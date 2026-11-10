// SPDX-License-Identifier: GPL-3.0
// visa_encode.rs --- VISA variable-width instruction encoder for SHARC+
// Copyright (c) 2026 Jakob Kastelic

//! SHARC+ VISA (Variable Instruction Set Architecture) encoder.
//!
//! Given an instruction already encoded as a 48-bit ISA word, attempts to
//! compress it into 16-bit or 32-bit VISA form. Falls back to 48-bit if
//! no compact representation exists.
//!
//! The encoding strategy reuses the proven ISA encoder and works backward
//! from the VISA decoder's bit layouts in `visa.rs`.

use crate::encode::{
    self, AluOp, ComputeOp, DagWidth, EncodeError, FaluOp, Instruction, MemWidth, MulOp, MultiOp,
    ShiftOp,
};

// Type 2c short-compute is a 16-bit `Rn = Rn OP Ry` family. The 5-bit
// opcode field at bits[12:8] selects which compute operation runs on
// real silicon. The mapping below comes from observing easm21k and ADI
// `elfdump` round-trip behaviour, not from a self-consistency check
// inside this crate. Several earlier entries (`AluOp::Add` at op 4 →
// FAdd, `FaluOp::Pass` at op 11 → FPass, FMul missing) round-tripped
// inside selas but the same bytes execute different operations on
// hardware, silently turning `F6 = F6 + F14` into `R6 = NOT R14`.
// The truth-table:
//   op 0  Rn = Rn + Ry  (Alu Add)
//   op 1  Rn = Rn - Ry  (Alu Sub)
//   op 2  Rn = pass Ry  (Alu Pass)
//   op 3  comp(Rx, Ry)  (Alu Comp)
//   op 4  Rn = NOT Ry   (Alu Not)
//   op 5  Rn = Ry + 1   (Alu Inc)
//   op 6  Rn = Ry - 1   (Alu Dec)
//   op 7  Rn = Rn * Ry (SSI)  (Mul MulSsi; the reference assembler c701. It was listed
//         as Rn = Rn + Ry + ci, which has no 16-bit form)
//   op 8  Fn = Fn + Fy  (Falu Add)
//   op 9  Fn = Fn - Fy  (Falu Sub)
//   op 10 Fn = FLOAT Ry (Falu Float; all 323 vendor rows, e.g. The reference assembler
//         `f0 = float r0` = ca00, `f1 = float r2` = ca12)
//   op 11 comp(Fx, Fy)  (Falu Comp)
//   op 12 Rn = Rn AND Ry (Alu And)
//   op 13 Rn = Rn OR Ry  (Alu Or)
//   op 14 Rn = Rn XOR Ry (Alu Xor)
//   op 15 Fn = Fn * Fy   (Mul FMul)
// The int-Neg/int-Abs aliases the older table had at 8/9 round-trip
// cleanly inside this crate but do not match hardware semantics, so
// they are intentionally absent from `try_type2c`. Falu Pass also has no Type 2c form: the reference assembler routes
// `Fn = pass Fy` through the 32-bit Type 2b path (`018a 12rr`).

/// Result of VISA compression.
#[derive(Debug, Clone)]
pub enum VisaEncoded {
    /// 16-bit instruction (1 parcel).
    W16(u16),
    /// 32-bit instruction (2 parcels, big-endian).
    W32(u32),
    /// 48-bit instruction (3 parcels, big-endian).
    W48([u8; 6]),
}

impl VisaEncoded {
    /// Whether the encoded instruction is zero-length (always false).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Number of bytes in the encoded instruction.
    pub fn len(&self) -> usize {
        match self {
            VisaEncoded::W16(_) => 2,
            VisaEncoded::W32(_) => 4,
            VisaEncoded::W48(_) => 6,
        }
    }

    /// Number of parcels (address units) consumed.
    pub fn parcels(&self) -> u32 {
        match self {
            VisaEncoded::W16(_) => 1,
            VisaEncoded::W32(_) => 2,
            VisaEncoded::W48(_) => 3,
        }
    }

    /// Emit bytes in big-endian order.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            VisaEncoded::W16(v) => vec![(v >> 8) as u8, *v as u8],
            VisaEncoded::W32(v) => vec![(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, *v as u8],
            VisaEncoded::W48(b) => b.to_vec(),
        }
    }
}

/// The compact SW/VISA 7-bit ureg field. Parsed instructions already carry
/// the universal register code that both widths use (vendor `astatx` is
/// 0x76 in 48-bit Type 15 `ac76…` and in 32-bit forms alike); codes above
/// 0x7F do not fit.
fn visa_ureg_code(ureg: u16) -> Option<u16> {
    (ureg <= 0x7F).then_some(ureg)
}

/// Attempt VISA compression of an instruction.
///
/// `instr` is the high-level instruction, `isa_bytes` is the 48-bit ISA
/// encoding (big-endian, as returned by `encode::encode`).
///
/// Returns the shortest VISA form possible.
pub fn visa_encode(instr: &Instruction, isa_bytes: &[u8; 6]) -> VisaEncoded {
    if let Some(w16) = try_16bit(instr) {
        return VisaEncoded::W16(w16);
    }
    if let Some(w32) = try_32bit(instr) {
        return VisaEncoded::W32(w32);
    }
    if let Some(w32) = try_marker32(instr, isa_bytes) {
        return VisaEncoded::W32(w32);
    }
    VisaEncoded::W48(*isa_bytes)
}

/// The 32-bit form of Types 1, 3, 4 and 5 without a compute: the top 32
/// bits of the 48-bit word with the marker 0x3f in bits 5:0 (word bits
/// 21:16, an empty compute's). The reference assembler never emits such a word in 48-bit
/// form: 0 of its accepted rows, against 6,202 in 32-bit, e.g.
/// `if ac r0 = m6` = 7247003f (48-bit 724700000000), `curlcntr = dm ( i4 ,
/// m2 )` = 58be33bf, `dm ( i0 , m0 ) = r4 , pm ( i8 , m10 ) = r8` =
/// 3028143f. selas left the conditional, PM and system-register forms
/// 48-bit. Short and byte widths use other markers (0x3c, 0x38) and stay
/// with their own encoders.
fn try_marker32(instr: &Instruction, isa: &[u8; 6]) -> Option<u32> {
    let normal_width = match *instr {
        // No vendor vector for a register swap (`r5 <-> s5`); it stays
        // 48-bit.
        Instruction::DualMove { .. }
        | Instruction::UregTransfer { .. }
        | Instruction::URegMove { .. } => true,
        Instruction::UregDagMove { width, .. } | Instruction::ComputeLoadStore { width, .. } => {
            width == DagWidth::Normal
        }
        _ => false,
    };
    let word = isa.iter().fold(0u64, |w, &b| (w << 8) | u64::from(b));
    let top3 = word >> 45;
    if !normal_width || !(1..=3).contains(&top3) || word & 0x7F_FFFF != 0 {
        return None;
    }
    Some(((word >> 16) as u32) | 0x3F)
}

/// Encode VISA, propagating encoder errors. Unsupported forms (including
/// access widths without a vendor-confirmed encoding) fail in `encode::encode`
/// rather than falling back to a different memory access width.
pub fn encode_visa(instr: &Instruction) -> Result<VisaEncoded, EncodeError> {
    match encode::encode(instr) {
        Ok(isa) => Ok(visa_encode(instr, &isa)),
        // A width-suffixed Type 4 access exists only as 32-bit Type 4b, and
        // the Type 3 widths without a 48-bit vector only as Type 3b. PM
        // Type 4 has only 32-bit vendor vectors (617e003f).
        Err(e) => match instr {
            Instruction::ComputeLoadStore { width, .. }
            | Instruction::UregDagMove { width, .. }
                if *width != DagWidth::Normal =>
            {
                try_32bit(instr).map(VisaEncoded::W32).ok_or(e)
            }
            Instruction::ComputeLoadStore {
                access,
                compute: None,
                ..
            } if access.pm => try_32bit(instr).map(VisaEncoded::W32).ok_or(e),
            _ => Err(e),
        },
    }
}

// ---------------------------------------------------------------------------
// 16-bit forms
// ---------------------------------------------------------------------------

fn try_16bit(instr: &Instruction) -> Option<u16> {
    match *instr {
        Instruction::Nop => Some(0x0001),
        Instruction::Idle => Some(0x0081),
        Instruction::EmuIdle => Some(0x00C1),
        Instruction::Rframe => Some(0x1901),
        // 16-bit RTS/RTI: 0x0A80 | RTI << 8 | DB << 6 | cond << 1 | LR.
        // The reference assembler: `rts` = 0abe, `rts (db)` = 0afe, `rti` = 0bbe,
        // `if eq rts (db)` = 0ac0, `if ne rti (db)` = 0be0, `if ne rts
        // (lr)` = 0aa1. No vector combines DB with LR,
        // so that form stays 48-bit.
        Instruction::Return {
            interrupt,
            cond,
            delayed,
            lr,
            compute: None,
        } if cond < 32 && !(delayed && lr) => Some(
            0x0A80
                | (u16::from(interrupt) << 8)
                | (u16::from(delayed) << 6)
                | (u16::from(cond) << 1)
                | u16::from(lr),
        ),
        Instruction::Compute { cond: 31, compute } => try_type2c(&compute),
        Instruction::UregDagMove {
            pm: false,
            write,
            ureg,
            i_reg,
            m_reg,
            cond: 31,
            compute: None,
            post_modify: true,
            width: DagWidth::Normal,
        } => try_type3c(write, ureg, i_reg, m_reg),
        _ => None,
    }
}

/// Type 2c: 16-bit short compute.
///
/// bits[15:13]=110, bits[12:8]=opcode, bits[7:4]=rn, bits[3:0]=ry.
/// Only available when rn==rx (operation is `rn = rn OP ry`).
///
/// See the module-level truth table comment for the full hardware
/// opcode mapping. The previous version of this function had several
/// entries off-by-row (FAdd at op 4 instead of 8, FPass at op 11
/// instead of un-compressed, FMul missing entirely, Not at op 15
/// instead of 4), which encoded but ran the wrong operation on real
/// silicon.
fn try_type2c(compute: &ComputeOp) -> Option<u16> {
    let (op, rn, ry) = match *compute {
        // Integer ALU ops where rn==rx
        ComputeOp::Alu(AluOp::Add { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => (0, rn, ry),
        ComputeOp::Alu(AluOp::Sub { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => (1, rn, ry),
        ComputeOp::Alu(AluOp::Pass { rn, rx }) if rn < 16 && rx < 16 => (2, rn, rx),
        ComputeOp::Alu(AluOp::Comp { rx, ry }) if rx < 16 && ry < 16 => (3, rx, ry),
        ComputeOp::Alu(AluOp::Not { rn, rx }) if rn < 16 && rx < 16 => (4, rn, rx),
        ComputeOp::Alu(AluOp::Inc { rn, rx }) if rn < 16 && rx < 16 => (5, rn, rx),
        ComputeOp::Alu(AluOp::Dec { rn, rx }) if rn < 16 && rx < 16 => (6, rn, rx),
        // Op 7 is the SSI multiply (the reference assembler `r0 = r0 * r1 (ssi)` = c701,
        // 7 rows). selas used it for `Rn = Rn + Ry + CI`, so an
        // add-with-carry ran as a multiply; that has no 16-bit form
        // (`r0 = r0 + r1 + ci` = 01805001).
        ComputeOp::Mul(MulOp::MulSsi { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => {
            (7, rn, ry)
        }
        // Abs and Neg are intentionally *not* compressed into the
        // Type 2c short-compute form. The 16-bit opcode-8/9 encoding
        // round-trips fine inside this crate, but the SHARC+ core
        // does not execute them as the expected single-source
        // `Rn = -Rx` / `Rn = ABS Rx` pair -- the compact form the
        // hardware actually implements for these opcodes treats the
        // bits[3:0] field differently, and a straight `Rn = ABS Rx`
        // compressed this way returns -1 from ABS(-7) on real
        // silicon. Let the 48-bit Type 2 encoder handle them.
        ComputeOp::Falu(FaluOp::Add { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => {
            (8, rn, ry)
        }
        ComputeOp::Falu(FaluOp::Sub { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => {
            (9, rn, ry)
        }
        ComputeOp::Falu(FaluOp::Float { rn, rx }) if rn < 16 && rx < 16 => (10, rn, rx),
        ComputeOp::Falu(FaluOp::Comp { rx, ry }) if rx < 16 && ry < 16 => (11, rx, ry),
        ComputeOp::Alu(AluOp::And { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => (12, rn, ry),
        ComputeOp::Alu(AluOp::Or { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => (13, rn, ry),
        ComputeOp::Alu(AluOp::Xor { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => (14, rn, ry),
        ComputeOp::Mul(MulOp::FMul { rn, rx, ry }) if rn == rx && rn < 16 && ry < 16 => {
            (15, rn, ry)
        }
        _ => return None,
    };
    Some(0b110u16 << 13 | (op as u16) << 8 | rn << 4 | ry)
}

/// Type 3c: 16-bit short data move (dreg <-> DM with DAG1 register addressing).
///
/// Only available for DAG1 (I0-I7, M0-M7), dreg R0-R15, no compute,
/// unconditional.
///
/// Layout: bits[15:12]=1001, bits[11:9]=i, bits[8:6]=m, bit[5]=d,
///         bit[4]=1(fixed), bits[3:0]=dreg.
fn try_type3c(write: bool, ureg: u16, i_reg: u8, m_reg: u8) -> Option<u16> {
    // ureg must be a data register (R0-R15): ureg codes 0x00-0x0F
    if ureg > 15 {
        return None;
    }
    // DAG1 only: I0-I7, M0-M7
    if i_reg > 7 || m_reg > 7 {
        return None;
    }
    let d = if write { 1u16 } else { 0 };
    let dreg = ureg;
    Some(
        0b1001u16 << 12
            | (i_reg as u16 & 7) << 9
            | (m_reg as u16 & 7) << 6
            | d << 5
            | 1u16 << 4
            | dreg,
    )
}

// ---------------------------------------------------------------------------
// 32-bit forms
// ---------------------------------------------------------------------------

fn try_32bit(instr: &Instruction) -> Option<u32> {
    match *instr {
        Instruction::Compute { cond, compute } => try_type2b(cond, &compute),
        Instruction::DagModify {
            pm,
            i_reg,
            m_reg,
            cond,
            compute: None,
            dest_xor,
            width: MemWidth::Normal,
        } => try_type7b(pm, i_reg, m_reg, cond, dest_xor),
        Instruction::LoadImm { ureg, value } => try_type17b(ureg, value),
        Instruction::UregDagMove {
            pm,
            write,
            ureg,
            i_reg,
            m_reg,
            cond,
            compute: None,
            post_modify,
            width,
        } => try_type3b(pm, write, ureg, i_reg, m_reg, cond, post_modify, width),
        Instruction::ComputeLoadStore {
            compute: None,
            access,
            dreg,
            offset,
            cond,
            post_modify,
            width,
        } => try_type4b(
            access.pm,
            access.write,
            access.i_reg,
            dreg,
            offset,
            cond,
            post_modify,
            width,
        ),
        // 32-bit VISA Group-4 encoding for Type 15 (ureg <-> DM with
        // 7-bit signed offset, DAG1 only). selcc's stack-frame setup
        // emits `DM(-N,I6)=Rx;` and the matching reload as
        // `UregMemAccess`, and they fit this encoding exactly. The
        // 48-bit ISA fall-through emits the same access in 6 bytes,
        // but the SHARC+ SW PM alias maps each PM word to 2 bytes,
        // so a 6-byte instruction in `seg_swco` straddles three SW
        // slots and the core decodes the middle slot as a separate
        // (garbage) instruction. Compressing here keeps every
        // selcc-emitted instruction within a single SW slot.
        Instruction::UregMemAccess {
            pm,
            i_reg,
            write,
            lw,
            ureg,
            offset,
        } if i_reg < 8 && (-64..=63).contains(&offset) && ureg <= 0x7F => {
            try_type15b(pm, write, lw, i_reg, ureg, offset)
        }
        Instruction::ImmStore {
            pm,
            i_reg,
            m_reg,
            value,
        } => try_type16b(pm, i_reg, m_reg, value),
        Instruction::URegMove { dest, src } => try_type5b_move(dest, src),
        Instruction::UregTransfer {
            src_ureg,
            dst_ureg,
            cond: 31,
            compute: None,
        } => try_type5b_move(dst_ureg, src_ureg),
        Instruction::IndirectBranch {
            call,
            cond,
            pm_i,
            pm_m,
            delayed,
            compute: None,
            else_compute: false,
        } => try_type9a_32bit(call, cond, pm_i, pm_m, delayed),
        _ => None,
    }
}

/// Type 9a: 32-bit VISA indirect jump/call through a PM DAG register pair.
///
/// Layout:
///   p1[15:13]=000, p1[12:8]=01000 (Type 9a), p1[7]=0 (32-bit flag),
///   p1[6]=J, p1[5:1]=cond, p1[0]=1 (reserved-set)
///   p2[15:14]=pmi_off (I12-I15 → 0-3), p2[13:11]=pmm (M8-M15 → 0-7),
///   p2[10]=db, p2[5:0]=0x3f (W32 marker)
///
/// Only reachable for I12-I15; other PM I-registers fall through to 48-bit.
/// `pm_i` and `pm_m` arrive as the full register index (0-15) from the
/// parser, and the encoder only uses the low 3 bits for I8-I15 / M8-M15.
/// Type 9a 32-bit form: the top 32 bits of the 48-bit word (`encode_type9a`)
/// with the 0x3f marker. The reference assembler: `call (m14, i12)` = 08bf303f,
/// `if lt jump (m14, i8) (db)` = 0802343f. The CALL bit used to sit at
/// p1[6] and only I12-I15 were accepted.
fn try_type9a_32bit(call: bool, cond: u8, pm_i: u8, pm_m: u8, delayed: bool) -> Option<u32> {
    let i = (pm_i & 7) as u16;
    let m = (pm_m & 7) as u16;
    let j = if call { 1u16 } else { 0 };
    let db = if delayed { 1u16 } else { 0 };
    let p1 = (0b01000u16 << 8) | (j << 7) | ((cond as u16 & 0x1F) << 1) | (i >> 2);
    let p2 = ((i & 3) << 14) | (m << 11) | (db << 10) | 0x3f;
    Some((p1 as u32) << 16 | p2 as u32)
}

/// Type 2b: 32-bit compute.
///
/// Layout: `[31:29]=000  [28:24]=00001  [23]=1  [22:0]=compute`
fn try_type2b(cond: u8, compute: &ComputeOp) -> Option<u32> {
    let comp23 = compute_to_23bit(compute)?;
    // Condition is encoded in p1[5:1] for 32-bit: bit7=1 flag + cond
    // Actually, Type 2b layout: p1 = 000_00001_1_CCCCC_0, p2 = compute low 16
    // Full 32-bit: bits[31:24]=0x01, bit[23]=1(32b), bits[22:18]=cond, ... no.
    //
    // From visa.rs decode_32_type2b: compute = {p1[6:0], p2[15:0]} = 23 bits
    // p1[15:13]=000, p1[12:8]=00001, p1[7]=1 (32-bit flag)
    // p1[6:0] = compute[22:16]
    // p2 = compute[15:0]
    //
    // But the decoder says "unconditional" — check: the decoder doesn't extract
    // a condition field. 32-bit Type 2b is always unconditional.
    if cond != 31 {
        return None;
    }
    let p1 = 0x0180u16 | ((comp23 >> 16) as u16 & 0x7F);
    let p2 = comp23 as u16;
    Some((p1 as u32) << 16 | p2 as u32)
}

/// Type 7b: 32-bit MODIFY with result assignment.
///
/// `i_dest = MODIFY(i_src, m)` — only supports i_dest == i_src (dest_xor=0)
/// or when the XOR trick works.
///
/// Layout: p1[7]=0 (32-bit flag), p1[6]=g, p1[5:1]=cond, p1[0]=i_src[2]
///         p2[15:14]=i_src[1:0], p2[13:11]=m, p2[10:8]=dest_xor
fn try_type7b(pm: bool, i_reg: u8, m_reg: u8, cond: u8, dest_xor: u8) -> Option<u32> {
    // Top two parcels of the 48-bit Type 7 word plus the 0x3f 32-bit
    // marker: the reference assembler `i1 = modify(i1,m4)` = 043e603f,
    // `i1 = modify(i0,m4)` = 043e213f.
    let g = if pm { 1u16 } else { 0 };
    let i_local = i_reg & 7;
    let m_local = m_reg & 7;
    let dest_xor = (dest_xor & 7) as u16;

    let p1 =
        (0b00000100u16 << 8) | (g << 6) | ((cond as u16 & 0x1F) << 1) | ((i_local as u16 >> 2) & 1);
    let p2 = ((i_local as u16 & 3) << 14) | ((m_local as u16) << 11) | (dest_xor << 8) | 0x3f;
    Some((p1 as u32) << 16 | p2 as u32)
}

/// Type 17b: 32-bit LoadImm where value fits signed 16-bit.
///
/// Layout: p1[15:13]=000, p1[12:8]=01111, p1[7]=1 (32-bit flag),
///         p1[6:0]=ureg code, p2=imm16.
fn try_type17b(ureg: u16, value: u32) -> Option<u32> {
    // Value must fit in signed 16-bit range
    let v = value as i32;
    if !(-32768..=32767).contains(&v) {
        return None;
    }
    let ureg = visa_ureg_code(ureg)?;
    let p1 = 0x0F80u16 | (ureg & 0x7F);
    let p2 = value as u16;
    Some((p1 as u32) << 16 | p2 as u32)
}

/// Type 3b: 32-bit ureg <-> DM|PM with register addressing.
///
///   p1[12]   = post-modify (0=pre-modify, 1=post-modify)
///   p1[11:9] = I, p1[8:6] = M (within the DAG)
///   p1[5:1]  = cond, p1[0] = G (PM, DAG2)
///   p2[15]   = direction (0=read, 1=write)
///   p2[14]   = long/short-word flag
///   p2[13:7] = ureg code
///   p2[5:0]  = 0x3c | width (11 normal/long, 00 short/byte, 01 sign-ext)
#[allow(clippy::too_many_arguments)]
fn try_type3b(
    pm: bool,
    write: bool,
    ureg: u16,
    i_reg: u8,
    m_reg: u8,
    cond: u8,
    post_modify: bool,
    width: DagWidth,
) -> Option<u32> {
    // p1 is the 48-bit word's top parcel: 010 U III MMM ccccc G (M is
    // three bits; bit 8 used to be forced, so only M4-M7 compressed).
    // p2 = D W uuuuuuu 1111LL: W (bit 14) long/short, LL 11 normal or long,
    // 00 short or byte, 01 sign-extended. The reference assembler, all M0-M7:
    // `curlcntr = dm ( i4 , m2 )` = 58be33bf, `r0 = dm(i0, m4)(lw)` =
    // 513e403f, `r0 = dm(i0, m0)(sw)` = 503e403c, `r0 = dm(i0, m0)(bw)` =
    // 503e003c, `r0 = dm(i0, m0)(swse)` = 503e403d, `dm(i1, m2) = r2(sw)`
    // = 52bec13c.
    if i_reg > 7 || m_reg > 7 {
        return None;
    }
    let ureg = visa_ureg_code(ureg)?;
    let (w, ll) = match width {
        DagWidth::Normal => (0u16, 0b11u16),
        DagWidth::Long => (1, 0b11),
        DagWidth::Short => (1, 0b00),
        DagWidth::Byte => (0, 0b00),
        DagWidth::ShortSe => (1, 0b01),
        DagWidth::ByteSe => (0, 0b01),
    };
    // Widths are vendor-observed on DM only (any register: `dm ( i4 , m6 )
    // = m13 ( bw )` = 59be96bc); sign extension only on loads.
    if width != DagWidth::Normal && pm {
        return None;
    }
    if matches!(width, DagWidth::ShortSe | DagWidth::ByteSe) && write {
        return None;
    }
    let p1 = (0b010u16 << 13)
        | (u16::from(post_modify) << 12)
        | (u16::from(i_reg) << 9)
        | (u16::from(m_reg) << 6)
        | ((u16::from(cond) & 0x1F) << 1)
        | u16::from(pm);
    let p2 = (u16::from(write) << 15) | (w << 14) | ((ureg & 0x7F) << 7) | 0x3C | ll;
    Some((u32::from(p1) << 16) | u32::from(p2))
}

/// Type 4b: 32-bit dreg <-> DM with 6-bit immediate offset, no compute.
///
/// From visa.rs decode_32_type4b:
///   p1[12:9] = i-register index
///   p1[10]   = direction
///   p1[5:1]  = cond
///   {p1[0], p2[15:11]} = 6-bit signed offset
///   p2[10:7] = dreg
///   p2[1:0]  = 11 (normal word)
fn try_type4b(
    pm: bool,
    write: bool,
    i_reg: u8,
    dreg: u16,
    offset: i8,
    cond: u8,
    post_modify: bool,
    width: DagWidth,
) -> Option<u32> {
    if i_reg > 7 || dreg > 15 {
        return None;
    }
    // Width markers: 0x3f normal, 0x3c short word, 0x38 byte word
    // (the reference assembler 6a7e113f, 683e163c `r12 = dm(0x2, i4)(sw)`, 683e0938
    // `r2 = dm(1, i4)(bw)`). The marker is independent of the update
    // bit: 687e0838 is `r0 = dm(i4, 1)(bw)`.
    // Sign-extending widths end 01: `r0 = dm ( i4 , 0x2 ) ( swse )` =
    // 687e103d, `r0 = dm ( -0x5 , i2 ) ( bwse )` = 643fd839.
    let marker = match width {
        DagWidth::Normal => 0x3f,
        DagWidth::Short => 0x3c,
        DagWidth::Byte => 0x38,
        DagWidth::ShortSe => 0x3d,
        DagWidth::ByteSe => 0x39,
        DagWidth::Long => return None,
    };
    // Offset must fit in 6-bit signed (-32..31)
    if !(-32..=31).contains(&offset) {
        return None;
    }
    let d = if write { 1u16 } else { 0 };
    let raw6 = (offset as u8 as u16) & 0x3F;
    let off_hi = (raw6 >> 5) & 1; // bit 5 → p1[0]
    let off_lo = raw6 & 0x1F; // bits 4:0 → p2[15:11]

    // p1 is the top parcel of the 48-bit Type 4 word: i at [11:9], d at 7,
    // u (post-modify) at 6, cond at [5:1]. The reference assembler: `r2=dm(i5,2)` =
    // 6a7e113f, `if sz r15=dm(-3,i6)` = 6c11efbf.
    // p1[8] (word bit 40) is G, PM through DAG2: `r0 = pm ( i8 , 0x0 )` =
    // 617e003f, `pm ( i10 , 0x0 ) = r2` = 65fe013f.
    let p1 = (0b011u16 << 13)
        | ((i_reg as u16 & 0x7) << 9)
        | ((pm as u16) << 8)
        | (d << 7)
        | ((post_modify as u16) << 6)
        | ((cond as u16 & 0x1F) << 1)
        | off_hi;
    let p2 = (off_lo << 11) | ((dreg & 0xF) << 7) | marker;
    Some((p1 as u32) << 16 | p2 as u32)
}

/// 32-bit VISA encoding for Type 15 ureg <-> DM with 7-bit
/// signed offset. The SHARC+ Group-4 ureg-mem-access form sits at
/// p1[15:13] = 100 and accepts any 7-bit ureg destination/source,
/// so it covers selcc's stack-frame stores (`DM(-N,I6)=Rx`) and
/// their matching reloads in a single 32-bit slot. The Type 16b
/// (`dm(Ii,Mm)=imm16`) sub-form within Group 4 is gated on p1[2]
/// = 1; we leave that bit clear so the decoder routes back to the
/// standard ureg-mem path. p1 bits 3:0 are fixed to 0x8 to match
/// easm21k's encoding -- the ADSP-21569 ROM rejects forms with
/// other lower-nibble values for this access type.
///
/// p1 bit 5 is G (PM, DAG2): the reference assembler `pm ( 0x0 , i10 ) = r2` = 95280100,
/// `m15 = pm ( 0x0 , i11 )` = 96281780; selas emitted every PM access
/// as the 48-bit Type 15.
fn try_type15b(pm: bool, write: bool, lw: bool, i_reg: u8, ureg: u16, offset: i32) -> Option<u32> {
    if i_reg > 7 {
        return None;
    }
    if !(-64..=63).contains(&offset) {
        return None;
    }
    let ureg = visa_ureg_code(ureg)?;
    let d = if write { 1u16 } else { 0 };
    let lw_bit = if lw { 1u16 } else { 0 };
    let raw7 = (offset as u8 as u16) & 0x7F;
    // p1[12] is the W32 marker the SHARC+ pipeline uses to tell a
    // Group-4 32-bit instruction apart from a 16-bit Type 3c data
    // move that shares the top4 = 1001 prefix. Without it set, real
    // silicon decodes the same byte stream as a 16-bit Type 3c
    // (DAG-register-addressed move) followed by a stray 16-bit
    // parcel, which writes the wrong dreg and corrupts the spill
    // slot. easm21k always sets this bit for the standard ureg
    // <-> DM access form.
    let p1 = (0b100u16 << 13)
        | (1u16 << 12)
        | ((i_reg as u16 & 7) << 9)
        | (d << 8)
        | (lw_bit << 7)
        | (u16::from(pm) << 5)
        | 0x08;
    let p2 = ((ureg & 0x7F) << 7) | raw7;
    Some((p1 as u32) << 16 | p2 as u32)
}

/// Type 16b (32-bit `DM|PM(Ii, Mm) = imm16`): the 48-bit Type 16 first
/// parcel (I 11:9, M 8:6, G 5) with bits 4:0 = 00100, then the immediate,
/// sign-extended. The reference assembler uses it exactly when the value fits a signed
/// 16-bit field: `dm ( i7 , m7 ) = 0x1000` = 9fc41000, `dm(i0, m0) = -0x1`
/// = 9004ffff, `pm ( i10 , m8 ) = 0x0` = 94240000; 0x8000 stays 48-bit.
fn try_type16b(pm: bool, i_reg: u8, m_reg: u8, value: u32) -> Option<u32> {
    let value = value as i32;
    if i_reg > 7 || m_reg > 7 || !(-0x8000..=0x7FFF).contains(&value) {
        return None;
    }
    let p1 = (0b1001u16 << 12)
        | (u16::from(i_reg) << 9)
        | (u16::from(m_reg) << 6)
        | (u16::from(pm) << 5)
        | 0x04;
    Some((u32::from(p1) << 16) | u32::from(value as u16))
}

/// Type 5b (32-bit ureg move): `dest_ureg = src_ureg`.
///
/// From visa.rs decode_32_ureg_move:
///   p1[15:13] = 011 (group-3 selector; required by hardware and by
///               decode_32's top-3 routing — forgetting this byte made
///               the encoded word look like a group-0 Type 17b load, so
///               `I4 = R1` silently assembled to `L14 = DM(...)` and
///               trampled the DAG on real silicon)
///   p1[12:8]  = sub5 = 10_group[2:0]
///   p1[7]     = src_idx[3]
///   p1[6]     = src_idx[2]
///   p1[5:1]   = cond
///   p1[0]     = src_idx[1]
///   p2[15]    = src_idx[0]
///   p2[13:7]  = dst_ureg (7-bit code)
///   p2[5:0]   = 0x3f (W32 width marker; group-3 ambiguous opcodes are
///               classified W32 vs W48 by `p2 & 0x3F >= 0x38`. Without
///               the marker the assembled bytes decode on hardware as
///               a 48-bit Type 5a parallel compute + ureg move, which
///               clobbers a dreg adjacent to the move).
fn try_type5b_move(dest: u16, src: u16) -> Option<u32> {
    let dest = visa_ureg_code(dest)?;
    let src = visa_ureg_code(src)?;
    let src_group = (src >> 4) & 7;
    let src_idx = src & 0xF;

    let sub5 = 0b10_000u16 | src_group;
    let cond = 31u16; // unconditional

    let p1 = (0b011u16 << 13)
        | (sub5 << 8)
        | ((src_idx >> 3) & 1) << 7
        | ((src_idx >> 2) & 1) << 6
        | (cond << 1)
        | ((src_idx >> 1) & 1);
    let p2 = (src_idx & 1) << 15 | ((dest & 0x7F) << 7) | 0x3f;
    Some((p1 as u32) << 16 | p2 as u32)
}

// ---------------------------------------------------------------------------
// Compute field → 23-bit encoding for 32-bit instructions
// ---------------------------------------------------------------------------

/// Convert a ComputeOp to a 23-bit compute field for 32-bit VISA instructions.
///
/// The 23-bit field layout matches the ISA 48-bit compute field:
///   bit[22]    = multi (0 for single)
///   bits[21:20] = cu (0=ALU, 1=MUL, 2=SHIFT)
///   bits[19:12] = opcode
///   bits[11:8]  = Rn
///   bits[7:4]   = Rx
///   bits[3:0]   = Ry
fn compute_to_23bit(compute: &ComputeOp) -> Option<u32> {
    match compute {
        ComputeOp::Alu(alu) => alu_to_23bit(alu),
        ComputeOp::Falu(falu) => falu_to_23bit(falu),
        ComputeOp::Mul(mul) => mul_to_23bit(mul),
        ComputeOp::Shift(shift) => shift_to_23bit(shift),
        // A float multiply + ALU multifunction has the 32-bit Type 2b form
        // (the reference assembler `f0 = f2 * f4, f1 = max (f8, f15)` = 01de0183). Other
        // multifunctions have no 32-bit vendor vector and stay 48-bit.
        // So does the float multiply + dual add/subtract (all 46 vendor
        // rows, e.g. `f10 = f0 * f4 , f12 = f8 + f12 , f8 = f8 - f12` =
        // 01f8ac00). No fixed-point vector exists for either.
        ComputeOp::Multi(MultiOp::MulAlu { fp: true, .. })
        | ComputeOp::Multi(MultiOp::MulDualAddSub { fp: true, .. }) => {
            crate::encode::encode_compute(compute).ok()
        }
        ComputeOp::Multi(_) => None,
    }
}

fn alu_to_23bit(alu: &AluOp) -> Option<u32> {
    let (opcode, rn, rx, ry): (u8, u16, u16, u16) = match *alu {
        AluOp::Add { rn, rx, ry } => (0x01, rn, rx, ry),
        AluOp::Sub { rn, rx, ry } => (0x02, rn, rx, ry),
        AluOp::AddCi { rn, rx, ry } => (0x05, rn, rx, ry),
        AluOp::SubCi { rn, rx, ry } => (0x06, rn, rx, ry),
        AluOp::Avg { rn, rx, ry } => (0x09, rn, rx, ry),
        AluOp::Comp { rx, ry } => (0x0A, 0, rx, ry),
        AluOp::CompU { rx, ry } => (0x0B, 0, rx, ry),
        AluOp::Pass { rn, rx } => (0x21, rn, rx, 0),
        AluOp::Neg { rn, rx } => (0x22, rn, rx, 0),
        AluOp::PassCi { rn, rx } => (0x25, rn, rx, 0),
        AluOp::PassCiMinus1 { rn, rx } => (0x26, rn, rx, 0),
        AluOp::Inc { rn, rx } => (0x29, rn, rx, 0),
        AluOp::Dec { rn, rx } => (0x2A, rn, rx, 0),
        AluOp::Abs { rn, rx } => (0x30, rn, rx, 0),
        AluOp::And { rn, rx, ry } => (0x40, rn, rx, ry),
        AluOp::Or { rn, rx, ry } => (0x41, rn, rx, ry),
        AluOp::Xor { rn, rx, ry } => (0x42, rn, rx, ry),
        AluOp::Not { rn, rx } => (0x43, rn, rx, 0),
        AluOp::Min { rn, rx, ry } => (0x61, rn, rx, ry),
        AluOp::Max { rn, rx, ry } => (0x62, rn, rx, ry),
        AluOp::Clip { rn, rx, ry } => (0x63, rn, rx, ry),
    };
    if rn > 15 || rx > 15 || ry > 15 {
        return None;
    }
    // cu=0 (ALU), multi=0
    Some((opcode as u32) << 12 | (rn as u32) << 8 | (rx as u32) << 4 | ry as u32)
}

fn falu_to_23bit(falu: &FaluOp) -> Option<u32> {
    let (opcode, rn, rx, ry): (u8, u16, u16, u16) = match *falu {
        FaluOp::DualAddSub { rn, rs, rx, ry } if rs < 16 => (0xF0 | rs as u8, rn, rx, ry),
        FaluOp::Add { rn, rx, ry } => (0x81, rn, rx, ry),
        FaluOp::Sub { rn, rx, ry } => (0x82, rn, rx, ry),
        FaluOp::Avg { rn, rx, ry } => (0x89, rn, rx, ry),
        FaluOp::Comp { rx, ry } => (0x8A, 0, rx, ry),
        FaluOp::AbsAdd { rn, rx, ry } => (0x91, rn, rx, ry),
        FaluOp::AbsSub { rn, rx, ry } => (0x92, rn, rx, ry),
        FaluOp::Pass { rn, rx } => (0xA1, rn, rx, 0),
        FaluOp::Neg { rn, rx } => (0xA2, rn, rx, 0),
        FaluOp::Rnd { rn, rx } => (0xA5, rn, rx, 0),
        FaluOp::Mant { rn, rx } => (0xAD, rn, rx, 0),
        FaluOp::Abs { rn, rx } => (0xB0, rn, rx, 0),
        FaluOp::Scalb { rn, rx, ry } => (0xBD, rn, rx, ry),
        FaluOp::Logb { rn, rx } => (0xC1, rn, rx, 0),
        FaluOp::Recips { rn, rx } => (0xC4, rn, rx, 0),
        FaluOp::Rsqrts { rn, rx } => (0xC5, rn, rx, 0),
        FaluOp::Fix { rn, rx } => (0xC9, rn, rx, 0),
        FaluOp::Float { rn, rx } => (0xCA, rn, rx, 0),
        FaluOp::Trunc { rn, rx } => (0xCD, rn, rx, 0),
        FaluOp::FixBy { rn, rx, ry } => (0xD9, rn, rx, ry),
        FaluOp::FloatBy { rn, rx, ry } => (0xDA, rn, rx, ry),
        FaluOp::TruncBy { rn, rx, ry } => (0xDD, rn, rx, ry),
        FaluOp::Copysign { rn, rx, ry } => (0xE0, rn, rx, ry),
        FaluOp::Min { rn, rx, ry } => (0xE1, rn, rx, ry),
        FaluOp::Max { rn, rx, ry } => (0xE2, rn, rx, ry),
        FaluOp::Clip { rn, rx, ry } => (0xE3, rn, rx, ry),
        FaluOp::DualAddSub { .. } => return None,
    };
    if rn > 15 || rx > 15 || ry > 15 {
        return None;
    }
    // cu=0 (ALU — FP opcodes share the ALU CU), multi=0
    Some((opcode as u32) << 12 | (rn as u32) << 8 | (rx as u32) << 4 | ry as u32)
}

fn mul_to_23bit(mul: &MulOp) -> Option<u32> {
    // MR-register transfers (`Rn = MRxF/B`, `MRxF/B = Rn`) live in
    // their own sub-encoding inside the 23-bit compute field rather
    // than in the MULOP opcode table. Detect them up front so the
    // 32-bit Type 2b short form picks up the correct bit pattern that
    // matches the 48-bit ISA encoder (and easm21k/elfdump on
    // hardware).
    if let Some((t, ai, rk)) = mr_transfer_fields(mul) {
        if rk > 15 {
            return None;
        }
        return Some(
            1u32 << 22 | (t as u32) << 16 | ((ai as u32) & 0xF) << 12 | ((rk as u32) & 0xF) << 8,
        );
    }
    // Opcodes match the 48-bit `encode_mul`; the 32-bit VISA
    // short-compute form shares the full 8-bit opcode field with
    // the 48-bit form.
    let (opcode, rn, rx, ry): (u8, u16, u16, u16) = match *mul {
        MulOp::MulSsi { rn, rx, ry } => (0x70, rn, rx, ry),
        MulOp::MulSsf { rn, rx, ry } => (0x78, rn, rx, ry),
        MulOp::MrfMulSsi { rx, ry } => (0x74, 0, rx, ry),
        MulOp::MrfMulSsf { rx, ry } => (0x7C, 0, rx, ry),
        MulOp::MrbMulSsf { rx, ry } => (0x7E, 0, rx, ry),
        MulOp::MrfMulUuf { rx, ry } => (0x4C, 0, rx, ry),
        MulOp::MrfMulUui { rx, ry } => (0x44, 0, rx, ry),
        MulOp::FixedMul { opcode, rn, rx, ry } if opcode >= 0x40 => (opcode, rn, rx, ry),
        MulOp::MacSsf { rn, rx, ry } => (0xB8, rn, rx, ry),
        MulOp::MrfMacSsf { rx, ry } => (0xBC, 0, rx, ry),
        MulOp::MrbMacSsf { rx, ry } => (0xBE, 0, rx, ry),
        MulOp::MrfMsubSsf { rx, ry } => (0xFC, 0, rx, ry),
        MulOp::MrbMsubSsf { rx, ry } => (0xFE, 0, rx, ry),
        // Unsuffixed `Rn = SAT MRF|MRB` is the (SF) form: the reference assembler 01909000 /
        // 0190b000. 0x08/0x0A are (UF).
        MulOp::SatMrf { rn } => (0x09, rn, 0, 0),
        MulOp::SatMrb { rn } => (0x0B, rn, 0, 0),
        MulOp::MrToReg { opcode, rn } => (opcode, rn, 0, 0),
        MulOp::ClrMrf => (0x14, 0, 0, 0),
        MulOp::ClrMrb => (0x16, 0, 0, 0),
        MulOp::FMul { rn, rx, ry } => (0x30, rn, rx, ry),
        _ => return None, // TRNC etc.
    };
    if rn > 15 || rx > 15 || ry > 15 {
        return None;
    }
    // cu=1 (MUL), multi=0
    Some(1u32 << 20 | (opcode as u32) << 12 | (rn as u32) << 8 | (rx as u32) << 4 | ry as u32)
}

/// Return `(t, ai, rk)` for MR-transfer ops (`Rn = MRxF/B` /
/// `MRxF/B = Rn`), or `None` otherwise. `t = 0` means read MR -> Rn,
/// `t = 1` means write Rn -> MR; `ai` is the MR-register selector
/// (0=MR0F, 1=MR1F, 2=MR2F, 4=MR0B, 5=MR1B, 6=MR2B); `rk` is the
/// data register.
fn mr_transfer_fields(mul: &MulOp) -> Option<(u8, u8, u16)> {
    let triple = match *mul {
        MulOp::ReadMr0f { rn } => (0u8, 0u8, rn),
        MulOp::ReadMr1f { rn } => (0, 1, rn),
        MulOp::ReadMr2f { rn } => (0, 2, rn),
        MulOp::ReadMr0b { rn } => (0, 4, rn),
        MulOp::ReadMr1b { rn } => (0, 5, rn),
        MulOp::ReadMr2b { rn } => (0, 6, rn),
        MulOp::WriteMr0f { rn } => (1, 0, rn),
        MulOp::WriteMr1f { rn } => (1, 1, rn),
        MulOp::WriteMr2f { rn } => (1, 2, rn),
        MulOp::WriteMr0b { rn } => (1, 4, rn),
        MulOp::WriteMr1b { rn } => (1, 5, rn),
        MulOp::WriteMr2b { rn } => (1, 6, rn),
        _ => return None,
    };
    Some(triple)
}

fn shift_to_23bit(shift: &ShiftOp) -> Option<u32> {
    // Opcodes match the 48-bit `encode_shift`.
    let (opcode, rn, rx, ry): (u8, u16, u16, u16) = match *shift {
        ShiftOp::Lshift { rn, rx, ry } => (0x00, rn, rx, ry),
        ShiftOp::Ashift { rn, rx, ry } => (0x04, rn, rx, ry),
        ShiftOp::Rot { rn, rx, ry } => (0x08, rn, rx, ry),
        ShiftOp::OrLshift { rn, rx, ry } => (0x20, rn, rx, ry),
        ShiftOp::OrAshift { rn, rx, ry } => (0x24, rn, rx, ry),
        ShiftOp::Fext { rn, rx, ry } => (0x40, rn, rx, ry),
        ShiftOp::Fdep { rn, rx, ry } => (0x44, rn, rx, ry),
        ShiftOp::Exp { rn, rx } => (0x80, rn, rx, 0),
        ShiftOp::ExpEx { rn, rx } => (0x84, rn, rx, 0),
        ShiftOp::Leftz { rn, rx } => (0x88, rn, rx, 0),
        ShiftOp::Lefto { rn, rx } => (0x8C, rn, rx, 0),
        ShiftOp::Fpack { rn, rx } => (0x90, rn, rx, 0),
        ShiftOp::Funpack { rn, rx } => (0x94, rn, rx, 0),
        ShiftOp::BffwrpRead { rn } => (0x70, rn, 0, 0),
        ShiftOp::BffwrpWrite { ry } => (0x7C, 0, 0, ry),
        ShiftOp::Bset { rn, rx, ry } => (0xC0, rn, rx, ry),
        ShiftOp::Bclr { rn, rx, ry } => (0xC4, rn, rx, ry),
        ShiftOp::Btgl { rn, rx, ry } => (0xC8, rn, rx, ry),
        ShiftOp::Btst { rx, ry } => (0xCC, 0, rx, ry),
        _ => return None,
    };
    if rn > 15 || rx > 15 || ry > 15 {
        return None;
    }
    // cu=2 (SHIFT), multi=0
    Some(2u32 << 20 | (opcode as u32) << 12 | (rn as u32) << 8 | (rx as u32) << 4 | ry as u32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{self, Instruction};

    fn encode_isa(instr: &Instruction) -> [u8; 6] {
        encode::encode(instr).expect("ISA encode failed")
    }

    #[test]
    fn nop_compresses_to_16bit() {
        let instr = Instruction::Nop;
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        assert!(matches!(result, VisaEncoded::W16(0x0001)));
        assert_eq!(result.len(), 2);
        assert_eq!(result.parcels(), 1);
    }

    #[test]
    fn idle_compresses_to_16bit() {
        let instr = Instruction::Idle;
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        assert!(matches!(result, VisaEncoded::W16(0x0081)));
    }

    #[test]
    fn emuidle_compresses_to_16bit() {
        let instr = Instruction::EmuIdle;
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        assert!(matches!(result, VisaEncoded::W16(0x00C1)));
    }

    #[test]
    fn short_compute_add() {
        // R3 = R3 + R5 → Type 2c, op=0
        let instr = Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Add {
                rn: 3,
                rx: 3,
                ry: 5,
            }),
        };
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        match result {
            VisaEncoded::W16(v) => {
                assert_eq!(v, 0b1100_0000_0011_0101);
            }
            _ => panic!("expected 16-bit"),
        }
    }

    #[test]
    fn short_compute_rn_ne_rx_stays_48() {
        // R3 = R4 + R5: rn != rx, cannot use Type 2c
        let instr = Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Add {
                rn: 3,
                rx: 4,
                ry: 5,
            }),
        };
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        // Should be 32-bit Type 2b (no rn==rx constraint at 32-bit level)
        assert!(matches!(result, VisaEncoded::W32(_)));
    }

    #[test]
    fn load_imm16_compresses() {
        let instr = Instruction::LoadImm {
            ureg: 0x06,
            value: 0x1234,
        };
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        match result {
            VisaEncoded::W32(v) => {
                let p1 = (v >> 16) as u16;
                let p2 = v as u16;
                assert_eq!(p1, 0x0F80 | 0x06);
                assert_eq!(p2, 0x1234);
            }
            _ => panic!("expected 32-bit"),
        }
    }

    #[test]
    fn system_uregs_keep_the_universal_code() {
        assert_eq!(visa_ureg_code(0x63), Some(0x63)); // PC
        assert_eq!(visa_ureg_code(0x72), Some(0x72)); // MODE1
        assert_eq!(visa_ureg_code(0x76), Some(0x76)); // ASTATx
        assert_eq!(visa_ureg_code(0x7F), Some(0x7F)); // USTAT4
        assert_eq!(visa_ureg_code(0x80), None);
    }

    #[test]
    fn astatx_dag_move_matches_easm21k_swc() {
        let instr = Instruction::UregDagMove {
            pm: false,
            write: true,
            ureg: 0x76,
            i_reg: 6,
            m_reg: 5,
            cond: 31,
            compute: None,
            post_modify: false,
            width: DagWidth::Normal,
        };
        let isa = encode_isa(&instr);
        assert!(matches!(
            visa_encode(&instr, &isa),
            VisaEncoded::W32(0x4d7ebb3f)
        ));
    }

    #[test]
    fn astatx_stack_access_matches_easm21k_swc() {
        let instr = Instruction::UregMemAccess {
            pm: false,
            i_reg: 6,
            write: true,
            lw: false,
            ureg: 0x76,
            offset: -1,
        };
        let isa = encode_isa(&instr);
        assert!(matches!(
            visa_encode(&instr, &isa),
            VisaEncoded::W32(0x9d083b7f)
        ));
    }

    #[test]
    fn load_imm32_stays_48() {
        let instr = Instruction::LoadImm {
            ureg: 0x06,
            value: 0x12345678,
        };
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        assert!(matches!(result, VisaEncoded::W48(_)));
    }

    #[test]
    fn branch_stays_48() {
        use crate::encode::BranchTarget;
        let instr = Instruction::Branch {
            call: false,
            cond: 31,
            target: BranchTarget::Absolute(0x100),
            delayed: false,
            loop_abort: false,
        };
        let isa = encode_isa(&instr);
        let result = visa_encode(&instr, &isa);
        // Branches are always 48-bit in VISA
        assert!(matches!(result, VisaEncoded::W48(_)));
    }

    #[test]
    fn to_bytes_correctness() {
        assert_eq!(VisaEncoded::W16(0xABCD).to_bytes(), vec![0xAB, 0xCD]);
        assert_eq!(
            VisaEncoded::W32(0x12345678).to_bytes(),
            vec![0x12, 0x34, 0x56, 0x78]
        );
    }

    // Ported from ElekLabs/selache fix/encodings (Em D).
    #[test]
    fn modify_with_m_register_is_type_7b_with_its_fixed_bits() {
        // MODIFY (I5, M2): PRM Figure 14-20 fixes frame bits 22-16 at 0111111. Without them the word reads as a 48-bit Type 7a.
        let instr = Instruction::DagModify {
            pm: false,
            dest_xor: 0,
            i_reg: 5,
            m_reg: 2,
            cond: 31,
            compute: None,
            width: MemWidth::Normal,
        };
        let isa = encode_isa(&instr);
        match visa_encode(&instr, &isa) {
            // p1 = 0000 0100 0 g=0 cond=11111 is[2]=1 -> 0x043F; p2 = is[1:0]=01, m=010, idis=000, 0x3F -> 0x503F
            VisaEncoded::W32(v) => assert_eq!(v, 0x043F_503F),
            _ => panic!("expected 32-bit"),
        }
    }
}
