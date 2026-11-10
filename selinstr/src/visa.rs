// SPDX-License-Identifier: GPL-3.0
// visa.rs --- VISA variable-width instruction decoder for SHARC+
// Copyright (c) 2026 Jakob Kastelic

//! SHARC+ VISA (Variable Instruction Set Architecture) decoder.
//!
//! VISA instructions are 16, 32, or 48 bits wide. Width is determined from
//! the first 16-bit parcel. Parcels are stored big-endian in the ELF.

/// Instruction width in VISA mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrWidth {
    W16,
    W32,
    W48,
}

impl InstrWidth {
    /// Byte length of this instruction width.
    pub fn bytes(self) -> usize {
        match self {
            InstrWidth::W16 => 2,
            InstrWidth::W32 => 4,
            InstrWidth::W48 => 6,
        }
    }
}

/// Return the byte length of the VISA instruction at `code[offset..]`.
///
/// Reads one or two 16-bit big-endian parcels to determine whether the
/// instruction is 16-bit (2 bytes), 32-bit (4 bytes), or 48-bit (6 bytes).
/// Returns 2 if the offset is at or past the end of the data.
pub fn instruction_len(code: &[u8], offset: usize) -> usize {
    if offset + 2 > code.len() {
        return 2;
    }
    let p1 = u16::from_be_bytes([code[offset], code[offset + 1]]);
    if is_16bit(p1) {
        return 2;
    }
    if offset + 4 > code.len() {
        return 2;
    }
    let p2 = u16::from_be_bytes([code[offset + 2], code[offset + 3]]);
    visa_width(p1, p2).bytes()
}

/// One disassembled VISA instruction.
pub struct VisaLine {
    /// Parcel address.
    pub addr: u32,
    /// Raw instruction bytes (2, 4, or 6).
    pub raw: Vec<u8>,
    /// Formatted disassembly text.
    pub text: String,
}

/// Disassemble a code section in VISA mode (BW=2).
///
/// VISA parcels are always big-endian (MSB first) regardless of ELF endianness.
/// The `decode_48` parameter is a callback for 48-bit instruction decoding
/// (typically `disasm::decode_instruction`).
pub fn disassemble_visa(
    data: &[u8],
    base_addr: u32,
    decode_48: fn(u64) -> String,
) -> Vec<VisaLine> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    let mut addr = base_addr;
    while offset + 2 <= data.len() {
        let p1 = read_16_be(&data[offset..]);

        if is_16bit(p1) {
            let text = decode_16(p1);
            result.push(VisaLine {
                addr,
                raw: data[offset..offset + 2].to_vec(),
                text,
            });
            offset += 2;
            addr += 1;
            continue;
        }

        if offset + 4 > data.len() {
            result.push(VisaLine {
                addr,
                raw: data[offset..].to_vec(),
                text: format!("??? (truncated at 0x{p1:04x})"),
            });
            break;
        }

        let p2 = read_16_be(&data[offset + 2..]);
        let width = visa_width(p1, p2);

        match width {
            InstrWidth::W16 => {
                let text = decode_16(p1);
                result.push(VisaLine {
                    addr,
                    raw: data[offset..offset + 2].to_vec(),
                    text,
                });
                offset += 2;
                addr += 1;
            }
            InstrWidth::W32 => {
                let word32 = (p1 as u32) << 16 | (p2 as u32);
                let text = decode_32(word32);
                result.push(VisaLine {
                    addr,
                    raw: data[offset..offset + 4].to_vec(),
                    text,
                });
                offset += 4;
                addr += 2;
            }
            InstrWidth::W48 => {
                if offset + 6 > data.len() {
                    result.push(VisaLine {
                        addr,
                        raw: data[offset..].to_vec(),
                        text: format!("??? (truncated 48-bit at 0x{p1:04x} 0x{p2:04x})"),
                    });
                    break;
                }
                let p3 = read_16_be(&data[offset + 4..]);
                let word48 = (p1 as u64) << 32 | (p2 as u64) << 16 | (p3 as u64);
                let text = compact_isa_output(&decode_48(word48));
                result.push(VisaLine {
                    addr,
                    raw: data[offset..offset + 6].to_vec(),
                    text,
                });
                offset += 6;
                addr += 3;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// ISA output normalization
// ---------------------------------------------------------------------------

/// Convert ISA decoder uppercase output to standard lowercase format.
/// Removes spaces around `=`, `+`, `-` etc. to match compact style.
fn compact_isa_output(s: &str) -> String {
    let lower = s.to_lowercase();
    // Remove spaces around = but preserve spaces in keywords like "by", "and"
    let mut result = String::with_capacity(lower.len());
    let chars: Vec<char> = lower.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' && i + 1 < chars.len() && chars[i + 1] == '=' {
            // Skip space before =
            i += 1;
        } else if chars[i] == '=' && i + 1 < chars.len() && chars[i + 1] == ' ' {
            result.push('=');
            i += 2; // Skip = and space after
        } else if chars[i] == ' '
            && i + 1 < chars.len()
            && chars[i + 1] == '*'
            && i + 2 < chars.len()
            && chars[i + 2] == ' '
        {
            // " * " → compact to "*"
            result.push('*');
            i += 3;
        } else if chars[i] == ' '
            && i + 1 < chars.len()
            && (chars[i + 1] == '+' || chars[i + 1] == '-')
        {
            // Check if this is an operator (not a negative number)
            if i > 0 && i + 2 < chars.len() && chars[i + 2] == ' ' {
                // " + " or " - " → compact to operator without spaces
                result.push(chars[i + 1]);
                i += 3;
            } else {
                result.push(chars[i]);
                i += 1;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    // Ensure "dm(" and "pm(" have a space: "dm (" and "pm ("
    result.replace("dm(", "dm (").replace("pm(", "pm (")
}

// ---------------------------------------------------------------------------
// Width detection
// ---------------------------------------------------------------------------

fn read_16_be(data: &[u8]) -> u16 {
    ((data[0] as u16) << 8) | (data[1] as u16)
}

/// Check if a first parcel is a 16-bit instruction.
///
/// Determined entirely from the first parcel — no ambiguity with 32/48-bit.
/// Covers all VISA instruction forms.
fn is_16bit(parcel: u16) -> bool {
    // Known exact values
    if matches!(parcel, 0x0001 | 0x0081 | 0x00C1 | 0x1901) {
        return true;
    }
    // RTS/RTI: 0000 101X 1D cccc cL (the reference assembler `rti (db)` = 0bfe, which
    // this used to frame as the start of a 48-bit word; `if ne rts (lr)`
    // = 0aa1). A 48-bit RTS/RTI keeps bit 7 of its first parcel clear.
    if parcel & 0xFE80 == 0x0A80 {
        return true;
    }
    // Type 2c (short compute): bits[15:13] = 110
    if parcel >> 13 == 0b110 {
        return true;
    }
    // Type 3c (short data move): bits[15:12]=1001, bit4=1.
    // Bit 8 belongs to the modify-register field, including M0-M3.
    if (parcel & 0xF010) == 0x9010 {
        return true;
    }
    false
}

/// The 32-bit marker in the second parcel of Types 1, 3, 4/5, 7 and 9:
/// word bits 22:19 = 0111, i.e. a single-function compute field with the
/// reserved CU = 11 and opcode bit 19 set. Every the reference assembler 32-bit form we
/// hold ends 0x3f (normal), 0x3c (short or byte word, Type 3b) or 0x38
/// (byte word, Type 4b). A multifunction compute sets bit 22, so a 48-bit
/// float multiply + dual add/sub with Fs = f8..f15 (bits 21:16 = 0x38..0x3f)
/// is not a marker.
fn is_32bit_marker(parcel2: u16) -> bool {
    parcel2 & 0x78 == 0x38
}

/// Determine VISA instruction width from the first two parcels.
///
/// Uses type-specific rules derived from the SHARC+ instruction decode tree.
/// For types that exist in both 32 and 48-bit form with the same first-parcel
/// value, uses the second parcel's marker (`is_32bit_marker`) to distinguish.
fn visa_width(parcel1: u16, parcel2: u16) -> InstrWidth {
    if is_16bit(parcel1) {
        return InstrWidth::W16;
    }
    let top3 = parcel1 >> 13;
    let sub5 = (parcel1 >> 8) & 0x1F;
    let bit7 = (parcel1 >> 7) & 1;

    match top3 {
        0b000 => match sub5 {
            // Types that are always 48-bit (no 32-bit variant)
            0b00000 // Type 21a/22a (NOP/IDLE 48-bit)
            | 0b00010 // Type 6b (imm shift alt)
            | 0b00110 // Type 8a (direct jump, absolute)
            | 0b00111 // Type 8b (direct jump, PC-relative)
            | 0b01010 // Type 11a (RTS/RTI + compute)
            | 0b01011 // Type 11b (RTS/RTI alt)
            | 0b01100 // Type 12a (LCNTR=imm16, DO)
            | 0b01101 // Type 12b (LCNTR=ureg, DO)
            | 0b01110 // Type 13 (DO UNTIL)
            | 0b10100 // Type 18 (BIT ops)
            | 0b10101 // Enhanced MODIFY with scaling (nw/sw)
            | 0b10110 // Type 19 (MODIFY/BITREV imm32)
            | 0b10111 // Type 20 (push/pop/flush)
            | 0b11000 // Type 25a (CJUMP)
            | 0b11010 // Type 14b read (absolute address with a width)
            | 0b11011 // Type 14b write
            => InstrWidth::W48,
            _ if (sub5 >> 2) == 0b100 => InstrWidth::W48, // Type 14a/d
            // Type 2: sub5=00001, bit7=1→32b, bit7=0→48b
            0b00001 => {
                if bit7 == 1 { InstrWidth::W32 } else { InstrWidth::W48 }
            }
            // Type 7: the second parcel's low six bits are the 32-bit marker;
            // p1 bit 7 is the (NW) flag. The reference assembler: 043e603f is 32-bit,
            // 043e61000000 and 043e611308a7 are 48-bit.
            0b00100 => {
                if is_32bit_marker(parcel2) { InstrWidth::W32 } else { InstrWidth::W48 }
            }
            // Type 17: sub5=01111, bit7=1→32b, bit7=0→48b
            0b01111 => {
                if bit7 == 1 { InstrWidth::W32 } else { InstrWidth::W48 }
            }
            // Type 9a: sub5=01000, ambiguous → p2 check
            // Type 9b: sub5=01001, ambiguous → p2 check
            _ => {
                if is_32bit_marker(parcel2) {
                    InstrWidth::W32
                } else {
                    InstrWidth::W48
                }
            }
        },
        0b001 => {
            // Type 1: bit7=1→32b (Type 1b), bit7=0→48b (Type 1a)
            // Actually Type 1 always has compute in 48-bit form.
            // Check p2 for ambiguous values.
            if is_32bit_marker(parcel2) {
                InstrWidth::W32
            } else {
                InstrWidth::W48
            }
        }
        0b010 => {
            // Type 3: has 3a (48), 3b (32), 3d (48) variants
            // The Type 3b marker also applies to post-modify I4 accesses
            // (sub5=11001), including M4/M5 where bit7 is clear.
            // The second parcel's low six bits are the 32-bit marker
            // (0x3f normal word, 0x3c short/byte word). The reference assembler 48-bit
            // forms with the same first parcel: `r2 = dm(i4, m6)` =
            // 59be01000000, `r0=dm(i4,m6)(sw)` = 59be40300000.
            if is_32bit_marker(parcel2) {
                InstrWidth::W32
            } else {
                InstrWidth::W48
            }
        }
        0b011 => {
            // Types 4, 5: the second parcel's low six bits are the 32-bit
            // marker (the reference assembler: 6a7e113f is 32-bit, 6a7e11000000 and
            // 6a7e11588f83 are 48-bit with the same first parcel).
            if is_32bit_marker(parcel2) {
                InstrWidth::W32
            } else {
                InstrWidth::W48
            }
        }
        0b100 => {
            // Type 3c (16-bit) already handled by is_16bit.
            let bit12 = (parcel1 >> 12) & 1;
            if bit12 == 0 {
                // Type 6a: shift + mem (always 48-bit)
                InstrWidth::W48
            } else {
                // bit12=1: Type 16a (48-bit, p1 bits 4:0 = 0) or the
                // 32-bit Type 15b/16b (bits 4:0 = 01000 / 00100). Holds
                // for all 2,165 such reference assembler rows: 48-bit
                // `dm(i2, m0) = 0x1c0111` = 9400001c0111, 32-bit
                // `pm ( i10 , m8 ) = 0x0` = 94240000.
                if parcel1 & 0x1F == 0 {
                    InstrWidth::W48
                } else {
                    InstrWidth::W32
                }
            }
        }
        0b101 => InstrWidth::W48, // Type 15a (always 48)
        0b110 => InstrWidth::W48, // Type 10a (always 48) — Type 2c handled by is_16bit
        0b111 => InstrWidth::W48, // Type 10b (always 48)
        _ => InstrWidth::W48,
    }
}

// ---------------------------------------------------------------------------
// 16-bit instruction decoders
// ---------------------------------------------------------------------------

fn decode_16(parcel: u16) -> String {
    match parcel {
        0x0001 => return "nop".into(),
        0x0081 => return "idle".into(),
        0x00C1 => return "emuidle".into(),
        p if p & 0xFE80 == 0x0A80 => {
            let op = if p & 0x0100 != 0 { "rti" } else { "rts" };
            let db = match (p & 0x40 != 0, p & 1 != 0) {
                (true, true) => " (db,lr)",
                (true, false) => " (db)",
                (false, true) => " (lr)",
                (false, false) => "",
            };
            let cond = (p >> 1) & 0x1F;
            return if cond == 31 {
                format!("{op}{db}")
            } else {
                format!("if {} {op}{db}", cond_name_lower(cond))
            };
        }
        0x1901 => return "rframe".into(),
        _ => {}
    }

    if parcel >> 13 == 0b110 {
        return decode_type2c(parcel);
    }
    if (parcel & 0xF010) == 0x9010 {
        return decode_type3c(parcel);
    }

    format!("??? 0x{parcel:04x}")
}

/// Type 2c: short compute (16-bit).
/// bits[15:13]=110, bits[12:0]=opcode+rn+ry.
fn decode_type2c(parcel: u16) -> String {
    decode_short_compute(parcel)
}

/// Decode a short compute from Type 2c.
/// Layout: bits[12:8]=opcode, bits[7:4]=rn, bits[3:0]=ry.
/// For most ops: rn is both destination and first source (rn = rn OP ry).
///
/// Opcode mapping mirrors `try_type2c` in `visa_encode.rs`; both came
/// from cross-checking against easm21k and ADI elfdump output. The
/// previous table had several rows in the wrong slot (op 4 was
/// labelled FAdd, op 8 labelled int Neg, op 11 labelled FPass, op 15
/// labelled int Not) which made encode/decode round-trip cleanly but
/// disagreed with what the SHARC+ core actually executes.
fn decode_short_compute(parcel: u16) -> String {
    let op = (parcel >> 8) & 0x1F;
    let rn = (parcel >> 4) & 0xF;
    let ry = parcel & 0xF;
    match op {
        0 => format!("r{rn}=r{rn}+r{ry}"),
        1 => format!("r{rn}=r{rn}-r{ry}"),
        2 => format!("r{rn}=pass r{ry}"),
        3 => format!("comp (r{rn},r{ry})"),
        4 => format!("r{rn}=not r{ry}"),
        5 => format!("r{rn}=r{ry}+1"),
        6 => format!("r{rn}=r{ry}-1"),
        // The reference assembler `r0 = r0 * r1 (ssi)` = c701; `r0 = r0 + r1 + ci` has
        // no 16-bit form (01805001).
        7 => format!("r{rn}=r{rn}*r{ry} (ssi)"),
        8 => format!("f{rn}=f{rn}+f{ry}"),
        9 => format!("f{rn}=f{rn}-f{ry}"),
        10 => format!("f{rn}=float r{ry}"),
        11 => format!("comp (f{rn},f{ry})"),
        12 => format!("r{rn}=r{rn} and r{ry}"),
        13 => format!("r{rn}=r{rn} or r{ry}"),
        14 => format!("r{rn}=r{rn} xor r{ry}"),
        15 => format!("f{rn}=f{rn}*f{ry}"),
        _ => format!("shortcompute op={op} r{rn} r{ry}"),
    }
}

/// Type 3c: short data move (16-bit).
/// bits[15:12]=1001, bit[4]=1, variable register fields between.
fn decode_type3c(parcel: u16) -> String {
    // Bit layout:
    //   bits[11:9] = i[2:0] (index register)
    //   bits[8:6]  = m[2:0] (modify register)
    //   bit[5]     = d (0=read, 1=write)
    //   bits[3:0]  = dreg (register number)
    let i_val = (parcel >> 9) & 7;
    let m_val = (parcel >> 6) & 7;
    let d_bit = (parcel >> 5) & 1;
    let dreg_val = parcel & 0xF;

    let ir = format!("i{i_val}");
    let mr = format!("m{m_val}");
    let dr = format!("r{dreg_val}");

    if d_bit == 1 {
        format!("dm ({ir},{mr})={dr}")
    } else {
        format!("{dr}=dm ({ir},{mr})")
    }
}

// ---------------------------------------------------------------------------
// 32-bit instruction decoder
// ---------------------------------------------------------------------------

/// Decode a 32-bit VISA instruction.
///
/// The 32-bit word is formed from two big-endian parcels: p1 (high) and p2 (low).
/// Dispatch is on the same top3 + sub5 fields used for width detection.
fn decode_32(word: u32) -> String {
    let p1 = (word >> 16) as u16;
    let p2 = word as u16;
    let top3 = p1 >> 13;

    match top3 {
        0b000 => decode_32_group0(p1, p2),
        // Type 1 (dual move): the 48-bit word with an empty compute and the
        // marker in its place (`dm ( i0 , m0 ) = r4 , pm ( i8 , m10 ) =
        // r8` = 3028143f, 48-bit 302814000000).
        0b001 => compact_isa_output(&crate::disasm::decode_instruction(
            u64::from(word & !0x3F) << 16,
        )),
        0b010 => decode_32_type3b(p1, p2),
        0b011 => decode_32_group3(p1, p2),
        0b100 => decode_32_group4(p1, p2),
        _ => format!("??? 0x{p1:04x} 0x{p2:04x}"),
    }
}

// ---------------------------------------------------------------------------
// Group 0 (top3=000): Type 2b, 7b, 9b, 17b
// ---------------------------------------------------------------------------

fn decode_32_group0(p1: u16, p2: u16) -> String {
    let sub5 = (p1 >> 8) & 0x1F;
    match sub5 {
        0b00001 => decode_32_type2b(p1, p2),
        0b00100 => decode_32_type7b(p1, p2),
        0b01000 => decode_32_type9a(p1, p2),
        0b01001 => decode_32_type9b(p1, p2),
        0b01111 => decode_32_type17b(p1, p2),
        _ => format!("??? 0x{p1:04x} 0x{p2:04x}"),
    }
}

/// Type 2b: 32-bit compute (unconditional or conditional).
///
/// Layout: `[31:29]=000  [28:24]=00001  [23]=1(32b flag)  [22:0]=compute`
///
/// The 23-bit compute field uses the same structure as the 48-bit ISA:
///   - `bit[22]`    = multi (0 for single-function)
///   - `bits[21:20]` = cu (0=ALU, 1=MUL, 2=SHIFT)
///   - `bits[19:12]` = opcode
///   - `bits[11:8]`  = Rn (destination)
///   - `bits[7:4]`   = Rx (source 1)
///   - `bits[3:0]`   = Ry (source 2)
///
/// Opcodes match the 48-bit ISA for ALU, but MUL and SHIFT include additional
/// opcodes not present in the 48-bit tables (e.g. MUL 0x70 = SSI, SHIFT
/// 0x88 = LEFTZ).  This decoder handles the full 32-bit opcode set.
fn decode_32_type2b(p1: u16, p2: u16) -> String {
    let compute = ((p1 as u32 & 0x7F) << 16) | p2 as u32;
    decode_32_compute(compute)
}

/// Decode a 23-bit compute field from a 32-bit instruction.
fn decode_32_compute(field: u32) -> String {
    if field == 0 {
        return "nop".into();
    }
    // MR-register transfer (`Rn = MRxF/B` or `MRxF/B = Rn`):
    // `bits[22:17]=0b100000  bit[16]=T  bits[15:12]=Ai
    //  bits[11:8]=Rk  bits[7:0]=0`. Recognise it before the regular
    // ALU/MUL/SHIFT split so the round-trip matches the 48-bit
    // disassembler.
    if (field >> 17) & 0x3F == 0b100000 && field & 0xFF == 0 {
        let t = (field >> 16) & 1;
        let ai = (field >> 12) & 0xF;
        let rk = (field >> 8) & 0xF;
        let mr = match ai {
            0 => "mr0f",
            1 => "mr1f",
            2 => "mr2f",
            4 => "mr0b",
            5 => "mr1b",
            6 => "mr2b",
            _ => return format!("mr-transfer(0x{field:06x})"),
        };
        return if t == 0 {
            format!("r{rk}={mr}")
        } else {
            format!("{mr}=r{rk}")
        };
    }
    let cu = (field >> 20) & 3;
    let opcode = ((field >> 12) & 0xFF) as u8;
    let rn = (field >> 8) & 0xF;
    let rx = (field >> 4) & 0xF;
    let ry = field & 0xF;

    match cu {
        0 => decode_32_alu(opcode, rn, rx, ry),
        1 => decode_32_mul(opcode, rn, rx, ry),
        2 => decode_32_shift(opcode, rn, rx, ry),
        // A multifunction field (bit 22, CU = 11): print it through the
        // 48-bit decoder as an unconditional Type 2 word (the reference assembler `f10 = f0
        // * f4 , f12 = f8 + f12 , f8 = f8 - f12` = 01f8ac00).
        _ if field & (1 << 22) != 0 => compact_isa_output(&crate::disasm::decode_instruction(
            (0x01u64 << 40) | (31u64 << 33) | u64::from(field),
        )),
        _ => format!("compute(0x{field:06x})"),
    }
}

fn decode_32_alu(opcode: u8, rn: u32, rx: u32, ry: u32) -> String {
    let fp = opcode & 0x80 != 0;
    let r = |i: u32| -> String {
        if fp {
            format!("f{i}")
        } else {
            format!("r{i}")
        }
    };

    match opcode {
        // MR register transfers (reads at 0x00..0x06, writes at
        // 0x10..0x16) are encoded with `bit[22]=1` and a dedicated
        // sub-layout, decoded ahead of this function in
        // `decode_32_compute`. Reaching this arm means `bit[22]=0`,
        // i.e. a real ALU compute. Fall through to the integer/FP
        // tables below.
        // Standard ALU (integer)
        0x01 => format!("r{rn}=r{rx}+r{ry}"),
        0x02 => format!("r{rn}=r{rx}-r{ry}"),
        0x05 => format!("r{rn}=r{rx}+r{ry}+ci"),
        0x06 => format!("r{rn}=r{rx}-r{ry}+ci-1"),
        0x09 => format!("r{rn}=(r{rx}+r{ry})/2"),
        0x0A => format!("comp (r{rx},r{ry})"),
        0x0B => format!("compu (r{rx},r{ry})"),
        0x21 => format!("r{rn}=pass r{rx}"),
        0x22 => format!("r{rn}=-r{rx}"),
        0x25 => format!("r{rn}=r{rx}+ci"),
        0x26 => format!("r{rn}=r{rx}+ci-1"),
        0x29 => format!("r{rn}=r{rx}+1"),
        0x2A => format!("r{rn}=r{rx}-1"),
        0x30 => format!("r{rn}=abs r{rx}"),
        0x40 => format!("r{rn}=r{rx} and r{ry}"),
        0x41 => format!("r{rn}=r{rx} or r{ry}"),
        0x42 => format!("r{rn}=r{rx} xor r{ry}"),
        0x43 => format!("r{rn}=not r{rx}"),
        0x61 => format!("r{rn}=min(r{rx},r{ry})"),
        0x62 => format!("r{rn}=max(r{rx},r{ry})"),
        0x63 => format!("r{rn}=clip r{rx} by r{ry}"),
        // Standard ALU (floating-point)
        0x81 => format!("f{rn}=f{rx}+f{ry}"),
        0x82 => format!("f{rn}=f{rx}-f{ry}"),
        0x89 => format!("f{rn}=(f{rx}+f{ry})/2"),
        0x8A => format!("comp (f{rx},f{ry})"),
        0x91 => format!("f{rn}=abs(f{rx}+f{ry})"),
        0x92 => format!("f{rn}=abs(f{rx}-f{ry})"),
        0xA1 => format!("f{rn}=pass f{rx}"),
        0xA2 => format!("f{rn}=-f{rx}"),
        0xA5 => format!("f{rn}=rnd f{rx}"),
        0xAD => format!("r{rn}=mant f{rx}"),
        0xB0 => format!("f{rn}=abs f{rx}"),
        0xBD => format!("{}=scalb {} by r{ry}", r(rn), r(rx)),
        0xC1 => format!("r{rn}=logb {}", r(rx)),
        0xC4 => format!("{}=recips {}", r(rn), r(rx)),
        0xC5 => format!("{}=rsqrts {}", r(rn), r(rx)),
        0xC9 => format!("r{rn}=fix f{rx}"),
        0xCA => format!("f{rn}=float r{rx}"),
        0xCD => format!("r{rn}=trunc f{rx}"),
        0xD9 => format!("r{rn}=fix f{rx} by r{ry}"),
        0xDA => format!("f{rn}=float r{rx} by r{ry}"),
        0xDD => format!("r{rn}=trunc f{rx} by r{ry}"),
        0xE0 => format!("f{rn}=f{rx} copysign f{ry}"),
        0xE1 => format!("f{rn}=min(f{rx},f{ry})"),
        0xE2 => format!("f{rn}=max(f{rx},f{ry})"),
        0xE3 => format!("f{rn}=clip f{rx} by f{ry}"),
        _ => format!("alu 0x{opcode:02x} r{rn} r{rx} r{ry}"),
    }
}

fn decode_32_mul(opcode: u8, rn: u32, rx: u32, ry: u32) -> String {
    // 32-bit VISA compressed multiplier. The 8-bit MULOP field
    // mirrors the 48-bit encoder's templates.
    match opcode {
        0x08 => format!("r{rn}=sat mrf"),
        0x0A => format!("r{rn}=sat mrb"),
        0x14 => "mrf=0".into(),
        0x16 => "mrb=0".into(),
        0x30 => format!("f{rn}=f{rx}*f{ry}"),
        0x4C => format!("mrf=r{rx}*r{ry} (uuf)"),
        0x70 => format!("r{rn}=r{rx}*r{ry} (ssi)"),
        0x74 => format!("mrf=r{rx}*r{ry} (ssi)"),
        0x78 => format!("r{rn}=r{rx}*r{ry} (ssf)"),
        0x7C => format!("mrf=r{rx}*r{ry} (ssf)"),
        0x7E => format!("mrb=r{rx}*r{ry} (ssf)"),
        0xB8 => format!("r{rn}=mrf+r{rx}*r{ry} (ssf)"),
        0xBC => format!("mrf=mrf+r{rx}*r{ry} (ssf)"),
        0xBE => format!("mrb=mrb+r{rx}*r{ry} (ssf)"),
        0xFC => format!("mrf=mrf-r{rx}*r{ry} (ssf)"),
        0xFE => format!("mrb=mrb-r{rx}*r{ry} (ssf)"),
        0x40..=0xFF => crate::disasm::decode_fixed_mul(opcode, rn, rx, ry).to_lowercase(),
        _ => format!("mul 0x{opcode:02x} r{rn} r{rx} r{ry}"),
    }
}

fn decode_32_shift(opcode: u8, rn: u32, rx: u32, ry: u32) -> String {
    // 32-bit VISA compressed shifter. The 8-bit SHIFTOP field
    // mirrors the 48-bit encoder's templates.
    match opcode {
        0x00 => format!("r{rn}=lshift r{rx} by r{ry}"),
        0x04 => format!("r{rn}=ashift r{rx} by r{ry}"),
        0x08 => format!("r{rn}=rot r{rx} by r{ry}"),
        0x20 => format!("r{rn}=r{rn} or lshift r{rx} by r{ry}"),
        0x24 => format!("r{rn}=r{rn} or ashift r{rx} by r{ry}"),
        0x40 => format!("r{rn}=fext r{rx} by r{ry}"),
        0x44 => format!("r{rn}=fdep r{rx} by r{ry}"),
        0x80 => format!("r{rn}=exp r{rx}"),
        0x84 => format!("r{rn}=exp r{rx} (ex)"),
        0x88 => format!("r{rn}=leftz r{rx}"),
        0x8C => format!("r{rn}=lefto r{rx}"),
        0x90 => format!("r{rn}=fpack f{rx}"),
        0x94 => format!("f{rn}=funpack r{rx}"),
        0xC0 => format!("r{rn}=bset r{rx} by r{ry}"),
        0xC4 => format!("r{rn}=bclr r{rx} by r{ry}"),
        0xC8 => format!("r{rn}=btgl r{rx} by r{ry}"),
        0xCC => format!("btst r{rx} by r{ry}"),
        0x70 => format!("r{rn}=bffwrp"),
        0x7C => format!("bffwrp=r{ry}"),
        _ => format!("shift 0x{opcode:02x} r{rn} r{rx} r{ry}"),
    }
}

/// Type 7b: 32-bit modify with result assignment.
///
/// `i_dest = MODIFY(i_src, m)`
///
/// Layout (in full 32-bit word = `(p1 << 16) | p2`):
///   - `{p1[0], p2[15:14]}` = i_src (3 bits)
///   - `p2[13:11]`          = m (3 bits)
///   - `p2[10:8]`           = dest_xor (i_dest = i_src XOR dest_xor)
///   - `p1[6]`              = g (0=DAG1, 1=DAG2)
///   - `p1[5:1]`            = cond (31=TRUE)
fn decode_32_type7b(p1: u16, p2: u16) -> String {
    let cond = (p1 >> 1) & 0x1F;
    let g = (p1 >> 6) & 1;
    let i_src = ((p1 & 1) << 2) | ((p2 >> 14) & 3);
    let m_field = (p2 >> 11) & 7;
    let dest_xor = (p2 >> 8) & 7;
    let i_dest = i_src ^ dest_xor;

    let base = if g == 1 { 8u16 } else { 0 };
    let src = i_src + base;
    let dst = i_dest + base;
    let m = m_field + base;

    let mut out = String::new();
    if cond != 31 {
        out.push_str("if ");
        out.push_str(&cond_name_lower(cond));
        out.push(' ');
    }
    out.push_str(&format!("i{dst}=modify (i{src},m{m})"));
    out
}

/// Type 9a: 32-bit indirect jump/call, the top of the 48-bit word with the
/// 0x3f marker: CALL p1[7], cond p1[5:1], I = 8 + {p1[0], p2[15:14]},
/// M = 8 + p2[13:11], (DB) p2[10]. The reference assembler: `call (m14, i12)` = 08bf303f,
/// `if lt jump (m14, i8) (db)` = 0802343f.
fn decode_32_type9a(p1: u16, p2: u16) -> String {
    let j = (p1 >> 7) & 1;
    let cond = (p1 >> 1) & 0x1F;
    let i = 8 + (((p1 & 1) << 2) | ((p2 >> 14) & 3));
    let m = 8 + ((p2 >> 11) & 7);
    let db = (p2 >> 10) & 1;
    let op = if j == 1 { "call" } else { "jump" };
    let mut out = String::new();
    if cond != 31 {
        out.push_str("if ");
        out.push_str(&cond_name_lower(cond));
        out.push(' ');
    }
    out.push_str(&format!("{op} (m{m},i{i})"));
    if db == 1 {
        out.push_str(" (db)");
    }
    out
}

/// Type 9b: 32-bit indirect jump/call.
///
/// Layout:
///   - `p1[6]`     = j (0=JUMP, 1=CALL)
///   - `p1[5:1]`   = cond (5 bits, typically 31=TRUE)
///   - `p2[13:11]` = pmm field (M register = M(8+field))
///   - `p2[15:14]` = pmi offset (I register = I(12+offset))
///   - `p2[10]`    = DB (delayed branch) flag
fn decode_32_type9b(p1: u16, p2: u16) -> String {
    let j = (p1 >> 6) & 1;
    let cond = (p1 >> 1) & 0x1F;
    let pmi_off = (p2 >> 14) & 3;
    let pmm = (p2 >> 11) & 7;
    let db = (p2 >> 10) & 1;

    let op = if j == 1 { "call" } else { "jump" };
    let m_reg = format!("m{}", 8 + pmm);
    let i_reg = format!("i{}", 12 + pmi_off);

    let mut out = String::new();
    if cond != 31 {
        out.push_str("if ");
        out.push_str(&cond_name_lower(cond));
        out.push(' ');
    }
    out.push_str(op);
    out.push_str(&format!(" ({m_reg},{i_reg})"));
    if db == 1 {
        out.push_str(" (db)");
    }
    out
}

/// Type 17b: 32-bit ureg = imm16.
///
/// Layout: `p1[6:0]` = ureg code (7 bits), `p2` = 16-bit immediate.
fn decode_32_type17b(p1: u16, p2: u16) -> String {
    let ureg_code = (p1 & 0x7F) as u32;
    let imm = p2 as i16;
    let ur = ureg_name_lower(ureg_code);
    if imm < 0 {
        format!("{ur}=-0x{:x}", -(imm as i32))
    } else {
        format!("{ur}=0x{:x}", imm as u16)
    }
}

// ---------------------------------------------------------------------------
// Type 3b (top3=010): ureg <-> DM/PM with register addressing
// ---------------------------------------------------------------------------

/// Type 3b: ureg transfer with register modify.
///
/// Layout:
///   - `p1[12]`   = post-modify (0=pre-modify, 1=post-modify)
///   - `p1[11:9]` = i-register index (I0-I7 for DM)
///   - `p1[7:6]`  = m-register offset (m = field + 4, giving M4-M7)
///   - `p1[5:1]`  = cond (5 bits)
///   - `p2[15]`   = direction (0=read, 1=write)
///   - `p2[14]`   = short-word flag (when p2[1:0]=00)
///   - `p2[13:7]` = ureg code (7 bits)
///   - `p2[1:0]`  = access width (11=NW, 00=BW or SW depending on p2[14])
fn decode_32_type3b(p1: u16, p2: u16) -> String {
    let post_modify = p1 & 0x1000 != 0;
    let i_field = (p1 >> 9) & 7;
    // M is three bits (p1 8:6) and p1 bit 0 is G (PM, DAG2): the reference assembler
    // `curlcntr = dm ( i4 , m2 )` = 58be33bf, `pm(i8, m10) = f4` =
    // 50bf823f. M used to be read as two bits plus 4.
    let m_field = (p1 >> 6) & 7;
    let pm = p1 & 1 == 1;
    let dag = if pm { 8 } else { 0 };
    let cond = (p1 >> 1) & 0x1F;
    let d = (p2 >> 15) & 1; // 0=read, 1=write
    let sw_flag = (p2 >> 14) & 1;
    let ureg_code = ((p2 >> 7) & 0x7F) as u32;
    let width_bits = p2 & 3;

    let mem = if pm { "pm" } else { "dm" };
    let i_reg = format!("i{}", i_field + dag);
    let m_reg = format!("m{}", m_field + dag);
    let ur = ureg_name_lower(ureg_code);

    // Access width: p2[1:0] 11 normal/long, 00 short/byte, 01 sign-extended
    // short/byte; p2[14] picks long/short (the reference assembler 513e403f `(lw)`,
    // 503e403c `(sw)`, 503e003c `(bw)`, 503e403d `(swse)`, 51be00bd
    // `(bwse)`).
    let suffix = match (width_bits, sw_flag) {
        (3, 0) => "",
        (3, _) => "(lw)",
        (0, 1) => "(sw)",
        (0, _) => "(bw)",
        (1, 1) => "(swse)",
        (1, _) => "(bwse)",
        _ => "(?)",
    };

    let mut out = String::new();
    if cond != 31 {
        out.push_str("if ");
        out.push_str(&cond_name_lower(cond));
        out.push(' ');
    }

    let address = if post_modify {
        format!("{i_reg},{m_reg}")
    } else {
        format!("{m_reg},{i_reg}")
    };
    if d == 1 {
        out.push_str(&format!("{mem} ({address})={ur}"));
        if !suffix.is_empty() {
            out.push(' ');
        }
    } else {
        out.push_str(&format!("{ur}={mem} ({address})"));
    }
    out.push_str(suffix);
    out
}

// ---------------------------------------------------------------------------
// Group 3 (top3=011): ureg-to-ureg moves, Type 4b
// ---------------------------------------------------------------------------

fn decode_32_group3(p1: u16, p2: u16) -> String {
    let sub5 = (p1 >> 8) & 0x1F;

    // sub5[4:3] = 10 -> ureg-to-ureg move (Types 5a-32b)
    // sub5[2:0] = source register group
    if (sub5 >> 3) == 0b10 {
        return decode_32_ureg_move(p1, p2);
    }

    // Type 4b: sub5 = 0 iii G (p1[12] = 0, I at p1[11:9], p1[8] = PM).
    if sub5 & 0b10000 == 0 {
        decode_32_type4b(p1, p2)
    } else {
        format!("??? type4b 0x{p1:04x} 0x{p2:04x}")
    }
}

/// 32-bit ureg-to-ureg move: `destination_ureg = source_ureg`.
///
/// Source register is encoded with group from `sub5[2:0]` and index from
/// scattered bits `{p1[7], p1[6], p1[0], p2[15]}`.
///
/// Destination is a 7-bit ureg code at `p2[13:7]`.
fn decode_32_ureg_move(p1: u16, p2: u16) -> String {
    let sub5 = (p1 >> 8) & 0x1F;
    let cond = (p1 >> 1) & 0x1F;
    let src_group = sub5 & 7;
    let src_idx =
        (((p1 >> 7) & 1) << 3) | (((p1 >> 6) & 1) << 2) | ((p1 & 1) << 1) | ((p2 >> 15) & 1);
    let dst_ureg_code = ((p2 >> 7) & 0x7F) as u32;

    let src_ureg_code = (src_group << 4) as u32 | src_idx as u32;
    let src = ureg_name_lower(src_ureg_code);
    let dst = ureg_name_lower(dst_ureg_code);

    let mut out = String::new();
    if cond != 31 {
        out.push_str("if ");
        out.push_str(&cond_name_lower(cond));
        out.push(' ');
    }
    out.push_str(&format!("{dst}={src}"));
    out
}

/// Type 4b: 32-bit dreg <-> DM|PM with immediate offset.
///
/// Layout:
///   - `p1[11:9]`  = i-register index
///   - `p1[7]`     = direction (0=read, 1=write)
///   - `p1[6]`     = post-modify (1 = dm(Ii, offset))
///   - `p1[5:1]`   = cond (5 bits)
///   - `p2[15:11]` = offset (5-bit unsigned)
///   - `p2[10:7]`  = dreg (4-bit register index)
///   - `p2[2]`     = access width (1=SW, 0=BW)
///   - `p2[1:0]`   = width mode (00=sub-word, 11=normal word)
fn decode_32_type4b(p1: u16, p2: u16) -> String {
    // p1 is the top parcel of the 48-bit Type 4 word: d at p1[7]. The reference assembler
    // `if sz r15=dm(-3,i6)` = 6c11efbf is a read with p1[10] set by I6.
    // p1[8] = G, PM through DAG2 (`r0 = pm ( i8 , 0x0 )` = 617e003f).
    let pm = (p1 >> 8) & 1 == 1;
    let i_field = ((p1 >> 9) & 0x7) + if pm { 8 } else { 0 };
    let d_bit = (p1 >> 7) & 1;
    let cond = (p1 >> 1) & 0x1F;

    let dreg_val = ((p2 >> 7) & 0xF) as u32;
    // 6-bit signed offset: {p1[0], p2[15:11]}
    let raw6 = (((p1 & 1) as i32) << 5) | ((p2 >> 11) & 0x1F) as i32;
    let offset = (raw6 << 26) >> 26; // sign-extend 6-bit

    let width_bits = p2 & 3;
    let sw_bit = (p2 >> 2) & 1;
    // 01 = sign-extending (`( swse )` 687e103d, `( bwse )` 643fd839).
    let suffix = match (width_bits, sw_bit) {
        (0, 1) => "(sw)",
        (0, _) => "(bw)",
        (1, 1) => "(swse)",
        (1, _) => "(bwse)",
        _ => "",
    };

    // Operand order depends on p1[6]:
    //   p1[6]=0: offset first → dm (offset, Ii)
    //   p1[6]=1: I first → dm (Ii, offset)
    let off_first = (p1 >> 6) & 1 == 0;

    let off_str = if offset < 0 {
        format!("-0x{:x}", -offset)
    } else {
        format!("0x{:x}", offset as u32)
    };

    let mem = if pm { "pm" } else { "dm" };
    let mut out = String::new();
    if cond != 31 {
        out.push_str("if ");
        out.push_str(&cond_name_lower(cond));
        out.push(' ');
    }

    if off_first {
        if d_bit == 1 {
            out.push_str(&format!("{mem} ({off_str},i{i_field})=r{dreg_val}"));
        } else {
            out.push_str(&format!("r{dreg_val}={mem} ({off_str},i{i_field})"));
        }
    } else {
        if d_bit == 1 {
            out.push_str(&format!("{mem} (i{i_field},{off_str})=r{dreg_val}"));
        } else {
            out.push_str(&format!("r{dreg_val}={mem} (i{i_field},{off_str})"));
        }
    }
    if !suffix.is_empty() {
        if d_bit == 1 {
            out.push(' ');
        }
        out.push_str(suffix);
    }
    out
}

// ---------------------------------------------------------------------------
// Group 4 (top3=100): DM with immediate offset / Type 16b
// ---------------------------------------------------------------------------

fn decode_32_group4(p1: u16, p2: u16) -> String {
    // p1 bit 5 is G: PM through DAG2 (I8-I15, M8-M15).
    let pm = (p1 >> 5) & 1 == 1;
    let mem = if pm { "pm" } else { "dm" };
    let dag = if pm { 8 } else { 0 };
    let i_field = ((p1 >> 9) & 7) + dag;
    let d_bit = (p1 >> 8) & 1;
    let lw_flag = (p1 >> 7) & 1;

    // Type 16b: DM|PM(Ii, Mm) = imm16, p1 bits 4:0 = 00100. The reference assembler:
    // `dm ( i7 , m7 ) = 0x1000` = 9fc41000, `pm ( i10 , m8 ) = 0x0` =
    // 94240000, `dm(i0, m0) = -0x1` = 9004ffff.
    if p1 & 0x1F == 0x04 {
        let m_field = ((p1 >> 6) & 7) + dag;
        let imm = p2 as i16;
        return if imm < 0 {
            format!("{mem} (i{i_field},m{m_field})=-0x{:x}", -i32::from(imm))
        } else {
            format!("{mem} (i{i_field},m{m_field})=0x{imm:x}")
        };
    }

    // Standard: ureg <-> dm(offset, Ii)
    let ureg_code = ((p2 >> 7) & 0x7F) as u32;
    let raw_off = (p2 & 0x7F) as i32;
    let offset = if raw_off >= 64 {
        raw_off - 128
    } else {
        raw_off
    };
    let ur = ureg_name_lower(ureg_code);

    // Access width suffix
    let suffix = if lw_flag == 1 { "(lw)" } else { "" };

    if d_bit == 1 {
        // Write: dm(offset, Ii)=ureg
        if offset < 0 {
            if suffix.is_empty() {
                format!("{mem} (-0x{:x},i{i_field})={ur}", -offset)
            } else {
                format!("{mem} (-0x{:x},i{i_field})={ur} {suffix}", -offset)
            }
        } else {
            if suffix.is_empty() {
                format!("{mem} (0x{offset:x},i{i_field})={ur}")
            } else {
                format!("{mem} (0x{offset:x},i{i_field})={ur} {suffix}")
            }
        }
    } else {
        // Read: ureg=dm(offset, Ii)
        if offset < 0 {
            if suffix.is_empty() {
                format!("{ur}={mem} (-0x{:x},i{i_field})", -offset)
            } else {
                format!("{ur}={mem} (-0x{:x},i{i_field}){suffix}", -offset)
            }
        } else {
            if suffix.is_empty() {
                format!("{ur}={mem} (0x{offset:x},i{i_field})")
            } else {
                format!("{ur}={mem} (0x{offset:x},i{i_field}){suffix}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lowercase register name helpers
// ---------------------------------------------------------------------------

/// Convert a ureg code to a lowercase register name.
///
/// Uses the standard SHARC+ universal-register encoding:
///   - group 0: R0-R15     group 1: I0-I15   group 2: M0-M15
///   - group 3: L0-L15     group 4: B0-B15   group 5: S0-S15
///   - groups 6-7: system registers (FADDR, PC, MODE1, etc.)
fn ureg_name_lower(code: u32) -> String {
    crate::disasm::ureg_name(code).to_lowercase()
}

/// Lowercase condition name.
fn cond_name_lower(code: u16) -> String {
    match code & 0x1F {
        0 => "eq".into(),
        1 => "lt".into(),
        2 => "le".into(),
        3 => "ac".into(),
        4 => "av".into(),
        5 => "mv".into(),
        6 => "ms".into(),
        7 => "sv".into(),
        8 => "sz".into(),
        9 => "flag0_in".into(),
        10 => "flag1_in".into(),
        11 => "flag2_in".into(),
        12 => "flag3_in".into(),
        13 => "tf".into(),
        14 => "bm".into(),
        15 => "not lce".into(),
        16 => "ne".into(),
        17 => "ge".into(),
        18 => "gt".into(),
        19 => "not ac".into(),
        20 => "not av".into(),
        21 => "not mv".into(),
        22 => "not ms".into(),
        23 => "not sv".into(),
        24 => "not sz".into(),
        25 => "not flag0_in".into(),
        26 => "not flag1_in".into(),
        27 => "not flag2_in".into(),
        28 => "not flag3_in".into(),
        29 => "not tf".into(),
        30 => "nbm".into(),
        31 => "true".into(),
        _ => format!("cond({code})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_data_moves_preserve_vendor_stream_boundaries() {
        // The reference assembler: the M0 load
        // is one parcel, followed by NOP, a 32-bit shift, and a 48-bit
        // immediate load. Requiring bit 8 swallowed the following NOP.
        let code = [
            0x01, 0x80, 0x14, 0x01, 0xc0, 0x01, 0x0f, 0x82, 0x12, 0x34, 0x90, 0x14, 0x00, 0x01,
            0x01, 0xa0, 0x04, 0x01, 0x0f, 0x03, 0x12, 0x34, 0x56, 0x78, 0x0a, 0xbe,
        ];
        let lines = disassemble_visa(&code, 0, crate::disasm::decode_instruction);
        let expected = [
            (0, 4),
            (2, 2),
            (3, 4),
            (5, 2),
            (6, 2),
            (7, 4),
            (9, 6),
            (12, 2),
        ];
        assert_eq!(lines.len(), expected.len());
        for (line, (addr, len)) in lines.iter().zip(expected) {
            assert_eq!((line.addr, line.raw.len()), (addr, len));
            assert_eq!(instruction_len(&code, addr as usize * 2), len);
        }
        assert_eq!(lines[3].text, "r4=dm (i0,m0)");
        assert_eq!(lines[4].text, "nop");

        // R4/F4 aliases encode as
        // four separate 16-bit loads/stores, not two 32-bit instructions.
        let code = [0x90, 0x14, 0x90, 0x34, 0x90, 0x14, 0x90, 0x34];
        let lines = disassemble_visa(&code, 0, crate::disasm::decode_instruction);
        assert_eq!(lines.len(), 4);
        for (i, line) in lines.iter().enumerate() {
            assert_eq!((line.addr, line.raw.len()), (i as u32, 2));
            assert_eq!(
                line.text,
                if i % 2 == 0 {
                    "r4=dm (i0,m0)"
                } else {
                    "dm (i0,m0)=r4"
                }
            );
        }
    }

    #[test]
    fn short_data_move_mask_keeps_all_modify_bits_and_other_widths() {
        // Vendor rows: dm(i0, M0..M7) = R4/F4. Bit 8 is M[2].
        let stores = [
            0x9034u16, 0x9074, 0x90b4, 0x90f4, 0x9134, 0x9174, 0x91b4, 0x91f4,
        ];
        for (m, word) in stores.into_iter().enumerate() {
            let mut code = word.to_be_bytes().to_vec();
            code.extend_from_slice(&[0, 1]);
            assert_eq!(instruction_len(&code, 0), 2);
            let lines = disassemble_visa(&code, 0, crate::disasm::decode_instruction);
            assert_eq!(lines.len(), 2);
            assert_eq!(lines[0].text, format!("dm (i0,m{m})=r4"));
            assert_eq!(lines[1].text, "nop");
        }

        // Neighboring 0x9xxx families remain wide: vendor
        // pm(i10,m8)=0 is 32-bit; dm(i2,m0)=0x1c0111
        // is 48-bit. Neither has Type 3c's fixed bit 4.
        for code in [
            vec![0x94, 0x24, 0x00, 0x00],
            vec![0x94, 0x00, 0x00, 0x1c, 0x01, 0x11],
        ] {
            let len = code.len();
            let mut stream = code;
            stream.extend_from_slice(&[0, 1]);
            assert_eq!(instruction_len(&stream, 0), len);
            let lines = disassemble_visa(&stream, 0, crate::disasm::decode_instruction);
            assert_eq!(lines.len(), 2);
            assert_eq!(lines[0].raw.len(), len);
            assert_eq!(lines[1].text, "nop");
        }
    }
}
