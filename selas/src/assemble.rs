// SPDX-License-Identifier: GPL-3.0
// assemble.rs --- Assembler orchestrator: parse, encode, emit .doj
// Copyright (c) 2026 Jakob Kastelic

//! Multi-section assembler orchestrator: reads .s files, parses, encodes, and
//! emits .doj ELF objects with per-section code and relocation support.

use std::collections::HashMap;

use crate::asmparse::{AsmParser, DataInit, Directive, ParsedLine};
use crate::error::{Error, Result};

/// Per-section accumulated code and symbol data.
struct SectionData {
    code: Vec<u8>,
    symbols: Vec<(String, u32)>,
    /// Declared /DM object sizes, in native words like their symbol values.
    dm_symbol_sizes: HashMap<String, u32>,
    is_pm: bool,
    /// Whether this section uses VISA encoding.
    is_visa: bool,
    /// Parcel count for VISA mode (each parcel = 1 address unit).
    /// In ISA mode this is not used; word_offset() uses byte length.
    parcel_count: u32,
    /// Byte-addressed data section (`DOUBLE32` / `BW` qualifier).
    byte_addressed: bool,
    /// Plain `/DM`: five file bytes per word-addressed 32-bit value.
    word_addressed: bool,
    /// Byte-addressed data starts at alignment one; .ALIGN raises it.
    data_alignment: u32,
    /// Instructions emitted since the last conditional JUMP, while fewer
    /// than `COND_JUMP_TAIL` (see `pad_cond_jump_tail`).
    cond_tail: Option<u32>,
    /// VISA parcel offset just past the last JUMP/CALL (`branch_shadow`).
    jump_end: Option<u32>,
    /// Delay slots still open after a delayed branch (`delay_slot_illegal`).
    delay_slots: u32,
}

/// The reference assembler ends a section, and a function at a `<name>.end` label, with at
/// least this many instructions after its last conditional JUMP; it pads
/// with NOPs when there are fewer.
const COND_JUMP_TAIL: u32 = 6;

/// Conditional control transfers that start the reference assembler's six-instruction tail
/// (MEASURED): JUMP and
/// CALL in every form (direct, PC-relative, indirect, with a compute or an
/// ELSE compute, delayed, `(LA)`) and RTS/RTI (`(LR)`, with a compute), on
/// any condition but TRUE (31), including `NOT LCE`. Explicit delay-slot
/// instructions count toward the six (two slots + four
/// pads). Unconditional transfers and CJUMP never pad.
fn is_cond_transfer(instr: &selinstr::encode::Instruction) -> bool {
    use selinstr::encode::Instruction as I;
    match instr {
        I::Branch { cond, .. }
        | I::Return { cond, .. }
        | I::ComputeBranch { cond, .. }
        | I::IndirectBranch { cond, .. } => *cond != 31,
        _ => false,
    }
}

/// The reference assembler keeps a branch wide when it ends fewer than three parcels after
/// the end of a preceding JUMP/CALL
/// (MEASURED):
/// with g parcels between them, g = 0 makes `rts` and `jump (m14,i12) (db)`
/// 48-bit, g = 1 leaves the 32-bit indirect JUMP
/// but makes `rts`/`rti`/`if ne rts`/`rts (db)` 48-bit,
/// g >= 2 compresses. The JUMP/CALL
/// may be direct, PC-relative or indirect, conditional or delayed; a 16-bit
/// RTS before a branch does not count. Returns the minimum
/// parcel count for `instr` at parcel `here`, if the rule applies.
fn branch_shadow(
    instr: &selinstr::encode::Instruction,
    here: u32,
    jump_end: Option<u32>,
) -> Option<u32> {
    use selinstr::encode::Instruction as I;
    let shadowed = matches!(instr, I::Return { .. } | I::IndirectBranch { .. });
    let gap = here.checked_sub(jump_end?)?;
    (shadowed && gap < 2).then(|| 3 - gap)
}

/// A delayed (DB) control transfer: the next two instructions are its
/// delay slots.
fn is_delayed_transfer(instr: &selinstr::encode::Instruction) -> bool {
    use selinstr::encode::Instruction as I;
    match instr {
        I::Branch { delayed, .. }
        | I::Return { delayed, .. }
        | I::ComputeBranch { delayed, .. }
        | I::IndirectBranch { delayed, .. }
        | I::CJump { delayed, .. } => *delayed,
        _ => false,
    }
}

/// The reference assembler rejects (`jump _x (db); rts;`): "Neither of the
/// delay branch slots can hold jump, call, return, do, or pc stack
/// manipulation instructions."
fn delay_slot_illegal(instr: &selinstr::encode::Instruction) -> bool {
    use selinstr::encode::Instruction as I;
    match instr {
        I::Branch { .. }
        | I::Return { .. }
        | I::ComputeBranch { .. }
        | I::IndirectBranch { .. }
        | I::CJump { .. }
        | I::DoLoop { .. }
        | I::DoUntil { .. } => true,
        // bit4 = PUSH PCSTK, bit5 = POP PCSTK
        I::StackOp { ops } => ops & 0x30 != 0,
        _ => false,
    }
}

/// The JUMP/CALL forms that start `branch_shadow`.
fn starts_branch_shadow(instr: &selinstr::encode::Instruction) -> bool {
    use selinstr::encode::Instruction as I;
    matches!(instr, I::Branch { .. } | I::IndirectBranch { .. })
}

impl SectionData {
    fn new(is_pm: bool, is_visa: bool) -> Self {
        Self {
            code: Vec::new(),
            symbols: Vec::new(),
            dm_symbol_sizes: HashMap::new(),
            is_pm,
            is_visa: is_visa && is_pm,
            parcel_count: 0,
            byte_addressed: false,
            word_addressed: false,
            data_alignment: 1,
            cond_tail: None,
            jump_end: None,
            delay_slots: 0,
        }
    }

    /// Track the distance from the last conditional control transfer
    /// (`is_cond_transfer`).
    fn note_instruction(&mut self, instr: &selinstr::encode::Instruction) {
        self.delay_slots = if is_delayed_transfer(instr) {
            2
        } else {
            self.delay_slots.saturating_sub(1)
        };
        self.cond_tail = if is_cond_transfer(instr) {
            Some(0)
        } else {
            self.cond_tail
                .map(|n| n + 1)
                .filter(|&n| n < COND_JUMP_TAIL)
        };
    }

    /// Append one NOP: a 16-bit parcel (`0001`) in a VISA section, the
    /// 48-bit word otherwise. Counts as an instruction.
    fn emit_nop(&mut self, wide: bool) -> Result<()> {
        let nop = selinstr::encode::Instruction::Nop;
        if self.is_visa {
            let encoded = if wide {
                selinstr::visa_encode::VisaEncoded::W48(selinstr::encode::encode(&nop)?)
            } else {
                selinstr::visa_encode::encode_visa(&nop)?
            };
            self.parcel_count += encoded.parcels();
            self.code.extend_from_slice(&encoded.to_bytes());
        } else {
            self.code
                .extend_from_slice(&selinstr::encode::encode(&nop)?);
        }
        self.note_instruction(&nop);
        Ok(())
    }

    /// The reference assembler: pad with NOPs until `COND_JUMP_TAIL` instructions follow
    /// the last conditional transfer (`if ne jump` last = 6 NOPs;
    /// three instructions after it = 3 NOPs; jumps, calls and returns
    /// without a condition never pad). The pads are 16-bit in a VISA
    /// section, also under `.NOCOMPRESS`, except the
    /// last `wide_last`, which fall in a DO loop's 48-bit tail
    /// (`LoopPlan::pad_room`).
    fn pad_cond_jump_tail(&mut self, wide_last: usize) -> Result<()> {
        if let Some(n) = self.cond_tail {
            let count = (COND_JUMP_TAIL - n) as usize;
            for k in 0..count {
                self.emit_nop(k + wide_last >= count)?;
            }
        }
        Ok(())
    }

    /// Current word offset.  In ISA mode: PM instructions are 6 bytes;
    /// DM data words occupy 5 file bytes. In VISA mode: parcel_count tracks
    /// address units directly.
    fn word_offset(&self) -> u32 {
        let unit = if self.is_pm {
            6
        } else if self.word_addressed {
            5
        } else {
            4
        };
        (self.code.len() / unit) as u32
    }

    /// Current address in VISA mode (parcel-based).
    fn visa_offset(&self) -> u32 {
        self.parcel_count
    }
}

/// Parse a `.SECTION` name string produced by the parser.
///
/// SHARC+ section directives can stack any number of slash-separated
/// qualifiers between the keyword and the bare section name, e.g.
///
/// ```text
///     .SECTION/PM seg_pmco;
///     .SECTION/SW seg_swco;
///     .SECTION/DOUBLE32 seg_dmda;
///     .SECTION/CODE/DOUBLEANY seg_pmco;
///     .SECTION/CODE/NW/DOUBLEANY iv_code;
/// ```
///
/// The parser hands us the substring after the keyword (still leading
/// `/` and trailing `;`). Strip the qualifier chain and return the bare
/// name plus whether any qualifier in the chain implies program memory
/// (PM / SW / CODE). The LDF matches sections by bare name, so emitting
/// `"CODE/NW/DOUBLEANY iv_code"` as the section name causes
/// `INPUT_SECTIONS( $OBJECTS(iv_code) )` to never match anything; the
/// output binary then ships without an IVT, a startup vector, or any
/// of the runtime objects.
///
/// `DOUBLE32` is a byte-addressed DM variant with 32-bit-aligned words
/// that a cross-section `R1 = symbol.;` reference from a code section
/// can resolve against; the word-addressed `DM` qualifier rejects such
/// relocations at link time.
fn parse_section_name(raw: &str) -> (String, bool, bool, bool) {
    let s = raw.trim().trim_end_matches(';').trim();
    // The parser preserves the leading `/`; strip it so the rest can
    // be split as a qualifier list.
    let s = s.strip_prefix('/').unwrap_or(s);
    // Split off the bare name: the section name is the final
    // whitespace-separated token. Everything before it is a chain of
    // slash-separated qualifiers.
    let (qualifier_str, name) = match s.rsplit_once(char::is_whitespace) {
        Some((q, n)) => (q, n.trim()),
        None => ("", s),
    };
    if name.is_empty() {
        return (s.to_string(), false, false, false);
    }
    let mut is_pm = false;
    // `NW` (Normal Word, 48-bit) on a code section forbids VISA
    // compression: the SHARC+ boot ROM steps the IVT in 48-bit slots,
    // and a compressed 16/32-bit instruction inside iv_code lands the
    // reset vector in the middle of the next slot. The .SECTION
    // directive lets the assembler know via a `/NW` qualifier; honour
    // it by reporting `force_nw` to the caller so the per-section
    // visa flag stays off even when the file is being assembled in
    // VISA mode globally.
    let mut force_nw = false;
    let mut force_sw = false;
    for tok in qualifier_str.split('/') {
        let upper = tok.trim().to_uppercase();
        if matches!(upper.as_str(), "PM" | "SW" | "NW" | "CODE") {
            is_pm = true;
        }
        if upper == "NW" {
            force_nw = true;
        }
        if upper == "SW" {
            force_sw = true;
        }
    }
    (name.to_string(), is_pm, force_nw, force_sw)
}

/// Parse a numeric literal: hex (0x...), octal (0...), or decimal.
fn parse_u32_literal(s: &str) -> Option<u32> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else if s.starts_with('0') && s.len() > 1 && s.chars().all(|c| c.is_ascii_digit()) {
        u32::from_str_radix(s, 8).ok()
    } else {
        s.parse::<u32>().ok()
    }
}

/// Preprocess source text before parsing.
///
/// Strips `.ENDSEG` directives (which the parser does not handle) by replacing
/// them with blank lines so that line numbering is preserved.
fn preprocess(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let stripped = line.trim().trim_end_matches(';').trim();
        if stripped.eq_ignore_ascii_case(".ENDSEG") {
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Whether a `.SECTION` qualifier chain names a byte-addressed data
/// section: `DOUBLE32` or `BW`.
fn is_byte_addressed(raw: &str) -> bool {
    let s = raw.trim().trim_end_matches(';').trim();
    let s = s.strip_prefix('/').unwrap_or(s);
    let qualifiers = s.rsplit_once(char::is_whitespace).map_or("", |(q, _)| q);
    qualifiers
        .split('/')
        .any(|t| matches!(t.trim().to_uppercase().as_str(), "DOUBLE32" | "BW"))
}

/// Ensure the named section exists in the section list, and return its index.
fn ensure_section(
    sections: &mut Vec<(String, SectionData)>,
    name: &str,
    is_pm: bool,
    visa: bool,
) -> usize {
    if let Some(idx) = sections.iter().position(|(n, _)| n == name) {
        return idx;
    }
    sections.push((name.to_string(), SectionData::new(is_pm, visa)));
    sections.len() - 1
}

/// Return the current section index, creating a default `.text` PM section if
/// none has been set yet.
fn current_or_default(
    sections: &mut Vec<(String, SectionData)>,
    current: &mut Option<usize>,
    default_name: &str,
    default_pm: bool,
    visa: bool,
) -> usize {
    if let Some(idx) = *current {
        return idx;
    }
    let idx = ensure_section(sections, default_name, default_pm, visa);
    *current = Some(idx);
    idx
}

/// An instruction that may need label resolution in a second pass.
struct PendingInstr {
    section_idx: usize,
    byte_offset: usize,
    byte_len: usize,
    /// Word/parcel offset of this instruction within its section, captured
    /// at encode time. In VISA mode this cannot be recovered from
    /// `byte_offset` alone because earlier instructions may be 2, 4, or 6
    /// bytes. Used by pass 2 to compute correct PC-relative offsets.
    word_offset: u32,
    instr: selinstr::encode::Instruction,
    label_ref: String,
    /// Additional label references for multi-symbol expressions (e.g.
    /// `I7 = ldf_stack_space + ldf_stack_length;`).  Each pair is
    /// `(symbol_name, signed_contribution)` and produces an extra Add
    /// relocation against the same operand slot in pass 2.
    extra_label_refs: Vec<(String, i64)>,
    /// A source `(PC, label)` branch: the reference assembler leaves 0 in the field and a
    /// PC-relative relocation even for a local label (`jump (pc,.P33L1)`
    /// = 073e00000000). Numeric targets keep
    /// n in the field and relocate against a generated label, in SW and NW.
    link_time: bool,
}

/// A relocation to emit against an undefined label. Produced in pass 2
/// when a pending label reference cannot be resolved in-file.
struct PendingReloc {
    section_idx: usize,
    byte_offset: u32,
    symbol: String,
    rela_type: u8,
    addend: i32,
}

/// Pick the ELF relocation type that matches the byte layout of a given
/// instruction. SHARC+ PM code references use a small family of
/// relocation types, one per instruction field layout:
///   * `R_SHARC_PM32` (0x0c) patches bytes 2..5 of a 48-bit instruction
///     with a 32-bit absolute value. Used by Type 17 `Ureg = imm32`
///     (`LoadImm`).
///   * `R_SHARC_PM24` (0x0b) patches bytes 3..5 of a 48-bit instruction
///     with a 24-bit absolute value. Used by Type 8a direct
///     `JUMP`/`CALL` and Type 25a `CJUMP`.
///   * `R_SHARC_PM_PCREL24` (0x0f) patches the same bytes 3..5, but
///     seld computes `target - pc` first. Used by Type 8b PC-relative
///     direct branches and by Type 12/13 `DO ... UNTIL` loops whose
///     RELADDR field is itself PC-relative to the DO instruction.
///
/// `R_SHARC_ADDR32` (0x01) writes a 32-bit value at the raw byte
/// offset and will clobber the opcode bytes of a 48-bit instruction,
/// so it must never be used for code-section relocations. It was the
/// type this code emitted originally, and it is the reason external
/// CALL/LoadImm targets silently produced broken bytes in the linked
/// image even when the link itself succeeded.
fn reloc_type_for(instr: &selinstr::encode::Instruction) -> u8 {
    use selinstr::encode::{BranchTarget, Instruction};
    match instr {
        Instruction::LoadImm { .. } => selelf::elf::R_SHARC_PM32 as u8,
        Instruction::Branch {
            target: BranchTarget::Absolute(_),
            ..
        } => selelf::elf::R_SHARC_PM24 as u8,
        Instruction::Branch {
            target: BranchTarget::PcRelative(_),
            ..
        } => selelf::elf::R_SHARC_PM_PCREL24 as u8,
        Instruction::CJump { relative: true, .. } => selelf::elf::R_SHARC_PM_PCREL24 as u8,
        Instruction::CJump {
            relative: false, ..
        } => selelf::elf::R_SHARC_PM24 as u8,
        Instruction::ComputeBranch { .. } => selelf::elf::R_SHARC_PM_PCREL6 as u8,
        // The reference assembler: every DO target, local or not
        // (`do _mod_copy_end until lce` = 0c0082800000 + 0x1b).
        Instruction::DoLoop { .. } | Instruction::DoUntil { .. } => {
            selelf::elf::R_SHARC_PM_LOOP16 as u8
        }
        Instruction::UregAbsAccess { .. } => selelf::elf::R_SHARC_PM32 as u8,
        Instruction::ImmStore { .. } => selelf::elf::R_SHARC_PM32 as u8,
        // Any other instruction landing in the relocation path is a
        // bug in either the parser or the backend: every case we
        // actually produce is handled above. Panicking here makes such
        // a bug loud instead of silently emitting the wrong reloc
        // type.
        other => panic!("relocation emitted for unsupported instruction shape: {other:?}"),
    }
}

/// Resolve a symbolic label reference inside an instruction.
///
/// Returns `Some(new_instr)` when the reference can be rewritten into a
/// position-independent form: an internal Type 12/13 DO loop whose RELADDR
/// field is `label_addr - instr_word_offset` (SHARC ISR, Program Flow
/// Control, Type 12/13: RELADDR is "the end-of-loop address relative
/// to the DO LOOP instruction address"). The rewritten bytes need no
/// further linker fixup and the caller patches the encoded instruction
/// in place.
///
/// Returns `None` when the reference must instead become a link-time
/// relocation:
///   - `LoadImm`: the immediate is a runtime address that the linker
///     must patch, regardless of whether the symbol is locally defined.
///   - Branches, including `CJump`: the source's absolute/relative mode
///     determines the relocation, even when its target is defined here.
///   - Bare symbolic direct branches: the reference assembler keeps these absolute with
///     PM24 relocations, including local targets in L2 sections
///     (063e00000000, type 0x0b).
///   - Any cross-section reference: a PC-relative offset between two
///     sections is not a fixed constant because the linker is free to
///     place the sections arbitrarily far apart.
///   - Any label that is not defined in this file at all.
///
/// The caller turns `None` into an R_SHARC_ADDR24 relocation against a
/// local symbol (if the label is defined somewhere in this file) or an
/// external symbol (if it is not).
///
/// `instr_section_idx` and `instr_word_offset` locate the instruction
/// being resolved in the section list.
fn resolve_labels(
    instr: &selinstr::encode::Instruction,
    label_name: &str,
    label_map: &std::collections::HashMap<String, (usize, u32)>,
    instr_section_idx: usize,
    instr_word_offset: u32,
) -> Option<selinstr::encode::Instruction> {
    use selinstr::encode::Instruction;
    // A symbolic load always goes through a relocation; a non-zero value
    // is its addend (`r0 = label + 16`).
    if matches!(*instr, Instruction::LoadImm { .. }) {
        return None;
    }
    if matches!(
        *instr,
        Instruction::Branch { .. } | Instruction::CJump { .. } | Instruction::ComputeBranch { .. }
    ) {
        return None;
    }
    // ImmStore with a symbolic value (e.g. a return-address label pushed
    // onto the stack in a CJUMP delay slot) needs a linker relocation for
    // the absolute address, just like LoadImm and CJump.
    if matches!(*instr, Instruction::ImmStore { .. }) {
        return None;
    }
    let (label_section, label_word_offset) = *label_map.get(label_name)?;
    if label_section != instr_section_idx {
        return None;
    }
    let offset = (label_word_offset as i32) - (instr_word_offset as i32);
    let resolved = match *instr {
        Instruction::DoLoop {
            counter,
            end_pc: 0,
            f,
        } => Instruction::DoLoop {
            counter,
            end_pc: offset as u32,
            f,
        },
        Instruction::DoUntil { addr: 0, term, f } => Instruction::DoUntil {
            addr: offset as u32,
            term,
            f,
        },
        other => other,
    };
    Some(resolved)
}

/// Assemble an input .s file to an output .doj ELF object.
///
/// When `visa` is true, PM code sections use VISA variable-width encoding
/// (16/32/48-bit instructions) targeting ADSP-21569.
pub fn assemble_file(input: &str, output: &str) -> Result<()> {
    let raw_src = std::fs::read_to_string(input)?;
    let bytes = assemble_source(&raw_src, false)?;
    std::fs::write(output, bytes)?;
    Ok(())
}

/// Assemble with VISA encoding for PM code sections.
pub fn assemble_file_visa(input: &str, output: &str) -> Result<()> {
    let raw_src = std::fs::read_to_string(input)?;
    let bytes = assemble_source(&raw_src, true)?;
    std::fs::write(output, bytes)?;
    Ok(())
}

/// Assemble already-preprocessed source text and return the raw .doj ELF
/// bytes. Callers that already have the source in memory can use this to
/// avoid a round-trip through the filesystem.
pub fn assemble_source(src: &str, visa: bool) -> Result<Vec<u8>> {
    assemble_source_with(src, visa, AsmOptions::default())
}

/// Assembler behaviour switches; the defaults follow the reference assembler.
#[derive(Debug, Clone, Copy)]
pub struct AsmOptions<'a> {
    /// The reference assembler `-nwc`: code defaults to 48-bit words, including labels,
    /// alignment and relocation offsets. Explicit `/SW` overrides it.
    pub nwc: bool,
    /// The reference assembler `-swc-exclude`: default these code sections to NW.
    /// An explicit /SW qualifier still overrides the exclusion.
    pub swc_exclude: &'a [String],
    /// The reference assembler `-auto-fast-loops` (default) or `-no-auto-fast-loops`;
    /// see `plan_loops`.
    pub auto_fast_loops: bool,
    /// Read a numeric `(PC, n)` target as raw address units instead of
    /// "n instructions away" (`label_pc_offsets`). Only for re-assembling
    /// selas's own disassembly, which prints raw offsets.
    pub raw_pc_offsets: bool,
    /// Pad section and function ends after a conditional JUMP like
    /// The reference assembler (`SectionData::pad_cond_jump_tail`). Off only for
    /// re-assembling single disassembled instructions.
    pub cond_jump_tail_pad: bool,
}

impl Default for AsmOptions<'_> {
    fn default() -> Self {
        Self {
            nwc: false,
            swc_exclude: &[],
            auto_fast_loops: true,
            raw_pc_offsets: false,
            cond_jump_tail_pad: true,
        }
    }
}

/// `assemble_source` with explicit `AsmOptions`.
pub fn assemble_source_with(src: &str, visa: bool, options: AsmOptions) -> Result<Vec<u8>> {
    assemble_source_inner(src, visa, options)
}

/// The reference assembler reads a numeric `(PC, n)` jump, call or DO target in a
/// compressed (VISA) section as "n instructions away" and puts a label
/// there: its objects carry `___ADI_AGL_JUMP_4` / `___ADI_AGL_DO_4`, and it
/// reports "Constant offset in JUMP is not recommended ... The
/// assembler has done this for you". Example:
/// `if ge jump (pc, 0x4)` targets parcel 6, the fourth instruction after
/// the 48-bit jump; `do (pc, 0x12)` ends on the 18th body
/// instruction. Used raw, the same numbers count 16-bit parcels and land
/// elsewhere once the instructions in between are compressed. Rewrite
/// such targets as labels for the linker, keeping n in the unlinked field
/// in both SW and NW.
fn label_pc_offsets(lines: &mut [ParsedLine]) -> Result<()> {
    use selinstr::encode::{BranchTarget, Instruction as I};
    // Instruction lines of each code section, in source order. A
    // reopened section keeps the kind it was first opened with.
    let mut kind: HashMap<String, bool> = HashMap::new();
    let mut order: HashMap<String, Vec<usize>> = HashMap::new();
    let mut place: Vec<Option<(String, usize)>> = vec![None; lines.len()];
    let mut current = (".text".to_string(), true);
    for (i, line) in lines.iter().enumerate() {
        if let Some(Directive::Section(raw)) = &line.directive {
            let (name, is_pm, _, _) = parse_section_name(raw);
            let section_kind = *kind.entry(name.clone()).or_insert(is_pm);
            current = (name, section_kind);
        }
        if line.instruction.is_some() && current.1 {
            let list = order.entry(current.0.clone()).or_default();
            place[i] = Some((current.0.clone(), list.len()));
            list.push(i);
        }
    }
    for i in 0..lines.len() {
        let Some((section, at)) = place[i].clone() else {
            continue;
        };
        if lines[i].label_ref.is_some() {
            continue;
        }
        // A self-relative direct branch needs no layout fixup. Vendor
        // object: 073e00000000, no relocation.
        if matches!(
            lines[i].instruction,
            Some(I::Branch {
                target: BranchTarget::PcRelative(0),
                ..
            })
        ) {
            continue;
        }
        // A DO keeps n in its field and relocates against a label the reference assembler
        // names after the DO's line (`do (pc, 0x1e)` = 0c001080001e +
        // R_SHARC_PM_LOOP16 against `___ADI_AGL_DO_5`).
        let (n, vendor_label) = match lines[i].instruction {
            // A CALL's label says CALL (`call (pc,1)` on
            // line 4 relocates against `___ADI_AGL_CALL_4`).
            Some(I::Branch {
                call,
                target: BranchTarget::PcRelative(n),
                ..
            }) => (
                n as i64,
                format!(
                    "___ADI_AGL_{}_{}",
                    if call { "CALL" } else { "JUMP" },
                    lines[i].line
                ),
            ),
            Some(I::CJump {
                addr,
                relative: true,
                ..
            }) => (
                addr as i32 as i64,
                format!("___ADI_AGL_CJUMP_{}", lines[i].line),
            ),
            Some(I::ComputeBranch { offset, .. }) => {
                (offset as i64, format!("___ADI_AGL_JUMP_{}", lines[i].line))
            }
            Some(I::DoLoop { end_pc, .. }) => (
                end_pc as i32 as i64,
                format!("___ADI_AGL_DO_{}", lines[i].line),
            ),
            Some(I::DoUntil { addr, .. }) => (
                addr as i32 as i64,
                format!("___ADI_AGL_DO_{}", lines[i].line),
            ),
            _ => continue,
        };
        let list = &order[&section];
        let target = at as i64 + n;
        if target < 0 || target >= list.len() as i64 {
            return Err(Error::Parse {
                line: 0,
                msg: format!(
                    "`(PC, {n})` target lies outside section `{section}` \
                     (counted in instructions, as the reference assembler does)"
                ),
            });
        }
        // The reference assembler generates the label even when the target already has one
        // (`___ADI_AGL_JUMP_3` and `___ADI_AGL_CALL_4`
        // beside `_target`), and the relocation names the generated one.
        let target_line = list[target as usize];
        lines[target_line]
            .generated_labels
            .push(vendor_label.clone());
        lines[i].label_ref = Some(vendor_label);
    }
    Ok(())
}

/// Number of trailing DO-loop body instructions the reference assembler keeps in 48-bit
/// form. In all 229 counter loops of the compressed-build listings
/// measured, exactly the last 11 body instructions (the end-label instruction
/// and the ten before it, nested DO instructions included) are 48-bit
/// and everything earlier is compressed as usual, e.g. a
/// 94-instruction loop compresses all but its final 11.
const LOOP_TAIL: usize = 11;

/// Per-line loop decisions made before encoding.
struct LoopPlan {
    /// Line indices whose instruction must be emitted in 48-bit form.
    force48: std::collections::HashSet<usize>,
    /// Line indices of `LCNTR = ..., DO end UNTIL LCE` instructions whose
    /// fast-loop bit (bit 23) the reference assembler sets without an explicit `(F)`.
    auto_fast: std::collections::HashSet<usize>,
    /// Line index of a `<name>.end` label -> how many of the tail pads
    /// emitted there (the last ones) fall in a DO loop's 48-bit tail.
    pad_room: std::collections::HashMap<usize, usize>,
    /// Instruction lines that are among the last ten of a DO tail-rule
    /// loop: an `.ALIGN` just before one is a reference assembler error in a
    /// VISA section.
    align_last_ten: std::collections::HashSet<usize>,
    /// Instruction lines of a whole-body 48-bit loop (`DO UNTIL` other
    /// than LCE): no vendor vector for `.ALIGN` there, so it stays refused.
    align_refused: std::collections::HashSet<usize>,
}

/// Tail instructions the reference assembler's auto-fast-loops pass accepts: straight-line
/// computes, data moves, immediate loads, MODIFY and nested counter loops.
/// It clears the bit for control flow in a short loop body. The longer
/// 20-, 30- and 40-instruction vectors permit a branch before the last 11
/// instructions; the 12-instruction one puts the same branch inside that tail and clears it.
/// SHARC+ Core Programming Reference rev. 1.5, p. 4-37, documents this
/// eleven-instruction window for flow changes and IDLE in F1-active loops.
/// Everything not listed here in the tail is treated conservatively.
fn fast_loop_body_instr(instr: &selinstr::encode::Instruction) -> bool {
    use selinstr::encode::Instruction as I;
    matches!(
        instr,
        I::Nop
            | I::LoadImm { .. }
            | I::Compute { .. }
            | I::ComputeLoadStore { .. }
            | I::DoLoop { .. }
            | I::DualMove { .. }
            | I::Modify { .. }
            | I::UregDagMove { .. }
            | I::DagModify { .. }
            | I::DagConvert { .. }
            | I::RegisterSwap { .. }
            | I::UregTransfer { .. }
            | I::URegMove { .. }
            | I::UregMemAccess { .. }
            | I::ImmShift { .. }
            | I::ImmShiftMem { .. }
            | I::ImmStore { .. }
            | I::UregAbsAccess { .. }
    )
}

/// Find each DO loop whose end label lies ahead in the same section and
/// decide which body instructions stay 48-bit and whether the reference assembler would
/// set the fast-loop bit by itself:
/// - `LCNTR = ..., DO end UNTIL LCE`: the last `LOOP_TAIL` body
///   instructions are 48-bit. With `auto_fast_loops`, the fast bit is
///   set when every instruction in that tail passes `fast_loop_body_instr`
///   (the reference assembler leaves it clear for an external or backward end label).
/// - `DO end UNTIL LCE`: the same 48-bit tail and automatic fast bit
/// - `DO end UNTIL cond` for any other condition: no compressed-build
///   vector, so the whole body stays 48-bit as before.
///
/// The 48-bit tail is counted over the emitted instruction stream, which
/// includes the conditional-transfer pads the reference assembler inserts before a
/// `<name>.end` label inside the body (after `if ne jump`,
/// six pads, `_f.end` and six more instructions, the first two pads are
/// 16-bit and everything after them is 48-bit; the source instructions
/// before the jump stay 16-bit). The automatic fast bit is decided over
/// the source instructions only (the jump is 11th from the
/// end in source, 17th in the emitted stream, and the bit stays clear;
/// without the pads it sets it). `.ALIGN` fill is not counted:
/// in a VISA section it can only precede the last eleven instructions.
fn plan_loops(lines: &[ParsedLine], auto_fast_loops: bool, cond_jump_tail_pad: bool) -> LoopPlan {
    use selinstr::encode::Instruction as I;
    let exported: std::collections::HashSet<&str> = lines
        .iter()
        .filter_map(|line| match &line.directive {
            Some(Directive::Global(names) | Directive::Weak(names)) => Some(names),
            _ => None,
        })
        .flatten()
        .map(String::as_str)
        .collect();
    let mut section_of = Vec::with_capacity(lines.len());
    let mut current: Option<String> = None;
    for line in lines {
        if let Some(Directive::Section(raw)) = &line.directive {
            current = Some(parse_section_name(raw).0);
        }
        section_of.push(current.clone());
    }
    // Pads emitted at each `<name>.end` label, replaying
    // `SectionData::note_instruction` / `pad_cond_jump_tail` per section.
    let mut pads_at = vec![0usize; lines.len()];
    if cond_jump_tail_pad {
        let mut tail: std::collections::HashMap<Option<String>, Option<u32>> =
            std::collections::HashMap::new();
        for (j, line) in lines.iter().enumerate() {
            let t = tail.entry(section_of[j].clone()).or_insert(None);
            let data_line = matches!(line.directive, Some(Directive::Data(_)));
            if !data_line && line.label.as_deref().is_some_and(|l| l.ends_with(".end")) {
                if let Some(n) = t.take() {
                    pads_at[j] = (COND_JUMP_TAIL - n) as usize;
                }
            }
            if let Some(instr) = &line.instruction {
                *t = if is_cond_transfer(instr) {
                    Some(0)
                } else {
                    t.map(|n| n + 1).filter(|&n| n < COND_JUMP_TAIL)
                };
            }
        }
    }
    let mut plan = LoopPlan {
        force48: std::collections::HashSet::new(),
        auto_fast: std::collections::HashSet::new(),
        pad_room: std::collections::HashMap::new(),
        align_last_ten: std::collections::HashSet::new(),
        align_refused: std::collections::HashSet::new(),
    };
    for (i, line) in lines.iter().enumerate() {
        // `DO end UNTIL LCE` (Type 13) follows the counter-loop tail rule:
        // The reference assembler places `do (pc, 0x1000) until lce`
        // at 0x1000 + 22 parcels, i.e. 11 trailing 48-bit instructions.
        const LCE: u8 = 15;
        let tail_only = match &line.instruction {
            Some(I::DoLoop { .. }) => true,
            Some(I::DoUntil { term, .. }) => *term == LCE,
            _ => continue,
        };
        let Some(end) = &line.label_ref else {
            continue;
        };
        // The body runs up to and including the first instruction at or
        // after the end label.
        let mut body = Vec::new();
        // Emitted stream: (line, is_pad); a label's pads precede its line's
        // instruction.
        let mut stream: Vec<(usize, bool)> = Vec::new();
        let mut label_seen = false;
        let mut closed = false;
        for (j, l) in lines.iter().enumerate().skip(i + 1) {
            if section_of[j] != section_of[i] {
                break;
            }
            if l.label.as_deref() == Some(end.as_str()) || l.generated_labels.contains(end) {
                label_seen = true;
            }
            stream.extend(std::iter::repeat((j, true)).take(pads_at[j]));
            if l.instruction.is_some() {
                body.push(j);
                stream.push((j, false));
                if label_seen {
                    closed = true;
                    break;
                }
            }
        }
        if !closed {
            continue;
        }
        let mut room: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        if tail_only {
            for (pos, &(j, pad)) in stream.iter().rev().take(LOOP_TAIL).enumerate() {
                if pad {
                    *room.entry(j).or_default() += 1;
                } else {
                    plan.force48.insert(j);
                    if pos < LOOP_TAIL - 1 {
                        plan.align_last_ten.insert(j);
                    }
                }
            }
        } else {
            plan.force48.extend(body.iter().copied());
            plan.align_refused.extend(body.iter().copied());
            for &(j, pad) in &stream {
                if pad {
                    *room.entry(j).or_default() += 1;
                }
            }
        }
        for (j, n) in room {
            let r = plan.pad_room.entry(j).or_default();
            *r = (*r).max(n);
        }
        let fast = body.iter().rev().take(LOOP_TAIL).all(|&j| {
            lines[j]
                .instruction
                .as_ref()
                .is_some_and(fast_loop_body_instr)
        });
        // A global end symbol is resolved by the linker. The reference assembler leaves
        // automatic fast mode clear,
        // even when its local definition has a straight-line body. Explicit
        // (F) stays in the parsed instruction and is unaffected here.
        if tail_only && auto_fast_loops && fast && !exported.contains(end.as_str()) {
            plan.auto_fast.insert(i);
        }
    }
    plan
}

/// Map every label defined in any section to its `(section_idx, word_offset)`.
/// Used by the label-resolution pass to decide whether a reference can be
/// rewritten position-independently (same section) or must become a
/// relocation (cross-section, external, or ISA-mandated absolute target).
fn build_label_map(
    sections: &[(String, SectionData)],
) -> std::collections::HashMap<String, (usize, u32)> {
    let mut map = std::collections::HashMap::new();
    for (sec_idx, (_, sec)) in sections.iter().enumerate() {
        for (name, offset) in &sec.symbols {
            map.insert(name.clone(), (sec_idx, *offset));
        }
    }
    map
}

fn assemble_source_inner(raw_src: &str, visa: bool, options: AsmOptions) -> Result<Vec<u8>> {
    let visa = visa && !options.nwc;
    let src = preprocess(raw_src);
    let mut parser = AsmParser::new(&src);
    let mut lines = parser.parse_all()?;
    if !options.raw_pc_offsets {
        label_pc_offsets(&mut lines)?;
    }
    let loop_plan = plan_loops(&lines, options.auto_fast_loops, options.cond_jump_tail_pad);

    let mut sections: Vec<(String, SectionData)> = Vec::new();
    let mut globals: Vec<String> = Vec::new();
    let mut weaks: Vec<String> = Vec::new();
    let mut externs: Vec<String> = Vec::new();
    let mut aliases: HashMap<String, String> = HashMap::new();
    let mut current_section_idx: Option<usize> = None;
    let mut pending: Vec<PendingInstr> = Vec::new();
    // Which DO-loop body instructions stay 48-bit, and which loops get
    // the automatic fast bit, is decided up front by `plan_loops`.
    // Set by `.NOCOMPRESS;`, cleared by `.COMPRESS;`.  While true, every
    // PM-code instruction is emitted in 48-bit ISA form even in a VISA
    // section.  This is required around hardware errata sequences where
    // the SHARC+ counts in 48-bit instruction units (e.g. the 12 NOPs
    // after a SHBTB_CFG write) and a compressed parcel would silently
    // shorten the window and hang the core.
    let mut no_compress_dir: bool = false;
    // Relocations produced by `.VAR sym = other_symbol;` directives.
    // Folded into the main relocation table at the end of pass 2 so
    // the linker patches symbol addresses into data words for
    // file-scope arrays of function pointers and pointer-typed
    // globals initialised with an address.
    let mut var_relocs: Vec<PendingReloc> = Vec::new();

    // Pass 1: encode instructions, build label map, track unresolved refs
    for (line_idx, line) in lines.iter().enumerate() {
        // `.ALIGN` before one of a DO loop's last ten instructions:
        // The reference assembler rejects it in a VISA section (with eleven
        // following it is accepted); NW accepts, with 48-bit fill NOPs.
        // No vendor vector covers a whole-body 48-bit loop.
        if matches!(line.directive, Some(Directive::Align(_))) {
            let next = lines[line_idx + 1..]
                .iter()
                .position(|l| l.instruction.is_some())
                .map(|p| line_idx + 1 + p);
            let sec_visa = current_section_idx
                .map(|i| sections[i].1.is_visa)
                .unwrap_or(visa);
            if next.is_some_and(|n| {
                loop_plan.align_refused.contains(&n)
                    || (sec_visa && loop_plan.align_last_ten.contains(&n))
            }) {
                return Err(Error::Parse {
                    line: line.line,
                    msg: "`.ALIGN` inside the last ten instructions of a DO loop (reference assembler)"
                        .into(),
                });
            }
        }
        let mut state = DirectiveState {
            sections: &mut sections,
            globals: &mut globals,
            weaks: &mut weaks,
            externs: &mut externs,
            aliases: &mut aliases,
            current_section_idx: &mut current_section_idx,
            var_relocs: &mut var_relocs,
            visa,
            nwc: options.nwc,
            swc_exclude: options.swc_exclude,
        };
        process_directives(line, &mut state)?;

        // `.NOCOMPRESS` / `.COMPRESS` toggles affect every subsequent
        // PM-code instruction.  Track them outside `process_directives`
        // so the flag lives next to the existing `no_compress_label`
        // tracking that drives the encoder's `force_isa` decision.
        match &line.directive {
            Some(Directive::NoCompress) => no_compress_dir = true,
            Some(Directive::Compress) => no_compress_dir = false,
            _ => {}
        }

        // A label on a data line was placed before its data by
        // `process_directives`.
        let data_line = matches!(line.directive, Some(Directive::Data(_)));
        if let Some(label) = line.label.as_ref().filter(|_| !data_line) {
            let idx =
                current_or_default(&mut sections, &mut current_section_idx, ".text", true, visa);
            let sec = &mut sections[idx].1;
            // A `<name>.end` label closes a function: the padding goes
            // before it (`.LN.fill..end` = 44, after two NOPs).
            if options.cond_jump_tail_pad && sec.is_pm && label.ends_with(".end") {
                sec.pad_cond_jump_tail(loop_plan.pad_room.get(&line_idx).copied().unwrap_or(0))?;
            }
            // Use per-section visa, not global: a `.SECTION/CODE/NW`
            // section keeps `is_visa = false` even when assembling in
            // VISA mode globally, and label offsets must be measured
            // in the same units the encoder will emit.
            // Data sections are byte-addressed: a label there takes the
            // byte offset, like a `.VAR` symbol (the reference assembler: `.ldr..end`
            // = 216 after `ldr.[54]`).
            let word_off = if sec.is_visa {
                sec.visa_offset()
            } else if sec.byte_addressed {
                sec.code.len() as u32
            } else {
                sec.word_offset()
            };
            sec.symbols.push((label.clone(), word_off));
        }
        // Generated `___ADI_AGL_*` labels sit at the instruction itself,
        // after any `.end` pads emitted for this line's source label.
        if !line.generated_labels.is_empty() {
            let idx =
                current_or_default(&mut sections, &mut current_section_idx, ".text", true, visa);
            let sec = &mut sections[idx].1;
            let word_off = if sec.is_visa {
                sec.visa_offset()
            } else {
                sec.word_offset()
            };
            for label in &line.generated_labels {
                sec.symbols.push((label.clone(), word_off));
            }
        }

        if let Some(instr) = &line.instruction {
            let mut instr = *instr;
            if loop_plan.auto_fast.contains(&line_idx) {
                match &mut instr {
                    selinstr::encode::Instruction::DoLoop { f, .. }
                    | selinstr::encode::Instruction::DoUntil { f, .. } => *f = true,
                    _ => {}
                }
            }
            let instr = &instr;
            // A symbolic load or store carries its constant as the
            // relocation addend; like the reference assembler, leave 0 in the field.
            let field_instr = match *instr {
                selinstr::encode::Instruction::LoadImm { ureg, .. } if line.label_ref.is_some() => {
                    selinstr::encode::Instruction::LoadImm { ureg, value: 0 }
                }
                selinstr::encode::Instruction::ImmStore {
                    pm, i_reg, m_reg, ..
                } if line.label_ref.is_some() => selinstr::encode::Instruction::ImmStore {
                    pm,
                    i_reg,
                    m_reg,
                    value: 0,
                },
                selinstr::encode::Instruction::UregAbsAccess {
                    pm,
                    write,
                    ureg,
                    width,
                    ..
                } if line.label_ref.is_some() => selinstr::encode::Instruction::UregAbsAccess {
                    pm,
                    write,
                    ureg,
                    addr: 0,
                    width,
                },
                other => other,
            };
            let idx =
                current_or_default(&mut sections, &mut current_section_idx, ".text", true, visa);
            let sec = &mut sections[idx].1;
            if sec.is_pm && sec.delay_slots > 0 && delay_slot_illegal(instr) {
                return Err(Error::Parse {
                    line: line.line,
                    msg: "jump, call, return, DO or PC stack instruction in a delay slot \
                          (reference assembler)"
                        .into(),
                });
            }
            let byte_offset = sec.code.len();
            // Capture the word/parcel offset of this instruction *before*
            // encoding it, so pass 2 can compute PC-relative offsets
            // correctly even when earlier instructions in the section had
            // varying widths (VISA: 2/4/6 bytes).
            let word_offset = if sec.is_visa {
                sec.visa_offset()
            } else {
                sec.word_offset()
            };
            if sec.is_visa {
                // Force the 48-bit ISA form under either of two
                // conditions:
                //   1. The instruction is one of the last `LOOP_TAIL`
                //      instructions of a DO loop body (`plan_loops`),
                //      which the reference assembler always keeps in 48-bit form.
                //   2. The instruction carries a pending relocation
                //      against a symbolic label. The SHARC+ PM reloc
                //      types (`R_SHARC_PM32` / `R_SHARC_PM24` / their
                //      PC-relative cousins) all patch byte slots 2..5
                //      or 3..5 of a 48-bit instruction. If selas
                //      compresses the instruction into a 2/4-byte VISA
                //      form, the linker's patcher writes past the
                //      instruction end -- or, in the former (mis-
                //      typed) case, straight over the opcode -- and
                //      the resulting code is corrupt. The only
                //      instruction that visa_encode currently
                //      compresses in practice is `LoadImm` via the
                //      Type 17b path, but checking `label_ref` is
                //      future-proof and costs nothing at call sites
                //      that do not have a reloc.
                let force_isa = loop_plan.force48.contains(&line_idx)
                    || no_compress_dir
                    || line.label_ref.is_some();
                let wide = || -> Result<selinstr::visa_encode::VisaEncoded> {
                    Ok(selinstr::visa_encode::VisaEncoded::W48(
                        selinstr::encode::encode(&field_instr)?,
                    ))
                };
                let encoded = if force_isa {
                    wide()?
                } else {
                    let short = selinstr::visa_encode::encode_visa(&field_instr)?;
                    match branch_shadow(instr, sec.parcel_count, sec.jump_end) {
                        Some(min) if short.parcels() < min => wide()?,
                        _ => short,
                    }
                };
                let bytes = encoded.to_bytes();
                sec.parcel_count += encoded.parcels();
                if starts_branch_shadow(instr) {
                    sec.jump_end = Some(sec.parcel_count);
                }
                sec.code.extend_from_slice(&bytes);
            } else {
                let bytes = selinstr::encode::encode(&field_instr)?;
                sec.code.extend_from_slice(&bytes);
            }
            let byte_len = sec.code.len() - byte_offset;
            if sec.is_pm {
                sec.note_instruction(instr);
            }
            if let Some(ref label_ref) = line.label_ref {
                pending.push(PendingInstr {
                    section_idx: idx,
                    byte_offset,
                    byte_len,
                    word_offset,
                    instr: *instr,
                    label_ref: label_ref.clone(),
                    extra_label_refs: line.extra_label_refs.clone(),
                    // The reference assembler never resolves a DO target in place: the field
                    // keeps 0 (or the numeric n) and a LOOP16 relocation does
                    // the rest, even for a local label. selas's own
                    // `.L_` loop labels still resolve here.
                    link_time: !line.extra_label_refs.is_empty()
                        || (matches!(
                            instr,
                            selinstr::encode::Instruction::Branch {
                                target: selinstr::encode::BranchTarget::PcRelative(_),
                                ..
                            }
                        ))
                        || (matches!(
                            instr,
                            selinstr::encode::Instruction::DoLoop { .. }
                                | selinstr::encode::Instruction::DoUntil { .. }
                        ) && !label_ref.starts_with(".L_")),
                });
            }
        }
    }

    // A section's end is padded after its trailing labels (a module's
    // `_mod_code_tail_end` = 0x1cb, section 0x1d0 parcels). Done once, at
    // the end: a section left and reopened later keeps its tail open.
    if options.cond_jump_tail_pad {
        for (_, sec) in sections.iter_mut().filter(|(_, sec)| sec.is_pm) {
            sec.pad_cond_jump_tail(0)?;
        }
    }

    // Pass 2: resolve label references in branch and loop instructions.
    // Labels defined in the same section as the referring instruction
    // become a position-independent rewrite (PC-relative branch or DO
    // loop RELADDR). Cross-section, external, and ISA-mandated-absolute
    // targets become R_SHARC_ADDR24 relocations that the linker patches.
    // Local-label relocations are backed by STB_LOCAL symbols emitted
    // into the symbol table so the linker can resolve them against the
    // in-file definition.
    let mut relocs: Vec<PendingReloc> = Vec::new();
    let mut local_refs: Vec<(String, usize, u32)> = Vec::new();
    if !pending.is_empty() {
        let label_map = build_label_map(&sections);
        for pi in &pending {
            let resolved = if pi.link_time {
                None
            } else {
                resolve_labels(
                    &pi.instr,
                    &pi.label_ref,
                    &label_map,
                    pi.section_idx,
                    pi.word_offset,
                )
            };
            match resolved {
                Some(resolved) => {
                    let sec = &mut sections[pi.section_idx].1;
                    if sec.is_visa {
                        let isa_bytes = selinstr::encode::encode(&resolved)?;
                        let encoded = selinstr::visa_encode::visa_encode(&resolved, &isa_bytes);
                        let bytes = encoded.to_bytes();
                        sec.code[pi.byte_offset..pi.byte_offset + pi.byte_len]
                            .copy_from_slice(&bytes);
                    } else {
                        let bytes = selinstr::encode::encode(&resolved)?;
                        sec.code[pi.byte_offset..pi.byte_offset + pi.byte_len]
                            .copy_from_slice(&bytes);
                    }
                }
                None => {
                    match label_map.get(&pi.label_ref) {
                        Some(&(sec_idx, word_offset)) => {
                            if !local_refs.iter().any(|(n, _, _)| n == &pi.label_ref) {
                                local_refs.push((pi.label_ref.clone(), sec_idx, word_offset));
                            }
                        }
                        None => {
                            if !externs.contains(&pi.label_ref) {
                                externs.push(pi.label_ref.clone());
                            }
                        }
                    }
                    // Keep the legacy handling of missing compiler-generated
                    // targets, but never suppress a defined target's relocation
                    // based on its spelling. Explicit (PC,.L_branch_*) needs
                    // PCREL24 just like (PC,.P33L1).
                    let unresolved_local_branch = matches!(
                        pi.instr,
                        selinstr::encode::Instruction::Branch { .. }
                            | selinstr::encode::Instruction::DoLoop { .. }
                            | selinstr::encode::Instruction::DoUntil { .. }
                    ) && (pi.label_ref.starts_with(".L_branch_")
                        || pi.label_ref.starts_with(".L_doloop_"));
                    if pi.label_ref.starts_with(".L_")
                        && !pi.label_ref.starts_with(".L_ret_")
                        && unresolved_local_branch
                        && !label_map.contains_key(&pi.label_ref)
                    {
                        continue;
                    }
                    // All code-section relocations use the SHARC+ PM
                    // relocation family.  `r_offset` is counted in the
                    // section's *native instruction unit*, which the
                    // linker then converts back to a byte offset in
                    // `reloc_byte_offset`:
                    //
                    //   * SW (VISA) sections address in 16-bit parcels,
                    //     so `byte = r_offset * 2`.
                    //   * PM48 (`/NW`) sections address in 48-bit
                    //     instructions, so `byte = r_offset * 6`.
                    //
                    // selas previously divided every byte offset by 2
                    // unconditionally, which placed iv_code (NW)
                    // relocations at the wrong site after the linker
                    // recovered the byte offset as `r_offset * 6`.  The
                    // visible symptom was a board hang on reset because
                    // the slot-1 `JUMP start` field never received its
                    // patched address.
                    //
                    // For ImmStore with a non-zero value (e.g. from
                    // a `symbol-1` expression), the value becomes the
                    // relocation addend so the linker computes
                    // symbol_address + addend.
                    let addend = match &pi.instr {
                        selinstr::encode::Instruction::ImmStore { value, .. }
                        | selinstr::encode::Instruction::LoadImm { value, .. } => *value as i32,
                        selinstr::encode::Instruction::UregAbsAccess { addr, .. } => *addr as i32,
                        _ => 0,
                    };
                    let sec = &sections[pi.section_idx].1;
                    let unit = if sec.is_visa { 2 } else { 6 };
                    let r_offset = (pi.byte_offset / unit) as u32;
                    // The relocation sequence is
                    // PUSH _e, PUSH _t, SUBTRACT, LOOP16 .__operator.
                    // These are stack operations, distinct from selas's
                    // older additive LoadImm relocation convention below.
                    if matches!(
                        pi.instr,
                        selinstr::encode::Instruction::DoLoop { .. }
                            | selinstr::encode::Instruction::DoUntil { .. }
                    ) && !pi.extra_label_refs.is_empty()
                    {
                        let (rhs, sign) = &pi.extra_label_refs[0];
                        debug_assert_eq!((*sign, pi.extra_label_refs.len()), (-1, 1));
                        // Only the section-start base is evidenced.
                        // Then a-b has a's section offset and plan_loops can
                        // safely use a as the end. Other bases need a reference
                        // vector before choosing the compressed loop tail.
                        if label_map.get(rhs) != Some(&(pi.section_idx, 0)) {
                            return Err(Error::Parse {
                                line: 0,
                                msg: "DO label difference requires a base at this section's start"
                                    .into(),
                            });
                        }
                        if let Some(&(idx, offset)) = label_map.get(rhs) {
                            if !local_refs.iter().any(|(n, _, _)| n == rhs) {
                                local_refs.push((rhs.clone(), idx, offset));
                            }
                        } else if !externs.contains(rhs) {
                            externs.push(rhs.clone());
                        }
                        // Its value is the DO's own offset (DO at parcel 1,
                        // `.__operator` = 1; at parcel 0: 0).
                        if !local_refs.iter().any(|(n, _, _)| n == ".__operator") {
                            local_refs.push((".__operator".into(), pi.section_idx, r_offset));
                        }
                        for (symbol, rela_type) in [
                            (pi.label_ref.as_str(), 0xe0),
                            (rhs.as_str(), 0xe0),
                            (".__operator", 0xe3),
                            (".__operator", reloc_type_for(&pi.instr)),
                        ] {
                            relocs.push(PendingReloc {
                                section_idx: pi.section_idx,
                                byte_offset: r_offset,
                                symbol: symbol.into(),
                                rela_type,
                                addend: 0,
                            });
                        }
                        continue;
                    }
                    // For multi-symbol expressions like
                    // `I7 = ldf_stack_space + ldf_stack_length;`,
                    // emit the primary and each extra symbol as
                    // `R_SHARC_PM_SW_BRANCHRETURN` (Add semantic).
                    // The encoded imm32 field is zero, so the first
                    // Add writes the primary symbol's value and each
                    // subsequent Add accumulates.  Subtraction is
                    // expressed via the addend: an extra entry with
                    // sign=-1 carries `-symbol` by emitting the
                    // primary-write reloc followed by an Add of the
                    // negated symbol's resolved value, but seld's
                    // Add path takes the symbol's positive address;
                    // libsel only ever uses `+ symbol` between
                    // externs, and a negative connector falls back
                    // to a single-symbol relocation by leaving the
                    // primary as a plain write.
                    let has_extras = !pi.extra_label_refs.is_empty()
                        && pi.extra_label_refs.iter().all(|(_, s)| *s >= 0);
                    let primary_type = if has_extras {
                        selelf::elf::R_SHARC_PM_SW_BRANCHRETURN as u8
                    } else {
                        reloc_type_for(&pi.instr)
                    };
                    relocs.push(PendingReloc {
                        section_idx: pi.section_idx,
                        byte_offset: r_offset,
                        symbol: pi.label_ref.clone(),
                        rela_type: primary_type,
                        addend,
                    });
                    if has_extras {
                        for (extra_sym, _sign) in &pi.extra_label_refs {
                            if !externs.contains(extra_sym) && !label_map.contains_key(extra_sym) {
                                externs.push(extra_sym.clone());
                            }
                            relocs.push(PendingReloc {
                                section_idx: pi.section_idx,
                                byte_offset: r_offset,
                                symbol: extra_sym.clone(),
                                rela_type: selelf::elf::R_SHARC_PM_SW_BRANCHRETURN as u8,
                                addend: 0,
                            });
                        }
                    }
                }
            }
        }
    }

    // Resolve .SET aliases into section symbol tables.
    resolve_aliases(&mut sections, &aliases);

    // Fold in relocations harvested from `.VAR sym = other_symbol;`
    // directives. Each one targets either a label defined elsewhere
    // in this file (recorded as a local symbol so the linker resolves
    // it locally) or an external symbol (added to the externs list
    // so the linker resolves it at link time).
    {
        let label_map = build_label_map(&sections);
        for vr in &var_relocs {
            match label_map.get(&vr.symbol) {
                Some(&(sec_idx, word_offset)) => {
                    if !local_refs.iter().any(|(n, _, _)| n == &vr.symbol) {
                        local_refs.push((vr.symbol.clone(), sec_idx, word_offset));
                    }
                }
                None => {
                    if !externs.contains(&vr.symbol) {
                        externs.push(vr.symbol.clone());
                    }
                }
            }
        }
    }
    relocs.extend(var_relocs);

    Ok(emit_elf_bytes(
        &sections,
        &globals,
        &weaks,
        &externs,
        &local_refs,
        &relocs,
    ))
}

/// Mutable state threaded through directive processing. Bundling the
/// references into one struct keeps `process_directives` to a small
/// argument count and makes the data flow obvious: pass 1 mutates
/// `sections`, `globals`, `externs`, `aliases`, and `var_relocs`
/// while advancing `current_section_idx`.
struct DirectiveState<'a> {
    sections: &'a mut Vec<(String, SectionData)>,
    globals: &'a mut Vec<String>,
    weaks: &'a mut Vec<String>,
    externs: &'a mut Vec<String>,
    aliases: &'a mut HashMap<String, String>,
    current_section_idx: &'a mut Option<usize>,
    /// Link-time relocations produced by `.VAR sym = other_symbol;`
    /// directives, folded into the final relocation table after
    /// pass 2 alongside the instruction-level relocations.
    var_relocs: &'a mut Vec<PendingReloc>,
    visa: bool,
    nwc: bool,
    swc_exclude: &'a [String],
}

/// Process directive and section-state effects from a parsed line.
fn process_directives(line: &ParsedLine, state: &mut DirectiveState<'_>) -> Result<()> {
    let directive = match &line.directive {
        Some(d) => d,
        None => return Ok(()),
    };

    match directive {
        Directive::Section(raw_name) => {
            let (name, is_pm, force_nw, force_sw) = parse_section_name(raw_name);
            // The reference assembler: explicit /SW still compresses under
            // -nwc. /PM and /CODE follow the command-line default.
            // A section is NW under -swc-exclude; explicit /SW
            // overrides that exclusion.
            let excluded = state.swc_exclude.contains(&name);
            let visa_for_section =
                (state.visa || (state.nwc && force_sw)) && !force_nw && (force_sw || !excluded);
            let reopened = state.sections.iter().any(|(n, _)| *n == name);
            let idx = ensure_section(state.sections, &name, is_pm, visa_for_section);
            let sec = &mut state.sections[idx].1;
            let byte_addressed = !sec.is_pm && is_byte_addressed(raw_name);
            // A bare reopen (`.SECTION name;`) keeps the first qualifiers;
            // an explicit one must agree with them.
            // The parser strips the leading `/`: qualifiers are present
            // when the operand has more than the bare name.
            let explicit = raw_name
                .trim()
                .trim_end_matches(';')
                .trim()
                .contains(char::is_whitespace);
            if reopened && explicit && byte_addressed != sec.byte_addressed {
                return Err(Error::Parse {
                    line: 0,
                    msg: format!("section `{name}` reopened with conflicting qualifiers"),
                });
            }
            if !reopened {
                sec.byte_addressed = byte_addressed;
                sec.word_addressed = !sec.is_pm
                    && !byte_addressed
                    && raw_name
                        .rsplit_once(char::is_whitespace)
                        .is_some_and(|(q, _)| q.split('/').any(|t| t.eq_ignore_ascii_case("DM")));
            }
            *state.current_section_idx = Some(idx);
        }
        Directive::Global(names) => {
            for name in names {
                if !state.globals.contains(name) {
                    state.globals.push(name.clone());
                }
            }
        }
        Directive::Weak(names) => {
            for name in names {
                if !state.weaks.contains(name) {
                    state.weaks.push(name.clone());
                }
            }
        }
        Directive::Extern(names) => {
            for name in names {
                if !state.externs.contains(name) {
                    state.externs.push(name.clone());
                }
            }
        }
        Directive::Data(decl) => {
            let idx = current_or_default(
                state.sections,
                state.current_section_idx,
                ".data",
                false,
                state.visa,
            );
            let sec = &mut state.sections[idx].1;
            if !sec.byte_addressed && !sec.word_addressed {
                return Err(Error::Parse {
                    line: 0,
                    msg: format!(
                        "data directive for `{}` needs a data section \
                         (/DM, /DOUBLE32 or /BW), `{}` is not one",
                        decl.name, state.sections[idx].0
                    ),
                });
            }
            // Only constant .VAR words have a retained vendor /DM vector.
            if sec.word_addressed
                && (decl.width != 4 || decl.inits.iter().any(|i| matches!(i, DataInit::Sym(_))))
            {
                return Err(Error::Parse {
                    line: 0,
                    msg: "word-addressed /DM supports constant .VAR initializers only".into(),
                });
            }
            let byte_off = sec.code.len() as u32;
            let address = if sec.word_addressed {
                sec.word_offset()
            } else {
                byte_off
            };
            // A bare `.VAR = v;` (no name) is a continuation word that
            // extends the most recently named array; it registers no
            // symbol.
            if let Some(label) = &line.label {
                sec.symbols.push((label.clone(), address));
            }
            if !decl.name.is_empty() {
                sec.symbols.push((decl.name.clone(), address));
                if sec.word_addressed {
                    sec.dm_symbol_sizes
                        .insert(decl.name.clone(), decl.len as u32);
                }
            }
            if decl.inits.is_empty() {
                sec.code.extend(std::iter::repeat_n(
                    0u8,
                    decl.len * if sec.word_addressed { 5 } else { decl.width },
                ));
            }
            for (k, init) in decl.inits.iter().enumerate() {
                match init {
                    DataInit::Num(v) => {
                        if sec.word_addressed {
                            sec.code.extend_from_slice(&(*v as u32).to_be_bytes());
                            sec.code.push(0);
                        } else {
                            let bytes = (*v as u32).to_le_bytes();
                            sec.code.extend_from_slice(&bytes[..decl.width]);
                        }
                    }
                    DataInit::Sym(target) => {
                        // A zero word patched by the linker with the
                        // absolute address of `target`. The reference assembler uses
                        // R_SHARC_DM_ADDR32 (0x0d) for these in every
                        // vendor object.
                        sec.code.extend_from_slice(&0u32.to_le_bytes());
                        state.var_relocs.push(PendingReloc {
                            section_idx: idx,
                            byte_offset: byte_off + (k * decl.width) as u32,
                            symbol: target.clone(),
                            rela_type: selelf::elf::R_SHARC_DM_ADDR32 as u8,
                            addend: 0,
                        });
                    }
                }
            }
        }
        Directive::Align(boundary) => {
            if let Some(idx) = *state.current_section_idx {
                let sec = &mut state.sections[idx].1;
                // Code aligns in address units and fills with NOPs. A VISA
                // section counts parcels: a gap of 1 = `0001`, 2 = `0001
                // 0001`, 3 = one 48-bit NOP (the reference assembler `.align 4`;
                // no larger gap seen), and the parcel count
                // moves with it so later labels stay right.
                // Data sections are byte-addressed, so `.ALIGN 4` there
                // means four bytes (the reference assembler: a data label stays at 216).
                let b = *boundary;
                if sec.byte_addressed {
                    sec.data_alignment = sec.data_alignment.max(b);
                }
                let here = if sec.is_visa {
                    sec.parcel_count
                } else if sec.word_addressed {
                    sec.word_offset()
                } else {
                    sec.code.len() as u32 / if sec.is_pm { 6 } else { 1 }
                };
                let mut gap = if b > 0 { (b - here % b) % b } else { 0 };
                if sec.word_addressed && gap != 0 {
                    return Err(Error::Parse {
                        line: 0,
                        msg: "alignment padding in word-addressed /DM lacks a vendor layout".into(),
                    });
                }
                while gap > 0 {
                    if !sec.is_pm {
                        sec.code.push(0);
                        gap -= 1;
                    } else if sec.is_visa && gap < 3 {
                        sec.emit_nop(false)?;
                        gap -= 1;
                    } else {
                        sec.emit_nop(true)?;
                        gap -= if sec.is_visa { 3 } else { 1 };
                    }
                }
            }
        }
        Directive::Set(name, value) => {
            state.aliases.insert(name.clone(), value.clone());
        }
        Directive::NoCompress | Directive::Compress => {
            // The `.NOCOMPRESS;` / `.COMPRESS;` toggles do not change
            // section state, label tables, or symbol kind: they only
            // tell the encoder whether to emit subsequent instructions
            // in compressed (VISA) form or in fixed 48-bit ISA form.
            // The flag is tracked alongside `no_compress_label` in
            // pass 1 of the main encoding loop.
        }
    }
    Ok(())
}

/// Resolve `.SET` aliases into section symbol tables.
///
/// For each alias, look up the value: if it names an existing symbol in any
/// section, copy that symbol's address; if it parses as a numeric literal,
/// use that value directly.  The value may also be `<sym> + <num>` or
/// `<sym> - <num>`, in which case the alias resolves to the named symbol's
/// address adjusted by the constant offset.  This sub-word interior form
/// is what selcc emits when a file-scope initialiser takes the address of
/// a byte / short interior of a packed aggregate, e.g. `(char *)&g + 1`,
/// where the resulting address is not word-aligned and therefore cannot
/// be planted as a fresh `.VAR` slot owner.  Chains (A = B, B = C) are
/// resolved transitively up to a bounded depth.
fn resolve_aliases(sections: &mut [(String, SectionData)], aliases: &HashMap<String, String>) {
    for (alias_name, raw_value) in aliases {
        // Resolve transitive chains: follow symbol names through the alias map.
        let mut value = raw_value.clone();
        for _ in 0..16 {
            match aliases.get(&value) {
                Some(next) => value = next.clone(),
                None => break,
            }
        }

        // Decompose the (possibly sym +/- num) form into a base name and a
        // signed addend. Bare names produce addend 0; bare numerics get
        // an empty base and the parsed value as the addend so the numeric
        // literal arm below picks them up unchanged.
        let (base, addend) = parse_sym_plus_offset(&value);

        // Try to find the resolved base as an existing symbol.
        let mut found = false;
        for sec in sections.iter_mut() {
            let hit = sec
                .1
                .symbols
                .iter()
                .find(|(n, _)| *n == base)
                .map(|(_, off)| *off);
            if let Some(addr) = hit {
                let final_addr = (addr as i64 + addend) as u32;
                sec.1.symbols.push((alias_name.clone(), final_addr));
                found = true;
                break;
            }
        }

        if !found {
            // Try as a numeric literal; add to the first section (or skip).
            if let Some(num) = parse_u32_literal(&value) {
                if let Some(first) = sections.first_mut() {
                    first.1.symbols.push((alias_name.clone(), num));
                }
            }
        }
    }
}

/// Decompose a `.SET` value into a `(base_symbol, signed_addend)` pair.
/// Recognises `name`, `name + N`, `name - N`, where `N` is a numeric
/// literal (decimal, hex, or octal). A bare `name` yields `(name, 0)`.
/// A value that does not parse as `sym +/- num` returns `(value, 0)` so
/// the existing numeric-literal fall-back can still apply.
fn parse_sym_plus_offset(value: &str) -> (String, i64) {
    let v = value.trim();
    // Walk the string and find the rightmost top-level `+` or `-` that
    // is not part of an identifier.  Identifiers may contain `_` and
    // `.`; numeric literals start with a digit. The simplest reliable
    // split is to scan from the right looking for `+` or `-` preceded
    // by whitespace or an identifier character so a leading `-` on the
    // numeric literal does not get mistaken for a connector. Any
    // failure to parse falls back to the bare-name path.
    let bytes = v.as_bytes();
    for i in (1..bytes.len()).rev() {
        let c = bytes[i];
        if c != b'+' && c != b'-' {
            continue;
        }
        let lhs = v[..i].trim();
        let rhs = v[i + 1..].trim();
        if lhs.is_empty() || rhs.is_empty() {
            continue;
        }
        // Numeric literal on the right and identifier on the left:
        // accept and return the sym +/- num decomposition.
        if let Some(num) = parse_u32_literal(rhs) {
            let signed = if c == b'-' { -(num as i64) } else { num as i64 };
            return (lhs.to_string(), signed);
        }
    }
    (v.to_string(), 0)
}

/// Serialize the accumulated sections and symbols to ELF .doj bytes.
fn emit_elf_bytes(
    sections: &[(String, SectionData)],
    globals: &[String],
    weaks: &[String],
    externs: &[String],
    local_refs: &[(String, usize, u32)],
    relocs: &[PendingReloc],
) -> Vec<u8> {
    let mut writer = selelf::elf_write::ElfWriter::new();

    // Add sections and collect their ELF section indices. VISA-encoded PM
    // sections use the short-word layout (alignment 1, entsize 1) so the
    // linker accepts them when cross-linking against `-swc` C objects.
    let mut elf_indices: Vec<u16> = Vec::new();
    for (name, sec_data) in sections {
        let idx = if sec_data.is_visa {
            writer.add_text_section_sw(name, &sec_data.code)
        } else if sec_data.is_pm {
            writer.add_text_section(name, &sec_data.code)
        } else if sec_data.word_addressed {
            writer.add_data_section_dm(name, &sec_data.code)
        } else {
            writer.add_data_section_aligned(name, &sec_data.code, sec_data.data_alignment)
        };
        elf_indices.push(idx);
    }

    // Register locally-defined labels that are the targets of relocations
    // in this file. These have to appear in the symbol table as STB_LOCAL
    // so the linker can resolve the relocation against the in-file
    // definition instead of treating the label as an unresolved external.
    //
    // Labels that originated from a `.VAR` directive in a data section
    // (i.e. they appear in `sections[*].symbols`) are emitted as
    // `STT_OBJECT`. the SHARC+ linker treats `STT_NOTYPE` locals as plain
    // labels and resolves them with *word*-granular addresses, while
    // it walks `STT_OBJECT` symbols for byte-granular layout in BW
    // data sections. Without the OBJECT type the relocation against
    // a string literal points 4 byte-addresses higher than the actual
    // first byte (for SwCode-aligned symbols this is hidden by the
    // /4 multiplier; in BW memory it leaves printf reading the
    // padding byte 0x00 instead of the real first character of the
    // format string).
    for (name, sec_idx, word_offset) in local_refs {
        // Skip names that are already declared `.GLOBAL` — those will
        // be emitted as STT_FUNC/STT_OBJECT globals later in this
        // function. Adding a duplicate STB_LOCAL entry for the same
        // name produces two symtab entries that point at the same
        // address; the SHARC+ linker resolves later in-file references
        // (CJUMPs, address-of) to the first match it sees, which is
        // the local NOTYPE — and a NOTYPE symbol is treated as a
        // word-granular label, so byte-addressed data references
        // (and even some PM cjumps) resolve to the wrong runtime
        // value.
        if globals.iter().any(|g| g == name) || weaks.iter().any(|g| g == name) {
            continue;
        }
        let elf_idx = elf_indices[*sec_idx];
        let from_var = sections
            .get(*sec_idx)
            .map(|(_, sd)| sd.symbols.iter().any(|(n, _)| n == name))
            .unwrap_or(false);
        let is_data_sec = sections
            .get(*sec_idx)
            .map(|(_, sd)| !sd.is_pm)
            .unwrap_or(false);
        if from_var && is_data_sec {
            let size = sections[*sec_idx]
                .1
                .dm_symbol_sizes
                .get(name)
                .copied()
                .unwrap_or(0);
            writer.add_local_object(name, elf_idx, *word_offset, size);
        } else {
            writer.add_local(name, elf_idx, *word_offset);
        }
    }

    // Register extern (undefined) symbols. Skip any extern that is
    // also declared `.GLOBAL`/`.WEAK` or defined as a label in the
    // same file: hand-written runtime assembly (e.g.
    // libsel/support/startup.s) often pairs a forward `.EXTERN start;`
    // with a later `.GLOBAL start;` + `start:` definition, so emitting
    // an UNDEF entry alongside the defined GLOBAL would produce two
    // `start` symtab rows. cc21k and elfar's archive indexer both
    // treat the duplicate as a multi-defined symbol, prevent the
    // archive member from being pulled, and silently strand the
    // entry point.
    let defined_in_file: std::collections::HashSet<&str> = sections
        .iter()
        .flat_map(|(_, sd)| sd.symbols.iter().map(|(n, _)| n.as_str()))
        .chain(globals.iter().map(|s| s.as_str()))
        .chain(weaks.iter().map(|s| s.as_str()))
        .collect();
    for ext in externs {
        if defined_in_file.contains(ext.as_str()) {
            continue;
        }
        writer.add_undefined(ext);
    }

    // Register symbols per section. Track globally-exported function names
    // for each PM section so we can populate the `.attributes` metadata.
    // Each entry is (elf index, section name, VISA flag, exported functions).
    let mut pm_section_funcs: Vec<(u16, String, bool, Vec<String>)> = Vec::new();
    for (sec_idx, (name, sec_data)) in sections.iter().enumerate() {
        let elf_idx = elf_indices[sec_idx];
        let mut funcs = Vec::new();
        for (sym_name, word_off) in &sec_data.symbols {
            let is_global = globals.iter().any(|g| g == sym_name);
            let is_weak = weaks.iter().any(|g| g == sym_name);
            if is_global || is_weak {
                if sec_data.is_pm {
                    if is_weak {
                        writer.add_weak_function(sym_name, elf_idx, *word_off, 0);
                    } else {
                        writer.add_function(sym_name, elf_idx, *word_off, 0);
                    }
                    funcs.push(sym_name.clone());
                } else {
                    let size = sec_data.dm_symbol_sizes.get(sym_name).copied().unwrap_or(0);
                    if is_weak {
                        writer.add_weak_object(sym_name, elf_idx, *word_off, size);
                    } else {
                        writer.add_object(sym_name, elf_idx, *word_off, size);
                    }
                }
            }
        }
        if sec_data.is_pm {
            pm_section_funcs.push((elf_idx, name.clone(), sec_data.is_visa, funcs));
        }
    }

    // Globals declared but not found as labels: attach to first section.
    let first_elf_idx = elf_indices.first().copied().unwrap_or(0);
    for g in globals {
        let found = sections
            .iter()
            .any(|(_, sd)| sd.symbols.iter().any(|(n, _)| n == g));
        if !found {
            writer.add_function(g, first_elf_idx, 0, 0);
        }
    }
    for g in weaks {
        let found = sections
            .iter()
            .any(|(_, sd)| sd.symbols.iter().any(|(n, _)| n == g));
        if !found {
            writer.add_weak_function(g, first_elf_idx, 0, 0);
        }
    }

    // Emit deferred relocations against undefined symbols.
    for r in relocs {
        let elf_idx = elf_indices[r.section_idx];
        writer.add_relocation(elf_idx, r.byte_offset, &r.symbol, r.rela_type, r.addend);
    }

    // Emit `.align.<name>` sections for every PM section. The linker
    // requires this placement metadata to accept the object file.
    for (idx, name, _, _) in &pm_section_funcs {
        writer.add_align_section(name, *idx);
    }

    // Emit a `.attributes` string table naming the exported functions and
    // the encoding / content tags the linker uses to classify code sections.
    // VISA sections are tagged as SW (short-word) for compatibility with
    // `-swc` compiled C objects; non-VISA PM sections stay NW.
    if !pm_section_funcs.is_empty() {
        let any_visa = pm_section_funcs.iter().any(|(_, _, v, _)| *v);
        let encoding: &[u8] = if any_visa { b"SW" } else { b"NW" };
        let mut attrs = Vec::new();
        attrs.push(0u8); // leading null
        for (_, _, _, funcs) in &pm_section_funcs {
            for f in funcs {
                attrs.extend_from_slice(b"FuncName\t");
                attrs.extend_from_slice(f.as_bytes());
                attrs.push(0);
            }
        }
        attrs.extend_from_slice(b"Encoding\t");
        attrs.extend_from_slice(encoding);
        attrs.push(0);
        attrs.extend_from_slice(b"Content\tCode\0");
        writer.add_section(".attributes", selelf::elf::SHT_STRTAB, 0, &attrs);
    }

    // Emit `.adi.attributes` with the processor identity and per-section
    // encoding metadata. Without this section the linker rejects the file.
    let mut encoded_sections: Vec<_> = pm_section_funcs
        .iter()
        .map(|(i, _, v, _)| (*i, if *v { 3 } else { 2 }))
        .collect();
    encoded_sections.extend(
        sections
            .iter()
            .zip(&elf_indices)
            .filter_map(|((_, sec), idx)| sec.word_addressed.then_some((*idx, 1))),
    );
    writer.add_section(
        ".adi.attributes",
        selelf::elf::SHT_SHARC_ATTR,
        0,
        &build_adi_attributes(&encoded_sections),
    );

    writer.finish()
}

/// Build the `.adi.attributes` blob for ADSP-21569.
///
/// An ARM-style build-attributes envelope carrying file-level tags
/// (processor name, short-word mode, char size, silicon revision, tool
/// version) and per-section subsections. Vendor Tag_Encoding is 0x02 for
/// NW code, 0x03 for SW code, and
/// 0x01 for plain /DM words.
fn build_adi_attributes(encoded_sections: &[(u16, u8)]) -> Vec<u8> {
    let proc_name = b"ADSP-21569\0";

    let mut file_attrs = Vec::new();
    file_attrs.push(0x04);
    file_attrs.extend_from_slice(proc_name);
    file_attrs.extend_from_slice(&[0x05, 0x0b]); // short-word mode
    file_attrs.extend_from_slice(&[0x06, 0x64]); // char size 8-bit
    file_attrs.extend_from_slice(&[0x07, 0xff, 0xff, 0x03]); // silicon rev = any (ULEB128 0xffff)
    file_attrs.extend_from_slice(&[0x08, 0x01]); // build tool version

    let file_block_size = 1 + 4 + file_attrs.len() as u32;
    let mut file_block = Vec::new();
    file_block.push(0x01); // Tag_File
    file_block.extend_from_slice(&file_block_size.to_le_bytes());
    file_block.extend_from_slice(&file_attrs);

    for &(idx, encoding) in encoded_sections {
        let mut sec_content = Vec::new();
        let mut idx = idx;
        loop {
            let byte = (idx & 0x7f) as u8;
            idx >>= 7;
            sec_content.push(byte | if idx != 0 { 0x80 } else { 0 });
            if idx == 0 {
                break;
            }
        }
        sec_content.extend_from_slice(&[0, 0x12, encoding]);

        let sec_block_size = 1 + 4 + sec_content.len() as u32;
        file_block.push(0x02); // Tag_Section
        file_block.extend_from_slice(&sec_block_size.to_le_bytes());
        file_block.extend_from_slice(&sec_content);
    }

    let producer = b"AnonADI\0";
    let total_size = 4 + producer.len() as u32 + file_block.len() as u32;
    let mut blob = Vec::new();
    blob.push(b'A');
    blob.extend_from_slice(&total_size.to_le_bytes());
    blob.extend_from_slice(producer);
    blob.extend_from_slice(&file_block);
    blob
}

// -----------------------------------------------------------------------
// Helpers for tests: parse ELF section headers
// -----------------------------------------------------------------------

#[cfg(test)]
fn find_section_by_name(
    data: &[u8],
    hdr: &selelf::elf::Elf32Header,
    target_name: &str,
) -> Option<selelf::elf::Elf32Shdr> {
    let shoff = hdr.e_shoff as usize;
    let shentsize = hdr.e_shentsize as usize;
    let shnum = hdr.e_shnum as usize;

    // Locate the section-header string table.
    let strtab_off = shoff + hdr.e_shstrndx as usize * shentsize;
    let strtab_shdr = selelf::elf::parse_section_header(&data[strtab_off..], hdr.ei_data);
    let strtab_start = strtab_shdr.sh_offset as usize;
    let strtab_end = strtab_start + strtab_shdr.sh_size as usize;
    let strtab = &data[strtab_start..strtab_end];

    for i in 0..shnum {
        let off = shoff + i * shentsize;
        let shdr = selelf::elf::parse_section_header(&data[off..], hdr.ei_data);
        let name = selelf::elf::read_string_at(strtab, shdr.sh_name);
        if name == target_name {
            return Some(shdr);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_SEQ: AtomicU32 = AtomicU32::new(0);

    fn visa_instruction_bytes(instruction: &str) -> Vec<u8> {
        let source = format!(".section/SW/DOUBLE32 code;\n{instruction};\n");
        // The instruction alone, without a conditional JUMP's tail padding.
        let options = AsmOptions {
            cond_jump_tail_pad: false,
            ..AsmOptions::default()
        };
        let object = assemble_source_with(&source, true, options).unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        let section = find_section_by_name(&object, &header, "code").unwrap();
        object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize].to_vec()
    }

    #[test]
    fn dag_pre_modify_matches_vendor_vectors() {
        // The reference assembler. These are
        // independent vendor bytes, not encoder/decoder self-consistency.
        let vectors: &[(&str, &[u8])] = &[
            ("r0=dm(m6,i0)", &[0x41, 0xbe, 0x00, 0x3f]),
            ("r0=dm(i0,m6)", &[0x91, 0x90]),
            ("r0=dm(m7,i0)", &[0x41, 0xfe, 0x00, 0x3f]),
            ("r0=dm(i0,m7)", &[0x91, 0xd0]),
            ("dm(m5,i1)=r0", &[0x43, 0x7e, 0x80, 0x3f]),
            ("dm(i1,m5)=r0", &[0x93, 0x70]),
        ];
        for (source, expected) in vectors {
            assert_eq!(visa_instruction_bytes(source), *expected, "{source}");
        }
    }

    #[test]
    fn dag_post_modify_is_preserved_when_16_bit_form_is_unavailable() {
        // Exercise conditional and non-data-register accesses as well as
        // the vendor's unconditional R0 examples.
        for (pre, post) in [
            ("if eq r3=dm(m6,i2)", "if eq r3=dm(i2,m6)"),
            ("dm(m5,i6)=astatx", "dm(i6,m5)=astatx"),
        ] {
            let pre = visa_instruction_bytes(pre);
            let post = visa_instruction_bytes(post);
            assert_eq!(pre.len(), 4);
            assert_eq!(post.len(), 4);
            assert_eq!(post[0], pre[0] | 0x10);
            assert_eq!(post[1..], pre[1..]);
        }
    }

    #[test]
    fn dag_short_word_matches_vendor_vectors() {
        // The reference assembler.
        let vectors: &[(&str, &[u8])] = &[
            ("dm(i1,m6)=r0 (sw)", &[0x53, 0xbe, 0xc0, 0x3c]),
            ("dm(i1,m6)=r0", &[0x93, 0xb0]),
            ("r0=dm(m5,i0)(sw)", &[0x41, 0x7e, 0x40, 0x3c]),
            // The evaluation explicitly corrects this source register to R3.
            ("r3=dm(i0,m6)", &[0x91, 0x93]),
            // An earlier evaluation paired these bytes with R0; the
            // reference assembler's output shows the source register is R2 (a transcription
            // error, like the R3 row above).
            ("r2=dm(i0,m6)(sw)", &[0x51, 0xbe, 0x41, 0x3c]),
            // Same source.
            ("r2=dm(i4,m6)(sw)", &[0x59, 0xbe, 0x41, 0x3c]),
            ("r5=dm(m5,i4)(sw)", &[0x49, 0x7e, 0x42, 0xbc]),
            ("r2=dm(m5,i5)(sw)", &[0x4b, 0x7e, 0x41, 0x3c]),
        ];
        for (source, expected) in vectors {
            assert_eq!(visa_instruction_bytes(source), *expected, "{source}");
        }
    }

    #[test]
    fn dag_byte_word_matches_vendor_vectors() {
        let vectors: &[(&str, &[u8])] = &[
            ("r2=dm(m5,i1)(bw)", &[0x43, 0x7e, 0x01, 0x3c]),
            ("r1=dm(m5,i4)(bw)", &[0x49, 0x7e, 0x00, 0xbc]),
            ("dm(m5,i4)=r0 (bw)", &[0x49, 0x7e, 0x80, 0x3c]),
            ("dm(m5,i4)=r8 (bw)", &[0x49, 0x7e, 0x84, 0x3c]),
            ("r4=dm(m5,i4)(bw)", &[0x49, 0x7e, 0x02, 0x3c]),
        ];
        for (source, expected) in vectors {
            assert_eq!(visa_instruction_bytes(source), *expected, "{source}");
        }
    }

    #[test]
    fn do_loop_f_suffix_matches_vendor_bytes() {
        // The reference assembler: the (F) suffix sets bit 23;
        // the body is 48-bit. The loop end is a LOOP16 relocation (field 0),
        // which the linker resolves to 6 parcels (0c004c800006).
        let source = ".section/SW/DOUBLE32 code;\n\
                      lcntr = 0x4c, do _copy_end until lce (f);\n\
                      r0 = dm(i0, m6)(sw);\n\
                      _copy_end:\n\
                      dm(i1, m6) = r0 (sw);\n";
        let object = assemble_source(source, true).unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        let section = find_section_by_name(&object, &header, "code").unwrap();
        let bytes =
            &object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize];
        assert_eq!(
            bytes,
            [
                0x0c, 0x00, 0x4c, 0x80, 0x00, 0x00, 0x51, 0xbe, 0x40, 0x30, 0x00, 0x00, 0x53, 0xbe,
                0xc0, 0x30, 0x00, 0x00
            ]
        );
        let relas = object_relas(&object, ".rela.code");
        assert_eq!(relas.len(), 1);
        assert_eq!(
            (relas[0].0, relas[0].1 & 0xff),
            (0, selelf::elf::R_SHARC_PM_LOOP16)
        );
        assert_eq!(object_symbols(&object)["_copy_end"], 6);
        let listing =
            selinstr::visa::disassemble_visa(&bytes[..6], 0, selinstr::disasm::decode_instruction);
        assert!(
            listing[0].text.to_uppercase().ends_with("(F)"),
            "{}",
            listing[0].text
        );
        for bad in [
            "lcntr = 2, do end until lce (x)",
            "lcntr = 2, do end until ne",
        ] {
            let source = format!(".section/SW/DOUBLE32 code;\n{bad};\nend: nop;\n");
            assert!(assemble_source(&source, true).is_err(), "{bad}");
        }
    }

    fn section_bytes(source: &str, name: &str) -> Vec<u8> {
        let object = assemble_source(source, true).unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        let section = find_section_by_name(&object, &header, name).unwrap();
        object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize].to_vec()
    }

    #[test]
    fn immediate_offset_width_matches_vendor_vectors() {
        // 32-bit Type 4b width markers: 0x3c short word, 0x38 byte word.
        assert_eq!(
            visa_instruction_bytes("r12 = dm(0x2, i4)(sw)"),
            [0x68, 0x3e, 0x16, 0x3c]
        );
        assert_eq!(
            visa_instruction_bytes("r2 = dm(1, i4)(bw)"),
            [0x68, 0x3e, 0x09, 0x38]
        );
        // Post-modify: same marker, update bit p1[6] set.
        assert_eq!(
            visa_instruction_bytes("r0 = dm(i4, 1)(bw)"),
            [0x68, 0x7e, 0x08, 0x38]
        );
        // Stores set the write bit p1[7]. No listing row has one; a binary
        // round-trip reproduces 86 of them byte for byte.
        assert_eq!(
            visa_instruction_bytes("dm(1, i4) = r2 (bw)"),
            [0x68, 0xbe, 0x09, 0x38]
        );
        assert_eq!(
            visa_instruction_bytes("dm(i4, 1) = r2 (sw)"),
            [0x68, 0xfe, 0x09, 0x3c]
        );
        // Forced 48-bit slots (loop end, .NOCOMPRESS, NW sections) carry the
        // width in the empty compute slot: (bw) 0x30, (sw) 0x34, (swse) 0x35
        // (the reference assembler objects 687e08300000, 647e10340000, 687e11350000). (BWSE)
        // is 32-bit only (643fd839); no 48-bit form has been seen.
        for (src, word) in [
            (
                ".section/SW/DOUBLE32 code;\n.nocompress;\nr0 = dm(i4, 1)(bw);\n",
                [0x68, 0x7e, 0x08, 0x30, 0x00, 0x00],
            ),
            (
                ".section/SW/DOUBLE32 code;\n.nocompress;\nr0 = dm(i2, 0x2)(sw);\n",
                [0x64, 0x7e, 0x10, 0x34, 0x00, 0x00],
            ),
            (
                ".section/SW/DOUBLE32 code;\n.nocompress;\nr2 = dm(i4, 2)(swse);\n",
                [0x68, 0x7e, 0x11, 0x35, 0x00, 0x00],
            ),
            (
                ".section/SW/DOUBLE32 code;\n.nocompress;\nr0 = dm(m4, i2)(swse);\n",
                [0x45, 0x3e, 0x40, 0x31, 0x00, 0x00],
            ),
            (
                ".section/SW/DOUBLE32 code;\n.nocompress;\nr12 = dm(i0, m6)(bwse);\n",
                [0x51, 0xbe, 0x06, 0x31, 0x00, 0x00],
            ),
        ] {
            let bytes = section_bytes(src, "code");
            assert_eq!(bytes[bytes.len() - 6..], word, "{src}");
        }
        let bad = ".section/SW/DOUBLE32 code;\n.nocompress;\nr2 = dm(1, i4)(bwse);\n";
        assert!(assemble_source(bad, true).is_err(), "{bad}");
    }

    #[test]
    fn register_copy_is_a_transfer_not_pass() {
        // `Rn = Rm` is a Type 5 register transfer that leaves ASTAT
        // alone; it used to assemble as `Rn = PASS Rm` (c28d), which sets
        // the ALU flags.
        assert_eq!(visa_instruction_bytes("r8 = r13"), [0x70, 0xfe, 0x84, 0x3f]);
        assert_eq!(visa_instruction_bytes("r15=r0"), [0x70, 0x3e, 0x07, 0xbf]);
        // An explicit PASS stays a compute.
        assert_eq!(visa_instruction_bytes("r8 = pass r13"), [0xc2, 0x8d]);
        for bad in ["f1 = f2", "r1 = f2", "f1 = r2", "if eq f1 = f2"] {
            let source = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_source(&source, true).is_err(), "{bad}");
        }
    }

    /// Instruction widths (in bits) of section `code` of a VISA object.
    fn visa_widths(object: &[u8]) -> Vec<usize> {
        let code = object_section(object, "code");
        let mut out = Vec::new();
        let mut at = 0;
        while at < code.len() {
            let len = selinstr::visa::instruction_len(&code, at);
            out.push(len * 8);
            at += len;
        }
        out
    }

    #[test]
    fn pc_oracle_forms_match_vendor_bytes() {
        // The reference assembler (the numeric
        // target fields here are unlinked; the reference run used no-auto).
        for (ins, expected) in [
            ("if eq jump (pc, 0) (la)", "074000000000"),
            ("if ge jump (pc, 1) (db,la)", "076204000001"),
            ("cjump (pc, 1) (db)", "184400000001"),
            ("if eq jump (pc, 1), else f12=f0*f1", "09000a130c01"),
            (
                "if not tf jump (pc, 1) (db), else f12=f2-f12",
                "093a0e082c2c",
            ),
            ("if ne jump (pc, 1) (db), f4=f12*f1", "09200c1304c1"),
            ("if tf jump (pc, 1) (db), r12=r12+1", "091a0c029cc0"),
            ("if not sz jump (pc, 0) (db), r0=r1+1", "093004029010"),
            ("if ge jump (pc, 1), else f0=rsqrts f4", "09220a0c5040"),
            ("do (pc, 1) until lce (f)", "0e1e00800001"),
        ] {
            let source = format!(".section/SW code;\n{ins};\n{}", "nop;\n".repeat(6));
            let object = assemble_source_with(
                &source,
                true,
                AsmOptions {
                    auto_fast_loops: false,
                    ..AsmOptions::default()
                },
            )
            .unwrap();
            let code = object_section(&object, "code");
            assert_eq!(
                code[..6]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
                expected,
                "{ins}"
            );
            // Round-trip the modifiers too; no auto-loop/layout context is
            // needed to reconstruct the raw field printed by seldump.
            let decoded = selinstr::disasm::decode_instruction(read_word48(&code[..6]));
            let parsed = crate::asmparse::parse_instruction(&decoded).unwrap();
            assert_eq!(
                selinstr::encode::encode(&parsed).unwrap(),
                code[..6],
                "{decoded}"
            );
        }
    }

    #[test]
    fn do_label_difference_matches_vendor_stack_relocations() {
        // 0c0010800000; the reference object has
        // (e0,_e), (e0,_t), (e3,.__operator), (1b,.__operator) at word 0.
        let source = format!(
            ".section/pm seg_pmco;\n.extern _ext;\n.global _t;\n_t:\n\
             lcntr = 0x10, do (pc, _e - _t) until lce;\n\
             nop;\nnop;\n_e: nop;\n{}",
            "nop;\n".repeat(40)
        );
        let object = assemble_source_with(
            &source,
            true,
            AsmOptions {
                nwc: true,
                ..AsmOptions::default()
            },
        )
        .unwrap();
        let code = object_section(&object, "seg_pmco");
        assert_eq!(code.len(), 264);
        assert_eq!(code[..6], [0x0c, 0, 0x10, 0x80, 0, 0]);
        assert!(code[6..].iter().all(|b| *b == 0));
        assert_eq!(object_symbols(&object)["_e"], 3);
        assert_eq!(object_symbols(&object)["_t"], 0);
        assert_eq!(
            reloc_tuples(&object, ".rela.seg_pmco"),
            vec![
                (0, 0xe0, "_e".into(), 0),
                (0, 0xe0, "_t".into(), 0),
                (0, 0xe3, ".__operator".into(), 0),
                (0, 0x1b, ".__operator".into(), 0),
            ]
        );
        let shifted = source.replace("_t:\n", "nop;\n_t:\n");
        assert!(assemble_source_with(
            &shifted,
            true,
            AsmOptions {
                nwc: true,
                ..AsmOptions::default()
            }
        )
        .is_err());
    }

    #[test]
    fn nwc_numeric_targets_match_vendor_words_and_relocations() {
        let options = AsmOptions {
            nwc: true,
            ..AsmOptions::default()
        };
        // -nwc: DO keeps 0x1e, its generated
        // symbol is word 30, and all 40 following NOPs occupy six bytes.
        let source = format!(
            ".section/pm seg_pmco;\n.extern _ext;\n.global _t;\n_t:\n\
             lcntr = 0x10, do (pc, 0x1e) until lce;\n{}",
            "nop;\n".repeat(40)
        );
        for (auto_fast_loops, fast_byte) in [(true, 0x80), (false, 0)] {
            // Same vector with -no-auto-fast-loops.
            let object = assemble_source_with(
                &source,
                true,
                AsmOptions {
                    auto_fast_loops,
                    ..options
                },
            )
            .unwrap();
            let code = object_section(&object, "seg_pmco");
            assert_eq!(code.len(), 246);
            assert_eq!(code[..6], [0x0c, 0, 0x10, fast_byte, 0, 0x1e]);
            assert!(code[6..].iter().all(|b| *b == 0));
            assert_eq!(object_symbols(&object)["___ADI_AGL_DO_5"], 30);
            let relas = object_relas(&object, ".rela.seg_pmco");
            assert_eq!(relas.len(), 1);
            assert_eq!((relas[0].0, relas[0].1 & 0xff), (0, 0x1b));
        }
        // A numeric JUMP also keeps its word offset,
        // with R_SHARC_PM_PCREL24 against ___ADI_AGL_JUMP_4 at word 4.
        let source = format!(
            ".section/pm seg_pmco;\n.global _t;\n_t:\n\
             if ge jump (pc, 0x4);\n{}",
            "nop;\n".repeat(40)
        );
        let object = assemble_source_with(&source, true, options).unwrap();
        assert_eq!(
            object_section(&object, "seg_pmco")[..6],
            [7, 0x22, 0, 0, 0, 4]
        );
        assert_eq!(object_symbols(&object)["___ADI_AGL_JUMP_4"], 4);
        let relas = object_relas(&object, ".rela.seg_pmco");
        assert_eq!(relas.len(), 1);
        assert_eq!((relas[0].0, relas[0].1 & 0xff), (0, 0xf));
        // The DO after two NW NOPs relocates at
        // word 2, not parcel 6 or byte 12; its backward target is word 0.
        let object = assemble_source_with(
            ".section/pm seg_pmco;\n_b: nop;\nnop;\n\
             lcntr = 0x10, do _b until lce;\nnop;\n",
            true,
            options,
        )
        .unwrap();
        assert_eq!(
            object_section(&object, "seg_pmco")[12..18],
            [0xc, 0, 0x10, 0, 0, 0]
        );
        assert_eq!(object_symbols(&object)["_b"], 0);
        let relas = object_relas(&object, ".rela.seg_pmco");
        assert_eq!((relas[0].0, relas[0].1 & 0xff), (2, 0x1b));
    }

    #[test]
    fn non_counter_loop_fast_bit_matches_vendor() {
        // 0e1e0080001e; the same DO with
        // -no-auto-fast-loops = 0e1e0000001e.
        let source = format!(
            ".section/pm code;\n.extern _ext;\n.global _t;\n_t:\n\
             do (pc, 0x1e) until lce;\n{}",
            "nop;\n".repeat(40)
        );
        for (auto_fast_loops, fast_byte) in [(true, 0x80), (false, 0)] {
            let object = assemble_source_with(
                &source,
                true,
                AsmOptions {
                    nwc: true,
                    auto_fast_loops,
                    ..AsmOptions::default()
                },
            )
            .unwrap();
            assert_eq!(
                object_section(&object, "code")[..6],
                [0xe, 0x1e, 0, fast_byte, 0, 0x1e]
            );
            assert_eq!(object_symbols(&object)["___ADI_AGL_DO_5"], 30);
            let relas = object_relas(&object, ".rela.code");
            assert_eq!((relas[0].0, relas[0].1 & 0xff), (0, 0x1b));
        }
        // A condition-terminated DO has no fast bit.
        let object = assemble_source_with(
            &source.replace("until lce", "until eq"),
            true,
            AsmOptions {
                nwc: true,
                ..AsmOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            object_section(&object, "code")[..6],
            [0xe, 0, 0, 0, 0, 0x1e]
        );
    }

    #[test]
    fn nwc_tail_padding_and_section_modes() {
        let options = AsmOptions {
            nwc: true,
            ..AsmOptions::default()
        };
        // Four instructions follow
        // the conditional JUMP; the reference inserts two NW NOPs before .end.
        // This excerpt starts at the vendor's word 12, so .end=8 here.
        let source = ".section/pm code;\n.global .LN.fill..end;\n\
            btst r8 by 0;\nif sz jump (pc,.P33L2);\ndm(i4,m5)=r12;\n\
            .P33L2: jump (m14,i12) (db);\nrframe;\nnop;\n.LN.fill..end:\n";
        let object = assemble_source_with(source, true, options).unwrap();
        let code = object_section(&object, "code");
        let hex: String = code.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            concat!(
                "023e00330008",
                "071000000000",
                "597e86000000",
                "083f34000000",
                "190000000000",
                "000000000000",
                "000000000000",
                "000000000000"
            )
        );
        assert_eq!(object_symbols(&object)[".LN.fill..end"], 8);
        let relas = object_relas(&object, ".rela.code");
        assert_eq!((relas[0].0, relas[0].1 & 0xff), (1, 0xf));

        // An explicit /SW overrides -nwc.
        // .COMPRESS only lifts instruction compression suppression; it
        // cannot turn an NW section into a parcel-addressed section.
        let source = "nop;\n.section/pm pm;\n.nocompress;\nnop;\n.compress;\nnop;\n\
            .section/code code;\nnop;\n.section/sw sw;\nnop;\n\
            .section/code/nw nw;\nnop;\n.section/double32 data;\n.var x=0x12345678;\n";
        let object = assemble_source_with(source, true, options).unwrap();
        assert_eq!(object_section(&object, ".text"), [0; 6]);
        assert_eq!(object_section(&object, "pm"), [0; 12]);
        assert_eq!(object_section(&object, "code"), [0; 6]);
        assert_eq!(object_section(&object, "sw"), [0, 1]);
        assert_eq!(object_section(&object, "nw"), [0; 6]);
        assert_eq!(object_section(&object, "data"), [0x78, 0x56, 0x34, 0x12]);
        let attrs = object_section(&object, ".adi.attributes");
        for (idx, encoding) in [(1, 2), (2, 2), (3, 2), (4, 3), (5, 2)] {
            assert!(attrs
                .windows(9)
                .any(|w| w == [2, 9, 0, 0, 0, idx, 0, 0x12, encoding]));
        }
        // Inherited NW .align semantics: whole 48-bit NOP words, symbols
        // in words. No vendor NW code .align vector is available.
        let object = assemble_source_with(
            ".section/pm code;\n.global aligned;\nnop;\n.align 4;\naligned: nop;\n",
            true,
            options,
        )
        .unwrap();
        assert_eq!(object_section(&object, "code"), [0; 30]);
        assert_eq!(object_symbols(&object)["aligned"], 4);
    }

    #[test]
    fn loop_bodies_keep_only_their_last_11_instructions_48_bit() {
        // The reference assembler keeps exactly the last 11 body instructions of a counter
        // loop in 48-bit form (all 229 loops in the compressed listings).
        // The body has no branch, so the fast bit is set automatically
        // (`-auto-fast-loops`).
        let body = vec!["r0 = r0 + r1;"; 12].join("\n");
        let source = format!(
            ".section/SW/DOUBLE32 code;\nlcntr = 4, do end until lce;\n{body}\nend: r0 = r0 + r1;\nnop;\n"
        );
        let object = assemble_preprocessed(&source).unwrap();
        let mut expected = vec![48, 16, 16];
        expected.extend([48; 11]);
        expected.push(16);
        assert_eq!(visa_widths(&object), expected);
        let code = object_section(&object, "code");
        assert_eq!(code[3] & 0x80, 0x80, "fast bit");
        // `-no-auto-fast-loops` leaves the bit to an explicit (F).
        let options = AsmOptions {
            auto_fast_loops: false,
            ..AsmOptions::default()
        };
        let plain = assemble_source_with(&source, true, options).unwrap();
        assert_eq!(object_section(&plain, "code")[3] & 0x80, 0);
    }

    #[test]
    fn automatic_fast_loop_checks_only_the_last_11_instructions() {
        // Four reference vectors: bodies of 12, 20, 30 and 40 instructions.
        // All four bodies branch at instruction 5 to instruction 9.
        // The 12 one has a 12-instruction body and emits 0c0004000000; the
        // 20/30/40-instruction bodies emit 0c0004800000. Only the branch
        // inside the last 11 instructions prevents automatic fast mode.
        // The exact window is also documented in the SHARC+ Core
        // Programming Reference rev. 1.5, p. 4-37; these retained vectors
        // do not independently test both sides of that boundary.
        for count in [12, 20, 30, 40] {
            let mut source = String::from(
                ".section/SW/DOUBLE32 seg_l2code;\n.extern _ext;\n_t:\n\
                 lcntr = 4, do _e until lce;\n",
            );
            let mut expected = vec![0x0c, 0, 4, if count == 12 { 0 } else { 0x80 }, 0, 0];
            for i in 0..count {
                if i == 4 {
                    source.push_str("if ne jump _x;\n");
                    expected.extend([0x06, 0x20, 0, 0, 0, 0]);
                } else {
                    if i == 8 {
                        source.push_str("_x: ");
                    } else if i == count - 1 {
                        source.push_str("_e: ");
                    }
                    source.push_str("r1 = r1 + r2;\n");
                    if i >= count - 11 {
                        expected.extend([0x01, 0x3e, 0, 0, 0x11, 0x12]);
                    } else {
                        expected.extend([0xc0, 0x12]);
                    }
                }
            }
            source.push_str("r3 = r3 + r4;\nrts;\nnop;\nnop;\n");
            expected.extend([0xc0, 0x34, 0x0a, 0xbe, 0, 1, 0, 1]);
            let object = assemble_preprocessed(&source).unwrap();
            assert_eq!(object_section(&object, "seg_l2code"), expected, "L{count}m");
            // The command-line opt-out changes only the automatic bit;
            // width planning remains the same. The reference
            // establishes -no-auto-fast-loops for compressed code.
            let no_auto = assemble_source_with(
                &source,
                true,
                AsmOptions {
                    auto_fast_loops: false,
                    ..AsmOptions::default()
                },
            )
            .unwrap();
            expected[3] = 0;
            assert_eq!(object_section(&no_auto, "seg_l2code"), expected);
        }
    }

    #[test]
    fn vendor_short_loop_control_flow_keeps_fast_bit_clear() {
        // Negative vectors (jump, conditional jump, call and return in the body):
        // lcntr=0x10, do _e until lce -> 0c0010000000. Every control
        // flow instruction is in the three-instruction loop tail.
        for instr in ["jump _x;", "if ne jump _x;", "call _ext;", "rts;"] {
            let source = format!(
                ".section/pm seg_pmco;\n.extern _ext;\n.global _t;\n_t:\n\
                 lcntr = 0x10, do _e until lce;\n{instr}\nnop;\n_e: nop;\n_x: nop;\n{}",
                "nop;\n".repeat(40)
            );
            let object = assemble_source_with(
                &source,
                true,
                AsmOptions {
                    nwc: true,
                    ..AsmOptions::default()
                },
            )
            .unwrap();
            assert_eq!(
                object_section(&object, "seg_pmco")[..6],
                [0x0c, 0, 0x10, 0, 0, 0],
                "{instr}"
            );
        }
    }

    #[test]
    fn nested_loop_keeps_the_outer_tail_48_bit() {
        // A nested loop: the outer tail spans the inner
        // DO and its body, and the outer loop's last instruction stays
        // 48-bit. Upstream tracked one end label, lost the outer one at the
        // inner loop and emitted that instruction in 16 bits.
        let source = ".section/SW/DOUBLE32 code;\n\
                      lcntr = 0x28, do outer until lce;\n\
                      r1 = r1 + r2;\nr1 = r1 + r2;\nr1 = r1 + r2;\n\
                      r1 = r1 + r2;\nr1 = r1 + r2;\nr1 = r1 + r2;\n\
                      lcntr = 8, do inner until lce;\n\
                      r0 = r0 + r1;\nr0 = r0 + r1;\nr0 = r0 + r1;\n\
                      inner: r0 = r0 + r1;\n\
                      outer: r15 = r15 + r14;\n";
        let object = assemble_preprocessed(source).unwrap();
        let widths = visa_widths(&object);
        assert_eq!(widths, [48, 16, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]);
    }

    #[test]
    fn numeric_pc_offsets_count_instructions_in_compressed_code() {
        // The reference assembler reads `(pc, n)` as "n instructions away" and labels
        // that instruction (___ADI_AGL_JUMP_4 / ___ADI_AGL_DO_4).
        // `if ge jump (pc, 0x4)` targets parcel 6.
        let nops = vec!["nop;"; 30].join("\n");
        let source = format!(".section/SW/DOUBLE32 code;\nif ge jump (pc, 0x4);\n{nops}\n");
        let code = object_section(&assemble_preprocessed(&source).unwrap(), "code");
        assert_eq!(code[..6], [0x07, 0x22, 0x00, 0x00, 0x00, 0x04]);
        let object = assemble_preprocessed(&source).unwrap();
        assert_eq!(object_symbols(&object)["___ADI_AGL_JUMP_2"], 6);
        assert_eq!(
            reloc_tuples(&object, ".rela.code"),
            vec![(0, 0x0f, "___ADI_AGL_JUMP_2".into(), 0)]
        );
        // The zero-offset case is the exception.
        let zero = assemble_preprocessed(".section/SW code;\njump (pc,0);\n").unwrap();
        assert_eq!(object_section(&zero, "code"), [7, 0x3e, 0, 0, 0, 0]);
        let hdr = selelf::elf::parse_header(&zero).unwrap();
        assert!(find_section_by_name(&zero, &hdr, ".rela.code").is_none());
        // Raw offsets stay available for re-assembling disassembly.
        let raw = AsmOptions {
            raw_pc_offsets: true,
            ..AsmOptions::default()
        };
        let code = object_section(&assemble_source_with(&source, true, raw).unwrap(), "code");
        assert_eq!(code[..6], [0x07, 0x22, 0x00, 0x00, 0x00, 0x04]);
        // (-no-auto-fast-loops): `do (pc, 0x12)` ends on the 18th
        // body instruction, 7 compressed NOPs plus the 11-instruction
        // 48-bit tail, at parcel 40 (0x28).
        let source =
            format!(".section/SW/DOUBLE32 code;\nlcntr = 0x10, do (pc, 0x12) until lce;\n{nops}\n");
        let no_auto = AsmOptions {
            auto_fast_loops: false,
            ..AsmOptions::default()
        };
        // The reference assembler keeps n in the field and relocates against
        // `___ADI_AGL_DO_<line>` at the end (0c001080001e,
        // the symbol at parcel 52); the linker writes the parcel delta.
        let object = assemble_source_with(&source, true, no_auto).unwrap();
        let code = object_section(&object, "code");
        assert_eq!(code[..6], [0x0c, 0x00, 0x10, 0x00, 0x00, 0x12]);
        assert_eq!(object_symbols(&object)["___ADI_AGL_DO_2"], 0x28);
        let relas = object_relas(&object, ".rela.code");
        assert_eq!(
            (relas[0].0, relas[0].1 & 0xff),
            (0, selelf::elf::R_SHARC_PM_LOOP16)
        );
        let mut widths = vec![48];
        widths.extend([16; 7]);
        widths.extend([48; 11]);
        widths.extend([16; 12]);
        assert_eq!(visa_widths(&object), widths);
        // A target past the section end is an error, and `(pc, n)` in a
        // condition-terminated DO is an offset, not a symbol named
        // "pc, 0x800".
        for bad in [
            ".section/SW/DOUBLE32 code;\nif ge jump (pc, 0x40);\nnop;\n",
            ".section/SW/DOUBLE32 code;\ndo (pc, 0x800) until lce;\nnop;\n",
        ] {
            assert!(assemble_preprocessed(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn backward_loop_end_is_a_relocation() {
        // Vendor behaviour: `do _b until lce` with `_b`
        // before the DO used to abort selas in pass 2, then was an error.
        // The reference assembler assembles it like any DO label: 0c0010000000 (no fast
        // bit) + R_SHARC_PM_LOOP16 against `_b` at the DO.
        let source = ".section/SW/DOUBLE32 code;\n_b: nop;\nnop;\n\
                      lcntr = 0x10, do _b until lce;\nnop;\n";
        let object = assemble_preprocessed(source).unwrap();
        assert_eq!(
            object_section(&object, "code")[4..10],
            [0x0c, 0x00, 0x10, 0x00, 0x00, 0x00]
        );
        let relas = object_relas(&object, ".rela.code");
        assert_eq!(relas.len(), 1);
        assert_eq!(
            (relas[0].0, relas[0].1 & 0xff),
            (2, selelf::elf::R_SHARC_PM_LOOP16)
        );
        assert_eq!(object_symbols(&object)["_b"], 0);
    }

    #[test]
    fn branches_clear_the_automatic_fast_bit() {
        // The reference assembler leaves bit 23 clear.
        let source = ".section/SW/DOUBLE32 code;\n\
                      lcntr = 4, do end until lce;\n\
                      if eq jump skip;\nr0 = r0 + r1;\nskip: r0 = r0 + r1;\n\
                      end: nop;\n";
        let object = assemble_preprocessed(source).unwrap();
        assert_eq!(object_section(&object, "code")[3] & 0x80, 0);
        // `DO end UNTIL cond` has no compressed vendor vector: its whole
        // body stays 48-bit.
        let body = vec!["r0 = r0 + r1;"; 13].join("\n");
        let source = format!(".section/SW/DOUBLE32 code;\ndo end until eq;\n{body}\nend: nop;\n");
        let object = assemble_preprocessed(&source).unwrap();
        assert!(visa_widths(&object).iter().all(|&w| w == 48));
    }

    #[test]
    fn symbolic_loads_keep_their_constant_as_the_addend() {
        // `r2 = _mod_params + 4` and `_second_params + 48/96`
        // lost their constant:
        // relocations carried addend 0. The reference assembler keeps 0
        // in the field and the constant in the relocation expression.
        let source = ".section/SW/DOUBLE32 code;\n_x: nop;\n\
                      r0 = _x - 1;\nr2 = _mod_params + 4;\n\
                      r3 = _second_params + 3 * 16 * 0 + 96;\ni7 = ldf_a + ldf_b;\n";
        let object = assemble_preprocessed(source).unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        let syms: Vec<String> = {
            let hdr = &header;
            let sections: Vec<_> = (0..hdr.e_shnum as usize)
                .map(|i| {
                    let off = hdr.e_shoff as usize + i * hdr.e_shentsize as usize;
                    selelf::elf::parse_section_header(&object[off..], hdr.ei_data)
                })
                .collect();
            let symtab = sections
                .iter()
                .find(|s| s.sh_type == selelf::elf::SHT_SYMTAB)
                .unwrap();
            let strtab = &sections[symtab.sh_link as usize];
            let names =
                &object[strtab.sh_offset as usize..(strtab.sh_offset + strtab.sh_size) as usize];
            (0..(symtab.sh_size / symtab.sh_entsize) as usize)
                .map(|i| {
                    let off = symtab.sh_offset as usize + i * symtab.sh_entsize as usize;
                    let sym = selelf::elf::parse_symbol(&object[off..], hdr.ei_data);
                    selelf::elf::read_string_at(names, sym.st_name).to_string()
                })
                .collect()
        };
        let data = object_section(&object, ".rela.code");
        let relas: Vec<(String, i32)> = data
            .chunks(12)
            .map(|e| {
                let r = selelf::elf::parse_rela(e, header.ei_data);
                (syms[(r.r_info >> 8) as usize].clone(), r.r_addend)
            })
            .collect();
        assert_eq!(
            relas,
            [
                ("_x".to_string(), -1),
                ("_mod_params".to_string(), 4),
                ("_second_params".to_string(), 96),
                ("ldf_a".to_string(), 0),
                ("ldf_b".to_string(), 0),
            ]
        );
        // The field itself stays 0, as the reference assembler leaves it.
        let code = object_section(&object, "code");
        assert_eq!(code[2..8], [0x0f, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // A label difference or an unknown operation is an error, not a
        // load of the first identifier.
        for bad in ["r1 = (_y - _x)", "r2 = b2w(i7)", "r3 = -_x"] {
            let source = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_preprocessed(&source).is_err(), "{bad}");
        }
    }

    #[test]
    fn float_multifunctions_match_vendor_vectors() {
        // Dropped operations. Reference vectors: float
        // multiply + ALU opcodes 0x18 +, 0x19 -, 0x1a FLOAT BY, 0x1c
        // (a+b)/2, 0x1d ABS, 0x1e MAX, 0x1f MIN; selas encoded (a+b)/2 as
        // 0x1a and dropped MAX/MIN/ABS next to a memory access.
        for (source, word) in [
            (
                "f0 = f3 * f4 , f8 = max ( f8 , f14 ) , dm ( -0x9 , i6 ) = r1",
                &[0x6c, 0xbf, 0xb8, 0xde, 0x08, 0xc2][..],
            ),
            (
                "f0 = f1 * f7 , f12 = abs f10 , dm ( i5 , 0x2 ) = r2",
                &[0x6a, 0xfe, 0x11, 0x5d, 0x0c, 0x78],
            ),
            (
                "f10 = f2 * f7 , f2 = min ( f11 , f12 ) , r15 = dm ( 0x2 , i6 )",
                &[0x6c, 0x3e, 0x17, 0xdf, 0xa2, 0xbc],
            ),
            (
                "f4 = f0 * f4 , f0 = ( f10 + f12 ) / 2 , r12 = i13",
                &[0x71, 0xfe, 0x86, 0x5c, 0x40, 0x08],
            ),
            (
                "f0 = f2 * f4 , f1 = max ( f8 , f15 )",
                &[0x01, 0xde, 0x01, 0x83],
            ),
            // Used to drop the subtract and the register swap.
            (
                "f0 = f0 * f4 , f12 = f8 - f12 , r5 <-> s5",
                &[0x79, 0x7e, 0x02, 0xd9, 0x0c, 0x00],
            ),
            (
                "f0 = f1 * f4 , f13 = float r11 by r14",
                &[0x01, 0xda, 0x0d, 0x4e],
            ),
            // ALU dual add/subtract (no multiply); it used to add F0 = F0 * F4.
            ("f12 = f2 + f6 , f8 = f2 - f6", &[0x01, 0x8f, 0x8c, 0x26]),
            (
                "f0 = f14 + f0 , f2 = f14 - f0 , dm ( -0x5 , i6 ) = r15",
                &[0x6c, 0xbf, 0xdf, 0x8f, 0x20, 0xe0],
            ),
        ] {
            assert_eq!(visa_instruction_bytes(source), word, "{source}");
        }
    }

    #[test]
    fn indirect_branches_match_vendor_vectors() {
        // The reference assembler. Before:
        // the compute and ELSE were dropped, `( db )` with spaces was lost,
        // the 32-bit CALL bit sat at p1[6] (`call` became 087f303f), and the
        // 48-bit layout put I at 26:24 and (DB) at 32.
        for (source, word) in [
            (
                "jump ( m12 , i12 ) , r4 = r1 + r2",
                &[0x08, 0x3f, 0x20, 0x00, 0x14, 0x12][..],
            ),
            (
                "if eq jump ( m14 , i8 ) , else f1 = abs f8",
                &[0x08, 0x00, 0x32, 0x0b, 0x01, 0x80],
            ),
            (
                "if eq jump ( m14 , i8 ) , f0 = f0 copysign f4",
                &[0x08, 0x00, 0x30, 0x0e, 0x00, 0x04],
            ),
            ("call (m14, i12)", &[0x08, 0xbf, 0x30, 0x3f]),
            ("if lt jump ( m14 , i8 ) ( db )", &[0x08, 0x02, 0x34, 0x3f]),
            ("if lt jump ( m8 , i12 ) ( db )", &[0x08, 0x03, 0x04, 0x3f]),
            ("jump (m14, i12) (db)", &[0x08, 0x3f, 0x34, 0x3f]),
        ] {
            assert_eq!(visa_instruction_bytes(source), word, "{source}");
        }
        // Vendor-rejected or unpinned forms fail instead of encoding.
        for bad in [
            "call ( m14 , i4 )",
            "jump (m14, i12) (db), else r0 = r1 + r2",
            "jump (m14, i12), else r0 = r1 + r2",
            "jump (m14, i12) (la)",
        ] {
            let source = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_source(&source, true).is_err(), "{bad}");
        }
    }

    #[test]
    fn modify_by_i32_min_matches_vendor_vector() {
        // Reference vector: `modify(i8, 0x80000000)` = 164080000000. The
        // disassembler negated i32::MIN while printing it and panicked.
        assert_eq!(
            visa_instruction_bytes("modify(i8, 0x80000000)"),
            [0x16, 0x40, 0x80, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            visa_instruction_bytes("bitrev ( i0 , -0x80000000 )"),
            [0x16, 0x80, 0x80, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn type19_bitrev_and_dag_select_match_vendor_vectors() {
        // The reference assembler: bit 39 is
        // BITREV and bit 38 the DAG select; selas had them swapped.
        for (source, word) in [
            ("bitrev (i0, 0x10)", [0x16, 0x80, 0x00, 0x00, 0x00, 0x10]),
            (
                "i10 = modify(i10, 0x4)",
                [0x16, 0x42, 0x00, 0x00, 0x00, 0x04],
            ),
            (
                "i14 = modify (i14, 0x8)",
                [0x16, 0x46, 0x00, 0x00, 0x00, 0x08],
            ),
            (
                "i0 = modify (i0, 0x1)",
                [0x16, 0x00, 0x00, 0x00, 0x00, 0x01],
            ),
        ] {
            assert_eq!(visa_instruction_bytes(source), word, "{source}");
        }
    }

    #[test]
    fn system_registers_use_the_vendor_codes() {
        // The reference assembler: one code
        // table serves every form. selas numbered groups 6/7 itself and put
        // USTATx/TPERIOD/TCOUNT at 0x80-0x91, so 48-bit forms and the
        // USTAT/timer transfers went to the wrong register.
        for (source, word) in [
            ("pcstk = 0x100000", vec![0x0f, 0x64, 0x00, 0x10, 0x00, 0x00]),
            ("px = 0x100000", vec![0x0f, 0x6b, 0x00, 0x10, 0x00, 0x00]),
            ("mmask = 0x100000", vec![0x0f, 0x73, 0x00, 0x10, 0x00, 0x00]),
            ("irptl = 0x100000", vec![0x0f, 0x7a, 0x00, 0x10, 0x00, 0x00]),
            ("mmask = 0x0", vec![0x0f, 0xf3, 0x00, 0x00]),
            ("tperiod = r0", vec![0x70, 0x3e, 0x37, 0x3f]),
            ("tcount = r0", vec![0x70, 0x3e, 0x37, 0xbf]),
            ("ustat1 = r0", vec![0x70, 0x3e, 0x38, 0x3f]),
            ("ustat4 = r0", vec![0x70, 0x3e, 0x3f, 0xbf]),
            ("mode1 = r0", vec![0x70, 0x3e, 0x39, 0x3f]),
            ("pcstkp = r0", vec![0x70, 0x3e, 0x32, 0xbf]),
            ("r0 = ustat3", vec![0x77, 0xff, 0x00, 0x3f]),
            (
                "astaty = dm(-0x100, i6)",
                vec![0xac, 0x77, 0xff, 0xff, 0xff, 0x00],
            ),
            (
                "bit tst ustat1 0x0",
                vec![0x14, 0x80, 0x00, 0x00, 0x00, 0x00],
            ),
        ] {
            assert_eq!(visa_instruction_bytes(source), word, "{source}");
        }
    }

    #[test]
    fn multiplier_modes_match_vendor_vectors() {
        // The reference assembler. selas
        // encoded every suffix it had no variant for as SSF, MRB (SSI) as
        // MRB (SSF), and ignored the suffix of MR accumulates.
        for (source, word) in [
            ("r0 = r0 * r1 (uui)", vec![0x01, 0x94, 0x00, 0x01]),
            ("r0 = r0 * r1 (uuf)", vec![0x01, 0x94, 0x80, 0x01]),
            ("r0 = r0 * r1 (sui)", vec![0x01, 0x95, 0x00, 0x01]),
            ("r0 = r0 * r1 (suf)", vec![0x01, 0x95, 0x80, 0x01]),
            ("r0 = r0 * r1 (usi)", vec![0x01, 0x96, 0x00, 0x01]),
            ("r0 = r0 * r1 (usf)", vec![0x01, 0x96, 0x80, 0x01]),
            ("r0 = r2 * r1 ( ssi )", vec![0x01, 0x97, 0x00, 0x21]),
            ("mrb = r0 * r0 (ssi)", vec![0x01, 0x97, 0x60, 0x00]),
            ("mrf = mrf + r0 * r0 (ssi)", vec![0x01, 0x9b, 0x40, 0x00]),
            ("mrb = mrb + r0 * r0 (ssi)", vec![0x01, 0x9b, 0x60, 0x00]),
            ("mrf = mrf - r0 * r0 (ssi)", vec![0x01, 0x9f, 0x40, 0x00]),
            ("mrb = mrb - r0 * r0 (ssi)", vec![0x01, 0x9f, 0x60, 0x00]),
            ("r0 = mrf + r2 * r15 ( ssi )", vec![0x01, 0x9b, 0x00, 0x2f]),
            (
                "r1 = r2 * r0 ( uui ) , r4 = r14",
                vec![0x70, 0xff, 0x02, 0x14, 0x01, 0x20],
            ),
        ] {
            assert_eq!(visa_instruction_bytes(source), word, "{source}");
        }
        for bad in [
            "r0 = r0 * r1 (ssir)",
            "r0 = r0 * r1 (xyz)",
            "mrf = mrb + r0 * r1 (ssf)",
        ] {
            let src = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_source(&src, true).is_err(), "{bad}");
        }
    }

    #[test]
    fn shift_memory_bit_and_modify_forms_match_vendor_vectors() {
        // The reference assembler.
        for (source, word) in [
            // Type 6a sign-extends a negative count into bits 30:27.
            (
                "r0 = ashift r12 by -0x10 , dm ( i5 , m0 ) = r11",
                vec![0x8a, 0x3e, 0xfd, 0x81, 0xf0, 0x0c],
            ),
            (
                "r13 = r13 or lshift r15 by -0x1f , r7 = dm ( i1 , m3 )",
                vec![0x82, 0xfe, 0x7b, 0x88, 0xe1, 0xdf],
            ),
            (
                "r0 = lshift r10 by 0xffffffff , dm ( i7 , m7 ) = r2",
                vec![0x8f, 0xfe, 0xf9, 0x00, 0xff, 0x0a],
            ),
            // a spaced `( nw )` is still the NW form.
            (
                "modify ( i0 , 0x7 ) ( nw )",
                vec![0x15, 0x80, 0, 0, 0, 0x07],
            ),
            (
                "i7 = modify ( i7 , 0x200 ) ( nw )",
                vec![0x15, 0x87, 0, 0, 0x02, 0x00],
            ),
            (
                "i6 = modify ( i0 , -0x1 ) ( nw )",
                vec![0x15, 0xb0, 0xff, 0xff, 0xff, 0xff],
            ),
            (
                "modify(i0,-1) (nw)",
                vec![0x15, 0x80, 0xff, 0xff, 0xff, 0xff],
            ),
            // A 32-bit field wraps any magnitude below 2^32.
            (
                "astatx = dm ( -0x80000001 , i6 )",
                vec![0xac, 0x76, 0x7f, 0xff, 0xff, 0xff],
            ),
            // BOP is bits 39:37; TGL was encoded as XOR.
            ("bit tgl mode1 0x0", vec![0x14, 0x42, 0, 0, 0, 0]),
            ("bit xor mode1 0x0", vec![0x14, 0xa2, 0, 0, 0, 0]),
            ("bit tst astatx 0x0", vec![0x14, 0x86, 0, 0, 0, 0]),
            (
                "bit clr imask -0x1f0608",
                vec![0x14, 0x2b, 0xff, 0xe0, 0xf9, 0xf8],
            ),
        ] {
            assert_eq!(visa_instruction_bytes(source), word, "{source}");
        }
    }

    #[test]
    fn plain_pm_dag_forms_match_vendor_vectors_and_decode_back() {
        // The reference assembler. Each
        // vendor word must also disassemble to its source (the decoder
        // misframed 48-bit Type 16 and read the PM bit of the 32-bit
        // Type 15b/16b forms as DM).
        for (source, word, text) in [
            // ABS (Fx +/- Fy) encoded ABS Fx.
            ("f0 = abs(f0 - f1)", "01892001", "f0=abs(f0-f1)"),
            ("f0 = abs(f0 + f1)", "01891001", "f0=abs(f0+f1)"),
            // the third part of mul + add was assumed to be the
            // subtraction.
            (
                "f0 = f1 * f4 , f12 = f11 + f12 , r14 = r13",
                "70fe87580c4c",
                "f0=f1*f4 , f12=f11+f12 , r14=r13",
            ),
            // bit FIFO forms were symbol loads.
            ("r0 = bitext 0x20", "023e00142000", "r0=bitext 0x20"),
            ("bitdep r2 by 0x800", "023e401d0002", "bitdep r2 by 0x800"),
            ("bffwrp = 0x10", "023e001f1000", "bffwrp=0x10"),
            ("r1 = bffwrp", "01a70100", "r1=bffwrp"),
            ("bffwrp = r1", "01a7c001", "bffwrp=r1"),
            // immediate OR FDEP is 0x19 (selas: 0x11, FDEP).
            (
                "r0 = r0 or fdep r12 by 0x10:0x1",
                "023e0019500c",
                "r0=r0 or fdep r12 by 0x10:0x1",
            ),
            // 32-bit Type 16b and PM Type 15b.
            ("pm ( i10 , m8 ) = 0x0", "94240000", "pm (i10,m8)=0x0"),
            ("dm(i0, m0) = -0x1", "9004ffff", "dm (i0,m0)=-0x1"),
            ("dm ( i7 , m7 ) = 0x1000", "9fc41000", "dm (i7,m7)=0x1000"),
            (
                "dm ( i7 , m7 ) = 0x8000",
                "9fc000008000",
                "dm (i7,m7)=0x8000",
            ),
            ("pm ( 0x0 , i10 ) = r2", "95280100", "pm (0x0,i10)=r2"),
            ("m15 = pm ( 0x0 , i11 )", "96281780", "m15=pm (0x0,i11)"),
            ("i11 = pm ( 0x1 , i8 )", "90280d81", "i11=pm (0x1,i8)"),
            // PM Type 3 (G = bit 32) was refused.
            // (F8 and R8 share the code; the decoder prints R.)
            ("f8 = pm ( i8 , m8 )", "503f043f", "r8=pm (i8,m8)"),
            ("pm(m10, i8) = r2", "40bf813f", "pm (m10,i8)=r2"),
            ("pm(m11, i8) = r4", "40ff823f", "pm (m11,i8)=r4"),
            ("i3 = dm(m0, i0)", "403e09bf", "i3=dm (m0,i0)"),
            // float multiply + dual add/sub is 32-bit Type 2b.
            (
                "f10 = f0 * f4 , f12 = f8 + f12 , f8 = f8 - f12",
                "01f8ac00",
                "f10=f0*f4 , f12=f8+f12 , f8=f8-f12",
            ),
            // RTS/RTI without a compute are 16-bit.
            ("rts", "0abe", "rts"),
            ("rts (db)", "0afe", "rts (db)"),
            ("rti", "0bbe", "rti"),
            ("rti (db)", "0bfe", "rti (db)"),
            ("if eq rts (db)", "0ac0", "if eq rts (db)"),
            ("if ne rti (db)", "0be0", "if ne rti (db)"),
            (
                "rts ( db ) , f0 = f0 + f0",
                "0a3e04081000",
                "rts (db) , f0=f0+f0",
            ),
            // 16-bit op 7 is the SSI multiply; selas used it for an
            // add with carry, which has only the 32-bit form.
            ("r0 = r0 * r1 (ssi)", "c701", "r0=r0*r1 (ssi)"),
            ("r0 = r0 + r1 + ci", "01805001", "r0=r0+r1+ci"),
            // Types 1/3/4/5 without a compute are 32-bit (marker 0x3f).
            ("if ac r0 = m6", "7247003f", "if ac r0=m6"),
            (
                "curlcntr = dm ( i4 , m2 )",
                "58be33bf",
                "curlcntr=dm (i4,m2)",
            ),
            (
                "if eq dm ( m1 , i4 ) = m15",
                "484097bf",
                "if eq dm (m1,i4)=m15",
            ),
            (
                "dm ( i0 , m0 ) = r4 , pm ( i8 , m10 ) = r8",
                "3028143f",
                "dm (i0, m0)=r4, pm (i8, m10)=r8",
            ),
        ] {
            let bytes = visa_instruction_bytes(source);
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, word, "{source}");
            let lines =
                selinstr::visa::disassemble_visa(&bytes, 0, selinstr::disasm::decode_instruction);
            assert_eq!(lines.len(), 1, "{source}");
            let squash = |t: &str| {
                t.to_lowercase()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            assert_eq!(squash(&lines[0].text), squash(text), "{source}");
        }
        // A vendor library interrupt dispatcher ends `RTI (DB);
        // I7 = PM(5, I12);`: the 16-bit return must frame on its own.
        let lines = selinstr::visa::disassemble_visa(
            &[0x0b, 0xfe, 0x98, 0x28, 0x0b, 0x85],
            0,
            selinstr::disasm::decode_instruction,
        );
        let texts: Vec<_> = lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["rti (db)", "i7=pm (0x5,i12)"]);
        for bad in [
            "r0 = fdep r1 by 0x4:0x8",
            "r0 = r0 or fext r1 by 0x4:0x8",
            "r0 = abs (r1 - r2)",
            "r1 = bitext 0x1000",
            "r0 = bitext -0x1",
            "bffwrp = 0x80",
            // An I/M register through its own DAG: only pre-modify loads.
            "i10 = pm(i8, m8)",
            "i1 = dm(i0, m0)",
            "dm(m7, i2) = m4",
            "dm(i5, m4) = i5",
        ] {
            let src = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_source(&src, true).is_err(), "{bad}");
        }
    }

    fn section_hex(source: &str, name: &str) -> String {
        section_bytes(source, name)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    #[test]
    fn cond_jump_tails_match_vendor_padding() {
        // The reference assembler: a section ends with at least six
        // instructions after its last conditional JUMP.
        let nops = |n: usize| "0001".repeat(n);
        let code = |body: &str| format!(".section/SW/DOUBLE32 seg_l2code;\n_t:\n{body}\n");
        // e3_ifjump_end: the conditional JUMP is last, six NOPs.
        assert_eq!(
            section_hex(&code("r1 = r1 + r2;\nif ne jump 0x1c2cb5;"), "seg_l2code"),
            format!("c0120620001c2cb5{}", nops(6))
        );
        // Three instructions after it (an unconditional JUMP
        // among them), three NOPs.
        assert_eq!(
            section_hex(
                &code("if ne jump 0x1c2cb5;\nr1 = r1 + r2;\nr1 = r1 + r2;\njump 0x1c2cb5;"),
                "seg_l2code"
            ),
            format!("0620001c2cb5c012c012063e001c2cb5{}", nops(3))
        );
        // JUMP, RTS, JUMP (DB) and CALL-free ends without
        // a condition are never padded.
        for (body, word) in [
            ("jump 0x1c2cb5;", "063e001c2cb5"),
            ("r1 = r1 + r2;\nrts;", "c0120abe"),
            (
                "r1 = r1 + r2;\njump 0x1c2cb5 (db);\nr1 = r1 + r2;\nr1 = r1 + r2;",
                "c012063e041c2cb5c012c012",
            ),
            ("btst r15 by r12;\njump 0x1c2cb5;", "01acc0fc063e001c2cb5"),
        ] {
            assert_eq!(section_hex(&code(body), "seg_l2code"), word, "{body}");
        }
        // Before a `<name>.end` label, so the label
        // and the next function start after the two NOPs (vendor
        // `.LN.fill..end` = `.fill..end` = `dot.` = 0x2c, 13 past `btst`).
        let source = ".section/SW/DOUBLE32 seg_swco;\n.global .LN.fill..end, .fill..end, dot.;\n\
                      fill.:\nbtst r8 by 0;\n\
                      if sz jump (pc,.P33L2);\n.P33L9:\ndm(i4,m5)=r12;\n.P33L2:\n\
                      jump (m14,i12) (db);\nrframe;\nnop;\n.LN.fill..end:\n.fill..end:\n\
                      dot.:\nmodify(i7,-2) (nw);\n";
        assert_eq!(
            section_hex(source, "seg_swco"),
            format!(
                "023e00330008071000000000997c083f343f1901{}1587fffffffe",
                nops(3)
            )
        );
        let syms = object_symbols(&assemble_source(source, true).unwrap());
        for name in [".LN.fill..end", ".fill..end", "dot."] {
            assert_eq!(syms[name], 13, "{name}");
        }
        // A label that only ends in `_end` is not a function end
        // (`_m2_copy_end`, five instructions after `if ne jump`).
        assert_eq!(
            section_hex(
                &code("if ne jump 0x1c2cb5;\nr1 = r1 + r2;\n_t_end:\nr1 = r1 + r2;\nrts;"),
                "seg_l2code"
            ),
            format!("0620001c2cb5c012c0120abe{}", nops(3))
        );
        // The section end comes after trailing labels (a module's
        // `_mod_code_tail_end` = 0x1cb, the section 0x1d0 parcels).
        let source = code(".global _t_end;\nif ne jump 0x1c2cb5;\n_t_end:");
        let syms = object_symbols(&assemble_source(&source, true).unwrap());
        assert_eq!(syms["_t_end"], 3);
        // A section left and reopened is padded once, at its final end.
        let source = ".section/SW/DOUBLE32 a;\nif ne jump 0x1c2cb5;\n\
                      .section/SW/DOUBLE32 b;\nr1 = r1 + r2;\n\
                      .section/SW/DOUBLE32 a;\nr1 = r1 + r2;\n";
        assert_eq!(
            section_hex(source, "a"),
            format!("0620001c2cb5c012{}", nops(5))
        );
        assert_eq!(section_hex(source, "b"), "c012");
        // 48-bit sections pad with 48-bit NOPs.
        assert_eq!(
            section_hex(
                ".section/CODE/NW/DOUBLE32 iv;\nif ne jump 0x1c2cb5;\n",
                "iv"
            ),
            format!("0620001c2cb5{}", "000000000000".repeat(6))
        );
    }

    #[test]
    fn code_align_fills_with_nops() {
        // The reference assembler `.align 4` in a VISA section counts parcels and
        // fills with NOPs (0x245 -> one 48-bit NOP, 0x33 ->
        // `0001`; a gap of two = `0001 0001`).
        for (lead, fill) in [
            ("", ""),
            ("r1 = r1 + r2;\n", "000000000000"),
            ("btst r15 by r12;\n", "00010001"),
            ("r1 = r1 + r2;\nbtst r15 by r12;\n", "0001"),
        ] {
            let source =
                format!(".section/SW/DOUBLE32 c;\n.global _x;\n{lead}.align 4;\n_x:\nrts;\n");
            let hex = section_hex(&source, "c");
            assert!(hex.ends_with(&format!("{fill}0abe")), "{lead:?}: {hex}");
            let syms = object_symbols(&assemble_source(&source, true).unwrap());
            assert_eq!(syms["_x"] % 4, 0, "{lead:?}");
            assert_eq!(hex.len() / 4, syms["_x"] as usize + 1, "{lead:?}");
        }
        // Inside a DO body's 48-bit tail the fill could not be forced wide
        // refused.
        let looped = ".section/SW/DOUBLE32 c;\nlcntr = 4, do end until lce;\n.align 4;\n\
                      end: r0 = r0 + r1;\n";
        assert!(assemble_source(looped, true).is_err());
    }

    #[test]
    fn branch_targets_match_vendor_vectors() {
        // Numeric targets are a
        // signed 24-bit field; CJUMP needs (DB), written with or without
        // spaces.
        for (source, word) in [
            ("if eq jump -0x436fa5", "060000bc905b"),
            ("call 0x100000", "06be00100000"),
            ("cjump -0x436e00 ( db )", "180400bc9200"),
        ] {
            let bytes = visa_instruction_bytes(source);
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, word, "{source}");
        }
        for bad in [
            "call 0x800000",
            "if eq call 0xb8d588",
            "jump -0xc76fa5",
            "cjump 0x0",
            // IF LCE assembled as IF NOT LCE.
            "if lce r0 = r1 + r2",
            "if lce jump (pc, 0x1)",
            // 32-bit values, the 16-bit loop count, unresolved BIT data.
            "i0 = 0x100000000",
            "b0 = -0x100000001",
            "dm ( i7 , m7 ) = 0x100000000",
            "dm ( -0x100000000 ) = r0",
            "lcntr = 0x10000 , do _pf_lp until lce",
            "bit clr mode1 0xzz",
            // MODIFY across DAGs (Type 7 kept M's low three bits).
            "modify(i0, m10)",
            "r10 = r10 + r14 , i5 = modify ( i5 , m12 )",
            // G5: Type 15 stores of the access DAG's own registers.
            "dm ( -0x21 , i6 ) = b4",
            "pm ( 0x10 , i12 ) = b8",
        ] {
            let src = format!(".section/SW/DOUBLE32 code;\n{bad};\n_pf_lp: nop;\n");
            assert!(assemble_source(&src, true).is_err(), "{bad}");
        }
        // `(PC, label)` became offset 0, a jump to itself. The reference assembler
        // leaves 0 and a PC-relative relocation (`jump (pc,.P33L1)` =
        // 073e00000000); numeric `(PC, n)`
        // keeps n with a generated-label relocation.
        let src = ".section/SW/DOUBLE32 code;\njump (pc, foo);\nnop;\n\
                   foo: jump (pc, 2);\nnop;\nnop;\n";
        let object = assemble_source(src, true).unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        let section = find_section_by_name(&object, &header, "code").unwrap();
        let code =
            &object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize];
        assert_eq!(&code[..6], &[0x07, 0x3e, 0, 0, 0, 0]);
        assert_eq!(&code[8..14], &[0x07, 0x3e, 0, 0, 0, 0x02]);
        let rela = find_section_by_name(&object, &header, ".rela.code").unwrap();
        let entries = &object[rela.sh_offset as usize..(rela.sh_offset + rela.sh_size) as usize];
        assert_eq!(entries.len(), 24, "both targets have relocations");
        assert_eq!(entries[4], selelf::elf::R_SHARC_PM_PCREL24 as u8);
    }

    #[test]
    fn widths_match_vendor_vectors() {
        // 32-bit Type 3b widths for
        // every M register (selas took only M4-M7, no (lw)/(swse)/(bwse))
        // and spaced suffixes; the literal 1 as `-0x1`, unspaced.
        for (source, word) in [
            ("r0 = dm ( i0 , m4 ) ( lw )", "513e403f"),
            ("dm ( i0 , m0 ) = r0 ( lw )", "503ec03f"),
            ("dm ( i1 , m2 ) = r2 ( sw )", "52bec13c"),
            ("r0 = dm(i0, m0)(bw)", "503e003c"),
            ("r0 = dm(i0, m0)(swse)", "503e403d"),
            ("r0 = dm ( i4 , m6 ) ( bwse )", "59be003d"),
            ("dm ( i0 , m6 ) = r8 ( bw )", "51be843c"),
            ("dm ( 0x0 , i2 ) = r10 ( lw )", "95880500"),
            ("r0 = r11 -0x1 , s0 = r0", "703e2802a0b0"),
            ("r0 = r5 - r10 + ci -0x1 , r14 = r0", "703e0700605a"),
            ("r1 = r4 -0x1 , i4 = modify ( i4 , m4 )", "043f2002a140"),
            // absolute-address widths ((lw) = Type 14 bit 39; the
            // others Type 14b, opcode 0x1a/0x1b) were refused.
            ("r0 = dm ( 0x252718 ) ( bw )", "1a0000252718"),
            ("r0 = dm ( 0x2d721c ) ( bwse )", "1a10002d721c"),
            ("dm ( 0x1 ) = r4 ( sw )", "1b4400000001"),
            ("r11 = dm(0x1)(swse)", "1a5b00000001"),
            ("r0 = dm ( 0x1 ) ( lw )", "108000000001"),
            ("dm(-0x7d1fff18)=r10(lw)", "118a82e000e8"),
            // Widths take any register in a DAG move.
            ("dm ( i4 , m6 ) = m13 ( bw )", "59be96bc"),
            // Type 7 widths, (nw) = bit 39 and (sw) = bit 23,
            // never compressed to 32 bits.
            ("i0 = modify ( i4 , m4 ) ( sw )", "043f24800000"),
            (
                "r12 = r12 + r10 , i0 = modify ( i2 , m4 ) ( sw )",
                "043ea2801cca",
            ),
            (
                "r0 = r2 and r1 , i2 = modify ( i2 , m6 ) ( nw )",
                "04beb0040021",
            ),
            (
                "r15 = r7 - 1 , i3 = modify ( i5 , m5 ) ( nw )",
                "04bf6e02af70",
            ),
            ("i10 = modify (i13, m10) (sw)", "047f57800000"),
            ("if eq i0 = modify ( i4 , m4 ) ( nw )", "048124000000"),
            ("f0 = scalb f2 by r4 , r4 < - > s5", "797e020bd024"),
            ("jump 0x1 ( db )", "063e04000001"),
            ("f0 = min ( f2 , f13 ) , r4 = r15", "70ff820e102d"),
            ("f0 = scalb f0 by r4 , r5 < - > s5", "797e028bd004"),
            ("if ne call 0x0 ( db )", "06a004000000"),
            ("r0 = dm(i0, 0x3f)", "607ff83f"),
            ("dm ( i4 , 0x20 ) = r2", "68ff013f"),
            ("r1 = r1 + r10 , dm ( 0x20 , i5 ) = r4", "6abf0200111a"),
            ("r0 = sat mrf (si)", "01901000"),
            ("r10 = sat mrf ( sf )", "01909a00"),
            ("r0 = sat mrb (si)", "01903000"),
            ("r0 = rnd mrf (sf)", "01919000"),
            ("r0 = rnd mrb (sf)", "0191b000"),
            ("r0 = sat mrf ( sf ) , dm ( i0 , m6 ) = r12", "51be86109000"),
            ("r0 = sat mrf ( sf ) , r2 = r1", "703e81109000"),
            ("r0 = rot r0 by 5", "023e00020500"),
            ("r0 = rot r0 by -0x1", "023e7802ff00"),
            (
                "comp ( f12 , f0 ) , r0 = dm ( i4 , m4 ) ( lw )",
                "593e4008a0c0",
            ),
            ("r0 = pass r9 , r14 = dm ( i3 , m2 ) ( lw )", "56be47021090"),
            ("r1 = pass r9 , r15 = dm ( i3 , m2 ) ( lw )", "56be47821190"),
            ("r0 = dm ( i4 , 0x2 ) ( swse )", "687e103d"),
            ("r0 = dm ( -0x2 , i4 ) ( swse )", "683ff03d"),
            ("r0 = dm ( -0x5 , i2 ) ( bwse )", "643fd839"),
            ("r2 = dm ( 0x20 , i6 ) ( bw )", "6c3f0138"),
            ("r7 = dm ( 0x20 , i3 ) ( sw )", "663f03bc"),
            ("r0 = dm ( i4 , 0xffffffff ) ( swse )", "687ff83d"),
            ("r0 = pm ( i8 , 0x0 )", "617e003f"),
            ("pm ( i10 , 0x0 ) = r2", "65fe013f"),
            ("r0 = pm ( i12 , 0x1 ) ( bw )", "697e0838"),
            ("r0 = pm ( 0x0 , i12 ) ( sw )", "693e003c"),
            ("if not sz r0 = pm ( i12 , 0x1 ) ( bw )", "69700838"),
            ("if not sz r2 = dm ( 0x20 , i5 )", "6a31013f"),
            ("if eq dm ( 0x20 , i5 ) = r15", "6a8107bf"),
            ("i3 = b2w ( i7 )", "04bfc4800000"),
            ("i3 = w2b ( i7 )", "04bfcc800000"),
            ("b3 = b2w ( b7 )", "04bfd4800000"),
            ("b3 = w2b ( b7 )", "04bfdc800000"),
            ("if not sv i12 = w2b ( i13 )", "04ef49800000"),
            ("r0 = pass r15 , i7 = b2w ( i7 )", "04bfc08210f0"),
            (
                "f1 = f1 * f7 , f12 = f8 + f12 , i1 = modify ( i1 , m4 )",
                "043e60581c70",
            ),
            (
                "f1 = f3 * f4 , f12 = f9 + f12 , i1 = modify ( i2 , m5 ) ( nw )",
                "04beab581cc4",
            ),
        ] {
            let bytes = visa_instruction_bytes(source);
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, word, "{source}");
        }
        for bad in [
            "r0 = r9-0x2",
            "r0 = r8 + -0x1",
            "dm(0x1)=r4(swse)",
            "r0=pm(0x1)(sw)",
            "if not eq r0 = r1 + r2",
            "cjump -0x100000000 ( db )",
            "if not ge jump (pc, 0x1)",
        ] {
            let src = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_source(&src, true).is_err(), "{bad}");
        }
    }

    #[test]
    fn absolute_address_label_offsets_are_relocation_addends() {
        // `r0 = dm(ldr.+4)` used to relocate against a
        // symbol named `LDR.+4`. The field is 0 and the constant is the
        // addend, as for symbolic loads.
        let src = ".section/SW/DOUBLE32 code;\nr0 = dm(ldr.+4);\ndm(_files.-8) = r1;\n\
                   .section/DM/DOUBLE32 data;\n.var ldr.[4];\n.var _files.[4];\n";
        let object = assemble_source(src, true).unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        let section = find_section_by_name(&object, &header, "code").unwrap();
        let code =
            &object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize];
        assert_eq!(code, [0x10, 0, 0, 0, 0, 0, 0x11, 0x01, 0, 0, 0, 0]);
        let rela = find_section_by_name(&object, &header, ".rela.code").unwrap();
        let entries = &object[rela.sh_offset as usize..(rela.sh_offset + rela.sh_size) as usize];
        let addends: Vec<i32> = entries
            .chunks(12)
            .map(|e| i32::from_le_bytes([e[8], e[9], e[10], e[11]]))
            .collect();
        assert_eq!(addends, [4, -8]);
    }

    #[test]
    fn vendor_rejected_forms_are_errors() {
        // Every row here is REJECTED by the reference assembler;
        // selas encoded each one (truncating, dropping or picking a write).
        for bad in [
            "r0 = r0 - 1 , r0 = dm ( i0 , 0x0 )",
            "f0 = - f2 , r0 = dm ( -0x5 , i4 )",
            "r0 = r1 or lshift r0 by 16",
            "r0 = r2 or lshift r12 by r3",
            "r0 = r1 or fdep r12 by 0x1f:0x1",
            "r0 = r0 or lshift r0 by -0x81",
            "r1 = lshift r1 by 0x100 , r0 = dm ( i5 , m4 )",
            "if eq r0 = lshift r0 by 0x100000",
            "btst r0 by 0x20",
            "r1 = btgl r13 by -0x1",
            "r8 = bset r8 by 0x100 , r2 = dm ( i4 , m4 )",
            "r2 = fext r15 by 0x0:0x100000",
            "r2 = fext r15 by -0x1:0x20",
            "btst r10 by 0xa , dm ( i11 , m5 ) = r2",
            "btst r10 by 0xa , dm ( i3 , m13 ) = r2",
            "modify ( i0 , 0x100000000 ) ( nw )",
            "modify ( i0 , -0x100000000 )",
            "i15 = modify ( i7 , -0x2 ) ( nw )",
            "bit set mode1 0x100000000",
            "bit tst ustat1 -0x100000001",
            "astatx = dm ( -0x75 , i14 )",
            "b7 = pm ( 0x10 , i3 )",
            "r0 = dm ( i0 , m8 )",
            "astatx = dm ( -0x100000000 , i6 )",
            "r15 = mrf + r11 * r6 ( ssi ) , dm ( 0x100000000 , i6 ) = r7",
            // The swap writes Rn; I/M registers stop at 15.
            "f0 = scalb f0 by r4 , r0 < - > s5",
            "f5 = f0 * f4 , f12 = f8 - f12 , r5 < - > s5",
            "i99 = modify (i99, m14) (sw)",
            "r0 = dm(i0, 0x40)",
            "r0 = dm(i0, -0x21)",
            "r15 = pass r9 , r14 = dm ( i3 , m2 ) ( lw )",
            "r0 = pass r9 , r1 = dm ( i3 , m2 ) ( lw )",
            "r0 = dm ( i4 , 0x40 ) ( swse )",
            "r0 = dm ( i4 , -0x21 ) ( swse )",
            "i15 = b2w ( i7 )",
            "if not sv i4 = w2b ( i12 )",
            // The last accepted-invalid oracle rows.
            "r4 = fpack r0",
            "r4 = funpack r0",
            "f4 = mant f0",
            "f4 = logb f0",
            "r5 = emuidle",
            "daddr = r5",
            "faddr = r5",
            "r0 = 0x100000000 -0x1",
            "r0 = -0x100000001 -0x1",
        ] {
            let src = format!(".section/SW/DOUBLE32 code;\n{bad};\n");
            assert!(assemble_source(&src, true).is_err(), "{bad}");
        }
        // The spaced sign-extend suffix still parses.
        assert_eq!(
            visa_instruction_bytes("r1 = fext r4 by 0x10:0x8 ( se )"),
            vec![0x02, 0x3e, 0x10, 0x12, 0x10, 0x14]
        );
    }

    #[test]
    fn float_conversion_uses_the_16_bit_form() {
        // Type 2c op 10 is `Fn = FLOAT Ry`: every one of the 323 vendor
        // rows with that opcode, e.g.
        // `f0 = float r0` = ca00 and `f1 = float r2` = ca12.
        assert_eq!(visa_instruction_bytes("f0 = float r0"), [0xca, 0x00]);
        assert_eq!(visa_instruction_bytes("f1 = float r2"), [0xca, 0x12]);
        assert_eq!(visa_instruction_bytes("f12 = float r0"), [0xca, 0xc0]);
    }

    #[test]
    fn visa_width_marker_ignores_multifunction_computes() {
        // 32-bit markers keep word bit 22 clear: the reference assembler 51be413c, 683e0938,
        // 043e603f, 083f343f. A 48-bit Type 3 carrying a float multiply +
        // dual add/sub with Fs = f8 has compute bits 22:16 = 0x78 and stays
        // 48-bit.
        for word in [
            [0x51, 0xbe, 0x41, 0x3c],
            [0x68, 0x3e, 0x09, 0x38],
            [0x04, 0x3e, 0x60, 0x3f],
            [0x08, 0x3f, 0x34, 0x3f],
        ] {
            assert_eq!(selinstr::visa::instruction_len(&word, 0), 4, "{word:02x?}");
        }
        assert_eq!(
            selinstr::visa::instruction_len(&[0x51, 0xbe, 0x00, 0x78, 0, 0], 0),
            6
        );
    }

    #[test]
    fn immediate_modify_type4_matches_vendor_vectors() {
        // Type 4 bit 38 (p1[6] in 32-bit form) is the update bit: DM(Ii,imm)
        // is post-modify, DM(imm,Ii) pre-modify. The reference assembler listings
        // are the source of these vectors.
        let visa: &[(&str, &[u8])] = &[
            ("if sz r15=dm(-3,i6)", &[0x6c, 0x11, 0xef, 0xbf]),
            ("r2=dm(i5,2)", &[0x6a, 0x7e, 0x11, 0x3f]),
        ];
        for (source, expected) in visa {
            assert_eq!(visa_instruction_bytes(source), *expected, "{source}");
        }
        let normal: &[(&str, &[u8])] = &[
            // oO_nwc lines 129 and 111.
            ("r2=dm(i5,2)", &[0x6a, 0x7e, 0x11, 0x00, 0x00, 0x00]),
            (
                "r2=pass r0, dm(-3,i6)=r15",
                &[0x6c, 0xbf, 0xef, 0x82, 0x12, 0x00],
            ),
        ];
        for (source, expected) in normal {
            let bytes = section_bytes(&format!(".section/CODE/NW code;\n{source};\n"), "code");
            assert_eq!(bytes, *expected, "{source}");
        }
        // In a DO body: post-modify store.
        let bytes = section_bytes(
            ".section/SW/DOUBLE32 code;\nlcntr = 2, do end until lce;\nend: dm(i1, 0x2) = r4;\n",
            "code",
        );
        assert_eq!(
            bytes[bytes.len() - 6..],
            [0x62, 0xfe, 0x12, 0x00, 0x00, 0x00]
        );
        // The pre-modify spelling must differ (it was silently identical).
        assert_ne!(
            visa_instruction_bytes("dm(i1, 0x2) = r4"),
            visa_instruction_bytes("dm(0x2, i1) = r4")
        );
        for source in ["if eq r0=dm(0x1000,i1)", "if eq i1=dm(1,i6)"] {
            let source = format!(".section/SW/DOUBLE32 code;\n{source};\n");
            assert!(assemble_source(&source, true).is_err(), "{source}");
        }
    }

    #[test]
    fn modify_with_destination_matches_vendor_vectors() {
        // `Ia = MODIFY(Ib, ...)` carries Ia XOR Ib. The reference assembler
        // vectors.
        let visa: &[(&str, &[u8])] = &[
            ("i1 = modify(i1,m4)", &[0x04, 0x3e, 0x60, 0x3f]),
            ("i1 = modify(i0,m4)", &[0x04, 0x3e, 0x21, 0x3f]),
            ("modify(i0, 0x2)", &[0x16, 0x00, 0x00, 0x00, 0x00, 0x02]),
            (
                "i4 = modify(i4, 0x10)",
                &[0x16, 0x04, 0x00, 0x00, 0x00, 0x10],
            ),
            (
                "i2 = modify(i4, 0x1a)",
                &[0x16, 0x34, 0x00, 0x00, 0x00, 0x1a],
            ),
            (
                "i4 = modify(i7, 1)(nw)",
                &[0x15, 0x9f, 0x00, 0x00, 0x00, 0x01],
            ),
        ];
        for (source, expected) in visa {
            let bytes = visa_instruction_bytes(source);
            assert_eq!(bytes, *expected, "{source}");
            let listing =
                selinstr::visa::disassemble_visa(&bytes, 0, selinstr::disasm::decode_instruction);
            assert_eq!(listing.len(), 1, "{source}");
            assert_eq!(visa_instruction_bytes(&listing[0].text), bytes, "{source}");
        }
        // Lines 668 and 659: 48-bit forms in a DO body, with and without compute.
        for (body, expected) in [
            ("i0 = modify(i1,m4)", [0x04, 0x3e, 0x61, 0x00, 0x00, 0x00]),
            (
                "f8 = f10*f7, i0 = modify(i1,m4)",
                [0x04, 0x3e, 0x61, 0x13, 0x08, 0xa7],
            ),
        ] {
            let bytes = section_bytes(
                &format!(
                    ".section/SW/DOUBLE32 code;\nlcntr = 2, do end until lce;\nend: {body};\n"
                ),
                "code",
            );
            assert_eq!(bytes[bytes.len() - 6..], expected, "{body}");
        }
        // `i1 = modify(i1,m4)(nw)` was refused here; the reference assembler takes it
        // (widths_match_vendor_vectors).
        for source in ["i1 = modify(i9,m4)", "i1 = bitrev(i0, 4)"] {
            let source = format!(".section/SW/DOUBLE32 code;\n{source};\n");
            assert!(assemble_source(&source, true).is_err(), "{source}");
        }
    }

    #[test]
    fn immediate_shifts_match_vendor_vectors() {
        // The reference assembler listings; SHIFTOP = register opcode >> 2 and a
        // negative count is sign-extended into DATAEX bits[30:27].
        let vectors: &[(&str, &[u8])] = &[
            (
                "r2 = lshift r2 by 0xe",
                &[0x02, 0x3e, 0x00, 0x00, 0x0e, 0x22],
            ),
            (
                "r0 = fext r0 by 0x0:0x2",
                &[0x02, 0x3e, 0x00, 0x10, 0x80, 0x00],
            ),
            (
                "r0 = lshift r0 by -4",
                &[0x02, 0x3e, 0x78, 0x00, 0xfc, 0x00],
            ),
            (
                "r2 = ashift r2 by -15",
                &[0x02, 0x3e, 0x78, 0x01, 0xf1, 0x22],
            ),
            // The OR form.
            (
                "r0 = r0 or lshift r0 by 16",
                &[0x02, 0x3e, 0x00, 0x08, 0x10, 0x00],
            ),
            (
                "r2 = r2 or lshift r12 by 16",
                &[0x02, 0x3e, 0x00, 0x08, 0x10, 0x2c],
            ),
            (
                "r12 = lshift r15 by -1",
                &[0x02, 0x3e, 0x78, 0x00, 0xff, 0xcf],
            ),
            ("btst r0 by 0x0", &[0x02, 0x3e, 0x00, 0x33, 0x00, 0x00]),
        ];
        for (source, expected) in vectors {
            let bytes = visa_instruction_bytes(source);
            assert_eq!(bytes, *expected, "{source}");
            let listing =
                selinstr::visa::disassemble_visa(&bytes, 0, selinstr::disasm::decode_instruction);
            assert_eq!(listing.len(), 1, "{source}");
            assert_eq!(visa_instruction_bytes(&listing[0].text), bytes, "{source}");
        }
    }

    #[test]
    fn dag_width_in_loop_body_matches_vendor_48_bit_vectors() {
        // The reference assembler: SW load and store in a DO body, and a
        // BW load.
        let cases: &[(&str, &[u8])] = &[
            ("r0 = dm(i0, m6)(sw)", &[0x51, 0xbe, 0x40, 0x30, 0x00, 0x00]),
            (
                "dm(i1, m6) = r0 (sw)",
                &[0x53, 0xbe, 0xc0, 0x30, 0x00, 0x00],
            ),
            ("r8 = dm(m5, i1)(bw)", &[0x43, 0x7e, 0x04, 0x30, 0x00, 0x00]),
            // Normal word in a DO body.
            ("dm(i1, m6) = r0", &[0x53, 0xbe, 0x80, 0x00, 0x00, 0x00]),
        ];
        for (body, expected) in cases {
            let source =
                format!(".section/SW/DOUBLE32 code;\nlcntr = 2, do end until lce;\nend: {body};\n");
            let object = assemble_source(&source, true).unwrap();
            let header = selelf::elf::parse_header(&object).unwrap();
            let section = find_section_by_name(&object, &header, "code").unwrap();
            let bytes =
                &object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize];
            assert_eq!(&bytes[bytes.len() - 6..], *expected, "{body}");
            let listing = selinstr::visa::disassemble_visa(
                &bytes[bytes.len() - 6..],
                0,
                selinstr::disasm::decode_instruction,
            );
            assert_eq!(listing.len(), 1, "{body}");
            let reparsed = crate::asmparse::parse_instruction(&listing[0].text).unwrap();
            let original = crate::asmparse::parse_instruction(body).unwrap();
            assert_eq!(format!("{reparsed:?}"), format!("{original:?}"), "{body}");
        }
    }

    #[test]
    fn dag_short_word_preserves_register_condition_and_address_order() {
        // Derived field checks, not additional vendor observations.
        for source in [
            "r0=dm(i0,m6)(sw)",
            "r2=dm(i0,m6)(sw)",
            "if eq r15=dm(m7,i3)(sw)",
            "if ne dm(i7,m4)=r5 (sw)",
            "r0=dm(i0,m6)(bw)",
            "if eq r15=dm(m7,i3)(bw)",
            "if ne dm(i7,m4)=r5 (bw)",
        ] {
            let bytes = visa_instruction_bytes(source);
            assert_eq!(bytes.len(), 4, "{source}");
            let listing =
                selinstr::visa::disassemble_visa(&bytes, 0, selinstr::disasm::decode_instruction);
            assert_eq!(listing.len(), 1);
            assert_eq!(visa_instruction_bytes(&listing[0].text), bytes, "{source}");
            let original = crate::asmparse::parse_instruction(source).unwrap();
            let decoded = crate::asmparse::parse_instruction(&listing[0].text).unwrap();
            assert_eq!(format!("{decoded:?}"), format!("{original:?}"));
        }
    }

    #[test]
    fn memory_se_is_rejected_but_compute_se_is_preserved() {
        for memory in ["dm(i0,m6)", "dm(m6,i0)", "pm(i8,m14)", "pm(m14,i8)"] {
            for suffix in ["(se)", "( se )", "(se )", "( se)"] {
                for source in [
                    format!("r0={memory}{suffix}"),
                    format!("{memory}=r0 {suffix}"),
                ] {
                    assert!(
                        crate::asmparse::parse_instruction(&source).is_err(),
                        "{source}"
                    );
                }
            }
        }
        for source in [
            "r0=r0 or fext r1 by r2 (se)",
            "r0=r0 or fext r1 by r2 (se), r3=dm(i0,m6)",
        ] {
            assert!(!visa_instruction_bytes(source).is_empty(), "{source}");
        }
    }

    #[test]
    fn unsupported_memory_widths_fail_without_falling_back_to_normal_word() {
        // `dm(imm, Ii)(sw|bw)` left this list once vendor vectors pinned it
        // (immediate_offset_width_matches_vendor_vectors).
        // Plain PM DAG moves left this list (G = bit 32,
        // plain_pm_dag_forms_match_vendor_vectors_and_decode_back).
        // M0-M3, (lw), (swse) and (bwse) DAG moves left this list
        // too (32-bit Type 3b vectors in widths_match_vendor_vectors).
        for source in [
            "r0=pm(i8,m14)(sw)",
            "pm(m14,i8)=r0 (sw)",
            "dm(i0,m6)=42 (sw)",
            "dm(i0,m6)=r0 (swse)",
            "r0=r1+r2, r3=dm(i0,m6)(sw)",
            "r0=r1+r2, r3=dm(i0,m6)(bw)",
            "r0=dm(i0,m6)(sw), r1=pm(i8,m14)",
            ".nocompress; r0=dm(i0,m0)(sw)",
            "lcntr=2, do end until lce; end: r0=r1+r2, r3=dm(i0,m6)(sw)",
        ] {
            let source = format!(".section/SW/DOUBLE32 code;\n{source};\n");
            assert!(assemble_source(&source, true).is_err(), "{source}");
        }
    }

    #[test]
    fn dag_type3b_keeps_stream_boundaries_for_all_dag1_pairs() {
        for i in 0..8 {
            for m in 4..8 {
                for post in [false, true] {
                    for suffix in ["", "(sw)", "(bw)"] {
                        let address = if post {
                            format!("i{i},m{m}")
                        } else {
                            format!("m{m},i{i}")
                        };
                        // A condition prevents the unmodified access from
                        // compressing into Type 3c; append NOP to catch overreads.
                        let source = format!("if eq r3=dm({address}){suffix}");
                        let mut bytes = visa_instruction_bytes(&source);
                        assert_eq!(bytes.len(), 4);
                        bytes.extend([0, 1]);
                        let listing = selinstr::visa::disassemble_visa(
                            &bytes,
                            0,
                            selinstr::disasm::decode_instruction,
                        );
                        assert_eq!(listing.len(), 2, "{source}");
                        assert_eq!(listing[0].raw.len(), 4, "{source}");
                        assert_eq!(listing[1].text, "nop", "{source}");
                        assert_eq!(
                            visa_instruction_bytes(&listing[0].text),
                            bytes[..4],
                            "{source}"
                        );
                    }
                }
            }
        }
    }
    fn assemble_str(source: &str) -> Vec<u8> {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        // Keep scratch inside the checkout and unique to this test run.
        let dir_name = format!("selas_mt_{}_{seq}_{tid:?}", std::process::id());
        let base = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/test-tmp"));
        let dir = base.join(dir_name);
        let _ = std::fs::create_dir_all(&dir);
        let input_path = dir.join("test.s");
        let output_path = dir.join("test.doj");
        std::fs::write(&input_path, source).unwrap();
        assemble_file(input_path.to_str().unwrap(), output_path.to_str().unwrap()).unwrap();
        let data = std::fs::read(&output_path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        data
    }

    #[test]
    fn test_assemble_and_readback() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _main;\n\
             _main:\n\
                 NOP\n\
                 RTS\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_machine, 0x85);
        assert_eq!(hdr.e_type, 1); // ET_REL

        let syms = selelf::elf::extract_global_symbols(&data).unwrap();
        assert!(syms.contains(&"_main".to_string()));
    }

    #[test]
    fn weak_directive_emits_weak_symbol() {
        let data = assemble_str(
            ".SECTION/SW seg_swco;\n\
             .WEAK hook.;\n\
             hook.:\n\
                 RTS\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        let sections: Vec<_> = (0..hdr.e_shnum as usize)
            .map(|i| {
                let off = hdr.e_shoff as usize + i * hdr.e_shentsize as usize;
                selelf::elf::parse_section_header(&data[off..], hdr.ei_data)
            })
            .collect();
        let symtab = sections
            .iter()
            .find(|s| s.sh_type == selelf::elf::SHT_SYMTAB)
            .expect("missing symtab");
        let strtab = &sections[symtab.sh_link as usize];
        let strtab_data =
            &data[strtab.sh_offset as usize..(strtab.sh_offset + strtab.sh_size) as usize];
        let nsyms = symtab.sh_size as usize / symtab.sh_entsize as usize;
        let hook = (0..nsyms)
            .map(|i| {
                let off = symtab.sh_offset as usize + i * symtab.sh_entsize as usize;
                selelf::elf::parse_symbol(&data[off..], hdr.ei_data)
            })
            .find(|sym| selelf::elf::read_string_at(strtab_data, sym.st_name) == "hook.")
            .expect("missing hook. symbol");
        assert_eq!(hook.bind(), selelf::elf::STB_WEAK);
    }

    #[test]
    fn test_assemble_multiple_instructions() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _func;\n\
             _func:\n\
                 R0 = 0x0000002A\n\
                 R8 = R6 + R7\n\
                 RTS\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_machine, 0x85);

        let shdr =
            find_section_by_name(&data, &hdr, "seg_pmco").expect("seg_pmco section not found");
        assert_eq!(shdr.sh_size, 18); // 3 instructions * 6 bytes
    }

    #[test]
    fn test_multi_section() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _main;\n\
             _main: NOP;\n\
                 RTS;\n\
             .ENDSEG;\n\
             .SECTION/DOUBLE32 seg_dmda;\n\
             .GLOBAL _data;\n\
             .VAR _data = 0x12345678;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();

        let pmco = find_section_by_name(&data, &hdr, "seg_pmco").expect("seg_pmco not found");
        assert_eq!(pmco.sh_size, 12); // 2 instructions * 6 bytes

        let dmda = find_section_by_name(&data, &hdr, "seg_dmda").expect("seg_dmda not found");
        assert_eq!(dmda.sh_size, 4); // one 32-bit value

        let syms = selelf::elf::extract_global_symbols(&data).unwrap();
        assert!(syms.contains(&"_main".to_string()));
        assert!(syms.contains(&"_data".to_string()));
    }

    #[test]
    fn test_endseg_handling() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             NOP;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);
    }

    #[test]
    fn test_var_data() {
        let data = assemble_str(
            ".SECTION/DOUBLE32 seg_dmda;\n\
             .VAR _val = 0xDEADBEEF;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();

        let shdr = find_section_by_name(&data, &hdr, "seg_dmda").expect("seg_dmda not found");
        assert_eq!(shdr.sh_size, 4);

        let off = shdr.sh_offset as usize;
        let bytes = &data[off..off + 4];
        let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(val, 0xDEADBEEF);
    }

    #[test]
    fn test_extern_symbol() {
        let data = assemble_str(
            ".EXTERN _ext_func;\n\
             .SECTION/PM seg_pmco;\n\
             .GLOBAL _main;\n\
             _main: NOP;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);

        let syms = selelf::elf::extract_global_symbols(&data).unwrap();
        assert!(syms.contains(&"_main".to_string()));
    }

    #[test]
    fn test_forward_label_reference() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _start;\n\
             .GLOBAL _end;\n\
             _start: NOP;\n\
             _end: RTS;\n\
             .ENDSEG;\n",
        );
        let syms = selelf::elf::extract_global_symbols(&data).unwrap();
        assert!(syms.contains(&"_start".to_string()));
        assert!(syms.contains(&"_end".to_string()));
    }

    #[test]
    fn test_no_section_fallback() {
        let data = assemble_str(".GLOBAL _main;\n_main:\n    NOP\n    RTS\n");
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_machine, 0x85);
        let syms = selelf::elf::extract_global_symbols(&data).unwrap();
        assert!(syms.contains(&"_main".to_string()));
    }

    #[test]
    fn test_parse_section_name_pm() {
        let (name, is_pm, _, _) = parse_section_name("PM seg_pmco;");
        assert_eq!(name, "seg_pmco");
        assert!(is_pm);
    }

    #[test]
    fn test_parse_section_name_dm() {
        let (name, is_pm, _, _) = parse_section_name("DM seg_dmda;");
        assert_eq!(name, "seg_dmda");
        assert!(!is_pm);
    }

    #[test]
    fn test_parse_section_name_bare() {
        let (name, is_pm, _, _) = parse_section_name("seg_pmco;");
        assert_eq!(name, "seg_pmco");
        assert!(!is_pm);
    }

    /// Bytes of section `name` in a `.doj`.
    fn object_section(object: &[u8], name: &str) -> Vec<u8> {
        let header = selelf::elf::parse_header(object).unwrap();
        let section = find_section_by_name(object, &header, name).unwrap();
        object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize].to_vec()
    }

    /// Value of every named symbol in a `.doj`.
    fn object_symbols(object: &[u8]) -> HashMap<String, u32> {
        let hdr = selelf::elf::parse_header(object).unwrap();
        let sections: Vec<_> = (0..hdr.e_shnum as usize)
            .map(|i| {
                let off = hdr.e_shoff as usize + i * hdr.e_shentsize as usize;
                selelf::elf::parse_section_header(&object[off..], hdr.ei_data)
            })
            .collect();
        let symtab = sections
            .iter()
            .find(|s| s.sh_type == selelf::elf::SHT_SYMTAB)
            .unwrap();
        let strtab = &sections[symtab.sh_link as usize];
        let names =
            &object[strtab.sh_offset as usize..(strtab.sh_offset + strtab.sh_size) as usize];
        (0..(symtab.sh_size / symtab.sh_entsize) as usize)
            .map(|i| {
                let off = symtab.sh_offset as usize + i * symtab.sh_entsize as usize;
                let sym = selelf::elf::parse_symbol(&object[off..], hdr.ei_data);
                (
                    selelf::elf::read_string_at(names, sym.st_name).to_string(),
                    sym.st_value,
                )
            })
            .collect()
    }

    /// `(r_offset, r_info)` of every entry of relocation section `name`.
    fn object_relas(object: &[u8], name: &str) -> Vec<(u32, u32)> {
        let header = selelf::elf::parse_header(object).unwrap();
        let data = object_section(object, name);
        data.chunks(12)
            .map(|entry| {
                let r = selelf::elf::parse_rela(entry, header.ei_data);
                (r.r_offset, r.r_info)
            })
            .collect()
    }

    fn reloc_tuples(object: &[u8], name: &str) -> Vec<(u32, u8, String, i32)> {
        let header = selelf::elf::parse_header(object).unwrap();
        let symtab = object_section(object, ".symtab");
        let names = object_section(object, ".strtab");
        object_section(object, name)
            .chunks_exact(12)
            .map(|entry| {
                let r = selelf::elf::parse_rela(entry, header.ei_data);
                let sym = selelf::elf::parse_symbol(
                    &symtab[(r.r_info >> 8) as usize * 16..],
                    header.ei_data,
                );
                (
                    r.r_offset,
                    r.r_info as u8,
                    selelf::elf::read_string_at(&names, sym.st_name).to_string(),
                    r.r_addend,
                )
            })
            .collect()
    }

    fn assemble_preprocessed(source: &str) -> Result<Vec<u8>> {
        crate::assemble_text(source, None, &[], &[], true)
    }

    #[test]
    fn var_arrays_match_vendor_object() {
        // The reference assembler object: seg_dmda is 284 bytes,
        // `ldr.[54]` is 54 words with an R_SHARC_DM_ADDR32 (0x0d) at byte 0
        // and 0x1c7586 at byte 4, `.align 4` after it keeps `ldr_nib.` at
        // byte 216, `ldr_tag.` is at 280 and the label `.ldr..end` is 216.
        let nib: [u32; 16] = [
            0x00000000, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4, 0x4db26158,
            0x5005713c, 0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c, 0x9b64c2b0, 0x86d3d2d4,
            0xa00ae278, 0xbdbdf21c,
        ];
        let words = |w: &[u32]| w.iter().map(|v| format!("{v:#010x}")).collect::<Vec<_>>();
        let source = format!(
            ".section/BW/DOUBLE32 seg_dmda;\n.align 4;\n\
             .global ldr., ldr_nib., ldr_tag., .ldr..end;\n\
             .var ldr.[54] = ldr_other_frame., 0x1c7586,\n    {};\n.ldr..end:\n.align 4;\n\
             .var ldr_nib.[16] = {},\n    {};\n\
             .var ldr_tag. = 0x444c3244;\n",
            vec!["0"; 52].join(","),
            words(&nib[..6]).join(", "),
            words(&nib[6..]).join(", "),
        );
        let object = assemble_preprocessed(&source).unwrap();
        let mut expected = vec![0u8; 4];
        expected.extend(0x1c7586u32.to_le_bytes());
        expected.extend([0u8; 52 * 4]);
        nib.iter().for_each(|w| expected.extend(w.to_le_bytes()));
        expected.extend(0x444c3244u32.to_le_bytes());
        assert_eq!(object_section(&object, "seg_dmda"), expected);
        let syms = object_symbols(&object);
        assert_eq!(syms["ldr."], 0);
        assert_eq!(syms[".ldr..end"], 216);
        assert_eq!(syms["ldr_nib."], 216);
        assert_eq!(syms["ldr_tag."], 280);
        let relas = object_relas(&object, ".rela.seg_dmda");
        assert_eq!(relas.len(), 1);
        assert_eq!((relas[0].0, relas[0].1 & 0xff), (0, 0x0d));
    }

    #[test]
    fn byte_arrays_match_vendor_module_object() {
        // The reference assembler object:
        // seg_mod_params is 260 zero bytes then 160 bytes of 0xff
        // (`_mod_owners` at 260, `_mod_curves` at 292), and
        // seg_mod_precision_data starts f2ff7fbf efff7fbf from LE32().
        let zero16 = vec!["0"; 16].join(",");
        let ff16 = vec!["255"; 16].join(",");
        let source = format!(
            "#define ZERO16 {zero16}\n#define FF16 {ff16}\n\
             #define LE32(x) ((x)&0xff), (((x)>>8)&0xff), (((x)>>16)&0xff), (((x)>>24)&0xff)\n\
             .section/DOUBLE32 seg_mod_params;\n.align 4;\n\
             .byte _mod_params[260] = 0,0,0,0,\n\
             \tZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,\n\
             \tZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16;\n\
             .byte _mod_owners[32] = FF16,FF16;\n\
             .byte _mod_curves[128] = FF16,FF16,FF16,FF16,FF16,FF16,FF16,FF16;\n\
             .section/DOUBLE32 seg_mod_precision_data;\n.align 4;\n\
             .global _mod_owners, _mod_curves, _mod_fine_pole;\n\
             .byte _mod_fine_pole[8] =\n    LE32(0xbf7ffff2), LE32(0xbf7fffef);\n"
        );
        let object = assemble_preprocessed(&source).unwrap();
        let mut params = vec![0u8; 260];
        params.extend([0xffu8; 160]);
        assert_eq!(object_section(&object, "seg_mod_params"), params);
        assert_eq!(
            object_section(&object, "seg_mod_precision_data"),
            [0xf2, 0xff, 0x7f, 0xbf, 0xef, 0xff, 0x7f, 0xbf]
        );
        let syms = object_symbols(&object);
        assert_eq!(syms["_mod_owners"], 260);
        assert_eq!(syms["_mod_curves"], 292);
    }

    #[test]
    fn var_reserves_its_declared_size() {
        // `.var buf[4];` used to reserve nothing, so `after`
        // overlapped it. selcc's bare `.VAR = v;` continuation word and
        // `.VAR name;` keep one word each.
        // `.byte storage.[9492] = 0;`
        // is 9492 zero bytes in the reference assembler object.
        let object = assemble_preprocessed(
            ".section/DOUBLE32 seg_dmda;\n.global buf, after, g, z, st;\n\
             .var buf[4];\n.var after = 7;\n.VAR g = 0x1;\n.VAR = 0x2;\n.VAR z;\n\
             .byte st[9492] = 0;\n",
        )
        .unwrap();
        let mut expected = vec![0u8; 16];
        for w in [7u32, 1, 2, 0] {
            expected.extend(w.to_le_bytes());
        }
        expected.extend([0u8; 9492]);
        assert_eq!(object_section(&object, "seg_dmda"), expected);
        let syms = object_symbols(&object);
        assert_eq!(
            (syms["after"], syms["g"], syms["z"], syms["st"]),
            (16, 20, 28, 32)
        );
        // A label before a multi-line statement, and a bare reopen.
        let object = assemble_preprocessed(
            ".section/DOUBLE32 seg_dmda;\n.global t, u;\nt: .var w[2] = 1,\n    2;\n\
             .section/SW/DOUBLE32 code;\nnop;\n.section seg_dmda;\n.var u = 3;\n",
        )
        .unwrap();
        let mut expected = Vec::new();
        for w in [1u32, 2, 3] {
            expected.extend(w.to_le_bytes());
        }
        assert_eq!(object_section(&object, "seg_dmda"), expected);
        let syms = object_symbols(&object);
        assert_eq!((syms["t"], syms["u"]), (0, 8));
    }

    #[test]
    fn standalone_nw_section_keeps_code_symbols_and_tail_padding() {
        // The reference declares
        // /NW/DOUBLE32 seg_l2code without /PM or /CODE and ends
        // with the following seven NW words,
        // including the vendor's conditional-tail NOP. _stereo_other
        // is word 126 in the full object, word 3 in this excerpt.
        let object = assemble_preprocessed(
            ".section/NW/DOUBLE32 seg_l2code;\n.global _stereo_other;\n\
             if ne jump _stereo_other;\nbtst r2 by 4;\njump 0x1c3360;\n\
             _stereo_other: r12=r12-r12;\nbtst r15 by r12;\njump 0x1c3360;\n",
        )
        .unwrap();
        let expected =
            "062000000000023e00330402063e001c3360013e00002ccc013e002cc0fc063e001c3360000000000000";
        let actual: String = object_section(&object, "seg_l2code")
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(object_symbols(&object)["_stereo_other"], 3);
        let header = selelf::elf::parse_header(&object).unwrap();
        assert_eq!(
            find_section_by_name(&object, &header, "seg_l2code")
                .unwrap()
                .sh_flags,
            6
        );
        assert!(object_section(&object, ".adi.attributes")
            .windows(9)
            .any(|w| w == [2, 9, 0, 0, 0, 1, 0, 0x12, 2]));
    }

    #[test]
    fn exported_loop_end_does_not_enable_automatic_fast_mode() {
        // 0c0024000000,
        // _m2_copy_end is global. Its two body instructions stay wide.
        let source = ".section/SW/DOUBLE32 code;\n.global _m2_copy_end;\n\
            lcntr=36, do _m2_copy_end until lce;\n\
            r0=dm(i4,m6);\n_m2_copy_end: dm(i5,m6)=r0;\n";
        let object = assemble_preprocessed(source).unwrap();
        assert_eq!(
            object_section(&object, "code")[..6],
            [0x0c, 0, 0x24, 0, 0, 0]
        );
        // Explicit (F) is retained
        // even for an externally bound end. Do not clear parsed flags.
        let object = assemble_preprocessed(&source.replace("until lce", "until lce (f)")).unwrap();
        assert_eq!(object_section(&object, "code")[3], 0x80);
        // A module's local two-instruction SW loop IS fast.
        let object = assemble_preprocessed(
            ".section/SW/DOUBLE32 code;\nlcntr=130, do _mod_copy_end until lce;\n\
             r0=dm(i4,m6)(sw);\n_mod_copy_end: dm(i5,m6)=r0(sw);\n",
        )
        .unwrap();
        assert_eq!(
            object_section(&object, "code")[..6],
            [0x0c, 0, 0x82, 0x80, 0, 0]
        );
    }

    #[test]
    fn byte_data_alignment_matches_vendor_sections() {
        // /DOUBLE32 seg_diag_glo and seg_diag_ghi have no .ALIGN; their
        // object alignment is 1. .ALIGN 4 on
        // seg_m2_state, whose object alignment is 4.
        let object = assemble_preprocessed(
            ".section/DOUBLE32 seg_diag_glo;\n.byte guard[4]=0;\n\
             .section/DOUBLE32 seg_m2_state;\n.align 4;\n.byte state[152]=0;\n",
        )
        .unwrap();
        let header = selelf::elf::parse_header(&object).unwrap();
        for (name, alignment) in [("seg_diag_glo", 1), ("seg_m2_state", 4)] {
            let section = find_section_by_name(&object, &header, name).unwrap();
            assert_eq!(section.sh_addralign, alignment);
        }
    }

    #[test]
    fn dm_words_match_vendor_data_and_symbol_units() {
        // The reference assembler:
        // bf7ffff2 -> bf7ffff200. Plain /DM
        // uses five file bytes per word, unlike /DOUBLE32's LE32 bytes.
        let object = assemble_preprocessed(
            ".section/DM seg_mod_precision_data;\n.global _mod_fine_pole;\n\
             .var _mod_fine_pole[2] = 0xbf7ffff2, 0xbf7fffef;\n",
        )
        .unwrap();
        assert_eq!(
            object_section(&object, "seg_mod_precision_data"),
            [0xbf, 0x7f, 0xff, 0xf2, 0, 0xbf, 0x7f, 0xff, 0xef, 0]
        );
        let header = selelf::elf::parse_header(&object).unwrap();
        let section = find_section_by_name(&object, &header, "seg_mod_precision_data").unwrap();
        assert_eq!(
            (section.sh_flags, section.sh_addralign, section.sh_entsize),
            (3, 1, 1)
        );
        assert!(object_section(&object, ".adi.attributes")
            .windows(9)
            .any(|w| w == [2, 9, 0, 0, 0, 1, 0, 0x12, 1]));
        // A test source: _srcA=8 after _dest[8],
        // 81960 file bytes for 16392 words in o_0_31.doj/seg_dmda.
        let object = assemble_preprocessed(
            ".section/DM seg_dmda;\n.global _dest, _srcA, _srcB, after;\n\
             .align 4;\n.var _dest[8];\n.var _srcA[4096];\n\
             .section/DOUBLE32 bytes;\n.var x=1;\n.section seg_dmda;\n\
             _srcB: .var words[4096];\n.var _bufA[4096];\n.var _bufB[4096];\nafter:\n",
        )
        .unwrap();
        assert_eq!(object_section(&object, "seg_dmda"), vec![0; 81960]);
        let syms = object_symbols(&object);
        assert_eq!(
            (syms["_dest"], syms["_srcA"], syms["_srcB"], syms["after"]),
            (0, 8, 4104, 16392)
        );
    }

    #[test]
    fn data_forms_without_a_vendor_layout_are_rejected() {
        for bad in [
            ".var t[3] = 1, 2;",
            ".var t[3] = 1, 2, 3, 4;",
            ".var a, b;",
            ".var x = 1, 2;",
            ".var y[];",
            ".var w = 0x100000000;",
            ".var f = 1.5;",
            ".byte b = 256;",
            ".byte b = -129;",
            ".byte b = sym.;",
            ".byte2 h = 1;",
            ".byte4 h = 1;",
            ".var/init24 x = 1;",
            ".var t[2] = 1,\n 2",
        ] {
            let source = format!(".section/DOUBLE32 seg_dmda;\n{bad}\n");
            assert!(assemble_preprocessed(&source).is_err(), "{bad}");
        }
        assert!(assemble_preprocessed(".section/SW/DOUBLE32 seg_swco;\n.var x = 1;\n").is_err());
        // these /DM forms still lack retained vendor vectors.
        for bad in [".byte b = 1;", ".var p = symbol;", ".var x;\n.align 4;"] {
            let source = format!(".section/DM seg_dmda;\n{bad}\n");
            assert!(assemble_preprocessed(&source).is_err(), "{bad}");
        }
        assert!(assemble_preprocessed(".var orphan = 1;\n").is_err());
        // Reopening with conflicting qualifiers, or a PM section.
        assert!(
            assemble_preprocessed(".section/DOUBLE32 s;\n.section/DM s;\n.var x = 1;\n").is_err()
        );
        assert!(
            assemble_preprocessed(".section/PM s;\nnop;\n.section/DOUBLE32 s;\n.var x = 1;\n")
                .is_err()
        );
    }

    /// Read a 48-bit big-endian instruction word from 6 bytes.
    fn read_word48(bytes: &[u8]) -> u64 {
        ((bytes[0] as u64) << 40)
            | ((bytes[1] as u64) << 32)
            | ((bytes[2] as u64) << 24)
            | ((bytes[3] as u64) << 16)
            | ((bytes[4] as u64) << 8)
            | (bytes[5] as u64)
    }

    #[test]
    fn test_set_directive() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _start;\n\
             _start: NOP;\n\
             .SET _alias = _start;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);
    }

    #[test]
    fn test_set_numeric() {
        let data = assemble_str(
            ".SET _CONST = 0x1000;\n\
             .SECTION/PM seg_pmco;\n\
             NOP;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);
    }

    #[test]
    fn test_set_comma_syntax() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _start;\n\
             _start: NOP;\n\
             .SET _alias, _start;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);
    }

    /// `.SET name = sym + N` resolves to the named symbol's address with
    /// the constant offset added.  selcc uses this form when a file-scope
    /// pointer initialiser takes the address of a sub-word interior of a
    /// packed aggregate (e.g. `(char *)&g_root + 1`); the resulting
    /// address is not word-aligned and therefore cannot be planted as a
    /// fresh `.VAR` slot owner, so the offset is folded into a `.SET`
    /// alias whose value is `slot_owner + remainder_bytes`.
    #[test]
    fn test_parse_sym_plus_offset() {
        assert_eq!(super::parse_sym_plus_offset("g."), ("g.".to_string(), 0));
        assert_eq!(
            super::parse_sym_plus_offset("g. + 3"),
            ("g.".to_string(), 3)
        );
        assert_eq!(
            super::parse_sym_plus_offset("g. - 1"),
            ("g.".to_string(), -1)
        );
        assert_eq!(
            super::parse_sym_plus_offset("g. + 0x10"),
            ("g.".to_string(), 0x10)
        );
        // Bare numeric falls back to (literal, 0) so the numeric arm of
        // resolve_aliases handles it unchanged.
        assert_eq!(
            super::parse_sym_plus_offset("0x1000"),
            ("0x1000".to_string(), 0)
        );
    }

    #[test]
    fn test_set_sym_plus_offset() {
        // The alias `_byte1` resolves to `_root + 1`, so the assembled
        // ELF carries an `_byte1` symbol one byte past `_root`.
        let data = assemble_str(
            ".SECTION/DOUBLE32 seg_dmda;\n\
             .GLOBAL _root;\n\
             .GLOBAL _byte1;\n\
             .VAR _root = 0x11111111;\n\
             .SET _byte1 = _root + 1;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);
        // Walk the symbol table and verify both symbols exist with the
        // expected delta. The .doj is local-symbols-then-globals so the
        // walk covers everything emitted by `emit_elf_bytes`.
        let sections: Vec<_> = (0..hdr.e_shnum as usize)
            .map(|i| {
                let off = hdr.e_shoff as usize + i * hdr.e_shentsize as usize;
                selelf::elf::parse_section_header(&data[off..], hdr.ei_data)
            })
            .collect();
        let symtab = sections
            .iter()
            .find(|s| s.sh_type == selelf::elf::SHT_SYMTAB)
            .expect("missing symtab");
        let strtab = &sections[symtab.sh_link as usize];
        let strtab_data =
            &data[strtab.sh_offset as usize..(strtab.sh_offset + strtab.sh_size) as usize];
        let nsyms = symtab.sh_size as usize / symtab.sh_entsize as usize;
        let mut root_val: Option<u32> = None;
        let mut byte1_val: Option<u32> = None;
        for i in 0..nsyms {
            let off = symtab.sh_offset as usize + i * symtab.sh_entsize as usize;
            let sym = selelf::elf::parse_symbol(&data[off..], hdr.ei_data);
            let name = selelf::elf::read_string_at(strtab_data, sym.st_name);
            if name == "_root" {
                root_val = Some(sym.st_value);
            } else if name == "_byte1" {
                byte1_val = Some(sym.st_value);
            }
        }
        let root_val = root_val.expect("_root symbol present");
        let byte1_val = byte1_val.expect("_byte1 symbol present");
        assert_eq!(
            byte1_val,
            root_val + 1,
            "alias `_byte1` should be one byte past `_root` (got root={root_val:#x}, byte1={byte1_val:#x})",
        );
    }

    #[test]
    fn test_branch_label_resolution() {
        // A bare local target stays absolute
        // (063e00000000 + PM24), regardless of the section's L2 name.
        let data = assemble_str(
            ".SECTION/PM seg_l2_swco;\n\
             .GLOBAL _start;\n\
             _start: NOP;\n\
             NOP;\n\
             JUMP _start;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);

        let shdr = find_section_by_name(&data, &hdr, "seg_l2_swco")
            .expect("seg_l2_swco section not found");
        assert_eq!(shdr.sh_size, 18); // 3 instructions * 6 bytes

        let off = shdr.sh_offset as usize;
        let word = read_word48(&data[off + 12..off + 18]);
        let decoded = selinstr::disasm::decode_instruction(word);
        assert_eq!(decoded, "JUMP 0x000000");
        let rels = reloc_tuples(&data, ".rela.seg_l2_swco");
        assert_eq!(rels, vec![(2, 0x0b, "_start".into(), 0)]);
        // An explicit PC label uses zero + PCREL24.
        // A defined compiler-style label must obey the same rule.
        let data = assemble_source(
            ".section/SW code;\njump (pc,.L_branch_1);\n.L_branch_1: nop;\n",
            true,
        )
        .unwrap();
        assert_eq!(object_section(&data, "code")[..6], [7, 0x3e, 0, 0, 0, 0]);
        assert_eq!(
            reloc_tuples(&data, ".rela.code"),
            vec![(0, 0xf, ".L_branch_1".into(), 0)]
        );
    }

    #[test]
    fn test_forward_branch_label() {
        // Forward local labels also use PM24.
        let data = assemble_str(
            ".SECTION/PM seg_l2_swco;\n\
             .GLOBAL _start;\n\
             _start: JUMP _end;\n\
             NOP;\n\
             _end: NOP;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);

        let shdr = find_section_by_name(&data, &hdr, "seg_l2_swco")
            .expect("seg_l2_swco section not found");
        let off = shdr.sh_offset as usize;
        let word = read_word48(&data[off..off + 6]);
        let decoded = selinstr::disasm::decode_instruction(word);
        assert_eq!(decoded, "JUMP 0x000000");
        assert_eq!(
            reloc_tuples(&data, ".rela.seg_l2_swco"),
            vec![(0, 0x0b, "_end".into(), 0)]
        );
        // CALL is 06be00000000 + PM24; IF NE JUMP is 062000000000 + PM24;
        // `if ne call 0x0` supplies the conditional CALL opcode.
        // Defining the target locally must not change either branch mode.
        for (ins, opcode) in [("call", 0xbe), ("if ne jump", 0x20), ("if ne call", 0xa0)] {
            for visa in [false, true] {
                let src = format!(
                    ".section/PM seg_l2code;\n{ins} _local;\n_local: nop;\n{}",
                    "nop;\n".repeat(6)
                );
                let obj = assemble_source(&src, visa).unwrap();
                assert_eq!(
                    object_section(&obj, "seg_l2code")[..6],
                    [6, opcode, 0, 0, 0, 0]
                );
                assert_eq!(
                    reloc_tuples(&obj, ".rela.seg_l2code"),
                    vec![(0, 0xb, "_local".into(), 0)]
                );
            }
        }
    }

    #[test]
    fn test_l1_local_branch_stays_pm24_reloc() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _start;\n\
             _start: JUMP _end;\n\
             NOP;\n\
             _end: NOP;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        let rela = find_section_by_name(&data, &hdr, ".rela.seg_pmco")
            .expect(".rela.seg_pmco section not found");
        assert_eq!(rela.sh_size, 12);
        let rela_off = rela.sh_offset as usize;
        let r_offset = u32::from_le_bytes(data[rela_off..rela_off + 4].try_into().unwrap());
        let r_info = u32::from_le_bytes(data[rela_off + 4..rela_off + 8].try_into().unwrap());
        assert_eq!(r_offset, 0);
        assert_eq!(r_info & 0xff, selelf::elf::R_SHARC_PM24);
    }

    #[test]
    fn test_do_loop_label() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             LCNTR = 10, DO _end UNTIL LCE;\n\
             NOP;\n\
             _end: NOP;\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();
        assert_eq!(hdr.e_type, 1);

        let shdr =
            find_section_by_name(&data, &hdr, "seg_pmco").expect("seg_pmco section not found");
        assert_eq!(shdr.sh_size, 18); // 3 instructions * 6 bytes
    }

    /// In VISA mode, the body of a hardware `DO ... UNTIL LCE` loop
    /// must be emitted as full 48-bit ISA instructions even when one
    /// or more of the individual instructions would compress down to
    /// 16/32 bits on their own. The hardware DO unit steps its
    /// internal PC in 48-bit instruction units, so a compressed body
    /// instruction desynchronises the loop-end compare and the board
    /// hangs: a hand-written `LCNTR=10, DO sum_loop_end UNTIL LCE`
    /// built with the old code path hit `R3 = R2 + R1`, the
    /// compressor emitted the 32-bit Type 6b form, and the board
    /// locked up every time. This test assembles a hardware loop in
    /// VISA mode with a body instruction that *would* compress in
    /// normal code, then walks the seg_pmco bytes and asserts each
    /// body slot occupies a full six bytes.
    #[test]
    fn test_hardware_loop_body_is_48_bit_in_visa() {
        let src = ".SECTION/PM seg_pmco;\n\
                   .GLOBAL loop_test;\n\
                   loop_test:\n\
                   LCNTR = 0xA, DO sum_loop_end UNTIL LCE;\n\
                   R0 = R0 + R1;\n\
                   R2 = PASS R0;\n\
                   sum_loop_end:\n\
                   R3 = R2 + R1;\n\
                   RTS;\n\
                   .ENDSEG;\n";
        let data = assemble_source(src, true).expect("assemble");
        let hdr = selelf::elf::parse_header(&data).unwrap();
        let shdr =
            find_section_by_name(&data, &hdr, "seg_pmco").expect("seg_pmco section not found");
        // DO + 3 body instructions, all full-width 48-bit (6 bytes
        // each), then the RTS after the loop in its 16-bit form (the reference assembler
        // `rts` = 0abe) = 26 bytes of code.
        assert_eq!(
            shdr.sh_size, 26,
            "expected DO + body 48-bit and a 16-bit RTS; got section size {} instead of 26",
            shdr.sh_size
        );
    }

    /// A LoadImm of an external symbol's address has to end up in the
    /// full 48-bit Type 17 form, with an R_SHARC_PM32 relocation whose
    /// `r_offset` is in parcel units. The old code path emitted
    /// R_SHARC_ADDR32 (value 0x01, which seld patches as a raw
    /// 4-byte LE write at the byte offset) and let visa_encode
    /// compress a LoadImm of zero to the 4-byte Type 17b form. When
    /// seld then patched that slot with a 32-bit address, the write
    /// ran off the end of the 4-byte instruction and the opcode bytes
    /// were destroyed. This test pins the fix: the section must hold
    /// a single 6-byte LoadImm, the relocation must be PM32 at parcel
    /// offset 0, and the instruction bytes must match an unresolved
    /// 48-bit `Ureg = 0x0` Type 17 encoding.
    #[test]
    fn test_load_imm_of_external_is_48_bit_pm32_reloc() {
        let src = ".SECTION/PM seg_pmco;\n\
                   .GLOBAL _start;\n\
                   .EXTERN _global;\n\
                   _start: R0 = _global;\n\
                   RTS;\n\
                   .ENDSEG;\n";
        let data = assemble_source(src, true).expect("assemble");
        let hdr = selelf::elf::parse_header(&data).unwrap();

        // seg_pmco: one LoadImm (48-bit, 6 bytes) + RTS (16-bit 0abe).
        let seg =
            find_section_by_name(&data, &hdr, "seg_pmco").expect("seg_pmco section not found");
        assert_eq!(
            seg.sh_size, 8,
            "LoadImm of an extern must not compress; \
             expected 8 bytes of code, got {}",
            seg.sh_size,
        );
        // Bytes 0..6 are the LoadImm with a 0x0 immediate placeholder.
        let off = seg.sh_offset as usize;
        assert_eq!(
            &data[off..off + 6],
            &[0x0f, 0x00, 0x00, 0x00, 0x00, 0x00],
            "unresolved Type 17 LoadImm encoding",
        );

        // .rela must have exactly one entry: type PM32 (0x0c), offset 0
        // (parcel units), symbol `_global`.
        let rela = find_section_by_name(&data, &hdr, ".rela.seg_pmco")
            .expect(".rela.seg_pmco section not found");
        assert_eq!(rela.sh_size, 12, "expected exactly one rela entry");
        let rela_off = rela.sh_offset as usize;
        let r_offset = u32::from_le_bytes(data[rela_off..rela_off + 4].try_into().unwrap());
        let r_info = u32::from_le_bytes(data[rela_off + 4..rela_off + 8].try_into().unwrap());
        let r_type = r_info & 0xff;
        assert_eq!(r_offset, 0, "r_offset must be parcel 0 of seg_pmco");
        assert_eq!(
            r_type,
            selelf::elf::R_SHARC_PM32,
            "LoadImm of extern must use R_SHARC_PM32 (0x0c), got 0x{r_type:x}",
        );
    }

    #[test]
    fn test_load_imm_of_local_cross_section_label_emits_pm32_reloc() {
        let src = ".SECTION/SW seg_swco;\n\
                   .GLOBAL _start;\n\
                   _start: I12 = .L_sel_l2_target;\n\
                   JUMP (M13,I12);\n\
                   .SECTION/SW seg_l2_swco;\n\
                   .L_sel_l2_target: NOP;\n\
                   .ENDSEG;\n";
        let data = assemble_source(src, true).expect("assemble");
        let hdr = selelf::elf::parse_header(&data).unwrap();

        let rela = find_section_by_name(&data, &hdr, ".rela.seg_swco")
            .expect(".rela.seg_swco section not found");
        assert_eq!(rela.sh_size, 12, "expected exactly one thunk relocation");
        let rela_off = rela.sh_offset as usize;
        let r_offset = u32::from_le_bytes(data[rela_off..rela_off + 4].try_into().unwrap());
        let r_info = u32::from_le_bytes(data[rela_off + 4..rela_off + 8].try_into().unwrap());
        let r_type = r_info & 0xff;
        assert_eq!(r_offset, 0, "LoadImm relocation should patch parcel 0");
        assert_eq!(
            r_type,
            selelf::elf::R_SHARC_PM32,
            "L2 thunk LoadImm must use a PM32 relocation, got 0x{r_type:x}",
        );

        let symtab_shdr = find_section_by_name(&data, &hdr, ".symtab").expect(".symtab not found");
        let strtab_shdr = find_section_by_name(&data, &hdr, ".strtab").expect(".strtab not found");
        let symtab_off = symtab_shdr.sh_offset as usize;
        let symtab_end = symtab_off + symtab_shdr.sh_size as usize;
        let strtab = &data[strtab_shdr.sh_offset as usize
            ..(strtab_shdr.sh_offset + strtab_shdr.sh_size) as usize];

        let mut found = false;
        let mut cursor = symtab_off;
        while cursor + 16 <= symtab_end {
            let st_name = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap());
            let st_info = data[cursor + 12];
            let name = selelf::elf::read_string_at(strtab, st_name);
            if name == ".L_sel_l2_target" {
                found = true;
                assert_eq!(st_info >> 4, selelf::elf::STB_LOCAL);
                break;
            }
            cursor += 16;
        }
        assert!(
            found,
            "cross-section L2 thunk label must be emitted as a local symbol"
        );
    }

    /// A CALL of an external function must likewise be 48-bit (Branches
    /// already always are in VISA mode), and its relocation must be
    /// R_SHARC_PM24 at a parcel offset -- not R_SHARC_ADDR32 at a byte
    /// offset, which would patch bytes 0..3 and clobber the opcode.
    #[test]
    fn test_call_extern_is_pm24_reloc() {
        let src = ".SECTION/PM seg_pmco;\n\
                   .GLOBAL _start;\n\
                   .EXTERN _helper;\n\
                   _start: CALL _helper;\n\
                   RTS;\n\
                   .ENDSEG;\n";
        let data = assemble_source(src, true).expect("assemble");
        let hdr = selelf::elf::parse_header(&data).unwrap();

        // CALL (48-bit) + RTS, which stays 48-bit right after a CALL
        // (the reference assembler `call _x; rts` = 06be00000000 0a3e00000000).
        let seg =
            find_section_by_name(&data, &hdr, "seg_pmco").expect("seg_pmco section not found");
        assert_eq!(seg.sh_size, 12);

        let rela = find_section_by_name(&data, &hdr, ".rela.seg_pmco")
            .expect(".rela.seg_pmco section not found");
        assert_eq!(rela.sh_size, 12);
        let rela_off = rela.sh_offset as usize;
        let r_offset = u32::from_le_bytes(data[rela_off..rela_off + 4].try_into().unwrap());
        let r_info = u32::from_le_bytes(data[rela_off + 4..rela_off + 8].try_into().unwrap());
        let r_type = r_info & 0xff;
        assert_eq!(r_offset, 0, "r_offset must be parcel 0 of seg_pmco");
        assert_eq!(
            r_type,
            selelf::elf::R_SHARC_PM24,
            "CALL of extern must use R_SHARC_PM24 (0x0b), got 0x{r_type:x}",
        );
    }

    /// `CJUMP local_label` has to become an R_SHARC_PM24 relocation
    /// against a STB_LOCAL symbol: Type 25a has no PC-relative variant,
    /// so the only way to produce position-independent bytes is to let
    /// the linker patch the final address from an in-file symbol table
    /// entry. Without the local-symbol path this would either silently
    /// encode a section-relative value as absolute (wrong at runtime)
    /// or declare the local label as an undefined external (fails to
    /// link).
    #[test]
    fn test_cjump_local_label_emits_local_symbol() {
        let data = assemble_str(
            ".SECTION/PM seg_pmco;\n\
             .GLOBAL _start;\n\
             _start: NOP;\n\
             my_label: NOP;\n\
             CJUMP my_label (DB);\n\
             .ENDSEG;\n",
        );
        let hdr = selelf::elf::parse_header(&data).unwrap();

        // The symbol table must contain `my_label` as a defined local
        // symbol pointing at word 1 of seg_pmco.
        let symtab_shdr = find_section_by_name(&data, &hdr, ".symtab").expect(".symtab not found");
        let strtab_shdr = find_section_by_name(&data, &hdr, ".strtab").expect(".strtab not found");
        let symtab_off = symtab_shdr.sh_offset as usize;
        let symtab_end = symtab_off + symtab_shdr.sh_size as usize;
        let strtab_off = strtab_shdr.sh_offset as usize;
        let strtab_end = strtab_off + strtab_shdr.sh_size as usize;
        let strtab = &data[strtab_off..strtab_end];

        let mut found = false;
        let mut cursor = symtab_off;
        while cursor + 16 <= symtab_end {
            let st_name = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            let st_value = u32::from_le_bytes([
                data[cursor + 4],
                data[cursor + 5],
                data[cursor + 6],
                data[cursor + 7],
            ]);
            let st_info = data[cursor + 12];
            let st_shndx = u16::from_le_bytes([data[cursor + 14], data[cursor + 15]]);
            let name = selelf::elf::read_string_at(strtab, st_name);
            if name == "my_label" {
                let binding = st_info >> 4;
                assert_eq!(
                    binding,
                    selelf::elf::STB_LOCAL,
                    "my_label must be STB_LOCAL"
                );
                assert_ne!(st_shndx, 0, "my_label must be defined (not SHN_UNDEF)");
                assert_eq!(st_value, 1, "my_label must point at word 1 of seg_pmco");
                found = true;
                break;
            }
            cursor += 16;
        }
        assert!(found, "my_label not found in symbol table");
    }

    /// Vendor objects: whole code section bytes, SW and NW: the six-instruction tail after
    /// every conditional transfer form, 48-bit pads inside a DO tail counted
    /// over the emitted stream, the auto fast bit over source instructions,
    /// `.ALIGN` before the last eleven, `RTS (LR)` and direct `JUMP (LA)`.
    #[test]
    fn tail_window_and_transfer_vectors() {
        let cases: &[(&str, bool, &str, &str, &str)] = &[
            ("tail_nocompress", false, ".section/PM/DOUBLE32 tnc;\n.extern _ext;\n.nocompress;\nif ne call _ext;\n", "tnc", "06a000000000000100010001000100010001"),
            ("tail_nocompress", true, ".section/PM/DOUBLE32 tnc;\n.extern _ext;\n.nocompress;\nif ne call _ext;\n", "tnc", "06a000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("tail_two_transfers", false, ".section/PM/DOUBLE32 ttt;\n.extern _ext, _ext2;\nif ne jump _ext;\nnop;\nnop;\nif eq call _ext2;\nnop;\n", "ttt", "06200000000000010001068000000000000100010001000100010001"),
            ("tail_two_transfers", true, ".section/PM/DOUBLE32 ttt;\n.extern _ext, _ext2;\nif ne jump _ext;\nnop;\nnop;\nif eq call _ext2;\nnop;\n", "ttt", "062000000000000000000000000000000000068000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("rts_lr", false, ".section/PM/DOUBLE32 t_rts_lr;\nif ne rts (lr);\n", "t_rts_lr", "0aa1000100010001000100010001"),
            ("rts_lr", true, ".section/PM/DOUBLE32 t_rts_lr;\nif ne rts (lr);\n", "t_rts_lr", "0a2001000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("rts_compute", false, ".section/PM/DOUBLE32 t_rts_cmp;\nif ne rts, r0=r0+r1;\n", "t_rts_cmp", "0a2000001001000100010001000100010001"),
            ("rts_compute", true, ".section/PM/DOUBLE32 t_rts_cmp;\nif ne rts, r0=r0+r1;\n", "t_rts_cmp", "0a2000001001000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("call_db_empty", false, ".section/PM/DOUBLE32 t_call_db;\n.extern _ext;\nif ne call _ext (db);\n", "t_call_db", "06a004000000000100010001000100010001"),
            ("call_db_empty", true, ".section/PM/DOUBLE32 t_call_db;\n.extern _ext;\nif ne call _ext (db);\n", "t_call_db", "06a004000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("jump_la", false, ".section/PM/DOUBLE32 t_jump_la;\n.extern _ext;\nif ne jump _ext (la);\n", "t_jump_la", "066000000000000100010001000100010001"),
            ("jump_la", true, ".section/PM/DOUBLE32 t_jump_la;\n.extern _ext;\nif ne jump _ext (la);\n", "t_jump_la", "066000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("ind_jump_else", false, ".section/PM/DOUBLE32 t_ind_else;\nif ne jump (m8,i8), else r0=r0+r1;\n", "t_ind_else", "082002001001000100010001000100010001"),
            ("ind_jump_else", true, ".section/PM/DOUBLE32 t_ind_else;\nif ne jump (m8,i8), else r0=r0+r1;\n", "t_ind_else", "082002001001000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("ind_call_db", false, ".section/PM/DOUBLE32 t_ind_call_db;\nif ne call (m8,i8) (db);\n", "t_ind_call_db", "08a0043f000100010001000100010001"),
            ("ind_call_db", true, ".section/PM/DOUBLE32 t_ind_call_db;\nif ne call (m8,i8) (db);\n", "t_ind_call_db", "08a004000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("not_lce", false, ".section/PM/DOUBLE32 t_not_lce;\n.extern _ext;\nif not lce jump _ext;\n", "t_not_lce", "061e00000000000100010001000100010001"),
            ("not_lce", true, ".section/PM/DOUBLE32 t_not_lce;\n.extern _ext;\nif not lce jump _ext;\n", "t_not_lce", "061e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("pc_else", false, ".section/PM/DOUBLE32 t_pc_else;\n_t: nop;\nif ne jump (pc,_t), else r0=r0+r1;\n", "t_pc_else", "0001092002001001000100010001000100010001"),
            ("pc_else", true, ".section/PM/DOUBLE32 t_pc_else;\n_t: nop;\nif ne jump (pc,_t), else r0=r0+r1;\n", "t_pc_else", "000000000000092002001001000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("tf_jump", false, ".section/PM/DOUBLE32 t_tf;\n.extern _ext;\nif tf jump _ext;\n", "t_tf", "061a00000000000100010001000100010001"),
            ("tf_jump", true, ".section/PM/DOUBLE32 t_tf;\n.extern _ext;\nif tf jump _ext;\n", "t_tf", "061a00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("loop_end_transfer", false, ".section/PM/DOUBLE32 t_loop_end;\n.extern _ext;\nlcntr=4, do _le until lce;\nnop;\nnop;\n_le: if ne jump _ext;\n", "t_loop_end", "0c0004000000000000000000000000000000062000000000000100010001000100010001"),
            ("loop_end_transfer", true, ".section/PM/DOUBLE32 t_loop_end;\n.extern _ext;\nlcntr=4, do _le until lce;\nnop;\nnop;\n_le: if ne jump _ext;\n", "t_loop_end", "0c0004000000000000000000000000000000062000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("tail_window_m2", false, ".section/PM/DOUBLE32 tw_m2;\n.extern _ext;\n.global _f.end, _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m2", "0c00040000000001000100010001000100010001000100010001000100010620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000abe"),
            ("tail_window_m2", true, ".section/PM/DOUBLE32 tw_m2;\n.extern _ext;\n.global _f.end, _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m2", "0c00040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a3e00000000"),
            ("tail_window_m5", false, ".section/PM/DOUBLE32 tw_m5;\n.extern _ext;\n.global _f.end, _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m5", "0c0004000000000100010001000100010001000100010001000100010001062000000000000100010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000abe"),
            ("tail_window_m5", true, ".section/PM/DOUBLE32 tw_m5;\n.extern _ext;\n.global _f.end, _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m5", "0c00040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a3e00000000"),
            ("tail_window_m8", false, ".section/PM/DOUBLE32 tw_m8;\n.extern _ext;\n.global _f.end, _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m8", "0c0004000000000100010001000100010001000100010001000100010001062000000000000100010001000100010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000abe"),
            ("tail_window_m8", true, ".section/PM/DOUBLE32 tw_m8;\n.extern _ext;\n.global _f.end, _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m8", "0c00040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a3e00000000"),
            ("align_follow11", false, ".section/PM/DOUBLE32 al_eleven;\n.global _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\n.align 2;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\n", "al_eleven", "0c0004000000000100010001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("align_follow11", true, ".section/PM/DOUBLE32 al_eleven;\n.global _le;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\n.align 2;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\n", "al_eleven", "0c0004000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("tail_window_m8_local", false, ".section/PM/DOUBLE32 tw_m8l;\n.extern _ext;\n.global _f.end;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m8l", "0c0004000000000100010001000100010001000100010001000100010001062000000000000100010001000100010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000abe"),
            ("tail_window_m8_local", true, ".section/PM/DOUBLE32 tw_m8l;\n.extern _ext;\n.global _f.end;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\n_f.end: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_m8l", "0c00040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a3e00000000"),
            ("tail_window_control", false, ".section/PM/DOUBLE32 tw_ctl;\n.extern _ext;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_ctl", "0c00048000000001000100010001000100010001000100010001000100010620000000000001000100010001000100010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000abe"),
            ("tail_window_control", true, ".section/PM/DOUBLE32 tw_ctl;\n.extern _ext;\nlcntr=4, do _le until lce;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nif ne jump _ext;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\nrts;\n", "tw_ctl", "0c00048000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a3e00000000"),
        ];
        for &(case, nwc, source, section, vendor) in cases {
            let options = AsmOptions {
                nwc,
                ..AsmOptions::default()
            };
            let object = assemble_source_with(source, true, options)
                .unwrap_or_else(|e| panic!("{case} nwc={nwc}: {e}"));
            let hex: String = object_section(&object, section)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(hex, vendor, "{case} nwc={nwc}");
        }
    }

    /// SW `.ALIGN` before one of a DO's last ten instructions is
    /// rejected by the reference assembler; NW accepts it.
    /// `(F)` with a non-LCE termination is rejected.
    #[test]
    fn vendor_rejections() {
        let align_ten = ".section/PM/DOUBLE32 al_ten;\n.global _le;\nlcntr=4, do _le until lce;\n\
            nop;\nnop;\nnop;\n.align 2;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n_le: nop;\n";
        assert!(assemble_source(align_ten, true).is_err());
        let nw = AsmOptions {
            nwc: true,
            ..AsmOptions::default()
        };
        assert!(assemble_source_with(align_ten, true, nw).is_ok());
        let f_eq = ".section/PM/DOUBLE32 s;\ndo _x until eq (f);\nnop;\n_x: nop;\n";
        assert!(assemble_source(f_eq, true).is_err());
        let f_lce = ".section/PM/DOUBLE32 s;\ndo _x until lce (f);\nnop;\n_x: nop;\n";
        assert!(assemble_source(f_lce, true).is_ok());
    }

    /// Generated `___ADI_AGL_*` labels beside an existing label
    /// (`jump (pc,2)` line 3, `call (pc,1)` line 4, both at
    /// `_named` = parcel 6) and the `.__operator` value (1).
    #[test]
    fn generated_label_names_and_operator_value() {
        let source = ".section/PM/DOUBLE32 pc_existing_label;\njump (pc,2);\ncall (pc,1);\n\
            _named: nop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n";
        let object = assemble_source(source, true).unwrap();
        let symbols = object_symbols(&object);
        assert_eq!(symbols["___ADI_AGL_JUMP_2"], 6);
        assert_eq!(symbols["___ADI_AGL_CALL_3"], 6);
        let source = ".section/PM/DOUBLE32 expr_do_shifted;\n.global _base, _end;\n_base: nop;\n\
            lcntr=16, do (pc,_end-_base) until lce;\nnop;\nnop;\n_end: nop;\nnop;\n";
        let object = assemble_source(source, true).unwrap();
        assert_eq!(object_symbols(&object)[".__operator"], 1);
    }

    /// Every SAT/RND form the reference assembler accepts. TRNC
    /// is not reference-assembler syntax.
    #[test]
    fn mr_saturate_round_vectors() {
        let cases: &[(&str, &str)] = &[
            ("r5 = sat mrf (si);", "01901500"),
            ("r5 = sat mrf (ui);", "01900500"),
            ("r5 = sat mrf (sf);", "01909500"),
            ("r5 = sat mrf (uf);", "01908500"),
            ("r5 = sat mrb (si);", "01903500"),
            ("r5 = sat mrb (ui);", "01902500"),
            ("r5 = sat mrb (sf);", "0190b500"),
            ("r5 = sat mrb (uf);", "0190a500"),
            ("r5 = rnd mrf (sf);", "01919500"),
            ("r5 = rnd mrf (uf);", "01918500"),
            ("r5 = rnd mrb (sf);", "0191b500"),
            ("r5 = rnd mrb (uf);", "0191a500"),
            ("mrf = sat mrf;", "0190d000"),
            ("mrf = sat mrf (si);", "01905000"),
            ("mrf = sat mrf (ui);", "01904000"),
            ("mrf = sat mrf (sf);", "0190d000"),
            ("mrf = sat mrf (uf);", "0190c000"),
            ("mrb = sat mrb;", "0190f000"),
            ("mrb = sat mrb (si);", "01907000"),
            ("mrb = sat mrb (ui);", "01906000"),
            ("mrb = sat mrb (sf);", "0190f000"),
            ("mrb = sat mrb (uf);", "0190e000"),
            ("mrf = rnd mrf;", "0191d000"),
            ("mrf = rnd mrf (sf);", "0191d000"),
            ("mrf = rnd mrf (uf);", "0191c000"),
            ("mrb = rnd mrb;", "0191f000"),
            ("mrb = rnd mrb (sf);", "0191f000"),
            ("mrb = rnd mrb (uf);", "0191e000"),
        ];
        for &(line, vendor) in cases {
            let source = format!(".section/PM/DOUBLE32 s;\n{line}\n");
            let hex: String = section_bytes(&source, "s")
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(hex, vendor, "{line}");
        }
        for line in [
            "mrf = trnc mrf;",
            "mrb = trnc mrb;",
            "r5 = trnc mrf;",
            "r5 = trnc mrb;",
        ] {
            let source = format!(".section/PM/DOUBLE32 s;\n{line}\n");
            assert!(assemble_source(&source, true).is_err(), "{line}");
        }
    }

    /// A branch ends at least three parcels after a preceding
    /// JUMP/CALL ends (`branch_shadow`), vendor objects, SW and NW; and the
    /// reference assembler rejects a branch in a delay slot.
    #[test]
    fn branch_shadow_vectors() {
        let cases: &[(&str, bool, &str, &str, &str)] = &[
            ("shadow_a_01", false, ".section/PM/DOUBLE32 br_a01;\n.extern _x;\njump _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a01", "063e000000000a3e00000000000100010001000100010001"),
            ("shadow_a_02", false, ".section/PM/DOUBLE32 br_a02;\n.extern _x;\njump _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a02", "063e0000000000010a3e00000000000100010001000100010001"),
            ("shadow_a_03", false, ".section/PM/DOUBLE32 br_a03;\n.extern _x;\njump _x;\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a03", "063e00000000000100010abe000100010001000100010001"),
            ("shadow_a_04", false, ".section/PM/DOUBLE32 br_a04;\n.extern _x;\njump _x;\nnop;\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a04", "063e000000000001000100010abe000100010001000100010001"),
            ("shadow_a_05", false, ".section/PM/DOUBLE32 br_a05;\n.extern _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a05", "00010abe000100010001000100010001"),
            ("shadow_a_06", false, ".section/PM/DOUBLE32 br_a06;\n.extern _x;\ncall _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a06", "06be000000000a3e00000000000100010001000100010001"),
            ("shadow_a_07", false, ".section/PM/DOUBLE32 br_a07;\n.extern _x;\nrts;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a07", "0abe0abe000100010001000100010001"),
            ("shadow_a_08", false, ".section/PM/DOUBLE32 br_a08;\n.extern _x;\nif ne jump _x;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a08", "062000000000083f3400000019010001000100010001000100010001"),
            ("shadow_a_09", false, ".section/PM/DOUBLE32 br_a09;\n.extern _x;\nif ne jump _x;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a09", "0620000000000001083f343f19010001000100010001000100010001"),
            ("shadow_a_10", false, ".section/PM/DOUBLE32 br_a10;\n.extern _x;\nif ne jump _x;\nnop;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a10", "06200000000000010001083f343f19010001000100010001000100010001"),
            ("shadow_a_11", false, ".section/PM/DOUBLE32 br_a11;\n.extern _x;\njump _x;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a11", "063e00000000083f3400000019010001000100010001000100010001"),
            ("shadow_a_12", false, ".section/PM/DOUBLE32 br_a12;\n.extern _x;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a12", "0001083f343f19010001000100010001000100010001"),
            ("shadow_a_13", false, ".section/PM/DOUBLE32 br_a13;\n.extern _x;\njump (m14,i12) (db);\nrframe;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a13", "083f343f190100010abe000100010001000100010001"),
            ("shadow_a_14", false, ".section/PM/DOUBLE32 br_a14;\n.extern _x;\nif ne jump _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a14", "0620000000000a3e00000000000100010001000100010001"),
            ("shadow_a_15", false, ".section/PM/DOUBLE32 br_a15;\n.extern _x;\njump (m14,i12);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a15", "083f303f0a3e00000000000100010001000100010001"),
            ("shadow_a_16", false, ".section/PM/DOUBLE32 br_a16;\n.extern _x;\njump (pc,3);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a16", "073e000000030a3e0000000000010001000100010001000100010001"),
            ("shadow_a_17", false, ".section/PM/DOUBLE32 br_a17;\n.extern _x;\nif ne rts;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a17", "0aa00abe000100010001000100010001"),
            ("shadow_a_18", false, ".section/PM/DOUBLE32 br_a18;\n.extern _x;\nidle;\nnop;\nemuidle;\njump (pc,0);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a18", "0081000100c1073e000000000a3e00000000000100010001000100010001"),
            ("shadow_a_01", true, ".section/PM/DOUBLE32 br_a01;\n.extern _x;\njump _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a01", "063e000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_02", true, ".section/PM/DOUBLE32 br_a02;\n.extern _x;\njump _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a02", "063e000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_03", true, ".section/PM/DOUBLE32 br_a03;\n.extern _x;\njump _x;\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a03", "063e000000000000000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_04", true, ".section/PM/DOUBLE32 br_a04;\n.extern _x;\njump _x;\nnop;\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a04", "063e000000000000000000000000000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_05", true, ".section/PM/DOUBLE32 br_a05;\n.extern _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a05", "0000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_06", true, ".section/PM/DOUBLE32 br_a06;\n.extern _x;\ncall _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a06", "06be000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_07", true, ".section/PM/DOUBLE32 br_a07;\n.extern _x;\nrts;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a07", "0a3e000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_08", true, ".section/PM/DOUBLE32 br_a08;\n.extern _x;\nif ne jump _x;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a08", "062000000000083f34000000190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_09", true, ".section/PM/DOUBLE32 br_a09;\n.extern _x;\nif ne jump _x;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a09", "062000000000000000000000083f34000000190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_10", true, ".section/PM/DOUBLE32 br_a10;\n.extern _x;\nif ne jump _x;\nnop;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a10", "062000000000000000000000000000000000083f34000000190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_11", true, ".section/PM/DOUBLE32 br_a11;\n.extern _x;\njump _x;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a11", "063e00000000083f34000000190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_12", true, ".section/PM/DOUBLE32 br_a12;\n.extern _x;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a12", "000000000000083f34000000190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_13", true, ".section/PM/DOUBLE32 br_a13;\n.extern _x;\njump (m14,i12) (db);\nrframe;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a13", "083f340000001900000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_14", true, ".section/PM/DOUBLE32 br_a14;\n.extern _x;\nif ne jump _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a14", "0620000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_15", true, ".section/PM/DOUBLE32 br_a15;\n.extern _x;\njump (m14,i12);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a15", "083f300000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_16", true, ".section/PM/DOUBLE32 br_a16;\n.extern _x;\njump (pc,3);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a16", "073e000000030a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_17", true, ".section/PM/DOUBLE32 br_a17;\n.extern _x;\nif ne rts;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a17", "0a20000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_a_18", true, ".section/PM/DOUBLE32 br_a18;\n.extern _x;\nidle;\nnop;\nemuidle;\njump (pc,0);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_a18", "00800000000000000000000000c000000000073e000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_01", false, ".section/PM/DOUBLE32 br_c01;\n.extern _x;\nif ne jump _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c01", "06200000000000010a3e00000000000100010001000100010001"),
            ("shadow_c_02", false, ".section/PM/DOUBLE32 br_c02;\n.extern _x;\njump _x;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c02", "063e000000000001083f343f19010001000100010001000100010001"),
            ("shadow_c_03", false, ".section/PM/DOUBLE32 br_c03;\n.extern _x;\nif ne jump _x (db);\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c03", "062004000000000100010abe000100010001000100010001"),
            ("shadow_c_04", false, ".section/PM/DOUBLE32 br_c04;\n.extern _x;\njump _x (db);\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c04", "063e04000000000100010abe000100010001000100010001"),
            ("shadow_c_06", false, ".section/PM/DOUBLE32 br_c06;\n.extern _x;\ncall _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c06", "06be0000000000010a3e00000000000100010001000100010001"),
            ("shadow_c_07", false, ".section/PM/DOUBLE32 br_c07;\n.extern _x;\njump _x;\nrti;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c07", "063e000000000b3e00000000000100010001000100010001"),
            ("shadow_c_08", false, ".section/PM/DOUBLE32 br_c08;\n.extern _x;\njump (m14,i12);\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c08", "083f303f00010a3e00000000000100010001000100010001"),
            ("shadow_c_09", false, ".section/PM/DOUBLE32 br_c09;\n.extern _x;\njump _x;\nnop;\nif ne rts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c09", "063e0000000000010a2000000000000100010001000100010001"),
            ("shadow_c_10", false, ".section/PM/DOUBLE32 br_c10;\n.extern _x;\nif ne jump _x;\nif ne rts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c10", "0620000000000a2000000000000100010001000100010001"),
            ("shadow_c_11", false, ".section/PM/DOUBLE32 br_c11;\n.extern _x;\nif ne call _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c11", "06a0000000000a3e00000000000100010001000100010001"),
            ("shadow_c_12", false, ".section/PM/DOUBLE32 br_c12;\n.extern _x;\nif ne call _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c12", "06a00000000000010a3e00000000000100010001000100010001"),
            ("shadow_c_13", false, ".section/PM/DOUBLE32 br_c13;\n.extern _x;\nif ne jump (m14,i12);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c13", "0821303f0a3e00000000000100010001000100010001"),
            ("shadow_c_14", false, ".section/PM/DOUBLE32 br_c14;\n.extern _x;\nif ne jump (m14,i12);\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c14", "0821303f00010a3e00000000000100010001000100010001"),
            ("shadow_c_15", false, ".section/PM/DOUBLE32 br_c15;\n.extern _x;\njump _x;\nr0 = r1 + r2;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c15", "063e00000000018010120abe000100010001000100010001"),
            ("shadow_c_16", false, ".section/PM/DOUBLE32 br_c16;\n.extern _x;\njump _x;\nr0 = 0x12345678;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c16", "063e000000000f00123456780abe000100010001000100010001"),
            ("shadow_c_17", false, ".section/PM/DOUBLE32 br_c17;\n.extern _x;\njump _x;\nrts (db);\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c17", "063e000000000a3e0400000000010001000100010001000100010001"),
            ("shadow_c_18", false, ".section/PM/DOUBLE32 br_c18;\n.extern _x;\njump _x;\nnop;\nnop;\n_l: rts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c18", "063e00000000000100010abe000100010001000100010001"),
            ("shadow_c_01", true, ".section/PM/DOUBLE32 br_c01;\n.extern _x;\nif ne jump _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c01", "0620000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_02", true, ".section/PM/DOUBLE32 br_c02;\n.extern _x;\njump _x;\nnop;\njump (m14,i12) (db);\nrframe;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c02", "063e00000000000000000000083f34000000190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_03", true, ".section/PM/DOUBLE32 br_c03;\n.extern _x;\nif ne jump _x (db);\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c03", "0620040000000000000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_04", true, ".section/PM/DOUBLE32 br_c04;\n.extern _x;\njump _x (db);\nnop;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c04", "063e040000000000000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_06", true, ".section/PM/DOUBLE32 br_c06;\n.extern _x;\ncall _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c06", "06be000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_07", true, ".section/PM/DOUBLE32 br_c07;\n.extern _x;\njump _x;\nrti;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c07", "063e000000000b3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_08", true, ".section/PM/DOUBLE32 br_c08;\n.extern _x;\njump (m14,i12);\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c08", "083f300000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_09", true, ".section/PM/DOUBLE32 br_c09;\n.extern _x;\njump _x;\nnop;\nif ne rts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c09", "063e000000000000000000000a2000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_10", true, ".section/PM/DOUBLE32 br_c10;\n.extern _x;\nif ne jump _x;\nif ne rts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c10", "0620000000000a2000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_11", true, ".section/PM/DOUBLE32 br_c11;\n.extern _x;\nif ne call _x;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c11", "06a0000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_12", true, ".section/PM/DOUBLE32 br_c12;\n.extern _x;\nif ne call _x;\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c12", "06a0000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_13", true, ".section/PM/DOUBLE32 br_c13;\n.extern _x;\nif ne jump (m14,i12);\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c13", "0821300000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_14", true, ".section/PM/DOUBLE32 br_c14;\n.extern _x;\nif ne jump (m14,i12);\nnop;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c14", "0821300000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_15", true, ".section/PM/DOUBLE32 br_c15;\n.extern _x;\njump _x;\nr0 = r1 + r2;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c15", "063e00000000013e000010120a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_16", true, ".section/PM/DOUBLE32 br_c16;\n.extern _x;\njump _x;\nr0 = 0x12345678;\nrts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c16", "063e000000000f00123456780a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_17", true, ".section/PM/DOUBLE32 br_c17;\n.extern _x;\njump _x;\nrts (db);\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c17", "063e000000000a3e04000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
            ("shadow_c_18", true, ".section/PM/DOUBLE32 br_c18;\n.extern _x;\njump _x;\nnop;\nnop;\n_l: rts;\nnop;\nnop;\nnop;\nnop;\nnop;\nnop;\n", "br_c18", "063e000000000000000000000000000000000a3e00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        ];
        for &(case, nwc, source, section, vendor) in cases {
            let options = AsmOptions {
                nwc,
                ..AsmOptions::default()
            };
            let object = assemble_source_with(source, true, options)
                .unwrap_or_else(|e| panic!("{case} nwc={nwc}: {e}"));
            let hex: String = object_section(&object, section)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(hex, vendor, "{case} nwc={nwc}");
        }
        let delay_slot_branch = ".section/PM/DOUBLE32 br_c05;\n.extern _x;\njump _x (db);\nrts;\nnop;\n";
        assert!(assemble_source(delay_slot_branch, true).is_err());
        let nw = AsmOptions {
            nwc: true,
            ..AsmOptions::default()
        };
        assert!(assemble_source_with(delay_slot_branch, true, nw).is_err());
    }

    // Ported from ElekLabs/selache fix/encodings (Em D).
    fn seg_pmco_size(src: &str) -> u32 {
        let data = assemble_source(src, true).expect("assemble");
        let hdr = selelf::elf::parse_header(&data).unwrap();
        find_section_by_name(&data, &hdr, "seg_pmco")
            .expect("seg_pmco section not found")
            .sh_size
    }

    /// An inner DO loop must not end the 48-bit rule of the outer
    /// loop: the outer loop's last instruction (a NOP, which VISA
    /// would compress) is encoded in full form, and code after the
    /// outer end is compressed again.
    #[test]
    fn test_nested_hardware_loop_stays_48_bit_until_outer_end() {
        let nested = ".SECTION/PM seg_pmco;\n\
                      LCNTR = 4, DO outer_end UNTIL LCE;\n\
                      R0 = DM(I1, M6);\n\
                      LCNTR = 4, DO inner_end UNTIL LCE;\n\
                      R1 = R0 AND R6;\n\
                      inner_end: DM(I0, M6) = R1;\n\
                      R2 = R1 + R0;\n\
                      outer_end: NOP;\n\
                      NOP;\n\
                      .ENDSEG;\n";
        // 2 DO + 5 body instructions, all 6 bytes, then one NOP
        // after the loop that compresses to 2 bytes.
        assert_eq!(seg_pmco_size(nested), 7 * 6 + 2);
        // Same loops with the end labels on their own lines, and two
        // loops ending on the same instruction.
        let own_line = ".SECTION/PM seg_pmco;\n\
                        LCNTR = 4, DO outer_end UNTIL LCE;\n\
                        LCNTR = 4, DO outer_end UNTIL LCE;\n\
                        R1 = R0 AND R6;\n\
                        outer_end:\n\
                        NOP;\n\
                        NOP;\n\
                        .ENDSEG;\n";
        assert_eq!(seg_pmco_size(own_line), 4 * 6 + 2);
    }

    /// A single loop still lets code after the loop compress.
    #[test]
    fn test_single_hardware_loop_compresses_after_end() {
        let src = ".SECTION/PM seg_pmco;\n\
                   LCNTR = 4, DO loop_end UNTIL LCE;\n\
                   R1 = R0 AND R6;\n\
                   loop_end: NOP;\n\
                   NOP;\n\
                   .ENDSEG;\n";
        assert_eq!(seg_pmco_size(src), 3 * 6 + 2);
    }
}
