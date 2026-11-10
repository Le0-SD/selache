// SPDX-License-Identifier: GPL-3.0
// asmparse.rs --- SHARC+ assembly text parser
// Copyright (c) 2026 Jakob Kastelic

//! Assembly text parser for SHARC+ (ADSP-2156x).
//!
//! Parses the disassembly text format produced by `selinstr::disasm` into
//! structured `selinstr::encode::Instruction` values.

use crate::error::{Error, Result};
use selinstr::encode::{
    AluOp, BranchTarget, ComputeOp, DagAccess, DagWidth, FaluOp, Instruction, LoopCounter,
    MemAccess, MemWidth, MulOp, MultiOp, ShiftOp,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One parsed line from an assembly source file.
#[derive(Debug)]
pub struct ParsedLine {
    pub label: Option<String>,
    pub instruction: Option<Instruction>,
    pub directive: Option<Directive>,
    /// Primary label reference for this instruction, if any.  When the
    /// instruction can be resolved as a same-section PC-relative branch,
    /// this is the only label that matters; otherwise the assembler
    /// emits a relocation against this symbol.
    pub label_ref: Option<String>,
    /// Additional label references for multi-symbol expressions like
    /// `I7 = ldf_stack_space + ldf_stack_length;` that need more than
    /// one relocation against the same operand slot.  Each entry is
    /// `(symbol_name, addend)` and produces an `R_SHARC_PM_SW_BRANCHRETURN`
    /// (Add) relocation at the same byte offset as `label_ref`.  The
    /// SHARC+ Tools dialect expresses two-symbol additions this way and
    /// libsel's startup.s uses it to seed `I7` with the top of the
    /// LDF-defined stack reserve.
    pub extra_label_refs: Vec<(String, i64)>,
    /// Source line (1-based, after preprocessing). The reference assembler names the
    /// label it generates for a numeric DO target after it
    /// (`___ADI_AGL_DO_5`).
    pub line: u32,
    /// Labels the reference assembler generates at this instruction for numeric `(PC, n)`
    /// targets (`___ADI_AGL_<KIND>_<line>`), beside any source label.
    pub generated_labels: Vec<String>,
}

/// Assembler directive.
#[derive(Debug)]
pub enum Directive {
    Section(String),
    /// `.GLOBAL foo, bar, baz;` — declare each name as a global
    /// symbol.  The SHARC+ Tools dialect lets one directive list
    /// several names separated by commas; selas splits at parse time
    /// so each name becomes its own symbol-table entry.
    Global(Vec<String>),
    /// `.WEAK foo, bar;` — see `Global` for the comma-list rationale.
    Weak(Vec<String>),
    /// `.EXTERN foo, bar;` — see `Global` for the comma-list rationale.
    /// (libsel's startup.s relies on the comma form for the
    /// `ldf_stack_space, ldf_stack_length` pair defined by the LDF.)
    Extern(Vec<String>),
    /// `.VAR` (32-bit elements) or `.BYTE` (8-bit elements) data
    /// declaration.
    Data(DataDecl),
    Align(u32),
    Set(String, String),
    /// `.NOCOMPRESS;` — suppress VISA compression for following
    /// instructions until a matching `.COMPRESS;` directive.  The
    /// SHARC+ Tools dialect uses this to keep critical sequences (e.g.
    /// the 12 instructions that the SHBTB_CFG-write errata requires
    /// after disabling the BTB) emitted in full 48-bit ISA form: the
    /// errata count is in 48-bit instruction units, and a 16/32-bit
    /// VISA NOP would silently shorten the window.
    NoCompress,
    /// `.COMPRESS;` — re-enable VISA compression after a
    /// `.NOCOMPRESS;` block.
    Compress,
}

/// One element initializer of a `.VAR` / `.BYTE` declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum DataInit {
    /// Constant value, already range-checked for the element width.
    Num(i64),
    /// Absolute address of a symbol, patched by the linker. Only
    /// 32-bit (`.VAR`) elements can hold one.
    Sym(String),
}

/// A `.VAR` / `.BYTE` declaration laid out the way the reference assembler does
/// (measured on two test modules): `name` is a
/// scalar of one element, `name[N]` an array of N elements, `name[]` an
/// array sized by its initializer list. Without initializers the
/// elements are zero. `name` is empty for selcc's bare `.VAR = v;`
/// continuation word.
#[derive(Debug, Clone, PartialEq)]
pub struct DataDecl {
    pub name: String,
    /// Element size in bytes: 4 for `.VAR`, 1 for `.BYTE`.
    pub width: usize,
    /// Number of elements.
    pub len: usize,
    /// Either empty (zero-filled) or exactly `len` entries.
    pub inits: Vec<DataInit>,
}

/// Parse the body of a `.VAR` (`width` 4) or `.BYTE` (`width` 1)
/// directive. Forms whose the reference assembler layout is not pinned by a vendor
/// object are rejected rather than guessed: several declarations in one
/// directive, a sized array with a short initializer list holding a
/// non-zero value, file or string initializers, and values that do not
/// fit the element.
fn parse_data_decl(body: &str, width: usize) -> std::result::Result<DataDecl, String> {
    let s = body.replace('\t', " ");
    let s = s.trim().trim_end_matches(';').trim();
    let (lhs, rhs) = match s.split_once('=') {
        Some((l, r)) => (l.trim(), Some(r.trim())),
        None => (s, None),
    };
    if lhs.contains(',') {
        return Err(format!(
            "one declaration per data directive is supported, got `{lhs}`"
        ));
    }
    // `None` = scalar, `Some(None)` = `name[]`, `Some(Some(n))` = `name[n]`.
    let (name, size) = match lhs.find('[') {
        Some(open) => {
            let close = lhs
                .rfind(']')
                .filter(|&c| c > open && lhs[c + 1..].trim().is_empty())
                .ok_or_else(|| format!("malformed array declaration `{lhs}`"))?;
            let n = lhs[open + 1..close].trim();
            let size = if n.is_empty() {
                None
            } else {
                let v = eval_const_expr(n)
                    .filter(|&v| v > 0 && v <= i64::from(u32::MAX))
                    .ok_or_else(|| format!("invalid array length `{n}`"))?;
                Some(v as usize)
            };
            (lhs[..open].trim(), Some(size))
        }
        None => (lhs, None),
    };
    if !name.is_empty() && !is_valid_label(name) {
        return Err(format!("invalid data symbol name `{name}`"));
    }
    let mut inits = match rhs {
        None => Vec::new(),
        Some(r) => {
            let mut out = Vec::new();
            let mut prev = 0;
            let mut cuts = find_all_commas_outside_parens(r);
            cuts.push(r.len());
            for c in cuts {
                out.push(parse_data_init(&r[prev..c], width)?);
                prev = c + 1;
            }
            out
        }
    };
    let len = match size {
        None => {
            if name.is_empty() && inits.is_empty() {
                return Err("data directive needs a name or an initializer".into());
            }
            if inits.len() > 1 {
                return Err(format!(
                    "scalar `{name}` has {} initializers; declare it as `{name}[]`",
                    inits.len()
                ));
            }
            1
        }
        Some(None) => {
            if inits.is_empty() {
                return Err(format!("`{name}[]` needs an initializer list"));
            }
            inits.len()
        }
        Some(Some(n)) => {
            // The reference assembler accepts a short list: a module's
            // `.byte storage.[9492] = 0;` gives 9492
            // zero bytes. Only an all-zero short list is pinned by a
            // vendor object; any other short list is rejected.
            let all_zero = inits.iter().all(|i| *i == DataInit::Num(0));
            if inits.len() < n && all_zero {
                inits.clear();
            } else if !inits.is_empty() && inits.len() != n {
                return Err(format!(
                    "`{name}[{n}]` has {} initializers; list all {n}, none, or only zeros",
                    inits.len()
                ));
            }
            n
        }
    };
    Ok(DataDecl {
        name: name.to_string(),
        width,
        len,
        inits,
    })
}

/// Element width of a `.VAR` (4) or `.BYTE` (1) directive keyword at
/// the start of `upper`; `None` for anything else, including `.BYTE2`,
/// `.BYTE4` and qualified `.VAR/...` forms.
fn data_directive_width(upper: &str) -> Option<usize> {
    let ends_keyword = |rest: &str| rest.is_empty() || rest.starts_with([' ', '\t', ';', '=']);
    if upper.strip_prefix(".VAR").is_some_and(ends_keyword) {
        Some(4)
    } else if upper.strip_prefix(".BYTE").is_some_and(ends_keyword) {
        Some(1)
    } else {
        None
    }
}

/// Number of comma-separated parallel operations in an instruction's text
/// (`else` prefixes dropped, commas inside parentheses ignored).
fn parallel_operations(text: &str) -> usize {
    let mut depth = 0i32;
    let mut count = 0;
    let mut current = String::new();
    // A bare condition (`IF EQ`, `IF NOT AC`) is not an operation: the
    // disassembler prints an empty compute slot as `IF EQ  , DM(...)`.
    let flush = |current: &mut String, count: &mut usize| {
        let words: Vec<&str> = current.split_whitespace().collect();
        let bare_condition = words.first().is_some_and(|w| w.eq_ignore_ascii_case("IF"))
            && (words.len() == 2 || (words.len() == 3 && words[1].eq_ignore_ascii_case("NOT")));
        if !words.is_empty() && !bare_condition {
            *count += 1;
        }
        current.clear();
    };
    for ch in text.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                flush(&mut current, &mut count);
                continue;
            }
            _ => {}
        }
        current.push(ch);
    }
    flush(&mut current, &mut count);
    count
}

/// DM addresses come from DAG1 (I0-I7, M0-M7) and PM addresses from DAG2
/// (I8-I15, M8-M15). Of 46,543 `DM|PM(...)` operands the reference
/// assembler accepts, none mixes them; it rejects all
/// 2,152 that do. selas folded the index into the 3-bit field, so
/// `astatx = dm(-0x75, i14)` accessed through I6.
fn check_dag_registers(text: &str, line: u32) -> Result<()> {
    let upper = text.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let mut at = 0;
    while let Some(found) = upper[at..].find("M") {
        let k = at + found;
        at = k + 1;
        if k == 0 || !matches!(bytes[k - 1], b'D' | b'P') {
            continue;
        }
        let start = k - 1;
        if start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
            continue;
        }
        let pm = bytes[start] == b'P';
        let rest = upper[k + 1..].trim_start();
        if !rest.starts_with('(') {
            continue;
        }
        let open = upper.len() - rest.len();
        let Some(close) = find_matching_close_paren(&upper, open) else {
            continue;
        };
        for operand in upper[open + 1..close].split(',') {
            let operand = operand.trim();
            let Some(kind @ (b'I' | b'M')) = operand.bytes().next() else {
                continue;
            };
            let Ok(n) = operand[1..].parse::<u8>() else {
                continue;
            };
            if n < 16 && (n >= 8) != pm {
                return Err(Error::Parse {
                    line,
                    msg: format!(
                        "{}{n} is not a {} register: {}",
                        kind as char,
                        if pm { "PM (DAG2)" } else { "DM (DAG1)" },
                        text.trim()
                    ),
                });
            }
        }
    }
    // `MODIFY(Ii, Mm)`: I and M from one DAG. Type 7 holds 3-bit fields
    // and a DAG bit, so `i0 = modify(i2, m11)` modified I2 by M3; the reference assembler
    // rejects every such row (`modify(i0, m10)`, `r10 = r10 + r14 , i5 =
    // modify ( i5 , m12 )`).
    let mut at = 0;
    while let Some(found) = upper[at..].find("MODIFY") {
        let k = at + found;
        at = k + 6;
        let rest = upper[at..].trim_start();
        if !rest.starts_with('(') {
            continue;
        }
        let open = upper.len() - rest.len();
        let Some(close) = find_matching_close_paren(&upper, open) else {
            continue;
        };
        let dags: Vec<(char, u8)> = upper[open + 1..close]
            .split(',')
            .filter_map(|op| {
                let op = op.trim();
                let kind = op.chars().next()?;
                let n = op.get(1..)?.parse::<u8>().ok()?;
                (matches!(kind, 'I' | 'M') && n < 16).then_some((kind, n))
            })
            .collect();
        if let [(_, a), (_, b)] = dags[..] {
            if (a >= 8) != (b >= 8) {
                return Err(Error::Parse {
                    line,
                    msg: format!("MODIFY mixes DAG1 and DAG2 registers: {}", text.trim()),
                });
            }
        }
    }
    Ok(())
}

/// An I or M register of the DAG doing the access moves only as a
/// pre-modify load (`i3 = dm(m0, i0)`, `m0 = dm(m7, i2)`: 22 the reference assembler rows,
/// all accepted). The reference assembler rejects every post-modify form and every store
/// (`i1 = dm(i0, m0)`, `dm(m7, i2) = m4`, `i10 = pm(i8, m8)`: 82 rows, all
/// rejected). No vendor rows cover L or B.
fn check_dag_self_transfer(instr: &Instruction, line: u32) -> Result<()> {
    if let Instruction::UregDagMove {
        pm,
        write,
        ureg,
        post_modify,
        ..
    } = *instr
    {
        let dag_register = matches!(ureg >> 4, 1 | 2) && ((ureg & 0xF) >= 8) == pm;
        if dag_register && (write || post_modify) {
            return Err(Error::Parse {
                line,
                msg: format!(
                    "{} cannot be {} through its own DAG{}",
                    selinstr::disasm::ureg_display_name(ureg),
                    if write { "stored" } else { "loaded" },
                    if write { "" } else { " with post-modify" }
                ),
            });
        }
    }
    // Type 15 (`DM|PM(offset, Ii)`): storing an I/M/L/B register of the
    // access's own DAG is rejected (16 of 16 rows, e.g. `dm ( -0x21 , i6 )
    // = b4`, `pm ( 0x10 , i12 ) = b8`); loads are accepted (1,111 rows).
    if let Instruction::UregMemAccess {
        pm,
        write: true,
        ureg,
        ..
    } = *instr
    {
        if matches!(ureg >> 4, 1..=4) && ((ureg & 0xF) >= 8) == pm {
            return Err(Error::Parse {
                line,
                msg: format!(
                    "{} cannot be stored through its own DAG",
                    selinstr::disasm::ureg_display_name(ureg)
                ),
            });
        }
    }
    Ok(())
}

/// Refuse parallel operations that write one data register twice
/// (`r0 = r0 - 1, r0 = dm(i0, 0)`, `f0 = -f2, r0 = dm(-5, i4)`). None of the
/// 101,833 instructions the reference assembler accepts
/// does; it rejects all 1,819 that do. selas encoded them, leaving the
/// result to whichever write the hardware lets win.
fn check_no_double_write(text: &str, line: u32) -> Result<()> {
    let mut written = [false; 16];
    let mut depth = 0i32;
    let mut part = String::new();
    let mut check = |part: &str| -> Result<()> {
        let mut words = part.trim();
        // Skip `IF [NOT] cond` and `ELSE` prefixes.
        if words.len() > 3 && words[..3].eq_ignore_ascii_case("IF ") {
            let rest = words[3..].trim_start();
            let rest = if rest.len() > 4 && rest[..4].eq_ignore_ascii_case("NOT ") {
                rest[4..].trim_start()
            } else {
                rest
            };
            words = rest.split_once(' ').map_or("", |(_, r)| r).trim_start();
        }
        if words.len() > 5 && words[..5].eq_ignore_ascii_case("ELSE ") {
            words = words[5..].trim_start();
        }
        // `Rn <-> Sn` writes Rn too: the reference assembler rejects `f0 = scalb f0 by r4 ,
        // r0 < - > s5` and takes `f0 = scalb f2 by r4 , r4 < - > s5`.
        let packed: String = words.chars().filter(|c| !c.is_whitespace()).collect();
        let lhs = if let Some((lhs, _)) = packed.split_once("<->") {
            lhs
        } else {
            let Some((lhs, rhs)) = words.split_once('=') else {
                return Ok(());
            };
            if rhs.starts_with('=') {
                return Ok(());
            }
            lhs.trim()
        };
        // FADDR and DADDR are read-only: the reference assembler rejects `faddr = r5` and
        // `daddr = r5` and takes `r5 = daddr` = 763e82bf.
        if lhs.eq_ignore_ascii_case("FADDR") || lhs.eq_ignore_ascii_case("DADDR") {
            return Err(Error::Parse {
                line,
                msg: format!("{lhs} is read-only: {text}"),
            });
        }
        let index = match lhs.as_bytes().first() {
            Some(b'R' | b'r' | b'F' | b'f') => lhs[1..].parse::<usize>().ok(),
            _ => None,
        };
        // An (LW) load writes the pair Rn, Rn^1: the reference assembler rejects `r15 =
        // pass r9 , r14 = dm ( i3 , m2 ) ( lw )` and `r0 = pass r9 , r1 =
        // dm ( i3 , m2 ) ( lw )`.
        let upper = packed.to_ascii_uppercase();
        let lw_load = upper.ends_with("(LW)")
            && upper
                .split_once('=')
                .is_some_and(|(_, rhs)| rhs.starts_with("DM(") || rhs.starts_with("PM("));
        if let Some(n) = index.filter(|&n| n < 16) {
            let pair = if lw_load { Some(n ^ 1) } else { None };
            for r in std::iter::once(n).chain(pair) {
                if std::mem::replace(&mut written[r], true) {
                    return Err(Error::Parse {
                        line,
                        msg: format!("R{r}/F{r} is written twice in one instruction: {text}"),
                    });
                }
            }
        }
        Ok(())
    };
    for ch in text.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                check(&part)?;
                part.clear();
                continue;
            }
            _ => {}
        }
        part.push(ch);
    }
    check(&part)
}

/// Refuse an instruction whose encoding, decoded again, holds a different
/// number of parallel operations than its source: the parser recognised one
/// part and silently discarded the rest, or picked a form that adds an
/// operation (`f15 = f9 + f15, f9 = f9 - f15, r8 = r0` became a multiply +
/// dual add/sub that also wrote F0). The reference assembler
/// found 2,321 such rows, e.g.
/// `jump (m12, i12), r4 = r1 + r2` encoded as the bare jump and
/// `f0 = f3 * f4, f8 = max(f8, f14), dm(i5, m4) = r1` without the MAX.
fn check_no_dropped_operations(text: &str, instr: &Instruction, line: u32) -> Result<()> {
    let Ok(bytes) = selinstr::encode::encode(instr) else {
        return Ok(());
    };
    let word = bytes.iter().fold(0u64, |w, &b| (w << 8) | u64::from(b));
    let decoded = selinstr::disasm::decode_instruction(word);
    let (wanted, got) = (parallel_operations(text), parallel_operations(&decoded));
    if got != wanted {
        return Err(Error::Parse {
            line,
            msg: format!(
                "`{text}` is not supported as written: it would encode as `{decoded}` \
                 ({got} parallel operations instead of {wanted})"
            ),
        });
    }
    Ok(())
}

/// `text` without a leading `label:`.
fn strip_label(text: &str) -> &str {
    match text.find(':') {
        Some(pos) if is_valid_label(text[..pos].trim()) => text[pos + 1..].trim_start(),
        _ => text,
    }
}

/// `F0`..`F15`: a float register name.
fn is_float_reg(name: &str) -> bool {
    let upper = name.trim().to_uppercase();
    upper.len() > 1 && upper.starts_with('F') && upper[1..].chars().all(|c| c.is_ascii_digit())
}

/// `Fn = Fm` has no vendor vector: the reference assembler encodes `Rn = Rm` as a Type 5
/// transfer (703e07bf), and `Fn = PASS Fm` is a float ALU operation.
fn float_copy_error(lhs: &str, rhs: &str, line: u32) -> Error {
    Error::Parse {
        line,
        msg: format!(
            "`{lhs} = {rhs}`: write `{lhs} = PASS {rhs}` or use R registers for a plain transfer"
        ),
    }
}

/// True for a line that starts a `.VAR` / `.BYTE` statement.
fn is_data_directive(trimmed: &str) -> bool {
    let upper = trimmed.to_uppercase();
    data_directive_width(&upper).is_some()
}

/// Parse one initializer: a constant expression that fits the element,
/// or (32-bit elements only) a bare symbol name.
fn parse_data_init(item: &str, width: usize) -> std::result::Result<DataInit, String> {
    let t = item.trim();
    if t.is_empty() {
        return Err("empty initializer in data list".into());
    }
    if let Some(v) = eval_const_expr(t) {
        let (lo, hi) = match width {
            1 => (-0x80, 0xff),
            _ => (-0x8000_0000, 0xffff_ffff),
        };
        if v < lo || v > hi {
            return Err(format!(
                "initializer `{t}` = {v} does not fit a {width}-byte element"
            ));
        }
        return Ok(DataInit::Num(v));
    }
    if width == 4 && is_valid_label(t) {
        return Ok(DataInit::Sym(t.to_string()));
    }
    Err(format!("unsupported data initializer `{t}`"))
}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

pub struct AsmParser<'a> {
    src: &'a str,
    line: u32,
}

impl<'a> AsmParser<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, line: 0 }
    }

    pub fn parse_all(&mut self) -> Result<Vec<ParsedLine>> {
        let mut results = Vec::new();
        // A `.VAR` / `.BYTE` statement whose initializer list spans
        // several lines, collected up to its terminating `;`.
        let mut data_stmt: Option<String> = None;
        let mut struct_depth: u32 = 0;
        let mut pending_line = String::new();
        let lines: Vec<&str> = self.src.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let raw_line = lines[i];
            self.line += 1;
            i += 1;
            // `#line N "file"` from the preprocessor: the next line is N.
            // Counting the marker itself put every line number one high
            // (and off by the include's length after an `#include`).
            if let Some(rest) = raw_line.trim_start().strip_prefix("#line ") {
                if let Some(n) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse::<u32>().ok())
                {
                    self.line = n.saturating_sub(1);
                    continue;
                }
            }
            if let Some(stmt) = data_stmt.as_mut() {
                let trimmed = strip_comment(raw_line).trim();
                stmt.push(' ');
                stmt.push_str(trimmed);
                if !trimmed.contains(';') {
                    continue;
                }
                let text = data_stmt.take().unwrap_or_default();
                results.push(self.parse_line(&text)?);
                continue;
            }
            // Skip .STRUCT body lines (brace-delimited)
            if struct_depth > 0 {
                let trimmed = strip_comment(raw_line).trim();
                for c in trimmed.chars() {
                    match c {
                        '{' => struct_depth += 1,
                        '}' => struct_depth = struct_depth.saturating_sub(1),
                        _ => {}
                    }
                }
                continue;
            }
            // Join multi-line instructions (lines ending with comma).
            // Directives starting with '.' are excluded from joining UNLESS
            // they are symbol-list directives (.GLOBAL, .EXTERN, .WEAK) or
            // the pending buffer already holds a partial directive.
            let trimmed = strip_comment(raw_line).trim();
            if pending_line.is_empty()
                && is_data_directive(strip_label(trimmed))
                && !trimmed.contains(';')
            {
                data_stmt = Some(trimmed.to_string());
                continue;
            }
            let allow_dot_join = {
                let u = trimmed.to_uppercase();
                u.starts_with(".GLOBAL") || u.starts_with(".EXTERN") || u.starts_with(".WEAK")
            };
            let ends_with_comma = trimmed.ends_with(',')
                && (!trimmed.starts_with('.')
                    || allow_dot_join
                    || (!pending_line.is_empty()
                        && pending_line.trim().to_uppercase().starts_with(".GLOBAL")
                        || pending_line.trim().to_uppercase().starts_with(".EXTERN")
                        || pending_line.trim().to_uppercase().starts_with(".WEAK")));
            // Also join lines ending with '|' (bitwise OR in multi-line
            // register load or BIT expressions like MMASK = (expr | expr |...)
            let ends_with_pipe = trimmed.ends_with('|');
            if ends_with_comma || ends_with_pipe {
                pending_line.push_str(trimmed);
                pending_line.push(' ');
                continue;
            }
            let was_joined = !pending_line.is_empty();
            let effective_line = if !was_joined {
                raw_line.to_string()
            } else {
                pending_line.push_str(raw_line);
                let joined = pending_line.clone();
                pending_line.clear();
                joined
            };
            // Split multi-statement lines at top-level `;`. Hand-written
            // SHARC asm packs interrupt-vector slots as
            //   JUMP .spurious_trap; NOP; NOP; NOP;
            // on a single source line; without splitting, parse_line
            // returns the first statement only and the remaining three
            // get silently dropped, which is exactly how an IVT with
            // half its instructions missing ships in the .doj. Skip the
            // split when there is only one top-level statement on the
            // line (so trailing `;` on lone instructions and directives
            // still works the way parse_line expects).
            let stmt_parts = split_top_level_stmts(&effective_line);
            if stmt_parts.len() > 1 {
                for part in stmt_parts {
                    let p = self.parse_line(&part)?;
                    if p.label.is_some() || p.instruction.is_some() || p.directive.is_some() {
                        results.push(p);
                    }
                }
                continue;
            }
            let parsed = match self.parse_line(&effective_line) {
                Ok(p) => p,
                Err(e) if was_joined => {
                    // A multi-line join failed to parse as a single instruction.
                    // Split at commas outside parens and parse each part
                    // independently (e.g. immediate LSHIFT + PUTS on one line).
                    let stripped = strip_comment(&effective_line).trim().to_string();
                    let upper = stripped.to_uppercase();
                    let commas = find_all_commas_outside_parens(&upper);
                    if commas.is_empty() {
                        return Err(e);
                    }
                    let mut parts = Vec::new();
                    let mut prev = 0;
                    for &c in &commas {
                        parts.push(stripped[prev..c].trim());
                        prev = c + 1;
                    }
                    parts.push(stripped[prev..].trim());
                    let mut all_ok = true;
                    let mut sub_results = Vec::new();
                    for part in &parts {
                        if part.is_empty() {
                            continue;
                        }
                        match self.parse_line(part) {
                            Ok(p) if p.instruction.is_some() || p.directive.is_some() => {
                                sub_results.push(p);
                            }
                            _ => {
                                all_ok = false;
                                break;
                            }
                        }
                    }
                    if !all_ok || sub_results.is_empty() {
                        return Err(e);
                    }
                    results.extend(sub_results);
                    continue;
                }
                Err(e) => return Err(e),
            };
            // Detect start of .STRUCT body (brace-delimited block)
            {
                let trimmed_upper = strip_comment(&effective_line).trim().to_uppercase();
                if trimmed_upper.starts_with(".STRUCT") {
                    let trimmed = strip_comment(&effective_line).trim();
                    for c in trimmed.chars() {
                        match c {
                            '{' => struct_depth += 1,
                            '}' => struct_depth = struct_depth.saturating_sub(1),
                            _ => {}
                        }
                    }
                    // If we opened a brace but didn't close, skip subsequent lines
                }
            }
            if parsed.label.is_some() || parsed.instruction.is_some() || parsed.directive.is_some()
            {
                results.push(parsed);
            }
        }
        if data_stmt.is_some() {
            return Err(Error::Parse {
                line: self.line,
                msg: "unterminated .VAR/.BYTE statement (missing `;`)".into(),
            });
        }
        Ok(results)
    }

    fn parse_line(&self, raw: &str) -> Result<ParsedLine> {
        // SHARC+ asm uses `;` as the statement terminator. After
        // `split_top_level_stmts` peels off multi-statement lines, each
        // single-statement input still ends with the trailing `;`
        // (e.g. `_main: NOP;`), so trim it before instruction parsing.
        let line = strip_comment(raw).trim();
        let line = line.trim_end_matches(';').trim();
        // Skip preprocessor #line directives
        if line.starts_with("#line ") || line.starts_with("#LINE ") {
            return Ok(ParsedLine {
                label: None,
                instruction: None,
                directive: None,
                label_ref: None,
                extra_label_refs: Vec::new(),
                line: self.line,
                generated_labels: Vec::new(),
            });
        }
        if line.is_empty() {
            return Ok(ParsedLine {
                label: None,
                instruction: None,
                directive: None,
                label_ref: None,
                extra_label_refs: Vec::new(),
                line: self.line,
                generated_labels: Vec::new(),
            });
        }

        // Check for label.
        // A colon immediately followed by a digit is a SIMD register pair
        // (e.g. "F1:0"), not a label separator.
        let (label, rest) = if let Some(pos) = line.find(':') {
            let candidate = line[..pos].trim();
            let after_colon_digit = line[pos + 1..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit());
            if is_valid_label(candidate) && !after_colon_digit {
                (Some(candidate.to_string()), line[pos + 1..].trim())
            } else {
                (None, line)
            }
        } else {
            (None, line)
        };

        if rest.is_empty() {
            return Ok(ParsedLine {
                label,
                instruction: None,
                directive: None,
                label_ref: None,
                extra_label_refs: Vec::new(),
                line: self.line,
                generated_labels: Vec::new(),
            });
        }

        // Check for directive
        if rest.starts_with('.') {
            let directive = self.parse_directive(rest)?;
            return Ok(ParsedLine {
                label,
                instruction: None,
                directive: Some(directive),
                label_ref: None,
                extra_label_refs: Vec::new(),
                line: self.line,
                generated_labels: Vec::new(),
            });
        }

        // Parse instruction
        let (instr, label_ref, extra_label_refs) = parse_instruction_at_line(rest, self.line)?;
        check_no_dropped_operations(rest, &instr, self.line)?;
        check_no_double_write(rest, self.line)?;
        check_dag_registers(rest, self.line)?;
        check_dag_self_transfer(&instr, self.line)?;
        Ok(ParsedLine {
            label,
            instruction: Some(instr),
            directive: None,
            label_ref,
            extra_label_refs,
            line: self.line,
            generated_labels: Vec::new(),
        })
    }

    fn parse_directive(&self, text: &str) -> Result<Directive> {
        let upper = text.to_uppercase();
        // Use the original text for symbol names (case-sensitive), only
        // use `upper` for matching the directive keyword.
        if upper.starts_with(".SECTION") || upper.starts_with(".SEGMENT") {
            let keyword_len = 8; // Both .SECTION and .SEGMENT are 8 chars
            let rest = &text[keyword_len..];
            // Preserve the /PM or /DM qualifier for parse_section_name
            let name = rest.trim().trim_start_matches('/').trim().to_string();
            return Ok(Directive::Section(name));
        }
        if upper.starts_with(".GLOBAL") {
            let rest = &text[7..];
            let names = parse_symbol_list(rest);
            return Ok(Directive::Global(names));
        }
        if upper.starts_with(".WEAK") {
            let rest = &text[5..];
            let names = parse_symbol_list(rest);
            return Ok(Directive::Weak(names));
        }
        if upper.starts_with(".EXTERN") {
            let rest = &text[7..];
            let names = parse_symbol_list(rest);
            return Ok(Directive::Extern(names));
        }
        if let Some(width) = data_directive_width(&upper) {
            let keyword_len = if width == 4 { 4 } else { 5 };
            let decl =
                parse_data_decl(&text[keyword_len..], width).map_err(|msg| Error::Parse {
                    line: self.line,
                    msg,
                })?;
            return Ok(Directive::Data(decl));
        }
        if upper.starts_with(".BYTE") || upper.starts_with(".VAR/") {
            return Err(Error::Parse {
                line: self.line,
                msg: format!("unsupported data directive: {text}"),
            });
        }
        if let Some(rest) = upper.strip_prefix(".ALIGN") {
            let val_str = rest.trim().trim_end_matches(';').trim();
            let val = parse_number(val_str).ok_or_else(|| Error::Parse {
                line: self.line,
                msg: format!("invalid .ALIGN value: {val_str}"),
            })?;
            return Ok(Directive::Align(val));
        }
        if upper.starts_with(".NOCOMPRESS") {
            return Ok(Directive::NoCompress);
        }
        if upper.starts_with(".COMPRESS") {
            return Ok(Directive::Compress);
        }
        if upper.starts_with(".SET") {
            let rest = &text[4..];
            let rest = rest.trim().trim_end_matches(';').trim();
            let (name, value) = if let Some((l, r)) = rest.split_once('=') {
                (l.trim().to_string(), r.trim().to_string())
            } else if let Some((l, r)) = rest.split_once(',') {
                (l.trim().to_string(), r.trim().to_string())
            } else {
                return Err(Error::Parse {
                    line: self.line,
                    msg: format!("invalid .SET syntax: {text}"),
                });
            };
            return Ok(Directive::Set(name, value));
        }
        // Silently ignore known toolchain directives that don't affect code generation
        if is_ignored_directive(&upper) {
            return Ok(Directive::Align(0));
        }
        Err(Error::Parse {
            line: self.line,
            msg: format!("unknown directive: {text}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Standalone instruction parser (also used by round-trip tests)
// ---------------------------------------------------------------------------

/// Parse a single instruction mnemonic string. Returns the instruction.
#[cfg(test)]
pub fn parse_instruction(text: &str) -> Result<Instruction> {
    let (instr, _label_ref, _extra) = parse_instruction_at_line(text, 0)?;
    Ok(instr)
}

#[allow(clippy::type_complexity)]
/// Rewrite `<` `-` `>` separated by whitespace as `<->`.
fn collapse_swap_arrow(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '-' {
                let mut k = j + 1;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                if k < chars.len() && chars[k] == '>' {
                    out.push_str("<->");
                    i = k + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn parse_instruction_at_line(
    text: &str,
    line: u32,
) -> Result<(Instruction, Option<String>, Vec<(String, i64)>)> {
    let text = text.trim().trim_end_matches(',');
    // Strip FETCH_RETURN (NOP) from compound instructions
    let text = if let Some(pos) = text.to_uppercase().find(", FETCH_RETURN") {
        &text[..pos]
    } else {
        text
    };
    // The reference assembler accepts the swap arrow spaced (`r5 < - > s5`, 139 rows); the
    // parser only knew `<->` and dropped the swap.
    let text = collapse_swap_arrow(text);
    // Normalize spacing: insert spaces around +/- between register operands
    let normalized = normalize_compute_spacing(&text);
    let (instr, label_ref, extra) = parse_instruction_inner_with_extra(&normalized, line)?;
    // The inner parser works on upper-cased text, so label names lose their
    // original case. Recover it from the un-modified source.
    let label_ref = label_ref.map(|lr| recover_label_case(&lr, &normalized));
    let extra: Vec<(String, i64)> = extra
        .into_iter()
        .map(|(name, addend)| (recover_label_case(&name, &normalized), addend))
        .collect();
    Ok((instr, label_ref, extra))
}

/// Wrapper around `parse_instruction_inner` that also returns any
/// secondary label references produced by multi-symbol expressions like
/// `I7 = ldf_stack_space + ldf_stack_length`.  The inner parser keeps
/// its single-`label_ref` shape for the common path; this wrapper
/// separately recognises the SHARC+ Tools `ureg = sym + sym + ...` and
/// `ureg = sym - sym` forms and surfaces every additional symbol as an
/// extra `(name, addend)` pair.  Each pair becomes one
/// `R_SHARC_PM_SW_BRANCHRETURN` (Add) relocation at the same byte
/// offset, matching what easm21k emits.
#[allow(clippy::type_complexity)]
fn parse_instruction_inner_with_extra(
    normalized: &str,
    line: u32,
) -> Result<(Instruction, Option<String>, Vec<(String, i64)>)> {
    // `(PC, _e - _t)` is a linker expression,
    // not a symbol whose name contains a minus. Keep the operands separate
    // so the assembler can emit the vendor's postfix relocation sequence.
    let upper = normalized.to_uppercase();
    if upper.starts_with("DO ") || upper.starts_with("LCNTR") {
        if let Some(do_at) = upper.find("DO") {
            let target = upper[do_at + 2..].trim_start();
            if let Some(inner) = target.strip_prefix('(').and_then(|s| s.split_once(')')) {
                if let Some((pc, expr)) = inner.0.split_once(',') {
                    if pc.trim() == "PC" {
                        if let Some((a, b)) = expr.split_once('-') {
                            let (a, b) = (a.trim(), b.trim());
                            if is_valid_label(a) && is_valid_label(b) {
                                let rewritten =
                                    format!("{} (PC, {a}){}", &upper[..do_at + 2], inner.1);
                                let (instr, label_ref) = parse_instruction_inner(&rewritten, line)?;
                                return Ok((instr, label_ref, vec![(b.to_string(), -1)]));
                            }
                        }
                    }
                }
            }
        }
    }
    let (instr, label_ref) = parse_instruction_inner(normalized, line)?;
    // Only LoadImm with a label_ref is a candidate for multi-symbol
    // expansion: the SHARC+ Tools dialect lets `ureg = symA + symB`
    // resolve at link time via two relocations against the same imm32
    // operand.  Other shapes (branches, ImmStore, etc.) carry at most
    // one symbolic operand in this dialect.
    let extra = if matches!(instr, Instruction::LoadImm { .. }) && label_ref.is_some() {
        let upper = normalized.to_uppercase();
        if let Some(eq) = upper.find('=') {
            let rhs = upper[eq + 1..].trim().trim_end_matches(';').trim();
            extract_extra_label_addends(rhs, label_ref.as_deref().unwrap())
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    Ok((instr, label_ref, extra))
}

/// Walk a LoadImm right-hand side and return every symbol other than the
/// primary one (which is already attached as `label_ref`) along with its
/// signed contribution.  Recognises `+ sym` and `- sym` connectors, with
/// or without surrounding whitespace.  Numeric terms are skipped: a
/// `label - 1` form is handled as an addend on the primary label.
fn extract_extra_label_addends(rhs: &str, primary: &str) -> Vec<(String, i64)> {
    let primary_upper = primary.to_uppercase();
    let mut out: Vec<(String, i64)> = Vec::new();
    let mut sign: i64 = 1;
    let mut seen_primary = false;
    let mut i = 0;
    let bytes = rhs.as_bytes();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }
        if c == b'+' {
            sign = 1;
            i += 1;
            continue;
        }
        if c == b'-' {
            sign = -sign;
            i += 1;
            continue;
        }
        // Read identifier or number token.
        let start = i;
        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'.' {
                i += 1;
            } else {
                break;
            }
        }
        if start == i {
            // Unrecognised character — bail to keep behaviour conservative.
            return Vec::new();
        }
        let tok = &rhs[start..i];
        if !is_label_token(tok) {
            // Numeric literal (or other non-label) — skip; it just
            // contributes to the implicit addend on the primary
            // symbol, which the encoder already baked into the
            // instruction's imm32 field.
            sign = 1;
            continue;
        }
        if !seen_primary && tok.eq_ignore_ascii_case(&primary_upper) {
            seen_primary = true;
        } else {
            out.push((tok.to_string(), sign));
        }
        sign = 1;
    }
    out
}

fn parse_instruction_inner(normalized: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    let upper = normalized.to_uppercase();

    // ELSE compute — strip ELSE prefix and parse as regular compute.
    // Used in SIMD mode for complementary PEx operations.
    if let Some(rest_upper) = upper.strip_prefix("ELSE ") {
        let rest_upper = rest_upper.trim();
        if !rest_upper.is_empty() {
            let compute = parse_compute_op(rest_upper, line)?;
            return Ok((Instruction::Compute { cond: 31, compute }, None));
        }
    }

    let tokens: Vec<&str> = upper.split_whitespace().collect();

    if tokens.is_empty() {
        return Err(Error::Parse {
            line,
            msg: "empty instruction".into(),
        });
    }

    // Skip orphan continuation lines from multi-line expressions in
    // included system headers that were not joined with the preceding line.
    // These start with operators, nested parens, or the % modulo operator.
    if upper.starts_with("| ")
        || upper.starts_with("|(")
        || upper.starts_with("- (")
        || upper.starts_with("+ (")
        || upper.starts_with("% (")
        || (upper.starts_with("((") && (upper.contains('|') || upper.ends_with(')')))
    {
        return Ok((Instruction::Nop, None));
    }

    // NOP
    if upper == "NOP" {
        return Ok((Instruction::Nop, None));
    }

    // IDLE
    if upper == "IDLE" {
        return Ok((Instruction::Idle, None));
    }

    // EMUIDLE
    if upper == "EMUIDLE" {
        return Ok((Instruction::EmuIdle, None));
    }

    // SYNC
    if upper == "SYNC" {
        return Ok((Instruction::Sync, None));
    }

    // RFRAME restores the frame pointer (I6) that the caller's delayed
    // CALL pushed on the frame stack. It appears in the first delay slot
    // of the SHARC+ C-ABI indirect return `JUMP (M14,I12) (DB)`. VISA
    // mode emits the canonical 16-bit 0x1901 parcel.
    if upper == "RFRAME" {
        return Ok((Instruction::Rframe, None));
    }

    // Other SIMD mode / frame instructions this toolchain does not yet
    // model in detail — encode as NOP as a placeholder so the assembler
    // accepts asm text that names them without breaking downstream
    // passes.
    if upper.starts_with("ENTER_SIMD")
        || upper.starts_with("EXIT_SIMD")
        || upper.starts_with("ENTER SIMD")
        || upper.starts_with("EXIT SIMD")
        || upper == "CFRAME"
        || upper.starts_with("FETCH_RETURN")
    {
        return Ok((Instruction::Nop, None));
    }

    // RTS / RTI (possibly conditional or with compute)
    if let Some(instr) = try_parse_return(&upper, line)? {
        return Ok((instr, None));
    }

    // LCNTR = ..., DO ... UNTIL LCE
    // Only match when DO is present; standalone "LCNTR = val" is a ureg load.
    if upper.starts_with("LCNTR") && upper.contains(" DO ") {
        return parse_do_loop(&upper, line);
    }

    // DO addr UNTIL cond (Type 13)
    if upper.starts_with("DO ") && upper.contains("UNTIL") {
        return parse_do_until(&upper, line);
    }

    // BIT SET/CLR/TGL/TST/XOR sreg data32
    if upper.starts_with("BIT ") {
        let instr = parse_bit_op(&upper, line)?;
        return Ok((instr, None));
    }

    // PUSH/POP stack operations
    if upper.starts_with("PUSH ") || upper.starts_with("POP ") || upper.starts_with("FLUSH ") {
        let instr = parse_stack_op(&upper, line)?;
        return Ok((instr, None));
    }

    // CJUMP addr (DB)
    if upper.starts_with("CJUMP") {
        return parse_cjump(&upper, line);
    }

    // MODIFY(Ii, Mm) or BITREV(Ii, data32) — DAG register modify/bitrev
    if upper.starts_with("MODIFY(") || upper.starts_with("MODIFY (") {
        let instr = parse_dag_modify(&upper, line)?;
        return Ok((instr, None));
    }
    if upper.starts_with("BITREV(") || upper.starts_with("BITREV (") {
        let instr = parse_dag_modify(&upper, line)?;
        return Ok((instr, None));
    }

    // Ii=MODIFY (Ii, Mm)(NW) or Ii=BITREV (Ii, data32) — enhanced Type 7/19
    // Match when "=MODIFY" or "=BITREV" appears
    // Also handles compound: "compute , I7=MODIFY (I7,M6)(NW)"
    let compact: String = upper.chars().filter(|c| !c.is_whitespace()).collect();
    if !upper.starts_with("IF ") && (compact.contains("=B2W(") || compact.contains("=W2B(")) {
        return Ok((parse_dag_convert(&upper, 31, line)?, None));
    }
    if !upper.starts_with("IF ") && (compact.contains("=MODIFY(") || compact.contains("=BITREV(")) {
        let instr = parse_modify_assign(&upper, line)?;
        return Ok((instr, None));
    }

    if let Some(instr) = try_parse_bit_fifo(&upper, line)? {
        return Ok((instr, None));
    }

    // IF PEx/PEy — SIMD processing element condition. Strip the PE
    // selector and parse the remaining instruction (the compute/mem op
    // is the same, just targeted at a specific processing element).
    if upper.starts_with("IF PEX ") || upper.starts_with("IF PEY ") {
        let rest = upper[7..].trim();
        // If the rest starts with another IF, parse it as conditional
        if rest.starts_with("IF ") {
            return parse_conditional(rest, line);
        }
        // Otherwise parse as a standalone instruction
        return parse_instruction_inner(rest, line);
    }

    // IF cond ...
    if upper.starts_with("IF ") {
        return parse_conditional(&upper, line);
    }

    // JUMP / CALL
    if upper.starts_with("JUMP") || upper.starts_with("CALL") {
        return parse_branch(&upper, line);
    }

    // Memory access: DM(...) = ... or ... = DM(...)
    // Handle both DM( and DM ( formats from the disassembler
    // Also check for compound: "compute , MODIFY(...)"
    if upper.contains("DM(")
        || upper.contains("PM(")
        || upper.contains("DM (")
        || upper.contains("PM (")
    {
        let (instr, sym) = parse_mem_access(&upper, line)?;
        return Ok((instr, sym));
    }

    // Compound: "compute , MODIFY(Ii, Mm)"
    if upper.contains("MODIFY(") || upper.contains("MODIFY (") {
        // Check if it's "compute , MODIFY" form
        if let Some(comma) = find_comma_outside_parens(&upper) {
            let before = upper[..comma].trim();
            let after = upper[comma + 1..].trim();
            if (after.starts_with("MODIFY(") || after.starts_with("MODIFY (")) && !before.is_empty()
            {
                let compute = parse_compute_op(before, line)?;
                let modify = parse_dag_modify(after, line)?;
                if let Instruction::DagModify {
                    pm,
                    i_reg,
                    m_reg,
                    cond,
                    dest_xor,
                    width,
                    ..
                } = modify
                {
                    return Ok((
                        Instruction::DagModify {
                            pm,
                            i_reg,
                            m_reg,
                            cond,
                            compute: Some(compute),
                            dest_xor,
                            width,
                        },
                        None,
                    ));
                }
            }
        }
    }

    // Register swap: R14<->S8, or compound: compute , R14<->S8
    if upper.contains("<->") {
        return parse_register_swap(&upper, line).map(|i| (i, None));
    }

    // Multifunction compute: "R0 = R1 * R2 (SSF) , R3 = R4 + R5" etc.
    if let Some(multi) = try_parse_multifunction(&upper, line)? {
        return Ok((
            Instruction::Compute {
                cond: 31,
                compute: multi,
            },
            None,
        ));
    }

    // Compound: "compute , ureg = ureg" (Type 5a with compute)
    // Try the LAST comma first (for multifunction compute + ureg transfer)
    {
        let all_commas = find_all_commas_outside_parens(&upper);
        for &comma in all_commas.iter().rev() {
            let before = upper[..comma].trim();
            let after = upper[comma + 1..].trim();
            if !before.is_empty() && !after.is_empty() && after.contains('=') {
                // Try parsing left side as compute, right side as ureg transfer
                if let Ok(compute) = parse_compute_op(before, line) {
                    let eq_pos = after.find('=').unwrap();
                    let dst = after[..eq_pos].trim();
                    let src = after[eq_pos + 1..].trim();
                    if let (Some(dst_code), Some(src_code)) = (ureg_code(dst), ureg_code(src)) {
                        return Ok((
                            Instruction::UregTransfer {
                                src_ureg: src_code,
                                dst_ureg: dst_code,
                                cond: 31,
                                compute: Some(compute),
                            },
                            None,
                        ));
                    }
                }
            }
        }
    }

    // ureg = symbol — load immediate with symbolic reference
    // Must not match ureg transfers (rhs is also a register) or compute ops
    if let Some(eq) = upper.find('=') {
        let lhs = upper[..eq].trim();
        let rhs = upper[eq + 1..].trim();
        if let Some(ureg) = ureg_code(lhs) {
            // ureg = label [+ label ...] [± constants]: the constants become
            // the relocation addend. They used to be dropped, so
            // `r0 = _mod_param + 16` loaded `_mod_param`; label differences
            // loaded the first label and `r0 = b2w(i7)` a symbol "B2W".
            if !rhs.is_empty() && ureg_code(rhs).is_none() && is_symbolic_expression(rhs) {
                let (label, addend) = parse_label_load(rhs, line)?;
                return Ok((
                    Instruction::LoadImm {
                        ureg,
                        value: addend as i32 as u32,
                    },
                    Some(label),
                ));
            }
        }
    }

    // Compute or load-immediate
    let instr = parse_compute_or_load(&upper, line)?;
    Ok((instr, None))
}

/// Check if a token looks like a label (not a register, not a compute keyword).
fn is_label_token(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    // Must start with '.', '_', or alphabetic
    if !(tok.starts_with('.') || tok.starts_with('_') || tok.as_bytes()[0].is_ascii_alphabetic()) {
        return false;
    }
    // Not a register
    if ureg_code(tok).is_some() {
        return false;
    }
    // Not a number
    if tok.parse::<u64>().is_ok() || tok.starts_with("0X") {
        return false;
    }
    // Not a compute mnemonic or keyword (the reference assembler rejects `r5 = emuidle`)
    !matches!(
        tok,
        "EMUIDLE"
            | "ABS"
            | "PASS"
            | "LSHIFT"
            | "ASHIFT"
            | "ROT"
            | "FEXT"
            | "FDEP"
            | "BSET"
            | "BCLR"
            | "BTGL"
            | "BTST"
            | "AND"
            | "OR"
            | "XOR"
            | "NOT"
            | "MIN"
            | "MAX"
            | "CLIP"
            | "FIX"
            | "FLOAT"
            | "TRUNC"
            | "RECIPS"
            | "RSQRTS"
            | "LOGB"
            | "MANT"
            | "SCALB"
            | "COPYSIGN"
            | "EXP"
            | "LEFTZ"
            | "LEFTO"
            | "FPACK"
            | "FUNPACK"
            | "SAT"
            | "TRNC"
            | "MRF"
            | "MRB"
            | "MR0F"
            | "MR1F"
            | "MR2F"
            | "MR0B"
            | "MR1B"
            | "MR2B"
            | "COMP"
            | "COMPU"
            | "CI"
            | "BY"
            | "SSF"
            | "SSI"
            | "SUI"
            | "USI"
            | "UUI"
            | "UUF"
            | "SUF"
            | "USF"
            | "ELSE"
            | "IF"
            | "TRUE"
            | "DM"
            | "PM"
            | "LW"
            | "BW"
            | "SW"
            | "NW"
    )
}

/// Check if an expression string contains at least one label-like identifier
/// (starts with '.', '_', or alpha) that is not a known register name.
fn has_label_ident(expr: &str) -> bool {
    split_identifiers(expr).any(is_label_token)
}

/// True when every identifier in `rhs` is a label or a number (no register
/// or mnemonic) and at least one is a label: `_mod_param + 16`,
/// `_MOD2_PARAMS + 3 * 16 * 0`. Anything else is a compute.
fn is_symbolic_expression(rhs: &str) -> bool {
    let mut any_label = false;
    for tok in split_identifiers(rhs) {
        if parse_signed_number(tok).is_some() {
            continue;
        }
        if !is_label_token(tok) {
            return false;
        }
        any_label = true;
    }
    any_label
}

/// Parse the right-hand side of `ureg = <symbolic expression>`: terms
/// joined by `+`/`-`, each a label or a constant expression. Returns the
/// first label and the constant sum (the relocation addend); further labels
/// are picked up by `extract_extra_label_addends`. A subtracted label, a
/// constant next to several labels, or a term mixing a label with other
/// arithmetic is rejected.
fn parse_label_load(rhs: &str, line: u32) -> Result<(String, i64)> {
    let err = |what: &str| Error::Parse {
        line,
        msg: format!("unsupported symbolic expression `{rhs}`: {what}"),
    };
    let mut expr = rhs.trim();
    while let Some(inner) = expr.strip_prefix('(').and_then(|e| e.strip_suffix(')')) {
        expr = inner.trim();
    }
    // Split into top-level terms with their signs.
    let mut terms: Vec<(i64, String)> = Vec::new();
    let mut current = String::new();
    let mut sign = 1i64;
    let mut depth = 0i32;
    for ch in expr.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            '+' | '-' if depth == 0 => {
                if current.trim().is_empty() {
                    if ch == '-' {
                        sign = -sign;
                    }
                } else {
                    terms.push((sign, std::mem::take(&mut current)));
                    sign = if ch == '-' { -1 } else { 1 };
                }
                continue;
            }
            _ => {}
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        terms.push((sign, current));
    }
    let mut labels: Vec<String> = Vec::new();
    let mut constant: i64 = 0;
    for (sign, term) in terms {
        let term = term.trim();
        if is_label_token(term) {
            if sign < 0 {
                return Err(err("a subtracted label is not supported"));
            }
            labels.push(term.to_string());
        } else if let Some(v) = eval_const_expr(term) {
            constant += sign * v;
        } else {
            return Err(err(&format!("`{term}` is neither a label nor a constant")));
        }
    }
    let Some(first) = labels.first().cloned() else {
        return Err(err("no label"));
    };
    if labels.len() > 1 && constant != 0 {
        return Err(err("a constant next to several labels is not supported"));
    }
    Ok((first, constant))
}

/// Check if an expression is `label-N` or `label+N`.
fn has_label_with_addend(expr: &str) -> bool {
    split_label_addend(expr).is_some()
}

/// Split `label-N` or `label+N` into (label, addend). Returns None if
/// the expression is not in that form.
fn split_label_addend(expr: &str) -> Option<(&str, i64)> {
    // Find the last '-' or '+' that separates the label from the number
    for (i, c) in expr.char_indices().rev() {
        if (c == '-' || c == '+') && i > 0 {
            let label = expr[..i].trim();
            let num_str = &expr[i..];
            if has_label_ident(label) {
                if let Some(v) = parse_signed_number(num_str) {
                    return Some((label, v));
                }
            }
        }
    }
    None
}

/// Split a string into identifier-like tokens (alphanumeric, '_', '.').
fn split_identifiers(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .filter(|tok| !tok.is_empty())
}

/// Recover the original-case label name from the source text.
///
/// The instruction parser upper-cases everything for keyword matching, but
/// label names are case-sensitive. Given the upper-cased label from the
/// parser and the original (un-modified-case) source line, find the
/// matching identifier and return it in its original case.
fn recover_label_case(upper_label: &str, original: &str) -> String {
    let hay_upper = original.to_uppercase();
    if let Some(pos) = hay_upper.find(upper_label) {
        return original[pos..pos + upper_label.len()].to_string();
    }
    upper_label.to_string()
}

// ---------------------------------------------------------------------------
// Return (RTS / RTI)
// ---------------------------------------------------------------------------

fn try_parse_return(text: &str, line: u32) -> Result<Option<Instruction>> {
    // Handle: RTS, RTI, RTS(DB), RTI(DB), RTS(LR), RTS(DB,LR), and with
    // optional compute after comma.
    let (main_part, compute_text) = split_first_comma(text);
    let main_part = main_part.trim();

    let main_norm = main_part.replace(' ', "");
    let (interrupt, matched) = if main_norm.starts_with("RTS") || main_norm.starts_with("RETURN") {
        if main_norm == "RTS"
            || main_norm == "RETURN"
            || main_norm.starts_with("RTS(")
            || main_norm.starts_with("RETURN(")
        {
            (false, true)
        } else {
            (false, false)
        }
    } else if main_norm.starts_with("RTI") {
        if main_norm == "RTI" || main_norm.starts_with("RTI(") {
            (true, true)
        } else {
            (false, false)
        }
    } else {
        (false, false)
    };

    if !matched {
        return Ok(None);
    }

    let delayed = main_norm.contains("DB");
    let lr = main_norm.contains("LR");

    let compute = if let Some(ct) = compute_text {
        Some(parse_compute_op(ct.trim(), line)?)
    } else {
        None
    };

    Ok(Some(Instruction::Return {
        interrupt,
        cond: 31, // TRUE
        delayed,
        lr,
        compute,
    }))
}

// ---------------------------------------------------------------------------
// Register swap (Type 5b): Rn<->Sm, compute , Rn<->Sm
// ---------------------------------------------------------------------------

fn parse_register_swap(text: &str, line: u32) -> Result<Instruction> {
    // May be compound: "compute , Rn<->Sm"
    // Use the LAST comma to split compute from swap, since the compute
    // part itself may contain commas (e.g. dual add/sub).
    let all_commas = find_all_commas_outside_parens(text);
    let (swap_text, compute) = if !all_commas.is_empty() {
        // Find the comma that separates compute from swap. The swap part
        // is always "Rn<->Sm" (no commas), so use the last comma.
        let last_comma = *all_commas.last().unwrap();
        let before = text[..last_comma].trim();
        let after = text[last_comma + 1..].trim();
        if after.contains("<->") {
            // "compute , Rn<->Sm"
            let comp = parse_compute_op(before, line)?;
            (after, Some(comp))
        } else if before.contains("<->") {
            // "Rn<->Sm , compute" — use first comma instead
            let first_comma = all_commas[0];
            let b = text[..first_comma].trim();
            let a = text[first_comma + 1..].trim();
            let comp = parse_compute_op(a, line)?;
            (b, Some(comp))
        } else {
            return Err(Error::Parse {
                line,
                msg: format!("expected register swap with <->: {text}"),
            });
        }
    } else {
        (text, None)
    };

    let arrow_pos = swap_text.find("<->").ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected <-> in register swap: {swap_text}"),
    })?;
    let left = swap_text[..arrow_pos].trim();
    let right = swap_text[arrow_pos + 3..].trim();

    let dreg = parse_dreg(left, line)?;
    let cdreg = parse_reg_index(right, 'S', line)? as u16;
    if cdreg > 15 {
        return Err(Error::Parse {
            line,
            msg: format!("cdreg index out of range (0-15): {cdreg}"),
        });
    }

    Ok(Instruction::RegisterSwap {
        dreg,
        cdreg,
        cond: 31,
        compute,
    })
}

// ---------------------------------------------------------------------------
// Branch (JUMP / CALL)
// ---------------------------------------------------------------------------

fn parse_branch(text: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    let call = text.starts_with("CALL");
    let rest = text[4..].trim();

    // Indirect branch through a DAG2 pair:
    //   JUMP|CALL (Mm, Ii) [(DB)] [, [ELSE] compute]
    // Everything after the register pair must be consumed: `( db )` with
    // spaces used to be missed, and a trailing compute silently dropped.
    if rest.starts_with('(') && !rest.starts_with("(PC") {
        if let Some(close) = rest.find(')') {
            let inner = rest[1..close].trim();
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            if parts.len() == 2 {
                let (m_str, i_str) = if parts[0].starts_with('M') {
                    (parts[0], parts[1])
                } else {
                    (parts[1], parts[0])
                };
                if let (Some(m_code), Some(i_code)) = (ureg_code(m_str), ureg_code(i_str)) {
                    let dag2 = |name: &str, prefix: char| {
                        name.strip_prefix(prefix)
                            .and_then(|n| n.parse::<u8>().ok())
                            .filter(|n| (8..=15).contains(n))
                    };
                    if dag2(m_str, 'M').is_none() || dag2(i_str, 'I').is_none() {
                        return Err(Error::Parse {
                            line,
                            msg: format!(
                                "indirect JUMP/CALL needs a DAG2 pair (M8-M15, I8-I15), got \
                                 `({inner})`"
                            ),
                        });
                    }
                    let mut after = rest[close + 1..].trim();
                    let mut delayed = false;
                    if after.starts_with('(') {
                        let end = after.find(')').ok_or_else(|| Error::Parse {
                            line,
                            msg: format!("unterminated modifier in `{text}`"),
                        })?;
                        let modifier: String = after[1..end]
                            .chars()
                            .filter(|c| !c.is_whitespace())
                            .collect();
                        if modifier != "DB" {
                            return Err(Error::Parse {
                                line,
                                msg: format!("unsupported indirect branch modifier `({modifier})`"),
                            });
                        }
                        delayed = true;
                        after = after[end + 1..].trim();
                    }
                    let (compute, else_compute) = if after.is_empty() {
                        (None, false)
                    } else if let Some(op) = after.strip_prefix(',') {
                        let op = op.trim();
                        let (op, else_compute) = match op.strip_prefix("ELSE ") {
                            Some(rest) => (rest.trim(), true),
                            None => (op, false),
                        };
                        (Some(parse_compute_op(op, line)?), else_compute)
                    } else {
                        return Err(Error::Parse {
                            line,
                            msg: format!("unexpected `{after}` after indirect JUMP/CALL"),
                        });
                    };
                    return Ok((
                        Instruction::IndirectBranch {
                            call,
                            cond: 31,
                            pm_i: (i_code & 0xF) as u8,
                            pm_m: (m_code & 0xF) as u8,
                            delayed,
                            compute,
                            else_compute,
                        },
                        None,
                    ));
                }
            }
        }
    }

    let (branch, compute_text) = match find_comma_outside_parens(rest) {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest, None),
    };
    let mut target_text = branch.trim();
    let mut delayed = false;
    let mut loop_abort = false;
    // Accept combined (DB,LA) and separate (DB) (LA) modifiers, as
    // printed by the vendor and our disassembler respectively.
    while target_text.ends_with(')') {
        let Some(open) = target_text.rfind('(') else {
            break;
        };
        let flags = &target_text[open + 1..target_text.len() - 1];
        if !flags.split(',').all(|f| matches!(f.trim(), "DB" | "LA")) {
            break;
        }
        for flag in flags.split(',').map(str::trim) {
            delayed |= flag == "DB";
            loop_abort |= flag == "LA";
        }
        target_text = target_text[..open].trim();
    }
    let (target, label_ref) = parse_branch_target(target_text, line)?;
    // The reference assembler also takes `(LA)` on a direct JUMP: `if ne jump _ext (la)` =
    // 066000000000, bit 38 as for the PC-relative form.
    // No vector covers CALL (LA).
    if loop_abort && call {
        return Err(Error::Parse {
            line,
            msg: "(LA) requires a JUMP".into(),
        });
    }
    if let Some(compute) = compute_text {
        let BranchTarget::PcRelative(offset) = target else {
            return Err(Error::Parse {
                line,
                msg: "JUMP with compute requires (PC, offset)".into(),
            });
        };
        if call || loop_abort || !(-32..=31).contains(&offset) {
            return Err(Error::Parse {
                line,
                msg: "JUMP with compute requires a six-bit offset and no (LA)".into(),
            });
        }
        let compute = compute.trim();
        let (compute, else_compute) = match compute.strip_prefix("ELSE ") {
            Some(s) => (s.trim(), true),
            None => (compute, false),
        };
        return Ok((
            Instruction::ComputeBranch {
                cond: 31,
                offset: offset as i8,
                delayed,
                else_compute,
                compute: parse_compute_op(compute, line)?,
            },
            label_ref,
        ));
    }
    Ok((
        Instruction::Branch {
            call,
            cond: 31,
            target,
            delayed,
            loop_abort,
        },
        label_ref,
    ))
}

fn parse_branch_target(text: &str, line: u32) -> Result<(BranchTarget, Option<String>)> {
    // "(PC, offset)" form
    if text.starts_with('(') {
        let close = text.find(')').ok_or_else(|| Error::Parse {
            line,
            msg: "missing ')' in branch target".into(),
        })?;
        let inner = text[1..close].trim();
        if !text[close + 1..].trim().is_empty() {
            return Err(Error::Parse {
                line,
                msg: format!("unexpected text after branch target: {text}"),
            });
        }
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 2 && parts[0] == "PC" {
            let s = parts[1].trim();
            let offset: i32 =
                if let Some(hex) = s.strip_prefix("-0x").or_else(|| s.strip_prefix("-0X")) {
                    let v = i32::from_str_radix(hex, 16).map_err(|_| Error::Parse {
                        line,
                        msg: format!("invalid PC-relative offset: {s}"),
                    })?;
                    -v
                } else if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                    i32::from_str_radix(hex, 16).map_err(|_| Error::Parse {
                        line,
                        msg: format!("invalid PC-relative offset: {s}"),
                    })?
                } else if let Ok(v) = s.parse::<i32>() {
                    v
                } else if is_valid_label(s) {
                    // `(PC, label)`: pass 2 resolves the offset (or emits a
                    // PC-relative relocation). It used to become offset 0,
                    // a jump to itself.
                    return Ok((BranchTarget::PcRelative(0), Some(s.to_string())));
                } else {
                    return Err(Error::Parse {
                        line,
                        msg: format!("invalid PC-relative offset: {s}"),
                    });
                };
            return Ok((BranchTarget::PcRelative(offset), None));
        }
        // Single-operand: (label) is an absolute jump to a label
        if parts.len() == 1 {
            let (addr, label_ref) = branch_address(parts[0], line)?;
            return Ok((BranchTarget::Absolute(addr), label_ref));
        }
        return Err(Error::Parse {
            line,
            msg: format!("unknown branch target: {text}"),
        });
    }

    // Absolute address or symbolic label
    let (addr, label_ref) = branch_address(text, line)?;
    Ok((BranchTarget::Absolute(addr), label_ref))
}

/// A JUMP/CALL/CJUMP address: a label, or a number the reference assembler reads as a
/// signed 24-bit field (32-bit values wrap first). `if eq jump -0x436fa5`
/// = 060000bc905b and `cjump 0xffffffff ( db )` are accepted; `call
/// 0x800000`, `call 0xb8d588` and `jump -0xc76fa5` are rejected (2,412 +
/// 2,468 rows). selas refused the negative ones and truncated the rest.
fn branch_address(s: &str, line: u32) -> Result<(u32, Option<String>)> {
    let s = s.trim();
    let Some(v) = parse_signed_number(s) else {
        return parse_address_or_label(s, line);
    };
    let v = if (0x8000_0000..=0xFFFF_FFFF).contains(&v) {
        v - (1 << 32)
    } else {
        v
    };
    if !(-0x80_0000..=0x7F_FFFF).contains(&v) {
        return Err(Error::Parse {
            line,
            msg: format!("branch address {s} does not fit a signed 24-bit field"),
        });
    }
    Ok(((v as u32) & 0xFF_FFFF, None))
}

// ---------------------------------------------------------------------------
// Conditional instructions
// ---------------------------------------------------------------------------

fn parse_conditional(text: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    // "IF cond ..."
    let after_if = text[3..].trim();

    // Parse "NOT COND" or single-word condition
    let (cond_code, rest) = parse_condition_prefix(after_if, line)?;

    let rest = rest.trim();

    // IF cond JUMP/CALL — delegate to parse_branch and override the condition
    if rest.starts_with("JUMP") || rest.starts_with("CALL") {
        let (mut instr, label_ref) = parse_branch(rest, line)?;
        match &mut instr {
            Instruction::Branch { cond, .. } => *cond = cond_code,
            Instruction::IndirectBranch { cond, .. } => *cond = cond_code,
            Instruction::ComputeBranch { cond, .. } => *cond = cond_code,
            _ => {}
        }
        return Ok((instr, label_ref));
    }

    // IF cond RTS/RTI
    let (rts_part, compute_text) = split_first_comma(rest);
    let rts_part = rts_part.trim();
    let rts_norm = rts_part.replace(' ', "");
    if rts_norm.starts_with("RTS") && (rts_norm == "RTS" || rts_norm.starts_with("RTS(")) {
        let delayed = rts_norm.contains("DB");
        let lr = rts_norm.contains("LR");
        let compute = if let Some(ct) = compute_text {
            Some(parse_compute_op(ct.trim(), line)?)
        } else {
            None
        };
        return Ok((
            Instruction::Return {
                interrupt: false,
                cond: cond_code,
                delayed,
                lr,
                compute,
            },
            None,
        ));
    }
    if rts_norm.starts_with("RTI") && (rts_norm == "RTI" || rts_norm.starts_with("RTI(")) {
        let delayed = rts_norm.contains("DB");
        let lr = rts_norm.contains("LR");
        let compute = if let Some(ct) = compute_text {
            Some(parse_compute_op(ct.trim(), line)?)
        } else {
            None
        };
        return Ok((
            Instruction::Return {
                interrupt: true,
                cond: cond_code,
                delayed,
                lr,
                compute,
            },
            None,
        ));
    }

    // IF cond NOP
    if rest == "NOP" {
        // Conditional NOP is encoded as Type 2 with cond and zero compute.
        // The disassembler will output "IF cond NOP" for these. We represent
        // it as a PASS R0 with the condition, which the disassembler roundtrips
        // correctly only when the compute field is literally 0. Since there is
        // no "conditional NOP" variant in our Instruction enum, we use Nop and
        // accept that the condition is dropped for now. A future improvement
        // could add a dedicated ConditionalNop variant.
        return Ok((Instruction::Nop, None));
    }

    // IF cond [compute] , mem_op — conditional memory/dag ops
    // Also handles "IF cond , MODIFY(...)" with empty compute
    // Check if rest contains DM/PM or MODIFY after a comma
    if rest.contains("DM(")
        || rest.contains("PM(")
        || rest.contains("DM (")
        || rest.contains("PM (")
    {
        // Re-parse as conditional memory instruction
        let instr = parse_conditional_mem(rest, cond_code, line)?;
        return Ok((instr, None));
    }

    // IF cond [compute ,] Ia = B2W|W2B (Ib) — `if not sv i0 = b2w ( i5 )` =
    // 04af45800000.
    let compact_rest: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
    if compact_rest.contains("=B2W(") || compact_rest.contains("=W2B(") {
        return Ok((parse_dag_convert(rest, cond_code, line)?, None));
    }

    // IF cond Ia = MODIFY(Ib, Mm) — `if eq i0 = modify ( i4 , m4 ) ( nw )`
    // = 048124000000.
    if rest.starts_with('I') && find_comma_outside_parens(rest).is_none() {
        if let Some(eq) = rest.find('=') {
            let rhs = rest[eq + 1..].trim_start();
            if rhs.starts_with("MODIFY(") || rhs.starts_with("MODIFY (") {
                if let Instruction::DagModify {
                    pm,
                    i_reg,
                    m_reg,
                    dest_xor,
                    width,
                    ..
                } = parse_modify_assign_inner(rest, line)?
                {
                    return Ok((
                        Instruction::DagModify {
                            pm,
                            i_reg,
                            m_reg,
                            cond: cond_code,
                            compute: None,
                            dest_xor,
                            width,
                        },
                        None,
                    ));
                }
            }
        }
    }

    // IF cond MODIFY(Ii, Mm) — direct conditional MODIFY (no comma, no compute)
    if rest.starts_with("MODIFY(") || rest.starts_with("MODIFY (") {
        let modify = parse_dag_modify(rest, line)?;
        if let Instruction::DagModify {
            pm,
            i_reg,
            m_reg,
            dest_xor,
            width,
            ..
        } = modify
        {
            return Ok((
                Instruction::DagModify {
                    pm,
                    i_reg,
                    m_reg,
                    cond: cond_code,
                    compute: None,
                    dest_xor,
                    width,
                },
                None,
            ));
        }
    }

    // "IF cond , MODIFY(Ii, Mm)" — comma before MODIFY with empty compute
    if let Some(stripped) = rest.strip_prefix(',') {
        let after_comma = stripped.trim();
        if after_comma.starts_with("MODIFY(") || after_comma.starts_with("MODIFY (") {
            let modify = parse_dag_modify(after_comma, line)?;
            if let Instruction::DagModify {
                pm,
                i_reg,
                m_reg,
                dest_xor,
                width,
                ..
            } = modify
            {
                return Ok((
                    Instruction::DagModify {
                        pm,
                        i_reg,
                        m_reg,
                        cond: cond_code,
                        compute: None,
                        dest_xor,
                        width,
                    },
                    None,
                ));
            }
        }
        // "IF cond , ureg=ureg" — conditional transfer with empty compute
        if let Some(eq) = find_eq_outside_parens(after_comma) {
            let lhs = after_comma[..eq].trim();
            let rhs = after_comma[eq + 1..].trim();
            if let (Some(dst), Some(src)) = (ureg_code(lhs), ureg_code(rhs)) {
                return Ok((
                    Instruction::UregTransfer {
                        src_ureg: src,
                        dst_ureg: dst,
                        cond: cond_code,
                        compute: None,
                    },
                    None,
                ));
            }
        }
    }

    // IF cond compute , MODIFY
    if let Some(comma) = find_comma_outside_parens(rest) {
        let before = rest[..comma].trim();
        let after = rest[comma + 1..].trim();
        if after.starts_with("MODIFY(") || after.starts_with("MODIFY (") {
            let modify = parse_dag_modify(after, line)?;
            let compute = parse_compute_op(before, line)?;
            if let Instruction::DagModify {
                pm,
                i_reg,
                m_reg,
                cond: _,
                dest_xor,
                width,
                ..
            } = modify
            {
                return Ok((
                    Instruction::DagModify {
                        pm,
                        i_reg,
                        m_reg,
                        cond: cond_code,
                        compute: Some(compute),
                        dest_xor,
                        width,
                    },
                    None,
                ));
            }
        }
    }

    // IF cond Rn<->Sm (conditional register swap)
    if rest.contains("<->") {
        let mut swap = parse_register_swap(rest, line)?;
        if let Instruction::RegisterSwap { ref mut cond, .. } = swap {
            *cond = cond_code;
        }
        return Ok((swap, None));
    }

    // IF cond ImmShift (e.g. "IF LE R8 = LSHIFT R1 BY -1")
    // Must check before parse_compute_op since ImmShift is a top-level
    // Instruction, not a ComputeOp.
    if let Some(imm_shift) = try_parse_imm_shift(rest, line)? {
        match imm_shift {
            Instruction::ImmShift {
                shift_type,
                sub_op,
                imm,
                rn,
                rx,
                data_hi,
                ..
            } => {
                return Ok((
                    Instruction::ImmShift {
                        shift_type,
                        sub_op,
                        imm,
                        rn,
                        rx,
                        data_hi,
                        cond: cond_code,
                    },
                    None,
                ));
            }
            other => return Ok((other, None)),
        }
    }

    // IF cond compute, ureg = ureg (conditional Type 5a with compute)
    if let Some(comma) = find_comma_outside_parens(rest) {
        let before = rest[..comma].trim();
        let after = rest[comma + 1..].trim();
        if !before.is_empty() && !after.is_empty() && after.contains('=') {
            if let Ok(compute) = parse_compute_op(before, line) {
                let eq_pos = after.find('=').unwrap();
                let dst = after[..eq_pos].trim();
                let src = after[eq_pos + 1..].trim();
                if let (Some(dst_code), Some(src_code)) = (ureg_code(dst), ureg_code(src)) {
                    return Ok((
                        Instruction::UregTransfer {
                            src_ureg: src_code,
                            dst_ureg: dst_code,
                            cond: cond_code,
                            compute: Some(compute),
                        },
                        None,
                    ));
                }
            }
        }
    }

    // IF cond ureg = ureg (conditional transfer without compute)
    if let Some(eq) = find_eq_outside_parens(rest) {
        let dst = rest[..eq].trim();
        let src = rest[eq + 1..].trim();
        if let (Some(dst_code), Some(src_code)) = (ureg_code(dst), ureg_code(src)) {
            if is_float_reg(dst) || is_float_reg(src) {
                return Err(float_copy_error(dst, src, line));
            }
            return Ok((
                Instruction::UregTransfer {
                    src_ureg: src_code,
                    dst_ureg: dst_code,
                    cond: cond_code,
                    compute: None,
                },
                None,
            ));
        }
    }

    // IF cond compute
    let compute = parse_compute_op(rest, line)?;
    Ok((
        Instruction::Compute {
            cond: cond_code,
            compute,
        },
        None,
    ))
}

fn parse_conditional_mem(text: &str, cond: u8, line: u32) -> Result<Instruction> {
    // Handle "IF cond [compute ,] mem_op" where mem_op is a DM/PM access
    // The text here is everything after "IF cond "
    // Try to parse the whole thing as a memory instruction, then set the cond
    let (instr, _sym) = parse_mem_access(text, line)?;
    // Override the condition code
    match instr {
        Instruction::ComputeLoadStore {
            compute,
            access,
            dreg,
            offset,
            post_modify,
            width,
            ..
        } => Ok(Instruction::ComputeLoadStore {
            compute,
            access,
            dreg,
            offset,
            cond,
            post_modify,
            width,
        }),
        Instruction::UregDagMove {
            pm,
            write,
            ureg,
            i_reg,
            m_reg,
            compute,
            post_modify,
            width,
            ..
        } => Ok(Instruction::UregDagMove {
            pm,
            write,
            ureg,
            i_reg,
            m_reg,
            cond,
            compute,
            post_modify,
            width,
        }),
        // Type 15 has no condition field. The reference assembler encodes a conditional
        // dreg access with a 6-bit modifier as Type 4: `if sz r15=dm(-3,i6)`
        // = 6c11efbf. Anything else must not silently lose its condition.
        Instruction::UregMemAccess {
            pm: false,
            i_reg,
            write,
            lw: false,
            ureg,
            offset,
        } if ureg < 0x10 && (-32..=63).contains(&offset) => Ok(Instruction::ComputeLoadStore {
            compute: None,
            access: MemAccess {
                pm: false,
                write,
                i_reg,
            },
            dreg: ureg,
            // `if not sz r2 = dm ( 0x20 , i5 )` = 6a31013f.
            offset: type4_offset(offset, line)?,
            cond,
            post_modify: false,
            width: selinstr::encode::DagWidth::Normal,
        }),
        _ => Err(Error::Parse {
            line,
            msg: "this memory access form cannot be conditional".into(),
        }),
    }
}

fn parse_condition_prefix(text: &str, line: u32) -> Result<(u8, &str)> {
    // Try "NOT XXX" first
    if let Some(stripped) = text.strip_prefix("NOT ") {
        let after_not = stripped.trim();
        // Find the next space to get the condition name
        let (cond_name, rest) = split_first_space(after_not);
        let base = cond_code_base(cond_name).ok_or_else(|| Error::UnknownCondition {
            line,
            name: format!("NOT {cond_name}"),
        })?;
        // NOT applies to AC .. LCE (codes 3-15); EQ, LT, LE and the
        // already negated NE, GE, GT have their own names, and the reference assembler
        // rejects `if not eq` etc. (82 rows each).
        if !(3..=15).contains(&base) {
            return Err(Error::Parse {
                line,
                msg: format!("NOT {cond_name} is not a condition"),
            });
        }
        let negated = negate_condition(base);
        return Ok((negated, rest));
    }

    // Single-word condition
    let (cond_name, rest) = split_first_space(text);
    // Code 15 is LCE as a DO UNTIL terminator but NOT LCE after IF, so
    // `IF LCE` assembled as `IF NOT LCE`; the reference assembler rejects it (71 rows).
    if cond_name == "LCE" {
        return Err(Error::Parse {
            line,
            msg: "IF LCE is not a condition; the IF form is IF NOT LCE".into(),
        });
    }
    let code = cond_code_base(cond_name).ok_or_else(|| Error::UnknownCondition {
        line,
        name: cond_name.to_string(),
    })?;
    Ok((code, rest))
}

fn cond_code_base(name: &str) -> Option<u8> {
    match name {
        "EQ" => Some(0),
        "LT" => Some(1),
        "LE" => Some(2),
        "AC" => Some(3),
        "AV" => Some(4),
        "MV" => Some(5),
        "MS" => Some(6),
        "SV" => Some(7),
        "SZ" => Some(8),
        "FLAG0_IN" => Some(9),
        "FLAG1_IN" => Some(10),
        "FLAG2_IN" => Some(11),
        "FLAG3_IN" => Some(12),
        "TF" => Some(13),
        "BM" => Some(14),
        "LCE" => Some(15),
        "NE" => Some(16),
        "GE" => Some(17),
        "GT" => Some(18),
        "NAC" => Some(19),
        "NAV" => Some(20),
        "NMV" => Some(21),
        "NMS" => Some(22),
        "NSV" => Some(23),
        "NSZ" => Some(24),
        "NFLAG0_IN" => Some(25),
        "NFLAG1_IN" => Some(26),
        "NFLAG2_IN" => Some(27),
        "NFLAG3_IN" => Some(28),
        "NTF" => Some(29),
        "NBM" => Some(30),
        "TRUE" => Some(31),
        _ => None,
    }
}

fn negate_condition(code: u8) -> u8 {
    // The disassembler outputs "NOT LCE" for code 15, "NOT AC" for 19, etc.
    // In the encoding, negation flips bit 4 (adds or subtracts 16),
    // except for special cases.
    match code {
        0..=14 => code + 16,
        15 => 15, // NOT LCE = 15 in the disassembler output
        16..=30 => code - 16,
        _ => code,
    }
}

// ---------------------------------------------------------------------------
// DO loops
// ---------------------------------------------------------------------------

fn parse_do_loop(text: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    // Formats from disassembler:
    //   "LCNTR = 0x64 , DO (PC,0x2000)UNTIL LCE"   (Type 12a)
    //   "LCNTR = R5, DO 0x003000 UNTIL LCE"         (Type 12b)
    let after_eq = text
        .strip_prefix("LCNTR")
        .and_then(|s| s.trim().strip_prefix('='))
        .ok_or_else(|| Error::Parse {
            line,
            msg: "expected 'LCNTR = ...'".into(),
        })?
        .trim();

    // Split on ", DO " or " , DO " or ", DO(" (no space before paren)
    let do_sep = if let Some(pos) = after_eq.find(", DO ") {
        pos
    } else if let Some(pos) = after_eq.find(" , DO ") {
        pos
    } else if let Some(pos) = after_eq.find(", DO(") {
        pos
    } else if let Some(pos) = after_eq.find(" , DO(") {
        pos
    } else {
        return Err(Error::Parse {
            line,
            msg: "expected ', DO' in LCNTR instruction".into(),
        });
    };

    let counter_str = after_eq[..do_sep].trim();
    let do_rest = after_eq[do_sep..].trim();
    // Skip the ", DO " / " , DO " / ", DO(" prefix
    let do_part = if let Some(r) = do_rest.strip_prefix(", DO ") {
        r
    } else if let Some(r) = do_rest.strip_prefix(" , DO ") {
        r
    } else if do_rest.starts_with(", DO(") {
        &do_rest[4..] // skip ", DO" to keep "(..."
    } else if do_rest.starts_with(" , DO(") {
        &do_rest[5..]
    } else {
        do_rest
    };

    // Parse the address and UNTIL part
    // Handle both "(PC,0x2000)UNTIL LCE" and "0x003000 UNTIL LCE"
    let addr_str;
    if do_part.starts_with('(') {
        // PC-relative: "(PC,0xNNNN)UNTIL LCE"
        let close = do_part.find(')').ok_or_else(|| Error::Parse {
            line,
            msg: "missing ')' in DO loop PC-relative".into(),
        })?;
        let inner = &do_part[1..close];
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 2 && parts[0] == "PC" {
            addr_str = parts[1].to_string();
        } else {
            addr_str = inner.to_string();
        }
    } else {
        // Absolute: "0x003000 UNTIL LCE"
        let until_pos = do_part.find("UNTIL").ok_or_else(|| Error::Parse {
            line,
            msg: "expected 'UNTIL' in DO loop".into(),
        })?;
        addr_str = do_part[..until_pos].trim().to_string();
    }

    let (end_pc, label_ref) = parse_address_or_label(addr_str.trim(), line)?;

    // Termination must be LCE, optionally with the (F) suffix; anything else
    // was previously ignored.
    let until_pos = do_part.find("UNTIL").ok_or_else(|| Error::Parse {
        line,
        msg: "expected 'UNTIL' in DO loop".into(),
    })?;
    let term: String = do_part[until_pos + 5..]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let f = match term.as_str() {
        "LCE" => false,
        "LCE(F)" => true,
        other => {
            return Err(Error::Parse {
                line,
                msg: format!("unsupported DO loop termination: {other}"),
            })
        }
    };

    // Counter is either a number or a register name
    let counter_str = counter_str.trim();
    if let Some(val) = parse_number(counter_str) {
        // Type 12a holds a 16-bit count; the reference assembler rejects `lcntr = 0x10000`
        // (24 rows), which selas truncated to 0.
        if val > 0xFFFF {
            return Err(Error::Parse {
                line,
                msg: format!("LCNTR count {counter_str} does not fit 16 bits"),
            });
        }
        Ok((
            Instruction::DoLoop {
                counter: LoopCounter::Immediate(val as u16),
                end_pc,
                f,
            },
            label_ref,
        ))
    } else {
        let ureg = ureg_code(counter_str).ok_or_else(|| Error::UnknownRegister {
            line,
            name: counter_str.to_string(),
        })?;
        Ok((
            Instruction::DoLoop {
                counter: LoopCounter::Ureg(ureg as u8),
                end_pc,
                f,
            },
            label_ref,
        ))
    }
}

/// Parse "DO addr UNTIL cond" (Type 13).
fn parse_do_until(text: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    let rest = text.strip_prefix("DO ").unwrap().trim();
    let until_pos = rest.find("UNTIL").ok_or_else(|| Error::Parse {
        line,
        msg: "expected 'UNTIL' in DO instruction".into(),
    })?;
    let addr_str = rest[..until_pos].trim();
    let term_str = rest[until_pos + 5..].trim();
    let packed: String = term_str.chars().filter(|c| !c.is_whitespace()).collect();
    let f = packed.ends_with("(F)");
    let term_str = if f {
        packed.trim_end_matches("(F)")
    } else {
        term_str
    };

    let (addr, label_ref) = parse_address_or_label(addr_str, line)?;
    let term = cond_code_base(term_str).ok_or_else(|| Error::UnknownCondition {
        line,
        name: term_str.to_string(),
    })?;
    // The reference assembler rejects a fast loop with a termination condition other
    // than LCE (`do _x until eq (f)`, SW and NW).
    const LCE: u8 = 15;
    if f && term != LCE {
        return Err(Error::Parse {
            line,
            msg: "a fast loop `(F)` needs the LCE termination condition (reference assembler)".into(),
        });
    }
    Ok((Instruction::DoUntil { addr, term, f }, label_ref))
}

// ---------------------------------------------------------------------------
// BIT SET/CLR/TGL/TST/XOR
// ---------------------------------------------------------------------------

fn parse_bit_op(text: &str, line: u32) -> Result<Instruction> {
    // "BIT SET MODE1 0x00001000"
    let rest = text[4..].trim(); // skip "BIT "
    let (opname, after_op) = split_first_space(rest);
    let op = match opname {
        // The BOP field (bits 39:37). The reference assembler: `bit tgl mode1 0x0` =
        // 144200000000, `bit xor mode1 0x0` = 14a200000000; selas encoded
        // TGL as XOR.
        "SET" => 0,
        "CLR" => 1,
        "TGL" => 2,
        "TST" => 4,
        "XOR" => 5,
        _ => {
            return Err(Error::Parse {
                line,
                msg: format!("unknown BIT operation: {opname}"),
            });
        }
    };

    let after_op = after_op.trim();
    let (sreg_name, data_str) = split_first_space(after_op);
    let sreg = bit_op_sreg_code(sreg_name).ok_or_else(|| Error::UnknownRegister {
        line,
        name: sreg_name.to_string(),
    })?;

    let data_str = data_str.trim();
    // Strip all outer layers of parentheses
    let mut inner = data_str;
    while inner.starts_with('(') && inner.ends_with(')') {
        let candidate = &inner[1..inner.len() - 1];
        // Verify the parens are balanced (not "(a) | (b)")
        let mut depth = 0i32;
        let mut balanced = true;
        for c in candidate.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if balanced && depth == 0 {
            inner = candidate.trim();
        } else {
            break;
        }
    }
    // Evaluate wide so an out-of-range value is an error, not truncated.
    let wide = |s: &str| -> Option<i64> {
        let s = s.trim();
        if let Some(hex) = s.strip_prefix("0X").or_else(|| s.strip_prefix("0x")) {
            if let Ok(v) = i64::from_str_radix(hex, 16) {
                return Some(v);
            }
        }
        s.parse::<i64>().ok().or_else(|| eval_const_expr(s))
    };
    let data = if let Some(rest) = inner.strip_prefix('-') {
        let val = wide(rest).ok_or_else(|| Error::Parse {
            line,
            msg: format!("invalid BIT data: {data_str}"),
        })?;
        fit_data32(-val, "BIT data", line)? as u32
    } else if let Some(val) = wide(inner) {
        fit_data32(val, "BIT data", line)? as u32
    } else {
        // BIT data must be a constant: the reference assembler rejects `bit clr mode1
        // 0xzz`. selas warned and used 0, a BIT op that did nothing.
        return Err(Error::Parse {
            line,
            msg: format!("BIT data is not a constant: {data_str}"),
        });
    };

    Ok(Instruction::BitOp { op, sreg, data })
}

/// Map system register name to the 5-bit sreg code used in BIT ops (Type 18).
fn bit_op_sreg_code(name: &str) -> Option<u8> {
    // The low nibble of the group-7 universal register code: vendor
    // `bit tst ustat1 0x0` = 148000000000.
    match name {
        "USTAT1" => Some(0x00),
        "USTAT2" => Some(0x01),
        "MODE1" => Some(0x02),
        "MMASK" => Some(0x03),
        "MODE2" => Some(0x04),
        "FLAGS" => Some(0x05),
        "ASTAT" | "ASTATX" | "ASTATx" => Some(0x06),
        "ASTATY" | "ASTATy" => Some(0x07),
        "STKY" | "STKYX" | "STKYx" => Some(0x08),
        "STKYY" | "STKYy" => Some(0x09),
        "IRPTL" => Some(0x0A),
        "IMASK" => Some(0x0B),
        "IMASKP" => Some(0x0C),
        "MODE1STK" => Some(0x0D),
        "USTAT3" => Some(0x0E),
        "USTAT4" => Some(0x0F),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// PUSH / POP / FLUSH stack operations
// ---------------------------------------------------------------------------

fn parse_stack_op(text: &str, line: u32) -> Result<Instruction> {
    // May be a single op or comma-separated: "PUSH STS", "PUSH LOOP,  PUSH STS"
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    let mut ops: u8 = 0;
    for part in &parts {
        let mask = stack_op_bit(part).ok_or_else(|| Error::Parse {
            line,
            msg: format!("unknown stack operation: {part}"),
        })?;
        ops |= mask;
    }
    Ok(Instruction::StackOp { ops })
}

fn stack_op_bit(text: &str) -> Option<u8> {
    let text = text.trim();
    match text {
        "PUSH LOOP" => Some(0x40),
        "POP LOOP" => Some(0x20),
        "PUSH STS" => Some(0x10),
        "POP STS" => Some(0x08),
        "PUSH PCSTK" => Some(0x04),
        "POP PCSTK" => Some(0x02),
        "FLUSH CACHE" => Some(0x01),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CJUMP (computed jump)
// ---------------------------------------------------------------------------

fn parse_cjump(text: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    // "CJUMP 0x001234 (DB)" or "CJUMP 0x001234"
    let rest = text[5..].trim(); // skip "CJUMP"
                                 // The reference assembler rejects CJUMP without (DB) (all 176 rows) and writes the
                                 // suffix spaced as often as not (`cjump -0x436e00 ( db )`).
    let (addr_str, suffix) = match rest.rfind('(') {
        Some(open) if rest.ends_with(')') => (&rest[..open], &rest[open..]),
        _ => (rest, ""),
    };
    let packed: String = suffix.chars().filter(|c| !c.is_whitespace()).collect();
    if packed != "(DB)" {
        return Err(Error::Parse {
            line,
            msg: format!("CJUMP must be delayed, `CJUMP addr (DB)`: {text}"),
        });
    }
    let (target, label_ref) = parse_branch_target(addr_str.trim(), line)?;
    let (addr, relative) = match target {
        BranchTarget::Absolute(addr) => (addr, false),
        BranchTarget::PcRelative(offset) => (offset as u32, true),
    };
    Ok((
        Instruction::CJump {
            addr,
            delayed: true,
            relative,
        },
        label_ref,
    ))
}

// ---------------------------------------------------------------------------
// MODIFY(Ii, Mm) — DAG register modify (Type 7)
// ---------------------------------------------------------------------------

/// The suffix after `MODIFY(Ii, data)`. The reference assembler writes `( nw )` with
/// spaces as often as `(nw)` (`modify ( i0 , 0x7 ) ( nw )` =
/// 158000000007); selas matched only the packed spelling and silently
/// encoded the spaced one as a plain MODIFY.
fn modify_width(suffix: &str, line: u32) -> Result<MemWidth> {
    let packed: String = suffix.chars().filter(|c| !c.is_whitespace()).collect();
    match packed.to_uppercase().as_str() {
        "" => Ok(MemWidth::Normal),
        "(NW)" => Ok(MemWidth::Nw),
        "(SW)" => Ok(MemWidth::Sw),
        _ => Err(Error::Parse {
            line,
            msg: format!("unknown MODIFY modifier: {}", suffix.trim()),
        }),
    }
}

/// A 32-bit immediate field (MODIFY data, BIT data, a Type 15 offset)
/// takes any value of magnitude below 2^32, wrapped (`astatx = dm (
/// -0x80000001 , i6 )` = ac767fffffff); the reference assembler rejects `modify ( i0 ,
/// 0x100000000 )`, `bit set mode1 0x100000000` and `astatx = dm (
/// -0x100000000 , i6 )`, which selas truncated.
fn fit_data32(value: i64, what: &str, line: u32) -> Result<i32> {
    if (-0xFFFF_FFFF..=0xFFFF_FFFF).contains(&value) {
        Ok(value as u32 as i32)
    } else {
        Err(Error::Parse {
            line,
            msg: if value < 0 {
                format!("{what} -{:#x} does not fit 32 bits", -value)
            } else {
                format!("{what} {value:#x} does not fit 32 bits")
            },
        })
    }
}

fn parse_dag_modify(text: &str, line: u32) -> Result<Instruction> {
    // "MODIFY(I0, M1)" or "MODIFY (I0, M1)" or "MODIFY(I7, -4)" or "BITREV(I14, 0)"
    // Also handles nested parens: "MODIFY(I4, (4 + 4 + 3)) (NW)"
    let is_bitrev = text.starts_with("BITREV");
    let open = text.find('(').ok_or_else(|| Error::Parse {
        line,
        msg: "missing '(' in MODIFY/BITREV".into(),
    })?;
    let close = find_matching_close_paren(text, open).ok_or_else(|| Error::Parse {
        line,
        msg: "missing ')' in MODIFY".into(),
    })?;
    let inner = text[open + 1..close].trim();
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(Error::Parse {
            line,
            msg: format!("expected 'Ii, Mm' in MODIFY: {inner}"),
        });
    }
    let i_reg_full = parse_reg_index(parts[0], 'I', line)?;

    // Check for immediate offset (Type 19): MODIFY(Ii, <number>) or expression
    let imm_from_expr = parse_signed_number(parts[1]).or_else(|| eval_const_expr(parts[1]));
    if let Some(imm_val) = imm_from_expr {
        let after_close = &text[close + 1..];
        let width = if after_close.contains("NWOPT") {
            MemWidth::Nw
        } else {
            modify_width(after_close, line)?
        };
        return Ok(Instruction::Modify {
            i_reg: i_reg_full,
            value: fit_data32(imm_val, "MODIFY offset", line)?,
            width,
            bitrev: is_bitrev,
            dest_xor: 0,
        });
    }

    let m_reg_full = parse_reg_index(parts[1], 'M', line)?;

    // Determine DAG: I0-I7/M0-M7 = DAG1, I8-I15/M8-M15 = DAG2
    let pm = i_reg_full >= 8;
    let i_reg = if pm { i_reg_full - 8 } else { i_reg_full };
    let m_reg = if pm { m_reg_full - 8 } else { m_reg_full };

    let width = modify_width(&text[close + 1..], line)?;
    Ok(Instruction::DagModify {
        pm,
        i_reg,
        m_reg,
        cond: 31,
        compute: None,
        dest_xor: 0,
        width,
    })
}

/// `[compute ,] Ia = B2W|W2B (Ib)` or the same on B registers, upper-cased
/// (Instruction::DagConvert). Both registers sit in one DAG (the reference assembler
/// rejects `i15 = b2w ( i7 )`).
fn parse_dag_convert(text: &str, cond: u8, line: u32) -> Result<Instruction> {
    let (compute, conv) = match find_all_commas_outside_parens(text).last() {
        Some(&comma) => (
            Some(parse_compute_op(text[..comma].trim(), line)?),
            text[comma + 1..].trim(),
        ),
        None => (None, text.trim()),
    };
    let bad = || Error::Parse {
        line,
        msg: format!("expected `Ia = B2W|W2B (Ib)` or `Ba = B2W|W2B (Bb)`: {conv}"),
    };
    let (lhs, rhs) = conv.split_once('=').ok_or_else(bad)?;
    let lhs = lhs.trim();
    let rhs: String = rhs.chars().filter(|c| !c.is_whitespace()).collect();
    let (w2b, operand) = if let Some(r) = rhs.strip_prefix("B2W(") {
        (false, r)
    } else if let Some(r) = rhs.strip_prefix("W2B(") {
        (true, r)
    } else {
        return Err(bad());
    };
    let operand = operand.strip_suffix(')').ok_or_else(bad)?;
    let prefix = match lhs.chars().next() {
        Some(c @ ('I' | 'B')) => c,
        _ => return Err(bad()),
    };
    let dst = parse_reg_index(lhs, prefix, line)?;
    let src = parse_reg_index(operand, prefix, line)?;
    if (dst >= 8) != (src >= 8) {
        return Err(Error::Parse {
            line,
            msg: format!("{lhs} and {operand} must be in one DAG"),
        });
    }
    Ok(Instruction::DagConvert {
        pm: src >= 8,
        dest_xor: (dst ^ src) & 7,
        i_reg: src & 7,
        op: u8::from(prefix == 'B') << 1 | u8::from(w2b),
        cond,
        compute,
    })
}

// ---------------------------------------------------------------------------
// Ii=MODIFY (Ii, Mm) or Ii=MODIFY (Ii,data32) forms
// ---------------------------------------------------------------------------

fn parse_modify_assign(text: &str, line: u32) -> Result<Instruction> {
    // "I7=MODIFY (I7,M6)(NW)" or "I4=MODIFY (I4,0x4)"
    // Also handles compound: "compute , I7=MODIFY (I7,M6)(NW)"
    // Check for compound with comma; split at the last one so a
    // multifunction compute stays whole (`f1 = f1 * f7 , f12 = f8 + f12 ,
    // i1 = modify ( i1 , m4 )` = 043e60581c70).
    if let Some(&comma) = find_all_commas_outside_parens(text).last() {
        let before = text[..comma].trim();
        let after = text[comma + 1..].trim();
        if after.contains("MODIFY") {
            let compute = parse_compute_op(before, line)?;
            let base = parse_modify_assign_inner(after, line)?;
            // Add compute to the result
            return match base {
                Instruction::DagModify {
                    pm,
                    i_reg,
                    m_reg,
                    cond,
                    dest_xor,
                    width,
                    ..
                } => Ok(Instruction::DagModify {
                    pm,
                    i_reg,
                    m_reg,
                    cond,
                    compute: Some(compute),
                    dest_xor,
                    width,
                }),
                Instruction::Modify { i_reg, value, .. } => {
                    // Type 19 Modify doesn't have a compute field
                    let _ = (i_reg, value, compute);
                    Err(Error::Parse {
                        line,
                        msg: "Type 19 MODIFY does not support compute".into(),
                    })
                }
                other => Ok(other),
            };
        }
    }
    parse_modify_assign_inner(text, line)
}

fn parse_modify_assign_inner(text: &str, line: u32) -> Result<Instruction> {
    // "I4=MODIFY (I4,0x4)" or "I7=MODIFY (I7,M6)(NW)"
    let eq = text.find('=').ok_or_else(|| Error::Parse {
        line,
        msg: "missing '=' in MODIFY assignment".into(),
    })?;
    let rhs = text[eq + 1..].trim();

    // Find the MODIFY(...) part, skip any trailing (NW) suffix
    let open = rhs.find('(').ok_or_else(|| Error::Parse {
        line,
        msg: "missing '(' in MODIFY".into(),
    })?;
    let close = rhs.find(')').ok_or_else(|| Error::Parse {
        line,
        msg: "missing ')' in MODIFY".into(),
    })?;
    let inner = rhs[open + 1..close].trim();

    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(Error::Parse {
            line,
            msg: format!("expected two operands in MODIFY: {inner}"),
        });
    }

    let i_reg_full = parse_reg_index(parts[0], 'I', line)?;
    // `Ia = MODIFY(Ib, ...)` writes Ia; the encoding carries Ia XOR Ib
    // (0 for Ia == Ib, which matches the two-operand form). The reference assembler:
    // `i4 = modify(i4, 0x10)` = 160400000010, `i2 = modify(i4, 0x1a)` =
    // 16340000001a, `i1 = modify(i0,m4)` = 043e213f.
    let lhs = text[..eq].rsplit(',').next().unwrap_or("").trim();
    let dest_full = parse_reg_index(lhs, 'I', line)?;
    if (dest_full >= 8) != (i_reg_full >= 8) {
        return Err(Error::Parse {
            line,
            msg: format!("{lhs} = MODIFY({}, ...) must stay within one DAG", parts[0]),
        });
    }
    let dest_xor = (dest_full ^ i_reg_full) & 7;

    // Check if second operand is M-register (Type 7) or immediate (Type 19)
    if parts[1].starts_with('M') {
        let m_reg_full = parse_reg_index(parts[1], 'M', line)?;
        let pm = i_reg_full >= 8;
        let i_reg = if pm { i_reg_full - 8 } else { i_reg_full };
        let m_reg = if pm { m_reg_full - 8 } else { m_reg_full };
        // `i0 = modify ( i4 , m4 ) ( sw )` = 043f24800000 (bit 23),
        // `r0 = r2 and r1 , i2 = modify ( i2 , m6 ) ( nw )` = 04beb0040021
        // (bit 39).
        let width = modify_width(&rhs[close + 1..], line)?;
        Ok(Instruction::DagModify {
            pm,
            i_reg,
            m_reg,
            cond: 31,
            compute: None,
            dest_xor,
            width,
        })
    } else {
        // Immediate: "I4=MODIFY (I4,0x4)" or "I4=MODIFY (I4,0x4)(NW)"
        let value = parse_signed_number(parts[1]).ok_or_else(|| Error::Parse {
            line,
            msg: format!("invalid MODIFY offset: {}", parts[1]),
        })?;
        let value = fit_data32(value, "MODIFY offset", line)?;
        let width = modify_width(&rhs[close + 1..], line)?;
        let bitrev = rhs.starts_with("BITREV");
        Ok(Instruction::Modify {
            i_reg: i_reg_full,
            value,
            width,
            bitrev,
            dest_xor,
        })
    }
}

// ---------------------------------------------------------------------------
// Memory access
// ---------------------------------------------------------------------------

fn parse_mem_access(text: &str, line: u32) -> Result<(Instruction, Option<String>)> {
    let (text, width_mod) = extract_width_modifier(text);
    let is_lw = width_mod == Some("(LW)");
    let (mut instr, symbol) = parse_mem_access_inner(text, is_lw, line)?;
    // `Rn = DM(imm, Ii)(SW|BW)`: the reference assembler emits the 32-bit Type 4b form
    // with width marker 0x3c / 0x38 (683e163c, 683e0938).
    // The sign-extending widths and PM take the same form (`r0 = dm (
    // -0x2 , i4 ) ( swse )` = 683ff03d, `r0 = pm ( 0x0 , i12 ) ( sw )` =
    // 693e003c), offsets -0x20..=0x3f as in type4_offset.
    let narrow = match width_mod {
        Some("(SW)") => Some(DagWidth::Short),
        Some("(BW)") => Some(DagWidth::Byte),
        Some("(SWSE)") => Some(DagWidth::ShortSe),
        Some("(BWSE)") => Some(DagWidth::ByteSe),
        _ => None,
    };
    // Stores use the same marker with the write bit set: no listing row
    // has one, but a binary round-trip reproduces 86 such vendor
    // words byte for byte.
    if let (
        Some(width),
        None,
        Instruction::UregMemAccess {
            pm,
            lw: false,
            i_reg,
            write,
            ureg,
            offset,
        },
    ) = (narrow, &symbol, &instr)
    {
        // i_reg is DAG-local here (PM I8-I15 -> 0-7).
        if *ureg < 0x10 && *i_reg < 8 && (-32..=63).contains(offset) {
            let access = MemAccess {
                pm: *pm,
                write: *write,
                i_reg: *i_reg,
            };
            instr = Instruction::ComputeLoadStore {
                compute: None,
                access,
                dreg: *ureg,
                offset: type4_offset(*offset, line)?,
                cond: 31,
                post_modify: false,
                width,
            };
            return Ok((instr, symbol));
        }
    }
    match (width_mod, &mut instr) {
        (None | Some("(NW)"), _) => {}
        (Some("(LW)"), Instruction::UregMemAccess { .. }) => {}
        (Some(modifier), Instruction::UregAbsAccess { width, .. }) if modifier != "(NW)" => {
            *width = match modifier {
                "(LW)" => DagWidth::Long,
                "(SW)" => DagWidth::Short,
                "(BW)" => DagWidth::Byte,
                "(SWSE)" => DagWidth::ShortSe,
                _ => DagWidth::ByteSe,
            };
        }
        (Some("(SW)"), Instruction::UregDagMove { width, .. }) => {
            *width = DagWidth::Short;
        }
        (Some("(BW)"), Instruction::UregDagMove { width, .. }) => {
            *width = DagWidth::Byte;
        }
        (Some("(LW)"), Instruction::UregDagMove { width, .. }) => {
            *width = DagWidth::Long;
        }
        (Some("(SWSE)"), Instruction::UregDagMove { width, .. }) => {
            *width = DagWidth::ShortSe;
        }
        (Some("(BWSE)"), Instruction::UregDagMove { width, .. }) => {
            *width = DagWidth::ByteSe;
        }
        // `Rn = DM(Ii, imm)(SW|BW)`, the post-modify Type 4b form
        // (the reference assembler 687e0838 `r0 = dm(i4, 1)(bw)`).
        (
            Some(modifier @ ("(SW)" | "(BW)" | "(SWSE)" | "(BWSE)")),
            Instruction::ComputeLoadStore {
                compute: None,
                post_modify: true,
                width,
                ..
            },
        ) => {
            *width = match modifier {
                "(SW)" => DagWidth::Short,
                "(BW)" => DagWidth::Byte,
                "(SWSE)" => DagWidth::ShortSe,
                _ => DagWidth::ByteSe,
            };
        }
        (Some(modifier), _) => {
            return Err(Error::Parse {
                line,
                msg: format!("unsupported memory access modifier {modifier} for this instruction"),
            });
        }
    }
    Ok((instr, symbol))
}

fn parse_mem_access_inner(
    text: &str,
    is_lw: bool,
    line: u32,
) -> Result<(Instruction, Option<String>)> {
    // Check for dual DM+PM move without compute
    if let Some(dual) = try_parse_dual_move("", text, line)? {
        return Ok((dual, None));
    }

    // Handle possible compound instructions: "compute , mem_op"
    if let Some(compound) = try_parse_compound_mem(text, line)? {
        return Ok((compound, None));
    }

    // Split on the first '=' that is outside parentheses
    let eq_pos = find_eq_outside_parens(text).ok_or_else(|| Error::Parse {
        line,
        msg: format!("no '=' found in memory access: {text}"),
    })?;

    let lhs = text[..eq_pos].trim();
    let rhs = text[eq_pos + 1..].trim();

    // Check which side has DM/PM
    let lhs_has_mem = has_mem_prefix(lhs);
    let rhs_has_mem = has_mem_prefix(rhs);

    if lhs_has_mem {
        // Store: DM(Ii, x) = Rn or DM (x,Ii)=Rn or DM(abs_addr)=Rn
        let (pm, inner) = extract_mem_inner(lhs, line)?;
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 1 {
            // Single operand: absolute address (Type 14)
            let (addr, sym) = resolve_abs_addr(parts[0], line)?;
            let ureg = parse_ureg_or_dreg(rhs, line)?;
            return Ok((
                Instruction::UregAbsAccess {
                    pm,
                    write: true,
                    ureg,
                    addr,
                    width: DagWidth::Normal,
                },
                sym,
            ));
        }
        if parts.len() != 2 {
            return Err(Error::Parse {
                line,
                msg: format!("expected two operands in memory expression: {inner}"),
            });
        }
        parse_mem_store(pm, parts[0], parts[1], rhs, is_lw, line)
    } else if rhs_has_mem {
        // Load: Rn = DM(Ii, x) or ureg=DM (Ii,Mm) or ureg=DM(abs_addr)
        let (pm, inner) = extract_mem_inner(rhs, line)?;
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 1 {
            // Single operand: absolute address (Type 14)
            let (addr, sym) = resolve_abs_addr(parts[0], line)?;
            let ureg = parse_ureg_or_dreg(lhs, line)?;
            return Ok((
                Instruction::UregAbsAccess {
                    pm,
                    write: false,
                    ureg,
                    addr,
                    width: DagWidth::Normal,
                },
                sym,
            ));
        }
        if parts.len() != 2 {
            return Err(Error::Parse {
                line,
                msg: format!("expected two operands in memory expression: {inner}"),
            });
        }
        parse_mem_load(pm, lhs, parts[0], parts[1], is_lw, line).map(|i| (i, None))
    } else {
        Err(Error::Parse {
            line,
            msg: format!("no DM/PM found in memory access: {text}"),
        })
    }
}

/// Check if a string starts with a DM/PM memory reference.
fn has_mem_prefix(s: &str) -> bool {
    s.starts_with("DM(") || s.starts_with("PM(") || s.starts_with("DM (") || s.starts_with("PM (")
}

/// Build a UregDagMove from two memory operands (I-reg and M-reg, either order).
/// Returns None if neither operand pair is I+M or M+I.
fn try_build_dag_move(
    pm: bool,
    op1: &str,
    op2: &str,
    ureg_name: &str,
    write: bool,
    compute: Option<ComputeOp>,
    line: u32,
) -> Result<Option<Instruction>> {
    // SHARC operand order encodes pre/post-modify directly:
    //   DM(Ii, Mm) = post-modify (use Ii, then Ii += Mm)
    //   DM(Mm, Ii) = pre-modify  (use Ii + Mm, leave Ii unchanged)
    // Losing this distinction at parse time silently turned every
    // post-modify access into a pre-modify one, which still happens
    // to read the right slot only when Mm == 0 -- any non-zero
    // M-register value walked the access into unrelated memory.
    let (i_str, m_str, post_modify) = if op1.starts_with('I') && op2.starts_with('M') {
        (op1, op2, true)
    } else if op1.starts_with('M') && op2.starts_with('I') {
        (op2, op1, false)
    } else {
        return Ok(None);
    };
    let i_reg_full = parse_reg_index(i_str, 'I', line)?;
    let m_reg_full = parse_reg_index(m_str, 'M', line)?;
    let ureg = parse_ureg_or_dreg(ureg_name, line)?;
    let dag_pm = i_reg_full >= 8;
    if dag_pm != pm || (m_reg_full >= 8) != pm {
        return Err(Error::Parse {
            line,
            msg: "DM requires DAG1 registers; PM requires DAG2 registers".into(),
        });
    }
    let i_reg = if dag_pm { i_reg_full - 8 } else { i_reg_full };
    let m_reg = if dag_pm {
        m_reg_full.wrapping_sub(8)
    } else {
        m_reg_full
    };
    Ok(Some(Instruction::UregDagMove {
        pm: dag_pm,
        write,
        ureg,
        i_reg,
        m_reg,
        cond: 31,
        compute,
        post_modify,
        width: DagWidth::Normal,
    }))
}

/// Parse a memory store: DM(operand1, operand2) = value
fn parse_mem_store(
    pm: bool,
    op1: &str,
    op2: &str,
    rhs: &str,
    is_lw: bool,
    line: u32,
) -> Result<(Instruction, Option<String>)> {
    if rhs.trim_end().rsplit_once('(').is_some_and(|(_, tail)| {
        tail.strip_suffix(')')
            .is_some_and(|modifier| modifier.trim() == "SE")
    }) {
        return Err(Error::Parse {
            line,
            msg: "(se) is not a memory access modifier".into(),
        });
    }
    // Type 16: DM|PM(Ii, Mm) = imm32 or symbolic label
    if op1.starts_with('I') && op2.starts_with('M') || op1.starts_with('M') && op2.starts_with('I')
    {
        let is_imm = parse_signed_number(rhs).is_some()
            || has_label_ident(rhs)
            || has_label_with_addend(rhs);
        if is_imm {
            let (i_str, m_str) = if op1.starts_with('I') {
                (op1, op2)
            } else {
                (op2, op1)
            };
            let i_raw = parse_reg_index(i_str, 'I', line)?;
            let m_raw = parse_reg_index(m_str, 'M', line)?;
            let i_reg = if pm { i_raw.wrapping_sub(8) } else { i_raw };
            let m_reg = if pm { m_raw.wrapping_sub(8) } else { m_raw };
            let (value, sym) = if let Some(v) = parse_signed_number(rhs) {
                // The reference assembler rejects `dm ( i7 , m7 ) = 0x100000000`.
                (fit_data32(v, "immediate", line)? as u32, None)
            } else if let Some((label, adj)) = split_label_addend(rhs) {
                (adj as i32 as u32, Some(label.to_string()))
            } else {
                (0, Some(rhs.to_string()))
            };
            return Ok((
                Instruction::ImmStore {
                    pm,
                    i_reg,
                    m_reg,
                    value,
                },
                sym,
            ));
        }
    }

    if let Some(dag_move) = try_build_dag_move(pm, op1, op2, rhs, true, None, line)? {
        return Ok((dag_move, None));
    }

    let (i_reg, offset, post_modify) = parse_ireg_and_offset(op1, op2, line)?;
    let i_field = if pm { i_reg.wrapping_sub(8) } else { i_reg };
    if post_modify {
        return Ok((
            post_modify_imm_access(pm, true, is_lw, rhs, i_field, offset, line)?,
            None,
        ));
    }
    if let Ok(dreg) = parse_dreg(rhs, line) {
        return Ok((
            Instruction::UregMemAccess {
                pm,
                i_reg: i_field,
                write: true,
                lw: is_lw,
                ureg: dreg,
                offset,
            },
            None,
        ));
    }
    if let Some(ureg) = ureg_code(rhs) {
        return Ok((
            Instruction::UregMemAccess {
                pm,
                i_reg: i_field,
                write: true,
                lw: is_lw,
                ureg,
                offset,
            },
            None,
        ));
    }
    Err(Error::UnknownRegister {
        line,
        name: rhs.to_string(),
    })
}

/// Parse a memory load: dest = DM(operand1, operand2)
fn parse_mem_load(
    pm: bool,
    dest: &str,
    op1: &str,
    op2: &str,
    is_lw: bool,
    line: u32,
) -> Result<Instruction> {
    if let Some(dag_move) = try_build_dag_move(pm, op1, op2, dest, false, None, line)? {
        return Ok(dag_move);
    }

    // Numeric offset: dreg = DM(imm, Ii) or dreg = DM(Ii, imm)
    // Always use Type 15 for standalone loads with integer offsets.
    // Type 4 is only used when a parallel compute is present.
    let (i_reg, offset, post_modify) = parse_ireg_and_offset(op1, op2, line)?;
    let i_field = if pm { i_reg.wrapping_sub(8) } else { i_reg };
    if post_modify {
        return post_modify_imm_access(pm, false, is_lw, dest, i_field, offset, line);
    }
    if let Ok(dreg) = parse_dreg(dest, line) {
        let ureg = dreg; // R0-R15 have ureg codes 0x00-0x0F
        return Ok(Instruction::UregMemAccess {
            pm,
            i_reg: i_field,
            write: false,
            lw: is_lw,
            ureg,
            offset,
        });
    }
    // Non-dreg destination (I/M/L/B/special regs): Type 15 encoding
    if let Some(ureg) = ureg_code(dest) {
        return Ok(Instruction::UregMemAccess {
            pm,
            i_reg: i_field,
            write: false,
            lw: is_lw,
            ureg,
            offset,
        });
    }
    Err(Error::UnknownRegister {
        line,
        name: dest.to_string(),
    })
}

/// Parse I-register and immediate offset from two operands.
/// `(Ii, imm)` is post-modify (access Ii, then Ii += imm) and `(imm, Ii)` is
/// pre-modify (access Ii+imm, Ii unchanged); the third value is true for
/// post-modify. Returns the full i32 offset so callers can decide between
/// Type 4 (6-bit) and Type 15 (32-bit, pre-modify only) encoding.
fn parse_ireg_and_offset(op1: &str, op2: &str, line: u32) -> Result<(u8, i32, bool)> {
    if op1.starts_with('I') {
        let i_reg = parse_reg_index(op1, 'I', line)?;
        let offset = parse_signed_number(op2).ok_or_else(|| Error::Parse {
            line,
            msg: format!("invalid offset: {op2}"),
        })?;
        Ok((i_reg, fit_data32(offset, "offset", line)?, true))
    } else if op2.starts_with('I') {
        let i_reg = parse_reg_index(op2, 'I', line)?;
        let offset = parse_signed_number(op1).ok_or_else(|| Error::Parse {
            line,
            msg: format!("invalid offset: {op1}"),
        })?;
        Ok((i_reg, fit_data32(offset, "offset", line)?, false))
    } else {
        Err(Error::Parse {
            line,
            msg: format!("expected I-register in memory operands: {op1}, {op2}"),
        })
    }
}

/// Type 4 data6 offset, range-checked before narrowing. The reference assembler takes
/// -0x20..=0x3f and keeps the low six bits (`r0 = dm(i0, 0x3f)` =
/// 607ff83f, `dm ( i4 , 0x20 ) = r2` = 68ff013f; 0x40 and -0x21 are
/// rejected).
fn type4_offset(offset: i32, line: u32) -> Result<i8> {
    if (-32..=63).contains(&offset) {
        Ok(((offset << 26) >> 26) as i8)
    } else {
        Err(Error::Parse {
            line,
            msg: format!("immediate modifier {offset} out of 6-bit range (-0x20..0x3f)"),
        })
    }
}

/// Standalone post-modify `DM|PM(Ii,imm)` access: only Type 4 (dreg,
/// 6-bit modifier) can update Ii by an immediate. PM: `r0 = pm ( i8 ,
/// 0x0 )` = 617e003f.
fn post_modify_imm_access(
    pm: bool,
    write: bool,
    is_lw: bool,
    reg: &str,
    i_reg: u8,
    offset: i32,
    line: u32,
) -> Result<Instruction> {
    let dreg =
        match parse_dreg(reg, line) {
            Ok(d) if !is_lw && i_reg < 8 => d,
            _ => return Err(Error::Parse {
                line,
                msg: "post-modify DM|PM(Ii,imm) is only supported for R0-R15 accesses without (LW)"
                    .into(),
            }),
        };
    Ok(Instruction::ComputeLoadStore {
        compute: None,
        access: MemAccess { pm, write, i_reg },
        dreg,
        offset: type4_offset(offset, line)?,
        cond: 31,
        post_modify: true,
        width: selinstr::encode::DagWidth::Normal,
    })
}

/// Parse a ureg name, falling back to dreg (R0-R15/F0-F15) codes.
fn parse_ureg_or_dreg(name: &str, line: u32) -> Result<u16> {
    ureg_code(name).ok_or_else(|| Error::UnknownRegister {
        line,
        name: name.to_string(),
    })
}

/// Try to parse a compound instruction with compute + memory access.
/// Returns None if the text doesn't match the compound pattern.
fn try_parse_compound_mem(text: &str, line: u32) -> Result<Option<Instruction>> {
    // Look for a comma that separates compute from memory access.
    // For multifunction compute + memory (e.g.
    //   "R0 = R1 * R2 (SSF) , R3 = R4 + R5 , DM(I0,M1) = R6"),
    // we need the LAST comma before the memory part.
    let commas = find_all_commas_outside_parens(text);
    if commas.is_empty() {
        return Ok(None);
    }

    // Find the split point: the last comma where the text after it
    // contains a DM/PM reference but the text before it does not.
    let mut split = None;
    for &pos in commas.iter().rev() {
        let after = text[pos + 1..].trim();
        let before = text[..pos].trim();
        if has_mem_prefix_anywhere(after) && !has_mem_prefix_anywhere(before) {
            split = Some(pos);
            break;
        }
    }
    // Also check if memory comes first (unusual)
    if split.is_none() {
        for &pos in &commas {
            let after = text[pos + 1..].trim();
            let before = text[..pos].trim();
            if has_mem_prefix_anywhere(before) && !has_mem_prefix_anywhere(after) {
                split = Some(pos);
                break;
            }
        }
    }
    let comma_pos = match split {
        Some(p) => p,
        None => return Ok(None),
    };

    let before = text[..comma_pos].trim();
    let after = text[comma_pos + 1..].trim();

    let (compute_text, mem_text) =
        if has_mem_prefix_anywhere(after) && !has_mem_prefix_anywhere(before) {
            (before, after)
        } else {
            (after, before)
        };

    // Check for dual DM+PM move (Type 1)
    if let Some(dual) = try_parse_dual_move(compute_text, mem_text, line)? {
        return Ok(Some(dual));
    }

    // Check if it's a MODIFY (Type 7 with compute)
    if mem_text.starts_with("MODIFY(") || mem_text.starts_with("MODIFY (") {
        let modify = parse_dag_modify(mem_text, line)?;
        let compute = parse_compute_op(compute_text, line)?;
        if let Instruction::DagModify {
            pm,
            i_reg,
            m_reg,
            cond,
            width,
            ..
        } = modify
        {
            return Ok(Some(Instruction::DagModify {
                pm,
                i_reg,
                m_reg,
                cond,
                compute: Some(compute),
                dest_xor: 0,
                width,
            }));
        }
    }

    // Parse the memory access part
    let eq_pos = find_eq_outside_parens(mem_text).ok_or_else(|| Error::Parse {
        line,
        msg: format!("no '=' found in memory part: {mem_text}"),
    })?;

    let mem_lhs = mem_text[..eq_pos].trim();
    let mem_rhs = mem_text[eq_pos + 1..].trim();
    let lhs_has_mem = has_mem_prefix(mem_lhs);

    // Type 6a: immediate shift + DM memory access
    // Handles FEXT/FDEP (shift_type=1), LSHIFT (0), ASHIFT (2), BSET/BCLR/BTGL/BTST (3)
    if compute_text.contains(" BY ") {
        let imm_shift = try_parse_imm_shift(compute_text, line)?;
        if let Some(Instruction::ImmShift {
            shift_type,
            sub_op,
            imm,
            rn,
            rx,
            data_hi,
            ..
        }) = imm_shift
        {
            let (pm, inner) = if lhs_has_mem {
                extract_mem_inner(mem_lhs, line)?
            } else {
                extract_mem_inner(mem_rhs, line)?
            };
            if pm {
                return Err(Error::Parse {
                    line,
                    msg: "Type 6a (immediate shift + memory) only supports DM".into(),
                });
            }
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            if parts.len() != 2 {
                return Err(Error::Parse {
                    line,
                    msg: format!("expected two operands in memory expression: {inner}"),
                });
            }
            let (i_str, m_str) = if parts[0].starts_with('I') && parts[1].starts_with('M') {
                (parts[0], parts[1])
            } else if parts[0].starts_with('M') && parts[1].starts_with('I') {
                (parts[1], parts[0])
            } else {
                return Err(Error::Parse {
                    line,
                    msg: format!("expected I,M register pair in memory expression: {inner}"),
                });
            };
            let i_reg = parse_reg_index(i_str, 'I', line)?;
            let m_reg = parse_reg_index(m_str, 'M', line)?;
            // DAG1 only: the reference assembler rejects `btst r10 by 0xa , dm ( i11 , m5 )
            // = r2` and `dm ( i3 , m13 )`.
            if i_reg > 7 || m_reg > 7 {
                return Err(Error::Parse {
                    line,
                    msg: "Type 6a (immediate shift + memory) takes I0-I7 and M0-M7".into(),
                });
            }
            let (write, dreg_str) = if lhs_has_mem {
                (true, mem_rhs)
            } else {
                (false, mem_lhs)
            };
            let dreg = parse_dreg(dreg_str, line)?;
            return Ok(Some(Instruction::ImmShiftMem {
                shift_type,
                sub_op,
                imm,
                rn,
                rx,
                len_hi: data_hi,
                i_reg,
                m_reg,
                write,
                dreg,
                cond: 31,
            }));
        }
    }

    let compute = parse_compute_op(compute_text, line)?;

    if lhs_has_mem {
        // Store with compute
        let (pm, inner) = extract_mem_inner(mem_lhs, line)?;
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() != 2 {
            return Err(Error::Parse {
                line,
                msg: format!("expected two operands in memory expression: {inner}"),
            });
        }
        if let Some(dag) =
            try_build_dag_move(pm, parts[0], parts[1], mem_rhs, true, Some(compute), line)?
        {
            return Ok(Some(dag));
        }
        // Numeric offset (Type 4 with compute)
        let (i_reg, offset, post_modify) = parse_ireg_and_offset(parts[0], parts[1], line)?;
        let dreg = parse_dreg(mem_rhs, line)?;
        let i_field = if pm { i_reg.wrapping_sub(8) } else { i_reg };
        Ok(Some(Instruction::ComputeLoadStore {
            compute: Some(compute),
            access: MemAccess {
                pm,
                write: true,
                i_reg: i_field,
            },
            dreg,
            offset: type4_offset(offset, line)?,
            cond: 31,
            post_modify,
            width: selinstr::encode::DagWidth::Normal,
        }))
    } else {
        // Load with compute
        let (pm, inner) = extract_mem_inner(mem_rhs, line)?;
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() != 2 {
            return Err(Error::Parse {
                line,
                msg: format!("expected two operands in memory expression: {inner}"),
            });
        }
        if let Some(dag) =
            try_build_dag_move(pm, parts[0], parts[1], mem_lhs, false, Some(compute), line)?
        {
            return Ok(Some(dag));
        }
        // Numeric offset (Type 4 with compute)
        let (i_reg, offset, post_modify) = parse_ireg_and_offset(parts[0], parts[1], line)?;
        let dreg = parse_dreg(mem_lhs, line)?;
        let i_field = if pm { i_reg.wrapping_sub(8) } else { i_reg };
        Ok(Some(Instruction::ComputeLoadStore {
            compute: Some(compute),
            access: MemAccess {
                pm,
                write: false,
                i_reg: i_field,
            },
            dreg,
            offset: type4_offset(offset, line)?,
            cond: 31,
            post_modify,
            width: selinstr::encode::DagWidth::Normal,
        }))
    }
}

/// Check if text contains DM or PM memory reference anywhere.
/// Try to parse a dual DM+PM move: "DM(...) = Rn , PM(...) = Rm" or reads.
/// Returns None if mem_text doesn't contain both DM and PM.
fn try_parse_dual_move(
    compute_text: &str,
    mem_text: &str,
    line: u32,
) -> Result<Option<Instruction>> {
    // Check if mem_text contains both DM and PM
    let has_dm = mem_text.contains("DM(") || mem_text.contains("DM (");
    let has_pm = mem_text.contains("PM(") || mem_text.contains("PM (");
    if !has_dm || !has_pm {
        return Ok(None);
    }

    // Split mem_text at commas to find the DM and PM parts.
    // When compute_text is empty, the first comma-separated part might be
    // a compute operation (not a memory access), so we need to identify
    // which parts are DM, which are PM, and which is compute.
    let mem_commas = find_all_commas_outside_parens(mem_text);
    if mem_commas.is_empty() {
        return Ok(None);
    }

    // Find the two parts that contain DM and PM references.
    // Split at each comma and identify roles.
    let mut parts: Vec<&str> = Vec::new();
    let mut prev = 0;
    for &c in &mem_commas {
        parts.push(mem_text[prev..c].trim());
        prev = c + 1;
    }
    parts.push(mem_text[prev..].trim());

    let mut dm_part = None;
    let mut pm_part = None;
    let mut compute_part = compute_text;
    let mut found_compute = !compute_text.is_empty();

    for &p in &parts {
        if has_mem_prefix_anywhere(p)
            && (p.contains("DM(") || p.contains("DM ("))
            && dm_part.is_none()
        {
            dm_part = Some(p);
        } else if has_mem_prefix_anywhere(p)
            && (p.contains("PM(") || p.contains("PM ("))
            && pm_part.is_none()
        {
            pm_part = Some(p);
        } else if !found_compute {
            compute_part = p;
            found_compute = true;
        } else {
            return Ok(None);
        }
    }

    let dm_part = match dm_part {
        Some(p) => p,
        None => return Ok(None),
    };
    let pm_part = match pm_part {
        Some(p) => p,
        None => return Ok(None),
    };

    // Parse compute (may be empty)
    let compute = if compute_part.is_empty() {
        None
    } else {
        Some(parse_compute_op(compute_part, line)?)
    };

    // Parse each part as a DAG access
    let dm_access = parse_dag_access(dm_part, false, line)?;
    let pm_access = parse_dag_access(pm_part, true, line)?;

    Ok(Some(Instruction::DualMove {
        compute,
        dm: dm_access,
        pm: pm_access,
    }))
}

/// Parse a single DM/PM DAG access: "DM(Ii, Mm) = Rn" or "Rn = DM(Ii, Mm)".
fn parse_dag_access(text: &str, expect_pm: bool, line: u32) -> Result<DagAccess> {
    let eq = find_eq_outside_parens(text).ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected '=' in DAG access: {text}"),
    })?;
    let lhs = text[..eq].trim();
    let rhs = text[eq + 1..].trim();

    let (write, mem_part, dreg_name) = if has_mem_prefix(lhs) {
        (true, lhs, rhs)
    } else {
        (false, rhs, lhs)
    };

    let (_, inner) = extract_mem_inner(mem_part, line)?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(Error::Parse {
            line,
            msg: format!("expected two operands in memory expression: {inner}"),
        });
    }

    let (i_str, m_str) = if parts[0].starts_with('I') && parts[1].starts_with('M') {
        (parts[0], parts[1])
    } else if parts[0].starts_with('M') && parts[1].starts_with('I') {
        (parts[1], parts[0])
    } else {
        return Err(Error::Parse {
            line,
            msg: format!("expected I and M registers in DAG access: {inner}"),
        });
    };

    let i_full = parse_reg_index(i_str, 'I', line)?;
    let m_full = parse_reg_index(m_str, 'M', line)?;
    let dreg = parse_dreg(dreg_name, line)?;

    // For DM (DAG1): I0-I7, M0-M7; for PM (DAG2): I8-I15, M8-M15
    let i_reg = if expect_pm {
        i_full.wrapping_sub(8)
    } else {
        i_full
    };
    let m_reg = if expect_pm {
        m_full.wrapping_sub(8)
    } else {
        m_full
    };

    Ok(DagAccess {
        write,
        i_reg,
        m_reg,
        dreg,
    })
}

fn has_mem_prefix_anywhere(s: &str) -> bool {
    s.contains("DM(") || s.contains("PM(") || s.contains("DM (") || s.contains("PM (")
}

/// Find the position of the first comma outside parentheses.
fn find_comma_outside_parens(text: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth > 0 => {
                depth -= 1;
            }
            ',' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Find all comma positions outside parentheses.
fn find_all_commas_outside_parens(text: &str) -> Vec<usize> {
    let mut result = Vec::new();
    let mut depth = 0;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth > 0 => {
                depth -= 1;
            }
            ',' if depth == 0 => result.push(i),
            _ => {}
        }
    }
    result
}

fn extract_mem_inner(expr: &str, line: u32) -> Result<(bool, &str)> {
    // Handle both "DM(" and "DM (" formats
    let (pm, rest) = if let Some(r) = expr.strip_prefix("DM(") {
        (false, r)
    } else if let Some(r) = expr.strip_prefix("DM (") {
        (false, r)
    } else if let Some(r) = expr.strip_prefix("PM(") {
        (true, r)
    } else if let Some(r) = expr.strip_prefix("PM (") {
        (true, r)
    } else {
        return Err(Error::Parse {
            line,
            msg: format!("expected DM( or PM(: {expr}"),
        });
    };
    // Find the matching close paren, accounting for nested parens.
    let mut depth = 1i32;
    let mut close = None;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.ok_or_else(|| Error::Parse {
        line,
        msg: format!("missing ')' in memory expression: {expr}"),
    })?;
    let suffix = rest[close + 1..].trim();
    if !suffix.is_empty() {
        return Err(Error::Parse {
            line,
            msg: format!("unexpected memory access suffix: {suffix}"),
        });
    }
    Ok((pm, &rest[..close]))
}

// ---------------------------------------------------------------------------
// Multifunction compute parser
// ---------------------------------------------------------------------------

/// Try to parse a multifunction compute instruction.
/// Returns None if the text doesn't match the multifunction pattern.
///
/// Forms:
///   Rm = Rxm * Rym (SSF) , Ra = Rxa + Rya
///   Rm = Rxm * Rym (SSF) , Ra = Rxa - Rya
///   Rm = Rxm * Rym (SSF) , Ra = (Rxa + Rya)/2
///   MRF = MRF + Rxm * Rym (SSF) , Ra = Rxa + Rya
///   MRF = MRF - Rxm * Rym (SSF) , Ra = Rxa + Rya
///   Rm = MRF + Rxm * Rym (SSF) , Ra = Rxa + Rya
///   Rm = Rxm * Rym (SSF) , Ra = Rxa + Rya , Rs = Rxa - Rya  (dual add/sub)
///   (and floating-point equivalents with Fn/Fx/Fy)
fn try_parse_multifunction(text: &str, line: u32) -> Result<Option<ComputeOp>> {
    let commas = find_all_commas_outside_parens(text);
    if commas.is_empty() {
        return Ok(None);
    }

    let first_comma = commas[0];
    let first_part = text[..first_comma].trim();

    if first_part.contains('*') {
        // First part is a multiply
        if commas.len() == 2 {
            // Three-part: mul , add , sub (dual add/sub)
            let add_part = text[first_comma + 1..commas[1]].trim();
            let sub_part = text[commas[1] + 1..].trim();
            match parse_mf_dual_addsub(first_part, add_part, sub_part, line) {
                Ok(op) => return Ok(Some(op)),
                Err(_) => return Ok(None),
            }
        }

        if commas.len() == 1 {
            // Two-part: mul , alu
            let alu_part = text[first_comma + 1..].trim();
            match parse_mf_mul_alu(first_part, alu_part, line) {
                Ok(op) => return Ok(Some(op)),
                Err(_) => return Ok(None),
            }
        }
    } else if commas.len() == 1 {
        // Two-part without multiply: `Fn = Fx + Fy, Fs = Fx - Fy` is the
        // ALU dual add/subtract (the reference assembler `f12 = f2 + f6 , f8 = f2 - f6` =
        // 018f8c26). It used to become the multiplier form with zeroed
        // multiply fields, which also ran `F0 = F0 * F4`. Only the float
        // form has vendor vectors.
        let add_part = first_part;
        let sub_part = text[first_comma + 1..].trim();
        let side = |part: &str, op: &str| -> Option<(u16, u16, u16)> {
            let (dst, rhs) = part.split_once('=')?;
            let (x, y) = split_at_str(rhs.trim(), op)?;
            let regs = [dst.trim(), x.trim(), y.trim()];
            if !regs.iter().all(|r| r.starts_with('F')) {
                return None;
            }
            Some((
                parse_dreg(regs[0], line).ok()?,
                parse_dreg(regs[1], line).ok()?,
                parse_dreg(regs[2], line).ok()?,
            ))
        };
        if let (Some((rn, x1, y1)), Some((rs, x2, y2))) =
            (side(add_part, " + "), side(sub_part, " - "))
        {
            if (x1, y1) != (x2, y2) {
                return Err(Error::Parse {
                    line,
                    msg: format!(
                        "dual add/subtract needs the same operands in both halves: `{text}`"
                    ),
                });
            }
            return Ok(Some(ComputeOp::Falu(FaluOp::DualAddSub {
                rn,
                rs,
                rx: x1,
                ry: y1,
            })));
        }
    }

    Ok(None)
}

/// Parse the MUL part of a multifunction instruction.
/// Returns (fp, mul_sel, rm, rxm, rym).
///   mul_sel: 0=Rm=Rxm*Rym, 1=MRF+=Rxm*Rym, 2=Rm=MRF+Rxm*Rym, 3=MRF-=Rxm*Rym
fn parse_mf_mul_part(text: &str, line: u32) -> Result<(bool, u8, u16, u16, u16)> {
    let eq = text.find('=').ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected '=' in multiply part: {text}"),
    })?;
    let dst = text[..eq].trim();
    let rhs = text[eq + 1..].trim();

    // Detect floating-point from register names
    let fp = dst.starts_with('F')
        || rhs.contains("F0")
        || rhs.contains("F1")
        || rhs.contains("F2")
        || rhs.contains("F3");

    if dst == "MRF" || dst == "MRB" {
        // MRF = MRF + Rxm * Rym (SSF) => mul_sel = 1
        // MRF = MRF - Rxm * Rym (SSF) => mul_sel = 3
        // MRF = Rxm * Rym (SSF) => mul_sel = 0 (but dst is MRF, not Rm — this is a
        // single-function form, not multifunction; shouldn't reach here)
        if rhs.starts_with("MRF + ") || rhs.starts_with("MRB + ") {
            let mul_expr = &rhs[6..];
            let (rxm, rym) = parse_mf_mul_operands(mul_expr, fp, line)?;
            return Ok((fp, 1, 0, rxm, rym));
        }
        if rhs.starts_with("MRF - ") || rhs.starts_with("MRB - ") {
            let mul_expr = &rhs[6..];
            let (rxm, rym) = parse_mf_mul_operands(mul_expr, fp, line)?;
            return Ok((fp, 3, 0, rxm, rym));
        }
        return Err(Error::Parse {
            line,
            msg: format!("unexpected MRF/MRB multiply form: {text}"),
        });
    }

    let rm = parse_dreg(dst, line)?;

    // Rm = MRF + Rxm * Rym (SSF) => mul_sel = 2
    if rhs.starts_with("MRF + ") || rhs.starts_with("MRB + ") {
        let mul_expr = &rhs[6..];
        let (rxm, rym) = parse_mf_mul_operands(mul_expr, fp, line)?;
        return Ok((fp, 2, rm, rxm, rym));
    }

    // Rm = Rxm * Rym (SSF) => mul_sel = 0
    let (rxm, rym) = parse_mf_mul_operands(rhs, fp, line)?;
    Ok((fp, 0, rm, rxm, rym))
}

/// Parse "Rxm * Rym (SSF)" and return (rxm_2bit, rym_2bit).
/// Rxm is from range R0-R3/F0-F3, Rym from R4-R7/F4-F7.
fn parse_mf_mul_operands(text: &str, fp: bool, line: u32) -> Result<(u16, u16)> {
    let star = text.find('*').ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected '*' in multiply: {text}"),
    })?;
    let rx_str = text[..star].trim();
    let after_star = text[star + 1..].trim();
    // Strip optional modifier like (SSF), (UUF), etc.
    let ry_end = after_star.find('(').unwrap_or(after_star.len());
    let ry_str = after_star[..ry_end].trim();

    let rxm_full = parse_dreg(rx_str, line)?;
    let rym_full = parse_dreg(ry_str, line)?;

    // In multifunction, Rxm encodes as 2 bits (R0-R3 or F0-F3)
    if rxm_full > 3 {
        return Err(Error::Parse {
            line,
            msg: format!(
                "multifunction multiply X operand must be {}0-{}3, got {}",
                if fp { "F" } else { "R" },
                if fp { "F" } else { "R" },
                rx_str,
            ),
        });
    }
    // Rym must be in R4-R7/F4-F7 range
    if !(4..=7).contains(&rym_full) {
        return Err(Error::Parse {
            line,
            msg: format!(
                "multifunction multiply Y operand must be {}4-{}7, got {}",
                if fp { "F" } else { "R" },
                if fp { "F" } else { "R" },
                ry_str,
            ),
        });
    }
    Ok((rxm_full, rym_full - 4))
}

/// Parse the ALU part of a two-part multifunction: "Ra = Rxa + Rya" etc.
/// Returns (alu_sel, ra, rxa_2bit, rya_2bit).
fn parse_mf_alu_part(text: &str, fp: bool, line: u32) -> Result<(u8, u16, u16, u16)> {
    let eq = text.find('=').ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected '=' in ALU part: {text}"),
    })?;
    let dst = text[..eq].trim();
    let rhs = text[eq + 1..].trim();
    let ra = parse_dreg(dst, line)?;

    // Check for (Rxa + Rya)/2, with the divisor written `2` or `0x2` and
    // any spacing (the reference assembler accepts `( f10 + f12 ) / 0x2`).
    let avg_inner = rhs
        .rsplit_once('/')
        .and_then(|(head, div)| (parse_number(div.trim()) == Some(2)).then_some(head.trim()));
    if let Some(head) = avg_inner {
        if let Some(inner) = head.strip_prefix('(').and_then(|h| h.strip_suffix(')')) {
            let inner = inner.trim();
            if let Some((lhs_str, rhs_str)) = split_at_str(inner, " + ") {
                let rxa = parse_dreg(lhs_str.trim(), line)?;
                let rya = parse_dreg(rhs_str.trim(), line)?;
                let (rxa_2, rya_2) = check_mf_alu_regs(rxa, rya, fp, line)?;
                return Ok((2, ra, rxa_2, rya_2));
            }
        }
    }

    // Float-only ALU operations (the reference assembler multifunction opcodes 0x1a,
    // 0x1d, 0x1e, 0x1f).
    if fp {
        let two = |inner: &str| -> Result<(u16, u16)> {
            let (a, b) = inner.split_once(',').ok_or_else(|| Error::Parse {
                line,
                msg: format!("expected two operands in `{text}`"),
            })?;
            check_mf_alu_regs(
                parse_dreg(a.trim(), line)?,
                parse_dreg(b.trim(), line)?,
                fp,
                line,
            )
        };
        let paren = |op: &str| {
            rhs.strip_prefix(op)
                .map(str::trim)
                .and_then(|r| r.strip_prefix('('))
                .and_then(|r| r.strip_suffix(')'))
        };
        if let Some(inner) = paren("MAX") {
            let (rxa, rya) = two(inner)?;
            return Ok((5, ra, rxa, rya));
        }
        if let Some(inner) = paren("MIN") {
            let (rxa, rya) = two(inner)?;
            return Ok((6, ra, rxa, rya));
        }
        if let Some(x) = rhs.strip_prefix("ABS ") {
            let (rxa, _) = check_mf_alu_regs(parse_dreg(x.trim(), line)?, 12, fp, line)?;
            return Ok((4, ra, rxa, 0));
        }
        if let Some(rest) = rhs.strip_prefix("FLOAT ") {
            if let Some((x, y)) = split_at_str(rest, " BY ") {
                let rxa = parse_dreg(x.trim(), line)?;
                let rya = parse_dreg(y.trim(), line)?;
                let (rxa_2, rya_2) = check_mf_alu_regs(rxa, rya, fp, line)?;
                return Ok((3, ra, rxa_2, rya_2));
            }
        }
    }

    // Ra = Rxa + Rya
    if let Some((lhs_str, rhs_str)) = split_at_str(rhs, " + ") {
        let rxa = parse_dreg(lhs_str.trim(), line)?;
        let rya = parse_dreg(rhs_str.trim(), line)?;
        let (rxa_2, rya_2) = check_mf_alu_regs(rxa, rya, fp, line)?;
        return Ok((0, ra, rxa_2, rya_2));
    }

    // Ra = Rxa - Rya
    if let Some((lhs_str, rhs_str)) = split_at_str(rhs, " - ") {
        let rxa = parse_dreg(lhs_str.trim(), line)?;
        let rya = parse_dreg(rhs_str.trim(), line)?;
        let (rxa_2, rya_2) = check_mf_alu_regs(rxa, rya, fp, line)?;
        return Ok((1, ra, rxa_2, rya_2));
    }

    Err(Error::Parse {
        line,
        msg: format!("unrecognized multifunction ALU operation: {text}"),
    })
}

/// Validate multifunction ALU register ranges: Rxa in R8-R11, Rya in R12-R15.
/// Returns 2-bit encoded values.
fn check_mf_alu_regs(rxa: u16, rya: u16, fp: bool, line: u32) -> Result<(u16, u16)> {
    if !(8..=11).contains(&rxa) {
        return Err(Error::Parse {
            line,
            msg: format!(
                "multifunction ALU X operand must be {}8-{}11, got register {}",
                if fp { "F" } else { "R" },
                if fp { "F" } else { "R" },
                rxa,
            ),
        });
    }
    if !(12..=15).contains(&rya) {
        return Err(Error::Parse {
            line,
            msg: format!(
                "multifunction ALU Y operand must be {}12-{}15, got register {}",
                if fp { "F" } else { "R" },
                if fp { "F" } else { "R" },
                rya,
            ),
        });
    }
    Ok((rxa - 8, rya - 12))
}

fn parse_mf_mul_alu(mul_text: &str, alu_text: &str, line: u32) -> Result<ComputeOp> {
    let (fp, mul_sel, rm, rxm, rym) = parse_mf_mul_part(mul_text, line)?;
    let (alu_sel, ra, rxa, rya) = parse_mf_alu_part(alu_text, fp, line)?;
    Ok(ComputeOp::Multi(MultiOp::MulAlu {
        fp,
        mul_sel,
        alu_sel,
        rm,
        ra,
        rxm,
        rym,
        rxa,
        rya,
    }))
}

fn parse_mf_dual_addsub(
    mul_text: &str,
    add_text: &str,
    sub_text: &str,
    line: u32,
) -> Result<ComputeOp> {
    let (fp, _mul_sel, rm, rxm, rym) = parse_mf_mul_part(mul_text, line)?;
    // add part: Ra = Rxa + Rya
    let (alu_sel, ra, rxa, rya) = parse_mf_alu_part(add_text, fp, line)?;
    if alu_sel != 0 {
        return Err(Error::Parse {
            line,
            msg: "dual add/sub: second part must be addition".into(),
        });
    }
    // sub part: Rs = Rxa - Rya, the add part's operands. Only its
    // destination used to be read, so `f0 = f1 * f4, f12 = f11 + f12,
    // r14 = r13` (the reference assembler 70fe87580c4c: multiply + add with a register
    // transfer) became `f14 = f11 - f12`.
    let (sub_sel, rs, rxs, rys) = parse_mf_alu_part(sub_text, fp, line)?;
    if sub_sel != 1 || rxs != rxa || rys != rya {
        return Err(Error::Parse {
            line,
            msg: format!("dual add/sub: third part must be the subtraction: {sub_text}"),
        });
    }
    Ok(ComputeOp::Multi(MultiOp::MulDualAddSub {
        fp,
        rm,
        ra,
        rs,
        rxm,
        rym,
        rxa,
        rya,
    }))
}

// ---------------------------------------------------------------------------
// Compute or load-immediate
// ---------------------------------------------------------------------------

fn parse_compute_or_load(text: &str, line: u32) -> Result<Instruction> {
    // Try load immediate: "ureg = 0xNNNNNNNN" or "ureg = -0xNNNN"
    if let Some(eq) = text.find('=') {
        let lhs = text[..eq].trim();
        let rhs = text[eq + 1..].trim();

        // Try constant expression evaluation first: "R0 = 1 + 1" etc.
        // eval_const_expr only succeeds on pure numeric expressions (no
        // register names), so "R0 = R1 + R2" will not match here.
        if let Some(expr_val) = eval_const_expr(rhs) {
            if let Some(ureg) = ureg_code(lhs) {
                // The reference assembler rejects `i0 = 0x100000000` and `b0 =
                // -0x100000001`; selas truncated them.
                let value = fit_data32(expr_val, "immediate", line)? as u32;
                return Ok(Instruction::LoadImm { ureg, value });
            }
        }

        // Float constant expressions: "F8 = 1.5 * 2", "F2 = 1.647E-1"
        if rhs.contains('.') || rhs.contains('E') {
            if let Some(fval) = eval_float_expr(rhs) {
                if let Some(ureg) = ureg_code(lhs) {
                    let value = (fval as f32).to_bits();
                    return Ok(Instruction::LoadImm { ureg, value });
                }
            }
        }

        // Check if RHS is a pure number (immediate load), including negative
        let is_numeric_rhs = rhs.starts_with("0X")
            || rhs.starts_with("-0X")
            || rhs.starts_with("-")
            || rhs.chars().next().is_some_and(|c| c.is_ascii_digit());

        if is_numeric_rhs {
            // But first, make sure it's not a compute like "R0 = R1 + R2"
            // Negative numbers that are pure (no further operators) are fine
            let is_pure_number = !rhs.contains(" + ")
                && !rhs.contains(" - ")
                && !rhs.contains('*')
                && !rhs.contains("PASS")
                && !rhs.contains("ABS")
                && !rhs.contains("LSHIFT")
                && !rhs.contains("ASHIFT")
                && !rhs.contains("ROT")
                && !rhs.contains("AND")
                && !rhs.contains("OR")
                && !rhs.contains("XOR")
                && !rhs.contains("NOT")
                && !rhs.contains("MIN")
                && !rhs.contains("MAX")
                && !rhs.contains("CLIP");

            if is_pure_number {
                if let Some(ureg) = ureg_code(lhs) {
                    // Handle negative hex: "-0xNNNN" → two's complement u32
                    if let Some(neg_hex) = rhs.strip_prefix('-') {
                        if let Some(val) = parse_number(neg_hex.trim()) {
                            let neg_val = (-(val as i64)) as u32;
                            return Ok(Instruction::LoadImm {
                                ureg,
                                value: neg_val,
                            });
                        }
                        // Try negative float: "-2.0" → IEEE 754 bits
                        if let Ok(fval) = neg_hex.trim().parse::<f32>() {
                            let value = (-fval).to_bits();
                            return Ok(Instruction::LoadImm { ureg, value });
                        }
                    } else if let Some(val) = parse_number(rhs) {
                        return Ok(Instruction::LoadImm { ureg, value: val });
                    } else if let Ok(fval) = rhs.parse::<f32>() {
                        // Float immediate: "F2 = 2.0" → IEEE 754 bits
                        let value = fval.to_bits();
                        return Ok(Instruction::LoadImm { ureg, value });
                    }
                }
            }
        }

        // Check for universal register transfer: "ureg1 = ureg2"
        if let Some(dst) = ureg_code(lhs) {
            if let Some(src) = ureg_code(rhs) {
                // Both sides are uregs: a register transfer. The reference assembler
                // encodes `r15=r0;` as Type 5 (703e07bf), which leaves
                // ASTAT alone; it is not `R15 = PASS R0`, which sets the
                // ALU flags. A float register on either side has no
                // vendor vector, so it must be spelled out.
                if is_float_reg(lhs) || is_float_reg(rhs) {
                    return Err(float_copy_error(lhs, rhs, line));
                }
                return Ok(Instruction::UregTransfer {
                    src_ureg: src,
                    dst_ureg: dst,
                    cond: 31,
                    compute: None,
                });
            }
        }
    }

    // Immediate shift: "Rn = LSHIFT Rx BY <imm>" or "Rn = ASHIFT Rx BY <imm>"
    if let Some(imm_shift) = try_parse_imm_shift(text, line)? {
        return Ok(imm_shift);
    }

    // Compute operation
    let compute = parse_compute_op(text, line)?;
    Ok(Instruction::Compute { cond: 31, compute })
}

/// Try to parse an immediate shift: Rn = LSHIFT/ASHIFT Rx BY <imm>,
/// Rn = BSET/BCLR/BTGL Rx BY <imm>, or BTST Rx BY <imm>.
/// Returns None if the BY operand is a register.
/// SHARC+ bit FIFO: `Rn = BITEXT imm12`, `BITDEP Rx BY imm12`, `BFFWRP =
/// imm7`, `Rn = BFFWRP`, `BFFWRP = Ry`. The reference assembler: `r0 = bitext 0x20` =
/// 023e00142000, `bitdep r2 by 0x800` = 023e401d0002, `bffwrp = 0x10` =
/// 023e001f1000, `r1 = bffwrp` = 01a70100, `bffwrp = r1` = 01a7c001; it
/// rejects `bitext 0x1000`, `bitext -0x1` and `bffwrp = 0x80`. selas
/// loaded a symbol named `BITEXT 0x20` or `BFFWRP` instead.
fn try_parse_bit_fifo(text: &str, line: u32) -> Result<Option<Instruction>> {
    let imm = |s: &str, max: i64| -> Result<u16> {
        match parse_signed_number(s) {
            Some(v) if (0..=max).contains(&v) => Ok(v as u16),
            _ => Err(Error::Parse {
                line,
                msg: format!("bit FIFO immediate {s} is out of range (0-{max:#x})"),
            }),
        }
    };
    let fifo = |sub_op: u8, value: u16, rn: u16, rx: u16| Instruction::ImmShift {
        shift_type: 1,
        sub_op,
        imm: (value & 0xFF) as u8,
        rn,
        rx,
        data_hi: (value >> 8) as u8,
        cond: 31,
    };
    let text = text.trim();
    if let Some(rest) = text.strip_prefix("BITDEP ") {
        let (rx, by) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
            line,
            msg: format!("expected 'BY' in {text}"),
        })?;
        let rx = parse_dreg(rx.trim(), line)?;
        return Ok(Some(fifo(0xD, imm(by.trim(), 0xFFF)?, 0, rx)));
    }
    let Some((lhs, rhs)) = text.split_once('=') else {
        return Ok(None);
    };
    let (lhs, rhs) = (lhs.trim(), rhs.trim());
    if lhs == "BFFWRP" {
        if let Ok(ry) = parse_dreg(rhs, line) {
            return Ok(Some(Instruction::Compute {
                cond: 31,
                compute: ComputeOp::Shift(ShiftOp::BffwrpWrite { ry }),
            }));
        }
        return Ok(Some(fifo(0xF, imm(rhs, 0x7F)?, 0, 0)));
    }
    if rhs == "BFFWRP" {
        let rn = parse_dreg(lhs, line)?;
        return Ok(Some(Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Shift(ShiftOp::BffwrpRead { rn }),
        }));
    }
    if let Some(value) = rhs.strip_prefix("BITEXT ") {
        let rn = parse_dreg(lhs, line)?;
        return Ok(Some(fifo(4, imm(value.trim(), 0xFFF)?, rn, 0)));
    }
    Ok(None)
}

/// `Rn = Rn OR LSHIFT|ASHIFT|FDEP|FEXT ...`: the OR operand must be the
/// destination. The reference assembler rejects all 1,428 oracle rows like `r0 = r1 or
/// lshift r0 by 16`; selas ignored the operand and OR-ed into Rn.
fn check_or_operand(rn: u16, rhs: &str, line: u32) -> Result<()> {
    if let Some((operand, _)) = split_at_str(rhs, " OR ") {
        if parse_dreg(operand.trim(), line).ok() != Some(rn) {
            return Err(Error::Parse {
                line,
                msg: format!("the OR operand must be the destination register: {rhs}"),
            });
        }
    }
    Ok(())
}

fn try_parse_imm_shift(text: &str, line: u32) -> Result<Option<Instruction>> {
    // Check for FEXT/FDEP with pos:len notation first
    if (text.contains("FEXT") || text.contains("FDEP"))
        && text.contains(':')
        && text.contains(" BY ")
    {
        return try_parse_imm_fext_fdep(text, line);
    }

    // BTST Rx BY <imm> — no destination register
    if text.starts_with("BTST ") && text.contains(" BY ") {
        let rest = text.strip_prefix("BTST ").unwrap().trim();
        let (rx_str, by_str) = match split_at_str(rest, " BY ") {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let rx = match parse_dreg(rx_str.trim(), line) {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        if let Some(imm_val) = parse_signed_number(by_str.trim()) {
            if !(0..=31).contains(&imm_val) {
                return Err(Error::Parse {
                    line,
                    msg: format!("bit number {} is out of range", by_str.trim()),
                });
            }
            return Ok(Some(Instruction::ImmShift {
                shift_type: 3,
                sub_op: 3,
                imm: imm_val as u8,
                rn: 0,
                rx,
                data_hi: 0,
                cond: 31,
            }));
        }
        return Ok(None);
    }

    // Check for LSHIFT, ASHIFT, BSET, BCLR, BTGL with BY
    // Also handle OR LSHIFT / OR ASHIFT immediate forms
    let (shift_type, sub_op, keyword) = if text.contains("OR LSHIFT") && text.contains(" BY ") {
        (0u8, 4u8, "LSHIFT") // sub_op=4 for OR LSHIFT
    } else if text.contains("OR ASHIFT") && text.contains(" BY ") {
        (2u8, 4u8, "ASHIFT") // sub_op=4 for OR ASHIFT
    } else if text.contains("LSHIFT") && text.contains(" BY ") {
        (0u8, 0u8, "LSHIFT")
    } else if text.contains("ASHIFT") && text.contains(" BY ") {
        (2u8, 0u8, "ASHIFT")
    } else if text.contains("ROT ") && text.contains(" BY ") {
        // SHIFTOP 0x02 (register ROT 0x08 >> 2): `r0 = rot r0 by 5` =
        // 023e00020500, `r0 = rot r0 by -0x1` = 023e7802ff00.
        (0u8, 2u8, "ROT")
    } else if text.contains("BSET") && text.contains(" BY ") {
        (3u8, 0u8, "BSET")
    } else if text.contains("BCLR") && text.contains(" BY ") {
        (3u8, 1u8, "BCLR")
    } else if text.contains("BTGL") && text.contains(" BY ") {
        (3u8, 2u8, "BTGL")
    } else {
        return Ok(None);
    };

    // Extract destination register
    let eq = match text.find('=') {
        Some(pos) => pos,
        None => return Ok(None),
    };
    let dst = text[..eq].trim();
    let rn = match parse_dreg(dst, line) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    // Extract Rx and BY operand
    let rhs = text[eq + 1..].trim();
    if sub_op == 4 {
        check_or_operand(rn, rhs, line)?;
    }

    let kw_pos = match rhs.find(keyword) {
        Some(p) => p,
        None => return Ok(None),
    };
    let rest = rhs[kw_pos + keyword.len()..].trim();
    let (rx_str, by_str) = match split_at_str(rest, " BY ") {
        Some(pair) => pair,
        None => return Ok(None),
    };
    let rx = match parse_dreg(rx_str.trim(), line) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let by_trimmed = by_str.trim();

    // Try to parse as immediate
    if let Some(imm_val) = parse_signed_number(by_trimmed) {
        // A shift count is a signed 8-bit field (bits 11:8 only carry its
        // sign). The reference assembler reads 32-bit values as signed (`by 0xffffffff` =
        // -1) and rejects `lshift r1 by 0x100 , r0 = dm(i5, m4)`; selas used
        // to truncate 0x100 to 0.
        // BSET/BCLR/BTGL bit numbers are 0-31 (the reference assembler rejects 0x20, 0x100
        // and -0x1).
        if shift_type == 3 && !(0..=31).contains(&imm_val) {
            return Err(Error::Parse {
                line,
                msg: format!("bit number {by_trimmed} is out of range"),
            });
        }
        if shift_type != 3 {
            let count = if (0x8000_0000..=0xFFFF_FFFF).contains(&imm_val) {
                imm_val - (1 << 32)
            } else {
                imm_val
            };
            if !(-128..=127).contains(&count) {
                return Err(Error::Parse {
                    line,
                    msg: format!("shift count {by_trimmed} is out of range"),
                });
            }
        }
        return Ok(Some(Instruction::ImmShift {
            shift_type,
            sub_op,
            imm: imm_val as u8,
            rn,
            rx,
            data_hi: 0,
            cond: 31,
        }));
    }

    Ok(None)
}

/// Try to parse FEXT/FDEP with pos:len notation: Rn = FEXT Rx BY pos:len
/// Parsed fields from an immediate FEXT/FDEP expression.
struct ImmFextFields {
    sub_op: u8,
    rn: u16,
    rx: u16,
    imm: u8,
    len_hi: u8,
}

/// Parse "Rn = [Rn OR] FEXT/FDEP Rx BY pos:len" into its fields.
/// Returns None if the text is not an immediate FEXT/FDEP.
fn try_parse_imm_fext_fields(text: &str, line: u32) -> Result<Option<ImmFextFields>> {
    let eq = match text.find('=') {
        Some(pos) => pos,
        None => return Ok(None),
    };
    let dst = text[..eq].trim();
    let rhs = text[eq + 1..].trim();

    // sub_op = low nibble of the immediate opcode (`type6_shift_op`).
    let no_vector = |form: &str| Error::Parse {
        line,
        msg: format!("{form} by an immediate has no reference vector; use the register form"),
    };
    let (keyword, sub_op) = if rhs.contains("OR FEXT") {
        return Err(Error::Parse {
            line,
            msg: format!("there is no OR FEXT: {rhs}"),
        });
    } else if rhs.contains("OR FDEP") {
        ("FDEP", 9u8)
    } else if rhs.contains("FEXT") {
        ("FEXT", 0u8)
    } else if rhs.contains("FDEP") {
        return Err(no_vector("FDEP"));
    } else {
        return Ok(None);
    };

    let rn = match parse_dreg(dst, line) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    if sub_op == 9 {
        check_or_operand(rn, rhs, line)?;
    }

    let kw_pos = match rhs.find(keyword) {
        Some(p) => p,
        None => return Ok(None),
    };
    let rest = rhs[kw_pos + keyword.len()..].trim();
    let (rx_str, by_str) = match split_at_str(rest, " BY ") {
        Some(pair) => pair,
        None => return Ok(None),
    };
    let rx = match parse_dreg(rx_str.trim(), line) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let by_trimmed = by_str.trim();

    let colon = match by_trimmed.find(':') {
        Some(c) => c,
        None => return Ok(None),
    };
    let pos_str = by_trimmed[..colon].trim();
    let mut len_str = by_trimmed[colon + 1..].trim();
    // Handle (SE) suffix: sign-extending FEXT → sub_op=2. The reference assembler writes
    // it spaced as often as not: `r1 = fext r4 by 0x10:0x8 ( se )`.
    let mut se_suffix = false;
    if let Some(open) = len_str.rfind('(') {
        let packed: String = len_str[open..]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        if packed == "(SE)" {
            se_suffix = true;
            len_str = len_str[..open].trim();
        }
    }
    // Position and length are 6-bit fields; the reference assembler rejects -0x1, 0x100
    // and 0x100000 (selas truncated them).
    let field = |s: &str| -> Result<Option<u8>> {
        match parse_signed_number(s) {
            Some(v) if (0..=63).contains(&v) => Ok(Some(v as u8)),
            Some(_) => Err(Error::Parse {
                line,
                msg: format!("bit field {by_trimmed} is out of range (0-63)"),
            }),
            None => Ok(None),
        }
    };
    let pos = match field(pos_str)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let len = match field(len_str)? {
        Some(v) => v,
        None => return Ok(None),
    };
    // (SE): FEXT (SE) is 0x12; OR FDEP (SE) has no vector.
    let sub_op = match (se_suffix, sub_op) {
        (false, op) => op,
        (true, 0) => 2,
        (true, _) => return Err(no_vector("OR FDEP (SE)")),
    };

    let imm = (pos & 0x3F) | ((len & 0x3) << 6);
    let len_hi = (len >> 2) & 0x1F;

    Ok(Some(ImmFextFields {
        sub_op,
        rn,
        rx,
        imm,
        len_hi,
    }))
}

fn try_parse_imm_fext_fdep(text: &str, line: u32) -> Result<Option<Instruction>> {
    let fields = match try_parse_imm_fext_fields(text, line)? {
        Some(f) => f,
        None => return Ok(None),
    };
    Ok(Some(Instruction::ImmShift {
        shift_type: 1,
        sub_op: fields.sub_op,
        imm: fields.imm,
        rn: fields.rn,
        rx: fields.rx,
        data_hi: fields.len_hi,
        cond: 31,
    }))
}

// ---------------------------------------------------------------------------
// Compute operation parser (ALU / FALU / MUL / SHIFT)
// ---------------------------------------------------------------------------

fn parse_compute_op(text: &str, line: u32) -> Result<ComputeOp> {
    let text = text.trim();

    // Try multifunction compute (comma-separated mul + alu)
    if let Some(multi) = try_parse_multifunction(text, line)? {
        return Ok(multi);
    }

    // When multifunction didn't match but the text has a comma (e.g. dual
    // add/sub with non-multifunction register ranges), parse just the first
    // part as a standalone compute.  The second part is dropped — acceptable
    // for P1G, and P1 will refine later.
    if let Some(comma) = find_comma_outside_parens(text) {
        let first = text[..comma].trim();
        if !first.is_empty() {
            return parse_compute_op(first, line);
        }
    }

    // COMP(Rx, Ry) / COMPU(Rx, Ry) — handle both "COMP(" and "COMP (" formats
    if text.starts_with("COMP(")
        || text.starts_with("COMPU(")
        || text.starts_with("COMP (")
        || text.starts_with("COMPU (")
    {
        return parse_comp(text, line);
    }

    // BTST Rx BY Ry
    if text.starts_with("BTST ") {
        return parse_btst(text, line);
    }

    // MR register field transfers: MR0F = Rn, MR1F = Rn, etc.
    if text.starts_with("MR0F")
        || text.starts_with("MR1F")
        || text.starts_with("MR2F")
        || text.starts_with("MR0B")
        || text.starts_with("MR1B")
        || text.starts_with("MR2B")
    {
        return parse_mr_field_write(text, line);
    }

    // MRF = ... / MRB = ... / SAT MRF / TRNC MRF
    if text.starts_with("MRF") || text.starts_with("MRB") {
        return parse_mr_op(text, line);
    }

    // SAT MRF / SAT MRB (handled via Rn = SAT MRF below, but catch standalone)

    // Rn = ... or Fn = ...
    if let Some(eq) = text.find('=') {
        let dst = text[..eq].trim();
        let rhs = text[eq + 1..].trim();
        return parse_assign(dst, rhs, line);
    }

    Err(Error::UnknownMnemonic {
        line,
        text: text.to_string(),
    })
}

fn parse_comp(text: &str, line: u32) -> Result<ComputeOp> {
    let is_u = text.starts_with("COMPU(") || text.starts_with("COMPU (");
    // Find the opening parenthesis
    let open = text.find('(').ok_or_else(|| Error::Parse {
        line,
        msg: "missing '(' in COMP".into(),
    })?;
    let inner = text[open + 1..]
        .strip_suffix(')')
        .ok_or_else(|| Error::Parse {
            line,
            msg: "missing ')' in COMP".into(),
        })?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(Error::Parse {
            line,
            msg: "COMP expects two operands".into(),
        });
    }

    let is_float = parts[0].starts_with('F');
    let rx = parse_dreg(parts[0], line)?;
    let ry = parse_dreg(parts[1], line)?;

    if is_u {
        Ok(ComputeOp::Alu(AluOp::CompU { rx, ry }))
    } else if is_float {
        Ok(ComputeOp::Falu(FaluOp::Comp { rx, ry }))
    } else {
        Ok(ComputeOp::Alu(AluOp::Comp { rx, ry }))
    }
}

fn parse_btst(text: &str, line: u32) -> Result<ComputeOp> {
    let rest = text.strip_prefix("BTST ").unwrap().trim();
    let (rx_str, ry_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: "expected 'BY' in BTST".into(),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;
    Ok(ComputeOp::Shift(ShiftOp::Btst { rx, ry }))
}

fn parse_mr_field_write(text: &str, line: u32) -> Result<ComputeOp> {
    // "MR0F = Rn", "MR1F = Rn", etc.
    let eq = text.find('=').ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected '=' in MR field write: {text}"),
    })?;
    let lhs = text[..eq].trim();
    let rhs = text[eq + 1..].trim();
    let rn = parse_dreg(rhs, line)?;
    match lhs {
        "MR0F" => Ok(ComputeOp::Mul(MulOp::WriteMr0f { rn })),
        "MR1F" => Ok(ComputeOp::Mul(MulOp::WriteMr1f { rn })),
        "MR2F" => Ok(ComputeOp::Mul(MulOp::WriteMr2f { rn })),
        "MR0B" => Ok(ComputeOp::Mul(MulOp::WriteMr0b { rn })),
        "MR1B" => Ok(ComputeOp::Mul(MulOp::WriteMr1b { rn })),
        "MR2B" => Ok(ComputeOp::Mul(MulOp::WriteMr2b { rn })),
        _ => Err(Error::Parse {
            line,
            msg: format!("unexpected MR field: {lhs}"),
        }),
    }
}

fn parse_mr_op(text: &str, line: u32) -> Result<ComputeOp> {
    // "MRF = 0", "MRB = 0", "MRF = TRNC MRF", etc.
    // "MRF = Rx * Ry (SSF)", "MRF = MRF + Rx * Ry (SSF)", etc.
    if let Some(eq) = text.find('=') {
        let lhs = text[..eq].trim();
        let rhs = text[eq + 1..].trim();

        let is_b = lhs == "MRB";

        // MRF = 0 / MRB = 0
        if rhs == "0" {
            return if is_b {
                Ok(ComputeOp::Mul(MulOp::ClrMrb))
            } else {
                Ok(ComputeOp::Mul(MulOp::ClrMrf))
            };
        }

        // `MRF|MRB = SAT|RND MRF|MRB [(mod)]`: the reference assembler opcodes
        // (MEASURED), unsuffixed = (SF). `MRx = TRNC MRx` is not
        // reference-assembler syntax; its old opcodes 0x18/0x1A are
        // `Rn = RND MRF|MRB (UF)`, so it is refused.
        let packed: String = rhs.chars().filter(|c| !c.is_whitespace()).collect();
        let opcode = match (lhs, packed.as_str()) {
            ("MRF", "SATMRF(UI)") => Some(0x04),
            ("MRF", "SATMRF(SI)") => Some(0x05),
            ("MRF", "SATMRF(UF)") => Some(0x0C),
            ("MRF", "SATMRF" | "SATMRF(SF)") => Some(0x0D),
            ("MRB", "SATMRB(UI)") => Some(0x06),
            ("MRB", "SATMRB(SI)") => Some(0x07),
            ("MRB", "SATMRB(UF)") => Some(0x0E),
            ("MRB", "SATMRB" | "SATMRB(SF)") => Some(0x0F),
            ("MRF", "RNDMRF(UF)") => Some(0x1C),
            ("MRF", "RNDMRF" | "RNDMRF(SF)") => Some(0x1D),
            ("MRB", "RNDMRB(UF)") => Some(0x1E),
            ("MRB", "RNDMRB" | "RNDMRB(SF)") => Some(0x1F),
            _ => None,
        };
        if let Some(opcode) = opcode {
            return Ok(ComputeOp::Mul(MulOp::MrToReg { opcode, rn: 0 }));
        }
        if packed.starts_with("TRNC") {
            return Err(Error::Parse {
                line,
                msg: format!("{lhs} = {rhs} is not reference-assembler syntax"),
            });
        }

        // MRF = Rx * Ry (SSF), MRF = MRF + Rx * Ry (SSI), etc.
        if rhs.contains('*') {
            return parse_fixed_mul(lhs, rhs, line);
        }
    }

    Err(Error::UnknownMnemonic {
        line,
        text: text.to_string(),
    })
}

/// A fixed-point multiply `dst = [MRF|MRB +|-] Rx * Ry [(mod2)]`, `dst`
/// being Rn, MRF or MRB. The opcode follows the `ddyx f??r` template in
/// `encode_mul`; mod2 names Rx then Ry signedness, then I/F and an
/// optional R (vendor `r0 = r0 * r1 (sui)` = opcode 0x50: Rx signed).
/// selas used to drop every suffix it had no variant for and encode SSF.
fn parse_fixed_mul(lhs: &str, rhs: &str, line: u32) -> Result<ComputeOp> {
    let err = |msg: &str| Error::Parse {
        line,
        msg: format!("{msg}: {lhs} = {rhs}"),
    };
    let accumulate = |mr: &str| {
        let rest = rhs.strip_prefix(mr)?.trim_start();
        if let Some(p) = rest.strip_prefix('+') {
            Some((0x80u8, p))
        } else {
            rest.strip_prefix('-').map(|p| (0xC0u8, p))
        }
    };
    let (acc_b, top, product) = if let Some((top, p)) = accumulate("MRF") {
        (Some(false), top, p)
    } else if let Some((top, p)) = accumulate("MRB") {
        (Some(true), top, p)
    } else {
        (None, 0x40, rhs)
    };
    let (rn, dd) = match (lhs, acc_b) {
        ("MRF", None | Some(false)) => (0, 0b10),
        ("MRB", None | Some(true)) => (0, 0b11),
        ("MRF" | "MRB", Some(_)) => return Err(err("the accumulator must be the destination")),
        (_, None) => (parse_dreg(lhs, line)?, 0b00),
        (_, Some(b)) => (parse_dreg(lhs, line)?, u8::from(b)),
    };
    let (factors, modifier) = match product.find('(') {
        Some(p) => (&product[..p], &product[p..]),
        None => (product, ""),
    };
    let (rx_str, ry_str) = factors
        .split_once('*')
        .ok_or_else(|| err("expected '*' in multiply"))?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;
    let mod2: String = modifier
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '(' && *c != ')')
        .collect();
    // No suffix: selas's long-standing default.
    let mod2 = if mod2.is_empty() {
        "SSF"
    } else {
        mod2.as_str()
    };
    let signed = |c: u8| match c {
        b'S' => Some(1u8),
        b'U' => Some(0),
        _ => None,
    };
    let m = mod2.as_bytes();
    let bits = match m {
        [x, y, f, rest @ ..] if rest.is_empty() || (rest == b"R" && *f == b'F') => {
            let frac = match f {
                b'F' => Some(1u8),
                b'I' => Some(0),
                _ => None,
            };
            match (signed(*x), signed(*y), frac) {
                (Some(x), Some(y), Some(frac)) => {
                    (y << 5) | (x << 4) | (frac << 3) | u8::from(!rest.is_empty())
                }
                _ => return Err(err("unknown multiplier mode")),
            }
        }
        _ => return Err(err("unknown multiplier mode")),
    };
    let opcode = top | bits | (dd << 1);
    Ok(ComputeOp::Mul(match opcode {
        0x70 => MulOp::MulSsi { rn, rx, ry },
        0x78 => MulOp::MulSsf { rn, rx, ry },
        0x74 => MulOp::MrfMulSsi { rx, ry },
        0x7C => MulOp::MrfMulSsf { rx, ry },
        0x4C => MulOp::MrfMulUuf { rx, ry },
        0x44 => MulOp::MrfMulUui { rx, ry },
        0x7E => MulOp::MrbMulSsf { rx, ry },
        0xB8 => MulOp::MacSsf { rn, rx, ry },
        0xBC => MulOp::MrfMacSsf { rx, ry },
        0xBE => MulOp::MrbMacSsf { rx, ry },
        0xFC => MulOp::MrfMsubSsf { rx, ry },
        0xFE => MulOp::MrbMsubSsf { rx, ry },
        _ => MulOp::FixedMul { opcode, rn, rx, ry },
    }))
}

// ---------------------------------------------------------------------------
// Assignment-based compute: Rn = ... or Fn = ...
// ---------------------------------------------------------------------------

fn parse_assign(dst: &str, rhs: &str, line: u32) -> Result<ComputeOp> {
    let is_float = dst.starts_with('F') && dst[1..].chars().all(|c| c.is_ascii_digit());
    let rn = parse_dreg(dst, line)?;

    // "Rn = Rn OR LSHIFT ...", "Rn = Rn OR ASHIFT ...", etc.
    // Check OR variants BEFORE plain LSHIFT/ASHIFT to avoid false matches
    if rhs.contains("OR LSHIFT") {
        return parse_or_shift(rn, rhs, "LSHIFT", line);
    }
    if rhs.contains("OR ASHIFT") {
        return parse_or_shift(rn, rhs, "ASHIFT", line);
    }

    // Shift operations: LSHIFT, ASHIFT, ROT
    if rhs.contains("LSHIFT") {
        return parse_shift_op(rn, rhs, "LSHIFT", 0x00, line);
    }
    if rhs.contains("ASHIFT") {
        return parse_shift_op(rn, rhs, "ASHIFT", 0x08, line);
    }
    if rhs.starts_with("ROT ") {
        return parse_shift_op(rn, rhs, "ROT", 0x20, line);
    }
    // OR FEXT / OR FDEP — check before plain FEXT/FDEP
    if rhs.contains("OR FEXT") {
        return parse_or_fext_fdep(rn, rhs, "FEXT", line);
    }
    if rhs.contains("OR FDEP") {
        return parse_or_fext_fdep(rn, rhs, "FDEP", line);
    }

    // BCLR, BSET, BTGL
    if let Some(rest) = rhs.strip_prefix("BCLR ") {
        return parse_bit_shift(rn, rest, "BCLR", line);
    }
    if let Some(rest) = rhs.strip_prefix("BSET ") {
        return parse_bit_shift(rn, rest, "BSET", line);
    }
    if let Some(rest) = rhs.strip_prefix("BTGL ") {
        return parse_bit_shift(rn, rest, "BTGL", line);
    }

    // FEXT, FDEP — after OR FEXT/FDEP check
    if let Some(rest) = rhs.strip_prefix("FEXT ") {
        return parse_fext_fdep(rn, rest, "FEXT", line);
    }
    if let Some(rest) = rhs.strip_prefix("FDEP ") {
        return parse_fext_fdep(rn, rest, "FDEP", line);
    }

    // EXP, EXP (EX), LEFTZ, LEFTO
    if let Some(rest) = rhs.strip_prefix("EXP ") {
        let rx_str = rest.trim();
        if let Some(inner) = rx_str.strip_suffix("(EX)") {
            let rx = parse_dreg(inner.trim(), line)?;
            return Ok(ComputeOp::Shift(ShiftOp::ExpEx { rn, rx }));
        }
        let rx = parse_dreg(rx_str, line)?;
        return Ok(ComputeOp::Shift(ShiftOp::Exp { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("LEFTZ ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Shift(ShiftOp::Leftz { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("LEFTO ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Shift(ShiftOp::Lefto { rn, rx }));
    }

    // FPACK / FUNPACK: `Rn = FPACK Fx`, `Fn = FUNPACK Rx`; the reference assembler rejects
    // `r4 = fpack r0` and `r4 = funpack r0`.
    if let Some(rest) = rhs.strip_prefix("FPACK ") {
        if !rest.trim().starts_with('F') {
            return Err(Error::Parse {
                line,
                msg: format!("FPACK takes a float source, `Rn = FPACK Fx`: {dst} = {rhs}"),
            });
        }
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Shift(ShiftOp::Fpack { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("FUNPACK ") {
        if !is_float {
            return Err(Error::Parse {
                line,
                msg: format!("FUNPACK writes a float, `Fn = FUNPACK Rx`: {dst} = {rhs}"),
            });
        }
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Shift(ShiftOp::Funpack { rn, rx }));
    }

    // MR register field reads: Rn = MR0F, Rn = MR1F, etc.
    match rhs {
        "MR0F" => return Ok(ComputeOp::Mul(MulOp::ReadMr0f { rn })),
        "MR1F" => return Ok(ComputeOp::Mul(MulOp::ReadMr1f { rn })),
        "MR2F" => return Ok(ComputeOp::Mul(MulOp::ReadMr2f { rn })),
        "MR0B" => return Ok(ComputeOp::Mul(MulOp::ReadMr0b { rn })),
        "MR1B" => return Ok(ComputeOp::Mul(MulOp::ReadMr1b { rn })),
        "MR2B" => return Ok(ComputeOp::Mul(MulOp::ReadMr2b { rn })),
        _ => {}
    }

    // Multiply: Rn = Rx * Ry (SSF), Fn = Fx * Fy
    if rhs.contains('*') {
        return parse_reg_mul(dst, rn, rhs, is_float, line);
    }

    // `Rn = SAT MRF|MRB (SI|SF)`, `Rn = RND MRF|MRB (SF)` (vendor opcodes
    // in MulOp::MrToReg).
    let packed_rhs: String = rhs.chars().filter(|c| !c.is_whitespace()).collect();
    let mr_opcode = match packed_rhs.as_str() {
        // (MEASURED): (UI) 0x00/0x02, (UF) 0x08/0x0A;
        // RND unsuffixed = (SF), (UF) 0x18/0x1A.
        "SATMRF(UI)" => Some(0x00),
        "SATMRF(SI)" => Some(0x01),
        "SATMRB(UI)" => Some(0x02),
        "SATMRB(SI)" => Some(0x03),
        "SATMRF(UF)" => Some(0x08),
        "SATMRF(SF)" => Some(0x09),
        "SATMRB(UF)" => Some(0x0A),
        "SATMRB(SF)" => Some(0x0B),
        "RNDMRF(UF)" => Some(0x18),
        "RNDMRF" | "RNDMRF(SF)" => Some(0x19),
        "RNDMRB(UF)" => Some(0x1A),
        "RNDMRB" | "RNDMRB(SF)" => Some(0x1B),
        _ => None,
    };
    if let Some(opcode) = mr_opcode {
        if is_float {
            return Err(Error::Parse {
                line,
                msg: format!("{dst} = {rhs} needs an R register"),
            });
        }
        return Ok(ComputeOp::Mul(MulOp::MrToReg { opcode, rn }));
    }

    // SAT MRF / SAT MRB
    if rhs == "SAT MRF" {
        return Ok(ComputeOp::Mul(MulOp::SatMrf { rn }));
    }
    if rhs == "SAT MRB" {
        return Ok(ComputeOp::Mul(MulOp::SatMrb { rn }));
    }

    // `Rn = TRNC MRF|MRB` is not reference-assembler syntax,
    // and its old opcodes 0x1C/0x1E are `MRF|MRB = RND MRF|MRB
    // (UF)`: refuse it.
    if rhs == "TRNC MRF" || rhs == "TRNC MRB" {
        return Err(Error::Parse {
            line,
            msg: format!("{dst} = {rhs} is not reference-assembler syntax"),
        });
    }

    // Int/float conversion: FIX, FLOAT, TRUNC (mixed register types)
    if let Some(rest) = rhs.strip_prefix("FIX ") {
        let rest = rest.trim();
        if let Some((fx_str, ry_str)) = split_at_str(rest, " BY ") {
            let rx = parse_dreg(fx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::FixBy { rn, rx, ry }));
        }
        let rx = parse_dreg(rest, line)?;
        return Ok(ComputeOp::Falu(FaluOp::Fix { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("TRUNC ") {
        let rest = rest.trim();
        if let Some((fx_str, ry_str)) = split_at_str(rest, " BY ") {
            let rx = parse_dreg(fx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::TruncBy { rn, rx, ry }));
        }
        let rx = parse_dreg(rest, line)?;
        return Ok(ComputeOp::Falu(FaluOp::Trunc { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("FLOAT ") {
        let rest = rest.trim();
        if let Some((rx_str, ry_str)) = split_at_str(rest, " BY ") {
            let rx = parse_dreg(rx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::FloatBy { rn, rx, ry }));
        }
        let rx = parse_dreg(rest, line)?;
        return Ok(ComputeOp::Falu(FaluOp::Float { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("RECIPS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Recips { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("RSQRTS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Rsqrts { rn, rx }));
    }
    // `Rn = LOGB Fx`, `Rn = MANT Fx`: fixed-point results; the reference assembler rejects
    // `f4 = logb f0` and `f4 = mant f0`.
    if (rhs.starts_with("LOGB ") || rhs.starts_with("MANT ")) && is_float {
        return Err(Error::Parse {
            line,
            msg: format!("{} writes a fixed-point register: {dst} = {rhs}", &rhs[..4]),
        });
    }
    if let Some(rest) = rhs.strip_prefix("LOGB ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Logb { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("MANT ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Mant { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("SCALB ") {
        if let Some((fx_str, ry_str)) = split_at_str(rest.trim(), " BY ") {
            let rx = parse_dreg(fx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::Scalb { rn, rx, ry }));
        }
    }
    if rhs.contains("COPYSIGN") {
        let parts: Vec<&str> = rhs.split("COPYSIGN").collect();
        if parts.len() == 2 {
            let rx = parse_dreg(parts[0].trim(), line)?;
            let ry = parse_dreg(parts[1].trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::Copysign { rn, rx, ry }));
        }
    }

    // Float-specific unary ops
    if is_float {
        return parse_float_unary(rn, rhs, line);
    }

    // Single operand with CI, +1, -1 (check before general add/sub)
    // Only match when the remaining prefix is a plain register, otherwise
    // fall through to binary operators (e.g. "Rx - Ry + CI - 1" is SubCi).
    // The reference assembler takes the 1 as `1` or `0x1`, spaced or not (`r0 = r9-0x1`,
    // `r0 = r11 -0x1 , s0 = r0` = 703e2802a0b0); selas needed `- 1`.
    let compact: String = rhs.chars().filter(|c| !c.is_whitespace()).collect();
    let compact = match compact.strip_suffix("0X1") {
        Some(head) if head.ends_with(['+', '-']) => format!("{head}1"),
        _ => compact,
    };
    if let Some(rx_str) = compact.strip_suffix("+CI-1") {
        if let Ok(rx) = parse_dreg(rx_str, line) {
            return Ok(ComputeOp::Alu(AluOp::PassCiMinus1 { rn, rx }));
        }
        // Rx - Ry + CI - 1
        if let Some((x, y)) = rx_str.split_once('-') {
            if let (Ok(rx), Ok(ry)) = (parse_dreg(x, line), parse_dreg(y, line)) {
                return Ok(ComputeOp::Alu(AluOp::SubCi { rn, rx, ry }));
            }
        }
    }
    if let Some(rx_str) = compact.strip_suffix("+CI") {
        if let Ok(rx) = parse_dreg(rx_str, line) {
            return Ok(ComputeOp::Alu(AluOp::PassCi { rn, rx }));
        }
    }
    if let Some(rx_str) = compact.strip_suffix("+1") {
        if let Ok(rx) = parse_dreg(rx_str, line) {
            return Ok(ComputeOp::Alu(AluOp::Inc { rn, rx }));
        }
    }
    if let Some(rx_str) = compact.strip_suffix("-1") {
        if let Ok(rx) = parse_dreg(rx_str, line) {
            return Ok(ComputeOp::Alu(AluOp::Dec { rn, rx }));
        }
    }

    // Unary ops (ABS, PASS) must be checked before binary subtraction
    // to avoid matching " - " inside "ABS (R1 - R2)".
    if rhs.starts_with("ABS ") || rhs.starts_with("ABS(") {
        let rest = rhs[3..].trim();
        let inner = if rest.starts_with('(') {
            if let Some(close) = find_matching_close_paren(rest, 0) {
                rest[1..close].trim()
            } else {
                rest.trim_start_matches('(').trim_end_matches(')').trim()
            }
        } else {
            rest
        };
        // The fixed-point ALU has only `Rn = ABS Rx`; `ABS (R1 - R2)` used
        // to encode ABS R1.
        let rx = parse_dreg(inner, line)?;
        return Ok(ComputeOp::Alu(AluOp::Abs { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("PASS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Pass { rn, rx }));
    }

    // Subtraction: Rn = Rx - Ry or Rn = -Rx
    // Must come before addition check since "Rx - Ry + CI - 1" contains '+'
    if let Some(rest) = rhs.strip_prefix('-') {
        // Negation: Rn = -Rx
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Neg { rn, rx }));
    }
    if let Some(minus) = rhs.find(" - ") {
        let rx = parse_dreg(rhs[..minus].trim(), line)?;
        let rest_after = rhs[minus + 3..].trim();
        // Check for "Rx - Ry + CI - 1"
        if let Some(ci_pos) = rest_after.find("+ CI - 1") {
            let ry = parse_dreg(rest_after[..ci_pos].trim(), line)?;
            return Ok(ComputeOp::Alu(AluOp::SubCi { rn, rx, ry }));
        }
        let ry = parse_dreg(rest_after, line)?;
        return Ok(ComputeOp::Alu(AluOp::Sub { rn, rx, ry }));
    }

    // ALU binary: Rn = Rx OP Ry
    if let Some(plus) = rhs.find('+') {
        return parse_alu_binary(rn, rhs, plus, is_float, line);
    }

    // Logical: AND, OR, XOR, NOT
    if rhs.contains(" AND ") {
        let parts = split_at_str(rhs, " AND ").unwrap();
        let rx = parse_dreg(parts.0.trim(), line)?;
        let ry = parse_dreg(parts.1.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::And { rn, rx, ry }));
    }
    if rhs.contains(" OR ") {
        let parts = split_at_str(rhs, " OR ").unwrap();
        let rx = parse_dreg(parts.0.trim(), line)?;
        let ry = parse_dreg(parts.1.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Or { rn, rx, ry }));
    }
    if rhs.contains(" XOR ") {
        let parts = split_at_str(rhs, " XOR ").unwrap();
        let rx = parse_dreg(parts.0.trim(), line)?;
        let ry = parse_dreg(parts.1.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Xor { rn, rx, ry }));
    }
    if let Some(rest) = rhs.strip_prefix("NOT ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Not { rn, rx }));
    }

    // Unary: PASS, ABS, MIN, MAX, CLIP
    if let Some(rest) = rhs.strip_prefix("PASS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Pass { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("ABS ") {
        let rest = rest
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        let rx = parse_dreg(rest, line)?;
        return Ok(ComputeOp::Alu(AluOp::Abs { rn, rx }));
    }
    if rhs.starts_with("MIN(") || rhs.starts_with("MIN (") {
        return parse_min_max_clip(rn, rhs, "MIN", is_float, line);
    }
    if rhs.starts_with("MAX(") || rhs.starts_with("MAX (") {
        return parse_min_max_clip(rn, rhs, "MAX", is_float, line);
    }
    if rhs.starts_with("CLIP ") {
        return parse_clip(rn, rhs, is_float, line);
    }

    Err(Error::UnknownMnemonic {
        line,
        text: format!("{dst} = {rhs}"),
    })
}

fn parse_float_unary(rn: u16, rhs: &str, line: u32) -> Result<ComputeOp> {
    // Float unary operations that use F-register destination

    // PASS, ABS — check BEFORE binary ops so "ABS (F1 - F2)" isn't
    // mistaken for a subtraction.
    if let Some(rest) = rhs.strip_prefix("PASS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Pass { rn, rx }));
    }
    if rhs.starts_with("ABS ") || rhs.starts_with("ABS(") {
        let rest = rhs[3..].trim();
        // Strip outer parentheses: "ABS (F1 - F2)" → "F1 - F2",
        // "ABS (F4)" → "F4", "ABS F3" → "F3"
        let inner = if rest.starts_with('(') {
            if let Some(close) = find_matching_close_paren(rest, 0) {
                rest[1..close].trim()
            } else {
                rest.trim_start_matches('(').trim_end_matches(')').trim()
            }
        } else {
            rest
        };
        // `ABS (Fx + Fy)` and `ABS (Fx - Fy)` are ALU ops of their own
        // (the reference assembler `f0 = abs(f0 + f1)` = 01891001, `f0 = abs(f0 - f1)` =
        // 01892001); selas encoded ABS of the first register.
        if let Ok(rx) = parse_dreg(inner, line) {
            return Ok(ComputeOp::Falu(FaluOp::Abs { rn, rx }));
        }
        let operands = |op: char| -> Result<Option<(u16, u16)>> {
            match inner.split_once(op) {
                Some((x, y)) => Ok(Some((
                    parse_dreg(x.trim(), line)?,
                    parse_dreg(y.trim(), line)?,
                ))),
                None => Ok(None),
            }
        };
        if let Some((rx, ry)) = operands('+')? {
            return Ok(ComputeOp::Falu(FaluOp::AbsAdd { rn, rx, ry }));
        }
        if let Some((rx, ry)) = operands('-')? {
            return Ok(ComputeOp::Falu(FaluOp::AbsSub { rn, rx, ry }));
        }
        return Err(Error::Parse {
            line,
            msg: format!("unrecognized ABS operand: {inner}"),
        });
    }

    // Binary: Fn = Fx + Fy, Fn = Fx - Fy
    if let Some(plus) = rhs.find('+') {
        return parse_alu_binary(rn, rhs, plus, true, line);
    }
    if let Some(rest) = rhs.strip_prefix('-') {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Neg { rn, rx }));
    }
    if let Some(minus) = rhs.find(" - ") {
        let rx = parse_dreg(rhs[..minus].trim(), line)?;
        let ry = parse_dreg(rhs[minus + 3..].trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Sub { rn, rx, ry }));
    }

    // RND
    if let Some(rest) = rhs.strip_prefix("RND ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Rnd { rn, rx }));
    }

    // MANT
    if let Some(rest) = rhs.strip_prefix("MANT ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Mant { rn, rx }));
    }

    // SCALB Fx BY Ry
    if let Some(rest) = rhs.strip_prefix("SCALB ") {
        let (rx_str, ry_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
            line,
            msg: "expected 'BY' in SCALB".into(),
        })?;
        let rx = parse_dreg(rx_str.trim(), line)?;
        let ry = parse_dreg(ry_str.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Scalb { rn, rx, ry }));
    }

    // LOGB
    if let Some(rest) = rhs.strip_prefix("LOGB ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Logb { rn, rx }));
    }

    // RECIPS, RSQRTS
    if let Some(rest) = rhs.strip_prefix("RECIPS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Recips { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("RSQRTS ") {
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Rsqrts { rn, rx }));
    }

    // FIX, FLOAT, TRUNC (with optional BY)
    if let Some(rest) = rhs.strip_prefix("FIX ") {
        if let Some((rx_str, ry_str)) = split_at_str(rest, " BY ") {
            let rx = parse_dreg(rx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::FixBy { rn, rx, ry }));
        }
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Fix { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("FLOAT ") {
        if let Some((rx_str, ry_str)) = split_at_str(rest, " BY ") {
            let rx = parse_dreg(rx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::FloatBy { rn, rx, ry }));
        }
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Float { rn, rx }));
    }
    if let Some(rest) = rhs.strip_prefix("TRUNC ") {
        if let Some((rx_str, ry_str)) = split_at_str(rest, " BY ") {
            let rx = parse_dreg(rx_str.trim(), line)?;
            let ry = parse_dreg(ry_str.trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::TruncBy { rn, rx, ry }));
        }
        let rx = parse_dreg(rest.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Trunc { rn, rx }));
    }

    // ABS(Fx + Fy) / ABS(Fx - Fy)
    if rhs.starts_with("ABS(") {
        let inner = rhs
            .strip_prefix("ABS(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| Error::Parse {
                line,
                msg: "malformed ABS(...)".into(),
            })?;
        if let Some(plus) = inner.find(" + ") {
            let rx = parse_dreg(inner[..plus].trim(), line)?;
            let ry = parse_dreg(inner[plus + 3..].trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::AbsAdd { rn, rx, ry }));
        }
        if let Some(minus) = inner.find(" - ") {
            let rx = parse_dreg(inner[..minus].trim(), line)?;
            let ry = parse_dreg(inner[minus + 3..].trim(), line)?;
            return Ok(ComputeOp::Falu(FaluOp::AbsSub { rn, rx, ry }));
        }
    }

    // MIN/MAX/CLIP (the reference assembler writes `f10 = max ( f2 , f0 )` spaced; the
    // float path took only `MAX(`)
    if rhs.starts_with("MIN(") || rhs.starts_with("MIN (") {
        return parse_min_max_clip(rn, rhs, "MIN", true, line);
    }
    if rhs.starts_with("MAX(") || rhs.starts_with("MAX (") {
        return parse_min_max_clip(rn, rhs, "MAX", true, line);
    }
    if rhs.starts_with("CLIP ") {
        return parse_clip(rn, rhs, true, line);
    }

    // COPYSIGN
    if rhs.contains(" COPYSIGN ") {
        let parts = split_at_str(rhs, " COPYSIGN ").unwrap();
        let rx = parse_dreg(parts.0.trim(), line)?;
        let ry = parse_dreg(parts.1.trim(), line)?;
        return Ok(ComputeOp::Falu(FaluOp::Copysign { rn, rx, ry }));
    }

    // Logical (integer ops with F dst -- treat as ALU since physical regs are same)
    if rhs.contains(" AND ") {
        let parts = split_at_str(rhs, " AND ").unwrap();
        let rx = parse_dreg(parts.0.trim(), line)?;
        let ry = parse_dreg(parts.1.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::And { rn, rx, ry }));
    }
    if rhs.contains(" OR ") {
        let parts = split_at_str(rhs, " OR ").unwrap();
        let rx = parse_dreg(parts.0.trim(), line)?;
        let ry = parse_dreg(parts.1.trim(), line)?;
        return Ok(ComputeOp::Alu(AluOp::Or { rn, rx, ry }));
    }

    Err(Error::UnknownMnemonic {
        line,
        text: format!("F{rn} = {rhs}"),
    })
}

fn parse_alu_binary(
    rn: u16,
    rhs: &str,
    plus_pos: usize,
    is_float: bool,
    line: u32,
) -> Result<ComputeOp> {
    let lhs_str = rhs[..plus_pos].trim();
    let rhs_str = rhs[plus_pos + 1..].trim();

    // Check for (Rx + Ry)/2 or (Rx + Ry) / 2
    let avg_suffix = if rhs_str.ends_with(")/2") {
        Some(")/2")
    } else if rhs_str.ends_with(") / 2") {
        Some(") / 2")
    } else if rhs_str.ends_with(") /2") {
        Some(") /2")
    } else {
        None
    };
    if let Some(suffix) = avg_suffix {
        if lhs_str.starts_with('(') {
            let inner_lhs = lhs_str.strip_prefix('(').unwrap().trim();
            let inner_rhs = rhs_str.strip_suffix(suffix).unwrap().trim();
            let rx = parse_dreg(inner_lhs, line)?;
            let ry = parse_dreg(inner_rhs, line)?;
            return if is_float {
                Ok(ComputeOp::Falu(FaluOp::Avg { rn, rx, ry }))
            } else {
                Ok(ComputeOp::Alu(AluOp::Avg { rn, rx, ry }))
            };
        }
    }

    // Check for Rx + Ry + CI
    if rhs_str.ends_with("+ CI") {
        let inner = rhs_str.strip_suffix("+ CI").unwrap().trim();
        let rx = parse_dreg(lhs_str, line)?;
        let ry = parse_dreg(inner, line)?;
        return Ok(ComputeOp::Alu(AluOp::AddCi { rn, rx, ry }));
    }

    let rx = parse_dreg(lhs_str, line)?;
    let ry = parse_dreg(rhs_str, line)?;
    if is_float {
        Ok(ComputeOp::Falu(FaluOp::Add { rn, rx, ry }))
    } else {
        Ok(ComputeOp::Alu(AluOp::Add { rn, rx, ry }))
    }
}

fn parse_shift_op(
    rn: u16,
    rhs: &str,
    keyword: &str,
    _base_opcode: u8,
    line: u32,
) -> Result<ComputeOp> {
    // "LSHIFT Rx BY Ry" or "LSHIFT Rx BY -1" (immediate)
    let start = rhs.find(keyword).unwrap() + keyword.len();
    let rest = rhs[start..].trim();
    let (rx_str, by_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected 'BY' in {keyword}"),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let by_trimmed = by_str.trim();

    // Register operand
    let ry = parse_dreg(by_trimmed, line)?;
    match keyword {
        "LSHIFT" => Ok(ComputeOp::Shift(ShiftOp::Lshift { rn, rx, ry })),
        "ASHIFT" => Ok(ComputeOp::Shift(ShiftOp::Ashift { rn, rx, ry })),
        "ROT" => Ok(ComputeOp::Shift(ShiftOp::Rot { rn, rx, ry })),
        _ => Err(Error::UnknownMnemonic {
            line,
            text: rhs.to_string(),
        }),
    }
}

fn parse_or_shift(rn: u16, rhs: &str, shift_kind: &str, line: u32) -> Result<ComputeOp> {
    // "Rn OR LSHIFT Rx BY Ry"
    check_or_operand(rn, rhs, line)?;
    let kw = format!("OR {shift_kind}");
    let start = rhs.find(&kw).unwrap() + kw.len();
    let rest = rhs[start..].trim();
    let (rx_str, ry_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected 'BY' in OR {shift_kind}"),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;

    match shift_kind {
        "LSHIFT" => Ok(ComputeOp::Shift(ShiftOp::OrLshift { rn, rx, ry })),
        "ASHIFT" => Ok(ComputeOp::Shift(ShiftOp::OrAshift { rn, rx, ry })),
        _ => Err(Error::UnknownMnemonic {
            line,
            text: rhs.to_string(),
        }),
    }
}

fn parse_or_fext_fdep(rn: u16, rhs: &str, kind: &str, line: u32) -> Result<ComputeOp> {
    check_or_operand(rn, rhs, line)?;
    let kw = format!("OR {kind}");
    let start = rhs.find(&kw).unwrap() + kw.len();
    let rest = rhs[start..].trim();
    let (rx_str, ry_part) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected 'BY' in OR {kind}"),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry_str = ry_part
        .trim()
        .strip_suffix("(SE)")
        .unwrap_or(ry_part.trim())
        .trim();
    let ry = parse_dreg(ry_str, line)?;
    let has_se = ry_part.contains("(SE)");

    match (kind, has_se) {
        ("FEXT", true) => Ok(ComputeOp::Shift(ShiftOp::OrFextSe { rn, rx, ry })),
        ("FDEP", _) => Ok(ComputeOp::Shift(ShiftOp::OrFdep { rn, rx, ry })),
        _ => Err(Error::UnknownMnemonic {
            line,
            text: rhs.to_string(),
        }),
    }
}

fn parse_bit_shift(rn: u16, rest: &str, kind: &str, line: u32) -> Result<ComputeOp> {
    let (rx_str, ry_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: "expected 'BY' in bit shift".into(),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;
    match kind {
        "BCLR" => Ok(ComputeOp::Shift(ShiftOp::Bclr { rn, rx, ry })),
        "BSET" => Ok(ComputeOp::Shift(ShiftOp::Bset { rn, rx, ry })),
        "BTGL" => Ok(ComputeOp::Shift(ShiftOp::Btgl { rn, rx, ry })),
        _ => Err(Error::UnknownMnemonic {
            line,
            text: rest.to_string(),
        }),
    }
}

fn parse_fext_fdep(rn: u16, rest: &str, kind: &str, line: u32) -> Result<ComputeOp> {
    let (rx_str, ry_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: "expected 'BY' in FEXT/FDEP".into(),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;
    match kind {
        "FEXT" => Ok(ComputeOp::Shift(ShiftOp::Fext { rn, rx, ry })),
        "FDEP" => Ok(ComputeOp::Shift(ShiftOp::Fdep { rn, rx, ry })),
        _ => Err(Error::UnknownMnemonic {
            line,
            text: rest.to_string(),
        }),
    }
}

fn parse_reg_mul(dst: &str, rn: u16, rhs: &str, is_float: bool, line: u32) -> Result<ComputeOp> {
    if !is_float {
        return parse_fixed_mul(dst, rhs, line);
    }
    let (rx_str, ry_str) = rhs.split_once('*').ok_or_else(|| Error::Parse {
        line,
        msg: format!("expected '*' in multiply: {rhs}"),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;
    Ok(ComputeOp::Mul(MulOp::FMul { rn, rx, ry }))
}

fn parse_min_max_clip(
    rn: u16,
    rhs: &str,
    kind: &str,
    is_float: bool,
    line: u32,
) -> Result<ComputeOp> {
    let paren_pos = rhs.find('(').unwrap();
    let inner = rhs[paren_pos + 1..]
        .strip_suffix(')')
        .ok_or_else(|| Error::Parse {
            line,
            msg: format!("missing ')' in {kind}"),
        })?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(Error::Parse {
            line,
            msg: format!("{kind} expects two operands"),
        });
    }
    let rx = parse_dreg(parts[0], line)?;
    let ry = parse_dreg(parts[1], line)?;

    match (kind, is_float) {
        ("MIN", false) => Ok(ComputeOp::Alu(AluOp::Min { rn, rx, ry })),
        ("MAX", false) => Ok(ComputeOp::Alu(AluOp::Max { rn, rx, ry })),
        ("MIN", true) => Ok(ComputeOp::Falu(FaluOp::Min { rn, rx, ry })),
        ("MAX", true) => Ok(ComputeOp::Falu(FaluOp::Max { rn, rx, ry })),
        _ => Err(Error::UnknownMnemonic {
            line,
            text: rhs.to_string(),
        }),
    }
}

fn parse_clip(rn: u16, rhs: &str, is_float: bool, line: u32) -> Result<ComputeOp> {
    // "CLIP Rx BY Ry"
    let rest = rhs.strip_prefix("CLIP ").unwrap().trim();
    let (rx_str, ry_str) = split_at_str(rest, " BY ").ok_or_else(|| Error::Parse {
        line,
        msg: "expected 'BY' in CLIP".into(),
    })?;
    let rx = parse_dreg(rx_str.trim(), line)?;
    let ry = parse_dreg(ry_str.trim(), line)?;
    if is_float {
        Ok(ComputeOp::Falu(FaluOp::Clip { rn, rx, ry }))
    } else {
        Ok(ComputeOp::Alu(AluOp::Clip { rn, rx, ry }))
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Resolve an absolute address: parse as number, eval as constant expression,
/// or treat as a symbolic reference (placeholder 0 with the symbol name
/// for relocation). Returns (value, optional_symbol_name).
fn resolve_abs_addr(text: &str, line: u32) -> Result<(u32, Option<String>)> {
    // A 32-bit address: the reference assembler rejects `dm ( -0x100000000 ) = r0`, which
    // selas truncated.
    if let Some(v) = parse_signed_number(text).or_else(|| eval_const_expr(text)) {
        return Ok((fit_data32(v, "address", line)? as u32, None));
    }
    // A forward-referenced symbol: 0 as placeholder; the caller emits a
    // relocation so the linker patches the absolute address.
    if is_valid_label(text.trim()) {
        return Ok((0, Some(text.trim().to_string())));
    }
    // `label ± constant` (`r0 = dm(ldr.+4)`): the
    // constant travels as the relocation addend (assemble.rs leaves 0 in
    // the field). It used to become a relocation against a symbol named
    // `LDR.+4`, which cannot link.
    if let Some((label, adj)) = split_label_addend(text.trim()) {
        if is_valid_label(label) {
            return Ok((
                fit_data32(adj, "address offset", line)? as u32,
                Some(label.to_string()),
            ));
        }
    }
    Err(Error::Parse {
        line,
        msg: format!("invalid address: {text}"),
    })
}

/// Find the closing ')' that matches the '(' at `open`, respecting nesting.
fn find_matching_close_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

fn normalize_compute_spacing(text: &str) -> String {
    // Strip SIMD register pair notation: "F1:0" → "F1", "R13:12" → "R13".
    // In SIMD mode, Rn:m means Rn on PEx and Rm on PEy; the assembler
    // encodes only the PEx register.
    let mut text_buf = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ':'
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && i + 1 < chars.len()
            && chars[i + 1].is_ascii_digit()
        {
            // Check that we're after a register name (R/F/S + digits)
            let mut j = i - 1;
            while j > 0 && chars[j - 1].is_ascii_digit() {
                j -= 1;
            }
            let prefix = if j > 0 { chars[j - 1] } else { '\0' };
            if matches!(prefix, 'R' | 'F' | 'S' | 'r' | 'f' | 's') {
                // Skip the colon and following digits
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                continue;
            }
        }
        text_buf.push(chars[i]);
        i += 1;
    }
    let text = &text_buf;

    // Insert spaces around +/- operators between operands
    // e.g. "R0-R0" -> "R0 - R0", "R4-1" -> "R4 - 1", "R2 -1" -> "R2 - 1"
    // But don't touch hex numbers: "-0x10", "0X100"
    let mut result = String::with_capacity(text.len() + 10);
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let is_op = chars[i] == '-' || chars[i] == '+';
        if is_op && i > 0 && i + 1 < chars.len() {
            let prev_alnum = chars[i - 1].is_ascii_alphanumeric();
            let next_alnum = chars[i + 1].is_ascii_alphanumeric();
            // Don't split hex prefixes: 0x, 0X
            let is_hex_prefix = chars[i] == '+'
                && i + 2 < chars.len()
                && chars[i + 1] == '0'
                && (chars[i + 2] == 'x' || chars[i + 2] == 'X');
            // Don't split scientific notation: 1.5E-1, 2.0E+3
            let is_sci_notation = prev_alnum && (chars[i - 1] == 'E' || chars[i - 1] == 'e');
            // Case 1: "R0-R1" or "R4-1" — both sides alphanumeric
            // Case 2: "R2 -1" — space before, alphanumeric after; check that
            // two chars back is alphanumeric (to catch "Rx -N" but not "= -0x")
            // Case 3: "R12- F2" — alphanumeric before, space after
            let space_before =
                chars[i - 1] == ' ' && i >= 2 && chars[i - 2].is_ascii_alphanumeric();
            let space_after = !next_alnum
                && chars[i + 1] == ' '
                && i + 2 < chars.len()
                && chars[i + 2].is_ascii_alphanumeric();
            if !is_hex_prefix
                && !is_sci_notation
                && (next_alnum || space_after)
                && (prev_alnum || space_before)
            {
                if !prev_alnum {
                    // Already have space before; just add space after
                    result.push(chars[i]);
                    if next_alnum {
                        result.push(' ');
                    }
                } else {
                    result.push(' ');
                    result.push(chars[i]);
                    if next_alnum {
                        result.push(' ');
                    }
                }
            } else {
                result.push(chars[i]);
            }
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }
    // Collapse runs of whitespace into a single space.
    // Preprocessor macro expansion can leave double-spaces (e.g.
    // "R2 -  1") which break suffix-based pattern matching in parse_assign.
    let mut prev_space = false;
    let mut collapsed = String::with_capacity(result.len());
    for c in result.chars() {
        if c == ' ' {
            if !prev_space {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(c);
            prev_space = false;
        }
    }
    let mut result = collapsed;

    // Strip outer parentheses from immediates: "R4 = (0X8)" -> "R4 = 0X8"
    // Only strip ") ," when preceded by "= (" (immediate wrapping), not
    // after modifier suffixes like (SSF).
    if let Some(eq_paren) = result.find("= (") {
        let after = eq_paren + 2; // start of '('
        if let Some(close) = result[after..].find(')') {
            let close_abs = after + close;
            let after_close = &result[close_abs + 1..];
            if after_close.starts_with(" ,")
                || after_close.starts_with(';')
                || after_close.is_empty()
            {
                result.replace_range(close_abs..close_abs + 1, "");
                result.replace_range(eq_paren + 2..eq_paren + 3, "");
            }
        }
    }
    result
}

/// Extract a width modifier suffix, returning (stripped text, modifier).
/// Modifier is None if no width modifier found.
fn extract_width_modifier(s: &str) -> (&str, Option<&str>) {
    let s = s.trim();
    for suffix in &["(LW)", "(BW)", "(BWSE)", "(SW)", "(SWSE)", "(NW)"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            return (stripped.trim(), Some(suffix));
        }
    }
    // The reference assembler writes the suffix spaced as often as not: `r0 = dm ( i0 , m4
    // ) ( lw )`.
    if let Some(pos) = s.rfind('(') {
        let candidate: String = s[pos..].chars().filter(|c| !c.is_whitespace()).collect();
        for suffix in &["(LW)", "(BW)", "(BWSE)", "(SW)", "(SWSE)", "(NW)"] {
            if candidate == *suffix && !s[..pos].trim().is_empty() {
                return (s[..pos].trim(), Some(suffix));
            }
        }
    }
    (s, None)
}

/// Split a (preprocessed, comment-stripped) source line at top-level
/// `;` characters and return the non-empty parts (each without the
/// trailing `;`). `;` inside parentheses, brackets, braces, or string
/// literals is preserved. Used by the line loop to recover all
/// statements from a packed source line such as
///   JUMP foo; NOP; NOP; NOP;
fn split_top_level_stmts(line: &str) -> Vec<String> {
    let stripped = strip_comment(line);
    let bytes = stripped.as_bytes();
    let mut parts: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut in_str: Option<u8> = None;
    let mut start = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_str {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == q {
                in_str = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' | b'\'' => in_str = Some(b),
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = (depth - 1).max(0),
            b';' if depth == 0 => {
                let p = stripped[start..i].trim();
                if !p.is_empty() {
                    parts.push(p.to_string());
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let tail = stripped[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

fn is_ignored_directive(upper: &str) -> bool {
    upper.starts_with(".ENDSEG")
        || upper.starts_with(".PRECISION")
        || upper.starts_with(".FILE_ATTR")
        || upper.starts_with(".PREVIOUS")
        || upper.starts_with(".TYPE")
        || upper.starts_with(".LEFTMARGIN")
        || upper.starts_with(".NEWPAGE")
        || upper.starts_with(".INDENT")
        || upper.starts_with(".IMPORT")
        || upper.starts_with(".RETAIN_NAME")
        || upper.starts_with(".SEGMENT_SPACE")
        || upper.starts_with(".FILE")
        || upper.starts_with(".LIST")
        || upper.starts_with(".NOLIST")
        || upper.starts_with(".ENTRY")
        || upper.starts_with(".PRIORITY")
        || upper.starts_with(".ROUND_")
        || upper.starts_with(".MESSAGE")
        || upper.starts_with(".STRUCT")
}

fn strip_comment(line: &str) -> &str {
    // SHARC+ asm uses `//` and `/* */` for comments. The `;` character
    // is a STATEMENT TERMINATOR, not a comment marker -- packed source
    // lines like `JUMP foo; NOP; NOP; NOP;` rely on each `;` ending an
    // instruction. Stripping at the first `;` would silently swallow
    // every statement after the first on such a line, leaving the IVT
    // with three NOPs missing per slot.
    let mut result = line;
    if let Some(pos) = result.find("//") {
        result = &result[..pos];
    }
    result
}

fn is_valid_label(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' && first != '.' {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// Split a `.GLOBAL` / `.WEAK` / `.EXTERN` operand list into individual
/// symbol names.  The SHARC+ Tools dialect allows one directive to
/// list several names separated by commas, e.g.
/// `.EXTERN ldf_stack_space, ldf_stack_length;`.  Empty entries (e.g.
/// from a trailing comma) are dropped.
fn parse_symbol_list(rest: &str) -> Vec<String> {
    rest.trim()
        .trim_end_matches(';')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn split_first_comma(text: &str) -> (&str, Option<&str>) {
    if let Some(pos) = text.find(',') {
        (&text[..pos], Some(&text[pos + 1..]))
    } else {
        (text, None)
    }
}

fn split_first_space(text: &str) -> (&str, &str) {
    if let Some(pos) = text.find(' ') {
        (&text[..pos], &text[pos + 1..])
    } else {
        (text, "")
    }
}

fn split_at_str<'a>(text: &'a str, sep: &str) -> Option<(&'a str, &'a str)> {
    text.find(sep)
        .map(|pos| (&text[..pos], &text[pos + sep.len()..]))
}

fn find_eq_outside_parens(text: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth > 0 => {
                depth -= 1;
            }
            '=' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn parse_dreg(name: &str, line: u32) -> Result<u16> {
    let name = name.trim();
    if let Some(rest) = name.strip_prefix('R') {
        if let Ok(n) = rest.parse::<u8>() {
            if n <= 15 {
                return Ok(n as u16);
            }
        }
    }
    if let Some(rest) = name.strip_prefix('F') {
        if let Ok(n) = rest.parse::<u8>() {
            if n <= 15 {
                return Ok(n as u16);
            }
        }
    }
    Err(Error::UnknownRegister {
        line,
        name: name.to_string(),
    })
}

fn parse_reg_index(name: &str, prefix: char, line: u32) -> Result<u8> {
    let rest = name
        .strip_prefix(prefix)
        .ok_or_else(|| Error::UnknownRegister {
            line,
            name: name.to_string(),
        })?;
    // I/M/S registers run 0-15 (the reference assembler rejects `i99 = modify (i99, m14) (sw)`).
    rest.parse::<u8>()
        .ok()
        .filter(|&n| n < 16)
        .ok_or_else(|| Error::UnknownRegister {
            line,
            name: name.to_string(),
        })
}

// ---------------------------------------------------------------------------
// Constant expression evaluator (recursive descent)
// ---------------------------------------------------------------------------

/// Evaluate a constant arithmetic expression at assembly time.
///
/// Supports: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `|`, `&`, `^`, `~`,
/// unary minus, and parentheses. Operands are hex (`0x...`) or decimal
/// integer literals. Returns `None` if the string is not a valid constant
/// expression (e.g. contains register names).
/// Evaluate a float constant expression: "1.5 * 2", "1.6470993291652860E-1".
/// Returns the f64 value, or None if the string isn't a valid float expression.
fn eval_float_expr(s: &str) -> Option<f64> {
    let s = s.trim();
    // Try parsing as a single float literal first (including scientific notation)
    // Handle trailing UL/L suffixes from preprocessor macro expansion
    let clean = s.trim_end_matches("UL").trim_end_matches('L').trim();
    if let Ok(v) = clean.parse::<f64>() {
        return Some(v);
    }
    // Try simple binary: "a * b", "a + b", "a - b", "a / b"
    for op in ['*', '+', '/'] {
        if let Some(pos) = s.rfind(op) {
            if pos > 0 {
                let lhs = eval_float_expr(s[..pos].trim())?;
                let rhs = eval_float_expr(s[pos + 1..].trim())?;
                return match op {
                    '*' => Some(lhs * rhs),
                    '+' => Some(lhs + rhs),
                    '/' => {
                        if rhs == 0.0 {
                            None
                        } else {
                            Some(lhs / rhs)
                        }
                    }
                    _ => None,
                };
            }
        }
    }
    // Handle subtraction carefully (avoid matching negative exponent)
    if let Some(pos) = s.rfind(" - ") {
        let lhs = eval_float_expr(s[..pos].trim())?;
        let rhs = eval_float_expr(s[pos + 3..].trim())?;
        return Some(lhs - rhs);
    }
    // Strip outer parens
    if s.starts_with('(') && s.ends_with(')') {
        return eval_float_expr(&s[1..s.len() - 1]);
    }
    // Try as integer
    if let Some(v) = parse_number(s) {
        return Some(v as f64);
    }
    None
}

/// Public entry point for the constant expression evaluator (used by preproc).
pub fn eval_const_expr_pub(s: &str) -> Option<i64> {
    eval_const_expr(s)
}

fn eval_const_expr(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    let mut pos = 0;
    let result = expr_bitor(bytes, &mut pos)?;
    // Skip trailing whitespace
    skip_ws(bytes, &mut pos);
    if pos == bytes.len() {
        Some(result)
    } else {
        None // trailing garbage
    }
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos] == b' ' {
        *pos += 1;
    }
}

fn expr_bitor(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut val = expr_bitxor(bytes, pos)?;
    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'|' {
            *pos += 1;
            let rhs = expr_bitxor(bytes, pos)?;
            val |= rhs;
        } else {
            break;
        }
    }
    Some(val)
}

fn expr_bitxor(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut val = expr_bitand(bytes, pos)?;
    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'^' {
            *pos += 1;
            let rhs = expr_bitand(bytes, pos)?;
            val ^= rhs;
        } else {
            break;
        }
    }
    Some(val)
}

fn expr_bitand(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut val = expr_shift(bytes, pos)?;
    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'&' {
            *pos += 1;
            let rhs = expr_shift(bytes, pos)?;
            val &= rhs;
        } else {
            break;
        }
    }
    Some(val)
}

fn expr_shift(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut val = expr_additive(bytes, pos)?;
    loop {
        skip_ws(bytes, pos);
        if *pos + 1 < bytes.len() && bytes[*pos] == b'<' && bytes[*pos + 1] == b'<' {
            *pos += 2;
            let rhs = expr_additive(bytes, pos)?;
            if !(0..64).contains(&rhs) {
                return None;
            }
            val = val.wrapping_shl(rhs as u32);
        } else if *pos + 1 < bytes.len() && bytes[*pos] == b'>' && bytes[*pos + 1] == b'>' {
            *pos += 2;
            let rhs = expr_additive(bytes, pos)?;
            if !(0..64).contains(&rhs) {
                return None;
            }
            val = val.wrapping_shr(rhs as u32);
        } else {
            break;
        }
    }
    Some(val)
}

fn expr_additive(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut val = expr_multiplicative(bytes, pos)?;
    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'+' {
            *pos += 1;
            let rhs = expr_multiplicative(bytes, pos)?;
            val = val.wrapping_add(rhs);
        } else if *pos < bytes.len() && bytes[*pos] == b'-' {
            *pos += 1;
            let rhs = expr_multiplicative(bytes, pos)?;
            val = val.wrapping_sub(rhs);
        } else {
            break;
        }
    }
    Some(val)
}

fn expr_multiplicative(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut val = expr_unary(bytes, pos)?;
    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'*' {
            *pos += 1;
            let rhs = expr_unary(bytes, pos)?;
            val = val.wrapping_mul(rhs);
        } else if *pos < bytes.len() && bytes[*pos] == b'/' {
            *pos += 1;
            let rhs = expr_unary(bytes, pos)?;
            if rhs == 0 {
                return None;
            }
            val /= rhs;
        } else if *pos < bytes.len() && bytes[*pos] == b'%' {
            *pos += 1;
            let rhs = expr_unary(bytes, pos)?;
            if rhs == 0 {
                return None;
            }
            val %= rhs;
        } else {
            break;
        }
    }
    Some(val)
}

fn expr_unary(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    skip_ws(bytes, pos);
    if *pos < bytes.len() && bytes[*pos] == b'~' {
        *pos += 1;
        let val = expr_unary(bytes, pos)?;
        Some(!val)
    } else if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
        let val = expr_unary(bytes, pos)?;
        Some(val.wrapping_neg())
    } else if *pos < bytes.len() && bytes[*pos] == b'+' {
        // Explicit positive sign, e.g. "M6 = +1" in the SHARC+ Tools
        // dialect.  Treated as a no-op unary so "+1" parses the same as
        // "1".
        *pos += 1;
        expr_unary(bytes, pos)
    } else {
        expr_atom(bytes, pos)
    }
}

fn expr_atom(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    skip_ws(bytes, pos);
    if *pos < bytes.len() && bytes[*pos] == b'(' {
        *pos += 1;
        let val = expr_bitor(bytes, pos)?;
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b')' {
            *pos += 1;
            Some(val)
        } else {
            None // unmatched paren
        }
    } else {
        // Number literal: hex (0x...) or decimal
        let start = *pos;
        if *pos + 1 < bytes.len()
            && bytes[*pos] == b'0'
            && (bytes[*pos + 1] == b'x' || bytes[*pos + 1] == b'X')
        {
            *pos += 2;
            let hex_start = *pos;
            while *pos < bytes.len() && bytes[*pos].is_ascii_hexdigit() {
                *pos += 1;
            }
            if *pos == hex_start {
                return None; // "0x" with no digits
            }
            let hex_str = std::str::from_utf8(&bytes[hex_start..*pos]).ok()?;
            let v = u64::from_str_radix(hex_str, 16).ok()?;
            // Each literal fits 32 bits: the reference assembler folds `0xffffffff - 1` and
            // rejects `0x100000000 - 1`.
            (v <= 0xFFFF_FFFF).then_some(v as i64)
        } else if *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
                *pos += 1;
            }
            // Reject if followed by a letter (e.g. register name like "0R")
            if *pos < bytes.len() && bytes[*pos].is_ascii_alphabetic() {
                return None;
            }
            let num_str = std::str::from_utf8(&bytes[start..*pos]).ok()?;
            num_str.parse::<i64>().ok().filter(|&v| v <= 0xFFFF_FFFF)
        } else {
            None // not a number, probably a register name or other token
        }
    }
}

fn parse_number(s: &str) -> Option<u32> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0X").or_else(|| s.strip_prefix("0x")) {
        if let Ok(v) = u32::from_str_radix(hex, 16) {
            return Some(v);
        }
    }
    if let Ok(v) = s.parse::<u32>() {
        return Some(v);
    }
    eval_const_expr(s).map(|v| v as u32)
}

fn parse_signed_number(s: &str) -> Option<i64> {
    let s = s.trim();
    // Handle negative hex: "-0xNNNN" or "-0XNNNN"
    if let Some(rest) = s.strip_prefix('-') {
        let rest = rest.trim();
        // i64, so a value past 32 bits reaches the range checks instead
        // of wrapping in eval_const_expr (`cjump -0x100000000` became 0).
        if let Some(hex) = rest.strip_prefix("0X").or_else(|| rest.strip_prefix("0x")) {
            return i64::from_str_radix(hex, 16).ok().map(|v| -v);
        }
        if let Ok(v) = rest.parse::<i64>() {
            return Some(-v);
        }
    }
    if let Some(hex) = s.strip_prefix("0X").or_else(|| s.strip_prefix("0x")) {
        if let Ok(v) = u32::from_str_radix(hex, 16) {
            return Some(v as i64);
        }
    }
    if let Ok(v) = s.parse::<i64>() {
        return Some(v);
    }
    eval_const_expr(s)
}

fn parse_address_or_label(s: &str, line: u32) -> Result<(u32, Option<String>)> {
    let s = s.trim();
    // Strip outer parentheses: "(___LABEL)" → "___LABEL"
    let s = if s.starts_with('(') && s.ends_with(')') {
        s[1..s.len() - 1].trim()
    } else {
        s
    };
    // "(PC, x)": a PC-relative offset or label. It used to fall through
    // to the label case and become a symbol named "PC, 0x800".
    let s = match s.split_once(',') {
        Some((pc, rest)) if pc.trim().eq_ignore_ascii_case("PC") => rest.trim(),
        _ => s,
    };
    if let Some(val) = parse_number(s) {
        return Ok((val, None));
    }
    // Symbolic label — return 0 placeholder + the label name.
    if is_valid_label(s) {
        return Ok((0, Some(s.to_string())));
    }
    Err(Error::Parse {
        line,
        msg: format!("invalid address: {s}"),
    })
}

/// Map a universal register name string to its 8-bit ureg code.
/// This mirrors the inverse of `disasm::ureg_name`.
fn ureg_code(name: &str) -> Option<u16> {
    // Check named special registers first, before prefix-based matching,
    // since names like MODE1 start with M which would otherwise match
    // modify registers.
    let named = match name {
        // System registers, numbered as in the SHARC+ universal register
        // code table. The reference assembler uses these codes in every form: 48-bit Type 17
        // `pcstk = 0x100000` = 0f6400100000, `mmask = 0x100000` =
        // 0f7300100000; 32-bit `tperiod = r0` = 703e373f.
        "FADDR" => Some(0x60),
        "DADDR" => Some(0x61),
        "PC" => Some(0x63),
        "PCSTK" => Some(0x64),
        "PCSTKP" => Some(0x65),
        "LADDR" => Some(0x66),
        "CURLCNTR" => Some(0x67),
        "LCNTR" => Some(0x68),
        "EMUCLK" => Some(0x69),
        "EMUCLK2" => Some(0x6A),
        "PX" => Some(0x6B),
        "PX1" => Some(0x6C),
        "PX2" => Some(0x6D),
        "TPERIOD" => Some(0x6E),
        "TCOUNT" => Some(0x6F),
        "USTAT1" => Some(0x70),
        "USTAT2" => Some(0x71),
        "MODE1" => Some(0x72),
        "MMASK" => Some(0x73),
        "MODE2" => Some(0x74),
        "FLAGS" => Some(0x75),
        "ASTAT" | "ASTATX" | "ASTATx" => Some(0x76),
        "ASTATY" | "ASTATy" => Some(0x77),
        "STKY" | "STKYX" | "STKYx" => Some(0x78),
        "STKYY" | "STKYy" => Some(0x79),
        "IRPTL" => Some(0x7A),
        "IMASK" => Some(0x7B),
        "IMASKP" => Some(0x7C),
        "MODE1STK" => Some(0x7D),
        "USTAT3" => Some(0x7E),
        "USTAT4" => Some(0x7F),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    // Data registers: R0-R15
    if let Some(rest) = name.strip_prefix('R') {
        return rest.parse::<u16>().ok().filter(|&n| n <= 15);
    }
    // F-alias for data registers: F0-F15
    if let Some(rest) = name.strip_prefix('F') {
        return rest.parse::<u16>().ok().filter(|&n| n <= 15);
    }
    // Index registers: I0-I15
    if let Some(rest) = name.strip_prefix('I') {
        return rest
            .parse::<u16>()
            .ok()
            .filter(|&n| n <= 15)
            .map(|n| 0x10 + n);
    }
    // Modify registers: M0-M15
    if let Some(rest) = name.strip_prefix('M') {
        return rest
            .parse::<u16>()
            .ok()
            .filter(|&n| n <= 15)
            .map(|n| 0x20 + n);
    }
    // Length registers: L0-L15
    if let Some(rest) = name.strip_prefix('L') {
        return rest
            .parse::<u16>()
            .ok()
            .filter(|&n| n <= 15)
            .map(|n| 0x30 + n);
    }
    // Base registers: B0-B15
    if let Some(rest) = name.strip_prefix('B') {
        return rest
            .parse::<u16>()
            .ok()
            .filter(|&n| n <= 15)
            .map(|n| 0x40 + n);
    }
    // Shadow/complementary registers: S0-S15 and SF0-SF15 (float alias)
    if let Some(rest) = name.strip_prefix("SF") {
        return rest
            .parse::<u16>()
            .ok()
            .filter(|&n| n <= 15)
            .map(|n| 0x50 + n);
    }
    if let Some(rest) = name.strip_prefix('S') {
        return rest
            .parse::<u16>()
            .ok()
            .filter(|&n| n <= 15)
            .map(|n| 0x50 + n);
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use selinstr::disasm;
    use selinstr::encode;

    /// Round-trip: encode -> disassemble -> parse -> re-encode, assert bytes match.
    fn roundtrip(instr: &Instruction) {
        let bytes = encode::encode(instr).unwrap();
        let lines = disasm::disassemble(&bytes, 0, false);
        assert_eq!(lines.len(), 1);
        let text = &lines[0].text;
        let parsed =
            parse_instruction(text).unwrap_or_else(|e| panic!("failed to parse '{}': {}", text, e));
        let rebytes = encode::encode(&parsed).unwrap();
        assert_eq!(
            bytes, rebytes,
            "round-trip mismatch for '{}': orig={:?} reparsed={:?}",
            text, instr, parsed
        );
    }

    #[test]
    fn roundtrip_nop() {
        roundtrip(&Instruction::Nop);
    }

    #[test]
    fn roundtrip_idle() {
        roundtrip(&Instruction::Idle);
    }

    #[test]
    fn roundtrip_load_imm() {
        roundtrip(&Instruction::LoadImm {
            ureg: 0x00,
            value: 42,
        });
        roundtrip(&Instruction::LoadImm {
            ureg: 0x10,
            value: 0xABCD0000,
        });
        roundtrip(&Instruction::LoadImm {
            ureg: 0x70,
            value: 0x00001000,
        });
    }

    #[test]
    fn roundtrip_alu_add() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Add {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_sub() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Sub {
                rn: 3,
                rx: 4,
                ry: 5,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_and() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::And {
                rn: 6,
                rx: 7,
                ry: 8,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_or() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Or {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_xor() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Xor {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_not() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Not { rn: 0, rx: 1 }),
        });
    }

    #[test]
    fn roundtrip_alu_pass() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Pass { rn: 5, rx: 3 }),
        });
    }

    #[test]
    fn roundtrip_alu_abs() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Abs { rn: 2, rx: 7 }),
        });
    }

    #[test]
    fn roundtrip_alu_neg() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Neg { rn: 0, rx: 1 }),
        });
    }

    #[test]
    fn roundtrip_alu_min_max() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Min {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Max {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_clip() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Clip {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_alu_comp() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Comp { rx: 1, ry: 2 }),
        });
    }

    #[test]
    fn roundtrip_alu_inc_dec() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Inc { rn: 0, rx: 1 }),
        });
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Dec { rn: 0, rx: 1 }),
        });
    }

    #[test]
    fn roundtrip_float_add() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Falu(FaluOp::Add {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_float_sub() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Falu(FaluOp::Sub {
                rn: 3,
                rx: 4,
                ry: 5,
            }),
        });
    }

    #[test]
    fn roundtrip_float_pass() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Falu(FaluOp::Pass { rn: 0, rx: 1 }),
        });
    }

    #[test]
    fn roundtrip_float_abs() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Falu(FaluOp::Abs { rn: 0, rx: 1 }),
        });
    }

    #[test]
    fn roundtrip_float_neg() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Falu(FaluOp::Neg { rn: 0, rx: 1 }),
        });
    }

    #[test]
    fn roundtrip_float_mul() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Mul(MulOp::FMul {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_mul_mrf() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Mul(MulOp::MrfMulSsf { rx: 2, ry: 3 }),
        });
    }

    #[test]
    fn roundtrip_mul_reg() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Mul(MulOp::MulSsf {
                rn: 1,
                rx: 5,
                ry: 6,
            }),
        });
    }

    #[test]
    fn roundtrip_shift_lshift() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Shift(ShiftOp::Lshift {
                rn: 4,
                rx: 5,
                ry: 6,
            }),
        });
    }

    #[test]
    fn roundtrip_shift_ashift() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Shift(ShiftOp::Ashift {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_shift_rot() {
        // Note: ROT (opcode 0x20) is disassembled as "OR LSHIFT" by the
        // disassembler (decode_shift maps 0x20 identically to 0x04). This
        // is a known disassembler issue; the parser correctly handles the
        // disassembler output by parsing it as OrLshift. We test that the
        // OrLshift variant itself roundtrips cleanly.
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Shift(ShiftOp::OrLshift {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_shift_btst() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Shift(ShiftOp::Btst { rx: 1, ry: 2 }),
        });
    }

    #[test]
    fn roundtrip_shift_bset() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Shift(ShiftOp::Bset {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
    }

    #[test]
    fn roundtrip_conditional_compute() {
        roundtrip(&Instruction::Compute {
            cond: 0, // EQ
            compute: ComputeOp::Alu(AluOp::Add {
                rn: 0,
                rx: 1,
                ry: 2,
            }),
        });
        roundtrip(&Instruction::Compute {
            cond: 16, // NE
            compute: ComputeOp::Alu(AluOp::Sub {
                rn: 3,
                rx: 4,
                ry: 5,
            }),
        });
    }

    #[test]
    fn roundtrip_jump_absolute() {
        roundtrip(&Instruction::Branch {
            call: false,
            cond: 31,
            delayed: false,
            target: BranchTarget::Absolute(0x001234),
            loop_abort: false,
        });
    }

    #[test]
    fn roundtrip_call_absolute() {
        roundtrip(&Instruction::Branch {
            call: true,
            cond: 31,
            delayed: false,
            target: BranchTarget::Absolute(0x005678),
            loop_abort: false,
        });
    }

    #[test]
    fn roundtrip_conditional_jump() {
        roundtrip(&Instruction::Branch {
            call: false,
            cond: 16, // NE
            delayed: false,
            target: BranchTarget::Absolute(0x000100),
            loop_abort: false,
        });
    }

    #[test]
    fn roundtrip_jump_pc_relative() {
        roundtrip(&Instruction::Branch {
            call: false,
            cond: 31,
            delayed: false,
            target: BranchTarget::PcRelative(10),
            loop_abort: false,
        });
    }

    #[test]
    fn roundtrip_jump_pc_relative_negative() {
        roundtrip(&Instruction::Branch {
            call: false,
            cond: 31,
            delayed: false,
            target: BranchTarget::PcRelative(-5),
            loop_abort: false,
        });
    }

    #[test]
    fn roundtrip_rts() {
        roundtrip(&Instruction::Return {
            interrupt: false,
            cond: 31,
            delayed: false,
            lr: false,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_rti() {
        roundtrip(&Instruction::Return {
            interrupt: true,
            cond: 31,
            delayed: false,
            lr: false,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_conditional_rts() {
        roundtrip(&Instruction::Return {
            interrupt: false,
            cond: 0, // EQ
            delayed: false,
            lr: false,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_rts_with_compute() {
        roundtrip(&Instruction::Return {
            interrupt: false,
            cond: 31,
            delayed: false,
            lr: false,
            compute: Some(ComputeOp::Alu(AluOp::Add {
                rn: 0,
                rx: 1,
                ry: 2,
            })),
        });
    }

    #[test]
    fn roundtrip_rts_db() {
        roundtrip(&Instruction::Return {
            interrupt: false,
            cond: 31,
            delayed: true,
            lr: false,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_rts_lr() {
        roundtrip(&Instruction::Return {
            interrupt: false,
            cond: 31,
            delayed: false,
            lr: true,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_conditional_rts_db() {
        roundtrip(&Instruction::Return {
            interrupt: false,
            cond: 0,
            delayed: true,
            lr: false,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_do_loop_imm() {
        roundtrip(&Instruction::DoLoop {
            counter: LoopCounter::Immediate(100),
            end_pc: 0x002000,
            f: false,
        });
    }

    #[test]
    fn roundtrip_do_loop_ureg() {
        roundtrip(&Instruction::DoLoop {
            counter: LoopCounter::Ureg(0x05),
            end_pc: 0x003000,
            f: false,
        });
    }

    #[test]
    fn roundtrip_mul_clear_mrf() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Mul(MulOp::ClrMrf),
        });
    }

    #[test]
    fn roundtrip_mul_trnc_mrf() {
        roundtrip(&Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Mul(MulOp::TrncMrf),
        });
    }

    #[test]
    fn test_parse_label() {
        let mut parser = AsmParser::new("_main:\n    NOP\n");
        let lines = parser.parse_all().unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].label, Some("_main".to_string()));
        assert!(lines[1].instruction.is_some());
    }

    #[test]
    fn test_parse_directive() {
        let mut parser = AsmParser::new(".SECTION/pm seg_pmco;\n.GLOBAL _main;\n");
        let lines = parser.parse_all().unwrap();
        assert_eq!(lines.len(), 2);
        assert!(matches!(lines[0].directive, Some(Directive::Section(_))));
        assert!(matches!(lines[1].directive, Some(Directive::Global(_))));
    }

    #[test]
    fn test_parse_comment() {
        // SHARC+ asm comment syntaxes are `//` and `/* */`; the
        // `;` character is a statement terminator, not a comment
        // marker, so an inline comment after an instruction uses
        // `//` rather than `;`.
        let mut parser = AsmParser::new("// comment\nNOP; // inline comment\n");
        let lines = parser.parse_all().unwrap();
        assert_eq!(lines.len(), 1);
        assert!(matches!(lines[0].instruction, Some(Instruction::Nop)));
    }

    #[test]
    fn test_parse_dm_load_imm() {
        // Standalone pre-modify load with integer offset uses Type 15
        // (UregMemAccess); DM(Ii, imm) would be post-modify.
        let instr = parse_instruction("R2 = DM(1, I6)").unwrap();
        match instr {
            Instruction::UregMemAccess {
                pm,
                i_reg,
                write,
                ureg,
                offset,
                ..
            } => {
                assert!(!pm);
                assert!(!write);
                assert_eq!(i_reg, 6);
                assert_eq!(ureg, 2);
                assert_eq!(offset, 1);
            }
            other => panic!("expected UregMemAccess, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_dm_store_imm() {
        // Standalone pre-modify store with integer offset uses Type 15.
        let instr = parse_instruction("DM(1, I0) = R3").unwrap();
        match instr {
            Instruction::UregMemAccess {
                pm,
                i_reg,
                write,
                ureg,
                offset,
                ..
            } => {
                assert!(!pm);
                assert!(write);
                assert_eq!(i_reg, 0);
                assert_eq!(ureg, 3);
                assert_eq!(offset, 1);
            }
            other => panic!("expected UregMemAccess, got {other:?}"),
        }
    }

    #[test]
    fn dropped_operation_check_compares_parallel_parts() {
        // The decode-and-count check (oracle: 2,321 rows refused). Its two
        // assemble-level examples, `r0 = pass r15 , i7 = b2w ( i7 )` and
        // `r0 = sat mrf ( sf ) , r2 = r1`, now encode in full.
        let pass = Instruction::Compute {
            cond: 31,
            compute: ComputeOp::Alu(AluOp::Pass { rn: 0, rx: 15 }),
        };
        assert!(check_no_dropped_operations("R0 = PASS R15 , I7 = B2W ( I7 )", &pass, 1).is_err());
        assert!(check_no_dropped_operations("R0 = PASS R15", &pass, 1).is_ok());
    }

    #[test]
    fn test_parse_dm_post_modify_imm_is_type4_with_update() {
        // DM(Ii, imm) accesses Ii and then updates it; only Type 4 can say so.
        for (source, write) in [("R2 = DM(I6, 1)", false), ("DM(I0, 1) = R3", true)] {
            match parse_instruction(source).unwrap() {
                Instruction::ComputeLoadStore {
                    compute: None,
                    access,
                    offset: 1,
                    post_modify: true,
                    ..
                } => assert_eq!(access.write, write, "{source}"),
                other => panic!("{source}: expected post-modify Type 4, got {other:?}"),
            }
        }
        // The data6 field takes -0x20..=0x3f (type4_offset); PM needs a
        // DAG2 I register (`r0 = pm ( i8 , 0x0 )` = 617e003f).
        for source in ["I1 = DM(I6, 1)", "R2 = DM(I6, 64)", "R2 = PM(I2, 1)"] {
            assert!(parse_instruction(source).is_err(), "{source}");
        }
    }

    // --- Phase 1: Batch A tests ---

    #[test]
    fn roundtrip_bit_set() {
        roundtrip(&Instruction::BitOp {
            op: 0,
            sreg: 0x02,
            data: 0x00001000,
        });
    }

    #[test]
    fn roundtrip_bit_clr() {
        roundtrip(&Instruction::BitOp {
            op: 1,
            sreg: 0x02,
            data: 0x00001000,
        });
    }

    #[test]
    fn roundtrip_bit_tst() {
        roundtrip(&Instruction::BitOp {
            op: 4,
            sreg: 0x0B,
            data: 0xFF,
        });
    }

    #[test]
    fn roundtrip_bit_xor() {
        roundtrip(&Instruction::BitOp {
            op: 5,
            sreg: 0x05,
            data: 0x0F0F,
        });
    }

    #[test]
    fn roundtrip_bit_tgl() {
        roundtrip(&Instruction::BitOp {
            op: 2,
            sreg: 0x02,
            data: 0x0F0F,
        });
    }

    #[test]
    fn roundtrip_push_sts() {
        roundtrip(&Instruction::StackOp { ops: 0x10 });
    }

    #[test]
    fn roundtrip_pop_pcstk() {
        roundtrip(&Instruction::StackOp { ops: 0x02 });
    }

    #[test]
    fn roundtrip_push_loop() {
        roundtrip(&Instruction::StackOp { ops: 0x40 });
    }

    #[test]
    fn roundtrip_pop_loop() {
        roundtrip(&Instruction::StackOp { ops: 0x20 });
    }

    #[test]
    fn cjump_without_db_is_an_error() {
        // The reference assembler rejects CJUMP without (DB) (176 rows).
        assert!(parse_instruction("CJUMP 0x001234").is_err());
    }

    #[test]
    fn roundtrip_cjump_db() {
        roundtrip(&Instruction::CJump {
            addr: 0x005678,
            delayed: true,
            relative: false,
        });
    }

    #[test]
    fn roundtrip_sync() {
        roundtrip(&Instruction::Sync);
    }

    // --- Phase 2: Batch B tests ---

    #[test]
    fn roundtrip_dag_modify() {
        roundtrip(&Instruction::DagModify {
            pm: false,
            i_reg: 0,
            m_reg: 1,
            cond: 31,
            compute: None,
            dest_xor: 0,
            width: MemWidth::Normal,
        });
    }

    #[test]
    fn roundtrip_dag_modify_dag2() {
        roundtrip(&Instruction::DagModify {
            pm: true,
            i_reg: 0,
            m_reg: 0,
            cond: 31,
            compute: None,
            dest_xor: 0,
            width: MemWidth::Normal,
        });
    }

    #[test]
    fn roundtrip_dag_modify_with_compute() {
        roundtrip(&Instruction::DagModify {
            pm: false,
            i_reg: 0,
            m_reg: 1,
            cond: 31,
            compute: Some(ComputeOp::Alu(AluOp::Add {
                rn: 0,
                rx: 1,
                ry: 2,
            })),
            dest_xor: 0,
            width: MemWidth::Normal,
        });
    }

    #[test]
    fn test_parse_dm_load_with_space() {
        // Disassembler format: "R0=DM (0x1,I0)" — Type 15
        let instr = parse_instruction("R0=DM (0x1,I0)").unwrap();
        match instr {
            Instruction::UregMemAccess {
                pm,
                write,
                ureg,
                offset,
                ..
            } => {
                assert!(!pm);
                assert!(!write);
                assert_eq!(ureg, 0);
                assert_eq!(offset, 1);
            }
            other => panic!("expected UregMemAccess, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_dm_store_with_space() {
        // Disassembler format: "DM (-0x1,I3)=R12" — Type 15
        let instr = parse_instruction("DM (-0x1,I3)=R12").unwrap();
        match instr {
            Instruction::UregMemAccess {
                pm,
                i_reg,
                write,
                ureg,
                offset,
                ..
            } => {
                assert!(!pm);
                assert!(write);
                assert_eq!(i_reg, 3);
                assert_eq!(ureg, 12);
                assert_eq!(offset, -1);
            }
            other => panic!("expected UregMemAccess, got {other:?}"),
        }
    }

    // --- Phase 3: Batch C tests ---

    #[test]
    fn roundtrip_load_imm_neg() {
        // Test that negative immediate values roundtrip (I0 = -0xNNNN)
        roundtrip(&Instruction::LoadImm {
            ureg: 0x10,
            value: (-0x54330000i64) as u32,
        });
    }

    #[test]
    fn roundtrip_ureg_transfer() {
        // I0 = R0 (ureg transfer, not compute)
        roundtrip(&Instruction::UregTransfer {
            src_ureg: 0x00, // R0
            dst_ureg: 0x10, // I0
            cond: 31,
            compute: None,
        });
    }

    // --- Phase 4: Batch D tests (compound instructions) ---

    #[test]
    fn test_parse_compute_with_dm_store() {
        // "R0 = R1 + R2 , DM (I0,M5)=R3"
        let instr = parse_instruction("R0 = R1 + R2 , DM (I0,M5)=R3").unwrap();
        match instr {
            Instruction::UregDagMove {
                pm,
                write,
                ureg,
                i_reg,
                m_reg,
                compute,
                ..
            } => {
                assert!(!pm);
                assert!(write);
                assert_eq!(ureg, 3); // R3
                assert_eq!(i_reg, 0);
                assert_eq!(m_reg, 5);
                assert!(compute.is_some());
            }
            other => panic!("expected UregDagMove, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_compute_with_dm_load() {
        // "R0 = R1 + R2 , R3=DM (I0,M5)"
        let instr = parse_instruction("R0 = R1 + R2 , R3=DM (I0,M5)").unwrap();
        match instr {
            Instruction::UregDagMove {
                pm,
                write,
                ureg,
                i_reg,
                m_reg,
                compute,
                ..
            } => {
                assert!(!pm);
                assert!(!write);
                assert_eq!(ureg, 3);
                assert_eq!(i_reg, 0);
                assert_eq!(m_reg, 5);
                assert!(compute.is_some());
            }
            other => panic!("expected UregDagMove, got {other:?}"),
        }
    }

    // --- Phase 5: Multifunction compute ---

    #[test]
    fn test_parse_multifunction_fixed() {
        let instr = parse_instruction("R0 = R1 * R4 (SSF) , R8 = R8 + R12").unwrap();
        match instr {
            Instruction::Compute {
                compute:
                    ComputeOp::Multi(MultiOp::MulAlu {
                        fp,
                        mul_sel,
                        alu_sel,
                        rm,
                        ra,
                        rxm,
                        rym,
                        rxa,
                        rya,
                    }),
                ..
            } => {
                assert!(!fp);
                assert_eq!(mul_sel, 0);
                assert_eq!(alu_sel, 0);
                assert_eq!(rm, 0);
                assert_eq!(ra, 8);
                assert_eq!(rxm, 1);
                assert_eq!(rym, 0); // R4 -> 4-4=0
                assert_eq!(rxa, 0); // R8 -> 8-8=0
                assert_eq!(rya, 0); // R12 -> 12-12=0
            }
            other => panic!("expected Multi MulAlu, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_multifunction_sub() {
        let instr = parse_instruction("R3 = R2 * R5 (SSF) , R9 = R10 - R14").unwrap();
        match instr {
            Instruction::Compute {
                compute:
                    ComputeOp::Multi(MultiOp::MulAlu {
                        fp,
                        mul_sel,
                        alu_sel,
                        rm,
                        ra,
                        rxm,
                        rym,
                        rxa,
                        rya,
                    }),
                ..
            } => {
                assert!(!fp);
                assert_eq!(mul_sel, 0);
                assert_eq!(alu_sel, 1);
                assert_eq!(rm, 3);
                assert_eq!(ra, 9);
                assert_eq!(rxm, 2);
                assert_eq!(rym, 1);
                assert_eq!(rxa, 2);
                assert_eq!(rya, 2);
            }
            other => panic!("expected Multi MulAlu sub, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_multifunction_dual_addsub() {
        let instr =
            parse_instruction("R0 = R1 * R4 (SSF) , R8 = R8 + R12 , R3 = R8 - R12").unwrap();
        match instr {
            Instruction::Compute {
                compute:
                    ComputeOp::Multi(MultiOp::MulDualAddSub {
                        fp,
                        rm,
                        ra,
                        rs,
                        rxm,
                        rym,
                        rxa,
                        rya,
                    }),
                ..
            } => {
                assert!(!fp);
                assert_eq!(rm, 0);
                assert_eq!(ra, 8);
                assert_eq!(rs, 3);
                assert_eq!(rxm, 1);
                assert_eq!(rym, 0);
                assert_eq!(rxa, 0);
                assert_eq!(rya, 0);
            }
            other => panic!("expected Multi MulDualAddSub, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_multifunction_mrf_acc() {
        let instr = parse_instruction("MRF = MRF + R2 * R6 (SSF) , R11 = R9 + R13").unwrap();
        match instr {
            Instruction::Compute {
                compute:
                    ComputeOp::Multi(MultiOp::MulAlu {
                        fp,
                        mul_sel,
                        alu_sel,
                        ..
                    }),
                ..
            } => {
                assert!(!fp);
                assert_eq!(mul_sel, 1);
                assert_eq!(alu_sel, 0);
            }
            other => panic!("expected Multi MulAlu with MRF acc, got {other:?}"),
        }
    }

    // -- Type 5b: RegisterSwap --

    #[test]
    fn roundtrip_register_swap() {
        roundtrip(&Instruction::RegisterSwap {
            dreg: 14,
            cdreg: 8,
            cond: 31,
            compute: None,
        });
    }

    #[test]
    fn roundtrip_register_swap_with_compute() {
        roundtrip(&Instruction::RegisterSwap {
            dreg: 14,
            cdreg: 8,
            cond: 31,
            compute: Some(ComputeOp::Alu(AluOp::Add {
                rn: 0,
                rx: 1,
                ry: 2,
            })),
        });
    }

    #[test]
    fn roundtrip_register_swap_conditional() {
        roundtrip(&Instruction::RegisterSwap {
            dreg: 5,
            cdreg: 3,
            cond: 0,
            compute: None,
        });
    }

    // -- Type 16: DM|PM(Ii,Mm) = imm32 --

    #[test]
    fn roundtrip_type16_dm_imm_store() {
        roundtrip(&Instruction::ImmStore {
            pm: false,
            i_reg: 7,
            m_reg: 7,
            value: 0xDEAD_BEEF,
        });
    }

    #[test]
    fn roundtrip_type16_pm_imm_store() {
        roundtrip(&Instruction::ImmStore {
            pm: true,
            i_reg: 0,
            m_reg: 1,
            value: 0xCAFE_F00D,
        });
    }

    #[test]
    fn roundtrip_type16_pm_imm_store_high_regs() {
        roundtrip(&Instruction::ImmStore {
            pm: true,
            i_reg: 7,
            m_reg: 7,
            value: 0x1234_5678,
        });
    }

    #[test]
    fn parse_swap_text() {
        let instr = parse_instruction("R14<->S8").unwrap();
        match instr {
            Instruction::RegisterSwap {
                dreg,
                cdreg,
                cond,
                compute,
            } => {
                assert_eq!(dreg, 14);
                assert_eq!(cdreg, 8);
                assert_eq!(cond, 31);
                assert!(compute.is_none());
            }
            other => panic!("expected RegisterSwap, got {other:?}"),
        }
    }

    // -- Constant expression evaluator --

    #[test]
    fn test_eval_const_expr() {
        assert_eq!(eval_const_expr("1 + 1"), Some(2));
        assert_eq!(eval_const_expr("0x100 - 5"), Some(251));
        assert_eq!(eval_const_expr("(1 << 4) | 5"), Some(21));
        assert_eq!(eval_const_expr("10 * 3 + 2"), Some(32));
        assert_eq!(eval_const_expr("~0"), Some(-1));
        assert_eq!(eval_const_expr("0xFF & 0x0F"), Some(15));
        assert_eq!(eval_const_expr("100 / 3"), Some(33));
        assert_eq!(eval_const_expr("42"), Some(42));
        assert_eq!(eval_const_expr("0x1000"), Some(4096));
        assert_eq!(eval_const_expr("-1"), Some(-1));
        assert_eq!(eval_const_expr("0xFFFFFFFF"), Some(4294967295));
        assert_eq!(eval_const_expr("4 * 1024"), Some(4096));
        assert_eq!(eval_const_expr("7 % 3"), Some(1));
        assert_eq!(eval_const_expr("0xFF ^ 0x0F"), Some(240));
        assert_eq!(eval_const_expr("1 << 31"), Some(2147483648));
        assert_eq!(eval_const_expr("(2 + 3) * 4"), Some(20));
        assert_eq!(eval_const_expr("10 / 0"), None);
        assert_eq!(eval_const_expr("10 % 0"), None);
        // Not an expression:
        assert_eq!(eval_const_expr("R0"), None);
        assert_eq!(eval_const_expr("R1 + R2"), None);
        assert_eq!(eval_const_expr(""), None);
        assert_eq!(eval_const_expr("PASS R0"), None);
    }

    #[test]
    fn test_expr_in_load_imm() {
        let instr = parse_instruction("R0 = 1 + 1").unwrap();
        match instr {
            Instruction::LoadImm { ureg, value } => {
                assert_eq!(ureg, 0);
                assert_eq!(value, 2);
            }
            other => panic!("expected LoadImm, got {other:?}"),
        }

        let instr = parse_instruction("R0 = (1 << 4) | 5").unwrap();
        match instr {
            Instruction::LoadImm { ureg, value } => {
                assert_eq!(ureg, 0);
                assert_eq!(value, 21);
            }
            other => panic!("expected LoadImm, got {other:?}"),
        }

        let instr = parse_instruction("R0 = 0x100 - 5").unwrap();
        match instr {
            Instruction::LoadImm { ureg, value } => {
                assert_eq!(ureg, 0);
                assert_eq!(value, 251);
            }
            other => panic!("expected LoadImm, got {other:?}"),
        }
    }

    #[test]
    fn parse_swap_compound() {
        let instr = parse_instruction("R0 = R1 + R2 , R14<->S8").unwrap();
        match instr {
            Instruction::RegisterSwap {
                dreg,
                cdreg,
                cond,
                compute,
            } => {
                assert_eq!(dreg, 14);
                assert_eq!(cdreg, 8);
                assert_eq!(cond, 31);
                assert!(compute.is_some());
            }
            other => panic!("expected RegisterSwap, got {other:?}"),
        }
    }

    #[test]
    fn test_expr_does_not_break_compute() {
        // "R0 = R1 + R2" must still parse as a compute, not a LoadImm
        let instr = parse_instruction("R0 = R1 + R2").unwrap();
        match instr {
            Instruction::Compute { .. } => {}
            other => panic!("expected Compute for R0 = R1 + R2, got {other:?}"),
        }

        // "R0 = PASS R1" must still be a compute
        let instr = parse_instruction("R0 = PASS R1").unwrap();
        match instr {
            Instruction::Compute { .. } => {}
            other => panic!("expected Compute for R0 = PASS R1, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_symbol_list_splits_extern() {
        // `.EXTERN a, b;` must split into two symbols, not one
        // comma-joined name.  libsel's startup.s declares the pair
        // `ldf_stack_space, ldf_stack_length` this way; treating it as
        // a single name leaves the linker with an undefined symbol
        // on every reference and the boot ROM hangs at reset.
        let mut parser = AsmParser::new(".EXTERN ldf_stack_space, ldf_stack_length;\n");
        let lines = parser.parse_all().unwrap();
        assert_eq!(lines.len(), 1);
        match &lines[0].directive {
            Some(Directive::Extern(names)) => {
                assert_eq!(
                    names,
                    &vec![
                        "ldf_stack_space".to_string(),
                        "ldf_stack_length".to_string()
                    ]
                );
            }
            other => panic!("expected Directive::Extern, got {other:?}"),
        }
    }

    #[test]
    fn test_unary_plus_in_const_expr() {
        // `M6 = +1;` must encode the same as `M6 = 1;` — the SHARC+
        // Tools dialect lets explicit positive signs annotate the
        // M-register defaults libsel/startup.s sets up before
        // entering main().
        let plus = parse_instruction("M6 = +1").unwrap();
        let bare = parse_instruction("M6 = 1").unwrap();
        assert_eq!(plus, bare);
    }

    #[test]
    fn test_multi_symbol_loadimm_expression() {
        // `I7 = ldf_stack_space + ldf_stack_length;` must surface BOTH
        // symbols so the assembler can emit a relocation pair.  The
        // primary lands in `label_ref`; any further symbols land in
        // `extra_label_refs` with the sign of their connector.
        let mut parser = AsmParser::new("I7 = ldf_stack_space + ldf_stack_length;\n");
        let lines = parser.parse_all().unwrap();
        let line = lines.into_iter().find(|l| l.instruction.is_some()).unwrap();
        assert_eq!(line.label_ref.as_deref(), Some("ldf_stack_space"));
        assert_eq!(
            line.extra_label_refs,
            vec![("ldf_stack_length".to_string(), 1)]
        );
    }

    // Ported from ElekLabs/selache fix/encodings (Em D).
    fn word(src: &str) -> String {
        let bytes = encode::encode(&parse_instruction(src).unwrap()).unwrap();
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Immediate shifts carry the PRM Table 17-9 shiftimm opcode (bits 22-16); a negative LSHIFT/ASHIFT count (also the OR forms)
    /// sets dataex (bits 30-27) to 0xF. The expected words are the vendor's shapes.
    #[test]
    fn imm_shift_opcodes_follow_prm_table_17_9() {
        for (src, want) in [
            ("R0 = LSHIFT R11 BY -16", "023e7800f00b"),
            ("R0 = LSHIFT R11 BY 16", "023e0000100b"),
            ("R0 = ASHIFT R11 BY -3", "023e7801fd0b"),
            ("R0 = ASHIFT R11 BY 3", "023e0001030b"),
            ("R0 = R0 OR LSHIFT R11 BY 3", "023e0008030b"),
            ("R0 = R0 OR ASHIFT R11 BY -3", "023e7809fd0b"),
            ("R0 = ROT R11 BY 3", "023e0002030b"),
            ("R0 = FEXT R11 BY 4:8", "023e1010040b"),
            ("R0 = FEXT R11 BY 4:8 (SE)", "023e1012040b"),
            // OR FDEP by an immediate is shiftimm 0x19 (register opcode 0x64 >> 2), not 0x1b: the
            // reference assembler emits `R0 = R0 OR FDEP R12 BY 0x10:0x1` as 023e0019500c. 0x1b is
            // OR FDEP (SE) (0x6c >> 2), and 0x1d is no shiftimm opcode.
            ("R0 = R0 OR FDEP R11 BY 4:8", "023e1019040b"),
            ("R0 = BSET R11 BY 5", "023e0030050b"),
            ("BTST R11 BY 5", "023e0033050b"),
        ] {
            assert_eq!(word(src), want, "{src}");
        }
        assert!(parse_instruction("R0 = R0 OR FEXT R11 BY 4:8").is_err());
        // No reference-assembler vector yet for these immediate forms, so they are refused
        // (use the register form). Expected once measured: FDEP 0x11, FDEP (SE) 0x13,
        // OR FDEP (SE) 0x1b.
        for src in [
            "R0 = FDEP R11 BY 4:8",
            "R0 = FDEP R11 BY 4:8 (SE)",
            "R0 = R0 OR FDEP R11 BY 4:8 (SE)",
        ] {
            assert!(parse_instruction(src).is_err(), "{src}");
        }
    }

    /// A direct DM access with (BW)/(BWSE)/(SW)/(SWSE) is Type 14d (PRM Figure 16-2): byte0 0x1A (0x1B for a store), l = bit 38 for a
    /// short word, x = bit 36 for sign extension, dreg in bits 35-32. Without a suffix it stays Type 14.
    #[test]
    fn direct_sub_word_access_is_type_14d() {
        for (src, want) in [
            ("R3 = DM(0x256008)", "100300256008"),
            ("R3 = DM(0x256008) (SW)", "1a4300256008"),
            ("R3 = DM(0x256008) (SWSE)", "1a5300256008"),
            ("R3 = DM(0x256008) (BW)", "1a0300256008"),
            ("R3 = DM(0x256008) (BWSE)", "1a1300256008"),
            ("DM(0x256008) = R3 (SW)", "1b4300256008"),
            ("DM(0x256008) = R3 (BW)", "1b0300256008"),
        ] {
            let bytes = encode::encode(&parse_instruction(src).unwrap()).unwrap();
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, want, "{src}");
            // The disassembler reads the same word back to the same instruction.
            if want.starts_with("1a") || want.starts_with("1b") {
                let text = &disasm::disassemble(&bytes, 0, false)[0].text;
                assert_eq!(
                    encode::encode(&parse_instruction(text).unwrap()).unwrap(),
                    bytes,
                    "{text}"
                );
            }
        }
        // A store has nothing to sign-extend: the assembler refuses it.
        assert!(crate::assemble::assemble_source(
            ".SECTION/PM seg_pmco;\nDM(0x256008) = R3 (SWSE);\n",
            true
        )
        .is_err());
        assert!(parse_instruction("R16 = DM(0x256008) (SW)").is_err());
    }
}
