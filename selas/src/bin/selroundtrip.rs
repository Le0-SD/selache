// SPDX-License-Identifier: GPL-3.0
//! Byte-exact VISA disassembly/reassembly check. No input-derived files are written.

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use selelf::elf;
use selinstr::visa::VisaLine;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ParcelEndian {
    #[default]
    Big,
    Little,
}

impl ParcelEndian {
    /// Convert between input storage order and the decoder's big-endian parcels.
    fn convert(self, bytes: &mut [u8]) {
        if self == Self::Little {
            for parcel in bytes.chunks_exact_mut(2) {
                parcel.swap(0, 1);
            }
        }
    }
}

#[derive(Default, Debug)]
struct Counts {
    total: usize,
    mismatches: usize,
    errors: usize,
}

impl Counts {
    fn record(&mut self, matches: bool, error: bool) {
        self.total += 1;
        self.mismatches += usize::from(!matches);
        self.errors += usize::from(error);
    }
}

#[derive(Default, Debug)]
struct Report {
    endian: ParcelEndian,
    preserve_48bit: bool,
    all: Counts,
    pre_modify: Counts,
    short_word: Counts,
    type3b_pre_modify: Counts,
    type3b_short_word: Counts,
    mnemonics: BTreeMap<String, Counts>,
    first: Option<String>,
    input_bytes: usize,
    assembled_bytes: usize,
}

/// Assemble the listing one instruction at a time so a shorter encoding or a
/// parser failure cannot turn every subsequent instruction into a mismatch.
/// This exercises selas's real parser, encoder, VISA selection and ELF writer.
fn reassemble(text: &str, force_48bit: bool) -> Result<Vec<u8>, String> {
    let compression = if force_48bit { ".NOCOMPRESS;\n" } else { "" };
    let source = format!(".section/SW/DOUBLE32 roundtrip;\n{compression}{text};\n");
    // Reject unresolved symbols and diagnostic text that parses as no instruction.
    let parsed = selas::asmparse::AsmParser::new(&source)
        .parse_all()
        .map_err(|e| e.to_string())?;
    let mut instructions = 0;
    for line in parsed {
        if line.label_ref.is_some() || !line.extra_label_refs.is_empty() {
            return Err("disassembly requires an unresolved symbol".into());
        }
        if line.instruction.is_some() {
            instructions += 1;
        }
    }
    if instructions != 1 {
        return Err(format!("expected one instruction, parsed {instructions}"));
    }
    // The disassembly prints raw `(pc, n)` offsets.
    // One instruction, not a section end.
    let options = selas::assemble::AsmOptions {
        raw_pc_offsets: true,
        cond_jump_tail_pad: false,
        ..selas::assemble::AsmOptions::default()
    };
    let object =
        selas::assemble::assemble_source_with(&source, true, options).map_err(|e| e.to_string())?;
    let header = elf::parse_header(&object).map_err(|e| e.to_string())?;
    let section = |index: usize| {
        let start = header.e_shoff as usize + index * header.e_shentsize as usize;
        elf::parse_section_header(&object[start..], header.ei_data)
    };
    let names = section(header.e_shstrndx as usize);
    let names = &object[names.sh_offset as usize..(names.sh_offset + names.sh_size) as usize];
    for index in 0..header.e_shnum as usize {
        let section = section(index);
        if elf::read_string_at(names, section.sh_name) == "roundtrip" {
            return Ok(object
                [section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize]
                .to_vec());
        }
    }
    Err("selas emitted no roundtrip section".into())
}

/// Group spelling variants by replacing register numbers and numeric operands.
/// Keep operators, conditions, address ordering and access modifiers visible.
fn mnemonic(text: &str) -> String {
    if text.starts_with("???") {
        return "<undecoded>".into();
    }
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_alphanumeric() || c == '_' {
            let mut word = String::from(c);
            while chars
                .peek()
                .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
            {
                word.push(chars.next().unwrap());
            }
            if c.is_ascii_digit() {
                result.push('#');
            } else if matches!(c, 'r' | 'f' | 'i' | 'm' | 'b' | 'l' | 's')
                && word.len() > 1
                && word[1..].bytes().all(|c| c.is_ascii_digit())
            {
                result.push(c);
                result.push('N');
            } else {
                result.push_str(&word);
            }
        } else if !c.is_whitespace() {
            result.push(c);
        }
    }
    result
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::new();
    for byte in bytes {
        write!(text, "{byte:02x}").unwrap();
    }
    text
}

fn roundtrip(data: &[u8], endian: ParcelEndian, preserve_48bit: bool) -> Report {
    let mut normalized = data.to_vec();
    endian.convert(&mut normalized);
    // Exactly the decoder/callback used by seldump's VISA mnemonic listing.
    let mut listing =
        selinstr::visa::disassemble_visa(&normalized, 0, selinstr::disasm::decode_instruction);
    let consumed: usize = listing.iter().map(|line| line.raw.len()).sum();
    // disassemble_visa ignores a final unpaired byte; it must still fail this check.
    if consumed < data.len() {
        listing.push(VisaLine {
            addr: (consumed / 2) as u32,
            raw: normalized[consumed..].to_vec(),
            text: "??? (trailing byte)".into(),
        });
    }
    compare_listing(&listing, data.len(), endian, preserve_48bit)
}

fn compare_listing(
    listing: &[VisaLine],
    input_bytes: usize,
    endian: ParcelEndian,
    preserve_48bit: bool,
) -> Report {
    let mut report = Report {
        endian,
        preserve_48bit,
        input_bytes,
        ..Report::default()
    };
    let mut cache = HashMap::new();
    let mut offset = 0;
    for line in listing {
        let mut original = line.raw.clone();
        endian.convert(&mut original);
        // The same mnemonic can occur in both compressed and 48-bit form.
        // Keep their encodings separate when preserving the original width.
        let force_48bit = preserve_48bit && line.raw.len() == 6;
        let encoded = cache
            .entry((line.text.clone(), force_48bit))
            .or_insert_with(|| {
                reassemble(&line.text, force_48bit).map(|mut bytes| {
                    endian.convert(&mut bytes);
                    bytes
                })
            });
        let matches = encoded.as_ref().is_ok_and(|bytes| *bytes == original);
        let error = encoded.is_err();
        if let Ok(bytes) = encoded {
            report.assembled_bytes += bytes.len();
        }
        report.all.record(matches, error);
        let compact: String = line.text.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.contains("dm(m") || compact.contains("pm(m") {
            report.pre_modify.record(matches, error);
        }
        if compact.contains("(sw)") {
            report.short_word.record(matches, error);
        }
        // Stable encoding subsets, independent of changes to decoder spelling.
        // Use canonical big-endian parcels at the decoder's original boundaries.
        if line.raw.len() == 4 {
            let p1 = u16::from_be_bytes([line.raw[0], line.raw[1]]);
            let p2 = u16::from_be_bytes([line.raw[2], line.raw[3]]);
            if p1 >> 13 == 0b010 && matches!(p2 & 0x3f, 0x3f | 0x3c) {
                if p1 & 0x1000 == 0 {
                    report.type3b_pre_modify.record(matches, error);
                }
                if p2 & 0x403f == 0x403c {
                    report.type3b_short_word.record(matches, error);
                }
            }
        }
        if !matches {
            report
                .mnemonics
                .entry(mnemonic(&line.text))
                .or_default()
                .record(false, error);
            if report.first.is_none() {
                let (within, actual) = match encoded {
                    Ok(bytes) => (
                        original
                            .iter()
                            .zip(bytes.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(original.len().min(bytes.len())),
                        hex(bytes),
                    ),
                    Err(error) => (0, format!("assembly error: {error}")),
                };
                report.first = Some(format!(
                    "First mismatch: byte 0x{:x}, instruction at byte 0x{offset:x} (parcel 0x{:x})\n  {}\n  original: {}\n  assembled: {actual}",
                    offset + within, line.addr, line.text, hex(&original)
                ));
            }
        }
        offset += line.raw.len();
    }
    report
}

fn print_report(report: &Report) {
    println!("VISA listing round-trip (per instruction, original boundaries)");
    if report.preserve_48bit {
        println!("48-bit input encodings: preserved with .NOCOMPRESS");
    }
    println!(
        "Input parcel byte order: {} (reported bytes/offsets use input storage order)",
        match report.endian {
            ParcelEndian::Big => "big",
            ParcelEndian::Little => "little",
        }
    );
    println!(
        "Input: {} bytes; assembled: {} bytes{}",
        report.input_bytes,
        report.assembled_bytes,
        if report.all.errors == 0 {
            ""
        } else {
            " (incomplete: assembly errors)"
        }
    );
    for (name, counts) in [
        ("All", &report.all),
        ("Pre-modify", &report.pre_modify),
        ("Short-word (sw)", &report.short_word),
        (
            "Type 3b pre-modify (raw encoding)",
            &report.type3b_pre_modify,
        ),
        (
            "Type 3b short-word (raw encoding)",
            &report.type3b_short_word,
        ),
    ] {
        println!(
            "{name}: {} instructions, {} matches, {} mismatches ({} assembly/decode errors)",
            counts.total,
            counts.total - counts.mismatches,
            counts.mismatches,
            counts.errors
        );
    }
    if let Some(first) = &report.first {
        println!("{first}");
        println!("Mismatches by mnemonic (register numbers and constants normalized):");
        for (name, counts) in &report.mnemonics {
            println!(
                "  {:6} ({:6} errors)  {name}",
                counts.mismatches, counts.errors
            );
        }
    }
}

fn parse_args(
    args: impl IntoIterator<Item = OsString>,
) -> Result<(PathBuf, ParcelEndian, bool), String> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut endian = ParcelEndian::Big;
    let mut preserve_48bit = false;
    while let Some(arg) = args.next() {
        if arg == "--parcel-endian" {
            endian = match args.next().as_deref().and_then(|value| value.to_str()) {
                Some("big") => ParcelEndian::Big,
                Some("little") => ParcelEndian::Little,
                _ => return Err("--parcel-endian requires big or little".into()),
            };
        } else if arg == "--preserve-48bit" {
            preserve_48bit = true;
        } else if arg.to_string_lossy().starts_with('-') {
            return Err(format!("unknown option: {}", arg.to_string_lossy()));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err("expected one blob path".into());
        }
    }
    Ok((path.ok_or("missing blob path")?, endian, preserve_48bit))
}

fn main() -> ExitCode {
    let (path, endian, preserve_48bit) = match parse_args(std::env::args_os().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("selroundtrip: {error}\nUsage: selroundtrip [--parcel-endian big|little] [--preserve-48bit] <blob.bin>\nReads a raw VISA code section; defaults to big-endian parcels; writes no files.\n--preserve-48bit disables compression when reassembling 48-bit input instructions.");
            return ExitCode::from(2);
        }
    };
    let data = match std::fs::read(&path) {
        Ok(data) => data,
        Err(error) => {
            eprintln!(
                "selroundtrip: {}: {error}",
                std::path::Path::new(&path).display()
            );
            return ExitCode::from(2);
        }
    };
    let report = roundtrip(&data, endian, preserve_48bit);
    print_report(&report);
    ExitCode::from(u8::from(report.all.mismatches != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_stream() {
        let report = roundtrip(&[0x00, 0x01, 0x91, 0x90], ParcelEndian::Big, false); // NOP; R0=DM(I0,M6)
        assert_eq!(report.all.total, 2);
        assert_eq!(report.all.mismatches, 0);
        assert_eq!(report.assembled_bytes, 4);
    }

    #[test]
    fn shrinking_instruction_does_not_desynchronize() {
        let listing = vec![
            VisaLine {
                addr: 0,
                raw: vec![0; 6],
                text: "nop".into(),
            },
            VisaLine {
                addr: 3,
                raw: vec![0, 1],
                text: "nop".into(),
            },
        ];
        let report = compare_listing(&listing, 8, ParcelEndian::Big, false);
        assert_eq!(report.all.total, 2);
        assert_eq!(report.all.mismatches, 1);
        assert_eq!(report.assembled_bytes, 4);
        assert!(report.first.unwrap().contains("nop"));
    }

    #[test]
    fn malformed_tail_is_not_silently_skipped() {
        let report = roundtrip(&[0, 1, 0xff], ParcelEndian::Big, false);
        assert_eq!(report.all.total, 2);
        assert_eq!(report.all.mismatches, 1);
        assert_eq!(report.all.errors, 1);
        assert!(report.first.unwrap().contains("byte 0x2"));
        let truncated = roundtrip(&[0x41, 0xbe], ParcelEndian::Big, false);
        assert_eq!(truncated.all.mismatches, 1);
        assert_eq!(truncated.all.errors, 1);
    }

    #[test]
    fn mnemonic_preserves_access_semantics() {
        assert_eq!(mnemonic("r0=dm (m6,i0) (sw)"), "rN=dm(mN,iN)(sw)");
        assert_ne!(mnemonic("r0=dm (m6,i0)"), mnemonic("r0=dm (i0,m6)"));
    }

    #[test]
    fn little_endian_parcels_and_original_mismatch_offset() {
        let exact = roundtrip(&[0x01, 0x00, 0x90, 0x91], ParcelEndian::Little, false);
        assert_eq!(exact.all.total, 2);
        assert_eq!(exact.all.mismatches, 0);
        // Compressed NOP followed by a 48-bit NOP that selas shortens.
        let report = roundtrip(&[1, 0, 0, 0, 0, 0, 0, 0], ParcelEndian::Little, false);
        assert_eq!(report.all.total, 2);
        assert_eq!(report.all.mismatches, 1);
        let first = report.first.unwrap();
        assert!(first.contains("byte 0x2, instruction at byte 0x2"));
        assert!(first.contains("assembled: 0100"));
        // Odd tails remain errors even when complete parcels are byte-swapped.
        assert_eq!(
            roundtrip(&[1, 0, 0xff], ParcelEndian::Little, false)
                .all
                .errors,
            1
        );
    }

    #[test]
    fn parcel_order_is_explicit_and_defaults_to_big() {
        let args = |items: &[&str]| items.iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(
            parse_args(args(&["blob.bin"])).unwrap().1,
            ParcelEndian::Big
        );
        assert_eq!(
            parse_args(args(&["--parcel-endian", "little", "blob.bin"]))
                .unwrap()
                .1,
            ParcelEndian::Little
        );
        assert!(parse_args(args(&["--parcel-endian", "auto", "blob.bin"])).is_err());
        assert!(parse_args(args(&["--parcel-endian"])).is_err());
        assert!(parse_args(args(&["one.bin", "two.bin"])).is_err());
        assert!(!parse_args(args(&["blob.bin"])).unwrap().2);
        assert!(
            parse_args(args(&["--preserve-48bit", "blob.bin"]))
                .unwrap()
                .2
        );
    }

    #[test]
    fn raw_type3b_subsets_do_not_depend_on_reassembly_success() {
        // Vendor vectors: normal pre-modify load and short-word post-modify load.
        // Count membership even on an unfixed encoder that changes their bytes.
        let big = [0x41, 0xbe, 0x00, 0x3f, 0x51, 0xbe, 0x41, 0x3c];
        let mut little = big;
        ParcelEndian::Little.convert(&mut little);
        for (data, endian) in [(&big, ParcelEndian::Big), (&little, ParcelEndian::Little)] {
            let report = roundtrip(data, endian, false);
            assert_eq!(report.type3b_pre_modify.total, 1);
            assert_eq!(report.type3b_short_word.total, 1);
        }
    }

    #[test]
    fn preserving_48bit_keeps_mixed_width_vendor_returns_and_nops() {
        // The reference assembler emits RTS as 0abe in one place and
        // 0a3e00000000 in another. Its NOPs likewise differ
        // in width. Identical text must not share a cached encoding.
        let big = [
            0x0a, 0xbe, 0x0a, 0x3e, 0, 0, 0, 0, 0x0a, 0xbe, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        for endian in [ParcelEndian::Big, ParcelEndian::Little] {
            let mut data = big;
            endian.convert(&mut data);
            let baseline = roundtrip(&data, endian, false);
            assert_eq!(baseline.all.mismatches, 2);
            let kept = roundtrip(&data, endian, true);
            assert_eq!(kept.all.total, 6);
            assert_eq!(kept.all.mismatches, 0);
            assert_eq!(kept.assembled_bytes, data.len());
        }
    }

    #[test]
    fn preserving_48bit_does_not_hide_malformed_or_changed_instructions() {
        // Negative controls use the vendor RTS/NOP pair above, with
        // deliberately inconsistent listing text and a truncated opcode.
        let wrong = [VisaLine {
            addr: 0,
            raw: vec![0x0a, 0x3e, 0, 0, 0, 0],
            text: "nop".into(),
        }];
        let report = compare_listing(&wrong, 6, ParcelEndian::Big, true);
        assert_eq!(report.all.mismatches, 1);
        assert_eq!(report.all.errors, 0);
        for data in [vec![0, 1, 0xff], vec![0x41, 0xbe]] {
            let report = roundtrip(&data, ParcelEndian::Big, true);
            assert_eq!(report.all.mismatches, 1);
            assert_eq!(report.all.errors, 1);
        }
    }
}
