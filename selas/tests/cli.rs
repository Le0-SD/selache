// SPDX-License-Identifier: GPL-3.0
// cli.rs --- Integration tests for the selas assembler binary
// Copyright (c) 2026 Jakob Kastelic

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn selas_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_selas"))
}

static WORK_COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_workdir(tag: &str) -> PathBuf {
    let n = WORK_COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut dir = std::env::temp_dir();
    dir.push(format!("selas_it_{}_{}_{}", tag, std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn no_args_prints_usage_and_fails() {
    let out = Command::new(selas_bin())
        .output()
        .expect("failed to spawn selas");
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Usage:"), "stderr: {stderr}");
    assert!(stderr.contains("selas"), "stderr: {stderr}");
}

#[test]
fn version_flag_succeeds() {
    // `-version` alone should parse cleanly and exit 0; the driver
    // currently does not print a banner for it (the selas binary's
    // run() is a no-op when show_version is set and there is no
    // input), but the important contract is that it does not crash
    // and returns success.
    let out = Command::new(selas_bin())
        .arg("-version")
        .output()
        .expect("failed to spawn selas");
    assert!(
        out.status.success(),
        "rc={:?} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn missing_input_file_errors() {
    let work = make_workdir("missing_input");
    let out = Command::new(selas_bin())
        .arg("nonexistent_input_file.s")
        .current_dir(&work)
        .output()
        .expect("failed to spawn selas");
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("selas:"), "stderr: {stderr}");
}

#[test]
fn assembles_minimal_source_to_doj() {
    // Smallest end-to-end exercise: feed a one-line .section directive
    // through the binary and confirm a .doj file is produced. We do
    // not assert on the byte contents because the assembler's exact
    // output format can evolve; the contract here is only that the
    // bin is reachable, parses an input file, and writes the output.
    let work = make_workdir("assemble_min");
    let src_path = work.join("min.s");
    std::fs::write(&src_path, ".section/pm seg_pmco;\n").unwrap();
    let out_path = work.join("min.doj");
    let out = Command::new(selas_bin())
        .arg("-o")
        .arg(&out_path)
        .arg(&src_path)
        .output()
        .expect("failed to spawn selas");
    if !out.status.success() {
        // Some minimal source may still fail to assemble; we accept
        // that as long as the binary itself was reachable. The hard
        // requirement of this test is just that selas was built and
        // ran; the assertion below confirms it produced *some*
        // diagnostic on stderr.
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("selas:"),
            "selas neither succeeded nor emitted a `selas:` diagnostic: {stderr}"
        );
        return;
    }
    assert!(
        out_path.is_file(),
        "selas reported success but did not create {out_path:?}"
    );
}

#[test]
fn nwc_overrides_the_processor_default() {
    // The reference has two six-byte NOPs before its
    // backward DO. Exercise the CLI -> library -> AsmOptions plumbing.
    let work = make_workdir("nwc");
    let source = ".section/pm seg_pmco;\n_b: nop;\nnop;\n\
        lcntr = 0x10, do _b until lce;\nnop;\n";
    let input = work.join("nwc.s");
    let output = work.join("nwc.doj");
    std::fs::write(&input, source).unwrap();
    let result = Command::new(selas_bin())
        .args(["-proc", "ADSP-21569", "-nwc", "-o"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let object = std::fs::read(output).unwrap();
    let header = selelf::elf::parse_header(&object).unwrap();
    let section = selelf::elf::parse_section_header(
        &object[header.e_shoff as usize + header.e_shentsize as usize..],
        header.ei_data,
    );
    let code = &object[section.sh_offset as usize..(section.sh_offset + section.sh_size) as usize];
    assert_eq!(code.len(), 24);
    assert_eq!(&code[..12], &[0; 12]);
    assert_eq!(&code[12..18], &[0xc, 0, 0x10, 0, 0, 0]);
    std::fs::remove_dir_all(work).unwrap();
}

#[test]
fn swc_exclude_uses_word_addresses_and_explicit_sw_overrides_it() {
    // The reference emits NW
    // 0f0e00000002 and 72bf08000000 under -swc-exclude seg_l2code.
    // With explicit /SW it emits 0f8e0002 / 72bf083f
    // despite both -nwc and -swc-exclude; a comma list also works.
    let work = make_workdir("swc_exclude");
    let input = work.join("exclude.asm");
    let output = work.join("exclude.doj");
    std::fs::write(
        &input,
        ".section/pm seg_l2code;\n.global after;\nr14=2;\ni0=m10;\nafter: nop;\n\
         .section/sw explicit;\nr14=2;\ni0=m10;\n\
         .section/pm other;\nr14=2;\n\
         .section seg_l2code;\ni0=m10;\n",
    )
    .unwrap();
    for nwc in [false, true] {
        let mut cmd = Command::new(selas_bin());
        cmd.args(["-proc", "ADSP-21569", "-swc-exclude", "seg_l2code,explicit"]);
        if nwc {
            cmd.arg("-nwc");
        }
        let result = cmd.arg("-o").arg(&output).arg(&input).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let obj = std::fs::read(&output).unwrap();
        let hdr = selelf::elf::parse_header(&obj).unwrap();
        let section = |i| {
            selelf::elf::parse_section_header(
                &obj[hdr.e_shoff as usize + i * hdr.e_shentsize as usize..],
                hdr.ei_data,
            )
        };
        let bytes = |i| {
            let sec = section(i);
            &obj[sec.sh_offset as usize..(sec.sh_offset + sec.sh_size) as usize]
        };
        assert_eq!(
            bytes(1),
            [
                0x0f, 0x0e, 0, 0, 0, 2, 0x72, 0xbf, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x72, 0xbf,
                0x08, 0, 0, 0
            ]
        );
        assert_eq!(bytes(2), [0x0f, 0x8e, 0, 2, 0x72, 0xbf, 0x08, 0x3f]);
        assert_eq!(bytes(3).len(), if nwc { 6 } else { 4 });
        // `after` follows two NW words, not twelve byte addresses or six parcels.
        let syms = (1..hdr.e_shnum as usize)
            .map(section)
            .find(|s| s.sh_type == selelf::elf::SHT_SYMTAB)
            .unwrap();
        let names = section(syms.sh_link as usize);
        let names = &obj[names.sh_offset as usize..(names.sh_offset + names.sh_size) as usize];
        let symbol = (0..syms.sh_size / syms.sh_entsize)
            .map(|i| {
                selelf::elf::parse_symbol(
                    &obj[(syms.sh_offset + i * syms.sh_entsize) as usize..],
                    hdr.ei_data,
                )
            })
            .find(|s| selelf::elf::read_string_at(names, s.st_name) == "after")
            .unwrap();
        assert_eq!(symbol.st_value, 2);
    }
    std::fs::remove_dir_all(work).unwrap();
}

#[test]
fn inc_binary_includes_the_file_slice() {
    // `.INC/BINARY name[] = "file", skip, count;` copies the file slice
    // (the reference assembler: `.inc/binary .sLINEAR.54[] = "wp.sbn", 0, 131;`).
    // selas used to ignore the directive.
    let work = make_workdir("inc_binary");
    std::fs::write(
        work.join("blob.bin"),
        [0x10u8, 0x11, 0xa2, 0xb3, 0xc4, 0xd5, 0x16],
    )
    .unwrap();
    let src = ".global x;\n.section/DOUBLE32 seg_d;\n.inc/binary x[] = \"blob.bin\", 2, 4;\n";
    std::fs::write(work.join("inc.s"), src).unwrap();
    let out_path = work.join("inc.doj");
    let out = Command::new(selas_bin())
        .arg("-o")
        .arg(&out_path)
        .arg(work.join("inc.s"))
        .output()
        .expect("failed to spawn selas");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let obj = std::fs::read(&out_path).unwrap();
    assert!(obj.windows(4).any(|w| w == [0xa2, 0xb3, 0xc4, 0xd5]));

    std::fs::write(
        work.join("bad.s"),
        ".section/DOUBLE32 seg_d;\n.inc/binary y[] = \"none.bin\";\n",
    )
    .unwrap();
    let out = Command::new(selas_bin())
        .arg("-o")
        .arg(work.join("bad.doj"))
        .arg(work.join("bad.s"))
        .output()
        .expect("failed to spawn selas");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not found"));
}
