// SPDX-License-Identifier: GPL-3.0
// cli.rs --- Command-line argument parsing
// Copyright (c) 2026 Jakob Kastelic

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct Options {
    pub input: String,
    pub output: String,
    pub verbose: bool,
    pub show_version: bool,
    pub proc: Option<String>,
    pub include_dirs: Vec<String>,
    pub defines: Vec<(String, String)>,
    pub preprocess_only: bool,
    /// The reference assembler `-no-auto-fast-loops`: set a DO loop's fast bit only for
    /// an explicit `(F)`.
    pub no_auto_fast_loops: bool,
    /// The reference assembler `-nwc`: default code to word-addressed 48-bit form.
    pub nwc: bool,
    /// Exact section names excluded from default short-word encoding.
    pub swc_exclude: Vec<String>,
}

pub fn parse_args(args: &[String]) -> Result<Options> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut verbose = false;
    let mut show_version = false;
    let mut proc: Option<String> = None;
    let mut include_dirs: Vec<String> = Vec::new();
    let mut defines: Vec<(String, String)> = Vec::new();
    let mut preprocess_only = false;
    let mut no_auto_fast_loops = false;
    let mut nwc = false;
    let mut swc_exclude = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    return Err(Error::MissingArgument("-o".into()));
                }
                output = Some(args[i].clone());
            }
            "-v" => verbose = true,
            "-version" | "--version" => show_version = true,
            "-proc" => {
                i += 1;
                if i >= args.len() {
                    return Err(Error::MissingArgument("-proc".into()));
                }
                proc = Some(args[i].clone());
            }
            "--preprocess-only" => preprocess_only = true,
            "-no-auto-fast-loops" => no_auto_fast_loops = true,
            "-nwc" => nwc = true,
            "-swc-exclude" => {
                i += 1;
                let value = args
                    .get(i)
                    .filter(|s| !s.starts_with('-'))
                    .ok_or_else(|| Error::MissingArgument("-swc-exclude".into()))?;
                for name in value.split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(Error::MissingArgument("-swc-exclude section name".into()));
                    }
                    swc_exclude.push(name.to_owned());
                }
            }
            arg if arg.starts_with("-I") => {
                let dir = &arg[2..];
                if dir.is_empty() {
                    i += 1;
                    if i >= args.len() {
                        return Err(Error::MissingArgument("-I".into()));
                    }
                    include_dirs.push(args[i].clone());
                } else {
                    include_dirs.push(dir.to_string());
                }
            }
            arg if arg.starts_with("-D") => {
                let def = &arg[2..];
                let def_str = if def.is_empty() {
                    i += 1;
                    if i >= args.len() {
                        return Err(Error::MissingArgument("-D".into()));
                    }
                    args[i].as_str()
                } else {
                    def
                };
                let (name, value) = match def_str.split_once('=') {
                    Some((n, v)) => (n.to_string(), v.to_string()),
                    None => (def_str.to_string(), "1".to_string()),
                };
                defines.push((name, value));
            }
            other if other.starts_with('-') => {
                // Silently ignore unknown flags for toolchain compatibility.
                // Consume the next argument if it looks like a value
                // (i.e. does not start with '-' and is not the last arg).
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    // Peek: if the next arg looks like a file that could be
                    // our input (ends in .s or .asm), don't consume it.
                    let next = &args[i + 1];
                    let looks_like_input =
                        next.ends_with(".s") || next.ends_with(".asm") || next.ends_with(".S");
                    if !looks_like_input {
                        i += 1; // consume the value argument
                    }
                }
            }
            other => {
                if input.is_none() {
                    input = Some(other.to_string());
                } else {
                    return Err(Error::Parse {
                        line: 0,
                        msg: format!("unexpected argument: {other}"),
                    });
                }
            }
        }
        i += 1;
    }

    if show_version && input.is_none() {
        return Ok(Options {
            input: String::new(),
            output: String::new(),
            verbose,
            show_version,
            proc,
            include_dirs,
            defines,
            preprocess_only,
            no_auto_fast_loops,
            nwc,
            swc_exclude,
        });
    }

    let input = input.ok_or(Error::NoInput)?;
    // `--preprocess-only` without `-o` prints to stdout: defaulting to
    // `<stem>.doj` overwrote the object next to the source with text.
    let output = output.unwrap_or_else(|| {
        if preprocess_only {
            return String::new();
        }
        let stem = input
            .strip_suffix(".s")
            .or_else(|| input.strip_suffix(".asm"))
            .unwrap_or(&input);
        format!("{stem}.doj")
    });

    Ok(Options {
        input,
        output,
        verbose,
        show_version,
        proc,
        include_dirs,
        defines,
        preprocess_only,
        no_auto_fast_loops,
        nwc,
        swc_exclude,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_basic() {
        let opts = parse_args(&args(&["test.s"])).unwrap();
        assert_eq!(opts.input, "test.s");
        assert_eq!(opts.output, "test.doj");
    }

    #[test]
    fn test_output_flag() {
        let opts = parse_args(&args(&["-o", "out.doj", "test.s"])).unwrap();
        assert_eq!(opts.input, "test.s");
        assert_eq!(opts.output, "out.doj");
    }

    #[test]
    fn test_version_flag() {
        let opts = parse_args(&args(&["-version"])).unwrap();
        assert!(opts.show_version);
    }

    #[test]
    fn test_no_input_error() {
        let result = parse_args(&args(&[]));
        assert!(result.is_err());
    }

    #[test]
    fn test_proc_flag() {
        let opts = parse_args(&args(&["-proc", "ADSP-21569", "test.s"])).unwrap();
        assert_eq!(opts.proc.as_deref(), Some("ADSP-21569"));
        assert!(!opts.nwc);
        let opts = parse_args(&args(&["-nwc", "-proc", "ADSP-21569", "test.s"])).unwrap();
        assert!(opts.nwc);
        assert_eq!(opts.input, "test.s");
    }

    #[test]
    fn swc_exclude_section_lists() {
        // The reference object retains
        // `-swc-exclude seg_l2code,seg_p_fanout` in .commandLine.
        let opts = parse_args(&args(&[
            "-swc-exclude",
            "seg_l2code,seg_p_fanout",
            "-swc-exclude",
            "extra",
            "test.s",
        ]))
        .unwrap();
        assert_eq!(opts.swc_exclude, ["seg_l2code", "seg_p_fanout", "extra"]);
        assert_eq!(opts.input, "test.s");
        assert!(parse_args(&args(&["test.s", "-swc-exclude"])).is_err());
        assert!(parse_args(&args(&["-swc-exclude", "-nwc", "test.s"])).is_err());
    }

    #[test]
    fn test_include_dirs() {
        let opts = parse_args(&args(&["-Ifoo", "-I", "bar", "test.s"])).unwrap();
        assert_eq!(opts.include_dirs, vec!["foo", "bar"]);
    }

    #[test]
    fn test_defines() {
        let opts = parse_args(&args(&["-DFOO", "-DBAR=42", "test.s"])).unwrap();
        assert_eq!(opts.defines[0], ("FOO".into(), "1".into()));
        assert_eq!(opts.defines[1], ("BAR".into(), "42".into()));
    }

    #[test]
    fn test_preprocess_only() {
        let opts = parse_args(&args(&["--preprocess-only", "test.s"])).unwrap();
        assert!(opts.preprocess_only);
        assert_eq!(opts.output, "", "must print to stdout, not write test.doj");
        assert!(!opts.no_auto_fast_loops);
        let opts = parse_args(&args(&["-no-auto-fast-loops", "test.s"])).unwrap();
        assert!(opts.no_auto_fast_loops);
    }

    #[test]
    fn test_unknown_flags_ignored() {
        let opts = parse_args(&args(&[
            "-si-revision",
            "any",
            "-flags-link",
            "-e",
            "test.s",
        ]))
        .unwrap();
        assert_eq!(opts.input, "test.s");
    }
}
