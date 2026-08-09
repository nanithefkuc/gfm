//! Build script: probe for the foreign libraries the differential tests in
//! `tests/foreign.rs` compare against, compile the small C/C++ shims that
//! reach their header-inline setters, and set the gate cfgs. Nothing here is
//! required to build the crate itself; a missing library only skips a test.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gfm_flint)");
    println!("cargo::rustc-check-cfg=cfg(gfm_m4ri)");
    println!("cargo::rustc-check-cfg=cfg(gfm_m4rie)");
    println!("cargo::rustc-check-cfg=cfg(gfm_fflas)");
    println!("cargo::rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo::rerun-if-changed=shim");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    // FLINT is reached through a plain `#[link]` in the test; no shim needed,
    // its functions are real exported symbols.
    if pkg_exists("flint") {
        println!("cargo::rustc-cfg=gfm_flint");
    }

    // M4RI (GF(2)) and M4RIE (GF(2^e)) expose their bit/element setters as
    // `static inline`, so each needs a compiled shim linked to its library.
    if pkg_exists("m4ri")
        && build_shim(
            &out_dir,
            "m4ri_shim",
            "shim/m4ri_shim.c",
            Language::C,
            &["m4ri"],
        )
    {
        println!("cargo::rustc-cfg=gfm_m4ri");
    }
    if pkg_exists("m4rie")
        && build_shim(
            &out_dir,
            "m4rie_shim",
            "shim/m4rie_shim.c",
            Language::C,
            &["m4rie"],
        )
    {
        println!("cargo::rustc-cfg=gfm_m4rie");
    }

    // FFLAS-FFPACK is a C++ template library; its entry point lives in a C++
    // shim linked against its BLAS/Givaro backend.
    if pkg_exists("fflas-ffpack")
        && build_shim(
            &out_dir,
            "fflas_shim",
            "shim/fflas_shim.cpp",
            Language::Cpp,
            &["fflas-ffpack"],
        )
    {
        println!("cargo::rustc-cfg=gfm_fflas");
    }
}

enum Language {
    C,
    Cpp,
}

/// Whether `pkg-config` reports the package present.
fn pkg_exists(pkg: &str) -> bool {
    Command::new("pkg-config")
        .args(["--exists", pkg])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// The whitespace-split output of `pkg-config <flag> <pkgs...>`, or empty on
/// failure.
fn pkg_config(flag: &str, pkgs: &[&str]) -> Vec<String> {
    let out = Command::new("pkg-config").arg(flag).args(pkgs).output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// Compiles `src` into a static archive `lib<name>.a` in `out_dir`, emits the
/// link directives for it and the package's own libraries, and returns
/// whether it succeeded. A compile or archive failure is reported and treated
/// as "library unusable" — the test skips loudly rather than breaking the
/// build.
fn build_shim(out_dir: &Path, name: &str, src: &str, lang: Language, pkgs: &[&str]) -> bool {
    println!("cargo::rerun-if-changed={src}");
    let (compiler_env, default, std_flag) = match lang {
        Language::C => ("CC", "cc", None),
        Language::Cpp => ("CXX", "c++", Some("-std=gnu++17")),
    };
    let compiler = env::var(compiler_env).unwrap_or_else(|_| default.to_owned());
    let obj = out_dir.join(format!("{name}.o"));

    let mut cmd = Command::new(&compiler);
    cmd.args(["-O2", "-fPIC", "-c"]);
    if let Some(flag) = std_flag {
        cmd.arg(flag);
    }
    for cflag in pkg_config("--cflags", pkgs) {
        cmd.arg(cflag);
    }
    cmd.arg(src).arg("-o").arg(&obj);
    match cmd.status() {
        Ok(s) if s.success() => {}
        other => {
            println!("cargo::warning=failed to compile {src}: {other:?}; test skipped");
            return false;
        }
    }

    // The shim object first, then the package's libraries that satisfy its
    // undefined symbols (link order matters for GNU ld). Scope the flags to
    // integration tests and benchmarks, leaving the library and cross builds
    // free of these host-only dependencies.
    for target in ["tests", "benches"] {
        println!("cargo::rustc-link-arg-{target}={}", obj.display());
        for token in pkg_config("--libs", pkgs) {
            println!("cargo::rustc-link-arg-{target}={token}");
        }
        if matches!(lang, Language::Cpp) {
            println!("cargo::rustc-link-arg-{target}=-lstdc++");
        }
    }
    true
}
