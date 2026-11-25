//! Builds libmaclator_appsandbox.dylib from src/appsandbox.c (see there for why it exists), links maclator against
//! it weakly, and puts a copy beside the binaries cargo writes (the profile directory and its deps/, where the test
//! executables are), since maclator loads it from its own directory (@loader_path).

use std::path::PathBuf;
use std::process::Command;

const LIB: &str = "libmaclator_appsandbox.dylib";

fn main() {
    println!("cargo:rerun-if-changed=src/appsandbox.c");
    println!("cargo:rerun-if-changed=build.rs");
    for var in ["TARGET_CC", "CC", "MACOSX_DEPLOYMENT_TARGET"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default().as_str() {
        "x86_64" => "x86_64",
        "aarch64" => "arm64",
        other => panic!("appsandbox: no clang -arch for target architecture {other:?}"),
    };
    let min = std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "10.12".into());
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let lib = out.join(LIB);
    let cc = std::env::var("TARGET_CC").or_else(|_| std::env::var("CC")).unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .args(["-arch", arch, &format!("-mmacosx-version-min={min}"), "-O2", "-Wall", "-dynamiclib", "-Wl,-adhoc_codesign"])
        .args(["-install_name", &format!("@loader_path/{LIB}"), "-o"])
        .arg(&lib)
        .arg("src/appsandbox.c")
        .status()
        .unwrap_or_else(|e| panic!("appsandbox: cannot run {cc}: {e}"));
    assert!(status.success(), "appsandbox: {cc} failed to build {LIB}: {status}");
    // OUT_DIR is <target dir>/[<triple>/]<profile>/build/<package>-<hash>/out in cargo's default layout; anything
    // else would put the copies where no binary looks, so say so. With a separate build directory the copies land
    // beside the test executables there, not beside the final binary, which is where scripts/install.sh takes the
    // library from: copy it there from OUT_DIR by hand first. Without it maclator runs as before, and says so when
    // that matters.
    let profile_dir = match out.ancestors().nth(3) {
        Some(d) if out.ancestors().nth(2).and_then(|b| b.file_name()).is_some_and(|n| n == "build") => d.to_path_buf(),
        _ => panic!("appsandbox: OUT_DIR {} is not <profile>/build/<package>/out: cannot place {LIB} beside the binaries", out.display()),
    };
    for dir in [profile_dir.clone(), profile_dir.join("deps")] {
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("appsandbox: {}: {e}", dir.display()));
        // A new file renamed over the old one, so a process that has the old one mapped keeps it intact.
        let tmp = dir.join(format!(".{LIB}.{}.tmp", std::process::id()));
        std::fs::copy(&lib, &tmp)
            .and_then(|_| std::fs::rename(&tmp, dir.join(LIB)))
            .unwrap_or_else(|e| panic!("appsandbox: copy to {}: {e}", dir.display()));
    }
    println!("cargo:rustc-link-arg=-Wl,-weak_library,{}", lib.display());
}
