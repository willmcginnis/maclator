//! Guest path redirection.
//!
//! On an Intel Mac the arm64 dyld shared cache is not installed; the user
//! supplies it (e.g. extracted from an Apple Silicon IPSW of the same macOS
//! version) and points Maclator at it with `--sysroot`. dyld's lookups in the
//! standard cache directories are redirected there.

use crate::forksync::FMutex;
use std::path::{Component, Path, PathBuf};

pub(crate) static CACHE_DIR: FMutex<Option<String>> = FMutex::new(None);
pub(crate) static OVERLAY_ROOT: FMutex<Option<PathBuf>> = FMutex::new(None);

// Keep experimental Xcode components local to the emulated process. These
// prefixes contain the unmodified payloads of Xcode's prerequisite packages.
const OVERLAY_PREFIXES: &[&str] = &[
    "/Library/Developer/PrivateFrameworks",
    "/Library/Developer/DeviceKit",
    "/Library/Developer/CoreSimulator",
    "/Library/Developer/CoreDevice",
    "/Library/Apple",
    "/System/Library/CoreServices/CoreTypes.bundle",
];

const CACHE_PREFIXES: &[&str] = &[
    "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld",
    "/System/Cryptexes/OS/System/Library/dyld",
    "/System/Library/dyld",
];

pub fn set_cache_dir(d: &str) {
    let c = canonical_dir(d);
    note_cache_dir_id(&c);
    *CACHE_DIR.lock().unwrap() = Some(c);
    REWRITING.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Whether a cache dir or an overlay root is set (rewriting). Set only by the two setters, which main calls
/// before the guest starts; an atomic, so the per-syscall check takes no lock.
static REWRITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The sysroot's (device, inode), taken when the sysroot is set,
/// at startup, while its path can still be searched. A working directory inside the sysroot is held by the host
/// whatever happens to the sysroot's ancestors later (a chmod 0 of one makes stat of the sysroot's path fail while
/// relative names from the held directory still walk), so the escape check compares against this, not against a
/// stat of the path at the time of the call. Atomics, not a lock: no fork-safety registration needed.
static CACHE_DEV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CACHE_INO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CACHE_ID_SET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn note_cache_dir_id(c: &str) {
    use std::os::unix::fs::MetadataExt;
    use std::sync::atomic::Ordering::SeqCst;
    match std::fs::metadata(c) {
        Ok(m) if m.is_dir() => {
            CACHE_DEV.store(m.dev(), SeqCst);
            CACHE_INO.store(m.ino(), SeqCst);
            CACHE_ID_SET.store(true, SeqCst);
        }
        _ => CACHE_ID_SET.store(false, SeqCst),
    }
}

/// The sysroot's (device, inode) as taken when it was set; None when it could not be read then.
pub fn cache_dir_id() -> Option<(u64, u64)> {
    use std::sync::atomic::Ordering::SeqCst;
    CACHE_ID_SET.load(SeqCst).then(|| (CACHE_DEV.load(SeqCst), CACHE_INO.load(SeqCst)))
}

/// Directory `d` (--sysroot, or one discover_sysroot found) by its canonical path,
/// resolved once, here. The redirect builds host paths by string from this, so a sysroot spelled
/// through a symlink (/var and /tmp are symlinks on macOS) put that symlink into every rewritten
/// path, and a NOFOLLOW_ANY call through one gave ELOOP where XNU succeeds (rows Y5, Y6, K2).
/// The overlay root is canonicalized already (set_overlay_root). When `d` cannot be
/// resolved, it is kept as given, trimmed, as before.
fn canonical_dir(d: &str) -> String {
    let d = d.trim_end_matches('/');
    std::fs::canonicalize(d).ok().and_then(|p| p.to_str().map(str::to_string)).unwrap_or_else(|| d.to_string())
}

pub fn set_overlay_root(d: &str) -> std::io::Result<()> {
    let root = std::fs::canonicalize(d)?;
    if !root.is_dir() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "overlay root is not a directory"));
    }
    *OVERLAY_ROOT.lock().unwrap() = Some(root);
    REWRITING.store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(())
}

pub fn overlay_root() -> Option<PathBuf> {
    OVERLAY_ROOT.lock().unwrap().clone()
}

fn normalized_absolute(path: &str) -> Option<PathBuf> {
    if !path.starts_with('/') {
        return None;
    }
    let mut result = PathBuf::from("/");
    for c in Path::new(path).components() {
        match c {
            Component::RootDir | Component::CurDir => {},
            Component::Normal(part) => result.push(part),
            Component::ParentDir => { if !result.pop() { return None; } },
            Component::Prefix(_) => return None,
        }
    }
    Some(result)
}

fn component_path(path: &Path) -> bool {
    OVERLAY_PREFIXES.iter().any(|p| path.starts_with(p))
}

fn overlay_path(root: &Path, path: &str) -> Option<PathBuf> {
    let normalized = normalized_absolute(path)?;
    if !component_path(&normalized) {
        return None;
    }
    let candidate = root.join(normalized.strip_prefix("/").ok()?);
    // Do not follow the final symlink here: readlink/stat must still be routed
    // when the original payload contains a symlink to another guest path.
    std::fs::symlink_metadata(&candidate).ok()?;
    Some(candidate)
}

/// Translate a physical overlay path returned by the host into its guest name.
pub fn guest_path(path: &str) -> Option<String> {
    let root = overlay_root()?;
    let relative = Path::new(path).strip_prefix(root).ok()?;
    let guest = Path::new("/").join(relative);
    component_path(&guest).then(|| guest.to_string_lossy().into_owned())
}

/// If `path` should be redirected, return the replacement.
pub fn redirect(path: &str) -> Option<String> {
    if let Some(dir) = CACHE_DIR.lock().unwrap().clone() {
        for p in CACHE_PREFIXES {
            if let Some(rest) = path.strip_prefix(p) {
                if rest.is_empty() || rest.starts_with('/') {
                    return Some(format!("{dir}{rest}"));
                }
            }
        }
    }
    if let Some(root) = overlay_root() {
        if let Some(candidate) = overlay_path(&root, path) {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

/// The cache prefix guest path `g` is under, as the physical spelling XNU keeps a working directory
/// by: the one spelled through the /System/Cryptexes/OS link (to ../../System/Volumes/Preboot/Cryptexes/OS, measured
/// on macOS 27.0.1, arm64: /bin/pwd -P there prints the Preboot path) is the Preboot one it names. /System/Library/dyld keeps
/// its own spelling: natively it is another directory, which the redirect merges with the cryptex one. None when `g`
/// is under no cache prefix.
pub fn physical_prefix(g: &str) -> Option<&'static str> {
    let p = CACHE_PREFIXES.iter().find(|p| g.strip_prefix(**p).is_some_and(|r| r.is_empty() || r.starts_with('/')))?;
    Some(if *p == CACHE_PREFIXES[1] { CACHE_PREFIXES[0] } else { p })
}

pub fn cache_dir() -> Option<String> {
    CACHE_DIR.lock().unwrap().clone()
}

/// Whether any guest path can be redirected: a cache dir or an overlay root is set. With neither,
/// the guest's paths are the host's and path-taking calls go through untouched.
pub fn rewriting() -> bool {
    REWRITING.load(std::sync::atomic::Ordering::SeqCst)
}

/// Whether `--sysroot DIR` can be used. The sysroot stands in for the cache directory, not for a
/// root filesystem: `/` would send `<prefix>/dyld_shared_cache_arm64e` to `/dyld_shared_cache_arm64e`.
pub fn check_sysroot(d: &str) -> Result<(), String> {
    let c = canonical_dir(d);
    if c.is_empty() || c == "/" {
        return Err(format!(
            "--sysroot {d:?} is not supported: the sysroot names the directory holding dyld_shared_cache_arm64e, \
             not a root filesystem; to use the host's own cache, leave it out or pass --no-sysroot"
        ));
    }
    Ok(())
}

/// Whether to look for a sysroot (discover_sysroot): not with one set, not with `--no-sysroot`, and
/// not when the host already has the arm64e cache (host_has_cache): then `$MACLATOR_SYSROOT` is not
/// read either; pass `--sysroot` to use another cache.
pub fn discovery_needed(set: bool, no_sysroot: bool, host_cache: bool) -> bool {
    !set && !no_sysroot && !host_cache
}

/// Whether the host has the arm64e cache under one of the standard directories: on macOS 26 and
/// later it sits in the OS cryptex, not in /System/Library/dyld.
pub fn host_has_cache() -> bool {
    cache_in(CACHE_PREFIXES)
}

fn cache_in(dirs: &[&str]) -> bool {
    dirs.iter().any(|d| Path::new(d).join("dyld_shared_cache_arm64e").is_file())
}

/// Whether guest path `path` is exactly one of the cache prefixes while a cache dir is set: the
/// guest directory that IS the sysroot (the path rewrite treats it as a directory however the host
/// reaches the sysroot).
pub fn is_cache_root(path: &str) -> bool {
    cache_root_for(path, CACHE_DIR.lock().unwrap().is_some())
}

fn cache_root_for(path: &str, set: bool) -> bool {
    set && CACHE_PREFIXES.contains(&path)
}

/// Find a directory holding the arm64 dyld shared cache when `--sysroot` was not given:
/// `$MACLATOR_SYSROOT`, then `~/.maclator/sysroot` (a directory, or a file containing one),
/// then any `~/maclator-sysroot/*` subdirectory that has a `dyld_shared_cache_arm64e`.
pub fn discover_sysroot() -> Option<String> {
    let has_cache = |d: &std::path::Path| d.join("dyld_shared_cache_arm64e").is_file();
    if let Ok(d) = std::env::var("MACLATOR_SYSROOT") {
        if has_cache(std::path::Path::new(&d)) {
            return Some(d);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let cfg = std::path::Path::new(&home).join(".maclator/sysroot");
    if cfg.is_dir() && has_cache(&cfg) {
        return Some(cfg.to_string_lossy().into_owned());
    }
    if let Ok(t) = std::fs::read_to_string(&cfg) {
        let p = std::path::PathBuf::from(t.trim());
        if has_cache(&p) {
            return Some(p.to_string_lossy().into_owned());
        }
    }
    let root = std::path::Path::new(&home).join("maclator-sysroot");
    let mut dirs: Vec<_> = std::fs::read_dir(&root).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| has_cache(p)).collect();
    dirs.sort();
    dirs.pop().map(|p| p.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_routes_only_existing_component_paths() {
        let root = std::env::temp_dir().join(format!("maclator-overlay-test-{}", std::process::id()));
        let dir = root.join("Library/Developer/PrivateFrameworks/Example.framework");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("Info.plist");
        std::fs::write(&file, b"original package data").unwrap();
        assert_eq!(overlay_path(&root, "/Library/Developer/PrivateFrameworks/Example.framework/Info.plist"), Some(file));
        assert!(overlay_path(&root, "/Library/Developer/PrivateFrameworks/Missing.framework").is_none());
        assert!(overlay_path(&root, "/Library/Developer/PrivateFrameworksOther/Info.plist").is_none());
        assert!(overlay_path(&root, "/Library/Developer/PrivateFrameworks/../../Apple").is_none());
        assert_eq!(overlay_path(&root, "/Library/Developer/PrivateFrameworks/Example.framework/../Example.framework/Info.plist"), Some(dir.join("Info.plist")));
        assert!(normalized_absolute("/../../Library").is_none());
        assert!(overlay_path(&root, "Library/Developer/PrivateFrameworks/Example.framework").is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn canonical_dir_resolves_the_sysroot_once() {
        let base = std::env::temp_dir().join(format!("maclator-f-test-{}", std::process::id()));
        let real = base.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let want = std::fs::canonicalize(&real).unwrap().to_string_lossy().into_owned();
        let l = link.to_string_lossy().into_owned();
        assert_eq!(canonical_dir(&l), want);
        assert_eq!(canonical_dir(&format!("{l}/")), want);
        assert_eq!(canonical_dir(&format!("{l}//")), want);
        assert_ne!(canonical_dir(&l), l, "must-fail control: the link's own spelling kept");
        // a path that cannot be resolved is kept as given, trimmed
        let missing = base.join("missing").to_string_lossy().into_owned();
        assert_eq!(canonical_dir(&format!("{missing}/")), missing);
        std::fs::remove_file(&link).unwrap();
        std::fs::remove_dir(&real).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn sysroot_slash_is_refused() {
        for d in ["/", "//", "/.", "/usr/.."] {
            assert!(check_sysroot(d).is_err(), "{d}");
        }
        let t = std::env::temp_dir();
        assert!(check_sysroot(&t.to_string_lossy()).is_ok(), "must-fail control: a real directory");
    }

    #[test]
    fn cache_in_checks_every_directory() {
        let base = std::env::temp_dir().join(format!("maclator-n-test-{}", std::process::id()));
        let (with, without) = (base.join("with"), base.join("without"));
        std::fs::create_dir_all(&with).unwrap();
        std::fs::create_dir_all(&without).unwrap();
        std::fs::write(with.join("dyld_shared_cache_arm64e"), b"").unwrap();
        let (w, wo) = (with.to_string_lossy().into_owned(), without.to_string_lossy().into_owned());
        // the cache in the first directory (the cryptex one), none in the last (/System/Library/dyld)
        assert!(cache_in(&[&w, &wo]), "the cache in the first directory not seen");
        assert!(!cache_in(&[&wo]), "must-fail control: no cache");
        std::fs::remove_file(with.join("dyld_shared_cache_arm64e")).unwrap();
        std::fs::remove_dir(&with).unwrap();
        std::fs::remove_dir(&without).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn no_sysroot_skips_discovery() {
        assert!(discovery_needed(false, false, false), "must-fail control: neither set, no host cache");
        assert!(!discovery_needed(false, true, false), "--no-sysroot still discovers");
        assert!(!discovery_needed(true, false, false), "a sysroot set");
        assert!(!discovery_needed(false, false, true), "the host has the cache");
    }

    #[test]
    fn host_cache_seen_in_the_cryptex() {
        // On macOS 26 and later (measured on 27.0.1) the cache is only in the OS cryptex; skipped elsewhere.
        let c = Path::new(CACHE_PREFIXES[0]).join("dyld_shared_cache_arm64e");
        if !c.is_file() || Path::new(CACHE_PREFIXES[2]).join("dyld_shared_cache_arm64e").is_file() {
            return;
        }
        assert!(host_has_cache(), "the cryptex cache not seen");
    }

    #[test]
    fn cache_root_is_a_prefix_exactly() {
        for p in CACHE_PREFIXES {
            assert!(cache_root_for(p, true), "{p}");
            assert!(!cache_root_for(p, false), "{p}: no cache dir set"); // must-fail control
        }
        for p in ["/System/Library/dyld/", "/System/Library/dyld/x", "/System/Library", "/", ""] {
            assert!(!cache_root_for(p, true), "{p}");
        }
    }
}
