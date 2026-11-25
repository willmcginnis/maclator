//! Ahead-of-time translation cache.
//!
//! Images that Maclator maps at fixed addresses (the main executable and the
//! arm64 shared cache) get a persistent translation file in
//! `~/Library/Caches/Maclator/aot/`. Each file holds position-independent x86
//! code for a set of guest blocks plus an index.
//!
//! Where the blocks come from:
//!  * static discovery: `LC_FUNCTION_STARTS` + following direct branches
//!    (main executable only; the shared cache is too large to translate whole);
//!  * feedback: every block the JIT translated during a run is recorded in a
//!    profile, and the next AOT build includes it.
//!
//! On exit, if new blocks were seen, a forked child rebuilds the AOT files in
//! the background, so the next launch starts with warm translations.

use crate::jit::emit::{self, StubKind, Unit};
use iced_x86::BlockEncoderOptions;
use std::collections::{BTreeSet, HashSet};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::forksync::FMutex;

const MAGIC: &[u8; 8] = b"MCLAOT04";
/// Bump when the translator's output changes.
const TRANSLATOR_VERSION: u32 = 26;

#[derive(Clone)]
pub(crate) struct Image {
    name: String,
    key: String,
    start: u64,
    end: u64,
    /// Static entry points (function starts), if known.
    entries: Vec<u64>,
    loaded: bool,
    /// Whether a cache for this translator version was successfully loaded.
    cached: bool,
}

pub(crate) static IMAGES: FMutex<Vec<Image>> = FMutex::new(Vec::new());
/// Blocks translated by the JIT this run (per image index).
pub(crate) static SEEN: FMutex<Vec<HashSet<u64>>> = FMutex::new(Vec::new());
static ENABLED: AtomicBool = AtomicBool::new(true);
static SAVED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::SeqCst);
}

/// Whether the AOT cache is in use. It is turned off when the JIT code cache is refused.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

pub(crate) fn cache_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("MACLATOR_CACHE") {
        return d.into();
    }
    match std::env::var("HOME") {
        Ok(home) => std::path::Path::new(&home).join("Library/Caches/Maclator/aot"),
        // No home (a daemon): a directory of our own uid, not a shared one under /tmp that anyone could make first.
        Err(_) => format!("/tmp/maclator-{}", unsafe { libc::geteuid() }).into(),
    }
}

/// Owned by `uid` and writable by no group or other: what the cache trusts. Only the mode bits are read; an
/// access control list granting another user write access is not seen here.
pub(crate) fn owned(md: &std::fs::Metadata, uid: u32) -> bool {
    use std::os::unix::fs::MetadataExt;
    md.uid() == uid && md.mode() & 0o022 == 0
}

/// `dir`, made first if missing (mode 0700), is a real directory (not a symlink) that is `owned` by `uid`.
pub(crate) fn owned_dir(dir: &std::path::Path, uid: u32) -> bool {
    use std::os::unix::fs::DirBuilderExt;
    let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir);
    std::fs::symlink_metadata(dir).is_ok_and(|md| md.is_dir() && owned(&md, uid))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// FNV-1a, enough to name cache files.
fn fnv(parts: &[&[u8]]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The AOT and profile identity of the main executable or dyld: its UUID, its path and the base it is actually
/// loaded at. Translated code embeds guest addresses (the block index's PCs, branch targets, ADR/ADRP results),
/// so a translation is valid only at the base it was made for; a library is keyed the same way
/// (`register_image_at`). An image that slid off its preferred base therefore gets a cache of its own.
fn image_key(kind: &str, uuid: &[u8; 16], path: &str, start: u64) -> String {
    format!("{kind}-{}-{}-{start:x}", hex(uuid), fnv(&[path.as_bytes()]))
}

/// How many entries of one image identity the store keeps. 1 means only the entry for the base in use.
///
/// The store is keyed by identity PLUS the base the image is loaded at (`image_key`, `cache_key`), because
/// translated code embeds absolute guest addresses and is valid only at the base it was made for. An image whose
/// preferred base is declined slides to a fresh base on nearly every exec, so nearly every slid load mints a new key. Nothing retired the
/// old ones, and the store grew without bound: 999 files and 4.03 GiB in one boot, about 70 MiB a minute, until
/// the volume filled. Keeping the base in the key is what keeps a slid load from running another base's code;
/// retiring what that base left behind is what keeps the store finite.
const AOT_KEEP: usize = 1;

/// `key` without its trailing `-<base>`: the part that names the IMAGE rather than the run. Returns `None` when
/// the key carries no base, and such a key is never retired.
///
/// Every keyed family ends in `-{start:x}` and the hex digits carry no `-`, so the last `-` is the split:
/// `main-<uuid>-<fnv>-<base>`, `dyld-<uuid>-<fnv>-<base>`, `dsc-<uuid>-<base>`, `lib-<uuid>-<base>`. An
/// unpatched build's `main-<uuid>-<fnv>` would split into a plausible-looking prefix too -- its fnv is hex as
/// well -- so the prefix must ITSELF still contain a `-`, which is true of every keyed family above and false of
/// the two-field unpatched form.
fn identity_prefix(key: &str) -> Option<&str> {
    match key.rsplit_once('-') {
        Some((prefix, base))
            if !prefix.is_empty()
                && !base.is_empty()
                && base.bytes().all(|b| b.is_ascii_hexdigit())
                && prefix.contains('-') =>
        {
            Some(prefix)
        }
        _ => None,
    }
}

/// Remove the `.aot` and `.profile` of entries in `dir` that share `keep_key`'s identity but name a different
/// base, newest first, keeping `AOT_KEEP` entries in all (`keep_key` itself is always one of them and is never
/// removed). Returns how many keys were retired. `site` names the caller, and appears in the line below so a log
/// shows WHICH call site fired rather than only that retirement happened somewhere.
///
/// A `.aot.<pid>.tmp` (build()'s temporary file since upstream #7; `.aot.tmp` before) is a write in flight and is never considered. Unlinking a file another process has open is safe
/// on this platform, and `load_file` copies an AOT into the JIT code cache rather than mapping it, so a retired
/// entry cannot be pulled out from under a running process.
fn retire_superseded(dir: &std::path::Path, keep_key: &str, site: &str) -> usize {
    let prefix = match identity_prefix(keep_key) {
        Some(p) => p,
        None => return 0,
    };
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return 0,
    };
    let mut others: Vec<(std::time::SystemTime, String)> = Vec::new();
    for ent in rd.flatten() {
        let name = ent.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        let stem = match name.strip_suffix(".aot") {
            Some(s) => s,
            None => match name.strip_suffix(".profile") {
                Some(s) => s,
                None => continue,
            },
        };
        if stem == keep_key || identity_prefix(stem) != Some(prefix) {
            continue;
        }
        let t = ent
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        match others.iter_mut().find(|(_, s)| s == stem) {
            // One key, two files: rank it by the newer of the pair.
            Some(slot) => {
                if t > slot.0 {
                    slot.0 = t;
                }
            }
            None => others.push((t, stem.to_string())),
        }
    }
    others.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut retired = 0;
    for (_, stem) in others.into_iter().skip(AOT_KEEP.saturating_sub(1)) {
        let _ = std::fs::remove_file(dir.join(format!("{stem}.aot")));
        let _ = std::fs::remove_file(dir.join(format!("{stem}.profile")));
        retired += 1;
    }
    // Say so, in the engine's own shape, and name the call site. Ungated, so a daemon emits it too -- but only
    // when something really went, so a run whose base did not move stays silent. One line per slid exec at most.
    if retired > 0 {
        eprintln!(
            "[maclator] aot: retired {retired} superseded {} at {site} for {prefix}, keeping {AOT_KEEP} ({keep_key})",
            if retired == 1 { "entry" } else { "entries" }
        );
    } else if std::env::var_os("MACLATOR_TRACE").is_some() || std::env::var_os("MACLATOR_STATS").is_some() {
        eprintln!("[maclator] aot: retired 0 superseded entries at {site} for {prefix} ({keep_key})");
    }
    retired
}

fn flags_word() -> u32 {
    crate::jit::commpage_checks() as u32 | ((crate::jit::emit::native_fma() as u32) << 1)
        | ((crate::jit::emit::direct_operands() as u32) << 2)
        | ((crate::jit::emit::loop_cache() as u32) << 3)
        | ((maclator_core::mem::native_pair_atomic() as u32) << 4)
        | ((crate::jit::emit::register_budget() as u32) << 5)
        | ((crate::jit::emit::cold_budget() as u32) << 7)
        | ((crate::jit::emit::csel_chain() as u32) << 8)
        | ((crate::jit::emit::packed_flags() as u32) << 11)
}

/// Register the main executable (mapped at a fixed address).
pub fn register_main(path: &str, uuid: [u8; 16], start: u64, end: u64, entries: Vec<u64>) {
    let key = image_key("main", &uuid, path, start);
    add_image(Image { name: path.rsplit('/').next().unwrap_or(path).to_string(), key, start, end, entries, loaded: false, cached: false });
}

/// Register dyld (also mapped at a fixed address).
pub fn register_dyld(path: &str, uuid: [u8; 16], start: u64, end: u64) {
    let key = image_key("dyld", &uuid, path, start);
    add_image(Image { name: "dyld".into(), key, start, end, entries: vec![], loaded: false, cached: false });
}

/// The AOT and profile identity of the shared cache: its UUID and the base it is mapped at. Like an image's
/// (`image_key`), a translation of cache code is valid only at the base it was made for, and the cache slides.
fn cache_key(uuid: &[u8; 16], start: u64) -> String {
    format!("dsc-{}-{start:x}", hex(uuid))
}

/// Register the shared cache once dyld has had us map it.
pub fn register_cache(uuid: [u8; 16], start: u64, end: u64) {
    let key = cache_key(&uuid, start);
    add_image(Image { name: "dyld_shared_cache".into(), key, start, end, entries: vec![], loaded: false, cached: false });
    ensure_loaded();
}

/// Register the arm64 Mach-O whose header is mapped at `addr` (a dyld-loaded library or
/// helper image). Keyed by UUID and load address, so it only hits the cache when the
/// library loads at the same place again.
pub fn register_image_at(addr: u64) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let rd = |o: u64| unsafe { std::ptr::read_unaligned((addr + o) as *const u32) };
    // mach_header_64: magic, cputype, ..., ncmds at +16, sizeofcmds at +20
    if rd(0) != 0xfeed_facf || rd(4) != 0x0100_000c {
        if std::env::var_os("MACLATOR_STATS").is_some() {
            eprintln!("[maclator] aot: mmap exec at {addr:#x} not an arm64 mach-o ({:#x} {:#x})", rd(0), rd(4));
        }
        return;
    }
    let ncmds = rd(16);
    let mut off = 32u64;
    let mut uuid = [0u8; 16];
    let (mut text_vm, mut lo, mut hi) = (u64::MAX, u64::MAX, 0u64);
    for _ in 0..ncmds.min(512) {
        let cmd = rd(off);
        let size = rd(off + 4) as u64;
        if size < 8 || off + size > 0x4000 {
            return;
        }
        if cmd == 0x1b {
            for i in 0..16 {
                uuid[i] = unsafe { *((addr + off + 8 + i as u64) as *const u8) };
            }
        } else if cmd == 0x19 {
            let mut name = [0u8; 16];
            for i in 0..16 {
                name[i] = unsafe { *((addr + off + 8 + i as u64) as *const u8) };
            }
            let vmaddr = unsafe { std::ptr::read_unaligned((addr + off + 24) as *const u64) };
            let vmsize = unsafe { std::ptr::read_unaligned((addr + off + 32) as *const u64) };
            if name.starts_with(b"__PAGEZERO") {
                off += size;
                continue;
            }
            if name.starts_with(b"__TEXT\0") {
                text_vm = vmaddr;
            }
            lo = lo.min(vmaddr);
            hi = hi.max(vmaddr + vmsize);
        }
        off += size;
    }
    if text_vm == u64::MAX || uuid == [0u8; 16] || hi <= lo {
        return;
    }
    let slide = addr.wrapping_sub(text_vm);
    let (start, end) = (lo.wrapping_add(slide), hi.wrapping_add(slide));
    let key = format!("lib-{}-{:x}", hex(&uuid), start);
    if std::env::var_os("MACLATOR_STATS").is_some() {
        eprintln!("[maclator] aot: registered image {key} [{start:#x}, {end:#x})");
    }
    add_image(Image { name: format!("lib-{}", &hex(&uuid)[..8]), key, start, end, entries: vec![], loaded: false, cached: false });
    ensure_loaded();
}

fn add_image(img: Image) {
    let mut v = IMAGES.lock().unwrap();
    if v.iter().any(|i| i.key == img.key) {
        return;
    }
    v.push(img);
    SEEN.lock().unwrap().push(HashSet::new());
}

thread_local! {
    /// (start, end, image index) of the image this thread last recorded a block in.
    static LAST_IMG: std::cell::Cell<(u64, u64, usize)> = const { std::cell::Cell::new((1, 0, 0)) };
    static PENDING: std::cell::RefCell<Vec<(usize, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record that the guest block at `pc` executed (JIT-translated or cold-interpreted), so the
/// next AOT build covers it.
pub fn note_translated(pc: u64) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let (start, end, idx) = LAST_IMG.with(|l| l.get());
    let idx = if pc >= start && pc < end {
        idx
    } else {
        let imgs = IMAGES.lock().unwrap();
        match imgs.iter().position(|im| pc >= im.start && pc < im.end) {
            Some(i) => {
                LAST_IMG.with(|l| l.set((imgs[i].start, imgs[i].end, i)));
                i
            }
            None => return,
        }
    };
    // Batch inserts so the global lock is taken once per 256 blocks.
    let flush = PENDING.with(|p| {
        let mut p = p.borrow_mut();
        p.push((idx, pc));
        if p.len() >= 256 {
            Some(std::mem::take(&mut *p))
        } else {
            None
        }
    });
    if let Some(batch) = flush {
        let mut seen = SEEN.lock().unwrap();
        for (i, pc) in batch {
            seen[i].insert(pc);
        }
    }
}

/// Publish this thread's not-yet-flushed blocks (called before saving the profile).
pub fn flush_pending() {
    let batch = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
    let mut seen = SEEN.lock().unwrap();
    for (i, pc) in batch {
        seen[i].insert(pc);
    }
}

fn file_for(img: &Image, ext: &str) -> std::path::PathBuf {
    cache_dir().join(format!("{}.{}", img.key, ext))
}

/// Load AOT code for any registered image not loaded yet.
pub fn ensure_loaded() {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut imgs = IMAGES.lock().unwrap();
    for img in imgs.iter_mut() {
        if img.loaded {
            continue;
        }
        img.loaded = true;
        let begun = std::time::Instant::now();
        match load_file(&file_for(img, "aot"), img.start, img.end) {
            Ok(n) if n > 0 => {
                img.cached = true;
                if std::env::var_os("MACLATOR_TRACE").is_some() || std::env::var_os("MACLATOR_STATS").is_some() {
                    // The file's name too, so a test can match a load to the file it read.
                    eprintln!("[maclator] aot: loaded {n} blocks for {} (file {}.aot) in {:.1} ms", img.name, img.key, begun.elapsed().as_secs_f64() * 1000.0);
                }
            }
            _ => {}
        }
    }
}

const CODE_ALIGN: usize = 0x4000;

fn code_offset(n: usize) -> usize {
    (32 + n * 16 + CODE_ALIGN - 1) & !(CODE_ALIGN - 1)
}

// Bound disk-controlled sizes before allocating memory or publishing executable
// code. The current cache is normally a few MiB to tens of MiB per image.
const MAX_AOT_BLOCKS: usize = 4_000_000;
const MAX_AOT_CODE: usize = 512 << 20;

fn cache_layout(head: &[u8; 32], file_len: u64) -> Option<(usize, usize, usize)> {
    let rd32 = |o| u32::from_le_bytes(head[o..o + 4].try_into().unwrap());
    let rd64 = |o| u64::from_le_bytes(head[o..o + 8].try_into().unwrap());
    if &head[..8] != MAGIC || rd32(8) != TRANSLATOR_VERSION || rd32(12) != flags_word() {
        return None;
    }
    let n = usize::try_from(rd64(16)).ok()?;
    let code_len = usize::try_from(rd64(24)).ok()?;
    if n == 0 || n > MAX_AOT_BLOCKS || code_len == 0 || code_len > MAX_AOT_CODE {
        return None;
    }
    let code_off = code_offset(n);
    if (code_off + code_len) as u64 > file_len {
        return None;
    }
    Some((n, code_len, code_off))
}

fn load_file(path: &std::path::Path, start: u64, end: u64) -> std::io::Result<usize> {
    let mut f = std::fs::File::open(path)?;
    // The code is run as it is, so only a file of ours in a directory of ours, both writable by no one else.
    let uid = unsafe { libc::geteuid() };
    let dir_ok = path.parent().is_some_and(|d| std::fs::symlink_metadata(d).is_ok_and(|md| md.is_dir() && owned(&md, uid)));
    if !dir_ok || !f.metadata().is_ok_and(|md| md.is_file() && owned(&md, uid)) {
        return Ok(0);
    }
    let mut head = [0u8; 32];
    if f.read_exact(&mut head).is_err() {
        return Ok(0);
    }
    let rd64 = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let Some((n, code_len, code_off)) = cache_layout(&head, f.metadata()?.len()) else {
        return Ok(0);
    };
    let mut idx = vec![0u8; n * 16];
    f.read_exact(&mut idx)?;
    // Never let an invalid index publish arbitrary host addresses. Validate the
    // whole image before registering any entry; a partial load is not useful.
    for entry in idx.chunks_exact(16) {
        let pc = rd64(entry, 0);
        if pc < start || pc >= end || pc & 3 != 0 || rd64(entry, 8) >= code_len as u64 {
            return Ok(0);
        }
    }
    // Copy the code into the JIT code cache. (Mapping the file itself PROT_EXEC makes macOS
    // run a Gatekeeper assessment on every cache file and pop up "could not verify" dialogs.)
    let cc = crate::jit::code_cache();
    let base = cc.alloc(code_len as u64);
    {
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(code_off as u64))?;
        let dst = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, code_len) };
        f.read_exact(dst)?;
    }
    crate::jit::register_blocks(idx.chunks_exact(16).map(|entry| (rd64(entry, 0), base + rd64(entry, 8))));
    Ok(n)
}

fn read_profile(path: &std::path::Path) -> BTreeSet<u64> {
    let mut s = BTreeSet::new();
    if let Ok(d) = std::fs::read(path) {
        for c in d.chunks_exact(8) {
            s.insert(u64::from_le_bytes(c.try_into().unwrap()));
        }
    }
    s
}

/// Save this run's profile; if it contains new blocks, rebuild AOT files in
/// a background (forked) process.
pub fn save_profile() {
    if !ENABLED.load(Ordering::Relaxed) || SAVED.swap(true, Ordering::SeqCst) {
        return;
    }
    flush_pending();
    let imgs = IMAGES.lock().unwrap().clone();
    let seen = SEEN.lock().unwrap().clone();
    if !owned_dir(&cache_dir(), unsafe { libc::geteuid() }) {
        return;
    }
    let mut rebuild = Vec::new();
    for (i, img) in imgs.iter().enumerate() {
        let path = file_for(img, "profile");
        let mut prof = read_profile(&path);
        let before = prof.len();
        prof.extend(seen[i].iter().copied());
        // A file left by an older translator is present but unusable. Rebuild
        // it even when this run saw no blocks beyond the existing profile.
        let aot_missing = !img.cached;
        if prof.len() != before || (aot_missing && (!prof.is_empty() || !img.entries.is_empty())) {
            let mut buf = Vec::with_capacity(prof.len() * 8);
            for pc in &prof {
                buf.extend_from_slice(&pc.to_le_bytes());
            }
            let _ = std::fs::write(&path, buf);
            retire_superseded(&cache_dir(), &img.key, "save_profile");
            rebuild.push((img.clone(), prof));
        }
    }
    if rebuild.is_empty() || std::env::var_os("MACLATOR_NO_AOT_BUILD").is_some() {
        return;
    }
    let sync = std::env::var_os("MACLATOR_AOT_SYNC").is_some();
    // Fork: the child still has the guest's memory mapped, so it can read
    // the code it needs to translate.
    let pid = if sync { 0 } else { crate::forksafe::fork() };
    if pid != 0 {
        return;
    }
    if !sync {
        unsafe {
            libc::setsid();
            let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
            libc::dup2(null, 0);
            libc::dup2(null, 1);
            libc::dup2(null, 2);
        }
    }
    // With MACLATOR_AOT_DONE_FILE set, the builder
    // says it finished, and how each image went, in that file (written whole, by rename) as its
    // last act: "aot-build pid=<pid> images=<n> ok=<k> failed=<f>". A test can then tell a builder
    // that finished from one still running or killed, which a count of .aot files cannot.
    let done = std::env::var_os("MACLATOR_AOT_DONE_FILE");
    let (mut ok, mut failed) = (0usize, 0usize);
    let images = rebuild.len();
    for (img, prof) in rebuild {
        match build(&img, &prof) {
            Ok(()) => ok += 1,
            Err(_) => failed += 1,
        }
    }
    if let Some(done) = done {
        let done = std::path::PathBuf::from(done);
        let tmp = done.with_extension("tmp");
        let line = format!("aot-build pid={} images={images} ok={ok} failed={failed}\n", std::process::id());
        if std::fs::write(&tmp, line).is_ok() {
            let _ = std::fs::rename(&tmp, &done);
        }
    }
    if !sync {
        unsafe { libc::_exit(0) };
    }
}

/// Collect block starts reachable from `entries` via direct branches.
fn discover(entries: &[u64], start: u64, end: u64, read: &dyn Fn(u64) -> Option<u32>, limit: usize) -> BTreeSet<u64> {
    let mut out = BTreeSet::new();
    let mut work: Vec<u64> = entries.to_vec();
    while let Some(pc) = work.pop() {
        if pc < start || pc >= end || pc & 3 != 0 || !out.insert(pc) || out.len() > limit {
            continue;
        }
        let mut cur = pc;
        for _ in 0..emit::MAX_BLOCK_INSNS {
            let Some(insn) = read(cur) else { break };
            if insn == 0 {
                break;
            }
            let next = cur + 4;
            let sext = |v: u64, n: u32| (((v << (64 - n)) as i64) >> (64 - n)) as u64;
            if insn & 0x7C00_0000 == 0x1400_0000 {
                let t = cur.wrapping_add(sext((insn & 0x03ff_ffff) as u64, 26) << 2);
                if insn & 0x8000_0000 != 0 {
                    work.push(next); // return point after BL
                } else {
                    work.push(t);
                }
                break;
            } else if insn & 0xFF00_0000 == 0x5400_0000 || insn & 0x7E00_0000 == 0x3400_0000 {
                work.push(cur.wrapping_add(sext(((insn >> 5) & 0x7ffff) as u64, 19) << 2));
                work.push(next);
                break;
            } else if insn & 0x7E00_0000 == 0x3600_0000 {
                work.push(cur.wrapping_add(sext(((insn >> 5) & 0x3fff) as u64, 14) << 2));
                work.push(next);
                break;
            } else if emit::is_terminator(insn) {
                if insn & 0xFE00_0000 == 0xD600_0000 && (insn >> 21) & 0xf == 1 {
                    work.push(next); // BLR returns here
                }
                if insn & 0xFFE0_001F == 0xD400_0001 {
                    work.push(next); // SVC continues
                }
                break;
            }
            cur = next;
        }
    }
    out
}

/// Open `p` for writing as a file only we can write, whatever the umask. `load_file` refuses an AOT file that
/// group or other can write, so one created 0664 under umask 0002 would be refused at every load and rebuilt at
/// every exit. A file left at `p` by an earlier process with our pid keeps its old mode, so the mode is set too.
fn create_private(p: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(p)?;
    f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    Ok(f)
}

fn build(img: &Image, profile: &BTreeSet<u64>) -> std::io::Result<()> {
    let read = crate::jit::guest_reader();
    let mut blocks: BTreeSet<u64> = profile.clone();
    if !img.entries.is_empty() {
        blocks.extend(discover(&img.entries, img.start, img.end, &read, 400_000));
    }
    // Profile-guided expansion: also follow direct branches from hot blocks.
    let hot: Vec<u64> = profile.iter().copied().collect();
    blocks.extend(discover(&hot, img.start, img.end, &read, profile.len() * 4 + 1000));
    let blocks: Vec<u64> = blocks.into_iter().filter(|&pc| read(pc).is_some()).collect();
    if blocks.is_empty() {
        return Ok(());
    }
    // Translate in chunks so a single assembler run stays manageable.
    let mut code: Vec<u8> = Vec::new();
    let mut index: Vec<(u64, u64)> = Vec::new();
    for chunk in blocks.chunks(4096) {
        let mut u = Unit::new(StubKind::Plain, crate::jit::commpage_checks());
        for &pc in chunk {
            let l = u.a.create_label();
            u.labels.insert(pc, l);
        }
        for &pc in chunk {
            emit::emit_block(&mut u, pc, &read);
        }
        u.finish();
        let base = code.len() as u64;
        let res = match u.a.assemble_options(base, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for &pc in chunk {
            if let Ok(ip) = res.label_ip(&u.labels[&pc]) {
                index.push((pc, ip));
            }
        }
        code.extend_from_slice(&res.inner.code_buffer);
        while code.len() % 16 != 0 {
            code.push(0xcc);
        }
    }
    let path = file_for(img, "aot");
    // Several applications can finish simultaneously. Each builder writes its
    // own temporary file, then atomically replaces the complete cache.
    let tmp = path.with_extension(format!("aot.{}.tmp", std::process::id()));
    let mut f = create_private(&tmp)?;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(MAGIC);
    hdr.extend_from_slice(&TRANSLATOR_VERSION.to_le_bytes());
    hdr.extend_from_slice(&flags_word().to_le_bytes());
    hdr.extend_from_slice(&(index.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&(code.len() as u64).to_le_bytes());
    for (pc, off) in &index {
        hdr.extend_from_slice(&pc.to_le_bytes());
        hdr.extend_from_slice(&off.to_le_bytes());
    }
    while hdr.len() < code_offset(index.len()) {
        hdr.push(0);
    }
    f.write_all(&hdr)?;
    f.write_all(&code)?;
    drop(f);
    std::fs::rename(tmp, path)?;
    // This image is now cached for the base it is loaded at; anything cached for a base it will never be at
    // again is dead weight. Without this the store grows by one entry per slid exec, for ever.
    retire_superseded(&cache_dir(), &img.key, "build");
    Ok(())
}

/// `maclator --aot <exe>`: translate the main executable ahead of time
/// without running it.
pub fn build_standalone(path: &str) -> std::io::Result<usize> {
    let m = crate::macho::MachO::open_arm64(path, false)?;
    let (lo, hi) = m.vm_range();
    let file = std::fs::File::open(path)?;
    use std::os::unix::io::AsRawFd;
    let slide = crate::loader::MAIN_BASE.wrapping_sub(lo);
    // Map __TEXT read-only at the runtime address.
    for seg in &m.segments {
        if seg.name != "__TEXT" {
            continue;
        }
        let addr = seg.vmaddr.wrapping_add(slide);
        let p = unsafe {
            libc::mmap(addr as *mut _, seg.filesize as usize, libc::PROT_READ, libc::MAP_PRIVATE | libc::MAP_FIXED, file.as_raw_fd(), (m.slice_offset + seg.fileoff) as i64)
        };
        if p as u64 != addr {
            return Err(std::io::Error::last_os_error());
        }
    }
    let entries = function_starts(path, &m, slide);
    let img_start = lo.wrapping_add(slide);
    let img_end = hi.wrapping_add(slide);
    let key = image_key("main", &m.uuid, path, img_start);
    let img = Image { name: path.to_string(), key, start: img_start, end: img_end, entries, loaded: false, cached: false };
    if !owned_dir(&cache_dir(), unsafe { libc::geteuid() }) {
        return Err(std::io::Error::other("the cache directory is not ours alone"));
    }
    let prof = read_profile(&file_for(&img, "profile"));
    build(&img, &prof)?;
    let n = load_count(&file_for(&img, "aot"));
    Ok(n)
}

fn load_count(p: &std::path::Path) -> usize {
    // This query only needs the header, not another read of the entire code.
    let Ok(mut f) = std::fs::File::open(p) else { return 0 };
    let mut head = [0u8; 32];
    if f.read_exact(&mut head).is_err() {
        return 0;
    }
    f.metadata().ok().and_then(|m| cache_layout(&head, m.len())).map(|(n, _, _)| n).unwrap_or(0)
}

/// Function entry points from LC_FUNCTION_STARTS (runtime addresses).
pub fn function_starts(path: &str, m: &crate::macho::MachO, slide: u64) -> Vec<u64> {
    let Some((off, size)) = m.function_starts else { return vec![] };
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return vec![],
    };
    use std::io::Seek;
    let mut buf = vec![0u8; size as usize];
    if f.seek(std::io::SeekFrom::Start(m.slice_offset + off as u64)).is_err() || f.read_exact(&mut buf).is_err() {
        return vec![];
    }
    crate::macho::decode_function_starts(&buf, m.text_vmaddr()).into_iter().map(|a| a.wrapping_add(slide)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(n: u64, code_len: u64) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&TRANSLATOR_VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&flags_word().to_le_bytes());
        h[16..24].copy_from_slice(&n.to_le_bytes());
        h[24..32].copy_from_slice(&code_len.to_le_bytes());
        h
    }

    #[test]
    fn cache_header_rejects_stale_truncated_and_oversized_files() {
        let h = header(20, 128);
        let expected_len = (code_offset(20) + 128) as u64;
        assert_eq!(cache_layout(&h, expected_len), Some((20, 128, code_offset(20))));
        assert_eq!(cache_layout(&h, expected_len - 1), None);
        assert_eq!(cache_layout(&header(u64::MAX, 128), u64::MAX), None);
        assert_eq!(cache_layout(&header(20, u64::MAX), u64::MAX), None);
        assert_eq!(cache_layout(&header(0, 128), expected_len), None);
        let mut stale = h;
        stale[8..12].copy_from_slice(&(TRANSLATOR_VERSION - 1).to_le_bytes());
        assert_eq!(cache_layout(&stale, expected_len), None);
        let mut different_features = h;
        different_features[12..16].copy_from_slice(&(flags_word() ^ 2).to_le_bytes());
        assert_eq!(cache_layout(&different_features, expected_len), None);
        different_features[12..16].copy_from_slice(&(flags_word() ^ 4).to_le_bytes());
        assert_eq!(cache_layout(&different_features, expected_len), None);
        different_features[12..16].copy_from_slice(&(flags_word() ^ 8).to_le_bytes());
        assert_eq!(cache_layout(&different_features, expected_len), None);
        different_features[12..16].copy_from_slice(&(flags_word() ^ 16).to_le_bytes());
        assert_eq!(cache_layout(&different_features, expected_len), None);
        different_features[12..16].copy_from_slice(&(flags_word() ^ (1 << 11)).to_le_bytes());
        assert_eq!(cache_layout(&different_features, expected_len), None);
    }

    #[test]
    fn load_rejects_invalid_index_before_publishing_code() {
        let p = std::env::temp_dir().join(format!("maclator-invalid-aot-{}", std::process::id()));
        let mut bytes = header(1, 16).to_vec();
        bytes.extend_from_slice(&0x1000u64.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes()); // exactly past the code
        bytes.resize(code_offset(1) + 16, 0);
        std::fs::write(&p, bytes).unwrap();
        assert_eq!(load_file(&p, 0x1000, 0x2000).unwrap(), 0);
        std::fs::remove_file(p).unwrap();
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;

    fn chmod(p: &std::path::Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    /// An AOT file of one block (16 bytes of int3) for guest pc `pc`, in `dir`.
    fn aot_file(dir: &std::path::Path, pc: u64) -> std::path::PathBuf {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&TRANSLATOR_VERSION.to_le_bytes());
        b.extend_from_slice(&flags_word().to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&16u64.to_le_bytes());
        b.extend_from_slice(&pc.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.resize(code_offset(1), 0);
        b.extend_from_slice(&[0xcc; 16]);
        let p = dir.join("t.aot");
        std::fs::write(&p, b).unwrap();
        p
    }

    /// A few arm64 `ret`s in this process's own memory: real guest code for build() to translate.
    pub(super) fn rets() -> Vec<u32> {
        vec![0xd65f_03c0; 16]
    }

    #[test]
    fn an_aot_file_built_under_umask_0002_is_loaded() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        if let Some(marker) = std::env::var_os("MCL_AOT_UMASK") {
            unsafe { libc::umask(0o002) };
            let d = cache_dir();
            let plain = d.join("plain");
            drop(std::fs::File::create(&plain).unwrap());
            assert_eq!(mode(&plain), 0o664, "control: the umask applies to a plain create");
            let code = rets();
            let (start, end) = (code.as_ptr() as u64, code.as_ptr() as u64 + 4 * code.len() as u64);
            let img = Image { name: "t".into(), key: "umask-test".into(), start, end, entries: vec![], loaded: false, cached: false };
            build(&img, &BTreeSet::from([start])).unwrap();
            let p = file_for(&img, "aot");
            assert_eq!(mode(&p), 0o600, "the published file is ours alone");
            assert!(load_file(&p, start, end).unwrap() >= 1, "and it is loaded");
            // A temporary file left by an earlier process with the same pid.
            let tmp = d.join("t.aot.1.tmp");
            std::fs::write(&tmp, b"old").unwrap();
            chmod(&tmp, 0o666);
            drop(create_private(&tmp).unwrap());
            assert_eq!(mode(&tmp), 0o600, "a reused temporary file is made ours alone");
            std::fs::write(&marker, b"ran").unwrap();
            return;
        }
        let d = std::env::temp_dir().join(format!("mcl-aot-umask-{}", std::process::id()));
        let cache = d.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        chmod(&cache, 0o700);
        let marker = d.join("ran");
        // The child says it ran its checks: libtest exits 0 having run nothing if the name stops matching.
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "aot::owner_tests::an_aot_file_built_under_umask_0002_is_loaded", "--test-threads=1"])
            .env("MCL_AOT_UMASK", &marker)
            .env("MACLATOR_CACHE", &cache)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        let ran = std::fs::read(&marker).ok();
        std::fs::remove_dir_all(&d).ok();
        assert!(st.success(), "the child under umask 0002: {st}");
        assert_eq!(ran.as_deref(), Some(&b"ran"[..]), "the child ran its checks");
    }

    #[test]
    fn load_refuses_an_aot_file_or_directory_writable_by_others() {
        /// Removes the test's directory when the test ends, a panic included.
        struct Gone(std::path::PathBuf);
        impl Drop for Gone {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let d = std::env::temp_dir().join(format!("mcl-aot-owner-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        let _gone = Gone(d.clone());
        chmod(&d, 0o755);
        let p = aot_file(&d, 0x7d00_0000_0000);
        assert_eq!(load_file(&p, 0x7d00_0000_0000, 0x7d00_0000_1000).unwrap(), 1, "control: ours");
        for (what, file, dir) in [("group-writable file", 0o664, 0o755), ("other-writable file", 0o646, 0o755), ("group-writable directory", 0o644, 0o775), ("other-writable directory", 0o644, 0o757)] {
            chmod(&p, file);
            chmod(&d, dir);
            assert_eq!(load_file(&p, 0x7d00_0000_0000, 0x7d00_0000_1000).unwrap(), 0, "{what}");
        }
        chmod(&p, 0o644);
        chmod(&d, 0o755);
        assert_eq!(load_file(&p, 0x7d00_0000_0000, 0x7d00_0000_1000).unwrap(), 1, "control: ours again");
        let link = d.join("l");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&d, &link).unwrap();
        assert_eq!(load_file(&link.join("t.aot"), 0x7d00_0000_0000, 0x7d00_0000_1000).unwrap(), 0, "a directory reached through a symlink");
    }
}

#[cfg(test)]
mod image_key_tests {
    use super::*;

    /// A slid image must not share a cache with the same image at another base.
    #[test]
    fn the_key_names_the_base() {
        let u = [7u8; 16];
        let a = image_key("main", &u, "/bin/x", 0x5_0000_0000);
        let b = image_key("main", &u, "/bin/x", 0x5_0000_4000);
        assert_ne!(a, b, "two bases, one key: a slid image would run another base's translation");
        assert_eq!(a, image_key("main", &u, "/bin/x", 0x5_0000_0000), "one base, one key");
    }

    /// A slid shared cache must not share a cache with the same cache at another base.
    #[test]
    fn the_cache_key_names_the_base() {
        let u = [9u8; 16];
        assert_ne!(cache_key(&u, 0x6_0000_0000), cache_key(&u, 0x6_0000_4000), "two bases, one cache key");
        assert_eq!(cache_key(&u, 0x6_0000_0000), cache_key(&u, 0x6_0000_0000), "one base, one cache key");
    }
}

#[cfg(test)]
mod retire_tests {
    use super::*;

    /// A store of our own, so a test never touches the real one.
    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "maclator-retire-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map(|x| x.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn put(dir: &std::path::Path, stem: &str, bytes: usize) {
        std::fs::write(dir.join(format!("{stem}.aot")), vec![0u8; bytes]).unwrap();
        std::fs::write(dir.join(format!("{stem}.profile")), vec![0u8; 8]).unwrap();
    }

    fn count(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir).unwrap().flatten().count()
    }

    #[test]
    fn the_prefix_is_the_key_without_its_base() {
        let u = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            identity_prefix(&format!("main-{u}-aabbccddeeff0011-500000000")),
            Some(format!("main-{u}-aabbccddeeff0011").as_str())
        );
        assert_eq!(
            identity_prefix(&format!("dsc-{u}-600000000")),
            Some(format!("dsc-{u}").as_str())
        );
        assert_eq!(
            identity_prefix(&format!("lib-{u}-106340000")),
            Some(format!("lib-{u}").as_str())
        );
        // Two fields only: an unpatched build's key, which names no base and is never retired.
        assert_eq!(identity_prefix(&format!("dsc-{u}")), None);
        // A trailing field that is not hex is not a base.
        assert_eq!(identity_prefix("main-aa-bb-zznothex"), None);
        assert_eq!(identity_prefix("nodashes"), None);
    }

    /// ⭐ The reading this patch exists for: repeated slid loads of ONE binary leave the store bounded.
    #[test]
    fn repeated_slid_loads_of_one_binary_stay_bounded() {
        let d = tmpdir("bounded");
        let ident = "main-0123456789abcdef0123456789abcdef-aabbccddeeff0011";
        let mut high = 0usize;
        for i in 0..200u64 {
            // A declined hint slides to a fresh base, so each exec mints a new key.
            let key = format!("{ident}-{:x}", 0x5_0000_0000u64 + i * 0x4000);
            put(&d, &key, 4096);
            // The peak is HERE, between the write and the retire that follows it. Reading it after the retire
            // would only ever see the bound, so the high-water assertion below could not fail.
            high = high.max(count(&d));
            retire_superseded(&d, &key, "test");
        }
        assert_eq!(
            count(&d),
            AOT_KEEP * 2,
            "after 200 slid loads the store holds {} files, not the {} the bound allows",
            count(&d),
            AOT_KEEP * 2
        );
        // The peak is one pair over the bound: the new entry's two files exist for the moment between the write
        // and the retire that follows it. Asserting the PEAK and not only the end state is what rules out a
        // store that balloons mid-loop and is tidied only at the end.
        assert_eq!(
            high,
            AOT_KEEP * 2 + 2,
            "high-water mark {high} files, want {} -- the bound plus the pair just written",
            AOT_KEEP * 2 + 2
        );
        std::fs::remove_dir_all(&d).ok();
    }

    /// The negative control: WITHOUT the retire call the same loop grows without bound, so the test above is
    /// capable of failing. A test that cannot see the defect proves nothing about the fix.
    #[test]
    fn without_retirement_the_same_loop_grows() {
        let d = tmpdir("control");
        let ident = "main-0123456789abcdef0123456789abcdef-aabbccddeeff0011";
        for i in 0..200u64 {
            put(&d, &format!("{ident}-{:x}", 0x5_0000_0000u64 + i * 0x4000), 4096);
        }
        assert_eq!(count(&d), 400, "the control must accumulate every entry");
        std::fs::remove_dir_all(&d).ok();
    }

    /// Another image is untouched: retirement is scoped to one identity.
    #[test]
    fn another_images_entries_survive() {
        let d = tmpdir("scope");
        let mine = "main-0123456789abcdef0123456789abcdef-aabbccddeeff0011";
        let theirs = "main-fedcba9876543210fedcba9876543210-1122334455667788";
        let cache = "dsc-0123456789abcdef0123456789abcdef";
        put(&d, &format!("{mine}-500000000"), 16);
        put(&d, &format!("{theirs}-500000000"), 16);
        put(&d, &format!("{theirs}-500004000"), 16);
        put(&d, &format!("{cache}-600000000"), 16);
        let keep = format!("{mine}-500008000");
        put(&d, &keep, 16);
        let gone = retire_superseded(&d, &keep, "test");
        assert_eq!(gone, 1, "exactly my own superseded entry should go");
        assert!(d.join(format!("{keep}.aot")).exists(), "the entry in use must stay");
        assert!(!d.join(format!("{mine}-500000000.aot")).exists(), "my superseded entry must go");
        assert!(d.join(format!("{theirs}-500000000.aot")).exists(), "another image must be untouched");
        assert!(d.join(format!("{theirs}-500004000.aot")).exists(), "including its own older base");
        assert!(d.join(format!("{cache}-600000000.aot")).exists(), "and so must the shared cache's");
        std::fs::remove_dir_all(&d).ok();
    }

    /// A write in flight (`.aot.<pid>.tmp`, or the older `.aot.tmp`) is never removed, and a key naming no base is never retired.
    #[test]
    fn a_write_in_flight_and_an_unkeyed_entry_are_left_alone() {
        let d = tmpdir("inflight");
        let ident = "main-0123456789abcdef0123456789abcdef-aabbccddeeff0011";
        put(&d, &format!("{ident}-500000000"), 16);
        std::fs::write(d.join(format!("{ident}-500004000.aot.tmp")), [0u8; 4]).unwrap();
        std::fs::write(d.join(format!("{ident}-500004000.aot.4242.tmp")), [0u8; 4]).unwrap();
        std::fs::write(d.join(format!("{ident}.aot")), [0u8; 4]).unwrap();
        let keep = format!("{ident}-500008000");
        put(&d, &keep, 16);
        retire_superseded(&d, &keep, "test");
        assert!(d.join(format!("{ident}-500004000.aot.tmp")).exists(), "a temp file is a write in flight");
        assert!(d.join(format!("{ident}-500004000.aot.4242.tmp")).exists(), "so is build()'s per-pid temp file");
        assert!(d.join(format!("{ident}.aot")).exists(), "a key with no base names no run and is not ours to retire");
        assert_eq!(retire_superseded(&d, ident, "test"), 0, "and such a key retires nothing");
        std::fs::remove_dir_all(&d).ok();
    }

    /// Running it twice changes nothing the second time.
    #[test]
    fn retirement_is_idempotent() {
        let d = tmpdir("idem");
        let ident = "dyld-0123456789abcdef0123456789abcdef-aabbccddeeff0011";
        put(&d, &format!("{ident}-580000000"), 16);
        let keep = format!("{ident}-580004000");
        put(&d, &keep, 16);
        assert_eq!(retire_superseded(&d, &keep, "test"), 1);
        assert_eq!(retire_superseded(&d, &keep, "test"), 0, "the second run has nothing left to do");
        assert_eq!(count(&d), 2);
        std::fs::remove_dir_all(&d).ok();
    }

    /// An empty or missing directory is not an error.
    #[test]
    fn a_missing_store_is_not_an_error() {
        let d = tmpdir("missing");
        let gone = d.join("no-such-dir");
        assert_eq!(retire_superseded(&gone, "main-aa-bb-500000000", "test"), 0);
        assert_eq!(retire_superseded(&d, "main-aa-bb-500000000", "test"), 0);
        std::fs::remove_dir_all(&d).ok();
    }
}
