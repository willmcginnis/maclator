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
//! On exit, an image with no translation the run could load is built by a forked child in the background, so the
//! next launch starts with warm translations: once per image and translation format, by one builder at a time (see
//! "One build per image" below). Blocks a later run sees beyond the translation go into the profile, which the next
//! engine's build of the image uses.

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

/// `dir`, made first if missing (mode 0700, not in a sandboxed process), is a real directory (not a symlink) that is
/// `owned` by `uid`.
pub(crate) fn owned_dir(dir: &std::path::Path, uid: u32) -> bool {
    use std::os::unix::fs::DirBuilderExt;
    if !crate::sandboxcheck::sandboxed() {
        let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir);
    }
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
/// the last field is not hex or what precedes it has no `-` left, and such a key is never retired.
///
/// Every keyed family ends in `-{start:x}` and the hex digits carry no `-`, so the last `-` is the split:
/// `main-<uuid>-<fnv>-<base>`, `dyld-<uuid>-<fnv>-<base>`, `dsc-<uuid>-<base>`, `lib-<uuid>-<base>`. Keys from
/// before the base was part of them are not told apart by this split. The shared cache's `dsc-<uuid>` has no `-`
/// left in its prefix and gets `None`. The main executable's `main-<uuid>-<fnv>` (and dyld's) ends in a hex fnv,
/// so it splits into `main-<uuid>`; no current key has that prefix, since theirs carry the fnv, so such a file is
/// never retired either.
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
        // A key counts as retired when a file of it was removed and none is left; a file already gone is not an error.
        let (mut removed, mut left) = (false, false);
        for ext in ["aot", "profile"] {
            let p = dir.join(format!("{stem}.{ext}"));
            match std::fs::remove_file(&p) {
                Ok(()) => removed = true,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    left = true;
                    eprintln!("[maclator] aot: could not retire {}: {e}", p.display());
                }
            }
        }
        retired += (removed && !left) as usize;
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

/// A translation file at `path` that this engine would load for an image in [start, end): ours, in a directory of ours,
/// this format's header with sizes that fit the file, and an index every entry of which is in the image. The open file,
/// its index, and (blocks, code length, code offset). None: it would not be loaded.
fn validated(path: &std::path::Path, start: u64, end: u64) -> std::io::Result<Option<(std::fs::File, Vec<u8>, usize, usize, usize)>> {
    let mut f = std::fs::File::open(path)?;
    // The code is run as it is, so only a file of ours in a directory of ours, both writable by no one else.
    let uid = unsafe { libc::geteuid() };
    let dir_ok = path.parent().is_some_and(|d| std::fs::symlink_metadata(d).is_ok_and(|md| md.is_dir() && owned(&md, uid)));
    if !dir_ok || !f.metadata().is_ok_and(|md| md.is_file() && owned(&md, uid)) {
        return Ok(None);
    }
    let mut head = [0u8; 32];
    if f.read_exact(&mut head).is_err() {
        return Ok(None);
    }
    let rd64 = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let Some((n, code_len, code_off)) = cache_layout(&head, f.metadata()?.len()) else {
        return Ok(None);
    };
    let mut idx = vec![0u8; n * 16];
    f.read_exact(&mut idx)?;
    // Never let an invalid index publish arbitrary host addresses. Validate the
    // whole image before registering any entry; a partial load is not useful.
    for entry in idx.chunks_exact(16) {
        let pc = rd64(entry, 0);
        if pc < start || pc >= end || pc & 3 != 0 || rd64(entry, 8) >= code_len as u64 {
            return Ok(None);
        }
    }
    Ok(Some((f, idx, n, code_len, code_off)))
}

fn load_file(path: &std::path::Path, start: u64, end: u64) -> std::io::Result<usize> {
    // A sandboxed process reads the store only where its sandbox lets it, asked first so a refusal is never logged.
    if !crate::sandboxcheck::may_read(path) {
        crate::sandboxcheck::note_left_alone("not reading saved translations its sandbox does not let it read");
        return Ok(0);
    }
    let Some((mut f, idx, n, code_len, code_off)) = validated(path, start, end)? else { return Ok(0) };
    let rd64 = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
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

/// Save this run's profile; if an image has no translation that this run could load, and none has been made or tried
/// for it (see "One build per image" below), build it in a background (forked) process.
pub fn save_profile() {
    if !ENABLED.load(Ordering::Relaxed) || SAVED.swap(true, Ordering::SeqCst) {
        return;
    }
    // A sandboxed process writes nothing to the store, and so reads no profile and builds nothing, as Rosetta's
    // translations are made by a service of their own and never by the sandboxed program.
    if crate::sandboxcheck::sandboxed() {
        crate::sandboxcheck::note_left_alone("not saving translations");
        return;
    }
    flush_pending();
    let imgs = IMAGES.lock().unwrap().clone();
    let seen = SEEN.lock().unwrap().clone();
    let dir = cache_dir();
    if !owned_dir(&dir, unsafe { libc::geteuid() }) {
        return;
    }
    let mut rebuild = Vec::new();
    for (i, img) in imgs.iter().enumerate() {
        let path = file_for(img, "profile");
        let mut prof = read_profile(&path);
        let before = prof.len();
        prof.extend(seen[i].iter().copied());
        // A file left by an older translator is present but unusable: such an image is built even when this run saw
        // no blocks beyond the existing profile. An image whose translation was loaded is built already, and blocks
        // this run saw beyond it only go into its profile, for the next engine's build.
        let aot_missing = !img.cached;
        let wanted = aot_missing && (!prof.is_empty() || !img.entries.is_empty());
        if prof.len() != before || wanted {
            let mut buf = Vec::with_capacity(prof.len() * 8);
            for pc in &prof {
                buf.extend_from_slice(&pc.to_le_bytes());
            }
            // Retire what this image left at other bases only once its new profile is written: under disk pressure
            // the write can fail, and the old entry is then all there is.
            if std::fs::write(&path, buf).is_ok() {
                retire_superseded(&dir, &img.key, "save_profile");
            }
        }
        if wanted {
            rebuild.push((img.clone(), prof));
        }
    }
    if rebuild.is_empty() || std::env::var_os("MACLATOR_NO_AOT_BUILD").is_some() {
        return;
    }
    // Asked before forking, so an exit with nothing to build forks nothing. Nothing is locked or written here: the
    // builder decides again under the image's lock, and this answer only spares it the fork.
    let me = Asker::now();
    // A dead builder's temporary file may be what takes the room, and only a builder sweeps: sweep here too (it removes
    // only files older than any builder lives, AOT_TMP_STALE_S).
    if !has_room(&dir) && (sweep_stale_tmps(&dir) == 0 || !has_room(&dir)) {
        trace_build(format_args!("not building: under {} MiB free in the store", AOT_MIN_FREE >> 20));
        return;
    }
    // A library unmapped and replaced since it was registered is not this run's to build (nor to defer: the identity's
    // record is shared with a copy of it mapped elsewhere, which may be live).
    let read = crate::jit::guest_reader();
    rebuild.retain(|(img, _)| still_there(img, &read));
    rebuild.retain(|(img, _)| wants_build(&dir, img, &me));
    if rebuild.is_empty() {
        return;
    }
    if queue_full(&dir) {
        trace_build(format_args!("not building: a builder is running and {AOT_QUEUE} are waiting"));
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
        #[cfg(test)]
        once_tests::fork_hook();
        unsafe {
            libc::setsid();
            // What the program had open goes first, so the builder has descriptors to open anything with, and keeps
            // none of the program's lock files or sockets.
            close_inherited(&crate::engine::dump_pipe());
            let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
            libc::dup2(null, 0);
            libc::dup2(null, 1);
            libc::dup2(null, 2);
            if null > 2 {
                libc::close(null);
            }
            become_background_builder();
        }
    }
    // With MACLATOR_AOT_DONE_FILE set, the builder
    // says it finished, and how each image went, in that file (written whole, by rename) as its
    // last act: "aot-build pid=<pid> images=<n> ok=<k> skipped=<s> failed=<f>", where ok counts images whose
    // file was published and skipped those with no block to translate. A test can then tell a builder
    // that finished from one still running or killed, which a count of .aot files cannot. A builder that found
    // nothing left to build, or no room in the queue, writes no line.
    if let Some((images, ok, skipped, failed)) = build_queued(&dir, rebuild, &me, sync) {
        if let Some(done) = std::env::var_os("MACLATOR_AOT_DONE_FILE") {
            let pid = std::process::id();
            write_done(std::path::Path::new(&done), pid, &format!("aot-build pid={pid} images={images} ok={ok} skipped={skipped} failed={failed}\n"));
        }
    }
    if !sync {
        unsafe { libc::_exit(0) };
    }
}

// ---- One build per image, one builder at a time ----
//
// Every exit used to fork a builder when the image had no translation it could load or its run had seen a block the
// profile lacked. A daemon that dies and is started again every few seconds therefore kept many builders running at
// once, each translating the same image again, and they took half of a small machine's CPU. Now:
//  * an image is built at most once per AOT format: its identity (`identity_prefix`: kind, UUID, path) has a build
//    record, and an image whose record says it was built, or that its build failed with this engine, is not built
//    again (a failure is retried by another engine, a build deferred for want of space or another passing reason, a
//    translation found missing or unloadable and an image found at another base, each after an hour or in a later
//    boot);
//  * one builder at a time builds, per store: it holds `aot-builder.lock`. Up to AOT_QUEUE more wait their turn,
//    each holding one `aot-queue-<n>.lock`; a request that finds the queue full is dropped (a later exit asks again);
//  * a builder holds a write lock on each image record it took, while it waits and while it builds, so a second
//    request for an image that is queued or being built is a no-op. The kernel drops a dead process's locks, so a
//    builder that dies leaves no image held: its record says "building" with its lock free, which the next request
//    reads as a builder that died, and builds again, at most AOT_MAX_ATTEMPTS times per engine and boot;
//  * the builder keeps nothing the program had open (a lock file, a socket), runs at background priority, and is
//    ended after AOT_BUILD_CPU_S of CPU (a build that loops) or AOT_BUILD_WALL_S of time (one that is stuck).
// The locks are fcntl record locks on files of ours (0600, in the store's 0700 directory, never followed through a
// symlink). Rosetta's oahd translates each binary once and keeps it; this keeps that rule in the store, which every
// engine sharing it reads. An engine that predates the records ignores them, and is fought at most once an hour: an
// image whose record says built and whose file another engine's format replaced is built again after the hour.

/// Builders that may wait while one builds, per store.
const AOT_QUEUE: usize = 2;
/// How long a waiting builder waits for its turn before it gives up (a later exit asks again).
const AOT_QUEUE_WAIT: std::time::Duration = std::time::Duration::from_secs(20 * 60);
/// A builder that has used this much CPU is ended (SIGXCPU: a build that loops), and its images count one attempt each.
/// CPU, not time: a builder at background priority on a busy machine may take long without being stuck.
const AOT_BUILD_CPU_S: u64 = 20 * 60;
/// A builder still building after this long is ended (SIGALRM: one blocked without using CPU), its images counting an
/// attempt each.
const AOT_BUILD_WALL_S: u32 = 3 * 60 * 60;
/// Builds of one image by one engine that may die before the image is left alone for that engine.
const AOT_MAX_ATTEMPTS: u32 = 2;
/// Below this much free space in the store nothing is built (nor forked): a translation runs to tens of MiB.
const AOT_MIN_FREE: u64 = 256 << 20;
/// A deferred build (no space, no descriptors or memory), a translation found missing or unloadable since it was built,
/// and another engine's failed or dead build are tried again after this long, or in a later boot.
const AOT_RETRY_S: u64 = 60 * 60;
/// A builder's temporary file this old belongs to a builder that died: no builder lives longer (its queue wait and its
/// wall deadline), so one paused (a guest suspended) is not swept from under it.
const AOT_TMP_STALE_S: u64 = AOT_QUEUE_WAIT.as_secs() + AOT_BUILD_WALL_S as u64 + 60 * 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildState {
    Built,
    Failed,
    Building,
    /// Failed for a reason that passes: no space, no descriptors, no memory.
    Deferred,
}

impl BuildState {
    fn word(self) -> &'static str {
        match self {
            BuildState::Built => "built",
            BuildState::Failed => "failed",
            BuildState::Building => "building",
            BuildState::Deferred => "deferred",
        }
    }
    fn from_word(w: &str) -> Option<Self> {
        Some(match w {
            "built" => BuildState::Built,
            "failed" => BuildState::Failed,
            "building" => BuildState::Building,
            "deferred" => BuildState::Deferred,
            _ => return None,
        })
    }
}

/// One image identity's build record, one line in `<identity>.<format>.build`, ending in a checksum of the rest so a
/// torn write reads as no record (which costs one build at most).
#[derive(Clone, Debug, PartialEq, Eq)]
struct BuildRecord {
    state: BuildState,
    /// The engine that wrote it (`engine_id`).
    engine: String,
    /// The base the image was at (`Image::start`).
    base: u64,
    /// The boot it was written in (kern.bootsessionuuid; empty when unknown).
    boot: String,
    /// Builds of this image by this engine begun so far, the one in progress included.
    attempts: u32,
    /// Seconds since the epoch.
    time: u64,
    pid: u32,
    /// Blocks in the profile it was built from.
    blocks: usize,
}

impl BuildRecord {
    fn line(&self) -> String {
        let body = format!(
            "state={} engine={} base={:x} boot={} attempts={} time={} pid={} blocks={}",
            self.state.word(),
            self.engine,
            self.base,
            self.boot,
            self.attempts,
            self.time,
            self.pid,
            self.blocks
        );
        let sum = fnv(&[body.as_bytes()]);
        format!("{body} sum={sum}\n")
    }

    fn parse(bytes: &[u8]) -> Option<Self> {
        let line = std::str::from_utf8(bytes).ok()?.split_once('\n')?.0;
        let (body, sum) = line.rsplit_once(" sum=")?;
        if fnv(&[body.as_bytes()]) != sum {
            return None;
        }
        let mut fields = body.split(' ').map(|f| f.split_once('='));
        let mut next = |name: &str| match fields.next() {
            Some(Some((k, v))) if k == name => Some(v.to_string()),
            _ => None,
        };
        let state = BuildState::from_word(&next("state")?)?;
        let engine = next("engine")?;
        let base = u64::from_str_radix(&next("base")?, 16).ok()?;
        let boot = next("boot")?;
        let attempts = next("attempts")?.parse().ok()?;
        let time = next("time")?.parse().ok()?;
        let pid = next("pid")?.parse().ok()?;
        let blocks = next("blocks")?.parse().ok()?;
        Some(BuildRecord { state, engine, base, boot, attempts, time, pid, blocks })
    }
}

/// Who is asking whether an image is due: this engine, in this boot, now.
#[derive(Clone, Debug)]
struct Asker {
    engine: String,
    boot: String,
    now: u64,
}

impl Asker {
    fn now() -> Self {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        #[cfg(test)]
        if let Ok(boot) = std::env::var("MCL_TEST_BOOT") {
            return Asker { engine: engine_id(), boot, now };
        }
        Asker { engine: engine_id(), boot: crate::cache::boot_session_uuid().unwrap_or_default(), now }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Due {
    /// Build it; `attempts` builds of it by this engine began before and died.
    Yes { attempts: u32 },
    No(&'static str),
}

/// Whether the image at `base` is due a build, from its record (read under its lock, or as a hint before forking)
/// and whether a translation of this format for this base is in the store (`aot_loadable`). The caller already knows
/// that this run did not load one.
fn build_due(rec: Option<&BuildRecord>, me: &Asker, base: u64, loadable: bool) -> Due {
    let Some(r) = rec else { return Due::Yes { attempts: 0 } };
    let later_boot = !me.boot.is_empty() && !r.boot.is_empty() && r.boot != me.boot;
    let retry_due = later_boot || me.now >= r.time.saturating_add(AOT_RETRY_S);
    match r.state {
        // Its file is gone or cannot be loaded: the store was cleared, a run of the image at another base retired it
        // (a profile written for that base retires the other bases' entries), another engine's format replaced it, or
        // it was torn. Built again, but not at once: an image that alternates between two bases, or between two
        // engines' formats, would otherwise be built again after every run of the other.
        BuildState::Built if r.base == base => {
            if loadable {
                Due::No("built")
            } else if retry_due {
                Due::Yes { attempts: 0 }
            } else {
                Due::No("built, and its file gone or unloadable since, within the hour")
            }
        }
        // It moved. In the same boot it is sliding from exec to exec, and a translation for this base would be
        // used by no later run; in a later boot this is where it lives now.
        // An hour later too: a record written when the boot could not be read must not keep a moved image unbuilt.
        BuildState::Built => {
            if retry_due {
                Due::Yes { attempts: 0 }
            } else {
                Due::No("built at another base within the hour")
            }
        }
        // Another engine tries a failure again, but not at once: two engines of one format taking turns in one store
        // would otherwise each build the failing image after every run of the other.
        BuildState::Failed => {
            if r.engine == me.engine {
                Due::No("its build failed with this engine")
            } else if retry_due {
                Due::Yes { attempts: 0 }
            } else {
                Due::No("its build failed with another engine within the hour")
            }
        }
        BuildState::Deferred => {
            if retry_due {
                Due::Yes { attempts: 0 }
            } else {
                Due::No("its build was deferred within the hour")
            }
        }
        // Read with its lock free: the builder that wrote it is gone, killed or ended by its deadline, or by a reboot.
        // After AOT_MAX_ATTEMPTS in one boot it is left until a later boot or another engine: a builder that dies
        // because the machine is too busy may get through in another boot, and one that dies of the image itself
        // dies AOT_MAX_ATTEMPTS times a boot at most.
        BuildState::Building => {
            if loadable && r.base == base {
                // It published before it died, between the rename and the record.
                Due::No("built")
            } else if later_boot {
                Due::Yes { attempts: 0 }
            } else if r.engine != me.engine {
                if retry_due {
                    Due::Yes { attempts: 0 }
                } else {
                    Due::No("another engine's builder died within the hour")
                }
            } else if r.attempts >= AOT_MAX_ATTEMPTS {
                Due::No("its builder died every time")
            } else {
                Due::Yes { attempts: r.attempts }
            }
        }
    }
}

/// The translation format: what decides whether a file can be loaded (`cache_layout`).
fn format_tag() -> String {
    format!("v{TRANSLATOR_VERSION}-{:x}", flags_word())
}

/// The engine, for a build record: the format and this binary's LC_UUID, so a new build of the engine tries again
/// what an older one could not build.
fn engine_id() -> String {
    #[cfg(test)]
    if let Ok(id) = std::env::var("MCL_TEST_ENGINE_ID") {
        return id;
    }
    format!("{}-{}", format_tag(), own_uuid().map(|u| hex(&u)).unwrap_or_else(|| "nouuid".into()))
}

/// This binary's LC_UUID, read from its own Mach-O header in memory.
pub(crate) fn own_uuid() -> Option<[u8; 16]> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(own_uuid as *const libc::c_void, &mut info) } == 0 || info.dli_fbase.is_null() {
        return None;
    }
    let base = info.dli_fbase as u64;
    let rd = |o: u64| unsafe { std::ptr::read_unaligned((base + o) as *const u32) };
    // mach_header_64: magic, cputype, cpusubtype, filetype, ncmds at +16, sizeofcmds at +20; commands from +32.
    if rd(0) != 0xfeed_facf {
        return None;
    }
    let (ncmds, end) = (rd(16), 32 + rd(20) as u64);
    let mut off = 32u64;
    for _ in 0..ncmds.min(4096) {
        let (cmd, size) = (rd(off), rd(off + 4) as u64);
        if size < 8 || off + size > end {
            return None;
        }
        if cmd == 0x1b && size >= 24 {
            let mut uuid = [0u8; 16];
            uuid.copy_from_slice(unsafe { std::slice::from_raw_parts((base + off + 8) as *const u8, 16) });
            return Some(uuid);
        }
        off += size;
    }
    None
}

fn record_path(dir: &std::path::Path, img: &Image) -> std::path::PathBuf {
    let identity = identity_prefix(&img.key).unwrap_or(&img.key);
    dir.join(format!("{identity}.{}.build", format_tag()))
}

/// Open a file of the store's bookkeeping (a build record or a lock), made 0600 if `create`; refused unless it is a
/// regular file of ours that no one else can write, reached without a symlink.
fn open_store_file(path: &std::path::Path, create: bool) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let md = f.metadata()?;
    if !(md.is_file() && owned(&md, unsafe { libc::geteuid() })) {
        return Err(std::io::Error::other("not a file of ours alone"));
    }
    Ok(f)
}

fn whole_file_lock(kind: libc::c_short) -> libc::flock {
    let mut l: libc::flock = unsafe { std::mem::zeroed() };
    l.l_type = kind;
    l.l_whence = libc::SEEK_SET as libc::c_short;
    l
}

/// Take the write lock on all of `f` if no other process holds a lock on it. Locks are this process's: closing any
/// descriptor of the file in this process drops them, so a file locked here is opened nowhere else in the process.
fn try_lock(f: &std::fs::File) -> bool {
    use std::os::unix::io::AsRawFd;
    let mut l = whole_file_lock(libc::F_WRLCK as libc::c_short);
    unsafe { libc::fcntl(f.as_raw_fd(), libc::F_SETLK, &mut l) == 0 }
}

/// Whether another process holds a lock on `f`. When the kernel cannot say, the answer is yes: the caller then leaves
/// the file alone, which costs at most a build this exit does not ask for.
fn locked_elsewhere(f: &std::fs::File) -> bool {
    use std::os::unix::io::AsRawFd;
    let mut l = whole_file_lock(libc::F_WRLCK as libc::c_short);
    let r = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_GETLK, &mut l) };
    r != 0 || l.l_type != libc::F_UNLCK as libc::c_short
}

fn read_record(f: &std::fs::File) -> Option<BuildRecord> {
    use std::os::unix::fs::FileExt;
    let mut buf = [0u8; 512];
    let n = f.read_at(&mut buf, 0).ok()?;
    BuildRecord::parse(&buf[..n])
}

/// Replace the record in `f` (whose lock this process holds): one write of a line under 512 bytes, then the file cut
/// to it.
fn write_record(f: &std::fs::File, r: &BuildRecord) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    let line = r.line();
    f.write_all_at(line.as_bytes(), 0)?;
    f.set_len(line.len() as u64)
}

/// Whether the store holds a translation this engine would load for `img`: every check `load_file` makes before it
/// copies the code (`validated`). Not a file of another engine's format, a torn or truncated one, one whose index is
/// damaged, or one not ours alone.
fn aot_loadable(img: &Image) -> bool {
    validated(&file_for(img, "aot"), img.start, img.end).is_ok_and(|v| v.is_some())
}

/// Before forking: whether `img` is worth a builder (its record says it is due, and no builder holds it).
fn wants_build(dir: &std::path::Path, img: &Image, me: &Asker) -> bool {
    let why = match open_store_file(&record_path(dir, img), false) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return true,
        Err(_) => "its build record is not a file of ours",
        Ok(f) if locked_elsewhere(&f) => "a builder holds it",
        Ok(f) => match build_due(read_record(&f).as_ref(), me, img.start, aot_loadable(img)) {
            Due::Yes { .. } => return true,
            Due::No(why) => why,
        },
    };
    trace_build(format_args!("{}: not building ({why})", img.name));
    false
}

/// Before forking: whether a builder is building and every place in the queue is taken.
fn queue_full(dir: &std::path::Path) -> bool {
    let held = |name: &str| open_store_file(&dir.join(name), false).is_ok_and(|f| locked_elsewhere(&f));
    held("aot-builder.lock") && (0..AOT_QUEUE).all(|i| held(&format!("aot-queue-{i}.lock")))
}

/// Whether the store's volume has AOT_MIN_FREE free. When it cannot be read, yes: the build then fails or not on
/// its own, and a failure for want of space is recorded as such.
fn has_room(dir: &std::path::Path) -> bool {
    #[cfg(test)]
    if let Some(free) = std::env::var("MCL_TEST_FREE_BYTES").ok().and_then(|v| v.parse::<u64>().ok()) {
        return free >= AOT_MIN_FREE;
    }
    // A test's store has no room while the file it names exists.
    #[cfg(test)]
    if let Some(p) = std::env::var_os("MCL_TEST_NO_ROOM_IF") {
        return !std::path::Path::new(&p).exists();
    }
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else { return true };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return true;
    }
    // ST_RDONLY (1): a store on a read-only volume has no room, so no exit forks a builder that can record nothing.
    st.f_flag & 1 == 0 && (st.f_bavail as u64).saturating_mul(st.f_frsize as u64) >= AOT_MIN_FREE
}

fn trace_build(what: std::fmt::Arguments) {
    if std::env::var_os("MACLATOR_TRACE").is_some() || std::env::var_os("MACLATOR_STATS").is_some() {
        eprintln!("[maclator] aot: {what}");
    }
}

/// The builder process, once forked: in its own session at background priority (CPU and I/O); every signal it can
/// take at its default action and unblocked, so it ends when it is told to (it runs no guest code, so the guest's
/// handlers, which the fork kept, would only swallow a SIGTERM or a deadline's signal), but SIGUSR1/SIGUSR2, which are
/// the translator's own dumps; no core file; nothing the program had open but stdio and the dump pipe (a builder that
/// kept them would hold the program's lock files and sockets while it waits and builds, and a restarted program could
/// not take them back); and an alarm that ends it if it has not got its turn and finished by then.
unsafe fn become_background_builder() {
    for sig in 1..32 {
        if !matches!(sig, libc::SIGKILL | libc::SIGSTOP | libc::SIGUSR1 | libc::SIGUSR2) {
            libc::signal(sig, libc::SIG_DFL);
        }
    }
    let mut none: libc::sigset_t = std::mem::zeroed();
    libc::sigemptyset(&mut none);
    libc::sigprocmask(libc::SIG_SETMASK, &none, std::ptr::null_mut());
    libc::setpriority(libc::PRIO_DARWIN_PROCESS, 0, libc::PRIO_DARWIN_BG);
    let no_core = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    libc::setrlimit(libc::RLIMIT_CORE, &no_core);
    // Room for a descriptor per claimed image and the locks, whatever soft limit the program ran with.
    let mut files: libc::rlimit = std::mem::zeroed();
    if libc::getrlimit(libc::RLIMIT_NOFILE, &mut files) == 0 && files.rlim_cur < 1024 {
        files.rlim_cur = files.rlim_max.min(1024);
        libc::setrlimit(libc::RLIMIT_NOFILE, &files);
    }
    libc::alarm(AOT_QUEUE_WAIT.as_secs() as libc::c_uint + AOT_BUILD_WALL_S);
}

/// Close every descriptor above stdio but those in `keep`: those /dev/fd lists (the listing's own descriptor is closed
/// before the others are), or, when no descriptor is free to list them with, every number up to the soft limit.
fn close_inherited(keep: &[i32]) {
    let fds: Vec<i32> = match std::fs::read_dir("/dev/fd") {
        Ok(rd) => rd.flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).collect(),
        Err(_) => {
            let mut l: libc::rlimit = unsafe { std::mem::zeroed() };
            let top = if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut l) } == 0 { l.rlim_cur.min(1 << 16) as i32 } else { 1 << 16 };
            (3..top).collect()
        }
    };
    for fd in fds {
        if fd > 2 && !keep.contains(&fd) {
            unsafe { libc::close(fd) };
        }
    }
}

/// Once the builder has its turn: at most AOT_BUILD_CPU_S more of CPU (SIGXCPU, at its default action, ends it) and
/// AOT_BUILD_WALL_S of time (SIGALRM).
unsafe fn set_build_deadline() {
    let mut used: libc::rusage = std::mem::zeroed();
    libc::getrusage(libc::RUSAGE_SELF, &mut used);
    let spent = (used.ru_utime.tv_sec + used.ru_stime.tv_sec).max(0) as u64;
    let mut lim: libc::rlimit = std::mem::zeroed();
    libc::getrlimit(libc::RLIMIT_CPU, &mut lim);
    let want = spent.saturating_add(AOT_BUILD_CPU_S) as libc::rlim_t;
    lim.rlim_cur = want.min(lim.rlim_max);
    libc::setrlimit(libc::RLIMIT_CPU, &lim);
    libc::alarm(AOT_BUILD_WALL_S);
}

/// An image this builder holds: the open (and locked) record and the builds begun before.
struct Claim {
    file: std::fs::File,
    attempts: u32,
}

/// Take `img`'s record and decide under its lock. None: another builder holds it, or it is not due.
fn claim(dir: &std::path::Path, img: &Image, me: &Asker) -> Option<Claim> {
    let f = open_store_file(&record_path(dir, img), true).ok()?;
    if !try_lock(&f) {
        trace_build(format_args!("{}: not building (a builder holds it)", img.name));
        return None;
    }
    let rec = read_record(&f);
    match build_due(rec.as_ref(), me, img.start, aot_loadable(img)) {
        Due::Yes { attempts } => Some(Claim { file: f, attempts }),
        Due::No(why) => {
            trace_build(format_args!("{}: not building ({why})", img.name));
            None
        }
    }
}

enum Turn {
    Mine(std::fs::File),
    Dropped(&'static str),
}

/// Wait for this builder's turn: `aot-builder.lock`, at once, or from a place in the queue for at most `wait`.
fn wait_turn(dir: &std::path::Path, wait: std::time::Duration) -> Turn {
    let Ok(lock) = open_store_file(&dir.join("aot-builder.lock"), true) else { return Turn::Dropped("the builder lock is not a file of ours") };
    if try_lock(&lock) {
        return Turn::Mine(lock);
    }
    let place = (0..AOT_QUEUE).find_map(|i| open_store_file(&dir.join(format!("aot-queue-{i}.lock")), true).ok().filter(try_lock));
    let Some(place) = place else { return Turn::Dropped("the queue is full") };
    let end = std::time::Instant::now() + wait;
    while std::time::Instant::now() < end {
        std::thread::sleep(std::time::Duration::from_millis(200));
        if try_lock(&lock) {
            drop(place);
            return Turn::Mine(lock);
        }
    }
    Turn::Dropped("its turn did not come")
}

/// A builder's temporary file: `<key>.aot.<pid>.tmp`, or `<key>.aot.tmp` from before upstream #7.
fn is_builder_tmp(name: &str) -> bool {
    let Some(s) = name.strip_suffix(".tmp") else { return false };
    if s.ends_with(".aot") {
        return true;
    }
    s.rsplit_once('.').is_some_and(|(stem, pid)| stem.ends_with(".aot") && !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()))
}

/// Remove the temporary files of builders that died: older than AOT_TMP_STALE_S (no live builder's is that old). How
/// many went.
fn sweep_stale_tmps(dir: &std::path::Path) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    let now = std::time::SystemTime::now();
    let mut gone = 0;
    for ent in rd.flatten() {
        let name = ent.file_name();
        let Some(name) = name.to_str() else { continue };
        if !is_builder_tmp(name) {
            continue;
        }
        let age = ent.metadata().and_then(|m| m.modified()).ok().and_then(|t| now.duration_since(t).ok());
        if age.is_some_and(|a| a.as_secs() >= AOT_TMP_STALE_S) && std::fs::remove_file(ent.path()).is_ok() {
            trace_build(format_args!("removed {name}, left by a builder that died"));
            gone += 1;
        }
    }
    gone
}

/// A failure that passes: no space, no descriptors, no memory, interrupted, a file gone. Such a build is deferred, not
/// failed.
fn is_transient(e: &std::io::Error) -> bool {
    // ENOENT: the temporary file was removed before its rename (the store cleared under the builder).
    matches!(e.raw_os_error(), Some(libc::ENOSPC | libc::EDQUOT | libc::EMFILE | libc::ENFILE | libc::ENOMEM | libc::EAGAIN | libc::EINTR | libc::ENOENT))
}

/// Build the images this builder can claim, in its turn. None: it claimed none, or its turn did not come.
fn build_queued(dir: &std::path::Path, rebuild: Vec<(Image, BTreeSet<u64>)>, me: &Asker, sync: bool) -> Option<(usize, usize, usize, usize)> {
    // One claim per record: two images of one identity (at two bases) would lock one file twice, and closing either
    // descriptor would drop the lock both rely on.
    let mut seen = HashSet::new();
    let read = crate::jit::guest_reader();
    let claims: Vec<_> = rebuild
        .into_iter()
        // Before the de-duplication: a stale registration of a library must not stand in for a live copy of it.
        .filter(|(img, _)| still_there(img, &read))
        .filter(|(img, _)| seen.insert(record_path(dir, img)))
        .filter_map(|(img, prof)| claim(dir, &img, me).map(|c| (c, img, prof)))
        .collect();
    if claims.is_empty() {
        return None;
    }
    let lock = match wait_turn(dir, AOT_QUEUE_WAIT) {
        Turn::Mine(lock) => lock,
        Turn::Dropped(why) => {
            trace_build(format_args!("not building ({why})"));
            return None;
        }
    };
    if !sync {
        unsafe { set_build_deadline() };
    }
    sweep_stale_tmps(dir);
    let (mut ok, mut skipped, mut failed) = (0usize, 0usize, 0usize);
    let images = claims.len();
    for (c, img, prof) in claims {
        let rec = |state| BuildRecord {
            state,
            engine: me.engine.clone(),
            base: img.start,
            boot: me.boot.clone(),
            attempts: c.attempts + 1,
            time: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            pid: std::process::id(),
            blocks: prof.len(),
        };
        // Asked again for each image: the one before may have taken the room.
        if !has_room(dir) {
            let _ = write_record(&c.file, &rec(BuildState::Deferred));
            failed += 1;
            continue;
        }
        // A build it cannot record is not begun: a store that cannot take a line cannot take a translation, and
        // building without a record is building again at every exit.
        if write_record(&c.file, &rec(BuildState::Building)).is_err() {
            failed += 1;
            continue;
        }
        let state = match build(&img, &prof) {
            Ok(true) => {
                ok += 1;
                BuildState::Built
            }
            // Nothing could be read to translate, or the library is no longer mapped where it was registered: maybe
            // in another run, so deferred, not failed.
            Ok(false) => {
                skipped += 1;
                BuildState::Deferred
            }
            Err(e) => {
                failed += 1;
                if is_transient(&e) {
                    BuildState::Deferred
                } else {
                    BuildState::Failed
                }
            }
        };
        let _ = write_record(&c.file, &rec(state));
        trace_build(format_args!("{}: {}", img.name, state.word()));
        // Dropping the claim closes the record and frees the image.
    }
    drop(lock);
    Some((images, ok, skipped, failed))
}

/// The temporary file a builder writes its done line to, beside `done`. Builders inheriting one
/// MACLATOR_AOT_DONE_FILE each need their own, or one could rename another's half-written line into place.
fn done_tmp(done: &std::path::Path, pid: u32) -> std::path::PathBuf {
    let mut t = done.as_os_str().to_owned();
    t.push(format!(".{pid}.tmp"));
    t.into()
}

/// Write `line` to `done` whole: to this builder's temporary file, then by rename.
fn write_done(done: &std::path::Path, pid: u32, line: &str) {
    let tmp = done_tmp(done, pid);
    if std::fs::write(&tmp, line).is_err() || std::fs::rename(&tmp, done).is_err() {
        let _ = std::fs::remove_file(&tmp);
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

/// Translate `img` and publish its AOT file. `Ok(false)`: no block could be read, so nothing was published.
fn build(img: &Image, profile: &BTreeSet<u64>) -> std::io::Result<bool> {
    #[cfg(test)]
    once_tests::build_hook(img, "start");
    let r = build_image(img, profile);
    #[cfg(test)]
    once_tests::build_hook(img, if r.is_ok() { "end" } else { "fail" });
    r
}

fn build_image(img: &Image, profile: &BTreeSet<u64>) -> std::io::Result<bool> {
    let read = crate::jit::guest_reader();
    if !still_there(img, &read) {
        return Ok(false);
    }
    let mut blocks: BTreeSet<u64> = profile.clone();
    if !img.entries.is_empty() {
        blocks.extend(discover(&img.entries, img.start, img.end, &read, 400_000));
    }
    // Profile-guided expansion: also follow direct branches from hot blocks.
    let hot: Vec<u64> = profile.iter().copied().collect();
    blocks.extend(discover(&hot, img.start, img.end, &read, profile.len() * 4 + 1000));
    let blocks: Vec<u64> = blocks.into_iter().filter(|&pc| read(pc).is_some()).collect();
    if blocks.is_empty() {
        return Ok(false);
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
    let published = create_private(&tmp).and_then(|mut f| {
        f.write_all(&hdr)?;
        f.write_all(&code)?;
        drop(f);
        std::fs::rename(&tmp, &path)
    });
    // A file that was not published is removed: nothing else ever removes a temporary file (retirement leaves every
    // write in flight alone), so one left by a failed write or rename stayed in the store for good, and on a full
    // volume it was often a large part of a translation.
    if let Err(e) = published {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // This image is now cached for the base it is loaded at; anything cached for a base it will never be at
    // again is dead weight. Without this the store grows by one entry per slid exec, for ever.
    retire_superseded(&cache_dir(), &img.key, "build");
    Ok(true)
}

/// Whether a library is still the one its key names. It was registered when dyld mapped it (`register_image_at`), and
/// it may have been unmapped since and another mapped at its addresses; its code must not be published under the other's
/// UUID. Read from the Mach-O header at its start, where dyld maps a library's __TEXT. Other kinds are not unmapped.
fn still_there(img: &Image, read: &dyn Fn(u64) -> Option<u32>) -> bool {
    let Some(uuid) = img.key.strip_prefix("lib-").and_then(|k| k.split('-').next()) else { return true };
    let at = img.start;
    if read(at) != Some(0xfeed_facf) || read(at + 4) != Some(0x0100_000c) {
        return false;
    }
    let Some(ncmds) = read(at + 16) else { return false };
    let mut off = 32u64;
    for _ in 0..ncmds.min(512) {
        let (Some(cmd), Some(size)) = (read(at + off), read(at + off + 4)) else { return false };
        let size = size as u64;
        if size < 8 || off + size > 0x4000 {
            return false;
        }
        if cmd == 0x1b && size >= 24 {
            let mut bytes = Vec::with_capacity(16);
            for i in 0..4 {
                let Some(w) = read(at + off + 8 + 4 * i) else { return false };
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            return hex(&bytes) == uuid;
        }
        off += size;
    }
    false
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

    /// An image with no block that can be read publishes nothing, and build says so.
    #[test]
    fn a_build_with_nothing_to_translate_reports_nothing_published() {
        // Low guest addresses are never mapped in this process, so no block there can be read.
        let img = Image { name: "x".into(), key: format!("skip-{}", std::process::id()), start: 0x1000, end: 0x2000, entries: vec![], loaded: false, cached: false };
        assert_eq!(build(&img, &BTreeSet::from([0x1000u64])).unwrap(), false, "nothing could be read, so nothing was published");
        assert!(!file_for(&img, "aot").exists(), "and no file was written");
    }

    /// The done line counts an image whose file was published as ok and one with nothing to translate as skipped.
    /// Run in a child: save_profile saves once per process, and reads its settings from the environment.
    #[test]
    fn the_done_line_counts_published_images_ok_and_empty_ones_skipped() {
        if let Some(marker) = std::env::var_os("MCL_AOT_DONE") {
            let code = super::owner_tests::rets();
            let (start, end) = (code.as_ptr() as u64, code.as_ptr() as u64 + 4 * code.len() as u64);
            register_main("/bin/readable", [4u8; 16], start, end, vec![start]);
            // Low guest addresses are never mapped in this process, so no block there can be read.
            register_main("/bin/unreadable", [5u8; 16], 0x1000, 0x2000, vec![0x1000]);
            save_profile();
            let line = std::fs::read_to_string(std::env::var_os("MACLATOR_AOT_DONE_FILE").unwrap()).unwrap();
            assert_eq!(line, format!("aot-build pid={} images=2 ok=1 skipped=1 failed=0\n", std::process::id()));
            std::fs::write(&marker, b"ran").unwrap();
            return;
        }
        let d = std::env::temp_dir().join(format!("maclator-aot-done-line-{}", std::process::id()));
        let cache = d.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::set_permissions(&cache, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
        let marker = d.join("ran");
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "aot::tests::the_done_line_counts_published_images_ok_and_empty_ones_skipped", "--test-threads=1"])
            .env("MCL_AOT_DONE", &marker)
            .env("MACLATOR_CACHE", &cache)
            .env("MACLATOR_AOT_SYNC", "1")
            .env_remove("MACLATOR_NO_AOT_BUILD")
            .env("MACLATOR_AOT_DONE_FILE", d.join("aot-build.done"))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        let ran = std::fs::read(&marker).ok();
        std::fs::remove_dir_all(&d).ok();
        assert!(st.success(), "the child: {st}");
        assert_eq!(ran.as_deref(), Some(&b"ran"[..]), "the child ran its checks");
    }

    /// Builders sharing one done file write it through temporary files of their own, beside it.
    #[test]
    fn each_builder_writes_its_done_line_through_its_own_temporary_file() {
        let d = std::env::temp_dir().join(format!("maclator-aot-done-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let done = d.join("aot-build.done");
        assert_ne!(done_tmp(&done, 1), done_tmp(&done, 2), "two builders, one temporary file");
        assert_eq!(done_tmp(&done, 1).parent(), Some(d.as_path()), "the temporary file is beside the done file");
        assert_ne!(done_tmp(&done, 1), done, "and is not the done file itself");
        write_done(&done, 7, "aot-build pid=7 images=1 ok=1 skipped=0 failed=0\n");
        assert_eq!(std::fs::read_to_string(&done).unwrap(), "aot-build pid=7 images=1 ok=1 skipped=0 failed=0\n");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1, "the temporary file was renamed into place");
        std::fs::remove_dir_all(&d).ok();
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
            assert!(build(&img, &BTreeSet::from([start])).unwrap(), "build published a file");
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

    /// A build whose file cannot be published leaves no temporary file in the store. Run in a child, as the store is
    /// read from the environment.
    #[test]
    fn a_build_that_cannot_publish_leaves_no_temporary_file() {
        if let Some(marker) = std::env::var_os("MCL_AOT_TMP_LEFT") {
            let code = rets();
            let (start, end) = (code.as_ptr() as u64, code.as_ptr() as u64 + 4 * code.len() as u64);
            let img = Image { name: "t".into(), key: "tmp-left".into(), start, end, entries: vec![], loaded: false, cached: false };
            let d = cache_dir();
            // The published file's name is taken by a directory, so the rename that would publish it fails.
            std::fs::create_dir_all(d.join("tmp-left.aot").join("x")).unwrap();
            assert!(build(&img, &BTreeSet::from([start])).is_err(), "the rename onto a directory fails");
            let left: Vec<String> = std::fs::read_dir(&d)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".tmp"))
                .collect();
            assert!(left.is_empty(), "temporary files left in the store: {left:?}");
            std::fs::write(&marker, b"ran").unwrap();
            return;
        }
        let d = std::env::temp_dir().join(format!("mcl-aot-tmp-left-{}", std::process::id()));
        let cache = d.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        chmod(&cache, 0o700);
        let marker = d.join("ran");
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "aot::owner_tests::a_build_that_cannot_publish_leaves_no_temporary_file", "--test-threads=1", "--nocapture"])
            .env("MCL_AOT_TMP_LEFT", &marker)
            .env("MACLATOR_CACHE", &cache)
            .output()
            .unwrap();
        let ran = std::fs::read(&marker).ok();
        std::fs::remove_dir_all(&d).ok();
        assert!(out.status.success(), "the child: {}: {}", out.status, String::from_utf8_lossy(&out.stderr));
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

    /// The same run twice, each in a child: unsandboxed it writes the image's profile (the control); under a sandbox
    /// that allows everything it writes nothing, so what is tested is the translator's own choice, not a refusal.
    #[test]
    fn a_sandboxed_process_saves_no_translations() {
        // In a process of its own (the test binary run again for this test alone): save_profile saves once per process
        // and reads the store from the environment, which a fork child of the threaded test harness must not change
        // (a parent thread reading it as the fork came leaves its lock held in the child).
        if let Some(sbx) = std::env::var_os("MCL_AOT_SBX") {
            let d = std::path::PathBuf::from(std::env::var_os("MACLATOR_CACHE").unwrap());
            ENABLED.store(true, Ordering::SeqCst);
            SAVED.store(false, Ordering::SeqCst);
            if sbx == "1" && !crate::sandboxcheck::tests::apply("(version 2)(allow default)") {
                std::process::exit(20);
            }
            register_main("/x/mcl-sbx-save", [7; 16], 0x1_0000_0000, 0x1_0000_4000, vec![]);
            note_translated(0x1_0000_0000);
            save_profile();
            std::process::exit(std::fs::read_dir(&d).map(|r| r.count()).unwrap_or(0).min(19) as i32);
        }
        let run = |sandbox: bool| {
            // The store exists (0700, ours), so making it is not what keeps a sandboxed process out.
            let d = std::env::temp_dir().join(format!("mcl-sbx-save-{sandbox}-{}", std::process::id()));
            assert!(owned_dir(&d, unsafe { libc::geteuid() }), "control: the store is made");
            let st = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "aot::image_key_tests::a_sandboxed_process_saves_no_translations", "--test-threads=1"])
                .env("MCL_AOT_SBX", if sandbox { "1" } else { "0" })
                .env("MACLATOR_CACHE", &d)
                .env("MACLATOR_NO_AOT_BUILD", "1")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap();
            let _ = std::fs::remove_dir_all(&d);
            st.code().unwrap_or(1000)
        };
        let control = run(false);
        assert!((1..20).contains(&control), "control: unsandboxed, the profile is written (got {control})");
        assert_eq!(run(true), 0, "sandboxed: nothing is written (20: the sandbox was not applied)");
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
        // Two fields only: an older build's cache key, which names no base and is never retired.
        assert_eq!(identity_prefix(&format!("dsc-{u}")), None);
        // An older build's main key splits on its fnv, into a prefix no current key has.
        let old_main = format!("main-{u}-aabbccddeeff0011");
        assert_eq!(identity_prefix(&old_main), Some(format!("main-{u}").as_str()));
        assert_ne!(identity_prefix(&old_main), identity_prefix(&format!("{old_main}-500000000")), "so a current key never retires it");
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

    /// A key whose files could not be removed is not counted as retired.
    #[test]
    fn only_entries_really_removed_are_counted() {
        use std::os::unix::fs::PermissionsExt;
        if unsafe { libc::geteuid() } == 0 {
            eprintln!("skipped: root may unlink in a read-only directory");
            return;
        }
        let d = tmpdir("unlink");
        let ident = "main-0123456789abcdef0123456789abcdef-aabbccddeeff0011";
        put(&d, &format!("{ident}-500000000"), 16);
        let keep = format!("{ident}-500004000");
        put(&d, &keep, 16);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o555)).unwrap();
        let n = retire_superseded(&d, &keep, "test");
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(n, 0, "nothing could be removed, so nothing was retired");
        assert_eq!(count(&d), 4, "and every file is still there");
        assert_eq!(retire_superseded(&d, &keep, "test"), 1, "control: in a writable store it is retired");
        assert_eq!(count(&d), 2);
        // One of the pair already gone still retires the key: the other file was removed.
        put(&d, &format!("{ident}-500008000"), 16);
        std::fs::remove_file(d.join(format!("{ident}-500008000.profile"))).unwrap();
        assert_eq!(retire_superseded(&d, &keep, "test"), 1, "a key with one file left is retired by removing it");
        // One file removed and the other refused: the key is not retired, since its .aot is still there.
        let stuck = format!("{ident}-50000c000");
        std::fs::create_dir(d.join(format!("{stuck}.aot"))).unwrap();
        std::fs::write(d.join(format!("{stuck}.aot/x")), b"x").unwrap();
        std::fs::write(d.join(format!("{stuck}.profile")), [0u8; 8]).unwrap();
        assert_eq!(retire_superseded(&d, &keep, "test"), 0, "an .aot that could not be removed leaves the key unretired");
        assert!(d.join(format!("{stuck}.aot")).exists() && !d.join(format!("{stuck}.profile")).exists(), "control: only the profile went");
        std::fs::remove_dir_all(&d).ok();
    }

    /// When a new profile cannot be written, save_profile retires nothing, so with AOT building off the image's entry
    /// at its old base is kept. Run in a child, since save_profile saves once per process and reads the cache
    /// directory from the environment.
    #[test]
    fn a_profile_that_could_not_be_written_retires_nothing() {
        let ident_of = |start: u64| image_key("main", &[3u8; 16], "/bin/x", start);
        let (old, new) = (ident_of(0x5_0000_0000), ident_of(0x5_0000_4000));
        if std::env::var_os("MCL_AOT_PROFILE").is_some() {
            let d = std::path::PathBuf::from(std::env::var_os("MACLATOR_CACHE").unwrap());
            register_main("/bin/x", [3u8; 16], 0x5_0000_4000, 0x5_0000_8000, vec![0x5_0000_4000]);
            save_profile();
            assert!(d.join(format!("{old}.aot")).exists() && d.join(format!("{old}.profile")).exists(), "the old entry is kept");
            std::fs::write(std::env::var_os("MCL_AOT_PROFILE").unwrap(), b"ran").unwrap();
            return;
        }
        let run = |d: &std::path::Path| {
            let marker = d.join("ran");
            let st = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "aot::retire_tests::a_profile_that_could_not_be_written_retires_nothing", "--test-threads=1"])
                .env("MCL_AOT_PROFILE", &marker)
                .env("MACLATOR_CACHE", d)
                .env("MACLATOR_NO_AOT_BUILD", "1")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap();
            (st.success(), std::fs::read(&marker).ok().as_deref() == Some(&b"ran"[..]))
        };
        use std::os::unix::fs::PermissionsExt;
        // The new profile's path is a directory, so its write fails while the store itself stays writable.
        let d = tmpdir("profile");
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700)).unwrap();
        put(&d, &old, 16);
        std::fs::create_dir(d.join(format!("{new}.profile"))).unwrap();
        assert_eq!(run(&d), (true, true), "the child kept the old entry and ran its checks");
        // Control: with the new profile writable, the same run retires the old entry.
        let c = tmpdir("profile-control");
        std::fs::set_permissions(&c, std::fs::Permissions::from_mode(0o700)).unwrap();
        put(&c, &old, 16);
        assert_eq!(run(&c).0, false, "control: a written profile retires the old entry, so the child's check fails");
        assert!(c.join(format!("{new}.profile")).exists() && !c.join(format!("{old}.aot")).exists(), "control: retired");
        std::fs::remove_dir_all(&d).ok();
        std::fs::remove_dir_all(&c).ok();
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

#[cfg(test)]
#[path = "aot_once_tests.rs"]
mod once_tests;

#[cfg(test)]
mod build_record_tests {
    use super::*;

    fn rec(state: BuildState) -> BuildRecord {
        BuildRecord { state, engine: "e1".into(), base: 0x5_0000_0000, boot: "B1".into(), attempts: 1, time: 1000, pid: 7, blocks: 3 }
    }

    fn me() -> Asker {
        Asker { engine: "e1".into(), boot: "B1".into(), now: 1000 }
    }

    #[test]
    fn a_record_reads_back_and_a_torn_one_reads_as_none() {
        let r = rec(BuildState::Building);
        let line = r.line();
        assert!(line.len() < 512 && line.ends_with('\n'), "one short line: {line:?}");
        assert_eq!(BuildRecord::parse(line.as_bytes()), Some(r.clone()));
        // Bytes of an older, longer record after the line (a write cut off before the file was) do not count.
        let mut longer = line.clone().into_bytes();
        longer.extend_from_slice(b"state=built engine=x");
        assert_eq!(BuildRecord::parse(&longer), Some(r.clone()), "what follows the line is not read");
        for cut in [0, 1, line.len() / 2, line.len() - 1] {
            assert_eq!(BuildRecord::parse(&line.as_bytes()[..cut]), None, "a record cut at {cut} bytes");
        }
        // One byte changed in the body: the checksum no longer matches.
        let mut flipped = line.clone().into_bytes();
        let at = line.find("attempts=").unwrap() + "attempts=".len();
        flipped[at] = b'2';
        assert_eq!(BuildRecord::parse(&flipped), None, "a changed byte");
        // An older record's tail under a newer, shorter head (state=built over state=building): the checksum says no.
        let mut mixed = rec(BuildState::Built).line().into_bytes();
        mixed.truncate(20);
        mixed.extend_from_slice(&line.as_bytes()[20..]);
        assert_eq!(BuildRecord::parse(&mixed), None, "a head and a tail of two records");
        assert_eq!(BuildRecord::parse(b"garbage\n"), None);
    }

    #[test]
    fn whether_an_image_is_due() {
        let m = me();
        let at = |r: BuildRecord, f: &dyn Fn(&mut BuildRecord)| {
            let mut r = r;
            f(&mut r);
            r
        };
        let base = 0x5_0000_0000;
        assert_eq!(build_due(None, &m, base, false), Due::Yes { attempts: 0 }, "no record");
        assert_eq!(build_due(Some(&rec(BuildState::Built)), &m, base, true), Due::No("built"));
        assert_eq!(build_due(Some(&rec(BuildState::Built)), &m, base, false), Due::No("built, and its file gone or unloadable since, within the hour"));
        let hour = Asker { now: 1000 + AOT_RETRY_S, ..me() };
        assert_eq!(build_due(Some(&rec(BuildState::Built)), &hour, base, false), Due::Yes { attempts: 0 }, "built, and its file gone or unloadable an hour ago");
        let next = Asker { boot: "B2".into(), ..me() };
        assert_eq!(build_due(Some(&rec(BuildState::Built)), &next, base, false), Due::Yes { attempts: 0 }, "built, and its file gone in a later boot");
        let moved = rec(BuildState::Built);
        assert_eq!(build_due(Some(&moved), &m, base + 0x4000, false), Due::No("built at another base within the hour"));
        let earlier = at(rec(BuildState::Built), &|r| r.boot = "B0".into());
        assert_eq!(build_due(Some(&earlier), &m, base + 0x4000, false), Due::Yes { attempts: 0 }, "moved, in a later boot");
        let unknown = Asker { boot: String::new(), ..me() };
        assert_eq!(build_due(Some(&earlier), &unknown, base + 0x4000, false), Due::No("built at another base within the hour"), "boot unknown: not taken for a later one");
        let unknown_later = Asker { boot: String::new(), now: 1000 + AOT_RETRY_S, ..me() };
        let unrecorded = at(rec(BuildState::Built), &|r| r.boot = String::new());
        assert_eq!(build_due(Some(&unrecorded), &unknown_later, base + 0x4000, false), Due::Yes { attempts: 0 }, "moved, the boot unknown when it was built, an hour later");
        assert_eq!(build_due(Some(&rec(BuildState::Failed)), &m, base, false), Due::No("its build failed with this engine"));
        assert_eq!(build_due(Some(&rec(BuildState::Failed)), &m, base + 0x4000, false), Due::No("its build failed with this engine"), "a failure is the image's, wherever it is");
        let other = at(rec(BuildState::Failed), &|r| r.engine = "e0".into());
        assert_eq!(build_due(Some(&other), &m, base, false), Due::No("its build failed with another engine within the hour"));
        let later = Asker { now: 1000 + AOT_RETRY_S, ..me() };
        assert_eq!(build_due(Some(&other), &later, base, false), Due::Yes { attempts: 0 }, "failed with another engine an hour ago");
        let next_boot = Asker { boot: "B2".into(), ..me() };
        assert_eq!(build_due(Some(&other), &next_boot, base, false), Due::Yes { attempts: 0 }, "failed with another engine in an earlier boot");
        assert_eq!(build_due(Some(&rec(BuildState::Deferred)), &m, base, false), Due::No("its build was deferred within the hour"));
        let by_other_deferred = at(rec(BuildState::Deferred), &|r| r.engine = "e0".into());
        assert_eq!(build_due(Some(&by_other_deferred), &m, base, false), Due::No("its build was deferred within the hour"), "whichever engine deferred it");
        assert_eq!(build_due(Some(&rec(BuildState::Deferred)), &later, base, false), Due::Yes { attempts: 0 }, "deferred an hour ago");
        assert_eq!(build_due(Some(&rec(BuildState::Deferred)), &next_boot, base, false), Due::Yes { attempts: 0 }, "deferred in an earlier boot");
        assert_eq!(build_due(Some(&rec(BuildState::Building)), &m, base, false), Due::Yes { attempts: 1 }, "its builder died once");
        assert_eq!(build_due(Some(&rec(BuildState::Building)), &m, base, true), Due::No("built"), "its builder died after it published");
        let twice = at(rec(BuildState::Building), &|r| r.attempts = AOT_MAX_ATTEMPTS);
        assert_eq!(build_due(Some(&twice), &m, base, false), Due::No("its builder died every time"));
        assert_eq!(build_due(Some(&twice), &next_boot, base, false), Due::Yes { attempts: 0 }, "cut off by a reboot");
        let by_other = at(twice.clone(), &|r| r.engine = "e0".into());
        assert_eq!(build_due(Some(&by_other), &m, base, false), Due::No("another engine's builder died within the hour"));
        assert_eq!(build_due(Some(&by_other), &later, base, false), Due::Yes { attempts: 0 }, "another engine's builder died an hour ago");
        assert_eq!(build_due(Some(&by_other), &next_boot, base, false), Due::Yes { attempts: 0 }, "another engine's builder died in an earlier boot");
    }

    #[test]
    fn a_failure_that_passes_is_deferred_and_one_that_stays_is_failed() {
        for e in [libc::ENOSPC, libc::EDQUOT, libc::EMFILE, libc::ENFILE, libc::ENOMEM, libc::EAGAIN, libc::EINTR, libc::ENOENT] {
            assert!(is_transient(&std::io::Error::from_raw_os_error(e)), "errno {e}");
        }
        for e in [libc::EISDIR, libc::EACCES, libc::EINVAL] {
            assert!(!is_transient(&std::io::Error::from_raw_os_error(e)), "errno {e}");
        }
        assert!(!is_transient(&std::io::Error::other("no errno")));
        assert_eq!(BuildRecord::parse(rec(BuildState::Deferred).line().as_bytes()).map(|r| r.state), Some(BuildState::Deferred));
    }

    #[test]
    fn only_old_builder_temporary_files_are_swept() {
        let d = std::env::temp_dir().join(format!("mcl-sweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let names = ["k.aot.123.tmp", "k.aot.tmp", "k.aot.9.tmp", "k.profile.123.tmp", "k.aot", "k.aot.x.tmp", "done.123.tmp", "k.aot.123.tmp.keep"];
        for n in names {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(AOT_TMP_STALE_S + 60);
        for n in names.iter().filter(|n| **n != "k.aot.9.tmp") {
            std::fs::File::options().write(true).open(d.join(n)).unwrap().set_modified(old).unwrap();
        }
        sweep_stale_tmps(&d);
        let mut left: Vec<String> = std::fs::read_dir(&d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, vec!["done.123.tmp", "k.aot", "k.aot.123.tmp.keep", "k.aot.9.tmp", "k.aot.x.tmp", "k.profile.123.tmp"], "an old builder's temporary file goes; a new one and anything else stays");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn this_binary_has_an_engine_id() {
        assert!(own_uuid().is_some(), "the test binary's own LC_UUID");
        let id = engine_id();
        assert!(id.starts_with(&format_tag()) && !id.contains(' ') && !id.ends_with("nouuid"), "{id}");
    }
}
