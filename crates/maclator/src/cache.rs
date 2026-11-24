//! Emulation of the kernel's shared-region syscalls for the arm64e dyld
//! shared cache.
//!
//! dyld asks the kernel to map the cache (`shared_region_map_and_slide_2_np`)
//! and later asks where it is (`shared_region_check_np`). We perform the
//! mapping ourselves at a fixed address inside our process (so AOT
//! translations of cache code stay valid across runs) and apply the slide
//! info (rebasing; pointer-auth signing is the identity in Maclator).

use crate::hostsys;
use crate::forksync::FMutex;

/// Where we place the arm64 shared region (unslid start 0x180000000).
pub const CACHE_BASE: u64 = 0x6_0000_0000;

pub static CACHE: FMutex<Option<MappedCache>> = FMutex::new(None);

#[derive(Debug, Clone)]
pub struct MappedCache {
    pub base: u64,
    pub slide: u64,
    pub ranges: Vec<(u64, u64)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SharedFileNp {
    fd: i32,
    mappings_count: u32,
    slide: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MappingSlideNp {
    address: u64,
    size: u64,
    file_offset: u64,
    slide_size: u64,
    slide_start: u64,
    max_prot: u32,
    init_prot: u32,
}

const VM_PROT_ZF: u32 = 0x10;

fn pread_all(fd: i32, buf: &mut [u8], off: u64) -> bool {
    let mut done = 0usize;
    while done < buf.len() {
        let r = unsafe { libc::pread(fd, buf[done..].as_mut_ptr() as *mut _, buf.len() - done, (off + done as u64) as i64) };
        if r <= 0 {
            return false;
        }
        done += r as usize;
    }
    true
}

/// arm64 macOS pages are 16 KiB; the host's are 4 KiB.
const PAGE16: u64 = 0x4000;

/// Reserve `size` bytes, PROT_NONE, for the whole cache and return where. `hint` is only a hint: the kernel places the
/// reservation, as it places the shared region itself, and the cache slides to wherever that is. The base is rounded
/// up to 16 KiB by over-reserving one page and trimming, so the slide keeps the guest's page alignment. Every region is
/// then mapped MAP_FIXED inside this reservation, never over unmapped space: that raises EXC_GUARD
/// (kGUARD_EXC_DEALLOC_GAP) on current Darwin, fatal in a platform binary, whose mmap hints the kernel does not honour.
unsafe fn reserve_cache(hint: u64, size: u64) -> std::io::Result<u64> {
    let Some(over) = size.checked_add(PAGE16) else { return Err(std::io::Error::from_raw_os_error(libc::ENOMEM)) };
    // A test may have the hint ignored, as the kernel ignores a platform binary's.
    #[cfg(test)]
    let hint = if HINT_IGNORED.with(|h| h.get()) { 0 } else { hint };
    let p = libc::mmap(hint as *mut _, over as usize, libc::PROT_NONE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
    if p == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    let q = p as u64;
    let base = (q + PAGE16 - 1) & !(PAGE16 - 1);
    // A trim that fails leaves the reservation half-made: keep its errno, release what is still reserved (only that:
    // unmapping a gap is what raises EXC_GUARD), and fail.
    if base > q {
        if let Err(e) = trim(q, base - q) {
            libc::munmap(q as *mut _, over as usize);
            return Err(e);
        }
    }
    let tail = q + over - (base + size);
    if tail > 0 {
        if let Err(e) = trim(base + size, tail) {
            libc::munmap(base as *mut _, (q + over - base) as usize);
            return Err(e);
        }
    }
    Ok(base)
}

#[cfg(test)]
thread_local! {
    /// (trims so far, the trim that fails; 0 = none): a test makes the nth trim on its own thread fail with EINVAL.
    static TRIM_FAIL: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
}

/// Unmap one end of the cache's over-reservation.
unsafe fn trim(addr: u64, len: u64) -> std::io::Result<()> {
    #[cfg(test)]
    {
        let (n, at) = TRIM_FAIL.with(|c| c.get());
        TRIM_FAIL.with(|c| c.set((n + 1, at)));
        if at != 0 && n + 1 == at {
            return Err(std::io::Error::from_raw_os_error(libc::EINVAL));
        }
    }
    if libc::munmap(addr as *mut _, len as usize) == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// The shared cache at one base per boot, as XNU slides its shared region once per boot.
unsafe fn reserve_boot_base(hint: u64, span: u64) -> std::io::Result<u64> {
    // Under test, the store and the boot are the test's own (TEST_BOOT), never the real ones; with none set, S 1.4.
    #[cfg(test)]
    {
        let set = TEST_BOOT.with(|t| t.borrow().clone());
        return match set {
            Some((dir, boot)) => unsafe { reserve_boot_base_in(&dir, &boot, hint, span) },
            None => unsafe { reserve_cache(hint, span) },
        };
    }
    #[cfg(not(test))]
    match boot_session_uuid() {
        Some(boot) => unsafe { reserve_boot_base_in(&crate::aot::cache_dir(), &boot, hint, span) },
        None => unsafe { reserve_cache(hint, span) },
    }
}

#[cfg(test)]
thread_local! {
    /// A test sets this to have reserve_cache's hint ignored, as the kernel ignores a platform binary's.
    static HINT_IGNORED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// The store directory and boot UUID a test gives map_and_slide_at; None leaves the boot base out.
    static TEST_BOOT: std::cell::RefCell<Option<(std::path::PathBuf, String)>> = const { std::cell::RefCell::new(None) };
}

/// kern.bootsessionuuid: a new value each boot.
fn boot_session_uuid() -> Option<String> {
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    let r = unsafe { libc::sysctlbyname(c"kern.bootsessionuuid".as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut(), 0) };
    let s = std::str::from_utf8(&buf[..len.min(buf.len())]).ok()?.trim_end_matches('\0');
    (r == 0 && !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-')).then(|| s.to_string())
}

fn boot_base_path(dir: &std::path::Path, boot: &str, span: u64) -> std::path::PathBuf {
    dir.join(format!("dsc-base-{boot}-{span:x}"))
}

fn boot_base_record(boot: &str, span: u64, base: u64) -> String {
    format!("MCLBASE1 {boot} {span:x} {base:x}\n")
}

/// This boot's recorded base for a cache of `span` bytes, if a record of ours says so.
fn read_boot_base(dir: &std::path::Path, boot: &str, span: u64) -> Option<u64> {
    let uid = unsafe { libc::geteuid() };
    let dmd = std::fs::symlink_metadata(dir).ok()?;
    if !(dmd.is_dir() && crate::aot::owned(&dmd, uid)) {
        return None;
    }
    let path = boot_base_path(dir, boot, span);
    let md = std::fs::symlink_metadata(&path).ok()?;
    if !(md.is_file() && crate::aot::owned(&md, uid) && md.len() < 256) {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let mut f = text.trim_end().split(' ');
    let (magic, b, s, base) = (f.next()?, f.next()?, f.next()?, f.next()?);
    if magic != "MCLBASE1" || b != boot || s != format!("{span:x}") || f.next().is_some() {
        return None;
    }
    let base = u64::from_str_radix(base, 16).ok()?;
    (base != 0 && base % PAGE16 == 0).then_some(base)
}

/// Publish `base` as this boot's base: the first writer wins (link(2) fails on an existing name).
fn write_boot_base(dir: &std::path::Path, boot: &str, span: u64, base: u64) {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let uid = unsafe { libc::geteuid() };
    if !crate::aot::owned_dir(dir, uid) {
        return;
    }
    let path = boot_base_path(dir, boot, span);
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let tmp = dir.join(format!("dsc-base-{boot}-{span:x}.{}.{ns}.tmp", std::process::id()));
    let ok = (|| -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o644).open(&tmp)?;
        f.write_all(boot_base_record(boot, span, base).as_bytes())?;
        f.sync_all()?;
        std::fs::hard_link(&tmp, &path)
    })();
    let _ = std::fs::remove_file(&tmp);
    if let Err(e) = ok {
        if e.kind() != std::io::ErrorKind::AlreadyExists {
            eprintln!("maclator: could not record this boot's shared cache base in {}: {e}", path.display());
        }
    }
}

/// Take exactly [base, base + span): fixed, never overwriting, so an occupied range fails and nothing is replaced.
unsafe fn reserve_exact(base: u64, span: u64) -> std::io::Result<()> {
    extern "C" {
        static mach_task_self_: u32;
        fn mach_vm_allocate(task: u32, addr: *mut u64, size: u64, flags: i32) -> i32;
    }
    extern "C" {
        fn mach_vm_deallocate(task: u32, addr: u64, size: u64) -> i32;
    }
    const VM_FLAGS_FIXED: i32 = 0;
    if base == 0 || base % PAGE16 != 0 || span == 0 || base.checked_add(span).is_none() {
        return Err(std::io::Error::from_raw_os_error(libc::EINVAL));
    }
    let mut a = base;
    let kr = unsafe { mach_vm_allocate(mach_task_self_, &mut a, span, VM_FLAGS_FIXED) };
    if kr != 0 {
        return Err(std::io::Error::other(format!("mach_vm_allocate fixed: kern_return {kr}")));
    }
    if a != base || unsafe { libc::mprotect(base as *mut _, span as usize, libc::PROT_NONE) } != 0 {
        let e = if a != base { std::io::Error::other(format!("mach_vm_allocate fixed returned {a:#x}")) } else { std::io::Error::last_os_error() };
        unsafe { mach_vm_deallocate(mach_task_self_, a, span) };
        return Err(e);
    }
    Ok(())
}

/// `reserve_boot_base` with its directory and boot named, so a test can give its own.
unsafe fn reserve_boot_base_in(dir: &std::path::Path, boot: &str, hint: u64, span: u64) -> std::io::Result<u64> {
    if let Some(b) = read_boot_base(dir, boot, span) {
        match unsafe { reserve_exact(b, span) } {
            Ok(()) => return Ok(b),
            Err(e) => eprintln!("[maclator] this boot's shared cache base {b:#x} is occupied in this process ({e}); sliding this process alone"),
        }
        return unsafe { reserve_cache(hint, span) };
    }
    // The first process of the boot takes the preferred base exactly when it is free (a platform binary's hint is
    // ignored, but the fixed-mapping probe measured that it can still take the base fixed), else where the kernel puts it.
    // Only the preferred base is recorded: a base the kernel placed (the band it uses for this process's own
    // mappings) is likelier to be taken in the next process, and a first writer's record lasts the whole boot.
    match unsafe { reserve_exact(hint, span) } {
        Ok(()) => {
            write_boot_base(dir, boot, span, hint);
            Ok(hint)
        }
        Err(_) => unsafe { reserve_cache(hint, span) },
    }
}

/// The errno a failed cache reservation reports to dyld: the kernel's own, or ENOMEM when there is none.
fn reservation_errno(e: &std::io::Error) -> i64 {
    e.raw_os_error().unwrap_or(libc::ENOMEM) as i64
}

/// Handle `shared_region_map_and_slide_2_np(files_count, files, mappings_count, mappings)`.
/// Returns 0 or an errno.
pub fn map_and_slide_2(files_count: u32, files: u64, mappings_count: u32, mappings: u64) -> i64 {
    map_and_slide_at(CACHE_BASE, files_count, files, mappings_count, mappings)
}

/// `map_and_slide_2` with the preferred base as a parameter, so a test can use an address of its own.
fn map_and_slide_at(hint: u64, files_count: u32, files: u64, mappings_count: u32, mappings: u64) -> i64 {
    let files: &[SharedFileNp] = unsafe { std::slice::from_raw_parts(files as *const SharedFileNp, files_count as usize) };
    let maps: &[MappingSlideNp] = unsafe { std::slice::from_raw_parts(mappings as *const MappingSlideNp, mappings_count as usize) };
    if maps.is_empty() {
        return libc::EINVAL as i64;
    }
    let mut guard = CACHE.lock().unwrap();
    if guard.is_some() {
        return libc::EEXIST as i64;
    }
    let unslid_base = maps[0].address;
    let ends: Option<Vec<u64>> = maps.iter().map(|m| m.address.checked_add(m.size)).collect();
    let unslid_end = ends.and_then(|e| e.into_iter().max()).unwrap_or(unslid_base);
    let span = unslid_end.wrapping_sub(unslid_base).checked_add(PAGE16 - 1).map(|s| s & !(PAGE16 - 1));
    let span = match span {
        Some(s) if s > 0 && unslid_end > unslid_base && !maps.iter().any(|m| m.address < unslid_base) => s,
        _ => {
            eprintln!("maclator: shared cache mappings do not fit one span from the first mapping, {unslid_base:#x}");
            return libc::EINVAL as i64;
        }
    };
    let base = match unsafe { reserve_boot_base(hint, span) } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("maclator: reserving {span:#x} bytes for the shared cache failed: {e}");
            return reservation_errno(&e);
        }
    };
    if base != hint {
        eprintln!("[maclator] shared cache base {hint:#x} unavailable, mapping it at {base:#x} instead");
    }
    let slide = base.wrapping_sub(unslid_base);
    let mut ranges = Vec::new();
    let mut fds = Vec::new();
    let mut mi = 0usize;
    for f in files {
        for _ in 0..f.mappings_count {
            let m = maps[mi];
            mi += 1;
            let addr = m.address.wrapping_add(slide);
            let size = m.size as usize;
            let prot = (m.init_prot & 3) as i32; // never map guest code executable on the host
            unsafe {
                let p = if f.fd < 0 {
                    // Dynamic config data lives in the caller's memory.
                    let p = libc::mmap(addr as *mut _, size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0);
                    if p != libc::MAP_FAILED {
                        std::ptr::copy_nonoverlapping(m.file_offset as *const u8, p as *mut u8, size);
                    }
                    p
                } else if m.max_prot & VM_PROT_ZF != 0 {
                    libc::mmap(addr as *mut _, size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0)
                } else {
                    let wprot = if m.slide_size != 0 { libc::PROT_READ | libc::PROT_WRITE } else { prot.max(libc::PROT_READ) };
                    libc::mmap(addr as *mut _, size, wprot, libc::MAP_PRIVATE | libc::MAP_FIXED, f.fd, m.file_offset as i64)
                };
                if p == libc::MAP_FAILED || p as u64 != addr {
                    eprintln!("maclator: failed to map shared cache region {:#x}+{:#x}: {}", addr, size, std::io::Error::last_os_error());
                    libc::munmap(base as *mut _, span as usize);
                    return libc::ENOMEM as i64;
                }
                if m.slide_size == 0 {
                    let final_prot = if prot == 0 { libc::PROT_NONE } else { prot };
                    libc::mprotect(addr as *mut _, size, final_prot);
                }
            }
            ranges.push((addr, m.size));
            fds.push(f.fd);
        }
    }
    // Second pass: apply slide info. It lives in each file's __LINKEDIT
    // mapping (sms_slide_start is an unslid address), mapped above.
    for (i, m) in maps.iter().enumerate() {
        if m.slide_size == 0 {
            continue;
        }
        let addr = m.address.wrapping_add(slide);
        let info = unsafe { std::slice::from_raw_parts(m.slide_start.wrapping_add(slide) as *const u8, m.slide_size as usize) };
        if unsafe { share_slid(fds.get(i).copied().unwrap_or(-1), m, info, slide, unslid_base, addr) } {
            continue;
        }
        if let Err(e) = unsafe { apply_slide_info(info, addr, m.size, slide, unslid_base) } {
            eprintln!("maclator: slide info: {e}");
            unsafe { libc::munmap(base as *mut _, span as usize) };
            return libc::EINVAL as i64;
        }
        let prot = (m.init_prot & 3) as i32;
        unsafe { libc::mprotect(addr as *mut _, m.size as usize, if prot == 0 { libc::PROT_NONE } else { prot }) };
    }
    if std::env::var_os("MACLATOR_TRACE").is_some() {
        eprintln!("[maclator] mapped arm64 shared cache at {:#x} (slide {:#x}), {} regions", base, slide, ranges.len());
    }
    let end = ranges.iter().map(|(a, s)| a + s).max().unwrap_or(base);
    *guard = Some(MappedCache { base, slide, ranges });
    drop(guard);
    let mut uuid = [0u8; 16];
    unsafe { std::ptr::copy_nonoverlapping((base + 88) as *const u8, uuid.as_mut_ptr(), 16) };
    crate::aot::register_cache(uuid, base, end);
    0
}

pub fn check_np() -> Option<u64> {
    CACHE.lock().unwrap().as_ref().map(|c| c.base)
}

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn rd64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Apply dyld cache slide info (v3 or v5) to a mapped region.
unsafe fn apply_slide_info(info: &[u8], region: u64, size: u64, slide: u64, unslid_base: u64) -> Result<(), String> {
    let version = rd32(info, 0);
    match version {
        3 => {
            let page_size = rd32(info, 4) as u64;
            let count = rd32(info, 8) as usize;
            let auth_value_add = rd64(info, 16);
            for i in 0..count {
                let start = u16::from_le_bytes([info[24 + i * 2], info[25 + i * 2]]);
                if start == 0xFFFF {
                    continue;
                }
                let page = region + i as u64 * page_size;
                if page >= region + size {
                    break;
                }
                let mut loc = page + start as u64;
                loop {
                    let raw = std::ptr::read_unaligned(loc as *const u64);
                    let delta = (raw >> 51) & 0x7ff;
                    let auth = raw >> 63 != 0;
                    let v = if auth {
                        (raw & 0xffff_ffff).wrapping_add(auth_value_add).wrapping_add(slide)
                    } else {
                        let v51 = raw & ((1u64 << 51) - 1);
                        let top8 = v51 & 0x0007_F800_0000_0000;
                        let bottom43 = v51 & 0x0000_07FF_FFFF_FFFF;
                        ((top8 << 13) | bottom43).wrapping_add(slide)
                    };
                    std::ptr::write_unaligned(loc as *mut u64, v);
                    if delta == 0 {
                        break;
                    }
                    loc += delta * 8;
                }
            }
            Ok(())
        }
        5 => {
            let page_size = rd32(info, 4) as u64;
            let count = rd32(info, 8) as usize;
            let value_add = rd64(info, 16);
            let _ = unslid_base;
            for i in 0..count {
                let start = u16::from_le_bytes([info[24 + i * 2], info[25 + i * 2]]);
                if start == 0xFFFF {
                    continue;
                }
                let page = region + i as u64 * page_size;
                if page >= region + size {
                    break;
                }
                let mut loc = page + start as u64;
                loop {
                    let raw = std::ptr::read_unaligned(loc as *const u64);
                    let next = (raw >> 52) & 0x7ff;
                    let auth = raw >> 63 != 0;
                    let target = (raw & ((1u64 << 34) - 1)).wrapping_add(value_add).wrapping_add(slide);
                    let v = if auth { target } else { target | (((raw >> 34) & 0xff) << 56) };
                    std::ptr::write_unaligned(loc as *mut u64, v);
                    if next == 0 {
                        break;
                    }
                    loc += next * 8;
                }
            }
            Ok(())
        }
        v => Err(format!("unsupported slide info version {v}")),
    }
}

/// Sliding a mapping in place writes into nearly every page of the cache's data mappings, so each translated process
/// held its own private copy of them (about 390 MiB on macOS 27 with 4 KiB host pages), where the kernel shares one
/// slid copy among native processes. Instead the slid bytes are kept once in a file (in root's shared store when there
/// is one, `SHARED_STORE`, else beside the AOT files) and mapped
/// from it copy-on-write: pages the guest only reads are shared clean file pages, and only pages it writes become
/// its own, as with the kernel's shared region. `MACLATOR_PRIVATE_SLIDE` set keeps the old in-place slide.
///
/// A slid file is trusted only in a directory of its store's owner (our effective uid, or root for the shared store)
/// that no group or other can write, only if the file is that owner's and no one else can write it, and only if its record matches: after the slid bytes it records the mapping
/// it was slid from and a SHA-256 of each chunk. The first process in a boot to map a file checks every chunk and
/// leaves a stamp beside it; later processes in that boot find the stamp and check the record and some chunks.
/// Anything else is refused, and the mapping is slid privately as before.
const SLID_FORMAT: u32 = 2;
const SLID_MAGIC: &[u8; 8] = b"MCLSLID\0";
const SLID_STAMP_MAGIC: &[u8; 8] = b"MCLSTMP\0";
/// A refusal: a stamp's bytes under this magic instead, left beside a file one of whose chunks did not match when every
/// chunk was checked, so the rest of that boot refuses the file at once rather than check it whole again under its lock.
const SLID_REFUSED_MAGIC: &[u8; 8] = b"MCLRFSD\0";
/// Pages slid per step while building a slid file, so building costs one chunk of memory, not the mapping.
const SLID_CHUNK_PAGES: usize = 64;
/// Bytes per recorded hash. Small, because a process hashes a few of them each time it maps a file: SHA-256 ran at 40 to
/// 100 MiB/s in a macOS 27 x86 guest, where two 1 MiB chunks of each of 14 files took 250 to 620 ms.
const SLID_HASH_CHUNK: usize = 0x10000;
/// How long to wait for another process building the same file before sliding privately instead.
const SLID_LOCK_WAIT_MS: u64 = 30_000;
/// Chunks a process that finds a stamp hashes before mapping a slid file, besides the first: chosen from its pid, so
/// different processes check different chunks. Hashing every chunk on every start cost 7.6 to 11.8 s for macOS 27's
/// cache in that guest, so only the first process in a boot does it.
const SLID_CHECK_CHUNKS: usize = 1;

extern "C" {
    /// CommonCrypto's SHA-256 (libSystem).
    fn CC_SHA256(data: *const libc::c_void, len: u32, md: *mut u8) -> *mut u8;
}

fn sha256(b: &[u8]) -> [u8; 32] {
    let mut h = [0u8; 32];
    unsafe { CC_SHA256(b.as_ptr().cast(), b.len() as u32, h.as_mut_ptr()) };
    h
}

/// The mapping a slid file was slid from, and the chunk size of its hashes: what its record must say.
#[derive(Clone, Copy, Debug)]
struct SlidId {
    uuid: [u8; 16],
    file_offset: u64,
    size: u64,
    slide: u64,
    chunk: usize,
}

impl SlidId {
    fn chunks(&self) -> usize {
        (self.size as usize).div_ceil(self.chunk)
    }
    /// The record after the slid bytes is this header, a SHA-256 of each chunk, and a SHA-256 of both.
    fn header(&self) -> [u8; 64] {
        let mut h = [0u8; 64];
        h[..8].copy_from_slice(SLID_MAGIC);
        h[8..12].copy_from_slice(&SLID_FORMAT.to_le_bytes());
        h[12..16].copy_from_slice(&(self.chunk as u32).to_le_bytes());
        h[16..32].copy_from_slice(&self.uuid);
        h[32..40].copy_from_slice(&self.file_offset.to_le_bytes());
        h[40..48].copy_from_slice(&self.size.to_le_bytes());
        h[48..56].copy_from_slice(&self.slide.to_le_bytes());
        h[56..64].copy_from_slice(&(self.chunks() as u64).to_le_bytes());
        h
    }
    fn record_len(&self) -> usize {
        64 + 32 * self.chunks() + 32
    }
}

/// The chunks a process checks: the first, and `SLID_CHECK_CHUNKS` others spread from `seed`; all of them if that many.
/// The others are taken from chunks 1 to n - 1, so none is chunk 0 again; seed c (0 < c < n) starts at chunk c.
fn checked_chunks(n: usize, seed: usize) -> Vec<usize> {
    if SLID_CHECK_CHUNKS + 1 >= n {
        return (0..n).collect();
    }
    let rest = n - 1;
    let step = (rest / SLID_CHECK_CHUNKS).max(1);
    let mut v = vec![0];
    v.extend((0..SLID_CHECK_CHUNKS).map(|j| 1 + (seed % rest + rest - 1 + j * step) % rest));
    v
}

#[derive(Debug, PartialEq)]
enum Slid {
    Mapped,
    Missing,
    Refused,
}

/// Where root keeps one slid copy of each shared cache mapping for every process of the system, as Rosetta 2 keeps its
/// translations once, in `/var/db/oah`, for every user. A per-user store sits under `HOME`, and a sandbox container
/// gives each app a `HOME` of its own, so without this every container and every daemon's home held its own copy of the
/// same slid bytes: 16 stores and 8,946 MiB of identical files in one macOS 27 login boot, until the volume filled.
///
/// Used only when it is a real directory (not a symlink) owned by root that no group or other can write. Then only root
/// builds slid files there; every process maps root's files and checks them as it checks its own; and a process that
/// finds no file there, or may not read one (a sandbox), slides privately and writes no copy of its own. Without it
/// the per-user store is used as before, except by a process denied even the lookup of it (a sandbox), which slides privately. `MACLATOR_SHARED_STORE` names another such directory; empty, it turns it off.
const SHARED_STORE: &str = "/private/var/db/maclator";

/// The directory the shared store would be, from `MACLATOR_SHARED_STORE` as `var` gives it.
fn shared_store_path(var: Option<std::ffi::OsString>) -> Option<std::path::PathBuf> {
    match var {
        None => Some(SHARED_STORE.into()),
        Some(v) if v.is_empty() => None,
        Some(v) => Some(v.into()),
    }
}

/// Where this process's slid files are: the directory, the uid whose files (and directory) it trusts there, and
/// whether it may build a missing one.
#[derive(Debug, PartialEq)]
struct SlidStore {
    dir: std::path::PathBuf,
    owner: u32,
    build: bool,
}

/// The shared store `shared` if it is a real directory owned by `shared_owner` that no group or other can write, else
/// the per-user store `home` (owned by `euid`, which may build in it). Only the shared store's owner builds in it.
///
/// A shared store this process may not even look at (a sandbox denies it) is still the store, with no building: the
/// mapping then fails its own check and the cache is slid privately, rather than copied into this process's `HOME`.
/// Only a shared store that does not exist, or that is not safe to trust, leaves the per-user store in use.
fn slid_store_in(shared: Option<&std::path::Path>, shared_owner: u32, home: std::path::PathBuf, euid: u32) -> SlidStore {
    static UNSAFE_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if let Some(dir) = shared {
        match std::fs::symlink_metadata(dir) {
            Ok(md) if md.is_dir() && crate::aot::owned(&md, shared_owner) => {
                return SlidStore { dir: dir.to_path_buf(), owner: shared_owner, build: euid == shared_owner };
            }
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return SlidStore { dir: dir.to_path_buf(), owner: shared_owner, build: false };
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            _ if UNSAFE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) => {}
            _ => eprintln!("maclator: not using the shared slid store {}: not a directory of uid {shared_owner} that only it can write; using {}", dir.display(), home.display()),
        }
    }
    SlidStore { dir: home, owner: euid, build: true }
}

/// Map `m` (already mapped private from the cache file at `addr`) from its shared slid file, building the file first
/// if there is none and this process may. False means nothing changed and the caller slides in place as before.
unsafe fn share_slid(fd: i32, m: &MappingSlideNp, info: &[u8], slide: u64, unslid_base: u64, addr: u64) -> bool {
    let shared = shared_store_path(std::env::var_os("MACLATOR_SHARED_STORE"));
    let store = slid_store_in(shared.as_deref(), 0, crate::aot::cache_dir(), libc::geteuid());
    share_slid_in(&store, fd, m, info, slide, unslid_base, addr)
}

/// `share_slid` in the store `store`, so a test can give its own.
unsafe fn share_slid_in(store: &SlidStore, fd: i32, m: &MappingSlideNp, info: &[u8], slide: u64, unslid_base: u64, addr: u64) -> bool {
    static BOOT_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static MISSING_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let Some((path, id)) = slid_path(&store.dir, fd, m, info, slide) else { return false };
    let prot = match (m.init_prot & 3) as i32 {
        0 => libc::PROT_NONE,
        p => p,
    };
    let (size, uid, seed, boot) = (m.size as usize, store.owner, libc::getpid() as usize, boot_session());
    if boot.is_none() && !BOOT_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("maclator: kern.bootsessionuuid gave no answer, so every chunk of each slid file is checked at every start");
    }
    let boot = boot.as_deref();
    match map_slid(addr, size, prot, &path, &id, uid, seed, boot, SLID_LOCK_WAIT_MS) {
        Slid::Mapped => true,
        Slid::Refused => false,
        Slid::Missing if store.build => build_slid(fd, m, info, slide, unslid_base, &path, &id, uid, SLID_LOCK_WAIT_MS) && map_slid(addr, size, prot, &path, &id, uid, seed, boot, SLID_LOCK_WAIT_MS) == Slid::Mapped,
        Slid::Missing => {
            if !MISSING_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("maclator: {} is not in the shared store yet (only uid {uid} builds there); sliding privately", path.display());
            }
            false
        }
    }
}

/// This boot's session, `kern.bootsessionuuid`: new at each boot, so a stamp left in an earlier boot never matches.
/// None if the kernel does not say.
fn boot_session() -> Option<Vec<u8>> {
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    let r = unsafe { libc::sysctlbyname(c"kern.bootsessionuuid".as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) };
    let v = buf[..len.min(buf.len())].split(|&b| b == 0).next().unwrap_or(&[]);
    (r == 0 && !v.is_empty()).then(|| v.to_vec())
}

/// The slid file in `dir` for mapping `m` of the cache file `fd`: keyed by that file's UUID, the mapping's place in it,
/// and the slide, so every process that maps the same cache at the same slide names the same file.
fn slid_path(dir: &std::path::Path, fd: i32, m: &MappingSlideNp, info: &[u8], slide: u64) -> Option<(std::path::PathBuf, SlidId)> {
    if fd < 0 || info.len() < 24 || rd32(info, 4) == 0 || std::env::var_os("MACLATOR_PRIVATE_SLIDE").is_some() {
        return None;
    }
    let mut uuid = [0u8; 16];
    if !pread_all(fd, &mut uuid, 0x58) || uuid == [0u8; 16] {
        return None;
    }
    let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    let path = dir.join(format!("dsc-slid{SLID_FORMAT}-{hex}-{:x}-{:x}-{slide:x}.data", m.file_offset, m.size));
    Some((path, SlidId { uuid, file_offset: m.file_offset, size: m.size, slide, chunk: SLID_HASH_CHUNK }))
}

fn refuse(path: &std::path::Path, why: &str) -> Slid {
    eprintln!("maclator: not using {}: {why}; sliding privately (remove it to rebuild)", path.display());
    Slid::Refused
}

/// What a stamp says: this boot's session, the slid file it vouches for as `fstat` saw it when every chunk was checked
/// (its inode, size and modification time, which any write through the filesystem moves), and the file's record.
fn stamp_bytes(boot: &[u8], md: &std::fs::Metadata, sum: &[u8]) -> Vec<u8> {
    use std::os::unix::fs::MetadataExt;
    let mut s = SLID_STAMP_MAGIC.to_vec();
    s.extend_from_slice(&md.ino().to_le_bytes());
    s.extend_from_slice(&md.size().to_le_bytes());
    s.extend_from_slice(&md.mtime().to_le_bytes());
    s.extend_from_slice(&md.mtime_nsec().to_le_bytes());
    s.extend_from_slice(sum);
    s.extend_from_slice(boot);
    s
}

/// Whether the stamp at `stamp` says exactly `want`, as a regular file of ours that only we can write: opened without
/// following a symlink or waiting on a FIFO.
fn stamp_ok(stamp: &std::path::Path, want: &[u8], uid: u32) -> bool {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    let Ok(f) = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(stamp) else { return false };
    let mut got = Vec::new();
    f.metadata().is_ok_and(|md| md.is_file() && crate::aot::owned(&md, uid) && md.len() == want.len() as u64)
        && (&f).take(want.len() as u64 + 1).read_to_end(&mut got).is_ok()
        && got == want
}

/// Leave `want` as the stamp, written under a temporary name and renamed into place, so no reader sees part of one and
/// a symlink at its name is replaced, not followed. Called with the file's `BuildLock` held, so no other process is
/// writing one, and a temporary file left by one that died is ours to replace.
fn stamp_write(stamp: &std::path::Path, want: &[u8]) -> bool {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let tmp = stamp.with_extension("stamp-tmp");
    let _ = std::fs::remove_file(&tmp);
    // Readable by all, writable by its owner alone (set past the umask): in root's shared store every uid reads root's
    // stamp, or it would check every chunk at every start. A per-user store's directory is made 0700 when created.
    let ok = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o644)
        .open(&tmp)
        .and_then(|mut f| f.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o644)).and_then(|()| f.write_all(want)))
        .and_then(|()| std::fs::rename(&tmp, stamp))
        .is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp);
    }
    ok
}

/// Map the slid file at `path` over [addr, addr + size), copy-on-write, once it has passed the checks above; `uid` and
/// `seed` are the uid whose files are trusted (ours, or the shared store's owner) and our pid, `boot` this boot's session. Without a stamp of this boot for this file, every
/// chunk is checked, under the file's `BuildLock` (waited for at most `wait_ms`) so that a process going second finds
/// the first one's stamp rather than checking again, and a stamp is left if every chunk matches. With no `boot`, every
/// chunk is checked and no stamp is trusted or left. A refusal of this boot for this file, left when every chunk was
/// checked and one did not match, refuses it at once, before the lock and after it. On anything but `Mapped` the old
/// mapping stays.
#[allow(clippy::too_many_arguments)]
unsafe fn map_slid(addr: u64, size: usize, prot: i32, path: &std::path::Path, id: &SlidId, uid: u32, seed: usize, boot: Option<&[u8]>, wait_ms: u64) -> Slid {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    static DIR_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let Some(dir) = path.parent() else { return Slid::Refused };
    match std::fs::symlink_metadata(dir) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Slid::Missing,
        Ok(md) if md.is_dir() && crate::aot::owned(&md, uid) => {}
        _ if DIR_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) => return Slid::Refused,
        _ => return refuse(dir, "not a directory of ours that only we can write"),
    }
    let f = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Slid::Missing,
        Err(e) => return refuse(path, &e.to_string()),
    };
    let md = match f.metadata() {
        Ok(md) if md.is_file() && crate::aot::owned(&md, uid) && md.len() == (size + id.record_len()) as u64 => md,
        _ => return refuse(path, "not a file of ours, of the right size, that only we can write"),
    };
    let n = id.chunks();
    let mut rec = vec![0u8; id.record_len()];
    if !pread_all(f.as_raw_fd(), &mut rec, size as u64) || rec[..64] != id.header() || rec[64 + 32 * n..] != sha256(&rec[..64 + 32 * n]) {
        return refuse(path, "its record is not this mapping's");
    }
    let stamp = path.with_extension("stamp");
    let want = boot.map(|b| stamp_bytes(b, &md, &rec[64 + 32 * n..]));
    let stamped = || want.as_ref().is_some_and(|w| stamp_ok(&stamp, w, uid));
    let refusal_at = path.with_extension("refused");
    let refusal = want.as_ref().map(|w| [&SLID_REFUSED_MAGIC[..], &w[SLID_STAMP_MAGIC.len()..]].concat());
    let refused = || refusal.as_ref().is_some_and(|r| stamp_ok(&refusal_at, r, uid));
    if refused() {
        return refuse(path, "every chunk was checked earlier in this boot and one did not match");
    }
    let mut lock = None;
    let mut trusted = stamped();
    if !trusted && want.is_some() {
        lock = BuildLock::take(&path.with_extension("lock"), wait_ms);
        if refused() {
            return refuse(path, "every chunk was checked earlier in this boot and one did not match");
        }
        // Whether we got the lock or gave up waiting, the first process may have left its stamp meanwhile.
        trusted = stamped();
    }
    let chunks = if trusted { checked_chunks(n, seed) } else { (0..n).collect() };
    let mut buf = vec![0u8; id.chunk.min(size)];
    for c in chunks {
        let (off, len) = (c * id.chunk, (size - c * id.chunk).min(id.chunk));
        let read = pread_all(f.as_raw_fd(), &mut buf[..len], off as u64);
        if !read || sha256(&buf[..len])[..] != rec[64 + 32 * c..96 + 32 * c] {
            if trusted {
                // The file changed since its stamp was left: the next process checks every chunk, and refuses it too.
                let _ = std::fs::remove_file(&stamp);
            } else if let (true, Some(_), Some(r)) = (read, &lock, &refusal) {
                // Every chunk was checked and this one does not match: the rest of this boot refuses the file without
                // checking it again. A read that failed may not fail again, so it leaves no refusal.
                let _ = stamp_write(&refusal_at, r);
            }
            return refuse(path, &format!("chunk {c} {}", if read { "does not match its hash" } else { "could not be read" }));
        }
    }
    if let (false, Some(_), Some(w)) = (trusted, &lock, &want) {
        if !stamp_write(&stamp, w) {
            eprintln!("maclator: could not leave a stamp beside {}; the next process checks every chunk again", path.display());
        }
    }
    drop(lock);
    let p = libc::mmap(addr as *mut _, size, prot, libc::MAP_PRIVATE | libc::MAP_FIXED, f.as_raw_fd(), 0);
    if p != libc::MAP_FAILED && p as u64 == addr {
        Slid::Mapped
    } else {
        Slid::Refused
    }
}

/// `info` cut down to its pages [first, first + n): slide info v3 and v5 both keep the page count at 8 and the page
/// starts at 24.
fn sub_info(info: &[u8], first: usize, n: usize) -> Option<Vec<u8>> {
    let count = rd32(info, 8) as usize;
    if first + n > count || info.len() < 24 + count * 2 {
        return None;
    }
    let mut v = info[..24].to_vec();
    v[8..12].copy_from_slice(&(n as u32).to_le_bytes());
    v.extend_from_slice(&info[24 + first * 2..24 + (first + n) * 2]);
    Some(v)
}

/// A builder's `flock` on a slid file's `<file>.lock`. A flock lock belongs to the open file description, which a fork
/// shares: "file descriptors duplicated through dup(2) or fork(2) do not result in multiple instances of a lock, but
/// rather multiple references to a single lock" (flock(2)), and it is let go only when the last of them is closed. So a
/// child forked while we hold one, or exec'd with our descriptor, would keep the lock after we let it go, and other
/// processes would wait for it, give up and slide privately. The descriptor is opened close-on-exec, and is listed in
/// `SLID_LOCK_FDS`, whose fork child handler closes the child's copies. Closing a copy drops that reference only; the
/// child never unlocks, which would unlock for us too. (fcntl's process locks are not inherited by a fork, but do not
/// keep two threads or two lock files of one process apart, and any close of the file lets them go; macOS's OFD locks,
/// F_OFD_SETLK, are shared across a fork as flock's are.)
struct BuildLock {
    fd: i32,
    slot: usize,
}

/// The descriptors of the build locks this process holds, -1 a free slot: atomics only, because the fork child handler
/// runs where only async-signal-safe calls may be made.
static SLID_LOCK_FDS: [std::sync::atomic::AtomicI32; 64] = [const { std::sync::atomic::AtomicI32::new(-1) }; 64];
/// Held while a lock descriptor is opened and listed, or unlisted and closed, and across every fork (the prepare
/// handler takes it), so no child gets a descriptor it does not know to close.
static SLID_LOCK_GATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The fork handlers are registered once per process, when the image loads, before `main` and before any thread
/// exists -- the way QEMU's `rcu_init` constructor registers its `pthread_atfork` handlers. Any lazy registration
/// races a fork. Inside the gate, a first builder could block in `pthread_atfork` (Darwin holds its registration
/// lock across `fork`) while holding the gate, and the child inherited a locked gate with no child handler. Through
/// `pthread_once`, Darwin's once is owner-thread based with no fork recovery, so a child forked mid-registration
/// inherits an initialization owned by a thread that no longer exists. With nothing registered lazily, `take` has
/// no such window. `SLID_LOCK_REGISTERED` says the initializer ran.
static SLID_LOCK_REGISTERED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

extern "C" fn slid_lock_register() {
    unsafe { libc::pthread_atfork(Some(slid_lock_prepare), Some(slid_lock_parent), Some(slid_lock_child)) };
    SLID_LOCK_REGISTERED.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// dyld runs every `__mod_init_func` entry of the image before `main`, with libSystem already initialized.
#[used]
#[link_section = "__DATA,__mod_init_func,mod_init_funcs"]
static SLID_LOCK_INIT: extern "C" fn() = slid_lock_register;

/// `take` never registers. Debug and test builds assert that the initializer did; a release build where it did not
/// (a linker that stopped treating the section as initializers) takes no lock, so every caller slides privately
/// rather than build with no fork handlers.
fn slid_lock_registered() -> bool {
    let ok = SLID_LOCK_REGISTERED.load(std::sync::atomic::Ordering::SeqCst);
    #[cfg(any(debug_assertions, test))]
    assert!(ok, "fork handlers not registered at load");
    ok
}

fn slid_gate_take() {
    use std::sync::atomic::Ordering;
    while SLID_LOCK_GATE.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        unsafe { libc::sched_yield() };
    }
}

fn slid_gate_give() {
    SLID_LOCK_GATE.store(false, std::sync::atomic::Ordering::Release);
}

extern "C" fn slid_lock_prepare() {
    slid_gate_take();
}

extern "C" fn slid_lock_parent() {
    slid_gate_give();
}

/// In a fork child: close the copies of every build lock descriptor, then free the gate.
extern "C" fn slid_lock_child() {
    for s in &SLID_LOCK_FDS {
        let fd = s.swap(-1, std::sync::atomic::Ordering::SeqCst);
        if fd >= 0 {
            unsafe { libc::close(fd) };
        }
    }
    slid_gate_give();
}

impl BuildLock {
    /// Take the lock on `lock_path`, waiting at most `wait_ms` for another holder.
    fn take(lock_path: &std::path::Path, wait_ms: u64) -> Option<BuildLock> {
        use std::os::unix::ffi::OsStrExt;
        use std::sync::atomic::Ordering;
        let c = std::ffi::CString::new(lock_path.as_os_str().as_bytes()).ok()?;
        if !slid_lock_registered() {
            return None;
        }
        slid_gate_take();
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC, 0o644 as libc::c_uint) };
        let slot = if fd < 0 { None } else { SLID_LOCK_FDS.iter().position(|s| s.compare_exchange(-1, fd, Ordering::SeqCst, Ordering::SeqCst).is_ok()) };
        if fd >= 0 && slot.is_none() {
            unsafe { libc::close(fd) };
        }
        slid_gate_give();
        let lock = BuildLock { fd, slot: slot? };
        let mut waited = 0;
        while unsafe { libc::flock(lock.fd, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            if waited >= wait_ms {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
            waited += 20;
        }
        Some(lock)
    }
}

impl Drop for BuildLock {
    /// Close the descriptor, if our slot still names it: in a fork child the handler has closed it already, and the
    /// number may name another file by now.
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        slid_gate_take();
        if SLID_LOCK_FDS[self.slot].compare_exchange(self.fd, -1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            unsafe { libc::close(self.fd) };
        }
        slid_gate_give();
    }
}

/// Build the slid file for `m` if there is none: read the mapping from the cache file a chunk at a time, slide the
/// chunk in a buffer, append it and note its hash, append the record, then rename the whole file into place, so a
/// reader sees all of it or none. One builder per file at a time (a `BuildLock` on `<file>.lock`, waited for at most
/// `wait_ms`), in a directory of ours (made mode 0700 if missing).
#[allow(clippy::too_many_arguments)]
unsafe fn build_slid(fd: i32, m: &MappingSlideNp, info: &[u8], slide: u64, unslid_base: u64, path: &std::path::Path, id: &SlidId, uid: u32, wait_ms: u64) -> bool {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let Some(dir) = path.parent() else { return false };
    if info.len() < 24 || !crate::aot::owned_dir(dir, uid) {
        return false;
    }
    let Some(_lock) = BuildLock::take(&path.with_extension("lock"), wait_ms) else { return false };
    if std::fs::symlink_metadata(path).is_ok() {
        return true; // built by another process while we waited; mapping it checks it
    }
    let tmp = path.with_extension("tmp");
    let built = (|| -> Option<()> {
        let mut out = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o644).open(&tmp).ok()?;
        // 0644 past the umask: in root's shared store every uid must be able to read it.
        out.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o644)).ok()?;
        let page = rd32(info, 4) as usize;
        let count = rd32(info, 8) as usize;
        let size = m.size as usize;
        let mut buf = vec![0u8; SLID_CHUNK_PAGES * page];
        if page == 0 || id.chunk == 0 || buf.len() % id.chunk != 0 || size as u64 != id.size {
            return None;
        }
        let mut rec = id.header().to_vec();
        let mut off = 0usize;
        while off < size {
            let len = (size - off).min(buf.len());
            if !pread_all(fd, &mut buf[..len], m.file_offset + off as u64) {
                return None;
            }
            let first = off / page;
            let n = len.div_ceil(page).min(count.saturating_sub(first));
            if n > 0 {
                apply_slide_info(&sub_info(info, first, n)?, buf.as_mut_ptr() as u64, len as u64, slide, unslid_base).ok()?;
            }
            out.write_all(&buf[..len]).ok()?;
            for h in buf[..len].chunks(id.chunk) {
                rec.extend_from_slice(&sha256(h));
            }
            off += len;
        }
        let sum = sha256(&rec);
        rec.extend_from_slice(&sum);
        out.write_all(&rec).ok()?;
        out.sync_all().ok()?;
        std::fs::rename(&tmp, path).ok()
    })();
    if built.is_none() {
        let _ = std::fs::remove_file(&tmp);
    }
    built.is_some()
}

#[cfg(test)]
mod slid_tests {
    use super::*;

    const PAGE: usize = 0x4000;
    const PAGES: usize = 3;
    // One address per test, far above unhinted mappings and away from cache.rs's other tests.
    const MAP_AT: u64 = 0x2200_0000_0000;
    const MAP_AT_2: u64 = 0x2300_0000_0000;
    const MAP_AT_3: u64 = 0x2400_0000_0000;
    const MAP_AT_4: u64 = 0x2500_0000_0000;
    const MAP_AT_5: u64 = 0x2600_0000_0000;
    // The stamp tests: STAMP_AT + k * STAMP_STEP for the k-th, one each.
    const STAMP_AT: u64 = 0x3200_0000_0000;
    const STAMP_STEP: u64 = 0x10_0000_0000;
    const UNSLID: u64 = 0x1_8000_0000;
    const SLIDE: u64 = 0x4_8000_0000;
    // Two boots' sessions, as `boot_session` would give them.
    const BOOT: &[u8] = b"11111111-2222-3333-4444-555555555555";
    const BOOT_2: &[u8] = b"66666666-7777-8888-9999-000000000000";

    fn uid() -> u32 {
        unsafe { libc::geteuid() }
    }

    /// The directories a test made, removed when its thread ends, a panic included: libtest runs each test on a
    /// thread of its own, and a thread's locals are dropped as it ends, after unwinding (tests unwind whatever the
    /// profile's `panic` says). A forked child leaves with `_exit` and removes nothing. `the_suite_leaves_no_directory`
    /// checks it.
    struct Made(std::cell::RefCell<Vec<std::path::PathBuf>>);

    impl Drop for Made {
        fn drop(&mut self) {
            for d in self.0.borrow().iter() {
                let _ = std::fs::remove_dir_all(d);
            }
        }
    }

    thread_local! {
        static MADE: Made = const { Made(std::cell::RefCell::new(Vec::new())) };
    }

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mcl-slid-{name}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        MADE.with(|m| m.0.borrow_mut().push(d.clone()));
        d
    }

    /// R v3 registers its fork handlers when the image loads, so no `BuildLock::take` ever registers. A lazy
    /// registration races a fork: inside the gate, a child forked while the first builder blocked in
    /// `pthread_atfork` inherited a locked gate with no child handler and spun for ever; through `pthread_once`, it
    /// inherits a once owned by a vanished thread. The failure is a HANG, and the window is a process's FIRST
    /// `BuildLock::take`, so this runs racers as fresh child processes of its own, each with a deadline that kills
    /// its whole process group. MCL_SLID_FORK_RACE makes a process be one racer: a thread makes the first take while
    /// the main thread forks up to 48 children, each of which takes and drops a `BuildLock` of its own. Mutant m40
    /// restores the old shape (a Boolean marked, then `pthread_atfork`, under the gate) and must fail this.
    #[test]
    fn the_fork_handlers_are_registered_before_the_gate_is_taken() {
        if std::env::var_os("MCL_SLID_FORK_RACE").is_some() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static GO: AtomicBool = AtomicBool::new(false);
            static DONE: AtomicBool = AtomicBool::new(false);
            let d = dir("forkrace");
            let (lock, child_lock) = (d.join("r.lock"), d.join("c.lock"));
            let t = std::thread::spawn(move || {
                while !GO.load(Ordering::SeqCst) {
                    std::hint::spin_loop();
                }
                let _ = BuildLock::take(&lock, 0);
                DONE.store(true, Ordering::SeqCst);
            });
            let mut kids = Vec::new();
            GO.store(true, Ordering::SeqCst);
            while kids.len() < 48 && (kids.is_empty() || !DONE.load(Ordering::SeqCst)) {
                let c = unsafe { libc::fork() };
                if c == 0 {
                    let l = BuildLock::take(&child_lock, 0);
                    drop(l);
                    unsafe { libc::_exit(0) };
                }
                assert!(c > 0, "fork failed");
                kids.push(c);
            }
            let _ = t.join();
            for c in kids {
                let mut st = 0;
                let got = unsafe { libc::waitpid(c, &mut st, 0) };
                assert_eq!((got, st), (c, 0), "a fork child could not take and drop a BuildLock");
            }
            return;
        }
        let me = std::env::current_exe().unwrap();
        let (mut hung, mut failed) = (0, 0);
        for _ in 0..12 {
            // Its own process group, so the deadline can kill the whole tree. Killing the racer alone leaves the
            // fork child that inherited the locked gate spinning for ever, and those pile up across racers and
            // poison later timings -- which is the intended failure, so it is the likely case, not a corner.
            use std::os::unix::process::CommandExt;
            let mut cmd = std::process::Command::new(&me);
            cmd.args(["--exact", "cache::slid_tests::the_fork_handlers_are_registered_before_the_gate_is_taken", "--test-threads=1"])
                .env("MCL_SLID_FORK_RACE", "1")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setpgid(0, 0) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
            let mut ch = cmd.spawn().unwrap();
            let pgid = ch.id() as i32;
            let t0 = std::time::Instant::now();
            loop {
                match ch.try_wait().unwrap() {
                    Some(st) => {
                        if !st.success() {
                            failed += 1;
                        }
                        break;
                    }
                    None if t0.elapsed() > std::time::Duration::from_secs(20) => {
                        unsafe { libc::kill(-pgid, libc::SIGKILL) };
                        let _ = ch.kill();
                        let _ = ch.wait();
                        while unsafe { libc::waitpid(-pgid, std::ptr::null_mut(), libc::WNOHANG) } > 0 {}
                        // A killed racer cannot remove its directory; `dir` names it by the racer's pid.
                        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!("mcl-slid-forkrace-{pgid}")));
                        hung += 1;
                        break;
                    }
                    None => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
        }
        assert_eq!((hung, failed), (0, 0), "racers (killed at the 20 s deadline, exited non-zero), of 12");
    }

    /// Run by `the_suite_leaves_no_directory` alone, in a child with MCL_SLID_TEST_PANIC set: a test that fails after
    /// making its directory still removes it. Otherwise it only makes one.
    #[test]
    fn a_failing_test_removes_its_directory_too() {
        let d = dir("panic");
        std::fs::write(d.join("f"), b"x").unwrap();
        assert!(std::env::var_os("MCL_SLID_TEST_PANIC").is_none(), "failing on purpose");
    }

    /// Three 16 KiB pages and v5 slide info: page 0 a chain of two (0x10, then 0x20), page 1 none, page 2 one entry
    /// at its last 8 bytes, auth (no high byte kept).
    fn fixture() -> (Vec<u8>, Vec<u8>) {
        let mut data: Vec<u8> = (0..PAGES * PAGE).map(|i| (i * 7 + 3) as u8).collect();
        let put = |d: &mut Vec<u8>, at: usize, v: u64| d[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put(&mut data, 0x10, (2u64 << 52) | (0x12u64 << 34) | 0x1000); // next = 2 words, high8 0x12, target 0x1000
        put(&mut data, 0x20, 0x2000);
        put(&mut data, 2 * PAGE + PAGE - 8, (1u64 << 63) | 0x3000);
        let mut info = Vec::new();
        info.extend_from_slice(&5u32.to_le_bytes());
        info.extend_from_slice(&(PAGE as u32).to_le_bytes());
        info.extend_from_slice(&(PAGES as u32).to_le_bytes());
        info.extend_from_slice(&0u32.to_le_bytes());
        info.extend_from_slice(&UNSLID.to_le_bytes()); // value_add
        for s in [0x10u16, 0xFFFF, (PAGE - 8) as u16] {
            info.extend_from_slice(&s.to_le_bytes());
        }
        (data, info)
    }

    fn slid_in_place(data: &[u8], info: &[u8]) -> Vec<u8> {
        let mut d = data.to_vec();
        unsafe { apply_slide_info(info, d.as_mut_ptr() as u64, d.len() as u64, SLIDE, UNSLID).unwrap() };
        d
    }

    /// The cache file the fixture's mapping sits in: one page before it, so the file offset is not zero.
    fn source(d: &std::path::Path, data: &[u8]) -> (std::fs::File, MappingSlideNp) {
        use std::io::Write;
        let p = d.join("cache");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&vec![0u8; PAGE]).unwrap();
        f.write_all(data).unwrap();
        let m = MappingSlideNp { address: UNSLID, size: data.len() as u64, file_offset: PAGE as u64, slide_size: 1, slide_start: 0, max_prot: 3, init_prot: 3 };
        (std::fs::File::open(&p).unwrap(), m)
    }

    /// The fixture's identity, with one page per hash chunk, so it has three.
    fn ident(m: &MappingSlideNp, slide: u64) -> SlidId {
        SlidId { uuid: [0xab; 16], file_offset: m.file_offset, size: m.size, slide, chunk: PAGE }
    }

    /// A slid file built from the fixture in its own directory: (dir, file, identity, slid bytes).
    fn built(name: &str) -> (std::path::PathBuf, std::path::PathBuf, SlidId, Vec<u8>) {
        use std::os::unix::io::AsRawFd;
        let (data, info) = fixture();
        let d = dir(name);
        let (f, m) = source(&d, &data);
        let id = ident(&m, SLIDE);
        let path = d.join("x.data");
        assert!(unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 1000) });
        (d, path, id, slid_in_place(&data, &info))
    }

    /// Map `path` at `at` (over a fresh anonymous mapping that reads zero) and say what happened; unmaps after.
    fn try_map(at: u64, path: &std::path::Path, id: &SlidId, uid: u32, seed: usize) -> (Slid, u8) {
        try_map_in(at, path, id, uid, seed, Some(BOOT))
    }

    /// `try_map` in the boot `boot`, waiting up to 10 s for a process checking the file first.
    fn try_map_in(at: u64, path: &std::path::Path, id: &SlidId, uid: u32, seed: usize, boot: Option<&[u8]>) -> (Slid, u8) {
        try_map_waiting(at, path, id, uid, seed, boot, 10_000)
    }

    /// `try_map_in`, waiting up to `wait_ms` for a process checking the file first.
    fn try_map_waiting(at: u64, path: &std::path::Path, id: &SlidId, uid: u32, seed: usize, boot: Option<&[u8]>, wait_ms: u64) -> (Slid, u8) {
        let len = id.size as usize;
        unsafe {
            let p = libc::mmap(at as *mut _, len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0);
            assert_eq!(p as u64, at, "control: the range is ours");
            let r = map_slid(at, len, libc::PROT_READ, path, id, uid, seed, boot, wait_ms);
            let first = *(at as *const u8);
            libc::munmap(at as *mut _, len);
            (r, first)
        }
    }

    /// Leave the stamp a process would have left in `boot` had it checked every chunk of `path` as the file is now: the
    /// state after a change below the filesystem, which moves nothing a stamp names.
    fn stamp_as_is(path: &std::path::Path, id: &SlidId, boot: &[u8]) {
        let file = std::fs::read(path).unwrap();
        let sum = &file[file.len() - 32..];
        assert_eq!(file.len(), id.size as usize + id.record_len(), "control: the file is whole");
        assert!(stamp_write(&path.with_extension("stamp"), &stamp_bytes(boot, &std::fs::metadata(path).unwrap(), sum)));
    }

    /// `(inode, size, mtime, mtime_nsec)` of `path`.
    fn named(path: &std::path::Path) -> (u64, u64, i64, i64) {
        use std::os::unix::fs::MetadataExt;
        let md = std::fs::metadata(path).unwrap();
        (md.ino(), md.size(), md.mtime(), md.mtime_nsec())
    }

    fn flip(path: &std::path::Path, at: u64) {
        use std::os::unix::fs::FileExt;
        let f = std::fs::OpenOptions::new().read(true).write(true).open(path).unwrap();
        let mut b = [0u8; 1];
        f.read_exact_at(&mut b, at).unwrap();
        f.write_all_at(&[b[0] ^ 0x5a], at).unwrap();
    }

    fn chmod(p: &std::path::Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    fn chunked_slide_matches_in_place() {
        let (data, info) = fixture();
        let whole = slid_in_place(&data, &info);
        assert_ne!(whole, data, "control: the slide changes the data");
        for split in [1usize, 2] {
            let mut d = data.clone();
            for (first, n) in [(0, split), (split, PAGES - split)] {
                let sub = sub_info(&info, first, n).unwrap();
                let at = d[first * PAGE..].as_mut_ptr() as u64;
                unsafe { apply_slide_info(&sub, at, (n * PAGE) as u64, SLIDE, UNSLID).unwrap() };
            }
            assert_eq!(d, whole, "split after page {split}");
        }
        assert!(sub_info(&info, 2, 2).is_none(), "past the page count");
    }

    #[test]
    fn build_writes_the_slid_bytes_and_their_record_and_maps_them_copy_on_write() {
        let (_d, path, id, want) = built("build");
        let file = std::fs::read(&path).unwrap();
        assert_eq!(&file[..want.len()], &want[..]);
        assert_eq!(file.len(), want.len() + id.record_len());
        let rec = &file[want.len()..];
        assert_eq!(rec[..64], id.header());
        for c in 0..PAGES {
            assert_eq!(rec[64 + 32 * c..96 + 32 * c], sha256(&want[c * PAGE..(c + 1) * PAGE]), "chunk {c}'s hash");
        }
        assert!(!path.with_extension("tmp").exists(), "the temporary file is renamed, not left");
        unsafe {
            let p = libc::mmap(MAP_AT as *mut _, want.len(), libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(p as u64, MAP_AT, "control: the range starts free");
            assert_eq!(map_slid(MAP_AT, want.len(), libc::PROT_READ | libc::PROT_WRITE, &path, &id, uid(), 0, Some(BOOT), 1000), Slid::Mapped);
            assert_eq!(std::slice::from_raw_parts(MAP_AT as *const u8, want.len()), &want[..]);
            // Copy-on-write: a write by the process stays its own.
            *(MAP_AT as *mut u8) ^= 0xff;
            assert_eq!(std::fs::read(&path).unwrap()[0], want[0], "a write by the process stays private");
            libc::munmap(MAP_AT as *mut _, want.len());
        }
    }

    fn footprint() -> u64 {
        let mut ri: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let r = unsafe { libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V4, &mut ri as *mut _ as *mut libc::rusage_info_t) };
        assert_eq!(r, 0);
        ri.ri_phys_footprint
    }

    /// Touch every page of [at, at + len), reading only.
    unsafe fn read_all(at: u64, len: usize) {
        let mut sum = 0u64;
        for o in (0..len).step_by(4096) {
            sum = sum.wrapping_add(std::ptr::read_volatile((at + o as u64) as *const u8) as u64);
        }
        std::hint::black_box(sum);
    }

    /// The point of the patch: reading a mapped slid file costs no footprint, where the same bytes held privately do.
    #[test]
    fn reading_a_mapped_slid_file_costs_no_footprint() {
        use std::os::unix::io::AsRawFd;
        const LEN: usize = 16 << 20;
        let d = dir("cost");
        let data: Vec<u8> = (0..LEN).map(|i| (i / 4096) as u8 | 1).collect();
        let mut info = Vec::new(); // v5, 16 KiB pages, none with a rebase
        for v in [5u32, PAGE as u32, (LEN / PAGE) as u32, 0] {
            info.extend_from_slice(&v.to_le_bytes());
        }
        info.extend_from_slice(&UNSLID.to_le_bytes());
        for _ in 0..LEN / PAGE {
            info.extend_from_slice(&0xFFFFu16.to_le_bytes());
        }
        let (f, m) = source(&d, &data);
        drop(data);
        let id = SlidId { uuid: [1; 16], file_offset: m.file_offset, size: m.size, slide: SLIDE, chunk: PAGE * SLID_CHUNK_PAGES };
        let path = d.join("c.data");
        assert!(unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 1000) });
        unsafe {
            // Control: the same size, private and written, as the in-place slide leaves it.
            let c = libc::mmap(MAP_AT_4 as *mut _, LEN, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(c as u64, MAP_AT_4, "control: the range starts free");
            let before = footprint();
            for o in (0..LEN).step_by(4096) {
                *((MAP_AT_4 + o as u64) as *mut u8) = 1;
            }
            let private = footprint().saturating_sub(before);
            assert!(private >= (LEN as u64) * 3 / 4, "control: {private} bytes for {LEN} private bytes");
            libc::munmap(MAP_AT_4 as *mut _, LEN);
            let p = libc::mmap(MAP_AT_3 as *mut _, LEN, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(p as u64, MAP_AT_3, "control: the range starts free");
            let before = footprint();
            assert_eq!(map_slid(MAP_AT_3, LEN, libc::PROT_READ, &path, &id, uid(), 7, Some(BOOT), 1000), Slid::Mapped);
            read_all(MAP_AT_3, LEN);
            let shared = footprint().saturating_sub(before);
            assert!(shared < (LEN as u64) / 4, "checking, mapping and reading the slid file cost {shared} bytes of footprint");
            assert_eq!(*((MAP_AT_3 + 5 * 4096) as *const u8), 5 | 1);
            libc::munmap(MAP_AT_3 as *mut _, LEN);
        }
    }

    #[test]
    fn a_file_or_directory_not_ours_or_writable_by_others_is_refused() {
        let (d, path, id, want) = built("owner");
        assert_eq!(try_map(MAP_AT_2, &path, &id, uid(), 0), (Slid::Mapped, want[0]), "control: sound");
        let other = uid().wrapping_add(1);
        assert_eq!(try_map(MAP_AT_2, &path, &id, other, 0), (Slid::Refused, 0), "directory and file owned by another uid");
        for (mode, what) in [(0o664, "group-writable file"), (0o646, "other-writable file")] {
            chmod(&path, mode);
            assert_eq!(try_map(MAP_AT_2, &path, &id, uid(), 0), (Slid::Refused, 0), "{what}");
        }
        chmod(&path, 0o644);
        for (mode, what) in [(0o775, "group-writable directory"), (0o757, "other-writable directory")] {
            chmod(&d, mode);
            assert_eq!(try_map(MAP_AT_2, &path, &id, uid(), 0), (Slid::Refused, 0), "{what}");
        }
        chmod(&d, 0o700);
        assert_eq!(try_map(MAP_AT_2, &path, &id, uid(), 0), (Slid::Mapped, want[0]), "control: sound again");
        // The owner test proper: a file owned by root (another uid than ours, unless the tests run as root).
        let root = std::fs::metadata("/private/etc/hosts").unwrap();
        assert_eq!(crate::aot::owned(&root, uid()), uid() == 0, "a file owned by root is ours only if we are root");
        assert!(!crate::aot::owned(&std::fs::metadata(&path).unwrap(), other));
        // A symlink to a sound file, or a missing file.
        let link = d.join("l.data");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert_eq!(try_map(MAP_AT_2, &link, &id, uid(), 0).0, Slid::Refused, "a symlink");
        assert_eq!(try_map(MAP_AT_2, &d.join("none.data"), &id, uid(), 0).0, Slid::Missing);
        assert_eq!(try_map(MAP_AT_2, &d.join("no-dir/none.data"), &id, uid(), 0).0, Slid::Missing);
        // A sound file reached through a symlinked directory.
        let ldir = dir("owner-link");
        let lp = ldir.join("d");
        std::os::unix::fs::symlink(&d, &lp).unwrap();
        assert_eq!(try_map(MAP_AT_2, &lp.join("x.data"), &id, uid(), 0).0, Slid::Refused, "a symlinked directory");
    }

    #[test]
    fn a_short_or_long_file_is_refused() {
        let (_d, path, id, want) = built("short");
        let sound = std::fs::read(&path).unwrap();
        let mut long = sound.clone();
        long.push(0);
        for (bytes, what) in [(&sound[..sound.len() - 1], "short by a byte"), (&long[..], "long by a byte, record intact"), (&sound[..want.len()], "no record")] {
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(try_map(MAP_AT_2 + 0x1000_0000, &path, &id, uid(), 0), (Slid::Refused, 0), "{what}");
        }
        std::fs::write(&path, &sound).unwrap();
        assert_eq!(try_map(MAP_AT_2 + 0x1000_0000, &path, &id, uid(), 0), (Slid::Mapped, want[0]), "control: sound");
    }

    #[test]
    fn content_that_does_not_match_its_hash_is_refused() {
        // Seed c checks chunk c (and chunk 0): a byte changed in a checked chunk is refused, even with a stamp.
        assert_eq!(checked_chunks(PAGES, 2), vec![0, 2], "with SLID_CHECK_CHUNKS = 1, seed 2 checks chunks 0 and 2");
        for (c, at) in [(0usize, 0x30u64), (1, PAGE as u64 + 9), (2, 2 * PAGE as u64 + 100)] {
            let (_d, path, id, want) = built(&format!("hash{c}"));
            assert_eq!(try_map(MAP_AT_5, &path, &id, uid(), c), (Slid::Mapped, want[0]), "control: sound");
            flip(&path, at);
            stamp_as_is(&path, &id, BOOT);
            assert_eq!(try_map(MAP_AT_5, &path, &id, uid(), c), (Slid::Refused, 0), "a byte changed in chunk {c}");
        }
        // With no stamp every chunk is checked, so a change in a chunk the sample skips is seen too.
        let (_d, path, id, _) = built("hash-unchecked");
        flip(&path, PAGE as u64 + 9);
        assert_eq!(try_map(MAP_AT_5, &path, &id, uid(), 2), (Slid::Refused, 0), "with no stamp every chunk is checked: chunk 1, which seed 2's sample skips");
    }

    #[test]
    fn a_record_for_another_mapping_or_a_damaged_record_is_refused() {
        let (_d, path, id, want) = built("record");
        let len = want.len() as u64;
        assert_eq!(try_map(MAP_AT_5 + 0x1000_0000, &path, &id, uid(), 0).0, Slid::Mapped, "control: sound");
        for (other, what) in [
            (SlidId { slide: id.slide + 0x4000, ..id }, "another slide"),
            (SlidId { file_offset: id.file_offset + 0x4000, ..id }, "another place"),
            (SlidId { uuid: [0xcd; 16], ..id }, "another cache file"),
        ] {
            assert_eq!(try_map(MAP_AT_5 + 0x1000_0000, &path, &other, uid(), 0).0, Slid::Refused, "{what}");
        }
        flip(&path, len + 64 + 32 + 5); // chunk 1's hash
        assert_eq!(try_map(MAP_AT_5 + 0x1000_0000, &path, &id, uid(), 0).0, Slid::Refused, "a hash in the record changed");
    }

    #[test]
    fn checked_chunks_are_the_first_and_more_from_the_seed() {
        assert_eq!(checked_chunks(1, 5), vec![0]);
        assert_eq!(checked_chunks(2, 5), vec![0, 1]);
        let n = 100;
        let mut seen = std::collections::HashSet::new();
        for seed in 0..n {
            let v = checked_chunks(n, seed);
            assert_eq!(v.len(), 1 + SLID_CHECK_CHUNKS);
            assert_eq!(v[0], 0);
            assert!(v.iter().all(|&c| c < n));
            seen.extend(v);
        }
        assert_eq!(seen.len(), n, "over many processes every chunk is checked");
    }

    /// Each process checks as many different chunks as it says, whatever its seed: a seed that is a multiple of the
    /// chunk count must not check chunk 0 twice.
    #[test]
    fn a_sample_never_checks_a_chunk_twice() {
        for n in [3usize, 4, 5, 64, 100, 1888] {
            for seed in (0..3 * n).chain([usize::MAX, usize::MAX - 1]) {
                let v = checked_chunks(n, seed);
                let set: std::collections::HashSet<_> = v.iter().collect();
                assert_eq!(set.len(), v.len(), "n {n} seed {seed}: {v:?}");
                assert!(v.iter().all(|&c| c < n), "n {n} seed {seed}: {v:?}");
            }
            // The other tests choose a seed for the chunk it checks: seed c checks chunk c.
            for c in 1..n {
                assert_eq!(checked_chunks(n, c)[1], c, "n {n}: seed {c}");
            }
        }
    }

    #[test]
    fn a_missing_file_is_built_and_an_existing_one_never_rebuilt() {
        use std::os::unix::io::AsRawFd;
        let (data, info) = fixture();
        let d = dir("nobuild");
        let (f, m) = source(&d, &data);
        let id = ident(&m, SLIDE);
        let path = d.join("x.data");
        std::fs::write(&path, b"not slid").unwrap();
        assert!(unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 100) }, "present: leave it to the check");
        assert_eq!(std::fs::read(&path).unwrap(), b"not slid", "an existing file is not replaced");
        // A directory that is not ours is not built in.
        let d2 = dir("nobuild-dir");
        chmod(&d2, 0o777);
        let p2 = d2.join("y.data");
        assert!(!unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &p2, &id, uid(), 100) });
        assert!(!p2.exists());
        chmod(&d2, 0o700);
        assert!(unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &p2, &id, uid(), 100) }, "control: ours");
        // A directory reached through a symlink is not built in, even if it is ours.
        let d3 = dir("nobuild-link");
        let l3 = d3.join("d");
        std::os::unix::fs::symlink(dir("nobuild-target"), &l3).unwrap();
        assert!(!unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &l3.join("w.data"), &id, uid(), 100) });
        assert!(!l3.join("w.data").exists());
        // A missing directory is made, mode 0700.
        let p3 = d.join("new/z.data");
        assert!(unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &p3, &id, uid(), 100) });
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(d.join("new")).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn build_waits_for_another_builder_and_then_gives_up() {
        use std::os::unix::io::AsRawFd;
        let (data, info) = fixture();
        let d = dir("lock");
        let (f, m) = source(&d, &data);
        let id = ident(&m, SLIDE);
        let path = d.join("z.data");
        let held = BuildLock::take(&path.with_extension("lock"), 0).expect("free");
        assert!(!unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 100) });
        assert!(!path.exists() && !path.with_extension("tmp").exists());
        drop(held);
        assert!(unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 100) }, "control: free once released");
        assert!(path.exists());
    }

    #[test]
    fn a_fork_child_closes_its_copy_and_the_parent_keeps_the_lock() {
        use std::os::unix::ffi::OsStrExt;
        let _turn = Turn::take();
        let d = dir("forkkeep");
        let lock = d.join("k.lock");
        let held = BuildLock::take(&lock, 0).expect("free");
        // F_GETPATH answers with the real path, and $TMPDIR reaches /var through a symlink. Made
        // before the fork: a fork child of many threads may call only what a signal handler may.
        let want = std::ffi::CString::new(std::fs::canonicalize(&lock).unwrap().as_os_str().as_bytes()).unwrap();
        let mut p = [0i32; 2];
        assert_eq!(unsafe { libc::pipe(p.as_mut_ptr()) }, 0);
        let c = unsafe { libc::fork() };
        if c == 0 {
            // Every descriptor, not the one number the parent used: a copy at any other number would
            // keep the lock just as well. Stack buffers and libc only.
            let mut n = 0u8;
            for fd in 0..256i32 {
                let mut buf = [0i8; 1024];
                if unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_mut_ptr()) } == 0 && unsafe { libc::strcmp(buf.as_ptr(), want.as_ptr()) } == 0 {
                    n = n.saturating_add(1);
                }
            }
            unsafe {
                libc::write(p[1], [n].as_ptr().cast(), 1);
                libc::sleep(30);
                libc::_exit(0);
            }
        }
        assert!(c > 0);
        let mut b = [9u8; 1];
        let n = unsafe { libc::read(p[0], b.as_mut_ptr().cast(), 1) };
        let still = !lock_free(&lock);
        drop(held);
        let free = becomes_free(&lock, 2_000);
        let alive = unsafe { libc::kill(c, 0) } == 0;
        // BEFORE the reap: a SIGKILL takes the holder away, so after it holders() is empty whatever the truth
        // was, and m34 -- a real inherited lock -- printed that same empty answer.
        let who = if free { String::new() } else { holders(&lock) };
        reap(c);
        unsafe {
            libc::close(p[0]);
            libc::close(p[1]);
        }
        assert_eq!((n, b[0]), (1, 0), "the child has no descriptor of the lock, at any number");
        assert!(still, "the parent still holds the lock after the child closed its copy");
        assert!(alive, "control: the child runs on");
        assert!(free, "let go by the parent, it is free while the child runs on; open by: {who}");
    }

    /// The tests that fork or exec take this turn, one at a time: cargo runs every test in a thread of
    /// one process, so two of them at once makes each one's children inside the other's measurement,
    /// and a child that keeps a lock descriptor is the very thing they measure. An atomic rather than
    /// a `Mutex`, because a static std lock is one a fork child cannot reset; `Drop` gives the turn
    /// back, so a panic does not keep it.
    static FORK_TURN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    struct Turn;

    impl Turn {
        fn take() -> Turn {
            use std::sync::atomic::Ordering;
            while FORK_TURN.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
                std::thread::yield_now();
            }
            Turn
        }
    }

    impl Drop for Turn {
        fn drop(&mut self) {
            FORK_TURN.store(false, std::sync::atomic::Ordering::Release);
        }
    }

    /// Whether `lock` becomes free within `ms`. The property the three fork tests state is that a child does not
    /// KEEP the lock, which is a becomes and not an is: a release that lands a few milliseconds after the builder's
    /// thread is joined is still a release, and sampling once turns that into a red. Bounded, so a child that
    /// really does keep the lock -- which it would for its whole 30-second sleep -- still fails.
    fn becomes_free(lock: &std::path::Path, ms: u64) -> bool {
        let t0 = std::time::Instant::now();
        loop {
            if lock_free(lock) {
                return true;
            }
            if t0.elapsed() >= std::time::Duration::from_millis(ms) {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// Whether another open file description can take `lock` now; let go at once if it can.
    ///
    /// The whole probe -- the open, the lock, the unlock and the close -- is inside the gate, which
    /// the fork prepare handler also takes, so no child of this process can inherit this descriptor
    /// at all: a probe for a lock a child keeps must not be able to become one. Nothing between the
    /// take and the give may panic, because unwinding would never give the gate back, so the open's
    /// error is carried out and unwrapped after it.
    fn lock_free(lock: &std::path::Path) -> bool {
        use std::os::unix::io::AsRawFd;
        slid_gate_take();
        let opened = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(lock);
        let free = opened.as_ref().ok().map(|f| {
            let got = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0;
            if got {
                unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
            }
            got
        });
        drop(opened);
        slid_gate_give();
        free.expect("the probe opens the lock file")
    }

    /// For a failure message: the processes that have `p` open, as lsof sees them, with lsof's own exit status and
    /// standard error, and a CONTROL -- a file this process is holding open across the call, which lsof must report
    /// this pid for. An empty answer for `p` means "nothing holds it" only when the control came back; otherwise
    /// lsof is not answering and the reading is NO READING, never an absence. lsof exits 1 both for an error and
    /// for "found nothing", which is why the status alone cannot settle it. Control named without the `mcl-` prefix
    /// so that it is not counted as a leftover, and removed before returning.
    fn holders(p: &std::path::Path) -> String {
        let ctl = std::env::temp_dir().join(format!("lsof-control-{}", std::process::id()));
        let held = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&ctl);
        let look = |q: &std::path::Path| match std::process::Command::new("/usr/sbin/lsof").arg("-F").arg("pcfn").arg(q).output() {
            Ok(o) => (
                format!("{:?}", o.status.code()),
                String::from_utf8_lossy(&o.stdout).replace('\n', " "),
                String::from_utf8_lossy(&o.stderr).replace('\n', " "),
            ),
            Err(e) => ("spawn-failed".to_string(), String::new(), format!("{e}")),
        };
        let (trc, tout, terr) = look(p);
        let (crc, cout, cerr) = look(&ctl);
        let seen = held.is_ok() && cout.contains(&format!("p{}", std::process::id()));
        drop(held);
        let _ = std::fs::remove_file(&ctl);
        let verdict = if !seen {
            "NO READING: lsof did not report this process for a file it holds open, so an empty target is not an absence"
        } else if tout.trim().is_empty() {
            "NOTHING HOLDS IT, and the control was seen"
        } else {
            "HELD, see target"
        };
        format!("{verdict} | target rc={trc} out=[{tout}] err=[{terr}] | control rc={crc} out=[{cout}] err=[{cerr}]")
    }

    fn reap(pid: i32) {
        let mut st = 0;
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, &mut st, 0);
        }
    }

    /// A builder on a thread of its own, held inside its build with its lock taken: its temporary file is a FIFO, whose
    /// open waits for a reader. Returns the lock file, the FIFO and the thread once the lock is taken.
    fn held_build(name: &str) -> (std::path::PathBuf, std::path::PathBuf, std::thread::JoinHandle<bool>) {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::io::AsRawFd;
        let (data, info) = fixture();
        let d = dir(name);
        let (f, m) = source(&d, &data);
        let id = ident(&m, SLIDE);
        let path = d.join("h.data");
        let (lock, tmp) = (path.with_extension("lock"), path.with_extension("tmp"));
        let ct = std::ffi::CString::new(tmp.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(ct.as_ptr(), 0o600) }, 0);
        let t = std::thread::spawn(move || unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 10_000) });
        let t0 = std::time::Instant::now();
        while lock_free(&lock) {
            assert!(t0.elapsed() < std::time::Duration::from_secs(10), "the builder never took its lock");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        (lock, tmp, t)
    }

    /// Let a held builder finish: read its FIFO to the end, and wait for it to return.
    fn finish_build(tmp: &std::path::Path, t: std::thread::JoinHandle<bool>) {
        use std::io::Read;
        let mut v = Vec::new();
        std::fs::File::open(tmp).unwrap().read_to_end(&mut v).unwrap();
        let _ = t.join().unwrap();
    }

    #[test]
    fn a_child_forked_while_a_builder_holds_its_lock_does_not_keep_it() {
        let _turn = Turn::take();
        let (lock, tmp, t) = held_build("forkchild");
        let c = unsafe { libc::fork() };
        if c == 0 {
            unsafe {
                libc::sleep(30);
                libc::_exit(0);
            }
        }
        assert!(c > 0);
        finish_build(&tmp, t);
        let free = becomes_free(&lock, 2_000);
        let alive = unsafe { libc::kill(c, 0) } == 0;
        // BEFORE the reap: a SIGKILL takes the holder away, so after it holders() is empty whatever the truth
        // was, and m34 -- a real inherited lock -- printed that same empty answer.
        let who = if free { String::new() } else { holders(&lock) };
        reap(c);
        let after = lock_free(&lock);
        assert!(alive && after, "controls: the child ran on, and once it is gone the lock is free");
        assert!(free, "the builder let its lock go, and a child forked while it held it does not keep it; open by: {who}");
    }

    #[test]
    fn a_program_spawned_while_a_builder_holds_its_lock_does_not_keep_it() {
        let _turn = Turn::take();
        let (lock, tmp, t) = held_build("execchild");
        let (prog, arg) = (c"/bin/sleep", c"30");
        let argv = [prog.as_ptr() as *mut libc::c_char, arg.as_ptr() as *mut libc::c_char, std::ptr::null_mut()];
        let envp = [std::ptr::null_mut::<libc::c_char>()];
        let mut pid = 0;
        // posix_spawn, with no fork: the program keeps exactly what exec keeps, every descriptor not close-on-exec.
        let r = unsafe { libc::posix_spawn(&mut pid, prog.as_ptr(), std::ptr::null(), std::ptr::null(), argv.as_ptr(), envp.as_ptr()) };
        finish_build(&tmp, t);
        assert_eq!(r, 0, "posix_spawn");
        let free = becomes_free(&lock, 2_000);
        let alive = unsafe { libc::kill(pid, 0) } == 0;
        // BEFORE the reap: a SIGKILL takes the holder away, so after it holders() is empty whatever the truth
        // was, and m34 -- a real inherited lock -- printed that same empty answer.
        let who = if free { String::new() } else { holders(&lock) };
        reap(pid);
        let after = lock_free(&lock);
        assert!(alive && after, "controls: the program ran on, and once it is gone the lock is free");
        assert!(free, "the builder let its lock go, and a program spawned while it held it does not keep it; open by: {who}");
    }

    #[test]
    fn a_short_cache_file_builds_nothing() {
        use std::os::unix::io::AsRawFd;
        let (data, info) = fixture();
        let d = dir("shortsrc");
        let (f, mut m) = source(&d, &data);
        m.size += (PAGE * 64) as u64; // past the end of the file: the second chunk cannot be read
        let id = SlidId { chunk: PAGE * SLID_CHUNK_PAGES, ..ident(&m, SLIDE) };
        let path = d.join("s.data");
        assert!(!unsafe { build_slid(f.as_raw_fd(), &m, &info, SLIDE, UNSLID, &path, &id, uid(), 100) });
        assert!(!path.exists() && !path.with_extension("tmp").exists());
    }

    #[test]
    fn the_name_carries_the_uuid_the_place_and_the_slide() {
        use std::io::Write;
        use std::os::unix::io::AsRawFd;
        let (_, info) = fixture();
        let d = dir("name");
        let p = d.join("hdr");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&[0u8; 0x58]).unwrap();
        f.write_all(&[0xab; 16]).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let m = MappingSlideNp { address: 0, size: 0x8000, file_offset: 0x4000, slide_size: 1, slide_start: 0, max_prot: 3, init_prot: 3 };
        let (a, id) = slid_path(&d, f.as_raw_fd(), &m, &info, 0x10).unwrap();
        let name = a.file_name().unwrap().to_str().unwrap().to_string();
        assert!(name.starts_with("dsc-slid2-") && name.contains(&"ab".repeat(16)) && name.contains("-4000-8000-10."), "{name}");
        assert_eq!((id.uuid, id.file_offset, id.size, id.slide, id.chunk), ([0xab; 16], 0x4000, 0x8000, 0x10, SLID_HASH_CHUNK));
        assert_ne!(Some(a), slid_path(&d, f.as_raw_fd(), &m, &info, 0x20).map(|x| x.0));
        let zero = d.join("zero");
        std::fs::write(&zero, [0u8; 0x80]).unwrap();
        let zf = std::fs::File::open(&zero).unwrap();
        assert!(slid_path(&d, zf.as_raw_fd(), &m, &info, 0x10).is_none(), "no UUID, no shared file");
        assert!(slid_path(&d, -1, &m, &info, 0x10).is_none());
        assert_eq!(slid_path(&d.join("s"), f.as_raw_fd(), &m, &info, 0x10).unwrap().0.parent(), Some(d.join("s").as_path()), "the file is in the store it is given");
    }

    // The shared store tests: one address each, SHARE_AT + k * SHARE_STEP for the k-th.
    const SHARE_AT: u64 = 0x3400_0000_0000;
    const SHARE_STEP: u64 = 0x10_0000_0000;

    /// The fixture's cache file with a UUID in its header, as `slid_path` needs: the mapping one page in, as `source`.
    fn source_with_uuid(d: &std::path::Path, data: &[u8]) -> (std::fs::File, MappingSlideNp) {
        use std::io::Write;
        let p = d.join("cache-uuid");
        let mut f = std::fs::File::create(&p).unwrap();
        let mut head = vec![0u8; PAGE];
        head[0x58..0x68].copy_from_slice(&[0xcd; 16]);
        f.write_all(&head).unwrap();
        f.write_all(data).unwrap();
        let m = MappingSlideNp { address: UNSLID, size: data.len() as u64, file_offset: PAGE as u64, slide_size: 1, slide_start: 0, max_prot: 3, init_prot: 3 };
        (std::fs::File::open(&p).unwrap(), m)
    }

    /// `share_slid_in` over a fresh anonymous mapping at `at` that reads zero: (what it said, the mapping's first 16
    /// bytes); unmaps after. The fixture's first slid byte differs from 0, so a mapping that took reads non-zero.
    fn try_share(at: u64, store: &SlidStore, f: &std::fs::File, m: &MappingSlideNp, info: &[u8]) -> (bool, Vec<u8>) {
        use std::os::unix::io::AsRawFd;
        let len = m.size as usize;
        unsafe {
            let p = libc::mmap(at as *mut _, len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0);
            assert_eq!(p as u64, at, "control: the range is ours");
            let r = share_slid_in(store, f.as_raw_fd(), m, info, SLIDE, UNSLID, at);
            let first = std::slice::from_raw_parts(at as *const u8, 16).to_vec();
            libc::munmap(at as *mut _, len);
            (r, first)
        }
    }

    /// The `.data` files under `d`, at any depth.
    fn slid_files(d: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut todo = vec![d.to_path_buf()];
        while let Some(x) = todo.pop() {
            for e in std::fs::read_dir(&x).into_iter().flatten().flatten() {
                let p = e.path();
                if e.file_type().is_ok_and(|t| t.is_dir()) {
                    todo.push(p);
                } else if p.extension().is_some_and(|x| x == "data") {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    }

    /// Every entry directly in `d`, by name.
    fn names(d: &std::path::Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }

    /// Puts a directory's mode back when the test ends, a panic included, so `dir`'s cleanup can remove it.
    struct Mode(std::path::PathBuf, u32);

    impl Drop for Mode {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(self.1));
        }
    }

    #[test]
    fn the_shared_store_is_named_by_the_environment_or_turned_off_by_it() {
        assert_eq!(shared_store_path(None), Some(std::path::PathBuf::from(SHARED_STORE)));
        assert_eq!(shared_store_path(Some("".into())), None, "empty turns it off");
        assert_eq!(shared_store_path(Some("/x/y".into())), Some(std::path::PathBuf::from("/x/y")));
    }

    /// Which store a process uses: the shared one only when it is a real directory of its owner that no one else can
    /// write; then only the owner builds there. A missing or untrustworthy one leaves the per-user store as before.
    #[test]
    fn the_shared_store_is_used_only_when_its_owner_alone_can_write_it() {
        let me = uid();
        let d = dir("store-pick");
        let home = d.join("home");
        let shared = d.join("shared");
        let old = SlidStore { dir: home.clone(), owner: me, build: true };
        assert_eq!(slid_store_in(None, me, home.clone(), me), old, "no shared store: the per-user store, as before");
        assert_eq!(slid_store_in(Some(&shared), me, home.clone(), me), old, "a shared store that does not exist: the same");
        std::fs::create_dir(&shared).unwrap();
        chmod(&shared, 0o755);
        assert_eq!(slid_store_in(Some(&shared), me, home.clone(), me), SlidStore { dir: shared.clone(), owner: me, build: true }, "its owner builds there");
        assert_eq!(slid_store_in(Some(&shared), me, home.clone(), me + 1), SlidStore { dir: shared.clone(), owner: me, build: false }, "anyone else only reads");
        chmod(&shared, 0o555);
        assert_eq!(slid_store_in(Some(&shared), me, home.clone(), me + 1).dir, shared, "read-only to its owner too: still the store");
        for (what, mode) in [("group-writable", 0o775), ("other-writable", 0o757)] {
            chmod(&shared, mode);
            assert_eq!(slid_store_in(Some(&shared), me, home.clone(), me), old, "{what}: not trusted");
            assert_eq!(slid_store_in(Some(&shared), me, home.clone(), me + 1), SlidStore { dir: home.clone(), owner: me + 1, build: true }, "{what}: not trusted by another uid either");
        }
        chmod(&shared, 0o755);
        assert_eq!(slid_store_in(Some(&shared), me + 1, home.clone(), me), old, "owned by someone else than the owner named: not trusted");
        let link = d.join("link");
        std::os::unix::fs::symlink(&shared, &link).unwrap();
        assert_eq!(slid_store_in(Some(&link), me, home.clone(), me), old, "reached through a symlink: not trusted");
        let file = d.join("file");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(slid_store_in(Some(&file), me, home.clone(), me), old, "not a directory: not trusted");
    }

    /// ⭐ What this change is for. The control: with no shared store, two processes of one uid with two HOMEs (two
    /// sandbox containers) each keep a slid copy of the same mapping. With one, its owner builds the copy once, a process
    /// that may not build maps that copy, and neither writes a slid copy under its own HOME.
    #[test]
    fn with_a_shared_store_one_slid_copy_serves_every_home_and_none_is_written_there() {
        let me = uid();
        let (data, info) = fixture();
        let want = slid_in_place(&data, &info);
        let d = dir("share");
        let (f, m) = source_with_uuid(&d, &data);
        let homes: Vec<std::path::PathBuf> = (1..=4).map(|i| d.join(format!("home{i}/Library/Caches/Maclator/aot"))).collect();
        // The control, the old behaviour: no shared store.
        for (k, h) in homes[..2].iter().enumerate() {
            let s = slid_store_in(None, 0, h.clone(), me);
            assert_eq!(try_share(SHARE_AT + k as u64 * SHARE_STEP, &s, &f, &m, &info), (true, want[..16].to_vec()), "control: home {k} maps");
        }
        let copies = slid_files(&d);
        assert_eq!(copies.len(), 2, "control: one copy per HOME without a shared store: {copies:?}");
        assert_eq!(std::fs::read(&copies[0]).unwrap(), std::fs::read(&copies[1]).unwrap(), "control: and the two are the same bytes");
        // The change: a shared store, built by its owner.
        let shared = d.join("shared");
        std::fs::create_dir(&shared).unwrap();
        chmod(&shared, 0o755);
        let owner = slid_store_in(Some(&shared), me, homes[2].clone(), me);
        assert!(owner.build && owner.dir == shared, "control: the owner builds in the shared store");
        assert_eq!(try_share(SHARE_AT + 2 * SHARE_STEP, &owner, &f, &m, &info), (true, want[..16].to_vec()), "the owner builds and maps");
        {
            use std::os::unix::fs::PermissionsExt;
            let modes: Vec<(String, u32)> = names(&shared).into_iter().filter(|n| n.ends_with(".data") || n.ends_with(".stamp")).map(|n| { let md = std::fs::metadata(shared.join(&n)).unwrap(); (n, md.permissions().mode() & 0o777) }).collect();
            assert_eq!(modes.len(), 2, "the owner's copy and its stamp: {modes:?}");
            assert!(modes.iter().all(|(_, md)| *md == 0o644), "both readable by every uid, writable by the owner alone, whatever the umask: {modes:?}");
        }
        // Another process: the store read-only to it, as root's is to every other uid, and no stamp to trust.
        let _back = Mode(shared.clone(), 0o755);
        for n in names(&shared) {
            if n.ends_with(".stamp") {
                std::fs::remove_file(shared.join(n)).unwrap();
            }
        }
        chmod(&shared, 0o555);
        let before = names(&shared);
        let reader = slid_store_in(Some(&shared), me, homes[3].clone(), me + 1);
        assert!(!reader.build && reader.dir == shared, "control: the other process only reads");
        assert_eq!(try_share(SHARE_AT + 3 * SHARE_STEP, &reader, &f, &m, &info), (true, want[..16].to_vec()), "it maps the owner's copy");
        assert_eq!(names(&shared), before, "it wrote nothing in the shared store (no lock, stamp or file)");
        assert!(!d.join("home3").exists() && !d.join("home4").exists(), "and nothing under either HOME");
        let all = slid_files(&d);
        assert_eq!(all.len(), 3, "the two control copies and the one shared copy, no more: {all:?}");
        assert_eq!(all.iter().filter(|p| p.starts_with(&shared)).count(), 1, "one copy in the shared store");
    }

    /// A builder whose umask is 077 (a daemon's may be) still leaves the slid file and its stamp readable by every uid,
    /// or root's shared copy would serve root alone. The umask is the process's, so the build runs in a child of its own.
    #[test]
    fn the_shared_copy_and_its_stamp_are_readable_by_all_whatever_the_umask() {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        if std::env::var_os("MCL_SLID_UMASK").is_some() {
            unsafe { libc::umask(0o077) };
            let (data, info) = fixture();
            let d = dir("umask");
            let (f, m) = source_with_uuid(&d, &data);
            let shared = d.join("shared");
            std::fs::create_dir(&shared).unwrap();
            chmod(&shared, 0o755);
            let owner = slid_store_in(Some(&shared), uid(), d.join("home"), uid());
            assert!(try_share(SHARE_AT + 8 * SHARE_STEP, &owner, &f, &m, &info).0, "the owner builds and maps");
            let probe = d.join("probe");
            std::fs::OpenOptions::new().write(true).create_new(true).mode(0o644).open(&probe).unwrap();
            assert_eq!(std::fs::metadata(&probe).unwrap().permissions().mode() & 0o777, 0o600, "control: the umask applies here");
            for n in names(&shared).into_iter().filter(|n| n.ends_with(".data") || n.ends_with(".stamp")) {
                let md = std::fs::metadata(shared.join(&n)).unwrap().permissions().mode() & 0o777;
                assert_eq!(md, 0o644, "{n}");
            }
            assert_eq!(slid_files(&shared).len(), 1, "control: one copy was built");
            return;
        }
        let me = std::env::current_exe().unwrap();
        let st = std::process::Command::new(&me)
            .args(["--exact", "cache::slid_tests::the_shared_copy_and_its_stamp_are_readable_by_all_whatever_the_umask", "--test-threads=1"])
            .env("MCL_SLID_UMASK", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(st.success(), "the child under umask 077: {st}");
    }

    /// A process that may not build, finding no file in the shared store or not allowed to read it (as a sandbox
    /// refuses), slides privately: false, and nothing written anywhere, the shared store or its HOME.
    #[test]
    fn a_reader_that_finds_no_file_or_may_not_read_it_slides_privately_and_writes_nothing() {
        let me = uid();
        let (data, info) = fixture();
        let want = slid_in_place(&data, &info);
        let d = dir("share-deny");
        let (f, m) = source_with_uuid(&d, &data);
        let shared = d.join("shared");
        std::fs::create_dir(&shared).unwrap();
        chmod(&shared, 0o755);
        let home = d.join("home/Library/Caches/Maclator/aot");
        let reader = slid_store_in(Some(&shared), me, home.clone(), me + 1);
        assert!(!reader.build, "control: a reader");
        assert_eq!(try_share(SHARE_AT + 4 * SHARE_STEP, &reader, &f, &m, &info), (false, vec![0u8; 16]), "missing: slid privately, the mapping untouched");
        assert!(names(&shared).is_empty() && !d.join("home").exists(), "and nothing written: {:?}", names(&shared));
        let owner = slid_store_in(Some(&shared), me, home.clone(), me);
        assert_eq!(try_share(SHARE_AT + 5 * SHARE_STEP, &owner, &f, &m, &info), (true, want[..16].to_vec()), "control: the owner builds it");
        let file = slid_files(&shared).pop().expect("the owner's copy");
        if me == 0 {
            eprintln!("as root a mode-0 file is still readable, so the refused-read half is not tested here");
            return;
        }
        let _back = Mode(file.clone(), 0o644);
        chmod(&file, 0o000);
        let before = names(&shared);
        assert_eq!(try_share(SHARE_AT + 6 * SHARE_STEP, &reader, &f, &m, &info), (false, vec![0u8; 16]), "not readable: slid privately");
        assert_eq!(names(&shared), before, "nothing added to the shared store");
        assert!(!d.join("home").exists(), "and no copy under its HOME");
        // A shared store whose directory this process may not look at at all (a sandbox's EPERM): still the store.
        let _back_dir = Mode(shared.clone(), 0o755);
        chmod(&shared, 0o000);
        let blind = slid_store_in(Some(&shared.join("sub")), me, home.clone(), me + 1);
        assert_eq!(blind, SlidStore { dir: shared.join("sub"), owner: me, build: false }, "a store it may not look at is not swapped for its HOME");
        assert_eq!(try_share(SHARE_AT + 7 * SHARE_STEP, &blind, &f, &m, &info), (false, vec![0u8; 16]), "and it slides privately");
        assert!(!d.join("home").exists(), "with no copy under its HOME");
    }

    /// The stamp tests change chunk 1, which the sample of seed 2 (chunks 0 and 2) skips.
    const UNSAMPLED: u64 = PAGE as u64 + 9;

    #[test]
    fn a_file_whose_every_chunk_matches_gets_a_stamp() {
        use std::os::unix::fs::PermissionsExt;
        let (_d, path, id, want) = built("stamp-left");
        let stamp = path.with_extension("stamp");
        assert!(!stamp.exists(), "control: building leaves no stamp");
        assert_eq!(try_map(STAMP_AT, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "control: sound");
        let md = std::fs::symlink_metadata(&stamp).expect("a file whose every chunk matches gets a stamp");
        assert!(md.is_file() && md.permissions().mode() & 0o777 == 0o644, "the stamp is a file all can read and only we can write: {:o}", md.permissions().mode());
        let file = std::fs::read(&path).unwrap();
        assert_eq!(std::fs::read(&stamp).unwrap(), stamp_bytes(BOOT, &std::fs::metadata(&path).unwrap(), &file[file.len() - 32..]), "it names the boot, the file and its record");
        assert!(!path.with_extension("stamp-tmp").exists(), "the temporary file is renamed, not left");
    }

    #[test]
    fn a_stamp_that_matches_is_trusted_and_only_the_sample_is_checked() {
        let at = STAMP_AT + STAMP_STEP;
        let (_d, path, id, want) = built("stamp-trusted");
        flip(&path, UNSAMPLED);
        stamp_as_is(&path, &id, BOOT);
        // The limit, by design: a change that moves nothing the stamp names, in a chunk the sample skips.
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "a stamp that matches is trusted: only the sample is checked");
        assert_eq!(try_map(at, &path, &id, uid(), 1).0, Slid::Refused, "control: a sample holding chunk 1 sees the change");
    }

    #[test]
    fn a_stamp_left_in_another_boot_is_not_trusted() {
        let at = STAMP_AT + 2 * STAMP_STEP;
        let (_d, path, id, want) = built("stamp-boot");
        flip(&path, UNSAMPLED);
        stamp_as_is(&path, &id, BOOT);
        assert_eq!(try_map_in(at, &path, &id, uid(), 2, Some(BOOT)), (Slid::Mapped, want[0]), "control: in its own boot the stamp is trusted");
        assert_eq!(try_map_in(at, &path, &id, uid(), 2, Some(BOOT_2)), (Slid::Refused, 0), "a stamp left in another boot is not trusted: every chunk is checked");
    }

    #[test]
    fn with_no_boot_session_every_chunk_is_checked_and_no_stamp_is_left() {
        let at = STAMP_AT + 3 * STAMP_STEP;
        let (_d, path, id, want) = built("stamp-noboot");
        assert_eq!(try_map_in(at, &path, &id, uid(), 2, None), (Slid::Mapped, want[0]), "control: sound");
        assert!(!path.with_extension("stamp").exists(), "with no boot session no stamp is left");
        flip(&path, UNSAMPLED);
        assert_eq!(try_map_in(at, &path, &id, uid(), 2, None), (Slid::Refused, 0), "with no boot session every chunk is checked");
    }

    #[test]
    fn a_file_replaced_by_one_of_the_same_size_and_mtime_is_checked_whole() {
        let at = STAMP_AT + 4 * STAMP_STEP;
        let (d, path, id, want) = built("stamp-replaced");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "control: sound, and stamped");
        let before = named(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[UNSAMPLED as usize] ^= 0x5a;
        let new = d.join("x.new");
        std::fs::write(&new, &bytes).unwrap();
        chmod(&new, 0o644);
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::File::options().write(true).open(&new).unwrap().set_modified(mtime).unwrap();
        std::fs::rename(&new, &path).unwrap();
        let after = named(&path);
        assert!(after.0 != before.0 && (after.1, after.2, after.3) == (before.1, before.2, before.3), "control: another inode, the same size and mtime: {before:?} {after:?}");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "a file replaced by one of the same size and mtime: every chunk is checked");
    }

    #[test]
    fn a_file_written_in_place_is_checked_whole() {
        let at = STAMP_AT + 5 * STAMP_STEP;
        let (_d, path, id, want) = built("stamp-written");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "control: sound, and stamped");
        let before = named(&path);
        std::thread::sleep(std::time::Duration::from_millis(5));
        flip(&path, UNSAMPLED);
        let after = named(&path);
        assert!((after.0, after.1) == (before.0, before.1) && (after.2, after.3) != (before.2, before.3), "control: the same inode and size, a new mtime: {before:?} {after:?}");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "a file written in place since its stamp: every chunk is checked");
    }

    #[test]
    fn a_record_rewritten_in_place_with_its_mtime_kept_is_checked_whole() {
        use std::os::unix::fs::FileExt;
        let at = STAMP_AT + 6 * STAMP_STEP;
        let (_d, path, id, want) = built("stamp-record");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "control: sound, and stamped");
        let before = named(&path);
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        // A record whole in itself, whose hash for chunk 1 no longer matches chunk 1.
        let n = id.chunks();
        let mut rec = std::fs::read(&path).unwrap()[want.len()..].to_vec();
        rec[64 + 32] ^= 0x5a;
        let sum = sha256(&rec[..64 + 32 * n]);
        rec[64 + 32 * n..].copy_from_slice(&sum);
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all_at(&rec, want.len() as u64).unwrap();
        f.set_modified(mtime).unwrap();
        drop(f);
        assert_eq!(named(&path), before, "control: the inode, size and mtime the stamp names");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "a record rewritten with its mtime kept: every chunk is checked");
    }

    #[test]
    fn a_stamp_only_we_can_write_is_trusted_and_no_other() {
        let at = STAMP_AT + 7 * STAMP_STEP;
        let (d, path, id, want) = built("stamp-owner");
        let stamp = path.with_extension("stamp");
        flip(&path, UNSAMPLED);
        stamp_as_is(&path, &id, BOOT);
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "control: a stamp of ours that only we can write is trusted");
        // Each refusal below comes from checking every chunk, which leaves a refusal; it is taken away after each, so
        // the next step is a check of the stamp again and not an early refusal.
        let refusal = path.with_extension("refused");
        for (mode, what) in [(0o620, "a group-writable stamp"), (0o602, "an other-writable stamp")] {
            chmod(&stamp, mode);
            assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "{what} is not trusted");
            assert!(std::fs::remove_file(&refusal).is_ok(), "{what}: control: every chunk was checked, so a refusal was left");
        }
        stamp_as_is(&path, &id, BOOT);
        let real = d.join("elsewhere");
        std::fs::rename(&stamp, &real).unwrap();
        std::os::unix::fs::symlink(&real, &stamp).unwrap();
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "a symlink to a stamp that matches is not trusted");
        assert!(refusal.exists(), "control: every chunk was checked, so a refusal was left");
    }

    /// The refusal a process leaves when it checked every chunk of `path` in `boot` and one did not match.
    fn refusal_as_is(path: &std::path::Path, boot: &[u8]) -> Vec<u8> {
        let file = std::fs::read(path).unwrap();
        let stamp = stamp_bytes(boot, &std::fs::metadata(path).unwrap(), &file[file.len() - 32..]);
        [&SLID_REFUSED_MAGIC[..], &stamp[SLID_STAMP_MAGIC.len()..]].concat()
    }

    #[test]
    fn a_file_refused_when_every_chunk_was_checked_is_refused_at_once_for_the_rest_of_the_boot() {
        use std::os::unix::fs::PermissionsExt;
        let at = STAMP_AT + 13 * STAMP_STEP;
        let (_d, path, id, _) = built("refused-left");
        let refusal = path.with_extension("refused");
        flip(&path, UNSAMPLED);
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "control: with no stamp every chunk is checked");
        let md = std::fs::symlink_metadata(&refusal).expect("a file refused when every chunk was checked gets a refusal");
        assert!(md.is_file() && md.permissions().mode() & 0o777 == 0o644, "the refusal is a file all can read and only we can write: {:o}", md.permissions().mode());
        assert_eq!(std::fs::read(&refusal).unwrap(), refusal_as_is(&path, BOOT), "it names the boot, the file and its record");
        // Another process holds the file's lock now, longer than this one waits: without the refusal this start would
        // wait the whole 2 s for it and then check every chunk again.
        let held = BuildLock::take(&path.with_extension("lock"), 2_000).expect("free once the first check let it go");
        let t0 = std::time::Instant::now();
        let got = try_map_waiting(at, &path, &id, uid(), 2, Some(BOOT), 2_000);
        let took = t0.elapsed();
        drop(held);
        assert_eq!(got, (Slid::Refused, 0), "a refused file stays refused");
        assert!(took < std::time::Duration::from_millis(1_000), "the refusal is trusted at once, without waiting for the lock: {took:?}");
    }

    #[test]
    fn a_refusal_is_trusted_only_in_its_own_boot_and_for_the_file_as_it_names_it() {
        let at = STAMP_AT + 14 * STAMP_STEP;
        let (_d, path, id, want) = built("refused-scope");
        flip(&path, UNSAMPLED);
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "control: with no stamp every chunk is checked");
        // The byte put back below what the refusal names: the same inode, size, modification time and record.
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        flip(&path, UNSAMPLED);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(mtime).unwrap();
        assert_eq!(std::fs::read(path.with_extension("refused")).unwrap(), refusal_as_is(&path, BOOT), "control: the refusal names the file as it is now");
        // The limit, by design: in its own boot the refusal is trusted, though the file is sound again.
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "in its own boot a refusal is trusted");
        assert_eq!(try_map_in(at, &path, &id, uid(), 2, Some(BOOT_2)), (Slid::Mapped, want[0]), "a refusal left in another boot is not trusted: every chunk is checked");
        std::thread::sleep(std::time::Duration::from_millis(5));
        flip(&path, UNSAMPLED);
        flip(&path, UNSAMPLED);
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "a file written since its refusal is checked whole again");
    }

    #[test]
    fn a_process_waiting_for_the_check_trusts_the_refusal_it_left() {
        let at = STAMP_AT + 15 * STAMP_STEP;
        // A sound file, so a process that checked every chunk itself would map it: only the refusal refuses it.
        let (_d, path, id, _) = built("refused-race");
        let lock = path.with_extension("lock");
        let first = BuildLock::take(&lock, 2_000).expect("free once the builder let it go");
        let p = path.clone();
        let second = std::thread::spawn(move || try_map(at, &p, &id, uid(), 2));
        let went = goes_for_the_lock(&lock);
        // What the first process leaves when a chunk did not match (here as if the file changed back after its check).
        assert!(stamp_write(&path.with_extension("refused"), &refusal_as_is(&path, BOOT)));
        drop(first);
        let got = second.join().unwrap();
        assert!(went, "control: it found no stamp and went for the lock");
        assert_eq!(got, (Slid::Refused, 0), "a process that waited for the first one's check trusts the refusal it left");
    }

    #[test]
    fn processes_racing_to_check_first_check_once_and_the_second_trusts_the_first_ones_stamp() {
        let at = STAMP_AT + 8 * STAMP_STEP;
        let (_d, path, id, want) = built("stamp-race");
        flip(&path, UNSAMPLED);
        // The first process, checking every chunk: it holds the file's lock. Waited for, as `becomes_free` does: the
        // builder has just let it go, and a release can land a few milliseconds late.
        let lock = path.with_extension("lock");
        let first = BuildLock::take(&lock, 2_000).expect("free once the builder let it go");
        let p = path.clone();
        let second = std::thread::spawn(move || try_map(at, &p, &id, uid(), 2));
        let waiting = goes_for_the_lock(&lock) && !second.is_finished();
        // What the first process leaves once every chunk matched (here as if the change came after its check).
        stamp_as_is(&path, &id, BOOT);
        drop(first);
        let got = second.join().unwrap();
        assert!(waiting, "a process going second waits for the first one's check rather than checking at the same time");
        assert_eq!(got, (Slid::Mapped, want[0]), "and then trusts the stamp the first one left, checking only the sample");
    }

    #[test]
    fn a_process_that_gives_up_waiting_still_trusts_a_stamp_left_meanwhile() {
        let at = STAMP_AT + 12 * STAMP_STEP;
        let (_d, path, id, want) = built("stamp-gave-up");
        flip(&path, UNSAMPLED);
        let lock = path.with_extension("lock");
        let first = BuildLock::take(&lock, 2_000).expect("free once the builder let it go");
        let p = path.clone();
        // It waits 2 s, and the first process holds the lock longer than that.
        let second = std::thread::spawn(move || try_map_waiting(at, &p, &id, uid(), 2, Some(BOOT), 2_000));
        let went = goes_for_the_lock(&lock);
        stamp_as_is(&path, &id, BOOT);
        let got = second.join().unwrap();
        drop(first);
        assert!(went, "control: it found no stamp and went for the lock");
        assert_eq!(got, (Slid::Mapped, want[0]), "a process that gives up waiting still trusts a stamp left meanwhile");
    }

    /// Whether a second descriptor of `lock` opens in this process within 10 s, beside the one the test holds: a
    /// process that finds no stamp goes for the file's lock, and that is its descriptor. F_GETPATH answers with the
    /// real path, and $TMPDIR reaches /var through a symlink.
    fn goes_for_the_lock(lock: &std::path::Path) -> bool {
        use std::os::unix::ffi::OsStrExt;
        let want = std::ffi::CString::new(std::fs::canonicalize(lock).unwrap().as_os_str().as_bytes()).unwrap();
        let open = || {
            (0..unsafe { libc::getdtablesize() })
                .filter(|&fd| {
                    let mut buf = [0i8; 1024];
                    unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_mut_ptr()) == 0 && libc::strcmp(buf.as_ptr(), want.as_ptr()) == 0 }
                })
                .count()
        };
        let t0 = std::time::Instant::now();
        while open() < 2 {
            if t0.elapsed() > std::time::Duration::from_secs(10) {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        true
    }

    #[test]
    fn a_stamp_is_renamed_into_place_and_a_symlink_at_its_name_is_not_followed() {
        let at = STAMP_AT + 9 * STAMP_STEP;
        let (d, path, id, want) = built("stamp-rename");
        let stamp = path.with_extension("stamp");
        let target = d.join("target");
        std::fs::write(&target, b"keep").unwrap();
        std::os::unix::fs::symlink(&target, &stamp).unwrap();
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Mapped, want[0]), "control: sound");
        assert_eq!(std::fs::read(&target).unwrap(), b"keep", "a symlink at the stamp's name is not followed");
        assert!(std::fs::symlink_metadata(&stamp).unwrap().is_file(), "it is replaced by the stamp");
        assert!(!path.with_extension("stamp-tmp").exists(), "the temporary file is renamed, not left");
    }

    #[test]
    fn a_change_the_sample_finds_removes_the_stamp() {
        let at = STAMP_AT + 10 * STAMP_STEP;
        let (_d, path, id, _) = built("stamp-removed");
        flip(&path, UNSAMPLED);
        stamp_as_is(&path, &id, BOOT);
        assert_eq!(try_map(at, &path, &id, uid(), 1), (Slid::Refused, 0), "control: the sample of seed 1 holds chunk 1");
        assert!(!path.with_extension("stamp").exists(), "a change the sample finds removes the stamp");
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "so a process whose sample skips the change checks every chunk, and refuses too");
    }

    #[test]
    fn a_file_with_a_chunk_that_does_not_match_gets_no_stamp() {
        let at = STAMP_AT + 11 * STAMP_STEP;
        let (_d, path, id, _) = built("stamp-none");
        flip(&path, UNSAMPLED);
        assert_eq!(try_map(at, &path, &id, uid(), 2), (Slid::Refused, 0), "control: with no stamp every chunk is checked");
        assert!(!path.with_extension("stamp").exists(), "a file with a chunk that does not match gets no stamp");
    }

    #[test]
    fn the_boot_session_is_this_boots() {
        let a = boot_session().expect("the kernel says which boot this is");
        assert_eq!(boot_session().as_deref(), Some(&a[..]), "control: it does not change within a boot");
        let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", "kern.bootsessionuuid"]).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim().as_bytes(), &a[..], "the same as sysctl(8) says");
    }
}

#[cfg(test)]
mod slid_cleanup_tests {
    /// Run this test binary's `cache::slid_tests` in a child whose TMPDIR is a directory of its own, then count the
    /// mcl-slid-* directories left there: none, with 8 threads and with 1, and none after a test that fails.
    #[test]
    fn the_suite_leaves_no_directory() {
        let me = std::env::current_exe().unwrap();
        let tmp = std::env::temp_dir().join(format!("mcl-cleanup-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let left = || std::fs::read_dir(&tmp).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("mcl-slid-")).count();
        let run = |args: &[&str], panic: bool| {
            let mut c = std::process::Command::new(&me);
            c.args(args).env("TMPDIR", &tmp).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
            if panic {
                c.env("MCL_SLID_TEST_PANIC", "1");
            }
            let ok = c.status().unwrap().success();
            (ok, left())
        };
        std::fs::create_dir(tmp.join("mcl-slid-control")).unwrap();
        let seen = left();
        std::fs::remove_dir(tmp.join("mcl-slid-control")).unwrap();
        let r8 = run(&["--test-threads=8", "cache::slid_tests::"], false);
        let r1 = run(&["--test-threads=1", "cache::slid_tests::"], false);
        let made = {
            let mut c = std::process::Command::new(&me);
            c.args(["--exact", "cache::slid_tests::a_failing_test_removes_its_directory_too", "--test-threads=1"]).env("TMPDIR", &tmp);
            c.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
            c.status().unwrap().success()
        };
        let failed = run(&["--exact", "cache::slid_tests::a_failing_test_removes_its_directory_too"], true);
        let _ = std::fs::remove_dir_all(&tmp);
        // What this test is for is the COUNT. Whether the child suite passed is not its subject, and asserting it
        // made any flake in the three fork tests fail this test too, at about three times the rate (it runs the
        // suite twice). The pass is reported in each message instead. The controls that keep it non-vacuous stay:
        // `seen` proves the count catches a directory left on purpose, and the two --exact runs below exercise one
        // named test each, so no flaky sibling can reach them.
        assert_eq!(seen, 1, "control: the count sees a directory left there");
        assert_eq!(r8.1, 0, "the suite with 8 threads left directories behind (it passed: {})", r8.0);
        assert_eq!(r1.1, 0, "the suite with 1 thread left directories behind (it passed: {})", r1.0);
        assert!(made, "control: the one failing-test case passes when it is not told to fail");
        assert_eq!(failed, (false, 0), "a test that failed: (passed, directories left)");
    }
}

/// Keep `hostsys` referenced for callers that map via raw syscalls.
#[allow(dead_code)]
fn _unused() {
    let _ = hostsys::CLASS_UNIX;
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEN: u64 = 0x8000;
    const UNSLID: u64 = 0x1_8000_0000;
    // Far above where unhinted mappings land, and one address per case: cargo runs this beside other tests' mappings.
    const FREE_AT: u64 = 0x2000_0000_0000;
    const OCCUPIED_AT: u64 = 0x2100_0000_0000;

    /// A one-region cache: one file with fd -1, whose bytes are copied from `src`, `len` bytes at UNSLID.
    fn map_test_cache(hint: u64, src: &[u8], len: u64) -> (i64, Option<MappedCache>) {
        let files = [SharedFileNp { fd: -1, mappings_count: 1, slide: 0 }];
        let maps = [MappingSlideNp { address: UNSLID, size: len, file_offset: src.as_ptr() as u64, slide_size: 0, slide_start: 0, max_prot: 3, init_prot: 3 }];
        *CACHE.lock().unwrap() = None;
        let rc = map_and_slide_at(hint, 1, files.as_ptr() as u64, 1, maps.as_ptr() as u64);
        (rc, CACHE.lock().unwrap().take())
    }

    // One test, three cases in turn: CACHE is process-wide, and a static lock to take turns would be one more
    // std lock in a static, which the fork-safety scan rightly refuses.
    #[test]
    fn the_cache_reservation() {
        let src: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
        unsafe {
            // The negative control: with its base free, the cache lands exactly there and does not slide.
            let (rc, got) = map_test_cache(FREE_AT, &src, LEN);
            assert_eq!(rc, 0, "free base: the cache maps");
            let c = got.expect("free base: the cache is registered");
            assert_eq!(c.base, FREE_AT, "free base: it was not used");
            assert_eq!(c.slide, FREE_AT.wrapping_sub(UNSLID));
            assert_eq!(std::slice::from_raw_parts(c.base as *const u8, LEN as usize), &src[..]);
            libc::munmap(c.base as *mut _, LEN as usize);

            // Its base occupied: the cache slides to where the kernel put the reservation, 16 KiB-aligned, and
            // what held the base is left alone.
            // 4 KiB, so the kernel's next free address is not 16 KiB-aligned and the round-up is exercised.
            let occ = libc::mmap(OCCUPIED_AT as *mut _, 0x1000, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(occ as u64, OCCUPIED_AT, "control: the occupied base starts free");
            *(occ as *mut u8) = 0x5a;
            let (rc, got) = map_test_cache(OCCUPIED_AT, &src, LEN);
            assert_eq!(rc, 0, "occupied base: the cache maps");
            let c = got.expect("occupied base: the cache is registered");
            assert_ne!(c.base, OCCUPIED_AT, "occupied base: the cache was mapped over what holds it");
            assert_eq!(c.base % PAGE16, 0, "occupied base: a slid cache keeps 16 KiB alignment");
            assert_eq!(c.slide, c.base.wrapping_sub(UNSLID), "occupied base: slide = base - unslid base");
            assert_eq!(*(occ as *const u8), 0x5a, "occupied base: what held it was overwritten");
            assert_eq!(std::slice::from_raw_parts(c.base as *const u8, LEN as usize), &src[..]);
            libc::munmap(c.base as *mut _, LEN as usize);
            libc::munmap(occ, 0x1000);

            // reserve_cache itself: the range it returns is still mapped (a reservation given back, as the old
            // per-region hint did, would leave every MAP_FIXED that follows over unmapped space), and a base it
            // had to round up is 16 KiB-aligned. A hint lands on a free address and moves off a mapped one, so a
            // probe at each end says whether the range is still held (mincore cannot tell: it measured 0 either way).
            let occ = libc::mmap(OCCUPIED_AT as *mut _, 0x1000, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(occ as u64, OCCUPIED_AT, "control: the occupied base starts free again");
            let b = reserve_cache(OCCUPIED_AT, LEN).expect("reserve_cache");
            assert_ne!(b, OCCUPIED_AT);
            assert_eq!(b % PAGE16, 0, "reserve_cache: a rounded-up base is not 16 KiB-aligned");
            for at in [b, b + LEN - 0x1000] {
                let q = libc::mmap(at as *mut _, 0x1000, libc::PROT_NONE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
                assert!(q != libc::MAP_FAILED);
                libc::munmap(q, 0x1000);
                assert_ne!(q as u64, at, "reserve_cache: the range it returned is not mapped at {at:#x}");
            }
            libc::munmap(b as *mut _, LEN as usize);
            libc::munmap(occ, 0x1000);

            // With a boot record, a second mapping of the cache (a
            // second process, its hint elsewhere) lands at the first one's base.
            let d = boot_dir("wired");
            TEST_BOOT.with(|t| *t.borrow_mut() = Some((d.clone(), "B00T-W".to_string())));
            let (rc, got) = map_test_cache(FREE_AT, &src, LEN);
            assert_eq!(rc, 0, "boot base: the first mapping maps");
            let first = got.expect("boot base: the first mapping is registered").base;
            assert_eq!(first, FREE_AT, "control: the first mapping lands at its free hint");
            libc::munmap(first as *mut _, LEN as usize);
            let (rc, got) = map_test_cache(OCCUPIED_AT, &src, LEN);
            assert_eq!(rc, 0, "boot base: the second mapping maps");
            let second = got.expect("boot base: the second mapping is registered");
            assert_eq!(second.base, first, "boot base: map_and_slide_at did not take this boot's recorded base");
            assert_eq!(std::slice::from_raw_parts(second.base as *const u8, LEN as usize), &src[..]);
            libc::munmap(second.base as *mut _, LEN as usize);
            TEST_BOOT.with(|t| *t.borrow_mut() = None);
            let _ = std::fs::remove_dir_all(&d);

            // A reservation the kernel cannot make fails the call with ENOMEM and registers nothing.
            let (rc, got) = map_test_cache(FREE_AT, &src, 1 << 60);
            assert_eq!(rc, libc::ENOMEM as i64, "failed reservation: the call did not fail");
            assert!(got.is_none(), "failed reservation: a cache was registered without one");
        }
    }

    // A reservation that fails for a reason other than ENOMEM reports that reason, not ENOMEM (the kernel's mmap
    // gave ENOMEM for every oversized request measured on macOS 27.0.1, arm64, so the errno is set directly here).
    // A trim that fails, at the head (an unaligned hint, so the kernel's address needs rounding up) or at the tail:
    // reserve_cache returns that errno and nothing it reserved stays mapped. The probe is a second hint at the same
    // address, which lands there only if the whole over-reservation was released.
    #[test]
    fn a_failed_trim_releases_the_reservation_and_keeps_its_errno() {
        const TRIM_AT: u64 = 0x2700_0000_0000;
        let hint = TRIM_AT + 0x1000;
        unsafe {
            for (which, nth) in [("head", 1u32), ("tail", 2)] {
                let probe = libc::mmap(hint as *mut _, (LEN + PAGE16) as usize, libc::PROT_NONE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
                assert_eq!(probe as u64, hint, "control: the {which} range starts free");
                libc::munmap(probe, (LEN + PAGE16) as usize);
                TRIM_FAIL.with(|c| c.set((0, nth)));
                let r = reserve_cache(hint, LEN);
                let tried = TRIM_FAIL.with(|c| c.get()).0;
                TRIM_FAIL.with(|c| c.set((0, 0)));
                let e = r.expect_err(&format!("a failed {which} trim was ignored"));
                assert_eq!(e.raw_os_error(), Some(libc::EINVAL), "a failed {which} trim lost its errno");
                assert_eq!(tried, nth, "the {which} trim was never attempted, or trimming went on after it failed");
                let again = libc::mmap(hint as *mut _, (LEN + PAGE16) as usize, libc::PROT_NONE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
                libc::munmap(again, (LEN + PAGE16) as usize);
                assert_eq!(again as u64, hint, "after a failed {which} trim the reservation was not released");
            }
        }
    }

    /// A fresh store directory of ours, mode 0700, under $TMPDIR; the test removes it.
    fn boot_dir(tag: &str) -> std::path::PathBuf {
        use std::os::unix::fs::DirBuilderExt;
        let d = std::env::temp_dir().join(format!("mcl-boot-base-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::DirBuilder::new().mode(0o700).create(&d).expect("boot dir");
        d
    }

    // A second process of the same boot takes the first one's base, even with its own hint elsewhere.
    #[test]
    fn a_second_process_of_the_boot_gets_the_same_base() {
        const FIRST: u64 = 0x2800_0000_0000;
        const SECOND: u64 = 0x2900_0000_0000;
        let d = boot_dir("same");
        unsafe {
            let b1 = reserve_boot_base_in(&d, "B00T-1", FIRST, LEN).expect("first process");
            assert_eq!(b1, FIRST, "control: the first process lands at its free hint");
            assert_eq!(read_boot_base(&d, "B00T-1", LEN), Some(b1), "the first process recorded no base for its boot");
            libc::munmap(b1 as *mut _, LEN as usize); // the first process is gone
            let c = reserve_cache(SECOND, LEN).expect("control");
            assert_eq!(c, SECOND, "control: without a record the second hint lands at its own address");
            libc::munmap(c as *mut _, LEN as usize);
            let b2 = reserve_boot_base_in(&d, "B00T-1", SECOND, LEN).expect("second process");
            assert_eq!(b2, b1, "the second process of the boot did not get the recorded base");
            libc::munmap(b2 as *mut _, LEN as usize);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    // The recorded base occupied in this process: it is left alone, this process slides alone, the record stays.
    #[test]
    fn an_occupied_boot_base_falls_back_to_a_slide_of_its_own() {
        const AT: u64 = 0x2a00_0000_0000;
        let d = boot_dir("occupied");
        write_boot_base(&d, "B00T-2", LEN, AT);
        unsafe {
            let occ = libc::mmap(AT as *mut _, 0x4000, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(occ as u64, AT, "control: the recorded base starts free");
            *(occ as *mut u8) = 0x5a;
            let b = reserve_boot_base_in(&d, "B00T-2", AT, LEN).expect("an occupied boot base must fall back, not fail");
            // Read through the fault-tolerant path: a range replaced and made PROT_NONE must fail this assertion,
            // not crash the test.
            let mut held = [0u8; 1];
            let readable = crate::guestmem::read(occ as u64, &mut held);
            assert!(readable && held[0] == 0x5a, "occupied boot base: what held it was overwritten (readable {readable}, byte {:#x})", held[0]);
            assert_ne!(b, AT, "occupied boot base: the cache was put over what holds it");
            assert_eq!(b % PAGE16, 0, "occupied boot base: the fallback is not 16 KiB-aligned");
            assert_eq!(read_boot_base(&d, "B00T-2", LEN), Some(AT), "occupied boot base: the record was changed");
            libc::munmap(b as *mut _, LEN as usize);
            libc::munmap(occ, 0x4000);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    // A platform binary's hint is ignored, but the first process of the boot still takes the free preferred base
    // exactly (fixed), and records it; a base the kernel placed is never recorded.
    #[test]
    fn the_first_process_takes_the_preferred_base_exactly_when_its_hint_is_ignored() {
        const AT: u64 = 0x2d00_0000_0000;
        const BUSY: u64 = 0x2e00_0000_0000;
        let d = boot_dir("exact");
        HINT_IGNORED.with(|h| h.set(true));
        unsafe {
            let c = reserve_cache(AT, LEN).expect("control");
            assert_ne!(c, AT, "control: with the hint ignored, the kernel places the reservation elsewhere");
            libc::munmap(c as *mut _, LEN as usize);
            let b = reserve_boot_base_in(&d, "B00T-X", AT, LEN).expect("first process");
            assert_eq!(b, AT, "the first process did not take the free preferred base exactly");
            assert_eq!(read_boot_base(&d, "B00T-X", LEN), Some(AT), "the preferred base was not recorded");
            libc::munmap(b as *mut _, LEN as usize);
            // The preferred base occupied in the first process of another boot: it slides, and records nothing.
            let occ = libc::mmap(BUSY as *mut _, 0x4000, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            assert_eq!(occ as u64, BUSY, "control: the busy base starts free");
            let b = reserve_boot_base_in(&d, "B00T-Y", BUSY, LEN).expect("a busy preferred base must slide, not fail");
            assert_ne!(b, BUSY);
            assert_eq!(read_boot_base(&d, "B00T-Y", LEN), None, "a base the kernel placed was recorded for the boot");
            libc::munmap(b as *mut _, LEN as usize);
            libc::munmap(occ, 0x4000);
        }
        HINT_IGNORED.with(|h| h.set(false));
        let _ = std::fs::remove_dir_all(&d);
    }

    // A record that is not exactly ours, for this boot and this span, is not read.
    #[test]
    fn a_malformed_boot_record_is_ignored() {
        const AT: u64 = 0x2f00_0000_0000;
        let d = boot_dir("malformed");
        let path = boot_base_path(&d, "B00T-M", LEN);
        std::fs::write(&path, boot_base_record("B00T-M", LEN, AT)).unwrap();
        assert_eq!(read_boot_base(&d, "B00T-M", LEN), Some(AT), "control: a well-formed record reads");
        let bad = [
            ("magic", format!("MCLBASE2 B00T-M {LEN:x} {AT:x}\n")),
            ("span", format!("MCLBASE1 B00T-M {:x} {AT:x}\n", LEN + 0x4000)),
            ("unaligned", format!("MCLBASE1 B00T-M {LEN:x} {:x}\n", AT + 0x1000)),
            ("zero", format!("MCLBASE1 B00T-M {LEN:x} 0\n")),
            ("extra field", format!("MCLBASE1 B00T-M {LEN:x} {AT:x} 1\n")),
        ];
        for (what, text) in bad {
            std::fs::write(&path, text).unwrap();
            assert_eq!(read_boot_base(&d, "B00T-M", LEN), None, "a record with a bad {what} was read");
        }
        std::fs::write(&path, boot_base_record("B00T-M", LEN, AT)).unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
            assert_eq!(read_boot_base(&d, "B00T-M", LEN), None, "a record others can write was read");
            std::fs::remove_file(&path).unwrap();
            let real = d.join("real");
            std::fs::write(&real, boot_base_record("B00T-M", LEN, AT)).unwrap();
            std::os::unix::fs::symlink(&real, &path).unwrap();
            assert_eq!(read_boot_base(&d, "B00T-M", LEN), None, "a symlinked record was read");
        }
        unsafe {
            assert!(reserve_exact(AT + 0x1000, LEN).is_err(), "reserve_exact took an unaligned base");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    // A record from another boot, free and well formed, is never taken, and this boot gets a record of its own.
    #[test]
    fn a_record_from_another_boot_is_ignored() {
        const OLD: u64 = 0x2b00_0000_0000;
        const NEW: u64 = 0x2c00_0000_0000;
        let d = boot_dir("stale");
        write_boot_base(&d, "OLD-B00T", LEN, OLD);
        assert_eq!(read_boot_base(&d, "OLD-B00T", LEN), Some(OLD), "control: the other boot's record reads back");
        // The same record copied under this boot's name still names the other boot inside, so it is refused too.
        std::fs::write(boot_base_path(&d, "NEW-B00T", LEN + 0x4000), boot_base_record("OLD-B00T", LEN + 0x4000, OLD)).unwrap();
        assert_eq!(read_boot_base(&d, "NEW-B00T", LEN + 0x4000), None, "a record naming another boot inside was read");
        unsafe {
            let b = reserve_boot_base_in(&d, "NEW-B00T", NEW, LEN).expect("this boot");
            assert_ne!(b, OLD, "a record from another boot was taken");
            assert_eq!(b, NEW, "control: with no record of its own boot it lands at its hint");
            assert_eq!(read_boot_base(&d, "NEW-B00T", LEN), Some(NEW), "this boot recorded no base of its own");
            libc::munmap(b as *mut _, LEN as usize);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_reservation_failure_keeps_its_errno() {
        assert_eq!(reservation_errno(&std::io::Error::from_raw_os_error(libc::EINVAL)), libc::EINVAL as i64);
        assert_eq!(reservation_errno(&std::io::Error::from_raw_os_error(libc::EACCES)), libc::EACCES as i64);
        assert_eq!(reservation_errno(&std::io::Error::other("no errno")), libc::ENOMEM as i64);
    }
}
