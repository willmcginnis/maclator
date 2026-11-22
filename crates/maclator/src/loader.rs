//! Process image setup, doing what the XNU exec path does for an arm64
//! process: map the main executable and dyld, build the initial stack
//! (mach header, argc/argv/envp/apple strings) and point the CPU at dyld.

use crate::macho::MachO;
use maclator_core::cpu::Cpu;
use std::ffi::CString;
use std::os::unix::io::AsRawFd;

/// Fixed load addresses keep AOT translations position-stable.
pub const MAIN_BASE: u64 = 0x5_0000_0000;
pub const DYLD_BASE: u64 = 0x5_8000_0000;
pub const STACK_SIZE: u64 = 8 << 20;

pub struct Image {
    pub path: String,
    pub macho: MachO,
    pub slide: u64,
    pub base: u64,
}

pub struct Loaded {
    pub main: Image,
    pub dyld: Image,
    pub entry: u64,
    pub sp: u64,
    pub stack_lo: u64,
    pub stack_hi: u64,
}

fn map_image(path: &str, macho: &MachO, want_base: u64) -> std::io::Result<Image> {
    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();
    let (lo, hi) = macho.vm_range();
    // Reserve the whole range first so segments land contiguously. The preferred base is a HINT, not a demand
    // (no MAP_FIXED): reserve_image places an image that may slide wherever the kernel offers when the base is taken.
    let reserved_base = unsafe { reserve_image(path, macho, want_base, hi - lo)? };
    let slide = reserved_base.wrapping_sub(lo);
    // On any failure below, give the whole reservation back: every segment is mapped MAP_FIXED inside it,
    // so one munmap releases the segments already mapped as well.
    let release = || unsafe {
        libc::munmap(reserved_base as *mut _, (hi - lo) as usize);
    };
    for seg in &macho.segments {
        if seg.name == "__PAGEZERO" || seg.vmsize == 0 {
            continue;
        }
        let addr = seg.vmaddr.wrapping_add(slide);
        // Guest code is never executed natively; map R/RW only.
        let prot = (seg.initprot & 3) as i32;
        unsafe {
            if seg.filesize > 0 {
                let p = libc::mmap(addr as *mut _, seg.filesize as usize, libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_FIXED, fd, (macho.slice_offset + seg.fileoff) as i64);
                if p as u64 != addr {
                    let e = std::io::Error::last_os_error();
                    release();
                    return Err(std::io::Error::other(format!("{path}: mapping {} failed: {e}", seg.name)));
                }
            }
            let file_end = (addr + seg.filesize + 0xfff) & !0xfff;
            let seg_end = addr + seg.vmsize;
            if seg_end > file_end {
                let p = libc::mmap(file_end as *mut _, (seg_end - file_end) as usize, libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0);
                if p as u64 != file_end {
                    let e = std::io::Error::last_os_error();
                    release();
                    return Err(std::io::Error::other(format!("{path}: zerofill {} failed: {e}", seg.name)));
                }
            }
            if seg.filesize & 0xfff != 0 && seg.vmsize > seg.filesize {
                // zero the tail of the last file page
                let tail = addr + seg.filesize;
                let end = ((tail + 0xfff) & !0xfff).min(seg_end);
                std::ptr::write_bytes(tail as *mut u8, 0, (end - tail) as usize);
            }
            libc::mprotect(addr as *mut _, seg.vmsize as usize, if prot == 0 { libc::PROT_READ } else { prot });
        }
    }
    let base = macho.text_vmaddr().wrapping_add(slide);
    Ok(Image { path: path.to_string(), macho: macho.clone(), slide, base })
}

/// Give back an image's whole mapped range: its reservation and every segment in it.
fn unmap_image(img: &Image) {
    let (lo, hi) = img.macho.vm_range();
    unsafe { libc::munmap(lo.wrapping_add(img.slide) as *mut _, (hi - lo) as usize) };
}

/// arm64 macOS pages are 16 KiB; the host's are 4 KiB.
const PAGE16: u64 = 0x4000;
/// The arm64 shared region's size (SHARED_REGION_SIZE_ARM64, XNU osfmk/mach/shared_region.h). The shared cache is
/// mapped from crate::cache::CACHE_BASE when that is free, so an image kept out of this window leaves the cache its base.
const CACHE_WINDOW: u64 = 0x1_8000_0000;

fn in_cache_window(start: u64, size: u64) -> bool {
    let (cs, ce) = (crate::cache::CACHE_BASE, crate::cache::CACHE_BASE + CACHE_WINDOW);
    start < ce && start.saturating_add(size) > cs
}

/// Reserve [base, base + size), PROT_NONE, and return base. The preferred base is a hint. If the kernel declines it,
/// an image that may slide goes where the kernel offers instead, rounded up to a 16 KiB boundary by over-reserving one
/// page and trimming, as syscalls.rs::mmap_16k does, so its slide keeps the guest's page alignment. A range in the
/// shared-cache window is never returned: it is retried once above the window, and otherwise refused.
unsafe fn reserve_image(path: &str, macho: &MachO, want: u64, size: u64) -> std::io::Result<u64> {
    let anon = libc::MAP_ANON | libc::MAP_PRIVATE;
    let p = libc::mmap(want as *mut _, size as usize, libc::PROT_NONE, anon, -1, 0);
    if p == libc::MAP_FAILED {
        return Err(std::io::Error::other(format!(
            "{path}: reserving {size:#x} bytes for base {want:#x} failed: {}", std::io::Error::last_os_error())));
    }
    let got = p as u64;
    if got == want && !in_cache_window(got, size) {
        return Ok(got);
    }
    if got != want && std::env::var_os("MACLATOR_TRACE").is_some() {
        trace_declined_hint(path, want, size, got);
    }
    libc::munmap(p, size as usize);
    if !macho.may_slide() {
        return Err(std::io::Error::other(format!(
            "{path}: address {want:#x} unavailable and the image may not slide (not PIE, not dyld; got {got:#x})")));
    }
    let mut hint = want;
    for _ in 0..2 {
        let q = libc::mmap(hint as *mut _, (size + PAGE16) as usize, libc::PROT_NONE, anon, -1, 0);
        if q == libc::MAP_FAILED {
            return Err(std::io::Error::other(format!(
                "{path}: reserving {size:#x} bytes near {hint:#x} failed: {}", std::io::Error::last_os_error())));
        }
        let q = q as u64;
        let aligned = (q + PAGE16 - 1) & !(PAGE16 - 1);
        if aligned > q {
            libc::munmap(q as *mut _, (aligned - q) as usize);
        }
        let tail = q + size + PAGE16 - (aligned + size);
        if tail > 0 {
            libc::munmap((aligned + size) as *mut _, tail as usize);
        }
        if !in_cache_window(aligned, size) {
            eprintln!("[maclator] base {want:#x} unavailable, loading at {aligned:#x} instead");
            return Ok(aligned);
        }
        libc::munmap(aligned as *mut _, size as usize);
        hint = crate::cache::CACHE_BASE + CACHE_WINDOW;
    }
    Err(std::io::Error::other(format!(
        "{path}: no room for {size:#x} bytes outside the shared-cache window [{:#x}, {:#x})",
        crate::cache::CACHE_BASE, crate::cache::CACHE_BASE + CACHE_WINDOW)))
}

/// Under MACLATOR_TRACE, when the kernel declines a preferred base: what was wanted, what came back, and this
/// process's own mappings within 4 GiB of the wanted range (a mach_vm_region walk, with each region's protection and
/// user tag). The region the kernel just reserved for us is labelled as ours; any other region overlapping the wanted
/// range is a holder.
fn trace_declined_hint(path: &str, want: u64, size: u64, got: u64) {
    extern "C" {
        static mach_task_self_: u32;
        fn mach_vm_region(task: u32, address: *mut u64, size: *mut u64, flavor: i32, info: *mut u32, count: *mut u32, object_name: *mut u32) -> i32;
    }
    const VM_REGION_EXTENDED_INFO: i32 = 13;
    const WINDOW: u64 = 1 << 32;
    const MAX_REGIONS: usize = 64;
    let want_end = want.saturating_add(size);
    eprintln!("[maclator] trace: {path}: hint [{want:#x}, {want_end:#x}) declined; the kernel returned {got:#x}");
    let task = unsafe { mach_task_self_ };
    let mut addr = want.saturating_sub(WINDOW);
    let stop = want_end.saturating_add(WINDOW);
    let mut shown = 0;
    while shown < MAX_REGIONS {
        let (mut len, mut obj) = (0u64, 0u32);
        let mut info = [0u32; 16];
        let mut count = info.len() as u32;
        let kr = unsafe { mach_vm_region(task, &mut addr, &mut len, VM_REGION_EXTENDED_INFO, info.as_mut_ptr(), &mut count, &mut obj) };
        if kr != 0 {
            eprintln!("[maclator] trace:   no more regions (kr {kr}) at or above {addr:#x}");
            return;
        }
        if addr >= stop {
            return;
        }
        let end = addr.saturating_add(len);
        let what = if addr == got {
            "  <- Maclator's own reservation, where the kernel put it"
        } else if addr < want_end && end > want {
            "  <- holds part of the wanted range"
        } else {
            ""
        };
        eprintln!("[maclator] trace:   [{addr:#x}, {end:#x}) prot {} user_tag {}{what}", info[0], info[1]);
        addr = end;
        shown += 1;
    }
    eprintln!("[maclator] trace:   region walk stopped after {MAX_REGIONS} regions, below {addr:#x}");
}

fn file_ids(path: &str) -> (u64, u64) {
    let c = CString::new(path).unwrap();
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    unsafe { libc::stat(c.as_ptr(), &mut st) };
    (st.st_dev as u64, st.st_ino)
}

pub fn load(exe: &str, dyld_path: &str, argv: &[String], envp: &[String]) -> std::io::Result<(Loaded, Box<Cpu>)> {
    let main_m = MachO::open_arm64(exe, false)?;
    if main_m.filetype != crate::macho::MH_EXECUTE {
        return Err(std::io::Error::other(format!("{exe}: not an executable")));
    }
    let dyld_m = MachO::open_arm64(dyld_path, true)?;
    if dyld_m.filetype != crate::macho::MH_DYLINKER {
        return Err(std::io::Error::other(format!("{dyld_path}: not dyld (filetype {}, want MH_DYLINKER)", dyld_m.filetype)));
    }
    let main = map_image(exe, &main_m, MAIN_BASE)?;
    let dyld = match map_image(dyld_path, &dyld_m, DYLD_BASE) {
        Ok(d) => d,
        Err(e) => {
            unmap_image(&main);
            return Err(e);
        }
    };
    let Some(entry) = dyld_m.thread_pc.map(|pc| pc.wrapping_add(dyld.slide)) else {
        unmap_image(&dyld);
        unmap_image(&main);
        return Err(std::io::Error::other("dyld has no LC_UNIXTHREAD"));
    };

    // Stack
    let stack_lo = unsafe {
        let p = libc::mmap(std::ptr::null_mut(), (STACK_SIZE + 0x10000) as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
        assert!(p != libc::MAP_FAILED);
        libc::mprotect(p, 0x10000, libc::PROT_NONE); // guard
        p as u64 + 0x10000
    };
    let stack_hi = stack_lo + STACK_SIZE;

    // apple[] strings
    let (edev, eino) = file_ids(exe);
    let (ddev, dino) = file_ids(dyld_path);
    let mut rnd = [0u64; 4];
    for r in rnd.iter_mut() {
        let mut b = [0u8; 8];
        unsafe { libc::arc4random_buf(b.as_mut_ptr() as *mut _, 8) };
        *r = u64::from_le_bytes(b);
    }
    let apple: Vec<String> = vec![
        format!("executable_path={exe}"),
        format!("ptr_munge=0x{:x}", rnd[0]),
        format!("main_stack=0x{:x},0x{:x},0x{:x},0x{:x}", stack_hi, STACK_SIZE, stack_lo - 0x10000, 0x10000),
        format!("executable_file=0x{:x},0x{:x}", edev, eino),
        format!("dyld_file=0x{:x},0x{:x}", ddev, dino),
        format!("stack_guard=0x{:x}", rnd[1] & !0xff),
        format!("malloc_entropy=0x{:x},0x{:x}", rnd[2], rnd[3]),
        "arm64e_abi=os".to_string(),
        format!("th_port=0x{:x}", unsafe { libc::mach_thread_self() }),
    ];

    // Build the string area at the top of the stack.
    let mut sp = stack_hi;
    let mut push_str = |s: &str| -> u64 {
        let bytes = s.as_bytes();
        sp -= bytes.len() as u64 + 1;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), sp as *mut u8, bytes.len());
            *((sp + bytes.len() as u64) as *mut u8) = 0;
        }
        sp
    };
    let exec_path_ptr = push_str(exe);
    let _ = exec_path_ptr;
    let argv_p: Vec<u64> = argv.iter().map(|s| push_str(s)).collect();
    let envp_p: Vec<u64> = envp.iter().map(|s| push_str(s)).collect();
    let apple_p: Vec<u64> = apple.iter().map(|s| push_str(s)).collect();
    sp &= !0xf;
    // pointer area: mh, argc, argv..., 0, envp..., 0, apple..., 0
    let nwords = 2 + argv_p.len() + 1 + envp_p.len() + 1 + apple_p.len() + 1;
    sp -= (nwords as u64) * 8;
    sp &= !0xf;
    let mut w = sp;
    let mut put = |v: u64| {
        unsafe { *(w as *mut u64) = v };
        w += 8;
    };
    put(main.base);
    put(argv_p.len() as u64);
    for p in &argv_p {
        put(*p);
    }
    put(0);
    for p in &envp_p {
        put(*p);
    }
    put(0);
    for p in &apple_p {
        put(*p);
    }
    put(0);

    let mut cpu = Cpu::new();
    cpu.pc = entry;
    cpu.x[31] = sp;
    Ok((Loaded { main, dyld, entry, sp, stack_lo, stack_hi }, cpu))
}

#[cfg(test)]
mod slide_retry_tests {
    use super::*;

    /// Occupy MAIN_BASE, then load a PIE image that prefers it. Before the main image could slide, this returned
    /// "address 0x500000000 unavailable"; after it, the load succeeds at a slid address.
    ///
    /// The occupying mapping uses MAP_FIXED so the preferred base is genuinely taken, which is the
    /// condition the fix exists for. The NEGATIVE CONTROL is the same load with the base free: it must
    /// land exactly on MAIN_BASE, or the test would pass even if the preferred base were never honoured.
    #[test]
    fn pie_image_slides_when_preferred_base_is_taken() {
        let exe = match std::env::var("MACLATOR_TEST_ARM64_PIE") {
            Ok(p) => p,
            // No fixture, no reading: skip loudly rather than passing vacuously.
            Err(_) => {
                eprintln!("SKIP: set MACLATOR_TEST_ARM64_PIE to a thin arm64 PIE Mach-O");
                return;
            }
        };
        let macho = MachO::open_arm64(&exe, true).expect("parse the fixture");
        assert!(macho.is_pie(), "fixture must be PIE for this test to mean anything");

        // control first: with the base free, the load must take it
        let free = map_image(&exe, &macho, MAIN_BASE).expect("load with the base free");
        assert_eq!(free.slide, MAIN_BASE.wrapping_sub(macho.vm_range().0),
                   "CONTROL FAILED: a free preferred base was not used, so the test below proves nothing");
        unmap_image(&free);

        // now occupy it and load again
        let (lo, hi) = macho.vm_range();
        let squatter = unsafe {
            libc::mmap(MAIN_BASE as *mut _, (hi - lo) as usize, libc::PROT_NONE,
                libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0)
        };
        assert_ne!(squatter, libc::MAP_FAILED, "could not occupy MAIN_BASE");

        let slid = map_image(&exe, &macho, MAIN_BASE).expect("a PIE image must load at a slid address");
        assert_ne!(slid.slide, free.slide, "it reported success but did not actually slide");
        unmap_image(&slid);
        unsafe { libc::munmap(squatter, (hi - lo) as usize) };
    }

    /// dyld is MH_DYLINKER and carries no MH_PIE, so a PIE-only rule refused it ("address 0x580000000 unavailable
    /// and the image is not PIE", amfid's second exit on 27). Occupy DYLD_BASE and load dyld: it must slide.
    /// The same NEGATIVE CONTROL first: with the base free it must land exactly on DYLD_BASE.
    #[test]
    fn dyld_slides_when_its_preferred_base_is_taken() {
        let path = std::env::var("MACLATOR_TEST_DYLD").unwrap_or_else(|_| "/usr/lib/dyld".into());
        let macho = MachO::open_arm64(&path, true).expect("parse dyld");
        assert_eq!(macho.filetype, crate::macho::MH_DYLINKER, "fixture must be dyld for this test to mean anything");
        assert!(!macho.is_pie(), "this dyld carries MH_PIE, so the test would not reach the dyld rule");

        let free = map_image(&path, &macho, DYLD_BASE).expect("load dyld with its base free");
        assert_eq!(free.slide, DYLD_BASE.wrapping_sub(macho.vm_range().0),
                   "CONTROL FAILED: a free preferred base was not used, so the test below proves nothing");
        unmap_image(&free);

        let (lo, hi) = macho.vm_range();
        let squatter = unsafe {
            libc::mmap(DYLD_BASE as *mut _, (hi - lo) as usize, libc::PROT_NONE,
                libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED, -1, 0)
        };
        assert_ne!(squatter, libc::MAP_FAILED, "could not occupy DYLD_BASE");
        let slid = map_image(&path, &macho, DYLD_BASE).expect("dyld must load at a slid address");
        assert_ne!(slid.slide, free.slide, "it reported success but did not actually slide");
        unmap_image(&slid);
        unsafe { libc::munmap(squatter, (hi - lo) as usize) };
    }

    /// A 4 KiB squatter makes the kernel offer a base that is 4 KiB- but not 16 KiB-aligned; the slid image must still
    /// land on a 16 KiB boundary, since arm64 macOS pages are 16 KiB. The test first confirms the kernel really offers
    /// an unaligned address here, and fails loudly if not, because then it could not see the alignment at all.
    #[test]
    fn a_slid_image_is_16k_aligned_when_the_kernel_offers_4k() {
        let exe = match std::env::var("MACLATOR_TEST_ARM64_PIE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set MACLATOR_TEST_ARM64_PIE to a thin arm64 PIE Mach-O");
                return;
            }
        };
        let macho = MachO::open_arm64(&exe, true).expect("parse the fixture");
        let (lo, hi) = macho.vm_range();
        let want = MAIN_BASE + 0x0100_0000; // clear of the other tests' MAIN_BASE squatters
        let anon = libc::MAP_ANON | libc::MAP_PRIVATE;
        let squatter = unsafe { libc::mmap(want as *mut _, 0x1000, libc::PROT_NONE, anon | libc::MAP_FIXED, -1, 0) };
        assert_eq!(squatter as u64, want, "could not occupy the base");
        let probe = unsafe { libc::mmap(want as *mut _, (hi - lo) as usize, libc::PROT_NONE, anon, -1, 0) };
        assert_ne!(probe, libc::MAP_FAILED);
        unsafe { libc::munmap(probe, (hi - lo) as usize) };
        assert_ne!(probe as u64 % PAGE16, 0, "the kernel offered a 16 KiB-aligned address, so this test cannot see alignment");
        let slid = map_image(&exe, &macho, want).expect("a PIE image must load at a slid address");
        let base = lo.wrapping_add(slid.slide);
        assert_ne!(base, want, "it did not slide");
        assert_eq!(base % PAGE16, 0, "slid to {base:#x}, which is not 16 KiB-aligned");
        unmap_image(&slid);
        unsafe { libc::munmap(squatter, 0x1000) };
    }

    /// The shared cache is mapped MAP_FIXED over the cache window, so an image must never be reserved inside it. A
    /// preferred base that would put the image across CACHE_BASE must end up outside the window, or be refused.
    #[test]
    fn an_image_is_never_reserved_in_the_shared_cache_window() {
        let exe = match std::env::var("MACLATOR_TEST_ARM64_PIE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set MACLATOR_TEST_ARM64_PIE to a thin arm64 PIE Mach-O");
                return;
            }
        };
        let macho = MachO::open_arm64(&exe, true).expect("parse the fixture");
        let (lo, hi) = macho.vm_range();
        // the window, computed here rather than by the code under test
        let (cs, ce) = (crate::cache::CACHE_BASE, crate::cache::CACHE_BASE + 0x1_8000_0000);
        let inside = |b: u64| b < ce && b + (hi - lo) > cs;
        let want = cs - PAGE16; // the image would straddle CACHE_BASE
        assert!(inside(want), "the fixture is too small to straddle CACHE_BASE");
        match map_image(&exe, &macho, want) {
            Ok(img) => {
                let base = lo.wrapping_add(img.slide);
                assert!(!inside(base), "reserved [{base:#x}, {:#x}) inside the shared-cache window", base + (hi - lo));
                unmap_image(&img);
            }
            Err(e) => assert!(e.to_string().contains("shared-cache window"), "refused, but not for the window: {e}"),
        }
    }
}
