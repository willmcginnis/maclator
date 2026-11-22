//! Fault-tolerant access to guest memory from the runtime (syscall
//! emulation must return EFAULT rather than crash on bad pointers).

extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_read_overwrite(task: u32, addr: u64, size: u64, data: u64, outsize: *mut u64) -> i32;
    fn mach_vm_write(task: u32, addr: u64, data: u64, cnt: u32) -> i32;
    // The fallback below. mach_vm_copy has the KERNEL copy between two addresses in this task, so a
    // destination it cannot write is a kern_return_t and never a fault in this process.
    fn mach_vm_copy(task: u32, src: u64, size: u64, dst: u64) -> i32;
}

pub fn read(addr: u64, buf: &mut [u8]) -> bool {
    let mut out = 0u64;
    unsafe { mach_vm_read_overwrite(mach_task_self(), addr, buf.len() as u64, buf.as_mut_ptr() as u64, &mut out) == 0 }
}

/// `mach_vm_write` answering KERN_DENIED: the sandbox policy refuses the Mach call itself and says
/// nothing about the address. Measured twice in the same process at the same moment, on a destination
/// whose protection was already READ|WRITE and on a freshly allocated private page, both refused, so
/// the denial follows the process and not the target page.
const KERN_DENIED: i32 = 53;

/// The address the last fallback on THIS THREAD copied from, recorded so a test can observe it.
///
/// The bytes land the same whether or not the copy goes through an intermediate mapping, so without
/// this no test could tell the two apart. Thread-local because `cargo test` runs tests in parallel, and
/// `#[cfg(test)]`, so the shipped binary carries neither the slot nor the store.
#[cfg(test)]
thread_local! {
    pub(crate) static LAST_SOURCE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_source(src: u64) {
    LAST_SOURCE.with(|c| c.set(src));
}

#[cfg(not(test))]
fn note_source(_src: u64) {}

/// Copy `data` to the guest address `addr` with `mach_vm_copy`, straight from the caller's buffer.
///
/// This is the same kernel path `mach_vm_write` takes. XNU's `mach_vm_write` is
/// `vm_map_copy_overwrite` on a copy the MIG server made of the caller's buffer, and `mach_vm_copy` is
/// `vm_map_copyin` of the source range followed by the same `vm_map_copy_overwrite`
/// (osfmk/vm/vm_user.c). The copyin is complete before the overwrite starts, so a source that overlaps
/// the destination behaves like `memmove`.
///
/// NO MAPPING OF OUR OWN. The copy needs no intermediate region, so the fallback never maps or unmaps
/// anything in this task. That matters because guest threads share the address space and can replace
/// any range with MAP_FIXED or a fixed Mach allocation at any moment, and nothing serialises those
/// calls against us. A region we created, wrote into and then freed could, by the time we wrote or
/// freed it, be a mapping the guest had just put there.
///
/// True only if every byte was copied. A FALSE MAY HAVE WRITTEN A PREFIX: the kernel's overwrite path
/// copies in pieces and reports the first piece it cannot write, so a destination that stops being
/// writable partway keeps the bytes before that point. `mach_vm_write` tears in the same shapes, since
/// it is the same path. Callers treat false as EFAULT and none of them reads the destination back;
/// `guest_copyout` in syscalls.rs deliberately reproduces such a prefix itself.
fn copy_copyout(addr: u64, data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    let len = data.len() as u64;
    let src = data.as_ptr() as u64;
    note_source(src);
    let kr = unsafe { mach_vm_copy(mach_task_self(), src, len, addr) };
    kr == 0 // only a whole copy is a success; a failure may already have written a prefix
}

/// Decide the outcome of a write from the kern_return_t `mach_vm_write` gave.
///
/// THIS IS THE INJECTION SEAM, and it is the whole reason this function exists separately. The fallback
/// can only be reached on a kernel that answers KERN_DENIED, which no development machine here does --
/// they answer 0 for a writable page and 1 for NULL or a read-only one. Tests call this with the kr
/// they want, so the fallback branch runs on any machine.
pub(crate) fn finish_write(kr: i32, addr: u64, data: &[u8]) -> bool {
    if kr == 0 {
        return true;
    }
    if kr != KERN_DENIED {
        return false;
    }
    copy_copyout(addr, data)
}

/// Write `data` to the guest address `addr`.
///
/// True only if every byte was written, and a false may have written a prefix (see `copy_copyout`).
/// `mach_vm_write` is still the first and normal path, and every kern_return_t other than KERN_DENIED
/// is still a plain failure, so a bad out-pointer is still EFAULT from the kernel's own validation and
/// still never a crash. KERN_DENIED alone takes mach_vm_copy.
pub fn write(addr: u64, data: &[u8]) -> bool {
    let kr = unsafe { mach_vm_write(mach_task_self(), addr, data.as_ptr() as u64, data.len() as u32) };
    finish_write(kr, addr, data)
}

pub fn read_u64(addr: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    read(addr, &mut b).then(|| u64::from_le_bytes(b))
}

pub fn write_u64(addr: u64, v: u64) -> bool {
    write(addr, &v.to_le_bytes())
}
#[cfg(test)]
mod patch_y_copy_copyout_tests {
    //! The mach_vm_copy fallback. These drive finish_write with an INJECTED kern_return_t, so the
    //! KERN_DENIED branch executes on this machine even though this machine's kernel never produces it.
    //!
    //! Every failure message is an assert! with a single-line lowercase marker, never assert_eq!,
    //! whose first output line is "assertion `left == right` failed" rather than the message.
    //!
    //! SIZES ABOVE THE KERNEL'S msg_ool_size_small. A copy at or below that threshold becomes a
    //! kernel-buffer copy and cannot reach the entry-list path, so a test that means to exercise the
    //! entry-list path must be bigger than it: 16 KiB for an x86_64 kernel and 32 KiB otherwise.
    //! `big()` is 16 pages, which clears both.
    use super::{copy_copyout, finish_write, LAST_SOURCE};

    const DENIED: i32 = 53;

    /// VM_INHERIT_SHARE from <mach/vm_inherit.h>. 0, not 1 -- 1 is VM_INHERIT_COPY, which would make a
    /// private copy and the alias test would pass whether or not the destination pages were replaced.
    const VM_INHERIT_SHARE: u32 = 0;
    /// VM_FLAGS_ANYWHERE from <mach/vm_statistics.h>.
    const VM_FLAGS_ANYWHERE: i32 = 0x0001;

    extern "C" {
        fn mach_vm_remap(target: u32, addr: *mut u64, size: u64, mask: u64, flags: i32, src_task: u32,
                         src_addr: u64, copy: i32, cur: *mut i32, max: *mut i32, inherit: u32) -> i32;
    }

    /// The RUNTIME page size, never a 4096 literal: this host's arm64 processes have 16384-byte pages
    /// and its x86_64 ones 4096.
    fn page() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    /// A page-aligned length above every msg_ool_size_small.
    fn big() -> usize {
        page() * 16
    }

    fn map(prot: i32, len: usize) -> u64 {
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0)
        };
        assert!(p != libc::MAP_FAILED, "the test could not map its own memory");
        p as u64
    }

    /// A PAGE-ALIGNED source holding a position-dependent pattern, so a copy from it can take XNU's
    /// aligned entry-list path. 251 is prime, so no page boundary lands on a repeat.
    fn aligned_pattern(n: usize) -> u64 {
        let s = map(libc::PROT_READ | libc::PROT_WRITE, n);
        for i in 0..n {
            unsafe { *((s + i as u64) as *mut u8) = (i % 251) as u8 };
        }
        s
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn denied_over_a_writable_page_writes_the_bytes() {
        // The test a fallback that reports success without copying must break.
        let p = map(libc::PROT_READ | libc::PROT_WRITE, page());
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x11, 8) };
        assert!(finish_write(DENIED, p, &[0xAAu8; 8]), "the fallback must complete the write");
        let got = unsafe { std::slice::from_raw_parts(p as *const u8, 8) };
        assert!(got == &[0xAAu8; 8], "the bytes must actually land through mach_vm_copy: {got:?}");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn denied_over_null_is_refused_and_does_not_crash() {
        assert!(!finish_write(DENIED, 0, &[1u8; 8]), "null must be refused by the kernel, not stored to");
    }

    #[test]
    fn denied_over_a_read_only_page_is_refused_and_does_not_crash() {
        let p = map(libc::PROT_READ, page());
        assert!(!finish_write(DENIED, p, &[1u8; 8]), "a read-only page must be refused");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn denied_over_an_unreachable_reservation_is_refused_and_does_not_crash() {
        // A PROT_NONE reservation this test holds for its lifetime, not a freed page or a constant: a
        // freed page can be reused by a parallel test thread, and the write would then land in it.
        let p = map(libc::PROT_NONE, page());
        assert!(!finish_write(DENIED, p, &[1u8; 8]), "an unwritable reservation must be refused");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn a_non_denied_error_never_reaches_the_copy() {
        // kr 1 is KERN_INVALID_ADDRESS. It must fail WITHOUT writing, which keeps the fallback narrow.
        let p = map(libc::PROT_READ | libc::PROT_WRITE, page());
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x11, 8) };
        assert!(!finish_write(1, p, &[0xAAu8; 8]), "a non-denied error must stay a failure");
        let got = unsafe { std::slice::from_raw_parts(p as *const u8, 8) };
        assert!(got == &[0x11u8; 8], "nothing may be copied when the kr was not kern_denied: {got:?}");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn success_never_reaches_the_copy_either() {
        let p = map(libc::PROT_READ | libc::PROT_WRITE, page());
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x22, 8) };
        assert!(finish_write(0, p, &[0xAAu8; 8]), "kr 0 is already a success");
        let got = unsafe { std::slice::from_raw_parts(p as *const u8, 8) };
        assert!(got == &[0x22u8; 8], "kr 0 must not copy anything a second time: {got:?}");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn bytes_past_the_end_are_untouched() {
        // An unaligned destination, an odd length and a source on the stack: both neighbours must be
        // left alone, so the copy is byte-accurate from an arbitrary caller buffer.
        let p = map(libc::PROT_READ | libc::PROT_WRITE, page() * 2);
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x11, page() * 2) };
        let at = p + 37;
        assert!(finish_write(DENIED, at, &[0x7Eu8; 13]), "the fallback must complete an odd-length write");
        let body = unsafe { std::slice::from_raw_parts(at as *const u8, 13) };
        assert!(body == &[0x7Eu8; 13], "the 13 requested bytes must land: {body:?}");
        let after = unsafe { std::slice::from_raw_parts((at + 13) as *const u8, 16) };
        assert!(after == &[0x11u8; 16], "bytes past the end were overwritten, so the length is wrong: {after:?}");
        let before = unsafe { *((at - 1) as *const u8) };
        assert!(before == 0x11, "the byte before the destination was overwritten: {before:#x}");
        unsafe { libc::munmap(p as *mut libc::c_void, page() * 2) };
    }

    #[test]
    fn the_copy_comes_from_the_callers_own_buffer() {
        // THE FALLBACK CREATES NO MAPPING OF ITS OWN. Guest threads can replace any range of this
        // address space with MAP_FIXED, unserialised against us, so a region the fallback maps, writes
        // and frees could by then be the guest's. A copy straight from the caller's buffer has no such
        // region. The bytes land either way, so the test reads what the copy came FROM.
        //
        // The source is taken at an odd offset into a stack array, so any staged copy, which would
        // come from a fresh page-aligned region, cannot coincide with it.
        let buf = [0x5Cu8; 24];
        let data = &buf[1..17];
        let p = map(libc::PROT_READ | libc::PROT_WRITE, page());
        LAST_SOURCE.with(|c| c.set(0));
        assert!(finish_write(DENIED, p, data), "the fallback must succeed here");
        let src = LAST_SOURCE.with(|c| c.get());
        assert!(src == data.as_ptr() as u64, "the copy source {src:#x} is not the caller's own buffer, so the fallback copied through a mapping of its own");
        let got = unsafe { std::slice::from_raw_parts(p as *const u8, 16) };
        assert!(got == data, "the bytes from the caller's buffer must land");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn a_straddling_write_reports_failure_and_may_tear_the_prefix() {
        // A straddle into an unwritable page is REPORTED AS A FAILURE, which is what callers rely on.
        // The prefix is deliberately not asserted: from 64 bytes up the kernel leaves it torn, as
        // mach_vm_write does. Both sides of msg_ool_size_small are driven, because they are different
        // code and answer with different kern_return_t values. The byte BEFORE the destination is the
        // bound that is asserted: a tear must not reach backwards out of the requested range.
        for len in [16usize, big()] {
            let ps = page();
            let span = ((len + ps) / ps + 2) * ps;
            let base = map(libc::PROT_READ | libc::PROT_WRITE, span);
            unsafe { std::ptr::write_bytes(base as *mut u8, 0x00, span) };
            let guard = base + (span - ps) as u64;
            let r = unsafe { libc::mprotect(guard as *mut libc::c_void, ps, libc::PROT_NONE) };
            assert!(r == 0, "the test could not make its guard page unwritable");
            // The write ends 4 bytes inside the guard page, so its last 4 bytes cannot land.
            let at = guard + 4 - len as u64;
            unsafe { *((at - 1) as *mut u8) = 0x5A };
            assert!(!finish_write(DENIED, at, &vec![0x7Eu8; len]), "a straddling write must report failure");
            let before = unsafe { *((at - 1) as *const u8) };
            assert!(before == 0x5A, "the byte before the destination was overwritten by a straddle: {before:#x}");
            unsafe {
                libc::mprotect(guard as *mut libc::c_void, ps, libc::PROT_READ | libc::PROT_WRITE);
                libc::munmap(base as *mut libc::c_void, span);
            }
        }
    }

    #[test]
    fn a_buffer_far_larger_than_a_page_lands_in_order() {
        // 1 MiB from a heap buffer to an unaligned destination, with a position-dependent pattern, so a
        // copy that moved bytes to the wrong offsets shows.
        const N: usize = 1024 * 1024;
        let p = map(libc::PROT_READ | libc::PROT_WRITE, N + page());
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x00, N + page()) };
        let src = pattern(N);
        let at = p + 9;
        assert!(copy_copyout(at, &src), "a large buffer must copy in one call");
        let got = unsafe { std::slice::from_raw_parts(at as *const u8, N) };
        assert!(got == &src[..], "every byte must land at its own offset and in order");
        let tail = unsafe { *((at + N as u64) as *const u8) };
        assert!(tail == 0x00, "the byte past the end was overwritten: {tail:#x}");
        unsafe { libc::munmap(p as *mut libc::c_void, N + page()) };
    }

    #[test]
    fn a_page_aligned_copy_above_the_kernel_buffer_limit_lands_in_order() {
        // THE SHAPE THAT CAN TAKE XNU's ALIGNED ENTRY-LIST PATH: page-aligned source and destination and
        // a length above every msg_ool_size_small. The source is a page-aligned mapping on purpose,
        // since a caller's buffer usually is not and the path would otherwise go unexercised.
        let n = big();
        let s = aligned_pattern(n);
        let p = map(libc::PROT_READ | libc::PROT_WRITE, n + page());
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x00, n + page()) };
        let src = unsafe { std::slice::from_raw_parts(s as *const u8, n) };
        assert!(finish_write(DENIED, p, src), "a page-aligned copy above the limit must succeed");
        let got = unsafe { std::slice::from_raw_parts(p as *const u8, n) };
        assert!(got == &pattern(n)[..], "every byte of a page-aligned copy must land at its own offset");
        let tail = unsafe { *((p + n as u64) as *const u8) };
        assert!(tail == 0x00, "the byte past the end of a page-aligned copy was overwritten: {tail:#x}");
        unsafe {
            libc::munmap(s as *mut libc::c_void, n);
            libc::munmap(p as *mut libc::c_void, n + page());
        }
    }

    #[test]
    fn an_empty_write_is_trivially_fine() {
        assert!(copy_copyout(0, &[]), "an empty write needs no destination");
        assert!(finish_write(DENIED, 0, &[]), "and it is fine through the seam too");
    }

    #[test]
    fn threads_write_concurrently_without_mixing() {
        // Eight threads, each with its own destination page and byte value, 64 fallbacks apiece. Every
        // byte of every page is checked, so one thread's bytes landing in another's page would show.
        const T: usize = 8;
        const ITERS: usize = 64;
        const LEN: usize = 300;
        let mut hands = Vec::new();
        for t in 0..T {
            hands.push(std::thread::spawn(move || {
                let ps = page();
                let p = unsafe {
                    libc::mmap(std::ptr::null_mut(), ps, libc::PROT_READ | libc::PROT_WRITE,
                               libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0)
                };
                assert!(p != libc::MAP_FAILED, "a thread could not map its own page");
                let me = 0x30u8 + t as u8;
                let mine = vec![me; LEN];
                for i in 0..ITERS {
                    assert!(finish_write(DENIED, p as u64, &mine), "thread {t} iteration {i} failed");
                    let got = unsafe { std::slice::from_raw_parts(p as *const u8, LEN) };
                    assert!(got == &mine[..], "thread {t} saw another thread's bytes in its own page");
                }
                unsafe { libc::munmap(p, ps) };
            }));
        }
        for h in hands {
            h.join().expect("a thread panicked, so its own assertion message is above");
        }
    }

    #[test]
    fn the_destination_keeps_the_bytes_after_the_source_is_overwritten_and_freed() {
        // LIFETIME. The aligned path can hand the destination the source's pages copy-on-write, so the
        // source is then overwritten and unmapped; the destination must still hold what was copied.
        // Page-aligned and above the kernel-buffer limit, the only shape that moves VM structure.
        let n = big();
        let s = aligned_pattern(n);
        let p = map(libc::PROT_READ | libc::PROT_WRITE, n);
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x77, n) };
        let src = unsafe { std::slice::from_raw_parts(s as *const u8, n) };
        assert!(finish_write(DENIED, p, src), "the fallback must succeed here");
        unsafe {
            std::ptr::write_bytes(s as *mut u8, 0x00, n);
            libc::munmap(s as *mut libc::c_void, n);
        }
        let got = unsafe { std::slice::from_raw_parts(p as *const u8, n) };
        assert!(got == &pattern(n)[..], "the destination lost the bytes once the source was overwritten and freed");
        unsafe { libc::munmap(p as *mut libc::c_void, n) };
    }

    #[test]
    fn a_shared_alias_of_the_destination_sees_the_copy() {
        // XNU's aligned copy path can install the source's VM object in the destination entry instead of
        // copying bytes; a guard in the kernel confines that to an unshared destination. This asserts
        // the property the patch needs, that a destination with a live shared mapping has the copied
        // bytes visible through it. It does not gate any single condition of that guard.
        let n = big();
        let s = aligned_pattern(n);
        let d = map(libc::PROT_READ | libc::PROT_WRITE, n);
        unsafe { std::ptr::write_bytes(d as *mut u8, 0x44, n) };
        let mut alias = 0u64;
        let (mut cur, mut max) = (0i32, 0i32);
        let kr = unsafe {
            mach_vm_remap(super::mach_task_self(), &mut alias, n as u64, 0, VM_FLAGS_ANYWHERE,
                          super::mach_task_self(), d, 0, &mut cur, &mut max, VM_INHERIT_SHARE)
        };
        assert!(kr == 0, "the test could not make a shared alias of its destination (kr {kr})");
        let src = unsafe { std::slice::from_raw_parts(s as *const u8, n) };
        assert!(finish_write(DENIED, d, src), "the fallback must succeed here");
        let through_alias = unsafe { std::slice::from_raw_parts(alias as *const u8, n) };
        assert!(through_alias == &pattern(n)[..], "a shared alias did not see the whole copy, so the destination pages were replaced");
        unsafe {
            libc::munmap(alias as *mut libc::c_void, n);
            libc::munmap(d as *mut libc::c_void, n);
            libc::munmap(s as *mut libc::c_void, n);
        }
    }
}
