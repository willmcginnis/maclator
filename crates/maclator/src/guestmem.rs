//! Fault-tolerant access to guest memory from the runtime (syscall
//! emulation must return EFAULT rather than crash on bad pointers).

extern "C" {
    /// libSystem's name for this task's control port: mach_init sets it from task_self_trap() when the process
    /// starts, and again in the child of a libc fork (the only fork this runtime makes), so it is declared mut and
    /// read afresh each time. The kernel pins the send right under this name, so the task cannot give up its last
    /// reference: mach_port_deallocate, mach_port_mod_refs and mach_port_destroy all answer
    /// KERN_INVALID_CAPABILITY for it instead.
    static mut mach_task_self_: u32;
    fn mach_vm_read_overwrite(task: u32, addr: u64, size: u64, data: u64, outsize: *mut u64) -> i32;
    fn mach_vm_write(task: u32, addr: u64, data: u64, cnt: u32) -> i32;
    // The fallback below. mach_vm_copy has the KERNEL copy between two addresses in this task, so a
    // destination it cannot write is a kern_return_t and never a fault in this process.
    fn mach_vm_copy(task: u32, src: u64, size: u64, dst: u64) -> i32;
}

/// The task port every read and write of guest memory goes through: libSystem's cached name, the one native code
/// uses through the mach_task_self() macro. Not libSystem's mach_task_self() FUNCTION, which is task_self_trap():
/// that asks the kernel for another send right on every call, adds a user reference each time (so a busy process
/// pins the count at its ceiling of 65535), and answers whatever the kernel copies out at that moment, so every
/// access would depend on that copyout succeeding and naming the control port. A process has been seen where it
/// did not: every read through the function's answer failed with MACH_SEND_INVALID_DEST while reads through this
/// name, in the same process at the same moment, succeeded.
fn task() -> u32 {
    unsafe { std::ptr::addr_of!(mach_task_self_).read_volatile() }
}

pub fn read(addr: u64, buf: &mut [u8]) -> bool {
    let mut out = 0u64;
    let kr = match test_refusal(addr, buf.len()) {
        Some(kr) => kr,
        None => unsafe { mach_vm_read_overwrite(task(), addr, buf.len() as u64, buf.as_mut_ptr() as u64, &mut out) },
    };
    finish_read(kr, addr, buf.len())
}

/// The kern_return_t values that mean the guest's address is bad: nothing readable is mapped there. Measured on macOS
/// 27.0.1, both from an arm64 process and from an x86_64 one under Rosetta 2: NULL, an unmapped page, a PROT_NONE
/// page, a read straddling into either, the top of user space, a non-canonical address, the kernel half and a range
/// that wraps all answer KERN_INVALID_ADDRESS (1). KERN_PROTECTION_FAILURE (2) is the kernel's other answer for a page
/// it will not read, so it is counted with it.
fn bad_address(kr: i32) -> bool {
    kr == 1 || kr == 2
}

/// Decide the outcome of a read from the kern_return_t `mach_vm_read_overwrite` gave. Any failure is still a plain
/// failure (callers answer EFAULT, or the JIT interprets), but one that does not mean a bad address -- KERN_DENIED from
/// a policy that refuses the Mach call itself, say -- is reported, since otherwise it looks exactly like a bad guest
/// pointer, even under MACLATOR_TRACE.
pub(crate) fn finish_read(kr: i32, addr: u64, len: usize) -> bool {
    if kr == 0 {
        return true;
    }
    if !bad_address(kr) {
        note_refused_read(kr, addr, len);
    }
    false
}

/// Refused reads of this process so far. The 1st, 2nd, 4th, 8th ... are reported, each with its number, so a refusal
/// on a hot path cannot flood the guest's stderr (a process gets at most 64 lines) and one that goes on stays visible.
/// `REFUSED_READS_PID` is the process they were counted in: a forked child starts again at 1, so its first refusal is
/// reported too.
static REFUSED_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REFUSED_READS_PID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The number of this refusal in process `pid`: `count` starts again when `owner` names another process.
fn next_refusal(count: &std::sync::atomic::AtomicU64, owner: &std::sync::atomic::AtomicU32, pid: u32) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    if owner.swap(pid, Relaxed) != pid {
        count.store(0, Relaxed);
    }
    count.fetch_add(1, Relaxed) + 1
}

/// The line for the `n`th refused read of this process, if it is reported.
fn refused_read_line(n: u64, kr: i32, addr: u64, len: usize) -> Option<String> {
    n.is_power_of_two().then(|| format!("[maclator] warning: refused read #{n} (reported at #1, 2, 4, 8, ...): reading {len} bytes of guest memory at {addr:#x} answered kern_return_t {kr}, which is not a bad address; treated as unreadable"))
}

fn note_refused_read(kr: i32, addr: u64, len: usize) {
    #[cfg(test)]
    REFUSALS_NOTED.with(|c| c.set(c.get() + 1));
    let n = next_refusal(&REFUSED_READS, &REFUSED_READS_PID, std::process::id());
    if let Some(line) = refused_read_line(n, kr, addr, len) {
        eprintln!("{line}");
    }
}

/// Refused reads noted on THIS THREAD, so a test can see a report without reading stderr. Test builds only.
#[cfg(test)]
thread_local! {
    pub(crate) static REFUSALS_NOTED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// [lo, hi): a read on THIS THREAD overlapping it is answered KERN_DENIED without the Mach call, as a policy that
    /// refuses mach_vm_read_overwrite would answer it. Test builds only, so the shipped binary has no seam.
    pub(crate) static REFUSE_READS_IN: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
fn test_refusal(addr: u64, len: usize) -> Option<i32> {
    let (lo, hi) = REFUSE_READS_IN.with(|c| c.get());
    (addr < hi && addr.saturating_add(len.max(1) as u64) > lo).then_some(KERN_DENIED)
}

#[cfg(not(test))]
#[inline(always)]
fn test_refusal(_addr: u64, _len: usize) -> Option<i32> {
    None
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
    let kr = unsafe { mach_vm_copy(task(), src, len, addr) };
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
    let kr = unsafe { mach_vm_write(task(), addr, data.as_ptr() as u64, data.len() as u32) };
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
mod refused_read_tests {
    //! guestmem::read's report of a refusal that is not a bad address.
    use super::{finish_read, next_refusal, read, refused_read_line, REFUSALS_NOTED, REFUSE_READS_IN};

    fn noted() -> u64 {
        REFUSALS_NOTED.with(|c| c.get())
    }

    fn page() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    #[test]
    fn a_refusal_that_is_not_a_bad_address_is_reported() {
        let before = noted();
        assert!(!finish_read(53, 0x1000, 8), "kern_denied is still a failure");
        assert!(noted() == before + 1, "kern_denied must be reported, since it is not a bad address");
        assert!(!finish_read(4, 0x1000, 8), "kern_invalid_argument is still a failure");
        assert!(noted() == before + 2, "any kern_return_t but a bad address is reported");
    }

    #[test]
    fn a_bad_address_or_a_success_is_not_reported() {
        let before = noted();
        assert!(!finish_read(1, 0, 8), "kern_invalid_address is a failure");
        assert!(!finish_read(2, 0, 8), "kern_protection_failure is a failure");
        assert!(finish_read(0, 0x1000, 8), "kr 0 is a success");
        assert!(noted() == before, "a bad address and a success are not reported");
    }

    #[test]
    fn the_kernels_answers_for_bad_addresses_are_not_reported() {
        // The real kernel, so the classification matches what it answers. A PROT_NONE page and an unmapped one held
        // by this test, NULL, a read straddling into the PROT_NONE page, and a range that wraps.
        let ps = page();
        let base = unsafe { libc::mmap(std::ptr::null_mut(), ps * 2, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
        assert!(base != libc::MAP_FAILED, "the test could not map its own memory");
        let base = base as u64;
        assert!(unsafe { libc::mprotect((base + ps as u64) as *mut libc::c_void, ps, libc::PROT_NONE) } == 0, "control: mprotect");
        let mut b = [0u8; 8];
        assert!(read(base, &mut b), "control: a readable page reads");
        let before = noted();
        for (what, at) in [("null", 0u64), ("prot_none", base + ps as u64), ("straddle", base + ps as u64 - 4), ("wraps", u64::MAX - 3)] {
            assert!(!read(at, &mut b), "{what}: a bad address must not read");
        }
        assert!(noted() == before, "the kernel's answers for bad addresses were reported as refusals");
        unsafe { libc::munmap(base as *mut libc::c_void, ps * 2) };
    }

    #[test]
    fn a_refused_read_of_a_readable_page_is_reported_and_fails() {
        let ps = page();
        let p = unsafe { libc::mmap(std::ptr::null_mut(), ps, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
        assert!(p != libc::MAP_FAILED, "the test could not map its own memory");
        let p = p as u64;
        let mut b = [0u8; 8];
        assert!(read(p, &mut b), "control: the page reads");
        let before = noted();
        REFUSE_READS_IN.with(|c| c.set((p, p + ps as u64)));
        let got = read(p, &mut b);
        REFUSE_READS_IN.with(|c| c.set((0, 0)));
        assert!(!got, "a refused read is still a failure");
        assert!(noted() == before + 1, "a refused read of a readable page must be reported");
        unsafe { libc::munmap(p as *mut libc::c_void, ps) };
    }

    #[test]
    fn a_forked_child_numbers_its_refusals_from_one() {
        use std::sync::atomic::{AtomicU32, AtomicU64};
        // The counter and its owner as a fork leaves them: the parent (pid 100) had 5 refusals.
        let (count, owner) = (AtomicU64::new(0), AtomicU32::new(0));
        let parent: Vec<u64> = (0..5).map(|_| next_refusal(&count, &owner, 100)).collect();
        assert!(parent == [1, 2, 3, 4, 5], "control: the parent counts from 1: {parent:?}");
        let child: Vec<u64> = (0..3).map(|_| next_refusal(&count, &owner, 200)).collect();
        assert!(child == [1, 2, 3], "a forked child must number its own refusals from 1, so its first is reported: {child:?}");
    }

    #[test]
    fn refusals_are_reported_at_powers_of_two_with_their_number() {
        let first = refused_read_line(1, 53, 0x1234, 4).expect("the first refusal is reported");
        assert!(first.contains("kern_return_t 53") && first.contains("0x1234") && first.contains("4 bytes") && first.contains("#1 "), "the line names the number, the kr, the address and the length: {first}");
        let reported: Vec<u64> = (1..=1100u64).filter(|&n| refused_read_line(n, 53, 0, 4).is_some()).collect();
        assert!(reported == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], "refusals must be reported at powers of two only: {reported:?}");
        let late = refused_read_line(1 << 40, 53, 0, 4).expect("a late power of two is still reported");
        assert!(late.contains(&format!("#{} ", 1u64 << 40)), "a late report carries its number: {late}");
    }
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
            mach_vm_remap(super::task(), &mut alias, n as u64, 0, VM_FLAGS_ANYWHERE,
                          super::task(), d, 0, &mut cur, &mut max, VM_INHERIT_SHARE)
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

#[cfg(test)]
mod task_port_tests {
    //! The task port the reader and writer use. The checks that need a process of their own (a fresh count of the
    //! task name's user references, or a task that has tried to give its name up) re-run this test binary for one
    //! ignored test, which does its work only when MCL_GUESTMEM_CHILD names it, and the parent checks its marker.
    use super::{finish_write, read, task, write};

    const MACH_PORT_RIGHT_SEND: u32 = 0;
    const KERN_INVALID_CAPABILITY: i32 = 20;
    /// KERN_DENIED, which sends a write through the mach_vm_copy fallback (finish_write).
    const KERN_DENIED: i32 = 53;

    extern "C" {
        fn pid_for_task(task: u32, pid: *mut i32) -> i32;
        fn mach_port_get_refs(task: u32, name: u32, right: u32, refs: *mut u32) -> i32;
        fn mach_port_deallocate(task: u32, name: u32) -> i32;
        fn mach_port_mod_refs(task: u32, name: u32, right: u32, delta: i32) -> i32;
        fn mach_port_destroy(task: u32, name: u32) -> i32;
    }

    fn send_refs(t: u32) -> u32 {
        let mut r = 0u32;
        let kr = unsafe { mach_port_get_refs(t, t, MACH_PORT_RIGHT_SEND, &mut r) };
        assert!(kr == 0, "control: mach_port_get_refs on the task's own name answered {kr}");
        r
    }

    /// Run `name` (an ignored test of this module) alone in a fresh copy of this test binary; its stdout.
    fn in_a_fresh_process(name: &str) -> String {
        let exe = std::env::current_exe().expect("the test binary's path");
        let full = format!("guestmem::task_port_tests::{name}");
        let out = std::process::Command::new(exe)
            .args([full.as_str(), "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env("MCL_GUESTMEM_CHILD", name)
            .output()
            .expect("the child test process could not be started");
        let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "the child {name} failed ({}):\n{text}", out.status);
        assert!(text.contains(&format!("GUESTMEM-CHILD {name} ok")), "the child {name} never reached its end (it did not run?):\n{text}");
        text
    }

    fn is_child(name: &str) -> bool {
        std::env::var("MCL_GUESTMEM_CHILD").map(|v| v == name).unwrap_or(false)
    }

    #[test]
    fn reads_and_writes_take_no_new_right_to_the_task() {
        let out = in_a_fresh_process("child_counts_the_task_names_references");
        assert!(out.contains("TASK-REFS"), "the child printed no counts:\n{out}");
    }

    #[test]
    #[ignore = "run by reads_and_writes_take_no_new_right_to_the_task, in a process of its own"]
    fn child_counts_the_task_names_references() {
        if !is_child("child_counts_the_task_names_references") {
            return;
        }
        // libSystem's mach_task_self() function would add one user reference per call (it is task_self_trap), so
        // 1000 each of reads, writes and mach_vm_copy fallbacks through it would raise the count by about 3000,
        // which is why the count is read in a fresh process, far from the ceiling of 65535 where the kernel stops
        // counting.
        let t = task();
        let before = send_refs(t);
        assert!(before < 60000, "precondition: the task name's references ({before}) are too near their ceiling to count");
        let mut word = 0u64;
        let p = &mut word as *mut u64 as u64;
        let mut b = [0u8; 8];
        // Each round writes two values no earlier step wrote, so a write or a copy that does nothing is seen.
        for i in 0..1000u64 {
            let (w, c) = (3 * i + 1, 3 * i + 2);
            assert!(write(p, &w.to_le_bytes()), "control: write {i} of a stack word failed");
            assert!(read(p, &mut b) && u64::from_le_bytes(b) == w, "control: read {i} of a stack word did not see the write");
            assert!(finish_write(KERN_DENIED, p, &c.to_le_bytes()) && word == c, "control: mach_vm_copy fallback {i} failed");
        }
        let after = send_refs(t);
        println!("TASK-REFS before={before} after={after}");
        assert!(after == before, "1000 reads, writes and copies changed the task name's user references from {before} to {after}: guest memory took or gave up references to the task's name");
        println!("GUESTMEM-CHILD child_counts_the_task_names_references ok");
    }

    #[test]
    fn the_task_cannot_give_up_the_name_guest_memory_goes_through() {
        let out = in_a_fresh_process("child_tries_to_release_its_task_name");
        assert!(out.contains("RELEASE"), "the child printed no release results:\n{out}");
    }

    #[test]
    #[ignore = "run by the_task_cannot_give_up_the_name_guest_memory_goes_through, in a process of its own"]
    fn child_tries_to_release_its_task_name() {
        if !is_child("child_tries_to_release_its_task_name") {
            return;
        }
        // What a guest that over-releases its own task port can do: every reference one at a time, then all at
        // once, then the name itself. The kernel pins this right, so the last of each is refused, and the reader
        // and writer still work. On a kernel that let a task release it, this is the test that would show it.
        // (Each refusal may also raise a guard exception, by the task's policy: none, a crash report, or the end of
        // the process, which this test then reports as a failed child.)
        let t = task();
        let refs = send_refs(t);
        let mut ok = 0u32;
        let mut last = 0i32;
        for _ in 0..refs + 3 {
            last = unsafe { mach_port_deallocate(t, t) };
            if last != 0 {
                break;
            }
            ok += 1;
        }
        let left = send_refs(t);
        let all = unsafe { mach_port_mod_refs(t, t, MACH_PORT_RIGHT_SEND, -(left as i32)) };
        let destroy = unsafe { mach_port_destroy(t, t) };
        println!("RELEASE refs={refs} deallocated={ok} then kr={last}; mod_refs -{left} kr={all}; destroy kr={destroy}");
        assert!(left == 1 && last == KERN_INVALID_CAPABILITY && all == KERN_INVALID_CAPABILITY && destroy == KERN_INVALID_CAPABILITY,
                "the kernel let the task release its own task name, or refused an earlier step (left {left}; kr {last}, {all}, {destroy}): the reader's name is not protected as expected");
        let mut word = 0x0123_4567_89ab_cdefu64;
        let p = &mut word as *mut u64 as u64;
        let mut b = [0u8; 8];
        assert!(read(p, &mut b) && u64::from_le_bytes(b) == 0x0123_4567_89ab_cdef, "after the release attempts a read of a stack word failed");
        assert!(write(p, &7u64.to_le_bytes()) && word == 7, "after the release attempts a write of a stack word failed");
        println!("GUESTMEM-CHILD child_tries_to_release_its_task_name ok");
    }

    #[test]
    fn a_child_of_libc_fork_reads_and_writes_through_its_own_task_name() {
        // The runtime forks only through libc's fork, whose child handler sets mach_task_self_ again, so the
        // child's reads go through the child's own task, and the name must name it (pid_for_task). The child's
        // name is usually the same number as the parent's (the control port is the first right in a new space),
        // so this cannot tell a stale copy of that number from a fresh read; it shows the child works. The child
        // runs only Mach calls on its own stack words (a failed read would also format a warning), and an alarm
        // ends a child that hangs.
        let parent = task();
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "control: fork failed");
        if pid == 0 {
            unsafe { libc::alarm(10) };
            let mut owner = 0i32;
            let names_me = unsafe { pid_for_task(task(), &mut owner) } == 0 && owner == unsafe { libc::getpid() };
            let mut word = 41u64;
            let p = &mut word as *mut u64 as u64;
            let mut b = [0u8; 8];
            let read_ok = read(p, &mut b) && u64::from_le_bytes(b) == 41;
            let write_ok = write(p, &42u64.to_le_bytes()) && word == 42;
            let code = if !names_me { 4 } else if !read_ok { 3 } else if !write_ok { 2 } else { 0 };
            unsafe { libc::_exit(code) };
        }
        let mut status = 0;
        let w = loop {
            let r = unsafe { libc::waitpid(pid, &mut status, 0) };
            if r != -1 || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                break r;
            }
        };
        assert!(w == pid, "control: waitpid");
        assert!(libc::WIFEXITED(status), "the forked child did not exit normally (an alarm ends a hang): status {status:#x}");
        let code = libc::WEXITSTATUS(status);
        let what = match code { 4 => "task name does not name the child", 3 => "read", _ => "write" };
        assert!(code == 0, "in a child of libc fork the {what} failed (exit {code})");
        assert!(task() == parent, "control: the parent's task name changed across the fork");
    }
}
