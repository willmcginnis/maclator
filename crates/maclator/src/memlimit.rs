//! Charge the translator's own memory to the process's memory limit, the way dyld charges the
//! pages it dirties: `memorystatus_control(MEMORYSTATUS_CMD_INCREASE_JETSAM_TASK_LIMIT, getpid(),
//! bytes, NULL, 0)` (dyld's `RuntimeState::increaseJetsamLimit`). A daemon with a small jetsam
//! limit is sized for its native code; under translation the JIT code and the translator's heap
//! count against that limit too.
//!
//! Counted: JIT code cache in use (`jit::code_committed`) and the translator's own heap (the
//! counting global allocator below); not the guest's memory, which the guest maps itself. Until the
//! guest starts, each time the total passes another MiB, the limit is raised by the MiBs passed. As
//! the guest starts (`guest_starts`), the limit is raised once more, to cover what the translator will
//! add while the program runs, and the call is never made again: every call comes before the
//! program's own code runs, as dyld's come while it loads the program. A program may apply a sandbox
//! that lists the system calls it may make, and most that do (Apple's WindowManager among them) do not
//! list memorystatus_control, so a call after that would be refused, and the refusal logged, in every
//! such process. (A process started in such a sandbox is refused its first call, once.) Increase only, as
//! the kernel's call is: memory freed later is not given back. Kernels that do not accept the call
//! (Apple compiles it into DEVELOPMENT kernels only) refuse it; the first refusal turns this off,
//! and nothing changes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

const STEP: u64 = 1 << 20;
/// The kernel takes the increase as a u32 byte count and keeps the total under INT32_MAX.
const MAX_STEPS_PER_CALL: u64 = i32::MAX as u64 / STEP;
/// MEMORYSTATUS_CMD_INCREASE_JETSAM_TASK_LIMIT, bsd/sys/kern_memorystatus.h.
const CMD_INCREASE_JETSAM_TASK_LIMIT: u32 = 22;
/// What the limit is raised to cover as the guest starts, at least: the translator's memory for the rest of the run,
/// which it cannot ask for later. 64 MiB is several times the translator's own state in a small daemon (about 16 MiB);
/// a kernel that caps these increases lower answers ENOMEM, and less is asked for.
const UPFRONT: u64 = 64 << 20;

extern "C" {
    fn memorystatus_control(command: u32, pid: i32, flags: u32, buffer: *mut libc::c_void, buffersize: usize) -> i32;
}

/// Bytes live in the translator's heap.
pub static HEAP_LIVE: AtomicU64 = AtomicU64::new(0);
static TRACKER: Tracker = Tracker::new();

/// The system allocator, counting the bytes live in it.
pub struct Counting {
    live: &'static AtomicU64,
    report: bool,
}

impl Counting {
    pub const fn new(live: &'static AtomicU64, report: bool) -> Self {
        Counting { live, report }
    }
    fn grew(&self, n: usize) {
        self.live.fetch_add(n as u64, Relaxed);
        if self.report {
            check();
        }
    }
    fn shrank(&self, n: usize) {
        self.live.fetch_sub(n as u64, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            self.grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            self.grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        self.shrank(l.size());
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = System.realloc(p, l, new_size);
        if !q.is_null() {
            if new_size >= l.size() {
                self.grew(new_size - l.size());
            } else {
                self.shrank(l.size() - new_size);
            }
        }
        q
    }
}

/// The MiBs already added to the limit, and whether the kernel has refused.
pub struct Tracker {
    reported: AtomicU64,
    off: AtomicBool,
}

impl Tracker {
    pub const fn new() -> Self {
        Tracker { reported: AtomicU64::new(0), off: AtomicBool::new(false) }
    }

    /// `total` bytes are now counted: add the whole MiBs not yet added, through `increase`, which
    /// is given a byte count and answers the kernel's errno if it refused.
    pub fn note(&self, total: u64, increase: impl FnOnce(u32) -> Result<(), i32>) {
        if self.off.load(Relaxed) {
            return;
        }
        let want = total / STEP;
        let mut have = self.reported.load(Relaxed);
        while want > have {
            let add = (want - have).min(MAX_STEPS_PER_CALL);
            match self.reported.compare_exchange_weak(have, have + add, Relaxed, Relaxed) {
                Ok(_) => {
                    if increase((add * STEP) as u32).is_err() {
                        self.off.store(true, Relaxed);
                    }
                    return;
                }
                Err(now) => have = now,
            }
        }
    }

    /// The guest starts, and `total` bytes are counted: raise the limit to cover `total`, or `UPFRONT` if that is more,
    /// in one call (halved while the kernel answers ENOMEM, its answer at its cap), then never call again.
    pub fn guest_starts(&self, total: u64, mut increase: impl FnMut(u32) -> Result<(), i32>) -> u64 {
        if self.off.swap(true, Relaxed) {
            return 0;
        }
        let have = self.reported.load(Relaxed);
        let mut add = (total.max(UPFRONT) / STEP).saturating_sub(have).min(MAX_STEPS_PER_CALL);
        while add > 0 {
            match increase((add * STEP) as u32) {
                Ok(()) => {
                    self.reported.fetch_add(add, Relaxed);
                    return add * STEP;
                }
                Err(libc::ENOMEM) => add /= 2,
                Err(_) => return 0,
            }
        }
        0
    }

    /// A forked child is a new process to the kernel, whose increase starts at zero (and after the guest started, it
    /// makes no call).
    fn forked(&self) {
        self.reported.store(0, Relaxed);
    }
}

/// Raise the limit if the translator's memory has passed another MiB.
pub fn check() {
    TRACKER.note(HEAP_LIVE.load(Relaxed) + crate::jit::code_committed(), increase_limit);
}

fn increase_limit(bytes: u32) -> Result<(), i32> {
    if unsafe { memorystatus_control(CMD_INCREASE_JETSAM_TASK_LIMIT, libc::getpid(), bytes, std::ptr::null_mut(), 0) } == 0 {
        Ok(())
    } else {
        Err(unsafe { *libc::__error() })
    }
}

/// Just before the guest's first instruction: the last raise of the limit (see the module comment).
pub fn guest_starts() {
    let raised = TRACKER.guest_starts(HEAP_LIVE.load(Relaxed) + crate::jit::code_committed(), increase_limit);
    if crate::syscalls::TRACE.load(Relaxed) {
        eprintln!("[memlimit] raised by {} MiB as the program starts; no call after this", raised >> 20);
    }
}

/// Once, at start-up: a forked child counts from zero again.
pub fn init() {
    extern "C" fn child() {
        TRACKER.forked();
    }
    unsafe {
        libc::pthread_atfork(None, None, Some(child));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    const MIB: u64 = 1 << 20;

    fn run(t: &Tracker, total: u64, ok: bool, calls: &Mutex<Vec<u32>>) {
        t.note(total, |b| {
            calls.lock().unwrap().push(b);
            if ok {
                Ok(())
            } else {
                Err(libc::EPERM)
            }
        });
    }

    #[test]
    fn under_a_mib_no_call() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, MIB - 1, true, &c);
        assert!(c.lock().unwrap().is_empty());
    }

    #[test]
    fn each_mib_passed_once_in_bytes() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, MIB, true, &c);
        run(&t, MIB + 5, true, &c);
        run(&t, 3 * MIB + MIB / 2, true, &c);
        assert_eq!(*c.lock().unwrap(), vec![MIB as u32, 2 * MIB as u32]);
    }

    #[test]
    fn increase_only() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, 3 * MIB, true, &c);
        run(&t, 0, true, &c);
        run(&t, 3 * MIB + 7, true, &c);
        assert_eq!(c.lock().unwrap().len(), 1);
        run(&t, 4 * MIB, true, &c);
        assert_eq!(*c.lock().unwrap(), vec![3 * MIB as u32, MIB as u32]);
    }

    #[test]
    fn stops_after_a_refusal() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, MIB, false, &c);
        run(&t, 5 * MIB, true, &c);
        run(&t, 50 * MIB, true, &c);
        assert_eq!(*c.lock().unwrap(), vec![MIB as u32]);
    }

    #[test]
    fn a_call_stays_under_int32_max() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, 5 << 30, true, &c);
        run(&t, 5 << 30, true, &c);
        run(&t, 5 << 30, true, &c);
        let v = c.lock().unwrap();
        assert!(v.iter().all(|&b| b as u64 <= i32::MAX as u64 && b as u64 % MIB == 0), "{v:?}");
        assert_eq!(v.iter().map(|&b| b as u64).sum::<u64>(), 5 << 30);
    }

    #[test]
    fn forked_counts_from_zero() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, 3 * MIB, true, &c);
        t.forked();
        run(&t, 3 * MIB, true, &c);
        assert_eq!(*c.lock().unwrap(), vec![3 * MIB as u32, 3 * MIB as u32]);
    }

    #[test]
    fn threads_add_each_mib_once() {
        // Eight threads note the same totals at once, so they race for every MiB.
        let t = Tracker::new();
        let sum = AtomicU64::new(0);
        let go = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for _ in 0..8 {
                let (t, sum, go) = (&t, &sum, &go);
                s.spawn(move || {
                    go.wait();
                    for i in 0..20000u64 {
                        t.note(i * MIB / 4, |b| {
                            sum.fetch_add(b as u64, Relaxed);
                            Ok(())
                        });
                    }
                });
            }
        });
        // The largest total noted is 19999 / 4 MiB = 4999.75 MiB.
        assert_eq!(sum.load(Relaxed), 4999 * MIB);
    }

    #[test]
    fn as_the_guest_starts_one_call_covers_the_run_and_none_follows() {
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        run(&t, 3 * MIB, true, &c);
        let raised = t.guest_starts(5 * MIB, |b| {
            c.lock().unwrap().push(b);
            Ok(())
        });
        run(&t, 100 * MIB, true, &c);
        run(&t, 200 * MIB, true, &c);
        assert_eq!(*c.lock().unwrap(), vec![3 * MIB as u32, (UPFRONT - 3 * MIB) as u32], "before, at and after the start");
        assert_eq!(raised, UPFRONT - 3 * MIB);
        // More than UPFRONT counted already: that is what is covered.
        let (t2, c2) = (Tracker::new(), Mutex::new(vec![]));
        t2.guest_starts(UPFRONT + 7 * MIB, |b| {
            c2.lock().unwrap().push(b);
            Ok(())
        });
        assert_eq!(*c2.lock().unwrap(), vec![(UPFRONT + 7 * MIB) as u32]);
    }

    #[test]
    fn at_the_start_a_raise_the_kernel_caps_is_tried_smaller_and_a_refusal_ends_it() {
        // ENOMEM above 16 MiB (a kernel whose cap is lower): 64, 32, then 16 accepted.
        let (t, c) = (Tracker::new(), Mutex::new(vec![]));
        let raised = t.guest_starts(0, |b| {
            c.lock().unwrap().push(b);
            if b as u64 > 16 * MIB {
                Err(libc::ENOMEM)
            } else {
                Ok(())
            }
        });
        assert_eq!((c.lock().unwrap().clone(), raised), (vec![64 * MIB as u32, 32 * MIB as u32, 16 * MIB as u32], 16 * MIB));
        // Any other answer is a kernel that does not take the call: one try.
        let (t2, c2) = (Tracker::new(), Mutex::new(vec![]));
        t2.guest_starts(0, |b| {
            c2.lock().unwrap().push(b);
            Err(libc::EPERM)
        });
        assert_eq!(c2.lock().unwrap().len(), 1);
        // A tracker already off (refused before the start) makes no call at the start.
        let (t3, c3) = (Tracker::new(), Mutex::new(vec![]));
        run(&t3, MIB, false, &c3);
        t3.guest_starts(0, |b| {
            c3.lock().unwrap().push(b);
            Ok(())
        });
        assert_eq!(c3.lock().unwrap().len(), 1, "only the refused call before the start");
    }

    #[test]
    fn counting_allocator_counts_live_bytes() {
        static LIVE: AtomicU64 = AtomicU64::new(0);
        let a = Counting::new(&LIVE, false);
        unsafe {
            let l = Layout::from_size_align(1000, 8).unwrap();
            let p = a.alloc(l);
            assert_eq!(LIVE.load(Relaxed), 1000);
            let p = a.realloc(p, l, 3000);
            assert_eq!(LIVE.load(Relaxed), 3000);
            let l3 = Layout::from_size_align(3000, 8).unwrap();
            let p = a.realloc(p, l3, 500);
            assert_eq!(LIVE.load(Relaxed), 500);
            a.dealloc(p, Layout::from_size_align(500, 8).unwrap());
            let z = a.alloc_zeroed(l);
            assert_eq!(LIVE.load(Relaxed), 1000);
            a.dealloc(z, l);
        }
        assert_eq!(LIVE.load(Relaxed), 0);
    }

    #[test]
    fn the_counting_allocator_is_the_global_one() {
        let before = HEAP_LIVE.load(Relaxed) as i64;
        let v: Vec<u8> = std::hint::black_box(Vec::with_capacity(256 << 20));
        let after = HEAP_LIVE.load(Relaxed) as i64;
        drop(v);
        // Other tests allocate and free meanwhile; they do not free 64 MiB between two loads.
        assert!(after - before >= 192 << 20, "{before} -> {after}");
    }

    #[test]
    fn a_forked_child_counts_from_zero() {
        init();
        TRACKER.reported.fetch_max(1, Relaxed);
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            let r = TRACKER.reported.load(Relaxed);
            unsafe { libc::_exit(if r == 0 { 0 } else { 1 }) };
        }
        assert!(pid > 0);
        let mut st = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut st, 0) }, pid);
        assert!(libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0, "child status {st:#x}");
    }
}
