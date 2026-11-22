//! Fork without copying a translator lock another thread holds, and
//! without leaving the child anything that waits on, or wakes, a thread of the parent.
//!
//! libc::fork copies the whole address space but only the calling thread. A lock another host
//! thread held at that instant is copied held, and nothing in the child will ever release it.
//! `fork` takes every translator lock first (the pthread_atfork prepare/parent/child pattern,
//! done at the two places that fork: the guest's fork and vfork, and the AOT builder), forks,
//! releases them in the parent, and in the child writes a fresh initialiser over each one
//! (forksync), so the child starts with every lock free and with no waiter state from the
//! parent.
//!
//! Why the locks are forksync's and not std's: on macOS std's RwLock, Once,
//! OnceLock and mpsc channels park a waiting thread on its own parker, whose kernel semaphore is a
//! Mach port, and the child of a fork does not get the parent's ports. A child that wakes a thread
//! parked in the parent, or parks its forking thread when that thread's parker was used in the
//! parent, dies (libdispatch: "Use-after-free of dispatch_semaphore_t"). forksync's locks wait in
//! the kernel by address and have no parker. What else the child resets is `reset_in_child`.
//!
//! One fixed order, outermost first. An edge A -> B is a place that takes B while holding A:
//!   jit::TRANSLATE -> jit::BLOCKS (read), jit::INDEX; jit::INDEX -> aot::IMAGES, aot::SEEN,
//!   jit::BLOCKS; aot::IMAGES -> aot::SEEN, jit::BLOCKS; symbols::CACHE_IMAGES_LOADED ->
//!   cache::CACHE -> symbols::IMAGES; workq::WL -> workq::IDLE, workq::WLOG, workq::WL_MON_START.
//! The others are leaves. forksync::LAZY_GATE is taken last, and then the fork waits until no
//! FOnceLock initialiser is in flight (an initialiser takes no translator lock). The caller must
//! hold none of them (the guest's syscall path holds none). stdout and stderr are not taken: the
//! translator writes its own output without them, and a thread holding a lock keeps its output
//! until it releases it (forksync::write_out).
//!
//! Signals: the forking thread blocks every signal that is not a synchronous fault for the hold
//! window, so a guest signal handler cannot run on it while it holds every lock. The debug
//! handlers take no lock, allocate nothing and never write to fd 2 on any thread (engine.rs:
//! SIGUSR1 hands the dump, and SIGUSR2 its two lines, to a dump thread over a non-blocking
//! pipe), so a signal on another thread cannot wait for a lock the fork holds, nor block in a
//! write while its thread holds one.
//!
//! Thread state the child keeps: the forking thread's thread-locals survive. Those that name a
//! thread or kernel object of the parent are dropped or made stale by `GENERATION`, which the child
//! advances: the GPU bridge (gpu.rs), a workqueue worker's exit, parked or terminated
//! (workq::worker_after_run), and the workqueue's manager and workloop bindings
//! (workq::reset_in_child). The thread-creation handshake (threads::spawn_ready) is a heap lock
//! outside hold_all: once its thread is ready it is never locked again, so a child whose
//! forking thread was such a thread does not wait on a copy the creator held.

use crate::forksync::{FMutex, ForkHeld};
use std::sync::atomic::{AtomicU64, Ordering};

/// Every translator lock, held. Dropping it releases them in the reverse of the order taken.
pub(crate) struct Held(Vec<Box<dyn ForkHeld>>);

impl Drop for Held {
    fn drop(&mut self) {
        while let Some(g) = self.0.pop() {
            drop(g);
        }
    }
}

impl Held {
    /// Make every lock fresh (see forksync), none of them unlocked.
    ///
    /// # Safety
    /// As `ForkHeld::reinit_in_child`: only on the forking thread of a fork child, right after
    /// libc::fork returned 0, with `self` taken by this thread before the fork. `fork` below is the
    /// only caller.
    unsafe fn reinit_in_child(mut self) {
        while let Some(g) = self.0.pop() {
            g.reinit_in_child();
        }
    }
}

fn take<T: Send + 'static>(v: &mut Vec<Box<dyn ForkHeld>>, m: &'static FMutex<T>) {
    v.push(Box::new(m.lock().unwrap()));
}

/// Take every translator lock, in the order above, then the lazy-init gate, and wait until no
/// initialiser is in flight.
pub(crate) fn hold_all() -> Held {
    let mut v: Vec<Box<dyn ForkHeld>> = Vec::with_capacity(32);
    take(&mut v, &crate::jit::TRANSLATE);
    take(&mut v, &crate::jit::INDEX);
    take(&mut v, &crate::aot::IMAGES);
    take(&mut v, &crate::aot::SEEN);
    v.push(Box::new(crate::jit::BLOCKS.write().unwrap()));
    take(&mut v, &crate::symbols::CACHE_IMAGES_LOADED);
    take(&mut v, &crate::cache::CACHE);
    take(&mut v, &crate::symbols::IMAGES);
    take(&mut v, &crate::workq::WL);
    take(&mut v, &crate::workq::MANAGER);
    take(&mut v, &crate::workq::IDLE);
    take(&mut v, &crate::workq::ANON_QOS);
    take(&mut v, &crate::workq::WLOG);
    take(&mut v, &crate::workq::ANON_START);
    take(&mut v, &crate::workq::WL_MON_START);
    take(&mut v, &crate::syscalls::FD_VPATHS);
    take(&mut v, &crate::syscalls::CWD_VPATHS);
    take(&mut v, &crate::paths::CACHE_DIR);
    take(&mut v, &crate::paths::OVERLAY_ROOT);
    take(&mut v, &crate::signals::ACTIONS);
    take(&mut v, &crate::iokit::FAKE_PORTS);
    take(&mut v, &crate::spawn::FORWARD);
    take(&mut v, &crate::jit::HIST);
    take(&mut v, &crate::engine::CPUS);
    take(&mut v, &crate::forksync::LAZY_GATE);
    crate::forksync::wait_for_lazy_inits();
    Held(v)
}

/// Advanced in every fork child. State kept in a thread-local that names a thread or kernel object
/// of the process that made it records the generation it was made in, and is not used once the
/// generation has moved on.
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::SeqCst)
}

/// Block every signal but the synchronous faults on this thread; returns the old mask.
fn block_async_signals() -> libc::sigset_t {
    unsafe {
        let mut all: libc::sigset_t = std::mem::zeroed();
        let mut old: libc::sigset_t = std::mem::zeroed();
        libc::sigfillset(&mut all);
        for s in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE, libc::SIGTRAP] {
            libc::sigdelset(&mut all, s);
        }
        libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut old);
        old
    }
}

fn restore_signals(old: &libc::sigset_t) {
    unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, old, std::ptr::null_mut()) };
}

/// In the fork child, with every lock fresh: drop state that names threads or kernel objects
/// the child does not have (workq's idle workers, manager slot, workloops, kqueues and the forking
/// thread's own bindings; the CPU registry; the dump thread; the live-thread count).
fn reset_in_child() {
    GENERATION.fetch_add(1, Ordering::SeqCst);
    crate::workq::reset_in_child();
    crate::engine::reset_in_child();
    crate::threads::LIVE_THREADS.store(1, std::sync::atomic::Ordering::SeqCst);
}

/// libc::fork with every translator lock held across it (see the module comment).
pub fn fork() -> libc::pid_t {
    let mask = block_async_signals();
    let held = hold_all();
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        // Safety: the forking thread of the child, right after the fork, with `held` taken by this
        // thread before it; the child has no other thread yet.
        unsafe {
            held.reinit_in_child();
            crate::forksync::reset_held_in_child();
            crate::forksync::reset_lazy_in_child();
        }
        reset_in_child();
    } else {
        drop(held);
    }
    restore_signals(&mask);
    pid
}

#[cfg(test)]
#[path = "forksafe_adapt.rs"]
mod adapt;
#[cfg(test)]
#[path = "forksafe_v4tests.rs"]
pub(crate) mod v4tests;
#[cfg(test)]
#[path = "forksafe_v5tests.rs"]
mod v5tests;

#[cfg(test)]
mod tests {
    use super::v4tests::{code_only, rs_files, scan};
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Every translator lock, by name, enumerated here and NOT through hold_all, so a lock left
    /// out of hold_all is caught. Each entry try-locks its lock and releases it at once: true
    /// when the lock was free.
    fn named() -> Vec<(&'static str, fn() -> bool)> {
        vec![
            ("jit::TRANSLATE", || crate::jit::TRANSLATE.try_lock().is_ok()),
            ("jit::INDEX", || crate::jit::INDEX.try_lock().is_ok()),
            ("jit::BLOCKS", || crate::jit::BLOCKS.try_write().is_ok()),
            ("jit::HIST", || crate::jit::HIST.try_lock().is_ok()),
            ("aot::IMAGES", || crate::aot::IMAGES.try_lock().is_ok()),
            ("aot::SEEN", || crate::aot::SEEN.try_lock().is_ok()),
            ("symbols::CACHE_IMAGES_LOADED", || crate::symbols::CACHE_IMAGES_LOADED.try_lock().is_ok()),
            ("symbols::IMAGES", || crate::symbols::IMAGES.try_lock().is_ok()),
            ("cache::CACHE", || crate::cache::CACHE.try_lock().is_ok()),
            ("workq::WL", || crate::workq::WL.try_lock().is_ok()),
            ("workq::MANAGER", || crate::workq::MANAGER.try_lock().is_ok()),
            ("workq::IDLE", || crate::workq::IDLE.try_lock().is_ok()),
            ("workq::ANON_QOS", || crate::workq::ANON_QOS.try_lock().is_ok()),
            ("workq::WLOG", || crate::workq::WLOG.try_lock().is_ok()),
            ("workq::ANON_START", || crate::workq::ANON_START.try_lock().is_ok()),
            ("workq::WL_MON_START", || crate::workq::WL_MON_START.try_lock().is_ok()),
            ("syscalls::FD_VPATHS", || crate::syscalls::FD_VPATHS.try_lock().is_ok()),
            ("syscalls::CWD_VPATHS", || crate::syscalls::CWD_VPATHS.try_lock().is_ok()),
            ("paths::CACHE_DIR", || crate::paths::CACHE_DIR.try_lock().is_ok()),
            ("paths::OVERLAY_ROOT", || crate::paths::OVERLAY_ROOT.try_lock().is_ok()),
            ("signals::ACTIONS", || crate::signals::ACTIONS.try_lock().is_ok()),
            ("iokit::FAKE_PORTS", || crate::iokit::FAKE_PORTS.try_lock().is_ok()),
            ("spawn::FORWARD", || crate::spawn::FORWARD.try_lock().is_ok()),
            ("engine::CPUS", || crate::engine::CPUS.try_lock().is_ok()),
            ("forksync::LAZY_GATE", || crate::forksync::LAZY_GATE.try_lock().is_ok()),
        ]
    }

    /// hold_all's order (the module comment's), outermost first.
    const HOLD_ORDER: &[&str] = &[
        "jit::TRANSLATE", "jit::INDEX", "aot::IMAGES", "aot::SEEN", "jit::BLOCKS", "symbols::CACHE_IMAGES_LOADED",
        "cache::CACHE", "symbols::IMAGES", "workq::WL", "workq::MANAGER", "workq::IDLE", "workq::ANON_QOS",
        "workq::WLOG", "workq::ANON_START", "workq::WL_MON_START", "syscalls::FD_VPATHS", "syscalls::CWD_VPATHS", "paths::CACHE_DIR",
        "paths::OVERLAY_ROOT", "signals::ACTIONS", "iokit::FAKE_PORTS", "spawn::FORWARD", "jit::HIST",
        "engine::CPUS", "forksync::LAZY_GATE",
    ];

    /// Every static FCondvar, each made fresh by a reset_in_child.
    const CONDVARS: &[&str] = &["workq::WL_CV"];

    /// Every source file compiled into maclator (this crate and maclator-core): (module, path from
    /// the workspace's crates/ directory, text as compiled). `every_compiled_source_is_scanned`
    /// checks the paths against the files on disk.
    const SOURCES: &[(&str, &str, &str)] = &[
        ("aot", "maclator/src/aot.rs", include_str!("aot.rs")),
        ("bundle", "maclator/src/bundle.rs", include_str!("bundle.rs")),
        ("cache", "maclator/src/cache.rs", include_str!("cache.rs")),
        ("commpage", "maclator/src/commpage.rs", include_str!("commpage.rs")),
        ("engine", "maclator/src/engine.rs", include_str!("engine.rs")),
        ("forksafe", "maclator/src/forksafe.rs", include_str!("forksafe.rs")),
        ("forksafe::adapt", "maclator/src/forksafe_adapt.rs", include_str!("forksafe_adapt.rs")),
        ("forksafe::v4tests", "maclator/src/forksafe_v4tests.rs", include_str!("forksafe_v4tests.rs")),
        ("forksafe::v5tests", "maclator/src/forksafe_v5tests.rs", include_str!("forksafe_v5tests.rs")),
        ("forksync", "maclator/src/forksync.rs", include_str!("forksync.rs")),
        ("gpu", "maclator/src/gpu.rs", include_str!("gpu.rs")),
        ("guestmem", "maclator/src/guestmem.rs", include_str!("guestmem.rs")),
        ("hostsys", "maclator/src/hostsys.rs", include_str!("hostsys.rs")),
        ("iokit", "maclator/src/iokit.rs", include_str!("iokit.rs")),
        ("jit", "maclator/src/jit/mod.rs", include_str!("jit/mod.rs")),
        ("jit::budget_flags_selftest", "maclator/src/jit/budget_flags_selftest.rs", include_str!("jit/budget_flags_selftest.rs")),
        ("jit::csel_selftest", "maclator/src/jit/csel_selftest.rs", include_str!("jit/csel_selftest.rs")),
        ("jit::emit", "maclator/src/jit/emit.rs", include_str!("jit/emit.rs")),
        ("jit::fp_mode_selftest", "maclator/src/jit/fp_mode_selftest.rs", include_str!("jit/fp_mode_selftest.rs")),
        ("jit::loop_selftest", "maclator/src/jit/loop_selftest.rs", include_str!("jit/loop_selftest.rs")),
        ("jit::selftest", "maclator/src/jit/selftest.rs", include_str!("jit/selftest.rs")),
        ("jit::signal_budget_selftest", "maclator/src/jit/signal_budget_selftest.rs", include_str!("jit/signal_budget_selftest.rs")),
        ("loader", "maclator/src/loader.rs", include_str!("loader.rs")),
        ("macho", "maclator/src/macho.rs", include_str!("macho.rs")),
        ("memlimit", "maclator/src/memlimit.rs", include_str!("memlimit.rs")),
        ("main", "maclator/src/main.rs", include_str!("main.rs")),
        ("paths", "maclator/src/paths.rs", include_str!("paths.rs")),
        ("signals", "maclator/src/signals.rs", include_str!("signals.rs")),
        ("slices", "maclator/src/slices.rs", include_str!("slices.rs")),
        ("spawn", "maclator/src/spawn.rs", include_str!("spawn.rs")),
        ("symbols", "maclator/src/symbols.rs", include_str!("symbols.rs")),
        ("syscalls", "maclator/src/syscalls.rs", include_str!("syscalls.rs")),
        ("threads", "maclator/src/threads.rs", include_str!("threads.rs")),
        ("workq", "maclator/src/workq.rs", include_str!("workq.rs")),
        ("core", "core/src/lib.rs", include_str!("../../core/src/lib.rs")),
        ("core::cpu", "core/src/cpu.rs", include_str!("../../core/src/cpu.rs")),
        ("core::disasm", "core/src/disasm.rs", include_str!("../../core/src/disasm.rs")),
        ("core::mem", "core/src/mem.rs", include_str!("../../core/src/mem.rs")),
        ("core::softfloat", "core/src/softfloat.rs", include_str!("../../core/src/softfloat.rs")),
        ("core::interp", "core/src/interp/mod.rs", include_str!("../../core/src/interp/mod.rs")),
        ("core::interp::dp", "core/src/interp/dp.rs", include_str!("../../core/src/interp/dp.rs")),
        ("core::interp::fp16", "core/src/interp/fp16.rs", include_str!("../../core/src/interp/fp16.rs")),
        ("core::interp::ldst", "core/src/interp/ldst.rs", include_str!("../../core/src/interp/ldst.rs")),
        ("core::interp::simd", "core/src/interp/simd.rs", include_str!("../../core/src/interp/simd.rs")),
        ("core::interp::sys", "core/src/interp/sys.rs", include_str!("../../core/src/interp/sys.rs")),
    ];

    /// Statics whose type holds a lock or channel inside one of the crate's structs, and why that is
    /// safe across a fork. The scan must find each, and nothing else.
    const STRUCT_HELD_OK: &[(&str, &str)] = &[
        ("workq::IDLE", "each Worker holds an mpsc Sender; the child forgets every Worker (workq::reset_in_child)"),
        ("gpu::BRIDGE_THREAD", "a thread-local; a BridgeThread made before a fork is stale in the child and forgotten, never used or dropped (gpu.rs)"),
        ("forksafe::v4tests::LAZY_UNDER_TEST", "test only: an FOnceLock through the adapter's alias"),
        ("forksafe::v4tests::LAZY_RAW", "test only: an FOnceLock through the adapter's alias"),
    ];

    /// SOURCES is every .rs file of this crate and of maclator-core, so a new module cannot escape
    /// the scan.
    #[test]
    fn every_compiled_source_is_scanned() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut want: Vec<String> = rs_files(&crates.join("maclator/src")).into_iter().map(|f| format!("maclator/src/{f}")).collect();
        want.extend(rs_files(&crates.join("core/src")).into_iter().map(|f| format!("core/src/{f}")));
        want.sort();
        let mut listed: Vec<String> = SOURCES.iter().map(|(_, p, _)| p.to_string()).collect();
        listed.sort();
        assert!(want.len() > 30, "found only {} files on disk", want.len());
        assert_eq!(listed, want, "SOURCES and the .rs files on disk differ");
        for (_, p, text) in SOURCES {
            let disk = std::fs::read_to_string(crates.join(p)).unwrap();
            assert_eq!(&disk, text, "{p}: the compiled text differs from the file on disk (rebuild)");
        }
    }

    /// The named list is every lock the source declares, the source declares no std lock (or
    /// channel, Once, OnceLock, LazyLock) in a static, every static FCondvar is reset, no static
    /// holds a lock inside another type or a struct. What this cannot see: locks made at run time
    /// in heap objects (threads::spawn_ready's handshake, workq's per-worker channels, gpu's
    /// Shared), locks inside dependencies (std's own: the environment lock, the thread table,
    /// stdio; libc; iced-x86), and statics a macro other than thread_local! would generate.
    #[test]
    fn the_named_locks_are_every_static_lock_in_the_source() {
        let sc = scan(SOURCES);
        // Control: a plain count of declarations that start a line (J v3's parser, on the source
        // with comments and strings blanked) is a floor for the scan, which also reads the rest.
        let line_start = SOURCES
            .iter()
            .map(|(_, _, t)| {
                code_only(t)
                    .lines()
                    .filter(|l| {
                        let l = l.trim_start();
                        ["static ", "pub static ", "pub(crate) static ", "pub(super) static "].iter().any(|p| l.starts_with(p))
                    })
                    .count()
            })
            .sum::<usize>();
        assert!(line_start > 50 && sc.statics >= line_start, "the scan found {} statics, the line count {line_start}", sc.statics);
        assert!(sc.std_sync.is_empty(), "std synchronisation in statics (a fork child cannot reset them): {:?}", sc.std_sync);
        let mut held: Vec<&str> = sc.held_in_struct.iter().map(|h| h.split(": ").next().unwrap()).collect();
        held.sort();
        let mut ok: Vec<&str> = STRUCT_HELD_OK.iter().map(|(n, _)| *n).collect();
        ok.sort();
        assert_eq!(held, ok, "statics holding a lock inside a struct (each needs a reason in STRUCT_HELD_OK): {:?}", sc.held_in_struct);
        assert!(sc.nested.is_empty(), "a forksync type nested inside a static's type: {:?}", sc.nested);
        let names: Vec<&str> = named().iter().map(|(n, _)| *n).collect();
        for f in &sc.locks {
            assert!(names.contains(&f.as_str()), "{f} is declared but not in the named list");
        }
        for n in &names {
            assert!(sc.locks.iter().any(|f| f == n), "{n} is named but not declared");
        }
        assert_eq!(sc.locks.len(), names.len());
        let mut cv = sc.condvars.clone();
        cv.sort();
        assert_eq!(cv, CONDVARS, "static FCondvars");
        assert!(sc.oncelocks.len() >= 10, "FOnceLocks found: {:?}", sc.oncelocks);
    }

    /// hold_all takes the locks in HOLD_ORDER, read from its source.
    #[test]
    fn hold_all_takes_the_locks_in_the_documented_order() {
        let code = code_only(include_str!("forksafe.rs"));
        let start = code.find("pub(crate) fn hold_all() -> Held {").expect("hold_all");
        let body = &code[start..start + code[start..].find("\n}\n").unwrap()];
        let mut got = Vec::new();
        for line in body.lines() {
            let l = line.trim();
            if let Some(r) = l.strip_prefix("take(&mut v, &crate::") {
                got.push(r.trim_end_matches(");").to_string());
            } else if l.starts_with("v.push(Box::new(crate::jit::BLOCKS.write()") {
                got.push("jit::BLOCKS".to_string());
            }
        }
        assert_eq!(got, HOLD_ORDER);
        let mut sorted_named: Vec<&str> = named().iter().map(|(n, _)| *n).collect();
        let mut sorted_order = HOLD_ORDER.to_vec();
        sorted_named.sort();
        sorted_order.sort();
        assert_eq!(sorted_named, sorted_order, "HOLD_ORDER and the named list differ");
    }

    /// While one thread holds hold_all(), every named lock is busy; after, every one is free.
    #[test]
    fn hold_all_takes_every_named_lock() {
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let h = std::thread::spawn(move || {
            let g = hold_all();
            held_tx.send(()).unwrap();
            let _ = go_rx.recv();
            drop(g);
        });
        held_rx.recv().unwrap();
        let free: Vec<&str> = named().iter().filter(|(_, t)| t()).map(|(n, _)| *n).collect();
        go_tx.send(()).unwrap();
        h.join().unwrap();
        assert!(free.is_empty(), "free while hold_all held: {free:?}");
        // Other tests in this binary take the locks too, so a lock may be busy for a moment.
        for (n, t) in named() {
            let t0 = std::time::Instant::now();
            while !t() {
                assert!(t0.elapsed() < std::time::Duration::from_secs(5), "{n} still held 5 s after release");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// The exit status of a child: 0 when every named lock was free in it (exit 10 + i names the
    /// first that was not), checked under a 3 s alarm.
    fn child_checks_every_lock() -> ! {
        unsafe { libc::alarm(3) };
        for (i, (_, t)) in named().iter().enumerate() {
            if !t() {
                unsafe { libc::_exit(10 + i as i32) };
            }
        }
        drop(hold_all());
        unsafe { libc::_exit(0) };
    }

    fn wait_status(pid: libc::pid_t) -> i32 {
        let mut st = 0;
        loop {
            let r = unsafe { libc::waitpid(pid, &mut st, 0) };
            if r == pid {
                return st;
            }
            assert!(r < 0 && unsafe { *libc::__error() } == libc::EINTR, "waitpid");
        }
    }

    fn exited_0(st: i32) -> bool {
        libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0
    }

    /// A helper thread holds every lock while this thread forks.
    fn fork_beside_a_holder(raw: bool) -> i32 {
        let ready = Arc::new(AtomicBool::new(false));
        let r = ready.clone();
        let h = std::thread::spawn(move || {
            let g = hold_all();
            r.store(true, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(g);
        });
        while !ready.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        let pid = if raw { unsafe { libc::fork() } } else { fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            child_checks_every_lock();
        }
        h.join().unwrap();
        wait_status(pid)
    }

    #[test]
    fn a_child_forked_beside_a_lock_holder_finds_every_lock_free() {
        let st = fork_beside_a_holder(false);
        assert!(exited_0(st), "child status {st:#x}");
    }

    #[test]
    fn must_fail_control_a_raw_fork_beside_a_lock_holder_leaves_a_lock_held() {
        let st = fork_beside_a_holder(true);
        assert!(!exited_0(st), "the raw fork's child found every lock free");
    }

    /// Start `f(i)` on `n` new host threads (forksync::spawn_host); the returned closure waits for
    /// all of them. The wait is a pthread mutex and condvar, so it parks no std parker.
    fn on_host_threads(n: usize, f: fn(usize)) -> impl FnOnce() {
        let left = Arc::new((std::sync::Mutex::new(n), std::sync::Condvar::new()));
        for i in 0..n {
            let l = left.clone();
            crate::forksync::spawn_host(
                "",
                1 << 20,
                Box::new(move || {
                    f(i);
                    *l.0.lock().unwrap() -= 1;
                    l.1.notify_all();
                }),
            )
            .unwrap();
        }
        move || {
            let mut g = left.0.lock().unwrap();
            while *g > 0 {
                g = left.1.wait(g).unwrap();
            }
        }
    }

    /// Four threads of the parent keep starting and ending std threads (std's thread table and its
    /// mutex are busy) while this thread forks; each child starts host threads, which must not
    /// wait on that table. 1,000 forks: the hazard hits about 1 child in a few hundred.
    fn churn_and_fork(child: fn() -> i32, forks: usize, stop_at_first_bad: bool) -> usize {
        let stop = Arc::new(AtomicBool::new(false));
        let churn: Vec<_> = (0..4)
            .map(|_| {
                let s = stop.clone();
                std::thread::spawn(move || {
                    while !s.load(Ordering::Relaxed) {
                        let _ = std::thread::spawn(|| {}).join();
                    }
                })
            })
            .collect();
        let mut bad = 0;
        for _ in 0..forks {
            if stop_at_first_bad && bad > 0 {
                break;
            }
            let pid = fork();
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                unsafe { libc::alarm(3) };
                let rc = child();
                unsafe { libc::_exit(rc) };
            }
            if !exited_0(wait_status(pid)) {
                bad += 1;
            }
        }
        stop.store(true, Ordering::Relaxed);
        for c in churn {
            c.join().unwrap();
        }
        bad
    }

    #[test]
    fn a_child_starts_host_threads_while_the_parent_churns_std_threads() {
        let bad = churn_and_fork(|| {
            for _ in 0..5 {
                if crate::threads::spawn_ready(1 << 20, |ready| ready()).is_err() {
                    return 7;
                }
            }
            on_host_threads(2, |_| {})();
            0
        }, 1000, false);
        assert_eq!(bad, 0, "children that hung or died, of 1,000");
    }

    /// The hazard spawn_host avoids, shown on std: a child that starts std threads while the parent
    /// churns them. A few children hang, so this is a must-fail control: it forks until one does (up
    /// to 1,000) and passes when one did.
    #[test]
    fn must_fail_control_a_child_starting_std_threads_beside_std_churn_sometimes_hangs() {
        let bad = churn_and_fork(|| {
            for _ in 0..5 {
                let _ = std::thread::spawn(|| {}).join();
            }
            0
        }, 1000, true);
        assert!(bad > 0, "no child of 1,000 hung: the control saw nothing");
    }

    /// Make sure this thread's parker has slept once, so its semaphore exists in the parent.
    fn use_my_parker() {
        // Twice: a stale unpark token makes the first return at once.
        std::thread::park_timeout(std::time::Duration::from_millis(2));
        std::thread::park_timeout(std::time::Duration::from_millis(2));
    }

    /// Threads queue on the locks while fork holds them; this thread's parker has slept in the
    /// parent; each child then contends BLOCKS and a path lock from two new threads and from the
    /// forking thread. Every child must exit 0.
    #[test]
    fn a_child_forked_while_threads_wait_on_the_locks_survives() {
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let s = stop.clone();
                std::thread::spawn(move || {
                    while !s.load(Ordering::Relaxed) {
                        let _ = crate::jit::lookup(0);
                        drop(crate::paths::cache_dir());
                    }
                })
            })
            .collect();
        use_my_parker();
        let mut bad = 0;
        for _ in 0..200 {
            let pid = fork();
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                unsafe { libc::alarm(3) };
                let done = on_host_threads(2, |i| {
                    for _ in 0..200 {
                        crate::jit::register_block(0x1000 + i as u64, 0x2000);
                        let _ = crate::jit::lookup(0x1000);
                        drop(crate::paths::cache_dir());
                    }
                });
                for _ in 0..200 {
                    let _ = crate::jit::lookup(0x1001);
                    drop(crate::paths::cache_dir());
                }
                done();
                unsafe { libc::_exit(0) };
            }
            if !exited_0(wait_status(pid)) {
                bad += 1;
            }
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(bad, 0, "children that hung or died, of 200");
    }

    /// The hazard the design avoids, shown on std: a child that wakes a thread parked in the parent
    /// (here through an mpsc channel) does not exit 0.
    #[test]
    fn must_fail_control_waking_a_parent_era_receiver_kills_the_child() {
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        let h = std::thread::spawn(move || rx.recv().unwrap_or(0));
        std::thread::sleep(std::time::Duration::from_millis(100));
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            unsafe { libc::alarm(3) };
            let _ = tx.send(1);
            std::thread::sleep(std::time::Duration::from_millis(200));
            unsafe { libc::_exit(0) };
        }
        let st = wait_status(pid);
        tx.send(2).unwrap();
        h.join().unwrap();
        assert!(!exited_0(st), "a child woke a parent-era receiver and exited 0");
    }

    /// workq: a worker idle in the parent (parked on its channel) is not in the child's IDLE.
    #[test]
    fn a_child_has_no_parent_era_idle_worker() {
        let h = crate::workq::test_park_an_idle_worker();
        assert_eq!(crate::workq::test_idle_len(), 1);
        let pid = fork();
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            unsafe { libc::alarm(3) };
            let n = crate::workq::test_idle_len();
            unsafe { libc::_exit(if n == 0 { 0 } else { 1 }) };
        }
        let st = wait_status(pid);
        crate::workq::test_clear_idle();
        h.join().unwrap();
        assert!(exited_0(st), "the child still had a parent-era idle worker (status {st:#x})");
    }

    /// threads::spawn_ready from a child's forking thread, whose parker slept in the parent: the
    /// handshake must not park that thread.
    #[test]
    fn a_child_creates_host_threads_after_its_forking_thread_slept_in_the_parent() {
        let mut bad = 0;
        for _ in 0..20 {
            use_my_parker();
            let pid = fork();
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                unsafe { libc::alarm(3) };
                for _ in 0..20 {
                    if crate::threads::spawn_ready(1 << 20, |ready| ready()).is_err() {
                        unsafe { libc::_exit(7) };
                    }
                }
                unsafe { libc::_exit(0) };
            }
            if !exited_0(wait_status(pid)) {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "children that hung or died creating threads, of 20");
    }

    /// Run one #[ignore] scenario below in a fresh copy of this test binary, with stdout and
    /// stderr on pipes: stderr is never read. Returns its stdout when it exited 0 within 20 s.
    fn run_scenario(name: &str) -> Result<String, String> {
        use std::io::Read;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut c = std::process::Command::new(exe)
            .args([&format!("forksafe::tests::{name}"), "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env("MCL_FORKSAFE_SCENARIO", name)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        let t0 = std::time::Instant::now();
        let st = loop {
            if let Some(s) = c.try_wait().map_err(|e| e.to_string())? {
                break s;
            }
            if t0.elapsed() > std::time::Duration::from_secs(20) {
                let _ = c.kill();
                let _ = c.wait();
                return Err(format!("{name}: still running after 20 s (killed)"));
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        let mut out = String::new();
        let _ = c.stdout.take().unwrap().read_to_string(&mut out);
        if st.success() {
            Ok(out)
        } else {
            Err(format!("{name}: {st}"))
        }
    }

    fn scenario(name: &str) -> bool {
        std::env::var("MCL_FORKSAFE_SCENARIO").map(|v| v == name).unwrap_or(false)
    }

    fn say(s: &str) {
        unsafe { libc::write(1, s.as_ptr() as *const libc::c_void, s.len()) };
    }

    /// fd 2 is a pipe nobody reads, filled; one thread is blocked in the crate's eprintln! and
    /// one holds std's stderr lock while blocked writing. A fork must still return.
    #[test]
    fn a_fork_returns_while_threads_are_blocked_on_a_full_stderr_pipe() {
        let out = run_scenario("scenario_full_stderr").unwrap();
        assert!(out.contains("FORK-RETURNED"), "no marker: {out}");
    }

    #[test]
    #[ignore]
    fn scenario_full_stderr() {
        if !scenario("scenario_full_stderr") {
            return;
        }
        unsafe {
            let fl = libc::fcntl(2, libc::F_GETFL);
            libc::fcntl(2, libc::F_SETFL, fl | libc::O_NONBLOCK);
            let b = [b'x'; 4096];
            while libc::write(2, b.as_ptr() as *const libc::c_void, b.len()) > 0 {}
            libc::fcntl(2, libc::F_SETFL, fl & !libc::O_NONBLOCK);
        }
        std::thread::spawn(|| eprintln!("blocked on a full pipe"));
        std::thread::spawn(|| {
            use std::io::Write;
            let mut l = std::io::stderr().lock();
            let _ = l.write_all(&[b'y'; 8192]);
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        let pid = fork();
        if pid == 0 {
            unsafe { libc::_exit(0) };
        }
        let st = wait_status(pid);
        say(&format!("FORK-RETURNED child {}\n", if exited_0(st) { "ok" } else { "bad" }));
        unsafe { libc::_exit(if exited_0(st) { 0 } else { 1 }) };
    }

    extern "C" fn take_symbol_locks(_: i32) {
        let a = crate::symbols::CACHE_IMAGES_LOADED.lock();
        let b = crate::symbols::IMAGES.lock();
        drop(b);
        drop(a);
    }

    /// SIGUSR1 aimed at the forking thread, over and over, with a handler that takes the symbol
    /// locks as dump_threads does. Every fork must return and every child exit 0.
    #[test]
    fn a_fork_survives_a_signal_storm_on_the_forking_thread() {
        let out = run_scenario("scenario_signal_storm").unwrap();
        assert!(out.contains("FORKS-DONE bad=0"), "{out}");
    }

    #[test]
    #[ignore]
    fn scenario_signal_storm() {
        if !scenario("scenario_signal_storm") {
            return;
        }
        unsafe { libc::signal(libc::SIGUSR1, take_symbol_locks as usize) };
        let me = unsafe { libc::pthread_self() } as usize;
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let sender = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                unsafe { libc::pthread_kill(me as libc::pthread_t, libc::SIGUSR1) };
                unsafe { libc::usleep(20) };
            }
        });
        let mut bad = 0;
        for _ in 0..300 {
            let pid = fork();
            if pid == 0 {
                unsafe { libc::_exit(0) };
            }
            if !exited_0(wait_status(pid)) {
                bad += 1;
            }
        }
        stop.store(true, Ordering::Relaxed);
        sender.join().unwrap();
        say(&format!("FORKS-DONE bad={bad}\n"));
        unsafe { libc::_exit(0) };
    }
}
