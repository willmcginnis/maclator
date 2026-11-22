//! Tests for the fork-safe translator locks (second set). This file
//! is the same in this tree and in the earlier-version tree used to show each test failing
//! there; what differs between the two trees is forksafe_adapt.rs and the scanner in
//! forksafe_v4tests.rs.
#![allow(clippy::all)]

use super::adapt;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};

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

/// Run `child` in a fork child (through forksafe::fork) under a 3 s alarm: its return is the exit
/// status. Returns the raw wait status.
fn in_child(child: impl FnOnce() -> i32) -> i32 {
    let pid = crate::forksafe::fork();
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        unsafe { libc::alarm(3) };
        let rc = child();
        unsafe { libc::_exit(rc) };
    }
    wait_status(pid)
}

fn say(s: &str) {
    unsafe { libc::write(1, s.as_ptr() as *const libc::c_void, s.len()) };
}

/// Run `scenario_<name>` (an #[ignore] test below) in a fresh copy of this test binary, with stdout
/// and stderr on pipes; stderr is never read. Its stdout when it exited 0 within `secs`.
fn run_scenario(name: &str, secs: u64) -> Result<String, String> {
    use std::io::Read;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut c = std::process::Command::new(exe)
        .args([&format!("forksafe::v5tests::{name}"), "--exact", "--ignored", "--nocapture", "--test-threads=1"])
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
        if t0.elapsed() > std::time::Duration::from_secs(secs) {
            let _ = c.kill();
            let _ = c.wait();
            return Err(format!("{name}: still running after {secs} s (killed)"));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let mut out = String::new();
    let _ = c.stdout.take().unwrap().read_to_string(&mut out);
    if st.success() {
        Ok(out)
    } else {
        Err(format!("{name}: {st}; stdout: {out}"))
    }
}

fn scenario(name: &str) -> bool {
    std::env::var("MCL_FORKSAFE_SCENARIO").map(|v| v == name).unwrap_or(false)
}

/// Fill fd 2 (a pipe nobody reads) so that the next write to it blocks.
fn fill_stderr_pipe() {
    unsafe {
        let fl = libc::fcntl(2, libc::F_GETFL);
        libc::fcntl(2, libc::F_SETFL, fl | libc::O_NONBLOCK);
        let b = [b'x'; 4096];
        while libc::write(2, b.as_ptr() as *const libc::c_void, b.len()) > 0 {}
        libc::fcntl(2, libc::F_SETFL, fl & !libc::O_NONBLOCK);
    }
}

// ---- the handshake's cleanup in a fork child ----

/// A thread made by spawn_ready calls ready, then forks while its creator holds the handshake's
/// mutex (the creator woke from the wait and has not let go yet; here the test holds it on purpose).
/// In the child the thread returns: its cleanup must not wait on the copied mutex, so the child
/// ends (exit 0). Exit 5: the control failed (the child's copy of the mutex was not held, so the
/// scenario did not set up the hazard); SIGALRM (0xe): the cleanup waited for good.
#[test]
fn a_ready_thread_that_forks_ends_in_the_child_without_the_handshake_lock() {
    let out = run_scenario("scenario_handshake_held_at_fork", 30).unwrap();
    assert!(out.contains("CHILD exit 0"), "{out}");
}

#[test]
#[ignore]
fn scenario_handshake_held_at_fork() {
    if !scenario("scenario_handshake_held_at_fork") {
        return;
    }
    static GO: AtomicBool = AtomicBool::new(false);
    static PID: AtomicI32 = AtomicI32::new(0);
    let r = adapt::spawn_ready(|ready| {
        ready();
        while !GO.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        let pid = crate::forksafe::fork();
        if pid == 0 {
            unsafe { libc::alarm(3) };
            // The control: the child's copy of the handshake mutex is held (by the creator, which
            // is not in the child).
            if crate::threads::test_last_handshake().try_lock().is_ok() {
                unsafe { libc::_exit(5) };
            }
            return; // the thread ends here, in the child: its cleanup runs, then the process exits
        }
        PID.store(pid, Ordering::SeqCst);
    });
    assert!(r.is_ok(), "spawn_ready");
    let hs = crate::threads::test_last_handshake();
    let g = hs.lock().unwrap();
    GO.store(true, Ordering::SeqCst);
    let t0 = std::time::Instant::now();
    while PID.load(Ordering::SeqCst) == 0 {
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "the thread did not fork");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let st = wait_status(PID.load(Ordering::SeqCst));
    drop(g);
    if libc::WIFEXITED(st) {
        say(&format!("CHILD exit {}\n", libc::WEXITSTATUS(st)));
    } else {
        say(&format!("CHILD status {st:#x}\n"));
    }
    unsafe { libc::_exit(0) };
}

// ---- a stale workqueue worker that terminated ----

/// A workqueue worker made before the fork is the forking thread of the child. Whether its guest
/// thread parked or terminated, it must forget its channel ends in the child, never drop them;
/// here "forgotten" is seen as a send on another clone still succeeding (a dropped receiver makes
/// it fail). Controls in the parent: a worker that terminated drops its ends (the send fails), and
/// one that parked goes back to the idle wait with them.
#[test]
fn a_stale_worker_forgets_its_channel_ends_whether_it_parked_or_terminated() {
    std::thread::spawn(|| {
        let gen0 = adapt::generation();
        // controls, in this process (the worker is not stale)
        let (tx, rx) = std::sync::mpsc::channel::<u8>();
        let other = tx.clone();
        assert!(adapt::worker_exit(false, adapt::worker_is_stale(gen0), rx, tx).is_none(), "a terminated worker did not end");
        assert!(other.send(1).is_err(), "control: a worker of this process that terminated did not drop its receiver");
        let (tx, rx) = std::sync::mpsc::channel::<u8>();
        assert!(adapt::worker_exit(true, adapt::worker_is_stale(gen0), rx, tx).is_some(), "control: a parked worker did not go back to the idle wait");
        // the child: made before the fork, so stale
        for parked in [true, false] {
            let (tx, rx) = std::sync::mpsc::channel::<u8>();
            let other = tx.clone();
            let st = in_child(move || {
                if !adapt::worker_is_stale(gen0) {
                    return 4;
                }
                if adapt::worker_exit(parked, true, rx, tx).is_some() {
                    return 3;
                }
                if other.send(1).is_ok() { 0 } else { 1 }
            });
            assert!(exited_0(st), "parked={parked}: child status {st:#x} (exit 1: the channel ends were dropped; 3: it went back to the idle wait; 4: not stale)");
        }
    })
    .join()
    .unwrap();
}

// ---- SIGUSR2 at a thread that holds a translator lock, with stderr full ----

/// fd 2 is a full pipe nobody reads; a thread with a guest CPU holds jit::TRANSLATE and gets
/// SIGUSR2 (the stress tool sends it to a hung child). A fork from another thread must still
/// return, and the handler must have handled the signal (the positive control).
#[test]
fn a_fork_returns_while_a_lock_holder_takes_sigusr2_with_stderr_full() {
    let out = run_scenario("scenario_usr2_under_lock", 20).unwrap();
    assert!(out.contains("FORK-RETURNED child ok"), "{out}");
    let n: u64 = out.split("usr2=").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse().ok()).unwrap_or(0);
    assert!(n > 0, "positive control: the SIGUSR2 handler did not run: {out}");
}

#[test]
#[ignore]
fn scenario_usr2_under_lock() {
    if !scenario("scenario_usr2_under_lock") {
        return;
    }
    crate::engine::install_debug_handler();
    fill_stderr_pipe();
    static TID: AtomicUsize = AtomicUsize::new(0);
    static HOLDING: AtomicBool = AtomicBool::new(false);
    let h = std::thread::spawn(|| {
        let cpu: &'static mut maclator_core::cpu::Cpu = Box::leak(maclator_core::cpu::Cpu::new());
        crate::engine::test_set_current_cpu(cpu as *mut _ as usize);
        TID.store(unsafe { libc::pthread_self() } as usize, Ordering::SeqCst);
        let g = crate::jit::TRANSLATE.lock().unwrap();
        HOLDING.store(true, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(500));
        drop(g);
        crate::engine::test_set_current_cpu(0);
    });
    while !HOLDING.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    unsafe { libc::pthread_kill(TID.load(Ordering::SeqCst) as libc::pthread_t, libc::SIGUSR2) };
    std::thread::sleep(std::time::Duration::from_millis(100));
    let pid = crate::forksafe::fork();
    if pid == 0 {
        unsafe { libc::_exit(0) };
    }
    let st = wait_status(pid);
    h.join().unwrap();
    say(&format!("FORK-RETURNED child {} usr2={}\n", if exited_0(st) { "ok" } else { "bad" }, adapt::usr2_done()));
    unsafe { libc::_exit(0) };
}

// ---- the kept output of a lock holder is bounded ----

/// 1 MiB written by a thread that holds jit::TRANSLATE: once it lets go, less than 1 MiB reaches
/// the file, and the last line says how many bytes were dropped. The control: the same 1 MiB
/// written with no lock held arrives whole.
#[test]
fn output_kept_under_a_lock_is_bounded_and_the_drop_is_reported() {
    let dir = std::env::temp_dir();
    let piece = "y".repeat(4095);
    let write_mib = |fd: i32, locked: bool| {
        let g = if locked { Some(crate::jit::TRANSLATE.lock().unwrap()) } else { None };
        for _ in 0..256 {
            crate::forksync::write_out(fd, piece.clone(), true);
        }
        drop(g);
    };
    let run = |locked: bool| -> String {
        let p = dir.join(format!("mcl-j5-defer-{}-{locked}.txt", std::process::id()));
        let f = std::fs::File::create(&p).unwrap();
        use std::os::fd::AsRawFd;
        write_mib(f.as_raw_fd(), locked);
        drop(f);
        let s = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        s
    };
    let free = run(false);
    assert_eq!(free.len(), 1 << 20, "control: 1 MiB written with no lock held did not arrive whole");
    let kept = run(true);
    assert!(kept.len() < 1 << 20, "all {} bytes written under the lock were kept (no bound)", kept.len());
    assert!(kept.len() > 0, "nothing written under the lock arrived");
    let last = kept.lines().last().unwrap_or("");
    assert!(last.contains("bytes of output dropped"), "the drop is not reported: last line {:?}", &last[..last.len().min(80)]);
}

// ---- the static scanner reads nested brackets and renamed imports ----

/// Forms J v4's scanner missed: a struct whose body has a `;` inside brackets before its lock
/// (`[u8; 1]`), a tuple struct likewise, and locks under a renamed import. J v4's own forms are the
/// control (forksafe_v4tests' the_scan_catches_every_declaration_form, which still passes).
#[test]
fn the_scan_reads_nested_brackets_and_renamed_imports() {
    let fake: &[(&str, &str, &str)] = &[(
        "fake",
        "fake.rs",
        "use crate::forksync::FMutex as Lk;\n\
         use std::sync::{Mutex as StdM, Once};\n\
         struct Holder { pad: [u8; 1], lock: FMutex<()> }\n\
         static A: Holder = x;\n\
         static B: Lk<u8> = Lk::new(0);\n\
         static C: StdM<u8> = x;\n\
         struct Tup(u8, [u16; 2], FCondvar);\n\
         static D: Tup = x;\n\
         struct Plain { pad: [u8; 4], n: u32 }\n\
         static E: Plain = x;\n",
    )];
    let sc = super::v4tests::scan(fake);
    let has = |v: &Vec<String>, n: &str| v.iter().any(|s| s == n || s.starts_with(&format!("{n}:")));
    assert!(has(&sc.held_in_struct, "fake::A"), "a lock after [u8; 1] in a struct: {:?}", sc.held_in_struct);
    assert!(has(&sc.held_in_struct, "fake::D"), "a lock in a tuple struct after [u16; 2]: {:?}", sc.held_in_struct);
    assert!(has(&sc.locks, "fake::B"), "FMutex imported as Lk: {:?}", sc.locks);
    assert!(has(&sc.std_sync, "fake::C"), "std's Mutex imported as StdM: {:?}", sc.std_sync);
    assert!(!has(&sc.held_in_struct, "fake::E"), "control: a struct with no lock read as holding one: {:?}", sc.held_in_struct);
}
