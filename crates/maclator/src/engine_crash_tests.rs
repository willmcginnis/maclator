//! A guest fault seen from outside: the test binary is run again for one case, the guest CPU in it executes the
//! faulting exit, and this process reads how the child ended and what it left in its store. Only what the engine had
//! before (handle_exit, the store from MACLATOR_CACHE) is used, so the same file runs against the engine as it was,
//! where the child exited with status 134 and wrote no file.

use super::*;
use std::os::unix::process::ExitStatusExt;

const NAME: &str = "engine::crash_tests::a_guest_fault_ends_the_process_by_its_signal";

/// In the child: fault as MCL_CRASH_CASE says.
fn fault_in_child(case: &str) {
    if case.ends_with("-stderr-full") {
        // Fill stderr, a pipe the parent does not read, then make it blocking again: a write to it now waits forever.
        unsafe {
            let fl = libc::fcntl(2, libc::F_GETFL);
            libc::fcntl(2, libc::F_SETFL, fl | libc::O_NONBLOCK);
            let junk = [b'x'; 4096];
            while libc::write(2, junk.as_ptr() as *const libc::c_void, junk.len()) > 0 {}
            libc::fcntl(2, libc::F_SETFL, fl & !libc::O_NONBLOCK);
        }
    }
    if case.ends_with("-fsize") {
        // No file may grow: the report cannot be written, and SIGXFSZ, at its default action, would end the process.
        let mut l: libc::rlimit = unsafe { std::mem::zeroed() };
        unsafe { libc::getrlimit(libc::RLIMIT_FSIZE, &mut l) };
        l.rlim_cur = 0;
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &l) }, 0, "the file size limit was set");
    }
    if case.ends_with("-sandboxed") {
        assert!(crate::sandboxcheck::tests::apply("(version 2)(allow default)"), "the sandbox was applied");
    }
    let mut cpu = Cpu::new();
    let e = if case.starts_with("brk") { Exit::Brk(1) } else { Exit::Udf(0) };
    handle_exit(&mut cpu, e);
    // handle_exit came back: the fault did not end the process.
    std::process::exit(3);
}

struct Store(std::path::PathBuf);

impl Drop for Store {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(case: &str) -> (std::process::Output, Store) {
    let d = std::env::temp_dir().join(format!("mcl-crash-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let store = d.join("cache");
    std::fs::create_dir_all(&store).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&store, std::fs::Permissions::from_mode(0o700)).unwrap();
    if case.ends_with("-sandboxed") {
        // The crash directory made already, as a sandboxed process would not make it: what keeps it from writing a
        // report is then its own choice, not a directory it could not create.
        std::fs::create_dir(store.join("crash")).unwrap();
        std::fs::set_permissions(store.join("crash"), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", NAME, "--test-threads=1", "--nocapture"])
        .env("MCL_CRASH_CASE", case)
        .env("MACLATOR_CACHE", &store)
        .output()
        .unwrap();
    (out, Store(d))
}

fn reports(store: &Store) -> Vec<(String, String, u32)> {
    let dir = store.0.join("cache").join("crash");
    let Ok(rd) = std::fs::read_dir(&dir) else { return vec![] };
    let mut v: Vec<_> = rd
        .flatten()
        .map(|e| {
            use std::os::unix::fs::PermissionsExt;
            let mode = e.metadata().map(|m| m.permissions().mode() & 0o777).unwrap_or(0);
            (e.file_name().to_string_lossy().into_owned(), std::fs::read_to_string(e.path()).unwrap_or_default(), mode)
        })
        .collect();
    v.sort();
    v
}

#[test]
fn a_guest_fault_ends_the_process_by_its_signal() {
    if let Ok(case) = std::env::var("MCL_CRASH_CASE") {
        fault_in_child(&case);
        return;
    }
    for (case, sig, what) in [("brk", libc::SIGTRAP, "guest executed BRK #0x1 (trap)"), ("udf", libc::SIGILL, "guest executed undefined instruction 0x00000000")] {
        let (out, store) = run(case);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.signal(), Some(sig), "{case}: the child ended {}, not by signal {sig}: {stderr}", out.status);
        assert!(stderr.contains(&format!("maclator: {what}")), "{case}: stderr says what happened as before: {stderr}");
        let r = reports(&store);
        assert_eq!(r.len(), 1, "{case}: one report file: {r:?}");
        let (name, text, mode) = &r[0];
        assert!(name.starts_with("crash-") && name.ends_with(".log"), "{case}: {name}");
        assert_eq!(*mode, 0o600, "{case}: the report is the user's alone");
        assert!(text.starts_with("maclator crash report\n"), "{case}: {text}");
        assert!(text.contains(&format!("\nreason: {what}\n")) && text.contains(&format!("\nsignal: {} ({sig})\n", if sig == libc::SIGTRAP { "SIGTRAP" } else { "SIGILL" })), "{case}: {text}");
        assert!(text.contains(&format!("maclator: {what}")), "{case}: and the text stderr got: {text}");
    }
    // A process that may not grow a file writes no report and still ends by its signal, not by SIGXFSZ.
    let (out, store) = run("brk-fsize");
    assert_eq!(out.status.signal(), Some(libc::SIGTRAP), "no file may grow: {}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    assert_eq!(reports(&store), vec![], "no report, nor an empty file left behind");
    // A sandboxed process leaves the store alone, and still ends by its signal.
    let (out, store) = run("brk-sandboxed");
    assert_eq!(out.status.signal(), Some(libc::SIGTRAP), "sandboxed: {}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    assert_eq!(reports(&store), vec![], "sandboxed: no report file");
}

const FULL: &str = "engine::crash_tests::a_guest_fault_ends_by_its_signal_though_stderr_is_full";

/// stderr a full pipe its reader does not read (a parent that reads only after waitpid): the report still goes to the
/// file, and the process still ends by its signal, not blocked in the write.
#[test]
fn a_guest_fault_ends_by_its_signal_though_stderr_is_full() {
    if let Ok(case) = std::env::var("MCL_CRASH_CASE") {
        fault_in_child(&case);
        return;
    }
    let d = std::env::temp_dir().join(format!("mcl-crash-full-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let store = d.join("cache");
    std::fs::create_dir_all(&store).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&store, std::fs::Permissions::from_mode(0o700)).unwrap();
    let guard = Store(d);
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", FULL, "--test-threads=1", "--nocapture"])
        .env("MCL_CRASH_CASE", "brk-stderr-full")
        .env("MACLATOR_CACHE", &store)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let end = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let status = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break Some(st);
        }
        if std::time::Instant::now() >= end {
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    assert_eq!(status.and_then(|s| s.signal()), Some(libc::SIGTRAP), "the child, its stderr full: {status:?} (None: still blocked after 30 s)");
    assert_eq!(reports(&guard).len(), 1, "and its report is in the file");
}
