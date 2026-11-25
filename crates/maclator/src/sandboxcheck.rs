//! Whether this process is sandboxed, and whether its sandbox lets it read a file, asked so that the sandbox logs no
//! refusal (SANDBOX_CHECK_NO_REPORT).
//!
//! The translator's own stores (the AOT translations and the slid shared cache) keep out of a sandboxed program's way:
//! such a process never writes them, reads them only where its sandbox lets it, and otherwise translates and slides
//! as it goes, with no refusal and no line of its own (one, with tracing). Rosetta's saved translations are written
//! by a service of their own, never by the sandboxed program. A program that sandboxes itself after it starts has
//! done its start-up reads unsandboxed; this is for what comes after, and for a process started sandboxed.

use std::os::unix::ffi::OsStrExt;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

extern "C" {
    fn sandbox_check(pid: libc::pid_t, operation: *const libc::c_char, filter_type: libc::c_int, ...) -> libc::c_int;
}

const SANDBOX_FILTER_PATH: libc::c_int = 1;
const SANDBOX_CHECK_NO_REPORT: libc::c_int = 0x4000_0000;

static SANDBOXED: AtomicBool = AtomicBool::new(false);

/// Whether this process is sandboxed. Asked each time until it is, then kept: a sandbox is never taken away, and a
/// fork child keeps its parent's.
pub fn sandboxed() -> bool {
    if SANDBOXED.load(Relaxed) {
        return true;
    }
    #[cfg(test)]
    tests::ASKED.fetch_add(1, Relaxed);
    // 0: not sandboxed. Anything else is taken as sandboxed, -1 (the check itself failed) included.
    let yes = unsafe { sandbox_check(libc::getpid(), std::ptr::null(), 0) } != 0;
    if yes {
        SANDBOXED.store(true, Relaxed);
    }
    yes
}

/// Whether this process may read the file at `path`: always, unsandboxed; sandboxed, only a file that exists and that
/// its sandbox lets it read (the sandbox answers no for a path that does not exist, whatever its rules). The path is
/// matched as given.
pub fn may_read(path: &std::path::Path) -> bool {
    if !sandboxed() {
        return true;
    }
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return false };
    unsafe { sandbox_check(libc::getpid(), c"file-read-data".as_ptr(), SANDBOX_FILTER_PATH | SANDBOX_CHECK_NO_REPORT, c.as_ptr()) == 0 }
}

/// With tracing on, say once per process that the stores were left alone because the process is sandboxed.
pub fn note_left_alone(what: &str) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if crate::syscalls::TRACE.load(Relaxed) && !SAID.swap(true, Relaxed) {
        eprintln!("[maclator] sandboxed: {what}; translating and sliding as it goes");
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    extern "C" {
        fn sandbox_init_with_parameters(profile: *const libc::c_char, flags: u64, parameters: *const *const libc::c_char, errorbuf: *mut *mut libc::c_char) -> libc::c_int;
    }

    /// How many times `sandboxed` asked the sandbox.
    pub(crate) static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Apply the sandbox `profile` (SBPL) to this process: in a child only, as it cannot be taken away.
    pub(crate) fn apply(profile: &str) -> bool {
        let p = std::ffi::CString::new(profile).unwrap();
        let mut err: *mut libc::c_char = std::ptr::null_mut();
        unsafe { sandbox_init_with_parameters(p.as_ptr(), 0, std::ptr::null(), &mut err) == 0 }
    }

    /// Run `f` in a child forked with every translator lock held, killed after 30 s; the code it exits with.
    pub(crate) fn in_child(f: impl FnOnce() -> i32) -> i32 {
        let pid = crate::forksafe::fork();
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            unsafe { libc::alarm(30) };
            let code = f();
            unsafe { libc::_exit(code) };
        }
        let mut st = 0;
        while unsafe { libc::waitpid(pid, &mut st, 0) } != pid {
            assert!(unsafe { *libc::__error() } == libc::EINTR, "waitpid");
        }
        if libc::WIFEXITED(st) {
            libc::WEXITSTATUS(st)
        } else {
            1000 + libc::WTERMSIG(st)
        }
    }

    #[test]
    fn the_sandbox_is_asked_quietly_and_its_answers_are_kept() {
        // Two files that exist, one in a directory the child's sandbox refuses. Canonical paths, as the sandbox matches
        // the paths a process opens once they are resolved.
        let base = std::env::temp_dir().join(format!("mcl-sbx-{}", std::process::id()));
        let (no, yes) = (base.join("refused"), base.join("allowed"));
        for d in [&no, &yes] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("f"), b"x").unwrap();
        }
        let (no, yes) = (std::fs::canonicalize(&no).unwrap(), std::fs::canonicalize(&yes).unwrap());
        assert!(!sandboxed() && may_read(&no.join("f")) && may_read(&no.join("missing")), "control: this test process is not sandboxed");
        let code = in_child(|| {
            if !apply(&format!("(version 2)(allow default)(deny file-read-data (subpath \"{}\"))", no.display())) {
                return 20;
            }
            match (sandboxed(), may_read(&no.join("f")), may_read(&yes.join("f")), may_read(&yes.join("missing"))) {
                // Kept: not asked again.
                (true, false, true, false) if !SANDBOXED.load(Relaxed) || {
                    let n = ASKED.load(Relaxed);
                    sandboxed();
                    ASKED.load(Relaxed) != n
                } => 7,
                (true, false, true, false) => 0,
                (false, ..) => 3,
                (_, true, ..) => 4,
                (_, _, false, _) => 5,
                (.., true) => 6,
            }
        });
        let _ = std::fs::remove_dir_all(&base);
        assert_eq!(code, 0, "20 the sandbox was not applied, 3 not seen as sandboxed, 4 a refused file said readable, 5 a readable file said refused, 6 a missing file said readable, 7 the answer not kept");
    }
}
