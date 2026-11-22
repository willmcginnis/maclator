//! Tests for the fork-safe translator locks (first set). This file is the
//! same in this tree and in the earlier-version tree used to show each test failing there; what
//! differs between the two trees is forksafe_adapt.rs.
#![allow(clippy::all)]

use super::adapt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

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
        .args([&format!("forksafe::v4tests::{name}"), "--exact", "--ignored", "--nocapture", "--test-threads=1"])
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

// ---- A1: the forksync API refuses misuse at compile time ----

/// Compile `body` as a library holding forksync.rs (as `mod forksync`), with this build's libc.
/// (true when it compiled, rustc's stderr).
fn rustc_lib(tag: &str, body: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("mcl-compile-fail-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fs = concat!(env!("CARGO_MANIFEST_DIR"), "/src/forksync.rs");
    let lib = format!("#![allow(dead_code, unused)]\n#[path = {fs:?}]\nmod forksync;\n{body}\n");
    std::fs::write(dir.join("lib.rs"), lib).unwrap();
    let deps = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let libc_rlib = std::fs::read_dir(&deps)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            n.starts_with("liblibc-") && n.ends_with(".rlib")
        })
        .expect("this build's liblibc rlib beside the test binary");
    let rustc = std::path::Path::new(env!("CARGO")).with_file_name("rustc");
    let out = std::process::Command::new(rustc)
        .args(["--edition", "2021", "--crate-type", "lib", "--crate-name", "compile_fail_probe", "--target", "x86_64-apple-darwin", "--emit=metadata", "-o"])
        .arg(dir.join("probe.rmeta"))
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("--extern")
        .arg(format!("libc={}", libc_rlib.display()))
        .arg(dir.join("lib.rs"))
        .output()
        .expect("run rustc");
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

const GOOD_USE: &str = "static M: forksync::FMutex<u32> = forksync::FMutex::new(0);
static R: forksync::FRwLock<u32> = forksync::FRwLock::new(0);
fn need_sync<T: Sync>(_: &T) {}
pub fn good() {
    std::thread::spawn(|| { *M.lock().unwrap() += 1; }).join().unwrap();
    let g = M.lock().unwrap();
    need_sync(&g);
    drop(g);
    let r = R.read().unwrap();
    need_sync(&r);
}";

/// Each misuse must be refused with its error code; the good use (the positive control) compiles.
#[test]
fn the_forksync_api_refuses_misuse_at_compile_time() {
    let (ok, err) = rustc_lib("good", GOOD_USE);
    assert!(ok, "the positive control did not compile:\n{err}");
    let cases: &[(&str, &str, &str)] = &[
        ("mutex-guard-send", "static M: forksync::FMutex<u32> = forksync::FMutex::new(0);
pub fn bad() { let g = M.lock().unwrap(); std::thread::spawn(move || drop(g)); }", "E0277"),
        ("read-guard-send", "static R: forksync::FRwLock<u32> = forksync::FRwLock::new(0);
pub fn bad() { let g = R.read().unwrap(); std::thread::spawn(move || drop(g)); }", "E0277"),
        ("write-guard-send", "static R: forksync::FRwLock<u32> = forksync::FRwLock::new(0);
pub fn bad() { let g = R.write().unwrap(); std::thread::spawn(move || drop(g)); }", "E0277"),
        ("mutex-guard-sync-over-cell", "static M: forksync::FMutex<std::cell::Cell<u32>> = forksync::FMutex::new(std::cell::Cell::new(0));
fn need_sync<T: Sync>(_: &T) {}
pub fn bad() { let g = M.lock().unwrap(); need_sync(&g); }", "E0277"),
        ("reinit-without-unsafe", "use forksync::ForkHeld;
static M: forksync::FMutex<u32> = forksync::FMutex::new(0);
pub fn bad() { let g = M.lock().unwrap(); let b: Box<dyn ForkHeld> = Box::new(g); b.reinit_in_child(); }", "E0133"),
    ];
    let mut wrong = Vec::new();
    for (tag, body, code) in cases {
        let (ok, err) = rustc_lib(tag, body);
        if ok {
            wrong.push(format!("{tag}: compiled"));
        } else if !err.contains(&format!("error[{code}]")) {
            wrong.push(format!("{tag}: refused, but not with {code}: {}", err.lines().find(|l| l.starts_with("error")).unwrap_or("")));
        }
    }
    assert!(wrong.is_empty(), "misuse the API should refuse: {wrong:?}");
}

// ---- A2: a thread blocked writing log output while it holds a translator lock ----

/// The mechanism, directly: text written while a lock is held reaches the fd only once the lock is
/// released.
#[test]
fn output_written_under_a_translator_lock_comes_out_after_the_release() {
    let mut fds = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) };
    let read_now = || {
        let mut b = [0u8; 64];
        let n = unsafe { libc::read(fds[0], b.as_mut_ptr() as *mut libc::c_void, b.len()) };
        if n > 0 { String::from_utf8_lossy(&b[..n as usize]).into_owned() } else { String::new() }
    };
    let g = crate::jit::TRANSLATE.lock().unwrap();
    crate::forksync::write_out(fds[1], "under-lock".to_string(), true);
    let during = read_now();
    drop(g);
    let after = read_now();
    crate::forksync::write_out(fds[1], "no-lock".to_string(), true);
    let direct = read_now();
    unsafe {
        libc::close(fds[0]);
        libc::close(fds[1]);
    }
    assert_eq!(direct, "no-lock\n", "positive control: text written with no lock held");
    assert_eq!(during, "", "text written under TRANSLATE reached the fd before the release");
    assert_eq!(after, "under-lock\n");
}

/// fd 2 is a full pipe nobody reads; one thread holds jit::TRANSLATE and logs (as MACLATOR_TRACE's
/// invalidation message does, under TRANSLATE and INDEX). A fork must still return.
#[test]
fn a_fork_returns_while_a_lock_holder_logs_to_a_full_stderr_pipe() {
    let out = run_scenario("scenario_log_under_lock", 20).unwrap();
    assert!(out.contains("FORK-RETURNED child ok"), "{out}");
}

#[test]
#[ignore]
fn scenario_log_under_lock() {
    if !scenario("scenario_log_under_lock") {
        return;
    }
    fill_stderr_pipe();
    std::thread::spawn(|| {
        let _t = crate::jit::TRANSLATE.lock().unwrap();
        let _i = crate::jit::INDEX.lock().unwrap();
        eprintln!("[maclator] invalidating 3 units (0x1000+0x40)");
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    let pid = crate::forksafe::fork();
    if pid == 0 {
        unsafe { libc::_exit(0) };
    }
    let st = wait_status(pid);
    say(&format!("FORK-RETURNED child {}\n", if exited_0(st) { "ok" } else { "bad" }));
    unsafe { libc::_exit(0) };
}

// ---- A3: SIGUSR1 at other threads, inside malloc, while one thread forks ----

/// Maclator's own SIGUSR1 handler installed, one guest CPU registered (so a dump has work), three
/// threads allocating in a loop and a fourth sending SIGUSR1 to them in turn, while this thread
/// forks 300 times. Every fork must return, every child exit 0, and the dump must have run.
#[test]
fn forks_survive_a_sigusr1_storm_at_threads_inside_malloc() {
    let out = run_scenario("scenario_usr1_storm_elsewhere", 60).unwrap();
    assert!(out.contains("FORKS-DONE bad=0"), "{out}");
    let dumps: usize = out.split("dumps=").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse().ok()).unwrap_or(0);
    assert!(dumps > 0, "positive control: no dump ran, so no signal was handled: {out}");
}

#[test]
#[ignore]
fn scenario_usr1_storm_elsewhere() {
    if !scenario("scenario_usr1_storm_elsewhere") {
        return;
    }
    unsafe {
        let null = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
        libc::dup2(null, 2);
    }
    crate::engine::install_debug_handler();
    let cpu: &'static mut maclator_core::cpu::Cpu = Box::leak(maclator_core::cpu::Cpu::new());
    crate::engine::CPUS.lock().unwrap().push(cpu as *mut _ as usize);
    let stop = Arc::new(AtomicBool::new(false));
    let ids = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut hs = Vec::new();
    for t in 0..3 {
        let (s, ids) = (stop.clone(), ids.clone());
        hs.push(std::thread::spawn(move || {
            ids.lock().unwrap().push(unsafe { libc::pthread_self() } as usize);
            let mut keep: Vec<Vec<u8>> = Vec::new();
            let mut i = t;
            while !s.load(Ordering::Relaxed) {
                keep.push(vec![i as u8; 16 + (i * 37) % 4000]);
                if keep.len() > 64 {
                    keep.clear();
                }
                i += 1;
            }
        }));
    }
    while ids.lock().unwrap().len() < 3 {
        std::thread::yield_now();
    }
    let targets: Vec<usize> = ids.lock().unwrap().clone();
    let s = stop.clone();
    let sender = std::thread::spawn(move || {
        let mut k = 0;
        while !s.load(Ordering::Relaxed) {
            unsafe { libc::pthread_kill(targets[k % 3] as libc::pthread_t, libc::SIGUSR1) };
            k += 1;
            unsafe { libc::usleep(50) };
        }
    });
    let mut bad = 0;
    for _ in 0..300 {
        let pid = crate::forksafe::fork();
        if pid == 0 {
            unsafe { libc::_exit(0) };
        }
        if !exited_0(wait_status(pid)) {
            bad += 1;
        }
    }
    stop.store(true, Ordering::Relaxed);
    sender.join().unwrap();
    for h in hs {
        h.join().unwrap();
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    say(&format!("FORKS-DONE bad={bad} dumps={}\n", adapt::dumps_done()));
    unsafe { libc::_exit(0) };
}

// ---- A4: an initialiser in flight at the fork; a GPU bridge and workqueue bindings in the child ----

static LAZY_UNDER_TEST: adapt::Lazy<u32> = adapt::Lazy::new();
static LAZY_RAW: adapt::Lazy<u32> = adapt::Lazy::new();

/// Start initialising `l` on another thread (the initialiser sleeps 300 ms); returns once it runs.
fn init_in_flight(l: &'static adapt::Lazy<u32>) -> std::thread::JoinHandle<()> {
    let started = Arc::new(AtomicBool::new(false));
    let s = started.clone();
    let h = std::thread::spawn(move || {
        l.get_or_init(|| {
            s.store(true, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(300));
            7
        });
    });
    while !started.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    h
}

/// A fork while another thread is inside a lazy static's initialiser: the child must get the value.
#[test]
fn a_child_forked_during_a_lazy_init_gets_the_value() {
    let h = init_in_flight(&LAZY_UNDER_TEST);
    let st = in_child(|| if *LAZY_UNDER_TEST.get_or_init(|| 9) == 7 { 0 } else { 1 });
    h.join().unwrap();
    assert!(exited_0(st), "child status {st:#x}");
}

/// The hazard the gate removes, shown with a raw fork: the child of a fork taken while the
/// initialiser runs waits for an initialiser it does not have.
#[test]
fn must_fail_control_a_raw_fork_during_a_lazy_init_strands_the_child() {
    let h = init_in_flight(&LAZY_RAW);
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);
    if pid == 0 {
        unsafe { libc::alarm(3) };
        let v = *LAZY_RAW.get_or_init(|| 9);
        unsafe { libc::_exit(if v == 7 { 0 } else { 1 }) };
    }
    let st = wait_status(pid);
    h.join().unwrap();
    assert!(!exited_0(st), "the raw fork's child got the value: the control saw nothing");
}

unsafe extern "C" fn fake_bridge(op: u64, _input: *const u8, _len: u64, outlen: *mut u64) -> *mut u8 {
    *outlen = op + 1;
    (op * 2) as *mut u8
}

/// A thread makes a GPU bridged call (so its bridge thread exists and has gone to sleep), forks,
/// and in the child makes another: it must be answered. The parent's bridge keeps working.
#[test]
fn a_child_makes_a_gpu_bridged_call_after_its_thread_used_the_bridge_in_the_parent() {
    let st = std::thread::spawn(|| {
        assert_eq!(crate::gpu::test_call(fake_bridge, 5), (10, 6));
        std::thread::sleep(std::time::Duration::from_millis(50));
        let st = in_child(|| if crate::gpu::test_call(fake_bridge, 7) == (14, 8) { 0 } else { 1 });
        assert_eq!(crate::gpu::test_call(fake_bridge, 9), (18, 10), "the parent's bridge after the fork");
        st
    })
    .join()
    .unwrap();
    assert!(exited_0(st), "child status {st:#x}");
}

/// The forking thread's workqueue bindings (manager, bound workloop) name state the child drops,
/// and a worker made before the fork is stale in the child; in the parent nothing changes.
#[test]
fn a_child_forgets_the_forking_threads_workqueue_bindings() {
    let st = std::thread::spawn(|| {
        crate::workq::test_set_manager_and_wl();
        let gen0 = adapt::generation();
        let st = in_child(move || {
            let ok = !crate::workq::test_is_manager() && crate::workq::test_bound_wl().is_none() && adapt::worker_is_stale(gen0);
            if ok { 0 } else { 1 }
        });
        assert!(crate::workq::test_is_manager() && crate::workq::test_bound_wl() == Some(7), "the parent's bindings changed");
        assert!(!adapt::worker_is_stale(gen0), "a parent worker read stale in the parent");
        st
    })
    .join()
    .unwrap();
    assert!(exited_0(st), "child status {st:#x}");
}

// ---- A5: the thread-creation handshake when the thread ends before it is ready ----

/// spawn_ready returns (an error), not a wait for good, when the new thread ends without calling
/// ready; and Ok when it does (the positive control).
#[test]
fn spawn_ready_returns_when_the_thread_ends_before_ready() {
    let st = in_child(|| {
        if adapt::spawn_ready(|ready| ready()).is_err() {
            return 2;
        }
        if adapt::spawn_ready(|_ready| {}).is_err() { 0 } else { 3 }
    });
    assert!(exited_0(st), "child status {st:#x} (exit 2: the control failed; 3: reported ready; SIGALRM: it waited for good)");
}

// ---- A10: the static scanner ----

/// The .rs files under a directory, relative to it, sorted.
pub(crate) fn rs_files(root: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {}: {e}", d.display())) {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p.strip_prefix(root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    out.sort();
    out
}

/// The source with comments, string literals and char literals blanked (lifetimes kept).
pub(crate) fn code_only(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut o = String::with_capacity(src.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '/' && c.get(i + 1) == Some(&'/') {
            while i < c.len() && c[i] != '\n' {
                i += 1;
            }
        } else if c[i] == '/' && c.get(i + 1) == Some(&'*') {
            let mut depth = 0;
            while i < c.len() {
                if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            o.push(' ');
        } else if c[i] == 'r' && (c.get(i + 1) == Some(&'"') || (c.get(i + 1) == Some(&'#') && (i == 0 || !(c[i - 1].is_alphanumeric() || c[i - 1] == '_')))) {
            let mut j = i + 1;
            let mut hashes = 0;
            while c.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if c.get(j) != Some(&'"') {
                o.push(c[i]);
                i += 1;
                continue;
            }
            j += 1;
            loop {
                if j >= c.len() {
                    break;
                }
                if c[j] == '"' && (0..hashes).all(|k| c.get(j + 1 + k) == Some(&'#')) {
                    j += 1 + hashes;
                    break;
                }
                j += 1;
            }
            o.push_str("\"\"");
            i = j;
        } else if c[i] == '"' {
            i += 1;
            while i < c.len() && c[i] != '"' {
                if c[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            o.push_str("\"\"");
        } else if c[i] == '\'' {
            if c.get(i + 1) == Some(&'\\') {
                i += 3;
                while i < c.len() && c[i] != '\'' {
                    i += 1;
                }
                i += 1;
                o.push_str("' '");
            } else if c.get(i + 2) == Some(&'\'') {
                i += 3;
                o.push_str("' '");
            } else {
                o.push('\'');
                i += 1;
            }
        } else {
            o.push(c[i]);
            i += 1;
        }
    }
    o
}

fn idents(s: &str) -> Vec<&str> {
    s.split(|ch: char| !(ch.is_alphanumeric() || ch == '_')).filter(|w| !w.is_empty()).collect()
}

const STD_SYNC: &[&str] = &[
    "Mutex", "RwLock", "Condvar", "Once", "OnceLock", "LazyLock", "OnceCell", "LazyCell", "Barrier", "Sender",
    "SyncSender", "Receiver", "ReentrantLock", "mpsc", "mpmc",
];
const OURS: &[&str] = &["FMutex", "FRwLock", "FCondvar", "FOnceLock"];

pub(crate) struct Scan {
    /// module::NAME of each static FMutex or FRwLock (the outer type).
    pub(crate) locks: Vec<String>,
    pub(crate) condvars: Vec<String>,
    pub(crate) oncelocks: Vec<String>,
    /// Statics whose type mentions a std synchronisation type anywhere.
    pub(crate) std_sync: Vec<String>,
    /// Statics whose type mentions a struct or alias of the crate that holds a lock type.
    pub(crate) held_in_struct: Vec<String>,
    /// Statics of a forksync type nested inside another type (not a bare static of it).
    pub(crate) nested: Vec<String>,
    pub(crate) statics: usize,
}

/// The end of an item that starts at `rest[0]` (`struct X ...`,
/// `type X = ...;`): the first `;` outside any bracket, or the `}` that closes its first `{`.
/// Brackets are (), [] and {}; so `[u8; 1]` or `(u8, u8)` inside a body does not end it (J v4 stopped
/// at the first `;` or `}`). Angle brackets are not counted: a `;` cannot stand inside them alone.
pub(crate) fn item_end(rest: &str) -> usize {
    let mut depth: i32 = 0;
    for (i, ch) in rest.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' => depth -= 1,
            '}' => {
                depth -= 1;
                if depth <= 0 {
                    return i + 1;
                }
            }
            ';' if depth <= 0 => return i + 1,
            _ => {}
        }
    }
    rest.len()
}

/// `use ... X as Y` where X is a synchronisation type (ours or std's),
/// in any source: (Y, X). Grouped imports (`{Mutex as M, ..}`) included; a re-export under a new
/// name through `pub use` is the same text.
pub(crate) fn renamed_imports(codes: &[(&str, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (_, code) in codes {
        let w = idents(code);
        for k in 1..w.len().saturating_sub(1) {
            if w[k] == "as" && (OURS.contains(&w[k - 1]) || STD_SYNC.contains(&w[k - 1])) && !out.iter().any(|(a, _)| a == w[k + 1]) {
                out.push((w[k + 1].to_string(), w[k - 1].to_string()));
            }
        }
    }
    out
}

/// Every `static` in every source, multi-line declarations and paths included.
pub(crate) fn scan(sources: &[(&str, &str, &str)]) -> Scan {
    let raw: Vec<(&str, String)> = sources.iter().map(|(m, _, s)| (*m, code_only(s))).collect();
    // A name a synchronisation type was imported under is read as that type, everywhere.
    let aliases = renamed_imports(&raw);
    let codes: Vec<(&str, String)> = raw
        .iter()
        .map(|(m, c)| {
            let mut o = String::with_capacity(c.len());
            let mut word = String::new();
            for ch in c.chars().chain(std::iter::once(' ')) {
                if ch.is_alphanumeric() || ch == '_' {
                    word.push(ch);
                    continue;
                }
                match aliases.iter().find(|(a, _)| *a == word) {
                    Some((_, orig)) => o.push_str(orig),
                    None => o.push_str(&word),
                }
                word.clear();
                o.push(ch);
            }
            o.pop();
            (*m, o)
        })
        .collect();
    // Structs and type aliases that hold a lock type, by name (one level, then to a fixpoint).
    let mut holders: Vec<String> = Vec::new();
    let mut defs: Vec<(String, String)> = Vec::new();
    for (_, code) in &codes {
        let b = code.as_bytes();
        for kw in ["struct", "type", "enum", "union"] {
            let mut from = 0;
            while let Some(off) = code[from..].find(kw) {
                let at = from + off;
                from = at + kw.len();
                let before_ok = at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_');
                let after_ok = b.get(at + kw.len()).is_some_and(|c| c.is_ascii_whitespace());
                if !before_ok || !after_ok {
                    continue;
                }
                let rest = &code[at..];
                let Some(name) = idents(&rest[kw.len()..]).first().map(|n| n.to_string()) else { continue };
                defs.push((name, rest[..item_end(rest)].to_string()));
            }
        }
    }
    loop {
        let before = holders.len();
        for (name, body) in &defs {
            if holders.contains(name) || OURS.contains(&name.as_str()) {
                continue;
            }
            let ids = idents(body);
            if ids.iter().skip(2).any(|i| OURS.contains(i) || STD_SYNC.contains(i) || holders.iter().any(|h| h == i)) {
                holders.push(name.clone());
            }
        }
        if holders.len() == before {
            break;
        }
    }
    let mut sc = Scan { locks: vec![], condvars: vec![], oncelocks: vec![], std_sync: vec![], held_in_struct: vec![], nested: vec![], statics: 0 };
    for (module, code) in &codes {
        let b = code.as_bytes();
        let mut from = 0;
        while let Some(off) = code[from..].find("static") {
            let at = from + off;
            from = at + 6;
            let before_ok = at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_' || b[at - 1] == b'\'');
            let after_ok = b.get(at + 6).is_some_and(|c| c.is_ascii_whitespace());
            if !before_ok || !after_ok {
                continue;
            }
            let rest = &code[at + 6..];
            let Some(end) = rest.find(|ch| ch == '=' || ch == ';') else { continue };
            let decl: String = rest[..end].split_whitespace().collect::<Vec<_>>().join(" ");
            let decl = decl.strip_prefix("mut ").unwrap_or(&decl).to_string();
            let Some((name, ty)) = decl.split_once(':') else { continue };
            let (name, ty) = (name.trim(), ty.trim());
            sc.statics += 1;
            let full = format!("{module}::{name}");
            let ids = idents(ty);
            let outer = ty.split('<').next().unwrap_or(ty).rsplit("::").next().unwrap_or(ty).trim();
            if ids.iter().any(|i| STD_SYNC.contains(i)) {
                sc.std_sync.push(format!("{full}: {ty}"));
            }
            match outer {
                "FMutex" | "FRwLock" => sc.locks.push(full.clone()),
                "FCondvar" => sc.condvars.push(full.clone()),
                "FOnceLock" => sc.oncelocks.push(full.clone()),
                _ => {
                    if ids.iter().any(|i| OURS.contains(i)) {
                        sc.nested.push(format!("{full}: {ty}"));
                    }
                }
            }
            let inner: Vec<&str> = if OURS.contains(&outer) { ids.iter().skip_while(|i| **i != outer).skip(1).copied().collect() } else { ids.clone() };
            if inner.iter().any(|i| holders.iter().any(|h| h == i)) {
                sc.held_in_struct.push(format!("{full}: {ty}"));
            }
            if OURS.contains(&outer) && inner.iter().any(|i| OURS.contains(i)) {
                sc.nested.push(format!("{full}: {ty}"));
            }
        }
    }
    sc
}

/// The scan's own controls: each declaration form it must catch, caught.
#[test]
fn the_scan_catches_every_declaration_form() {
    let fake: &[(&str, &str, &str)] = &[(
        "fake",
        "fake.rs",
        "static A: FMutex<u8> = FMutex::new(0);\n\
         pub(crate) static\n    B:\n    crate::forksync::FRwLock<Vec<u8>> = x;\n\
         static C: std::sync::Mutex<u8> = x;\n\
         static D: std::sync::OnceLock<u8> = x;\n\
         struct Holder { m: FMutex<u8> }\n\
         static E: Holder = x;\n\
         type Alias = std::sync::RwLock<u8>;\n\
         static F: Alias = x;\n\
         static G: FCondvar = FCondvar::new();\n\
         static H: FMutex<Option<FMutex<u8>>> = x;\n\
         // static I: std::sync::Mutex<u8> = x;\n\
         const S: &str = \"static J: std::sync::Mutex<u8> = x;\";\n\
         fn f<'a>(x: &'a u8) { let c = 'x'; }\n\
         static K: FOnceLock<u8> = FOnceLock::new();\n\
         fn g() -> &'static str { x }\n",
    )];
    let sc = scan(fake);
    assert_eq!(sc.locks, ["fake::A", "fake::B", "fake::H"]);
    assert_eq!(sc.condvars, ["fake::G"]);
    assert_eq!(sc.oncelocks, ["fake::K"]);
    assert_eq!(sc.std_sync.len(), 2, "{:?}", sc.std_sync); // C and D; F names RwLock through Alias only
    assert!(sc.std_sync.iter().any(|s| s.starts_with("fake::C")) && sc.std_sync.iter().any(|s| s.starts_with("fake::D")));
    assert!(sc.held_in_struct.iter().any(|s| s.starts_with("fake::E")), "{:?}", sc.held_in_struct);
    assert!(sc.held_in_struct.iter().any(|s| s.starts_with("fake::F")), "{:?}", sc.held_in_struct);
    assert!(sc.nested.iter().any(|s| s.starts_with("fake::H")), "{:?}", sc.nested);
    assert!(!sc.std_sync.iter().any(|s| s.contains("fake::I") || s.contains("fake::J")), "a comment or string was read as code");
}

