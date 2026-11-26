//! How often an image is built, seen from outside: each "exit" is this test binary run again for one test, which
//! registers an image of real arm64 code at a fixed address, notes the blocks it ran and saves its profile, as a
//! translated program does at its exit. Every build appends a line to the file named by MCL_TEST_BUILD_LOG
//! (`build_hook`), so a test counts builds and sees when each one ran, in whichever process built it.
//!
//! These tests use only what the store had before builds were counted (register_main, ensure_loaded,
//! note_translated, save_profile, the environment), so the same file runs against the engine as it was, where they
//! fail: there, every exit that saw a new block or could not load a translation built the image again, at once, in
//! a builder of its own.

use super::*;
use std::time::{Duration, Instant};

/// Called by build(): a line per build start and end in MCL_TEST_BUILD_LOG, and, at the start, a sleep of
/// MCL_TEST_BUILD_DELAY_MS so a test can watch a builder that is building, or kill it there.
pub(super) fn build_hook(img: &Image, what: &str) {
    let Some(log) = std::env::var_os("MCL_TEST_BUILD_LOG") else { return };
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let bg = unsafe { libc::getpriority(libc::PRIO_DARWIN_PROCESS, 0) };
    let limit = |r| {
        let mut l: libc::rlimit = unsafe { std::mem::zeroed() };
        unsafe { libc::getrlimit(r, &mut l) };
        l.rlim_cur
    };
    let line = format!(
        "{what} pid={} key={} t={ns} bg={bg} cpu={} core={}\n",
        std::process::id(),
        img.key,
        limit(libc::RLIMIT_CPU),
        limit(libc::RLIMIT_CORE)
    );
    {
        use std::os::unix::fs::OpenOptionsExt;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).mode(0o600).open(&log) {
            let _ = f.write_all(line.as_bytes());
        }
    }
    if what == "start" {
        if let Some(ms) = std::env::var("MCL_TEST_BUILD_DELAY_MS").ok().and_then(|v| v.parse::<u64>().ok()) {
            std::thread::sleep(Duration::from_millis(ms));
        }
    }
}

/// Called in the builder child right after the fork: a "forked" line, so a test can tell an exit that forked nothing.
pub(super) fn fork_hook() {
    let Some(log) = std::env::var_os("MCL_TEST_BUILD_LOG") else { return };
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    use std::os::unix::fs::OpenOptionsExt;
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).mode(0o600).open(&log) {
        let _ = f.write_all(format!("forked pid={} key=- t={ns}\n", std::process::id()).as_bytes());
    }
}

/// Where the image's code lives in every exit: one address, so every run registers the same image (the key names
/// the base).
const CODE_AT: u64 = 0x7e00_0000_0000;
const CODE_INSNS: usize = 64;

/// Map CODE_INSNS arm64 `ret`s at CODE_AT in this process (a hint, checked: never MAP_FIXED over a mapping we do not
/// know). Each `ret` is a block of its own.
fn map_code(at: u64) -> (u64, u64) {
    let len = 0x4000usize;
    let p = unsafe { libc::mmap(at as *mut _, len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
    assert_eq!(p as u64, at, "the code could not be mapped at its address");
    let words = unsafe { std::slice::from_raw_parts_mut(p as *mut u32, CODE_INSNS) };
    words.fill(0xd65f_03c0);
    (at, at + 4 * CODE_INSNS as u64)
}

/// A second address for the same image: a run whose preferred base is taken slides there.
const CODE_AT_SLID: u64 = 0x7e00_0010_0000;
/// Where a library (a Mach-O header, then code) is mapped and registered as dyld's mmap of its __TEXT would.
const LIB_AT: u64 = 0x7e00_0020_0000;

/// A second address for a library: the same library loaded again after it was unloaded.
const LIB_AT_AGAIN: u64 = 0x7e00_0030_0000;

/// Map a library with UUID [tag; 16] at `at`: a header of one __TEXT segment (vmaddr 0, 16 KiB) and an LC_UUID, and
/// CODE_INSNS `ret`s at +0x1000.
fn map_library(at: u64, tag: u8) -> (u64, u64) {
    let p = unsafe { libc::mmap(at as *mut _, 0x4000, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
    assert_eq!(p as u64, at, "the library could not be mapped at its address");
    let mut h: Vec<u8> = Vec::new();
    let w32 = |h: &mut Vec<u8>, v: u32| h.extend_from_slice(&v.to_le_bytes());
    let w64 = |h: &mut Vec<u8>, v: u64| h.extend_from_slice(&v.to_le_bytes());
    for v in [0xfeed_facf, 0x0100_000c, 0, 6, 2, 72 + 24, 0, 0] {
        w32(&mut h, v);
    }
    w32(&mut h, 0x19);
    w32(&mut h, 72);
    let mut name = [0u8; 16];
    name[..6].copy_from_slice(b"__TEXT");
    h.extend_from_slice(&name);
    for v in [0u64, 0x4000, 0, 0x4000] {
        w64(&mut h, v);
    }
    for v in [5u32, 5, 0, 0] {
        w32(&mut h, v);
    }
    w32(&mut h, 0x1b);
    w32(&mut h, 24);
    h.extend_from_slice(&[tag; 16]);
    let mem = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, 0x4000) };
    mem[..h.len()].copy_from_slice(&h);
    let code = unsafe { std::slice::from_raw_parts_mut((at + 0x1000) as *mut u32, CODE_INSNS) };
    code.fill(0xd65f_03c0);
    (at, at + 0x4000)
}

/// The UUID offset of map_library's header: 32 + 72 + 8.
const LIB_UUID_AT: u64 = 112;

/// The image of one test: its UUID's bytes are `tag`.
fn uuid(tag: u8) -> [u8; 16] {
    [tag; 16]
}

fn path_of(tag: u8) -> String {
    format!("/x/mcl-once-{tag}")
}

/// The store key of image `tag`, as register_main makes it.
fn key_of(tag: u8) -> String {
    image_key("main", &uuid(tag), &path_of(tag), CODE_AT)
}

/// One exit, run in the child: register image MCL_ONCE_TAG (at its slid address with MCL_ONCE_SLID), load what the store
/// has, run block MCL_ONCE_BLOCK, save. With MCL_ONCE_HOLD, it first takes a flock on that file and keeps it open, as a
/// daemon holds its single-instance lock.
fn exit_in_child() {
    if let Ok(tag) = std::env::var("MCL_ONCE_LIB") {
        // A library: registered as dyld's mapping registers it, run, and (MCL_ONCE_REPLACE) another library mapped over
        // its addresses before the exit: its header now names that one.
        let tag: u8 = tag.parse().unwrap();
        let (start, _) = map_library(LIB_AT, tag);
        register_image_at(start);
        note_translated(start + 0x1000);
        if std::env::var_os("MCL_ONCE_REPLACE").is_some() {
            let uuid = unsafe { std::slice::from_raw_parts_mut((LIB_AT + LIB_UUID_AT) as *mut u8, 16) };
            uuid.fill(tag.wrapping_add(1));
        }
        // MCL_ONCE_LIB_AGAIN: the same library loaded again, elsewhere, and run there.
        if std::env::var_os("MCL_ONCE_LIB_AGAIN").is_some() {
            let (again, _) = map_library(LIB_AT_AGAIN, tag);
            register_image_at(again);
            note_translated(again + 0x1000);
        }
        save_profile();
        return;
    }
    if let Some(hold) = std::env::var_os("MCL_ONCE_HOLD") {
        use std::os::unix::io::IntoRawFd;
        let fd = std::fs::OpenOptions::new().read(true).write(true).create(true).open(hold).unwrap().into_raw_fd();
        assert_eq!(unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) }, 0, "the child takes its lock");
    }
    let tag: u8 = std::env::var("MCL_ONCE_TAG").unwrap().parse().unwrap();
    let block: u64 = std::env::var("MCL_ONCE_BLOCK").unwrap().parse().unwrap();
    let at = if std::env::var_os("MCL_ONCE_SLID").is_some() { CODE_AT_SLID } else { CODE_AT };
    // MCL_ONCE_UNMAPPED: the image registered but its code not mapped, so nothing of it can be read at the exit.
    let (start, end) = if std::env::var_os("MCL_ONCE_UNMAPPED").is_some() { (at, at + 4 * CODE_INSNS as u64) } else { map_code(at) };
    register_main(&path_of(tag), uuid(tag), start, end, vec![start]);
    // MCL_ONCE_TAG2: a second image in the same exit, at the slid address, built after the first.
    if let Ok(tag2) = std::env::var("MCL_ONCE_TAG2") {
        let tag2: u8 = tag2.parse().unwrap();
        let (s2, e2) = map_code(CODE_AT_SLID);
        register_main(&path_of(tag2), uuid(tag2), s2, e2, vec![s2]);
        note_translated(s2);
    }
    // MCL_ONCE_NOLOAD: an exit before the store was read (an image registered after the last load, say).
    if std::env::var_os("MCL_ONCE_NOLOAD").is_none() {
        ensure_loaded();
    }
    note_translated(start + 4 * (block % CODE_INSNS as u64));
    save_profile();
    // MCL_ONCE_EXEC: not an exit but an exec, as execve saves the profile first: this binary again, for the test it
    // names, in its after-exec role (the kept re-launch execs the translator itself, as this does).
    if let Ok(test) = std::env::var("MCL_ONCE_EXEC") {
        exec_self(&test);
    }
}

/// Exec this test binary for `test` in its after-exec role, with a descriptor of a file in the store left open across
/// the exec on purpose (no close-on-exec): the control that the new program's listing sees what an exec hands it.
fn exec_self(test: &str) -> ! {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::IntoRawFd;
    let store = std::path::PathBuf::from(std::env::var_os("MACLATOR_CACHE").unwrap());
    let canary = std::fs::OpenOptions::new().read(true).write(true).create(true).open(store.join("exec-canary")).unwrap().into_raw_fd();
    assert_eq!(unsafe { libc::fcntl(canary, libc::F_SETFD, 0) }, 0, "the canary is kept open across the exec");
    let exe = std::ffi::CString::new(std::env::current_exe().unwrap().as_os_str().as_bytes()).unwrap();
    let argv: Vec<std::ffi::CString> =
        [exe.to_bytes(), b"--exact", test.as_bytes(), b"--test-threads=1", b"--nocapture"].iter().map(|a| std::ffi::CString::new(*a).unwrap()).collect();
    let envp: Vec<std::ffi::CString> = std::env::vars_os()
        .filter(|(k, _)| k != "MCL_ONCE_ROLE")
        .map(|(k, v)| [k.as_bytes(), b"=", v.as_bytes()].concat())
        .chain([b"MCL_ONCE_ROLE=after-exec".to_vec()])
        .map(|e| std::ffi::CString::new(e).unwrap())
        .collect();
    let mut av: Vec<*const libc::c_char> = argv.iter().map(|a| a.as_ptr()).collect();
    av.push(std::ptr::null());
    let mut ev: Vec<*const libc::c_char> = envp.iter().map(|e| e.as_ptr()).collect();
    ev.push(std::ptr::null());
    unsafe { libc::execve(exe.as_ptr(), av.as_ptr(), ev.as_ptr()) };
    panic!("the exec failed: {}", std::io::Error::last_os_error());
}

/// The files of the store this process has open: the name of each open descriptor's file (by device and inode) among
/// the store's files.
fn store_fds(store: &std::path::Path) -> Vec<String> {
    use std::os::unix::fs::MetadataExt;
    let files: Vec<((u64, u64), String)> = std::fs::read_dir(store)
        .unwrap()
        .flatten()
        .filter_map(|e| Some(((e.metadata().ok()?.dev() as u64, e.metadata().ok()?.ino()), e.file_name().to_string_lossy().into_owned())))
        .collect();
    let fds: Vec<i32> = std::fs::read_dir("/dev/fd").unwrap().flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).collect();
    let mut names = Vec::new();
    for fd in fds {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            continue;
        }
        if let Some((_, name)) = files.iter().find(|(id, _)| *id == (st.st_dev as u64, st.st_ino)) {
            names.push(name.clone());
        }
    }
    names.sort();
    names
}

/// The pid of the process holding a lock on `path` (another one: a process's own locks do not show), "none", or
/// "missing" when there is no such file.
fn lock_holder(path: &std::path::Path) -> String {
    use std::os::unix::io::AsRawFd;
    let Ok(f) = std::fs::OpenOptions::new().read(true).write(true).open(path) else { return "missing".into() };
    let mut l = whole_file_lock(libc::F_WRLCK as libc::c_short);
    assert_eq!(unsafe { libc::fcntl(f.as_raw_fd(), libc::F_GETLK, &mut l) }, 0, "F_GETLK on {path:?}");
    if l.l_type == libc::F_UNLCK as libc::c_short {
        "none".into()
    } else {
        l.l_pid.to_string()
    }
}

/// The program an exit exec'd into: what of the store it was handed, in MCL_ONCE_EXEC_REPORT (the store's files it has
/// open; who holds the record of the image asked for before the exec, once that image's builder, if it forked one, is
/// building), then its own exit, of that image again (as dyld and the shared cache are the same images in the new
/// program) and of MCL_ONCE_EXEC_TAG2 (its own main image).
fn after_exec_in_child() {
    let store = std::path::PathBuf::from(std::env::var_os("MACLATOR_CACHE").unwrap());
    let tag: u8 = std::env::var("MCL_ONCE_TAG").unwrap().parse().unwrap();
    let mut report: String = store_fds(&store).iter().map(|n| format!("fd {n}\n")).collect();
    if std::env::var_os("MACLATOR_AOT_SYNC").is_none() {
        let log = std::env::var_os("MCL_TEST_BUILD_LOG").unwrap();
        let building = || std::fs::read_to_string(&log).unwrap_or_default().lines().any(|l| l.starts_with("start ") && l.contains(&format!(" key={} ", key_of(tag))));
        let end = Instant::now() + Duration::from_secs(30);
        while !building() && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    let record = store.join(format!("{}.{}.build", identity_prefix(&key_of(tag)).unwrap(), format_tag()));
    report.push_str(&format!("holder {}\nself {}\n", lock_holder(&record), std::process::id()));
    std::fs::write(std::env::var_os("MCL_ONCE_EXEC_REPORT").unwrap(), report).unwrap();
    let (start, end) = map_code(CODE_AT);
    register_main(&path_of(tag), uuid(tag), start, end, vec![start]);
    if let Ok(tag2) = std::env::var("MCL_ONCE_EXEC_TAG2") {
        let tag2: u8 = tag2.parse().unwrap();
        let (s2, e2) = map_code(CODE_AT_SLID);
        register_main(&path_of(tag2), uuid(tag2), s2, e2, vec![s2]);
        note_translated(s2);
    }
    ensure_loaded();
    note_translated(start + 4);
    save_profile();
}

/// A test's own store and build log, and the builders it started: any still running when the test ends are killed.
struct Run {
    dir: std::path::PathBuf,
    test: &'static str,
}

impl Run {
    fn new(test: &'static str, tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("mcl-once-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
        Run { dir, test }
    }

    fn cache(&self) -> std::path::PathBuf {
        self.dir.join("cache")
    }

    fn log(&self) -> std::path::PathBuf {
        self.dir.join("builds.log")
    }

    /// The command for one exit of image `tag` that ran block `block`, with `env` added.
    fn exit_cmd(&self, tag: u8, block: u64, env: &[(&str, &str)]) -> std::process::Command {
        let mut c = std::process::Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", self.test, "--test-threads=1", "--nocapture"])
            .env("MCL_ONCE_ROLE", "exit")
            .env("MCL_ONCE_TAG", tag.to_string())
            .env("MCL_ONCE_BLOCK", block.to_string())
            .env("MACLATOR_CACHE", self.cache())
            .env("MCL_TEST_BUILD_LOG", self.log())
            .env_remove("MACLATOR_AOT_SYNC")
            .env_remove("MACLATOR_NO_AOT_BUILD")
            .env_remove("MACLATOR_AOT_DONE_FILE")
            .env_remove("MCL_TEST_BUILD_DELAY_MS")
            .env_remove("MCL_TEST_ENGINE_ID")
            .env_remove("MCL_TEST_FREE_BYTES")
            .env_remove("MCL_TEST_BOOT")
            .env_remove("MCL_ONCE_SLID")
            .env_remove("MCL_ONCE_HOLD")
            .env_remove("MCL_ONCE_LIB")
            .env_remove("MCL_ONCE_REPLACE")
            .env_remove("MCL_ONCE_UNMAPPED")
            .env_remove("MCL_ONCE_NOLOAD")
            .env_remove("MCL_ONCE_LIB_AGAIN")
            .env_remove("MCL_ONCE_TAG2")
            .env_remove("MCL_TEST_NO_ROOM_IF")
            .env_remove("MCL_ONCE_EXEC")
            .env_remove("MCL_ONCE_EXEC_REPORT")
            .env_remove("MCL_ONCE_EXEC_TAG2")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());
        for (k, v) in env {
            c.env(k, v);
        }
        c
    }

    /// Run one exit to its end (the builder it may fork is not waited for); it must succeed.
    fn exit(&self, tag: u8, block: u64, env: &[(&str, &str)]) {
        let out = self.exit_cmd(tag, block, env).output().unwrap();
        assert!(out.status.success(), "an exit failed: {}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    }

    /// The build log's lines, as (what, pid, key, t).
    fn builds(&self) -> Vec<(String, u32, String, u128)> {
        let text = std::fs::read_to_string(self.log()).unwrap_or_default();
        text.lines()
            .filter_map(|l| {
                let mut it = l.split(' ');
                let what = it.next()?.to_string();
                let field = |s: Option<&str>, k: &str| s.and_then(|s| s.strip_prefix(k)).map(str::to_string);
                let pid = field(it.next(), "pid=")?.parse().ok()?;
                let key = field(it.next(), "key=")?;
                let t = field(it.next(), "t=")?.parse().ok()?;
                Some((what, pid, key, t))
            })
            .collect()
    }

    /// The value of `name=` on each build log line of `what`.
    fn field(&self, what: &str, name: &str) -> Vec<String> {
        let text = std::fs::read_to_string(self.log()).unwrap_or_default();
        let pre = format!("{name}=");
        text.lines()
            .filter(|l| l.split(' ').next() == Some(what))
            .filter_map(|l| l.split(' ').find_map(|f| f.strip_prefix(pre.as_str())).map(str::to_string))
            .collect()
    }

    fn count(&self, what: &str) -> usize {
        self.builds().iter().filter(|b| b.0 == what).count()
    }

    /// Wait (at most `secs`) until the log has `n` lines of `what`; whether it got there.
    fn wait_for(&self, what: &str, n: usize, secs: u64) -> bool {
        let end = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < end {
            if self.count(what) >= n {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.count(what) >= n
    }

    /// Builders that logged a start and no end or failure.
    fn running(&self) -> Vec<u32> {
        let b = self.builds();
        b.iter().filter(|s| s.0 == "start" && !b.iter().any(|e| e.1 == s.1 && (e.0 == "end" || e.0 == "fail"))).map(|s| s.1).collect()
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        for pid in self.running() {
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn gone(pid: u32, secs: u64) -> bool {
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        if unsafe { libc::kill(pid as i32, 0) } != 0 && unsafe { *libc::__error() } == libc::ESRCH {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

macro_rules! child_or {
    () => {
        match std::env::var("MCL_ONCE_ROLE").as_deref() {
            Ok("exit") => {
                exit_in_child();
                return;
            }
            Ok("after-exec") => {
                after_exec_in_child();
                return;
            }
            _ => {}
        }
    };
}

/// (1) N exits of one image build it once, though each saw a block the last build lacked.
#[test]
fn n_exits_of_one_image_build_it_once() {
    child_or!();
    let r = Run::new("aot::once_tests::n_exits_of_one_image_build_it_once", "n");
    for i in 0..6u64 {
        r.exit(31, i, &[("MACLATOR_AOT_SYNC", "1")]);
    }
    assert_eq!(r.count("start"), 1, "builds over six exits of one image: {:?}", r.builds());
    assert_eq!(r.count("end"), 1, "and the one build finished");
}

/// (1) Two engine formats sharing a store (the same engine with a code-generation switch, as two engines that
/// produce different translations) build an image once each, not once per exit: each cannot load the other's file.
#[test]
fn two_formats_sharing_a_store_build_once_each() {
    child_or!();
    let r = Run::new("aot::once_tests::two_formats_sharing_a_store_build_once_each", "fmt");
    for i in 0..6u64 {
        let env: &[(&str, &str)] = if i % 2 == 0 { &[("MACLATOR_AOT_SYNC", "1")] } else { &[("MACLATOR_AOT_SYNC", "1"), ("MACLATOR_JIT_OPERANDS", "0")] };
        r.exit(32, 0, env);
    }
    assert_eq!(r.count("start"), 2, "builds over six exits alternating between two formats, once each: {:?}", r.builds());
}

/// (1) An image that alternates between two bases (its preferred one taken now and then) is built once, not again after
/// every run at the other base: such a run's profile retires the translation at the first base.
#[test]
fn an_image_that_alternates_between_two_bases_is_built_once() {
    child_or!();
    let r = Run::new("aot::once_tests::an_image_that_alternates_between_two_bases_is_built_once", "alt");
    for i in 0..6u64 {
        let env: &[(&str, &str)] = if i % 2 == 0 { &[("MACLATOR_AOT_SYNC", "1")] } else { &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_SLID", "1")] };
        r.exit(33, i, env);
    }
    assert_eq!(r.count("start"), 1, "builds over six exits alternating between two bases: {:?}", r.builds());
}

/// (2) Two images whose exits come together are built one after the other, never at once, and both are built.
#[test]
fn two_images_queue_and_never_build_at_once() {
    child_or!();
    let r = Run::new("aot::once_tests::two_images_queue_and_never_build_at_once", "queue");
    let slow = [("MCL_TEST_BUILD_DELAY_MS", "1500")];
    let mut a = r.exit_cmd(41, 0, &slow).spawn().unwrap();
    let mut b = r.exit_cmd(42, 0, &slow).spawn().unwrap();
    assert!(a.wait().unwrap().success() && b.wait().unwrap().success(), "the two exits");
    assert!(r.wait_for("end", 2, 60), "both images were built within 60 s: {:?}", r.builds());
    let b = r.builds();
    let span = |key: &str| {
        let s = b.iter().find(|x| x.0 == "start" && x.2 == key).map(|x| x.3).unwrap();
        let e = b.iter().find(|x| x.0 == "end" && x.2 == key).map(|x| x.3).unwrap();
        (s, e)
    };
    let (a, c) = (span(&key_of(41)), span(&key_of(42)));
    assert!(a.1 <= c.0 || c.1 <= a.0, "the two builds overlapped: {a:?} and {c:?} ({b:?})");
    assert_eq!(r.count("start"), 2, "one build each");
    // A builder at background priority, with a CPU deadline and no core file.
    assert_eq!(r.field("start", "bg"), vec!["1", "1"], "PRIO_DARWIN_BG");
    assert!(r.field("start", "cpu").iter().all(|v| v.parse::<u64>().is_ok_and(|s| (AOT_BUILD_CPU_S..AOT_BUILD_CPU_S + 60).contains(&s))), "a CPU limit: {:?}", r.field("start", "cpu"));
    assert_eq!(r.field("start", "core"), vec!["0", "0"], "no core file");
}

/// (3) A builder keeps nothing the program had open: the program's lock is free while the builder it left waits or
/// builds, so the program can be started again.
#[test]
fn a_builder_keeps_nothing_the_program_had_open() {
    child_or!();
    let r = Run::new("aot::once_tests::a_builder_keeps_nothing_the_program_had_open", "fds");
    let lock = r.dir.join("held.lock");
    r.exit(52, 0, &[("MCL_TEST_BUILD_DELAY_MS", "5000"), ("MCL_ONCE_HOLD", lock.to_str().unwrap())]);
    assert!(r.wait_for("start", 1, 30), "the builder started");
    use std::os::unix::io::AsRawFd;
    let f = std::fs::OpenOptions::new().read(true).write(true).open(&lock).unwrap();
    let got = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(r.running().len(), 1, "control: the builder was still building when the lock was asked for");
    assert_eq!(got, 0, "the program's lock is held by the builder it left (errno {})", std::io::Error::last_os_error());
}

/// (1) A translation found unloadable since it was built (torn, or another engine's format over it) is built again,
/// but not at the next exit: after the hour, or in a later boot.
#[test]
fn an_unloadable_translation_is_built_again_later_not_at_once() {
    child_or!();
    let r = Run::new("aot::once_tests::an_unloadable_translation_is_built_again_later_not_at_once", "torn");
    r.exit(34, 0, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("end"), 1, "the first exit builds: {:?}", r.builds());
    std::fs::write(r.cache().join(format!("{}.aot", key_of(34))), b"MCLAOT04 torn by an unclean shutdown").unwrap();
    r.exit(34, 1, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("start"), 1, "an exit right after it found the file torn and built at once: {:?}", r.builds());
    r.exit(34, 2, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_BOOT", "a-later-boot")]);
    assert_eq!(r.count("end"), 2, "an exit in a later boot builds it again: {:?}", r.builds());
}

/// (3) A builder killed while building is recovered: while it lives, another exit of its image builds nothing; once
/// it is dead, the next exit builds the image, once, and later exits build nothing.
#[test]
fn a_killed_builder_is_recovered() {
    child_or!();
    let r = Run::new("aot::once_tests::a_killed_builder_is_recovered", "kill");
    r.exit(51, 0, &[("MCL_TEST_BUILD_DELAY_MS", "30000")]);
    assert!(r.wait_for("start", 1, 30), "the first builder started");
    let victim = r.builds()[0].1;
    r.exit(51, 1, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("start"), 1, "an exit while the image is being built built it again: {:?}", r.builds());
    unsafe { libc::kill(victim as i32, libc::SIGKILL) };
    assert!(gone(victim, 10), "the builder died");
    r.exit(51, 2, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!((r.count("start"), r.count("end")), (2, 1), "the exit after the kill built the image: {:?}", r.builds());
    r.exit(51, 3, &[("MACLATOR_AOT_SYNC", "1")]);
    r.exit(51, 4, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("start"), 2, "and later exits built nothing: {:?}", r.builds());
}

/// (4) A build that fails is not tried again at every exit, only when the engine changes, and then not within the
/// hour; and it leaves no temporary file.
#[test]
fn a_failed_build_is_not_retried_until_the_engine_changes() {
    child_or!();
    let r = Run::new("aot::once_tests::a_failed_build_is_not_retried_until_the_engine_changes", "fail");
    // The translation's own name is taken by a directory: every build of this image fails at its last step.
    std::fs::create_dir_all(r.cache().join(format!("{}.aot", key_of(61))).join("x")).unwrap();
    let one = [("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_ENGINE_ID", "engine-one")];
    let two = [("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_ENGINE_ID", "engine-two")];
    // Engine two in a later boot: another engine retries a failure, once, but not within the hour of the first.
    let two_later = [("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_ENGINE_ID", "engine-two"), ("MCL_TEST_BOOT", "a-later-boot")];
    let mut seen = Vec::new();
    for (i, env) in [&one[..], &one[..], &one[..], &two[..], &one[..], &two_later[..], &two_later[..]].into_iter().enumerate() {
        r.exit(61, i as u64, env);
        seen.push(r.count("start"));
    }
    assert_eq!(seen, vec![1, 1, 1, 1, 1, 2, 2], "builds begun after each exit (engine one three times, the two engines taking turns, then engine two in a later boot): {:?}", r.builds());
    assert_eq!(r.count("fail"), 2, "and each of them failed");
    let tmps: Vec<String> = std::fs::read_dir(r.cache()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".tmp")).collect();
    assert!(tmps.is_empty(), "temporary files left: {tmps:?}");
}

/// A full queue drops the request: with a build running and every place in the queue taken, an exit builds nothing;
/// once they are free, the next exit builds.
#[test]
fn a_full_queue_drops_the_request() {
    child_or!();
    let r = Run::new("aot::once_tests::a_full_queue_drops_the_request", "full");
    // This test process holds the builder lock and every queue place, as a running builder and its waiters would.
    let hold = |name: &str| {
        use std::os::unix::fs::OpenOptionsExt;
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).mode(0o600).open(r.cache().join(name)).unwrap();
        let mut l: libc::flock = unsafe { std::mem::zeroed() };
        l.l_type = libc::F_WRLCK as libc::c_short;
        l.l_whence = libc::SEEK_SET as libc::c_short;
        use std::os::unix::io::AsRawFd;
        assert_eq!(unsafe { libc::fcntl(f.as_raw_fd(), libc::F_SETLK, &mut l) }, 0, "lock {name}");
        f
    };
    let held: Vec<_> = ["aot-builder.lock", "aot-queue-0.lock", "aot-queue-1.lock"].iter().map(|n| hold(n)).collect();
    r.exit(71, 0, &[]);
    assert!(!r.wait_for("start", 1, 3), "an exit built while the queue was full: {:?}", r.builds());
    assert_eq!(r.count("forked"), 0, "the exit forked a builder for a full queue: {:?}", r.builds());
    drop(held);
    // Dropped, not waiting: nothing builds once the queue is free.
    assert!(!r.wait_for("start", 1, 2), "the request made while the queue was full built later: {:?}", r.builds());
    r.exit(71, 1, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("end"), 1, "control: with the queue free the next exit builds: {:?}", r.builds());
}

/// Under the floor of free space nothing is built, nor forked for; above it the next exit builds.
#[test]
fn nothing_is_built_without_room() {
    child_or!();
    let r = Run::new("aot::once_tests::nothing_is_built_without_room", "room");
    r.exit(81, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_FREE_BYTES", "4096")]);
    assert_eq!(r.count("start"), 0, "a build with 4 KiB free: {:?}", r.builds());
    r.exit(81, 1, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("end"), 1, "control: with room the next exit builds: {:?}", r.builds());
}

/// (1) A translation this engine can load, which the run did not load (it exited before it read the store), is not built
/// again: rebuilding would change nothing.
#[test]
fn a_translation_this_run_did_not_load_is_not_built_again() {
    child_or!();
    let r = Run::new("aot::once_tests::a_translation_this_run_did_not_load_is_not_built_again", "noload");
    r.exit(35, 0, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("end"), 1, "the first exit builds: {:?}", r.builds());
    r.exit(35, 1, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_NOLOAD", "1")]);
    r.exit(35, 2, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_NOLOAD", "1"), ("MCL_TEST_BOOT", "a-later-boot")]);
    assert_eq!(r.count("start"), 1, "exits that did not load a loadable translation built it again: {:?}", r.builds());
}

/// (1) A translation whose header and length are sound but whose index is damaged cannot be loaded, so it is built
/// again, after the hour or in a later boot, as a torn one is.
#[test]
fn a_translation_with_a_damaged_index_is_built_again_later() {
    child_or!();
    let r = Run::new("aot::once_tests::a_translation_with_a_damaged_index_is_built_again_later", "index");
    r.exit(36, 0, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("end"), 1, "the first exit builds: {:?}", r.builds());
    let file = r.cache().join(format!("{}.aot", key_of(36)));
    let mut bytes = std::fs::read(&file).unwrap();
    // The first index entry's pc (right after the 32-byte header), set outside the image.
    bytes[32..40].copy_from_slice(&1u64.to_le_bytes());
    std::fs::write(&file, &bytes).unwrap();
    r.exit(36, 1, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("start"), 1, "an exit right after it found the index damaged built at once: {:?}", r.builds());
    r.exit(36, 2, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_BOOT", "a-later-boot")]);
    assert_eq!(r.count("end"), 2, "an exit in a later boot builds it again: {:?}", r.builds());
}

/// (4) A build with nothing it can read (the image's code not mapped at that exit) is deferred, not failed: an exit in
/// a later boot, with the code mapped, builds it.
#[test]
fn a_build_with_nothing_to_read_is_deferred_not_failed() {
    child_or!();
    let r = Run::new("aot::once_tests::a_build_with_nothing_to_read_is_deferred_not_failed", "unread");
    r.exit(37, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_UNMAPPED", "1")]);
    assert_eq!(r.count("end"), 1, "the build ran and published nothing: {:?}", r.builds());
    assert!(!r.cache().join(format!("{}.aot", key_of(37))).exists(), "nothing was published");
    r.exit(37, 1, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("start"), 1, "the next exit, in the same boot, built at once: {:?}", r.builds());
    r.exit(37, 2, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_BOOT", "a-later-boot")]);
    assert_eq!(r.count("start"), 2, "an exit in a later boot, with the code mapped, builds it: {:?}", r.builds());
    assert!(r.cache().join(format!("{}.aot", key_of(37))).exists(), "and publishes it");
}

/// (1) A library unmapped and replaced by another at its addresses before the exit is not built: its translation would
/// be the other library's code under its UUID. The same library still in place is built (the control).
#[test]
fn a_library_replaced_at_its_addresses_is_not_built_under_its_name() {
    child_or!();
    let r = Run::new("aot::once_tests::a_library_replaced_at_its_addresses_is_not_built_under_its_name", "lib");
    let key = |tag: u8| format!("lib-{}-{:x}", hex(&[tag; 16]), LIB_AT);
    r.exit(0, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_LIB", "38"), ("MCL_ONCE_REPLACE", "1")]);
    assert!(!r.cache().join(format!("{}.aot", key(38))).exists(), "a translation was published under the replaced library's name: {:?}", r.builds());
    r.exit(0, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_LIB", "39")]);
    assert!(r.cache().join(format!("{}.aot", key(39))).exists(), "control: a library still in place is built: {:?}", r.builds());
}

/// (1) A library unloaded, and loaded again elsewhere before the exit, is built where it lives now: its stale first
/// registration does not stand in for the live copy (they share one build record).
#[test]
fn a_live_copy_of_a_library_is_built_though_a_stale_one_came_first() {
    child_or!();
    let r = Run::new("aot::once_tests::a_live_copy_of_a_library_is_built_though_a_stale_one_came_first", "relib");
    let key = |at: u64| format!("lib-{}-{:x}", hex(&[40u8; 16]), at);
    r.exit(0, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_LIB", "40"), ("MCL_ONCE_REPLACE", "1"), ("MCL_ONCE_LIB_AGAIN", "1")]);
    assert!(r.cache().join(format!("{}.aot", key(LIB_AT_AGAIN))).exists(), "the live copy was not built: {:?}", r.builds());
    assert!(!r.cache().join(format!("{}.aot", key(LIB_AT))).exists(), "and the stale one was not");
}

/// (3) A dead builder's temporary file may be what leaves the store without room, and only a builder used to sweep: the
/// exit sweeps it, finds room, and builds.
#[test]
fn an_exit_sweeps_a_dead_builders_file_before_it_gives_up_for_room() {
    child_or!();
    let r = Run::new("aot::once_tests::an_exit_sweeps_a_dead_builders_file_before_it_gives_up_for_room", "sweep");
    let tmp = r.cache().join(format!("{}.aot.999.tmp", key_of(93)));
    std::fs::write(&tmp, b"half a translation").unwrap();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(AOT_TMP_STALE_S + 60);
    std::fs::File::options().write(true).open(&tmp).unwrap().set_modified(old).unwrap();
    r.exit(93, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_TEST_NO_ROOM_IF", tmp.to_str().unwrap())]);
    assert!(!tmp.exists(), "the dead builder's file is still there");
    assert_eq!(r.count("end"), 1, "and the image was built: {:?}", r.builds());
}

/// (3) Room is asked about before each image: the first image's build can take the last of it, and the second is then
/// deferred, not built.
#[test]
fn room_is_asked_about_before_each_image() {
    child_or!();
    let r = Run::new("aot::once_tests::room_is_asked_about_before_each_image", "rooms");
    let first = r.cache().join(format!("{}.aot", key_of(94)));
    let second = r.cache().join(format!("{}.aot", image_key("main", &uuid(95), &path_of(95), CODE_AT_SLID)));
    r.exit(94, 0, &[("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_TAG2", "95"), ("MCL_TEST_NO_ROOM_IF", first.to_str().unwrap())]);
    assert!(first.exists(), "control: the first image was built: {:?}", r.builds());
    assert!(!second.exists(), "the second was built though the first took the room: {:?}", r.builds());
}

/// The report an after-exec program wrote: the store's files it had open, who held the record, and its pid.
fn exec_report(path: &std::path::Path) -> (Vec<String>, String, u32) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("no report from the program the exit exec'd ({e})"));
    let fds = text.lines().filter_map(|l| l.strip_prefix("fd ")).map(str::to_string).collect();
    let holder = text.lines().find_map(|l| l.strip_prefix("holder ")).unwrap().to_string();
    let me = text.lines().find_map(|l| l.strip_prefix("self ")).unwrap().parse().unwrap();
    (fds, holder, me)
}

/// Run one exit that execs (MCL_ONCE_EXEC) to the end of the program it exec'd, which keeps its pid; that pid.
fn exec_exit(r: &Run, tag: u8, env: &[(&str, &str)]) -> u32 {
    let child = r.exit_cmd(tag, 0, env).spawn().unwrap();
    let pid = child.id();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "the exit or the program it exec'd failed: {}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    pid
}

/// Across an exec: the image the exiting program asked to build stays held by its builder (the lock is the builder's,
/// not the program's, so the exec neither carries it nor drops it), the new program is handed no file of the store, and
/// it asks for the same image no second time while its own image is asked for once. The exec is of this binary itself,
/// as the kept re-launch execs the translator; a program the kernel starts is a new image of the translator too.
#[test]
fn an_exec_hands_the_new_program_nothing_of_the_store_and_asks_once() {
    child_or!();
    const NAME: &str = "aot::once_tests::an_exec_hands_the_new_program_nothing_of_the_store_and_asks_once";
    let r = Run::new(NAME, "exec");
    // An earlier exit, so the store has a builder lock that the exiting program opens before its exec (queue_full).
    r.exit(43, 0, &[("MACLATOR_AOT_SYNC", "1")]);
    assert_eq!(r.count("end"), 1, "the earlier exit built its image: {:?}", r.builds());
    let report = r.dir.join("exec-report.txt");
    let env = [("MCL_ONCE_EXEC", NAME), ("MCL_ONCE_EXEC_REPORT", report.to_str().unwrap()), ("MCL_ONCE_EXEC_TAG2", "45"), ("MCL_TEST_BUILD_DELAY_MS", "3000")];
    let pid = exec_exit(&r, 44, &env);
    let (fds, holder, me) = exec_report(&report);
    assert_eq!(me, pid, "the report is the exec'd program's, in the exit's own process");
    assert_eq!(fds, vec!["exec-canary".to_string()], "the store's files open in the program the exit exec'd (the canary, kept open on purpose, is the control)");
    let builder = r.builds().iter().find(|b| b.0 == "start" && b.2 == key_of(44)).map(|b| b.1);
    assert!(builder.is_some_and(|b| b != pid), "the image asked for before the exec was built by a builder of its own: {:?}", r.builds());
    assert_eq!(holder, builder.unwrap().to_string(), "who held the image's record while the exec'd program ran (the builder forked before the exec)");
    assert!(r.wait_for("end", 3, 60), "the builds ended within 60 s: {:?}", r.builds());
    let starts = |key: &str| r.builds().iter().filter(|b| b.0 == "start" && b.2 == key).count();
    assert_eq!(starts(&key_of(44)), 1, "builds of the image asked for before the exec and seen again after it: {:?}", r.builds());
    assert_eq!(starts(&image_key("main", &uuid(45), &path_of(45), CODE_AT_SLID)), 1, "builds of the exec'd program's own image: {:?}", r.builds());
    assert_eq!(r.count("forked"), 2, "one builder for the exit's exec and one for the exec'd program's exit: {:?}", r.builds());
}

/// A build made in the exiting process itself (MACLATOR_AOT_SYNC) lets go of every lock and file of the store before
/// the exec: the new program holds none, and its own image is built once.
#[test]
fn a_build_in_the_exiting_process_holds_nothing_across_its_exec() {
    child_or!();
    const NAME: &str = "aot::once_tests::a_build_in_the_exiting_process_holds_nothing_across_its_exec";
    let r = Run::new(NAME, "execsync");
    let report = r.dir.join("exec-report.txt");
    let env = [("MACLATOR_AOT_SYNC", "1"), ("MCL_ONCE_EXEC", NAME), ("MCL_ONCE_EXEC_REPORT", report.to_str().unwrap()), ("MCL_ONCE_EXEC_TAG2", "47")];
    let pid = exec_exit(&r, 46, &env);
    let (fds, holder, me) = exec_report(&report);
    assert_eq!(me, pid, "the report is the exec'd program's, in the exit's own process");
    assert_eq!(fds, vec!["exec-canary".to_string()], "the store's files open in the program the exit exec'd (the canary, kept open on purpose, is the control)");
    assert_eq!(holder, "none", "who held the record of the image built before the exec, after it");
    let by = |key: &str| r.builds().iter().filter(|b| b.0 == "start" && b.2 == key).map(|b| b.1).collect::<Vec<_>>();
    assert_eq!(by(&key_of(46)), vec![pid], "builds of the image built before the exec (and loaded after it): {:?}", r.builds());
    assert_eq!(by(&image_key("main", &uuid(47), &path_of(47), CODE_AT_SLID)), vec![pid], "builds of the exec'd program's own image: {:?}", r.builds());
    assert_eq!(r.count("forked"), 0, "nothing forked in-process builds");
}
