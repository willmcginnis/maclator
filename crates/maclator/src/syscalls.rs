//! Guest system call emulation.
//!
//! arm64 Darwin: `svc #0x80` with the number in x16 (positive: BSD, negative:
//! Mach trap, 0x80000000: platform call selected by x3). Arguments in x0..x7.
//! BSD results go in x0/x1 with the carry flag reporting errors.
//! Most calls pass straight through to the host kernel.

use crate::cache;
use crate::hostsys::{self, SysResult};
use crate::threads;
use maclator_core::cpu::Cpu;
use std::sync::atomic::{AtomicBool, Ordering};

pub static TRACE: AtomicBool = AtomicBool::new(false);

const PLATFORM_SYSCALL: i32 = 0x8000_0000u32 as i32;

// BSD numbers we care about
const SYS_SYSCALL: u64 = 0;
const SYS_EXIT: u64 = 1;
const SYS_VFORK: u64 = 66;
const SYS_FORK: u64 = 2;
const SYS_SIGACTION: u64 = 46;
const SYS_SIGALTSTACK: u64 = 53;
const SYS_MPROTECT: u64 = 74;
const SYS_SIGRETURN: u64 = 184;
const SYS_MMAP: u64 = 197;
const SYS_MUNMAP: u64 = 73;
const SYS_SYSCTL: u64 = 202;
const SYS_SYSCTLBYNAME: u64 = 274;
const SYS_SHARED_REGION_CHECK_NP: u64 = 294;
const SYS_BSDTHREAD_CREATE: u64 = 360;
const SYS_BSDTHREAD_TERMINATE: u64 = 361;
const SYS_BSDTHREAD_REGISTER: u64 = 366;
const SYS_WORKQ_OPEN: u64 = 367;
const SYS_WORKQ_KERNRETURN: u64 = 368;
const SYS_SHARED_REGION_MAP_AND_SLIDE_2_NP: u64 = 536;
const SYS_MAP_WITH_LINKING_NP: u64 = 550;
const SYS_EXECVE: u64 = 59;
const SYS_POSIX_SPAWN: u64 = 244;

const PROT_EXEC: u64 = 4;
const MAP_JIT: u64 = 0x800;

#[inline]
fn set_ok(cpu: &mut Cpu, r0: u64, r1: u64) {
    cpu.x[0] = r0;
    cpu.x[1] = r1;
    cpu.cf = 0;
}
#[inline]
fn set_err(cpu: &mut Cpu, errno: u64) {
    cpu.x[0] = errno;
    cpu.cf = 1;
}
#[inline]
fn set_res(cpu: &mut Cpu, r: SysResult) {
    if r.carry {
        set_err(cpu, r.rax);
    } else {
        set_ok(cpu, r.rax, r.rdx);
    }
}

/// Longest path name read from guest memory, NUL included. Above XNU's MAXPATHLEN (1024), so any
/// name the kernel would accept fits; a longer one is passed to the host kernel unread.
const GUEST_NAME_MAX: usize = 8192;

/// The NUL-terminated string at guest address `p`, read with `read` (a fault-safe copy) in pieces
/// that stop at each 4 KiB boundary, so a name that ends just before an unmapped page is still read.
/// None if any byte before the NUL cannot be read, or no NUL comes within `max` bytes.
fn guest_cstr(p: u64, max: usize, mut read: impl FnMut(u64, &mut [u8]) -> bool) -> Option<String> {
    const PIECE: u64 = 4096;
    let mut name = Vec::new();
    let mut at = p;
    while name.len() < max {
        let n = ((PIECE - at % PIECE) as usize).min(max - name.len());
        let mut buf = vec![0u8; n];
        if !read(at, &mut buf) {
            return None;
        }
        if let Some(end) = buf.iter().position(|&c| c == 0) {
            name.extend_from_slice(&buf[..end]);
            return Some(String::from_utf8_lossy(&name).into_owned());
        }
        name.extend_from_slice(&buf);
        at = at.checked_add(n as u64)?;
    }
    None
}

/// XNU's limit on a path name, its NUL included (sys/syslimits.h PATH_MAX, sys/param.h MAXPATHLEN):
/// namei's copyinstr refuses a name of 1024 bytes or more with ENAMETOOLONG before any lookup
/// (measured on macOS 27.0.1, arm64, natively and under Rosetta, and on 27.0.1's x86 kernel: 1023 looks up).
const MAXPATHLEN: usize = 1024;

/// Whether bsd() may rewrite the name it read. A name XNU would refuse
/// as too long is passed through untouched, so the host kernel refuses it the same way; translated,
/// walk_dotdot or the redirect could otherwise shrink it into one that runs. `from_utf8_lossy`
/// only ever lengthens a name, so a name under the limit in guest bytes may be passed through too:
/// the host then answers for it, which is the call as it came.
fn rewritable(name: &str) -> bool {
    name.len() < MAXPATHLEN
}

fn guest_name(p: u64) -> Option<String> {
    guest_cstr(p, GUEST_NAME_MAX, crate::guestmem::read)
}

/// Whether XNU never reads call `n`'s name: fstatat and fstatat64 with AT_FDONLY use the descriptor
/// alone (xnu-12377.1.9 bsd/vfs/vfs_syscalls.c, fstatat_internal: fp_getfvp, no nameiat).
fn name_unread(n: u64, args: &[u64; 8]) -> bool {
    matches!(n, 469 | 470) && args[3] & AT_FDONLY != 0
}

/// The name bsd() looks at for path argument `i`, or None to pass the call through as it came:
/// a NULL name, a name XNU never reads (name_unread; `read` is not called), or one `read` cannot
/// read in full. Passed through, the host kernel answers as XNU does: EFAULT where it reads the
/// name, and its own earlier answer where it checks something first (an invalid fstatat or
/// faccessat flag is EINVAL before the name is read).
fn call_name(n: u64, args: &[u64; 8], i: usize, read: impl FnOnce(u64) -> Option<String>) -> Option<String> {
    let p = args[i];
    if p == 0 {
        return None;
    }
    let trace = TRACE.load(Ordering::Relaxed);
    if name_unread(n, args) {
        if trace {
            eprintln!("[path] #{n} {p:#x} not read (AT_FDONLY): passed through");
        }
        return None;
    }
    let name = read(p);
    if name.is_none() && trace {
        eprintln!("[path] #{n} {p:#x} unreadable: passed through");
    }
    name
}

/// What to say when call `n`'s name at `p` could not be read and the host call still succeeded:
/// the kernel read what guestmem::read could not, so it walked the guest's path as it came, with
/// no redirect, into the host's own files. Reaches stderr without --trace: nothing else shows it.
/// A failed host call says nothing: that is the answer an unreadable name is expected to get.
fn unread_name_warning(n: u64, p: u64, host_ok: bool) -> Option<String> {
    host_ok.then(|| {
        format!("[maclator] warning: #{n} name at {p:#x} could not be read but the host call succeeded: it ran unredirected")
    })
}

/// Handle SVC. `cpu.pc` already points after the SVC.
pub fn handle_svc(cpu: &mut Cpu) {
    let num = cpu.x[16] as u32 as i32;
    if num == PLATFORM_SYSCALL {
        match cpu.x[3] {
            2 => cpu.tpidrro_el0 = cpu.x[0],
            3 => cpu.x[0] = cpu.tpidrro_el0,
            _ => {}
        }
        return;
    }
    if num < 0 {
        mach_trap(cpu, (-num) as u64);
        return;
    }
    let mut n = num as u64;
    let mut args = [cpu.x[0], cpu.x[1], cpu.x[2], cpu.x[3], cpu.x[4], cpu.x[5], cpu.x[6], cpu.x[7]];
    let original_args = args;
    if n == SYS_SYSCALL {
        n = args[0];
        args = [args[1], args[2], args[3], args[4], args[5], args[6], args[7], 0];
    }
    let trace = TRACE.load(Ordering::Relaxed);
    if trace {
        eprintln!("[sys] {}#{}({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}) pc={:#x}", syscall_name(n), n, args[0], args[1], args[2], args[3], args[4], args[5], cpu.pc - 4);
    }
    bsd(cpu, n, args);
    if trace {
        eprintln!("[sys]   -> {}{:#x}", if cpu.cf != 0 { "ERR " } else { "" }, cpu.x[0]);
    }
    // Host callbacks never run ARM code, and must interrupt host waits so we
    // can deliver at a guest boundary. For SA_RESTART, save a context that
    // retries this SVC after the guest signal trampoline has returned.
    if cpu.cf != 0 && cpu.x[0] == libc::EINTR as u64 && crate::signals::should_restart(n) {
        cpu.x[..8].copy_from_slice(&original_args);
        cpu.pc = cpu.pc.wrapping_sub(4);
    }
}

/// Guest paths of open fds, so relative *at lookups can resolve both the dyld
/// cache and optional Xcode component overlay without changing the host files.
pub(crate) static FD_VPATHS: crate::forksync::FMutex<Option<std::collections::HashMap<i32, String>>> = crate::forksync::FMutex::new(None);

fn vpath_join(base: &str, rel: &str) -> String {
    if base == "/" {
        format!("/{rel}")
    } else {
        format!("{}/{rel}", base.trim_end_matches('/'))
    }
}

fn fd_guest_path(fd: i32) -> Option<String> {
    if let Some(path) = FD_VPATHS.lock().unwrap().as_ref().and_then(|m| m.get(&fd).cloned()) {
        return Some(path);
    }
    // A descriptor may have been inherited from a native parent. Recover its
    // guest spelling from the host path rather than requiring a prior open.
    if crate::paths::overlay_root().is_some() {
        let mut path = [0u8; 1024];
        if unsafe { libc::fcntl(fd, libc::F_GETPATH, path.as_mut_ptr()) } == 0 {
            let end = path.iter().position(|&c| c == 0)?;
            let host = std::str::from_utf8(&path[..end]).ok()?;
            return crate::paths::guest_path(host).or_else(|| Some(host.to_string()));
        }
    }
    None
}

/// The host's own path for descriptor `fd`: F_GETPATH, mapped back
/// to guest space through the overlay where one is set, else the host path itself. Unlike
/// fd_guest_path this is NOT gated on an overlay root, because the firmlink correction it serves
/// applies with only a sysroot set. None if the descriptor has no path (F_GETPATH failed).
fn host_fd_path(fd: i32) -> Option<String> {
    if fd < 0 {
        return None;
    }
    let mut path = [0u8; 1024];
    if unsafe { libc::fcntl(fd, libc::F_GETPATH, path.as_mut_ptr()) } != 0 {
        return None;
    }
    let end = path.iter().position(|&c| c == 0)?;
    let host = std::str::from_utf8(&path[..end]).ok()?;
    Some(crate::paths::guest_path(host).unwrap_or_else(|| host.to_string()))
}

/// Whether host path `h` is the sysroot or beneath it: only there is the guest path H
/// computed the one a descriptor records, since F_GETPATH gives the sysroot's host spelling.
fn inside_sysroot(h: &str, dir: Option<&str>) -> bool {
    dir.is_some_and(|d| h.strip_prefix(d).is_some_and(|r| r.is_empty() || r.starts_with('/')))
}

/// The guest path a descriptor opened by a successful open records. When the host opened
/// something inside the sysroot: the guest path H
/// computed, which F_GETPATH cannot map back out of the sysroot. Otherwise the host's own F_GETPATH
/// (`host`), because what the host opened is where a later relative *at call starts: an un-redirected
/// open and also a redirected one that climbed out of the prefix, which round 4 recorded
/// lexically (`<cache dir>/<up>System/Volumes/Data/Applications/..` recorded /System/Volumes/Data, and
/// a later `../../Library/dyld/<cache file>` then reached the sysroot). If F_GETPATH fails, nothing is
/// recorded, so a later relative call goes to the host as it came; round 4 fell back to the lexical
/// path there, which is round 3's bug again. A record is a snapshot taken at the open: a later rename
/// of the directory is not seen (the host's own descriptor follows it; the record does not), and the
/// host's spelling is recorded (/private/var where the guest said /var).
fn descriptor_record(guest: &str, in_sysroot: bool, host: impl FnOnce() -> Option<String>) -> Option<String> {
    if in_sysroot {
        Some(recorded_path(guest))
    } else {
        host().map(|h| recorded_path(&h))
    }
}

/// A working directory the guest reached through a cache prefix. The
/// host stands in the sysroot's own directory (chdir and __pthread_chdir are redirected like open), so every
/// relative name that stays inside the sysroot is walked by the host kernel from the right directory, with XNU's
/// own credential, length and permission checks, whether Maclator rewrites that call or not. What the host
/// cannot know is the guest's parent of the sysroot's root: that is the prefix's own parent, not the sysroot's
/// host parent. So a record keeps the prefix (its physical spelling, paths::physical_prefix) and the directory's
/// physical path below the sysroot, and is used only where a `..` climbs out of the sysroot (escape_at) or a
/// child is re-launched (cwd_forward). `key` is the directory's (device, inode) when recorded.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CwdRec {
    key: (u64, u64),
    prefix: &'static str,
    rel: String,
}

impl CwdRec {
    /// The guest path of the directory.
    fn guest(&self) -> String {
        format!("{}{}", self.prefix, self.rel)
    }
}

/// The process working directory's record (None: the host's own path is the guest's). H v8.1 kept up
/// to 16 records keyed by directory, shared by every thread; the oldest was pushed out by sixteen other
/// directories and a record for one spelling answered for a thread standing there under another.
/// XNU keeps one process working directory and one per thread, so this does too.
pub(crate) static CWD_VPATHS: crate::forksync::FMutex<Option<CwdRec>> = crate::forksync::FMutex::new(None);

thread_local! {
    /// This thread's own working directory (__pthread_chdir, __pthread_fchdir): None when it has none
    /// and uses the process one; Some(None) when it has one the redirect did not take. Guest threads are host
    /// threads (threads::bsdthread_create), so this lives exactly as long as the host's per-thread directory.
    static THREAD_CWD: std::cell::RefCell<Option<Option<CwdRec>>> = const { std::cell::RefCell::new(None) };
}

/// Whether this thread has its own working directory.
fn thread_has_cwd() -> bool {
    THREAD_CWD.with(|t| t.borrow().is_some())
}

/// The record lookups from this thread use: its own, else the process one.
fn cwd_slot() -> Option<CwdRec> {
    THREAD_CWD.with(|t| t.borrow().clone()).unwrap_or_else(|| CWD_VPATHS.lock().unwrap().clone())
}

/// Record `r` for the process working directory, or for this thread's own (`per_thread`).
fn cwd_set(per_thread: bool, r: Option<CwdRec>) {
    if per_thread {
        THREAD_CWD.with(|t| *t.borrow_mut() = Some(r));
    } else {
        *CWD_VPATHS.lock().unwrap() = r;
    }
}

/// In a forked child. XNU gives the child, as its process working directory, the forking thread's own
/// one if it has one (measured on macOS 27.0.1, arm64: fork, posix_spawn and execve from a thread after __pthread_chdir
/// all start in that directory), and the child's one thread has none of its own.
pub(crate) fn cwd_after_fork_child() {
    let r = cwd_slot();
    THREAD_CWD.with(|t| *t.borrow_mut() = None);
    *CWD_VPATHS.lock().unwrap() = r;
}

/// The sysroot's path as getattrlist FULLPATH and F_GETPATH spell it, which is
/// what rec_at strips: realpath (paths::set_cache_dir) can keep another spelling of the same directory (measured by
/// /System/Volumes/Data/private/var against /private/var). The cache dir itself if that cannot be read.
fn sysroot_spelling() -> Option<String> {
    let d = crate::paths::cache_dir()?;
    std::ffi::CString::new(d.as_str()).ok().and_then(|c| full_path_of(&c)).or(Some(d))
}

/// The record for the directory at `key` whose host path, read now, is `host`: its path below the sysroot
/// with the prefix `prefix`. None when the directory is not inside the sysroot (then the host's path is the
/// guest's: a directory reached through a link out of the sysroot) or its path cannot be read.
fn rec_at(key: (u64, u64), prefix: &'static str, host: Option<String>, sysroot: Option<&str>) -> Option<CwdRec> {
    let h = host?;
    let rel = h.strip_prefix(sysroot?)?;
    (rel.is_empty() || rel.starts_with('/')).then(|| CwdRec { key, prefix, rel: rel.to_string() })
}

/// What a record `r` for this thread's working directory is now, "." being at `here`. A record for
/// another directory is no record (something Maclator did not see moved the working directory; the host's own
/// path answers, as before H v8). A record whose guest path no longer leads here (the directory or one above it
/// was renamed) is recomputed from where the host's working directory is now (`host`): its prefix
/// stays, its path below the sysroot is read again; if it is no longer inside the sysroot there is no record.
/// Never a guess at the parent: the prefix is the one the guest came in by, and the rest is the kernel's.
/// The physical path, read now, always decides the place below the
/// sysroot when it can be read: a link left where the directory used to be (S/a renamed to S/x/y, then S/a -> x/y)
/// still leads the old spelling to the same inode, which kept H v9's stale `rel`. Only when "." cannot be read
/// (deeper than MAXPATHLEN) does the record stand on its guest path still leading here; nothing depends on its
/// `rel` for a climb any more (cwd_depth counts the `..` physically).
/// A place the spelling test cannot put inside the sysroot (the sysroot's FULLPATH
/// spelling unreadable while an ancestor is unsearchable, so `sysroot` is the realpath spelling, which can differ) keeps
/// the record when "." is inside the sysroot by identity (`inside_id`: cwd_depth reaches the root).
fn cwd_check(r: CwdRec, here: (u64, u64), at: impl FnOnce(&str) -> Option<(u64, u64)>, host: impl FnOnce() -> Option<String>,
             sysroot: Option<&str>, inside_id: impl FnOnce() -> bool) -> Option<CwdRec> {
    if r.key != here {
        return None;
    }
    match host() {
        Some(h) => rec_at(here, r.prefix, Some(h), sysroot).or_else(|| inside_id().then_some(r)),
        None => (at(&r.guest()) == Some(here) || inside_id()).then_some(r),
    }
}

/// The sysroot's root (device, inode): the one taken when the sysroot
/// was set (paths::cache_dir_id), else a stat of its path now. H v9 only stat'ed the path, and a failed stat (an
/// ancestor made unsearchable after the guest chdir'd in, while relative names from the held directory still walk)
/// was read as "no `..` climbs out". None now means unknown, and the callers refuse (escape_closed).
/// When both are known they must agree. A sysroot replaced since startup (a new
/// directory renamed over the old path) has a new identity at its path; the guest's chdir by that path lands in it,
/// and a check against the startup identity alone would miss its root. Disagreeing is unknown (refused).
fn sysroot_root() -> Option<(u64, u64)> {
    root_decision(crate::paths::cache_dir_id(), crate::paths::cache_dir().and_then(|s| path_dev_ino(&s)))
}

/// The sysroot's root from the identity taken at startup and a stat of its path now: either when only one
/// is known, the one when both agree, None (unknown) when they disagree or neither is known.
fn root_decision(startup: Option<(u64, u64)>, now: Option<(u64, u64)>) -> Option<(u64, u64)> {
    match (startup, now) {
        (Some(a), Some(b)) => (a == b).then_some(a),
        (a, b) => a.or(b),
    }
}

/// Whether relative name `name` has a `..` component (one that could climb out).
fn has_dotdot(name: &str) -> bool {
    !name.starts_with('/') && name.split('/').any(|c| c == "..")
}

/// escape_at against the sysroot's root `root`, closed when that is unknown: Err(()) for a name
/// with a `..` then (the callers refuse it with a warning), never "nothing climbs out".
fn escape_closed(name: &str, root: Option<(u64, u64)>, at: impl Fn(&str) -> Option<(u64, u64)>) -> Result<Option<usize>, ()> {
    if !has_dotdot(name) {
        return Ok(None);
    }
    root.map(|k| escape_at(name, k, at)).ok_or(())
}

/// How many `..` from the working directory reach the sysroot's root (`root`), by (device,
/// inode): the depth the host really stands at, whatever spelling a record keeps. `at` stats a relative name. None when
/// a stat fails or the root is not reached within MAXPATHLEN / 3 levels.
fn cwd_depth(root: (u64, u64), at: impl Fn(&str) -> Option<(u64, u64)>) -> Option<usize> {
    let mut p = String::from(".");
    for k in 0..=MAXPATHLEN / 3 {
        if at(&p)? == root {
            return Some(k);
        }
        p = if k == 0 { "..".to_string() } else { format!("{p}/..") };
    }
    None
}

/// The name the host walks to check the leading part `name[..off]` that an escaped name drops:
/// that part, then "./" padding of the same length as what follows it, so a symlink in it is expanded against the
/// same remaining length XNU's lookup sees (its ENAMETOOLONG test is the link's length plus the rest of the name).
fn lead_probe(name: &str, off: usize) -> String {
    let l = name.len() - off;
    let mut s = name[..off].to_string();
    s.push_str(&"./".repeat(l / 2));
    if l % 2 == 1 {
        s.push('.');
    }
    s
}

/// The host's walk of the dropped leading part (lead_probe), with the call's credentials
/// (`ids`): faccessat(F_OK) from the working directory, so a directory in it the call may not search (EACCES with
/// the real ids for access), a link too long for the rest of the name (ENAMETOOLONG) or a loop (ELOOP) fails as
/// XNU's walk of the whole name does. Ok when there is no leading part. Not carried: the symlink count, which
/// starts again for the escaped form.
fn lead_walk(name: &str, off: usize, ids: Ids) -> Result<(), u64> {
    if off == 0 {
        return Ok(());
    }
    let Ok(c) = std::ffi::CString::new(lead_probe(name, off)) else { return Err(libc::EINVAL as u64) };
    let flag = if ids == Ids::Real { 0 } else { libc::AT_EACCESS };
    if unsafe { libc::faccessat(libc::AT_FDCWD, c.as_ptr(), libc::F_OK, flag) } == 0 {
        return Ok(());
    }
    Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EACCES) as u64)
}

/// This thread's working-directory record, checked (cwd_check) and, if recomputed, stored back where it
/// came from (unless that slot changed meanwhile). One stat of ".", one of the record's redirect.
fn cwd_now() -> Option<CwdRec> {
    let r = cwd_slot()?;
    let here = path_dev_ino(".")?;
    let checked = cwd_check(r.clone(), here, redirected_dev_ino, || full_path_of(c"."), sysroot_spelling().as_deref(), || here_depth().is_some());
    if checked.as_ref() != Some(&r) {
        let own = thread_has_cwd();
        if own {
            THREAD_CWD.with(|t| {
                let mut t = t.borrow_mut();
                if t.as_ref() == Some(&Some(r.clone())) {
                    *t = Some(checked.clone());
                }
            });
        } else {
            let mut p = CWD_VPATHS.lock().unwrap();
            if p.as_ref() == Some(&r) {
                *p = checked.clone();
            }
        }
    }
    checked
}

/// The (device, inode) guest path `g` names through the redirect; None when the redirect does
/// not take it or the stat fails.
fn redirected_dev_ino(g: &str) -> Option<(u64, u64)> {
    crate::paths::redirect(g).and_then(|h| path_dev_ino(&h))
}

/// The byte offset in relative name `name` of the first `..` the host
/// would take AT the sysroot's root (`root`, its device and inode), where it climbs to the sysroot's host parent
/// while the guest climbs to the cache prefix's own parent. `at(lead)` is where the name's leading part `lead` lands
/// from the working directory, links followed ("." for none). Every `..` before that one is taken inside the
/// sysroot and is the host's to walk. None when no `..` climbs out, or when a leading part does not land (the
/// host's walk stops there too, for the reason it gives). One stat per `..`.
fn escape_at(name: &str, root: (u64, u64), at: impl Fn(&str) -> Option<(u64, u64)>) -> Option<usize> {
    let mut off = 0;
    for c in name.split('/') {
        if c == ".." {
            let lead = &name[..off];
            match at(if lead.is_empty() { "." } else { lead }) {
                Some(k) if k == root => return Some(off),
                Some(_) => {}
                None => return None,
            }
        }
        off += c.len() + 1;
    }
    None
}

/// Whether relative leading part `lead` (what escaped_path drops) crosses a symlink,
/// which a NOFOLLOW_ANY call refuses with ELOOP natively: an lstat of each of its components from the working directory
/// (every one is walked, since a `..` follows). A first try asked faccessat(AT_SYMLINK_NOFOLLOW_ANY) about `lead`
/// itself, which does not refuse a link that is the last component (row N2 still opened, measured on macOS 27.0.1, arm64).
fn lead_has_link(lead: &str) -> bool {
    let mut p = String::new();
    for c in lead.split('/').filter(|c| !c.is_empty()) {
        if !p.is_empty() {
            p.push('/');
        }
        p.push_str(c);
        if c != "." && c != ".." && std::fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink()) {
            return true;
        }
    }
    false
}

/// A name that climbed out of the sysroot at `off` (escape_at), as the guest path it names: the prefix the
/// working directory was reached by, then the name from that `..` on. What comes before it is inside the sysroot
/// and ends at its root, which is the prefix.
fn escaped_path(r: &CwdRec, name: &str, off: usize) -> String {
    format!("{}/{}", r.prefix, &name[off..])
}

/// For host path `h` inside the sysroot, the same place as a name relative to a working directory `depth`
/// levels below the sysroot's root ("../" `depth` times, then `h`'s path below the sysroot); None when `h` is not inside
/// the sysroot. `depth` is counted physically (cwd_depth), not from a record's spelling.
fn reentry_relative(h: &str, depth: usize, sysroot: &str) -> Option<String> {
    let below = h.strip_prefix(sysroot)?;
    if !(below.is_empty() || below.starts_with('/')) {
        return None;
    }
    // The sysroot's root itself would become "." or "..", which rmdir and rename
    // refuse with EINVAL where the guest's name does not: it keeps its absolute form.
    if below.trim_matches('/').is_empty() {
        return None;
    }
    // Measured on macOS 27.0.1, arm64: a target still holding `.` or `..` (an unresolved walk) is not a place
    // below the sysroot; made relative, its `..` climbed out again from the working directory.
    if below.split('/').any(|c| c == "." || c == "..") {
        return None;
    }
    let mut parts: Vec<&str> = vec![".."; depth];
    parts.extend(below.split('/').filter(|c| !c.is_empty()));
    let mut s = parts.join("/");
    if s.is_empty() {
        s.push('.');
    } else if below.ends_with('/') {
        s.push('/');
    }
    Some(s)
}

/// Where a relative name through descriptor `fd` starts when the descriptor
/// has no guest path (fd_guest_path: opened outside /System, or inherited, with no overlay): the host's own
/// path for it, read now (F_GETPATH), unless that is inside the sysroot, whose host path is not the guest's.
/// None leaves the name unrewritten, as before H v8. One fcntl per such relative name.
fn unrecorded_fd_base(fd: i32, getpath: impl FnOnce(i32) -> Option<String>, sysroot: Option<&str>) -> Option<String> {
    let h = getpath(fd)?;
    (!inside_sysroot(&h, sysroot)).then_some(h)
}

/// Where a relative name in a plain call or through AT_FDCWD starts when the working directory has no
/// record: the host's own working directory (getattrlist FULLPATH of ".", no descriptor), mapped back through the
/// overlay; None when no sysroot or overlay is set (nothing could be redirected) or it is inside the sysroot
/// (reached by its host path: the host's walk is the guest's).
fn cwd_host_base(host: impl FnOnce() -> Option<String>, sysroot: Option<&str>, overlay: bool) -> Option<String> {
    if sysroot.is_none() && !overlay {
        return None;
    }
    let h = host()?;
    if inside_sysroot(&h, sysroot) {
        return None;
    }
    Some(crate::paths::guest_path(&h).unwrap_or(h))
}

/// A kernel probe name for a name joined to the working directory's host path `base`, made
/// relative again, so the host looks it up from the working directory it holds, with the call's credentials, as
/// XNU does, not from / through ancestors those credentials may not search. `p` itself when it is not below `base`.
fn probe_from_cwd<'a>(p: &'a str, base: Option<&str>) -> &'a str {
    base.and_then(|b| p.strip_prefix(b)).and_then(|r| r.strip_prefix('/')).filter(|r| !r.is_empty()).unwrap_or(p)
}

/// Whether a relative name joined to its base may be rewritten. A descriptor's recorded guest path
/// keeps H v7's rule (always); a new base (the working directory, an unrecorded descriptor) only below MAXPATHLEN,
/// so the host otherwise answers for the relative name as before H v8.
fn joined_rewritable(recorded: bool, orig: &str, full: &str) -> bool {
    recorded || orig.starts_with('/') || full.len() < MAXPATHLEN
}

/// Which base a relative name starts from: the working directory for a plain call or AT_FDCWD.
fn cwd_is_base(at: bool, fd: i32) -> bool {
    !at || fd == libc::AT_FDCWD
}

/// The prefix a chdir-like call's new working directory was reached by, if any: the guest path H computed
/// for it when that is under a cache prefix (an absolute name, or one that climbed out of the sysroot), else the
/// current record's when the name was left to the host from inside the sysroot. None: the host's path is the guest's.
fn chdir_prefix(guest: Option<&str>, left_from: Option<&CwdRec>) -> Option<&'static str> {
    guest.and_then(crate::paths::physical_prefix).or_else(|| left_from.map(|r| r.prefix))
}

/// What an arm64 child re-launched through Maclator is told about its working directory: the guest path
/// of this thread's record, if it has a live one. The child keeps only its prefix (cwd_adopt).
pub(crate) fn cwd_forward() -> Option<String> {
    cwd_now().map(|r| r.guest())
}

/// The record --guest-cwd `arg` gives a child standing at `here` whose host path is `host`: the prefix
/// `arg` names, if any, with the child's own path below the sysroot. A posix_spawn chdir file action relative to a
/// working directory inside the sysroot keeps the child under the same prefix (row J5); one that left the sysroot
/// gives no record (`host` outside it).
fn adopt_arg(arg: &str, here: Option<(u64, u64)>, host: impl FnOnce() -> Option<String>, sysroot: Option<&str>) -> Option<CwdRec> {
    let prefix = crate::paths::physical_prefix(arg)?;
    rec_at(here?, prefix, host(), sysroot)
}

/// Take the prefix a parent forwarded (main.rs, --guest-cwd). Returns whether a record was taken.
pub(crate) fn cwd_adopt(arg: &str) -> bool {
    let r = adopt_arg(arg, path_dev_ino("."), || full_path_of(c"."), sysroot_spelling().as_deref());
    let taken = r.is_some();
    if taken {
        cwd_set(false, r);
    }
    taken
}

/// The name arguments of the path-taking calls bsd_read does not otherwise rewrite, as (name
/// index, descriptor index for an *at call): every call in the 27.0.1 SDK's syscall.h that takes a path XNU looks up
/// (bsd/kern/syscalls.master), less those path_arg lists. symlink's and symlinkat's first argument is the link's
/// text, not a lookup. access_extended (284) takes an array of names: escape_accessx. posix_spawn (244) is not here:
/// XNU looks its program up after the file actions run. Not covered: AF_UNIX socket paths
/// (bind, connect, connectx: a sockaddr, not a name), graftdmg and ungraftdmg (549, 555), fs_snapshot's names (518).
fn escape_slots(n: u64) -> &'static [(usize, Option<usize>)] {
    match n {
        10 | 14 | 15 | 16 | 34 | 51 | 56 | 59 | 61 | 132 | 136 | 137 | 138 | 157 | 159 | 161 | 165 | 191 | 200 | 205 | 216
        | 221 | 225 | 226 | 234 | 236 | 238 | 240 | 242 | 277 | 279 | 280 | 282 | 291 | 292 | 341 | 342 | 345 | 364 | 380
        | 382 | 383 | 384 | 385 | 441 | 484 => &[(0, None)],
        9 | 128 | 223 | 227 | 537 => &[(0, None), (1, None)],
        57 | 167 | 424 => &[(1, None)],
        214 | 218 | 249 | 467 | 468 | 472 | 475 | 524 | 553 | 554 => &[(1, Some(0))],
        462 | 465 | 471 | 488 => &[(1, Some(0)), (3, Some(2))],
        474 | 517 => &[(2, Some(1))],
        _ => &[],
    }
}

/// A relative name from a working directory with NO record that climbs out of the
/// sysroot (escape_at, by device and inode, so whatever the spelling): the guest's parent there is unknown (it came in
/// through something Maclator did not see: an inherited descriptor, a path too deep to read), and the host's is the
/// sysroot's host parent, which is never the guest's. Refused, with a warning, rather than sent there. A guest that
/// entered the sysroot by its own host path is refused too.
/// With the sysroot's root unknown, any `..` from inside it is refused (closed, not open).
fn unrecorded_climb(name: &str) -> bool {
    if !has_dotdot(name) {
        return false;
    }
    // only from inside the sysroot: "." below it, or "." whose path cannot be read (deeper than MAXPATHLEN)
    // (or by identity, when the spellings differ)
    let inside = match full_path_of(c".") {
        Some(h) => sysroot_spelling().is_some_and(|s| inside_sysroot(&h, Some(&s))) || here_depth().is_some(),
        None => true,
    };
    inside && escape_closed(name, sysroot_root(), path_dev_ino).map_or(true, |o| o.is_some())
}

/// Where a name that climbed out of the sysroot goes, for a call bsd_read does not otherwise rewrite:
/// its guest path (escaped_path) with the `..` taken in the guest's view as for any absolute name (resolve_dotdot,
/// the call's effective ids), through the redirect when it lands under a prefix again, as the host path itself when
/// it does not. When it lands inside the sysroot, the same place relative to the working directory
/// (fit_target). Err(ENAMETOOLONG) when nothing fits: refused, with a warning, never sent on to walk from the
/// sysroot's host parent.
/// With the call's credentials `ids`.
fn escape_target(full: &str, ids: Ids) -> Result<String, u64> {
    let dotdot = resolve_dotdot(full, true, ids, |p| guest_node_as(p, ids), kernel_lookup, |m| crate::paths::redirect(m).is_some());
    let target = match dotdot {
        Dotdot::Keep => crate::paths::redirect(full).unwrap_or_else(|| full.to_string()),
        Dotdot::Path(p) => crate::paths::redirect(&p).unwrap_or(p),
        Dotdot::Host(k) => k,
    };
    fit_target(target, crate::paths::cache_dir().as_deref(), here_depth)
}

/// This thread's working directory's depth below the sysroot's root (cwd_depth), if known.
fn here_depth() -> Option<usize> {
    sysroot_root().and_then(|k| cwd_depth(k, path_dev_ino))
}

/// What an escaped name is sent as. A `target` inside the
/// sysroot `sysroot` goes relative to the working directory (reentry_relative, `depth` levels down, counted
/// physically) whenever that fits: the host then walks only the sysroot's own directories from the one it holds, as
/// XNU walks the guest's name from the guest's, never the sysroot's ancestors (which the guest's walk never searches,
/// and which may have been made unsearchable since). Else `target` as it is, below MAXPATHLEN; else ENAMETOOLONG.
fn fit_target(target: String, sysroot: Option<&str>, depth: impl FnOnce() -> Option<usize>) -> Result<String, u64> {
    if inside_sysroot(&target, sysroot) {
        if let Some(rel) = depth().and_then(|d| reentry_relative(&target, d, sysroot?)).filter(|r| r.len() < MAXPATHLEN) {
            return Ok(rel);
        }
    }
    if target.len() < MAXPATHLEN {
        return Ok(target);
    }
    Err(libc::ENAMETOOLONG as u64)
}

/// Rewrite each relative name of call `n` (escape_slots) that climbs out of the sysroot from
/// a working directory reached through a prefix. Names that stay inside, absolute names and names through a real
/// descriptor are left as they came. The new names are pushed onto `keep`, which must outlive the call. Err: refuse
/// the call with that errno (escape_target).
fn escape_unhandled(n: u64, args: &mut [u64; 8], keep: &mut Vec<std::ffi::CString>) -> Result<(), u64> {
    let slots = escape_slots(n);
    if slots.is_empty() || crate::paths::cache_dir().is_none() {
        return Ok(());
    }
    let mut rec: Option<Option<CwdRec>> = None;
    for &(i, fdi) in slots {
        if fdi.is_some_and(|f| args[f] as i32 != libc::AT_FDCWD) || args[i] == 0 {
            continue;
        }
        let Some(name) = guest_name(args[i]).filter(|o| rewritable(o)) else { continue };
        if let Some(t) = escape_name(n, &name, &mut rec, Ids::Effective, beneath(n, args))? {
            let Ok(c) = std::ffi::CString::new(t) else { continue };
            args[i] = c.as_ptr() as u64;
            keep.push(c);
        }
    }
    Ok(())
}

/// What relative name `name` of call `n` (one bsd_read does not otherwise rewrite) is sent as: None to
/// leave it as it came (absolute, no `..`, no record and no climb, or a `..` that stays inside the sysroot); Some(the
/// escaped target) when a `..` climbs out of the sysroot from a recorded working directory. Err refuses the call,
/// with a warning: a climb from a directory with no record (ENOENT), the sysroot's root unknown (ENOENT),
/// the dropped leading part failing the host's walk with the call's credentials `ids` (its errno),
/// or a target too long (ENAMETOOLONG). `rec` caches this call's record.
/// `bound`, the call's RESOLVE_BENEATH flag (beneath): a climb out is refused
/// ENOTCAPABLE, never rewritten into a name that stays beneath.
fn escape_name(n: u64, name: &str, rec: &mut Option<Option<CwdRec>>, ids: Ids, bound: bool) -> Result<Option<String>, u64> {
    if !has_dotdot(name) {
        return Ok(None);
    }
    let Some(r) = rec.get_or_insert_with(cwd_now).clone() else {
        if unrecorded_climb(name) {
            eprintln!("[maclator] warning: #{n} {name} climbs out of the sysroot from a working directory with no record: refused (ENOENT)");
            return Err(libc::ENOENT as u64);
        }
        return Ok(None);
    };
    let Ok(esc) = escape_closed(name, sysroot_root(), path_dev_ino) else {
        eprintln!("[maclator] warning: #{n} {name}: the sysroot's root cannot be identified, so a climb out of it cannot be ruled out: refused (ENOENT)");
        return Err(libc::ENOENT as u64);
    };
    let Some(off) = esc else { return Ok(None) };
    lead_walk(name, off, ids)?;
    if bound {
        return Err(libc::ENOTCAPABLE as u64);
    }
    let full = escaped_path(&r, name, off);
    let target = match escape_target(&full, ids) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[maclator] warning: #{n} {name} climbs out of the sysroot to {full}, too long to send: refused (ENAMETOOLONG)");
            return Err(e);
        }
    };
    if still_climbs(&target) {
        eprintln!("[maclator] warning: #{n} {name} climbs out of the sysroot to {full}, which could not be resolved: refused (ENOENT)");
        return Err(libc::ENOENT as u64);
    }
    if TRACE.load(Ordering::Relaxed) {
        eprintln!("[path] #{n} {name}: climbs out of the sysroot at byte {off} -> {full} -> {target}");
    }
    Ok(Some(target))
}

/// Whether a name an escaped one was rewritten to is relative and would itself still take a `..` at the
/// sysroot's root (or the root is unknown): it must never be sent, since the host would climb to the sysroot's host
/// parent; nor an absolute one inside the sysroot with a `..` below it (the same climb, spelled from /). A last guard
/// after the rewrite.
fn still_climbs(target: &str) -> bool {
    let sys = crate::paths::cache_dir();
    if inside_sysroot(target, sys.as_deref()) && sys.as_deref().and_then(|d| target.strip_prefix(d)).is_some_and(|b| b.split('/').any(|c| c == "..")) {
        return true;
    }
    has_dotdot(target) && escape_closed(target, sysroot_root(), path_dev_ino).map_or(true, |o| o.is_some())
}

/// accessx_descriptor's size (sys/unistd.h) and the most XNU takes (ACCESSX_MAX_TABLESIZE).
const ACCESSX_DESC: usize = 16;
const ACCESSX_MAX_TABLESIZE: usize = 16 * 1024;

/// access_extended (284)'s table `tbl` with each name `f` maps to Some
/// replaced. The table is read as XNU's access_extended reads it: descriptors of
/// 16 bytes, a u32 name offset first; the descriptors end at the lowest name offset (rounded down to a descriptor), so
/// the names may be in any order; an offset of 0 repeats the previous name (not for the first); an offset must be past
/// its own descriptor and below the table's size; the table's last byte must be NUL. Anything else XNU refuses
/// (EINVAL), so the table is left as it came (Ok(None)) for the host to refuse. A changed table is laid out again: the
/// descriptors, then each named descriptor's name (replaced or as it came), NUL-terminated, in order, the offsets
/// moved to match, which keeps every rule above. (A first draft appended the new name and pointed the first descriptor
/// at it, which moved the table's end: EINVAL on macOS 27.0.1, arm64, row AX1.) Err from `f`, or E2BIG past 16 KiB.
fn accessx_rewrite(tbl: &[u8], mut f: impl FnMut(&str) -> Result<Option<String>, u64>) -> Result<Option<Vec<u8>>, u64> {
    let size = tbl.len();
    if size < ACCESSX_DESC + 2 || tbl[size - 1] != 0 {
        return Ok(None);
    }
    let off_at = |i: usize| u32::from_ne_bytes([tbl[i * ACCESSX_DESC], tbl[i * ACCESSX_DESC + 1], tbl[i * ACCESSX_DESC + 2], tbl[i * ACCESSX_DESC + 3]]) as usize;
    let desc_max = (size - 2) / ACCESSX_DESC;
    let mut count = desc_max;
    let mut i = 0;
    while i < count {
        let o = off_at(i);
        let j = o / ACCESSX_DESC;
        if j > desc_max || (j != 0 && j <= i) || o >= size || (j == 0 && i == 0) {
            return Ok(None);
        }
        if j != 0 && j < count {
            count = j;
        }
        i += 1;
    }
    let mut names: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut changed = false;
    for i in 0..count {
        let o = off_at(i);
        if o / ACCESSX_DESC == 0 {
            continue;
        }
        let rest = &tbl[o..];
        let raw = &rest[..rest.iter().position(|&b| b == 0).unwrap_or(rest.len())];
        let new = match std::str::from_utf8(raw) {
            Ok(name) => f(name)?,
            Err(_) => None,
        };
        changed |= new.is_some();
        names.push((i, new.map_or_else(|| raw.to_vec(), String::into_bytes)));
    }
    if !changed {
        return Ok(None);
    }
    let mut v = tbl[..count * ACCESSX_DESC].to_vec();
    for (i, name) in names {
        let at = v.len();
        v.extend_from_slice(&name);
        v.push(0);
        v[i * ACCESSX_DESC..i * ACCESSX_DESC + 4].copy_from_slice(&(at as u32).to_ne_bytes());
    }
    if v.len() > ACCESSX_MAX_TABLESIZE {
        return Err(libc::E2BIG as u64);
    }
    Ok(Some(v))
}

/// access_extended's names, each handled as escape_name handles a relative name (H v9 read
/// none of them), with the real ids, as lookup_ids says the call uses. The table is read from guest memory (args[0],
/// args[1] bytes, at most 16 KiB); a rewritten copy goes in `keep`, which must outlive the call, and args point at it.
fn escape_accessx(args: &mut [u64; 8], keep: &mut Vec<u8>) -> Result<(), u64> {
    let len = args[1] as usize;
    if crate::paths::cache_dir().is_none() || args[0] == 0 || len == 0 || len > ACCESSX_MAX_TABLESIZE {
        return Ok(());
    }
    let mut tbl = vec![0u8; len];
    if !crate::guestmem::read(args[0], &mut tbl) {
        return Ok(());
    }
    let mut rec: Option<Option<CwdRec>> = None;
    let r = accessx_rewrite(&tbl, |name| escape_name(284, name, &mut rec, Ids::Real, false));
    if r == Err(libc::E2BIG as u64) {
        eprintln!("[maclator] warning: #284: the rewritten access table would pass 16 KiB: refused (E2BIG)");
    }
    if let Some(v) = r? {
        *keep = v;
        args[0] = keep.as_ptr() as u64;
        args[1] = keep.len() as u64;
    }
    Ok(())
}

fn rewrite_fd_path(cpu: &mut Cpu, args: &[u64; 8]) {
    let mut path = [0u8; 1024];
    if !crate::guestmem::read(args[2], &mut path) {
        return;
    }
    let Some(end) = path.iter().position(|&c| c == 0) else { return };
    let Ok(host) = std::str::from_utf8(&path[..end]) else { return };
    let Some(guest) = crate::paths::guest_path(host) else { return };
    let Ok(guest) = std::ffi::CString::new(guest) else { return };
    if !crate::guestmem::write(args[2], guest.as_bytes_with_nul()) {
        set_err(cpu, libc::EFAULT as u64);
    }
}

fn rewrite_fullpath_attribute(cpu: &mut Cpu, n: u64, args: &[u64; 8]) {
    // The general attribute buffer contains variable layouts. Handle only the
    // exact FULLPATH-only request whose ABI is a length and one attrreference;
    // never reinterpret a mixed attribute list using this small layout.
    let (attrs, out, capacity) = if n == 476 { (args[2], args[3], args[4]) } else { (args[1], args[2], args[3]) };
    let mut requested = [0u8; 24];
    if !crate::guestmem::read(attrs, &mut requested) ||
        u16::from_le_bytes(requested[0..2].try_into().unwrap()) != 5 ||
        u32::from_le_bytes(requested[4..8].try_into().unwrap()) != 0x0800_0000 ||
        requested[8..24].iter().any(|&b| b != 0) {
        return;
    }
    let mut header = [0u8; 12];
    if capacity < 12 || !crate::guestmem::read(out, &mut header) {
        return;
    }
    let old_size = u32::from_le_bytes(header[0..4].try_into().unwrap()) as u64;
    let offset = i32::from_le_bytes(header[4..8].try_into().unwrap()) as i64;
    let length = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;
    if offset < 8 || length == 0 || length > 1024 || old_size > capacity ||
        (offset as u64).checked_add(4 + length).is_none_or(|end| end > old_size) {
        return;
    }
    let mut path = vec![0u8; length as usize];
    if !crate::guestmem::read(out + 4 + offset as u64, &mut path) {
        return;
    }
    let Some(end) = path.iter().position(|&c| c == 0) else { return };
    let Ok(host) = std::str::from_utf8(&path[..end]) else { return };
    let Some(guest) = crate::paths::guest_path(host) else { return };
    let Ok(guest) = std::ffi::CString::new(guest) else { return };
    let bytes = guest.as_bytes_with_nul();
    let size = (12 + bytes.len() + 3) & !3;
    if size as u64 > capacity {
        set_err(cpu, libc::ERANGE as u64);
        return;
    }
    let mut rewritten = vec![0u8; size];
    rewritten[0..4].copy_from_slice(&(size as u32).to_le_bytes());
    rewritten[4..8].copy_from_slice(&8i32.to_le_bytes());
    rewritten[8..12].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
    rewritten[12..12 + bytes.len()].copy_from_slice(bytes);
    if !crate::guestmem::write(out, &rewritten) {
        set_err(cpu, libc::EFAULT as u64);
    }
}

/// Path-taking syscalls: (index of the path argument, whether it is an *at call). Every *at call
/// here takes ONE path, relative to its descriptor in args[0]. The two-path *at calls (renameat
/// 465, linkat 471, symlinkat 474) are deliberately absent: they pass through unrewritten, each
/// path against its own descriptor.
fn path_arg(n: u64) -> (Option<usize>, bool) {
    match n {
        5 | 398 => (Some(0), false),
        // chdir and __pthread_chdir too, so a working directory under a cache prefix is the
        // sysroot's directory, as it is the guest's.
        12 | 33 | 58 | 188 | 190 | 220 | 338 | 340 | 348 => (Some(0), false),
        463 | 464 | 466 | 469 | 470 | 473 | 476 => (Some(1), true),
        _ => (None, false),
    }
}

/// The guest path a syscall's name refers to: itself if absolute, else the name joined to `dir`,
/// the guest path it starts from: its descriptor's in an *at call, the working directory's
/// (cwd_host_base, or a record's) in a plain call or through AT_FDCWD.
/// `dir` is asked only for a relative, non-empty name; None from it leaves the name
/// unrewritten, as before.
fn guest_full_path(name: &str, _at: bool, dir: impl FnOnce() -> Option<String>) -> Option<String> {
    if name.starts_with('/') {
        Some(name.to_string())
    } else if !name.is_empty() {
        // An empty name is not the directory itself: XNU answers ENOENT for
        // openat(dirfd, "") and so does the host, so leave it unrewritten. macOS 27's
        // dyld relies on that: for each name in {arm64e_x1, arm64e} it skips building
        // the x1 file name unless its own slice is arm64e.x1, calls openat(dirfd, "")
        // anyway, and moves on only on failure. Joined to the directory's path, the
        // empty name opened the cache directory and dyld stopped with "shared cache
        // file pread() failed".
        dir().map(|b| vpath_join(&b, name))
    } else {
        None
    }
}

/// What a guest path names, as far as `walk_dotdot` needs to know.
#[derive(Debug, PartialEq)]
enum Node {
    /// a directory that is not a symlink
    Dir,
    /// a symlink, with its target as stored
    Link(String),
    /// anything else: absent, unreadable, not a directory, a target that is not UTF-8
    Other,
}

/// What walk_dotdot decided.
#[derive(Debug, PartialEq)]
enum Walk {
    /// No symlink was met: the path with its `..` taken in the guest's view, or None to keep it as it came.
    Lexical(Option<String>),
    /// A symlink stood in what a `.` or `..` looks through: the name for the kernel to look up itself, which
    /// is the directories walked before the first such link (their `..` already taken) and then the rest of
    /// the path exactly as written, from that link on.
    Kernel(String),
}

/// Where bsd() sends a name that has a `..`.
#[derive(Debug, PartialEq)]
enum Dotdot {
    /// unchanged: the redirect sees the name as it came
    Keep,
    /// this guest path, which the redirect then sees
    Path(String),
    /// this name to the host as it is, never redirected: the host kernel does every check of the lookup
    Host(String),
}

/// What the host kernel answered for a probe name.
#[derive(Debug, PartialEq)]
enum Probe {
    /// the lookup failed (any errno): the real call fails on the way too, for the same reason
    Refused,
    /// the lookup succeeded and landed here (the host's spelling, mapped back through an overlay)
    At(String),
    /// the lookup succeeded, but where it landed could not be read
    Unplaced,
}

/// The `..` components of absolute guest path `path`, taken in the guest's view: a directory
/// before a `..` is popped, and `..` at / stays at / (xnu bsd/vfs/vfs_lookup.c, lookup's ISDOTDOT case). The
/// guest's view matters because the redirect matches a cache prefix as a string: before this,
/// `<cache dir>/../dyld/x` became `<sysroot>/../dyld/x`, the sysroot's HOST parent (test
/// rows Y7, Y8). `node` says what a guest path names (guest_node_as). Lexical(None), and the caller keeps
/// `path` exactly as it was, when the path is relative or has no `..` component.
///
/// H never walks a symlink. At each `.` and `..`, if any
/// component of what has been walked is a symlink (explicit, right before the `..`, or implicit, further back,
/// which the host's own lstat expands), the walk stops with Kernel: the host kernel looks the name up itself,
/// so its length (linklen + what follows the link >= MAXPATHLEN: ENAMETOOLONG), credential (access and
/// faccessat without AT_EACCESS walk as the real ids), loop (more than 32 links: ELOOP) and permission checks are XNU's own,
/// not an emulation of them. Six rounds of review each found another corner where H's own link walk diverged
/// from XNU's (an expansion's length counted against the wrong remainder, through `.`, after a later link;
/// a link read with the effective ids where the call looks up with the real ones); H v7 has no link walk.
/// Links never reach this walk in real use: of 6,442 path calls of 13 real programs on macOS 27.0.1, arm64, none had an
/// absolute or joined name with a `..`.
///
/// Without a link, as H v2-v6: where XNU refuses (a `..` after something that is not a searchable directory
/// whose `..` is its lexical parent: ENOENT, ENOTDIR, EACCES, a firmlink) the walk stops, and the `..` already
/// taken are kept while the refused `..` and the rest stay as written, so the host refuses inside the
/// sysroot rather than in its parent; Lexical(None) if nothing was walked. A `.` after something that is not
/// a searchable directory stops the same way. A trailing `/` or `/.` is kept: `file/.` is
/// ENOTDIR where `file` opens. A trailing `..` (or `.`) ends the result in `/.`: rmdir of `x/..` is
/// EINVAL, which `x/.` is too and `x/` is not (measured on macOS 27.0.1, arm64).
fn walk_dotdot(path: &str, node: impl Fn(&str) -> Node) -> Walk {
    if !path.starts_with('/') || !path.split('/').any(|c| c == "..") {
        return Walk::Lexical(None);
    }
    let comps: Vec<&str> = path.split('/').collect();
    let abs = |acc: &[(&str, usize)]| format!("/{}", acc.iter().map(|(c, _)| *c).collect::<Vec<&str>>().join("/"));
    let with_rest = |mut out: String, rest: &[&str]| {
        for r in rest {
            out.push('/');
            out.push_str(r);
        }
        out
    };
    // acc: each walked component, with its index in `comps`, so the kernel's name keeps the rest as written
    let mut acc: Vec<(&str, usize)> = Vec::new();
    let mut walked = false;
    for (i, &c) in comps.iter().enumerate() {
        if c.is_empty() {
            continue;
        }
        if c != "." && c != ".." {
            acc.push((c, i));
            continue;
        }
        // A symlink anywhere in what is walked: the kernel looks the name up itself.
        if let Some(j) = (0..acc.len()).find(|&j| matches!(node(&abs(&acc[..=j])), Node::Link(_))) {
            let tail = comps[acc[j].1..].join("/");
            return Walk::Kernel(if j == 0 { format!("/{tail}") } else { format!("{}/{tail}", abs(&acc[..j])) });
        }
        if c == "." {
            // A `.` is a no-op only after a directory the caller can search.
            // After a file XNU answers ENOTDIR, after a non-searchable dir EACCES: hand the host the `.` and
            // the rest, which it refuses exactly as XNU does.
            if !acc.is_empty() && !matches!(node(&abs(&acc)), Node::Dir) {
                let out = with_rest(format!("{}/.", abs(&acc)), &comps[i + 1..]);
                return Walk::Lexical((out != path).then_some(out));
            }
            continue;
        }
        if acc.is_empty() {
            continue; // `..` at / is /
        }
        if matches!(node(&abs(&acc)), Node::Dir) {
            acc.pop();
            walked = true;
            continue;
        }
        if !walked {
            return Walk::Lexical(None);
        }
        let out = with_rest(format!("{}/..", abs(&acc)), &comps[i + 1..]);
        return Walk::Lexical((out != path).then_some(out));
    }
    let mut out = abs(&acc);
    if out.len() > 1 {
        if matches!(path.rsplit('/').find(|c| !c.is_empty()), Some(".." | ".")) {
            out.push_str("/.");
        } else if path.ends_with('/') {
            out.push('/');
        }
    }
    Walk::Lexical(Some(out))
}

/// The name the kernel is asked about for Kernel name `k`, and what follows where it lands. The
/// last component is replaced by `.` components of the same length in bytes: XNU refuses a link whose target
/// plus what follows it in the name reaches MAXPATHLEN, so the probe keeps every link's remainder exactly as
/// long as in the real name (measured on macOS 27.0.1, arm64: the probe fails with the real name's errno at every
/// boundary), while the last component itself, which may exist only in the sysroot,
/// is not looked up. A name ending in `..`, `.` or `/` is asked about whole.
fn kernel_name(k: &str) -> (String, String) {
    let (dir, last) = k.rsplit_once('/').unwrap_or(("", k));
    match last {
        "" => (k.to_string(), "/".to_string()),
        "." | ".." => (k.to_string(), "/.".to_string()),
        _ => {
            let mut pad = "./".repeat(last.len() / 2);
            if last.len() % 2 == 1 {
                pad.push('.');
            }
            (format!("{dir}/{pad}"), format!("/{last}"))
        }
    }
}

/// Where bsd() sends absolute guest path `path`. Without a symlink, walk_dotdot's answer. With one:
/// a NOFOLLOW_ANY call (`follow` false) goes to the host, which answers ELOOP as XNU does; otherwise `kernel`
/// looks the probe name up with the call's credentials `ids`. If it lands where `redirected` says the redirect
/// takes over (a cache prefix, or an overlaid path), the call goes there, through the redirect; in every other
/// case the host gets the name and does every check itself. A stated limit: the probe walks the host's tree,
/// so a symlink INSIDE a cache prefix is looked up in the host's own directory there, not in the sysroot
/// (there are none in macOS 27.0.1's system, its cryptex or the sysroots, measured); and when
/// the lookup succeeds with the call's credentials but getattrlist (the effective ids) cannot place it, the
/// host gets the name unredirected.
fn resolve_dotdot(path: &str, follow: bool, ids: Ids, node: impl Fn(&str) -> Node, kernel: impl Fn(&str, Ids) -> Probe,
                  redirected: impl Fn(&str) -> bool) -> Dotdot {
    match walk_dotdot(path, node) {
        Walk::Lexical(None) => Dotdot::Keep,
        Walk::Lexical(Some(p)) => Dotdot::Path(p),
        Walk::Kernel(k) => {
            if !follow {
                return Dotdot::Host(k);
            }
            let (probe, after) = kernel_name(&k);
            match kernel(&probe, ids) {
                Probe::At(at) => {
                    let landed = format!("{}{after}", at.trim_end_matches('/'));
                    if redirected(&landed) { Dotdot::Path(landed) } else { Dotdot::Host(k) }
                }
                Probe::Refused | Probe::Unplaced => Dotdot::Host(k),
            }
        }
    }
}

/// The host kernel's lookup of `probe` for resolve_dotdot: faccessat(F_OK) with the call's own
/// credentials (the real ids for access and faccessat without AT_EACCESS), then where it landed.
fn kernel_lookup(probe: &str, ids: Ids) -> Probe {
    let Ok(c) = std::ffi::CString::new(probe) else { return Probe::Refused };
    let flag = if ids == Ids::Real { 0 } else { libc::AT_EACCESS };
    if unsafe { libc::faccessat(libc::AT_FDCWD, c.as_ptr(), libc::F_OK, flag) } != 0 {
        return Probe::Refused;
    }
    full_path_of(&c).map_or(Probe::Unplaced, Probe::At)
}

/// getattrlist(ATTR_CMN_FULLPATH) of `c`, following links: where a lookup lands, spelled as F_GETPATH
/// spells it (measured on macOS 27.0.1, arm64: the cryptex for /System/Cryptexes/OS/..., /Applications for the
/// firmlinked /System/Volumes/Data/Applications, /private/tmp for /tmp), but with no descriptor, so no number in
/// the guest's descriptor table is ever taken, even briefly. Mapped back through an overlay as host_fd_path is.
fn full_path_of(c: &std::ffi::CStr) -> Option<String> {
    #[repr(C)]
    struct AttrList {
        bitmapcount: u16,
        reserved: u16,
        commonattr: u32,
        volattr: u32,
        dirattr: u32,
        fileattr: u32,
        forkattr: u32,
    }
    extern "C" {
        fn getattrlist(path: *const libc::c_char, list: *mut AttrList, buf: *mut libc::c_void, size: libc::size_t,
                       options: libc::c_uint) -> libc::c_int;
    }
    // ATTR_BIT_MAP_COUNT 5, ATTR_CMN_FULLPATH 0x0800_0000 (sys/attr.h)
    let mut al = AttrList { bitmapcount: 5, reserved: 0, commonattr: 0x0800_0000, volattr: 0, dirattr: 0, fileattr: 0, forkattr: 0 };
    let mut buf = vec![0u8; 4 + 8 + MAXPATHLEN + 8];
    if unsafe { getattrlist(c.as_ptr(), &mut al, buf.as_mut_ptr().cast(), buf.len(), 0) } != 0 {
        return None;
    }
    // u32 length, then an attrreference_t (i32 offset from itself, u32 length), then the string
    let off = i32::from_ne_bytes(buf.get(4..8)?.try_into().ok()?);
    let len = u32::from_ne_bytes(buf.get(8..12)?.try_into().ok()?) as usize;
    let start = usize::try_from(4i64 + i64::from(off)).ok()?;
    let bytes = buf.get(start..start.checked_add(len)?)?;
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let host = std::str::from_utf8(&bytes[..end]).ok()?;
    Some(crate::paths::guest_path(host).unwrap_or_else(|| host.to_string()))
}

/// Where a path H changed goes when the redirect no longer takes it. `before` is the path
/// as joined, `full` as normalised. When `before` was under a cache prefix and `full` has climbed out
/// of it, the name and descriptor the call came with would walk from the sysroot, whose parent is
/// not the guest's: the host gets `full`, which is also what the descriptor's guest path records. A
/// path that was never under a prefix is left to the host as it came.
fn left_the_prefix(before: &str, full: &str, redirect: impl Fn(&str) -> Option<String>) -> Option<String> {
    (before != full && redirect(before).is_some()).then(|| full.to_string())
}

/// The guest path a descriptor opened by `p` records: `p` without the trailing `/` and
/// `/.` that walk_dotdot keeps for the host's sake, so `<cache dir>/sub/..` records `<cache dir>`.
fn recorded_path(p: &str) -> String {
    let mut s = p.to_string();
    while s.len() > 1 {
        if let Some(t) = s.strip_suffix("/.") {
            s = if t.is_empty() { "/".to_string() } else { t.to_string() };
        } else if let Some(t) = s.strip_suffix('/') {
            s = t.to_string();
        } else {
            break;
        }
    }
    s
}

/// What guest path `p` names, through Maclator's redirect. A cache prefix itself names
/// the sysroot, which the guest sees as a directory however the host reaches it (a sysroot given
/// through a symlink is followed: that link is Maclator's plumbing, not a guest symlink); any
/// other path is the host's lstat of where the redirect puts it, or of the path itself.
/// `ids` are the credentials the call's own lookup uses.
fn guest_node_as(p: &str, ids: Ids) -> Node {
    let host = crate::paths::redirect(p);
    if crate::paths::is_cache_root(p) {
        let followed = host.as_deref().and_then(|h| std::fs::metadata(h).ok());
        // Measured on macOS 27.0.1, arm64: a sysroot whose path cannot be searched now (an ancestor made
        // mode 0 after the guest went in) is still the directory whose identity was taken at startup
        // (paths::cache_dir_id); as Other, a `..` after the prefix stopped the walk and the name went out unresolved.
        let dir = match followed {
            Some(m) => m.is_dir(),
            None => crate::paths::cache_dir_id().is_some(),
        };
        return if dir { Node::Dir } else { Node::Other };
    }
    host_node_as(&host.unwrap_or_else(|| p.to_string()), ids)
}

#[cfg(test)]
fn guest_node(p: &str) -> Node {
    guest_node_as(p, Ids::Effective)
}

/// The host's lstat of `h`, as a Node; a directory is searched with the call's credentials `ids`.
fn host_node_as(h: &str, ids: Ids) -> Node {
    match std::fs::symlink_metadata(h) {
        Ok(m) if m.file_type().is_symlink() => {
            std::fs::read_link(h).ok().and_then(|t| t.to_str().map(str::to_string)).map_or(Node::Other, Node::Link)
        }
        Ok(m) if m.is_dir() => {
            if dir_searchable(h, ids) && dotdot_is_lexical(h) { Node::Dir } else { Node::Other }
        }
        _ => Node::Other,
    }
}

#[cfg(test)]
fn host_node(h: &str) -> Node {
    host_node_as(h, Ids::Effective)
}

/// Whether the `..` of host directory `h` is its lexical parent. A firmlink breaks that:
/// /System/Volumes/Data/Applications is firmlinked to
/// /Applications, so XNU's `..` from it is /, not /System/Volumes/Data (xnu vfs_cache.c, the
/// firmlink-parent case; measured on macOS 27.0.1, arm64). A lexical pop there hands the host a different
/// directory, so such an `h` is not a directory H may pop: the walk stops and the host takes that `..`
/// itself. `stat` follows a link in the parent's spelling, so `lnk/q`'s parent `lnk` compares as
/// where the link leads, which is what a `..` from `lnk/q` reaches. Two stats per popped `..`.
fn dotdot_is_lexical(h: &str) -> bool {
    use std::os::unix::fs::MetadataExt;
    let parent = match h.rsplit_once('/') {
        Some(("", _)) => "/",
        Some((p, _)) => p,
        None => return false,
    };
    match (std::fs::metadata(format!("{h}/..")), std::fs::metadata(parent)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

/// Whether the caller may SEARCH directory `h`. XNU checks search
/// permission on each directory a `..` leaves; `symlink_metadata` does not, so a lexical pop of a
/// directory the caller cannot search opened a name XNU refuses with EACCES. `faccessat(X_OK,
/// AT_EACCESS)` is the kernel's own check against the effective ids, so root's directory-search
/// bypass is honoured (measured on macOS 27.0.1, arm64: a 0600 dir is searchable as root, EACCES as a non-root admin user).
/// `h` is already known to be a directory, so X_OK here means search, not file-execute.
/// With the credentials the call itself looks
/// up with. access() and faccessat() without AT_EACCESS walk the path as the REAL ids (XNU's access1 and
/// faccessat_internal take a copy of the credential with the real ids for the whole namei), so a real
/// uid 501 with effective uid 0 is refused at a 0600 directory there while open() is not (measured on
/// macOS 27.0.1, arm64). `faccessat` with flags 0 is the kernel's own check against the real ids.
fn dir_searchable(h: &str, ids: Ids) -> bool {
    let flag = match ids {
        Ids::Effective => libc::AT_EACCESS,
        Ids::Real => 0,
    };
    std::ffi::CString::new(h).is_ok_and(|c| unsafe {
        libc::faccessat(libc::AT_FDCWD, c.as_ptr(), libc::X_OK, flag) == 0
    })
}

/// Which credentials a call's lookup uses.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ids {
    Effective,
    Real,
}

/// faccessat's flag to check the effective ids (sys/fcntl.h AT_EACCESS).
const AT_EACCESS: u64 = 0x0010;

/// The credentials call `n`'s lookup uses: the real ids for access (33) and for
/// faccessat (466) without AT_EACCESS in its flags slot, the effective ids for everything else bsd()
/// rewrites. access_extended (284) also uses the real ids, but bsd() does not read its names.
fn lookup_ids(n: u64, args: &[u64; 8]) -> Ids {
    match n {
        33 => Ids::Real,
        466 if args[3] & AT_EACCESS == 0 => Ids::Real,
        _ => Ids::Effective,
    }
}

/// Whether call `n` may follow a symlink before a `..`: not with a NOFOLLOW_ANY flag,
/// in its *at slot (nofollow_any) or in open's, open_nocancel's or getattrlist's own.
fn follows_symlinks(n: u64, args: &[u64; 8]) -> bool {
    let plain = match n {
        5 | 398 => args[1] & O_NOFOLLOW_ANY,
        220 => args[4] & FSOPT_NOFOLLOW_ANY,
        _ => 0,
    };
    plain == 0 && !nofollow_any(n, args)
}

#[cfg(test)]
mod dotdot_tests {
    use super::*;

    // A guest tree: /System/Library/dyld and /t/real/sub are directories, /t/lnk -> real/sub,
    // /t/abs -> /System/Library/dyld, /t/up -> ../t/real, /t/loop -> loop, /t/empty -> "".
    fn tree(p: &str) -> Node {
        match p {
            "/System" | "/System/Library" | "/System/Library/dyld" | "/t" | "/t/real" | "/t/real/sub" => Node::Dir,
            "/t/lnk" => Node::Link("real/sub".into()),
            "/t/abs" => Node::Link("/System/Library/dyld".into()),
            "/t/up" => Node::Link("../t/real".into()),
            "/t/loop" => Node::Link("loop".into()),
            "/t/empty" => Node::Link(String::new()),
            "/t/two" => Node::Link("lnk".into()),
            _ => Node::Other,
        }
    }
    /// the walk's lexical answer; a Kernel answer is spelled "kernel:<name>" so one assert covers both
    fn w(p: &str, node: impl Fn(&str) -> Node) -> Option<String> {
        match walk_dotdot(p, node) {
            Walk::Lexical(x) => x,
            Walk::Kernel(k) => Some(format!("kernel:{k}")),
        }
    }
    fn n(p: &str) -> Option<String> {
        w(p, tree)
    }

    #[test]
    fn dotdot_through_directories_is_popped() {
        // Rows Y7, Y8: out of the cache dir and back in by its basename
        assert_eq!(n("/System/Library/dyld/../dyld/x").as_deref(), Some("/System/Library/dyld/x"));
        assert_eq!(n("/System/Library/dyld/../../Library/dyld/x").as_deref(), Some("/System/Library/dyld/x"));
        assert_eq!(n("/System/./Library//dyld/../dyld/x").as_deref(), Some("/System/Library/dyld/x"));
        // `..` at / is /
        assert_eq!(n("/../System/Library/dyld/x").as_deref(), Some("/System/Library/dyld/x"));
        assert_eq!(n("/..").as_deref(), Some("/"));
        // a trailing `..` ends in `/.`, with or without a slash after it; `/x/` keeps its slash
        assert_eq!(n("/t/real/sub/../").as_deref(), Some("/t/real/."));
        assert_eq!(n("/t/real/sub/..").as_deref(), Some("/t/real/."));
        assert_eq!(n("/t/real/../real/sub/").as_deref(), Some("/t/real/sub/"));
    }

    #[test]
    fn no_dotdot_or_a_relative_path_is_left_alone() {
        for p in ["/System/Library/dyld/x", "/System/./Library/dyld/x", "/a..b/c", "/x/..y", "/t/lnk/x", "/", ""] {
            assert_eq!(n(p), None, "{p}");
        }
        assert_eq!(n("System/Library/dyld/../dyld/x"), None); // relative: no guest anchor
    }

    #[test]
    fn hv7_a_symlink_in_what_is_walked_goes_to_the_kernel() {
        // explicit: the link right before the `..`; H v6 followed it itself (/t/real/x)
        assert_eq!(n("/t/lnk/../x").as_deref(), Some("kernel:/t/lnk/../x"));
        assert_ne!(n("/t/lnk/../x").as_deref(), Some("/t/x"), "must-fail control: the lexical answer");
        assert_ne!(n("/t/lnk/../x").as_deref(), Some("/t/real/x"), "must-fail control: H v6's own walk");
        // the directories walked before the link keep their `..`; the rest goes as written, from the link on
        assert_eq!(n("/t/real/../lnk/../x").as_deref(), Some("kernel:/t/lnk/../x"));
        assert_eq!(n("/t/real/sub/../../abs/../dyld/x").as_deref(), Some("kernel:/t/abs/../dyld/x"));
        // implicit: the link further back, which the host's lstat expands
        assert_eq!(n("/t/real/../lnk/q/../f").as_deref(), Some("kernel:/t/lnk/q/../f"));
        // The `.` is the kernel's to see, not dropped before it measures the link
        assert_eq!(n("/t/lnk/q/./../f").as_deref(), Some("kernel:/t/lnk/q/./../f"));
        assert_ne!(n("/t/lnk/q/./../f").as_deref(), Some("kernel:/t/lnk/q/../f"), "must-fail control: the `.` dropped");
        // a link before a `.` only: the kernel too
        assert_eq!(n("/t/lnk/./x/../y").as_deref(), Some("kernel:/t/lnk/./x/../y"));
        // a loop, an empty target, a link to a link: the kernel answers (ELOOP, ENOENT) or resolves
        assert_eq!(n("/t/loop/../x").as_deref(), Some("kernel:/t/loop/../x"));
        assert_eq!(n("/t/empty/../x").as_deref(), Some("kernel:/t/empty/../x"));
        assert_eq!(n("/t/two/../x").as_deref(), Some("kernel:/t/two/../x"));
        // a link AFTER the last `..` is the host's to walk, as before (no `.` or `..` looks through it)
        assert_eq!(n("/t/real/sub/../../lnk").as_deref(), Some("/t/lnk"));
        assert_eq!(n("/t/real/../lnk/x").as_deref(), Some("/t/lnk/x"));
    }

    #[test]
    fn what_xnu_refuses_is_left_for_the_host_to_refuse() {
        // a missing component, or a file, before `..` (XNU: ENOENT, ENOTDIR)
        assert_eq!(n("/System/Library/dyld/no-such/../x"), None);
        assert_eq!(n("/System/Library/dyld/x/../x"), None);
    }

    #[test]
    fn a_refusal_keeps_the_dotdots_already_taken() {
        // The first `..` leaves the prefix's spelling alone, the second is refused
        assert_eq!(n("/System/Library/dyld/../dyld/x/../x").as_deref(), Some("/System/Library/dyld/x/../x"));
        assert_eq!(n("/System/Library/dyld/../dyld/no-such/../x").as_deref(), Some("/System/Library/dyld/no-such/../x"));
        // the rest comes through as written, its trailing slash too
        assert_eq!(n("/t/real/../no-such/../x/../y/").as_deref(), Some("/t/no-such/../x/../y/"));
    }

    #[test]
    fn a_trailing_dot_is_kept() {
        // file/. is ENOTDIR where file opens
        assert_eq!(n("/System/Library/dyld/../dyld/x/.").as_deref(), Some("/System/Library/dyld/x/."));
        assert_eq!(n("/t/real/sub/../.").as_deref(), Some("/t/real/."));
        assert_eq!(n("/t/real/./sub/../y").as_deref(), Some("/t/real/y"), "must-fail control: a . inside still goes");
        assert_eq!(n("/t/real/../..").as_deref(), Some("/"));
        assert_eq!(n("/t/..").as_deref(), Some("/"));
    }

    #[test]
    fn hv4_a_dotdot_through_an_unsearchable_dir_is_left_for_the_host() {
        // node() is Other for a directory the caller cannot search, so the `..`
        // is not popped; the host gets the path and answers EACCES. dir_searchable, which makes that
        // decision from the real mode and euid, is exercised by host_node_reads_the_host below.
        let t2 = |p: &str| match p {
            "/g" | "/g/ok_x" | "/g/ok_x/f" => Node::Dir,
            "/g/no_x" => Node::Other, // a directory the caller cannot search
            _ => Node::Other,
        };
        assert_eq!(w("/g/no_x/../f", t2), None); // unchanged -> host -> EACCES
        assert_eq!(w("/g/ok_x/../f", t2).as_deref(), Some("/g/f")); // control: popped
    }

    #[test]
    fn hv4_a_dot_after_a_non_directory_is_left_for_the_host() {
        // /t/real/../f/./../f, f a file -- native ENOTDIR at f/. ; dropping the `.`
        // gave root's lenient f/../f. The `.` now stops where the preceding path is not a searchable dir.
        let t2 = |p: &str| match p {
            "/t" | "/t/real" | "/t/d" => Node::Dir,
            "/t/f" => Node::Other, // a file
            _ => Node::Other,
        };
        assert_eq!(w("/t/real/../f/./../f", t2).as_deref(), Some("/t/f/./../f"));
        assert_ne!(w("/t/real/../f/./../f", t2).as_deref(), Some("/t/f/../f"),
            "must-fail control: the `.`-dropped path that succeeds as root");
        // a `.` after a searchable directory is still a no-op
        assert_eq!(w("/t/real/../d/./f", t2).as_deref(), Some("/t/d/f"));
    }

    #[test]
    fn hv7_kernel_name_keeps_every_remainder_as_long() {
        // the last component becomes `.` components of the same length: an odd and an even one
        assert_eq!(kernel_name("/w/l/q/../f"), ("/w/l/q/../.".to_string(), "/f".to_string()));
        assert_eq!(kernel_name("/w/l/q/../ff"), ("/w/l/q/.././".to_string(), "/ff".to_string()));
        let k = "/w/lok/t/../../System/Library/dyld/dyld_shared_cache_arm64e";
        let (probe, after) = kernel_name(k);
        assert_eq!(probe.len(), k.len(), "every link's remainder is as long as in the real name");
        assert!(probe.starts_with("/w/lok/t/../../System/Library/dyld/./") && !probe.contains("dyld_shared"));
        assert_eq!(after, "/dyld_shared_cache_arm64e");
        // a name ending in `..`, `.` or `/` is asked about whole; where it lands takes `/.` or `/`
        assert_eq!(kernel_name("/w/l/sub/.."), ("/w/l/sub/..".to_string(), "/.".to_string()));
        assert_eq!(kernel_name("/w/l/../d/."), ("/w/l/../d/.".to_string(), "/.".to_string()));
        assert_eq!(kernel_name("/w/l/../d/"), ("/w/l/../d/".to_string(), "/".to_string()));
        assert_ne!(kernel_name("/w/l/q/../f").0, "/w/l/q/..", "must-fail control: the last component dropped, not padded");
    }

    #[test]
    fn hv7_the_kernel_decides_and_the_redirect_takes_only_a_prefix() {
        let tree2 = |p: &str| match p {
            "/w" | "/w/d" | "/c" => Node::Dir,
            "/w/l" => Node::Link("d".into()),
            _ => Node::Other,
        };
        let pre = |m: &str| m.starts_with("/c/");
        let never = |_: &str, _: Ids| -> Probe { panic!("the kernel must not be asked") };
        // no link: the walk's answer, the kernel never asked
        assert_eq!(resolve_dotdot("/w/d/../c/x", true, Ids::Effective, tree2, never, pre), Dotdot::Path("/w/c/x".into()));
        assert_eq!(resolve_dotdot("/w/x", true, Ids::Effective, tree2, never, pre), Dotdot::Keep);
        // NOFOLLOW_ANY with a link: the host, unasked (it answers ELOOP)
        assert_eq!(resolve_dotdot("/w/l/../f", false, Ids::Effective, tree2, never, pre), Dotdot::Host("/w/l/../f".into()));
        // the kernel refuses (ENAMETOOLONG, EACCES, ELOOP...): the host gets the name and refuses it the same way
        let refuse = |_: &str, _: Ids| Probe::Refused;
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Effective, tree2, refuse, pre), Dotdot::Host("/w/l/../f".into()));
        // it lands under a prefix: the redirect takes it
        let at_c = |p: &str, _: Ids| { assert_eq!(p, "/w/l/../."); Probe::At("/c".into()) };
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Effective, tree2, at_c, pre), Dotdot::Path("/c/f".into()));
        // it lands elsewhere, or cannot be placed: the host, never the redirect
        let at_w = |_: &str, _: Ids| Probe::At("/w".into());
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Effective, tree2, at_w, pre), Dotdot::Host("/w/l/../f".into()));
        let lost = |_: &str, _: Ids| Probe::Unplaced;
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Effective, tree2, lost, pre), Dotdot::Host("/w/l/../f".into()));
        // the credentials reach the kernel; a trailing `..` lands with `/.`, at / with no doubled slash
        let real_only = |_: &str, ids: Ids| if ids == Ids::Real { Probe::Refused } else { Probe::At("/c".into()) };
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Real, tree2, real_only, pre), Dotdot::Host("/w/l/../f".into()));
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Effective, tree2, real_only, pre), Dotdot::Path("/c/f".into()),
            "must-fail control: the effective ids");
        assert_eq!(resolve_dotdot("/w/l/x/..", true, Ids::Effective, tree2, at_c_whole, pre), Dotdot::Path("/c/sub/.".into()));
        let at_root = |_: &str, _: Ids| Probe::At("/".into());
        assert_eq!(resolve_dotdot("/w/l/../f", true, Ids::Effective, tree2, at_root, |m: &str| m == "/f"), Dotdot::Path("/f".into()));
    }

    fn at_c_whole(p: &str, _: Ids) -> Probe {
        assert_eq!(p, "/w/l/x/..");
        Probe::At("/c/sub".into())
    }

    #[test]
    fn hv7_the_kernel_lookup_on_the_host() {
        // On the host's own filesystem through kernel_lookup, measured
        // natively first: l15/q/./../f is ENAMETOOLONG and l14 opens; a link whose
        // expansion fits natively (lok) lands back in a stand-in cache dir, one byte longer (lno) is refused.
        let base = std::env::temp_dir().join(format!("maclator-hv7-test-{}", std::process::id()));
        let a40 = "a".repeat(40);
        std::fs::create_dir_all(base.join("w/d/q")).unwrap();
        std::fs::create_dir_all(base.join(format!("w/d/{a40}"))).unwrap();
        std::fs::create_dir_all(base.join("c")).unwrap();
        std::fs::write(base.join("w/d/f"), b"x").unwrap();
        std::fs::write(base.join("c/cache"), b"x").unwrap();
        std::os::unix::fs::symlink(&a40, base.join("w/d/t")).unwrap();
        let mk = |l: usize| format!("{}{}", "./".repeat(if l % 2 == 1 { (l - 1) / 2 } else { (l - 2) / 2 }), if l % 2 == 1 { "d" } else { "/d" });
        std::os::unix::fs::symlink(mk(1015), base.join("w/l15")).unwrap();
        std::os::unix::fs::symlink(mk(1014), base.join("w/l14")).unwrap();
        let b = std::fs::canonicalize(&base).unwrap().to_string_lossy().into_owned();
        let none = |_: &str| false;
        // Refused with the `.` kept, which an earlier version dropped
        let p15 = format!("{b}/w/l15/q/./../f");
        assert_eq!(kernel_lookup(&kernel_name(&p15).0, Ids::Effective), Probe::Refused);
        assert_eq!(resolve_dotdot(&p15, true, Ids::Effective, guest_node, kernel_lookup, none), Dotdot::Host(p15.clone()));
        let p14 = format!("{b}/w/l14/q/./../f");
        assert_eq!(kernel_lookup(&kernel_name(&p14).0, Ids::Effective), Probe::At(format!("{b}/w/d")), "control: 1014 bytes");
        assert_eq!(resolve_dotdot(&p14, true, Ids::Effective, guest_node, kernel_lookup, |m: &str| m == format!("{b}/w/d/f")),
                   Dotdot::Path(format!("{b}/w/d/f")));
        // rest climbs from w/d/<a40> to / and back into c; lok fits natively, lno does not
        let depth = b.matches('/').count() + 3;
        let rest = format!("{}{}/c/cache", "../".repeat(depth), &b[1..]);
        let lok = 1020 - rest.len();
        std::os::unix::fs::symlink(mk(lok), base.join("w/lok")).unwrap();
        std::os::unix::fs::symlink(mk(lok + 1), base.join("w/lno")).unwrap();
        let c = |m: &str| m.starts_with(&format!("{b}/c/"));
        let pok = format!("{b}/w/lok/t/{rest}");
        assert_eq!(resolve_dotdot(&pok, true, Ids::Effective, guest_node, kernel_lookup, c), Dotdot::Path(format!("{b}/c/cache")));
        let pno = format!("{b}/w/lno/t/{rest}");
        assert_eq!(resolve_dotdot(&pno, true, Ids::Effective, guest_node, kernel_lookup, c), Dotdot::Host(pno.clone()),
            "must-fail control: one byte more is ENAMETOOLONG");
        for l in ["w/l15", "w/l14", "w/lok", "w/lno", "w/d/t", "w/d/f", "c/cache"] {
            std::fs::remove_file(base.join(l)).unwrap();
        }
        for d in [format!("w/d/{a40}"), "w/d/q".into(), "w/d".into(), "w".into(), "c".into()] {
            std::fs::remove_dir(base.join(d)).unwrap();
        }
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn hv7_full_path_of_spells_the_hosts_path() {
        // getattrlist's FULLPATH as F_GETPATH spells it (/private/var for /var); a missing name gives None
        let base = std::env::temp_dir().join(format!("maclator-hv7-fp-{}", std::process::id()));
        std::fs::create_dir_all(base.join("d")).unwrap();
        std::os::unix::fs::symlink("d", base.join("l")).unwrap();
        let canon = std::fs::canonicalize(base.join("d")).unwrap().to_string_lossy().into_owned();
        let cs = |p: std::path::PathBuf| std::ffi::CString::new(p.to_string_lossy().into_owned()).unwrap();
        assert_eq!(full_path_of(&cs(base.join("l"))).as_deref(), Some(canon.as_str()), "a link is followed");
        assert_eq!(full_path_of(&cs(base.join("d/."))).as_deref(), Some(canon.as_str()));
        assert_eq!(full_path_of(&cs(base.join("missing"))), None, "must-fail control: nothing there");
        std::fs::remove_file(base.join("l")).unwrap();
        std::fs::remove_dir(base.join("d")).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn hv5_a_descriptor_records_the_hosts_truth_outside_the_sysroot() {
        // Inside the sysroot the guest path; elsewhere F_GETPATH; F_GETPATH failing: nothing
        assert_eq!(descriptor_record("/System/Library/dyld/.", true, || Some("/sys".into())).as_deref(), Some("/System/Library/dyld"));
        assert_eq!(descriptor_record("/System/Volumes/Data/.", false, || Some("/".into())).as_deref(), Some("/"),
            "must-fail control: round 4 recorded the lexical /System/Volumes/Data for a redirected climb-out");
        assert_eq!(descriptor_record("/System/Volumes/Data/.", false, || None), None,
            "must-fail control: round 4 fell back to the lexical path when F_GETPATH failed");
        assert!(inside_sysroot("/sys", Some("/sys")) && inside_sysroot("/sys/x", Some("/sys")));
        assert!(!inside_sysroot("/sysx", Some("/sys")), "must-fail control: a sibling that shares the prefix string");
        assert!(!inside_sysroot("/System/Volumes/Data/.", Some("/sys")) && !inside_sysroot("/sys", None));
    }

    #[test]
    fn hv5_host_fd_path_reads_the_host() {
        // A unit test that reaches host_fd_path. F_GETPATH gives the host's own spelling
        // (/private/var for /var); a bad descriptor gives None, which descriptor_record records as nothing.
        use std::os::unix::io::AsRawFd;
        let base = std::env::temp_dir().join(format!("maclator-fdpath-test-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let f = std::fs::File::open(&base).unwrap();
        let canon = std::fs::canonicalize(&base).unwrap().to_string_lossy().into_owned();
        assert_eq!(host_fd_path(f.as_raw_fd()).as_deref(), Some(canon.as_str()));
        assert_eq!(host_fd_path(-1), None, "must-fail control: no descriptor");
        drop(f);
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn hv5_a_firmlinked_directory_is_not_popped() {
        // On macOS /System/Volumes/Data/Applications is firmlinked to /Applications, so its
        // `..` is /. It must not be popped to /System/Volumes/Data. Skipped where that firmlink is absent.
        let a = "/System/Volumes/Data/Applications";
        let root = std::fs::metadata("/").unwrap();
        let up = std::fs::metadata(format!("{a}/.."));
        use std::os::unix::fs::MetadataExt;
        if up.is_ok_and(|m| m.ino() == root.ino() && m.dev() == root.dev()) {
            assert!(!dotdot_is_lexical(a));
            assert_eq!(host_node(a), Node::Other);
        }
        // control: an ordinary directory's `..` is its lexical parent
        let base = std::env::temp_dir().join(format!("maclator-firm-test-{}", std::process::id()));
        std::fs::create_dir_all(base.join("d")).unwrap();
        let d = base.join("d").to_string_lossy().into_owned();
        assert!(dotdot_is_lexical(&d), "must-fail control: an ordinary directory");
        assert_eq!(host_node(&d), Node::Dir);
        std::fs::remove_dir(base.join("d")).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn hv6_the_lookup_uses_the_calls_credentials() {
        // access, and faccessat without AT_EACCESS, look up
        // with the real ids; everything else with the effective ids.
        let a = |i: usize, v: u64| {
            let mut x = [0u64; 8];
            x[i] = v;
            x
        };
        assert_eq!(lookup_ids(33, &a(1, 0)), Ids::Real);
        assert_eq!(lookup_ids(466, &a(3, 0)), Ids::Real);
        assert_eq!(lookup_ids(466, &a(3, AT_EACCESS)), Ids::Effective);
        assert_eq!(lookup_ids(466, &a(3, AT_EACCESS | AT_SYMLINK_NOFOLLOW_ANY)), Ids::Effective);
        assert_eq!(lookup_ids(466, &a(3, 0x0020)), Ids::Real, "must-fail control: AT_SYMLINK_NOFOLLOW is not AT_EACCESS");
        assert_eq!(lookup_ids(466, &a(2, AT_EACCESS)), Ids::Real, "must-fail control: the mode slot, not the flags");
        for n in [5, 398, 58, 188, 190, 220, 338, 340, 463, 464, 469, 470, 473, 476] {
            assert_eq!(lookup_ids(n, &a(3, 0)), Ids::Effective, "#{n}");
        }
        // As non-root the real and effective ids are the same: a 0600 directory is unsearchable both ways.
        // Root with a real uid of 501 is a row (H34-H37 of mcl-atpaths 1.9.0), not a unit test.
        if unsafe { libc::geteuid() } != 0 && unsafe { libc::getuid() } == unsafe { libc::geteuid() } {
            use std::os::unix::fs::PermissionsExt;
            let base = std::env::temp_dir().join(format!("maclator-hv6-ids-{}", std::process::id()));
            std::fs::create_dir_all(base.join("nx")).unwrap();
            std::fs::create_dir_all(base.join("ok")).unwrap();
            std::fs::set_permissions(base.join("nx"), std::fs::Permissions::from_mode(0o600)).unwrap();
            let p = |s: &str| base.join(s).to_string_lossy().into_owned();
            for ids in [Ids::Real, Ids::Effective] {
                assert!(!dir_searchable(&p("nx"), ids), "{ids:?}");
                assert!(dir_searchable(&p("ok"), ids), "control {ids:?}");
            }
            std::fs::remove_dir(base.join("nx")).unwrap();
            std::fs::remove_dir(base.join("ok")).unwrap();
            std::fs::remove_dir(&base).unwrap();
        }
    }

    #[test]
    fn a_name_too_long_for_xnu_is_not_rewritten() {
        // 1023 bytes looks up, 1024 is ENAMETOOLONG
        assert!(rewritable(&"a".repeat(MAXPATHLEN - 1)));
        assert!(!rewritable(&"a".repeat(MAXPATHLEN)), "must-fail control: the 1024-byte name");
        assert!(!rewritable(&"a".repeat(GUEST_NAME_MAX - 1)));
    }

    #[test]
    fn a_descriptor_records_the_path_without_the_marker() {
        // The `/.` and `/` walk_dotdot keeps for the host are not part of the guest path
        assert_eq!(recorded_path("/System/Library/dyld/."), "/System/Library/dyld");
        assert_eq!(recorded_path("/System/Library/dyld/"), "/System/Library/dyld");
        assert_eq!(recorded_path("/a/./"), "/a");
        assert_eq!(recorded_path("/."), "/");
        assert_eq!(recorded_path("/"), "/");
        assert_eq!(recorded_path("/a/b.c"), "/a/b.c", "must-fail control: a dot inside a name stays");
        assert_eq!(recorded_path("/a/.."), "/a/..", "must-fail control: only `.` is a marker");
    }

    #[test]
    fn a_path_that_climbed_out_goes_to_the_host_normalised() {
        let redirect = |p: &str| p.starts_with("/System/Library/dyld").then(|| format!("/sys{p}"));
        assert_eq!(left_the_prefix("/System/Library/dyld/../Frameworks/x", "/System/Library/Frameworks/x", redirect).as_deref(),
                   Some("/System/Library/Frameworks/x"));
        // must-fail controls: never under a prefix; nothing changed
        assert_eq!(left_the_prefix("/tmp/a/../b", "/tmp/b", redirect), None);
        assert_eq!(left_the_prefix("/System/Library/dyld/x", "/System/Library/dyld/x", redirect), None);
    }

    #[test]
    fn follows_symlinks_reads_every_nofollow_any_slot() {
        let a = |i: usize, v: u64| {
            let mut x = [0u64; 8];
            x[i] = v;
            x
        };
        assert!(follows_symlinks(463, &a(2, 0)));
        assert!(!follows_symlinks(463, &a(2, O_NOFOLLOW_ANY)));
        assert!(!follows_symlinks(470, &a(3, AT_SYMLINK_NOFOLLOW_ANY)));
        for k in [5, 398] {
            assert!(follows_symlinks(k, &a(1, 0x0100))); // O_NOFOLLOW: the last component only
            assert!(!follows_symlinks(k, &a(1, O_NOFOLLOW_ANY)));
            assert!(follows_symlinks(k, &a(2, O_NOFOLLOW_ANY))); // must-fail control: wrong slot
        }
        assert!(!follows_symlinks(220, &a(4, FSOPT_NOFOLLOW_ANY)));
        assert!(follows_symlinks(220, &a(3, FSOPT_NOFOLLOW_ANY)));
        assert!(follows_symlinks(188, &[u64::MAX; 8])); // stat has no flags
    }

    #[test]
    fn host_node_reads_the_host() {
        let base = std::env::temp_dir().join(format!("maclator-h-test-{}", std::process::id()));
        std::fs::create_dir_all(base.join("d")).unwrap();
        std::fs::write(base.join("f"), b"x").unwrap();
        std::os::unix::fs::symlink("d", base.join("l")).unwrap();
        let p = |s: &str| base.join(s).to_string_lossy().into_owned();
        assert_eq!(host_node(&p("d")), Node::Dir);
        assert_eq!(host_node(&p("l")), Node::Link("d".into())); // lstat: the link, not its target
        assert_eq!(host_node(&p("f")), Node::Other);
        assert_eq!(host_node(&p("missing")), Node::Other);
        // guest_node of a path no redirect touches is the host's
        assert_eq!(guest_node(&p("l")), Node::Link("d".into()));
        assert!(!crate::paths::is_cache_root(&p("d")));
        // A directory the caller cannot search is Other, not Dir. Root's directory-
        // search bypass makes every directory searchable, so only assert this as a non-root caller.
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            std::fs::create_dir(base.join("nx")).unwrap();
            std::fs::set_permissions(base.join("nx"), std::fs::Permissions::from_mode(0o600)).unwrap();
            assert_eq!(host_node(&p("nx")), Node::Other);
            std::fs::remove_dir(base.join("nx")).unwrap();
        }
        std::fs::remove_file(base.join("l")).unwrap();
        std::fs::remove_file(base.join("f")).unwrap();
        std::fs::remove_dir(base.join("d")).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }
}

// Flags that make an *at call's descriptor itself load-bearing (values from the macOS SDK's
// sys/fcntl.h and sys/attr.h).
const O_RESOLVE_BENEATH: u64 = 0x1000;
const AT_FDONLY: u64 = 0x0400;
const AT_RESOLVE_BENEATH: u64 = 0x2000;
const FSOPT_RESOLVE_BENEATH: u64 = 0x1000;

/// Whether *at call `n`'s flags make its descriptor load-bearing, so that it must not be replaced
/// by AT_FDCWD and an absolute path: fstatat's AT_FDONLY uses only the descriptor and ignores the
/// name, and the RESOLVE_BENEATH flags keep the lookup beneath the descriptor. Rewritten, those
/// calls answered EBADF and ENOTCAPABLE where XNU succeeds; left alone, the host kernel answers
/// as XNU does, through the host descriptor (for a redirected directory, the sysroot's).
fn descriptor_bound(n: u64, args: &[u64; 8]) -> bool {
    let bits = match n {
        463 | 464 => args[2] & O_RESOLVE_BENEATH,
        466 => args[3] & AT_RESOLVE_BENEATH,
        469 | 470 => args[3] & (AT_FDONLY | AT_RESOLVE_BENEATH),
        476 => args[5] & FSOPT_RESOLVE_BENEATH,
        _ => 0,
    };
    bits != 0
}

/// Whether call `n` carries
/// a RESOLVE_BENEATH flag (its lookup may not climb above where it starts), for the calls H rewrites (open's and
/// openat's O_RESOLVE_BENEATH, faccessat's, fstatat's and fstatat64's AT_RESOLVE_BENEATH, getattrlist's and
/// getattrlistat's FSOPT_RESOLVE_BENEATH) and the ones escape_name rewrites (fchmodat, fchownat, linkat, unlinkat:
/// AT_RESOLVE_BENEATH; renameatx_np: RENAME_RESOLVE_BENEATH; clonefileat: CLONE_RESOLVE_BENEATH; setattrlistat:
/// FSOPT_RESOLVE_BENEATH; getxattr, setxattr, removexattr, listxattr: XATTR_RESOLVE_BENEATH; values from the 27.0 SDK).
/// Not AT_FDONLY (descriptor_bound's other flag): such a name is never read (name_unread).
fn beneath(n: u64, args: &[u64; 8]) -> bool {
    let bits = match n {
        5 | 398 => args[1] & O_RESOLVE_BENEATH,
        463 | 464 => args[2] & O_RESOLVE_BENEATH,
        466 | 469 | 470 | 467 => args[3] & AT_RESOLVE_BENEATH,
        468 | 471 => args[4] & AT_RESOLVE_BENEATH,
        472 => args[2] & AT_RESOLVE_BENEATH,
        220 => args[4] & FSOPT_RESOLVE_BENEATH,
        476 | 524 => args[5] & FSOPT_RESOLVE_BENEATH,
        488 => args[4] & 0x20,
        462 => args[4] & 0x10,
        234 | 236 => args[5] & 0x80,
        238 => args[2] & 0x80,
        240 => args[3] & 0x80,
        _ => 0,
    };
    bits != 0
}

// The NOFOLLOW_ANY flags (same source): no symlink may be crossed. XNU checks only the components
// it actually traverses from its starting point, so through a descriptor they depend on it too.
// AT_SYMLINK_NOFOLLOW_ANY and FSOPT_NOFOLLOW_ANY share 0x0800 (O_EXCL in openat's oflag).
const O_NOFOLLOW_ANY: u64 = 0x2000_0000;
const AT_SYMLINK_NOFOLLOW_ANY: u64 = 0x0800;
const FSOPT_NOFOLLOW_ANY: u64 = 0x0800;

/// Whether *at call `n` carries a NOFOLLOW_ANY flag: openat's O_NOFOLLOW_ANY, faccessat's,
/// fstatat's or fstatat64's AT_SYMLINK_NOFOLLOW_ANY, getattrlistat's FSOPT_NOFOLLOW_ANY.
fn nofollow_any(n: u64, args: &[u64; 8]) -> bool {
    let bits = match n {
        463 | 464 => args[2] & O_NOFOLLOW_ANY,
        466 | 469 | 470 => args[3] & AT_SYMLINK_NOFOLLOW_ANY,
        476 => args[5] & FSOPT_NOFOLLOW_ANY,
        _ => 0,
    };
    bits != 0
}

/// Whether an *at call must keep its own descriptor and name (no rewrite to AT_FDCWD and an
/// absolute path). Always when descriptor_bound. With a NOFOLLOW_ANY flag, only when `inside()`
/// (see `nofollow_any_inside`): the host descriptor is the redirected directory and the name never
/// leaves it. Rewritten, the absolute path walks every component of the sysroot's path, and a
/// symlink there (a sysroot reached through one; on macOS /var and /tmp are symlinks) gave ELOOP
/// where XNU succeeds. Anything else is left to the rewrite. `inside` is asked only for a
/// NOFOLLOW_ANY call that descriptor_bound does not already keep.
fn keep_descriptor(n: u64, args: &[u64; 8], inside: impl FnOnce() -> bool) -> bool {
    descriptor_bound(n, args) || (nofollow_any(n, args) && inside())
}

/// Whether relative name `name` stays beneath its starting directory: no `..` component ever
/// climbs above it ("a/../b" stays, "../x" and "a/../../x" do not). Lexical; a symlink among the
/// components is the host kernel's to judge, and under NOFOLLOW_ANY it refuses one.
fn stays_beneath(name: &str) -> bool {
    let mut depth = 0i64;
    for c in name.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => depth += 1,
        }
    }
    true
}

/// Whether host descriptor `fd` is the directory at host path `host` (same device and inode;
/// `host` is followed through symlinks, as the descriptor's open was). False when either is
/// unreadable (None). Returns that (device, inode). The descriptor is read once, and the
/// pair compared is the pair returned: read twice, a guest thread's dup2 between the reads let the
/// check pass on one directory and act on the other.
fn same_host_dir(fd: i32, host: &str) -> Option<(u64, u64)> {
    let k = fd_dev_ino(fd)?;
    let same = path_dev_ino(host) == Some(k);
    same.then_some(k)
}

/// The (device, inode) host descriptor `fd` is on; None when fstat fails.
fn fd_dev_ino(fd: i32) -> Option<(u64, u64)> {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    (unsafe { libc::fstat(fd, &mut st) } == 0).then_some((st.st_dev as u64, st.st_ino as u64))
}

/// The (device, inode) at host path `p`, followed through symlinks; None when stat fails.
fn path_dev_ino(p: &str) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).ok().map(|m| (m.dev(), m.ino()))
}

/// The host's own path for descriptor `fd` (F_GETPATH), unmapped; None when it cannot be read.
fn fd_host_getpath(fd: i32) -> Option<String> {
    let mut path = [0u8; 1024];
    if fd < 0 || unsafe { libc::fcntl(fd, libc::F_GETPATH, path.as_mut_ptr()) } != 0 {
        return None;
    }
    let end = path.iter().position(|&c| c == 0)?;
    std::str::from_utf8(&path[..end]).ok().map(str::to_string)
}

/// Where a kept NOFOLLOW_ANY call goes. Not
/// through the guest's descriptor number (a guest thread's dup2 after the check would move the
/// call; E2 v2's race) and not through a pin: a pin is a descriptor of the
/// directory, and closing any descriptor of a vnode releases the process's POSIX locks on it, so a
/// pin could never be closed, a full table sent the call to the absolute rewrite (ELOOP through a
/// symlinked sysroot path), a duplicate kept the guest's flock locks alive past its own last
/// close, and every pin took a number from the guest's descriptor table. Instead the call is
/// rewritten to the directory's canonical host path, F_GETPATH of the checked descriptor, which
/// has no symlink in it (so NOFOLLOW_ANY walks it as XNU walks the descriptor: measured on macOS 27.0.1, arm64,
/// /var/... gives ELOOP and its F_GETPATH spelling /private/var/... opens), then `name`. The path
/// is accepted only when it is still the directory the check saw (`dir`, from the check's single
/// fstat): a dup2 between the check and F_GETPATH gives another directory's path, and the caller
/// checks again. None when F_GETPATH or the stat fails or the directory differs. No descriptor is
/// made, so nothing is ever closed, nothing is shared, and nothing is held.
fn canonical_dir(fd: i32, dir: (u64, u64), getpath: impl FnOnce(i32) -> Option<String>, stat: impl FnOnce(&str) -> Option<(u64, u64)>) -> Option<String> {
    let p = getpath(fd)?;
    (stat(&p) == Some(dir)).then_some(p)
}

/// How many times a kept NOFOLLOW_ANY call re-checks when the descriptor's path is not the
/// directory just checked (a guest thread's dup2 or rename in between) before it goes to the
/// rewrite, as a call that fails the check does.
const CANONICAL_TRIES: usize = 3;

/// The `inside` test for a NOFOLLOW_ANY call through descriptor `fd` with name `name`, whose
/// recorded guest path is `guest` (fd_guest_path). All three must hold:
///  - the name is relative and stays beneath the descriptor (stays_beneath). A `..` that climbs
///    out walks the HOST parent of the sysroot from a kept descriptor, not the guest parent;
///    the rewrite is then left to decide;
///  - the guest path is redirected, to host path `h`;
///  - the descriptor really is `h` (same_host_dir). The guest path alone does not say so: a
///    descriptor opened by a call that was itself kept (openat(rootfd, "System/Library/dyld",
///    O_RESOLVE_BENEATH)) is recorded with the redirected guest path while it sits on the
///    host's real directory.
///
/// Returns the directory, as (device, inode), when all three hold. The lexical
/// tests run first and the redirect next, so a call that fails them costs no system call.
fn nofollow_any_inside(name: &str, fd: i32, guest: Option<String>, redirect: impl FnOnce(&str) -> Option<String>) -> Option<((u64, u64), String)> {
    if name.starts_with('/') || !stays_beneath(name) {
        return None;
    }
    let h = guest.and_then(|d| redirect(&d))?;
    same_host_dir(fd, &h).map(|k| (k, h))
}

/// Where a kept NOFOLLOW_ANY call that passed the check goes.
#[derive(Debug, PartialEq, Eq)]
enum Kept {
    /// To `path`, the checked directory's canonical host path and the name, at AT_FDCWD.
    Path { path: String, redirected: String },
    /// Through the guest's own descriptor number, as it came: `path` would reach MAXPATHLEN.
    Number { redirected: String },
}

/// The route for a kept call through a
/// directory whose canonical host path (F_GETPATH) is `canonical` and whose redirected path (the
/// guest's record through paths::redirect, which the check found to be the same directory) is
/// `redirected`, for relative name `name`. The joined path `canonical/name` can reach
/// MAXPATHLEN where the name alone, through the descriptor, does not (natively XNU limits the name a
/// call passes, not the directory's own path); the host would answer ENAMETOOLONG, as the absolute
/// rewrite does. Then the call goes through the guest's number as it came, which E2 v2 did for every
/// such call: right unless a guest thread's dup2 moves the number between the check and the call
/// (Inferred; E2 v2's race, now only for a joined path of 1,024 bytes or more). The
/// place is spelled by the redirect (`redirected/name`) for inside_sysroot, whose prefix is the
/// sysroot as canonicalised when it was set (realpath); F_GETPATH spells some of the same directories
/// otherwise (measured on macOS 27.0.1, arm64: a sysroot named under /System/Volumes/Data/private/var keeps that
/// spelling in realpath while F_GETPATH answers /private/var; case and Unicode variants came back
/// alike from both), so the canonical path's byte prefix can miss the sysroot.
fn kept_route(canonical: &str, redirected: &str, name: &str) -> Kept {
    let redirected = format!("{redirected}/{name}");
    if canonical.len() + 1 + name.len() >= MAXPATHLEN {
        Kept::Number { redirected }
    } else {
        Kept::Path { path: format!("{canonical}/{name}"), redirected }
    }
}

fn bsd(cpu: &mut Cpu, n: u64, args: [u64; 8]) {
    if let Some(warning) = bsd_read(cpu, n, args, guest_name) {
        eprintln!("{warning}");
    }
}

/// bsd() with its name reader a parameter, so a test can give it a memory that refuses a name the
/// kernel can read. Returns unread_name_warning's answer for this call.
fn bsd_read(cpu: &mut Cpu, n: u64, mut args: [u64; 8], read: impl FnOnce(u64) -> Option<String>) -> Option<String> {
    let (pidx, at) = path_arg(n);
    let mut keep: Option<std::ffi::CString> = None;
    let mut guest_path: Option<String> = None;
    let mut in_sysroot = false;
    let mut unread: Option<u64> = None;
    let mut canon: Option<Kept> = None;
    // The record of a working directory inside the sysroot that a relative name was left to the host
    // from (the name stays inside), for what the call records afterwards; and the names escape_unhandled rewrote.
    let mut left_from: Option<CwdRec> = None;
    // The name the host was given, for a chdir's record (bsd_read's chdir branch).
    let mut sent: Option<String> = None;
    let mut esc_keep: Vec<std::ffi::CString> = Vec::new();
    let mut ax_keep: Vec<u8> = Vec::new();
    // The prefix of the record in force before a chdir-like call, kept for a new
    // directory inside the sysroot that names none (getcwd answers the sysroot's host path, so a guest
    // that saves and restores its working directory by path comes back by that path).
    let prev_prefix = if matches!(n, 12 | 13 | 348 | 349) { cwd_slot().map(|r| r.prefix) } else { None };
    if pidx.is_none() {
        let r = if n == 284 { escape_accessx(&mut args, &mut ax_keep) } else { escape_unhandled(n, &mut args, &mut esc_keep) };
        if let Err(e) = r {
            set_err(cpu, e);
            return None;
        }
    }
    if let Some(i) = pidx {
        let name = call_name(n, &args, i, |p| {
            let name = read(p);
            if name.is_none() {
                unread = Some(p);
            }
            name
        });
        if let Some(orig) = name.filter(|o| rewritable(o)) {
            // The base of a relative name. `recorded` says it is a descriptor's guest path
            // (fd_guest_path), the only base before H v8; the new bases are the working directory's and an
            // unrecorded descriptor's host path. A name joined to a new base that reaches MAXPATHLEN is not
            // rewritten: the host answers for the relative name, as it did before H v8.
            // From a working directory with a record (inside the sysroot,
            // reached through a prefix) the name goes to the host as it came, which walks it from the right
            // directory with XNU's own checks, unless a `..` in it climbs out of the sysroot (escape_at): then its
            // guest path (escaped_path) goes through H as an absolute name would. With no record, a new base's
            // result is used only when it lands under the redirect (`new_base`).
            let mut recorded = false;
            let mut new_base = false;
            let mut escaped: Option<CwdRec> = None;
            let mut probe_base: Option<String> = None;
            let fd = args[0] as i32;
            sent = Some(orig.clone());
            let rec = if !orig.starts_with('/') && !orig.is_empty() && cwd_is_base(at, fd) { cwd_now() } else { None };
            if rec.is_none() && cwd_is_base(at, fd) && !orig.starts_with('/') && crate::paths::cache_dir().is_some() && unrecorded_climb(&orig) {
                eprintln!("[maclator] warning: #{n} {orig} climbs out of the sysroot from a working directory with no record: refused (ENOENT)");
                set_err(cpu, libc::ENOENT as u64);
                return None;
            }
            // Moved up from below, so an escaped name can be refused before it is rewritten.
            let bound = beneath(n, &args);
            let full = if let Some(r) = rec {
                // The root taken at startup; unknown is refused, never "no climb".
                let Ok(esc) = escape_closed(&orig, sysroot_root(), path_dev_ino) else {
                    eprintln!("[maclator] warning: #{n} {orig}: the sysroot's root cannot be identified, so a climb out of it cannot be ruled out: refused (ENOENT)");
                    set_err(cpu, libc::ENOENT as u64);
                    return None;
                };
                match esc {
                    Some(off) if !follows_symlinks(n, &args) && lead_has_link(&orig[..off]) => {
                        // The leading part is dropped from the escaped name, and with it XNU's
                        // NOFOLLOW_ANY refusal of a link on it.
                        set_err(cpu, libc::ELOOP as u64);
                        return None;
                    }
                    Some(off) => {
                        // The dropped part, walked by the host with the call's credentials and
                        // the name's remaining length.
                        if let Err(e) = lead_walk(&orig, off, lookup_ids(n, &args)) {
                            set_err(cpu, e);
                            return None;
                        }
                        // A RESOLVE_BENEATH call whose name climbs out of the sysroot climbs above
                        // its working directory, which XNU refuses: never rewritten into a name that stays beneath.
                        if bound {
                            set_err(cpu, libc::ENOTCAPABLE as u64);
                            return None;
                        }
                        let f = escaped_path(&r, &orig, off);
                        escaped = Some(r);
                        Some(f)
                    }
                    None => {
                        guest_path = Some(vpath_join(&r.guest(), &orig));
                        in_sysroot = true;
                        left_from = Some(r);
                        None
                    }
                }
            } else {
                guest_full_path(&orig, at, || {
                    if cwd_is_base(at, fd) {
                        new_base = true;
                        probe_base = cwd_host_base(|| full_path_of(c"."), crate::paths::cache_dir().as_deref(), crate::paths::overlay_root().is_some());
                        probe_base.clone()
                    } else if let Some(g) = fd_guest_path(fd) {
                        recorded = true;
                        Some(g)
                    } else {
                        new_base = true;
                        unrecorded_fd_base(fd, fd_host_getpath, crate::paths::cache_dir().as_deref())
                    }
                })
            };
            let full = full.filter(|f| joined_rewritable(recorded || escaped.is_some(), &orig, f));
            if TRACE.load(Ordering::Relaxed) {
                if left_from.is_some() {
                    eprintln!("[path] #{n} {orig}: inside the sysroot from the working directory, left to the host");
                } else {
                    eprintln!("[path] #{n} {orig} -> {full:?}{}", if escaped.is_some() { " (climbed out of the sysroot)" } else { "" });
                }
            }
            if let Some(full) = full {
                // Resolve a `..` in the guest's view before the redirect sees the path.
                let before = full;
                let ids = lookup_ids(n, &args);
                // A symlink before a `.` or `..` is the kernel's to look up (resolve_dotdot); a name
                // it sends to the host goes as it is, never through the redirect.
                // A probe for a name joined to the working directory's host path looks it up from
                // the working directory, as the call itself does.
                let dotdot = resolve_dotdot(&before, follows_symlinks(n, &args), ids, |p| guest_node_as(p, ids),
                                            |p, ids| kernel_lookup(probe_from_cwd(p, probe_base.as_deref()), ids),
                                            |m| crate::paths::redirect(m).is_some());
                let (full, host_as_is) = match dotdot {
                    Dotdot::Keep => (before.clone(), false),
                    Dotdot::Path(p) => (p, false),
                    Dotdot::Host(p) => (p, true),
                };
                // A relative path beneath an overlaid directory may be absent
                // from the payload. In that case fall back to the real guest
                // absolute path, rather than looking only inside the overlay.
                // Never when the flags bind the call to its descriptor (keep_descriptor).
                // A NOFOLLOW_ANY call that passes the check goes to the checked
                // directory's canonical host path (canonical_dir), not through the guest's number except as kept_route says,
                // and never through a descriptor Maclator made.
                // kept_route decides between that path and
                // the guest's number (a joined path reaching MAXPATHLEN), and spells the call's place
                // by the redirect for inside_sysroot.
                let check = || {
                    let fd = args[0] as i32;
                    for _ in 0..CANONICAL_TRIES {
                        let (dir, h) = nofollow_any_inside(&orig, fd, fd_guest_path(fd), crate::paths::redirect)?;
                        if let Some(p) = canonical_dir(fd, dir, fd_host_getpath, path_dev_ino) {
                            return Some(kept_route(&p, &h, &orig));
                        }
                    }
                    None
                };
                let inside = || {
                    canon = check();
                    canon.is_some()
                };
                let mut spelled: Option<String> = None;
                let replacement = if at && keep_descriptor(n, &args, inside) {
                    match canon.take() {
                        Some(Kept::Path { path, redirected }) => {
                            if TRACE.load(Ordering::Relaxed) {
                                eprintln!("[canonical] #{n} {orig} -> {path}");
                            }
                            spelled = Some(redirected);
                            Some(path)
                        }
                        Some(Kept::Number { redirected }) => {
                            if TRACE.load(Ordering::Relaxed) {
                                eprintln!("[canonical] #{n} {orig}: kept on descriptor {} (the joined path reaches MAXPATHLEN)", args[0] as i32);
                            }
                            spelled = Some(redirected);
                            None
                        }
                        None => None,
                    }
                } else if host_as_is && new_base {
                    // The host's own lookup of the name as it came, from the base it holds.
                    None
                } else if host_as_is {
                    Some(full.clone())
                } else {
                    crate::paths::redirect(&full).or_else(|| {
                        // For a recorded descriptor's base only, as before H v8.
                        (at && recorded && !orig.starts_with('/') && crate::paths::overlay_root().is_some()).then(|| full.clone())
                    }).or_else(|| left_the_prefix(&before, &full, crate::paths::redirect))
                };
                // An escaped name whose host path reaches MAXPATHLEN goes relative to the
                // working directory when it lands inside the sysroot again; else the call is refused, loudly.
                // An escaped name is never sent as it came (the host would climb to the sysroot's host parent): with
                // no replacement from above it goes as its guest path, unless the call is bound to its descriptor
                // (RESOLVE_BENEATH: the host refuses the climb itself, as XNU does).
                let replacement = match (replacement, escaped.as_ref()) {
                    (None, Some(_)) if !bound => Some(full.clone()),
                    (x, _) => x,
                };
                let replacement = match (replacement, escaped.as_ref()) {
                    (Some(new), Some(_)) => match fit_target(new, crate::paths::cache_dir().as_deref(), here_depth) {
                        Ok(t) if still_climbs(&t) => {
                            eprintln!("[maclator] warning: #{n} {orig} climbs out of the sysroot to {before}, which could not be resolved: refused (ENOENT)");
                            set_err(cpu, libc::ENOENT as u64);
                            return None;
                        }
                        Ok(t) => Some(t),
                        Err(e) => {
                            eprintln!("[maclator] warning: #{n} {orig} climbs out of the sysroot to {before}, too long to send: refused (ENAMETOOLONG)");
                            set_err(cpu, e);
                            return None;
                        }
                    },
                    (x, _) => x,
                };
                if let Some(new) = replacement {
                    let ins = inside_sysroot(spelled.as_deref().unwrap_or(&new), crate::paths::cache_dir().as_deref())
                        || (escaped.is_some() && !new.starts_with('/'));
                    sent = Some(new.clone());
                    if let Ok(c) = std::ffi::CString::new(new) {
                        args[i] = c.as_ptr() as u64;
                        if at {
                            args[0] = (-2i64) as u64; // AT_FDCWD; path is absolute now
                        }
                        keep = Some(c);
                        in_sysroot = ins;
                    }
                } else if let Some(sp) = spelled {
                    in_sysroot = inside_sysroot(&sp, crate::paths::cache_dir().as_deref());
                }
                guest_path = Some(full);
            }
        }
    }
    bsd_inner(cpu, n, args);
    drop(keep);
    drop(esc_keep);
    drop(ax_keep);
    let warning = unread.and_then(|p| unread_name_warning(n, p, cpu.cf == 0));
    if cpu.cf == 0 && crate::paths::overlay_root().is_some() {
        if matches!(n, 92 | 406) && matches!(args[1], 50 | 102) {
            rewrite_fd_path(cpu, &args);
        } else if matches!(n, 220 | 228 | 476) {
            rewrite_fullpath_attribute(cpu, n, &args);
        }
    }
    if matches!(n, 5 | 398 | 463 | 464) && cpu.cf == 0 {
        // An untracked open can reuse a descriptor number previously populated
        // by dup2 or inherited files: do not leave a stale virtual path behind.
        if let Some(m) = FD_VPATHS.lock().unwrap().as_mut() {
            m.remove(&(cpu.x[0] as i32));
        }
        if let Some(g) = guest_path {
            // Record where the host really is, unless it is inside
            // the sysroot (descriptor_record). The FD_VPATHS entry was removed just above.
            let g = match left_from.as_ref() {
                // A name left to the host from inside the sysroot: the prefix, then where the host opened it.
                Some(r) => {
                    let h = fd_host_getpath(cpu.x[0] as i32);
                    let sys = sysroot_spelling();
                    match h.as_deref().and_then(|h| rec_at((0, 0), r.prefix, Some(h.to_string()), sys.as_deref())) {
                        Some(x) => Some(x.guest()),
                        None => h.map(|h| recorded_path(&crate::paths::guest_path(&h).unwrap_or(h))),
                    }
                }
                None => descriptor_record(&g, in_sysroot, || host_fd_path(cpu.x[0] as i32)),
            };
            if let Some(g) = g.filter(|g| g == "/" || g.starts_with("/System") || crate::paths::overlay_root().is_some()) {
                FD_VPATHS.lock().unwrap().get_or_insert_with(Default::default).insert(cpu.x[0] as i32, g);
            }
        }
    } else if cpu.cf == 0 && (matches!(n, 41 | 90) || (matches!(n, 92 | 406) && matches!(args[1], 0 | 67))) {
        let source = fd_guest_path(args[0] as i32);
        let target = if n == 90 { args[1] as i32 } else { cpu.x[0] as i32 };
        let mut paths = FD_VPATHS.lock().unwrap();
        let paths = paths.get_or_insert_with(Default::default);
        if let Some(path) = source {
            paths.insert(target, path);
        } else {
            paths.remove(&target);
        }
    } else if matches!(n, 12 | 13 | 348 | 349) && cpu.cf == 0 {
        // The new working directory's record (CwdRec): the prefix it was reached by (chdir_prefix: the
        // guest path H computed for chdir and __pthread_chdir, the descriptor's guest path for fchdir and
        // __pthread_fchdir) with its physical path below the sysroot, read from the host now; the process's for
        // chdir and fchdir, the thread's own for the per-thread forms, as XNU keeps them. __pthread_fchdir(-1)
        // gives the thread back the process's.
        let per_thread = matches!(n, 348 | 349);
        if n == 349 && args[0] as i32 == -1 {
            THREAD_CWD.with(|t| *t.borrow_mut() = None);
        } else {
            let sys = sysroot_spelling();
            let (prefix, at) = if matches!(n, 12 | 348) {
                (chdir_prefix(guest_path.as_deref(), left_from.as_ref()), None)
            } else {
                let fd = args[0] as i32;
                (fd_guest_path(fd).as_deref().and_then(crate::paths::physical_prefix), Some(fd))
            };
            // "." is the new directory unless a process chdir came from a thread with its own (then its fd or path)
            let r = prefix.or(prev_prefix).and_then(|p| {
                if let Some(fd) = at {
                    rec_at(fd_dev_ino(fd)?, p, fd_host_getpath(fd), sys.as_deref())
                } else if per_thread || !thread_has_cwd() {
                    rec_at(path_dev_ino(".")?, p, full_path_of(c"."), sys.as_deref())
                } else {
                    let h = sent.as_deref()?;
                    let c = std::ffi::CString::new(h).ok()?;
                    rec_at(path_dev_ino(h)?, p, full_path_of(&c), sys.as_deref())
                }
            });
            cwd_set(per_thread, r);
        }
    } else if matches!(n, 6 | 399) && cpu.cf == 0 {
        if let Some(m) = FD_VPATHS.lock().unwrap().as_mut() {
            m.remove(&(args[0] as i32));
        }
    }
    warning
}

fn bsd_inner(cpu: &mut Cpu, n: u64, mut args: [u64; 8]) {
    match n {
        crate::gpu::SYS_MCL_HOSTCALL => crate::gpu::hostcall(cpu),
        SYS_VFORK | SYS_FORK => {
            // vfork shares our address space and stack: use fork instead. Go through libc's
            // fork (not the raw syscall) so the host libSystem re-initialises in the child
            // (mach_task_self, MIG reply ports, ...); the JIT reads guest memory through
            // mach_vm_read with the cached task port.
            crate::engine::flush_profile();
            // With every translator lock held across it.
            let pid = crate::forksafe::fork();
            if pid < 0 {
                set_err(cpu, unsafe { *libc::__error() } as u64);
            } else if pid == 0 {
                // The child's working directory is the forking thread's (cwd_after_fork_child).
                cwd_after_fork_child();
                // Child: XNU returns the parent's pid with the "is child" flag in x1.
                set_ok(cpu, unsafe { libc::getppid() } as u64, 1);
            } else {
                set_ok(cpu, pid as u64, 0);
            }
        }
        SYS_EXIT => {
            crate::engine::flush_profile();
            let r = hostsys::unix(SYS_EXIT, &args);
            set_res(cpu, r);
        }
        SYS_SIGACTION => {
            let r = crate::signals::sigaction(args[0] as i32, args[1], args[2]);
            match r {
                Ok(()) => set_ok(cpu, 0, 0),
                Err(e) => set_err(cpu, e as u64),
            }
        }
        SYS_SIGALTSTACK => {
            match crate::signals::sigaltstack(args[0], args[1]) {
                Ok(()) => set_ok(cpu, 0, 0),
                Err(e) => set_err(cpu, e as u64),
            }
        }
        SYS_SIGRETURN => {
            if let Err(e) = crate::signals::sigreturn(cpu, args[0], args[1] as i32, args[2]) {
                set_err(cpu, e as u64);
            }
        }
        111 => { // sigsuspend: its argument is a mask value, not a pointer.
            let r = hostsys::unix(n, &args);
            if r.carry && r.rax == libc::EINTR as u64 {
                crate::signals::interrupted_suspend(args[0] as u32);
            }
            set_res(cpu, r);
        }
        SYS_MMAP => {
            let was_exec = args[2] & PROT_EXEC != 0;
            args[2] &= !PROT_EXEC;
            args[3] &= !MAP_JIT;
            let r = mmap_16k(&args);
            // dyld maps a library's __TEXT (file offset 0, executable) at its final address:
            // register it with the AOT cache so translations persist across runs.
            if !r.carry && was_exec && args[5] == 0 && (args[4] as i64) >= 0 && args[1] >= 0x1000 {
                crate::aot::register_image_at(r.rax);
            }
            set_res(cpu, r);
        }
        SYS_MPROTECT => {
            let exec = args[2] & PROT_EXEC != 0;
            args[2] &= !PROT_EXEC;
            let r = hostsys::unix(n, &args);
            if !r.carry && exec {
                crate::engine::invalidate_range(args[0], args[1]);
            }
            set_res(cpu, r);
        }
        SYS_SYSCTLBYNAME => {
            // The name is read fault-safely; an unreadable one is passed through,
            // as for path names, and the host kernel answers (EFAULT, or EINVAL for a
            // length it refuses first).
            // A name never read (NULL, length 0, or not below MAXPATHLEN) is
            // traced as that, not as unreadable, and gets no warning: no override is that name.
            let name = sysctl_name(args[0], args[1], crate::guestmem::read);
            let not_read = sysctl_name_not_read(args[0], args[1]);
            if name.is_none() && TRACE.load(Ordering::Relaxed) {
                let why = if not_read { "not read (NULL, empty or too long)" } else { "unreadable" };
                eprintln!("[sysctl] #{n} {:#x} len {} {why}: passed through", args[0], args[1]);
            }
            // One table, every answer as native gives it
            if let Some(a) = name.as_deref().and_then(sysctl_override) {
                sysctl_finish(cpu, sysctl_reply(a, args[2], args[3], args[4], args[5]));
                return;
            }
            let r = hostsys::unix(n, &args);
            set_res(cpu, r);
            if let Some(w) = unread_sysctl_warning(n, args[0], name.is_none() && !not_read, cpu.cf == 0) {
                eprintln!("{w}");
            }
        }
        SYS_SYSCTL => {
            // hw.machine (CTL_HW=6, HW_MACHINE=1) is what uname() reports. The MIB
            // is read fault-safely; an unreadable one is passed through and the host kernel answers
            // EFAULT. Any MIB that names an overridden sysctl
            // (sysctl_mib_answer) answers as that name does, so sysctlnametomib + sysctl and
            // getpagesize ({CTL_HW, HW_PAGESIZE}, hw.pagesize_compat) agree with sysctlbyname.
            let (mib, unread) = sysctl_mib(args[0], args[1], crate::guestmem::read);
            if unread && TRACE.load(Ordering::Relaxed) {
                eprintln!("[sysctl] #{n} {:#x} mib unreadable: passed through", args[0]);
            }
            if let Some(a) = mib.as_deref().and_then(|m| sysctl_mib_answer(m, host_oid2name)) {
                if TRACE.load(Ordering::Relaxed) {
                    eprintln!("[sysctl] #{n} mib {:?} overridden: {a:?}", mib.as_deref().unwrap_or_default());
                }
                sysctl_finish(cpu, sysctl_reply(a, args[2], args[3], args[4], args[5]));
                return;
            }
            let r = hostsys::unix(n, &args);
            set_res(cpu, r);
            if let Some(w) = unread_sysctl_warning(n, args[0], unread, cpu.cf == 0) {
                eprintln!("{w}");
            }
        }
        SYS_SHARED_REGION_CHECK_NP => {
            // -1 (DYLD_VM_END_MWL) marks the end of map_with_linking; 0 just
            // refreshes the task's shared region.
            if args[0] == u64::MAX || args[0] == 0 {
                set_ok(cpu, 0, 0);
                return;
            }
            match cache::check_np() {
                Some(base) => {
                    if crate::guestmem::write_u64(args[0], base) {
                        set_ok(cpu, 0, 0);
                    } else {
                        set_err(cpu, libc::EFAULT as u64);
                    }
                }
                None => set_err(cpu, libc::ENOMEM as u64),
            }
        }
        SYS_SHARED_REGION_MAP_AND_SLIDE_2_NP => {
            let e = cache::map_and_slide_2(args[0] as u32, args[1], args[2] as u32, args[3]);
            if e == 0 {
                set_ok(cpu, 0, 0);
            } else {
                set_err(cpu, e as u64);
            }
        }
        SYS_MAP_WITH_LINKING_NP => set_err(cpu, libc::ENOTSUP as u64),
        SYS_BSDTHREAD_REGISTER => {
            let r = threads::bsdthread_register(&args);
            match r {
                Ok(v) => set_ok(cpu, v, 0),
                Err(e) => set_err(cpu, e),
            }
        }
        SYS_BSDTHREAD_CREATE => match threads::bsdthread_create(&args) {
            Ok(v) => set_ok(cpu, v, 0),
            Err(e) => set_err(cpu, e),
        },
        SYS_BSDTHREAD_TERMINATE => threads::bsdthread_terminate(cpu, &args),
        SYS_WORKQ_OPEN => set_ok(cpu, 0, 0),
        SYS_WORKQ_KERNRETURN => match crate::workq::kernreturn(cpu, &args) {
            Ok(v) => set_ok(cpu, v, 0),
            Err(e) => set_err(cpu, e),
        },
        SYS_EXECVE | SYS_POSIX_SPAWN => {
            if n == SYS_EXECVE {
                crate::engine::flush_profile();
            }
            let r = crate::spawn::exec_like(n, &args);
            set_res(cpu, r);
        }
        374 | 375 => {
            if TRACE.load(Ordering::Relaxed) {
                eprintln!("[kev] #{n} fd/id={:#x} nchanges={} nevents={} flags={:#x}", args[0], args[2], args[4], args[7]);
                for i in 0..(args[2].min(4)) {
                    let e = args[1] + i * 72;
                    let v = unsafe { std::slice::from_raw_parts(e as *const u64, 9) };
                    eprintln!("[kev]   change ident={:#x} filter={} flags={:#x} qos={:#x} udata={:#x} fflags={:#x}", v[0], (v[1] & 0xffff) as i16, (v[1] >> 16) & 0xffff, v[1] >> 32, v[2], v[3] & 0xffffffff);
                }
            }
            let r = crate::workq::kevent_redirect(n, &args).unwrap_or_else(|| hostsys::unix(n, &args));
            set_res(cpu, r);
        }
        // bsdthread_ctl(BSDTHREAD_CTL_SET_SELF): thread QoS/priority/voucher. Guest threads are
        // host threads we schedule ourselves, and the host kernel rejects some arm64-era
        // priority combinations with EINVAL (libdispatch then aborts), so just accept it.
        478 if args[0] == 0x100 => set_ok(cpu, 0, 0),
        92 if matches!(args[1], 59 | 61 | 97 | 98 | 103) => {
            // Code-signing fcntls (F_ADDSIGS, F_ADDFILESIGS[_RETURN/_INFO], F_CHECK_LV):
            // the Intel host kernel cannot validate arm64 signatures, so accept them.
            // F_ADDFILESIGS_RETURN/_INFO report the signed extent back in fs_file_start;
            // the signature blob sits at the tail, so everything before it is signed.
            if matches!(args[1], 97 | 103) && args[2] != 0 {
                unsafe {
                    let fs = args[2] as *mut u64;
                    let blob_off = *fs.add(1);
                    *fs = blob_off;
                }
            }
            set_ok(cpu, 0, 0);
        }
        92 if args[1] == 105 && args[2] != 0 => {
            // F_GETSIGSINFO: dyld asks whether a file slice is a platform (OS) binary before it
            // lets an arm64 process load an arm64e OS dylib from disk (e.g. libobjc-trampolines,
            // which is not in the macOS 13 shared cache). The faked F_ADDFILESIGS above never
            // registered a signature with the Intel host kernel, so answer for system files.
            let mut path = [0u8; 1024];
            let g = hostsys::unix(92, &[args[0], 50 /* F_GETPATH */, path.as_mut_ptr() as u64, 0, 0, 0, 0, 0]);
            let p = std::ffi::CStr::from_bytes_until_nul(&path).ok().and_then(|c| c.to_str().ok()).unwrap_or("");
            if !g.carry && (p.starts_with("/usr/lib/") || p.starts_with("/System/")) {
                // struct fgetsigsinfo { off_t fg_file_start; int fg_info_request; int fg_sig_is_platform; }
                unsafe { *((args[2] + 12) as *mut i32) = 1 };
                set_ok(cpu, 0, 0);
            } else {
                let r = hostsys::unix(n, &args);
                set_res(cpu, r);
            }
        }
        _ => {
            let r = hostsys::unix(n, &args);
            set_res(cpu, r);
        }
    }
}

/// arm64 macOS has 16 KiB pages; the Intel host has 4 KiB. Guest allocators (V8,
/// PartitionAlloc, libmalloc) rely on mmap returning 16 KiB-aligned, 16 KiB-sized
/// regions, so over-reserve, align, and trim.
fn mmap_16k(args: &[u64; 8]) -> hostsys::SysResult {
    const PAGE: u64 = 0x4000;
    const MAP_FIXED: u64 = 0x10;
    const MAP_ANON: u64 = 0x1000;
    const MAP_PRIVATE: u64 = 0x2;
    let (len, flags, fd) = (args[1], args[3], args[4] as i64);
    if len == 0 || flags & MAP_FIXED != 0 {
        return hostsys::unix(SYS_MMAP, args);
    }
    let len16 = (len + PAGE - 1) & !(PAGE - 1);
    // Reserve len16 + one page of slack, address space only.
    let reserve = hostsys::unix(SYS_MMAP, &[0, len16 + PAGE, 0, MAP_ANON | MAP_PRIVATE, u64::MAX, 0, 0, 0]);
    if reserve.carry {
        return reserve;
    }
    let base = reserve.rax;
    let aligned = (base + PAGE - 1) & !(PAGE - 1);
    // Map the real thing over the aligned window.
    let mut a = *args;
    a[0] = aligned;
    a[1] = len16;
    a[3] = flags | MAP_FIXED;
    let _ = fd;
    let r = hostsys::unix(SYS_MMAP, &a);
    if r.carry {
        hostsys::unix(SYS_MUNMAP, &[base, len16 + PAGE, 0, 0, 0, 0, 0, 0]);
        return r;
    }
    // Give back the slack before and after the aligned window.
    if aligned > base {
        hostsys::unix(SYS_MUNMAP, &[base, aligned - base, 0, 0, 0, 0, 0, 0]);
    }
    let end = aligned + len16;
    let reserve_end = base + len16 + PAGE;
    if reserve_end > end {
        hostsys::unix(SYS_MUNMAP, &[end, reserve_end - end, 0, 0, 0, 0, 0, 0]);
    }
    r
}

/// The name of a sysctlbyname call at guest address `p`, read as XNU reads it:
/// `namelen` bytes by copyin (no NUL needed), then the string up to its first NUL (name2oid stops
/// there). `read` is guestmem::read in use: a fault-safe copy, never a dereference. None, and the
/// call goes to the host as it came, when `p` is NULL, `namelen` is 0 or not below MAXPATHLEN, or
/// a byte cannot be read. The cap only bounds the read: a name that long matches no override, so
/// the host would answer for it either way, and decides its own length limit. Then one terminal
/// dot is dropped, as name2oid drops it (xnu-12377.1.9
/// bsd/kern/kern_newsysctl.c, name2oid: `if (*p == '.') *p = '\0'` on the last byte before the NUL),
/// so "hw.machine." is hw.machine. Measured natively on macOS 27.0.1, arm64:
/// "hw.machine." answers arm64 with namelen 10, 11 or 12; "hw.machine..", ".hw.machine" and
/// "hw..machine" answer ENOENT. Only the one dot: a second, a leading or a doubled one is left, so it
/// matches no override and the host answers ENOENT as native does. This is the name looked up in the
/// overrides only; a call that is passed through goes to the host with the guest's own bytes.
fn sysctl_name(p: u64, namelen: u64, read: impl FnOnce(u64, &mut [u8]) -> bool) -> Option<String> {
    if sysctl_name_not_read(p, namelen) {
        return None;
    }
    let mut buf = vec![0u8; namelen as usize];
    if !read(p, &mut buf) {
        return None;
    }
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let name = &buf[..end];
    let name = name.strip_suffix(b".").unwrap_or(name);
    Some(String::from_utf8_lossy(name).into_owned())
}

/// The sysctlbyname names sysctl_name does not read at all: a NULL pointer, a
/// length of 0, or a length not below MAXPATHLEN.
fn sysctl_name_not_read(p: u64, namelen: u64) -> bool {
    p == 0 || namelen == 0 || namelen >= MAXPATHLEN as u64
}

/// XNU's limit on a MIB's length (sys/sysctl.h CTL_MAXNAME): a longer one is the host's EINVAL.
const CTL_MAXNAME: u64 = 12;

/// The MIB of a __sysctl call at `p` of
/// `namelen` entries, read fault-safely (`read` is guestmem::read in use), and whether it was
/// unreadable. A length outside 2..=CTL_MAXNAME is not read at all, so XNU's own checks (EINVAL for
/// fewer than 2 or more than CTL_MAXNAME entries) are the host's; an unreadable MIB is passed through
/// and the host kernel answers EFAULT. K v1 to v3 read only a two-entry MIB, the only one that could
/// be {CTL_HW, HW_MACHINE}.
fn sysctl_mib(p: u64, namelen: u64, read: impl FnOnce(u64, &mut [u8]) -> bool) -> (Option<Vec<i32>>, bool) {
    if !(2..=CTL_MAXNAME).contains(&namelen) {
        return (None, false);
    }
    let mut b = vec![0u8; 4 * namelen as usize];
    if !read(p, &mut b) {
        return (None, true);
    }
    (Some(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()), false)
}

/// The override a __sysctl MIB names, so a MIB
/// answers as its name does by sysctlbyname. {CTL_HW, HW_MACHINE} is hw.machine without asking. Any
/// other MIB under sysctl (0, past its meta entries 0 to 5: name, next, name2oid, oidfmt, oiddescr),
/// vm (2) or hw (6) is named by the host kernel itself (`oid2name`, host_oid2name in use: XNU's
/// {0, 1, mib...}), and that name is looked up in sysctl_override. The MIB numbers are the host
/// kernel's own (sysctlnametomib asks it, through __sysctl {0, 3}), so the host's name for them is
/// the one the guest asked about; {CTL_HW, HW_PAGESIZE} is hw.pagesize_compat (measured on macOS 27.0.1, arm64:
/// oid2name of {6, 7}), which getpagesize asks. A MIB longer than CTL_MAXNAME - 2 has no host name
/// to ask for and is passed through. What remains (Inferred): a name the host kernel does not
/// have has no MIB there (on an x86 host, hw.optional.arm64 and the arm FEAT names), so
/// sysctlnametomib answers ENOENT for it where native gives a MIB.
/// {CTL_HW, HW_PAGESIZE} is hw.pagesize_compat without asking too, as native arm64 names
/// it. Published XNU names it so on x86 as well (xnu-12377.1.9
/// bsd/kern/kern_mib.c: SYSCTL_PROC(_hw, HW_PAGESIZE, pagesize_compat, ...), with hw.pagesize and
/// hw.pagesize32 as OID_AUTO, none of them under an architecture guard); answering it directly keeps
/// the host's page size (4096 under Rosetta) out of it. The x86 kernel's own answer is not measured.
fn sysctl_mib_answer(mib: &[i32], oid2name: impl FnOnce(&[i32]) -> Option<String>) -> Option<SysctlAnswer> {
    match mib {
        [6, 1] => sysctl_override("hw.machine"),
        [6, 7] => sysctl_override("hw.pagesize_compat"),
        [0, m, ..] if *m <= 5 => None,
        [0 | 2 | 6, ..] if mib.len() as u64 <= CTL_MAXNAME - 2 => oid2name(mib).as_deref().and_then(sysctl_override),
        _ => None,
    }
}

/// The host kernel's name for MIB `mib` (XNU's oid2name, __sysctl {0, 1, mib...}); None when it
/// has none. An unknown entry is named by its number ("hw.8"), which matches no override.
fn host_oid2name(mib: &[i32]) -> Option<String> {
    let q: Vec<i32> = [0, 1].iter().chain(mib).copied().collect();
    let mut buf = [0u8; 256];
    let mut len = buf.len();
    let r = unsafe { libc::sysctl(q.as_ptr() as *mut i32, q.len() as u32, buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) };
    if r != 0 {
        return None;
    }
    let end = buf[..len.min(buf.len())].iter().position(|&c| c == 0)?;
    std::str::from_utf8(&buf[..end]).ok().map(str::to_string)
}

/// What to say when a sysctl's name or MIB at `p` could not be read and the host call still
/// succeeded: the kernel read what guestmem::read could not, so no arm64 override was applied
/// and the guest got the host's own answer. Reaches stderr without --trace, as the path-name warning does.
fn unread_sysctl_warning(n: u64, p: u64, unread: bool, host_ok: bool) -> Option<String> {
    (unread && host_ok).then(|| {
        // __sysctl (202) takes a MIB, not a name
        let what = if n == 202 { "MIB" } else { "name" };
        format!("[maclator] warning: #{n} sysctl {what} at {p:#x} could not be read but the host call succeeded: no override applied")
    })
}

/// Write `data` at guest address `p` as XNU's copyout does:
/// fault-safely (`write` is guestmem::write in use, never a dereference), and when the whole cannot
/// be written, the bytes before the first unwritable one still are. Measured on macOS 27.0.1, arm64: native
/// sysctlbyname("hw.machine") into a buffer 3 bytes before a PROT_NONE page leaves "arm" there and
/// answers EFAULT, while a whole guestmem::write of those 6 bytes from inside the translator writes
/// nothing (probe row N5; from the unit-test binary the same whole write did write the prefix).
/// False when not every byte was written.
fn guest_copyout(p: u64, data: &[u8], write: impl Fn(u64, &[u8]) -> bool) -> bool {
    if write(p, data) {
        return true;
    }
    for (i, b) in data.iter().enumerate() {
        if !write(p + i as u64, std::slice::from_ref(b)) {
            break;
        }
    }
    false
}

/// Hand an override's answer back as XNU's sysctl does,
/// through guestmem (fault-safe; the old writers dereferenced the guest's `oldp` and `oldlenp`
/// raw, so a bad pointer killed the translator). `oldlenp` NULL: nothing is read or written, and
/// the call answers ENOMEM when `oldp` is not NULL. Else `oldlenp` is read (EFAULT when it
/// cannot be). With `oldp` NULL only the answer's length `query` is written back. Else
/// `answer(oldlen)` gives the bytes, or None for a short buffer: ENOMEM, nothing written to `oldp`
/// and 0 written to `oldlenp`. Else the bytes are copied out (EFAULT, with `oldlenp`
/// untouched, when they cannot all be), then their length is written to `oldlenp` (EFAULT when it
/// cannot be). Measured on macOS 27.0.1, arm64, natively and the same for a host sysctl (kern.ostype):
/// an unreadable or unwritable `oldlenp` never returns (3 s at 99% CPU, 78% of it system time, so
/// the call is presumably restarted), and EFAULT, what sysctl(3) documents, is answered instead
/// of a hang; `oldp` with a NULL `oldlenp` answers ENOMEM and writes nothing; a short buffer
/// (oldlen 0, 2 or 3) answers ENOMEM, writes no prefix, and sets oldlen to 0 (probe rows 6 to 8).
fn sysctl_copyout(oldp: u64, oldlenp: u64, query: u64, answer: impl FnOnce(u64) -> Option<Vec<u8>>) -> Result<(), u64> {
    let (efault, enomem) = (libc::EFAULT as u64, libc::ENOMEM as u64);
    if oldlenp == 0 {
        return if oldp == 0 { Ok(()) } else { Err(enomem) };
    }
    let oldlen = crate::guestmem::read_u64(oldlenp).ok_or(efault)?;
    if oldp == 0 {
        return crate::guestmem::write_u64(oldlenp, query).then_some(()).ok_or(efault);
    }
    let Some(bytes) = answer(oldlen) else {
        return Err(if crate::guestmem::write_u64(oldlenp, 0) { enomem } else { efault });
    };
    if !guest_copyout(oldp, &bytes, crate::guestmem::write) {
        return Err(efault);
    }
    crate::guestmem::write_u64(oldlenp, bytes.len() as u64).then_some(()).ok_or(efault)
}

fn sysctl_finish(cpu: &mut Cpu, r: Result<(), u64>) {
    match r {
        Ok(()) => set_ok(cpu, 0, 0),
        Err(e) => set_err(cpu, e),
    }
}

/// What an overridden sysctl answers,
/// in the shape native gives it. Measured on macOS 27.0.1, arm64 and arm64e alike, by name and by
/// MIB alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SysctlAnswer {
    /// A string, its NUL included: the whole string, or ENOMEM and oldlen 0 for a shorter buffer.
    Str(&'static str),
    /// A 4-byte integer: 4 bytes and oldlen 4 for any buffer of 4 or more, an 8-byte one included;
    /// ENOMEM and oldlen 0 below 4; the length query answers 4. K v1 to v3 wrote 8 bytes into a
    /// buffer of 8 or more.
    Int(i32),
    /// An 8-byte integer (hw.pagesize, hw.pagesize32): 8 bytes for a buffer of 8 or more, 4 bytes
    /// for a buffer of exactly 4, ENOMEM and oldlen 0 otherwise (5 to 7 included); the length
    /// query answers 8. K v1 to v3 answered 4 to the query and 4 bytes to a buffer of 5 to 7.
    Quad(i64),
    /// A name that exists and answers `errno` to every read before touching oldp or oldlenp, oldlen
    /// left as it was (hw.optional.x86_64 and the SSE and AVX names on arm64: ENOTSUP). A write still
    /// gets EPERM first.
    Fails(i32),
    /// A name that does not exist natively (hw.optional.AdvSIMD on 27.0.1): ENOENT to every call,
    /// a write included, nothing touched.
    Absent,
}

/// Answer an overridden sysctl `a` as XNU
/// does, for a call with old (`oldp`, `oldlenp`) and new (`newp`, `newlen`) values. Measured on
/// macOS 27.0.1, arm64: a name that does not exist answers ENOENT before anything else; a write (newp not NULL
/// and newlen not 0; newp NULL or newlen 0 is a read) to a read-only name, which every overridden
/// name is, answers EPERM before oldlenp is read, with oldp and oldlen untouched, whatever oldp,
/// oldlenp and newp point at (an unmapped oldlenp too, where a read never returns); a Fails name
/// answers its errno with nothing touched. Anything else is sysctl_copyout. K v1 to v3 ignored
/// newp and newlen and answered a write as a read.
fn sysctl_reply(a: SysctlAnswer, oldp: u64, oldlenp: u64, newp: u64, newlen: u64) -> Result<(), u64> {
    if a == SysctlAnswer::Absent {
        return Err(libc::ENOENT as u64);
    }
    if newp != 0 && newlen != 0 {
        return Err(libc::EPERM as u64);
    }
    match a {
        SysctlAnswer::Str(v) => {
            let bytes = [v.as_bytes(), &[0]].concat();
            sysctl_copyout(oldp, oldlenp, bytes.len() as u64, |len| (len >= bytes.len() as u64).then_some(bytes))
        }
        SysctlAnswer::Int(v) => sysctl_copyout(oldp, oldlenp, 4, |len| (len >= 4).then(|| v.to_le_bytes().to_vec())),
        SysctlAnswer::Quad(v) => sysctl_copyout(oldp, oldlenp, 8, |len| match len {
            l if l >= 8 => Some(v.to_le_bytes().to_vec()),
            4 => Some((v as i32).to_le_bytes().to_vec()),
            _ => None,
        }),
        SysctlAnswer::Fails(e) => Err(e as u64),
        SysctlAnswer::Absent => Err(libc::ENOENT as u64),
    }
}

/// sysctls whose answer must describe the emulated arm64 machine (an M1), by name. The values are
/// the translator's choice (the features its JIT gives the guest); the shapes are native's
/// (SysctlAnswer), measured on macOS 27.0.1, arm64 for every name here.
fn sysctl_override(name: &str) -> Option<SysctlAnswer> {
    use SysctlAnswer::{Absent, Int, Quad, Str};
    Some(match name {
        "hw.machine" => Str("arm64"),
        "sysctl.proc_translated" => Int(0),
        "sysctl.proc_native" => Int(1),
        "hw.cputype" => Int(0x0100_000c),
        "hw.cpusubtype" => Int(2),
        "hw.optional.arm64" => Int(1),
        // hw.optional.AdvSIMD does not exist on 27.0.1 (ENOENT natively); an earlier version answered 1
        "hw.optional.AdvSIMD" => Absent,
        "hw.optional.neon" | "hw.optional.floatingpoint" => Int(1),
        // The arm names the commpage says the translator honours (commpage.rs: kHasAdvSIMD,
        // kHasAdvSIMD_HPFPCvt, kHasUCNormalMemory, kHasFeatCSV2, kHasFeatCSV3). Every other arm name native
        // has answers 0 (sysctl_optional_default), hw.optional.neon_fp16 with FEAT_FP16's 0 among them.
        "hw.optional.arm.AdvSIMD" | "hw.optional.arm.AdvSIMD_HPFPCvt" | "hw.optional.neon_hpfp"
        | "hw.optional.ucnormal_mem" | "hw.optional.arm.FEAT_CSV2" | "hw.optional.arm.FEAT_CSV3" => Int(1),
        "hw.optional.armv8_crc32" | "hw.optional.arm.FEAT_CRC32" => Int(1),
        "hw.optional.armv8_1_atomics" | "hw.optional.arm.FEAT_LSE" => Int(1),
        // Aligned 128-bit support is narrower than the full LSE2 contract (upstream #7).
        "hw.optional.arm.FEAT_LSE2" => Int(0),
        "hw.optional.arm.FEAT_LRCPC" | "hw.optional.arm.FEAT_LRCPC2" => Int(1),
        "hw.optional.arm.FEAT_PAuth" | "hw.optional.arm.FEAT_JSCVT" | "hw.optional.arm.FEAT_FlagM"
        | "hw.optional.arm.FEAT_FlagM2" | "hw.optional.arm.FEAT_DotProd" | "hw.optional.arm.FEAT_FRINTTS"
        | "hw.optional.arm.FEAT_SB" | "hw.optional.arm.FEAT_DPB" | "hw.optional.arm.FEAT_DPB2" => Int(1),
        "hw.optional.arm.FEAT_FP16" | "hw.optional.arm.FEAT_FHM" | "hw.optional.arm.FEAT_FCMA"
        | "hw.optional.arm.FEAT_RDM" | "hw.optional.arm.FEAT_SHA3" | "hw.optional.arm.FEAT_SHA512"
        | "hw.optional.arm.FEAT_SME" | "hw.optional.arm.FEAT_SME2" | "hw.optional.arm.FEAT_BF16"
        | "hw.optional.arm.FEAT_I8MM" | "hw.optional.armv8_2_fhm" | "hw.optional.armv8_2_sha512"
        | "hw.optional.armv8_2_sha3" | "hw.optional.armv8_3_compnum" | "hw.optional.arm.FEAT_SHA256"
        | "hw.optional.arm.FEAT_AES" | "hw.optional.arm.FEAT_PMULL" | "hw.optional.arm.FEAT_SHA1"
        | "hw.optional.arm.FEAT_AFP" | "hw.optional.arm.FEAT_RPRES" | "hw.optional.arm.FEAT_ECV"
        | "hw.optional.arm.FEAT_WFxT" | "hw.optional.arm.FEAT_CSSC" | "hw.optional.arm.FEAT_HBC" => Int(0),
        "hw.pagesize" | "hw.pagesize32" => Quad(16384),
        // {CTL_HW, HW_PAGESIZE}'s own name (getpagesize), and vm.pagesize: 4 bytes
        "hw.pagesize_compat" | "vm.pagesize" => Int(16384),
        "hw.cpufamily" => Int(crate::commpage::CPUFAMILY as i32),
        // Any other hw.optional name native has, the x86 names included
        _ => return sysctl_optional_default(name),
    })
}

/// An hw.optional name sysctl_override does not
/// list, answered as native arm64 has it (the measured lists below). Every x86 feature name native
/// answers ENOTSUP for (29 on 27.0.1: K v4 listed 9, so on an x86 host the other 20 got the host's 1),
/// answers ENOTSUP. Every other name native has as a 4-byte integer answers 0, a feature the translator
/// does not promise: on an x86 host its kernel has no such name and would answer ENOENT, and on macOS 27.0.1, arm64 the
/// host answered M1's own value. A name native has in another shape (hw.optional.arm.caps, 13 bytes) and
/// any name native does not have are passed through (None).
fn sysctl_optional_default(name: &str) -> Option<SysctlAnswer> {
    if NATIVE_OPTIONAL_ENOTSUP.contains(&name) {
        Some(SysctlAnswer::Fails(libc::ENOTSUP))
    } else if NATIVE_OPTIONAL_INT.contains(&name) {
        Some(SysctlAnswer::Int(0))
    } else {
        None
    }
}

// BEGIN GENERATED from the hw.optional names of macOS 27.0.1, arm64e (
// every name `sysctl -aN` lists under hw.optional, read natively as arm64e; arm64 identical). Do not edit by hand.
/// The hw.optional names native arm64 answers ENOTSUP for (the x86 feature names): 29.
const NATIVE_OPTIONAL_ENOTSUP: [&str; 29] = [
    "hw.optional.mmx",
    "hw.optional.sse",
    "hw.optional.sse2",
    "hw.optional.sse3",
    "hw.optional.supplementalsse3",
    "hw.optional.sse4_1",
    "hw.optional.sse4_2",
    "hw.optional.x86_64",
    "hw.optional.aes",
    "hw.optional.avx1_0",
    "hw.optional.rdrand",
    "hw.optional.f16c",
    "hw.optional.enfstrg",
    "hw.optional.fma",
    "hw.optional.avx2_0",
    "hw.optional.bmi1",
    "hw.optional.bmi2",
    "hw.optional.rtm",
    "hw.optional.hle",
    "hw.optional.adx",
    "hw.optional.mpx",
    "hw.optional.sgx",
    "hw.optional.avx512f",
    "hw.optional.avx512cd",
    "hw.optional.avx512dq",
    "hw.optional.avx512bw",
    "hw.optional.avx512vl",
    "hw.optional.avx512ifma",
    "hw.optional.avx512vbmi",
];
/// The other hw.optional names native arm64 has, each a 4-byte integer: 95.
const NATIVE_OPTIONAL_INT: [&str; 95] = [
    "hw.optional.arm.FEAT_FP8",
    "hw.optional.arm.FEAT_CRC32",
    "hw.optional.arm.FEAT_FlagM",
    "hw.optional.arm.FEAT_FlagM2",
    "hw.optional.arm.FEAT_FHM",
    "hw.optional.arm.FEAT_DotProd",
    "hw.optional.arm.FEAT_SHA3",
    "hw.optional.arm.FEAT_RDM",
    "hw.optional.arm.FEAT_LSE",
    "hw.optional.arm.FEAT_SHA256",
    "hw.optional.arm.FEAT_SHA512",
    "hw.optional.arm.FEAT_SHA1",
    "hw.optional.arm.FEAT_AES",
    "hw.optional.arm.FEAT_PMULL",
    "hw.optional.arm.FEAT_SPECRES",
    "hw.optional.arm.FEAT_SPECRES2",
    "hw.optional.arm.FEAT_SB",
    "hw.optional.arm.FEAT_FRINTTS",
    "hw.optional.arm.FEAT_PACIMP",
    "hw.optional.arm.FEAT_LRCPC",
    "hw.optional.arm.FEAT_LRCPC2",
    "hw.optional.arm.FEAT_FCMA",
    "hw.optional.arm.FEAT_JSCVT",
    "hw.optional.arm.FEAT_PAuth",
    "hw.optional.arm.FEAT_PAuth2",
    "hw.optional.arm.FEAT_FPAC",
    "hw.optional.arm.FEAT_FPACCOMBINE",
    "hw.optional.arm.FEAT_PAuth_LR",
    "hw.optional.arm.FEAT_DPB",
    "hw.optional.arm.FEAT_DPB2",
    "hw.optional.arm.FEAT_BF16",
    "hw.optional.arm.FEAT_EBF16",
    "hw.optional.arm.FEAT_I8MM",
    "hw.optional.arm.FEAT_WFxT",
    "hw.optional.arm.FEAT_RPRES",
    "hw.optional.arm.FEAT_CSSC",
    "hw.optional.arm.FEAT_HBC",
    "hw.optional.arm.FEAT_LUT",
    "hw.optional.arm.FEAT_FAMINMAX",
    "hw.optional.arm.FEAT_CPA",
    "hw.optional.arm.FEAT_CPA2",
    "hw.optional.arm.FEAT_ECV",
    "hw.optional.arm.FEAT_AFP",
    "hw.optional.arm.FEAT_LSE2",
    "hw.optional.arm.FEAT_CSV2",
    "hw.optional.arm.FEAT_CSV3",
    "hw.optional.arm.FEAT_DIT",
    "hw.optional.arm.AdvSIMD",
    "hw.optional.arm.AdvSIMD_HPFPCvt",
    "hw.optional.arm.FEAT_FP16",
    "hw.optional.arm.FEAT_SSBS",
    "hw.optional.arm.FEAT_BTI",
    "hw.optional.arm.FEAT_SME",
    "hw.optional.arm.FEAT_SME2",
    "hw.optional.arm.FEAT_SME2p1",
    "hw.optional.arm.FEAT_MTE",
    "hw.optional.arm.FEAT_MTE2",
    "hw.optional.arm.FEAT_MTE3",
    "hw.optional.arm.FEAT_MTE4",
    "hw.optional.arm.FEAT_MTE_ASYNC",
    "hw.optional.arm.FEAT_MTE_STORE_ONLY",
    "hw.optional.arm.FEAT_FPMR",
    "hw.optional.arm.SME_F32F32",
    "hw.optional.arm.SME_BI32I32",
    "hw.optional.arm.SME_B16F32",
    "hw.optional.arm.SME_F16F32",
    "hw.optional.arm.SME_I8I32",
    "hw.optional.arm.SME_I16I32",
    "hw.optional.arm.FEAT_SME_F64F64",
    "hw.optional.arm.FEAT_SME_I16I64",
    "hw.optional.arm.FEAT_SME_F16F16",
    "hw.optional.arm.FEAT_SME_B16B16",
    "hw.optional.arm.FEAT_SME_LUTv2",
    "hw.optional.arm.FEAT_SME_F8F16",
    "hw.optional.arm.FEAT_SME_F8F32",
    "hw.optional.arm.FP_SyncExceptions",
    "hw.optional.arm.FEAT_SVE_B16B16",
    "hw.optional.arm.FEAT_MTE_CANONICAL_TAGS",
    "hw.optional.arm.FEAT_MTE_NO_ADDRESS_TAGS",
    "hw.optional.arm.sme_max_svl_b",
    "hw.optional.floatingpoint",
    "hw.optional.neon",
    "hw.optional.neon_hpfp",
    "hw.optional.neon_fp16",
    "hw.optional.armv8_crc32",
    "hw.optional.armv8_gpi",
    "hw.optional.armv8_1_atomics",
    "hw.optional.armv8_2_fhm",
    "hw.optional.armv8_2_sha512",
    "hw.optional.armv8_2_sha3",
    "hw.optional.armv8_3_compnum",
    "hw.optional.watchpoint",
    "hw.optional.breakpoint",
    "hw.optional.ucnormal_mem",
    "hw.optional.arm64",
];
/// hw.optional names native arm64 has in another shape (not overridden; passed through): 1.
#[cfg(test)]
const NATIVE_OPTIONAL_OTHER: [&str; 1] = [
    "hw.optional.arm.caps",
];
// END GENERATED

fn mach_trap(cpu: &mut Cpu, n: u64) {
    let trace = TRACE.load(Ordering::Relaxed);
    match n {
        3 => {
            cpu.x[0] = unsafe { mach_absolute_time() };
            return;
        }
        4 => {
            cpu.x[0] = unsafe { mach_continuous_time() };
            return;
        }
        _ => {}
    }
    let args = [cpu.x[0], cpu.x[1], cpu.x[2], cpu.x[3], cpu.x[4], cpu.x[5], cpu.x[6], cpu.x[7]];
    if n == 10 && args[3] & 1 != 0 && args[1] != 0 && args[2] != 0 {
        // mach_vm_allocate(VM_FLAGS_ANYWHERE): place it from a deterministic arena so that
        // dyld-loaded libraries land at the same addresses every run (persistent AOT cache).
        // The arena is a large PROT_NONE reservation; allocations overwrite pieces of it.
        const ARENA_SIZE: u64 = 64 << 30;
        static ARENA: crate::forksync::FOnceLock<(u64, std::sync::atomic::AtomicU64)> = crate::forksync::FOnceLock::new();
        let (base, next) = ARENA.get_or_init(|| {
            for cand in [0x40_0000_0000u64, 0x20_0000_0000, 0x10_0000_0000, 0x0c_0000_0000, 0x08_0000_0000] {
                let p = unsafe {
                    libc::mmap(cand as *mut _, ARENA_SIZE as usize, libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE, -1, 0)
                };
                if p as u64 == cand {
                    return (cand, std::sync::atomic::AtomicU64::new(0));
                }
                if p != libc::MAP_FAILED {
                    unsafe { libc::munmap(p, ARENA_SIZE as usize) };
                }
            }
            (0, std::sync::atomic::AtomicU64::new(0))
        });
        let sz = (args[2] + 0x3fff) & !0x3fff;
        if *base != 0 {
            let off = next.fetch_add(sz, Ordering::SeqCst);
            if off + sz <= ARENA_SIZE {
                let at = base + off;
                let orig_hint = unsafe { *(args[1] as *const u64) };
                unsafe { *(args[1] as *mut u64) = at };
                let mut fixed = args;
                fixed[3] = (fixed[3] & !1) | 0x4000; // fixed + VM_FLAGS_OVERWRITE
                let r = hostsys::mach(n, &fixed);
                if r.rax == 0 {
                    cpu.x[0] = 0;
                    return;
                }
                if trace {
                    eprintln!("[mach] arena vm_allocate at {at:#x} size {sz:#x} failed: {:#x}", r.rax);
                }
                unsafe { *(args[1] as *mut u64) = orig_hint };
            }
        }
    }
    if n == 47 && crate::iokit::intercept(args[0], args[1]) {
        if trace {
            eprintln!("[mach] msg2 answered by iokit shim");
        }
        cpu.x[0] = 0;
        return;
    }
    // mach_msg2: remember the request id/size before the reply overwrites the header.
    let req_id = if n == 47 && args[0] != 0 { unsafe { *((args[0] + 20) as *const u32) } } else { 0 };
    if trace && n == 47 && args[0] != 0 {
        unsafe {
            let h = args[0] as *const u32;
            let size = (*h.add(1)).min(512) as usize;
            let body = std::slice::from_raw_parts(args[0] as *const u8, size);
            let mut strs = String::new();
            let mut cur = String::new();
            for &b in body {
                if (0x20..0x7f).contains(&b) {
                    cur.push(b as char);
                } else {
                    if cur.len() >= 4 {
                        strs.push_str(&format!(" \"{cur}\""));
                    }
                    cur.clear();
                }
            }
            eprintln!("[mach] msg2> bits={:#x} size={} rport={:#x} lport={:#x} id={} opt={:#x}{}", *h, *h.add(1), *h.add(2), *h.add(3), *h.add(5), args[1], strs);
        }
    }
    let r = hostsys::mach(n, &args);
    if n == 47 && args[0] != 0 {
        let reply_id = unsafe { *((args[0] + 20) as *const u32) };
        let retcode = unsafe { *((args[0] + 32) as *const u32) };
        if trace {
            eprintln!("[mach] msg2 id={req_id} reply_id={reply_id} retcode={retcode:#x}");
            if (2800..3000).contains(&req_id) {
                let w: Vec<String> = (0..16).map(|i| format!("{:08x}", unsafe { *((args[0] + i * 4) as *const u32) })).collect();
                eprintln!("[mach] reply words: {}", w.join(" "));
            }
        }
        // task_restartable_ranges_register: the Intel host kernel does not support it, and
        // libobjc treats failure as fatal. Restartable ranges only matter for arm64 threads
        // we already run in the interpreter/JIT, so pretend it worked.
        if req_id == 8000 && reply_id == req_id + 100 && retcode == 46 {
            unsafe { *((args[0] + 32) as *mut u32) = 0 };
        }
    }
    if trace {
        eprintln!("[mach] trap {}({:#x}, {:#x}, {:#x}, {:#x}) -> {:#x}", n, args[0], args[1], args[2], args[3], r.rax);
    }
    cpu.x[0] = r.rax;
}

extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_continuous_time() -> u64;
}

pub fn syscall_name(n: u64) -> &'static str {
    match n {
        1 => "exit",
        2 => "fork",
        3 => "read",
        4 => "write",
        5 => "open",
        6 => "close",
        20 => "getpid",
        33 => "access",
        46 => "sigaction",
        48 => "sigprocmask",
        54 => "ioctl",
        59 => "execve",
        73 => "munmap",
        74 => "mprotect",
        92 => "fcntl",
        116 => "gettimeofday",
        153 => "pread",
        169 => "csops",
        184 => "sigreturn",
        194 => "getrlimit",
        197 => "mmap",
        202 => "sysctl",
        220 => "getattrlist",
        244 => "posix_spawn",
        274 => "sysctlbyname",
        294 => "shared_region_check_np",
        327 => "issetugid",
        336 => "proc_info",
        338 => "stat64",
        339 => "fstat64",
        340 => "lstat64",
        344 => "getdirentries64",
        360 => "bsdthread_create",
        361 => "bsdthread_terminate",
        366 => "bsdthread_register",
        367 => "workq_open",
        368 => "workq_kernreturn",
        372 => "thread_selfid",
        396 => "read_nocancel",
        397 => "write_nocancel",
        398 => "open_nocancel",
        399 => "close_nocancel",
        463 => "openat",
        464 => "openat_nocancel",
        500 => "getentropy",
        515 => "ulock_wait",
        516 => "ulock_wake",
        520 => "terminate_with_payload",
        521 => "abort_with_payload",
        536 => "shared_region_map_and_slide_2_np",
        550 => "map_with_linking_np",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guest memory for the tests: `bytes` at address `base`; reads below `limit` succeed.
    fn reader(base: u64, bytes: &[u8], limit: u64) -> impl FnMut(u64, &mut [u8]) -> bool + '_ {
        move |at, buf| {
            if at < base || at + buf.len() as u64 > limit {
                return false;
            }
            let off = (at - base) as usize;
            buf.copy_from_slice(&bytes[off..off + buf.len()]);
            true
        }
    }

    #[test]
    fn guest_cstr_reads_up_to_an_unreadable_page_and_no_further() {
        let base = 0x10_0000u64;
        let mut mem = vec![b'a'; 8192];
        // "name\0" ending exactly at the end of the readable page.
        mem[4096 - 5..4096].copy_from_slice(b"name\0");
        let p = base + 4096 - 5;
        assert_eq!(guest_cstr(p, GUEST_NAME_MAX, reader(base, &mem, base + 4096)).as_deref(), Some("name"));
        // The same bytes with no NUL before the unreadable page: unreadable, not a crash.
        mem[4095] = b'x';
        assert_eq!(guest_cstr(p, GUEST_NAME_MAX, reader(base, &mem, base + 4096)), None);
        // A pointer into the unreadable page itself.
        assert_eq!(guest_cstr(base + 4096, GUEST_NAME_MAX, reader(base, &mem, base + 4096)), None);
    }

    #[test]
    fn guest_cstr_reads_across_pages_and_stops_at_max() {
        let base = 0x20_0000u64;
        let mut mem = vec![b'b'; 8192];
        mem[5000] = 0;
        let p = base + 3000;
        assert_eq!(guest_cstr(p, GUEST_NAME_MAX, reader(base, &mem, base + 8192)).map(|s| s.len()), Some(2000));
        assert_eq!(guest_cstr(p, 2000, reader(base, &mem, base + 8192)), None);
        assert_eq!(guest_cstr(p, 2001, reader(base, &mem, base + 8192)).map(|s| s.len()), Some(2000));
    }

    #[test]
    fn path_arg_covers_only_one_path_calls() {
        // every *at call's path is args[1], so its descriptor is args[0]
        for n in [463, 464, 466, 469, 470, 473, 476] {
            assert_eq!(path_arg(n), (Some(1), true), "*at call {n}");
        }
        for n in [5, 398, 12, 33, 58, 188, 190, 220, 338, 340, 348] {
            assert_eq!(path_arg(n), (Some(0), false), "plain call {n}");
        }
        // the two-path calls pass through untouched: renameat, linkat, symlinkat, rename, link, symlink
        for n in [465, 471, 474, 128, 9, 57] {
            assert_eq!(path_arg(n), (None, false), "two-path call {n}");
        }
    }

    #[test]
    fn guest_full_path_leaves_an_empty_name_alone() {
        // An empty name is never joined, and the descriptor's path is not even asked for
        let asked = std::cell::Cell::new(false);
        let dir = || { asked.set(true); Some("/System/Library/dyld".to_string()) };
        assert_eq!(guest_full_path("", true, dir), None);
        assert!(!asked.get(), "the empty name asked for the descriptor's path");
        assert_eq!(guest_full_path("", false, || Some("/x".into())), None);
    }

    #[test]
    fn guest_full_path_joins_only_relative_names_in_at_calls() {
        let d = || Some("/System/Library/dyld".to_string());
        assert_eq!(guest_full_path("dyld_shared_cache_arm64e", true, d).as_deref(),
                   Some("/System/Library/dyld/dyld_shared_cache_arm64e"));
        assert_eq!(guest_full_path("x", true, || Some("/".into())).as_deref(), Some("/x"));
        assert_eq!(guest_full_path("/etc/hosts", true, || None).as_deref(), Some("/etc/hosts"));
        assert_eq!(guest_full_path("/etc/hosts", false, || None).as_deref(), Some("/etc/hosts"));
        // A relative name in a plain call is joined to the base it is given (the working
        // directory's); no base, no join
        assert_eq!(guest_full_path("x", false, || Some("/System".into())).as_deref(), Some("/System/x"));
        assert_eq!(guest_full_path("x", true, || None), None);
        assert_eq!(guest_full_path("x", false, || None), None);
    }

    #[test]
    fn hv9_a_record_is_kept_while_its_guest_path_leads_here_else_recomputed_from_the_host() {
        let pre = "/System/Library/dyld";
        let r = CwdRec { key: (1, 2), prefix: pre, rel: "/sub".into() };
        let sys = Some("/sysroot");
        // The physical path decides when it can be read, even while the guest path still leads here
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| panic!("asked the guest path"), || Some("/sysroot/sub".into()), sys, || false), Some(r.clone()));
        let aliased = CwdRec { key: (1, 2), prefix: pre, rel: "/x/y".into() };
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| Some((1, 2)), || Some("/sysroot/x/y".into()), sys, || false), Some(aliased), "an alias left at the old spelling");
        // unreadable "." (deeper than MAXPATHLEN): the record stands while its guest path leads here
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| Some((1, 2)), || None, sys, || false), Some(r.clone()));
        // a record for another directory is no record (must-fail control for the key test)
        assert_eq!(cwd_check(r.clone(), (1, 3), |_| Some((1, 3)), || panic!("asked the host"), sys, || false), None);
        // renamed: the prefix stays, the place below the sysroot is read again
        let moved = CwdRec { key: (1, 2), prefix: pre, rel: "/sub3".into() };
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| None, || Some("/sysroot/sub3".into()), sys, || false), Some(moved));
        // moved out of the sysroot, or its path unreadable: no record (the host's path is the guest's)
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| None, || Some("/private/tmp/x".into()), sys, || false), None);
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| None, || None, sys, || false), None);
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| None, || Some("/sysroot2/sub".into()), sys, || false), None, "a sibling with the sysroot's name as prefix");
        // A spelling the strip misses, inside by identity: kept
        assert_eq!(cwd_check(r.clone(), (1, 2), |_| None, || Some("/System/Volumes/Data/sysroot/sub".into()), sys, || true), Some(r.clone()));
        assert_eq!(cwd_check(r, (1, 2), |_| None, || None, sys, || true).is_some(), true);
    }

    #[test]
    fn hv9_rec_at_takes_only_a_place_inside_the_sysroot() {
        let sys = Some("/sysroot");
        assert_eq!(rec_at((1, 2), "/p", Some("/sysroot".into()), sys).map(|r| r.guest()).as_deref(), Some("/p"));
        assert_eq!(rec_at((1, 2), "/p", Some("/sysroot/a/b".into()), sys).map(|r| r.guest()).as_deref(), Some("/p/a/b"));
        assert_eq!(rec_at((1, 2), "/p", Some("/private/tmp/out/d".into()), sys), None, "a link out of the sysroot");
        assert_eq!(rec_at((1, 2), "/p", Some("/sysrootx".into()), sys), None);
        assert_eq!(rec_at((1, 2), "/p", None, sys), None);
        assert_eq!(rec_at((1, 2), "/p", Some("/sysroot".into()), None), None);
    }

    #[test]
    fn hv9_escape_at_is_the_first_dotdot_taken_at_the_sysroots_root() {
        let root = (9, 9);
        // the working directory is the root: "." lands there
        let at_root = |p: &str| match p { "." | "./" | "a/.." | "a/../" => Some(root), "a" | "a/" => Some((9, 1)), _ => None };
        assert_eq!(escape_at("../victim", root, at_root), Some(0));
        assert_eq!(escape_at("./../x", root, at_root), Some(2));
        assert_eq!(escape_at("a/../../x", root, at_root), Some(5));
        assert_eq!(escape_at("a/../b", root, at_root), None, "a `..` taken below the root stays inside");
        assert_eq!(escape_at("a/b", root, at_root), None);
        assert_eq!(escape_at("..", root, at_root), Some(0));
        // 500 "./" then "../victim"; the stat sees the same root for every "./" prefix
        let long = format!("{}../victim", "./".repeat(500));
        assert_eq!(escape_at(&long, root, |_| Some(root)), Some(1000));
        // one level down: the first `..` stays, the second climbs out
        let down = |p: &str| match p { "." => Some((9, 1)), "../" => Some(root), _ => None };
        assert_eq!(escape_at("../../x", root, down), Some(3));
        assert_eq!(escape_at("../x", root, down), None);
        // a leading part that does not land: the host's walk stops there too
        assert_eq!(escape_at("nope/../../x", root, |p| if p == "." { Some((9, 1)) } else { None }), None);
        // must-fail control: a root that never matches finds nothing
        assert_eq!(escape_at("../victim", (0, 0), at_root), None);
    }

    #[test]
    fn hv9_escaped_path_and_the_relative_reentry() {
        let r = CwdRec { key: (1, 2), prefix: "/System/Library/dyld", rel: "/a/b".into() };
        assert_eq!(escaped_path(&r, "x/../../../victim", 8), "/System/Library/dyld/../victim");
        assert_eq!(reentry_relative("/sysroot/f", 2, "/sysroot").as_deref(), Some("../../f"));
        assert_eq!(reentry_relative("/sysroot/f", 0, "/sysroot").as_deref(), Some("f"));
        assert_eq!(reentry_relative("/sysroot/d/", 1, "/sysroot").as_deref(), Some("../d/"));
        assert_eq!(reentry_relative("/elsewhere/f", 0, "/sysroot"), None);
        assert_eq!(reentry_relative("/sysrootx/f", 0, "/sysroot"), None);
    }

    #[test]
    fn hv8_an_unrecorded_descriptor_starts_from_its_host_path_outside_the_sysroot() {
        let sys = Some("/sysroot");
        assert_eq!(unrecorded_fd_base(3, |_| Some("/private/tmp/ec".into()), sys).as_deref(), Some("/private/tmp/ec"));
        assert_eq!(unrecorded_fd_base(3, |_| Some("/sysroot/sub".into()), sys), None);
        assert_eq!(unrecorded_fd_base(3, |_| Some("/sysroot".into()), sys), None);
        assert_eq!(unrecorded_fd_base(3, |_| None, sys), None);
        assert_eq!(unrecorded_fd_base(3, |_| Some("/sysroot".into()), None).as_deref(), Some("/sysroot"));
    }

    #[test]
    fn hv9_the_pure_parts_of_the_bsd_read_branches() {
        // cwd_host_base: nothing to redirect, nothing joined; inside the sysroot by its host path, nothing joined
        assert_eq!(cwd_host_base(|| panic!("asked the host"), None, false), None);
        assert_eq!(cwd_host_base(|| Some("/private/tmp/fa".into()), Some("/sysroot"), false).as_deref(), Some("/private/tmp/fa"));
        assert_eq!(cwd_host_base(|| Some("/sysroot/sub".into()), Some("/sysroot"), false), None);
        assert_eq!(cwd_host_base(|| Some("/private/tmp/fa".into()), None, true).as_deref(), Some("/private/tmp/fa"));
        // probe_from_cwd: relative again below the base, else as it came
        assert_eq!(probe_from_cwd("/private/tmp/secret/d/l/../f", Some("/private/tmp/secret/d")), "l/../f");
        assert_eq!(probe_from_cwd("/private/tmp/secret/d", Some("/private/tmp/secret/d")), "/private/tmp/secret/d");
        assert_eq!(probe_from_cwd("/private/tmp/secret/dx/f", Some("/private/tmp/secret/d")), "/private/tmp/secret/dx/f");
        assert_eq!(probe_from_cwd("/a/b", None), "/a/b");
        // joined_rewritable: H v7's rule for a recorded base; the limit for a new one
        let long = "/".to_string() + &"a".repeat(MAXPATHLEN);
        assert!(joined_rewritable(true, "x", &long));
        assert!(!joined_rewritable(false, "x", &long));
        assert!(!joined_rewritable(false, "x", &long[..MAXPATHLEN]), "exactly MAXPATHLEN is too long");
        assert!(joined_rewritable(false, "x", &long[..MAXPATHLEN - 1]));
        // cwd_is_base: plain calls and AT_FDCWD; a real descriptor is not
        assert!(cwd_is_base(false, 3) && cwd_is_base(true, libc::AT_FDCWD) && !cwd_is_base(true, 3));
        // chdir_prefix: the guest path's physical prefix, else the record the name was left from
        let r = CwdRec { key: (1, 2), prefix: "/System/Library/dyld", rel: String::new() };
        let pre = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld";
        assert_eq!(chdir_prefix(Some("/System/Cryptexes/OS/System/Library/dyld/sub"), None), Some(pre));
        assert_eq!(chdir_prefix(Some("/System/Library/dyld"), Some(&r)), Some("/System/Library/dyld"));
        assert_eq!(chdir_prefix(Some("/System/Library"), None), None, "climbed out of the prefix: no record");
        assert_eq!(chdir_prefix(None, Some(&r)), Some("/System/Library/dyld"));
        assert_eq!(chdir_prefix(None, None), None);
        // physical_prefix
        assert_eq!(crate::paths::physical_prefix(pre), Some(pre));
        assert_eq!(crate::paths::physical_prefix("/System/Library/dyld/x"), Some("/System/Library/dyld"));
        assert_eq!(crate::paths::physical_prefix("/System/Library/dyldx"), None);
        assert_eq!(crate::paths::physical_prefix("/tmp/a"), None);
    }

    #[test]
    fn hv9_a_forwarded_prefix_is_taken_with_the_childs_own_place() {
        let sys = Some("/sysroot");
        let pre = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld";
        // the parent's prefix, the child's own place below the sysroot (a spawn chdir file action inside it, row J5)
        let got = adopt_arg("/System/Cryptexes/OS/System/Library/dyld", Some((7, 9)), || Some("/sysroot/child-subdir".into()), sys);
        assert_eq!(got.map(|r| r.guest()).as_deref(), Some(&*format!("{pre}/child-subdir")));
        // a child that a file action took out of the sysroot, or a forward with no prefix, takes nothing
        assert_eq!(adopt_arg("/System/Library/dyld", Some((7, 9)), || Some("/private/tmp/x".into()), sys), None);
        assert_eq!(adopt_arg("/private/tmp/x", Some((7, 9)), || panic!("asked the host"), sys), None);
        assert_eq!(adopt_arg("relative", Some((7, 9)), || panic!("asked the host"), sys), None);
        assert_eq!(adopt_arg("/System/Library/dyld", None, || Some("/sysroot".into()), sys), None);
    }

    #[test]
    fn hv9_every_path_slot_is_a_call_bsd_read_does_not_already_take() {
        for n in 0..600u64 {
            let slots = escape_slots(n);
            if !slots.is_empty() {
                assert_eq!(path_arg(n), (None, false), "call {n} is in both tables");
                for &(i, f) in slots {
                    assert!(i < 8 && f.is_none_or(|f| f < 8 && f != i), "call {n}");
                }
            }
        }
        assert_eq!(escape_slots(10), &[(0, None)]);
        assert_eq!(escape_slots(128), &[(0, None), (1, None)]);
        assert_eq!(escape_slots(57), &[(1, None)], "symlink's first argument is the link's text");
        assert_eq!(escape_slots(474), &[(2, Some(1))]);
        assert_eq!(escape_slots(472), &[(1, Some(0))]);
        assert_eq!(escape_slots(465), &[(1, Some(0)), (3, Some(2))]);
        assert!(escape_slots(5).is_empty() && escape_slots(12).is_empty() && escape_slots(3).is_empty());
    }

    #[test]
    fn hv10_an_unknown_sysroot_root_refuses_a_climb_and_never_reads_as_none() {
        let root = (9, 9);
        let at = |p: &str| if p == "." { Some(root) } else { None };
        assert_eq!(escape_closed("../x", Some(root), at), Ok(Some(0)));
        assert_eq!(escape_closed("../x", None, at), Err(()), "unknown is closed");
        assert_eq!(escape_closed("x", None, |_| panic!("stat")), Ok(None), "no `..`: nothing to rule out");
        assert_eq!(escape_closed("/abs/../x", None, |_| panic!("stat")), Ok(None));
        assert!(has_dotdot("a/../b") && has_dotdot("..") && !has_dotdot("a/..b") && !has_dotdot("/a/.."));
    }

    #[test]
    fn hv10_the_root_is_known_only_when_its_two_readings_agree() {
        assert_eq!(root_decision(Some((1, 1)), Some((1, 1))), Some((1, 1)));
        assert_eq!(root_decision(Some((1, 1)), None), Some((1, 1)), "a path made unsearchable: the startup identity");
        assert_eq!(root_decision(None, Some((1, 2))), Some((1, 2)));
        assert_eq!(root_decision(Some((1, 1)), Some((1, 2))), None, "replaced");
        assert_eq!(root_decision(None, None), None);
        // beneath: RESOLVE_BENEATH only, never AT_FDONLY
        let mut a = [0u64; 8];
        a[3] = AT_FDONLY;
        assert!(!beneath(469, &a));
        a[3] = AT_RESOLVE_BENEATH;
        assert!(beneath(469, &a) && beneath(467, &a) && !beneath(472, &a));
        a[2] = AT_RESOLVE_BENEATH;
        assert!(beneath(472, &a));
        a[1] = O_RESOLVE_BENEATH;
        assert!(beneath(5, &a));
        // the sysroot's root itself keeps its absolute form
        assert_eq!(reentry_relative("/sysroot", 2, "/sysroot"), None);
        assert_eq!(reentry_relative("/sysroot/", 0, "/sysroot"), None);
        // an unresolved walk is not a place below the sysroot (measured, row LK1)
        assert_eq!(reentry_relative("/sysroot/../dyld/x", 0, "/sysroot"), None);
        assert_eq!(reentry_relative("/sysroot/a/./b", 0, "/sysroot"), None);
    }

    #[test]
    fn hv10_cwd_depth_counts_the_physical_climb_to_the_root() {
        let root = (9, 9);
        assert_eq!(cwd_depth(root, |p| if p == "." { Some(root) } else { None }), Some(0));
        assert_eq!(cwd_depth(root, |p| match p { "." => Some((1, 1)), ".." => Some((1, 2)), "../.." => Some(root), _ => None }), Some(2));
        assert_eq!(cwd_depth(root, |p| if p == "." { Some((1, 1)) } else { None }), None, "a failed stat");
        assert_eq!(cwd_depth(root, |_| Some((1, 1))), None, "never reached");
    }

    #[test]
    fn hv10_fit_target_goes_relative_inside_the_sysroot() {
        let sys = Some("/sysroot");
        assert_eq!(fit_target("/sysroot/f".into(), sys, || Some(2)).as_deref(), Ok("../../f"));
        assert_eq!(fit_target("/sysroot/f".into(), sys, || None).as_deref(), Ok("/sysroot/f"), "depth unknown: absolute");
        assert_eq!(fit_target("/System/Library/x".into(), sys, || panic!("depth")).as_deref(), Ok("/System/Library/x"));
        let long = format!("/sysroot/{}", "a".repeat(MAXPATHLEN));
        assert_eq!(fit_target(long.clone(), sys, || Some(0)), Err(libc::ENAMETOOLONG as u64));
        let long_out = format!("/elsewhere/{}", "a".repeat(MAXPATHLEN));
        assert_eq!(fit_target(long_out, sys, || Some(0)), Err(libc::ENAMETOOLONG as u64));
        let near = format!("/sysroot/{}", "b".repeat(MAXPATHLEN - 20));
        assert_eq!(fit_target(near, sys, || Some(1)).map(|r| r.starts_with("../b")), Ok(true), "too long absolute, fits relative");
    }

    #[test]
    fn hv10_lead_probe_keeps_the_names_length() {
        for (name, off) in [("l/../dyld/f", 2), ("a/b/../../../x", 10), ("../x", 0), ("q/..", 2), ("q/../y", 2)] {
            let p = lead_probe(name, off);
            assert_eq!(p.len(), name.len(), "{name}");
            assert!(p.starts_with(&name[..off]) && !p[off..].contains(".."), "{name} -> {p}");
        }
        assert_eq!(lead_probe("l/../d", 2), "l/././");
        assert_eq!(lead_probe("l/../dy", 2), "l/././.");
        assert_eq!(lead_probe("l/..", 2), "l/./");
        assert_eq!(lead_walk("../x", 0, Ids::Effective), Ok(()), "no leading part: nothing to walk");
    }

    #[test]
    fn hv10_accessx_names_are_read_and_rewritten_in_a_copy() {
        // two descriptors (the first offset, 32, ends the table), names at 32 and 42
        let mut t = vec![0u8; 48];
        t[0..4].copy_from_slice(&32u32.to_ne_bytes());
        t[16..20].copy_from_slice(&42u32.to_ne_bytes());
        t.truncate(32);
        t.extend_from_slice(b"../dyld/a\0");
        t.extend_from_slice(b"b\0");
        let seen = std::cell::RefCell::new(Vec::new());
        let got = accessx_rewrite(&t, |n| { seen.borrow_mut().push(n.to_string()); Ok((n == "../dyld/a").then(|| "a".to_string())) }).unwrap().unwrap();
        assert_eq!(*seen.borrow(), vec!["../dyld/a".to_string(), "b".to_string()]);
        // laid out again: the table still ends at 32 (the first offset), then "a", then "b"
        assert_eq!(&got[0..4], &32u32.to_ne_bytes(), "the first offset is the table's end and must not move");
        assert_eq!(&got[4..16], &t[4..16]);
        assert_eq!(&got[16..20], &34u32.to_ne_bytes());
        assert_eq!(&got[32..], b"a\0b\0");
        // an offset of 0 (repeat the previous name) stays 0
        let mut t3 = vec![0u8; 48];
        t3[0..4].copy_from_slice(&48u32.to_ne_bytes());
        t3.extend_from_slice(b"../x\0");
        let got3 = accessx_rewrite(&t3, |_| Ok(Some("y".into()))).unwrap().unwrap();
        assert_eq!(&got3[0..4], &48u32.to_ne_bytes());
        assert_eq!(&got3[16..20], &0u32.to_ne_bytes());
        assert_eq!(&got3[32..36], &0u32.to_ne_bytes());
        assert_eq!(&got3[48..], b"y\0");
        // nothing to rewrite: no copy; a refusal is passed on
        assert_eq!(accessx_rewrite(&t, |_| Ok(None)), Ok(None));
        assert_eq!(accessx_rewrite(&t, |_| Err(2)), Err(2));
        // not a table XNU would read: left alone
        assert_eq!(accessx_rewrite(&[0u8; 8], |_| panic!("read")), Ok(None));
        assert_eq!(accessx_rewrite(&[0u8; 20], |_| panic!("read")), Ok(None), "the first offset 0");
        assert_eq!(accessx_rewrite(&{ let mut b = t.clone(); b.push(b'x'); b }, |_| panic!("read")), Ok(None), "no NUL at the end");
        assert_eq!(accessx_rewrite(&{ let mut b = t.clone(); b[16..20].copy_from_slice(&16u32.to_ne_bytes()); b }, |_| panic!("read")), Ok(None), "an offset into its own descriptor");
        // Names out of order and an unaligned first offset are read as XNU reads them
        let mut t2 = vec![0u8; 32];
        t2[0..4].copy_from_slice(&34u32.to_ne_bytes());
        t2[16..20].copy_from_slice(&32u32.to_ne_bytes());
        t2.extend_from_slice(b"x\0../dyld/m\0");
        let seen2 = std::cell::RefCell::new(Vec::new());
        let got2 = accessx_rewrite(&t2, |n| { seen2.borrow_mut().push(n.to_string()); Ok((n == "../dyld/m").then(|| "m".to_string())) }).unwrap().unwrap();
        assert_eq!(*seen2.borrow(), vec!["../dyld/m".to_string(), "x".to_string()]);
        assert_eq!(&got2[32..], b"m\0x\0");
        assert_eq!(&got2[0..4], &32u32.to_ne_bytes());
        assert_eq!(&got2[16..20], &34u32.to_ne_bytes());
        // growing past 16 KiB is refused
        let big = "x".repeat(ACCESSX_MAX_TABLESIZE);
        assert_eq!(accessx_rewrite(&t, |_| Ok(Some(big.clone()))), Err(libc::E2BIG as u64));
    }

    fn a(i: usize, v: u64) -> [u64; 8] {
        let mut x = [0u64; 8];
        x[i] = v;
        x
    }

    #[test]
    fn descriptor_bound_flags() {
        // fstatat, fstatat64: AT_FDONLY and AT_RESOLVE_BENEATH bind; AT_SYMLINK_NOFOLLOW does not
        for n in [469, 470] {
            assert!(descriptor_bound(n, &a(3, AT_FDONLY)));
            assert!(descriptor_bound(n, &a(3, AT_RESOLVE_BENEATH)));
            assert!(descriptor_bound(n, &a(3, AT_FDONLY | 0x20)));
            assert!(!descriptor_bound(n, &a(3, 0x20)));
            assert!(!descriptor_bound(n, &a(3, 0)));
        }
        for n in [463, 464] {
            assert!(descriptor_bound(n, &a(2, O_RESOLVE_BENEATH)));
            assert!(!descriptor_bound(n, &a(2, 0x0201))); // O_WRONLY | O_CREAT
            // must-fail control: AT_FDONLY's value in openat's oflag is O_TRUNC, not a binding
            assert!(!descriptor_bound(n, &a(2, AT_FDONLY)));
            // a flag in the wrong argument slot does not count
            assert!(!descriptor_bound(n, &a(3, O_RESOLVE_BENEATH)));
        }
        assert!(descriptor_bound(466, &a(3, AT_RESOLVE_BENEATH)));
        assert!(!descriptor_bound(466, &a(3, 0x10))); // AT_EACCESS
        assert!(!descriptor_bound(466, &a(3, AT_FDONLY))); // fd-only means nothing to faccessat
        assert!(descriptor_bound(476, &a(5, FSOPT_RESOLVE_BENEATH)));
        assert!(!descriptor_bound(476, &a(5, 0x1))); // FSOPT_NOFOLLOW
        assert!(!descriptor_bound(476, &a(3, FSOPT_RESOLVE_BENEATH)));
        assert!(!descriptor_bound(473, &[u64::MAX; 8])); // readlinkat has no flags
        assert!(!descriptor_bound(5, &[u64::MAX; 8])); // not an *at call
    }

    #[test]
    fn nofollow_any_flags() {
        for n in [463, 464] {
            assert!(nofollow_any(n, &a(2, O_NOFOLLOW_ANY)));
            assert!(nofollow_any(n, &a(2, O_NOFOLLOW_ANY | 0x0201)));
            // must-fail controls: AT_SYMLINK_NOFOLLOW_ANY's value in openat's oflag is O_EXCL,
            // O_NOFOLLOW (0x100) is not NOFOLLOW_ANY, and the right bit in the wrong slot
            assert!(!nofollow_any(n, &a(2, AT_SYMLINK_NOFOLLOW_ANY)));
            assert!(!nofollow_any(n, &a(2, 0x0100)));
            assert!(!nofollow_any(n, &a(3, O_NOFOLLOW_ANY)));
        }
        for n in [466, 469, 470] {
            assert!(nofollow_any(n, &a(3, AT_SYMLINK_NOFOLLOW_ANY)));
            assert!(!nofollow_any(n, &a(3, 0x0020))); // AT_SYMLINK_NOFOLLOW
            assert!(!nofollow_any(n, &a(3, O_NOFOLLOW_ANY))); // the openat value in an AT_ flag
            assert!(!nofollow_any(n, &a(2, AT_SYMLINK_NOFOLLOW_ANY))); // wrong slot
        }
        assert!(nofollow_any(476, &a(5, FSOPT_NOFOLLOW_ANY)));
        assert!(!nofollow_any(476, &a(5, 0x1))); // FSOPT_NOFOLLOW
        assert!(!nofollow_any(476, &a(3, FSOPT_NOFOLLOW_ANY))); // wrong slot
        assert!(!nofollow_any(473, &[u64::MAX; 8])); // readlinkat has no flags
        assert!(!nofollow_any(5, &[u64::MAX; 8])); // not an *at call
        assert!(!nofollow_any(463, &a(2, 0)));
    }

    #[test]
    fn keep_descriptor_needs_inside_only_for_nofollow_any() {
        let asked = std::cell::Cell::new(false);
        let yes = || { asked.set(true); true };
        // a NOFOLLOW_ANY call keeps its descriptor exactly when inside() says so
        assert!(keep_descriptor(463, &a(2, O_NOFOLLOW_ANY), yes));
        assert!(asked.get());
        assert!(!keep_descriptor(463, &a(2, O_NOFOLLOW_ANY), || false));
        assert!(keep_descriptor(469, &a(3, AT_SYMLINK_NOFOLLOW_ANY), || true));
        assert!(!keep_descriptor(469, &a(3, AT_SYMLINK_NOFOLLOW_ANY), || false));
        assert!(keep_descriptor(466, &a(3, AT_SYMLINK_NOFOLLOW_ANY), || true));
        assert!(keep_descriptor(476, &a(5, FSOPT_NOFOLLOW_ANY), || true));
        assert!(!keep_descriptor(476, &a(5, FSOPT_NOFOLLOW_ANY), || false));
        // descriptor-bound flags keep it whatever inside() says, and do not ask it
        let asked = std::cell::Cell::new(false);
        assert!(keep_descriptor(469, &a(3, AT_FDONLY), || { asked.set(true); false }));
        assert!(keep_descriptor(463, &a(2, O_RESOLVE_BENEATH), || { asked.set(true); false }));
        assert!(!asked.get(), "inside() asked for a call descriptor_bound already keeps");
        // must-fail control: no flag at all never keeps it, and inside() is not asked
        let asked = std::cell::Cell::new(false);
        assert!(!keep_descriptor(463, &a(2, 0), || { asked.set(true); true }));
        assert!(!keep_descriptor(469, &a(3, 0x0020), || { asked.set(true); true }));
        assert!(!asked.get(), "inside() asked for a call with no binding flag");
    }

    #[test]
    fn stays_beneath_refuses_a_climb_out() {
        for n in ["x", "a/b", "./x", "a/../x", "a/./../b", "a/b/../../c", "", "a//b"] {
            assert!(stays_beneath(n), "{n} stays beneath");
        }
        // must-fail controls: out of the cache dir and back in, and a climb hidden behind a descent
        for n in ["..", "../x", "../dyld/dyld_shared_cache_arm64e", "a/../../x", "./../x", "a/b/../../../c"] {
            assert!(!stays_beneath(n), "{n} climbs out");
        }
    }

    #[test]
    fn nofollow_any_inside_needs_the_host_descriptor_to_be_the_redirect() {
        use std::os::unix::io::AsRawFd;
        let base = std::env::temp_dir().join(format!("maclator-e2v2-test-{}", std::process::id()));
        let (sys, other) = (base.join("sys"), base.join("other"));
        std::fs::create_dir_all(&sys).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let link = base.join("sys-link");
        std::os::unix::fs::symlink(&sys, &link).unwrap();
        let s = std::fs::File::open(&sys).unwrap();
        let o = std::fs::File::open(&other).unwrap();
        let g = || Some("/System/Library/dyld".to_string());
        let to = |p: &std::path::Path| { let p = p.to_string_lossy().into_owned(); move |_: &str| Some(p.clone()) };
        // the descriptor is the redirected directory: by its own path, or through a symlink to it
        assert!(nofollow_any_inside("dyld_shared_cache_arm64e", s.as_raw_fd(), g(), to(&sys)).is_some());
        assert!(nofollow_any_inside("dyld_shared_cache_arm64e", s.as_raw_fd(), g(), to(&link)).is_some());
        assert!(nofollow_any_inside("a/../x", s.as_raw_fd(), g(), to(&sys)).is_some());
        // The guest path is redirected, but the host descriptor sits elsewhere
        assert!(nofollow_any_inside("dyld_shared_cache_arm64e", o.as_raw_fd(), g(), to(&sys)).is_none());
        // A name that climbs out of the descriptor
        assert!(nofollow_any_inside("../sys/x", s.as_raw_fd(), g(), to(&sys)).is_none());
        // an absolute name, no guest path, a guest path that is not redirected, a closed fd
        assert!(nofollow_any_inside("/x", s.as_raw_fd(), g(), to(&sys)).is_none());
        assert!(nofollow_any_inside("x", s.as_raw_fd(), None, to(&sys)).is_none());
        assert!(nofollow_any_inside("x", s.as_raw_fd(), g(), |_: &str| None).is_none());
        assert!(nofollow_any_inside("x", -1, g(), to(&sys)).is_none());
        drop(s);
        drop(o);
        std::fs::remove_file(&link).unwrap();
        std::fs::remove_dir(&sys).unwrap();
        std::fs::remove_dir(&other).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    /// The piece size upstream's guest_cstr reads in (its PIECE): no read crosses one.
    const NAME_READ_CHUNK: u64 = 4096;

    /// A fake guest memory: `bytes` at address `base`, readable below `limit` only. It records
    /// every read that crosses a NAME_READ_CHUNK boundary (guest_cstr must make none).
    struct Fake { base: u64, bytes: Vec<u8>, limit: u64, crossed: std::cell::Cell<u32>, reads: std::cell::Cell<u32> }
    impl Fake {
        fn new(base: u64, bytes: &[u8], limit: u64) -> Fake {
            Fake { base, bytes: bytes.to_vec(), limit, crossed: 0.into(), reads: 0.into() }
        }
        fn read(&self, at: u64, buf: &mut [u8]) -> bool {
            self.reads.set(self.reads.get() + 1);
            let end = at + buf.len() as u64;
            if buf.len() > 0 && at / NAME_READ_CHUNK != (end - 1) / NAME_READ_CHUNK {
                self.crossed.set(self.crossed.get() + 1);
            }
            if at < self.base || end > self.limit || end > self.base + self.bytes.len() as u64 {
                return false;
            }
            buf.copy_from_slice(&self.bytes[(at - self.base) as usize..(end - self.base) as usize]);
            true
        }
    }

    #[test]
    fn guest_cstr_reads_a_name_chunk_by_chunk_up_to_a_fault() {
        let c = NAME_READ_CHUNK;
        // a name that ends with its NUL on the last byte before an unreadable chunk
        let mut m = vec![b'x'; 2 * c as usize];
        let at = 0x10000 + c - 5;
        m[(c - 5) as usize..c as usize].copy_from_slice(b"dyld\0");
        let f = Fake::new(0x10000, &m, 0x10000 + c);
        assert_eq!(guest_cstr(at, 8192, |a, b| f.read(a, b)).as_deref(), Some("dyld"));
        // must-fail control: the same name with no NUL runs into the unreadable chunk
        m[(c - 1) as usize] = b'1';
        let f = Fake::new(0x10000, &m, 0x10000 + c);
        assert_eq!(guest_cstr(at, 8192, |a, b| f.read(a, b)), None);
        // and readable all the way, it is read across the boundary in two reads, neither crossing it
        let f = Fake::new(0x10000, &m, 0x10000 + 2 * c);
        assert_eq!(guest_cstr(at, 8192, |a, b| f.read(a, b)), None, "no NUL in the whole memory");
        m[c as usize + 3] = 0;
        let f = Fake::new(0x10000, &m, 0x10000 + 2 * c);
        assert_eq!(guest_cstr(at, 8192, |a, b| f.read(a, b)).as_deref(), Some("dyld1xxx"));
        assert_eq!(f.reads.get(), 2);
        assert_eq!(f.crossed.get(), 0, "a read crossed a chunk boundary");
        // an empty name; an unreadable first byte; a NUL that comes only after `max` bytes
        // (a fake chunk is readable whole or not at all, as a real page is)
        let f = Fake::new(0x20000, &[0u8; 4096], 0x21000);
        assert_eq!(guest_cstr(0x20000, 8192, |a, b| f.read(a, b)).as_deref(), Some(""));
        assert_eq!(guest_cstr(0x1000, 8192, |a, b| f.read(a, b)), None);
        let f = Fake::new(0x20000, b"aaaaaaaaaaaaaaaa\0", 0x30000);
        assert_eq!(guest_cstr(0x20000, 16, |a, b| f.read(a, b)), None, "16 bytes and no NUL: too long");
        assert_eq!(guest_cstr(0x20000, 17, |a, b| f.read(a, b)).as_deref(), Some("aaaaaaaaaaaaaaaa"));
    }

    /// Two host pages, the second PROT_NONE; the name `tail` placed to end on the first page's last
    /// byte. Returns (mapping, page size, address of the name).
    fn guard_page(tail: &[u8]) -> (*mut libc::c_void, usize, u64) {
        let pg = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let m = unsafe { libc::mmap(std::ptr::null_mut(), 2 * pg, libc::PROT_READ | libc::PROT_WRITE,
                                    libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) };
        assert_ne!(m, libc::MAP_FAILED);
        let at = m as u64 + (pg - tail.len()) as u64;
        unsafe {
            std::ptr::copy_nonoverlapping(tail.as_ptr(), at as *mut u8, tail.len());
            assert_eq!(libc::mprotect((m as u64 + pg as u64) as *mut libc::c_void, pg, libc::PROT_NONE), 0);
        }
        (m, pg, at)
    }

    #[test]
    fn guest_name_never_faults_on_guest_memory() {
        // positive control: an ordinary name
        let c = std::ffi::CString::new("/System/Library/dyld/x").unwrap();
        assert_eq!(guest_name(c.as_ptr() as u64).as_deref(), Some("/System/Library/dyld/x"));
        // a name ending, NUL and all, just before a PROT_NONE page; then one that runs into it
        let (m, pg, at) = guard_page(b"dyld_shared_cache_arm64e\0");
        assert_eq!(guest_name(at).as_deref(), Some("dyld_shared_cache_arm64e"));
        let (m2, _, at2) = guard_page(b"dyld_sha");
        assert_eq!(guest_name(at2), None);
        assert_eq!(guest_name(m2 as u64 + pg as u64), None, "the PROT_NONE page itself");
        // address 0x1000: __PAGEZERO (4 GiB on x86_64), as in rows U5-U7
        assert_eq!(guest_name(0x1000), None);
        unsafe {
            libc::munmap(m, 2 * pg);
            libc::munmap(m2, 2 * pg);
        }
    }

    #[test]
    fn call_name_reads_only_what_xnu_reads() {
        let asked = std::cell::Cell::new(false);
        let read = |_: u64| { asked.set(true); Some("x".to_string()) };
        // fstatat and fstatat64 with AT_FDONLY: the name is not read, whatever else is set
        for n in [469, 470] {
            let mut x = a(3, AT_FDONLY);
            x[1] = 0x1000;
            assert_eq!(call_name(n, &x, 1, read), None);
            x[3] = AT_FDONLY | AT_SYMLINK_NOFOLLOW_ANY | 0x80;
            assert_eq!(call_name(n, &x, 1, read), None);
        }
        assert!(!asked.get(), "an AT_FDONLY name was read");
        // NULL is not read either
        assert_eq!(call_name(470, &[0; 8], 1, read), None);
        assert!(!asked.get(), "a NULL name was read");
        // must-fail controls: 0x400 means nothing of the kind to openat (O_TRUNC) or faccessat
        let mut x = a(2, AT_FDONLY);
        x[1] = 0x1000;
        assert_eq!(call_name(463, &x, 1, read).as_deref(), Some("x"));
        assert!(asked.get(), "openat's name was not read");
        let asked2 = std::cell::Cell::new(false);
        let mut x = a(3, AT_FDONLY);
        x[1] = 0x1000;
        assert_eq!(call_name(466, &x, 1, |_| { asked2.set(true); None }), None);
        assert!(asked2.get(), "faccessat's name was not read");
        // a plain call's name, in args[0]
        assert_eq!(call_name(5, &a(0, 0x1000), 0, |p| Some(format!("{p:#x}"))).as_deref(), Some("0x1000"));
    }

    #[test]
    fn bsd_passes_an_unreadable_name_to_the_host_kernel() {
        // Real calls: the host kernel answers what XNU answers natively (rows U5-U7, and the
        // EINVAL-before-the-name case). Unpatched, every one of these crashed the process.
        use std::os::unix::io::AsRawFd;
        let base = std::env::temp_dir().join(format!("maclator-g-test-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("f"), b"").unwrap();
        let d = std::fs::File::open(&base).unwrap();
        let fd = d.as_raw_fd() as u64;
        let mut st = [0u8; 512];
        let sb = st.as_mut_ptr() as u64;
        let call = |n: u64, args: [u64; 8]| -> Result<u64, u64> {
            let mut cpu = Cpu::new();
            bsd(&mut cpu, n, args);
            if cpu.cf == 0 { Ok(cpu.x[0]) } else { Err(cpu.x[0]) }
        };
        let e = |v: i32| -> Result<u64, u64> { Err(v as u64) };
        // positive control: a readable name through the same descriptor
        let f = std::ffi::CString::new("f").unwrap();
        let r = call(463, [fd, f.as_ptr() as u64, 0, 0, 0, 0, 0, 0]);
        assert!(r.is_ok(), "openat(fd, \"f\") {r:?}");
        unsafe { libc::close(r.unwrap() as i32) };
        assert!(call(470, [fd, f.as_ptr() as u64, sb, 0, 0, 0, 0, 0]).is_ok());
        // U5-U7 on address 0x1000
        assert!(call(470, [fd, 0x1000, sb, AT_FDONLY, 0, 0, 0, 0]).is_ok(), "U5: AT_FDONLY ignores the name");
        assert!(call(469, [fd, 0x1000, sb, AT_FDONLY, 0, 0, 0, 0]).is_ok(), "fstatat AT_FDONLY");
        assert_eq!(call(470, [fd, 0x1000, sb, 0, 0, 0, 0, 0]), e(libc::EFAULT), "U6");
        assert_eq!(call(463, [fd, 0x1000, 0, 0, 0, 0, 0, 0]), e(libc::EFAULT), "U7");
        assert_eq!(call(466, [fd, 0x1000, 0, 0, 0, 0, 0, 0]), e(libc::EFAULT), "faccessat");
        assert_eq!(call(5, [0x1000, 0, 0, 0, 0, 0, 0, 0]), e(libc::EFAULT), "open");
        // XNU checks fstatat's flags before the name: an invalid one is EINVAL, not EFAULT
        assert_eq!(call(470, [fd, 0x1000, sb, 0x80, 0, 0, 0, 0]), e(libc::EINVAL), "AT_REMOVEDIR");
        // a name running into a PROT_NONE page; 1024 bytes with no NUL before one
        let (m, pg, at) = guard_page(b"dyld_sha");
        assert_eq!(call(463, [fd, at, 0, 0, 0, 0, 0, 0]), e(libc::EFAULT), "into the guard page");
        let (m2, _, at2) = guard_page(&[b'a'; 1024]);
        assert_eq!(call(463, [fd, at2, 0, 0, 0, 0, 0, 0]), e(libc::ENAMETOOLONG), "1024 bytes, no NUL");
        unsafe {
            libc::munmap(m, 2 * pg);
            libc::munmap(m2, 2 * pg);
        }
        drop(d);
        std::fs::remove_file(base.join("f")).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn hv6_a_redirected_climb_out_records_the_hosts_truth() {
        // A mutant that survived an earlier test set, sysroot-gate-true: bsd_read with a sysroot set, on W3's name. The name
        // starts under a cache prefix, climbs out of it, and stops at the firmlink
        // /System/Volumes/Data/Applications, whose `..` is /. The host opens /, so the descriptor records
        // "/" (F_GETPATH); recording the guest's spelling (the mutant) keeps ".../Applications/..". Skipped
        // where that firmlink is absent. This sets the process's sysroot: no other test uses a cache prefix.
        let a = "/System/Volumes/Data/Applications";
        let root = std::fs::metadata("/").unwrap();
        use std::os::unix::fs::MetadataExt;
        if !std::fs::metadata(format!("{a}/..")).is_ok_and(|m| m.ino() == root.ino() && m.dev() == root.dev()) {
            return;
        }
        let sys = std::env::temp_dir().join(format!("maclator-hv6-sys-{}", std::process::id()));
        std::fs::create_dir_all(&sys).unwrap();
        crate::paths::set_cache_dir(&sys.to_string_lossy());
        let name = std::ffi::CString::new(format!("/System/Library/dyld/../../../{}/..", &a[1..])).unwrap();
        let mut cpu = Cpu::new();
        let w = bsd_read(&mut cpu, 5, [name.as_ptr() as u64, libc::O_RDONLY as u64 | libc::O_DIRECTORY as u64, 0, 0, 0, 0, 0, 0], guest_name);
        assert_eq!(w, None);
        assert_eq!(cpu.cf, 0, "open failed: {}", cpu.x[0]);
        let fd = cpu.x[0] as i32;
        let st = unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            assert_eq!(libc::fstat(fd, &mut st), 0);
            st
        };
        assert!(st.st_ino as u64 == root.ino() && st.st_dev as u64 == root.dev(), "the host opened / (control)");
        let recorded = FD_VPATHS.lock().unwrap().as_ref().and_then(|m| m.get(&fd).cloned());
        let mut c2 = Cpu::new();
        bsd(&mut c2, 6, [fd as u64, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(recorded.as_deref(), Some("/"), "must-fail control: the guest's spelling recorded for a climb-out");
        assert!(FD_VPATHS.lock().unwrap().as_ref().is_none_or(|m| !m.contains_key(&fd)), "close removed the record");
        std::fs::remove_dir(&sys).unwrap();
    }

    #[test]
    fn unread_name_warning_needs_a_host_success() {
        let w = unread_name_warning(463, 0x1000, true).unwrap();
        assert!(w.starts_with("[maclator] warning: #463 name at 0x1000 "), "{w}");
        assert_eq!(unread_name_warning(463, 0x1000, false), None, "must-fail control: the host refused it");
    }

    #[test]
    fn bsd_warns_when_the_host_read_what_guest_memory_refused() {
        // A fake memory that refuses every read, in front of a name the kernel can read: the shape
        // of a disagreement between guestmem::read and the kernel (rows U15 and U16 make a real one).
        use std::os::unix::io::AsRawFd;
        let base = std::env::temp_dir().join(format!("maclator-g2-test-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("f"), b"").unwrap();
        let d = std::fs::File::open(&base).unwrap();
        let fd = d.as_raw_fd() as u64;
        let mut st = [0u8; 512];
        let sb = st.as_mut_ptr() as u64;
        let f = std::ffi::CString::new("f").unwrap();
        let fp = f.as_ptr() as u64;
        let refuses = Fake::new(0x20000, b"", 0x20000);
        let refused = |p: u64| guest_cstr(p, GUEST_NAME_MAX, |a, b| refuses.read(a, b));
        let call = |n: u64, args: [u64; 8], read: &dyn Fn(u64) -> Option<String>| {
            let mut cpu = Cpu::new();
            let w = bsd_read(&mut cpu, n, args, |p| read(p));
            (cpu.cf == 0, cpu.x[0], w)
        };
        // the host opens it, read or not; only the refused read warns
        let (ok, r, w) = call(463, [fd, fp, 0, 0, 0, 0, 0, 0], &refused);
        assert!(ok, "the host call failed: {r}");
        unsafe { libc::close(r as i32) };
        assert!(w.is_some_and(|w| w.contains("#463")), "a refused name the host read gave no warning");
        assert!(refuses.reads.get() > 0, "the fake memory was not asked");
        // must-fail controls: a name that reads; a host call that fails; AT_FDONLY, which reads nothing
        let (ok, r, w) = call(463, [fd, fp, 0, 0, 0, 0, 0, 0], &guest_name);
        assert!(ok);
        unsafe { libc::close(r as i32) };
        assert_eq!(w, None, "a name that read warned");
        let (ok, r, w) = call(463, [fd, 0x1000, 0, 0, 0, 0, 0, 0], &refused);
        assert_eq!((ok, r), (false, libc::EFAULT as u64));
        assert_eq!(w, None, "EFAULT warned");
        let (ok, _, w) = call(470, [fd, 0x1000, sb, AT_FDONLY, 0, 0, 0, 0], &refused);
        assert!(ok);
        assert_eq!(w, None, "AT_FDONLY warned");
        let (ok, _, w) = call(470, [fd, 0, sb, AT_FDONLY, 0, 0, 0, 0], &refused);
        assert!(ok);
        assert_eq!(w, None, "NULL with AT_FDONLY warned");
        drop(d);
        std::fs::remove_file(base.join("f")).unwrap();
        std::fs::remove_dir(&base).unwrap();
    }

    #[test]
    fn guest_name_reads_real_memory_across_a_4k_boundary() {
        // Two readable host pages (4 KiB or more each) and then a PROT_NONE one.
        let c = NAME_READ_CHUNK as usize;
        let pg = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let m = unsafe { libc::mmap(std::ptr::null_mut(), 3 * pg, libc::PROT_READ | libc::PROT_WRITE,
                                    libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) };
        assert_ne!(m, libc::MAP_FAILED);
        let guard = m as u64 + 2 * pg as u64;
        assert_eq!(unsafe { libc::mprotect(guard as *mut libc::c_void, pg, libc::PROT_NONE) }, 0);
        // a short name across a 4 KiB boundary, in the middle of readable memory
        let at = m as u64 + c as u64 - 4;
        unsafe { std::ptr::copy_nonoverlapping(b"dyld_shared_cache\0".as_ptr(), at as *mut u8, 18) };
        assert_eq!(guest_name(at).as_deref(), Some("dyld_shared_cache"));
        // longer than one chunk, its NUL the last byte before the PROT_NONE page: one read from the
        // start to GUEST_NAME_MAX would run into that page, and stopping at the first boundary would
        // lose the tail
        let len = c + 900;
        let at = guard - len as u64;
        unsafe {
            std::ptr::write_bytes(at as *mut u8, b'a', len - 1);
            *(guard as *mut u8).sub(1) = 0;
        }
        assert_ne!(at / NAME_READ_CHUNK, (guard - 1) / NAME_READ_CHUNK, "the name does not cross");
        assert_eq!(guest_name(at).map(|n| n.len()), Some(len - 1));
        // must-fail control: the same length with no NUL runs into the page
        unsafe { *(guard as *mut u8).sub(1) = b'a' };
        assert_eq!(guest_name(at), None);
        unsafe { libc::munmap(m, 3 * pg) };
    }

    #[test]
    fn sysctl_names_are_read_fault_safely() {
        // namelen bytes, no NUL needed, cut at the first NUL; never past the cap
        let ok = |at: u64, buf: &mut [u8]| crate::guestmem::read(at, buf);
        let c = b"hw.machine";
        assert_eq!(sysctl_name(c.as_ptr() as u64, 10, ok).as_deref(), Some("hw.machine"));
        assert_eq!(sysctl_name(c.as_ptr() as u64, 2, ok).as_deref(), Some("hw"));
        let z = b"hw.machine\0zz";
        assert_eq!(sysctl_name(z.as_ptr() as u64, 13, ok).as_deref(), Some("hw.machine"));
        assert_eq!(sysctl_name(0x1000, 10, ok), None);
        assert_eq!(sysctl_name(0, 10, |_: u64, _: &mut [u8]| panic!("read a NULL name")), None);
        assert_eq!(sysctl_name(c.as_ptr() as u64, 0, |_: u64, _: &mut [u8]| panic!("read an empty name")), None);
        assert_eq!(sysctl_name(c.as_ptr() as u64, MAXPATHLEN as u64, |_: u64, _: &mut [u8]| panic!("read past the cap")), None);
        // a name ending, with no NUL, just before a PROT_NONE page: XNU copies namelen bytes only
        let (m, pg, at) = guard_page(b"hw.machine");
        assert_eq!(sysctl_name(at, 10, ok).as_deref(), Some("hw.machine"));
        // must-fail control: one byte more runs into the page
        assert_eq!(sysctl_name(at, 11, ok), None);
        // A MIB of 2 to CTL_MAXNAME entries is read, fault-safely; any other length is not
        let mib: [i32; 2] = [6, 1];
        assert_eq!(sysctl_mib(mib.as_ptr() as u64, 2, ok), (Some(vec![6, 1]), false));
        let three: [i32; 3] = [6, 103, 109];
        assert_eq!(sysctl_mib(three.as_ptr() as u64, 3, ok), (Some(vec![6, 103, 109]), false));
        assert_eq!(sysctl_mib(0x1000, 2, ok), (None, true));
        assert_eq!(sysctl_mib(0x1000, 3, ok), (None, true));
        for n in [0, 1, CTL_MAXNAME + 1] {
            assert_eq!(sysctl_mib(0x1000, n, |_: u64, _: &mut [u8]| panic!("read a MIB of {n} entries")), (None, false));
        }
        // the second entry in the PROT_NONE page (probe row U7)
        let (gm, gpg, gat) = guard_page(&6i32.to_le_bytes());
        assert_eq!(sysctl_mib(gat, 2, ok), (None, true));
        assert_eq!(sysctl_mib(gat, 1, ok), (None, false), "must-fail control: a 1-entry MIB is the host's EINVAL, not read");
        unsafe { libc::munmap(gm, 2 * gpg) };
        assert_eq!(unread_sysctl_warning(274, 0x1000, true, false), None);
        assert!(unread_sysctl_warning(274, 0x1000, true, true).is_some());
        assert_eq!(unread_sysctl_warning(274, 0x1000, false, true), None);
        unsafe { libc::munmap(m, 2 * pg) };
    }

    #[test]
    fn sysctl_names_drop_one_terminal_dot_as_xnu() {
        // name2oid drops one terminal dot, after the
        // cut at the first NUL; measured natively on macOS 27.0.1, arm64
        let ok = |at: u64, buf: &mut [u8]| crate::guestmem::read(at, buf);
        let name = |b: &[u8], n: u64| sysctl_name(b.as_ptr() as u64, n, ok);
        assert_eq!(name(b"hw.machine.", 11).as_deref(), Some("hw.machine"));
        assert_eq!(name(b"hw.machine.", 10).as_deref(), Some("hw.machine"), "namelen not counting the dot");
        assert_eq!(name(b"hw.machine.\0", 12).as_deref(), Some("hw.machine"), "namelen counting the NUL");
        assert_eq!(name(b"hw.machine.\0junk", 16).as_deref(), Some("hw.machine"), "the dot is the last byte before the NUL");
        assert_eq!(name(b"hw.machine\0.", 12).as_deref(), Some("hw.machine"), "a dot after the NUL is not read");
        assert_eq!(name(b"hw.optional.mmx.", 16).as_deref(), Some("hw.optional.mmx"));
        assert_eq!(name(b"hw.optional.", 12).as_deref(), Some("hw.optional"));
        // must-fail controls: only one, and only at the end (native ENOENT for each)
        assert_eq!(name(b"hw.machine..", 12).as_deref(), Some("hw.machine."));
        assert_eq!(name(b".hw.machine", 11).as_deref(), Some(".hw.machine"));
        assert_eq!(name(b"hw..machine", 11).as_deref(), Some("hw..machine"));
        assert_eq!(name(b".", 1).as_deref(), Some(""));
        assert_eq!(name(b"hw.machine ", 11).as_deref(), Some("hw.machine "));
        assert_eq!(sysctl_override("hw.machine."), None, "the table itself is not widened");
        // real calls through bsd(274): the overrides answer, the doubled dot goes to the host (its ENOENT)
        let call = |n: u64, args: [u64; 8]| -> Result<u64, u64> {
            let mut cpu = Cpu::new();
            bsd(&mut cpu, n, args);
            if cpu.cf == 0 { Ok(cpu.x[0]) } else { Err(cpu.x[0]) }
        };
        let mut out = [0x5au8; 64];
        let mut len: u64 = 64;
        let o = out.as_mut_ptr() as u64;
        let l = &mut len as *mut u64 as u64;
        let hm = b"hw.machine.";
        assert!(call(274, [hm.as_ptr() as u64, 11, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!((&out[..6], len), (&b"arm64\0"[..], 6));
        len = 64;
        let mmx = b"hw.optional.mmx.";
        assert_eq!(call(274, [mmx.as_ptr() as u64, 16, o, l, 0, 0, 0, 0]), Err(libc::ENOTSUP as u64));
        len = 64;
        let pt = b"sysctl.proc_translated.";
        assert!(call(274, [pt.as_ptr() as u64, 23, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!((i32::from_le_bytes(out[..4].try_into().unwrap()), len), (0, 4));
        len = 64;
        let mm = b"hw.machine..";
        assert_eq!(call(274, [mm.as_ptr() as u64, 12, o, l, 0, 0, 0, 0]), Err(libc::ENOENT as u64));
        // control: a host sysctl with the dot is the host's, and answers
        len = 64;
        let ot = b"kern.ostype.";
        assert!(call(274, [ot.as_ptr() as u64, 12, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(&out[..7], b"Darwin\0");
    }

    #[test]
    fn bsd_passes_an_unreadable_sysctl_name_to_the_host_kernel() {
        // Real calls. Unpatched, each unreadable case crashed the process.
        let call = |n: u64, args: [u64; 8]| -> Result<u64, u64> {
            let mut cpu = Cpu::new();
            bsd(&mut cpu, n, args);
            if cpu.cf == 0 { Ok(cpu.x[0]) } else { Err(cpu.x[0]) }
        };
        let e = |v: i32| -> Result<u64, u64> { Err(v as u64) };
        let mut out = [0u8; 64];
        let mut len: u64 = 64;
        let o = out.as_mut_ptr() as u64;
        let l = &mut len as *mut u64 as u64;
        // positive controls: an override, an integer override, and a host sysctl
        let hm = b"hw.machine";
        assert!(call(274, [hm.as_ptr() as u64, 10, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(&out[..6], b"arm64\0");
        len = 64;
        let pt = b"sysctl.proc_translated";
        assert!(call(274, [pt.as_ptr() as u64, pt.len() as u64, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(len, 4, "K v4: a 4-byte integer, as native, in a 64-byte buffer");
        assert_eq!(i32::from_le_bytes(out[..4].try_into().unwrap()), 0);
        len = 64;
        let ot = b"kern.ostype";
        assert!(call(274, [ot.as_ptr() as u64, ot.len() as u64, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(&out[..7], b"Darwin\0");
        // unreadable names: the host kernel's EFAULT
        len = 64;
        assert_eq!(call(274, [0x1000, 10, o, l, 0, 0, 0, 0]), e(libc::EFAULT), "sysctlbyname at 0x1000");
        let (m, pg, at) = guard_page(b"hw.machine");
        assert_eq!(call(274, [at, 11, o, l, 0, 0, 0, 0]), e(libc::EFAULT), "into the guard page");
        // the same name, namelen stopping before the page: the override, as XNU reads it
        len = 64;
        assert!(call(274, [at, 10, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(&out[..6], b"arm64\0");
        // __sysctl
        let mib: [i32; 2] = [6, 1];
        len = 64;
        assert!(call(202, [mib.as_ptr() as u64, 2, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(&out[..6], b"arm64\0");
        let kost: [i32; 2] = [1, 1];
        len = 64;
        assert!(call(202, [kost.as_ptr() as u64, 2, o, l, 0, 0, 0, 0]).is_ok());
        assert_eq!(&out[..7], b"Darwin\0");
        assert_eq!(call(202, [0x1000, 2, o, l, 0, 0, 0, 0]), e(libc::EFAULT), "__sysctl MIB at 0x1000");
        assert_eq!(call(202, [0x1000, 3, o, l, 0, 0, 0, 0]), e(libc::EFAULT), "__sysctl 3-entry MIB at 0x1000");
        unsafe { libc::munmap(m, 2 * pg) };
    }

    #[test]
    fn sysctl_answers_are_written_fault_safely() {
        // Real calls through bsd(). Under K, each EFAULT case
        // below dereferenced the guest's oldp or oldlenp raw and crashed the process.
        let call = |n: u64, args: [u64; 8]| -> Result<u64, u64> {
            let mut cpu = Cpu::new();
            bsd(&mut cpu, n, args);
            if cpu.cf == 0 { Ok(cpu.x[0]) } else { Err(cpu.x[0]) }
        };
        let efault: Result<u64, u64> = Err(libc::EFAULT as u64);
        let enomem: Result<u64, u64> = Err(libc::ENOMEM as u64);
        let pg = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let m = unsafe { libc::mmap(std::ptr::null_mut(), 3 * pg, libc::PROT_READ | libc::PROT_WRITE,
                                    libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) };
        assert_ne!(m, libc::MAP_FAILED);
        let (buf, lenp, ro) = (m as u64, m as u64 + pg as u64, m as u64 + 2 * pg as u64);
        unsafe { *(ro as *mut u64) = 64 };
        unsafe { *((ro + 8) as *mut u64) = 2 }; // a short oldlen on the read-only page
        assert_eq!(unsafe { libc::mprotect(ro as *mut libc::c_void, pg, libc::PROT_READ) }, 0);
        let set_len = |v: u64| unsafe { *(lenp as *mut u64) = v };
        let len = || unsafe { *(lenp as *const u64) };
        let hm = b"hw.machine";
        let pt = b"sysctl.proc_translated";
        let mib: [i32; 2] = [6, 1];
        let calls: [(u64, u64, u64, u64); 3] = [
            (274, hm.as_ptr() as u64, 10, 6),
            (202, mib.as_ptr() as u64, 2, 6),
            (274, pt.as_ptr() as u64, pt.len() as u64, 4),
        ];
        for (n, name, nl, need) in calls {
            // positive control: an ordinary buffer gets the override's answer, not the host's (under
            // Rosetta the host says x86_64 and 1). It comes first: a call that missed the override
            // would take the bad pointers below to the host kernel, which natively never returns.
            set_len(64);
            assert!(call(n, [name, nl, buf, lenp, 0, 0, 0, 0]).is_ok(), "#{n} control");
            let got = unsafe { std::slice::from_raw_parts(buf as *const u8, 8) };
            if need == 6 {
                assert_eq!(&got[..6], b"arm64\0", "#{n} control: the override answered");
            } else {
                assert_eq!((len(), i32::from_le_bytes(got[..4].try_into().unwrap())), (4, 0), "#{n} control: the override answered");
            }
            // oldlenp unmapped, and read-only
            assert_eq!(call(n, [name, nl, buf, 0x1000, 0, 0, 0, 0]), efault, "#{n} oldlenp 0x1000");
            assert_eq!(call(n, [name, nl, buf, ro, 0, 0, 0, 0]), efault, "#{n} read-only oldlenp");
            assert_eq!(unsafe { *(ro as *const u64) }, 64);
            // oldp read-only: EFAULT, and oldlen is left as it was
            set_len(64);
            assert_eq!(call(n, [name, nl, ro + 64, lenp, 0, 0, 0, 0]), efault, "#{n} read-only oldp");
            assert_eq!(len(), 64);
            // oldp NULL: the length query
            set_len(64);
            assert!(call(n, [name, nl, 0, lenp, 0, 0, 0, 0]).is_ok(), "#{n} length query");
            assert_eq!(len(), need);
            // As measured natively (probe rows 6 to 8). oldp with
            // oldlenp NULL: ENOMEM, nothing written; both NULL: success
            let fill = || unsafe { std::ptr::write_bytes(buf as *mut u8, 0x5a, 64) };
            let untouched = || unsafe { std::slice::from_raw_parts(buf as *const u8, 64) }.iter().all(|&b| b == 0x5a);
            fill();
            assert_eq!(call(n, [name, nl, buf, 0, 0, 0, 0, 0]), enomem, "#{n} oldp, oldlenp NULL");
            assert!(untouched(), "#{n} oldlenp NULL: nothing written");
            assert!(call(n, [name, nl, 0, 0, 0, 0, 0, 0]).is_ok(), "#{n} oldp NULL, oldlenp NULL");
            // a short buffer (3, 2 and 0 bytes): ENOMEM, no prefix written, oldlen set to 0
            for short in [3, 2, 0] {
                fill();
                set_len(short);
                assert_eq!(call(n, [name, nl, buf, lenp, 0, 0, 0, 0]), enomem, "#{n} oldlen {short}");
                assert!(untouched(), "#{n} oldlen {short}: no prefix written");
                assert_eq!(len(), 0, "#{n} oldlen {short}: oldlen set to 0");
            }
            // must-fail control: a buffer that fits is written, and oldlen is the answer's length
            fill();
            set_len(64);
            assert!(call(n, [name, nl, buf, lenp, 0, 0, 0, 0]).is_ok());
            assert!(!untouched() && len() == need, "#{n} control: written");
            // a short oldlen on a read-only page: the 0 cannot be written, EFAULT (natively such a
            // oldlenp never returns, rows N2, M2, I2)
            assert_eq!(call(n, [name, nl, buf, ro + 8, 0, 0, 0, 0]), efault, "#{n} short, read-only oldlenp");
            assert_eq!(unsafe { *((ro + 8) as *const u64) }, 2);
        }
        // oldp 3 bytes before a PROT_NONE page: what fits is written, then EFAULT (as native)
        let (g, gpg, at) = guard_page(b"zzz");
        set_len(64);
        assert_eq!(call(274, [hm.as_ptr() as u64, 10, at, lenp, 0, 0, 0, 0]), efault);
        assert_eq!(unsafe { std::slice::from_raw_parts(at as *const u8, 3) }, b"arm");
        assert_eq!(len(), 64);
        // guest_copyout with a memory that refuses, whole, any write reaching past `limit` (what
        // guestmem::write did from inside the translator, probe row N5): the bytes below the limit
        // are still written, then false; a write that fits is whole
        let mem = std::cell::RefCell::new([b'z'; 8]);
        let limit = 3u64;
        let fake = |at: u64, d: &[u8]| -> bool {
            if at + d.len() as u64 > limit {
                return false;
            }
            mem.borrow_mut()[at as usize..at as usize + d.len()].copy_from_slice(d);
            true
        };
        assert!(!guest_copyout(0, b"arm64\0", fake));
        assert_eq!(&mem.borrow()[..], b"armzzzzz");
        assert!(guest_copyout(0, b"ab", fake));
        assert_eq!(&mem.borrow()[..2], b"ab");
        // A MIB is called a MIB, a name a name; a name never read gets no trace
        // of "unreadable" (sysctl_name_not_read) and no warning
        assert!(unread_sysctl_warning(202, 0x1000, true, true).unwrap().contains("sysctl MIB at"));
        assert!(unread_sysctl_warning(274, 0x1000, true, true).unwrap().contains("sysctl name at"));
        assert!(sysctl_name_not_read(0, 10) && sysctl_name_not_read(0x1000, 0) && sysctl_name_not_read(0x1000, 1024));
        assert!(!sysctl_name_not_read(0x1000, 10), "must-fail control: an unreadable name is read");
        unsafe {
            libc::munmap(g, 2 * gpg);
            libc::munmap(m, 3 * pg);
        }
    }

    #[test]
    fn sysctl_optional_names_answer_as_native_without_the_host() {
        // Every hw.optional name native arm64 has as an integer
        // (the hw.optional names of macOS 27.0.1 on arm64e) is answered by the table alone, so an x86 host's own
        // answers (1 for its features, ENOENT for the arm names) never reach the guest. None of these calls
        // reaches the host: sysctl_override and sysctl_mib_answer are pure, and the MIB route is given a host
        // namer that panics.
        use SysctlAnswer::{Absent, Fails, Int};
        assert_eq!(NATIVE_OPTIONAL_ENOTSUP.len() + NATIVE_OPTIONAL_INT.len() + NATIVE_OPTIONAL_OTHER.len(), 125);
        for name in NATIVE_OPTIONAL_ENOTSUP {
            assert_eq!(sysctl_override(name), Some(Fails(libc::ENOTSUP)), "{name}");
        }
        // the 20 K v4 did not list (an x86 host answers 1 for several of them)
        for x in ["mmx", "aes", "rdrand", "f16c", "enfstrg", "fma", "bmi1", "bmi2", "rtm", "hle", "adx", "mpx", "sgx",
                  "avx512f", "avx512cd", "avx512dq", "avx512bw", "avx512vl", "avx512ifma", "avx512vbmi"] {
            let name = format!("hw.optional.{x}");
            assert!(NATIVE_OPTIONAL_ENOTSUP.contains(&name.as_str()), "{name} measured ENOTSUP");
            assert_eq!(sysctl_override(&name), Some(Fails(libc::ENOTSUP)), "{name}");
        }
        for name in NATIVE_OPTIONAL_INT {
            assert!(matches!(sysctl_override(name), Some(Int(_))), "{name}: a 4-byte integer, as native");
        }
        // the commpage's honoured features answer 1; an arm name the translator does not promise answers 0
        // (natively M1 answers 1 for FEAT_DIT and neon_fp16, 4 for watchpoint: the translator's choice)
        for name in ["hw.optional.arm.AdvSIMD", "hw.optional.arm.AdvSIMD_HPFPCvt", "hw.optional.neon_hpfp",
                     "hw.optional.ucnormal_mem", "hw.optional.arm.FEAT_CSV2", "hw.optional.arm.FEAT_CSV3"] {
            assert!(NATIVE_OPTIONAL_INT.contains(&name), "{name} exists natively");
            assert_eq!(sysctl_override(name), Some(Int(1)), "{name}");
        }
        for name in ["hw.optional.arm.FEAT_DIT", "hw.optional.neon_fp16", "hw.optional.watchpoint",
                     "hw.optional.breakpoint", "hw.optional.arm.FEAT_BTI", "hw.optional.arm.sme_max_svl_b"] {
            assert!(NATIVE_OPTIONAL_INT.contains(&name), "{name} exists natively");
            assert_eq!(sysctl_override(name), Some(Int(0)), "{name}");
        }
        // the table's own values stand (precedence over the default)
        assert_eq!(sysctl_override("hw.optional.arm.FEAT_PAuth"), Some(Int(1)));
        assert_eq!(sysctl_override("hw.optional.arm64"), Some(Int(1)));
        assert_eq!(sysctl_override("hw.optional.AdvSIMD"), Some(Absent));
        // must-fail controls: another shape, a name native lacks, a host sysctl, a prefix are passed through
        for name in NATIVE_OPTIONAL_OTHER.iter().copied().chain(["hw.optional.nope", "hw.optional", "hw.optional.mmx2",
                                                                 "kern.ostype", "hw.optional.arm."]) {
            assert_eq!(sysctl_override(name), None, "{name}");
        }
        // by MIB: an x86 name and an arm name answer as by name, whatever number the host gave
        let never = |m: &[i32]| -> Option<String> { panic!("asked the host to name {m:?}") };
        let named = |name: &'static str| move |_: &[i32]| Some(name.to_string());
        assert_eq!(sysctl_mib_answer(&[6, 103, 200], named("hw.optional.avx512vbmi")), Some(Fails(libc::ENOTSUP)));
        assert_eq!(sysctl_mib_answer(&[6, 103, 201], named("hw.optional.watchpoint")), Some(Int(0)));
        // {CTL_HW, HW_PAGESIZE} is hw.pagesize_compat (4 bytes) without asking, so an x86 kernel's
        // name for it (hw.pagesize, 8 bytes) is never what answers
        assert_eq!(sysctl_mib_answer(&[6, 7], never), Some(Int(16384)));
        assert_eq!(sysctl_override("hw.pagesize"), Some(SysctlAnswer::Quad(16384)), "the name it must not take");
    }

    #[test]
    fn sysctl_writes_widths_and_mibs_answer_as_native() {
        // Against what macOS 27.0.1 on arm64
        // answers natively.
        use SysctlAnswer::{Absent, Fails, Int, Quad, Str};
        let (eperm, enomem, enoent, enotsup) =
            (Err(libc::EPERM as u64), Err(libc::ENOMEM as u64), Err(libc::ENOENT as u64), Err(libc::ENOTSUP as u64));
        let pg = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let m = unsafe { libc::mmap(std::ptr::null_mut(), 2 * pg, libc::PROT_READ | libc::PROT_WRITE,
                                    libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) };
        assert_ne!(m, libc::MAP_FAILED);
        let (buf, lenp) = (m as u64, m as u64 + pg as u64);
        let newv: i32 = 0;
        let newp = &newv as *const i32 as u64;
        let fill = || unsafe { std::ptr::write_bytes(buf as *mut u8, 0x5a, 64) };
        let set_len = |v: u64| unsafe { *(lenp as *mut u64) = v };
        let len = || unsafe { *(lenp as *const u64) };
        let got = |n: usize| unsafe { std::slice::from_raw_parts(buf as *const u8, n) }.to_vec();
        let untouched = || got(64).iter().all(|&b| b == 0x5a);
        // A write to a read-only name is EPERM before anything is read or written: oldp and
        // oldlen untouched, whatever oldp, oldlenp and newp are (an unmapped oldlenp, an unmapped newp)
        for a in [Str("arm64"), Int(0), Quad(16384), Fails(libc::ENOTSUP)] {
            for (op, lp, np, nl) in [(buf, lenp, newp, 4), (0, 0, newp, 4), (buf, 0, newp, 4), (0, lenp, newp, 4),
                                     (buf, 0x1000, newp, 4), (buf, lenp, 0x1000, 4), (buf, lenp, newp, 1 << 30)] {
                fill();
                set_len(64);
                assert_eq!(sysctl_reply(a, op, lp, np, nl), eperm, "{a:?} write {op:#x} {lp:#x} {np:#x} {nl}");
                assert!(untouched() && len() == 64, "{a:?} write: nothing touched");
            }
            // a short oldlen: still EPERM, oldlen left (native: l2)
            set_len(2);
            assert_eq!(sysctl_reply(a, buf, lenp, newp, 4), eperm);
            assert_eq!(len(), 2);
        }
        // must-fail controls: newp with newlen 0, and newlen with newp NULL, are reads (native: ok)
        for (np, nl) in [(newp, 0), (0, 4)] {
            fill();
            set_len(64);
            assert_eq!(sysctl_reply(Str("arm64"), buf, lenp, np, nl), Ok(()), "newp {np:#x} newlen {nl}: a read");
            assert_eq!((got(6), len()), (b"arm64\0".to_vec(), 6));
        }
        // a name that does not exist: ENOENT first, a write included; a Fails name: its errno to a read,
        // nothing touched (an unmapped oldlenp is never read: native returns at once there)
        for (op, lp, np, nl) in [(buf, lenp, 0, 0), (buf, lenp, newp, 4), (buf, 0x1000, 0, 0), (0, 0, 0, 0), (buf, 0, 0, 0)] {
            fill();
            set_len(64);
            assert_eq!(sysctl_reply(Absent, op, lp, np, nl), enoent);
            if np == 0 {
                assert_eq!(sysctl_reply(Fails(libc::ENOTSUP), op, lp, np, nl), enotsup);
            }
            assert!(untouched() && len() == 64);
        }
        // Widths. A 4-byte integer: 4 bytes and oldlen 4 for any buffer of 4 or more, the rest of an
        // 8-byte buffer untouched; ENOMEM, oldlen 0 below 4; the query answers 4
        for cap in [4u64, 5, 7, 8, 9, 64] {
            fill();
            set_len(cap);
            assert_eq!(sysctl_reply(Int(0x0100_000c), buf, lenp, 0, 0), Ok(()), "Int cap {cap}");
            assert_eq!((len(), got(8)), (4, vec![0x0c, 0, 0, 1, 0x5a, 0x5a, 0x5a, 0x5a]), "Int cap {cap}");
        }
        for cap in [0u64, 3] {
            fill();
            set_len(cap);
            assert_eq!(sysctl_reply(Int(1), buf, lenp, 0, 0), enomem);
            assert!(untouched() && len() == 0);
        }
        set_len(999);
        assert_eq!(sysctl_reply(Int(1), 0, lenp, 0, 0), Ok(()));
        assert_eq!(len(), 4);
        // an 8-byte integer (hw.pagesize, hw.pagesize32): 8 bytes from 8, 4 bytes at exactly 4, ENOMEM at 5 to 7
        // and below 4; the query answers 8
        set_len(999);
        assert_eq!(sysctl_reply(Quad(16384), 0, lenp, 0, 0), Ok(()));
        assert_eq!(len(), 8);
        for cap in [8u64, 9, 64] {
            fill();
            set_len(cap);
            assert_eq!(sysctl_reply(Quad(16384), buf, lenp, 0, 0), Ok(()));
            assert_eq!((len(), got(8)), (8, 16384i64.to_le_bytes().to_vec()));
        }
        fill();
        set_len(4);
        assert_eq!(sysctl_reply(Quad(16384), buf, lenp, 0, 0), Ok(()));
        assert_eq!((len(), got(8)), (4, vec![0, 0x40, 0, 0, 0x5a, 0x5a, 0x5a, 0x5a]));
        for cap in [0u64, 3, 5, 6, 7] {
            fill();
            set_len(cap);
            assert_eq!(sysctl_reply(Quad(16384), buf, lenp, 0, 0), enomem, "Quad cap {cap}");
            assert!(untouched() && len() == 0, "Quad cap {cap}");
        }
        // the table's shapes, one per kind measured
        assert_eq!(sysctl_override("hw.cputype"), Some(Int(0x0100_000c)));
        assert_eq!(sysctl_override("sysctl.proc_translated"), Some(Int(0)));
        assert_eq!(sysctl_override("vm.pagesize"), Some(Int(16384)));
        assert_eq!(sysctl_override("hw.pagesize_compat"), Some(Int(16384)));
        assert_eq!(sysctl_override("hw.pagesize32"), Some(Quad(16384)));
        assert_eq!(sysctl_override("hw.optional.sse4_2"), Some(Fails(libc::ENOTSUP)));
        assert_eq!(sysctl_override("hw.optional.AdvSIMD"), Some(Absent));
        assert_eq!(sysctl_override("kern.ostype"), None, "must-fail control: a host sysctl");
        // The MIB names the override. {6, 1} without asking the host; the sysctl meta entries, other
        // top levels and a MIB too long to name are never asked about
        let never = |m: &[i32]| -> Option<String> { panic!("asked the host to name {m:?}") };
        assert_eq!(sysctl_mib_answer(&[6, 1], never), Some(Str("arm64")));
        for mib in [&[0, 3][..], &[0, 1, 6, 1], &[0, 5], &[1, 1], &[7, 1], &[6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] {
            assert_eq!(sysctl_mib_answer(mib, never), None, "{mib:?}");
        }
        let named = |name: &'static str| move |_: &[i32]| Some(name.to_string());
        assert_eq!(sysctl_mib_answer(&[6, 7], named("hw.pagesize_compat")), Some(Int(16384)));
        assert_eq!(sysctl_mib_answer(&[0, 100], named("sysctl.proc_translated")), Some(Int(0)));
        assert_eq!(sysctl_mib_answer(&[2, 257], named("vm.pagesize")), Some(Int(16384)));
        assert_eq!(sysctl_mib_answer(&[6, 103, 109], named("hw.optional.x86_64")), Some(Fails(libc::ENOTSUP)));
        assert_eq!(sysctl_mib_answer(&[6, 1, 2, 3, 4, 5, 6, 7, 8, 9], named("hw.pagesize")), Some(Quad(16384)), "10 entries: asked");
        assert_eq!(sysctl_mib_answer(&[6, 2], named("hw.model")), None, "must-fail control: a host name");
        assert_eq!(sysctl_mib_answer(&[6, 8], |_: &[i32]| None), None);
        // real calls through bsd(202), with the MIBs the host kernel gives (sysctlnametomib); under Rosetta the
        // host answers 4096, 1, 4096 and 1 for these four, so a call that missed the override shows
        let call = |n: u64, args: [u64; 8]| -> Result<u64, u64> {
            let mut cpu = Cpu::new();
            bsd(&mut cpu, n, args);
            if cpu.cf == 0 { Ok(cpu.x[0]) } else { Err(cpu.x[0]) }
        };
        let host_mib = |name: &str| -> Vec<i32> {
            let c = std::ffi::CString::new(name).unwrap();
            let mut mib = [0i32; 12];
            let mut n: libc::size_t = 12;
            assert_eq!(unsafe { libc::sysctlnametomib(c.as_ptr(), mib.as_mut_ptr(), &mut n) }, 0, "{name}");
            mib[..n].to_vec()
        };
        assert_eq!(host_oid2name(&[6, 7]).as_deref(), Some("hw.pagesize_compat"));
        assert_eq!(host_oid2name(&host_mib("hw.pagesize")).as_deref(), Some("hw.pagesize"));
        for (name, want_len, want) in [("hw.pagesize", 8u64, 16384i64), ("sysctl.proc_translated", 4, 0),
                                       ("hw.pagesize_compat", 4, 16384), ("vm.pagesize", 4, 16384)] {
            let mib = if name == "hw.pagesize_compat" { vec![6, 7] } else { host_mib(name) };
            fill();
            set_len(64);
            assert_eq!(call(202, [mib.as_ptr() as u64, mib.len() as u64, buf, lenp, 0, 0, 0, 0]), Ok(0), "{name} by MIB");
            let mut v = [0u8; 8];
            v[..want_len as usize].copy_from_slice(&got(want_len as usize));
            assert_eq!((len(), i64::from_le_bytes(v)), (want_len, want), "{name} by MIB");
            set_len(64);
            assert_eq!(call(202, [mib.as_ptr() as u64, mib.len() as u64, buf, lenp, newp, 4, 0, 0]), Err(libc::EPERM as u64), "{name} written by MIB");
        }
        let x = host_mib("hw.optional.x86_64");
        set_len(64);
        assert_eq!(call(202, [x.as_ptr() as u64, x.len() as u64, buf, lenp, 0, 0, 0, 0]), Err(libc::ENOTSUP as u64));
        assert_eq!(len(), 64);
        // must-fail control: a host MIB is the host's
        let k: [i32; 2] = [1, 1];
        set_len(64);
        assert_eq!(call(202, [k.as_ptr() as u64, 2, buf, lenp, 0, 0, 0, 0]), Ok(0));
        assert_eq!(got(7), b"Darwin\0".to_vec());
        // and by name, a write: EPERM through bsd(274) too
        let hm = b"hw.machine";
        set_len(64);
        assert_eq!(call(274, [hm.as_ptr() as u64, 10, buf, lenp, newp, 4, 0, 0]), Err(libc::EPERM as u64));
        unsafe { libc::munmap(m, 2 * pg) };
    }

    #[test]
    fn a_kept_call_goes_to_the_canonical_path_of_the_directory_that_was_checked() {
        // No descriptor is made for a kept call.
        use std::os::unix::io::AsRawFd;
        let base = std::env::temp_dir().join(format!("maclator-l3-test-{}", std::process::id()));
        let (a, b) = (base.join("a"), base.join("b"));
        for d in [&a, &b] {
            std::fs::create_dir_all(d).unwrap();
        }
        std::fs::write(a.join("only-a"), b"").unwrap();
        // a, reached through a symlink: the test's temp dir is /var/folders/..., and /var is one
        let link = base.join("link-a");
        std::os::unix::fs::symlink(&a, &link).unwrap();
        let da = std::fs::File::open(&link).unwrap();
        let db = std::fs::File::open(&b).unwrap();
        let (fa, fb) = (da.as_raw_fd(), db.as_raw_fd());
        let (ka, kb) = (fd_dev_ino(fa).unwrap(), fd_dev_ino(fb).unwrap());
        let nofollow_any_open = |path: &str| -> Result<(), i32> {
            let c = std::ffi::CString::new(path).unwrap();
            let r = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY | O_NOFOLLOW_ANY as i32) };
            if r < 0 {
                return Err(unsafe { *libc::__error() });
            }
            unsafe { libc::close(r) };
            Ok(())
        };
        // 1. the canonical path is a's, has no symlink in it, and opens under NOFOLLOW_ANY
        let p = canonical_dir(fa, ka, fd_host_getpath, path_dev_ino).expect("a's own path");
        assert_eq!(path_dev_ino(&p), Some(ka));
        assert_eq!(nofollow_any_open(&format!("{p}/only-a")), Ok(()));
        // must-fail control: the same file by the path the descriptor was opened with crosses a
        // symlink (link-a, and /var), which NOFOLLOW_ANY refuses: the rewrite L v3 never takes
        let given = format!("{}/only-a", link.to_str().unwrap());
        assert_eq!(nofollow_any_open(&given), Err(libc::ELOOP));
        // 2. the check saw a, then a guest thread's dup2 put b on the number: F_GETPATH answers b's
        //    path, which is not a, so no path (the caller checks again)
        let g = unsafe { libc::dup(fa) };
        assert_eq!(canonical_dir(g, ka, |fd| { assert_eq!(unsafe { libc::dup2(fb, fd) }, fd); fd_host_getpath(fd) }, path_dev_ino), None);
        assert_eq!(canonical_dir(g, kb, fd_host_getpath, path_dev_ino).map(|p| path_dev_ino(&p)), Some(Some(kb)), "control: b's own");
        // 3. a path that names something else by the time it is stat'd (a rename in between) is
        //    refused; an unreadable descriptor or path gives none
        assert_eq!(canonical_dir(fa, ka, fd_host_getpath, |_: &str| Some(kb)), None);
        assert_eq!(canonical_dir(-1, ka, fd_host_getpath, path_dev_ino), None);
        assert_eq!(canonical_dir(fa, ka, |_| None, path_dev_ino), None);
        // 4. the guest's POSIX lock on a survives (no descriptor of a was made and closed): another
        //    process still sees it
        let mut lk: libc::flock = unsafe { std::mem::zeroed() };
        lk.l_type = libc::F_RDLCK as i16;
        lk.l_whence = libc::SEEK_SET as i16;
        assert_eq!(unsafe { libc::fcntl(fa, libc::F_SETLK, &lk as *const libc::flock) }, 0);
        assert!(canonical_dir(fa, ka, fd_host_getpath, path_dev_ino).is_some());
        let ca = std::ffi::CString::new(a.to_str().unwrap()).unwrap();
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            let f = unsafe { libc::open(ca.as_ptr(), libc::O_RDONLY) };
            let mut q: libc::flock = unsafe { std::mem::zeroed() };
            q.l_type = libc::F_WRLCK as i16;
            q.l_whence = libc::SEEK_SET as i16;
            let r = unsafe { libc::fcntl(f, libc::F_GETLK, &mut q as *mut libc::flock) };
            unsafe { libc::_exit(if r != 0 { 3 } else if q.l_type == libc::F_UNLCK as i16 { 1 } else { 0 }) };
        }
        let mut st = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut st, 0) }, pid);
        assert!(libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0, "the lock is still held (child status {st:#x})");
        unsafe { libc::close(g) };
        drop((da, db));
        std::fs::remove_file(&link).unwrap();
        std::fs::remove_file(a.join("only-a")).unwrap();
        for d in [&a, &b, &base] {
            std::fs::remove_dir(d).unwrap();
        }
    }

    #[test]
    fn a_kept_route_reaching_maxpathlen_or_spelled_otherwise_stays_right() {
        // The joined path is used up to 1,023 bytes; at 1,024 the call keeps the guest's number
        let d = format!("/{}", "d".repeat(600));
        let name = |k: usize| "n".repeat(k);
        let r = "/r/sys/x";
        let fits = 1023 - d.len() - 1;
        assert_eq!(kept_route(&d, r, &name(fits)), Kept::Path { path: format!("{d}/{}", name(fits)), redirected: format!("{r}/{}", name(fits)) });
        assert_eq!(kept_route(&d, r, &name(fits + 1)), Kept::Number { redirected: format!("{r}/{}", name(fits + 1)) });
        assert!(matches!(kept_route("/a", r, "b"), Kept::Path { .. }), "control: a short path");
        // the boundary is the host's: a 1,023-byte path is looked up, a 1,024-byte one is ENAMETOOLONG
        let probe = |len: usize| -> i32 {
            let p = std::ffi::CString::new(format!("/{}", "x".repeat(len - 1))).unwrap();
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            unsafe { libc::stat(p.as_ptr(), &mut st) };
            unsafe { *libc::__error() }
        };
        assert_eq!(probe(1023), libc::ENAMETOOLONG, "a 255+ byte component is too long on its own: the probe's own control");
        let deep = |len: usize| -> i32 {
            let mut p = String::new();
            while p.len() + 101 <= len {
                p.push('/');
                p.push_str(&"y".repeat(100));
            }
            while p.len() < len {
                p.push('/');
                if p.len() < len {
                    p.push_str(&"z".repeat(len - p.len()));
                }
            }
            assert_eq!(p.len(), len);
            let c = std::ffi::CString::new(p).unwrap();
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            unsafe { libc::stat(c.as_ptr(), &mut st) };
            unsafe { *libc::__error() }
        };
        assert_eq!(deep(1023), libc::ENOENT, "a 1,023-byte path is looked up");
        assert_eq!(deep(1024), libc::ENAMETOOLONG, "a 1,024-byte path is not");
        // inside_sysroot is asked with the redirect's spelling, whose prefix is the sysroot's
        let sysroot = "/System/Volumes/Data/private/var/w/sys";
        let k = kept_route("/private/var/w/sys/d00", &format!("{sysroot}/d00"), "racefile");
        let Kept::Path { path, redirected } = k else { panic!("a short path is a Path") };
        assert!(inside_sysroot(&redirected, Some(sysroot)));
        assert!(!inside_sysroot(&path, Some(sysroot)), "must-fail control: the F_GETPATH spelling misses the sysroot");
        // and the spelling difference is real: a directory opened by its /System/Volumes/Data name has a
        // /private F_GETPATH (the test's temp dir is /var/folders/...), while realpath keeps the name given
        use std::os::unix::io::AsRawFd;
        let t = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        let svd = format!("/System/Volumes/Data{}", t.display());
        if let Ok(f) = std::fs::File::open(&svd) {
            let gp = fd_host_getpath(f.as_raw_fd()).unwrap();
            assert!(!gp.starts_with("/System/Volumes/Data/"), "F_GETPATH {gp}");
            assert_eq!(path_dev_ino(&gp), fd_dev_ino(f.as_raw_fd()));
            let rp = std::fs::canonicalize(&svd).unwrap();
            assert!(rp.starts_with("/System/Volumes/Data/"), "realpath {}", rp.display());
        } else {
            eprintln!("no /System/Volumes/Data spelling of {} here: the measurement half skipped", t.display());
        }
    }
}
