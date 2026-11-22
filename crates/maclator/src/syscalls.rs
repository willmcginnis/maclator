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

fn cstr_at(p: u64) -> String {
    if p == 0 {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p as *const libc::c_char).to_string_lossy().into_owned() }
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
static FD_VPATHS: std::sync::Mutex<Option<std::collections::HashMap<i32, String>>> = std::sync::Mutex::new(None);

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
        33 | 58 | 188 | 190 | 220 | 338 | 340 => (Some(0), false),
        463 | 464 | 466 | 469 | 470 | 473 | 476 => (Some(1), true),
        _ => (None, false),
    }
}

/// The guest path a syscall's name refers to: itself if absolute, else (in an *at call) the
/// name joined to its descriptor's guest path. `dir` is asked only for a relative, non-empty name
/// in an *at call. There is no virtual working directory: a relative name outside an *at call, or
/// through a descriptor with no guest path (AT_FDCWD included), gives None and is not rewritten.
fn guest_full_path(name: &str, at: bool, dir: impl FnOnce() -> Option<String>) -> Option<String> {
    if name.starts_with('/') {
        Some(name.to_string())
    } else if at && !name.is_empty() {
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
        return if followed.is_some_and(|m| m.is_dir()) { Node::Dir } else { Node::Other };
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
/// unreadable.
fn same_host_dir(fd: i32, host: &str) -> bool {
    use std::os::unix::fs::MetadataExt;
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return false;
    }
    std::fs::metadata(host).is_ok_and(|m| m.dev() == st.st_dev as u64 && m.ino() == st.st_ino as u64)
}

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
fn nofollow_any_inside(name: &str, fd: i32, guest: Option<String>, redirect: impl FnOnce(&str) -> Option<String>) -> bool {
    !name.starts_with('/')
        && stays_beneath(name)
        && guest.and_then(|d| redirect(&d)).is_some_and(|h| same_host_dir(fd, &h))
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
    if let Some(i) = pidx {
        let name = call_name(n, &args, i, |p| {
            let name = read(p);
            if name.is_none() {
                unread = Some(p);
            }
            name
        });
        if let Some(orig) = name.filter(|o| rewritable(o)) {
            let full = guest_full_path(&orig, at, || fd_guest_path(args[0] as i32));
            if TRACE.load(Ordering::Relaxed) {
                eprintln!("[path] #{n} {orig} -> {full:?}");
            }
            if let Some(full) = full {
                // Resolve a `..` in the guest's view before the redirect sees the path.
                let before = full;
                let ids = lookup_ids(n, &args);
                // A symlink before a `.` or `..` is the kernel's to look up (resolve_dotdot); a name
                // it sends to the host goes as it is, never through the redirect.
                let dotdot = resolve_dotdot(&before, follows_symlinks(n, &args), ids, |p| guest_node_as(p, ids), kernel_lookup,
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
                let inside = || {
                    let fd = args[0] as i32;
                    nofollow_any_inside(&orig, fd, fd_guest_path(fd), crate::paths::redirect)
                };
                let replacement = if at && keep_descriptor(n, &args, inside) {
                    None
                } else if host_as_is {
                    Some(full.clone())
                } else {
                    crate::paths::redirect(&full).or_else(|| {
                        (at && !orig.starts_with('/') && crate::paths::overlay_root().is_some()).then(|| full.clone())
                    }).or_else(|| left_the_prefix(&before, &full, crate::paths::redirect))
                };
                if let Some(new) = replacement {
                    let ins = inside_sysroot(&new, crate::paths::cache_dir().as_deref());
                    if let Ok(c) = std::ffi::CString::new(new) {
                        args[i] = c.as_ptr() as u64;
                        if at {
                            args[0] = (-2i64) as u64; // AT_FDCWD; path is absolute now
                        }
                        keep = Some(c);
                        in_sysroot = ins;
                    }
                }
                guest_path = Some(full);
            }
        }
    }
    bsd_inner(cpu, n, args);
    drop(keep);
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
            let g = descriptor_record(&g, in_sysroot, || host_fd_path(cpu.x[0] as i32));
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
            let pid = unsafe { libc::fork() };
            if pid < 0 {
                set_err(cpu, unsafe { *libc::__error() } as u64);
            } else if pid == 0 {
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
            let name = cstr_at(args[0]);
            if name == "hw.machine" {
                write_sysctl_str(cpu, args[2], args[3], "arm64");
                return;
            }
            if let Some(v) = sysctl_override(&name) {
                write_sysctl_int(cpu, args[2], args[3], v);
                return;
            }
            let r = hostsys::unix(n, &args);
            set_res(cpu, r);
        }
        SYS_SYSCTL => {
            // hw.machine (CTL_HW=6, HW_MACHINE=1) is what uname() reports.
            let mib = unsafe { std::slice::from_raw_parts(args[0] as *const i32, (args[1] as usize).min(8)) };
            if args[1] == 2 && mib == [6, 1] {
                write_sysctl_str(cpu, args[2], args[3], "arm64");
                return;
            }
            let r = hostsys::unix(n, &args);
            set_res(cpu, r);
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

fn write_sysctl_str(cpu: &mut Cpu, oldp: u64, oldlenp: u64, v: &str) {
    let bytes = [v.as_bytes(), &[0]].concat();
    unsafe {
        if oldlenp != 0 {
            let len = *(oldlenp as *const u64) as usize;
            if oldp != 0 {
                if len < bytes.len() {
                    set_err(cpu, libc::ENOMEM as u64);
                    return;
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), oldp as *mut u8, bytes.len());
            }
            *(oldlenp as *mut u64) = bytes.len() as u64;
        }
    }
    set_ok(cpu, 0, 0);
}

fn write_sysctl_int(cpu: &mut Cpu, oldp: u64, oldlenp: u64, v: i64) {
    unsafe {
        if oldlenp != 0 {
            let len = *(oldlenp as *const u64);
            if oldp != 0 {
                if len >= 8 {
                    *(oldp as *mut i64) = v;
                    *(oldlenp as *mut u64) = 8;
                } else if len >= 4 {
                    *(oldp as *mut i32) = v as i32;
                    *(oldlenp as *mut u64) = 4;
                } else {
                    set_err(cpu, libc::ENOMEM as u64);
                    return;
                }
            } else {
                *(oldlenp as *mut u64) = 4;
            }
        }
    }
    set_ok(cpu, 0, 0);
}

/// sysctls whose answer must describe the emulated arm64 machine.
fn sysctl_override(name: &str) -> Option<i64> {
    Some(match name {
        "sysctl.proc_translated" => 0,
        "sysctl.proc_native" => 1,
        "hw.cputype" => 0x0100_000c,
        "hw.cpusubtype" => 2,
        "hw.optional.arm64" => 1,
        "hw.optional.neon" | "hw.optional.AdvSIMD" | "hw.optional.floatingpoint" => 1,
        "hw.optional.armv8_crc32" | "hw.optional.arm.FEAT_CRC32" => 1,
        "hw.optional.armv8_1_atomics" | "hw.optional.arm.FEAT_LSE" => 1,
        // Aligned 128-bit support is narrower than the full LSE2 contract.
        "hw.optional.arm.FEAT_LSE2" => 0,
        "hw.optional.arm.FEAT_LRCPC" | "hw.optional.arm.FEAT_LRCPC2" => 1,
        "hw.optional.arm.FEAT_PAuth" | "hw.optional.arm.FEAT_JSCVT" | "hw.optional.arm.FEAT_FlagM"
        | "hw.optional.arm.FEAT_FlagM2" | "hw.optional.arm.FEAT_DotProd" | "hw.optional.arm.FEAT_FRINTTS"
        | "hw.optional.arm.FEAT_SB" | "hw.optional.arm.FEAT_DPB" | "hw.optional.arm.FEAT_DPB2" => 1,
        "hw.optional.arm.FEAT_FP16" | "hw.optional.arm.FEAT_FHM" | "hw.optional.arm.FEAT_FCMA"
        | "hw.optional.arm.FEAT_RDM" | "hw.optional.arm.FEAT_SHA3" | "hw.optional.arm.FEAT_SHA512"
        | "hw.optional.arm.FEAT_SME" | "hw.optional.arm.FEAT_SME2" | "hw.optional.arm.FEAT_BF16"
        | "hw.optional.arm.FEAT_I8MM" | "hw.optional.armv8_2_fhm" | "hw.optional.armv8_2_sha512"
        | "hw.optional.armv8_2_sha3" | "hw.optional.armv8_3_compnum" | "hw.optional.arm.FEAT_SHA256"
        | "hw.optional.arm.FEAT_AES" | "hw.optional.arm.FEAT_PMULL" | "hw.optional.arm.FEAT_SHA1"
        | "hw.optional.arm.FEAT_AFP" | "hw.optional.arm.FEAT_RPRES" | "hw.optional.arm.FEAT_ECV"
        | "hw.optional.arm.FEAT_WFxT" | "hw.optional.arm.FEAT_CSSC" | "hw.optional.arm.FEAT_HBC" => 0,
        "hw.pagesize" | "hw.pagesize32" | "vm.pagesize" => 16384,
        "hw.cpufamily" => crate::commpage::CPUFAMILY as i64,
        "hw.optional.x86_64" | "hw.optional.sse" | "hw.optional.sse2" | "hw.optional.sse3"
        | "hw.optional.supplementalsse3" | "hw.optional.sse4_1" | "hw.optional.sse4_2"
        | "hw.optional.avx1_0" | "hw.optional.avx2_0" => 0,
        _ => return None,
    })
}

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
        static ARENA: std::sync::OnceLock<(u64, std::sync::atomic::AtomicU64)> = std::sync::OnceLock::new();
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
        for n in [5, 398, 33, 58, 188, 190, 220, 338, 340] {
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
        // no virtual cwd: a relative name outside an *at call, or through AT_FDCWD (no guest path)
        assert_eq!(guest_full_path("x", false, || Some("/System".into())), None);
        assert_eq!(guest_full_path("x", true, || None), None);
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
        assert!(nofollow_any_inside("dyld_shared_cache_arm64e", s.as_raw_fd(), g(), to(&sys)));
        assert!(nofollow_any_inside("dyld_shared_cache_arm64e", s.as_raw_fd(), g(), to(&link)));
        assert!(nofollow_any_inside("a/../x", s.as_raw_fd(), g(), to(&sys)));
        // The guest path is redirected, but the host descriptor sits elsewhere
        assert!(!nofollow_any_inside("dyld_shared_cache_arm64e", o.as_raw_fd(), g(), to(&sys)));
        // A name that climbs out of the descriptor
        assert!(!nofollow_any_inside("../sys/x", s.as_raw_fd(), g(), to(&sys)));
        // an absolute name, no guest path, a guest path that is not redirected, a closed fd
        assert!(!nofollow_any_inside("/x", s.as_raw_fd(), g(), to(&sys)));
        assert!(!nofollow_any_inside("x", s.as_raw_fd(), None, to(&sys)));
        assert!(!nofollow_any_inside("x", s.as_raw_fd(), g(), |_: &str| None));
        assert!(!nofollow_any_inside("x", -1, g(), to(&sys)));
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
}
