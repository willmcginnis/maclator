//! execve / posix_spawn. Native (x86-64 or universal) programs run directly on
//! the host; arm64-only programs are re-launched through Maclator, carrying over
//! the sysroot and options so children (e.g. Chromium/Electron helper processes)
//! run in the same environment.
//!
//! A process the kernel started as the program itself (a kernel with a foreign-architecture
//! exec, which loads the translator beside the arm64 program, as Rosetta 2's does) hands such an
//! exec to the kernel unchanged instead: the kernel then starts the child as the program too, so
//! what the kernel answers about it (its code signature and entitlements, as a peer reads them by
//! pid or audit token, and its path) is the program's own, not the translator's.

use crate::hostsys::{self, SysResult};
use std::ffi::CString;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::slices::{arm64_only, CPU_TYPE_ARM64};
use crate::forksync::FMutex;

const SYS_EXECVE: u64 = 59;

/// The most arguments the kernel's translator runtime hands an engine (its own limit: it refuses more, counting the
/// arguments its configuration adds), less room for those.
const RUNTIME_ARGS: usize = 4090 - 64;

/// Options to forward to re-launched children (`--sysroot X`, `--dyld D`, ...).
pub(crate) static FORWARD: FMutex<Vec<String>> = FMutex::new(Vec::new());

/// Whether the kernel started this process as the program (`route_from`); read once by `detect_route`.
static KERNEL_ROUTE: AtomicBool = AtomicBool::new(false);

pub fn configure(flags: Vec<String>) {
    *FORWARD.lock().unwrap() = flags;
}

/// Read once, before the program runs, whether the kernel started this process as the program, and keep the answer
/// for every exec. A fork child keeps it, as it keeps its parent's main image; an exec starts a new translator, which
/// reads it again.
pub fn detect_route() -> bool {
    let r = route_from(main_image().as_deref(), engine_image().as_deref());
    KERNEL_ROUTE.store(r, Ordering::SeqCst);
    r
}

/// The file the kernel holds as this process's main image (proc_pidpath).
fn main_image() -> Option<String> {
    let mut buf = vec![0u8; 4096];
    let n = unsafe { libc::proc_pidpath(libc::getpid(), buf.as_mut_ptr() as *mut libc::c_void, buf.len() as u32) };
    if n <= 0 || n as usize > buf.len() {
        return None;
    }
    buf.truncate(n as usize);
    String::from_utf8(buf).ok()
}

extern "C" {
    /// dyld's (mach-o/dyld.h); declared here, as the libc crate's binding is deprecated.
    fn _NSGetExecutablePath(buf: *mut libc::c_char, bufsize: *mut u32) -> libc::c_int;
}

/// The translator's own image as dyld names it (_NSGetExecutablePath): the file the kernel exec'd when the translator
/// was started directly; under a foreign-architecture exec, the translator file the kernel's runtime started.
fn engine_image() -> Option<String> {
    let mut size: u32 = 0;
    unsafe { _NSGetExecutablePath(std::ptr::null_mut(), &mut size) };
    if size == 0 {
        return None;
    }
    let mut buf = vec![0u8; size as usize];
    if unsafe { _NSGetExecutablePath(buf.as_mut_ptr() as *mut libc::c_char, &mut size) } != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0)?;
    buf.truncate(end);
    String::from_utf8(buf).ok()
}

/// (device, inode) of the file `p` names.
fn file_id(p: &str) -> Option<(u64, u64)> {
    let c = CString::new(p).ok()?;
    let mut s: libc::stat = unsafe { std::mem::zeroed() };
    (unsafe { libc::stat(c.as_ptr(), &mut s) } == 0).then_some((s.st_dev as u64, s.st_ino as u64))
}

/// Whether the file `p` holds an arm64 slice.
fn has_arm64(p: &str) -> bool {
    let Ok(mut f) = std::fs::File::open(p) else { return false };
    crate::slices::read_slices(&mut f).is_ok_and(|s| s.iter().any(|a| a.cpu == CPU_TYPE_ARM64))
}

/// Whether the kernel started this process as the program: its main image (`main`) is another file than the
/// translator's own (`engine`), and one with an arm64 slice, the kind a foreign-architecture exec starts (an arm64-only
/// file, or a universal one whose arm64 slice a spawn's binary preference chose). A translator started directly (by
/// hand, by a plist, by Rosetta 2, or re-launched by a translated parent) is its own main image. Files are compared by
/// device and inode, so another spelling of one file (a link, /var for /private/var) is that file. Any answer that
/// cannot be read is no: the re-launch, as before.
fn route_from(main: Option<&str>, engine: Option<&str>) -> bool {
    let (Some(m), Some(e)) = (main, engine) else { return false };
    if m == e {
        return false;
    }
    match (file_id(m), file_id(e)) {
        (Some(a), Some(b)) => a != b && has_arm64(m),
        _ => false,
    }
}

/// How an exec of an arm64-only program is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// To the kernel unchanged: the kernel starts the child as the program.
    Kernel,
    /// The translator with the program as its argument, and why.
    Relaunch(&'static str),
}

/// The route for an exec of an arm64-only program. `kept` gives a reason the kernel cannot be handed this exec as it is,
/// if one applies; it is asked only of a process the kernel started as the program.
fn route(kernel_route: bool, kept: impl FnOnce() -> Option<&'static str>) -> Route {
    if !kernel_route {
        return Route::Relaunch("not started by the kernel as the program");
    }
    match kept() {
        Some(why) => Route::Relaunch(why),
        None => Route::Kernel,
    }
}

/// Why this process must still re-launch its arm64-only children although the kernel started it as the program, if
/// it must: names mean something else to it than to the kernel (a sysroot or overlay root is in force, or the working
/// directory has a guest record, which only the re-launch's options carry); it is sandboxed (the child inherits the
/// sandbox, and a child the kernel starts must then read the translator's configuration and file under it, which a
/// re-launched one does not); or the GPU bridge is on (an app's helper processes take it from the options a re-launch
/// passes, and a child the kernel starts takes its own configuration's).
fn kept_reason(rewriting: bool, cwd_record: bool, sandboxed: bool, gpu: bool) -> Option<&'static str> {
    if rewriting {
        return Some("a sysroot or overlay root is in force");
    }
    if cwd_record {
        return Some("the working directory has a guest record");
    }
    if sandboxed {
        return Some("this process is sandboxed");
    }
    if gpu {
        return Some("the GPU bridge is on");
    }
    None
}

/// Why this exec, by its own program and arguments, must keep the re-launch under the kernel route, if it must. A
/// child the kernel starts is started by the kernel's translator runtime, which hands the engine the program's real path
/// as the child's argv[0] and the arguments as they are; the re-launch passes the path as the caller spelled it and the
/// arguments converted. So it keeps:
/// - a relative path (its argv[0] would become absolute, so a program that finds files from it after a chdir, or a
///   chdir file action before the kernel's lookup, would see others);
/// - an argument that is not UTF-8 (the engine reads its arguments as UTF-8), and more arguments than the runtime hands
///   on (RUNTIME_ARGS); arguments that cannot be read are the kernel's to refuse (EFAULT);
/// - a path with an empty, `.` or `..` name in it (`/opt/pkg/bin/../helper`: its argv[0] would lose the directory the
///   caller named, so a program that finds files from its argv[0]'s directories would find others);
/// - a path that does not resolve, whose last name is a link, or whose last name is spelled otherwise than its file's
///   (the child's argv[0], and the directory it names, would be another's: /usr/bin/view to vim, say);
/// - a directory in the path, other than a link, spelled otherwise than its own name (another case, or another Unicode
///   form, on a volume that ignores them: the child's argv[0] would carry the directory's own spelling). A link to a
///   directory is followed as the kernel follows it, which gives the real path as argv[0] (accepted, as for a program
///   launchd starts);
/// - a set-user-ID or set-group-ID program (the kernel would start the child, and with it the runtime and the engine,
///   with the file's owner or group; the re-launch execs the translator file, so the child runs with the caller's, as
///   before);
/// - a file the kernel does not exec: only a thin 64-bit Mach-O or a big-endian 32-bit fat file (the translator also
///   reads little-endian and 64-bit fat files);
/// - a file whose arm64 slice the kernel would choose otherwise than the translator: the kernel's signature for the
///   process is the slice it runs, the code is the slice the translator loads. The kernel takes the first arm64e slice,
///   else the first arm64 one, or, given a binary preference, the one it names (so any preference, where there is more
///   than one arm64 slice);
/// - a main-thread stack size the program declares (LC_MAIN): the kernel gives the runtime that stack, and the runtime
///   needs more of it than a small one has (the re-launch's engine gets an ordinary stack).
fn kept_for_call(n: u64, args: &[u64; 8], path: &str) -> Option<&'static str> {
    if !path.starts_with('/') {
        return Some("a relative path");
    }
    if path.split('/').skip(1).any(|name| name.is_empty() || name == "." || name == "..") {
        return Some("the path has an empty, . or .. name");
    }
    let aidx = if n == SYS_EXECVE { 1 } else { 3 };
    if let Some(why) = argv_kept(args[aidx]) {
        return Some(why);
    }
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Some("its last name is a link");
    }
    let Ok(real) = std::fs::canonicalize(path) else { return Some("the path does not resolve") };
    let spelled = std::path::Path::new(path).file_name();
    if spelled.is_none() || spelled != real.file_name() {
        return Some("its last name is spelled otherwise than its file's");
    }
    let names: Vec<&str> = path.split('/').skip(1).collect();
    let mut dir = String::new();
    for name in &names[..names.len() - 1] {
        dir.push('/');
        dir.push_str(name);
        if std::fs::symlink_metadata(&dir).is_ok_and(|m| m.file_type().is_symlink()) {
            continue;
        }
        let Ok(own) = std::fs::canonicalize(&dir) else { return Some("the path does not resolve") };
        if own.file_name() != Some(std::ffi::OsStr::new(name)) {
            return Some("a directory in it is spelled otherwise than its own name");
        }
    }
    let Ok(meta) = std::fs::metadata(&real) else { return Some("the path does not resolve") };
    if std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o6000 != 0 {
        return Some("a set-user-ID or set-group-ID program");
    }
    let mut magic = [0u8; 4];
    let Ok(mut f) = std::fs::File::open(path) else { return Some("the path does not resolve") };
    if f.read_exact(&mut magic).is_err() || !matches!(magic, [0xca, 0xfe, 0xba, 0xbe] | [0xcf, 0xfa, 0xed, 0xfe]) {
        return Some("a file the kernel does not exec");
    }
    let Ok(slices) = crate::slices::read_slices(&mut f) else { return Some("a file the kernel does not exec") };
    // Closed before open_arm64 opens the file again: a caller with one descriptor free must get the same answer.
    drop(f);
    let arm64: Vec<u32> = slices.iter().filter(|s| s.cpu == CPU_TYPE_ARM64).map(|s| s.subtype).collect();
    let kernel = arm64.iter().position(|&s| s & 0x00ff_ffff == 2).unwrap_or(0);
    let engine = (0..arm64.len()).min_by_key(|&i| crate::macho::slice_rank(arm64[i], false)).unwrap_or(0);
    if kernel != engine {
        return Some("the kernel and the translator would run different arm64 slices");
    }
    if arm64.len() > 1 && n != SYS_EXECVE && binary_preference(args[2]) != Some(false) {
        return Some("a binary preference, with more than one arm64 slice");
    }
    match crate::macho::MachO::open_arm64(path, false) {
        Ok(m) if main_stack_size(&m.header_bytes) == 0 => None,
        Ok(_) => Some("it declares a main-thread stack size"),
        Err(_) => Some("a file the kernel does not exec"),
    }
}

/// Of the guest's argv at `p`: a reason to keep the re-launch (an argument that is not UTF-8, too many arguments), or
/// None, also where it cannot be read (the kernel answers that).
fn argv_kept(p: u64) -> Option<&'static str> {
    if p == 0 {
        return None;
    }
    for i in 0..=RUNTIME_ARGS as u64 {
        let e = crate::guestmem::read_u64(p + i * 8)?;
        if e == 0 {
            return None;
        }
        if i as usize == RUNTIME_ARGS {
            return Some("more arguments than the kernel's runtime hands on");
        }
        if std::str::from_utf8(&guest_cstr(e)?).is_err() {
            return Some("an argument is not UTF-8");
        }
    }
    None
}

/// The guest's NUL-terminated string at `a`, read without faulting; None if it cannot be read (or runs past 1 MiB).
fn guest_cstr(mut a: u64) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = [0u8; 256];
    while out.len() < 1 << 20 {
        let n = (4096 - (a % 4096) as usize).min(buf.len());
        if !crate::guestmem::read(a, &mut buf[..n]) {
            return None;
        }
        if let Some(z) = buf[..n].iter().position(|&b| b == 0) {
            out.extend_from_slice(&buf[..z]);
            return Some(out);
        }
        out.extend_from_slice(&buf[..n]);
        a += n as u64;
    }
    None
}

/// Whether posix_spawn's argument descriptor at `adesc` gives a binary preference (psa_binprefs, in the attribute
/// block it points to: XNU's bsd/sys/spawn_internal.h). Some(false) for no descriptor, no attributes, or none set; None
/// where it cannot be read.
fn binary_preference(adesc: u64) -> Option<bool> {
    if adesc == 0 {
        return Some(false);
    }
    let size = crate::guestmem::read_u64(adesc)?;
    let attrp = crate::guestmem::read_u64(adesc + 8)?;
    if attrp == 0 {
        return Some(false);
    }
    if size < 32 {
        return None;
    }
    let mut prefs = [0u8; 16];
    if !crate::guestmem::read(attrp + 16, &mut prefs) {
        return None;
    }
    Some(prefs.iter().any(|&b| b != 0))
}

/// LC_MAIN's stacksize in a Mach-O's header and load commands (0 if it has none).
fn main_stack_size(h: &[u8]) -> u64 {
    let rd32 = |o: usize| h.get(o..o + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    let ncmds = rd32(16);
    let mut p = 32usize;
    for _ in 0..ncmds {
        let (cmd, size) = (rd32(p), rd32(p + 4) as usize);
        if size < 8 {
            break;
        }
        if cmd == crate::macho::LC_MAIN {
            return h.get(p + 16..p + 24).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
        }
        p += size;
    }
    0
}

unsafe fn cstr(p: u64) -> String {
    if p == 0 {
        return String::new();
    }
    std::ffi::CStr::from_ptr(p as *const libc::c_char).to_string_lossy().into_owned()
}

unsafe fn read_argv(p: u64) -> Vec<String> {
    let mut v = Vec::new();
    if p == 0 {
        return v;
    }
    let mut i = 0;
    loop {
        let e = *((p + i * 8) as *const u64);
        if e == 0 || i > 4096 {
            break;
        }
        v.push(cstr(e));
        i += 1;
    }
    v
}

/// Host processes (native children and the maclator re-launcher) must not inherit the arm64
/// GPU shim in DYLD_INSERT_LIBRARIES; maclator injects it into guests itself (`--gpu`).
/// Returns the scrubbed environment (keep alive until the call), and points `args` at it.
fn scrub_shim(n: u64, args: &mut [u64; 8]) -> Option<(Vec<CString>, Vec<u64>)> {
    let eidx = if n == SYS_EXECVE { 2 } else { 4 };
    let shim = crate::gpu::shim_path()?;
    let env = unsafe { read_argv(args[eidx]) };
    if !env.iter().any(|e| e.starts_with("DYLD_INSERT_LIBRARIES=") && e.contains(&shim)) {
        return None;
    }
    let cleaned: Vec<CString> = env
        .into_iter()
        .filter_map(|e| {
            if let Some(v) = e.strip_prefix("DYLD_INSERT_LIBRARIES=") {
                let rest: Vec<&str> = v.split(':').filter(|p| *p != shim && !p.is_empty()).collect();
                if rest.is_empty() {
                    return None;
                }
                return CString::new(format!("DYLD_INSERT_LIBRARIES={}", rest.join(":"))).ok();
            }
            CString::new(e).ok()
        })
        .collect();
    let mut ptrs: Vec<u64> = cleaned.iter().map(|c| c.as_ptr() as u64).collect();
    ptrs.push(0);
    args[eidx] = ptrs.as_ptr() as u64;
    Some((cleaned, ptrs))
}

pub fn exec_like(n: u64, args: &[u64; 8]) -> SysResult {
    exec_with(n, args, KERNEL_ROUTE.load(Ordering::SeqCst), || {
        let gpu = FORWARD.lock().unwrap().iter().any(|f| f == "--gpu");
        kept_reason(crate::paths::rewriting(), crate::syscalls::cwd_forward().is_some(), crate::sandboxcheck::sandboxed(), gpu)
    })
}

/// exec_like with the gate and this process's kept reason given (tests give their own).
fn exec_with(n: u64, args: &[u64; 8], kernel_route: bool, kept: impl FnOnce() -> Option<&'static str>) -> SysResult {
    let (pidx, aidx) = if n == SYS_EXECVE { (0, 1) } else { (1, 3) };
    let original_path = unsafe { cstr(args[pidx]) };
    let path = crate::paths::redirect(&original_path).unwrap_or(original_path);
    if path.is_empty() {
        return hostsys::unix(n, args);
    }
    let mut args = *args;
    let physical_path = CString::new(path.as_str()).ok();
    if let Some(ref c) = physical_path {
        args[pidx] = c.as_ptr() as u64;
    }
    let _env = scrub_shim(n, &mut args);
    let args = &args;
    if !arm64_only(&path) {
        // A native (x86-64 or universal) program starts where the
        // calling thread stands: for a guest working directory under a cache prefix, the sysroot's directory, so its
        // relative names inside reach the sysroot's files, as the guest's do. H v9 moved the thread to the guest path
        // on the host first (native_cwd), which gave the right parent for `..` but the host cache's files (or none)
        // for every relative name inside; no one host directory gives both.
        return hostsys::unix(n, args);
    }
    let how = route(kernel_route, || kept().or_else(|| kept_for_call(n, args, &path)));
    if crate::syscalls::TRACE.load(Ordering::Relaxed) {
        match how {
            Route::Kernel => eprintln!("[spawn] #{n} {path}: to the kernel"),
            Route::Relaunch(why) => eprintln!("[spawn] #{n} {path}: re-launched ({why})"),
        }
    }
    if how == Route::Kernel {
        // The program, its arguments, environment, file actions and attributes, as the guest gave them.
        return hostsys::unix(n, args);
    }
    let Ok(me) = std::env::current_exe() else { return hostsys::unix(n, args) };
    let argv = unsafe { read_argv(args[aidx]) };
    let mut new_argv: Vec<String> = vec![me.to_string_lossy().into_owned()];
    new_argv.extend(FORWARD.lock().unwrap().iter().cloned());
    // The working directory survives exec, its record does not; the child checks it.
    // Only the prefix is passed; the child reads its own place below the sysroot (syscalls::cwd_adopt).
    if let Some(c) = crate::syscalls::cwd_forward() {
        new_argv.push("--guest-cwd".into());
        new_argv.push(c);
    }
    new_argv.push(path);
    new_argv.extend(argv.into_iter().skip(1));
    let cstrs: Vec<CString> = new_argv.iter().filter_map(|s| CString::new(s.as_str()).ok()).collect();
    let mut ptrs: Vec<u64> = cstrs.iter().map(|c| c.as_ptr() as u64).collect();
    ptrs.push(0);
    let mut a = *args;
    a[pidx] = cstrs[0].as_ptr() as u64;
    a[aidx] = ptrs.as_ptr() as u64;
    let r = hostsys::unix(n, &a);
    drop(cstrs);
    r
}


#[cfg(test)]
mod tests {
    //! The child processes here are a thin arm64 program built with the system's clang. On an Apple silicon Mac (the
    //! tests under Rosetta 2) the kernel runs it natively when it is handed the exec: a stand-in for the
    //! foreign-architecture exec, which only a kernel that has one can give. On an Intel Mac the tests that need the
    //! child to run as itself say NO READING and return; the rest run there too.
    use super::*;
    use std::cell::RefCell;
    use std::path::PathBuf;

    const SYS_POSIX_SPAWN: u64 = 244;

    const CHILD: &str = r#"#include <libproc.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv) {
    char p[4096];
    int n = proc_pidpath(getpid(), p, sizeof p);
    if (argc > 1 && argv[1][0] == '/') {
        FILE *f = fopen(argv[1], "w");
        if (!f) return 3;
        fprintf(f, "%s\n%s\n", n > 0 ? p : "?", argv[0]);
        fclose(f);
    }
    return 7;
}
"#;

    /// Whether this Mac's kernel runs an arm64 program itself (Apple silicon; also for a test under Rosetta 2).
    fn host_runs_arm64() -> bool {
        let mut v: i32 = 0;
        let mut n = std::mem::size_of::<i32>();
        let r = unsafe {
            libc::sysctlbyname(c"hw.optional.arm64".as_ptr(), &mut v as *mut i32 as *mut libc::c_void, &mut n, std::ptr::null_mut(), 0)
        };
        r == 0 && v == 1
    }

    fn no_reading_without_arm64() -> bool {
        if host_runs_arm64() {
            return false;
        }
        eprintln!("NO READING: this Mac's kernel does not run arm64 programs, so a child handed to it cannot run as itself here");
        true
    }

    /// A test's own directory (by its real path) and, in it, a thin arm64 program (linker-signed ad hoc) that writes its
    /// main image as the kernel names it (proc_pidpath) and its argv[0] to the file argv[1] names, if that is an
    /// absolute path (so a child started by mistake with other arguments writes nothing), and exits 7. Made
    /// per test, not kept in a static: the crate keeps no std lock in a static (forksafe's census). What it made is
    /// removed when it is dropped, file by file, then its directories (left if anything else is in them).
    struct Fx {
        dir: PathBuf,
        child: String,
        made: RefCell<Vec<PathBuf>>,
    }

    impl Fx {
        fn made(&self, p: PathBuf) -> PathBuf {
            self.made.borrow_mut().push(p.clone());
            p
        }
    }

    impl Drop for Fx {
        fn drop(&mut self) {
            for p in self.made.borrow().iter().rev() {
                let _ = std::fs::remove_file(p);
            }
            let _ = std::fs::remove_dir(self.dir.join("sub"));
            let _ = std::fs::remove_dir(&self.dir);
        }
    }

    fn fx(test: &str) -> Fx {
        let d = std::env::temp_dir().join(format!("mcl-spawn-{}-{test}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let mut f = Fx { dir: std::fs::canonicalize(&d).unwrap(), child: String::new(), made: RefCell::new(Vec::new()) };
        let src = f.made(f.dir.join("child.c"));
        std::fs::write(&src, CHILD).unwrap();
        let out = f.made(f.dir.join("arm64-child"));
        // On Apple silicon through arch: a child of a process under Rosetta 2 takes a universal program's x86_64 slice,
        // and the compiler driver's own library has none. An Intel Mac's clang builds for arm64 as it is.
        let mut cc = if host_runs_arm64() {
            let mut c = std::process::Command::new("/usr/bin/arch");
            c.args(["-arm64", "/usr/bin/clang"]);
            c
        } else {
            std::process::Command::new("/usr/bin/clang")
        };
        let st = cc.args(["-arch", "arm64", "-O1", "-o"]).arg(&out).arg(&src).status().expect("run /usr/bin/clang");
        assert!(st.success(), "clang: {st}");
        f.child = out.to_string_lossy().into_owned();
        assert!(arm64_only(&f.child), "the child program is not arm64-only");
        f
    }

    fn link(f: &Fx, name: &str, to: &str) -> String {
        let p = f.made(f.dir.join(name));
        let _ = std::fs::remove_file(&p);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).unwrap();
        }
        std::os::unix::fs::symlink(to, &p).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn file(f: &Fx, name: &str, bytes: &[u8]) -> String {
        let p = f.made(f.dir.join(name));
        std::fs::write(&p, bytes).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn report(f: &Fx, name: &str) -> PathBuf {
        let r = f.made(f.dir.join(name));
        let _ = std::fs::remove_file(&r);
        r
    }

    fn me() -> String {
        std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap().to_string_lossy().into_owned()
    }

    /// The same file spelled without the /private firmlink's prefix (/var for /private/var), if it has one.
    fn other_spelling(p: &str) -> Option<String> {
        let s = p.strip_prefix("/private")?.to_string();
        (file_id(&s).is_some() && file_id(&s) == file_id(p)).then_some(s)
    }

    /// A fat file of `slices` (cpu, subtype, bytes), 32- or 64-bit, in either byte order, each slice 16 KiB aligned
    /// (tests/check_spawn_formats.py's layout).
    fn fat(slices: &[(u32, u32, &[u8])], wide: bool, be: bool) -> Vec<u8> {
        let w32 = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&if be { x.to_be_bytes() } else { x.to_le_bytes() });
        let w64 = |v: &mut Vec<u8>, x: u64| v.extend_from_slice(&if be { x.to_be_bytes() } else { x.to_le_bytes() });
        let stride = if wide { 32 } else { 20 };
        let mut offset = (8 + slices.len() * stride + 16383) & !16383;
        let mut table = Vec::new();
        w32(&mut table, if wide { 0xcafe_babf } else { 0xcafe_babe });
        w32(&mut table, slices.len() as u32);
        let mut body = vec![0u8; offset];
        for (cpu, sub, payload) in slices {
            w32(&mut table, *cpu);
            w32(&mut table, *sub);
            if wide {
                w64(&mut table, offset as u64);
                w64(&mut table, payload.len() as u64);
                w32(&mut table, 14);
                w32(&mut table, 0);
            } else {
                w32(&mut table, offset as u32);
                w32(&mut table, payload.len() as u32);
                w32(&mut table, 14);
            }
            body.extend_from_slice(payload);
            offset = (body.len() + 16383) & !16383;
            body.resize(offset, 0);
        }
        body[..table.len()].copy_from_slice(&table);
        body
    }

    fn cstrings(v: &[&str]) -> (Vec<CString>, Vec<u64>) {
        let c: Vec<CString> = v.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut p: Vec<u64> = c.iter().map(|s| s.as_ptr() as u64).collect();
        p.push(0);
        (c, p)
    }

    /// posix_spawn `path` with `argv` and `env` (and PATH) through exec_with, the gate given and no reason of this
    /// process's: the child's pid, or the errno.
    fn spawn(path: &str, argv: &[&str], env: &[&str], kernel_route: bool) -> Result<i32, i32> {
        let p = CString::new(path).unwrap();
        let (_a, av) = cstrings(argv);
        let mut e = vec!["PATH=/usr/bin:/bin"];
        e.extend_from_slice(env);
        let (_e, ev) = cstrings(&e);
        let mut pid: i32 = 0;
        let args = [&mut pid as *mut i32 as u64, p.as_ptr() as u64, 0, av.as_ptr() as u64, ev.as_ptr() as u64, 0, 0, 0];
        let r = exec_with(SYS_POSIX_SPAWN, &args, kernel_route, || None);
        if r.carry {
            Err(r.rax as i32)
        } else {
            Ok(pid)
        }
    }

    /// The child's exit status (1000 + the signal if one ended it), waiting at most 30 s.
    fn wait(pid: i32) -> i32 {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut st = 0;
        loop {
            let r = unsafe { libc::waitpid(pid, &mut st, libc::WNOHANG) };
            if r == pid {
                break;
            }
            assert!(r == 0 || (r < 0 && unsafe { *libc::__error() } == libc::EINTR), "waitpid");
            if std::time::Instant::now() > deadline {
                unsafe { libc::kill(pid, libc::SIGKILL) };
                unsafe { libc::waitpid(pid, &mut st, 0) };
                panic!("child {pid} still running after 30 s");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if libc::WIFEXITED(st) {
            libc::WEXITSTATUS(st)
        } else {
            1000 + libc::WTERMSIG(st)
        }
    }

    /// The test a re-launch runs when it starts this test binary as the translator (`--exact MARKER` among the guest's
    /// arguments): it writes its own arguments, one a line, to MCL_SPAWN_MARK. Without that variable it does nothing.
    const MARKER: &str = "spawn::tests::re_launch_marker";

    #[test]
    fn re_launch_marker() {
        if let Some(m) = std::env::var_os("MCL_SPAWN_MARK") {
            let a: String = std::env::args().map(|x| format!("{x}\n")).collect();
            std::fs::write(m, a).unwrap();
        }
    }

    #[test]
    fn a_translator_started_directly_reads_as_not_started_by_the_kernel_as_the_program() {
        // This test process is its own main image, as a translator started directly is. Both answers must be read, and
        // name one file, for the "no" to mean anything.
        let (m, e) = (main_image(), engine_image());
        let (m, e) = (m.expect("proc_pidpath of this process"), e.expect("_NSGetExecutablePath"));
        assert!(file_id(&m).is_some() && file_id(&m) == file_id(&e), "{m} and {e} are not one file");
        assert!(!route_from(Some(&m), Some(&e)));
        assert!(!detect_route());
        assert!(!KERNEL_ROUTE.load(Ordering::SeqCst));
    }

    #[test]
    fn the_gate_needs_another_file_with_an_arm64_slice_as_the_main_image() {
        let f = fx("gate");
        let (c, me) = (f.child.as_str(), me());
        // The control: an arm64-only main image that is not the translator.
        assert!(route_from(Some(c), Some(&me)));
        // A universal main image whose arm64 slice the kernel ran (a spawn's binary preference): the kernel route too.
        assert!(file_id("/usr/bin/true").is_some() && file_id("/usr/bin/true") != file_id(&me));
        assert!(!arm64_only("/usr/bin/true") && has_arm64("/usr/bin/true"), "/usr/bin/true is expected to be universal");
        assert!(route_from(Some("/usr/bin/true"), Some(&me)));
        // One file, however spelled, is not two.
        assert!(!route_from(Some(c), Some(c)));
        assert!(!route_from(Some(&link(&f, "same-child", c)), Some(c)), "a link to the main image");
        assert!(!route_from(Some(c), Some(&link(&f, "same-child-2", c))), "a link to the translator");
        if let Some(s) = other_spelling(c) {
            assert!(!route_from(Some(&s), Some(c)), "{s} for {c}");
        }
        // A main image with no arm64 slice (an x86_64 copy of this test binary; a text file) is not one.
        let x = f.made(f.dir.join("x86-copy"));
        std::fs::copy(&me, &x).unwrap();
        let x = x.to_string_lossy().into_owned();
        assert!(file_id(&x).is_some() && file_id(&x) != file_id(&me), "the copy is another file");
        assert!(!has_arm64(&x), "this test binary is expected to have no arm64 slice");
        assert!(!route_from(Some(&x), Some(&me)));
        let src = f.dir.join("child.c").to_string_lossy().into_owned();
        assert!(!route_from(Some(&src), Some(&me)));
        // Anything unread is no.
        let missing = f.dir.join("missing").to_string_lossy().into_owned();
        assert!(!route_from(None, Some(&me)));
        assert!(!route_from(Some(c), None));
        assert!(!route_from(Some(&missing), Some(&me)));
        assert!(!route_from(Some(c), Some(&missing)));
    }

    #[test]
    fn only_a_process_the_kernel_started_as_the_program_hands_the_exec_to_the_kernel() {
        assert_eq!(route(false, || None), Route::Relaunch("not started by the kernel as the program"));
        assert_eq!(route(false, || panic!("asked outside the kernel route")), Route::Relaunch("not started by the kernel as the program"));
        assert_eq!(route(true, || None), Route::Kernel);
        assert_eq!(route(true, || Some("why")), Route::Relaunch("why"));
        assert_eq!(kept_reason(false, false, false, false), None);
        assert_eq!(kept_reason(true, false, false, false), Some("a sysroot or overlay root is in force"));
        assert_eq!(kept_reason(false, true, false, false), Some("the working directory has a guest record"));
        assert_eq!(kept_reason(false, false, true, false), Some("this process is sandboxed"));
        assert_eq!(kept_reason(false, false, false, true), Some("the GPU bridge is on"));
    }

    #[test]
    fn an_exec_the_kernel_would_start_otherwise_keeps_the_re_launch() {
        let f = fx("call");
        let c = f.child.as_str();
        // kept_for_call on a call made of `path`, `argv` (raw bytes), and for posix_spawn an argument descriptor.
        let call = |n: u64, path: &str, argv: &[&[u8]], adesc: u64| -> Option<&'static str> {
            let raw: Vec<CString> = argv.iter().map(|a| CString::new(a.to_vec()).unwrap()).collect();
            let mut ptrs: Vec<u64> = raw.iter().map(|x| x.as_ptr() as u64).collect();
            ptrs.push(0);
            let p = CString::new(path).unwrap();
            let mut args = [0u64; 8];
            if n == SYS_EXECVE {
                args[0] = p.as_ptr() as u64;
                args[1] = ptrs.as_ptr() as u64;
            } else {
                args[1] = p.as_ptr() as u64;
                args[2] = adesc;
                args[3] = ptrs.as_ptr() as u64;
            }
            kept_for_call(n, &args, path)
        };
        let both = [SYS_EXECVE, SYS_POSIX_SPAWN];
        // The control: the thin child by its real path, either call.
        for n in both {
            assert_eq!(call(n, c, &[b"x" as &[u8], b"y"], 0), None, "#{n}");
        }
        // A relative path, either call.
        for n in both {
            assert_eq!(call(n, "arm64-child", &[b"x" as &[u8]], 0), Some("a relative path"), "#{n}");
        }
        // An argument that is not UTF-8; more arguments than the runtime hands on, and no more than that; either call.
        // The runtime refuses 4,090 engine arguments, and the recorded configuration adds 3 to the guest's argv[1..]:
        // the limit must leave room for them (so this fails if RUNTIME_ARGS is raised past what the runtime takes).
        assert!(RUNTIME_ARGS + 3 < 4090);
        let many: Vec<&[u8]> = vec![&b"a"[..]; RUNTIME_ARGS + 1];
        let enough: Vec<&[u8]> = vec![&b"a"[..]; RUNTIME_ARGS];
        for n in both {
            assert_eq!(call(n, c, &[b"x" as &[u8], b"\xff"], 0), Some("an argument is not UTF-8"), "#{n}");
            assert_eq!(call(n, c, &many, 0), Some("more arguments than the kernel's runtime hands on"), "#{n}");
            assert_eq!(call(n, c, &enough, 0), None, "#{n}");
        }
        // An argv the engine cannot read is the kernel's to refuse: no reason here.
        let mut args = [0u64; 8];
        let p = CString::new(c).unwrap();
        args[0] = p.as_ptr() as u64;
        args[1] = 0x10;
        assert_eq!(kept_for_call(SYS_EXECVE, &args, c), None, "an unreadable argv");
        let bad = [0x10u64, 0];
        args[1] = bad.as_ptr() as u64;
        assert_eq!(kept_for_call(SYS_EXECVE, &args, c), None, "an unreadable argument");
        // A link as the last name, of another name or the same; a link in a directory above is fine.
        assert_eq!(call(SYS_EXECVE, &link(&f, "view", c), &[b"x" as &[u8]], 0), Some("its last name is a link"));
        assert_eq!(call(SYS_EXECVE, &link(&f, "sub/arm64-child", c), &[b"x" as &[u8]], 0), Some("its last name is a link"));
        let here = f.dir.to_string_lossy().into_owned();
        let up = link(&f, "dir-link", &here);
        assert_eq!(call(SYS_EXECVE, &format!("{up}/arm64-child"), &[b"x" as &[u8]], 0), None, "a link in a directory above the program");
        // A path with an empty, . or .. name, either call: the child itself, spelled through them (sub/ is the directory
        // the link above was made in).
        for p in [format!("{here}/sub/../arm64-child"), format!("{here}/./arm64-child"), format!("{here}//arm64-child")] {
            assert!(std::fs::canonicalize(&p).is_ok_and(|r| r.to_string_lossy() == c), "{p} is not the child");
            for n in both {
                assert_eq!(call(n, &p, &[b"x" as &[u8]], 0), Some("the path has an empty, . or .. name"), "{p} #{n}");
            }
        }
        let missing = f.dir.join("missing").to_string_lossy().into_owned();
        assert_eq!(call(SYS_EXECVE, &missing, &[b"x" as &[u8]], 0), Some("the path does not resolve"));
        // The last name spelled otherwise than its file's: on a case-insensitive volume (macOS's default) the upper-case
        // name opens the child; on a case-sensitive one it names nothing.
        let upper = f.dir.join("ARM64-CHILD").to_string_lossy().into_owned();
        let want = if std::fs::metadata(&upper).is_ok() { "its last name is spelled otherwise than its file's" } else { "the path does not resolve" };
        assert_eq!(call(SYS_EXECVE, &upper, &[b"x" as &[u8]], 0), Some(want));
        // Formats: a big-endian 32-bit fat file goes to the kernel; the other three, which the translator also reads, do not.
        let thin = std::fs::read(c).unwrap();
        for (wide, be, want) in [
            (false, true, None),
            (false, false, Some("a file the kernel does not exec")),
            (true, true, Some("a file the kernel does not exec")),
            (true, false, Some("a file the kernel does not exec")),
        ] {
            let name = format!("fat{}-{}", if wide { 64 } else { 32 }, if be { "be" } else { "le" });
            let p = file(&f, &name, &fat(&[(CPU_TYPE_ARM64, 0u32, &thin[..])], wide, be));
            assert!(arm64_only(&p), "{name} is not arm64-only to the translator");
            assert_eq!(call(SYS_EXECVE, &p, &[b"x" as &[u8]], 0), want, "{name}");
        }
        // Slices, in fat-table order: the kernel takes the first arm64e (any ABI bits), else the first arm64 slice; the
        // translator the first of its best rank (plain arm64, then arm64e, then any other).
        let diff = Some("the kernel and the translator would run different arm64 slices");
        for (subs, want) in [
            (&[0u32, 2][..], diff),
            (&[1, 2][..], diff),
            (&[0, 0x8000_0002][..], diff),
            (&[0x0c, 0][..], diff),
            (&[2, 0x0c][..], None),
            (&[0x0c, 2][..], None),
            (&[0, 0x0c][..], None),
            (&[0, 1][..], None),
        ] {
            let sl: Vec<(u32, u32, &[u8])> = subs.iter().map(|s| (CPU_TYPE_ARM64, *s, &thin[..])).collect();
            let name = format!("fat-{}", subs.iter().map(|s| format!("{s:x}")).collect::<Vec<_>>().join("-"));
            let p = file(&f, &name, &fat(&sl, false, true));
            assert!(arm64_only(&p), "{name} is not arm64-only to the translator");
            assert_eq!(call(SYS_EXECVE, &p, &[b"x" as &[u8]], 0), want, "{name}");
        }
        // A binary preference in posix_spawn's attributes, where there is more than one arm64 slice (here arm64e and
        // arm64e.x1, where the two agree without one): the attribute block (psa_binprefs at offset 16) through the
        // argument descriptor (attr_size, attrp).
        let two = file(&f, "fat-2-c", &fat(&[(CPU_TYPE_ARM64, 2, &thin[..]), (CPU_TYPE_ARM64, 0x0c, &thin[..])], false, true));
        let mut attr = [0u8; 256];
        let desc = |attr: &[u8; 256]| [attr.len() as u64, attr.as_ptr() as u64, 0, 0, 0, 0, 0, 0];
        let d0 = desc(&attr);
        assert_eq!(call(SYS_POSIX_SPAWN, &two, &[b"x" as &[u8]], d0.as_ptr() as u64), None, "attributes, no preference");
        attr[16..20].copy_from_slice(&CPU_TYPE_ARM64.to_le_bytes());
        let d1 = desc(&attr);
        assert_eq!(call(SYS_POSIX_SPAWN, &two, &[b"x" as &[u8]], d1.as_ptr() as u64), Some("a binary preference, with more than one arm64 slice"));
        assert_eq!(call(SYS_POSIX_SPAWN, c, &[b"x" as &[u8]], d1.as_ptr() as u64), None, "a preference, one arm64 slice");
        // execve has no attributes: its third argument is the environment, which must not be read as a descriptor even
        // where it would read as one with a preference (here d1 itself).
        let mut ex = [0u64; 8];
        let (tp, (_xa, xv)) = (CString::new(two.as_str()).unwrap(), cstrings(&["x"]));
        ex[0] = tp.as_ptr() as u64;
        ex[1] = xv.as_ptr() as u64;
        ex[2] = d1.as_ptr() as u64;
        assert_eq!(binary_preference(ex[2]), Some(true), "the control: read as a descriptor, it gives a preference");
        assert_eq!(kept_for_call(SYS_EXECVE, &ex, &two), None, "execve has no preference");
        // A main-thread stack size the program declares: the child with LC_MAIN's stacksize set.
        let mut sized = thin.clone();
        let h = crate::macho::MachO::open_arm64(c, false).unwrap().header_bytes;
        let rd32 = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let (mut at, mut q) = (None, 32usize);
        for _ in 0..rd32(16) {
            if rd32(q) == crate::macho::LC_MAIN {
                at = Some(q + 16);
            }
            q += rd32(q + 4) as usize;
        }
        let at = at.expect("the child has an LC_MAIN");
        assert_eq!(main_stack_size(&h), 0);
        sized[at..at + 8].copy_from_slice(&0x4000u64.to_le_bytes());
        let sized = file(&f, "arm64-child-stack", &sized);
        assert_eq!(call(SYS_EXECVE, &sized, &[b"x" as &[u8]], 0), Some("it declares a main-thread stack size"));
        // A set-user-ID or set-group-ID program, either call: copies of the child with the bit set (the group made this
        // process's own first, as the owner may set S_ISGID only for a group it is in); the mode is read back, so a bit
        // the volume dropped fails here and not as a pass.
        use std::os::unix::fs::PermissionsExt;
        for (name, mode) in [("arm64-child-suid", 0o4755), ("arm64-child-sgid", 0o2755)] {
            let p = file(&f, name, &thin);
            std::os::unix::fs::chown(&p, None, Some(unsafe { libc::getegid() })).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o7777, mode, "{name}: the mode was not set");
            for n in both {
                assert_eq!(call(n, &p, &[b"x" as &[u8]], 0), Some("a set-user-ID or set-group-ID program"), "{name} #{n}");
            }
        }
        // A directory in the path spelled otherwise than its own name (another case), either call: a copy of the child in
        // sub/ (the directory the links above were made in), by sub/ and by SUB/. On a case-sensitive volume SUB/ names
        // nothing.
        let copy = file(&f, "sub/arm64-copy", &thin);
        assert_eq!(call(SYS_EXECVE, &copy, &[b"x" as &[u8]], 0), None, "the control: the copy by its own spelling");
        let other = format!("{here}/SUB/arm64-copy");
        let want = if std::fs::metadata(&other).is_ok() { "a directory in it is spelled otherwise than its own name" } else { "the path does not resolve" };
        for n in both {
            assert_eq!(call(n, &other, &[b"x" as &[u8]], 0), Some(want), "SUB/ #{n}");
        }
        // The same by the path's first directory in another case (/PRIVATE/var/... for a fixture under /private/var),
        // so a check that left out the leading directories fails here too.
        let first = here.split('/').nth(1).unwrap_or_default();
        let upper = first.to_uppercase();
        if upper != first {
            let other = format!("/{upper}{}/sub/arm64-copy", &here[1 + first.len()..]);
            let want = if std::fs::metadata(&other).is_ok() { "a directory in it is spelled otherwise than its own name" } else { "the path does not resolve" };
            for n in both {
                assert_eq!(call(n, &other, &[b"x" as &[u8]], 0), Some(want), "{upper}/ #{n}");
            }
        }
    }

    #[test]
    fn a_caller_with_one_descriptor_free_gets_the_same_answer() {
        // kept_for_call opens the file one descriptor at a time. In a fresh process with every descriptor below its soft
        // limit in use but one, the child by its real path still goes to the kernel; the control shows exactly one was
        // free there, and that the call goes to the kernel with descriptors to spare.
        let f = fx("one-fd");
        let child = f.child.clone();
        let p = CString::new(child.as_str()).unwrap();
        let (_a, av) = cstrings(&["x"]);
        let mut args = [0u64; 8];
        args[0] = p.as_ptr() as u64;
        args[1] = av.as_ptr() as u64;
        assert_eq!(kept_for_call(SYS_EXECVE, &args, &child), None, "the control: with descriptors to spare");
        let code = crate::sandboxcheck::tests::in_child(|| {
            let null = c"/dev/null";
            let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) } != 0 {
                return 6;
            }
            rl.rlim_cur = rl.rlim_cur.min(512);
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rl) } != 0 {
                return 6;
            }
            // open gives the lowest free descriptor: open until the limit refuses, then free the last one given.
            let mut last = -1;
            loop {
                let fd = unsafe { libc::open(null.as_ptr(), libc::O_RDONLY) };
                if fd >= 0 {
                    last = fd;
                } else if unsafe { *libc::__error() } == libc::EMFILE && last >= 0 {
                    break;
                } else {
                    return 5;
                }
            }
            unsafe { libc::close(last) };
            let one = unsafe { libc::open(null.as_ptr(), libc::O_RDONLY) };
            let two = unsafe { libc::open(null.as_ptr(), libc::O_RDONLY) };
            if one != last || two != -1 {
                return 4;
            }
            unsafe { libc::close(one) };
            match kept_for_call(SYS_EXECVE, &args, &child) {
                None => 0,
                Some(_) => 2,
            }
        });
        assert_eq!(code, 0, "2 a reason given with one descriptor free, 4 the control (not exactly one free), 5/6 setup");
    }

    #[test]
    fn under_the_kernel_route_an_arm64_only_child_is_started_by_the_kernel_as_itself() {
        if no_reading_without_arm64() {
            return;
        }
        let f = fx("as-itself");
        let c = f.child.as_str();
        let r = report(&f, "as-itself.txt");
        let pid = spawn(c, &["given-argv0", r.to_str().unwrap()], &[], true).expect("posix_spawn");
        assert_eq!(wait(pid), 7, "the child is not the program (a re-launch runs this test binary, which exits 0)");
        let text = std::fs::read_to_string(&r).expect("the child wrote no report");
        let mut l = text.lines();
        assert_eq!(l.next(), Some(c), "the kernel's main image of the child");
        assert_eq!(l.next(), Some("given-argv0"), "the child's argv[0]");
    }

    #[test]
    fn outside_the_kernel_route_an_arm64_only_child_is_still_re_launched() {
        // The re-launch starts the translator, here this test binary, with the program's path and the guest's arguments
        // after its argv[0]; given `--exact MARKER` among them, it runs the marker test, which writes the arguments it
        // got and exits 0. So the mark is exactly [translator, path, guest's argv[1..]].
        let f = fx("re-launched");
        let c = f.child.as_str();
        let exe = std::env::current_exe().unwrap().to_string_lossy().into_owned();
        for (name, path, gate) in [("outside", c.to_string(), false), ("kept", link(&f, "view-2", c), true)] {
            let mark = report(&f, &format!("mark-{name}.txt"));
            let env = format!("MCL_SPAWN_MARK={}", mark.display());
            let pid = spawn(&path, &["given-argv0", "--exact", MARKER], &[env.as_str()], gate).expect("posix_spawn");
            assert_eq!(wait(pid), 0, "{name}: the re-launched translator did not exit 0");
            let got = std::fs::read_to_string(&mark).unwrap_or_else(|_| panic!("{name}: the re-launch did not run the marker"));
            assert_eq!(got, format!("{exe}\n{path}\n--exact\n{MARKER}\n"), "{name}");
        }
    }

    #[test]
    fn the_kernel_answers_an_exec_it_refuses() {
        // An arm64-only file without execute permission: the kernel's EACCES, where a re-launch returned a pid.
        let f = fx("refused");
        let no = f.made(f.dir.join("arm64-child-no-x"));
        std::fs::copy(&f.child, &no).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&no, std::fs::Permissions::from_mode(0o644)).unwrap();
        let no = no.to_string_lossy().into_owned();
        assert!(arm64_only(&no));
        match spawn(&no, &["x"], &[], true) {
            Err(e) => assert_eq!(e, libc::EACCES),
            Ok(pid) => {
                wait(pid);
                panic!("a non-executable program was started");
            }
        }
        // The control: a missing file is ENOENT either way (not arm64-only, so it always went to the kernel).
        let missing = f.dir.join("missing").to_string_lossy().into_owned();
        assert_eq!(spawn(&missing, &["x"], &[], true), Err(libc::ENOENT));
        assert_eq!(spawn(&missing, &["x"], &[], false), Err(libc::ENOENT));
    }

    #[test]
    fn under_the_kernel_route_an_execve_becomes_the_program() {
        if no_reading_without_arm64() {
            return;
        }
        let f = fx("execve");
        let c = f.child.as_str();
        let r = report(&f, "execve.txt");
        let rs = r.to_string_lossy().into_owned();
        let code = crate::sandboxcheck::tests::in_child(|| {
            let p = CString::new(c).unwrap();
            let (_a, av) = cstrings(&["given-argv0", &rs]);
            let (_e, ev) = cstrings(&["PATH=/usr/bin:/bin"]);
            let args = [p.as_ptr() as u64, av.as_ptr() as u64, ev.as_ptr() as u64, 0, 0, 0, 0, 0];
            let res = exec_with(SYS_EXECVE, &args, true, || None);
            if res.carry {
                100 + res.rax as i32
            } else {
                99
            }
        });
        assert_eq!(code, 7, "100+errno: the exec failed; 0: this test binary was re-launched instead");
        let text = std::fs::read_to_string(&r).expect("the child wrote no report");
        let mut l = text.lines();
        assert_eq!(l.next(), Some(c), "the kernel's main image of the child");
        assert_eq!(l.next(), Some("given-argv0"), "the child's argv[0]");
    }

    /// exec_like itself: the gate it reads, and this process's own reasons. Each case runs in a fresh process (this test
    /// binary again), as the state is the process's: a sysroot another test set, a sandbox, a configured GPU bridge.
    /// Each must show the route it took twice over: in the trace line exec_like writes, with its reason, and in the
    /// child: the program as itself, or this test binary re-launched with exactly the translator's arguments (running
    /// the marker test), or, with --gpu among them, refused by the test harness (exit 101).
    #[test]
    fn exec_like_reads_the_gate_and_this_processs_state() {
        if let Ok(mode) = std::env::var("MCL_SPAWN_WIRING") {
            std::process::exit(wiring(&mode));
        }
        if no_reading_without_arm64() {
            return;
        }
        let f = fx("wiring");
        for mode in ["kernel", "gate-off", "gpu", "sysroot", "sandbox"] {
            let r = report(&f, &format!("wiring-{mode}.txt"));
            let t = report(&f, &format!("wiring-{mode}.trace"));
            let st = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "spawn::tests::exec_like_reads_the_gate_and_this_processs_state", "--test-threads=1"])
                .env("MCL_SPAWN_WIRING", mode)
                .env("MCL_SPAWN_CHILD", &f.child)
                .env("MCL_SPAWN_REPORT", &r)
                .env("MCL_SPAWN_TRACE", &t)
                .env("MCL_SPAWN_SYSROOT", &f.dir)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap();
            assert_eq!(st.code(), Some(0), "{mode}: 1 the trace line, 2 the child, 3 no sandbox, 4 the spawn failed");
        }
    }

    /// In the fresh process: set the state `mode` names, posix_spawn the child through exec_like with tracing on (the
    /// trace to MCL_SPAWN_TRACE), and check the trace line and what started.
    fn wiring(mode: &str) -> i32 {
        let c = std::env::var("MCL_SPAWN_CHILD").unwrap();
        let r = std::env::var("MCL_SPAWN_REPORT").unwrap();
        let t = std::env::var("MCL_SPAWN_TRACE").unwrap();
        KERNEL_ROUTE.store(mode != "gate-off", Ordering::SeqCst);
        let why = match mode {
            "kernel" => None,
            "gate-off" => Some("not started by the kernel as the program"),
            "gpu" => {
                configure(vec!["--gpu".into()]);
                Some("the GPU bridge is on")
            }
            "sysroot" => {
                crate::paths::set_cache_dir(&std::env::var("MCL_SPAWN_SYSROOT").unwrap());
                Some("a sysroot or overlay root is in force")
            }
            _ => {
                if !crate::sandboxcheck::tests::apply("(version 1)(allow default)") {
                    return 3;
                }
                Some("this process is sandboxed")
            }
        };
        let p = CString::new(c.as_str()).unwrap();
        let argv: Vec<&str> = if why.is_none() { vec!["given-argv0", &r] } else { vec!["given-argv0", "--exact", MARKER] };
        let (_a, av) = cstrings(&argv);
        let mark = format!("MCL_SPAWN_MARK={r}");
        let (_e, ev) = cstrings(&["PATH=/usr/bin:/bin", &mark]);
        let mut pid: i32 = 0;
        let args = [&mut pid as *mut i32 as u64, p.as_ptr() as u64, 0, av.as_ptr() as u64, ev.as_ptr() as u64, 0, 0, 0];
        crate::syscalls::TRACE.store(true, Ordering::SeqCst);
        let tc = CString::new(t.as_str()).unwrap();
        let (fd, saved) = unsafe { (libc::open(tc.as_ptr(), libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC, 0o644), libc::dup(2)) };
        unsafe { libc::dup2(fd, 2) };
        let res = exec_like(SYS_POSIX_SPAWN, &args);
        unsafe {
            libc::dup2(saved, 2);
            libc::close(fd);
            libc::close(saved);
        }
        crate::syscalls::TRACE.store(false, Ordering::SeqCst);
        if res.carry {
            return 4;
        }
        let code = wait(pid);
        let line = match why {
            None => format!("[spawn] #244 {c}: to the kernel"),
            Some(w) => format!("[spawn] #244 {c}: re-launched ({w})"),
        };
        if !std::fs::read_to_string(&t).is_ok_and(|x| x.lines().any(|l| l == line)) {
            return 1;
        }
        let wrote = std::fs::read_to_string(&r).ok();
        let exe = std::env::current_exe().unwrap().to_string_lossy().into_owned();
        let child_ok = match (mode, why) {
            (_, None) => code == 7 && wrote.as_deref().and_then(|x| x.lines().next()) == Some(c.as_str()),
            ("gpu", _) => code == 101 && wrote.is_none(),
            _ => code == 0 && wrote.as_deref() == Some(format!("{exe}\n{c}\n--exact\n{MARKER}\n").as_str()),
        };
        if child_ok {
            0
        } else {
            2
        }
    }
}
