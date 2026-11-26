//! Execution engine: runs a guest thread until it exits.
//!
//! Tiering: translated host code (AOT cache or JIT) when available, with the
//! reference interpreter as the fallback for anything not translated.

use maclator_core::cpu::Cpu;
use maclator_core::disasm::disasm;
use maclator_core::interp::{self, Exit};
use std::cell::Cell;
use std::sync::atomic::{AtomicI32, Ordering};

thread_local! {
    static EXIT_REQUESTED: Cell<bool> = const { Cell::new(false) };
}

/// Ask the current guest thread to stop after the current instruction.
pub fn exit_current_thread() {
    EXIT_REQUESTED.with(|f| f.set(true));
}

pub fn invalidate_range(_addr: u64, _len: u64) {
    crate::jit::invalidate_range(_addr, _len);
}

pub fn flush_profile() {
    crate::jit::flush_profile();
}

/// Registry of running guest CPUs (for SIGUSR1 diagnostics).
pub(crate) static CPUS: crate::forksync::FMutex<Vec<usize>> = crate::forksync::FMutex::new(Vec::new());

/// In a fork child, only the forking thread's CPU is running.
/// And the parent's dump thread is not there; start the child's own, on a
/// pipe of its own (the inherited one is shared with the parent's dump thread).
pub(crate) fn reset_in_child() {
    let me = CURRENT_CPU.with(|c| c.get());
    CPUS.lock().unwrap().retain(|&k| k == me);
    if DUMP_W.load(Ordering::SeqCst) >= 0 {
        let (r, w) = (DUMP_R.swap(-1, Ordering::SeqCst), DUMP_W.swap(-1, Ordering::SeqCst));
        unsafe {
            libc::close(r);
            libc::close(w);
        }
        start_dump_thread();
    }
}

// The SIGUSR1 handler takes no lock and allocates nothing. It
// writes one byte to a pipe, and a dump thread (started with the handler) does the dump outside
// any signal handler. J v3's handler formatted text and took the symbol locks on whatever thread
// the signal landed on: on a thread interrupted inside malloc, while another thread held those
// locks inside fork() waiting for malloc's lock, that was a deadlock.
static DUMP_R: AtomicI32 = AtomicI32::new(-1);
/// Dumps done (tests: the SIGUSR1 storm's positive control).
static DUMPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dumps_done() -> u64 {
    DUMPS.load(Ordering::Relaxed)
}
static DUMP_W: AtomicI32 = AtomicI32::new(-1);

/// The dump request pipe's two ends (-1 when there is none): what an AOT builder keeps when it closes the rest.
pub(crate) fn dump_pipe() -> [i32; 2] {
    [DUMP_R.load(Ordering::SeqCst), DUMP_W.load(Ordering::SeqCst)]
}

/// SIGUSR2 lines handled (tests: the positive control of the SIGUSR2 scenario).
static USR2_DONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn usr2_done() -> u64 {
    USR2_DONE.load(Ordering::Relaxed)
}

extern "C" fn request_dump(_: i32) {
    let w = DUMP_W.load(Ordering::Relaxed);
    if w >= 0 {
        let b = 1u8;
        let e = unsafe { *libc::__error() };
        unsafe { libc::write(w, &b as *const u8 as *const libc::c_void, 1) };
        unsafe { *libc::__error() = e };
    }
}

fn start_dump_thread() {
    let mut fds = [-1i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return;
    }
    unsafe {
        for fd in fds {
            libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
        }
        // A full pipe drops a request rather than block the handler.
        let fl = libc::fcntl(fds[1], libc::F_GETFL);
        libc::fcntl(fds[1], libc::F_SETFL, fl | libc::O_NONBLOCK);
    }
    let r = fds[0];
    let started = crate::forksync::spawn_host("maclator-dump", 1 << 20, Box::new(move || {
        // No signal is handled on this thread (its handlers would only ask it to dump).
        unsafe {
            let mut all: libc::sigset_t = std::mem::zeroed();
            libc::sigfillset(&mut all);
            for s in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE, libc::SIGTRAP] {
                libc::sigdelset(&mut all, s);
            }
            libc::pthread_sigmask(libc::SIG_BLOCK, &all, std::ptr::null_mut());
        }
        // The pipe also carries SIGUSR2's text lines (each
        // written whole: shorter than PIPE_BUF), which this thread copies to fd 2; a 0x01 byte
        // is a SIGUSR1 dump request. A write blocked on a full stderr blocks only this thread,
        // which holds no translator lock.
        let mut buf = [0u8; 1024];
        loop {
            let n = unsafe { libc::read(r, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
            if n > 0 {
                let got = &buf[..n as usize];
                let text: Vec<u8> = got.iter().copied().filter(|&c| c != 1).collect();
                let mut b = &text[..];
                while !b.is_empty() {
                    let k = unsafe { libc::write(2, b.as_ptr() as *const libc::c_void, b.len()) };
                    if k > 0 {
                        b = &b[k as usize..];
                    } else if k < 0 && unsafe { *libc::__error() } == libc::EINTR {
                        continue;
                    } else {
                        break;
                    }
                }
                if got.contains(&1) {
                    dump_threads(0);
                }
            } else if n < 0 && unsafe { *libc::__error() } == libc::EINTR {
                continue;
            } else {
                return; // the pipe was closed (a fork child replaced it)
            }
        }
    }));
    if started.is_ok() {
        DUMP_R.store(fds[0], Ordering::SeqCst);
        DUMP_W.store(fds[1], Ordering::SeqCst);
    } else {
        unsafe {
            libc::close(fds[0]);
            libc::close(fds[1]);
        }
    }
}

/// Append `s` to `buf` at `*at` (truncating at the end). Signal-safe: no allocation.
fn put(buf: &mut [u8], at: &mut usize, s: &[u8]) {
    for &c in s {
        if *at < buf.len() {
            buf[*at] = c;
            *at += 1;
        }
    }
}

/// Append `v` in hex (no 0x). Signal-safe.
fn put_hex(buf: &mut [u8], at: &mut usize, v: u64) {
    let mut d = [0u8; 16];
    let mut n = 0;
    let mut x = v;
    loop {
        d[n] = b"0123456789abcdef"[(x & 15) as usize];
        n += 1;
        x >>= 4;
        if x == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        put(buf, at, &d[n..n + 1]);
    }
}

/// Append `v` in decimal. Signal-safe.
fn put_dec(buf: &mut [u8], at: &mut usize, v: u64) {
    let mut d = [0u8; 20];
    let mut n = 0;
    let mut x = v;
    loop {
        d[n] = b'0' + (x % 10) as u8;
        n += 1;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        put(buf, at, &d[n..n + 1]);
    }
}

thread_local! {
    static CURRENT_CPU: Cell<usize> = const { Cell::new(0) };
}

/// SIGUSR2: print this thread's guest registers without taking any locks
/// (usable in forked children where other threads' locks may be stuck).
/// The handler never writes to fd 2, which
/// can block (a full pipe) while this thread holds a translator lock that a fork waits for. It
/// hands each line to the dump thread through the non-blocking request pipe (a full pipe drops the
/// line), and the dump thread writes it. With no dump thread the line is dropped. errno is kept.
extern "C" fn dump_current(_: i32) {
    let w = DUMP_W.load(Ordering::Relaxed);
    if w < 0 {
        return;
    }
    let e = unsafe { *libc::__error() };
    let mut sent = false;
    format_current(|line| {
        unsafe { libc::write(w, line.as_ptr() as *const libc::c_void, line.len()) };
        sent = true;
    });
    if sent {
        USR2_DONE.fetch_add(1, Ordering::Relaxed);
    }
    unsafe { *libc::__error() = e };
}

/// fault_dump's use: the same lines, written straight to fd 2 (fault_dump has pointed it at a file
/// and is about to end the process).
fn dump_current_direct() {
    format_current(|line| {
        unsafe { libc::write(2, line.as_ptr() as *const libc::c_void, line.len()) };
    });
}

/// Format this thread's registers and the JIT counters as two lines (each under 512 bytes, so a
/// pipe write of one is atomic) into a stack buffer, handing each to `emit`. Signal-safe.
fn format_current(mut emit: impl FnMut(&[u8])) {
    let p = CURRENT_CPU.with(|c| c.get());
    if p == 0 {
        return;
    }
    let cpu = unsafe { &*(p as *const Cpu) };
    // Formatted by hand into a stack buffer; snprintf is not async-signal-safe.
    let mut buf = [0u8; 512];
    let mut at = 0;
    put(&mut buf, &mut at, b"[usr2] pid ");
    put_dec(&mut buf, &mut at, unsafe { libc::getpid() } as u64);
    let regs: [(&[u8], u64); 13] = [
        (b" pc=", cpu.pc), (b" lr=", cpu.x[30]), (b" sp=", cpu.x[31]), (b" x0=", cpu.x[0]), (b" x1=", cpu.x[1]),
        (b" x2=", cpu.x[2]), (b" x3=", cpu.x[3]), (b" x8=", cpu.x[8]), (b" x16=", cpu.x[16]), (b" x19=", cpu.x[19]),
        (b" x20=", cpu.x[20]), (b" x21=", cpu.x[21]), (b" x22=", cpu.x[22]),
    ];
    for (k, v) in regs {
        put(&mut buf, &mut at, k);
        put_hex(&mut buf, &mut at, v);
    }
    put(&mut buf, &mut at, b"\n");
    emit(&buf[..at]);
    let (blocks, ms, interp, entries, fb) = crate::jit::stats_snapshot();
    let mut at = 0;
    put(&mut buf, &mut at, b"[stats] pid ");
    put_dec(&mut buf, &mut at, unsafe { libc::getpid() } as u64);
    for (k, v) in [(&b" translated="[..], blocks), (b" blocks in ", ms), (b" ms, cold-interpreted=", interp), (b" blocks, block-entries=", entries), (b", fallback-insns=", fb)] {
        put(&mut buf, &mut at, k);
        put_dec(&mut buf, &mut at, v);
    }
    put(&mut buf, &mut at, b"\n");
    emit(&buf[..at]);
}

/// Tests: make `p` this thread's current guest CPU, as run_thread does.
#[cfg(test)]
pub(crate) fn test_set_current_cpu(p: usize) {
    CURRENT_CPU.with(|c| c.set(p));
}

pub fn run_thread(cpu: &mut Cpu) {
    crate::signals::enter_thread();
    CURRENT_CPU.with(|c| c.set(cpu as *mut Cpu as usize));
    EXIT_REQUESTED.with(|f| f.set(false));
    let key = cpu as *mut Cpu as usize;
    CPUS.lock().unwrap().push(key);
    crate::jit::run(cpu);
    CPUS.lock().unwrap().retain(|&k| k != key);
    CURRENT_CPU.with(|c| c.set(0));
    crate::signals::leave_thread();
}

/// The SIGUSR1 dump, run on the dump thread (and by fault_dump, on a thread that is about to exit
/// the process). It takes locks and allocates, so it must never run inside a signal handler on a
/// thread that may hold a lock or be inside malloc.
fn dump_threads(_: i32) {
    DUMPS.fetch_add(1, Ordering::Relaxed);
    crate::workq::dump_log();
    // Best effort, racy by design: this is a debugging aid.
    let list = CPUS.try_lock().map(|g| g.clone()).unwrap_or_default();
    for (i, &p) in list.iter().enumerate() {
        let cpu = unsafe { &*(p as *const Cpu) };
        let me = CURRENT_CPU.with(|c| c.get()) == p;
        eprintln!("=== guest thread {i}{}: pc {:#x} {}  lr {}", if me { " (THIS THREAD)" } else { "" }, cpu.pc, crate::symbols::describe(cpu.pc), crate::symbols::describe(cpu.x[30]));
        eprintln!("{}", cpu.dump());
        for i in 0..14u64 {
            let a = cpu.pc.wrapping_sub(10 * 4) + i * 4;
            if let Some(w) = crate::guestmem::read_u64(a & !7) {
                let insn = if a & 4 != 0 { (w >> 32) as u32 } else { w as u32 };
                eprintln!("  {} {:#x}: {:08x}  {}", if a == cpu.pc { "=>" } else { "  " }, a, insn, disasm(insn));
            }
        }
        crate::symbols::backtrace(cpu);
        for r in [0usize, 1, 19, 20, 21, 22] {
            if let Some(v) = crate::guestmem::read_u64(cpu.x[r]) {
                let v2 = crate::guestmem::read_u64(cpu.x[r] + 8).unwrap_or(0);
                eprintln!("  [x{r}] = {:#x} {:#x}", v, v2);
            }
        }
    }
}

pub fn install_debug_handler() {
    if DUMP_W.load(Ordering::SeqCst) < 0 {
        start_dump_thread();
    }
    unsafe { libc::signal(libc::SIGUSR1, request_dump as usize) };
    unsafe { libc::signal(libc::SIGUSR2, dump_current as usize) };
    if std::env::var_os("MACLATOR_DUMP_ON_FAULT").is_some() {
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = fault_dump as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut());
            libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut());
        }
    }
}

// libc does not expose Darwin's ucontext_t. This prefix follows the public
// sys/_types/_ucontext.h layout; the x86_64 mcontext begins with a 16-byte
// exception state, followed by x86_thread_state64 (RIP is register 16).
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[repr(C)]
struct HostUContext {
    onstack: i32,
    sigmask: u32,
    stack: libc::stack_t,
    link: *mut libc::c_void,
    mcsize: usize,
    mcontext: *mut libc::c_void,
}

extern "C" fn fault_dump(sig: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // Helper processes often have stderr closed or /dev/null: write the dump to a file.
    unsafe {
        let path = format!("/tmp/maclator-fault-{}.log\0", libc::getpid());
        let fd = libc::open(path.as_ptr() as *const libc::c_char, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC, 0o644);
        if fd >= 0 {
            libc::dup2(fd, 2);
        }
    }
    crate::forksync::output_direct_on_this_thread();
    eprintln!("maclator: host signal {sig} in emulator; guest state:");
    // Always emit the current thread first: the registry mutex may be held by
    // another thread, especially in a fork child or during thread creation.
    dump_current_direct();
    unsafe {
        let address = if info.is_null() { std::ptr::null_mut() } else { (*info).si_addr };
        let code = if info.is_null() { 0 } else { (*info).si_code };
        let mut rip = 0u64;
        let mut rsp = 0u64;
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        if !context.is_null() {
            let uc = &*(context as *const HostUContext);
            if !uc.mcontext.is_null() && uc.mcsize >= 16 + 17 * 8 {
                let regs = (uc.mcontext as *const u8).add(16) as *const u64;
                rip = *regs.add(16);
                rsp = *regs.add(7);
            }
        }
        let mut buf = [0u8; 192];
        let n = libc::snprintf(
            buf.as_mut_ptr() as *mut libc::c_char,
            buf.len(),
            b"[host fault] pid=%d code=%d address=%p rip=%llx rsp=%llx\n\0".as_ptr() as *const libc::c_char,
            libc::getpid(), code, address, rip, rsp,
        );
        libc::write(2, buf.as_ptr() as *const libc::c_void, (n.max(0) as usize).min(buf.len() - 1));
    }
    dump_threads(sig);
    unsafe { libc::_exit(128 + sig) };
}

/// Handle an exit from interpreter/JIT. Returns false when the thread should stop.
pub fn handle_exit(cpu: &mut Cpu, e: Exit) -> bool {
    match e {
        Exit::Svc(_) => {
            crate::syscalls::handle_svc(cpu);
            !EXIT_REQUESTED.with(|f| f.get())
        }
        Exit::IcInvalidate(addr) => {
            crate::jit::invalidate_range(addr & !63, 64);
            true
        }
        // A native run of these gets EXC_BREAKPOINT (SIGTRAP) and EXC_BAD_INSTRUCTION (SIGILL). An instruction the
        // translator lacks is one the hardware would run: the program ends as for an undefined one.
        Exit::Brk(imm) => {
            crash(cpu, &format!("guest executed BRK #{imm:#x} (trap)"), libc::SIGTRAP);
        }
        Exit::Udf(insn) => {
            crash(cpu, &format!("guest executed undefined instruction {insn:#010x}"), libc::SIGILL);
        }
        Exit::Unimplemented(insn) => {
            crash(cpu, &format!("unimplemented instruction {insn:#010x}  {}", disasm(insn)), libc::SIGILL);
        }
    }
}

/// End the process for a guest fault, or a failure of the translator's own, after saying what happened: on stderr as
/// before, and in a crash report file (`write_crash_report`), since a launchd daemon's stderr is /dev/null. The process
/// then ends by `sig` at its default action, as a native run ends for a fault it does not handle, so launchd, a
/// waiting parent and the crash reporter see that signal (it used to exit with status 134).
pub fn crash(cpu: &Cpu, why: &str, sig: i32) -> ! {
    use std::fmt::Write as _;
    // Until the one bounded write at the end, fd 2 is /dev/null: a note written on the way (a refused guest read) must
    // not block on a full stderr before the report is saved and the signal sent.
    let stderr = quiet_stderr();
    let mut text = String::new();
    let _ = writeln!(text, "\nmaclator: {why}");
    let _ = writeln!(text, "  at pc {:#x} {}", cpu.pc, crate::symbols::describe(cpu.pc));
    let _ = writeln!(text, "{}", cpu.dump());
    let _ = writeln!(text, "  lr {}", crate::symbols::describe(cpu.x[30]));
    let n: u64 = std::env::var("MACLATOR_DISAS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    let start = cpu.pc.wrapping_sub(n * 4);
    for i in 0..(n + 4) {
        let a = start.wrapping_add(i * 4);
        if let Some(w) = crate::guestmem::read_u64(a & !7) {
            let insn = if a & 4 != 0 { (w >> 32) as u32 } else { w as u32 };
            let _ = writeln!(text, "  {} {:#x}: {:08x}  {}", if a == cpu.pc { "=>" } else { "  " }, a, insn, disasm(insn));
        }
    }
    for line in crate::symbols::backtrace_lines(cpu) {
        let _ = writeln!(text, "{line}");
    }
    // Strings materialised by ADRP+ADD just before the crash (crash messages).
    for i in 1..24u64 {
        let a = cpu.pc.wrapping_sub(i * 4);
        let (Some(w1), Some(w2)) = (read_insn(a), read_insn(a.wrapping_add(4))) else { continue };
        if w1 & 0x9F00_0000 == 0x9000_0000 && w2 & 0xFFC0_0000 == 0x9100_0000 && (w1 & 31) == ((w2 >> 5) & 31) {
            let immlo = ((w1 >> 29) & 3) as u64;
            let immhi = ((w1 >> 5) & 0x7ffff) as u64;
            let imm = (((immhi << 2) | immlo) << 43) as i64 >> 31;
            let page = (a & !0xfff).wrapping_add(imm as u64);
            let target = page.wrapping_add(((w2 >> 10) & 0xfff) as u64);
            let mut b = [0u8; 160];
            if crate::guestmem::read(target, &mut b) {
                let len = b.iter().position(|&c| c == 0).unwrap_or(0);
                if len >= 6 && b[..len].iter().all(|&c| (0x20..0x7f).contains(&c)) {
                    let _ = writeln!(text, "  message? \"{}\"", String::from_utf8_lossy(&b[..len]));
                }
            }
        }
    }
    // Registers that look like C strings often carry the crash reason.
    for i in 0..31 {
        let mut b = [0u8; 96];
        if crate::guestmem::read(cpu.x[i], &mut b) {
            let len = b.iter().position(|&c| c == 0).unwrap_or(0);
            if len >= 6 && b[..len].iter().all(|&c| (0x20..0x7f).contains(&c) || c == b'\n') {
                let _ = writeln!(text, "  x{i} -> \"{}\"", String::from_utf8_lossy(&b[..len]));
            }
        }
    }
    // The file first, then stderr for at most CRASH_STDERR_MS: a stderr that blocks (a full pipe no one reads) must
    // cost neither the report nor the signal. A report file that may not grow (RLIMIT_FSIZE) fails rather than ending
    // the process by SIGXFSZ.
    unsafe { libc::signal(libc::SIGXFSZ, libc::SIG_IGN) };
    write_crash_report(&crate::aot::cache_dir().join("crash"), &report_header(why, sig), &text);
    write_stderr_bounded(stderr, text.into_bytes());
    end_by_signal(sig)
}

/// What stderr was, once a crash has put /dev/null on fd 2: -1 before, -2 while a thread is putting it there.
static KEPT_STDERR: AtomicI32 = AtomicI32::new(-1);

/// Put /dev/null on fd 2, and return a descriptor of what stderr was (fd 2 itself when it cannot be kept). Once per
/// process: a second thread crashing at once gets the same descriptor, not a duplicate of /dev/null.
fn quiet_stderr() -> i32 {
    match KEPT_STDERR.compare_exchange(-1, -2, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => {}
        Err(_) => {
            // Another thread is crashing too: wait (briefly) for what it kept.
            for _ in 0..200 {
                let k = KEPT_STDERR.load(Ordering::SeqCst);
                if k != -2 {
                    return k;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            return 2;
        }
    }
    let kept = quiet_stderr_once();
    KEPT_STDERR.store(kept, Ordering::SeqCst);
    kept
}

fn quiet_stderr_once() -> i32 {
    unsafe {
        let kept = libc::fcntl(2, libc::F_DUPFD_CLOEXEC, 3);
        if kept < 0 {
            return 2;
        }
        let null = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC);
        if null < 0 || libc::dup2(null, 2) < 0 {
            if null >= 0 {
                libc::close(null);
            }
            libc::close(kept);
            return 2;
        }
        libc::close(null);
        kept
    }
}

/// How long the crash path waits for stderr to take its report.
const CRASH_STDERR_MS: u64 = 2000;

/// Write `bytes` to `fd` on a thread of its own and wait for it at most CRASH_STDERR_MS. A write that blocks (a full
/// pipe no one reads, or one another writer filled after it looked free) then blocks only that thread, which ends with
/// the process. Not by making the descriptor non-blocking: that flag belongs to the open file, which other processes
/// share. SIGPIPE is ignored first, so a reader that has gone ends the write, not the process by the wrong signal. The
/// wait is a flag polled with sleeps, not a channel: in a fork child a channel's wait would park on the parent's
/// semaphore, which the child does not have.
fn write_stderr_bounded(fd: i32, bytes: Vec<u8>) {
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let told = done.clone();
    let writer = Box::new(move || {
        let mut rest = &bytes[..];
        while !rest.is_empty() {
            let k = unsafe { libc::write(fd, rest.as_ptr() as *const libc::c_void, rest.len()) };
            if k > 0 {
                rest = &rest[k as usize..];
            } else if !(k < 0 && unsafe { *libc::__error() } == libc::EINTR) {
                break;
            }
        }
        told.store(true, Ordering::SeqCst);
    });
    if crate::forksync::spawn_host("maclator-crash-stderr", 1 << 16, writer).is_ok() {
        let end = std::time::Instant::now() + std::time::Duration::from_millis(CRASH_STDERR_MS);
        while !done.load(Ordering::SeqCst) && std::time::Instant::now() < end {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

/// How many crash reports the store keeps, how many of them one program may hold, and the most bytes one may take; the
/// oldest is written over first. A program that crashes every few seconds keeps CRASH_PER_PROGRAM, not all of them.
const CRASH_KEEP: usize = 16;
const CRASH_PER_PROGRAM: usize = 4;
const CRASH_MAX_BYTES: usize = 64 << 10;

fn signal_name(sig: i32) -> &'static str {
    match sig {
        libc::SIGTRAP => "SIGTRAP",
        libc::SIGILL => "SIGILL",
        libc::SIGABRT => "SIGABRT",
        _ => "signal",
    }
}

/// The lines a report file starts with: nothing the translator did not already know or print, and no guest data
/// beyond the text it prints on stderr.
fn report_header(why: &str, sig: i32) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let engine = crate::aot::own_uuid().map(|u| u.iter().map(|b| format!("{b:02x}")).collect::<String>()).unwrap_or_else(|| "unknown".into());
    format!(
        "maclator crash report\nprogram: {}\npid: {}\ntime: {now}\nsignal: {} ({sig})\nreason: {why}\nengine: {engine}\n",
        crate::symbols::main_name().unwrap_or_else(|| "unknown".into()),
        unsafe { libc::getpid() },
        signal_name(sig)
    )
}

/// The "program: " line of a report's header (empty when it has none).
fn report_program(header: &str) -> &str {
    header.lines().find_map(|l| l.strip_prefix("program: ")).unwrap_or("")
}

/// The slot to write `program`'s report in: its own oldest once it holds CRASH_PER_PROGRAM, else the first one unused,
/// else the one written longest ago.
fn crash_slot(dir: &std::path::Path, program: &str) -> usize {
    let mut oldest: Option<(std::time::SystemTime, usize)> = None;
    let mut mine: Vec<(std::time::SystemTime, usize)> = Vec::new();
    let mut unused = None;
    for i in 0..CRASH_KEEP {
        let path = dir.join(format!("crash-{i:02}.log"));
        match std::fs::symlink_metadata(&path) {
            Err(_) => {
                unused.get_or_insert(i);
            }
            Ok(md) => {
                let t = md.modified().unwrap_or(std::time::UNIX_EPOCH);
                if oldest.is_none_or(|(o, _)| t < o) {
                    oldest = Some((t, i));
                }
                // The header is in the first few hundred bytes.
                let mut head = [0u8; 512];
                let n = std::fs::File::open(&path).and_then(|mut f| std::io::Read::read(&mut f, &mut head)).unwrap_or(0);
                if report_program(&String::from_utf8_lossy(&head[..n])) == program {
                    mine.push((t, i));
                }
            }
        }
    }
    if mine.len() >= CRASH_PER_PROGRAM {
        return mine.iter().min().map_or(0, |&(_, i)| i);
    }
    unused.or(oldest.map(|(_, i)| i)).unwrap_or(0)
}

/// Write `header` and `text` to a report file in `dir`, cut to CRASH_MAX_BYTES, in one of CRASH_KEEP slots (at most
/// CRASH_PER_PROGRAM of them the header's program's), through a temporary file of this process renamed into place;
/// temporary files a minute old (a writer that died) are removed.
/// Nothing is written in a sandboxed process (the translator's stores are left alone there) or in a directory that is
/// not ours alone. The report's path, if one was written.
pub(crate) fn write_crash_report(dir: &std::path::Path, header: &str, text: &str) -> Option<std::path::PathBuf> {
    if crate::sandboxcheck::sandboxed() {
        return None;
    }
    let uid = unsafe { libc::geteuid() };
    if !dir.parent().is_some_and(|store| crate::aot::owned_dir(store, uid)) || !crate::aot::owned_dir(dir, uid) {
        return None;
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        let now = std::time::SystemTime::now();
        for ent in rd.flatten() {
            let stale = ent.metadata().and_then(|m| m.modified()).ok().and_then(|t| now.duration_since(t).ok()).is_some_and(|a| a.as_secs() >= 60);
            if ent.file_name().to_str().is_some_and(|n| n.starts_with("crash-") && n.ends_with(".tmp")) && stale {
                let _ = std::fs::remove_file(ent.path());
            }
        }
    }
    let mut report = format!("{header}\n{text}");
    if report.len() > CRASH_MAX_BYTES {
        let note = format!("\n[maclator: report cut at {CRASH_MAX_BYTES} bytes]\n");
        let mut cut = CRASH_MAX_BYTES - note.len();
        while !report.is_char_boundary(cut) {
            cut -= 1;
        }
        report.truncate(cut);
        report.push_str(&note);
    }
    let slot = crash_slot(dir, report_program(header));
    let path = dir.join(format!("crash-{slot:02}.log"));
    // Two threads of one process can crash at once: each writes a temporary file of its own.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!("crash-{slot:02}.log.{}-{seq}.tmp", unsafe { libc::getpid() }));
    let written = {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&tmp)
            .and_then(|mut f| f.write_all(report.as_bytes()))
            .and_then(|()| std::fs::rename(&tmp, &path))
    };
    match written {
        Ok(()) => Some(path),
        Err(_) => {
            let _ = std::fs::remove_file(&tmp);
            None
        }
    }
}

/// End this process by `sig` at its default action, whatever the guest set for it (a native process that does not
/// handle a fault dies of it). `raise` targets this thread, with the signal unblocked here. Not by an instruction that
/// faults (int3, ud2), which would give the crash reporter EXC_BREAKPOINT or EXC_BAD_INSTRUCTION as natively: the
/// guest's Mach messages reach the kernel as they are, task_set_exception_ports among them, so a guest's own exception
/// handler would be sent an x86 thread state it cannot read, and could resume it into the same fault. A signal that
/// kills the process reaches the crash reporter as EXC_CRASH, which no handler can resume.
fn end_by_signal(sig: i32) -> ! {
    unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        act.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut act.sa_mask);
        libc::sigaction(sig, &act, std::ptr::null_mut());
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, sig);
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
        libc::raise(sig);
        // Not reached unless something holds the signal off (a tracer): say it with the status a shell would.
        libc::_exit(128 + sig)
    }
}

fn read_insn(a: u64) -> Option<u32> {
    let w = crate::guestmem::read_u64(a & !7)?;
    Some(if a & 4 != 0 { (w >> 32) as u32 } else { w as u32 })
}

/// Plain interpreter loop (used by `--interp` and as the reference path).
pub fn interp_loop(cpu: &mut Cpu) {
    loop {
        if let Err(errno) = crate::signals::poll(cpu) {
            crash(cpu, &format!("guest signal delivery failed: errno {errno}"), libc::SIGABRT);
        }
        match interp::step(cpu) {
            Ok(()) => {}
            Err(e) => {
                if !handle_exit(cpu, e) {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "engine_crash_tests.rs"]
mod crash_tests;

#[cfg(test)]
mod crash_report_tests {
    use super::*;

    /// A store of a test's own (0700) and the crash directory in it; removed when the test ends.
    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("mcl-crash-report-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(d.join("store")).unwrap();
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(d.join("store"), std::fs::Permissions::from_mode(0o700)).unwrap();
            Dir(d)
        }
        fn crash(&self) -> std::path::PathBuf {
            self.0.join("store").join("crash")
        }
        fn names(&self) -> Vec<String> {
            let mut v: Vec<String> = std::fs::read_dir(self.crash()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
            v.sort();
            v
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn reports_are_kept_to_their_count_and_the_oldest_goes_first() {
        let d = Dir::new("count");
        for i in 0..40 {
            // Ten programs, four reports each: the store's count binds, not a program's.
            let header = format!("maclator crash report\nprogram: p{}\n", i % 10);
            assert!(write_crash_report(&d.crash(), &header, &format!("report {i}\n")).is_some(), "report {i} written");
        }
        let names = d.names();
        assert_eq!(names.len(), CRASH_KEEP, "files after 40 reports: {names:?}");
        assert!(names.iter().all(|n| n.starts_with("crash-") && n.ends_with(".log")), "and no temporary file: {names:?}");
        let mut kept: Vec<usize> = names
            .iter()
            .map(|n| std::fs::read_to_string(d.crash().join(n)).unwrap())
            .map(|t| t.trim_end().rsplit(' ').next().unwrap().parse().unwrap())
            .collect();
        kept.sort();
        assert_eq!(kept, (40 - CRASH_KEEP..40).collect::<Vec<_>>(), "the newest {CRASH_KEEP} are kept");
    }

    #[test]
    fn a_program_that_crashes_again_and_again_keeps_its_share_and_no_more() {
        let d = Dir::new("share");
        let w = |program: &str, i: usize| write_crash_report(&d.crash(), &format!("maclator crash report\nprogram: {program}\n"), &format!("report {program} {i}\n")).unwrap();
        for i in 0..5 {
            w("quiet", i);
        }
        for i in 0..200 {
            w("loops", i);
        }
        let texts: Vec<String> = d.names().iter().map(|n| std::fs::read_to_string(d.crash().join(n)).unwrap()).collect();
        let of = |p: &str| texts.iter().filter(|t| report_program(t) == p).count();
        assert_eq!(of("loops"), CRASH_PER_PROGRAM, "the program that loops keeps its share: {texts:?}");
        assert_eq!(of("quiet"), CRASH_PER_PROGRAM, "and the other keeps its newest reports");
        assert!(texts.iter().any(|t| t.ends_with("report loops 199\n")) && texts.iter().any(|t| t.ends_with("report quiet 4\n")), "the newest of each");
        assert_eq!(texts.len(), 2 * CRASH_PER_PROGRAM);
    }

    #[test]
    fn a_report_is_cut_to_its_size() {
        let d = Dir::new("size");
        let big = "x".repeat(3 * CRASH_MAX_BYTES);
        let p = write_crash_report(&d.crash(), "header\n", &big).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.len() <= CRASH_MAX_BYTES, "{} bytes", text.len());
        assert!(text.starts_with("header\n") && text.ends_with(&format!("[maclator: report cut at {CRASH_MAX_BYTES} bytes]\n")), "cut, and says so");
        // A cut inside a character keeps the file valid text: one byte first, so the cut falls inside a two-byte "é".
        let wide = format!("x{}", "é".repeat(CRASH_MAX_BYTES));
        assert!(!format!("h\n\n{wide}").is_char_boundary(CRASH_MAX_BYTES - format!("\n[maclator: report cut at {CRASH_MAX_BYTES} bytes]\n").len()), "control: the cut falls inside a character");
        let p = write_crash_report(&d.crash(), "h\n", &wide).unwrap();
        assert!(std::fs::read_to_string(&p).is_ok(), "valid UTF-8 after the cut");
        // A small one is written whole.
        let p = write_crash_report(&d.crash(), "h\n", "small\n").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "h\n\nsmall\n");
    }

    #[test]
    fn a_dead_writers_temporary_file_goes_and_a_live_ones_stays() {
        let d = Dir::new("tmp");
        assert!(write_crash_report(&d.crash(), "h\n", "first\n").is_some());
        let (old, new) = (d.crash().join("crash-05.log.123.tmp"), d.crash().join("crash-06.log.124.tmp"));
        std::fs::write(&old, b"half").unwrap();
        std::fs::write(&new, b"half").unwrap();
        let then = std::time::SystemTime::now() - std::time::Duration::from_secs(120);
        std::fs::File::options().write(true).open(&old).unwrap().set_modified(then).unwrap();
        assert!(write_crash_report(&d.crash(), "h\n", "second\n").is_some());
        assert!(!old.exists() && new.exists(), "the two-minute-old one went, the new one stayed: {:?}", d.names());
    }

    #[test]
    fn no_report_where_the_store_is_not_ours_alone() {
        let d = Dir::new("mode");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(d.0.join("store"), std::fs::Permissions::from_mode(0o777)).unwrap();
        assert_eq!(write_crash_report(&d.crash(), "h\n", "x\n"), None, "a store others can write");
        std::fs::set_permissions(d.0.join("store"), std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(write_crash_report(&d.crash(), "h\n", "x\n").is_some(), "control: ours again");
        let mode = std::fs::metadata(d.crash()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "the crash directory is made ours alone");
    }

    #[test]
    fn the_header_names_the_signal_and_no_guest_data() {
        let h = report_header("guest executed BRK #0x1 (trap)", libc::SIGTRAP);
        let lines: Vec<&str> = h.lines().collect();
        assert_eq!(lines[0], "maclator crash report");
        let keys: Vec<&str> = lines[1..].iter().map(|l| l.split_once(": ").unwrap().0).collect();
        assert_eq!(keys, ["program", "pid", "time", "signal", "reason", "engine"], "{h}");
        assert!(h.contains("\nsignal: SIGTRAP (5)\n"), "{h}");
    }
}
