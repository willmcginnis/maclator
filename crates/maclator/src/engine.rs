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
        Exit::Brk(imm) => {
            crash(cpu, &format!("guest executed BRK #{imm:#x} (trap)"));
            false
        }
        Exit::Udf(insn) => {
            crash(cpu, &format!("guest executed undefined instruction {insn:#010x}"));
            false
        }
        Exit::Unimplemented(insn) => {
            crash(cpu, &format!("unimplemented instruction {insn:#010x}  {}", disasm(insn)));
            false
        }
    }
}

pub fn crash(cpu: &Cpu, why: &str) -> ! {
    eprintln!("\nmaclator: {why}");
    eprintln!("  at pc {:#x} {}", cpu.pc, crate::symbols::describe(cpu.pc));
    eprintln!("{}", cpu.dump());
    eprintln!("  lr {}", crate::symbols::describe(cpu.x[30]));
    let n: u64 = std::env::var("MACLATOR_DISAS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    let start = cpu.pc.wrapping_sub(n * 4);
    for i in 0..(n + 4) {
        let a = start + i * 4;
        if let Some(w) = crate::guestmem::read_u64(a & !7) {
            let insn = if a & 4 != 0 { (w >> 32) as u32 } else { w as u32 };
            eprintln!("  {} {:#x}: {:08x}  {}", if a == cpu.pc { "=>" } else { "  " }, a, insn, disasm(insn));
        }
    }
    crate::symbols::backtrace(cpu);
    // Strings materialised by ADRP+ADD just before the crash (crash messages).
    for i in 1..24u64 {
        let a = cpu.pc.wrapping_sub(i * 4);
        let (Some(w1), Some(w2)) = (read_insn(a), read_insn(a + 4)) else { continue };
        if w1 & 0x9F00_0000 == 0x9000_0000 && w2 & 0xFFC0_0000 == 0x9100_0000 && (w1 & 31) == ((w2 >> 5) & 31) {
            let immlo = ((w1 >> 29) & 3) as u64;
            let immhi = ((w1 >> 5) & 0x7ffff) as u64;
            let imm = (((immhi << 2) | immlo) << 43) as i64 >> 31;
            let page = (a & !0xfff).wrapping_add(imm as u64);
            let target = page + ((w2 >> 10) & 0xfff) as u64;
            let mut b = [0u8; 160];
            if crate::guestmem::read(target, &mut b) {
                let len = b.iter().position(|&c| c == 0).unwrap_or(0);
                if len >= 6 && b[..len].iter().all(|&c| (0x20..0x7f).contains(&c)) {
                    eprintln!("  message? \"{}\"", String::from_utf8_lossy(&b[..len]));
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
                eprintln!("  x{i} -> \"{}\"", String::from_utf8_lossy(&b[..len]));
            }
        }
    }
    std::process::exit(134);
}

fn read_insn(a: u64) -> Option<u32> {
    let w = crate::guestmem::read_u64(a & !7)?;
    Some(if a & 4 != 0 { (w >> 32) as u32 } else { w as u32 })
}

/// Plain interpreter loop (used by `--interp` and as the reference path).
pub fn interp_loop(cpu: &mut Cpu) {
    loop {
        if let Err(errno) = crate::signals::poll(cpu) {
            crash(cpu, &format!("guest signal delivery failed: errno {errno}"));
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
