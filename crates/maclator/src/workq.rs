//! Workqueue emulation: libdispatch worker threads, kernel event delivery and
//! workloops.
//!
//! XNU owns three things libdispatch relies on:
//!  * the per-process *workq kqueue* (`KEVENT_FLAG_WORKQ`): knotes fire by
//!    spawning a workqueue thread that receives the events on its stack;
//!  * *workloops* (`kevent_id` + `KEVENT_FLAG_WORKLOOP`): dynamic kqueues, one
//!    per serial queue, with EVFILT_WORKLOOP knotes that request a servicer
//!    thread or implement dispatch_sync waits;
//!  * workqueue thread creation (`workq_kernreturn`).
//!
//! We back the workq kqueue with an ordinary host kqueue and every workloop's
//! regular knotes with one more, shared by all workloops, implement
//! EVFILT_WORKLOOP in user space, and run guest worker threads laid out
//! exactly as the kernel lays them out.

use crate::hostsys;
use maclator_core::cpu::Cpu;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use crate::forksync::{FCondvar, FMutex};

// ---- constants (workqueue_syscalls.h, event_private.h, priority_private.h) ----

const WQOPS_THREAD_RETURN: u64 = 0x004;
const WQOPS_QUEUE_NEWSPISUPP: u64 = 0x010;
const WQOPS_QUEUE_REQTHREADS: u64 = 0x020;
const WQOPS_QUEUE_REQTHREADS2: u64 = 0x030;
const WQOPS_THREAD_KEVENT_RETURN: u64 = 0x040;
const WQOPS_SET_EVENT_MANAGER_PRIORITY: u64 = 0x080;
const WQOPS_THREAD_WORKLOOP_RETURN: u64 = 0x100;
const WQOPS_SHOULD_NARROW: u64 = 0x200;
const WQOPS_SETUP_DISPATCH: u64 = 0x400;

const WQ_FLAG_THREAD_PRIO_QOS: u64 = 0x0000_4000;
const WQ_FLAG_THREAD_OVERCOMMIT: u64 = 0x0001_0000;
const WQ_FLAG_THREAD_REUSE: u64 = 0x0002_0000;
const WQ_FLAG_THREAD_NEWSPI: u64 = 0x0004_0000;
const WQ_FLAG_THREAD_KEVENT: u64 = 0x0008_0000;
const WQ_FLAG_THREAD_EVENT_MANAGER: u64 = 0x0010_0000;
const WQ_FLAG_THREAD_TSD_BASE_SET: u64 = 0x0020_0000;
const WQ_FLAG_THREAD_WORKLOOP: u64 = 0x0040_0000;
const WQ_FLAG_THREAD_COOPERATIVE: u64 = 0x0100_0000;

const PP_OVERCOMMIT: u64 = 0x8000_0000;
const PP_COOPERATIVE: u64 = 0x0800_0000;
const PP_EVENT_MANAGER: u64 = 0x0200_0000;

const KEVENT_FLAG_IMMEDIATE: u64 = 0x001;
const KEVENT_FLAG_ERROR_EVENTS: u64 = 0x002;
const KEVENT_FLAG_STACK_DATA: u64 = 0x008;
const KEVENT_FLAG_WORKQ: u64 = 0x020;
const KEVENT_FLAG_PARKING: u64 = 0x800;

const EVFILT_WORKLOOP: i16 = -17;
const EVFILT_USER_: i16 = -10;
const EVFILT_TIMER: i16 = -7;
const EVFILT_READ: i16 = -1;
const EVFILT_WRITE: i16 = -2;
const EVFILT_VNODE: i16 = -4;
const EVFILT_EXCEPT: i16 = -15;
const EVFILT_SIGNAL: i16 = -6;
const EVFILT_SOCK: i16 = -13;
const EVFILT_NW_CHANNEL: i16 = -16;
/// The flags the kernel keeps as a knote is attached: a later change of the knote cannot alter them.
const PROTOCOL_FLAGS: u16 = EV_DISPATCH | EV_ONESHOT | EV_UDATA_SPECIFIC | EV_VANISHED;
const EV_ADD: u16 = 0x1;
const EV_DELETE: u16 = 0x2;
const EV_ENABLE: u16 = 0x4;
const EV_DISABLE: u16 = 0x8;
const EV_ONESHOT: u16 = 0x10;
const EV_CLEAR: u16 = 0x20;
const EV_DISPATCH: u16 = 0x80;
const EV_UDATA_SPECIFIC: u16 = 0x100;
const EV_VANISHED: u16 = 0x200;
const EV_ERROR: u16 = 0x4000;
const EV_EOF: u16 = 0x8000;

const NOTE_WL_THREAD_REQUEST: u32 = 0x1;
const NOTE_WL_SYNC_WAIT: u32 = 0x4;
const NOTE_WL_SYNC_WAKE: u32 = 0x8;
const NOTE_WL_SYNC_IPC: u32 = 0x8000_0000;
const NOTE_WL_END_OWNERSHIP: u32 = 0x20;
const NOTE_WL_DISCOVER_OWNER: u32 = 0x80;
const NOTE_WL_IGNORE_ESTALE: u32 = 0x100;
const NOTE_WL_COMMANDS_MASK: u32 = 0x8000_000f;

const SYS_KEVENT_QOS: u64 = 374;
const SYS_KEVENT_ID: u64 = 375;
const SYS_KQUEUE: u64 = 362;

const KEV_SIZE: u64 = 72;
const WQ_KEVENT_LIST_LEN: u64 = 16;
const WQ_KEVENT_DATA_SIZE: u64 = 32 * 1024;
const STACK_SIZE: u64 = 512 * 1024;
const GUARD: u64 = 0x4000;
const PTHREAD_T_OFFSET: u64 = 12 * 1024;

// ---- raw kevent_qos_s helpers ----

#[derive(Clone, Copy, Default, Debug)]
struct Kev {
    ident: u64,
    filter: i16,
    flags: u16,
    qos: i32,
    udata: u64,
    fflags: u32,
    xflags: u32,
    data: i64,
    ext: [u64; 4],
}

impl Kev {
    unsafe fn read(p: u64) -> Kev {
        let b = p as *const u8;
        let r64 = |o: usize| std::ptr::read_unaligned(b.add(o) as *const u64);
        Kev {
            ident: r64(0),
            filter: std::ptr::read_unaligned(b.add(8) as *const i16),
            flags: std::ptr::read_unaligned(b.add(10) as *const u16),
            qos: std::ptr::read_unaligned(b.add(12) as *const i32),
            udata: r64(16),
            fflags: std::ptr::read_unaligned(b.add(24) as *const u32),
            xflags: std::ptr::read_unaligned(b.add(28) as *const u32),
            data: r64(32) as i64,
            ext: [r64(40), r64(48), r64(56), r64(64)],
        }
    }
    unsafe fn write(&self, p: u64) {
        let b = p as *mut u8;
        let w64 = |o: usize, v: u64| std::ptr::write_unaligned(b.add(o) as *mut u64, v);
        w64(0, self.ident);
        std::ptr::write_unaligned(b.add(8) as *mut i16, self.filter);
        std::ptr::write_unaligned(b.add(10) as *mut u16, self.flags);
        std::ptr::write_unaligned(b.add(12) as *mut i32, self.qos);
        w64(16, self.udata);
        std::ptr::write_unaligned(b.add(24) as *mut u32, self.fflags);
        std::ptr::write_unaligned(b.add(28) as *mut u32, self.xflags);
        w64(32, self.data as u64);
        for i in 0..4 {
            w64(40 + i * 8, self.ext[i]);
        }
    }
}

/// A new host kqueue, or the errno the host refused it with.
fn host_kqueue() -> Result<i32, i32> {
    kqueue_from(hostsys::unix(SYS_KQUEUE, &[0; 8]))
}

/// `host_kqueue` from the host's answer `r`. A refusal is never taken for a descriptor: with the carry set, rax holds
/// the errno, and a sandbox that lists the system calls a program may make refuses kqueue with EPERM, which is 1.
/// Taken as a descriptor, that 1 was duplicated and then closed: the program's standard output.
fn kqueue_from(r: hostsys::SysResult) -> Result<i32, i32> {
    kqueue_from_with(r, libc::setrlimit)
}

/// Said once per process: the soft descriptor limit raised for a kqueue's move could not be put back.
static SOFT_LIMIT_KEPT_RAISED: AtomicBool = AtomicBool::new(false);

/// `kqueue_from`, with `setrlimit` the call that raises the soft descriptor limit and puts it back.
fn kqueue_from_with(
    r: hostsys::SysResult,
    setrlimit: unsafe extern "C" fn(libc::c_int, *const libc::rlimit) -> libc::c_int,
) -> Result<i32, i32> {
    if r.carry {
        return Err(r.rax as i32);
    }
    let fd = r.rax as i32;
    // Move it out of the guest's usual descriptor range.
    unsafe {
        let mut high = libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 900);
        let mut rl: libc::rlimit = std::mem::zeroed();
        if high < 0 && libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) == 0 && rl.rlim_cur <= 900 && rl.rlim_max > 901 {
            // The soft descriptor limit is below that range (256 for most processes launchd starts): raised for the
            // move alone and put back, so the program keeps its limit and never reaches the kqueue's number (natively
            // the kernel's kqueues are no descriptors of the program's).
            let soft = rl.rlim_cur;
            let raised = rl.rlim_max.min(1024);
            rl.rlim_cur = raised;
            if setrlimit(libc::RLIMIT_NOFILE, &rl) == 0 {
                high = libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 900);
                rl.rlim_cur = soft;
                // Lowering the soft limit is always allowed, so this should not fail; if it does, the program keeps
                // the raised limit, and that is said.
                if setrlimit(libc::RLIMIT_NOFILE, &rl) != 0 {
                    let e = *libc::__error();
                    if !SOFT_LIMIT_KEPT_RAISED.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "maclator: could not put the soft descriptor limit back to {soft} after moving a kqueue \
                             (errno {e}); it stays at {raised}"
                        );
                    }
                }
            }
        }
        if high >= 0 {
            libc::close(fd);
            return Ok(high);
        }
        // Where even the hard limit does not reach that range: kept low, but never in a standard descriptor's place
        // (one the program closed would be found open), and still not inherited by an exec.
        if fd <= 2 {
            let moved = libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3);
            let e = *libc::__error();
            libc::close(fd);
            return if moved >= 0 { Ok(moved) } else { Err(e) };
        }
        libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    Ok(fd)
}

/// What a monitor does when its wait on its kqueue failed (EINTR excepted): say so once per process, then pause before
/// waiting again, 1 ms after the first failure in a row and twice as long after each, at most about a second. It does
/// not park: nothing would wake it, since the errors its own kqueue can give (a descriptor the guest closed or
/// replaced, a shortage) end on their own or never. So a wait that keeps failing costs one call a second, not a core.
fn pause_after_failed_wait(which: &str, warned: &std::sync::atomic::AtomicBool, errno: i32, fails: &mut u32) {
    if !warned.swap(true, Ordering::Relaxed) {
        eprintln!("maclator: the {which} monitor's wait on its kqueue failed (errno {errno}); waiting again after a pause");
    }
    std::thread::sleep(std::time::Duration::from_millis(1 << (*fails).min(10)));
    *fails = fails.saturating_add(1);
}

/// What the guest is answered when a host kqueue the emulation needs could not be made: the kernel's answer when it
/// cannot allocate one.
const NO_KQUEUE: i64 = libc::ENOMEM as i64;

/// A kqueue slot that holds no kqueue yet.
const NOT_MADE: i32 = -1;
/// A kqueue slot whose kqueue this process may not make (its sandbox refuses the call): it is not asked again.
const REFUSED: i32 = -2;

/// The host kqueue kept in `slot`, made now if it is not yet; or the errno to answer the guest. A refusal (EPERM, from
/// a sandbox) is kept, so the call is made once per process; another failure (a shortage) is tried again next time.
/// Two threads may make one at once: one is kept, the other closed.
fn kept_kq(slot: &AtomicI32) -> Result<i32, i64> {
    static REFUSAL_SAID: AtomicBool = AtomicBool::new(false);
    match slot.load(Ordering::SeqCst) {
        kq if kq >= 0 => return Ok(kq),
        REFUSED => return Err(NO_KQUEUE),
        _ => {}
    }
    match host_kqueue() {
        Ok(kq) => match slot.compare_exchange(NOT_MADE, kq, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Ok(kq),
            Err(other) => {
                unsafe { libc::close(kq) };
                if other >= 0 {
                    Ok(other)
                } else {
                    Err(NO_KQUEUE)
                }
            }
        },
        Err(e) => {
            if e == libc::EPERM && slot.compare_exchange(NOT_MADE, REFUSED, Ordering::SeqCst, Ordering::SeqCst).is_ok() && !REFUSAL_SAID.swap(true, Ordering::Relaxed) {
                eprintln!("maclator: this process may not make a kqueue (its sandbox refuses the call), which the workqueue emulation needs: its dispatch event registrations are answered ENOMEM");
            }
            Err(NO_KQUEUE)
        }
    }
}

/// Make the host kqueues the workqueue emulation needs, before the guest's first instruction. A program may apply a
/// sandbox that lists the system calls it may make, and on macOS the kernel's workqueue and workloops need only kevent
/// calls of a program: Apple's WindowManager allows those and not kqueue. Made first, the emulation's kqueues are
/// there when the program's sandbox comes, and it makes no call the program would not.
///
/// A process that starts in such a sandbox (exec'd or spawned by a process in one, or given one as it is spawned), and
/// a fork child of a process that is in one (a fork does not hand on kqueues), cannot make them: its first dispatch
/// registration is answered ENOMEM, as the kernel answers when it cannot allocate a kqueue, where natively it works.
/// Natively the kernel makes those kqueues itself; here only its own per-process workqueue kqueue would need no call,
/// and it hands events to host workqueue threads, not to the guest's. A fork child makes its kqueues when it first
/// needs them, as most children exec first and need none.
pub fn prepare() {
    let _ = kept_kq(&ANON_KQ);
    let _ = kept_kq(&WL_KQ);
}

/// Whether `fd` is one of the emulation's kqueues. Natively the workqueue's and the workloops' kqueues are not the
/// program's descriptors, so the program's close of one (a program closing every descriptor it was not given, as it
/// starts), or dup2 onto one, is answered EBADF, and the kqueue kept. (Above the program's descriptor limit, where
/// they are put, the kernel answers a dup2 onto that number EBADF too.)
pub(crate) fn is_own_kqueue(fd: i32) -> bool {
    fd >= 0 && (ANON_KQ.load(Ordering::SeqCst) == fd || WL_KQ.load(Ordering::SeqCst) == fd)
}

/// Start `monitor` on a thread of its own, once per process: `started` says it was, under `gate`.
fn start_monitor_once(started: &AtomicBool, gate: &FMutex<()>, name: &str, monitor: fn()) {
    if started.load(Ordering::SeqCst) {
        return;
    }
    let _g = gate.lock().unwrap();
    if !started.load(Ordering::SeqCst) {
        crate::forksync::spawn_host(name, 2 << 20, Box::new(monitor)).unwrap();
        started.store(true, Ordering::SeqCst);
    }
}

fn kevent_qos(kq: i32, changes: u64, nchanges: u64, out: u64, nout: u64, data: u64, avail: u64, flags: u64) -> hostsys::SysResult {
    hostsys::unix(SYS_KEVENT_QOS, &[kq as u64, changes, nchanges, out, nout, data, avail, flags])
}

// ---- workers ----

struct Job {
    flags: u64,
    kevent_list: u64,
    nevents: u64,
    stack_top: u64,
    wl: Option<u64>,
}

pub(crate) struct Worker {
    stack_low: u64,
    pthread: u64,
    tx: Option<Sender<Job>>,
}

impl Worker {
    fn new() -> Option<Worker> {
        let (_, _, _, pthsize) = crate::threads::wq_registration();
        let pthsize = pthsize.max(0x1000);
        let total = GUARD + STACK_SIZE + PTHREAD_T_OFFSET + ((pthsize + 0x3fff) & !0x3fff);
        let base = unsafe {
            let p = libc::mmap(std::ptr::null_mut(), total as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
            if p == libc::MAP_FAILED {
                return None;
            }
            libc::mprotect(p, GUARD as usize, libc::PROT_NONE);
            p as u64
        };
        let stack_low = base + GUARD;
        Some(Worker { stack_low, pthread: stack_low + STACK_SIZE + PTHREAD_T_OFFSET, tx: None })
    }
    fn kevent_list(&self) -> u64 {
        self.pthread - WQ_KEVENT_LIST_LEN * KEV_SIZE
    }
    fn data_buf(&self) -> u64 {
        self.kevent_list() - WQ_KEVENT_DATA_SIZE
    }
}

pub(crate) static IDLE: FMutex<Vec<Worker>> = FMutex::new(Vec::new());

thread_local! {
    static PARK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static BOUND_WL: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

fn take_worker() -> Option<Worker> {
    if let Some(w) = IDLE.lock().unwrap().pop() {
        return Some(w);
    }
    Worker::new()
}

fn start_job(mut w: Worker, job: Job) {
    match &w.tx {
        Some(tx) => {
            let _ = tx.send(Job { flags: job.flags | WQ_FLAG_THREAD_REUSE, ..job });
        }
        None => {
            let (tx, rx) = channel::<Job>();
            w.tx = Some(tx);
            spawn_worker_thread(w, job, rx);
        }
    }
}

fn spawn_worker_thread(w: Worker, job: Job, rx: Receiver<Job>) {
    let (start, tsd_off, mts_off, _) = crate::threads::wq_registration();
    let (stack_low, pthread) = (w.stack_low, w.pthread);
    let tx = w.tx.clone().unwrap();
    crate::threads::LIVE_THREADS.fetch_add(1, Ordering::SeqCst);
    // A plain pthread, not a std::thread (forksync::spawn_host).
    crate::forksync::spawn_host("", 8 << 20, Box::new(move || {
            // The process this worker was made in. If a guest
            // fork on this thread made it the forking thread of a child, it must not return to the
            // idle list: its channel's receiver parks this thread's std parker, made in the parent.
            let gen0 = crate::forksafe::generation();
            let (mut rx, mut tx) = (rx, tx);
            let port = unsafe { libc::mach_thread_self() } as u64;
            if tsd_off != 0 && mts_off != 0 {
                unsafe { *((pthread + tsd_off + mts_off) as *mut u64) = port };
            }
            let mut cpu = Cpu::new();
            let mut job = job;
            loop {
                *cpu = Cpu::default();
                cpu.pc = start;
                cpu.x[0] = pthread;
                cpu.x[1] = port;
                cpu.x[2] = stack_low;
                cpu.x[3] = job.kevent_list;
                cpu.x[4] = job.flags;
                cpu.x[5] = job.nevents;
                cpu.tpidrro_el0 = pthread + tsd_off;
                cpu.x[31] = job.stack_top & !0xf;
                PARK.with(|p| p.set(false));
                IS_MANAGER.with(|m| m.set(job.flags & WQ_FLAG_THREAD_EVENT_MANAGER != 0));
                BOUND_WL.with(|b| b.set(job.wl));
                if let Some(id) = job.wl {
                    if let Some(st) = WL.lock().unwrap().as_mut() {
                        if let Some(w) = st.loops.get_mut(&id) {
                            w.servicer = owner_key(port);
                        }
                    }
                }
                crate::engine::run_thread(&mut cpu);
                // Every way out of run_thread goes through
                // worker_after_run, so a stale worker forgets its channel ends whether it parked
                // or terminated.
                let (r, t) = match worker_after_run(PARK.with(|p| p.get()), worker_is_stale(gen0), rx, tx) {
                    AfterRun::Idle(r, t) => (r, t),
                    AfterRun::End => break,
                };
                (rx, tx) = (r, t);
                // The workloop (if any) was released by the return path.
                IDLE.lock().unwrap().push(Worker { stack_low, pthread, tx: Some(tx.clone()) });
                match rx.recv() {
                    Ok(j) => job = j,
                    Err(_) => break,
                }
            }
            crate::threads::LIVE_THREADS.fetch_sub(1, Ordering::SeqCst);
    }))
    .expect("failed to spawn workqueue thread");
}

/// What a workqueue worker's host thread does once the guest thread has left run_thread.
pub(crate) enum AfterRun<J> {
    /// Go back to the idle list and wait on the receiver (the ends, handed back).
    Idle(Receiver<J>, Sender<J>),
    /// End the host thread. The channel ends are already gone: dropped (a worker of this process
    /// that terminated) or forgotten (a stale one).
    End,
}

/// `parked`: the guest thread returned to the workqueue
/// (WQOPS_THREAD_RETURN); otherwise it terminated. `stale`: this worker was made before the latest
/// fork, so this is the forking thread of a child. A stale worker ends its thread and forgets its
/// channel ends, never drops them: the parent's other clones and its receiver's state were copied,
/// and dropping takes the channel's internal lock, which a parent thread may have held at the fork.
/// J v4 checked staleness only on the parked path, so a stale worker that terminated dropped them.
pub(crate) fn worker_after_run<J>(parked: bool, stale: bool, rx: Receiver<J>, tx: Sender<J>) -> AfterRun<J> {
    if stale {
        // (Natively the guest's libdispatch kills such a child before a parked worker gets here;
        // the stress tool's workq-return mode compares.)
        std::mem::forget(rx);
        std::mem::forget(tx);
        return AfterRun::End;
    }
    if !parked {
        drop(rx);
        drop(tx);
        return AfterRun::End;
    }
    AfterRun::Idle(rx, tx)
}

fn flags_for(pp: u64) -> u64 {
    let mut f = WQ_FLAG_THREAD_NEWSPI | WQ_FLAG_THREAD_TSD_BASE_SET;
    if pp & PP_EVENT_MANAGER != 0 {
        return f | WQ_FLAG_THREAD_KEVENT | WQ_FLAG_THREAD_EVENT_MANAGER;
    }
    let qos_bits = (pp >> 8) & 0x3f;
    let qos = if qos_bits == 0 { 4 } else { 64 - qos_bits.leading_zeros() as u64 };
    f |= WQ_FLAG_THREAD_PRIO_QOS | qos.min(6);
    if pp & PP_OVERCOMMIT != 0 {
        f |= WQ_FLAG_THREAD_OVERCOMMIT;
    }
    if pp & PP_COOPERATIVE != 0 {
        f |= WQ_FLAG_THREAD_COOPERATIVE;
    }
    f
}

// ---- single event-manager thread ----

/// libdispatch requires at most one event-manager thread at a time; the
/// kernel keeps the manager's events pending while one is bound. Events that
/// fire while the manager runs are held here and handed over when it parks.
#[derive(Default)]
pub(crate) struct ManagerState {
    active: bool,
    pending: Vec<Kev>,
}

pub(crate) static MANAGER: FMutex<Option<ManagerState>> = FMutex::new(None);

thread_local! {
    static IS_MANAGER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Claim the manager slot for a new manager thread. Returns false when one is
/// already running (the event, if any, is queued for later).
fn claim_manager(ev: Option<Kev>) -> bool {
    let mut g = MANAGER.lock().unwrap();
    let st = g.get_or_insert_with(ManagerState::default);
    if st.active {
        if let Some(ev) = ev {
            // EV_CLEAR user events coalesce; other events are kept in order.
            if !st.pending.iter().any(|k| k.filter == ev.filter && k.ident == ev.ident && ev.filter == EVFILT_USER_) {
                st.pending.push(ev);
            }
        }
        return false;
    }
    st.active = true;
    true
}

/// The manager thread is parking: release the slot and start a new manager
/// for any events that fired meanwhile.
fn release_manager() {
    let pending = {
        let mut g = MANAGER.lock().unwrap();
        let st = g.get_or_insert_with(ManagerState::default);
        st.active = false;
        std::mem::take(&mut st.pending)
    };
    if pending.is_empty() {
        return;
    }
    let Some(w) = take_worker() else { return };
    if !claim_manager(None) {
        IDLE.lock().unwrap().push(w);
        return;
    }
    let list = w.kevent_list();
    let n = pending.len().min(WQ_KEVENT_LIST_LEN as usize);
    for (i, k) in pending.iter().take(n).enumerate() {
        unsafe { k.write(list + i as u64 * KEV_SIZE) };
    }
    let flags = flags_for(PP_EVENT_MANAGER) | WQ_FLAG_THREAD_KEVENT;
    let stack_top = w.data_buf() + WQ_KEVENT_DATA_SIZE;
    start_job(w, Job { flags, kevent_list: list, nevents: n as u64, stack_top, wl: None });
}

// ---- the anonymous workq kqueue ----

/// QoS/priority each knote on the workq kqueue was registered with, keyed by
/// (filter, ident). The host kernel does not hand the guest's pthread priority
/// (e.g. the event-manager flag) back on fired events, but the thread we start
/// for the event depends on it.
pub(crate) static ANON_QOS: FMutex<Option<HashMap<(i16, u64), i32>>> = FMutex::new(None);

fn record_anon_qos(changes: u64, nchanges: u64) {
    let mut g = ANON_QOS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    for i in 0..nchanges {
        let ke = unsafe { Kev::read(changes + i * KEV_SIZE) };
        if ke.flags & EV_DELETE != 0 {
            m.remove(&(ke.filter, ke.ident));
        } else if ke.flags & EV_ADD != 0 {
            m.insert((ke.filter, ke.ident), ke.qos);
        }
    }
}

fn restore_anon_qos(k: &mut Kev) {
    if k.qos != 0 {
        return;
    }
    if let Some(q) = ANON_QOS.lock().unwrap().as_ref().and_then(|m| m.get(&(k.filter, k.ident)).copied()) {
        k.qos = q;
    }
}

static ANON_KQ: AtomicI32 = AtomicI32::new(NOT_MADE);
static ANON_MONITOR: AtomicBool = AtomicBool::new(false);
/// Starts the monitor once per process (a fork child starts its own);
/// a std Once cannot be reset in a fork child.
pub(crate) static ANON_START: FMutex<()> = FMutex::new(());

/// The workq kqueue, its monitor started on first use; or the errno to answer the guest.
fn anon_kq() -> Result<i32, i64> {
    let kq = kept_kq(&ANON_KQ)?;
    start_monitor_once(&ANON_MONITOR, &ANON_START, "maclator-workq", anon_monitor);
    Ok(kq)
}

/// Deliver each event fired on the workq kqueue to a worker thread.
fn anon_monitor() {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let kq = ANON_KQ.load(Ordering::SeqCst);
    let mut fails = 0;
    loop {
        let Some(w) = take_worker() else {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        };
        let list = w.kevent_list();
        let mut avail: u64 = WQ_KEVENT_DATA_SIZE;
        let r = kevent_qos(kq, 0, 0, list, 1, w.data_buf(), &mut avail as *mut u64 as u64, KEVENT_FLAG_STACK_DATA);
        if r.carry || r.rax == 0 {
            IDLE.lock().unwrap().push(w);
            if r.carry && r.rax as i32 != libc::EINTR {
                pause_after_failed_wait("workqueue", &WARNED, r.rax as i32, &mut fails);
            }
            continue;
        }
        fails = 0;
        let mut ev = unsafe { Kev::read(list) };
        restore_anon_qos(&mut ev);
        unsafe { ev.write(list) };
        let qos = ev.qos as u32 as u64;
        let flags = flags_for(qos) | WQ_FLAG_THREAD_KEVENT;
        if crate::syscalls::TRACE.load(Ordering::Relaxed) {
            let k = unsafe { Kev::read(list) };
            eprintln!("[workq] anon event ident={:#x} filter={} flags={:#x} qos={:#x} fflags={:#x} data={:#x} udata={:#x} -> thread flags={:#x} n={}", k.ident, k.filter, k.flags, k.qos, k.fflags, k.data, k.udata, flags, r.rax);
        }
        let stack_top = w.data_buf() + avail;
        if flags & WQ_FLAG_THREAD_EVENT_MANAGER != 0 && !claim_manager(Some(ev)) {
            IDLE.lock().unwrap().push(w);
            continue;
        }
        start_job(w, Job { flags, kevent_list: list, nevents: r.rax, stack_top, wl: None });
    }
}

// ---- workloops ----

/// Every workloop's regular knotes live on one host kqueue, made before the guest starts (`prepare`): a workloop is
/// a kqueue of the kernel's that needs no call of the program's, and a kqueue per workloop would need one each time
/// a workloop first gets a knote, which a sandbox that lists the program's system calls refuses. On the shared kqueue
/// each knote is the host's knote under a token of its own (its udata), which says whose it is; the monitor takes each
/// event as it fires and keeps it with its workloop (`Workloop::staged`) until a servicer or a poll of that workloop
/// takes it, as the kernel keeps a workloop's fired knotes queued until its servicer runs.
static WL_KQ: AtomicI32 = AtomicI32::new(NOT_MADE);
static WL_MONITOR: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct Workloop {
    /// Events of this workloop's regular knotes, taken from the shared kqueue and not yet delivered, oldest first.
    staged: std::collections::VecDeque<Staged>,
    /// The thread-request knote (EVFILT_WORKLOOP), if any.
    thread_request: Option<Kev>,
    /// The thread request is active (fired and not yet delivered). Each
    /// delivery hands libdispatch a +1 reference, so it must be delivered
    /// once per activation.
    tr_active: bool,
    /// A servicer thread is currently bound.
    bound: bool,
    /// Thread (mach port name) that owns the workloop, e.g. a dispatch_sync
    /// drainer. No async servicer is started while there is an owner.
    owner: u64,
    /// Port of the bound servicer thread.
    servicer: u64,
}

/// An event of a workloop's knote, taken from the shared kqueue.
struct Staged {
    token: u64,
    ev: Kev,
    /// For a message the kernel received into the data area as the event was taken (EVFILT_MACHPORT with
    /// MACH_RCV_MSG and no buffer of its own): the bytes it used there, and where in them ext[0] points.
    msg: Option<(Vec<u8>, usize)>,
}

/// A regular knote of a workloop, kept on the shared kqueue under its token.
#[derive(Clone)]
struct Knote {
    wl: u64,
    key: KnoteKey,
    filter: i16,
    ident: u64,
    /// The guest's udata, given back in its events.
    udata: u64,
    /// The guest's EV_DISPATCH, EV_ONESHOT, EV_UDATA_SPECIFIC and EV_VANISHED, as it attached the knote: the kernel keeps
    /// them from then on.
    gflags: u16,
    /// The guest's latest fflags, data, ext and qos: a change to the host knote of the emulation's own gives them again, as
    /// the kernel gives every change of a knote to its filter, which takes them for the knote's (a timer's interval, a
    /// port's receive options).
    params: Kev,
    /// The host knote has EV_UDATA_SPECIFIC: the guest's, or added so that knotes of different workloops with one ident
    /// and filter stay apart, as on kqueues of their own. Not added to an EV_DISPATCH knote, whose deletion the host
    /// would then defer as it defers an EV_DISPATCH2 knote's.
    specific: bool,
    /// Enabled, as the guest sees it.
    enabled: bool,
    /// Its deletion is deferred until it is enabled (an EV_DISPATCH2 knote the guest deleted while it was disabled, or
    /// an EV_ONESHOT one whose event was delivered): enabled, it gives that deletion as an event, and nothing else.
    deferred: bool,
    /// The host knote is enabled as far as the emulation's own changes go (the host also disables an EV_DISPATCH knote,
    /// and drops or defers an EV_ONESHOT one, as it gives an event; staging notes that too).
    host_enabled: bool,
    /// An EV_DISPATCH knote whose event was delivered: natively it gives no other event until the guest enables it.
    spent: bool,
}

/// (workloop, filter, ident, and for an EV_UDATA_SPECIFIC knote the guest's udata): what names a knote on its kqueue.
type KnoteKey = (u64, i16, u64, Option<u64>);

/// A NOTE_WL_SYNC_WAIT/WAKE knote (ident = waiter thread id).
#[derive(Default)]
struct SyncKnote {
    /// NOTE_WL_SYNC_WAKE has been applied (sticky until the knote is deleted).
    woken: bool,
}

pub(crate) static WLOG: FMutex<std::collections::VecDeque<String>> = FMutex::new(std::collections::VecDeque::new());

pub fn wlog_enabled() -> bool {
    static V: crate::forksync::FOnceLock<bool> = crate::forksync::FOnceLock::new();
    *V.get_or_init(|| std::env::var_os("MACLATOR_WL_LOG").is_some())
}

pub fn wlog(msg: String) {
    let mut l = WLOG.lock().unwrap();
    if l.len() >= 600 {
        l.pop_front();
    }
    l.push_back(msg);
}

pub fn dump_log() {
    if let Ok(l) = WLOG.try_lock() {
        for m in l.iter() {
            eprintln!("  wl| {m}");
        }
    }
}

fn owner_tracking() -> bool {
    static V: crate::forksync::FOnceLock<bool> = crate::forksync::FOnceLock::new();
    *V.get_or_init(|| std::env::var_os("MACLATOR_WL_NO_OWNER").is_none())
}

/// Thread identity as stored in dispatch locks: the port name with its low
/// two bits masked off (the kernel's ipc_entry_name_mask undoes this).
fn current_port() -> u64 {
    owner_key(unsafe { libc::pthread_mach_thread_np(libc::pthread_self()) as u64 })
}

fn owner_key(port: u64) -> u64 {
    port & !3
}

pub(crate) struct WlState {
    loops: HashMap<u64, Workloop>,
    sync: HashMap<(u64, u64), SyncKnote>,
    /// Every workloop's regular knotes, by token.
    knotes: HashMap<u64, Knote>,
    by_key: HashMap<KnoteKey, u64>,
    /// The host knotes without EV_UDATA_SPECIFIC, by (filter, ident): the shared kqueue can hold only one each.
    plain: HashMap<(i16, u64), u64>,
    next_token: u64,
}

impl WlState {
    fn new() -> WlState {
        WlState { loops: HashMap::new(), sync: HashMap::new(), knotes: HashMap::new(), by_key: HashMap::new(), plain: HashMap::new(), next_token: 0 }
    }
}

pub(crate) static WL: FMutex<Option<WlState>> = FMutex::new(None);
static WL_CV: FCondvar = FCondvar::new();
/// As ANON_START, for the workloop monitor.
pub(crate) static WL_MON_START: FMutex<()> = FMutex::new(());

/// In a fork child, with every lock fresh. The parent's workers, manager
/// thread and monitors are not in the child, and its kqueues are not inherited by a fork. An
/// idle worker's channel is forgotten, never dropped or sent to: its receiver is parked in the
/// parent, and waking it from the child is fatal. The child makes kqueues of its own, new workers
/// and monitors, when it first needs them.
pub(crate) fn reset_in_child() {
    // Safety: forksafe::fork's child branch, before the child starts a thread.
    unsafe { WL_CV.reinit_in_child() };
    // The forking thread's own bindings name the parent's
    // manager slot and workloops, which are dropped below.
    IS_MANAGER.with(|m| m.set(false));
    BOUND_WL.with(|b| b.set(None));
    std::mem::forget(std::mem::take(&mut *IDLE.lock().unwrap()));
    std::mem::forget(MANAGER.lock().unwrap().take());
    std::mem::forget(WL.lock().unwrap().take());
    std::mem::forget(ANON_QOS.lock().unwrap().take());
    // A refusal is kept: the child has its parent's sandbox.
    for slot in [&ANON_KQ, &WL_KQ] {
        if slot.load(Ordering::SeqCst) >= 0 {
            slot.store(NOT_MADE, Ordering::SeqCst);
        }
    }
    ANON_MONITOR.store(false, Ordering::SeqCst);
    WL_MONITOR.store(false, Ordering::SeqCst);
}

/// A worker made before the latest fork in this process (so in a parent)
/// is the forking thread of a child and must not wait on its parent-made receiver.
pub(crate) fn worker_is_stale(gen0: u64) -> bool {
    crate::forksafe::generation() != gen0
}

#[cfg(test)]
pub(crate) fn test_is_manager() -> bool {
    IS_MANAGER.with(|m| m.get())
}

#[cfg(test)]
pub(crate) fn test_set_manager_and_wl() {
    IS_MANAGER.with(|m| m.set(true));
    BOUND_WL.with(|b| b.set(Some(7)));
}

#[cfg(test)]
pub(crate) fn test_bound_wl() -> Option<u64> {
    BOUND_WL.with(|b| b.get())
}

/// Tests: park a fake idle worker's receiver on a thread of this process.
#[cfg(test)]
pub(crate) fn test_park_an_idle_worker() -> std::thread::JoinHandle<()> {
    let (tx, rx) = channel::<Job>();
    let h = std::thread::spawn(move || drop(rx.recv()));
    std::thread::sleep(std::time::Duration::from_millis(100));
    let mut w = Worker::new().expect("worker stack");
    w.tx = Some(tx);
    IDLE.lock().unwrap().push(w);
    h
}

#[cfg(test)]
pub(crate) fn test_idle_len() -> usize {
    IDLE.lock().unwrap().len()
}

#[cfg(test)]
pub(crate) fn test_clear_idle() {
    IDLE.lock().unwrap().clear();
}

fn with_wl<R>(f: impl FnOnce(&mut WlState) -> R) -> R {
    let mut g = WL.lock().unwrap();
    let st = g.get_or_insert_with(WlState::new);
    f(st)
}

/// The shared kqueue of the workloops' knotes, its monitor started on first use; or the errno to answer the guest.
fn wl_kq() -> Result<i32, i64> {
    let kq = kept_kq(&WL_KQ)?;
    start_monitor_once(&WL_MONITOR, &WL_MON_START, "maclator-workloops", wl_monitor);
    Ok(kq)
}

extern "C" {
    fn mach_msg_destroy(msg: *mut libc::c_void);
}

/// Apply one change to the shared kqueue: the errno it was answered, 0 when it was applied.
fn host_change(kq: i32, h: &Kev) -> i64 {
    let mut b = [0u8; 72];
    let mut out = [0u8; 72];
    unsafe { h.write(b.as_mut_ptr() as u64) };
    let r = kevent_qos(kq, b.as_ptr() as u64, 1, out.as_mut_ptr() as u64, 1, 0, 0, KEVENT_FLAG_IMMEDIATE | KEVENT_FLAG_ERROR_EVENTS);
    if r.carry {
        return r.rax as i64;
    }
    if r.rax > 0 {
        let e = unsafe { Kev::read(out.as_ptr() as u64) };
        if e.flags & EV_ERROR != 0 {
            return e.data;
        }
    }
    0
}

/// The host change `flags` (an action, no EV_ADD) for knote `kn` under `token`, with the knote's own parameters, so that
/// the filter is given what it has (an EVFILT_USER knote no command: NOTE_TRIGGER and the fflags operations are
/// commands).
fn host_kev(kn: &Knote, token: u64, flags: u16) -> Kev {
    let p = &kn.params;
    let fflags = if kn.filter == EVFILT_USER_ { 0 } else { p.fflags };
    Kev { ident: kn.ident, filter: kn.filter, flags: flags | if kn.specific { EV_UDATA_SPECIFIC } else { 0 }, udata: token, fflags, data: p.data, ext: p.ext, qos: p.qos, xflags: 0 }
}

/// The event ends its knote's events: a deletion the host had deferred, or a source that vanished.
fn gone(ev: &Kev) -> bool {
    ev.flags & (EV_DELETE | EV_VANISHED) != 0
}

/// The event is a deferred deletion, done: natively it is what a knote whose deletion is deferred delivers, a
/// vanishing not excepted.
fn deleted(ev: &Kev) -> bool {
    ev.flags & EV_DELETE != 0
}

/// The event's knote stays active while what it reports holds (no EV_CLEAR, EV_DISPATCH or EV_ONESHOT on the host
/// knote, whose flags the event carries, the ones the kernel adds included: a timer's or a process's EV_CLEAR): taken,
/// it would be taken again and again, so it is kept quiet until its event is delivered. Any other knote's activation is
/// used up as its event is taken.
fn level(ev: &Kev) -> bool {
    ev.flags & (EV_CLEAR | EV_DISPATCH | EV_ONESHOT) == 0 && !gone(ev)
}

/// Release what a taken message carries (its rights and memory), when it will not be delivered: its knote was deleted
/// before it was. Natively it would still be queued on its port, and libdispatch deletes a receive knote as it is done
/// with the port, which destroys what is queued there too.
fn destroy_msg(s: Staged) {
    if let Some((mut b, off)) = s.msg {
        if s.ev.fflags == 0 && b.len() >= off.saturating_add(24) {
            unsafe { mach_msg_destroy(b.as_mut_ptr().add(off).cast()) };
        }
    }
}

/// Take the staged events of `token` that `which` picks (kept in order) out of workloop `wl_id`.
fn take_staged(st: &mut WlState, wl_id: u64, token: u64, which: impl Fn(&Staged) -> bool) -> Vec<Staged> {
    let Some(wl) = st.loops.get_mut(&wl_id) else { return Vec::new() };
    let (taken, kept): (std::collections::VecDeque<Staged>, std::collections::VecDeque<Staged>) =
        std::mem::take(&mut wl.staged).into_iter().partition(|s| s.token == token && which(s));
    wl.staged = kept;
    taken.into_iter().collect()
}

/// What a disable takes back: natively a disabled knote's queued event is dequeued, and comes again once it is enabled
/// if its filter still says so. For a level-triggered knote the host says so again once its knote is enabled; a message
/// already received, or an activation used up as the event was taken, is kept here until then.
fn taken_back(s: &Staged) -> bool {
    s.msg.is_none() && level(&s.ev)
}

fn all(_: &Staged) -> bool {
    true
}

fn staged_count(st: &WlState, wl_id: u64, token: u64) -> usize {
    st.loops.get(&wl_id).map_or(0, |wl| wl.staged.iter().filter(|s| s.token == token).count())
}

fn forget_knote(st: &mut WlState, token: u64) {
    if let Some(kn) = st.knotes.remove(&token) {
        st.by_key.remove(&kn.key);
        if !kn.specific {
            st.plain.remove(&(kn.filter, kn.ident));
        }
    }
}

/// A regular knote's change on workloop `wl_id`, applied to the shared kqueue. Returns an errno (0 on success).
fn wl_apply_regular(st: &mut WlState, wl_id: u64, ke: &Kev) -> i64 {
    let kq = match wl_kq() {
        Ok(kq) => kq,
        Err(e) => return e,
    };
    let key: KnoteKey = (wl_id, ke.filter, ke.ident, (ke.flags & EV_UDATA_SPECIFIC != 0).then_some(ke.udata));
    let found = st.by_key.get(&key).copied();
    let err = match found {
        _ if ke.flags & EV_DELETE != 0 => found.map_or(libc::ENOENT as i64, |t| knote_delete(st, kq, t, ke)),
        _ if ke.flags & EV_ADD != 0 => knote_add(st, kq, wl_id, key, found, ke),
        Some(t) => knote_toggle(st, kq, t, ke),
        None => libc::ENOENT as i64,
    };
    schedule(st, wl_id);
    err
}

/// EV_ADD of a new knote, or of an existing one (a change: the host re-reads its state, as the kernel does).
fn knote_add(st: &mut WlState, kq: i32, wl_id: u64, key: KnoteKey, found: Option<u64>, ke: &Kev) -> i64 {
    static CLASH_SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let old = found.map(|t| (t, st.knotes[&t].clone()));
    if let Some((t, o)) = &old {
        if o.deferred {
            // Its deletion is deferred: the kernel gives the change to no filter, and an EV_ENABLE in it delivers the
            // deletion.
            return knote_toggle(st, kq, *t, ke);
        }
    }
    // A timer given new values before its event is delivered, as libdispatch programs its timers: the kernel discards a
    // fire not yet delivered. The host may have fired, that fire still with the monitor, so the timer goes on under a
    // token of its own, and the old fire, staged or still to come, is not taken for the new one's. A timer named by its
    // udata is attached anew, with the flags it was attached with, and the old one deleted once the host takes the new
    // values; one that is not (the host has one such per ident and filter) is changed in place under the new token.
    let retimed = old.as_ref().filter(|(_, o)| o.filter == EVFILT_TIMER);
    let (token, mut kn, new) = match (&old, retimed) {
        (Some((t, o)), None) => (*t, o.clone(), false),
        _ => {
            let specific = match retimed {
                Some((_, o)) => o.specific,
                None => ke.flags & (EV_UDATA_SPECIFIC | EV_DISPATCH) != EV_DISPATCH,
            };
            if !specific && retimed.is_none() && st.plain.contains_key(&(ke.filter, ke.ident)) {
                // Another workloop's EV_DISPATCH knote of the same ident and filter, neither EV_UDATA_SPECIFIC: on kqueues of
                // their own both could be, on the shared one only one. libdispatch registers its EV_DISPATCH knotes
                // with EV_UDATA_SPECIFIC, so it does not ask this.
                if !CLASH_SAID.swap(true, Ordering::Relaxed) {
                    eprintln!("maclator: two workloops' EV_DISPATCH knotes with filter {} and ident {:#x}, without EV_UDATA_SPECIFIC: the second is answered ENOMEM", ke.filter, ke.ident);
                }
                return NO_KQUEUE;
            }
            st.next_token += 1;
            let (gflags, enabled) = match retimed {
                Some((_, o)) => (o.gflags, o.enabled),
                None => (ke.flags & PROTOCOL_FLAGS, true),
            };
            let kn = Knote { wl: wl_id, key, filter: ke.filter, ident: ke.ident, udata: ke.udata, gflags, params: *ke, specific, enabled, deferred: false, host_enabled: false, spent: false };
            (st.next_token, kn, true)
        }
    };
    let enabled = if ke.flags & EV_DISABLE != 0 { false } else { ke.flags & EV_ENABLE != 0 || kn.enabled };
    let staged = staged_count(st, wl_id, token);
    if !new && staged > 0 && st.loops[&wl_id].staged.iter().any(|s| s.token == token && s.ev.flags & EV_ONESHOT != 0) {
        // An EV_ONESHOT knote whose event waits: the host dropped it, or deferred its deletion, as it gave that event.
        // Natively the knote is there until the event is delivered, and is then gone; the change is kept for it.
        kn.udata = ke.udata;
        kn.params = *ke;
        kn.enabled = enabled;
        st.knotes.insert(token, kn);
        if !enabled {
            drop(take_staged(st, wl_id, token, taken_back));
        }
        return 0;
    }
    let mut h = *ke;
    h.udata = token;
    // The flags the knote was attached with: the kernel keeps them, and a later change cannot alter them.
    if old.is_some() {
        h.flags = (h.flags & !PROTOCOL_FLAGS) | (kn.gflags & PROTOCOL_FLAGS);
    }
    if kn.specific {
        h.flags |= EV_UDATA_SPECIFIC;
    }
    let host_enabled;
    if new {
        // As the guest gave it, EV_ENABLE included or not, unless it is to be disabled.
        h.flags |= EV_ADD;
        if !enabled {
            h.flags = (h.flags | EV_DISABLE) & !EV_ENABLE;
        }
        host_enabled = enabled;
    } else {
        // The host knote is enabled when the guest's is and none of its events is taken and not yet delivered: an
        // enabled one is left as the host has it (natively enabling it changes nothing and its event stays queued),
        // quiet if it gave an event that is taken, staged or still with the monitor.
        h.flags &= !(EV_ENABLE | EV_DISABLE);
        host_enabled = if !enabled {
            h.flags |= EV_DISABLE;
            false
        } else if !kn.host_enabled && staged == 0 {
            h.flags |= EV_ENABLE;
            true
        } else {
            kn.host_enabled
        };
    }
    let err = host_change(kq, &h);
    if err != 0 {
        return err;
    }
    if let Some((t, o)) = retimed {
        if o.specific {
            host_change(kq, &host_kev(o, *t, EV_DELETE | EV_ENABLE));
        }
        drop(take_staged(st, wl_id, *t, all));
        forget_knote(st, *t);
    }
    kn.udata = ke.udata;
    kn.params = *ke;
    kn.enabled = enabled;
    kn.host_enabled = host_enabled;
    if enabled && ke.flags & EV_ENABLE != 0 {
        kn.spent = false;
    }
    if new {
        st.by_key.insert(key, token);
        if !kn.specific {
            st.plain.insert((kn.filter, kn.ident), token);
        }
    }
    st.knotes.insert(token, kn);
    if !enabled {
        drop(take_staged(st, wl_id, token, taken_back));
    }
    0
}

/// EV_ENABLE or EV_DISABLE of an existing knote (or another change without EV_ADD or EV_DELETE, given as it is).
fn knote_toggle(st: &mut WlState, kq: i32, token: u64, ke: &Kev) -> i64 {
    let kn = st.knotes[&token].clone();
    if kn.filter == EVFILT_TIMER && !kn.deferred {
        // Natively the timer's filter is given the change, as with EV_ADD: its new values, its pending fire discarded.
        return knote_add(st, kq, kn.wl, kn.key, Some(token), ke);
    }
    let mut h = Kev { fflags: ke.fflags, data: ke.data, ext: ke.ext, qos: ke.qos, ..host_kev(&kn, token, ke.flags & !EV_UDATA_SPECIFIC) };
    let mut send = true;
    let mut enables = false;
    if ke.flags & EV_ENABLE != 0 && ke.flags & EV_DISABLE == 0 {
        if kn.deferred {
            // Enabled, the host gives its deferred deletion as an event, as the kernel does, and the knote delivers nothing
            // else: a message taken for it is released. Once that deletion is taken, the host has no knote left, and the
            // kernel takes a further enable as done.
            for s in take_staged(st, kn.wl, token, |s| !deleted(&s.ev)) {
                destroy_msg(s);
            }
            send = staged_count(st, kn.wl, token) == 0;
            enables = send;
        } else if kn.enabled || staged_count(st, kn.wl, token) > 0 {
            // Natively enabling an enabled knote changes nothing, and its event, if one is queued, stays so. The host knote
            // may be quiet with an event of it taken (staged, or still with the monitor): it stays so until that event is
            // delivered, and is given the rest of the change.
            h.flags &= !EV_ENABLE;
        } else {
            enables = true;
        }
    }
    // The host dropped an EV_ONESHOT knote as it gave an event that is taken: natively the knote is still there.
    let err = if send { host_change(kq, &h) } else { 0 };
    let err = if err == libc::ENOENT as i64 { 0 } else { err };
    if err != 0 {
        return err;
    }
    let k = st.knotes.get_mut(&token).unwrap();
    // The kernel gives every change to the filter, and takes a new udata for a knote not named by it.
    k.params = *ke;
    if k.gflags & EV_UDATA_SPECIFIC == 0 {
        k.udata = ke.udata;
    }
    if ke.flags & EV_DISABLE != 0 {
        k.enabled = false;
        k.host_enabled = false;
        drop(take_staged(st, kn.wl, token, taken_back));
    } else if ke.flags & EV_ENABLE != 0 {
        k.enabled = true;
        k.spent = false;
        if enables {
            k.host_enabled = true;
        }
    }
    0
}

/// EV_DELETE of an existing knote: `ke` is the guest's change.
fn knote_delete(st: &mut WlState, kq: i32, token: u64, ke: &Kev) -> i64 {
    let kn = st.knotes[&token].clone();
    // As the guest asked (the host refuses what the kernel would), with EV_ENABLE added for a knote the guest sees
    // enabled: the host may have disabled it as it gave an event that is taken (staged, or still with the monitor),
    // where natively that event is still queued and the knote enabled. So the host defers the deletion of a disabled
    // EV_DISPATCH2 knote (EINPROGRESS) exactly when the kernel would: when the guest's knote is disabled and the guest
    // does not enable it in the same change.
    let (enable, keep) = if kn.enabled { (EV_ENABLE, !(EV_UDATA_SPECIFIC | EV_ENABLE | EV_DISABLE)) } else { (ke.flags & EV_ENABLE, !(EV_UDATA_SPECIFIC | EV_ENABLE)) };
    let h = Kev { fflags: ke.fflags, data: ke.data, ext: ke.ext, qos: ke.qos, ..host_kev(&kn, token, (ke.flags & keep) | enable) };
    let err = match host_change(kq, &h) {
        // The host dropped it as it gave an event that is taken: an EV_ONESHOT knote's, or one saying it is gone. Natively
        // that event is still queued, and the knote with it. (A host knote dropped with no event, its descriptor closed and
        // the number reused, is answered the same, where natively the guest's would be gone too: libdispatch deletes its
        // knotes before it closes their descriptors.)
        e if e == libc::ENOENT as i64 => 0,
        e => e,
    };
    if err == 0 {
        // Natively a deleted knote's queued event goes with it.
        for s in take_staged(st, kn.wl, token, all) {
            destroy_msg(s);
        }
        forget_knote(st, token);
    } else if err == libc::EINPROGRESS as i64 {
        // A knote whose deletion is deferred delivers that deletion and nothing else, a vanishing included.
        for s in take_staged(st, kn.wl, token, |s| !deleted(&s.ev)) {
            destroy_msg(s);
        }
        let k = st.knotes.get_mut(&token).unwrap();
        k.deferred = true;
        k.host_enabled = false;
    }
    err
}

/// Before the guest closes `fd` (close, or dup2 onto it): the tokens of the workloop knotes on it, for `fd_closed`.
/// Taken first, so a knote registered on the number once it is free again is not taken for one of them.
pub(crate) fn fd_closing(fd: i32) -> Vec<u64> {
    let mut g = WL.lock().unwrap();
    let Some(st) = g.as_mut() else { return Vec::new() };
    st.knotes.iter().filter(|(_, kn)| on_fd(kn) && kn.ident == fd as u64).map(|(t, _)| *t).collect()
}

/// After the guest closed a descriptor whose knotes were `tokens` (fd_closing): natively the kernel drops them with
/// their queued events, except one that asked for EV_VANISHED, which is told; the host has done the same with its own.
/// The dropped ones' records and events go here. A vanishing one with an event taken and not yet delivered: natively
/// that event becomes the vanishing, so it is dropped here and the host knote (quiet with it) enabled to give that.
pub(crate) fn fd_closed(tokens: &[u64]) {
    if tokens.is_empty() {
        return;
    }
    let mut g = WL.lock().unwrap();
    let Some(st) = g.as_mut() else { return };
    let kq = WL_KQ.load(Ordering::SeqCst);
    for t in tokens {
        let Some(kn) = st.knotes.get(t).cloned() else { continue };
        if kn.gflags & EV_VANISHED == 0 {
            for s in take_staged(st, kn.wl, *t, all) {
                destroy_msg(s);
            }
            forget_knote(st, *t);
        } else if !take_staged(st, kn.wl, *t, |s| !gone(&s.ev)).is_empty() && kn.enabled {
            host_change(kq, &host_kev(&kn, *t, EV_ENABLE));
            st.knotes.get_mut(t).unwrap().host_enabled = true;
        }
    }
}

/// A knote on a descriptor, which the kernel drops when the descriptor is closed (its filter's knotes are attached to
/// the file).
fn on_fd(kn: &Knote) -> bool {
    matches!(kn.filter, EVFILT_READ | EVFILT_WRITE | EVFILT_VNODE | EVFILT_EXCEPT | EVFILT_SOCK | EVFILT_NW_CHANNEL)
}

#[cfg(test)]
static TAKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The monitor: take each event as it fires on the shared kqueue, one at a time (so a message the kernel receives
/// into the data area is that one event's), keep it with its workloop, and start a servicer if that workloop needs one.
fn wl_monitor() {
    static WARNED: AtomicBool = AtomicBool::new(false);
    // As much room for a message as a servicer's data area has (schedule).
    let mut data = vec![0u8; (WQ_KEVENT_DATA_SIZE - 8) as usize];
    let mut b = [0u8; 72];
    let mut fails = 0;
    let kq = WL_KQ.load(Ordering::SeqCst);
    loop {
        let mut avail = data.len() as u64;
        let r = kevent_qos(kq, 0, 0, b.as_mut_ptr() as u64, 1, data.as_mut_ptr() as u64, &mut avail as *mut u64 as u64, KEVENT_FLAG_STACK_DATA);
        if r.carry {
            if r.rax as i32 != libc::EINTR {
                pause_after_failed_wait("workloop", &WARNED, r.rax as i32, &mut fails);
            }
            continue;
        }
        fails = 0;
        if r.rax == 0 {
            continue;
        }
        #[cfg(test)]
        TAKEN.fetch_add(1, Ordering::SeqCst);
        let ev = unsafe { Kev::read(b.as_ptr() as u64) };
        // The kernel fills the data area from its top down: [free, end) is what this event's message took.
        let free = (avail as usize).min(data.len());
        let base = data.as_ptr() as u64 + free as u64;
        let len = (data.len() - free) as u64;
        let msg = (len > 0 && ev.ext[0] >= base && ev.ext[0] < base + len).then(|| (data[free..].to_vec(), (ev.ext[0] - base) as usize));
        let mut g = WL.lock().unwrap();
        let st = g.get_or_insert_with(WlState::new);
        stage(st, kq, ev, msg);
    }
}

/// Keep the event `ev` the monitor took (with its message, if any) with its knote's workloop.
fn stage(st: &mut WlState, kq: i32, ev: Kev, msg: Option<(Vec<u8>, usize)>) {
    let token = ev.udata;
    let s = Staged { token, ev, msg };
    let Some(kn) = st.knotes.get(&token) else {
        // Deleted after the event was taken: natively its queued event goes with it. A host knote left under a token
        // that names no knote (one the host made again as the guest changed a knote whose event was being taken) is
        // deleted, so it fires no more.
        if ev.flags & EV_UDATA_SPECIFIC != 0 && !gone(&ev) {
            host_change(kq, &Kev { ident: ev.ident, filter: ev.filter, flags: EV_DELETE | EV_ENABLE | EV_UDATA_SPECIFIC, udata: token, ..Default::default() });
        }
        destroy_msg(s);
        return;
    };
    let wl_id = kn.wl;
    if kn.deferred && !deleted(&ev) {
        // Its deletion is deferred: natively it delivers that, and nothing else.
        destroy_msg(s);
        return;
    }
    if !kn.enabled && taken_back(&s) {
        // Disabled after the event was taken: natively a disabled knote's queued event is taken back.
        return;
    }
    let quiet = ev.flags & (EV_DISPATCH | EV_ONESHOT) != 0 || level(&ev);
    let waiting = st.loops.get(&wl_id).and_then(|wl| wl.staged.iter().rposition(|x| x.token == token && x.msg.is_none() && !gone(&x.ev)));
    if ev.flags & EV_DISPATCH != 0 && s.msg.is_none() && !gone(&ev) && (kn.spent || waiting.is_some()) {
        // Natively an EV_DISPATCH knote gives one event until the guest enables it again: a second comes here only of
        // an enable while the first was being taken, and the host has disabled the knote again with it.
        st.knotes.get_mut(&token).unwrap().host_enabled = false;
        return;
    }
    if let (Some(i), false, true) = (waiting, level(&ev), s.msg.is_none() && !gone(&ev)) {
        // A knote whose activation its event used up, fired again while that event waits: natively the two are one
        // event, the counts added (a timer's fires, a signal's), the flags reported joined, and the end of its source
        // (EV_EOF, EV_ONESHOT) and ext as the later fire has them.
        let w = &mut st.loops.get_mut(&wl_id).unwrap().staged[i].ev;
        if matches!(ev.filter, EVFILT_TIMER | EVFILT_SIGNAL) {
            w.data = w.data.saturating_add(ev.data);
        } else {
            w.data = ev.data;
        }
        w.fflags |= ev.fflags;
        w.flags |= ev.flags & (EV_EOF | EV_ONESHOT | EV_ERROR);
        w.ext = ev.ext;
        return;
    }
    // The host knote stays quiet until this event is delivered, so a level-triggered knote is not taken again and again;
    // it is enabled again once its event is delivered. An EV_DISPATCH or EV_ONESHOT knote the host disabled, dropped or
    // deferred as it gave the event.
    if level(&ev) {
        host_change(kq, &host_kev(kn, token, EV_DISABLE));
    }
    if quiet {
        st.knotes.get_mut(&token).unwrap().host_enabled = false;
    }
    st.loops.entry(wl_id).or_default().staged.push_back(s);
    schedule(st, wl_id);
}

/// Deliver workloop `wl_id`'s staged events, oldest first, into the event list at `out` (at most `nmax`), as the kernel
/// would hand them to its servicer: each with the guest's udata, each message placed in the data area of `size` bytes
/// at `data`, of which `*avail` are left (`*avail` shrinks): from the top of what is left down with `top_down`, as
/// KEVENT_FLAG_STACK_DATA has it, else from the bottom of what is left up. With `data` 0 an event that holds a message
/// is not delivered. At most one event per knote; a knote's events stay in order; an event of a disabled knote waits
/// until the knote is enabled. Returns how many were delivered.
#[allow(clippy::too_many_arguments)]
fn deliver(st: &mut WlState, kq: i32, wl_id: u64, out: u64, nmax: u64, data: u64, size: u64, avail: &mut u64, top_down: bool) -> u64 {
    let Some(wl) = st.loops.get_mut(&wl_id) else { return 0 };
    let mut staged = std::mem::take(&mut wl.staged);
    let mut kept = std::collections::VecDeque::new();
    // Knotes whose next event waits for another delivery (one was delivered, or an earlier one waits).
    let mut later: Vec<u64> = Vec::new();
    let mut n = 0;
    while let Some(s) = staged.pop_front() {
        if n >= nmax {
            kept.push_back(s);
            kept.extend(staged.drain(..));
            break;
        }
        let Some(kn) = st.knotes.get(&s.token).cloned() else {
            destroy_msg(s);
            continue;
        };
        let at = match &s.msg {
            None => Some(0),
            Some((b, _)) if top_down => (data + *avail).checked_sub(b.len() as u64).map(|a| a & !15).filter(|&a| data != 0 && a >= data),
            Some((b, _)) => Some((data + (size - *avail) + 15) & !15).filter(|&a| data != 0 && a + b.len() as u64 <= data + size),
        };
        // Natively a disabled knote's event, whatever it reports, is not delivered until the knote is enabled.
        let held = !kn.enabled;
        let Some(at) = at.filter(|_| !held && !later.contains(&s.token)) else {
            later.push(s.token);
            kept.push_back(s);
            continue;
        };
        let mut e = s.ev;
        e.udata = kn.udata;
        if kn.gflags & EV_UDATA_SPECIFIC == 0 {
            e.flags &= !EV_UDATA_SPECIFIC;
        }
        if let Some((b, off)) = &s.msg {
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), at as *mut u8, b.len()) };
            e.ext[0] = at + *off as u64;
            *avail = if top_down { at - data } else { data + size - (at + b.len() as u64) };
        }
        unsafe { e.write(out + n * KEV_SIZE) };
        n += 1;
        later.push(s.token);
        // The knote as the kernel leaves it once its event is delivered: one whose deferred deletion this was is dropped;
        // an EV_ONESHOT one too, except an EV_DISPATCH2 one (a vanished one among them), which stays, disabled, its
        // deletion deferred until the guest deletes it; an EV_DISPATCH one is disabled until the guest enables it; any
        // other the host keeps quiet is enabled again (its event comes again if it still holds), unless another of its
        // events waits. EV_ONESHOT is the event's: the guest's, or the kernel's own (end of file on a port that died, a
        // vanished source, an absolute timer).
        let dispatch2 = kn.gflags & (EV_DISPATCH | EV_UDATA_SPECIFIC) == EV_DISPATCH | EV_UDATA_SPECIFIC;
        if deleted(&e) || (e.flags & EV_ONESHOT != 0 && !(dispatch2 && !kn.deferred)) {
            forget_knote(st, s.token);
        } else if e.flags & EV_ONESHOT != 0 {
            let k = st.knotes.get_mut(&s.token).unwrap();
            k.enabled = false;
            k.deferred = true;
        } else if kn.gflags & EV_DISPATCH != 0 {
            let k = st.knotes.get_mut(&s.token).unwrap();
            k.enabled = false;
            k.spent = true;
        } else if kn.enabled && !kn.host_enabled && !staged.iter().chain(kept.iter()).any(|x| x.token == s.token) {
            host_change(kq, &host_kev(&kn, s.token, EV_ENABLE));
            st.knotes.get_mut(&s.token).unwrap().host_enabled = true;
        }
    }
    if let Some(wl) = st.loops.get_mut(&wl_id) {
        wl.staged = kept;
    }
    n
}

/// If the workloop needs a servicer and has none, bind a worker to it.
fn schedule(st: &mut WlState, wl_id: u64) {
    let Some(wl) = st.loops.get_mut(&wl_id) else { return };
    let tr_pending = wl.thread_request.is_some() && wl.tr_active;
    let owned = wl.owner != 0 && owner_tracking();
    if wl.bound || owned || (!tr_pending && wl.staged.is_empty()) {
        return;
    }
    let Some(w) = take_worker() else { return };
    let list = w.kevent_list();
    let mut n: u64 = 0;
    let mut qos: u64 = 0;
    if let (Some(tr), true) = (wl.thread_request, wl.tr_active) {
        unsafe { tr.write(list) };
        qos = tr.qos as u32 as u64;
        n = 1;
        wl.tr_active = false; // delivery deactivates the knote
    }
    let has_tr = wl.thread_request.is_some();
    // kqueue id lives just below the event list; data goes below that.
    unsafe { *((list - 8) as *mut u64) = wl_id };
    let data_buf = w.data_buf();
    let mut avail: u64 = WQ_KEVENT_DATA_SIZE - 8;
    let got = deliver(st, WL_KQ.load(Ordering::SeqCst), wl_id, list + n * KEV_SIZE, WQ_KEVENT_LIST_LEN - n, data_buf, avail, &mut avail, true);
    if n == 0 && got > 0 {
        qos = unsafe { Kev::read(list) }.qos as u32 as u64;
    }
    n += got;
    if n == 0 {
        // Only events that must wait (a disabled knote's message).
        IDLE.lock().unwrap().push(w);
        return;
    }
    st.loops.get_mut(&wl_id).unwrap().bound = true;
    if wlog_enabled() { wlog(format!("schedule wl{:x}: deliver n={} (tr={})", wl_id & 0xffffff, n, has_tr)); }
    let flags = (flags_for(qos) & !(WQ_FLAG_THREAD_EVENT_MANAGER)) | WQ_FLAG_THREAD_KEVENT | WQ_FLAG_THREAD_WORKLOOP;
    start_job(w, Job { flags, kevent_list: list, nevents: n, stack_top: data_buf + avail, wl: Some(wl_id) });
}

/// The workloop state word at the guest's `addr` (EV_EXTIDX_WL_ADDR), or the errno to answer.
///
/// The kernel (filt_wlupdate) loads it with copyin_atomic64 and answers EFAULT only when that load faults, after
/// retrying: EFAULT means the guest's pointer is bad, and libdispatch treats it as a fatal bug. So EFAULT is answered
/// only when our reader says the address is bad. When the reader was refused for a reason that says nothing about
/// the address (guestmem has reported it), the kernel would still have loaded the word, so load it here: it is in
/// this address space, in the guest's own queue, which the guest has just handed us. An aligned word is loaded
/// atomically, as copyin_atomic64 loads it.
fn wl_state_word(addr: u64) -> Result<u64, i64> {
    match crate::guestmem::read_u64_or_why(addr) {
        Ok(v) => Ok(v),
        Err(crate::guestmem::Unread::BadAddress) => Err(libc::EFAULT as i64),
        Err(crate::guestmem::Unread::Refused) => Ok(unsafe {
            if addr & 7 == 0 {
                (*(addr as *const std::sync::atomic::AtomicU64)).load(Ordering::SeqCst)
            } else {
                std::ptr::read_unaligned(addr as *const u64)
            }
        }),
    }
}

/// Apply a single change to a workloop. Returns an errno (0 on success).
fn wl_apply(st: &mut WlState, wl_id: u64, ke: &Kev) -> i64 {
    if ke.filter != EVFILT_WORKLOOP {
        return wl_apply_regular(st, wl_id, ke);
    }
    let ff = ke.fflags;
    let cmd = ff & NOTE_WL_COMMANDS_MASK;
    if cmd == NOTE_WL_SYNC_IPC {
        return libc::ENOENT as i64;
    }
    let me = current_port();
    let wl = st.loops.entry(wl_id).or_default();
    // State check and owner discovery (filt_wlupdate).
    // EV_EXTIDX_WL_ADDR = 1, _MASK = 2, _VALUE = 3 (0 is the lane)
    let (addr, mask, value) = (ke.ext[1], ke.ext[2], ke.ext[3]);
    let mut err: i64 = 0;
    let mut new_owner = wl.owner;
    if addr != 0 {
        match wl_state_word(addr) {
            Err(e) => return e,
            Ok(cur) => {
                if cur & mask != value & mask {
                    err = libc::ESTALE as i64;
                } else if ff & NOTE_WL_DISCOVER_OWNER != 0 {
                    let name = (cur as u32 & !3) as u64;
                    if name != 0 {
                        new_owner = name;
                    }
                }
            }
        }
    }
    if ff & NOTE_WL_END_OWNERSHIP != 0 && new_owner == me {
        new_owner = 0;
    }
    if wl.bound && new_owner == wl.servicer {
        new_owner = 0;
    }
    let owner_released = wl.owner != 0 && new_owner == 0;
    if wlog_enabled() { wlog(format!("t{:x} wl{:x} apply id={:x} fl={:x} ff={:x} qos={:x} err={} owner {:x}->{:x} bound={} tr={}/{}", me, wl_id & 0xffffff, ke.ident & 0xffffff, ke.flags, ke.fflags, ke.qos, err, wl.owner, new_owner, wl.bound, wl.thread_request.is_some(), wl.tr_active)); }
    wl.owner = new_owner;
    if err == libc::ESTALE as i64 && ff & NOTE_WL_IGNORE_ESTALE != 0 {
        // The update is dropped silently.
        if owner_released {
            schedule(st, wl_id);
        }
        return 0;
    }
    let mut result: i64 = err;
    if err == 0 {
        if cmd == NOTE_WL_THREAD_REQUEST {
            if ke.flags & EV_DELETE != 0 {
                wl.thread_request = None;
                wl.tr_active = false;
            } else if ke.flags & EV_ADD != 0 {
                // Attaching or touching the thread request fires it.
                let mut tr = *ke;
                tr.flags = EV_ADD | EV_ENABLE;
                wl.thread_request = Some(tr);
                wl.tr_active = true;
            }
        } else if cmd == NOTE_WL_SYNC_WAKE || cmd == NOTE_WL_SYNC_WAIT || cmd == 0 {
            let key = (wl_id, ke.ident);
            if ke.flags & EV_DELETE != 0 {
                // Dropping a waiter knote wakes the waiter.
                if st.sync.remove(&key).is_none() {
                    result = libc::ENOENT as i64;
                }
                WL_CV.notify_all();
            } else if cmd == NOTE_WL_SYNC_WAKE {
                st.sync.entry(key).or_default().woken = true;
                WL_CV.notify_all();
            } else if cmd == NOTE_WL_SYNC_WAIT {
                st.sync.entry(key).or_default();
            } else {
                result = libc::EINVAL as i64;
            }
        } else {
            result = libc::EINVAL as i64;
        }
    }
    schedule(st, wl_id);
    let _ = owner_released;
    result
}

/// kevent_id() on a workloop.
fn kevent_id(args: &[u64; 8]) -> hostsys::SysResult {
    let (wl_id, changes, nchanges, out, nout, data, avail, flags) = (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
    let mut nerr: u64 = 0;
    let mut wait_on: Option<u64> = None;
    {
        let mut g = WL.lock().unwrap();
        let st = g.get_or_insert_with(WlState::new);
        for i in 0..nchanges {
            let ke = unsafe { Kev::read(changes + i * KEV_SIZE) };
            let err = wl_apply(st, wl_id, &ke);
            if crate::syscalls::TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[kev]   wl={:#x} chg ident={:#x} filter={} flags={:#x} fflags={:#x} data={:#x} qos={:#x} -> err={}", wl_id, ke.ident, ke.filter, ke.flags, ke.fflags, ke.data, ke.qos, err);
            }
            if err == 0 && ke.filter == EVFILT_WORKLOOP && ke.fflags & NOTE_WL_COMMANDS_MASK == NOTE_WL_SYNC_WAIT && ke.flags & EV_DELETE == 0 {
                wait_on = Some(ke.ident);
            }
            if err != 0 && flags & KEVENT_FLAG_ERROR_EVENTS != 0 && nerr < nout {
                let mut e = ke;
                e.flags |= EV_ERROR;
                e.data = err;
                unsafe { e.write(out + nerr * KEV_SIZE) };
                nerr += 1;
            }
        }
    }
    if let Some(tid) = wait_on {
        if wlog_enabled() { wlog(format!("t{:x} wl{:x} SYNC_WAIT id={:x} begin", current_port(), wl_id & 0xffffff, tid & 0xffffff)); }
        // dispatch_sync waiter: block until SYNC_WAKE for our tid.
        let g = WL.lock().unwrap();
        return finish_wait(g, (wl_id, tid), nerr);
    }
    if nerr > 0 || flags & KEVENT_FLAG_ERROR_EVENTS != 0 || nout == 0 {
        return hostsys::SysResult { rax: nerr, rdx: 0, carry: false };
    }
    // A poll of the workloop's regular knotes: their events taken from the shared kqueue and not yet delivered.
    // A data area: filled from its top down with KEVENT_FLAG_STACK_DATA, else from its bottom up (libdispatch's polls of
    // a workloop give one without that flag).
    let area = data != 0 && avail != 0;
    let given = if area { unsafe { *(avail as *const u64) } } else { 0 };
    let mut room = given;
    let top_down = flags & KEVENT_FLAG_STACK_DATA != 0;
    let n = with_wl(|st| deliver(st, WL_KQ.load(Ordering::SeqCst), wl_id, out, nout, if area { data } else { 0 }, given, &mut room, top_down));
    // As the kernel, which writes it back only when a message took some of it.
    if room != given {
        unsafe { *(avail as *mut u64) = room };
    }
    hostsys::SysResult { rax: n, rdx: 0, carry: false }
}

fn finish_wait(mut g: crate::forksync::FMutexGuard<'_, Option<WlState>>, key: (u64, u64), nerr: u64) -> hostsys::SysResult {
    loop {
        let st = g.as_mut().unwrap();
        match st.sync.get(&key) {
            Some(k) if k.woken => {
                return hostsys::SysResult { rax: nerr, rdx: 0, carry: false };
            }
            // Deleted by the waker: also a wake-up.
            None => return hostsys::SysResult { rax: nerr, rdx: 0, carry: false },
            _ => {}
        }
        g = WL_CV.wait(g).unwrap();
    }
}

// ---- syscall entry points ----

/// Guest kevent_qos / kevent_id aimed at the workq kqueue or a workloop.
/// Returns None when the call should go to the host unchanged.
pub fn kevent_redirect(n: u64, args: &[u64; 8]) -> Option<hostsys::SysResult> {
    match n {
        SYS_KEVENT_QOS => {
            let flags = args[7];
            if flags & KEVENT_FLAG_WORKQ == 0 {
                return None;
            }
            let kq = match anon_kq() {
                Ok(kq) => kq,
                Err(e) => return Some(hostsys::SysResult { rax: e as u64, rdx: 0, carry: true }),
            };
            record_anon_qos(args[1], args[2]);
            let mut a = *args;
            a[0] = kq as u64;
            a[7] = flags & !(KEVENT_FLAG_WORKQ | KEVENT_FLAG_PARKING);
            if a[4] == 0 {
                a[7] |= KEVENT_FLAG_IMMEDIATE;
            }
            Some(hostsys::unix(SYS_KEVENT_QOS, &a))
        }
        SYS_KEVENT_ID => Some(kevent_id(args)),
        _ => None,
    }
}

/// workq_kernreturn(options, item, affinity, prio)
pub fn kernreturn(_cpu: &mut Cpu, a: &[u64; 8]) -> Result<u64, u64> {
    match a[0] {
        WQOPS_QUEUE_NEWSPISUPP | WQOPS_SET_EVENT_MANAGER_PRIORITY | WQOPS_SETUP_DISPATCH | WQOPS_SHOULD_NARROW => Ok(0),
        WQOPS_QUEUE_REQTHREADS | WQOPS_QUEUE_REQTHREADS2 => {
            let count = requested_thread_count(a)?;
            let pp = a[3];
            // Starts the workq kqueue's monitor. The threads asked for need no kqueue, so they start without one.
            let _ = anon_kq();
            for _ in 0..count {
                let Some(w) = take_worker() else { return Err(libc::EAGAIN as u64) };
                let mut flags = flags_for(pp);
                if flags & WQ_FLAG_THREAD_EVENT_MANAGER != 0 && !claim_manager(None) {
                    IDLE.lock().unwrap().push(w);
                    continue;
                }
                if flags & WQ_FLAG_THREAD_EVENT_MANAGER == 0 {
                    flags &= !WQ_FLAG_THREAD_KEVENT;
                }
                let stack_top = w.pthread;
                start_job(w, Job { flags, kevent_list: 0, nevents: 0, stack_top, wl: None });
            }
            Ok(0)
        }
        WQOPS_THREAD_KEVENT_RETURN | WQOPS_THREAD_RETURN => {
            if a[0] == WQOPS_THREAD_KEVENT_RETURN && a[1] != 0 && (a[2] as i64) > 0 {
                if let Ok(kq) = anon_kq() {
                    record_anon_qos(a[1], a[2]);
                    kevent_qos(kq, a[1], a[2], 0, 0, 0, 0, KEVENT_FLAG_IMMEDIATE);
                }
            }
            park();
            Ok(0)
        }
        WQOPS_THREAD_WORKLOOP_RETURN => {
            let wl = BOUND_WL.with(|b| b.get());
            if let Some(wl_id) = wl {
                let mut g = WL.lock().unwrap();
                let st = g.get_or_insert_with(WlState::new);
                if a[1] != 0 && (a[2] as i64) > 0 {
                    for i in 0..a[2] {
                        let ke = unsafe { Kev::read(a[1] + i * KEV_SIZE) };
                        wl_apply(st, wl_id, &ke);
                    }
                }
                let me = current_port();
                if wlog_enabled() { wlog(format!("t{:x} wl{:x} WORKLOOP_RETURN nchanges={}", me, wl_id & 0xffffff, a[2])); }
                if let Some(w) = st.loops.get_mut(&wl_id) {
                    w.bound = false;
                    w.servicer = 0;
                    if w.owner == me {
                        w.owner = 0;
                    }
                }
                // Parking happens after this returns; the next servicer is
                // picked by schedule() (possibly this same thread later).
                drop(g);
                park();
                let mut g = WL.lock().unwrap();
                if let Some(st) = g.as_mut() {
                    schedule(st, wl_id);
                }
            } else {
                park();
            }
            Ok(0)
        }
        op => {
            if std::env::var_os("MACLATOR_TRACE").is_some() {
                eprintln!("[maclator] unhandled workq op {:#x}", op);
            }
            Ok(0)
        }
    }
}

fn requested_thread_count(a: &[u64; 8]) -> Result<u64, u64> {
    // libpthread calls __workq_kernreturn(op, NULL, numthreads, priority):
    // `item` is not the count. dispatch_apply waits for every requested worker,
    // so starting one worker for a multi-thread request can deadlock the guest.
    let count = a[2] as u32 as i32;
    if count <= 0 {
        return Err(libc::EINVAL as u64);
    }
    Ok(count as u64)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    extern "C" {
        fn sandbox_init_with_parameters(profile: *const libc::c_char, flags: u64, parameters: *const *const libc::c_char, errorbuf: *mut *mut libc::c_char) -> libc::c_int;
    }

    /// Run `f` in a child forked with every translator lock held (forksafe::fork), and return the code it exits with.
    /// The child is killed after `secs`.
    fn in_child(secs: u32, f: impl FnOnce() -> i32) -> i32 {
        let pid = crate::forksafe::fork();
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            unsafe { libc::alarm(secs) };
            let code = f();
            unsafe { libc::_exit(code) };
        }
        let mut st = 0;
        loop {
            let r = unsafe { libc::waitpid(pid, &mut st, 0) };
            if r == pid {
                break;
            }
            assert!(r < 0 && unsafe { *libc::__error() } == libc::EINTR, "waitpid");
        }
        if libc::WIFEXITED(st) {
            libc::WEXITSTATUS(st)
        } else {
            1000 + libc::WTERMSIG(st)
        }
    }

    /// In a child: apply a sandbox that allows everything but the kqueue system call, as Apple's profiles that list
    /// their calls do (WindowManager's allows the kevent calls and not kqueue). 0 when it is in force: the host then
    /// refuses kqueue with EPERM.
    fn refuse_kqueue_from_now_on() -> i32 {
        let p = c"(version 2)(allow default)(deny syscall-unix (syscall-number SYS_kqueue))";
        let mut err: *mut libc::c_char = std::ptr::null_mut();
        if unsafe { sandbox_init_with_parameters(p.as_ptr(), 0, std::ptr::null(), &mut err) } != 0 {
            return 20;
        }
        let r = hostsys::unix(SYS_KQUEUE, &[0; 8]);
        if !r.carry || r.rax != libc::EPERM as u64 {
            if !r.carry {
                unsafe { libc::close(r.rax as i32) };
            }
            return 21;
        }
        0
    }

    fn fd_open(fd: i32) -> bool {
        unsafe { libc::fcntl(fd, libc::F_GETFD) != -1 }
    }

    #[test]
    fn a_refused_kqueue_is_an_errno_and_standard_output_stays_open() {
        let code = in_child(20, || match kqueue_from(hostsys::SysResult { rax: libc::EPERM as u64, rdx: 0, carry: true }) {
            Err(e) if e == libc::EPERM && fd_open(1) => 0,
            Err(_) => 2,
            Ok(_) if !fd_open(1) => 3,
            Ok(_) => 4,
        });
        assert_eq!(code, 0, "2 another errno, 3 a descriptor and standard output closed, 4 a descriptor");
    }

    #[test]
    fn a_kqueue_the_host_made_is_kept_and_not_inherited_by_an_exec() {
        let r = hostsys::unix(SYS_KQUEUE, &[0; 8]);
        assert!(!r.carry, "control: the host makes a kqueue here");
        let fd = kqueue_from(r).expect("a kqueue");
        let mut b = [0u8; 72];
        let probe = kevent_qos(fd, 0, 0, b.as_mut_ptr() as u64, 1, 0, 0, KEVENT_FLAG_IMMEDIATE);
        let cloexec = unsafe { libc::fcntl(fd, libc::F_GETFD) } & libc::FD_CLOEXEC != 0;
        unsafe { libc::close(fd) };
        assert!(!probe.carry && probe.rax == 0, "not a working kqueue: {probe:?}");
        assert!(cloexec, "an exec would inherit it");
    }

    #[test]
    fn a_kqueue_is_put_above_a_low_soft_limit_and_the_limit_kept() {
        // As launchd starts most processes: a soft descriptor limit of 256, a high hard one.
        let code = in_child(20, || unsafe {
            let mut rl: libc::rlimit = std::mem::zeroed();
            libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl);
            rl.rlim_cur = 256;
            if rl.rlim_max <= 901 || libc::setrlimit(libc::RLIMIT_NOFILE, &rl) != 0 {
                return 20;
            }
            let fd = kqueue_from(hostsys::unix(SYS_KQUEUE, &[0; 8]));
            let mut now: libc::rlimit = std::mem::zeroed();
            libc::getrlimit(libc::RLIMIT_NOFILE, &mut now);
            match fd {
                Ok(fd) if fd >= 900 && now.rlim_cur == 256 => 0,
                Ok(_) if now.rlim_cur != 256 => 3,
                Ok(_) => 4,
                Err(_) => 5,
            }
        });
        assert_eq!(code, 0, "20 no reading (limits), 3 the soft limit not put back, 4 kept in the program's range, 5 refused");
    }

    #[test]
    fn a_soft_limit_that_could_not_be_put_back_is_said() {
        // Lowering the soft limit is always allowed, so the restore cannot be made to fail for real: a setrlimit that
        // takes the raise and refuses the lowering stands in for one that did.
        unsafe extern "C" fn refuse_lowering(res: libc::c_int, rl: *const libc::rlimit) -> libc::c_int {
            if (*rl).rlim_cur > 900 {
                return libc::setrlimit(res, rl);
            }
            *libc::__error() = libc::EPERM;
            -1
        }
        let code = in_child(20, || unsafe {
            let mut rl: libc::rlimit = std::mem::zeroed();
            libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl);
            rl.rlim_cur = 256;
            if rl.rlim_max <= 901 || libc::setrlimit(libc::RLIMIT_NOFILE, &rl) != 0 {
                return 20;
            }
            if SOFT_LIMIT_KEPT_RAISED.load(Ordering::SeqCst) {
                return 21;
            }
            // What it says on standard error, read from a pipe put in its place for the call.
            let mut e = [-1i32; 2];
            let saved = libc::dup(2);
            if libc::pipe(e.as_mut_ptr()) != 0 || saved < 0 || libc::dup2(e[1], 2) != 2 {
                return 22;
            }
            let made = kqueue_from_with(hostsys::unix(SYS_KQUEUE, &[0; 8]), refuse_lowering);
            libc::dup2(saved, 2);
            libc::close(e[1]);
            let mut buf = [0u8; 512];
            let n = libc::read(e[0], buf.as_mut_ptr().cast(), buf.len());
            let said = String::from_utf8_lossy(&buf[..n.max(0) as usize]).into_owned();
            match made {
                Ok(fd) if fd >= 900 => {}
                Ok(_) => return 4,
                Err(_) => return 5,
            }
            let raised = rl.rlim_max.min(1024);
            let mut now: libc::rlimit = std::mem::zeroed();
            libc::getrlimit(libc::RLIMIT_NOFILE, &mut now);
            if !said.contains("back to 256") || !said.contains(&format!("it stays at {raised}")) {
                3
            } else if now.rlim_cur != raised {
                6
            } else {
                0
            }
        });
        assert_eq!(code, 0, "20/21/22 no reading (limits, said already, or the pipe), 3 the failed restore not said with its limits, 4 kept low, 5 refused, 6 the limit kept is not the raised one");
    }

    #[test]
    fn a_kqueue_kept_low_never_takes_a_standard_descriptors_place() {
        // A descriptor limit below the high range, and standard output closed by the program: the kqueue made in its
        // place is moved, and the program still finds it closed.
        let code = in_child(20, || unsafe {
            // Both limits low, so the high range cannot be reached at all.
            let rl = libc::rlimit { rlim_cur: 64, rlim_max: 64 };
            if libc::setrlimit(libc::RLIMIT_NOFILE, &rl) != 0 || libc::close(1) != 0 {
                return 20;
            }
            match kqueue_from(hostsys::unix(SYS_KQUEUE, &[0; 8])) {
                Ok(fd) if fd > 2 && !fd_open(1) => 0,
                Ok(_) => 3,
                Err(_) => 4,
            }
        });
        assert_eq!(code, 0, "20 no reading (limit or close), 3 kept in standard output's place, 4 refused");
    }

    #[test]
    fn a_fork_child_of_a_process_that_may_not_make_a_kqueue_answers_enomem_and_stdout_stays_open() {
        // The residual of making the kqueues first: a child forked after its parent's sandbox came cannot make its own.
        let code = in_child(30, || {
            prepare();
            let c = refuse_kqueue_from_now_on();
            if c != 0 {
                return c;
            }
            in_child(20, || {
                let mut st = WlState::new();
                let ke = Kev { ident: 0x5157, filter: EVFILT_USER_, flags: EV_ADD, ..Default::default() };
                let e = wl_apply(&mut st, 0x6b71, &ke);
                let w = kevent_redirect(SYS_KEVENT_QOS, &[u64::MAX, 0, 0, 0, 0, 0, 0, KEVENT_FLAG_WORKQ | KEVENT_FLAG_IMMEDIATE]);
                if ANON_KQ.load(Ordering::SeqCst) != REFUSED || WL_KQ.load(Ordering::SeqCst) != REFUSED {
                    return 6;
                }
                match (e == NO_KQUEUE, w.is_some_and(|r| r.carry && r.rax as i64 == NO_KQUEUE), fd_open(1)) {
                    (true, true, true) => 0,
                    (_, _, false) => 3,
                    (false, _, _) => 4,
                    (true, false, _) => 5,
                }
            })
        });
        assert_eq!(code, 0, "20 the sandbox was not applied, 21 it did not refuse kqueue (no reading), 3 standard output closed, 4 the knote not ENOMEM, 5 the workq kevent not ENOMEM, 6 the child's kqueues not refused");
    }

    #[test]
    fn a_fork_child_makes_its_kqueues_when_it_needs_them_and_keeps_its_parents_refusal() {
        // A native fork child makes no kqueue call, and one made in the child of a process whose sandbox refuses the
        // call would be refused (and logged) again: most children exec first and need none.
        let code = in_child(30, || {
            prepare();
            if ANON_KQ.load(Ordering::SeqCst) < 0 || WL_KQ.load(Ordering::SeqCst) < 0 {
                return 3;
            }
            let child = in_child(20, || if ANON_KQ.load(Ordering::SeqCst) == NOT_MADE && WL_KQ.load(Ordering::SeqCst) == NOT_MADE { 0 } else { 4 });
            if child != 0 {
                return child;
            }
            let c = refuse_kqueue_from_now_on();
            if c != 0 {
                return c;
            }
            in_child(20, || {
                // First use: refused, and kept so; a child of this one keeps the refusal and is not refused again.
                if kept_kq(&WL_KQ) != Err(NO_KQUEUE) || WL_KQ.load(Ordering::SeqCst) != REFUSED {
                    return 5;
                }
                in_child(20, || if WL_KQ.load(Ordering::SeqCst) == REFUSED { 0 } else { 6 })
            })
        });
        assert_eq!(code, 0, "20/21 no reading (sandbox), 3 not made before the program, 4 the fork child made its kqueues at once, 5 the first use not refused, 6 the refusal not kept in a child");
    }

    /// The workloop monitor, started on `fd` in place of the shared kqueue.
    fn start_wl_monitor_on(fd: i32) {
        WL_KQ.store(fd, Ordering::SeqCst);
        crate::forksync::spawn_host("test-monitor", 2 << 20, Box::new(wl_monitor)).unwrap();
    }

    const NOTE_TRIGGER: u32 = 0x0100_0000;
    const EVFILT_MACHPORT: i16 = -8;

    /// A change on workloop `wl` through the global state, as kevent_id applies it.
    fn apply(wl: u64, ke: Kev) -> i64 {
        with_wl(|st| wl_apply(st, wl, &ke))
    }

    /// Mark workloop `wl` as having a servicer, so its events stay staged: a test starts no guest thread.
    fn hold(wl: u64) {
        with_wl(|st| st.loops.entry(wl).or_default().bound = true);
    }

    fn staged(wl: u64) -> usize {
        with_wl(|st| st.loops.get(&wl).map_or(0, |w| w.staged.len()))
    }

    /// Wait up to `ms` for workloop `wl` to have `want` staged events; how many it has then.
    fn wait_staged(wl: u64, want: usize, ms: u64) -> usize {
        let t0 = std::time::Instant::now();
        loop {
            let n = staged(wl);
            if n == want || t0.elapsed() > std::time::Duration::from_millis(ms) {
                return n;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// Poll workloop `wl` as a guest's kevent_id does (at most 4 events), with a data area when `data` is given.
    fn poll(wl: u64, data: Option<&mut [u8]>) -> Vec<Kev> {
        let mut out = [0u8; 72 * 4];
        let (d, mut avail, f) = match data {
            Some(b) => (b.as_mut_ptr() as u64, b.len() as u64, KEVENT_FLAG_IMMEDIATE | KEVENT_FLAG_STACK_DATA),
            None => (0, 0, KEVENT_FLAG_IMMEDIATE),
        };
        let r = kevent_id(&[wl, 0, 0, out.as_mut_ptr() as u64, 4, d, if d != 0 { &mut avail as *mut u64 as u64 } else { 0 }, f]);
        assert!(!r.carry, "poll: {r:?}");
        (0..r.rax).map(|i| unsafe { Kev::read(out.as_ptr() as u64 + i * KEV_SIZE) }).collect()
    }

    #[test]
    fn under_a_sandbox_that_refuses_kqueue_workloops_made_after_it_get_their_knotes_and_events() {
        // The kqueues were made before the sandbox (as main makes them before the guest starts), so after it no kqueue
        // call is needed: one would be refused, and the registration answered ENOMEM.
        let code = in_child(30, || {
            prepare();
            let c = refuse_kqueue_from_now_on();
            if c != 0 {
                return c;
            }
            let (w1, w2, w3) = (0x7a01, 0x7a02, 0x7a03);
            for w in [w1, w2, w3] {
                hold(w);
            }
            // One ident and filter on two workloops, as on kqueues of their own; and an EV_DISPATCH2 knote.
            let user = |udata: u64, flags: u16| Kev { ident: 0x51, filter: EVFILT_USER_, flags, udata, ..Default::default() };
            if apply(w1, user(0xa1, EV_ADD | EV_CLEAR)) != 0 || apply(w2, user(0xa2, EV_ADD | EV_CLEAR)) != 0 {
                return 3;
            }
            let d2 = Kev { ident: 0x52, filter: EVFILT_USER_, flags: EV_ADD | EV_CLEAR | EV_DISPATCH | EV_UDATA_SPECIFIC, udata: 0xa3, ..Default::default() };
            if apply(w3, d2) != 0 {
                return 4;
            }
            let wq = kevent_redirect(SYS_KEVENT_QOS, &[u64::MAX, 0, 0, 0, 0, 0, 0, KEVENT_FLAG_WORKQ | KEVENT_FLAG_IMMEDIATE]);
            if !wq.is_some_and(|r| !r.carry) {
                return 5;
            }
            // Fire w2's knote: only w2 gets an event, with its own udata.
            if apply(w2, Kev { fflags: NOTE_TRIGGER, ..user(0xa2, 0) }) != 0 {
                return 6;
            }
            if wait_staged(w2, 1, 3000) != 1 || staged(w1) != 0 {
                return 7;
            }
            let got = poll(w2, None);
            if got.len() != 1 || got[0].udata != 0xa2 || got[0].ident != 0x51 || got[0].flags & EV_UDATA_SPECIFIC != 0 || !poll(w1, None).is_empty() {
                return 8;
            }
            // An EV_DISPATCH2 knote whose event waits is deleted at once (natively it is still queued, so not disabled
            // and its deletion not deferred), and its event goes with it.
            if apply(w3, Kev { fflags: NOTE_TRIGGER, flags: EV_UDATA_SPECIFIC, ..d2 }) != 0 || wait_staged(w3, 1, 3000) != 1 {
                return 9;
            }
            let del = apply(w3, Kev { flags: EV_DELETE | EV_UDATA_SPECIFIC, ..d2 });
            if del != 0 || staged(w3) != 0 {
                return 10;
            }
            if !fd_open(1) {
                return 11;
            }
            0
        });
        assert_eq!(code, 0, "20/21 no reading (sandbox), 3 the shared ident refused, 4 the EV_DISPATCH2 knote refused, 5 the workq kevent refused, 6 the trigger refused, 7 the event not staged on its own workloop, 8 not delivered as the guest's, 9 the second trigger, 10 the waiting knote's deletion not immediate, 11 stdout closed");
    }

    #[test]
    fn a_level_triggered_knote_is_taken_once_until_its_event_is_delivered() {
        // A readable pipe stays readable: taken again and again, the monitor would spin and stage copies.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"x".as_ptr().cast(), 1) } != 1 {
                return 20;
            }
            let w = 0x7b01;
            hold(w);
            if apply(w, Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD, udata: 0xb1, ..Default::default() }) != 0 {
                return 3;
            }
            if wait_staged(w, 1, 3000) != 1 {
                return 4;
            }
            let before = cpu_ms();
            std::thread::sleep(std::time::Duration::from_millis(300));
            if staged(w) != 1 || cpu_ms() - before > 100 {
                return 5;
            }
            let got = poll(w, None);
            if got.len() != 1 || got[0].udata != 0xb1 || got[0].data != 1 {
                return 6;
            }
            // Delivered, it is armed again, and as the pipe is still readable its event comes again.
            if wait_staged(w, 1, 3000) != 1 {
                return 7;
            }
            0
        });
        assert_eq!(code, 0, "3 refused, 4 not staged, 5 taken more than once (or the monitor spun), 6 not delivered as the guest's, 7 not armed again after delivery");
    }

    extern "C" {
        static mach_task_self_: u32;
        fn mach_port_allocate(task: u32, right: u32, name: *mut u32) -> i32;
        fn mach_port_insert_right(task: u32, name: u32, poly: u32, poly_poly: u32) -> i32;
        fn mach_msg(msg: *mut u8, option: i32, send_size: u32, rcv_size: u32, rcv_name: u32, timeout: u32, notify: u32) -> i32;
    }

    #[test]
    fn a_message_received_as_its_event_was_taken_is_delivered_in_the_data_area() {
        // EVFILT_MACHPORT with MACH_RCV_MSG and no buffer of its own (ext[0] and ext[1] 0): the kernel receives the message
        // into the data area as the monitor takes the event, and a servicer or poll must find it in its own, at ext[0].
        let code = in_child(30, || unsafe {
            let task = mach_task_self_;
            let mut port = 0u32;
            if mach_port_allocate(task, 1, &mut port) != 0 || mach_port_insert_right(task, port, port, 20) != 0 {
                return 20;
            }
            let w = 0x7c01;
            hold(w);
            const MACH_RCV_MSG: u32 = 0x2;
            const MACH_RCV_LARGE: u32 = 0x4;
            let ke = Kev { ident: port as u64, filter: EVFILT_MACHPORT, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC, udata: 0xc1, fflags: MACH_RCV_MSG | MACH_RCV_LARGE, ..Default::default() };
            if apply(w, ke) != 0 {
                return 3;
            }
            // A 32-byte message: header (bits COPY_SEND, size, remote, local, voucher, id) and 8 bytes of body.
            let mut m = [0u8; 32];
            let put = |m: &mut [u8; 32], o: usize, v: u32| m[o..o + 4].copy_from_slice(&v.to_le_bytes());
            put(&mut m, 0, 19);
            put(&mut m, 4, 32);
            put(&mut m, 8, port);
            put(&mut m, 20, 0x6b71);
            m[24..].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
            if mach_msg(m.as_mut_ptr(), 1, 32, 0, 0, 0, 0) != 0 {
                return 21;
            }
            if wait_staged(w, 1, 3000) != 1 {
                return 4;
            }
            let mut area = vec![0u8; 0x8000];
            let (lo, hi) = (area.as_ptr() as u64, area.as_ptr() as u64 + area.len() as u64);
            let got = poll(w, Some(&mut area));
            if got.len() != 1 || got[0].udata != 0xc1 || got[0].fflags != 0 {
                return 5;
            }
            let at = got[0].ext[0];
            if at < lo || at + 32 > hi {
                return 6;
            }
            let rd32 = |o: u64| std::ptr::read_unaligned((at + o) as *const u32);
            if rd32(20) != 0x6b71 || std::ptr::read_unaligned((at + 24) as *const u64) != 0x1122_3344_5566_7788 {
                return 7;
            }
            0
        });
        assert_eq!(code, 0, "20/21 the test's port or send failed, 3 refused, 4 not staged, 5 not delivered as the guest's, 6 ext[0] not in the poll's data area, 7 not the message sent");
    }

    extern "C" {
        fn mach_port_get_refs(task: u32, name: u32, right: u32, refs: *mut u32) -> i32;
    }

    const MACH_RCV_MSG_LARGE: u32 = 0x2 | 0x4;

    /// A new port with its receive right and a send right (0 if the kernel would not make one).
    unsafe fn new_port() -> u32 {
        let mut port = 0u32;
        if mach_port_allocate(mach_task_self_, 1, &mut port) != 0 || mach_port_insert_right(mach_task_self_, port, port, 20) != 0 {
            return 0;
        }
        port
    }

    /// Send `port` a 32-byte message with `body`, and, unless `reply` is 0, a send right to `reply` made as it is sent.
    unsafe fn send(port: u32, reply: u32, body: u64) -> i32 {
        let mut m = [0u8; 32];
        let put = |m: &mut [u8; 32], o: usize, v: u32| m[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut m, 0, 19 | if reply != 0 { 20 << 8 } else { 0 });
        put(&mut m, 4, 32);
        put(&mut m, 8, port);
        put(&mut m, 12, reply);
        put(&mut m, 20, 0x6b72);
        m[24..].copy_from_slice(&body.to_le_bytes());
        mach_msg(m.as_mut_ptr(), 1, 32, 0, 0, 0, 0)
    }

    unsafe fn send_refs(port: u32) -> u32 {
        let mut n = u32::MAX;
        mach_port_get_refs(mach_task_self_, port, 0, &mut n);
        n
    }

    #[test]
    fn a_message_taken_for_a_knote_deleted_before_its_delivery_is_released() {
        // Natively it would still be on its port, and go with it: the send right it carries must not stay in the task.
        let code = in_child(30, || unsafe {
            let (port, reply) = (new_port(), new_port());
            if port == 0 || reply == 0 || mach_port_get_refs(mach_task_self_, reply, 0, &mut 0) != 0 {
                return 20;
            }
            // The reply port's own send right, made by new_port, is taken back so that only the message's remains.
            extern "C" {
                fn mach_port_mod_refs(task: u32, name: u32, right: u32, delta: i32) -> i32;
            }
            if mach_port_mod_refs(mach_task_self_, reply, 0, -1) != 0 || send_refs(reply) != 0 {
                return 21;
            }
            let w = 0x7c02;
            hold(w);
            let ke = Kev { ident: port as u64, filter: EVFILT_MACHPORT, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC, udata: 0xc2, fflags: MACH_RCV_MSG_LARGE, ..Default::default() };
            if apply(w, ke) != 0 {
                return 3;
            }
            if send(port, reply, 1) != 0 {
                return 22;
            }
            if wait_staged(w, 1, 3000) != 1 {
                return 4;
            }
            // Control: received as its event was taken, the message's send right is in the task.
            if send_refs(reply) != 1 {
                return 5;
            }
            if apply(w, Kev { flags: EV_DELETE | EV_UDATA_SPECIFIC, ..ke }) != 0 || staged(w) != 0 {
                return 6;
            }
            if send_refs(reply) != 0 {
                return 7;
            }
            0
        });
        assert_eq!(code, 0, "20-22 the test's ports or send failed, 3 refused, 4 not staged, 5 the message's right not received (no reading), 6 the deletion, 7 the right was kept");
    }

    #[test]
    fn a_poll_without_stack_data_fills_its_data_area_from_the_bottom() {
        // As libdispatch's drain of a workloop polls: a data area and its size, without KEVENT_FLAG_STACK_DATA.
        let code = in_child(30, || unsafe {
            let p = new_port();
            if p == 0 {
                return 20;
            }
            let w = 0x7c04;
            hold(w);
            let ke = Kev { ident: p as u64, filter: EVFILT_MACHPORT, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC, udata: 0xc4, fflags: MACH_RCV_MSG_LARGE, ..Default::default() };
            if apply(w, ke) != 0 || send(p, 0, 0x4747) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            let mut area = vec![0u8; 0x8000];
            let lo = area.as_ptr() as u64;
            let mut out = [0u8; 72 * 4];
            let mut avail = area.len() as u64;
            let r = kevent_id(&[w, 0, 0, out.as_mut_ptr() as u64, 4, lo, &mut avail as *mut u64 as u64, KEVENT_FLAG_IMMEDIATE]);
            if r.carry || r.rax != 1 {
                return 4;
            }
            let e = Kev::read(out.as_ptr() as u64);
            if e.ext[0] < lo || e.ext[0] >= lo + 64 || avail >= area.len() as u64 || std::ptr::read_unaligned((e.ext[0] + 24) as *const u64) != 0x4747 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "20 the test's port, 3 not staged, 4 not delivered, 5 not placed at the bottom of the area");
    }

    #[test]
    fn two_messages_delivered_at_once_each_have_their_own_place_in_the_data_area() {
        let code = in_child(30, || unsafe {
            let ports = [new_port(), new_port()];
            if ports.contains(&0) {
                return 20;
            }
            let w = 0x7c03;
            hold(w);
            for (i, p) in ports.iter().enumerate() {
                let ke = Kev { ident: *p as u64, filter: EVFILT_MACHPORT, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC, udata: 0xd0 + i as u64, fflags: MACH_RCV_MSG_LARGE, ..Default::default() };
                if apply(w, ke) != 0 {
                    return 3;
                }
                if send(*p, 0, 0x1111 * (i as u64 + 1)) != 0 {
                    return 21;
                }
            }
            if wait_staged(w, 2, 3000) != 2 {
                return 4;
            }
            let mut area = vec![0u8; 0x8000];
            let got = poll(w, Some(&mut area));
            if got.len() != 2 {
                return 5;
            }
            for e in &got {
                let i = e.udata.wrapping_sub(0xd0);
                if i > 1 || std::ptr::read_unaligned((e.ext[0] + 24) as *const u64) != 0x1111 * (i + 1) {
                    return 6;
                }
            }
            if got[0].ext[0].abs_diff(got[1].ext[0]) < 32 {
                return 7;
            }
            0
        });
        assert_eq!(code, 0, "20/21 the test's ports or send failed, 3 refused, 4 not staged, 5 not both delivered, 6 a message not its own, 7 the two overlap");
    }

    fn user(ident: u64, flags: u16, udata: u64) -> Kev {
        Kev { ident, filter: EVFILT_USER_, flags, udata, ..Default::default() }
    }

    fn trigger(k: Kev) -> Kev {
        Kev { fflags: NOTE_TRIGGER, flags: k.flags & EV_UDATA_SPECIFIC, ..k }
    }

    /// Deliver the one staged event of held workloop `wl`, once it is staged; its flags, or None.
    fn deliver_one(wl: u64) -> Option<u16> {
        if wait_staged(wl, 1, 3000) != 1 {
            return None;
        }
        let got = poll(wl, None);
        (got.len() == 1).then(|| got[0].flags)
    }

    #[test]
    fn a_delivered_dispatch2_knote_is_deleted_as_the_kernel_deletes_it() {
        // Measured natively (a plain kqueue): delivered, an EV_DISPATCH2 knote is disabled; EV_DELETE|EV_ENABLE drops it at
        // once (0); a bare EV_DELETE is deferred (EINPROGRESS), and once enabled it gives its deletion as an event.
        let code = in_child(30, || {
            let w = 0x7d01;
            hold(w);
            for oneshot in [0, EV_ONESHOT] {
                let d2 = EV_CLEAR | EV_DISPATCH | EV_UDATA_SPECIFIC | oneshot;
                let a = user(0x61 + oneshot as u64, EV_ADD | d2, 0xe1);
                if apply(w, a) != 0 || apply(w, trigger(a)) != 0 {
                    return 3;
                }
                // Its flags as the guest attached it (measured natively: 0x1b1 with EV_ONESHOT).
                match deliver_one(w) {
                    Some(f) if f == EV_ADD | d2 => {}
                    Some(_) => return 10,
                    None => return 3,
                }
                if apply(w, Kev { flags: EV_DELETE | EV_ENABLE | EV_UDATA_SPECIFIC, ..a }) != 0 {
                    return 4;
                }
                let b = user(0x71 + oneshot as u64, EV_ADD | d2, 0xe2);
                if apply(w, b) != 0 || apply(w, trigger(b)) != 0 || deliver_one(w).is_none() {
                    return 5;
                }
                if apply(w, Kev { flags: EV_DELETE | EV_UDATA_SPECIFIC, ..b }) != libc::EINPROGRESS as i64 {
                    return 6;
                }
                if apply(w, Kev { flags: EV_ENABLE | EV_UDATA_SPECIFIC, ..b }) != 0 || wait_staged(w, 1, 3000) != 1 {
                    return 7;
                }
                // Enabled again before its deletion is delivered: natively a no-op, and the deletion still comes, once.
                if apply(w, Kev { flags: EV_ENABLE | EV_UDATA_SPECIFIC, ..b }) != 0 {
                    return 11;
                }
                match deliver_one(w) {
                    Some(f) if f & EV_DELETE != 0 => {}
                    _ => return 8,
                }
                if apply(w, Kev { flags: EV_DELETE | EV_UDATA_SPECIFIC, ..b }) != libc::ENOENT as i64 {
                    return 9;
                }
            }
            0
        });
        assert_eq!(code, 0, "3/5 not delivered, 4 EV_DELETE|EV_ENABLE not 0, 6 a bare EV_DELETE not EINPROGRESS, 7 the enable refused or no deletion came, 8 its deletion not delivered, 9 the knote still there, 10 delivered with other flags than it was attached with, 11 a second enable refused");
    }

    /// While holding the workloop lock: `fire` (false if it failed), wait for the monitor to take the event that makes
    /// (it then waits for the lock), and apply `changes` to workloop `wl`, as guest changes that come as the event is
    /// taken; then let the monitor stage it. Their answers, or None (no reading) if the monitor took no event in 3 s.
    fn changes_as_it_is_taken(wl: u64, fire: impl FnOnce(&mut WlState) -> bool, changes: &[Kev]) -> Option<Vec<i64>> {
        let mut g = WL.lock().unwrap();
        let st = g.get_or_insert_with(WlState::new);
        let before = TAKEN.load(Ordering::SeqCst);
        if !fire(st) {
            return Some(vec![-1]);
        }
        let t0 = std::time::Instant::now();
        while TAKEN.load(Ordering::SeqCst) == before {
            if t0.elapsed() > std::time::Duration::from_secs(3) {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let r = changes.iter().map(|c| wl_apply(st, wl, c)).collect();
        // Let the monitor have the lock and stage what it took before the caller takes the lock again.
        drop(g);
        std::thread::sleep(std::time::Duration::from_millis(100));
        Some(r)
    }

    /// changes_as_it_is_taken with one change; `fire` 0 triggers `k` (EVFILT_USER), otherwise it is a pipe's write end,
    /// written to.
    fn change_as_it_is_taken(wl: u64, k: Kev, fire: i32, change: Kev) -> Option<i64> {
        let f = |st: &mut WlState| if fire == 0 { wl_apply(st, wl, &trigger(k)) == 0 } else { (unsafe { libc::write(fire, b"x".as_ptr().cast(), 1) }) == 1 };
        changes_as_it_is_taken(wl, f, &[change]).map(|v| v[0])
    }

    #[test]
    fn a_change_that_comes_as_a_knotes_event_is_taken_is_answered_as_natively() {
        // Natively the event is still queued, the knote enabled with it.
        let code = in_child(30, || {
            let w = 0x7e01;
            hold(w);
            // An EV_DISPATCH2 knote and an EV_ONESHOT one deleted as their events are taken: 0, and no event comes.
            let d2 = user(0x81, EV_ADD | EV_CLEAR | EV_DISPATCH | EV_UDATA_SPECIFIC, 0xf1);
            if apply(w, d2) != 0 {
                return 3;
            }
            match change_as_it_is_taken(w, d2, 0, Kev { flags: EV_DELETE | EV_UDATA_SPECIFIC, ..d2 }) {
                Some(0) => {}
                None => return 21,
                Some(_) => return 4,
            }
            let os = user(0x82, EV_ADD | EV_CLEAR | EV_ONESHOT, 0xf2);
            if apply(w, os) != 0 {
                return 3;
            }
            match change_as_it_is_taken(w, os, 0, Kev { flags: EV_DELETE, ..os }) {
                Some(0) => {}
                None => return 22,
                Some(_) => return 5,
            }
            if wait_staged(w, 1, 300) != 0 {
                return 6;
            }
            // An enabled EV_DISPATCH knote on a readable pipe enabled again (EV_ENABLE, or EV_ADD|EV_ENABLE) as its event
            // is taken: still one event, as the host knote stays disabled until it is delivered.
            for again in [EV_ENABLE, EV_ADD | EV_ENABLE | EV_DISPATCH] {
                let mut p = [-1i32; 2];
                if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                    return 20;
                }
                let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH, udata: 0xf3, ..Default::default() };
                if apply(w, r) != 0 {
                    return 3;
                }
                match change_as_it_is_taken(w, r, p[1], Kev { flags: again, ..r }) {
                    Some(0) => {}
                    None => return 23,
                    Some(_) => return 7,
                }
                std::thread::sleep(std::time::Duration::from_millis(300));
                if staged(w) != 1 || poll(w, None).len() != 1 {
                    return 8;
                }
                if apply(w, Kev { flags: EV_DELETE, ..r }) != 0 {
                    return 9;
                }
            }
            0
        });
        assert_eq!(code, 0, "3 refused, 4 the EV_DISPATCH2 knote's deletion not 0, 5 the EV_ONESHOT knote's not 0, 6 a deleted knote's event came, 7 the enable refused, 8 not one event, 9 the pipe knote's deletion; 20 no pipe, 21-23 no reading (the monitor had not taken the event)");
    }

    #[test]
    fn a_oneshot_timer_programmed_again_as_it_fires_gives_the_new_fire_only() {
        // Measured natively: programmed again while its fire is queued, a oneshot timer's fire is discarded and the new
        // one comes at the new time. libdispatch programs its timers so.
        let code = in_child(30, || {
            let w = 0x7f01;
            hold(w);
            let t = Kev { ident: 0x91, filter: EVFILT_TIMER, flags: EV_ADD | EV_ENABLE | EV_ONESHOT, data: 10, udata: 0x101, ..Default::default() };
            if apply(w, t) != 0 {
                return 3;
            }
            let before = TAKEN.load(Ordering::SeqCst);
            let mut g = WL.lock().unwrap();
            let t1 = std::time::Instant::now();
            while TAKEN.load(Ordering::SeqCst) == before {
                if t1.elapsed() > std::time::Duration::from_secs(3) {
                    return 22;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            let t0 = std::time::Instant::now();
            let e = wl_apply(g.as_mut().unwrap(), w, &Kev { data: 300, ..t });
            drop(g);
            if e != 0 {
                return 4;
            }
            if wait_staged(w, 1, 3000) != 1 {
                return 5;
            }
            if t0.elapsed() < std::time::Duration::from_millis(250) {
                return 6;
            }
            let got = poll(w, None);
            if got.len() != 1 || got[0].udata != 0x101 {
                return 7;
            }
            if wait_staged(w, 1, 500) != 0 {
                return 8;
            }
            0
        });
        assert_eq!(code, 0, "22 no reading (the monitor took no fire), 3/4 refused, 5 no fire came, 6 the old fire was delivered, 7 not delivered as the guest's, 8 a second fire");
    }

    #[test]
    fn a_knote_added_again_keeps_its_queued_event() {
        // Measured natively: an EV_CLEAR knote triggered and added again keeps its queued event.
        let code = in_child(30, || {
            let w = 0x7f02;
            hold(w);
            let u = user(0xa1, EV_ADD | EV_CLEAR, 0x102);
            if apply(w, u) != 0 || apply(w, trigger(u)) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            if apply(w, u) != 0 {
                return 4;
            }
            if staged(w) != 1 || poll(w, None).len() != 1 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "3 not staged, 4 refused, 5 its event not kept");
    }

    #[test]
    fn a_plain_message_knote_keeps_its_receive_options() {
        // The emulation's own disable and enable of a knote are changes the kernel gives to the filter: across them a
        // port's knote must still receive each message with its event.
        let code = in_child(30, || unsafe {
            let p = new_port();
            if p == 0 {
                return 20;
            }
            let w = 0x7f03;
            hold(w);
            let ke = Kev { ident: p as u64, filter: EVFILT_MACHPORT, flags: EV_ADD | EV_UDATA_SPECIFIC, udata: 0x103, fflags: MACH_RCV_MSG_LARGE, ..Default::default() };
            if apply(w, ke) != 0 {
                return 3;
            }
            for body in [0x5151u64, 0x6262] {
                if send(p, 0, body) != 0 {
                    return 21;
                }
                if wait_staged(w, 1, 3000) != 1 {
                    return 4;
                }
                let mut area = vec![0u8; 0x8000];
                let (lo, hi) = (area.as_ptr() as u64, area.as_ptr() as u64 + area.len() as u64);
                let got = poll(w, Some(&mut area));
                if got.len() != 1 || got[0].fflags != 0 || got[0].ext[0] < lo || got[0].ext[0] + 32 > hi {
                    return 5;
                }
                if std::ptr::read_unaligned((got[0].ext[0] + 24) as *const u64) != body {
                    return 6;
                }
            }
            0
        });
        assert_eq!(code, 0, "20/21 the test's port or send failed, 3 refused, 4 not staged, 5 a message not received with its event, 6 not the message sent");
    }

    #[test]
    fn a_repeating_timer_keeps_its_interval() {
        // A 50 ms timer gives about 8 events in 400 ms; one given its interval again as 0 gives one at each poll.
        let code = in_child(30, || {
            let w = 0x7f04;
            hold(w);
            if apply(w, Kev { ident: 0x92, filter: EVFILT_TIMER, flags: EV_ADD | EV_ENABLE, data: 50, udata: 0x104, ..Default::default() }) != 0 {
                return 3;
            }
            let t0 = std::time::Instant::now();
            let mut n = 0;
            while t0.elapsed() < std::time::Duration::from_millis(400) {
                n += poll(w, None).len();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            if n < 2 {
                return 4;
            }
            if n > 12 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "3 refused, 4 fewer than two events, 5 more events than its interval gives");
    }

    #[test]
    fn a_read_knote_keeps_its_low_water_mark() {
        // The emulation's own disable and enable of a level-triggered knote are changes the kernel gives to the filter:
        // given with no fflags or data, a read knote would lose NOTE_LOWAT and fire for fewer bytes than it asks.
        const NOTE_LOWAT: u32 = 1;
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"abcd".as_ptr().cast(), 4) } != 4 {
                return 20;
            }
            let w = 0x7f0e;
            hold(w);
            if apply(w, Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD, fflags: NOTE_LOWAT, data: 4, udata: 0x10e, ..Default::default() }) != 0 {
                return 3;
            }
            if wait_staged(w, 1, 3000) != 1 {
                return 4;
            }
            // Three of the four bytes read while its event waits: delivered, the knote is armed again with one byte
            // left, under its mark, so no event comes until there are four again.
            let mut b = [0u8; 3];
            if unsafe { libc::read(p[0], b.as_mut_ptr().cast(), 3) } != 3 {
                return 21;
            }
            if poll(w, None).len() != 1 {
                return 5;
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            if staged(w) != 0 {
                return 6;
            }
            if unsafe { libc::write(p[1], b"efg".as_ptr().cast(), 3) } != 3 {
                return 22;
            }
            if wait_staged(w, 1, 3000) != 1 {
                return 7;
            }
            0
        });
        assert_eq!(code, 0, "3 refused, 4 not staged, 5 not delivered, 6 an event for fewer bytes than its mark, 7 none at its mark; 20-22 the test's pipe");
    }

    #[test]
    fn a_deferred_deletion_delivers_the_deletion_only() {
        // A message knote disabled, deleted and enabled as its message is being taken: natively it delivers its deletion,
        // never the message.
        let code = in_child(30, || unsafe {
            let p = new_port();
            if p == 0 {
                return 20;
            }
            let w = 0x7f05;
            hold(w);
            let ke = Kev { ident: p as u64, filter: EVFILT_MACHPORT, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC, udata: 0x105, fflags: MACH_RCV_MSG_LARGE, ..Default::default() };
            if apply(w, ke) != 0 {
                return 3;
            }
            let changes = [EV_DISABLE, EV_DELETE, EV_ENABLE].map(|f| Kev { flags: f | EV_UDATA_SPECIFIC, ..ke });
            match changes_as_it_is_taken(w, |_| send(p, 0, 7) == 0, &changes) {
                Some(r) if r == [0, libc::EINPROGRESS as i64, 0] => {}
                None => return 22,
                Some(_) => return 4,
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
            let mut area = vec![0u8; 0x8000];
            let got = poll(w, Some(&mut area));
            if got.len() != 1 || got[0].flags & EV_DELETE == 0 || !poll(w, Some(&mut area)).is_empty() {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "20 the test's port, 22 no reading (the monitor took no event), 3 refused, 4 the changes not answered 0, EINPROGRESS, 0, 5 not its deletion alone");
    }

    #[test]
    fn a_vanished_knote_stays_until_the_guest_deletes_it() {
        // An EV_DISPATCH2 knote that asked for EV_VANISHED, its descriptor closed: the kernel gives the vanishing as an
        // EV_ONESHOT event and defers the knote's deletion, which libdispatch then makes.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                return 20;
            }
            let w = 0x7f06;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC | EV_VANISHED, udata: 0x106, ..Default::default() };
            if apply(w, r) != 0 {
                return 3;
            }
            unsafe { libc::close(p[0]) };
            match deliver_one(w) {
                Some(f) if f & EV_VANISHED != 0 => {}
                _ => return 4,
            }
            if apply(w, Kev { flags: EV_DELETE | EV_ENABLE | EV_UDATA_SPECIFIC, ..r }) != 0 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 3 refused, 4 no vanished event, 5 its deletion not 0");
    }

    #[test]
    fn an_ev_add_of_a_dispatch_knote_keeps_it_dispatch() {
        // The kernel keeps the flags a knote was attached with: EV_ADD alone later is a change of its filter's values.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"x".as_ptr().cast(), 1) } != 1 {
                return 20;
            }
            let w = 0x7f07;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH, udata: 0x107, ..Default::default() };
            if apply(w, r) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            if apply(w, Kev { flags: EV_ADD, ..r }) != 0 || poll(w, None).len() != 1 {
                return 4;
            }
            // Delivered, it is disabled until the guest enables it, though the pipe is still readable.
            if wait_staged(w, 1, 300) != 0 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 3 not staged, 4 not delivered, 5 enabled again after delivery");
    }

    #[test]
    fn a_disabled_and_enabled_knote_keeps_an_activation_its_event_used() {
        // An EV_CLEAR knote's activation is used up as its event is taken: natively the disable only dequeues the event,
        // and the enable queues it again. A plain oneshot knote enabled while its event waits is still there.
        let code = in_child(30, || {
            let w = 0x7f08;
            hold(w);
            let c = user(0xb1, EV_ADD | EV_CLEAR, 0x108);
            if apply(w, c) != 0 || apply(w, trigger(c)) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            if apply(w, Kev { flags: EV_DISABLE, ..c }) != 0 || apply(w, Kev { flags: EV_ENABLE, ..c }) != 0 || poll(w, None).len() != 1 {
                return 4;
            }
            let o = user(0xb2, EV_ADD | EV_CLEAR | EV_ONESHOT, 0x109);
            if apply(w, o) != 0 || apply(w, trigger(o)) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            if apply(w, Kev { flags: EV_ENABLE, ..o }) != 0 || poll(w, None).len() != 1 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "3 not staged, 4 the activation lost, 5 the enable of a waiting oneshot knote refused or its event lost");
    }

    #[test]
    fn a_dispatch_knote_disabled_and_enabled_as_its_event_is_taken_gives_one_event() {
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                return 20;
            }
            let w = 0x7f09;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH, udata: 0x10a, ..Default::default() };
            if apply(w, r) != 0 {
                return 3;
            }
            let fd = p[1];
            match changes_as_it_is_taken(w, |_| (unsafe { libc::write(fd, b"x".as_ptr().cast(), 1) }) == 1, &[Kev { flags: EV_DISABLE, ..r }, Kev { flags: EV_ENABLE, ..r }]) {
                Some(v) if v == [0, 0] => {}
                None => return 22,
                Some(_) => return 4,
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            if staged(w) != 1 || poll(w, None).len() != 1 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 22 no reading (the monitor took no event), 3 refused, 4 a change refused, 5 not one event");
    }

    #[test]
    fn a_timer_given_values_the_host_refuses_is_left_as_it_was() {
        let code = in_child(30, || {
            let w = 0x7f0a;
            hold(w);
            let t = Kev { ident: 0x93, filter: EVFILT_TIMER, flags: EV_ADD | EV_ENABLE | EV_ONESHOT, data: 300, udata: 0x10b, ..Default::default() };
            if apply(w, t) != 0 {
                return 3;
            }
            // Seconds and nanoseconds at once: refused.
            if apply(w, Kev { fflags: 0x1 | 0x4, ..t }) != libc::EINVAL as i64 {
                return 4;
            }
            if wait_staged(w, 1, 2000) != 1 || poll(w, None).len() != 1 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "3 refused, 4 the bad values not EINVAL, 5 the timer as it was did not fire");
    }

    #[test]
    fn a_oneshot_knote_added_again_while_its_event_waits_leaves_no_host_knote() {
        let code = in_child(30, || {
            let w = 0x7f0b;
            hold(w);
            let o = user(0xb3, EV_ADD | EV_CLEAR | EV_ONESHOT, 0x10c);
            if apply(w, o) != 0 || apply(w, trigger(o)) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            let token = with_wl(|st| st.by_key.get(&(w, EVFILT_USER_, 0xb3, None)).copied());
            let Some(token) = token else { return 4 };
            if apply(w, o) != 0 || poll(w, None).len() != 1 {
                return 5;
            }
            // Delivered, the knote is gone, and so is any host knote under its token.
            let host = host_change(WL_KQ.load(Ordering::SeqCst), &Kev { ident: 0xb3, filter: EVFILT_USER_, flags: EV_DELETE | EV_UDATA_SPECIFIC, udata: token, ..Default::default() });
            if host != libc::ENOENT as i64 {
                return 6;
            }
            0
        });
        assert_eq!(code, 0, "3 not staged, 4 no record, 5 not delivered once, 6 a host knote left");
    }

    #[test]
    fn a_change_without_ev_add_gives_a_knote_a_new_udata() {
        // Natively a knote not named by its udata takes the udata of every change.
        let code = in_child(30, || {
            let w = 0x7f0c;
            hold(w);
            let u = user(0xb4, EV_ADD | EV_CLEAR, 0xa);
            if apply(w, u) != 0 || apply(w, Kev { udata: 0xb, ..trigger(u) }) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            match poll(w, None).as_slice() {
                [e] if e.udata == 0xb => 0,
                _ => 4,
            }
        });
        assert_eq!(code, 0, "3 not staged, 4 not delivered with the new udata");
    }

    #[test]
    fn a_closed_descriptors_knotes_and_events_go_with_it() {
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"x".as_ptr().cast(), 1) } != 1 {
                return 20;
            }
            let w = 0x7f0d;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD, udata: 0x10d, ..Default::default() };
            if apply(w, r) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            let tokens = fd_closing(p[0]);
            unsafe { libc::close(p[0]) };
            fd_closed(&tokens);
            if staged(w) != 0 || !poll(w, None).is_empty() || apply(w, Kev { flags: EV_DELETE, ..r }) != libc::ENOENT as i64 {
                return 4;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 3 not staged, 4 its record or event left");
    }

    #[test]
    fn an_edge_knote_added_again_or_disabled_and_enabled_while_its_event_waits_fires_again_later() {
        // Its activation was used up as its event was taken: the host knote stays enabled, so a trigger before its event
        // is delivered is one event with it, and a later trigger gives an event, as natively.
        let code = in_child(30, || {
            let w = 0x7f10;
            hold(w);
            for (i, again) in [[EV_ADD | EV_CLEAR, 0], [EV_DISABLE, EV_ENABLE]].into_iter().enumerate() {
                let c = user(0xc1 + i as u64, EV_ADD | EV_CLEAR, 0x110);
                if apply(w, c) != 0 || apply(w, trigger(c)) != 0 || wait_staged(w, 1, 3000) != 1 {
                    return 3;
                }
                for f in again.into_iter().filter(|f| *f != 0) {
                    if apply(w, Kev { flags: f, ..c }) != 0 {
                        return 4;
                    }
                }
                if i == 0 {
                    // Triggered again before its event is delivered.
                    let before = TAKEN.load(Ordering::SeqCst);
                    if apply(w, trigger(c)) != 0 {
                        return 4;
                    }
                    let t0 = std::time::Instant::now();
                    while TAKEN.load(Ordering::SeqCst) == before && t0.elapsed() < std::time::Duration::from_secs(1) {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                if poll(w, None).len() != 1 {
                    return 5;
                }
                if i == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    if staged(w) != 0 {
                        return 8;
                    }
                }
                if apply(w, trigger(c)) != 0 || wait_staged(w, 1, 3000) != 1 || poll(w, None).len() != 1 {
                    return 6 + i as i32;
                }
            }
            0
        });
        assert_eq!(code, 0, "3 not staged, 4 refused, 5 not delivered, 6 no event after it was added again, 7 none after disable and enable, 8 a trigger while its event waited came as an event of its own");
    }

    #[test]
    fn a_registration_on_a_reused_descriptor_outlives_the_close_of_the_old_one() {
        // A close gives up the knotes it found before the descriptor was closed: one made on the number once it was free
        // again (another thread's) is not one of them.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                return 20;
            }
            let (w1, w2) = (0x7f11, 0x7f12);
            hold(w1);
            hold(w2);
            let r = |udata| Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD, udata, ..Default::default() };
            if apply(w1, r(0x111)) != 0 {
                return 3;
            }
            let tokens = fd_closing(p[0]);
            if tokens.len() != 1 || apply(w2, r(0x112)) != 0 {
                return 4;
            }
            fd_closed(&tokens);
            let left = with_wl(|st| (st.by_key.contains_key(&(w1, EVFILT_READ, p[0] as u64, None)), st.by_key.contains_key(&(w2, EVFILT_READ, p[0] as u64, None))));
            if left != (false, true) {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 3/4 refused, 5 the wrong knote given up");
    }

    #[test]
    fn a_vanished_knotes_deferred_deletion_is_delivered_as_a_deletion() {
        // The kernel gives a knote whose deletion is deferred its deletion, a vanishing not excepted.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                return 20;
            }
            let w = 0x7f13;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC | EV_VANISHED, udata: 0x113, ..Default::default() };
            if apply(w, r) != 0 {
                return 3;
            }
            unsafe { libc::close(p[0]) };
            if wait_staged(w, 1, 3000) != 1 {
                return 4;
            }
            let spec = |f| Kev { flags: f | EV_UDATA_SPECIFIC, ..r };
            let got = [apply(w, spec(EV_DISABLE)), apply(w, spec(EV_DELETE)), apply(w, spec(EV_ENABLE))];
            if got != [0, libc::EINPROGRESS as i64, 0] {
                return 5;
            }
            match deliver_one(w) {
                Some(f) if f & EV_DELETE != 0 => {}
                _ => return 6,
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 3 refused, 4 no vanishing staged, 5 not 0, EINPROGRESS, 0, 6 not delivered as a deletion");
    }

    #[test]
    fn a_disabled_knotes_staged_deletion_waits_for_the_enable() {
        let code = in_child(30, || {
            let w = 0x7f14;
            hold(w);
            let d2 = user(0xc4, EV_ADD | EV_CLEAR | EV_DISPATCH | EV_UDATA_SPECIFIC, 0x114);
            if apply(w, d2) != 0 || apply(w, trigger(d2)) != 0 || deliver_one(w).is_none() {
                return 3;
            }
            let spec = |f| Kev { flags: f | EV_UDATA_SPECIFIC, ..d2 };
            if apply(w, spec(EV_DELETE)) != libc::EINPROGRESS as i64 || apply(w, spec(EV_ENABLE)) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 4;
            }
            if apply(w, spec(EV_DISABLE)) != 0 || !poll(w, None).is_empty() {
                return 5;
            }
            if apply(w, spec(EV_ENABLE)) != 0 {
                return 6;
            }
            match deliver_one(w) {
                Some(f) if f & EV_DELETE != 0 => 0,
                _ => 7,
            }
        });
        assert_eq!(code, 0, "3 not delivered, 4 the deferred deletion not staged, 5 delivered while disabled, 6 the enable refused, 7 not delivered once enabled");
    }

    #[test]
    fn a_vanishing_knotes_waiting_event_becomes_its_vanishing_when_its_descriptor_is_closed() {
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"x".as_ptr().cast(), 1) } != 1 {
                return 20;
            }
            let w = 0x7f15;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH | EV_UDATA_SPECIFIC | EV_VANISHED, udata: 0x115, ..Default::default() };
            if apply(w, r) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            let tokens = fd_closing(p[0]);
            unsafe { libc::close(p[0]) };
            fd_closed(&tokens);
            match deliver_one(w) {
                Some(f) if f & EV_VANISHED != 0 => 0,
                Some(_) => 4,
                None => 5,
            }
        });
        assert_eq!(code, 0, "20 no pipe, 3 not staged, 4 the read delivered, 5 nothing delivered");
    }

    #[test]
    fn a_delivered_dispatch_knote_takes_no_event_the_guest_did_not_enable() {
        // A second fire of an EV_DISPATCH knote whose event was delivered (the host enabled by an enable that came as
        // the first was being taken) is not delivered: natively it gives one event until the guest enables it again.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"x".as_ptr().cast(), 1) } != 1 {
                return 20;
            }
            let w = 0x7f16;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_DISPATCH, udata: 0x116, ..Default::default() };
            if apply(w, r) != 0 || wait_staged(w, 1, 3000) != 1 || poll(w, None).len() != 1 {
                return 3;
            }
            let Some(token) = with_wl(|st| st.by_key.get(&(w, EVFILT_READ, p[0] as u64, None)).copied()) else { return 4 };
            let kq = WL_KQ.load(Ordering::SeqCst);
            let before = TAKEN.load(Ordering::SeqCst);
            if host_change(kq, &Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ENABLE, udata: token, ..Default::default() }) != 0 {
                return 5;
            }
            let t0 = std::time::Instant::now();
            while TAKEN.load(Ordering::SeqCst) == before {
                if t0.elapsed() > std::time::Duration::from_secs(3) {
                    return 22;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
            if staged(w) != 0 {
                return 6;
            }
            // Enabled by the guest, it gives its event again, the pipe being still readable.
            if apply(w, Kev { flags: EV_ENABLE, ..r }) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 7;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 22 no reading (the monitor took no second event), 3 not delivered, 4 no record, 5 the host refused, 6 a second event kept, 7 no event after the guest's enable");
    }

    #[test]
    fn a_timer_given_new_values_keeps_the_flags_it_was_attached_with() {
        let code = in_child(30, || {
            let w = 0x7f17;
            hold(w);
            // A oneshot timer given new values with EV_ADD alone is still oneshot.
            let t = Kev { ident: 0x94, filter: EVFILT_TIMER, flags: EV_ADD | EV_ONESHOT, data: 300, udata: 0x117, ..Default::default() };
            if apply(w, t) != 0 || apply(w, Kev { flags: EV_ADD, data: 30, ..t }) != 0 {
                return 3;
            }
            let t0 = std::time::Instant::now();
            let mut n = 0;
            while t0.elapsed() < std::time::Duration::from_millis(400) {
                n += poll(w, None).len();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            if n != 1 {
                return 4;
            }
            // A timer the host has one of per ident (EV_DISPATCH, not EV_UDATA_SPECIFIC) given new values: it fires.
            let p = Kev { ident: 0x95, filter: EVFILT_TIMER, flags: EV_ADD | EV_ONESHOT | EV_DISPATCH, data: 300, udata: 0x118, ..Default::default() };
            if apply(w, p) != 0 || apply(w, Kev { data: 50, ..p }) != 0 {
                return 5;
            }
            if wait_staged(w, 1, 2000) != 1 || poll(w, None).len() != 1 {
                return 6;
            }
            0
        });
        assert_eq!(code, 0, "3/5 refused, 4 not one fire (a repeating timer), 6 the timer did not fire");
    }

    #[test]
    fn a_delivered_oneshot_dispatch2_timer_given_new_values_delivers_its_deletion() {
        // Its deletion is deferred: natively the change goes to no filter, and EV_ENABLE in it delivers the deletion.
        let code = in_child(30, || {
            let w = 0x7f18;
            hold(w);
            let t = Kev { ident: 0x96, filter: EVFILT_TIMER, flags: EV_ADD | EV_ENABLE | EV_ONESHOT | EV_DISPATCH | EV_UDATA_SPECIFIC, data: 10, udata: 0x119, ..Default::default() };
            if apply(w, t) != 0 || deliver_one(w).is_none() {
                return 3;
            }
            if apply(w, Kev { data: 20, ..t }) != 0 {
                return 4;
            }
            match deliver_one(w) {
                Some(f) if f & EV_DELETE != 0 => 0,
                _ => 5,
            }
        });
        assert_eq!(code, 0, "3 the first fire, 4 refused, 5 not its deletion");
    }

    #[test]
    fn a_timer_enabled_with_new_values_while_its_fire_waits_fires_at_the_new_time() {
        let code = in_child(30, || {
            let w = 0x7f19;
            hold(w);
            let t = Kev { ident: 0x97, filter: EVFILT_TIMER, flags: EV_ADD | EV_ENABLE | EV_ONESHOT, data: 10, udata: 0x11a, ..Default::default() };
            if apply(w, t) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            let t0 = std::time::Instant::now();
            if apply(w, Kev { flags: EV_ENABLE, data: 300, ..t }) != 0 || !poll(w, None).is_empty() {
                return 4;
            }
            if wait_staged(w, 1, 2000) != 1 || t0.elapsed() < std::time::Duration::from_millis(250) {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "3 not staged, 4 refused or the old fire delivered, 5 no fire at the new time");
    }

    #[test]
    fn a_oneshot_knote_added_again_as_its_event_is_taken_leaves_no_knote_firing() {
        // The host knote made again under the old token keeps the flags the knote was attached with (oneshot), so it
        // does not fire on and on with no knote of its own.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                return 20;
            }
            let w = 0x7f1a;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_ONESHOT, udata: 0x11b, ..Default::default() };
            if apply(w, r) != 0 {
                return 3;
            }
            let fd = p[1];
            match change_as_it_is_taken(w, r, fd, Kev { flags: EV_ADD, ..r }) {
                Some(0) => {}
                None => return 22,
                Some(_) => return 4,
            }
            if poll(w, None).len() != 1 {
                return 5;
            }
            let before = cpu_ms();
            std::thread::sleep(std::time::Duration::from_millis(400));
            if cpu_ms() - before > 150 {
                return 6;
            }
            0
        });
        assert_eq!(code, 0, "20 no pipe, 22 no reading (the monitor took no event), 3/4 refused, 5 not delivered, 6 a host knote fired on and on");
    }

    #[test]
    fn a_repeating_timers_fires_while_its_event_waits_are_one_event() {
        // Natively a timer's fires not yet delivered are one event, its data their count.
        let code = in_child(30, || {
            let w = 0x7f1b;
            hold(w);
            if apply(w, Kev { ident: 0x98, filter: EVFILT_TIMER, flags: EV_ADD | EV_ENABLE, data: 10, udata: 0x11c, ..Default::default() }) != 0 {
                return 3;
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            if staged(w) != 1 {
                return 4;
            }
            match poll(w, None).as_slice() {
                [e] if e.data >= 5 => 0,
                _ => 5,
            }
        });
        assert_eq!(code, 0, "3 refused, 4 not one staged event, 5 its count not the fires'");
    }

    #[test]
    fn an_edge_knotes_end_of_file_while_its_event_waits_is_reported() {
        // An EV_CLEAR read knote with data waiting, then the writer gone: natively its one event says EV_EOF.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 || unsafe { libc::write(p[1], b"x".as_ptr().cast(), 1) } != 1 {
                return 20;
            }
            let w = 0x7f1d;
            hold(w);
            let r = Kev { ident: p[0] as u64, filter: EVFILT_READ, flags: EV_ADD | EV_CLEAR, udata: 0x11e, ..Default::default() };
            if apply(w, r) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 3;
            }
            let before = TAKEN.load(Ordering::SeqCst);
            unsafe { libc::close(p[1]) };
            let t0 = std::time::Instant::now();
            while TAKEN.load(Ordering::SeqCst) == before {
                if t0.elapsed() > std::time::Duration::from_secs(3) {
                    return 22;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
            match poll(w, None).as_slice() {
                [e] if e.flags & EV_EOF != 0 => 0,
                [_] => 4,
                _ => 5,
            }
        });
        assert_eq!(code, 0, "20 no pipe, 22 no reading (the monitor took no second event), 3 not staged, 4 no EV_EOF, 5 not one event");
    }

    #[test]
    fn a_deletion_the_kernel_would_refuse_is_refused_and_the_knote_kept() {
        // EV_VANISHED is not for a deletion: the kernel answers EINVAL.
        let code = in_child(30, || {
            let w = 0x7f1c;
            hold(w);
            let u = user(0xc5, EV_ADD | EV_CLEAR, 0x11d);
            if apply(w, u) != 0 {
                return 3;
            }
            if apply(w, Kev { flags: EV_DELETE | EV_VANISHED, ..u }) != libc::EINVAL as i64 {
                return 4;
            }
            if apply(w, trigger(u)) != 0 || wait_staged(w, 1, 3000) != 1 {
                return 5;
            }
            0
        });
        assert_eq!(code, 0, "3 refused, 4 not EINVAL, 5 the knote gone");
    }


    /// Tests elsewhere: hold workloop `wl`, watch `fd` for reading on it, and wait for its event; how many are staged.
    pub(crate) fn test_watch_read(wl: u64, fd: i32) -> usize {
        hold(wl);
        if apply(wl, Kev { ident: fd as u64, filter: EVFILT_READ, flags: EV_ADD, udata: 0x10e, ..Default::default() }) != 0 {
            return 0;
        }
        wait_staged(wl, 1, 3000)
    }

    pub(crate) fn test_staged(wl: u64) -> usize {
        staged(wl)
    }

    fn cpu_ms() -> u64 {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        let ms = |t: libc::timeval| t.tv_sec as u64 * 1000 + t.tv_usec as u64 / 1000;
        ms(ru.ru_utime) + ms(ru.ru_stime)
    }

    #[test]
    fn a_monitor_whose_wait_fails_pauses_rather_than_spin() {
        // A pipe is no kqueue: every wait on it fails with EBADF, as on a kqueue descriptor the guest closed and
        // reused. 600 ms of a spinning monitor costs about as much processor time; a pausing one a few milliseconds.
        let code = in_child(30, || {
            let mut p = [-1i32; 2];
            if unsafe { libc::pipe(p.as_mut_ptr()) } != 0 {
                return 20;
            }
            let before = cpu_ms();
            start_wl_monitor_on(p[0]);
            std::thread::sleep(std::time::Duration::from_millis(600));
            let used = cpu_ms() - before;
            if used < 150 {
                0
            } else {
                3
            }
        });
        assert_eq!(code, 0, "3: the monitor used 150 ms or more of processor time in 600 ms");
    }

    #[test]
    fn libpthread_thread_request_uses_affinity_argument() {
        // The Xcode linker requests 31 parallel workers with a NULL item.
        for op in [WQOPS_QUEUE_REQTHREADS, WQOPS_QUEUE_REQTHREADS2] {
            assert_eq!(requested_thread_count(&[op, 0, 31, 0x0400_08ff, 0, 0, 0, 0]), Ok(31));
        }
    }

    #[test]
    fn invalid_thread_requests_do_not_create_a_worker() {
        for count in [0, u32::MAX as u64] {
            assert_eq!(requested_thread_count(&[WQOPS_QUEUE_REQTHREADS, 0, count, 0, 0, 0, 0, 0]), Err(libc::EINVAL as u64));
        }
    }

    /// One NOTE_WL_SYNC_WAKE change whose state check loads the word at `addr` and compares it with `value` under
    /// `mask`, on a workloop of its own: the errno wl_apply answers. SYNC_WAKE requests no servicer, so nothing is
    /// scheduled.
    fn state_check(addr: u64, mask: u64, value: u64) -> i64 {
        let mut st = WlState::new();
        let ke = Kev { ident: 0x1234, filter: EVFILT_WORKLOOP, flags: EV_ADD, fflags: NOTE_WL_SYNC_WAKE, ext: [0, addr, mask, value], ..Default::default() };
        wl_apply(&mut st, 0x77, &ke)
    }

    fn page() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    fn map(prot: i32) -> u64 {
        let p = unsafe { libc::mmap(std::ptr::null_mut(), page(), prot, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
        assert!(p != libc::MAP_FAILED, "the test could not map its own page");
        p as u64
    }

    const STATE: u64 = 0x0000_0123_4000_0001;

    #[test]
    fn a_readable_state_word_is_compared_with_the_value() {
        let p = map(libc::PROT_READ | libc::PROT_WRITE);
        let w = p + 64;
        unsafe { *(w as *mut u64) = STATE };
        assert!(state_check(w, !0, STATE) == 0, "a matching state word must apply");
        assert!(state_check(w, !0, STATE + 1) == libc::ESTALE as i64, "a state word that differs must answer ESTALE");
        assert!(state_check(w, 0xff, STATE + 0x100) == 0, "only the masked bits are compared");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn a_state_word_at_a_bad_address_is_efault() {
        // As the kernel's copyin_atomic64 answers it. A PROT_NONE page this test holds, not a freed one, which a
        // parallel test thread could map again.
        let p = map(libc::PROT_NONE);
        assert!(state_check(p + 64, !0, STATE) == libc::EFAULT as i64, "an unreadable state word must answer EFAULT");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn a_state_word_our_reader_was_refused_is_still_checked() {
        // The reader refused for a reason that says nothing about the address, as a policy refusing
        // mach_vm_read_overwrite would answer: the guest's word is readable, so the kernel would have compared it,
        // and EFAULT (which libdispatch treats as fatal) would be wrong.
        let p = map(libc::PROT_READ | libc::PROT_WRITE);
        let w = p + 64;
        unsafe { *(w as *mut u64) = STATE };
        let noted = || crate::guestmem::REFUSALS_NOTED.with(|c| c.get());
        let before = noted();
        crate::guestmem::REFUSE_READS_IN.with(|c| c.set((p, p + page() as u64)));
        let matching = state_check(w, !0, STATE);
        let differing = state_check(w, !0, STATE + 1);
        crate::guestmem::REFUSE_READS_IN.with(|c| c.set((0, 0)));
        assert!(noted() == before + 2, "control: both reads must have been refused and reported");
        assert!(matching == 0, "a refused read of a matching state word answered {matching}, not 0");
        assert!(differing == libc::ESTALE as i64, "a refused read of a differing state word answered {differing}, not ESTALE");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }

    #[test]
    fn a_refused_read_of_an_unaligned_word_compares_what_a_read_would() {
        // The kernel answers EINVAL for an unaligned address (copyin_atomic64); the state check has always read an
        // unaligned word instead (the reader copies bytes). After a refusal it must still answer what the read
        // answers. (It cannot see a misaligned atomic load in place of read_unaligned: x86 performs one.)
        let p = map(libc::PROT_READ | libc::PROT_WRITE);
        let w = p + 64;
        unsafe { *(w as *mut u64) = STATE };
        let (mask, value) = (0xffff_ffffu64, STATE >> 32); // bytes 4..12: the word's high half, then zeros
        let read = (state_check(w + 4, mask, value), state_check(w + 4, mask, value + 1));
        crate::guestmem::REFUSE_READS_IN.with(|c| c.set((p, p + page() as u64)));
        let refused = (state_check(w + 4, mask, value), state_check(w + 4, mask, value + 1));
        crate::guestmem::REFUSE_READS_IN.with(|c| c.set((0, 0)));
        assert!(read == (0, libc::ESTALE as i64), "control: an unaligned word read normally answered {read:?}");
        assert!(refused == read, "after a refusal an unaligned word answered {refused:?}, where the read answered {read:?}");
        unsafe { libc::munmap(p as *mut libc::c_void, page()) };
    }
}

fn park() {
    if IS_MANAGER.with(|m| m.replace(false)) {
        release_manager();
    }
    PARK.with(|p| p.set(true));
    BOUND_WL.with(|b| b.set(None));
    crate::engine::exit_current_thread();
}
