//! Guest threads: emulation of the libpthread kernel interface
//! (bsdthread_register / bsdthread_create / bsdthread_terminate) and a
//! minimal workqueue so libdispatch can obtain worker threads.
//!
//! Every guest thread runs on its own host thread with its own `Cpu`.

use maclator_core::cpu::Cpu;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

static THREAD_START: AtomicU64 = AtomicU64::new(0);
static WQTHREAD_START: AtomicU64 = AtomicU64::new(0);
static PTHSIZE: AtomicU64 = AtomicU64::new(0);
static TSD_OFFSET: AtomicU32 = AtomicU32::new(0);
static MACH_THREAD_SELF_OFFSET: AtomicU32 = AtomicU32::new(0);
static DISPATCH_QUEUE_OFFSET: AtomicU64 = AtomicU64::new(0);
static REGISTERED: AtomicU32 = AtomicU32::new(0);
pub static LIVE_THREADS: AtomicUsize = AtomicUsize::new(1);

const PTHREAD_START_CUSTOM: u64 = 0x0100_0000;
const PTHREAD_START_TSD_BASE_SET: u64 = 0x1000_0000;
const PTHREAD_START_SUSPENDED: u64 = 0x2000_0000;

// DISPATCHFUNC | FINEPRIO | BSDTHREADCTL | SETSELF | QOS_MAINTENANCE | QOS_DEFAULT | KEVENT.
// Workloops are not offered: libdispatch then drives everything through the
// kevent workqueue, which Maclator emulates on top of a host kqueue.
const PTHREAD_FEATURE_SUPPORTED: u64 = 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x4000_0000 | 0x40 | 0x80;

/// bsdthread_register(threadstart, wqthread, pthsize, init_data, init_data_size, dispatchqueue_offset)
pub fn bsdthread_register(a: &[u64; 8]) -> Result<u64, u64> {
    if REGISTERED.swap(1, Ordering::SeqCst) != 0 {
        return Err(libc::EINVAL as u64);
    }
    THREAD_START.store(a[0], Ordering::SeqCst);
    WQTHREAD_START.store(a[1], Ordering::SeqCst);
    let pthsize = a[2] as u32 as u64;
    PTHSIZE.store(pthsize, Ordering::SeqCst);
    let data = a[3];
    let size = a[4];
    if data != 0 {
        unsafe {
            let version = *(data as *const u64);
            if version != size {
                return Err(libc::EINVAL as u64);
            }
            let p = data as *mut u8;
            let rd32 = |o: usize| std::ptr::read_unaligned(p.add(o) as *const u32);
            let dq = std::ptr::read_unaligned(p.add(8) as *const u64);
            let tsd = rd32(24);
            let mts = if size >= 36 { rd32(32) } else { 0 };
            let max_tsd = pthsize.saturating_sub(tsd as u64 + 8);
            TSD_OFFSET.store(if (tsd as u64) <= pthsize.saturating_sub(8) { tsd } else { 0 }, Ordering::SeqCst);
            MACH_THREAD_SELF_OFFSET.store(if (mts as u64) <= max_tsd { mts } else { 0 }, Ordering::SeqCst);
            DISPATCH_QUEUE_OFFSET.store(if dq <= max_tsd { dq } else { 0 }, Ordering::SeqCst);
            // copy-out fields
            let n = size.min(56) as usize;
            let mut out = [0u8; 56];
            std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), n);
            out[0..8].copy_from_slice(&56u64.min(size).to_le_bytes());
            out[16..24].copy_from_slice(&0u64.to_le_bytes()); // main_qos: unspecified
            if n >= 44 {
                out[36..44].copy_from_slice(&0x7000_0000_0000u64.to_le_bytes()); // stack_addr_hint
            }
            if n >= 48 {
                out[44..48].copy_from_slice(&0u32.to_le_bytes()); // mutex_default_policy
            }
            std::ptr::copy_nonoverlapping(out.as_ptr(), p, n);
        }
    } else {
        DISPATCH_QUEUE_OFFSET.store(a[5], Ordering::SeqCst);
    }
    Ok(PTHREAD_FEATURE_SUPPORTED)
}

struct SendCpu(Box<Cpu>);
unsafe impl Send for SendCpu {}

/// Spawn a host thread running guest code from `cpu`. Err when the thread could not be started.
fn spawn_guest(mut cpu: Box<Cpu>, setup: impl FnOnce(&mut Cpu) + Send + 'static) -> std::io::Result<()> {
    let sc = SendCpu(cpu_take(&mut cpu));
    // XNU writes the new thread's Mach port to the guest TSD before
    // bsdthread_create returns. The caller may inspect that slot immediately
    // (or detach the thread), so do not expose a partially initialized pthread.
    LIVE_THREADS.fetch_add(1, Ordering::SeqCst);
    let r = spawn_ready(8 << 20, move |ready| {
        let mut c = sc;
        setup(&mut c.0);
        ready();
        crate::engine::run_thread(&mut c.0);
        LIVE_THREADS.fetch_sub(1, Ordering::SeqCst);
    });
    if r.is_err() {
        LIVE_THREADS.fetch_sub(1, Ordering::SeqCst);
    }
    r
}

/// Spawn a host thread running `body`, and return once it has called `ready`.
/// The wait is a pthread mutex and condvar, not an mpsc channel. A
/// channel parks this thread on its std parker, and in a fork child the forking thread's parker
/// may hold a semaphore made in the parent, which the child does not have. And the thread is a
/// plain pthread (forksync::spawn_host), not a std::thread.
/// Err, not a wait for good, when the thread ends without calling `ready`
/// (or could not be started): the creator's state is set to "gone" when the thread's closure is
/// dropped. A panic in `body` aborts the process (panic = "abort"); under the unwinding test
/// harness it ends the thread and also reads as "gone".
/// Once `ready` has published, the thread's end takes no lock.
/// The guest code that runs after `ready` may fork, and the child's copy of the handshake mutex may
/// be held by the creator, which is not in the child; a cleanup that locked it would wait for good.
/// `ready` sets an atomic flag (under the mutex, so the creator still sees the state change under
/// it) and the drop returns at once when the flag is set: nobody waits on a handshake after ready.
pub(crate) fn spawn_ready(stack: usize, body: impl FnOnce(&dyn Fn()) + Send + 'static) -> std::io::Result<()> {
    use crate::forksync::{FCondvar, FMutex};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    /// 0 waiting, 1 ready, 2 the thread is gone without calling ready; and "ready was published"
    struct Handshake(FMutex<u8>, FCondvar, AtomicBool);
    struct GoneOnDrop(Arc<Handshake>);
    impl Drop for GoneOnDrop {
        fn drop(&mut self) {
            if self.0 .2.load(Ordering::Acquire) {
                return;
            }
            let mut g = self.0 .0.lock().unwrap();
            if *g == 0 {
                *g = 2;
            }
            drop(g);
            self.0 .1.notify_all();
        }
    }
    let hs = Arc::new(Handshake(FMutex::new(0), FCondvar::new(), AtomicBool::new(false)));
    #[cfg(test)]
    test_note_handshake(&hs.0);
    let theirs = GoneOnDrop(hs.clone());
    let started = crate::forksync::spawn_host("", stack, Box::new(move || {
        let t = theirs;
        body(&|| {
            let mut g = t.0 .0.lock().unwrap();
            *g = 1;
            t.0 .2.store(true, Ordering::Release);
            drop(g);
            t.0 .1.notify_all();
        });
    }));
    started?;
    let mut g = hs.0.lock().unwrap();
    while *g == 0 {
        g = hs.1.wait(g).unwrap();
    }
    if *g == 1 {
        Ok(())
    } else {
        Err(std::io::Error::other("the new host thread ended before it was ready"))
    }
}

/// Tests: the handshake mutex of the latest spawn_ready, so a test can hold it across a
/// fork the new thread makes. Valid while that thread runs (its GoneOnDrop holds the Arc).
#[cfg(test)]
static TEST_LAST_HANDSHAKE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
fn test_note_handshake(m: &crate::forksync::FMutex<u8>) {
    TEST_LAST_HANDSHAKE.store(m as *const _ as usize, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn test_last_handshake() -> &'static crate::forksync::FMutex<u8> {
    let p = TEST_LAST_HANDSHAKE.load(Ordering::SeqCst);
    assert!(p != 0, "no handshake yet");
    unsafe { &*(p as *const crate::forksync::FMutex<u8>) }
}

fn cpu_take(c: &mut Box<Cpu>) -> Box<Cpu> {
    std::mem::replace(c, Cpu::new())
}

/// bsdthread_create(func, func_arg, stack, pthread, flags)
pub fn bsdthread_create(a: &[u64; 8]) -> Result<u64, u64> {
    let (func, arg, stack, pthread, flags) = (a[0], a[1], a[2], a[3], a[4]);
    if REGISTERED.load(Ordering::SeqCst) == 0 || flags & PTHREAD_START_CUSTOM == 0 {
        return Err(libc::EINVAL as u64);
    }
    let start = THREAD_START.load(Ordering::SeqCst);
    let tsd_off = TSD_OFFSET.load(Ordering::SeqCst) as u64;
    let mts_off = MACH_THREAD_SELF_OFFSET.load(Ordering::SeqCst) as u64;
    let mut cpu = Cpu::new();
    cpu.pc = start;
    cpu.x[0] = pthread;
    cpu.x[2] = func;
    cpu.x[3] = arg;
    cpu.x[4] = stack;
    let mut fl = flags & !PTHREAD_START_SUSPENDED;
    if tsd_off != 0 {
        cpu.tpidrro_el0 = pthread + tsd_off;
        fl |= PTHREAD_START_TSD_BASE_SET;
    }
    cpu.x[5] = fl;
    cpu.x[31] = stack;
    spawn_guest(cpu, move |c| {
        let port = unsafe { libc::mach_thread_self() } as u64;
        c.x[1] = port;
        if tsd_off != 0 && mts_off != 0 {
            unsafe { *((pthread + tsd_off + mts_off) as *mut u64) = port };
        }
    })
    .map_err(|_| libc::EAGAIN as u64)?;
    Ok(pthread)
}

/// bsdthread_terminate(stackaddr, freesize, port, sem)
pub fn bsdthread_terminate(_cpu: &mut Cpu, a: &[u64; 8]) {
    let (stackaddr, freesize, _port, sem) = (a[0], a[1], a[2], a[3]);
    unsafe {
        if freesize != 0 {
            libc::munmap(stackaddr as *mut _, freesize as usize);
        }
        if sem != 0 {
            // semaphore_signal_internal_trap
            crate::hostsys::mach(33, &[sem, 0, 0, 0, 0, 0, 0, 0]);
        }
    }
    crate::engine::exit_current_thread();
}

// The workqueue (libdispatch worker threads and kernel event delivery) lives
// in `workq.rs`; it needs the registration data kept here.
pub fn wq_registration() -> (u64, u64, u64, u64) {
    (
        WQTHREAD_START.load(Ordering::SeqCst),
        TSD_OFFSET.load(Ordering::SeqCst) as u64,
        MACH_THREAD_SELF_OFFSET.load(Ordering::SeqCst) as u64,
        PTHSIZE.load(Ordering::SeqCst),
    )
}
