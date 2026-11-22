//! Translator locks that a fork child can make fresh in place.
//!
//! std's locks cannot be reset after a fork. On macOS std's RwLock, Once, OnceLock and mpsc park
//! waiters on a per-thread parker whose kernel semaphore is a Mach port, and the child of a fork
//! does not get the parent's ports, so a child that waits on or wakes a parker made in the parent
//! dies (libdispatch: "Use-after-free of dispatch_semaphore_t").
//! These types wrap the pthread objects directly. They wait in the kernel by address, with no
//! per-thread parker, and `ForkHeld::reinit_in_child` writes a fresh initialiser over a lock the
//! forking thread holds, so the child starts with every lock free and with no waiter state copied
//! from the parent. Each translator lock is one of these; std's Mutex, RwLock, Condvar, Once,
//! OnceLock and LazyLock are not used for a static (forksafe's tests check the source for that).
//!
//! The API matches the part of std's that the translator uses: `lock().unwrap()`,
//! `try_lock()`, `read()`, `write()`, `Condvar::wait`, `OnceLock::get_or_init`. There is no
//! poisoning: `lock` never fails.
//!
//! Soundness. A guard is neither Send nor Sync of itself (a pthread lock must be
//! released by the thread that took it); it is Sync only when T is Sync, as std's guards are.
//! Making a held lock fresh is `unsafe` and reachable only through `forksafe::fork`'s child branch:
//! see `ForkHeld` for the invariant.
//!
//! Output while a lock is held: this thread counts the forksync locks it holds,
//! and `write_out` keeps the text of a thread that holds one in a per-thread buffer, written when
//! its last lock is released. So no thread blocks in write(2) on a full pipe while it holds a lock
//! that a fork has to take.
//!
//! The kept text is bounded: at most DEFER_CAP bytes per thread; text
//! past that is dropped and counted, and the flush ends with one line saying how many bytes were
//! dropped. A thread waiting on an FCondvar still counts its mutex as held (the wait gives the mutex
//! back but not the count), so its output during the wait is kept too, under the same bound.

use std::cell::{Cell, RefCell, UnsafeCell};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

thread_local! {
    /// forksync locks this thread holds (each guard counts one).
    static HELD: Cell<u32> = const { Cell::new(0) };
    /// write_out text kept while HELD > 0: (fd, bytes).
    static DEFERRED: RefCell<Vec<(i32, Vec<u8>)>> = const { RefCell::new(Vec::new()) };
    /// Bytes kept in DEFERRED, and bytes dropped past DEFER_CAP since the last flush.
    static DEFERRED_BYTES: Cell<usize> = const { Cell::new(0) };
    static DEFERRED_DROPPED: Cell<usize> = const { Cell::new(0) };
    /// FOnceLock initialisers this thread is inside (see FOnceLock).
    static LAZY_DEPTH: Cell<usize> = const { Cell::new(0) };
    /// write_out writes at once on this thread even while it holds a lock (fault_dump).
    static DIRECT: Cell<bool> = const { Cell::new(false) };
}

/// The process is about to exit from this thread (engine::fault_dump): write what this thread
/// kept, and from now on write at once, lock or no lock, so nothing is lost at _exit.
pub(crate) fn output_direct_on_this_thread() {
    DIRECT.with(|d| d.set(true));
    flush_deferred();
}

fn held_inc() {
    HELD.with(|h| h.set(h.get() + 1));
}

fn held_dec() {
    let left = HELD.with(|h| {
        let n = h.get().saturating_sub(1);
        h.set(n);
        n
    });
    if left == 0 {
        flush_deferred();
    }
}

/// Forksync locks this thread holds (tests and forksafe).
pub(crate) fn held_count() -> u32 {
    HELD.with(|h| h.get())
}

/// In the fork child, on the forking thread, after every lock was made fresh: it holds none.
/// Safety: only from forksafe::fork's child branch (see ForkHeld).
pub(crate) unsafe fn reset_held_in_child() {
    HELD.with(|h| h.set(0));
}

/// The most text one thread keeps while it holds a lock: 256 KiB.
pub(crate) const DEFER_CAP: usize = 256 << 10;

fn flush_deferred() {
    // try_with: a guard may be dropped while this thread's thread-locals are being destroyed
    // (a thread-local's destructor releasing a lock), when DEFERRED is already gone.
    let out = DEFERRED.try_with(|d| std::mem::take(&mut *d.borrow_mut())).unwrap_or_default();
    let _ = DEFERRED_BYTES.try_with(|n| n.set(0));
    let dropped = DEFERRED_DROPPED.try_with(|n| n.replace(0)).unwrap_or(0);
    let mut last_fd = 2;
    for (fd, b) in out {
        write_all(fd, &b);
        last_fd = fd;
    }
    if dropped > 0 {
        write_all(last_fd, format!("[maclator] {dropped} bytes of output dropped: written while a lock was held, past the {DEFER_CAP}-byte cap\n").as_bytes());
    }
}

/// Never constructed: the `Err` side of `lock`, so `.lock().unwrap()` and
/// `.unwrap_or_else(|e| e.into_inner())` read as they do for std's locks.
pub struct Poisoned<G>(G);
impl<G> Poisoned<G> {
    #[allow(dead_code)] // the std-shaped API: `.unwrap_or_else(|e| e.into_inner())`
    pub fn into_inner(self) -> G {
        self.0
    }
}
impl<G> std::fmt::Debug for Poisoned<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Poisoned")
    }
}

/// `try_lock` found the lock held.
#[derive(Debug)]
pub struct WouldBlock;

/// Makes a guard neither Send nor Sync (a raw pointer is neither).
type NotSendSync = PhantomData<*const ()>;

pub struct FMutex<T> {
    m: UnsafeCell<libc::pthread_mutex_t>,
    v: UnsafeCell<T>,
}
unsafe impl<T: Send> Sync for FMutex<T> {}
unsafe impl<T: Send> Send for FMutex<T> {}

impl<T> FMutex<T> {
    pub const fn new(v: T) -> Self {
        FMutex { m: UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER), v: UnsafeCell::new(v) }
    }
    pub fn lock(&self) -> Result<FMutexGuard<'_, T>, Poisoned<FMutexGuard<'_, T>>> {
        let rc = unsafe { libc::pthread_mutex_lock(self.m.get()) };
        assert_eq!(rc, 0, "pthread_mutex_lock");
        held_inc();
        Ok(FMutexGuard { l: self, _n: PhantomData })
    }
    pub fn try_lock(&self) -> Result<FMutexGuard<'_, T>, WouldBlock> {
        if unsafe { libc::pthread_mutex_trylock(self.m.get()) } == 0 {
            held_inc();
            Ok(FMutexGuard { l: self, _n: PhantomData })
        } else {
            Err(WouldBlock)
        }
    }
}

pub struct FMutexGuard<'a, T> {
    l: &'a FMutex<T>,
    _n: NotSendSync,
}
unsafe impl<T: Sync> Sync for FMutexGuard<'_, T> {}
impl<T> Deref for FMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.l.v.get() }
    }
}
impl<T> DerefMut for FMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.l.v.get() }
    }
}
impl<T> Drop for FMutexGuard<'_, T> {
    fn drop(&mut self) {
        unsafe { libc::pthread_mutex_unlock(self.l.m.get()) };
        held_dec();
    }
}

pub struct FRwLock<T> {
    l: UnsafeCell<libc::pthread_rwlock_t>,
    v: UnsafeCell<T>,
}
unsafe impl<T: Send + Sync> Sync for FRwLock<T> {}
unsafe impl<T: Send> Send for FRwLock<T> {}

impl<T> FRwLock<T> {
    pub const fn new(v: T) -> Self {
        FRwLock { l: UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER), v: UnsafeCell::new(v) }
    }
    pub fn read(&self) -> Result<FReadGuard<'_, T>, Poisoned<FReadGuard<'_, T>>> {
        let rc = unsafe { libc::pthread_rwlock_rdlock(self.l.get()) };
        assert_eq!(rc, 0, "pthread_rwlock_rdlock");
        held_inc();
        Ok(FReadGuard { l: self, _n: PhantomData })
    }
    pub fn write(&self) -> Result<FWriteGuard<'_, T>, Poisoned<FWriteGuard<'_, T>>> {
        let rc = unsafe { libc::pthread_rwlock_wrlock(self.l.get()) };
        assert_eq!(rc, 0, "pthread_rwlock_wrlock");
        held_inc();
        Ok(FWriteGuard { l: self, _n: PhantomData })
    }
    #[cfg_attr(not(test), allow(dead_code))] // forksafe's tests
    pub fn try_write(&self) -> Result<FWriteGuard<'_, T>, WouldBlock> {
        if unsafe { libc::pthread_rwlock_trywrlock(self.l.get()) } == 0 {
            held_inc();
            Ok(FWriteGuard { l: self, _n: PhantomData })
        } else {
            Err(WouldBlock)
        }
    }
}

pub struct FReadGuard<'a, T> {
    l: &'a FRwLock<T>,
    _n: NotSendSync,
}
unsafe impl<T: Sync> Sync for FReadGuard<'_, T> {}
impl<T> Deref for FReadGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.l.v.get() }
    }
}
impl<T> Drop for FReadGuard<'_, T> {
    fn drop(&mut self) {
        unsafe { libc::pthread_rwlock_unlock(self.l.l.get()) };
        held_dec();
    }
}

pub struct FWriteGuard<'a, T> {
    l: &'a FRwLock<T>,
    _n: NotSendSync,
}
unsafe impl<T: Sync> Sync for FWriteGuard<'_, T> {}
impl<T> Deref for FWriteGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.l.v.get() }
    }
}
impl<T> DerefMut for FWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.l.v.get() }
    }
}
impl<T> Drop for FWriteGuard<'_, T> {
    fn drop(&mut self) {
        unsafe { libc::pthread_rwlock_unlock(self.l.l.get()) };
        held_dec();
    }
}

pub struct FCondvar {
    c: UnsafeCell<libc::pthread_cond_t>,
}
unsafe impl Sync for FCondvar {}
unsafe impl Send for FCondvar {}

impl FCondvar {
    pub const fn new() -> Self {
        FCondvar { c: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER) }
    }
    pub fn wait<'a, T>(&self, g: FMutexGuard<'a, T>) -> Result<FMutexGuard<'a, T>, Poisoned<FMutexGuard<'a, T>>> {
        let rc = unsafe { libc::pthread_cond_wait(self.c.get(), g.l.m.get()) };
        assert_eq!(rc, 0, "pthread_cond_wait");
        Ok(g)
    }
    pub fn notify_all(&self) {
        unsafe { libc::pthread_cond_broadcast(self.c.get()) };
    }
    /// A fresh condvar, so no waiter counted in the parent can take a wake-up meant for a child
    /// thread.
    ///
    /// # Safety
    /// Only in a fork child, on the forking thread, before the child starts any thread, while no
    /// thread of the child waits on it (forksafe::fork's child branch, through reset_in_child).
    pub unsafe fn reinit_in_child(&self) {
        std::ptr::write(self.c.get(), libc::PTHREAD_COND_INITIALIZER);
    }
}

/// A translator lock held across a fork: released in the parent, made fresh in the child.
///
/// Implemented only here, for FMutexGuard and FWriteGuard, and called only from
/// `forksafe::Held::reinit_in_child`, which `forksafe::fork` calls on its child branch.
pub(crate) trait ForkHeld {
    /// Forget the guard without unlocking, and write a fresh initialiser over the lock, so no state
    /// copied from the parent survives in it.
    ///
    /// # Safety
    /// The caller must be the forking thread of a fork child, between libc::fork returning 0 and
    /// any other use of the lock, with this guard taken by this thread before the fork, and with no
    /// other thread in the child (a fork child has exactly one until it starts more). Called
    /// anywhere else it overwrites a lock that another thread may be using.
    unsafe fn reinit_in_child(self: Box<Self>);
}

impl<T> ForkHeld for FMutexGuard<'static, T> {
    unsafe fn reinit_in_child(self: Box<Self>) {
        let l = self.l;
        std::mem::forget(*self);
        std::ptr::write(l.m.get(), libc::PTHREAD_MUTEX_INITIALIZER);
    }
}

impl<T> ForkHeld for FWriteGuard<'static, T> {
    unsafe fn reinit_in_child(self: Box<Self>) {
        let l = self.l;
        std::mem::forget(*self);
        std::ptr::write(l.l.get(), libc::PTHREAD_RWLOCK_INITIALIZER);
    }
}

/// The gate every FOnceLock initialiser passes on its way in; forksafe::hold_all takes it last.
pub(crate) static LAZY_GATE: FMutex<()> = FMutex::new(());
/// Threads inside an FOnceLock initialiser, or waiting for another thread's (counted per level).
static LAZY_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Counts this thread in LAZY_IN_FLIGHT while it is alive. The outermost level enters through
/// LAZY_GATE, so no initialiser can start while a fork holds the gate.
struct LazyEnter;
impl LazyEnter {
    fn new() -> LazyEnter {
        let depth = LAZY_DEPTH.with(|d| {
            let n = d.get();
            d.set(n + 1);
            n
        });
        if depth == 0 {
            let _g = LAZY_GATE.lock().unwrap();
            LAZY_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
        } else {
            LAZY_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
        }
        LazyEnter
    }
}
impl Drop for LazyEnter {
    fn drop(&mut self) {
        LAZY_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        LAZY_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// With LAZY_GATE held (no initialiser can start): wait until every initialiser in flight on
/// another thread has finished. forksafe::hold_all calls it with every translator lock held, and an
/// initialiser takes no translator lock (forksafe's tests check that for each one), so the wait ends.
pub(crate) fn wait_for_lazy_inits() {
    let mine = LAZY_DEPTH.with(|d| d.get());
    while LAZY_IN_FLIGHT.load(Ordering::SeqCst) > mine {
        unsafe { libc::usleep(100) };
    }
}

/// In the fork child: no other thread is inside an initialiser.
/// Safety: only from forksafe::fork's child branch (see ForkHeld).
pub(crate) unsafe fn reset_lazy_in_child() {
    LAZY_IN_FLIGHT.store(LAZY_DEPTH.with(|d| d.get()), Ordering::SeqCst);
}

/// std's OnceLock without its parker. A fork can never copy an initialiser in
/// flight on another thread: `get_or_init` counts itself in LAZY_IN_FLIGHT, entering through
/// LAZY_GATE, and forksafe::fork takes the gate and waits for that count to drop before it forks.
/// Waiters wait on a pthread mutex, not on a parker. An initialiser must take no translator lock.
pub struct FOnceLock<T> {
    ready: AtomicBool,
    m: FMutex<()>,
    v: UnsafeCell<MaybeUninit<T>>,
}
unsafe impl<T: Send + Sync> Sync for FOnceLock<T> {}
unsafe impl<T: Send> Send for FOnceLock<T> {}

impl<T> FOnceLock<T> {
    pub const fn new() -> Self {
        FOnceLock { ready: AtomicBool::new(false), m: FMutex::new(()), v: UnsafeCell::new(MaybeUninit::uninit()) }
    }
    pub fn get(&self) -> Option<&T> {
        if self.ready.load(Ordering::Acquire) {
            Some(unsafe { (*self.v.get()).assume_init_ref() })
        } else {
            None
        }
    }
    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &T {
        if let Some(v) = self.get() {
            return v;
        }
        let enter = LazyEnter::new();
        let g = self.m.lock().unwrap();
        if !self.ready.load(Ordering::Acquire) {
            let v = f();
            unsafe { (*self.v.get()).write(v) };
            self.ready.store(true, Ordering::Release);
        }
        // Leave the count before the lock: releasing the last lock writes this thread's deferred
        // output, which may block on a full pipe, and a fork must not wait for that. Once `ready`
        // is set no thread takes `m` again, so a fork between these two lines is harmless.
        drop(enter);
        drop(g);
        self.get().expect("FOnceLock initialised")
    }
}

impl<T> Drop for FOnceLock<T> {
    fn drop(&mut self) {
        if *self.ready.get_mut() {
            unsafe { self.v.get_mut().assume_init_drop() };
        }
    }
}

/// Start a detached host thread with pthread_create, not std::thread. std records every thread it
/// starts, and removes it again at the thread's exit, in a process-wide table behind a pthread mutex
/// (its stack-overflow guard information, std::sys::pal::unix::stack_overflow::thread_info). A fork
/// while another thread is inside either copies that mutex held, and then the child's first
/// std::thread start or exit waits for good. Measured: a child stuck in set_current_info, in
/// _pthread_mutex_firstfit_lock_slow, about 1 child in 1,200 in the unit tests. A thread started here
/// never touches that table.
///
/// What a thread started here does not get from std: std's thread name (the OS name
/// is set, from inside the thread, when `name` is not empty, so `std::thread::current().name()` is
/// None) and std's stack-overflow handler (the pthread guard page is still there, so an overflow is
/// a plain SIGSEGV with no "has overflowed its stack" message). A panic: Maclator builds with
/// panic = "abort" in both profiles, so a panic in such a thread ends the process; only under the
/// test harness, which unwinds, does catch_unwind below end just the thread.
pub fn spawn_host(name: &str, stack: usize, f: Box<dyn FnOnce() + Send + 'static>) -> std::io::Result<()> {
    type Start = (Option<std::ffi::CString>, Box<dyn FnOnce() + Send + 'static>);
    extern "C" fn start(p: *mut libc::c_void) -> *mut libc::c_void {
        let b: Box<Start> = unsafe { Box::from_raw(p as *mut Start) };
        let (name, f) = *b;
        if let Some(n) = name {
            unsafe { libc::pthread_setname_np(n.as_ptr()) };
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        std::ptr::null_mut()
    }
    let name = if name.is_empty() { None } else { std::ffi::CString::new(name).ok() };
    let p = Box::into_raw(Box::new((name, f) as Start)) as *mut libc::c_void;
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        let mut rc = libc::pthread_attr_init(&mut attr);
        if rc == 0 {
            rc = libc::pthread_attr_setstacksize(&mut attr, stack.max(libc::PTHREAD_STACK_MIN));
        }
        if rc == 0 {
            rc = libc::pthread_attr_setdetachstate(&mut attr, libc::PTHREAD_CREATE_DETACHED);
        }
        if rc == 0 {
            let mut t: libc::pthread_t = std::mem::zeroed();
            rc = libc::pthread_create(&mut t, &attr, start, p);
        }
        libc::pthread_attr_destroy(&mut attr);
        if rc != 0 {
            drop(Box::from_raw(p as *mut Start));
            return Err(std::io::Error::from_raw_os_error(rc));
        }
    }
    Ok(())
}

fn write_all(fd: i32, mut b: &[u8]) {
    while !b.is_empty() {
        let n = unsafe { libc::write(fd, b.as_ptr() as *const libc::c_void, b.len()) };
        if n < 0 {
            if unsafe { *libc::__error() } == libc::EINTR {
                continue;
            }
            return;
        }
        b = &b[n as usize..];
    }
}

/// Write `s` (and a newline) to `fd` with plain write(2), taking no lock. The crate's print! /
/// eprintln! family (main.rs) comes here. A thread that holds a forksync lock does not write: its
/// text is kept and written when it releases its last one, so a write blocked on a full
/// pipe never holds a lock that a fork waits for. Errors are dropped, where std's eprintln! would
/// panic. Output is not serialised between threads: a long write from one thread can interleave
/// with another's, and kept text comes out after text written meanwhile by other threads. Kept
/// text is bounded by DEFER_CAP per thread; past it, text is dropped and the flush says how much.
pub fn write_out(fd: i32, mut s: String, newline: bool) {
    if newline {
        s.push('\n');
    }
    if held_count() > 0 && !DIRECT.with(|d| d.get()) {
        let len = s.len();
        let room = DEFERRED_BYTES.try_with(|n| n.get() + len <= DEFER_CAP);
        if matches!(room, Ok(false)) {
            let _ = DEFERRED_DROPPED.try_with(|n| n.set(n.get() + len));
            return;
        }
        let mut b = Some(s.into_bytes());
        let kept = DEFERRED.try_with(|d| d.borrow_mut().push((fd, b.take().unwrap()))).is_ok();
        if kept {
            let _ = DEFERRED_BYTES.try_with(|n| n.set(n.get() + len));
            return;
        }
        // This thread's thread-locals are being destroyed: write now.
        write_all(fd, &b.unwrap_or_default());
        return;
    }
    write_all(fd, s.as_bytes());
}

// Compile-time checks: a guard is not Send, and a guard over a !Sync value is
// not Sync. Each line below fails to compile (E0283, ambiguous impl) if the type DOES implement the
// trait; this is static_assertions' assert_not_impl_any. forksafe's tests also compile small
// programs that misuse the API and require them to be refused (compile-fail tests).
#[allow(dead_code)]
mod not_send {
    use super::*;
    trait AmbiguousIfSend<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    struct IsSend;
    impl<T: ?Sized + Send> AmbiguousIfSend<IsSend> for T {}
    const _: fn() = || {
        let _ = <FMutexGuard<'static, u32> as AmbiguousIfSend<_>>::some_item;
        let _ = <FReadGuard<'static, u32> as AmbiguousIfSend<_>>::some_item;
        let _ = <FWriteGuard<'static, u32> as AmbiguousIfSend<_>>::some_item;
    };
}

#[allow(dead_code)]
mod not_sync {
    use super::*;
    trait AmbiguousIfSync<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSync<()> for T {}
    struct IsSync;
    impl<T: ?Sized + Sync> AmbiguousIfSync<IsSync> for T {}
    const _: fn() = || {
        let _ = <FMutexGuard<'static, std::cell::Cell<u32>> as AmbiguousIfSync<_>>::some_item;
        let _ = <FReadGuard<'static, std::cell::Cell<u32>> as AmbiguousIfSync<_>>::some_item;
        let _ = <FWriteGuard<'static, std::cell::Cell<u32>> as AmbiguousIfSync<_>>::some_item;
    };
}
