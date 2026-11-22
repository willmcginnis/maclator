//! What the fork-safety tests (forksafe_v4tests.rs, forksafe_v5tests.rs) need from the code under test,
//! kept in this one file so the same tests can also be built against an earlier version of that code
//! to show each of them failing there. This is the current version's.

pub(crate) type Lazy<T> = crate::forksync::FOnceLock<T>;

pub(crate) fn generation() -> u64 {
    crate::forksafe::generation()
}

pub(crate) fn worker_is_stale(gen0: u64) -> bool {
    crate::workq::worker_is_stale(gen0)
}

pub(crate) fn spawn_ready(body: impl FnOnce(&dyn Fn()) + Send + 'static) -> Result<(), ()> {
    crate::threads::spawn_ready(1 << 20, body).map_err(|_| ())
}

pub(crate) fn dumps_done() -> u64 {
    crate::engine::dumps_done()
}

/// A workqueue worker's exit, as the worker loop runs it (workq::worker_after_run): None when
/// the host thread ends, else the channel ends handed back for the idle wait.
pub(crate) fn worker_exit(parked: bool, stale: bool, rx: std::sync::mpsc::Receiver<u8>, tx: std::sync::mpsc::Sender<u8>) -> Option<(std::sync::mpsc::Receiver<u8>, std::sync::mpsc::Sender<u8>)> {
    match crate::workq::worker_after_run(parked, stale, rx, tx) {
        crate::workq::AfterRun::Idle(r, t) => Some((r, t)),
        crate::workq::AfterRun::End => None,
    }
}

/// SIGUSR2 lines the handler handed to the dump thread.
pub(crate) fn usr2_done() -> u64 {
    crate::engine::usr2_done()
}
