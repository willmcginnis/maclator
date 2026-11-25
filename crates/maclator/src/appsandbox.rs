//! The App Sandbox is the translated program's to enter, once. maclator's own libSystem start-up would otherwise enter
//! it first on the program's behalf, and a kernel can then refuse the program's own request: appsandbox.c, the library
//! maclator is linked against for this, says how. Here: ending that library's filter as the guest starts.

use std::sync::atomic::Ordering;

const LIB: &str = "libmaclator_appsandbox.dylib";

extern "C" {
    fn sandbox_check(pid: libc::pid_t, operation: *const libc::c_char, filter_type: libc::c_int, ...) -> libc::c_int;
}

/// The library's `name`, if the library is loaded (it is linked weakly: a maclator without it still runs).
fn library_fn(name: &std::ffi::CStr) -> Option<*mut libc::c_void> {
    let f = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    (!f.is_null()).then_some(f)
}

/// Called once, just before the guest's first instruction: from here on every caller of
/// xpc_copy_entitlements_for_self in this process gets the real answer, and the program's own libsecinit makes the
/// App Sandbox request.
pub fn guest_starts() {
    let Some(disarm) = library_fn(c"maclator_appsandbox_disarm") else {
        // Without the library, the start-up may have entered the App Sandbox already, and a kernel can then refuse
        // the program's own request (its libsecinit traps); or the sandbox was inherited from a parent.
        if unsafe { sandbox_check(libc::getpid(), std::ptr::null(), 0) } > 0 {
            eprintln!("maclator: {LIB} is not loaded and this process is already sandboxed before the program starts; if maclator's own start-up entered the App Sandbox, the program's own sandbox request may be refused");
        }
        return;
    };
    let disarm: unsafe extern "C" fn() -> libc::c_uint = unsafe { std::mem::transmute(disarm) };
    let filtered = unsafe { disarm() };
    if crate::syscalls::TRACE.load(Ordering::Relaxed) {
        eprintln!("[appsandbox] {filtered} start-up read(s) of the entitlements answered without the App Sandbox key");
    }
}

#[cfg(test)]
mod tests {
    use super::library_fn;
    use std::ffi::{c_char, c_void, CStr};

    type Obj = *mut c_void;
    const KEY: &CStr = c"com.apple.security.app-sandbox";
    const OTHER: &CStr = c"com.example.other";

    extern "C" {
        fn xpc_copy_entitlements_for_self() -> Obj;
        fn xpc_dictionary_create(keys: *const *const c_char, values: *const Obj, count: usize) -> Obj;
        fn xpc_dictionary_set_bool(d: Obj, key: *const c_char, value: bool);
        fn xpc_dictionary_get_bool(d: Obj, key: *const c_char) -> bool;
        fn xpc_dictionary_get_value(d: Obj, key: *const c_char) -> Obj;
        fn xpc_dictionary_get_count(d: Obj) -> usize;
        fn xpc_bool_create(value: bool) -> Obj;
        fn xpc_retain(o: Obj) -> Obj;
        fn xpc_release(o: Obj);
    }

    fn filter(e: Obj) -> Obj {
        let f = library_fn(c"maclator_appsandbox_filter").expect("the library is loaded");
        let f: unsafe extern "C" fn(Obj) -> Obj = unsafe { std::mem::transmute(f) };
        unsafe { f(e) }
    }

    /// (reads, filtered, image of the first read) before the disarm.
    fn stats() -> (u32, u32, String) {
        let f = library_fn(c"maclator_appsandbox_stats").expect("the library is loaded");
        let f: unsafe extern "C" fn(*mut u32, *mut u32, *mut *const c_void) = unsafe { std::mem::transmute(f) };
        let (mut r, mut n, mut first) = (0u32, 0u32, std::ptr::null());
        unsafe { f(&mut r, &mut n, &mut first) };
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        let image = if !first.is_null() && unsafe { libc::dladdr(first, &mut info) } != 0 && !info.dli_fname.is_null() {
            unsafe { CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned()
        } else {
            String::new()
        };
        (r, n, image)
    }

    fn dict(app_sandbox: Option<bool>) -> Obj {
        let d = unsafe { xpc_dictionary_create(std::ptr::null(), std::ptr::null(), 0) };
        assert!(!d.is_null());
        unsafe { xpc_dictionary_set_bool(d, OTHER.as_ptr(), true) };
        if let Some(v) = app_sandbox {
            unsafe { xpc_dictionary_set_bool(d, KEY.as_ptr(), v) };
        }
        d
    }

    /// Whether `d` (built by `dict(Some(true))`) still holds both its keys as they were.
    fn intact(d: Obj) -> bool {
        unsafe {
            xpc_dictionary_get_bool(d, KEY.as_ptr()) && xpc_dictionary_get_bool(d, OTHER.as_ptr()) && xpc_dictionary_get_count(d) == 2
        }
    }

    /// The start-up's own read of the entitlements (libsystem_secinit's, from libSystem's initializer) reached the
    /// library's replacement in this executable, which is linked as maclator is: dyld applied the interposing to a
    /// call between two images of the shared cache. This process has no App Sandbox key, so nothing was filtered.
    #[test]
    fn the_start_up_read_of_the_entitlements_reaches_the_library() {
        for name in [c"maclator_appsandbox_disarm", c"maclator_appsandbox_filter", c"maclator_appsandbox_answer", c"maclator_appsandbox_stats"] {
            assert!(library_fn(name).is_some(), "{name:?} is not loaded: the library was not linked or not found");
        }
        let (reads, filtered, image) = stats();
        assert!(reads >= 1, "no read of the entitlements reached the replacement before main");
        assert!(image.ends_with("/libsystem_secinit.dylib"), "the first read came from {image:?}, not libsystem_secinit");
        assert_eq!(filtered, 0, "this test process has no App Sandbox key to take out");
    }

    #[test]
    fn the_filter_takes_out_the_app_sandbox_key_and_only_it() {
        let d = dict(Some(true));
        unsafe { xpc_retain(d) }; // one reference goes to the filter, one stays here to look at the original
        let f = filter(d);
        assert!(!f.is_null() && f != d, "a dictionary with the key true comes back as a new dictionary");
        unsafe {
            assert!(xpc_dictionary_get_value(f, KEY.as_ptr()).is_null(), "the App Sandbox key is gone");
            assert!(xpc_dictionary_get_bool(f, OTHER.as_ptr()), "the other key is kept with its value");
            assert_eq!(xpc_dictionary_get_count(f), 1, "and nothing else is added or lost");
            assert!(intact(d), "the original, which libxpc may share with other callers, is unchanged");
            xpc_release(f);
            xpc_release(d);
        }
    }

    #[test]
    fn the_filter_leaves_every_other_answer_as_it_is() {
        for app_sandbox in [None, Some(false)] {
            let d = dict(app_sandbox);
            assert_eq!(filter(d), d, "{app_sandbox:?}: the same object comes back");
            unsafe {
                assert_eq!(xpc_dictionary_get_count(d), 1 + app_sandbox.is_some() as usize, "{app_sandbox:?}: unchanged");
                xpc_release(d);
            }
        }
        assert!(filter(std::ptr::null_mut()).is_null(), "no entitlements stay none");
        let b = unsafe { xpc_bool_create(true) };
        assert_eq!(filter(b), b, "an answer that is not a dictionary comes back as it is");
        unsafe { xpc_release(b) };
    }

    /// The disarm ends the replacement's part. Before it, a read is the replacement's (counted), and an answer holding
    /// the App Sandbox key goes out without it; after it, a read goes straight to libxpc (not counted), and the same
    /// answer goes out as it is. In a fresh copy of the test binary, so no other test sees the disarm.
    #[test]
    fn after_the_disarm_a_read_goes_straight_through() {
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args(["appsandbox::tests::child_reads_before_and_after_the_disarm", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env("MCL_APPSANDBOX_CHILD", "1")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "the child failed: {}\n{text}", out.status);
        // The harness prints the test's name on the same line, before its output.
        let line = text.lines().find_map(|l| l.split_once("DISARM ").map(|(_, v)| v)).unwrap_or_else(|| panic!("no DISARM line in:\n{text}"));
        let n: Vec<u32> = line.split_whitespace().map(|v| v.parse().unwrap()).collect();
        assert_eq!(n.len(), 6, "{line}");
        assert_eq!(n[1], n[0] + 1, "a read before the disarm reached the replacement: {line}");
        assert_eq!(n[2], 1, "before the disarm, an answer holding the key went out as a new dictionary without it: {line}");
        assert_eq!(n[3], 1, "the disarm reports the one answer it filtered: {line}");
        assert_eq!(n[4], n[1] + 1, "a read after the disarm was not counted (the one count after the read is the check's): {line}");
        assert_eq!(n[5], 2, "after the disarm, an answer holding the key went out as the same object, unchanged: {line}");
    }

    #[test]
    #[ignore = "run by after_the_disarm_a_read_goes_straight_through in a process of its own"]
    fn child_reads_before_and_after_the_disarm() {
        if std::env::var_os("MCL_APPSANDBOX_CHILD").is_none() {
            return;
        }
        let read = || unsafe {
            let e = xpc_copy_entitlements_for_self();
            if !e.is_null() {
                xpc_release(e);
            }
        };
        let answer = library_fn(c"maclator_appsandbox_answer").expect("the library is loaded");
        let answer: unsafe extern "C" fn(Obj) -> Obj = unsafe { std::mem::transmute(answer) };
        // The replacement's answer to a real answer holding the key (and OTHER), the original left as it was: 1 if it
        // is a new dictionary without the key and with OTHER, 2 if it is the same dictionary with both keys, 0 if
        // anything else (none, or the original changed).
        let answered = || unsafe {
            let d = dict(Some(true));
            xpc_retain(d); // one reference goes to the replacement, one stays here to look at the original
            let a = answer(d);
            let has = |k: &CStr| xpc_dictionary_get_bool(a, k.as_ptr());
            let none = |k: &CStr| xpc_dictionary_get_value(a, k.as_ptr()).is_null();
            let v = if a.is_null() || !intact(d) {
                0
            } else if a != d && none(KEY) && has(OTHER) && xpc_dictionary_get_count(a) == 1 {
                1
            } else if a == d {
                2
            } else {
                0
            };
            if !a.is_null() {
                xpc_release(a);
            }
            xpc_release(d);
            v
        };
        let before = stats().0;
        read();
        let armed = stats().0;
        let armed_answer = answered();
        let disarm = library_fn(c"maclator_appsandbox_disarm").expect("the library is loaded");
        let disarm: unsafe extern "C" fn() -> libc::c_uint = unsafe { std::mem::transmute(disarm) };
        let filtered = unsafe { disarm() };
        read();
        let after = stats().0;
        let disarmed_answer = answered();
        println!("DISARM {before} {armed} {armed_answer} {filtered} {after} {disarmed_answer}");
    }
}
