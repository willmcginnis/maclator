// The App Sandbox is the translated program's to enter, and it enters it once.
//
// maclator is a dynamically linked x86-64 program, so libSystem's initializer runs in every process it
// translates before the program's own libSystem does. Its libsystem_secinit part reads the process's
// entitlements with xpc_copy_entitlements_for_self(), and when com.apple.security.app-sandbox is true it asks
// secinitd for the container profile and installs it. When a kernel runs maclator as an arm64 program's
// translator inside the program's own process (as macOS runs Rosetta 2), those are the program's entitlements, so
// maclator's own start-up sandboxes the process; then the program's libsecinit asks again, a kernel can refuse a
// second, different profile with EPERM ("forbidden-sandbox-reinit"), and libsecinit then traps ("Could not set
// sandbox profile data"). Natively, and under Rosetta 2, whose runtime is static and runs no libSystem
// initializer, only the program's own start-up asks.
//
// maclator is linked against this library (dyld applies the __interpose section of a dylib, not of the main
// executable), which answers that read without the App Sandbox key until maclator calls
// maclator_appsandbox_disarm() just before the program's first instruction. maclator's libsecinit then takes no
// App Sandbox step, and the program's libsecinit, which runs after the disarm and is translated code reading the
// entitlements through the program's own libxpc, makes the one request, with the program as the main executable,
// as it would natively. From the disarm on, every caller sees the real answer.
//
// So maclator's own start-up runs outside the sandbox, as dyld and Rosetta 2's runtime do natively before the
// program's libsecinit. A run of maclator that starts no program (--aot, --dry-run, an error) ends there; that
// matters only if maclator's own signature carries the App Sandbox key, which it does not as built.
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <xpc/xpc.h>

// libxpc; listed in the SDK's libxpc.tbd, not declared in its headers.
extern xpc_object_t xpc_copy_entitlements_for_self(void);

#define APP_SANDBOX "com.apple.security.app-sandbox"

static atomic_bool disarmed;
static atomic_uint reads, filtered;
static _Atomic(const void *) first_reader;

// `e` (owned; may be NULL) without the App Sandbox key: `e` itself unless the key is true there, else a new
// dictionary holding every other key and value (and `e` released), with `*taken` set. The dictionary is new rather
// than a copy edited in place, so an object libxpc may share with other callers is never changed. If it cannot be
// made, `e` comes back as it is: maclator's start-up then enters the App Sandbox as it did without this library.
static xpc_object_t filter(xpc_object_t e, bool *taken) {
    *taken = false;
    if (e == NULL || xpc_get_type(e) != XPC_TYPE_DICTIONARY || !xpc_dictionary_get_bool(e, APP_SANDBOX)) return e;
    xpc_object_t without = xpc_dictionary_create(NULL, NULL, 0);
    if (without == NULL) return e;
    (void)xpc_dictionary_apply(e, ^bool(const char *key, xpc_object_t value) {
        if (strcmp(key, APP_SANDBOX) != 0) xpc_dictionary_set_value(without, key, value);
        return true;
    });
    xpc_release(e);
    *taken = true;
    return without;
}

xpc_object_t maclator_appsandbox_filter(xpc_object_t e) {
    bool taken;
    return filter(e, &taken);
}

// The replacement's answer to a read whose real answer is `e` (owned), made from `reader`: filtered (and counted)
// before the disarm, `e` itself after it.
static xpc_object_t answer(xpc_object_t e, const void *reader) {
    if (atomic_load(&disarmed)) return e;
    const void *none = NULL;
    atomic_compare_exchange_strong(&first_reader, &none, reader);
    atomic_fetch_add(&reads, 1);
    bool taken;
    xpc_object_t a = filter(e, &taken);
    if (taken) atomic_fetch_add(&filtered, 1);
    return a;
}

static xpc_object_t copy_entitlements_for_self(void) {
    return answer(xpc_copy_entitlements_for_self(), __builtin_return_address(0));
}

// What the replacement would answer now if libxpc's answer were `e` (owned): for tests, which cannot give the
// process the App Sandbox key.
xpc_object_t maclator_appsandbox_answer(xpc_object_t e) {
    return answer(e, __builtin_return_address(0));
}

__attribute__((used, section("__DATA,__interpose"))) static const struct {
    const void *replacement;
    const void *replacee;
} interpose[] = {
    {(const void *)copy_entitlements_for_self, (const void *)xpc_copy_entitlements_for_self},
};

// From now on every caller sees the real answer. Returns how many answers went out without the key.
unsigned maclator_appsandbox_disarm(void) {
    atomic_store(&disarmed, true);
    return atomic_load(&filtered);
}

// What the replacement saw before the disarm: the reads, how many were answered without the key, and the return
// address of the first read (in libsystem_secinit when the start-up's read reached it).
void maclator_appsandbox_stats(unsigned *r, unsigned *f, const void **first) {
    *r = atomic_load(&reads);
    *f = atomic_load(&filtered);
    *first = atomic_load(&first_reader);
}
