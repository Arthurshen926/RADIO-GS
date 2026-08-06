#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

/*
 * A deliberately tiny audit interposer for frozen NVOS evaluation.  Only the
 * exact target-mask path named by RADIO_GS_GATED_TARGET_PATH is delayed.  The
 * evaluator remains byte-for-byte unchanged; a supervisor seals its primitive
 * and rendered-score receipt, then creates RADIO_GS_GT_RELEASE_PATH.
 */

static bool is_gated_path(const char *path) {
    const char *target = getenv("RADIO_GS_GATED_TARGET_PATH");
    return path != NULL && target != NULL && target[0] != '\0' &&
           strcmp(path, target) == 0;
}

static void mark_and_wait(void) {
    const char *marker = getenv("RADIO_GS_GT_BLOCKED_MARKER_PATH");
    const char *release = getenv("RADIO_GS_GT_RELEASE_PATH");
    if (marker == NULL || marker[0] == '\0' ||
        release == NULL || release[0] == '\0') {
        _exit(125);
    }
    int fd = (int)syscall(SYS_openat, AT_FDCWD, marker,
                          O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) {
        (void)syscall(SYS_close, fd);
    }
    while (syscall(SYS_faccessat, AT_FDCWD, release, F_OK, 0) != 0) {
        usleep(10000);
    }
}

static int open_impl(const char *symbol, const char *path, int flags,
                     mode_t mode, bool has_mode) {
    typedef int (*open_fn)(const char *, int, ...);
    open_fn real_open = (open_fn)dlsym(RTLD_NEXT, symbol);
    if (real_open == NULL) {
        errno = ENOSYS;
        return -1;
    }
    if (is_gated_path(path)) {
        mark_and_wait();
    }
    return has_mode ? real_open(path, flags, mode) : real_open(path, flags);
}

int open(const char *path, int flags, ...) {
    mode_t mode = 0;
    bool has_mode = (flags & O_CREAT) != 0;
    if (has_mode) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
    }
    return open_impl("open", path, flags, mode, has_mode);
}

int open64(const char *path, int flags, ...) {
    mode_t mode = 0;
    bool has_mode = (flags & O_CREAT) != 0;
    if (has_mode) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
    }
    return open_impl("open64", path, flags, mode, has_mode);
}

static int openat_impl(const char *symbol, int dirfd, const char *path,
                       int flags, mode_t mode, bool has_mode) {
    typedef int (*openat_fn)(int, const char *, int, ...);
    openat_fn real_openat = (openat_fn)dlsym(RTLD_NEXT, symbol);
    if (real_openat == NULL) {
        errno = ENOSYS;
        return -1;
    }
    if (is_gated_path(path)) {
        mark_and_wait();
    }
    return has_mode ? real_openat(dirfd, path, flags, mode)
                    : real_openat(dirfd, path, flags);
}

int openat(int dirfd, const char *path, int flags, ...) {
    mode_t mode = 0;
    bool has_mode = (flags & O_CREAT) != 0;
    if (has_mode) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
    }
    return openat_impl("openat", dirfd, path, flags, mode, has_mode);
}

int openat64(int dirfd, const char *path, int flags, ...) {
    mode_t mode = 0;
    bool has_mode = (flags & O_CREAT) != 0;
    if (has_mode) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
    }
    return openat_impl("openat64", dirfd, path, flags, mode, has_mode);
}
