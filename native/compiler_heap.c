// Storage-backed allocations for the SDXL context-generator subprocess.
// Small allocations stay with libc. Bootstrap allocations during dlsym use
// anonymous mappings, tracked separately so they never reach libc's free.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MIB (1024UL * 1024UL)
#define DISK_THRESHOLD (64UL * 1024UL)
#define POOL_MAX MIB

typedef struct Allocation {
    struct Allocation *next;
    void *pointer;
    void *base;
    size_t mapping_size;
    size_t backing_size;
    size_t size;
    int disk;
    int pool_bin;
} Allocation;

static void *(*libc_malloc)(size_t);
static void (*libc_free)(void *);
static void *(*libc_realloc)(void *, size_t);
static int (*libc_posix_memalign)(void **, size_t, size_t);
static size_t (*libc_usable_size)(const void *);
static pthread_once_t initialization = PTHREAD_ONCE_INIT;
static pthread_mutex_t allocations_lock = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local int resolving;
static _Atomic int ready;
static _Atomic unsigned long sequence;
static Allocation *allocations[4096];
// One idle mapping per size class: 64, 128, 256, 512 and 1024 KiB.
// Reuse costs no open/unlink/fallocate/mmap, and retains less than 2 MiB total.
static Allocation *idle_mappings[5];
static Allocation *deferred_frees;
static int directory_fd = -1;
static size_t page_size;
static size_t disk_bytes;
static size_t peak_disk_bytes;
static size_t disk_count;
static time_t last_report;

// All allocation APIs use this index. File mappings are page-aligned, so mix
// the address before selecting a bucket instead of using its zero low bits.
static Allocation **allocation_bucket(const void *pointer) {
    uintptr_t key = (uintptr_t)pointer >> 12;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    return &allocations[key & 4095];
}

static void initialize(void) {
    resolving = 1;
    page_size = (size_t)sysconf(_SC_PAGESIZE);
    // dlsym can allocate while resolving these symbols. Those allocations use
    // mapped_allocate below and remain tracked after initialization finishes.
    libc_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
    libc_malloc = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
    libc_realloc = (void *(*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
    libc_posix_memalign = (int (*)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");
    libc_usable_size = (size_t (*)(const void *))dlsym(RTLD_NEXT, "malloc_usable_size");
    if (!libc_free || !libc_malloc || !libc_realloc || !libc_posix_memalign || !libc_usable_size) {
        static const char message[] = "[compiler heap] cannot resolve libc allocator\n";
        write(STDERR_FILENO, message, sizeof(message) - 1);
        _exit(127);
    }
    const char *directory = getenv("QNN_COMPILER_HEAP_DIR");
    directory_fd = open(directory ? directory : "", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory_fd < 0) {
        static const char message[] = "[compiler heap] cannot open backing directory\n";
        write(STDERR_FILENO, message, sizeof(message) - 1);
        _exit(127);
    }
    // Complete frees made by the linker before libc_free was resolved.
    pthread_mutex_lock(&allocations_lock);
    Allocation *pending = deferred_frees;
    deferred_frees = NULL;
    pthread_mutex_unlock(&allocations_lock);
    while (pending) {
        Allocation *next = pending->next;
        libc_free(pending->pointer);
        munmap(pending, page_size);
        pending = next;
    }
    atomic_store_explicit(&ready, 1, memory_order_release);
    resolving = 0;
    static const char message[] =
        "[compiler heap] active: 64 KiB cutoff, indexed lookup, reusable backing up to 1 MiB\n";
    write(STDERR_FILENO, message, sizeof(message) - 1);
}

// Owns the storage and lifetime metadata shared by malloc and aligned allocation.
static void *mapped_allocate(size_t size, size_t alignment, int disk) {
    size_t bytes = size ? size : 1;
    size_t extra = alignment > page_size ? alignment : 0;
    if (bytes > PTRDIFF_MAX || bytes > SIZE_MAX - (page_size - 1) - extra) {
        errno = ENOMEM;
        return NULL;
    }
    size_t length = (bytes + page_size - 1) & ~(page_size - 1);
    int pool_bin = -1;
    if (disk && alignment <= page_size && bytes <= POOL_MAX) {
        pool_bin = 0;
        length = DISK_THRESHOLD;
        while (length < bytes) {
            length *= 2;
            pool_bin++;
        }
    }
    size_t reservation = length + extra;
    Allocation *entry = NULL;
    int fd = -1;
    void *base = MAP_FAILED;
    void *pointer = MAP_FAILED;
    if (pool_bin >= 0) {
        pthread_mutex_lock(&allocations_lock);
        entry = idle_mappings[pool_bin];
        idle_mappings[pool_bin] = NULL;
        pthread_mutex_unlock(&allocations_lock);
        if (entry) {
            base = entry->base;
            pointer = entry->pointer;
            goto register_allocation;
        }
    }
    entry = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (entry == MAP_FAILED) return NULL;
    if (disk) {
        char name[80];
        snprintf(name, sizeof(name), ".compiler-heap-%d-%lu", getpid(),
                 atomic_fetch_add_explicit(&sequence, 1, memory_order_relaxed));
        fd = openat(directory_fd, name, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) goto failed;
        // The mapping retains the backing inode; even SIGKILL leaves no files.
        if (unlinkat(directory_fd, name, 0) != 0) goto failed;
        // Allocate actual disk space before returning writable memory, so a
        // later page fault cannot discover a sparse file has no backing space.
        if (fallocate(fd, 0, 0, (off_t)length) != 0) goto failed;
    }
    if (extra) {
        base = mmap(NULL, reservation, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) goto failed;
        pointer = (void *)(((uintptr_t)base + alignment - 1) & ~(uintptr_t)(alignment - 1));
        // MAP_FIXED replaces only pages in the reservation owned above.
        pointer = mmap(pointer, length, PROT_READ | PROT_WRITE,
                       MAP_FIXED | (disk ? MAP_SHARED : MAP_PRIVATE | MAP_ANONYMOUS), fd, 0);
    } else {
        base = mmap(NULL, length, PROT_READ | PROT_WRITE,
                    disk ? MAP_SHARED : MAP_PRIVATE | MAP_ANONYMOUS, fd, 0);
        pointer = base;
    }
    if (pointer == MAP_FAILED) goto failed;
    if (fd >= 0) close(fd);

register_allocation:
    entry->pointer = pointer;
    entry->base = base;
    entry->mapping_size = reservation;
    entry->backing_size = length;
    entry->size = size;
    entry->disk = disk;
    entry->pool_bin = pool_bin;
    char report[240];
    int report_length = 0;
    pthread_mutex_lock(&allocations_lock);
    Allocation **bucket = allocation_bucket(pointer);
    entry->next = *bucket;
    *bucket = entry;
    if (disk) {
        disk_bytes += length;
        disk_count++;
        if (disk_bytes > peak_disk_bytes) peak_disk_bytes = disk_bytes;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (size >= 64 * MIB || now.tv_sec - last_report >= 5) {
            last_report = now.tv_sec;
            report_length = snprintf(report, sizeof(report),
                "[compiler heap] allocation %zu KiB; storage-backed %zu MiB live, %zu MiB peak, %zu allocations (not resident RAM)\n",
                size / 1024, disk_bytes / MIB, peak_disk_bytes / MIB, disk_count);
        }
    }
    pthread_mutex_unlock(&allocations_lock);
    if (report_length > 0) write(STDERR_FILENO, report, (size_t)report_length);
    return pointer;

failed:;
    int error = errno;
    if (base != MAP_FAILED) munmap(base, reservation);
    if (fd >= 0) close(fd);
    munmap(entry, page_size);
    if (disk) {
        char report[120];
        int count = snprintf(report, sizeof(report),
            "[compiler heap] backing allocation failed: bytes=%zu errno=%d\n", size, error);
        write(STDERR_FILENO, report, (size_t)count);
    }
    errno = ENOMEM;
    return NULL;
}

void *malloc(size_t size) {
    if (!atomic_load_explicit(&ready, memory_order_acquire)) {
        if (resolving) return mapped_allocate(size, _Alignof(max_align_t), 0);
        pthread_once(&initialization, initialize);
    }
    return size >= DISK_THRESHOLD ? mapped_allocate(size, _Alignof(max_align_t), 1) : libc_malloc(size);
}

void free(void *pointer) {
    if (!pointer) return;
    int saved_errno = errno;
    if (!atomic_load_explicit(&ready, memory_order_acquire) && !resolving)
        pthread_once(&initialization, initialize);
    pthread_mutex_lock(&allocations_lock);
    Allocation **link = allocation_bucket(pointer);
    while (*link && (*link)->pointer != pointer) link = &(*link)->next;
    Allocation *entry = *link;
    int cached = 0;
    if (entry) {
        *link = entry->next;
        if (entry->disk) {
            disk_bytes -= entry->backing_size;
            disk_count--;
            if (entry->pool_bin >= 0 && !idle_mappings[entry->pool_bin]) {
                idle_mappings[entry->pool_bin] = entry;
                cached = 1;
            }
        }
    }
    pthread_mutex_unlock(&allocations_lock);
    if (entry) {
        if (!cached) {
            munmap(entry->base, entry->mapping_size);
            munmap(entry, page_size);
        }
    } else if (libc_free) {
        libc_free(pointer);
    } else {
        // Only reachable during dlsym initialization; no libc ownership guessed.
        Allocation *pending = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pending == MAP_FAILED) _exit(127);
        pending->pointer = pointer;
        pthread_mutex_lock(&allocations_lock);
        pending->next = deferred_frees;
        deferred_frees = pending;
        pthread_mutex_unlock(&allocations_lock);
    }
    errno = saved_errno;
}

size_t malloc_usable_size(const void *pointer) {
    if (!pointer) return 0;
    if (!atomic_load_explicit(&ready, memory_order_acquire) && !resolving)
        pthread_once(&initialization, initialize);
    pthread_mutex_lock(&allocations_lock);
    Allocation *entry = *allocation_bucket(pointer);
    while (entry && entry->pointer != pointer) entry = entry->next;
    size_t size = entry ? entry->size : 0;
    pthread_mutex_unlock(&allocations_lock);
    return entry ? size : libc_usable_size(pointer);
}

void *calloc(size_t count, size_t size) {
    size_t bytes;
    if (__builtin_mul_overflow(count, size, &bytes)) {
        errno = ENOMEM;
        return NULL;
    }
    void *pointer = malloc(bytes);
    // Reused mappings and libc memory need explicit zeroing. Larger mappings
    // are always fresh files, already zero-filled without touching every page.
    if (pointer && bytes <= POOL_MAX)
        memset(pointer, 0, bytes);
    return pointer;
}

void *realloc(void *pointer, size_t size) {
    if (!pointer) return malloc(size);
    if (!size) {
        free(pointer);
        return NULL;
    }
    if (!atomic_load_explicit(&ready, memory_order_acquire) && !resolving)
        pthread_once(&initialization, initialize);
    pthread_mutex_lock(&allocations_lock);
    Allocation *entry = *allocation_bucket(pointer);
    while (entry && entry->pointer != pointer) entry = entry->next;
    size_t old_size = entry ? entry->size : 0;
    pthread_mutex_unlock(&allocations_lock);
    if (!entry && size < DISK_THRESHOLD && libc_realloc) return libc_realloc(pointer, size);
    if (!entry) old_size = libc_usable_size(pointer);
    void *replacement = malloc(size);
    if (replacement) {
        memcpy(replacement, pointer, old_size < size ? old_size : size);
        free(pointer);
    }
    return replacement;
}

void *reallocarray(void *pointer, size_t count, size_t size) {
    size_t bytes;
    if (__builtin_mul_overflow(count, size, &bytes)) {
        errno = ENOMEM;
        return NULL;
    }
    return realloc(pointer, bytes);
}

int posix_memalign(void **result, size_t alignment, size_t size) {
    if (alignment < sizeof(void *) || (alignment & (alignment - 1))) return EINVAL;
    int saved_errno = errno;
    if (!atomic_load_explicit(&ready, memory_order_acquire) && !resolving)
        pthread_once(&initialization, initialize);
    int disk = atomic_load_explicit(&ready, memory_order_acquire);
    if (disk && size < DISK_THRESHOLD && alignment < DISK_THRESHOLD)
        return libc_posix_memalign(result, alignment, size);
    void *pointer = mapped_allocate(size, alignment, disk);
    int error = pointer ? 0 : ENOMEM;
    if (pointer) *result = pointer;
    errno = saved_errno;
    return error;
}

void *memalign(size_t alignment, size_t size) {
    // Bionic accepts power-of-two alignments smaller than sizeof(void *).
    if (!alignment || (alignment & (alignment - 1))) {
        errno = EINVAL;
        return NULL;
    }
    void *pointer;
    int error = posix_memalign(&pointer, alignment < sizeof(void *) ? sizeof(void *) : alignment, size);
    if (error) {
        errno = error;
        return NULL;
    }
    return pointer;
}

void *aligned_alloc(size_t alignment, size_t size) {
    if (!alignment || size % alignment) {
        errno = EINVAL;
        return NULL;
    }
    return memalign(alignment, size);
}

__attribute__((constructor)) static void start_heap(void) {
    pthread_once(&initialization, initialize);
}
