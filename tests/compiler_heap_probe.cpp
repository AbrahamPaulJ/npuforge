// Black-box allocator probes. Keep bookkeeping static so tiny-object backing
// cannot be mistaken for a large test-framework allocation.
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <initializer_list>
#include <malloc.h>
#include <new>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static unsigned char pattern(size_t key, size_t offset) {
    return static_cast<unsigned char>((key * 97 + offset * 29) ^ (offset >> 3));
}

static void fill(void* pointer, size_t size, size_t key) {
    auto* bytes = static_cast<unsigned char*>(pointer);
    for (size_t i = 0; i < size; ++i) bytes[i] = pattern(key, i);
}

static void verify(const void* pointer, size_t size, size_t key) {
    const auto* bytes = static_cast<const unsigned char*>(pointer);
    for (size_t i = 0; i < size; ++i)
        require(bytes[i] == pattern(key, i), "allocation contents changed");
}

static void zeroed(const void* pointer, size_t size) {
    require(pointer != nullptr || size == 0, "calloc returned null");
    const auto* bytes = static_cast<const unsigned char*>(pointer);
    for (size_t i = 0; i < size; ++i) require(bytes[i] == 0, "calloc reused nonzero bytes");
}

struct Region { uintptr_t first, last; };
struct MappingSnapshot {
    size_t total = 0, backed = 0;
    Region regions[512]{};
};

static MappingSnapshot mappings() {
    MappingSnapshot result;
    // Raw I/O avoids malloc-backed stdio buffers perturbing the measurement.
    const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "open proc maps");
    char buffer[4096], line[4096];
    size_t used = 0;
    ssize_t count;
    while ((count = read(fd, buffer, sizeof(buffer))) > 0) {
        for (ssize_t i = 0; i < count; ++i) {
            if (buffer[i] == '\n') {
                line[used] = '\0';
                ++result.total;
                if (std::strstr(line, ".compiler-heap-")) {
                    require(result.backed < 512, "too many backing mappings");
                    char* end = nullptr;
                    const uintptr_t first = std::strtoull(line, &end, 16);
                    const uintptr_t last = std::strtoull(end + 1, nullptr, 16);
                    result.regions[result.backed++] = {first, last};
                }
                used = 0;
            } else {
                require(used + 1 < sizeof(line), "proc maps line too long");
                line[used++] = buffer[i];
            }
        }
    }
    require(count == 0, "read proc maps");
    close(fd);
    return result;
}

static bool backed(const MappingSnapshot& snapshot, const void* pointer, size_t size) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    for (size_t i = 0; i < snapshot.backed; ++i) {
        if (address >= snapshot.regions[i].first && address + size <= snapshot.regions[i].last)
            return true;
    }
    return false;
}

static uint32_t random_word(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static constexpr size_t small_count = 300000;
static void* small_objects[small_count];
static uint32_t order[small_count];
static unsigned char replaced[small_count];

static void small() {
    const auto before = mappings();
    for (size_t i = 0; i < small_count; ++i) {
        small_objects[i] = std::malloc(40);
        require(small_objects[i] != nullptr, "tiny malloc");
        require(reinterpret_cast<uintptr_t>(small_objects[i]) % alignof(std::max_align_t) == 0,
                "malloc alignment");
        fill(small_objects[i], 40, i);
        order[i] = static_cast<uint32_t>(i);
    }
    const auto full = mappings();
    require(full.backed < before.backed + 100, "tiny objects consume too many backing mappings");
    require(full.total < before.total + 200, "tiny objects consume too many VMAs");
    for (void* pointer : small_objects)
        require(backed(full, pointer, 40), "tiny object did not use storage-backed heap");

    uint32_t state = 0x71ca1234;
    for (size_t i = small_count - 1; i > 0; --i) {
        const size_t other = random_word(state) % (i + 1);
        const auto value = order[i]; order[i] = order[other]; order[other] = value;
    }
    for (size_t i = 0; i < small_count; i += 2) {
        const auto slot = order[i];
        verify(small_objects[slot], 40, slot);
        std::free(small_objects[slot]);
    }
    for (size_t i = 0; i < small_count; i += 2) {
        const auto slot = order[i];
        small_objects[slot] = std::calloc(5, 8);
        zeroed(small_objects[slot], 40);
        fill(small_objects[slot], 40, slot + small_count);
        replaced[slot] = 1;
    }
    const auto reused = mappings();
    require(reused.backed < before.backed + 100, "tiny reuse causes mapping growth");
    require(reused.total < before.total + 200, "tiny reuse causes VMA growth");
    for (size_t i = 0; i < small_count; ++i) {
        const auto slot = order[i];
        verify(small_objects[slot], 40, slot + replaced[slot] * small_count);
        std::free(small_objects[slot]);
    }
    std::printf("small maps: baseline=%zu full=%zu reused=%zu backed=%zu\n",
                before.total, full.total, reused.total, full.backed);
}

static void boundaries() {
    const size_t sizes[] = {1, 15, 16, 17, 31, 32, 40, 63, 64, 65, 127, 128,
        255, 256, 511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192, 8193,
        1048575, 1048576, 1048577};
    void* pointer = nullptr;
    size_t previous = 0;
    for (size_t size : sizes) {
        void* next = std::realloc(pointer, size);
        require(next != nullptr, "growing realloc");
        verify(next, previous, 91);
        require(malloc_usable_size(next) >= size, "malloc_usable_size too small");
        fill(next, size, 91);
        pointer = next;
        previous = size;
    }
    for (size_t i = sizeof(sizes) / sizeof(*sizes); i > 0; --i) {
        const size_t size = sizes[i - 1];
        void* next = std::realloc(pointer, size);
        require(next != nullptr, "shrinking realloc");
        verify(next, size, 91);
        pointer = next;
    }
    require(std::realloc(pointer, 0) == nullptr, "realloc(p, 0) contract");
    for (size_t size : sizes) {
        auto* dirty = std::malloc(size);
        require(dirty != nullptr, "boundary malloc");
        std::memset(dirty, 0xa5, size);
        std::free(dirty);
        auto* clean = std::calloc(size, 1);
        zeroed(clean, size);
        std::free(clean);
    }
}

static void alignment() {
    for (size_t align : {size_t(8), size_t(16), size_t(32), size_t(64), size_t(256), size_t(4096), size_t(65536)}) {
        for (size_t size : {size_t(0), size_t(40), size_t(48), size_t(8191), size_t(8192), size_t(8193)}) {
            void* pointer = nullptr;
            errno = EDOM;
            require(posix_memalign(&pointer, align, size) == 0, "posix_memalign failed");
            require(errno == EDOM, "posix_memalign changed errno");
            require(reinterpret_cast<uintptr_t>(pointer) % align == 0, "posix_memalign alignment");
            require(pointer != nullptr || size == 0, "aligned null pointer");
            fill(pointer, size, align);
            verify(pointer, size, align);
            std::free(pointer);
        }
        auto* pointer = std::aligned_alloc(align, align * 3);
        require(pointer && reinterpret_cast<uintptr_t>(pointer) % align == 0, "aligned_alloc alignment");
        fill(pointer, align * 3, 19);
        verify(pointer, align * 3, 19);
        std::free(pointer);
    }
    auto* pointer = memalign(1, 40);
    require(pointer != nullptr, "Bionic memalign(1) contract");
    std::free(pointer);
    void* unchanged = reinterpret_cast<void*>(uintptr_t(0x1234));
    errno = ERANGE;
    require(posix_memalign(&unchanged, 3, 40) == EINVAL, "invalid posix alignment accepted");
    require(unchanged == reinterpret_cast<void*>(uintptr_t(0x1234)) && errno == ERANGE,
            "failed posix_memalign changed pointer/errno");
    require(std::aligned_alloc(64, 65) == nullptr && errno == EINVAL, "invalid aligned size accepted");
    require(memalign(3, 40) == nullptr && errno == EINVAL, "invalid memalign accepted");
    std::free(std::malloc(0));
    std::free(std::calloc(0, 40));
    std::free(std::calloc(40, 0));
    std::free(std::realloc(nullptr, 0));
    std::free(nullptr);

    volatile size_t huge = SIZE_MAX;
    require(std::malloc(huge) == nullptr && errno == ENOMEM, "malloc overflow");
    require(std::calloc(huge, 2) == nullptr && errno == ENOMEM, "calloc overflow");
    pointer = std::malloc(40);
    require(pointer != nullptr, "overflow test malloc");
    fill(pointer, 40, 73);
    auto realloc_array = reinterpret_cast<void* (*)(void*, size_t, size_t)>(
        dlsym(RTLD_DEFAULT, "reallocarray"));
    require(realloc_array != nullptr, "reallocarray hook missing");
    require(realloc_array(pointer, huge, 2) == nullptr && errno == ENOMEM, "reallocarray overflow");
    verify(pointer, 40, 73);
    require(std::realloc(pointer, huge) == nullptr && errno == ENOMEM, "realloc overflow");
    verify(pointer, 40, 73);
    errno = EDOM;
    std::free(pointer);
    require(errno == EDOM, "free changed errno");
}

static unsigned handler_calls;
static void allocation_failure() { ++handler_calls; throw std::bad_alloc(); }

static void cpp() {
    struct Object { uint64_t data[5]; };
    struct alignas(256) Aligned { unsigned char data[40]; };
    auto* object = new Object;
    fill(object, sizeof(*object), 42);
    verify(object, sizeof(*object), 42);
    delete object;
    auto* array = new Object[257];
    fill(array, sizeof(Object) * 257, 43);
    verify(array, sizeof(Object) * 257, 43);
    delete[] array;
    auto* aligned = new Aligned;
    require(reinterpret_cast<uintptr_t>(aligned) % alignof(Aligned) == 0, "aligned C++ new");
    fill(aligned, sizeof(*aligned), 44);
    verify(aligned, sizeof(*aligned), 44);
    delete aligned;
    auto* aligned_array = new (std::nothrow) Aligned[5];
    require(aligned_array && reinterpret_cast<uintptr_t>(aligned_array) % alignof(Aligned) == 0,
            "aligned C++ nothrow new[]");
    fill(aligned_array, sizeof(Aligned) * 5, 45);
    verify(aligned_array, sizeof(Aligned) * 5, 45);
    delete[] aligned_array;
    auto* zero = ::operator new(0);
    require(zero != nullptr, "C++ zero new");
    ::operator delete(zero, std::nothrow);
    auto* nothrow = ::operator new[](40, std::nothrow);
    require(nothrow != nullptr, "C++ nothrow new[]");
    ::operator delete[](nothrow, std::nothrow);
    auto* explicit_aligned = ::operator new(40, std::align_val_t(65536), std::nothrow);
    require(explicit_aligned && reinterpret_cast<uintptr_t>(explicit_aligned) % 65536 == 0,
            "explicit aligned nothrow new");
    ::operator delete(explicit_aligned, std::align_val_t(65536), std::nothrow);
    auto* sized = ::operator new[](40, std::align_val_t(64));
    ::operator delete[](sized, size_t(40), std::align_val_t(64));
    volatile size_t huge = SIZE_MAX;
    const auto old_handler = std::set_new_handler(allocation_failure);
    require(::operator new(huge, std::nothrow) == nullptr, "nothrow failure did not return null");
    require(handler_calls == 1, "new_handler was not called");
    bool threw = false;
    try { auto* p = ::operator new[](huge); ::operator delete[](p); }
    catch (const std::bad_alloc&) { threw = true; }
    require(threw && handler_calls == 2, "throwing new failure contract");
    std::set_new_handler(old_handler);
}

static constexpr size_t thread_count = 8, thread_objects = 2048;
static void* shared_objects[thread_count][thread_objects];
static size_t shared_sizes[thread_count][thread_objects];
static pthread_barrier_t barrier;

static void* worker(void* argument) {
    const size_t thread = reinterpret_cast<uintptr_t>(argument);
    const size_t choices[] = {1, 40, 64, 257, 1024, 8191, 8192, 8193};
    for (size_t round = 0; round < 4; ++round) {
        for (size_t i = 0; i < thread_objects; ++i) {
            const size_t size = choices[(i + thread + round) % 8];
            void* pointer;
            if (i % 3 == 0) {
                pointer = std::calloc(size, 1);
                zeroed(pointer, size);
            } else if (i % 3 == 1) {
                pointer = nullptr;
                require(posix_memalign(&pointer, 64, size) == 0, "thread aligned allocation");
            } else pointer = std::malloc(size);
            require(pointer != nullptr, "thread allocation");
            fill(pointer, size, thread * thread_objects + i + round);
            shared_objects[thread][i] = pointer;
            shared_sizes[thread][i] = size;
        }
        pthread_barrier_wait(&barrier);
        // Release another thread's objects to exercise ownership across threads.
        const size_t owner = (thread + 1) % thread_count;
        for (size_t i = 0; i < thread_objects; ++i) {
            void* pointer = shared_objects[owner][i];
            const size_t size = shared_sizes[owner][i];
            const size_t key = owner * thread_objects + i + round;
            verify(pointer, size, key);
            if (i % 5 == 0) {
                void* next = std::realloc(pointer, 32769);
                require(next != nullptr, "thread realloc");
                verify(next, size, key);
                pointer = next;
            }
            std::free(pointer);
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

static void threads() {
    pthread_t workers[thread_count];
    require(pthread_barrier_init(&barrier, nullptr, thread_count) == 0, "barrier init");
    for (size_t i = 0; i < thread_count; ++i)
        require(pthread_create(&workers[i], nullptr, worker, reinterpret_cast<void*>(i)) == 0,
                "pthread_create");
    for (auto thread : workers) require(pthread_join(thread, nullptr) == 0, "pthread_join");
    pthread_barrier_destroy(&barrier);
}

static void foreign() {
    // Models allocations that originate inside libc/loader and arrive at hooks.
    void* libc = dlopen("libc.so.6", RTLD_NOW | RTLD_LOCAL);
    require(libc != nullptr, "open host libc");
    auto libc_malloc = reinterpret_cast<void* (*)(size_t)>(dlsym(libc, "malloc"));
    require(libc_malloc != nullptr, "resolve original libc malloc");
    for (size_t target : {size_t(17), size_t(8192), size_t(32769)}) {
        void* pointer = libc_malloc(73);
        require(pointer != nullptr, "foreign allocation");
        fill(pointer, 73, 51);
        require(malloc_usable_size(pointer) >= 73, "foreign usable size");
        void* next = std::realloc(pointer, target);
        require(next != nullptr, "foreign realloc");
        verify(next, target < 73 ? target : 73, 51);
        std::free(next);
    }
    void* pointer = libc_malloc(40);
    require(pointer != nullptr, "foreign free allocation");
    std::free(pointer);
    dlclose(libc);
}

static void crash() {
    const rlimit no_core{0, 0};
    require(setrlimit(RLIMIT_CORE, &no_core) == 0, "disable core files");
    for (size_t i = 0; i < 512; ++i) {
        small_objects[i] = std::malloc(40);
        require(small_objects[i] != nullptr, "crash probe allocation");
        fill(small_objects[i], 40, i);
    }
    raise(SIGABRT);
    require(false, "SIGABRT did not terminate process");
}

int main(int argc, char** argv) {
    require(argc == 2, "scenario required");
    Dl_info origin{};
    require(dladdr(dlsym(RTLD_DEFAULT, "malloc"), &origin) != 0 &&
            std::strstr(origin.dli_fname, "libcompiler_heap.so"), "malloc hook not preloaded");
    require(dladdr(dlsym(RTLD_DEFAULT, "_Znam"), &origin) != 0 &&
            std::strstr(origin.dli_fname, "libcompiler_heap.so"), "C++ hook not preloaded");
    if (std::strcmp(argv[1], "small") == 0) small();
    else if (std::strcmp(argv[1], "boundaries") == 0) boundaries();
    else if (std::strcmp(argv[1], "alignment") == 0) alignment();
    else if (std::strcmp(argv[1], "cpp") == 0) cpp();
    else if (std::strcmp(argv[1], "threads") == 0) threads();
    else if (std::strcmp(argv[1], "foreign") == 0) foreign();
    else if (std::strcmp(argv[1], "crash") == 0) crash();
    else require(false, "unknown scenario");
    std::printf("%s OK\n", argv[1]);
}
