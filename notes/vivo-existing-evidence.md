# Vivo compiler failure: findings from existing evidence

Investigation date: 2026-09-15. No conversion, APK build, phone deployment,
or new tester run was performed for this investigation. Runtime code is unchanged.

## Signal sender recovered without another run

Report:
`/home/j/Documents/Testers Logcat/conversion_1789464273853_451fa6ed_e7a2_476e_a5b5_345fcdfaacc6.txt`

Use the report contents, not its filename: this copy starts at 11:23:46 UTC,
identifies Vivo V2307A/SM8650, and ends at 11:31:47 UTC.

Lines 517–526 contain SIGABRT, SI_QUEUE, and
`fault_address=0x00002c26000072f3`. On Android arm64 little-endian, siginfo_t's
si_addr overlaps the queued-signal si_pid/si_uid fields. This value therefore
contains reported sender PID 0x72f3 = 29427 and UID 0x2c26 = 11302. These match
the compiler's PID and UID in the same report. It is not a fault address for
this signal. Sender fields alone are not an authenticated syscall trace.

The earlier claim that normal Bionic abort necessarily produces SI_TKILL was
wrong. Current Bionic abort calls inline_raise, whose implementation fills
SI_QUEUE and invokes rt_tgsigqueueinfo. The existing signal is consistent with
ordinary in-process abort; it does not establish an external OEM watchdog kill.

Sources inspected:
- https://android.googlesource.com/platform/bionic/+/refs/heads/main/libc/bionic/abort.cpp
- https://android.googlesource.com/platform/bionic/+/refs/heads/main/libc/private/bionic_inline_raise.h
- NDK 29 sysroot usr/include/asm-generic/siginfo.h (union layout).

## Mapping failure evidence and its limits

The Vivo report reaches 65,530 total mappings, of which 64,575 are labelled
scudo:secondary and 503 are compiler storage mappings. VmSize falls while
mapping count rises. This strongly implicates mapping exhaustion but does not
identify the failed syscall, requested allocation size, or native caller.
The report has no unwind or original abort message. max_map_count was unreadable.

An older Samsung compiler crash is preserved in:
`/home/j/qnn/npuforge-sdxl-int8/phone-profile/compiler-heap-64k-run/crash.txt`

That crash explicitly reports Scudo internal map failure and a stack through
QNN FancyAllocator::link_blocks / link_source_destructive_operands. It predates
the successful source-destructive configuration change. It must not be used as
the missing Vivo stack or proof that the same QNN stage still fails on Vivo.
It also shows SI_QUEUE for a normal libc abort on Samsung Android 16.

## Aura versus NPUForge, checked against the actual worktrees

Both SDXL htp_config.json files have SHA-256:
`4897c2df1c53aba6eb42aac0912760fdfed88193f558275f108fecf25c91af51`

Both SDXL libqnn_model.so files have SHA-256:
`089b81628ef026628472e90ad826b7401037dde935c3164cf14c4e869c85313c`

Both launch the compiler with LD_PRELOAD and QNN_COMPILER_HEAP_DIR.
The allocator implementations currently differ:
- Aura: 64 KiB cutoff; individual mappings and one idle mapping per size class.
- NPUForge: 8 KiB cutoff; shared 8 MiB slabs through 1 MiB; C++ hooks and crash capture.

NPUForge's newer allocator still failed in the supplied Vivo report. Matching
template/configuration does not prove identical runtime allocation behavior.

## Unresolved cause

Scudo primary exhaustion of delegated small requests is possible, as are
allocation paths not covered by preload interception. Neither is proven by
the existing counters. A 512-byte threshold change, disabling MTE, or declaring
an OEM watchdog responsible would be speculative. Existing evidence cannot
justify claiming a runtime fix. No such changes were made.
