# Running QNN from inside an app — four traps

Everything before this document ran the converter as `shell` from
`/data/local/tmp`. Moving the identical binaries into an app broke it, and every
failure surfaced as the same useless message:

```
Device Creation failure
QnnDsp <E> Failed to load skel, error: 4000
```

Four separate causes produce that one message. They were found one at a time, by
running the same binary from different places until the difference showed.
Written down because "Device Creation failure" reads like broken hardware.

## 1. `/vendor/lib64` must be on `LD_LIBRARY_PATH`

**This was the actual blocker.** `libQnnHtpV<arch>Stub.so` links against
`libcdsprpc.so`, the vendor FastRPC client, which is **not** in an app's default
linker namespace:

```
dlopen failed: library "libcdsprpc.so" not found:
  needed by .../lib/arm64/libQnnHtpV79Stub.so in namespace (default)
```

As `shell` it resolves and everything works, which is why no earlier test caught
it. The fix:

```
LD_LIBRARY_PATH = <nativeLibraryDir>:/system/lib64:/vendor/lib64:/vendor/lib64/egl
```

⚠ Note it is a **colon** list, while `ADSP_LIBRARY_PATH` is **semicolon**
separated. They are not the same syntax.

## 2. The DSP cannot read most of an app's storage

`ADSP_LIBRARY_PATH` is read by the DSP itself, not by the app, so it must name a
directory the DSP can open. Measured, one at a time:

| location | exec | dlopen | DSP can read |
|---|---|---|---|
| `/data/app/.../lib/arm64` (nativeLibraryDir) | ✅ | ✅ | ❌ |
| `filesDir` | ❌ noexec | ✅ | ✅ |
| `cacheDir` | ❌ | ✅ | ❌ |
| `/sdcard/Android/data/<pkg>` | ❌ noexec | ❌ | ❌ |

So the pieces have to be split: **executables and CPU-side libraries from
`nativeLibraryDir`, hexagon skels copied into `filesDir`.** No single directory
satisfies everything.

## 3. The skels must be world-readable

`File.copyTo` creates `0600`. The DSP runs outside the app's UID, so a `0600`
skel fails exactly like a missing one — right directory, right bytes, still
`error: 4000`. An `adb cp` preserves `0755`, which is what made a hand-run test
pass where the app failed.

```kotlin
out.setReadable(true, /* ownerOnly = */ false)
```

## 4. AGP strips native libraries, which corrupts the skels

By default the Android Gradle Plugin strips everything in `jniLibs`. The
packaged `libQnnHtpV79Skel.so` came out with a **different md5** from the SDK
file — and the **same size**, so a directory listing looked correct.

```kotlin
packaging {
    jniLibs {
        useLegacyPackaging = true          // real files in nativeLibraryDir, so they can be exec'd
        keepDebugSymbols += "**/libQnn*.so"
        keepDebugSymbols += "**/libtplconv.so"
        keepDebugSymbols += "**/libqnncontextgen.so"
    }
}
```

⚠ Verify by md5 against the SDK file, never by size.

## Executables shipped as `lib*.so`

`tplconv` and `qnn-context-binary-generator` are executables, not libraries.
Android refuses to execute anything in the writable app data directory;
`nativeLibraryDir` is the one allowed location. Naming them `libtplconv.so` and
`libqnncontextgen.so` and putting them in `jniLibs` is what gets them there.
`useLegacyPackaging = true` is required or they stay compressed inside the APK.

## How to debug this class of failure

The message never names the cause, so bisect by *location and identity*:

1. Run the same binary as `shell` from `/data/local/tmp`. If that works, the
   binaries are fine and it is an app-context problem.
2. Run it under `run-as <pkg>`. ⚠ `run-as` keeps a different SELinux domain, so
   a pass there does **not** prove an app can do it — it only narrows things.
3. Load a tiny prebuilt context binary to separate "cannot reach the DSP at all"
   from "cannot prepare a graph". The 54 KB fp16 canary is ideal.
   ⚠ Make the probe key off a positive signal. A first attempt used
   `qnn-net-run` with an empty input list, which aborted during argument parsing
   and reported a pass without ever creating a device.
4. md5 every library against its SDK original.

`ProbeService` in `src/debug` does step 3 and prints a verdict. It is in the
debug source set so it never ships.
