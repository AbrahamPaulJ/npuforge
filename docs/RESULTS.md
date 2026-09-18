# Results and evaluation

Updated: **16 September 2026**. Results below separate measured comparisons from
reported phone outcomes. Historical timings describe the pipeline used at the
time; adding checkpoint-owned CLIP and VAE conversion changes that workload.

## Phone results

| Experiment                             | Configuration                                                        | Result                                                                                       | Scope                                                                                                                     |
|----------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Original SD1.5 UNet conversion         | Galaxy S25 Ultra / SM-S938B, SM8750; 512 × 512; QAIRT 2.49           | 24 s weight conversion + 93 s compilation = **117 s**                                        | Measured original UNet path; excludes today's CLIP/VAE conversion                                                         |
| Original SD1.5 equivalence             | DreamShaper 8, identical reference pack and generation inputs        | Phone pack byte-identical to Python/host pack; **5/5 rendered PNGs identical** to PC compile | Specific reference model and test inputs                                                                                  |
| Earlier SDXL conversion                | Galaxy S25 Ultra; O=3, source-destructive reuse disabled; QAIRT 2.50 | **437 s total conversion**, reported                                                         | Earlier pipeline using shared components; not a current full-pipeline benchmark                                           |
| Generation from that SDXL export       | 1024 × 1024, 8 steps, CFG 1, LCM/Karras                              | **15 s**, reported screenshot                                                                | Generation in the consuming app; one run                                                                                  |
| Pony CLIP diagnostic                   | Pony v6; only seven CLIP-related files replaced                      | Successful output reported; UNet/VAE/tokenizer/markers preserved by hash                     | Isolates the text-encoder replacement for this case                                                                       |
| Full-component Illustrious             | `waiIllustriousSDXL_v170`; 1024 × 1024, 30 steps, CFG 7              | Successful output reported; screenshot shows **45.8 s generation**                           | End-to-end conversion and text-to-image report; no conversion timing or peak memory supplied                              |
| SDXL DMD2 LoRA before component update | nubia NX789J / SM8750; F16-labeled adapter at strength 0.8           | Log records completion in **687.678 s**; F32 success also reported                           | Historical evidence; earlier merger matched 722 UNet modules and lacked full ResNet/sampler coverage                      |
| Current conversion fixes               | `0.2.1-small-pool-lora-test3`                                        | Working conversion confirmed by maintainer after testing                                     | Confirmation lacks a new complete per-device log and timing record; does not establish a device-wide compatibility matrix |

Detailed records: [SD1.5 derivation](PIPELINE.md), [SDXL measurements](SDXL.md),
[component/device investigation](SDXL-INVESTIGATION.md).

The main demonstrated phone has 12 GB installed RAM. Reports from phones with
more RAM exposed separate allocator and workspace failures. Installed RAM alone
is not a compatibility measurement.

## Weight and component comparisons

| Comparison                     | Recorded result                                                                               | Reference                              |
|--------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------|
| DreamShaper 8 weight pack      | Python, host C++ and phone C++ have identical bytes; MD5 `042b99cf38b37cecc15483f37118bb46`   | [Pack comparisons](PIPELINE.md)        |
| AbsoluteReality weight pack    | Python and host C++ have identical bytes; MD5 `52ca4f492e9f43634d08698ea9a5360b`              | [Pack comparisons](PIPELINE.md)        |
| Trimmed vs full SD1.5 template | Identical generated pack; constants template reduced to 43,328 bytes                          | [Template derivation](PIPELINE.md)     |
| SDXL CLIP writer               | All seven native output files match the verified Pony reference by full-file size and SHA-256 | [SDXL components](SDXL-COMPONENTS.md)  |
| SD1.5 CLIP writer              | All three native output files match the reference by size and SHA-256                         | [SD1.5 components](SD15-COMPONENTS.md) |
| VAE weight conversion          | Native packs reproduce reference/QNN-converter weight bytes; both graph interfaces compile    | [VAE templates](VAE-TEMPLATES.md)      |

Hash values here identify historical comparison artifacts; the checkpoints and
large reference artifacts are not distributed in this repository. Host regression
tests instead construct small synthetic fixtures that can be reproduced from a
fresh clone. See [Testing](TESTING.md).

## How to interpret the evidence

**Pack equality** establishes that two converters produced the same weight and
encoding bytes for the tested inputs. It does not establish that a borrowed
activation calibration is suitable for every fine-tune.

**Successful compilation** establishes that the compiler emitted a context.
Generation must still be checked for the intended graph interfaces, scheduler,
conditioning and image quality. Context binaries themselves are not reliably
byte-reproducible across compiler runs.

**Reported renders** establish useful end-to-end examples. They are not a
controlled image-quality study, repeated performance benchmark or guarantee for
all members of a model family. The repository records report metadata without
publishing private tester conversations or unsanitized device logs.

## Reproduce an evaluation

1. Record the app commit/version, QAIRT version, phone model/SoC, Android version,
   installed RAM and available storage. Identify the checkpoint by public model
   revision and SHA-256, and record adapter files and strengths if used.
2. Run conversion with the APK's bundled assets. Save the full troubleshooting
   log, total duration and output ZIP file list. Distinguish component timings,
   resident memory, file backing and virtual mapping counts.
3. For converter changes, compare native weight/component output to the matching
   reference recipe and source weights. Keep compiler configuration fixed.
4. Import the ZIP into [Fancy-Ai](https://github.com/Mr-J-369/Fancy-Ai) or
   [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile), the supported apps. Record
   app/runtime version, prompt, negative prompt, seed, sampler/schedule,
   steps, CFG, resolution and prediction mode.
5. Check text-to-image and, separately, image-to-image. For timing claims, repeat
   trials under recorded thermal conditions and distinguish cold and warm runs.
6. Share a sanitized summary and representative output. Remove personal paths,
   device serials and unrelated user data from any public log.

## Measurements still needed

- Repeated timings and peak resident memory for the current complete SDXL pipeline.
- Fresh reports by device/firmware after the allocator and workspace fixes.
- End-to-end rendering of the expanded SD1.5 component pipeline.
- VAE encoder/image-to-image evaluation on phone.
- Controlled quality comparisons across more fine-tunes and supported LoRA layers.
