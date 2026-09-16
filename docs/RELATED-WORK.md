# Related work and the on-device conversion claim

Review date: **16 September 2026**.

npuforge's specific contribution is an Android application that accepts an SDXL
checkpoint, converts its weights and compiles QNN context binaries **on the
phone**, then exports a package for Local Dream or Fancy-Ai. This includes
checkpoint-owned CLIP graphs in MNN format and QNN VAE encoder/decoder graphs.
Templates are prepared in advance; conversion of each selected checkpoint does
not need a workstation or cloud compiler.

## Public workflows reviewed

| Project | Documented workflow | Where conversion happens |
|---|---|---|
| [Local Dream](https://ld.chino.icu/conversion/) | Custom SD1.5/SDXL checkpoints are converted to QNN packages, then imported into the Android app | Its conversion guide explicitly requires a Linux or WSL host |
| [Model-to-NPU](https://github.com/VitalikDen0/Model-To-NPU/blob/main/README_EN.md) | Export checkpoint components, convert to QNN, build Android model libraries, deploy and generate on the phone | The build/export pipeline precedes deployment; the APK is the mobile generation interface |
| [TokForge SDXL QNN model](https://huggingface.co/darkmaniac7/TokForge-SDXL-QNN-NPU) | A precompiled RealVisXL Lightning QNN package for the TokForge Android app | The model card documents consumption of compiled contexts, not an in-app checkpoint converter |

These projects demonstrate relevant mobile inference and deployment work. Their
linked documentation does not describe the same checkpoint-to-QNN conversion
workflow running entirely inside an Android app.

## Search scope and conclusion

The review searched public web results and project documentation for combinations
of SDXL, QNN, Android, checkpoint conversion, on-device conversion, on-phone
conversion and mobile converter. The comparison above uses the projects' own
published descriptions rather than third-party summaries.

**No other publicly documented Android app with this capability was found in
this review.** This supports the project's qualified claim: “To our knowledge,
the first publicly documented Android app to convert SDXL checkpoints into QNN
models entirely on-device.” It is a dated literature/project search, not proof
that an unpublished or unindexed implementation does not exist.

A reproducible counterexample should show a phone taking an SDXL checkpoint and
producing new QNN contexts locally. Downloading precompiled models, generating
images on the NPU, importing an ONNX graph, or invoking a remote conversion
service establishes a different capability. Related implementations and earlier
work can be submitted through the repository's issues.
