# Contributing

Start with [README.md](README.md), [Build](docs/BUILD.md) and
[Testing](docs/TESTING.md). The host regression suite runs without Qualcomm
binaries, Android tooling or model downloads.

## Changes

- Keep changes focused and explain the concrete behavior they address.
- Preserve working checkpoint and adapter paths. A new rejection condition
  changes compatibility and needs an explicit reason and a representative case.
- For conversion arithmetic, tensor mapping or allocation changes, include a
  small regression case that checks the result. Do not rely on successful graph
  compilation as evidence of correct output.
- Report what you tested, with the command and result. Separate host results
  from phone conversion and rendering results.
- If a user-visible limit changes, update the relevant documentation and the
  app's Info screen together.

Run the relevant tests from [Testing](docs/TESTING.md). For Android changes,
also run the documented build and lint when the required external assets are
available. State any checks you could not run.

## Reporting a conversion problem

Include the app version, phone model/SoC, Android version, checkpoint family,
checkpoint filename, LoRA filenames and strengths, and the stage that failed.
Attach the in-app troubleshooting log. If conversion finishes but rendering
fails, include the consuming app, sampler, prediction mode and generation
settings as well.

Review logs before publishing them: they can contain filenames, paths and device
details. A small synthetic reproducer is preferable to uploading a checkpoint.

## Source and external artifacts

Original source contributions use the repository's [MIT license](LICENSE).
[NOTICE](NOTICE) describes separate third-party and generated-artifact terms.
Do not commit Qualcomm SDK/runtime files, checkpoints, adapters, generated model
libraries, component bundles, credentials or signing keys. Adding a file to
`.gitignore` does not remove it from an existing commit.

No Qualcomm affiliation or endorsement is implied by a contribution or a
successful device test.
