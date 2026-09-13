# The app — not written yet

Phase 5. The converter currently runs as a binary driven over `adb`
(`../docs/BUILD.md`). This directory is where the Android app goes.

⛔ **Do not start this before the QAIRT redistribution question in `../NOTICE` is
answered.** The app must bundle Qualcomm's runtime to work at all, and if that
cannot be distributed then the app cannot ship in this form — which changes what
gets built, not just how it is licensed.

What the app has to do, once that is settled:

1. Pick a `.safetensors` the user already has.
2. Run the weight stage (~24 s) and the compile (~93 s), with progress — the
   compile peaks around 4.8 GB, so it needs a foreground service and a story for
   8 GB devices.
3. Write a model directory alongside the template's CLIP/VAE that Local Dream or
   DreamUI can load.
