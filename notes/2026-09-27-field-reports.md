# 2026-09-27 — field reports, 1.0.4 preparation

Outcome: two field reports explained and addressed, three wishlist items built,
release 1.0.4 prepared (committed, signed APK built, not published).
Measurements live in `docs/LIMITS.md`, `docs/SD15-INPAINT.md` and
`docs/CLIP-COMPONENTS.md`; this note records the process and what went wrong.

## Reports and what they turned out to be

- **China: "the in-app download stops after a little, dozens of times."** The
  developer suspected the inpaint difference. More likely the SDXL VAE: the app
  offered it at every launch, and its downloader had no read timeout, no retry
  and no resume, so every stall restarted from zero. Both downloads now share
  `HfDownload` (resume, huggingface.co ⇄ hf-mirror.com, a source setting,
  China-timezone default) and each has an offline import.
- **8 GB phone: inpaint conversion killed 5/5, plain SD1.5 fine.** The user
  concluded the 9-channel graph needs more compile RAM. Their two reports show
  identical memory profiles; both compiles peak near 4.3 GB anonymous memory,
  and whether Android's killer fires at 0.6 GB free is luck. The phone is an
  SM8450 (8 Gen 1), which is unsupported anyway.
- **8 Gen 2 limit.** Asked whether Mr.J introduced it: no. The first commits
  (14 September) already targeted v73 and said 8 Gen 1/888 cannot load the
  graphs. Mr.J's v1.0.3 only removed the V68/V69 libraries used while
  converting. Decision (user): keep them out.

## Decisions (user)

- Low-RAM mode (storage-backed allocator for the SD1.5 UNet below 10 GiB) over
  always-on or a warning.
- Wishlist: other file managers in the picker, lower storage, clip skip 1/2
  with a `_cs1` name suffix. Export stays automatic, with no separate Export
  step. Other resolutions are left for Mr.J.
- Download source as a Utility-tab setting, like Nightmare; the SDXL VAE is
  asked for only when an SDXL conversion starts.
- Prepare 1.0.4; the user publishes.

## What was wrong, and why

- **The local tree was 5 commits behind `origin/main`** (Mr.J's v1.0.2/1.0.3)
  under a large uncommitted change. Fast-forwarded with a stash; no conflicts.
- **Docs claimed conversions never download VAE weights**, while SDXL
  downloads Mr.J's VAE contexts. LIMITS/BUILD/HANDOFF/CLAUDE were corrected.
  CLAUDE/HANDOFF also still called BF16 unsupported after BF16 support landed.
- **UI automation tapped into the user's Instagram.** A `uiautomator dump`
  was stale while the user was using the phone; three taps landed in another
  app. Take a screenshot before every tap, and ask before driving the phone.
- Self-inflicted shell failures: a stray `cat >` waited on stdin until timeout;
  `python - <<EOF < /dev/null` ran nothing, because the redirect replaced the
  heredoc; `sh -c` quoting through PowerShell broke an awk one-liner. Script
  files fixed all three.
- The inpaint-diff parity test failed until the venv was activated; it shells
  out to a bare `python3` (docs/TESTING.md says so).
- NDK 30 was not installed and there were no SDK command-line tools; the NDK was
  fetched from Google's repository index (SHA-1 checked) and unpacked by hand.
