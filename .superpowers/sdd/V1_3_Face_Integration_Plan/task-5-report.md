# Task 5 report: real-clip validation and final results

Status: **DONE**. No production fitting behavior needed a change. The validator and results
document complete Task 5, and the full real-cache run covered every available paired clip.

## Implementation and measurements

Added `scripts/validate_face_integration.py`. It discovers paired face/hand archives, rejects an
output directory inside either input tree, writes versioned feature archives and a JSON report to
the requested separate destination, and records the processed clips, frames, input hashes,
coverage, geometry acceptance and runtime. Defaults are all paired clips in `data/test_output` and
`data/round_trip_eval`; both frame stride and clip limit are explicit and reported. The script also
checks face attachment inverses, exact-fit-context attachment on/off body/hand outputs, invalid
face masking, a controlled producer body-fit failure, corrected-but-unreached masking, and
collated classifier optimization with synthetic labels.

The full real-cache command was:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python scripts/validate_face_integration.py \
  --output-dir /tmp/face-v1.3-validation-full
```

It processed **25/25 clips and 1,646/1,646 frames** in 245.2 seconds. Of 1,550 observed face
frames, 1,549 attached. Body fitting succeeded on 1,643 frames; geometry masks accepted 1,472
hand-frame rows. Zero invalid-face frames had accepted geometry. All 50 input NPZ hashes matched
before and after. Feature archives and detailed JSON were written under `/tmp`; no cache, licensed
asset or video was overwritten. The detailed per-clip table, pixel-unit inversion, arbitrary
rigid re-pose and aggregate metrics are in `docs/V1_3_Face_Integration_Results.md` and
`docs/V1_3_Face_Integration_Evidence.json`.

Round-trip errors were recalculated from the saved per-frame feature archives, so the final totals
use the identical output of the 25-clip run. Across 740,422 finite face points, world reconstruction
error was 9.7e-17 m median, 1.8e-16 m p95 and 4.1e-16 m maximum. Returning through the inverse
pixel transform had 6.4e-14 px median and 7.3e-13 px maximum error. Arbitrary rigid re-pose/inverse
error peaked at 9.3e-17 m.

Attachment on/off was rerun with identical full-clip fitting context for representative pilot
clips, then compared on three geometry-valid rows: 01986 frames 10–12 and 01991 frames 0–2.
Body rotations, translations and hand-to-face distances matched exactly. The script keeps the
full-clip fit context because MANO hand shape can change when only a subset of frames is fitted.

On 01986 frames 14–21, the corrected run marked 16 hands as corrected but unreached; none had
valid geometry. On a controlled copy of 01986 frame 0, `run_clip` failed the body fit while
preserving observed face points. The image-face adapter kept 52 values available and masked all
body/hand values. The 4-sequence, three-stream classifier probe had finite collated tensors and
logits and completed an optimizer update with synthetic labels only.

No usable WLASL or equivalent sign label annotations were found in the project or parent research
workspace. Therefore the task reports **no meaningful real recognition accuracy**.

## Verification

The required full suite command was:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest -q --tb=long
```

Result: 279 passed, 1 xfailed, 3 failures and 3 errors in 88.86 seconds. All six unsuccessful tests
were MediaPipe face/hand/world extraction tests. Their full log showed NSOpenGL failing to create
a pixel format in the sandbox. As allowed by the Task 5 brief, only those extraction tests were
retried with graphics access:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_extract_face_landmarks.py tests/test_extract_hand_landmarks.py \
  tests/test_extract_world_landmarks.py -q --tb=long
```

The graphics-enabled retry passed all 6 tests in 14.63 seconds. The full output is saved in
`task-5-full-suite.log`; the targeted retry is in `task-5-extraction-graphics.log`. The validation
script also passed `py_compile`, a maximum 100-character source-line check, and `git diff --check`.
Black is not installed in this environment.

## Self-review

- The real clips come from the provided observation caches; no fresh video extraction is claimed.
- Exact inverse errors establish transform reversibility, not absolute metric truth or contact-label
  accuracy. The audit and results preserve that distinction.
- The initial three-frame-only attachment comparison changed the clip-fitted hand shape and was
  discarded. Both reported comparisons refit all clip frames with attachment toggled, then compare
  the same finite geometry rows.
- The untracked `data/round_trip_eval` symlink remains intact. No production source was modified.
