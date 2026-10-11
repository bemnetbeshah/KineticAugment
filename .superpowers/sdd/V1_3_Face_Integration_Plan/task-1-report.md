# Task 1 implementation report

## Implementation

Added `face_stream.py` with immutable `FaceAttachment`, one weighted Umeyama similarity from
observed `RIGID` landmarks to posed rigid points, and explicit homogeneous transforms for the
pixel-to-world similarity and head-frame-to-world SE(3) pose. All observed rows are transformed,
so mouth, brow, and other nonrigid observations remain intact. Missing point rows remain NaN while
unrelated finite rows are preserved. Missing/zero-weight anchors are excluded; insufficient,
weighted-degenerate, or non-finite head transforms return an invalid all-NaN attachment. Input
shapes and weight values receive explicit `ValueError`s, and rotations must be proper orthonormal
matrices. No coordinate or unit conversion is implicit.

## Tests

- Command: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_face_stream.py`
- Result: 9 passed.
- Command: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_face_stream.py tests/test_alignment.py`
- Result: 21 passed.
- Command: `rtk proxy git diff --check`
- Result: passed with no whitespace errors.

## Files changed

- `src/kinetic_augment/body_model/face_stream.py`
- `tests/test_face_stream.py`
- `src/kinetic_augment/body_model/index.md`
- `tests/index.md`
- This report and `.superpowers/sdd/V1_3_Face_Integration_Plan/index.md`

## Self-review

Reviewed input shape checks, proper-rotation validation, weighted anchor filtering, finite-row
preservation, invalid attachment contents, matrix conventions, and separation of similarity from
head SE(3). The tests cover nonidentity rotation, translation and scale, inverse mapping, preserved
nonrigid changes, rigid reposing, missing data, zero and degenerate anchors, invalid head transforms,
and shape errors. No remaining concerns.

## Commit

The implementation, tests, indexes, and this report are included in the local task commit.
