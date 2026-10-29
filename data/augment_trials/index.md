# Augmentation trials index

Output of `scripts/augment_face_trials.py` (V1.3: body + hands + observed face). Derived from WLASL footage outside the repo; regenerate rather than commit.

- `<clip>_face_augmentations.mp4` — Source video beside the SMPL-X figure carrying the observed MediaPipe face: round trip, yaw, mirror, scale, uncentre, arm pose, body shape, head turn and time warp, with a metrics panel and a hand-to-face distance strip.
- `augment_face_trials.json` — Augmentation parameters, whether the depth-corrected pass was used, and per-clip measurements (zero-strength identity, face rigidity, face and wrist displacement, hand-to-face distances).
- `index.md` — This index.
