# Scripts index

- `download_models.py` — Checks required model assets and prints setup or download guidance.
- `extract_hand_landmarks.py` — Extracts MediaPipe hand world and image landmarks with the paired pose and a geometric left/right assignment.
- `extract_world_landmarks.py` — Extracts metric MediaPipe world landmarks and visibility for body fitting.
- `measure_noise_floor.py` — Measures MediaPipe's frame-to-frame noise floor to set the round-trip error budget.
- `measure_round_trip.py` — Measures identity round-trip error and decomposes it by cause.
- `render_augmented_from_yaml.py` — Applies YAML-configured augmentations to videos and renders overlays.
- `render_landmarks.py` — Extracts landmarks, reconstructs motion with SMPL-X, and renders comparisons.
- `test_phase1.py` — Checks joint mappings, landmark extraction, and MediaPipe-to-SMPL-X conversion.
- `test_phase2.py` — Validates augmentation methods, constraints, and end-to-end augmentation.
- `test_phase3.py` — Exercises collision geometry, detection, resolution, and constraint integration.
- `test_phase4.py` — Checks pipeline configuration, presets, datasets, and legacy compatibility.
- `test_phase5.py` — Runs evaluation metrics, datasets, classifiers, benchmarks, and integration checks.
- `index.md` — Index of setup, rendering, and phase-check scripts.
