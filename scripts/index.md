# Scripts index

- `diagnose_residual.py` — Diagnoses round-trip error below the rigid-skeleton floor (MediaPipe non-rigidity).
- `demo_smplx_augmentation.py` — Demo of SMPL-X-space augmentation (elbow bend, arm raise, body shape) via fit_body → edit → reconstruct_body; renders videos and measures each edit.
- `download_models.py` — Checks required model assets and prints setup or download guidance.
- `extract_world_landmarks.py` — Extracts metric MediaPipe world landmarks and visibility, one MediaPipe instance per clip.
- `measure_noise_floor.py` — Measures MediaPipe's frame-to-frame noise floor to set the round-trip error budget.
- `measure_round_trip.py` — Measures the shape-fitted identity round trip against the rigid-skeleton floor (the V1 gate), fitted and held out.
- `review_checks/` — Frozen verification scripts and outputs behind `docs/V1_Review_Assessment.md` (not project code).
- `view_round_trip.py` — Renders per-clip videos of the V1 round trip: input vs SMPL-X reconstruction, joints coloured by error, error-over-time strip.
- `render_augmented_from_yaml.py` — Applies YAML-configured augmentations to videos and renders overlays.
- `render_landmarks.py` — Extracts landmarks, reconstructs motion with SMPL-X, and renders comparisons.
- `study_shape_fit.py` — Reproduces the Step 10 shape-fit evidence: prior, identifiability, ladder, and design sweeps.
- `test_phase1.py` — Checks joint mappings, landmark extraction, and MediaPipe-to-SMPL-X conversion.
- `test_phase2.py` — Validates augmentation methods, constraints, and end-to-end augmentation.
- `test_phase3.py` — Exercises collision geometry, detection, resolution, and constraint integration.
- `test_phase4.py` — Checks pipeline configuration, presets, datasets, and legacy compatibility.
- `test_phase5.py` — Runs evaluation metrics, datasets, classifiers, benchmarks, and integration checks.
- `index.md` — Index of setup, rendering, and phase-check scripts.
