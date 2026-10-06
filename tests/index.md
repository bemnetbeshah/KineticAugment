# Tests index

- `__init__.py` — Marks the automated test directory as a Python package.
- `test_alignment.py` — Tests least-squares similarity alignment between coordinate spaces.
- `test_body_model.py` — Tests joint mappings, SMPL-X wrapper, and both landmark converters.
- `test_converter.py` — Tests the V1 converter contract and its round trip through SMPL-X.
- `test_kinematics.py` — Tests skeleton sources and forward kinematics against the real SMPL-X model.
- `test_swing_ik.py` — Tests analytic swing inverse kinematics against known synthetic ground truth.
- `hand_assets.py` — Shared model-asset paths, skip marks and a cached hand-model loader for the hand tests.
- `test_hand_model.py` — Tests the 21-point hand operator against the full MANO/SMPL-X forward pass and hand anatomy.
- `test_hand_converter.py` — Tests the hand fit: analytic start, real-model refinement, decoder, serialisation and shape.
- `index.md` — Index of automated test files.
