# Task 4 report: masked temporal facial classifier

## Implementation

- Added `FaceStreamClassifier` with an independent batch-first GRU per configured stream.
  Each stream receives sanitized values concatenated with its per-feature validity mask; masked
  NaN/Inf values are replaced with `torch.where` before entering the encoder. Packed sequences
  exclude padding, and an explicit presence gate zeros an encoder contribution for a fully
  unobserved stream. `stream_dims` configures body-only and multimodal ablations.
- Updated `SLRTrainer` to move nested dictionary batches recursively and compute batch metrics
  from labels, retaining the legacy tensor batch path.
- Exported the classifier and added `examples/train_face_streams.py`. It reads only official
  `train` and `val` splits, defaults to CPU, accepts feature/annotation roots, face space, epochs,
  seed, checkpoint output, and a body-only ablation.
- Added classifier behavior tests and updated the relevant folder indexes. The synthetic
  face-discriminative experiment is wiring evidence only; it is not a recognition benchmark.

## Verification

Command:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_face_classifier.py tests/test_face_training_dataset.py
```

Result: **12 passed in 1.53s**. Full output: `logs/task-4-focused-tests.txt`.

Coverage includes padding invariance, masked nonfinite-value invariance, fully missing stream
gating, finite outputs/gradients with all streams missing, a real optimizer update, optimization
and validation through `collate_face_samples` and `SLRTrainer`, a body-only comparison, and legacy
LSTM trainer/evaluation/prediction batches. The full multimodal synthetic model reached at least
95% validation accuracy and exceeded the body-only model by more than 25 percentage points under
the test's assertions; this demonstrates wiring only.

Command:

```sh
PYTHONPATH=src .venv/bin/python examples/train_face_streams.py --help
```

Result: exit 0; the requested CLI options are shown. Output: `logs/task-4-cli-help.txt`.

Command:

```sh
.venv/bin/black --check src/kinetic_augment/evaluation/models/face_classifier.py tests/test_face_classifier.py examples/train_face_streams.py
```

Result: passed after formatting. `git diff --check` also passed. The existing trainer file was
not run through Black because that would reformat unrelated legacy code.

## Files changed

- `src/kinetic_augment/evaluation/models/face_classifier.py`
- `src/kinetic_augment/evaluation/models/trainer.py`
- `src/kinetic_augment/evaluation/models/__init__.py`
- `src/kinetic_augment/evaluation/models/index.md`
- `src/kinetic_augment/evaluation/__init__.py`
- `src/kinetic_augment/evaluation/index.md`
- `tests/test_face_classifier.py`
- `tests/index.md`
- `examples/train_face_streams.py`
- `examples/index.md`
- `index.md`, `logs/index.md`, and `logs/task-4-*.txt` in this task workspace

## Self-review and limits

No remaining implementation concerns found. Verification used synthetic feature samples and did
not measure real sign-recognition accuracy. The optional checkpoint path and train/validation
workflow are exposed by the CLI; this task did not run on licensed or downloaded assets.
