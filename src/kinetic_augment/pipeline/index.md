# Pipeline index

- `__init__.py` — Exports pipeline configuration, chains, presets, and optional datasets.
- `chain.py` — Composes ordered augmentation steps and builds chains from presets.
- `config.py` — Defines pipeline and constraint configuration objects and YAML loading.
- `dataset.py` — Provides PyTorch datasets and loaders for augmented landmark sequences.
- `pipeline.py` — Applies configured augmentation chains and constraints to motion data.
- `presets.py` — Defines named none, conservative, moderate, and aggressive profiles.
- `index.md` — Index of pipeline configuration, execution, and dataset modules.
