# MANO model index

- `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` — Official MANO v1.2 hand models (licensed, git-ignored; needs chumpy to unpickle).
- `MANO_LEFT.npz`, `MANO_RIGHT.npz` — Chumpy-free copies from `scripts/convert_mano.py`; left includes the shapedirs x-sign fix. Load with `smplx.create("models", model_type="mano", ext="npz")`.
- `SMPL-X__FLAME_vertex_ids.npy` — NumPy vertex-index mapping from the SMPL-X mesh to FLAME.
- `index.md` — Index of MANO-related mapping assets in this folder.
