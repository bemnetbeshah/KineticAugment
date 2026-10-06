#!/usr/bin/env python3
"""
V1.2 pilot: MediaPipe hand world landmarks -> MANO / SMPL-X -> landmarks.

For every clip and anatomical side, fits each frame's hand, rebuilds its 21
landmarks from the fitted parameters alone, and scores the plan's primary metric:
millimetre distance between predicted and observed wrist-relative positions of
landmarks 1-20, with no rotation or scale registration
(docs/V1_2_Hand_Roundtrip_Plan.md, Section 6).

The ladder separates optimiser error from model mismatch:
    swing start      analytic palm rotation + finger swing, neutral shape
    refined          real-model refinement, neutral shape
    held-out shape   shape fitted on the other half of the track
    in-sample shape  shape fitted on the frames it is scored on (optimistic)
    per-frame shape  every frame its own plausible shape (diagnostic): how much
                     error MediaPipe's frame-to-frame change of hand size causes
    rigid reference  a per-track rigid palm, anchored at the wrist like the fits,
                     and median bone lengths along the observed bone directions;
                     model-free, not a lower bound; also scaled per frame

Also reported per track and backend: how many rotation directions the landmarks
barely constrain (observability), and the error when the other side's model is
used (chirality). Five clips are a pilot, not a held-out test set.

Usage:
    python scripts/extract_hand_landmarks.py   # once
    python scripts/measure_hand_round_trip.py
"""

import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kinetic_augment.body_model.hand_converter import (
    fit_hand_shape,
    fit_hands,
    fit_hands_per_frame_shape,
    initial_pose,
    landmark_errors_mm,
    palm_rotation,
    plausible_radius,
    reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import (
    FINGERS,
    HAND_PARENTS,
    LANDMARK_NAMES,
    POSE_SLOT_LANDMARKS,
    TIPS,
    WRIST,
    HandModel,
)
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip

DATA_DIR = Path("data/test_output")
MODEL_DIR = "models"
OUTPUT = DATA_DIR / "hand_round_trip.json"
BACKENDS = ("mano", "smplx")
TARGETS_MM = {"mean": 5.0, "tips": 8.0, "p95": 15.0}  # plan Section 6, provisional
PALM = [1, 5, 9, 13, 17]  # wrist-relative, so the wrist is the origin
MIDDLE_MCP = 9
OBSERVABILITY_STRIDE = 5
WEAK_FRACTION = 0.01


def tracks():
    """Yield (clip, side, landmarks (n, 21, 3), frames_in_clip) per clip-side, in time order."""
    for path in sorted(DATA_DIR.glob("*_hands.npz")):
        clip = load_hand_clip(path)
        for s, side in enumerate(SIDES):
            frames = np.flatnonzero(clip.present[:, s])
            if len(frames) >= 2:
                landmarks = clip.world[frames, s].astype(np.float64)
                yield path.stem.replace("_hands", ""), side, landmarks, len(clip.present)


def rebuild(fits, model):
    return np.stack([reconstruct_hand(fit, model) for fit in fits])


def start_points(model, landmarks):
    """Landmarks rebuilt from the analytic start alone, wrists placed on the observed ones."""
    skeleton = model.skeleton()
    out = np.full_like(landmarks, np.nan)
    for t, frame in enumerate(landmarks):
        start = initial_pose(frame, np.ones(len(frame), dtype=bool), skeleton)
        if start is not None:
            with torch.no_grad():
                points = model.points(start[0], start[1], np.zeros(model.num_betas))[0].numpy()
            out[t] = points - points[WRIST] + frame[WRIST]
    return out


def rigid_reference(landmarks, per_frame_scale=False, iterations=10):
    """
    A rigid hand built from the track itself, anchored at the wrist like the model fits:
    a wrist-relative palm template (rotation-only Procrustes) and every finger bone at
    its median length along the observed direction. With `per_frame_scale` the whole
    hand is also scaled each frame by its least-squares factor. Model-free; a
    reference, not a lower bound.
    """
    every = np.ones(landmarks.shape[1], dtype=bool)
    relative = landmarks - landmarks[:, :1]
    template = relative[0].copy()
    for _ in range(iterations):
        template = np.mean(
            [frame @ palm_rotation(frame, template, every).T for frame in relative], axis=0)
    rebuilt = np.zeros_like(relative)
    for t, frame in enumerate(relative):
        rebuilt[t, PALM] = template[PALM] @ palm_rotation(template, frame, every).T
    for parent, child in [(int(p), c) for c, p in enumerate(HAND_PARENTS) if p > 0]:
        offset = relative[:, child] - relative[:, parent]
        length = np.median(np.linalg.norm(offset, axis=-1))
        unit = offset / np.linalg.norm(offset, axis=-1, keepdims=True)
        rebuilt[:, child] = rebuilt[:, parent] + length * unit
    if per_frame_scale:
        factor = (rebuilt * relative).sum(axis=(1, 2)) / (rebuilt ** 2).sum(axis=(1, 2))
        rebuilt *= factor[:, None, None]
    return rebuilt + landmarks[:, :1]


def summary(errors_mm, scale_mm):
    """Pooled statistics of (n, 20) errors; `scale_mm` (n,) is each frame's hand scale."""
    finite = np.isfinite(errors_mm)
    tips = [t - 1 for t in TIPS]
    out = {
        "frames": int(np.isfinite(errors_mm).any(axis=1).sum()),
        "mean_mm": float(np.mean(errors_mm[finite])),
        "tips_mm": float(np.nanmean(errors_mm[:, tips])),
        "p95_mm": float(np.percentile(errors_mm[finite], 95)),
        "mean_pct_hand": float(np.nanmean(100.0 * errors_mm / scale_mm[:, None])),
    }
    for name, chain in FINGERS.items():
        out[f"{name}_mm"] = float(np.nanmean(errors_mm[:, [i - 1 for i in chain]]))
    return out


def ladder(model, landmarks):
    """Per-row (n, 20) errors for one track, plus the fitted shapes."""
    valid = np.ones(landmarks.shape[:2], dtype=bool)
    rows = {"swing start": landmark_errors_mm(start_points(model, landmarks), landmarks)}
    neutral = fit_hands(landmarks, valid, model)
    rows["refined"] = landmark_errors_mm(rebuild(neutral, model), landmarks)

    half = len(landmarks) // 2
    held = np.full_like(rows["refined"], np.nan)
    held_norms = []
    halves = (slice(0, half), slice(half, None))
    for fit_on, score_on in (halves, halves[::-1]):
        betas = fit_hand_shape(landmarks[fit_on], valid[fit_on], model)
        held_norms.append(float(np.linalg.norm(betas)))
        fits = fit_hands(landmarks[score_on], valid[score_on], model, betas)
        held[score_on] = landmark_errors_mm(rebuild(fits, model), landmarks[score_on])
    rows["held-out shape"] = held

    in_sample = fit_hand_shape(landmarks, valid, model)
    fits = fit_hands(landmarks, valid, model, in_sample)
    rows["in-sample shape"] = landmark_errors_mm(rebuild(fits, model), landmarks)
    per_frame = fit_hands_per_frame_shape(landmarks, valid, model)
    rows["per-frame shape (diagnostic)"] = landmark_errors_mm(rebuild(per_frame, model), landmarks)
    shapes = {
        "radius": plausible_radius(model.num_betas),
        "held_out_norms": held_norms,
        "in_sample_norm": float(np.linalg.norm(in_sample)),
        "failed_neutral_fits": int(sum(f.status != "ok" for f in neutral)),
    }
    return rows, neutral, shapes


def observability(model, fits):
    """How many of the 48 rotation directions barely move the landmarks; where the weakest is."""
    names = ["wrist"] + [LANDMARK_NAMES[i] for i in POSE_SLOT_LANDMARKS]
    counts, weakest = [], {}
    for fit in [f for f in fits if f.status == "ok"][::OBSERVABILITY_STRIDE]:
        betas = torch.as_tensor(fit.betas, dtype=torch.float64)

        def relative(x):
            points = model.points(x[:3], x[3:], betas)[0]
            return (points[1:] - points[:1]).reshape(-1)

        x0 = torch.as_tensor(np.concatenate([fit.wrist_orient, fit.hand_pose]), dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(relative, x0).numpy()
        _, singular, vt = np.linalg.svd(jacobian)
        counts.append(int((singular < WEAK_FRACTION * singular[0]).sum()))
        joint = names[int(np.argmax(np.linalg.norm(vt[-1].reshape(16, 3), axis=1)))]
        weakest[joint] = weakest.get(joint, 0) + 1
    return {
        "frames": len(counts),
        "median_weak_directions": float(np.median(counts)) if counts else float("nan"),
        "weakest_direction_joint": weakest,
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    assets = [Path(MODEL_DIR) / "mano" / f"MANO_{s}.npz" for s in ("LEFT", "RIGHT")]
    assets.append(Path(MODEL_DIR) / "smplx" / "SMPLX_NEUTRAL.npz")
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "smplx")},
        "assets": {str(p): sha256(p) for p in assets},
        "data": {str(p): sha256(p) for p in sorted(DATA_DIR.glob("*_hands.npz"))},
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'row':<36}{'frames':>7}{'mean':>8}{'tips':>8}{'p95':>8}{'% hand':>8}")
    for name, s in rows.items():
        print(f"  {name:<36}{s['frames']:>7}{s['mean_mm']:>8.2f}{s['tips_mm']:>8.2f}"
              f"{s['p95_mm']:>8.2f}{s['mean_pct_hand']:>8.2f}")


def main():
    began = time.time()
    found = list(tracks())
    if not found:
        raise SystemExit("No data/test_output/*_hands.npz; run scripts/extract_hand_landmarks.py")

    models = {(b, s): HandModel(b, s, model_dir=MODEL_DIR) for b in BACKENDS for s in SIDES}
    pooled_errors, pooled_scale = {}, {}
    report = {"environment": environment(), "targets_mm": TARGETS_MM, "tracks": [], "pooled": {}}

    for clip, side, landmarks, total in found:
        span = np.linalg.norm(landmarks[:, MIDDLE_MCP] - landmarks[:, WRIST], axis=-1)
        scale = np.full(len(landmarks), 1000.0 * np.median(span))
        entry = {"clip": clip, "side": side, "frames_in_clip": int(total),
                 "frames_with_hand": len(landmarks), "hand_scale_mm": float(scale[0]), "rows": {}}
        errors = {
            "rigid reference": landmark_errors_mm(rigid_reference(landmarks), landmarks),
            "rigid reference, per-frame scale": landmark_errors_mm(
                rigid_reference(landmarks, per_frame_scale=True), landmarks),
        }
        other = "right" if side == "left" else "left"
        for backend in BACKENDS:
            model = models[(backend, side)]
            rows, neutral, shapes = ladder(model, landmarks)
            for row, value in rows.items():
                errors[f"{backend}: {row}"] = value
            valid = np.ones(landmarks.shape[:2], dtype=bool)
            wrong = fit_hands(landmarks, valid, models[(backend, other)])
            entry[backend] = {
                "shape": shapes,
                "observability": observability(model, neutral),
                "chirality_mean_mm": {
                    "own_side": float(np.nanmean(rows["refined"])),
                    "other_side": float(np.nanmean(
                        landmark_errors_mm(rebuild(wrong, models[(backend, other)]), landmarks))),
                },
            }
        for row, value in errors.items():
            entry["rows"][row] = summary(value, scale)
            pooled_errors.setdefault(row, []).append(value)
            pooled_scale.setdefault(row, []).append(scale)
        report["tracks"].append(entry)
        print_table(f"{clip} {side}  ({len(landmarks)}/{total} frames, hand "
                    f"{scale[0]:.0f} mm)", entry["rows"])

    for row in pooled_errors:
        report["pooled"][row] = summary(
            np.concatenate(pooled_errors[row]), np.concatenate(pooled_scale[row]))
        report["pooled"][row]["track_mean_mm"] = float(np.mean(
            [t["rows"][row]["mean_mm"] for t in report["tracks"]]))
    print_table("POOLED over all tracks (mm; % of wrist-to-middle-MCP length)", report["pooled"])
    print(f"\nProvisional targets (plan Section 6): mean <= {TARGETS_MM['mean']} mm, "
          f"tips <= {TARGETS_MM['tips']} mm, p95 <= {TARGETS_MM['p95']} mm")

    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUTPUT} in {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
