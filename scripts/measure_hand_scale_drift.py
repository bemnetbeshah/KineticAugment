#!/usr/bin/env python3
"""
Prototype: does a per-frame hand scale, smoothed in time, carry MediaPipe's size drift?

MediaPipe's metric hand changes size within a track. This measures, on MANO and both sides:

    in-sample shape            one shape per track, fitted and scored on the same frames
    per-frame scale            that shape plus one uniform scale per frame, fitted jointly
                               (diagnostic: isolates size change; compare per-frame shape)
    per-frame shape            every frame its own 10 shape values (diagnostic, as in V1.2)
    held-out, constant         shape fitted on even frames, scored on odd ones, and back
    held-out, scale sigma=s    shape and per-frame scales fitted on even frames; the scales are
                               smoothed in time (Gaussian, sigma s source frames; 0 = nearest
                               neighbour) and evaluated at the odd frames, which are then fitted
                               at that shape and scale and scored

The held-out rows are the test: a scale curve that predicts frames it never saw is following
slow drift, not frame noise. Every row fits each frame's pose to that frame, as V1.2 does.

Depth: hand_depth turns pixels per metre into depth, Z = f / scale, and pixels per metre is
image size over MediaPipe's metric size. Any spurious change of metric size therefore goes
straight into depth. Per track this reports how the depth series changes when the per-frame
fitted scale is divided out (a constant-size hand) or replaced by its smoothed curve. There is
no depth ground truth: the roughness of depth (frame-to-frame second differences of log Z)
is a proxy, not a measurement of error.

Usage:
    python scripts/measure_hand_scale_drift.py --clip-dir data/test_output \\
        --clip-dir data/round_trip_eval --model-dir models
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from kinetic_augment.body_model.hand_converter import (
    fit_hand_shape,
    fit_hands,
    fit_hands_per_frame_shape,
    landmark_errors_mm,
    reconstruct_hand,
)
from kinetic_augment.body_model.hand_depth import hand_scale_px_per_m
from kinetic_augment.body_model.hand_model import TIPS, WRIST, HandModel
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip
from kinetic_augment.body_model.hand_scale import (
    fit_hand_scales,
    fit_hands_at_scale,
    smooth_log_scale,
)

SIGMAS = (0.0, 1.5, 3.0, 6.0, 12.0)  # source frames
DEPTH_SIGMA = 3.0
MIN_FRAMES = 10
MIDDLE_MCP = 9


def tracks(clip_dirs):
    """(clip, side, world (n, 21, 3), pixels (n, 21, 2), source frames (n,)) per clip-side."""
    for clip_dir in clip_dirs:
        for path in sorted(Path(clip_dir).glob("*_hands.npz")):
            clip = load_hand_clip(path)
            for s, side in enumerate(SIDES):
                frames = np.flatnonzero(clip.present[:, s])
                if len(frames) >= MIN_FRAMES:
                    pixels = clip.image[frames, s, :, :2] * np.array([clip.width, clip.height])
                    yield (path.stem.replace("_hands", ""), side,
                           clip.world[frames, s].astype(np.float64),
                           pixels.astype(np.float64), frames)


def rebuild(fits, model):
    return np.stack([reconstruct_hand(fit, model) for fit in fits])


def summary(errors, hand_mm):
    finite = np.isfinite(errors)
    return {
        "frames": int(finite.any(axis=1).sum()),
        "mean_mm": float(np.mean(errors[finite])),
        "tips_mm": float(np.nanmean(errors[:, [t - 1 for t in TIPS]])),
        "p95_mm": float(np.percentile(errors[finite], 95)),
        "mean_pct_hand": float(np.nanmean(100.0 * errors / hand_mm)),
    }


def held_out(model, landmarks, valid, frames):
    """Rows of (n, 20) errors, every frame scored once by a fit on the other parity."""
    rows = {"held-out, constant": np.full((len(landmarks), 20), np.nan)}
    rows.update({f"held-out, scale sigma={s:g}": np.full((len(landmarks), 20), np.nan)
                 for s in SIGMAS})
    parts = (np.arange(0, len(landmarks), 2), np.arange(1, len(landmarks), 2))
    for fit_on, score_on in (parts, parts[::-1]):
        betas = fit_hand_shape(landmarks[fit_on], valid[fit_on], model)
        fits = fit_hands(landmarks[score_on], valid[score_on], model, betas)
        rows["held-out, constant"][score_on] = landmark_errors_mm(
            rebuild(fits, model), landmarks[score_on])
        betas, scales = fit_hand_scales(landmarks[fit_on], valid[fit_on], model)
        for sigma in SIGMAS:
            predicted = smooth_log_scale(frames[fit_on], scales, sigma, at=frames[score_on])
            _, rebuilt = fit_hands_at_scale(landmarks[score_on], valid[score_on], model,
                                            betas, predicted)
            rows[f"held-out, scale sigma={sigma:g}"][score_on] = landmark_errors_mm(
                rebuilt, landmarks[score_on])
    return rows


def roughness(log_values, frames):
    """RMS second difference over runs of consecutive source frames, as a white-noise sd."""
    steps = [log_values[i - 1] - 2 * log_values[i] + log_values[i + 1]
             for i in range(1, len(frames) - 1)
             if frames[i + 1] - frames[i] == 1 and frames[i] - frames[i - 1] == 1
             and np.isfinite(log_values[i - 1:i + 2]).all()]
    return float(np.sqrt(np.mean(np.square(steps)) / 6.0)) if steps else float("nan")


def depth_effect(world, pixels, frames, scales):
    """How the hand's depth series changes when its metric size is held constant or smoothed."""
    ppm = np.array([hand_scale_px_per_m(w, p) for w, p in zip(world, pixels)])
    log_z = -np.log(ppm)  # log depth, up to the focal length
    log_s = np.log(scales)
    level = np.nanmean(log_s)
    smooth = np.log(smooth_log_scale(frames, scales, DEPTH_SIGMA))
    variants = {
        "mediapipe size": log_z,
        "constant size": log_z - (log_s - level),
        f"smoothed size (sigma={DEPTH_SIGMA:g})": log_z - (log_s - smooth),
    }
    ok = np.isfinite(log_z) & np.isfinite(log_s)
    log_px = (log_s - log_z)[ok]  # image size, up to a constant
    return {
        "frames": int(ok.sum()),
        "size_sd_pct": float(100 * np.nanstd(log_s)),
        "size_max_over_min": float(np.exp(np.nanmax(log_s) - np.nanmin(log_s))),
        "size_vs_image_size_corr": float(np.corrcoef(log_s[ok], log_px)[0, 1]),
        "depth_sd_pct": {k: float(100 * np.nanstd(v)) for k, v in variants.items()},
        "depth_roughness_pct": {k: 100 * roughness(v, frames) for k, v in variants.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip-dir", action="append", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output", default="data/test_output/hand_scale_drift.json")
    args = parser.parse_args()

    began = time.time()
    models = {side: HandModel("mano", side, model_dir=args.model_dir) for side in SIDES}
    pooled, pooled_hand = {}, []
    report = {"settings": {"backend": "mano", "sigmas": SIGMAS, "depth_sigma": DEPTH_SIGMA,
                           "min_frames": MIN_FRAMES, "clip_dirs": args.clip_dir},
              "tracks": []}
    for clip, side, world, pixels, frames in tracks(args.clip_dir):
        model, valid = models[side], np.ones(world.shape[:2], dtype=bool)
        hand_mm = 1000.0 * np.median(np.linalg.norm(world[:, MIDDLE_MCP] - world[:, WRIST], -1))

        shape = fit_hand_shape(world, valid, model)
        rows = {"in-sample shape": landmark_errors_mm(
            rebuild(fit_hands(world, valid, model, shape), model), world)}
        betas, scales = fit_hand_scales(world, valid, model)
        _, rebuilt = fit_hands_at_scale(world, valid, model, betas, scales)
        rows["per-frame scale (diagnostic)"] = landmark_errors_mm(rebuilt, world)
        rows["per-frame shape (diagnostic)"] = landmark_errors_mm(
            rebuild(fit_hands_per_frame_shape(world, valid, model), model), world)
        rows.update(held_out(model, world, valid, frames))

        entry = {"clip": clip, "side": side, "frames": len(frames), "hand_mm": float(hand_mm),
                 "scale_sd_pct": float(100 * np.nanstd(np.log(scales))),
                 "rows": {k: summary(v, hand_mm) for k, v in rows.items()},
                 "depth": depth_effect(world, pixels, frames, scales)}
        report["tracks"].append(entry)
        for k, v in rows.items():
            pooled.setdefault(k, []).append(v)
        pooled_hand.append(np.full(len(frames), hand_mm))
        print(f"{clip} {side:5s} n={len(frames):3d} scale sd {entry['scale_sd_pct']:5.1f}%  "
              + "  ".join(f"{k.split(', ')[-1]}={v['mean_mm']:.2f}"
                          for k, v in entry["rows"].items()), flush=True)

    hand = np.concatenate(pooled_hand)[:, None]
    report["pooled"] = {k: summary(np.concatenate(v), hand) for k, v in pooled.items()}
    report["seconds"] = round(time.time() - began, 1)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print("\npooled (MANO)")
    for k, v in report["pooled"].items():
        print(f"  {k:34s} mean {v['mean_mm']:6.2f}  tips {v['tips_mm']:6.2f}  "
              f"p95 {v['p95_mm']:6.2f}  % hand {v['mean_pct_hand']:6.2f}")


if __name__ == "__main__":
    main()
