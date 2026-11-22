#!/usr/bin/env python3
"""
Prototype: does holding MediaPipe's metric hand at constant size change depth-corrected contact?

`hand_depth.wrist_targets` reads a hand's depth from its pixels per metre, image size over
MediaPipe's metric size, so a spurious change of metric size moves the hand in depth. This runs
the depth-corrected face pipeline twice per clip: on MediaPipe's hands as they are, and on hands
rescaled about their centroid to a constant size (each frame divided by its fitted MANO scale,
`hand_scale.fit_hand_scales`, relative to the track's geometric mean).

Reported over hand-near-face frames: how many frames get a depth-corrected wrist, how many of
those the arm reaches, contact (any hand point within the contact band of the head surface),
the closest hand-head distance, and the frame-to-frame roughness of log(Z_hand / Z_head).
There is no contact ground truth except where a clip is known to touch throughout (08925:
fingers on the cheek in every frame).

Usage:
    python scripts/measure_hand_size_contact.py --clip-dir data/round_trip_eval
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from kinetic_augment.body_model.face_observations import load_face_clip
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip
from kinetic_augment.body_model.hand_scale import fit_hand_scales

MIN_FRAMES = 10


def constant_size(hands, mano):
    """A copy of `hands` with every track rescaled about its centroid to constant size."""
    out = copy.deepcopy(hands)
    for s, side in enumerate(SIDES):
        frames = np.flatnonzero(hands.present[:, s])
        if len(frames) < MIN_FRAMES:
            continue
        world = hands.world[frames, s].astype(np.float64)
        _, scales = fit_hand_scales(world, np.isfinite(world).all(-1), mano[side])
        factor = np.exp(np.nanmean(np.log(scales))) / scales
        factor = np.where(np.isfinite(factor), factor, 1.0)
        centre = world.mean(axis=1, keepdims=True)
        out.world[frames, s] = (centre + (world - centre) * factor[:, None, None]).astype(
            hands.world.dtype)
    return out


def roughness(series):
    """Per hand, RMS second difference of a (T,) log series over finite triples, pooled."""
    steps = []
    for x in series:
        d = x[:-2] - 2 * x[1:-1] + x[2:]
        steps.extend(d[np.isfinite(d)])
    return float(np.sqrt(np.mean(np.square(steps)) / 6.0)) if steps else float("nan")


def measure(result):
    near = result.hand_near_face
    contact, closest = [], []
    for i in np.flatnonzero(near):
        g = result.geometry[i]
        if g is not None and np.isfinite(g.distance).any():
            contact.append(float(g.contact.any()))
            closest.append(float(np.nanmin(g.distance)))
    corrected = result.wrist_corrected[near].any(axis=1)
    reached = result.wrist_reached[near].any(axis=1)
    log_ratio = np.log(result.depth_ratio)
    return {
        "frames": int(len(near)),
        "near_face": int(near.sum()),
        "corrected": int(corrected.sum()),
        "reached": int(reached.sum()),
        "contact": int(np.sum(contact)),
        "posed": len(contact),
        "closest_median_m": float(np.median(closest)) if closest else None,
        "all_frames_contact": int(sum(g is not None and np.isfinite(g.distance).any()
                                      and bool(g.contact.any()) for g in result.geometry)),
        "depth_ratio_roughness_pct": 100 * roughness(log_ratio.T),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip-dir", action="append", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output", default="data/test_output/hand_size_contact.json")
    args = parser.parse_args()

    models = load_face_models(args.model_dir)
    report = {"clips": {}}
    for clip_dir in map(Path, args.clip_dir):
        for face_path in sorted(clip_dir.glob("*_face.npz")):
            clip = face_path.name[: -len("_face.npz")]
            hands = load_hand_clip(clip_dir / f"{clip}_hands.npz")
            face = load_face_clip(face_path)
            entry = {}
            for name, variant in (("mediapipe size", hands),
                                  ("constant size", constant_size(hands, models.mano))):
                result = run_clip(variant, face, models, depth_corrected=True,
                                  attach_face_stream=False)
                entry[name] = measure(result)
            report["clips"][clip] = entry
            print(clip, json.dumps(entry), flush=True)

    keys = ("near_face", "corrected", "reached", "contact", "posed", "all_frames_contact")
    report["pooled"] = {
        name: {k: int(sum(c[name][k] for c in report["clips"].values())) for k in keys}
        for name in ("mediapipe size", "constant size")}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
