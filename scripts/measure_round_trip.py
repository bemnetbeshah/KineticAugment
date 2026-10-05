#!/usr/bin/env python3
"""
Measure what the V1 MediaPipe -> SMPL-X -> MediaPipe round trip costs.

Runs the converter's identity round trip with no augmentation in between:

    fit_body(landmarks, validity)  ->  SMPL-X parameters
    reconstruct_body(parameters)   ->  landmarks

Reconstruction sees only the parameters, so this is the error a consumer of the
converter actually gets.

Two error figures are reported and must not be conflated:

  direct   reconstruct_body output vs. input. The converter's own transform is
           rigid, because both spaces are metric. This is the real number.
  aligned  the same output after an *extra* free similarity (scale included) is
           fitted at evaluation time. Diagnostic only: the gap between the two
           shows how much error a free transform would have hidden.

Part A uses synthetic ground truth, where the answer is known exactly. It reports
joint *positions*: twist is unobservable from positions, so rotation recovery is
not claimed.

The noise floor is printed as a reference scale, not an acceptance threshold:
temporal jitter and per-frame reconstruction error are different statistics, and
MediaPipe's default smoothing makes the jitter estimate an underestimate. See
docs/V1_Problem_Scope.md, Section 2.

Usage:
    python scripts/extract_world_landmarks.py   # once
    python scripts/measure_round_trip.py
"""

import glob
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.converter import (
    ARM_CORRESPONDENCE,
    fit_body,
    reconstruct_body,
)
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES

NAMES = {11: "L shoulder", 12: "R shoulder", 13: "L elbow",
         14: "R elbow", 15: "L wrist", 16: "R wrist"}
BONES = [(11, 13, "L upper arm"), (13, 15, "L forearm"),
         (12, 14, "R upper arm"), (14, 16, "R forearm")]
L_SHOULDER, R_SHOULDER = 11, 12
VISIBILITY_THRESHOLD = 0.5
NOISE_FLOOR_REFERENCE_PCT = 0.18


def synthetic_landmarks(skeleton, rotations):
    joints = forward_kinematics(rotations, skeleton)
    landmarks = np.full((33, 3), np.nan)
    for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
        landmarks[mp_idx] = joints[smplx_idx]
    return landmarks, np.isfinite(landmarks).all(axis=1)


def run_synthetic(skeleton, root_deg, trials=200):
    rng = np.random.default_rng(42)
    worst = 0.0
    for _ in range(trials):
        truth = np.zeros((skeleton.num_joints, 3))
        if root_deg:
            axis = rng.normal(size=3)
            truth[0] = axis / np.linalg.norm(axis) * np.deg2rad(root_deg)
        for joint, _, _, _ in SMPLX_ARM_BONES:
            truth[joint] = rng.normal(0, 0.4, 3)
        landmarks, validity = synthetic_landmarks(skeleton, truth)
        out = reconstruct_body(fit_body(landmarks, validity, skeleton), skeleton)
        for _, mp_idx in ARM_CORRESPONDENCE:
            worst = max(worst, np.linalg.norm(out[mp_idx] - landmarks[mp_idx]))
    return worst


def load_clip(path):
    data = np.load(path)
    pose = data["landmarks"].astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (data["visibility"] >= VISIBILITY_THRESHOLD)
    return pose, ok


def run_real(skeleton):
    files = sorted(glob.glob("data/test_output/*_world.npz"))
    if not files:
        raise SystemExit("No world clips. Run scripts/extract_world_landmarks.py first.")

    direct = {m: [] for m in NAMES}
    aligned = {m: [] for m in NAMES}
    coverage = {m: [0, 0] for m in NAMES}
    length = {name: [] for *_, name in BONES}
    per_clip, skipped, frames = {}, 0, 0

    for path in files:
        pose, ok_all = load_clip(path)
        both = ok_all[:, L_SHOULDER] & ok_all[:, R_SHOULDER]
        width = float(np.median(np.linalg.norm(
            pose[both, L_SHOULDER] - pose[both, R_SHOULDER], axis=-1)))
        clip_direct = []

        for frame, ok in zip(pose, ok_all):
            frames += 1
            for m in NAMES:
                coverage[m][1] += 1
                coverage[m][0] += int(ok[m])
            try:
                out = reconstruct_body(fit_body(frame, ok, skeleton), skeleton)
            except ValueError:
                skipped += 1
                continue

            present = [m for m in NAMES if ok[m]]
            for m in present:
                e = np.linalg.norm(out[m] - frame[m]) / width
                direct[m].append(e)
                clip_direct.append(e)

            src = np.array([out[m] for m in present])
            tgt = np.array([frame[m] for m in present])
            realigned = apply_similarity(src, *similarity_transform(src, tgt))
            for m, r, t in zip(present, realigned, tgt):
                aligned[m].append(np.linalg.norm(r - t) / width)

            for a, b, name in BONES:
                if ok[a] and ok[b]:
                    length[name].append(abs(np.linalg.norm(out[b] - out[a])
                                            - np.linalg.norm(frame[b] - frame[a])) / width)

        per_clip[Path(path).stem.replace("_world", "")] = np.median(clip_direct)

    return direct, aligned, coverage, length, per_clip, skipped, frames


def pct(values, q=50):
    return np.percentile(values, q) * 100 if len(values) else float("nan")


def main():
    skeleton = SMPLXSkeleton(model_path="models")

    print("=" * 70)
    print("PART A - synthetic ground truth (joint positions; twist not claimed)")
    print("=" * 70)
    for deg in (0, 30):
        worst = run_synthetic(skeleton, deg)
        status = "exact" if worst < 1e-9 else "NOT exact -- root not yet estimated (Step 8)"
        print(f"  root rotated {deg:>2} deg   worst error {worst:.3e} m   {status}")

    direct, aligned, coverage, length, per_clip, skipped, frames = run_real(skeleton)

    print("\n" + "=" * 70)
    print("PART B - real clips, world landmarks, identity round trip")
    print("=" * 70)
    print(f"  frames: {frames}   skipped (a shoulder invalid, so no torso frame): {skipped}\n")

    print(f"  {'joint':<12}{'coverage':>10}{'direct med':>12}{'direct p95':>12}"
          f"{'aligned med':>13}")
    print("  " + "-" * 58)
    for m, name in NAMES.items():
        cov = 100 * coverage[m][0] / coverage[m][1]
        print(f"  {name:<12}{cov:>9.0f}%{pct(direct[m]):>11.2f}%{pct(direct[m], 95):>11.2f}%"
              f"{pct(aligned[m]):>12.2f}%")

    print(f"\n  {'clip':<10}{'direct median':>16}")
    for clip, value in per_clip.items():
        print(f"  {clip:<10}{value * 100:>15.2f}%")

    print(f"\n  {'bone':<14}{'length error':>14}")
    for *_, name in BONES:
        print(f"  {name:<14}{pct(length[name]):>13.2f}%")

    all_direct = np.concatenate([direct[m] for m in NAMES])
    all_aligned = np.concatenate([aligned[m] for m in NAMES])
    d, a = pct(all_direct), pct(all_aligned)

    print("\n" + "=" * 70)
    print(f"  DIRECT  round-trip error : {d:.2f}% of shoulder width   <- the real number")
    print(f"  ALIGNED round-trip error : {a:.2f}%   (diagnostic; an evaluation-time "
          f"similarity hides {d - a:.2f} pts)")
    print(f"  noise-floor reference    : {NOISE_FLOOR_REFERENCE_PCT:.2f}%   (reference "
          f"scale, NOT a threshold)")
    print("=" * 70)


if __name__ == "__main__":
    main()
