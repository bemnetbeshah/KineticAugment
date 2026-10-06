"""
V1.2 MediaPipe hand landmarks <-> MANO / SMPL-X hand parameters.

The analytic start follows the V1 body converter. MANO's palm is rigid -- the
thumb CMC and the four finger MCPs are children of the wrist -- so a Kabsch
rotation about the wrist over those five points fixes the wrist rotation. Each
finger bone is then swung onto its observed direction with zero twist
(`solve_swing_ik`), parents before children. On a hand the model can represent
exactly this reproduces every joint landmark; only the fingertips, which are
skinned mesh vertices, are approximate.

Contract, per frame and hand:
    21 world landmarks (metres) + per-landmark validity
        -> genuine model parameters (wrist_orient, hand_pose, betas, transl)
        -> 21 landmarks rebuilt from those parameters alone

`fit_hands` refines the analytic start through the real model forward pass
(L-BFGS on the summed squared wrist-relative landmark error), which settles the
fingertips -- skinned mesh vertices the analytic start only approximates -- and
whatever the model cannot represent exactly. `reconstruct_hand` reads nothing
but a fit's parameters and the model.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.hand_model import (
    HAND_PARENTS,
    NUM_HAND_LANDMARKS,
    POSE_SLOT_LANDMARKS,
    WRIST,
)
from kinetic_augment.body_model.swing_ik import solve_swing_ik

PALM_LANDMARKS = (1, 5, 9, 13, 17)

# (driven node, child node, observed parent, observed child): each finger joint's
# rotation aims the bone to its child. Bones leaving the wrist form the palm,
# which `palm_rotation` places instead.
FINGER_BONES = [(int(p), c, int(p), c) for c, p in enumerate(HAND_PARENTS) if p > 0]

FIT_DTYPE = torch.float64
POSE_PRIOR = 1e-4  # mm^2 per rad^2: only settles directions the landmarks leave free
MAX_ITERATIONS = 1000  # dry run: 200 stops short of convergence (0.07 mm); 1000 reaches 0.02 mm
FRAMES_PER_BATCH = 32
FIT_STATUSES = ("ok", "no_wrist", "no_palm", "diverged")


@dataclass
class HandFit:
    """
    Genuine model parameters for one hand in one frame.

    MANO takes them as they are: global_orient=wrist_orient, hand_pose, betas,
    transl. In SMPL-X, wrist_orient is the wrist joint's local rotation in
    body_pose with every other body joint at rest, and hand_pose is
    left_hand_pose or right_hand_pose. Both use the flat hand mean and no PCA.

    `transl` puts the model's wrist on the observed wrist. A frame without a fit
    has its reason in `status` (one of FIT_STATUSES) and NaN parameters.
    """

    backend: str
    side: str
    wrist_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    transl: np.ndarray
    status: str = "ok"
    rms_mm: float = float("nan")


def palm_rotation(rest: np.ndarray, observed: np.ndarray,
                  valid: np.ndarray) -> Optional[np.ndarray]:
    """
    Proper rotation about the wrist best carrying the rest palm onto the observed one.

    Args:
        rest, observed: (21, 3) positions relative to their own wrist.
        valid: (21,) bool.

    Returns:
        (3, 3) rotation, or None when the valid palm points cannot fix one:
        fewer than two of them, or all on one line through the wrist.
    """
    idx = [i for i in PALM_LANDMARKS if valid[i]]
    if len(idx) < 2:
        return None
    u, s, vt = np.linalg.svd(rest[idx].T @ observed[idx])
    if s[1] <= 1e-6 * s[0]:
        return None
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, flip]) @ u.T


def initial_pose(observed, valid, skeleton) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Analytic start for one frame: (wrist_orient (3,), hand_pose (45,)), axis-angle.

    Args:
        observed: (21, 3) landmark positions, any origin.
        valid: (21,) bool; invalid landmarks never influence the result.
        skeleton: rest skeleton from `HandModel.skeleton(betas)`.

    Returns None when the wrist is invalid or the palm cannot fix a rotation.
    """
    observed = np.asarray(observed, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(observed).all(axis=1)
    if not valid[WRIST]:
        return None
    rest = np.asarray(skeleton.rest_joints, dtype=np.float64)
    rotation = palm_rotation(rest - rest[WRIST], observed - observed[WRIST], valid)
    if rotation is None:
        return None
    # Swing IK reads only directions, and takes an all-zero point as missing, so it
    # gets the observations as given rather than wrist-relative.
    masked = np.where(valid[:, None], observed, np.nan)
    local = solve_swing_ik(masked, skeleton, FINGER_BONES,
                           global_orient=Rotation.from_matrix(rotation).as_rotvec())
    return local[WRIST], local[list(POSE_SLOT_LANDMARKS)].reshape(-1)


def node_rotations(wrist_orient, hand_pose) -> np.ndarray:
    """(21, 3) local rotation per landmark node -- the layout `forward_kinematics` takes."""
    nodes = np.zeros((NUM_HAND_LANDMARKS, 3))
    nodes[WRIST] = wrist_orient
    nodes[list(POSE_SLOT_LANDMARKS)] = np.asarray(hand_pose, dtype=np.float64).reshape(-1, 3)
    return nodes


def landmark_errors_mm(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """
    The plan's primary metric: distance in millimetres between predicted and
    observed wrist-relative positions of landmarks 1-20, with no rotation or scale
    registration. Shape (..., 20); NaN where either side is missing.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    gap = (predicted - predicted[..., :1, :]) - (observed - observed[..., :1, :])
    return 1000.0 * np.linalg.norm(gap, axis=-1)[..., 1:]


def _prepared(observed, valid):
    """float64 observations, and validity that also excludes non-finite landmarks."""
    observed = np.asarray(observed, dtype=np.float64)
    return observed, np.asarray(valid, dtype=bool) & np.isfinite(observed).all(axis=-1)


def _unfitted(model, status: str) -> HandFit:
    return HandFit(
        model.backend, model.side, np.full(3, np.nan), np.full(45, np.nan),
        np.full(model.num_betas, np.nan), np.full(3, np.nan), status,
    )


def _rms_mm(residual: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-frame RMS of the masked landmark residuals."""
    squared = (residual ** 2).sum(dim=-1) * mask
    return torch.sqrt(squared.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).numpy()


def _optimise(model, observed, valid, start, betas):
    """
    L-BFGS through the real forward pass on the wrist-relative landmark error (mm).
    The frames are independent and their summed loss separates, so one run fits them all.

    Args:
        observed, valid: (F, 21, 3) and (F, 21) for the frames being fitted.
        start: (F, 48) starting wrist_orient and hand_pose.
        betas: (K,) shape.

    Returns (params (F, 48), rms_mm (F,)).
    """
    kept = np.where(valid[..., None], observed, 0.0)
    target = torch.as_tensor(1000.0 * (kept - kept[:, :1]), dtype=FIT_DTYPE)
    mask = torch.tensor(valid)
    mask[:, WRIST] = False  # zero by construction
    shape = torch.as_tensor(betas, dtype=FIT_DTYPE)
    params = torch.tensor(start, dtype=FIT_DTYPE, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [params], lr=1.0, max_iter=MAX_ITERATIONS, tolerance_grad=1e-12,
        tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe",
    )

    def residuals():
        points = 1000.0 * model.points(params[:, :3], params[:, 3:], shape)
        return (points - points[:, :1]) - target

    def closure():
        optimiser.zero_grad()
        loss = (residuals()[mask] ** 2).sum() + POSE_PRIOR * (params[:, 3:] ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        return params.detach().numpy().copy(), _rms_mm(residuals(), mask)


def fit_hands(observed, valid, model, betas=None) -> List[HandFit]:
    """
    Fit every frame of one hand track independently.

    Args:
        observed: (T, 21, 3) world landmarks of one hand, metres.
        valid: (T, 21) bool; invalid or non-finite landmarks never influence a fit.
        model: `HandModel` of the hand's side.
        betas: (K,) shape used for every frame; neutral when omitted.

    Returns one `HandFit` per frame, in order.
    """
    observed, valid = _prepared(observed, valid)
    betas = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=np.float64)
    skeleton = model.skeleton(betas)

    fits: List[Optional[HandFit]] = [None] * len(observed)
    starts = {}
    for t in range(len(observed)):
        start = initial_pose(observed[t], valid[t], skeleton)
        if start is None:
            fits[t] = _unfitted(model, "no_palm" if valid[t, WRIST] else "no_wrist")
        else:
            starts[t] = np.concatenate(start)

    frames = sorted(starts)
    for first in range(0, len(frames), FRAMES_PER_BATCH):
        batch = frames[first:first + FRAMES_PER_BATCH]
        params, rms = _optimise(
            model, observed[batch], valid[batch], np.stack([starts[t] for t in batch]), betas
        )
        for k, t in enumerate(batch):
            if not (np.isfinite(rms[k]) and np.isfinite(params[k]).all()):
                fits[t] = _unfitted(model, "diverged")
                continue
            fits[t] = HandFit(
                model.backend, model.side, params[k, :3], params[k, 3:], betas.copy(),
                observed[t, WRIST] - skeleton.rest_joints[WRIST], "ok", float(rms[k]),
            )
    return fits


def reconstruct_hand(fit: HandFit, model) -> np.ndarray:
    """
    Rebuild the 21 world landmarks from a fit's parameters alone.

    Returns (21, 3) metres; all NaN when the frame has no fit.

    Raises:
        ValueError: `model` is a different backend or side from the fit's.
    """
    if (fit.backend, fit.side) != (model.backend, model.side):
        raise ValueError(
            f"fit is {fit.backend}/{fit.side} but the model is {model.backend}/{model.side}"
        )
    if fit.status != "ok":
        return np.full((NUM_HAND_LANDMARKS, 3), np.nan)
    with torch.no_grad():
        points = model.points(fit.wrist_orient, fit.hand_pose, fit.betas)[0].numpy()
    return points + fit.transl


def save_hand_fits(path, fits: Sequence[HandFit]) -> None:
    """Write one hand track's fits to an .npz; every fit must share backend and side."""
    tracks = {(f.backend, f.side) for f in fits}
    if len(tracks) != 1:
        raise ValueError(f"one hand track per file, got {sorted(tracks)}")
    (backend, side), = tracks
    np.savez_compressed(
        path,
        backend=np.array(backend),
        side=np.array(side),
        wrist_orient=np.stack([f.wrist_orient for f in fits]),
        hand_pose=np.stack([f.hand_pose for f in fits]),
        betas=np.stack([f.betas for f in fits]),
        transl=np.stack([f.transl for f in fits]),
        status=np.array([f.status for f in fits]),
        rms_mm=np.array([f.rms_mm for f in fits], dtype=np.float64),
    )


def load_hand_fits(path) -> List[HandFit]:
    """Read fits written by `save_hand_fits`."""
    data = np.load(path)
    backend, side = str(data["backend"]), str(data["side"])
    return [
        HandFit(
            backend, side, data["wrist_orient"][t], data["hand_pose"][t], data["betas"][t],
            data["transl"][t], str(data["status"][t]), float(data["rms_mm"][t]),
        )
        for t in range(len(data["status"]))
    ]
