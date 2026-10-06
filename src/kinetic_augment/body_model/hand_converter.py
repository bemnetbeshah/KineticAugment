"""
V1.2 MediaPipe hand landmarks <-> MANO / SMPL-X hand parameters.

The analytic start follows the V1 body converter. MANO's palm is rigid -- the
thumb CMC and the four finger MCPs are children of the wrist -- so a Kabsch
rotation about the wrist over those five points fixes the wrist rotation. Each
finger bone is then swung onto its observed direction with zero twist
(`solve_swing_ik`), parents before children. On a hand the model can represent
exactly this reproduces every joint landmark; only the fingertips, which are
skinned mesh vertices, are approximate.

Refinement and shape fitting follow in later sections of this module.
"""

from typing import Optional, Tuple

import numpy as np
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
