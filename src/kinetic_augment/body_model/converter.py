"""
V1 MediaPipe <-> SMPL-X body converter.

Contract:
    world-coordinate body landmarks + per-landmark validity
        -> SMPL-X parameters (body_pose, global_orient, transl, betas)
        -> reconstructed world-coordinate body landmarks

Reconstruction receives only the parameters. Everything needed to rebuild the
landmarks lives in them, so the round trip genuinely passes through SMPL-X
rather than subtracting and re-adding the input.

Fitting runs in an order that keeps the solved quantities from fighting:

  1. Root rotation from the torso: the shoulder axis fixes two degrees of
     freedom, and the nose, when visible, fixes rotation about that axis.
  2. Arm rotations by swing IK, solved *relative to* that root, so each bone
     points along its observed direction in world space.
  3. Translation by least squares over the valid correspondences.

Rotating the body after the arms are solved -- what a whole-body rigid alignment
does -- turns correct bone directions away from the observations, so no rotation
is fitted after step 2. Scale is never fitted: both spaces are metric, and a free
scale absorbs correspondence error into something that looks like a fit.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik, swing_rotation

NUM_POSE_LANDMARKS = 33
NUM_BETAS = 10

NOSE_MP = 0
LEFT_SHOULDER_MP, RIGHT_SHOULDER_MP = 11, 12
LEFT_SHOULDER_SMPLX, RIGHT_SHOULDER_SMPLX = 16, 17

# (SMPL-X joint, MediaPipe pose landmark) for the joints V1 models. Legs are
# excluded because MediaPipe reports visibility near zero for them in
# upper-body-framed video; face and hands are out of V1's scope.
ARM_CORRESPONDENCE = [
    (16, 11),  # left shoulder
    (17, 12),  # right shoulder
    (18, 13),  # left elbow
    (19, 14),  # right elbow
    (20, 15),  # left wrist
    (21, 16),  # right wrist
]


@dataclass
class BodyFit:
    """Genuine SMPL-X parameters; valid input to the real model's forward pass."""

    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    betas: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BETAS))


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(v)
    return None if norm < 1e-12 else v / norm


def _torso_frame(axis: np.ndarray, toward_apex: np.ndarray) -> Optional[np.ndarray]:
    """Right-handed frame: x along the shoulder axis, y toward the apex."""
    up = _unit(toward_apex)
    if up is None:
        return None
    z = np.cross(axis, up)
    norm = np.linalg.norm(z)
    if norm < 1e-9:
        return None
    z = z / norm
    return np.column_stack([axis, np.cross(z, axis), z])


def torso_rotation(
    rest_left: np.ndarray,
    rest_right: np.ndarray,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    rest_apex: Optional[np.ndarray] = None,
    obs_apex: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rotation carrying the rest torso orientation onto the observed one.

    The shoulder axis fixes two of the three rotational degrees of freedom. The
    third, rotation about the shoulder axis, moves neither shoulder. It is taken
    from the apex (the nose) when one is supplied and not degenerate, and
    otherwise from the minimal rotation between the two shoulder axes.

    That third freedom cannot change any reconstructed arm landmark: the arms are
    solved in world directions and hang from shoulders it leaves in place. The
    apex makes the root anatomically plausible without touching round-trip error.
    """
    rest_left, rest_right = np.asarray(rest_left, float), np.asarray(rest_right, float)
    obs_left, obs_right = np.asarray(obs_left, float), np.asarray(obs_right, float)

    axis_rest = _unit(rest_right - rest_left)
    axis_obs = _unit(obs_right - obs_left)
    if axis_rest is None or axis_obs is None:
        raise ValueError("shoulders coincide, so the torso axis is undefined")

    if rest_apex is not None and obs_apex is not None:
        frame_rest = _torso_frame(axis_rest, np.asarray(rest_apex, float)
                                  - 0.5 * (rest_left + rest_right))
        frame_obs = _torso_frame(axis_obs, np.asarray(obs_apex, float)
                                 - 0.5 * (obs_left + obs_right))
        if frame_rest is not None and frame_obs is not None:
            return frame_obs @ frame_rest.T

    return swing_rotation(axis_rest, axis_obs)


def fit_body(landmarks: np.ndarray, validity: np.ndarray, skeleton) -> BodyFit:
    """
    Fit SMPL-X parameters to one frame of world-coordinate body landmarks.

    Args:
        landmarks: (33, 3) MediaPipe world landmarks, metres.
        validity: (33,) bool; False marks a landmark to ignore, e.g. low
            visibility. Invalid landmarks never influence the fit.
        skeleton: SMPL-X skeleton source exposing `rest_nose`.

    Raises:
        ValueError: either shoulder is invalid. The shoulders anchor the torso
            frame; without both, the root rotation is undefined.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    usable = np.asarray(validity, dtype=bool) & np.isfinite(landmarks).all(axis=1)

    if not (usable[LEFT_SHOULDER_MP] and usable[RIGHT_SHOULDER_MP]):
        raise ValueError("both shoulders must be valid: they anchor the torso frame")

    rest = skeleton.rest_joints
    nose_visible = bool(usable[NOSE_MP])
    root = torso_rotation(
        rest[LEFT_SHOULDER_SMPLX], rest[RIGHT_SHOULDER_SMPLX],
        landmarks[LEFT_SHOULDER_MP], landmarks[RIGHT_SHOULDER_MP],
        rest_apex=skeleton.rest_nose if nose_visible else None,
        obs_apex=landmarks[NOSE_MP] if nose_visible else None,
    )

    masked = np.where(usable[:, None], landmarks, np.nan)
    rotations = solve_swing_ik(masked, skeleton, SMPLX_ARM_BONES,
                               global_orient=Rotation.from_matrix(root).as_rotvec())
    posed = forward_kinematics(rotations, skeleton)

    residuals = [landmarks[m] - posed[s] for s, m in ARM_CORRESPONDENCE if usable[m]]

    return BodyFit(
        body_pose=rotations[1:22].reshape(63).copy(),
        global_orient=rotations[0].copy(),
        transl=np.mean(residuals, axis=0),
    )


def reconstruct_body(fit: BodyFit, skeleton) -> np.ndarray:
    """
    Rebuild world-coordinate landmarks from SMPL-X parameters alone.

    Returns (33, 3). Landmarks outside V1's model are NaN rather than invented.
    """
    rotations = np.zeros((skeleton.num_joints, 3))
    rotations[0] = fit.global_orient
    rotations[1:22] = np.asarray(fit.body_pose).reshape(21, 3)

    joints = forward_kinematics(rotations, skeleton) + np.asarray(fit.transl)

    out = np.full((NUM_POSE_LANDMARKS, 3), np.nan)
    for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
        out[mp_idx] = joints[smplx_idx]
    return out
