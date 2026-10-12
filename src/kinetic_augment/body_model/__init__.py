"""
Body Model Module for KineticAugment.

This module provides SMPL-X integration for parametric human body representation,
enabling proper forward/inverse kinematics for anatomically plausible augmentations.

Components:
    - SMPLXWrapper: Thin wrapper around the SMPL-X model
    - MediaPipeToSMPLX: Fits SMPL-X parameters to MediaPipe landmarks
    - SMPLXToMediaPipe: Projects SMPL-X joints to MediaPipe landmark format
    - joint_mapping: Correspondence tables between SMPL-X and MediaPipe

Example:
    >>> from kinetic_augment.body_model import SMPLXWrapper, MediaPipeToSMPLX
    >>> model = SMPLXWrapper(model_path="models")  # dir containing smplx/
    >>> fitter = MediaPipeToSMPLX(model)
    >>> smplx_params = fitter.fit(mediapipe_landmarks)
"""

# Lazy imports to avoid requiring torch/smplx for basic functionality
def get_smplx_wrapper():
    """Get SMPLXWrapper class (lazy import)."""
    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
    return SMPLXWrapper


def get_mp_to_smplx():
    """Get MediaPipeToSMPLX class (lazy import)."""
    from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX
    return MediaPipeToSMPLX


def get_smplx_to_mp():
    """Get SMPLXToMediaPipe class (lazy import)."""
    from kinetic_augment.body_model.smplx_to_mp import SMPLXToMediaPipe
    return SMPLXToMediaPipe


# Always available
from kinetic_augment.body_model.joint_mapping import (
    SMPLX_TO_MEDIAPIPE_POSE,
    MEDIAPIPE_TO_SMPLX_POSE,
    SMPLX_JOINT_NAMES,
    MEDIAPIPE_POSE_NAMES,
    get_corresponding_joints,
)

__all__ = [
    "get_smplx_wrapper",
    "get_mp_to_smplx",
    "get_smplx_to_mp",
    "SMPLX_TO_MEDIAPIPE_POSE",
    "MEDIAPIPE_TO_SMPLX_POSE",
    "SMPLX_JOINT_NAMES",
    "MEDIAPIPE_POSE_NAMES",
    "get_corresponding_joints",
]
