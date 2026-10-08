"""
Wrist positions relative to the head, recovered from the image (V1.3 task 14).

MediaPipe's pose-world wrists can sit 15-25 cm from the face while the video shows fingers at
the chin (ledger O16). The image carries the missing depth cue: a hand's world landmarks are
metric and its image landmarks are in pixels, so their ratio is the hand's own pixels per metre,
and a pinhole camera turns pixels per metre into depth (`Z = f / scale`). The head's scale comes
from the head projection of task 13. With both depths and both image positions, the wrist can be
placed relative to the nose in camera axes.

Assumptions: pinhole camera, principal point at the image centre, a nominal focal length from a
vertical field of view (MediaPipe's face-geometry default, 63 degrees), a hand and a head that
are each shallow compared with their distance from the camera (weak perspective per object), and
camera axes aligned with the pose-world axes (ledger O11, measured about 1 degree).
"""

from dataclasses import dataclass

import numpy as np

from .alignment import similarity_transform

DEFAULT_VERTICAL_FOV_DEG = 63.0
DEPTH_RATIO_RANGE = (0.5, 1.5)  # plausible Z_hand / Z_head; outside it the correction is skipped
SIDES = 2


def focal_px(height, fov_deg=DEFAULT_VERTICAL_FOV_DEG) -> float:
    """Focal length in pixels for a vertical field of view over an image `height` pixels tall."""
    return float((height / 2.0) / np.tan(np.radians(fov_deg) / 2.0))


def camera_point(u, v, depth, focal, cx, cy) -> np.ndarray:
    """(3,) camera-frame point (x right, y down, z away) of pixel (u, v) at `depth` metres."""
    return np.array([(u - cx) * depth / focal, (v - cy) * depth / focal, depth])


def hand_scale_px_per_m(hand_world, hand_px) -> float:
    """
    Pixels per metre of one hand: the scale of a 2-D similarity from its world landmarks' x, y
    (metres) to its image landmarks (pixels).

    Args:
        hand_world: (21, 3) world landmarks, metres.
        hand_px: (21, 2) the same landmarks in pixels.

    Returns:
        The scale, or NaN when either input has a non-finite value or the fit is degenerate.
    """
    world = np.asarray(hand_world, dtype=np.float64)[:, :2]
    pixels = np.asarray(hand_px, dtype=np.float64)[:, :2]
    if not (np.isfinite(world).all() and np.isfinite(pixels).all()):
        return float("nan")
    zeros = np.zeros((len(world), 1))
    try:
        scale, _, _ = similarity_transform(np.hstack([world, zeros]),
                                           np.hstack([pixels, zeros]), allow_scale=True)
    except ValueError:
        return float("nan")
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


@dataclass
class WristTargets:
    target: np.ndarray       # (2, 3) world wrist targets; NaN where not corrected
    corrected: np.ndarray    # (2,) bool
    depth_ratio: np.ndarray  # (2,) Z_hand / Z_head; NaN where unavailable


def wrist_targets(nose_px, nose_world, head_scale, hand_world, hand_px, width, height,
                  fov_deg=DEFAULT_VERTICAL_FOV_DEG) -> WristTargets:
    """
    World wrist targets from the image: each wrist's offset from the nose, in camera axes, added
    to the nose's world position.

    Args:
        nose_px: (2,) or (3,) image position of the nose tip, pixels.
        nose_world: (3,) world position of the nose tip on the first-pass posed mesh.
        head_scale: pixels per metre at the head (`ProjectedHead.scale`).
        hand_world: (2, 21, 3) world landmarks per side, metres; NaN where absent.
        hand_px: (2, 21, 2) image landmarks per side in pixels; NaN where absent.
        width, height: image size in pixels.

    A hand is left uncorrected (target NaN) when any input it needs is missing or its depth ratio
    `Z_hand / Z_head` falls outside `DEPTH_RATIO_RANGE`; the ratio is still reported when both
    depths exist.
    """
    out = WristTargets(np.full((SIDES, 3), np.nan), np.zeros(SIDES, bool), np.full(SIDES, np.nan))
    nose_px = np.asarray(nose_px, dtype=np.float64)[:2]
    nose_world = np.asarray(nose_world, dtype=np.float64)
    if not (np.isfinite(nose_px).all() and np.isfinite(nose_world).all()
            and np.isfinite(head_scale) and head_scale > 0):
        return out
    focal, cx, cy = focal_px(height, fov_deg), width / 2.0, height / 2.0
    z_head = focal / head_scale
    nose_cam = camera_point(nose_px[0], nose_px[1], z_head, focal, cx, cy)
    for s in range(SIDES):
        scale = hand_scale_px_per_m(hand_world[s], hand_px[s])
        wrist = np.asarray(hand_px[s][0], dtype=np.float64)[:2]
        if not (np.isfinite(scale) and np.isfinite(wrist).all()):
            continue
        z_hand = focal / scale
        out.depth_ratio[s] = z_hand / z_head
        if not DEPTH_RATIO_RANGE[0] <= out.depth_ratio[s] <= DEPTH_RATIO_RANGE[1]:
            continue
        wrist_cam = camera_point(wrist[0], wrist[1], z_hand, focal, cx, cy)
        out.target[s] = nose_world + (wrist_cam - nose_cam)
        out.corrected[s] = True
    return out
