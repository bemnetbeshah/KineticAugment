"""
SMPL-X neck, head and jaw from MediaPipe FaceLandmarker landmarks (V1.3 design §4).

Head orientation: a weighted Umeyama fit from the 17 rigid face points on the SMPL-X rest mesh to
the observed points in pixels (x·W, y·H, z·W). At rest every SMPL-X rotation is the identity, so
the fitted rotation is the head's global rotation; scale and translation are discarded. Rigid
points under either hand's padded image outline get zero weight, and one robust pass drops
outliers, because hands over the face are the failure mode prior sign-language work reports.

Using the rotation in the pose world frame assumes the pixel axes match the pose world axes
(assumption F2): both are x right, y down, z away from the camera.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import ConvexHull, QhullError

from .alignment import apply_similarity, similarity_transform

OCCLUSION_PAD = 0.15     # padding around a hand outline, fraction of the outer-eye-corner distance
OUTLIER_FACTOR = 2.5     # residual above this many weighted medians is dropped
RESIDUAL_FLOOR = 0.01    # ... but never below this fraction of the outer-eye-corner distance
MIN_RIGID_POINTS = 8     # fewer surviving rigid points: the frame is unreliable
MAX_RMS_FRACTION = 0.10  # provisional; revisited once the floor (metric 1) is measured


def hand_outlines(hand_image: np.ndarray, width: int, height: int) -> List[np.ndarray]:
    """
    Each present hand's convex outline in pixels.

    Args:
        hand_image: (2, 21, 3) one frame of `HandClip.image` (x / width, y / height, z); NaN
            for an absent hand.
    """
    outlines = []
    for side in range(hand_image.shape[0]):
        points = hand_image[side, :, :2].astype(np.float64) * [width, height]
        if not np.isfinite(points).all():
            continue
        try:
            outlines.append(points[ConvexHull(points).vertices])
        except QhullError:  # degenerate (collinear) detection: fall back to its bounding box
            lo, hi = points.min(axis=0), points.max(axis=0)
            outlines.append(np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]))
    return outlines


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def occluded(points: np.ndarray, outlines: List[np.ndarray], pad: float) -> np.ndarray:
    """(N,) True where a 2-D point is inside an outline or within `pad` pixels of one."""
    points = np.asarray(points, dtype=np.float64)
    hidden = np.zeros(len(points), dtype=bool)
    for outline in outlines:
        inside = Polygon(outline).contains_points(points)
        edge = np.min([_segment_distance(points, outline[i], outline[(i + 1) % len(outline)])
                       for i in range(len(outline))], axis=0)
        hidden |= inside | (edge <= pad)
    return hidden


@dataclass
class HeadRotation:
    rotation: np.ndarray  # (3, 3) head's global rotation in camera axes; NaN if not fitted
    rms_fraction: float   # weighted RMS residual / reference_length
    points_used: int      # points with non-zero final weight
    weights: np.ndarray   # (N,) final weights after the outlier pass


def head_rotation(template, observed, weights, reference_length: float) -> HeadRotation:
    """
    Head rotation from rigid points, robust to occlusion and a few outliers.

    Args:
        template: (N, 3) the rigid points on the SMPL-X rest mesh, metres.
        observed: (N, 3) the same points observed, in pixels.
        weights: (N,) initial weights; 0 for occluded points.
        reference_length: the observed outer-eye-corner distance, pixels.
    """
    template = np.asarray(template, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    if (w > 0).sum() < 3:
        return HeadRotation(np.full((3, 3), np.nan), float("nan"), int((w > 0).sum()), w)

    def fit(w):
        s, r, t = similarity_transform(template, observed, weights=w)
        return r, np.linalg.norm(apply_similarity(template, s, r, t) - observed, axis=1)

    rotation, residual = fit(w)
    cut = max(OUTLIER_FACTOR * float(np.median(residual[w > 0])),
              RESIDUAL_FLOOR * reference_length)
    trimmed = np.where(residual > cut, 0.0, w)
    if (trimmed > 0).sum() >= 3 and (trimmed != w).any():
        w = trimmed
        rotation, residual = fit(w)
    rms = float(np.sqrt((w @ residual ** 2) / w.sum())) / reference_length
    return HeadRotation(rotation, rms, int((w > 0).sum()), w)
