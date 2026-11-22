"""
Prototype: one hand shape per track plus a uniform scale per frame.

MediaPipe's metric hand changes size within a track (a median 15.9% coefficient of variation
in bone length, docs/V1_2_Findings.md section 4.2). A per-track shape cannot follow that; a
per-frame shape can, but with 10 more values per frame it absorbs more than size. This module
isolates size: every frame keeps the track's shape and gets one extra number, a uniform scale
about the wrist.

The scale is not a MANO or SMPL-X parameter. A scaled hand is the model's hand, posed by
genuine parameters, then scaled about its wrist, so a pipeline using it has to carry the scale
beside the model parameters and apply it after the forward pass.

Scaling about the wrist commutes with the wrist-relative fit: fitting a hand s times the
model's size to `observed` is fitting the model to `observed` shrunk by s about its wrist, so
`fit_hands` does the work unchanged (up to its pose prior, which is not rescaled).
"""

from typing import List, Tuple

import numpy as np

from kinetic_augment.body_model.hand_converter import (
    SHAPE_PLAUSIBILITY,
    HandFit,
    _optimise,
    _prepared,
    _starts_from,
    fit_hands,
    reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import WRIST


def rescale_about_wrist(points, scales) -> np.ndarray:
    """(T, 21, 3) points with frame t scaled by scales[t] about its own wrist."""
    points = np.asarray(points, dtype=np.float64)
    wrist = points[:, WRIST:WRIST + 1]
    return wrist + (points - wrist) * np.asarray(scales, dtype=np.float64)[:, None, None]


def wrist_relative_scale(predicted, observed, valid=None) -> np.ndarray:
    """
    Per frame, the factor s minimising |s (P - P_wrist) - (O - O_wrist)|^2 over valid landmarks.

    Returns (T,) scales; NaN where a frame has no finite valid landmark besides the wrist.
    """
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    p, o = p - p[:, WRIST:WRIST + 1], o - o[:, WRIST:WRIST + 1]
    use = np.isfinite(p).all(-1) & np.isfinite(o).all(-1)
    if valid is not None:
        use &= np.asarray(valid, dtype=bool)
    p, o = np.where(use[..., None], p, 0.0), np.where(use[..., None], o, 0.0)
    num, den = (p * o).sum(axis=(1, 2)), (p * p).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def smooth_log_scale(times, scales, sigma, at=None) -> np.ndarray:
    """
    Gaussian kernel regression of log scale over time, evaluated at `at` (default `times`).

    `sigma` is in the units of `times` (frames). Non-finite scales are ignored. With sigma 0
    each query takes the scale of its nearest observed time (ties: the mean of the nearest).
    """
    times = np.asarray(times, dtype=np.float64)
    at = times if at is None else np.asarray(at, dtype=np.float64)
    log = np.log(np.asarray(scales, dtype=np.float64))
    keep = np.isfinite(log)
    times, log = times[keep], log[keep]
    if not len(log):
        return np.full(len(at), np.nan)
    gap = at[:, None] - times[None, :]
    if sigma > 0:
        weight = np.exp(-0.5 * (gap / sigma) ** 2)
        # Far from every observation all weights underflow; fall back to the nearest.
        empty = weight.sum(axis=1) == 0
        weight[empty] = np.abs(gap[empty]) == np.abs(gap[empty]).min(axis=1, keepdims=True)
    else:
        weight = (np.abs(gap) == np.abs(gap).min(axis=1, keepdims=True)).astype(np.float64)
    return np.exp((weight @ log) / weight.sum(axis=1))


def fit_hands_at_scale(observed, valid, model, betas, scales) -> Tuple[List[HandFit], np.ndarray]:
    """
    Fit each frame's pose for a hand of shape `betas` scaled by scales[t] about the wrist.

    Returns the fits (to the shrunk observations, so their `rms_mm` is in the model's units)
    and the (T, 21, 3) landmarks rebuilt from them and scaled back up.
    """
    observed = np.asarray(observed, dtype=np.float64)
    fits = fit_hands(rescale_about_wrist(observed, 1.0 / np.asarray(scales)), valid, model, betas)
    rebuilt = np.stack([reconstruct_hand(fit, model) for fit in fits])
    return fits, rescale_about_wrist(rebuilt, scales)


def fit_hand_scales(observed, valid, model,
                    plausibility=SHAPE_PLAUSIBILITY) -> Tuple[np.ndarray, np.ndarray]:
    """
    The track's shape and every frame's scale, fitted jointly with the frames' poses in one
    L-BFGS run (started from neutral-shape fits, like `fit_hand_shape`).

    Overall size is shared between the shape and the scales, so the scales' level is not
    pinned to 1; only their product with the shape's size is determined.

    Returns ((K,) betas, (T,) scales); a frame the neutral fit cannot place has scale NaN.
    """
    observed, valid = _prepared(observed, valid)
    neutral = fit_hands(observed, valid, model)
    frames = [t for t, fit in enumerate(neutral) if fit.status == "ok"]
    scales = np.full(len(observed), np.nan)
    if not frames:
        return np.zeros(model.num_betas), scales
    _, betas, _, fitted = _optimise(model, observed[frames], valid[frames],
                                    _starts_from(neutral, frames), shape="shared",
                                    plausibility=plausibility, scale=True)
    scales[frames] = fitted
    return betas[0], scales
