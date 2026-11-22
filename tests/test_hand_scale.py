from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.body_model.hand_scale import (
    rescale_about_wrist,
    smooth_log_scale,
    wrist_relative_scale,
)

MANO_DIR = Path(__file__).parent.parent / "models"
MANO_AVAILABLE = (MANO_DIR / "mano" / "MANO_RIGHT.npz").exists()


def _hands(count=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.05, size=(count, 21, 3)) + rng.normal(size=(count, 1, 3))


def test_rescale_keeps_the_wrist_and_scales_offsets():
    points = _hands()
    scales = np.linspace(0.5, 2.0, len(points))
    out = rescale_about_wrist(points, scales)
    np.testing.assert_allclose(out[:, 0], points[:, 0])
    np.testing.assert_allclose(out - out[:, :1], (points - points[:, :1]) * scales[:, None, None])


def test_wrist_relative_scale_recovers_a_known_factor():
    points = _hands()
    scales = np.linspace(0.7, 1.4, len(points))
    moved = rescale_about_wrist(points, scales) + np.array([0.3, -0.1, 0.2])  # wrist moves too
    np.testing.assert_allclose(wrist_relative_scale(points, moved), scales)


def test_wrist_relative_scale_ignores_invalid_landmarks():
    points = _hands(2)
    observed = rescale_about_wrist(points, [1.5, 1.5])
    observed[:, 5] += 10.0  # a wild landmark, masked out
    valid = np.ones((2, 21), bool)
    valid[:, 5] = False
    np.testing.assert_allclose(wrist_relative_scale(points, observed, valid), [1.5, 1.5])
    assert np.isnan(wrist_relative_scale(points[:1] * 0, points[:1])).all()


def test_smooth_log_scale_keeps_a_constant_and_interpolates_geometrically():
    times = np.array([0.0, 2.0, 4.0])
    np.testing.assert_allclose(smooth_log_scale(times, [1.3] * 3, 2.0, at=[1.0, 3.0]), 1.3)
    # Nearest-neighbour mode: a query halfway between two scales takes their geometric mean.
    np.testing.assert_allclose(smooth_log_scale([0.0, 2.0], [0.5, 2.0], 0.0, at=[1.0]), [1.0])
    np.testing.assert_allclose(smooth_log_scale(times, [1.0, 2.0, 4.0], 0.0), [1.0, 2.0, 4.0])


def test_smooth_log_scale_skips_missing_and_survives_far_queries():
    out = smooth_log_scale([0.0, 1.0, 2.0], [1.0, np.nan, 1.0], 0.5, at=[1.0, 1e6])
    np.testing.assert_allclose(out, [1.0, 1.0])


@pytest.mark.skipif(not MANO_AVAILABLE, reason="MANO model files not downloaded")
def test_fit_hand_scales_follows_a_hand_that_changes_size():
    import torch

    from kinetic_augment.body_model.hand_model import HandModel
    from kinetic_augment.body_model.hand_scale import fit_hand_scales, fit_hands_at_scale

    model = HandModel("mano", "right", model_dir=str(MANO_DIR))
    rng = np.random.default_rng(3)
    count = 12
    pose = rng.normal(scale=0.2, size=(count, 48))
    with torch.no_grad():
        hands = model.points(pose[:, :3], pose[:, 3:], np.zeros(model.num_betas)).numpy()
    truth = np.exp(0.15 * np.sin(np.linspace(0, np.pi, count)))
    truth /= np.exp(np.mean(np.log(truth)))
    observed = rescale_about_wrist(hands, truth)
    valid = np.ones(observed.shape[:2], bool)

    betas, scales = fit_hand_scales(observed, valid, model)
    np.testing.assert_allclose(scales / scales.mean(), truth / truth.mean(), atol=0.005)
    _, rebuilt = fit_hands_at_scale(observed, valid, model, betas, scales)
    assert np.linalg.norm(rebuilt - observed, axis=-1).mean() < 1e-3  # metres
