"""Tests for re-placing the wrists in 3-D from the image (V1.3 task 14)."""

from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.body_model.hand_depth import (
    DEFAULT_VERTICAL_FOV_DEG, REACH_TOLERANCE, camera_point, focal_px, hand_scale_px_per_m,
    wrist_targets,
)
from tests.face_assets import MODEL_DIR, requires_face_assets

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data" / "test_output" / "01991"
W, H = 640, 480
F = focal_px(H)
CX, CY = W / 2, H / 2


def project(point):
    x, y, z = point
    return np.array([F * x / z + CX, F * y / z + CY])


def synthetic_hand(wrist_cam, rng):
    """21 hand points (metres) around a wrist: world offsets, camera-frame positions."""
    offsets = rng.uniform(-0.05, 0.05, (21, 3))
    offsets[0] = 0.0
    return offsets, wrist_cam + offsets


def test_focal_px_for_63_degrees():
    assert DEFAULT_VERTICAL_FOV_DEG == 63.0
    assert focal_px(480) == pytest.approx(240 / np.tan(np.radians(31.5)))
    assert focal_px(480, 90.0) == pytest.approx(240.0)


def test_camera_point_round_trips_a_perspective_projection():
    p = np.array([0.12, -0.2, 0.9])
    u, v = project(p)
    np.testing.assert_allclose(camera_point(u, v, p[2], F, CX, CY), p, atol=1e-12)


def test_hand_scale_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    world = rng.uniform(-0.05, 0.05, (21, 3))
    px = world[:, :2] * 850.0 + np.array([300.0, 200.0])
    assert hand_scale_px_per_m(world, px) == pytest.approx(850.0)


def test_hand_scale_is_nan_for_an_absent_hand():
    world = np.full((21, 3), np.nan)
    assert np.isnan(hand_scale_px_per_m(world, np.full((21, 2), np.nan)))


def scene(wrist_cam, z_head=1.0, nose_cam=None):
    rng = np.random.default_rng(1)
    nose_cam = np.array([0.0, 0.0, z_head]) if nose_cam is None else nose_cam
    offsets, cam = synthetic_hand(wrist_cam, rng)
    hand_world = np.full((2, 21, 3), np.nan)
    hand_px = np.full((2, 21, 2), np.nan)
    hand_world[1] = offsets
    hand_px[1] = [project(p) for p in cam]
    return nose_cam, hand_world, hand_px


def test_wrist_targets_recover_the_offset_from_the_nose():
    nose_cam = np.array([0.02, -0.05, 1.0])
    wrist_cam = nose_cam + np.array([0.03, 0.12, -0.15])  # lower and 15 cm nearer
    nose_cam, hand_world, hand_px = scene(wrist_cam, nose_cam=nose_cam)
    nose_world = np.array([0.4, -0.3, 0.1])               # arbitrary world position
    targets = wrist_targets(project(nose_cam), nose_world, F / nose_cam[2], hand_world, hand_px,
                            W, H)
    assert targets.corrected.tolist() == [False, True]
    assert np.isnan(targets.target[0]).all() and np.isnan(targets.depth_ratio[0])
    np.testing.assert_allclose(targets.target[1] - nose_world, wrist_cam - nose_cam, atol=0.015)
    assert targets.depth_ratio[1] == pytest.approx(wrist_cam[2] / nose_cam[2], abs=0.02)


def test_an_implausible_depth_ratio_is_left_uncorrected():
    nose_cam = np.array([0.0, 0.0, 1.0])
    nose_cam, hand_world, hand_px = scene(np.array([0.0, 0.1, 3.0]), nose_cam=nose_cam)
    targets = wrist_targets(project(nose_cam), np.zeros(3), F / 1.0, hand_world, hand_px, W, H)
    assert not targets.corrected.any() and np.isnan(targets.target).all()
    assert targets.depth_ratio[1] == pytest.approx(3.0, abs=0.1)


def test_an_invalid_arm_leaves_that_side_uncorrected():
    nose_cam, hand_world, hand_px = scene(np.array([0.0, 0.1, 0.9]))
    nose_px = project(nose_cam)
    both = wrist_targets(nose_px, np.zeros(3), F / 1.0, hand_world, hand_px, W, H)
    assert both.corrected.tolist() == [False, True]  # only the right hand is present
    gated = wrist_targets(nose_px, np.zeros(3), F / 1.0, hand_world, hand_px, W, H,
                          arm_valid=np.array([True, False]))  # right elbow or shoulder invalid
    assert not gated.corrected.any() and np.isnan(gated.target).all()
    assert np.isfinite(gated.depth_ratio[1])  # the ratio is still reported
    kept = wrist_targets(nose_px, np.zeros(3), F / 1.0, hand_world, hand_px, W, H,
                         arm_valid=np.array([False, True]))
    assert kept.corrected.tolist() == [False, True]


def test_missing_inputs_leave_both_hands_uncorrected():
    nose_cam, hand_world, hand_px = scene(np.array([0.0, 0.1, 0.9]))
    for nose_px, scale in ((np.full(2, np.nan), F), (project(nose_cam), np.nan)):
        targets = wrist_targets(nose_px, np.zeros(3), scale, hand_world, hand_px, W, H)
        assert not targets.corrected.any()
        assert np.isnan(targets.depth_ratio).all()


@requires_face_assets
def test_run_clip_depth_corrected_on_pilot_frames():
    hands_path, face_path = Path(f"{CLIP}_hands.npz"), Path(f"{CLIP}_face.npz")
    if not (hands_path.exists() and face_path.exists()):
        pytest.skip("pilot hand/face clips absent")
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.face_pipeline import (
        load_face_models, run_clip, save_face_features,
    )
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    hands, face = load_hand_clip(hands_path), load_face_clip(face_path)
    models = load_face_models(MODEL_DIR)
    frames = np.arange(0, 20)
    plain = run_clip(hands, face, models, frames=frames)
    result = run_clip(hands, face, models, frames=frames, depth_corrected=True)
    assert plain.wrist_corrected is not None and not plain.wrist_corrected.any()
    assert result.wrist_target.shape == (20, 2, 3)
    for name in ("wrist_corrected", "depth_ratio", "wrist_residual_m", "wrist_shift_m"):
        assert getattr(result, name).shape == (20, 2), name
    corrected = result.wrist_corrected
    assert np.isfinite(result.wrist_residual_m[corrected]).all()
    assert np.isfinite(result.wrist_shift_m[corrected]).all()
    assert np.isnan(result.wrist_residual_m[~corrected]).all()
    reached = result.wrist_reached
    assert reached.shape == (20, 2) and reached.dtype == bool
    assert not reached[~corrected].any()
    np.testing.assert_array_equal(
        reached[corrected], result.wrist_residual_m[corrected] <= REACH_TOLERANCE)
    assert not plain.wrist_reached.any()
    assert len(result.geometry) == 20 and len(result.image_geometry) == 20
    # Frames with no correction are exactly the first pass.
    untouched = ~corrected.any(axis=1)
    np.testing.assert_array_equal(result.rotations[untouched], plain.rotations[untouched])
    for key, values in (("wrist_shift_m", result.wrist_shift_m),
                        ("wrist_residual_m", result.wrist_residual_m),
                        ("depth_ratio", result.depth_ratio)):
        if corrected.any():
            print(f"median {key}:", np.nanmedian(values[corrected]))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        save_face_features(Path(tmp) / "x_dc.npz", result, face)
        data = np.load(Path(tmp) / "x_dc.npz")
        assert data["wrist_target"].shape == (20, 2, 3)
        assert data["wrist_corrected"].shape == (20, 2)
        np.testing.assert_array_equal(data["wrist_reached"], reached)
