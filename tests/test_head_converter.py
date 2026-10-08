"""
Tests for the head converter (design §4): occlusion, head rotation, jaw, neck/head split, fit_head.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.head_converter import (
    HeadRotation, head_rotation, occluded, points_in_polygon,
)

PIXELS_PER_METRE = 4000.0


def synthetic_face(seed=0, noise_px=0.0):
    rng = np.random.default_rng(seed)
    template = 0.04 * rng.normal(size=(17, 3))
    rotation = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    observed = PIXELS_PER_METRE * template @ rotation.T + [320.0, 200.0, 50.0]
    observed += noise_px * rng.normal(size=observed.shape)
    return template, observed, rotation


def angle_deg(a, b):
    return np.degrees(Rotation.from_matrix(a.T @ b).magnitude())


def test_head_rotation_recovers_the_rotation():
    template, observed, rotation = synthetic_face(noise_px=0.5)
    fit = head_rotation(template, observed, np.ones(17), reference_length=250.0)
    assert isinstance(fit, HeadRotation)
    assert angle_deg(fit.rotation, rotation) < 0.5
    assert fit.points_used == 17
    assert fit.rms_fraction < 0.01


def test_outliers_are_dropped_in_the_robust_pass():
    template, observed, rotation = synthetic_face(noise_px=0.5)
    observed[[2, 7]] += 30.0
    fit = head_rotation(template, observed, np.ones(17), reference_length=250.0)
    assert fit.points_used == 15
    assert angle_deg(fit.rotation, rotation) < 0.5


def test_noise_free_points_are_all_kept():
    template, observed, _ = synthetic_face()
    assert head_rotation(template, observed, np.ones(17), 250.0).points_used == 17


def test_too_few_points_give_nan():
    template, observed, _ = synthetic_face()
    weights = np.zeros(17)
    weights[:2] = 1.0
    fit = head_rotation(template, observed, weights, 250.0)
    assert np.isnan(fit.rotation).all() and fit.points_used == 2


def test_occluded_marks_inside_and_near_points():
    square = [np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])]
    points = np.array([[50.0, 50.0], [105.0, 50.0], [120.0, 50.0], [-3.0, -3.0]])
    np.testing.assert_array_equal(occluded(points, square, pad=10.0),
                                  [True, True, False, True])
    assert not occluded(points, [], pad=10.0).any()


def test_points_in_polygon_convex_square():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    points = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, -1.0], [-1.0, 5.0]])
    np.testing.assert_array_equal(points_in_polygon(points, square),
                                  [True, False, False, False])


def test_points_in_polygon_non_convex_u_shape():
    u_shape = np.array([[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [6.0, 9.0], [6.0, 3.0],
                        [3.0, 3.0], [3.0, 9.0], [0.0, 9.0]])
    points = np.array([[1.5, 6.0],   # left arm
                       [7.5, 6.0],   # right arm
                       [4.5, 6.0],   # notch between the arms
                       [4.5, 1.5],   # base
                       [4.5, 12.0]])  # above everything
    np.testing.assert_array_equal(points_in_polygon(points, u_shape),
                                  [True, True, False, True, False])


def test_points_in_polygon_empty_points():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert points_in_polygon(np.empty((0, 2)), square).shape == (0,)


from tests.face_assets import MODEL_DIR, requires_face_assets  # noqa: E402

BETAS = np.zeros(300)
BETAS[[0, 2, 7]] = [0.8, -0.6, 1.2]


@pytest.fixture(scope="module")
def body():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=MODEL_DIR).shaped(BETAS)


@pytest.fixture(scope="module")
def corr():
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    return load_face_correspondence(MODEL_DIR)


@pytest.fixture(scope="module")
def curve(body, corr):
    from kinetic_augment.body_model.head_converter import jaw_curve
    return jaw_curve(body, corr)


@requires_face_assets
def test_jaw_curve_opens_with_positive_pitch(curve):
    from kinetic_augment.body_model.head_converter import JAW_GRID_DEG

    assert len(curve.pitch_rad) == len(JAW_GRID_DEG)
    assert np.all(np.diff(curve.gap_ratio) >= 0)
    assert curve.gap_ratio[40] > curve.gap_ratio[0] + 0.05  # 10 degrees opens the lips visibly


@requires_face_assets
def test_fit_jaw_pitch_recovers_a_posed_jaw(body, corr, curve):
    from kinetic_augment.body_model.assembly import JAW, pose_mesh
    from kinetic_augment.body_model.head_converter import JAW_AXIS, fit_jaw_pitch, lip_gap_ratio

    rotations = np.zeros((55, 3))
    rotations[JAW] = np.radians(12.3) * JAW_AXIS
    vertices = pose_mesh(body, rotations, body.betas, np.zeros(3)).vertices
    upper, lower, right, left = corr.points(vertices, (13, 14, 33, 263))
    pitch = fit_jaw_pitch(lip_gap_ratio(upper, lower, right, left), curve)
    assert abs(np.degrees(pitch) - 12.3) < 1.0


@requires_face_assets
def test_fit_jaw_pitch_clamps_to_the_grid(curve):
    from kinetic_augment.body_model.head_converter import fit_jaw_pitch

    assert fit_jaw_pitch(-1.0, curve) == 0.0
    assert np.isclose(np.degrees(fit_jaw_pitch(10.0, curve)), 30.0)
