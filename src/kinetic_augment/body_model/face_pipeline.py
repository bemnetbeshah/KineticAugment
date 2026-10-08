"""
One clip end to end: body, hands and head fitted per frame, assembled into one SMPL-X pose,
and hand-to-face geometry measured on it (V1.3 design §3).

The body is fitted to the pose landmarks stored in the hand clip, so every stream shares the
same decoded frames. Hands are fitted with MANO and attached as the V1.2 viewer attaches them.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .assembly import attach_hands, attach_head, body_rotations, pose_mesh
from .converter import fit_body
from .face_correspondence import (
    LEFT_EYE_OUTER, NOSE_TIP, RIGID, RIGHT_EYE_OUTER, load_face_correspondence,
)
from .face_regions import REGIONS, SUBREGIONS, build_head_surface
from .hand_converter import fit_hand_shape, fit_hands, reconstruct_hand
from .hand_depth import DEFAULT_VERTICAL_FOV_DEG, wrist_targets
from .hand_face_geometry import HAND_POINTS, HandFaceGeometry, hand_face_geometry
from .hand_model import HandModel
from .head_converter import (
    OCCLUSION_PAD, HeadFit, fit_head, hand_outlines, head_template, points_in_polygon,
)
from .image_proximity import (
    ImageProximity, image_hand_points, image_proximity, project_head,
)
from .shape import fit_shape
from .skeleton import SMPLXSkeleton

VISIBILITY_THRESHOLD = 0.5  # matches measure_round_trip.py
EARS = (7, 8)               # MediaPipe pose: left ear, right ear
SIDES = ("left", "right")
NECK_MODES = ("rest", "swing", "split")
POSE_WRISTS = (15, 16)      # MediaPipe pose: left wrist, right wrist
SMPLX_WRISTS = (20, 21)     # SMPL-X joints: left wrist, right wrist


@dataclass
class FaceModels:
    skeleton: SMPLXSkeleton
    corr: object
    mano: Dict[str, HandModel]         # fitting backend
    smplx_hands: Dict[str, HandModel]  # SMPL-X palms for attaching the wrists
    model_dir: str


def load_face_models(model_dir="models") -> FaceModels:
    return FaceModels(
        skeleton=SMPLXSkeleton(model_dir),
        corr=load_face_correspondence(model_dir),
        mano={s: HandModel("mano", s, model_dir=model_dir) for s in SIDES},
        smplx_hands={s: HandModel("smplx", s, model_dir=model_dir) for s in SIDES},
        model_dir=model_dir,
    )


def hand_near_face(face_px, outlines, pad: float) -> bool:
    """True if a hand outline comes within `pad` pixels of the face's bounding box."""
    points = np.asarray(face_px)[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points) or not outlines:
        return False
    lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
    corners = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    for outline in outlines:
        if ((outline >= lo) & (outline <= hi)).all(axis=1).any():
            return True
        if points_in_polygon(corners, outline).any():
            return True
    return False


@dataclass
class ClipFaceResult:
    head_fits: List[HeadFit]
    rotations: np.ndarray         # (T, 55, 3); NaN where the body could not be fitted
    transl: np.ndarray            # (T, 3)
    betas: np.ndarray             # (300,)
    geometry: List[Optional[HandFaceGeometry]]
    hand_near_face: np.ndarray    # (T,) bool
    body_ok: np.ndarray           # (T,) bool
    frames: np.ndarray            # (T,) source frame indices
    image_geometry: List[Optional[ImageProximity]] = field(default_factory=list)
    head_scale_px_per_m: np.ndarray = None       # (T,) px per metre at the head; NaN if none
    head_camera_rotation_deg: np.ndarray = None  # (T,) world->pixel rotation angle; NaN if none
    # Depth correction (task 14); filled in every run, empty unless depth_corrected is set.
    wrist_target: np.ndarray = None       # (T, 2, 3) image-derived world wrist; NaN if none
    wrist_corrected: np.ndarray = None    # (T, 2) bool: the re-fit used the target
    depth_ratio: np.ndarray = None        # (T, 2) Z_hand / Z_head; NaN where unavailable
    wrist_residual_m: np.ndarray = None   # (T, 2) |SMPL-X wrist - target| after the re-fit
    wrist_shift_m: np.ndarray = None      # (T, 2) |target - MediaPipe pose-world wrist|


def _fit_hands(hands, mano, frames):
    fits, recon = {}, {}
    for s, side in enumerate(SIDES):
        observed = hands.world[frames, s].astype(np.float64)
        valid = np.isfinite(observed).all(axis=-1)
        betas = fit_hand_shape(observed, valid, mano[side])
        fits[side] = fit_hands(observed, valid, mano[side], betas)
        recon[side] = np.stack([reconstruct_hand(f, mano[side]) for f in fits[side]])
    return fits, recon


def run_clip(hands, face, models: FaceModels, frames=None, depth_corrected=False,
             fov_deg=DEFAULT_VERTICAL_FOV_DEG) -> ClipFaceResult:
    """
    Fit and measure every frame (or `frames`, source indices) of one clip.

    With `depth_corrected`, each frame whose head projection exists and where at least one hand's
    wrist can be placed from the image (`hand_depth.wrist_targets`) gets a second pass: the pose
    wrist is replaced by that target, the body is re-fitted with the same shaped skeleton, and
    hands, head, mesh and both geometries are recomputed. Those frames then hold the second-pass
    values; every other frame, and every frame with `depth_corrected` False, is the first pass.

    Raises:
        ValueError: the hand and face clips do not share frames.
    """
    if not np.array_equal(hands.timestamp_ms, face.timestamp_ms):
        raise ValueError("hand and face clips are not frame-aligned")
    frames = np.arange(len(face.valid)) if frames is None else np.asarray(frames)
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (hands.pose_visibility >= VISIBILITY_THRESHOLD)
    betas = fit_shape(pose, ok, models.skeleton)
    body = models.skeleton.shaped(betas)
    template = head_template(body, models.corr)
    surface = build_head_surface(body, models.corr, models.model_dir)
    hand_fits, hand_recon = _fit_hands(hands, models.mano, frames)
    face_px = face.pixels()

    n = len(frames)
    result = ClipFaceResult([], np.full((n, 55, 3), np.nan), np.full((n, 3), np.nan), betas,
                            [], np.zeros(n, bool), np.zeros(n, bool), frames, [],
                            np.full(n, np.nan), np.full(n, np.nan),
                            np.full((n, 2, 3), np.nan), np.zeros((n, 2), bool),
                            np.full((n, 2), np.nan), np.full((n, 2), np.nan),
                            np.full((n, 2), np.nan))
    for i, t in enumerate(frames):
        outlines = hand_outlines(hands.image[t], hands.width, hands.height)
        observed_face = face_px[t] if face.valid[t] else None
        if observed_face is not None:
            reference = np.linalg.norm(observed_face[RIGHT_EYE_OUTER]
                                       - observed_face[LEFT_EYE_OUTER])
            result.hand_near_face[i] = hand_near_face(observed_face, outlines,
                                                      OCCLUSION_PAD * reference)
        try:
            body_fit = fit_body(pose[t], ok[t], body)
        except ValueError:
            result.head_fits.append(fit_head(None, [], np.zeros((55, 3)), np.zeros(3), template))
            result.geometry.append(None)
            result.image_geometry.append(None)
            continue
        result.body_ok[i] = True
        hand_fit_i = {side: hand_fits[side][i] for side in SIDES}
        hand_recon_i = {side: hand_recon[side][i] for side in SIDES}
        ears = pose[t, list(EARS)].mean(axis=0) if ok[t, list(EARS)].all() else None
        rotations, head, mesh, posed = _assemble(
            body, betas, models, body_fit, hand_fit_i, hand_recon_i, observed_face, outlines,
            template, ears)
        geometry = hand_face_geometry(mesh, surface, posed)
        image_geometry = _image_geometry(result, i, mesh, surface, models.corr, head,
                                         observed_face, hands.image[t], hands.width, hands.height)
        transl = body_fit.transl
        if depth_corrected and image_geometry is not None:
            targets = wrist_targets(
                observed_face[NOSE_TIP], models.corr.points(mesh.vertices, (NOSE_TIP,))[0],
                result.head_scale_px_per_m[i], hands.world[t],
                hands.image[t][..., :2] * np.array([hands.width, hands.height], np.float64),
                hands.width, hands.height, fov_deg)
            result.depth_ratio[i] = targets.depth_ratio
            if targets.corrected.any():
                second = _depth_pass(
                    result, i, t, targets, pose, ok, body, betas, models, surface, hand_fit_i,
                    hand_recon_i, observed_face, outlines, template, ears, hands)
                if second is not None:
                    rotations, head, geometry, image_geometry, transl = second
        result.head_fits.append(head)
        result.rotations[i], result.transl[i] = rotations, transl
        result.geometry.append(geometry)
        result.image_geometry.append(image_geometry)
    return result


def _assemble(body, betas, models, body_fit, hand_fit, hand_recon, observed_face, outlines,
              template, ears):
    """Hands, then head, attached to one body fit; returns rotations, head, mesh, hands posed."""
    rotations, posed = attach_hands(body_rotations(body_fit), body.parents, hand_fit,
                                    hand_recon, models.smplx_hands)
    head = fit_head(observed_face, outlines, rotations, body_fit.transl, template, ears)
    rotations = attach_head(rotations, head)
    return rotations, head, pose_mesh(body, rotations, betas, body_fit.transl), posed


def _depth_pass(result, i, t, targets, pose, ok, body, betas, models, surface, hand_fit,
                hand_recon, observed_face, outlines, template, ears, hands):
    """
    Second pass of one frame with the corrected wrists (see `run_clip`).

    Fills the wrist target, corrected, residual and shift fields of `result` row `i` and
    returns (rotations, head, geometry, image geometry, transl), or None if the body cannot be
    re-fitted.
    """
    landmarks, valid = pose[t].copy(), ok[t].copy()
    for s, wrist in enumerate(POSE_WRISTS):
        if targets.corrected[s]:
            landmarks[wrist], valid[wrist] = targets.target[s], True
    try:
        body_fit = fit_body(landmarks, valid, body)
    except ValueError:
        return None
    rotations, head, mesh, posed = _assemble(
        body, betas, models, body_fit, hand_fit, hand_recon, observed_face, outlines, template,
        ears)
    geometry = hand_face_geometry(mesh, surface, posed)
    image_geometry = _image_geometry(result, i, mesh, surface, models.corr, head, observed_face,
                                     hands.image[t], hands.width, hands.height)
    result.wrist_target[i] = targets.target
    result.wrist_corrected[i] = targets.corrected
    for s in np.flatnonzero(targets.corrected):
        result.wrist_residual_m[i, s] = np.linalg.norm(
            mesh.joints[SMPLX_WRISTS[s]] - targets.target[s])
        result.wrist_shift_m[i, s] = np.linalg.norm(
            targets.target[s] - pose[t, POSE_WRISTS[s]])
    return rotations, head, geometry, image_geometry, body_fit.transl


def _image_geometry(result, i, mesh, surface, corr, head, observed_face, hand_image,
                    width, height) -> Optional[ImageProximity]:
    """Image-space proximity for one frame, or None without a usable head fit."""
    if observed_face is None or (head.rigid_weights > 0).sum() < 3:
        return None
    projected = project_head(
        mesh.vertices[surface.vertex_ids], surface.faces, corr.points(mesh.vertices, RIGID),
        observed_face[list(RIGID)], head.rigid_weights)
    result.head_scale_px_per_m[i] = projected.scale
    result.head_camera_rotation_deg[i] = projected.rotation_deg
    return image_proximity(image_hand_points(hand_image, width, height), projected, surface)


def save_face_features(path, result: ClipFaceResult, face) -> None:
    """Write the per-frame face features (design §3) to .npz."""
    fits, n = result.head_fits, len(result.head_fits)
    shape = (n, 2, len(HAND_POINTS))
    distance = np.full(shape, np.nan)
    region = np.full(shape, -1, np.int8)
    subregion = np.full(shape, -1, np.int8)
    contact = np.zeros(shape, bool)
    for i, g in enumerate(result.geometry):
        if g is not None:
            distance[i], region[i], subregion[i], contact[i] = (g.distance, g.region,
                                                                g.subregion, g.contact)
    image = {"image_distance": np.full(shape, np.nan), "image_overlap": np.zeros(shape, bool),
             "image_region": np.full(shape, -1, np.int8),
             "image_subregion": np.full(shape, -1, np.int8), "image_near": np.zeros(shape, bool)}
    for i, g in enumerate(result.image_geometry):
        if g is not None:
            image["image_distance"][i], image["image_overlap"][i] = g.distance, g.overlap
            image["image_region"][i], image["image_subregion"][i] = g.region, g.subregion
            image["image_near"][i] = g.near
    meta = {"regions": list(REGIONS), "subregions": list(SUBREGIONS),
            "hand_points": list(HAND_POINTS), "neck_modes": list(NECK_MODES),
            "blendshape_names": list(face.blendshape_names)}
    np.savez_compressed(
        path,
        frames=result.frames,
        neck_pose=np.array([f.neck_pose for f in fits]),
        head_pose=np.array([f.head_pose for f in fits]),
        jaw_pose=np.array([f.jaw_pose for f in fits]),
        valid=np.array([f.valid for f in fits]),
        reliable=np.array([f.reliable for f in fits]),
        jaw_reliable=np.array([f.jaw_reliable for f in fits]),
        rms_fraction=np.array([f.rms_fraction for f in fits]),
        points_used=np.array([f.points_used for f in fits]),
        neck_mode=np.array([NECK_MODES.index(f.neck_mode) for f in fits], np.int8),
        ear_residual_m=np.array([f.ear_residual_m for f in fits]),
        distance=distance, region=region, subregion=subregion, contact=contact,
        **image,
        head_scale_px_per_m=result.head_scale_px_per_m,
        head_camera_rotation_deg=result.head_camera_rotation_deg,
        wrist_target=result.wrist_target, wrist_corrected=result.wrist_corrected,
        depth_ratio=result.depth_ratio, wrist_residual_m=result.wrist_residual_m,
        wrist_shift_m=result.wrist_shift_m,
        hand_near_face=result.hand_near_face, body_ok=result.body_ok,
        blendshapes=face.blendshapes[result.frames],
        betas=result.betas,
        meta=np.array(json.dumps(meta)),
    )
