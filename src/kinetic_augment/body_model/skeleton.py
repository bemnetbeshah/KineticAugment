"""
Rest-pose skeleton sources for forward and inverse kinematics.

Fitting needs only two things from a body model: where each joint sits at rest,
and which joint is its parent. Isolating that behind a small interface lets the
kinematics run against a hand-checkable synthetic chain as well as the real
SMPL-X skeleton, so a failure can be localised to the maths or to SMPL-X's
specific rest geometry.
"""

from typing import List, Optional

import numpy as np


class SyntheticSkeleton:
    """
    A straight 3-bone chain of unit-length bones along +x.

    Deliberately trivial: every posed position is computable by hand, so a
    failing kinematics test can be checked on paper.
    """

    def __init__(self):
        self.rest_joints = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        self.parents = np.array([-1, 0, 1, 2])
        self.names: List[str] = ["root", "a", "b", "c"]

    @property
    def num_joints(self) -> int:
        return len(self.parents)


class SMPLXSkeleton:
    """
    The real SMPL-X rest skeleton: 55 model joints and their kinematic tree.

    Rest joints are the shape-blended template regressed to joint centres, which
    is the same quantity SMPL-X's own skinning poses. Only the 55 model joints
    are exposed; the 72 extra landmarks SMPL-X appends to its `joints` output are
    derived from mesh vertices and are not part of the kinematic chain.
    """

    NUM_MODEL_JOINTS = 55

    def __init__(self, model_path: str = "models", gender: str = "neutral",
                 betas: Optional[np.ndarray] = None):
        import torch
        import smplx
        from smplx.vertex_ids import vertex_ids

        # flat_hand_mean=True so a zero rotation genuinely means no rotation.
        # SMPL-X otherwise adds a non-zero mean hand pose (up to ~0.85 rad) to
        # whatever hand parameters it is given, which would make "zero pose"
        # silently differ from the rest skeleton and corrupt round-trip error.
        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        )

        v = self.model.v_template
        if betas is not None:
            shape_offset = torch.einsum(
                "l,mkl->mk",
                torch.as_tensor(betas, dtype=v.dtype),
                self.model.shapedirs[:, :, : len(betas)],
            )
            v = v + shape_offset

        with torch.no_grad():
            joints = self.model.J_regressor @ v

        self.rest_joints = joints[: self.NUM_MODEL_JOINTS].detach().numpy().astype(np.float64)
        self.parents = self.model.parents[: self.NUM_MODEL_JOINTS].detach().numpy().astype(int)

        # The nose tip on the mesh surface: the same anatomical point MediaPipe
        # reports, unlike the head joint, which sits inside the skull and points in
        # a direction ~30 degrees different from the shoulders.
        self.rest_nose = (
            v[vertex_ids["smplx"]["nose"]].detach().numpy().astype(np.float64)
        )
        self.names: List[str] = [f"joint_{i}" for i in range(self.NUM_MODEL_JOINTS)]

    @property
    def num_joints(self) -> int:
        return self.NUM_MODEL_JOINTS
