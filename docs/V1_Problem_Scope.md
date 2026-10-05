# V1 Problem Scope: MediaPipe ↔ SMPL-X Round-Trip Conversion

**Status:** Proposed work. **Date:** 2026-10-01. **Source snapshot:** `45472c6`.

This document scopes the first deliverable for the MediaPipe ↔ SMPL-X conversion task:
convert MediaPipe landmarks into genuine SMPL-X parameters and back, losing as little
information as the representation allows, and measure precisely how much is lost.

Evidence labels follow the [research guide](research-guide/index.md#how-to-interpret-evidence):
**Code present**, **Verified now**, **Historical claim**, **Inference**, **Proposed work**.

---

## 1. The ask, stated precisely

Go from MediaPipe landmarks to SMPL-X parameters and back without losing information.

Restated as an engineering problem:

> Given MediaPipe Holistic landmarks `P` for a frame, produce SMPL-X parameters `θ` that are
> **valid inputs to the real SMPL-X model**, such that running SMPL-X forward on `θ` and
> reading off the corresponding joints reconstructs `P` to within a stated tolerance.

Two requirements, and both must hold:

1. **Round-trip fidelity.** `P → θ → P'` where `P' ≈ P`.
2. **Parameter validity.** `θ` must be real axis-angle rotations and shape coefficients that
   drive the actual SMPL-X mesh. A `(1, 63)` array of the right shape is not enough.

Requirement 2 is the one that makes this hard, and the one the current code skips.

---

## 2. Why "lossless" must be defined before it can be delivered

Bit-exact round-trip through real SMPL-X is **not achievable**, for three independent reasons.
This is a property of the representations, not a limitation of any implementation.

| Reason | Consequence |
| --- | --- |
| **Bone lengths are constrained.** SMPL-X limb lengths are determined by 10 shape coefficients plus the template. MediaPipe's observed bone lengths are whatever the detector reports, and they drift frame to frame on a rigid limb. | No `betas` vector reproduces arbitrary per-frame bone lengths. Residual length error is structural. |
| **MediaPipe's `z` is not metric.** It is a relative pseudo-depth in units of image width, not a calibrated Euclidean coordinate. | The target is not a metric 3D point cloud. Fitting treats it as one only by convention. |
| **Joint centres do not coincide.** MediaPipe's `left_shoulder` is a surface-adjacent keypoint; SMPL-X's `left_shoulder` is an internal rotation centre. | A systematic offset exists between corresponded points even for a perfect fit. |

Therefore "lossless" is operationalised as:

> **Round-trip error is at or below MediaPipe's own frame-to-frame noise floor.**

If the reconstruction differs from the input by less than the detector's own jitter on an
unmoving limb, the conversion has discarded nothing the input reliably carried. The noise
floor is **measured** (Step 1), not assumed.

#### ⚠️ Correction (2026-10-04): the noise floor is a reference, not a threshold

The definition above conflates two different statistics and must not be used as a pass/fail gate
in its current form:

- **Temporal jitter** is frame-to-frame variation. A conversion with a *systematic* bias — every
  wrist displaced 2 cm the same way every frame — has a jitter of zero.
- **Reconstruction error** is per-frame deviation from the target, systematic parts included.

A reconstruction can therefore sit far above the jitter floor while discarding nothing random,
and the two numbers are not commensurable.

The estimators are also undermined at the source: MediaPipe Holistic defaults to
`smooth_landmarks=True` (**verified now**), which deliberately correlates consecutive frames.
Method B's second-difference estimator assumes independent per-frame noise, so smoothing makes
the measured floor an **underestimate** of true per-frame uncertainty.

Treat **0.18% as a reference scale**, not an acceptance threshold. A defensible threshold needs
to be derived from the same statistic as the reconstruction score — for example, the spread
between repeated detections of the same pose, or a task-level criterion.

**Rejected alternative.** Exactness is reachable by storing a per-landmark residual alongside
`θ`, making reconstruction exact by construction. This was considered and declined for V1: it
guarantees the number without earning it, and the residual would silently absorb exactly the
error this study exists to quantify. It is retained as a documented fallback (Section 9).

---

## 3. What exists today, and why it does not satisfy the ask

**Verified now.** The current converter round-trips to a maximum absolute error of
`3.7e-09` ([05-project-status.md:17](research-guide/05-project-status.md:17)). That number is real
and reproducible. It also does not mean what it appears to mean.

[`mp_to_smplx.py:1-30`](../src/kinetic_augment/body_model/mp_to_smplx.py:1) states in its own module
docstring that it stores the raw bone displacement `child − parent` directly into the axis-angle
slots of `body_pose`, and that the result is **"NOT standard SMPL-X axis-angle, so you cannot
feed these params directly into an SMPL-X model and expect meaningful output."**

[`smplx_to_mp.py`](../src/kinetic_augment/body_model/smplx_to_mp.py) then reconstructs by adding
that displacement back onto the parent position. It accepts an `smplx_wrapper` argument and
**never calls it** ([smplx_to_mp.py:14](../src/kinetic_augment/body_model/smplx_to_mp.py:14)).

So the round-trip is exact because it is a subtraction followed by an addition. The SMPL-X model
is never evaluated, no rotation is ever computed, and `betas` and `global_orient` have no effect
on the output. It satisfies requirement 1 by entirely bypassing requirement 2.

Consequences that follow directly from this, and that V1 exists to fix:

- Intrinsic augmentations treat displacements as angles. Zero-noise `JointAnglePerturbation`
  changes projected landmarks by `0.56` — a no-op operation that is not a no-op
  (**Verified now**, [05-project-status.md:18](research-guide/05-project-status.md:18)).
- Joint-limit clamping applies radian bounds to displacement values, which has no anatomical meaning.
- No mesh, no collision geometry, and no body-shape variation is reachable through this path.

**Historical claim, relevant.** A real Adam-based fitting loop existed before commit `f9c6d7b`
and was replaced by the current encoding. Recover it for reference with:

```bash
git show f9c6d7b^:src/kinetic_augment/body_model/mp_to_smplx.py
```

It is prior art to learn from, not a working baseline — it was never verified, and its
`smoothing_weight` argument was accepted and never used.

---

## 4. Scope boundaries

### In scope for V1

- **Body only.** The 12 reliably corresponded MediaPipe pose landmarks:
  shoulders (11, 12), elbows (13, 14), wrists (15, 16), hips (23, 24), knees (25, 26),
  ankles (27, 28).
- Fitting genuine `body_pose` rotations and `global_orient` for those limbs.
- A measured noise floor, and a round-trip error decomposed by cause.

### Explicitly out of scope for V1

| Deferred | Why |
| --- | --- |
| Hands (MANO, 21 landmarks/hand) | Roughly triples the difficulty: wrist coupling, finger twist ambiguity, and MediaPipe's separate hand coordinate frame. V2. |
| Face (468 landmarks) | No correspondence to SMPL-X joints exists. Expression fitting is a separate problem. |
| Mesh vertices, collision geometry | Downstream of valid parameters. Nothing to do until V1 lands. |
| Rewiring the augmentation pipeline | V1 delivers a verified converter. Integration is a separate change with its own risks. |
| Temporal smoothing, real-time performance | Correctness first. Optimisation against an unverified result is wasted work. |

### What the 12 landmarks actually constrain

**Inference.** Of SMPL-X's 21 `body_pose` joints, the in-scope landmarks constrain **8**:

| Constrained (8) | Driven by bone |
| --- | --- |
| `left_hip`, `right_hip` | hip → knee |
| `left_knee`, `right_knee` | knee → ankle |
| `left_shoulder`, `right_shoulder` | shoulder → elbow |
| `left_elbow`, `right_elbow` | elbow → wrist |

The remaining 13 (`spine1/2/3`, `neck`, `head`, `left/right_collar`, `left/right_foot`,
`left/right_wrist`, and others) have **no MediaPipe landmark constraining them**. MediaPipe
provides no spine landmarks at all.

V1 policy, stated explicitly rather than left implicit:

- **Spine, neck, collars:** held at zero. Torso orientation is absorbed into `global_orient`,
  derived from the shoulder–hip frame.
- **Wrists:** held at zero. Wrist rotation requires hand landmarks (out of scope).
- **Feet, head:** held at zero. The existing `head → nose` and `foot → foot_index` entries in
  [`joint_mapping.py:175-187`](../src/kinetic_augment/body_model/joint_mapping.py:175) are marked
  `# Approximate` in the source and are **excluded** from the V1 correspondence set.

Reconstruction of unconstrained joints therefore reflects a documented default, not a fit.

---

## 5. Definition of done

V1 is complete when all of the following are true:

1. `MediaPipeToSMPLX.fit()` returns parameters that run through the real SMPL-X forward pass
   without error and produce a plausible mesh.
2. `SMPLXToMediaPipe.project()` reconstructs landmarks **by evaluating the SMPL-X model**,
   not by adding stored displacements.
3. A synthetic ground-truth test recovers known rotations to near machine precision (Step 5a).
4. Round-trip error on the five cached real clips is reported with the decomposition in Step 5b,
   and compared against the Step 1 noise floor.
5. The result is written down **whether or not it passes**. A measured failure is a valid V1
   outcome; an unmeasured success is not.

Point 5 is deliberate. The prior `0.5974 RMSE` figure in the roadmap is uninterpretable because
no one recorded what it measured ([05-project-status.md:36](research-guide/05-project-status.md:36)).

---

## 6. Approach

Analytic swing-twist inverse kinematics, with shape refinement added only if measurement
demands it.

**Core idea.** For each bone, the rotation carrying the rest-pose bone direction onto the
observed bone direction decomposes into a *swing* (the minimal rotation between two directions)
and a *twist* (rotation about the bone's own axis). Joint positions constrain the swing exactly
and constrain the twist not at all. So the swing is computed in closed form and the twist is set
by explicit policy — zero for V1 — rather than left to an optimiser to resolve arbitrarily.

Walking the kinematic chain parent-to-child and composing these rotations yields local
axis-angle values for each constrained joint.

**Why this over gradient-based fitting:** deterministic, fast, has no local minima, is
unit-testable bone by bone, and makes the twist ambiguity an explicit documented decision
instead of an implicit one. Gradient fitting remains available as a fallback (Section 9).

**What this leaves behind.** Every bone points in the correct direction; bone *lengths* remain
those of SMPL-X's default-shape skeleton. If the subject's proportions differ from the template,
each joint lands on the correct ray at the wrong distance, and the error compounds down the chain.

**Shape refinement, if needed.** Solving for `betas` and a global scale closes that gap. This is
not a patch over residual error — those coefficients exist precisely to match a specific person's
proportions, and the core step simply held them at zero. It breaks a chicken-and-egg dependency:
rotations need the skeleton's proportions, proportions need an approximate pose. The analytic
solve supplies the approximate pose, shape is solved against it, and rotations are optionally
re-solved against the corrected skeleton.

Shape is solved **once per sequence** from median bone lengths, not per frame. Shape does not
change during a clip, and the median rejects per-frame detector jitter.

**This step is evidence-gated.** It is built only if Step 6 shows length error is what pushes
the round-trip outside the noise floor. If the core step already lands inside, it is not built.

---

## 7. Steps

Step 0 is complete, so every step below is unblocked. The `SkeletonSource` seam in Step 2 is
still worth building — not to work around missing weights, but because the synthetic skeleton in
Step 5a is what makes the math testable against known ground truth.

### Step 0 — Unblock the environment — ✅ **COMPLETE (2026-10-01)**

**Verified now:** the weights were **never** in this repository.
[`.gitignore:219-224`](../.gitignore:219) excludes `models/smplx/*.npz`, `*.pkl`, and
`models/vposer/*.pt`; there is no Git LFS; and the only files ever committed under `models/`
were `version.txt`, `README.md`, and one vertex-ID array. **A fresh clone cannot produce them.**
They are licensed assets requiring registration at `smpl-x.is.tue.mpg.de`.

- [x] `SMPLX_NEUTRAL.npz` (103.7 MB) installed at `models/smplx/SMPLX_NEUTRAL.npz`, from the
      `models_smplx_v1_1.zip` release. Confirmed ignored by git.
- [x] `torch 2.14.1` and `smplx 0.1.28` installed into `.venv` (Python 3.11.15).
- [x] Forward pass verified. Three findings recorded below.

#### Finding 1 — `model_path` must be `"models"`, not `"models/smplx"`

**Verified now.** `smplx.create()` appends the model type to the given directory, so
`"models/smplx"` resolves to `models/smplx/smplx/` and raises
`AssertionError: Path models/smplx/smplx does not exist!`

The docstrings in
[smplx_wrapper.py:19](../src/kinetic_augment/body_model/smplx_wrapper.py:19) and
[body_model/\_\_init\_\_.py:15](../src/kinetic_augment/body_model/__init__.py:15) are **wrong**;
the runnable scripts ([test_phase1.py:324](../scripts/test_phase1.py:324),
[test_phase3.py:473](../scripts/test_phase3.py:473),
[render_landmarks.py:170](../scripts/render_landmarks.py:170)) are right.

#### Finding 2 — the forward pass returns **127** joints, not 55

**Verified now.** `output.joints` has shape `(1, 127, 3)`, while `J_regressor` is `(55, 10475)`.
SMPL-X returns its 55 model joints **plus 72 extra landmarks** appended after them.

The wrapper docstring's `(1, 55, 3)` claim
([smplx_wrapper.py:21](../src/kinetic_augment/body_model/smplx_wrapper.py:21)) is wrong, and
`SMPLXWrapper.num_joints` reports `55` — the `J_regressor` row count — while `forward()`
returns 127. **Index into the first 55 only.** Anything that trusts `num_joints` to describe the
`forward()` output is reading the wrong axis length.

#### Finding 3 — the repo's joint mapping is correct

**Verified now.** The model's `parents` array `(55,)` begins:

```
[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
```

This matches [`SMPLX_BODY_JOINTS`](../src/kinetic_augment/body_model/joint_mapping.py:32) exactly
for all 22 body joints. That table can be trusted; Step 3 should read `model.parents` directly
rather than re-deriving the tree.

#### Rest-pose reference skeleton (neutral, `betas=0`, metres)

The numbers `SkeletonSource` must supply, and the baseline Step 7 adjusts away from:

| Bone | Length | | Joint | Index |
| --- | --- | --- | --- | --- |
| Upper arm | 0.2676 | | `left_shoulder` | 16 |
| Forearm | 0.2531 | | `left_elbow` | 18 |
| Thigh | 0.3828 | | `left_wrist` | 20 |
| Shin | 0.4066 | | `left_hip` / `left_knee` / `left_ankle` | 1 / 4 / 7 |
| Shoulder width | 0.3159 | | | |

#### Note on `scripts/download_models.py --check`

It exits `1` on a correct V1 install. Its `REQUIRED_FILES` list
([download_models.py:25-30](../scripts/download_models.py:25)) demands `SMPLX_MALE.npz` and
`SMPLX_FEMALE.npz`, which **no code path in this repository requests** — every call site
defaults to `gender="neutral"`. Treat its failure as a false negative, or narrow the list.

### Step 1 — Measure the noise floor — ✅ **COMPLETE (2026-10-01)**

Reproduce with [`scripts/measure_noise_floor.py`](../scripts/measure_noise_floor.py).

Two independent estimators were used, because each fails differently and agreement between
them is the evidence that the number is real:

- **Method A** — 10th-percentile frame-to-frame displacement. On near-stationary frames the
  observed step is mostly jitter.
- **Method B** — second-difference estimator. For `p[t] = s[t] + n[t]` with `s` smooth,
  `D[t] = p[t-1] − 2p[t] + p[t+1]` annihilates constant velocity and leaves
  `Var(D) = 6σ²`, so `σ ≈ std(D)/√6`. Restricted to the slowest 25% of frames, since rapid
  direction changes violate local linearity.

#### Result

| Band | Method A | Method B |
| --- | --- | --- |
| x/y | 0.08% | 0.10% |
| z | 0.48% | 0.46% |
| 3-D | 0.73% | 0.46% |

> ~~V1 error budget: 0.73% of shoulder width~~ — **superseded.** Measured on the distorted
> image-space data (Finding 9).

**Re-measured on world landmarks (2026-10-04):**

| Band | Method A | Method B |
| --- | --- | --- |
| x/y | 0.07% | 0.06% |
| z | 0.06% | 0.07% |
| 3-D | 0.18% | 0.10% |

> **V1 error budget: 0.18% of shoulder width (≈ 0.6 mm at SMPL-X's 0.3159 m scale).**

Landmark validity now comes from MediaPipe's own per-landmark `visibility` rather than the
image-bounds heuristic, which does not apply to world coordinates. It confirms Finding 5
independently and more directly — the detector rates knees at **0.02** and ankles at **0.00**,
stating plainly that those are its prior and not an observation. Shoulders read 1.00, elbows
0.98, wrists 0.96.

#### Finding 4 — ~~MediaPipe's `z` is ~4.8× noisier than `x`/`y`~~ **SUPERSEDED**

Originally measured as `0.46%` versus `0.10%`, with the conclusion that a fitting objective
should down-weight `z`. **That conclusion was wrong and was never implemented.**

The 4.8× ratio was an artefact of the image-space anisotropy described in Finding 9, not a
property of MediaPipe's depth estimate. Re-measured on world landmarks, `z` is **1.2×** `x`/`y`
— close to isotropic. No axis weighting is needed.

A useful caution: this looked like a clean, actionable result and would have led to a fitting
objective tuned against a coordinate-system bug.

#### Finding 5 — half the in-scope landmarks are not observed in this data

**Verified now.** These are upper-body-framed signing videos. MediaPipe reports leg coordinates
by extrapolating past the image boundary, so they are model prior, not observation:

| Landmark | Observed in frame | 3-D noise |
| --- | --- | --- |
| `left_shoulder` / `right_shoulder` | 100% / 100% | 0.37% / 0.29% |
| `left_elbow` / `right_elbow` | 100% / 100% | 0.47% / 0.49% |
| `left_wrist` / `right_wrist` | 63% / 82% | 1.17% / 1.01% |
| `left_hip` / `right_hip` | **32% / 32%** | excluded |
| `left_knee` / `right_knee` | **0% / 0%** | excluded |
| `left_ankle` / `right_ankle` | **0% / 0%** | excluded |

**This changes Section 4.** Two consequences:

1. **Only 4 joints are constrainable on this data**, not 8: `left/right_shoulder` (via
   shoulder→elbow) and `left/right_elbow` (via elbow→wrist). The hip and knee rotations have no
   observed evidence behind them. Fitting legs here would be fitting to MediaPipe's guesses, and
   any round-trip error reported for them would be meaningless.
2. **`global_orient` cannot use the shoulder–hip frame** as Section 4 assumed — hips are observed
   in only 32% of frames. Derive the torso frame from **shoulders and nose** instead, as
   [`canonicalization.py:35-50`](../src/kinetic_augment/canonicalization.py:35) already does.

Legs remain in the method's scope; they are simply unverifiable until a full-body clip exists.
Record that as a limitation rather than quietly fitting them.

### Steps 2 & 3 — ✅ **COMPLETE (2026-10-04)**

Implemented test-first in [`skeleton.py`](../src/kinetic_augment/body_model/skeleton.py) and
[`kinematics.py`](../src/kinetic_augment/body_model/kinematics.py), with 15 tests in
[`tests/test_kinematics.py`](../tests/test_kinematics.py), all passing.

#### Result — forward kinematics matches SMPL-X

> Across 20 random body poses, our FK agrees with `model(...).joints[:, :22]` to a maximum
> absolute error of **1.96e-07 m (0.0002 mm)** — float32 round-off against float64 arithmetic.

This confirms the assumption Step 3 was built to test: SMPL-X's posed joints are a rigid
transform chain over the shaped rest skeleton, so pose-corrective blendshapes move mesh vertices
without displacing joints. **Requirement 2 of Section 1 now has a test behind it.** Parameters
we fit will mean to us exactly what they mean to the model.

#### Finding 6 — `flat_hand_mean=False` silently poisons the zero pose

**Verified now.** The first FK/SMPL-X comparison failed on exactly the finger joints
(26, 27, 29, 30, 32, 33, …) with a maximum error of 0.047 m. The cause was not the kinematics:
SMPL-X defaults to `flat_hand_mean=False`, which **adds a mean hand pose of up to 0.85 rad** to
whatever hand parameters it is given. Passing zeros does not produce a zero pose.

[`SMPLXSkeleton`](../src/kinetic_augment/body_model/skeleton.py) therefore constructs the model
with `flat_hand_mean=True`, so a zero rotation means no rotation and the rest pose is genuinely
the rest pose. Note that
[`SMPLXWrapper`](../src/kinetic_augment/body_model/smplx_wrapper.py:121) does **not** pass this
flag and so still carries the offset — harmless for V1, which touches no hand parameters, but it
must be settled before V2 fits hands, and any hand pose is only comparable between the two paths
once both agree on the flag.

#### Finding 7 — installing the weights un-skipped 10 broken tests

**Verified now.** `tests/test_body_model.py` now reports 2 failures and 10 errors. Neither is
caused by this work, and both were previously hidden:

- **10 errors** — `AssertionError: Path models/smplx/smplx does not exist!`
  [`test_body_model.py:40`](../tests/test_body_model.py:40) sets `MODEL_PATH = models/smplx`,
  the same wrong spelling as Finding 1. These tests were skipped while the weights were absent,
  so the bug sat undetected behind a skip marker.
- **2 failures** — `ImportError: cannot import name 'SimpleFitter'`, already recorded in
  [05-project-status.md:36](research-guide/05-project-status.md:36).

**Resolved 2026-10-04.** `MODEL_PATH` was serving two incompatible purposes: the asset check
needs `models/smplx`, while `SMPLXWrapper` needs `models`. Splitting them into `MODEL_PATH` and
`SMPLX_ASSET_DIR` fixed 8 of the 10. The suite is now **43 passed, 7 xfailed, 0 failures**.

The 7 remaining are marked `xfail` with reasons rather than deleted or rewritten, because they
fall into two groups that must not be conflated:

**Group 1 — one real production bug (1 test).** `test_rest_pose_joints` asserts
`joints.shape == (1, num_joints, 3)` and gets `(1, 127, 3) == (1, 55, 3)`. This is Finding 2
reproduced as a test failure: the **test is correct and the wrapper is wrong**.
`SMPLXWrapper.num_joints` reports the `J_regressor` row count while `forward()` returns the
appended-landmark count. This needs a production fix, not a test edit.

**Group 2 — six tests written against the API deleted in `f9c6d7b`.** They are a precise
inventory of what the bone-vector encoding gave up:

| Failure | Capability it describes |
| --- | --- |
| `__init__() got an unexpected keyword argument 'num_iterations'` | a fitter that optimises |
| `project() missing 1 required positional argument: 'target_mp'` | a projector that reconstructs from parameters **alone** |
| `cannot import name 'SimpleFitter'` | a model-free fitter |

The middle row is the diagnostic one. A genuine converter does not need the original landmarks
handed back to it; the current signature *requires* `target_mp` precisely because it cannot
reconstruct anything, only offset what it is given. These six tests already encode V1's target
contract, so rewriting them to match the current hack would cement the broken design and force
them to be un-written at Step 4. **When Step 4 lands they should flip to XPASS** — that is the
signal the capability has been restored.

---

### Step 2 — `SkeletonSource` interface

A narrow seam that decouples all fitting work from the blocked weights.

- Interface: rest-pose joint positions `(J, 3)`, parent index array `(J,)`, joint names
- `SyntheticSkeleton`: hardcoded plausible proportions. No weights required.
- `SMPLXSkeleton`: wraps the existing [`SMPLXWrapper`](../src/kinetic_augment/body_model/smplx_wrapper.py). Requires Step 0.

Fitting code depends on the interface only, so swapping in real SMPL-X changes no fitting logic.

### Step 3 — Forward kinematics and projection (SMPL-X → MediaPipe)

- Read the parent tree from `model.parents` directly (Finding 3), not a re-derived table
- Compose rotations down the parent chain to produce world joint positions
- Index into the **first 55 joints only** — `forward()` returns 127 (Finding 2)
- Select corresponded joints and emit a MediaPipe-shaped array
- **Test:** zero rotations reproduce the rest pose exactly

**Store and invert the normalisation.** Fitting happens in SMPL-X's metric, pelvis-centred space;
MediaPipe lives in image-normalised coordinates. The similarity transform between them — roughly
7 numbers (scale, rotation, translation) — is **round-trip state that must be carried alongside
the parameters**. Omit it and reconstruction lands in metres around the hip while the input sits
in `[0,1]` image coordinates, looking catastrophically wrong despite a perfect fit.

This is *not* the residual channel rejected in §2. The distinction is dimensionality and meaning:
a similarity transform is 7 numbers with a physical interpretation (effectively where the camera
stood), whereas the residual would be 99 unexplained numbers per frame absorbing exactly the
error this study exists to measure.

Build this before the inverse. It is the thing the inverse is checked against.

### Step 4 — ✅ **COMPLETE (2026-10-04)**

Implemented test-first in [`swing_ik.py`](../src/kinetic_augment/body_model/swing_ik.py), with 11
tests in [`tests/test_swing_ik.py`](../tests/test_swing_ik.py). Full suite: **54 passed,
7 xfailed**.

#### Result — bone directions are reproduced essentially exactly

Run over all 410 frames of the five cached clips:

| Bone | Solved | Median angular error | Max |
| --- | --- | --- | --- |
| Left shoulder | 100% | 0.0045° | 0.026° |
| Left elbow | 100% | 0.0036° | 0.025° |
| Right shoulder | 100% | 0.0050° | 0.026° |
| Right elbow | 100% | 0.0036° | 0.026° |

No NaNs, no unsolved frames. The residual is float round-off through the
matrix → axis-angle → matrix conversion, not fitting error.

**This is the decomposition Section 6 predicted, now confirmed.** Direction error is
effectively zero *by construction*, because the swing rotation solves it in closed form. So any
round-trip position error Step 5 finds must come from **bone length** — SMPL-X's default-shape
skeleton versus the subject's actual proportions — and not from the fit missing. That is exactly
the condition under which Step 7's shape solve is the right remedy.

#### ~~Deliberate deferral — `global_orient` is not estimated~~ **RETRACTED**

The original claim was that leaving `global_orient` unestimated "provably cannot affect the V1
gate," because with spine and collar at zero the swing solve places each bone along its observed
direction regardless of root rotation.

**That claim is false.** It conflates bone *direction* with joint *position*. Directions are
rotation-invariant; positions are not. Shoulder rotations move elbows and wrists but **cannot
relocate the shoulders themselves**, so a rotated root leaves the shoulders at rest position with
no parameter able to correct them. Procrustes alignment cannot absorb it either: the rotation
that would carry the shoulders into place simultaneously rotates the arms *away* from directions
that were already correct.

**Measured counterexample** — same skeleton for truth and fit, so bone lengths are identical by
construction and root rotation is the only variable:

| Root rotation | Mean position error | Max |
| --- | --- | --- |
| 0° | 0.00 mm | 0.00 mm |
| 10° | 9.87 mm | 81 mm |
| 20° | 16.55 mm | 119 mm |
| 45° | 47.04 mm | 307 mm |

The claim holds only in the degenerate 0° case, which is what made it look true.

Root orientation is therefore a **real error term**, quantified in the revised Step 6 below.

#### Finding 8 — the all-zeros sentinel is a live hazard

**Verified now.** MediaPipe marks undetected landmarks with exact zeros, which is
indistinguishable from a landmark legitimately at the origin. This caused two separate false
results while building Step 4, both in test fixtures that placed a joint at the origin: a test
that passed for the wrong reason, and a test that failed against correct code.

Real image-space landmarks are never exactly at `(0, 0, 0)`, so the sentinel is safe for actual
MediaPipe input. It is **not** safe for synthetic or canonicalised data, where the origin is a
natural place for a joint to sit. The same hazard is already noted for canonicalisation in
[03-math-and-geometry.md:142](research-guide/03-math-and-geometry.md:142). Any future code
generating synthetic landmarks must keep joints off the origin, or carry an explicit validity
mask instead.

---

### Step 4 — Analytic swing IK (MediaPipe → SMPL-X) *(core deliverable)*

- Derive the torso frame from shoulders and hips → `global_orient`
- Per bone: compute the swing rotation from rest direction to observed direction
- Apply the twist policy (zero for V1)
- Convert world rotations to parent-relative local rotations, in chain order
- Write unconstrained joints as zero per the Section 4 policy

### Step 5 — ✅ **COMPLETE (2026-10-04)**

Re-run on world landmarks extracted by
[`scripts/extract_world_landmarks.py`](../scripts/extract_world_landmarks.py) (409/410 frames
detected). The blocking data problem is recorded as Finding 9 below.

#### Result

| | Value |
| --- | --- |
| Part A — synthetic ground truth (**positions**) | **2.99e-16 m** (machine precision) |
| Part B — round-trip error | **5.94%** of shoulder width |
| Step 1 noise floor | 0.18% |
| Verdict | **33× the budget — above the floor** |

Per joint (median / p95, % shoulder width): shoulders 6.2 / 11.6 and 6.5 / 10.6, elbows
5.7 / 9.1 and 4.2 / 8.9, wrists 6.9 / 15.5 and 6.5 / 14.8. Error is spread evenly rather than
accumulating down the chain, which already argues against a kinematics fault.

**Part A measures positions, not rotations.** It must not be described as recovering the
original rotations: twist is underdetermined, so many rotation vectors produce identical joint
positions. Exact *position* recovery is the correct test, and rotation recovery should only be
asserted for cases where the ambiguity is explicitly pinned
([`test_recovers_a_known_single_rotation`](../tests/test_swing_ik.py) uses a pure-swing rotation
for exactly that reason). Analytic swing is well-founded — HybrIK
([arXiv:2011.14672](https://arxiv.org/abs/2011.14672)) uses the same decomposition, and resolves
twist from additional image evidence rather than from positions.

**The reported number is an *aligned* residual.** The harness fits a fresh rotation, translation
and scale per frame before scoring
([`measure_round_trip.py`](../scripts/measure_round_trip.py)). That is a useful diagnostic, but
the transform is **not part of the converter's output**, so this is not yet the error a consumer
of the converter would see. Direct and aligned error must be reported separately, and the
converter's contract must say what coordinate state it returns.

#### The decomposition answers Step 6 cleanly

| Component | Value | Reading |
| --- | --- | --- |
| Bone **length** error | 6.66% | **dominant** |
| Bone **direction** error | exact by construction | fit is not missing |

Observed bone lengths against the SMPL-X neutral rest skeleton:

| Bone | SMPL-X | Observed | Ratio |
| --- | --- | --- | --- |
| L upper arm | 0.2676 | 0.2388 | 0.893 |
| L forearm | 0.2531 | 0.2677 | 1.058 |
| R upper arm | 0.2749 | 0.2429 | 0.884 |
| R forearm | 0.2498 | 0.2458 | 0.984 |

The ratios are **not uniform** — spread 0.071. That is the decisive detail. A single global
scale cannot reconcile them, and the Procrustes alignment already absorbs uniform scale, so
what remains is a *proportion* difference rather than a size difference.

#### Finding 10 — the five clips are different people, and the lengths are partly a prior

**Verified now.** The table above pools five clips. They are **not one subject**: the videos
come from at least three sources and show at least four different people. Per clip:

| Clip | Shoulder W | L upper | L fore | R upper | R fore | fore/upper |
| --- | --- | --- | --- | --- | --- | --- |
| 01986 | 0.3327 | 0.2433 | 0.2765 | 0.2487 | 0.2487 | 1.068 |
| 01987 | 0.3337 | 0.2406 | **0.1909** | 0.2506 | 0.2207 | **0.837** |
| 01988 | 0.3253 | 0.2569 | 0.2715 | 0.2362 | 0.2569 | 1.072 |
| 01991 | 0.3467 | 0.2285 | 0.2388 | 0.2598 | 0.2718 | 1.046 |
| 01992 | 0.3296 | 0.2291 | 0.2748 | 0.2339 | 0.2353 | 1.103 |
| **spread** | **2.1%** | 4.4% | 13.1% | 3.9% | 7.1% | 9.3% |

`01987` is the one male signer and is also the proportion outlier, so world landmarks do carry
real individual signal. But shoulder width varies by only **2.1%** across one man and three
women (four people over five clips — `01991` and `01992` show the same signer), which is far
tighter than adult anthropometry allows. `pose_world_landmarks` are model
output fitted to the image, carrying MediaPipe's own body prior — not an anthropometric
measurement.

**Two consequences for Step 7:**

1. **Fit `betas` per clip, not globally.** This answers the open question in Section 10 for V1:
   the clips are different bodies, so a shared shape vector would be fitting an average of
   strangers.
2. **State the objective accurately.** Step 7 reconciles SMPL-X's template with MediaPipe's
   *estimate* of each body. That is valid for the round-trip gate, which asks only whether the
   conversion preserves what the input carried. It is **not** recovery of a person's true
   proportions, and no write-up should claim it is.

#### ~~Step 6 verdict: length error dominates → proceed to Step 7~~ **RETRACTED**

That verdict was reached without ever testing the root contribution, so it was unfounded. The
decomposition below runs each candidate fix as an **oracle** — the best any estimator of that
kind could do — to measure what each actually explains.

| Configuration | Pooled error | Reduction |
| --- | --- | --- |
| Baseline (current) | 5.94% | — |
| Root orientation only | 5.13% | 13.7% |
| Bone length only | 3.86% | 35.0% |
| **Both** | **2.64%** | **55.5%** |

Shape is the larger single contributor, so the original verdict was right in direction. But two
things it asserted are wrong:

1. **Root is a real term, not zero.** It explains 13.7% on its own.
2. **Neither fix reaches the gate, nor do both together.** With *oracle* shape and *oracle* root,
   **44.5% of the error remains unexplained**. Step 7 alone was never going to close this.

The residual needs its own investigation before more parameters are added. The leading candidate
is the joint-centre offset named in Section 2: MediaPipe's `left_shoulder` is a surface-adjacent
keypoint while SMPL-X's is an internal rotation centre, and nothing in the current pipeline
models that difference.

Corroborating evidence: the per-frame Procrustes fits a median scale of **0.9726** (5th–95th
percentile 0.91–1.02) when both spaces are metric and the true scale is 1.0. A free scale
parameter absorbing ±5% per frame is a symptom of systematic correspondence error, not noise.

---

### Step 5 — the data problem that had to be fixed first

Harness built: [`scripts/measure_round_trip.py`](../scripts/measure_round_trip.py), with
[`alignment.py`](../src/kinetic_augment/body_model/alignment.py) (6 tests) for the
Umeyama similarity used to compare SMPL-X's metric space to MediaPipe's.

#### Part A — synthetic ground truth: ✅ PASS

> 200 random poses, worst joint position error: **2.99e-16 m**.

Rotations we generated are recovered to machine precision. **The mathematics is correct.**

#### Part B — real clips: 50.05% of shoulder width, 68× the Step 1 budget

Before treating that as a result, note it contradicts Step 4, which measured bone-direction
error at 0.004°. Here the same bones read ~5° after alignment. A fit that is exact in direction
cannot produce 5° of direction error unless the alignment is fighting the data.

#### Finding 9 — the cached landmarks are not in a Euclidean space

**Verified now.** The forearm-to-upper-arm length ratio, which anatomy fixes at roughly 0.95:

| Source | fore/upper |
| --- | --- |
| SMPL-X rest skeleton | 0.946 |
| Cached landmarks, xyz | 1.228 |
| Cached landmarks, xy only | 0.614 |

The cached values bracket the truth instead of approximating it, because the axes are not on a
common scale. MediaPipe normalises **x by image width and y by image height**, so any non-square
video is anisotropically stretched — and the five clips have **three different aspect ratios**
(320×240 and 640×480 at 1.333, 1280×720 at 1.778, 288×192 at 1.500). There is no single
correction, because the distortion differs per clip.

**Root cause.** [`test_phase1.py:146-147`](../scripts/test_phase1.py:146) captures
`results.pose_landmarks` — image-normalised coordinates with a predicted pseudo-depth.
MediaPipe also exposes `results.pose_world_landmarks`: metric, hip-centred, in real units.
That is the correct input for fitting a metric body model.

Re-extracting clip `01986` both ways:

| | fore/upper | upper-arm length |
| --- | --- | --- |
| `pose_landmarks` (cached) | 1.567 | n/a — not metric |
| `pose_world_landmarks` | **1.066** | **0.2433 m** |
| SMPL-X rest | 0.946 | 0.2676 m |

World landmarks were present in **83/83 frames** and give anatomically plausible proportions.
The residual 13% is ordinary individual variation — precisely what `betas` exists to absorb, and
exactly the case Step 7 was scoped for.

#### Consequence

A metric body model cannot be fitted to anisotropically distorted image coordinates. This is a
correctness requirement, not a preference. Two things follow:

1. **Part B's 50% is not a round-trip result** and must not be reported as one. It measures
   coordinate distortion, and would have been published as a finding about SMPL-X.
2. **The Step 1 noise floor (0.73%) is also measured on distorted data** and must be re-measured
   once the input is fixed. Treat that number as provisional.

Steps 2, 3 and 4 are unaffected — swing IK reads directions and is scale-invariant, and the
synthetic verification never used this data.

---

### Step 5 — Round-trip harness and error decomposition

**5a — Synthetic ground truth** *(no weights needed; the rigorous test)*

Generate random valid rotations → forward kinematics → landmarks → fit → compare against the
rotations you started from. The true answer is known exactly, which is never true of real data.

**Pass condition:** recovered rotations and positions match to near machine precision. Failure
here is a bug in the math, isolated from every question about real-world data quality.

**5b — Real data** *(needs Step 0)*

Run the five cached clips through `P → θ → P'` and report three separate quantities:

| Metric | Question it answers |
| --- | --- |
| **Bone direction error** (angular, per bone) | Did the fit miss? |
| **Bone length error** (per bone) | What can SMPL-X structurally not represent? |
| **Joint position error** (MPJPE, % shoulder width) | The headline round-trip number. |

Reporting these separately is the point. A single RMSE conflates a fitting bug with a
representational limit, and they call for opposite responses.

**Compute every metric over the 12 fitted landmarks only.** `project()` copies the other 531
landmarks — face, hands, and the unconstrained pose points — straight through from the input
frame. An error averaged over all 543 would be measuring copied data against itself, and could
report an impressively small number while the fit is completely wrong. This is the most likely
way to manufacture an unfalsifiable success, and plausibly how the historical `0.5974 RMSE`
became uninterpretable: nobody recorded which landmarks it covered.

### Step 6 — Decision gate — ✅ **RUN (2026-10-04)**

The original gate assumed a single dominant cause. The oracle decomposition in Step 5 shows
there are at least three, so the gate is replaced by its result: **shape 35%, root 13.7%,
unexplained 44.5%.** See the retraction above.

---

## Revised plan (2026-10-04)

Steps 7–9 replace the original conditional Step 7. The change is driven by two measured facts:
root orientation is a real error term, and shape plus root together leave nearly half the error
unexplained.

### Step 7 — ✅ **COMPLETE (2026-10-04)**

**Wiring decision: keep the paths separate and mark the legacy one.** The new converter is the
V1 deliverable. `MediaPipeToSMPLX` and `SMPLXToMediaPipe` remain, because image-space callers
depend on them, but now emit a `DeprecationWarning` naming what they actually do and pointing to
`converter.py`. This keeps to the original Section 4 boundary, which put pipeline integration
outside V1.

The six legacy `xfail` tests are retired in favour of
[`tests/test_converter.py`](../tests/test_converter.py), which encodes the correct contract. They
described an optimisation-based fitter that was removed in `f9c6d7b` and deliberately not
rebuilt, so keeping them as permanent `xfail`s would have recorded a rejected design as pending
work. Two tests replace them, asserting that each legacy class warns and names the converter.

Suite: **72 passed, 2 xfailed.** The two are intentional: the `SMPLXWrapper.num_joints` bug
(Finding 2, being fixed separately) and the `strict=True` rotated-root case that is Step 8's
target.

Pipeline integration remains open and needs a camera model to project world-space fits back to
the classifier's image-space format — recorded as V2 scope.

---

### Step 7 — implementation record

Implemented test-first in [`converter.py`](../src/kinetic_augment/body_model/converter.py) with
8 tests in [`tests/test_converter.py`](../tests/test_converter.py), plus a rigid mode for
[`alignment.py`](../src/kinetic_augment/body_model/alignment.py). Full suite: **71 passed,
8 xfailed.**

#### The contract, as built

```python
fit_body(landmarks, validity, skeleton) -> BodyFit   # body_pose, global_orient, transl, betas
reconstruct_body(fit, skeleton)         -> (33, 3)   # NaN where V1 does not model
```

- **Reconstruction takes only the parameters.** A test asserts its signature is exactly
  `(fit, skeleton)`, so the displacement-encoding failure mode cannot return.
- **The output is pure SMPL-X.** The rigid frame transform folds exactly into SMPL-X's own
  `global_orient` and `transl`, compensating for `global_orient` pivoting about the root's rest
  position rather than the origin. No side-channel coordinate state.
- **Verified against the real model:** SMPL-X's forward pass on the returned parameters
  reproduces `reconstruct_body` to within 1e-5 m.
- **Rigid, not similarity.** Scale is fixed at 1.0 because both spaces are metric.
- Unmodelled landmarks return NaN rather than invented values; invalid landmarks never influence
  the fit; fewer than three valid correspondences raises rather than guessing.

#### Result — direct versus aligned

| | Value |
| --- | --- |
| Synthetic, root unrotated | **5.3e-15 m** — exact |
| Synthetic, root rotated 30° | **70 mm worst** — the Step 8 gap, now visible in the harness |
| **Real clips, direct** | **6.55%** of shoulder width — the number a consumer gets |
| Real clips, aligned | 5.94% — the figure previously reported |

**The previously reported 5.94% was the aligned figure.** A free scale was hiding 0.61
percentage points. Every earlier comparison to 5.94% should be read as optimistic by that much.

Per joint, direct median / p95, with coverage: shoulders 6.43 / 12.56 and 7.80 / 12.62 (100%),
elbows 6.04 / 9.58 and 4.98 / 10.73 (100%), wrists 7.85 / 17.33 (87%) and 7.01 / 16.63 (89%).

The rotated-root case is captured as a `strict=True` xfail in `test_converter.py`: it fails today
and will fail the suite *loudly* the moment Step 8 makes it pass, forcing the marker's removal.

#### Blocked — wiring into the public classes

The review is right that the verified maths is not yet reachable through `MediaPipeToSMPLX` and
`SMPLXToMediaPipe`. But this is not a drop-in swap, for three reasons:

1. **Every existing caller passes image-space data.** `scripts/render_landmarks.py` and
   `scripts/test_phase2.py` call `fit(frame)` on 543-landmark image-space frames — exactly what
   Finding 9 shows cannot support a metric fit. Rewiring would make them silently wrong.
2. **The framework's interchange format is image space.** Augmented landmarks feed a classifier
   as `(T, 543, 3)` image-space arrays. Fitting in world space produces world-space output, and
   no camera model exists to project it back.
3. **The six legacy `xfail` tests describe a rejected design.** They expect a
   `num_iterations` optimiser argument and random `(543, 3)` input; analytic IK was chosen
   deliberately and has no iterations.

This is an architecture decision about the framework's data format, not a V1 implementation
detail, and is raised for a decision rather than resolved unilaterally.

---

### Step 7 (original text) — Define and enforce the converter's contract

Nothing further should be measured until the deliverable's interface is pinned down, because the
current headline is an aligned residual whose transform the converter does not return.

> **Contract.** World-coordinate body landmarks + per-landmark validity → SMPL-X parameters
> **and the coordinate state needed to reconstruct** → world-coordinate body landmarks.

- Decide explicitly what coordinate state is returned. Both spaces are metric, so a **rigid**
  transform should suffice; the fitted scale of 0.9726 is evidence of correspondence error, and
  a free scale parameter hides it. Prefer rigid, and report any residual scale as a finding.
- Report **direct** reconstruction error (the number a consumer sees) **and** aligned error
  (diagnostic) side by side, per joint, with median, p95 and observation coverage.
- **Wire the verified maths into the public classes.** `MediaPipeToSMPLX` and `SMPLXToMediaPipe`
  still use the displacement encoding of Section 3; the new modules are parallel to them, not
  connected. The six `xfail` tests in `tests/test_body_model.py` flip when this lands and are the
  completion signal.

### Step 8 — ✅ **COMPLETE (2026-10-05)**

Implemented test-first in [`converter.py`](../src/kinetic_augment/body_model/converter.py)
(`torso_rotation`, and a rewritten `fit_body`), with `SMPLXSkeleton.rest_nose` added in
[`skeleton.py`](../src/kinetic_augment/body_model/skeleton.py). Full suite: **81 passed,
2 xfailed** — the `num_joints` bug (Finding 2) and the strict collar case below.

#### What changed

Fitting now runs in an order where nothing fights:

1. **Root from the torso.** The shoulder axis sets two rotational degrees of freedom; the nose,
   when visible, sets the third.
2. **Arms by swing IK solved *relative to* that root**, so each bone points along its observed
   direction in world space.
3. **Translation by least squares** over the valid correspondences. No rotation is fitted after
   step 2, because that is what turned correct arm directions away from the observations.

The shoulders now anchor the frame, so `fit_body` requires both and raises without them. This
replaces the earlier "three correspondences" rule, which belonged to the whole-body rigid fit.

#### Synthetic cases added — the ones that could have caught the retracted claim

| Case | Result |
| --- | --- |
| Root rotated 30° (the reviewer's counterexample) | **exact**, 2.9e-16 m (was 70 mm) |
| Spine bend at all three spine joints | exact |
| Root rotation recovered from shoulders + nose | exact, < 1e-9 rad |
| Missing wrist with a rotated root | remaining joints exact |
| Different body shape (`betas` ≠ 0) | **bone directions exact**; positions differ, honestly |
| Collar motion (a shrug) | **not representable** — `strict=True` xfail, see below |
| Both shoulders required | raises without them |

The body-shape case failed under the old converter: its whole-body rigid fit bent bone directions
to compensate for length mismatch. That directly confirms the review's point that a final
rotation fights the solved arms.

#### Finding 11 — torso pitch is invisible to the arm landmarks (a gauge freedom)

**Verified now, by test and on real data.** Rotation about the shoulder axis moves neither
shoulder, and the arms are solved in world directions, so it cannot change any reconstructed arm
landmark. Only the shoulder-axis *direction* — two of the root's three degrees of freedom — is
observable from V1's correspondences.

- **Test:** `test_torso_pitch_does_not_affect_arm_reconstruction` fits the same frame with and
  without the nose. The roots disagree by more than 5°; the arm reconstructions agree to 1e-9 m.
- **Real data:** the earlier decomposition estimated the root using the SMPL-X *head joint* as the
  nose's counterpart, which points **29.9°** away from the true nose vertex about the shoulder
  axis. Despite that 30° pitch error, its "root only" result was **5.13%** — identical to the
  corrected converter's aligned figure. A 30° error in the gauge direction changed nothing.

Consequences. The nose makes `global_orient` anatomically plausible (an upright, correctly facing
torso) but **cannot improve round-trip error**. And it explains why fixing the root helped only
13.7%: signers mostly face the camera squarely, and for a frontal torso the shoulder axis of
MediaPipe's world frame and SMPL-X's rest frame already agree — the ~180° difference between them
is a rotation *about* that axis, which is gauge.

`SMPLXSkeleton.rest_nose` uses SMPL-X's own nose vertex (vertex 9120, which the model also emits
as output joint 55) rather than the head joint, so the pitch that is fitted is the anatomically
correct one.

#### Collar motion is a real V1 limitation

A shrug rotates one shoulder about its collar joint, changing the shoulder geometry in a way no
root rotation reproduces. V1 holds the collars at zero, so it cannot represent this, and shoulder
movement is meaningful in some signs. Recorded as a `strict=True` xfail so the marker is removed
the moment collars are fitted.

#### Result on real data

| | Step 7 | Step 8 |
| --- | --- | --- |
| **Direct round-trip error** | 6.55% | **6.02%** |
| Aligned (diagnostic) | 5.94% | 5.13% |
| Arm direction error | 2–3° | **0.00°** |

#### The `01988` question, resolved

The plan required understanding why root refinement made `01988` slightly worse before
accepting an estimator. A **paired, per-frame comparison with bootstrap confidence intervals**
separates two different effects:

| Clip | Direct, new − old | 95% CI | Frames improved |
| --- | --- | --- | --- |
| 01986 | −0.96 | [−1.19, −0.68] | 82% |
| 01987 | −0.35 | [−1.08, +0.16] | 81% |
| 01988 | **+0.35** | **[+0.09, +0.67]** | 44% |
| 01991 | +0.27 | [−2.66, +0.45] | 47% |
| 01992 | −0.28 | [−0.41, −0.13] | 68% |

1. **The originally flagged regression was noise.** It was measured under the *aligned* metric,
   where the paired difference for `01988` is +0.07 with CI [−0.23, +0.15].
2. **A different, real regression appears under the *direct* metric** — and the harness's median
   over all joint samples reports `01988` *improving* (5.57% → 5.40%). When two reasonable
   aggregations disagree on the sign, the effect is mixed across joints, and the per-joint split
   shows how: `01988`'s shoulders improve while its wrists worsen by 1.21 and 0.71 points.

The mechanism is a trade-off, not an estimator fault. The old converter fitted one extra rotation
over all six points, and that rotation bent every arm bone **1.9–3.2°** away from what MediaPipe
observed, which spread bone-length mismatch across the body. The new converter keeps directions
exact (0.00° on every clip), so the same length mismatch accumulates at the wrists instead.
`01988` is tied for the most frontal torso (3.8° yaw), so it gains least from the corrected root,
and the trade-off shows; `01986`, the most rotated (7.9°), gains most.

**Decision: accept the new estimator.** The old converter's lower error on `01988` was error
moved out of positions and into joint angles, where the round-trip metric cannot see it. For a
framework whose purpose is to perturb joint angles, a 2–3° distortion of every elbow and
shoulder is not acceptable. The wrist error that replaced it is honest length mismatch — the
quantity Step 10 targets.

---

### Step 8 (original text) — Fix root and shoulder-anchor handling

The largest *correctly understood* gap, and a prerequisite for interpreting any shape fit.

- Estimate `global_orient` from the shoulders-and-nose frame. Nose visibility is **1.00** in all
  five clips (**verified now**), so the frame is available on every usable frame.
- Extend the synthetic suite to cover what it currently cannot distinguish: **rotated roots**,
  torso motion, differing body shapes, and missing landmarks. The retracted claim above survived
  precisely because no synthetic case had a rotated root.
- Note that root-only refinement made clip `01988` slightly *worse* (5.06% → 5.11%). Understand
  that before accepting any estimator.

### Step 9 — Investigate the unexplained 44.5%

Do not add parameters before the dominant unexplained term is identified. Adding `betas` now
risks absorbing a correspondence bias into a shape estimate and reporting it as anatomy.

Leading hypothesis: the **joint-centre offset** of Section 2 — MediaPipe's shoulder keypoint sits
near the body surface, SMPL-X's is an internal rotation centre. Testable by estimating a
per-joint offset and measuring what it explains, exactly as the Step 5 oracles did.

### Step 10 — Shape, with a prior and a bounded claim *(only after Step 9)*

- Four arm lengths plus shoulder width cannot determine 10 `betas` plus scale. The problem is
  **underdetermined** and needs regularisation toward the shape prior; an unregularised fit will
  find a shape that matches five numbers and means nothing.
- Keep shape **shared across a sequence** but **per clip** — Finding 10 establishes the clips are
  different people.
- Compare against analytic IK as baseline, and consider a small confidence-weighted optimisation
  using `visibility` as weights, initialised from the analytic solution.
- **Bound the claim.** A better reconstruction does not establish that a signer has the inferred
  proportions; detector bias and correspondence offsets remain live alternatives.

---

## 8. Deliverables

1. A working `MediaPipeToSMPLX` producing genuine SMPL-X parameters.
2. A working `SMPLXToMediaPipe` that evaluates the real model.
3. A round-trip test suite: synthetic exactness plus real-data error decomposition.
4. A results document recording the measured noise floor, the three error metrics, and the
   Step 6 verdict — including a negative verdict.

---

## 9. Fallbacks

Ordered by what the measurement would be telling us.

| Trigger | Fallback |
| --- | --- |
| Length error survives Step 7 — `betas` cannot span the subject's proportions | Accept and report it. This is a genuine finding about SMPL-X's shape space, not a failure. |
| Analytic IK accumulates error down long chains | **Approach A:** gradient-based fitting (SMPLify-X style), optimising all parameters jointly against a global objective. Adam loop recoverable from `f9c6d7b^` as a starting reference. |
| Direction error persists — joint centres offset systematically | Calibrate a per-joint correspondence offset from data, then re-fit. |
| Round-trip cannot reach the noise floor by any route | **Residual channel:** store `θ` plus a per-landmark residual, making reconstruction exact by construction, and report residual magnitude as the measure of what SMPL-X cannot represent. Declined for V1 (Section 2); legitimate as a last resort *provided the residual is reported, not hidden*. |
| ~~Weights never arrive~~ | Resolved 2026-10-01 — see Step 0. |

---

## 10. Open questions

- Which twist policy for V1 — zero, or minimal-twist propagated along the chain? Zero is assumed
  above; it does not affect joint positions, so it does not affect the round-trip gate. It does
  affect mesh plausibility, which V1 does not measure.
- Does the shoulder–hip torso frame suffice for `global_orient`, or does depth noise in
  MediaPipe's `z` make it unstable? Testable in Step 1.
- Should `betas` be shared across all clips from one signer rather than per clip? Relevant to V2;
  V1 has too few subjects to tell.

---

## 11. Known downstream defect — temporal incoherence in intrinsic augmentation

**Verified now (source).** Outside V1 scope, since §4 defers pipeline integration. Recorded here
because it is the first thing that will break *after* V1 lands: a correct round-trip feeding a
temporally incoherent augmenter still produces unusable training data.

**The classifier is a sequence model.**
[`lstm_classifier.py:114`](../src/kinetic_augment/evaluation/models/lstm_classifier.py:114) reads
`(batch, T, 543, 3)` and takes **only the final hidden state** (line 128), emitting one label for
the whole clip. A training example is a clip, not a frame, and
[`dataset.py`](../src/kinetic_augment/pipeline/dataset.py) multiplies the dataset by
`num_augmented_versions` copies of each clip, all sharing its label.

An augmentation must therefore be **temporally coherent**: one perturbation sampled per clip and
applied to all `T` frames.

**Extrinsic augmentations do this correctly.** `GlobalRotation` samples a single rotation and
applies it to the flattened `(T*J, 3)` clip, so every relative motion inside the clip survives.

**Intrinsic augmentations do not.**
[`intrinsic.py:133-136`](../src/kinetic_augment/augmentations/intrinsic.py:133) draws fresh noise
*inside* the batch-row loop:

```python
for b in range(batch_size):
    noise = np.random.randn(3) * stddev     # fresh noise every row
    body_pose[b, pose_idx:pose_idx + 3] += noise
```

and [`mp_to_smplx.py:96`](../src/kinetic_augment/body_model/mp_to_smplx.py:96) sets
`B = landmarks.shape[0]`, turning a clip's `T` frames into `B` independent batch rows. Every
frame therefore receives an unrelated perturbation.

**Consequence.** Instead of *"the same sign, performed with the elbow held 8° differently"*, the
augmenter produces *"a sign in which the elbow twitches randomly every frame"* — high-frequency
noise in exactly the temporal dimension an LSTM exists to read. `BodyProportionVariation` has the
same defect, varying body shape per frame within a single clip.

**Root cause is structural, not a typo.** No parameter dictionary marks that a batch is one
temporal sequence, and `fit()` has no concept of time, so an intrinsic operation cannot
distinguish 32 unrelated poses from a 32-frame clip.

**Requirement for whoever integrates V1:** sample augmentation parameters **once per clip** and
apply them across all `T` frames. Per-frame is correct for the *geometry*; once-per-clip is
correct for the *augmentation*. These are different time constants and the current code conflates
them — the same distinction §6 already makes for shape.
