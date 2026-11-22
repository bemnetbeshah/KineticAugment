# Hand Scale Prototype: Letting Hand Size Vary Within a Clip

**Question.** The hand round trip fits one shape per track. MediaPipe's metric hand changes size
from frame to frame. Would letting the hand's size vary within a clip make the round trip more
accurate? If it would, what does that do to depth and hand-to-face contact?

**Short answer.** One extra number per frame, a uniform scale about the wrist, removes about as
much error as giving every frame its own 10 shape values: 5.81 → 3.79 mm against 3.93 mm. The
scale can be predicted from neighbouring frames, so it is not frame noise: predicted for frames it
never saw, it keeps 68% of that gain (4.43 mm). But the size change is MediaPipe's error, not the
signer's anatomy: the palm, which is rigid in a real hand, changes size by a median 10.8%. That
same error moves the hand in depth. Holding the hand at constant size cuts 3-D contacts that the
image contradicts (hand clear of the face) from 26 to 3. It also cuts contact on near-face frames
from 52 to 31, and the labels to say whether those were real do not exist yet.

## 1. Setup

- **Clips.** 26 clips: the 5 V1.2 pilot clips (`data/test_output`), the 20 V1.3 evaluation clips
  (`data/round_trip_eval`) and 08925 (fingers on the cheek in all 41 frames; from the head
  placement spike). 35 hand tracks with at least 10 frames. MANO only; V1.2 found SMPL-X within
  0.1 mm of MANO on every ladder row.
- **Prototype.** `kinetic_augment.body_model.hand_scale`. `fit_hand_scales` fits one shape and
  one scale per frame jointly with every frame's pose, in one L-BFGS run (the hand optimiser's new
  `scale` option). The scale is not a MANO or SMPL-X parameter: the model's hand is posed, then
  scaled about its wrist, so a pipeline using it has to carry the scale beside the parameters.
  `test_fit_hand_scales_follows_a_hand_that_changes_size` recovers a known ±7% size swing to
  within 0.5%.
- **Sources.** *Round-trip JSON*: `data/test_output/hand_scale_drift.json`, written by
  `scripts/measure_hand_scale_drift.py`. *Contact JSON*: `data/test_output/hand_size_contact.json`,
  written by `scripts/measure_hand_size_contact.py`. The palm figures come from a one-off probe
  that is not committed. Error is the V1.2 metric: mean millimetre distance of landmarks 1–20,
  wrist-relative, with no registration.

## 2. How much the hand's size moves

| Measure (median over 35 tracks) | Value |
| --- | ---: |
| Fitted per-frame scale, standard deviation | 10.7% |
| Fitted per-frame scale, largest / smallest in the track | 1.48× |
| Palm size (wrist to the four finger MCPs), standard deviation | 10.8% |
| Correlation, palm size against how squarely the palm faces the camera | +0.13 |
| Correlation, fitted scale against the hand's size in the image | +0.09 |

A real palm does not change size. MediaPipe's changes as much as the whole hand does, and neither
the palm's orientation nor its size in the image explains it. V1 found the same for the body's
bone lengths (V1 Findings, finding 6). The size change is detector error.

## 3. Round trip

Pooled over all 35 tracks (round-trip JSON, `pooled`):

| Row | Mean mm | Tips mm | p95 mm |
| --- | ---: | ---: | ---: |
| in-sample shape (current contract) | 5.81 | 2.77 | 16.60 |
| per-frame scale (diagnostic; 1 value per frame) | **3.79** | 2.10 | 10.19 |
| per-frame shape (diagnostic; 10 values per frame) | 3.93 | 2.04 | 10.99 |
| held-out, constant shape | 5.82 | 2.77 | 16.64 |
| held-out, scale from neighbours (sigma 0, nearest) | **4.43** | 2.44 | 12.37 |
| held-out, scale smoothed, sigma 1.5 frames | 4.46 | 2.44 | 12.61 |
| held-out, scale smoothed, sigma 3 frames | 4.59 | 2.46 | 13.07 |
| held-out, scale smoothed, sigma 6 frames | 4.90 | 2.59 | 14.09 |
| held-out, scale smoothed, sigma 12 frames | 5.38 | 2.82 | 15.80 |

The held-out rows fit the shape and the scales on the even frames and score the odd frames, then
the reverse. A scored frame's scale is predicted from the other frames' scales, never fitted to
itself. Its pose is fitted to itself, as in every V1.2 row.

- **The V1.2 per-frame-shape gain is size.** One scale per frame does slightly better than ten
  shape values per frame, on all 35 tracks pooled and on the five pilot clips (7.24 → 4.09 mm,
  against 4.44 mm for per-frame shape). This answers the question V1.2 left open (§4.2: "the model
  ladder has no scale-only row"). Across tracks the gain correlates 0.93 with how much the scale
  moves.
- **Neighbouring frames predict the scale.** Every held-out row with sigma ≤ 3 beats the
  constant shape on 34 or 35 of 35 tracks. The nearest-neighbour row keeps 68% of the in-sample
  gain: (5.82 − 4.43) / (5.82 − 3.79).
- **The size change is fast, not a slow drift.** More smoothing does worse at every step, and the
  nearest neighbour is the best predictor on 20 of 35 tracks. The size changes over a few frames.
  It is coherent from frame to frame, which MediaPipe's video-mode tracking would produce even for
  an error. This corrects the earlier guess that a slowly varying scale with a strong smoothness
  penalty would capture it. A heavily smoothed scale loses most of the gain.

## 4. Depth and contact

`hand_depth.wrist_targets` sets a hand's depth from pixels per metre, which is image size over
MediaPipe's metric size. A wrong metric size therefore moves the hand in depth by the same factor.

**Depth series** (round-trip JSON, `tracks[].depth`). Median over 35 tracks, depth up to the
focal length:

| Hand size used | Depth sd | Frame-to-frame roughness | Rougher than MediaPipe size? |
| --- | ---: | ---: | --- |
| MediaPipe's, per frame (current) | 14.9% | 4.16% | |
| constant (divided by the fitted scale) | 10.6% | 3.81% | smoother on 21/35 tracks |
| smoothed, sigma 3 frames | 12.9% | 3.83% | smoother on 21/35 tracks |

About half of the depth variance (1 − 10.6² / 14.9²) comes from MediaPipe's size changes, not from
the hand's apparent size in the image. Roughness falls on only 21 of 35 tracks, so a constant size
is not clearly more plausible frame to frame.

**Contact** (contact JSON). The face pipeline was run with depth correction on the 20 evaluation
clips and 08925, once with MediaPipe's hands and once with each hand held at constant size (each
frame rescaled about the hand's centroid by its fitted scale relative to the track's geometric
mean):

| Over hand-near-face frames (184) | MediaPipe size | Constant size |
| --- | ---: | ---: |
| wrist depth-corrected | 182 | 182 |
| arm reaches the target | 126 | 124 |
| contact | 52 | 31 |
| median of each clip's median closest hand–head distance | 46.4 mm | 48.6 mm |
| median depth-ratio roughness per clip | 4.12% | 3.64% |

- **Contact on frames where the hand is away from the face drops from 26 to 3.** These frames are
  not near-face: in the image the hand outline stays clear of the face's padded bounding box, so
  3-D contact there contradicts the image. With MediaPipe's size, 21871 alone has 21 such frames.
  At constant size it has 1. This is the one contact error measurable without labels, and the
  constant size makes it much smaller.
- **On near-face frames, contact falls from 52 to 31,** and 18 of the 21 lost frames are in two
  clips: 63669 (12 → 1; median closest distance −2.1 → 38.4 mm) and 57949 (13 → 6; 20.4 →
  34.9 mm). Whether these were real contacts is unknown.
- **08925, the one labelled clip,** touches in all 41 frames. Contact goes from 23 to 20. The
  median closest distance goes from −13.1 mm (the hand inside the head) to −3.7 mm, nearer the
  skin. Fewer frames reach, but those that do are less embedded. This does not decide between the
  two.

## 5. What this means

1. **For round-trip fidelity to MediaPipe,** a per-frame hand scale is the cheapest large gain
   available: one number per frame, 5.8 → 3.8 mm in-sample and 4.4 mm held out. It belongs
   outside the body model, saved per frame and re-applied after augmentation. Baking it into the
   shape would make the hand grow and shrink under every augmentation.
2. **It is not more accurate about the signer.** The size change is detector error (§2). Copying
   it reproduces MediaPipe more closely, so the result depends on what the round trip is for. For
   augmented landmarks meant to look like MediaPipe's, carrying the scale is right. For a
   physically consistent hand, a constant size is right.
3. **For hand-to-face proximity, the size choice is a live variable,** not a detail. MediaPipe's
   per-frame size puts the hand on the face in frames where the image shows it clear of the face
   (26 frames, against 3 at constant size). That favours the constant size for depth. But the
   size choice also moves contact on near-face frames by 40%, and only labels can settle those:
   contact or no contact on the near-face frames of a handful of clips (63669, 57949, 63229,
   08917, 08925), judged from the video. The rule to carry forward: the round trip may keep
   MediaPipe's per-frame size to reproduce its landmarks, but the depth estimate should not
   read hand size from it.

Nothing in the pipeline changed. `run_clip`, `fit_hands` and the V1.2 ladder behave as before.
The optimiser's `scale` option is off by default.

## 6. Limits

- MANO only, a single held-out split (interleaved). The time-halves split was not run, because a
  per-frame scale cannot be extrapolated across a half-track gap.
- No ground truth for size or depth. Depth roughness is a proxy, and contact is labelled only on
  08925.
- The pose prior is not rescaled with the hand. Its weight is 1e-4 mm² per rad², small against
  the landmark terms.
- The constant-size contact run also refits the hands at constant size, so it changes the hand
  geometry as well as the depth.

## Reproduce

```sh
python scripts/measure_hand_scale_drift.py --clip-dir data/test_output \
    --clip-dir data/round_trip_eval --clip-dir <dir with 08925_hands.npz> --model-dir models
python scripts/measure_hand_size_contact.py --clip-dir data/round_trip_eval \
    --clip-dir <dir with 08925_hands.npz and 08925_face.npz> --model-dir models
```

The first takes about 25 minutes, the second about 10.
