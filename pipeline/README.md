# JawTrack pipeline

Python processing for the JawTrack prototype: iPhone video of a HEAD board (headband) and a JAW
board (paraocclusal clutch) → 6-DoF lower-jaw motion in upper-STL coordinates (`motion.json`,
SPEC 5.2) → virtual-articulator parameters, key-position STLs and a report.

## Install

Python 3, OpenCV 5.0 (contrib, `cv2.aruco.ArucoDetector`), numpy, scipy, matplotlib, plus:

    pip install --break-system-packages trimesh

`ffmpeg`/`ffprobe` should be on PATH (rotation tag cross-check, decode fallback).

## Commands for Morten's test

```bash
cd pipeline
# 1. camera calibration (ChArUco clip, same settings as the recording). Print-scale = measured/100 mm.
python3 jawtrack_cli.py calibrate --video charuco.mov --out work/camera.json --print-scale 1.0
#    (no calibration clip: --preset iphone_main_4k  or  --preset iphone_main_1080p --video rec.mov)

# 2. tracking (4K60: --every 2 gives 30 Hz and halves the time)
python3 jawtrack_cli.py track --video rec.mov --camera work/camera.json --out work/raw_tracking.json \
        --every 2 [--reg-image reg.mov|reg.jpg] [--print-scale 1.0]
#    -> work/raw_tracking.json + work/reg_frame.png

# 3. viewer: load reg_frame.png + upper/lower STL, click >= 6 matching points (image <-> STL),
#    export work/registration_landmarks.json

# 4. solve
python3 jawtrack_cli.py solve --raw work/raw_tracking.json --upper upper.stl --lower lower.stl \
        --landmarks work/registration_landmarks.json --out work/motion.json --out-positions work/positions
#    options: --mi-interval T0 T1, --segments segments.json, --condyles kinematic, --smooth 0.12,
#             --refine-registration auto|on|off

# 5. report
python3 jawtrack_cli.py report --motion work/motion.json --outdir work/report
```

One command: `python3 jawtrack_cli.py run --video rec.mov --calib charuco.mov --upper upper.stl
--lower lower.stl --outdir work [--every 2]` — stops after tracking with instructions if
`work/registration_landmarks.json` does not exist yet; run it again after the clicks.

Outputs: `motion.json` (viewer contract), `positions/lower_{MI,max_opening,max_protrusion,
edge_to_edge,max_laterotrusion_right,max_laterotrusion_left}.stl` (lower STL moved into each
position, in upper-STL coordinates — import next to the upper STL in any CAD), `positions/
articulator_values.txt` (Danish/English), `report/*.png` + `report.md`.

## What each stage does

| stage | module | method |
|---|---|---|
| calibrate | `calib.py` | ~300 sampled frames, ChArUco detection, blur filter (Laplacian variance, lowest 30 % dropped), views spread over 3x3 image cells x 3 sizes, `calibrateCamera` (k3 fixed), outlier views (> 3x median error) dropped and re-run. Preset: fx = fy = long side x 26/36, centred principal point, zero distortion, `source: "preset"`. |
| video IO | `io.py` | `cv2.VideoCapture` with **auto-rotation disabled**; the rotation tag (`CAP_PROP_ORIENTATION_META`, cross-checked with ffprobe's display matrix) is applied explicitly so frames are always in display orientation. If OpenCV cannot decode (e.g. HEVC unsupported), frames are piped through ffmpeg. Camera intrinsics are scaled (same aspect) or transposed (portrait/landscape) with a warning if the video size differs from the calibration. |
| detect | `detect.py` | `ArucoDetector` DICT_4X4_50 (adaptive threshold windows 5-35 px, SUBPIX). Full frame downscaled to long side <= 1920, afterwards only a region of interest around the last detections, scaled so markers are ~48 px; corners mapped back and refined with `cornerSubPix` on the **full-resolution** frame (window < half the 3 mm gap). |
| pose | `pose.py`, `track.py` | Both IPPE solutions per board and frame; **planar ambiguity resolved by a Viterbi pass** over the whole clip (reprojection cost + rotational smoothness), for the JAW board plus a prior on its orientation relative to the HEAD board learned from unambiguous frames; then `solvePnPRefineLM`. Per frame: pose, n_markers, reproj_px, ambiguity ratio. |
| reg frame | `track.py` | sharpest frame (ROI Laplacian variance x marker count) with both boards inside the first MI hold (detected from board motion); or `--reg-image`: a still (own head pose, intrinsics adapted, warning if the size differs) or a **short registration video** (`.mov/.mp4`: sharpest frame with both boards, temporally disambiguated poses). |
| registration refinement | `registration_refine.py` | optional image-based search along the weak click directions (see below) |
| registration | `registration.py` | SQPnP of STL landmarks to clicked pixels, then LM of X = T_H_S over all points; upper points through T_C_H, lower points through T_C_J inv(M0) (correct even if the reg frame is not exactly MI). RMS px reported, warning > 4 px. |
| motion | `motion.py`, `solve.py` | M(t) = inv(T_C_H) T_C_J; frames with reprojection > max(3x median, 1.5 px) rejected; MI = first stable run (incisal displacement below max(0.5 mm, 4x noise)) in the first 40 % of the clip, M0 = robust SE(3) mean (chordal rotation mean + median translation with MAD rejection); MI and X re-estimated twice; T_S = inv(X) M inv(M0) X; Hampel outlier filter; gaps <= 0.25 s slerp-interpolated (`interpolated: true`), longer gaps `valid: false`; Savitzky-Golay (0.12 s, order 2) on the rotation vector and the incisal-point position. Check: warns if the incisal point rises > 1 mm above "MI" (wrong MI interval). |
| landmarks | `landmarks.py` | incisal point + occlusal plane from the landmarks file, otherwise automatic (PCA normal towards the upper arch, anterior = narrow end of the U, most anterior-superior midline vertex; plane through the highest posterior cusps L/R). Condyles: Bonwill 105 mm triangle with Balkwill 22 deg built in the occlusal-plane frame; kinematic hinge axis fitted on opening frames with 1-10 mm incisal displacement (axis direction = weighted mean rotation axis; point = least squares on the mid-sagittal plane; fit RMS = incisal error of the pure-rotation model); `--condyles kinematic` puts condyles on it at +-52.5 mm. |
| segments | `segments.py` | excursions from MI (> max(1.5 mm, 2x MI threshold)); per excursion, costs from the share of opening / anterior / right / left displacement at its peak (chewing: short cycles); labels by dynamic programming constrained to the protocol order open_close → protrusion → laterotrusion_right → laterotrusion_left → chewing; MI segments for rests >= 0.4 s. `--segments` overrides. |
| params | `params.py` | see docstring; angles relative to the **occlusal plane**. SCI = mean sagittal direction of each condyle point's protrusion path for 0.5-5 mm condylar advance; Bennett = horizontal-plane angle of the **orbiting** condyle (left value from right laterotrusion); incisal guidance = incisal path for 0.3-3.5 mm advance; edge-to-edge = protrusion frame where the lower incisal point is closest to the upper incisal edge (if within 2 mm). |
| quality | `solve.py` | tracked fraction, registration RMS, `jitter_mm_rms_in_MI` (incisal-point RMS scatter in MI holds >= 1 s, filtered; unfiltered also given — the honest noise floor), MI start-vs-end repeatability (headband/clutch slip shows up here), marker reprojection median/p95, X and M0. |

Board frame convention actually observed (OpenCV 5.0 GridBoard, unit test `test_gridboard_convention`
and cross-check against `capture_kit/board_geometry.json`): origin at the top-left outer corner of
the top-left marker, **x right, y down, z = x × y into the board** (away from the camera), corners
TL, TR, BR, BL. `T` in all JSON files is row-major (`T[3]`, `T[7]`, `T[11]` = translation).

## Verification (all numbers measured by running the code)

`python3 -m pytest -q tests/` — 12 tests (11 unit + 1 end-to-end), all pass (~60 s).

Unit tests (`tests/test_math.py`): T_S formula vs. a known jaw motion (20 random scenes, 1e-9),
row-major serialisation, GridBoard orientation + kit geometry, robust averaging with outliers,
**rotation tag** (H.264 stored rotated with a display-rotation tag, both OpenCV and ffmpeg
backends), ChArUco calibration on 25 rendered views (fx error 0.15 %, RMS 0.16 px), Viterbi flip
resolution, parameter definitions on analytic motions (SCI/Bennett/incisal guidance exact), hinge
axis fit exact.

End-to-end synthetic (`tests/test_synthetic_e2e.py`, also `jawtrack_cli.py selftest`): the real
cv2.aruco board images are warped (2x supersampled homography K[r1 r2 t]) into a 26 mm-equivalent
camera at ~35 cm over a textured background, with head motion (+-2.5 deg, +-4 mm), Gaussian blur
(0-0.8 px), noise (sigma 2.5), H.264 CRF 20, stored portrait + rotation tag; jaw motion follows the
protocol (2x open/close to ~45 mm, 2x protrusion, 2x laterotrusion R/L with 35 deg SCI and 10 deg
Bennett, 3 s chewing, MI holds); exact landmarks (8 points, 2 on the lower jaw).

T_S(t) vs. ground truth, all frames (error of the incisal point in mm, rotation in deg):

| synthetic run | incisal median | incisal p95 | rot median | rot p95 | jitter in MI | registration RMS |
|---|---|---|---|---|---|---|
| 1920x1080, 30 fps (pytest default, 663 frames) | 0.33 | 0.73 | 0.30 | 0.75 | 0.35 | 0.49 px |
| 1920x1080, 60 fps (1326 frames) | 0.32 | 0.69 | 0.31 | 0.81 | 0.40 | 0.30 px |
| 1920x1080, 60 fps, `--every 2` | 0.33 | 0.68 | 0.32 | 0.86 | 0.36 | 0.31 px |
| 3840x2160, 30 fps (663 frames) | **0.12** | **0.25** | 0.12 | 0.29 | 0.14 | 0.36 px |

Segmentation: all 6 labels found with boundaries within ~0.3 s of the truth in every run.
Parameters, measured vs. same definitions applied to the ground-truth motion:

| parameter | truth | 1080p60 | 4K30 |
|---|---|---|---|
| max opening (mm) | 44.95 | 45.61 | 45.08 |
| max protrusion (mm) | 6.98 | 7.17 | 7.10 |
| max laterotrusion R / L (mm) | 7.71 / 7.71 | 7.76 / 7.82 | 7.77 / 7.74 |
| SCI L / R (deg, to occlusal plane) | 37.6 / 37.6 | 36.1 / 38.2 | 37.7 / 37.2 |
| Bennett L / R (deg) | 11.2 / 11.2 | 7.1 / 15.9 | 12.9 / 10.3 |
| incisal guidance (deg) | 26.3 | 22.8 | 25.5 |
| kinematic hinge axis vs true condyle axis | - | 1.0 deg, ~2 mm, fit RMS 0.58 mm | - |

Error budget (1080p30 run): with ground-truth HEAD pose and estimated JAW pose the incisal error is
0.52 mm median, with estimated HEAD and true JAW only 0.17 mm — **the small 2x2 JAW board dominates**,
almost entirely through its depth (board-normal) estimate (std 0.22 mm, plus a ~0.5 mm bias that is
partly common to both boards). The synthetic corner bias is small (texture mapping verified to
0.01 mm). A larger jaw board, or 4K, is the most effective improvement.

### Synth agent's dataset (`testdata/synth_quick`, 10 s 1080p60, lens distortion, blur, occlusions)

Full chain with **our own ChArUco calibration** of `charuco_calib.mp4` (40 views, RMS 0.07 px,
fx 1390.9 vs true 1391.7 = 0.06 %, distortion recovered), then track + solve + `validate.py`:

| landmarks | incisal median | incisal p95 | rot median | max opening (true 45.0) | SCI L / R error |
|---|---|---|---|---|---|
| exact (6 upper points) | 0.56 mm | 1.23 mm | 0.39 deg | 45.3 mm | 4.3 / 2.6 deg (PASS) |
| simulated clicks, sigma 1.5 px | 0.63 mm | **4.5 mm** | 0.37 deg | 49.2 mm | 2.2 / 0.8 deg (PASS) |

With noisy clicks the error during wide opening is **registration-dominated**: 6 upper-arch points
constrain the depth of X poorly, and an X translation error acts as a lever (error ≈ jaw rotation x
X error). `solve` now estimates this itself: the LM covariance of X is propagated to the incisal
point at max opening (`quality.registration_incisal_uncertainty_at_max_opening_mm`: 2.95 mm here,
0.6 mm for the exact clicks) and a note asks for more landmarks when it exceeds 1 mm.
Incisal guidance differs from the synth agent's value by definition (theirs: chord at 5 mm
protrusion; ours: path over the first 0.3-3.5 mm); recomputed with their definition the error is
0.3 deg (exact clicks).

### Registration refinement (`solve --refine-registration auto|on|off`, default auto)

Click-based PnP is ill-conditioned in essentially ONE direction (a depth/tilt combination): the
click covariance has one eigen-direction with sd ~5-6 mm-equivalent, all others < 0.4 mm. With
1.5 px click noise that direction produced an 8 mm X error on synth_quick. Steps added:

1. head/jaw pose of the registration frame averaged over neighbouring frames (+-0.2 s, only frames
   within the pose-noise band) instead of a single frame;
2. if the registration image is found (next to the landmarks file or raw_tracking.json), X is
   searched on a grid of +-3 sd along the two weakest click-covariance directions (13x13 coarse,
   7x7 fine). Each candidate renders upper+lower STL (small painter's rasteriser, Lambert shading)
   and scores the rendered crown edges (|height| < 8 mm from the occlusal plane, boards masked)
   by Gaussian proximity to Canny edges of the image. A small prior keeps it near the clicks;
3. sanity checks (otherwise the click X is kept and a warning logged): edge-score gain >= 0.005,
   click RMS may not more than double, optimum not on the search border. Uncertainty before/after
   is propagated to the incisal point at max opening (`registration.refinement` in motion.json).
   On an image without visible teeth (our own synthetic set) it correctly declines.
   A local least-squares edge alignment (silhouette/crease/depth edges) was tried first and
   dropped: its cost was flat along the weak direction and its minimum was not at the truth.

Measured with NOISY landmarks (sigma 1.5 px, 6 points), own ChArUco calibration, `validate.py`:

| dataset | refinement | incisal median | incisal p95 | open_close p95 | max opening (true 45.0) |
|---|---|---|---|---|---|
| synth_quick (10 s) | off | 0.63 mm | 4.52 mm | 4.67 mm | 49.2 mm |
| synth_quick | on | 0.63 mm | **1.45 mm** | 1.68 mm | 44.4 mm |
| synth_01 (31 s, full protocol) | off | 0.53 mm | 1.28 mm | 3.64 mm | 43.8 mm |
| synth_01 | on | 0.50 mm | **1.12 mm** | 2.17 mm | 46.2 mm |

synth_01 other parameters (off → on): protrusion 8.81 → 8.84 (true 9.0), laterotrusion R/L
9.79/9.94 → 10.27/10.32 (10.0), SCI L/R 33.1/37.3 → 33.0/37.3 (35/38). Bennett (recomputed by
validate.py with its definition) L/R 7.1/5.3 → 10.2/7.6 (10/8). Our own Bennett definition
(mean path over 0.5-5 mm) gives 16-20 deg for L on this dataset. That is a definition/early-path
effect still under investigation. synth_01 has a few tracking outlier frames in open_close
(max 8 mm / 10 deg), probably occlusions. Runtime of the refinement: ~35 s.

**Honest caveat:** the synthetic teeth are flat-shaded renders with crisp crown edges and a clean
retractor. Real enamel, gingiva, saliva highlights and lips give weaker, noisier edges. The checks
make the step fall back to the clicks, but on real photos expect a smaller gain.

### Recommendations for the registration capture

* Record the registration as a short **4K video** (same camera settings as the recording, not
  photo mode), cheek **retractor**, teeth in MI, both plates visible, phone on the tripod.
* **Diffuse light** (no flash hot spots), dry teeth (air-blow), focus on the teeth.
* Click **8-10 sharp, widely spread landmarks on BOTH jaws**: incisal edges 11/21, canine tips
  13/23 and 33/43, mesiobuccal cusps 16/26 and 36/46. Spread in depth (molars) is what constrains
  the weak direction. Check `quality.registration_incisal_uncertainty_at_max_opening_mm` (< 1 mm).

## Performance (2 CPUs)

Measured (2 cores, other agents running in parallel, so conservative):

* 1080p: 14 fps processing every frame, 25 fps with `--every 2` (synth_quick: 53 fps) →
  a 60 s 1080p60 clip takes ~1-4 min.
* 4K (H.264): 10.2 fps processing every frame (98 ms/frame: decode ~30 ms, ROI detection ~40 ms,
  full-res corner refinement, PnP). HEVC 4K decode measured 29 ms/frame (38 ms/frame including
  skipped frames with `--every 2`). **Estimate for a 60 s 4K60 clip: ~6 min (every frame),
  ~4 min with `--every 2`** (30 Hz is enough for jaw movement).
* calibrate: ~20 s for an 8 s clip; solve: < 1 s; report: ~3 s; full pytest: 66 s.
* Detector tuning that mattered: `minMarkerPerimeterRate` 0.01 → 0.03 made ROI detection 7x faster.

## Assumptions

* Intrinsics from ChArUco calibration at the same resolution/orientation/zoom/focus as the
  recording (lens switching, stabilisation and auto-macro off; AF/AE locked). The preset is a
  fallback with a few % scale error; this mostly affects absolute distances, not angles.
* Upper and lower STLs exported in MI in the same frame; registration landmarks clicked on a frame
  in MI (lower-jaw points are nevertheless handled through the jaw pose).
* Head board rigid with the maxilla, jaw board rigid with the mandible.
* The first protocol segment is an MI hold (or give `--mi-interval`).

## Known limitations

* **Planar pose ambiguity** of small boards: handled by IPPE candidates + temporal Viterbi + jaw
  orientation prior, audited per frame (`ambiguity` ratio, count of corrected frames in the log).
  A frame sequence that is ambiguous from start to end could still choose the wrong branch.
* **Depth precision of the 33 mm jaw board** is the dominant error (see above); jitter of a few
  tenths of a mm at the incisal point is the floor at 1080p. Not verified on real footage yet.
* **Rolling shutter** (iPhone readout ~10-20 ms): fast chewing moves the jaw board during readout;
  not modelled. Slow movements are fine; chewing paths are approximate.
* **Headband/skin slip** and **clutch flex** are invisible to the cameras and go straight into the
  jaw motion; the MI start-vs-end repeatability value is the practical check.
* **Registration is the largest error source with real clicks** (see synth_quick numbers): click
  8-10 sharp, well-spread points on both arches (incisal edges, canine tips, molar cusps L/R).
* **Registration error** acts as a lever: ~1 px RMS ≈ 0.3 mm at the teeth at 1080p, but rotation
  errors of X move the condyle points ~100 mm away → condyle-derived angles (SCI, Bennett) are the
  least reliable numbers (synthetic 1080p: SCI within 1.5 deg, Bennett within 5 deg; 4K: 0.4 / 1.7 deg).
* **Average-value condyles** (Bonwill/Balkwill) are not the patient's condyles; SCI/Bennett values
  depend on where the condyle points are. The kinematic hinge axis is an alternative but needs a
  clean slow opening and is sensitive to noise (synthetic: ~2-3 mm from the true axis; synth_quick: 3 deg, 7-11 mm).
* Angles are relative to the occlusal plane, not Frankfort.
* Edge-to-edge detection uses the upper STL geometry (lowest upper vertex near the midline); it is
  skipped when no protrusion frame comes within 2 mm.
* The automatic incisal point / occlusal plane fallback is heuristic — prefer clicking them.
* Video timestamps come from the container (CAP_PROP_POS_MSEC); variable frame rate is respected,
  dropped frames are not detected.

## Next steps

1. Run on Morten's real recording; compare MI jitter / MI repeatability with the synthetic floor.
2. Bigger or non-planar jaw target (e.g. two plates at an angle) to fix depth precision and the
   ambiguity; or a 3x2 jaw board.
3. Bundle-adjust X using several registration frames; refine registration by ICP of the STL
   against a cheek-retractor photo silhouette.
4. Rolling-shutter-aware pose (per-row timing) for chewing.
5. Frankfort reference (orbitale/porion clicks on a profile photo) for articulator-standard angles.
