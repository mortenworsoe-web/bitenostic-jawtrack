# JawTrack prototype — shared spec (all agents MUST follow this)

Goal: overnight prototype for Bitenostic/Bitedesign. A dentist (Morten) records an iPhone video of
his own jaw movements wearing two small printed marker plates, plus intraoral-scan STLs of his
upper and lower jaw exported IN OCCLUSION (maximal intercuspation, MI, same coordinate frame).
The system reconstructs the 6-DoF motion of the lower jaw relative to the upper jaw, animates the
STLs, shows contacts, and derives virtual-articulator parameters. Simple, cheap, convincing.

Root: repo root
- SPEC.md            this file (read-only for agents; ask the orchestrator to change it)
- capture_kit/       printable markers, calibration board, 3D-printable holders, capture protocol (Agent: kit)
- pipeline/          Python processing (Agent: pipeline)
- testdata/          synthetic dataset with ground truth + validation harness (Agent: synth)
- viewer/            single-file HTML/JS 3D viewer (Agent: viewer)
- docs/              user-facing docs (Danish), written by the kit agent / orchestrator

Environment: Linux container, Python 3, OpenCV 5.0 (contrib, cv2.aruco.ArucoDetector API),
numpy 2.5, scipy 1.18, matplotlib, reportlab installed. `pip install --break-system-packages X`
works for PyPI packages (trimesh, numpy-stl, etc.). ffmpeg available. Node 22 available.
Only 2 CPU cores / 7 GB RAM: keep renders modest (1920x1080, <= 30 s).
No internet except package registries and GitHub. Do not touch other agents' folders.

## 1. Physical setup (what the patient wears)

Two rigid planar ArUco GridBoards, dictionary DICT_4X4_50:

| Board | Attached to | Grid (cols x rows) | Marker IDs (row-major, top-left first) | Marker side | Gap |
|-------|-------------|--------------------|----------------------------------------|-------------|-----|
| HEAD  | headband on forehead (maxilla reference) | 3 x 2 | 0,1,2,3,4,5     | 15.0 mm | 3.0 mm |
| JAW   | paraocclusal clutch on lower anterior teeth (labial, cervical half 33–43), stem out between the lips | 2 x 2 | 10,11,12,13 | 15.0 mm | 3.0 mm |

Board coordinate frame = OpenCV GridBoard convention (origin at top-left corner of top-left
marker, x right, y down, z out of the board — use cv2.aruco.GridBoard and its objPoints, and
state the convention you actually observed in your README). Units: millimetres everywhere.

Camera calibration board: ChArUco, DICT_5X5_100, 7 x 5 squares, square 25.0 mm, marker 18.0 mm,
printed on A4. Every printed sheet carries a 100.0 mm scale bar so the user can verify print scale;
the pipeline accepts a `--print-scale` factor (measured_mm/100) applied to all printed geometry.

## 2. Capture protocol (summary; full Danish version in docs/)

iPhone rear main camera (1x), 4K or 1080p, 60 fps, HDR off, enhanced stabilization/action mode
off, camera lens switching locked (no auto-macro), AE/AF lock, phone on tripod 30–40 cm in front,
slightly below, face + both plates in view. Recording sequence (each starting/ending in MI):
1. MI 3 s (teeth firmly together)  2. open/close x3 (slow, max)  3. protrusion x3 (edge-to-edge and
beyond)  4. right laterotrusion x3  5. left laterotrusion x3  6. free chewing 5 s  7. MI 3 s.
Plus: a separate 20–30 s calibration clip of the ChArUco board (same settings, AF lock at ~35 cm)
and one "registration still": a frame/photo with cheek retractor, teeth in MI, both plates visible.

## 3. Coordinate frames and the core math

- S  = STL frame (upper and lower STLs share it in MI). Units mm.
- C(t) = camera frame at frame t. H(t) = HEAD board frame, J(t) = JAW board frame.
- Pipeline estimates T_C_H(t) and T_C_J(t) (4x4, maps board coords -> camera coords).
- Relative jaw pose in head frame: M(t) = inv(T_C_H(t)) @ T_C_J(t).
- MI reference M0 = robust average of M(t) over the MI interval (first MI hold).
- Registration X = T_H_S (maps STL coords -> HEAD board coords), obtained once (sec. 4).
- Lower-jaw motion in STL coordinates (THE deliverable):
      T_S(t) = inv(X) @ M(t) @ inv(M0) @ X
  T_S(t) maps a lower-STL vertex (in MI) to its position at time t, expressed in the upper STL
  frame. T_S = identity in MI. The upper STL never moves.
- Note: rotations of X do matter (conjugation), translation of X matters a lot (lever arm). This is
  why registration cannot be skipped.

## 4. Registration (STL <-> markers)

Landmark-based, semi-manual (prototype): on the registration still (teeth in MI, both plates
visible) the user clicks N >= 4 (recommend 6) clearly identifiable points in the image (e.g. incisal
edge of 11, 21, cusp tips 13, 23, mesiobuccal cusps 16, 26, or lower 33/43 cusp tips) and clicks the
same points on the STL in the viewer. The viewer exports `registration_landmarks.json`:

```json
{ "version": 1,
  "image": {"file": "reg_frame.png", "frame_index": 123, "width": 3840, "height": 2160},
  "points": [ {"name": "11_incisal", "img_px": [u, v], "stl_mm": [x, y, z], "jaw": "upper"} ],
  "incisal_point_mm": [x, y, z] | null,
  "occlusal_plane_mm": [[x,y,z],[x,y,z],[x,y,z]] | null
}
```
`incisal_point_mm` = lower incisal point (between 31/41 incisal edges), `occlusal_plane_mm` =
3 points defining the occlusal plane (lower incisal point + mesiobuccal cusps 36, 46). Both optional
— pipeline falls back to automatic estimates (most anterior-superior lower vertex; PCA plane of
the occlusal-most lower vertices) and states that it did.

Pipeline solves PnP (camera intrinsics known) for T_C_S at that frame, combined with T_C_H at the
same frame: X = inv(T_C_H) @ T_C_S. Report reprojection RMS (px) and warn if > 4 px.

## 5. File formats

### 5.1 `raw_tracking.json` (pipeline stage 1 output)
```json
{ "version": 1, "video": "file.mov", "fps": 59.94, "width": 3840, "height": 2160,
  "camera": {"K": [[...3x3...]], "dist": [k1,k2,p1,p2,k3], "source": "charuco|preset|selfcal"},
  "frames": [ {"i": 0, "t": 0.0, "head": {"T": [16 floats, row-major], "n_markers": 6, "reproj_px": 0.4} | null,
                                 "jaw":  {"T": [...], "n_markers": 4, "reproj_px": 0.5} | null } ] }
```

### 5.2 `motion.json` (final output; consumed by the viewer) — the contract between pipeline and viewer
```json
{ "version": 1, "units": "mm", "fps": 59.94,
  "frame_of_reference": "upper STL coordinates; T maps lower STL (as exported in MI) to its pose at time t",
  "source": {"video": "file.mov", "video_offset_s": 0.0},
  "frames": [ {"t": 0.0, "T": [16 floats row-major 4x4], "valid": true, "quality": 0.0-1.0} ],
  "segments": [ {"label": "MI|open_close|protrusion|laterotrusion_right|laterotrusion_left|chewing", "t0": 0.0, "t1": 3.0} ],
  "landmarks_mm": { "incisal_point": [x,y,z], "condyle_left": [x,y,z], "condyle_right": [x,y,z],
                    "occlusal_plane": {"point": [x,y,z], "normal": [x,y,z]},
                    "condyle_method": "average_bonwill|kinematic_axis" },
  "parameters": {
     "max_opening_mm": 0, "max_protrusion_mm": 0, "max_laterotrusion_right_mm": 0, "max_laterotrusion_left_mm": 0,
     "sagittal_condylar_inclination_left_deg": 0, "sagittal_condylar_inclination_right_deg": 0,
     "bennett_angle_left_deg": 0, "bennett_angle_right_deg": 0,
     "incisal_guidance_deg": 0, "kinematic_hinge_axis": {"point": [x,y,z], "direction": [x,y,z], "fit_rms_mm": 0} | null,
     "notes": ["human-readable caveats"] },
  "quality": { "tracked_fraction": 0.0, "registration_rms_px": 0.0, "jitter_mm_rms_in_MI": 0.0 } }
```
Angles are measured relative to the occlusal plane (state this in notes; Frankfort is not available).
`T` is row-major: T[0..3] is the first row. Viewer must convert to three.js (column-major) correctly.

### 5.3 Upper/lower STL: binary or ASCII STL, mm, exported in occlusion. File names are free; the
viewer and pipeline take them as arguments / drag-and-drop.

## 6. Anatomical defaults (for synthetic data and fallbacks)

Upper STL frame for synthetic data: origin near the upper incisal point,
x = patient's LEFT, y = SUPERIOR, z = ANTERIOR (right-handed). Real scanner
STLs will have arbitrary frames; never assume axes for real data — derive from landmarks/PCA.
Average-value articulator (Bonwill): condyles ~105 mm apart, each ~105 mm from the lower incisal
point; Balkwill angle ~22°; sagittal condylar inclination ~30–40° to occlusal plane (≈ 25–30° to
Frankfort ± ); Bennett angle ~7–15°; max opening ~45–55 mm; protrusion ~8–10 mm; laterotrusion ~8–12 mm.

## 7. Quality bar

- Python: CLI with argparse, `--help` works, runs end-to-end with one command, clear errors.
- Validation numbers must come from actually running code, never estimated.
- Each agent writes a README.md in its folder: what it built, how to run, what is verified, known
  limitations. Keep it honest.
