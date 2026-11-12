# JawTrack capture kit

Printable markers, calibration board, 3D-printable holders and the geometry contract for the
JawTrack prototype. Danish step-by-step capture guide: `../docs/optageguide_DA.md`.

## Files

| File | What |
|---|---|
| `markers_head_jaw.pdf` | A4, 100 % scale. 3x HEAD (3x2, IDs 0-5) + 3x JAW (2x2, IDs 10-13), DICT_4X4_50, marker 15.0 mm, gap 3.0 mm. Dashed cut line = board + 4 mm white quiet zone (HEAD sticker 59x41 mm, JAW 41x41 mm). Horizontal + vertical 100.0 mm scale bars, Danish print note. |
| `charuco_calibration_A4.pdf` | ChArUco 7x5, square 25.0 mm, marker 18.0 mm, DICT_5X5_100 (default, non-legacy pattern), 175x125 mm, two 100.0 mm scale bars. |
| `board_geometry.json` | All three boards: dictionary, ids, sizes, and every marker corner (and ChArUco chessboard corner) in board coordinates exactly as OpenCV 5.0.0 reports them. |
| `jaw_clutch_paddle.stl` | Generic paraocclusal mandibular clutch (bar + stem + JAW plate with 41.4x41.4x0.3 mm pocket). |
| `head_marker_plate.stl` | Forehead plate, 59.4x41.4x0.3 mm pocket, 12° downward tilt, wings with 2x2 headband slots. |
| `make_markers.py` | Generates the two PDFs + `board_geometry.json`. |
| `verify_markers.py` | Rasterises the PDFs (pdftoppm and PyMuPDF, 600 dpi), re-detects, measures. |
| `make_holders.py` | Generates the STLs, watertightness checks, mass, previews (`verify/*_preview.png`). |
| `make_custom_clutch.py` | Stretch goal, implemented: custom-fit clutch from the patient's lower STL. `--selftest` runs on a procedural arch. |
| `verify/` | `verification.json`, `stl_checks.json`, `custom_clutch_selftest.json`, previews, self-test STLs. |

Regenerate everything: `python3 make_markers.py && python3 verify_markers.py && python3 make_holders.py && python3 make_custom_clutch.py --selftest`
(needs `pip install --break-system-packages trimesh manifold3d pymupdf scikit-image`).

## Board frame convention (observed, OpenCV 5.0.0)

`cv2.aruco.GridBoard(...).getObjPoints()`: origin at the outer top-left corner of the top-left
marker (as printed, upright), **x right, y DOWN the sheet**, z = 0 on the board. Hence +z = x × y
points **into** the board (away from the viewer); the board normal toward the camera is −z.
Corners per marker: TL, TR, BR, BL. Example: HEAD id 3 = (0,18),(15,18),(15,33),(0,33).
(Older OpenCV 4.x GridBoards used y up / origin bottom-left — do not mix.) All markers are drawn by
converting the `generateImage()` bitmap (28 px/mm, every module an integer pixel count, asserted
binary) losslessly to vector rectangles, so the print is bit-identical to the OpenCV board.

## Verification (real numbers, `verify/verification.json`, 600 dpi rasters)

| Check | pdftoppm | PyMuPDF |
|---|---|---|
| HEAD/JAW markers detected | 30/30, ids exactly {0-5}x3 + {10-13}x3 | 30/30, same |
| Marker side mean (min–max), mm | 14.990 (14.981–15.024) | 14.993 (14.987–15.002) |
| Board diagonal vs objPoints, max abs err | 0.009 mm | 0.012 mm |
| Marker sheet scale bar H / V, mm | 99.991 / 99.991 | 100.000 / 100.000 |
| ChArUco corners / markers | 24/24, 17/17 | 24/24, 17/17 |
| ChArUco square mean (min–max), mm | 25.003 (24.977–25.020) | 25.000 (24.994–25.005) |
| ChArUco affine scale x / y, RMS | 0.99987 / 0.99975, 0.016 mm | 1.00000 / 0.99998, 0.003 mm |
| ChArUco scale bar H / V, mm | 99.991 / 100.012 | 100.000 / 100.000 |

Residuals (≤0.02 mm) are at the raster pixel level (1 px = 0.042 mm). This verifies the PDF; the
**printer** is the real error source — the user must measure the bar on paper.

## Holders (`verify/stl_checks.json`)

| Part | Watertight | Bodies | Volume | Est. mass (resin 1.18 g/cm³) | BBox mm |
|---|---|---|---|---|---|
| `jaw_clutch_paddle.stl` | yes | 1 | 4.71 cm³ | **5.6 g** | 45.4 x 61.8 x 48.3 |
| `head_marker_plate.stl` | yes | 1 | 10.53 cm³ | 12.4 g | 97.7 x 45.4 x 18.1 |

Model frame: x = patient left/right, y = up, z = anterior. **Clutch:** labial bar inner radius
22.3 mm (arch radius 22 + 0.3 mm luting gap; `--arch-radius` to change), 1.5 mm thick, 6 mm tall,
104° arc (chord 35 mm ≈ 33–43), 3 horizontal + 6 vertical retention grooves (0.6 x 0.4 mm) on the
tooth side. Stem 6 (w) x 4 (h) mm from the lower half of the bar: 12 mm straight forward, then 30 mm
at 40° down-forward (42 mm total). Plate 45.4 x 45.4 x 1.8 mm, face normal (0, −0.26, 0.97) = 15°
down, centre ≈ 34 mm below / 56 mm anterior of the bar → hangs in front of the chin and stays below
the camera's line of sight to the teeth for a camera slightly below the mouth. **Head plate:** flat
2 mm panel (sticker must stay planar), rib frame whose back plane tilts the face 12° down
(stand-off 4 mm top / 13.7 mm bottom), wings swept 25° back with two 26 x 2.6 mm slots each.

**Pocket size deviation:** pockets are 41.4 x 41.4 / 59.4 x 41.4 mm, not 33 x 33 / 51 x 33 — the
sticker includes a 4 mm white quiet zone because ArUco detection needs white around the outer black
border, and grey/clear resin does not reliably provide it.

### Printing
- **Resin (Formlabs-type), preferred for the clutch.** Rigid resin; for intraoral use preferably a
  biocompatible dental resin (surgical-guide / splint class) — standard resins are not certified for
  intraoral contact (short self-test is the clinician's call). Wash + full post-cure.
  Clutch orientation: plate face (pocket) **up**, part tilted ~30–45°, supports only on the plate back,
  stem underside and bar outer face; keep pocket and bar inner (grooved) side support-free.
  Head plate: panel face up, supports on rib frame/wings.
- **FDM (PLA/PETG), fine for the head plate** at 0.2 mm, face down on the bed is OK if the bed is flat
  and smooth (pocket depth 0.3 mm = 1–2 layers). The clutch is better in resin (thin bar).
- Glue paper stickers **flat** in the pockets (spray adhesive / full-surface double-sided tape),
  matte laminate or none; glossy lamination causes glare and lost detections.

## Custom-fit clutch (stretch goal — implemented, tested only on a procedural arch)

`python3 make_custom_clutch.py --lower lower.stl --p33 x,y,z --p43 x,y,z --incisal x,y,z [--upper upper.stl] --out custom_clutch.stl`

p33/p43 = labial gingival-margin points mid-33/mid-43, incisal = lower incisal point (STL coords,
pick in any viewer). It builds a local frame (x = 43→33, y = toward incisal, z = labial), voxelises
(0.2 mm) a 1.5 mm labial shell with 0.1 mm relief from −0.3 to min(4.5 mm, half crown height) above
the margin, only on labial-facing surface, marching-cubes it, unions the same stem + JAW plate, and
writes the result **in the lower-STL frame** (so the clutch geometry relative to the teeth is known —
a future route to automatic registration). With `--upper` it reports min clearance to the upper teeth
in MI and warns below 0.5 mm.
Self-test (`verify/custom_clutch_selftest.json`): watertight, 1 body, 4.66 cm³ / 5.5 g, min distance
to lower 0.015 mm (smoothing eats most of the 0.1 mm relief at the gingival edge — increase
`--relief` to 0.15–0.2 for real scans), clearance to upper in MI 0.88 mm, runtime ~15 s.

## Limitations
- Generic clutch is a one-size arch (r = 22 mm); it **must** be relined/luted with bis-acryl to fit.
  Bar top must sit ≥ overbite + 1.5 mm below the lower incisal edges; deep bites need the bar trimmed.
- Head plate on forehead skin: frowning/eyebrow movement moves it by mm (soft-tissue artefact).
- Previews are matplotlib renders (depth-sorting artefacts); shape sanity only. No physical fit test.
- Custom clutch: not tested on a real scan; landmarks are manual; scanner meshes with holes in the
  labial region give holes in the shell.
- iOS setting names in the guide were checked against Apple's iOS 26 iPhone User Guide (Record Video,
  HDR Video, Enhanced Stabilization, Lock Camera, Lock White Balance, Auto FPS); "Macro Control" and
  "Preserve Settings" are from memory, and Danish UI names are unverified translations.
