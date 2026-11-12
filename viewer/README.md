# JawTrack Viewer (`viewer/index.html`)

A single self-contained HTML file. It shows the upper and lower STL, animates the lower jaw from
`motion.json` (SPEC 5.2), and draws occlusal contacts, movement traces and Posselt-style plots.
It also provides the registration tool that writes `registration_landmarks.json` (SPEC 4).
The UI is in Danish.

**Privacy:** every file is read and processed locally in the browser (FileReader, a Web Worker,
WebGL). Nothing is uploaded. The UI says so ("Data forlader ikke din computer"). The only
network traffic is the one-time download of three.js and three-mesh-bvh from jsdelivr.

## How to open

* **Double-click `index.html`** to open it in Chrome or Safari. It needs internet the first time
  so it can load three.js 0.186.0 and three-mesh-bvh 0.9.16 from `cdn.jsdelivr.net`. The browser
  then caches them.
* Or host the file on any static web server (`python3 -m http.server` in `viewer/`). This also
  works on an iPad or iPhone over the local network.
* Drag files into the window, or click "Vælg filer…". The viewer guesses each file's role from its
  name and contents. You can change any role in the file list.

| Role | Recognised by |
|---|---|
| Overkæbe | `.stl` with upper / over / maxil / `OK` in the name |
| Underkæbe | `.stl` with lower / under / mand / `UK` in the name |
| Unnamed STL | goes to the first free slot (upper first) and is marked "rolle gættet" |
| Bevægelse | `.json` whose `frames` contain `T` (`raw_tracking.json` is ignored) |
| Landmarks | `.json` with `points` (registration_landmarks.json) |
| Facit | `.json` named ground_truth / truth / gt / facit, or `parameters` without frames |
| Video | mp4 / mov / m4v / webm |
| Reg.-billede | png / jpg |

## Panels

* **Top bar:** switch between Analyse and Registrering. Export the lower jaw in its current
  position as a binary STL (in upper-STL coordinates), a PNG screenshot, or a short text summary
  (Resumé). Light/dark theme.
* **Filer:** the drop zone, plus a list of files with their role, size, triangle count and load
  progress.
* **Visning:** show or hide each jaw and set its transparency. "Facit-position (spøgelse)" shows
  the lower jaw at its ground-truth pose when ground truth is loaded. Orientation comes from PCA:
  the smallest axis is the occlusal normal. Up/down is set by the upper versus lower centroid.
  Front/back: the end of the arch with the narrower lateral spread is taken as anterior.
  "Drej X/Y/Z" rotates the view 90°. "Auto-orientér" resets it. "Tilpas" fits the model (F).
* **Camera presets** (bottom left of the 3D view, keys 1–6):
  * Frontal, Lateral H, Lateral V.
  * Okklusal OK: seen from below, with the lower jaw hidden.
  * Okklusal UK: seen from above, with the upper jaw hidden.
  * Sagittalsnit: a lateral view with a clipping plane through the incisal point, keeping the
    patient's left half.
  * Mouse/touch: rotate (TrackballControls), right-drag or two fingers to pan, wheel or pinch to
    zoom.
* **Kontakter:** colours each vertex by its signed distance to the opposing jaw, using
  three-mesh-bvh `closestPointToPoint` and the face normal for the sign:
  * dark red: overlap (inside the other jaw)
  * red: contact, ≤ 0.05 mm
  * orange: 0.05–0.3 mm
  * yellow: 0.3–1 mm
  * otherwise the base colour.

  Only candidate vertices are tested: those in a 9 mm occlusal band that face the opposing jaw.
  During playback or scrubbing ("hurtig"), the test uses ≤ 15,000 voxel-clustered representatives
  and runs at most every 110 ms. When paused ("fuld opløsning"), it uses up to 120,000 voxel
  representatives (every candidate on normal scans), computed in time-sliced chunks with
  progress shown. The panel also shows counts per class, the minimum distance or largest overlap,
  and contact areas grouped by arch region (front / præmolarer / molarer H/V).
  "Også på overkæbe" colours the upper jaw too.
* **Spor:** traces for the incisal point and both condyle points from `landmarks_mm`. The full
  path is faint, the last 1 s is drawn bright, and a sphere marks the current position. If
  `landmarks_mm` is missing, the viewer uses fallbacks and says so under "Forbehold":
  * incisal point: the registration incisal point, or else the most anterior-superior lower
    vertex
  * condyles: Bonwill average (105 mm triangle, 22° Balkwill)
* **Timeline:**
  * Play/pause (space), frame step (←/→, Shift = 1 s), speed 0.25×–2×, scrub on the bar.
  * The bar shows segment bands, a quality sparkline and red ticks for invalid frames.
  * On an invalid frame, the HUD says "Ugyldigt billede" and the pose of the last valid frame is
    held.
  * Segment chips (MI, Åbning/lukning, Protrusion, Laterotrusion H/V, Tygning) jump to a segment
    and loop it. Click the chip again to stop looping.
* **Video:** a floating, draggable panel. Video time = motion `t + source.video_offset_s` + a nudge
  (±1 frame, ±0.1 s). The motion clock is the master; the video is re-seeked if it drifts more
  than 0.2 s while playing.
* **Bevægelsesbaner (right panel):** Posselt-style projections of the incisal point displacement
  relative to MI (Frontal / Sagittal / Horisontal tabs).
  * Frame used: occlusal-plane normal (from `landmarks_mm` if present), anterior =
    incisal − condyle midpoint.
  * Lines are coloured by segment, a cross marks MI, and the grid is 5 or 10 mm.
* **Artikulatorparametre:** the `parameters` from motion.json with Danish labels and units, the
  hinge-axis fit RMS, quality (with warnings for registration RMS > 4 px, tracked fraction < 90 %
  and jitter), and `notes` listed as caveats under "Forbehold".
  * If ground truth is loaded, the panel shows **Estimeret / Facit / Δ**. Δ is red beyond
    2 mm (opening), 1 mm (other distances) or 3° (angles).
  * It also shows the RMS and maximum incisal-point deviation over all frames, and the live
    deviation in the HUD.
  * Ground truth alone (without motion.json) is animated if it has frames with `T`.
* **Registrering:** registration mode (described below).

### Registration mode

1. Load `reg_frame.png` and both STLs. The lower jaw is shown in MI.
2. Click a landmark in the image, then the same point on the STL. The ray hits the upper or the
   lower mesh, and `jaw` is set to whichever was hit.
3. Names: the next unused suggestion (11_incisal, 21_incisal, 13_cusp, 23_cusp, 16_mb, 26_mb,
   33_cusp, 43_cusp, 36_mb, 46_mb), or type your own in "Navn på næste punkt". Names can be
   edited in the list.
4. Points can be clicked in either order: image first or STL first.
5. Other controls:
   * "Fortryd" (Ctrl/⌘-Z) undoes, × deletes a point, and "Ryd alle" clears all points.
   * Image navigation: wheel or pinch to zoom, drag to pan, plus "Hele billedet" and +/−.
   * Separate buttons set the lower incisal point and the 3 occlusal-plane points.
   * "Billednummer i video" fills `image.frame_index`. It is guessed from digits in the image
     file name.
6. Click "Eksporter registration_landmarks.json". The output follows SPEC 4 exactly:
   * `img_px` is in **original image pixels**, with the OpenCV convention (pixel centres at
     integer coordinates), whatever the zoom or devicePixelRatio.
   * `stl_mm` is in STL coordinates.
   * Only complete points are exported. A warning appears if there are fewer than 4.
7. To review an existing `registration_landmarks.json`, drop it in. The points are drawn on the
   image and on the STL, and you can edit and re-export them.

## Tests (actually run, headless)

`dev/` holds the test tooling, which the shipped page does not use:

* `make_fixture.py` writes `dev/fixture/`:
  * simple STL arches, binary plus an ASCII copy
  * a SPEC 5.2 `motion.json`: 34 s of MI, open/close ×3, protrusion ×2, laterotrusion R/L ×2,
    chewing and MI, with 33 invalid frames (`T: null`)
  * a `ground_truth.json`
  * a `registration_landmarks.json`, and a `reg_frame.png` with red dots at its `img_px`
  * a webm/mp4 counter video.

  `--res 14 --no-video --out dev/big` writes two ~100 MB STLs (2.09 M triangles each). They were deleted after the test to save space; regenerate them with this command.
* `test_viewer.py` uses Playwright with Chromium and SwiftShader WebGL. It writes
  `dev/index.test.html`, which is identical to the shipped page except that the importmap points
  to `dev/node_modules` because the sandbox blocks the CDN. It then serves `viewer/` and runs the
  checks. Results go to `dev/test_results*.json`. Run it with
  `python3 dev/test_viewer.py [--data DIR --motion FILE --prefix P --chrome]`.
* `test_big.py`: the large-STL load and performance test.

Verified on the fixture (all 31 checks pass) and on the synth agent's
`testdata/synth_quick` (all 23 applicable checks pass; run with `--chrome` for H.264 video):

* **Row-major → three.js:** `Matrix4.set(...T)` puts the translation at elements[12..14].
  A point transformed by the Matrix4 equals the row-major math. The round trip back to row-major
  is exact, nested 4×4 input is accepted, and malformed input is rejected. The pose used for
  export equals the nearest frame's row-major `T` from the file.
* **File-role detection:** 20 name/content cases.
* **Contact thresholds:** the boundaries (−0.05 / 0.05 / 0.3 / 1.0 mm) map to the right classes.
* **PCA orientation** recovers the known S frame on both the fixture and the synth STLs
  (L≈+x, U≈+y, A≈+z; synth up axis tilted 6.6°, which is the real occlusal-plane tilt).
* **STL handling:**
  * binary and ASCII parse identically
  * vertex welding works
  * exporting the lower jaw at the current pose and parsing it back matches the
    transformed vertices (max error 2e-6 mm)
  * the downloaded STL is valid binary.
* **Contacts:** present in MI, none at maximal opening. Paused mode uses full resolution; playback
  switches to the fast mode (2–43 ms per update).
* **Timeline:**
  * Playback advances time.
  * A segment chip jumps to its segment and loops it.
  * An invalid frame is flagged in the HUD.
* **Video sync:** `currentTime = t + video_offset_s` (offset 1.0 s in the fixture), and the
  +0.1 s nudge moves the video. Tested with webm in Chromium and H.264 mp4 in Chrome.
* **Ground-truth comparison:** the Estimeret / Facit / Δ table renders. Summary, STL and PNG
  downloads work.
* **Registration:**
  * After zooming and panning (display scale 0.68), the red dots were found in the
    rendered canvas pixels and clicked with the mouse.
  * The exported `img_px` are within 0.04–0.34 px of the true dot centres (in 1920×1080 image
    pixels).
  * A raycast click on the 3D model sets the incisal point at the clicked tooth.
  * The export schema matches SPEC 4.
  * Importing `registration_landmarks.json` and exporting again gives an identical file
    (fixture and synth).
* **Layout:** at phone width there is no horizontal page scroll. No page errors occurred.
* **Large files:** 2 × 104 MB binary STL plus motion loaded in 18.8 s in total in this 2-core
  container:
  * parse+weld in the worker 1.1–1.5 s per file, BVH ~1.2–2.3 s, orientation/traces/contact
    setup 0.6 s
  * load progress is shown per stage
  * the fast contact mode took 43 ms per update.

  Software (CPU) rendering of 4 M triangles caused 9–15 s frame gaps here. That is a SwiftShader
  limitation, not a GPU number, and is not representative of a Mac or iPad.

Screenshots (`screenshots/`):

| File | Shows |
|---|---|
| `01_empty.png` | empty state |
| `02_loaded_open.png` | loaded, at maximal opening |
| `03_contacts_MI.png` | contacts in MI |
| `04_contacts_occlusal.png` | lower occlusal view with the contact map |
| `05_playing_laterotrusion.png` | during playback |
| `06_protrusion_lateral_video.png` | protrusion, lateral view, with video |
| `07_sagittal_section.png` | sagittal section |
| `08_compare_truth.png` | estimated vs ground truth |
| `09_dark.png` | dark theme |
| `10_registration_clicking.png` | registration, clicking points |
| `11_registration_review.png` | registration, reviewing imported landmarks |
| `12_ipad_portrait.png` | iPad portrait |
| `13_phone.png` | phone |
| `14_big_loading_progress.png`, `15_big_loaded.png` | 100 MB STLs |
| `synth_*.png` | the same states with the synth dataset |

In screenshots where the video panel is open and the ghost is on, a rectangular gap appears in the
3D view below the video panel. This is a headless compositor artefact. It disappears with the
video closed (checked separately) and has not been seen outside headless runs.

## Known limitations / unverified

* **Not tested on real Safari, iPad or iPhone, or a real GPU.** Only headless Chromium/Chrome was
  run. Safari needs 16.4+ for import maps. iOS Safari may not autoplay or seek video precisely
  until the user has tapped play once.
* The fixture and synth STLs come from scanner-free generators. Real intraoral scans have
  arbitrary axes, open bases and possibly inconsistent winding:
  * The PCA anterior/posterior guess could be wrong. Use "Drej X/Y/Z".
  * The overlap sign uses the opposing mesh's global orientation, estimated from its signed
    volume. On a badly oriented open scan, "overlap" can be misclassified as distance. Contact
    classes 1–3 use |distance| and are unaffected.
* Contact counts are counts of vertices or voxel representatives, not areas. In fast mode they
  are approximate.
* Full-resolution contacts on dense scans take a few seconds on a laptop: 16 s for 450,000
  candidates in this container before the 120,000-representative cap was added.
* Memory: two 100 MB STLs use roughly 0.6–0.8 GB in the tab (positions, normals, index, colours
  and BVH). Older iPads may refuse this.
* A `.mov` from the iPhone (HEVC) plays in Safari. Chrome needs H.264 or HEVC hardware support.
* Video sync is accurate to about one frame when paused. During playback it is re-seeked only when
  drift exceeds 0.2 s.
* The condyle traces are only as good as `landmarks_mm`. Bonwill fallbacks are labelled as such.
