#!/usr/bin/env python3
"""JawTrack pipeline CLI.

  calibrate  ChArUco video -> camera.json   (or --preset iphone_main_4k|iphone_main_1080p)
  track      video + camera.json -> raw_tracking.json + reg_frame.png
  solve      raw_tracking.json + upper/lower STL + registration_landmarks.json -> motion.json
  run        calibrate/preset + track (+ solve + report if a landmarks file exists)
  report     motion.json -> PNG plots + report.md
  selftest   synthetic end-to-end check with ground truth

Run `python3 jawtrack_cli.py <command> --help` for options.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("jawtrack")


def cmd_calibrate(a):
    from jawtrack.calib import calibrate_video, preset_camera
    from jawtrack.io import VideoReader, write_json
    if a.preset:
        w = h = None
        if a.video:
            vr = VideoReader(a.video)
            w, h = vr.width, vr.height
        cam = preset_camera(a.preset, w, h)
        log.warning("using PRESET intrinsics (%s): %s", a.preset, cam["assumptions"])
    else:
        if not a.video:
            sys.exit("calibrate: give --video charuco.mov or --preset")
        cam, info = calibrate_video(a.video, a.print_scale)
        print(f"RMS reprojection error: {info['rms_px']:.3f} px over {info['n_views']} views")
    write_json(a.out, cam, indent=1)
    return cam


def cmd_track(a):
    from jawtrack.track import track_video
    raw = track_video(a.video, a.camera, a.out, print_scale=a.print_scale, every=a.every, max_frames=a.max_frames,
                      reg_image=a.reg_image, max_side=a.max_side, refine=a.refine, force_ffmpeg=a.ffmpeg)
    n = len(raw["frames"])
    both = sum(1 for f in raw["frames"] if f["head"] and f["jaw"])
    print(f"tracked {both}/{n} frames with both boards ({raw['processing']['fps']:.1f} fps processing)")
    return raw


def cmd_solve(a):
    from jawtrack.solve import solve
    m = solve(a.raw, a.upper, a.lower, a.landmarks, a.out, out_positions=a.out_positions,
              mi_interval=a.mi_interval, segments_path=a.segments, condyles=a.condyles,
              smooth_s=a.smooth, max_gap_s=a.max_gap, refine_registration=a.refine_registration)
    q, p = m["quality"], m["parameters"]
    print(f"registration RMS {q['registration_rms_px']:.2f} px, tracked {100 * q['tracked_fraction']:.1f}%, "
          f"jitter in MI {q['jitter_mm_rms_in_MI'] or float('nan'):.3f} mm")
    for k, v in p.items():
        if k.endswith(("_mm", "_deg")):
            print(f"  {k:42s} {'n/a' if v is None else f'{v:.1f}'}")
    return m


def cmd_track_skin(a):
    from jawtrack.skin import CLIP_LABELS, track_skin
    clips = []
    for c in a.clip:
        if "=" not in c:
            sys.exit(f"--clip expects label=file (labels: {', '.join(CLIP_LABELS)}), got {c!r}")
        lab, f = c.split("=", 1)
        clips.append((lab, f))
    raw = track_skin(clips, a.camera, a.out, marker_mm=a.marker_mm, print_scale=a.print_scale, every=a.every,
                     max_side=a.max_side, mi_hold_s=a.mi_hold)
    d = raw["skin_deformation"]
    print(f"skin-marker tracking written to {a.out}")
    for c in raw["clips"]:
        print(f"  clip {c['label']:11s} {c['frames']:5d} frames, all 6 markers in {100 * c['all_markers_fraction']:.0f}%")
    print(f"  constellation misfit: head {d['head']['median_mm']:.2f} mm, jaw {d['jaw']['median_mm']:.2f} mm "
          f"(p95 {d['jaw']['p95_mm']:.2f})")
    print(f"next: click landmarks on {Path(a.out).parent / 'reg_frame.png'} in the viewer, then\n"
          f"  solve --raw {a.out} --segments {Path(a.out).parent / 'segments.json'} --upper ... --lower ... "
          f"--landmarks ... --out motion.json")


def cmd_skin_selftest(a):
    import json
    from jawtrack.skin_synth import run_skin_e2e
    r = run_skin_e2e(a.outdir, lag=a.lag, noise_px=a.click_noise)
    txt = json.dumps(r, indent=1, default=lambda x: round(float(x), 3))
    (Path(a.outdir) / "skin_selftest_result.json").write_text(txt)
    print(txt)
    print(f"SKIN SELFTEST lag={a.lag}: incisal error median {r['incisal_err_median_mm']:.2f} mm, "
          f"p95 {r['incisal_err_p95_mm']:.2f} mm, tracked {100 * r['tracked_fraction']:.0f}%")


def cmd_report(a):
    from jawtrack.report import make_report
    make_report(a.motion, a.outdir)


def cmd_run(a):
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cam = out / "camera.json"
    if a.camera:
        cam = Path(a.camera)
    else:
        cmd_calibrate(argparse.Namespace(video=a.calib if not a.preset else a.video, preset=a.preset,
                                         out=cam, print_scale=a.print_scale))
    raw = out / "raw_tracking.json"
    cmd_track(argparse.Namespace(video=a.video, camera=cam, out=raw, print_scale=a.print_scale, every=a.every,
                                 max_frames=a.max_frames, reg_image=a.reg_image, max_side=a.max_side,
                                 refine="subpix", ffmpeg=False))
    lmk = Path(a.landmarks) if a.landmarks else out / "registration_landmarks.json"
    if not lmk.exists():
        print("\n" + "=" * 78 + "\nTRACKING DONE. Registration landmarks are missing:\n"
              f"  open the viewer, load {out / 'reg_frame.png'} and the STLs, click >= 6 matching points\n"
              f"  (image <-> STL), export registration_landmarks.json to {lmk}, then run:\n"
              f"  python3 jawtrack_cli.py solve --raw {raw} --upper {a.upper} --lower {a.lower} "
              f"--landmarks {lmk} --out {out / 'motion.json'} --out-positions {out / 'positions'}\n" + "=" * 78)
        return
    if not (a.upper and a.lower):
        sys.exit("run: --upper and --lower STLs are required for solve")
    cmd_solve(argparse.Namespace(raw=raw, upper=a.upper, lower=a.lower, landmarks=lmk, out=out / "motion.json",
                                 out_positions=out / "positions", mi_interval=None, segments=None,
                                 condyles=a.condyles, smooth=0.12, max_gap=0.25, refine_registration="auto"))
    cmd_report(argparse.Namespace(motion=out / "motion.json", outdir=out / "report"))


def cmd_selftest(a):
    import json
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    from jawtrack.synthetic import SynthConfig
    from test_synthetic_e2e import run_e2e
    cfg = SynthConfig(width=a.width, height=a.height, fps=a.fps)
    out = a.outdir or tempfile.mkdtemp(prefix="jawtrack_selftest_")
    res, _, _ = run_e2e(out, cfg)
    print(json.dumps(res, indent=1, default=float))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="camera calibration from a ChArUco video, or a preset")
    c.add_argument("--video", help="ChArUco calibration video (DICT_5X5_100, 7x5, 25/18 mm)")
    c.add_argument("--out", required=True, help="camera.json")
    c.add_argument("--print-scale", type=float, default=1.0, help="measured_mm/100 of the printed scale bar")
    c.add_argument("--preset", choices=["iphone_main_4k", "iphone_main_1080p"],
                   help="approximate intrinsics instead of calibration (fx=long_side*26/36, centred, no distortion);"
                        " with --video the resolution/orientation is taken from that video")
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("track", help="detect boards and estimate HEAD/JAW poses per frame")
    t.add_argument("--video", required=True)
    t.add_argument("--camera", required=True, help="camera.json from calibrate")
    t.add_argument("--out", required=True, help="raw_tracking.json (reg_frame.png is written next to it)")
    t.add_argument("--print-scale", type=float, default=1.0)
    t.add_argument("--every", type=int, default=1, help="process every N-th frame (2 for 4K60 -> 30 Hz)")
    t.add_argument("--max-frames", type=int, default=None)
    t.add_argument("--reg-image", default=None,
                   help="separate registration still OR short registration video (.mov/.mp4); default: sharpest "
                        "frame of the first MI hold of the main video")
    t.add_argument("--max-side", type=int, default=1920, help="long side for full-frame detection (downscale)")
    t.add_argument("--refine", choices=["subpix", "apriltag", "contour"], default="subpix",
                   help="corner refinement (subpix = full-resolution cornerSubPix; best in synthetic tests)")
    t.add_argument("--ffmpeg", action="store_true", help="force decoding through an ffmpeg pipe")
    t.set_defaults(func=cmd_track)

    k = sub.add_parser("track-skin", help="v2: six small skin markers (IDs 20-22 cranium, 30-32 mandible), "
                                          "one or more clips -> raw_tracking.json + segments.json")
    k.add_argument("--clip", action="append", required=True, metavar="LABEL=FILE",
                   help="repeat per clip, in recording order; labels: open, right, left, protrusion, chew")
    k.add_argument("--camera", required=True, help="camera.json (calibrate, or calibrate --preset)")
    k.add_argument("--out", required=True, help="raw_tracking.json (reg_frame.png, segments.json, combined_1080p.mp4 next to it)")
    k.add_argument("--marker-mm", type=float, default=12.0)
    k.add_argument("--print-scale", type=float, default=1.0)
    k.add_argument("--every", type=int, default=1)
    k.add_argument("--max-side", type=int, default=1920)
    k.add_argument("--mi-hold", type=float, default=1.0, help="seconds of MIP hold assumed at the start of each clip")
    k.set_defaults(func=cmd_track_skin)

    ks = sub.add_parser("skin-selftest", help="v2: render 4 synthetic clips with 6 skin markers, track, solve "
                                              "and compare with ground truth")
    ks.add_argument("--outdir", required=True)
    ks.add_argument("--lag", type=float, default=0.0, help="simulated skin lag (0 = skin follows bone, 0.15 = 15%% lag)")
    ks.add_argument("--click-noise", type=float, default=1.0, help="landmark click noise in px")
    ks.set_defaults(func=cmd_skin_selftest)

    s = sub.add_parser("solve", help="registration, motion, segmentation, parameters -> motion.json")
    s.add_argument("--raw", required=True)
    s.add_argument("--upper", required=True)
    s.add_argument("--lower", required=True)
    s.add_argument("--landmarks", required=True, help="registration_landmarks.json from the viewer")
    s.add_argument("--out", required=True, help="motion.json")
    s.add_argument("--out-positions", default=None, help="dir for key-position STLs + articulator_values.txt")
    s.add_argument("--mi-interval", nargs=2, type=float, metavar=("T0", "T1"), help="override MI reference interval (s)")
    s.add_argument("--segments", default=None, help="segments.json override ({'segments': [{label,t0,t1}]})")
    s.add_argument("--condyles", choices=["average", "kinematic"], default="average",
                   help="average = Bonwill/Balkwill average-value; kinematic = on the fitted hinge axis +-52.5 mm")
    s.add_argument("--smooth", type=float, default=0.12, help="Savitzky-Golay window in seconds (0 = off)")
    s.add_argument("--max-gap", type=float, default=0.25, help="max dropout (s) bridged by interpolation")
    s.add_argument("--refine-registration", choices=["auto", "on", "off"], default="auto",
                   help="refine X by aligning rendered STL silhouette/crease edges to image edges of the registration "
                        "image (auto = when the image is found next to the landmarks/raw file; before/after reported)")
    s.set_defaults(func=cmd_solve)

    r = sub.add_parser("run", help="end-to-end: calibrate|preset -> track -> (solve -> report)")
    r.add_argument("--video", required=True)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib", help="ChArUco calibration video")
    g.add_argument("--preset", choices=["iphone_main_4k", "iphone_main_1080p"])
    g.add_argument("--camera", help="existing camera.json")
    r.add_argument("--upper"); r.add_argument("--lower")
    r.add_argument("--landmarks", default=None, help="default: <outdir>/registration_landmarks.json")
    r.add_argument("--reg-image", default=None)
    r.add_argument("--outdir", required=True)
    r.add_argument("--print-scale", type=float, default=1.0)
    r.add_argument("--every", type=int, default=1)
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--max-side", type=int, default=1920)
    r.add_argument("--condyles", choices=["average", "kinematic"], default="average")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="plots + report.md from motion.json")
    rp.add_argument("--motion", required=True)
    rp.add_argument("--outdir", required=True)
    rp.set_defaults(func=cmd_report)

    st = sub.add_parser("selftest", help="synthetic end-to-end test with ground truth")
    st.add_argument("--outdir", default=None)
    st.add_argument("--width", type=int, default=1920)
    st.add_argument("--height", type=int, default=1080)
    st.add_argument("--fps", type=float, default=30.0)
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    try:
        a.func(a)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
