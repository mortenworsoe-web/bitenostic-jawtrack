"""End-to-end synthetic check (self-contained; does not need testdata/).

Renders the real ArUco boards into a camera with known poses following the SPEC protocol
(H.264, stored rotated with an iPhone-style rotation tag), runs track + solve with exact
landmarks and compares T_S(t) to ground truth.

Run:  python3 -m pytest -q tests/test_synthetic_e2e.py      (or)   python3 tests/test_synthetic_e2e.py [outdir]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jawtrack import se3, synthetic  # noqa: E402
from jawtrack.io import read_json  # noqa: E402
from jawtrack.params import compute as compute_params  # noqa: E402
from jawtrack.landmarks import occlusal_frame  # noqa: E402
from jawtrack.solve import solve  # noqa: E402
from jawtrack.track import track_video  # noqa: E402


def ts_errors(motion, gt, inc):
    """Per-frame translation error at the incisal point (mm) and rotation error (deg)."""
    et, er, tt = [], [], []
    gtt = gt["times"]
    for f in motion["frames"]:
        if not f["valid"]:
            continue
        k = int(np.argmin(np.abs(gtt - f["t"])))
        if abs(gtt[k] - f["t"]) > 1e-3:
            continue
        T = se3.from_list(f["T"]); G = gt["T_S"][k]
        et.append(np.linalg.norm(se3.apply(T, inc) - se3.apply(G, inc)))
        er.append(se3.rot_dist_deg(T[:3, :3], G[:3, :3]))
        tt.append(f["t"])
    return np.array(et), np.array(er), np.array(tt)


def gt_params(motion, gt):
    """Parameters computed by the same definitions from the GROUND-TRUTH motion and GT segments."""
    lm = motion["landmarks_mm"]
    of_d = lm["occlusal_frame"]
    from jawtrack.landmarks import OcclusalFrame
    of = OcclusalFrame(of_d["point"], of_d["normal"], of_d["anterior"])
    segs = read_json(Path(motion["_dir"]) / "segments_gt.json")["segments"]
    P, _, _ = compute_params(gt["times"], gt["T_S"], np.ones(len(gt["times"]), bool), segs, of,
                             np.asarray(lm["incisal_point"]), np.asarray(lm["condyle_left"]),
                             np.asarray(lm["condyle_right"]))
    return P


def run_e2e(outdir, cfg=None, noise_px=0.0, verbose=True):
    outdir = Path(outdir)
    t0 = time.time()
    gt = synthetic.generate(outdir, cfg, log=print if verbose else (lambda *a: None))
    t1 = time.time()
    raw = track_video(outdir / "video.mov", outdir / "camera.json", outdir / "raw_tracking.json")
    t2 = time.time()
    rf = raw["registration_frame"]
    synthetic.make_landmarks_json(gt, rf["frame_index"], rf["t"], raw["width"], raw["height"], noise_px=noise_px,
                                  path=outdir / "registration_landmarks.json")
    motion = solve(outdir / "raw_tracking.json", outdir / "upper.stl", outdir / "lower.stl",
                   outdir / "registration_landmarks.json", outdir / "motion.json", out_positions=outdir / "positions")
    t3 = time.time()
    inc = synthetic.LOWER_INCISAL
    et, er, tt = ts_errors(motion, gt, inc)
    motion["_dir"] = str(outdir)
    Pg = gt_params(motion, gt)
    P = motion["parameters"]
    res = {
        "frames": len(gt["times"]), "valid_frames_compared": int(len(et)),
        "incisal_err_mm_median": float(np.median(et)), "incisal_err_mm_p95": float(np.percentile(et, 95)),
        "incisal_err_mm_max": float(et.max()),
        "rot_err_deg_median": float(np.median(er)), "rot_err_deg_p95": float(np.percentile(er, 95)),
        "registration_rms_px": motion["quality"]["registration_rms_px"],
        "jitter_mm_rms_in_MI": motion["quality"]["jitter_mm_rms_in_MI"],
        "tracked_fraction": motion["quality"]["tracked_fraction"],
        "segments": [(s["label"], round(s["t0"], 2), round(s["t1"], 2)) for s in motion["segments"]],
        "params_measured_vs_gt": {k: (None if P.get(k) is None else round(P[k], 2),
                                      None if Pg.get(k) is None else round(Pg[k], 2))
                                  for k in P if k.endswith(("_mm", "_deg"))},
        "hinge_axis": P.get("kinematic_hinge_axis"),
        "seconds": {"generate": round(t1 - t0, 1), "track": round(t2 - t1, 1), "solve": round(t3 - t2, 1)},
        "track_fps": raw["processing"]["fps"],
    }
    (outdir / "e2e_result.json").write_text(json.dumps(res, indent=1, default=float))
    return res, motion, gt


def test_synthetic_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        res, motion, gt = run_e2e(d, verbose=False)
    print(json.dumps(res, indent=1, default=float))
    assert res["tracked_fraction"] > 0.95
    assert res["registration_rms_px"] < 1.0
    assert res["incisal_err_mm_median"] < 0.5
    assert res["incisal_err_mm_p95"] < 1.0
    assert res["rot_err_deg_median"] < 0.5
    labels = [s[0] for s in res["segments"]]
    for lab in ("MI", "open_close", "protrusion", "laterotrusion_right", "laterotrusion_left", "chewing"):
        assert lab in labels, lab
    pm = res["params_measured_vs_gt"]
    assert abs(pm["max_opening_mm"][0] - pm["max_opening_mm"][1]) < 1.5
    assert abs(pm["max_protrusion_mm"][0] - pm["max_protrusion_mm"][1]) < 1.0
    for k in ("sagittal_condylar_inclination_left_deg", "sagittal_condylar_inclination_right_deg"):
        assert abs(pm[k][0] - pm[k][1]) < 5.0, k


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="jawtrack_e2e_")
    res, _, _ = run_e2e(out)
    print(json.dumps(res, indent=1, default=float))
