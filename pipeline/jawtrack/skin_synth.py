"""Synthetic test for skin-marker tracking (version 2): four clips, six individual 12 mm markers,
with a simple soft-tissue model: every jaw marker LAGS behind the bone by a fraction ``lag`` of its
own displacement (skin does not follow the mandible fully), different per marker so the
constellation also deforms. lag = 0 gives the ideal (rigid) case.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import se3
from .boards import aruco_dictionary
from .skin import HEAD_IDS, JAW_IDS, MARKER_MM
from .synthetic import (LOWER_INCISAL, R_S_BOARD, JawModel, Renderer, SynthConfig, camera_K, head_motion,
                        make_arch_mesh, scene)


def _mpose(pos, yaw=0.0, pitch=0.0):
    R = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix() @ R_S_BOARD
    return se3.make_T(R, pos)


# marker poses in the STL frame at MI (x = patient's left, y = up, z = anterior; origin upper incisal point)
HEAD_MARKERS = {20: _mpose([0, 46, 16], pitch=-20), 21: _mpose([-30, 86, 6], yaw=-12, pitch=-8),
                22: _mpose([30, 86, 6], yaw=12, pitch=-8)}
JAW_MARKERS = {30: _mpose([0, -43, -1], pitch=12), 31: _mpose([-37, -38, -24], yaw=-38),
               32: _mpose([37, -38, -24], yaw=38)}
LAG_WEIGHT = {30: 0.6, 31: 1.0, 32: 0.85}  # relative lag per jaw marker

CLIPS = [("open", "open", 2, 2.6), ("right", "right", 1, 3.0), ("left", "left", 1, 3.0),
         ("protrusion", "protrusion", 1, 3.0)]


class MarkerRenderer(Renderer):
    PPM = 16

    def __init__(self, cfg, rng, side=MARKER_MM, margin_mm=3.0):
        super().__init__(cfg, rng)
        self.tex = {}
        m = int(round(margin_mm * self.PPM)); s = int(round(side * self.PPM))
        for i in HEAD_IDS + JAW_IDS:
            img = np.full((s + 2 * m, s + 2 * m), 255, np.uint8)
            img[m:m + s, m:m + s] = cv2.aruco.generateImageMarker(aruco_dictionary(), i, s)
            off = (0.5 - m) / self.PPM - side / 2.0
            A = np.array([[1.0 / self.PPM, 0, off], [0, 1.0 / self.PPM, off], [0, 0, 1]])
            self.tex[i] = (img, A)


def generate_skin(outdir, lag=0.0, width=1920, height=1080, fps=30.0, seed=11, mi_s=2.0, log=print):
    """Writes clip_<label>.mp4 x4, camera.json, upper/lower.stl; returns ground truth dict."""
    from .io import camera_dict, write_json
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = SynthConfig(width=width, height=height, fps=fps, seed=seed, rotation_tag=False)
    rng = np.random.default_rng(seed)
    model = JawModel()
    _, _, T_C_S0 = scene()
    K = camera_K(cfg)
    rend = MarkerRenderer(cfg, rng)
    fns = {"open": model.open, "protrusion": model.protrude, "right": lambda s: model.lateral(s, "right"),
           "left": lambda s: model.lateral(s, "left")}
    gt = {"times": [], "T_S": [], "T_C_S": [], "K": K, "clips": []}
    t_glob = 0.0
    dt = 1.0 / fps
    for label, kind, reps, dur in CLIPS:
        n = int(round((mi_s + reps * dur + 0.5) * fps))
        path = outdir / f"clip_{label}.mp4"
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s",
                                 f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset",
                                 "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        t0 = t_glob
        for k in range(n):
            t = k * dt
            if t < mi_s or t >= mi_s + reps * dur:
                T = np.eye(4)
            else:
                tau = ((t - mi_s) % dur) / dur
                T = fns[kind](np.sin(np.pi * tau) ** 2)
            # pretend the head keeps drifting between clips: offset the head-motion clock per clip
            A = T_C_S0 @ head_motion(t_glob + t + 17.0 * len(gt["clips"]))
            poses = {i: A @ Tm for i, Tm in HEAD_MARKERS.items()}
            for i, Tm in JAW_MARKERS.items():
                p = Tm[:3, 3]
                d = se3.apply(T, p) - p
                poses[i] = A @ se3.make_T(None, -lag * LAG_WEIGHT[i] * d) @ T @ Tm
            proc.stdin.write(rend.render(K, poses).tobytes())
            gt["times"].append(t_glob + t); gt["T_S"].append(T); gt["T_C_S"].append(A)
        proc.stdin.close(); proc.wait()
        t_glob += n * dt
        gt["clips"].append((label, str(path), t0, t_glob))
        log(f"synthetic skin clip '{label}': {n} frames -> {path}")
    for k in ("times", "T_S", "T_C_S"):
        gt[k] = np.array(gt[k])
    gt["condyle_left"], gt["condyle_right"], gt["incisal"] = model.cL, model.cR, LOWER_INCISAL
    write_json(outdir / "camera.json", camera_dict(K, [0] * 5, width, height, "synthetic"))
    make_arch_mesh("upper").export(str(outdir / "upper.stl"))
    make_arch_mesh("lower").export(str(outdir / "lower.stl"))
    return gt


def run_skin_e2e(outdir, lag=0.0, noise_px=1.0, log=print, **kw):
    """Generate -> track_skin -> solve -> compare with ground truth. Returns a result dict."""
    from .skin import track_skin
    from .solve import solve
    from .synthetic import BENNETT_GT, SCI_GT, make_landmarks_json
    outdir = Path(outdir)
    gt = generate_skin(outdir, lag=lag, log=log, **kw)
    raw = track_skin([(lab, p) for lab, p, _, _ in gt["clips"]], outdir / "camera.json", outdir / "raw_tracking.json")
    rf = raw["registration_frame"]
    make_landmarks_json(gt, rf["frame_index"], rf["t"], raw["width"], raw["height"], noise_px=noise_px,
                        rng=np.random.default_rng(3), path=outdir / "registration_landmarks.json")
    motion = solve(outdir / "raw_tracking.json", outdir / "upper.stl", outdir / "lower.stl",
                   outdir / "registration_landmarks.json", outdir / "motion.json",
                   segments_path=outdir / "segments.json", refine_registration="off")
    inc = LOWER_INCISAL
    err, lab = [], []
    for f in motion["frames"]:
        if not f.get("valid", True):
            continue
        k = int(np.argmin(np.abs(gt["times"] - f["t"])))
        e = np.linalg.norm(se3.apply(se3.from_list(f["T"]), inc) - se3.apply(gt["T_S"][k], inc))
        err.append(e)
        lab.append(next(c[0] for c in gt["clips"] if c[2] <= gt["times"][k] < c[3] + 1e-6))
    err, lab = np.array(err), np.array(lab)
    P = motion["parameters"]
    true_max = {}
    for c in gt["clips"]:
        m = (gt["times"] >= c[2]) & (gt["times"] < c[3])
        d = np.array([se3.apply(T, inc) - inc for T in gt["T_S"][m]])
        # same sense as the pipeline: 3-D for opening, anterior (z) for protrusion, lateral (x) for laterotrusion
        true_max[c[0]] = float({"open": np.linalg.norm(d, axis=1).max(), "protrusion": d[:, 2].max()}.get(
            c[0], np.abs(d[:, 0]).max()))
    res = {"lag": lag, "click_noise_px": noise_px,
           "incisal_err_median_mm": float(np.median(err)), "incisal_err_p95_mm": float(np.percentile(err, 95)),
           "incisal_err_p95_per_clip_mm": {c[0]: float(np.percentile(err[lab == c[0]], 95)) for c in gt["clips"]},
           "tracked_fraction": motion["quality"]["tracked_fraction"],
           "max_opening_mm": (P.get("max_opening_mm"), true_max["open"]),
           "max_protrusion_mm": (P.get("max_protrusion_mm"), true_max["protrusion"]),
           "max_laterotrusion_right_mm": (P.get("max_laterotrusion_right_mm"), true_max["right"]),
           "max_laterotrusion_left_mm": (P.get("max_laterotrusion_left_mm"), true_max["left"]),
           "sci_left_deg": (P.get("sagittal_condylar_inclination_left_deg"), SCI_GT),
           "sci_right_deg": (P.get("sagittal_condylar_inclination_right_deg"), SCI_GT),
           "bennett_left_deg": (P.get("bennett_angle_left_deg"), BENNETT_GT),
           "bennett_right_deg": (P.get("bennett_angle_right_deg"), BENNETT_GT),
           "skin_misfit": raw["skin_deformation"]}
    return res
