"""Self-contained synthetic data generator for tests (independent of the testdata/ agent).

Renders the real cv2.aruco HEAD/JAW board images into a pinhole camera with known poses
(homography K [r1 r2 t] for the board plane z=0), 2x supersampled, over a textured background,
with blur, noise and H.264/HEVC compression, following the SPEC section 2 protocol. Ground truth:
camera K, X = T_H_S, per-frame T_S(t), and simple arch STLs.

Synthetic STL frame (SPEC 6): x = patient's LEFT, y = SUPERIOR, z = ANTERIOR, origin at the upper
incisal point.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import se3
from .boards import HEAD, JAW, board_size_mm, make_gridboard

R_S_BOARD = np.diag([1.0, -1.0, -1.0])  # board x = patient's left, y = inferior, z = posterior
LOWER_INCISAL = np.array([0.0, -1.5, -1.0])
SCI_GT = 35.0
BENNETT_GT = 10.0


def bonwill_condyles(inc=LOWER_INCISAL, balkwill=22.0):
    back = np.sqrt(105.0 ** 2 - 52.5 ** 2)
    v = back * (np.cos(np.radians(balkwill)) * np.array([0, 0, -1.0]) + np.sin(np.radians(balkwill)) * np.array([0, 1.0, 0]))
    return inc + v + [52.5, 0, 0], inc + v + [-52.5, 0, 0]  # left, right


def rot_about(point, axis, deg):
    R = Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()
    p = np.asarray(point, float)
    return se3.make_T(R, p - R @ p)


def _align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


class JawModel:
    """Ground-truth kinematics: condyle paths at SCI_GT, Bennett ~BENNETT_GT, hinge opening."""

    def __init__(self):
        self.cL, self.cR = bonwill_condyles()
        self.cm = 0.5 * (self.cL + self.cR)
        sci = np.radians(SCI_GT)
        self.emin = np.array([0, -np.sin(sci), np.cos(sci)])  # forward-down along eminence

    def open(self, s):
        theta = 28.0 * s
        tr = 10.0 * max(0.0, s - 0.3) / 0.7
        return se3.make_T(None, tr * self.emin) @ rot_about(self.cm, [1, 0, 0], theta)

    def protrude(self, s):
        # condyles 8 mm along the 35 deg eminence; slight closing rotation about the condylar axis so the
        # incisal path is anterior-dominant (~7 mm forward, ~3 mm down)
        return se3.make_T(None, 8.0 * s * self.emin) @ rot_about(self.cm, [1, 0, 0], -1.0 * s)

    def lateral(self, s, side):
        """side='right': mandible to the patient's right. The orbiting condyle moves 8*s mm forward,
        down (SCI_GT in the sagittal projection) and medially (BENNETT_GT in the horizontal projection);
        the working condyle shifts laterally just enough to keep the intercondylar distance (Bennett
        side shift); no rotation about the intercondylar axis apart from a 1.5 deg opening."""
        sci, ben = np.radians(SCI_GT), np.radians(BENNETT_GT)
        if side == "right":
            work, orb, med = self.cR, self.cL, -1.0
        else:
            work, orb, med = self.cL, self.cR, 1.0
        v = np.array([med * np.sin(ben), -np.cos(ben) * np.tan(sci), np.cos(ben)])
        v /= np.linalg.norm(v) / np.hypot(np.sin(ben), np.cos(ben))  # 8*s = horizontal path length
        d_o = 8.0 * s * v
        D = work - orb
        L = np.linalg.norm(D); e = D / L
        perp = d_o - np.dot(d_o, e) * e
        lam = -L + np.dot(d_o, e) + np.sqrt(L ** 2 - np.dot(perp, perp))
        o2 = orb + d_o
        w2 = work + lam * e
        R = _align(work - orb, w2 - o2)
        T = se3.make_T(R, o2 - R @ orb)
        return T @ rot_about(self.cm, [1, 0, 0], 1.5 * s)


@dataclass
class SynthConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    seed: int = 7
    noise_sigma: float = 2.5
    blur_max: float = 0.8
    crf: int = 20
    codec: str = "libx264"          # or libx265
    rotation_tag: bool = True        # store rotated + display-rotation tag like an iPhone
    head_motion: bool = True
    durations: dict = field(default_factory=lambda: {"MI": 2.0, "open": 2.0, "protrusion": 1.6,
                                                     "lat": 1.6, "chew": 3.0, "MI_end": 1.5, "pause": 0.25})
    reps: int = 2


def protocol(cfg: SynthConfig, model: JawModel):
    """-> (times, T_S list, segments)."""
    d = cfg.durations
    ev = [("MI", d["MI"], None)]
    for lab, dur, fn in [("open_close", d["open"], model.open), ("protrusion", d["protrusion"], model.protrude),
                         ("laterotrusion_right", d["lat"], lambda s: model.lateral(s, "right")),
                         ("laterotrusion_left", d["lat"], lambda s: model.lateral(s, "left"))]:
        for _ in range(cfg.reps):
            ev.append((lab, dur, fn))
            ev.append(("pause", d["pause"], None))
    ev.append(("chewing", d["chew"], "chew"))
    ev.append(("MI", d["MI_end"], None))
    times, Ts, segs = [], [], []
    t = 0.0
    dt = 1.0 / cfg.fps
    tt = 0.0
    for lab, dur, fn in ev:
        t0 = tt
        while t < tt + dur - 1e-9:
            tau = (t - t0) / dur
            if fn is None:
                T = np.eye(4)
            elif fn == "chew":
                ph = 1.3 * (t - t0)
                s = np.sin(np.pi * (ph % 1.0)) ** 2
                side = "right" if int(ph) % 2 == 0 else "left"
                T = model.lateral(0.35 * s, side) @ model.open(0.35 * s)
            else:
                T = fn(np.sin(np.pi * tau) ** 2)
            times.append(t); Ts.append(T)
            t += dt
        if lab != "pause":
            if segs and segs[-1]["label"] == lab:
                segs[-1]["t1"] = tt + dur
            else:
                segs.append({"label": lab, "t0": tt, "t1": tt + dur})
        else:
            segs[-1]["t1"] = tt + dur
        tt += dur
    return np.array(times), np.array(Ts), segs


def scene():
    """Fixed geometry: T_S_H (head board in S), B_S (jaw board in S at MI), T_C_S0 (camera)."""
    wh, hh = board_size_mm(HEAD)
    T_S_H = se3.make_T(R_S_BOARD, [-wh / 2, 95.0, 20.0])
    wj, hj = board_size_mm(JAW)
    B_S = se3.make_T(R_S_BOARD, [-wj / 2, -30.0 + hj / 2, 32.0])
    pos = np.array([0.0, -20.0, 360.0]); target = np.array([0.0, 30.0, 0.0]); up = np.array([0, 1.0, 0])
    f = target - pos; f /= np.linalg.norm(f)
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.stack([x, y, f])
    T_C_S0 = se3.make_T(R, -R @ pos)
    return T_S_H, B_S, T_C_S0


def head_motion(t, on=True):
    if not on:
        return np.eye(4)
    rv = np.radians([2.0 * np.sin(0.31 * t), 2.5 * np.sin(0.23 * t + 1), 1.5 * np.sin(0.41 * t + 2)])
    tr = [3.0 * np.sin(0.27 * t), 2.0 * np.sin(0.19 * t + 0.5), 4.0 * np.sin(0.21 * t + 1.3)]
    return se3.make_T(Rotation.from_rotvec(rv).as_matrix(), tr)


def camera_K(cfg):
    f = max(cfg.width, cfg.height) * 26.0 / 36.0
    return np.array([[f, 0, (cfg.width - 1) / 2], [0, f, (cfg.height - 1) / 2], [0, 0, 1.0]])


class Renderer:
    PPM = 8  # board texture pixels per mm

    def __init__(self, cfg: SynthConfig, rng):
        self.cfg = cfg
        self.rng = rng
        self.tex = {}
        for spec in (HEAD, JAW):
            w, h = board_size_mm(spec)
            m = 5 * self.PPM  # 5 mm white plate margin
            img = make_gridboard(spec).generateImage((int(round(w * self.PPM)) + 2 * m, int(round(h * self.PPM)) + 2 * m),
                                                     marginSize=m, borderBits=1)
            A = np.array([[1.0 / self.PPM, 0, (0.5 - m) / self.PPM], [0, 1.0 / self.PPM, (0.5 - m) / self.PPM], [0, 0, 1]])
            self.tex[spec.name] = (img, A)
        W, H = cfg.width * 2, cfg.height * 2
        bg = rng.normal(140, 40, (H // 16, W // 16)).astype(np.float32)
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:H, 0:W]
        face = ((xx - W / 2) / (0.28 * W)) ** 2 + ((yy - H / 2) / (0.48 * H)) ** 2 < 1
        bg[face] = 0.5 * bg[face] + 0.5 * 175
        self.bg = np.clip(bg, 0, 255).astype(np.uint8)

    def render(self, K, poses):
        """poses: {'head': T_C_H, 'jaw': T_C_J} -> grayscale uint8 image at (width,height)."""
        K2 = K.copy(); K2[:2] *= 2; K2[0, 2] += 0.5; K2[1, 2] += 0.5
        W, H = self.cfg.width * 2, self.cfg.height * 2
        img = self.bg.copy()

        order = sorted(poses.items(), key=lambda kv: -kv[1][2, 3])  # far first
        for name, T in order:
            tex, A = self.tex[name]
            Hm = K2 @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ A
            th, tw = tex.shape[:2]
            c = cv2.perspectiveTransform(np.array([[[0, 0], [tw, 0], [tw, th], [0, th]]], np.float64), Hm)[0]
            x0, y0 = np.floor(c.min(axis=0)).astype(int) - 2
            x1, y1 = np.ceil(c.max(axis=0)).astype(int) + 2
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, W), min(y1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            Hs = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ Hm
            size = (x1 - x0, y1 - y0)
            warped = cv2.warpPerspective(tex, Hs, size, flags=cv2.INTER_LINEAR, borderValue=0)
            mask = cv2.warpPerspective(np.full_like(tex, 255), Hs, size, flags=cv2.INTER_LINEAR, borderValue=0)
            gain = self.rng.uniform(0.75, 0.95)
            a = mask.astype(np.float32) / 255
            sub = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = (sub * (1 - a) + warped.astype(np.float32) * gain * a).astype(np.uint8)
        img = cv2.resize(img, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
        s = self.rng.uniform(0.0, self.cfg.blur_max)
        if s > 0.15:
            img = cv2.GaussianBlur(img, (0, 0), s)
        img = img.astype(np.float32) + self.rng.normal(0, self.cfg.noise_sigma, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def make_arch_mesh(jaw="upper"):
    """Simple tooth-row mesh (ellipsoid 'teeth' along a parabolic arch) in the synthetic S frame."""
    import trimesh
    parts = []
    n = 14
    for k in range(n):
        u = (k - (n - 1) / 2) / ((n - 1) / 2)  # -1..1, 0 = midline
        x = 24.0 * u
        z = -40.0 * u ** 2
        size = 3.5 + 2.5 * abs(u)
        sp = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sp.apply_scale([size, 4.0, size * 0.8])
        if jaw == "upper":
            sp.apply_translation([x, 4.0, z - 1.0])
        else:
            sp.apply_translation([x, -5.5, z - 2.5])
        parts.append(sp)
    return trimesh.util.concatenate(parts)


# Landmarks (S frame) used by the synthetic registration: points on/near the arch
SYNTH_LANDMARKS = [
    ("11_incisal", [-3.0, 0.0, -0.3], "upper"), ("21_incisal", [3.0, 0.0, -0.3], "upper"),
    ("13_cusp", [-14.0, 1.0, -9.0], "upper"), ("23_cusp", [14.0, 1.0, -9.0], "upper"),
    ("16_mb", [-22.0, 2.0, -32.0], "upper"), ("26_mb", [22.0, 2.0, -32.0], "upper"),
    ("33_cusp", [-12.0, -2.0, -10.5], "lower"), ("43_cusp", [12.0, -2.0, -10.5], "lower"),
]


def generate(outdir, cfg: SynthConfig | None = None, log=print):
    """Write video.mov (+ camera.json, upper/lower.stl, ground_truth.npz) into outdir."""
    from .io import camera_dict, write_json
    cfg = cfg or SynthConfig()
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    model = JawModel()
    times, TS, segs = protocol(cfg, model)
    T_S_H, B_S, T_C_S0 = scene()
    X = se3.inv(T_S_H)
    K = camera_K(cfg)
    rend = Renderer(cfg, rng)
    T_C_S = np.array([T_C_S0 @ head_motion(t, cfg.head_motion) for t in times])
    T_C_H = np.array([A @ T_S_H for A in T_C_S])
    T_C_J = np.array([A @ T @ B_S for A, T in zip(T_C_S, TS)])
    W, H = cfg.width, cfg.height
    stored = (H, W) if cfg.rotation_tag else (W, H)
    tmp = outdir / "_stored.mov"
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg needed for synthetic video")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{stored[0]}x{stored[1]}",
           "-r", str(cfg.fps), "-i", "-", "-c:v", cfg.codec, "-crf", str(cfg.crf), "-pix_fmt", "yuv420p"]
    if cfg.codec == "libx265":
        cmd += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    proc = subprocess.Popen(cmd + [str(tmp)], stdin=subprocess.PIPE)
    for k in range(len(times)):
        img = rend.render(K, {"head": T_C_H[k], "jaw": T_C_J[k]})
        if cfg.rotation_tag:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        proc.stdin.write(img.tobytes())
    proc.stdin.close(); proc.wait()
    video = outdir / "video.mov"
    if cfg.rotation_tag:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(tmp), "-c", "copy",
                        str(video)], check=True)
        tmp.unlink()
    else:
        tmp.rename(video)
    write_json(outdir / "camera.json", camera_dict(K, [0] * 5, W, H, "synthetic"))
    make_arch_mesh("upper").export(str(outdir / "upper.stl"))
    make_arch_mesh("lower").export(str(outdir / "lower.stl"))
    np.savez(outdir / "ground_truth.npz", times=times, T_S=TS, T_C_S=T_C_S, T_C_H=T_C_H, T_C_J=T_C_J, X=X, K=K,
             B_S=B_S, condyle_left=model.cL, condyle_right=model.cR, incisal=LOWER_INCISAL)
    write_json(outdir / "segments_gt.json", {"segments": segs})
    log(f"synthetic: {len(times)} frames {W}x{H} @ {cfg.fps} fps -> {video}")
    return {"times": times, "T_S": TS, "T_C_S": T_C_S, "X": X, "K": K, "segments": segs}


def make_landmarks_json(gt, frame_index, t, width, height, noise_px=0.0, rng=None, path=None):
    """Exact (or noisy) registration landmarks for a frame of the synthetic video (in MI)."""
    from .io import write_json
    rng = rng or np.random.default_rng(0)
    k = int(np.argmin(np.abs(gt["times"] - t)))
    T = gt["T_C_S"][k]
    pts = []
    for name, p, jaw in SYNTH_LANDMARKS:
        q = se3.apply(T if jaw == "upper" else gt["T_C_S"][k] @ gt["T_S"][k], np.array(p))
        uv = (gt["K"] @ q)[:2] / q[2] + rng.normal(0, noise_px, 2)
        pts.append({"name": name, "img_px": uv.tolist(), "stl_mm": p, "jaw": jaw})
    plane = [LOWER_INCISAL.tolist(), [22.0, -3.0, -33.0], [-22.0, -3.0, -33.0]]
    lm = {"version": 1, "image": {"file": "reg_frame.png", "frame_index": int(frame_index), "width": width, "height": height},
          "points": pts, "incisal_point_mm": LOWER_INCISAL.tolist(), "occlusal_plane_mm": plane}
    if path:
        write_json(path, lm, indent=1)
    return lm
