"""Version 2 tracking: small individual paper markers stuck on the SKIN (no clutch, no plates).

Three markers on the cranium (HEAD_IDS: nasal dorsum + 2 forehead) and three on the mandible
(JAW_IDS: chin + 2 along the mandibular border). Their mutual geometry is unknown (they are stuck
on freely), so it is estimated from the video itself:

1. every marker is a square of known side -> a single-marker pose (IPPE) gives its 4 corners in 3-D;
2. the 3 markers of a segment form a "constellation" (12 corners); it is initialised in a reference
   frame and refined by bundle adjustment over frames spread through the clips, each marker kept a
   rigid square of the known size;
3. per frame, the constellation pose is solved by PnP on all visible corners.

The result is written in the same raw_tracking.json format as the board tracker, so ``solve``
(registration, T_S(t), parameters) is unchanged. The per-frame fit residual of the JAW constellation
is reported in mm: it is a direct measure of how much the skin deforms (soft-tissue artefact).

Several clips (open, right, left, protrusion) are tracked onto one timeline; a segments.json and a
combined 1080p H.264 preview video for the viewer are written next to the output.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import se3
from .detect import MarkerDetector, sharpness
from .io import VideoReader, adapt_camera, load_camera, write_json

log = logging.getLogger("jawtrack.skin")

HEAD_IDS = (20, 21, 22)
JAW_IDS = (30, 31, 32)
MARKER_MM = 12.0
CLIP_LABELS = {"open": "open_close", "right": "laterotrusion_right", "left": "laterotrusion_left",
               "protrusion": "protrusion", "chew": "chewing", "free": "chewing"}


def square(side):
    """Marker corners TL, TR, BR, BL in the marker frame (x right, y down, z into the paper)."""
    h = side / 2.0
    return np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)


def _proj(P, T, K, dist):
    rv, tv = se3.to_rvec_tvec(T)
    return cv2.projectPoints(P.reshape(-1, 1, 3), rv, tv, K, dist)[0].reshape(-1, 2)


def marker_pose(img_pts, side, K, dist):
    """Single-marker IPPE -> list of (T_C_marker, reproj_px) sorted by error (1 or 2 candidates)."""
    try:
        n, rvs, tvs, errs = cv2.solvePnPGeneric(square(side), img_pts.reshape(-1, 1, 2).astype(np.float64), K, dist,
                                                flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    out = [(se3.from_rvec_tvec(rv, tv), float(np.ravel(e)[0])) for rv, tv, e in zip(rvs, tvs, errs)]
    return sorted(out, key=lambda c: c[1])


def _pnp(obj, img, K, dist, T_prev=None):
    """Constellation pose from >= 8 corners. Tries SQPNP and (if given) the previous pose."""
    cands = []
    o = obj.reshape(-1, 1, 3).astype(np.float64); i = img.reshape(-1, 1, 2).astype(np.float64)
    try:
        ok, rv, tv = cv2.solvePnP(o, i, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            cands.append((rv, tv))
    except cv2.error:
        pass
    if T_prev is not None:
        cands.append(se3.to_rvec_tvec(T_prev))
    best = None
    for rv, tv in cands:
        rv, tv = cv2.solvePnPRefineLM(o, i, K, dist, np.array(rv, float).reshape(3, 1), np.array(tv, float).reshape(3, 1))
        T = se3.from_rvec_tvec(rv, tv)
        if T[2, 3] <= 0:
            continue
        r = _proj(obj, T, K, dist) - img
        e = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
        if best is None or e < best[1]:
            best = (T, e)
    return best


def _kabsch(A, B):
    """Rigid T with B ~ T(A)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return se3.make_T(R, cb - R @ ca)


class Constellation:
    """Three (or more) square markers with fixed mutual pose. ``poses[id]`` = T_seg_marker."""

    def __init__(self, ids, side):
        self.ids = tuple(ids)
        self.side = side
        self.poses = {}

    def corners(self, i):
        return se3.apply(self.poses[i], square(self.side))

    def gather(self, found):
        ids = [i for i in self.ids if i in found and i in self.poses]
        if not ids:
            return None, None, ids
        return np.concatenate([self.corners(i) for i in ids]), np.concatenate([found[i] for i in ids]), ids

    def to_json(self):
        return {"ids": list(self.ids), "marker_mm": self.side,
                "T_seg_marker": {str(i): se3.to_list(T) for i, T in self.poses.items()},
                "centres_mm": {str(i): T[:3, 3].tolist() for i, T in self.poses.items()}}


def build_constellation(dets, ids, side, K, dist, init_frames, ba_frames, name=""):
    """dets: list of {id: (4,2)}; init_frames/ba_frames: indices into dets (all markers visible)."""
    con = Constellation(ids, side)
    if not init_frames:
        raise RuntimeError(f"{name}: no frame shows all markers {ids}; check placement, light and focus")
    # --- initial structure from single-marker poses, averaged over the init frames in the frame of k0
    k0 = init_frames[0]
    def cam_corners(k):
        out = {}
        for i in ids:
            c = marker_pose(dets[k][i], side, K, dist)
            if not c:
                return None
            out[i] = se3.apply(c[0][0], square(side))
        return out
    P0 = cam_corners(k0)
    if P0 is None:
        raise RuntimeError(f"{name}: marker pose failed in the reference frame")
    ref = np.concatenate([P0[i] for i in ids])
    acc = [ref]
    for k in init_frames[1:12]:
        Pk = cam_corners(k)
        if Pk is None:
            continue
        cur = np.concatenate([Pk[i] for i in ids])
        acc.append(se3.apply(_kabsch(cur, ref), cur))
    mean = np.median(np.array(acc), axis=0)
    centre = mean.mean(0)
    for n, i in enumerate(ids):
        con.poses[i] = _kabsch(square(side), mean[4 * n:4 * n + 4] - centre)
    # --- bundle adjustment: marker poses (6 each) + frame poses (6 each); reprojection residuals
    frames = list(ba_frames)
    Ts = []
    keep = []
    T_prev = None
    for k in frames:
        obj, img, _ = con.gather(dets[k])
        r = _pnp(obj, img, K, dist, T_prev)
        if r is None:
            continue
        Ts.append(r[0]); keep.append(k); T_prev = r[0]
    frames = keep
    if len(frames) < 4:
        log.warning("%s: too few frames for bundle adjustment; using the single-frame constellation", name)
        return con, {"ba_frames": len(frames), "rms_px_before": None, "rms_px_after": None}
    sq = square(side)
    nm = len(ids)

    def pack(mposes, fposes):
        v = []
        for T in list(mposes) + list(fposes):
            rv, tv = se3.to_rvec_tvec(T)
            v += [*np.ravel(rv), *np.ravel(tv)]
        return np.array(v, float)

    x0 = pack([con.poses[i] for i in ids], Ts)
    obs = np.array([np.concatenate([dets[k][i] for i in ids]) for k in frames])  # (F, 4*nm, 2)
    prior_w = 0.05  # soft gauge: keep the constellation near its initial pose

    def resid(x):
        m = x[:6 * nm].reshape(nm, 6)
        P = np.concatenate([se3.apply(se3.from_rvec_tvec(mm[:3], mm[3:]), sq) for mm in m])
        res = []
        for f in range(len(frames)):
            p = x[6 * nm + 6 * f:6 * nm + 6 * f + 6]
            uv = cv2.projectPoints(P.reshape(-1, 1, 3), p[:3], p[3:], K, dist)[0].reshape(-1, 2)
            res.append((uv - obs[f]).ravel())
        res.append(prior_w * (x[:6 * nm] - x0[:6 * nm]))
        return np.concatenate(res)

    n_obs = len(frames) * nm * 8
    rms0 = float(np.sqrt(np.mean(resid(x0)[:n_obs] ** 2) * 2))
    from scipy.sparse import lil_matrix
    S = lil_matrix((n_obs + 6 * nm, len(x0)), dtype=int)
    for f in range(len(frames)):
        r0 = f * nm * 8
        S[r0:r0 + nm * 8, 6 * nm + 6 * f:6 * nm + 6 * f + 6] = 1
        for j in range(nm):
            S[r0 + 8 * j:r0 + 8 * j + 8, 6 * j:6 * j + 6] = 1
    for j in range(6 * nm):
        S[n_obs + j, j] = 1
    sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=60)
    rms1 = float(np.sqrt(np.mean(resid(sol.x)[:n_obs] ** 2) * 2))
    m = sol.x[:6 * nm].reshape(nm, 6)
    for n, i in enumerate(ids):
        con.poses[i] = se3.from_rvec_tvec(m[n, :3], m[n, 3:])
    log.info("%s constellation: %d markers, BA over %d frames, reprojection RMS %.2f -> %.2f px", name, nm,
             len(frames), rms0, rms1)
    return con, {"ba_frames": len(frames), "rms_px_before": rms0, "rms_px_after": rms1}


def _entry(T, n, rms, deform_mm):
    return {"T": se3.to_list(T), "n_markers": int(n), "reproj_px": round(float(rms), 4), "ambiguity": 99.0,
            "deform_mm": round(float(deform_mm), 4)}


def _track(con, dets, K, dist):
    """Per-frame constellation pose -> {k: (T, n_markers, rms_px, deform_mm)}."""
    out = {}
    T_prev = None
    f = 0.5 * (K[0, 0] + K[1, 1])
    for k, found in enumerate(dets):
        obj, img, ids = con.gather(found)
        if len(ids) < 2:
            T_prev = None if len(ids) == 0 else T_prev
            continue
        r = _pnp(obj, img, K, dist, T_prev)
        if r is None:
            continue
        T, rms = r
        out[k] = (T, len(ids), rms, rms * T[2, 3] / f)
        T_prev = T
    return out


def _make_preview(videos, out, fps):
    n = len(videos)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for v in videos:
        cmd += ["-i", str(v)]
    filt = "".join(f"[{i}:v]scale=-2:1080,fps={fps:.5f},setsar=1[v{i}];" for i in range(n))
    filt += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[o]"
    cmd += ["-filter_complex", filt, "-map", "[o]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:  # preview is a convenience only
        log.warning("could not write the combined preview video: %s", e)
        return False


def track_skin(clips, camera, out, marker_mm=MARKER_MM, print_scale=1.0, every=1, max_side=1920, mi_hold_s=1.0,
               preview=True):
    """clips: list of (label, path), label in CLIP_LABELS (or any text). Writes raw_tracking.json
    (+ reg_frame.png, segments.json, combined_1080p.mp4) next to ``out``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = marker_mm * print_scale
    cam0 = load_camera(camera)
    dets, times, idx, clip_of, clip_info = [], [], [], [], []
    K = dist = None
    W = H = None
    fps = None
    t_off = 0.0
    reg = None  # (sharpness, global k, image)
    n_glob = 0
    for ci, (label, path) in enumerate(clips):
        vr = VideoReader(path)
        if W is None:
            W, H, fps = vr.width, vr.height, float(vr.fps)
            cam = adapt_camera(cam0, W, H)
            K, dist = np.asarray(cam["K"], float), np.asarray(cam["dist"], float)
        elif (vr.width, vr.height) != (W, H):
            raise RuntimeError(f"{path}: {vr.width}x{vr.height} differs from the first clip {W}x{H}; "
                               "record all clips with the same camera settings")
        det = MarkerDetector(max_side=max_side, n_expected=6)
        n_local = 0
        last_t = 0.0
        for i, t, fr in vr.frames(every=every):
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            found = det.detect(gray)
            found = {m: c for m, c in found.items() if m in HEAD_IDS or m in JAW_IDS}
            k = len(dets)
            dets.append(found); times.append(t_off + t); idx.append(n_glob + i); clip_of.append(ci)
            if ci == 0 and t <= max(mi_hold_s, 0.5) + 1.0 and len(found) == 6:
                pts = np.concatenate(list(found.values()))
                s = sharpness(gray, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
                if reg is None or s > reg[0]:
                    reg = (s, k, fr.copy())
            n_local += 1; last_t = t
        dur = last_t + every / float(vr.fps)
        n_all = sum(1 for k in range(len(dets) - n_local, len(dets)) if len(dets[k]) == 6)
        log.info("clip %d '%s' %s: %d frames, %.1f s, all 6 markers in %.0f%%", ci + 1, label, Path(path).name,
                 n_local, dur, 100.0 * n_all / max(n_local, 1))
        clip_info.append({"label": label, "file": str(path), "t0": t_off, "t1": t_off + dur, "frames": n_local,
                          "all_markers_fraction": n_all / max(n_local, 1)})
        t_off += dur
        n_glob += int(round(dur * float(vr.fps)))
    times = np.array(times)
    clip_of = np.array(clip_of)

    def full(ids):
        return [k for k, f in enumerate(dets) if all(i in f for i in ids)]

    cons, stats, tracks = {}, {}, {}
    for name, ids in (("head", HEAD_IDS), ("jaw", JAW_IDS)):
        fk = full(ids)
        if name == "jaw":
            # shape of the jaw constellation = its shape at MI: use only the MI hold at the start of each clip
            hold = [k for k in fk if times[k] - clip_info[clip_of[k]]["t0"] <= mi_hold_s]
            pool = hold if len(hold) >= 8 else fk
        else:
            pool = fk
        init = [k for k in fk if clip_of[k] == 0][:12] or fk[:12]
        sel = [pool[j] for j in np.unique(np.linspace(0, len(pool) - 1, min(len(pool), 48)).astype(int))] if pool else []
        cons[name], stats[name] = build_constellation(dets, ids, side, K, dist, init, sel, name)
        tracks[name] = _track(cons[name], dets, K, dist)

    frames = []
    for k in range(len(dets)):
        frames.append({"i": int(idx[k]), "t": round(float(times[k]), 5), "clip": int(clip_of[k]),
                       "head": _entry(*tracks["head"][k]) if k in tracks["head"] else None,
                       "jaw": _entry(*tracks["jaw"][k]) if k in tracks["jaw"] else None})
    both = [k for k in range(len(dets)) if k in tracks["head"] and k in tracks["jaw"]]
    if not both:
        raise RuntimeError("no frame with both the head and the jaw markers tracked")

    # --- MI hold per clip: jaw constellation origin still (relative to the head) at the clip start
    segs = []
    M = {k: se3.inv(tracks["head"][k][0]) @ tracks["jaw"][k][0] for k in both}
    for ci, c in enumerate(clip_info):
        ks = [k for k in both if clip_of[k] == ci]
        t_move = c["t0"] + mi_hold_s
        if ks:
            ref = np.median([M[k][:3, 3] for k in ks if times[k] - c["t0"] <= 0.5] or [M[ks[0]][:3, 3]], axis=0)
            d = np.array([np.linalg.norm(M[k][:3, 3] - ref) for k in ks])
            moving = np.where(d > 1.0)[0]
            if len(moving):
                t_move = max(c["t0"] + 0.3, times[ks[moving[0]]] - 0.15)
            else:
                t_move = c["t1"]
        c["mi_hold_end"] = float(t_move)
        segs.append({"label": "MI", "t0": float(c["t0"]), "t1": float(t_move)})
        if t_move < c["t1"]:
            segs.append({"label": CLIP_LABELS.get(c["label"], c["label"]), "t0": float(t_move), "t1": float(c["t1"])})

    def dstat(name, ks):
        v = np.array([tracks[name][k][3] for k in ks if k in tracks[name]])
        return {"median_mm": float(np.median(v)), "p95_mm": float(np.percentile(v, 95)), "max_mm": float(v.max())} \
            if len(v) else None

    deform = {"head": dstat("head", range(len(dets))), "jaw": dstat("jaw", range(len(dets))),
              "jaw_per_clip": {c["label"]: dstat("jaw", [k for k in range(len(dets)) if clip_of[k] == ci])
                               for ci, c in enumerate(clip_info)},
              "jaw_in_MI_holds": dstat("jaw", [k for k in range(len(dets))
                                               if times[k] <= clip_info[clip_of[k]]["mi_hold_end"]]),
              "note": "RMS misfit of the rigid 3-marker constellation, in mm at the face. Head: noise floor. "
                      "Jaw minus head/MI level = skin deformation (soft-tissue artefact indicator; a rigid "
                      "slide of all three markers together is NOT visible here)."}

    raw = {"version": 1, "mode": "skin_markers", "video": str(clips[0][1]), "fps": fps / max(1, every),
           "video_fps": fps, "every": int(every), "width": W, "height": H,
           "camera": {"K": cam["K"], "dist": cam["dist"], "source": cam.get("source", "unknown")},
           "print_scale": print_scale, "marker_mm": marker_mm, "clips": clip_info,
           "constellations": {n: {**c.to_json(), **stats[n]} for n, c in cons.items()},
           "skin_deformation": deform,
           "first_mi_hold_estimate": [float(clip_info[0]["t0"]), float(clip_info[0]["mi_hold_end"])],
           "frames": frames}
    if reg is not None and reg[1] in tracks["head"]:
        cv2.imwrite(str(out.parent / "reg_frame.png"), reg[2])
        k = reg[1]
        raw["registration_frame"] = {"source": "video", "file": "reg_frame.png", "frame_index": int(idx[k]),
                                     "t": float(times[k]), "width": W, "height": H, "sharpness": float(reg[0]),
                                     "head": frames[k]["head"], "jaw": frames[k]["jaw"]}
        log.info("registration frame: clip 1 frame %d (t=%.2f s) -> %s", idx[k], times[k], out.parent / "reg_frame.png")
    else:
        raw["registration_frame"] = None
        log.warning("no sharp frame with all 6 markers at the start of clip 1: no reg_frame.png")
    write_json(out, raw)
    write_json(out.parent / "segments.json", {"segments": segs}, indent=1)
    if preview:
        if _make_preview([p for _, p in clips], out.parent / "combined_1080p.mp4", fps):
            raw["preview_video"] = "combined_1080p.mp4"
    log.info("tracked both segments in %.1f%% of %d frames; skin misfit jaw median %.2f mm (p95 %.2f), head %.2f mm",
             100.0 * len(both) / len(dets), len(dets), deform["jaw"]["median_mm"], deform["jaw"]["p95_mm"],
             deform["head"]["median_mm"])
    return raw
