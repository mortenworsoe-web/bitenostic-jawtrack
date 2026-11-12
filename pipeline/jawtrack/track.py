"""Stage 1 (track): video -> raw_tracking.json (SPEC 5.1) + reg_frame.png."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import se3
from .boards import HEAD, JAW, board_object_points
from .detect import MarkerDetector, sharpness
from .io import VideoReader, adapt_camera, load_camera, read_image, write_json
from .pose import gather_points, ippe_candidates, refine_lm, viterbi_select

log = logging.getLogger("jawtrack.track")


def _pose_entry(T, n_markers, rms, amb):
    return {"T": se3.to_list(T), "n_markers": int(n_markers), "reproj_px": round(float(rms), 4),
            "ambiguity": round(float(amb), 3)}


def _candidates_for(found, objpts, K, dist):
    obj, img, ids = gather_points(found, objpts)
    if obj is None:
        return None
    cands = ippe_candidates(obj, img, K, dist)
    if not cands:
        return None
    amb = cands[1][1] / max(cands[0][1], 1e-6) if len(cands) > 1 else 99.0
    return {"obj": obj, "img": img, "n": len(ids), "cands": [(T, e, len(obj)) for T, e in cands], "amb": amb}


def select_and_refine(per_frame, key, frame_idx, K, dist, unary_extra=None):
    """Viterbi candidate choice + LM refinement for one board. per_frame[k][key] may be None."""
    sel = [k for k in range(len(per_frame)) if per_frame[k][key] is not None]
    if not sel:
        return {}
    seq = [per_frame[k][key]["cands"] for k in sel]
    extra = None
    if unary_extra is not None:
        extra = lambda f, T: unary_extra(sel[f], T)  # noqa: E731
    path = viterbi_select(seq, [frame_idx[k] for k in sel], unary_extra=extra)
    out, n_flip = {}, 0
    for k, c in zip(sel, path):
        d = per_frame[k][key]
        n_flip += int(c != 0)
        T, rms = refine_lm(d["obj"], d["img"], K, dist, d["cands"][c][0])
        out[k] = (T, d["n"], rms, d["amb"])
    log.info("%s: %d frames posed, %d chose the 2nd IPPE solution (flip correction by temporal consistency)",
             key, len(sel), n_flip)
    return out


def jaw_prior_fn(per_frame, head_sel, min_amb=2.0, sigma_deg=20.0):
    """Prior on the jaw-board orientation relative to the head board, learned from frames where the
    jaw pose is unambiguous (2nd IPPE solution clearly worse)."""
    rels = []
    for k, d in enumerate(per_frame):
        if d["jaw"] is not None and k in head_sel and d["jaw"]["amb"] >= min_amb:
            rels.append(se3.inv(head_sel[k][0]) @ d["jaw"]["cands"][0][0])
    if len(rels) < 5:
        log.info("jaw orientation prior: too few unambiguous frames (%d) -> not used", len(rels))
        return None
    ref, _ = se3.robust_average(rels)
    Rref = ref[:3, :3]
    log.info("jaw orientation prior from %d unambiguous frames", len(rels))

    def f(k, T):
        if k not in head_sel:
            return 0.0
        Rrel = head_sel[k][0][:3, :3].T @ T[:3, :3]
        return 0.5 * (se3.rot_dist_deg(Rref, Rrel) / sigma_deg) ** 2
    return f


def find_first_mi_hold(times, Ms, max_mm=0.6, max_deg=1.0, ref_s=0.5, min_s=0.5):
    """First stable interval (t0, t1) of the relative pose M at the start of the recording.
    Stability is measured on the jaw-board centre (in head coordinates) vs. the median of the first
    ``ref_s`` seconds. Returns None if nothing stable is found."""
    if len(Ms) < 3:
        return None
    times = np.asarray(times)
    ref_mask = times <= times[0] + ref_s
    ref, _ = se3.robust_average(np.asarray(Ms)[ref_mask])
    c = np.array([16.5, 16.5, 0.0])
    bad = []
    for M in Ms:
        d = np.linalg.norm(se3.apply(M, c) - se3.apply(ref, c))
        a = se3.rot_dist_deg(M[:3, :3], ref[:3, :3])
        bad.append(d > max_mm or a > max_deg)
    bad = np.array(bad)
    # end = first index where 3 consecutive frames are unstable
    end = len(bad)
    for k in range(len(bad) - 2):
        if bad[k] and bad[k + 1] and bad[k + 2]:
            end = k
            break
    if end == 0 or times[end - 1] - times[0] < min_s:
        return None
    return float(times[0]), float(times[end - 1])


def pose_still(image_path, cam, ref_M=None, print_scale=1.0):
    """Head (and jaw) pose of a separate registration still. Returns dict for raw_tracking."""
    img = read_image(image_path)
    h, w = img.shape[:2]
    if (cam.get("width"), cam.get("height")) not in ((w, h), (None, None)):
        log.warning("registration still is %dx%d but the camera was calibrated at %sx%s: intrinsics are "
                    "adapted (scaled/transposed). Photo mode may use a different crop/lens -> prefer a video frame.",
                    w, h, cam.get("width"), cam.get("height"))
    c = adapt_camera(cam, w, h)
    K, dist = np.asarray(c["K"]), np.asarray(c["dist"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = MarkerDetector(max_side=max(w, h))
    found = det.detect(gray)
    res = {"source": "still", "file": str(image_path), "frame_index": None, "width": w, "height": h,
           "K": c["K"], "dist": c["dist"], "head": None, "jaw": None}
    hc = _candidates_for(found, board_object_points(HEAD, print_scale), K, dist)
    jc = _candidates_for(found, board_object_points(JAW, print_scale), K, dist)
    if hc is None:
        raise RuntimeError(f"HEAD board not found in registration still {image_path}")
    best = None
    for Th, eh, _ in hc["cands"]:
        for Tj, ej, _ in (jc["cands"] if jc else [(None, 0, 0)]):
            cost = eh ** 2 + ej ** 2
            if ref_M is not None and Tj is not None:
                cost += (se3.rot_dist_deg((se3.inv(Th) @ Tj)[:3, :3], ref_M[:3, :3]) / 5.0) ** 2
            if best is None or cost < best[0]:
                best = (cost, Th, Tj)
    Th, rms = refine_lm(hc["obj"], hc["img"], K, dist, best[1])
    res["head"] = _pose_entry(Th, hc["n"], rms, hc["amb"])
    if jc is not None and best[2] is not None:
        Tj, rmsj = refine_lm(jc["obj"], jc["img"], K, dist, best[2])
        res["jaw"] = _pose_entry(Tj, jc["n"], rmsj, jc["amb"])
    return res


VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def detect_sequence(vr, K, dist, print_scale=1.0, every=1, max_frames=None, max_side=1920, refine="subpix",
                    reg_window_s=8.0, label="video"):
    """Detection + IPPE candidates for every processed frame of a VideoReader.
    Also keeps, per 0.5 s bin within the first ``reg_window_s`` seconds, the sharpest frame with
    both boards visible (registration-frame candidates)."""
    obj_h, obj_j = board_object_points(HEAD, print_scale), board_object_points(JAW, print_scale)
    det = MarkerDetector(max_side=max_side, refine=refine)
    per_frame, idx, times = [], [], []
    reg_cands = {}  # 0.5 s bin -> (sharpness, frame_index, t, bgr)
    t_start = time.time()
    last_log = t_start
    total = (vr.n_frames // max(1, every)) if vr.n_frames else 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    for n, (i, t, fr) in enumerate(vr.frames(every=every, max_frames=max_frames)):
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        found = det.detect(gray)
        d = {"head": _candidates_for(found, obj_h, K, dist), "jaw": _candidates_for(found, obj_j, K, dist)}
        per_frame.append(d)
        idx.append(i)
        times.append(t)
        if t <= reg_window_s and d["head"] is not None and d["jaw"] is not None and \
                d["head"]["n"] >= 3 and d["jaw"]["n"] >= 2:
            pts = np.concatenate([d["head"]["img"], d["jaw"]["img"]])
            roi = (pts[:, 0].min() - 80, pts[:, 1].min() - 80, pts[:, 0].max() + 80, pts[:, 1].max() + 200)
            s = sharpness(gray, roi) * (d["head"]["n"] + d["jaw"]["n"]) / 10.0
            b = int(t / 0.5)
            if b not in reg_cands or s > reg_cands[b][0]:
                reg_cands[b] = (s, i, t, fr.copy())
        now = time.time()
        if now - last_log > 5 or n == 0:
            rate = (n + 1) / max(now - t_start, 1e-6)
            eta = (total - n - 1) / rate if total else float("nan")
            log.info("%s frame %d (t=%.2fs) %d/%s processed, %.1f fps, ETA %.0f s, markers H%s J%s", label, i, t,
                     n + 1, total or "?", rate, eta, d["head"]["n"] if d["head"] else 0,
                     d["jaw"]["n"] if d["jaw"] else 0)
            last_log = now
    elapsed = time.time() - t_start
    log.info("%s detection done: %d frames in %.1f s (%.2f fps); ROI passes %d, full-frame passes %d", label,
             len(per_frame), elapsed, len(per_frame) / max(elapsed, 1e-6), det.stats["roi"], det.stats["full"])
    if not per_frame:
        raise RuntimeError(f"no frames decoded from {vr.path}")
    return per_frame, idx, times, reg_cands, elapsed


def solve_sequence(per_frame, idx, K, dist):
    """Viterbi + LM for head, then jaw (with the relative-orientation prior)."""
    head_sel = select_and_refine(per_frame, "head", idx, K, dist)
    prior = jaw_prior_fn(per_frame, head_sel)
    jaw_sel = select_and_refine(per_frame, "jaw", idx, K, dist, unary_extra=prior)
    return head_sel, jaw_sel


def pose_reg_video(path, cam, out_png, print_scale=1.0, max_side=1920, refine="subpix", every=1):
    """Registration from a SHORT VIDEO (retractor, MI, both plates): sharpest frame with both boards
    (all markers preferred), poses chosen with the same temporal disambiguation as the main video."""
    vr = VideoReader(path)
    c = adapt_camera(cam, vr.width, vr.height)
    K, dist = np.asarray(c["K"], float), np.asarray(c["dist"], float)
    per_frame, idx, times, reg_cands, _ = detect_sequence(vr, K, dist, print_scale, every, None, max_side, refine,
                                                          reg_window_s=1e9, label="reg-video")
    head_sel, jaw_sel = solve_sequence(per_frame, idx, K, dist)
    pool = [r for r in reg_cands.values() if idx.index(r[1]) in head_sel]
    if not pool:
        raise RuntimeError(f"registration video {path}: no frame with HEAD (>=3 markers) and JAW (>=2) visible")
    s, i, t, img = max(pool, key=lambda r: r[0])
    k = idx.index(i)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    log.info("registration video %s: frame %d (t=%.2fs) -> %s", path, i, t, out_png)
    return {"source": "reg_video", "video": str(path), "file": Path(out_png).name, "frame_index": int(i),
            "t": float(t), "width": vr.width, "height": vr.height, "K": c["K"], "dist": c["dist"], "sharpness": s,
            "head": _pose_entry(*head_sel[k]), "jaw": _pose_entry(*jaw_sel[k]) if k in jaw_sel else None}


def track_video(video, camera, out, print_scale=1.0, every=1, max_frames=None, reg_image=None,
                max_side=1920, refine="subpix", reg_window_s=8.0, force_ffmpeg=False):
    """Run stage 1. Writes ``out`` (raw_tracking.json) and reg_frame.png next to it.

    ``reg_image`` may be a still photo OR a short registration video (by extension)."""
    out = Path(out)
    vr = VideoReader(video, force_ffmpeg=force_ffmpeg)
    cam0 = load_camera(camera)
    cam = adapt_camera(cam0, vr.width, vr.height)
    K, dist = np.asarray(cam["K"], float), np.asarray(cam["dist"], float)
    per_frame, idx, times, reg_cands, elapsed = detect_sequence(vr, K, dist, print_scale, every, max_frames,
                                                                max_side, refine, reg_window_s)

    head_sel, jaw_sel = solve_sequence(per_frame, idx, K, dist)

    frames = []
    for k, (i, t) in enumerate(zip(idx, times)):
        fh = _pose_entry(*head_sel[k]) if k in head_sel else None
        fj = _pose_entry(*jaw_sel[k]) if k in jaw_sel else None
        frames.append({"i": int(i), "t": round(float(t), 5), "head": fh, "jaw": fj})
    nh, nj = len(head_sel), len(jaw_sel)
    nb = sum(1 for k in head_sel if k in jaw_sel)
    log.info("tracked: head %.1f%%, jaw %.1f%%, both %.1f%% of %d frames", 100 * nh / len(frames),
             100 * nj / len(frames), 100 * nb / len(frames), len(frames))

    raw = {"version": 1, "video": str(video), "fps": float(vr.fps) / max(1, every), "video_fps": float(vr.fps),
           "every": int(every), "width": vr.width, "height": vr.height,
           "rotation_applied_cw_deg": vr.rotation_cw, "decoder": vr.backend,
           "camera": {"K": cam["K"], "dist": cam["dist"], "source": cam.get("source", "unknown")},
           "print_scale": print_scale,
           "board_convention": "OpenCV GridBoard: origin top-left corner of top-left marker, x right, y down, z into board",
           "processing": {"seconds": round(elapsed, 2), "fps": round(len(per_frame) / max(elapsed, 1e-6), 2),
                          "detector_max_side": max_side, "refine": refine},
           "frames": frames}

    # ---- registration frame
    Ms, ts, ks = [], [], []
    for k in range(len(frames)):
        if k in head_sel and k in jaw_sel:
            Ms.append(se3.inv(head_sel[k][0]) @ jaw_sel[k][0])
            ts.append(times[k])
            ks.append(k)
    hold = find_first_mi_hold(ts, Ms) if Ms else None
    raw["first_mi_hold_estimate"] = list(hold) if hold else None
    reg_png = out.parent / "reg_frame.png"
    if reg_image and Path(reg_image).suffix.lower() in VIDEO_EXT:
        raw["registration_frame"] = pose_reg_video(reg_image, cam0, reg_png, print_scale, max_side, refine)
    elif reg_image:
        ref_M = se3.robust_average([M for M, t in zip(Ms, ts) if hold and hold[0] <= t <= hold[1]])[0] \
            if hold else None
        raw["registration_frame"] = pose_still(reg_image, cam, ref_M, print_scale)
        log.info("registration still %s posed (head reproj %.2f px)", reg_image,
                 raw["registration_frame"]["head"]["reproj_px"])
    else:
        pool = [c for c in reg_cands.values() if hold is None or hold[0] <= c[2] <= hold[1]]
        if not pool:
            pool = list(reg_cands.values())
            if pool:
                log.warning("no sharp both-boards frame inside the detected first MI hold; using the sharpest "
                            "early frame instead - check that the teeth are in MI on reg_frame.png")
        if pool:
            s, i, t, img = max(pool, key=lambda c: c[0])
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(reg_png), img)
            k = idx.index(i)
            raw["registration_frame"] = {"source": "video", "file": reg_png.name, "frame_index": int(i),
                                         "t": float(t), "width": vr.width, "height": vr.height,
                                         "sharpness": s, "head": frames[k]["head"], "jaw": frames[k]["jaw"]}
            log.info("registration frame: index %d (t=%.2fs, sharpness %.0f) -> %s", i, t, s, reg_png)
        else:
            raw["registration_frame"] = None
            log.warning("no frame with both boards in the first %.0f s: no reg_frame.png written", reg_window_s)
    write_json(out, raw)
    return raw
