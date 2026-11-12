"""Camera calibration: ChArUco video (DICT_5X5_100, 7x5, 25/18 mm) or an iPhone preset."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .boards import make_charuco
from .detect import sharpness
from .io import VideoReader, camera_dict

log = logging.getLogger("jawtrack.calib")

PRESETS = {
    # ~26 mm (35 mm-equivalent) main camera; fx = width * 26/36 (36 mm = full-frame width).
    # Principal point at the image centre, zero distortion (iPhone video is already
    # distortion-corrected in-camera, but residual distortion is NOT modelled). Approximate only.
    "iphone_main_4k": (3840, 2160),
    "iphone_main_1080p": (1920, 1080),
}


def preset_camera(name: str, width: int | None = None, height: int | None = None) -> dict:
    """Approximate intrinsics for the iPhone main camera (see PRESETS comment)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name}; choose from {sorted(PRESETS)}")
    w, h = PRESETS[name]
    if width and height:
        w, h = width, height
    long_side = max(w, h)
    f = long_side * 26.0 / 36.0
    K = [[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1]]
    return camera_dict(K, [0] * 5, w, h, "preset", preset=name,
                       assumptions="fx=fy=long_side*26/36 (26 mm-eq. main camera), principal point at centre, "
                                   "zero distortion; expect a few % scale error -> calibrate with ChArUco if possible")


def _charuco_detector(board):
    cp = cv2.aruco.CharucoParameters()
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cp, dp)


def detect_charuco(det, gray):
    """-> (charuco_corners (N,1,2), charuco_ids (N,1)) or (None, None)."""
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < 6:
        return None, None
    return cc, ci


def charuco_view(det, board, gray, min_corners=8):
    """Detect a ChArUco view -> (obj (N,3), img (N,2), sharpness) or None."""
    cc, ci = detect_charuco(det, gray)
    if cc is None or len(ci) < min_corners:
        return None
    obj, img = board.matchImagePoints(cc, ci)
    if obj is None or len(obj) < min_corners:
        return None
    return (obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32), sharpness(gray))


def calibrate_from_frames(grays, board, image_size, min_corners=8, max_views=40):
    """Calibrate from a list of grayscale frames. Returns (camera dict, info)."""
    det = _charuco_detector(board)
    views = [v for v in (charuco_view(det, board, g, min_corners) for g in grays) if v is not None]
    return calibrate_from_views(views, image_size, max_views)


def calibrate_from_views(views, image_size, max_views=40):
    """Calibrate from detected views [(obj, img, sharpness)]."""
    if len(views) < 5:
        raise RuntimeError(f"only {len(views)} usable ChArUco views (need >= 5): move the board more slowly, "
                           "fill more of the frame, check focus / lighting")
    views = _select_distributed(views, image_size, max_views)
    flags = cv2.CALIB_FIX_K3
    w, h = image_size

    def run(vs):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([v[0] for v in vs], [v[1] for v in vs], (w, h),
                                                         None, None, flags=flags)
        per = []
        for (o, i, _), rv, tv in zip(vs, rvecs, tvecs):
            p, _ = cv2.projectPoints(o, rv, tv, K, dist)
            per.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
        return rms, K, dist, np.array(per)

    rms, K, dist, per = run(views)
    keep = per < max(3 * np.median(per), 0.5)
    if keep.sum() >= 5 and keep.sum() < len(views):
        log.info("dropping %d outlier views", int((~keep).sum()))
        views = [v for v, k in zip(views, keep) if k]
        rms, K, dist, per = run(views)
    info = {"n_views": len(views), "rms_px": float(rms), "per_view_rms_px": per.tolist()}
    log.info("calibration: %d views, RMS %.3f px, fx=%.1f fy=%.1f cx=%.1f cy=%.1f dist=%s", len(views), rms,
             K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.round(dist.ravel(), 4).tolist())
    if rms > 1.0:
        log.warning("calibration RMS %.2f px is high (expect < 0.6 px): blur, rolling shutter or a bent board?", rms)
    return camera_dict(K, dist.ravel(), w, h, "charuco", rms_px=float(rms), n_views=len(views)), info


def _select_distributed(views, image_size, max_views):
    """Pick sharp views spread over image position and apparent board size (3x3 x 3 bins)."""
    w, h = image_size
    bins = {}
    for v in views:
        c = v[1].mean(axis=0)
        area = cv2.contourArea(cv2.convexHull(v[1]))
        key = (int(3 * c[0] / w), int(3 * c[1] / h), int(np.clip(np.log2(max(area, 1) / (w * h)) + 6, 0, 2)))
        bins.setdefault(key, []).append(v)
    picked = []
    per_bin = max(1, int(np.ceil(max_views / max(1, len(bins)))))
    for k in sorted(bins):
        picked += sorted(bins[k], key=lambda v: -v[2])[:per_bin]
    picked = sorted(picked, key=lambda v: -v[2])[:max_views]
    return picked


def calibrate_video(path, print_scale=1.0, max_candidates=300, blur_quantile=0.3):
    """Calibrate from a ChArUco video: sample ~max_candidates frames, detect the board, drop the
    blurriest views (Laplacian variance below the ``blur_quantile`` quantile), select well
    distributed sharp views, calibrate (k1,k2,p1,p2; k3 fixed), reject outlier views, recalibrate."""
    vr = VideoReader(path)
    every = max(1, int(vr.n_frames / max_candidates)) if vr.n_frames else 5
    board = make_charuco(print_scale)
    det = _charuco_detector(board)
    views, n = [], 0
    for i, t, fr in vr.frames(every=every):
        n += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        v = charuco_view(det, board, g)
        if v is not None:
            views.append(v)
    if not views:
        raise RuntimeError(f"no ChArUco board found in {path} (DICT_5X5_100 7x5? print scale?)")
    thr = np.quantile([v[2] for v in views], blur_quantile)
    views = [v for v in views if v[2] >= thr]
    log.info("calibration video: %d frames sampled, %d sharp views with the board", n, len(views))
    cam, info = calibrate_from_views(views, (vr.width, vr.height))
    cam["video"] = str(path)
    cam["print_scale"] = print_scale
    return cam, info
