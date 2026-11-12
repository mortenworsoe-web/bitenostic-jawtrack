"""ArUco marker detection tuned for 1080p/4K phone video.

Strategy (keeps 4K affordable on 2 cores):
1. Detect on a DOWNSCALED image: either the whole frame (long side <= ``max_side``) or, when the
   previous frame had detections, a region of interest around them, scaled so that markers are
   ~``target_marker_px`` pixels wide.
2. Map the corners back to full resolution and refine them there (``refine='subpix'``:
   cv2.cornerSubPix with a window smaller than the 3 mm marker gap), so localisation accuracy comes
   from the full-resolution image.
3. If the ROI pass finds fewer markers than expected, fall back to a full-frame pass.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boards import aruco_dictionary

VALID_IDS = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 20, 21, 22, 30, 31, 32}


def make_detector(refine: str = "subpix"):
    """ArucoDetector with parameters for small markers in phone video."""
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 5
    p.adaptiveThreshWinSizeMax = 35
    p.adaptiveThreshWinSizeStep = 10
    p.minMarkerPerimeterRate = 0.03  # 0.01 was 7x slower (background clutter candidates)
    p.polygonalApproxAccuracyRate = 0.05
    p.errorCorrectionRate = 0.6
    if refine == "apriltag":
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif refine == "contour":
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    else:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 3
        p.cornerRefinementMaxIterations = 50
        p.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(aruco_dictionary(), p)


def sharpness(gray: np.ndarray, roi=None) -> float:
    """Variance of the Laplacian (higher = sharper), optionally inside roi=(x0,y0,x1,y1)."""
    if roi is not None:
        x0, y0, x1, y1 = [int(v) for v in roi]
        gray = gray[max(0, y0):y1, max(0, x0):x1]
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class MarkerDetector:
    """Stateful detector (remembers the last ROI for the next frame)."""

    def __init__(self, max_side: int = 1920, target_marker_px: float = 48.0, refine: str = "subpix",
                 n_expected: int = 10):
        self.max_side = max_side
        self.target = target_marker_px
        self.refine = refine
        self.det = make_detector(refine)
        self.n_expected = n_expected
        self.prev = None  # (roi, marker side px, n)
        self.stats = {"roi": 0, "full": 0}

    def _detect_scaled(self, gray, x0, y0, scale):
        img = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        corners, ids, _ = self.det.detectMarkers(img)
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i not in VALID_IDS:
                continue
            # pixel-centre convention: x_full = (x_small + 0.5)/scale - 0.5
            c = (c.reshape(4, 2).astype(np.float64) + 0.5) / scale - 0.5
            c[:, 0] += x0
            c[:, 1] += y0
            out[i] = c
        return out

    def _refine_full(self, gray, found):
        if self.refine != "subpix" or not found:
            return found
        ids = list(found)
        pts = np.concatenate([found[i] for i in ids]).astype(np.float32).reshape(-1, 1, 2)
        side = np.median([np.linalg.norm(found[i][0] - found[i][1]) for i in ids])
        # gap = 0.2 * marker side; keep the half-window below ~1/2 gap so neighbours do not interfere
        w = int(np.clip(side * 0.08, 2, 10))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.005)
        ref = cv2.cornerSubPix(gray, pts.copy(), (w, w), (-1, -1), crit).reshape(-1, 4, 2)
        out = {}
        for k, i in enumerate(ids):
            r = ref[k].astype(np.float64)
            # reject refinement jumps (corner snapped to something else)
            if np.max(np.linalg.norm(r - found[i], axis=1)) > max(2.0, 0.6 * w / 1.0 + 1.0):
                r = found[i]
            out[i] = r
        return out

    def detect(self, gray: np.ndarray) -> dict:
        """Detect markers in a full-resolution grayscale frame -> {id: (4,2) corners, full-res px}."""
        H, W = gray.shape[:2]
        found = {}
        if self.prev is not None:
            (x0, y0, x1, y1), side, nprev = self.prev
            scale = float(np.clip(self.target / max(side, 1e-6), 0.1, 1.0))
            found = self._detect_scaled(gray[y0:y1, x0:x1], x0, y0, scale)
            self.stats["roi"] += 1
            if len(found) < nprev:
                found = {}
        if not found:
            scale = min(1.0, self.max_side / max(H, W))
            found = self._detect_scaled(gray, 0, 0, scale)
            self.stats["full"] += 1
        found = self._refine_full(gray, found)
        if found:
            allc = np.concatenate(list(found.values()))
            side = float(np.median([np.linalg.norm(c[0] - c[1]) for c in found.values()]))
            m = 2.5 * side + 40
            x0 = int(max(0, allc[:, 0].min() - m)); x1 = int(min(W, allc[:, 0].max() + m))
            y0 = int(max(0, allc[:, 1].min() - m)); y1 = int(min(H, allc[:, 1].max() + m))
            self.prev = ((x0, y0, x1, y1), side, len(found))
        else:
            self.prev = None
        return found
