"""Board pose estimation with explicit planar-pose-ambiguity handling.

A small planar target seen from 30-40 cm has two physically plausible poses ("flip" ambiguity:
the board normal mirrored about the line of sight). Both reproject almost equally well when the
board is small or nearly fronto-parallel. We therefore:

1. compute BOTH IPPE solutions per frame (cv2.solvePnPGeneric, SOLVEPNP_IPPE) and their RMS
   reprojection errors;
2. choose one candidate per frame by a Viterbi pass over the whole sequence that minimises
   reprojection cost + rotational smoothness between consecutive tracked frames (+ for the JAW
   board optionally a prior that the jaw board orientation relative to the head board stays close
   to its typical value: the jaw rotates < ~40 deg relative to the head, a flip is usually larger);
3. refine the chosen candidate with cv2.solvePnPRefineLM on all marker corners of the board.

The ambiguity ratio (err_second / err_best) is stored per frame so flips can be audited.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import se3


def gather_points(found: dict, objpts: dict):
    """Stack object/image points of all detected markers that belong to a board."""
    ids = sorted(i for i in found if i in objpts)
    if not ids:
        return None, None, []
    obj = np.concatenate([objpts[i] for i in ids]).astype(np.float64)
    img = np.concatenate([found[i] for i in ids]).astype(np.float64)
    return obj, img, ids


def reproj_rms(obj, img, T, K, dist) -> float:
    rv, tv = se3.to_rvec_tvec(T)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def ippe_candidates(obj, img, K, dist):
    """Both IPPE solutions as [(T, rms_px)], best first. Falls back to SQPNP if IPPE fails."""
    cands = []
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        for rv, tv in zip(rvecs, tvecs):
            T = se3.from_rvec_tvec(rv, tv)
            if T[2, 3] > 0:  # in front of the camera
                cands.append((T, reproj_rms(obj, img, T, K, dist)))
    except cv2.error:
        pass
    if not cands:
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            T = se3.from_rvec_tvec(rv, tv)
            cands.append((T, reproj_rms(obj, img, T, K, dist)))
    cands.sort(key=lambda c: c[1])
    return cands


def refine_lm(obj, img, K, dist, T):
    rv, tv = se3.to_rvec_tvec(T)
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv,
                                  criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-10))
    T2 = se3.from_rvec_tvec(rv, tv)
    return T2, reproj_rms(obj, img, T2, K, dist)


def viterbi_select(cand_seq, frame_idx, sigma_px=0.7, sigma_deg=2.0, unary_extra=None):
    """Choose one candidate per frame.

    cand_seq: list (per tracked frame) of lists of (T, rms, n_pts); frame_idx: frame indices
    (for gap-dependent smoothness). unary_extra(k_frame, T) -> extra cost (optional).
    Returns list of chosen candidate indices.
    """
    n = len(cand_seq)
    if n == 0:
        return []
    costs, back = [], []
    for f, cands in enumerate(cand_seq):
        emin = min(c[1] for c in cands)
        u = np.array([c[2] * (c[1] ** 2 - emin ** 2) / (2 * sigma_px ** 2) for c in cands])
        if unary_extra is not None:
            u = u + np.array([unary_extra(f, c[0]) for c in cands])
        if f == 0:
            costs.append(u)
            back.append(np.zeros(len(cands), int))
            continue
        gap = max(1, frame_idx[f] - frame_idx[f - 1])
        sa = sigma_deg * (1.0 + 0.5 * (gap - 1))
        prev = cand_seq[f - 1]
        P = np.array([[0.5 * (se3.rot_dist_deg(p[0][:3, :3], c[0][:3, :3]) / sa) ** 2 for p in prev]
                      for c in cands])  # (cur, prev)
        if gap > 30:  # long dropout: no smoothness link
            P[:] = 0.0
        tot = P + costs[-1][None, :]
        back.append(np.argmin(tot, axis=1))
        costs.append(u + np.min(tot, axis=1))
    k = int(np.argmin(costs[-1]))
    path = [k]
    for f in range(n - 1, 0, -1):
        k = int(back[f][k])
        path.append(k)
    return path[::-1]
