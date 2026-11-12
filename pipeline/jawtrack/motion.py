"""Relative jaw motion: M(t), MI detection, M0, T_S(t), outlier rejection, gap filling, smoothing."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from . import se3

log = logging.getLogger("jawtrack.motion")


@dataclass
class RawSeries:
    times: np.ndarray        # (N,)
    frame_idx: np.ndarray    # (N,)
    M: np.ndarray            # (N,4,4) inv(T_C_H) @ T_C_J (identity where invalid)
    valid: np.ndarray        # (N,) bool: both boards tracked and accepted
    quality: np.ndarray      # (N,) 0..1
    head_reproj: np.ndarray
    jaw_reproj: np.ndarray
    n_head: np.ndarray
    n_jaw: np.ndarray


def load_series(raw: dict, max_reproj_px=None) -> RawSeries:
    """Build M(t) and per-frame quality from raw_tracking.json frames.

    Frames are rejected when a board is missing, or its reprojection error exceeds
    ``max(3 x median, 1.5 px)`` (or ``max_reproj_px``)."""
    fr = raw["frames"]
    N = len(fr)
    times = np.array([f["t"] for f in fr], float)
    idx = np.array([f["i"] for f in fr], int)
    M = np.repeat(np.eye(4)[None], N, axis=0)
    hr = np.full(N, np.nan); jr = np.full(N, np.nan)
    nh = np.zeros(N, int); nj = np.zeros(N, int)
    for k, f in enumerate(fr):
        if f.get("head"):
            hr[k] = f["head"]["reproj_px"]; nh[k] = f["head"]["n_markers"]
        if f.get("jaw"):
            jr[k] = f["jaw"]["reproj_px"]; nj[k] = f["jaw"]["n_markers"]
        if f.get("head") and f.get("jaw"):
            M[k] = se3.inv(se3.from_list(f["head"]["T"])) @ se3.from_list(f["jaw"]["T"])
    valid = (nh > 0) & (nj > 0)
    if valid.sum() == 0:
        raise RuntimeError("no frame has both HEAD and JAW boards tracked")
    thr_h = max_reproj_px or max(3 * np.nanmedian(hr[valid]), 1.5)
    thr_j = max_reproj_px or max(3 * np.nanmedian(jr[valid]), 1.5)
    bad = valid & ((hr > thr_h) | (jr > thr_j))
    if bad.any():
        log.info("rejected %d frames with high reprojection error (thr head %.2f, jaw %.2f px)",
                 int(bad.sum()), thr_h, thr_j)
    valid &= ~bad
    q = np.where(valid, np.minimum(1, nh / 6) ** 0.5 * np.minimum(1, nj / 4) ** 0.5 *
                 np.exp(-np.maximum(0, np.nan_to_num(np.fmax(hr, jr)) - 0.5) / 2.0), 0.0)
    return RawSeries(times, idx, M, valid, q, hr, jr, nh, nj)


def t_s_series(X, M, M0):
    """T_S(t) = inv(X) M(t) inv(M0) X for all frames (SPEC section 3)."""
    A = se3.inv(X)
    B = se3.inv(M0) @ X
    return np.einsum("ij,njk,kl->nil", A, M, B)


def incisal_displacement(TS, p):
    """Incisal-point displacement (N,3) in S: T_S(t) p - p."""
    return np.einsum("nij,j->ni", TS[:, :3, :3], p) + TS[:, :3, 3] - p


def detect_mi_interval(times, disp_mm, valid, thr_mm=None, min_s=0.5, search_frac=0.4):
    """First stable segment at the start: the first run (within the first ``search_frac`` of the
    recording) of >= ``min_s`` seconds where the incisal displacement (relative to a provisional M0)
    stays below a threshold (default max(0.5 mm, 4 x robust noise)). Returns (t0, t1) or None."""
    d = np.where(valid, disp_mm, np.inf)
    v = disp_mm[valid]
    if v.size < 5:
        return None
    if thr_mm is None:
        early = disp_mm[valid & (times <= times[valid][0] + 1.0)]
        noise = 1.4826 * np.median(np.abs(early - np.median(early))) if early.size > 3 else 0.1
        thr_mm = max(0.5, 4 * noise + np.median(early) if early.size else 0.5)
    ok = d < thr_mm
    # tolerate single-frame glitches
    okc = ok.copy()
    for k in range(1, len(ok) - 1):
        if not ok[k] and ok[k - 1] and ok[k + 1]:
            okc[k] = True
    tmax = times[0] + search_frac * (times[-1] - times[0])
    k = 0
    while k < len(okc) and times[k] <= tmax:
        if okc[k]:
            j = k
            while j + 1 < len(okc) and okc[j + 1]:
                j += 1
            if times[j] - times[k] >= min_s:
                return float(times[k]), float(times[j])
            k = j + 1
        else:
            k += 1
    return None


def mi_holds(times, disp_mm, valid, thr_mm, min_s=0.4):
    """All intervals where the jaw rests in MI (displacement < thr for >= min_s)."""
    ok = valid & (disp_mm < thr_mm)
    out, k = [], 0
    while k < len(ok):
        if ok[k]:
            j = k
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if times[j] - times[k] >= min_s:
                out.append((float(times[k]), float(times[j])))
            k = j + 1
        else:
            k += 1
    return out


def reject_outliers(times, TS, valid, p, win=7, k=5.0, floor_mm=1.0, floor_deg=2.0):
    """Hampel filter on incisal position and rotation vector; returns the updated valid mask."""
    q = incisal_displacement(TS, p)
    r = Rotation.from_matrix(TS[:, :3, :3]).as_rotvec() * 180 / np.pi
    out = valid.copy()
    vi = np.flatnonzero(valid)
    h = win // 2
    n_rej = 0
    for a, (sig, floor) in enumerate(((q, floor_mm), (r, floor_deg))):
        s = sig[vi]
        for j in range(len(vi)):
            lo, hi = max(0, j - h), min(len(vi), j + h + 1)
            med = np.median(s[lo:hi], axis=0)
            mad = 1.4826 * np.median(np.abs(s[lo:hi] - med), axis=0)
            if np.any(np.abs(s[j] - med) > k * mad + floor):
                if out[vi[j]]:
                    n_rej += 1
                out[vi[j]] = False
    if n_rej:
        log.info("Hampel filter rejected %d outlier frames", n_rej)
    return out


def fill_and_smooth(times, TS, valid, p, max_gap_s=0.25, smooth_s=0.12, polyorder=2):
    """Interpolate gaps (slerp / linear), Savitzky-Golay smooth the rotation vector and the incisal
    point position, and rebuild T. Returns (TS_out, status) with status 0=measured, 1=interpolated
    (short gap, considered valid), 2=invalid (long gap / outside tracked range; T interpolated or held
    for display only)."""
    N = len(times)
    vi = np.flatnonzero(valid)
    if vi.size < 2:
        raise RuntimeError("fewer than 2 valid frames")
    TSf = se3.interpolate_poses(times[vi], TS[vi], times)
    status = np.where(valid, 0, 1)
    # long gaps -> invalid
    for a, b in zip(vi[:-1], vi[1:]):
        if b - a > 1 and times[b] - times[a] > max_gap_s:
            status[a + 1:b] = 2
    status[:vi[0]] = 2
    status[vi[-1] + 1:] = 2
    dt = np.median(np.diff(times)) if N > 1 else 1.0
    w = int(round(smooth_s / dt)) | 1
    if smooth_s > 0 and w >= polyorder + 2 and N >= w:
        r = Rotation.from_matrix(TSf[:, :3, :3]).as_rotvec()
        q = np.einsum("nij,j->ni", TSf[:, :3, :3], p) + TSf[:, :3, 3]
        # smooth each contiguous run of non-invalid frames separately
        good = status < 2
        k = 0
        while k < N:
            if good[k]:
                j = k
                while j + 1 < N and good[j + 1]:
                    j += 1
                if j - k + 1 >= w:
                    r[k:j + 1] = savgol_filter(r[k:j + 1], w, polyorder, axis=0)
                    q[k:j + 1] = savgol_filter(q[k:j + 1], w, polyorder, axis=0)
                k = j + 1
            else:
                k += 1
        R = Rotation.from_rotvec(r).as_matrix()
        TSf[:, :3, :3] = R
        TSf[:, :3, 3] = q - np.einsum("nij,j->ni", R, p)
        log.info("Savitzky-Golay smoothing: window %d frames (%.3f s), order %d", w, w * dt, polyorder)
    return TSf, status
