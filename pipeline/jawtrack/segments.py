"""Automatic segmentation of the recording into SPEC labels.

1. Incisal-point displacement from MI is expressed in the occlusal-plane frame:
   lateral (+ = patient's left), vertical opening (+ = down), anterior (+ = forward).
2. "Excursions" = runs where |displacement| exceeds a movement threshold (short gaps merged).
3. Each excursion gets a cost per label from its displacement at peak (share of the opening,
   anterior, right-lateral and left-lateral components; chewing favours short cycles).
4. Labels are assigned by dynamic programming constrained to the protocol order of SPEC 2
   (open_close -> protrusion -> laterotrusion_right -> laterotrusion_left -> chewing, labels may be
   skipped but never go backwards).
5. Consecutive excursions with the same label form one segment; rests between groups (>= 0.4 s)
   and the start/end holds become MI segments.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("jawtrack.segments")

ORDER = ["open_close", "protrusion", "laterotrusion_right", "laterotrusion_left", "chewing"]


def excursions(times, mag, valid, thr, merge_s=0.15, min_s=0.15):
    mv = valid & (mag > thr)
    runs, k, N = [], 0, len(mv)
    while k < N:
        if mv[k]:
            j = k
            while j + 1 < N and (mv[j + 1] or not valid[j + 1]):
                j += 1
            runs.append([k, j])
            k = j + 1
        else:
            k += 1
    merged = []
    for r in runs:
        if merged and times[r[0]] - times[merged[-1][1]] < merge_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if times[r[1]] - times[r[0]] >= min_s]


def label_costs(loc_peak, duration_s):
    """loc_peak = (lateral_left, up, anterior) displacement at the excursion peak."""
    lat, up, ant = loc_peak
    comps = np.array([max(-up, 0) * 0.8, max(ant, 0), max(-lat, 0), max(lat, 0)])
    share = comps / max(comps.sum(), 1e-9)
    c = list(1.0 - share)
    c.append(0.45 if duration_s < 1.3 else 1.5)  # chewing: short cycles
    return np.array(c)


def segment(times, disp_local, valid, mi_thr_mm=1.0, move_thr_mm=None):
    """Return list of {"label","t0","t1"} sorted by time."""
    times = np.asarray(times, float)
    mag = np.linalg.norm(disp_local, axis=1)
    thr = move_thr_mm or max(1.5, 2.0 * mi_thr_mm)
    ex = excursions(times, mag, valid, thr)
    segs = []
    if not ex:
        return [{"label": "MI", "t0": float(times[0]), "t1": float(times[-1])}]
    C = []
    for a, b in ex:
        pk = a + int(np.argmax(np.where(valid[a:b + 1], mag[a:b + 1], -1)))
        C.append(label_costs(disp_local[pk], times[b] - times[a]))
    C = np.array(C)  # (n_ex, 5)
    n, L = C.shape
    D = np.full((n, L), np.inf); B = np.zeros((n, L), int)
    D[0] = C[0]
    for i in range(1, n):
        for l in range(L):
            j = int(np.argmin(D[i - 1, :l + 1]))
            D[i, l] = C[i, l] + D[i - 1, j]
            B[i, l] = j
    lab = [int(np.argmin(D[-1]))]
    for i in range(n - 1, 0, -1):
        lab.append(B[i, lab[-1]])
    lab = lab[::-1]
    # group
    groups = []
    for (a, b), l in zip(ex, lab):
        if groups and groups[-1][2] == l:
            groups[-1][1] = b
        else:
            groups.append([a, b, l])
    t_prev = times[0]
    for a, b, l in groups:
        if times[a] - t_prev >= 0.4:
            segs.append({"label": "MI", "t0": float(t_prev), "t1": float(times[a])})
        segs.append({"label": ORDER[l], "t0": float(times[a]), "t1": float(times[b])})
        t_prev = times[b]
    if times[-1] - t_prev >= 0.2:
        segs.append({"label": "MI", "t0": float(t_prev), "t1": float(times[-1])})
    log.info("segments: %s", ", ".join(f"{s['label']} {s['t0']:.1f}-{s['t1']:.1f}s" for s in segs))
    found = {s["label"] for s in segs}
    missing = [o for o in ORDER if o not in found]
    if missing:
        log.warning("segments not detected: %s (use --segments to override)", ", ".join(missing))
    return segs


def mask_for(times, segs, label):
    m = np.zeros(len(times), bool)
    for s in segs:
        if s["label"] == label:
            m |= (times >= s["t0"]) & (times <= s["t1"])
    return m
