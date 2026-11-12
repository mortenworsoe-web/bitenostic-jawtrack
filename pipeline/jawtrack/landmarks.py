"""Anatomical landmarks in the STL frame S: incisal point, occlusal-plane frame, condyles,
kinematic hinge axis.

Occlusal-plane frame (OPF): origin at the lower incisal point; axes
  e_left (patient's left), e_up (normal of the occlusal plane, towards the upper jaw),
  e_ant (anterior, in the plane) with e_left x e_up = e_ant (same handedness as SPEC 6).
"""
from __future__ import annotations

import logging

import numpy as np

from . import se3

log = logging.getLogger("jawtrack.landmarks")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class OcclusalFrame:
    """Orthonormal occlusal-plane frame. ``R`` has columns (e_left, e_up, e_ant)."""

    def __init__(self, origin, e_up, e_ant):
        self.origin = np.asarray(origin, float)
        self.e_up = _unit(e_up)
        a = np.asarray(e_ant, float)
        self.e_ant = _unit(a - np.dot(a, self.e_up) * self.e_up)
        self.e_left = np.cross(self.e_up, self.e_ant)
        self.R = np.column_stack([self.e_left, self.e_up, self.e_ant])

    def to_local(self, v):
        """Displacement / direction vectors in S -> (left, up, ant)."""
        return np.asarray(v, float) @ self.R

    def to_S(self, v):
        return np.asarray(v, float) @ self.R.T

    def as_dict(self):
        return {"point": self.origin.tolist(), "normal": self.e_up.tolist(), "anterior": self.e_ant.tolist(),
                "left": self.e_left.tolist()}


def auto_lower_landmarks(lower_vertices, upper_vertices):
    """Fallback incisal point + occlusal plane from the meshes (no axis assumptions).

    * occlusal normal: smallest PCA axis of the lower arch, signed towards the upper arch centroid;
    * anterior: in-plane direction whose extreme slab of vertices is narrowest (the incisor end of
      the U-shaped arch);
    * incisal point: within 3 mm of the arch midline, the vertex maximising anterior + superior;
    * occlusal plane: incisal point + the highest points (top 5 % along the normal) of the
      posterior third, left and right.
    Returns (incisal_point, [incisal, posterior_left, posterior_right]).
    """
    V = np.asarray(lower_vertices, float)
    c = V.mean(axis=0)
    _, _, Vt = np.linalg.svd(V - c, full_matrices=False)
    n = Vt[2] * np.sign(np.dot(Vt[2], np.asarray(upper_vertices, float).mean(axis=0) - c) or 1.0)
    best = None
    for a in (Vt[0], -Vt[0], Vt[1], -Vt[1]):
        lat = np.cross(n, a)
        pa = (V - c) @ a
        slab = V[pa > np.quantile(pa, 0.92)]
        spread = np.ptp((slab - c) @ lat)
        if best is None or spread < best[0]:
            best = (spread, a)
    a = best[1]
    left = np.cross(n, a)
    loc = np.column_stack([(V - c) @ left, (V - c) @ n, (V - c) @ a])
    mid = np.median(loc[:, 0])
    near = np.abs(loc[:, 0] - mid) < 3.0
    score = loc[:, 2] + 0.5 * loc[:, 1]
    inc = V[near][np.argmax(score[near])] if near.any() else V[np.argmax(score)]
    post = loc[:, 2] < np.quantile(loc[:, 2], 0.33)
    pts = []
    for side in (1, -1):
        m = post & (np.sign(loc[:, 0] - mid) == side)
        if m.sum() < 3:
            m = post
        h = loc[m, 1]
        top = V[m][h >= np.quantile(h, 0.95)]
        pts.append(top.mean(axis=0))
    return inc, [inc, pts[0], pts[1]]


def occlusal_frame(incisal, plane_pts, upper_vertices=None, lower_vertices=None):
    """Build the OPF from incisal point + 3 plane points ([incisal, 36 (left), 46 (right)]).
    The normal is signed towards the upper arch if meshes are given; e_ant points from the midpoint
    of the posterior points to the incisal point."""
    p0, pl, pr = [np.asarray(p, float) for p in plane_pts]
    n = _unit(np.cross(pl - p0, pr - p0))
    if upper_vertices is not None and lower_vertices is not None:
        d = np.mean(upper_vertices, axis=0) - np.mean(lower_vertices, axis=0)
        if np.dot(n, d) < 0:
            n = -n
    ant = np.asarray(incisal, float) - 0.5 * (pl + pr)
    of = OcclusalFrame(incisal, n, ant)
    # sanity: 36 should be on the patient's left
    if np.dot(pl - pr, of.e_left) < 0:
        log.warning("occlusal_plane_mm point 2 (36) lies on the patient's RIGHT of point 3 (46) given the "
                    "normal direction: check landmark order or mirrored STL")
    return of


def bonwill_condyles(of: OcclusalFrame, intercondylar=105.0, bonwill=105.0, balkwill_deg=22.0):
    """Average-value condyles: Bonwill equilateral triangle (105 mm), Balkwill angle 22 deg above
    the occlusal plane, built in the OPF from the incisal point. Returns (left, right) in S."""
    half = intercondylar / 2.0
    back = np.sqrt(bonwill ** 2 - half ** 2)
    b = np.radians(balkwill_deg)
    base = of.origin + back * (np.cos(b) * -of.e_ant + np.sin(b) * of.e_up)
    return base + half * of.e_left, base - half * of.e_left


def fit_hinge_axis(Ts, incisal, of: OcclusalFrame):
    """Least-squares fixed rotation axis for early opening poses (T_S, relative to MI).

    Direction u: angle-weighted mean of the rotation axes (signed towards e_left);
    point c: solves P_u (I - R_i) c = P_u t_i (P_u projects out u, which absorbs any slide along
    the axis) with c constrained to the mid-sagittal plane through the incisal point.
    Fit RMS: incisal-point error (mm) of the pure-rotation model (per-frame angle) vs. observed.
    Returns dict(point, direction, fit_rms_mm, n_frames, angle_to_left_deg) or None.
    """
    Ts = np.asarray(Ts, float)
    if len(Ts) < 5:
        return None
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(Ts[:, :3, :3]).as_rotvec()
    ang = np.linalg.norm(rv, axis=1)
    keep = ang > np.radians(1.0)
    if keep.sum() < 4:
        return None
    axes = rv[keep] / ang[keep, None]
    axes *= np.sign(axes @ of.e_left)[:, None]
    u = _unit((axes * ang[keep, None]).sum(axis=0))
    Pu = np.eye(3) - np.outer(u, u)
    A, b = [], []
    for T in Ts[keep]:
        A.append(Pu @ (np.eye(3) - T[:3, :3]))
        b.append(Pu @ T[:3, 3])
    A.append(100.0 * of.e_left[None, :])
    b.append(np.array([100.0 * np.dot(of.e_left, incisal)]))
    c, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    errs = []
    for T in Ts[keep]:
        th = np.dot(Rotation.from_matrix(T[:3, :3]).as_rotvec(), u)
        Rm = Rotation.from_rotvec(th * u).as_matrix()
        pm = Rm @ (incisal - c) + c
        errs.append(np.linalg.norm(pm - se3.apply(T, incisal)))
    return {"point": c.tolist(), "direction": u.tolist(), "fit_rms_mm": float(np.sqrt(np.mean(np.square(errs)))),
            "n_frames": int(keep.sum()),
            "angle_to_left_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(u, of.e_left)), 0, 1))))}


def kinematic_condyles(axis: dict, half=52.5):
    c = np.asarray(axis["point"], float)
    u = _unit(axis["direction"])
    return c + half * u, c - half * u
