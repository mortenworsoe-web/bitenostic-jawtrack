"""Rigid-transform (SE(3)) utilities.

All transforms are 4x4 numpy arrays mapping points of the source frame into the target frame
(``T_A_B`` maps B coordinates -> A coordinates). Serialisation is ROW-MAJOR (SPEC 5.1/5.2):
``T.reshape(-1)`` -> ``[r00, r01, r02, tx, r10, ..., 0, 0, 0, 1]``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def make_T(R: np.ndarray | None = None, t=None) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation and translation (defaults: identity / zero)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def inv(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform."""
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def from_rvec_tvec(rvec, tvec) -> np.ndarray:
    """OpenCV (rvec, tvec) -> 4x4."""
    return make_T(Rotation.from_rotvec(np.asarray(rvec, float).reshape(3)).as_matrix(),
                  np.asarray(tvec, float).reshape(3))


def to_rvec_tvec(T: np.ndarray):
    """4x4 -> OpenCV (rvec (3,1), tvec (3,1))."""
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return rv.reshape(3, 1).copy(), T[:3, 3].reshape(3, 1).copy()


def to_list(T: np.ndarray) -> list:
    """Row-major 16-float list (SPEC serialisation)."""
    return [float(x) for x in np.asarray(T, float).reshape(-1)]


def from_list(v) -> np.ndarray:
    """Row-major 16-float list -> 4x4."""
    return np.asarray(v, float).reshape(4, 4)


def apply(T: np.ndarray, pts) -> np.ndarray:
    """Apply T to points (N,3) or (3,)."""
    p = np.asarray(pts, float)
    return p @ T[:3, :3].T + T[:3, 3]


def rot_angle_deg(R: np.ndarray) -> float:
    """Rotation angle of a 3x3 rotation matrix in degrees."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rot_dist_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic distance between two rotations in degrees."""
    return rot_angle_deg(Ra.T @ Rb)


def project_to_SO3(R: np.ndarray) -> np.ndarray:
    """Closest rotation matrix (SVD)."""
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def robust_average(Ts, n_iter: int = 3, k: float = 3.0):
    """Robust average of rigid transforms.

    Rotation: chordal L2 mean (scipy ``Rotation.mean``) with iterative rejection of samples whose
    geodesic distance to the mean exceeds ``max(k*MAD, 0.05 deg)``; translation: per-axis median of
    the inliers.  Returns (T_mean, inlier_mask).
    """
    Ts = np.asarray(Ts, float)
    n = len(Ts)
    if n == 0:
        raise ValueError("robust_average: no transforms")
    mask = np.ones(n, bool)
    rots = Rotation.from_matrix(Ts[:, :3, :3])
    for _ in range(n_iter):
        Rm = rots[mask].mean()
        d = (Rm.inv() * rots).magnitude()
        tm = np.median(Ts[mask, :3, 3], axis=0)
        dt = np.linalg.norm(Ts[:, :3, 3] - tm, axis=1)
        md = np.median(d[mask]); mt = np.median(dt[mask])
        new = (d <= max(k * 1.4826 * md, np.radians(0.05)) + md) & (dt <= max(k * 1.4826 * mt, 0.02) + mt)
        if new.sum() < max(3, 0.3 * n) or np.array_equal(new, mask):
            break
        mask = new
    Rm = rots[mask].mean().as_matrix()
    tm = np.median(Ts[mask, :3, 3], axis=0)
    return make_T(Rm, tm), mask


def interpolate_poses(times_known, Ts_known, times_query):
    """Slerp rotations / linearly interpolate translations at query times (inside the known range)."""
    tk = np.asarray(times_known, float)
    Ts_known = np.asarray(Ts_known, float)
    tq = np.clip(np.asarray(times_query, float), tk[0], tk[-1])
    sl = Slerp(tk, Rotation.from_matrix(Ts_known[:, :3, :3]))
    Rq = sl(tq).as_matrix()
    out = np.repeat(np.eye(4)[None], len(tq), axis=0)
    out[:, :3, :3] = Rq
    for a in range(3):
        out[:, a, 3] = np.interp(tq, tk, Ts_known[:, a, 3])
    return out


def ts_formula(X: np.ndarray, M: np.ndarray, M0: np.ndarray) -> np.ndarray:
    """SPEC section 3: T_S(t) = inv(X) @ M(t) @ inv(M0) @ X  (X = T_H_S)."""
    return inv(X) @ M @ inv(M0) @ X
