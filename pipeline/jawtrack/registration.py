"""Registration X = T_H_S (STL -> HEAD board) from clicked landmarks (SPEC section 4).

Upper-jaw landmarks are rigid with the head:   x_img ~ K [T_C_H X] p
Lower-jaw landmarks are rigid with the jaw:    x_img ~ K [T_C_J inv(M0) X] p
(the latter equals the former when the registration frame is exactly in MI, but stays correct if
the registration frame is slightly off MI and JAW is tracked in that frame). X is initialised by
SQPnP on all points (treating everything as upper) and refined by Levenberg-Marquardt on the
reprojection error of all points.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import se3

log = logging.getLogger("jawtrack.registration")


def _project(K, dist, T_C_P, pts):
    rv, tv = se3.to_rvec_tvec(T_C_P)
    p, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 3), rv, tv, K, dist)
    return p.reshape(-1, 2)


def register(landmarks: dict, T_C_H, K, dist, T_C_J=None, M0=None):
    """Return (X, info). info: rms_px, per-point residuals, method."""
    pts = landmarks.get("points", [])
    if len(pts) < 4:
        raise ValueError(f"registration needs >= 4 landmark points, got {len(pts)}")
    P = np.array([p["stl_mm"] for p in pts], float)
    U = np.array([p["img_px"] for p in pts], float)
    is_lower = np.array([p.get("jaw", "upper") == "lower" for p in pts])
    use_jaw = T_C_J is not None and M0 is not None and is_lower.any()
    K = np.asarray(K, float); dist = np.asarray(dist, float)

    ok, rv, tv = cv2.solvePnP(P, U, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        ok, rv, tv = cv2.solvePnP(P, U, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("PnP failed for registration landmarks")
    rv, tv = cv2.solvePnPRefineLM(P, U, K, dist, rv, tv)
    X0 = se3.inv(T_C_H) @ se3.from_rvec_tvec(rv, tv)
    pre_lower = (T_C_J @ se3.inv(M0)) if use_jaw else T_C_H

    def model(x):
        Xc = se3.make_T(Rotation.from_rotvec(x[:3]).as_matrix() @ X0[:3, :3], X0[:3, 3] + x[3:])
        pr = np.empty_like(U)
        if (~is_lower).any():
            pr[~is_lower] = _project(K, dist, T_C_H @ Xc, P[~is_lower])
        if is_lower.any():
            pr[is_lower] = _project(K, dist, pre_lower @ Xc, P[is_lower])
        return Xc, pr

    res = least_squares(lambda x: (model(x)[1] - U).ravel(), np.zeros(6), method="lm")
    X, pr = model(res.x)
    r = np.linalg.norm(pr - U, axis=1)
    rms = float(np.sqrt(np.mean(r ** 2)))
    # covariance of the 6 registration parameters (rotvec, translation) from the LM Jacobian
    n_res = 2 * len(pts)
    sigma = max(rms * np.sqrt(n_res / max(n_res - 6, 1)), 0.5)  # floor: 0.5 px click noise
    try:
        cov = sigma ** 2 * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = None
    info = {"rms_px": rms, "sigma_px_used": float(sigma),
            "cov_rotvec_t": None if cov is None else cov.tolist(),
            "X0_rot": X0[:3, :3].tolist(), "n_points": len(pts), "lower_points_via_jaw_pose": bool(use_jaw),
            "residuals_px": {p.get("name", str(k)): round(float(e), 3) for k, (p, e) in enumerate(zip(pts, r))}}
    # depth of the landmark centroid, for the mm-equivalent of 1 px
    zc = float(np.mean(se3.apply(T_C_H @ X, P)[:, 2]))
    info["mm_per_px_at_teeth"] = zc / K[0, 0]
    log.info("registration: %d points, RMS %.2f px (~%.2f mm at the teeth), lower points via jaw pose: %s",
             len(pts), rms, rms * zc / K[0, 0], use_jaw)
    if rms > 4.0:
        log.warning("registration RMS %.1f px > 4 px: check the landmark clicks / correspondence order", rms)
    if len(pts) < 6:
        log.warning("only %d landmarks: 6 or more (spread over the arch) are recommended", len(pts))
    return X, info


def lever_uncertainty(info, X, M, M0, point, n=300, seed=0, cov=None, x_of=None):
    """Monte-Carlo propagation of a registration covariance to the position of ``point`` (S) moved by
    the relative pose M (e.g. max opening): RMS 3-D error (mm) of T_S(M) @ point.
    Default parametrisation is the click registration's (rotvec left-multiplied, translation added);
    pass ``cov`` and ``x_of`` (delta -> X) for another one."""
    cov = info.get("cov_rotvec_t") if cov is None else cov
    if cov is None:
        return None
    if x_of is None:
        def x_of(x):
            return se3.make_T(Rotation.from_rotvec(x[:3]).as_matrix() @ X[:3, :3], X[:3, 3] + x[3:])
        base = np.zeros(6)
    else:
        base = None
    rng = np.random.default_rng(seed)
    d = rng.multivariate_normal(np.zeros(6), np.asarray(cov), n)
    X_ref = X if base is not None else x_of(np.zeros(6))
    ref = se3.apply(se3.ts_formula(X_ref, M, M0), point)
    errs = [np.linalg.norm(se3.apply(se3.ts_formula(x_of(x), M, M0), point) - ref) for x in d]
    return float(np.sqrt(np.mean(np.square(errs))))
