"""Image-based refinement of the registration X (optional, `solve --refine-registration`).

The click-based X (registration.py) is limited by click noise: 6-10 points constrain the depth of
the dentition along the viewing ray poorly, and X acts as a lever on everything downstream.
Here X is refined against the registration image itself:

(Implemented method: see refine_registration — a grid search along the weakest directions of the
click covariance, scored by rendered-shading edges; an earlier local least-squares variant on
silhouette/crease edges was dropped because its cost was flat/biased on the test data.)

Original design notes:
1. model edges: silhouette edges (adjacent faces front/back facing w.r.t. the camera centre) and
   sharp crease edges (dihedral > ``crease_deg``) of the upper + lower (MI) meshes, sampled every
   ~0.3 mm, kept only where visible (vertex-splat z-buffer of both meshes, min-filtered);
2. image edges: Canny on the blurred registration image inside the projected-dentition ROI, both
   marker boards masked out; distance transform (truncated) of the edge map;
3. 6-DoF correction of X (rotation about / translation at the landmark centroid, in S) by
   scipy least_squares (TRF, soft-L1 loss), residuals = edge distances (sigma_edge px) + the click
   reprojection residuals (sigma_click px) as a prior, bounded to +-max_mm / +-max_deg around the
   click solution; the visible edge set is recomputed in 3 outer iterations;
4. sanity checks (reject and keep the click X if): any parameter ends at its bound, the click RMS
   grows by > max(1.5 px, 1.5x), or the median edge distance does not improve.

Caveat: synthetic test renders are flat-shaded with crisp geometric edges; real enamel/gingiva,
saliva highlights and the retractor give weaker, noisier edges.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import se3

log = logging.getLogger("jawtrack.regrefine")


class MeshRenderer:
    """Tiny painter's-algorithm rasteriser (OpenCV fillConvexPoly, 1/16 px precision) producing a
    triangle-id image and a Lambert shading image (light at the camera) inside an ROI."""

    def __init__(self, meshes):
        self.parts = []
        for m in meshes:
            self.parts.append((np.ascontiguousarray(np.asarray(m.vertices, float)), np.asarray(m.faces)))

    def render(self, Ts, K, dist, roi):
        """Ts: one camera-from-S transform per mesh. Returns (tri_id image (-1 = background), shade
        image uint8, list of per-triangle (part, face) lookup arrays)."""
        x0, y0, x1, y1 = roi
        W, H = x1 - x0, y1 - y0
        tri_uv, tri_z, tri_shade, lut, tri_n = [], [], [], [], []
        for k, ((V, F), T) in enumerate(zip(self.parts, Ts)):
            Vc = se3.apply(T, V)
            uv, _ = _project(K, dist, T, V)
            A, B, C = Vc[F[:, 0]], Vc[F[:, 1]], Vc[F[:, 2]]
            n = np.cross(B - A, C - A)
            nn = np.linalg.norm(n, axis=1) + 1e-12
            cen = (A + B + C) / 3
            facing = np.einsum("ij,ij->i", n, cen) < 0
            tuv = uv[F]  # (nf,3,2)
            inroi = ((tuv[:, :, 0] >= x0) & (tuv[:, :, 0] < x1) & (tuv[:, :, 1] >= y0) & (tuv[:, :, 1] < y1)).all(1)
            sel = np.flatnonzero(facing & inroi)
            tri_uv.append(tuv[sel] - [x0, y0])
            tri_z.append(cen[sel, 2])
            tri_shade.append(np.abs(np.einsum("ij,ij->i", n[sel], cen[sel])) / nn[sel] / np.linalg.norm(cen[sel], axis=1))
            tri_n.append(n[sel] / nn[sel, None])
            lut.append(np.column_stack([np.full(len(sel), k), sel]))
        tuv = np.concatenate(tri_uv); tz = np.concatenate(tri_z); ts = np.concatenate(tri_shade)
        lut = np.concatenate(lut)
        order = np.argsort(-tz)  # far -> near
        ids = np.full((H, W), -1, np.int32)
        pts16 = np.round(tuv * 16).astype(np.int32)
        for i in order:
            cv2.fillConvexPoly(ids, pts16[i], int(i), lineType=cv2.LINE_8, shift=4)
        shade = np.where(ids >= 0, (255 * ts[np.maximum(ids, 0)]), 0).astype(np.uint8)
        self.depth = np.where(ids >= 0, tz[np.maximum(ids, 0)], 1e6).astype(np.float32)
        self.part = np.where(ids >= 0, lut[np.maximum(ids, 0), 0], -1)
        tn = np.concatenate(tri_n)
        self.normal = np.where((ids >= 0)[..., None], tn[np.maximum(ids, 0)], 0).astype(np.float32)
        return ids, shade, lut


def _project(K, dist, T_C_X, P):
    rv, tv = se3.to_rvec_tvec(T_C_X)
    uv, _ = cv2.projectPoints(np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3)), rv, tv, K, dist)
    z = se3.apply(T_C_X, P)[:, 2]
    return uv.reshape(-1, 2), z


def _zbuffer(shape, uvz_list, radius=2):
    """Min-depth splat of vertices (several (uv, z) arrays), min-filtered to close gaps."""
    H, W = shape
    zb = np.full((H, W), np.inf, np.float32)
    for uv, z in uvz_list:
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        np.minimum.at(zb, (v[ok], u[ok]), z[ok].astype(np.float32))
    zf = np.where(np.isfinite(zb), zb, 1e9).astype(np.float32)
    k = 2 * radius + 1
    return cv2.erode(zf, np.ones((k, k), np.uint8))


def image_edges(gray, roi, masks_poly=(), blur=1.2):
    """Canny edge map inside roi=(x0,y0,x1,y1) with polygons masked out; returns full-size uint8."""
    x0, y0, x1, y1 = roi
    sub = cv2.GaussianBlur(gray[y0:y1, x0:x1], (0, 0), blur)
    v = np.median(sub)
    e = cv2.Canny(sub, int(max(10, 0.5 * v)), int(min(255, max(30, 1.2 * v))), L2gradient=True)
    out = np.zeros_like(gray)
    out[y0:y1, x0:x1] = e
    for poly in masks_poly:
        cv2.fillPoly(out, [np.round(poly).astype(np.int32)], 0)
    return out


def refine_registration(X0, landmarks, image, T_C_H, K, dist, upper_mesh, lower_mesh, T_C_J=None, M0=None,
                        board_polys=(), click_cov=None, occlusal_frame=None, crown_mm=8.0, sigma_kernel_px=1.5,
                        coarse=13, span_sigma=3.0, fine=7, prior_weight=0.001, min_gain=0.005):
    """Refine the click-based X along its poorly determined directions using image edges.

    The click PnP is ill-conditioned in essentially one or two directions (a depth/tilt combination:
    eigen-decomposition of the click covariance typically gives one sd of several mm and the rest
    < 0.4 mm). Instead of a local optimisation (whose edge cost is flat and biased), X is searched on
    a grid spanning +-``span_sigma`` sd along the two weakest eigen-directions (coarse grid, then a
    fine grid around the best cell). Each candidate is scored by rendering the dentition (upper + lower
    in MI) with Lambert shading, extracting rendered Canny edges on the crowns (|height| < crown_mm from
    the occlusal plane; boards and the render outline excluded) and computing the mean Gaussian
    proximity exp(-d^2 / 2 s^2) to the image's Canny edges (distance transform), minus a small prior
    on the click solution. Returns (X, info); info['applied'] is False if the gain is insignificant or
    a sanity check fails.
    """
    K = np.asarray(K, float); dist = np.asarray(dist, float)
    if click_cov is None:
        return X0, {"applied": False, "reason": "no click covariance"}
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    pts = landmarks["points"]
    P = np.array([p["stl_mm"] for p in pts], float)
    U = np.array([p["img_px"] for p in pts], float)
    low = np.array([p.get("jaw", "upper") == "lower" for p in pts])
    pre_low = (T_C_J @ se3.inv(M0)) if (T_C_J is not None and M0 is not None) else T_C_H
    rend = MeshRenderer([upper_mesh, lower_mesh])
    Vu = np.ascontiguousarray(np.asarray(upper_mesh.vertices, float))
    Vl = np.ascontiguousarray(np.asarray(lower_mesh.vertices, float))

    def X_of(x):  # same parametrisation as registration.register's covariance
        return se3.make_T(Rotation.from_rotvec(x[:3]).as_matrix() @ X0[:3, :3], X0[:3, 3] + x[3:])

    def click_rms(Xc):
        pr = np.empty_like(U)
        if (~low).any():
            pr[~low] = _project(K, dist, T_C_H @ Xc, P[~low])[0]
        if low.any():
            pr[low] = _project(K, dist, pre_low @ Xc, P[low])[0]
        return float(np.sqrt(np.mean(np.sum((pr - U) ** 2, axis=1))))

    uv = np.vstack([_project(K, dist, T_C_H @ X0, Vu[::7])[0], _project(K, dist, pre_low @ X0, Vl[::7])[0]])
    pad = 40
    roi = (int(max(0, uv[:, 0].min() - pad)), int(max(0, uv[:, 1].min() - pad)),
           int(min(W, uv[:, 0].max() + pad)), int(min(H, uv[:, 1].max() + pad)))
    if roi[2] - roi[0] < 30 or roi[3] - roi[1] < 30:
        return X0, {"applied": False, "reason": "projected dentition outside the image"}
    edges = image_edges(gray, roi, board_polys)
    dt = np.minimum(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)[roi[1]:roi[3], roi[0]:roi[2]], 50.0)
    board_mask = np.zeros_like(gray)
    for poly in board_polys:
        cv2.fillPoly(board_mask, [np.round(poly).astype(np.int32)], 255)
    bm = board_mask[roi[1]:roi[3], roi[0]:roi[2]] > 0
    up_n = None
    if occlusal_frame is not None:
        up_n = (occlusal_frame.origin, occlusal_frame.e_up)
    crown = []
    for V in (Vu, Vl):
        crown.append(np.ones(len(V), bool) if up_n is None else np.abs((V - up_n[0]) @ up_n[1]) < crown_mm)

    def score(Xc):
        ids, shade, lut = rend.render([T_C_H @ Xc, pre_low @ Xc], K, dist, roi)
        e = cv2.Canny(cv2.GaussianBlur(shade, (0, 0), 1.2), 20, 50, L2gradient=True) > 0
        bg = cv2.dilate((ids < 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        e &= ~bg & ~bm
        if up_n is not None:
            # crown filter via the triangle's first vertex
            tid = ids[e]
            part, face = lut[tid, 0], lut[tid, 1]
            keep = np.zeros(len(tid), bool)
            for k, (msh, cm) in enumerate(zip((upper_mesh, lower_mesh), crown)):
                m = part == k
                keep[m] = cm[np.asarray(msh.faces)[face[m], 0]]
            d = dt[e][keep]
        else:
            d = dt[e]
        if d.size < 50:
            return -1.0, int(d.size)
        return float(np.mean(np.exp(-d ** 2 / (2 * sigma_kernel_px ** 2)))), int(d.size)

    cov = np.asarray(click_cov, float)
    w, V = np.linalg.eigh(cov)
    dirs = [(np.sqrt(max(w[-1], 0)), V[:, -1]), (np.sqrt(max(w[-2], 0)), V[:, -2])]
    s0, n0 = score(X0)
    if s0 < 0:
        return X0, {"applied": False, "reason": f"too few rendered crown edges ({n0})"}

    def grid(c_ab, half, n):
        A = np.linspace(c_ab[0] - half, c_ab[0] + half, n)
        B = np.linspace(c_ab[1] - half, c_ab[1] + half, n)
        S = np.zeros((n, n))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                x = a * dirs[0][0] * dirs[0][1] + b * dirs[1][0] * dirs[1][1]
                S[i, j] = score(X_of(x))[0] - prior_weight * (a * a + b * b)
        return A, B, S

    A, B, S = grid((0.0, 0.0), span_sigma, coarse)
    Ss = cv2.blur(S.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)  # robust to single-cell noise
    i, j = np.unravel_index(np.argmax(Ss), Ss.shape)
    step = A[1] - A[0]
    A2, B2, S2 = grid((A[i], B[j]), step, fine)
    i2, j2 = np.unravel_index(np.argmax(S2), S2.shape)
    a_best, b_best = A2[i2], B2[j2]
    x_best = a_best * dirs[0][0] * dirs[0][1] + b_best * dirs[1][0] * dirs[1][1]
    X = X_of(x_best)
    s1, n1 = score(X)
    c0, c1 = click_rms(X0), click_rms(X)
    # spread of near-optimal coarse cells -> remaining uncertainty along the searched directions
    near = S >= S.max() - 0.5 * min_gain
    AA, BB = np.meshgrid(A, B, indexing="ij")
    var_a = float(np.var(AA[near]) + (step / 2) ** 2)
    var_b = float(np.var(BB[near]) + (step / 2) ** 2)
    cov_after = cov.copy()
    for (sd, v), var in zip(dirs, (var_a, var_b)):
        cov_after += (var * sd ** 2 - sd ** 2) * np.outer(v, v)
    info = {"applied": True, "method": "weak-direction grid search, rendered shading edges",
            "click_sd_mm_weakest": float(dirs[0][0]), "a_sigma": float(a_best), "b_sigma": float(b_best),
            "edge_score_before": s0, "edge_score_after": s1, "edge_points": n1,
            "click_rms_px_before": c0, "click_rms_px_after": c1, "roi": list(roi), "cov_after": cov_after.tolist()}
    reasons = []
    if s1 - s0 < min_gain:
        reasons.append(f"edge score gain {s1 - s0:.4f} < {min_gain}")
    if c1 > max(c0 + 2.0, 2.0 * c0):
        reasons.append(f"click RMS grew {c0:.2f} -> {c1:.2f} px")
    if max(abs(a_best), abs(b_best)) >= span_sigma + step:
        reasons.append("optimum at the search border")
    if reasons:
        log.warning("registration refinement not applied (%s); keeping the click-based X", "; ".join(reasons))
        info.update(applied=False, reason="; ".join(reasons))
        return X0, info
    log.info("registration refined along the weak click directions: %.2f / %.2f sd (weakest sd %.1f mm-equiv.); "
             "edge score %.3f -> %.3f (%d edge px); click RMS %.2f -> %.2f px",
             a_best, b_best, dirs[0][0], s0, s1, n1, c0, c1)
    return X, info


def board_polygons(T_C_H, T_C_J, K, dist, print_scale=1.0, margin_mm=6.0):
    """Image polygons of the HEAD and JAW plates (+ margin) to mask out of the edge map."""
    from .boards import HEAD, JAW, board_size_mm
    polys = []
    for spec, T in ((HEAD, T_C_H), (JAW, T_C_J)):
        if T is None:
            continue
        w, h = board_size_mm(spec, print_scale)
        m = margin_mm
        corners = np.array([[-m, -m, 0], [w + m, -m, 0], [w + m, h + m, 0], [-m, h + m, 0]], float)
        polys.append(_project(K, dist, T, corners)[0])
    return polys
