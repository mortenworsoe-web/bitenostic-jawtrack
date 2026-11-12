"""Tiny flat-shaded mesh rasteriser (painter's algorithm + cv2.fillConvexPoly) with optional
lens distortion (vertices projected with cv2.projectPoints)."""
import numpy as np
import cv2


def render_mesh(img, verts_cam, faces, K, dist, color_fn=None, base_color=(200, 220, 235),
                light_dir=(0.3, -0.5, -1.0), ambient=0.35, zbuf=None, shift=4):
    """verts_cam: (N,3) in camera frame (OpenCV: z forward). Draws in-place on img (BGR uint8).
    color_fn(face_centers_cam_idx) -> (F,3) BGR colours, optional. Returns per-face drawn mask count."""
    V = np.asarray(verts_cam, np.float64)
    F = np.asarray(faces)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    cen = tri.mean(1)
    front = (nn * cen).sum(1) < 0  # facing camera
    vis = front & (cen[:, 2] > 1)
    px, _ = cv2.projectPoints(V.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    px = px.reshape(-1, 2)
    L = np.asarray(light_dir, float)
    L = L / np.linalg.norm(L)
    lam = np.clip(nn @ L, 0, 1) ** 0.8
    shade = ambient + (1 - ambient) * lam
    if color_fn is not None:
        cols = np.asarray(color_fn(np.arange(len(F))), float)
    else:
        cols = np.tile(np.asarray(base_color, float), (len(F), 1))
    cols = np.clip(cols * shade[:, None], 0, 255).astype(np.uint8)
    order = np.where(vis)[0]
    order = order[np.argsort(-cen[order, 2])]
    P = np.round(px * (1 << shift)).astype(np.int32)
    for f in order:
        cv2.fillConvexPoly(img, P[F[f]], tuple(int(c) for c in cols[f]), lineType=cv2.LINE_AA, shift=shift)
    return img
