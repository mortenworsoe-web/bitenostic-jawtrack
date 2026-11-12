"""Scene, rig, camera and video rendering for the synthetic datasets."""
import json
import os
import subprocess
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import gtdefs as G
from .raster import render_mesh

DICT_BOARDS = cv2.aruco.DICT_4X4_50
MARKER_MM, GAP_MM = 15.0, 3.0
HEAD_GRID, HEAD_IDS = (3, 2), [0, 1, 2, 3, 4, 5]
JAW_GRID, JAW_IDS = (2, 2), [10, 11, 12, 13]
QUIET_MM, PLATE_MM = 4.0, 2.5   # white quiet zone around the grid, grey plastic rim
TEX_PPM = 12.0                  # texture pixels per mm for boards


def flat(T):
    return [float(x) for x in np.asarray(T).reshape(-1)]


def tolist(v):
    return [float(x) for x in np.asarray(v).reshape(-1)]


def Rx(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def HT(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# board frame (OpenCV GridBoard objPoints): origin top-left corner of top-left marker, x right,
# y DOWN (row index increases), z = x cross y = INTO the board (away from the viewer).
R_BOARD_IN_S = np.diag([1.0, -1.0, -1.0])  # board x -> patient's left, y -> inferior, z -> posterior


# ============================================================================ textures
class PlanarTexture:
    """A textured plane: texture BGR + alpha, and A (3x3) texel(u,v,1) -> plane mm (X,Y,1)."""

    def __init__(self, bgr, alpha, A):
        self.bgr = bgr
        self.alpha = alpha.astype(np.float32)
        self.A = A


def make_board_texture(grid, ids, dictionary):
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.GridBoard(grid, MARKER_MM, GAP_MM, d, np.array(ids))
    bw = grid[0] * MARKER_MM + (grid[0] - 1) * GAP_MM
    bh = grid[1] * MARKER_MM + (grid[1] - 1) * GAP_MM
    q = int(round(QUIET_MM * TEX_PPM))
    size = (int(round(bw * TEX_PPM)) + 2 * q, int(round(bh * TEX_PPM)) + 2 * q)
    img = board.generateImage(size, marginSize=q, borderBits=1)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, dids, _ = det.detectMarkers(img)
    obj = board.getObjPoints()
    bid = list(board.getIds().ravel())
    src, dst = [], []
    for c, i in zip(corners, dids.ravel()):
        src.append(c.reshape(4, 2))
        dst.append(np.asarray(obj[bid.index(int(i))])[:, :2])
    src, dst = np.vstack(src), np.vstack(dst)
    assert len(dids) == len(ids), "board texture detection failed"
    # analytic texel -> mm: generateImage scales the board by exactly TEX_PPM here; pixel centres
    # are integer coordinates, so board coordinate 0 lies on the pixel edge at q - 0.5
    A = np.array([[1 / TEX_PPM, 0, -(q - 0.5) / TEX_PPM], [0, 1 / TEX_PPM, -(q - 0.5) / TEX_PPM], [0, 0, 1.0]])
    Xs = np.hstack([src, np.ones((len(src), 1))])
    resid = np.abs((Xs @ A.T)[:, :2] - dst)
    assert resid.mean() < 0.06 and resid.max() < 0.15, resid.max()
    # add grey plastic rim
    p = int(round(PLATE_MM * TEX_PPM))
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(150, 152, 155))
    T = np.array([[1, 0, -p], [0, 1, -p], [0, 0, 1.0]])  # padded texel -> original texel
    A = A @ T
    bgr = cv2.GaussianBlur(bgr, (0, 0), 0.7)
    alpha = np.ones(bgr.shape[:2], np.float32)
    # rounded corners
    r = int(3 * TEX_PPM)
    hh, ww = alpha.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    for cx, cy in ((r, r), (ww - r, r), (r, hh - r), (ww - r, hh - r)):
        m = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r) & (((xx < r) | (xx > ww - r)) & ((yy < r) | (yy > hh - r)))
        alpha[m & ((xx - cx) ** 2 + (yy - cy) ** 2 > r * r)] = 0
    return PlanarTexture(bgr, alpha, A), board, (bw, bh)


def _noise(shape, rng, sigma, scale):
    n = rng.standard_normal((shape[0] // scale + 2, shape[1] // scale + 2)).astype(np.float32)
    n = cv2.resize(n, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(n, (0, 0), scale / 2.0) * sigma


def make_face_textures(rng, ppm=3.0):
    """Upper face (head frame) and chin (lower-jaw frame) textures. Plane frames like boards:
    x -> patient's left, y -> inferior (texel rows), origin at top-left."""
    skin = np.array([120, 150, 200], np.float32)  # BGR
    # ---- upper face: x in [-95, 95], y_S in [130, -110]
    x0, x1, ytop, ybot = -95.0, 95.0, 135.0, -110.0
    w, h = int((x1 - x0) * ppm), int((ytop - ybot) * ppm)
    X, Y = np.meshgrid(x0 + (np.arange(w) + 0.5) / ppm, ytop - (np.arange(h) + 0.5) / ppm)
    img = np.zeros((h, w, 3), np.float32) + skin
    shade = 1.0 - 0.35 * (np.abs(X) / 85.0) ** 2.5
    img *= shade[..., None]
    img += _noise((h, w), rng, 6, 6)[..., None]
    head = ((X / 78.0) ** 2 + ((Y - 30) / 115.0) ** 2) < 1.0
    neck = (np.abs(X) < 48) & (Y < -40)
    alpha = (head | neck).astype(np.float32)
    hair = head & (Y > 105 - 0.004 * X ** 2)
    img[hair] = np.array([40, 50, 65]) + _noise((h, w), rng, 8, 3)[hair][:, None]
    for sx in (-1, 1):  # eyes, brows
        eye = (((X - sx * 32) / 14.0) ** 2 + ((Y - 52) / 5.5) ** 2) < 1
        img[eye] = (230, 235, 240)
        iris = ((X - sx * 32) ** 2 + (Y - 52) ** 2) < 4.5 ** 2
        img[iris] = (60, 70, 50)
        brow = (np.abs(X - sx * 32) < 17) & (np.abs(Y - 64 - 0.02 * (X - sx * 32) ** 2 * 0) < 2.2)
        img[brow] = (45, 55, 70)
    nose = (np.abs(X) < 14) & (Y > 14) & (Y < 50)
    img[nose] *= (0.9 + 0.1 * np.cos(X[nose] / 14 * np.pi / 2))[:, None]
    nost = (((np.abs(X) - 6) / 4.0) ** 2 + ((Y - 17) / 2.0) ** 2) < 1
    img[nost] = (60, 70, 100)
    lip_u = (((X / 26.0) ** 2 + ((Y - 6) / 7.0) ** 2) < 1) & (Y > 0)
    img[lip_u] = (95, 95, 175)
    below = Y < 0
    cav = below & (((X / 27.0) ** 2 + ((Y + 18) / 30.0) ** 2) < 1)
    img[below] *= 0.85
    img[cav] = (35, 30, 45)
    up = PlanarTexture(np.clip(img, 0, 255).astype(np.uint8), alpha, np.array([[1 / ppm, 0, x0], [0, 1 / ppm, -ytop], [0, 0, 1]]))
    up.z_S = 6.0  # plane depth in S (anterior)
    # ---- chin plane (lower jaw frame): x in [-70,70], y in [0,-85]
    x0c, ytopc, ybotc = -70.0, 0.0, -88.0
    w, h = int(140 * ppm), int((ytopc - ybotc) * ppm)
    X, Y = np.meshgrid(x0c + (np.arange(w) + 0.5) / ppm, ytopc - (np.arange(h) + 0.5) / ppm)
    img = np.zeros((h, w, 3), np.float32) + skin
    img *= (1.0 - 0.35 * (np.abs(X) / 70.0) ** 2.5)[..., None]
    img += _noise((h, w), rng, 6, 6)[..., None]
    jaw = (((X / 62.0) ** 2 + ((Y + 0) / 85.0) ** 2) < 1) & (Y < -0.5)
    lip_l = (((X / 25.0) ** 2 + ((Y + 0.5) / 9.0) ** 2) < 1) & (Y < -0.5)
    img[lip_l] = (90, 90, 170)
    chin = PlanarTexture(np.clip(img, 0, 255).astype(np.uint8), jaw.astype(np.float32),
                         np.array([[1 / ppm, 0, x0c], [0, 1 / ppm, -ytopc], [0, 0, 1]]))
    chin.z_S = 8.0
    return up, chin


def make_wall(size, rng):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 175 - 35 * (yy / h) + 10 * np.sin(xx / w * 3)
    img = np.stack([base * 0.95, base * 0.98, base * 1.0], -1)
    img += _noise((h, w), rng, 5, 24)[..., None] + _noise((h, w), rng, 2, 3)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================================ camera
class Camera:
    def __init__(self, K, dist, size):
        self.K, self.dist, self.size = np.asarray(K, float), np.asarray(dist, float), size
        w, h = size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        und = cv2.undistortPoints(pts, self.K, self.dist, R=None, P=self.K, criteria=crit).reshape(h, w, 2)
        self.mapx = und[..., 0].astype(np.float32)
        self.mapy = und[..., 1].astype(np.float32)
        # verify inverse mapping
        sel = np.random.default_rng(0).choice(w * h, 2000, replace=False)
        idl = und.reshape(-1, 2)[sel]
        nrm = np.hstack([(idl - self.K[:2, 2]) / [self.K[0, 0], self.K[1, 1]], np.ones((len(sel), 1))])
        rep, _ = cv2.projectPoints(nrm.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        self.map_err = float(np.abs(rep.reshape(-1, 2) - np.stack([u.ravel()[sel], v.ravel()[sel]], -1)).max())

    def project(self, P_cam):
        p, _ = cv2.projectPoints(np.asarray(P_cam, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return p.reshape(-1, 2)

    def ideal_coords(self, x0, y0, x1, y1, ss=1):
        """Ideal (undistorted) pixel coords of a distorted-image ROI sampled ss x ss per pixel."""
        if ss == 1:
            return self.mapx[y0:y1, x0:x1], self.mapy[y0:y1, x0:x1]
        xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss - 0.5
        ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss - 0.5
        gx, gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        return (cv2.remap(self.mapx, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                cv2.remap(self.mapy, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def warp_plane(cam, tex, T_C_P, roi=None, ss=1):
    """Render textured plane (pose T_C_P: plane mm -> camera) into ROI of the distorted image.
    Returns (bgr float32, alpha float32, roi)."""
    w, h = cam.size
    if roi is None:
        hh, ww = tex.alpha.shape
        cs = np.array([[0, 0, 1], [ww, 0, 1], [ww, hh, 1], [0, hh, 1]], float) @ tex.A.T
        P = (T_C_P[:3, :3] @ np.c_[cs[:, :2], np.zeros(4)].T).T + T_C_P[:3, 3]
        if (P[:, 2] < 10).any():
            return None
        px = cam.project(P)
        x0, y0 = np.floor(px.min(0)).astype(int) - 3
        x1, y1 = np.ceil(px.max(0)).astype(int) + 4
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = (x0, y0, x1, y1)
    x0, y0, x1, y1 = roi
    ix, iy = cam.ideal_coords(x0, y0, x1, y1, ss)
    Hm = cam.K @ np.c_[T_C_P[:3, 0], T_C_P[:3, 1], T_C_P[:3, 3]]
    Hinv = np.linalg.inv(Hm @ tex.A)
    den = Hinv[2, 0] * ix + Hinv[2, 1] * iy + Hinv[2, 2]
    mx = ((Hinv[0, 0] * ix + Hinv[0, 1] * iy + Hinv[0, 2]) / den).astype(np.float32)
    my = ((Hinv[1, 0] * ix + Hinv[1, 1] * iy + Hinv[1, 2]) / den).astype(np.float32)
    col = cv2.remap(tex.bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    al = cv2.remap(tex.alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if ss > 1:
        sz = (x1 - x0, y1 - y0)
        col = cv2.resize(col * al[..., None], sz, interpolation=cv2.INTER_AREA)
        al = cv2.resize(al, sz, interpolation=cv2.INTER_AREA)
        col = col / np.maximum(al[..., None], 1e-6)
    return col, al, roi


def composite(img, layer):
    if layer is None:
        return
    col, al, (x0, y0, x1, y1) = layer
    a = al[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col * a


# ============================================================================ motion
class HeadMotion:
    """Smooth, seeded head motion relative to the camera: rotations about a pivot near the
    atlanto-occipital joint, translations, nodding, and a small extension when opening wide."""

    def __init__(self, rng, duration, amp_rot=9.0, amp_tr=18.0):
        self.terms = []
        for k in range(6):  # yaw, pitch, roll, tx, ty, tz
            amp = amp_rot if k < 3 else amp_tr
            if k == 2:
                amp *= 0.5
            if k == 5:
                amp *= 0.7
            fs = rng.uniform(0.03, 0.12), rng.uniform(0.12, 0.3), rng.uniform(0.3, 0.6)
            ws = np.array([0.65, 0.3, 0.08])
            ph = rng.uniform(0, 2 * np.pi, 3)
            self.terms.append((amp, fs, ws, ph))
        self.nod = (rng.uniform(0.35, 0.5), rng.uniform(0, 2 * np.pi))
        self.pivot = np.array([0.0, 5.0, -85.0])
        self.phi_fn = None

    def _sig(self, k, t):
        amp, fs, ws, ph = self.terms[k]
        return amp * sum(w * np.sin(2 * np.pi * f * t + p) for f, w, p in zip(fs, ws, ph))

    def __call__(self, t):
        yaw, pitch, roll = (self._sig(k, t) for k in range(3))
        # nodding bursts
        f, p = self.nod
        pitch += 3.0 * np.sin(2 * np.pi * f * t + p) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t))
        if self.phi_fn is not None:
            pitch -= 5.0 * self.phi_fn(t)  # head extension (chin up) when opening wide
        R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
        tr = np.array([self._sig(3, t), self._sig(4, t), self._sig(5, t)])
        return HT(R, self.pivot - R @ self.pivot + tr)


def look_at_T_C_S(cam_pos_S, target_S):
    z = target_S - cam_pos_S
    z /= np.linalg.norm(z)
    ydown = np.array([0, -1.0, 0])
    y = ydown - z * (ydown @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z])
    return HT(R, -R @ cam_pos_S)


class JawInterp:
    """Interpolates T_S between frames (sub-frame motion blur)."""

    def __init__(self, times, Ts):
        self.times, self.Ts = times, Ts
        self.slerp = Slerp(times, Rotation.from_matrix(Ts[:, :3, :3]))

    def __call__(self, t):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        R = self.slerp(t).as_matrix()
        tr = np.array([np.interp(t, self.times, self.Ts[:, k, 3]) for k in range(3)])
        return HT(R, tr)


# ============================================================================ rig
def rig_transforms(head_size, jaw_size, lm):
    """T_S_H: HEAD board -> STL(upper) frame. T_L_J: JAW board -> lower-STL frame (in MI)."""
    bw, bh = head_size
    R_SH = Rz(2.0) @ Rx(15.0) @ R_BOARD_IN_S  # tilted: normal points anterior-inferior to the camera
    cH = np.array([0.0, 76.0, 10.0])
    T_S_H = HT(R_SH, cH - R_SH @ np.array([bw / 2, bh / 2, 0]))
    jw, jh = jaw_size
    R_LJ = Rz(-1.5) @ Rx(10.0) @ R_BOARD_IN_S
    cJ = np.array([0.0, -37.0, 38.0])
    T_L_J = HT(R_LJ, cJ - R_LJ @ np.array([jw / 2, jh / 2, 0]))
    return T_S_H, T_L_J


def box_mesh(p0, p1, width, thick):
    """Thin rectangular bar from p0 to p1 (stem)."""
    import trimesh
    d = p1 - p0
    L = np.linalg.norm(d)
    b = trimesh.creation.box(extents=(width, thick, L))
    z = d / L
    x = np.array([1.0, 0, 0])
    y = np.cross(z, x); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    b.apply_transform(HT(np.stack([x, y, z], 1), 0.5 * (p0 + p1)))
    return b


# ============================================================================ frame renderer
class Renderer:
    def __init__(self, cam, rng, lm, U, L, size):
        self.cam = cam
        self.rng = rng
        self.size = size
        self.head_tex, self.head_board, self.head_size = make_board_texture(HEAD_GRID, HEAD_IDS, DICT_BOARDS)
        self.jaw_tex, self.jaw_board, self.jaw_size = make_board_texture(JAW_GRID, JAW_IDS, DICT_BOARDS)
        self.face_up, self.face_chin = make_face_textures(rng)
        wall = make_wall(size, rng)
        self.wall = cv2.remap(wall, cam.mapx, cam.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        self.T_S_H, self.T_L_J = rig_transforms(self.head_size, self.jaw_size, lm)
        # stem from the jaw board's top edge to the labial surface of the lower incisors
        top_mid = (self.T_L_J @ np.array([self.jaw_size[0] / 2, -QUIET_MM - PLATE_MM, 0, 1]))[:3]
        ip = lm["lower_incisal_point"]
        self.stem = box_mesh(top_mid + np.array([0, 0, 0.0]), np.array([0.0, ip[1] - 7.0, ip[2] + 2.0]), 6.0, 2.5)
        self.U, self.L = U, L
        self.noise_bank = [rng.normal(0, 1, (size[1], size[0], 3)).astype(np.float32) for _ in range(4)]
        yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
        r2 = ((xx - size[0] / 2) ** 2 + (yy - size[1] / 2) ** 2) / (size[0] / 2) ** 2
        self.vignette = (1 - 0.18 * r2)[..., None]
        self.objpts = {"head": self.head_board.getObjPoints(), "jaw": self.jaw_board.getObjPoints()}
        self.occluders = []

    def face_pose(self, T_C_X, tex):
        # plane frame: x -> +x_S, y -> -y_S, z -> -z_S, origin at plane top-left mm (A handles)
        return T_C_X @ HT(R_BOARD_IN_S, np.array([0, 0, tex.z_S]))

    def board_poses(self, T_C_S, T_S):
        return T_C_S @ self.T_S_H, T_C_S @ T_S @ self.T_L_J

    def marker_px(self, T_C_B, which):
        out = []
        for c in self.objpts[which]:
            P = (T_C_B[:3, :3] @ np.asarray(c, float).T).T + T_C_B[:3, 3]
            out.append(self.cam.project(P))
        return out

    def render(self, i, t, T_C_S, T_S, sub_poses, gain, frame_noise=True, teeth=False, occl=None):
        cam = self.cam
        img = self.wall.copy()
        # faces (at frame centre time)
        T_C_L = T_C_S @ T_S
        composite(img, warp_plane(cam, self.face_up, self.face_pose(T_C_S, self.face_up), roi=(0, 0) + tuple(self.size)))
        if teeth:
            self._draw_teeth(img, T_C_S)
        else:
            composite(img, warp_plane(cam, self.face_chin, self.face_pose(T_C_L, self.face_chin)))
        # stem (moves with jaw)
        V = (T_C_L[:3, :3] @ self.stem.vertices.T).T + T_C_L[:3, 3]
        tmp = img.astype(np.uint8)
        render_mesh(tmp, V, self.stem.faces, cam.K, cam.dist, base_color=(150, 152, 155))
        img = tmp.astype(np.float32)
        # boards with motion blur: average over sub-exposure samples
        for which, tex in (("head", self.head_tex), ("jaw", self.jaw_tex)):
            layers = []
            for (TCS, TS) in sub_poses:
                TH, TJ = self.board_poses(TCS, TS)
                layers.append(warp_plane(cam, tex, TH if which == "head" else TJ, ss=2))
            layers = [l for l in layers if l is not None]
            if not layers:
                continue
            x0 = min(l[2][0] for l in layers); y0 = min(l[2][1] for l in layers)
            x1 = max(l[2][2] for l in layers); y1 = max(l[2][3] for l in layers)
            acc = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            aacc = np.zeros((y1 - y0, x1 - x0), np.float32)
            for col, al, (a0, b0, a1, b1) in layers:
                acc[b0 - y0:b1 - y0, a0 - x0:a1 - x0] += col * al[..., None]
                aacc[b0 - y0:b1 - y0, a0 - x0:a1 - x0] += al
            n = len(layers)
            al = aacc / n
            col = acc / np.maximum(aacc[..., None], 1e-6)
            # boards are lit a little differently than the face
            col *= 1.0 if which == "head" else 0.96
            composite(img, (col, al, (x0, y0, x1, y1)))
        # occluder (lip / finger) over a marker corner, drawn in the board plane
        if occl is not None:
            which, poly_mm = occl
            TH, TJ = self.board_poses(T_C_S, T_S)
            TB = TH if which == "head" else TJ
            P = (TB[:3, :3] @ np.c_[poly_mm, np.full(len(poly_mm), -1.0)].T).T + TB[:3, 3]
            px = cam.project(P)
            ov = img.copy()
            cv2.fillPoly(ov, [np.round(px * 16).astype(np.int32)], (110, 120, 185), lineType=cv2.LINE_AA, shift=4)
            img = cv2.GaussianBlur(ov, (0, 0), 0.8) * 0.5 + ov * 0.5
        img *= gain * self.vignette
        if frame_noise:
            nz = self.noise_bank[i % len(self.noise_bank)]
            sh = (i * 7919) % 97
            img += np.roll(nz, (sh, 3 * sh), axis=(0, 1)) * (1.5 + 0.012 * img)
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)

    def _draw_teeth(self, img, T_C_S):
        """Registration still: cheek retractor opening, teeth in MI rendered with a painter's
        rasteriser (flat shading) into the mouth opening."""
        cam = self.cam
        h, w = img.shape[:2]
        layer = np.full((h, w, 3), (30, 25, 40), np.uint8)
        for M, col in ((self.L, (205, 222, 236)), (self.U, (212, 228, 240))):
            V = (T_C_S[:3, :3] @ M.vertices.T).T + T_C_S[:3, 3]
            ys = M.vertices[:, 1]
            fy = ys[M.faces].mean(1)
            gum = (fy > 8.5) if M is self.U else (fy < -6.5)

            def cfn(idx, gum=gum, col=col):
                c = np.tile(np.array(col, float), (len(idx), 1))
                c[gum[idx]] = (120, 120, 205)
                return c
            render_mesh(layer, V, M.faces, cam.K, cam.dist, color_fn=cfn, light_dir=(0.2, -0.4, -1.0))
        # retractor opening (ellipse in the lip plane, head frame)
        ang = np.linspace(0, 2 * np.pi, 120)
        ell = np.c_[33 * np.cos(ang), 1.0 + 17 * np.sin(ang), np.full_like(ang, 7.0)]
        P = (T_C_S[:3, :3] @ ell.T).T + T_C_S[:3, 3]
        px = cam.project(P)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(px * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
        m = (mask.astype(np.float32) / 255)[..., None]
        img[:] = img * (1 - m) + layer.astype(np.float32) * m
        cv2.polylines(img, [np.round(px * 16).astype(np.int32)], True, (215, 225, 230), 7, cv2.LINE_AA, shift=4)


# ============================================================================ dataset writer
def ffmpeg_writer(path, size, fps):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(int(fps)), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def visible_in_image(px_list, size, margin=2):
    w, h = size
    return [bool(((p[:, 0] >= margin) & (p[:, 0] <= w - 1 - margin) & (p[:, 1] >= margin) & (p[:, 1] <= h - 1 - margin)).all())
            for p in px_list]


def make_occlusion_events(rng, n_frames, fps, segs):
    """A few short partial occlusions of one marker corner (lip / finger)."""
    ev = []
    cand = [("jaw", 12, 2), ("jaw", 13, 3), ("head", 0, 0), ("jaw", 10, 1)]
    k = max(1, int(n_frames / fps / 8))
    starts = np.sort(rng.choice(np.arange(int(1 * fps), n_frames - int(1 * fps)), k, replace=False))
    for j, s in enumerate(starts):
        which, mid, corner = cand[j % len(cand)]
        dur = int(rng.integers(6, 16))
        ev.append(dict(board=which, marker_id=mid, corner=corner, f0=int(s), f1=int(min(s + dur, n_frames - 1))))
    return ev


def occluder_poly(rend, ev):
    """Polygon (board mm) covering ~35% of a marker around one of its corners."""
    obj = rend.objpts[ev["board"]]
    ids = HEAD_IDS if ev["board"] == "head" else JAW_IDS
    c = np.asarray(obj[ids.index(ev["marker_id"])])[:, :2]
    corner = c[ev["corner"]]
    centre = c.mean(0)
    d = centre - corner
    ang = np.linspace(0, 2 * np.pi, 24)
    r = 0.5 * MARKER_MM
    poly = corner - 0.15 * d + np.c_[r * np.cos(ang), 0.8 * r * np.sin(ang)]
    return poly


def write_dataset(out, variant, U, L, lm, kin, P, occ, seed, K, dist, fps, size, skip_video=False, skip_calib=False):
    rng = np.random.default_rng(seed)
    t_start = time.time()
    U.export(os.path.join(out, "upper.stl"))
    L.export(os.path.join(out, "lower.stl"))
    times, Ts = kin["times"], kin["Ts"]
    n = len(times)
    cam = Camera(K, dist, size)
    print(f"    camera distortion map max reprojection error {cam.map_err:.2e} px")
    rend = Renderer(cam, rng, lm, U, L, size)
    head = HeadMotion(rng, times[-1])
    phis = kin["phis"]
    phimax = max(phis.max(), 1e-6)
    head.phi_fn = lambda t: np.interp(t, times, phis) / phimax
    cam_pos = np.array([4.0, -55.0, 360.0])
    T_C_S0 = look_at_T_C_S(cam_pos, np.array([0.0, 18.0, 0.0]))
    T_C_S = np.array([T_C_S0 @ head(t) for t in times])
    jaw = JawInterp(times, Ts)
    T_C_H = np.array([T_C_S[i] @ rend.T_S_H for i in range(n)])
    T_C_J = np.array([T_C_S[i] @ Ts[i] @ rend.T_L_J for i in range(n)])
    segs = kin["segments"]
    occl_events = make_occlusion_events(rng, n, fps, segs)
    # registration frame: middle of the first MI hold
    reg_i = int(round(segs[0]["t1"] * 0.5 * fps))
    gains = 1.0 + 0.035 * np.sin(2 * np.pi * 0.11 * times + rng.uniform(0, 6)) + 0.015 * np.sin(2 * np.pi * 0.9 * times)
    # visibility stats
    vis_rows = []
    all_in = 0
    for i in range(n):
        vh = visible_in_image(rend.marker_px(T_C_H[i], "head"), size)
        vj = visible_in_image(rend.marker_px(T_C_J[i], "jaw"), size)
        occ_ids = [e["marker_id"] for e in occl_events if e["f0"] <= i <= e["f1"]]
        vis_rows.append((vh, vj, occ_ids))
        all_in += all(vh) and all(vj)
    frac_in = all_in / n
    print(f"    frames with all 10 markers fully inside the image: {100 * frac_in:.1f}%")
    exposure = 1.0 / 120.0

    def sub_poses(i):
        # number of sub-samples proportional to the image-space speed of the jaw/head boards
        t = times[i]
        a, b = max(0, i - 1), min(n - 1, i + 1)
        pj = rend.marker_px(T_C_J[a], "jaw")[0]
        pj2 = rend.marker_px(T_C_J[b], "jaw")[0]
        ph = rend.marker_px(T_C_H[a], "head")[0]
        ph2 = rend.marker_px(T_C_H[b], "head")[0]
        speed = max(np.abs(pj2 - pj).max(), np.abs(ph2 - ph).max()) / ((b - a) / fps)  # px/s
        blur_px = speed * exposure
        ns = int(np.clip(np.ceil(blur_px / 0.6), 1, 9))
        if ns == 1:
            return [(T_C_S[i], Ts[i])], blur_px
        offs = (np.arange(ns) + 0.5) / ns - 0.5
        return [(T_C_S0 @ head(t + o * exposure), jaw(t + o * exposure)) for o in offs], blur_px

    if not skip_video:
        vpath = os.path.join(out, "video.mp4")
        proc = ffmpeg_writer(vpath, size, fps)
        blur_list = []
        t0 = time.time()
        for i in range(n):
            sp, bp = sub_poses(i)
            blur_list.append(bp)
            occl = None
            for e in occl_events:
                if e["f0"] <= i <= e["f1"]:
                    occl = (e["board"], occluder_poly(rend, e))
            frame = rend.render(i, times[i], T_C_S[i], Ts[i], sp, gains[i], occl=occl)
            proc.stdin.write(frame.tobytes())
            if i == reg_i:
                cv2.imwrite(os.path.join(out, "_video_frame_reg.png"), frame)
            if i % 120 == 0:
                el = time.time() - t0
                print(f"    video frame {i}/{n}  ({el:.0f}s elapsed, ~{el / max(i, 1) * (n - i):.0f}s left)")
        proc.stdin.close()
        proc.wait()
        print(f"    max motion blur {max(blur_list):.1f} px")
    # registration still (teeth visible, same pose as video frame reg_i, no blur)
    reg = rend.render(reg_i, times[reg_i], T_C_S[reg_i], Ts[reg_i], [(T_C_S[reg_i], Ts[reg_i])], gains[reg_i], teeth=True)
    cv2.imwrite(os.path.join(out, "reg_frame.png"), reg)
    lmk = registration_landmarks(out, rend, cam, U, L, lm, T_C_S[reg_i], reg_i, size, rng)
    # ground truth json
    tl = kin["tl"]
    fr = tl["frame"]
    X = np.linalg.inv(rend.T_S_H)
    gt = {
        "version": 1, "units": "mm", "fps": fps, "n_frames": n, "variant": variant,
        "frame_of_reference": "upper STL coordinates (x = patient's left, y = superior, z = anterior); T maps lower STL (as exported in MI) to its pose at time t",
        "matrix_convention": "4x4 row-major, T[0..3] = first row; maps column vectors p' = T p",
        "files": {"video": "video.mp4", "upper_stl": "upper.stl", "lower_stl": "lower.stl", "reg_frame": "reg_frame.png"},
        "camera": {"K": K.tolist(), "dist": tolist(dist), "width": size[0], "height": size[1], "model": "OpenCV (k1,k2,p1,p2,k3)"},
        "frames": [{"i": i, "t": round(float(times[i]), 6), "T": flat(Ts[i])} for i in range(n)],
        "segments": segs,
        "landmarks_mm": {
            "incisal_point": tolist(tl["incisal_point"]),
            "condyle_left": tolist(tl["condyle_left"]), "condyle_right": tolist(tl["condyle_right"]),
            "occlusal_plane": {"point": tolist(fr["origin"]), "normal": tolist(fr["n"])},
            "occlusal_plane_points": [tolist(tl["incisal_point"]), tolist(lm["36_mb"]), tolist(lm["46_mb"])],
            "occlusal_frame": {"f_anterior": tolist(fr["f"]), "l_left": tolist(fr["l"]), "n_superior": tolist(fr["n"])},
            "condyle_method": "average_bonwill (also the exact rotation centres of the synthetic kinematics)",
            "upper_incisal_point": tolist(lm["upper_incisal_point"]),
            "anatomical": {k: tolist(v) for k, v in lm.items() if k[0].isdigit()},
            "lower_molar_cusp_points": {"36_mb": tolist(lm["36_mb"]), "46_mb": tolist(lm["46_mb"])},
        },
        "parameters": P,
        "model_inputs": {"sagittal_condylar_inclination_left_deg": 35.0, "sagittal_condylar_inclination_right_deg": 38.0,
                         "bennett_angle_left_deg": 10.0, "bennett_angle_right_deg": 8.0,
                         "eminence_path_curvature_per_mm": kin["model"].curv,
                         "working_condyle_lateral_shift_per_mm_orbit": kin["model"].side_shift,
                         "amplitudes": {k: float(v) for k, v in kin["amps"].items()}},
        "rig": {
            "board_convention": "OpenCV GridBoard objPoints: origin = top-left corner of top-left marker, x right, y down (rows), z = x cross y = INTO the board (away from the camera). Units mm.",
            "head_board": {"dictionary": "DICT_4X4_50", "grid": list(HEAD_GRID), "ids": HEAD_IDS, "marker_mm": MARKER_MM, "gap_mm": GAP_MM},
            "jaw_board": {"dictionary": "DICT_4X4_50", "grid": list(JAW_GRID), "ids": JAW_IDS, "marker_mm": MARKER_MM, "gap_mm": GAP_MM},
            "T_S_H": flat(rend.T_S_H), "X_T_H_S": flat(X),
            "T_L_J": flat(rend.T_L_J),
            "T_L_J_note": "JAW board pose in the lower-STL frame (MI). M0 = X @ T_L_J; M(t) = X @ T_S(t) @ T_L_J",
        },
        "registration": {"frame_index": reg_i, "t": float(times[reg_i]), "file": "reg_frame.png",
                         "T_C_S_at_reg": flat(T_C_S[reg_i]), "landmark_files": ["registration_landmarks.json", "registration_landmarks_exact.json"],
                         "landmark_noise_px_sigma": 1.5},
        "occlusions": occl_events,
        "rendering": {"exposure_s": exposure, "motion_blur": "board layers averaged over sub-exposure samples (count ~ image speed)",
                      "noise": "gaussian ~1.5-4.5 DN + brightness drift +-4% + vignetting", "codec": "H.264 CRF 18 veryfast yuv420p",
                      "rolling_shutter": False, "frames_all_markers_in_image_fraction": frac_in},
        "verification": {"occlusion_in_MI": occ},
    }
    # self-checks
    gt["verification"].update(self_checks(kin, lm, P, tl, frac_in, U, L))
    json.dump(gt, open(os.path.join(out, "ground_truth.json"), "w"), indent=1)
    json.dump(G.gt_to_motion(gt), open(os.path.join(out, "motion_true.json"), "w"))
    # raw tracking truth (SPEC 5.1 format) for stage-1 validation
    raw = {"version": 1, "video": "video.mp4", "fps": fps, "width": size[0], "height": size[1],
           "camera": {"K": K.tolist(), "dist": tolist(dist), "source": "synthetic_truth"},
           "frames": []}
    for i in range(n):
        vh, vj, occ_ids = vis_rows[i]
        nh = sum(v and (mid not in occ_ids) for v, mid in zip(vh, HEAD_IDS))
        nj = sum(v and (mid not in occ_ids) for v, mid in zip(vj, JAW_IDS))
        raw["frames"].append({"i": i, "t": round(float(times[i]), 6),
                              "head": {"T": flat(T_C_H[i]), "n_markers": nh, "reproj_px": 0.0},
                              "jaw": {"T": flat(T_C_J[i]), "n_markers": nj, "reproj_px": 0.0}})
    json.dump(raw, open(os.path.join(out, "raw_tracking_true.json"), "w"))
    json.dump({"camera": {"K": K.tolist(), "dist": tolist(dist), "source": "synthetic_truth"},
               "width": size[0], "height": size[1], "fps": fps}, open(os.path.join(out, "camera_true.json"), "w"), indent=1)
    if not skip_calib:
        render_charuco_clip(os.path.join(out, "charuco_calib.mp4"), cam, rng, fps, size,
                            duration=8.0 if variant == "quick" else 20.0)
    preview_images(out, U, L, Ts, times, segs)
    print(f"    {out} written in {time.time() - t_start:.0f}s")


def registration_landmarks(out, rend, cam, U, L, lm, T_C_S, reg_i, size, rng):
    import trimesh
    names = ["11_incisal", "21_incisal", "13_cusp", "23_cusp", "16_mb", "26_mb", "33_cusp", "43_cusp"]
    both = trimesh.util.concatenate([U, L])
    Rinv = T_C_S[:3, :3].T
    cam_S = -Rinv @ T_C_S[:3, 3]  # camera centre in S
    pts = []
    for nm in names:
        p = lm[nm]
        d = p - cam_S
        dist = np.linalg.norm(d)
        locs, _, _ = both.ray.intersects_location([cam_S], [d / dist])
        first = np.min(np.linalg.norm(locs - cam_S, axis=1)) if len(locs) else dist
        visible = first > dist - 0.3
        pts.append((nm, p, visible))
    chosen = [x for x in pts if x[2]][:6]
    if len(chosen) < 6:
        chosen += [x for x in pts if not x[2]][:6 - len(chosen)]
    res = {}
    for tag, sigma in (("exact", 0.0), ("noisy", 1.5)):
        P = []
        for nm, p, vis in chosen:
            Pc = T_C_S[:3, :3] @ p + T_C_S[:3, 3]
            uv = cam.project(Pc)[0]
            if sigma > 0:
                uv = uv + rng.normal(0, sigma, 2)
            P.append({"name": nm, "img_px": tolist(uv), "stl_mm": tolist(p),
                      "jaw": "upper" if nm[0] in "12" else "lower", "visible_in_reg_frame": bool(vis)})
        d = {"version": 1, "image": {"file": "reg_frame.png", "frame_index": int(reg_i), "width": size[0], "height": size[1]},
             "points": P, "incisal_point_mm": tolist(lm["lower_incisal_point"]),
             "occlusal_plane_mm": [tolist(lm["lower_incisal_point"]), tolist(lm["36_mb"]), tolist(lm["46_mb"])]}
        fn = "registration_landmarks_exact.json" if tag == "exact" else "registration_landmarks.json"
        json.dump(d, open(os.path.join(out, fn), "w"), indent=1)
        res[tag] = d
    # mark landmarks on a debug copy
    img = cv2.imread(os.path.join(out, "reg_frame.png"))
    for p in res["noisy"]["points"]:
        u, v = p["img_px"]
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, p["name"], (int(u) + 7, int(v) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, "_reg_frame_landmarks_debug.png"), img)
    return res


def self_checks(kin, lm, P, tl, frac_in, U, L):
    from .dentition import signed_dist
    Ts, times, segs = kin["Ts"], kin["times"], kin["segments"]
    mi = G.seg_mask(times, segs, "MI")
    ident = float(np.abs(Ts[mi] - np.eye(4)).max())
    ip = tl["incisal_point"]
    dI = G.apply(Ts, ip) - ip
    fr = tl["frame"]
    # condyle chord inclinations at several protrusion levels
    chk = {"max_abs_T_minus_I_in_MI_frames": ident,
           "incisal_point_max_displacement_mm": float(np.linalg.norm(dI, axis=1).max()),
           "condyles_are_rotation_centres": "phi rotates about the instantaneous intercondylar axis; in the pure-hinge phase (<~18 mm) the condyles do not move",
           "frames_all_markers_in_image_fraction": frac_in}
    # max interpenetration over a subsample of moving frames (exact closest-point distances)
    idx = np.where(~mi)[0][::max(1, int((~mi).sum() / 60))]
    v = L.vertices
    sel = v[:, 1] > np.percentile(v[:, 1], 55)
    pts = v[sel][::3]
    pens = []
    for i in idx:
        p = pts @ Ts[i][:3, :3].T + Ts[i][:3, 3]
        near = p[:, 1] > -2.0
        pens.append(float(signed_dist(U, p[near]).max()) if near.any() else -9.0)
    chk["max_interpenetration_during_motion_mm"] = max(pens)
    chk["interpenetration_checked_frames"] = int(len(idx))
    print("    self-checks:", {k: v for k, v in chk.items() if not isinstance(v, str)})
    return chk


def preview_images(out, U, L, Ts, times, segs):
    """Static previews of the STLs in MI and at a few poses (for humans)."""
    K = np.array([[2600, 0, 640], [0, 2600, 400], [0, 0, 1.0]])
    tiles = []
    poses = [("MI", np.eye(4))]
    for lab in ("open_close", "protrusion", "laterotrusion_right"):
        m = G.seg_mask(times, segs, lab)
        if m.any():
            idx = np.where(m)[0]
            disp = np.linalg.norm(Ts[idx][:, :3, 3], axis=1)
            poses.append((lab, Ts[idx[np.argmax(disp)]]))
    for (lab, T), (ang, name) in zip(poses + poses, [(0, "front")] * len(poses) + [(-75, "side")] * len(poses)):
        img = np.full((800, 1280, 3), 45, np.uint8)
        R = Ry(ang) if ang else np.eye(3)
        Rc = np.diag([1, -1, -1.0]) @ R
        c = np.array([0, -5.0, -25])
        for M, Tm, col in ((U, np.eye(4), (215, 230, 240)), (L, T, (195, 215, 235))):
            V = (Tm[:3, :3] @ M.vertices.T).T + Tm[:3, 3]
            Vc = (Rc @ (V - c).T).T + [0, 0, 260]
            render_mesh(img, Vc, M.faces, K, np.zeros(5), base_color=col, light_dir=(0.3, -0.4, -1.0))
        cv2.putText(img, f"{lab} ({name})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (640, 400), interpolation=cv2.INTER_AREA))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    cv2.imwrite(os.path.join(out, "preview_stl.png"), np.vstack(rows))


# ============================================================================ calibration clip
def render_charuco_clip(path, cam, rng, fps, size, duration=20.0):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((7, 5), 25.0, 18.0, d)
    ppm = 8.0
    margin_mm = 10.0
    img = board.generateImage((int((175 + 2 * margin_mm) * ppm), int((125 + 2 * margin_mm) * ppm)),
                              marginSize=int(margin_mm * ppm), borderBits=1)
    # texel -> board mm: chessboard corners (objPoints) known; board origin at top-left of squares
    A = np.array([[1 / ppm, 0, -margin_mm], [0, 1 / ppm, -margin_mm], [0, 0, 1.0]])
    # verify the mapping with the detector on the texture
    cd = cv2.aruco.CharucoDetector(board)
    cc, cid, _, _ = cd.detectBoard(img)
    obj = board.getChessboardCorners()
    pix = np.c_[cc.reshape(-1, 2), np.ones(len(cc))] @ A.T
    err = np.abs(pix[:, :2] - obj[cid.ravel(), :2]).max()
    assert err < 0.15, f"charuco texture mapping error {err}"
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = cv2.GaussianBlur(bgr, (0, 0), 0.6)
    tex = PlanarTexture(bgr, np.ones(img.shape, np.float32), A)
    wall = make_wall(size, rng).astype(np.float32) * 0.9
    n = int(duration * fps)
    # keyframes: board centre position (camera frame) and orientation, smooth cubic interpolation
    from scipy.interpolate import CubicSpline
    nk = int(duration / 1.6) + 2
    tk = np.linspace(0, duration, nk)
    pos, rot = [], []
    for k in range(nk):
        z = rng.uniform(270, 430)
        x = rng.uniform(-0.38, 0.38) * z * size[0] / cam.K[0, 0]
        y = rng.uniform(-0.33, 0.33) * z * size[1] / cam.K[1, 1]
        pos.append([x, y, z])
        rot.append([rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(-25, 25)])
    cp = CubicSpline(tk, np.array(pos), bc_type="clamped")
    cr = CubicSpline(tk, np.array(rot), bc_type="clamped")
    centre = np.array([87.5, 62.5, 0])

    def pose(t):
        a = cr(t)
        R = Rx(a[0]) @ Ry(a[1]) @ Rz(a[2])
        return HT(R, cp(t) - R @ centre)

    proc = ffmpeg_writer(path, size, fps)
    nz = [rng.normal(0, 1, (size[1], size[0], 3)).astype(np.float32) for _ in range(3)]
    exposure = 1 / 120.0
    poses_out = []
    for i in range(n):
        t = i / fps
        T0 = pose(t)
        poses_out.append(flat(T0))
        P0 = cam.project((T0[:3, :3] @ obj.T).T + T0[:3, 3])
        T1 = pose(min(t + 1 / fps, duration))
        P1 = cam.project((T1[:3, :3] @ obj.T).T + T1[:3, 3])
        blur = np.abs(P1 - P0).max() * fps * exposure
        ns = int(np.clip(np.ceil(blur / 0.6), 1, 9))
        frame = wall.copy()
        offs = [0.0] if ns == 1 else (np.arange(ns) + 0.5) / ns - 0.5
        acc = None
        for o in offs:
            L_ = warp_plane(cam, tex, pose(np.clip(t + o * exposure, 0, duration)), ss=2)
            if L_ is None:
                continue
            col, al, roi = L_
            full_c = np.zeros_like(frame); full_a = np.zeros(frame.shape[:2], np.float32)
            x0, y0, x1, y1 = roi
            full_c[y0:y1, x0:x1] = col * al[..., None]
            full_a[y0:y1, x0:x1] = al
            acc = (full_c, full_a) if acc is None else (acc[0] + full_c, acc[1] + full_a)
        if acc is not None:
            a = acc[1] / len(offs)
            c = acc[0] / np.maximum(acc[1][..., None], 1e-6)
            frame = frame * (1 - a[..., None]) + c * a[..., None]
        frame *= 1.0 + 0.03 * np.sin(t * 0.7)
        frame += nz[i % 3] * 2.0
        proc.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())
        if i % 300 == 0:
            print(f"    calib frame {i}/{n}")
    proc.stdin.close()
    proc.wait()
    json.dump({"board": {"type": "ChArUco", "dictionary": "DICT_5X5_100", "squares": [7, 5], "square_mm": 25.0, "marker_mm": 18.0},
               "fps": fps, "T_C_board_per_frame": poses_out},
              open(path.replace(".mp4", "_poses_true.json"), "w"))
