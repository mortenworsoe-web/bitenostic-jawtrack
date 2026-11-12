"""Procedural dentition in the SPEC synthetic frame (x = patient's LEFT, y = SUPERIOR,
z = ANTERIOR, mm, origin near the upper incisal point).

Each crown is a closed ring-mesh: an occlusal height-field (cusps, fossa, ridges) on a
superelliptic table, flowing into bulging sides, a cervical constriction and a root stub that
is buried in a horseshoe gingiva/base block. Crowns + base are boolean-unioned with manifold3d
so each arch is a single watertight mesh.
"""
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline

# FDI tooth types per position 1..7 (central incisor .. 2nd molar)
TYPES = ["incisor", "incisor", "canine", "premolar", "premolar", "molar", "molar"]
# (mesiodistal width, buccolingual width, crown height) in mm
UPPER_DIMS = [(8.6, 7.0, 10.5), (6.6, 6.2, 9.0), (7.6, 8.0, 10.0), (7.0, 9.0, 8.0),
              (6.6, 9.0, 7.5), (10.2, 11.0, 7.0), (9.2, 11.0, 6.5)]
LOWER_DIMS = [(5.3, 5.8, 9.0), (5.9, 6.2, 9.0), (6.9, 7.6, 10.0), (7.0, 7.6, 8.0),
              (7.1, 8.0, 7.5), (11.0, 10.4, 7.0), (10.4, 10.0, 6.5)]


# ----------------------------------------------------------------------------- arch curves
class Arch:
    """Planar (x,z) symmetric arch curve parameterised by arc length s from the midline,
    s > 0 towards the patient's LEFT (+x)."""

    def __init__(self, ctrl):
        ctrl = np.asarray(ctrl, float)  # right-half control points (x>=0,z), starting at (0,z0)
        full = np.vstack([ctrl[::-1] * [-1, 1], ctrl[1:]])
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
        d -= d[len(ctrl) - 1]
        self.cs = CubicSpline(d, full, bc_type="natural")
        tt = np.linspace(d[0], d[-1], 20001)
        p = self.cs(tt)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        s -= np.interp(0.0, tt, s)
        self._t_of_s = (s, tt)

    def point(self, s):
        t = np.interp(s, *self._t_of_s)
        return self.cs(t)

    def tangent(self, s):
        t = np.interp(s, *self._t_of_s)
        v = self.cs(t, 1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)



UPPER_CTRL = [(0, -0.8), (5, -1.6), (10, -3.6), (14.5, -6.6), (18.0, -10.5), (20.5, -15.5),
              (22.4, -21.0), (23.6, -27.0), (24.6, -33.0), (25.4, -39.0), (26.0, -45.0),
              (26.4, -51.0)]


def outward_normal(arch, s):
    tg = arch.tangent(s)
    # tangent runs in +s direction (right-distal -> midline -> left-distal); rotate +90 deg in
    # (x,z) to get the buccal/labial (outward) normal
    return np.array([-tg[1], tg[0]])


def offset_arch(arch, d_of_s, smax=60.0, n=400):
    """Lingual offset of an arch (positive d moves lingually)."""
    ss = np.linspace(0, smax, n)
    pts = []
    for s in ss:
        p = arch.point(s)
        nn = outward_normal(arch, s)
        pts.append(p - d_of_s(s) * nn)
    pts = np.array(pts)
    pts[0, 0] = 0.0
    return Arch(pts[::8])


# ----------------------------------------------------------------------------- crowns
def _superellipse(theta, n):
    c, s = np.cos(theta), np.sin(theta)
    return np.sign(c) * np.abs(c) ** (2.0 / n), np.sign(s) * np.abs(s) ** (2.0 / n)


def _gauss(u, v, uc, vc, su, sv):
    return np.exp(-0.5 * (((u - uc) / su) ** 2 + ((v - vc) / sv) ** 2))


def occlusal_height(kind, u, v, buccal_sign=1.0):
    """Occlusal height field (mm, along occlusal direction) on the normalised table (u,v in
    [-1,1], u mesiodistal, v buccolingual with +v = buccal). Rim (r=1) is ~0."""
    r2 = u ** 2 + v ** 2
    if kind == "molar":
        h = 0.6 + 0.0 * u
        for (uc, vc, a) in [(0.45, 0.6, 1.9), (-0.42, 0.6, 1.7), (0.45, -0.58, 2.0), (-0.45, -0.58, 1.5)]:
            h = h + a * _gauss(u, v, uc, vc, 0.30, 0.30)
        h -= 0.6 * _gauss(u, v, 0, 0, 0.25, 0.25)
        h -= 1.2 * np.clip(r2 - 0.55, 0, None) ** 1.2
    elif kind == "premolar":
        h = 0.5 + 2.3 * _gauss(u, v, 0, 0.5, 0.42, 0.32) + 1.5 * _gauss(u, v, 0, -0.5, 0.42, 0.32)
        h += 0.5 * (_gauss(u, v, 0.8, 0, 0.2, 0.5) + _gauss(u, v, -0.8, 0, 0.2, 0.5))  # marginal ridges
        h -= 1.2 * np.clip(r2 - 0.55, 0, None) ** 1.2
    elif kind == "canine":
        h = 2.3 * np.exp(-0.5 * ((u / 0.6) ** 2 + ((v - 0.15) / 0.55) ** 2)) ** 0.9
        h += 0.5 * _gauss(u, v, 0, -0.3, 0.3, 0.4)  # cingulum-ish
        h -= 0.8 * np.clip(r2 - 0.5, 0, None)
    elif kind == "incisor":
        # chisel: straight incisal edge along u near v~0.2 (labial), fall-off at corners
        h = 1.6 * np.exp(-0.5 * ((v - 0.2) / 0.45) ** 2) * (1 - 0.25 * np.abs(u) ** 4)
    else:
        raise ValueError(kind)
    return h


def crown_mesh(kind, md, bl, height, n_theta=56, n_top=11, n_side=16, root=6.0, upper=True):
    """Crown in local coords: u = mesiodistal (x), w = occlusal (y, +up), v = buccal (z).
    Reference (0,0,0) = centre of the occlusal-table rim plane. Returns trimesh.Trimesh."""
    a, b = md / 2.0, bl / 2.0
    nexp = {"molar": 3.0, "premolar": 2.6, "canine": 2.3, "incisor": 2.8}[kind]
    table = {"molar": (0.86, 0.70), "premolar": (0.82, 0.62), "canine": (0.72, 0.62),
             "incisor": (0.98, 0.22)}[kind]
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cu, cv = _superellipse(theta, nexp)
    rings = []
    # top: occlusal table r in (0,1]
    for k in range(1, n_top + 1):
        r = k / n_top
        u, v = r * cu, r * cv
        x = a * table[0] * u
        z = b * table[1] * v
        y = occlusal_height(kind, u, v)
        rings.append(np.stack([x, y, z], -1))
    # sides: from rim downwards. profile scale (sx, sz) as a function of depth fraction f
    total = height + root
    y0 = rings[-1][:, 1].copy()
    for j in range(1, n_side + 1):
        f = j / n_side
        depth = f * total
        dc = depth / height  # fraction of crown height
        if kind == "incisor":
            sx = table[0] + (1 - table[0]) * np.clip(dc / 0.5, 0, 1) - 0.25 * np.clip((dc - 0.6) / 0.4, 0, 1)
            sz = table[1] + (1 - table[1]) * np.sin(np.clip(dc, 0, 1) * np.pi / 2) ** 0.8
            sz = sz - 0.15 * np.clip((dc - 0.85) / 0.3, 0, 1)
        else:
            bulge = np.sin(np.clip(dc / 0.35, 0, 1) * np.pi / 2)
            sx = table[0] + (1 - table[0]) * bulge - 0.18 * np.clip((dc - 0.45) / 0.55, 0, 1)
            sz = table[1] + (1 - table[1]) * bulge - 0.16 * np.clip((dc - 0.45) / 0.55, 0, 1)
        if dc > 1:  # root stub
            sx = sx - 0.05 * min(dc - 1, 1)
            sz = sz - 0.05 * min(dc - 1, 1)
        x = a * sx * cu
        z = b * sz * cv
        # blend rim height (varies with theta) into a flat level as we go down
        w = np.clip(dc / 0.4, 0, 1)
        y = (1 - w) * y0 + w * y0.mean() - depth
        rings.append(np.stack([x, y, z], -1))
    rings = np.array(rings)  # (R, T, 3)
    R, T = rings.shape[:2]
    top = np.array([[0.0, occlusal_height(kind, np.array(0.0), np.array(0.0)), 0.0]])
    bot = np.array([[0.0, rings[-1, :, 1].mean() - 0.5, 0.0]])
    verts = np.vstack([top, rings.reshape(-1, 3), bot])
    faces = []
    idx = lambda r, t: 1 + r * T + (t % T)
    for t in range(T):
        faces.append([0, idx(0, t + 1), idx(0, t)])
    for r in range(R - 1):
        for t in range(T):
            a0, a1, b0, b1 = idx(r, t), idx(r, t + 1), idx(r + 1, t), idx(r + 1, t + 1)
            faces.append([a0, a1, b1])
            faces.append([a0, b1, b0])
    last = len(verts) - 1
    for t in range(T):
        faces.append([idx(R - 1, t), idx(R - 1, t + 1), last])
    m = trimesh.Trimesh(verts, np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


# ----------------------------------------------------------------------------- arch assembly
def rot_axis(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def min_rot(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    return rot_axis(v, np.arctan2(np.linalg.norm(v), c))


def spee_level(z, depth=1.2, z0=-22.0):
    """Lower contact level (y) as function of depth z: overbite 2 mm at the incisors plus a
    curve of Spee ~1.2 mm deep with its lowest point around the 2nd premolar / 1st molar."""
    k = depth / (z0 ** 2)
    return 2.0 + k * (z - z0) ** 2 - depth


def build_arch_teeth(upper, arch, dims, s_offset=0.0, level_fn=None):
    """Return list of dict(tooth, mesh(local->S transformed), frame info)."""
    teeth = []
    for side in (+1, -1):  # +1 = patient's left (quadrant 2 / 3), -1 = right (1 / 4)
        s = s_offset
        for i, kind in enumerate(TYPES):
            md, bl, h = dims[i]
            sc = s + md / 2.0 - 0.15  # slight overlap for contact areas
            s += md - 0.3
            sa = side * sc
            p2 = arch.point(sa)
            tg2 = arch.tangent(sa) * side  # mesial -> distal direction
            nb2 = outward_normal(arch, sa)
            T = np.array([tg2[0], 0, tg2[1]])
            B = np.array([nb2[0], 0, nb2[1]])
            B /= np.linalg.norm(B)
            T = T - B * (T @ B)
            T /= np.linalg.norm(T)
            O = np.array([0, -1.0, 0]) if upper else np.array([0, 1.0, 0])
            # curve of Spee: occlusal direction = normal of the sagittal contact curve y(z)
            z = p2[1]
            dz = 0.5
            yp = (spee_level(z + dz) - spee_level(z - dz)) / (2 * dz)
            up = np.array([0.0, 1.0, -yp])
            up /= np.linalg.norm(up)
            Rb = min_rot(O, up if not upper else -up)
            # Wilson (molars/premolars) & incisor proclination: tilt O towards +/-B about T
            if kind == "molar":
                tilt = np.radians(-9 if not upper else 8)
            elif kind == "premolar":
                tilt = np.radians(-4 if not upper else 4)
            elif kind == "incisor":
                tilt = np.radians(20 if upper else 18)
            else:
                tilt = np.radians(12 if upper else 10)
            # rotation about T that moves O towards +B by 'tilt'
            sgn = np.sign(np.cross(T, O) @ B)
            Rt = rot_axis(T, tilt * sgn)
            Rm = Rt @ Rb
            Ow = Rm @ O
            Bw = Rm @ B
            Tw = Rm @ T
            # local (u,w,v) -> world: u along T (mesial->distal is +s; use Tw), w along Ow, v along Bw
            Rl = np.stack([Tw, Ow, Bw], axis=1)
            if np.linalg.det(Rl) < 0:
                Rl[:, 0] *= -1
            mesh = crown_mesh(kind, md, bl, h, upper=upper)
            ylev = level_fn(kind, z, i)
            origin = np.array([p2[0], ylev, p2[1]])
            M = np.eye(4)
            M[:3, :3] = Rl
            M[:3, 3] = origin
            mesh.apply_transform(M)
            quad = (2 if side > 0 else 1) if upper else (3 if side > 0 else 4)
            teeth.append(dict(fdi=quad * 10 + i + 1, kind=kind, mesh=mesh, M=M, md=md, bl=bl,
                              upper=upper))
    return teeth


def base_block(arch, upper, s_max, width_fn, y_near, y_far, n_ring=40, n_s=160):
    """Horseshoe gingiva/base: swept rounded rectangle along the arch."""
    ss = np.linspace(-s_max, s_max, n_s)
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    cu, cv = _superellipse(th, 3.5)
    rings = []
    for s in ss:
        p = arch.point(s)
        nb = outward_normal(arch, s)
        hw = width_fn(s)
        ymid = 0.5 * (y_near + y_far)
        hh = 0.5 * abs(y_far - y_near)
        bx = p[0] + hw * cu * nb[0]
        bz = p[1] + hw * cu * nb[1]
        by = ymid + hh * cv
        rings.append(np.stack([bx, by, bz], -1))
    rings = np.array(rings)
    R, T = rings.shape[:2]
    verts = rings.reshape(-1, 3)
    c0 = rings[0].mean(0)
    c1 = rings[-1].mean(0)
    verts = np.vstack([verts, c0, c1])
    i0, i1 = len(verts) - 2, len(verts) - 1
    idx = lambda r, t: r * T + (t % T)
    f = []
    for r in range(R - 1):
        for t in range(T):
            f += [[idx(r, t), idx(r, t + 1), idx(r + 1, t + 1)], [idx(r, t), idx(r + 1, t + 1), idx(r + 1, t)]]
    for t in range(T):
        f.append([i0, idx(0, t + 1), idx(0, t)])
        f.append([i1, idx(R - 1, t), idx(R - 1, t + 1)])
    m = trimesh.Trimesh(verts, np.array(f), process=True)
    if m.volume < 0:
        m.invert()
    return m


def union_meshes(meshes):
    import manifold3d as mf
    ms = []
    for m in meshes:
        mm = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                                 tri_verts=np.asarray(m.faces, np.uint32)))
        if mm.status() != mf.Error.NoError:
            raise RuntimeError(f"manifold error {mm.status()}")
        ms.append(mm)
    u = mf.Manifold.batch_boolean(ms, mf.OpType.Add)
    out = u.to_mesh()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)


# ----------------------------------------------------------------------------- heightfields
def heightfield(meshes, lo=True, res=0.2, bounds=(-40, 40, -70, 15), samples=4):
    """Min (lo=True) or max y of a set of meshes on an (x,z) grid, by dense triangle sampling."""
    x0, x1, z0, z1 = bounds
    nx, nz = int((x1 - x0) / res) + 1, int((z1 - z0) / res) + 1
    hf = np.full((nz, nx), np.inf if lo else -np.inf)
    rng = np.random.default_rng(0)
    for m in meshes:
        tri = m.triangles
        pts = [m.vertices]
        n = samples
        for k in range(n * 3):
            w = rng.dirichlet([1, 1, 1], size=len(tri))
            pts.append(np.einsum("ij,ijk->ik", w, tri))
        p = np.vstack(pts)
        ix = np.clip(np.round((p[:, 0] - x0) / res).astype(int), 0, nx - 1)
        iz = np.clip(np.round((p[:, 2] - z0) / res).astype(int), 0, nz - 1)
        if lo:
            np.minimum.at(hf, (iz, ix), p[:, 1])
        else:
            np.maximum.at(hf, (iz, ix), p[:, 1])
    return dict(hf=hf, x0=x0, z0=z0, res=res, lo=lo)


def hf_lookup(H, x, z):
    """Nearest-cell lookup with 3x3 min/max dilation (conservative)."""
    hf = H["hf"]
    ix = np.round((x - H["x0"]) / H["res"]).astype(int)
    iz = np.round((z - H["z0"]) / H["res"]).astype(int)
    ok = (ix >= 1) & (iz >= 1) & (ix < hf.shape[1] - 1) & (iz < hf.shape[0] - 1)
    out = np.full(x.shape, np.inf if H["lo"] else -np.inf)
    ixo, izo = ix[ok], iz[ok]
    vals = []
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            vals.append(hf[izo + dz, ixo + dx])
    out[ok] = np.min(vals, 0) if H["lo"] else np.max(vals, 0)
    return out


# ----------------------------------------------------------------------------- full build
def build_dentition(d_post=2.2, d_ant=3.4, lower_scale=0.93, verbose=True):
    upper_arch = Arch(UPPER_CTRL)

    def d_of_s(s):
        w = np.clip((s - 8.0) / 14.0, 0, 1)
        w = w * w * (3 - 2 * w)
        return d_ant * (1 - w) + d_post * w

    lower_arch = offset_arch(upper_arch, d_of_s)

    def upper_level(kind, z, i):
        if i == 0:
            return 1.6
        if i == 1:
            return 2.2
        if kind == "canine":
            return 0.3 + 2.3
        h00 = float(occlusal_height(kind, np.array(0.0), np.array(0.0)))
        return spee_level(z) + h00

    def lower_level(kind, z, i):
        hmax = {"incisor": 1.6, "canine": 2.3, "premolar": 2.5, "molar": 2.4}[kind]
        return spee_level(z) - 1.0 - hmax

    up = build_arch_teeth(True, upper_arch, UPPER_DIMS, level_fn=upper_level)
    lo_dims = [(a * lower_scale, b, c) for (a, b, c) in LOWER_DIMS]
    lo = build_arch_teeth(False, lower_arch, lo_dims, level_fn=lower_level)

    # --- per-tooth vertical equilibration of lower teeth against the upper crowns
    H = heightfield([t["mesh"] for t in up], lo=True)
    for t in lo:
        v = t["mesh"].vertices
        top = v[:, 1] > np.percentile(v[:, 1], 40)
        gap = hf_lookup(H, v[top, 0], v[top, 2]) - v[top, 1]
        g = np.min(gap[np.isfinite(gap)])
        shift = g + 0.02  # leave 0.02 mm (sampling) interpenetration -> contact
        t["mesh"].apply_translation([0, shift, 0])
        t["M"][1, 3] += shift
        t["shift"] = shift
        if verbose:
            print(f"  lower {t['fdi']}: vertical shift {shift:+.2f} mm")
    return up, lo, upper_arch, lower_arch


def make_bases(up, lo, upper_arch, lower_arch):
    def w_up(s):
        a = abs(s)
        return np.interp(a, [0, 12, 22, 60], [3.6, 4.2, 5.2, 5.8])

    def w_lo(s):
        a = abs(s)
        return np.interp(a, [0, 12, 22, 60], [3.0, 3.6, 5.2, 6.6])

    ub = base_block(upper_arch, True, 54.0, w_up, 6.0, 20.0)
    lb = base_block(lower_arch, False, 51.0, w_lo, -4.0, -20.0)
    return ub, lb


def signed_dist(mesh, pts, chunk=4000):
    """Signed distance of pts to a watertight mesh, positive INSIDE (penetration). Sign from the
    normal of the closest triangle (robust enough for small penetrations)."""
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        c, d, tid = trimesh.proximity.closest_point(mesh, p)
        s = np.einsum("ij,ij->i", p - c, mesh.face_normals[tid])
        out[i:i + chunk] = np.where(s < 0, d, -d)
    return out


def refine_contacts(lo_teeth, upper_mesh, target=0.03, iters=6, verbose=True):
    """Raise each lower tooth vertically until its closest distance to the upper arch is ~target
    (exact closest-point distances; the vertical gap is >= the Euclidean one, so no overshoot)."""
    for t in lo_teeth:
        total = 0.0
        for _ in range(iters):
            v = t["mesh"].vertices
            top = v[:, 1] > np.percentile(v[:, 1], 60)
            sd = signed_dist(upper_mesh, v[top])
            gap = -sd.max()
            if gap <= target + 0.01:
                break
            step = gap - target
            t["mesh"].apply_translation([0, step, 0])
            t["M"][1, 3] += step
            total += step
        t["shift"] = t.get("shift", 0) + total
        if verbose:
            print(f"  lower {t['fdi']}: refine {total:+.3f} mm, final gap {gap:.3f}")


TABLE = {"molar": (0.86, 0.70), "premolar": (0.82, 0.62), "canine": (0.72, 0.62), "incisor": (0.98, 0.22)}


def tooth_point(t, u, v):
    """World position of the occlusal-surface point at normalised table coords (u,v) of tooth t."""
    a, b = t["md"] / 2, t["bl"] / 2
    tb = TABLE[t["kind"]]
    h = float(occlusal_height(t["kind"], np.array(u), np.array(v)))
    p = np.array([a * tb[0] * u, h, b * tb[1] * v, 1.0])
    return (t["M"] @ p)[:3]


def snap(mesh, p):
    c, d, _ = trimesh.proximity.closest_point(mesh, np.atleast_2d(p))
    return c[0]


def landmarks(up, lo, U, L):
    """True anatomical landmarks (snapped onto the final meshes)."""
    by = {t["fdi"]: t for t in up + lo}
    out = {}
    for f in (11, 21):
        out[f"{f}_incisal"] = snap(U, tooth_point(by[f], 0.0, 0.2))
    for f in (13, 23):
        out[f"{f}_cusp"] = snap(U, tooth_point(by[f], 0.0, 0.15))
    for f in (33, 43):
        out[f"{f}_cusp"] = snap(L, tooth_point(by[f], 0.0, 0.15))
    for f, M in ((16, U), (26, U), (36, L), (46, L)):
        c1 = tooth_point(by[f], 0.45, 0.6)
        c2 = tooth_point(by[f], -0.42, 0.6)
        mb = c1 if c1[2] > c2[2] else c2  # more anterior buccal cusp = mesiobuccal
        out[f"{f}_mb"] = snap(M, mb)
    # lower incisal point: midpoint of the mesial ends of the 31/41 incisal edges
    pts = []
    for f in (31, 41):
        e1 = tooth_point(by[f], 0.85, 0.2)
        e2 = tooth_point(by[f], -0.85, 0.2)
        pts.append(snap(L, e1 if abs(e1[0]) < abs(e2[0]) else e2))
    out["lower_incisal_point"] = 0.5 * (pts[0] + pts[1])
    pts = [snap(U, tooth_point(by[f], 0.85 * s, 0.2)) for f in (11, 21) for s in (1, -1)]
    pts = sorted(pts, key=lambda p: abs(p[0]))[:2]
    out["upper_incisal_point"] = 0.5 * (pts[0] + pts[1])
    return out
