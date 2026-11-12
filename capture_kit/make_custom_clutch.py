#!/usr/bin/env python3
"""Custom-fit paraocclusal clutch from the patient's own LOWER intraoral scan (stretch goal).

Builds a thin labial shell (default 1.5 mm, 0.1 mm cement relief) over the CERVICAL part of the
labial surfaces 33-43, then attaches the same stem + JAW-sticker plate as the generic clutch.
The output STL is in the SAME coordinate frame as the input lower STL (so it can be checked
against the scan, and the clutch-to-teeth geometry is known exactly).

Landmarks (mm, in STL coordinates; pick them in any STL viewer / the JawTrack viewer):
  --p33  labial gingival-margin point at the middle of 33 (lower LEFT canine)
  --p43  labial gingival-margin point at the middle of 43 (lower RIGHT canine)
  --incisal  lower incisal point (between 31/41 incisal edges)
Optional: --upper upper.stl (in occlusion) -> reports minimum clearance clutch<->upper teeth in MI.

Usage:
  python3 make_custom_clutch.py --lower lower.stl --p33 x,y,z --p43 x,y,z --incisal x,y,z [--upper upper.stl]
  python3 make_custom_clutch.py --selftest      # procedural arch test, writes verify/custom_clutch_selftest*.stl
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from trimesh.transformations import rotation_matrix, translation_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_holders import JAW_STICKER, RESIN_DENSITY, beam, box, plate_with_pocket, union  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def vec(s):
    return np.array([float(v) for v in s.split(",")])


def local_frame(p33, p43, incisal):
    """x = patient left (43->33), y = up (cervical->incisal), z = anterior/labial. Origin = mid p33/p43."""
    o = (p33 + p43) / 2
    ex = (p33 - p43) / np.linalg.norm(p33 - p43)
    up = incisal - o
    ey = up - ex * (up @ ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    T = np.eye(4)
    T[:3, :3] = np.c_[ex, ey, ez]
    T[:3, 3] = o
    return T  # local -> STL


def build_shell(lower_local, half_width, relief=0.1, thick=1.5, y0=-0.3, y1=4.5, vox=0.2):
    """Voxel shell on the labial side. lower_local: lower mesh already in the local frame."""
    pts, fi = trimesh.sample.sample_surface_even(lower_local, 400000, radius=0.08)
    nrm = lower_local.face_normals[fi]
    # keep only surface patches in the target band and labial-facing
    sel = (np.abs(pts[:, 0]) < half_width + 3) & (pts[:, 1] > y0 - 3) & (pts[:, 1] < y1 + 3)
    tree = cKDTree(pts[sel])
    nrm_s = nrm[sel]
    lo = np.array([-half_width - 2, y0 - 1, pts[sel][:, 2].min() - 1])
    hi = np.array([half_width + 2, y1 + 1, pts[sel][:, 2].max() + thick + 2])
    axes = [np.arange(lo[i], hi[i] + vox, vox) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, j = tree.query(G, workers=-1)
    s = np.einsum("ij,ij->i", G - tree.data[j], nrm_s[j])
    n_lab = nrm_s[j][:, 2]
    inside = (s > 0) & (d >= relief) & (d <= relief + thick) & (n_lab > 0.2) \
        & (G[:, 1] >= y0) & (G[:, 1] <= y1) & (np.abs(G[:, 0]) <= half_width)
    vol = inside.reshape([len(a) for a in axes]).astype(float)
    vol = np.pad(gaussian_filter(vol, 0.7), 2)
    v, f, _, _ = marching_cubes(vol, 0.5, spacing=(vox, vox, vox))
    v += lo - 2 * vox
    m = trimesh.Trimesh(v, f, process=True)
    if m.volume < 0:
        m.invert()
    parts = m.split(only_watertight=True)
    m = max(parts, key=lambda p: p.volume)  # drop speckles
    return m


def stem_and_plate(P0):
    """Same stem + plate geometry as the generic clutch, starting at P0 (local frame)."""
    w, h = 6.0, 4.0
    P0 = np.asarray(P0, float)
    P1 = P0 + [0, 0, 12.6]
    dn = np.radians(40)
    P2 = P1 + 30.0 * np.array([0, -np.sin(dn), np.cos(dn)])
    parts = [beam(P0, P1, w, h), box((w, h, h), translation_matrix(P1) @ rotation_matrix(-dn / 2, [1, 0, 0])),
             beam(P1, P2, w, h)]
    plate, (W, H) = plate_with_pocket(JAW_STICKER, margin=2.0, thick=1.8)
    Rt = rotation_matrix(np.radians(15), [1, 0, 0])
    Tp = translation_matrix(P2 - (Rt @ [0, H / 2 - 6.0, -1.8, 1.0])[:3]) @ Rt
    plate.apply_transform(Tp)
    parts += [plate, box((w + 4, h + 4, 3.0), Tp @ translation_matrix((0, H / 2 - 6.0, -3.2))),
              beam(P2 - 3 * np.array([0, -np.sin(dn), np.cos(dn)]), P2 + [0, 0, 1.0], w, h)]
    return parts


def make_custom(lower, p33, p43, incisal, upper=None, thick=1.5, relief=0.1, y1=None):
    T = local_frame(p33, p43, incisal)
    Tinv = np.linalg.inv(T)
    low = lower.copy()
    low.apply_transform(Tinv)
    inc_h = (Tinv @ np.r_[incisal, 1])[1]
    if y1 is None:
        y1 = min(4.5, 0.5 * inc_h)  # cervical half of the crown
    half_w = np.linalg.norm(p33 - p43) / 2 + 1.0
    shell = build_shell(low, half_w, relief=relief, thick=thick, y1=y1)
    # stem starts at the labial-most shell point near the midline, in the lower half of the shell
    v = shell.vertices
    mid = v[(np.abs(v[:, 0]) < 3) & (v[:, 1] < y1 * 0.6)]
    P0 = np.array([0.0, max(1.6, y1 * 0.35), mid[:, 2].max() - 0.8])
    m = union([shell] + stem_and_plate(P0))
    m.apply_transform(T)
    rep = dict(watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)),
               volume_cm3=round(m.volume / 1000, 3), mass_g_resin=round(m.volume / 1000 * RESIN_DENSITY, 2),
               shell_height_mm=[-0.3, round(float(y1), 2)], crown_height_to_incisal_mm=round(float(inc_h), 2),
               relief_mm=relief, thickness_mm=thick)
    # min distance clutch -> lower (should be ~relief) and clutch -> upper (must be clearly > 0)
    pl, _ = trimesh.sample.sample_surface_even(lower, 200000, radius=0.08)
    rep["min_dist_to_lower_mm"] = round(float(cKDTree(pl).query(m.vertices)[0].min()), 3)
    if upper is not None:
        pu, _ = trimesh.sample.sample_surface_even(upper, 200000, radius=0.08)
        du = cKDTree(pu).query(m.vertices)[0]
        rep["min_clearance_to_upper_in_MI_mm"] = round(float(du.min()), 3)
        if du.min() < 0.5:
            rep["WARNING"] = "clutch comes within 0.5 mm of the upper teeth in MI - lower --height or check landmarks"
    return m, rep


# ---------------- procedural test arch ----------------
def procedural_jaws():
    """Lower 43-33 crowns (ellipsoids on an arc, radius 21 mm, crown 0..9 mm) on a gingiva block;
    upper incisors/canines overlapping labially (overjet ~2.5, overbite ~2.5 mm). Frame: x left, y up, z anterior."""
    def crowns(R, ang_deg, sizes, yc, ry, z_off=0.0):
        out = []
        for a, (mx, lz) in zip(ang_deg, sizes):
            e = trimesh.creation.icosphere(subdivisions=3)
            e.apply_scale((mx, ry, lz))
            a = np.radians(a)
            e.apply_transform(translation_matrix((R * np.sin(a), yc, R * np.cos(a) + z_off)) @ rotation_matrix(a, [0, 1, 0]))
            out.append(e)
        return out
    ang = [-42, -26, -9, 9, 26, 42]                    # 43 42 41 31 32 33 (x = patient left positive)
    sz = [(3.6, 3.4), (3.1, 3.0), (2.8, 2.9), (2.8, 2.9), (3.1, 3.0), (3.6, 3.4)]
    low = crowns(19.0, ang, sz, 4.5, 5.0)
    ging = trimesh.creation.cylinder(radius=20.5, height=8, sections=96)
    ging.apply_transform(translation_matrix((0, -3.0, 0)) @ rotation_matrix(np.pi / 2, [1, 0, 0]))
    ging = trimesh.boolean.difference([ging, trimesh.creation.cylinder(radius=14, height=10, sections=96,
                                      transform=translation_matrix((0, -3, 0)) @ rotation_matrix(np.pi / 2, [1, 0, 0]))], engine="manifold")
    lower = union(low + [ging])
    up = crowns(21.6, [-40, -24, -8, 8, 24, 40], [(4.0, 3.6), (3.4, 3.1), (4.3, 3.2), (4.3, 3.2), (3.4, 3.1), (4.0, 3.6)], 11.0, 5.0)
    upper = union(up)
    p43 = np.array([19.0 * np.sin(np.radians(-42)) * 1.18, 0.0, 19.0 * np.cos(np.radians(-42)) * 1.18])
    p33 = p43 * [-1, 1, 1]
    incisal = np.array([0.0, 9.3, 21.9])
    return lower, upper, p33, p43, incisal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lower")
    ap.add_argument("--upper")
    ap.add_argument("--p33", type=vec)
    ap.add_argument("--p43", type=vec)
    ap.add_argument("--incisal", type=vec)
    ap.add_argument("--thickness", type=float, default=1.5)
    ap.add_argument("--relief", type=float, default=0.1)
    ap.add_argument("--height", type=float, default=None, help="shell height above gingival margin (default min(4.5, half crown))")
    ap.add_argument("--out", default="custom_clutch.stl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        lower, upper, p33, p43, inc = procedural_jaws()
        os.makedirs(os.path.join(HERE, "verify"), exist_ok=True)
        lower.export(os.path.join(HERE, "verify", "selftest_lower.stl"))
        upper.export(os.path.join(HERE, "verify", "selftest_upper.stl"))
        m, rep = make_custom(lower, p33, p43, inc, upper, a.thickness, a.relief, a.height)
        out = os.path.join(HERE, "verify", "custom_clutch_selftest.stl")
        m.export(out)
        rep["output"] = out
        json.dump(rep, open(os.path.join(HERE, "verify", "custom_clutch_selftest.json"), "w"), indent=1)
        try:
            from make_holders import preview
            sc = trimesh.util.concatenate([m, lower])
            preview(sc, os.path.join(HERE, "verify", "custom_clutch_selftest_preview.png"), "custom clutch + test arch")
        except Exception as e:  # preview is optional
            print("preview failed:", e)
        print(json.dumps(rep, indent=1))
        return
    if not (a.lower and a.p33 is not None and a.p43 is not None and a.incisal is not None):
        ap.error("--lower, --p33, --p43 and --incisal are required (or use --selftest)")
    lower = trimesh.load(a.lower, force="mesh")
    upper = trimesh.load(a.upper, force="mesh") if a.upper else None
    m, rep = make_custom(lower, a.p33, a.p43, a.incisal, upper, a.thickness, a.relief, a.height)
    m.export(a.out)
    rep["output"] = os.path.abspath(a.out)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
