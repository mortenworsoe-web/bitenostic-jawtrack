#!/usr/bin/env python3
"""Generate the 3D-printable marker holders (STL, mm) for the JawTrack capture kit.

  jaw_clutch_paddle.stl  generic paraocclusal mandibular clutch (labial bar 33-43 + stem + JAW plate)
  head_marker_plate.stl  forehead plate for the HEAD sticker, headband slots, 12 deg downward tilt

Model frame (both parts): x = patient's left/right, y = up, z = anterior (toward the camera).
The clutch bar's arch centre is the origin; its lower (gingival) edge is y = 0.

Usage: python3 make_holders.py [--outdir DIR] [--arch-radius 22] [--no-preview]
"""
import argparse
import json
import os

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

RESIN_DENSITY = 1.18  # g/cm^3, typical SLA/DLP standard resin (PLA ~1.24)

# sticker sizes = board + 2 x 4 mm quiet zone (see make_markers.py / board_geometry.json)
JAW_STICKER = (41.0, 41.0)
HEAD_STICKER = (59.0, 41.0)
POCKET_CLEAR = 0.4   # total clearance on pocket width/height
POCKET_DEPTH = 0.3


def annulus_sector(r_in, r_out, a0, a1, y0, y1, n=64):
    """Watertight curved slab: radii r_in..r_out around the y axis, angles a0..a1 (rad, 0 = +z), y0..y1."""
    a = np.linspace(a0, a1, n)
    verts = []
    for r in (r_in, r_out):
        for y in (y0, y1):
            verts.append(np.c_[r * np.sin(a), np.full(n, y), r * np.cos(a)])
    V = np.vstack(verts)  # blocks: 0 in-lo, 1 in-hi, 2 out-lo, 3 out-hi
    idx = lambda b, i: b * n + i
    F = []
    for i in range(n - 1):
        for (b0, b1, flip) in ((0, 1, False), (2, 3, True), (0, 2, True), (1, 3, False)):
            q = [idx(b0, i), idx(b0, i + 1), idx(b1, i + 1), idx(b1, i)]
            t = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            F += [tt[::-1] for tt in t] if flip else t
    for i, flip in ((0, False), (n - 1, True)):
        q = [idx(0, i), idx(1, i), idx(3, i), idx(2, i)]
        t = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        F += [tt[::-1] for tt in t] if flip else t
    m = trimesh.Trimesh(V, np.array(F), process=True)
    if m.volume < 0:
        m.invert()
    assert m.is_watertight
    return m


def box(ext, T=None):
    b = trimesh.creation.box(extents=ext)
    if T is not None:
        b.apply_transform(T)
    return b


def beam(p0, p1, w, h):
    """Rectangular beam from p0 to p1 (in the y-z plane), width w along x, height h perpendicular."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    b = trimesh.creation.box(extents=(w, h, L))
    ang = np.arctan2(-d[1], d[2])  # rotation about x taking +z to d
    T = translation_matrix((p0 + p1) / 2) @ rotation_matrix(ang, [1, 0, 0])
    b.apply_transform(T)
    return b


def union(ms):
    return trimesh.boolean.union(ms, engine="manifold")


def diff(a, bs):
    return trimesh.boolean.difference([a] + list(bs), engine="manifold")


def plate_with_pocket(sticker, margin=2.0, thick=2.0):
    """Plate in local frame: face (sticker side) at z = 0 facing +z, back at z = -thick."""
    W, H = sticker[0] + POCKET_CLEAR + 2 * margin, sticker[1] + POCKET_CLEAR + 2 * margin
    p = box((W, H, thick), translation_matrix((0, 0, -thick / 2)))
    pocket = box((sticker[0] + POCKET_CLEAR, sticker[1] + POCKET_CLEAR, POCKET_DEPTH + 1),
                 translation_matrix((0, 0, -POCKET_DEPTH + (POCKET_DEPTH + 1) / 2)))
    # small up-arrow notch at the top edge so the sticker orientation is unambiguous
    notch = box((3, 1.2, 0.6), translation_matrix((0, H / 2 - 0.4, -0.2)))
    return diff(p, [pocket, notch]), (W, H)


def make_clutch(R=22.0, half_angle_deg=52.0, bar_t=1.5, bar_h=6.0, lute_gap=0.3):
    r_in = R + lute_gap
    r_out = r_in + bar_t
    a = np.radians(half_angle_deg)
    bar = annulus_sector(r_in, r_out, -a, a, 0.0, bar_h, n=96)
    # retention grooves on the tooth-facing (inner) side: 3 horizontal grooves 0.6 x 0.4 mm
    grooves = [annulus_sector(r_in - 0.5, r_in + 0.4, -a + 0.06, a - 0.06, y - 0.3, y + 0.3, n=96)
               for y in (1.5, 3.0, 4.5)]
    # vertical grooves (anti-rotation) every ~12 deg
    for ang in np.radians(np.arange(-42, 43, 14)):
        g = box((0.6, bar_h - 1.6, 0.9), translation_matrix((0, bar_h / 2, r_in + 0.25)))
        g.apply_transform(rotation_matrix(ang, [0, 1, 0]))
        grooves.append(g)
    bar = diff(bar, grooves)

    # stem: 6 mm wide (x) x 4 mm tall, from the lower half of the bar straight forward, then down/forward
    w, h = 6.0, 4.0
    yc = 2.0
    P0 = (0, yc, r_out - 0.6)
    P1 = (0, yc, r_out + 12.0)
    dn = np.radians(40)
    L2 = 30.0
    P2 = (0, yc - L2 * np.sin(dn), P1[2] + L2 * np.cos(dn))
    s1 = beam(P0, P1, w, h)
    s2 = beam(P1, P2, w, h)
    knee = box((w, h, h), translation_matrix(P1) @ rotation_matrix(-dn / 2, [1, 0, 0]))
    # JAW plate, face tilted 15 deg downward toward a camera slightly below the face
    plate, (W, H) = plate_with_pocket(JAW_STICKER, margin=2.0, thick=1.8)
    tilt = np.radians(15)
    Rt = rotation_matrix(tilt, [1, 0, 0])  # face normal -> (0, -sin, cos)
    attach_local = np.array([0, H / 2 - 6.0, -1.8, 1.0])  # stem meets the back of the upper plate part
    off = np.array(P2) + np.array([0, 0, 0.0]) - (Rt @ attach_local)[:3]
    Tp = translation_matrix(off) @ Rt
    plate.apply_transform(Tp)
    # gusset/boss on the back of the plate where the stem lands
    boss = box((w + 4, h + 4, 3.0), Tp @ translation_matrix((0, H / 2 - 6.0, -1.8 - 1.4)))
    s3 = beam(np.array(P2) - 3 * np.array([0, -np.sin(dn), np.cos(dn)]), np.array(P2) + np.array([0, 0, 1.0]), w, h)
    m = union([bar, s1, knee, s2, s3, boss, plate])
    info = dict(arch_radius_mm=R, lute_gap_mm=lute_gap, bar_inner_radius_mm=r_in, bar_thickness_mm=bar_t,
                bar_height_mm=bar_h, bar_arc_deg=2 * half_angle_deg,
                bar_chord_mm=round(2 * r_in * np.sin(a), 1), stem_section_mm=[w, h],
                stem_length_mm=round(12.0 + L2, 1), stem_down_angle_deg=40, plate_size_mm=[W, H],
                plate_tilt_down_deg=15, pocket_mm=[JAW_STICKER[0] + POCKET_CLEAR, JAW_STICKER[1] + POCKET_CLEAR, POCKET_DEPTH],
                plate_centre_mm=np.round(Tp[:3, 3] + (Rt[:3, :3] @ [0, 0, 0]), 1).tolist(),
                plate_face_normal=np.round(Rt[:3, :3] @ [0, 0, 1], 3).tolist())
    return m, info


def make_head_plate(tilt_deg=12.0, wing_angle_deg=25.0):
    plate, (W, H) = plate_with_pocket(HEAD_STICKER, margin=2.0, thick=2.0)
    # perimeter + centre rib frame behind the panel; its back is a plane tilted so the face looks down
    depth_top, t = 2.0, 2.0
    depth_bot = depth_top + H * np.tan(np.radians(tilt_deg))
    D = depth_bot + 1
    ribs_outer = box((W, H, D), translation_matrix((0, 0, -2.0 - D / 2)))
    hole_w = (W - 3 * t) / 2
    holes = [box((hole_w, H - 2 * t, D + 2), translation_matrix((sx * (t / 2 + hole_w / 2), 0, -2.0 - D / 2)))
             for sx in (-1, 1)]
    ribs = diff(ribs_outer, holes)
    # cut the rib back with the tilted plane: keep z > -2 - depth(y), depth linear top->bottom
    cutter = box((W + 40, H * 3, 60), translation_matrix((0, 0, -30)))
    ang = np.arctan2(depth_bot - depth_top, H)
    yc_plane = 0.0
    zc_plane = -2.0 - (depth_top + depth_bot) / 2
    cutter.apply_transform(translation_matrix((0, yc_plane, zc_plane)) @ rotation_matrix(-ang, [1, 0, 0]))
    ribs = diff(ribs, [cutter])
    body = union([plate, ribs])
    # wings angled back toward the temples, each with two 26 x 2.5 mm slots (ladder-lock threading)
    wing_len, wing_h, wing_t = 20.0, 34.0, 2.0
    wings = []
    for sx in (-1, 1):
        wg = box((wing_len, wing_h, wing_t), translation_matrix((sx * wing_len / 2, 0, -wing_t / 2)))
        slots = [box((2.6, 26.0, wing_t + 2), translation_matrix((sx * xs, 0, -wing_t / 2))) for xs in (7.0, 13.5)]
        wg = diff(wg, slots)
        # hinge at panel edge, mid-depth of the rib frame, rotate back
        zh = -2.0 - (depth_top + depth_bot) / 2 + wing_t / 2
        wg.apply_transform(translation_matrix((sx * (W / 2 - 1.0), 0, zh)) @ rotation_matrix(sx * np.radians(wing_angle_deg), [0, 1, 0]))
        wings.append(wg)
    m = union([body] + wings)
    info = dict(sticker_mm=list(HEAD_STICKER), pocket_mm=[HEAD_STICKER[0] + POCKET_CLEAR, HEAD_STICKER[1] + POCKET_CLEAR, POCKET_DEPTH],
                panel_mm=[W, H], tilt_down_deg=tilt_deg, standoff_top_mm=2.0 + depth_top, standoff_bottom_mm=round(2.0 + depth_bot, 1),
                wing_angle_back_deg=wing_angle_deg, slots="2 per wing, 26.0 x 2.6 mm, for 20-25 mm elastic band")
    return m, info


def check(m, name):
    r = dict(name=name, watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent),
             volume_positive=bool(m.volume > 0), euler=int(m.euler_number), bodies=len(m.split(only_watertight=False)),
             volume_cm3=round(m.volume / 1000, 3), mass_g_resin=round(m.volume / 1000 * RESIN_DENSITY, 2),
             bbox_mm=np.round(m.extents, 1).tolist(), n_faces=len(m.faces))
    return r


def preview(m, path, title, elev=15, azims=(-60, 0, 90)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15, 5))
    V = m.vertices[:, [0, 2, 1]]  # plot with y (up) as matplotlib z
    tri = V[m.faces]
    n = m.face_normals[:, [0, 2, 1]]
    light = np.array([0.3, -0.6, 0.75]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    cols = np.c_[shade * 0.75, shade * 0.8, shade * 0.9, np.ones_like(shade)]
    for k, az in enumerate(azims):
        ax = fig.add_subplot(1, len(azims), k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolor="none"))
        lo, hi = V.min(0), V.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=az)
        ax.set_xlabel("x (L/R)"); ax.set_ylabel("z (anterior)"); ax.set_zlabel("y (up)")
        ax.set_title(f"{title} az={az}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--arch-radius", type=float, default=22.0, help="labial arch radius at the cervical third, mm")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    res = {}
    clutch, ci = make_clutch(R=a.arch_radius)
    head, hi = make_head_plate()
    for m, name, info in ((clutch, "jaw_clutch_paddle", ci), (head, "head_marker_plate", hi)):
        p = os.path.join(a.outdir, name + ".stl")
        m.export(p)
        back = trimesh.load(p)
        r = check(back, name)
        r["design"] = info
        res[name] = r
        if not a.no_preview:
            os.makedirs(os.path.join(a.outdir, "verify"), exist_ok=True)
            preview(back, os.path.join(a.outdir, "verify", name + "_preview.png"), name)
    json.dump(res, open(os.path.join(a.outdir, "verify", "stl_checks.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
