#!/usr/bin/env python3
"""Generate synthetic JawTrack test datasets with exact ground truth.

    python3 make_synth.py                 # synth_quick/ then synth_01/
    python3 make_synth.py --variant quick # only synth_quick/ (10 s)
    python3 make_synth.py --variant full --skip-calib

Deterministic (seeded). See README.md for file contents and definitions.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from synthlib import dentition as D  # noqa: E402
from synthlib import gtdefs as G  # noqa: E402
from synthlib.kinematics import JawModel, build_timeline  # noqa: E402

FPS = 60.0
W, H = 1920, 1080
K_TRUE = np.array([[1391.7, 0.0, 963.4], [0.0, 1389.9, 543.1], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([0.05, -0.03, 0.0004, -0.0003, 0.0])
TRUE_ANGLES = dict(theta_L=35.0, theta_R=38.0, bennett_L=10.0, bennett_R=8.0)


def flat(T):
    return [float(x) for x in np.asarray(T).reshape(-1)]


def tolist(v):
    return [float(x) for x in np.asarray(v).reshape(-1)]


# ============================================================================ dentition
def make_dentition(cache_dir, verbose=True):
    """Build (or load cached) upper/lower meshes + landmarks. Shared by all variants."""
    import trimesh
    cu, cl, cj = (os.path.join(cache_dir, n) for n in ("upper.stl", "lower.stl", "landmarks.json"))
    if all(os.path.exists(p) for p in (cu, cl, cj)):
        U, L = trimesh.load(cu), trimesh.load(cl)
        lm = {k: np.array(v) for k, v in json.load(open(cj)).items()}
        return U, L, lm
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    up, lo, ua, la = D.build_dentition(verbose=False)
    ub, lb = D.make_bases(up, lo, ua, la)
    U = D.union_meshes([t["mesh"] for t in up] + [ub])
    D.refine_contacts(lo, U, verbose=False)
    L = D.union_meshes([t["mesh"] for t in lo] + [lb])
    lm = D.landmarks(up, lo, U, L)
    if verbose:
        print(f"  dentition built in {time.time() - t0:.1f}s: upper {len(U.faces)} tris, lower {len(L.faces)} tris")
    U.export(cu)
    L.export(cl)
    json.dump({k: tolist(v) for k, v in lm.items()}, open(cj, "w"), indent=1)
    return U, L, lm


def check_occlusion(U, L):
    v = L.vertices
    sel = v[:, 1] > -4
    sd = D.signed_dist(U, v[sel])
    v2 = U.vertices
    sd2 = D.signed_dist(L, v2[v2[:, 1] < 6])
    return dict(max_penetration_lower_into_upper_mm=float(sd.max()),
                max_penetration_upper_into_lower_mm=float(sd2.max()),
                n_lower_vertices_within_0p1mm=int((np.abs(sd) < 0.1).sum()),
                upper_watertight=bool(U.is_watertight), lower_watertight=bool(L.is_watertight),
                upper_triangles=int(len(U.faces)), lower_triangles=int(len(L.faces)))


def true_landmarks(lm):
    ip = lm["lower_incisal_point"]
    fr = G.occlusal_frame(ip, lm["36_mb"], lm["46_mb"])
    cL, cR = G.bonwill_condyles(fr)
    return dict(frame=fr, condyle_left=cL, condyle_right=cR, incisal_point=ip)


# ============================================================================ kinematics
def make_kinematics(variant, U, L, lm, verbose=True, iters=2):
    """Generate T_S(t). The path angles are corrected by a fixed-point iteration so that the
    parameters MEASURED by the shared definitions (gtdefs) land on the intended values."""
    tl = true_landmarks(lm)
    model = JawModel(tl, U, L, **TRUE_ANGLES)
    segs, events, dur = build_timeline(variant, FPS)
    for it in range(iters + 1):
        amps = model.tune()
        times, Ts, phis, state = model.generate(segs, events, dur, FPS, verbose=verbose and it == iters)
        kin = dict(times=times, Ts=Ts, phis=phis, state=state, segments=segs, amps=amps, tl=tl, model=model)
        if it == iters:
            break
        P = ground_truth_params(kin, lm)
        for key, attr, tgt in (("sagittal_condylar_inclination_left_deg", "thL", "theta_L"),
                               ("sagittal_condylar_inclination_right_deg", "thR", "theta_R"),
                               ("bennett_angle_left_deg", "bL", "bennett_L"),
                               ("bennett_angle_right_deg", "bR", "bennett_R")):
            if P.get(key) is not None:
                setattr(model, attr, getattr(model, attr) + np.radians(TRUE_ANGLES[tgt] - P[key]))
    return kin


def ground_truth_params(kin, lm):
    tl = kin["tl"]
    P, info = G.compute_parameters(kin["Ts"], kin["times"], kin["segments"], tl["incisal_point"],
                                   lm["36_mb"], lm["46_mb"],
                                   condyles=(tl["condyle_left"], tl["condyle_right"]))
    fr = tl["frame"]
    P["kinematic_hinge_axis"] = dict(point=tolist(0.5 * (tl["condyle_left"] + tl["condyle_right"])),
                                     direction=tolist((tl["condyle_left"] - tl["condyle_right"]) / G.BONWILL_SIDE_MM),
                                     fit_rms_mm=0.0)
    P["notes"] = P["notes"] + [
        "Angles relative to the occlusal plane (IP, MB36, MB46).",
        f"Inclinations read at {G.PROTRUSIVE_RECORD_MM} mm incisal protrusion, Bennett at {G.LATERAL_RECORD_MM} mm incisal laterotrusion (median over repetitions).",
    ]
    return P


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["quick", "full", "all"], default="all")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--skip-video", action="store_true", help="only STLs + ground truth (fast)")
    ap.add_argument("--skip-calib", action="store_true", help="do not render charuco_calib.mp4")
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()
    from synthlib import scene  # heavy imports only when needed

    cache = os.path.join(args.outdir, ".cache")
    print("[1] dentition")
    U, L, lm = make_dentition(cache)
    occ = check_occlusion(U, L)
    print("    occlusion check:", occ)
    variants = ["quick", "full"] if args.variant == "all" else [args.variant]
    for v in variants:
        name = "synth_quick" if v == "quick" else "synth_01"
        out = os.path.join(args.outdir, name)
        os.makedirs(out, exist_ok=True)
        print(f"[2] {name}: kinematics")
        kin = make_kinematics(v, U, L, lm)
        P = ground_truth_params(kin, lm)
        print("    true parameters:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in P.items() if k not in ("notes", "kinematic_hinge_axis")})
        scene.write_dataset(out, v, U, L, lm, kin, P, occ, seed=args.seed + (0 if v == "quick" else 1),
                            K=K_TRUE, dist=DIST_TRUE, fps=FPS, size=(W, H),
                            skip_video=args.skip_video, skip_calib=args.skip_calib)
    print("done.")


if __name__ == "__main__":
    main()
