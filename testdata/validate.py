#!/usr/bin/env python3
"""Validate a pipeline motion.json against synthetic ground truth.

    python3 validate.py --motion ../pipeline/out/motion.json --gt synth_01/ground_truth.json \
        [--raw ../pipeline/out/raw_tracking.json] [--registration reg.json] [--outdir DIR]

Writes DIR/validation_report.md and DIR/*.png (default DIR = <motion dir>/validation) and prints
the report. Exit code: 0 = PASS, 1 = WARN, 2 = FAIL, 3 = input error.
Robust to missing optional fields (invalid frames, missing parameters/landmarks/segments).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthlib import gtdefs as G  # noqa: E402

# target thresholds: (PASS if <= target, WARN if <= warn, else FAIL)
TARGETS = {
    "incisal_median_mm": (0.3, 0.6),
    "incisal_p95_mm": (0.8, 1.6),
    "condylar_inclination_err_deg": (5.0, 10.0),
    "bennett_err_deg": (5.0, 10.0),
}


def M4(v):
    a = np.asarray(v, float)
    return a.reshape(4, 4) if a.size == 16 else a


def rot_angle_deg(R):
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, median=np.nan, p95=np.nan, max=np.nan, mean=np.nan)
    return dict(n=int(len(x)), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                max=float(np.max(x)), mean=float(np.mean(x)))


def verdict(val, key):
    if val is None or not np.isfinite(val):
        return "N/A"
    t, w = TARGETS[key]
    return "PASS" if val <= t else ("WARN" if val <= w else "FAIL")


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def load_json(p, what):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot read {what} '{p}': {e}", file=sys.stderr)
        sys.exit(3)


def find_registration(motion, reg_path):
    cand = []
    if reg_path:
        cand.append(load_json(reg_path, "registration"))
    cand.append(motion.get("registration") or {})
    cand.append(motion)
    for c in cand:
        if not isinstance(c, dict):
            continue
        for k in ("X", "X_T_H_S", "T_H_S"):
            if k in c and c[k] is not None:
                try:
                    return M4(c[k]), k
                except Exception:  # noqa: BLE001
                    pass
        if "T_S_H" in c and c["T_S_H"] is not None:
            return np.linalg.inv(M4(c["T_S_H"])), "inv(T_S_H)"
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", required=True, help="pipeline motion.json (SPEC 5.2)")
    ap.add_argument("--gt", required=True, help="ground_truth.json from make_synth.py")
    ap.add_argument("--raw", help="optional pipeline raw_tracking.json (SPEC 5.1)")
    ap.add_argument("--registration", help="optional json with X (=T_H_S, 16 floats row-major) or T_S_H")
    ap.add_argument("--outdir", help="output dir (default: <motion dir>/validation)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    gt = load_json(args.gt, "ground truth")
    mo = load_json(args.motion, "motion")
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.motion)), "validation")
    os.makedirs(outdir, exist_ok=True)

    fps = float(gt["fps"])
    gt_t = np.array([f["t"] for f in gt["frames"]])
    gt_T = np.array([M4(f["T"]) for f in gt["frames"]])
    lmk = gt["landmarks_mm"]
    ip = np.array(lmk["incisal_point"])
    molars = [np.array(v) for v in lmk.get("lower_molar_cusp_points", {}).values()] or \
        [np.array(p) for p in lmk["occlusal_plane_points"][1:]]
    segs = gt["segments"]
    lines = []
    out = lines.append
    warnings = []

    # ---------------------------------------------------------------- frame matching
    frames = mo.get("frames") or []
    off = float((mo.get("source") or {}).get("video_offset_s") or 0.0)
    est_T = np.full((len(gt_t), 4, 4), np.nan)
    est_valid = np.zeros(len(gt_t), bool)
    n_bad = 0
    for k, f in enumerate(frames):
        try:
            t = float(f["t"]) + off if "t" in f and f["t"] is not None else k / float(mo.get("fps") or fps)
            T = M4(f["T"])
            if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                raise ValueError
        except Exception:  # noqa: BLE001
            n_bad += 1
            continue
        i = int(round(t * fps))
        if i < 0 or i >= len(gt_t) or abs(gt_t[i] - t) > 0.5 / fps + 1e-6:
            continue
        if f.get("valid", True) is False:
            continue
        est_T[i] = T
        est_valid[i] = True
    if n_bad:
        warnings.append(f"{n_bad} motion frames had missing/invalid T and were ignored")
    tracked = est_valid.mean()
    if tracked == 0:
        print("ERROR: no motion frames matched the ground-truth timeline (check t / fps / video_offset_s)", file=sys.stderr)
        sys.exit(3)

    # ---------------------------------------------------------------- per-frame errors
    v = est_valid
    e_inc = np.full(len(gt_t), np.nan)
    e_mol = np.full(len(gt_t), np.nan)
    e_rot = np.full(len(gt_t), np.nan)
    pe = G.apply(est_T[v], ip)
    pg = G.apply(gt_T[v], ip)
    e_inc[v] = np.linalg.norm(pe - pg, axis=1)
    em = [np.linalg.norm(G.apply(est_T[v], m) - G.apply(gt_T[v], m), axis=1) for m in molars]
    e_mol[v] = np.max(em, axis=0)
    e_rot[v] = [rot_angle_deg(a[:3, :3] @ b[:3, :3].T) for a, b in zip(est_T[v], gt_T[v])]
    S_inc, S_mol, S_rot = stats(e_inc), stats(e_mol), stats(e_rot)

    out("# JawTrack validation report\n")
    out(f"- motion: `{os.path.abspath(args.motion)}`")
    out(f"- ground truth: `{os.path.abspath(args.gt)}` (variant `{gt.get('variant')}`, {len(gt_t)} frames @ {fps:g} fps)")
    out(f"- matched valid frames: {int(v.sum())}/{len(gt_t)} (tracked fraction {tracked:.3f}); pipeline-reported tracked_fraction: {fmt((mo.get('quality') or {}).get('tracked_fraction'))}")
    out("")
    out("## Motion accuracy (lower jaw in upper-STL frame)\n")
    out("| metric | n | median | 95th | max | mean |")
    out("|---|---|---|---|---|---|")
    for name, S in (("incisal-point error (mm)", S_inc), ("lower-molar MB cusp error, worse of 36/46 (mm)", S_mol), ("rotation error (deg)", S_rot)):
        out(f"| {name} | {S['n']} | {fmt(S['median'])} | {fmt(S['p95'])} | {fmt(S['max'])} | {fmt(S['mean'])} |")
    out("")
    out("### Per segment (incisal-point error, mm / rotation error, deg)\n")
    out("| segment | t0–t1 (s) | n | inc. median | inc. 95th | inc. max | rot. median | rot. max |")
    out("|---|---|---|---|---|---|---|---|")
    for s in segs:
        m = (gt_t >= s["t0"] - 1e-9) & (gt_t <= s["t1"] + 1e-9)
        a, r = stats(e_inc[m]), stats(e_rot[m])
        out(f"| {s['label']} | {s['t0']:.2f}–{s['t1']:.2f} | {a['n']} | {fmt(a['median'])} | {fmt(a['p95'])} | {fmt(a['max'])} | {fmt(r['median'], 2)} | {fmt(r['max'], 2)} |")
    out("")
    # MI jitter
    mi = G.seg_mask(gt_t, segs, "MI") & v
    if mi.any():
        d = G.apply(est_T[mi], ip) - ip
        out(f"Incisal-point RMS deviation from MI during MI holds (jitter + bias): {np.sqrt((d ** 2).sum(1).mean()):.3f} mm "
            f"(pipeline reported jitter_mm_rms_in_MI: {fmt((mo.get('quality') or {}).get('jitter_mm_rms_in_MI'))})\n")

    # ---------------------------------------------------------------- parameters
    Pg = gt.get("parameters") or {}
    Pe = mo.get("parameters") or {}
    est_T_fill = est_T.copy()
    est_T_fill[~v] = np.eye(4)  # invalid frames: excluded via valid mask below
    Pr, _ = G.compute_parameters(est_T_fill, gt_t, segs, ip, *[np.array(p) for p in lmk["occlusal_plane_points"][1:]],
                                 condyles=(np.array(lmk["condyle_left"]), np.array(lmk["condyle_right"])), valid=v)
    keys = ["max_opening_mm", "max_protrusion_mm", "max_laterotrusion_right_mm", "max_laterotrusion_left_mm",
            "sagittal_condylar_inclination_left_deg", "sagittal_condylar_inclination_right_deg",
            "bennett_angle_left_deg", "bennett_angle_right_deg", "incisal_guidance_deg"]
    out("## Parameters\n")
    out("*reported* = value in motion.json; *recomputed* = the GT definitions (README) applied to the pipeline's T(t) "
        "with the TRUE landmarks (isolates tracking error from definition/landmark differences).\n")
    out("| parameter | true | reported | abs err | recomputed | abs err | verdict |")
    out("|---|---|---|---|---|---|---|")
    perr = {}
    for k in keys:
        g = Pg.get(k)
        e = Pe.get(k)
        r = Pr.get(k)
        ea = abs(e - g) if isinstance(e, (int, float)) and isinstance(g, (int, float)) else None
        ra = abs(r - g) if isinstance(r, (int, float)) and isinstance(g, (int, float)) else None
        vk = None
        if "condylar_inclination" in k:
            vk = "condylar_inclination_err_deg"
        elif "bennett" in k:
            vk = "bennett_err_deg"
        vd = verdict(ea, vk) if vk else ""
        if vk:
            perr[k] = (ea, ra, vd)
        out(f"| {k} | {fmt(g, 2)} | {fmt(e, 2)} | {fmt(ea, 2)} | {fmt(r, 2)} | {fmt(ra, 2)} | {vd} |")
    out("")
    # hinge axis
    ha_g = Pg.get("kinematic_hinge_axis")
    ha_e = Pe.get("kinematic_hinge_axis")
    if ha_g and ha_e and ha_e.get("point") is not None and ha_e.get("direction") is not None:
        dg, de = np.array(ha_g["direction"]), np.array(ha_e["direction"])
        de = de / np.linalg.norm(de)
        ang = np.degrees(np.arccos(min(1, abs(dg @ de))))
        out_pts = []
        for c in (np.array(lmk["condyle_left"]), np.array(lmk["condyle_right"])):
            w = c - np.array(ha_e["point"])
            out_pts.append(np.linalg.norm(w - (w @ de) * de))
        out(f"Kinematic hinge axis: direction error {ang:.2f} deg; distance of true condyles from estimated axis "
            f"L {out_pts[0]:.2f} mm / R {out_pts[1]:.2f} mm; reported fit_rms {fmt(ha_e.get('fit_rms_mm'))} mm\n")
    else:
        out("Kinematic hinge axis: not reported (or null).\n")
    # landmarks
    le = mo.get("landmarks_mm") or {}
    out("## Landmarks (estimated vs true, mm)\n")
    out("| landmark | error (mm) |")
    out("|---|---|")
    for k in ("incisal_point", "condyle_left", "condyle_right"):
        if le.get(k) is not None:
            out(f"| {k} | {np.linalg.norm(np.array(le[k]) - np.array(lmk[k])):.2f} |")
        else:
            out(f"| {k} | not reported |")
    op = le.get("occlusal_plane") or {}
    if op.get("normal") is not None:
        ne = np.array(op["normal"], float)
        ne /= np.linalg.norm(ne)
        out(f"| occlusal plane normal (deg) | {np.degrees(np.arccos(min(1, abs(ne @ np.array(lmk['occlusal_plane']['normal']))))):.2f} |")
    out(f"\ncondyle_method reported: `{le.get('condyle_method')}` (true condyles = Bonwill points = rotation centres)\n")

    # ---------------------------------------------------------------- registration
    out("## Registration\n")
    Xg = M4(gt["rig"]["X_T_H_S"])
    Xe, src = find_registration(mo, args.registration)
    rms = (mo.get("quality") or {}).get("registration_rms_px")
    out(f"Pipeline-reported registration reprojection RMS: {fmt(rms, 2)} px")
    if Xe is not None:
        dR = rot_angle_deg(Xe[:3, :3] @ Xg[:3, :3].T)
        out(f"\nImplied X (`{src}`) vs true X = T_H_S: rotation error {dR:.3f} deg")
        out("\n| STL point mapped by X | error (mm) |\n|---|---|")
        for nm, p in (("incisal_point", ip), ("condyle_left", np.array(lmk["condyle_left"])), ("condyle_right", np.array(lmk["condyle_right"]))):
            e = np.linalg.norm((Xe[:3, :3] @ p + Xe[:3, 3]) - (Xg[:3, :3] @ p + Xg[:3, 3]))
            out(f"| {nm} | {e:.3f} |")
    else:
        out("\nNo X found (looked for `registration.X` / `X` / `T_H_S` in motion.json and --registration). "
            "Registration error is still reflected in the motion errors above.")
    out("")

    # ---------------------------------------------------------------- raw tracking
    if args.raw:
        raw = load_json(args.raw, "raw tracking")
        rtrue_path = os.path.join(os.path.dirname(os.path.abspath(args.gt)), "raw_tracking_true.json")
        out("## Raw board tracking (stage 1)\n")
        cam = raw.get("camera") or {}
        if cam.get("K") is not None:
            Ke, Kg = np.array(cam["K"]), np.array(gt["camera"]["K"])
            out(f"Camera: fx err {Ke[0, 0] - Kg[0, 0]:+.1f} px, fy err {Ke[1, 1] - Kg[1, 1]:+.1f} px, cx err {Ke[0, 2] - Kg[0, 2]:+.1f}, "
                f"cy err {Ke[1, 2] - Kg[1, 2]:+.1f}; dist est {np.round(cam.get('dist') or [], 4).tolist()} vs true {np.round(gt['camera']['dist'], 4).tolist()} (source {cam.get('source')})\n")
        if os.path.exists(rtrue_path):
            rt = load_json(rtrue_path, "raw truth")
            tr = {int(f["i"]): f for f in rt["frames"]}
            res = {"head": ([], []), "jaw": ([], []), "M": ([], [])}
            det = {"head": 0, "jaw": 0}
            for k, f in enumerate(raw.get("frames") or []):
                t = f.get("t")
                i = int(round(float(t) * fps)) if t is not None else int(f.get("i", k))
                if i not in tr:
                    continue
                Ts = {}
                for b in ("head", "jaw"):
                    fb = f.get(b)
                    if not fb or fb.get("T") is None:
                        continue
                    Te, Tg = M4(fb["T"]), M4(tr[i][b]["T"])
                    det[b] += 1
                    Ts[b] = (Te, Tg)
                    res[b][0].append(np.linalg.norm(Te[:3, 3] - Tg[:3, 3]))
                    res[b][1].append(rot_angle_deg(Te[:3, :3] @ Tg[:3, :3].T))
                if len(Ts) == 2:
                    Me = np.linalg.inv(Ts["head"][0]) @ Ts["jaw"][0]
                    Mg = np.linalg.inv(Ts["head"][1]) @ Ts["jaw"][1]
                    res["M"][0].append(np.linalg.norm(Me[:3, 3] - Mg[:3, 3]))
                    res["M"][1].append(rot_angle_deg(Me[:3, :3] @ Mg[:3, :3].T))
            out(f"Detected: head {det['head']}/{len(tr)} frames, jaw {det['jaw']}/{len(tr)} frames\n")
            out("| pose | origin err median (mm) | 95th | rot err median (deg) | 95th |")
            out("|---|---|---|---|---|")
            for b, nm in (("head", "T_C_H"), ("jaw", "T_C_J"), ("M", "M = inv(T_C_H) T_C_J")):
                a, r = stats(res[b][0]), stats(res[b][1])
                out(f"| {nm} | {fmt(a['median'])} | {fmt(a['p95'])} | {fmt(r['median'])} | {fmt(r['p95'])} |")
            out("\n(board-origin errors are dominated by depth error along the viewing ray; M is what matters)\n")
        else:
            out("raw_tracking_true.json not found next to ground truth; skipped.\n")

    # ---------------------------------------------------------------- verdict
    out("## Verdict\n")
    checks = [("incisal-point median error", S_inc["median"], "incisal_median_mm", "mm"),
              ("incisal-point 95th percentile error", S_inc["p95"], "incisal_p95_mm", "mm")]
    for k, (ea, ra, vd) in perr.items():
        if Pg.get(k) is None:
            continue  # parameter not defined for this dataset (e.g. no laterotrusion in synth_quick)
        checks.append((k + " |err| (reported)", ea, "condylar_inclination_err_deg" if "condylar" in k else "bennett_err_deg", "deg"))
    out("| check | value | target | verdict |")
    out("|---|---|---|---|")
    vs = []
    for nm, val, key, unit in checks:
        vd = verdict(val, key)
        vs.append(vd)
        out(f"| {nm} | {fmt(val)} {unit} | ≤ {TARGETS[key][0]} {unit} (WARN ≤ {TARGETS[key][1]}) | **{vd}** |")
    if tracked < 0.9:
        warnings.append(f"only {100 * tracked:.1f}% of frames valid")
        vs.append("WARN")
    overall = "FAIL" if "FAIL" in vs else ("WARN" if ("WARN" in vs or "N/A" in vs) else "PASS")
    out(f"\n**OVERALL: {overall}**\n")
    if "N/A" in vs:
        out("(N/A = parameter missing in motion.json → counted as WARN)\n")
    for wmsg in warnings:
        out(f"- warning: {wmsg}")

    # ---------------------------------------------------------------- plots
    if not args.no_plots:
        try:
            plots(outdir, gt_t, segs, e_inc, e_mol, e_rot, est_T, gt_T, v, ip, lmk)
            out("\n## Plots\n\n![errors](errors_vs_time.png)\n\n![incisal path](incisal_path.png)\n")
        except Exception as e:  # noqa: BLE001
            out(f"\n(plotting failed: {e})")
    rep = "\n".join(lines)
    with open(os.path.join(outdir, "validation_report.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)
    print(f"\n[report written to {os.path.join(outdir, 'validation_report.md')}]")
    sys.exit({"PASS": 0, "WARN": 1, "FAIL": 2}[overall])


def plots(outdir, t, segs, e_inc, e_mol, e_rot, est_T, gt_T, v, ip, lmk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"MI": "#dddddd", "open_close": "#cfe2f3", "protrusion": "#d9ead3", "laterotrusion_right": "#fce5cd",
            "laterotrusion_left": "#fff2cc", "chewing": "#ead1dc"}
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for a, y, lab in zip(ax, (e_inc, e_mol, e_rot), ("incisal err (mm)", "molar cusp err (mm)", "rotation err (deg)")):
        for s in segs:
            a.axvspan(s["t0"], s["t1"], color=cols.get(s["label"], "#eee"), lw=0)
        a.plot(t, y, lw=0.8, color="#1f4e79")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].axhline(0.3, color="g", ls="--", lw=0.8)
    ax[0].axhline(0.8, color="r", ls="--", lw=0.8)
    ax[-1].set_xlabel("t (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()]
    ax[0].legend(handles, list(cols), ncol=6, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "errors_vs_time.png"), dpi=110)
    plt.close(fig)
    fr = G.occlusal_frame(ip, *[np.array(p) for p in lmk["occlusal_plane_points"][1:]])
    B = np.stack([fr["l"], fr["n"], fr["f"]])
    dg = (G.apply(gt_T, ip) - ip) @ B.T
    de = np.full_like(dg, np.nan)
    de[v] = (G.apply(est_T[v], ip) - ip) @ B.T
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    for a, (i, j, ti) in zip(ax, ((2, 1, "sagittal (anterior vs superior)"), (0, 1, "frontal (left vs superior)"), (0, 2, "horizontal (left vs anterior)"))):
        a.plot(dg[:, i], dg[:, j], color="k", lw=1.2, label="truth")
        a.plot(de[:, i], de[:, j], color="#c00000", lw=0.7, alpha=0.8, label="pipeline")
        a.set_title(ti)
        a.set_aspect("equal", adjustable="datalim")
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.suptitle("lower incisal-point path in the occlusal frame (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "incisal_path.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
