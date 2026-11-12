"""Stage 3 (report): PNG plots + report.md from motion.json."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import se3  # noqa: E402
from .io import read_json  # noqa: E402
from .landmarks import OcclusalFrame  # noqa: E402

log = logging.getLogger("jawtrack.report")

SEG_COLORS = {"MI": "#d9d9d9", "open_close": "#8fb8de", "protrusion": "#f4b183", "laterotrusion_right": "#a9d18e",
              "laterotrusion_left": "#c5a3d9", "chewing": "#ffd966"}
PATH_COLORS = {"open_close": "#1f5f99", "protrusion": "#c55a11", "laterotrusion_right": "#548235",
               "laterotrusion_left": "#7030a0", "chewing": "#bf9000", "MI": "#7f7f7f"}


def _load(motion):
    t = np.array([f["t"] for f in motion["frames"]])
    T = np.array([se3.from_list(f["T"]) for f in motion["frames"]])
    ok = np.array([f["valid"] for f in motion["frames"]])
    lm = motion["landmarks_mm"]
    ofd = lm.get("occlusal_frame") or {}
    of = OcclusalFrame(lm["occlusal_plane"]["point"], lm["occlusal_plane"]["normal"],
                       ofd.get("anterior") or (np.asarray(lm["incisal_point"]) -
                                               0.5 * (np.asarray(lm["condyle_left"]) + np.asarray(lm["condyle_right"]))))
    return t, T, ok, lm, of


def _disp(T, p, of):
    return of.to_local(np.einsum("nij,j->ni", T[:, :3, :3], p) + T[:, :3, 3] - p)  # left, up, ant


def _seg_label(t, segs):
    lab = np.array(["MI"] * len(t), dtype=object)
    for s in segs:
        lab[(t >= s["t0"]) & (t <= s["t1"])] = s["label"]
    return lab


def make_report(motion_path, outdir):
    motion = read_json(motion_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t, T, ok, lm, of = _load(motion)
    segs = motion["segments"]
    lab = _seg_label(t, segs)
    inc = np.asarray(lm["incisal_point"])
    di = _disp(T, inc, of)
    dL = _disp(T, np.asarray(lm["condyle_left"]), of)
    dR = _disp(T, np.asarray(lm["condyle_right"]), of)
    files = []

    # 1. Posselt-style incisal envelope: frontal, sagittal, horizontal
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    views = [("Frontal (lateral vs vertical)", 0, 1, "patient's right  <-  lateral [mm]  ->  left", "vertical [mm] (up +)", 1),
             ("Sagittal (anterior vs vertical)", 2, 1, "anterior [mm]", "vertical [mm] (up +)", 1),
             ("Horizontal (lateral vs anterior)", 0, 2, "patient's right  <-  lateral [mm]  ->  left", "anterior [mm]", -1)]
    for a, (title, i, j, xl, yl, xs) in zip(ax, views):
        for L in PATH_COLORS:
            m = ok & (lab == L)
            if m.any():
                a.plot(di[m, i], di[m, j], ".", ms=2, color=PATH_COLORS[L], label=L)
        a.plot(0, 0, "k+", ms=12)
        a.set_title(title); a.set_xlabel(xl); a.set_ylabel(yl)
        a.set_aspect("equal", adjustable="datalim"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, markerscale=4, loc="lower right")
    fig.suptitle("Incisal point trajectories (occlusal-plane frame, origin = MI)")
    fig.tight_layout()
    f = outdir / "incisal_envelope.png"; fig.savefig(f, dpi=120); plt.close(fig); files.append(f)

    # 2. Condyle traces sagittal
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for a, d, name in zip(ax, (dR, dL), ("right", "left")):
        for L in PATH_COLORS:
            m = ok & (lab == L)
            if m.any():
                a.plot(d[m, 2], d[m, 1], ".", ms=2, color=PATH_COLORS[L], label=L)
        a.set_title(f"Condyle {name} - sagittal"); a.set_xlabel("anterior [mm]"); a.grid(alpha=0.3)
        a.set_aspect("equal", adjustable="datalim")
    ax[0].set_ylabel("vertical [mm] (up +)")
    ax[1].legend(fontsize=7, markerscale=4)
    P = motion["parameters"]
    fig.suptitle(f"Condyle points ({lm.get('condyle_method')}); SCI L {_fmt(P.get('sagittal_condylar_inclination_left_deg'))} deg, "
                 f"R {_fmt(P.get('sagittal_condylar_inclination_right_deg'))} deg")
    fig.tight_layout()
    f = outdir / "condyle_sagittal.png"; fig.savefig(f, dpi=120); plt.close(fig); files.append(f)

    # 3. Time series with segments
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for a in ax:
        for s in segs:
            a.axvspan(s["t0"], s["t1"], color=SEG_COLORS.get(s["label"], "#eee"), alpha=0.5, lw=0)
    names = ["lateral (left +)", "vertical (up +)", "anterior (+)"]
    for k, c in zip(range(3), ("#548235", "#1f5f99", "#c55a11")):
        ax[0].plot(t, np.where(ok, di[:, k], np.nan), color=c, lw=1, label=names[k])
    ax[0].set_ylabel("incisal disp. [mm]"); ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=0.3)
    from scipy.spatial.transform import Rotation
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(T[:, :3, :3]).as_rotvec(), axis=1))
    ax[1].plot(t, np.where(ok, ang, np.nan), color="#333", lw=1)
    ax[1].set_ylabel("jaw rotation [deg]"); ax[1].grid(alpha=0.3)
    q = np.array([f["quality"] for f in motion["frames"]])
    ax[2].plot(t, q, color="#7f7f7f", lw=1, label="quality")
    ax[2].plot(t, (~ok).astype(float), color="#c00000", lw=1, label="invalid")
    ax[2].set_ylabel("quality"); ax[2].set_xlabel("time [s]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in SEG_COLORS.values()]
    ax[0].figure.legend(handles, SEG_COLORS.keys(), loc="upper center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = outdir / "timeseries.png"; fig.savefig(f, dpi=110); plt.close(fig); files.append(f)

    # 4. Quality: MI jitter histogram + per-frame reprojection (from quality block)
    Q = motion["quality"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mi = ok & (lab == "MI")
    if mi.any():
        r = np.linalg.norm(di[mi] - di[mi].mean(axis=0), axis=1)
        ax[0].hist(r, bins=40, color="#1f5f99")
    ax[0].set_title(f"Incisal scatter in MI segments (jitter RMS {_fmt(Q.get('jitter_mm_rms_in_MI'), 3)} mm)")
    ax[0].set_xlabel("distance from MI mean [mm]")
    txt = [f"tracked fraction: {Q.get('tracked_fraction', 0):.3f}",
           f"registration RMS: {_fmt(Q.get('registration_rms_px'))} px",
           f"jitter in MI (filtered): {_fmt(Q.get('jitter_mm_rms_in_MI'), 3)} mm",
           f"jitter in MI (unfiltered): {_fmt(Q.get('jitter_mm_rms_in_MI_unfiltered'), 3)} mm",
           f"MI start vs end: {_fmt(Q.get('mi_repeatability_mm'), 3)} mm"]
    mr = Q.get("marker_reproj_px") or {}
    txt += [f"marker reproj head median/p95: {_fmt(mr.get('head_median'), 2)}/{_fmt(mr.get('head_p95'), 2)} px",
            f"marker reproj jaw median/p95: {_fmt(mr.get('jaw_median'), 2)}/{_fmt(mr.get('jaw_p95'), 2)} px"]
    ax[1].axis("off"); ax[1].text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=10)
    fig.tight_layout()
    f = outdir / "quality.png"; fig.savefig(f, dpi=110); plt.close(fig); files.append(f)

    # report.md
    md = ["# JawTrack report", "", f"Source: `{motion['source'].get('video')}`", "",
          "## Parameters (angles relative to the occlusal plane)", "", "| parameter | value |", "|---|---|"]
    for k, v in P.items():
        if k in ("notes", "kinematic_hinge_axis"):
            continue
        md.append(f"| {k} | {_fmt(v)} |")
    ax_ = P.get("kinematic_hinge_axis")
    if ax_:
        md.append(f"| kinematic hinge axis fit RMS (mm) | {_fmt(ax_['fit_rms_mm'], 2)} |")
    md += ["", "## Quality", ""] + [f"- {x}" for x in txt]
    md += ["", "## Segments", ""] + [f"- {s['label']}: {s['t0']:.2f}-{s['t1']:.2f} s" for s in segs]
    md += ["", "## Notes", ""] + [f"- {n}" for n in P.get("notes", [])]
    md += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in files]
    (outdir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("report written to %s (%d figures)", outdir, len(files))
    return files


def _fmt(v, nd=1):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))
