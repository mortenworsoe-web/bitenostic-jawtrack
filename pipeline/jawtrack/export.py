"""Exports: key-position STLs and articulator_values.txt."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import se3
from .io import save_mesh

log = logging.getLogger("jawtrack.export")


def edge_to_edge_frame(TS, prot_mask, incisal, upper_vertices, of, max_dist_mm=2.0):
    """Frame in protrusion where the lower incisal point is closest to the upper incisal edge.
    Upper incisal edge = lowest (along the occlusal normal) upper vertex within +-4 mm of the
    midline and in the most anterior 10 mm. Returns (frame index, distance mm) or (None, None)."""
    if not prot_mask.any():
        return None, None
    U = np.asarray(upper_vertices, float)
    loc = of.to_local(U - of.origin)
    m = (np.abs(loc[:, 0]) < 4.0) & (loc[:, 2] > loc[:, 2].max() - 10.0)
    if m.sum() < 3:
        return None, None
    cand = U[m]
    edge = cand[np.argmin(loc[m, 1])]
    idx = np.flatnonzero(prot_mask)
    q = np.einsum("nij,j->ni", TS[idx, :3, :3], incisal) + TS[idx, :3, 3]
    d = np.linalg.norm(q - edge, axis=1)
    k = int(np.argmin(d))
    if d[k] > max_dist_mm:
        return None, float(d[k])
    return int(idx[k]), float(d[k])


def write_positions(outdir, lower_mesh, TS, times, key_frames: dict):
    """Write lower STL transformed to each key frame -> {name: file}."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, k in key_frames.items():
        T = np.eye(4) if k is None else TS[k]
        m = lower_mesh.copy()
        m.apply_transform(T)
        fn = outdir / f"lower_{name}.stl"
        save_mesh(m, fn)
        files[name] = {"file": fn.name, "t": None if k is None else float(times[k]), "T": se3.to_list(T)}
    log.info("wrote %d position STLs to %s", len(files), outdir)
    return files


LABELS = [
    ("max_opening_mm", "Maksimal gabning / Max opening (incisal point, 3-D)", "mm"),
    ("max_opening_vertical_mm", "Maksimal gabning, vertikal / Max opening, vertical component", "mm"),
    ("max_protrusion_mm", "Maksimal protrusion / Max protrusion", "mm"),
    ("max_laterotrusion_right_mm", "Maksimal laterotrusion højre / Max laterotrusion right", "mm"),
    ("max_laterotrusion_left_mm", "Maksimal laterotrusion venstre / Max laterotrusion left", "mm"),
    ("sagittal_condylar_inclination_left_deg", "Sagittal kondylbanehældning venstre / Sagittal condylar inclination left", "deg"),
    ("sagittal_condylar_inclination_right_deg", "Sagittal kondylbanehældning højre / Sagittal condylar inclination right", "deg"),
    ("bennett_angle_left_deg", "Bennett-vinkel venstre / Bennett angle left", "deg"),
    ("bennett_angle_right_deg", "Bennett-vinkel højre / Bennett angle right", "deg"),
    ("incisal_guidance_deg", "Incisalføring (sagittal) / Incisal guidance (sagittal)", "deg"),
]


def articulator_text(motion: dict, files: dict | None = None) -> str:
    P = motion["parameters"]
    Q = motion["quality"]
    lm = motion["landmarks_mm"]
    out = ["JawTrack - artikulatorværdier / articulator values",
           "Vinkler er målt i forhold til okklusionsplanet (ikke Frankfort) / "
           "Angles relative to the OCCLUSAL PLANE (not Frankfort horizontal)",
           f"Kondylmetode / condyle method: {lm.get('condyle_method')}", ""]
    for key, label, unit in LABELS:
        v = P.get(key)
        out.append(f"{label:<95s} {('%.1f' % v) if isinstance(v, (int, float)) else 'n/a':>7s} {unit}")
    ax = P.get("kinematic_hinge_axis")
    if ax:
        out.append(f"{'Kinematisk hængselakse, fit-RMS / Kinematic hinge axis fit RMS':<95s} {ax['fit_rms_mm']:>7.2f} mm")
    out += ["", "Kvalitet / Quality:",
            f"  tracked fraction                  {Q.get('tracked_fraction', 0):.3f}",
            f"  registration RMS                  {Q.get('registration_rms_px', float('nan')):.2f} px",
            f"  jitter in MI (incisal point, RMS) {Q.get('jitter_mm_rms_in_MI', float('nan')):.3f} mm"]
    if Q.get("mi_repeatability_mm") is not None:
        out.append(f"  MI start vs end (incisal point)   {Q['mi_repeatability_mm']:.3f} mm")
    if files:
        out += ["", "Positioner (underkæbe-STL i overkæbens koordinater) / Positions (lower STL in upper-STL coordinates):"]
        for name, f in files.items():
            t = "" if f["t"] is None else f"  t={f['t']:.2f}s"
            out.append(f"  {f['file']}{t}")
    out += ["", "Noter / Notes:"] + [f"  - {n}" for n in P.get("notes", [])]
    return "\n".join(out) + "\n"
