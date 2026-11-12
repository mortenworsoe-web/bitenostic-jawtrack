"""Virtual-articulator parameters (SPEC 5.2). All angles relative to the OCCLUSAL PLANE.

Definitions (displacements from MI, in the occlusal-plane frame):
* max_opening_mm: max 3-D incisal-point displacement in open_close (clinical interincisal opening
  additionally includes the vertical overbite).
* max_protrusion_mm: max anterior incisal displacement in protrusion.
* max_laterotrusion_{right,left}_mm: max lateral incisal displacement towards that side.
* sagittal_condylar_inclination_{L,R}_deg: direction of the condyle-point path in the sagittal
  projection (anterior vs. downward), mean direction over the part of the protrusion path with
  0.5..5 mm anterior condylar displacement; + = descending forward.
* bennett_angle_{L,R}_deg: angle in the horizontal (occlusal-plane) projection between the
  ORBITING condyle path and the sagittal direction, from the contralateral laterotrusion
  (left value from right laterotrusion), over 0.5..5 mm of horizontal condylar displacement.
* incisal_guidance_deg: sagittal direction of the incisal-point protrusion path over the first
  0.3..3.5 mm of anterior displacement; + = descending forward.
"""
from __future__ import annotations

import numpy as np

from . import se3
from .landmarks import OcclusalFrame
from .segments import mask_for


def point_disp_local(TS, p, of: OcclusalFrame):
    d = np.einsum("nij,j->ni", TS[:, :3, :3], p) + TS[:, :3, 3] - p
    return of.to_local(d)  # (left, up, ant)


def _mean_dir_angle(x, y, xr):
    """Angle (deg) of the mean displacement direction for samples with xr[0] <= x <= xr[1]."""
    m = (x >= xr[0]) & (x <= xr[1])
    if m.sum() < 3:
        return None, int(m.sum())
    # direction of the least-squares line through the origin weighted by distance
    return float(np.degrees(np.arctan2(np.sum(y[m] * x[m]), np.sum(x[m] * x[m])))), int(m.sum())


def compute(times, TS, ok, segs, of: OcclusalFrame, incisal, condyle_left, condyle_right):
    """Return (parameters dict, key_frames dict, notes list)."""
    notes = ["Angles are measured relative to the occlusal plane (Frankfort horizontal not available); "
             "sagittal condylar inclination to the occlusal plane is typically ~5-10 deg larger than to FH."]
    P, key = {}, {"MI": None}
    di = point_disp_local(TS, incisal, of)
    dL = point_disp_local(TS, condyle_left, of)
    dR = point_disp_local(TS, condyle_right, of)
    mag = np.linalg.norm(di, axis=1)

    def seg_mask(label):
        return mask_for(times, segs, label) & ok

    m = seg_mask("open_close")
    if m.any():
        k = int(np.flatnonzero(m)[np.argmax(mag[m])])
        P["max_opening_mm"] = float(mag[k]); key["max_opening"] = k
        P["max_opening_vertical_mm"] = float(-di[k, 1])
    else:
        P["max_opening_mm"] = None
        notes.append("no open_close segment detected")
    m = seg_mask("protrusion")
    if m.any():
        k = int(np.flatnonzero(m)[np.argmax(di[m, 2])])
        P["max_protrusion_mm"] = float(di[k, 2]); key["max_protrusion"] = k
        P["sagittal_condylar_inclination_left_deg"], nl = _mean_dir_angle(dL[m, 2], -dL[m, 1], (0.5, 5.0))
        P["sagittal_condylar_inclination_right_deg"], nr = _mean_dir_angle(dR[m, 2], -dR[m, 1], (0.5, 5.0))
        P["incisal_guidance_deg"], ni = _mean_dir_angle(di[m, 2], -di[m, 1], (0.3, 3.5))
        if P["sagittal_condylar_inclination_left_deg"] is None:
            notes.append("condylar protrusion < 0.5 mm: sagittal condylar inclination not measurable")
    else:
        P.update(max_protrusion_mm=None, sagittal_condylar_inclination_left_deg=None,
                 sagittal_condylar_inclination_right_deg=None, incisal_guidance_deg=None)
        notes.append("no protrusion segment detected")
    m = seg_mask("laterotrusion_right")
    if m.any():
        k = int(np.flatnonzero(m)[np.argmax(-di[m, 0])])
        P["max_laterotrusion_right_mm"] = float(-di[k, 0]); key["max_laterotrusion_right"] = k
        # orbiting = LEFT condyle; medial for the left condyle = -left
        P["bennett_angle_left_deg"], _ = _bennett(dL[m], medial_sign=-1)
    else:
        P.update(max_laterotrusion_right_mm=None, bennett_angle_left_deg=None)
        notes.append("no right laterotrusion segment detected")
    m = seg_mask("laterotrusion_left")
    if m.any():
        k = int(np.flatnonzero(m)[np.argmax(di[m, 0])])
        P["max_laterotrusion_left_mm"] = float(di[k, 0]); key["max_laterotrusion_left"] = k
        P["bennett_angle_right_deg"], _ = _bennett(dR[m], medial_sign=+1)
    else:
        P.update(max_laterotrusion_left_mm=None, bennett_angle_right_deg=None)
        notes.append("no left laterotrusion segment detected")
    notes.append("Bennett angle LEFT is measured on the left (orbiting) condyle during RIGHT laterotrusion and "
                 "vice versa; condyle points are model points, so condyle-derived angles inherit "
                 "registration and condyle-position errors.")
    return P, key, notes


def _bennett(dc, medial_sign):
    ant = dc[:, 2]
    med = medial_sign * dc[:, 0]
    h = np.hypot(ant, med)
    m = (h >= 0.5) & (h <= 5.0)
    if m.sum() < 3:
        return None, int(m.sum())
    return float(np.degrees(np.arctan2(np.sum(med[m] * h[m]), np.sum(ant[m] * h[m])))), int(m.sum())
