"""Ground-truth definitions shared by make_synth.py and validate.py.

All parameter definitions are documented in testdata/README.md; keep them in sync.
Occlusal frame (from landmarks): origin = lower incisal point IP; n = unit normal of the plane
through IP, MB36, MB46 oriented superior (towards the upper arch); f = anterior = unit
projection onto the plane of (IP - midpoint(MB36, MB46)); l = n x f = patient's LEFT.
Sagittal plane = span(f, n); horizontal plane = occlusal plane.
"""
import numpy as np

PROTRUSIVE_RECORD_MM = 5.0   # incisal-point anterior displacement at which inclinations are read
LATERAL_RECORD_MM = 5.0      # incisal-point lateral displacement at which Bennett angles are read
BONWILL_SIDE_MM = 105.0
BALKWILL_DEG = 22.0


def occlusal_frame(ip, mb36, mb46, up_hint=None):
    ip, mb36, mb46 = (np.asarray(v, float) for v in (ip, mb36, mb46))
    n = np.cross(mb36 - ip, mb46 - ip)
    n /= np.linalg.norm(n)
    if up_hint is not None and n @ np.asarray(up_hint) < 0:
        n = -n
    mid = 0.5 * (mb36 + mb46)
    f = ip - mid
    f -= n * (f @ n)
    f /= np.linalg.norm(f)
    l = np.cross(n, f)
    # make sure l points towards 36 (patient's left); if not, the normal was flipped
    if l @ (mb36 - mb46) < 0:
        n = -n
        l = np.cross(n, f)
    return dict(origin=ip, n=n, f=f, l=l)


def bonwill_condyles(fr):
    """Equilateral Bonwill triangle (side 105 mm) tilted by the Balkwill angle (22 deg) above the
    occlusal plane, apex at the lower incisal point."""
    h = BONWILL_SIDE_MM * np.sqrt(3) / 2
    b = np.radians(BALKWILL_DEG)
    c = fr["origin"] - h * np.cos(b) * fr["f"] + h * np.sin(b) * fr["n"]
    half = BONWILL_SIDE_MM / 2
    return c + half * fr["l"], c - half * fr["l"]  # left, right


def apply(T, p):
    T = np.asarray(T)
    return np.einsum("nij,j->ni", T[:, :3, :3], p) + T[:, :3, 3]


def seg_mask(times, segments, label):
    m = np.zeros(len(times), bool)
    for s in segments:
        if s["label"] == label:
            m |= (times >= s["t0"] - 1e-9) & (times <= s["t1"] + 1e-9)
    return m


def _crossings(x, level, mask):
    """Indices i (with fractional part) where x crosses `level` upwards inside mask runs."""
    out = []
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for r in runs:
        xr = x[r]
        for k in range(len(r) - 1):
            if xr[k] < level <= xr[k + 1]:
                a = (level - xr[k]) / (xr[k + 1] - xr[k])
                out.append(r[k] + a)
    return out


def _interp(arr, fi):
    i0 = int(np.floor(fi))
    a = fi - i0
    i1 = min(i0 + 1, len(arr) - 1)
    return (1 - a) * arr[i0] + a * arr[i1]


def compute_parameters(Ts, times, segments, ip, mb36, mb46, condyles=None, valid=None):
    """Parameters per SPEC 5.2 from a motion (Ts: (N,4,4) row-major matrices as arrays)."""
    Ts = np.asarray(Ts, float)
    times = np.asarray(times, float)
    if valid is None:
        valid = np.ones(len(Ts), bool)
    fr = occlusal_frame(ip, mb36, mb46, up_hint=None)
    cL, cR = condyles if condyles is not None else bonwill_condyles(fr)
    n, f, l = fr["n"], fr["f"], fr["l"]
    dI = apply(Ts, np.asarray(ip, float)) - ip
    dL = apply(Ts, cL) - cL
    dR = apply(Ts, cR) - cR
    fwd = dI @ f
    lat = dI @ l  # + = to patient's left
    notes = []
    P = {}

    def segmax(label, val):
        m = seg_mask(times, segments, label) & valid
        return float(np.max(val[m])) if m.any() else None

    P["max_opening_mm"] = segmax("open_close", np.linalg.norm(dI, axis=1))
    P["max_protrusion_mm"] = segmax("protrusion", fwd)
    P["max_laterotrusion_right_mm"] = segmax("laterotrusion_right", -lat)
    P["max_laterotrusion_left_mm"] = segmax("laterotrusion_left", lat)

    def angle_at(label, x, level, vecs, fn):
        m = seg_mask(times, segments, label) & valid
        cr = _crossings(x, level, m)
        if not cr:
            return None, 0
        vals = [fn(_interp(vecs, c)) for c in cr]
        return float(np.median(vals)), len(cr)

    sag = lambda d: np.degrees(np.arctan2(-(d @ n), d @ f))
    P["sagittal_condylar_inclination_left_deg"], nl = angle_at("protrusion", fwd, PROTRUSIVE_RECORD_MM, dL, sag)
    P["sagittal_condylar_inclination_right_deg"], nr = angle_at("protrusion", fwd, PROTRUSIVE_RECORD_MM, dR, sag)
    P["incisal_guidance_deg"], _ = angle_at("protrusion", fwd, PROTRUSIVE_RECORD_MM, dI, sag)
    # Bennett: orbiting condyle; left condyle orbits during RIGHT laterotrusion (medial = -l)
    benL = lambda d: np.degrees(np.arctan2(-(d @ l), d @ f))
    benR = lambda d: np.degrees(np.arctan2(d @ l, d @ f))
    P["bennett_angle_left_deg"], _ = angle_at("laterotrusion_right", -lat, LATERAL_RECORD_MM, dL, benL)
    P["bennett_angle_right_deg"], _ = angle_at("laterotrusion_left", lat, LATERAL_RECORD_MM, dR, benR)
    for k in list(P):
        if P[k] is None:
            notes.append(f"{k}: not measurable (segment missing or record level not reached)")
    P["notes"] = notes
    return P, dict(frame=fr, condyle_left=cL, condyle_right=cR)


def gt_to_motion(gt):
    """Ground truth as a SPEC 5.2 motion.json (handy for the viewer and for validate.py self-tests)."""
    lm = gt["landmarks_mm"]
    P = dict(gt["parameters"])
    return {"version": 1, "units": "mm", "fps": gt["fps"],
            "frame_of_reference": "upper STL coordinates; T maps lower STL (as exported in MI) to its pose at time t",
            "source": {"video": gt["files"]["video"], "video_offset_s": 0.0, "note": "GROUND TRUTH (synthetic)"},
            "frames": [{"t": f["t"], "T": f["T"], "valid": True, "quality": 1.0} for f in gt["frames"]],
            "segments": gt["segments"],
            "landmarks_mm": {"incisal_point": lm["incisal_point"], "condyle_left": lm["condyle_left"],
                             "condyle_right": lm["condyle_right"], "occlusal_plane": lm["occlusal_plane"],
                             "condyle_method": "average_bonwill"},
            "parameters": P,
            "quality": {"tracked_fraction": 1.0, "registration_rms_px": 0.0, "jitter_mm_rms_in_MI": 0.0}}
