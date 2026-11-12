"""Stage 2 (solve): raw_tracking.json + STLs + registration landmarks -> motion.json."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import se3
from .export import articulator_text, edge_to_edge_frame, write_positions
from .io import load_mesh, read_json, write_json
from .landmarks import auto_lower_landmarks, bonwill_condyles, fit_hinge_axis, kinematic_condyles, occlusal_frame
from .motion import (detect_mi_interval, fill_and_smooth, incisal_displacement, load_series, mi_holds,
                     reject_outliers, t_s_series)
from .params import compute as compute_params
from .registration import lever_uncertainty, register
from .segments import mask_for, segment
from .track import find_first_mi_hold

log = logging.getLogger("jawtrack.solve")


def _local_average(raw, k, key, half_s=0.2, max_mm=0.8, max_deg=0.5):
    """Robust average of a board pose over neighbouring frames that differ from frame k by less than
    the pose noise level (reduces single-frame pose noise in the registration frame)."""
    fr = raw["frames"]
    T0 = se3.from_list(fr[k][key]["T"])
    Ts = []
    for f in fr:
        if f.get(key) and abs(f["t"] - fr[k]["t"]) <= half_s:
            T = se3.from_list(f[key]["T"])
            if np.linalg.norm(T[:3, 3] - T0[:3, 3]) < max_mm and se3.rot_dist_deg(T[:3, :3], T0[:3, :3]) < max_deg:
                Ts.append(T)
    if len(Ts) < 3:
        return T0, 1
    return se3.robust_average(Ts)[0], len(Ts)


def _reg_poses(raw, lm, average=True):
    """Head/jaw pose and intrinsics for the registration image referenced by the landmarks."""
    rf = raw.get("registration_frame")
    fi = (lm.get("image") or {}).get("frame_index")
    K, dist = np.asarray(raw["camera"]["K"]), np.asarray(raw["camera"]["dist"])
    if rf and rf.get("source") != "video":
        K, dist = np.asarray(rf["K"]), np.asarray(rf["dist"])
        if not rf.get("head"):
            raise RuntimeError("registration frame has no HEAD pose")
        return (se3.from_list(rf["head"]["T"]), se3.from_list(rf["jaw"]["T"]) if rf.get("jaw") else None, K, dist,
                f"{rf.get('source')} frame {rf.get('frame_index')}")
    if fi is None and rf:
        fi = rf.get("frame_index")
    for k, f in enumerate(raw["frames"]):
        if f["i"] == fi:
            if not f.get("head"):
                raise RuntimeError(f"landmark frame {fi} has no HEAD pose in raw_tracking.json")
            if average:
                Th, nh = _local_average(raw, k, "head")
                Tj, nj = _local_average(raw, k, "jaw") if f.get("jaw") else (None, 0)
                log.info("registration frame %s: head pose averaged over %d frames, jaw over %d", fi, nh, nj)
            else:
                Th = se3.from_list(f["head"]["T"])
                Tj = se3.from_list(f["jaw"]["T"]) if f.get("jaw") else None
            return Th, Tj, K, dist, f"video frame {fi}"
    raise RuntimeError(f"landmarks refer to frame_index {fi}, which is not in raw_tracking.json "
                       "(was the video tracked with --every N? pick a tracked frame)")


def _reg_image_path(raw_path, landmarks_path, lm, raw):
    from pathlib import Path as P
    cands = []
    f = (lm.get("image") or {}).get("file")
    if f:
        cands += [P(landmarks_path).parent / f, P(raw_path).parent / f, P(f)]
    rf = raw.get("registration_frame") or {}
    if rf.get("file"):
        cands.append(P(raw_path).parent / rf["file"])
    for c in cands:
        if c.exists():
            return c
    return None


def solve(raw_path, upper, lower, landmarks, out, out_positions=None, mi_interval=None, segments_path=None,
          condyles="average", smooth_s=0.12, max_gap_s=0.25, refine_registration="auto", print_scale=None):
    raw = read_json(raw_path)
    lm = read_json(landmarks)
    up_mesh, lo_mesh = load_mesh(upper), load_mesh(lower)
    notes = []
    ser = load_series(raw)
    times, valid0 = ser.times, ser.valid.copy()
    fps = float(raw.get("fps") or (1.0 / np.median(np.diff(times))))

    # --- provisional MI / M0 from board motion only
    vt = np.flatnonzero(valid0)
    if mi_interval:
        hold = tuple(mi_interval)
    else:
        hold = find_first_mi_hold(times[vt], ser.M[vt]) or (times[vt[0]], times[vt[min(len(vt) - 1, 10)]])
    sel = valid0 & (times >= hold[0]) & (times <= hold[1])
    M0, _ = se3.robust_average(ser.M[sel])

    # --- registration
    T_C_H_reg, T_C_J_reg, K, dist, reg_desc = _reg_poses(raw, lm)
    X, reg_info = register(lm, T_C_H_reg, K, dist, T_C_J_reg, M0)

    # --- landmarks in S
    if lm.get("incisal_point_mm") is not None:
        inc = np.asarray(lm["incisal_point_mm"], float)
        inc_src = "landmarks file"
    else:
        inc, _ = auto_lower_landmarks(lo_mesh.vertices, up_mesh.vertices)
        inc_src = "automatic (most anterior-superior lower vertex)"
        notes.append("incisal point estimated automatically from the lower STL (no incisal_point_mm given)")
    if lm.get("occlusal_plane_mm"):
        plane = lm["occlusal_plane_mm"]
    else:
        _, plane = auto_lower_landmarks(lo_mesh.vertices, up_mesh.vertices)
        plane = [inc] + list(plane[1:])
        notes.append("occlusal plane estimated automatically (incisal point + highest posterior cusps L/R)")
    of = occlusal_frame(inc, plane, up_mesh.vertices, lo_mesh.vertices)
    log.info("incisal point (%s): %s", inc_src, np.round(inc, 2).tolist())

    # --- MI refinement with the incisal point, M0, re-registration
    for it in range(2):
        TS = t_s_series(X, ser.M, M0)
        disp = np.linalg.norm(incisal_displacement(TS, inc), axis=1)
        if not mi_interval:
            h = detect_mi_interval(times, disp, valid0)
            if h is None:
                log.warning("no stable MI hold found at the start; using the provisional interval %.2f-%.2f s",
                            *hold)
                notes.append("MI hold not detected automatically; provisional interval used")
            else:
                hold = h
        sel = valid0 & (times >= hold[0]) & (times <= hold[1])
        M0, inl = se3.robust_average(ser.M[sel])
        X, reg_info = register(lm, T_C_H_reg, K, dist, T_C_J_reg, M0)
    log.info("MI reference interval %.2f-%.2f s (%d frames)", hold[0], hold[1], int(sel.sum()))

    # --- optional image-based refinement of X (edges of the rendered dentition vs. the reg image)
    refine_info = None
    if refine_registration != "off":
        img_path = _reg_image_path(raw_path, landmarks, lm, raw)
        if img_path is None:
            msg = "registration refinement skipped: registration image not found"
            (log.warning if refine_registration == "on" else log.info)(msg)
            refine_info = {"applied": False, "reason": "registration image not found"}
        else:
            import cv2
            from .registration import lever_uncertainty as _lev
            from .registration_refine import board_polygons, refine_registration as _refine
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            iw, ih = (lm.get("image") or {}).get("width"), (lm.get("image") or {}).get("height")
            if img is None or (iw and (img.shape[1], img.shape[0]) != (iw, ih)):
                refine_info = {"applied": False, "reason": f"cannot use {img_path} (missing or size mismatch)"}
                log.warning("registration refinement skipped: %s", refine_info["reason"])
            else:
                ps = print_scale or raw.get("print_scale", 1.0)
                polys = board_polygons(T_C_H_reg, T_C_J_reg, K, dist, ps)
                X_click = X
                X, refine_info = _refine(X_click, lm, img, T_C_H_reg, K, dist, up_mesh, lo_mesh, T_C_J_reg, M0,
                                         board_polys=polys, occlusal_frame=of, click_cov=reg_info.get("cov_rotvec_t"))
                refine_info["image"] = str(img_path)
                # uncertainty before / after at the max-opening-like pose (largest jaw rotation)
                TS_tmp = t_s_series(X_click, ser.M, M0)
                ang = np.array([se3.rot_angle_deg(T[:3, :3]) if v else 0 for T, v in zip(TS_tmp, valid0)])
                kk = int(np.argmax(ang))
                refine_info["uncertainty_before_mm"] = _lev(reg_info, X_click, ser.M[kk], M0, inc)
                if refine_info.get("applied"):
                    Xr = X

                    def _xo(x, X_click=X_click, Xr=Xr):  # perturbation about the refined X, click parametrisation
                        return se3.make_T(Rotation.from_rotvec(x[:3]).as_matrix() @ Xr[:3, :3], Xr[:3, 3] + x[3:])
                    refine_info["uncertainty_after_mm"] = _lev({}, X, ser.M[kk], M0, inc,
                                                               cov=refine_info["cov_after"], x_of=_xo)
                    refine_info["X_click"] = se3.to_list(X_click)
                    log.info("registration incisal uncertainty at max rotation: %.2f mm (clicks) -> %.2f mm "
                             "(refined; edge points treated as ~200 independent samples)",
                             refine_info["uncertainty_before_mm"], refine_info["uncertainty_after_mm"])
                refine_info.pop("cov_after", None)

    TS = t_s_series(X, ser.M, M0)
    disp_raw_vec = incisal_displacement(TS, inc)
    # MI must be the most closed position: vertical displacement above MI indicates a wrong hold
    up_comp = of.to_local(disp_raw_vec)[:, 1]
    if valid0.any() and np.percentile(up_comp[valid0], 99) > 1.0:
        msg = (f"incisal point rises {np.percentile(up_comp[valid0], 99):.1f} mm above the MI reference in some "
               "frames: the detected MI interval may not be true MI (use --mi-interval)")
        log.warning(msg)
        notes.append(msg)

    valid = reject_outliers(times, TS, valid0, inc)
    TSf, status = fill_and_smooth(times, TS, valid, inc, max_gap_s=max_gap_s, smooth_s=smooth_s)
    ok = status < 2
    di = of.to_local(incisal_displacement(TSf, inc))
    mag = np.linalg.norm(di, axis=1)

    # --- noise floor in MI holds (before and after filtering)
    d_raw = np.linalg.norm(disp_raw_vec, axis=1)
    noise = d_raw[sel]
    mi_thr = max(0.5, float(np.percentile(noise, 95)) * 2.0) if noise.size else 0.5
    holds = mi_holds(times, mag, ok, mi_thr)

    def jitter(T_arr, mask):
        q = incisal_displacement(T_arr, inc)[mask]
        return float(np.sqrt(np.mean(np.sum((q - q.mean(axis=0)) ** 2, axis=1)))) if len(q) > 2 else None

    # noise floor: long holds only (>= 1 s, trimmed 0.2 s each side) so slow movement onsets do not count
    long_holds = [(a + 0.2, b - 0.2) for a, b in holds if b - a >= 1.0]
    hold_masks = [(times >= a) & (times <= b) for a, b in (long_holds or holds)] or [sel]
    jit_f = [jitter(TSf, m & ok) for m in hold_masks if (m & ok).sum() > 2]
    jit_r = [jitter(TS, m & valid) for m in hold_masks if (m & valid).sum() > 2]
    w = [int((m & ok).sum()) for m in hold_masks if (m & ok).sum() > 2]
    jitter_f = float(np.sqrt(np.average(np.square(jit_f), weights=w))) if jit_f else None
    jitter_r = float(np.sqrt(np.average(np.square(jit_r), weights=w[:len(jit_r)]))) if jit_r else None
    mi_rep = None
    if len(holds) >= 2:
        a, b = holds[0], holds[-1]
        qa = incisal_displacement(TSf, inc)[(times >= a[0]) & (times <= a[1]) & ok].mean(axis=0)
        qb = incisal_displacement(TSf, inc)[(times >= b[0]) & (times <= b[1]) & ok].mean(axis=0)
        mi_rep = float(np.linalg.norm(qb - qa))

    # --- segments
    if segments_path:
        segs = read_json(segments_path)
        segs = segs["segments"] if isinstance(segs, dict) else segs
        log.info("segments from %s", segments_path)
    else:
        segs = segment(times, di, ok, mi_thr_mm=mi_thr)

    # --- condyles
    cL, cR = bonwill_condyles(of)
    cond_method = "average_bonwill"
    oc = mask_for(times, segs, "open_close") & ok
    axis = None
    if oc.any():
        # opening phases with incisal displacement 1..10 mm (first ~10 mm of each opening)
        m = oc & (mag >= 1.0) & (mag <= 10.0) & (-di[:, 1] > 0.5 * mag)
        axis = fit_hinge_axis(TSf[m], inc, of) if m.sum() >= 5 else None
    if axis:
        log.info("kinematic hinge axis: point %s dir %s, fit RMS %.2f mm (%d frames, %.1f deg from occlusal "
                 "lateral axis)", np.round(axis["point"], 1).tolist(), np.round(axis["direction"], 3).tolist(),
                 axis["fit_rms_mm"], axis["n_frames"], axis["angle_to_left_deg"])
    if condyles == "kinematic":
        if axis:
            cL, cR = kinematic_condyles(axis)
            cond_method = "kinematic_axis"
        else:
            notes.append("kinematic condyles requested but the hinge axis could not be fitted -> Bonwill used")
    P, key, pnotes = compute_params(times, TSf, ok, segs, of, inc, cL, cR)
    P["kinematic_hinge_axis"] = ({k: axis[k] for k in ("point", "direction", "fit_rms_mm")} if axis else None)
    if axis:
        P["kinematic_hinge_axis"].update(n_frames=axis["n_frames"], angle_to_occlusal_lateral_deg=axis["angle_to_left_deg"])
    prot = mask_for(times, segs, "protrusion") & ok
    e2e, e2e_d = edge_to_edge_frame(TSf, prot, inc, up_mesh.vertices, of)
    if e2e is not None:
        key["edge_to_edge"] = e2e
    notes += pnotes
    notes.append(f"condyles: {cond_method}" + (" (Bonwill 105 mm triangle, Balkwill 22 deg, built in the "
                                               "occlusal-plane frame)" if cond_method == "average_bonwill" else ""))
    notes.append("max_opening_mm is the 3-D incisal-point displacement from MI; add the vertical overbite for the "
                 "clinical interincisal distance.")
    P["notes"] = notes

    tracked_fraction = float(valid.mean())
    q = np.where(status == 0, ser.quality, np.where(status == 1, 0.5 * ser.quality.max(initial=0), 0.0))
    if np.any(status == 1):
        # interpolated frames: quality from neighbours
        q[status == 1] = 0.5 * np.interp(times[status == 1], times[status == 0], ser.quality[status == 0])
    motion = {
        "version": 1, "units": "mm", "fps": fps,
        "frame_of_reference": "upper STL coordinates; T maps lower STL (as exported in MI) to its pose at time t",
        "source": {"video": raw.get("video"), "video_offset_s": 0.0, "raw_tracking": str(raw_path),
                   "upper_stl": str(upper), "lower_stl": str(lower)},
        "frames": [{"t": round(float(t), 5), "T": [round(v, 6) for v in se3.to_list(T)], "valid": bool(s < 2),
                    "quality": round(float(qq), 3), **({"interpolated": True} if s == 1 else {})}
                   for t, T, s, qq in zip(times, TSf, status, q)],
        "segments": segs,
        "landmarks_mm": {"incisal_point": inc.tolist(), "condyle_left": np.asarray(cL).tolist(),
                         "condyle_right": np.asarray(cR).tolist(),
                         "occlusal_plane": {"point": of.origin.tolist(), "normal": of.e_up.tolist()},
                         "condyle_method": cond_method,
                         "occlusal_frame": of.as_dict()},
        "parameters": P,
        "quality": {"tracked_fraction": tracked_fraction, "registration_rms_px": reg_info["rms_px"],
                    "jitter_mm_rms_in_MI": jitter_f, "jitter_mm_rms_in_MI_unfiltered": jitter_r,
                    "mi_repeatability_mm": mi_rep, "mi_interval_s": list(hold), "mi_holds_s": holds, "jitter_holds_s": long_holds or holds,
                    "registration": reg_info, "registration_image": reg_desc,
                    "marker_reproj_px": {
                        "head_median": float(np.nanmedian(ser.head_reproj)),
                        "head_p95": float(np.nanpercentile(ser.head_reproj, 95)),
                        "jaw_median": float(np.nanmedian(ser.jaw_reproj)),
                        "jaw_p95": float(np.nanpercentile(ser.jaw_reproj, 95))},
                    "frames_interpolated": int((status == 1).sum()), "frames_invalid": int((status == 2).sum()),
                    "X_T_H_S": se3.to_list(X), "M0": se3.to_list(M0)},
        "key_frames": {k: (None if v is None else {"index": int(v), "t": float(times[v])}) for k, v in key.items()},
    }
    if e2e_d is not None:
        motion["quality"]["edge_to_edge_distance_mm"] = e2e_d
    # registration lever: predicted incisal error from registration uncertainty at max opening
    kmo = key.get("max_opening")
    if kmo is not None and valid0[kmo]:
        lu = (refine_info.get("uncertainty_after_mm") if refine_info and refine_info.get("applied")
              else lever_uncertainty(reg_info, X, ser.M[kmo], M0, inc))
        motion["quality"]["registration_incisal_uncertainty_at_max_opening_mm"] = lu
        if lu is not None:
            log.info("registration uncertainty (from %.2f px residual) -> +-%.2f mm at the incisal point at max opening",
                     reg_info["sigma_px_used"], lu)
            if lu > 1.0:
                notes.append(f"registration uncertainty propagates to ~{lu:.1f} mm incisal error at max opening: "
                             "click more (8-10) well-spread landmarks incl. molars / lower teeth")
    motion["registration"] = {"X": se3.to_list(X), "convention": "X = T_H_S maps STL -> HEAD board coordinates",
                              "rms_px": reg_info["rms_px"], "refinement": refine_info}
    write_json(out, motion)
    log.info("tracked %.1f%%, registration %.2f px, jitter in MI %.3f mm (unfiltered %.3f mm)",
             100 * tracked_fraction, reg_info["rms_px"], jitter_f or float("nan"), jitter_r or float("nan"))
    if out_positions:
        files = write_positions(out_positions, lo_mesh, TSf, times, key)
        Path(out_positions, "articulator_values.txt").write_text(articulator_text(motion, files), encoding="utf-8")
        write_json(Path(out_positions) / "positions.json", files, indent=1)
    return motion
