"""Unit tests: SE(3) math, T_S formula, frame conventions, serialisation, video rotation,
ChArUco calibration, pose disambiguation and parameter definitions.   python3 -m pytest -q tests/"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jawtrack import se3  # noqa: E402
from jawtrack.boards import HEAD, JAW, board_object_points, make_charuco  # noqa: E402

RNG = np.random.default_rng(1)


def rand_T(rot_deg=30, trans=50):
    rv = RNG.normal(size=3); rv *= np.radians(rot_deg) * RNG.uniform(0.2, 1) / np.linalg.norm(rv)
    return se3.make_T(Rotation.from_rotvec(rv).as_matrix(), RNG.uniform(-trans, trans, 3))


def test_ts_formula_recovers_ground_truth():
    # world: camera C, STL S (rigid with head), head board H, jaw board J
    for _ in range(20):
        X = rand_T(90, 100)            # T_H_S
        B_S = rand_T(90, 60)           # jaw board in S at MI
        T_C_S = rand_T(20, 300)
        T_S_gt = rand_T(25, 10)        # lower-jaw motion
        T_C_H = T_C_S @ se3.inv(X)
        T_C_J = T_C_S @ T_S_gt @ B_S
        T_C_J0 = T_C_S @ B_S
        M = se3.inv(T_C_H) @ T_C_J
        M0 = se3.inv(T_C_H) @ T_C_J0
        assert np.allclose(se3.ts_formula(X, M, M0), T_S_gt, atol=1e-9)
        assert np.allclose(se3.ts_formula(X, M0, M0), np.eye(4), atol=1e-9)


def test_inverse_and_rvec_roundtrip():
    T = rand_T()
    assert np.allclose(se3.inv(T) @ T, np.eye(4))
    rv, tv = se3.to_rvec_tvec(T)
    assert np.allclose(se3.from_rvec_tvec(rv, tv), T)


def test_row_major_serialisation():
    T = se3.make_T(np.eye(3), [1.0, 2.0, 3.0])
    v = se3.to_list(T)
    assert v[3] == 1.0 and v[7] == 2.0 and v[11] == 3.0 and v[12:] == [0, 0, 0, 1]
    assert np.allclose(se3.from_list(v), T)


def test_gridboard_convention():
    op = board_object_points(HEAD)
    assert np.allclose(op[0][0], [0, 0, 0]) and np.allclose(op[0][1], [15, 0, 0])  # x right
    assert np.allclose(op[0][3], [0, 15, 0])                                        # y down (BL corner)
    assert np.allclose(op[3][0], [0, 18, 0])                                        # second row has larger y
    assert np.allclose(op[2][1], [51, 0, 0])
    assert sorted(board_object_points(JAW)) == [10, 11, 12, 13]
    assert np.allclose(board_object_points(HEAD, 1.02)[5][2], [51 * 1.02, 33 * 1.02, 0])


def test_board_geometry_matches_capture_kit():
    p = Path(__file__).resolve().parents[2] / "capture_kit" / "board_geometry.json"
    if not p.exists():
        pytest.skip("capture_kit/board_geometry.json not present")
    g = json.loads(p.read_text())["boards"]
    for spec, key in ((HEAD, "HEAD"), (JAW, "JAW")):
        op = board_object_points(spec)
        kit = {int(k): np.array(v) for k, v in g[key]["marker_corners_mm"].items()}
        assert set(kit) == set(op)
        for i in op:
            assert np.allclose(kit[i], op[i], atol=1e-6), (key, i)


def test_robust_average_rejects_outliers():
    base = rand_T()
    Ts = [base @ se3.make_T(Rotation.from_rotvec(RNG.normal(0, 0.002, 3)).as_matrix(), RNG.normal(0, 0.05, 3))
          for _ in range(50)]
    Ts += [base @ rand_T(20, 5) for _ in range(5)]
    A, mask = se3.robust_average(Ts)
    assert se3.rot_dist_deg(A[:3, :3], base[:3, :3]) < 0.1
    assert np.linalg.norm(A[:3, 3] - base[:3, 3]) < 0.05
    assert mask[-5:].sum() == 0


def test_video_rotation_tag_is_applied():
    """A landscape frame stored rotated with a display-rotation tag must come back as displayed."""
    from jawtrack.io import VideoReader
    with tempfile.TemporaryDirectory() as d:
        img = np.zeros((90, 160), np.uint8); img[:30, :40] = 255  # bright block top-left (display)
        stored = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        raw = Path(d) / "s.mov"; out = Path(d) / "r.mov"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "90x160",
                              "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", str(raw)],
                             stdin=subprocess.PIPE)
        for _ in range(5):
            p.stdin.write(stored.tobytes())
        p.stdin.close(); p.wait()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(raw), "-c", "copy",
                        str(out)], check=True)
        for force in (False, True):
            vr = VideoReader(out, force_ffmpeg=force)
            assert (vr.width, vr.height) == (160, 90)
            _, _, fr = next(vr.frames())
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            assert g[:30, :40].mean() > 200 and g[60:, 120:].mean() < 50, f"wrong orientation (ffmpeg={force})"


def _render_plane(tex, A, K, T, size):
    Hm = K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ A
    return cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR, borderValue=128)


def test_charuco_calibration_recovers_intrinsics():
    from jawtrack.calib import calibrate_from_frames
    board = make_charuco()
    ppm = 6
    tex = board.generateImage((175 * ppm, 125 * ppm), marginSize=0)
    A = np.array([[1 / ppm, 0, 0.5 / ppm], [0, 1 / ppm, 0.5 / ppm], [0, 0, 1]])
    W, H = 1280, 720
    K = np.array([[1000.0, 0, 640.5], [0, 1000.0, 359.5], [0, 0, 1]])
    frames = []
    rng = np.random.default_rng(3)
    for k in range(25):
        R = Rotation.from_euler("xyz", rng.uniform(-30, 30, 3), degrees=True).as_matrix()
        z = rng.uniform(320, 480)
        c = np.array([87.5, 62.5, 0])
        t = np.array([rng.uniform(-60, 60), rng.uniform(-30, 30), z]) - R @ c
        img = _render_plane(tex, A, K, se3.make_T(R, t), (W, H))
        frames.append(cv2.GaussianBlur(img, (0, 0), 0.6))
    cam, info = calibrate_from_frames(frames, board, (W, H))
    Kc = np.array(cam["K"])
    print("calib rms", info["rms_px"], "fx", Kc[0, 0], "fy", Kc[1, 1], "cx", Kc[0, 2], "cy", Kc[1, 2])
    assert info["rms_px"] < 0.5
    assert abs(Kc[0, 0] / 1000 - 1) < 0.01 and abs(Kc[1, 1] / 1000 - 1) < 0.01
    assert abs(Kc[0, 2] - 640.5) < 5 and abs(Kc[1, 2] - 359.5) < 5


def test_viterbi_resolves_flip():
    from jawtrack.pose import viterbi_select
    # true orientation drifts slowly; the alternative is a 30 deg "flip"; in some frames the flip has
    # slightly LOWER reprojection error (ambiguous) - temporal consistency must still pick the truth
    seq, truth = [], []
    for k in range(40):
        Rt = Rotation.from_euler("y", 0.3 * k, degrees=True).as_matrix()
        Rf = Rotation.from_euler("y", 0.3 * k + 30, degrees=True).as_matrix()
        et, ef = (0.30, 0.29) if k % 3 == 0 else (0.30, 0.6)
        c = [(se3.make_T(Rt), et, 24), (se3.make_T(Rf), ef, 24)]
        if k % 5 == 0:
            c = c[::-1]
        seq.append(c); truth.append(0 if k % 5 else 1)
    path = viterbi_select(seq, list(range(40)))
    assert path == truth


def test_parameter_definitions_on_analytic_motion():
    from jawtrack.landmarks import OcclusalFrame
    from jawtrack.params import compute
    of = OcclusalFrame([0, 0, 0], [0, 1, 0], [0, 0, 1])  # S axes = (left, up, ant)
    cL, cR = np.array([52.5, 30, -90.0]), np.array([-52.5, 30, -90.0])
    t = np.arange(0, 4, 0.01)
    s = np.sin(np.pi * t / 2) ** 2
    sci, ig, ben = np.radians(35), np.radians(50), np.radians(12)
    Ts = []
    for k, tk in enumerate(t):
        if tk < 2:   # protrusion: pure translation along 35 deg -> SCI = 35 on both sides
            d = 6 * s[k] * np.array([0, -np.sin(sci), np.cos(sci)])
        else:        # right laterotrusion: pure translation with 12 deg medial (to the right) component
            d = 6 * s[k] * np.array([-np.sin(ben), -0.3, np.cos(ben)])
        Ts.append(se3.make_T(None, d))
    segs = [{"label": "protrusion", "t0": 0, "t1": 1.999}, {"label": "laterotrusion_right", "t0": 2.0, "t1": 4}]
    P, key, notes = compute(t, np.array(Ts), np.ones(len(t), bool), segs, of, np.zeros(3), cL, cR)
    assert abs(P["sagittal_condylar_inclination_left_deg"] - 35) < 1e-6
    assert abs(P["sagittal_condylar_inclination_right_deg"] - 35) < 1e-6
    assert abs(P["incisal_guidance_deg"] - 35) < 1e-6
    assert abs(P["bennett_angle_left_deg"] - 12) < 1e-6
    assert abs(P["max_protrusion_mm"] - 6 * np.cos(sci)) < 1e-3


def test_hinge_axis_fit_exact():
    from jawtrack.landmarks import OcclusalFrame, fit_hinge_axis
    from jawtrack.synthetic import rot_about
    of = OcclusalFrame([0, 0, 0], [0, 1, 0], [0, 0, 1])
    c = np.array([0.0, 30.0, -85.0])
    Ts = [rot_about(c, [1, 0, 0], a) for a in np.linspace(1, 8, 15)]
    ax = fit_hinge_axis(Ts, np.zeros(3), of)
    assert np.allclose(ax["point"], c, atol=1e-6) and abs(ax["direction"][0] - 1) < 1e-9
    assert ax["fit_rms_mm"] < 1e-6
