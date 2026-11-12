#!/usr/bin/env python3
"""Verify the printable PDFs: rasterise (pdftoppm and/or PyMuPDF), re-detect all markers with
cv2.aruco, and measure physical sizes in the raster (marker side, board extent, scale bars,
ChArUco square). Writes verify/verification.json and prints a summary.

Usage: python3 verify_markers.py [--dpi 600]
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rasterise(pdf, dpi, engine):
    if engine == "pdftoppm":
        tmp = tempfile.mkdtemp()
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", "-singlefile", pdf, os.path.join(tmp, "p")], check=True)
        img = cv2.imread(os.path.join(tmp, "p.png"), cv2.IMREAD_GRAYSCALE)
        shutil.rmtree(tmp)
        return img
    import pymupdf
    doc = pymupdf.open(pdf)
    pix = doc[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def detect_aruco(img, dict_id):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(d, p).detectMarkers(img)
    return corners, (ids.ravel().tolist() if ids is not None else [])


def tick_centres(img, y_px, x_lo, x_hi):
    """Centres (px, subpixel via intensity weighting) of dark runs along row y_px."""
    row = 255.0 - img[y_px, x_lo:x_hi].astype(float)
    dark = row > 127
    cents, x = [], 0
    while x < len(row):
        if dark[x]:
            x0 = x
            while x < len(row) and dark[x]:
                x += 1
            a, b = max(x0 - 2, 0), min(x + 2, len(row))
            w = row[a:b]
            cents.append(x_lo + (np.arange(a, b) * w).sum() / w.sum())
        else:
            x += 1
    return cents


def measure_hbar(img, ppmm, x_mm, y_mm):
    # ticks occupy y in [y+1, y+1+2.5] mm below the 1 mm bar; sample at y+2.2 mm
    y = int(round((y_mm + 2.2) * ppmm))
    c = tick_centres(img, y, int((x_mm - 3) * ppmm), int((x_mm + 103) * ppmm))
    return (c[-1] - c[0]) / ppmm, len(c)


def measure_vbar(img, ppmm, x_mm, y_mm):
    x = int(round((x_mm + 1 + 1.2) * ppmm))
    c = tick_centres(np.ascontiguousarray(img.T), x, int((y_mm - 3) * ppmm), int((y_mm + 103) * ppmm))
    return (c[-1] - c[0]) / ppmm, len(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    lay = json.load(open(os.path.join(HERE, "verify", "layout.json")))
    geo = json.load(open(os.path.join(HERE, "board_geometry.json")))
    ppmm = a.dpi / 25.4
    engines = ["pdftoppm"] if shutil.which("pdftoppm") else []
    try:
        import pymupdf  # noqa
        engines.append("pymupdf")
    except ImportError:
        pass
    report = {"dpi": a.dpi}
    for eng in engines:
        r = {}
        # ---------- HEAD/JAW sheet ----------
        img = rasterise(os.path.join(HERE, "markers_head_jaw.pdf"), a.dpi, eng)
        corners, ids = detect_aruco(img, cv2.aruco.DICT_4X4_50)
        sides = [np.linalg.norm(np.roll(c.reshape(4, 2), -1, 0) - c.reshape(4, 2), axis=1) for c in corners]
        sides_mm = np.concatenate(sides) / ppmm
        # board-level check: distance between corner 0 of first and last marker vs objPoints
        board_err = []
        cmap = {}
        for c, i in zip(corners, ids):
            cmap.setdefault(i, []).append(c.reshape(4, 2))
        for bname in ("HEAD", "JAW"):
            g = geo["boards"][bname]
            i0, i1 = g["ids"][0], g["ids"][-1]
            d_true = np.linalg.norm(np.array(g["marker_corners_mm"][str(i1)][2]) - np.array(g["marker_corners_mm"][str(i0)][0]))
            for c0, c1 in zip(sorted(cmap.get(i0, []), key=lambda c: c[0, 0]), sorted(cmap.get(i1, []), key=lambda c: c[0, 0])):
                board_err.append(np.linalg.norm(c1[2] - c0[0]) / ppmm - d_true)
        sb_x, sb_y = lay["marker_sheet"]["scale_bar_x_mm"], lay["marker_sheet"]["scale_bar_y_mm"]
        hbar, nh = measure_hbar(img, ppmm, sb_x, sb_y)
        vbar, nv = measure_vbar(img, ppmm, 180.0, sb_y - 5)
        expected = sorted([0, 1, 2, 3, 4, 5] * 3 + [10, 11, 12, 13] * 3)
        r["markers_sheet"] = {
            "n_detected": len(ids), "ids_ok": sorted(ids) == expected, "ids": sorted(ids),
            "marker_side_mm_mean": round(float(sides_mm.mean()), 4), "marker_side_mm_min": round(float(sides_mm.min()), 4),
            "marker_side_mm_max": round(float(sides_mm.max()), 4),
            "board_diagonal_error_mm_max_abs": round(float(np.max(np.abs(board_err))), 4), "n_board_diagonals": len(board_err),
            "scale_bar_h_mm": round(hbar, 3), "scale_bar_h_ticks": nh, "scale_bar_v_mm": round(vbar, 3), "scale_bar_v_ticks": nv,
        }
        # ---------- ChArUco sheet ----------
        img = rasterise(os.path.join(HERE, "charuco_calibration_A4.pdf"), a.dpi, eng)
        g = geo["boards"]["CHARUCO"]
        board = cv2.aruco.CharucoBoard(tuple(g["squares_x_y"]), g["square_mm"], g["marker_mm"],
                                       cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
        ch_corners, ch_ids, m_corners, m_ids = cv2.aruco.CharucoDetector(board).detectBoard(img)
        cc = ch_corners.reshape(-1, 2)
        cid = ch_ids.ravel()
        obj = np.array(g["chessboard_corners_mm"])[cid][:, :2]
        # similarity fit image->board (should be pure scale ppmm, no anisotropy)
        A = np.c_[cc, np.ones(len(cc))]
        sol, *_ = np.linalg.lstsq(A, obj, rcond=None)
        resid = A @ sol - obj
        sx_ = np.linalg.norm(sol[0]); sy_ = np.linalg.norm(sol[1])
        nx = g["squares_x_y"][0] - 1
        sq = []
        for k, i in enumerate(cid):
            for j in (i + 1, i + nx):
                if (j == i + 1 and (i + 1) % nx == 0):
                    continue
                w = np.where(cid == j)[0]
                if len(w):
                    sq.append(np.linalg.norm(cc[w[0]] - cc[k]) / ppmm)
        ch = lay["charuco_sheet"]
        hbar, nh = measure_hbar(img, ppmm, ch["board"]["x_mm"], ch["scale_bar_y_mm"])
        vbar, nv = measure_vbar(img, ppmm, ch["board"]["x_mm"] + ch["board"]["w_mm"] - 10, ch["scale_bar_y_mm"])
        r["charuco_sheet"] = {
            "n_charuco_corners": int(len(cid)), "expected": 24, "n_markers": int(len(m_ids)), "expected_markers": 17,
            "square_mm_mean": round(float(np.mean(sq)), 4), "square_mm_min": round(float(np.min(sq)), 4), "square_mm_max": round(float(np.max(sq)), 4),
            "affine_fit_mm_per_mm_x": round(float(sx_ * ppmm), 5), "affine_fit_mm_per_mm_y": round(float(sy_ * ppmm), 5),
            "affine_fit_rms_mm": round(float(np.sqrt((resid ** 2).sum(1).mean())), 4),
            "scale_bar_h_mm": round(hbar, 3), "scale_bar_v_mm": round(vbar, 3),
        }
        report[eng] = r
    json.dump(report, open(os.path.join(HERE, "verify", "verification.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    ok = all(v["markers_sheet"]["ids_ok"] and abs(v["markers_sheet"]["marker_side_mm_mean"] - 15) < 0.1
             and abs(v["markers_sheet"]["scale_bar_h_mm"] - 100) < 0.1 and v["charuco_sheet"]["n_charuco_corners"] == 24
             for k, v in report.items() if k != "dpi")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
