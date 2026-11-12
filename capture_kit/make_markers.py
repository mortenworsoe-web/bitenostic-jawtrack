#!/usr/bin/env python3
"""Generate the printable marker sheets + board_geometry.json for the JawTrack capture kit.

All marker bitmaps come from cv2.aruco (GridBoard / CharucoBoard.generateImage) at an integer
number of pixels per mm, and are then converted losslessly to vector rectangles in the PDF
(reportlab), so the printed geometry is bit-for-bit the OpenCV board at exact millimetre scale.

Usage:  python3 make_markers.py [--outdir DIR]
Outputs: markers_head_jaw.pdf, charuco_calibration_A4.pdf, board_geometry.json
"""
import argparse
import json
import os

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

PX_PER_MM = 28  # 711.2 dpi; makes every module an integer number of pixels (15mm/6=70px, 18mm/7=72px)

# ---- SPEC geometry (SPEC.md section 1) ----
MARKER_DICT = cv2.aruco.DICT_4X4_50
HEAD = dict(name="HEAD", grid=(3, 2), ids=[0, 1, 2, 3, 4, 5], marker_mm=15.0, gap_mm=3.0)
JAW = dict(name="JAW", grid=(2, 2), ids=[10, 11, 12, 13], marker_mm=15.0, gap_mm=3.0)
CHARUCO = dict(name="CHARUCO", dict=cv2.aruco.DICT_5X5_100, squares=(7, 5), square_mm=25.0, marker_mm=18.0)
QUIET_MM = 4.0  # white quiet zone on the cut-out sticker around the outer marker borders
PRINT_NOTE = "Udskriv i 100 % / faktisk størrelse — mål skalabjælken"


def make_gridboard(spec):
    d = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    return cv2.aruco.GridBoard(spec["grid"], spec["marker_mm"], spec["gap_mm"], d,
                               ids=np.array(spec["ids"], dtype=np.int32))


def make_charuco():
    d = cv2.aruco.getPredefinedDictionary(CHARUCO["dict"])
    return cv2.aruco.CharucoBoard(CHARUCO["squares"], CHARUCO["square_mm"], CHARUCO["marker_mm"], d)


def board_image(board, w_mm, h_mm):
    w, h = int(round(w_mm * PX_PER_MM)), int(round(h_mm * PX_PER_MM))
    img = board.generateImage((w, h), marginSize=0, borderBits=1)
    assert img.shape == (h, w)
    # must be pure binary (no interpolation) for a lossless vector conversion
    assert set(np.unique(img).tolist()) <= {0, 255}, "non-binary pixels -> resampling happened"
    return img


def image_to_rects(img):
    """Lossless conversion of a binary image to black rectangles (pixel units, y down).
    Horizontal runs per row, merged vertically over identical consecutive rows."""
    black = img < 128
    rects, prev_key, start_row = [], None, 0

    def runs(row):
        r, x, n = [], 0, len(row)
        while x < n:
            if row[x]:
                x0 = x
                while x < n and row[x]:
                    x += 1
                r.append((x0, x))
            else:
                x += 1
        return tuple(r)

    for y in range(black.shape[0] + 1):
        key = runs(black[y]) if y < black.shape[0] else None
        if key != prev_key:
            if prev_key:
                for x0, x1 in prev_key:
                    rects.append((x0, start_row, x1 - x0, y - start_row))
            prev_key, start_row = key, y
    return rects


def draw_bitmap(c, img, x_mm, y_top_mm):
    """Draw binary image with its top-left at (x_mm, y_top_mm) measured from page top-left."""
    page_h = A4[1]
    s = mm / PX_PER_MM
    c.setFillColorRGB(0, 0, 0)
    for px, py, pw, ph in image_to_rects(img):
        c.rect(x_mm * mm + px * s, page_h - y_top_mm * mm - (py + ph) * s, pw * s, ph * s, stroke=0, fill=1)


def scale_bar(c, x_mm, y_mm, vertical=False, length=100.0):
    """100.0 mm scale bar with 10 mm ticks. (x_mm, y_mm) = start point from page top-left."""
    page_h = A4[1]
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    t = 1.0  # bar thickness mm
    if not vertical:
        c.rect(x_mm * mm, page_h - (y_mm + t) * mm, length * mm, t * mm, stroke=0, fill=1)
        for i in range(0, int(length) + 1, 10):
            hgt = 4 if i % 50 == 0 else 2.5
            c.rect((x_mm + i) * mm - 0.15 * mm, page_h - (y_mm + t + hgt) * mm, 0.3 * mm, (t + hgt) * mm, stroke=0, fill=1)
        c.setFont("Helvetica", 7)
        c.drawString(x_mm * mm, page_h - (y_mm + t + 8) * mm, "0")
        c.drawRightString((x_mm + length) * mm, page_h - (y_mm + t + 8) * mm, "100.0 mm")
    else:
        c.rect(x_mm * mm, page_h - (y_mm + length) * mm, t * mm, length * mm, stroke=0, fill=1)
        for i in range(0, int(length) + 1, 10):
            hgt = 4 if i % 50 == 0 else 2.5
            c.rect((x_mm + t) * mm, page_h - (y_mm + i) * mm - 0.15 * mm, hgt * mm, 0.3 * mm, stroke=0, fill=1)
        c.saveState()
        c.setFont("Helvetica", 7)
        c.translate((x_mm + t + 6) * mm, page_h - (y_mm + length) * mm)
        c.rotate(90)
        c.drawString(0, 0, "100.0 mm (lodret)")
        c.restoreState()


def cut_frame(c, x_mm, y_mm, w_mm, h_mm, label):
    page_h = A4[1]
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.25)
    c.setDash(2, 2)
    c.rect(x_mm * mm, page_h - (y_mm + h_mm) * mm, w_mm * mm, h_mm * mm, stroke=1, fill=0)
    c.setDash()
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica", 6.5)
    c.drawString(x_mm * mm, page_h - (y_mm - 1.5) * mm, label)
    c.setFillColorRGB(0, 0, 0)


def header(c, title):
    page_h = A4[1]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(15 * mm, page_h - 15 * mm, title)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(15 * mm, page_h - 21 * mm, PRINT_NOTE)


def board_wh(spec):
    cols, rows = spec["grid"]
    return (cols * spec["marker_mm"] + (cols - 1) * spec["gap_mm"],
            rows * spec["marker_mm"] + (rows - 1) * spec["gap_mm"])


def make_marker_pdf(path):
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("JawTrack HEAD + JAW markers (DICT_4X4_50)")
    header(c, "JawTrack markører — HEAD (ID 0-5) + JAW (ID 10-13), 4X4_50, 15.0/3.0 mm")
    c.setFont("Helvetica", 8)
    c.drawString(15 * mm, A4[1] - 26 * mm,
                 "Klip langs den stiplede linje (hvid kant 4 mm er nødvendig). Lim helt plant (spraylim), helst mat laminering. Printerskalering: 'Faktisk størrelse'/'100 %'.")
    layout = []
    q = QUIET_MM
    head_img = board_image(make_gridboard(HEAD), *board_wh(HEAD))
    jaw_img = board_image(make_gridboard(JAW), *board_wh(JAW))
    hw, hh = board_wh(HEAD)
    jw, jh = board_wh(JAW)
    # Row of 3 HEAD boards, row of 3 JAW boards
    y0 = 40.0
    for k in range(3):
        x = 15.0 + k * (hw + 2 * q + 3)
        cut_frame(c, x - q, y0 - q, hw + 2 * q, hh + 2 * q, f"HEAD #{k+1}  (59 x 41 mm)  ^ op/top")
        draw_bitmap(c, head_img, x, y0)
        layout.append(dict(board="HEAD", copy=k + 1, x_mm=x, y_mm=y0, w_mm=hw, h_mm=hh))
    y1 = y0 + hh + 2 * q + 12
    for k in range(3):
        x = 20.0 + k * (jw + 2 * q + 12)
        cut_frame(c, x - q, y1 - q, jw + 2 * q, jh + 2 * q, f"JAW #{k+1}  (41 x 41 mm)  ^ op/top")
        draw_bitmap(c, jaw_img, x, y1)
        layout.append(dict(board="JAW", copy=k + 1, x_mm=x, y_mm=y1, w_mm=jw, h_mm=jh))
    # scale bars
    sb_y = y1 + jh + 2 * q + 15
    scale_bar(c, 20.0, sb_y)
    scale_bar(c, 180.0, sb_y - 5, vertical=True)
    c.setFont("Helvetica", 8)
    c.drawString(20 * mm, A4[1] - (sb_y + 16) * mm,
                 "Kontrol: vandret og lodret bjælke skal måle 100.0 mm (± 0.5). Én markør (sort kvadrat) skal måle 15.0 mm.")
    c.drawString(20 * mm, A4[1] - (sb_y + 21) * mm,
                 "Afviger målet: angiv --print-scale = målt_mm/100 til pipeline (fx 99.2 mm -> 0.992).")
    c.showPage()
    c.save()
    return layout, sb_y


def make_charuco_pdf(path):
    board = make_charuco()
    sx, sy = CHARUCO["squares"]
    w, h = sx * CHARUCO["square_mm"], sy * CHARUCO["square_mm"]
    img = board_image(board, w, h)
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("JawTrack ChArUco calibration 7x5, 25/18 mm, DICT_5X5_100")
    header(c, "JawTrack kalibrering — ChArUco 7x5, felt 25.0 / markør 18.0 mm, 5X5_100")
    x0, y0 = (210 - w) / 2, 40.0
    draw_bitmap(c, img, x0, y0)
    sb_y = y0 + h + 15
    scale_bar(c, x0, sb_y)
    scale_bar(c, x0 + w - 10, sb_y, vertical=True)
    c.setFont("Helvetica", 8)
    c.drawString(x0 * mm, A4[1] - (sb_y + 16) * mm,
                 "Lim arket helt plant på en stiv plade (karton/forex/glas). Ét felt skal måle 25.0 mm, bjælken 100.0 mm.")
    c.showPage()
    c.save()
    return dict(x_mm=x0, y_mm=y0, w_mm=w, h_mm=h), sb_y


def geometry_json(path):
    out = {
        "version": 1,
        "units": "mm",
        "generator": f"cv2 {cv2.__version__} cv2.aruco.GridBoard / CharucoBoard",
        "frame_convention_observed": (
            "OpenCV 5 GridBoard objPoints: origin = top-left (outer) corner of the top-left marker "
            "(as printed, upright), x to the right, y DOWN the sheet, z = 0 on the board plane. "
            "Right-handed => +z = x cross y points INTO the board (away from the viewer). "
            "Corner order per marker: TL, TR, BR, BL (clockwise as viewed)."),
        "sticker_quiet_zone_mm": QUIET_MM,
        "boards": {},
    }
    for spec in (HEAD, JAW):
        b = make_gridboard(spec)
        op = b.getObjPoints()
        out["boards"][spec["name"]] = {
            "type": "GridBoard", "dictionary": "DICT_4X4_50",
            "grid_cols_rows": list(spec["grid"]), "ids": [int(i) for i in b.getIds().ravel()],
            "marker_side_mm": float(b.getMarkerLength()), "gap_mm": float(b.getMarkerSeparation()),
            "board_size_mm": list(board_wh(spec)),
            "sticker_size_mm": [board_wh(spec)[0] + 2 * QUIET_MM, board_wh(spec)[1] + 2 * QUIET_MM],
            "marker_corners_mm": {str(int(i)): np.asarray(p).reshape(4, 3).tolist() for i, p in zip(b.getIds().ravel(), op)},
        }
    cb = make_charuco()
    out["boards"]["CHARUCO"] = {
        "type": "CharucoBoard", "dictionary": "DICT_5X5_100",
        "squares_x_y": list(CHARUCO["squares"]), "square_mm": float(cb.getSquareLength()),
        "marker_mm": float(cb.getMarkerLength()), "legacy_pattern": bool(cb.getLegacyPattern()),
        "ids": [int(i) for i in cb.getIds().ravel()],
        "board_size_mm": [CHARUCO["squares"][0] * CHARUCO["square_mm"], CHARUCO["squares"][1] * CHARUCO["square_mm"]],
        "chessboard_corners_mm": np.asarray(cb.getChessboardCorners()).reshape(-1, 3).tolist(),
        "marker_corners_mm": {str(int(i)): np.asarray(p).reshape(4, 3).tolist() for i, p in zip(cb.getIds().ravel(), cb.getObjPoints())},
        "note": "Top-left square is black (first marker id 0 sits in square col 1,row 0). Construct with cv2.aruco.CharucoBoard((7,5),25.0,18.0,DICT_5X5_100), default (non-legacy) pattern.",
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    lay, sb = make_marker_pdf(os.path.join(a.outdir, "markers_head_jaw.pdf"))
    ch, sb2 = make_charuco_pdf(os.path.join(a.outdir, "charuco_calibration_A4.pdf"))
    geometry_json(os.path.join(a.outdir, "board_geometry.json"))
    json.dump({"marker_sheet": {"boards": lay, "scale_bar_y_mm": sb, "scale_bar_x_mm": 20.0},
               "charuco_sheet": {"board": ch, "scale_bar_y_mm": sb2}},
              open(os.path.join(a.outdir, "verify", "layout.json"), "w"), indent=1)
    print("wrote PDFs + board_geometry.json to", a.outdir)


if __name__ == "__main__":
    main()
