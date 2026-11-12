"""Skin markers laid out for a pre-cut A4 label sheet (default: assumed HERMA 10107, 25.4 mm squares, 6 x 11).
The grid numbers are parameters: measure the real sheet and pass them if they differ."""
import argparse, cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

ap = argparse.ArgumentParser()
ap.add_argument("--label", type=float, default=25.4); ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--rows", type=int, default=11); ap.add_argument("--left", type=float, default=None)
ap.add_argument("--top", type=float, default=None); ap.add_argument("--pitch-x", type=float, default=27.94)
ap.add_argument("--pitch-y", type=float, default=25.4); ap.add_argument("--outline", action="store_true")
ap.add_argument("--out", default="hudmarkoerer_etiketark_HERMA10107.pdf")
a = ap.parse_args()
SIDE = 12.0
W, H = A4
left = a.left if a.left is not None else (210 - ((a.cols - 1) * a.pitch_x + a.label)) / 2
top = a.top if a.top is not None else (297 - ((a.rows - 1) * a.pitch_y + a.label)) / 2
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
NAMES = {20: "20 næseryg", 21: "21 pande H", 22: "22 pande V", 30: "30 hage", 31: "31 kæbe H", 32: "32 kæbe V"}
ids = [20, 21, 22, 30, 31, 32]
c = canvas.Canvas(a.out, pagesize=A4)
for r in range(a.rows):
    for q in range(a.cols):
        mid = ids[q % 6]
        x = left + q * a.pitch_x; y = 297 - top - r * a.pitch_y - a.label   # lower-left of the label, mm
        if a.outline:
            c.setStrokeColorRGB(0.7, 0.7, 0.7); c.setLineWidth(0.3)
            c.roundRect(x * mm, y * mm, a.label * mm, a.label * mm, 1.5 * mm)
        bits = cv2.aruco.generateImageMarker(D, mid, 6); cell = SIDE / 6
        mx = x + (a.label - SIDE) / 2; my = y + (a.label - SIDE) / 2 + 1.0
        c.setFillColorRGB(0, 0, 0)
        for i in range(6):
            for j in range(6):
                if bits[i, j] < 128:
                    c.rect((mx + j * cell) * mm, (my + SIDE - (i + 1) * cell) * mm, cell * mm + 0.05, cell * mm + 0.05, stroke=0, fill=1)
        c.setFont("Helvetica", 5.5); c.drawCentredString((x + a.label / 2) * mm, (y + 1.6) * mm, NAMES[mid])
c.save()
print(f"{a.out}: {a.cols}x{a.rows}, left {left:.2f} top {top:.2f} pitch {a.pitch_x}/{a.pitch_y}")
