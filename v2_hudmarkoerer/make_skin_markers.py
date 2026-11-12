"""A4 sheet with small individual skin markers (DICT_4X4_50, 12 mm, IDs 20-22 cranium, 30-32 mandible)."""
import cv2, numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

SIDE, MARGIN = 12.0, 3.0           # marker side, white border (mm)
CELL = SIDE + 2 * MARGIN           # 18 mm sticker
LABELS = {20: "20 NÆSERYG", 21: "21 PANDE H", 22: "22 PANDE V", 30: "30 HAGE", 31: "31 KÆBERAND H", 32: "32 KÆBERAND V"}
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def draw_marker(c, mid, x, y):
    """x, y = lower-left corner of the 18 mm sticker (points)."""
    bits = cv2.aruco.generateImageMarker(D, mid, 6)  # 6x6 cells incl. black border
    c.setStrokeColorRGB(0.6, 0.6, 0.6); c.setLineWidth(0.3); c.setDash(1.5, 1.5)
    c.rect(x, y, CELL * mm, CELL * mm, stroke=1, fill=0)
    c.setDash(); c.setFillColorRGB(0, 0, 0)
    cell = SIDE / 6.0
    for r in range(6):
        for q in range(6):
            if bits[r, q] < 128:
                c.rect(x + (MARGIN + q * cell) * mm, y + (MARGIN + SIDE - (r + 1) * cell) * mm,
                       cell * mm + 0.05, cell * mm + 0.05, stroke=0, fill=1)
    c.setFont("Helvetica", 6); c.drawCentredString(x + CELL * mm / 2, y - 7, LABELS[mid])

c = canvas.Canvas("hudmarkoerer_A4.pdf", pagesize=A4)
W, H = A4
c.setFont("Helvetica-Bold", 13); c.drawString(20 * mm, H - 18 * mm, "JawTrack hudmarkører (12 mm) — 5 sæt")
c.setFont("Helvetica", 9)
c.drawString(20 * mm, H - 24 * mm, "Udskriv i 100 % / faktisk størrelse på mat papir eller mat etiketark. Mål skalabjælken: 100,0 mm.")
c.drawString(20 * mm, H - 29 * mm, "Klip langs den stiplede linje (behold den hvide kant). 20–22 på kraniet, 30–32 på underkæben.")
y0 = H - 36 * mm
c.setLineWidth(0.8); c.line(20 * mm, y0, 120 * mm, y0)
for xx in (20, 120):
    c.line(xx * mm, y0 - 2 * mm, xx * mm, y0 + 2 * mm)
c.setFont("Helvetica", 7); c.drawString(124 * mm, y0 + 4 * mm, "100,0 mm")
ids = [20, 21, 22, 30, 31, 32]
for row in range(5):
    y = H - (48 + CELL + row * 30) * mm
    for k, mid in enumerate(ids):
        draw_marker(c, mid, (20 + k * 28) * mm, y)
c.setFont("Helvetica", 8)
c.drawString(20 * mm, 22 * mm, "H/V = patientens højre/venstre. Placeringen inden for hver gruppe er vejledende; kun gruppen (20–22 / 30–32) er afgørende.")
c.save()
