"""Marker-board definitions (SPEC section 1).

Observed OpenCV 5.0 GridBoard convention (verified in tests/test_math.py): object points have the
origin at the TOP-LEFT corner of the top-left marker, +x to the right, +y DOWN (marker row 1 has
larger y), all z = 0.  The implied right-handed +z = x × y points INTO the board (away from a
camera that looks at the printed face).  Marker corners are ordered TL, TR, BR, BL.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
CHARUCO_DICT = cv2.aruco.DICT_5X5_100


@dataclass(frozen=True)
class BoardSpec:
    name: str
    cols: int
    rows: int
    ids: tuple
    marker_mm: float
    gap_mm: float


HEAD = BoardSpec("head", 3, 2, (0, 1, 2, 3, 4, 5), 15.0, 3.0)
JAW = BoardSpec("jaw", 2, 2, (10, 11, 12, 13), 15.0, 3.0)
BOARDS = (HEAD, JAW)

CHARUCO_SQUARES = (7, 5)
CHARUCO_SQUARE_MM = 25.0
CHARUCO_MARKER_MM = 18.0


def aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)


def make_gridboard(spec: BoardSpec, print_scale: float = 1.0):
    """cv2.aruco.GridBoard for a board spec, geometry multiplied by ``print_scale``."""
    return cv2.aruco.GridBoard((spec.cols, spec.rows), spec.marker_mm * print_scale,
                               spec.gap_mm * print_scale, aruco_dictionary(),
                               np.array(spec.ids, dtype=np.int32))


def board_object_points(spec: BoardSpec, print_scale: float = 1.0) -> dict:
    """{marker_id: (4,3) object corners in mm} in the board frame."""
    b = make_gridboard(spec, print_scale)
    return {int(i): np.asarray(p, np.float64).reshape(4, 3)
            for i, p in zip(b.getIds().ravel(), b.getObjPoints())}


def board_size_mm(spec: BoardSpec, print_scale: float = 1.0):
    w = (spec.cols * spec.marker_mm + (spec.cols - 1) * spec.gap_mm) * print_scale
    h = (spec.rows * spec.marker_mm + (spec.rows - 1) * spec.gap_mm) * print_scale
    return w, h


def make_charuco(print_scale: float = 1.0):
    """ChArUco calibration board (DICT_5X5_100, 7x5, 25/18 mm)."""
    return cv2.aruco.CharucoBoard(CHARUCO_SQUARES, CHARUCO_SQUARE_MM * print_scale,
                                  CHARUCO_MARKER_MM * print_scale,
                                  cv2.aruco.getPredefinedDictionary(CHARUCO_DICT))
