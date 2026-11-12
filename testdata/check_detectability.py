#!/usr/bin/env python3
"""Sanity check that a synthetic video is trackable: detect both GridBoards with cv2.aruco (OpenCV 5),
solvePnP with the TRUE camera, and compare T_C_H, T_C_J and M = inv(T_C_H) T_C_J with ground truth.
This is a minimal reference tracker (no filtering), not the pipeline.

    python3 check_detectability.py synth_quick [--step 5]
"""
import argparse, json, os, sys
import cv2
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dataset")
ap.add_argument("--step", type=int, default=5, help="analyse every N-th frame")
a = ap.parse_args()
gt = json.load(open(os.path.join(a.dataset, "ground_truth.json")))
raw = json.load(open(os.path.join(a.dataset, "raw_tracking_true.json")))
K, dist = np.array(gt["camera"]["K"]), np.array(gt["camera"]["dist"])
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
boards = {"head": cv2.aruco.GridBoard((3, 2), 15.0, 3.0, d, np.arange(6)),
          "jaw": cv2.aruco.GridBoard((2, 2), 15.0, 3.0, d, np.array([10, 11, 12, 13]))}
prm = cv2.aruco.DetectorParameters()
prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
det = cv2.aruco.ArucoDetector(d, prm)
cap = cv2.VideoCapture(os.path.join(a.dataset, "video.mp4"))
X = np.array(gt["rig"]["X_T_H_S"]).reshape(4, 4)
TLJ = np.array(gt["rig"]["T_L_J"]).reshape(4, 4)
ip = np.array(gt["landmarks_mm"]["incisal_point"])
res = {"head": [], "jaw": [], "M": [], "ip": []}
nd = {"head": 0, "jaw": 0}
i, n = 0, 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i % a.step == 0:
        n += 1
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        T = {}
        for name, b in boards.items():
            if ids is None:
                continue
            obj, img = b.matchImagePoints(corners, ids)
            if obj is None or len(obj) < 8:
                continue
            okp, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not okp:
                continue
            Te = np.eye(4); Te[:3, :3] = cv2.Rodrigues(rv)[0]; Te[:3, 3] = tv.ravel()
            Tg = np.array(raw["frames"][i][name]["T"]).reshape(4, 4)
            nd[name] += 1
            T[name] = (Te, Tg)
            res[name].append(np.linalg.norm(Te[:3, 3] - Tg[:3, 3]))
        if len(T) == 2:
            Me = np.linalg.inv(T["head"][0]) @ T["jaw"][0]
            Mg = np.linalg.inv(T["head"][1]) @ T["jaw"][1]
            M0 = X @ TLJ
            TSe = np.linalg.inv(X) @ Me @ np.linalg.inv(M0) @ X
            TSg = np.array(gt["frames"][i]["T"]).reshape(4, 4)
            res["ip"].append(np.linalg.norm(TSe[:3, :3] @ ip + TSe[:3, 3] - TSg[:3, :3] @ ip - TSg[:3, 3]))
    i += 1
print(f"frames analysed: {n}; head detected {nd['head']}, jaw detected {nd['jaw']}")
for k, lab in (("head", "T_C_H origin err (mm)"), ("jaw", "T_C_J origin err (mm)"), ("ip", "incisal point err, raw per-frame PnP, true X and M0 (mm)")):
    x = np.array(res[k])
    if len(x):
        print(f"{lab}: median {np.median(x):.3f}  95th {np.percentile(x, 95):.3f}  max {x.max():.3f}")
