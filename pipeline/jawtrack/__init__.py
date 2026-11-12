"""JawTrack processing pipeline.

iPhone video of a HEAD and a JAW ArUco GridBoard -> 6-DoF lower-jaw motion in upper-STL
coordinates (motion.json, see SPEC.md section 5.2) -> virtual-articulator parameters.

Modules
-------
se3           rigid-transform utilities (row-major serialisation, averaging, log/exp, slerp)
boards        board definitions (HEAD/JAW GridBoards, ChArUco) incl. print-scale
io            JSON / STL / camera file IO and the rotation-aware video reader
calib         ChArUco camera calibration and iPhone intrinsics presets
detect        ArUco marker detection (downscaled ROI detection + full-res corner refinement)
pose          board pose with explicit planar-ambiguity handling (IPPE candidates + Viterbi)
track         stage 1: video -> raw_tracking.json (+ reg_frame.png)
registration  STL <-> HEAD board registration X from clicked landmarks (SPEC section 4)
landmarks     incisal point, occlusal plane frame, condyles (Bonwill / kinematic hinge axis)
motion        M(t), MI detection, M0, T_S(t), filtering, gap interpolation
segments      automatic movement segmentation
params        articulator parameters (SPEC 5.2)
export        motion.json, key-position STLs, articulator_values.txt
report        PNG plots + report.md
"""

__version__ = "0.1.0"
