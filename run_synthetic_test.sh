#!/usr/bin/env bash
# Kører hele den syntetiske test fra start til slut. Brug: ./run_synthetic_test.sh [output-mappe]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/_out}"
CLI="python3 $ROOT/pipeline/jawtrack_cli.py"
mkdir -p "$OUT"

echo "== 1/4  Enhedstests og indbygget ende-til-ende-test (pipeline/tests)"
(cd "$ROOT/pipeline" && python3 -m pytest -q tests)

echo "== 2/4  Generér syntetisk datasæt synth_quick (STL, video, kalibreringsklip, facit)"
python3 "$ROOT/testdata/make_synth.py" --variant quick --outdir "$OUT/testdata"

echo "== 3/4  Version 1 (markørplader): calibrate -> track -> solve -> report -> validate"
D="$OUT/testdata/synth_quick"; W="$OUT/v1_quick"; mkdir -p "$W"
$CLI calibrate --video "$D/charuco_calib.mp4" --out "$W/camera.json"
$CLI track --video "$D/video.mp4" --camera "$W/camera.json" --out "$W/raw_tracking.json"
$CLI solve --raw "$W/raw_tracking.json" --upper "$D/upper.stl" --lower "$D/lower.stl" \
     --landmarks "$D/registration_landmarks.json" --out "$W/motion.json" --out-positions "$W/positions"
$CLI report --motion "$W/motion.json" --outdir "$W/report"
python3 "$ROOT/testdata/validate.py" --motion "$W/motion.json" --gt "$D/ground_truth.json" \
     --raw "$W/raw_tracking.json" --outdir "$W/validation" || echo "(validate.py meldte FAIL/WARN - se rapporten)"

echo "== 4/4  Version 2 (hudmarkører): syntetisk selvtest, hud følger knoglen 100 %"
$CLI skin-selftest --outdir "$OUT/v2_skin" --lag 0.0

echo "FÆRDIG. Resultater i $OUT (se README.md, afsnittet 'Hvad resultatet skal vise')"
