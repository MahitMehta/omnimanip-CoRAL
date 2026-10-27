#!/usr/bin/env bash
# N full sim runs of one task (perception with ground-truth poses, planning stages 11-16, execution), each in
# outputs/sim/<name>_<i> with its exec.mp4 / exec.json; run i uses noise seed i.
# Usage: [VLM_MODEL=...] [GRASP_ARGS=--sim-mesh-grasps] sim/run_batch.sh <frame_dir> <task> <name> <first_i> <n> [distance_cm: TEST-ONLY d used when the VLM's d is 0]
set -uo pipefail
cd "$(dirname "$0")/.."
FRAME=$1 TASK=$2 NAME=$3 FIRST=$4 N=$5 DIST=${6:-}
export PYTHONUNBUFFERED=1
for i in $(seq "$FIRST" $((FIRST + N - 1))); do
  OUT=outputs/sim/${NAME}_$i
  rm -rf "$OUT" && mkdir -p "$OUT"
  echo "== $OUT $(date +%H:%M:%S)"
  ok=1
  uv run python -m omnimanip.run --out "$OUT" --sim "$FRAME" --gt 0.005 2 --seed "$i" --task "$TASK" ${GRASP_ARGS:-} \
    > "$OUT/log_plan.txt" 2>&1 || { ok=0; echo "   failed at $(grep '^== ' "$OUT/log_plan.txt" | tail -1)"; }
  grep -h "ACCEPTED\|TASK FAILED\|grasps;\|OK |\|FAILED" "$OUT/log_plan.txt" 2>/dev/null | cut -c1-150 | sed 's/^/   /'
  [ $ok = 1 ] || continue
  (cd third_party/ur5e-manip-sim && MUJOCO_GL=cgl PYTHONPATH=.:../..:../ur5e-manip-hardware/ros2_ws/src/manip_bridge \
    ../../sim/.venv/bin/python ../../sim/execute.py --run "../../$OUT" --frame "../../$FRAME" \
    ${DIST:+--distance-cm "$DIST"} > "../../$OUT/log_execute.txt" 2>&1) || echo "   execute failed"
  python3 -c "
import json; r = json.load(open('$OUT/exec.json')); x = r['result']; a = [k for k in r if k not in ('grasp', 'result')][0]
print('   exec:', {k: round(x[k], 2) for k in ('active_point_to_target_cm', 'axis_angle_deg', 'slip_max_mm', 'passive_moved_max_mm')}, '| contacts', r['grasp'].get('contacts'), '->', x['contacts_end'], '|', r[a].get('failed', 'completed'))" 2>/dev/null || echo "   no exec.json"
done
