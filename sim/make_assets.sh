#!/usr/bin/env bash
# Sim assets for sim/scenes/pour_tea_cup.json, built with Albus's converter (CoACD hulls + robosuite sites) into
# sim/assets/ (his checkout is left untouched):
#   mug     YCB 025_mug (google_16k scan), the red mug of the real scene; 118 g (YCB object list)
#   teapot  his teapot asset scaled to 0.75 (closer to the real teapot's size); frames.json points scaled to match
#   marker  YCB 040_large_marker (google_16k), for the paper's insert-pen-into-holder style task; 15.8 g (YCB)
set -euo pipefail
cd "$(dirname "$0")"
SIM=../third_party/ur5e-manip-sim
PY=$PWD/.venv/bin/python
mkdir -p downloads/ycb assets
for o in 025_mug 040_large_marker; do
  [ -f downloads/ycb/$o/google_16k/textured.obj ] || curl -sSL \
    http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/${o}_google_16k.tgz | tar -xz -C downloads/ycb
done

cd "$SIM"
"$PY" scripts/convert_asset.py ../../sim/downloads/ycb/025_mug/google_16k/textured.obj --name mug --mass 0.118 \
  --texture ../../sim/downloads/ycb/025_mug/google_16k/texture_map.png --out ../../sim/assets
"$PY" scripts/convert_asset.py ../../sim/downloads/ycb/040_large_marker/google_16k/textured.obj --name marker --mass 0.0158 \
  --texture ../../sim/downloads/ycb/040_large_marker/google_16k/texture_map.png --out ../../sim/assets
"$PY" scripts/convert_asset.py assets/objects/teapot/meshes/teapot_visual.obj --name teapot --scale 0.75 \
  --out ../../sim/assets
"$PY" - <<'EOF'
import json
f = json.load(open("assets/objects/teapot/frames.json"))
for p in f["points"].values():
    p["xyz"] = [0.75 * v for v in p["xyz"]]
f["comment_scale"] = "points scaled by 0.75 with the mesh (sim/make_assets.sh); axes unchanged"
json.dump(f, open("../../sim/assets/teapot/frames.json", "w"), indent=1)
EOF
