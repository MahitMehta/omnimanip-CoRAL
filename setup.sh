#!/usr/bin/env bash
# Environment: uv venv + deps + Albus's shared-perception code (third_party/ submodules, pinned to what the CoRAL
# server runs). His repos are not pip packages (the sim has no packaging), so a .pth puts them on the venv's path.
set -euo pipefail
cd "$(dirname "$0")"

uv sync
git submodule update --init

SITE=$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$PWD/third_party/ur5e-manip-hardware/ros2_ws/src/manip_bridge" "$PWD/third_party/ur5e-manip-sim" > "$SITE/third_party.pth"
uv run python -c 'import manip_bridge.frame_packet, manip_sim.refine; print("shared perception importable")'
