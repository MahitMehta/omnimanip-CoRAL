# OmniManip baseline on the CoRAL UR5e

A reimplementation of [OmniManip](https://arxiv.org/abs/2501.03841) (`paper/`) on a UR5e with a Robotiq 2F-85 and a fixed RealSense. Perception runs on the CoRAL GPU sidecars (Albus's shared perception), the VLM is Gemini through litellm.

- `PIPELINE.md`: one line per stage, `sim/PIPELINE.md` for sim.
- `DEVIATIONS.md`: every departure from the paper (real pipeline only).

## Layout

| | |
|---|---|
| `omnimanip/` | the pipeline: one module per stage, `run.py` (offline runner), `server.py` (stage sidecar for ROS), `extract_bag.py` |
| `ros2_ws/` | `omnimanip_interfaces` (typed services) + `omnimanip_ros` (`bridge`, `run_task`, `execute`), docker compose for CoRAL |
| `sim/` | MuJoCo capture + execution on Albus's ur5e-manip-sim (own venv) |
| `assets/` | robot URDF |
| `third_party/` | Albus's repos (git submodules, pinned) |

## Setup

```bash
git clone --recursive <this repo>   # or: git submodule update --init
./setup.sh                   # uv env + the pinned third_party submodules on the path
cp .env.example .env         # GEMINI_API_KEY, HOST (the GPU box running the sidecars)
```

## Offline (a bag)

```bash
uv run python -m omnimanip.extract_bag bag/<stamp> extracted
uv run python -m omnimanip.run --out outputs/<name> [--start 58]     # stages 1-16; --from <stage> resumes
```

## Real robot

Everything runs on CoRAL (`ssh coral`) except the robot laptop's driver, camera and gripper; ROS domain 66, CycloneDDS. Runs are written on CoRAL in `~/omnimanip/outputs/runs/<stamp>/`.

1. **Pendant**: robot on, External Control program loaded. Objects within ~0.65 m of the base, fully in view. Someone on the e-stop.
2. **Laptop**: Albus's ros2-essentials (UR driver, RealSense, extrinsics); gripper-control with `GRIPPER_BACKEND=urcap`, `ROS_DOMAIN_ID=66`. Press play on External Control.
3. **Gripper check**: it must physically close and open, report `stalled: true` on an object, and close along tool0 x.
   ```bash
   ros2 action send_goal /gripper/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 60.0}}"
   ros2 action send_goal /gripper/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.085, max_effort: 60.0}}"
   ```
4. **CoRAL**: Albus's sidecars up (5667, 5669, 5670, 5672, 5673; restart PoseServer / Trellis2Server if a GPU is near full), then ours:
   ```bash
   docker stop gripper 2>/dev/null                       # the mock gripper must not run
   docker start CuroboServer-OmniManip       # AnyGrasp: Albus's on 5666; if it's down, start AnyGraspServer-OmniManip and set ANYGRASP_PORT=5676 in .env
   cd ~/omnimanip && PYOPENGL_PLATFORM=egl nohup .venv/bin/python -m omnimanip.server > outputs/server.log 2>&1 &
   cd ros2_ws && RUNS_DIR=$HOME/omnimanip/outputs/runs docker compose up -d --force-recreate   # domain 66
   ```
5. **Check** (camera ~30 Hz at 640x480, trajectory and gripper actions listed):
   ```bash
   docker exec -it OmniManipROS bash -c 'source install/setup.bash; ros2 topic hz /camera/camera/color/image_raw; \
     ros2 topic echo --once /camera/camera/color/camera_info | head -4; ros2 action list'
   ```
6. **Plan** (no motion, ~9 min), then check `canonical_overlay_*`, `grasp/`, `rrc/`, `targetpose/` in the run dir. Don't move the objects after this.
   ```bash
   docker exec -it OmniManipROS bash -c 'umask 0 && source install/setup.bash && \
     ros2 run omnimanip_ros run_task --task "Pour tea into the cup" --out $RUNS_DIR'
   ```
7. **Execute**: 10 % speed, Enter at every motion (anything else stops; Ctrl-C cancels a motion). Writes `exec.json`.
   ```bash
   docker exec -it OmniManipROS bash -c 'umask 0 && source install/setup.bash && \
     ros2 run omnimanip_ros execute --run $RUNS_DIR/$(ls -t $RUNS_DIR | head -1)'
   ```
8. **After**: `docker stop OmniManipROS`, kill the sidecar. Copy a run home with
   `rsync -a --exclude '*.obj' --exclude '*.glb' coral:omnimanip/outputs/runs/<stamp> outputs/runs/`.

Dry runs without the robot: replay a bag (or launch the UR driver with `use_fake_hardware:=true`) and gripper-control with `GRIPPER_BACKEND=mock`, all on another domain (`ROS_DOMAIN_ID=67`), and `execute --mock-tracking --controller joint_trajectory_controller`.

## Pending (optional)

- **Closed-loop grasp.** The grasp is open-loop: `execute` (and `sim/execute.py`) approach the planning frame's `cam_T_grasp`, so a teapot moved before the grasp is missed (the `stalled` check stops the run). Option: keep the AnyGrasp grasp fixed in the object frame and re-anchor it to the tracked pose (`T_obj_now · T_obj_plan⁻¹ · cam_T_grasp`) before the pre-grasp plan and before the straight approach, re-planning past 1 cm / 3°; freeze it during the approach (the gripper occludes the handle). The paper's grasp primitive (supp. A.4 (a)) doesn't say whether it is tracked, so this would be a DEVIATIONS row.
- **Held object in O9's collision loss.** Eq. 4 checks only the end-effector against the scene (with the task objects' masks removed), so a target can put the held object (through Φ) inside the passive object or the table; cuRobo then fails the plan at the goal (e.g. `pour_c_4`). Option: add the held object's spheres at `P_ee · Φ`, and the passive object's points, to `L_collision`. Departs from the paper's O9, so a DEVIATIONS row.
