# OmniManip baseline on the CoRAL UR5e

A reimplementation of [OmniManip](https://arxiv.org/abs/2501.03841) on a UR5e with a Robotiq 2F-85 and a fixed RealSense: a language instruction and an RGB-D view of a tabletop scene in, a grasp and an interaction stage executed out. Perception runs on the CoRAL GPU sidecars (Albus's shared perception), the VLM defaults to Gemini via litellm (easily configurable to other models).

- [Pipeline](#pipeline) below: one line per stage; `sim/PIPELINE.md` for sim.
- `DEVIATIONS.md`: every departure from the paper (real pipeline only, not sim).

## Layout

| | |
|---|---|
| `omnimanip/` | the pipeline: one module per stage, `run.py` (offline runner), `server.py` (stage sidecar for ROS), `extract_bag.py` |
| `ros2_ws/` | `omnimanip_interfaces` (typed services) + `omnimanip_ros` (`bridge`, `run_task`, `execute`), docker compose for CoRAL |
| `sim/` | MuJoCo capture + execution on Albus's ur5e-manip-sim (own venv) |
| `assets/` | robot URDF |
| `third_party/` | Albus's repos (git submodules, pinned) |

## Pipeline

Planning, stages 1–16, as `omnimanip.run` stages: `perception` (1–10) → `points` (11) → `directions` (12) → `constraints` (13) → `rrc` (14) → `grasp` (15) → `targetpose` (16). On the robot each stage is a typed service `/omnimanip/<stage>` (`omnimanip_interfaces`). The `bridge` node serves those services and forwards each call to our stage sidecar `omnimanip.server` (ZMQ, our Python 3.12 env); capture (1) runs in the bridge itself, on the live camera topics. `run_task` calls stages 1–16 and writes the same run directory as `omnimanip.run`, and `execute` then runs 17–20.

1. **Capture** (`perception`): takes 10 frames, makes a per-pixel median depth, fits the table plane, and asks cuRobo which pixels are the robot.
2. **Marks**: SAM 3 segments everything as "object", and Albus's selection drops the table, arm, background and duplicates, leaving numbered masks on `marked.png`.
3. **Stage split**: the VLM looks at `marked.png` and returns which mark IDs matter for the task, the stages (each an action from grasp, place, push, pull, rotate, pour, with an active and a passive object) and the interaction part on each object.
4. **Mesh** (per task object): TRELLIS.2 builds a textured 3D mesh from the masked object crop, in a unit-size box.
5. **Scale**: Any6D stretches that mesh along each axis to the object's real size, using the observed depth.
6. **Pose**: FoundationPose finds the scaled mesh's 6D pose in the camera and re-ranks its candidate poses against the mask.
7. **Yaw choice**: Orient Anything compares the real object crop with a render of each candidate and picks the one facing the right way (`--no-decide` on `omnimanip.run` or `run_task` skips this and the re-rank, keeping FoundationPose's own pick).
8. **Mesh copy**: the scaled mesh and its texture are copied from Any6D's per-object folder (a local copy on CoRAL, scp elsewhere).
9. **Canonical frame**: sets up = the table normal (refined by the object's symmetry axis) and front = Orient Anything's front, giving the object's canonical pose and mesh.
10. **Overlays**: draws the canonical axes and the mesh-versus-mask accuracy image (also scored for FoundationPose's own pick and Any6D's pose), and saves per-stage timings.
11. **Interaction points** (`points`): VLM picks dots on a SCAFFOLD grid over the object crop (upscaled when small; 8 dots along its long side, fewer along the short one by aspect ratio; visible parts, lifted with foreground depth), on front and side renders of the canonical mesh (invisible parts), or several on-object dots for the grasp part.
12. **Interaction directions** (`directions`): renders the object six times with one ± canonical axis drawn as a red arrow, VLM captions each arrow, and a text-only call ranks them for the stage (passive axis = the way the interaction enters it).
13. **Constraints** (`constraints`): fills the paper's template "active axis and passive axis are aligned; active point is d along the passive axis from the passive point" for each ranked active direction, with d set later by RRC.
14. **RRC** (`rrc`): moves the active object's mesh to the pose each candidate constraint implies, renders it over the inpainted frame, and the VLM answers success, failure or refine (refine resamples 6 nearby directions) plus the distance.
15. **Grasp** (`grasp`): AnyGrasp proposes grasps around the target object, those that would put the real Robotiq within 2 cm of the table (at the grasp or the 16 cm pre-grasp) or into the scene are dropped, a 1 cm Gaussian heatmap on the VLM's grasp-part dots keeps those with at least half the peak heat, and the highest-scoring of them gives the gripper-to-object transform Φ.
16. **Target pose** (`targetpose`): SLSQP finds the gripper pose whose held object (through Φ) best meets the accepted constraint while keeping the gripper spheres 2 cm from the scene and moving as little as possible; accepted if ρ ≤ 0.05.

Execution (O10), stages 17–20, runs the grasp and then the first interaction stage RRC accepted: `sim/execute.py` in sim, `omnimanip_ros execute` on the robot (UR driver trajectories, gripper-control's Robotiq action, FoundationPose tracking on live frames, an operator gate before every motion, its path latched first on `/omnimanip/trajectory` and `/omnimanip/ee_path` for RViz). Straight tool moves use IK on the planner's URDF, and every move is timed by a minimum-jerk law at 10 % of the joint limits.

17. **Grasp primitive**: move to 8 cm before the grasp pose along its approach axis (12 or 16 cm if that is refused), move straight in, close the Robotiq, lift 10 cm.
18. **Interaction primitive**: one collision-free planned move of the gripper, carrying the held object, to the O9 target pose, held there (pour); place instead goes to |d| before the target along the passive axis, then straight in, and opens.
19. **Closed-loop tracking**: FoundationPose tracks the active and passive objects on live frames, and O9 is re-solved on every update (Φ re-measured from the tracked active object) until the gripper is within 2 cm of the target, re-planning when the target moves 1 cm / 3°.
20. **Motion** (`omnimanip/motion.py`): cuRobo maps the planning frame's depth (robot masked out) and attaches the held object as covering spheres fitted to its mesh. Its MotionPlanner plans a collision-free joint path (5 mm buffer) to each pose, on our own cuRobo container. The goal is the IK solution nearest the current branch, with joints wrapped the short way round.

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
     ros2 run omnimanip_ros run_task --task "<instruction>" --out $RUNS_DIR'   # e.g. "Pour tea into the cup"; add --no-decide to skip the yaw decision
   ```
7. **Execute**: 10 % speed, Enter at every motion (anything else stops; Ctrl-C cancels a motion). Before each gate, check the path in RViz: a Path display on `/omnimanip/ee_path`, fixed frame `base_link`. Writes `exec.json`.
   ```bash
   docker exec -it OmniManipROS bash -c 'umask 0 && source install/setup.bash && \
     ros2 run omnimanip_ros execute --run $RUNS_DIR/$(ls -t $RUNS_DIR | head -1)'
   ```
8. **After**: `docker stop OmniManipROS`, kill the sidecar. Copy a run home with
   `rsync -a --exclude '*.obj' --exclude '*.glb' coral:omnimanip/outputs/runs/<stamp> outputs/runs/`.

Dry runs without the robot: replay a bag (or launch the UR driver with `use_fake_hardware:=true`) and gripper-control with `GRIPPER_BACKEND=mock`, all on another domain (`ROS_DOMAIN_ID=67`), and `execute --mock-tracking --controller joint_trajectory_controller`.

## Pending (optional)

- **Closed-loop grasp.** The grasp is open-loop: `execute` (and `sim/execute.py`) approach the planning frame's `cam_T_grasp`, so an object moved before the grasp is missed (the `stalled` check stops the run). Option: keep the AnyGrasp grasp fixed in the object frame and re-anchor it to the tracked pose (`T_obj_now · T_obj_plan⁻¹ · cam_T_grasp`) before the pre-grasp plan and before the straight approach, re-planning past 1 cm / 3°; freeze it during the approach (the gripper occludes the grasped part). The paper's grasp primitive (supp. A.4 (a)) doesn't say whether it is tracked, so this would be a DEVIATIONS row.
- **Held object in O9's collision loss.** Eq. 4 checks only the end-effector against the scene (with the task objects' masks removed), so a target can put the held object (through Φ) inside the passive object or the table; cuRobo then fails the plan at the goal (e.g. `pour_c_4`). Option: add the held object's spheres at `P_ee · Φ`, and the passive object's points, to `L_collision`. Departs from the paper's O9, so a DEVIATIONS row.
