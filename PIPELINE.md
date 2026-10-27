# Pipeline (one line per stage, kept current)

Run everything: `uv run python -m omnimanip.run --out <dir> [--start <frame>]` (one function per stage in `omnimanip/run.py`, all writing into `<dir>`; `--from <stage>` resumes). Stage order on a bag:
`perception` (1–10) → `points` (11) → `directions` (12) → `constraints` (13) → `rrc` (14) → `grasp` (15) → `targetpose` (16).

ROS 2 (hardware): each stage is a typed service `/omnimanip/<stage>` (`ros2_ws/src/omnimanip_interfaces`), served by `omnimanip_ros bridge`, a thin ROS 2 Humble node in Albus's Ros2Bridge image that forwards to our stage sidecar `python -m omnimanip.server` (ZMQ, our Python 3.12 env); capture (1) runs in the bridge on the live camera topics with Albus's frame packet. `ros2 run omnimanip_ros run_task --task ... --out RUNS_DIR` calls stages 1–16 in order and writes the same run directory as `omnimanip.run`; `ros2 run omnimanip_ros execute --run <run>` then runs stages 17–20 on the robot (below), stopping at an operator gate before every motion. Deployment: `ros2_ws/docker-compose.yml` (domain 66, CycloneDDS, as the robot laptop).

Built:
1. **Capture** (`perception`): takes 10 bag frames, makes a per-pixel median depth, fits the table plane, and asks cuRobo which pixels are the robot.
2. **Marks**: SAM 3 segments everything as "object", and Albus's selection drops the table, arm, background and duplicates, leaving numbered masks on `marked.png`.
3. **Stage split**: Gemini looks at `marked.png` and returns which mark IDs matter for the task, the stages (grasp, then pour) and the interaction part on each object.
4. **Mesh** (per task object): TRELLIS.2 builds a textured 3D mesh from the masked object crop, in a unit-size box.
5. **Scale**: Any6D stretches that mesh along each axis to the object's real size, using the observed depth.
6. **Pose**: FoundationPose finds the scaled mesh's 6D pose in the camera and re-ranks its candidate poses against the mask.
7. **Yaw choice**: Orient Anything compares the real object crop with a render of each candidate and picks the one facing the right way (`--no-decide` skips this and the re-rank, keeping FoundationPose's own pick).
8. **Mesh copy**: the scaled mesh and its texture are copied from Any6D's per-object folder on the server with scp.
9. **Canonical frame**: sets up = the table normal (refined by the object's symmetry axis) and front = Orient Anything's front, giving the object's canonical pose and mesh.
10. **Overlays**: draws the canonical axes and the mesh-versus-mask accuracy image (also scored for FoundationPose's own pick and Any6D's pose), and saves per-stage timings.
11. **Interaction points** (`points`): Gemini picks dots on a SCAFFOLD grid over the object crop (upscaled when small; 8 dots along its long side, fewer along the short one by aspect ratio; visible parts, lifted with foreground depth), on front and side renders of the canonical mesh (invisible parts), or several on-object dots for the grasp part.
12. **Interaction directions** (`directions`): renders the object six times with one ± canonical axis drawn as a red arrow, Gemini captions each arrow, and a text-only call ranks them for the stage (passive axis = the way the interaction enters it).
13. **Constraints** (`constraints`): fills the paper's template "active axis and passive axis are aligned; active point is d along the passive axis from the passive point" for each ranked active direction, with d set later by RRC.
14. **RRC** (`rrc`): moves the teapot's mesh to the pose each candidate constraint implies, renders it over the inpainted frame, and Gemini answers success, failure or refine (refine resamples 6 nearby directions) plus the distance.
15. **Grasp** (`grasp`): AnyGrasp proposes grasps around the teapot, and a 1 cm Gaussian heatmap on the VLM's handle dots picks the best-scoring grasp near them, giving the gripper-to-teapot transform Φ.
16. **Target pose** (`targetpose`): SLSQP finds the gripper pose whose held teapot (through Φ) best meets the accepted constraint while keeping the gripper spheres 2 cm from the scene and moving as little as possible; accepted if ρ ≤ 0.05.

Execution (O10; built in sim, `sim/execute.py`, and for the robot, `omnimanip_ros execute`: UR driver trajectories, gripper-control's Robotiq action, FoundationPose tracking on live frames; straight tool moves by IK on the planner's URDF, timed by a minimum-jerk law at 10 % of the joint limits):
17. **Grasp primitive**: move to 8 cm before the grasp pose along its approach axis, move straight in, close the Robotiq.
18. **Pour primitive**: one collision-free planned move of the gripper to the O9 target pose, carrying the teapot.
19. **Closed-loop tracking**: FoundationPose tracks the teapot and cup on live frames and O9 is re-solved until the gripper is within 2 cm of the target.
20. **Motion** (`omnimanip/motion.py`, built and tested in sim): cuRobo maps the planning frame's depth (robot masked out), attaches the held object as covering spheres fitted to its mesh, and its MotionPlanner plans a collision-free joint path (5 mm buffer) to each pose, choosing the goal IK solution nearest the current branch with joints wrapped the short way round, on our own cuRobo container.
