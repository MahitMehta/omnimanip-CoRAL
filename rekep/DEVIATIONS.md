# ReKep deviations log

One row per difference between our ReKep and the paper (arXiv 2409.01652; the real-robot appendix A.5–A.9). We
import ReKep's released modules (github.com/huangwl18/ReKep @ 63c43fd, pinned in third_party/ReKep), set to the
paper's values by `rekep/paper.patch` and `rekep/core.config`; where the paper is silent, the code's behaviour stands.

| # | Module | Paper | Ours | Why |
|---|---|---|---|---|
| 1 | robot | Franka on a wheeled frame, 2 Orbbec Femto Bolt cameras; Deoxys joint impedance | UR5e + Robotiq 2F-85, one fixed RealSense (640×480 aligned depth) | Our hardware; one view means more occlusion for the tracker and a one-sided SDF (row 5) |
| 2 | workspace | Predefined workspace bounds | ±0.85 m in x and y around base_link (UR5e reach), 2 cm under the table plane to 50 cm above it | No per-scene tuning; the 2 cm keeps table-contact points that depth noise puts under the fitted plane |
| 3 | keypoints | Grasped keypoints: "a rigid group of keypoints that belong to the same object or part; obtained from the segmentation model" | The largest SAM mask (at most half the image, as the proposer's masks) holding the grasped keypoint's pixel; every keypoint whose pixel lies in it rides with the end effector | SAM's masks nest (part inside object) and overlap, so a keypoint's own mask can be a part that leaves the rest of the object fixed in the optimizer |
| 4 | solvers | Reachability: PyBullet IK residual | `motion.ik` (damped least squares on the planner URDF); the cost counts its steps out of 20, as the released solvers score Lula's descents | Our robot model |
| 5 | solvers | Scene SDF: nvblox ESDF fused from all cameras at 20 Hz, arm (cuRobo) and held object (Cutie) removed | Voxels (1 cm) holding a depth point are occupied, Euclidean distance transform; robot mask and the held object's mask removed; what lies behind a surface counts as free | User's choice: our own SDF first |
| 6 | solvers | Solution consistency (A.8, A.9) and table clearance (A.9), no weights given | Sub-goal: 1.0 × the released `consistency` to the previous solution (rot weight 1.5, as its pose regularization). Path: 1.0 × `consistency` between the dense samples and the previous ones (sets). Table: 0.5 × the gripper and held points' depth under the table plane, summed (the path's collision weight) | The paper names the terms only; the released code's own forms and weights |
| 7 | solvers | End-effector frame unstated (Franka gripper) | AnyGrasp's grasp frame, fingertips at the origin, x = approach | The released solvers' grasp cost and `grasp_depth` push along x |
| 8 | program | GPT-4o | Gemini 3.7 Flash at minimal thinking, provider-default temperature and no token limit; the released prompt template and parser | The same VLM and settings as our OmniManip |
