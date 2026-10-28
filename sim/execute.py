"""OmniManip execution in the sim (paper Sec. 3.3 closed-loop execution; supp. A.4 primitives; onboarding notes O10)
for a planned run (python -m omnimanip.run --sim ... --gt).

  grasp  move to 8 cm in front of the grasp pose, move forward, close (supp. A.4 (a)), lift 5 cm (DEVIATIONS #33)
  pour   move to the O9 target pose (supp. A.4 (f)); hold there
  place  move to v cm in front of the O9 target pose, move to the target, open the gripper (supp. A.4 (b); v = |d|, O10)
  Both: the target is re-solved on every tracker update, with Phi re-measured from the tracked active pose, until
  the EE is within 2 cm of it (O10), and the path is re-planned when it moves

Motion (omnimanip/motion.py): cuRobo plans the transit to the pre-grasp and the pour against the ESDF of the
planning frame's depth, the stage's move carrying the held object as spheres; the short forward move into the grasp is a
straight Cartesian line on mink IK. Joint paths are tracked by Albus's joint-position Tracker on the live
physics; the grasp is friction. Tracker stand-in: the sim's ground-truth body poses carrying the run's registration
error (each object's noise from objects.json, fixed). Writes exec.json and exec.mp4 under --run.

Run from third_party/ur5e-manip-sim (add --view, with mjpython in place of python, to watch it live):
  MUJOCO_GL=cgl PYTHONPATH=.:../..:../ur5e-manip-hardware/ros2_ws/src/manip_bridge ../../sim/.venv/bin/python \
      ../../sim/execute.py --run ../../outputs/sim/run_near3 --frame ../../outputs/sim/frame_near
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
from manip_sim.execution import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LiveArm,
    SlipMonitor,
    Tracker,
    tracking_controller_config,
)
from manip_sim.planning import ArmKinematics, MinkIK
from manip_sim.scene import load_scene, make_env
from scipy.spatial.transform import Rotation

from omnimanip import constraints as C
from omnimanip import motion, urdf_fk
from omnimanip import targetpose as TP
from omnimanip.shared_perception import read_obj

PRE_GRASP = (
    0.08,
    0.12,
    0.16,
)  # m in front of the grasp pose: the paper's 8 cm (supp. A.4) first, longer when
# cuRobo's conservative sphere/voxel model rejects it (onboarding notes: re-measure the 8 cm for the Robotiq)
STEP_T, STEP_R = 0.01, np.radians(3)  # max Cartesian waypoint spacing
RESOLVE_STOP = 0.02  # m: stop re-solving O9 this close to the target (O10)
REPLAN_T, REPLAN_R = (
    0.01,
    np.radians(3),
)  # re-plan when a re-solved target moves this far
CHUNK = 8  # path waypoints tracked per tracker update
LIFT = 0.05  # m straight up after the grasp closes, so the planned move to the target starts out of contact
SPHERE_MARGIN = 0.0  # m added to the held object's sphere radii: the covering spheres already enclose the mesh and
# cuRobo's 5 mm activation distance is the buffer (5 mm here put tight pour poses themselves in collision)
SETTLE_MM, SETTLE_STEPS = (
    1.0,
    100,
)  # before closing, hold the grasp pose until the EE is this close (step cap)
MASK_GROW = 5  # px: the held object's mask is grown by this before it is cleared from the map (edge pixels)
DWELL = 40  # control steps held at the pour pose
SITE = "gripper0_right_grip_site"  # robosuite: x = closing, z = approach
GRASP_R_SITE = np.array(
    [[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]
)  # site axes in AnyGrasp's frame (x approach, y closing)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument(
    "--frame", required=True, help="the sim/capture.py frame the run was planned on"
)
ap.add_argument(
    "--view",
    action="store_true",
    help="live MuJoCo viewer (macOS: run with mjpython)",
)
ap.add_argument(
    "--distance-cm",
    type=float,
    default=None,
    help="TEST ONLY: distance d to use when the VLM's (RRC interaction_value) is 0; negative = against the passive axis",
)
ap.add_argument(
    "--speed",
    type=float,
    default=3.0,
    help="viewer playback x real time (3 = exec.mp4, one frame per 3 steps at 20 fps)",
)
a = ap.parse_args()
run = Path(a.run)
meta = json.loads((Path(a.frame) / "meta.json").read_text())
objects = json.loads((run / "objects.json").read_text())
grasp = json.loads((run / "grasp" / "grasp.json").read_text())[0]
stage = json.loads((run / "rrc" / "rrc.json").read_text())[0]
c = {
    k: np.asarray(v) if isinstance(v, list) else v for k, v in stage["accepted"].items()
}
A, Pv, ACT = stage["active"], stage["passive"], stage["action"]
if a.distance_cm is not None and c["distance_m"] == 0:
    c["distance_m"] = a.distance_cm / 100
W_C, W_B = np.array(meta["world_T_cam"]), np.array(meta["world_T_base"])
C_W, B_W = np.linalg.inv(W_C), np.linalg.inv(W_B)
B_C = B_W @ W_C

env, _ = make_env(
    load_scene(meta["scene"]),
    has_renderer=False,
    controller_configs=tracking_controller_config("UR5e"),
)
arm = LiveArm(env, SITE)
tracker = Tracker(env, arm)
kin = ArmKinematics(env, SITE)
ik = MinkIK(kin, SITE)
m, d = arm.model, arm.data

# grasp frame -> grip site: the pads' tips sit `depth` past the grasp centre (gripper.py's convention)
s_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
Rs = d.site_xmat[s_id].reshape(3, 3)
pad_tip = max(
    (Rs.T @ (d.geom_xpos[g] - d.site_xpos[s_id]))[2]
    + np.abs((Rs.T @ d.geom_xmat[g].reshape(3, 3))[2]) @ m.geom_size[g]
    for g in range(m.ngeom)
    if "fingerpad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
)
G = np.eye(4)
G[:3, :3], G[:3, 3] = GRASP_R_SITE, [grasp["depth"] - pad_tip, 0, 0]
# grip site -> tool0 of the planner's robot model (robot0_base is base_link: same joints, checked to ~1 mm)
Y = (
    np.linalg.inv(
        urdf_fk.fk(
            urdf_fk.chain(motion.ROBOT_URDF),
            dict(zip(motion.UR_JOINTS, arm.q())),
        )
    )
    @ B_W
    @ arm.ee_pose()
)


def tracked(name):
    """Tracker stand-in: cam_T_can of a plan object from its ground-truth body pose plus its fixed registration
    error. -> (tracked, ground truth)."""
    o = objects[name]
    b = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, meta["objects"][o["sim_object"]]["body"]
    )
    gt = np.eye(4)
    gt[:3, :3], gt[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    gt = C_W @ gt
    T = gt.copy()
    T[:3, :3], T[:3, 3] = (
        np.array(o["noise"]["R_noise_cam"]) @ gt[:3, :3],
        gt[:3, 3] + o["noise"]["t_noise_cam"],
    )
    Rc = np.array(o["canonical_frame"]["R_body_from_canon"])
    T[:3, :3], gt[:3, :3] = T[:3, :3] @ Rc, gt[:3, :3] @ Rc
    return T, gt


def ee_cam():
    """Current EE (AnyGrasp grasp frame) in the camera frame."""
    return C_W @ arm.ee_pose() @ np.linalg.inv(G)


SEEDS = np.random.default_rng(0).uniform(-np.pi, np.pi, (20, 6))


def go(T_cam, gripper):
    """IK for the grasp-frame pose T_cam and track there; stays put if IK fails. -> reached.
    The solution seeded at the current joints is kept when it converges (continuous Cartesian moves); otherwise
    the converged random-seed solution closest to the current joints, collision-free if any is, each joint
    wrapped to its nearest equivalent (+-2 pi) so the arm never spins through full turns."""
    q0, T_w = arm.q(), W_C @ T_cam @ G
    q, ok = ik.solve(T_w, q0)
    if not ok:
        sols = []
        for s in SEEDS:
            q, ok = ik.solve(T_w, s)
            q = q0 + (q - q0 + np.pi) % (2 * np.pi) - np.pi
            if ok and np.all(
                (q >= kin.joint_range[:, 0]) & (q <= kin.joint_range[:, 1])
            ):
                sols.append((kin.in_collision(q), np.abs(q - q0).max(), q))
        if not sols:
            return False
        q = min(sols, key=lambda s: s[:2])[2]
    return not tracker.track(np.array([q0, q]), gripper, on_step=on_step).stalled


def tool0(T_cam):
    """base_T_tool0 for a grasp-frame pose in the camera frame."""
    return B_C @ T_cam @ G @ np.linalg.inv(Y)


def follow(path, gripper):
    """Track a planned joint path. -> reached."""
    return not tracker.track(np.asarray(path), gripper, on_step=on_step).stalled


def line(T0, T1, n_min=1):
    """Cartesian waypoints from T0 to T1 (excluding T0), <= STEP_T / STEP_R apart."""
    dr = Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T)
    n = max(
        n_min,
        int(
            np.ceil(
                max(
                    np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / STEP_T,
                    dr.magnitude() / STEP_R,
                )
            )
        ),
    )
    for k in range(1, n + 1):
        T = np.eye(4)
        T[:3, :3] = (
            Rotation.from_rotvec(dr.as_rotvec() * k / n).as_matrix() @ T0[:3, :3]
        )
        T[:3, 3] = T0[:3, 3] + (T1[:3, 3] - T0[:3, 3]) * k / n
        yield T


# video from the planning camera
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "frontview")
q = np.zeros(4)
mujoco.mju_mat2Quat(q, (W_C[:3, :3] @ np.diag([1.0, -1.0, -1.0])).ravel())
m.cam_pos[cam], m.cam_quat[cam] = W_C[:3, 3], q
m.cam_fovy[cam] = np.degrees(2 * np.arctan(240 / meta["K"][1][1]))
renderer, opt = mujoco.Renderer(m, 480, 640), mujoco.MjvOption()
opt.geomgroup[0] = 0
frames, n_steps = [], [0]
p0, passive_moved = (
    tracked(Pv)[1][:3, 3],
    [0.0],
)  # passive object's start position, max drift (m)
slip = None
if a.view:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(m, d)


def on_step():
    n_steps[0] += 1
    if a.view:
        viewer.sync()
        time.sleep(env.control_timestep / a.speed)
    passive_moved[0] = max(
        passive_moved[0], float(np.linalg.norm(tracked(Pv)[1][:3, 3] - p0))
    )
    if slip is not None:
        slip.update(arm.ee_pose(), W_C @ tracked(A)[1])
    if n_steps[0] % 3 == 0:
        renderer.update_scene(d, camera=cam, scene_option=opt)
        frames.append(renderer.render())


log = {"grasp": {}, ACT: {"updates": []}}

depth0 = (
    cv2.imread(str(run / "depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    / 1000
)
masks = {
    n: cv2.imread(str(run / "marks" / f"mask_{o['mark']}.png"), 0) > 127
    for n, o in objects.items()
}
q_cap = np.array(
    meta["joints"]["q"]
)  # the arm when the planning frame was captured (robot masking)
motion.map_scene(depth0, meta["K"], B_C, q_cap)

# grasp primitive: pre-grasp (planned), forward (Cartesian), close
T_g = np.array(grasp["cam_T_grasp"])
for pre in PRE_GRASP:
    T_pre = T_g.copy()
    T_pre[:3, 3] -= pre * T_g[:3, 0]
    path, rep = motion.plan(arm.q(), tool0(T_pre), q_goal_ref=arm.q())
    if path is not None:
        break
log["grasp"]["plan_pre"] = rep.get("reason") or bool(rep.get("success"))
log["grasp"]["pre_grasp_m"] = pre
if path is None:
    raise SystemExit(f"no plan to the pre-grasp: {rep.get('reason')}")
log["grasp"]["reached_pre"] = follow(path, GRIPPER_OPEN)
log["grasp"]["ik_forward"] = all(go(T, GRIPPER_OPEN) for T in line(T_pre, T_g))
# settle: the joint tracker advances at 0.015 rad per joint (a few mm at the pads); hold the grasp pose's IK
# solution until the EE is within SETTLE_MM, so the closing fingers meet the object together instead of one
# shoving it sideways first
q_g, _ = ik.solve(W_C @ T_g @ G, arm.q())
for k in range(SETTLE_STEPS):
    if np.linalg.norm(ee_cam()[:3, 3] - T_g[:3, 3]) * 1000 <= SETTLE_MM:
        break
    tracker.hold(q_g, GRIPPER_OPEN, 1, on_step=on_step)
log["grasp"]["settle_steps"], log["grasp"]["settle_error_mm"] = (
    k,
    float(np.linalg.norm(ee_cam()[:3, 3] - T_g[:3, 3]) * 1000),
)
log["grasp"]["contacts"] = tracker.close_gripper(
    arm.q(),
    meta["objects"][objects[A]["sim_object"]]["body"].split("_")[0],
    on_step=on_step,
)
log["grasp"]["ee_error_mm"] = float(np.linalg.norm(ee_cam()[:3, 3] - T_g[:3, 3]) * 1000)
# where each finger touches the object, in the grasp frame (x approach, y closing; origin = AnyGrasp's grasp point)
held = meta["objects"][objects[A]["sim_object"]]["body"].split("_")[0]
log["grasp"]["finger_contacts_grasp_frame_mm"] = [
    {
        "finger": (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        ).replace("gripper0_right_", ""),
        "xyz": np.round(
            (np.linalg.inv(T_g) @ C_W @ np.r_[con.pos, 1])[:3] * 1000, 1
        ).tolist(),
    }
    for con in d.contact[: d.ncon]
    for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1))
    if (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    ).startswith("gripper0")
    and (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[o]) or ""
    ).startswith(held)
]
print("grasp:", log["grasp"])
slip = SlipMonitor(np.linalg.inv(arm.ee_pose()) @ W_C @ tracked(A)[1])
T_up = ee_cam()
T_up[:3, 3] += LIFT * C_W[:3, 2]  # world +z in the camera frame
log["grasp"]["lift"] = all(go(T, GRIPPER_CLOSE) for T in line(ee_cam(), T_up))

# the stage's primitive (pour / place), closed loop. The held object leaves the map and rides on tool0 as spheres.
obst = TP.obstacles(depth0, np.array(meta["K"]), np.any(list(masks.values()), axis=0))
grown = (
    cv2.dilate(masks[A].astype(np.uint8), np.ones((2 * MASK_GROW + 1,) * 2, np.uint8))
    > 0
)
motion.map_scene(depth0, meta["K"], B_C, q_cap, exclude=grown)
Rc = np.eye(4)  # canonical -> body (the mesh is in the body frame)
Rc[:3, :3] = np.array(objects[A]["canonical_frame"]["R_body_from_canon"])
V, F = read_obj(Path(objects[A]["final_mesh"]))


def approach(T_t, T_p):
    """The planned goal for target T_t: the target itself (pour), or v = |d| in front of it along the passive
    axis (place)."""
    T = T_t.copy()
    if ACT == "place":
        T[:3, 3] -= abs(c["distance_m"]) * (T_p[:3, :3] @ c["passive_axis"])
    return T


def held_spheres(margin):
    """The held object's spheres on tool0 where the tracker sees it now (follows any slip in the grip)."""
    return motion.attached_spheres(
        V, F, np.linalg.inv(tool0(ee_cam())) @ B_C @ tracked(A)[0] @ Rc.T, margin
    )


T_t, T_plan, solving, path, k = None, np.eye(4), True, None, 0
T_start = ee_cam()  # O9's path cost is measured from the stage's start pose, so a static scene gives a fixed target
for it in range(200):
    T_a, _ = tracked(A)
    T_p, _ = tracked(Pv)
    T_e = ee_cam()
    if solving:  # O9 on this tracker update: Phi from the tracked active pose, warm-started at the last target
        phi = np.linalg.inv(T_e) @ T_a
        T_t, terms, ok = TP.solve(
            c, phi, T_start, T_p, obst, grasp["width"], x0_pose=T_t
        )
        solving = np.linalg.norm(T_t[:3, 3] - T_e[:3, 3]) > RESOLVE_STOP
        log[ACT]["updates"].append(
            {
                "it": it,
                "ok": ok,
                **terms,
                "slip_mm": slip.history[-1][0] * 1000 if slip.history else 0.0,
                "slip_deg": float(np.degrees(slip.history[-1][1]))
                if slip.history
                else 0.0,
            }
        )
        moved = path is None or (
            np.linalg.norm(T_t[:3, 3] - T_plan[:3, 3]) > REPLAN_T
            or Rotation.from_matrix(T_t[:3, :3] @ T_plan[:3, :3].T).magnitude()
            > REPLAN_R
        )
        if moved:
            spheres = held_spheres(SPHERE_MARGIN)
            ref = (
                arm.q() if path is None else path[-1]
            )  # stay on the previous plan's IK branch
            new, rep = motion.plan(arm.q(), tool0(approach(T_t, T_p)), spheres, ref)
            log[ACT].setdefault("plans", []).append(
                {
                    "it": it,
                    "success": bool(rep.get("success")),
                    "reason": rep.get("reason", ""),
                    "travel_rad": None
                    if new is None
                    else np.abs(np.diff(new, axis=0)).sum(0).round(2).tolist(),
                }
            )
            if new is None and path is not None:
                # e.g. already inside the cup's margin near the end: stop re-solving and finish the last
                # collision-checked path to its target
                T_t, solving = T_plan, False
            elif (path := new) is None:
                log[ACT]["failed"] = f"no plan at update {it}: {rep.get('reason')}"
                log[ACT]["q_start"], log[ACT]["spheres_tool0"] = (
                    arm.q().tolist(),
                    spheres.tolist(),
                )
                log[ACT]["base_T_tool0_goal"] = tool0(approach(T_t, T_p)).tolist()
                break
            else:
                T_plan, k = T_t, 0
    if k >= len(path):
        break
    if not follow(path[k : k + CHUNK + 1], GRIPPER_CLOSE):
        log[ACT]["failed"] = f"tracking stalled at update {it}"
        break
    if not arm.contacts_between("gripper0", held):
        log[ACT]["failed"] = f"object lost from the grip at update {it}"
        break
    k += CHUNK
slip_max = (slip.max_dpos, slip.max_drot)
if (
    ACT == "place" and "failed" not in log[ACT]
):  # forward to the target, release, let the object settle
    log[ACT]["forward"] = all(go(T, GRIPPER_CLOSE) for T in line(ee_cam(), T_t))
    slip = None
    tracker.hold(arm.q(), GRIPPER_OPEN, DWELL, on_step=on_step)
tracker.hold(
    arm.q(), GRIPPER_OPEN if ACT == "place" else GRIPPER_CLOSE, DWELL, on_step=on_step
)

# outcome, on the ground-truth poses
(Ta, Ta_gt), (_, Tp_gt) = tracked(A), tracked(Pv)
a_pt, va, b_pt, vb = C.world(c, Ta_gt, Tp_gt)
log["result"] = {
    "rho_true": C.residual(c, Ta_gt, Tp_gt),
    "active_point_to_target_cm": float(
        np.linalg.norm(a_pt - (b_pt + c["distance_m"] * vb)) * 100
    ),
    "axis_angle_deg": float(np.degrees(np.arccos(np.clip(va @ vb, -1, 1)))),
    "contacts_end": arm.contacts_between(
        "gripper0", meta["objects"][objects[A]["sim_object"]]["body"].split("_")[0]
    ),
    "slip_max_mm": slip_max[0] * 1000,
    "slip_max_deg": float(np.degrees(slip_max[1])),
    "passive_moved_max_mm": passive_moved[0] * 1000,
    "cam_T_active_final_gt": Ta_gt.tolist(),
    "cam_T_active_target_final": (T_t @ phi).tolist(),
    "updates": len(log[ACT]["updates"]),
    "iterations": it + 1,
    # world frame: how far the active point ends from the passive point sideways, and how far below it
    "active_point_xy_from_passive_cm": float(
        np.linalg.norm((W_C[:3, :3] @ (a_pt - b_pt))[:2]) * 100
    ),
    "active_point_below_passive_cm": float(-(W_C[:3, :3] @ (a_pt - b_pt))[2] * 100),
}
print("result:", log["result"])
(run / "exec.json").write_text(json.dumps(log, indent=1, default=float))
imageio.mimsave(run / "exec.mp4", frames, fps=20)
