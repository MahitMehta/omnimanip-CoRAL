"""Collision-free motion between O9 targets (the paper names no planner; DEVIATIONS #40), on the cuRobo sidecar.

Scene-agnostic: obstacles are the depth of the planning frame integrated into the sidecar's ESDF (robot masked
out by its own FK), and the held object travels as spheres fitted to its mesh, attached to tool0. Plans come from
cuRobo's MotionPlanner (IK, then trajectory optimisation, seeded by its PRM graph planner on retries), so paths
are smooth and short; our server fork (ur5e-manip-hardware-OmniManip) adds the attached spheres to its "plan"
command. All poses are in the robot base frame (the sidecar's map frame).
"""

from functools import cache

import numpy as np
from manip_bridge.zmq_client import SidecarClient
from scipy.spatial.transform import Rotation, Slerp

from . import gripper, urdf_fk
from .config import HOST, ROOT, SIDECARS

UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ROBOT_URDF = (
    ROOT / "assets" / "ur5e_robotiq.urdf"
)  # the cuRobo server's model: base_link -> tool0 -> Robotiq
LINE_STEP_T, LINE_STEP_R = (
    0.005,
    np.radians(1),
)  # straight tool moves: an IK waypoint every 5 mm / 1 deg
MAX_JUMP = 0.2  # rad: a larger joint change between line waypoints is a branch flip or a singularity
N_SPHERES = 256  # the robot model's attached_object slots (raised from 16 in our model copy: ~11 mm spheres hug the surface)


def cli(timeout_s=120):
    return SidecarClient(f"tcp://{HOST}:{SIDECARS['curobo']}", timeout_s * 1000)


def map_scene(depth, K, T_base_cam, q, exclude=None):
    """Rebuild the ESDF from one depth frame: robot pixels removed by the sidecar (load_robot + q), `exclude`
    pixels (e.g. the held object) zeroed first."""
    c = cli()
    c.call({"cmd": "load_robot"})
    c.call({"cmd": "reset"})
    d = np.where(exclude, 0.0, depth) if exclude is not None else depth
    c.call(
        {
            "cmd": "integrate",
            "depth": d.astype(np.float32),
            "intrinsics": np.asarray(K, np.float32),
            "pose": np.asarray(T_base_cam, np.float32),
            "q": [float(v) for v in q],
            "joint_names": UR_JOINTS,
        }
    )


def attached_spheres(V, F, tool0_T_obj, margin=0.0, n_samples=12000, seed=0):
    """The held object's collision spheres in tool0: its mesh surface sampled and clustered into N_SPHERES
    groups (k-means), each sphere at a group's centroid with the radius reaching its farthest point, so the
    spheres cover the surface for any mesh (the sidecar's voxel fit needs a watertight one); `margin` (m) inflates
    every radius."""
    import trimesh
    from scipy.cluster.vq import kmeans2

    P = trimesh.sample.sample_surface(
        trimesh.Trimesh(V, F, process=False), n_samples, seed=seed
    )[0]
    c, lab = kmeans2(P, N_SPHERES, minit="++", rng=np.random.default_rng(seed))
    r = np.array(
        [
            np.linalg.norm(P[lab == k] - c[k], axis=1).max()
            if (lab == k).any()
            else 0.0
            for k in range(len(c))
        ]
    )
    return np.c_[c @ tool0_T_obj[:3, :3].T + tool0_T_obj[:3, 3], r + margin]


def plan(q, base_T_tool0_goal, spheres=None, q_goal_ref=None):
    """Joint trajectory (n, 6; UR order) from q to the tool0 goal, collision-free against the mapped scene and,
    with `spheres`, carrying them. With `q_goal_ref` (UR order) the goal is the IK solution nearest it, so replans
    stay on one arm configuration instead of swinging between IK branches. -> (path or None, reply)."""
    msg = {
        "cmd": "plan",
        "q": [float(v) for v in q],
        "joint_names": UR_JOINTS,
        "T_base_goal": np.asarray(base_T_tool0_goal, np.float32),
    }
    if spheres is not None:
        msg["attached_spheres"] = np.asarray(spheres, np.float32)
    if q_goal_ref is not None:
        msg["q_goal_ref"] = [float(v) for v in q_goal_ref]
    rep = cli(180).call(msg)
    if not rep.get("success"):
        return None, {**rep, "reason": rep.get("error", "")}
    order = [list(rep["joint_names"]).index(n) for n in UR_JOINTS]
    return np.asarray(rep["positions"], float)[:, order], rep


@cache
def _chain(base="base_link", tip="tool0"):
    return urdf_fk.chain(ROBOT_URDF, base, tip)


def fk(q):
    """base_T_tool0 at joints q (UR order) on the planner's URDF."""
    return urdf_fk.fk(_chain(), dict(zip(UR_JOINTS, q)))


def tool0_T_grasp(depth):
    """AnyGrasp's grasp frame in tool0 for a grasp whose fingertips reach `depth` past its origin: tool0 ->
    Robotiq base from the URDF, then the gripper model's base -> grasp frame (gripper.py)."""
    base = urdf_fk.fk(_chain("tool0", "robotiq_85_base_link"), {})
    return base @ np.linalg.inv(gripper.grasp_T_base(gripper.GRIPPER, depth))


def ik(T_goal, q_seed, tol_t=1e-4, tol_r=1e-3, damping=0.01, iters=100):
    """Damped least squares from q_seed on fk (numeric Jacobian). -> (q, converged)."""

    def err(q):
        T = fk(q)
        return np.r_[
            T_goal[:3, 3] - T[:3, 3],
            Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec(),
        ]

    q = np.asarray(q_seed, float).copy()
    for _ in range(iters):
        e = err(q)
        if np.linalg.norm(e[:3]) < tol_t and np.linalg.norm(e[3:]) < tol_r:
            return q, True
        J = np.stack([(e - err(q + h)) / 1e-6 for h in np.eye(6) * 1e-6], axis=1)
        q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
    return q, False


def line(q0, base_T_tool0_goal):
    """tool0 along a straight line (orientation slerped) from fk(q0) to the goal: IK from the previous waypoint
    every LINE_STEP_T / LINE_STEP_R. -> (joint path (n, 6) from q0, or None; reason)."""
    T0, T1 = fk(q0), np.asarray(base_T_tool0_goal, float)
    R = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    n = max(
        1,
        int(
            np.ceil(
                max(
                    np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / LINE_STEP_T,
                    (R[1] * R[0].inv()).magnitude() / LINE_STEP_R,
                )
            )
        ),
    )
    slerp, Q = Slerp([0, 1], R), [np.asarray(q0, float)]
    for k in range(1, n + 1):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = (
            slerp(k / n).as_matrix(),
            T0[:3, 3] + k / n * (T1[:3, 3] - T0[:3, 3]),
        )
        q, ok = ik(T, Q[-1])
        if not ok:
            return None, f"IK did not converge at waypoint {k}/{n}"
        if np.abs(q - Q[-1]).max() > MAX_JUMP:
            return (
                None,
                f"joint jump of {np.abs(q - Q[-1]).max():.2f} rad at waypoint {k}/{n}",
            )
        Q.append(q)
    return np.array(Q), ""
