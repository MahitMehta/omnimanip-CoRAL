"""ReKep's closed loop (paper Alg. 1, Sec. 3.3; A.2, A.7–A.9): its released main loop (main.Main._execute: stages,
backtracking, sub-goal and path solves, `action_steps_per_iter` actions between solves) on an arm server (the
sim's sim/rekep_arm.py or the robot's omnimanip_ros rekep_arm), with our environment: tracked keypoints (rekep/
tracker.py at up to 20 Hz), the scene SDF from the latest frame, the gripper and held object's points, and the
grasp stage's sub-goal from AnyGrasp (the grasp closest to the grasp keypoint, paper footnote 1).

Arm server protocol (msgpack, REQ/REP; frames on port + 1 so they keep coming during a move):
    {"cmd": "capture"}                  -> {"rgb", "depth", "K", "packet", "robot_mask"}   the planning frame packet
    {"cmd": "frame"}       (port + 1)   -> {"rgb", "depth", "q", "t"[, "robot_mask"]}       the latest frame
    {"cmd": "joints"}                   -> {"q"}
    {"cmd": "move", "path", "speed"}    -> {"ok", "q"}      a joint path (path[0] = where the arm is), to its end
    {"cmd": "gripper", "close"}         -> {"ok", "grasped"}
"""

import json
import signal
import sys
import threading
import time
import types

import cv2
import numpy as np
from manip_bridge.zmq_client import SidecarClient, SidecarError
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from omnimanip import grasp as AG
from omnimanip import motion

from . import core as C
from .tracker import Tracker

# ReKep's main.py imports its OmniGibson environment; only its loop is used here
for _m, _cls in (
    ("environment", "ReKepOGEnv"),
    ("omnigibson", None),
    ("omnigibson.robots", None),
    ("omnigibson.robots.fetch", "Fetch"),
    ("ik_solver", "IKSolver"),
    ("visualizer", "Visualizer"),
):
    _mod = sys.modules.setdefault(_m, types.ModuleType(_m))
    if _cls:
        setattr(_mod, _cls, object)
from main import Main
from path_solver import PathSolver
from subgoal_solver import SubgoalSolver
from utils import get_linear_interpolation_steps, linear_interpolate_poses

STEP_T, STEP_R = (
    0.005,
    np.radians(1.0),
)  # paper A.2: interpolation to each target pose, 5 mm and 1 degree
TRACK_HZ = 20  # paper A.7
HELD_CLEAR = 0.02  # m: scene points this close to the held object's points leave the SDF (rekep/DEVIATIONS.md)
GRASPED_GAP = 0.002  # m: a close that stops wider than this holds something


class Arm:
    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.cmd = SidecarClient(addr, 600_000)
        self.frames = SidecarClient(f"{host}:{int(port) + 1}", 10_000)

    def __call__(self, cmd, **kw):
        return self.cmd.call({"cmd": cmd, **kw})

    def frame(self):
        return self.frames.call({"cmd": "frame"})


class Env:
    """ReKep's environment interface (environment.ReKepOGEnv's methods main.Main uses) on an arm server."""

    def __init__(
        self, arm, cfg, packet, masks, pixels, proposer, log, speed, cap_depth
    ):
        self.speed, self.cap_depth, self.grasp_keypoint = speed, cap_depth, -1
        self.table_z = cfg["path_solver"]["table_z"]
        self.arm, self.cfg, self.log = arm, cfg, log
        self.K, self.T = np.array(packet["K"]), np.array(packet["t_base_cam"]["T"])
        self.lo, self.hi = (
            np.array(cfg["main"]["bounds_min"]),
            np.array(cfg["main"]["bounds_max"]),
        )
        self.masks, self.pixels = masks, np.asarray(pixels)
        self.tracker, self.step_counter, self.held = (
            Tracker(proposer, self.K, self.T),
            0,
            None,
        )
        self.curobo = motion.cli(30)
        try:  # the sim's fingertips sit short of the planner model's (sim/rekep_arm.py); the robot's don't
            C.set_tip_offset(float(arm("info").get("tip_offset", 0.0)))
        except SidecarError:
            C.set_tip_offset(0.0)
        self.ik = C.IK(self.q())
        self.reset_joint_pos = self.q()

    # --- perception: tracked keypoints at up to TRACK_HZ, on a thread
    def register_keypoints(self, keypoints):
        self.kp_obj = [
            C.object_id(self.masks, p, self.cfg["keypoint_proposer"]["max_mask_ratio"])
            for p in self.pixels
        ]
        f = self.arm.frame()
        self.last, self.kps = f, self.tracker.start(f["rgb"], f["depth"], keypoints)
        self.kps0 = self.kps.copy()
        self.stop = threading.Event()
        threading.Thread(target=self._track, daemon=True).start()

    def _track(self):
        while not self.stop.is_set():
            t0 = time.time()
            try:
                f = self.arm.frame()
                self.kps, self.last = self.tracker.update(f["rgb"], f["depth"]), f
            except Exception as e:  # noqa: BLE001 -- a lost frame; the last keypoints stand
                self.log["events"].append({"t": time.time(), "track_error": str(e)})
            time.sleep(max(0.0, 1 / TRACK_HZ - (time.time() - t0)))

    def get_keypoint_positions(self):
        return self.kps.copy()

    # --- robot state
    def q(self):
        return np.asarray(self.arm("joints")["q"], float)

    def get_arm_joint_postions(self):
        return self.q()

    def get_ee_pose(self):
        return C.ee_pose(self.q())

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    # --- collision geometry (paper A.8): the scene SDF without the arm and the held object; gripper + held points
    def _robot_mask(self, f):
        if "robot_mask" in f:
            return np.asarray(f["robot_mask"], bool)
        rep = self.curobo.call(
            {
                "cmd": "robot_mask",
                "depth": np.asarray(f["depth"], np.float32),
                "intrinsics": self.K.astype(np.float32),
                "T_base_cam": self.T.astype(np.float32),
                "q": [float(x) for x in f["q"]],
                "joint_names": motion.UR_JOINTS,
            }
        )
        return np.asarray(rep["mask"], bool)

    def _held_points(self):
        if self.held is None:
            return np.zeros((0, 3))
        E = C.mat(self.get_ee_pose())
        return self.held[1] @ E[:3, :3].T + E[:3, 3]

    def get_sdf_voxels(self, res):
        f = self.last
        P = C.points(f["depth"], self.K, self.T)[
            (f["depth"] > 0) & ~self._robot_mask(f)
        ]
        H = self._held_points()
        if len(H):
            P = P[KDTree(H).query(P, distance_upper_bound=HELD_CLEAR)[0] > HELD_CLEAR]
        return C.sdf_voxels(P, self.lo, self.hi, res)

    def get_collision_points(self):
        return np.vstack([C.gripper_points(self.get_ee_pose()), self._held_points()])

    # --- objects (rigid groups, rekep/DEVIATIONS.md) and grasping
    def get_object_by_keypoint(self, i):
        return self.kp_obj[i]

    def is_grasping(self, candidate_obj=None):
        return self.held is not None and candidate_obj in (None, self.held[0])

    # --- actions (paper A.2): clip to the workspace, interpolate 5 mm / 1 deg, IK from the current joints, run
    def execute_action(self, action, precise=True):
        target = np.asarray(action[:7], float).copy()
        target[:3] = np.clip(target[:3], self.lo, self.hi)
        q, cur = self.q(), self.get_ee_pose()
        n = get_linear_interpolation_steps(cur, target, STEP_T, STEP_R)
        Q = [q]
        for pose in linear_interpolate_poses(cur, target, max(n, 2))[1:]:
            r = self.ik.solve(C.mat(pose), 100, Q[-1])
            if not r.success:
                self.log["events"].append(
                    {"t": time.time(), "ik_failed_at": pose.tolist()}
                )
                break
            Q.append(r.cspace_position)
        if len(Q) > 1:
            rep = self.arm("move", path=np.array(Q), speed=self.speed)
            if not rep["ok"]:
                raise RuntimeError(f"move failed: {rep}")
        self.step_counter += 1
        if action[7] == self.get_gripper_close_action():
            self.close_gripper()
        elif action[7] == self.get_gripper_open_action():
            self.open_gripper()

    def close_gripper(self):
        rep = self.arm("gripper", close=True)
        self.log["events"].append({"t": time.time(), "close": rep})
        if rep[
            "grasped"
        ]:  # the grasp keypoint's object, its capture points moved with its tracked keypoints
            obj = self.kp_obj[self.grasp_keypoint]
            on = [i for i, o in enumerate(self.kp_obj) if o == obj]
            P = C.points(self.cap_depth, self.K, self.T)[
                self.masks[obj] & (self.cap_depth > 0)
            ]
            P = P + (self.kps[on] - self.kps0[on]).mean(0)
            E = np.linalg.inv(C.mat(self.get_ee_pose()))
            self.held = (obj, P @ E[:3, :3].T + E[:3, 3])

    def open_gripper(self):
        self.log["events"].append(
            {"t": time.time(), "open": self.arm("gripper", close=False)}
        )
        self.held = None

    def get_gripper_open_action(self):
        return -1.0

    def get_gripper_close_action(self):
        return 1.0

    def get_gripper_null_action(self):
        return 0.0

    def sleep(self, s):
        time.sleep(s)

    def save_video(self):
        self.stop.set()


class Loop(Main):
    """main.Main with our environment and solvers; the grasp stage's sub-goal from AnyGrasp."""

    def __init__(self, env, cfg):
        self.config, self.env, self.visualize = cfg["main"], env, False
        self.subgoal_solver = SubgoalSolver(
            cfg["subgoal_solver"], env.ik, env.reset_joint_pos
        )
        self.path_solver = PathSolver(cfg["path_solver"], env.ik, env.reset_joint_pos)
        self.grasp = None

    def _update_stage(self, stage):
        super()._update_stage(stage)
        self.grasp = None
        self.env.grasp_keypoint = self.program_info["grasp_keypoints"][stage - 1]
        self.env.log["stages"].append({"t": time.time(), "stage": stage})

    def _get_next_subgoal(self, from_scratch):
        t0 = time.time()
        if not self.is_grasp_stage:
            pose = super()._get_next_subgoal(from_scratch)
        else:
            k = self.keypoints[1 + self.env.grasp_keypoint]
            if self.grasp is None:
                self.grasp = (choose_grasp(self.env, k), k)
            T = self.grasp[0].copy()
            T[:3, 3] += k - self.grasp[1]  # the grasp rides with its keypoint
            # main executes a grasp stage by reaching the sub-goal, then moving grasp_depth along its approach: the
            # sub-goal is the grasp grasp_depth back, so the fingertips close where AnyGrasp put them
            T[:3, 3] -= self.config["grasp_depth"] * T[:3, 0]
            pose = np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]
        self.env.log["solves"].append(
            {
                "t": t0,
                "stage": self.stage,
                "kind": "subgoal",
                "s": time.time() - t0,
                "from_scratch": from_scratch,
                "pose": pose.tolist(),
            }
        )
        return pose

    def _get_next_path(self, next_subgoal, from_scratch):
        t0 = time.time()
        path = super()._get_next_path(next_subgoal, from_scratch)
        self.env.log["solves"].append(
            {
                "t": t0,
                "stage": self.stage,
                "kind": "path",
                "s": time.time() - t0,
                "from_scratch": from_scratch,
                "n": len(path),
            }
        )
        return path


def choose_grasp(env, k, radius=0.10):
    """Paper footnote 1: AnyGrasp's grasp closest to the grasp keypoint k (its fingertips' centre), among those IK
    reaches, at least OmniManip's MIN_DEPTH deep, that face down as the sub-goal's rotation bounds require (A.8:
    approach z <= 0), and whose open gripper
    stays above the table plane (the push-in to a grasp is not collision-checked). AnyGrasp sees the
    latest frame's points within `radius` of k, without the arm, and returns its detections (after its NMS and
    scoring; its dense mode's every proposal includes millimetre-deep, near-zero-score ones). -> base_T_grasp (AnyGrasp frame: x approach,
    fingertips at the origin)."""
    f = env.last
    P = C.points(f["depth"], env.K, env.T)
    near = (
        (np.linalg.norm(P - k, axis=-1) < radius)
        & (f["depth"] > 0)
        & ~env._robot_mask(f)
    )
    rep = AG.anygrasp(
        f["rgb"], f["depth"], env.K, near, dense=False
    )  # its detector output, not every proposal
    tips = [env.T @ AG.pose(rep, i) @ _tips(rep, i) for i in range(rep["n"])]
    for i in np.argsort([np.linalg.norm(T[:3, 3] - k) for T in tips]):
        T = tips[i]
        z = C.gripper_points(
            np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]
        )[:, 2]
        deep = (
            rep["depths"][i] >= AG.MIN_DEPTH - 1e-6
        )  # OmniManip's real-gripper rule (its stage 15)
        if (
            deep
            and T[2, 0] <= 0
            and z.min() >= env.table_z
            and env.ik.solve(T, 100).success
        ):
            env.log["grasp"] = {
                "base_T_grasp": T.tolist(),
                "depth_m": float(rep["depths"][i]),
                "width_m": float(rep["widths"][i]),
                "score": float(rep["scores"][i]),
                "keypoint": k.tolist(),
                "index": int(i),
                "n": int(rep["n"]),
                "dist_m": float(np.linalg.norm(tips[i][:3, 3] - k)),
            }
            return tips[i]
    raise RuntimeError(
        f"none of AnyGrasp's {rep['n']} grasps near the keypoint is deep enough, faces down, clears the table and is reachable"
    )


def _tips(rep, i):
    T = np.eye(4)
    T[0, 3] = float(
        rep["depths"][i]
    )  # AnyGrasp's fingertips sit `depth` along the approach from its grasp point
    return T


def execute(d, a, cfg, packet, masks, kp, proposer):
    """Run the program in d on the arm server a.arm (SIDECARS["rekep_arm"]); writes exec.json."""
    log = {
        "args": {k: str(v) for k, v in vars(a).items()},
        "events": [],
        "stages": [],
        "solves": [],
    }
    depth = cv2.imread(str(d / "depth_mm.png"), cv2.IMREAD_UNCHANGED)
    assert depth is not None, d / "depth_mm.png"
    env = Env(
        Arm(a.arm),
        cfg,
        packet,
        masks,
        kp["pixels"],
        proposer,
        log,
        a.speed,
        depth.astype(np.float32) / 1000,
    )

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)  # as Ctrl-C: exec.json is still written
    try:
        Loop(env, cfg)._execute(str(d / "program"))
        log["status"] = "completed"
    except KeyboardInterrupt:
        log["status"] = "interrupted"
    except Exception as e:
        log["status"] = f"error: {e}"
        raise
    finally:
        env.save_video()
        (d / "exec.json").write_text(
            json.dumps(log, indent=1, default=lambda x: np.asarray(x).tolist())
        )
        print("exec:", log["status"])
