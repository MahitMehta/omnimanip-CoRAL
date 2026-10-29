"""OmniManip execution on the UR5e (paper Sec. 3.3 closed-loop execution; supp. A.4 primitives; onboarding notes
O10) for a run_task run: the hardware counterpart of sim/execute.py, same primitives and loop.

  grasp  cuRobo move to 8 cm in front of the grasp pose (12 / 16 cm if refused), straight in, close, straight
         10 cm up (DEVIATIONS #33)
  pour   cuRobo move to the O9 target pose carrying the object; hold
  place  cuRobo move to v = |d| in front of the target along the passive axis, straight to the target, open
  The target is re-solved on every tracker update (Phi re-measured from the tracked active pose) until the EE is
  within 2 cm of it, and re-planned when it moves 1 cm / 3 deg.

Tracking: FoundationPose "track" on every synced camera frame (drop-if-busy) for the run's registered sessions
(omni_m<id>, made by register_object). Motion: the arm through the UR driver's FollowJointTrajectory action, the
gripper through gripper-control's /gripper/gripper_cmd (GripperCommand). Planning, IK and O9 run on the
omnimanip.server sidecar (cuRobo behind it); kinematics are the planner's URDF (assets/ur5e_robotiq.urdf).

--mock-tracking (dry runs on the UR driver's mock hardware and gripper-control's mock backend): no camera or
FoundationPose; the objects stay at their planned poses and the held one moves rigidly with the grasp.

Every motion stops at a gate first (Enter = go, anything else = stop; --gate first: only the first move of each
phase), its path latched just before on /omnimanip/trajectory and its tool0 positions on /omnimanip/ee_path
(nav_msgs/Path, base_link) for RViz. Ctrl-C during a motion cancels the trajectory goal; the e-stop stays with
the operator. Writes <run>/exec.json.

    ros2 run omnimanip_ros execute --run $RUNS_DIR/<stamp> [--speed 0.1] [--controller scaled_joint_trajectory_controller]
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PoseStamped
from manip_bridge import DEPTH_TOPIC, INFO_TOPIC, RGB_TOPIC
from manip_bridge.img import camera_info_to_K, image_to_depth_m, image_to_rgb
from manip_bridge.zmq_client import SidecarClient
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Path as NavPath
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
OMNI_ADDR = os.environ.get("OMNIMANIP_ADDR", "tcp://127.0.0.1:5690")
POSE_ADDR = os.environ.get("POSE_ADDR", "tcp://127.0.0.1:5667")
PRE_GRASP = (0.08, 0.12, 0.16)  # m; as sim/execute.py
LIFT = 0.10  # m straight up after closing (DEVIATIONS #33)
RESOLVE_STOP = 0.02  # m (O10)
REPLAN_T, REPLAN_R = 0.01, np.radians(3)
CHUNK = 8  # planned waypoints sent per tracker update
MASK_GROW = 5  # px: held object's mask grown before it leaves the map
DWELL_S = 2.0  # held at the pour pose
OPEN_GAP, CLOSE_GAP = (
    0.085,
    0.0,
)  # m, Robotiq 2F-85 (gripper-control: position = jaw gap)
VMAX = np.pi  # rad/s and rad/s^2 per UR5e joint at --speed 1
START_TOL, ARRIVE_TOL = 0.01, 0.01  # rad
TRACK_MAX_AGE = 1.0  # s: an older tracked pose is not used


class Stop(Exception):
    """The operator declined a gate, or a check failed; nothing further moves."""


class Executor(Node):
    def __init__(self, keys, controller):
        super().__init__("omnimanip_execute")
        self.q, self.qlock = None, threading.Lock()
        self.poses, self.plock, self.busy = {}, threading.Lock(), False
        self.keys, self.info = keys, None
        self.pose_cli = SidecarClient(POSE_ADDR, 30_000)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(
            CameraInfo, INFO_TOPIC, self._on_info, qos_profile_sensor_data
        )
        track_group = MutuallyExclusiveCallbackGroup()
        self.sync = ApproximateTimeSynchronizer(
            [
                Subscriber(
                    self,
                    Image,
                    RGB_TOPIC,
                    qos_profile=qos_profile_sensor_data,
                    callback_group=track_group,
                ),
                Subscriber(
                    self,
                    Image,
                    DEPTH_TOPIC,
                    qos_profile=qos_profile_sensor_data,
                    callback_group=track_group,
                ),
            ],
            queue_size=2,
            slop=0.034,
        )
        self.sync.registerCallback(self._on_frame)
        self.traj = ActionClient(
            self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory"
        )
        self.grip = ActionClient(self, GripperCommand, "/gripper/gripper_cmd")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_traj = self.create_publisher(
            JointTrajectory, "/omnimanip/trajectory", latched
        )
        self.pub_ee = self.create_publisher(NavPath, "/omnimanip/ee_path", latched)

    def _on_joints(self, msg):
        if all(n in msg.name for n in UR_JOINTS):
            with self.qlock:
                self.q = np.array(
                    [msg.position[list(msg.name).index(n)] for n in UR_JOINTS]
                )

    def _on_info(self, msg):
        self.info = msg

    def _on_frame(self, rgb, depth):
        """FoundationPose track for every session on this frame (frames arriving while busy are dropped)."""
        if self.busy or self.info is None:
            return
        self.busy = True
        try:
            im, d, K = (
                image_to_rgb(rgb),
                image_to_depth_m(depth),
                camera_info_to_K(self.info),
            )
            for k in self.keys:
                rep = self.pose_cli.call(
                    {
                        "cmd": "track",
                        "obj": k,
                        "rgb": im,
                        "depth": d,
                        "K": K,
                        "track_refine_iter": 2,
                    }
                )
                with self.plock:
                    self.poses[k] = (time.time(), np.asarray(rep["pose"], float))
        except Exception as e:  # noqa: BLE001 -- a lost frame; staleness is checked where poses are used
            self.get_logger().warn(f"track: {e}")
        finally:
            self.busy = False

    def joints(self):
        with self.qlock:
            if self.q is None:
                raise Stop("no /joint_states")
            return self.q.copy()

    def pose(self, key):
        """Latest tracked cam_T_obj (body frame), waiting up to 5 s for a fresh one."""
        t0 = time.time()
        while time.time() - t0 < 5.0:
            with self.plock:
                p = self.poses.get(key)
            if p is not None and time.time() - p[0] < TRACK_MAX_AGE:
                return p[1]
            time.sleep(0.05)
        raise Stop(f"no fresh tracked pose for {key}")

    def _wait(self, fut, what):
        while not fut.done():
            time.sleep(0.01)
        return fut.result()

    def timed(self, Q, speed):
        """Joint path Q (UR order) timed by a minimum-jerk law over its joint-space length, peak joint speed and
        acceleration speed * VMAX; None when Q does not move."""
        L = np.r_[0.0, np.cumsum(np.abs(np.diff(Q, axis=0)).max(1))]
        if L[-1] < 1e-4:
            return None
        v = speed * VMAX
        T = max(
            1.875 * L[-1] / v, np.sqrt(5.774 * L[-1] / v), 0.5
        )  # min-jerk peak speed 1.875 L/T, accel 5.774 L/T^2
        tau = np.linspace(0, 1, 2001)
        t = np.interp(L / L[-1], 10 * tau**3 - 15 * tau**4 + 6 * tau**5, tau) * T
        msg, last = JointTrajectory(joint_names=UR_JOINTS), 0.0
        for i, (qi, ti) in enumerate(zip(Q[1:], t[1:]), 1):
            if (
                i < len(Q) - 1 and ti < last + 1e-3
            ):  # repeated waypoints: times must strictly increase
                continue
            last = ti = max(ti, last + 1e-3)
            msg.points.append(
                JointTrajectoryPoint(
                    positions=[float(x) for x in qi],
                    time_from_start=Duration(seconds=float(ti)).to_msg(),
                )
            )
        return msg

    def show(self, Q, ee, speed):
        """Latch a path before its gate: /omnimanip/trajectory as it will be sent, /omnimanip/ee_path its tool0
        positions (RViz: a Path display, fixed frame base_link)."""
        hdr = Header(frame_id="base_link", stamp=self.get_clock().now().to_msg())
        msg = self.timed(Q, speed) or JointTrajectory(joint_names=UR_JOINTS)
        msg.header = hdr
        self.pub_traj.publish(msg)
        path = NavPath(header=hdr)
        for p in ee:
            ps = PoseStamped(header=hdr)
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_ee.publish(path)

    def send(self, Q, speed):
        """Track joint path Q (Q[0] = where the robot is), timed as in timed(). Checks start and arrival."""
        q = self.joints()
        if np.abs(Q[0] - q).max() > START_TOL:
            raise Stop(
                f"path starts {np.abs(Q[0] - q).max():.3f} rad from the robot (stale plan)"
            )
        msg = self.timed(Q, speed)
        if msg is None:
            return
        if not self.traj.wait_for_server(timeout_sec=5.0):
            raise Stop(
                "FollowJointTrajectory server not available (UR driver up? --controller?)"
            )
        gh = self._wait(
            self.traj.send_goal_async(FollowJointTrajectory.Goal(trajectory=msg)),
            "goal",
        )
        if not gh.accepted:
            raise Stop("trajectory goal rejected")
        try:
            res = self._wait(gh.get_result_async(), "result").result
        except KeyboardInterrupt:  # stop the arm before exiting: wait (2 s at most) for the driver to take the cancel
            fut, t_end = gh.cancel_goal_async(), time.time() + 2.0
            while not fut.done() and time.time() < t_end:
                time.sleep(0.01)
            print("exec: trajectory cancel " + ("acknowledged" if fut.done() else "NOT acknowledged -- use the e-stop"))
            raise
        if res.error_code != 0:
            raise Stop(
                f"trajectory failed: error_code {res.error_code} {res.error_string}"
            )
        if np.abs(self.joints() - Q[-1]).max() > ARRIVE_TOL:
            raise Stop(
                f"arrived {np.abs(self.joints() - Q[-1]).max():.3f} rad from the path end"
            )

    def gripper(self, gap, effort):
        if not self.grip.wait_for_server(timeout_sec=5.0):
            raise Stop(
                "/gripper/gripper_cmd not available (gripper-control up on the laptop?)"
            )
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = float(gap), float(effort)
        gh = self._wait(self.grip.send_goal_async(goal), "goal")
        if not gh.accepted:
            raise Stop("gripper goal rejected")
        r = self._wait(gh.get_result_async(), "result").result
        return {
            "position": r.position,
            "stalled": r.stalled,
            "reached_goal": r.reached_goal,
        }


def gate(a, what, first=True):
    if (a.gate == "all" or (a.gate == "first" and first)) and input(
        f"[gate] {what} -- Enter to go, anything else stops: "
    ).strip():
        raise Stop(f"stopped at gate: {what}")


def run(node, a, log):
    run = Path(a.run)
    objects = json.loads((run / "objects.json").read_text())
    grasp = json.loads((run / "grasp" / "grasp.json").read_text())[0]
    stage = next(
        s for s in json.loads((run / "rrc" / "rrc.json").read_text()) if s["accepted"]
    )
    packet = json.loads((run / "packet.json").read_text())
    c = {
        k: np.asarray(v) if isinstance(v, list) else v
        for k, v in stage["accepted"].items()
    }
    A, Pv, ACT = stage["active"], stage["passive"], stage["action"]
    if a.distance_cm is not None and c["distance_m"] == 0:
        c["distance_m"] = a.distance_cm / 100
    B_C = np.asarray(packet["t_base_cam"]["T"], float)
    C_B, K = np.linalg.inv(B_C), np.asarray(packet["K"], float)
    jn = packet["joints"]["names"]
    q_cap = np.array([packet["joints"]["q"][jn.index(n)] for n in UR_JOINTS])
    depth0 = (
        cv2.imread(str(run / "depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        / 1000
    )
    masks = {
        n: cv2.imread(str(run / "marks" / f"mask_{o['mark']}.png"), 0) > 127
        for n, o in objects.items()
    }
    Rc = {
        n: np.asarray(o["canonical_frame"]["R_body_from_canon"], float)
        for n, o in objects.items()
    }
    omni = SidecarClient(OMNI_ADDR, 300_000)
    call = lambda cmd, **kw: omni.call({"cmd": cmd, **kw})

    TG = np.asarray(
        call("kinematics", q=list(q_cap), depth=grasp["depth"])["tool0_T_grasp"], float
    )
    fk = lambda q: np.asarray(
        call("kinematics", q=list(q), depth=grasp["depth"])["base_T_tool0"], float
    )
    ee_cam = lambda: C_B @ fk(node.joints()) @ TG
    tool0 = lambda T_cam: B_C @ T_cam @ np.linalg.inv(TG)

    held = []  # set once the gripper holds the active object

    def tracked(name):
        if a.mock_tracking:  # dry run: planned poses; the held object rigid in the grasp (the grasp's Phi)
            if name == A and held:
                return ee_cam() @ np.asarray(grasp["phi_grasp_T_obj"], float)
            return np.asarray(objects[name]["cam_T_can"], float)
        T = node.pose(f"omni_m{objects[name]['mark']}").copy()
        T[:3, :3] = T[:3, :3] @ Rc[name]
        return T

    show = lambda Q: node.show(Q, [fk(q)[:3, 3] for q in Q], a.speed)

    def planned(what, goal, first, **kw):
        r = call("plan", q=list(node.joints()), goal=goal, **kw)
        if r["path"] is None:
            return None, r["reason"]
        Q = np.asarray(r["path"], float)
        show(Q)
        gate(a, f"{what}: planned path of {len(Q)} waypoints", first)
        return Q, ""

    def straight(what, goal, gripper_note=""):
        r = call("line", q=list(node.joints()), goal=goal)
        if r["path"] is None:
            raise Stop(f"{what}: {r['reason']}")
        Q = np.asarray(r["path"], float)
        show(Q)
        gate(a, f"{what}: straight tool move{gripper_note}")
        node.send(Q, a.speed)

    # grasp primitive (supp. A.4 (a))
    call("map", depth=depth0, K=K, T_base_cam=B_C, q=list(q_cap), plane=packet["plane"])
    gate(a, "open the gripper")
    log["grasp"]["open"] = node.gripper(OPEN_GAP, a.effort)
    T_g = np.asarray(grasp["cam_T_grasp"], float)
    for pre in PRE_GRASP:
        T_pre = T_g.copy()
        T_pre[:3, 3] -= pre * T_g[:3, 0]
        path, why = planned(
            f"pre-grasp {pre * 100:.0f} cm",
            tool0(T_pre),
            True,
            q_goal_ref=list(node.joints()),
        )
        if path is not None:
            break
        log["grasp"].setdefault("refused", []).append(
            {"pre_grasp_m": pre, "reason": why}
        )
    if path is None:
        raise Stop(f"no plan to the pre-grasp: {why}")
    log["grasp"]["pre_grasp_m"] = pre
    node.send(path, a.speed)
    straight("approach", tool0(T_g))
    gate(a, "close the gripper")
    log["grasp"]["close"] = r = node.gripper(CLOSE_GAP, a.effort)
    if not r["stalled"] and not a.skip_grip_check:
        raise Stop(
            f"gripper closed without an object (gap {r['position'] * 1000:.1f} mm)"
        )
    held.append(A)
    T_up = ee_cam()
    T_up[:3, 3] += LIFT * C_B[:3, 2]  # base +z in the camera frame
    straight("lift", tool0(T_up))

    # the stage's primitive, closed loop; the held object leaves the map and rides on tool0 as spheres
    grown = (
        cv2.dilate(
            masks[A].astype(np.uint8), np.ones((2 * MASK_GROW + 1,) * 2, np.uint8)
        )
        > 0
    )
    call(
        "map", depth=depth0, K=K, T_base_cam=B_C, q=list(q_cap), exclude=grown, plane=packet["plane"]
    )
    exclude = np.any(list(masks.values()), axis=0)
    mesh = objects[A]["final_mesh"]

    def approach(T_t, T_p):
        T = T_t.copy()
        if ACT == "place":
            T[:3, 3] -= abs(c["distance_m"]) * (T_p[:3, :3] @ c["passive_axis"])
        return T

    T_t, T_plan, solving, path, k, first = None, np.eye(4), True, None, 0, True
    T_start = ee_cam()  # O9's path cost is measured from the stage's start pose
    log[ACT] = {"updates": [], "plans": []}
    for it in range(200):
        T_a, T_p, T_e = tracked(A), tracked(Pv), ee_cam()
        if solving:
            phi = np.linalg.inv(T_e) @ T_a
            r = call(
                "targetpose",
                depth=depth0,
                K=K,
                obstacle_exclude=exclude,
                constraint=c,
                phi=phi,
                cam_T_ee_start=T_start,
                cam_T_passive=T_p,
                width=grasp["width"],
                x0=T_t,
                overlay=False,
            )
            T_t = np.asarray(r["cam_T_ee_target"], float)
            solving = np.linalg.norm(T_t[:3, 3] - T_e[:3, 3]) > RESOLVE_STOP
            log[ACT]["updates"].append(
                {
                    "it": it,
                    "ok": bool(r["ok"]),
                    **{k_: float(v) for k_, v in r["terms"].items()},
                }
            )
            moved = path is None or (
                np.linalg.norm(T_t[:3, 3] - T_plan[:3, 3]) > REPLAN_T
                or Rotation.from_matrix(T_t[:3, :3] @ T_plan[:3, :3].T).magnitude()
                > REPLAN_R
            )
            if moved:
                ref = list(
                    node.joints() if path is None else path[-1]
                )  # stay on the previous plan's IK branch
                T_body = T_a.copy()  # the mesh is in the body frame: canonical -> body
                T_body[:3, :3] = T_a[:3, :3] @ Rc[A].T
                carry = {
                    "mesh_path": mesh,
                    "tool0_T_obj": np.linalg.inv(fk(node.joints())) @ B_C @ T_body,
                }
                new, why = planned(
                    f"{ACT} (update {it})",
                    tool0(approach(T_t, T_p)),
                    first,
                    held=carry,
                    q_goal_ref=ref,
                )
                log[ACT]["plans"].append(
                    {"it": it, "success": new is not None, "reason": why}
                )
                first = False
                if (
                    new is None and path is not None
                ):  # e.g. inside the passive object's buffer near the end
                    T_t, solving = T_plan, False
                elif (path := new) is None:
                    raise Stop(f"no plan at update {it}: {why}")
                else:
                    T_plan, k = T_t, 0
        if k >= len(path) - 1:
            break
        node.send(path[k : k + CHUNK + 1], a.speed)
        k += CHUNK
    if ACT == "place":  # straight to the target, release (supp. A.4 (b))
        straight("place", tool0(T_t))
        gate(a, "open the gripper (release)")
        log[ACT]["release"] = node.gripper(OPEN_GAP, a.effort)
    else:
        time.sleep(DWELL_S)
    T_a, T_p = tracked(A), tracked(Pv)
    log["result"] = {
        "cam_T_active_tracked_final": T_a,
        "cam_T_passive_tracked_final": T_p,
        "cam_T_ee_target": T_t,
        "cam_T_ee_final": ee_cam(),
        "iterations": it + 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="fraction of the UR5e joint limits (pi rad/s, rad/s^2)",
    )
    ap.add_argument(
        "--effort",
        type=float,
        default=60.0,
        help="gripper force, N (gripper-control's example)",
    )
    ap.add_argument("--controller", default="scaled_joint_trajectory_controller")
    ap.add_argument("--gate", choices=("all", "first", "none"), default="all")
    ap.add_argument(
        "--skip-grip-check",
        action="store_true",
        help="go on after the close whatever the gripper reports (gripper-control's URCap reply can be one move "
        "behind); the operator checks the grasp at the lift's gate",
    )
    ap.add_argument(
        "--mock-tracking",
        action="store_true",
        help="DRY RUN: no FoundationPose; objects at their planned poses, the held one rigid in the grasp",
    )
    ap.add_argument(
        "--distance-cm",
        type=float,
        default=None,
        help="TEST ONLY (as sim/execute.py): d to use when the VLM's is 0; negative = against the passive axis",
    )
    a, ros_args = ap.parse_known_args()
    objects = json.loads((Path(a.run) / "objects.json").read_text())
    rclpy.init(args=ros_args)
    keys = [] if a.mock_tracking else [f"omni_m{o['mark']}" for o in objects.values()]
    node = Executor(keys, a.controller)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    log = {"args": vars(a), "status": "error (see the traceback)", "grasp": {}}
    try:
        run(node, a, log)
        log["status"] = "completed"
    except Stop as e:
        log["status"] = f"stopped: {e}"
    except KeyboardInterrupt:
        log["status"] = "interrupted"
    finally:
        (Path(a.run) / "exec.json").write_text(
            json.dumps(
                log,
                indent=1,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
            )
        )
        print("exec:", log["status"])
        ex.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
