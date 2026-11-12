"""OmniManip execution on the UR5e (paper Sec. 3.3 closed-loop execution; supp. A.4 primitives; onboarding notes
O10) for a run_task run: the hardware counterpart of sim/execute.py, same primitives.

  grasp  cuRobo move to 8 cm in front of the grasp pose (12 / 16 cm if refused), straight in, close, straight
         10 cm up (DEVIATIONS #33)
  pour   cuRobo move to the O9 target pose carrying the object; hold
  place  cuRobo move to max(v, 4 cm) in front of the target along the passive axis (v = |d|), straight to the target,
         open
  The planned path goes to the controller as one goal, and the target is re-solved on every tracker update while the
  arm moves (Phi re-measured from the tracked active pose and the joints at its frame's arrival) until the EE is within
  2 cm of it. When the target moves 2 cm / 5 deg from the plan's, a replan is tried without stopping; if it can be
  planned the goal is cancelled and the arm re-plans from rest, and if not the current goal runs on.

Tracking: FoundationPose "track" on every synced camera frame (drop-if-busy) for the run's registered sessions
(omni_m<id>, made by register_object). With no fresh pose, execution goes on with a warning: the held object rides on
the gripper at its last measured grasp, others stay at their last pose; both are logged as track_events. Motion: the arm through the UR driver's FollowJointTrajectory action, the
gripper through gripper-control's /gripper/gripper_cmd (GripperCommand). Planning, IK and O9 run on the
omnimanip.server sidecar (cuRobo behind it); kinematics are the planner's URDF (assets/ur5e_robotiq.urdf).

--no-tracking: no camera or FoundationPose; the objects stay at their planned poses and the held one moves rigidly
with the grasp, at the plan's Phi, and the stage solves O9 and plans once, with no re-solves or re-plans. The arm and gripper still move: for an object placed by hand, which track can't
follow, or for dry runs on the UR driver's mock hardware and gripper-control's mock backend.

Every motion stops at a gate first (Enter = go, anything else = stop; --gate first: only the first move of each
phase), its path latched just before on /omnimanip/trajectory and its tool0 positions on /omnimanip/ee_path
(nav_msgs/Path, base_link) for RViz. Ctrl-C during a motion cancels the trajectory goal; the e-stop stays with
the operator. Writes <run>/exec.json, and records camera2's colour stream for the whole session to
<run>/videos/exec_<start time>_camera2.webm (VP8; a new file per execute; none when camera2 isn't publishing).

    ros2 run omnimanip_ros execute --run $RUNS_DIR/<stamp> [--speed 0.1] [--controller scaled_joint_trajectory_controller]
"""

import argparse
import json
import os
import threading
import time
from collections import deque
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
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
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
PRE_PLACE = 0.04  # m: place's planned move ends at least this far before the target; cuRobo's margins refused a
# marker 1.6 cm into a cup at v = 0 (run 20261111_221720); the straight move in isn't collision-checked
LIFT = 0.10  # m straight up after closing (DEVIATIONS #33)
RESOLVE_STOP = 0.02  # m (O10)
REPLAN_T, REPLAN_R = 0.02, np.radians(5)
IK_ALTERNATIVES = 5  # distinct IK solutions the stage's first plan tries, nearest first, before cuRobo chooses one
MASK_GROW = 10  # px at 480 rows: held object's mask grown before it leaves the map (depth bleeds past its edge)
DWELL_S = 2.0  # held at the pour pose
OPEN_GAP, CLOSE_GAP = (
    0.085,
    0.0,
)  # m, Robotiq 2F-85 (gripper-control: position = jaw gap)
VMAX = np.pi  # rad/s and rad/s^2 per UR5e joint at --speed 1
START_TOL, ARRIVE_TOL = 0.01, 0.01  # rad
TRACK_MAX_AGE = 1.0  # s: an older tracked pose is not used
TRACK_WAIT, STALE_WAIT = (
    5.0,
    0.2,
)  # s: wait for a fresh pose; the same once that object is known to be stale
REC_TOPIC, REC_FPS = (
    "/camera/camera2/color/image_raw/compressed",
    30,
)  # session video: the second camera, JPEG frames
TRACK_SLOW = 1.0  # s: a longer gap between synced frames, or track call, is logged
JOINT_HISTORY = 2000  # /joint_states samples kept, to pair a tracked pose with the joints at its frame's arrival
CANCEL_WAIT, SETTLE_TOL = (
    2.0,
    1e-3,
)  # s for the driver to take a cancel, and then for the arm to stop; rad per 50 ms


class Recorder:
    """REC_TOPIC to a VP8 webm (plays in VS Code and browsers; mp4v didn't, and this OpenCV has no H.264) at a fixed
    REC_FPS by message time: a frame is repeated across a gap (up to 1 s), so the video plays at real speed. The file is
    opened on the first frame."""

    def __init__(self, node, path):
        self.node, self.path, self.w, self.t0, self.n, self.lock = (
            node,
            path,
            None,
            None,
            0,
            threading.Lock(),
        )
        self.closed = False
        self.sub = node.create_subscription(
            CompressedImage,
            REC_TOPIC,
            self._on,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _on(self, msg):
        im = cv2.imdecode(
            np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
        )  # BGR, as VideoWriter wants
        if im is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            if (
                self.closed
            ):  # a frame after close() would reopen and truncate the finished file
                return
            if self.w is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.w = cv2.VideoWriter(
                    str(self.path),
                    cv2.VideoWriter_fourcc(*"VP80"),
                    REC_FPS,
                    (im.shape[1], im.shape[0]),
                )
                self.t0 = t
            for _ in range(min(int((t - self.t0) * REC_FPS) + 1 - self.n, REC_FPS)):
                self.w.write(im)
                self.n += 1

    def close(self):
        """-> the video's path, or None if no frame came."""
        with self.lock:
            self.closed = True
            self.node.destroy_subscription(self.sub)
            if self.w is None:
                return None
            self.w.release()
            return self.path


class Stop(Exception):
    """The operator declined a gate, or a check failed; nothing further moves."""


class Executor(Node):
    def __init__(self, keys, controller):
        super().__init__("omnimanip_execute")
        self.q, self.qlock, self.q_hist = (
            None,
            threading.Lock(),
            deque(maxlen=JOINT_HISTORY),
        )
        self.poses, self.plock, self.busy = {}, threading.Lock(), False
        self.keys, self.info = keys, None
        self.stale, self.events = (
            {},
            [],
        )  # key -> when it went stale; what track_events records in exec.json
        self.last_frame, self.call_t0 = None, None
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
                self.q_hist.append((time.time(), self.q))

    def _on_info(self, msg):
        self.info = msg

    def _on_frame(self, rgb, depth):
        """FoundationPose track for every session on this frame (frames arriving while busy are dropped)."""
        now = time.time()
        if self.last_frame is not None and now - self.last_frame > TRACK_SLOW:
            self.event("frame_gap", s=now - self.last_frame)
        self.last_frame = now
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
                self.call_t0 = time.time()
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
                took = time.time() - self.call_t0
                if took > TRACK_SLOW:
                    self.event("slow_track", key=k, s=took)
                with self.plock:
                    self.poses[k] = (time.time(), np.asarray(rep["pose"], float), now)
        except Exception as e:  # noqa: BLE001 -- a lost frame; staleness is checked where poses are used
            self.event("track_error", error=str(e), s=time.time() - self.call_t0)
        finally:
            self.call_t0, self.busy = None, False

    def event(self, kind, **kw):
        """A tracking problem: warned, and kept for exec.json's track_events."""
        kw = {k: round(v, 2) if isinstance(v, float) else v for k, v in kw.items()}
        self.events.append({"t": round(time.time(), 2), "event": kind, **kw})
        self.get_logger().warn(f"track: {kind} {kw}")

    def joints(self):
        with self.qlock:
            if self.q is None:
                raise Stop("no /joint_states")
            return self.q.copy()

    def joints_at(self, t):
        """The joints nearest time t in the /joint_states history."""
        with self.qlock:
            if not self.q_hist:
                raise Stop("no /joint_states")
            return min(self.q_hist, key=lambda s: abs(s[0] - t))[1].copy()

    def pose(self, key):
        """Latest tracked cam_T_obj (body frame), waiting for a fresh one. -> (pose, or None if never tracked; fresh;
        when its frame arrived, or None)."""
        wait = STALE_WAIT if key in self.stale else TRACK_WAIT
        t0 = time.time()
        while True:
            with self.plock:
                p = self.poses.get(key)
            if p is not None and time.time() - p[0] < TRACK_MAX_AGE:
                if key in self.stale:
                    self.event(
                        "recovered", key=key, s=time.time() - self.stale.pop(key)
                    )
                return p[1], True, p[2]
            if time.time() - t0 >= wait:
                break
            time.sleep(0.05)
        if key not in self.stale:
            self.stale[key] = t0
            call, frame = (
                self.call_t0,
                self.last_frame,
            )  # which of the two stopped the poses
            self.event(
                "stale",
                key=key,
                call_running_s=None if call is None else time.time() - call,
                frame_age_s=None if frame is None else time.time() - frame,
            )
        return (None, False, None) if p is None else (p[1], False, p[2])

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

    def start(self, Q, speed):
        """Send joint path Q (Q[0] = where the robot is), timed as in timed(), without waiting for it. Checks the start.
        -> (goal handle, result future, Q), or None when Q does not move."""
        q = self.joints()
        if np.abs(Q[0] - q).max() > START_TOL:
            raise Stop(
                f"path starts {np.abs(Q[0] - q).max():.3f} rad from the robot (stale plan)"
            )
        msg = self.timed(Q, speed)
        if msg is None:
            return None
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
        return gh, gh.get_result_async(), Q

    def cancel(self, motion):
        """Cancel a running goal; False if the driver did not take it within CANCEL_WAIT."""
        gh, result, _ = motion
        if result.done():
            return True
        fut, t_end = gh.cancel_goal_async(), time.time() + CANCEL_WAIT
        while not (fut.done() and result.done()) and time.time() < t_end:
            time.sleep(0.01)
        ok = fut.done() and result.done()
        print(
            "exec: trajectory cancel "
            + ("acknowledged" if ok else "NOT acknowledged -- use the e-stop")
        )
        return ok

    def stop(self, motion):
        """Bring the arm to rest from a running goal, for a re-plan."""
        if not self.cancel(motion):
            raise Stop("trajectory cancel not acknowledged")
        q, t_end = self.joints(), time.time() + CANCEL_WAIT
        while time.time() < t_end:
            time.sleep(0.05)
            q, prev = self.joints(), q
            if np.abs(q - prev).max() < SETTLE_TOL:
                return
        raise Stop("the arm kept moving after the cancel")

    def finish(self, motion):
        """Wait for a started goal; checks its result and the arrival. Ctrl-C cancels it."""
        _, result, Q = motion
        try:
            res = self._wait(result, "result").result
        except KeyboardInterrupt:
            self.cancel(motion)
            raise
        if res.error_code != 0:
            raise Stop(
                f"trajectory failed: error_code {res.error_code} {res.error_string}"
            )
        if np.abs(self.joints() - Q[-1]).max() > ARRIVE_TOL:
            raise Stop(
                f"arrived {np.abs(self.joints() - Q[-1]).max():.3f} rad from the path end"
            )

    def send(self, Q, speed):
        """Run joint path Q to its end."""
        motion = self.start(Q, speed)
        if motion is not None:
            self.finish(motion)

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


def delta(T1, T0):
    """(translation m, rotation rad) between two poses."""
    return float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), float(
        Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T).magnitude()
    )


def gate(a, what, first=True):
    if (a.gate == "all" or (a.gate == "first" and first)) and input(
        f"[gate] {what} -- Enter to go, anything else stops: "
    ).strip():
        raise Stop(f"stopped at gate: {what}")


def run(node, a, log):
    run = Path(a.run)
    objects = json.loads((run / "objects.json").read_text())
    stage = next(
        (
            s
            for s in json.loads((run / "rrc" / "rrc.json").read_text())
            if s["accepted"]
        ),
        None,
    )
    if stage is None:
        raise Stop("RRC accepted no constraint in this run: nothing to execute")
    grasp = json.loads((run / "grasp" / "grasp.json").read_text())[0]
    packet = json.loads((run / "packet.json").read_text())
    c = {
        k: np.asarray(v) if isinstance(v, list) else v
        for k, v in stage["accepted"].items()
    }
    A, Pv, ACT = stage["active"], stage["passive"], stage["action"]
    if a.distance_cm is not None:
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
    seen = {}  # the held object's last measured Phi (grasp -> object), for when tracking goes stale

    def tracked(name):
        if (
            a.no_tracking
        ):  # planned poses; the held object rigid in the grasp (the grasp's Phi)
            if name == A and held:
                return ee_cam() @ np.asarray(grasp["phi_grasp_T_obj"], float)
            return np.asarray(objects[name]["cam_T_can"], float)
        body, fresh, t_frame = node.pose(f"omni_m{objects[name]['mark']}")
        carried = name == A and held
        if not fresh and carried:
            return ee_cam() @ seen.get(
                "phi", np.asarray(grasp["phi_grasp_T_obj"], float)
            )
        if body is None:  # never tracked: where the plan saw it
            return np.asarray(objects[name]["cam_T_can"], float)
        T = body.copy()
        T[:3, :3] = T[:3, :3] @ Rc[name]
        if carried:
            # the joints when the frame arrived, not now: the arm has moved on while the frame was tracked
            seen["phi"] = np.linalg.inv(C_B @ fk(node.joints_at(t_frame)) @ TG) @ T
        return T

    show = lambda Q: node.show(Q, [fk(q)[:3, 3] for q in Q], a.speed)

    timing = {}

    def planned(what, goal, first, fallback=False, **kw):
        t0 = time.time()
        r = call("plan", q=list(node.joints()), goal=goal, **kw)
        failures = r[
            "goal_failures"
        ]  # the IK solutions that could not be planned to, nearest first
        for why in failures:
            print(f"exec: {what}: {why}")
        chose = r["path"] is None and fallback
        if chose:  # same tool goal, but cuRobo chooses the joint angles
            kw.pop("q_goal_ref")
            kw.pop("alternatives")
            r = call("plan", q=list(node.joints()), goal=goal, **kw)
            what += (
                ", to cuRobo's own IK solution, not the nearest: check the arm's motion"
            )
        elif failures:
            what += f", to the IK solution {len(failures) + 1} by distance, not the nearest: check the arm's motion"
        # sidecar + cuRobo round trips, before the gate
        timing.update(plan_s=time.time() - t0, failures=failures, fallback=chose)
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
    mesh = objects[A]["final_mesh"]
    t0 = time.time()
    call(
        "fit_held", mesh_path=mesh
    )  # at the start: the fit would otherwise hold up the stage's first plan
    log["grasp"]["fit_held_s"] = time.time() - t0
    t0 = time.time()
    call("map", depth=depth0, K=K, T_base_cam=B_C, q=list(q_cap), plane=packet["plane"])
    log["grasp"]["map_s"] = time.time() - t0
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
    log["grasp"]["close"] = node.gripper(CLOSE_GAP, a.effort)
    held.append(A)
    T_up = ee_cam()
    T_up[:3, 3] += LIFT * C_B[:3, 2]  # base +z in the camera frame
    straight("lift", tool0(T_up))

    # the stage's primitive, closed loop; the held object leaves the map and rides on tool0 as spheres
    grow = round(MASK_GROW * masks[A].shape[0] / 480)
    grown = (
        cv2.dilate(masks[A].astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8))
        > 0
    )
    t0 = time.time()
    call(
        "map",
        depth=depth0,
        K=K,
        T_base_cam=B_C,
        q=list(q_cap),
        exclude=grown,
        plane=packet["plane"],
    )
    map_s = time.time() - t0
    exclude = np.any(list(masks.values()), axis=0)

    def approach(T_t, T_p):
        T = T_t.copy()
        if ACT == "place":
            T[:3, 3] -= max(abs(c["distance_m"]), PRE_PLACE) * (
                T_p[:3, :3] @ c["passive_axis"]
            )
        return T

    T_t, T_plan, solving, path, motion = None, np.eye(4), True, None, None
    T_start, t_stage, it = (
        ee_cam(),
        time.time(),
        0,
    )  # O9's path cost is measured from the stage's start pose
    log[ACT] = {"map_s": map_s, "updates": [], "plans": [], "stops": []}

    def rest_of_path():
        """From the current joints along `path`, past its nearest waypoint."""
        q = node.joints()
        j = int(np.argmin(np.abs(path - q).max(1)))
        return np.vstack([q, path[min(j + 1, len(path) - 1) :]])

    try:
        while True:
            if solving:
                t0 = time.time()
                _, T_p = tracked(A), tracked(Pv)
                T_e = ee_cam()
                phi = seen.get("phi", np.asarray(grasp["phi_grasp_T_obj"], float))
                tracked_s = time.time() - t0
                t0 = time.time()
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
                    active={
                        "mesh_path": mesh,
                        "R": Rc[A],
                    },  # O9's L_collision: held and passive objects
                    passive={"mesh_path": objects[Pv]["final_mesh"], "R": Rc[Pv]},
                    x0=T_t,
                    overlay=False,
                )
                T_t = np.asarray(r["cam_T_ee_target"], float)
                solve_s = time.time() - t0
                accepted = bool(r["accepted"])
                # no tracking: nothing new to re-solve from, so one solve and one plan
                solving = not a.no_tracking and (
                    not accepted
                    or np.linalg.norm(T_t[:3, 3] - T_e[:3, 3]) > RESOLVE_STOP
                )
                phi0 = phi if it == 0 else phi0
                d_t, d_r = delta(T_t, T_plan)
                moved = accepted and (path is None or d_t > REPLAN_T or d_r > REPLAN_R)
                log[ACT]["updates"].append(
                    {
                        "it": it,
                        "t": t0
                        - t_stage,  # s since the stage began, at this solve's start
                        "moving": motion is not None and not motion[1].done(),
                        "ok": accepted,
                        **{k_: float(v) for k_, v in r["terms"].items()},
                        # what drives re-plans: the target's move from the current plan's target, and Phi's drift
                        # from the stage's first measurement (cm, deg)
                        "target_vs_plan": None
                        if path is None
                        else [d_t * 100, np.degrees(d_r)],
                        "phi_vs_first": [
                            x * f
                            for x, f in zip(delta(phi, phi0), (100, np.degrees(1)))
                        ],
                        "replan": bool(moved),
                        "tracked_s": tracked_s,  # poses and FK, sidecar round trips included
                        "solve_s": solve_s,  # O9 re-solve on the sidecar, round trip
                        "phi": phi.tolist(),
                        "cam_T_passive": T_p.tolist(),
                        "target": T_t.tolist(),
                    }
                )
                if (
                    not accepted
                ):  # O9 report failure: never move to it; a running path carries on
                    rho = float(r["terms"]["constraint"])
                    print(f"exec: update {it}: O9 solve rejected, rho {rho:.3f}")
                    if path is None:
                        raise Stop(
                            f"O9 rejected the first solve (rho {rho:.3f}); see the last update in exec.json"
                        )
                    T_t = T_plan
                if moved:
                    T_body = (
                        TG @ phi
                    )  # the mesh is in the body frame: canonical -> body
                    T_body[:3, :3] = T_body[:3, :3] @ Rc[A].T
                    carry = {"mesh_path": mesh, "tool0_T_obj": T_body}
                    goal = tool0(approach(T_t, T_p))
                    plan_log = lambda rec: log[ACT]["plans"].append({"it": it, **rec})
                    if path is None:  # at rest, before the stage's first motion
                        path, why = planned(
                            f"{ACT} (update {it})",
                            goal,
                            True,
                            alternatives=IK_ALTERNATIVES,
                            fallback=True,
                            held=carry,
                            q_goal_ref=list(node.joints()),
                        )
                        plan_log(
                            {
                                "success": path is not None,
                                "reason": why,
                                "plan_s": timing["plan_s"],
                                "failures": timing["failures"],
                                "fallback": timing["fallback"],
                            }
                        )
                        if path is None:
                            raise Stop(f"no plan at update {it}: {why}")
                        T_plan = T_t
                        motion = node.start(path, a.speed)
                    else:
                        ref = list(path[-1])  # stay on the previous plan's IK branch
                        # first from where the arm is now, still moving: if this fails the arm is not interrupted
                        t0 = time.time()
                        trial = call(
                            "plan",
                            q=list(node.joints()),
                            goal=goal,
                            held=carry,
                            q_goal_ref=ref,
                        )
                        plan_log(
                            {
                                "moving": True,
                                "success": trial["path"] is not None,
                                "reason": trial["reason"],
                                "plan_s": time.time() - t0,
                            }
                        )
                        if (
                            trial["path"] is None
                        ):  # e.g. inside the passive object's buffer near the end
                            T_t, solving = T_plan, False
                        else:
                            t_stop = time.time()
                            node.stop(motion)
                            new, why = planned(
                                f"{ACT} (update {it})",
                                goal,
                                False,
                                held=carry,
                                q_goal_ref=ref,
                            )
                            plan_log(
                                {
                                    "moving": False,
                                    "success": new is not None,
                                    "reason": why,
                                    "plan_s": timing["plan_s"],
                                }
                            )
                            if new is None:
                                T_t, solving = T_plan, False
                                motion = node.start(rest_of_path(), a.speed)
                            else:
                                path, T_plan = new, T_t
                                motion = node.start(path, a.speed)
                            log[ACT]["stops"].append(
                                {"it": it, "s": time.time() - t_stop}
                            )
            if motion is None:
                break
            if motion[1].done():
                node.finish(motion)
                break
            if not solving:
                time.sleep(0.02)
            it += 1
    except (
        BaseException
    ):  # Ctrl-C, a refused gate, a sidecar error: the arm must not run on
        if motion is not None:
            node.cancel(motion)
        raise
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
        "--no-tracking",
        action="store_true",
        help="no FoundationPose: objects at their planned poses, the held one rigid in the grasp; one O9 solve and one plan per stage. The robot still moves",
    )
    ap.add_argument(
        "--distance-cm",
        type=float,
        default=None,
        help="TEST ONLY: d to use instead of the VLM's; negative = against the passive axis",
    )
    a, ros_args = ap.parse_known_args()
    objects = json.loads((Path(a.run) / "objects.json").read_text())
    rclpy.init(args=ros_args)
    keys = [] if a.no_tracking else [f"omni_m{o['mark']}" for o in objects.values()]
    node = Executor(keys, a.controller)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    video = (
        Path(a.run) / "videos" / f"exec_{time.strftime('%Y%m%d_%H%M%S')}_camera2.webm"
    )
    while video.exists():  # two executes in the same second
        video = video.with_name(video.stem + "_1.webm")
    rec = Recorder(node, video)
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
        log["track_events"] = node.events
        log["video"] = rec.close()
        (Path(a.run) / "exec.json").write_text(
            json.dumps(
                log,
                indent=1,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
            )
        )
        print("exec:", log["status"])
        ex.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
