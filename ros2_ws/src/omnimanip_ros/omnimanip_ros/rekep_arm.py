"""ReKep's arm server on the robot (protocol: rekep/loop.py, which runs on the host in our uv env): the planning frame
from the bridge's /omnimanip/capture, the latest synced camera frame and joints, joint paths through the UR driver's
FollowJointTrajectory and the Robotiq through gripper-control's /gripper/gripper_cmd, all as `execute` does them
(its Executor, without FoundationPose: ReKep tracks its keypoints itself). No operator gates (the user's choice for
ReKep's streamed execution); Ctrl-C here or in the loop stops it, and the e-stop stays in reach. A close holds
something when the jaws stop more than GRASPED_GAP apart. With --out (the runs root), records camera 1 to
<out>/videos/rekep_<start>_camera1.webm and writes <out>/videos/rekep_<start>_joints.json on exit, as `execute` does.

    docker exec -it OmniManipROS bash -c 'umask 0 && source install/setup.bash && \\
        ros2 run omnimanip_ros rekep_arm --out $RUNS_DIR'
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import rclpy
import zmq
from manip_bridge.img import camera_info_to_K, image_to_depth_m, image_to_rgb
from omnimanip_interfaces import srv
from rclpy.executors import MultiThreadedExecutor

from . import conv
from .execute_node import CLOSE_GAP, OPEN_GAP, REC_FINISH, REC_TAIL, Executor

msgpack_numpy.patch()
GRASPED_GAP = 0.002  # m


class Arm(Executor):
    def __init__(self, controller):
        super().__init__([], controller)
        self.frame, self.flock = {}, threading.Lock()
        self.capture_cli = self.create_client(srv.Capture, "omnimanip/capture")

    def _on_frame(self, rgb, depth):
        q = self.q
        if self.info is None or q is None:
            return
        f = {
            "rgb": image_to_rgb(rgb),
            "depth": image_to_depth_m(depth),
            "q": q.copy(),
            "t": time.time(),
        }
        with self.flock:
            self.frame = f

    def capture(self):
        if not self.capture_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/omnimanip/capture not available (is the bridge up?)")
        r = self._wait(self.capture_cli.call_async(srv.Capture.Request()), "capture")
        if not r.success:
            raise RuntimeError(f"capture: {r.message}")
        m = conv.mask_of(r.robot_mask)
        return {
            "rgb": image_to_rgb(r.rgb),
            "depth": image_to_depth_m(r.depth),
            "K": camera_info_to_K(r.camera_info),
            "packet": r.packet_json,
            "robot_mask": np.zeros((0, 0), bool) if m is None else m,
        }

    def handle(self, req, speed_cap):
        c = req["cmd"]
        if c == "capture":
            return self.capture()
        if c == "joints":
            return {"q": self.joints()}
        if c == "move":
            self.send(
                np.asarray(req["path"], float), min(float(req["speed"]), speed_cap)
            )
            return {"q": self.joints()}
        if c == "gripper":
            r = self.gripper(CLOSE_GAP if req["close"] else OPEN_GAP, self.effort)
            return {"grasped": bool(req["close"] and r["position"] > GRASPED_GAP), **r}
        raise ValueError(f"unknown cmd {c!r}")


def serve(sock, fn):
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            rep = {"ok": True, **fn(req)}
        except Exception as e:  # noqa: BLE001 -- reported to the loop, which stops
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(rep, use_bin_type=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5684)
    ap.add_argument("--controller", default="scaled_joint_trajectory_controller")
    ap.add_argument(
        "--max-speed",
        type=float,
        default=0.1,
        help="cap on the loop's speed (fraction of the joint limits)",
    )
    ap.add_argument("--effort", type=float, default=60.0, help="gripper force, N")
    ap.add_argument(
        "--out",
        default=None,
        help="runs root: camera 1's video and the joints go to <out>/videos/",
    )
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Arm(a.controller)
    node.effort = a.effort
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    rec, stem = None, None
    if a.out:
        stem = Path(a.out) / "videos" / f"rekep_{time.strftime('%Y%m%d_%H%M%S')}"
        video = Path(f"{stem}_camera1.webm")
        rec = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omnimanip_ros.execute_node",
                "--record",
                str(video),
            ],
            start_new_session=True,
        )

    def frame(_):
        with node.flock:
            if not node.frame:
                raise RuntimeError("no synced camera frame yet")
            return dict(node.frame)

    ctx = zmq.Context.instance()
    fs, cs = ctx.socket(zmq.REP), ctx.socket(zmq.REP)
    fs.bind(f"tcp://*:{a.port + 1}")
    cs.bind(f"tcp://*:{a.port}")
    threading.Thread(target=serve, args=(fs, frame), daemon=True).start()
    print(f"[rekep-arm] commands on {a.port}, frames on {a.port + 1}", flush=True)
    try:
        serve(cs, lambda req: node.handle(req, a.max_speed))
    except KeyboardInterrupt:
        pass
    finally:
        if a.out:
            time.sleep(REC_TAIL)
            Path(f"{stem}_joints.json").write_text(
                json.dumps(
                    {
                        "t": [t for t, _ in node.q_log],
                        "q": [q.tolist() for _, q in node.q_log],
                        "motions": [
                            {"t": m["t"], "q": np.asarray(m["q"]).tolist()}
                            for m in node.sent
                        ],
                    }
                )
            )
        if rec is not None:
            rec.terminate()
            try:
                rec.wait(timeout=REC_FINISH)
            except subprocess.TimeoutExpired:
                rec.kill()
        ex.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
