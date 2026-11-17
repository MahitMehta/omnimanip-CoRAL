"""ReKep's arm server in the sim (protocol: rekep/loop.py): Albus's MuJoCo scene with the UR5e on its joint-position
tracking controller, a camera placed like the real RealSense (sim/capture.py's), frames rendered every control step
(20 Hz sim time) and served on port + 1 while the arm moves. The robot mask is MuJoCo's segmentation; a close holds
something when the fingers touch a scene object. --video records every rendered frame.

Run from third_party/ur5e-manip-sim (add --view, with mjpython in place of python, to watch it live):
  MUJOCO_GL=egl PYTHONPATH=.:../..:../ur5e-manip-hardware/ros2_ws/src/manip_bridge ../../sim/.venv/bin/python \\
      ../../sim/rekep_arm.py --video ../../outputs/rekep/sim_exec.mp4
then, in the main env: python -m rekep.run --out DIR --arm tcp://127.0.0.1:5684 --task ...
"""

import argparse
import json
import tempfile
import threading
from pathlib import Path

import cv2
import imageio
import msgpack
import msgpack_numpy
import mujoco
import numpy as np
import zmq
from manip_sim.execution import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LiveArm,
    Tracker,
    tracking_controller_config,
)
from manip_sim.scene import load_scene, make_env

from omnimanip import shared_perception as S

msgpack_numpy.patch()
# sim/capture.py's real camera (base_link) and its height above the table, intrinsics
BASE_LINK_T_CAM = np.array(
    [
        [0.008266, -0.366931, 0.930212, -1.193813],
        [-0.997900, -0.062791, -0.015901, 0.272376],
        [0.064244, -0.928127, -0.366679, 0.229418],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
CAM_HEIGHT, FY, H, W = 0.2547, 607.4, 480, 640
ROBOT_PREFIXES = ("robot0_", "gripper0_", "fixed_mount0_")
SITE, CAMERA = "gripper0_right_grip_site", "frontview"
GRIP_STEPS = 40  # control steps the gripper is held closing or opening

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="../../sim/scenes/pour_tea_real.json")
ap.add_argument("--port", type=int, default=5684)
ap.add_argument("--video", default=None, help="mp4 of the camera (every frame served)")
ap.add_argument(
    "--view", action="store_true", help="live MuJoCo viewer (macOS: run with mjpython)"
)
a = ap.parse_args()

scene = load_scene(a.scene)
env, _ = make_env(
    scene, has_renderer=False, controller_configs=tracking_controller_config("UR5e")
)
arm = LiveArm(env, SITE)
tracker = Tracker(env, arm)
m, d = arm.model, arm.data

# the camera, as sim/capture.py places it
b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
W_B = np.eye(4)
W_B[:3, :3], W_B[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
Rz = np.diag([-1.0, -1.0, 1.0])  # base_link -> robosuite base
W_C = np.eye(4)
W_C[:3, :3] = W_B[:3, :3] @ Rz @ BASE_LINK_T_CAM[:3, :3]
W_C[:3, 3] = W_B[:3, :3] @ Rz @ BASE_LINK_T_CAM[:3, 3] + W_B[:3, 3]
W_C[2, 3] = scene.table_top_z + CAM_HEIGHT
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
qc = np.zeros(4)
mujoco.mju_mat2Quat(qc, (W_C[:3, :3] @ np.diag([1.0, -1.0, -1.0])).ravel())
m.cam_pos[cam], m.cam_quat[cam], m.cam_fovy[cam] = (
    W_C[:3, 3],
    qc,
    np.degrees(2 * np.arctan(H / 2 / FY)),
)
f = H / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
T_base_cam = np.linalg.inv(W_B) @ W_C
robot_geoms = [
    g
    for g in range(m.ngeom)
    if (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    ).startswith(ROBOT_PREFIXES)
]
renderer, opt = mujoco.Renderer(m, H, W), mujoco.MjvOption()
opt.geomgroup[0] = 0  # robosuite: group 0 = collision geoms, group 1 = visuals
lock, latest, grip = threading.Lock(), {}, [GRIPPER_OPEN]
video = imageio.get_writer(a.video, fps=20) if a.video else None
viewer = None
if a.view:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(m, d)


def render():
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=CAMERA, scene_option=opt)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    seg = renderer.render()[..., 0]
    renderer.disable_segmentation_rendering()
    fr = {
        "rgb": rgb,
        "depth": np.where(depth < 10, depth, 0).astype(np.float32),
        "robot_mask": np.isin(seg, robot_geoms),
        "q": arm.q(),
        "t": float(d.time),
    }
    with lock:
        latest.update(fr)
    if video is not None:
        video.append_data(rgb)
    if viewer is not None:
        viewer.sync()


def serve_frames():
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://*:{a.port + 1}")
    while True:
        msgpack.unpackb(sock.recv(), raw=False)
        with lock:
            fr = dict(latest)
        sock.send(msgpack.packb({"ok": True, **fr}, use_bin_type=True))


def capture():
    """The frame packet as sim_packet builds it from a sim/capture.py frame."""
    render()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cv2.imwrite(str(t / "rgb.png"), cv2.cvtColor(latest["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(t / "depth_mm.png"), np.round(latest["depth"] * 1000).astype(np.uint16)
        )
        cv2.imwrite(
            str(t / "robot_mask.png"), latest["robot_mask"].astype(np.uint8) * 255
        )
        names = [
            m.joint(j).name
            for j in range(m.njnt)
            if m.joint(j).name.startswith("robot0_")
        ]
        meta = {
            "K": K.tolist(),
            "world_T_cam": W_C.tolist(),
            "world_T_base": W_B.tolist(),
            "joints": {"names": names, "q": arm.q().tolist()},
        }
        (t / "meta.json").write_text(json.dumps(meta))
        rgb, depth, K_, packet, rmask = S.sim_packet(t)
    return {
        "rgb": rgb,
        "depth": depth,
        "K": K_,
        "packet": json.dumps(packet, default=lambda x: np.asarray(x).tolist()),
        "robot_mask": rmask,
    }


def hold(g):
    grip[0], q = g, arm.q()
    for _ in range(GRIP_STEPS):
        tracker._step(q, g)
        render()


def handle(req):
    c = req["cmd"]
    if c == "capture":
        return capture()
    if c == "joints":
        return {"q": arm.q()}
    if c == "move":
        st = tracker.track(np.asarray(req["path"], float), grip[0], on_step=render)
        return {
            "ok": not st.stalled,
            "q": arm.q(),
            "error": "the arm stalled (obstructed)",
        }
    if c == "gripper":
        hold(GRIPPER_CLOSE if req["close"] else GRIPPER_OPEN)
        grasped = req["close"] and any(
            arm.contacts_between("gripper0", o) for o in scene.objects
        )
        return {"ok": True, "grasped": bool(grasped)}
    return {"ok": False, "error": f"unknown cmd {c!r}"}


render()
threading.Thread(target=serve_frames, daemon=True).start()
sock = zmq.Context.instance().socket(zmq.REP)
sock.bind(f"tcp://*:{a.port}")
print(f"[rekep-arm sim] commands on {a.port}, frames on {a.port + 1}", flush=True)
try:
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            rep = {"ok": True, **handle(req)}
        except Exception as e:  # noqa: BLE001 -- reported to the client
            rep = {"ok": False, "error": repr(e)}
        sock.send(msgpack.packb(rep, use_bin_type=True))
except KeyboardInterrupt:
    if video is not None:
        video.close()
        print("wrote", a.video)
