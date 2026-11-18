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

from omnimanip import motion, urdf_fk
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
MAX_DEPTH = 1.5  # m: farther has no depth, as the real cell's black backdrop (else the sim's wall wins the table fit)
# robosuite mounts its Robotiq 90 deg about the approach axis from the planner URDF's (its fingers close along the
# URDF gripper's z): the arm's last joint runs WRIST_OFF ahead of the joints the loop sees, so the two grippers match
WRIST_OFF = np.r_[0, 0, 0, 0, 0, np.pi / 2]
# robosuite's Robotiq ends 0.86 cm short of the real one (the planner URDF's, gripper.py's fingertips) along the
# approach: its gripper is moved out that far, so the sim's fingers are where the real gripper's are
GRIPPER_EXTEND = 0.0086
# before a close the arm holds the last move's end until within SETTLE_RAD (as sim/execute.py settles), so the fingers
# meet the object together; after first contact they keep squeezing for SQUEEZE_STEPS
SETTLE_RAD, SETTLE_STEPS, SQUEEZE_STEPS = 0.002, 100, 15

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="../../sim/scenes/pour_tea_real.json")
ap.add_argument("--port", type=int, default=5684)
ap.add_argument(
    "--camera-from",
    default=None,
    help="a JSON with base_T_cam (robot base, the sim's convention), K and cam_height_above_table: a real run's camera",
)
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


def body(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def pad_tip_gap():
    """How far the sim's finger pads end behind the planner model's fingertips along the approach (m)."""
    mujoco.mj_forward(m, d)
    s_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    z = d.site_xmat[s_id].reshape(3, 3)[:, 2]  # approach
    tips = (
        max(
            (d.geom_xpos[g] - d.site_xpos[s_id]) @ z
            + np.abs(d.geom_xmat[g].reshape(3, 3).T @ z) @ m.geom_size[g]
            for g in range(m.ngeom)
            if "fingerpad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        )
        + d.site_xpos[s_id] @ z
    )
    b = body("robot0_base")
    W_B = np.eye(4)
    W_B[:3, :3], W_B[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    ee = W_B @ urdf_fk.fk(
        urdf_fk.chain(motion.ROBOT_URDF),
        dict(zip(motion.UR_JOINTS, arm.q() - WRIST_OFF)),
    )
    ee = ee @ motion.tool0_T_grasp(0.0)
    return float(ee[:3, 3] @ z - tips)


root = (
    next(  # the gripper's root body: the first gripper0_ body whose parent is not one
        b
        for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
            "gripper0_"
        )
        and not (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]) or ""
        ).startswith("gripper0_")
    )
)
gap0 = pad_tip_gap()
s_z = d.site_xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)].reshape(3, 3)[
    :, 2
]
m.body_pos[root] += d.xmat[m.body_parentid[root]].reshape(3, 3).T @ (
    GRIPPER_EXTEND * s_z
)
print(
    f"[rekep-arm sim] gripper moved {GRIPPER_EXTEND * 100:.2f} cm out: pad tips {gap0 * 100:.2f} -> "
    f"{pad_tip_gap() * 100:.2f} cm short of the real gripper's",
    flush=True,
)

# the camera, as sim/capture.py places it
b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
W_B = np.eye(4)
W_B[:3, :3], W_B[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
if a.camera_from:
    c = json.loads(Path(a.camera_from).read_text())
    B_C, CAM_HEIGHT, FY = (
        np.array(c["base_T_cam"], float),
        float(c["cam_height_above_table"]),
        float(c["K"][1][1]),
    )
else:  # the bag's camera: its base_link faces the other way from robosuite's base
    B_C = np.diag([-1.0, -1.0, 1.0, 1.0]) @ BASE_LINK_T_CAM
W_C = W_B @ B_C
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


def _mesh(n, o):
    """Object n's visual mesh and its body, as sim/capture.py's ground truth (the OBJ's frame is the body's)."""
    vis = o.asset.name + "_visual_mesh"
    g = next(
        g
        for g in range(m.ngeom)
        if m.geom_dataid[g] >= 0 and m.mesh(m.geom_dataid[g]).name.endswith(vis)
    )
    V, F = S.read_obj(o.asset / "meshes" / f"{o.asset.name}_visual.obj")
    return {"body": int(m.geom_bodyid[g]), "V": np.asarray(V), "F": np.asarray(F)}


MESHES = {n: _mesh(n, o) for n, o in scene.objects.items()}


def mesh_near(p):
    """SIM ONLY (rekep.run --sim-mesh-grasps): the ground-truth mesh nearest base_link point p and its pose now."""
    best = None
    for n, o in MESHES.items():
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = d.xmat[o["body"]].reshape(3, 3), d.xpos[o["body"]]
        T = np.linalg.inv(W_B) @ T
        dist = np.linalg.norm(o["V"] @ T[:3, :3].T + T[:3, 3] - p, axis=1).min()
        if best is None or dist < best[0]:
            best = (dist, n, T)
    dist, n, T = best
    return {
        "name": n,
        "V": MESHES[n]["V"],
        "F": MESHES[n]["F"],
        "base_T_obj": T,
        "dist_m": float(dist),
    }


renderer, opt = mujoco.Renderer(m, H, W), mujoco.MjvOption()
opt.geomgroup[0] = 0  # robosuite: group 0 = collision geoms, group 1 = visuals
lock, latest, grip, target = threading.Lock(), {}, [GRIPPER_OPEN], [None]
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
        "depth": np.where(depth < MAX_DEPTH, depth, 0).astype(np.float32),
        "robot_mask": np.isin(seg, robot_geoms),
        "q": arm.q() - WRIST_OFF,
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
            "joints": {"names": names, "q": (arm.q() - WRIST_OFF).tolist()},
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
    q = arm.q() if target[0] is None else target[0]
    if g == GRIPPER_CLOSE:  # settle at the last move's end first
        for _ in range(SETTLE_STEPS):
            if np.abs(arm.q() - q).max() <= SETTLE_RAD:
                break
            tracker._step(q, grip[0])
            render()
    grip[0], first = g, None
    for k in range(GRIP_STEPS + SQUEEZE_STEPS):
        tracker._step(q, g)
        render()
        if (
            g == GRIPPER_CLOSE
            and first is None
            and any(arm.contacts_between("gripper0", o) for o in scene.objects)
        ):
            first = k
        if (first is not None and k - first >= SQUEEZE_STEPS) or (
            first is None and k >= GRIP_STEPS
        ):
            break


held = {}  # at a close that holds: the object and its pose in the grip site's frame


def site_T():
    s_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.site_xmat[s_id].reshape(3, 3), d.site_xpos[s_id]
    return T


def obj_T(o):
    b = next(
        b
        for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(o)
    )
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    return T


def grip_report(o):
    """Diagnostics of the grip on o: the pads' summed normal force on it (N) and how far back from the pads' tips
    (along the approach) the contacts sit (m)."""
    pads = {
        g
        for g in range(m.ngeom)
        if "fingerpad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
    }
    obj = {
        g
        for g in range(m.ngeom)
        if (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        ).startswith(o)
    }
    S = site_T()
    z = S[:3, 2]
    tips = (
        max(
            (d.geom_xpos[g] - S[:3, 3]) @ z
            + np.abs(d.geom_xmat[g].reshape(3, 3).T @ z) @ m.geom_size[g]
            for g in pads
        )
        + S[:3, 3] @ z
    )
    force, back = 0.0, []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in pads and c.geom2 in obj) or (c.geom2 in pads and c.geom1 in obj):
            mujoco.mj_contactForce(m, d, i, f6)
            force += abs(f6[0])
            back.append(float(tips - c.pos @ z))
    return {
        "pad_force_N": round(force, 2),
        "contact_back_from_tips_m": [round(b, 4) for b in back],
    }


def handle(req):
    c = req["cmd"]
    if c == "capture":
        return capture()
    if c == "grasping":  # the held object still between the fingers
        return {
            "grasping": bool(held) and arm.contacts_between("gripper0", held["o"]) > 0
        }
    if c == "mesh":
        return mesh_near(np.asarray(req["point"], float))
    if c == "joints":
        return {"q": arm.q() - WRIST_OFF}
    if c == "move":
        path = np.asarray(req["path"], float) + WRIST_OFF
        st = tracker.track(path, grip[0], on_step=render)
        target[0] = path[-1]
        rep = {
            "ok": not st.stalled,
            "q": arm.q() - WRIST_OFF,
            "error": "the arm stalled (obstructed)",
        }
        if held:  # how far the held object has moved in the grip since the close
            D = np.linalg.inv(held["T"]) @ np.linalg.inv(site_T()) @ obj_T(held["o"])
            rep["held_drift"] = {
                "pos_m": round(float(np.linalg.norm(D[:3, 3])), 4),
                "rot_deg": round(
                    float(
                        np.degrees(
                            np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))
                        )
                    ),
                    1,
                ),
                **grip_report(held["o"]),
            }
        return rep
    if c == "gripper":
        hold(GRIPPER_CLOSE if req["close"] else GRIPPER_OPEN)
        touch = {o: arm.contacts_between("gripper0", o) for o in scene.objects}
        pads = [
            d.geom_xpos[g]
            for g in range(m.ngeom)
            if "fingerpad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        ]
        gap = float(max(np.linalg.norm(p - q) for p in pads for q in pads))
        grasped = req["close"] and any(touch.values())
        held.clear()
        rep = {
            "ok": True,
            "grasped": bool(grasped),
            "touching": touch,
            "pad_span_m": gap,
        }
        if grasped:
            o = max(touch, key=lambda k: touch[k])
            held.update(o=o, T=np.linalg.inv(site_T()) @ obj_T(o))
            rep |= grip_report(o)
        return rep
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
