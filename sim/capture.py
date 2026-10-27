"""One RGB-D frame of a sim scene from a camera placed like the real RealSense. Writes rgb.png, depth_mm.png,
robot_mask.png, mask_<object>.png and meta.json (K, world_T_cam, world_T_base, table_top_z, joints, and per object
the ground-truth world_T_mesh of its visual mesh, the mesh path and its declared front) under --out.

Run from third_party/ur5e-manip-sim (its scene assets are relative to it):
  MUJOCO_GL=cgl PYTHONPATH=. ../../sim/.venv/bin/python ../../sim/capture.py --out ../../outputs/sim/frame
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from manip_sim.scene import load_scene, make_env

# The real camera (bag tf_static, camera_color_optical_frame in base_link) and its height above the table plane
# (RANSAC plane |d| of the frame packet); the robot's forward is -x in base_link and +x in robosuite's base frame,
# hence the 180 deg yaw between them.
BASE_LINK_T_CAM = np.array(
    [
        [0.008266, -0.366931, 0.930212, -1.193813],
        [-0.997900, -0.062791, -0.015901, 0.272376],
        [0.064244, -0.928127, -0.366679, 0.229418],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
CAM_HEIGHT = 0.2547  # m above the table
FY, H, W = (
    607.4,
    480,
    640,
)  # real intrinsics (fy; MuJoCo cameras have one fovy and a centred principal point)
ROBOT_PREFIXES = ("robot0_", "gripper0_", "fixed_mount0_")
CAMERA = "frontview"  # repurposed: moved to the real camera's pose

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="../../sim/scenes/pour_tea_real.json")
ap.add_argument("--out", required=True)
ap.add_argument(
    "--centre-camera",
    action="store_true",
    help="same height, tilt and distance from the base as the real camera, moved onto the robot's forward axis and levelled (no roll)",
)
ap.add_argument(
    "--tilt-up",
    type=float,
    default=0.0,
    help="deg: pitch the camera up from the real camera's tilt",
)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

scene = load_scene(a.scene)
env, _ = make_env(scene, has_renderer=False)
m, d = env.sim.model._model, env.sim.data._data

base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
world_T_base = np.eye(4)
world_T_base[:3, :3], world_T_base[:3, 3] = d.xmat[base].reshape(3, 3), d.xpos[base]
Rz = np.diag([-1.0, -1.0, 1.0])  # base_link -> robosuite base
world_T_cam = np.eye(4)  # optical frame: x right, y down, z forward
world_T_cam[:3, :3] = world_T_base[:3, :3] @ Rz @ BASE_LINK_T_CAM[:3, :3]
world_T_cam[:3, 3] = (
    world_T_base[:3, :3] @ Rz @ BASE_LINK_T_CAM[:3, 3] + world_T_base[:3, 3]
)
world_T_cam[2, 3] = (
    scene.table_top_z + CAM_HEIGHT
)  # the robot sits on a pedestal in sim, at table level for real
if a.centre_camera:  # same height, tilt and distance from the base, moved onto its +x axis, facing it head-on
    rel = world_T_cam[:2, 3] - world_T_base[:2, 3]
    world_T_cam[:2, 3] = world_T_base[:2, 3] + [np.linalg.norm(rel), 0.0]
    f = world_T_cam[:2, 2]  # optical axis, horizontal part
    yaw = np.pi - np.arctan2(f[1], f[0])  # turn it to point along -x, back at the base
    Ry = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    world_T_cam[:3, :3] = Ry @ world_T_cam[:3, :3]

t = np.radians(
    a.tilt_up
)  # about the optical x axis (right): the view axis moves toward -y (up)
world_T_cam[:3, :3] = world_T_cam[:3, :3] @ np.array(
    [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
)

if a.centre_camera:  # level: no roll, the image's horizontal axis parallel to the table
    z = world_T_cam[:3, 2]
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    world_T_cam[:3, :3] = np.c_[x, np.cross(z, x), z]

cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
q = np.zeros(4)
mujoco.mju_mat2Quat(
    q, (world_T_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])).ravel()
)  # MuJoCo: looks down -z, y up
m.cam_pos[cam], m.cam_quat[cam] = world_T_cam[:3, 3], q
m.cam_fovy[cam] = np.degrees(2 * np.arctan(H / 2 / FY))
mujoco.mj_forward(m, d)

r = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption()
opt.geomgroup[0] = 0  # robosuite: group 0 = collision geoms, group 1 = visuals
r.update_scene(d, camera=CAMERA, scene_option=opt)
rgb = r.render()
r.enable_depth_rendering()
depth = r.render()
r.disable_depth_rendering()
r.enable_segmentation_rendering()
seg = r.render()[..., 0]  # geom ids (-1 = none)
robot_geoms = [
    g
    for g in range(m.ngeom)
    if (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    ).startswith(ROBOT_PREFIXES)
]
# ground truth: each object's visual mesh geom (robosuite prefixes its names) and the world pose of its body, the
# OBJ's frame (MuJoCo re-centres the geom itself on the mesh's centre of mass and principal axes)
objects = {}
for n, o in scene.objects.items():
    vis = o.asset.name + "_visual_mesh"
    g = next(
        g
        for g in range(m.ngeom)
        if m.geom_dataid[g] >= 0 and m.mesh(m.geom_dataid[g]).name.endswith(vis)
    )
    b = m.geom_bodyid[g]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    front = scene.front(n)
    objects[n] = {
        "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
        "world_T_mesh": T.tolist(),
        "mesh": str((o.asset / "meshes" / f"{o.asset.name}_visual.obj").resolve()),
        "front": None if front is None else front.tolist(),
    }
    cv2.imwrite(str(out / f"mask_{n}.png"), (seg == g).astype(np.uint8) * 255)
arm = [
    j for j in range(m.njnt) if m.joint(j).name.startswith("robot0_")
]  # the UR5e's 6 joints
f = H / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(
    str(out / "depth_mm.png"),
    np.round(np.where(depth < 10, depth, 0) * 1000).astype(np.uint16),
)
cv2.imwrite(
    str(out / "robot_mask.png"), np.isin(seg, robot_geoms).astype(np.uint8) * 255
)
(out / "meta.json").write_text(
    json.dumps(
        {
            "scene": a.scene,
            "K": K.tolist(),
            "world_T_cam": world_T_cam.tolist(),
            "world_T_base": world_T_base.tolist(),
            "table_top_z": scene.table_top_z,
            "objects": objects,
            "joints": {
                "names": [m.joint(j).name for j in arm],
                "q": d.qpos[[m.jnt_qposadr[j] for j in arm]].tolist(),
            },
        },
        indent=1,
    )
)
print(
    "wrote",
    out,
    "| fovy",
    round(float(m.cam_fovy[cam]), 1),
    "| cam at",
    world_T_cam[:3, 3].round(3),
)
