"""Extract a ROS 2 bag into plain files: PNG images, camera intrinsics, TF, robot state CSVs, URDF.

uv run python -m omnimanip.extract_bag bag/<stamp> extracted
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

bag, out = Path(sys.argv[1]), Path(sys.argv[2])
IMAGES = {
    "/camera/camera/color/image_raw": "color",
    "/camera/camera/aligned_depth_to_color/image_raw": "aligned_depth",
    "/camera/camera/depth/image_rect_raw": "depth",
}
INFOS = {
    "/camera/camera/color/camera_info": "color",
    "/camera/camera/aligned_depth_to_color/camera_info": "aligned_depth",
    "/camera/camera/depth/camera_info": "depth",
}
for d in IMAGES.values():
    (out / d).mkdir(parents=True, exist_ok=True)

stamp = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
tf = lambda x: {
    "parent": x.header.frame_id,
    "child": x.child_frame_id,
    "t": stamp(x.header),
    "xyz": [
        x.transform.translation.x,
        x.transform.translation.y,
        x.transform.translation.z,
    ],
    "qxyzw": [
        x.transform.rotation.x,
        x.transform.rotation.y,
        x.transform.rotation.z,
        x.transform.rotation.w,
    ],
}
intr, tf_static, tf_dyn, rows = (
    {},
    [],
    [],
    {"joint_states": [], "tcp_pose": [], "wrench": []},
)

with AnyReader([bag]) as r:
    for c, _, raw in r.messages():
        m: Any = r.deserialize(raw, c.msgtype)
        if c.topic in IMAGES:
            img = np.frombuffer(
                m.data, np.uint8 if m.encoding == "rgb8" else np.uint16
            ).reshape(m.height, m.width, -1)
            if m.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(out / IMAGES[c.topic] / f"{stamp(m.header):.6f}.png"), img
            )  # depth: uint16, millimetres
            if (
                m.encoding == "16UC1"
            ):  # viewable preview: fixed 0-1.5 m turbo colormap, invalid (0) = black
                vis = cv2.applyColorMap(
                    (np.clip(img / 1500, 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_TURBO,
                )
                vis[img[..., 0] == 0] = 0
                (out / f"{IMAGES[c.topic]}_vis").mkdir(exist_ok=True)
                cv2.imwrite(
                    str(out / f"{IMAGES[c.topic]}_vis" / f"{stamp(m.header):.6f}.png"),
                    vis,
                )
        elif c.topic in INFOS and INFOS[c.topic] not in intr:
            intr[INFOS[c.topic]] = {
                "frame_id": m.header.frame_id,
                "width": m.width,
                "height": m.height,
                "K": m.k.reshape(3, 3).tolist(),
                "D": m.d.tolist(),
                "model": m.distortion_model,
            }
        elif c.topic == "/tf_static":
            tf_static += [tf(x) for x in m.transforms]
        elif c.topic == "/tf":
            tf_dyn += [tf(x) for x in m.transforms]
        elif c.topic == "/robot_description":
            (out / "robot.urdf").write_text(m.data)
        elif c.topic == "/joint_states":
            rows["joint_states"].append(
                {
                    "t": stamp(m.header),
                    **{f"pos_{n}": p for n, p in zip(m.name, m.position)},
                    **{f"vel_{n}": v for n, v in zip(m.name, m.velocity)},
                }
            )
        elif c.topic == "/tcp_pose_broadcaster/pose":
            p, q = m.pose.position, m.pose.orientation
            rows["tcp_pose"].append(
                {
                    "t": stamp(m.header),
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "qx": q.x,
                    "qy": q.y,
                    "qz": q.z,
                    "qw": q.w,
                }
            )
        elif c.topic == "/force_torque_sensor_broadcaster/wrench":
            f, k = m.wrench.force, m.wrench.torque
            rows["wrench"].append(
                {
                    "t": stamp(m.header),
                    "fx": f.x,
                    "fy": f.y,
                    "fz": f.z,
                    "tx": k.x,
                    "ty": k.y,
                    "tz": k.z,
                }
            )

(out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
(out / "tf_static.json").write_text(json.dumps(tf_static, indent=2))
(out / "tf.json").write_text(json.dumps(tf_dyn))
for name, rs in rows.items():
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rs[0].keys())
        w.writeheader()
        w.writerows(rs)
