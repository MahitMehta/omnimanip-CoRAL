"""omnimanip_interfaces messages <-> the omnimanip.server payloads (numpy arrays, plain dicts in the run-dir
formats: plan.json's keys, constraints as in omnimanip/constraints.py). Image decoding is manip_bridge.img's."""

import numpy as np
from geometry_msgs.msg import Point, Pose, Vector3
from manip_bridge.img import image_to_mono, mono_to_image
from omnimanip_interfaces.msg import (
    Constraint,
    InteractionElement,
    ObjectModel,
    Stage,
    TaskObject,
    TaskPlan,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image


def T_of(p: Pose) -> np.ndarray:
    T = np.eye(4)
    q = p.orientation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def pose_of(T) -> Pose:
    T = np.asarray(T, float)
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
    return p


def xyz(m) -> np.ndarray:
    return np.array([m.x, m.y, m.z], float)


def point(v) -> Point:
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def vector(v) -> Vector3:
    return Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def rgb_msg(img, header) -> Image:
    m = Image(
        header=header,
        height=img.shape[0],
        width=img.shape[1],
        encoding="rgb8",
        step=img.shape[1] * 3,
    )
    m.data = np.ascontiguousarray(img, np.uint8).tobytes()
    return m


def depth_msg(d, header) -> Image:
    m = Image(
        header=header,
        height=d.shape[0],
        width=d.shape[1],
        encoding="32FC1",
        step=d.shape[1] * 4,
    )
    m.data = np.ascontiguousarray(d, np.float32).tobytes()
    return m


def mask_of(msg: Image):
    """mono8 -> bool HxW, or None for an empty (0x0) image."""
    return image_to_mono(msg) > 0 if msg.width else None


def mask_msg(m, header) -> Image:
    return (
        Image(header=header)
        if m is None
        else mono_to_image(np.asarray(m, bool), header)
    )


def obj_of(o: ObjectModel) -> dict:
    return {
        "name": o.name,
        "mark_id": o.mark_id,
        "mask": mask_of(o.mask),
        "mesh_path": o.mesh_path,
        "cam_T_obj": T_of(o.cam_t_obj),
        "R": np.asarray(o.r_body_from_canon, float).reshape(3, 3),
        "cam_T_can": T_of(o.cam_t_can),
    }


def obj_msg(
    name, mark_id, mask, mesh_path, cam_T_obj, R, cam_T_can, header
) -> ObjectModel:
    return ObjectModel(
        name=name,
        mark_id=int(mark_id),
        mask=mask_msg(mask, header),
        mesh_path=mesh_path,
        cam_t_obj=pose_of(cam_T_obj),
        r_body_from_canon=[float(x) for x in np.asarray(R, float).ravel()],
        cam_t_can=pose_of(cam_T_can),
    )


def element_of(e: InteractionElement) -> dict:
    """-> plan.json's "element" dict."""
    return {
        "action": e.action,
        "type": e.role,
        "interaction_part": e.part,
        "part_visible_and_tangible": e.part_visible_and_tangible,
    }


def plan_msg(p: dict) -> TaskPlan:
    return TaskPlan(
        task_name=p["Task Name"],
        reasoning=p["Think Step by Step"],
        objects=[
            TaskObject(name=o["object_ID"], mark_id=int(o["bbox_ID"]))
            for o in p["objects"]
        ],
        stages=[
            Stage(
                action=s["action"],
                active_object=s["active_object_ID"],
                passive_object=s["passive_object_ID"],
            )
            for s in p["stages"]
        ],
        elements=[
            InteractionElement(
                object=e["object_ID"],
                action=e["element"]["action"],
                role=e["element"]["type"],
                part=e["element"]["interaction_part"],
                part_visible_and_tangible=bool(
                    e["element"]["part_visible_and_tangible"]
                ),
            )
            for e in p["interaction_elements"]
        ],
    )


def plan_of(p: TaskPlan) -> dict:
    """-> plan.json."""
    return {
        "Task Name": p.task_name,
        "Think Step by Step": p.reasoning,
        "objects": [{"object_ID": o.name, "bbox_ID": o.mark_id} for o in p.objects],
        "stages": [
            {
                "action": s.action,
                "active_object_ID": s.active_object,
                "passive_object_ID": s.passive_object,
            }
            for s in p.stages
        ],
        "interaction_elements": [
            {"object_ID": e.object, "element": element_of(e)} for e in p.elements
        ],
    }


def constraint_of(c: Constraint) -> dict:
    rank = int(c.rank) if c.rank.isdigit() else c.rank
    return {
        "rank": rank,
        "active_point": xyz(c.active_point),
        "active_axis": xyz(c.active_axis),
        "passive_point": xyz(c.passive_point),
        "passive_axis": xyz(c.passive_axis),
        "distance_m": float(c.distance_m),
    }


def constraint_msg(c: dict) -> Constraint:
    return Constraint(
        rank=str(c["rank"]),
        active_point=point(c["active_point"]),
        active_axis=vector(c["active_axis"]),
        passive_point=point(c["passive_point"]),
        passive_axis=vector(c["passive_axis"]),
        distance_m=float(c["distance_m"]),
    )
