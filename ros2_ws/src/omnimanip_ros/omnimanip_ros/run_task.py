"""The OmniManip planning pipeline (stages 1-16) over the /omnimanip/* services: the ROS counterpart of
python -m omnimanip.run, writing the same run directory (plan.json,
objects.json, points/, directions/, constraints/, rrc/, grasp/, targetpose/).

The run directory must be visible at the same path to this node and to omnimanip.server (both write into it).

    ros2 run omnimanip_ros run_task --task "Pour tea into the cup" --out /path/to/outputs/runs [--no-decide]
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from manip_bridge.img import image_to_depth_m, image_to_rgb
from omnimanip_interfaces import srv
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from . import conv

UR_JOINTS = [  # as execute_node
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
SERVICES = {
    "capture": srv.Capture,
    "marks": srv.Marks,
    "partition_task": srv.PartitionTask,
    "register_object": srv.RegisterObject,
    "canonicalize": srv.Canonicalize,
    "propose_points": srv.ProposePoints,
    "propose_directions": srv.ProposeDirections,
    "build_constraints": srv.BuildConstraints,
    "rrc": srv.Rrc,
    "select_grasp": srv.SelectGrasp,
    "target_pose": srv.TargetPose,
}


class Runner(Node):
    def __init__(self):
        super().__init__("omnimanip_run_task")
        self.cli = {
            k: self.create_client(t, f"omnimanip/{k}") for k, t in SERVICES.items()
        }

    def call(self, name, **fields):
        """Blocking call; raises on an unavailable service or success=False."""
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"{cli.srv_name} not available (is the omnimanip bridge up?)"
            )
        t0 = time.time()
        res = cli.call(SERVICES[name].Request(**fields))
        self.get_logger().info(
            f"{name}: {'ok' if res.success else 'FAILED'} ({time.time() - t0:.1f}s) {res.message}"
        )
        if not res.success:
            raise RuntimeError(f"{name}: {res.message}")
        return res


def _dump(o, path):
    path.write_text(
        json.dumps(
            o,
            indent=1,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
    )


def _save(msg, path):
    cv2.imwrite(str(path), cv2.cvtColor(image_to_rgb(msg), cv2.COLOR_RGB2BGR))


def run(node, task, run, decide=True):
    for d in (
        "marks",
        "objects",
        "points",
        "directions",
        "constraints",
        "rrc",
        "grasp",
        "targetpose",
    ):
        (run / d).mkdir(parents=True, exist_ok=True)

    # 1 capture
    cap = node.call("capture")
    rgb, depth, info, h = cap.rgb, cap.depth, cap.camera_info, cap.rgb.header
    packet = json.loads(cap.packet_json)
    (run / "packet.json").write_text(cap.packet_json)
    _save(rgb, run / "rgb.png")
    cv2.imwrite(
        str(run / "depth_mm.png"),
        np.round(image_to_depth_m(depth) * 1000).astype(np.uint16),
    )

    # 2 marks, 3 stage partition
    mk = node.call(
        "marks",
        rgb=rgb,
        depth=depth,
        camera_info=info,
        robot_mask=cap.robot_mask,
        packet_json=cap.packet_json,
        run_dir=str(run),
    )
    (run / "marks_report.json").write_text(mk.report_json)
    masks = dict(zip(mk.mark_ids, mk.masks))
    plan_msg = node.call("partition_task", task=task, marked=mk.marked).plan
    plan = conv.plan_of(plan_msg)
    _dump(plan, run / "plan.json")

    # 4-9 register + canonicalize each task object that is a mark
    objects, models = {}, {}
    for o in plan_msg.objects:
        if o.mark_id not in masks:
            continue
        reg = node.call(
            "register_object",
            rgb=rgb,
            depth=depth,
            camera_info=info,
            mask=masks[o.mark_id],
            arm_mask=mk.arm_mask,
            mark_id=o.mark_id,
            decide=decide,
            run_dir=str(run),
        )
        can = node.call(
            "canonicalize",
            rgb=rgb,
            camera_info=info,
            mesh_path=reg.mesh_path,
            cam_t_obj=reg.cam_t_obj,
            decision_json=reg.decision_json,
            table_normal=[float(x) for x in packet["plane"]["n"]],
        )
        _save(can.overlay, run / f"canonical_overlay_{o.name}.png")
        R = np.asarray(can.r_body_from_canon).reshape(3, 3)
        objects[o.name] = {
            **json.loads(reg.record_json),
            "cam_T_can": conv.T_of(can.cam_t_can),
            "canonical_frame": json.loads(can.frame_json),
        }
        models[o.name] = conv.obj_msg(
            o.name,
            o.mark_id,
            conv.mask_of(masks[o.mark_id]),
            reg.mesh_path,
            conv.T_of(reg.cam_t_obj),
            R,
            conv.T_of(can.cam_t_can),
            h,
        )
    _dump(objects, run / "objects.json")

    # 11 points, 12 directions (per element of a registered object)
    pts, dirs, parts = {}, {}, {}
    for e in plan_msg.elements:
        parts.setdefault(e.object, []).append(e.part)
    for e in plan_msg.elements:
        if e.object not in models:  # the gripper has no mesh
            continue
        p = node.call(
            "propose_points",
            rgb=rgb,
            depth=depth,
            camera_info=info,
            task=task,
            object=models[e.object],
            element=e,
        )
        tag = f"{e.action}_{e.object}_{e.part.replace(' ', '_')}"
        for i, g in enumerate(p.images):
            _save(g, run / "points" / f"{tag}_{i}.png")
        P = np.array([conv.xyz(q) for q in p.points])
        pts[e.object, e.action] = {
            "object": e.object,
            "stage_action": e.action,
            "role": e.role,
            "part": e.part,
            "kind": p.kind,
            "points": P,
            "point": P[0],
            "points_camera": [conv.xyz(q) for q in p.points_camera],
            "vlm": json.loads(p.vlm_json),
        }
        if e.action == "grasp":
            continue
        d = node.call(
            "propose_directions",
            camera_info=info,
            task=task,
            object=models[e.object],
            element=e,
            object_parts=parts[e.object],
        )
        _save(d.image, run / "directions" / f"{e.action}_{e.object}.png")
        dirs[e.object, e.action] = {
            "object": e.object,
            "stage_action": e.action,
            "role": e.role,
            "part": e.part,
            "ranking": list(d.ranking),
            "ranked_directions": [conv.xyz(v) for v in d.ranked_directions],
            "captions": list(d.captions),
            "expected": d.expected,
        }
    _dump(list(pts.values()), run / "points" / "points.json")
    _dump(list(dirs.values()), run / "directions" / "directions.json")

    # 13 constraints, 14 RRC (non-grasp stages)
    stages, accepted = [], {}
    part = {(e.object, e.action): e.part for e in plan_msg.elements}
    for s in plan_msg.stages:
        A, Pv = s.active_object, s.passive_object
        if s.action == "grasp":
            continue
        K = node.call(
            "build_constraints",
            active_point=conv.point(pts[A, s.action]["point"]),
            active_directions=[
                conv.vector(v) for v in dirs[A, s.action]["ranked_directions"]
            ],
            passive_point=conv.point(pts[Pv, s.action]["point"]),
            passive_direction=conv.vector(dirs[Pv, s.action]["ranked_directions"][0]),
        ).constraints
        stages.append(
            {
                "action": s.action,
                "active": A,
                "passive": Pv,
                "K": [conv.constraint_of(c) for c in K],
            }
        )
        r = node.call(
            "rrc",
            rgb=rgb,
            camera_info=info,
            task=task,
            stage=s,
            active_part=part[A, s.action],
            active=models[A],
            passive=models[Pv],
            constraints=K,
        )
        trace = json.loads(r.trace_json)
        for i, (t, g) in enumerate(zip(trace, r.images)):
            _save(
                g,
                run
                / "rrc"
                / f"{s.action}_{i:02d}_C{t['rank']}_{t['reply']['Reasonable']}.png",
            )
        acc = (
            {
                **conv.constraint_of(r.constraint),
                "cam_T_active_target": conv.T_of(r.cam_t_active_target),
            }
            if r.accepted
            else None
        )
        accepted[s.action] = (s, r.constraint) if r.accepted else None
        stages[-1]["rrc"] = {"accepted": acc, "trace": trace}
    _dump(
        [{k: v for k, v in st.items() if k != "rrc"} for st in stages],
        run / "constraints" / "constraints.json",
    )
    _dump(
        [
            {
                "action": st["action"],
                "active": st["active"],
                "passive": st["passive"],
                **st["rrc"],
            }
            for st in stages
        ],
        run / "rrc" / "rrc.json",
    )

    # 15 grasp (grasp stages)
    grasps = {}
    for s in plan_msg.stages:
        if s.action != "grasp":
            continue
        name = s.passive_object
        g = node.call(
            "select_grasp",
            rgb=rgb,
            depth=depth,
            camera_info=info,
            object=models[name],
            grasp_points=[conv.point(q) for q in pts[name, "grasp"]["points"]],
            table_plane=[float(x) for x in packet["plane"]["n"]] + [float(packet["plane"]["d"])],
            base_t_cam=[float(x) for x in np.asarray(packet["t_base_cam"]["T"], float).ravel()],
            q=[float(packet["joints"]["q"][packet["joints"]["names"].index(n)]) for n in UR_JOINTS],
        )
        _save(g.overlay, run / "grasp" / f"grasp_overlay_{name}.png")
        grasps[name] = g
    _dump(
        [
            {
                "object": n,
                "n": g.n_candidates,
                "n_feasible": g.n_feasible,
                "index": g.index,
                "score": g.score,
                "width": g.width,
                "depth": g.depth,
                "cam_T_grasp": conv.T_of(g.cam_t_grasp),
                "phi_grasp_T_obj": conv.T_of(g.grasp_t_obj),
            }
            for n, g in grasps.items()
        ],
        run / "grasp" / "grasp.json",
    )

    # 16 target pose (stages with an accepted constraint)
    exclude = np.any([conv.mask_of(m) for m in masks.values()], axis=0)
    results = []
    for act, a in accepted.items():
        if a is None:
            node.get_logger().warn(f"{act}: no accepted constraint (RRC failed)")
            continue
        s, c = a
        g = grasps[s.active_object]
        t = node.call(
            "target_pose",
            rgb=rgb,
            depth=depth,
            camera_info=info,
            obstacle_exclude=conv.mask_msg(exclude, h),
            constraint=c,
            active=models[s.active_object],
            passive=models[s.passive_object],
            cam_t_ee_start=g.cam_t_grasp,
            grasp_t_obj=g.grasp_t_obj,
            grasp_width=g.width,
            grasp_depth=g.depth,
        )
        _save(t.overlay, run / "targetpose" / f"{act}_target.png")
        results.append(
            {
                "action": act,
                "active": s.active_object,
                "passive": s.passive_object,
                "cam_T_ee_target": conv.T_of(t.cam_t_ee_target),
                "terms": {
                    "constraint": t.constraint_loss,
                    "collision": t.collision_loss,
                    "path": t.path_loss,
                },
                "ok": t.ok,
                "cam_T_ee_start": conv.T_of(g.cam_t_grasp),
                "cam_T_active_target": conv.T_of(t.cam_t_active_target),
            }
        )
    _dump(results, run / "targetpose" / "targetpose.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--out", required=True, help="runs root; the run goes to <out>/<timestamp>"
    )
    ap.add_argument(
        "--no-decide",
        action="store_true",
        help="FoundationPose's scorer pick, no re-rank / Orient Anything yaw decision",
    )
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Runner()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    out = Path(a.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    node.get_logger().info(f"run dir {out}")
    try:
        run(node, a.task, out, decide=not a.no_decide)
    finally:
        ex.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
