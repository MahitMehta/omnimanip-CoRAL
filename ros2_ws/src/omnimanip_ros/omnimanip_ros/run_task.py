"""The OmniManip planning pipeline (stages 1-16) over the /omnimanip/* services: the ROS counterpart of
python -m omnimanip.run, writing the same run directory (plan.json,
objects.json, points/, directions/, constraints/, rrc/, grasp/, targetpose/).

The run directory must be visible at the same path to this node and to omnimanip.server (both write into it).

    ros2 run omnimanip_ros run_task --task "Pour tea into the cup" --out /path/to/outputs/runs [--no-decide]
    ros2 run omnimanip_ros run_task --task "Pour tea into the cup" --resume /path/to/outputs/runs/<stamp> [--no-decide]

Every successful service reply is saved in <run>/calls/ in call order; --resume replays them, so the run picks up
at the calls that failed or never ran (same capture, same plan). Resume with the flags the run started with.

--grasps-only stops after stage 15 for the plan's grasp targets (capture, marks, partition, their registration and
grasp dots, select_grasp), publishes the result latched and stays up until Ctrl-C. Frame = the capture's camera:
    /omnimanip/grasps/markers              MarkerArray  namespaces <object>/proposed (red), /filtered (green), /chosen (blue)
    /omnimanip/grasps/<object>/proposed    PoseArray    every AnyGrasp proposal, before our filters, score-sorted
    /omnimanip/grasps/<object>/filtered    PoseArray    those passing real gripper, holds and near the dots
    /omnimanip/grasps/<object>/chosen      PoseArray    the heatmap pick that cuRobo reaches (empty if none)
    /omnimanip/grasps/cloud                PointCloud2  the captured frame
In RViz, use Durability: Transient Local on all of them.
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message, serialize_message
from geometry_msgs.msg import Point, PoseArray
from manip_bridge.grasp_viz_node import backproject, gripper_lines, xyzrgb_to_cloud
from manip_bridge.img import image_to_depth_m, image_to_rgb
from omnimanip_interfaces import srv
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

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
    def __init__(self, calls_dir):
        super().__init__("omnimanip_run_task")
        self.cli = {
            k: self.create_client(t, f"omnimanip/{k}") for k, t in SERVICES.items()
        }
        self.calls, self.n = calls_dir, 0  # saved replies, in call order

    def slot(self):
        """The next position in the call order; concurrent calls take theirs before they start, so it stays fixed."""
        self.n += 1
        return self.n - 1

    def call(self, name, slot=None, **fields):
        """Blocking call; raises on an unavailable service or success=False. A reply saved by an earlier attempt of
        this run (same position in the call order) is replayed instead."""
        saved = self.calls / f"{self.slot() if slot is None else slot:03d}_{name}.bin"
        if saved.exists():
            self.get_logger().info(f"{name}: replayed from {saved.name}")
            return deserialize_message(saved.read_bytes(), SERVICES[name].Response)
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
        saved.write_bytes(serialize_message(res))
        return res


def _parallel(fn, items):
    """fn over items at once, results in order. Waits for them all, then raises the first failure; the replies of
    those that succeeded are saved, so --resume only repeats the rest."""
    if not items:
        return []
    with ThreadPoolExecutor(len(items)) as pool:
        return list(pool.map(fn, items))


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


def _points_overlay(rgb, info, pts, path):
    """Every stage's points on the frame, as omnimanip.run draws them: grasp dots red, the others yellow."""
    im, K = image_to_rgb(rgb).copy(), np.asarray(info.k, float).reshape(3, 3)
    for p in pts:
        for x, y, z in p["points_camera"]:
            u, v = int(round(K[0, 0] * x / z + K[0, 2])), int(round(K[1, 1] * y / z + K[1, 2]))
            cv2.circle(im, (u, v), 6, (255, 0, 0) if p["kind"] == "grasp" else (255, 255, 0), -1)
            cv2.circle(im, (u, v), 6, (0, 0, 0), 1)
    cv2.imwrite(str(path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def _select_grasps(node, plan_msg, rgb, depth, info, models, pts, packet, run, top_down_deg, ideal_grasp, allow_no_pick=False):
    """15 for every grasp stage's object: writes grasp/ (picked grasps only) -> {object: SelectGrasp reply}."""
    grasps = {}
    for s in plan_msg.stages:
        if s.action != "grasp":
            continue
        name = s.passive_object
        if name not in models:
            node.get_logger().warn(f"grasp: skipped {name}, it has no mark")
            continue
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
            top_down_deg=top_down_deg,
            ideal_grasp=ideal_grasp,
            allow_no_pick=allow_no_pick,
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
            if g.index >= 0
        ],
        run / "grasp" / "grasp.json",
    )
    return grasps


def _publish_grasps(node, rgb, depth, info, grasps):
    """The latched topics of --grasps-only (module docstring). Zero stamps, so RViz needs no TF at the capture's time."""
    hdr = Header(frame_id=rgb.header.frame_id)
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = lambda typ, topic, msg: node.create_publisher(typ, topic, qos).publish(msg)

    d, im = image_to_depth_m(depth), image_to_rgb(rgb)
    valid = np.zeros(d.shape, bool)
    valid[::2, ::2] = True  # every 2nd pixel each way
    pts, (v, u) = backproject(d, np.asarray(info.k, float).reshape(3, 3), valid & (d > 0.05) & (d < 3.0))
    pub(PointCloud2, "/omnimanip/grasps/cloud", xyzrgb_to_cloud(pts, im[v, u].astype(np.float32) / 255.0, hdr))

    markers = MarkerArray()
    for name, g in grasps.items():
        n = len(g.candidates)
        sets = {
            "proposed": (range(n), (1.0, 0.1, 0.1, 0.35), 0.001),
            "filtered": ([i for i in range(n) if g.candidate_ok[i]], (0.1, 1.0, 0.1, 1.0), 0.002),
            "chosen": ([g.index] if g.index >= 0 else [], (0.1, 0.4, 1.0, 1.0), 0.005),
        }
        for kind, (idx, (cr, cg, cb, ca), thick) in sets.items():
            poses = [g.candidates[i] for i in idx]
            pub(PoseArray, f"/omnimanip/grasps/{re.sub(r'[^0-9A-Za-z_]', '_', name)}/{kind}", PoseArray(header=hdr, poses=poses))
            m = Marker(header=hdr, ns=f"{name}/{kind}", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
            m.pose.orientation.w, m.scale.x = 1.0, thick
            m.color.r, m.color.g, m.color.b, m.color.a = cr, cg, cb, ca
            for i in idx:
                T = conv.T_of(g.candidates[i])
                lines = gripper_lines(T[:3, 3], T[:3, :3], g.candidate_widths[i], g.candidate_depths[i])
                m.points.extend(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in lines)
            markers.markers.append(m)
        node.get_logger().info(
            f"{name}: {n} proposed, {sum(g.candidate_ok)} filtered, "
            + (f"chosen #{g.index}" if g.index >= 0 else "none chosen")
        )
    pub(MarkerArray, "/omnimanip/grasps/markers", markers)


def run(
    node, task, run, decide=True, mesh_cache=False, top_down_deg=0.0, ideal_grasp=False, grasps_only=False,
    low_res_mesh=False,
):
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
    cal = json.loads(mk.report_json).get("calibration", {})
    if not cal.get("ok", True):
        node.get_logger().warn(
            "calibration check: " + "; ".join(cal["problems"]) + f" (see {run / 'calibration_check.png'})"
        )
    masks = dict(zip(mk.mark_ids, mk.masks))
    plan_msg = node.call("partition_task", task=task, marked=mk.marked).plan
    plan = conv.plan_of(plan_msg)
    _dump(plan, run / "plan.json")

    # 4-9 register + canonicalize each task object that is a mark
    # (concurrently with --mesh-cache: FoundationPose queues them, the waits overlap the CPU work)
    objects, models = {}, {}
    targets = {s.passive_object for s in plan_msg.stages if s.action == "grasp"}

    def register(job):
        o, reg_slot, can_slot = job
        reg = node.call(
            "register_object",
            reg_slot,
            rgb=rgb,
            depth=depth,
            camera_info=info,
            mask=masks[o.mark_id],
            arm_mask=mk.arm_mask,
            mark_id=o.mark_id,
            decide=decide,
            run_dir=str(run),
            object_name=o.name if mesh_cache else "",
            low_res_mesh=low_res_mesh,
        )
        can = node.call(
            "canonicalize",
            can_slot,
            rgb=rgb,
            camera_info=info,
            mesh_path=reg.mesh_path,
            cam_t_obj=reg.cam_t_obj,
            decision_json=reg.decision_json,
            table_normal=[float(x) for x in packet["plane"]["n"]],
        )
        return o, reg, can

    jobs = [
        (o, node.slot(), node.slot())
        for o in plan_msg.objects
        if o.mark_id in masks and (not grasps_only or o.name in targets)
    ]
    for o, reg, can in _parallel(register, jobs) if mesh_cache else map(register, jobs):
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

    # 11 points, 12 directions, all concurrently (per element of a registered object)
    pts, dirs, parts = {}, {}, {}
    for e in plan_msg.elements:
        parts.setdefault(e.object, []).append(e.part)
    elements = [
        e
        for e in plan_msg.elements
        if e.object in models and (not grasps_only or e.action == "grasp")  # the gripper has no mesh
    ]
    calls = []
    for e in elements:
        calls.append((
            "propose_points",
            node.slot(),
            {"rgb": rgb, "depth": depth, "camera_info": info, "task": task, "object": models[e.object], "element": e},
        ))
        if e.action != "grasp":
            calls.append((
                "propose_directions",
                node.slot(),
                {"camera_info": info, "task": task, "object": models[e.object], "element": e, "object_parts": parts[e.object]},
            ))
    replies = iter(_parallel(lambda c: node.call(c[0], c[1], **c[2]), calls))
    for e in elements:
        p = next(replies)
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
        d = next(replies)
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
    _points_overlay(rgb, info, pts.values(), run / "points" / "points_overlay.png")
    _dump(list(dirs.values()), run / "directions" / "directions.json")

    if grasps_only:
        grasps = _select_grasps(
            node, plan_msg, rgb, depth, info, models, pts, packet, run, top_down_deg, ideal_grasp, allow_no_pick=True
        )
        if not grasps:
            raise RuntimeError("the plan has no grasp stage on a marked object: name what to grasp in --task")
        _publish_grasps(node, rgb, depth, info, grasps)
        return

    # 13 constraints, 14 RRC (non-grasp stages)
    stages, accepted = [], {}
    part = {(e.object, e.action): e.part for e in plan_msg.elements}
    for s in plan_msg.stages:
        A, Pv = s.active_object, s.passive_object
        if s.action == "grasp":
            continue
        if A not in models or Pv not in models:  # e.g. "place on the table": a surface has no mark, mesh or point
            missing = [o for o in (A, Pv) if o not in models]
            node.get_logger().warn(f"{s.action}: skipped, {', '.join(missing)} has no mark (a surface, or not detected)")
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
        if not r.accepted:
            node.get_logger().warn(f"{s.action}: RRC NOT accepted: none of its {len(trace)} checks succeeded")
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
    if not any(accepted.values()):
        raise RuntimeError(
            "RRC accepted no constraint, nothing to execute (to ask again: delete calls/*_rrc.bin and --resume)"
        )

    # 15 grasp (grasp stages)
    grasps = _select_grasps(node, plan_msg, rgb, depth, info, models, pts, packet, run, top_down_deg, ideal_grasp)

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
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="runs root; the run goes to <out>/<timestamp>")
    where.add_argument("--resume", help="a run dir to continue from its first failed or missing call")
    ap.add_argument(
        "--no-decide",
        action="store_true",
        help="FoundationPose's scorer pick, no re-rank / Orient Anything yaw decision",
    )
    ap.add_argument(
        "--mesh-cache",
        action="store_true",
        help="reuse each object's cached mesh when it matches (skips TRELLIS.2 + Any6D); cache new ones (omnimanip.mesh_cache)",
    )
    ap.add_argument(
        "--low-res-mesh",
        action="store_true",
        help="save VRAM and time on meshes: implies --mesh-cache; new meshes use low-resolution TRELLIS.2 settings "
        "(shared_perception.LOW_RES_TRELLIS, needs Trellis2Server-mahit) and are not cached",
    )
    ap.add_argument(
        "--top-down-deg",
        type=float,
        default=0.0,
        help="steer AnyGrasp to approaches within this angle of straight down the table's normal (0: off)",
    )
    ap.add_argument(
        "--ideal-grasp",
        action="store_true",
        help="objects with a hand-set grasp (grasp.IDEAL_GRASPS, e.g. the teapot's handle) use it instead of AnyGrasp; needs --mesh-cache",
    )
    ap.add_argument(
        "--grasps-only",
        action="store_true",
        help="stop after the grasp stage and publish the proposed, filtered and chosen grasps for RViz (see the module docstring)",
    )
    a, ros_args = ap.parse_known_args()
    a.mesh_cache = a.mesh_cache or a.low_res_mesh
    if a.ideal_grasp and not a.mesh_cache:
        ap.error("--ideal-grasp needs --mesh-cache: its poses are set on the cached meshes")
    out = Path(a.resume) if a.resume else Path(a.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "calls").mkdir(parents=True, exist_ok=True)
    rclpy.init(args=ros_args)
    node = Runner(out / "calls")
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.get_logger().info(f"run dir {out}" + (" (resumed)" if a.resume else ""))
    try:
        run(
            node, a.task, out, decide=not a.no_decide, mesh_cache=a.mesh_cache, top_down_deg=a.top_down_deg,
            ideal_grasp=a.ideal_grasp, grasps_only=a.grasps_only, low_res_mesh=a.low_res_mesh,
        )
        if a.grasps_only:
            node.get_logger().info("grasps latched on /omnimanip/grasps/*; Ctrl-C to exit")
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
