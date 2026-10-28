"""OmniManip pipeline stages as ROS 2 services (README Pipeline numbering), the same shape as Albus's manip_bridge
nodes: a thin rclpy layer over a ZMQ sidecar (omnimanip.server, our Python 3.12 env), so ROS Humble's Python 3.10
never imports the pipeline.

    /omnimanip/capture             Capture            1     frame packet from the live topics (Albus's run_scene)
    /omnimanip/marks               Marks              2
    /omnimanip/partition_task      PartitionTask      3
    /omnimanip/register_object     RegisterObject     4-8
    /omnimanip/canonicalize        Canonicalize       9
    /omnimanip/propose_points      ProposePoints      11
    /omnimanip/propose_directions  ProposeDirections  12
    /omnimanip/build_constraints   BuildConstraints   13
    /omnimanip/rrc                 Rrc                14
    /omnimanip/select_grasp        SelectGrasp        15
    /omnimanip/target_pose         TargetPose         16

Env: OMNIMANIP_ADDR (tcp://127.0.0.1:5690); capture's robot mask uses CUROBO_ADDR (run_scene's variable); camera
topics as manip_bridge (RGB_TOPIC / DEPTH_TOPIC / INFO_TOPIC).

    ros2 run omnimanip_ros bridge
"""

import argparse
import json
import os

import numpy as np
import rclpy
from manip_bridge import frame_packet as fp
from manip_bridge.img import camera_info_to_K, image_to_depth_m, image_to_rgb
from manip_bridge.run_scene import SceneRunner, build_packet
from manip_bridge.zmq_client import SidecarClient
from omnimanip_interfaces import srv
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from . import conv

ADDR = os.environ.get("OMNIMANIP_ADDR", "tcp://127.0.0.1:5690")
TIMEOUT_S = {  # sidecar reply budgets: the VLM stages and registration take minutes
    "marks": 180,
    "partition": 300,
    "register": 1800,
    "canonicalize": 120,
    "points": 600,
    "directions": 600,
    "constraints": 30,
    "rrc": 1800,
    "grasp": 300,
    "targetpose": 300,
}


class Bridge(Node):
    def __init__(self, capture: SceneRunner):
        super().__init__("omnimanip_bridge")
        self.capture_node = capture
        group = ReentrantCallbackGroup()
        # one REQ socket per service: a socket strictly alternates send / recv
        self.cli = {k: SidecarClient(ADDR, t * 1000) for k, t in TIMEOUT_S.items()}
        for name, typ, fn in (
            ("capture", srv.Capture, self.capture),
            ("marks", srv.Marks, self.marks),
            ("partition_task", srv.PartitionTask, self.partition),
            ("register_object", srv.RegisterObject, self.register),
            ("canonicalize", srv.Canonicalize, self.canonicalize),
            ("propose_points", srv.ProposePoints, self.points),
            ("propose_directions", srv.ProposeDirections, self.directions),
            ("build_constraints", srv.BuildConstraints, self.constraints),
            ("rrc", srv.Rrc, self.rrc),
            ("select_grasp", srv.SelectGrasp, self.grasp),
            ("target_pose", srv.TargetPose, self.targetpose),
        ):
            self.create_service(
                typ, f"omnimanip/{name}", self._guard(fn), callback_group=group
            )
        self.get_logger().info(f"omnimanip services up; sidecar {ADDR}")

    def _guard(self, fn):
        """Any failure (sidecar error or timeout, a frame-packet gate) -> success=False with the reason."""

        def cb(req, res):
            try:
                fn(req, res)
                res.success = True
            except Exception as e:  # noqa: BLE001
                res.success, res.message = False, f"{type(e).__name__}: {e}"
                self.get_logger().error(f"{fn.__name__}: {res.message}")
            return res

        return cb

    def call(self, cmd, **payload):
        return self.cli[cmd].call({"cmd": cmd, **payload})

    # 1
    def capture(self, req, res):
        n = req.n_frames or fp.PACKET_PARAMS["n_frames"][0]
        args = argparse.Namespace(
            t_base_cam=None, from_run=None, frames=n, frame_timeout=req.timeout_s
        )
        _, rgb_msg, info, header, depth, packet, rmask = build_packet(
            self.capture_node, args, self.get_logger()
        )
        res.rgb, res.camera_info = rgb_msg, info
        res.depth = conv.depth_msg(depth, header)
        res.robot_mask = conv.mask_msg(rmask, header)
        res.packet_json = json.dumps(
            packet, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)
        )

    # 2
    def marks(self, req, res):
        r = self.call(
            "marks",
            rgb=image_to_rgb(req.rgb),
            depth=image_to_depth_m(req.depth),
            K=camera_info_to_K(req.camera_info),
            robot_mask=conv.mask_of(req.robot_mask),
            packet_json=req.packet_json,
            run_dir=req.run_dir,
        )
        h = req.rgb.header
        res.mark_ids = [int(i) for i in r["mark_ids"]]
        res.masks = [conv.mask_msg(m, h) for m in r["masks"]]
        res.marked = conv.rgb_msg(r["marked"], h)
        res.arm_mask = conv.mask_msg(r["arm_mask"], h)
        res.report_json = r["report_json"]

    # 3
    def partition(self, req, res):
        r = self.call("partition", task=req.task, marked=image_to_rgb(req.marked))
        res.plan = conv.plan_msg(r["plan"])

    # 4-8
    def register(self, req, res):
        r = self.call(
            "register",
            rgb=image_to_rgb(req.rgb),
            depth=image_to_depth_m(req.depth),
            K=camera_info_to_K(req.camera_info),
            mask=conv.mask_of(req.mask),
            arm_mask=conv.mask_of(req.arm_mask),
            mark_id=req.mark_id,
            decide=req.decide,
            run_dir=req.run_dir,
        )
        res.mesh_path, res.cam_t_obj = r["mesh_path"], conv.pose_of(r["cam_T_obj"])
        res.decision_json, res.record_json = r["decision_json"], r["record_json"]

    # 9
    def canonicalize(self, req, res):
        r = self.call(
            "canonicalize",
            rgb=image_to_rgb(req.rgb),
            K=camera_info_to_K(req.camera_info),
            mesh_path=req.mesh_path,
            cam_T_obj=conv.T_of(req.cam_t_obj),
            decision_json=req.decision_json,
            table_normal=list(req.table_normal),
        )
        res.r_body_from_canon = [float(x) for x in np.asarray(r["R"]).ravel()]
        res.cam_t_can, res.frame_json = conv.pose_of(r["cam_T_can"]), r["frame_json"]
        res.overlay = conv.rgb_msg(r["overlay"], req.rgb.header)

    # 11
    def points(self, req, res):
        r = self.call(
            "points",
            rgb=image_to_rgb(req.rgb),
            depth=image_to_depth_m(req.depth),
            K=camera_info_to_K(req.camera_info),
            task=req.task,
            object=conv.obj_of(req.object),
            element=conv.element_of(req.element),
        )
        res.kind = r["kind"]
        res.points = [conv.point(p) for p in r["points"]]
        res.points_camera = [conv.point(p) for p in r["points_camera"]]
        res.images = [conv.rgb_msg(g, req.rgb.header) for g in r["images"]]
        res.vlm_json = r["vlm_json"]

    # 12
    def directions(self, req, res):
        info = req.camera_info
        r = self.call(
            "directions",
            K=camera_info_to_K(info),
            size=[info.height, info.width],
            task=req.task,
            object=conv.obj_of(req.object),
            element=conv.element_of(req.element),
            object_parts=list(req.object_parts),
        )
        res.ranked_directions = [conv.vector(v) for v in r["ranked_directions"]]
        res.ranking = [int(i) for i in r["ranking"]]
        res.captions, res.expected = list(r["captions"]), r["expected"]
        res.image = conv.rgb_msg(r["image"], info.header)

    # 13
    def constraints(self, req, res):
        r = self.call(
            "constraints",
            active_point=conv.xyz(req.active_point),
            active_directions=[conv.xyz(v) for v in req.active_directions],
            passive_point=conv.xyz(req.passive_point),
            passive_direction=conv.xyz(req.passive_direction),
        )
        res.constraints = [conv.constraint_msg(c) for c in r["constraints"]]

    # 14
    def rrc(self, req, res):
        r = self.call(
            "rrc",
            rgb=image_to_rgb(req.rgb),
            K=camera_info_to_K(req.camera_info),
            task=req.task,
            stage={"action": req.stage.action},
            active_part=req.active_part,
            active=conv.obj_of(req.active),
            passive=conv.obj_of(req.passive),
            constraints=[conv.constraint_of(c) for c in req.constraints],
        )
        acc = r["accepted"]
        res.accepted = acc is not None
        if acc is not None:
            res.constraint = conv.constraint_msg(acc)
            res.cam_t_active_target = conv.pose_of(acc["cam_T_active_target"])
        res.images = [conv.rgb_msg(g, req.rgb.header) for g in r["images"]]
        res.trace_json = r["trace_json"]

    # 15
    def grasp(self, req, res):
        r = self.call(
            "grasp",
            rgb=image_to_rgb(req.rgb),
            depth=image_to_depth_m(req.depth),
            K=camera_info_to_K(req.camera_info),
            object=conv.obj_of(req.object),
            grasp_points=np.array([conv.xyz(p) for p in req.grasp_points]),
        )
        res.cam_t_grasp, res.grasp_t_obj = (
            conv.pose_of(r["cam_T_grasp"]),
            conv.pose_of(r["phi"]),
        )
        res.width, res.depth, res.score = r["width"], r["depth"], r["score"]
        res.n_candidates, res.index = int(r["n"]), int(r["index"])
        res.overlay = conv.rgb_msg(r["overlay"], req.rgb.header)

    # 16
    def targetpose(self, req, res):
        r = self.call(
            "targetpose",
            rgb=image_to_rgb(req.rgb),
            depth=image_to_depth_m(req.depth),
            K=camera_info_to_K(req.camera_info),
            obstacle_exclude=conv.mask_of(req.obstacle_exclude),
            constraint=conv.constraint_of(req.constraint),
            active=conv.obj_of(req.active),
            passive=conv.obj_of(req.passive),
            cam_T_passive=conv.T_of(req.passive.cam_t_can),
            cam_T_ee_start=conv.T_of(req.cam_t_ee_start),
            phi=conv.T_of(req.grasp_t_obj),
            width=req.grasp_width,
            depth_grasp=req.grasp_depth,
        )
        t = r["terms"]
        res.ok = bool(r["ok"])
        res.cam_t_ee_target = conv.pose_of(r["cam_T_ee_target"])
        res.cam_t_active_target = conv.pose_of(r["cam_T_active_target"])
        res.constraint_loss, res.collision_loss, res.path_loss = (
            float(t["constraint"]),
            float(t["collision"]),
            float(t["path"]),
        )
        res.overlay = conv.rgb_msg(r["overlay"], req.rgb.header)


def main():
    rclpy.init()
    cap = SceneRunner(
        argparse.Namespace()
    )  # camera / joints / TF subscriptions for capture
    node = Bridge(cap)
    ex = MultiThreadedExecutor()
    ex.add_node(cap)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
