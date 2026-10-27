"""Shared perception: Albus's ur5e-manip-hardware vlm-port pipeline, driven from our code.

Same sidecar calls and payloads as his run_scene.py + ROS bridges (no ROS); every decision is made
by Albus's modules (third_party/ submodules), put on the path by setup.sh:
  frame_packet   N-frame median depth, table plane, gates
  scene_marks    label-free SAM 3 instances -> marks (set-of-mark image for the VLM)
  pose_decide    FoundationPose re-rank survivors -> Orient Anything yaw decision
  canonical_frame  z = table normal (revolution-refined), x = Orient Anything front
Session keys are omni_m<id> so his m<id> sidecar files and sessions are never touched.
"""

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from manip_bridge import canonical_frame as cf
from manip_bridge import frame_packet as fp
from manip_bridge import pose_decide
from manip_bridge import scene_marks as sm
from manip_bridge.render_asset import read_obj
from manip_bridge.zmq_client import SidecarClient
from scipy.spatial.transform import Rotation

from .config import HOST, SIDECARS

SSH_HOST = "coral"  # ~/.ssh/config alias
ANY6D_HOST_DIR = "/home/albusfang/ur5e-manip-hardware/any6d_runtime/outputs"  # mounted at /data/any6d
TOPICS = (
    "/camera/camera/color/image_raw",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/color/camera_info",
)  # recorded in the depth contract only
cli = lambda name, timeout_s: SidecarClient(
    f"tcp://{HOST}:{SIDECARS[name]}", int(timeout_s * 1000)
)


class Laps(dict):
    """Stage timer: laps("name") stores the seconds since the previous lap (or creation)."""

    def __init__(self):
        super().__init__()
        self.t = time.perf_counter()

    def __call__(self, name):
        now = time.perf_counter()
        self[name], self.t = now - self.t, now


def key(mark_id):
    return f"omni_{sm.obj_key(mark_id)}"


def imread(path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    im = cv2.imread(str(path), flags)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return np.asarray(im)


def _T(t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(t["qxyzw"]).as_matrix()
    T[:3, 3] = t["xyz"]
    return T


def bag_packet(extracted, start=0, n=fp.PACKET_PARAMS["n_frames"][0]):
    """run_scene's live capture, from an extract_bag.py dir: n synced frames from `start`, reference = middle.
    -> (rgb, median depth (m), K, packet, robot mask or None)."""
    d = Path(extracted)
    colors = sorted((d / "color").glob("*.png"))[start : start + n]
    depths = sorted((d / "aligned_depth").glob("*.png"))
    pairs = [
        (c, min(depths, key=lambda p: abs(float(p.stem) - float(c.stem))))
        for c in colors
    ]
    ref = pairs[len(pairs) // 2][0]
    rgb = cv2.cvtColor(imread(ref), cv2.COLOR_BGR2RGB)
    intr = json.loads((d / "intrinsics.json").read_text())
    K = np.array(intr["color"]["K"], float)
    meta = lambda i, t: {
        "topic": t,
        "frame_id": i["frame_id"],
        "width": i["width"],
        "height": i["height"],
    }
    contract = fp.depth_contract(
        {**meta(intr["color"], TOPICS[0]), "encoding": "rgb8"},
        {**meta(intr["aligned_depth"], TOPICS[1]), "encoding": "16UC1"},
        {
            **meta(intr["color"], TOPICS[2]),
            "distortion_model": intr["color"]["model"],
            "D": intr["color"]["D"],
        },
    )
    tf = {
        (t["parent"], t["child"]): t
        for t in json.loads((d / "tf_static.json").read_text())
    }
    T = (
        _T(tf["base_link", "camera_link"])
        @ _T(tf["camera_link", "camera_color_frame"])
        @ _T(tf["camera_color_frame", "camera_color_optical_frame"])
    )
    samples = []
    with open(d / "joint_states.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        names = [k[4:] for k in r if k.startswith("pos_")]
        s = fp.arm_sample(
            float(r["t"]),
            names,
            [float(r["pos_" + n]) for n in names],
            [float(r["vel_" + n]) for n in names],
        )
        if s is not None:
            samples.append(s)
    curobo = cli(
        "curobo", 180
    )  # first call loads the robot model; a failure is a packet degrade

    def robot_mask(depth, K, T, q, names):  # run_scene._robot_mask_fn
        return np.asarray(
            curobo.call(
                {
                    "cmd": "robot_mask",
                    "depth": np.asarray(depth, np.float32),
                    "intrinsics": np.asarray(K, np.float32),
                    "T_base_cam": np.asarray(T, np.float32),
                    "q": list(q),
                    "joint_names": list(names),
                }
            )["mask"],
            bool,
        )

    depth, packet, rmask = fp.build(
        depths=[
            imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3 for _, p in pairs
        ],
        K=K,
        source="live",
        contract=contract,
        stamps={
            "ref": float(ref.stem),
            "rgb": [float(c.stem) for c, _ in pairs],
            "depth": [float(p.stem) for _, p in pairs],
        },
        joints={"samples": samples, "topic": "/joint_states"},
        t_base_cam={
            "T": T,
            "source": "tf_static",
            "dt_s": 0.0,
            "base_frame": "base_link",
            "camera_frame": intr["color"]["frame_id"],
        },
        robot_mask_fn=robot_mask,
    )
    return rgb, depth, K, packet, rmask


def sim_packet(frame_dir):
    """The frame packet for a sim/capture.py frame: one rendered frame (exact depth, no median needed), T_base_cam
    from the sim, and the robot mask from MuJoCo segmentation in place of cuRobo's. Same fp.build gates and plane.
    -> (rgb, depth (m), K, packet, robot mask)."""
    d = Path(frame_dir)
    meta = json.loads((d / "meta.json").read_text())
    rgb = cv2.cvtColor(imread(d / "rgb.png"), cv2.COLOR_BGR2RGB)
    depth = imread(d / "depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
    rmask = imread(d / "robot_mask.png", cv2.IMREAD_GRAYSCALE) > 127
    K = np.array(meta["K"], float)
    h, w = depth.shape
    cam = {"frame_id": "sim_camera", "width": w, "height": h}
    depth, packet, rmask = fp.build(
        depths=[depth],
        K=K,
        source={"from_sim": str(d)},
        contract=fp.depth_contract(
            {**cam, "encoding": "rgb8"}, {**cam, "encoding": "16UC1"}, cam
        ),
        joints=meta["joints"],
        t_base_cam={
            "T": np.linalg.inv(meta["world_T_base"]) @ np.array(meta["world_T_cam"]),
            "source": "sim",
            "dt_s": 0.0,
            "base_frame": "robot0_base",
            "camera_frame": "sim_camera",
        },
        robot_mask_fn=lambda *_: rmask,
    )
    return rgb, depth, K, packet, rmask


def sim_gt(frame_dir, mark_id, rgb, mask, noise, rng):
    """Privileged register for a sim frame (stands in for register(); tests the later stages, not the pose
    models): the sim object whose ground-truth mask best overlaps the mark, its visual mesh, and its ground-truth
    pose with noise (translation N(0, noise[0]) m per axis, rotation vector N(0, noise[1]) deg per axis, in camera
    axes about the object's origin). The real crop's Orient Anything reading is taken as in register(), so
    canonicalize() sets the front as on the robot. -> record (register()'s keys used downstream)."""
    d = Path(frame_dir)
    meta = json.loads((d / "meta.json").read_text())
    iou = {
        n: (mask & (g := imread(d / f"mask_{n}.png", cv2.IMREAD_GRAYSCALE) > 127)).sum()
        / (mask | g).sum()
        for n in meta["objects"]
    }
    name = max(iou, key=lambda n: iou[n])
    o = meta["objects"][name]
    gt = np.linalg.inv(meta["world_T_cam"]) @ np.array(o["world_T_mesh"])
    Rn, tn = (
        Rotation.from_rotvec(np.radians(rng.normal(0, noise[1], 3))).as_matrix(),
        rng.normal(0, noise[0], 3),
    )
    T = gt.copy()
    T[:3, :3], T[:3, 3] = Rn @ gt[:3, :3], gt[:3, 3] + tn
    m8 = (mask > 0).astype(np.uint8)
    orient = pose_decide.sidecar_fns(cli("oriany", 120), None, key(mark_id))[0]
    real = pose_decide.real_crop(rgb, m8, pose_decide.crop_box(m8))
    return {
        "mark": mark_id,
        "sim_object": name,
        "mask_iou": iou[name],
        "cam_T_obj": T,
        "cam_T_obj_gt": gt,
        "noise": {
            "sigma_t_m": noise[0],
            "sigma_r_deg": noise[1],
            "R_noise_cam": Rn,
            "t_noise_cam": tn,
        },
        "final_mesh": o["mesh"],
        "decision": {
            "reason": "sim_gt",
            "hard_stop": False,
            "oa_real": pose_decide._oa(orient(real)),
        },
    }


def write_mark_set(rgb, kept, out_dir):
    """scene_marks.write_mark_set (vlm-port@a1883ac) with one fix: a matched instance leaves `unclaimed` by
    identity. The pinned list.remove compares the instance dicts with ==, which raises on their numpy masks
    whenever the mark reading order differs from SAM3's instance order."""
    from manip_bridge.marks_compat import build_marks

    out_dir = Path(out_dir)
    ms = build_marks(
        np.asarray(rgb, np.uint8),
        [it["mask"] for it in kept],
        "sam",
        out_dir,
        min_area=1,
    )
    if len(ms.ids()) != len(kept):
        raise RuntimeError(
            f"build_marks wrote {len(ms.ids())} marks for {len(kept)} instances"
        )
    id_map, unclaimed = {}, list(kept)
    for mid in ms.ids():
        m = ms.load_mask(mid)
        hit = [it for it in unclaimed if np.array_equal(it["mask"], m)]
        if len(hit) != 1:
            raise RuntimeError(
                f"mark {mid}: {len(hit)} instances with an identical mask"
            )
        unclaimed = [it for it in unclaimed if it is not hit[0]]
        id_map[mid] = hit[0]["idx"]
    by_idx = {it["idx"]: it for it in kept}
    (out_dir / "marks.sam3.json").write_text(
        json.dumps(
            {
                "note": "SAM3 provenance per mark. Never read by prompt builders: prompt strings "
                "reaching the VLM would turn marking into naming.",
                "marks": {
                    str(mid): {
                        "instance": i,
                        "prompt": by_idx[i]["prompt"],
                        "score": round(by_idx[i]["score"], 4),
                    }
                    for mid, i in id_map.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return ms, id_map


def marks(
    rgb,
    depth,
    K,
    packet,
    rmask,
    mark_dir,
    prompts=("object",),
    bg=("robot arm",),
    threshold=0.1,
):
    """run_scene's set-of-mark step. -> ({mark_id: HxW bool}, arm mask or None, report).
    mark_dir/marked.png is the image the VLM sees; bbox_IDs are the mark ids."""
    use_bg = rmask is None and bool(bg)
    ps = list(prompts) + (list(bg) if use_bg else [])
    res = cli("sam3", 120).call(
        {
            "cmd": "segment_multi",
            "rgb": rgb,
            "prompts": ps,
            "threshold": float(threshold),
        }
    )
    flat = [
        (p, m, s)
        for p, r in zip(ps, res["results"])
        for m, s in zip(r["masks"], r["scores"])
    ]
    inst, bgm, n_bg = sm.split_instances(
        [f[0] for f in flat],
        [f[1] for f in flat],
        [f[2] for f in flat],
        set(prompts),
        set(bg) if use_bg else set(),
        sm.values()["bg_score_min"],
    )
    arm, arm_source = (
        (rmask, "robot_mask")
        if rmask is not None
        else (
            (bgm if bgm is not None else np.zeros(depth.shape, bool))
            if use_bg
            else None,
            f"bg_prompt {list(bg)} ({n_bg} instance(s))" if use_bg else None,
        )
    )
    kept, report = sm.select_instances(
        inst,
        depth=depth,
        K=K,
        plane=packet["plane"],
        arm_mask=arm,
        arm_source=arm_source,
    )
    if not kept:
        raise fp.GateStop(
            "marks", f"no instance survived selection ({len(inst)} from SAM3)"
        )
    ms, id_map = write_mark_set(rgb, kept, mark_dir)
    report.pop("plane", None)
    report.update(
        id_map={str(k): v for k, v in id_map.items()},
        sam3={"prompts": ps, "threshold": threshold},
    )
    return {mid: ms.load_mask(mid) for mid in ms.ids()}, arm, report


def register(mark_id, rgb, depth, K, mask, arm, out_dir, decide=True):
    """run_scene step 3 for one mark: TRELLIS.2 canonical GLB -> Any6D metric final_mesh -> FoundationPose
    register on it with the re-rank + Orient Anything yaw decision (decide=False: the scorer's pick, as
    run_scene --no-decide). Copies final_mesh locally.
    -> record {cam_T_obj, fp_scorer, final_mesh (local path), decision, any6d, trellis2}."""
    k, out_dir = key(mark_id), Path(out_dir)
    dc = depth.copy()
    if arm is not None:
        dc[arm] = 0.0  # arm removed from the depth the pose models see
    m8 = (mask > 0).astype(np.uint8)
    laps = Laps()
    t2 = cli("trellis2", 400).call(
        {
            "op": "generate",
            "rgb": rgb,
            "seed": 42,
            "decimation_target": 500000,
            "texture_size": 2048,
            "output_name": f"{k}_{out_dir.name}",
            "return_glb": False,
            "mask": m8 * 255,
            # no depth/K: they only trigger the sidecar's metric-scale branch, which run_scene records but
            # never uses (Any6D does the scaling); the GLB does not depend on them (DEVIATIONS #15)
        }
    )
    laps("trellis2")
    a6 = cli("any6d", 1000).call(
        {
            "cmd": "estimate",
            "obj": k,
            "rgb": rgb,
            "depth": dc,
            "K": K,
            "mask": m8,
            "est_refine_iter": 5,
            "mesh": t2["glb_path"],
        }
    )
    laps("any6d")
    pose = cli("pose", 300)
    rep = pose.call(
        {
            "cmd": "register",
            "obj": k,
            "mesh": a6["mesh_path"],
            "rgb": rgb,
            "depth": dc,
            "K": K,
            "mask": m8,
            "est_refine_iter": 5,
            **(
                {"rerank": True, "survivor_crops": True, "return_all": True}
                if decide
                else {}
            ),
        }
    )
    laps("pose_register")
    orient, select, orient_rel = pose_decide.sidecar_fns(cli("oriany", 120), pose, k)
    if decide:
        T_sel, dec, _ = pose_decide.decide(
            rep, rgb, m8, orient, select, orient_rel=orient_rel
        )
    else:  # no yaw decision, but canonical_frame still needs the real crop's front (decide's framing)
        real = pose_decide.real_crop(rgb, m8, pose_decide.crop_box(m8))
        T_sel, dec = (
            None,
            {
                "reason": "decide_off",
                "hard_stop": False,
                "oa_real": pose_decide._oa(orient(real)),
            },
        )
    laps("yaw_decision")
    if dec["hard_stop"]:
        raise fp.GateStop("decision", f"{k}: {dec['reason']} -- {dec.get('note', '')}")
    # Any6D's per-object work dir holds the mesh with ITS texture; the top-level material_0.png is shared
    # and overwritten by every object, so it can belong to a different mesh.
    local = out_dir / "objects" / k / Path(a6["mesh_path"]).name
    local.parent.mkdir(parents=True, exist_ok=True)
    src = f"{ANY6D_HOST_DIR}/{k}"
    files = [local.name, "material.mtl", "material_0.png"]
    if Path(src).is_dir():  # running on the desktop itself (omnimanip.server)
        for f in files:
            shutil.copy(Path(src) / f, local.parent)
    else:
        subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes"]
            + [f"{SSH_HOST}:{src}/{f}" for f in files]
            + [str(local.parent)],
            check=True,
        )
    laps("mesh_copy")
    T = np.asarray(T_sel if T_sel is not None else rep["pose"], float)
    return {
        "mark": mark_id,
        "cam_T_obj": T,
        # FoundationPose's own scorer pick (hypotheses are scorer-sorted), before any re-rank / override
        "fp_scorer": np.asarray(rep["hypotheses"][0] if decide else rep["pose"], float),
        "final_mesh": str(local),
        "decision": dec,
        "any6d": {
            "cam_T_obj": np.asarray(a6["pose"], float),
            "extents": np.asarray(a6["extents"], float),
            "mesh": a6["mesh_path"],
        },
        "trellis2": {"glb": t2["glb_path"], "gen_time": t2["gen_time"]},
        "time_s": dict(laps),
    }


def canonicalize(rec, packet):
    """canonicalize_asset.py's frame step: R_body_from_canon (columns x=front, y=z cross x, z=up) for the
    tracked mesh. Returns (cam_T_can, canonical-frame vertices, faces, canonical_frame record)."""
    V, F = read_obj(Path(rec["final_mesh"]))
    R, crec = cf.canonical_frame(
        V,
        F,
        rec["cam_T_obj"],
        np.asarray(packet["plane"]["n"], float),
        rec["decision"].get("oa_real"),
    )
    T = rec["cam_T_obj"].copy()
    T[:3, :3] = T[:3, :3] @ R
    return T, cf.to_canonical(V, R), F, crec


def track(mark_id, rgb, depth, K, iters=2):
    """FoundationPose frame-to-frame tracking on the registered session (tracker_bridge payload)."""
    return np.asarray(
        cli("pose", 30).call(
            {
                "cmd": "track",
                "obj": key(mark_id),
                "rgb": rgb,
                "depth": depth,
                "K": K,
                "track_refine_iter": iters,
            }
        )["pose"],
        float,
    )


def release(mark_id):
    cli("pose", 5).call({"cmd": "release", "obj": key(mark_id)})
