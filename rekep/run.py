"""ReKep's pipeline (README "ReKep pipeline") offline, into one run directory, stage by stage; --from resumes.

uv run python -m rekep.run --out outputs/rekep/<name> --task "Pour tea into the cup"   # the bag in extracted/
uv run python -m rekep.run --out DIR --sim FRAME --task ...                            # a sim/capture.py frame
uv run python -m rekep.run --out DIR --run outputs/runs/<id> --task ...                # an OmniManip run's frame
uv run python -m rekep.run --out DIR --from program
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import litellm
import numpy as np
from constraint_generation import ConstraintGenerator
from path_solver import PathSolver
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation
from subgoal_solver import SubgoalSolver
from utils import (
    get_linear_interpolation_steps,
    load_functions_from_txt,
    spline_interpolate_poses,
)

from omnimanip import shared_perception as S
from omnimanip import viz
from omnimanip.config import ROOT, VLM_MODEL, VLM_REASONING
from omnimanip.run import dump, save

from . import core as C

STAGES = ("capture", "keypoints", "program", "preview")
PROMPT = ROOT / "third_party" / "ReKep" / "vlm_query" / "prompt_template.txt"
CAPTURE_SRV = ROOT / "ros2_ws" / "src" / "omnimanip_interfaces" / "srv" / "Capture.srv"


def run_packet(run):
    """An OmniManip run directory's frame: rgb, depth, packet, and the robot mask from its saved capture reply
    (calls/000_capture.bin, CDR), read without ROS. -> (rgb, depth (m), K, packet, robot mask)."""
    run = Path(run)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(
        get_types_from_msg(
            CAPTURE_SRV.read_text().split("---")[1],
            "omnimanip_interfaces/msg/CaptureReply",
        )
    )
    m = ts.deserialize_cdr(
        (run / "calls" / "000_capture.bin").read_bytes(),
        "omnimanip_interfaces/msg/CaptureReply",
    ).robot_mask  # type: ignore[attr-defined]
    rmask = (
        np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width] > 0
        if m.width
        else None
    )
    packet = json.loads((run / "packet.json").read_text())
    depth = (
        S.imread(run / "depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    )
    return rgb_of(run, "rgb.png"), depth, np.array(packet["K"]), packet, rmask


def rgb_of(d, name):
    return cv2.cvtColor(S.imread(d / name), cv2.COLOR_BGR2RGB)


def frame(d):
    """-> rgb, depth (m), K, packet, SAM masks (N x H x W bool), robot mask, T_base_cam."""
    packet = json.loads((d / "packet.json").read_text())
    depth = S.imread(d / "depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    masks = np.load(d / "masks.npz")["masks"]
    rmask = S.imread(d / "robot_mask.png", cv2.IMREAD_GRAYSCALE) > 127
    return (
        rgb_of(d, "rgb.png"),
        depth,
        np.array(packet["K"]),
        packet,
        masks,
        rmask,
        np.array(packet["t_base_cam"]["T"]),
    )


def masks_sheet(rgb, masks, max_ratio=0.5, cols=6, s=0.4):
    """One tile per SAM mask (the mask's id = its index in masks.npz, its % of the image; red = over max_ratio,
    which the keypoint proposer skips)."""
    tiles = []
    for i, m in enumerate(masks):
        t = (rgb * 0.35).astype(np.uint8)
        t[m] = rgb[m]
        t = cv2.resize(t, None, fx=s, fy=s)
        big = m.mean() > max_ratio
        cv2.putText(
            t,
            f"{i}  {100 * m.mean():.1f}%",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 64, 64) if big else (255, 255, 255),
            1,
        )
        tiles.append(
            cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        )
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    return np.vstack(
        [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    )


def capture(d, a):
    """Frame packet (OmniManip's stage 1), then every mask in the scene from SAM (paper A.5)."""
    rgb, depth, _, packet, rmask = (
        run_packet(a.run)
        if a.run
        else S.sim_packet(a.sim)
        if a.sim
        else S.bag_packet(a.extracted, a.start)
    )
    r = S.cli("sam", 120).call({"cmd": "everything", "rgb": rgb})
    save(rgb, d / "rgb.png")
    cv2.imwrite(str(d / "depth_mm.png"), np.round(depth * 1000).astype(np.uint16))
    np.savez_compressed(d / "masks.npz", masks=r["masks"], scores=r["scores"])
    save(masks_sheet(rgb, r["masks"]), d / "masks.png")
    rmask = np.zeros(depth.shape, bool) if rmask is None else rmask
    cv2.imwrite(str(d / "robot_mask.png"), rmask.astype(np.uint8) * 255)
    dump(packet, d / "packet.json")
    print(len(r["masks"]), "SAM masks")


def keypoints(d, a):
    """ReKep's keypoint proposal on the SAM masks; each keypoint keeps the mask it came from (its rigid group)."""
    rgb, depth, K, packet, masks, _, T = frame(d)
    cfg = C.config(*C.workspace(packet))
    kp = C.proposer(cfg)
    masks = masks & (depth > 0)  # pixels without depth have no point
    # k-means needs at least k points per mask; groups stay indices into masks.npz
    keep = np.flatnonzero(
        masks.sum((1, 2)) >= cfg["keypoint_proposer"]["num_candidates_per_mask"]
    )
    kps, img = kp.get_keypoints(rgb, C.points(depth, K, T), masks[keep])
    save(img, d / "keypoints.png")
    dump(
        {"positions": kps, "pixels": kp.pixels, "groups": keep[kp.groups]},
        d / "keypoints.json",
    )
    print(
        len(kps),
        "keypoints from",
        len(set(kp.groups.tolist())),
        "of",
        len(masks),
        "masks",
    )


def program(d, a):
    """ReKep's constraint generation: its prompt template and parser, our VLM."""
    png = (d / "keypoints.png").read_bytes()
    prompt = PROMPT.read_text().format(instruction=a.task)
    r = litellm.completion(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(png).decode()
                        },
                    },
                ],
            }
        ],
        reasoning_effort=VLM_REASONING,
    )
    assert isinstance(r, litellm.ModelResponse)
    out = r.choices[0].message.content  # type: ignore[union-attr]
    p = d / "program"
    p.mkdir(exist_ok=True)
    (p / "prompt.txt").write_text(prompt)
    (p / "output_raw.txt").write_text(out)
    ConstraintGenerator._parse_and_save_constraints(None, out, str(p))  # type: ignore[arg-type]
    kp = json.loads((d / "keypoints.json").read_text())
    meta = {
        "task": a.task,
        "init_keypoint_positions": kp["positions"],
        "num_keypoints": len(kp["positions"]),
        **ConstraintGenerator._parse_other_metadata(None, out),  # type: ignore[arg-type]
    }
    dump(meta, p / "metadata.json")
    print(json.dumps({k: v for k, v in meta.items() if k != "init_keypoint_positions"}))


def preview(d, a):
    """The program open loop on the captured frame: each stage's subgoal and path from ReKep's solvers, executed
    as main.py does (grasp: back off grasp_depth / 2, then in grasp_depth; the held object's keypoints ride with
    the end effector), with no tracking. preview.png: keypoints.png with each stage's path, its solved end-effector
    point (cross) and the gripper where the stage ends (at the grasp itself for a grasp stage), beside top and side
    views of the same."""
    rgb, depth, K, packet, masks, rmask, T_bc = frame(d)
    kp = json.loads((d / "keypoints.json").read_text())
    meta = json.loads((d / "program" / "metadata.json").read_text())
    lo, hi = C.workspace(packet)
    cfg = C.config(lo, hi)
    P = C.points(depth, K, T_bc)
    q = np.array(packet["joints"]["q"])
    ik = C.IK(q)
    sub = SubgoalSolver(cfg["subgoal_solver"], ik, q)
    path = PathSolver(cfg["path_solver"], ik, q)
    kps, px = np.array(kp["positions"], float), np.array(kp["pixels"])
    # the held object's mask: keypoints whose pixel lies in it ride with the end effector (rekep/DEVIATIONS.md)
    state = {"held": np.zeros_like(rmask)}
    on_held = lambda: state["held"][px[:, 0], px[:, 1]]
    grasp_cost = lambda i: float(
        not on_held()[i]
    )  # utils.get_callable_grasping_cost_fn
    fns = {
        s: {
            t: load_functions_from_txt(
                f
                if (f := d / "program" / f"stage{s}_{t}_constraints.txt").exists()
                else None,
                grasp_cost,
            )
            for t in ("subgoal", "path")
        }
        for s in range(1, meta["num_stages"] + 1)
    }
    pose, held_pts, report, drawn = C.ee_pose(q), np.zeros((0, 3)), [], []
    kps0 = kps.copy()
    img = rgb_of(d, "keypoints.png")
    cam_T_base = np.linalg.inv(T_bc)
    for s in range(1, meta["num_stages"] + 1):
        is_grasp, is_release = (
            meta["grasp_keypoints"][s - 1] != -1,
            meta["release_keypoints"][s - 1] != -1,
        )
        movable = np.r_[True, on_held()]
        keys = np.vstack([pose[:3], kps])
        scene = P[(depth > 0) & ~rmask & ~state["held"]]
        sdf = C.sdf_voxels(scene, lo, hi, cfg["main"]["sdf_voxel_size"])
        cpts = np.vstack([C.gripper_points(pose), held_pts])
        goal, gd = sub.solve(
            pose,
            keys,
            movable,
            fns[s]["subgoal"],
            fns[s]["path"],
            sdf,
            cpts,
            is_grasp,
            q,
            from_scratch=True,
        )
        solved = goal.copy()
        if is_grasp:
            goal[:3] += Rotation.from_quat(goal[3:]).as_matrix() @ [
                -cfg["main"]["grasp_depth"] / 2,
                0,
                0,
            ]
        ctrl, pd = path.solve(
            pose, goal, keys, movable, fns[s]["path"], sdf, cpts, q, from_scratch=True
        )
        full = np.vstack([pose, ctrl])
        n = get_linear_interpolation_steps(
            full[0],
            full[-1],
            cfg["main"]["interpolate_pos_step_size"],
            cfg["main"]["interpolate_rot_step_size"],
        )
        dense = spline_interpolate_poses(full, n)
        report.append(
            {
                "stage": s,
                "subgoal": goal,
                "subgoal_debug": gd,
                "path_control": ctrl,
                "path_debug": pd,
                "dense": dense,
            }
        )
        print(
            f"stage {s}: subgoal {gd['msg']} ({gd['solve_time']:.1f} s) | path {pd['msg']} ({pd['solve_time']:.1f} s)"
        )
        # execute: the held object's keypoints and points ride with the end effector
        D = C.mat(goal) @ np.linalg.inv(C.mat(pose))
        if state["held"].any():
            m = on_held()
            kps[m] = kps[m] @ D[:3, :3].T + D[:3, 3]
            held_pts = held_pts @ D[:3, :3].T + D[:3, 3]
        pose, q = goal, ik.solve(C.mat(goal), 100, q).cspace_position
        if is_grasp:  # main._execute_grasp_action, then the keypoint's object is held
            pose = pose.copy()
            pose[:3] += Rotation.from_quat(pose[3:]).as_matrix() @ [
                cfg["main"]["grasp_depth"],
                0,
                0,
            ]
            q = ik.solve(C.mat(pose), 100, q).cspace_position
            state["held"] = C.object_mask(
                masks,
                px[meta["grasp_keypoints"][s - 1]],
                cfg["keypoint_proposer"]["max_mask_ratio"],
            )
            held_pts = P[state["held"] & (depth > 0)]
        if is_release:
            state["held"], held_pts = np.zeros_like(rmask), np.zeros((0, 3))
        # the path, then (grasp) the move in to the grasp; the solved point; the gripper where the stage ends
        drawn.append(
            (
                np.vstack([dense[:, :3], pose[None, :3]]),
                solved[:3],
                C.gripper_points(pose)[::40],
            )
        )
    ok = depth > 0
    views = ortho_views(P[ok][::7], rgb[ok][::7], drawn, kps0)
    project = lambda X: viz.project(K, cam_T_base, X)
    img = np.hstack([draw(img, project, drawn, kps0, numbers=False), views])
    save(img, d / "preview.png")
    dump(report, d / "preview.json")


COLORS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 160, 0)]


def draw(img, f, drawn, kps, numbers=True):
    """Each stage's path (line), solved end-effector point (cross) and end gripper (dots), mapped to pixels by f;
    keypoint numbers if `numbers`."""
    for s, (path, solved, grip) in enumerate(drawn):
        c = COLORS[s % 4]
        cv2.polylines(img, [f(path).astype(int).reshape(-1, 1, 2)], False, c, 2)
        for x, y in f(grip).astype(int):
            cv2.circle(img, (int(x), int(y)), 1, c, -1)
        x, y = f(solved[None])[0].astype(int)
        cv2.drawMarker(img, (int(x), int(y)), c, cv2.MARKER_CROSS, 12, 2)
    if numbers:
        for i, (x, y) in enumerate(f(kps).astype(int)):
            cv2.putText(
                img,
                str(i),
                (int(x) - 4, int(y) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 0, 0),
                1,
            )
    return img


def ortho_views(P, colors, drawn, kps, size=480):
    """Top view (y right, x down: toward the camera) and side view (x right, z up) of the scene points, keypoints
    and the stages, at one scale around the keypoints and paths."""
    X = np.vstack([kps] + [p for p, _, _ in drawn])
    c, r = (X.min(0) + X.max(0)) / 2, (X.max(0) - X.min(0)).max() / 2 + 0.1
    out = []
    for name, h, v, vs, depth_axis in (("top", 1, 0, 1, 2), ("side", 0, 2, -1, 1)):
        f = lambda X, h=h, v=v, vs=vs: (
            np.c_[(X[:, h] - c[h] + r), (vs * (X[:, v] - c[v]) + r)] / (2 * r) * size
        )
        img = np.full((size, size, 3), 255, np.uint8)
        o = np.argsort(
            P[:, depth_axis] * (1 if name == "top" else -1)
        )  # nearer the viewer drawn last
        uv = f(P[o]).astype(int)
        m = np.all((uv >= 0) & (uv < size), axis=1)
        img[uv[m, 1], uv[m, 0]] = colors[o][m]
        draw(img, f, drawn, kps)
        cv2.putText(img, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        out.append(
            cv2.copyMakeBorder(
                img[:, :-2], 0, 0, 2, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        )
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="Pour tea into the cup")
    ap.add_argument("--extracted", default="extracted", help="extract_bag output")
    ap.add_argument(
        "--start", type=int, default=58, help="first bag frame of the packet"
    )
    ap.add_argument(
        "--sim", default=None, help="a sim/capture.py frame dir instead of the bag"
    )
    ap.add_argument(
        "--run",
        default=None,
        help="an OmniManip run dir: its captured frame instead of the bag",
    )
    ap.add_argument("--from", dest="start_stage", choices=STAGES, default=STAGES[0])
    a = ap.parse_args()
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for st in STAGES[STAGES.index(a.start_stage) :]:
        globals()[st](d, a)


if __name__ == "__main__":
    main()
