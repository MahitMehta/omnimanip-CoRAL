"""OmniManip stage sidecar: one ZMQ command per pipeline stage (PIPELINE.md 2-16) for the ROS 2 bridge in
ros2_ws/src/omnimanip_ros, plus the motion commands its executor uses (map, plan, line, kinematics). Same wire
protocol as the CoRAL sidecars (REQ/REP, msgpack + msgpack_numpy; {"ok": False, "error"} on failure), so the
bridge reuses manip_bridge's SidecarClient. Stage 1 (capture) runs in the bridge, which owns the camera topics.

Objects travel as {"name", "mark_id", "mask", "mesh_path", "cam_T_obj", "R" (body <- canonical), "cam_T_can"};
images are RGB uint8, depth float32 metres, masks bool or uint8.

    uv run python -m omnimanip.server          # OMNIMANIP_PORT (default 5690)
"""

import json
import os
import traceback
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from . import constraints as C
from . import directions as D
from . import grasp as G
from . import motion, rrc, viz
from . import perception as P
from . import points as Pt
from . import shared_perception as S
from . import targetpose as TP

msgpack_numpy.patch()
PORT = int(os.getenv("OMNIMANIP_PORT", "5690"))


def _json(o):
    return json.dumps(
        o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)
    )


def _mesh(o):
    return Pt.textured_canonical_mesh(o["mesh_path"], o["R"])


def _fields(r, obj, element):
    return {
        "instruction": r["task"],
        "action": element["action"],
        "object": obj,
        "role": element["type"],
        "part": element["interaction_part"],
    }


def marks(r):
    m, arm, report = S.marks(
        r["rgb"],
        r["depth"],
        r["K"],
        json.loads(r["packet_json"]),
        r.get("robot_mask"),
        Path(r["run_dir"]) / "marks",
    )
    ids = sorted(m)
    return {
        "mark_ids": ids,
        "masks": [m[i] for i in ids],
        "marked": S.imread(Path(r["run_dir"]) / "marks" / "marked.png")[
            ..., ::-1
        ].copy(),
        "arm_mask": arm,
        "report_json": _json(report),
    }


def partition(r):
    return {"plan": P.partition(r["task"], r["marked"])}


def register(r):
    rec = S.register(
        r["mark_id"],
        r["rgb"],
        r["depth"],
        r["K"],
        r["mask"],
        r.get("arm_mask"),
        Path(r["run_dir"]),
        decide=r["decide"],
    )
    return {
        "mesh_path": rec["final_mesh"],
        "cam_T_obj": rec["cam_T_obj"],
        "decision_json": _json(rec["decision"]),
        "record_json": _json(rec),
    }


def canonicalize(r):
    rec = {
        "final_mesh": r["mesh_path"],
        "cam_T_obj": np.asarray(r["cam_T_obj"], float),
        "decision": json.loads(r["decision_json"]),
    }
    cam_T_can, Vc, F, crec = S.canonicalize(rec, {"plane": {"n": r["table_normal"]}})
    return {
        "R": np.asarray(crec["R_body_from_canon"], float),
        "cam_T_can": cam_T_can,
        "frame_json": _json(crec),
        "overlay": viz.canonical_overlay(
            r["rgb"], r["K"], {"object": (cam_T_can, Vc, F)}
        ),
    }


def points(r):
    o, el = r["object"], r["element"]
    res = Pt.propose(
        r["rgb"],
        r["depth"],
        r["K"],
        np.asarray(o["mask"]) > 0,
        o["cam_T_can"],
        o["mesh_path"],
        o["R"],
        el,
        **_fields(r, o["name"], el),
    )
    T = np.asarray(o["cam_T_can"])
    pts = np.asarray(res["points"], float).reshape(-1, 3)
    return {
        "kind": res["kind"],
        "points": pts,
        "points_camera": pts @ T[:3, :3].T + T[:3, 3],
        "images": res["images"],
        "vlm_json": _json(res["vlm"]),
    }


def directions(r):
    o, el = r["object"], r["element"]
    f = _fields(r, o["name"], el)
    res = D.sample(
        _mesh(o),
        np.asarray(o["cam_T_can"]),
        r["K"],
        tuple(r["size"]),
        f["instruction"],
        f["action"],
        f["object"],
        f["role"],
        f["part"],
        r["object_parts"],
    )
    return {
        k: res[k]
        for k in ("ranked_directions", "ranking", "captions", "expected", "image")
    }


def constraints(r):
    return {
        "constraints": C.candidates(
            r["active_point"],
            r["active_directions"],
            r["passive_point"],
            r["passive_direction"],
        )
    }


def rrc_(r):
    a, p, st = r["active"], r["passive"], r["stage"]
    fields = {
        "instruction": r["task"],
        "action": st["action"],
        "active": a["name"],
        "passive": p["name"],
        "active_part": r["active_part"],
    }
    K_list = [
        {k: np.asarray(v) if isinstance(v, list) else v for k, v in c.items()}
        for c in r["constraints"]
    ]
    acc, trace = rrc.plan(
        K_list,
        rrc.inpaint(r["rgb"], np.asarray(a["mask"]) > 0),
        r["K"],
        _mesh(a),
        np.asarray(a["cam_T_can"]),
        _mesh(p),
        np.asarray(p["cam_T_can"]),
        fields,
    )
    images = [t.pop("image") for t in trace]
    return {"accepted": acc, "images": images, "trace_json": _json(trace)}


def grasp(r):
    o = r["object"]
    T = np.asarray(o["cam_T_can"])
    rep = G.anygrasp(r["rgb"], r["depth"], r["K"], np.asarray(o["mask"]) > 0)
    if not rep["n"]:
        raise RuntimeError("AnyGrasp returned no grasps")
    i, T_g, phi, p_cam, _ = G.choose(rep, T, r["grasp_points"])
    return {
        "cam_T_grasp": T_g,
        "phi": phi,
        "width": float(rep["widths"][i]),
        "depth": float(rep["depths"][i]),
        "score": float(rep["scores"][i]),
        "n": int(rep["n"]),
        "index": i,
        "overlay": G.overlay(r["rgb"], r["K"], rep, p_cam, T_g),
    }


def targetpose(r):
    c = {
        k: np.asarray(v) if isinstance(v, list) else v
        for k, v in r["constraint"].items()
    }
    T_ee0, phi, T_p = (
        np.asarray(r[k], float) for k in ("cam_T_ee_start", "phi", "cam_T_passive")
    )
    obst = TP.obstacles(r["depth"], r["K"], np.asarray(r["obstacle_exclude"]) > 0)
    x0 = r.get("x0")  # executor: warm start at the previous tracker update's target
    T_ee, terms, ok = TP.solve(
        c, phi, T_ee0, T_p, obst, r["width"], None if x0 is None else np.asarray(x0)
    )
    out = {
        "ok": ok,
        "cam_T_ee_target": T_ee,
        "cam_T_active_target": T_ee @ phi,
        "terms": terms,
    }
    if not r.get("overlay", True):  # the executor's per-update re-solves
        return out
    a, p = r["active"], r["passive"]
    return {
        **out,
        "overlay": TP.overlay(
            r["rgb"],
            r["K"],
            np.asarray(a["mask"]) > 0,
            _mesh(a),
            _mesh(p),
            T_p,
            T_ee0,
            T_ee,
            phi,
            r["width"],
            r["depth_grasp"],
        ),
    }


# execution (stages 17-20; the hardware executor omnimanip_ros execute): base-frame motion on the cuRobo sidecar
def map_(r):
    motion.map_scene(r["depth"], r["K"], r["T_base_cam"], r["q"], r.get("exclude"))
    return {}


def plan(r):
    """cuRobo plan to a tool0 goal; with "held" ({mesh_path, tool0_T_obj}) the object rides on tool0 as spheres."""
    h, spheres = r.get("held"), None
    if h is not None:
        V, F = S.read_obj(Path(h["mesh_path"]))
        spheres = motion.attached_spheres(V, F, np.asarray(h["tool0_T_obj"], float))
    path, rep = motion.plan(r["q"], r["goal"], spheres, r.get("q_goal_ref"))
    return {"path": path, "reason": rep.get("reason", "")}


def line(r):
    path, reason = motion.line(r["q"], r["goal"])
    return {"path": path, "reason": reason}


def kinematics(r):
    """base_T_tool0 at q, and the grasp frame in tool0 for a grasp depth."""
    return {
        "base_T_tool0": motion.fk(r["q"]),
        "tool0_T_grasp": motion.tool0_T_grasp(r["depth"]),
    }


CMDS = {
    "marks": marks,
    "partition": partition,
    "register": register,
    "canonicalize": canonicalize,
    "points": points,
    "directions": directions,
    "constraints": constraints,
    "rrc": rrc_,
    "grasp": grasp,
    "targetpose": targetpose,
    "map": map_,
    "plan": plan,
    "line": line,
    "kinematics": kinematics,
}


def main():
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{PORT}")
    print(f"[omnimanip-server] listening on :{PORT}", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        cmd = req.pop("cmd", None)
        try:
            rep = {"ok": True} if cmd == "ping" else {"ok": True, **CMDS[cmd](req)}
        except Exception as e:  # noqa: BLE001 -- reported to the bridge as success=False
            traceback.print_exc()
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(rep, use_bin_type=True) or b"")


if __name__ == "__main__":
    main()
