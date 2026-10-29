"""The offline pipeline (README Pipeline stages 1-16) into one run directory, stage by stage; each stage reads what the
earlier ones wrote, so --from resumes a run at any stage.

    uv run python -m omnimanip.run --out outputs/<name> [--start 58]          # the bag in extracted/
    uv run python -m omnimanip.run --out DIR --sim FRAME --gt 0.005 2         # a sim frame, GT perception
    uv run python -m omnimanip.run --out DIR --from rrc                       # redo stages rrc..targetpose
"""

import argparse
import json
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import trimesh

from . import constraints as C
from . import directions as D
from . import grasp as G
from . import perception as P
from . import points as Pt
from . import rrc as R
from . import shared_perception as S
from . import targetpose as TP
from . import viz


def dump(o, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            o,
            indent=1,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
    )


def save(im, path):
    cv2.imwrite(str(path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


class Run:
    """A run directory: the frame and the earlier stages' outputs, loaded on first use."""

    def __init__(self, path):
        self.dir = Path(path)

    def json(self, rel):
        return json.loads((self.dir / rel).read_text())

    def out(self, sub):
        (self.dir / sub).mkdir(parents=True, exist_ok=True)
        return self.dir / sub

    @cached_property
    def rgb(self):
        return cv2.cvtColor(S.imread(self.dir / "rgb.png"), cv2.COLOR_BGR2RGB)

    @cached_property
    def depth(self):
        return (
            S.imread(self.dir / "depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
            / 1000
        )

    @cached_property
    def K(self):
        return np.array(self.json("packet.json")["K"])

    @cached_property
    def plan(self):
        return self.json("plan.json")

    @cached_property
    def objects(self):
        return self.json("objects.json")

    def mask(self, name):
        m = self.objects[name]["mark"]
        return (
            S.imread(self.dir / "marks" / f"mask_{m}.png", cv2.IMREAD_GRAYSCALE) > 127
        )

    def mesh(self, name):
        o = self.objects[name]
        return Pt.textured_canonical_mesh(
            o["final_mesh"], o["canonical_frame"]["R_body_from_canon"]
        )

    def T(self, name):
        return np.array(self.objects[name]["cam_T_can"])

    def part(self, name, action):
        return next(
            e["element"]["interaction_part"]
            for e in self.plan["interaction_elements"]
            if e["object_ID"] == name and e["element"]["action"] == action
        )


def perception(r, a):
    """1-10: frame packet -> marks -> stage partition -> register + canonicalize the task objects."""
    rng = np.random.default_rng(a.seed)
    laps = S.Laps()
    rgb, depth, K, packet, rmask = (
        S.sim_packet(a.sim) if a.sim else S.bag_packet(a.extracted, a.start)
    )
    laps("capture")
    marks, arm, mrep = S.marks(rgb, depth, K, packet, rmask, r.dir / "marks")
    laps("marks")
    print("marks:", sorted(marks), "| packet degrades:", packet["degrades"])
    marked = cv2.cvtColor(S.imread(r.dir / "marks" / "marked.png"), cv2.COLOR_BGR2RGB)
    plan = P.partition(a.task, marked)
    laps("stage_partition")
    print(json.dumps(plan, indent=1))
    dump(packet, r.dir / "packet.json")
    dump(mrep, r.dir / "marks_report.json")
    dump(plan, r.dir / "plan.json")
    save(rgb, r.dir / "rgb.png")
    cv2.imwrite(str(r.dir / "depth_mm.png"), np.round(depth * 1000).astype(np.uint16))

    objects, posed = {}, {}
    for o in plan["objects"]:
        mid = o["bbox_ID"]
        if mid not in marks:  # -1 (gripper, table) or an id that is not a mark
            continue
        if a.gt:
            rec = S.sim_gt(a.sim, mid, rgb, marks[mid], a.gt, rng)
            print(
                o["object_ID"],
                "mark",
                mid,
                "->",
                rec["sim_object"],
                f"(mask IoU {rec['mask_iou']:.2f})",
            )
        else:
            rec = S.register(
                mid, rgb, depth, K, marks[mid], arm, r.dir, decide=not a.no_decide
            )
        laps(f"register:{o['object_ID']}")
        cam_T_can, Vc, F, crec = S.canonicalize(rec, packet)
        trimesh.Trimesh(Vc, F, process=False).export(
            r.out(f"objects/{S.key(mid)}") / "canonical_mesh.ply"
        )
        laps(f"canonicalize:{o['object_ID']}")
        posed[o["object_ID"]] = (cam_T_can, Vc, F)
        objects[o["object_ID"]] = {
            **rec,
            "cam_T_can": cam_T_can,
            "canonical_frame": crec,
        }
        print(
            o["object_ID"],
            "mark",
            mid,
            "| yaw decision:",
            rec["decision"]["reason"],
            "| up",
            crec["up"]["source"],
            f"{crec['up']['angle_to_table_deg']:.1f} deg to table | azimuth",
            crec["azimuth"]["route"],
            "accepted" if crec["azimuth"]["accepted"] else "NOT accepted",
        )
    dump(objects, r.dir / "objects.json")
    save(viz.canonical_overlay(rgb, K, posed), r.dir / "canonical_overlay.png")
    masks = {n: marks[o["mark"]] for n, o in objects.items()}
    acc, stats = viz.mesh_accuracy(rgb, depth, K, posed, masks)
    save(acc, r.dir / "mesh_accuracy.png")
    # the same check for each pose source on the same mesh: FoundationPose's scorer pick, the final (decided)
    # pose, and Any6D's own estimate
    for src in ("cam_T_obj_gt",) if a.gt else ("fp_scorer", "any6d"):
        for n, o in objects.items():
            T = np.array(o[src]["cam_T_obj"] if src == "any6d" else o[src])
            T[:3, :3] = T[:3, :3] @ np.array(o["canonical_frame"]["R_body_from_canon"])
            _, s = viz.mesh_accuracy(
                rgb, depth, K, {n: (T, *posed[n][1:])}, {n: masks[n]}
            )
            stats[n][src] = s[n]
    dump(stats, r.dir / "mesh_accuracy.json")
    print(stats)
    laps("overlays")
    dump(
        {
            "stages": dict(laps),
            "register": {n: o.get("time_s") for n, o in objects.items()},
        },
        r.dir / "timings.json",
    )
    print(
        "timings (s):",
        {k: round(v, 1) for k, v in laps.items()},
        "| total",
        round(sum(laps.values()), 1),
    )


def points(r, a):
    """11 (O4): one point (or grasp dots) per interaction element, in each object's canonical frame."""
    out, task = r.out("points"), r.plan["Task Name"]
    results, overlay = [], r.rgb.copy()
    for e in r.plan["interaction_elements"]:
        name, el = e["object_ID"], e["element"]
        if (
            name not in r.objects
        ):  # the gripper (and anything not registered) has no mesh
            continue
        o, T = r.objects[name], r.T(name)
        res = Pt.propose(
            r.rgb,
            r.depth,
            r.K,
            r.mask(name),
            T,
            o["final_mesh"],
            o["canonical_frame"]["R_body_from_canon"],
            el,
            instruction=task,
            action=el["action"],
            object=name,
            role=el["type"],
            part=el["interaction_part"],
        )
        kind, pts = res["kind"], res["points"]
        tag = f"{el['action']}_{name}_{el['interaction_part'].replace(' ', '_')}"
        for i, g in enumerate(res.pop("images")):
            save(g, out / f"{tag}_{i}.png")
        results.append(
            {
                "object": name,
                "stage_action": el["action"],
                "role": el["type"],
                "part": el["interaction_part"],
                "kind": kind,
                **res,
            }
        )
        for u, v in np.round(viz.project(r.K, T, pts)).astype(int):
            cv2.circle(
                overlay,
                (int(u), int(v)),
                6,
                (255, 0, 0) if kind == "grasp" else (255, 255, 0),
                -1,
            )
            cv2.circle(overlay, (int(u), int(v)), 6, (0, 0, 0), 1)
        print(
            f"{name:8s} {el['action']:6s} {el['interaction_part']:22s} {kind:9s}",
            res.get("dot", res.get("dots")),
            np.round(pts, 3).tolist(),
        )
    dump(results, out / "points.json")
    save(overlay, out / "points_overlay.png")


def directions(r, a):
    """12 (O5): rank the six canonical axes of every non-grasp stage's active and passive objects."""
    out = r.out("directions")
    parts = {}
    for e in r.plan["interaction_elements"]:
        parts.setdefault(e["object_ID"], []).append(e["element"]["interaction_part"])
    results = []
    for e in r.plan["interaction_elements"]:
        name, el = e["object_ID"], e["element"]
        if (
            el["action"] == "grasp" or name not in r.objects
        ):  # grasps use the grasp model; gripper has no mesh
            continue
        res = D.sample(
            r.mesh(name),
            r.T(name),
            r.K,
            r.rgb.shape[:2],
            r.plan["Task Name"],
            el["action"],
            name,
            el["type"],
            el["interaction_part"],
            parts[name],
        )
        save(res.pop("image"), out / f"{el['action']}_{name}.png")
        results.append(
            {
                "object": name,
                "stage_action": el["action"],
                "role": el["type"],
                "part": el["interaction_part"],
                **res,
            }
        )
        print(
            f"{name} ({el['type']}, {el['action']}): ranking {res['ranking']}, best {res['ranked_directions'][0].tolist()}"
        )
        print("   expected:", res["expected"])
        for cap in res["captions"]:
            print("  ", cap)
    dump(results, out / "directions.json")


def constraints(r, a):
    """13 (O6): the constraint list K_i of every non-grasp stage; d is set later by RRC."""
    pts, dirs = r.json("points/points.json"), r.json("directions/directions.json")
    pick = lambda rows, obj, act: next(
        x for x in rows if x["object"] == obj and x["stage_action"] == act
    )
    stages = []
    for s in r.plan["stages"]:
        act, A, Pv = s["action"], s["active_object_ID"], s["passive_object_ID"]
        if (
            act == "grasp"
        ):  # grasp stages: grasp model + heatmap (O8), no constraint list
            continue
        pa, pp = pick(pts, A, act), pick(pts, Pv, act)
        K = C.candidates(
            pa["point"],
            pick(dirs, A, act)["ranked_directions"],
            pp["point"],
            pick(dirs, Pv, act)["ranked_directions"][0],
        )
        print(
            f"stage {act}: active {A} ({pa['part']}), passive {Pv} ({pp['part']}), N = {len(K)}"
        )
        for c in K:
            print(
                f"  C({c['rank']}) active axis {c['active_axis'].tolist()} passive axis {c['passive_axis'].tolist()}"
                f" | residual at current poses {C.residual(c, r.T(A), r.T(Pv)):.2f} | {C.describe(c)}"
            )
        stages.append({"action": act, "active": A, "passive": Pv, "K": K})
    dump(stages, r.dir / "constraints" / "constraints.json")


def rrc(r, a):
    """14 (O7): Algorithm 1 over each stage's constraint list."""
    out, results = r.out("rrc"), []
    for st in r.json("constraints/constraints.json"):
        act, A, Pv = st["action"], st["active"], st["passive"]
        K_list = [
            {k: np.asarray(v) if isinstance(v, list) else v for k, v in c.items()}
            for c in st["K"]
        ]
        fields = {
            "instruction": r.plan["Task Name"],
            "action": act,
            "active": A,
            "passive": Pv,
            "active_part": r.part(A, act),
        }
        accepted, trace = R.plan(
            K_list,
            R.inpaint(r.rgb, r.mask(A)),
            r.K,
            r.mesh(A),
            r.T(A),
            r.mesh(Pv),
            r.T(Pv),
            fields,
        )
        for i, t in enumerate(trace):
            save(
                t.pop("image"),
                out / f"{act}_{i:02d}_C{t['rank']}_{t['reply']['Reasonable']}.png",
            )
        print(
            f"{act}: TASK FAILED (list exhausted)"
            if accepted is None
            else f"{act}: ACCEPTED C({accepted['rank']}) | {C.describe(accepted)}"
        )
        results.append(
            {
                "action": act,
                "active": A,
                "passive": Pv,
                "accepted": accepted,
                "trace": trace,
            }
        )
    dump(results, out / "rrc.json")


def grasp(r, a):
    """15 (O8): AnyGrasp candidates for every grasp stage's object, filtered by its grasp points' heatmap."""
    out, pts = r.out("grasp"), r.json("points/points.json")
    results, im = [], r.rgb.copy()
    for s in r.plan["stages"]:
        if s["action"] != "grasp":
            continue
        name = s["passive_object_ID"]
        o, mask = r.objects[name], r.mask(name)
        gp = next(x for x in pts if x["object"] == name and x["kind"] == "grasp")
        ok = None  # sim: mesh_grasps already checks its own grasps against the table and object
        if a.sim_mesh_grasps:
            V, F = S.read_obj(Path(o["final_mesh"]))
            rep = G.mesh_grasps(
                V, F, np.array(o["cam_T_obj_gt"]), r.json("packet.json")["plane"]
            )
        else:
            rep = G.anygrasp(r.rgb, r.depth, r.K, mask)
        if not rep["n"]:
            print(f"{name}: AnyGrasp returned no grasps")
            continue
        if not a.sim_mesh_grasps:
            ok = G.feasible(rep, r.depth, r.K, mask, r.json("packet.json")["plane"])
            print(f"{name}: {ok.sum()} of {rep['n']} grasps pass the real-gripper check")
            if not ok.any():
                continue
        i, T_g, phi, p_cam, heat = G.choose(rep, r.T(name), gp["points"], ok)
        dist = np.linalg.norm(rep["translations"][i] - p_cam, axis=1).min()
        print(
            f"{name}: {rep['n']} grasps; chosen #{i} score {rep['scores'][i]:.3f} (best score {rep['scores'][0]:.3f}),"
            f" width {rep['widths'][i] * 100:.1f} cm, {dist * 100:.1f} cm from the nearest grasp point"
        )
        results.append(
            {
                "object": name,
                "n": rep["n"],
                "n_feasible": None if ok is None else int(ok.sum()),
                "index": i,
                "score": rep["scores"][i],
                "width": rep["widths"][i],
                "depth": rep["depths"][i],
                "cam_T_grasp": T_g,
                "phi_grasp_T_obj": phi,
                "heat": heat[i],
                "grasp_points_cam": p_cam,
            }
        )
        im = G.overlay(im, r.K, rep, p_cam, i, r.T(name)[:3, 2])

        # heatmap image: the object's observed surface coloured by heat (0 = none, 1 = max over the object), candidate
        # centres coloured the same way (white ring = kept by the 0.5 x max filter), chosen grasp as a large ring
        obj, (v, u) = G.backproject(
            r.depth, r.K, mask & (r.depth > 0.05) & (r.depth < 3.0)
        )
        hm = G.heat(obj, p_cam)
        hm_img = r.rgb.copy() // 2
        jet = cv2.applyColorMap(
            (255 * hm / max(hm.max(), 1e-9)).astype(np.uint8)[:, None], cv2.COLORMAP_JET
        )
        hm_img[v, u] = jet[:, 0, ::-1]
        hc = heat[: rep["n"]] / max(heat.max(), 1e-9)
        uv = np.round(
            viz.project(r.K, np.eye(4), np.asarray(rep["translations"]))
        ).astype(int)
        for t, hv in zip(uv, hc):
            col = cv2.applyColorMap(np.array([[255 * hv]], np.uint8), cv2.COLORMAP_JET)[
                0, 0, ::-1
            ].tolist()
            cv2.circle(hm_img, (int(t[0]), int(t[1])), 4, col, -1)
            if hv >= G.KEEP:
                cv2.circle(hm_img, (int(t[0]), int(t[1])), 6, (255, 255, 255), 1)
        c0 = np.round(viz.project(r.K, T_g, np.zeros((1, 3)))).astype(int)[0]
        cv2.circle(hm_img, (int(c0[0]), int(c0[1])), 11, (255, 255, 255), 2)
        save(hm_img, out / f"grasp_heatmap_{name}.png")
    dump(results, out / "grasp.json")
    save(im, out / "grasp_overlay.png")


def targetpose(r, a):
    """16 (O9): the target EE pose meeting each stage's accepted constraint through the grasp's Phi."""
    out, results = r.out("targetpose"), []
    grasps = {g["object"]: g for g in r.json("grasp/grasp.json")}
    obst = TP.obstacles(r.depth, r.K, np.any([r.mask(n) for n in r.objects], axis=0))
    for st in r.json("rrc/rrc.json"):
        A, Pv, c = st["active"], st["passive"], st["accepted"]
        if c is None:
            print(f"{st['action']}: no accepted constraint (RRC failed)")
            continue
        c = {k: np.asarray(v) if isinstance(v, list) else v for k, v in c.items()}
        g = grasps[A]
        T_ee0, phi, T_p = (
            np.array(g["cam_T_grasp"]),
            np.array(g["phi_grasp_T_obj"]),
            r.T(Pv),
        )
        T_ee, terms, ok = TP.solve(c, phi, T_ee0, T_p, obst, g["width"])
        print(
            f"{st['action']}: {'OK' if ok else 'FAILED (rho above tolerance)'} | terms "
            + ", ".join(f"{k} {v:.4f}" for k, v in terms.items())
            + f" | EE moves {np.linalg.norm(T_ee[:3, 3] - T_ee0[:3, 3]) * 100:.1f} cm"
        )
        results.append(
            {
                "action": st["action"],
                "active": A,
                "passive": Pv,
                "cam_T_ee_target": T_ee,
                "terms": terms,
                "ok": ok,
                "cam_T_ee_start": T_ee0,
                "cam_T_active_target": T_ee @ phi,
            }
        )
        im = TP.overlay(
            r.rgb,
            r.K,
            r.mask(A),
            r.mesh(A),
            r.mesh(Pv),
            T_p,
            T_ee0,
            T_ee,
            phi,
            g["width"],
            g["depth"],
        )
        save(im, out / f"{st['action']}_target.png")
    dump(results, out / "targetpose.json")


STAGES = {
    f.__name__: f
    for f in (perception, points, directions, constraints, rrc, grasp, targetpose)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the run directory")
    ap.add_argument(
        "--from", dest="start_stage", default="perception", choices=list(STAGES)
    )
    ap.add_argument(
        "--to", dest="end_stage", default="targetpose", choices=list(STAGES)
    )
    ap.add_argument(
        "--task", default="Pour tea into the cup", help="perception: the instruction"
    )
    ap.add_argument("--extracted", default="extracted", help="extract_bag output")
    ap.add_argument(
        "--start", type=int, default=58, help="first bag frame of the packet"
    )
    ap.add_argument(
        "--sim", default=None, help="a sim/capture.py frame dir instead of the bag"
    )
    ap.add_argument(
        "--gt",
        nargs=2,
        type=float,
        default=None,
        metavar=("SIGMA_T_M", "SIGMA_R_DEG"),
        help="with --sim: ground-truth mesh + pose with this noise in place of register/canonicalize",
    )
    ap.add_argument("--seed", type=int, default=0, help="--gt noise seed")
    ap.add_argument(
        "--no-decide",
        action="store_true",
        help="FoundationPose's scorer pick, no re-rank / Orient Anything yaw decision",
    )
    ap.add_argument(
        "--sim-mesh-grasps",
        action="store_true",
        help="SIM ONLY (with --gt): privileged antipodal grasps on the ground-truth mesh instead of AnyGrasp",
    )
    a = ap.parse_args()
    assert a.gt is None or a.sim, "--gt needs --sim"
    names = list(STAGES)
    r = Run(a.out)
    r.dir.mkdir(parents=True, exist_ok=True)
    for name in names[names.index(a.start_stage) : names.index(a.end_stage) + 1]:
        print(f"== {name}", flush=True)
        STAGES[name](r, a)


if __name__ == "__main__":
    main()
