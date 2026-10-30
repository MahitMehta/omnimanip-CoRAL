"""Functional grasping (supp. A.3; onboarding notes O8).

AnyGrasp proposes 6-DoF grasps; a Gaussian at each VLM grasp point is summed into a heatmap that filters them.
The AnyGrasp call is the shared grasp service's, as in Albus's grasp_viz_node.py.
"""

from functools import cache

import cv2
import numpy as np
from manip_bridge.zmq_client import SidecarClient

from . import gripper, viz
from .config import HOST, SIDECARS

MARGIN, CONTEXT = (
    0.03,
    0.15,
)  # m: lims box around the object; table context sent for collision checking
SIGMA, KEEP = 0.01, 0.5  # heatmap Gaussian (m); keep candidates with heat >= KEEP * max
TABLE_CLEAR, PRE_GRASP_MAX = 0.02, 0.16  # m: gripper clearance above the table; execute's farthest pre-grasp
TIP_CLEAR, TIP_LEN = 0.003, 0.02  # m: the fingertips (their last TIP_LEN) may come this close to the table
PRE_GRASPS = (0.08, 0.12, 0.16)  # m: execute's pre-grasp back-offs, in the order it tries them
REACH_CAP = 0.04  # m: farthest a grasp may sit from a grasp dot to be considered
MIN_DEPTH, MAX_OPEN = 0.01, 0.08  # m: fingertips at least this far past the grasp centre; opening at most this


def backproject(depth, K, sel):
    v, u = np.nonzero(sel)
    z = depth[v, u]
    return np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z].astype(
        np.float32
    ), (v, u)


def anygrasp(rgb, depth, K, mask, dense=True):
    """grasp_viz_node.process: object bbox + MARGIN as lims, bbox + CONTEXT as the cloud, collision on; `dense`
    asks for AnyGrasp's dense proposals (no NMS). -> AnyGrasp reply (camera-frame grasps; frame x = approach,
    y = closing, z = normal; score-sorted)."""
    valid = (depth > 0.05) & (depth < 3.0)
    obj, _ = backproject(depth, K, valid & mask)
    lo, hi = obj.min(0) - MARGIN, obj.max(0) + MARGIN
    allp, (v, u) = backproject(depth, K, valid)
    inside = np.all((allp >= lo - CONTEXT) & (allp <= hi + CONTEXT), axis=1)
    pts, cols = allp[inside], (rgb[v[inside], u[inside]].astype(np.float32) / 255.0)
    lims = [float(x) for x in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])]
    cli = SidecarClient(f"tcp://{HOST}:{SIDECARS['anygrasp']}", 120_000, codec="pickle")
    return cli.call(
        {"points": pts, "colors": cols, "lims": lims, "collision_detection": True, "dense_grasp": dense}
    )


@cache
def _gripper_points():
    """The real Robotiq mesh's vertices in the grasp frame (fingertips at x = 0, fingers open), and per vertex how
    far along y it slides per metre of closing (the fingers; 0 for the body)."""
    g = gripper.GRIPPER
    T = gripper.grasp_T_base(g, 0.0)
    V, W = (np.asarray(gripper.mesh(g, w).vertices) @ T[:3, :3].T + T[:3, 3] for w in (g.max_width, 0.0))
    return V, (W[:, 1] - V[:, 1]) / g.max_width


def feasible(rep, depth, K, mask, plane):
    """Real-gripper check (DEVIATIONS #27): the real Robotiq mesh at each candidate (fingers at width + 1 cm) keeps
    its fingertips (last TIP_LEN) TIP_CLEAR and the rest TABLE_CLEAR above the table plane, at the grasp and at
    PRE_GRASP_MAX back along the approach (height is linear along the straight approach, so the two ends cover every
    pre-grasp); at the grasp the palm spheres also clear the scene cloud and the finger spheres every other object.
    The mesh, not the spheres, for the table: each finger is four large spheres that bulge below a fingertip near
    the table. -> bool per candidate."""
    from scipy.spatial import KDTree

    g, n_pl, d_pl = gripper.GRIPPER, np.asarray(plane["n"], float), float(plane["d"])
    valid = (depth > 0.05) & (depth < 3.0)
    scene, _ = backproject(depth, K, valid)
    others, _ = backproject(depth, K, valid & ~mask)
    off = lambda P: np.abs(P @ n_pl + d_pl) >= 0.01  # the table itself is checked analytically
    t_scene, t_others = KDTree(scene[off(scene)]), KDTree(others[off(others)])
    V0, slide = _gripper_points()
    tip = V0[:, 0] >= -TIP_LEN
    clear = np.where(tip, TIP_CLEAR, TABLE_CLEAR)
    ok = np.zeros(rep["n"], bool)
    for i in range(rep["n"]):
        R, t, d = rep["rotations"][i], rep["translations"][i], float(rep["depths"][i])
        w = min(float(rep["widths"][i]) + 0.01, g.max_width)
        S = gripper.spheres(g, w)
        palm = S[:, 3] > 0.02
        C = S[:, :3] @ R.T + t + d * R[:, 0]
        V = V0.copy()
        V[:, 1] += slide * (g.max_width - w)
        M = V @ R.T + t + d * R[:, 0]
        ok[i] = (
            all(((P @ n_pl + d_pl) >= clear).all() for P in (M, M - PRE_GRASP_MAX * R[:, 0]))
            and (t_scene.query(C[palm])[0] > S[palm, 3]).all()
            and (t_others.query(C[~palm])[0] > S[~palm, 3]).all()
        )
    return ok


def reachable(rep, idx, depth, K, T_base_cam, q, plane):
    """Candidates `idx` whose pre-grasp cuRobo can plan to from joints q (UR order), trying PRE_GRASPS in order as
    execute does, against the frame's depth and the table slab. -> bool per candidate of rep."""
    from . import motion

    motion.map_scene(depth, K, T_base_cam, q, plane=plane)
    ok = np.zeros(rep["n"], bool)
    for i in idx:
        T = T_base_cam @ pose(rep, i)
        tool0_T_g = motion.tool0_T_grasp(float(rep["depths"][i]))
        for pre in PRE_GRASPS:
            T_pre = T.copy()
            T_pre[:3, 3] -= pre * T[:3, 0]
            if motion.plan(q, T_pre @ np.linalg.inv(tool0_T_g), None, list(q))[0] is not None:
                ok[i] = True
                break
    return ok


def candidates(rep, depth, K, mask, plane, grasp_points_cam, T_base_cam, q):
    """The grasps the heatmap may pick (DEVIATIONS #27): pass the real-gripper check, hold the object (fingertips
    MIN_DEPTH past the grasp centre, opening at most MAX_OPEN: at the gripper's full 8.5 cm AnyGrasp found no pair of
    contacts and the jaws can land on the object), within REACH_CAP of a grasp dot, and reachable (a pre-grasp
    cuRobo plans to from joints q, against the depth and the table slab). -> (bool per candidate, counts)."""
    ok = feasible(rep, depth, K, mask, plane)
    hold = ok & (np.asarray(rep["depths"]) >= MIN_DEPTH - 1e-6) & (np.asarray(rep["widths"]) <= MAX_OPEN)
    d = np.linalg.norm(rep["translations"][:, None] - np.asarray(grasp_points_cam)[None], axis=2).min(1)
    near = hold & (d <= REACH_CAP)
    reach = reachable(rep, np.flatnonzero(near), depth, K, T_base_cam, q, plane)
    return reach, {"proposed": int(rep["n"]), "real_gripper": int(ok.sum()), "holds": int(hold.sum()), "near_dots": int(near.sum()),
                   "reachable": int(reach.sum())}


def heat(x, grasp_points_cam):
    """Heatmap at 3D points x (N, 3): sum_i exp(-|x - p_i|^2 / 2 sigma^2) over the VLM grasp points."""
    d2 = ((np.asarray(x)[:, None, :] - np.asarray(grasp_points_cam)[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * SIGMA**2)).sum(1)


def select(rep, grasp_points_cam, ok=None):
    """Heatmap filter: heat(t) = sum_i exp(-|t - p_i|^2 / 2 sigma^2) at each grasp's centre; keep heat >= KEEP *
    max heat among the `ok` candidates (all if None); take the highest AnyGrasp score among those. -> (index, heats)."""
    h = heat(rep["translations"], grasp_points_cam)
    ok = np.ones(len(h), bool) if ok is None else np.asarray(ok)
    keep = np.flatnonzero(ok & (h >= KEEP * h[ok].max()))
    return int(keep[np.argmax(np.asarray(rep["scores"])[keep])]), h


def pose(rep, i):
    """cam_T_grasp for grasp i."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rep["rotations"][i], rep["translations"][i]
    return T


def choose(rep, cam_T_can, grasp_points_can, ok=None):
    """The heatmap pick among the `ok` candidates for the object's VLM grasp points (canonical). -> (index,
    cam_T_grasp, Phi, grasp points in the camera frame, heats); Phi = grasp -> canonical, so P_active = P_grasp . Phi (Eq. 3)."""
    p_cam = np.asarray(grasp_points_can) @ cam_T_can[:3, :3].T + cam_T_can[:3, 3]
    i, h = select(rep, p_cam, ok)
    T_g = pose(rep, i)
    return i, T_g, np.linalg.inv(T_g) @ cam_T_can, p_cam, h


def overlay(rgb, K, rep, p_cam, i, up):
    """Candidate centres (grey), grasp points (blue), and grasp i: its two jaws (green) at +-width/2 along the
    closing axis y, from 2 cm behind the centre to the tips `depth` past it, the opening between them (yellow), the
    approach (x, red) into the palm, and a label with the pinch's tilt from the table (`up` = table normal)."""
    im = rgb.copy()
    for t in np.round(
        viz.project(K, np.eye(4), np.asarray(rep["translations"]))
    ).astype(int):
        cv2.circle(im, (int(t[0]), int(t[1])), 3, (160, 160, 160), -1)
    for q in np.round(viz.project(K, np.eye(4), p_cam)).astype(int):
        cv2.circle(im, (int(q[0]), int(q[1])), 6, (0, 0, 255), -1)
    T_g, w, d = pose(rep, i), float(rep["widths"][i]), float(rep["depths"][i])
    px = lambda X: [
        tuple(int(v) for v in p) for p in np.round(viz.project(K, T_g, np.array(X)))
    ]
    for s in (-1, 1):  # jaw: palm side to fingertip
        a, b = px([[-0.02, s * w / 2, 0], [d, s * w / 2, 0]])
        cv2.line(im, a, b, (0, 200, 0), 4)
    a, b = px([[0, -w / 2, 0], [0, w / 2, 0]])
    cv2.line(im, a, b, (220, 220, 0), 1)
    a, b = px([[-0.08, 0, 0], [-0.02, 0, 0]])
    cv2.arrowedLine(im, a, b, (255, 0, 0), 2, tipLength=0.3)
    tilt = np.degrees(np.arcsin(min(1.0, abs(float(T_g[:3, 1] @ up)))))
    kind = "vertical" if tilt > 60 else "horizontal" if tilt < 30 else "diagonal"
    cv2.putText(
        im,
        f"{kind} pinch ({tilt:.0f} deg from table)  width {w * 100:.1f} cm  score {rep['scores'][i]:.3f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    return im


def mesh_grasps(
    V,
    F,
    cam_T_obj,
    plane,
    n_pairs=400,
    n_approach=12,
    depths=(0.03, 0.02, 0.015, 0.01),
    keep=100,
    seed=0,
):
    """SIM ONLY, privileged stand-in for AnyGrasp (like Albus's sim proposer): antipodal grasps on the object's
    ground-truth mesh at its ground-truth pose, in AnyGrasp's reply format so selection and execution are unchanged.
    Surface point pairs whose normals oppose along the line between them (within 15 deg) and lie within the
    gripper's opening give the closing axis; approaches are swept about it every 360/n_approach deg. The grasp origin
    sits depth/2 before the pair's midpoint, so the pads (tips `depth` past it) centre on it; the deepest of `depths`
    whose palm spheres clear the object and the table, and whose fingertips stay 2 mm above the table, is kept (the
    coarse finger spheres themselves may touch a low object's table). Score = the worse normal alignment."""
    import trimesh
    from scipy.spatial import KDTree

    m = trimesh.Trimesh(V, F, process=False)
    sample = trimesh.sample.sample_surface(m, 6000, seed=seed)
    P, N = sample[0], m.face_normals[sample[1]]
    P, N = P @ cam_T_obj[:3, :3].T + cam_T_obj[:3, 3], N @ cam_T_obj[:3, :3].T
    tree, rng, g = KDTree(P), np.random.default_rng(seed), gripper.GRIPPER
    n_pl, d_pl = np.asarray(plane["n"], float), float(plane["d"])
    out = []
    for i in rng.choice(len(P), min(n_pairs * 5, len(P)), replace=False):
        if len(out) >= n_pairs * n_approach:
            break
        js = np.array(tree.query_ball_point(P[i], g.max_width - 0.01))
        js = js[js != i]
        if not len(js):
            continue
        v = P[js] - P[i]
        dist = np.linalg.norm(v, axis=1)
        y = v / np.maximum(dist, 1e-9)[:, None]
        q = np.minimum(
            -(N[i] @ y.T), np.einsum("ij,ij->i", N[js], y)
        )  # both normals face outward along the axis
        ok = (q > np.cos(np.radians(15))) & (dist > 0.005)
        if not ok.any():
            continue
        j = js[ok][np.argmax(q[ok])]
        w, qa = float(np.linalg.norm(P[j] - P[i])), float(q[ok].max())
        y = (P[j] - P[i]) / w
        a0 = np.cross(y, [0.0, 0, 1]) if abs(y[2]) < 0.9 else np.cross(y, [1.0, 0, 0])
        a0 /= np.linalg.norm(a0)
        S = gripper.spheres(g, min(w + 0.01, g.max_width))
        palm, tip = (
            S[:, 3] > 0.02,
            S[:, 0] == S[:, 0].max(),
        )  # the sphere model's fingertips sit at its origin
        for t in np.linspace(0, 2 * np.pi, n_approach, endpoint=False):
            x = a0 * np.cos(t) + np.cross(y, a0) * np.sin(
                t
            )  # approach, perpendicular to the closing axis
            R = np.c_[x, y, np.cross(x, y)]
            for depth in depths:
                o = (P[i] + P[j]) / 2 - depth / 2 * x
                C = S[:, :3] @ R.T + o + depth * x
                h = C @ n_pl + d_pl
                if (
                    (h[palm] > S[palm, 3]).all()
                    and (h[tip] > 0.002).all()
                    and (tree.query(C[palm])[0] > S[palm, 3]).all()
                ):
                    out.append((qa, o, R, min(w + 0.01, g.max_width), depth))
                    break
    out.sort(key=lambda c: -c[0])
    out = out[:keep]
    return {
        "n": len(out),
        "scores": np.array([c[0] for c in out]),
        "translations": np.array([c[1] for c in out]).reshape(-1, 3),
        "rotations": np.array([c[2] for c in out]).reshape(-1, 3, 3),
        "widths": np.array([c[3] for c in out]),
        "depths": np.array([c[4] for c in out]),
    }
