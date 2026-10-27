"""Functional grasping (supp. A.3; onboarding notes O8).

AnyGrasp proposes 6-DoF grasps; a Gaussian at each VLM grasp point is summed into a heatmap that filters them.
The AnyGrasp call is the shared grasp service's, as in Albus's grasp_viz_node.py.
"""

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


def backproject(depth, K, sel):
    v, u = np.nonzero(sel)
    z = depth[v, u]
    return np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z].astype(
        np.float32
    ), (v, u)


def anygrasp(rgb, depth, K, mask):
    """grasp_viz_node.process: object bbox + MARGIN as lims, bbox + CONTEXT as the cloud, collision on.
    -> AnyGrasp reply (camera-frame grasps; frame x = approach, y = closing, z = normal; score-sorted)."""
    valid = (depth > 0.05) & (depth < 3.0)
    obj, _ = backproject(depth, K, valid & mask)
    lo, hi = obj.min(0) - MARGIN, obj.max(0) + MARGIN
    allp, (v, u) = backproject(depth, K, valid)
    inside = np.all((allp >= lo - CONTEXT) & (allp <= hi + CONTEXT), axis=1)
    pts, cols = allp[inside], (rgb[v[inside], u[inside]].astype(np.float32) / 255.0)
    lims = [float(x) for x in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])]
    cli = SidecarClient(f"tcp://{HOST}:{SIDECARS['anygrasp']}", 120_000, codec="pickle")
    return cli.call(
        {"points": pts, "colors": cols, "lims": lims, "collision_detection": True}
    )


def heat(x, grasp_points_cam):
    """Heatmap at 3D points x (N, 3): sum_i exp(-|x - p_i|^2 / 2 sigma^2) over the VLM grasp points."""
    d2 = ((np.asarray(x)[:, None, :] - np.asarray(grasp_points_cam)[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * SIGMA**2)).sum(1)


def select(rep, grasp_points_cam):
    """Heatmap filter: heat(t) = sum_i exp(-|t - p_i|^2 / 2 sigma^2) at each grasp's centre; keep heat >= KEEP *
    max heat; take the highest AnyGrasp score among those. -> (index, heats)."""
    h = heat(rep["translations"], grasp_points_cam)
    keep = np.flatnonzero(h >= KEEP * h.max())
    return int(keep[np.argmax(np.asarray(rep["scores"])[keep])]), h


def pose(rep, i):
    """cam_T_grasp for grasp i."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rep["rotations"][i], rep["translations"][i]
    return T


def choose(rep, cam_T_can, grasp_points_can):
    """The heatmap pick for the object's VLM grasp points (canonical). -> (index, cam_T_grasp, Phi, grasp points in
    the camera frame, heats); Phi = grasp -> canonical, so P_active = P_grasp . Phi (Eq. 3)."""
    p_cam = np.asarray(grasp_points_can) @ cam_T_can[:3, :3].T + cam_T_can[:3, 3]
    i, h = select(rep, p_cam)
    T_g = pose(rep, i)
    return i, T_g, np.linalg.inv(T_g) @ cam_T_can, p_cam, h


def overlay(rgb, K, rep, p_cam, T_g):
    """Candidate centres (grey), grasp points (blue), the chosen grasp's centre with its approach (x, red) and
    closing (y, green) axes."""
    im = rgb.copy()
    for t in np.round(
        viz.project(K, np.eye(4), np.asarray(rep["translations"]))
    ).astype(int):
        cv2.circle(im, (int(t[0]), int(t[1])), 3, (160, 160, 160), -1)
    for q in np.round(viz.project(K, np.eye(4), p_cam)).astype(int):
        cv2.circle(im, (int(q[0]), int(q[1])), 6, (0, 0, 255), -1)
    c, ax, cl = np.round(
        viz.project(K, T_g, np.array([[0, 0, 0], [-0.06, 0, 0], [0, 0.04, 0]]))
    ).astype(int)
    cv2.arrowedLine(
        im,
        (int(ax[0]), int(ax[1])),
        (int(c[0]), int(c[1])),
        (255, 0, 0),
        2,
        tipLength=0.25,
    )
    cv2.line(
        im,
        (int(2 * c[0] - cl[0]), int(2 * c[1] - cl[1])),
        (int(cl[0]), int(cl[1])),
        (0, 200, 0),
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
