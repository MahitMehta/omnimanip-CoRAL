"""Closed-loop planning: Resampling, Rendering and Checking (paper Sec. 3.3, Algorithm 1; supp. A.3).

Each candidate constraint is rendered by moving the active object's mesh to the pose that satisfies it (the
object's original pixels inpainted first), and the VLM answers success / failure / refine plus the distance
("interaction_value"). The first "refine" swaps the list for six directions resampled around that candidate.
"""

import cv2
import numpy as np

from . import constraints as C
from . import vlm
from .directions import ARROW_LEN, CV_TO_GL, arrow

INPAINT_DILATE = (
    7  # px: cover the mask's edge so the object's outline is not left behind
)


def inpaint(rgb, mask):
    """Remove the active object's original appearance (supp. A.3: the paper's method [40] is not named;
    OpenCV's Telea inpainting here)."""
    m = cv2.dilate(
        mask.astype(np.uint8) * 255, np.ones((2 * INPAINT_DILATE + 1,) * 2, np.uint8)
    )
    return cv2.inpaint(rgb, m, 5, cv2.INPAINT_TELEA)


def render(bg, K, meshes, extras=(), pad=1.0, margin=0.15):
    """Composite Pyrender renders of (mesh, cam_T_mesh) pairs and camera-frame `extras` (arrows) over bg, on a
    canvas padded by `pad` x the image size on every side (grey beyond the real image) so a moved object never
    leaves the frame, then cropped to the rendered content plus `margin`."""
    import pyrender

    H, W = bg.shape[:2]
    ox, oy = int(pad * W), int(pad * H)
    canvas = np.full((H + 2 * oy, W + 2 * ox, 3), 160, np.uint8)
    canvas[oy : oy + H, ox : ox + W] = bg
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
    for m, T in meshes:
        scene.add(pyrender.Mesh.from_trimesh(m), pose=T)
    for e in extras:
        scene.add(pyrender.Mesh.from_trimesh(e))
    cam = pyrender.IntrinsicsCamera(
        K[0, 0], K[1, 1], K[0, 2] + ox, K[1, 2] + oy, znear=0.01, zfar=10
    )
    scene.add(cam, pose=CV_TO_GL)
    scene.add(pyrender.DirectionalLight(intensity=2.0), pose=CV_TO_GL)
    r = pyrender.OffscreenRenderer(canvas.shape[1], canvas.shape[0])
    rendered = r.render(scene)
    r.delete()
    assert rendered is not None
    color, depth = rendered
    canvas[depth > 0] = np.asarray(color)[depth > 0]
    ys, xs = np.nonzero(depth > 0)
    m = int(margin * max(np.ptp(xs), np.ptp(ys)))
    x0, y0 = max(xs.min() - m, 0), max(ys.min() - m, 0)
    x1, y1 = (
        min(xs.max() + m + 1, canvas.shape[1]),
        min(ys.max() + m + 1, canvas.shape[0]),
    )
    return canvas[y0:y1, x0:x1]


def interaction_image(c, bg, K, mesh_a, T_a, mesh_p, T_p):
    """The scene with the active mesh moved to satisfy c, both meshes rendered, the active axis as a red arrow."""
    T_star = C.solve_active_pose(c, T_a, T_p)
    a, va, _, _ = C.world(c, T_star, T_p)
    L = ARROW_LEN * float(np.max(mesh_a.extents))
    return render(bg, K, [(mesh_p, T_p), (mesh_a, T_star)], [arrow(a, va, L)]), T_star


def plan(K_list, bg, K, mesh_a, T_a, mesh_p, T_p, fields, M=6):
    """Algorithm 1. -> (accepted constraint with distance or None, trace of every check)."""
    todo, refine, trace = list(K_list), False, []
    k = 0
    while k < len(todo):
        c = todo[k]
        img, _ = interaction_image(c, bg, K, mesh_a, T_a, mesh_p, T_p)
        rep = vlm.query("rrc", [img], constraint=C.describe(c), **fields)
        trace.append(
            {
                "rank": c["rank"],
                "active_axis": c["active_axis"],
                "reply": rep,
                "image": img,
            }
        )
        print(  # as each verdict arrives (the images are written once the loop ends)
            f"{fields['action']} check {len(trace) - 1}: C({c['rank']}) -> {rep['Reasonable']},"
            f" value {rep['interaction_value']} cm |",
            rep["Think Step By Step"][:160],
            flush=True,
        )
        c = {**c, "distance_m": float(rep["interaction_value"]) / 100}
        if rep["Reasonable"] == "success":
            return {**c, "cam_T_active_target": C.solve_active_pose(c, T_a, T_p)}, trace
        if rep["Reasonable"] == "refine" and not refine:
            todo, k, refine = C.resample(c, n=M), 0, True
            continue
        k += 1
    return None, trace
