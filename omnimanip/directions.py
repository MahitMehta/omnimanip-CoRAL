"""Interaction direction sampling (paper Sec. 3.2, Fig. 4; supp. A.2, Figs. 12-14).

Candidates are the six signed principal axes of the canonical frame. The VLM captions each one from renders with
the candidate drawn as a red arrow; an LLM then ranks the captions for the task.
"""

import cv2
import numpy as np
import trimesh
from trimesh.visual import ColorVisuals

from . import vlm

# Panels 1-6: i and i+3 are the same axis with opposite signs (supp. Fig. 12). Canonical: x front, y, z up.
CANDIDATES = np.array(
    [[1, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, -1, 0]], float
)
ARROW_LEN = 0.75  # x the mesh's largest extent
CV_TO_GL = np.diag(
    [1.0, -1.0, -1.0, 1.0]
)  # OpenCV camera (z forward, y down) -> OpenGL camera


def arrow(start, direction, length, color=(200, 0, 0, 255)):
    """Arrow mesh (red by default): shaft + cone head."""
    r = 0.025 * length
    tip = start + direction * length
    shaft = trimesh.creation.cylinder(
        radius=r, segment=[start, tip - direction * 4 * r]
    )
    head = trimesh.creation.cone(radius=2.5 * r, height=4 * r)
    T = np.asarray(trimesh.geometry.align_vectors([0, 0, 1], direction))
    T[:3, 3] = tip - direction * 4 * r
    head.apply_transform(T)
    a = trimesh.util.concatenate([shaft, head])
    assert isinstance(a, trimesh.Trimesh)
    a.visual = ColorVisuals(a, vertex_colors=list(color))
    return a


def frame_arrows(T, length):
    """A pose's axes as camera-frame arrows: x red, y green, z blue."""
    cols = [(220, 0, 0, 255), (0, 190, 0, 255), (0, 60, 230, 255)]
    return [arrow(T[:3, 3], T[:3, i], length, cols[i]) for i in range(3)]


def render_candidates(mesh_can, cam_T_can, K, size):
    """The posed canonical mesh from the real camera, once per candidate with its arrow from the bbox center.
    The canvas is twice the image size with the principal point shifted to centre the object, so the viewpoint
    and focal length are the camera's but arrows leaving the camera's field of view stay in frame.
    -> six RGB panels (white background) cropped to the same box."""
    import pyrender

    H, W = 2 * size[0], 2 * size[1]
    lo, hi = mesh_can.bounds
    c, L = (lo + hi) / 2, ARROW_LEN * float(np.max(hi - lo))
    pc = cam_T_can[:3, :3] @ c + cam_T_can[:3, 3]
    u0, v0 = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    cx, cy = K[0, 2] - u0 + W / 2, K[1, 2] - v0 + H / 2
    renderer, panels, boxes = pyrender.OffscreenRenderer(W, H), [], []
    for d in CANDIDATES:
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5]
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh_can), pose=cam_T_can)
        scene.add(pyrender.Mesh.from_trimesh(arrow(c, d, L)), pose=cam_T_can)
        cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], cx, cy, znear=0.01, zfar=10)
        scene.add(cam, pose=CV_TO_GL)
        scene.add(pyrender.DirectionalLight(intensity=2.0), pose=CV_TO_GL)
        rendered = renderer.render(scene)
        assert rendered is not None
        color, depth = rendered
        ys, xs = np.nonzero(depth > 0)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        panels.append(np.asarray(color))
    renderer.delete()
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    m = int(0.1 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)
    return [p[y0:y1, x0:x1] for p in panels]


def composite(panels):
    """3 x 2 numbered sheet as in supp. Fig. 12: left column 1-3, right column 4-6."""
    tiles = []
    for i, p in enumerate(panels):
        t = cv2.copyMakeBorder(
            p.copy(), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        cv2.putText(
            t, str(i + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )
        tiles.append(t)
    return np.hstack([np.vstack(tiles[:3]), np.vstack(tiles[3:])])


def sample(mesh_can, cam_T_can, K, size, instruction, action, obj, role, part, parts):
    """Caption the six candidates (VLM) and rank them for the stage (text-only LLM call).
    -> {candidates, captions, expected caption, ranking (1-based), ranked directions, image}."""
    sheet = composite(render_candidates(mesh_can, cam_T_can, K, size))
    captions = vlm.query(
        "direction_caption", [sheet], object=obj, parts=", ".join(parts)
    )
    rank = vlm.query(
        "direction_rank",
        instruction=instruction,
        action=action,
        object=obj,
        role=role,
        part=part,
        captions="\n".join(captions),
    )
    order = [i for i in rank["ranked directions"] if 1 <= i <= 6]
    return {
        "candidates": CANDIDATES,
        "captions": captions,
        "expected": rank["expected interaction direction caption"],
        "ranking": order,
        "ranked_directions": CANDIDATES[[i - 1 for i in order]],
        "image": sheet,
    }
