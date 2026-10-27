"""Interaction point grounding (paper Sec. 3.2, Fig. 3; supp. A.3, Figs. 12-14).

Visible and tangible parts are picked on a SCAFFOLD-gridded crop of the real image and lifted with depth;
invisible or intangible parts are picked on gridded orthographic renders of the canonical mesh. Grasp parts get
several dots (only those on the object are drawn) for the grasp heatmap. All points are canonical-frame metres.
"""

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from . import vlm

GRID = (
    8,
    8,
)  # rows, cols: the size of the paper's real-crop examples (supp. Figs. 12-14); real crops scale it to their shape
MIN_CROP_PX = 512  # real crops whose long side is shorter are upscaled before gridding (thin / far objects)
CROP_MARGIN = 0.3  # of the mask bbox per side; the paper's crops show this much context
DEPTH_WIN = 5  # px radius for the median depth around a selected dot
FG_TOL = 0.02  # m: window depths this close to the nearest one are the object (thin parts bleed background)
VIEW_PX = 512


def scaffold(img, grid=GRID, keep=None):
    """SCAFFOLD's dot_matrix_two_dimensional (leixy20/Scaffold image_processor.py): interior dots at
    i * W / (cols + 1), labels "(row,col)" from 1, black or white against the pixel. keep (HxW bool) drops dots
    outside it. -> (gridded RGB, {(row, col): (u, v)})."""
    im = Image.fromarray(img).convert("RGB")
    src = np.asarray(im).astype(int)  # contrast test on the undrawn image
    draw, (W, H), (rows, cols) = ImageDraw.Draw(im), im.size, grid
    font = ImageFont.load_default(size=max(W // 40, 8))
    r, dots = max(W // 240, 1), {}
    for j in range(1, rows + 1):
        for i in range(1, cols + 1):
            u, v = int(i * W / (cols + 1)), int(j * H / (rows + 1))
            if keep is not None and not keep[v, u]:
                continue
            c = (0, 0, 0) if int(src[v, u].sum()) >= 255 * 3 / 2 else (255, 255, 255)
            draw.ellipse([(u - r, v - r), (u + r, v + r)], fill=c)
            draw.text((u + 3, v), f"({j},{i})", fill=c, font=font)
            dots[(j, i)] = (u, v)
    return np.asarray(im), dots


def grid_for(shape, n=GRID[1]):
    """(rows, cols) with n dots along the crop's long side and 1..n along the short one, by aspect ratio."""
    h, w = shape[:2]
    short = max(1, min(n, round(n * min(h, w) / max(h, w))))
    return (n, short) if h >= w else (short, n)


def gridded_crop(img, keep=None):
    """A real crop made legible for the VLM: upscaled so its long side is at least MIN_CROP_PX, then gridded with
    grid_for(its shape). -> (gridded RGB, {(row, col): (u, v) in the ORIGINAL crop's pixels}, grid)."""
    f = max(1.0, MIN_CROP_PX / max(img.shape[:2]))
    if f > 1:
        size = (round(img.shape[1] * f), round(img.shape[0] * f))
        img = np.asarray(Image.fromarray(img).resize(size, Image.Resampling.BICUBIC))
        if keep is not None:
            keep = (
                np.asarray(
                    Image.fromarray(keep.astype(np.uint8) * 255).resize(
                        size, Image.Resampling.NEAREST
                    )
                )
                > 127
            )
    grid = grid_for(img.shape)
    g, dots = scaffold(img, grid, keep)
    return (
        g,
        {
            k: (
                min(int(u / f), int(img.shape[1] / f) - 1),
                min(int(v / f), int(img.shape[0] / f) - 1),
            )
            for k, (u, v) in dots.items()
        },
        grid,
    )


def crop(rgb, mask, margin=CROP_MARGIN):
    """Object crop: mask bbox enlarged by `margin` per side. -> (crop, mask crop, (x0, y0))."""
    ys, xs = np.nonzero(mask)
    mx, my = int((np.ptp(xs) + 1) * margin), int((np.ptp(ys) + 1) * margin)
    x0, y0 = max(xs.min() - mx, 0), max(ys.min() - my, 0)
    x1, y1 = min(xs.max() + mx + 1, rgb.shape[1]), min(ys.max() + my + 1, rgb.shape[0])
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1], (x0, y0)


def lift(u, v, depth, K, mask, cam_T_can, win=DEPTH_WIN):
    """Pixel -> canonical point: median depth of valid mask pixels within `win` px (onboarding notes O4), over
    the foreground only (within FG_TOL of the nearest), since on thin parts (handle, spout) the aligned depth inside
    the mask is largely the background behind them. With no valid pixel there, the nearest valid mask pixel is
    used. -> (point, snapped)."""
    H, W = depth.shape
    y0, y1, x0, x1 = (
        max(v - win, 0),
        min(v + win + 1, H),
        max(u - win, 0),
        min(u + win + 1, W),
    )
    ok = mask[y0:y1, x0:x1] & (depth[y0:y1, x0:x1] > 0)
    snapped = not ok.any()
    if snapped:
        vv, uu = np.nonzero(mask & (depth > 0))
        i = np.argmin((uu - u) ** 2 + (vv - v) ** 2)
        u, v, z = int(uu[i]), int(vv[i]), float(depth[vv[i], uu[i]])
    else:
        d = depth[y0:y1, x0:x1][ok]
        z = float(np.median(d[d <= d.min() + FG_TOL]))
    p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    return cam_T_can[:3, :3].T @ (p - cam_T_can[:3, 3]), snapped


# Orthographic canonical views for invisible points. Canonical frame: x front, y = z x x, z up.
# Each view: camera axis (towards the camera) and the canonical axes shown as image right / up.
VIEWS = {"front": (np.array([1.0, 0, 0]), 1, 2), "side": (np.array([0, 1.0, 0]), 0, 2)}


def render_view(mesh, view, px=VIEW_PX):
    """Textured orthographic render of the canonical mesh, white background. -> (RGB, pixel -> canonical
    (right, up) coordinates)."""
    import pyrender

    axis, right_ax, up_ax = VIEWS[view]
    lo, hi = mesh.bounds
    c, half = (lo + hi) / 2, 0.6 * float(np.max(hi - lo))
    z, y = axis, np.eye(3)[up_ax]
    x = np.cross(y, z)  # image right in canonical coords (+y for front, -x for side)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = np.c_[x, y, z], c + axis * 3 * half
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    scene.add(
        pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=10 * half),
        pose=pose,
    )
    scene.add(pyrender.DirectionalLight(intensity=2.0), pose=pose)
    rendered = pyrender.OffscreenRenderer(px, px).render(scene)
    assert rendered is not None
    color = rendered[0]
    sign = float(x[right_ax])

    def to_canonical(u, v):  # -> {right axis index: coord, up axis index: coord}
        return {
            right_ax: c[right_ax] + sign * (u / px * 2 - 1) * half,
            up_ax: c[up_ax] - (v / px * 2 - 1) * half,
        }

    return np.asarray(color), to_canonical


def _ask_dot(img, keep, real=False, **fields):
    g, dots, grid = (
        gridded_crop(img, keep) if real else (*scaffold(img, keep=keep), GRID)
    )
    rep = vlm.query("interaction_point", [g], rows=grid[0], cols=grid[1], **fields)
    return tuple(rep["Selected dot"]), dots, g, rep


def visible_point(rgb, depth, K, mask, cam_T_can, **fields):
    """Visible and tangible part: SCAFFOLD dot on the real object crop, lifted with depth."""
    im, _, (x0, y0) = crop(rgb, mask)
    dot, dots, g, rep = _ask_dot(im, None, real=True, view="camera image", **fields)
    u, v = dots[dot]
    p, snapped = lift(u + x0, v + y0, depth, K, mask, cam_T_can)
    return {
        "point": p,
        "dot": dot,
        "pixel": (u + x0, v + y0),
        "snapped": snapped,
        "vlm": rep,
        "images": [g],
    }


def invisible_point(mesh_can, **fields):
    """Invisible or intangible part: a dot on an orthographic front view fixes (y, z), one on a side view
    fixes (x, z); z is averaged (onboarding notes O4)."""
    coords, dots_sel, images, reps = {}, {}, [], {}
    for view in VIEWS:
        img, to_can = render_view(mesh_can, view)
        dot, dots, g, rep = _ask_dot(img, None, view=f"rendered {view} view", **fields)
        for ax, val in to_can(*dots[dot]).items():
            coords.setdefault(ax, []).append(val)
        dots_sel[view], reps[view] = dot, rep
        images.append(g)
    p = np.array([np.mean(coords[a]) for a in range(3)])
    return {"point": p, "dot": dots_sel, "vlm": reps, "images": images}


def grasp_points(rgb, depth, K, mask, cam_T_can, **fields):
    """Grasp part: several dots on the object (only on-mask dots drawn), each lifted with depth."""
    im, m, (x0, y0) = crop(rgb, mask)
    g, dots, grid = gridded_crop(im, m)
    rep = vlm.query("grasp_points", [g], rows=grid[0], cols=grid[1], **fields)
    sel = [tuple(d) for d in rep["Selected dots"] if tuple(d) in dots]
    pts = [
        lift(dots[d][0] + x0, dots[d][1] + y0, depth, K, mask, cam_T_can)[0]
        for d in sel
    ]
    return {"points": np.array(pts), "dots": sel, "vlm": rep, "images": [g]}


def textured_canonical_mesh(final_mesh_path, R):
    """Any6D's textured final mesh rotated into the canonical frame (V_c = V @ R)."""
    m = trimesh.load(final_mesh_path, force="mesh", process=False)
    assert isinstance(m, trimesh.Trimesh)
    m.vertices = np.asarray(m.vertices) @ np.asarray(R)
    return m


def propose(rgb, depth, K, mask, cam_T_can, final_mesh, R, element, **fields):
    """Stage 11 for one plan element: grasp dots for a grasped (passive) object, a real-crop dot for a visible
    and tangible part, else front/side render dots. -> result + "kind" and "points" (N, 3) canonical."""
    if element["action"] == "grasp" and element["type"] == "passive":
        r = grasp_points(rgb, depth, K, mask, cam_T_can, **fields)
        return {**r, "kind": "grasp"}
    if element["part_visible_and_tangible"]:
        r = visible_point(rgb, depth, K, mask, cam_T_can, **fields)
        return {**r, "kind": "visible", "points": r["point"][None]}
    r = invisible_point(textured_canonical_mesh(final_mesh, R), **fields)
    return {**r, "kind": "invisible", "points": r["point"][None]}
