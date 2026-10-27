"""Reprojection overlays (onboarding notes Sec. 9: log mesh, pose and a reprojection overlay per object)."""

import cv2
import numpy as np

COLORS = [(255, 170, 0), (0, 200, 255), (255, 0, 200), (120, 255, 0)]


def project(K, T, X):
    p = X @ T[:3, :3].T + T[:3, 3]
    return p[:, :2] / p[:, 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def silhouette(K, T, V, F, shape):
    sil = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(sil, list(np.round(project(K, T, V)[F]).astype(np.int32)), 1)
    return sil > 0


def mesh_accuracy(rgb, depth, K, objects, masks):
    """Posed mesh vs the SAM 3 mask: green = both, red = mesh only, blue = mask only. Labels each object
    with silhouette IoU and the median distance from its observed depth points to the posed mesh surface.
    -> (image, {name: {iou, depth_res_mm}})."""
    import trimesh
    from scipy.spatial import KDTree

    im, stats = (0.6 * rgb).astype(np.uint8), {}
    for name, (T, V, F) in objects.items():
        sil, m = silhouette(K, T, V, F, im.shape), masks[name]
        im[sil & m] = (0, 200, 0)
        im[sil & ~m] = (255, 0, 0)
        im[m & ~sil] = (0, 90, 255)
        v, u = np.nonzero(m & (depth > 0))
        z = depth[v, u]
        obs = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
        surf = trimesh.Trimesh(V @ T[:3, :3].T + T[:3, 3], F, process=False).sample(
            30000
        )
        stats[name] = {
            "iou": float((sil & m).sum() / (sil | m).sum()),
            "depth_res_mm": float(np.median(KDTree(surf).query(obs)[0]) * 1000),
        }
        ys, xs = np.nonzero(m)
        txt = f"{name}: IoU {stats[name]['iou']:.2f}, depth {stats[name]['depth_res_mm']:.1f} mm"
        cv2.putText(
            im,
            txt,
            (int(xs.min()), int(ys.min()) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    cv2.putText(
        im,
        "green: mesh+mask   red: mesh only   blue: mask only",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return im, stats


def canonical_overlay(rgb, K, objects, axis_len=0.08):
    """objects: {name: (cam_T_can 4x4, canonical vertices Nx3, faces Mx3)}. Draws each posed mesh
    (tinted fill + outline) and its canonical axes: x = front red, y = green, z = up blue."""
    im = rgb.copy()
    for (name, (T, V, F)), col in zip(objects.items(), COLORS):
        sil = silhouette(K, T, V, F, im.shape).astype(np.uint8)
        im[sil > 0] = (0.55 * im[sil > 0] + 0.45 * np.array(col)).astype(np.uint8)
        cv2.drawContours(
            im,
            cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
            -1,
            col,
            2,
        )
        o, *ends = np.round(
            project(K, T, np.vstack([np.zeros(3), np.eye(3) * axis_len]))
        ).astype(int)
        for e, c, lab in zip(ends, [(255, 0, 0), (0, 255, 0), (0, 0, 255)], "xyz"):
            cv2.arrowedLine(im, tuple(o), tuple(e), c, 3, tipLength=0.15)
            cv2.putText(im, lab, tuple(e + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(
            im,
            name,
            tuple(o + [-30, 40]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            im, name, tuple(o + [-30, 40]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2
        )
    cv2.putText(
        im,
        "canonical frame: x front (red)  y = z cross x (green)  z up (blue)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return im
