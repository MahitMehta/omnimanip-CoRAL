"""Interaction primitives as spatial constraints (paper Sec. 3.1 Eq. 1, Sec. 3.2, Fig. 2).

C = {O_active, O_passive, d, theta}. Following the paper's artefacts (DEVIATIONS #20) there is no constraint
prompt: theta is always "aligned" (Fig. 2), the ordered list K_i follows the active object's direction ranking
(each paired with the passive object's top direction), and the signed distance d is RRC's "interaction_value"
(supp. A.4: "v is the distance predicted by VLM"), 0 until RRC sets it. Points/axes are canonical-frame.
"""

import numpy as np


def candidates(active_point, active_dirs, passive_point, passive_dir):
    """K_i = [C(1), ..., C(N)]: one constraint per ranked active direction."""
    return [
        {
            "rank": k + 1,
            "active_point": np.asarray(active_point, float),
            "active_axis": np.asarray(v, float),
            "passive_point": np.asarray(passive_point, float),
            "passive_axis": np.asarray(passive_dir, float),
            "distance_m": 0.0,
        }
        for k, v in enumerate(active_dirs)
    ]


def describe(c):
    """Fig. 2's constraint text, with the distance slot filled."""
    d = c["distance_m"] * 100
    return (
        "1. Active axis and passive axis are aligned. "
        f"2. Active point is {abs(d):.0f}cm {'negative' if d < 0 else 'positive'} along passive axis to passive point."
    )


def world(c, T_active, T_passive):
    """Constraint primitives in the camera frame through the current object poses (cam_T_can).
    -> (a, a_axis, b, b_axis)."""
    Ra, Rp = T_active[:3, :3], T_passive[:3, :3]
    return (
        Ra @ c["active_point"] + T_active[:3, 3],
        Ra @ c["active_axis"],
        Rp @ c["passive_point"] + T_passive[:3, 3],
        Rp @ c["passive_axis"],
    )


def solve_active_pose(c, T_active, T_passive):
    """Active-object pose (cam_T_can) that satisfies C exactly, for rendering it (RRC, onboarding notes O7: solve
    O9 with only L_C). Closed form: the smallest rotation taking the active axis onto the passive axis (the
    rotation about the axis is otherwise free), then the translation putting a at b + d * b_axis."""
    _, va, b, vb = world(c, T_active, T_passive)
    axis, cos = np.cross(va, vb), float(np.clip(va @ vb, -1, 1))
    s = float(np.linalg.norm(axis))
    if s < 1e-9:  # parallel or antiparallel
        axis = np.cross(va, [1.0, 0, 0] if abs(va[0]) < 0.9 else [0, 1.0, 0])
        s, axis = 0.0, axis / np.linalg.norm(axis)
    else:
        axis = axis / s
    ang = np.arctan2(s, cos)
    Kx = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R_align = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
    T = np.eye(4)
    T[:3, :3] = R_align @ T_active[:3, :3]
    T[:3, 3] = b + c["distance_m"] * vb - T[:3, :3] @ c["active_point"]
    return T


def resample(c, n=6, half_angle_deg=20.0):
    """RRC refinement (Alg. 1): n active directions uniformly around C's active axis on a cone (onboarding
    notes O7: 6 at 20 deg). -> n constraints differing only in the active axis."""
    v = c["active_axis"] / np.linalg.norm(c["active_axis"])
    u = np.cross(v, [1.0, 0, 0] if abs(v[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w, t = np.cross(v, u), np.radians(half_angle_deg)
    out = []
    for i in range(n):
        phi = 2 * np.pi * i / n
        d = np.cos(t) * v + np.sin(t) * (np.cos(phi) * u + np.sin(phi) * w)
        out.append({**c, "rank": f"{c['rank']}.{i + 1}", "active_axis": d})
    return out


def residual(c, T_active, T_passive, w_d=10.0):
    """rho(C, P_active, P_passive) of Eq. 3, as specified in the onboarding notes (O9):
    (1 - a_axis . b_axis) + w_d * ||a - (b + d * b_axis)||, with w_d in 1/m. Zero when the constraint holds."""
    a, va, b, vb = world(c, T_active, T_passive)
    return float((1 - va @ vb) + w_d * np.linalg.norm(a - (b + c["distance_m"] * vb)))
