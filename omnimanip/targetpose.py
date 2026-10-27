"""Target end-effector pose optimisation (paper Sec. 3.3, Eqs. 2-5; onboarding notes O9).

P_ee* = argmin L_C + L_collision + L_path, with the active object rigidly attached: P_active = P_ee . Phi.
EE = the grasp frame (gripper.py). All poses are camera-frame; the robot base frame comes in at execution.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from . import constraints as C
from . import gripper
from .gripper import GRIPPER

D_MIN, LAMBDA_T, LAMBDA_R = 0.02, 1.0, 0.1  # m; 1/m; 1/rad (onboarding notes O9)
RHO_TOL = 0.05  # accept the solve if rho <= this: ~18 deg of misalignment at 0 mm, or 5 mm at 0 deg
VOXEL = 0.005  # m: obstacle cloud downsampling


def obstacles(depth, K, exclude):
    """Scene points with the task objects' masks removed (onboarding notes O9), voxel-downsampled."""
    v, u = np.nonzero((depth > 0.05) & (depth < 3.0) & ~exclude)
    z = depth[v, u]
    p = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
    _, keep = np.unique(np.floor(p / VOXEL).astype(np.int64), axis=0, return_index=True)
    return p[keep]


def _pose(x, T0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[3:]).as_matrix() @ T0[:3, :3]
    T[:3, 3] = T0[:3, 3] + x[:3]
    return T


def losses(T_ee, T_ee0, c, phi, T_passive, tree, width):
    """{constraint, collision, path} for one EE pose (Eqs. 3-5)."""
    s = gripper.spheres(GRIPPER, width)  # fingers closed on the held object
    centres = s[:, :3] @ T_ee[:3, :3].T + T_ee[:3, 3]
    d = tree.query(centres)[0] - s[:, 3]
    drot = (
        Rotation.from_matrix(T_ee[:3, :3]) * Rotation.from_matrix(T_ee0[:3, :3]).inv()
    ).magnitude()
    return {
        "constraint": C.residual(c, T_ee @ phi, T_passive),
        "collision": float((np.maximum(0.0, D_MIN - d) ** 2).sum()),
        "path": float(
            LAMBDA_T * np.linalg.norm(T_ee[:3, 3] - T_ee0[:3, 3]) + LAMBDA_R * drot
        ),
    }


def solve(c, phi, T_ee0, T_passive, obstacle_points, width, x0_pose=None):
    """SLSQP over (translation, rotation vector) relative to the current EE pose, warm-started at x0_pose (the
    previous solution on tracker updates) or at the current pose. -> (P_ee*, terms, ok)."""
    tree = KDTree(obstacle_points)
    x0 = np.zeros(6)
    if x0_pose is not None:
        x0[:3] = x0_pose[:3, 3] - T_ee0[:3, 3]
        x0[3:] = (
            Rotation.from_matrix(x0_pose[:3, :3])
            * Rotation.from_matrix(T_ee0[:3, :3]).inv()
        ).as_rotvec()

    def f(x):
        return sum(
            losses(_pose(x, T_ee0), T_ee0, c, phi, T_passive, tree, width).values()
        )

    res = minimize(f, x0, method="SLSQP", options={"maxiter": 500, "ftol": 1e-9})
    T = _pose(res.x, T_ee0)
    terms = losses(T, T_ee0, c, phi, T_passive, tree, width)
    return T, terms, bool(terms["constraint"] <= RHO_TOL)


def overlay(rgb, K, mask_a, mesh_a, mesh_p, T_p, T_ee0, T_ee, phi, width, depth):
    """Padded-canvas render: the active object (canonical mesh) at the optimised pose over the frame with it
    inpainted; 6D frames as arrows -- grasp frame at the start and at the target (x approach red, y closing green,
    z normal blue), the active object's canonical frame at the target (x front, y, z up); the gripper closed at
    both grasp frames; the target grasp centre as a yellow sphere."""
    import trimesh
    from trimesh.visual import ColorVisuals

    from . import rrc
    from .directions import frame_arrows

    centre = trimesh.creation.icosphere(radius=0.006)
    centre.apply_translation(T_ee[:3, 3])
    centre.visual = ColorVisuals(centre, vertex_colors=[255, 220, 0, 255])
    extras = (
        frame_arrows(T_ee0, 0.05)
        + frame_arrows(T_ee, 0.05)
        + frame_arrows(T_ee @ phi, 0.08)
        + [centre]
    )
    grip = gripper.mesh(GRIPPER, width).copy()
    grip.visual = ColorVisuals(grip, vertex_colors=[90, 90, 100, 255])
    g_base = gripper.grasp_T_base(GRIPPER, depth)
    meshes = [
        (mesh_p, T_p),
        (mesh_a, T_ee @ phi),
        (grip, T_ee0 @ g_base),
        (grip, T_ee @ g_base),
    ]
    return rrc.render(rrc.inpaint(rgb, mask_a), K, meshes, extras)
