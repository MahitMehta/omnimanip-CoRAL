"""Target end-effector pose optimisation (paper Sec. 3.3, Eqs. 2-5; onboarding notes O9).

P_ee* = argmin L_C + L_collision + L_path, with the active object rigidly attached: P_active = P_ee . Phi.
EE = the grasp frame (gripper.py). All poses are camera-frame; the robot base frame comes in at execution.
L_collision checks the gripper and the held object against the scene and the passive object (its posed mesh).
"""

from functools import cache
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from . import constraints as C
from . import gripper, motion
from .gripper import GRIPPER
from .shared_perception import read_obj

D_MIN, LAMBDA_T, LAMBDA_R = 0.02, 1.0, 0.1  # m; 1/m; 1/rad (onboarding notes O9)
D_MIN_HELD_PASSIVE = 0.0  # m: the constraint brings them into contact (place at d = 0), so only overlap counts
RHO_TOL = 0.05  # accept the solve if rho <= this: ~18 deg of misalignment at 0 mm, or 5 mm at 0 deg
VOXEL = 0.005  # m: obstacle cloud downsampling


def obstacles(depth, K, exclude):
    """Scene points with the task objects' masks removed (onboarding notes O9), voxel-downsampled."""
    v, u = np.nonzero((depth > 0.05) & (depth < 3.0) & ~exclude)
    z = depth[v, u]
    return _voxel(np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def _voxel(p):
    _, keep = np.unique(np.floor(p / VOXEL).astype(np.int64), axis=0, return_index=True)
    return p[keep]


@cache
def body_spheres(mesh_path):
    """The object's covering spheres in its mesh's (body) frame, as cuRobo carries it (motion.attached_spheres);
    fitted once per mesh (~4 s)."""
    V, F = read_obj(Path(mesh_path))
    return motion.attached_spheres(V, F, np.eye(4))


@cache
def _body_surface(mesh_path):
    V, F = read_obj(Path(mesh_path))
    m = trimesh.Trimesh(V, F, process=False)
    return _voxel(
        trimesh.sample.sample_surface(m, max(2000, int(4 * m.area / VOXEL**2)), seed=0)[
            0
        ]
    )


def held_model(mesh_path, R):
    """The held object's spheres in its canonical frame (V_c = V @ R, as points.textured_canonical_mesh)."""
    s = body_spheres(mesh_path).copy()
    s[:, :3] = s[:, :3] @ np.asarray(R)
    return s


def passive_model(mesh_path, R):
    """The passive object's surface points (5 mm voxels) in its canonical frame, as a KDTree."""
    return KDTree(_body_surface(mesh_path) @ np.asarray(R))


def _pose(x, T0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[3:]).as_matrix() @ T0[:3, :3]
    T[:3, 3] = T0[:3, 3] + x[:3]
    return T


def _hinge(tree, T_inv, s, d_min):
    """Eq. 4's sum for spheres s (camera-frame centres, radius) against one obstacle tree, whose frame is T_inv."""
    d = tree.query(s[:, :3] @ T_inv[:3, :3].T + T_inv[:3, 3])[0] - s[:, 3]
    return float((np.maximum(0.0, d_min - d) ** 2).sum())


def losses(T_ee, T_ee0, c, phi, T_passive, tree, width, held, passive):
    """{constraint, collision_gripper, collision_held, path} for one EE pose (Eqs. 3-5): the gripper's spheres
    (fingers closed on the held object) and the held object's (`held`, its canonical frame, at T_ee . phi) against
    the scene cloud (`tree`) and the passive object's surface (`passive`, its canonical frame, at T_passive)."""
    T_a, P_inv = T_ee @ phi, np.linalg.inv(T_passive)
    g = gripper.spheres(GRIPPER, width)
    g = np.c_[g[:, :3] @ T_ee[:3, :3].T + T_ee[:3, 3], g[:, 3]]
    h = np.c_[held[:, :3] @ T_a[:3, :3].T + T_a[:3, 3], held[:, 3]]
    drot = (
        Rotation.from_matrix(T_ee[:3, :3]) * Rotation.from_matrix(T_ee0[:3, :3]).inv()
    ).magnitude()
    return {
        "constraint": C.residual(c, T_a, T_passive),
        "collision_gripper": _hinge(tree, np.eye(4), g, D_MIN)
        + _hinge(passive, P_inv, g, D_MIN),
        "collision_held": _hinge(tree, np.eye(4), h, D_MIN)
        + _hinge(passive, P_inv, h, D_MIN_HELD_PASSIVE),
        "path": float(
            LAMBDA_T * np.linalg.norm(T_ee[:3, 3] - T_ee0[:3, 3]) + LAMBDA_R * drot
        ),
    }


def solve(
    c, phi, T_ee0, T_passive, obstacle_points, width, held, passive, x0_pose=None
):
    """SLSQP over (translation, rotation vector) relative to the current EE pose, warm-started at x0_pose (the
    previous solution on tracker updates) or at the current pose; `held` / `passive` from held_model /
    passive_model. -> (P_ee*, terms with "collision" = gripper + held, ok)."""
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
            losses(
                _pose(x, T_ee0), T_ee0, c, phi, T_passive, tree, width, held, passive
            ).values()
        )

    res = minimize(f, x0, method="SLSQP", options={"maxiter": 500, "ftol": 1e-9})
    T = _pose(res.x, T_ee0)
    terms = losses(T, T_ee0, c, phi, T_passive, tree, width, held, passive)
    terms["collision"] = terms["collision_gripper"] + terms["collision_held"]
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
