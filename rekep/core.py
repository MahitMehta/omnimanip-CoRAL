"""ReKep (Huang et al., CoRL 2024) on the CoRAL UR5e: the few adapters ReKep's own modules (third_party/ReKep, pinned)
need to run on our robot and frames. base_link plays ReKep's world frame. The end effector is AnyGrasp's grasp
frame with the fingertips at its origin (x = approach, as ReKep's grasp cost and grasp_depth assume)."""

import copy
from types import SimpleNamespace

import numpy as np
import torch
from keypoint_proposal import KeypointProposer
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from utils import get_config

from omnimanip import grasp as G
from omnimanip import motion

TOOL0_T_EE = motion.tool0_T_grasp(0.0)
EE_T_TOOL0 = np.linalg.inv(TOOL0_T_EE)
# m: workspace = UR5e reach around the base, from just under the table (depth noise puts table-contact points
# ~1.5 cm under the fitted plane) up (rekep/DEVIATIONS.md)
REACH, BELOW, HEIGHT = 0.85, 0.02, 0.5


def workspace(packet):
    """ReKep's bounds_min/max in base_link: BELOW under the table's height under the base, up HEIGHT, +-REACH."""
    T = np.array(packet["t_base_cam"]["T"])
    n = T[:3, :3] @ packet["plane"]["n"]
    z = -(packet["plane"]["d"] - n @ T[:3, 3]) / n[2]
    return np.array([-REACH, -REACH, z - BELOW]), np.array([REACH, REACH, z + HEIGHT])


def config(lo, hi):
    """ReKep's configs/config.yaml with our workspace (rekep/DEVIATIONS.md) and the paper's real-robot values."""
    c = copy.deepcopy(get_config())
    for k in ("main", "path_solver", "subgoal_solver", "keypoint_proposer"):
        c[k]["bounds_min"], c[k]["bounds_max"] = lo.tolist(), hi.tolist()
    c["keypoint_proposer"]["min_dist_bt_keypoints"] = (
        0.06  # released code's (rekep/DEVIATIONS.md)
    )
    c["keypoint_proposer"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    for k in ("path_solver", "subgoal_solver"):
        c[k]["max_collision_points"] = 30  # paper A.8
    c["path_solver"]["table_z"] = float(
        lo[2] + BELOW
    )  # paper A.9: table clearance (rekep/paper.patch)
    return c


class Proposer(KeypointProposer):
    """ReKep's KeypointProposer on a list of (overlapping) SAM masks instead of a label image, keeping each final
    keypoint's mask index (its rigid group, paper A.5) and pixel."""

    def _preprocess(self, rgb, points, masks):
        t, rgb, points, _, info = super()._preprocess(
            rgb, points, np.zeros(rgb.shape[:2], int)
        )
        return t, rgb, points, list(masks), info

    def _project_keypoints_to_img(
        self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat
    ):
        self.pixels, self.groups = candidate_pixels, candidate_rigid_group_ids
        return super()._project_keypoints_to_img(
            rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat
        )


def proposer(cfg):
    kp = Proposer(cfg["keypoint_proposer"])
    m: torch.nn.Module = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")  # type: ignore[assignment]
    kp.dinov2 = m.eval().to(kp.device)  # paper A.5: DINOv2 with registers
    return kp


def object_mask(masks, px, max_ratio):
    """The object a keypoint belongs to (its rigid group once grasped): the largest SAM mask holding its pixel (row,
    col), among masks the proposer clusters (at most max_ratio of the image)."""
    area = masks.sum((1, 2))
    ok = masks[:, px[0], px[1]] & (area <= max_ratio * masks[0].size)
    return masks[np.flatnonzero(ok)[np.argmax(area[ok])]]


def points(depth, K, T_base_cam):
    """Per-pixel points in base_link (HxWx3), ReKep's cam_obs 'points'."""
    v, u = np.indices(depth.shape)
    p = np.stack(
        [(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1
    )
    return p @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def sdf_voxels(P, lo, hi, res):
    """ReKep's scene SDF (positive inside, negative outside, on the grid its solvers interpolate) from observed
    points: a voxel holding a point is occupied; one view, so what lies behind a surface counts as free."""
    shape = np.ceil((hi - lo) / res).astype(int)
    i = np.round((P - lo) / (hi - lo) * (shape - 1)).astype(int)
    i = i[np.all((i >= 0) & (i < shape), axis=1)]
    occ = np.zeros(shape, bool)
    occ[tuple(i.T)] = True
    inside, outside = (np.asarray(distance_transform_edt(m)) for m in (occ, ~occ))
    return np.where(occ, inside, -outside) * res


def mat(pose):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
    return T


def ee_pose(q):
    """[x y z qx qy qz qw] of the end effector at joints q."""
    T = motion.fk(q) @ TOOL0_T_EE
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]


def gripper_points(pose):
    """ReKep's collision points for the gripper: the real Robotiq mesh (fingers open) at an end-effector pose."""
    V = G._gripper_points()[0]
    T = mat(pose)
    return V @ T[:3, :3].T + T[:3, 3]


class IK:
    """ReKep's IKSolver (ik_solver.py: Lula CCD on the Fetch) on our planner URDF: motion.ik's damped least squares,
    one step per descent, from initial_joint_pos (q0 when None, as in the solvers' warm-up)."""

    def __init__(self, q0):
        self.q0 = np.asarray(q0, float)

    def solve(self, target_pose_homo, max_iterations=20, initial_joint_pos=None, **_):
        T = np.asarray(target_pose_homo) @ EE_T_TOOL0
        q, ok, n = (
            np.asarray(
                self.q0 if initial_joint_pos is None else initial_joint_pos, float
            ),
            False,
            0,
        )
        while n < max_iterations:
            q, ok = motion.ik(T, q, iters=1)
            if ok:
                break
            n += 1
        return SimpleNamespace(
            success=ok,
            num_descents=n,
            cspace_position=q,
            position_error=float(np.linalg.norm(motion.fk(q)[:3, 3] - T[:3, 3])),
        )
