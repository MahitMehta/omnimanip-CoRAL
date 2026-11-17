"""ReKep (Huang et al., CoRL 2024) on the CoRAL UR5e: the few adapters ReKep's own modules (third_party/ReKep, pinned)
need to run on our robot and frames. base_link plays ReKep's world frame. The end effector is AnyGrasp's grasp
frame with the fingertips at its origin (x = approach, as ReKep's grasp cost and grasp_depth assume)."""

import copy
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import transform_utils as TU
from keypoint_proposal import KeypointProposer
from kmeans_pytorch import kmeans
from numba import njit
from path_solver import PathSolver
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from subgoal_solver import SubgoalSolver
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
    c["keypoint_proposer"]["min_dist_bt_keypoints"] = 0.06  # rekep/DEVIATIONS.md
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

    def _cluster_features(self, points, features_flat, masks):
        """Paper A.5: per mask, k-means (k = 5, cosine) on its DINOv2 features; a centroid's candidate is the mask's
        pixel nearest it (cosine), at that pixel's point. Larger masks first, the order _merge_clusters keeps."""
        k, out = self.config["num_candidates_per_mask"], []
        for j in sorted(range(len(masks)), key=lambda j: -masks[j].sum()):
            m = masks[j]
            if m.mean() > self.config["max_mask_ratio"]:
                continue
            sel = torch.from_numpy(m.reshape(-1)).to(features_flat.device)
            F = torch.nn.functional.normalize(features_flat[sel].float(), dim=-1)
            _, centers = kmeans(
                X=F, num_clusters=k, distance="cosine", device=self.device
            )
            best = (
                (F @ torch.nn.functional.normalize(centers.to(F), dim=-1).T)
                .argmax(0)
                .cpu()
                .numpy()
            )
            out += [(points[m][i], np.argwhere(m)[i], j) for i in best]
        kps, pxs, ids = (np.array(x) for x in zip(*out))
        return kps, pxs, ids

    def _merge_clusters(self, candidate_keypoints):
        """Paper A.5: candidates within min_dist_bt_keypoints of a kept one are filtered out."""
        keep = []
        for i, p in enumerate(candidate_keypoints):
            if (
                not keep
                or np.linalg.norm(candidate_keypoints[keep] - p, axis=1).min()
                >= self.config["min_dist_bt_keypoints"]
            ):
                keep.append(i)
        return keep

    def _project_keypoints_to_img(
        self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat
    ):
        self.pixels, self.groups = candidate_pixels, candidate_rigid_group_ids
        return label_keypoints(rgb, candidate_pixels)


def label_keypoints(rgb, pixels):
    """ReKep's numbered keypoint boxes (red number in a white box centred on the pixel) at about half their size,
    so a box covers less of what lies around its keypoint."""
    img = rgb.copy()
    for i, (r, c) in enumerate(pixels):
        w, h = 10 + 7 * len(str(i)), 16
        cv2.rectangle(
            img, (c - w // 2, r - h // 2), (c + w // 2, r + h // 2), (255, 255, 255), -1
        )
        cv2.rectangle(
            img, (c - w // 2, r - h // 2), (c + w // 2, r + h // 2), (0, 0, 0), 1
        )
        cv2.putText(
            img,
            str(i),
            (c - 4 * len(str(i)), r + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )
    return img


def proposer(cfg):
    kp = Proposer(cfg["keypoint_proposer"])
    m: torch.nn.Module = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")  # type: ignore[assignment]
    kp.dinov2 = m.eval().to(kp.device)  # paper A.5: DINOv2 with registers
    return kp


def object_id(masks, px, max_ratio):
    """The object a keypoint belongs to (its rigid group once grasped): the index of the largest SAM mask holding its
    pixel (row, col), among masks the proposer clusters (at most max_ratio of the image)."""
    area = masks.sum((1, 2))
    ok = masks[:, px[0], px[1]] & (area <= max_ratio * masks[0].size)
    return int(np.flatnonzero(ok)[np.argmax(area[ok])])


def object_mask(masks, px, max_ratio):
    return masks[object_id(masks, px, max_ratio)]


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


# --- compiled stand-ins (same results) for the solvers' hot paths: IK, rotation conversions, SDF lookup ---
_CHAIN = motion._chain()  # base_link -> tool0: [(origin, joint name or None, axis)]
assert [n for _, n, _ in _CHAIN if n] == motion.UR_JOINTS
assert all(a is None or np.allclose(a, [0, 0, 1]) for _, _, a in _CHAIN)
_O = np.array([O for O, _, _ in _CHAIN])
_J = np.cumsum([n is not None for _, n, _ in _CHAIN]) - 1
_J = np.where([n is not None for _, n, _ in _CHAIN], _J, -1)


@njit(cache=True)
def _fk(q, O, J):
    """motion.fk (base_T_tool0) plus each joint's position and z axis in base_link (the geometric Jacobian's)."""
    T, P, A = np.eye(4), np.zeros((6, 3)), np.zeros((6, 3))
    for k in range(O.shape[0]):
        T = T @ O[k]
        j = J[k]
        if j >= 0:
            P[j], A[j] = T[:3, 3], T[:3, 2]
            Rz = np.eye(4)
            Rz[0, 0] = Rz[1, 1] = np.cos(q[j])
            Rz[1, 0] = np.sin(q[j])
            Rz[0, 1] = -Rz[1, 0]
            T = T @ Rz
    return T, P, A


@njit(cache=True)
def _cross(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@njit(cache=True)
def _rotvec(R):
    th = np.arccos(min(1.0, max(-1.0, (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return v / 2
    if th > np.pi - 1e-3:  # axis from R + I's largest column, signed as v
        B = R + np.eye(3)
        c = np.argmax(np.array([B[:, i] @ B[:, i] for i in range(3)]))
        a = B[:, c] / np.sqrt(B[:, c] @ B[:, c])
        return a * th if a @ v >= 0 else -a * th
    return v * th / (2 * np.sin(th))


@njit(cache=True)
def _ik(G, q, O, J, n_max, tol_t, tol_r, lam2):
    """motion.ik's damped least squares, one step per descent; -> (q, descents, converged, position error)."""
    n = 0
    while True:
        T, P, A = _fk(q, O, J)
        e = np.empty(6)
        e[:3] = G[:3, 3] - T[:3, 3]
        e[3:] = _rotvec(
            np.ascontiguousarray(G[:3, :3]) @ np.ascontiguousarray(T[:3, :3].T)
        )
        ok = np.sqrt(e[:3] @ e[:3]) < tol_t and np.sqrt(e[3:] @ e[3:]) < tol_r
        if ok or n == n_max:
            return q, n, ok, np.sqrt(e[:3] @ e[:3])
        Jm = np.empty((6, 6))
        for j in range(6):
            Jm[:3, j] = _cross(A[j], T[:3, 3] - P[j])
            Jm[3:, j] = A[j]
        q = q + Jm.T @ np.linalg.solve(Jm @ Jm.T + lam2 * np.eye(6), e)
        n += 1


class IK:
    """ReKep's IKSolver (ik_solver.py: Lula CCD on the Fetch) on our planner URDF: motion.ik's damped least squares
    and tolerances with the Jacobian from the joint axes, compiled; from initial_joint_pos (q0 when None, as in the
    solvers' warm-up)."""

    def __init__(self, q0, tol_t=1e-4, tol_r=1e-3, damping=0.01):
        self.q0, self.tol_t, self.tol_r, self.lam2 = (
            np.asarray(q0, float),
            tol_t,
            tol_r,
            damping**2,
        )

    def solve(self, target_pose_homo, max_iterations=20, initial_joint_pos=None, **_):
        G = np.ascontiguousarray(np.asarray(target_pose_homo, float) @ EE_T_TOOL0)
        q0 = np.array(
            self.q0 if initial_joint_pos is None else initial_joint_pos, float
        )
        q, n, ok, err = _ik(
            G, q0, _O, _J, max_iterations, self.tol_t, self.tol_r, self.lam2
        )
        return SimpleNamespace(
            success=ok, num_descents=n, cspace_position=q, position_error=err
        )


@njit(cache=True)
def _quat2mat(q):
    x, y, z, w = q / np.sqrt(q @ q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@njit(cache=True)
def _mat2quat(m):
    """scipy's Rotation.from_matrix(m).as_quat() for a rotation matrix (its branch on the largest of the diagonal
    and the trace, no sign canonicalisation)."""
    d = np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 0] + m[1, 1] + m[2, 2]])
    c, q = np.argmax(d), np.empty(4)
    if c != 3:
        i, j, k = c, (c + 1) % 3, (c + 2) % 3
        q[i] = 1 - d[3] + 2 * m[i, i]
        q[j] = m[j, i] + m[i, j]
        q[k] = m[k, i] + m[i, k]
        q[3] = m[k, j] - m[j, k]
    else:
        q[0], q[1], q[2], q[3] = (
            m[2, 1] - m[1, 2],
            m[0, 2] - m[2, 0],
            m[1, 0] - m[0, 1],
            1 + d[3],
        )
    return q / np.sqrt(q @ q)


@njit(cache=True)
def _euler2quat(e):
    """scipy's Rotation.from_euler("xyz", e).as_quat(): extrinsic x, then y, then z."""
    cx, sx, cy, sy, cz, sz = (
        np.cos(e[0] / 2),
        np.sin(e[0] / 2),
        np.cos(e[1] / 2),
        np.sin(e[1] / 2),
        np.cos(e[2] / 2),
        np.sin(e[2] / 2),
    )
    return np.array(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ]
    )


@njit(cache=True)
def _trilinear(V, lo, step, X):
    """scipy RegularGridInterpolator (linear, fill 0 outside) on the grid lo + i * step."""
    out = np.zeros(X.shape[0])
    n = V.shape
    for p in range(X.shape[0]):
        f = (X[p] - lo) / step
        if np.any(f < 0) or f[0] > n[0] - 1 or f[1] > n[1] - 1 or f[2] > n[2] - 1:
            continue
        i = np.minimum(
            np.floor(f).astype(np.int64), np.array([n[0] - 2, n[1] - 2, n[2] - 2])
        )
        t = f - i
        acc = 0.0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    w = (
                        (t[0] if a else 1 - t[0])
                        * (t[1] if b else 1 - t[1])
                        * (t[2] if c else 1 - t[2])
                    )
                    acc += w * V[i[0] + a, i[1] + b, i[2] + c]
        out[p] = acc
    return out


def _one_or_scipy(fast, slow, k):
    """A transform_utils conversion: the compiled one for a single input of k numbers, else the original."""
    return lambda x: (
        fast(np.ascontiguousarray(x, float)) if np.ndim(x) == k else slow(x)
    )


def _setup_sdf(self, sdf_voxels):
    """The solvers' _setup_sdf (a RegularGridInterpolator on linspace(bounds_min, bounds_max, shape)), compiled."""
    lo, hi = (
        np.array(self.config["bounds_min"], float),
        np.array(self.config["bounds_max"], float),
    )
    V, step = (
        np.ascontiguousarray(sdf_voxels, float),
        (hi - lo) / (np.array(sdf_voxels.shape) - 1),
    )
    return lambda X: _trilinear(V, lo, step, np.ascontiguousarray(X, float))


TU.quat2mat = _one_or_scipy(_quat2mat, TU.quat2mat, 1)
TU.mat2quat = _one_or_scipy(_mat2quat, TU.mat2quat, 2)
TU.euler2quat = _one_or_scipy(_euler2quat, TU.euler2quat, 1)
SubgoalSolver._setup_sdf = PathSolver._setup_sdf = _setup_sdf  # type: ignore[assignment]
