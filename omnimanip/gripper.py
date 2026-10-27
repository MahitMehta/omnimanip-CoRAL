"""Gripper models, in AnyGrasp's grasp frame (origin at the grasp centre between the fingertips; x = approach
into the object, y = closing, z = normal; the fingertips reach `depth` past the origin along x). Swap the gripper
by adding a Gripper here and selecting it in GRIPPER."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cache

import numpy as np
import trimesh

from .config import ROOT


@dataclass(frozen=True, eq=False)  # identity hash: cacheable despite array fields
class Gripper:
    name: str
    max_width: float  # m
    palm_spheres: np.ndarray  # (N, 4): x, y, z, radius in the grasp frame
    finger_spheres: (
        np.ndarray
    )  # (M, 4): the same for the fingers, gripper OPEN (y = +-closing offset)
    tip: float  # m: fingertip distance from the gripper's base frame along its approach axis
    grasp_R_base: np.ndarray  # 3x3: the gripper base frame's axes in the grasp frame
    urdf: str  # frozen-open URDF (links + fixed joints); meshes in assets/<name>/


def spheres(g, width):
    """Collision spheres with the fingers closed to `width` (each side slides in by (max_width - width) / 2)."""
    f = g.finger_spheres.copy()
    f[:, 1] -= np.sign(f[:, 1]) * (g.max_width - width) / 2
    return np.vstack([g.palm_spheres, f])


def grasp_T_base(g, depth):
    """Gripper base frame in the grasp frame: rotated by grasp_R_base, fingertips `depth` past the grasp centre."""
    T = np.eye(4)
    T[:3, :3] = g.grasp_R_base
    T[:3, 3] = [depth - g.tip, 0.0, 0.0]
    return T


def _attr(el, path, key):
    """URDF attribute that must exist (ElementTree returns None for missing ones)."""
    node = el.find(path)
    val = None if node is None else node.get(key)
    if val is None:
        raise ValueError(f"URDF: missing {path}@{key}")
    return val


@cache
def mesh(g, width=None):
    """The gripper assembled from its frozen-open URDF (fixed joints, no rotations), in its base frame. With a
    width, each finger chain (left_* / right_* links) slides in along the opening axis (base x) so the pads are
    `width` apart: the real linkage swings the pads in a small arc, so this is good to a few millimetres."""
    shift = 0.0 if width is None else (g.max_width - width) / 2
    root = ET.parse(ROOT / g.urdf).getroot()
    parent = {
        _attr(j, "child", "link"): (
            _attr(j, "parent", "link"),
            np.array([float(v) for v in _attr(j, "origin", "xyz").split()]),
        )
        for j in root.findall("joint")
    }
    parts = []
    for link in root.findall("link"):
        if link.find("collision/geometry/mesh") is None:
            continue
        stl = _attr(link, "collision/geometry/mesh", "filename").split("/")[-1]
        p, name = np.zeros(3), link.get("name")
        while name in parent:
            name, off = parent[name]
            p += off
        p[0] += shift * (
            ("right" in link.get("name", "")) - ("left" in link.get("name", ""))
        )
        part = trimesh.load(ROOT / "assets" / g.name / stl, force="mesh")
        assert isinstance(part, trimesh.Trimesh)
        parts.append(part.apply_translation(p))
    out = trimesh.util.concatenate(parts)
    assert isinstance(out, trimesh.Trimesh)
    return out


def _robotiq_2f85():
    """Robotiq 2F-85 (frozen URDF from ros-humble-robotiq-description, via Albus's cuRobo setup): base z = approach,
    base x = opening; fingertips at 149.3 mm, inner pads at +-42.5 mm open. The collision spheres approximate it to
    about a centimetre (cuRobo fits its own inside its container)."""
    palm = [[x, 0.0, 0.0, 0.035] for x in (-0.065, -0.095, -0.125)]
    fingers = [[x, s * 0.0475, 0.0, 0.012] for s in (-1, 1) for x in (0.0, -0.025)]
    return Gripper(
        name="robotiq",
        max_width=0.085,
        palm_spheres=np.array(palm),
        finger_spheres=np.array(fingers),
        tip=0.1493,
        grasp_R_base=np.array(
            [[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ),  # base x = grasp y, base y = grasp z, z = x
        urdf="third_party/ur5e-manip-hardware/test/curobo_incontainer/robotiq/robotiq_2f85_frozen.urdf",
    )


GRIPPER = _robotiq_2f85()
