"""Forward kinematics of a serial URDF chain (fixed and revolute joints): the cuRobo server's robot model
(assets/ur5e_robotiq.urdf, the URDF the server plans with), for the sim arm's frames and the executors' IK."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _origin(j):
    o = j.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, :3] = Rotation.from_euler(
            "xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]
        ).as_matrix()
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def chain(urdf, base="base_link", tip="tool0"):
    """[(origin 4x4, joint name or None, axis)] from base to tip."""
    joints = {
        j.find("child").attrib["link"]: j  # pyright: ignore[reportOptionalMemberAccess]
        for j in ET.parse(urdf)
        .getroot()
        .findall("joint")  # top level: not transmissions
    }
    out, link = [], tip
    while link != base:
        j = joints[link]
        ax = j.find("axis")
        axis = (
            None
            if ax is None
            else np.array([float(v) for v in ax.attrib["xyz"].split()])
        )
        out.append(
            (_origin(j), j.get("name") if j.get("type") == "revolute" else None, axis)
        )
        link = j.find("parent").attrib["link"]  # pyright: ignore[reportOptionalMemberAccess]
    return out[::-1]


def fk(ch, q):
    """base_T_tip for joint values q {name: rad}."""
    T = np.eye(4)
    for O, name, axis in ch:
        T = T @ O
        if name is not None:
            R = np.eye(4)
            R[:3, :3] = Rotation.from_rotvec(axis * q[name]).as_matrix()
            T = T @ R
    return T
