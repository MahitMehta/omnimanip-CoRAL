"""Mesh cache (run_task --mesh-cache): an object's metric mesh (TRELLIS.2 -> Any6D, from an earlier run) reused by
name, so registration skips TRELLIS.2 and Any6D and only FoundationPose registers it in the new scene. Used only when
the segmented points match its size (height above the table, footprint), and kept only when the registered mesh's
silhouette matches the mask; otherwise the object is generated afresh. PoseServer reads meshes from its own
container, so an entry is copied into it (docker cp) under /tmp, and again after a PoseServer restart.

    uv run python -m omnimanip.mesh_cache list
    uv run python -m omnimanip.mesh_cache set <name> <run_dir>      # that run's mesh of <name> becomes the entry
    uv run python -m omnimanip.mesh_cache drop <name>
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import ROOT

CACHE_DIR = ROOT / "outputs" / "mesh_cache"
POSE_CONTAINER, POSE_DIR = "PoseServer", "/tmp/omnimanip_mesh_cache"
SIZE_TOL, MIN_IOU = 0.015, 0.7  # m: observed vs cached height / footprint; silhouette-vs-mask IoU to keep a mesh
FILES = ("mesh.obj", "material.mtl", "material_0.png")


def _height(P, plane):
    return P @ np.asarray(plane["n"], float) + float(plane["d"])  # camera on the positive side: up is positive


def _footprint(P, plane):
    """Largest horizontal spread (m) of points P (camera frame) on the table plane, 2-98th percentile."""
    n = np.asarray(plane["n"], float)
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    uv = np.c_[P @ a, P @ b]
    return float(max(np.ptp(np.percentile(uv @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]), [2, 98], axis=0), 0).max()
                     for t in np.linspace(0, np.pi / 2, 7)))


def observed_size(depth, K, mask, plane):
    """(height of the object's top above the table, footprint) from its segmented depth, in m."""
    v, u = np.nonzero(mask & (depth > 0.05) & (depth < 3.0))
    z = depth[v, u]
    P = np.c_[(u - K[0][2]) * z / K[0][0], (v - K[1][2]) * z / K[1][1], z]
    return float(np.percentile(_height(P, plane), 98)), _footprint(P, plane)


def mesh_size(V, cam_T_obj, plane):
    """(height, footprint) of mesh vertices V (object frame) placed at cam_T_obj, in m."""
    P = V @ np.asarray(cam_T_obj)[:3, :3].T + np.asarray(cam_T_obj)[:3, 3]
    h = _height(P, plane)
    return float(h.max() - h.min()), _footprint(P, plane)


def entry(name):
    d = CACHE_DIR / name
    return d if (d / "meta.json").exists() and all((d / f).exists() for f in FILES) else None


def lookup(name, depth, K, mask, plane):
    """-> (entry dir or None, reason)."""
    d = entry(name)
    if d is None:
        return None, f"no cached mesh for {name!r}"
    m = json.loads((d / "meta.json").read_text())
    h, f = observed_size(depth, K, mask, plane)
    if abs(h - m["height_m"]) > SIZE_TOL or f > m["footprint_m"] + SIZE_TOL:
        return None, (f"{name!r} looks different from the cached one: {100 * h:.1f} cm tall, {100 * f:.1f} cm across "
                      f"(cached {100 * m['height_m']:.1f}, {100 * m['footprint_m']:.1f})")
    return d, f"cached mesh of {name!r} from {m['source_run']}"


def in_pose_container(d):
    """The entry's mesh path inside PoseServer's container, copying the entry there if it isn't yet."""
    m = json.loads((d / "meta.json").read_text())
    dst = f"{POSE_DIR}/{d.name}_{m['id']}"
    if subprocess.run(["docker", "exec", POSE_CONTAINER, "test", "-f", f"{dst}/mesh.obj"]).returncode != 0:
        subprocess.run(["docker", "exec", POSE_CONTAINER, "mkdir", "-p", dst], check=True)
        for f in FILES:
            subprocess.run(["docker", "cp", str(d / f), f"{POSE_CONTAINER}:{dst}/{f}"], check=True)
    return f"{dst}/mesh.obj"


def save(name, mesh_obj, cam_T_obj, plane, source_run):
    """A run's metric mesh (OBJ with material.mtl / material_0.png beside it) as the entry for name."""
    from . import shared_perception as S

    src = Path(mesh_obj)
    V, _ = S.read_obj(src)
    h, f = mesh_size(V, cam_T_obj, plane)
    d = CACHE_DIR / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(src, d / "mesh.obj")  # its mtllib is material.mtl, whose map_Kd is material_0.png
    for f_ in ("material.mtl", "material_0.png"):
        shutil.copy(src.parent / f_, d / f_)
    meta = {"name": name, "source_run": str(source_run), "height_m": h, "footprint_m": f, "id": int(time.time())}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for d in sorted(CACHE_DIR.glob("*/meta.json")):
            m = json.loads(d.read_text())
            print(f"{m['name']:10s} {100 * m['height_m']:5.1f} cm tall, {100 * m['footprint_m']:5.1f} cm across, from {m['source_run']}")
    elif cmd == "set":
        name, run = sys.argv[2], Path(sys.argv[3])
        o = json.loads((run / "objects.json").read_text())[name]
        plane = json.loads((run / "packet.json").read_text())["plane"]
        print(save(name, o["final_mesh"].replace("runs/bag0216_", "runs/1bag0216_"), o["cam_T_obj"], plane, run.name))
    elif cmd == "drop":
        shutil.rmtree(CACHE_DIR / sys.argv[2], ignore_errors=True)
