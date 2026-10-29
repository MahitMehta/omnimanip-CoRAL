"""Camera-to-robot calibration check, right after capture. The table's fitted plane, seen in the base frame, is fixed
by the hardware (the base is bolted to it), so a calibration that moved shows up as a different table height under
the base or a different tilt; the overlay projects the arm's links at the capture joints, for the sideways errors
the plane can't show. Writes calibration_check.png into the run.

uv run python -m omnimanip.calib <run_dir>       # the overlay and numbers for any run
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import motion, urdf_fk

TABLE_Z_UNDER_BASE = -0.009  # m: the table plane at base_link's origin (2026-10-29 calibration, checked by overlay)
TABLE_Z_TOL, TILT_MAX_DEG = 0.015, 2.0  # m, deg: a capture outside these stops the run
LINKS = ["base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
         "wrist_3_link", "tool0"]


def table_in_base(plane, T_base_cam):
    """(height of the table plane under base_link's origin (m), its tilt from base +z (deg))."""
    n = T_base_cam[:3, :3] @ np.asarray(plane["n"], float)
    d = float(plane["d"]) - n @ T_base_cam[:3, 3]
    n, d = (n, d) if n[2] > 0 else (-n, -d)
    return -d / n[2], float(np.degrees(np.arccos(min(1.0, n[2]))))


def check(rgb, K, packet):
    """-> (report {table_z, tilt_deg, ok, problems}, overlay RGB)."""
    T = np.asarray(packet["t_base_cam"]["T"], float)
    names = [n.removeprefix("robot0_") for n in packet["joints"]["names"]]
    q = dict(zip(motion.UR_JOINTS, [packet["joints"]["q"][names.index(n)] for n in motion.UR_JOINTS]))
    z, tilt = table_in_base(packet["plane"], T)
    problems = []
    if abs(z - TABLE_Z_UNDER_BASE) > TABLE_Z_TOL:
        problems.append(f"table {100 * z:+.1f} cm under the base, expected {100 * TABLE_Z_UNDER_BASE:+.1f} +- "
                        f"{100 * TABLE_Z_TOL:.1f} cm: the camera calibration moved")
    if tilt > TILT_MAX_DEG:
        problems.append(f"table tilted {tilt:.1f} deg from the base's up axis (max {TILT_MAX_DEG:.0f})")

    P = np.array([urdf_fk.fk(urdf_fk.chain(motion.ROBOT_URDF, "base_link", l), q)[:3, 3] if l != "base_link"
                  else np.zeros(3) for l in LINKS])
    C = (P - T[:3, 3]) @ T[:3, :3]  # base -> camera
    im = np.ascontiguousarray(rgb).copy()
    if (C[:, 2] > 0).all():
        uv = np.round(C[:, :2] / C[:, 2:3] * [K[0][0], K[1][1]] + [K[0][2], K[1][2]]).astype(int)
        for a, b in zip(uv[:-1], uv[1:]):
            cv2.line(im, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 0), 2)
        for p in uv:
            cv2.circle(im, tuple(map(int, p)), 5, (0, 255, 0), -1)
    lines = [f"table under base {100 * z:+.1f} cm (expected {100 * TABLE_Z_UNDER_BASE:+.1f}), tilt {tilt:.1f} deg",
             "green: robot links at the capture joints (should sit on the arm)"] + problems
    cv2.rectangle(im, (0, 0), (im.shape[1], 8 + 20 * len(lines)), (0, 0, 0), -1)
    for k, s in enumerate(lines):
        col = (255, 80, 80) if k >= 2 else (255, 255, 255)
        cv2.putText(im, s, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return {"table_z_under_base_m": z, "tilt_deg": tilt, "ok": not problems, "problems": problems}, im


def save(run_dir, rgb, K, packet):
    """check() plus calibration_check.png in run_dir. -> report."""
    rep, im = check(rgb, K, packet)
    cv2.imwrite(str(Path(run_dir) / "calibration_check.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return rep


if __name__ == "__main__":
    run = Path(sys.argv[1])
    pk = json.loads((run / "packet.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(run / "rgb.png")), cv2.COLOR_BGR2RGB)
    r = save(run, rgb, np.asarray(pk["K"], float), pk)
    print(json.dumps(r, indent=1), "\nwrote", run / "calibration_check.png")
