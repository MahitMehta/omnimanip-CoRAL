"""The arm server's camera to an mp4, in its own process (as OmniManip's execute_node --record): a thread in the loop's
process starves while ReKep's solvers hold the GIL, freezing the video for whole solves. Stops on SIGTERM
(Ctrl-C is left to the loop) or when the loop's process is gone.

    python -m rekep.record tcp://127.0.0.1:5684 out.mp4 [fps]
"""

import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from manip_bridge.zmq_client import SidecarClient


def main(addr, path, fps=10.0):
    host, port = addr.rsplit(":", 1)
    frames, out, stop = SidecarClient(f"{host}:{int(port) + 1}", 10_000), None, []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(
        signal.SIGINT, signal.SIG_IGN
    )  # Ctrl-C reaches the loop, which ends this; the mp4 stays playable
    parent = os.getppid()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    while not stop and os.getppid() == parent:  # or the loop's process is gone
        t0 = time.time()
        try:
            rgb = np.asarray(frames.call({"cmd": "frame"})["rgb"], np.uint8)
        except Exception:  # noqa: BLE001 -- a lost frame
            rgb = None
        if rgb is not None:
            if out is None:
                out = cv2.VideoWriter(
                    str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, rgb.shape[1::-1]
                )
            out.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        time.sleep(max(0.0, 1 / fps - (time.time() - t0)))
    if out is not None:
        out.release()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(float, sys.argv[3:4]))
