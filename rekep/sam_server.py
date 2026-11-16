"""SAM (v1) "segment everything" ZMQ REP server for ReKep's keypoint proposal (paper A.5: all masks in the scene from
Segment Anything). SamAutomaticMaskGenerator with its defaults on ViT-H. Same wire protocol as the CoRAL sidecars
(msgpack + msgpack_numpy):

    {"cmd": "ping"} -> {"ok": True}
    {"cmd": "everything", "rgb": HxWx3 uint8} -> {"ok": True, "masks": NxHxW bool, "scores": N float32 (predicted IoU)}

Runs on CoRAL as container SamServer-mahit (README "ReKep pipeline"); env SAM_CKPT, SAM_PORT (default 5683).
"""

import os
import traceback

import msgpack
import msgpack_numpy
import numpy as np
import torch
import zmq
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

msgpack_numpy.patch()
amg = SamAutomaticMaskGenerator(
    sam_model_registry["vit_h"](checkpoint=os.environ["SAM_CKPT"]).to("cuda")
)
sock = zmq.Context().socket(zmq.REP)
sock.bind(f"tcp://*:{os.getenv('SAM_PORT', '5683')}")
print("[sam-server] listening on", os.getenv("SAM_PORT", "5683"), flush=True)
while True:
    req = msgpack.unpackb(sock.recv(), raw=False)
    try:
        if req["cmd"] == "ping":
            rep = {"ok": True}
        elif req["cmd"] == "everything":
            with torch.inference_mode():
                out = amg.generate(np.asarray(req["rgb"], np.uint8))
            rep = {
                "ok": True,
                "masks": np.array([m["segmentation"] for m in out], bool),
                "scores": np.array([m["predicted_iou"] for m in out], np.float32),
            }
        else:
            rep = {"ok": False, "error": f"unknown cmd {req['cmd']!r}"}
    except Exception:  # noqa: BLE001 -- reported to the client as ok=False
        rep = {"ok": False, "error": traceback.format_exc()}
    sock.send(msgpack.packb(rep, use_bin_type=True))
