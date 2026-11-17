"""ReKep's point tracker (paper A.7) on our one camera: DINOv2 features as in keypoint proposal (A.5); each keypoint's
reference feature is the mean feature of the points within 2 cm of it, fixed for the task; each frame, the 100 pixels
most cosine-similar to it (at least 0.6), outliers rejected by median deviation (m = 2), their mean point; a uniform
filter over the last 10 positions. A keypoint only matches points within GATE of its last position (the paper
searches the whole image; alike parts of other objects, e.g. a cup's handle for a teapot's, pulled keypoints onto
them)."""

from collections import deque

import numpy as np
import torch

from . import core as C

REF_RADIUS, TOP, MIN_SIM, M_DEV, WINDOW = 0.02, 100, 0.6, 2.0, 10  # paper A.7
GATE = 0.10  # m


class Tracker:
    def __init__(self, proposer, K, T_base_cam):
        self.kp, self.K, self.T = (
            proposer,
            np.asarray(K, float),
            np.asarray(T_base_cam, float),
        )

    def _features(self, rgb, depth):
        """Unit DINOv2 features and base_link points of the pixels with depth."""
        t, _, _, _, info = self.kp._preprocess(rgb, None, [])
        F = self.kp._get_features(t, info)
        ok = torch.from_numpy((depth > 0).reshape(-1)).to(F.device)
        return torch.nn.functional.normalize(F[ok].float(), dim=-1), C.points(
            depth, self.K, self.T
        )[depth > 0]

    def start(self, rgb, depth, keypoints):
        F, P = self._features(rgb, depth)
        refs = []
        for k in np.asarray(keypoints, float):
            d = np.linalg.norm(P - k, axis=1)
            sel = np.flatnonzero(d <= REF_RADIUS)
            sel = (
                sel if len(sel) else [int(np.argmin(d))]
            )  # no depth within 2 cm: the nearest point's
            refs.append(F[torch.as_tensor(sel, device=F.device)].mean(0))
        self.ref = torch.nn.functional.normalize(torch.stack(refs), dim=-1)
        self.hist = deque([np.asarray(keypoints, float)], maxlen=WINDOW)
        return self.positions()

    @torch.inference_mode()
    def update(self, rgb, depth):
        F, P = self._features(rgb, depth)
        sim = F @ self.ref.T  # pixels x keypoints
        far = (
            torch.cdist(
                torch.from_numpy(P).to(sim), torch.from_numpy(self.hist[-1]).to(sim)
            )
            > GATE
        )
        sim = sim.masked_fill(far, -torch.inf)
        val, idx = sim.topk(min(TOP, len(F)), dim=0)
        val, idx = val.cpu().numpy(), idx.cpu().numpy()
        now = self.hist[-1].copy()
        for k in range(sim.shape[1]):
            pts = P[idx[val[:, k] >= MIN_SIM, k]]
            if len(pts):
                dev = np.linalg.norm(pts - np.median(pts, 0), axis=1)
                mdev = np.median(dev)
                now[k] = pts[dev <= M_DEV * mdev].mean(0) if mdev > 0 else pts.mean(0)
        self.hist.append(now)  # a keypoint with no match keeps its last position
        return self.positions()

    def positions(self):
        return np.mean(self.hist, axis=0)
