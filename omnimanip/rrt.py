"""RRT-Connect (Kuffner & LaValle 2000) in joint space, `motion.plan(..., planner="rrt")` (DEVIATIONS #33).

The planner is ours; cuRobo only answers two questions on our server fork: "goals" (the goal IK solutions its own
"plan" would try: nearest q_goal_ref, wrapped by whole turns, distinct) and "check" (which joint configs put a
robot or held-object sphere in contact with the ESDF, or self-collide: contact, since cuRobo's own goals and
paths sit inside its 5 mm activation band). No TSRs and no constrained planner (onboarding notes, Sec. 1).

Trees grow by EXTEND_STEP joint-space steps toward a uniform sample in the joint limits; the other tree then
connects greedily along the straight line, all of it checked in one call. Every edge is checked at RES (max joint
change between checked configs). The path is pruned greedily (each node straight to the farthest free one), then
shortcut N_SHORTCUT times at random, and returned densified to RES, so every waypoint sent to the arm was checked.
"""

import time

import numpy as np

EXTEND_STEP = 0.3  # rad (joint-space norm): one tree step toward a sample
RES = 0.01  # rad: the largest joint change between checked configs on an edge (~1 cm at the tool)
N_SHORTCUT = 80  # random shortcut attempts on the found path
TIMEOUT = 10.0  # s: the search budget per plan, shared across goals
SEED = 0  # fixed: the same scene and request give the same path
# UR5e joint limits (UR order): +-2 pi, elbow +-pi (as the server's goal wrapping)
LIMITS = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])


def _edge(qa, qb):
    """Configs from qa (exclusive) to qb (inclusive), at most RES apart in every joint."""
    n = max(1, int(np.ceil(np.abs(qb - qa).max() / RES)))
    return qa + (qb - qa) * (np.arange(1, n + 1)[:, None] / n)


class _Tree:
    def __init__(self, q):
        self.q, self.parent = [np.asarray(q, float)], [-1]

    def nearest(self, q):
        return int(np.argmin(np.linalg.norm(np.asarray(self.q) - q, axis=1)))

    def add(self, q, parent):
        self.q.append(q)
        self.parent.append(parent)
        return len(self.q) - 1

    def branch(self, i):
        out = []
        while i != -1:
            out.append(self.q[i])
            i = self.parent[i]
        return out  # from node i back to the root


def _grow(tree, q_target, collides, step):
    """From tree's nearest node toward q_target: at most `step` (EXTEND) or all the way (CONNECT, step=None), cut
    at the first colliding config; new nodes every EXTEND_STEP. -> (last node index, reached q_target)."""
    i = tree.nearest(q_target)
    q0 = tree.q[i]
    d = q_target - q0
    n = np.linalg.norm(d)
    if n < 1e-9:
        return i, True
    end = q_target if step is None or n <= step else q0 + d * (step / n)
    pts = _edge(q0, end)
    bad = np.flatnonzero(collides(pts))
    free = pts if not len(bad) else pts[: bad[0]]
    if not len(free):
        return i, False
    # nodes along the free part, EXTEND_STEP apart, ending at its last free config
    k = max(1, int(np.ceil(np.linalg.norm(free[-1] - q0) / EXTEND_STEP)))
    for j in range(1, k + 1):
        i = tree.add(q0 + (free[-1] - q0) * (j / k), i)
    return i, not len(bad) and end is q_target


def connect(q_start, q_goal, collides, deadline, rng):
    """RRT-Connect from q_start to q_goal (both collision-free). -> (path (n, 6) of tree nodes, or None; tree sizes)."""
    trees = [_Tree(q_start), _Tree(q_goal)]
    a = 0
    while time.time() < deadline:
        q_rand = rng.uniform(-LIMITS, LIMITS)
        ia, _ = _grow(trees[a], q_rand, collides, EXTEND_STEP)
        ib, reached = _grow(trees[1 - a], trees[a].q[ia], collides, None)
        if reached:
            pa, pb = trees[a].branch(ia)[::-1], trees[1 - a].branch(ib)[1:]
            path = np.array(pa + pb)  # root of a ... meeting point ... root of b
            return (path if a == 0 else path[::-1]), [len(t.q) for t in trees]
        a = 1 - a
    return None, [len(t.q) for t in trees]


def prune(path, collides):
    """Greedy shortcut: from each kept node, straight to the farthest later node whose edge is free."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and collides(_edge(path[i], path[j])).any():
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def shortcut(path, collides, rng):
    """N_SHORTCUT random straight shortcuts between two path nodes, kept when the straight edge is free."""
    for _ in range(N_SHORTCUT):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if not collides(_edge(path[i], path[j])).any():
            path = np.vstack([path[: i + 1], path[j:]])
    return path


def densify(path):
    return np.vstack([path[:1]] + [_edge(a, b) for a, b in zip(path[:-1], path[1:])])


def plan(q, goals, collides, timeout=TIMEOUT):
    """Joint path (n, 6) from q to the first of `goals` (UR order, nearest first) RRT-Connect reaches within the
    shared timeout. `collides(Q)` -> bool per config. -> (path or None, reply dict like the cuRobo planner's)."""
    rng, deadline = np.random.default_rng(SEED), time.time() + timeout
    q = np.asarray(q, float)
    if collides(q[None])[0]:
        return None, {"reason": "start configuration in collision", "goal_failures": []}
    failures, sizes = [], []
    for g in np.asarray(goals, float):
        if time.time() >= deadline:
            failures.append(f"q_goal {np.round(g, 2).tolist()}: not tried, the {timeout:.0f} s budget ran out")
            continue
        if collides(g[None])[0]:
            failures.append(f"q_goal {np.round(g, 2).tolist()}: in collision")
            continue
        path, size = connect(q, g, collides, deadline, rng)
        sizes.append(size)
        if path is not None:
            n_nodes = len(path)
            path = densify(shortcut(prune(path, collides), collides, rng))
            return path, {"reason": "", "goal_failures": failures, "tree_sizes": sizes, "nodes": n_nodes}
        failures.append(f"q_goal {np.round(g, 2).tolist()}: RRT-Connect found no path (trees {size})")
    return None, {
        "reason": "RRT-Connect failed: " + " | ".join(failures),
        "goal_failures": failures,
        "tree_sizes": sizes,
    }
