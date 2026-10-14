"""Shared geometry helpers for silhouettes."""
import numpy as np
from skimage.graph import MCP_Geometric

MAX_GRID = 400


def _snap(mask, p):
    ys, xs = np.nonzero(mask)
    i = np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2)
    return int(ys[i]), int(xs[i])


def geodesic(mask, p1, p2):
    """Shortest path from p1 to p2 (x, y) staying inside `mask`.

    Body length is measured on a stretched-out bird, so for curled necks this follows the
    body better than a straight line. Returns (length_px, [[x, y], ...] simplified path)."""
    h, w = mask.shape
    s = min(1.0, MAX_GRID / max(h, w))
    if s < 1:
        import cv2
        small = cv2.resize(mask.astype(np.uint8), (max(1, round(w * s)), max(1, round(h * s))),
                           interpolation=cv2.INTER_AREA) > 0
    else:
        small = mask
    a = _snap(small, (p1[0] * s, p1[1] * s))
    b = _snap(small, (p2[0] * s, p2[1] * s))
    cost = np.where(small, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    try:
        mcp.find_costs([a], [b])
        path = np.array(mcp.traceback(b), dtype=float)  # (row, col)
    except ValueError:  # endpoints in disconnected parts
        path = np.zeros((0, 2))
    if len(path) < 2:
        return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])), [list(p1), list(p2)]
    # Smooth the staircase path so its length isn't inflated by pixel steps
    k = max(1, len(path) // 40)
    if len(path) > 2 * k + 2:
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        sm = np.stack([np.convolve(np.pad(path[:, i], k, mode="edge"), kernel, "valid") for i in range(2)], 1)
        sm[0], sm[-1] = path[0], path[-1]
    else:
        sm = path
    xy = sm[:, ::-1] / s
    xy[0], xy[-1] = p1, p2
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    step = max(1, len(xy) // 30)
    pts = np.vstack([xy[::step], xy[-1:]])
    return length, [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def _downsample(mask):
    import cv2
    h, w = mask.shape
    s = min(1.0, MAX_GRID / max(h, w))
    if s == 1.0:
        return mask, s
    small = cv2.resize(mask.astype(np.uint8), (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA) > 0
    return small, s


def geodesic_diameter(mask):
    """Approximate the two points farthest apart *within* the shape (double sweep).
    For a bird with legs removed this is usually bill tip to tail tip, even with a cocked
    tail or curled neck. Returns ((x, y), (x, y), tangent_a, tangent_b) in mask coords, where
    tangents are unit vectors pointing outward along the path at each end."""
    small, s = _downsample(mask)
    cost = np.where(small, 1.0, np.inf)
    ys, xs = np.nonzero(small)
    seed = (int(ys[len(ys) // 2]), int(xs[len(xs) // 2]))

    def farthest(src):
        mcp = MCP_Geometric(cost, fully_connected=True)
        c, _ = mcp.find_costs([src])
        c = np.where(np.isfinite(c), c, -1)
        return np.unravel_index(np.argmax(c), c.shape), mcp

    a, _ = farthest(seed)
    b, mcp = farthest(a)
    path = np.array(mcp.traceback(b), dtype=float)[:, ::-1] / s  # (x, y), from a to b
    k = max(2, len(path) // 12)

    def tangent(end, inner):
        v = end - inner
        return v / (np.linalg.norm(v) + 1e-9)

    pa, pb = path[0], path[-1]
    return pa, pb, tangent(pa, path[min(k, len(path) - 1)]), tangent(pb, path[max(-k - 1, -len(path))])
