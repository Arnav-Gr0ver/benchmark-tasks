"""The buzz wire's path and the geometry the task, its scorer and its demo share. Pure numpy.

World frame: x forward from the arm bases, y left, z up (metres); heights are above the table top.
"""

from __future__ import annotations

import math

import numpy as np

# The wire's centre line runs through these points (a cubic B-spline; four in a row keep the ends straight), from the
# start post to the end post: a straight start away from the robot (the green sleeve); a bend a quarter turn to the
# robot's left, climbing 3.5 cm; a crest that swings round to the right, a half turn, and drops back down; a bend a
# quarter turn to the left; and a straight finish away from the robot (the red sleeve). The bends turn the wire 90
# degrees each way from the ring's starting direction, so a ring that is not turned with them touches the wire.
SHIFT = (0.0, 0.0)                         # moves the whole track (x, y)
H = 0.06                                   # the wire's height above the table, at both ends
CONTROL = [(0.15, 0.0, H), (0.165, 0.0, H), (0.18, 0.0, H), (0.195, 0.0, H), (0.225, 0.0, H), (0.22, 0.045, H + 0.02),
           (0.24, 0.075, H + 0.035), (0.275, 0.07, H + 0.03), (0.285, 0.03, H + 0.012), (0.28, -0.01, H),
           (0.305, -0.04, H), (0.325, -0.04, H), (0.34, -0.04, H), (0.355, -0.04, H)]
WIRE_R = 0.0015                            # 3 mm steel wire
SLEEVE_R = 0.0035                          # 7 mm insulating sleeves at the start (green) and the finish (red)
GREEN_LEN, RED_LEN = 0.035, 0.03           # their lengths along the wire: touching them does not count
RING_R = 0.0135                            # the ring's centre line: 24 mm inside, 30 mm outside
TUBE_R = 0.0015                            # 3 mm thick
TOUCH = WIRE_R + TUBE_R                    # the ring touches the bare wire when its centre line comes this close
STEP = 0.001                               # the path is sampled every 1 mm


def _bspline(points, n=200) -> np.ndarray:
    """A uniform cubic B-spline over the control points, its ends clamped to the first and last point. It stays
    inside the control polygon (no overshoot), and four control points in a row give a straight run."""
    p = np.asarray(points, float)
    p = np.vstack([p[0], p[0], p, p[-1], p[-1]])
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i:i + 4]
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            b = np.array([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4, -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3]) / 6
            out.append(b[0] * p0 + b[1] * p1 + b[2] * p2 + b[3] * p3)
    out.append(p[-1])
    return np.array(out)


def _resample(curve: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.linspace(0.0, s[-1], int(round(s[-1] / step)) + 1)
    return np.stack([np.interp(grid, s, curve[:, k]) for k in range(3)], axis=1)


PATH = _resample(_bspline(CONTROL), STEP) + np.array([*SHIFT, 0.0])  # (n, 3), relative to the table top in z
ARC = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(PATH, axis=0), axis=1))])
LENGTH = float(ARC[-1])
S_GREEN, S_RED = GREEN_LEN, LENGTH - RED_LEN          # the bare, scored wire runs between these
CHECKPOINTS = [S_GREEN + k / 5 * (S_RED - S_GREEN) for k in range(1, 6)]   # 20, 40, 60, 80, 100 %


def point(s: float) -> np.ndarray:
    return np.array([np.interp(s, ARC, PATH[:, k]) for k in range(3)])


def tangent(s: float, h: float = 0.004) -> np.ndarray:
    a, b = point(max(0.0, s - h)), point(min(LENGTH, s + h))
    return (b - a) / np.linalg.norm(b - a)


_YAW = np.unwrap(np.arctan2(np.gradient(PATH[:, 1]), np.gradient(PATH[:, 0])))   # the wire's heading, continuous


def heading(s: float) -> float:
    return float(np.interp(s, ARC, _YAW))


def ring_frame(s: float, follow: float = 1.0) -> np.ndarray:
    """A ring orientation at arc length s: its axis (x) pitched with the wire and turned ``follow`` times the wire's
    heading (1: along the wire), its handle (-y) level and square to the axis; z completes the frame."""
    t = tangent(s)
    yaw = follow * heading(s)
    pitch = math.asin(max(-1.0, min(1.0, float(t[2]))))
    axis = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    handle = np.cross(axis, [0.0, 0.0, 1.0])
    handle /= np.linalg.norm(handle)
    return np.column_stack([axis, -handle, np.cross(axis, -handle)])


def ring_points(centre, rot, n: int = 90) -> np.ndarray:
    """Points on the ring's centre line (a circle of RING_R in the ring's y-z plane)."""
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    local = np.stack([np.zeros(n), RING_R * np.cos(a), RING_R * np.sin(a)], axis=1)
    return np.asarray(centre, float) + local @ np.asarray(rot, float).T


def _segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to each segment a[j]-b[j]: (n_points, n_segments)."""
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


BARE = (ARC[:-1] >= S_GREEN) & (ARC[1:] <= S_RED)    # the segments of bare wire


def clearance(centre, rot) -> float:
    """The smallest distance between the ring's centre line and the bare wire's centre line (m)."""
    pts = ring_points(centre, rot)
    near = np.linalg.norm(PATH - np.asarray(centre, float), axis=1) < RING_R + 0.02
    seg = BARE & near[:-1] & near[1:]
    if not seg.any():
        return math.inf
    return float(_segment_distances(pts, PATH[:-1][seg], PATH[1:][seg]).min())


def progress(centre) -> float:
    """How far along the wire the ring is: the arc length of the wire point nearest its centre (m)."""
    return float(ARC[int(np.argmin(np.linalg.norm(PATH - np.asarray(centre, float), axis=1)))])
