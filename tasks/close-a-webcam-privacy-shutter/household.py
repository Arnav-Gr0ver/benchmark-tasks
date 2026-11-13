"""Shared layout and checks for the household mechanism tasks: the silicone mat, the collection tray, where a point is
relative to a housekit Mech, contact, "detached" and "stowed". Pure NumPy.

Coordinates. The briefs give millimetres from the mat's centre, x to the robot's right, y away from the robot, z up from
the mat's top. yam_arena's station frame has x away from the robot and y to its left, so a brief's (x, y) is the
station's (MAT_X + y, -x). ``brief_xy`` converts.
"""

from __future__ import annotations

import math

import numpy as np
from yam_arena.task_file import TABLE_TOP

from housekit import Mech, MechState, box

TOP = TABLE_TOP["station"]
MAT_X = 0.30                    # the mat's centre, on the station's x axis between the arms
MAT_T = 0.002
MAT_TOP = TOP + MAT_T
MAT_SIZE = (0.400, 0.600)       # station x (the brief's y, 400 mm), station y (the brief's x, 600 mm)
MAT_COLOR = (0.18, 0.19, 0.21)

TRAY_FLOOR = (0.100, 0.120)     # inside, station x by y (the brief's 120 mm across, 100 mm deep)
TRAY_WALL_H, TRAY_WALL_T, TRAY_FLOOR_T = 0.015, 0.002, 0.002
TRAY_MASS = 0.060               # estimate for a polypropylene tray this size; weigh the real one
TRAY_COLOR = (0.88, 0.88, 0.84)

STILL_S = 2.0                   # the observation after release
EDGE_TOL = 0.001                # footprint check against the tray's inner walls: 1 mm, the layout's measuring precision
CONTACT_MM = 2.0                # a finger-pad point (6 mm outside the pad) nearer than this to a part's surface: contact


def brief_xy(x_mm: float, y_mm: float) -> tuple[float, float]:
    """A brief's mat coordinates (mm) as station x, y (m)."""
    return MAT_X + y_mm / 1000.0, -x_mm / 1000.0


def mat() -> Mech:
    return Mech("mat", pos=(MAT_X, 0.0, TOP + MAT_T / 2), mass=None, friction=0.9,
                shapes=[box((0, 0, 0), (MAT_SIZE[0], MAT_SIZE[1], MAT_T), MAT_COLOR)])


def tray(name: str, xy) -> Mech:
    fx, fy = TRAY_FLOOR
    t, h, f = TRAY_WALL_T, TRAY_WALL_H, TRAY_FLOOR_T
    ox, oy = fx + 2 * t, fy + 2 * t
    shapes = [box((0, 0, f / 2), (ox, oy, f), TRAY_COLOR),
              box(((fx + t) / 2, 0, h / 2), (t, oy, h), TRAY_COLOR),
              box((-(fx + t) / 2, 0, h / 2), (t, oy, h), TRAY_COLOR),
              box((0, (fy + t) / 2, h / 2), (fx, t, h), TRAY_COLOR),
              box((0, -(fy + t) / 2, h / 2), (fx, t, h), TRAY_COLOR)]
    return Mech(name, pos=(xy[0], xy[1], MAT_TOP), mass=TRAY_MASS, friction=0.5, shapes=shapes)


# -- geometry of a Mech -----------------------------------------------------------------------------------------------

def _quat_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _axis_rot(axis: str, angle: float) -> np.ndarray:
    u = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return _quat_wxyz((math.cos(angle / 2), *(np.array(u) * math.sin(angle / 2))))


def pieces(spec: Mech, state: MechState):
    """Every shape of the Mech placed in the world: (kind, centre, rotation, size) with size as housekit gives it."""
    R0 = state.rotation()
    out = []
    for s in spec.shapes:
        out.append((s.kind, state.origin + R0 @ np.array(s.at), R0 @ _quat_wxyz(s.tilt), s.size))
    for j in spec.joints:
        v = state.joints.get(j.name, 0.0)
        at = np.array(j.at)
        if j.kind == "slide":
            Rj, oj = np.eye(3), at + np.array({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[j.axis]) * v
        else:
            Rj, oj = _axis_rot(j.axis, v), at
        for s in j.shapes:
            c = oj + Rj @ np.array(s.at)
            out.append((s.kind, state.origin + R0 @ c, R0 @ Rj @ _quat_wxyz(s.tilt), s.size))
    return out


def _point_shape(point, piece) -> float:
    """Signed distance from a point to one shape (negative inside)."""
    kind, c, R, size = piece
    q = R.T @ (np.asarray(point, float) - c)
    if kind == "box":
        h = np.array(size) / 2
        d = np.abs(q) - h
    else:
        r, _, ht = size
        d = np.array([math.hypot(q[0], q[1]) - r, abs(q[2]) - ht / 2])
    outside = np.linalg.norm(np.maximum(d, 0.0))
    return float(outside if outside > 0 else d.max())


def distance(spec: Mech, state: MechState, point) -> float:
    """Signed distance from a world point to the Mech's nearest shape (m)."""
    return min(_point_shape(point, pc) for pc in pieces(spec, state))


def corners(spec: Mech, state: MechState, joints_too: bool = True) -> np.ndarray:
    pts = []
    for kind, c, R, size in pieces(spec, state) if joints_too else pieces(spec, state)[: len(spec.shapes)]:
        if kind == "box":
            h = np.array(size) / 2
        else:
            h = np.array([size[0], size[0], size[2] / 2])
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + R @ (h * [sx, sy, sz]))
    return np.array(pts)


def surface_points(spec: Mech, state: MechState, per_edge: int = 3) -> np.ndarray:
    """Points spread over every shape's box (its corners, edge and face midpoints): for distances between parts."""
    pts = []
    ticks = np.linspace(-1, 1, per_edge)
    for kind, c, R, size in pieces(spec, state):
        h = np.array(size) / 2 if kind == "box" else np.array([size[0], size[0], size[2] / 2])
        for a in ticks:
            for b in ticks:
                for s in (-1, 1):
                    pts += [c + R @ (h * [s, a, b]), c + R @ (h * [a, s, b]), c + R @ (h * [a, b, s])]
    return np.array(pts)


def gap(spec_a: Mech, state_a: MechState, spec_b: Mech, state_b: MechState) -> float:
    """An estimate of the clearance between two Mechs (m): the least distance from points over A's shapes to B's.
    Exact for parallel faces; for a 10 mm clearance check the sampling error is well under a millimetre."""
    pts = surface_points(spec_a, state_a, 5)
    return min(distance(spec_b, state_b, q) for q in pts)


def touching(spec: Mech, state: MechState, fingers) -> bool:
    """A finger pad on the part: a pad point (6 mm outside its pad face) within CONTACT_MM of the part's surface."""
    return any(distance(spec, state, f) < CONTACT_MM / 1000.0 for f in fingers)


def in_tray(spec: Mech, state: MechState, tray_spec: Mech, tray_state: MechState) -> bool:
    """Resting on the tray floor with its whole footprint inside the floor's inner boundary (a part leaning on a wall's
    inner face counts as inside: EDGE_TOL)."""
    pts = corners(spec, state)
    R = tray_state.rotation()
    local = (pts - tray_state.origin) @ R
    fx, fy = TRAY_FLOOR
    inside = (np.abs(local[:, 0]) <= fx / 2 + EDGE_TOL).all() and (np.abs(local[:, 1]) <= fy / 2 + EDGE_TOL).all()
    low = local[:, 2].min()
    on_floor = TRAY_FLOOR_T - 0.003 <= low <= TRAY_FLOOR_T + 0.004
    return bool(inside and on_floor)


def still(state: MechState, lin=0.005, ang=0.05) -> bool:
    return float(np.linalg.norm(state.lin_vel)) < lin and float(np.linalg.norm(state.ang_vel)) < ang


def fallen(state: MechState) -> bool:
    """Off the work surface: its lowest point more than 1 cm below the table's top, or its middle past the table's
    edge."""
    c = state.pos
    return bool(state.low[2] < TOP - 0.01 or not (-0.19 < c[0] < 0.91) or abs(c[1]) > 0.65)


class Timer:
    """How long a condition has held, in control steps of 1/20 s."""

    def __init__(self, dt: float = 0.05) -> None:
        self.dt, self.n = dt, 0

    def update(self, ok: bool) -> float:
        self.n = self.n + 1 if ok else 0
        return self.n * self.dt


def state_of(spec: Mech, origin=None, yaw: float = 0.0, joints=None, quat_xyzw=None) -> MechState:
    """A MechState for a Mech placed by hand (for tests and counterexamples): its origin, a yaw or a full quaternion,
    and its joints' values (0 where not given)."""
    import housekit

    values = {j.name: 0.0 for j in spec.joints}
    values.update(joints or {})
    origin = np.array(spec.pos if origin is None else origin, dtype=float)
    quat = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)] if quat_xyzw is None else quat_xyzw, dtype=float)
    R = housekit._quat_xyzw_matrix(quat)
    pts = housekit._local_corners(spec, values) @ R.T + origin
    low, high = pts.min(0), pts.max(0)
    return MechState(pos=(low + high) / 2, quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=low,
                     high=high, joints=values, origin=origin)
