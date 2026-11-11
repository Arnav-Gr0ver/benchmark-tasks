"""The tilt maze's geometry, scoring and route. Pure Python (math only): no Isaac Sim, no NumPy.

Everything here is in the board frame: origin at the centre of the maze floor, x away from the robot, y to the
robot's left, z up out of the floor; metres and radians. Azimuths are counterclockwise from +x seen from above, so
0 deg is the far side of the board, 90 deg its left, 180 deg the side nearest the robot.

The board is a disc with five concentric walls. Between them lie four ring-shaped channels and, in the middle, the
centre cell. Gaps in the walls join neighbouring channels and radial "spokes" block a channel. The same tables build
the simulated colliders (``wall_boxes``), the cutting drawing (``make_cad.py``) and the scorer (``score_at``).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

# -- dimensions (m) ----------------------------------------------------------------------------------------------
BOARD_R = 0.140          # disc radius
BASE_T = 0.006           # base plate thickness; the maze floor is its top face (z = 0)
WALL_H = 0.018           # wall height above the floor
WALL_T = 0.004           # wall thickness
LID_T = 0.003            # clear lid, lying on the walls
BALL_R = 0.008           # 16 mm ball
BALL_MASS = 0.0054       # glass marble
BOARD_MASS = 0.90        # base, walls, lid and ring weight together

# The board rides on a two-axis gimbal (a small universal joint) on a post under its centre. Both gimbal axes pass
# through the pivot, PIVOT_BELOW_FLOOR under the maze floor's centre; twisting about the vertical is impossible. A ring
# weight under the base puts the centre of mass COM_BELOW_PIVOT under the pivot, so a board left alone swings back
# level. The post stands inside a fixed stand tube (STAND_R) whose top rim the base's underside lands on at TILT_STOP,
# whichever way the board tilts.
PIVOT_BELOW_FLOOR = 0.016                    # the joint's centre below the floor (10 mm under the base)
COM_BELOW_PIVOT = 0.030
FLOOR_ABOVE_TABLE = 0.060                    # the floor above the table when the board is level
TILT_STOP = math.radians(7.0)
STAND_R = 0.050
PIVOT_Z = -PIVOT_BELOW_FLOOR                 # board frame
BOARD_COM_Z = PIVOT_Z - COM_BELOW_PIVOT      # board frame
LID_TOP_Z = WALL_H + LID_T                   # top of the lid above the floor
_UNDERSIDE = PIVOT_BELOW_FLOOR - BASE_T      # the base's underside above the pivot
STAND_TOP_Z = PIVOT_Z + _UNDERSIDE * math.cos(TILT_STOP) - STAND_R * math.sin(TILT_STOP)  # board frame, board level

# Wall centre radii, innermost first. WALLS[0] encloses the centre cell; WALLS[4] is the closed outer wall.
WALLS = (0.028, 0.054, 0.080, 0.106, 0.132)
GAP_W = 0.028            # clear opening of a gap, measured along the wall's centre line
RINGS = len(WALLS) - 1   # channels; ring k lies between WALLS[k-1] and WALLS[k]; "ring 0" is the centre cell

# Gaps: wall index -> [(start azimuth deg, end azimuth deg)], counterclockwise from start to end.
# Spokes: ring index -> [azimuth deg]; a spoke crosses its channel from wall to wall.
# Two routes lead in. The long one: from the start (ring 4 at 180 deg) clockwise to the gap at 60-75 deg; in ring 3
# counterclockwise to the gap at 110-130 deg; in ring 2 clockwise, the long way round, to the gap at 300-320 deg; in
# ring 1 counterclockwise to the centre gap at 80-129 deg. The short one goes backward first: from ring 3 back out
# through the gap at 230-245 deg into the other half of ring 4, round to about 300 deg, where the gaps in walls 3, 2
# and 1 line up, and straight in to ring 1. Every other gap leads into a dead end. (Layout picked by a search that
# kept layouts a random tilter almost never gets out of the outer ring of: see README.)


def _gap(start_deg: float, wall: int, ccw: bool = True, width: float = GAP_W) -> tuple[float, float]:
    """A gap of clear width ``width`` in wall ``wall`` starting at ``start_deg`` and opening counterclockwise
    (``ccw``) or clockwise from there."""
    span = math.degrees(width / WALLS[wall])
    return (start_deg, start_deg + span) if ccw else (start_deg - span, start_deg)


GAPS = {
    0: [_gap(80.0, 0, width=0.024)],                      # into the centre cell
    1: [_gap(200.0, 1), _gap(300.0, 1)],
    2: [_gap(40.0, 2), _gap(110.0, 2), _gap(300.0, 2)],
    3: [_gap(60.0, 3), _gap(230.0, 3), _gap(290.0, 3)],
    4: [],
}
SPOKES = {
    1: [180.0, 240.0],
    2: [240.0, 290.0],
    3: [60.0, 270.0],
    4: [20.0, 210.0],
}
START_RING, START_DEG = 4, 180.0
START_R = (WALLS[3] + WALLS[4]) / 2
START_XY = (START_R * math.cos(math.radians(START_DEG)), START_R * math.sin(math.radians(START_DEG)))


# -- geometry ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WallBox:
    """An upright box in the board frame: centre (x, y), half sizes along its own x and y, and its yaw."""

    x: float
    y: float
    hx: float
    hy: float
    yaw: float
    kind: str  # "arc" or "spoke"


def _norm_deg(a: float) -> float:
    return a % 360.0


def solid_spans(wall: int) -> list[tuple[float, float]]:
    """The wall's solid stretches as (start, end) azimuths in degrees, counterclockwise, end > start."""
    gaps = sorted((_norm_deg(a), _norm_deg(a) + ((b - a) % 360.0)) for a, b in GAPS[wall])
    if not gaps:
        return [(0.0, 360.0)]
    spans = []
    for i, (_, end) in enumerate(gaps):
        nxt = gaps[(i + 1) % len(gaps)][0] + (360.0 if i + 1 == len(gaps) else 0.0)
        if nxt - end > 1e-9:
            spans.append((end, nxt))
    return spans


def wall_boxes(max_sag: float = 0.0002) -> list[WallBox]:
    """Every wall as boxes: each solid stretch of an arc as short chords (bowing at most ``max_sag`` off the true
    arc), and each spoke as one box."""
    boxes = []
    for k, r in enumerate(WALLS):
        step = math.degrees(2 * math.acos(1 - max_sag / r))
        for a, b in solid_spans(k):
            n = max(1, math.ceil((b - a) / step))
            d = math.radians((b - a) / n)
            for i in range(n):
                mid = math.radians(a) + (i + 0.5) * d
                # The chord's outer face spans the arc piece, so neighbours meet without a notch.
                half_len = (r + WALL_T / 2) * math.tan(d / 2)
                rc = r * math.cos(d / 2) if n > 1 else r
                boxes.append(WallBox(rc * math.cos(mid), rc * math.sin(mid), WALL_T / 2, half_len, mid, "arc"))
    for ring, spokes in SPOKES.items():
        r0, r1 = WALLS[ring - 1], WALLS[ring]
        for deg in spokes:
            a = math.radians(deg)
            rc = (r0 + r1) / 2
            boxes.append(WallBox(rc * math.cos(a), rc * math.sin(a), (r1 - r0) / 2, WALL_T / 2, a, "spoke"))
    return boxes


# -- scoring -------------------------------------------------------------------------------------------------------

def ring_at(x: float, y: float) -> int:
    """Which channel the ball's centre is in: 0 the centre cell, 1 the innermost ring ... 4 the outermost. The
    boundaries are the walls' centre circles, so a ball sitting in a gap counts for the side its centre is on."""
    r = math.hypot(x, y)
    for k, wall in enumerate(WALLS):
        if r < wall:
            return k
    return RINGS


def score_at(x: float, y: float) -> float:
    """0 in the outermost ring (where the ball starts), then 0.25 more per ring inward; 1.0 in the centre cell."""
    return (RINGS - ring_at(x, y)) / RINGS


# -- route ---------------------------------------------------------------------------------------------------------

def _blocked(ring: int, a: float, b: float) -> bool:
    """Whether a spoke of ``ring`` lies between azimuths a and b (deg, |b - a| < 180)."""
    lo, hi = min(a, b), max(a, b)
    return any(lo <= s + turn <= hi for s in SPOKES.get(ring, ()) for turn in (-360.0, 0.0, 360.0))


def _open(wall: int, deg: float, margin: float) -> bool:
    """Whether wall ``wall`` is open at azimuth ``deg``, at least ``margin`` metres from the gap's ends."""
    m = math.degrees(margin / WALLS[wall])
    for a, b in GAPS[wall]:
        span = (b - a) % 360.0
        off = (deg - a) % 360.0
        if m <= off <= span - m:
            return True
    return False


def route(start_xy: tuple[float, float] | None = None, step_deg: float = 2.0) -> list[tuple[float, float]]:
    """The shortest path for the ball's centre to the centre cell, as (x, y) points: along the channels' centre
    lines and straight through the gaps (Dijkstra over points every ``step_deg``). It starts at the ball's start
    mark, or with ``start_xy`` at the centre-line point of that position's ring nearest to it."""
    n = round(360.0 / step_deg)
    sx, sy = START_XY if start_xy is None else start_xy
    start_ring = ring_at(sx, sy)
    if start_ring == 0:
        return [(sx, sy), (0.0, 0.0)]

    def mid_r(ring: int) -> float:
        return (WALLS[ring - 1] + WALLS[ring]) / 2

    def xy(node) -> tuple[float, float]:
        ring, i = node
        if ring == 0:
            return (0.0, 0.0)
        a = math.radians(i * step_deg)
        return (mid_r(ring) * math.cos(a), mid_r(ring) * math.sin(a))

    def neighbours(node):
        ring, i = node
        if ring == 0:
            return
        deg = i * step_deg
        for j in (i - 1, i + 1):
            if not _blocked(ring, deg, j * step_deg):
                yield (ring, j % n), mid_r(ring) * math.radians(step_deg)
        # Through a gap, keeping the ball's edge a few millimetres off the gap's ends.
        if _open(ring - 1, deg, BALL_R + 0.003):
            inner = (ring - 1, i) if ring > 1 else (0, 0)
            yield inner, mid_r(ring) - (mid_r(ring - 1) if ring > 1 else 0.0)
        if ring < RINGS and _open(ring, deg, BALL_R + 0.003):
            yield (ring + 1, i), mid_r(ring + 1) - mid_r(ring)

    start = (start_ring, round(math.degrees(math.atan2(sy, sx)) / step_deg) % n)
    best, prev, heap = {start: 0.0}, {}, [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == (0, 0):
            break
        if d > best[node]:
            continue
        for nxt, cost in neighbours(node):
            if d + cost < best.get(nxt, math.inf):
                best[nxt], prev[nxt] = d + cost, node
                heapq.heappush(heap, (d + cost, nxt))
    if (0, 0) not in prev:
        raise ValueError("the maze has no route from the start to the centre")
    path, node = [], (0, 0)
    while node != start:
        path.append(xy(node))
        node = prev[node]
    path.append(xy(start))
    return path[::-1]
