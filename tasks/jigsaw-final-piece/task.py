"""On the station table, between the arms: a wooden peg puzzle in a tray, 3 x 3 square pieces (4.9 cm, each with a
small square knob on top), joined by rectangular tabs that fit into notches in their neighbours. The middle piece is
missing; it lies on the table to the robot's left, turned a quarter turn from the way it fits. Its tabs and notches
fit the hole only one way round.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    --policy yam_scripted
"""

from __future__ import annotations

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

from shapekit import Compound  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Finish the jigsaw puzzle: put the last piece into the empty space in the tray."

# -- pieces (m). Directions are the tray's (world) axes: +x away from the robot, +y to its left -----------------------
CELL = 0.052                 # piece pitch in the tray
HALF = 0.0245                # half the body: 1.5 mm clear of each neighbour and of the tray's walls
THICK = 0.008
TAB_W, TAB_D = 0.014, 0.011   # a tab reaches 8 mm into its neighbour's notch, past the 3 mm gap between bodies
NOTCH_W, NOTCH_D = 0.017, 0.0095  # 1.5 mm clear of the tab on every side
KNOB = (0.012, 0.012, 0.014)
DIRS = {"+x": (1, 0), "+y": (0, 1), "-x": (-1, 0), "-y": (0, -1)}


def piece_boxes(edges: dict[str, str]) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """A piece as boxes (centre, size; frame origin at the middle of its underside): the body minus a notch on each
    edge marked "N", plus a tab on each edge marked "T" ("F": flat), and the knob."""
    notches = []
    for d, kind in edges.items():
        if kind == "N":
            ux, uy = DIRS[d]
            if ux:
                x0, x1 = sorted((ux * HALF, ux * (HALF - NOTCH_D)))
                notches.append((x0, x1, -NOTCH_W / 2, NOTCH_W / 2))
            else:
                y0, y1 = sorted((uy * HALF, uy * (HALF - NOTCH_D)))
                notches.append((-NOTCH_W / 2, NOTCH_W / 2, y0, y1))
    xs = sorted({-HALF, HALF, *(v for n in notches for v in n[:2])})
    ys = sorted({-HALF, HALF, *(v for n in notches for v in n[2:])})
    boxes = []
    for x0, x1 in zip(xs, xs[1:]):
        run = None
        for y0, y1 in zip(ys, ys[1:]):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            solid = not any(a < mx < b and c < my < e for a, b, c, e in notches)
            if solid:
                run = (run[0], y1) if run else (y0, y1)
            if (not solid or y1 == ys[-1]) and run:
                boxes.append((((x0 + x1) / 2, (run[0] + run[1]) / 2, THICK / 2), (x1 - x0, run[1] - run[0], THICK)))
                run = None
    for d, kind in edges.items():
        if kind == "T":
            ux, uy = DIRS[d]
            c = HALF + TAB_D / 2
            size = (TAB_D, TAB_W, THICK) if ux else (TAB_W, TAB_D, THICK)
            boxes.append(((ux * c, uy * c, THICK / 2), size))
    boxes.append(((0.0, 0.0, THICK + KNOB[2] / 2), KNOB))
    return boxes


def edges_of(cell, owners) -> dict[str, str]:
    """A cell's four edges: "F" on the tray's border, else "T" if this cell owns the shared edge's tab, else "N"."""
    out = {}
    for d, (ux, uy) in DIRS.items():
        other = (cell[0] + ux, cell[1] + uy)
        if max(abs(other[0]), abs(other[1])) > 1:
            out[d] = "F"
        else:
            out[d] = "T" if owners[frozenset((cell, other))] == cell else "N"
    return out


def _owners():
    """Who has the tab on each of the twelve inner edges: the middle piece's four are fixed (tabs away from and to
    the right of the robot, notches toward it and to its left); the rest by a fixed draw."""
    rng = random.Random(7)
    cells = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    owners = {}
    for a in cells:
        for ux, uy in ((1, 0), (0, 1)):
            b = (a[0] + ux, a[1] + uy)
            if b in cells:
                owners[frozenset((a, b))] = rng.choice((a, b))
    middle = (0, 0)
    for d, tab in (("+x", True), ("-y", True), ("-x", False), ("+y", False)):
        ux, uy = DIRS[d]
        other = (ux, uy)
        owners[frozenset((middle, other))] = middle if tab else other
    return owners


OWNERS = _owners()
MIDDLE_EDGES = edges_of((0, 0), OWNERS)

# -- tray and layout (world frame) ----------------------------------------------------------------------------------
TRAY_XY = (0.27, 0.0)
FLOOR_T = 0.004
FLOOR = TOP + FLOOR_T
WALL_T, WALL_H = 0.008, 0.008
INNER = 1.5 * CELL           # the walls' inner faces, from the tray's centre
WOOD = (0.78, 0.62, 0.42)
COLOURS = [(0.35, 0.60, 0.85), (0.55, 0.75, 0.35), (0.90, 0.75, 0.25), (0.75, 0.45, 0.70), (0.40, 0.75, 0.75),
           (0.85, 0.50, 0.40), (0.55, 0.55, 0.85), (0.70, 0.80, 0.45)]
FINAL_COLOUR = (0.95, 0.45, 0.10)


def _tray():
    cx, cy = TRAY_XY
    outer = 2 * (INNER + WALL_T)
    parts = [Box("tray_floor", pos=(cx, cy, TOP + FLOOR_T / 2), size=(outer, outer, FLOOR_T), color=WOOD)]
    for name, (ux, uy) in DIRS.items():
        c = INNER + WALL_T / 2
        size = (WALL_T, outer, WALL_H) if ux else (outer, WALL_T, WALL_H)
        parts.append(Box(f"tray_wall{name[0] == '+' and 'p' or 'm'}{name[1]}",
                         pos=(cx + ux * c, cy + uy * c, FLOOR + WALL_H / 2), size=size, color=WOOD))
    return parts


def cell_xy(cell) -> tuple[float, float]:
    return (TRAY_XY[0] + cell[0] * CELL, TRAY_XY[1] + cell[1] * CELL)


def _nominal(edges):
    reach = [HALF + (TAB_D if edges[d] == "T" else 0.0) for d in ("+x", "+y", "-x", "-y")]
    return (reach[0] + reach[2], reach[1] + reach[3], THICK + KNOB[2])


PLACED = []
for k, cell in enumerate([c for c in ((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)) if c != (0, 0)]):
    e = edges_of(cell, OWNERS)
    PLACED.append(Compound(f"piece_{cell[0] + 1}{cell[1] + 1}", pos=(*cell_xy(cell), FLOOR), size=_nominal(e),
                           color=COLOURS[k], boxes=tuple(piece_boxes(e))))

START_XY, START_YAW = (0.20, 0.20), math.pi / 2   # a quarter turn counterclockwise from the way it fits
PIECE_MASS = 0.012                                # 8 mm plywood with a pine knob
FINAL = Compound("last_piece", pos=(*START_XY, TOP + 0.0005), yaw=START_YAW, size=_nominal(MIDDLE_EDGES),
                 mass=PIECE_MASS, color=FINAL_COLOUR, boxes=tuple(piece_boxes(MIDDLE_EDGES)), friction=0.6)
OBJECTS = [*_tray(), *PLACED, FINAL]

# -- scoring --------------------------------------------------------------------------------------------------------
SEATED_DIST, SEATED_YAW_DEG, SEATED_HEIGHT, SEATED_TILT_DEG = 0.003, 6.0, 0.0015, 3.0
OVER_DIST, TURNED_YAW_DEG = 0.015, 15.0


def pose_in_hole(objects):
    """The last piece against the hole: (centre distance m, yaw error deg, underside above the floor m, tilt deg)."""
    p = objects["last_piece"]
    dist = math.dist(p.pos[:2], cell_xy((0, 0)))
    yaw_err = abs((math.degrees(p.yaw()) + 180.0) % 360.0 - 180.0)
    return dist, yaw_err, float(p.pos[2]) - FLOOR, p.tilt_deg()


def outcome(objects) -> tuple[float, str]:
    dist, yaw_err, height, tilt = pose_in_hole(objects)
    if (dist <= SEATED_DIST and yaw_err <= SEATED_YAW_DEG and abs(height) <= SEATED_HEIGHT
            and tilt <= SEATED_TILT_DEG):
        return 1.0, "seated"
    if dist <= OVER_DIST and yaw_err <= TURNED_YAW_DEG:
        return 0.5, "turned the right way over the hole, not seated"
    if dist <= OVER_DIST:
        return 0.25, f"over the hole, turned {yaw_err:.0f} degrees from the way it fits"
    return 0.0, "not over the hole"


def success(objects) -> bool:
    """The last piece seated in the hole the right way round, at rest; holds 1 s."""
    return outcome(objects)[0] == 1.0 and objects["last_piece"].resting()


def score(objects) -> float:
    """1.0 seated (centre within 3 mm, turned within 6 degrees, flat on the tray floor); 0.5 turned the right way
    (within 15 degrees) with its centre within 1.5 cm of the hole's but not seated; 0.25 over the hole turned the
    wrong way; 0 otherwise."""
    return outcome(objects)[0]


# -- scripted demo --------------------------------------------------------------------------------------------------
PAD_TIP = 0.013
KNOB_GRIP = 0.002                                      # fingertips this far above the piece's top face
PICK_Z = TOP + THICK + KNOB_GRIP + PAD_TIP
PLACE_Z = FLOOR + 0.0015 + THICK + KNOB_GRIP + PAD_TIP  # the underside 1.5 mm above the floor when let go
TRAVEL_Z = TOP + 0.12
RELEASE = 0.45 * FINGER_OPEN                           # open only partly: the neighbours' knobs are 4.6 cm away


def demo():
    """The left arm grips the knob (fingers closing along y), lifts, turns the piece a quarter turn clockwise on the
    way, lowers it into the hole, lets go and lifts away."""
    f0, f1 = FINGER_OPEN, FINGER_CLOSED
    (px, py), (hx, hy) = START_XY, cell_xy((0, 0))
    turn = -(START_YAW % (2 * math.pi))                 # the grip turns with the piece: from 0 to -90 degrees
    return {"left": [
        Waypoint(1.5, (px, py, TRAVEL_Z), f0, "above the piece"),
        Waypoint(1.5, (px, py, PICK_Z), f0, "down to the knob"),
        Waypoint(0.8, (px, py, PICK_Z), f1, "grip"),
        Waypoint(1.2, (px, py, TRAVEL_Z), f1, "lift"),
        Waypoint(2.5, (hx, hy, TRAVEL_Z), f1, "over the hole, turning", turn),
        Waypoint(1.5, (hx, hy, PLACE_Z + 0.01), f1, "down", turn),
        Waypoint(1.0, (hx, hy, PLACE_Z), f1, "into the hole", turn),
        Waypoint(0.6, (hx, hy, PLACE_Z), RELEASE, "let go", turn),
        Waypoint(1.0, (hx, hy, TRAVEL_Z), RELEASE, "up", turn),
    ]}
