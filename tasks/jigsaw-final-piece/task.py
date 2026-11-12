"""On the station table, between the arms: a 4 x 4 jigsaw puzzle (17.6 cm square, a landscape printed on 6 mm
plywood pieces with round-headed tabs) on a dark base board, 15 pieces assembled and glued down. The piece in the
second row from the robot, second column from its right, is missing. Four loose pieces lie on the table, two on each
side of the board, each turned a different way, each with a small square peg glued at its centre to hold it by. One
is the missing piece; one has its shape but the wrong picture; one has the right picture but the wrong tabs; one has
neither.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
The pieces' shapes and pictures are in pieces.json, made by make_pieces.py.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

from puzzlekit import Piece  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Complete the jigsaw puzzle by choosing and inserting the correct missing piece."

DATA = json.loads((HERE / "pieces.json").read_text())
CELL = DATA["cell"]                      # 44 mm
THICK = 0.006                            # 6 mm plywood
HOLE_CELL = tuple(DATA["hole"])          # (row from the robot's side, column from the robot's right), 0-based
CORRECT = DATA["correct"]

# -- the board (world frame: x forward from the arms, y left, z up; metres) ----------------------------------------
BOARD_XY = (0.30, 0.0)
BASE_T = 0.004
BASE_SIZE = 0.20
BASE_TOP = TOP + BASE_T
PLYWOOD = (0.80, 0.66, 0.48)
BASE_COLOUR = (0.16, 0.16, 0.18)        # dark, so the hole's shape shows


def cell_xy(cell) -> tuple[float, float]:
    return (BOARD_XY[0] + (cell[0] - 1.5) * CELL, BOARD_XY[1] + (cell[1] - 1.5) * CELL)


HOLE_XY = cell_xy(HOLE_CELL)


def _tuples(points):
    return tuple(tuple(p) for p in points)


BEVEL_IN, BEVEL_UP = DATA["bevel"]       # a loose piece's bottom edge: bevelled 2 mm in over 2.5 mm up


def _piece(name, rec, pos, *, yaw=0.0, mass=None, knob=None):
    decals = tuple((tuple(d["color"]), tuple(_tuples(t) for t in d["triangles"])) for d in rec["decals"])
    xs = [p[0] for p in rec["outline"]]
    ys = [p[1] for p in rec["outline"]]
    size = (max(xs) - min(xs), max(ys) - min(ys), THICK + (knob[1] if knob else 0.0))
    common = dict(pos=pos, yaw=yaw, mass=mass, size=size, color=PLYWOOD, thick=THICK, decals=decals, knob=knob)
    if mass is None:  # glued down: one exact collider
        return Piece(name, outline=_tuples(rec["outline"]), triangles=tuple(_tuples(t) for t in rec["triangles"]),
                     **common)
    bottoms = tuple(None if b is None else _tuples(b) for b in rec["bottoms"])
    return Piece(name, parts=tuple(_tuples(p) for p in rec["parts"]), bottoms=bottoms, chamfer=BEVEL_UP, **common)


BASE = Box("board_base", pos=(*BOARD_XY, TOP + BASE_T / 2), size=(BASE_SIZE, BASE_SIZE, BASE_T), color=BASE_COLOUR)
PLACED = [_piece(name, rec, (*cell_xy(rec["cell"]), BASE_TOP)) for name, rec in DATA["placed"].items()]

# -- the four loose pieces ---------------------------------------------------------------------------------------------
KNOB = (0.012, 0.014)                    # a 12 mm square pine peg, 14 mm tall, glued at the piece's centre
PIECE_MASS = 0.008                       # 6 mm plywood and the peg
# where each lies on the table (x, y) and how far it is turned (degrees, counterclockwise seen from above) from the
# way it would sit in the hole
START = {
    "piece_a": ((0.25, 0.175), 200.0),
    "piece_b": ((0.35, 0.175), 70.0),
    "piece_c": ((0.35, -0.175), 125.0),
    "piece_d": ((0.25, -0.175), 320.0),
}
CANDIDATES = [_piece(name, DATA["candidates"][name], (*xy, TOP + 0.0005), yaw=math.radians(deg), mass=PIECE_MASS,
                     knob=KNOB) for name, (xy, deg) in START.items()]
OBJECTS = [BASE, *PLACED, *CANDIDATES]

# -- scoring --------------------------------------------------------------------------------------------------------
SEATED_DIST, SEATED_YAW_DEG, SEATED_HEIGHT, SEATED_TILT_DEG = 0.0025, 5.0, 0.0015, 3.0
OVER_DIST = 0.015
OVER_HEIGHT = THICK + 0.012              # the piece's underside at most this far above the base: lying in or on the
#                                          hole, not carried over it


def pose_in_hole(state):
    """A piece against the hole: (centre distance m, yaw error deg, underside above the base m, tilt deg)."""
    dist = math.dist(state.pos[:2], HOLE_XY)
    yaw_err = abs((math.degrees(state.yaw()) + 180.0) % 360.0 - 180.0)
    return dist, yaw_err, float(state.pos[2]) - BASE_TOP, state.tilt_deg()


def _over(state) -> bool:
    dist, _, height, _ = pose_in_hole(state)
    return dist <= OVER_DIST and height <= OVER_HEIGHT


def outcome(objects) -> tuple[float, str]:
    for name in START:
        if name != CORRECT and _over(objects[name]):
            return 0.0, f"{name}, a wrong piece, is in or on the hole"
    dist, yaw_err, height, tilt = pose_in_hole(objects[CORRECT])
    if (dist <= SEATED_DIST and yaw_err <= SEATED_YAW_DEG and abs(height) <= SEATED_HEIGHT
            and tilt <= SEATED_TILT_DEG):
        return 1.0, "the missing piece is seated"
    if _over(objects[CORRECT]):
        return 0.5, (f"the missing piece is over the hole, not seated (off by {1000 * dist:.1f} mm, turned "
                     f"{yaw_err:.0f} degrees, {1000 * height:.1f} mm up)")
    return 0.0, "the missing piece is not over the hole"


def success(objects) -> bool:
    """The missing piece seated in the hole, at rest; holds 1 s."""
    return outcome(objects)[0] == 1.0 and objects[CORRECT].resting()


_LAST = {"why": None}


def score(objects) -> float:
    """1.0 the missing piece seated: centre within 2.5 mm of the hole's, turned within 5 degrees of the way it fits,
    flat on the base (underside within 1.5 mm of it, tilted at most 3 degrees). 0.5 the missing piece over the hole
    but not seated: centre within 1.5 cm of the hole's, lying in or on it. 0 otherwise, and 0 whenever a wrong piece
    lies in or on the hole."""
    value, why = outcome(objects)
    if why.split(" (")[0] != _LAST["why"]:  # a line in the run's log each time the verdict changes
        _LAST["why"] = why.split(" (")[0]
        print(f"[task] {value}: {why}", flush=True)
    return value


# -- scripted demo --------------------------------------------------------------------------------------------------
PAD_TIP = 0.013                          # the fingertips, below the TCP
KNOB_GRIP = 0.002                        # fingertips this far above the piece's top face while holding the peg
PICK_Z = TOP + 0.0005 + THICK + KNOB_GRIP + PAD_TIP
DROP_Z = BASE_TOP + THICK + 0.0015 + THICK + KNOB_GRIP + PAD_TIP   # the piece's underside 1.5 mm above the board
PRESS_Z = BASE_TOP + THICK + KNOB[1] + PAD_TIP   # closed fingertips on the peg's top, the piece seated
TRAVEL_Z = TOP + 0.10
RELEASE = 0.45 * FINGER_OPEN
DEMO_ARM = "right"


def grip_yaws():
    """(at the pick, at the hole) in radians: the fingers square to the peg, turned with the piece (125 degrees at
    the pick, 0 at the hole: the wrist turns 125 degrees on the way, within its 120-degree-each-way range)."""
    _, deg = START[CORRECT]
    return math.radians(deg), 0.0


def demo():
    """The right arm takes the missing piece by its peg, lifts it, turns it the way it fits while carrying it over
    the hole, lowers it until it is just above the board and lets go (it drops into the hole, its bevel taking up a
    few millimetres of misplacement), then presses its peg down twice with closed fingers."""
    f0, f1 = FINGER_OPEN, FINGER_CLOSED
    (px, py), _ = START[CORRECT]
    hx, hy = HOLE_XY
    g0, g1 = grip_yaws()
    return {DEMO_ARM: [
        Waypoint(2.0, (px, py, TRAVEL_Z), f0, "above the piece", g0),
        Waypoint(1.5, (px, py, PICK_Z), f0, "down to the peg", g0),
        Waypoint(0.8, (px, py, PICK_Z), f1, "grip", g0),
        Waypoint(1.2, (px, py, TRAVEL_Z), f1, "lift", g0),
        Waypoint(3.0, (hx, hy, TRAVEL_Z), f1, "over the hole, turning", g1),
        Waypoint(1.5, (hx, hy, DROP_Z + 0.01), f1, "down", g1),
        Waypoint(1.0, (hx, hy, DROP_Z), f1, "just above the board", g1),
        Waypoint(0.5, (hx, hy, DROP_Z), f1, "steady", g1),
        Waypoint(0.5, (hx, hy, DROP_Z), RELEASE, "let go", g1),
        Waypoint(0.6, (hx, hy, PRESS_Z + 0.012), RELEASE, "up", g1),
        Waypoint(0.4, (hx, hy, PRESS_Z + 0.012), f1, "close", g1),
        Waypoint(1.0, (hx, hy, PRESS_Z - 0.002), f1, "press the peg", g1),
        Waypoint(0.5, (hx, hy, PRESS_Z + 0.004), f1, "ease off", g1),
        Waypoint(0.8, (hx, hy, PRESS_Z - 0.002), f1, "press again", g1),
        Waypoint(1.0, (hx, hy, TRAVEL_Z), f1, "up", g1),
    ]}
