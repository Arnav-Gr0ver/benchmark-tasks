"""On the station table, between the arms and 25 cm in front of them: an upright, standard-size Connect Four frame
(7 columns, 6 rows, 27 cm wide, 21 cm tall) facing the robot, holding twelve discs, six red and six blue. Columns
are numbered 1 to 7 from the robot's left. Bottom row, left to right: blue, blue, empty, blue, red, red, red. Second
row: column 4 red, column 5 blue, column 6 red, column 7 blue. Column 4 continues with blue in the third row and red
in the fourth. A spare red disc stands on its edge in a holder on the table to the left of the frame, and a spare
blue disc in one to the right. Red to move.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yam_arena.demo as _yam_demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

from shapekit import Compound, Disc  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("You are playing red in Connect Four and it is your turn. Drop the spare red disc into the column "
               "that wins the game.")

# -- frame (world frame: x forward from the arms, y left, z up; metres) ----------------------------------------------
COLS, ROWS = 7, 6
CX = 0.25                      # the slot's mid-plane
D, R, T = 0.032, 0.016, 0.007  # disc diameter, radius, thickness
PITCH = 0.038                  # column pitch: 36 mm inside each column, 2 mm dividers
GAP = 0.011                    # between the front and back plates (discs are 7 mm thick)
DIVIDER = 0.002
POST = 0.010                   # the front's posts at the column boundaries: 28 mm windows show the discs
PLATE = 0.003
BASE_H = 0.012
BASE_TOP = TOP + BASE_H
FRAME_H = ROWS * D + 0.002
FRAME_TOP = BASE_TOP + FRAME_H
WIDTH = COLS * PITCH
YELLOW, RED, BLUE = (0.95, 0.78, 0.10), (0.85, 0.10, 0.10), (0.12, 0.30, 0.85)


def col_y(col: int) -> float:
    """Column 0 is the robot's left (+y)."""
    return (3 - col) * PITCH


def cell_z(row: int) -> float:
    """Row 0 is the bottom; discs rest on each other."""
    return BASE_TOP + R + row * D


def _frame():
    zc = BASE_TOP + FRAME_H / 2
    parts = [
        Box("frame_base", pos=(CX, 0.0, TOP + BASE_H / 2), size=(0.06, WIDTH + POST + 0.02, BASE_H), color=YELLOW),
        Box("frame_back", pos=(CX + GAP / 2 + PLATE / 2, 0.0, zc), size=(PLATE, WIDTH + POST, FRAME_H), color=YELLOW),
        Box("frame_bottom_rail", pos=(CX - GAP / 2 - PLATE / 2, 0.0, BASE_TOP + 0.003),
            size=(PLATE, WIDTH + POST, 0.006), color=YELLOW),
    ]
    for i in range(COLS + 1):
        y = (3.5 - i) * PITCH
        parts.append(Box(f"frame_post{i}", pos=(CX - GAP / 2 - PLATE / 2, y, zc), size=(PLATE, POST, FRAME_H),
                         color=YELLOW))
        parts.append(Box(f"frame_divider{i}", pos=(CX, y, zc), size=(GAP, DIVIDER, FRAME_H), color=YELLOW))
    parts.append(_funnel())
    return parts


# The slot's flared mouth, as on a real frame: a lip along the top of the front and of the back, each sloping out and
# up at 45 degrees, so a disc let go a few millimetres off the slot's mid-plane slides in instead of landing on the rim.
LIP_LEN, LIP_T, LIP_DEG = 0.012, 0.002, 45.0
LIP_RISE = LIP_LEN * math.sin(math.radians(LIP_DEG))
MOUTH = GAP + 2 * LIP_LEN * math.cos(math.radians(LIP_DEG))   # the funnel's width across its top


def _funnel():
    run, rise = LIP_LEN * math.cos(math.radians(LIP_DEG)) / 2, LIP_RISE / 2
    lips = []
    for s in (-1, 1):  # -1: the front lip (toward the robot), +1: the back lip
        centre = (s * (GAP / 2 + run), 0.0, rise)
        # turning about y by +45 degrees lowers a box's +x end: the front lip's inner (+x) end sits on the slot's edge
        lips.append((centre, (LIP_LEN, WIDTH + POST, LIP_T), (0.0, -s * LIP_DEG)))
    # WHY slick: a disc that lands on a 45 degree lip must slide down it, and rubber-like friction (1.2) would hold it
    return Compound("frame_funnel", pos=(CX, 0.0, FRAME_TOP), size=(MOUTH, WIDTH + POST, LIP_RISE), color=YELLOW,
                    boxes=tuple(lips), friction=0.05)


# The position (column, row), 0-based. Red to move; twelve discs, six each.
PLACED_RED = [(4, 0), (5, 0), (6, 0), (3, 1), (5, 1), (3, 3)]
PLACED_BLUE = [(0, 0), (1, 0), (3, 0), (4, 1), (6, 1), (3, 2)]
WIN = (4, 2)       # completes red's diagonal (6,0), (5,1), (4,2), (3,3): column 5 from the left
BLOCK = (2, 0)     # completes blue's bottom row: column 3 from the left


RIDGES = 16                    # a real disc's ridged rim; the disc stands in its holder on one of the flats


def _placed(name, cell, color):
    return Disc(name, pos=(CX, col_y(cell[0]), cell_z(cell[1])), radius=R, height=T, axis="X", color=color,
                ridges=RIDGES)


DISCS = [_placed(f"red{i}", c, RED) for i, c in enumerate(PLACED_RED)]
DISCS += [_placed(f"blue{i}", c, BLUE) for i, c in enumerate(PLACED_BLUE)]

HOLDER_X = 0.20
HOLDER_Y = {"red": 0.24, "blue": -0.24}
HOLDER_WALL = (0.004, 0.020, 0.008)  # each holder: two low walls 1 mm off the disc's faces


def _holder(colour: str):
    y = HOLDER_Y[colour]
    off = T / 2 + 0.001 + HOLDER_WALL[0] / 2
    return [Box(f"holder_{colour}{i}", pos=(HOLDER_X + s * off, y, TOP + HOLDER_WALL[2] / 2), size=HOLDER_WALL,
                color=(0.3, 0.3, 0.32)) for i, s in enumerate((-1, 1))]


DISC_MASS = 0.006
SPARE_RED = Disc("red_spare", pos=(HOLDER_X, HOLDER_Y["red"], TOP + R + 0.0005), radius=R, height=T, axis="X",
                 mass=DISC_MASS, color=RED, ridges=RIDGES)
SPARE_BLUE = Disc("blue_spare", pos=(HOLDER_X, HOLDER_Y["blue"], TOP + R + 0.0005), radius=R, height=T, axis="X",
                  mass=DISC_MASS, color=BLUE, ridges=RIDGES)
OBJECTS = [*_frame(), *DISCS, *_holder("red"), *_holder("blue"), SPARE_RED, SPARE_BLUE]


# -- scoring --------------------------------------------------------------------------------------------------------
def slot_of(obj):
    """(column, row) of a disc standing in the frame's slot, or None."""
    x, y, z = (float(v) for v in obj.pos)
    if abs(x - CX) > GAP / 2 or abs(y) > WIDTH / 2 or not BASE_TOP < z < FRAME_TOP + 0.005:
        return None
    return round(3 - y / PITCH), round((z - BASE_TOP - R) / D)


def outcome(objects) -> tuple[float, str]:
    if slot_of(objects["blue_spare"]) is not None:
        return 0.0, "the blue disc was played"
    cell = slot_of(objects["red_spare"])
    if cell is None:
        return 0.0, "the red disc is not in the frame"
    if cell == WIN:
        return 1.0, "winning move"
    return 0.25, f"a legal move that does not win (column {cell[0] + 1})"


def success(objects) -> bool:
    """The spare red disc resting in the winning cell (column 5, third row); holds 1 s."""
    return outcome(objects)[0] == 1.0 and objects["red_spare"].resting()


def score(objects) -> float:
    """1.0 for the winning column; 0.25 for the red disc in any other column; 0 if the blue disc was played or the
    red disc is not in the frame."""
    return outcome(objects)[0]


# -- scripted demo --------------------------------------------------------------------------------------------------
# A hand holds a Connect Four disc by its two flat faces and reaches the frame's top leaning over, not pointing
# straight down (a straight-down grip cannot reach 23 cm up). yam_arena's scripted waypoints only turn the
# straight-down grip about the vertical (their `yaw`), so the demo tags its own grip by passing SIDE_GRIP as the yaw,
# and this file teaches yam_arena.demo.grip_rotation to read that tag. Ordinary yaws (within a turn of 0) are unchanged.
SIDE_GRIP = 100.0
LEAN = math.radians(40.0)      # the gripper leans 40 degrees from straight down, toward the robot's right

_plain_grip_rotation = getattr(_yam_demo.grip_rotation, "plain", _yam_demo.grip_rotation)


def _grip_rotation(yaw: float = 0.0) -> np.ndarray:
    """SIDE_GRIP: the fingers close along x (front to back, onto the disc's faces) and the approach leans LEAN from
    straight down toward -y."""
    if yaw < SIDE_GRIP / 2:
        return _plain_grip_rotation(yaw)
    c, s = math.cos(-LEAN), math.sin(-LEAN)
    lean = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return lean @ _plain_grip_rotation(-math.pi / 2)


_grip_rotation.plain = _plain_grip_rotation
_yam_demo.grip_rotation = _grip_rotation

HOLD_ABOVE_CENTRE = 0.008                       # the pads hold the disc's faces from 2 mm below its centre to its top
PICK_Z = SPARE_RED.pos[2] + HOLD_ABOVE_CENTRE   # the fingertips stay 6 mm above the holder's walls
DROP_Z = FRAME_TOP + 0.004 + R + HOLD_ABOVE_CENTRE  # the disc's bottom 4 mm above the slot, inside the funnel's mouth
#                                                 (19 mm wide there); the fingertips stay 9 mm above the funnel's lips
TRAVEL_Z = DROP_Z + 0.025                       # carried, the disc's bottom passes 18 mm above the funnel's lips
LIFT_Z = TOP + 0.12                             # WHY: leaning, the arm cannot reach TRAVEL_Z over the holder (it tops
VIA_Y = 0.17                                    # out 21 cm up there), so it climbs on its way in, to just left of the frame
RELEASE = 0.35 * FINGER_OPEN                    # open 16 mm a side: enough to let go, clear of everything


def demo():
    """The left arm, leaning over, takes the spare red disc by its two flat faces, lifts it out of its holder,
    carries it over column 5, lowers it into the slot's flared mouth and lets go."""
    f0, f1, g = FINGER_OPEN, FINGER_CLOSED, SIDE_GRIP
    px, py = SPARE_RED.pos[:2]
    wx, wy = CX, col_y(WIN[0])
    return {"left": [
        Waypoint(2.0, (px, py, PICK_Z + 0.06), f0, "above the disc", g),
        Waypoint(1.5, (px, py, PICK_Z), f0, "down", g),
        Waypoint(0.8, (px, py, PICK_Z), f1, "hold its faces", g),
        Waypoint(1.5, (px, py, LIFT_Z), f1, "lift", g),
        Waypoint(2.0, (wx, VIA_Y, TRAVEL_Z), f1, "up beside the frame", g),
        Waypoint(2.5, (wx, wy, TRAVEL_Z), f1, "over column 5", g),
        Waypoint(1.5, (wx, wy, DROP_Z), f1, "down into the slot's mouth", g),
        Waypoint(0.5, (wx, wy, DROP_Z), f1, "steady", g),
        Waypoint(0.6, (wx, wy, DROP_Z), RELEASE, "let go", g),
        Waypoint(1.0, (wx, wy, TRAVEL_Z), RELEASE, "up", g),
    ]}
