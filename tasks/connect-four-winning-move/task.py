"""On the station table, between the arms: an upright Connect Four frame (7 columns, 6 rows, 16 cm wide, 12 cm tall)
facing the robot, holding twelve discs, six red and six blue. Columns are numbered 1 to 7 from the robot's left.
Bottom row, left to right: blue, blue, empty, blue, red, red, red. Second row: column 4 red, column 5 blue, column 6
red, column 7 blue. Column 4 continues with blue in the third row and red in the fourth. A spare red disc stands in a
holder on the table to the left of the frame and a spare blue disc in one to the right. Red to move.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    --policy yam_scripted
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

from shapekit import Disc  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("You are playing red in Connect Four and it is your turn. Drop the spare red disc into the column "
               "that wins the game.")

# -- frame (world frame: x forward from the arms, y left, z up; metres) ----------------------------------------------
COLS, ROWS = 7, 6
CX = 0.18                     # the slot's mid-plane
D, R, T = 0.018, 0.009, 0.005  # disc diameter, radius, thickness
PITCH = 0.022                 # column pitch: 20 mm inside each column, 2 mm dividers
GAP = 0.010                   # between the front and back plates (discs are 5 mm thick)
DIVIDER = PITCH - 0.020
POST = 0.006                  # the front's posts at the column boundaries: 16 mm windows show the discs
PLATE = 0.003
BASE_H = 0.008
BASE_TOP = TOP + BASE_H
FRAME_H = ROWS * D + 0.002
FRAME_TOP = BASE_TOP + FRAME_H
WIDTH = COLS * PITCH
YELLOW, RED, BLUE = (0.95, 0.80, 0.10), (0.85, 0.10, 0.10), (0.12, 0.30, 0.85)


def col_y(col: int) -> float:
    """Column 0 is the robot's left (+y)."""
    return (3 - col) * PITCH


def cell_z(row: int) -> float:
    """Row 0 is the bottom; discs rest on each other."""
    return BASE_TOP + R + row * D


def _frame():
    zc = BASE_TOP + FRAME_H / 2
    parts = [
        Box("frame_base", pos=(CX, 0.0, TOP + BASE_H / 2), size=(GAP + 2 * PLATE + 0.03, WIDTH + POST + 0.02, BASE_H),
            color=YELLOW),
        Box("frame_back", pos=(CX + GAP / 2 + PLATE / 2, 0.0, zc), size=(PLATE, WIDTH + POST, FRAME_H), color=YELLOW),
    ]
    for i in range(COLS + 1):
        y = (3.5 - i) * PITCH
        parts.append(Box(f"frame_post{i}", pos=(CX - GAP / 2 - PLATE / 2, y, zc), size=(PLATE, POST, FRAME_H),
                         color=YELLOW))
        parts.append(Box(f"frame_divider{i}", pos=(CX, y, zc), size=(GAP, DIVIDER, FRAME_H), color=YELLOW))
    parts.append(Box("frame_bottom_rail", pos=(CX - GAP / 2 - PLATE / 2, 0.0, BASE_TOP + 0.002),
                     size=(PLATE, WIDTH + POST, 0.004), color=YELLOW))
    return parts


# The position (column, row): 0-based. Red to move; twelve discs, six each.
PLACED_RED = [(4, 0), (5, 0), (6, 0), (3, 1), (5, 1), (3, 3)]
PLACED_BLUE = [(0, 0), (1, 0), (3, 0), (4, 1), (6, 1), (3, 2)]
WIN = (4, 2)       # completes red's diagonal (6,0), (5,1), (4,2), (3,3): column 5 from the left
BLOCK = (2, 0)     # completes blue's bottom row: column 3 from the left


def _placed(name, cell, color):
    return Disc(name, pos=(CX, col_y(cell[0]), cell_z(cell[1])), radius=R, height=T, axis="X", color=color)


DISCS = [_placed(f"red{i}", c, RED) for i, c in enumerate(PLACED_RED)]
DISCS += [_placed(f"blue{i}", c, BLUE) for i, c in enumerate(PLACED_BLUE)]

HOLDER_X = 0.18
HOLDER_Y = {"red": 0.20, "blue": -0.20}
HOLDER_WALL = (0.004, 0.024, 0.006)  # each holder: two low walls 1 mm off the disc's faces


def _holder(colour: str):
    y = HOLDER_Y[colour]
    off = T / 2 + 0.001 + HOLDER_WALL[0] / 2
    return [Box(f"holder_{colour}{i}", pos=(HOLDER_X + s * off, y, TOP + HOLDER_WALL[2] / 2), size=HOLDER_WALL,
                color=(0.3, 0.3, 0.32)) for i, s in enumerate((-1, 1))]


DISC_MASS = 0.002
SPARE_RED = Disc("red_spare", pos=(HOLDER_X, HOLDER_Y["red"], TOP + R + 0.0005), radius=R, height=T, axis="X",
                 mass=DISC_MASS, color=RED)
SPARE_BLUE = Disc("blue_spare", pos=(HOLDER_X, HOLDER_Y["blue"], TOP + R + 0.0005), radius=R, height=T, axis="X",
                  mass=DISC_MASS, color=BLUE)
OBJECTS = [*_frame(), *DISCS, *_holder("red"), *_holder("blue"), SPARE_RED, SPARE_BLUE]


# -- scoring --------------------------------------------------------------------------------------------------------
def slot_of(obj):
    """(column, row) of a disc standing in the frame's slot, or None."""
    x, y, z = (float(v) for v in obj.pos)
    if abs(x - CX) > GAP / 2 or abs(y) > WIDTH / 2 or not BASE_TOP < z < FRAME_TOP + 0.005:
        return None
    col = round(3 - y / PITCH)
    row = round((z - BASE_TOP - R) / D)
    return col, row


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
PAD_TIP = 0.013
GRIP_YAW = -math.pi / 2                         # the left arm's fingers close along x, across the disc's faces
PICK_Z = SPARE_RED.pos[2] + 0.002 + PAD_TIP     # fingertips 2 mm above the disc's centre, above the holder's walls
DROP_Z = FRAME_TOP + 0.003 + PAD_TIP            # fingertips 3 mm above the frame: the disc's lower 9 mm is in the slot
TRAVEL_Z = FRAME_TOP + 0.05


def demo():
    """The left arm lifts the spare red disc out of its holder and drops it into column 5."""
    f0, f1, yaw = FINGER_OPEN, FINGER_CLOSED, GRIP_YAW
    px, py = SPARE_RED.pos[:2]
    wx, wy = CX, col_y(WIN[0])
    return {"left": [
        Waypoint(1.5, (px, py, TRAVEL_Z), f0, "above the disc", yaw),
        Waypoint(1.5, (px, py, PICK_Z), f0, "down", yaw),
        Waypoint(0.8, (px, py, PICK_Z), f1, "grip", yaw),
        Waypoint(1.2, (px, py, TRAVEL_Z), f1, "lift", yaw),
        Waypoint(2.0, (wx, wy, TRAVEL_Z), f1, "over column 5", yaw),
        Waypoint(1.5, (wx, wy, DROP_Z), f1, "into the slot", yaw),
        Waypoint(0.6, (wx, wy, DROP_Z), f0, "let go", yaw),
        Waypoint(1.0, (wx, wy, TRAVEL_Z), f0, "up", yaw),
    ]}
