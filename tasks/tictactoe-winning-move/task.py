"""On the station table, between the arms: a white tic-tac-toe board (24 cm square, a 3 x 3 grid of 8 cm cells drawn in
black) with four 4 cm wooden blocks on it as pieces, two red and two blue, and two loose blocks on the table: a red
one to the robot's left of the board and a blue one to its right. Red to move. Rows and columns as seen from behind the arms: row 1 is
the far row, column 1 the left column. Red holds row 1 and row 2 of the middle column; blue holds row 2 and row 3 of
the right column.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    --policy yam_scripted
"""

from __future__ import annotations

import math

import numpy as np
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("You are playing red in tic-tac-toe and it is your turn. Make the winning move: put the spare red "
               "block in the cell that wins the game.")

# -- board (world frame: x forward from the arms, y left, z up; metres) ----------------------------------------------
CENTRE = (0.28, 0.0)
PITCH = 0.08                     # cell size
BOARD_T = 0.006
BOARD_TOP = TOP + BOARD_T
BLOCK, BLOCK_MASS = 0.04, 0.032   # 4 cm pine toy blocks
WHITE, BLACK = (0.94, 0.94, 0.92), (0.08, 0.08, 0.09)
RED, BLUE = (0.85, 0.10, 0.10), (0.12, 0.30, 0.85)


def cell_xy(row: int, col: int) -> tuple[float, float]:
    """Centre of a cell; row 0 is the far row, col 0 the robot's left column."""
    return (CENTRE[0] + (1 - row) * PITCH, CENTRE[1] + (1 - col) * PITCH)


BOARD = Box("board", pos=(*CENTRE, TOP + BOARD_T / 2), size=(3 * PITCH, 3 * PITCH, BOARD_T), color=WHITE)
_LINE_H = 0.0006
LINES = [Box(f"line_x{i}", pos=(CENTRE[0] + s * PITCH / 2, CENTRE[1], BOARD_TOP + _LINE_H / 2),
             size=(0.004, 3 * PITCH - 0.01, _LINE_H), color=BLACK) for i, s in enumerate((-1, 1))]
LINES += [Box(f"line_y{i}", pos=(CENTRE[0], CENTRE[1] + s * PITCH / 2, BOARD_TOP + _LINE_H / 2),
              size=(3 * PITCH - 0.01, 0.004, _LINE_H), color=BLACK) for i, s in enumerate((-1, 1))]

# The position: red (to move) has the middle column's top two cells, blue the right column's lower two.
PLACED = {"red_a": ((0, 1), RED), "red_b": ((1, 1), RED), "blue_a": ((1, 2), BLUE), "blue_b": ((2, 2), BLUE)}
WIN_CELL = (2, 1)                # completes red's middle column
BLOCK_CELL = (0, 2)              # where blue would complete its right column
BLOCK_Z = BOARD_TOP + BLOCK / 2 + 0.0005


def _block(name, xy, color, z):
    return Box(name, pos=(*xy, z), size=(BLOCK, BLOCK, BLOCK), mass=BLOCK_MASS, color=color)


PIECES = [_block(n, cell_xy(*rc), c, BLOCK_Z) for n, (rc, c) in PLACED.items()]
SPARE_RED = _block("red_spare", (0.20, 0.20), RED, TOP + BLOCK / 2 + 0.0005)
SPARE_BLUE = _block("blue_spare", (0.20, -0.20), BLUE, TOP + BLOCK / 2 + 0.0005)
OBJECTS = [BOARD, *LINES, *PIECES, SPARE_RED, SPARE_BLUE]

# -- scoring --------------------------------------------------------------------------------------------------------
IN_CELL = 0.025       # a block's centre within this of a cell's centre
MOVED = 0.015         # a placed block moved further than this has been disturbed
FLAT_DEG = 15.0


def _tilt_deg(obj) -> float:
    """How far a block is from lying flat on one of its faces (deg): the angle between the vertical and the nearest
    of its own axes."""
    x, y, z, w = obj.quat_xyzw
    up = np.abs([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])  # world z in the block's frame
    return float(np.degrees(np.arccos(np.clip(up.max(), -1.0, 1.0))))


def _cell_of(obj):
    """The cell a block sits flat in, on the board (not stacked on another block); else None."""
    if _tilt_deg(obj) > FLAT_DEG or abs(float(obj.low[2]) - BOARD_TOP) > 0.003:
        return None
    for r in range(3):
        for c in range(3):
            if math.dist(obj.pos[:2], cell_xy(r, c)) <= IN_CELL:
                return (r, c)
    return None


def _on_board(obj) -> bool:
    half = 1.5 * PITCH
    return bool(abs(obj.pos[0] - CENTRE[0]) < half and abs(obj.pos[1] - CENTRE[1]) < half
                and float(obj.low[2]) < BOARD_TOP + BLOCK + 0.005)


def _disturbed(objects) -> bool:
    return any(math.dist(objects[n].pos[:2], cell_xy(*rc)) > MOVED or _tilt_deg(objects[n]) > FLAT_DEG
               for n, (rc, _) in PLACED.items())


def outcome(objects) -> tuple[float, str]:
    """(score, why), from the end state."""
    if _disturbed(objects):
        return 0.0, "a block already on the board was moved"
    if _on_board(objects["blue_spare"]):
        return 0.0, "the blue block was played"
    red = objects["red_spare"]
    cell = _cell_of(red)
    if cell == WIN_CELL:
        return 1.0, "winning move"
    taken = {rc for rc, _ in PLACED.values()}
    if cell is not None and cell not in taken:
        return 0.25, f"a legal move that does not win (row {cell[0] + 1}, column {cell[1] + 1})"
    return 0.0, "the red block is not flat in an empty cell"


def success(objects) -> bool:
    """The spare red block flat in the winning cell, at rest, nothing else disturbed; holds 1 s."""
    return outcome(objects)[0] == 1.0 and objects["red_spare"].resting()


def score(objects) -> float:
    """1.0 for the winning move; 0.25 for the red block flat in any other empty cell; 0 if a placed block was moved,
    the blue block played, or otherwise."""
    return outcome(objects)[0]


# -- scripted demo --------------------------------------------------------------------------------------------------
GRIP_Z = TOP + BLOCK / 2                       # the TCP at the block's centre, as the simulator's own block example
PLACE_Z = BOARD_TOP + BLOCK / 2 + 0.004        # the block's underside 4 mm above the board when let go
RELEASE = 0.7 * FINGER_OPEN                    # open only partly, so the fingers stay clear of the next cells


def demo():
    """The left arm picks the spare red block and sets it in the winning cell."""
    plan = pick_place_waypoints((*SPARE_RED.pos[:2], GRIP_Z), (*cell_xy(*WIN_CELL), PLACE_Z), travel_z=TOP + 0.14)
    plan[6] = Waypoint(plan[6].duration, plan[6].pos, RELEASE, "release")
    plan[7] = Waypoint(plan[7].duration, plan[7].pos, RELEASE, "retreat")
    return {"left": plan}
