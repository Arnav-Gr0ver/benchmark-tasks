"""On the station table, between the arms: a Rush Hour board (6 x 6 cells of 34 mm, walled, one gap in the right-hand
wall: the exit) with seven vehicles, each with a small square peg on its roof to hold it by. The red car is in the
fourth row from the robot, facing the exit; a yellow car stands across that row in the fifth column, between the red
car and the exit. The yellow car cannot go toward the robot (a green car is in the way); it must slide two cells
away from the robot. Then the red car can slide out of the exit onto the red apron.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

from shapekit import Compound  # noqa: E402
from stepkit import on_reset, on_step  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Move the blocking vehicle, then slide the red car out of the exit."

# -- board (world frame: x forward from the arms, y left, z up; metres) ---------------------------------------------
N = 6
C = 0.034                       # cell
CENTRE = (0.30, 0.0)
BASE_T = 0.006
BOARD_TOP = TOP + BASE_T
WALL_T, WALL_H = 0.006, 0.010
EXIT_ROW = 3                    # rows count from the robot's side (x), columns from the robot's left (y)
SLICK = 0.2                     # smooth plastic on smooth plastic: vehicles slide
GREY, WALL_GREY, APRON_RED = (0.78, 0.78, 0.80), (0.30, 0.30, 0.34), (0.95, 0.55, 0.55)


def row_x(r: float) -> float:
    return CENTRE[0] + (r - (N - 1) / 2) * C


def col_y(c: float) -> float:
    return CENTRE[1] + ((N - 1) / 2 - c) * C


INNER = N * C / 2                # the walls' inner faces, from the centre
OUTER = INNER + WALL_T
APRON_LEN = 0.10                 # beyond the exit, where the red car ends up


def _board():
    cx, cy = CENTRE
    side = 2 * OUTER
    base = [((0.0, 0.0, BASE_T / 2), (side, side, BASE_T))]
    walls = []
    for s in (-1, 1):
        walls.append(((s * (INNER + WALL_T / 2), 0.0, BASE_T + WALL_H / 2), (WALL_T, side, WALL_H)))  # near / far
        if s > 0:
            walls.append(((0.0, INNER + WALL_T / 2, BASE_T + WALL_H / 2), (side, WALL_T, WALL_H)))      # left
    # right wall, with the exit's gap in the red car's row
    ex = row_x(EXIT_ROW) - cx
    lo, hi = ex - C / 2, ex + C / 2
    for a, b in ((-OUTER, lo), (hi, OUTER)):
        walls.append((((a + b) / 2, -(INNER + WALL_T / 2), BASE_T + WALL_H / 2), (b - a, WALL_T, WALL_H)))
    board = Compound("board", pos=(cx, cy, TOP), size=(side, side, BASE_T + WALL_H), color=GREY,
                     boxes=tuple(base), friction=SLICK)
    rails = Compound("walls", pos=(cx, cy, TOP), size=(side, side, BASE_T + WALL_H), color=WALL_GREY,
                     boxes=tuple(walls), friction=SLICK)
    apron = Compound("exit_apron", pos=(row_x(EXIT_ROW), cy - OUTER - APRON_LEN / 2, TOP),
                     size=(C + 0.01, APRON_LEN, BASE_T), color=APRON_RED,
                     boxes=(((0.0, 0.0, BASE_T / 2), (C + 0.01, APRON_LEN, BASE_T)),), friction=SLICK)
    return [board, rails, apron]


# -- vehicles -------------------------------------------------------------------------------------------------------
BODY_W, BODY_H, END_GAP = C - 0.006, 0.016, 0.004    # 28 mm wide, 16 mm tall, 4 mm shorter than its cells
PEG = (0.012, 0.014)                                 # a 12 mm square peg, 14 mm tall, on the roof's middle
MASS = {2: 0.025, 3: 0.040}

# name: (colour, "H" (lies along a row: slides left and right) or "V" (along a column: slides toward and away from the
# robot), first row, first column, length in cells)
VEHICLES = {
    "red_car": ((0.85, 0.08, 0.08), "H", 3, 2, 2),
    "yellow_car": ((0.95, 0.80, 0.10), "V", 2, 4, 2),     # the blocker
    "green_car": ((0.15, 0.62, 0.25), "H", 1, 4, 2),      # stops the yellow car going toward the robot
    "purple_car": ((0.55, 0.30, 0.70), "V", 3, 1, 2),
    "teal_car": ((0.10, 0.65, 0.70), "V", 4, 5, 2),
    "blue_truck": ((0.15, 0.30, 0.85), "H", 5, 0, 3),
    "orange_truck": ((0.95, 0.50, 0.10), "V", 0, 0, 3),
}
TARGET, BLOCKER = "red_car", "yellow_car"


def cells(name, start=None) -> list[tuple[int, int]]:
    _, d, r, c, n = VEHICLES[name]
    r, c = start if start else (r, c)
    return [(r + k, c) if d == "V" else (r, c + k) for k in range(n)]


def centre_of(name, start=None) -> tuple[float, float]:
    cs = cells(name, start)
    return (sum(row_x(r) for r, _ in cs) / len(cs), sum(col_y(c) for _, c in cs) / len(cs))


def length_of(name) -> float:
    return VEHICLES[name][4] * C - END_GAP


def _vehicle(name):
    colour, d, _, _, n = VEHICLES[name]
    length = length_of(name)
    x, y = centre_of(name)
    boxes = (((0.0, 0.0, BODY_H / 2), (length, BODY_W, BODY_H)),
             ((0.0, 0.0, BODY_H + PEG[1] / 2), (PEG[0], PEG[0], PEG[1])))
    # a vehicle's own x is along its length: an "H" vehicle is turned a quarter turn
    return Compound(name, pos=(x, y, BOARD_TOP + 0.0005), yaw=0.0 if d == "V" else math.pi / 2,
                    size=(length, BODY_W, BODY_H + PEG[1]), mass=MASS[n], color=colour, boxes=boxes, friction=SLICK)


OBJECTS = [*_board(), *(_vehicle(n) for n in VEHICLES)]

# -- rules and scoring ----------------------------------------------------------------------------------------------
LIFT_LIMIT = 0.008              # a vehicle's underside more than this above the board: lifted (it could hop others)
LANE_OFF = 0.008                # off its lane's centre line by more than this, or
LANE_TURN_DEG = 15.0            # turned more than this: out of its lane
_STATE = {"lifted": set()}


@on_reset
def _clear(environment=None) -> None:
    _STATE["lifted"] = set()


def _watch(objects) -> None:
    for name in VEHICLES:
        st = objects.get(name)
        if st is not None and float(st.pos[2]) - BOARD_TOP > LIFT_LIMIT:
            _STATE["lifted"].add(name)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    _watch(object_states(environment, list(VEHICLES)))


def _yaw(state) -> float:
    if hasattr(state, "yaw"):
        return state.yaw()
    x, y, z, w = state.quat_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def in_lane(name, state) -> bool:
    _, d, _, _, _ = VEHICLES[name]
    x0, y0 = centre_of(name)
    off = abs(float(state.pos[1]) - y0) if d == "V" else abs(float(state.pos[0]) - x0)
    want = 0.0 if d == "V" else math.pi / 2
    turn = abs((math.degrees(_yaw(state) - want) + 90.0) % 180.0 - 90.0)
    return off <= LANE_OFF and turn <= LANE_TURN_DEG


def red_out(state) -> bool:
    """The red car's trailing end past the right wall's outer face (2 mm grace), on the apron."""
    return float(state.pos[1]) + length_of(TARGET) / 2 <= CENTRE[1] - OUTER + 0.002


def row_clear(state) -> bool:
    """The yellow car no longer reaches into the red car's row (2 mm grace)."""
    lo, hi = float(state.pos[0]) - length_of(BLOCKER) / 2, float(state.pos[0]) + length_of(BLOCKER) / 2
    band = (row_x(EXIT_ROW) - C / 2 + 0.002, row_x(EXIT_ROW) + C / 2 - 0.002)
    return hi <= band[0] or lo >= band[1]


def outcome(objects) -> tuple[float, str]:
    _watch(objects)
    if _STATE["lifted"]:
        return 0.0, f"lifted off the board: {', '.join(sorted(_STATE['lifted']))}"
    strays = [n for n in VEHICLES if not (n == TARGET and red_out(objects[n])) and not in_lane(n, objects[n])]
    if strays:
        return 0.0, f"out of its lane: {', '.join(strays)}"
    if red_out(objects[TARGET]):
        return 1.0, "the red car is out"
    if row_clear(objects[BLOCKER]):
        return 0.5, "the yellow car is clear of the red car's row; the red car is not out"
    return 0.0, "the red car's way is still blocked"


def success(objects) -> bool:
    """The red car out of the exit, nothing lifted, every vehicle in its lane, the red car at rest; holds 1 s."""
    return outcome(objects)[0] == 1.0 and objects[TARGET].resting()


def score(objects) -> float:
    """1.0 the red car out of the exit; 0.5 the yellow car slid clear of the red car's row but the red car not out;
    0 otherwise. 0 whenever any vehicle was lifted more than 8 mm off the board at any time, or ends out of its lane
    (more than 8 mm off its lane's centre line or turned more than 15 degrees)."""
    return outcome(objects)[0]


# -- scripted demo --------------------------------------------------------------------------------------------------
PAD_TIP = 0.013
HOLD = 0.002                    # fingertips this far above the roof, on the peg
CARRY = 0.003                   # vehicles slide 3 mm clear of the board (well under the 8 mm limit)
GRIP_Z = BOARD_TOP + 0.0005 + BODY_H + HOLD + PAD_TIP
TRAVEL_Z = TOP + 0.09
APPROACH = 0.28 * FINGER_OPEN   # 13 mm a side: clear of the neighbouring pegs
YELLOW_TO = (EXIT_ROW + 1, 4)   # the yellow car's first cell after its slide: rows 4 and 5
RED_TO_Y = CENTRE[1] - OUTER - length_of(TARGET) / 2 - 0.012


def demo():
    """The right arm takes the yellow car by its peg (fingers closing along its length, so they push on the peg's
    faces), slides it two cells away from the robot and lets go; then takes the red car the same way and slides it
    out of the exit onto the apron."""
    a, c, r = APPROACH, FINGER_CLOSED, APPROACH
    yx, yy = centre_of(BLOCKER)
    yx2, _ = centre_of(BLOCKER, YELLOW_TO)
    rx, ry = centre_of(TARGET)
    along_x, along_y = math.pi / 2, 0.0   # grip yaws: the fingers close along x, along y
    up = GRIP_Z + CARRY
    return {"right": [
        Waypoint(2.0, (yx, yy, TRAVEL_Z), a, "above the yellow car", along_x),
        Waypoint(1.5, (yx, yy, GRIP_Z), a, "down to its peg", along_x),
        Waypoint(0.8, (yx, yy, GRIP_Z), c, "grip", along_x),
        Waypoint(0.4, (yx, yy, up), c, "ease up", along_x),
        Waypoint(2.0, (yx2, yy, up), c, "slide two cells away", along_x),
        Waypoint(0.4, (yx2, yy, GRIP_Z), c, "set down", along_x),
        Waypoint(0.5, (yx2, yy, GRIP_Z), r, "let go", along_x),
        Waypoint(1.0, (yx2, yy, TRAVEL_Z), r, "up", along_x),
        Waypoint(2.0, (rx, ry, TRAVEL_Z), a, "above the red car", along_y),
        Waypoint(1.5, (rx, ry, GRIP_Z), a, "down to its peg", along_y),
        Waypoint(0.8, (rx, ry, GRIP_Z), c, "grip", along_y),
        Waypoint(0.4, (rx, ry, up), c, "ease up", along_y),
        Waypoint(3.0, (rx, RED_TO_Y, up), c, "slide out of the exit", along_y),
        Waypoint(0.4, (rx, RED_TO_Y, GRIP_Z), c, "set down", along_y),
        Waypoint(0.5, (rx, RED_TO_Y, GRIP_Z), r, "let go", along_y),
        Waypoint(1.0, (rx, RED_TO_Y, TRAVEL_Z), r, "up", along_y),
    ]}
