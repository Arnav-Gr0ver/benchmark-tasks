"""On the station table, between the arms and 22 cm in front of them: a round tilting maze, 28 cm across, on a
gimbal on a grey stand, so the board tilts in any direction when its rim is pressed down (until it rests on the
stand, at 7 degrees) and swings back level when let go. A clear lid covers the maze. Under the lid, four ring-shaped
channels with dark walls, gaps and dead ends surround a green centre cell. A red 16 mm ball lies in the outermost
channel on the side nearest the robot. Tilting the board rolls the ball downhill; the goal is the green centre.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    --policy yam_scripted (open loop), or python demo.py task.py demo_out (closed loop)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Cylinder  # noqa: E402

import maze  # noqa: E402
from mazekit import Ball, Board  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Roll the red ball through the maze into the green centre by pressing down on the edge of the clear "
               "lid to tilt the board.")

# -- layout (world frame: x forward from the arms, y left, z up; metres) --------------------------------------------
CENTRE_XY = (0.22, 0.0)                       # the board's centre, on the midline between the arms
FLOOR_Z = TOP + maze.FLOOR_ABOVE_TABLE        # the maze floor, board level
PIVOT_WORLD = np.array([*CENTRE_XY, FLOOR_Z + maze.PIVOT_Z])

GREY = (0.45, 0.47, 0.50)
RED = (0.85, 0.08, 0.08)

STAND_TOP = FLOOR_Z + maze.STAND_TOP_Z        # the base's underside lands on this rim at a 7 degree tilt
STAND = Cylinder("maze_stand", pos=(*CENTRE_XY, (TOP + STAND_TOP) / 2), radius=maze.STAND_R,
                 height=STAND_TOP - TOP, color=GREY)
BOARD = Board("maze_board", pos=(*CENTRE_XY, FLOOR_Z), size=(2 * maze.BOARD_R, 2 * maze.BOARD_R, 0.03),
              color=GREY)
BALL = Ball("ball", pos=(CENTRE_XY[0] + maze.START_XY[0], CENTRE_XY[1] + maze.START_XY[1],
                         FLOOR_Z + maze.BALL_R + 0.0005), radius=maze.BALL_R, color=RED)
OBJECTS = [STAND, BOARD, BALL]

# -- scoring --------------------------------------------------------------------------------------------------------
RESTING = 0.02  # m/s: the ball counts as resting below this speed


def ball_on_board(objects) -> np.ndarray:
    """The ball's centre in the board frame (m): x, y across the maze floor, z above it."""
    return objects["maze_board"].to_local(objects["ball"].pos)


def lost(local) -> bool:
    """The ball is not between the floor and the lid inside the outer wall (it cannot be, short of a broken board)."""
    return bool(math.hypot(local[0], local[1]) > maze.WALLS[-1] or abs(local[2] - maze.BALL_R) > 0.006)


def success(objects) -> bool:
    """The ball at rest in the green centre cell; must hold for 1 s."""
    local = ball_on_board(objects)
    resting = float(np.linalg.norm(objects["ball"].lin_vel)) < RESTING
    return bool(not lost(local) and maze.ring_at(local[0], local[1]) == 0 and resting)


def distance(objects) -> float:
    """The ball's straight-line distance from the board's centre (m)."""
    local = ball_on_board(objects)
    return float(math.hypot(local[0], local[1]))


def score(objects) -> float:
    """Partial credit by the channel the ball's centre is in at the end: 0 in the outer channel (where it starts),
    0.25 in the next, 0.5, 0.75 in the innermost channel, 1.0 in the centre cell (success, at rest for 1 s)."""
    local = ball_on_board(objects)
    return 0.0 if lost(local) else maze.score_at(local[0], local[1])


# -- scripted demo (open loop; demo.py has a closed-loop one) -------------------------------------------------------
# Each press takes the board down onto its stand (7 degrees), so the tilt is set by the hardware, not by how exactly
# the arm holds. The fewest presses that bring the ball home in plan.py's model at that tilt: plan.plan() returns
# them (about 10 s on a CPU); kept here so loading the task does not run the search. test_tilt_maze.py checks them.
DEMO_HOLDS = (60, 210, 0, 300, 90, 240)
DEMO_TILT = maze.TILT_STOP
PRESS_R = 0.12             # press the lid this far from the board's centre (over the outer channel)
PRESS_DEEPER = 0.002       # command the fingertips this far past the lid on the stand, to hold it there
HOVER = 0.03               # above the lid between presses
PAD_TIP = 0.013            # fingertips past the TCP (yam_arena.constants.PAD_TIP_PAST_TCP)
HOLD_S = 5.5               # long enough for the ball to come to rest in the model (at most 4.1 s)
PARK = {"left": (0.20, 0.25, TOP + 0.15), "right": (0.20, -0.25, TOP + 0.15)}


def arm_for(azimuth_deg: float) -> str:
    """The left arm presses the left half of the rim (0-180 degrees), the right arm the right half. Each arm reaches
    from 0 to 255 degrees (left) or 105 to 360 (right), so 15-90 degrees needs the left and 270-345 the right."""
    return "left" if azimuth_deg % 360.0 <= 180.0 else "right"


def press_point(azimuth_deg: float, tilt: float, above: float = 0.0) -> tuple[float, float, float]:
    """The TCP target for pressing the lid at ``azimuth_deg`` with the board tilted ``tilt`` (rad) toward that side:
    the lid point at PRESS_R, turned about the pivot, then the fingertips ``above`` it."""
    a = math.radians(azimuth_deg)
    u = np.array([math.cos(a), math.sin(a), 0.0])
    p = np.array([PRESS_R * u[0], PRESS_R * u[1], maze.LID_TOP_Z - maze.PIVOT_Z])  # from the pivot, board level
    k = np.array([-u[1], u[0], 0.0])  # z x u: turning about it by +tilt lowers the rim at the azimuth
    c, s = math.cos(tilt), math.sin(tilt)
    p = p * c + np.cross(k, p) * s + k * float(np.dot(k, p)) * (1 - c)
    point = PIVOT_WORLD + p + np.array([0.0, 0.0, PAD_TIP + above])
    return tuple(float(v) for v in point)


def press_waypoints(azimuth_deg: float, hold_s: float = HOLD_S) -> list[Waypoint]:
    """One press: above the rim, down onto the lid, press the board down onto its stand, hold, and back up."""
    f = FINGER_CLOSED
    return [
        Waypoint(1.5, press_point(azimuth_deg, 0.0, HOVER), f, f"above {azimuth_deg:.0f} deg"),
        Waypoint(0.8, press_point(azimuth_deg, 0.0, 0.002), f, "onto the lid"),
        Waypoint(1.0, press_point(azimuth_deg, DEMO_TILT, -PRESS_DEEPER), f, "tilt"),
        Waypoint(hold_s, press_point(azimuth_deg, DEMO_TILT, -PRESS_DEEPER), f, "hold"),
        Waypoint(0.8, press_point(azimuth_deg, 0.0, HOVER), f, "let it level"),
    ]


def demo():
    """Each press in DEMO_HOLDS by the arm that reaches it; the other arm waits parked over its own side. When the
    pressing arm changes, the one that pressed goes back to park before the other comes in."""
    plans = {arm: [Waypoint(1.5, PARK[arm], FINGER_CLOSED, "park")] for arm in ("left", "right")}
    clock = {arm: 1.5 for arm in plans}
    previous = None
    for az in DEMO_HOLDS:
        arm = arm_for(az)
        if previous is not None and previous != arm:
            plans[previous].append(Waypoint(1.2, PARK[previous], FINGER_CLOSED, "park"))
            clock[previous] += 1.2
        start = max(clock.values()) if previous != arm else clock[arm]
        if start > clock[arm]:
            plans[arm].append(Waypoint(start - clock[arm], plans[arm][-1].pos, FINGER_CLOSED, "wait"))
        moves = press_waypoints(az)
        plans[arm] += moves
        clock[arm] = start + sum(w.duration for w in moves)
        previous = arm
    return plans
