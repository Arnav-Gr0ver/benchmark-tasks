"""On the station table, between the arms: five square wooden sticks (14 cm x 8 mm x 8 mm) in a small pile on the table,
and a cream mat to the robot's right for removed sticks. The red stick lies along the table pointing away from the
robot, between the yellow stick (3 cm to its left) and the black stick (3 cm to its right), parallel to them. The green
stick lies across the far end of all three, 1.2 cm past their ends. The blue stick lies across the middle of the
yellow, red and black sticks, resting on all three: it traps the red stick.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

from shapekit import Compound  # noqa: E402
from stepkit import on_reset, on_step  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Remove the red stick from the pile. You may remove the blue stick first, but do not disturb the green, "
               "yellow, or black sticks.")

# -- sticks (world frame: x forward from the arms, y left, z up; metres) ----------------------------------------------
LENGTH, SIDE = 0.140, 0.008
MASS = 0.006                        # beech, 14 cm x 8 x 8 mm
FRICTION = 0.5                      # wood on wood
PILE = (0.30, 0.0)
GAP = 0.030                         # the yellow and black sticks' centre lines, either side of the red one's
COLOURS = {"red": (0.85, 0.10, 0.10), "blue": (0.15, 0.35, 0.90), "green": (0.10, 0.65, 0.25),
           "yellow": (0.95, 0.80, 0.10), "black": (0.08, 0.08, 0.08)}
LIFT = 0.0005                       # spawned this far above their supports, to settle

# name: (x, y, z of the centre above the table, yaw)
START = {
    "red": (PILE[0], PILE[1], SIDE / 2, 0.0),
    "yellow": (PILE[0], PILE[1] + GAP, SIDE / 2, 0.0),
    "black": (PILE[0], PILE[1] - GAP, SIDE / 2, 0.0),
    "green": (PILE[0] + LENGTH / 2 + 0.012 + SIDE / 2, PILE[1], SIDE / 2, math.pi / 2),
    "blue": (PILE[0], PILE[1], SIDE + SIDE / 2, math.pi / 2),       # across the middle of the three, on top of them
}
PROTECTED = ("green", "yellow", "black")


def _stick(name) -> Compound:
    x, y, z, yaw = START[name]
    return Compound(name, pos=(x, y, TOP + z + LIFT * (2 if name == "blue" else 1)), yaw=yaw,
                    size=(LENGTH, SIDE, SIDE), mass=MASS, color=COLOURS[name],
                    boxes=(((0.0, 0.0, 0.0), (LENGTH, SIDE, SIDE)),), friction=FRICTION)


STICKS = [_stick(n) for n in START]

# The mat for removed sticks: 16 x 16 cm, to the robot's right of the pile.
MAT_XY, MAT_SIDE, MAT_T = (0.30, -0.20), 0.16, 0.002
MAT = Box("mat", pos=(*MAT_XY, TOP + MAT_T / 2), size=(MAT_SIDE, MAT_SIDE, MAT_T), color=(0.93, 0.89, 0.78))
OBJECTS = [MAT, *STICKS]

# -- rules and scoring ----------------------------------------------------------------------------------------------
MAX_SHIFT, MAX_TURN_DEG = 0.003, 3.0    # a protected stick may move this much, at any time
_STATE: dict = {"ref": None, "disturbed": set()}


@on_reset
def _clear(environment=None) -> None:
    _STATE.update(ref=None, disturbed=set())


def _turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    m = r1 @ r0.T
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(m) - 1) / 2))))


def _watch(objects) -> None:
    """Latches any protected stick moving more than 3 mm or turning more than 3 degrees from where it first was."""
    states = {n: objects[n] for n in PROTECTED if n in objects}
    if _STATE["ref"] is None:
        _STATE["ref"] = {n: (np.array(s.pos, float), s.rotation()) for n, s in states.items()}
        return
    for n, s in states.items():
        p0, r0 = _STATE["ref"][n]
        if np.linalg.norm(np.asarray(s.pos) - p0) > MAX_SHIFT or _turn_deg(r0, s.rotation()) > MAX_TURN_DEG:
            _STATE["disturbed"].add(n)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    _watch(object_states(environment, list(PROTECTED)))


def on_mat(state) -> bool:
    """A stick lying on the mat: its centre over it and low (not held above it)."""
    x, y, z = (float(v) for v in state.pos)
    return (abs(x - MAT_XY[0]) < MAT_SIDE / 2 and abs(y - MAT_XY[1]) < MAT_SIDE / 2
            and z < TOP + MAT_T + 0.02)


def out_of_pile(state) -> bool:
    """A stick taken out of the pile: on the mat, or its centre more than 10 cm from the pile's centre."""
    return on_mat(state) or math.dist(state.pos[:2], PILE) > 0.10


def outcome(objects) -> tuple[float, str]:
    _watch(objects)
    if _STATE["disturbed"]:
        return 0.0, f"disturbed: {', '.join(sorted(_STATE['disturbed']))}"
    if on_mat(objects["red"]):
        return 1.0, "the red stick is on the mat"
    if out_of_pile(objects["blue"]):
        return 0.5, "the blue stick is out of the pile; the red stick is not on the mat"
    return 0.0, "neither stick is out"


def success(objects) -> bool:
    """The red stick at rest on the mat, no protected stick ever disturbed; holds 1 s."""
    return outcome(objects)[0] == 1.0 and objects["red"].resting()


_LAST = {"why": None}


def score(objects) -> float:
    """1.0 the red stick lying on the mat; 0.5 the blue stick out of the pile (on the mat or over 10 cm from its
    centre) but the red stick not on the mat; 0 otherwise. 0 whenever a green, yellow or black stick moves more than
    3 mm or turns more than 3 degrees at any time."""
    value, why = outcome(objects)
    if why != _LAST["why"]:
        _LAST["why"] = why
        print(f"[task] {value}: {why}", flush=True)
    return value


# -- scripted demo --------------------------------------------------------------------------------------------------
PAD_TIP = 0.013
TRAVEL_Z = TOP + 0.08
APPROACH = 0.25 * FINGER_OPEN       # open 12 mm a side: the fingers' outsides stay 6 mm clear of the yellow and black
RELEASE = 0.25 * FINGER_OPEN
ALONG_X, ALONG_Y = math.pi / 2, 0.0  # grip yaws: the fingers close along x / along y


def _grip_z(z_centre: float) -> float:
    """The TCP height that puts the fingertips 2 mm below a stick's centre."""
    return TOP + z_centre - 0.002 + PAD_TIP


BLUE_TO = (0.30, -0.17)              # on the mat, turned to lie along x
RED_TO = (0.30, -0.23)


def demo():
    """The right arm lifts the blue stick straight up by its middle (the fingers 2 mm above the red stick), lays it on
    the mat, then lifts the red stick straight up by its middle (the fingers between it and the yellow and black
    sticks) and lays it on the mat beside the blue one."""
    a, c = APPROACH, FINGER_CLOSED
    bx, by, bz, _ = START["blue"]
    rx, ry, rz, _ = START["red"]
    mat_z = MAT_T
    return {"right": [
        Waypoint(2.0, (bx, by, TRAVEL_Z), a, "above the blue stick", ALONG_X),
        Waypoint(1.5, (bx, by, _grip_z(bz)), a, "down", ALONG_X),
        Waypoint(0.8, (bx, by, _grip_z(bz)), c, "grip", ALONG_X),
        Waypoint(1.5, (bx, by, TRAVEL_Z), c, "straight up", ALONG_X),
        Waypoint(2.0, (*BLUE_TO, TRAVEL_Z), c, "over the mat, turning it along x", ALONG_Y),
        Waypoint(1.2, (*BLUE_TO, _grip_z(mat_z + SIDE / 2) + 0.003), c, "down", ALONG_Y),
        Waypoint(0.5, (*BLUE_TO, _grip_z(mat_z + SIDE / 2) + 0.003), RELEASE, "let go", ALONG_Y),
        Waypoint(1.0, (*BLUE_TO, TRAVEL_Z), RELEASE, "up", ALONG_Y),
        Waypoint(2.0, (rx, ry, TRAVEL_Z), a, "above the red stick", ALONG_Y),
        Waypoint(1.5, (rx, ry, _grip_z(rz)), a, "down between the yellow and black sticks", ALONG_Y),
        Waypoint(0.8, (rx, ry, _grip_z(rz)), c, "grip", ALONG_Y),
        Waypoint(1.5, (rx, ry, TRAVEL_Z), c, "straight up", ALONG_Y),
        Waypoint(2.0, (*RED_TO, TRAVEL_Z), c, "over the mat", ALONG_Y),
        Waypoint(1.2, (*RED_TO, _grip_z(mat_z + SIDE / 2) + 0.003), c, "down", ALONG_Y),
        Waypoint(0.5, (*RED_TO, _grip_z(mat_z + SIDE / 2) + 0.003), RELEASE, "let go", ALONG_Y),
        Waypoint(1.0, (*RED_TO, TRAVEL_Z), RELEASE, "up", ALONG_Y),
    ]}
