"""On the station table, between the arms: a buzz-wire game. A bent 3 mm steel wire runs away from the robot, 6 cm
above a wooden base, from a post near the robot to a post 23 cm further out. It starts straight under a green
insulating sleeve, bends to the robot's left while climbing 2.5 cm, swings back to the right over a crest, drops,
bends right again and ends straight under a red sleeve. A metal ring (24 mm inside) with a 7 cm handle sits around the
wire at the green sleeve, the ring's bottom in a grooved stand and the handle's end on a rest, the handle pointing to
the robot's right.

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

import track  # noqa: E402
from shapekit import Compound, Wire  # noqa: E402
from stepkit import on_reset, on_step  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Move the metal ring from the green starting point to the red endpoint along the wire without allowing "
               "the ring to touch the wire.")

STEEL, GREEN, RED = (0.72, 0.73, 0.76), (0.10, 0.70, 0.20), (0.85, 0.10, 0.10)
WOOD = (0.62, 0.45, 0.28)


def lift(p) -> tuple[float, float, float]:
    """A point of the track (heights above the table) in the world."""
    return (float(p[0]), float(p[1]), float(p[2]) + TOP)


# -- the base, the wire and its posts -------------------------------------------------------------------------------
BASE_T = 0.010
BASE_TOP = TOP + BASE_T
_lo, _hi = track.PATH.min(axis=0), track.PATH.max(axis=0)
BASE = Box("base", pos=((_lo[0] + _hi[0]) / 2, (_lo[1] + _hi[1]) / 2 - 0.02, TOP + BASE_T / 2),
           size=(_hi[0] - _lo[0] + 0.05, _hi[1] - _lo[1] + 0.14, BASE_T), color=WOOD)


def _wire() -> Wire:
    """Capsules every 6 mm along the path: under a sleeve, a 7 mm green or red capsule, else the 3 mm steel wire;
    and a post from each end straight down into the base."""
    segs = []
    marks = np.arange(0.0, track.LENGTH, 0.006)
    marks = np.append(marks, track.LENGTH)
    for s0, s1 in zip(marks, marks[1:]):
        mid = (s0 + s1) / 2
        if mid < track.S_GREEN:
            r, c = track.SLEEVE_R, 1
        elif mid > track.S_RED:
            r, c = track.SLEEVE_R, 2
        else:
            r, c = track.WIRE_R, 0
        segs.append((tuple(track.point(s0)), tuple(track.point(s1)), r, c))
    for end in (track.PATH[0], track.PATH[-1]):
        segs.append(((end[0], end[1], BASE_T - 0.002), tuple(end), 0.003, 0))
    return Wire("wire", pos=(0.0, 0.0, TOP), size=(0.25, 0.1, 0.1), segments=tuple(segs),
                colours=(STEEL, GREEN, RED))


# -- the ring and its handle ----------------------------------------------------------------------------------------
N_SEG = 24
HANDLE_LEN, HANDLE_W = 0.070, 0.010
RING_OUT = track.RING_R + track.TUBE_R          # 15 mm
GRIP_ALONG = 0.040                              # the fingers hold the handle 4 cm out from the ring
RING_MASS = 0.020                               # steel ring and a plastic handle
S0 = 0.02                                       # the ring starts here, over the green sleeve


def _ring_boxes():
    """In the ring's frame: x along its axis, the handle along -y, z completing the frame. 24 chords of the circle,
    each 3 x 3 mm across, tilted about x; and the handle."""
    chord = 2 * track.RING_R * math.sin(math.pi / N_SEG) * 1.08
    d = 2 * track.TUBE_R
    boxes = []
    for k in range(N_SEG):
        a = 2 * math.pi * k / N_SEG
        boxes.append(((0.0, track.RING_R * math.cos(a), track.RING_R * math.sin(a)), (d, chord, d),
                      (math.degrees(a) + 90.0, 0.0)))
    boxes.append(((0.0, -(RING_OUT - 0.001 + HANDLE_LEN / 2), 0.0), (HANDLE_W, HANDLE_LEN, HANDLE_W)))
    return tuple(boxes)


RING_START = lift(track.point(S0))
RING = Compound("ring", pos=RING_START, yaw=0.0, size=(2 * RING_OUT, 2 * RING_OUT + HANDLE_LEN, 2 * RING_OUT),
                mass=RING_MASS, color=(0.70, 0.70, 0.74), boxes=_ring_boxes(), friction=0.6)
assert np.allclose(track.ring_frame(S0), np.eye(3), atol=1e-6)   # the path starts straight along x

# The start stand: a pillar with a groove the ring's bottom sits in, and a rest under the handle's end.
STAND_TOP = RING_START[2] - RING_OUT
GROOVE = 2 * track.TUBE_R + 0.002               # 1 mm clear on each side of the ring
_stand_h = STAND_TOP - BASE_TOP
_rest_y = -(RING_OUT + HANDLE_LEN - 0.006)
_rest_top = RING_START[2] - HANDLE_W / 2
STAND = [
    Box("stand", pos=(RING_START[0], 0.0, BASE_TOP + _stand_h / 2), size=(0.018, 0.016, _stand_h), color=WOOD),
    *[Box(f"stand_lip{i}", pos=(RING_START[0] + s * (GROOVE / 2 + 0.0015), 0.0, STAND_TOP + 0.0015),
          size=(0.003, 0.016, 0.003), color=WOOD) for i, s in enumerate((-1, 1))],
    Box("handle_rest", pos=(RING_START[0], _rest_y, (BASE_TOP + _rest_top) / 2),
        size=(0.016, 0.010, _rest_top - BASE_TOP), color=WOOD),
]
OBJECTS = [BASE, _wire(), RING, *STAND]

# -- scoring --------------------------------------------------------------------------------------------------------
_STATE = {"touched": False, "best": 0.0}


@on_reset
def _clear(environment=None) -> None:
    _STATE.update(touched=False, best=0.0)


def _watch(ring) -> None:
    """Latches the first touch of the bare wire; until then, keeps the furthest point the ring has reached."""
    if _STATE["touched"]:
        return
    rot = ring.rotation()
    centre = np.asarray(ring.pos, float) - np.array([0.0, 0.0, TOP])   # track heights are above the table
    if track.clearance(centre, rot) < track.TOUCH:
        _STATE["touched"] = True
        return
    _STATE["best"] = max(_STATE["best"], track.progress(centre))


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    _watch(object_states(environment, ["ring"])["ring"])


def outcome(objects) -> tuple[float, str]:
    _watch(objects["ring"])
    reached = sum(_STATE["best"] >= c - 1e-9 for c in track.CHECKPOINTS)
    value = 1.0 if reached == 5 else 0.2 * reached
    note = " before touching the wire" if _STATE["touched"] else ""
    return value, f"{reached} of 5 checkpoints{note}"


def success(objects) -> bool:
    """The ring at the red endpoint without having touched the bare wire; holds 1 s."""
    return outcome(objects)[0] == 1.0 and not _STATE["touched"]


_LAST = {"why": None}


def score(objects) -> float:
    """0.2 for each checkpoint (20, 40, 60, 80 % of the bare wire) the ring passes, in order, before it first touches
    the bare wire; 1.0 for reaching the red sleeve without touching it. Touching the green or red sleeve does not
    count. Credit earned before the first touch is kept."""
    value, why = outcome(objects)
    if why != _LAST["why"]:
        _LAST["why"] = why
        print(f"[task] {value}: {why}", flush=True)
    return value


# -- scripted demo --------------------------------------------------------------------------------------------------
# The ring has to turn to follow the wire: about its vertical axis round the bends and about its handle over the
# crest. yam_arena's scripted waypoints only describe straight-down grips turned about the vertical (their `yaw`), so
# each waypoint here passes a tag (TAG + k) as its yaw, and this file teaches yam_arena.demo.grip_rotation to read
# the tag as the k-th gripper orientation in ROTATIONS. Ordinary yaws (within a turn of 0) are unchanged.
TAG = 100.0
# one list for every copy of this file yam_arena loads, since the tags index it
ROTATIONS: list[np.ndarray] = getattr(_yam_demo, "_buzz_wire_rotations", [])
_yam_demo._buzz_wire_rotations = ROTATIONS
_plain_grip_rotation = getattr(_yam_demo.grip_rotation, "plain", _yam_demo.grip_rotation)


def _grip_rotation(yaw: float = 0.0) -> np.ndarray:
    if yaw < TAG / 2:
        return _plain_grip_rotation(yaw)
    return ROTATIONS[int(round(yaw - TAG))]


_grip_rotation.plain = _plain_grip_rotation
_yam_demo.grip_rotation = _grip_rotation

PAD_TIP = 0.013
GRIP_YAW = math.pi / 2                           # the fingers close along x, across the handle
GRIP_POINT = np.array(RING_START) + np.array([0.0, -(RING_OUT + GRIP_ALONG), 0.0])
TCP0 = GRIP_POINT + np.array([0.0, 0.0, PAD_TIP - 0.002])   # fingertips 2 mm below the handle's centre line
UNSEAT = 0.005                                   # the ring rises 5 mm out of its groove first
STEP_S = 0.008                                   # a waypoint every 8 mm of wire
# The ring is round, so its axis need only lie within about 75 degrees of the wire's direction, either way along it;
# the wire's heading swings through 180 degrees (more than the wrist can turn), so the demo turns the ring through
# half of each change of heading: never more than about 45 degrees off the wire's line.
FOLLOW = 0.5
S_END = track.LENGTH - 0.015                     # the middle of the red sleeve


def _tagged(rot: np.ndarray) -> float:
    ROTATIONS.append(rot)
    return TAG + len(ROTATIONS) - 1


def pose_along(s: float, rise: float = 0.0):
    """Where the TCP and the gripper's orientation must be for the ring to sit centred on the wire at s, its axis
    along the wire and its handle level (``rise`` lifts the ring above the wire)."""
    g0 = _plain_grip_rotation(GRIP_YAW)
    r = track.ring_frame(s, FOLLOW)
    centre = np.array(lift(track.point(s))) + np.array([0.0, 0.0, rise])
    tcp = centre + r @ (TCP0 - np.array(RING_START))   # the ring's starting frame is the world's
    return tcp, r @ g0


_DEMO: dict = {}


def demo():
    """The right arm grips the handle from above, raises the ring 5 mm out of its stand, then carries it along the
    wire to the red sleeve, turning it to keep its axis along the wire, and holds it there."""
    if not _DEMO:  # built once: the waypoints' tags index ROTATIONS
        _DEMO["right"] = _plan()
    return dict(_DEMO)


def _plan() -> list:
    f0, f1 = 0.45 * FINGER_OPEN, FINGER_CLOSED
    above = TCP0 + np.array([0.0, 0.0, 0.05])
    way = [
        Waypoint(2.0, tuple(above), f0, "above the handle", GRIP_YAW),
        Waypoint(1.5, tuple(TCP0), f0, "down to the handle", GRIP_YAW),
        Waypoint(0.8, tuple(TCP0), f1, "grip", GRIP_YAW),
    ]
    tcp, rot = pose_along(S0, UNSEAT)
    way.append(Waypoint(0.8, tuple(tcp), f1, "raise the ring out of its stand", _tagged(rot)))
    s_marks = list(np.arange(S0 + STEP_S, S_END, STEP_S)) + [S_END]
    for s in s_marks:
        rise = UNSEAT * max(0.0, 1.0 - (s - S0) / 0.02)       # settle onto the wire's centre over the first 2 cm
        tcp, rot = pose_along(s, rise)
        way.append(Waypoint(0.45, tuple(tcp), f1, f"along the wire, {100 * s / track.LENGTH:.0f} %", _tagged(rot)))
    way.append(Waypoint(2.0, tuple(tcp), f1, "hold at the red sleeve", _tagged(rot)))
    return way
