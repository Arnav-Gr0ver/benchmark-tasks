"""On the station table, between the arms: a KerPlunk-style tower. A clear tube (10 cm across, 16 cm tall) stands on a
wooden base, its inside open down to the base (the collection chamber). Four coloured 6 mm rods cross it through holes
about 10 cm up, and five glass marbles rest on them. Three rods are ledges near the wall, each holding marbles against
the wall; the fourth (blue) runs beside the far ledge, level with its two marbles' middles, touching them. Only the
blue rod can come out without dropping a marble.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
The tower's geometry is in tower.py.
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

import tower  # noqa: E402
from shapekit import Compound, Disc, Marble  # noqa: E402
from stepkit import on_reset, on_step  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Remove exactly one stick from the tower without allowing any marbles to fall into the bottom chamber."

# -- placement (world frame: x forward from the arms, y left, z up; metres) -----------------------------------------
TUBE_XY = (0.30, 0.0)
BASE_T = 0.010
BASE_TOP = TOP + BASE_T


def world(p) -> tuple[float, float, float]:
    """A point in the tube's frame (origin on its axis at the base's top) in the world."""
    return (TUBE_XY[0] + float(p[0]), TUBE_XY[1] + float(p[1]), BASE_TOP + float(p[2]))


BASE = Box("base", pos=(*TUBE_XY, TOP + BASE_T / 2), size=(0.14, 0.14, BASE_T), color=(0.55, 0.40, 0.25))
TUBE = Compound("tube", pos=world((0.0, 0.0, 0.0)), size=(0.1, 0.1, tower.TUBE_H), color=(0.75, 0.88, 0.95),
                boxes=tuple(tower.wall_boxes()) + tuple((c, s) for c, s in tower.bores()), friction=0.4,
                opacity=0.35)

ROD_COLOURS = {"far": (0.95, 0.50, 0.10), "guard": (0.15, 0.35, 0.90), "left": (0.15, 0.65, 0.25),
               "right": (0.95, 0.80, 0.10)}     # orange, blue, green, yellow
ROD_NAMES = {n: f"rod_{c}" for n, c in (("far", "orange"), ("guard", "blue"), ("left", "green"), ("right", "yellow"))}
ROD_MASS = 0.003                                # 6 mm birch dowel, 16 cm
RODS = [Disc(ROD_NAMES[n], pos=world(tower.rod_centre(n)), radius=tower.ROD_R, height=tower.ROD_LEN,
             axis=tower.RODS[n][0].upper(), mass=ROD_MASS, color=ROD_COLOURS[n], friction=(0.4, 0.35))
        for n in tower.RODS]
MARBLE_MASS = 0.0055                            # 16 mm glass
MARBLES = [Marble(n, pos=world((x, y, z + 0.0003)), size=(0.016, 0.016, 0.016), radius=tower.MARBLE_R,
                  mass=MARBLE_MASS, color=(0.55, 0.80, 0.95), friction=0.3)
           for n, (x, y, z) in tower.MARBLES.items()]
OBJECTS = [BASE, TUBE, *RODS, *MARBLES]
SAFE_ROD = ROD_NAMES[tower.SAFE]

# -- rules and scoring ----------------------------------------------------------------------------------------------
LIFTED = 0.015                                  # a marble rising this far above where it started has been handled
_STATE: dict = {"fallen": set(), "handled": set()}


@on_reset
def _clear(environment=None) -> None:
    _STATE.update(fallen=set(), handled=set())


def _local(pos) -> np.ndarray:
    return np.asarray(pos, float) - np.array([TUBE_XY[0], TUBE_XY[1], BASE_TOP])


def _watch(objects) -> None:
    """Latches marbles that fall below the fallen plane (20 mm under the lowest rods), and marbles the robot has
    handled (lifted 15 mm above where they started, or taken out over the tube's rim)."""
    for name, start in tower.MARBLES.items():
        st = objects.get(name)
        if st is None:
            continue
        p = _local(st.pos)
        if p[2] < tower.FALLEN_Z:
            _STATE["fallen"].add(name)
        if p[2] > start[2] + LIFTED or math.hypot(p[0], p[1]) > tower.TUBE_R_IN + 0.002:
            _STATE["handled"].add(name)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    _watch(object_states(environment, list(tower.MARBLES)))


def extraction(objects) -> dict[str, float]:
    """Each rod: how far it has been slid out, 0 (in place) to 1 (out of the tube)."""
    return {tower_name: tower.extracted(tower_name, _local(objects[rod].pos))
            for tower_name, rod in ROD_NAMES.items()}


def outcome(objects) -> tuple[float, str]:
    _watch(objects)
    if _STATE["handled"]:
        return 0.0, f"marbles handled: {', '.join(sorted(_STATE['handled']))}"
    if _STATE["fallen"]:
        return 0.0, f"marbles fell: {', '.join(sorted(_STATE['fallen']))}"
    out = extraction(objects)
    half = [n for n, e in out.items() if e >= 0.5]
    if len(half) > 1:
        return 0.0, f"more than one rod out: {', '.join(ROD_NAMES[n] for n in half)}"
    if half and out[half[0]] >= 1.0:
        return 1.0, f"{ROD_NAMES[half[0]]} is out; no marble fell"
    if half:
        return 0.5, f"{ROD_NAMES[half[0]]} is {100 * out[half[0]]:.0f} % out; no marble fell"
    return 0.0, "no rod is half out"


def success(objects) -> bool:
    """One rod fully out, no marble fallen or handled, the marbles at rest; holds 1 s."""
    return outcome(objects)[0] == 1.0 and all(objects[m].resting() for m in tower.MARBLES)


_LAST = {"why": None}


def score(objects) -> float:
    """1.0 one rod slid fully out of the tube and every marble still on the rods; 0.5 one rod at least half out (its
    last end has gone half the way to the tube's outside) and every marble on the rods; 0 otherwise. 0 if any marble's
    centre ever drops 20 mm below the lowest rods, if more than one rod is half out, or if a marble is handled (lifted
    15 mm above where it started, or taken out over the rim)."""
    value, why = outcome(objects)
    if why.split(" is ")[0] != _LAST["why"]:
        _LAST["why"] = why.split(" is ")[0]
        print(f"[task] {value}: {why}", flush=True)
    return value


# -- scripted demo --------------------------------------------------------------------------------------------------
PAD_TIP = 0.013
ALONG_X = math.pi / 2                   # grip yaw: the fingers close along x, across a rod that runs along y
_g = world(tower.rod_centre(tower.SAFE))
GRIP_Y = _g[1] - tower.ROD_LEN / 2 + 0.015          # 1.5 cm in from the blue rod's long end, on the robot's right
GRIP = (_g[0], GRIP_Y, _g[2] - 0.002 + PAD_TIP)       # fingertips 2 mm below the rod's centre line
_chord = math.sqrt(tower.R_OUT ** 2 - tower.RODS[tower.SAFE][1] ** 2) + 0.0025   # to the outside of its hole
PULL = tower.SHIFT[tower.SAFE] + tower.ROD_LEN / 2 + _chord + 0.012          # its far end 1.2 cm past the wall
DROP_Z = TOP + 0.03


def demo():
    """The right arm takes the blue rod by its long end on the robot's right, slides it straight out (to its right)
    over 5 s, carries it clear of the tower and lays it on the table."""
    a, c = 0.30 * FINGER_OPEN, FINGER_CLOSED
    x, y, z = GRIP
    out = (x, y - PULL, z)
    return {"right": [
        Waypoint(2.0, (x, y, z + 0.05), a, "above the blue rod's end", ALONG_X),
        Waypoint(1.5, (x, y, z), a, "down to it", ALONG_X),
        Waypoint(0.8, (x, y, z), c, "grip", ALONG_X),
        Waypoint(5.0, out, c, "slide it straight out", ALONG_X),
        Waypoint(1.0, (x, y - PULL - 0.03, z), c, "clear of the tower", ALONG_X),
        Waypoint(1.5, (x, y - PULL - 0.03, DROP_Z), c, "down", ALONG_X),
        Waypoint(0.5, (x, y - PULL - 0.03, DROP_Z), a, "let go", ALONG_X),
        Waypoint(1.0, (x, y - PULL - 0.03, DROP_Z + 0.06), a, "up", ALONG_X),
    ]}
