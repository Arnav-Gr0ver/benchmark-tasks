"""On the station table, between the arms: a wrong equation, 9 - 6 = 7, laid out in wooden sticks on a black mat,
read left to right from the robot's side. Digits are seven-segment, one stick per segment; the minus is one stick,
the equals sign two. Moving one stick fixes it: the top of the 7 onto the 6's upper right makes 9 - 8 = 1. No other
single move makes a true equation.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
"""

from __future__ import annotations

import math

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Make the equation mathematically correct by moving exactly one matchstick."

# -- sticks and layout (world frame: x forward from the arms, y left, z up; metres) --------------------------------
STICK = (0.040, 0.008, 0.008)   # length, width, height
STICK_MASS = 0.0035             # 40 x 8 x 8 mm hardwood
WOOD = (0.93, 0.78, 0.50)
MAT = Box("mat", pos=(0.29, 0.0, TOP + 0.0015), size=(0.17, 0.36, 0.003), color=(0.08, 0.08, 0.09))
MAT_TOP = TOP + 0.003
ROW_X = 0.29                    # the equation's middle line
W = 0.052                       # a digit's segment pitch: its width, and half its height

# A digit's segments in the usual letters, in the digit's own frame (u to the reader's right = the robot's right, v
# away from the robot): centre and whether the stick runs along u (across) or v (up).
SEGMENTS = {
    "a": ((0.0, W), "u"), "b": ((W / 2, W / 2), "v"), "c": ((W / 2, -W / 2), "v"), "d": ((0.0, -W), "u"),
    "e": ((-W / 2, -W / 2), "v"), "f": ((-W / 2, W / 2), "v"), "g": ((0.0, 0.0), "u"),
}
DIGITS = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg",
          9: "abcdfg"}
# where people draw them otherwise (6 and 9 without their tails, 7 with a hook): any of these reads as the digit
DIGIT_FORMS = {**{d: (s,) for d, s in DIGITS.items()}, 6: ("acdefg", "cdefg"), 7: ("abc", "abcf"),
               9: ("abcdfg", "abcfg")}
OPERATOR_SLOTS = {"h": ((0.0, 0.0), "u"), "v": ((0.0, 0.0), "v")}            # minus: h; plus: h and v
EQUALS_SLOTS = {"top": ((0.0, 0.011), "u"), "bottom": ((0.0, -0.011), "u")}

# the five symbols, left to right as the robot reads them, and where each sits along y
SYMBOLS = [("d1", 0.135), ("op", 0.07), ("d2", 0.0), ("eq", -0.07), ("d3", -0.135)]
START = {"d1": 9, "op": "-", "d2": 6, "d3": 7}
SOLUTION = (9, "-", 8, 1)
MOVE = ("d3_a", "d2_b")         # the stick to move, the slot it goes to


def slots() -> dict[str, tuple[tuple[float, float], float]]:
    """Every place a stick can lie: name -> ((x, y) centre, yaw of the stick's length)."""
    out = {}
    for sym, y0 in SYMBOLS:
        table = OPERATOR_SLOTS if sym == "op" else EQUALS_SLOTS if sym == "eq" else SEGMENTS
        for seg, ((u, v), along) in table.items():
            out[f"{sym}_{seg}"] = ((ROW_X + v, y0 - u), 0.0 if along == "v" else math.pi / 2)
    return out


SLOTS = slots()


def start_slots() -> list[str]:
    out = [f"{sym}_{s}" for sym in ("d1", "d2", "d3") for s in DIGITS[START[sym]]]
    out += ["op_h"] + (["op_v"] if START["op"] == "+" else []) + ["eq_top", "eq_bottom"]
    return out


def _stick(slot: str) -> Box:
    (x, y), yaw = SLOTS[slot]
    return Box(slot, pos=(x, y, MAT_TOP + STICK[2] / 2 + 0.0005), size=STICK, yaw=yaw, mass=STICK_MASS, color=WOOD)


STICKS = [_stick(s) for s in start_slots()]   # each stick is named for the slot it starts in
OBJECTS = [MAT, *STICKS]


# -- reading the equation -------------------------------------------------------------------------------------------
SLOT_DIST = 0.012               # a stick's centre within 1.2 cm of a slot's
SLOT_ANGLE_DEG = 20.0           # and its length within 20 degrees of the slot's direction
LYING = MAT_TOP + 0.012         # its centre this low: lying on the mat, not held or on another stick


def _yaw(quat_xyzw) -> float:
    x, y, z, w = quat_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def slot_of(state) -> str | None:
    if float(state.pos[2]) > LYING:
        return None
    best = None
    for name, ((x, y), yaw) in SLOTS.items():
        d = math.dist(state.pos[:2], (x, y))
        turn = abs((math.degrees(_yaw(state.quat_xyzw) - yaw) + 90.0) % 180.0 - 90.0)
        if d <= SLOT_DIST and turn <= SLOT_ANGLE_DEG and (best is None or d < best[0]):
            best = (d, name)
    return best[1] if best else None


def read(occupied: set[str]):
    """The equation the occupied slots spell, as (d1, op, d2, d3), or None if any symbol is not one."""
    def digit(sym):
        segs = {s.split("_")[1] for s in occupied if s.startswith(sym + "_")}
        for d, forms in DIGIT_FORMS.items():
            if any(segs == set(f) for f in forms):
                return d
        return None

    op = {s.split("_")[1] for s in occupied if s.startswith("op_")}
    eq = {s.split("_")[1] for s in occupied if s.startswith("eq_")}
    sign = {frozenset("h"): "-", frozenset("hv"): "+"}.get(frozenset(op))
    out = (digit("d1"), sign, digit("d2"), digit("d3"))
    return None if None in out or eq != {"top", "bottom"} else out


def true(eq) -> bool:
    d1, op, d2, d3 = eq
    return (d1 + d2 if op == "+" else d1 - d2) == d3


def outcome(objects) -> tuple[float, str]:
    where = {s.name: slot_of(objects[s.name]) for s in STICKS}
    moved = [n for n, slot in where.items() if slot != n]
    occupied = {slot for slot in where.values() if slot}
    if len(occupied) < sum(slot is not None for slot in where.values()):
        return 0.0, "two sticks in one place"
    eq = read(occupied) if None not in where.values() else None
    if eq and true(eq) and len(moved) == 1:
        return 1.0, f"{eq[0]} {eq[1]} {eq[2]} = {eq[3]}, one stick moved"
    if moved == [MOVE[0]]:
        went = "is not lying in a slot" if where[MOVE[0]] is None else f"went to {where[MOVE[0]]}"
        return 0.25, f"the right stick was moved but {went}"
    if not moved:
        return 0.0, "nothing moved"
    return 0.0, f"moved {', '.join(moved)}"


def success(objects) -> bool:
    """A true equation with exactly one stick moved, every stick lying in a slot and at rest; holds 1 s."""
    return outcome(objects)[0] == 1.0 and all(objects[s.name].resting() for s in STICKS)


def score(objects) -> float:
    """1.0 the sticks spell a true equation (only 9 - 8 = 1 can be made) and exactly one stick moved; 0.25 only the
    right stick (the 7's top) moved but it does not lie in the slot that makes the equation true; 0 otherwise."""
    return outcome(objects)[0]


# -- scripted demo --------------------------------------------------------------------------------------------------
PAD_TIP = 0.013
GRIP_Z = MAT_TOP + STICK[2] / 2 + PAD_TIP       # fingertips at the stick's mid-height
PLACE_Z = GRIP_Z + 0.002                        # let go 2 mm up
TRAVEL_Z = TOP + 0.08
APPROACH = 0.30 * FINGER_OPEN                   # open 14 mm a side: clear of the neighbouring sticks
RELEASE = 0.25 * FINGER_OPEN


def demo():
    """The right arm picks up the 7's top stick (fingers across it), turns it a quarter turn on the way and lays it
    in the 6's upper right."""
    (px, py), pyaw = SLOTS[MOVE[0]]
    (qx, qy), qyaw = SLOTS[MOVE[1]]
    # the fingers close across the stick: at grip yaw 0 they close along y, so a stick along x is held at yaw 0
    g0, g1 = pyaw, qyaw
    a, c = APPROACH, FINGER_CLOSED
    return {"right": [
        Waypoint(2.0, (px, py, TRAVEL_Z), a, "above the 7's top stick", g0),
        Waypoint(1.5, (px, py, GRIP_Z), a, "down", g0),
        Waypoint(0.8, (px, py, GRIP_Z), c, "grip", g0),
        Waypoint(1.0, (px, py, TRAVEL_Z), c, "lift", g0),
        Waypoint(2.0, (qx, qy, TRAVEL_Z), c, "over the 6, turning", g1),
        Waypoint(1.5, (qx, qy, PLACE_Z), c, "down", g1),
        Waypoint(0.5, (qx, qy, PLACE_Z), RELEASE, "let go", g1),
        Waypoint(1.0, (qx, qy, TRAVEL_Z), RELEASE, "up", g1),
    ]}
