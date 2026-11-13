"""On the station table, between the arms: a compact cordless drill lying on its side (battery removed), its chuck
pointing away from the robot and its handle to the robot's left, with a 4 mm drill bit tight in its keyless chuck.
To the left of the chuck, a 6 mm bit (gold) stands shank-down in a rack, leaning away from the robot; beside it, an
empty shallow tray. The drill is loose on the table. Turning the chuck's sleeve opens and closes its three jaws.

Run it:  bash run.sh            (scripted demo)      bash run.sh dream      (Dream-1)
         bash run.sh pretest    (the retention test on the 4 mm bit as prepared, then the demo)
The drill's geometry and the chuck's thread are in chuck.py.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yam_arena.demo as _yam_demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

import chuck as C  # noqa: E402
from shapekit import Compound, Link, Mechanism  # noqa: E402
from stepkit import on_reset, on_step  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]
CONTROL_DT = 0.05               # run.sh and the README run at 20 Hz

INSTRUCTION = ("Replace the installed drill bit with the larger bit from the rack. Leave the old bit in the tray and "
               "the drill on the table with the new bit securely held in its chuck.")

# -- layout (world frame: x forward from the arms, y left, z up; metres) --------------------------------------------
DRILL_XY = (0.19, 0.0)
DRILL_AT = np.array([*DRILL_XY, TOP + C.AXIS_H])      # the drill frame's origin; its axes are the world's


def world(p) -> tuple[float, float, float]:
    """A point of the drill's frame (as the drill starts) in the world."""
    return tuple(float(v) for v in DRILL_AT + np.asarray(p, float))


TEAL, DARK, STEEL, WOOD = (0.05, 0.50, 0.55), (0.12, 0.12, 0.13), (0.62, 0.63, 0.66), (0.70, 0.55, 0.36)


def _drill() -> Mechanism:
    (hc, hr, hl), (bc, bs), (fc, fs) = C.HOUSING, C.HANDLE, C.FOOT
    parts = [(hc, ("cyl", hr, hl, "X")), (bc, bs), (fc, fs)]
    parts += [(centre, size, (angle, 0.0, 0.0)) for centre, size, angle in C.lands()]
    jaw_x = (C.JAW_X[0] + C.JAW_X[1]) / 2
    links = [Link("sleeve", "turn", origin=(sum(C.SLEEVE_X) / 2, 0.0, 0.0), mass=C.SLEEVE_MASS,
                  parts=(((0.0, 0.0, 0.0), ("cyl", C.SLEEVE_R, C.SLEEVE_X[1] - C.SLEEVE_X[0], "X")),),
                  damping=C.SLEEVE_DAMPING, friction=0.9, color=DARK)]
    for k, angle in enumerate(C.JAW_ANGLES):
        origin = np.array([jaw_x, 0.0, 0.0]) + C.jaw_dir(k) * (C.R0 + C.JAW_T / 2)
        links.append(Link(f"jaw_{k}", "slide", origin=tuple(origin), about_x_deg=angle, mass=C.JAW_MASS,
                          parts=(((0.0, 0.0, 0.0), (C.JAW_X[1] - C.JAW_X[0], C.JAW_T, C.JAW_W), (angle, 0.0, 0.0)),),
                          limits=(C.R_MIN - C.R0, C.R_MAX - C.R0), stiffness=C.JAW_K, damping=C.JAW_D,
                          friction=C.JAW_FRICTION, friction_min=True, color=STEEL))
    return Mechanism("drill", pos=tuple(DRILL_AT), size=(0.19, 0.21, 0.07), mass=C.BODY_MASS, com=C.BODY_COM,
                     color=TEAL, parts=tuple(parts), links=tuple(links), friction=0.6, slick=(3, 4, 5),
                     slick_friction=C.JAW_FRICTION, part_colors=(TEAL, DARK, DARK, STEEL, STEEL, STEEL))


def _bit(name, pos, quat_xyzw=None) -> Compound:
    d, length, _, mass, colour = C.BITS[name]
    return Compound(name, pos=tuple(float(v) for v in pos), quat_xyzw=quat_xyzw, size=(length, d, d), mass=mass,
                    color=colour, boxes=C.bit_parts(name), friction=C.BIT_FRICTION)


# The rack: a square socket (8 mm inside, 26 mm deep) on a block, leaning 45 degrees away from the robot. The 6 mm
# bit stands in it shank-down, lying against the socket's far wall.
RACK_LEAN = math.radians(45.0)
RACK_AT = np.array([0.290, 0.175, TOP + 0.012])       # the middle of the socket's floor
SOCKET, SOCKET_WALL, SOCKET_DEPTH = 0.008, 0.003, 0.026


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


RACK_ROT = _ry(RACK_LEAN)                              # the socket's axes in the world: its z runs up the bit
RACK_UP = RACK_ROT[:, 2]                               # (sin 45, 0, cos 45)


def _rack() -> Compound:
    off, outer = SOCKET / 2 + SOCKET_WALL / 2, SOCKET + 2 * SOCKET_WALL
    local = [((off, 0.0, SOCKET_DEPTH / 2), (SOCKET_WALL, outer, SOCKET_DEPTH)),
             ((-off, 0.0, SOCKET_DEPTH / 2), (SOCKET_WALL, outer, SOCKET_DEPTH)),
             ((0.0, off, SOCKET_DEPTH / 2), (SOCKET, SOCKET_WALL, SOCKET_DEPTH)),
             ((0.0, -off, SOCKET_DEPTH / 2), (SOCKET, SOCKET_WALL, SOCKET_DEPTH)),
             ((0.0, 0.0, -SOCKET_WALL / 2), (outer, outer, SOCKET_WALL))]
    lean = math.degrees(RACK_LEAN)
    boxes = [(tuple(RACK_ROT @ np.array(c)), size, (0.0, lean, 0.0)) for c, size in local]
    h = RACK_AT[2] - TOP
    boxes.append(((0.0, 0.0, -h / 2), (0.030, 0.030, h)))                    # the block under the socket
    boxes.append(((0.0, 0.0, -h + 0.002), (0.060, 0.045, 0.004)))            # its foot
    return Compound("rack", pos=tuple(RACK_AT), size=(0.06, 0.045, 0.05), color=WOOD, boxes=tuple(boxes),
                    friction=0.6)


def bit_in_rack(name: str = "new_bit"):
    """(middle, quaternion xyzw) of a bit standing in the rack: against the socket's far wall, on its floor."""
    d, length = C.BITS[name][:2]
    middle = RACK_AT + RACK_ROT @ np.array([SOCKET / 2 - d / 2, 0.0, length / 2])
    turn = RACK_LEAN - math.pi / 2                    # about y: takes the bit's x (shank to tip) to RACK_UP
    return middle, (0.0, math.sin(turn / 2), 0.0, math.cos(turn / 2))


# The tray for the old bit: 110 x 50 mm inside, 7 mm walls round a 3 mm floor.
TRAY_XY, TRAY_IN, TRAY_FLOOR, TRAY_WALL, TRAY_H = (0.370, 0.100), (0.110, 0.050), 0.003, 0.005, 0.010


def _tray() -> Compound:
    (lx, ly), w, h = TRAY_IN, TRAY_WALL, TRAY_H
    boxes = [((0.0, 0.0, TRAY_FLOOR / 2), (lx + 2 * w, ly + 2 * w, TRAY_FLOOR)),
             ((lx / 2 + w / 2, 0.0, h / 2), (w, ly + 2 * w, h)), ((-lx / 2 - w / 2, 0.0, h / 2), (w, ly + 2 * w, h)),
             ((0.0, ly / 2 + w / 2, h / 2), (lx, w, h)), ((0.0, -ly / 2 - w / 2, h / 2), (lx, w, h))]
    return Compound("tray", pos=(*TRAY_XY, TOP), size=(lx + 2 * w, ly + 2 * w, h), color=(0.20, 0.30, 0.55),
                    boxes=tuple(boxes), friction=0.6)


_new_at, _new_quat = bit_in_rack()
DRILL = _drill()
OLD_BIT = _bit("old_bit", world(C.bit_in_chuck("old_bit")))
NEW_BIT = _bit("new_bit", _new_at + RACK_UP * 0.0005, _new_quat)
OBJECTS = [DRILL, OLD_BIT, NEW_BIT, _rack(), _tray()]
JAWS = [f"jaw_{k}" for k in range(3)]

# -- the chuck's thread, the release check and the retention test: run after every control step ----------------------
CLEAR = 0.030                   # fingers this far from the drill and the bit count as let go
STILL_S, STILL_MM, STILL_DEG = 2.0, 0.002, 2.0
CLAMP_NEED = 1.5 * C.PULL_N / (3 * C.JAW_FRICTION)     # per jaw, if the pull cannot be applied (see _pull)
PRETEST = os.environ.get("DRILL_PRETEST") == "1"

_S: dict = {}


def _fresh() -> None:
    _S.clear()
    _S.update(thread=C.Thread(), steps=0, still=0, ref=None, pull=None, passed=None, method=None, fingers_ok=True,
              off_table=False, pretest=None, clear={"drill": True, "new_bit": True, "old_bit": True})


_fresh()


@on_reset
def _clear(environment=None) -> None:
    _fresh()
    if environment is not None:
        from shapekit import set_joint_targets

        set_joint_targets(environment, "drill", {j: _S["thread"].radius - C.R0 for j in JAWS})


def jaw_radii(drill) -> list[float]:
    return [C.R0 + drill.joints[j] for j in JAWS]


def in_drill(drill, state):
    """An object's middle and its x axis in the drill's frame."""
    rot = drill.rotation()
    return rot.T @ (np.asarray(state.pos, float) - drill.pos), rot.T @ state.rotation()[:, 0]


_DRILL_BOXES = [((-0.020, 0.0, 0.0), (0.110, 0.070, 0.070)), C.HANDLE, C.FOOT,
                (((C.SLEEVE_X[0] + C.JAW_X[1]) / 2, 0.0, 0.0), (C.JAW_X[1] - C.SLEEVE_X[0], 0.036, 0.036))]


def distance_to_drill(drill, points) -> float:
    local = (np.asarray(points, float) - drill.pos) @ drill.rotation()
    best = math.inf
    for centre, size in _DRILL_BOXES:
        d = np.maximum(np.abs(local - np.asarray(centre)) - np.asarray(size) / 2, 0.0)
        best = min(best, float(np.linalg.norm(d, axis=1).min()))
    return best


def distance_to_bit(name, state, points) -> float:
    d, length = C.BITS[name][:2]
    axis = state.rotation()[:, 0]
    rel = np.asarray(points, float) - np.asarray(state.pos, float)
    along = np.clip(rel @ axis, -length / 2, length / 2)
    return float(np.linalg.norm(rel - np.outer(along, axis), axis=1).min()) - d / 2


def _turn_deg(r0, r1) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r1 @ r0.T) - 1) / 2))))


def _pull(environment, drill, bit_name, bit, k: int) -> float:
    """Step k of the retention test: the pull on the bit along the chuck axis, and the same force the other way on
    the drill (held by its chuck sleeve, on the axis, never by the bit). Returns the force."""
    from shapekit import set_wrench

    force = C.pull_at(k * CONTROL_DT)
    axis = drill.rotation()[:, 0]
    set_wrench(environment, bit_name, axis * force)
    set_wrench(environment, "drill", -axis * force, body="sleeve")
    return force


def _end_pull(environment, bit_name) -> None:
    from shapekit import clear_wrench

    for name in (bit_name, "drill"):
        try:
            clear_wrench(environment, name)
        except Exception:  # noqa: BLE001
            pass


def _retention(environment, objects, name: str) -> bool | None:
    """Runs the retention test on a bit, one control step per call. None while it runs; then whether it passed."""
    drill, bit = objects["drill"], objects[name]
    centre, axis = in_drill(drill, bit)
    p = _S["pull"]
    if p is None:
        p = _S["pull"] = {"k": 0, "x0": float(centre[0]), "slip": 0.0, "name": name}
    try:
        _pull(environment, drill, name, bit, p["k"])
    except Exception as err:  # noqa: BLE001  (no way to apply a force in this build: judge from the jaws' clamp)
        _S["pull"] = None
        _S["method"] = f"clamp force (the pull could not be applied: {type(err).__name__}: {err})"
        force = _S["thread"].clamp_force(jaw_radii(drill))
        return force >= CLAMP_NEED and C.seated(name, centre, axis)
    p["slip"] = max(p["slip"], abs(float(centre[0]) - p["x0"]))
    p["k"] += 1
    if p["k"] * CONTROL_DT < C.PULL_TOTAL_S:
        return None
    _end_pull(environment, name)
    _S["pull"] = None
    _S["method"] = f"{C.PULL_N:.0f} N pull, slipped {1000 * p['slip']:.2f} mm"
    return p["slip"] <= C.SLIP_OK and C.seated(name, centre, axis)


def _watch(environment, objects) -> None:
    from shapekit import robot_fingers, set_joint_targets

    drill = objects["drill"]
    s = _S
    s["steps"] += 1
    radius = s["thread"].step(drill.joints["sleeve"], jaw_radii(drill))
    set_joint_targets(environment, "drill", {j: radius - C.R0 for j in JAWS})
    if any(float(objects[n].pos[2]) < TOP - 0.05 for n in ("drill", "old_bit", "new_bit")):
        s["off_table"] = True

    try:
        fingers = np.concatenate(list(robot_fingers(environment).values()))
        s["clear"] = {"drill": distance_to_drill(drill, fingers) >= CLEAR,
                      "new_bit": distance_to_bit("new_bit", objects["new_bit"], fingers) >= CLEAR,
                      "old_bit": distance_to_bit("old_bit", objects["old_bit"], fingers) >= CLEAR}
    except Exception as err:  # noqa: BLE001
        if s["fingers_ok"]:
            print(f"[task] finger positions unavailable ({type(err).__name__}: {err}); release is not checked",
                  flush=True)
        s["fingers_ok"] = False

    if PRETEST and s["pretest"] is None and s["steps"] >= 20:      # the 4 mm bit, as prepared
        result = _retention(environment, objects, "old_bit")
        if result is not None:
            s["pretest"] = result
            print(f"[task] retention test on the old bit as prepared: {'passed' if result else 'FAILED'} "
                  f"({s['method']})", flush=True)
        return

    centre, axis = in_drill(drill, objects["new_bit"])
    if s["passed"] and s["thread"].clamp_force(jaw_radii(drill)) < 0.2 * C.CLAMP_N:
        s["passed"] = None                                          # loosened again: it must be tested again
    released = s["clear"]["drill"] and s["clear"]["new_bit"]
    if s["pull"] is not None:                                       # the test is running
        if not released:                                            # touched during the test: not valid, start over
            _end_pull(environment, "new_bit")
            s["pull"], s["still"], s["ref"] = None, 0, None
            return
        result = _retention(environment, objects, "new_bit")
        if result is not None:
            s["passed"] = result
            print(f"[task] retention test on the new bit: {'passed' if result else 'failed'} ({s['method']})",
                  flush=True)
        return
    if s["passed"] is not None or not (released and C.inserted("new_bit", centre, axis)):
        s["still"], s["ref"] = 0, None
        return
    poses = [(np.array(objects[n].pos, float), objects[n].rotation()) for n in ("drill", "new_bit")]
    if s["ref"] is None or any(np.linalg.norm(p - p0) > STILL_MM or _turn_deg(r0, r) > STILL_DEG
                               for (p, r), (p0, r0) in zip(poses, s["ref"])):
        s["still"], s["ref"] = 0, poses
        return
    s["still"] += 1
    if s["still"] * CONTROL_DT >= STILL_S:                          # let go and still for 2 s: test it
        s["passed"] = _retention(environment, objects, "new_bit")


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    _watch(environment, object_states(environment, ["drill", "old_bit", "new_bit"]))


# -- scoring --------------------------------------------------------------------------------------------------------
def in_tray(state) -> bool:
    """The old bit lying entirely inside the tray: both its ends over the tray's floor, and low."""
    axis = state.rotation()[:, 0]
    half = C.BITS["old_bit"][1] / 2
    for end in (state.pos + axis * half, state.pos - axis * half):
        if (abs(end[0] - TRAY_XY[0]) > TRAY_IN[0] / 2 or abs(end[1] - TRAY_XY[1]) > TRAY_IN[1] / 2
                or end[2] > TOP + TRAY_H + 0.004):
            return False
    return True


def drill_resting(drill) -> bool:
    """The drill lying on the table, on its side."""
    return abs(float(drill.pos[2]) - (TOP + C.AXIS_H)) <= 0.004 and float(drill.rotation()[2, 2]) >= math.cos(
        math.radians(5.0))


def outcome(objects) -> tuple[float, str]:
    drill = objects["drill"]
    s = _S
    if s["off_table"]:
        return 0.0, "the drill or a bit fell off the table"
    if not C.removed("old_bit", *in_drill(drill, objects["old_bit"])):
        return 0.0, "the old bit is still in the chuck"
    centre, axis = in_drill(drill, objects["new_bit"])
    if not C.inserted("new_bit", centre, axis):
        return 0.25, "the old bit is out; the new bit is not in the chuck"
    depth, tilt, off = C.insertion("new_bit", centre, axis)
    where = f"the new bit is {1000 * depth:.0f} mm into the chuck"
    released = s["clear"]["drill"] and s["clear"]["new_bit"]
    if s["passed"] is None:
        return 0.5, f"{where}; not yet let go and tested"
    if not s["passed"]:
        return 0.5, f"{where}; it failed the retention test ({s['method']})"
    if not (released and C.seated("new_bit", centre, axis)):
        return 0.5, f"{where} and passed the retention test, but is held or no longer seated"
    held = f"{where} and holds ({s['method']})"
    if not (in_tray(objects["old_bit"]) and s["clear"]["old_bit"] and objects["old_bit"].resting()):
        return 0.75, f"{held}; the old bit is not lying in the tray"
    if not drill_resting(drill):
        return 0.75, f"{held}; the drill is not lying on the table"
    return 1.0, f"{held}; the old bit is in the tray"


def success(objects) -> bool:
    """Everything in the instruction, let go: holds 1 s."""
    return outcome(objects)[0] == 1.0


_LAST = {"why": None}


def score(objects) -> float:
    """The highest tier reached at the end. 0.25 the old bit fully out of the chuck. 0.5 also the 6 mm bit inserted:
    its shank 14 to 22.5 mm past the jaws' front, within 15 degrees and 3 mm of the chuck's axis (it may be held).
    0.75 also drill and new bit let go (every finger 30 mm or more from them), still for 2 s (within 2 mm, 2
    degrees), and the retention test passed: a 50 N axial pull on the bit (0.5 s ramp, 1 s hold, 0.5 s unload, 1 s
    watched) with the drill held by its chuck sleeve, the bit slipping 1 mm at most and seated afterwards (within 3
    degrees, 0.6 mm of the axis). 1.0 also the old bit let go and lying entirely inside the tray, and the drill lying
    on the table. 0 if the old bit is still between the jaws or anything fell off the table. A failed retention test
    caps the score at 0.5."""
    value, why = outcome(objects)
    key = (value, why.split(";")[-1])
    if key != _LAST["why"]:
        _LAST["why"] = key
        print(f"[task] {value}: {why}", flush=True)
    return value


# -- scripted demo --------------------------------------------------------------------------------------------------
# Some grips lean: the right hand rolls about the chuck's axis to turn the sleeve, the left leans 45 degrees to take
# the bit out of the rack. yam_arena's scripted waypoints only describe straight-down grips turned about the vertical
# (their `yaw`), so a leaning waypoint passes a tag (TAG + k) as its yaw, and this file teaches
# yam_arena.demo.grip_rotation to read the tag as the k-th orientation in ROTATIONS. Ordinary yaws are unchanged.
TAG = 100.0
ROTATIONS: list[np.ndarray] = getattr(_yam_demo, "_drill_bit_rotations", [])
_yam_demo._drill_bit_rotations = ROTATIONS
_plain_grip_rotation = getattr(_yam_demo.grip_rotation, "plain", _yam_demo.grip_rotation)


def _grip_rotation(yaw: float = 0.0) -> np.ndarray:
    if yaw < TAG / 2:
        return _plain_grip_rotation(yaw)
    return ROTATIONS[int(round(yaw - TAG))]


_grip_rotation.plain = _plain_grip_rotation
_yam_demo.grip_rotation = _grip_rotation


def _tagged(rot: np.ndarray) -> float:
    ROTATIONS.append(rot)
    return TAG + len(ROTATIONS) - 1


PAD_TIP = 0.013
ALONG_Y, ALONG_X = 0.0, -math.pi / 2          # plain grip yaws: the fingers close along y / along x
AXIS_Z = TOP + C.AXIS_H
# the right hand on the sleeve: its tool point on the chuck's axis, rolled about it
SLEEVE_AT = np.array([DRILL_XY[0] + 0.062, 0.0, AXIS_Z])
ROLL_OPEN, ROLL_SHUT = math.radians(40.0), math.radians(-25.0)   # an opening stroke ends at +40, a closing one at -25
STROKE_DEG = 65.0
SLEEVE_GAP = 0.55 * FINGER_OPEN                # 26 mm a side: 8 mm clear of the sleeve
BACK = 0.09                                    # the hand backs this far off along its own axis,
ROLL_AWAY = math.radians(15.0)                 # rolled this much (it cannot reach that far back at -25 degrees)
# the left hand
HANDLE_AT = np.array([DRILL_XY[0] + C.HANDLE[0][0], 0.105, TOP + 0.025 + PAD_TIP])   # tips 25 mm above the table
HANDLE_GAP = 0.60 * FINGER_OPEN
BIT_GAP = 0.30 * FINGER_OPEN                   # 14 mm a side
OLD_GRIP = 0.012                               # the old bit is held this far from its tip
OLD_TIPS_BELOW = 0.011                         # the fingertips this far below the chuck's axis: a loose bit droops
NEW_GRIP = 0.020                               # the new bit is held this far from its tip
NEW_TIPS_PAST = 0.006                          # the fingertips this far past its axis
PRE_INSERT = 0.012                             # the shank's end this far in front of the jaws before going in
JAW_FRONT_X = DRILL_XY[0] + C.JAW_FRONT


class _Arm:
    """One arm's waypoints, with a clock so the two arms can wait for each other."""

    def __init__(self) -> None:
        self.way: list = []
        self.t = 0.0
        self.last = None

    def go(self, seconds, pos, finger, label, rot) -> None:
        yaw = rot if isinstance(rot, float) else _tagged(rot)
        self.last = (tuple(float(v) for v in pos), finger, yaw)
        self.way.append(Waypoint(seconds, self.last[0], finger, label, yaw))
        self.t += seconds

    def wait(self, until, label="wait") -> None:
        if until > self.t + 1e-9:
            pos, finger, yaw = self.last
            self.way.append(Waypoint(until - self.t, pos, finger, label, yaw))
            self.t = until


def _roll(angle: float) -> np.ndarray:
    return _rx(angle) @ _plain_grip_rotation(ALONG_Y)


def _stroke(arm: _Arm, start: float, end: float, n: int, what: str, keep_hold: bool) -> None:
    """n strokes of the sleeve from roll ``start`` to roll ``end``; the hand starts and ends closed on the sleeve at
    ``start`` and (keep_hold) ``end``, or open at ``end``."""
    for k in range(n):
        arm.go(1.8, SLEEVE_AT, FINGER_CLOSED, f"{what} stroke", _roll(end))
        if k < n - 1 or not keep_hold:
            arm.go(0.5, SLEEVE_AT, SLEEVE_GAP, "let go of the sleeve", _roll(end))
        if k < n - 1:
            arm.go(1.3, SLEEVE_AT, SLEEVE_GAP, "back", _roll(start))
            arm.go(0.7, SLEEVE_AT, FINGER_CLOSED, "grip the sleeve", _roll(start))


def _back_off(angle: float) -> np.ndarray:
    """Where the hand's tool point is, backed off the sleeve along its own axis at this roll."""
    return SLEEVE_AT - (_rx(angle) @ np.array([0.0, 0.0, -1.0])) * BACK


_DEMO: dict = {}


def demo():
    """Left hand holds the drill's handle while the right hand turns the sleeve one stroke to loosen the chuck. The
    left hand draws the old bit out and lays it in the tray, then holds the handle again while the right hand opens
    the chuck fully (three more strokes) and keeps hold of the sleeve. The left hand takes the 6 mm bit from the rack,
    puts its shank 19 mm into the chuck and holds it there while the right hand closes the jaws round it (two
    strokes), lets go, and the right hand tightens (two more). Both hands back off."""
    if not _DEMO:  # built once: the waypoints' tags index ROTATIONS
        _DEMO.update(_plan())
    return dict(_DEMO)


def _plan() -> dict:
    left, right = _Arm(), _Arm()
    up = np.array([0.0, 0.0, 1.0])

    # 1. left: hold the handle. right: one opening stroke.
    left.go(2.0, HANDLE_AT + up * 0.06, HANDLE_GAP, "above the handle", ALONG_X)
    left.go(1.5, HANDLE_AT, HANDLE_GAP, "down round the handle", ALONG_X)
    left.go(0.8, HANDLE_AT, FINGER_CLOSED, "hold the handle", ALONG_X)
    right.go(2.5, _back_off(ROLL_AWAY), SLEEVE_GAP, "toward the sleeve", _roll(ROLL_AWAY))
    right.go(1.5, SLEEVE_AT, SLEEVE_GAP, "round the sleeve", _roll(ROLL_SHUT))
    right.wait(left.t + 0.2)
    right.go(0.7, SLEEVE_AT, FINGER_CLOSED, "grip the sleeve", _roll(ROLL_SHUT))
    _stroke(right, ROLL_SHUT, ROLL_OPEN, 1, "loosening", keep_hold=False)
    right.go(1.0, _back_off(ROLL_OPEN), SLEEVE_GAP, "back off", _roll(ROLL_OPEN))

    # 2. left: draw the old bit out and lay it in the tray; then hold the handle again.
    left.wait(right.t - 0.8, "hold the handle")
    left.go(0.5, HANDLE_AT, HANDLE_GAP, "let go of the handle", ALONG_X)
    left.go(1.0, HANDLE_AT + up * 0.07, HANDLE_GAP, "up", ALONG_X)
    tip_x = DRILL_XY[0] + C.bit_in_chuck("old_bit")[0] + C.BITS["old_bit"][1] / 2
    old = np.array([tip_x - OLD_GRIP, 0.0, AXIS_Z - OLD_TIPS_BELOW + PAD_TIP])
    left.go(2.0, old + up * 0.05, BIT_GAP, "above the old bit", ALONG_Y)
    left.go(1.5, old, BIT_GAP, "down round it", ALONG_Y)
    left.go(0.8, old, FINGER_CLOSED, "grip it", ALONG_Y)
    left.go(1.5, old + np.array([0.040, 0.0, 0.0]), FINGER_CLOSED, "draw it out", ALONG_Y)
    left.go(1.0, old + np.array([0.040, 0.0, 0.05]), FINGER_CLOSED, "up", ALONG_Y)
    # in the tray its middle goes to the tray's middle; the hand holds it 25.5 mm toward its tip from there
    lay = np.array([TRAY_XY[0] + C.BITS["old_bit"][1] / 2 - OLD_GRIP, TRAY_XY[1], 0.0])
    lay[2] = TOP + TRAY_FLOOR + 0.001 + PAD_TIP                              # fingertips 1 mm above the tray's floor
    left.go(2.0, lay + up * 0.05, FINGER_CLOSED, "over the tray", ALONG_Y)
    left.go(1.2, lay, FINGER_CLOSED, "down", ALONG_Y)
    left.go(0.5, lay, BIT_GAP, "let go", ALONG_Y)
    left.go(1.0, lay + up * 0.06, BIT_GAP, "up", ALONG_Y)
    left.go(2.0, HANDLE_AT + up * 0.06, HANDLE_GAP, "above the handle", ALONG_X)
    left.go(1.5, HANDLE_AT, HANDLE_GAP, "down round the handle", ALONG_X)
    left.go(0.8, HANDLE_AT, FINGER_CLOSED, "hold the handle", ALONG_X)

    # 3. right: three more opening strokes, ending with the sleeve held at ROLL_OPEN.
    right.go(1.2, _back_off(ROLL_AWAY), SLEEVE_GAP, "wait off the sleeve", _roll(ROLL_AWAY))
    right.wait(left.t - 1.0)
    right.go(1.5, SLEEVE_AT, SLEEVE_GAP, "round the sleeve", _roll(ROLL_SHUT))
    right.go(0.7, SLEEVE_AT, FINGER_CLOSED, "grip the sleeve", _roll(ROLL_SHUT))
    _stroke(right, ROLL_SHUT, ROLL_OPEN, 3, "opening", keep_hold=True)

    # 4. left: take the new bit from the rack, put it in the chuck and hold it there.
    left.wait(right.t + 0.3, "hold the handle")
    left.go(0.5, HANDLE_AT, HANDLE_GAP, "let go of the handle", ALONG_X)
    left.go(1.0, HANDLE_AT + up * 0.07, HANDLE_GAP, "up", ALONG_X)
    middle, _ = bit_in_rack()
    length = C.BITS["new_bit"][1]
    hold_from_shank = length - NEW_GRIP
    lean = _ry(-(math.pi / 2 - RACK_LEAN)) @ _plain_grip_rotation(ALONG_Y)   # the hand's axis square to the bit
    approach = lean @ _plain_grip_rotation(ALONG_Y).T @ np.array([0.0, 0.0, -1.0])
    at_rack = middle + RACK_UP * (hold_from_shank - length / 2) + approach * (NEW_TIPS_PAST - PAD_TIP)
    left.go(2.5, at_rack - approach * 0.06, BIT_GAP, "toward the new bit", lean)
    left.go(1.5, at_rack, BIT_GAP, "round it", lean)
    left.go(0.8, at_rack, FINGER_CLOSED, "grip it", lean)
    left.go(1.2, at_rack + RACK_UP * 0.045, FINGER_CLOSED, "out of the rack", lean)
    z_held = AXIS_Z - NEW_TIPS_PAST + PAD_TIP                               # the bit on the chuck's axis
    before = np.array([JAW_FRONT_X + PRE_INSERT + hold_from_shank, 0.0, z_held])
    left.go(3.0, before + up * 0.05, FINGER_CLOSED, "level, in front of the chuck", ALONG_Y)
    left.go(1.2, before, FINGER_CLOSED, "down to the chuck's axis", ALONG_Y)
    seat = before - np.array([PRE_INSERT + C.NOMINAL_DEPTH, 0.0, 0.0])
    left.go(2.0, seat, FINGER_CLOSED, "into the chuck", ALONG_Y)

    # 5. right: two closing strokes with the bit held; left lets go; right: two more.
    right.wait(left.t + 0.3, "hold the sleeve")
    _stroke(right, ROLL_OPEN, ROLL_SHUT, 2, "closing", keep_hold=False)
    right.go(1.3, SLEEVE_AT, SLEEVE_GAP, "back", _roll(ROLL_OPEN))
    right.go(0.7, SLEEVE_AT, FINGER_CLOSED, "grip the sleeve", _roll(ROLL_OPEN))
    left.wait(right.t - 0.6, "hold the bit")
    left.go(0.6, seat, BIT_GAP, "let go of the bit", ALONG_Y)
    left.go(1.2, seat + up * 0.065, BIT_GAP, "up", ALONG_Y)
    right.wait(left.t)
    _stroke(right, ROLL_OPEN, ROLL_SHUT, 2, "tightening", keep_hold=False)
    right.go(1.5, _back_off(ROLL_AWAY), SLEEVE_GAP, "back off", _roll(ROLL_AWAY))
    end = max(left.t, right.t) + STILL_S + C.PULL_TOTAL_S + SUCCESS_HOLD_S + 2.0
    left.wait(end, "clear")
    right.wait(end, "clear")
    return {"left": left.way, "right": right.way}
