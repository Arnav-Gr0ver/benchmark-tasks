"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_drill_bit_swap.py -q
The chuck's physics (jaws opening and closing, the bit held under the 50 N pull) is checked by check_physics.py
(needs pybullet)."""

from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
task_file = pytest.importorskip("yam_arena.task_file")
import chuck as C  # noqa: E402
import shapekit  # noqa: E402
import task  # noqa: E402
from shapekit import BodyState, MechState  # noqa: E402

STROKE = math.radians(task.STROKE_DEG)


# -- the chuck ------------------------------------------------------------------------------------------------------
def test_jaws_close_on_the_old_bit_and_open_wide_enough_for_the_new_one_inside_the_bore():
    assert math.isclose(C.R0, C.BITS["old_bit"][0] / 2)                      # the joints' zero: on the 4 mm bit
    assert C.R_MAX > C.BITS["new_bit"][0] / 2 + 0.002                        # open: 2 mm clear of the 6 mm bit
    assert C.R_MAX < C.BORE_R                                                # the lands stay outside the open jaws
    assert C.R_MIN < C.BITS["old_bit"][0] / 2
    assert C.DEPTH_OK[0] <= C.NOMINAL_DEPTH <= C.DEPTH_OK[1] <= C.MAX_DEPTH + 0.001
    for name in C.BITS:
        d, length, flute = C.BITS[name][:3]
        assert length - flute >= C.DEPTH_OK[1]                               # the jaws only ever grip smooth shank


def _turn(thread, radians, radii):
    """Turn the sleeve by ``radians`` (> 0 opens) in 1 degree control steps."""
    angle = thread.angle or 0.0
    out = thread.radius
    for _ in range(int(abs(math.degrees(radians)))):
        angle -= C.CLOSE_SIGN * math.copysign(math.radians(1.0), radians)
        out = thread.step(angle, radii(out))
    return out


def test_thread_opens_closes_and_only_winds_so_far_past_a_stalled_jaw():
    th = C.Thread()
    th.step(0.0, [C.R0] * 3)
    held = [C.R0] * 3
    assert math.isclose(th.clamp_force(held), C.CLAMP_N)                    # prepared: tight on the old bit
    r = _turn(th, STROKE, lambda r: [max(r, C.R0)] * 3)                      # one opening stroke
    assert math.isclose(r, C.R0 - C.WIND + C.LEAD * task.STROKE_DEG / 360, rel_tol=0.02)
    assert th.clamp_force(held) == 0.0                                       # loose
    for _ in range(3):
        r = _turn(th, STROKE, lambda r: [r] * 3)
    assert math.isclose(r, C.R_MAX)                                          # the end stop
    assert r - C.BITS["new_bit"][0] / 2 > 0.0015
    # closing round the new bit: the jaws stop on it and the thread winds only WIND further, however far it turns
    bit_r = C.BITS["new_bit"][0] / 2
    for _ in range(4):
        r = _turn(th, -STROKE, lambda r: [max(r, bit_r)] * 3)
    assert math.isclose(r, bit_r - C.WIND)
    assert math.isclose(th.clamp_force([bit_r] * 3), C.CLAMP_N)
    # a jaw pushed outward never loosens the thread; nothing but the sleeve moves it
    assert th.step(th.angle, [bit_r + 0.001] * 3) == r


def test_two_closing_strokes_leave_the_new_bit_loose_and_four_clamp_it():
    th = C.Thread()
    th.step(0.0, [C.R0] * 3)
    for _ in range(4):
        _turn(th, STROKE, lambda r: [max(r, C.R0)] * 3)
    bit_r = C.BITS["new_bit"][0] / 2
    _turn(th, -2 * STROKE, lambda r: [max(r, bit_r)] * 3)
    assert th.clamp_force([bit_r] * 3) == 0.0
    _turn(th, -2 * STROKE, lambda r: [max(r, bit_r)] * 3)
    assert th.clamp_force([bit_r] * 3) >= task.CLAMP_NEED


def test_insertion_measures():
    centre = C.bit_in_chuck("new_bit")
    x = np.array([1.0, 0.0, 0.0])
    depth, tilt, off = C.insertion("new_bit", centre, x)
    assert math.isclose(depth, C.NOMINAL_DEPTH) and tilt < 1e-6 and off < 1e-9
    assert C.inserted("new_bit", centre, x) and C.seated("new_bit", centre, x)
    assert not C.removed("new_bit", centre, x)
    shallow = C.bit_in_chuck("new_bit", 0.010)
    assert not C.inserted("new_bit", shallow, x) and not C.removed("new_bit", shallow, x)
    tilted = np.array([math.cos(math.radians(8)), 0.0, math.sin(math.radians(8))])   # pivoted at the jaws' middle
    mid = np.array([(C.JAW_X[0] + C.JAW_X[1]) / 2, 0.0, 0.0])
    s = C.BITS["new_bit"][1] / 2 + (C.JAW_FRONT - C.NOMINAL_DEPTH - mid[0]) / tilted[0]
    leaning = mid + tilted * s
    assert C.inserted("new_bit", leaning, tilted) and not C.seated("new_bit", leaning, tilted)
    assert not C.inserted("new_bit", centre + np.array([0.0, 0.004, 0.0]), x)     # beside the chuck: not in it
    assert C.removed("new_bit", C.bit_in_chuck("new_bit", -0.003), x)              # 3 mm in front of the jaws
    assert C.removed("old_bit", np.array([0.05, 0.08, 0.0]), np.array([0.0, 1.0, 0.0]))


def test_retention_pull_profile():
    assert C.pull_at(0.0) == 0.0 and math.isclose(C.pull_at(0.25), 25.0)
    assert C.pull_at(0.5) == C.PULL_N and C.pull_at(1.49) == C.PULL_N
    assert math.isclose(C.pull_at(1.75), 25.0) and C.pull_at(2.0) == 0.0 and C.pull_at(2.9) == 0.0
    assert math.isclose(C.PULL_TOTAL_S, 3.0)
    # the spec's 100 steps after release: 2 s still, then the 3 s test
    assert round((task.STILL_S + C.PULL_TOTAL_S) / task.CONTROL_DT) == 100


# -- the scene ------------------------------------------------------------------------------------------------------
def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert {"drill", "old_bit", "new_bit"} <= set(t.free_objects)
    assert "rack" not in t.free_objects and "tray" not in t.free_objects


def test_new_bit_stands_in_the_rack_and_the_old_bit_fits_the_tray():
    middle, quat = task.bit_in_rack()
    axis = BodyState(pos=middle, quat_xyzw=np.array(quat), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=middle,
                     high=middle).rotation()[:, 0]
    assert np.allclose(axis, task.RACK_UP)                                    # shank down, up the socket
    shank_end = middle - axis * C.BITS["new_bit"][1] / 2
    assert np.allclose(shank_end, task.RACK_AT + task.RACK_ROT @ np.array([task.SOCKET / 2 - 0.003, 0.0, 0.0]))
    assert task.SOCKET > C.BITS["new_bit"][0] + 0.001
    assert C.BITS["old_bit"][1] + 0.02 < task.TRAY_IN[0] and C.BITS["old_bit"][0] < task.TRAY_IN[1]
    # the rack and tray are clear of the drill's chuck end, and the tray clear of the rack
    assert task.RACK_AT[1] - 0.03 > 0.04 and task.TRAY_XY[1] - task.TRAY_IN[1] / 2 - task.TRAY_WALL > 0.04
    assert abs(task.TRAY_XY[0] - task.RACK_AT[0]) > 0.05 or abs(task.TRAY_XY[1] - task.RACK_AT[1]) > 0.05


def test_old_bit_starts_seated_in_the_jaws():
    centre = np.asarray(task.OLD_BIT.pos) - task.DRILL_AT
    assert C.seated("old_bit", centre, np.array([1.0, 0.0, 0.0]))


# -- the watcher and the score --------------------------------------------------------------------------------------
def _body(pos, rot=(0.0, 0.0, 0.0, 1.0)):
    pos = np.asarray(pos, float)
    return BodyState(pos=pos, quat_xyzw=np.asarray(rot, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos,
                     high=pos)


def _drill(jaws=0.0):
    pos = np.asarray(task.DRILL_AT, float)
    return MechState(pos=pos, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=pos, high=pos, joints={"sleeve": 0.0, **{j: jaws for j in task.JAWS}})


IN_TRAY = (task.TRAY_XY[0], task.TRAY_XY[1], task.TOP + task.TRAY_FLOOR + C.BITS["old_bit"][0] / 2)
ON_TABLE = (0.40, -0.10, task.TOP + C.BITS["old_bit"][0] / 2)
NEW_IN = task.world(C.bit_in_chuck("new_bit"))
FAR = np.array([[0.0, 0.40, 0.40]])                    # fingers well away from everything


@pytest.fixture()
def robot(monkeypatch):
    """Stand-ins for the simulator calls: where the fingers are, and a log of the forces and joint targets."""
    log = {"fingers": FAR, "wrench": [], "targets": [], "wrench_fails": False}

    def set_wrench(env, name, force, body=None):
        if log["wrench_fails"]:
            raise RuntimeError("no wrench composer")
        log["wrench"].append((name, float(np.linalg.norm(force))))

    monkeypatch.setattr(shapekit, "set_wrench", set_wrench)
    monkeypatch.setattr(shapekit, "clear_wrench", lambda env, name: None)
    monkeypatch.setattr(shapekit, "set_joint_targets", lambda env, name, t: log["targets"].append(t))
    monkeypatch.setattr(shapekit, "robot_fingers", lambda env: {"left": log["fingers"]})
    task._fresh()
    return log


def _objs(old=IN_TRAY, new=NEW_IN, jaws=0.0):
    return {"drill": _drill(jaws), "old_bit": _body(old), "new_bit": _body(new)}


def _clamp_on_new_bit():
    """The thread wound tight on the 6 mm bit (as after the demo's tightening strokes)."""
    task._S["thread"].radius = C.BITS["new_bit"][0] / 2 - C.WIND
    return C.BITS["new_bit"][0] / 2 - C.R0              # the jaws' joint position on the bit


def _run(objs, seconds):
    for _ in range(round(seconds / task.CONTROL_DT)):
        task._watch(None, objs)


def test_untouched_start_scores_zero(robot):
    old = task.world(C.bit_in_chuck("old_bit"))
    objs = {"drill": _drill(), "old_bit": _body(old), "new_bit": _body(task.NEW_BIT.pos, task.NEW_BIT.quat_xyzw)}
    _run(objs, 6.0)
    assert task.score(objs) == 0.0


def test_tiers(robot):
    rack = _body(task.NEW_BIT.pos, task.NEW_BIT.quat_xyzw)
    assert task.score({**_objs(), "new_bit": rack}) == 0.25                 # old bit out, new one still in the rack
    robot["fingers"] = np.array([NEW_IN])                                   # held in the chuck
    objs = _objs(jaws=_clamp_on_new_bit())
    _run(objs, 6.0)
    assert task.score(objs) == 0.5 and task._S["passed"] is None             # never tested while held
    robot["fingers"] = FAR
    _run(objs, task.STILL_S - 0.1)
    assert task._S["pull"] is None and task.score(objs) == 0.5               # let go, not yet still for 2 s
    _run(objs, 0.2)
    assert task._S["pull"] is not None                                       # the test has started
    _run(objs, C.PULL_TOTAL_S)
    assert task._S["passed"] is True
    assert max(f for n, f in robot["wrench"] if n == "new_bit") == pytest.approx(C.PULL_N)
    assert {n for n, _ in robot["wrench"]} == {"new_bit", "drill"}          # the drill is held, never the bit
    assert task.score(objs) == 1.0 and task.success(objs)
    assert task.score({**objs, "old_bit": _body(ON_TABLE)}) == 0.75          # old bit not in the tray
    up = np.array([0.0, 0.0, 0.02])
    lifted = MechState(**{**objs["drill"].__dict__, "pos": objs["drill"].pos + up})
    lifted_bit = _body(np.asarray(NEW_IN) + up)
    assert task.score({**objs, "drill": lifted, "new_bit": lifted_bit}) == 0.75   # the drill not on the table


def test_a_loose_bit_fails_the_test_and_is_capped_at_half(robot):
    objs = _objs(jaws=0.0005)                                                # jaws not wound onto it
    task._S["thread"].radius = C.R_MAX

    def slipping(objs, seconds):
        for k in range(round(seconds / task.CONTROL_DT)):
            if task._S["pull"] is not None and task._S["pull"]["k"] > 5:     # it slides out under the pull
                objs["new_bit"] = _body(np.asarray(NEW_IN) + np.array([0.004, 0.0, 0.0]))
            task._watch(None, objs)

    slipping(objs, task.STILL_S + C.PULL_TOTAL_S + 0.5)
    assert task._S["passed"] is False
    objs["new_bit"] = _body(NEW_IN)
    assert task.score(objs) == 0.5


def test_touching_the_drill_during_the_test_abandons_it(robot):
    objs = _objs(jaws=_clamp_on_new_bit())
    _run(objs, task.STILL_S + 1.0)
    assert task._S["pull"] is not None
    robot["fingers"] = np.array([task.world((0.06, 0.0, 0.0))])             # a hand back on the sleeve
    _run(objs, 0.1)
    assert task._S["pull"] is None and task._S["passed"] is None
    assert task.score(objs) == 0.5


def test_without_a_wrench_the_clamp_force_decides(robot):
    robot["wrench_fails"] = True
    objs = _objs(jaws=_clamp_on_new_bit())
    _run(objs, task.STILL_S + 0.2)
    assert task._S["passed"] is True and "clamp force" in task._S["method"]
    task._fresh()
    task._S["thread"].radius = C.BITS["new_bit"][0] / 2 + 0.0002            # touching, not wound on
    _run(objs, task.STILL_S + 0.2)
    assert task._S["passed"] is False


def test_loosening_after_a_pass_needs_a_new_test(robot):
    objs = _objs(jaws=_clamp_on_new_bit())
    _run(objs, task.STILL_S + C.PULL_TOTAL_S + 0.2)
    assert task.score(objs) == 1.0
    task._S["thread"].radius = C.R_MAX                                       # opened again
    _run(objs, 0.05)
    assert task._S["passed"] is None or task._S["pull"] is not None
    assert task.score(objs) == 0.5


def test_old_bit_left_in_the_chuck_or_anything_off_the_table_scores_zero(robot):
    old = task.world(C.bit_in_chuck("old_bit", 0.008))                       # drawn part way out
    assert task.score(_objs(old=old)) == 0.0
    objs = _objs()
    objs["old_bit"] = _body((0.40, 0.10, task.TOP - 0.2))
    _run(objs, 0.05)
    assert task.score(_objs()) == 0.0                                        # latched


def test_old_bit_in_the_tray():
    turned = (0.0, 0.0, math.sin(math.pi / 12), math.cos(math.pi / 12))          # 30 degrees across it
    across = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert task.in_tray(_body(IN_TRAY)) and task.in_tray(_body(IN_TRAY, turned))
    assert not task.in_tray(_body(IN_TRAY, across))                               # longer than the tray is wide
    assert not task.in_tray(_body(ON_TABLE))
    assert not task.in_tray(_body((IN_TRAY[0] + 0.03, IN_TRAY[1], IN_TRAY[2])))  # an end over the wall
    assert not task.in_tray(_body((IN_TRAY[0], IN_TRAY[1], IN_TRAY[2] + 0.02)))  # held above it


def test_release_distance():
    drill = _drill()
    assert task.distance_to_drill(drill, np.array([task.world((0.06, 0.0, 0.02))])) < 0.005    # on the sleeve
    assert task.distance_to_drill(drill, FAR) > task.CLEAR
    bit = _body(NEW_IN)
    assert task.distance_to_bit("new_bit", bit, np.array([NEW_IN])) < 0
    assert task.distance_to_bit("new_bit", bit, np.array([np.asarray(NEW_IN) + [0.0, 0.0, 0.04]])) > task.CLEAR


# -- the demo -------------------------------------------------------------------------------------------------------
def test_demo_fits_the_run_and_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    plan = task.demo()
    durations = {arm: sum(w.duration for w in way) for arm, way in plan.items()}
    assert max(durations.values()) / task.CONTROL_DT <= 3600
    for arm, waypoints in plan.items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)


def test_demo_strokes_open_then_close_the_chuck():
    """The right hand's sleeve strokes, replayed through the thread: open fully, then wound tight on the new bit."""
    right = task.demo()["right"]
    rolls = []
    for w in right:
        if w.label.endswith("stroke"):
            r = task.ROTATIONS[int(round(w.yaw - task.TAG))]
            rolls.append((w.label, math.atan2(r[2, 1], r[1, 1])))
    opening = [a for label, a in rolls if label.startswith(("loosening", "opening"))]
    closing = [a for label, a in rolls if label.startswith(("closing", "tightening"))]
    assert len(opening) == 4 and len(closing) == 4
    th = C.Thread()
    th.step(0.0, [C.R0] * 3)
    for _ in opening:
        _turn(th, STROKE, lambda r: [max(r, C.R0)] * 3)
    assert math.isclose(th.radius, C.R_MAX)
    bit_r = C.BITS["new_bit"][0] / 2
    for _ in closing:
        _turn(th, -STROKE, lambda r: [max(r, bit_r)] * 3)
    assert th.clamp_force([bit_r] * 3) >= task.CLAMP_NEED
