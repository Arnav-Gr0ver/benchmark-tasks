"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_pick_up_sticks.py -q"""

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
import task  # noqa: E402
from shapekit import BodyState  # noqa: E402


def _footprint(name):
    x, y, _, yaw = task.START[name]
    along = abs(math.cos(yaw)) > 0.5
    hx, hy = (task.LENGTH / 2, task.SIDE / 2) if along else (task.SIDE / 2, task.LENGTH / 2)
    return x - hx, x + hx, y - hy, y + hy


def _overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2])


def test_the_blue_stick_rests_on_the_red_one_and_traps_it():
    assert _overlap(_footprint("blue"), _footprint("red"))
    assert math.isclose(task.START["blue"][2] - task.SIDE / 2, task.START["red"][2] + task.SIDE / 2)  # sits on it
    for name in ("yellow", "black"):                                  # and on the yellow and black ones
        assert _overlap(_footprint("blue"), _footprint(name))
    assert not _overlap(_footprint("blue"), _footprint("green"))


def test_the_protected_sticks_hem_the_red_one_in():
    rx0, rx1, ry0, ry1 = _footprint("red")
    gx0 = _footprint("green")[0]
    assert 0.005 < gx0 - rx1 < 0.02                                   # the green one just past its far end
    assert _footprint("yellow")[2] - ry1 == pytest.approx(task.GAP - task.SIDE)   # 22 mm either side
    assert ry0 - _footprint("black")[3] == pytest.approx(task.GAP - task.SIDE)
    for a, b in itertools.combinations(task.START, 2):               # nothing else overlaps on the table
        if "blue" not in (a, b):
            assert not _overlap(_footprint(a), _footprint(b)), (a, b)


def test_the_fingers_fit_between_the_sticks():
    finger = 0.008                                                    # finger thickness, generous
    reach = -task.APPROACH + finger                                   # from the red stick's centre line, open
    assert -task.APPROACH > task.SIDE / 2 + 0.006                     # 6 mm or more clear of the red stick
    assert task.GAP - task.SIDE / 2 - reach > 0.005                   # and of the yellow and black ones
    blue_tips = task._grip_z(task.START["blue"][2]) - task.PAD_TIP
    assert blue_tips - (task.TOP + task.SIDE) > 0.0015                # over the red stick while gripping the blue


def _state(name, dx=0.0, dy=0.0, dz=0.0, dyaw_deg=0.0, at=None):
    x, y, z, yaw = task.START[name]
    if at is not None:
        x, y, z = at
    yaw += math.radians(dyaw_deg)
    pos = np.array([x + dx, y + dy, task.TOP + z + dz])
    return BodyState(pos=pos, quat_xyzw=np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
                     lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos, high=pos)


def _start():
    return {n: _state(n) for n in task.START}


def test_scores():
    task._clear()
    objs = _start()
    assert task.score(objs) == 0.0
    mat_z = task.MAT_T + task.SIDE / 2
    blue_out = {**objs, "blue": _state("blue", at=(*task.BLUE_TO, mat_z))}
    assert task.score(blue_out) == 0.5
    done = {**blue_out, "red": _state("red", at=(*task.RED_TO, mat_z))}
    assert task.score(done) == 1.0 and task.success(done)
    held = {**blue_out, "red": _state("red", at=(*task.RED_TO, mat_z + 0.05))}      # still held over the mat
    assert task.score(held) == 0.5
    nudged = {**done, "yellow": _state("yellow", dx=0.002, dyaw_deg=2.0)}           # within 3 mm, 3 degrees
    assert task.score(nudged) == 1.0

    task._clear()
    task.score(_start())
    bumped = {**_start(), "black": _state("black", dy=-0.004)}                       # 4 mm: latched
    assert task.score(bumped) == 0.0
    assert task.score(done) == 0.0                                                   # even if put back after
    task._clear()
    task.score(_start())
    assert task.score({**_start(), "green": _state("green", dyaw_deg=4.0)}) == 0.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert set(t.free_objects) == set(task.START)


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
        assert p.duration < 30.0
