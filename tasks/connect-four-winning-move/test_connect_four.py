"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_connect_four.py -q"""

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


def _board():
    b = {c: "R" for c in task.PLACED_RED}
    b.update({c: "B" for c in task.PLACED_BLUE})
    return b


def _four(b, p):
    for c, r in itertools.product(range(task.COLS), range(task.ROWS)):
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            if all(b.get((c + k * dc, r + k * dr)) == p for k in range(4)):
                return True
    return False


def _land(b, c):
    return sum((c, r) in b for r in range(task.ROWS))


def _wins(b, p):
    out = set()
    for c in range(task.COLS):
        r = _land(b, c)
        if r < task.ROWS and _four({**b, (c, r): p}, p):
            out.add((c, r))
    return out


def test_the_position_is_legal_with_one_win_and_one_threat():
    b = _board()
    assert len(task.PLACED_RED) == len(task.PLACED_BLUE)            # red, who moves first, is to move
    assert all(r == 0 or (c, r - 1) in b for c, r in b)            # no disc floats
    assert not _four(b, "R") and not _four(b, "B")
    assert _wins(b, "R") == {task.WIN}
    assert _wins(b, "B") == {task.BLOCK}


def _state(pos):
    pos = np.asarray(pos, float)
    return task_file.ObjectState(pos=pos, quat_xyzw=np.array([0.0, 0.7071, 0.0, 0.7071]), lin_vel=np.zeros(3),
                                 ang_vel=np.zeros(3), low=pos - task.R, high=pos + task.R)


def test_scores():
    start = {"red_spare": _state(task.SPARE_RED.pos), "blue_spare": _state(task.SPARE_BLUE.pos)}
    assert task.score(start) == 0.0
    win = {**start, "red_spare": _state((task.CX, task.col_y(task.WIN[0]), task.cell_z(task.WIN[1])))}
    assert task.score(win) == 1.0 and task.success(win)
    block = {**start, "red_spare": _state((task.CX, task.col_y(task.BLOCK[0]), task.cell_z(task.BLOCK[1])))}
    assert task.score(block) == 0.25
    both = {**win, "blue_spare": _state((task.CX, task.col_y(task.BLOCK[0]), task.cell_z(task.BLOCK[1])))}
    assert task.score(both) == 0.0
    top = {**start, "red_spare": _state((task.CX, task.col_y(task.WIN[0]), task.FRAME_TOP + 0.02))}  # on top, not in
    assert task.score(top) == 0.0


def test_discs_fit_the_columns_and_cannot_leave_through_the_front():
    assert task.PITCH - task.DIVIDER > task.D + 0.0015
    assert task.GAP >= task.T + 0.004 - 1e-9
    assert task.PITCH - task.POST < task.D


def test_ridged_discs_have_flats_facing_the_fingers_and_the_floor():
    shapekit = pytest.importorskip("shapekit")
    pts = np.array(shapekit.ridged_points(task.SPARE_RED))
    assert len(pts) == 2 * task.RIDGES
    assert np.allclose(np.abs(pts[:, 0]), task.T / 2)                 # faces normal to x, 7 mm apart
    for axis in (1, 2):                                               # across (y, the pinch) and down (z, the stand)
        extreme = np.abs(pts[:, axis]).max()
        assert math.isclose(extreme, task.R, rel_tol=1e-9)            # a flat, R from the centre, not a corner
        assert (np.isclose(np.abs(pts[:, axis]), extreme)).sum() == 8  # two corners per flat, on both faces, both sides


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert set(t.free_objects) == {"red_spare", "blue_spare"}


def test_the_tilted_grip_tag_leaves_ordinary_yaws_alone():
    demo = pytest.importorskip("yam_arena.demo")
    plain = demo.grip_rotation.plain
    assert np.allclose(demo.grip_rotation(0.3), plain(0.3))
    tilted = demo.grip_rotation(task.GRIP)
    assert np.isclose(tilted[:, 2] @ np.array([0.0, 0.0, -1.0]), math.cos(task.TILT))   # approach 50 deg off vertical
    assert tilted[0, 2] > 0                                                              # leaning forward
    assert np.allclose(abs(tilted[:, 0] @ np.array([0.0, 1.0, 0.0])), 1.0)               # fingers close along y


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
    assert math.isclose(task.col_y(task.WIN[0]), -task.PITCH)
