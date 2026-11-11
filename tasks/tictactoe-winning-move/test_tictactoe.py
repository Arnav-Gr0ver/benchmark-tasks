"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_tictactoe.py -q"""

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

LINES = [[(r, c) for c in range(3)] for r in range(3)] + [[(r, c) for r in range(3)] for c in range(3)]
LINES += [[(i, i) for i in range(3)], [(i, 2 - i) for i in range(3)]]


def winning_cells(mine, theirs):
    return {cell for line in LINES for cell in line
            if cell not in mine | theirs and sum(c in mine for c in line) == 2 and not any(c in theirs for c in line)}


def test_the_position_has_exactly_one_win_and_one_threat():
    red = {rc for rc, col in task.PLACED.values() if col == task.RED}
    blue = {rc for rc, col in task.PLACED.values() if col == task.BLUE}
    assert len(red) == len(blue)                       # red, who moves first, is to move
    assert not any(all(c in red for c in line) or all(c in blue for c in line) for line in LINES)
    assert winning_cells(red, blue) == {task.WIN_CELL}
    assert winning_cells(blue, red) == {task.BLOCK_CELL}


def _state(pos, tilt_deg=0.0):
    a = math.radians(tilt_deg) / 2
    pos = np.asarray(pos, float)
    half = task.BLOCK / 2 * (math.cos(math.radians(tilt_deg)) + abs(math.sin(math.radians(tilt_deg))))
    low = pos - [task.BLOCK / 2, half, half]
    return task_file.ObjectState(pos=pos, quat_xyzw=np.array([math.sin(a), 0, 0, math.cos(a)]), lin_vel=np.zeros(3),
                                 ang_vel=np.zeros(3), low=low, high=pos + [task.BLOCK / 2, half, half])


def _start():
    return {o.name: _state(o.pos) for o in task.OBJECTS if o.mass is not None}


def test_scores():
    s = _start()
    assert task.score(s) == 0.0
    s["red_spare"] = _state((*task.cell_xy(*task.WIN_CELL), task.BLOCK_Z))
    assert task.score(s) == 1.0 and task.success(s)
    s["red_spare"] = _state((*task.cell_xy(*task.BLOCK_CELL), task.BLOCK_Z))
    assert task.score(s) == 0.25
    s["red_spare"] = _state((*task.cell_xy(1, 1), task.BLOCK_Z + task.BLOCK))  # stacked on red
    assert task.score(s) == 0.0
    s["red_spare"] = _state((*task.cell_xy(*task.WIN_CELL), task.BLOCK_Z), tilt_deg=40)  # propped on an edge
    assert task.score(s) == 0.0
    s["red_spare"] = _state((*task.cell_xy(*task.WIN_CELL), task.BLOCK_Z), tilt_deg=90)  # flat on another face
    assert task.score(s) == 1.0
    s = _start()
    s["red_spare"] = _state((*task.cell_xy(*task.WIN_CELL), task.BLOCK_Z))
    s["blue_spare"] = _state((*task.cell_xy(*task.BLOCK_CELL), task.BLOCK_Z))
    assert task.score(s) == 0.0
    s = _start()
    s["red_spare"] = _state((*task.cell_xy(*task.WIN_CELL), task.BLOCK_Z))
    s["red_b"] = _state((task.cell_xy(1, 1)[0] + 0.03, 0.0, task.BLOCK_Z))   # knocked aside
    assert task.score(s) == 0.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert t.scene == "station"


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
