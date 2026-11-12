"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_matchstick.py -q"""

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

START = set(task.start_slots())


def test_the_start_reads_as_a_wrong_equation():
    eq = task.read(START)
    assert eq == (9, "-", 6, 7) and not task.true(eq)


def test_exactly_one_move_fixes_it():
    """Every way of taking one stick and laying it in any empty slot; the digits may be read in their other common
    forms too (6 and 9 without tails, 7 with a hook)."""
    fixes = []
    for src in START:
        for dst in set(task.SLOTS) - START:
            eq = task.read(START - {src} | {dst})
            if eq and task.true(eq):
                fixes.append((src, dst, eq))
    assert fixes == [(task.MOVE[0], task.MOVE[1], task.SOLUTION)]


def _state(slot, dx=0.0, dy=0.0, dyaw_deg=0.0, z=None):
    (x, y), yaw = task.SLOTS[slot]
    a = (yaw + math.radians(dyaw_deg)) / 2
    pos = np.array([x + dx, y + dy, z if z is not None else task.MAT_TOP + task.STICK[2] / 2])
    return task_file.ObjectState(pos=pos, quat_xyzw=np.array([0.0, 0.0, math.sin(a), math.cos(a)]),
                                 lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos, high=pos)


def _at_start():
    return {s.name: _state(s.name) for s in task.STICKS}


def test_scores():
    objs = _at_start()
    assert task.score(objs) == 0.0
    fixed = {**objs, task.MOVE[0]: _state(task.MOVE[1], 0.006, -0.004, 12.0)}   # within the slot's tolerance
    assert task.score(fixed) == 1.0 and task.success(fixed)
    near = {**objs, task.MOVE[0]: _state(task.MOVE[1], 0.02)}                     # 2 cm off: not in the slot
    assert task.score(near) == 0.25
    held = {**objs, task.MOVE[0]: _state(task.MOVE[1], z=task.MAT_TOP + 0.03)}    # still held over it
    assert task.score(held) == 0.25
    wrong = {**objs, "d2_g": _state("d3_d", 0.0, 0.0, 0.0, task.MAT_TOP + 0.012 + 0.001)}
    assert task.score(wrong) == 0.0
    two = {**fixed, "d1_d": _state("d1_d", 0.03)}                                  # a second stick knocked away
    assert task.score(two) == 0.0
    swap = {**objs, task.MOVE[0]: _state(task.MOVE[1]), "d1_a": _state(task.MOVE[0])}
    assert task.score(swap) == 0.0                                                 # two sticks moved
    stacked = {**fixed, "d2_c": _state(task.MOVE[1])}
    assert task.score(stacked) == 0.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert len(t.free_objects) == len(START) == 18


def test_sticks_do_not_touch():
    """Sticks' footprints stay 2 mm apart (their corners, any pair, in the start layout)."""
    def corners(slot):
        (x, y), yaw = task.SLOTS[slot]
        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = task.STICK[0] / 2, task.STICK[1] / 2
        return [(x + c * u - s * v, y + s * u + c * v) for u, v in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]

    def gap(p, q):  # separating-axis distance between two rectangles
        best = 0.0
        for poly in (p, q):
            for k in range(4):
                (x0, y0), (x1, y1) = poly[k], poly[(k + 1) % 4]
                n = np.array([y1 - y0, x0 - x1]) / math.hypot(x1 - x0, y1 - y0)
                pa = [n @ v for v in np.array(p)]
                qa = [n @ v for v in np.array(q)]
                best = max(best, min(qa) - max(pa), min(pa) - max(qa))
        return best

    names = list(task.SLOTS)
    for a, b in itertools.combinations(names, 2):
        if {a, b} == {"op_h", "op_v"}:
            continue                                                     # the plus's two sticks cross (unused here)
        assert gap(corners(a), corners(b)) >= 0.0019, (a, b)


def test_sticks_lie_on_the_mat():
    for (x, y), yaw in task.SLOTS.values():
        along_x = abs(math.cos(yaw)) > 0.5
        hx, hy = (task.STICK[0] / 2, task.STICK[1] / 2) if along_x else (task.STICK[1] / 2, task.STICK[0] / 2)
        assert abs(x - task.MAT.pos[0]) + hx < task.MAT.size[0] / 2 - 0.005
        assert abs(y - task.MAT.pos[1]) + hy < task.MAT.size[1] / 2 - 0.005


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
