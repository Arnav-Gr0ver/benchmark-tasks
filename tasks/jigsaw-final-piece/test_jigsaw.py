"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_jigsaw.py -q"""

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

shapekit = pytest.importorskip("shapekit")


def _rects(boxes, centre, quarter_turns=0):
    """The piece's footprint (below the knob) as axis-aligned rectangles (x0, x1, y0, y1), world frame."""
    out = []
    for (cx, cy, cz), (sx, sy, _) in boxes:
        if cz > task.THICK:
            continue
        for _ in range(quarter_turns % 4):
            cx, cy, sx, sy = -cy, cx, sy, sx
        out.append((centre[0] + cx - sx / 2, centre[0] + cx + sx / 2, centre[1] + cy - sy / 2, centre[1] + cy + sy / 2))
    return out


def _min_gap(a, b):
    """Smallest clearance between two sets of rectangles (negative: they overlap)."""
    best = math.inf
    for (a0, a1, a2, a3), (b0, b1, b2, b3) in itertools.product(a, b):
        gx = max(b0 - a1, a0 - b1)
        gy = max(b2 - a3, a2 - b3)
        best = min(best, max(gx, gy) if (gx > 0 or gy > 0) else -min(-gx, -gy))
    return best


def _neighbours():
    return [r for p in task.PLACED for r in _rects(p.boxes, p.pos[:2])]


def test_the_placed_pieces_fit_each_other():
    for a, b in itertools.combinations(task.PLACED, 2):
        assert _min_gap(_rects(a.boxes, a.pos[:2]), _rects(b.boxes, b.pos[:2])) > 0.001, (a.name, b.name)


def test_the_last_piece_fits_one_way_only():
    hole = task.cell_xy((0, 0))
    gaps = [_min_gap(_rects(task.FINAL.boxes, hole, k), _neighbours()) for k in range(4)]
    assert gaps[0] >= 0.0014, gaps                      # 1.5 mm clear all round the right way
    assert all(g < -0.005 for g in gaps[1:]), gaps       # every other way, a tab runs into a neighbour by 5 mm or more


def test_the_middle_piece_has_two_tabs_and_two_notches():
    assert sorted(task.MIDDLE_EDGES.values()) == ["N", "N", "T", "T"]


def _state(pos, yaw=0.0, tilt=0.0):
    q = np.array([math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2)])
    z = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    x1, y1, z1, w1 = z
    x2, y2, z2, w2 = q
    quat = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    pos = np.asarray(pos, float)
    return shapekit.BodyState(pos=pos, quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos, high=pos)


def test_scores():
    hole = task.cell_xy((0, 0))
    assert task.score({"last_piece": _state((*task.START_XY, task.TOP), task.START_YAW)}) == 0.0
    seated = {"last_piece": _state((*hole, task.FLOOR))}
    assert task.score(seated) == 1.0 and task.success(seated)
    assert task.score({"last_piece": _state((*hole, task.FLOOR), math.radians(360))}) == 1.0
    assert task.score({"last_piece": _state((hole[0] + 0.005, hole[1], task.FLOOR + 0.008))}) == 0.5
    assert task.score({"last_piece": _state((*hole, task.FLOOR + 0.008), math.radians(90))}) == 0.25
    assert task.score({"last_piece": _state((*hole, task.FLOOR + 0.008), math.radians(180))}) == 0.25
    assert task.score({"last_piece": _state((hole[0] + 0.05, hole[1], task.FLOOR + 0.008))}) == 0.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert t.free_objects == ("last_piece",)


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
