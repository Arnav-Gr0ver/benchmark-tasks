"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_kerplunk.py -q
The physics of which rod is safe is checked by check_physics.py (needs pybullet)."""

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
import tower  # noqa: E402
from shapekit import BodyState  # noqa: E402


def _to_rod(name, point):
    """Distance from a point to a rod's centre line (the rods are long: their ends are far outside the tube)."""
    c, d = tower.rod_centre(name), tower.rod_dir(name)
    v = np.asarray(point) - c
    return float(np.linalg.norm(v - (v @ d) * d))


TOUCH = tower.ROD_R + tower.MARBLE_R
SUPPORT = {"marble_far_l": {"far", "guard"}, "marble_far_r": {"far", "guard"}, "marble_left_n": {"left"},
           "marble_left_f": {"left"}, "marble_right": {"right"}}


def test_each_marble_touches_its_rods_and_the_wall_and_nothing_else():
    for m, pos in tower.MARBLES.items():
        touching = {r for r in tower.RODS if _to_rod(r, pos) < TOUCH + 1e-6}
        assert touching == SUPPORT[m], m
        for r in set(tower.RODS) - touching:
            assert _to_rod(r, pos) > TOUCH + 0.002, (m, r)
        assert math.isclose(math.hypot(pos[0], pos[1]), tower.TUBE_R_IN - tower.MARBLE_R, abs_tol=1e-9)
        assert pos[2] > tower.FALLEN_Z + 0.02                              # well above the fallen plane
    for a, b in itertools.combinations(tower.MARBLES.values(), 2):
        assert math.dist(a, b) >= 2 * tower.MARBLE_R - 1e-9


def test_ledge_marbles_sit_outside_their_ledge_and_the_guard_is_level_with_its_marbles():
    """A ledge marble's centre is on the wall side of its rod (so the rod and the wall hold it); the guard touches
    the far pair at their middles (horizontally: it cannot hold them up)."""
    for m, rods in SUPPORT.items():
        pos = np.array(tower.MARBLES[m])
        for r in rods:
            axis, off, z, _ = tower.RODS[r]
            across = pos[1] if axis == "x" else pos[0]
            if r == "guard":
                assert math.isclose(pos[2], z, abs_tol=1e-9) and abs(across) <= abs(off) + TOUCH + 1e-9
            else:
                assert abs(across) > abs(off) and pos[2] > z


def test_crossing_rods_clear_each_other():
    for a, b in itertools.combinations(tower.RODS, 2):
        if tower.RODS[a][0] == tower.RODS[b][0]:
            ca, cb = tower.rod_centre(a), tower.rod_centre(b)
            d = math.hypot(*(ca - cb)[[0 if tower.RODS[a][0] == "y" else 1, 2]])
        else:
            d = abs(tower.RODS[a][2] - tower.RODS[b][2])               # they cross: the heights separate them
        assert d >= 2 * tower.ROD_R + 0.0004, (a, b, d)


def test_extracted_reads_in_place_half_and_out():
    for name in tower.RODS:
        c = tower.rod_centre(name)
        d = tower.rod_dir(name)
        assert tower.extracted(name, c) == 0.0
        for sign in (-1, 1):
            lo, hi = 0.0, 0.3
            for _ in range(60):                                           # the slide that brings it to 1
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if tower.extracted(name, c + sign * d * mid) < 1.0 else (lo, mid)
            last_end = c + sign * d * hi - sign * d * tower.ROD_LEN / 2
            assert math.hypot(last_end[0], last_end[1]) > tower.R_OUT     # at 1, its last end is outside the tube


def _body(pos, rot=(0.0, 0.0, 0.0, 1.0)):
    pos = np.asarray(pos, float)
    return BodyState(pos=pos, quat_xyzw=np.asarray(rot, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos,
                     high=pos)


def _start():
    objs = {task.ROD_NAMES[n]: _body(task.world(tower.rod_centre(n))) for n in tower.RODS}
    objs.update({m: _body(task.world(p)) for m, p in tower.MARBLES.items()})
    return objs


def _slid(name, distance):
    return _body(task.world(tower.rod_centre(name) - tower.rod_dir(name) * distance))


def test_scores():
    task._clear()
    objs = _start()
    assert task.score(objs) == 0.0
    blue = task.SAFE_ROD
    assert task.score({**objs, blue: _slid("guard", task.PULL)}) == 1.0
    assert task.success({**objs, blue: _slid("guard", task.PULL)})
    assert task.score({**objs, blue: _slid("guard", 0.07)}) == 0.5
    assert task.score({**objs, blue: _slid("guard", 0.03)}) == 0.0
    two = {**objs, blue: _slid("guard", task.PULL), "rod_green": _slid("left", 0.09)}
    assert task.score(two) == 0.0                                          # a second rod half out

    task._clear()
    fell = {**objs, "rod_orange": _slid("far", 0.13), "marble_far_l": _body(task.world((0.03, 0.01, 0.008)))}
    assert task.score(fell) == 0.0
    assert task.score({**objs, blue: _slid("guard", task.PULL)}) == 0.0     # latched

    task._clear()
    lifted = dict(objs)
    x, y, z = tower.MARBLES["marble_right"]
    lifted["marble_right"] = _body(task.world((x, y, z + 0.02)))
    assert task.score({**lifted, blue: _slid("guard", task.PULL)}) == 0.0   # a marble picked up


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert set(t.free_objects) == set(task.ROD_NAMES.values()) | set(tower.MARBLES)


def test_demo_pulls_the_blue_rod_fully_out_and_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    assert tower.extracted(tower.SAFE, tower.rod_centre(tower.SAFE) - tower.rod_dir(tower.SAFE) * task.PULL) > 1.0
    far_end = task.world(tower.rod_centre("far") - tower.rod_dir("far") * tower.ROD_LEN / 2)
    assert task.GRIP_Y - 0.010 - far_end[1] < -0.01                           # the fingers clear the orange rod's end
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
