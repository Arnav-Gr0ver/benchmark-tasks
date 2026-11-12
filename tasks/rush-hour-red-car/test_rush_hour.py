"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_rush_hour.py -q"""

from __future__ import annotations

import collections
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

NAMES = list(task.VEHICLES)


def _occupied(starts, skip=None):
    return {c for n, s in starts.items() if n != skip for c in task.cells(n, s)}


def _moves(starts):
    """Every legal slide (any distance, one vehicle) under Rush Hour's rules; the red car may slide out of the exit
    once its row is clear to the wall."""
    for name in NAMES:
        _, d, _, _, n = task.VEHICLES[name]
        r, c = starts[name]
        full = _occupied(starts, skip=name)
        for step in (-1, 1):
            k = step
            while True:
                s = (r + k, c) if d == "V" else (r, c + k)
                cs = task.cells(name, s)
                if any(not (0 <= a < task.N and 0 <= b < task.N) for a, b in cs):
                    if name == task.TARGET and step == 1 and all(b >= task.N or (a, b) not in full for a, b in cs):
                        if all((r, b) not in full for b in range(c + n, task.N)):
                            yield name, "out"
                    break
                if any(x in full for x in cs):
                    break
                yield name, s
                k += step


def _solve():
    start = {n: (task.VEHICLES[n][2], task.VEHICLES[n][3]) for n in NAMES}
    key = lambda s: tuple(s[n] for n in NAMES)  # noqa: E731
    seen = {key(start): []}
    queue = collections.deque([start])
    found = []
    while queue:
        s = queue.popleft()
        path = seen[key(s)]
        if found and len(path) >= len(found[0]):
            continue
        for name, to in _moves(s):
            if to == "out":
                found.append(path + [(name, "out")])
                continue
            nxt = {**s, name: to}
            if key(nxt) not in seen:
                seen[key(nxt)] = path + [(name, to)]
                queue.append(nxt)
    return found


def test_the_only_shortest_solution_is_the_yellow_car_two_cells_away_then_the_red_car_out():
    sols = _solve()
    assert sols == [[(task.BLOCKER, task.YELLOW_TO), (task.TARGET, "out")]]


def test_the_layout_is_legal_and_fits():
    seen = set()
    for name in NAMES:
        for r, c in task.cells(name):
            assert 0 <= r < task.N and 0 <= c < task.N and (r, c) not in seen, name
            seen.add((r, c))
    assert (task.EXIT_ROW, task.VEHICLES[task.TARGET][3]) in task.cells(task.TARGET)
    assert task.VEHICLES[task.TARGET][1] == "H"


def _state(name, start=None, dx=0.0, dy=0.0, z=None, dyaw_deg=0.0):
    x, y = task.centre_of(name, start)
    yaw = (0.0 if task.VEHICLES[name][1] == "V" else math.pi / 2) + math.radians(dyaw_deg)
    pos = np.array([x + dx, y + dy, task.BOARD_TOP if z is None else z])
    return BodyState(pos=pos, quat_xyzw=np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
                     lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos, high=pos)


def _start():
    return {n: _state(n) for n in NAMES}


def test_scores():
    task._clear()
    objs = _start()
    assert task.score(objs) == 0.0
    moved = {**objs, task.BLOCKER: _state(task.BLOCKER, task.YELLOW_TO)}
    assert task.score(moved) == 0.5
    one = {**objs, task.BLOCKER: _state(task.BLOCKER, (task.EXIT_ROW, 4))}       # one cell: still in the way
    assert task.score(one) == 0.0
    done = {**moved, task.TARGET: _state(task.TARGET, dy=task.RED_TO_Y - task.centre_of(task.TARGET)[1])}
    assert task.score(done) == 1.0 and task.success(done)
    half = {**moved, task.TARGET: _state(task.TARGET, dy=-(task.OUTER - 0.01))}  # partly through the exit
    assert task.score(half) == 0.5
    skew = {**done, "purple_car": _state("purple_car", dy=0.012)}                # knocked out of its lane
    assert task.score(skew) == 0.0
    nudged = {**done, "purple_car": _state("purple_car", dx=0.01, dy=0.004, dyaw_deg=8.0)}  # nudged along it
    assert task.score(nudged) == 1.0
    task._watch({**objs, task.TARGET: _state(task.TARGET, z=task.BOARD_TOP + 0.02)})        # hopped once
    assert task.score(done) == 0.0
    task._clear()
    assert task.score(done) == 1.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert set(t.free_objects) == set(NAMES)


def test_vehicles_fit_their_cells_and_the_exit():
    for name in NAMES:
        assert task.length_of(name) < task.VEHICLES[name][4] * task.C
    assert task.BODY_W < task.C - 0.004                                    # 3 mm a side in the exit's gap
    x, _ = task.centre_of(task.BLOCKER, task.YELLOW_TO)
    assert x + task.length_of(task.BLOCKER) / 2 < task.CENTRE[0] + task.INNER   # clear of the far wall
    assert task.RED_TO_Y - task.length_of(task.TARGET) / 2 > task.CENTRE[1] - task.OUTER - task.APRON_LEN


def test_fingers_clear_other_pegs_and_the_walls():
    finger = 0.008                                                             # finger thickness, generous
    reach = -task.APPROACH + finger                                           # m from the peg's centre, along the grip
    for name in (task.BLOCKER, task.TARGET):
        x, y = task.centre_of(name)
        along_x = task.VEHICLES[name][1] == "V"
        for other in NAMES:
            if other == name:
                continue
            ox, oy = task.centre_of(other)
            dx, dy = abs(ox - x), abs(oy - y)
            along, across = (dx, dy) if along_x else (dy, dx)
            if across < task.PEG[0] / 2 + 0.012:                               # in the fingers' path
                assert along - task.PEG[0] / 2 > reach + 0.003, (name, other)
    assert task.GRIP_Z - task.PAD_TIP > task.BOARD_TOP + task.WALL_H + 0.005   # fingertips over the walls


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
