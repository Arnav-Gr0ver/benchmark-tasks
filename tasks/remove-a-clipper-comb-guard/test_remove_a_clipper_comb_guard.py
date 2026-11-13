"""Offline checks (no GPU, no Isaac Sim; needs yam_arena importable): python -m pytest -q
The physics (the snap clip holding against a straight pull, releasing when the guard's back is tipped up) is checked
by check_demo.py (PyBullet)."""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
pytest.importorskip("yam_arena.task_file")
from yam_arena import station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402

import household as H  # noqa: E402
import task as T  # noqa: E402

FAR = [np.array([0.0, 0.5, 1.2])] * 8
HOME = np.array(T.GUARD.pos)


def objs(guard=None, tray=None, clipper=None):
    return {"clipper": clipper or H.state_of(T.CLIPPER), "guard": guard or H.state_of(T.GUARD),
            "tray": tray or H.state_of(T.TRAY)}


def watch(o, fingers=FAR, steps=1):
    for _ in range(steps):
        T._watch(o, fingers)


def in_tray(dx=0.0, dy=0.0):
    """The guard lying level in the tray, its length along the tray's long side."""
    g = H.state_of(T.GUARD, origin=(0, 0, 0), yaw=math.pi / 2)
    c = H.corners(T.GUARD, g)
    mid = (c.min(0) + c.max(0)) / 2
    origin = np.array([T.TRAY_XY[0] - mid[0] + dx, T.TRAY_XY[1] - mid[1] + dy,
                       H.MAT_TOP + H.TRAY_FLOOR_T - c[:, 2].min()])
    return H.state_of(T.GUARD, origin=origin, yaw=math.pi / 2)


def on(guard):
    """Finger points with one pad point on the guard's spine."""
    return [guard.to_world((0.06, 0.0, sum(T.SPINE_Z) / 2))] + FAR[1:]


def test_fitted_guard_is_located_and_caught_and_scores_zero():
    T._fresh()
    o = objs()
    assert T.lips_on_blade(o["guard"], o["clipper"]) and T.hooked(o["guard"], o["clipper"])
    assert not T.detached(o)
    watch(o, steps=60)
    assert T.score(o) == 0.0


def test_the_clip_and_the_front_hook_overlap_their_catches():
    assert T.BARB_X[1] - T.CAP_X[0] >= 0.0015             # the barb reaches under the cap
    assert T.BARB_Z[1] < T.TOP_Z + T.POST_H               # and sits below it
    assert T.TIP_HOOK_X[1] - T.TIP_HOOK_X[0] >= 0.002     # the front hook under the blade's tip
    assert T.CLIP.target < T.CLIP.low                     # the clip's spring is preloaded against its stop


def test_guard_barely_off_is_not_detached():
    T._fresh()
    for shift in ((0.004, 0, 0), (0.012, 0, 0.004), (0.0, 0, 0.006)):
        o = objs(guard=H.state_of(T.GUARD, origin=HOME + shift))
        assert not T.detached(o), shift
        assert T.score(o) == 0.0


def test_guard_clear_of_the_clipper_but_held_scores_half():
    T._fresh()
    g = H.state_of(T.GUARD, origin=HOME + (0.03, 0, 0.06))
    o = objs(guard=g)
    assert T.detached(o)
    watch(o, on(g), steps=60)
    assert T.score(o) == 0.5


def test_guard_on_the_mat_beside_the_tray_scores_half():
    T._fresh()
    g = in_tray(dx=0.12)
    g = H.state_of(T.GUARD, origin=g.origin - (0, 0, H.TRAY_FLOOR_T), yaw=math.pi / 2)
    o = objs(guard=g)
    watch(o, steps=60)
    assert T.score(o) == 0.5


def test_guard_in_the_tray_but_held_scores_half():
    T._fresh()
    g = in_tray()
    o = objs(guard=g)
    watch(o, on(g), steps=60)
    assert T.S["held"] and T.score(o) == 0.5


def test_guard_in_the_tray_and_released_needs_two_seconds():
    T._fresh()
    o = objs(guard=in_tray())
    watch(o, steps=39)
    assert T.score(o) == 0.5
    watch(o, steps=1)
    assert T.score(o) == 1.0


def test_guard_over_the_tray_wall_is_not_stowed():
    T._fresh()
    o = objs(guard=in_tray(dy=0.025))                    # its end past the inner wall
    watch(o, steps=60)
    assert T.score(o) == 0.5
    T._fresh()
    high = in_tray()
    o = objs(guard=H.state_of(T.GUARD, origin=high.origin + (0, 0, 0.012), yaw=math.pi / 2))   # not on the floor
    watch(o, steps=60)
    assert T.score(o) == 0.5


def test_anything_off_the_table_scores_zero_for_good():
    T._fresh()
    o = objs(guard=H.state_of(T.GUARD, origin=(0.3, 0.9, H.TOP - 0.3)))
    watch(o)
    assert T.score(o) == 0.0
    o = objs(guard=in_tray())
    watch(o, steps=60)
    assert T.score(o) == 0.0                              # latched for the run
    T._fresh()
    watch(o, steps=60)
    assert T.score(o) == 1.0


def _play(plan):
    plans = {a: ArmPlan(list(w), station.BASES[a]) for a, w in plan.items()}
    end = max(p.duration for p in plans.values())
    t = 0.0
    while t <= end + 3.0:
        t += 0.05
        for p in plans.values():
            p.action7(t)
    return plans, int(round((end + 3.0) / 0.05))


def test_demo_is_reachable_and_inside_the_run_limit():
    plans, steps = _play(T.demo())
    for p in plans.values():
        assert p.worst_ik_mm < 1.0 and p.worst_ik_deg < 1.0
    assert steps <= 2400


def test_counterexamples_are_reachable():
    for name, (fn, _expect, _why) in T.COUNTEREXAMPLES.items():
        plans, _ = _play(fn())
        for p in plans.values():
            assert p.worst_ik_mm < 1.0, name


def test_task_json_matches_the_brief():
    meta = json.loads((HERE / "task.json").read_text())
    assert meta["category"] == "Societal relatability"
    assert meta["instruction"] == ("Remove the comb guard from the switched-off hair clipper and put the guard in the "
                                   "collection tray.")
    assert meta["context"].startswith("People who cut hair or trim beards at home remove a clipper's comb guard")
    assert len(meta["fidelity"]) <= 500
