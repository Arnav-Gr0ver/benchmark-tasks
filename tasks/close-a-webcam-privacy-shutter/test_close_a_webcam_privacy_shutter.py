"""Offline checks (no GPU, no Isaac Sim; needs yam_arena importable): python -m pytest -q
The physics (the tab pushed by the fingertips, the webcam staying on its clip) is checked by check_demo.py (PyBullet)."""

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

FAR = [np.array([0.0, 0.5, 1.2])] * 8                    # finger points nowhere near the webcam


def objs(shutter=0.0, origin=None, quat=None):
    return {"webcam": H.state_of(T.WEBCAM, origin=origin, joints={"shutter": shutter}, quat_xyzw=quat),
            "monitor": H.state_of(T.MONITOR), "mat": H.state_of(H.mat())}


def watch(o, fingers=FAR, steps=1):
    for _ in range(steps):
        T._watch(o, fingers)


def on_tab(shutter):
    """Finger points with one pad point on the tab."""
    tab = np.array([T._wc_x + T._tab_x, T.OPEN_Y + shutter, T.Z0 + T.BODY[2] + 0.003])
    return [tab] + FAR[1:]


def test_coverage_runs_from_open_to_fully_closed():
    assert T.coverage(0.0) == 0.0
    assert T.coverage(-T.TRAVEL) == 1.0
    values = [T.coverage(-s) for s in np.linspace(0, T.TRAVEL, 60)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    # full cover needs the plate's far edge past the aperture: not before OPEN_Y - FULL of travel
    assert T.coverage(-(T.OPEN_Y - T.FULL) + 0.0005) < 1.0
    assert T.coverage(-(T.OPEN_Y - T.FULL) - 0.0001) == 1.0
    assert T.TRAVEL > T.OPEN_Y - T.FULL                  # the stop lies beyond the fully covering position


def test_open_or_barely_moved_scores_zero():
    T._fresh()
    o = objs(0.0)
    watch(o, steps=60)
    assert T.score(o) == 0.0
    o = objs(-0.006)
    assert T.coverage(-0.006) < 0.25
    watch(o, steps=60)
    assert T.score(o) == 0.0


def test_partly_closed_scores_half():
    T._fresh()
    o = objs(-0.012)
    assert 0.25 <= T.coverage(-0.012) < 1.0
    watch(o, steps=60)
    assert T.score(o) == 0.5


def test_closed_but_held_scores_three_quarters():
    T._fresh()
    o = objs(-T.TRAVEL)
    watch(o, on_tab(-T.TRAVEL), steps=60)
    assert T.S["held"]
    assert T.score(o) == 0.75


def test_closed_and_released_needs_two_seconds():
    T._fresh()
    o = objs(-T.TRAVEL)
    watch(o, steps=39)                                   # 1.95 s
    assert T.score(o) == 0.75
    watch(o, steps=1)
    assert T.score(o) == 1.0


def test_shutter_sliding_back_open_during_the_observation_restarts_it():
    T._fresh()
    o = objs(-T.TRAVEL)
    watch(o, steps=30)
    watch(objs(-0.012), steps=1)
    watch(o, steps=30)
    assert T.score(o) == 0.75


def test_webcam_off_its_monitor_scores_zero_even_if_put_back():
    T._fresh()
    lifted = objs(-T.TRAVEL, origin=np.array(T.WEBCAM.pos) + [0, 0, 0.03])
    watch(lifted)
    assert T.score(lifted) == 0.0
    back = objs(-T.TRAVEL)
    watch(back, steps=60)
    assert T.score(back) == 0.0                          # "at any time"


def test_a_tilted_but_attached_webcam_keeps_its_score():
    T._fresh()
    a = math.radians(8)
    o = objs(-T.TRAVEL, quat=[0, math.sin(a / 2), 0, math.cos(a / 2)])
    assert T.attached(o["webcam"])
    watch(o, steps=60)
    assert T.score(o) == 1.0


def test_fresh_clears_latched_state():
    T._fresh()
    watch(objs(0.0, origin=np.array(T.WEBCAM.pos) + [0, 0, 0.03]))
    assert T.S["detached"]
    T._fresh()
    assert not T.S["detached"]


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
    assert meta["instruction"] == "Close the webcam's privacy shutter. Leave the webcam attached to its monitor."
    assert meta["context"].startswith("After a video call, people can close a webcam's physical privacy shutter")
    assert len(meta["fidelity"]) <= 500
