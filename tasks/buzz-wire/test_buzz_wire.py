"""Offline checks (no GPU, no Isaac Sim; needs yam_arena installed): python -m pytest test_buzz_wire.py -q"""

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
import track  # noqa: E402
from shapekit import BodyState  # noqa: E402


def _turn_rate():
    d = np.diff(track.PATH, axis=0)
    t = d / np.linalg.norm(d, axis=1)[:, None]
    ang = np.arccos(np.clip((t[1:] * t[:-1]).sum(1), -1.0, 1.0))
    return t, ang / np.linalg.norm(d[1:], axis=1)


def test_the_track_bends_in_three_dimensions_smoothly():
    t, k = _turn_rate()
    slope = np.degrees(np.arcsin(t[:, 2]))
    yaw = np.degrees(np.arctan2(t[:, 1], t[:, 0]))
    assert 0.27 <= track.LENGTH <= 0.36
    assert slope.max() > 20 and slope.min() < -20                 # climbs and drops (out of plane)
    assert yaw.max() > 80 and yaw.min() < -85                     # bends a quarter turn both ways
    assert 1 / k.max() > 0.012                                     # no bend tighter than 12 mm radius
    assert np.allclose(track.tangent(0.01), [1, 0, 0]) and np.allclose(track.tangent(track.LENGTH - 0.01), [1, 0, 0])


def test_a_ring_centred_on_the_wire_clears_it_everywhere():
    for s in np.arange(track.S_GREEN, track.S_RED, 0.002):
        assert track.clearance(track.point(s), track.ring_frame(s)) > 0.010, s


def test_no_fixed_orientation_gets_the_ring_through():
    """Carried along the wire centred but never turned, whichever way it is held (axis level, any heading), the ring
    hits the wire somewhere: it must be turned on the way."""
    for deg in range(0, 180, 5):
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        assert any(track.clearance(track.point(s), rot) < track.TOUCH
                   for s in np.arange(track.S_GREEN, track.S_RED, 0.002)), deg


def test_the_demo_turns_the_ring_through_a_wide_angle():
    yaws = [math.degrees(track.heading(s)) * task.FOLLOW for s in np.arange(task.S0, task.S_END, 0.004)]
    assert max(yaws) - min(yaws) > 80


def _quat(m):
    m = np.asarray(m, float)
    w = math.sqrt(max(1e-12, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])


def _on_wire(s, dz=0.0):
    """The ring centred on the wire at s (raised dz), turned with it, as the scorer sees it (world frame)."""
    pos = track.point(s) + np.array([0.0, 0.0, task.TOP + dz])
    return {"ring": BodyState(pos=pos, quat_xyzw=_quat(track.ring_frame(s)), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                              low=pos, high=pos)}


def test_rotation_round_trip():
    st = _on_wire(0.15)["ring"]
    assert np.allclose(st.rotation(), track.ring_frame(0.15), atol=1e-6)


def test_scores():
    task._clear()
    assert task.score(_on_wire(task.S0)) == 0.0
    for k, c in enumerate(track.CHECKPOINTS[:4], start=1):            # walking along, centred
        assert task.score(_on_wire(c + 0.001)) == pytest.approx(0.2 * k)
    assert task.score(_on_wire(track.S_RED + 0.01)) == 1.0 and task.success(_on_wire(track.S_RED + 0.01))

    task._clear()
    task.score(_on_wire(track.CHECKPOINTS[1] + 0.001))                # 40 %, then a touch: the ring dropped on it
    s = track.CHECKPOINTS[1] + 0.005
    assert task.score(_on_wire(s, dz=track.RING_R - 0.001)) == pytest.approx(0.4)
    assert task.score(_on_wire(track.S_RED + 0.01)) == pytest.approx(0.4)   # credit before the touch is kept

    task._clear()                                                      # touching a sleeve does not count
    assert task.score(_on_wire(0.02, dz=track.RING_R - 0.002)) == 0.0
    assert task.score(_on_wire(track.S_RED + 0.01)) == 1.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert t.free_objects == ("ring",)


def _slerp(r0, r1, a):
    m = r1 @ r0.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(m) - 1) / 2)))
    if ang < 1e-9:
        return r0
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * math.sin(ang))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + math.sin(a * ang) * k + (1 - math.cos(a * ang)) * (k @ k)) @ r0


def test_the_demo_carries_the_ring_clear_of_the_wire_all_the_way():
    """The ring as the gripper carries it between the demo's waypoints (straight lines, turning evenly): at least
    6 mm clear of touching the bare wire, and it ends on the red sleeve."""
    demo = pytest.importorskip("yam_arena.demo")
    way = task.demo()["right"]
    g0 = task._plain_grip_rotation(task.GRIP_YAW)
    off = task.TCP0 - np.array(task.RING_START)
    after_grip = way[3:]
    worst = math.inf
    for w0, w1 in zip(after_grip, after_grip[1:]):
        r0, r1 = demo.grip_rotation(w0.yaw), demo.grip_rotation(w1.yaw)
        for a in np.linspace(0.0, 1.0, 9):
            tcp = (1 - a) * np.asarray(w0.pos) + a * np.asarray(w1.pos)
            ring = _slerp(r0, r1, a) @ g0.T
            centre = tcp - ring @ off - np.array([0.0, 0.0, task.TOP])
            worst = min(worst, track.clearance(centre, ring))
    assert worst > track.TOUCH + 0.006, worst
    end = np.asarray(after_grip[-1].pos) - demo.grip_rotation(after_grip[-1].yaw) @ g0.T @ off
    assert track.progress(end - np.array([0.0, 0.0, task.TOP])) > track.S_RED


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
        assert p.duration < 30.0                                       # inside a 600-step demo run


def test_the_ring_starts_seated_and_clear():
    c = np.array(task.RING_START) - np.array([0.0, 0.0, task.TOP])
    assert track.progress(c) < track.S_GREEN                           # over the green sleeve
    assert track.clearance(c, np.eye(3)) > 0.010
