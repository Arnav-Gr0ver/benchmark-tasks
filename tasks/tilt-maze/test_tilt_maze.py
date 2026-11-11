"""Checks for the tilt maze that need no GPU and no Isaac Sim: python -m pytest test_tilt_maze.py -q

They need yam_arena installed (as on the simulation instance) for the task-file and demo-reach checks.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import maze  # noqa: E402
import plan  # noqa: E402


def test_walls_leave_room_for_the_ball():
    channel = min(maze.WALLS[k] - maze.WALLS[k - 1] for k in range(1, len(maze.WALLS))) - maze.WALL_T
    assert channel > 2 * maze.BALL_R + 0.004          # a 16 mm ball rolls freely in a 22 mm channel
    assert maze.WALL_H > 2 * maze.BALL_R + 0.001      # and fits under the lid
    assert maze.WALLS[0] - maze.WALL_T / 2 > 2 * maze.BALL_R  # the centre cell holds it
    for wall, gaps in maze.GAPS.items():
        for a, b in gaps:
            assert ((b - a) % 360) * math.pi / 180 * maze.WALLS[wall] > 2 * maze.BALL_R + 0.006


def test_spokes_do_not_sit_in_gaps():
    for ring, spokes in maze.SPOKES.items():
        for deg in spokes:
            for wall in (ring - 1, ring):
                assert not maze._open(wall, deg, 0.002), (ring, deg, wall)


def test_ball_start_scores_zero_and_centre_scores_one():
    assert maze.ring_at(*maze.START_XY) == 4
    assert maze.score_at(*maze.START_XY) == 0.0
    assert maze.score_at(0.0, 0.0) == 1.0
    assert [maze.score_at((maze.WALLS[k - 1] + maze.WALLS[k]) / 2, 0.0) for k in range(1, 5)] == [0.75, 0.5, 0.25, 0]


def test_the_stand_stops_the_board_at_the_tilt_stop_and_clears_it_level():
    under = maze.PIVOT_BELOW_FLOOR - maze.BASE_T
    level_gap = (maze.PIVOT_Z + under) - maze.STAND_TOP_Z
    assert 0.004 < level_gap < 0.010                     # a few millimetres of clearance when level
    tilt = maze.TILT_STOP
    rim = maze.PIVOT_Z + under * math.cos(tilt) - maze.STAND_R * math.sin(tilt)
    assert abs(rim - maze.STAND_TOP_Z) < 1e-9
    # the board's own rim stays well above the table at the stop
    edge = maze.FLOOR_ABOVE_TABLE + maze.PIVOT_Z + under * math.cos(tilt) - maze.BOARD_R * math.sin(tilt)
    assert edge > 0.02


def test_the_maze_has_a_route():
    path = maze.route()
    length = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    assert length > 0.6


def test_the_demo_presses_bring_the_ball_home_in_the_model():
    task = pytest.importorskip("task")
    model = plan.BallModel()
    rings = []
    for az in task.DEMO_HOLDS:
        model.hold(az, math.degrees(task.DEMO_TILT))
        rings.append(maze.ring_at(model.x, model.y))
    assert rings[-1] == 0, rings


def test_the_demo_presses_still_work_at_other_tilts():
    for tilt in (5.0, 6.0):
        assert plan.plan(tilt_deg=tilt) is not None


def test_random_tilting_rarely_leaves_the_outer_ring():
    counts = plan.random_tilting(trials=100, holds=30)
    assert counts[4] >= 85, counts      # at least 85 of 100 random 30-press runs end where they started (score 0)
    assert counts[0] <= 1, counts


def test_rocking_between_two_directions_does_not_get_past_ring_3():
    model = plan.BallModel()
    best = 4
    for a in range(0, 360, 30):
        for b in range(a + 30, 360, 30):
            model.reset()
            for i in range(10):
                model.hold(a if i % 2 == 0 else b, 6.0, max_s=3.0)
                best = min(best, maze.ring_at(model.x, model.y))
    assert best >= 3


def test_task_file_loads():
    task_file = pytest.importorskip("yam_arena.task_file")
    t = task_file.load(HERE / "task.py")
    assert t.scene == "station" and t.free_objects == ()  # nothing is jittered: board pinned, ball on its mark
    assert set(t.names) == {"maze_stand", "maze_board", "ball"}


def test_each_press_is_reachable_by_its_arm():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    task = pytest.importorskip("task")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        t = 0.0
        while t < p.duration:
            p.action7(t)
            t += 0.05
        assert p.worst_ik_mm < 2.0, (arm, p.worst_ik_mm)
        assert p.worst_ik_deg < 3.0, (arm, p.worst_ik_deg)


def test_both_arms_are_needed():
    task = pytest.importorskip("task")
    arms = {task.arm_for(az) for az in task.DEMO_HOLDS}
    assert arms == {"left", "right"}
    assert any(15 <= az <= 90 for az in task.DEMO_HOLDS) and any(270 <= az <= 345 for az in task.DEMO_HOLDS)
