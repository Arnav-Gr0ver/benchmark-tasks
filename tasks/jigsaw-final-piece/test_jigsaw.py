"""Offline checks (no GPU, no Isaac Sim; needs yam_arena, and shapely for the fit checks):
python -m pytest test_jigsaw.py -q"""

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
from puzzlekit import BodyState, prism_mesh  # noqa: E402

DECOYS = [n for n in task.START if n != task.CORRECT]


def _rot(points, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y, s * x + c * y) for x, y in points]


def _placed_outlines(shapely):
    from shapely.ops import unary_union
    out = []
    for rec in task.DATA["placed"].values():
        cx, cy = ((rec["cell"][0] - 1.5) * task.CELL, (rec["cell"][1] - 1.5) * task.CELL)
        out.append(shapely.Polygon([(x + cx, y + cy) for x, y in rec["outline"]]))
    return unary_union(out)


def test_only_the_missing_piece_fits_and_only_one_way_round():
    shapely = pytest.importorskip("shapely")
    board = _placed_outlines(shapely)
    hx, hy = ((task.HOLE_CELL[0] - 1.5) * task.CELL, (task.HOLE_CELL[1] - 1.5) * task.CELL)
    for name in task.START:
        for deg in (0, 90, 180, 270):
            p = shapely.Polygon([(x + hx, y + hy) for x, y in _rot(task.DATA["candidates"][name]["outline"], deg)])
            overlap = p.intersection(board).area * 1e6           # mm^2
            fits = name in (task.CORRECT, "piece_b") and deg == 0
            if fits:
                assert overlap == 0.0 and p.distance(board) >= 0.0009, (name, deg)   # 1 mm clear all round
            else:
                assert overlap > 20.0, (name, deg, overlap)       # blocked by a neighbour's tab or body


def _colour_at(rec, x, y):
    for d in rec["decals"]:
        for (ax, ay), (bx, by), (cx, cy) in d["triangles"]:
            d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
            d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
            d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
            if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
                return tuple(d["color"])
    return None


def _agreement(name, deg):
    """How much of the picture near the hole's edge (2 to 6 mm in from the cell's sides) a piece, turned ``deg``
    and set in the hole, shows the same as the missing piece does: the picture the neighbours continue."""
    truth, rec = task.DATA["candidates"][task.CORRECT], task.DATA["candidates"][name]
    h = task.CELL / 2
    same = total = 0
    for x, y in itertools.product(np.arange(-h + 0.002, h - 0.002, 0.0015), repeat=2):
        if max(abs(x), abs(y)) < h - 0.006:
            continue
        want = _colour_at(truth, x, y)
        (bx, by), = _rot([(x, y)], -deg)
        got = _colour_at(rec, bx, by)
        if want is None or got is None:
            continue
        total += 1
        same += want == got
    return same / total


def test_only_the_right_picture_turned_the_right_way_continues_the_board():
    assert _agreement(task.CORRECT, 0) == 1.0
    assert _agreement("piece_a", 0) == 1.0                    # the right picture, the wrong tabs
    for name in task.START:
        for deg in (0, 90, 180, 270):
            if name in (task.CORRECT, "piece_a") and deg == 0:
                continue
            assert _agreement(name, deg) < 0.75, (name, deg)


def test_the_peg_sits_on_solid_plywood():
    shapely = pytest.importorskip("shapely")
    side = task.KNOB[0]
    peg = shapely.box(-side / 2, -side / 2, side / 2, side / 2)
    for name in task.START:
        outline = shapely.Polygon(task.DATA["candidates"][name]["outline"])
        assert outline.buffer(-0.0008).contains(peg), name


def test_prism_meshes_are_closed_and_bevelled():
    rec = task.DATA["candidates"][task.CORRECT]
    for part, bottom in zip(rec["parts"], rec["bottoms"]):
        pts, counts, idx = prism_mesh(part, 0.0, task.THICK, bottom, 0.0008)
        assert sum(counts) == len(idx) and max(idx) == len(pts) - 1
        zs = sorted({round(p[2], 6) for p in pts})
        assert zs == ([0.0, 0.0008, task.THICK] if bottom else [0.0008, task.THICK])


def _state(pos, yaw_deg=0.0, tilt_deg=0.0):
    pos = np.asarray(pos, float)
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qx = np.array([math.sin(b), 0.0, 0.0, math.cos(b)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    return BodyState(pos=pos, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos, high=pos)


def test_scores():
    start = {n: _state((*xy, task.TOP + 0.0005), deg) for n, (xy, deg) in task.START.items()}
    assert task.score(start) == 0.0
    hole = (*task.HOLE_XY, task.BASE_TOP)
    seated = {**start, task.CORRECT: _state(hole)}
    assert task.score(seated) == 1.0 and task.success(seated)
    off = (task.HOLE_XY[0] + 0.001, task.HOLE_XY[1] - 0.001, task.BASE_TOP + 0.0005)
    assert task.score({**start, task.CORRECT: _state(off, 3.0, 1.0)}) == 1.0           # within the play of the fit
    on_tabs = (task.HOLE_XY[0] + 0.004, task.HOLE_XY[1], task.BASE_TOP + task.THICK)   # resting on the neighbours
    assert task.score({**start, task.CORRECT: _state(on_tabs, 180.0, 4.0)}) == 0.5
    assert task.score({**start, task.CORRECT: _state(hole, 180.0)}) == 0.5            # turned the wrong way
    held = (*task.HOLE_XY, task.BASE_TOP + 0.05)                                       # still carried over it
    assert task.score({**start, task.CORRECT: _state(held)}) == 0.0
    for name in DECOYS:                                                                # a wrong piece in the hole
        assert task.score({**start, name: _state(hole)}) == 0.0
        assert task.score({**seated, name: _state((*task.HOLE_XY, task.BASE_TOP + task.THICK))}) == 0.0


def test_task_file_loads():
    t = task_file.load(HERE / "task.py")
    assert set(t.free_objects) == set(task.START)
    assert len([o for o in t.objects if o.name.startswith("piece_") and not o.free]) == 15


def test_loose_pieces_lie_apart_and_off_the_board():
    reach = task.CELL / 2 + 0.015                                     # body half plus a tab
    xy = [v[0] for v in task.START.values()]
    for a, b in itertools.combinations(xy, 2):
        assert math.dist(a, b) > 2 * reach
    for x, y in xy:
        assert abs(y) - reach > task.BASE_SIZE / 2 + 0.02


def test_fingers_stay_above_the_board():
    tips = task.PLACE_Z - task.PAD_TIP
    assert tips >= task.BASE_TOP + task.THICK + 0.003                  # above the neighbours' tops while lowering
    assert task.PICK_Z - task.PAD_TIP > task.TOP + task.THICK + 0.001  # above the piece's top at the pick


def test_demo_is_reachable():
    demo = pytest.importorskip("yam_arena.demo")
    station = pytest.importorskip("yam_arena.station")
    for arm, waypoints in task.demo().items():
        p = demo.ArmPlan(waypoints, station.BASES[arm])
        for t in itertools.takewhile(lambda t: t < p.duration, itertools.count(0.0, 0.05)):
            p.action7(t)
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 3.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
