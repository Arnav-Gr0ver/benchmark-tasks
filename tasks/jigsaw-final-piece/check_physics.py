"""Physics check outside Isaac Sim (pip install pybullet): drops the loose pieces into the hole and reports which
seat. World scaled x10 with gravity x10 (same timing), so PyBullet's contact margins stay small.

    python check_physics.py          # every piece at every quarter turn, dropped centred, then pressed
    python check_physics.py offsets  # the missing piece dropped 2-6 mm off the hole's centre, 8 directions
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pybullet as p

D = json.loads((pathlib.Path(__file__).resolve().parent / "pieces.json").read_text())
S = 10.0
THICK, CELL = D["thick"], D["cell"]
BEV_IN, BEV_UP = D["bevel"]
HOLE = ((D["hole"][0] - 1.5) * CELL, (D["hole"][1] - 1.5) * CELL)
PEG_H = 0.014
DROP_GAP = 0.003


def world():
    p.resetSimulation()
    p.setGravity(0, 0, -9.81 * S)
    p.setTimeStep(1 / 1000)
    p.setPhysicsEngineParameter(numSolverIterations=80)
    base = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1 * S, 0.1 * S, 0.002 * S]),
                             basePosition=[0, 0, -0.002 * S])
    p.changeDynamics(base, -1, lateralFriction=0.6, collisionMargin=0.0002 * S)
    for rec in D["placed"].values():
        cx, cy = (rec["cell"][0] - 1.5) * CELL, (rec["cell"][1] - 1.5) * CELL
        cs = p.createCollisionShape(p.GEOM_MESH, vertices=[[v * S for v in q] for q in rec["mesh"]["points"]],
                                    indices=[i for t in rec["mesh"]["triangles"] for i in t],
                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        body = p.createMultiBody(0, cs, basePosition=[cx * S, cy * S, 0])
        p.changeDynamics(body, -1, lateralFriction=0.6, collisionMargin=0.0002 * S)


def piece(name, dx, dy, yaw):
    """The loose piece as in task.py: its bevelled convex parts and its peg, one rigid body (8 g), let go DROP_GAP
    above the board."""
    rec = D["candidates"][name]
    shapes = []
    for part, bottom in zip(rec["parts"], rec["bottoms"]):
        pts = [[x, y, BEV_UP] for x, y in part] + [[x, y, THICK] for x, y in part]
        pts += [[x, y, 0.0] for x, y in bottom] if bottom else []
        shapes.append(p.createCollisionShape(p.GEOM_MESH, vertices=[[v * S for v in q] for q in pts]))
    peg = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.006 * S, 0.006 * S, PEG_H / 2 * S])
    n = len(shapes)
    q = p.getQuaternionFromEuler([0, 0, yaw])
    body = p.createMultiBody(
        baseMass=0.002, baseCollisionShapeIndex=peg, baseOrientation=q,
        basePosition=[(HOLE[0] + dx) * S, (HOLE[1] + dy) * S, (THICK + DROP_GAP + THICK + PEG_H / 2) * S],
        linkMasses=[0.006 / n] * n, linkCollisionShapeIndices=shapes, linkVisualShapeIndices=[-1] * n,
        linkPositions=[[0, 0, -(THICK + PEG_H / 2) * S]] * n, linkOrientations=[[0, 0, 0, 1]] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n, linkJointAxis=[[0, 0, 1]] * n)
    for k in range(-1, n):
        p.changeDynamics(body, k, lateralFriction=0.6, collisionMargin=0.0002 * S)
    return body


def settle(body, press=True):
    for _ in range(1500):
        p.stepSimulation()
    if press:  # 3 N down on the peg for 0.8 s, as the demo's closed fingers do
        for _ in range(800):
            p.applyExternalForce(body, -1, [0, 0, -3.0 * S], [0, 0, 0], p.LINK_FRAME)
            p.stepSimulation()
        for _ in range(500):
            p.stepSimulation()
    pos, q = p.getBasePositionAndOrientation(body)
    m = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    under = np.array(pos) / S - m @ np.array([0, 0, THICK + PEG_H / 2])
    dist = math.dist(under[:2], HOLE) * 1000
    yaw = abs((math.degrees(math.atan2(m[1, 0], m[0, 0])) + 180) % 360 - 180)
    tilt = math.degrees(math.acos(min(1.0, m[2, 2])))
    seated = dist <= 2.5 and yaw <= 5 and under[2] <= 0.0015 and tilt <= 3
    return seated, dist, yaw, under[2] * 1000, tilt


def main():
    p.connect(p.DIRECT)
    if sys.argv[1:] == ["offsets"]:
        for r in (0.002, 0.003, 0.004, 0.005, 0.006):
            ok = 0
            for a in range(0, 360, 45):
                for dyaw in (-2.0, 2.0):
                    world()
                    ok += settle(piece(D["correct"], r * math.cos(math.radians(a)), r * math.sin(math.radians(a)),
                                       math.radians(dyaw)))[0]
            print(f"{D['correct']} let go {r * 1000:.0f} mm off the hole's centre (8 directions, turned +-2 deg): "
                  f"{ok}/16 seated")
        return
    for name in D["candidates"]:
        row = []
        for deg in (0, 90, 180, 270):
            world()
            seated, _, _, up, _ = settle(piece(name, 0.0, 0.0, math.radians(deg)))
            row.append(f"{deg:3d} deg: {'seated' if seated else 'not in'} ({up:.1f} mm up)")
        print(f"{name}: " + " | ".join(row))


if __name__ == "__main__":
    main()
