"""Physics check outside Isaac Sim (pip install pybullet): builds the tower, lets the marbles settle, then for each rod
in turn (each time from a fresh start) pulls it straight out along its length at 5 cm/s and counts the marbles that
fall below the fallen plane (20 mm under the lowest rods). Exactly one rod, the guard, must come out with none
falling. World scaled x10 with gravity x10 (same timing), so PyBullet's contact margins stay small.

    python check_physics.py          # the four removals
    python check_physics.py sweep    # the same under low, medium and high friction, two speeds, both directions
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tower  # noqa: E402

S = 10.0
DT = 1 / 1000
FRICTION = {"rod": 0.4, "marble": 0.3, "wall": 0.4}
MASS = {"rod": 0.003, "marble": 0.0055}


def build():
    p.resetSimulation()
    p.setGravity(0, 0, -9.81 * S)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=100)
    base = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.07 * S, 0.07 * S, 0.005 * S]),
                             basePosition=[0, 0, -0.005 * S])
    p.changeDynamics(base, -1, lateralFriction=0.4)
    for c, size, rot in tower.wall_boxes():
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in size])
        w = p.createMultiBody(0, cs, basePosition=[v * S for v in c],
                              baseOrientation=p.getQuaternionFromEuler([math.radians(a) for a in rot]))
        p.changeDynamics(w, -1, lateralFriction=FRICTION["wall"], collisionMargin=0.0002 * S)
    for centre, size in tower.bores():
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in size])
        bb = p.createMultiBody(0, cs, basePosition=[v * S for v in centre])
        p.changeDynamics(bb, -1, lateralFriction=FRICTION["wall"], collisionMargin=0.0001 * S)
    rods = {}
    for name in tower.RODS:
        cs = p.createCollisionShape(p.GEOM_CYLINDER, radius=tower.ROD_R * S, height=tower.ROD_LEN * S)
        d = tower.rod_dir(name)
        q = p.getQuaternionFromEuler([0, math.pi / 2, 0]) if d[0] else p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        b = p.createMultiBody(MASS["rod"], cs, basePosition=list(tower.rod_centre(name) * S), baseOrientation=q)
        p.changeDynamics(b, -1, lateralFriction=FRICTION["rod"], collisionMargin=0.0002 * S, rollingFriction=0.0)
        rods[name] = b
    marbles = {}
    for name, pos in tower.MARBLES.items():
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=tower.MARBLE_R * S)
        b = p.createMultiBody(MASS["marble"], cs, basePosition=[v * S for v in (pos[0], pos[1], pos[2] + 0.0003)])
        p.changeDynamics(b, -1, lateralFriction=FRICTION["marble"], collisionMargin=0.0002 * S,
                         rollingFriction=0.0001, spinningFriction=0.0001)
        marbles[name] = b
    return rods, marbles


def zpos(b):
    return p.getBasePositionAndOrientation(b)[0][2] / S


def settle(seconds=2.0):
    for _ in range(int(seconds / DT)):
        p.stepSimulation()


def pull(name, speed=0.05, sign=-1):
    """From a fresh start: pull one rod straight out; the marbles that fell."""
    rods, marbles = build()
    settle(1.0)
    b = rods[name]
    pos0, q0 = p.getBasePositionAndOrientation(b)
    grip = p.createConstraint(b, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos0, childFrameOrientation=q0)
    d = sign * tower.rod_dir(name)
    dist = tower.ROD_LEN / 2 + tower.TUBE_R_IN + 0.04
    n = int(dist / speed / DT)
    for k in range(1, n + 1):
        p.changeConstraint(grip, list(np.array(pos0) + d * dist * S * k / n), maxForce=200 * S)
        p.stepSimulation()
    p.removeConstraint(grip)
    settle(2.0)
    return sorted(m for m, mb in marbles.items() if zpos(mb) < tower.FALLEN_Z)


def sweep():
    p.connect(p.DIRECT)
    ok = True
    for rod_f, marble_f in ((0.2, 0.15), (0.4, 0.3), (0.7, 0.6)):
        FRICTION.update(rod=rod_f, marble=marble_f)
        for speed in (0.05, 0.15):
            for sign in (-1, 1):
                safe = [n for n in tower.RODS if not pull(n, speed, sign)]
                ok &= safe == [tower.SAFE]
                print(f"friction rod {rod_f} marble {marble_f}, {100 * speed:.0f} cm/s, "
                      f"{'toward' if sign < 0 else 'away from'} the robot / its right: safe {safe}")
    print("OK" if ok else "FAIL")


def main():
    if sys.argv[1:] == ["sweep"]:
        return sweep()
    p.connect(p.DIRECT)
    rods, marbles = build()
    start = {n: np.array(p.getBasePositionAndOrientation(b)[0]) / S for n, b in marbles.items()}
    settle(3.0)
    moved = max(np.linalg.norm(np.array(p.getBasePositionAndOrientation(b)[0]) / S - start[n])
                for n, b in marbles.items())
    fallen = [n for n, b in marbles.items() if zpos(b) < tower.FALLEN_Z]
    print(f"left alone 3 s: marbles moved at most {1000 * moved:.2f} mm, fallen: {fallen or 'none'}")
    safe = []
    for name in tower.RODS:
        fallen = pull(name)
        if not fallen:
            safe.append(name)
        print(f"pull out the {name} rod ({tower.RODS[name][3]}): fallen {fallen or 'none'}")
    print(f"safe to remove: {safe} -> {'OK' if safe == [tower.SAFE] else 'FAIL'}")


if __name__ == "__main__":
    main()
