"""Physics check outside Isaac Sim (pip install pybullet): builds the tower, lets the marbles settle, then for each rod
in turn (each time from a fresh start) pulls it straight out along its length at 5 cm/s and counts the marbles that
fall below the fallen plane (20 mm under the lowest rods). Exactly one rod, the guard, must come out with none
falling. World scaled x10 with gravity x10 (same timing), so PyBullet's contact margins stay small.

    python check_physics.py          # the four removals
    python check_physics.py sweep    # the same under low, medium and high friction, two speeds, both directions
    python check_physics.py demo     # the demo's grip, lift and pull on the blue rod, with the arm off by 1-2 mm
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


def demo_replay(err=(0.0, 0.0, 0.0)):
    """The demo's motion on the blue rod: held 6.5 cm from its middle toward its pulled end, that point lifted 2 mm
    (plus an arm error ``err``, m) and then drawn straight out. The marbles that fell, and how far out the rod got."""
    import task_geometry as tg
    rods, marbles = build()
    settle(1.0)
    b = rods[tower.SAFE]
    pos0, q0 = p.getBasePositionAndOrientation(b)
    d = -tower.rod_dir(tower.SAFE)
    grip_along = tg.GRIP_ALONG                     # from the rod's middle to the grip, along d
    grip0 = np.array(pos0) / S + d * grip_along
    local = [0.0, 0.0, grip_along * S]            # the rod's own axis is its local z in PyBullet
    hold = grip0 + np.array(err)
    lifted = hold + np.array([0.0, 0.0, tg.LIFT_BIAS])
    g = p.createConstraint(b, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], local, list(grip0 * S),
                           childFrameOrientation=q0)
    for a, b_, sec in ((grip0, hold, 0.3), (hold, lifted, 0.5), (lifted, lifted + d * tg.PULL, 5.0)):
        n = int(sec / DT)
        for k in range(1, n + 1):
            p.changeConstraint(g, list((a + (b_ - a) * k / n) * S), maxForce=30 * S)
            p.stepSimulation()
    out = tower.extracted(tower.SAFE, np.array(p.getBasePositionAndOrientation(b)[0]) / S)
    p.removeConstraint(g)
    settle(1.5)
    return sorted(m for m, mb in marbles.items() if zpos(mb) < tower.FALLEN_Z), out


def demo_check():
    p.connect(p.DIRECT)
    ok = True
    for err in ((0, 0, 0), (0, 0, -0.002), (0, 0, 0.002), (0.001, 0, 0), (-0.001, 0, 0), (0.001, 0, -0.002)):
        fallen, out = demo_replay(tuple(err))
        ok &= not fallen and out >= 1.0
        print(f"demo with the arm off by {[round(1000 * v, 1) for v in err]} mm: rod {100 * out:.0f} % out, "
              f"fallen {fallen or 'none'}")
    print("OK" if ok else "FAIL")


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
    if sys.argv[1:] == ["demo"]:
        return demo_check()
    p.connect(p.DIRECT)
    rods, marbles = build()
    p.performCollisionDetection()
    worst = min([c[8] for b in rods.values() for c in p.getContactPoints(b)] or [0.0]) / S
    print(f"at the start, the deepest overlap of any rod with anything: {max(0.0, -1000 * worst):.2f} mm")
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
