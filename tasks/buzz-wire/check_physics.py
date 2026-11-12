"""Physics check outside Isaac Sim (pip install pybullet): the ring and handle left on their stand settle without
touching the wire, and the bare wire does block the ring (it cannot be pulled straight off sideways).
World scaled x10 with gravity x10 (same timing), so PyBullet's contact margins stay small.

    python check_physics.py
"""

from __future__ import annotations

import math
import pathlib
import sys
import types

import numpy as np
import pybullet as p

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# task.py needs yam_arena only for its constants and the demo hook; stand-ins are enough here
for name in ("yam_arena", "yam_arena.constants", "yam_arena.demo", "yam_arena.task_file"):
    sys.modules.setdefault(name, types.ModuleType(name))
_tf = sys.modules["yam_arena.task_file"]
if not hasattr(_tf, "Box"):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Shape:
        name: str
        pos: tuple
        color: tuple = (0.7, 0.7, 0.7)
        mass: float | None = None
        yaw: float = 0.0

        def _check(self):
            pass

        @property
        def free(self):
            return self.mass is not None

    @dataclass(frozen=True)
    class Box(_Shape):
        size: tuple = (0.04, 0.04, 0.04)

    @dataclass(frozen=True)
    class Cylinder(_Shape):
        radius: float = 0.03
        height: float = 0.05

    @dataclass(frozen=True)
    class ObjectState:
        pos: object = None

    _tf.Box, _tf.Cylinder, _tf.ObjectState, _tf.TABLE_TOP = Box, Cylinder, ObjectState, {"station": 0.7475}
    sys.modules["yam_arena.constants"].FINGER_OPEN, sys.modules["yam_arena.constants"].FINGER_CLOSED = -0.04695, 0.0
    sys.modules["yam_arena.demo"].Waypoint = lambda *a, **k: (a, k)
    sys.modules["yam_arena.demo"].grip_rotation = lambda yaw=0.0: np.eye(3)

import task  # noqa: E402
import track  # noqa: E402

S = 10.0


def _rot_xyz(ax_deg, ay_deg):
    return p.getQuaternionFromEuler([math.radians(ax_deg), math.radians(ay_deg), 0.0])


def world():
    p.resetSimulation()
    p.setGravity(0, 0, -9.81 * S)
    p.setTimeStep(1 / 1000)
    p.setPhysicsEngineParameter(numSolverIterations=80)
    for obj in task.OBJECTS:
        if obj.name in ("ring", "wire"):
            continue
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in obj.size])
        b = p.createMultiBody(0, cs, basePosition=[v * S for v in obj.pos])
        p.changeDynamics(b, -1, lateralFriction=0.6, collisionMargin=0.0002 * S)
    wire = next(o for o in task.OBJECTS if o.name == "wire")
    for p0, p1, r, _ in wire.segments:
        p0 = np.array(p0) + np.array(wire.pos)
        p1 = np.array(p1) + np.array(wire.pos)
        d = p1 - p0
        length = float(np.linalg.norm(d))
        axis = np.cross([0, 0, 1], d / length)
        ang = math.acos(max(-1, min(1, d[2] / length)))
        if np.linalg.norm(axis) < 1e-9:
            q = [0, 0, 0, 1]
        else:
            q = list(np.append(axis / np.linalg.norm(axis) * math.sin(ang / 2), math.cos(ang / 2)))
        cs = p.createCollisionShape(p.GEOM_CAPSULE, radius=r * S, height=length * S)
        b = p.createMultiBody(0, cs, basePosition=list((p0 + p1) / 2 * S), baseOrientation=q)
        p.changeDynamics(b, -1, lateralFriction=0.4, collisionMargin=0.0001 * S)


def ring():
    boxes = task.RING.boxes
    shapes = [p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in b[1]]) for b in boxes]
    n = len(shapes) - 1
    hub = shapes[-1]  # the handle is the base link; the ring's chords are fixed links
    hb = boxes[-1]
    body = p.createMultiBody(
        baseMass=task.RING_MASS / 2, baseCollisionShapeIndex=hub,
        basePosition=[(task.RING.pos[k] + hb[0][k]) * S for k in range(3)],
        linkMasses=[task.RING_MASS / 2 / n] * n, linkCollisionShapeIndices=shapes[:-1],
        linkVisualShapeIndices=[-1] * n,
        linkPositions=[[(b[0][k] - hb[0][k]) * S for k in range(3)] for b in boxes[:-1]],
        linkOrientations=[_rot_xyz(*b[2]) for b in boxes[:-1]],
        linkInertialFramePositions=[[0, 0, 0]] * n, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n, linkJointAxis=[[0, 0, 1]] * n)
    for k in range(-1, n):
        p.changeDynamics(body, k, lateralFriction=0.6, collisionMargin=0.0001 * S)
    for a in range(n):
        for b in range(-1, n):
            p.setCollisionFilterPair(body, body, a, b, 0)
    return body


def ring_pose(body):
    pos, q = p.getBasePositionAndOrientation(body)
    rot = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    centre = np.array(pos) / S - rot @ np.array(task.RING.boxes[-1][0])
    return centre, rot


def main():
    p.connect(p.DIRECT)
    world()
    b = ring()
    c0, _ = ring_pose(b)
    worst = math.inf
    for k in range(3000):
        p.stepSimulation()
        if k % 50 == 0:
            c, rot = ring_pose(b)
            worst = min(worst, track.clearance(c - np.array([0, 0, task.TOP]), rot))
    c, rot = ring_pose(b)
    tilt = math.degrees(math.acos(min(1.0, rot[2, 2])))
    print(f"left on its stand for 3 s: moved {1000 * np.linalg.norm(c - c0):.2f} mm, tilted {tilt:.1f} deg, closest "
          f"to the bare wire {1000 * worst:.1f} mm (touching under {1000 * track.TOUCH:.1f}): "
          f"{'OK' if worst > track.TOUCH and np.linalg.norm(c - c0) < 0.002 else 'FAIL'}")
    # pull the ring sideways off the wire (toward the robot's left), hard: the wire must stop it
    for _ in range(1500):
        p.applyExternalForce(b, -1, [0, 0.5 * S, 0.5 * S], [0, 0, 0], p.LINK_FRAME)
        p.stepSimulation()
    c, rot = ring_pose(b)
    local = c - np.array([0, 0, task.TOP])
    off = np.linalg.norm(local[1:] - track.point(track.progress(local))[1:])
    print(f"pulled up and sideways: the ring's centre is {1000 * off:.1f} mm from the wire (it encircles the wire "
          f"while under {1000 * (track.RING_R - track.TUBE_R):.1f} mm): {'OK' if off < track.RING_R else 'FAIL'}")


if __name__ == "__main__":
    main()
