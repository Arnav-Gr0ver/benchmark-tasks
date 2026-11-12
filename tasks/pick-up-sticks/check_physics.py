"""Physics check outside Isaac Sim (pip install pybullet). Builds the pile, lets it settle, and replays two ways of
taking the red stick out with a stand-in gripper (two 8 mm fingers that close on the stick, then carry it):

  blue first  - lift the blue stick straight up by its middle and lay it on the mat, then the red one (the demo);
  red first   - lift the red stick straight up by its middle with the blue one still on it.

For each it reports how far the protected sticks (green, yellow, black) moved and turned at worst (limit: 3 mm,
3 degrees). World scaled x10 with gravity x10 (same timing), so PyBullet's contact margins stay small.

    python check_physics.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _standins import install  # noqa: E402

install()
import task  # noqa: E402

S = 10.0
DT = 1 / 1000
FINGER = (0.008, 0.020, 0.030)       # thickness (along the grip), width, height (m)


def build():
    p.resetSimulation()
    p.setGravity(0, 0, -9.81 * S)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=100)
    table = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5 * S, 0.5 * S, 0.01 * S]),
                              basePosition=[0.3 * S, 0, (task.TOP - 0.01) * S])
    p.changeDynamics(table, -1, lateralFriction=0.5)
    mat = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in task.MAT.size]),
                            basePosition=[v * S for v in task.MAT.pos])
    p.changeDynamics(mat, -1, lateralFriction=0.5)
    ids = {}
    for st in task.STICKS:
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in st.size])
        b = p.createMultiBody(task.MASS, cs, basePosition=[v * S for v in st.pos],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, st.yaw]))
        p.changeDynamics(b, -1, lateralFriction=task.FRICTION, collisionMargin=0.0002 * S)
        ids[st.name] = b
    fingers = []
    for _ in range(2):
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2 * S for v in FINGER])
        f = p.createMultiBody(0, cs, basePosition=[0, 0, 2 * S])
        p.changeDynamics(f, -1, lateralFriction=1.0)
        fingers.append(f)
    return ids, fingers


def pose(b):
    pos, q = p.getBasePositionAndOrientation(b)
    return np.array(pos) / S, np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


class Run:
    def __init__(self):
        self.ids, self.fingers = build()
        for _ in range(1000):
            p.stepSimulation()
        self.ref = {n: pose(self.ids[n]) for n in task.PROTECTED}
        self.worst = (0.0, 0.0)
        self.grip = None

    def _fingers(self, centre, yaw, gap):
        """Place the fingers: their tips at centre's height, ``gap`` apart, closing along the direction at angle
        ``yaw`` from x (across the stick)."""
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        q = p.getQuaternionFromEuler([0, 0, yaw])
        for s, f in zip((-1, 1), self.fingers):
            c = np.asarray(centre) + s * d * (gap / 2 + FINGER[0] / 2) + np.array([0, 0, FINGER[2] / 2])
            p.resetBasePositionAndOrientation(f, list(c * S), q)

    def step(self):
        p.stepSimulation()
        for n in task.PROTECTED:
            (p0, r0), (p1, r1) = self.ref[n], pose(self.ids[n])
            turn = math.degrees(math.acos(max(-1, min(1, (np.trace(r1 @ r0.T) - 1) / 2))))
            self.worst = (max(self.worst[0], np.linalg.norm(p1 - p0) * 1000), max(self.worst[1], turn))

    def move(self, a, b, yaw, gap, seconds, carry=None):
        n = int(seconds / DT)
        for k in range(1, n + 1):
            t = k / n
            s = t * t * (3 - 2 * t)
            c = (1 - s) * np.asarray(a) + s * np.asarray(b)
            self._fingers(c, yaw, gap)
            if self.grip is not None:
                p.changeConstraint(self.grip, list(c * S + np.array([0, 0, 0.002 * S])), maxForce=50 * S)
            self.step()

    def take(self, name, place_xy, yaw_pick, yaw_place):
        """Fingers come down beside the stick's middle, close, the stick is carried up and over to the mat."""
        b = self.ids[name]
        (c0, _) = pose(b)
        grip_pt = c0 - np.array([0, 0, 0.002])          # fingertips 2 mm below the stick's centre
        above = grip_pt + np.array([0, 0, 0.06])
        open_gap, closed_gap = 2 * 0.0117, task.SIDE
        self.move(above, above, yaw_pick, open_gap, 0.05)
        self.move(above, grip_pt, yaw_pick, open_gap, 1.0)
        self.move(grip_pt, grip_pt, yaw_pick, closed_gap, 0.3)
        self.grip = p.createConstraint(b, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                       list((grip_pt + np.array([0, 0, 0.002])) * S),
                                       childFrameOrientation=p.getBasePositionAndOrientation(b)[1])
        self.move(grip_pt, above, yaw_pick, closed_gap, 1.2)
        dest = np.array([*place_xy, task.TOP + task.MAT_T + task.SIDE / 2 + 0.002 - 0.002])
        high = np.array([*place_xy, above[2]])
        self.move(above, high, yaw_pick, closed_gap, 1.5)
        q = p.getQuaternionFromEuler([0, 0, yaw_place])
        p.changeConstraint(self.grip, list(high * S), jointChildFrameOrientation=q, maxForce=50 * S)
        self.move(high, dest, yaw_place, closed_gap, 1.0)
        p.removeConstraint(self.grip)
        self.grip = None
        self.move(dest, dest + np.array([0, 0, 0.06]), yaw_place, open_gap, 0.8)


def main():
    p.connect(p.DIRECT)
    for label, order in (("blue first (the demo)", ("blue", "red")), ("red first", ("red",))):
        r = Run()
        settle = r.worst
        for name in order:
            dest = task.BLUE_TO if name == "blue" else task.RED_TO
            # the fingers close across the stick: along x for the blue one (it lies along y), along y for the red one;
            # both end up lying along x on the mat
            r.take(name, dest, *(0.0, math.pi / 2) if name == "blue" else (math.pi / 2, math.pi / 2))
        for _ in range(500):
            r.step()
        red_on_mat = task.on_mat(type("S", (), {"pos": pose(r.ids["red"])[0]})())
        ok = r.worst[0] <= 3.0 and r.worst[1] <= 3.0
        print(f"{label}: protected sticks moved at most {r.worst[0]:.2f} mm, turned {r.worst[1]:.2f} deg "
              f"({'within' if ok else 'BEYOND'} the 3 mm / 3 deg limit); red on the mat: {red_on_mat}"
              f"{'' if settle == (0.0, 0.0) else ''}")


if __name__ == "__main__":
    main()
