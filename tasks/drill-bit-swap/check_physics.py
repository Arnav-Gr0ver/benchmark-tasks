"""Physics check of the chuck outside Isaac Sim (pip install pybullet). The drill lies on a table; its thread is the
same chuck.Thread the task uses, stepped at 20 Hz. The sleeve is turned and the bits are moved by stand-ins for the
robot (the sleeve by a motor, a bit by a constraint on its fluted end); the retention test is the one in chuck.py:
an axial pull on the bit, ramped to 50 N, with the drill held by its body only.

    python check_physics.py

World scaled x10 with gravity x10 (same timing), so PyBullet's contact margins stay small.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import chuck as C  # noqa: E402

S = 10.0
DT = 1 / 4000
CONTROL = 200            # sim steps per 20 Hz control step


def q_about_x(deg):
    return p.getQuaternionFromEuler([math.radians(deg), 0, 0])


class Sim:
    def __init__(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81 * S)
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        table = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[S, S, 0.01 * S]),
                                  basePosition=[0, 0, (-C.AXIS_H - 0.01) * S])
        p.changeDynamics(table, -1, lateralFriction=0.5)     # x the drill's 1.0
        (hc, hr, hl), (bc, bs), (fc, fs) = C.HOUSING, C.HANDLE, C.FOOT
        com = np.array(C.BODY_COM)
        y_to_x = p.getQuaternionFromEuler([0, math.pi / 2, 0])     # a cylinder's own z along x
        lands = C.lands()
        body = p.createCollisionShapeArray(
            [p.GEOM_CYLINDER] + [p.GEOM_BOX] * 5, radii=[hr * S] + [0] * 5, lengths=[hl * S] + [0] * 5,
            halfExtents=[[0, 0, 0], [v / 2 * S for v in bs], [v / 2 * S for v in fs]]
            + [[v / 2 * S for v in size] for _, size, _ in lands],
            collisionFramePositions=[list((np.array(c) - com) * S) for c in (hc, bc, fc)]
            + [list((np.array(c) - com) * S) for c, _, _ in lands],
            collisionFrameOrientations=[y_to_x, [0, 0, 0, 1], [0, 0, 0, 1]] + [q_about_x(a) for _, _, a in lands])
        sleeve = p.createCollisionShape(p.GEOM_CYLINDER, radius=C.SLEEVE_R * S,
                                        height=(C.SLEEVE_X[1] - C.SLEEVE_X[0]) * S)
        jaw_len = C.JAW_X[1] - C.JAW_X[0]
        jaw = p.createCollisionShape(p.GEOM_BOX, halfExtents=[jaw_len / 2 * S, C.JAW_T / 2 * S, C.JAW_W / 2 * S])
        jaw_x = (C.JAW_X[0] + C.JAW_X[1]) / 2
        link_pos = [list((np.array([sum(C.SLEEVE_X) / 2, 0, 0]) - com) * S)]
        link_orn = [y_to_x]
        axes = [[0, 0, 1]]
        for k, a in enumerate(C.JAW_ANGLES):
            link_pos.append(list((np.array([jaw_x, 0, 0]) + C.jaw_dir(k) * (C.R0 + C.JAW_T / 2) - com) * S))
            link_orn.append(q_about_x(a))       # the jaw's own y is its radial direction
            axes.append([0, 1, 0])
        n = 4
        self.drill = p.createMultiBody(
            C.BODY_MASS, body, basePosition=list(com * S), linkMasses=[C.SLEEVE_MASS] + [C.JAW_MASS] * 3,
            linkCollisionShapeIndices=[sleeve, jaw, jaw, jaw], linkVisualShapeIndices=[-1] * n,
            linkPositions=link_pos, linkOrientations=link_orn, linkInertialFramePositions=[[0, 0, 0]] * n,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * n, linkParentIndices=[0] * n,
            linkJointTypes=[p.JOINT_REVOLUTE] + [p.JOINT_PRISMATIC] * 3, linkJointAxis=axes)
        for k in range(-1, n):
            p.changeDynamics(self.drill, k, lateralFriction=1.0)
            p.changeDynamics(self.drill, k, collisionMargin=0.0001 * S)
        for k in range(1, 4):
            p.changeDynamics(self.drill, k, jointLowerLimit=(C.R_MIN - C.R0) * S, jointUpperLimit=(C.R_MAX - C.R0) * S)
        for a in range(-1, n):                  # no collisions inside the drill
            for b in range(a + 1, n):
                p.setCollisionFilterPair(self.drill, self.drill, a, b, 0)
        p.setJointMotorControl2(self.drill, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=0.1 * S * S)
        for k in range(1, 4):
            p.setJointMotorControl2(self.drill, k, p.VELOCITY_CONTROL, force=0)     # no built-in motor
        self.thread = C.Thread()
        self.radius = self.thread.radius
        self.bits = {}
        self.hold = None
        self.grab = None
        self.pull = None            # (bit name, force N)
        self.t = 0
        self.sleeve_rate = 0.0

    # -- bodies ---------------------------------------------------------------------------------------------------
    def add_bit(self, name, centre, quat=(0, 0, 0, 1)):
        d, length, flute, mass, _ = C.BITS[name]
        shank = length - flute
        y_to_x = p.getQuaternionFromEuler([0, math.pi / 2, 0])
        kinds, radii, lengths, halves, poss, orns = [p.GEOM_CYLINDER], [d / 2 * S], [shank * S], [[0, 0, 0]], \
            [[(-length / 2 + shank / 2) * S, 0, 0]], [y_to_x]
        for k in range(3):
            kinds.append(p.GEOM_BOX)
            radii.append(0)
            lengths.append(0)
            halves.append([flute / 2 * S, d / 2 * S, d * math.tan(math.radians(30)) / 2 * S])
            poss.append([(length / 2 - flute / 2) * S, 0, 0])
            orns.append(q_about_x(60 * k))
        cs = p.createCollisionShapeArray(kinds, radii=radii, lengths=lengths, halfExtents=halves,
                                         collisionFramePositions=poss, collisionFrameOrientations=orns)
        b = p.createMultiBody(mass, cs, basePosition=list(np.array(centre) * S), baseOrientation=quat)
        # PyBullet multiplies friction: 0.3 against the chuck (as in the task), 0.15 on the table (unimportant)
        p.changeDynamics(b, -1, lateralFriction=C.JAW_FRICTION, collisionMargin=0.0001 * S)
        self.bits[name] = b
        return b

    def pose(self, body, link=-1):
        if link < 0:
            pos, q = p.getBasePositionAndOrientation(body)
        else:
            pos, q = p.getLinkState(body, link)[4:6]
        return np.array(pos) / S, np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)

    def drill_frame(self):
        com, rot = self.pose(self.drill)
        return com - rot @ np.array(C.BODY_COM), rot

    def bit_in_drill(self, name):
        origin, rot = self.drill_frame()
        c, r = self.pose(self.bits[name])
        return rot.T @ (c - origin), rot.T @ r[:, 0]

    def jaw_radii(self):
        return [C.R0 + p.getJointState(self.drill, k)[0] / S for k in range(1, 4)]

    # -- stepping -------------------------------------------------------------------------------------------------
    def control(self):
        self.radius = self.thread.step(p.getJointState(self.drill, 0)[0], self.jaw_radii())

    def jaw_springs(self):
        """The jaws' drives, as the task's implicit drives are: a spring and damper to the thread's radius."""
        for k in range(1, 4):
            q, v = p.getJointState(self.drill, k)[:2]
            force = C.JAW_K * (self.radius - (C.R0 + q / S)) - C.JAW_D * v / S
            p.setJointMotorControl2(self.drill, k, p.TORQUE_CONTROL, force=force * S)

    def step(self, seconds=0.05):
        for _ in range(int(round(seconds / DT))):
            if self.t % CONTROL == 0:
                self.control()
            self.jaw_springs()
            p.setJointMotorControl2(self.drill, 0, p.VELOCITY_CONTROL, targetVelocity=self.sleeve_rate,
                                    force=(5.0 if self.sleeve_rate else 0.1) * S * S)
            if self.pull:
                name, force = self.pull
                origin, rot = self.drill_frame()
                c, _ = self.pose(self.bits[name])
                f = rot[:, 0] * force * S
                p.applyExternalForce(self.bits[name], -1, list(f), list(c * S), p.WORLD_FRAME)
                p.applyExternalForce(self.drill, -1, list(-f), list(c * S), p.WORLD_FRAME)   # held by its body
            p.stepSimulation()
            self.t += 1

    # -- stand-ins for the robot ----------------------------------------------------------------------------------
    def hold_body(self, on=True):
        if on and self.hold is None:
            pos, q = p.getBasePositionAndOrientation(self.drill)
            self.hold = p.createConstraint(self.drill, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos,
                                           childFrameOrientation=q)
            p.changeConstraint(self.hold, maxForce=40 * S)
        elif not on and self.hold is not None:
            p.removeConstraint(self.hold)
            self.hold = None

    def turn_sleeve(self, degrees, seconds=1.5):
        """+: closes."""
        self.sleeve_rate = C.CLOSE_SIGN * math.radians(degrees) / seconds
        self.step(seconds)
        self.sleeve_rate = 0.0
        self.step(0.3)

    def move_bit(self, name, to_centre_in_drill, seconds=1.5, quat=None):
        """Carry a bit by its fluted end to where its middle is at a point in the drill's frame, along the axis."""
        b = self.bits[name]
        origin, rot = self.drill_frame()
        pos, q = p.getBasePositionAndOrientation(b)
        g = p.createConstraint(b, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, childFrameOrientation=q)
        goal = (origin + rot @ np.asarray(to_centre_in_drill)) * S
        n = int(seconds / 0.05)
        for k in range(1, n + 1):
            a = k / n
            p.changeConstraint(g, list((1 - a) * np.array(pos) + a * goal),
                               jointChildFrameOrientation=quat if quat is not None else q, maxForce=20 * S)
            self.step(0.05)
        p.removeConstraint(g)

    def retention_test(self, name):
        """chuck.py's pull on a bit; (passed, most slip mm, still seated)."""
        c0, _ = self.bit_in_drill(name)
        slip = 0.0
        for k in range(int(C.PULL_TOTAL_S / 0.05)):
            self.pull = (name, C.pull_at(k * 0.05))
            self.step(0.05)
            c, _ = self.bit_in_drill(name)
            slip = max(slip, abs(c[0] - c0[0]))
        self.pull = None
        ok = C.seated(name, *self.bit_in_drill(name))
        return slip <= C.SLIP_OK and ok, slip * 1000, ok


def _slip(mm: float) -> str:
    return f"slipped {mm:.2f} mm" if mm < 50 else "pulled out of the chuck"


def report(sim, name, label):
    depth, tilt, off = C.insertion(name, *sim.bit_in_drill(name))
    print(f"  {label}: {name} {1000 * depth:.1f} mm in, tilted {tilt:.1f} deg, {1000 * off:.2f} mm off the axis; jaws at "
          f"{[round(1000 * r, 2) for r in sim.jaw_radii()]} mm, each pushing {sim.thread.clamp_force(sim.jaw_radii()):.0f} N")


STROKE = 65.0


def exchange(close_strokes, label, last=STROKE):
    """The whole exchange as the demo does it; returns whether the new bit passes the retention test."""
    print(label)
    sim = Sim()
    sim.add_bit("old_bit", C.bit_in_chuck("old_bit"))
    sim.step(1.0)
    report(sim, "old_bit", "start")
    sim.hold_body(True)
    passed, slip, _ = sim.retention_test("old_bit")
    print(f"  the old bit, as prepared: retention {'passed' if passed else 'FAILED'} ({_slip(slip)})")
    sim.turn_sleeve(-STROKE)
    sim.hold_body(False)                    # nobody holds the drill while the old bit is drawn out
    sim.step(0.5)
    report(sim, "old_bit", "after one opening stroke")
    o0, _ = sim.drill_frame()
    c, axis = sim.bit_in_drill("old_bit")
    sim.move_bit("old_bit", c + np.array([0.035, 0.0, 0.0]))
    sim.move_bit("old_bit", [0.19, 0.09, 0.03])
    o1, _ = sim.drill_frame()
    print(f"  old bit drawn out; the drill, unheld, moved {1000 * np.linalg.norm(o1 - o0):.2f} mm; removed: "
          f"{C.removed('old_bit', *sim.bit_in_drill('old_bit'))}")
    p.resetBasePositionAndOrientation(sim.bits["old_bit"], [3, 3, 0], [0, 0, 0, 1])
    sim.hold_body(True)
    for _ in range(3):
        sim.turn_sleeve(-STROKE)
    print(f"  after three more: jaws at {[round(1000 * r, 2) for r in sim.jaw_radii()]} mm")
    off = np.array([0.0, 0.0015, -0.0015])  # the hand holds the new bit 2 mm off the axis
    sim.add_bit("new_bit", C.bit_in_chuck("new_bit", -0.02) + off)
    held = HeldBit(sim, "new_bit")
    held.to(C.bit_in_chuck("new_bit") + off, 1.5)
    for k in range(close_strokes):
        if k == 2:
            held.let_go()
            sim.step(0.5)
            report(sim, "new_bit", "let go after two closing strokes")
        sim.turn_sleeve(STROKE if k < close_strokes - 1 else last)
    held.let_go()
    sim.step(0.5)
    report(sim, "new_bit", f"after {close_strokes} closing strokes")
    sim.hold_body(False)
    o0, _ = sim.drill_frame()
    sim.step(2.0)
    o1, _ = sim.drill_frame()
    print(f"  everything let go: the drill moved {1000 * np.linalg.norm(o1 - o0):.2f} mm in 2 s")
    sim.hold_body(True)
    passed, slip, ok = sim.retention_test("new_bit")
    print(f"  retention test: {'passed' if passed else 'failed'} ({_slip(slip)}, seated after: {ok})")
    return passed


class HeldBit:
    """A hand holding a bit by its fluted end: a constraint with the grip's strength (50 N)."""

    def __init__(self, sim, name):
        self.sim, self.b = sim, sim.bits[name]
        self.pos, self.q = p.getBasePositionAndOrientation(self.b)
        self.g = p.createConstraint(self.b, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.pos,
                                    childFrameOrientation=self.q)
        p.changeConstraint(self.g, maxForce=50 * S)

    def to(self, centre_in_drill, seconds):
        origin, rot = self.sim.drill_frame()
        goal = (origin + rot @ np.asarray(centre_in_drill)) * S
        n = int(seconds / 0.05)
        for k in range(1, n + 1):
            a = k / n
            p.changeConstraint(self.g, list((1 - a) * np.array(self.pos) + a * goal), maxForce=50 * S)
            self.sim.step(0.05)
        self.pos = list(goal)

    def let_go(self):
        if self.g is not None:
            p.removeConstraint(self.g)
            self.g = None


def main():
    p.connect(p.DIRECT)
    results = [exchange(4, "Tightened (four 65-degree closing strokes, as the demo):"),
               exchange(2, "Closed short of the bit (two strokes):"),
               exchange(3, "Barely snug (two strokes and 32 degrees: the jaws touch the bit, wound 0.05 mm):", last=32.0)]
    print("OK" if results == [True, False, False] else "FAIL")


if __name__ == "__main__":
    main()
