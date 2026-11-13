"""check_demo: a PyBullet stand-in for yam_arena, for checking a task file's scripted demo before running it in Isaac Sim.

    python check_demo.py [task.py] [--quiet] [--friction F] [--arm-stiffness A] [--sweep]

Needs pybullet and yam_arena (pip install pybullet; set YAM_ARENA_DIR to a checkout if yam_arena is not installed).

What it reproduces from yam_arena (read from its own modules where it can):
- the station table, the two arm bases (station.BASES) and the YAM arm (yam_arm.urdf, the same convex collision pieces);
- joint PD with yam_arena's gains and effort limits plus gravity compensation (actions.py drives the arm that way), the
  fingers on their own PD (both track g x FINGER_OPEN);
- the yam_scripted policy: the task file's demo() through yam_arena.demo.ArmPlan (the same IK), one action per control
  step at 20 Hz, the plan's end plus 3 s;
- housekit Mechs: a base body and one moving body per joint, its spring (stiffness x (target - q) - damping x qdot), its
  stops, collisions between moving pieces but not with their base;
- after every control step, the task's housekit watchers with the same object states and finger points, then the
  efforts they ask for; score() at the end.
Friction: PhysX averages the two materials' coefficients and the robot and table have PhysX's default 0.5; PyBullet
multiplies, so robot and table get 1.0 and each object (0.5 + mu) / 2, which matches every robot and table contact.
Contacts between two objects come out lower here than in Isaac (the product instead of the mean).

It is a check, not the benchmark: contact details differ from PhysX. A demo that fails here will very likely fail
there; one that passes here can still need tuning there.
"""

from __future__ import annotations

import argparse
import importlib
import math
import pathlib
import sys

import numpy as np
import pybullet as p

import os
import tempfile

if os.environ.get("YAM_ARENA_DIR"):                      # a checkout of yam-isaac-lab-arena, if it is not installed
    sys.path.insert(0, os.environ["YAM_ARENA_DIR"])
import yam_arena  # noqa: E402
from yam_arena import constants as K  # noqa: E402
from yam_arena import station, task_file  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402

URDF = pathlib.Path(yam_arena.__file__).resolve().parent / "assets/yam/yam_arm.urdf"


def _urdf_for_pybullet() -> pathlib.Path:
    """PyBullet clamps joint torques to the URDF's effort (1): a copy with the limit lifted (yam_arena has its own
    limits) and absolute mesh paths, in the temp folder."""
    out = pathlib.Path(tempfile.gettempdir()) / "yam_arm_check_demo.urdf"
    text = URDF.read_text().replace('effort="1"', 'effort="1000"')
    text = text.replace('filename="', f'filename="{URDF.parent}/')
    out.write_text(text)
    return out


CONTROL_HZ = 20.0
SUB = 48                        # physics steps per control step: 960 Hz, for the explicit PD and springs
DT = 1.0 / CONTROL_HZ / SUB
ARMS = ("left", "right")


def _q_wxyz_to_xyzw(q):
    w, x, y, z = q
    return (x, y, z, w)


def _rot(q_xyzw):
    return np.array(p.getMatrixFromQuaternion(q_xyzw)).reshape(3, 3)


def _yaw_q(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def _shape_volume_centre(shapes):
    vol, acc = 0.0, np.zeros(3)
    for s in shapes:
        v = s.size[0] * s.size[1] * s.size[2] if s.kind == "box" else math.pi * s.size[0] ** 2 * s.size[2]
        vol += v
        acc += v * np.array(s.at)
    return acc / vol if vol > 0 else np.zeros(3)


def _shape_array(shapes):
    types, half, radii, lengths, pos, orn = [], [], [], [], [], []
    for s in shapes:
        if s.kind == "box":
            types.append(p.GEOM_BOX)
            half.append([v / 2 for v in s.size])
            radii.append(0.0)
            lengths.append(0.0)
        else:
            types.append(p.GEOM_CYLINDER)
            half.append([0, 0, 0])
            radii.append(s.size[0])
            lengths.append(s.size[2])
        pos.append(list(s.at))
        orn.append(list(_q_wxyz_to_xyzw(s.tilt)))
    return p.createCollisionShapeArray(types, halfExtents=half, radii=radii, lengths=lengths,
                                       collisionFramePositions=pos, collisionFrameOrientations=orn)


class Body:
    """One housekit Mech in PyBullet."""

    def __init__(self, spec, hk):
        self.spec, self.hk = spec, hk
        com = _shape_volume_centre(spec.shapes)
        self.joints = list(spec.joints)
        n = len(self.joints)
        base_shape = _shape_array(spec.shapes)
        mass = 0.0 if spec.mass is None else float(spec.mass)
        kw = {}
        if n:
            kw = dict(linkMasses=[float(j.mass) for j in self.joints],
                      linkCollisionShapeIndices=[_shape_array(j.shapes) for j in self.joints],
                      linkVisualShapeIndices=[-1] * n, linkPositions=[list(j.at) for j in self.joints],
                      linkOrientations=[[0, 0, 0, 1]] * n,
                      linkInertialFramePositions=[list(_shape_volume_centre(j.shapes)) for j in self.joints],
                      linkInertialFrameOrientations=[[0, 0, 0, 1]] * n, linkParentIndices=[0] * n,
                      linkJointTypes=[p.JOINT_PRISMATIC if j.kind == "slide" else p.JOINT_REVOLUTE for j in self.joints],
                      linkJointAxis=[list(hk.AXES[j.axis]) for j in self.joints])
        self.id = p.createMultiBody(mass, base_shape, -1, list(spec.pos), _yaw_q(spec.yaw),
                                    baseInertialFramePosition=list(com),
                                    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT, **kw)
        self.com = com
        mu = (0.5 + spec.friction) / 2 * VARIANT["friction"]
        p.changeDynamics(self.id, -1, lateralFriction=mu, collisionMargin=0.0005)
        for i, j in enumerate(self.joints):
            jm = (0.5 + (spec.friction if j.friction is None else j.friction)) / 2 * VARIANT["friction"]
            p.changeDynamics(self.id, i, lateralFriction=jm, collisionMargin=0.0005)
            # (a separate call: PyBullet drops the limits when they come with other arguments)
            p.changeDynamics(self.id, i, jointLowerLimit=j.low, jointUpperLimit=j.high, jointLimitForce=1e4)
            p.setJointMotorControl2(self.id, i, p.VELOCITY_CONTROL, force=0)
        self.efforts: dict[str, float] = {}
        # PhysX drives are implicit; here the spring and damper are explicit, so keep them within what this step
        # length integrates stably (a tiny, heavily damped piece is overdamped either way)
        self.gains = []
        for i, j in enumerate(self.joints):
            info = p.getDynamicsInfo(self.id, i)
            m, inertia, com = info[0], np.array(info[2]), np.array(info[3])
            if j.kind == "slide":
                m_eff = m
            else:
                u = np.array(hk.AXES[j.axis])
                off = com - u * (com @ u)
                m_eff = float(inertia @ (u * u)) + m * float(off @ off)
            k = min(j.spring, 0.8 * m_eff / DT ** 2)
            c = min(j.damping, 0.8 * m_eff / DT)
            if k < j.spring or c < j.damping:
                print(f"  check_demo: {spec.name}.{j.name}: spring {j.spring} -> {k:.3g}, damping {j.damping} -> {c:.3g} "
                      f"(explicit-step limit)")
            self.gains.append((k, c))

    def origin(self):
        pos, q = p.getBasePositionAndOrientation(self.id)
        return np.array(pos) - _rot(q) @ self.com, np.array(q)

    def values(self):
        return {j.name: p.getJointState(self.id, i)[0] for i, j in enumerate(self.joints)}

    def springs(self):
        for i, j in enumerate(self.joints):
            q, v = p.getJointState(self.id, i)[:2]
            k, c = self.gains[i]
            # evaluated a step ahead, as an implicit drive would (PhysX's are implicit): damps the spring's chatter
            tau = k * (j.target - (q + v * DT)) - c * v + self.efforts.get(j.name, 0.0)
            p.setJointMotorControl2(self.id, i, p.TORQUE_CONTROL, force=tau)

    def state(self):
        hk = self.hk
        origin, quat = self.origin()
        vel, ang = p.getBaseVelocity(self.id)
        R = _rot(quat)
        lin = np.array(vel) - np.cross(np.array(ang), R @ self.com)
        values = self.values()
        corners = hk._local_corners(self.spec, values) @ R.T + origin
        low, high = corners.min(0), corners.max(0)
        return hk.MechState(pos=(low + high) / 2, quat_xyzw=quat, lin_vel=lin, ang_vel=np.array(ang), low=low,
                            high=high, joints=values, origin=origin)


VARIANT = {"friction": 1.0, "arm_stiffness": 1.0}   # robustness sweeps: scale every object's friction / the arm's gains


class Arm:
    GAINS = [K.SHOULDER] * 3 + [K.WRIST] * 3

    def __init__(self, name):
        self.name = name
        self.id = p.loadURDF(str(_urdf_for_pybullet()), basePosition=station.BASES[name], useFixedBase=True,
                             flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.j = {p.getJointInfo(self.id, i)[1].decode(): i for i in range(p.getNumJoints(self.id))}
        self.l = {p.getJointInfo(self.id, i)[12].decode(): i for i in range(p.getNumJoints(self.id))}
        self.order = [self.j[n] for n in K.ARM_JOINTS + K.FINGER_JOINTS]
        for i in range(-1, p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, lateralFriction=1.0, collisionMargin=0.0005)
        for k, n in enumerate(K.ARM_JOINTS):
            p.resetJointState(self.id, self.j[n], K.READY[k])
        for n in K.FINGER_JOINTS:
            p.resetJointState(self.id, self.j[n], K.FINGER_OPEN)
        p.setJointMotorControlArray(self.id, self.order, p.VELOCITY_CONTROL, forces=[0] * 8)
        self.target = np.r_[K.READY, K.FINGER_OPEN, K.FINGER_OPEN]
        self.masses = [p.getDynamicsInfo(self.id, i)[0] for i in range(p.getNumJoints(self.id))]
        self._tick, self._inertia = 0, np.ones(8)

    def set_action(self, a7):
        g = float(np.clip(a7[6], 0.0, 1.0))
        self.target = np.r_[a7[:6], g * K.FINGER_OPEN, g * K.FINGER_OPEN]

    def control(self):
        st = p.getJointStates(self.id, self.order)
        q = np.array([s[0] for s in st])
        v = np.array([s[1] for s in st])
        # gravity compensation: lift each link by its weight at its centre of mass (PyBullet's inverse dynamics
        # disagrees with its own forward dynamics on this URDF)
        for i, ls in enumerate(p.getLinkStates(self.id, list(range(len(self.masses))))):
            p.applyExternalForce(self.id, i, [0.0, 0.0, self.masses[i] * 9.81], ls[0], p.WORLD_FRAME)
        # PhysX's joint drives are implicit. An explicit PD at this step length is past its stability limit on the light
        # wrist and fingers (they chatter, and closing the fingers then twists the wrist), so each joint's PD is solved
        # implicitly against the inertia it sees: tau = (k e - (c + k dt) v) / (1 + dt (c + k dt) / M_jj).
        if self._tick % SUB == 0:
            self._inertia = np.diag(np.array(p.calculateMassMatrix(self.id, [float(x) for x in q])))
        self._tick += 1
        ks = VARIANT["arm_stiffness"]
        tau = np.zeros(8)
        for k in range(8):
            g = self.GAINS[k] if k < 6 else K.GRIPPER
            stiff, damp = ks * g["stiffness"], g["damping"]
            raw = (stiff * (self.target[k] - q[k]) - (damp + stiff * DT) * v[k])
            tau[k] = np.clip(raw / (1.0 + DT * (damp + stiff * DT) / self._inertia[k]), -g["effort"], g["effort"])
        p.setJointMotorControlArray(self.id, self.order, p.TORQUE_CONTROL, forces=list(tau))

    def finger_points(self):
        ls = p.getLinkState(self.id, self.l["gripper"], computeForwardKinematics=True)
        pos, R = np.array(ls[4]), _rot(ls[5])
        half_gap = max(abs(p.getJointState(self.id, self.j[n])[0]) for n in K.FINGER_JOINTS)
        pts = []
        for along in (0.0, K.PAD_TIP_PAST_TCP):
            c = pos + R @ (np.array(K.TCP_OFFSET) + [0.0, 0.0, along])
            pts += [c + R[:, 0] * s * (half_gap + 0.006) for s in (1.0, -1.0)]
        return pts

    def tcp(self):
        ls = p.getLinkState(self.id, self.l["gripper"], computeForwardKinematics=True)
        return np.array(ls[4]) + _rot(ls[5]) @ np.array(K.TCP_OFFSET)


class Sim:
    def __init__(self, task_path, verbose=True):
        self.path = pathlib.Path(task_path).resolve()
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=60)
        top = station.TABLE_TOP
        table = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.55, 0.65, 0.02])
        self.table = p.createMultiBody(0, table, -1, [0.36, 0.0, top - 0.02])
        p.changeDynamics(self.table, -1, lateralFriction=1.0)
        self.task = task_file.load(self.path)
        self.hk = importlib.import_module("housekit")
        self.bodies = {}
        for spec in self.task.objects:
            if not isinstance(spec, self.hk.Mech):
                raise TypeError(f"check_demo builds housekit Mechs only, not {spec!r}")
            self.bodies[spec.name] = Body(spec, self.hk)
        self.arms = {a: Arm(a) for a in ARMS}
        self.verbose = verbose
        self.t = 0.0
        self.log = []
        for fn in list(self.hk._RESET_HOOKS):
            if fn.__name__ != "_reset_targets":
                fn(None)
        self.settle(0.5)

    def settle(self, seconds):
        for _ in range(int(round(seconds * CONTROL_HZ))):
            self._physics()

    def _physics(self):
        for _ in range(SUB):
            for arm in self.arms.values():
                arm.control()
            for b in self.bodies.values():
                b.springs()
            p.stepSimulation()

    def states(self, names=None):
        names = names or list(self.bodies)
        return {n: self.bodies[n].state() for n in names}

    def fingers(self):
        return self.arms["left"].finger_points() + self.arms["right"].finger_points()

    def watch(self):
        hk = self.hk
        fingers = self.fingers()
        for names, fn in list(hk._WATCH.values()):
            fn(self.states(list(names)), fingers)
        for b in self.bodies.values():
            b.efforts = dict(hk._EFFORTS.get(b.spec.name, {}))
        hk._EFFORTS.clear()

    def run_demo(self, hold_s=3.0, max_s=None, report_every=5.0, plan=None):
        plan = dict(self.task.demo() if plan is None else plan)
        plans = {a: ArmPlan(list(plan[a]), station.BASES[a]) for a in plan}
        end = max(pl.duration for pl in plans.values()) + hold_s
        if max_s is not None:
            end = min(end, max_s)
        dt = 1.0 / CONTROL_HZ
        steps = 0
        next_report = 0.0
        while self.t <= end:
            t = self.t + dt
            for a in ARMS:
                if a in plans:
                    self.arms[a].set_action(plans[a].action7(t)[0])
            self._physics()
            self.t += dt
            steps += 1
            self.watch()
            if self.verbose and self.t >= next_report:
                next_report += report_every
                labels = {a: plans[a].target(self.t)[3] for a in plans}
                print(f"  t {self.t:6.1f}s  score {self.score():.2f}  {labels}", flush=True)
        worst = {a: (round(pl.worst_ik_mm, 2), round(pl.worst_ik_deg, 2)) for a, pl in plans.items()}
        return steps, worst

    def score(self):
        return float(self.task.score(self.states([o.name for o in self.task.objects])))


SWEEP = [(1.0, 1.0), (0.7, 1.0), (1.4, 1.0), (1.0, 0.7), (1.0, 1.4), (0.7, 0.7), (1.4, 1.4)]


def _sweep(task, extra):
    """The demo (or a counterexample) under seven friction x arm-stiffness variants, one process each."""
    import subprocess

    runs = []
    for fr, st in SWEEP:
        cmd = [sys.executable, __file__, task, "--quiet", "--friction", str(fr), "--arm-stiffness", str(st)] + extra
        runs.append((fr, st, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    ok = True
    for fr, st, proc in runs:
        last = proc.communicate()[0].strip().splitlines()[-1]
        print(f"friction x{fr}, arm stiffness x{st}: {last}")
        ok = ok and "MISMATCH" not in last and "Traceback" not in last
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default=str(pathlib.Path(__file__).resolve().parent / "task.py"))
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--friction", type=float, default=1.0, help="scale every object's friction")
    ap.add_argument("--arm-stiffness", type=float, default=1.0, help="scale the arm joints' stiffness")
    ap.add_argument("--counter", default=None, help="play one of the task file's COUNTEREXAMPLES instead of demo()")
    ap.add_argument("--list", action="store_true", help="list the task file's counterexamples")
    ap.add_argument("--sweep", action="store_true", help="run under seven friction and arm-stiffness variants")
    a = ap.parse_args()
    if a.sweep:
        sys.exit(0 if _sweep(a.task, ["--counter", a.counter] if a.counter else []) else 1)
    VARIANT.update(friction=a.friction, arm_stiffness=a.arm_stiffness)
    sys.path.insert(0, str(pathlib.Path(a.task).resolve().parent))
    if a.list:
        counters = getattr(task_file.load(a.task).demo, "__globals__", {}).get("COUNTEREXAMPLES", {})
        for name, (fn, expect, why) in counters.items():
            print(f"{name}: expect {expect}. {why}")
        return
    sim = Sim(a.task, verbose=not a.quiet)
    counters = getattr(sim.task.demo, "__globals__", {}).get("COUNTEREXAMPLES", {})
    plan, expect = None, 1.0
    if a.counter:
        fn, expect, _ = counters[a.counter]
        plan = fn()
    steps, worst = sim.run_demo(max_s=a.seconds, plan=plan)
    print(f"{a.counter or 'demo'}: {steps} control steps ({steps / CONTROL_HZ:.1f} s); worst IK (mm, deg): {worst}")
    got = sim.score()
    print(f"final score: {got}" + ("" if abs(got - expect) < 1e-9 else f"  MISMATCH (expected {expect})"))


if __name__ == "__main__":
    main()
