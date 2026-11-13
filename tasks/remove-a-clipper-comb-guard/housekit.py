"""housekit: multi-part objects, spring joints and per-step watchers for yam_arena task files.

Built on mechkit (from the accepted task hang-a-blind-headrail-in-gated-brackets, unchanged in how it builds
objects in Isaac Sim), with three additions at the end of this file: a joint's spring can aim past a stop to preload
it (``Joint.target``), watchers that run after every control step with the objects' states and the fingers'
positions (``every_step``), and joint efforts a watcher can apply (``effort``).

Original mechkit description:

yam_arena task files hold single primitives. A ``Mech`` is one object built from several boxes and cylinders, which can
carry moving pieces: a ``Joint`` is a slide or a hinge, with end stops and an optional spring that returns it to its rest
position. That is enough for hollow fixtures, sockets, keyed parts, drawers, doors, lids, spring latches and buttons.

    from mechkit import Mech, Joint, box, cyl

    DRAWER = Mech("cabinet", pos=(0.25, 0.0, TOP), mass=None,
                  shapes=[box((0, 0, 0.05), (0.2, 0.2, 0.01), GREY)],
                  joints=[Joint("drawer", "slide", axis="x", at=(0, 0, 0.03), low=-0.12, high=0.0,
                                shapes=[box((0, 0, 0), (0.16, 0.16, 0.03), BLUE)])])
    OBJECTS = [DRAWER, ...]

    def score(objects):
        return 1.0 if objects["cabinet"].joints["drawer"] < -0.1 else 0.0

Positions in a Mech are metres from its ``pos``, turned by its ``yaw``. Joint values are measured from rest: metres for a
slide, radians for a hinge, positive along ``axis`` by the right-hand rule. ``mass=None`` fixes the object to the world.

A sprung piece sags a little under its own weight before anything touches it, by mass x g / spring for a slide, so give
score() a margin: measured in Isaac, a 30 g button on 150 N/m sits at -0.002 m. Fixed and free Mechs, slides and hinges have
all been run in Isaac: buttons press and return, hinges turn, multi-part blocks and jointed objects lift and carry.

The task file runs with the standard ``yam_arena`` embodiment and ``task_score`` scorer: importing this module adds Mech
support to ``yam_arena.scene`` as that module loads, and changes nothing else. Keep this file beside the task file.
Importing it needs only NumPy and yam_arena; everything that needs Isaac Sim is imported when the simulator builds the scene.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Mech", "Joint", "MechState", "Shape", "box", "cyl", "on_reset", "every_step", "store", "effort",
           "finger_points"]

_RESET_HOOKS: list[Any] = []


def on_reset(fn):
    """Register ``fn(env)`` to run whenever a yam_arena episode resets (clears latched scorer state across episodes)."""
    _RESET_HOOKS.append(fn)
    return fn

Vec3 = tuple[float, float, float]
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
GREY = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class Shape:
    """A collider that also shows: a box (``size`` x, y, z) or an upright cylinder (``size`` radius, radius, height)."""

    kind: str
    at: Vec3
    size: Vec3
    color: Vec3 = GREY
    tilt: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # rotation in the part frame, (w, x, y, z)


def box(at, size, color=GREY, tilt=(1.0, 0.0, 0.0, 0.0)) -> Shape:
    return Shape("box", tuple(map(float, at)), tuple(map(float, size)), tuple(color), tuple(map(float, tilt)))


def cyl(at, radius, height, color=GREY) -> Shape:
    return Shape("cyl", tuple(map(float, at)), (float(radius), float(radius), float(height)), tuple(color))


@dataclass(frozen=True)
class Joint:
    """A moving piece. ``at`` is the joint's origin in the Mech frame and the moving piece's own frame at rest. ``low`` and
    ``high`` are its stops, which must include 0 (rest). ``spring`` is N/m or N m/rad toward rest; 0 leaves it free."""

    name: str
    kind: str  # "slide" or "hinge"
    shapes: tuple[Shape, ...]
    axis: str = "z"
    at: Vec3 = (0.0, 0.0, 0.0)
    low: float = 0.0
    high: float = 0.01
    spring: float = 0.0
    damping: float = 5.0
    mass: float = 0.02
    friction: float | None = None
    target: float = 0.0  # where the spring pulls toward; past a stop, it preloads the piece against that stop

    def __post_init__(self) -> None:
        if self.kind not in ("slide", "hinge"):
            raise ValueError(f"joint {self.name}: kind must be slide or hinge, got {self.kind!r}")
        if self.axis not in AXES:
            raise ValueError(f"joint {self.name}: axis must be x, y or z, got {self.axis!r}")
        if not self.low <= 0.0 <= self.high or self.low == self.high:
            raise ValueError(f"joint {self.name}: stops {self.low}..{self.high} must surround 0, the rest position")
        if not self.shapes:
            raise ValueError(f"joint {self.name}: needs at least one shape")
        object.__setattr__(self, "shapes", tuple(self.shapes))


@dataclass(frozen=True)
class Mech(Box):
    """A multi-part object. Subclasses Box because task files accept only the primitive types; ``size`` is filled in as
    the bounding box at rest. Give ``mass`` for a free object, None to fix it to the world."""

    shapes: tuple[Shape, ...] = ()
    joints: tuple[Joint, ...] = ()
    friction: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shapes", tuple(self.shapes))
        object.__setattr__(self, "joints", tuple(self.joints))
        if not self.shapes:
            raise ValueError(f"mech {self.name}: needs at least one shape")
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"mech {self.name}: joint names repeat: {names}")
        corners = _local_corners(self, {j.name: 0.0 for j in self.joints})
        object.__setattr__(self, "size", tuple(float(v) for v in corners.max(0) - corners.min(0)))
        self._check()


@dataclass(frozen=True)
class MechState(ObjectState):
    """An ObjectState plus each joint's value, and helpers to read a point in the Mech's own frame."""

    joints: dict[str, float] = field(default_factory=dict)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def rotation(self) -> np.ndarray:
        return _quat_xyzw_matrix(self.quat_xyzw)

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, dtype=float) - self.origin)

    def to_world(self, local_point) -> np.ndarray:
        return self.rotation() @ np.asarray(local_point, dtype=float) + self.origin


# -- geometry, pure NumPy ----------------------------------------------------------------------------------------------


def _quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat_xyzw_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return _quat_wxyz_matrix((w, x, y, z))


def _axis_matrix(axis: str, angle: float) -> np.ndarray:
    half = angle / 2
    v = np.array(AXES[axis]) * math.sin(half)
    return _quat_wxyz_matrix((math.cos(half), *v))


def _shape_corners(shape: Shape) -> np.ndarray:
    sx, sy, sz = shape.size
    half = np.array([sx, sy, sz / 2]) if shape.kind == "cyl" else np.array([sx, sy, sz]) / 2
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
    return signs * half @ _quat_wxyz_matrix(shape.tilt).T + np.array(shape.at)


def _joint_pose(joint: Joint, value: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and origin of a moving piece in the Mech frame at joint position ``value``."""
    at = np.array(joint.at)
    if joint.kind == "slide":
        return np.eye(3), at + np.array(AXES[joint.axis]) * value
    return _axis_matrix(joint.axis, value), at


def _local_corners(spec: Mech, values: dict[str, float]) -> np.ndarray:
    pts = [_shape_corners(s) for s in spec.shapes]
    for joint in spec.joints:
        rot, origin = _joint_pose(joint, values.get(joint.name, 0.0))
        pts += [_shape_corners(s) @ rot.T + origin for s in joint.shapes]
    return np.vstack(pts)


# -- building it in Isaac Sim ------------------------------------------------------------------------------------------

_MECHS: dict[str, Mech] = {}


def _author(prim_path: str, cfg: Any, translation=None, orientation=None, **_: Any):
    """Write the Mech to USD: an Xform holding a base body and one body per joint."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _MECHS[cfg.mech_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    looks: dict[Vec3, str] = {}

    def look(color: Vec3) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/color{len(looks)}"
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.4)
            mat.func(path, mat)
            looks[color] = path
        return looks[color]

    def material(tag: str, mu: float) -> str:
        path = f"{prim_path}/Looks/{tag}"
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu)
        mat.func(path, mat)
        return path

    def add_shape(body: str, i: int, shape: Shape, phys: str) -> None:
        path = f"{body}/shape{i}"
        if shape.kind == "cyl":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(shape.size[0])
            geom.CreateHeightAttr(shape.size[2])
            geom.CreateAxisAttr("Z")
            scale = Gf.Vec3f(1.0, 1.0, 1.0)
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            scale = Gf.Vec3f(*shape.size)
        geom.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*shape.at))
        tw, tx, ty, tz = shape.tilt
        geom.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(tw, tx, ty, tz))
        geom.AddScaleOp().Set(scale)
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(0.0015)
        contact.CreateRestOffsetAttr(0.0)
        bind_visual_material(path, look(shape.color))
        bind_physics_material(path, phys)

    def add_body(name: str, at, mass: float | None) -> str:
        path = f"{prim_path}/{name}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*at))
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        if mass is not None:
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(mass))
        return path

    fixed = spec.mass is None
    base_mass = (spec.mass if not fixed else (1.0 if spec.joints else None))
    base = add_body("base", (0.0, 0.0, 0.0), base_mass)
    phys = material("friction", spec.friction)
    for i, shape in enumerate(spec.shapes):
        add_shape(base, i, shape, phys)

    if spec.joints:
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(base))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(base)).CreateEnabledSelfCollisionsAttr(True)
    if fixed and spec.joints:
        anchor = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor")
        anchor.CreateBody1Rel().SetTargets([Sdf.Path(base)])
        ax, ay, az = translation or (0.0, 0.0, 0.0)
        anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(ax, ay, az))
        anchor.CreateLocalRot0Attr().Set(Gf.Quatf(w, x, y, z))
        anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        anchor.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    for joint in spec.joints:
        body = add_body(joint.name, joint.at, joint.mass)
        mu = spec.friction if joint.friction is None else joint.friction
        own = phys if joint.friction is None else material(f"friction_{joint.name}", mu)
        for i, shape in enumerate(joint.shapes):
            add_shape(body, i, shape, own)
        kind = UsdPhysics.PrismaticJoint if joint.kind == "slide" else UsdPhysics.RevoluteJoint
        j = kind.Define(stage, f"{prim_path}/{joint.name}_joint")
        j.CreateBody0Rel().SetTargets([Sdf.Path(base)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body)])
        j.CreateAxisAttr(joint.axis.upper())
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(*joint.at))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        degrees = joint.kind == "hinge"
        j.CreateLowerLimitAttr(math.degrees(joint.low) if degrees else joint.low)
        j.CreateUpperLimitAttr(math.degrees(joint.high) if degrees else joint.high)
        UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if degrees else "linear")
    return stage.GetPrimAtPath(prim_path)


def _arena_object(spec: Mech):
    """The Mech as an Arena object; Isaac Lab's implicit actuators carry the springs."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class MechSpawnerCfg(SpawnerCfg):
        func: object = clone(_author)
        mech_name: str = ""

    class MechObject(Object):
        def _generate_articulation_cfg(self):
            def spring(j: Joint) -> ImplicitActuatorCfg:
                return ImplicitActuatorCfg(joint_names_expr=[f"{j.name}_joint"], stiffness=j.spring, damping=j.damping)

            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg,
                                  actuators={j.name: spring(j) for j in spec.joints})
            return self._add_initial_pose_to_cfg(cfg)

    _MECHS[spec.name] = spec
    if spec.joints:
        kind = ObjectType.ARTICULATION
    else:
        kind = ObjectType.RIGID if spec.mass is not None else ObjectType.BASE
    half = spec.yaw / 2
    return MechObject(name=spec.name, object_type=kind, spawner_cfg=MechSpawnerCfg(mech_name=spec.name),
                      initial_pose=Pose(position_xyz=tuple(spec.pos),
                                        rotation_xyzw=(0.0, 0.0, math.sin(half), math.cos(half))))


def _to_numpy(value) -> np.ndarray:
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy().astype(np.float64)


def _mech_state(environment, spec: Mech) -> MechState:
    base = environment.unwrapped
    values: dict[str, float] = {}
    origin = np.array(spec.pos, dtype=float)
    half = spec.yaw / 2
    quat = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
    lin, ang = np.zeros(3), np.zeros(3)
    entity = base.scene[spec.name] if spec.name in {*base.scene.rigid_objects, *base.scene.articulations} else None
    if entity is not None:
        env0 = _to_numpy(base.scene.env_origins)[0]
        pose = _to_numpy(entity.data.root_link_pose_w)[0]
        origin, quat = pose[:3] - env0, pose[3:]
        lin = _to_numpy(entity.data.root_link_lin_vel_w)[0]
        ang = _to_numpy(entity.data.root_link_ang_vel_w)[0]
        if spec.joints:
            pos = _to_numpy(entity.data.joint_pos)[0]
            values = {n.removesuffix("_joint"): float(v) for n, v in zip(entity.joint_names, pos, strict=True)}
    corners = _local_corners(spec, values) @ _quat_xyzw_matrix(quat).T + origin
    low, high = corners.min(0), corners.max(0)
    return MechState(pos=(low + high) / 2, quat_xyzw=quat, lin_vel=lin, ang_vel=ang, low=low, high=high,
                     joints=values, origin=origin)


def _patch_scene(scene) -> None:
    plain_object, plain_states, plain_make = scene.task_object, scene.object_states, scene.make_env

    def task_object(spec):
        return _arena_object(spec) if isinstance(spec, Mech) else plain_object(spec)

    def object_states(environment, names):
        mechs = {n: _MECHS[n] for n in names if n in _MECHS}
        out = plain_states(environment, [n for n in names if n not in mechs])
        out.update({n: _mech_state(environment, spec) for n, spec in mechs.items()})
        return out

    def make_env(*args, **kwargs):
        env = plain_make(*args, **kwargs)
        orig_reset = env.reset

        def reset(*r_args, **r_kwargs):
            res = orig_reset(*r_args, **r_kwargs)
            for fn in list(_RESET_HOOKS):
                fn(env)
            return res

        env.reset = reset
        return env

    scene.task_object, scene.object_states, scene.make_env = task_object, object_states, make_env


# -- additions: spring targets, per-step watchers, joint efforts ---------------------------------------------------------

_WATCH: dict[str, tuple[tuple[str, ...], Any]] = {}
_STORES: dict[str, dict] = {}
_EFFORTS: dict[str, dict[str, float]] = {}
_APPLIED: set[str] = set()


def store(key: str) -> dict:
    """A dict kept here, in this module, under ``key``: a task file's watcher and score() share state through it even
    when yam_arena imports the task file more than once."""
    return _STORES.setdefault(key, {})


def every_step(key: str, names):
    """Register ``fn(objects, fingers)`` to run after every control step, before the task file is scored. ``objects``
    holds the named objects' states (as score() gets them); ``fingers`` is finger_points(). Registering again under the
    same key replaces the earlier watcher."""

    def deco(fn):
        _WATCH[key] = (tuple(names), fn)
        return fn

    return deco


def effort(mech: str, joint: str, value: float) -> None:
    """Apply ``value`` (N for a slide, N m for a hinge, along the joint's axis) to a joint during the next control
    step. Call it every step the effort should last; a joint not given one gets none."""
    _EFFORTS.setdefault(mech, {})[joint] = float(value)


def finger_points(environment) -> list:
    """Points on both arms' finger pads (world, relative to the env origin): at the TCP and at the pad tips, 6 mm
    outside each pad face (from partkit, accepted with snapfit-close)."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    base = environment.unwrapped
    origin = _to_numpy(base.scene.env_origins)[0]
    pts = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _to_numpy(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        jp, jn = _to_numpy(robot.data.joint_pos)[0], list(robot.joint_names)
        half_gap = max(abs(jp[jn.index(j)]) for j in ("joint7", "joint8") if j in jn)
        r = _quat_xyzw_matrix(pose[3:7])
        for along in (0.0, PAD_TIP_PAST_TCP):
            centre = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, along]) - origin
            pts += [centre + r[:, 0] * side * (half_gap + 0.006) for side in (1.0, -1.0)]
    return pts


def _joint_ids(asset, spec: Mech) -> list[int]:
    names = list(asset.joint_names)
    return [names.index(f"{j.name}_joint") for j in spec.joints]


def _set_targets(environment) -> None:
    """Point each preloaded spring at its target (scene resets clear the targets, so this runs after every reset and
    every step)."""
    import torch

    base = environment.unwrapped
    for spec in _MECHS.values():
        if not spec.joints or all(j.target == 0.0 for j in spec.joints):
            continue
        asset = base.scene[spec.name]
        ids = _joint_ids(asset, spec)
        device = getattr(asset.data.joint_pos, "torch", asset.data.joint_pos).device
        t = torch.tensor([[j.target for j in spec.joints]], dtype=torch.float32, device=device)
        if hasattr(asset, "set_joint_position_target_index"):
            asset.set_joint_position_target_index(target=t, joint_ids=ids)
        else:
            asset.set_joint_position_target(t, joint_ids=ids)


def _apply_efforts(environment) -> None:
    import torch

    base = environment.unwrapped
    for name in set(_EFFORTS) | _APPLIED:
        spec = _MECHS.get(name)
        if spec is None or not spec.joints:          # only jointed Mechs are articulations with joints to drive
            continue
        asset = base.scene[name]
        ids = _joint_ids(asset, spec)
        values = _EFFORTS.get(name, {})
        device = getattr(asset.data.joint_pos, "torch", asset.data.joint_pos).device
        t = torch.tensor([[values.get(j.name, 0.0) for j in spec.joints]], dtype=torch.float32, device=device)
        if hasattr(asset, "set_joint_effort_target_index"):
            asset.set_joint_effort_target_index(target=t, joint_ids=ids)
        else:
            asset.set_joint_effort_target(t, joint_ids=ids)
    _APPLIED.clear()
    _APPLIED.update(n for n, v in _EFFORTS.items() if v)
    _EFFORTS.clear()


def _after_step(environment) -> None:
    from yam_arena.scene import object_states

    _set_targets(environment)
    fingers = finger_points(environment)
    for names, fn in list(_WATCH.values()):
        fn(object_states(environment, list(names)), fingers)
    _apply_efforts(environment)


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_housekit_patched", False):
        return
    base_task_step = cls._task_step

    def _task_step(self, *a, **k):
        _after_step(self._env)
        return base_task_step(self, *a, **k)

    cls._task_step = _task_step
    cls._housekit_patched = True


def _reset_targets(environment) -> None:
    _EFFORTS.clear()
    _APPLIED.update(n for n, s in _MECHS.items() if s.joints)
    _apply_efforts(environment)
    _set_targets(environment)


_RESET_HOOKS.insert(0, _reset_targets)


class _AfterImport(importlib.abc.MetaPathFinder):
    """Patches yam_arena.scene and yam_arena.inspect_embodiment as they load (they need Isaac Sim)."""

    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        run, patch = spec.loader.exec_module, self.targets[fullname]

        def exec_module(module):
            run(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


for _name, _patch in _AfterImport.targets.items():
    if _name in sys.modules:
        _patch(sys.modules[_name])
if not any(type(f).__name__ == "_AfterImport" for f in sys.meta_path):
    sys.meta_path.insert(0, _AfterImport())
