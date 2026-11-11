"""mazekit: the tilt-maze board and its ball as objects for a yam_arena task file.

yam_arena's task files spawn single primitives. This module adds two object kinds and reads their state:

- :class:`Board`: one rigid body with many colliders (base plate, the walls of ``maze.wall_boxes()``, an invisible
  lid), held to the world by a joint at its gimbal's pivot: free to tilt about both horizontal axes, not free to slide,
  lift or turn about the vertical. The stand under it (a plain fixed cylinder in the task file) stops the tilt at
  ``maze.TILT_STOP``; the joint's own limit, a little further, is only a backstop. Its centre of mass hangs below the pivot, so
  gravity levels it when nothing presses on it.
- :class:`Ball`: a rigid sphere with rolling resistance (damping), which yam_arena's ``Sphere`` cannot set.

Both report ``mass=None`` to yam_arena so that a seeded reset never shifts them (``-E object_jitter``): the board is
pinned by its joint and the ball must start on its mark.

Importing this module hooks ``yam_arena.scene`` without changing yam_arena: ``success()`` and ``score()`` receive a
:class:`BodyState` (the body origin's pose, not a bounding-box centre) for these objects.

Pure Python at import time; everything that needs Isaac Sim is imported lazily, inside the functions.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, ObjectState, Sphere

import maze

__all__ = ["Ball", "Board", "BodyState", "install"]

Color = tuple[float, float, float]


@dataclass(frozen=True)
class Board(Box):
    """The maze board. ``pos`` is the centre of the maze floor (the board frame's origin) with the board level;
    ``size`` is only nominal. It is always loose on its pivot; ``mass`` stays None (see the module docstring)."""

    body_mass: float = maze.BOARD_MASS
    floor_color: Color = (0.90, 0.90, 0.87)
    wall_color: Color = (0.10, 0.10, 0.12)
    rim_color: Color = (0.45, 0.47, 0.50)
    goal_color: Color = (0.10, 0.65, 0.25)
    friction: float = 0.6        # floor and walls against the ball
    lid_friction: float = 0.3    # the lid against fingertips
    rocking_damping: float = 4.0  # 1/s, angular: the gimbal's pin friction, so a released board settles, not swings


@dataclass(frozen=True)
class Ball(Sphere):
    """The ball. ``pos`` is its centre. ``mass`` stays None (see the module docstring)."""

    ball_mass: float = maze.BALL_MASS
    friction: float = 0.6
    linear_damping: float = 0.3   # 1/s: rolling resistance; the planner's model assumes about 1.0 per second in all
    angular_damping: float = 1.0  # 1/s


@dataclass(frozen=True)
class BodyState(ObjectState):
    """The body origin's pose and velocities in the world (m, m/s, rad/s)."""

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def tilt(self) -> np.ndarray:
        """The body's tilt as a vector (rad): it points downhill, and its length is the tilt angle."""
        n = self.rotation()[:, 2]
        flat = float(np.hypot(n[0], n[1]))
        if flat < 1e-9:
            return np.zeros(2)
        return n[:2] / flat * math.atan2(flat, n[2])


# -- spawning (pxr only) -------------------------------------------------------------------------------------------

_SPECS: dict[str, Board | Ball] = {}


def _try(fn) -> None:
    """Author an optional attribute; older schema builds may lack it."""
    try:
        fn()
    except AttributeError:
        pass


def _root(stage, prim_path: str, translation, orientation):
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    return root


def _body(stage, path: str, mass: float, com=None, linear_damping=0.0, angular_damping=0.0, iters=32):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    rb.CreateSolverPositionIterationCountAttr(iters)
    rb.CreateLinearDampingAttr(linear_damping)
    rb.CreateAngularDampingAttr(angular_damping)
    _try(lambda: rb.CreateSleepThresholdAttr(0.0))  # a resting ball must wake when the board under it tilts
    _try(lambda: rb.CreateMaxAngularVelocityAttr(20000.0))  # deg/s: a 16 mm ball at 0.5 m/s spins at 3600
    return prim


class _Materials:
    """Preview-surface and physics materials under ``{prim_path}/Looks``, made on first use."""

    def __init__(self, stage, prim_path: str):
        self.stage, self.path = stage, f"{prim_path}/Looks"
        self.looks: dict[Color, str] = {}
        self.physics: dict[tuple[float, bool], str] = {}

    def look(self, color: Color) -> str:
        import isaaclab.sim as sim_utils

        from yam_arena.station import linear_rgb

        if color not in self.looks:
            path = f"{self.path}/c{len(self.looks)}"
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            cfg.func(path, cfg)
            self.looks[color] = path
        return self.looks[color]

    def friction(self, mu: float, combine_min: bool = False) -> str:
        import isaaclab.sim as sim_utils

        key = (mu, combine_min)
        if key not in self.physics:
            path = f"{self.path}/p{len(self.physics)}"
            cfg = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu,
                                                 friction_combine_mode="min" if combine_min else "average")
            cfg.func(path, cfg)
            self.physics[key] = path
        return self.physics[key]


def _collide(stage, prim, physics_material: str) -> None:
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    cp.CreateContactOffsetAttr(0.002)
    cp.CreateRestOffsetAttr(0.0)
    bind_physics_material(str(prim.GetPath()), physics_material, stage=stage)


def _paint(stage, prim, look: str) -> None:
    from isaaclab.sim.utils import bind_visual_material

    bind_visual_material(str(prim.GetPath()), look, stage=stage)


def _disc(stage, path: str, radius: float, z0: float, z1: float):
    """An upright cylinder from height z0 to z1 in the body frame."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(z1 - z0)
    cyl.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, (z0 + z1) / 2))
    return cyl


def _spawn_board(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the board under ``prim_path``: a plain Xform holding the body and its pivot joint."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    spec: Board = _SPECS[cfg.maze_name]
    stage = get_current_stage()
    root = _root(stage, prim_path, translation, orientation)
    mats = _Materials(stage, prim_path)
    grip = mats.friction(spec.friction)
    slick = mats.friction(spec.lid_friction, combine_min=True)

    body = f"{prim_path}/body"
    _body(stage, body, spec.body_mass, com=(0.0, 0.0, maze.BOARD_COM_Z), angular_damping=spec.rocking_damping)

    # Base plate: its top face is the maze floor (z = 0).
    base = _disc(stage, f"{body}/base", maze.BOARD_R, -maze.BASE_T, 0.0)
    _collide(stage, base.GetPrim(), grip)
    _paint(stage, base.GetPrim(), mats.look(spec.floor_color))
    # The green centre cell: a paper-thin disc, looks only.
    goal = _disc(stage, f"{body}/goal_look", maze.WALLS[0] - maze.WALL_T / 2, 0.0, 0.0005)
    _paint(stage, goal.GetPrim(), mats.look(spec.goal_color))

    walls = mats.look(spec.wall_color)
    for i, b in enumerate(maze.wall_boxes()):
        cube = UsdGeom.Cube.Define(stage, f"{body}/wall_{i}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(b.x, b.y, maze.WALL_H / 2))
        cube.AddRotateZOp().Set(math.degrees(b.yaw))
        cube.AddScaleOp().Set(Gf.Vec3f(2 * b.hx, 2 * b.hy, maze.WALL_H))
        _collide(stage, cube.GetPrim(), grip)
        _paint(stage, cube.GetPrim(), walls)

    # The clear lid: it collides (nothing reaches the ball, and the ball cannot hop a wall) but is not drawn.
    lid = _disc(stage, f"{body}/lid", maze.BOARD_R, maze.WALL_H, maze.WALL_H + maze.LID_T)
    lid.CreatePurposeAttr("guide")
    _collide(stage, lid.GetPrim(), slick)

    # The gimbal block under the base and the ring weight hanging below it: looks only (no colliders; the body's mass
    # and centre of mass above already include them).
    block = _disc(stage, f"{body}/gimbal_look", 0.018, maze.PIVOT_Z - 0.010, -maze.BASE_T)
    _paint(stage, block.GetPrim(), mats.look(spec.wall_color))
    weight = _disc(stage, f"{body}/weight_look", 0.045, maze.BOARD_COM_Z - 0.006, maze.BOARD_COM_Z + 0.006)
    _paint(stage, weight.GetPrim(), mats.look(spec.rim_color))

    # The pivot: a joint between the world and the body at the gimbal's centre. Its x axis lies along
    # the board's normal, so "twist" (rotX) is turning about the normal, which is locked along with all sliding;
    # the two swings (rotY, rotZ) are the tilt.
    body_prim = stage.GetPrimAtPath(body)
    world = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(0)
    pivot_world = world.Transform(Gf.Vec3d(0.0, 0.0, maze.PIVOT_Z))
    align = Gf.Quatd(math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)  # turns x onto z
    rot_world = world.ExtractRotationQuat() * align
    joint = UsdPhysics.Joint.Define(stage, f"{prim_path}/pivot")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot_world))
    joint.CreateLocalRot0Attr(Gf.Quatf(float(rot_world.GetReal()), *(float(v) for v in rot_world.GetImaginary())))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, maze.PIVOT_Z))
    joint.CreateLocalRot1Attr(Gf.Quatf(float(align.GetReal()), *(float(v) for v in align.GetImaginary())))
    prim = joint.GetPrim()
    for dof in ("transX", "transY", "transZ", "rotX"):
        lim = UsdPhysics.LimitAPI.Apply(prim, dof)
        lim.CreateLowAttr(1.0)  # low above high locks the axis
        lim.CreateHighAttr(-1.0)
    stop = math.degrees(maze.TILT_STOP) + 1.5  # backstop only: the stand stops the board first
    for dof in ("rotY", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, dof)
        lim.CreateLowAttr(-stop)
        lim.CreateHighAttr(stop)
    return root.GetPrim()


def _spawn_ball(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdGeom

    spec: Ball = _SPECS[cfg.maze_name]
    stage = get_current_stage()
    root = _root(stage, prim_path, translation, orientation)
    mats = _Materials(stage, prim_path)
    body = f"{prim_path}/body"
    prim = _body(stage, body, spec.ball_mass, linear_damping=spec.linear_damping,
                 angular_damping=spec.angular_damping, iters=16)
    from pxr import PhysxSchema

    _try(lambda: PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr(True))  # small and quick
    sphere = UsdGeom.Sphere.Define(stage, f"{body}/geom")
    sphere.CreateRadiusAttr(spec.radius)
    _collide(stage, sphere.GetPrim(), mats.friction(spec.friction))
    _paint(stage, sphere.GetPrim(), mats.look(spec.color))
    return root.GetPrim()


def _arena_object(spec: Board | Ball):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    spawn = _spawn_board if isinstance(spec, Board) else _spawn_ball

    @configclass
    class MazeSpawnerCfg(SpawnerCfg):
        func: object = clone(spawn)
        maze_name: str = ""

    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID,
        spawner_cfg=MazeSpawnerCfg(maze_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                          rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))),
    )


# -- state ---------------------------------------------------------------------------------------------------------

def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(environment, spec: Board | Ball) -> BodyState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    pos = pose[:3] - origin
    reach = spec.radius if isinstance(spec, Ball) else maze.BOARD_R
    return BodyState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos - reach,
                     high=pos + reach)


# -- hooking yam_arena ---------------------------------------------------------------------------------------------

def _patch_scene(scene) -> None:
    if getattr(scene, "_mazekit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, Board | Ball):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._mazekit = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches ``yam_arena.scene`` as it is imported (it needs Isaac Sim, which starts after the task file loads)."""

    target = "yam_arena.scene"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch_scene(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now if its scene module is already imported, otherwise when it is. Idempotent."""
    if _Finder.target in sys.modules:
        _patch_scene(sys.modules[_Finder.target])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
