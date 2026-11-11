"""shapekit: two object kinds yam_arena's task files lack, for a task file in this folder.

- :class:`Disc`: a cylinder lying along the world X or Y axis (a coin or game disc standing on its edge), loose (with a
  ``mass``) or fixed. With ``ridges`` its rim is that many flat ridges, as on a real game disc (a prism whose flats
  are ``radius`` from its axis, one flat facing each way across and down), so a gripper's pads press on flats rather
  than touching a round edge at one point; its state is then a :class:`BodyState` (the disc's centre). Without, it is
  a plain cylinder and its state is yam_arena's own.
- :class:`Compound`: one body made of several boxes (a jigsaw piece, a bracket), loose or fixed, turned ``yaw`` about the
  vertical. Its state is a :class:`BodyState`: the body origin's pose.

Importing this module hooks ``yam_arena.scene`` without changing yam_arena. Pure Python at import time; everything
that needs Isaac Sim is imported lazily, inside the functions.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, Cylinder, ObjectState

__all__ = ["BodyState", "Compound", "Disc", "install"]

# centre and size in the part frame (m), and optionally a tilt (degrees) about the part's x then y axes
BoxSpec = tuple


@dataclass(frozen=True)
class Disc(Cylinder):
    """A cylinder whose axis lies along world ``axis`` ("X" or "Y"); ``pos`` is its centre."""

    axis: str = "X"
    ridges: int = 0
    friction: tuple[float, float] = (1.2, 1.0)  # static, dynamic: yam_arena's rubber-like default

    def _check(self) -> None:
        super()._check()
        if self.axis not in ("X", "Y"):
            raise ValueError(f"{self.name}: axis must be 'X' or 'Y'")
        if self.ridges and (self.ridges < 8 or self.ridges % 4):
            raise ValueError(f"{self.name}: ridges must be a multiple of 4, at least 8")


@dataclass(frozen=True)
class Compound(Box):
    """One body of ``boxes`` (centre, size in the part frame, and optionally a tilt in degrees about the part's x then
    y axes); ``pos`` is the part frame's origin. ``size`` is only nominal (its bounding box). Loose with a ``mass``,
    fixed without."""

    boxes: tuple[BoxSpec, ...] = ()
    friction: float = 0.8

    def _check(self) -> None:
        super()._check()
        if not self.boxes:
            raise ValueError(f"{self.name}: a Compound needs at least one box")


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

    def yaw(self) -> float:
        r = self.rotation()
        return math.atan2(r[1, 0], r[0, 0])

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.rotation()[2, 2])))))


_SPECS: dict[str, Compound | Disc] = {}


def _yaw_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def _disc_object(scene, spec: Disc):
    """As yam_arena's own cylinder, lying along ``spec.axis``."""
    sim_utils = scene.sim_utils
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(spec.color), roughness=0.4)
    if spec.free:
        body = dict(rigid_props=scene.PhysxRigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass), collision_props=scene._contact_props())
    else:
        body = dict(collision_props=scene.PhysxCollisionPropertiesCfg())
    body.update(physics_material=scene._grippy(), visual_material=look)
    cfg = sim_utils.CylinderCfg(radius=spec.radius, height=spec.height, axis=spec.axis, **body)
    return scene.Object(name=spec.name, object_type=scene.ObjectType.RIGID if spec.free else scene.ObjectType.BASE,
                        spawner_cfg=cfg, initial_pose=scene.Pose(position_xyz=tuple(spec.pos),
                                                                 rotation_xyzw=_yaw_xyzw(spec.yaw)))


def _spawn_compound(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)
    look = f"{prim_path}/Looks/colour"
    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.5)
    vis.func(look, vis)

    body = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    if spec.free:
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
    for i, box in enumerate(spec.boxes):
        centre, size = box[0], box[1]
        tilt = box[2] if len(box) > 2 else (0.0, 0.0)
        cube = UsdGeom.Cube.Define(stage, f"{body}/box_{i}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        if any(tilt):
            cube.AddRotateXYZOp().Set(Gf.Vec3f(float(tilt[0]), float(tilt[1]), 0.0))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cp = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        cp.CreateContactOffsetAttr(0.001)  # under the fits' clearances, so neighbouring parts don't touch early
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(cube.GetPath()), phys, stage=stage)
        bind_visual_material(str(cube.GetPath()), look, stage=stage)
    return root.GetPrim()


def ridged_points(spec: Disc) -> list[tuple[float, float, float]]:
    """The prism's corners in the disc's own frame: two rings of ``ridges`` points, flats ``radius`` from the axis,
    a flat's middle (not a corner) facing each way across the axis and straight down."""
    n = spec.ridges
    corner = spec.radius / math.cos(math.pi / n)
    pts = []
    for side in (-spec.height / 2, spec.height / 2):
        for k in range(n):
            a = (k + 0.5) * 2 * math.pi / n
            u, v = corner * math.cos(a), corner * math.sin(a)
            pts.append((side, u, v) if spec.axis == "X" else (u, side, v))
    return pts


def _spawn_ridged(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    pm.func(phys, pm)
    look = f"{prim_path}/Looks/colour"
    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.4)
    vis.func(look, vis)

    body = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    if spec.free:
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)

    n = spec.ridges
    mesh = UsdGeom.Mesh.Define(stage, f"{body}/geom")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in ridged_points(spec)])
    counts = [n, n] + [4] * n
    idx = list(range(n - 1, -1, -1)) + list(range(n, 2 * n))  # the two flat faces
    for k in range(n):
        j = (k + 1) % n
        idx += [k, j, n + j, n + k]
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    geom = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(geom)
    UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr("convexHull")
    cp = PhysxSchema.PhysxCollisionAPI.Apply(geom)
    cp.CreateContactOffsetAttr(0.003)
    cp.CreateRestOffsetAttr(0.0)
    bind_physics_material(str(geom.GetPath()), phys, stage=stage)
    bind_visual_material(str(geom.GetPath()), look, stage=stage)
    return root.GetPrim()


def _ridged_object(spec: Disc):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class RidgedSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_ridged)
        part_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
                  spawner_cfg=RidgedSpawnerCfg(part_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=_yaw_xyzw(spec.yaw)))


def _compound_object(spec: Compound):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class CompoundSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_compound)
        part_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
                  spawner_cfg=CompoundSpawnerCfg(part_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=_yaw_xyzw(spec.yaw)))


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _compound_state(environment, spec: Compound | Disc) -> BodyState:
    if isinstance(spec, Disc):
        half = np.full(3, spec.radius)
        half[0 if spec.axis == "X" else 1] = spec.height / 2
    else:
        half = np.asarray(spec.size, float) / 2
    if not spec.free:
        pos = np.asarray(spec.pos, float)
        return BodyState(pos=pos, quat_xyzw=np.asarray(_yaw_xyzw(spec.yaw), float), lin_vel=np.zeros(3),
                         ang_vel=np.zeros(3), low=pos - half, high=pos + half)
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    pos = pose[:3] - origin
    reach = float(np.linalg.norm(half))
    return BodyState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos - reach,
                     high=pos + reach)


def _patch_scene(scene) -> None:
    if getattr(scene, "_shapekit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, Disc) and spec.ridges:
            _SPECS[spec.name] = spec
            return _ridged_object(spec)
        if isinstance(spec, Disc):
            return _disc_object(scene, spec)
        if isinstance(spec, Compound):
            _SPECS[spec.name] = spec
            return _compound_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _compound_state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._shapekit = True


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
