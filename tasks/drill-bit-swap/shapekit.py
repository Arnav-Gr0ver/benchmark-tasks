"""shapekit: two object kinds yam_arena's task files lack, for a task file in this folder.

- :class:`Disc`: a cylinder lying along the world X or Y axis (a coin or game disc standing on its edge), loose (with a
  ``mass``) or fixed. With ``ridges`` its rim is that many flat ridges, as on a real game disc (a prism whose flats
  are ``radius`` from its axis, one flat facing each way across and down), so a gripper's pads press on flats rather
  than touching a round edge at one point; its state is then a :class:`BodyState` (the disc's centre). Without, it is
  a plain cylinder and its state is yam_arena's own.
- :class:`Compound`: one body made of several boxes and cylinders (here the tube with its wall and holes, and each rod
  with its cap), loose or fixed, turned ``yaw`` about the vertical; a box can be turned about x, y and z; a part given
  as (centre, ("cyl", radius, length, axis)) is a cylinder; the body can be see-through (``opacity``) and its parts
  coloured one by one (``part_colors``). Its state is a :class:`BodyState`: the body origin's pose.
- :class:`Marble`: a ball with its own friction. Its state is a :class:`BodyState`.

A plain Disc (no ridges) is a cylinder along x or y with its own friction.

Importing this module hooks ``yam_arena.scene`` without changing yam_arena. Pure Python at import time; everything
that needs Isaac Sim is imported lazily, inside the functions.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, Cylinder, ObjectState

__all__ = ["BodyState", "Compound", "Disc", "Link", "Marble", "MechState", "Mechanism", "clear_wrench", "install",
           "robot_fingers", "set_joint_targets", "set_wrench"]

# centre and size in the part frame (m), and optionally a tilt (degrees) about the part's x then y axes
BoxSpec = tuple


@dataclass(frozen=True)
class Disc(Cylinder):
    """A cylinder whose axis lies along world ``axis`` ("X" or "Y"); ``pos`` is its centre."""

    axis: str = "X"
    ridges: int = 0
    friction: tuple[float, float] = (1.2, 1.0)  # static, dynamic: yam_arena's rubber-like default
    # m: lets a pinch on the rim resist the disc swivelling about the line between the pads, as rubber pads do;
    # without it a contact resists sliding but not twisting
    torsional_patch: float = 0.004
    angular_damping: float = 1.0

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
    opacity: float = 1.0
    part_colors: tuple = ()     # optional: one colour per part (else ``color`` for all)
    com: tuple | None = None    # optional: the centre of mass in the body's frame (else from its parts, as if solid)
    quat_xyzw: tuple | None = None   # optional: the starting orientation in full (else ``yaw`` about the vertical)

    def _check(self) -> None:
        super()._check()
        if not self.boxes:
            raise ValueError(f"{self.name}: a Compound needs at least one box")


@dataclass(frozen=True)
class Marble(Box):
    """A ball of ``radius``; ``size`` is only nominal. Loose with a ``mass``."""

    radius: float = 0.008
    friction: float = 0.3


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


@dataclass(frozen=True)
class Link:
    """A part of a :class:`Mechanism` on a joint to its root body. ``kind`` "turn": a hinge about the root's x axis
    through ``origin``; "slide": a slide along the root's y axis turned ``about_x_deg`` about x, with ``limits``
    (lower, upper; metres from where it is built). Its ``parts`` are given in the link's own frame (the root's axes,
    at ``origin``). ``stiffness`` and ``damping`` are its joint drive's (N/m and N s/m, or N m/rad and N m s/rad);
    the drive's target is set while the episode runs with :func:`set_joint_targets`. ``friction_min``: this link's
    friction rules its contacts (PhysX "min"), for hard smooth steel against anything."""

    name: str
    kind: str
    origin: tuple
    parts: tuple
    mass: float
    about_x_deg: float = 0.0
    limits: tuple | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    friction: float = 0.8
    friction_min: bool = False
    color: tuple = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class Mechanism(Box):
    """A loose body with parts on joints: one PhysX articulation whose root is the body (its ``parts``, as a
    Compound's) and whose ``links`` hang off it. ``slick`` lists root parts (by index) that get ``slick_friction``
    with "min". ``part_colors``: one colour per root part. Its state is a :class:`MechState`."""

    parts: tuple = ()
    links: tuple = ()
    com: tuple | None = None
    friction: float = 0.8
    slick: tuple = ()
    slick_friction: float = 0.3
    part_colors: tuple = ()

    def _check(self) -> None:
        super()._check()
        if not self.free or not self.parts or not self.links:
            raise ValueError(f"{self.name}: a Mechanism is loose (give it a mass) and has parts and links")


@dataclass(frozen=True)
class MechState(BodyState):
    """The root body's pose and velocities, and each link's joint position (``joints``: metres or radians)."""

    joints: dict = field(default_factory=dict)


_SPECS: dict = {}


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
    body.update(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0],
                                                                dynamic_friction=spec.friction[1]),
                visual_material=look)
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
    # A slick part (friction under 0.3) rules its contacts: "min" takes the lower of the two surfaces' frictions.
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction,
                                        friction_combine_mode="min" if spec.friction < 0.3 else "average")
    pm.func(phys, pm)
    look = f"{prim_path}/Looks/colour"
    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.5,
                                      opacity=float(getattr(spec, "opacity", 1.0)))
    vis.func(look, vis)
    looks = []
    for k, rgb in enumerate(getattr(spec, "part_colors", ()) or ()):
        path = f"{prim_path}/Looks/part_{k}"
        cfg_k = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(tuple(rgb)), roughness=0.5)
        cfg_k.func(path, cfg_k)
        looks.append(path)

    body = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    if spec.free:
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(spec.mass)
        if getattr(spec, "com", None) is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in spec.com)))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
    for i, box in enumerate(spec.boxes):
        centre, size = box[0], box[1]
        if isinstance(size, tuple) and size and size[0] == "cyl":   # ("cyl", radius, length, axis "X" / "Y" / "Z")
            _, radius, length, axis = size
            cyl = UsdGeom.Cylinder.Define(stage, f"{body}/cyl_{i}")
            cyl.CreateRadiusAttr(float(radius))
            cyl.CreateHeightAttr(float(length))
            cyl.CreateAxisAttr(axis)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*centre))
            prim = cyl.GetPrim()
        else:
            tilt = tuple(box[2]) + (0.0,) * (3 - len(box[2])) if len(box) > 2 else (0.0, 0.0, 0.0)
            cube = UsdGeom.Cube.Define(stage, f"{body}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            if any(tilt):
                cube.AddRotateXYZOp().Set(Gf.Vec3f(float(tilt[0]), float(tilt[1]), float(tilt[2])))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(0.001)  # under the fits' clearances, so neighbouring parts don't touch early
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(prim.GetPath()), looks[i] if looks else look, stage=stage)
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
        rb.CreateAngularDampingAttr(spec.angular_damping)

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
    if spec.free and spec.torsional_patch:
        cp.CreateTorsionalPatchRadiusAttr(spec.torsional_patch)
        cp.CreateMinTorsionalPatchRadiusAttr(0.8 * spec.torsional_patch)
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
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                    rotation_xyzw=tuple(spec.quat_xyzw) if spec.quat_xyzw else _yaw_xyzw(spec.yaw)))


def _marble_object(scene, spec: Marble):
    sim_utils = scene.sim_utils
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(spec.color), roughness=0.15, metallic=0.1)
    body = dict(rigid_props=scene.PhysxRigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
                collision_props=scene._contact_props(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction,
                                                                 dynamic_friction=spec.friction),
                visual_material=look)
    cfg = sim_utils.SphereCfg(radius=spec.radius, **body)
    return scene.Object(name=spec.name, object_type=scene.ObjectType.RIGID, spawner_cfg=cfg,
                        initial_pose=scene.Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _spawn_mechanism(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the mechanism under ``prim_path``: a plain Xform holding the root body and its links as siblings, the
    joints, and the articulation root on the body (the layout merged tasks' partkit uses)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Mechanism = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    made: dict = {}

    def material(friction: float, use_min: bool) -> str:
        key = (round(friction, 4), use_min)
        if key not in made:
            path = f"{prim_path}/Looks/physics_{len(made)}"
            pm = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction,
                                                friction_combine_mode="min" if use_min else "average")
            pm.func(path, pm)
            made[key] = path
        return made[key]

    looks: dict = {}

    def look(color) -> str:
        color = tuple(color)
        if color not in looks:
            path = f"{prim_path}/Looks/colour_{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, pos, mass: float, com=None):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(float(v) for v in pos)))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))
        if com is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in com)))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        return prim

    def shapes(path: str, parts, phys_of, colour_of):
        for i, part in enumerate(parts):
            centre, size = part[0], part[1]
            if isinstance(size, tuple) and size and size[0] == "cyl":   # ("cyl", radius, length, axis)
                _, radius, length, axis = size
                geom = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
                geom.CreateRadiusAttr(float(radius))
                geom.CreateHeightAttr(float(length))
                geom.CreateAxisAttr(axis)
                geom.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in centre)))
            else:
                tilt = tuple(part[2]) + (0.0,) * (3 - len(part[2])) if len(part) > 2 else (0.0, 0.0, 0.0)
                geom = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
                geom.CreateSizeAttr(1.0)
                geom.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in centre)))
                if any(tilt):
                    geom.AddRotateXYZOp().Set(Gf.Vec3f(*(float(v) for v in tilt)))
                geom.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in size)))
            prim = geom.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(0.0005)
            cp.CreateRestOffsetAttr(0.0)
            bind_physics_material(str(prim.GetPath()), phys_of(i), stage=stage)
            bind_visual_material(str(prim.GetPath()), look(colour_of(i)), stage=stage)

    main = f"{prim_path}/body"
    body(main, (0.0, 0.0, 0.0), spec.mass, spec.com)
    shapes(main, spec.parts,
           lambda i: material(spec.slick_friction, True) if i in spec.slick else material(spec.friction, False),
           lambda i: spec.part_colors[i] if i < len(spec.part_colors) else spec.color)
    for link in spec.links:
        path = f"{prim_path}/{link.name}"
        body(path, link.origin, link.mass)
        shapes(path, link.parts, lambda i, k=link: material(k.friction, k.friction_min), lambda i, k=link: k.color)
        a = math.radians(link.about_x_deg) / 2
        frame = Gf.Quatf(math.cos(a), math.sin(a), 0.0, 0.0)
        if link.kind == "turn":
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/{link.name}_joint")
            joint.CreateAxisAttr("X")
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")  # USD angular drives take degrees
            drive.CreateStiffnessAttr(link.stiffness * math.pi / 180.0)
            drive.CreateDampingAttr(link.damping * math.pi / 180.0)
        else:
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{link.name}_joint")
            joint.CreateAxisAttr("Y")
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateStiffnessAttr(link.stiffness)
            drive.CreateDampingAttr(link.damping)
        if link.limits is not None:
            joint.CreateLowerLimitAttr(float(link.limits[0]))
            joint.CreateUpperLimitAttr(float(link.limits[1]))
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(main)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in link.origin)))
        joint.CreateLocalRot0Attr(frame)
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(frame)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(main))
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(main))
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _mechanism_object(spec: Mechanism):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class MechanismSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_mechanism)
        part_name: str = ""

    class MechanismObject(Object):
        def _generate_articulation_cfg(self):
            actuators = {k.name: ImplicitActuatorCfg(joint_names_expr=[f"{k.name}_joint"], stiffness=k.stiffness,
                                                     damping=k.damping) for k in spec.links}
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    return MechanismObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                           spawner_cfg=MechanismSpawnerCfg(part_name=spec.name),
                           initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                             rotation_xyzw=_yaw_xyzw(spec.yaw)))


def _t(x):
    return getattr(x, "torch", x)


def _mechanism_state(environment, spec: Mechanism) -> MechState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    pos = pose[:3] - origin
    reach = float(np.linalg.norm(np.asarray(spec.size, float) / 2))
    names = list(asset.joint_names)
    q = _np(asset.data.joint_pos)[0]
    joints = {k.name: float(q[names.index(f"{k.name}_joint")]) for k in spec.links}
    return MechState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos - reach,
                     high=pos + reach, joints=joints)


def set_joint_targets(environment, name: str, targets: dict) -> None:
    """Set the position targets of a mechanism's joint drives ({link name: metres or radians}); the other joints'
    targets are set to where they are."""
    asset = environment.unwrapped.scene[name]
    names = list(asset.joint_names)
    q = _t(asset.data.joint_pos).clone()
    for link, value in targets.items():
        q[:, names.index(f"{link}_joint")] = float(value)
    command = getattr(getattr(asset, "actuators", None), "target_command", None)
    if command is not None:  # Isaac Lab 3.0, as yam_arena.actions uses it
        command.set_position_index(value=q)
    else:
        asset.set_joint_position_target(q)


def set_wrench(environment, name: str, force, body: str | None = None) -> None:
    """Hold a constant force (world frame, N) on a body until :func:`clear_wrench`: on a rigid object, or on the
    named link of a mechanism. Replaces any force set before."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    ids = torch.tensor([0], device=base.device)
    index = 0 if body is None else list(asset.body_names).index(body)
    composer = asset.permanent_wrench_composer
    composer.reset(env_ids=ids)
    f = torch.tensor([[[float(v) for v in force]]], device=base.device)
    composer.add_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), body_ids=[index], env_ids=ids,
                                          is_global=True)


def clear_wrench(environment, name: str) -> None:
    import torch

    base = environment.unwrapped
    base.scene[name].permanent_wrench_composer.reset(env_ids=torch.tensor([0], device=base.device))


def robot_fingers(environment) -> dict[str, np.ndarray]:
    """Each arm's finger envelope: ten points that bound where its two finger pads can be at any opening (station
    frame). From link positions only: the approach axis runs from the gripper link's origin through the midpoint of
    the two finger links' origins (56.7 mm along it); the pads' tips are 147 mm along it; the fingers open along
    the line between the finger links, up to 48 mm either side."""
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        names = list(robot.body_names)
        pos = _np(robot.data.body_pos_w)[0] - origin
        out[side] = finger_envelope(pos[names.index("gripper")], pos[names.index("tip_left")],
                                    pos[names.index("tip_right")])
    return out


def finger_envelope(gripper, tip_left, tip_right) -> np.ndarray:
    gripper, tip_left, tip_right = (np.asarray(v, float) for v in (gripper, tip_left, tip_right))
    approach = (tip_left + tip_right) / 2 - gripper
    approach /= np.linalg.norm(approach)
    across = tip_left - tip_right
    across -= approach * (across @ approach)
    across /= np.linalg.norm(across)
    return np.array([gripper + approach * h + across * s for h in (0.110, 0.147)
                     for s in (-0.048, -0.024, 0.0, 0.024, 0.048)])


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
        quat = getattr(spec, "quat_xyzw", None) or _yaw_xyzw(spec.yaw)
        return BodyState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.zeros(3),
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
            _SPECS[spec.name] = spec
            return _disc_object(scene, spec)
        if isinstance(spec, Marble):
            _SPECS[spec.name] = spec
            return _marble_object(scene, spec)
        if isinstance(spec, Mechanism):
            _SPECS[spec.name] = spec
            return _mechanism_object(spec)
        if isinstance(spec, Compound):
            _SPECS[spec.name] = spec
            return _compound_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                spec = _SPECS[n]
                out[n] = (_mechanism_state if isinstance(spec, Mechanism) else _compound_state)(environment, spec)
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
