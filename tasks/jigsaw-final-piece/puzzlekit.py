"""puzzlekit: jigsaw pieces for a yam_arena task file.

:class:`Piece` is one flat piece of plywood cut to a jigsaw outline, with a picture printed on top: its collision
shape is a set of convex prisms (``parts``, polygons in the piece's frame), its picture a set of thin coloured
triangle meshes on its top face (``decals``, visual only), and optionally a square peg ``knob`` glued at its centre.
Loose with a ``mass``, fixed without. A loose piece's prisms can be bevelled underneath (``bottoms``: each part's
underside, or None for a part that starts ``chamfer`` above the floor). A fixed piece can instead be one exact
triangle-mesh collider (``outline`` and ``triangles``, its top face). Its state is a :class:`BodyState`: the pose
of the piece's frame (origin at the middle of its cell, on its underside).

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
from yam_arena.task_file import Box, ObjectState

__all__ = ["BodyState", "Piece", "install"]


@dataclass(frozen=True)
class Piece(Box):
    """``size`` is only nominal (the bounding box). ``color`` is the plywood's edge colour."""

    parts: tuple = ()          # convex polygons ((x, y), ...) in the piece frame, m
    thick: float = 0.006
    decals: tuple = ()         # (rgb, ((x, y), (x, y), (x, y)) triangles ...)
    bottoms: tuple = ()        # per part: its underside polygon, or None; empty: no bevel
    chamfer: float = 0.0025    # the bevel's height
    outline: tuple = ()        # with ``triangles``: a fixed piece is one exact triangle-mesh collider instead of
    triangles: tuple = ()      # convex parts (its outline, and its top face's triangles)
    knob: tuple | None = None  # (side, height), m
    knob_color: tuple = (0.80, 0.62, 0.40)
    friction: float = 0.6

    def _check(self) -> None:
        super()._check()
        if not self.parts and not (self.triangles and self.outline):
            raise ValueError(f"{self.name}: a Piece needs its parts, or its outline and triangles")
        if self.triangles and self.free:
            raise ValueError(f"{self.name}: only a fixed piece can be a triangle mesh")
        if self.bottoms and len(self.bottoms) != len(self.parts):
            raise ValueError(f"{self.name}: one bottom per part")


@dataclass(frozen=True)
class BodyState(ObjectState):
    """The piece frame's pose and velocities in the world (m, m/s, rad/s)."""

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def yaw(self) -> float:
        r = self.rotation()
        return math.atan2(r[1, 0], r[0, 0])

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.rotation()[2, 2])))))


_SPECS: dict[str, Piece] = {}


def _yaw_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def prism_mesh(part, z0: float, z1: float, bottom=None, chamfer: float = 0.0):
    """A convex part as a mesh (points, face sizes, face indices; faces counterclockwise seen from outside): the part
    from z0 + ``chamfer`` (z0 without a bevel) up to z1, and with a ``bottom``, that underside at z0 below it. The
    collision shape is the convex hull of all the points: with a bottom, a bevel from it up to the part."""
    n = len(part)
    pts = [(x, y, z0 + chamfer) for x, y in part] + [(x, y, z1) for x, y in part]
    counts = [n, n]
    idx = list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
    for k in range(n):
        j = (k + 1) % n
        counts.append(4)
        idx += [k, j, n + j, n + k]
    if bottom is not None:
        m = len(bottom)
        pts += [(x, y, z0) for x, y in bottom]
        counts.append(m)
        idx += list(range(2 * n + m - 1, 2 * n - 1, -1))
    return pts, counts, idx


def solid_mesh(outline, triangles, z0: float, z1: float):
    """A fixed piece as one closed mesh: its top's triangles at z1, the same turned over at z0, and its sides."""
    pts, counts, idx = [], [], []
    for t in triangles:
        base = len(pts)
        pts += [(x, y, z1) for x, y in t] + [(x, y, z0) for x, y in t]
        counts += [3, 3]
        idx += [base, base + 1, base + 2, base + 5, base + 4, base + 3]
    n = len(outline)
    base = len(pts)
    pts += [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    for k in range(n):
        j = (k + 1) % n
        counts.append(4)
        idx += [base + k, base + j, base + n + j, base + n + k]
    return pts, counts, idx


def _spawn_piece(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
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

    def look(name, rgb, roughness=0.6):
        path = f"{prim_path}/Looks/{name}"
        vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(tuple(rgb)), roughness=roughness)
        vis.func(path, vis)
        return path

    wood = look("wood", spec.color)

    body = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    if spec.free:
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateAngularDampingAttr(0.5)

    def collider(gprim):
        UsdPhysics.CollisionAPI.Apply(gprim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(gprim)
        cp.CreateContactOffsetAttr(0.0008)   # under the 1 mm clearance, so neighbours do not touch early
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(gprim.GetPath()), phys, stage=stage)

    if spec.triangles:  # fixed: one exact triangle mesh (PhysX allows these for static colliders)
        pts, counts, idx = solid_mesh(spec.outline, spec.triangles, 0.0, spec.thick)
        mesh = UsdGeom.Mesh.Define(stage, f"{body}/solid")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateSubdivisionSchemeAttr("none")
        geom = mesh.GetPrim()
        collider(geom)
        UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr("none")
        bind_visual_material(str(geom.GetPath()), wood, stage=stage)

    for i, part in enumerate(() if spec.triangles else spec.parts):
        bottom = spec.bottoms[i] if spec.bottoms else None
        pts, counts, idx = prism_mesh(part, 0.0, spec.thick, bottom, spec.chamfer if spec.bottoms else 0.0)
        mesh = UsdGeom.Mesh.Define(stage, f"{body}/part_{i}")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateSubdivisionSchemeAttr("none")
        geom = mesh.GetPrim()
        collider(geom)
        UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr("convexHull")
        try:  # WHY: PhysX thickens hulls thinner than 1 mm by default, which would eat the fit's clearance
            hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(geom)
            hull.CreateMinThicknessAttr(0.0001)
            hull.CreateHullVertexLimitAttr(128)
        except Exception:  # noqa: BLE001  (older schemas)
            pass
        bind_visual_material(str(geom.GetPath()), wood, stage=stage)

    for k, (rgb, tris) in enumerate(spec.decals):  # the printed picture: visual only, 0.2 mm above the top face
        mesh = UsdGeom.Mesh.Define(stage, f"{body}/print_{k}")
        z = spec.thick + 0.0002
        mesh.CreatePointsAttr([Gf.Vec3f(px, py, z) for t in tris for px, py in t])
        mesh.CreateFaceVertexCountsAttr([3] * len(tris))
        mesh.CreateFaceVertexIndicesAttr(list(range(3 * len(tris))))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        bind_visual_material(str(mesh.GetPath()), look(f"print_{k}", rgb, 0.7), stage=stage)

    if spec.knob:
        side, height = spec.knob
        cube = UsdGeom.Cube.Define(stage, f"{body}/knob")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spec.thick + height / 2))
        cube.AddScaleOp().Set(Gf.Vec3f(side, side, height))
        collider(cube.GetPrim())
        bind_visual_material(str(cube.GetPath()), look("knob", spec.knob_color), stage=stage)
    return root.GetPrim()


def _piece_object(spec: Piece):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class PieceSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_piece)
        part_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
                  spawner_cfg=PieceSpawnerCfg(part_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=_yaw_xyzw(spec.yaw)))


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(environment, spec: Piece) -> BodyState:
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
    if getattr(scene, "_puzzlekit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, Piece):
            _SPECS[spec.name] = spec
            return _piece_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._puzzlekit = True


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
