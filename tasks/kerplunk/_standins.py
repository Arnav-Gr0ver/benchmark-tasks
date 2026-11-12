"""Stand-ins for yam_arena's task-file modules, so check_physics.py can load task.py without Isaac Sim."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np


def install() -> None:
    try:
        import yam_arena.task_file  # noqa: F401
        return
    except ImportError:
        pass
    for name in ("yam_arena", "yam_arena.constants", "yam_arena.demo", "yam_arena.task_file"):
        sys.modules[name] = types.ModuleType(name)

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
        quat_xyzw: object = None
        lin_vel: object = None
        ang_vel: object = None
        low: object = None
        high: object = None

    tf = sys.modules["yam_arena.task_file"]
    tf.Box, tf.Cylinder, tf.ObjectState, tf.TABLE_TOP = Box, Cylinder, ObjectState, {"station": 0.7475}
    const = sys.modules["yam_arena.constants"]
    const.FINGER_OPEN, const.FINGER_CLOSED = -0.04695, 0.0
    demo = sys.modules["yam_arena.demo"]
    demo.Waypoint = lambda *a, **k: (a, k)
    demo.grip_rotation = lambda yaw=0.0: np.eye(3)
