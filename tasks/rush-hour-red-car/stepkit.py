"""stepkit: run task-file callbacks after every control step and after every reset.

Callbacks registered with :func:`on_step` run after every control step, before the task file is scored; ones
registered with :func:`on_reset` run after every reset. Both get the Isaac Lab environment. Importing this module
hooks ``yam_arena.inspect_embodiment`` without changing yam_arena (the same hook as scenekit's in
tasks/lock-before-loading). Pure Python at import time.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys

__all__ = ["on_reset", "on_step"]

_STEP: list = []
_RESET: list = []


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset."""
    _RESET.append(fn)
    return fn


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_stepkit", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def task_step_hook(self, *a, **k):
        for fn in _STEP:
            fn(self._env)
        return task_step(self, *a, **k)

    def reset_hook(self, *a, **k):
        obs = reset(self, *a, **k)
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = task_step_hook, reset_hook
    cls._stepkit = True


class _Finder(importlib.abc.MetaPathFinder):
    target = "yam_arena.inspect_embodiment"

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
            _patch_embodiment(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    if _Finder.target in sys.modules:
        _patch_embodiment(sys.modules[_Finder.target])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
