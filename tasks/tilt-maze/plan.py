"""A 2D model of the ball in the maze, and a planner of tilts over it. Pure Python (math only).

The planner works in "holds": tilt the board a fixed angle toward an azimuth until the ball comes to rest, then
let the board level. Holds are quasi-static, so the plan barely depends on how fast the ball rolls: it only needs
where the ball comes to rest. The demo re-plans after every hold from where the simulator says the ball is, and
the tests use the same model to check that random tilting almost never gets out of the outer ring.

Board frame and azimuths as in ``maze.py``. Tilting toward azimuth phi lowers the rim at phi.
"""

from __future__ import annotations

import math
from collections import deque

import maze

G = 9.81
_CELL = 0.02


class BallModel:
    """A ball rolling on the maze floor: gravity along the tilt (5/7 g sin for a rolling solid sphere), rolling
    resistance (viscous plus a constant term), and walls as boxes with a little restitution and wall friction."""

    def __init__(self, dt: float = 0.002, restitution: float = 0.2, viscous: float = 1.0, coulomb: float = 0.03,
                 wall_mu: float = 0.2):
        self.dt, self.e, self.visc, self.coul, self.wall_mu = dt, restitution, viscous, coulomb, wall_mu
        self.boxes = maze.wall_boxes()
        self.trig = [(math.cos(b.yaw), math.sin(b.yaw)) for b in self.boxes]
        self.grid: dict[tuple[int, int], list[int]] = {}
        for i, b in enumerate(self.boxes):
            reach = math.hypot(b.hx, b.hy) + maze.BALL_R + 0.002
            for gx in range(int((b.x - reach) // _CELL), int((b.x + reach) // _CELL) + 1):
                for gy in range(int((b.y - reach) // _CELL), int((b.y + reach) // _CELL) + 1):
                    self.grid.setdefault((gx, gy), []).append(i)
        self.reset()

    def reset(self, xy=None) -> None:
        self.x, self.y = maze.START_XY if xy is None else (float(xy[0]), float(xy[1]))
        self.vx = self.vy = 0.0
        self.t = 0.0

    def _collide(self) -> None:
        r = maze.BALL_R
        for i in self.grid.get((int(self.x // _CELL), int(self.y // _CELL)), ()):
            b = self.boxes[i]
            c, s = self.trig[i]
            dx, dy = self.x - b.x, self.y - b.y
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            ex, ey = lx - min(max(lx, -b.hx), b.hx), ly - min(max(ly, -b.hy), b.hy)
            d = math.hypot(ex, ey)
            if d >= r:
                continue
            if d < 1e-9:  # centre inside the box: out along the shallower side
                if b.hx - abs(lx) < b.hy - abs(ly):
                    ex, ey, pen = math.copysign(1.0, lx), 0.0, r + b.hx - abs(lx)
                else:
                    ex, ey, pen = 0.0, math.copysign(1.0, ly), r + b.hy - abs(ly)
                d = 1.0
            else:
                pen = r - d
            nx, ny = ex / d, ey / d
            wx, wy = c * nx - s * ny, s * nx + c * ny
            self.x += wx * pen
            self.y += wy * pen
            vn = self.vx * wx + self.vy * wy
            if vn < 0:
                self.vx -= (1 + self.e) * vn * wx
                self.vy -= (1 + self.e) * vn * wy
                tx, ty = -wy, wx
                vt = self.vx * tx + self.vy * ty
                drop = min(abs(vt), self.wall_mu * (1 + self.e) * -vn)
                self.vx -= math.copysign(drop, vt) * tx
                self.vy -= math.copysign(drop, vt) * ty

    def step(self, tilt_x: float, tilt_y: float) -> None:
        """Advance one time step with the floor tilted down toward +x by ``tilt_x`` and toward +y by ``tilt_y``."""
        dt = self.dt
        self.vx += 5 / 7 * G * math.sin(tilt_x) * dt
        self.vy += 5 / 7 * G * math.sin(tilt_y) * dt
        speed = math.hypot(self.vx, self.vy)
        if speed > 0:
            k = max(0.0, 1 - (self.visc * speed + self.coul) * dt / speed)
            self.vx *= k
            self.vy *= k
        self.x += self.vx * dt
        self.y += self.vy * dt
        self._collide()
        self._collide()
        self.t += dt

    def hold(self, azimuth_deg: float, tilt_deg: float, max_s: float = 6.0) -> float:
        """Tilt toward ``azimuth_deg`` until the ball has rested for 0.25 s (or ``max_s``); returns the time."""
        a, p = math.radians(tilt_deg), math.radians(azimuth_deg)
        tx, ty = a * math.cos(p), a * math.sin(p)
        still, t0 = 0.0, self.t
        while self.t - t0 < max_s:
            self.step(tx, ty)
            still = still + self.dt if math.hypot(self.vx, self.vy) < 0.003 else 0.0
            if still > 0.25:
                break
        return self.t - t0


def _key(x: float, y: float) -> tuple[int, int]:
    """Rest positions closer than about 2 mm and 3 degrees count as the same."""
    return round(math.hypot(x, y) * 500), round((math.degrees(math.atan2(y, x)) % 360) / 3) % 120


def plan(xy=None, tilt_deg: float = 6.0, azimuths=tuple(range(0, 360, 30)), max_holds: int = 6000):
    """The fewest holds (azimuths, deg) that bring the ball from ``xy`` (default: the start mark) to rest in the
    centre cell, by breadth-first search over rest positions; None if none is found within ``max_holds`` holds."""
    model = BallModel()
    start = maze.START_XY if xy is None else (float(xy[0]), float(xy[1]))
    if maze.ring_at(*start) == 0:
        return []
    seen = {_key(*start)}
    queue = deque([(start, [])])
    tried = 0
    while queue and tried < max_holds:
        here, holds = queue.popleft()
        for az in azimuths:
            model.reset(here)
            model.hold(az, tilt_deg)
            tried += 1
            if maze.ring_at(model.x, model.y) == 0:
                return holds + [az]
            k = _key(model.x, model.y)
            if k not in seen:
                seen.add(k)
                queue.append(((model.x, model.y), holds + [az]))
    return None


def random_tilting(trials: int = 200, holds: int = 30, tilt_deg: float = 6.0, seed: int = 0) -> list[int]:
    """Holds toward random azimuths: how many trials end with the ball in each ring (index 0: the centre cell)."""
    import random

    rng = random.Random(seed)
    model = BallModel()
    counts = [0] * (maze.RINGS + 1)
    for _ in range(trials):
        model.reset()
        for _ in range(holds):
            model.hold(rng.uniform(0.0, 360.0), tilt_deg, max_s=3.0)
            if maze.ring_at(model.x, model.y) == 0:
                break
        counts[maze.ring_at(model.x, model.y)] += 1
    return counts
