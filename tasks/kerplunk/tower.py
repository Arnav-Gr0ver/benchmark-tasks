"""The KerPlunk tower's geometry, shared by the task, its scorer and its checks. Pure numpy.

Frame: the tube's own, in metres: origin on its axis at the base board's top, x away from the robot, y to its left,
z up. task.py places it on the table.

Four rods cross the tube through holes in its wall. Three are ledges: each runs 30 mm from the axis, and marbles sit
in the crescent between it and the wall (a marble's centre lies outside the rod, toward the wall, so the rod and the
wall hold it; without the rod nothing is under it). The fourth rod, the guard, runs just inside the ledge on the far
side, level with its marbles' middles, touching them: it looks like it holds them in, but gravity presses them
outward onto the ledge and the wall, not onto it.
"""

from __future__ import annotations

import math

import numpy as np

TUBE_R_IN, TUBE_WALL, TUBE_H = 0.047, 0.002, 0.160      # 10 cm across outside, 16 cm tall
TUBE_R_MID = TUBE_R_IN + TUBE_WALL / 2
ROD_R, ROD_LEN = 0.003, 0.160                           # 6 mm rods, 16 cm long
MARBLE_R = 0.008                                        # 16 mm marbles
LEDGE = 0.030                                           # ledge rods run this far from the axis
Z0 = 0.100                                              # the far ledge's height above the base board
LOW = Z0 - 0.0065                                       # the left and right ledges pass 6.5 mm below it
PAIR = 0.008                                            # a pair of marbles sits 8 mm either side of a ledge's middle

_ON_WALL = TUBE_R_IN - MARBLE_R                         # a marble touching the wall: its centre this far out


def _rest_height(offset: float) -> float:
    """How far above a ledge's axis a marble touching it sits, its centre ``offset`` outside the rod (horizontally)."""
    return math.sqrt((ROD_R + MARBLE_R) ** 2 - offset ** 2)


# The pair on the far ledge: at y = +-PAIR, touching the wall.
_FAR_X = math.sqrt(_ON_WALL ** 2 - PAIR ** 2)
_FAR_Z = Z0 + _rest_height(_FAR_X - LEDGE)
GUARD_X = _FAR_X - (ROD_R + MARBLE_R)                   # the guard touches their inner sides, level with their middles

# name: (axis "x" or "y" (its direction), its offset across the tube, its height, what it is)
RODS = {
    "far": ("y", LEDGE, Z0, "ledge under the far pair"),
    "guard": ("y", GUARD_X, _FAR_Z, "guard against the far pair's inner sides"),
    "left": ("x", LEDGE, LOW, "ledge under the left pair"),
    "right": ("x", -LEDGE, LOW, "ledge under the single marble on the right"),
}
# How far each rod's middle sits along its own length from the tube's axis: the far and guard rods run side by side,
# 2.8 mm apart, so they stick out unevenly (the guard 2 cm further on the robot's right, the far rod on its left),
# leaving each one's long end clear to hold.
SHIFT = {"far": 0.020, "guard": -0.020, "left": 0.0, "right": 0.0}
SAFE = "guard"

_SIDE_X = math.sqrt(_ON_WALL ** 2 - PAIR ** 2)
_LOW_Z = LOW + _rest_height(_SIDE_X - LEDGE)
_ONE_Z = LOW + _rest_height(_ON_WALL - LEDGE)
MARBLES = {
    "marble_far_l": (_FAR_X, PAIR, _FAR_Z),
    "marble_far_r": (_FAR_X, -PAIR, _FAR_Z),
    "marble_left_n": (-PAIR, _SIDE_X, _LOW_Z),
    "marble_left_f": (PAIR, _SIDE_X, _LOW_Z),
    "marble_right": (0.0, -_ON_WALL, _ONE_Z),
}
SUPPORT_LAYER = LOW                                     # the lowest rods
FALLEN_Z = SUPPORT_LAYER - 0.020                        # a marble whose centre drops below this has fallen


def rod_dir(name: str) -> np.ndarray:
    return np.array([1.0, 0.0, 0.0]) if RODS[name][0] == "x" else np.array([0.0, 1.0, 0.0])


def rod_centre(name: str) -> np.ndarray:
    axis, off, z, _ = RODS[name]
    return np.array([SHIFT[name], off, z]) if axis == "x" else np.array([off, SHIFT[name], z])


R_OUT = TUBE_R_IN + TUBE_WALL


def extracted(name: str, centre) -> float:
    """How far out of the tube a rod is, 0 to 1: the share of the slide (either way along it) that takes its last
    end out past the tube's outside, from where it started. 1 or more: out."""
    axis, off, _, _ = RODS[name]
    a = math.sqrt(R_OUT ** 2 - off ** 2) + 0.0025          # the chord inside the tube, and its sleeve
    k = 0 if axis == "x" else 1
    s0, s = SHIFT[name], float(np.asarray(centre)[k])
    if s < s0:
        return (s0 - s) / (s0 + ROD_LEN / 2 + a)
    return (s - s0) / (a - (s0 - ROD_LEN / 2))


def wall_crossings(name: str) -> list[tuple[float, float]]:
    """Where a rod passes through the wall's mid-surface: (angle about the axis, angle between rod and wall normal)."""
    axis, off, _, _ = RODS[name]
    along = math.sqrt(TUBE_R_MID ** 2 - off ** 2)
    out = []
    for s in (-1, 1):
        x, y = (s * along, off) if axis == "x" else (off, s * along)
        phi = math.atan2(y, x)
        d = rod_dir(name)[:2]
        n = np.array([math.cos(phi), math.sin(phi)])
        out.append((phi, math.acos(min(1.0, abs(float(d @ n))))))
    return out


N_WALL = 48
BORE_CLEAR = 0.001                                      # a drilled hole: 1 mm clear of the rod at either side
BORE_T, BORE_LEN = 0.001, 0.003                         # modelled as a 1 mm sill and two 1 mm cheeks, 3 mm long
WINDOW = 0.003                                          # the wall's panels are cut back this much round each hole


def crossing_points(name: str) -> list[np.ndarray]:
    axis, off, z, _ = RODS[name]
    along = math.sqrt(TUBE_R_MID ** 2 - off ** 2)
    return [np.array([s * along, off, z]) if axis == "x" else np.array([off, s * along, z]) for s in (-1, 1)]


def bores() -> list[tuple[tuple, tuple]]:
    return [(c, size) for _, c, size in bores_by_rod()]


def bores_by_rod() -> list[tuple[str, tuple, tuple]]:
    """Each rod's two holes through the wall: a sill the rod rests on (its top exactly at the rod's underside) and a
    cheek either side, 1 mm clear of it; open on top, as a loose hole is to a rod lying in it. (centre, size) in the
    tube's frame, axis-aligned. Kept thin so no part of one rod's hole reaches another rod (the far and guard rods run
    2.8 mm apart): see test_kerplunk.py."""
    out = []
    side = ROD_R + BORE_CLEAR
    for name in RODS:
        along_x = RODS[name][0] == "x"
        for c in crossing_points(name):
            span = 2 * (side + BORE_T)
            sill = (BORE_LEN, span, BORE_T) if along_x else (span, BORE_LEN, BORE_T)
            out.append((name, tuple(c + np.array([0.0, 0.0, -ROD_R - BORE_T / 2])), sill))
            height = 2 * ROD_R + BORE_T
            for ds in (-1, 1):
                shift = np.array([0.0, ds * (side + BORE_T / 2), 0.0]) if along_x else \
                    np.array([ds * (side + BORE_T / 2), 0.0, 0.0])
                size = (BORE_LEN, BORE_T, height) if along_x else (BORE_T, BORE_LEN, height)
                out.append((name, tuple(c + shift + np.array([0.0, 0.0, -BORE_T / 2])), size))
    return out


def wall_boxes() -> list[tuple[tuple, tuple, tuple]]:
    """The tube's wall as N_WALL flat panels round the axis, each cut away round where a rod passes through it (the
    rod's sleeve fills the window). (centre, size, (0, 0, turn about z in degrees)) in the tube's frame."""
    width = 2 * math.pi * TUBE_R_MID / N_WALL * 1.02
    holes = []                                   # (angle, half-width along the wall, z bottom, z top)
    for name in RODS:
        z = RODS[name][2]
        for phi, slant in wall_crossings(name):
            half = 0.5 * (2 * ROD_R / math.cos(slant) + TUBE_WALL * math.tan(slant)) + WINDOW
            holes.append((phi, half, z - ROD_R - WINDOW, z + ROD_R + WINDOW))
    out = []
    for k in range(N_WALL):
        theta = 2 * math.pi * k / N_WALL
        cuts = []
        for phi, half, z0, z1 in holes:
            gap = abs((theta - phi + math.pi) % (2 * math.pi) - math.pi) * TUBE_R_MID
            if gap < width / 2 + half:
                cuts.append((z0, z1))
        spans, z = [], 0.0
        for z0, z1 in sorted(cuts):
            if z0 > z:
                spans.append((z, z0))
            z = max(z, z1)
        spans.append((z, TUBE_H))
        for a, b in spans:
            c = (TUBE_R_MID * math.cos(theta), TUBE_R_MID * math.sin(theta), (a + b) / 2)
            out.append((c, (TUBE_WALL, width, b - a), (0.0, 0.0, math.degrees(theta))))
    return out
