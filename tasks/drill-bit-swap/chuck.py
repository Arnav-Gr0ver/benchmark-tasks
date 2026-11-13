"""The drill, its keyless chuck and the bits: geometry and the chuck's thread logic, shared by the task file, its
scorer, its tests and the physics check. Pure numpy.

Drill frame (metres): origin on the chuck axis, x along it toward the bit, the handle along +y, z up when the drill
lies on its side (as it does in the task). The drill is one articulation: the body (root, free on the table), the
sleeve on a hinge about the chuck axis, and three jaws on radial slides in front of the sleeve.

The thread inside a real keyless chuck turns sleeve rotation into jaw travel and cannot be driven backwards by the
jaws. Here that is the class Thread: each control step it reads how far the sleeve has turned and moves the jaws'
position drives by LEAD per turn. The drives are stiff springs (JAW_K), so the jaws move together and centre what they
hold; the thread winds at most WIND past a stalled jaw, so each jaw clamps with up to CLAMP_N (a real chuck's limit
is the torque a hand can put on the sleeve); turning further winds no tighter. Nothing moves the jaws but the sleeve.
"""

from __future__ import annotations

import math

import numpy as np

# -- drill body ------------------------------------------------------------------------------------------------------
AXIS_H = 0.035                                  # the chuck axis above the table, the drill on its side
HOUSING = ((-0.020, 0.0, 0.0), 0.035, 0.110)    # motor housing: a cylinder along x (centre, radius, length)
HANDLE = ((-0.045, 0.090, 0.0), (0.036, 0.120, 0.040))      # box (centre, size)
FOOT = ((-0.040, 0.160, 0.0), (0.075, 0.026, 0.070))        # the battery foot (battery removed)
BODY_MASS = 1.0
BODY_COM = (-0.030, 0.045, 0.0)

# -- chuck -----------------------------------------------------------------------------------------------------------
SLEEVE_R, SLEEVE_X = 0.018, (0.037, 0.082)      # the sleeve: a cylinder along x, 36 mm across, 45 mm long
SLEEVE_MASS = 0.08
SLEEVE_DAMPING = 0.15                           # N m s / rad: it turns stiffly and stays where it is left
JAW_X = (0.082, 0.104)                          # the jaws' gripping faces: 22 mm long, in front of the sleeve
JAW_FRONT = JAW_X[1]
JAW_ANGLES = (270.0, 30.0, 150.0)               # round the axis in the y-z plane, from +y toward +z: one underneath
JAW_T, JAW_W = 0.008, 0.005                     # a jaw: 8 mm radially, 5 mm across
JAW_MASS = 0.050
R_MIN, R_MAX = 0.0005, 0.0052                   # the gripping faces' distance from the axis: 1 to 10.4 mm opening
R0 = 0.0020                                     # at the start: closed on the 4 mm bit (the jaw joints' zero)
LEAD = 0.005                                    # radial jaw travel per turn of the sleeve (10 mm of opening per turn)
CLOSE_SIGN = -1.0                               # tightens clockwise seen from the front: a turn about the drill's -x
JAW_K, JAW_D = 5.0e5, 300.0                     # each jaw's position drive: N/m, N s/m
CLAMP_N = 150.0                                 # the most a jaw pushes
WIND = CLAMP_N / JAW_K                          # the drive target that far inside a stalled jaw: 0.3 mm
JAW_FRICTION, BIT_FRICTION = 0.3, 0.6           # hardened steel jaws and bore: 0.3 against anything ("min")
# The chuck body's nose: between the jaws, three fixed lands whose faces make the bore (with the open jaws, a
# six-sided hole 10.8 mm across) that keeps a loose bit from falling out sideways.
BORE_R = 0.0054
LAND_ANGLES = (90.0, 210.0, 330.0)
LAND_T, LAND_W = 0.006, 0.0062                  # 6 mm radially, 6.2 mm across
MAX_DEPTH = JAW_X[1] - SLEEVE_X[1]              # a bit bottoms out on the sleeve's face: 22 mm in


def jaw_dir(k: int) -> np.ndarray:
    a = math.radians(JAW_ANGLES[k])
    return np.array([0.0, math.cos(a), math.sin(a)])


def lands():
    """The nose's lands as boxes in the drill frame: (centre, size (along x, radial, across), turn about x in deg)."""
    out = []
    for a in LAND_ANGLES:
        u = np.array([0.0, math.cos(math.radians(a)), math.sin(math.radians(a))])
        centre = np.array([(JAW_X[0] + JAW_X[1]) / 2, 0.0, 0.0]) + u * (BORE_R + LAND_T / 2)
        out.append((tuple(centre), (JAW_X[1] - JAW_X[0], LAND_T, LAND_W), a))
    return out


class Thread:
    """The chuck's thread. ``radius`` is where the jaws' drives are told to be (the faces' distance from the axis).
    Turning the sleeve clockwise seen from the front (about the drill's -x) by 1 rad closes the jaws by
    LEAD / (2 pi)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.radius = R0 - WIND                 # tightened on the 4 mm bit
        self.angle = None

    def step(self, sleeve_angle: float, jaw_radii) -> float:
        """One control step: the sleeve's joint angle (rad, may wrap) and the three jaws' measured radii. Returns the
        new commanded radius."""
        turned = 0.0
        if self.angle is not None:
            turned = (sleeve_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.angle = sleeve_angle
        travel = CLOSE_SIGN * LEAD * turned / (2 * math.pi)     # > 0: closing
        if travel > 0:      # closing: no further than WIND inside the jaw that is furthest out (they move together)
            stalled = max(float(r) for r in jaw_radii) - WIND
            self.radius = min(self.radius, max(self.radius - travel, stalled, R_MIN - WIND))
        else:               # opening, up to the end stop. A jaw knocked outward never loosens the thread.
            self.radius = min(self.radius - travel, R_MAX)
        return self.radius

    def clamp_force(self, jaw_radii) -> float:
        """The least any jaw pushes on what it holds (N)."""
        return min(max(0.0, JAW_K * (float(r) - self.radius)) for r in jaw_radii)


# -- bits (bit frame: x along the bit from the shank's end toward the tip, origin at its middle) ----------------------
BITS = {
    # name: (diameter, length, flute length, mass, colour)
    "old_bit": (0.004, 0.075, 0.043, 0.0060, (0.30, 0.30, 0.33)),     # 4 mm jobber drill (DIN 338), bright steel
    "new_bit": (0.006, 0.093, 0.057, 0.0170, (0.85, 0.65, 0.20)),     # 6 mm jobber drill, gold (TiN)
}
NOMINAL_DEPTH = 0.019
DEPTH_OK = (0.014, MAX_DEPTH + 0.0005)          # how far the shank's end is past the jaws' front: valid insertion
TILT_OK_DEG, OFFSET_OK = 15.0, 0.003            # inserted (may be loose): within 15 degrees and 3 mm of the axis
SEATED_TILT_DEG, SEATED_OFFSET = 3.0, 0.0006    # clamped: within 3 degrees and 0.6 mm of it


def bit_parts(name: str):
    """The bit's colliders in its own frame: the smooth shank (a cylinder) and the fluted length (a hexagonal prism,
    across flats = the diameter: three boxes a sixth of a turn apart), for shapekit.Compound."""
    d, length, flute, _, _ = BITS[name]
    shank = length - flute
    parts = [((-length / 2 + shank / 2, 0.0, 0.0), ("cyl", d / 2, shank, "X"))]
    for k in range(3):
        parts.append(((length / 2 - flute / 2, 0.0, 0.0), (flute, d, d * math.tan(math.radians(30.0))),
                      (60.0 * k, 0.0, 0.0)))
    return tuple(parts)


def bit_in_chuck(name: str, depth: float = NOMINAL_DEPTH) -> np.ndarray:
    """The bit's middle in the drill frame when it is on the axis, its shank ``depth`` past the jaws' front."""
    return np.array([JAW_FRONT - depth + BITS[name][1] / 2, 0.0, 0.0])


def insertion(name: str, centre, axis) -> tuple[float, float, float]:
    """A bit against the chuck, from its middle and its direction (shank to tip) in the drill frame: (depth of its
    shank's end past the jaws' front, m; tilt from the chuck axis, deg; distance of its axis from the chuck's at the
    jaws' middle, m)."""
    centre, axis = np.asarray(centre, float), np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    end = centre - axis * BITS[name][1] / 2
    depth = JAW_FRONT - float(end[0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(axis[0])))))
    mid_x = (JAW_X[0] + JAW_X[1]) / 2
    if abs(axis[0]) < 1e-6:
        return depth, tilt, math.inf
    at_mid = centre + axis * (mid_x - centre[0]) / axis[0]
    return depth, tilt, float(math.hypot(at_mid[1], at_mid[2]))


def inserted(name: str, centre, axis) -> bool:
    depth, tilt, off = insertion(name, centre, axis)
    return DEPTH_OK[0] <= depth <= DEPTH_OK[1] and tilt <= TILT_OK_DEG and off <= OFFSET_OK


def seated(name: str, centre, axis) -> bool:
    depth, tilt, off = insertion(name, centre, axis)
    return DEPTH_OK[0] <= depth <= DEPTH_OK[1] and tilt <= SEATED_TILT_DEG and off <= SEATED_OFFSET


def removed(name: str, centre, axis) -> bool:
    """No part of the bit is between the jaws: every point of its centre line is in front of the jaws' front by
    2 mm, or more than 2 cm from the chuck axis."""
    centre, axis = np.asarray(centre, float), np.asarray(axis, float)
    half = BITS[name][1] / 2
    for s in np.linspace(-half, half, 23):
        p = centre + axis * s
        if p[0] < JAW_FRONT + 0.002 and math.hypot(p[1], p[2]) < 0.020:
            return False
    return True


# -- the retention test ------------------------------------------------------------------------------------------------
PULL_N = 50.0
PULL_RAMP_S, PULL_HOLD_S, PULL_UNLOAD_S, PULL_WATCH_S = 0.5, 1.0, 0.5, 1.0
SLIP_OK = 0.001


def pull_at(t: float) -> float:
    """The axial pull (N) t seconds into the test."""
    if t < PULL_RAMP_S:
        return PULL_N * t / PULL_RAMP_S
    if t < PULL_RAMP_S + PULL_HOLD_S:
        return PULL_N
    if t < PULL_RAMP_S + PULL_HOLD_S + PULL_UNLOAD_S:
        return PULL_N * (1 - (t - PULL_RAMP_S - PULL_HOLD_S) / PULL_UNLOAD_S)
    return 0.0


PULL_TOTAL_S = PULL_RAMP_S + PULL_HOLD_S + PULL_UNLOAD_S + PULL_WATCH_S
