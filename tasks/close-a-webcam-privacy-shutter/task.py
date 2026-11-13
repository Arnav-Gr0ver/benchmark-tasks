"""Close the webcam's privacy shutter. Leave the webcam attached to its monitor.

A Logitech C925e-type webcam sits on a small monitor's top edge on its own clip, lens toward the robot, shutter open.
The shutter is a thin opaque plate in front of the lens on a slide across the camera's face; a tab on its top edge,
just above the camera's body, is how it is moved. It stays wherever it is left (the slide's damping stands in for the
real slider's friction). The webcam is free: only its clip, hooked over the monitor's top edge, holds it there.

Scoring (final state): C is the fraction of the lens aperture covered by the shutter, from the slide's position.
- 0: anything fell, the webcam came off its monitor at any time, or C < 0.25.
- 0.5: the webcam is on its monitor and 0.25 <= C < 1.
- 0.75: C = 1 but a finger is still on the shutter.
- 1.0: C = 1 with no finger on the shutter, through 2 s of observation.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

import household as H  # noqa: E402
from housekit import Joint, Mech, box, cyl, every_step, on_reset, store  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
KEY = "close-a-webcam-privacy-shutter"

# -- the monitor: a 7-inch portable display on its own foot (estimates; measure the chosen model) ----------------------
PANEL_X = 0.300                 # the panel's front face (toward the robot), station x
PANEL_T = 0.010                 # top-bezel thickness
PANEL_W, PANEL_H = 0.172, 0.110
PANEL_BOTTOM = H.MAT_TOP + 0.001
PANEL_TOP = PANEL_BOTTOM + PANEL_H
BLACK, GREY, DARK = (0.06, 0.06, 0.07), (0.45, 0.46, 0.48), (0.12, 0.12, 0.13)
MONITOR = Mech("monitor", pos=(PANEL_X + PANEL_T / 2, 0.0, PANEL_BOTTOM), mass=None, friction=0.6, shapes=[
    box((0, 0, PANEL_H / 2), (PANEL_T, PANEL_W, PANEL_H), BLACK),
    box((0.035, 0, 0.004), (0.080, 0.120, 0.008), DARK),                        # foot, behind the panel
    box((0.022, 0, 0.030), (0.010, 0.060, 0.050), DARK, tilt=(math.cos(0.2), 0, math.sin(0.2), 0)),  # stand arm
])

# -- the webcam (Logitech C925e: 126 x 73 x 45 mm and 170 g with its clip, from Logitech; the parts below are
# estimates of its body, clip and shutter for this model, to be measured on the sample) -----------------------------
BODY = (0.030, 0.126, 0.030)            # depth (x), width (y), height (z)
CLIP_T = 0.003
BODY_FRONT = PANEL_X - 0.012            # the body overhangs the panel's front face by 12 mm
Z0 = PANEL_TOP + CLIP_T                 # the body's underside
LENS_R = 0.0045                         # the lens entrance aperture's radius
LENS_Z = Z0 + BODY[2] / 2               # its centre; lens centre at y = 0 on the front face
SHUTTER = (0.002, 0.016, 0.020)         # plate: thickness, width (y, across the lens), height
SHUTTER_GAP = 0.0005                    # between the plate and the body's front face
OPEN_Y = 0.016                          # the plate's centre when open (to the robot's left of the lens)
TRAVEL = 0.018                          # the slide's travel to its closing stop
TAB = (0.004, 0.005, 0.005)             # the tab on the plate's top, above the body's top front edge
WEBCAM_MASS = 0.170
FULL = SHUTTER[1] / 2 - LENS_R          # the plate covers the lens fully within this of the lens centre

_wc_x = BODY_FRONT + BODY[0] / 2                    # webcam origin: middle of the body's underside
_plate_x = BODY_FRONT - SHUTTER_GAP - SHUTTER[0] / 2 - _wc_x
_tab_x = BODY_FRONT - SHUTTER_GAP - SHUTTER[0] + TAB[0] / 2 - _wc_x
_TAB_Z = BODY[2] / 2 + 0.0005 + TAB[2] / 2           # the tab's centre above the plate's centre
_STEM_H = BODY[2] / 2 + 0.0005 - SHUTTER[2] / 2       # from the plate's top to the tab's underside
_STEM_Z = SHUTTER[2] / 2 + _STEM_H / 2
WEBCAM = Mech("webcam", pos=(_wc_x, 0.0, Z0), mass=WEBCAM_MASS, friction=0.6, shapes=[
    box((0, 0, BODY[2] / 2), BODY, DARK),
    cyl((BODY_FRONT - _wc_x - 0.0005, 0, BODY[2] / 2), 0.009, 0.001, (0.15, 0.25, 0.55)),  # (thin lens ring, visual)
    # the clip: a plate on the panel's top edge, a lip down its front face and a leg down its back
    box((PANEL_X + PANEL_T / 2 - _wc_x, 0, -CLIP_T / 2), (PANEL_T + 0.008, 0.050, CLIP_T), GREY),
    box((PANEL_X - 0.0025 - _wc_x, 0, -0.008), (0.003, 0.050, 0.010), GREY),
    box((PANEL_X + PANEL_T + 0.0025 - _wc_x, 0, -0.022), (0.003, 0.050, 0.038), GREY),
], joints=[
    Joint("shutter", "slide", axis="y", at=(_plate_x, OPEN_Y, BODY[2] / 2), low=-TRAVEL, high=0.0, spring=0.0,
          damping=4.0, mass=0.002, friction=0.5, shapes=[
              box((0, 0, 0), SHUTTER, (0.02, 0.02, 0.02)),
              # a stem up the plate's top to the tab, which sits 0.5 mm above the body's top front edge
              box((0, 0, _STEM_Z), (SHUTTER[0], TAB[1], _STEM_H), (0.85, 0.85, 0.85)),
              box((_tab_x - _plate_x, 0, _TAB_Z), TAB, (0.85, 0.85, 0.85)),
          ])])
OBJECTS = [H.mat(), MONITOR, WEBCAM]


# -- coverage ---------------------------------------------------------------------------------------------------------

def _strip(a: float, b: float, r: float) -> float:
    """Area of the part of a circle (radius r, centre 0) with a <= y <= b."""
    def cum(t):
        t = max(-r, min(r, t))
        return t * math.sqrt(r * r - t * t) + r * r * math.asin(t / r)
    return max(0.0, cum(b) - cum(a))


def coverage(shutter: float) -> float:
    """C: the fraction of the lens aperture behind the plate, for the slide's position (0 open, -TRAVEL at its stop).
    The plate is taller than the aperture, so only its sides matter."""
    centre = OPEN_Y + shutter
    covered = _strip(-centre - SHUTTER[1] / 2, -centre + SHUTTER[1] / 2, LENS_R)
    c = covered / (math.pi * LENS_R ** 2)
    return 1.0 if c > 1 - 1e-9 else c


def attached(webcam) -> bool:
    """The clip still over the panel's top edge: the webcam within 4 mm of where it was mounted front to back, its clip
    plate within 5 mm above the edge, tipped less than 20 degrees, and still over the panel's width."""
    o, R = webcam.origin, webcam.rotation()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    return bool(abs(o[0] - _wc_x) < 0.004 and -0.002 < o[2] - Z0 < 0.005 and tilt < 20.0
                and abs(o[1]) < PANEL_W / 2 - 0.025)


# -- the watcher ------------------------------------------------------------------------------------------------------

S = store(KEY)


def _fresh(_env=None) -> None:
    S.clear()
    S.update(detached=False, fell=False, timer=H.Timer(), held=False)


_fresh()
on_reset(_fresh)


@every_step(KEY, ["webcam", "monitor", "mat"])
def _watch(objs, fingers) -> None:
    cam = objs["webcam"]
    if not attached(cam):
        S["detached"] = True
    if H.fallen(cam):
        S["fell"] = True
    S["held"] = H.touching(WEBCAM, cam, fingers)
    closed = coverage(cam.joints.get("shutter", 0.0)) >= 1.0
    S["closed_for"] = S["timer"].update(closed and not S["held"])


def score(objs) -> float:
    cam = objs["webcam"]
    if S["fell"] or S["detached"] or not attached(cam):
        return 0.0
    c = coverage(cam.joints.get("shutter", 0.0))
    if c < 0.25:
        return 0.0
    if c < 1.0:
        return 0.5
    if S["held"] or S.get("closed_for", 0.0) < H.STILL_S:
        return 0.75
    return 1.0


# -- scripted demo: the left hand's closed fingertips push the tab across -----------------------------------------------

PAD_TIP = 0.0138                # the closed fingertips' lowest point below the TCP
FACE = 0.0047                   # the closed fingers' outer face from the TCP, near the tips


def demo():
    tab_x = _wc_x + _tab_x
    tab_top = Z0 + BODY[2] + 0.0005 + TAB[2]
    tip_z = Z0 + BODY[2] + 0.0015                       # fingertips 1.5 mm above the body's top
    tcp_z = tip_z + PAD_TIP
    start_y = OPEN_Y + TAB[1] / 2 + FACE + 0.003         # the -y finger face 3 mm clear of the tab
    end_y = start_y - (TRAVEL - 0.001)                  # 1 mm short of the stop
    side_y = BODY[1] / 2 + FACE + 0.045                 # beside the webcam's end, clear of it
    return {"left": [
        Waypoint(2.5, (tab_x, side_y, tcp_z + 0.030), FINGER_CLOSED, "close the fingers, beside the webcam"),
        Waypoint(1.5, (tab_x, side_y, tcp_z), FINGER_CLOSED, "fingertips down to just above the body's top"),
        Waypoint(2.0, (tab_x, start_y, tcp_z), FINGER_CLOSED, "along the top to beside the tab"),
        Waypoint(3.0, (tab_x, end_y, tcp_z), FINGER_CLOSED, "push the shutter across the lens"),
        Waypoint(0.5, (tab_x, end_y, tcp_z), FINGER_CLOSED, "settle"),
        Waypoint(0.5, (tab_x, end_y + 0.004, tcp_z), FINGER_CLOSED, "off the tab"),
        Waypoint(1.0, (tab_x, end_y + 0.004, tcp_z + 0.012), FINGER_CLOSED, "up off the body"),
        Waypoint(2.0, (tab_x, side_y, tcp_z + 0.030), FINGER_CLOSED, "away to the side"),
        Waypoint(3.0, (tab_x, side_y, tcp_z + 0.030), FINGER_CLOSED, "observe"),
    ]}


# -- counterexamples for the scorer (python check_demo.py --counter NAME) ----------------------------------------------

def _push(travel, leave=True):
    plan = demo()["left"]
    k = next(i for i, p in enumerate(plan) if p.label.startswith("push the shutter"))
    start = plan[k - 1].pos
    end = (start[0], start[1] - travel, start[2])
    out = plan[:k] + [Waypoint(3.0, end, FINGER_CLOSED, "push the shutter part of the way"),
                      Waypoint(0.5, end, FINGER_CLOSED, "settle")]
    if leave:
        out += [Waypoint(0.5, (end[0], end[1] + 0.004, end[2]), FINGER_CLOSED, "off the tab"),
                Waypoint(1.0, (end[0], end[1] + 0.004, end[2] + 0.012), FINGER_CLOSED, "up off the body"),
                Waypoint(3.0, (end[0], end[1] + 0.004, end[2] + 0.012), FINGER_CLOSED, "observe")]
    else:
        out += [Waypoint(4.0, end, FINGER_CLOSED, "stay on the tab")]
    return {"left": out}


def _half_closed():
    """Push the tab 10 mm of its 18 mm and leave: the lens is partly covered."""
    return _push(0.010)


def _keep_holding():
    """Close the shutter fully but keep the fingers against the tab to the end."""
    return _push(TRAVEL + 0.001, leave=False)


def _tip_off():
    """Push the top of the webcam's body backward until it tips off the monitor's top edge."""
    x0, y = BODY_FRONT - 0.022, 0.045
    z = Z0 + BODY[2] - 0.008 + PAD_TIP                  # fingertips 8 mm below the body's top, in front of its face
    return {"left": [
        Waypoint(2.5, (x0, 0.113, z + 0.020), FINGER_CLOSED, "close the fingers, beside the webcam"),
        Waypoint(2.0, (x0, y, z + 0.020), FINGER_CLOSED, "in front of the webcam's body"),
        Waypoint(1.5, (x0, y, z), FINGER_CLOSED, "down in front of its face"),
        Waypoint(3.0, (x0 + 0.045, y, z), FINGER_CLOSED, "push the body's top backward"),
        Waypoint(2.0, (x0 + 0.045, y, z), FINGER_CLOSED, "hold"),
    ]}


COUNTEREXAMPLES = {
    "half_closed": (_half_closed, 0.5, "0.25 <= C < 1"),
    "keep_holding": (_keep_holding, 0.75, "C = 1 but a finger is still on the shutter"),
    "tip_off": (_tip_off, 0.0, "the webcam came off its monitor"),
}
