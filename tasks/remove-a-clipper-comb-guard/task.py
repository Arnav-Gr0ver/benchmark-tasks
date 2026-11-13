"""Remove the comb guard from the switched-off hair clipper and put the guard in the collection tray.

A Wahl Elite Pro-type cordless clipper lies on its broad face on the mat, blade end away from the robot, its #4 comb
guard fitted. The guard is held two ways:
- at the front, a lip lies along the blade's top and a short hook tucks under the blade's tip, so the front can only
  leave the blade by sliding forward past its teeth (the back can tip up about the tip);
- at the back, a sprung metal clip hangs from a hinge on the guard's spine; its barb sits under a cap on the housing.
  Lifting the guard's back pushes the barb's sloped top against the cap's edge, springing the clip back until the barb
  slips past (a snap fit); pulling the guard forward instead runs the barb into the cap's post.
Nothing else attaches it: no joint, weld or release rule between the guard and the clipper.

Scoring (final state):
- 0: anything fell off the table, or the guard is not detached (its lips still on the blade, its hook still behind the
  ridge, or less than 10 mm from the clipper anywhere).
- 0.5: detached but not in the tray (it may still be held).
- 1.0: detached, lying on the tray's floor wholly inside it, no finger on it, still for 2 s.
The clipper and its blade are one rigid body here, so neither can break or come apart.
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
from housekit import Joint, Mech, box, every_step, on_reset, store  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
KEY = "remove-a-clipper-comb-guard"

# -- the clipper (estimates for a Wahl Elite Pro; measure the sample). Frame: origin at the housing's underside centre,
# x along the clipper toward its blade, z up from the face it lies on. -----------------------------------------------
HOUSING = (0.160, 0.046, 0.034)
CLIPPER_MASS = 0.300
BLADE = (0.008, 0.044, 0.004)           # the blade's exposed teeth plate: length past the housing, width, thickness
BLADE_X0 = HOUSING[0] / 2               # the housing's front face
BLADE_Z = 0.025                         # the blade plate's mid-plane
CLIPPER_XY = H.brief_xy(0.0, -80.0)     # 80 mm toward the robot from the mat's centre, for reach (frozen)
BLACK, SILVER, RED = (0.08, 0.08, 0.09), (0.75, 0.76, 0.78), (0.70, 0.08, 0.08)

TOP_Z = HOUSING[2]
GAP = 0.0005
# The housing's catch for the clip: a short post with a cap overhanging toward the back (an undercut the clip's barb
# sits under).
POST_X, POST_H = (0.0320, 0.0360), 0.0025
CAP_X, CAP_T = (0.0280, 0.0360), 0.0020
CAP_TOP = TOP_Z + POST_H + CAP_T

CLIPPER = Mech("clipper", pos=(CLIPPER_XY[0], CLIPPER_XY[1], H.MAT_TOP), mass=CLIPPER_MASS, friction=0.8, shapes=[
    box((0, 0, HOUSING[2] / 2), HOUSING, BLACK),
    box((BLADE_X0 + BLADE[0] / 2, 0, BLADE_Z), BLADE, SILVER),
    box(((POST_X[0] + POST_X[1]) / 2, 0, TOP_Z + POST_H / 2), (POST_X[1] - POST_X[0], 0.024, POST_H), BLACK),
    box(((CAP_X[0] + CAP_X[1]) / 2, 0, TOP_Z + POST_H + CAP_T / 2), (CAP_X[1] - CAP_X[0], 0.024, CAP_T), BLACK),
])


def _mid(a):
    return (a[0] + a[1]) / 2


def _len(a):
    return a[1] - a[0]


def _rot_y(deg):
    r = math.radians(deg) / 2
    return (math.cos(r), 0.0, math.sin(r), 0.0)


# -- the guard (a #4, 13 mm cutting length; plastic comb with a metal clip; estimates). Same frame as the clipper. ----
# At the front an upper lip lies along the blade's top and a short hook tucks under the blade's tip: the guard can only
# leave the blade by sliding forward, but its back can tip up about the tip. Along the housing's top runs the spine,
# with the comb's two side walls hugging the housing's sides. At the back, the metal clip hangs from a hinge on the
# spine; its barb sits under the housing's cap. Lifting the guard's back pushes the barb's sloped top against the cap's
# edge, which springs the clip back until the barb slips past: a snap fit.
LIP_GAP = 0.0003
LIP_X = (BLADE_X0 + GAP, BLADE_X0 + BLADE[0])
TIP_HOOK_X = (BLADE_X0 + BLADE[0] - 0.002, BLADE_X0 + BLADE[0])      # only under the blade's last 2 mm
COMB_X = (BLADE_X0 + BLADE[0] + GAP, BLADE_X0 + BLADE[0] + GAP + 0.013)
SPINE_Z = (CAP_TOP + GAP, CAP_TOP + GAP + 0.003)
SPINE_X = (0.022, COMB_X[0])
GUARD_W = 0.046
WALL_X = (0.040, COMB_X[1])
WALL_Y = (HOUSING[1] / 2 + GAP, HOUSING[1] / 2 + GAP + 0.002)
WALL_Z = (BLADE_Z - 0.004, TOP_Z + 0.013)
GUARD_MASS = 0.012
CLIP_MASS = 0.003
CLIP_SPRING = 0.05              # N m/rad, holding the barb under the cap (estimate; measure the clip's release force)
CLIP_TRAVEL = 0.50              # rad

_lz_lo, _lz_hi = BLADE_Z - BLADE[2] / 2, BLADE_Z + BLADE[2] / 2
GUARD_SHAPES = [
    box((_mid(COMB_X), 0, (0.004 + SPINE_Z[1]) / 2), (_len(COMB_X), GUARD_W, SPINE_Z[1] - 0.004), RED),     # nose
    box((_mid(TIP_HOOK_X), 0, _lz_lo - LIP_GAP - 0.002), (_len(TIP_HOOK_X), GUARD_W - 0.006, 0.004), RED),  # hook
    box((_mid(LIP_X), 0, _lz_hi + LIP_GAP + 0.002), (_len(LIP_X), GUARD_W - 0.006, 0.004), RED),           # upper lip
    box((_mid(LIP_X), 0, (_lz_hi + LIP_GAP + 0.004 + SPINE_Z[1]) / 2),
        (_len(LIP_X), GUARD_W, SPINE_Z[1] - _lz_hi - LIP_GAP - 0.004), RED),
    box((_mid(SPINE_X), 0, _mid(SPINE_Z)), (_len(SPINE_X), GUARD_W, _len(SPINE_Z)), RED),                  # spine
    box((_mid((0.040, LIP_X[0])), 0, (TOP_Z + GAP + SPINE_Z[0]) / 2),
        (_len((0.040, LIP_X[0])), GUARD_W, SPINE_Z[0] - TOP_Z - GAP), RED),                                 # under it
]
for _s in (1, -1):
    GUARD_SHAPES.append(box((_mid(WALL_X), _s * _mid(WALL_Y), _mid(WALL_Z)),
                            (_len(WALL_X), _len(WALL_Y), _len(WALL_Z)), RED))
# The clip: a strip hanging from a hinge on the spine's back end, a barb on its foot reaching forward under the cap.
HINGE = (0.0235, 0.0, SPINE_Z[0] + 0.001)
STRIP = (0.0015, 0.016, HINGE[2] - (TOP_Z + GAP))
BARB_X = (HINGE[0] + STRIP[0] / 2, CAP_X[0] + 0.0018)               # reaches 1.8 mm under the cap
BARB_Z = (TOP_Z + GAP, TOP_Z + POST_H - 0.0003)
RAMP_DEG = 35.0                 # the slope on the barb's top, which the cap's edge rides
CLIP = Joint("clip", "hinge", axis="y", at=HINGE, low=0.0, high=CLIP_TRAVEL, spring=CLIP_SPRING, target=-0.05,
             damping=0.002, mass=CLIP_MASS, friction=0.3, shapes=[
                 box((0, 0, -STRIP[2] / 2), STRIP, SILVER),
                 box((_mid(BARB_X) - HINGE[0], 0, _mid(BARB_Z) - HINGE[2] - 0.0004),
                     (_len(BARB_X), STRIP[1], _len(BARB_Z) - 0.0008), SILVER),
                 # its sloped top: a thin plate along the barb, its front end low
                 box((_mid(BARB_X) - HINGE[0], 0, BARB_Z[1] - HINGE[2] - 0.0002),
                     (0.004, STRIP[1], 0.0008), SILVER, tilt=_rot_y(RAMP_DEG)),
             ])
GUARD = Mech("guard", pos=(CLIPPER_XY[0], CLIPPER_XY[1], H.MAT_TOP), mass=GUARD_MASS, friction=0.5,
             shapes=GUARD_SHAPES, joints=[CLIP])

TRAY_XY = H.brief_xy(180.0, -80.0)
TRAY = H.tray("tray", TRAY_XY)
OBJECTS = [H.mat(), CLIPPER, GUARD, TRAY]


# -- checks -----------------------------------------------------------------------------------------------------------

def _local(clipper, point):
    return clipper.rotation().T @ (np.asarray(point, float) - clipper.origin)


def lips_on_blade(guard, clipper) -> bool:
    """The upper lip still over the blade's plate (the front still located on the blade)."""
    c = GUARD_SHAPES[2]
    p = _local(clipper, guard.to_world(c.at))
    return bool(abs(p[1]) < GUARD_W / 2 and abs(p[2] - BLADE_Z) < 0.008 and p[0] - c.size[0] / 2 < BLADE_X0 + BLADE[0])


def hooked(guard, clipper) -> bool:
    """The clip's barb still under the cap: its tip ahead of the cap's back edge and below the cap."""
    R = guard.rotation()
    a = guard.joints.get("clip", 0.0)
    ca, sa = math.cos(a), math.sin(a)
    Rj = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    tip = guard.origin + R @ (np.array(HINGE) + Rj @ np.array([BARB_X[1] - HINGE[0], 0, _mid(BARB_Z) - HINGE[2]]))
    p = _local(clipper, tip)
    return bool(p[0] > CAP_X[0] and p[0] < POST_X[0] + 0.001 and p[2] < TOP_Z + POST_H + 0.0005
                and abs(p[1]) < 0.012 + STRIP[1] / 2)


def detached(objs) -> bool:
    g, c = objs["guard"], objs["clipper"]
    return (not lips_on_blade(g, c) and not hooked(g, c) and H.gap(GUARD, g, CLIPPER, c) >= 0.010)


S = store(KEY)


def _fresh(_env=None) -> None:
    S.clear()
    S.update(fell=False, timer=H.Timer(), held=False, stowed_for=0.0)


_fresh()
on_reset(_fresh)


@every_step(KEY, ["clipper", "guard", "tray"])
def _watch(objs, fingers) -> None:
    if any(H.fallen(objs[n]) for n in ("clipper", "guard", "tray")):
        S["fell"] = True
    g = objs["guard"]
    S["held"] = H.touching(GUARD, g, fingers)
    ok = (not S["held"] and H.in_tray(GUARD, g, TRAY, objs["tray"]) and detached(objs))
    S["stowed_for"] = S["timer"].update(ok)


def score(objs) -> float:
    if S["fell"] or any(H.fallen(objs[n]) for n in ("clipper", "guard", "tray")):
        return 0.0
    if not detached(objs):
        return 0.0
    if S["stowed_for"] >= H.STILL_S:
        return 1.0
    return 0.5


# -- scripted demo: the right hand grips the comb's side walls near their back, tips the guard's back up (the clip snaps
# off its catch), draws the guard forward off the blade and carries it to the tray. ------------------------------------

PAD_TIP = 0.0138
HOVER = H.TOP + 0.11
GRIP_AT = 0.052                 # along the clipper, on the walls' back part
GRIP_Z = WALL_Z[0] + 0.002      # the fingertips' height above the mat: the pads cover the walls nearly top to bottom
TURN = math.pi / 2
SETTLE = 0.010
NOSE_CLEAR = 0.012              # the comb's nose above the tray floor when the guard is let go


def demo():
    cx, cy = CLIPPER_XY
    gx = cx + GRIP_AT
    gz = H.MAT_TOP + GRIP_Z + PAD_TIP               # fingertips beside the housing, the pads flat on the walls
    tip_up = 0.008                                  # lift at the grip: the guard tips ~10 degrees about the blade's tip
    slide = LIP_X[1] - LIP_X[0] + 0.020             # forward past the blade's teeth, with the arm's give
    # from the grip to the guard's middle, along the guard, less SETTLE: let go level, the guard rocks back onto its
    # walls' back corners and ends about 16 mm toward its back
    d = (SPINE_X[0] + COMB_X[1]) / 2 - GRIP_AT - SETTLE
    turn = TURN                                     # the guard turned so its length lies along the tray's long side
    tx, ty = TRAY_XY[0] - d * math.cos(turn), TRAY_XY[1] - d * math.sin(turn)
    drop_z = H.MAT_TOP + H.TRAY_FLOOR_T + (GRIP_Z - 0.004) + NOSE_CLEAR + PAD_TIP
    return {"right": [
        Waypoint(2.5, (gx, cy, HOVER), FINGER_OPEN, "over the comb's walls"),
        Waypoint(2.0, (gx, cy, gz), FINGER_OPEN, "down either side of the walls"),
        Waypoint(1.0, (gx, cy, gz), FINGER_CLOSED, "grip the walls"),
        Waypoint(1.5, (gx + 0.001, cy, gz + tip_up), FINGER_CLOSED, "tip the guard's back up: the clip snaps off"),
        Waypoint(0.5, (gx + 0.001, cy, gz + tip_up), FINGER_CLOSED, "hold"),
        Waypoint(2.5, (gx + slide, cy, gz + tip_up + 0.004), FINGER_CLOSED, "draw the guard forward off the blade"),
        Waypoint(1.5, (gx + slide, cy, HOVER), FINGER_CLOSED, "lift the guard clear"),
        Waypoint(3.0, (tx, ty, HOVER), FINGER_CLOSED, "over the tray, turning the guard along it", turn),
        Waypoint(2.5, (tx, ty, drop_z), FINGER_CLOSED, "lower into the tray", turn),
        Waypoint(1.0, (tx, ty, drop_z), FINGER_CLOSED, "settle", turn),
        Waypoint(1.0, (tx, ty, drop_z), FINGER_OPEN, "let go", turn),
        Waypoint(1.5, (tx, ty, HOVER), FINGER_OPEN, "up and away", turn),
        Waypoint(3.0, (tx, ty, HOVER), FINGER_OPEN, "observe", turn),
    ]}


# -- counterexamples for the scorer and the retention (python check_demo.py --counter NAME) --------------------------

def _pull_forward():
    """Grip the walls and draw the guard straight forward without tipping its back up."""
    cx, cy = CLIPPER_XY
    gx, gz = cx + GRIP_AT, H.MAT_TOP + GRIP_Z + PAD_TIP
    return {"right": [
        Waypoint(2.5, (gx, cy, HOVER), FINGER_OPEN, "over the comb's walls"),
        Waypoint(2.0, (gx, cy, gz), FINGER_OPEN, "down either side of the walls"),
        Waypoint(1.0, (gx, cy, gz), FINGER_CLOSED, "grip the walls"),
        Waypoint(2.5, (gx + 0.030, cy, gz), FINGER_CLOSED, "pull straight forward"),
        Waypoint(1.0, (gx + 0.030, cy, gz), FINGER_OPEN, "let go"),
        Waypoint(1.5, (gx + 0.030, cy, HOVER), FINGER_OPEN, "up"),
    ]}


def _keep_holding():
    """The full removal, but the guard is held over the tray and never put down."""
    plan = demo()["right"]
    k = next(i for i, w in enumerate(plan) if w.label.startswith("lower into"))
    over = plan[k - 1]
    return {"right": plan[:k] + [Waypoint(4.0, over.pos, FINGER_CLOSED, "hold it over the tray", over.yaw)]}


def _drop_beside_tray():
    """The full removal, but the guard is let go on the mat beside the tray."""
    out = []
    for w in demo()["right"]:
        if w.yaw:
            w = Waypoint(w.duration, (w.pos[0] + 0.100, w.pos[1], w.pos[2]), w.finger, w.label, w.yaw)
        out.append(w)
    return {"right": out}


COUNTEREXAMPLES = {
    "pull_forward": (_pull_forward, 0.0, "the clip's barb meets the catch's post: the guard stays on"),
    "keep_holding": (_keep_holding, 0.5, "detached but still held"),
    "drop_beside_tray": (_drop_beside_tray, 0.5, "detached and released, but not in the tray"),
}
