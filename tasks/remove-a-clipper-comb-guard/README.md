# Remove a clipper's comb guard and put it in a tray

A cordless hair clipper lies on the mat with its #4 comb guard fitted. The guard is located on the blade at the front
and held at the back by a sprung metal clip. The robot must take the guard off and leave it lying in the tray.

**Instruction:** `Remove the comb guard from the switched-off hair clipper and put the guard in the collection tray.`

**Category:** Societal relatability

**Context:** People who cut hair or trim beards at home remove a clipper's comb guard when changing cutting lengths and
exposing the blade area for cleaning. This task completes that familiar grooming preparation step: the fitted guard is
removed and put aside while the clipper and its blade assembly remain intact. It evaluates handling the real
attachment on an everyday personal-care tool.

## Status

Simulation only. No physical clipper, guard or robot was used. The guard's and clipper's dimensions, masses and the
clip's spring rate are estimates for a Wahl Elite Pro with a Secure-Snap #4 guard, and must be replaced by measurements
of the sample (including a five-cycle release check of the clip) before the task is frozen. The scripted demo has been
checked in a PyBullet stand-in (`check_demo.py`), not yet in Isaac Sim.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: clipper, guard, clip, tray; the detached and stowed checks; `score()`, `demo()` and the counterexamples |
| `housekit.py` | Multi-part objects with sliding and hinged pieces, spring preload, per-step watchers (mechkit from an accepted task, plus additions listed at its top) |
| `household.py` | The mat, the tray, and the geometric checks shared by this group of tasks |
| `check_demo.py` | PyBullet stand-in for yam_arena: plays `demo()` or a counterexample and prints the score (`pip install pybullet`) |
| `test_remove_a_clipper_comb_guard.py` | Offline tests: every score tier, the detached and stowed definitions, demo reach |
| `run.sh` | Runs the task in Isaac Sim and writes the videos to the home folder |
| `task.json` | Submission details |

## Objects

Mat coordinates in mm: origin at the mat's centre, x to the robot's right, y away from the robot, z up from the mat's
top. The mat's centre is 300 mm in front of the midpoint between the two arm bases.

| Object | Size and mass | Pose | Source |
|---|---|---|---|
| Silicone mat | 600 x 400 x 2 mm, fixed, friction 0.9 | centre at the origin | brief |
| Clipper housing | 160 long x 46 wide x 34 tall; clipper 300 g, friction 0.8 | long axis along y, blade end +y, centre x 0, y -80, lying on a broad face | estimate |
| Blade teeth plate | projects 8 mm past the housing's front, 44 wide, 4 thick, at z 23 to 27 | part of the clipper body | estimate |
| Catch on the housing | post 4 long x 24 wide x 2.5 tall with a 2 mm cap overhanging 4 mm toward the back | on the housing's top, 112 to 116 mm from its back end | estimate |
| Comb guard (#4, 13 mm) | 79.5 long x 51 wide x 43 tall, 12 g plastic, friction 0.5 | fully fitted | estimate |
| Guard's metal clip | strip 16 wide hanging from a hinge on the guard's back end, barb reaching 1.8 mm under the cap; 3 g; spring 0.05 N m/rad, preloaded | caught | estimate |
| Collection tray | inside floor 120 (x) x 100 (y), walls 15 tall, 2 thick, floor 2 thick, 60 g, loose | floor centre x +180, y -80 | brief; wall thickness and mass estimated |

Layout change from the brief, frozen: the clipper's centre is at y -80 instead of y 0, so that the guard's forward
withdrawal and the lift stay inside the arm's top-down reach. The clipper rests on the broad face that puts the
guard's clip on top. The clipper is switched off with no cable; blade and housing are one body.

How the guard is held:

- **Front:** an upper lip lies along the blade's top and a short hook tucks under the blade's last 2 mm. The front can
  only leave by sliding forward past the blade's tip, though the guard's back can tip up about that tip.
- **Sides:** the comb's two side walls hug the housing's sides.
- **Back:** the metal clip hangs from a hinge on the guard's spine. Its barb sits under the cap of the housing's catch,
  pressed there by the clip's spring.

Lifting the guard's back pushes the barb's sloped top against the cap's edge, which springs the clip back until the
barb slips past: a snap fit. Pulling the guard straight forward instead runs the barb into the catch's post. Nothing
else attaches the guard: there is no joint, weld or release rule between guard and clipper.

## Scoring

Final state, highest tier fully met.

| Score | Final conditions |
|---|---|
| 0 | The guard is not detached, or the clipper, guard or tray fell off the table at any time |
| 0.5 | Guard detached but not stowed in the tray (it may still be held) |
| 1.0 | Guard detached and stowed in the tray |

- **Detached:** the forward lip is off the blade, the clip's barb is out from under the catch, and the least distance
  between the guard and the clipper is at least 10 mm.
- **Stowed:** the guard rests on the tray's floor, its whole overhead footprint is inside the inner floor boundary (1 mm
  tolerance, so a guard leaning on a wall's inner face counts), no finger pad is within 2 mm of it, and this has
  lasted 2 s. Any orientation is accepted.
- Holding the clipper does not reduce credit. No clipper pose, arm pose or hand assignment is scored.
- The clipper and its blade are one rigid body here, so "blade assembly intact" cannot fail in simulation; in a real
  trial it is checked by inspection.

## Scripted demo and checks

`demo()`, right arm: grip the comb's side walls near their back, tip the guard's back up 8 mm (the clip snaps off its
catch), draw the guard 28 mm forward off the blade, lift, turn it along the tray's long side, lower, let go and wait.
530 control steps (26.5 s) of the 2,400 allowed.

PyBullet results (`python check_demo.py`, `--sweep`, `--counter NAME`):

| Run | Score |
|---|---|
| `demo()`, nominal | 1.0 |
| `demo()`, friction x0.7 / x1.4, arm stiffness x0.7 / x1.4, both low, both high | 1.0 in all seven |
| `pull_forward`: grip the walls and pull straight forward without tipping the back up | 0.0 (the barb meets the post; the guard stays on) |
| `keep_holding`: full removal, guard held over the tray to the end | 0.5 |
| `drop_beside_tray`: full removal, guard let go on the mat beside the tray | 0.5 |

These are simulation results in a stand-in, not real-robot trials and not Isaac Sim runs.

## Running it

```bash
python -m pytest -q                 # offline tests
python check_demo.py                # PyBullet check of the demo (needs pybullet)
bash run.sh                         # Isaac Sim, scripted demo
bash run.sh dream                   # Isaac Sim, Dream-1
```

## Reset

1. Inspect the clipper and guard for deformation or loosened fasteners.
2. Refit the same guard: forward lip and hook over the blade's tip first, then press the back down until the clip
   snaps under the catch. Check both are seated. A loose or pre-released guard is an invalid reset.
3. Put the clipper back on its marked pose and empty the tray.

## What is approximated

- All dimensions, masses and the clip's spring are estimates; the real Secure-Snap clip's geometry and release force
  were not measured.
- Everything is rigid boxes. The clip's flexing is reduced to one sprung hinge; the comb's teeth are a solid block.
- The clipper, blade and catch are one body, so breakage cannot occur.
- The mat is rigid.

## Duplicate check

A keyword search of the public repository's pull requests on 13 November 2026 for "clipper" and "comb guard" returned
no results. This is a keyword search only, not a review of the task board.
