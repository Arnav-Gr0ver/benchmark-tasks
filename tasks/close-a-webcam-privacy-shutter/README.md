# Close a mounted webcam's privacy shutter

A webcam sits on a small monitor's top edge on its own clip, lens toward the robot, with its sliding privacy shutter
open. The robot must slide the shutter fully across the lens and let go, without taking the webcam off the monitor.

**Instruction:** `Close the webcam's privacy shutter. Leave the webcam attached to its monitor.`

**Category:** Societal relatability

**Context:** After a video call, people can close a webcam's physical privacy shutter to cover the lens while leaving
the camera mounted for the next call. This task completes that familiar privacy-control action: the built-in shutter
fully covers the lens and stays closed after release, while the webcam remains attached to its monitor.

## Status

Simulation only. No physical webcam, monitor or robot was used. Every part size below that is not a published
whole-unit figure is an estimate and must be replaced by measurements of the chosen sample before the task is frozen.
The scripted demo has been checked in a PyBullet stand-in (`check_demo.py`), not yet in Isaac Sim.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, coverage, the per-step watcher, `score()`, the scripted `demo()` and the counterexamples |
| `housekit.py` | Multi-part objects with sliding and hinged pieces, per-step watchers (mechkit from an accepted task, plus additions listed at its top) |
| `household.py` | The mat, the tray, and the geometric checks shared by this group of tasks |
| `check_demo.py` | PyBullet stand-in for yam_arena: plays `demo()` or a counterexample and prints the score (`pip install pybullet`) |
| `test_close_a_webcam_privacy_shutter.py` | Offline tests: coverage, every score tier, the latched zero, demo reach |
| `run.sh` | Runs the task in Isaac Sim and writes the videos to the home folder |
| `task.json` | Submission details |

## Objects

Mat coordinates in mm: origin at the mat's centre, x to the robot's right, y away from the robot, z up from the mat's
top. The mat's centre is 300 mm in front of the midpoint between the two arm bases.

| Object | Size and mass | Pose | Source |
|---|---|---|---|
| Silicone mat | 600 x 400 x 2 mm, fixed, friction 0.9 | centre at the origin | brief |
| Monitor panel | 172 wide x 110 tall x 10 thick, fixed to the table | centred x 0, front face y 0, z 1 to 111 | estimate (a 7-inch portable display) |
| Monitor foot and arm | foot 120 x 80 x 8, arm 60 x 10 x 50 | behind the panel | estimate |
| Webcam body | 126 wide x 30 deep x 30 tall; whole webcam 170 g | centred x 0, front face y -12, underside z 114 | width and mass from Logitech (126 x 73 x 45 mm, 170 g with clip); body depth and height estimated |
| Clip | 3 mm plate on the panel's top edge, 10 mm lip down the front, 38 mm leg down the back, 50 wide | under the body | estimate |
| Lens aperture | circle, 9 mm across | centre x 0, z 129, on the body's front face | estimate |
| Shutter plate | 16 wide x 20 tall x 2 thick, 2 g, opaque | 0.5 mm in front of the body's face; open: centre x -16 | estimate |
| Shutter tab | 5 (along the slide) x 4 x 5 tall, on a stem from the plate's top | above the body's top front edge, top at z 149.5 | estimate |
| Slide | 18 mm from open to its closing stop, toward +x; damping 4 N s/m | | estimate |

The lens faces -y (toward the robot). The tab is the only part of the shutter that stands clear of the body; it is not
enlarged. The plate covers the whole aperture once it has moved 12.5 mm; the closing stop is at 18 mm.

Not in the scene: the webcam's cable (routed behind the monitor in the real setup, outside the manipulation path), a
tray, any loose accessory.

## Scoring

Final state, highest tier fully met. C is the fraction of the lens aperture covered by the shutter plate, measured in
the aperture plane. In simulation it is computed exactly from the slide's position and the plate and aperture
geometry (`coverage()`); in a real trial it is read from a close, front-on image with a calibrated overlay.

| Score | Final conditions |
|---|---|
| 0 | The webcam came off its monitor at any time, something fell off the table, or C < 0.25 |
| 0.5 | Webcam attached and 0.25 <= C < 1 |
| 0.75 | C = 1, but a finger is still on the webcam, or C = 1 has not yet lasted 2 s after release |
| 1.0 | C = 1 with no finger contact throughout 2 s; webcam still attached |

- **Attached** means the clip is still over the panel's top edge: the webcam has moved less than 4 mm front to back,
  is no more than 5 mm above the edge, is tipped less than 20 degrees and is still over the panel's width. Coming off
  is latched for the whole run. A harmless change of tilt is not penalised.
- **Finger contact** is a finger-pad point within 2 mm of any part of the webcam.
- No arm pose and no hand assignment is scored.

## How the shutter works in simulation

The webcam is one articulation: a free body (the camera and its clip) with one sliding piece (plate, stem and tab).
Nothing fixes the webcam to the monitor; the clip's lip and leg straddle the panel's top edge and the webcam's weight
keeps it there. The shutter stays wherever it is left because the slide is damped; there is no detent and no spring.
Pushing the tab is the only way to move the plate. No rule sets a "closed" flag.

## Scripted demo and checks

`demo()` closes the left hand's fingers, brings the fingertips down beside the webcam, slides them along the body's top
to the tab, pushes the tab 17 mm, backs off the tab, lifts away and waits. 380 control steps (19 s) of the 2,400
allowed.

PyBullet results (`python check_demo.py`, `--sweep`, `--counter NAME`):

| Run | Score |
|---|---|
| `demo()`, nominal | 1.0 |
| `demo()`, friction x0.7 / x1.4, arm stiffness x0.7 / x1.4, both low, both high | 1.0 in all seven |
| `half_closed`: push the tab 10 mm and leave | 0.5 |
| `keep_holding`: close fully, fingers stay against the tab | 0.75 |
| `tip_off`: push the top of the webcam's body backward until it tips off the monitor's edge | 0.0 |

These are simulation results in a stand-in, not real-robot trials and not Isaac Sim runs.

## Running it

```bash
python -m pytest -q                 # offline tests
python check_demo.py                # PyBullet check of the demo (needs pybullet)
bash run.sh                         # Isaac Sim, scripted demo
bash run.sh dream                   # Isaac Sim, Dream-1
```

## Reset

1. Seat the webcam's clip on the monitor's top edge at its marked position, lens toward the robot.
2. Slide the shutter fully open to its stop and check the tab is at its open mark.
3. Inspect the shutter's stops and the clip.

## What is approximated

- All part sizes except the webcam's overall width and mass are estimates.
- Rigid boxes; the lens ring is visual only.
- The slider's real friction and any detent are replaced by viscous damping, so the force needed is lower than on a
  real unit at slow speeds.
- The cable is omitted.
- The monitor is fixed to the table.

## Duplicate check

A keyword search of the public repository's pull requests on 13 November 2026 for "webcam" and "shutter" returned no
results. This is a keyword search only, not a review of the task board.
