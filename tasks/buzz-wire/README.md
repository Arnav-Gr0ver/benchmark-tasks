# Buzz wire: carry the ring along a bent wire without touching it

A buzz-wire game between the arms. A 3 mm steel wire runs away from the robot, 6 cm above a wooden base: it turns a
quarter turn to the robot's left while climbing 3 cm, swings a half turn back to the right while dropping, turns a
quarter turn to the left again, and ends under a red sleeve. A steel ring (24 mm inside) with a handle sits around the
wire at the green sleeve. The robot must carry the ring to the red sleeve without the ring touching the bare wire.

Because the wire's direction swings through 180 degrees, the ring has to be turned on the way: `test_buzz_wire.py`
checks that a ring carried along the wire centred but never turned, held any way round, touches the wire somewhere.
It also has to follow the wire up and down (the crest is 3.2 cm above the ends).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: base, wire, ring, stand, the touch watcher, `success()`, `score()` and the scripted `demo()` |
| `track.py` | The wire's path (a cubic B-spline), the checkpoints, and the ring-to-wire distance the scorer uses |
| `shapekit.py` | The ring (one body of tilted boxes) and the wire (a fixed chain of capsules) for yam_arena task files |
| `stepkit.py` | Runs the touch watcher after every control step |
| `check_physics.py` | PyBullet check: the ring rests on its stand clear of the wire, and the wire traps it (`pip install pybullet`) |
| `wire_path.csv` | The wire's centre line every 5 mm (mm), for bending a real one |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_buzz_wire.py` | Offline checks: the path, that turning is required, the scorer, the demo's clearance and reach |

The kits hook yam_arena without modifying it, so the task runs with the standard `yam_arena` embodiment and
`task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/buzz-wire_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1, 3,600 steps: ~/buzz-wire_dream_<time>_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras; `<time>` is when the run started. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the metal ring from the green starting point to the red endpoint along the wire without allowing the ring to touch the wire." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/buzz-wire-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The right arm grips the handle from
above, raises the ring 5 mm out of its stand, and carries it along the wire in 8 mm steps to the red sleeve, pitching
it with the wire and turning it through half of each change in the wire's heading (a round ring only has to stay
within about 75 degrees of the wire's line; turning it the full 180 degrees would exceed the wrist's range). Offline,
its ring stays at least 9.8 mm clear of touching the wire (it touches at 3 mm). About 25 s (500 steps).

## Scoring

Watched every step, scored from what was reached (`SCORE_OVER = "end"`):

- 0.2, 0.4, 0.6, 0.8 for passing the checkpoints at 20, 40, 60, 80 % of the bare wire, in order;
- 1.0 (success, held 1 s) for reaching the red sleeve;
- credit counts only until the ring first touches the bare wire (its centre line within 3 mm of the wire's); credit
  earned before that is kept. Touching the green or red sleeve does not count.

The ring encircles the wire and cannot leave it, so the track cannot be bypassed.

## Runtime and seeds

About 3 to 4 minutes per Dream-1 trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_buzz_wire.py -q
python check_physics.py
```

## Real-world build

Bend 3 mm steel wire to `wire_path.csv` (print the x-y path 1:1 as a template; the heights are in the last column) and
fix its ends in two posts on a 10 mm plywood base. Slip a 7 mm heat-shrink sleeve over the first 3.5 cm (green) and
the last 3 cm (red). Make the ring from 3 mm steel wire, 24 mm inside, with a 70 mm handle of 10 mm square plastic.
Wire the ring and the bare wire to a 3 V buzzer or continuity tester. Make the stand (a pillar with a 5 mm groove, its
top 4.5 cm up) and the handle rest from wood at the positions in `task.py`.
