# Pick-up sticks: take out the trapped red stick

Five square wooden sticks lie in a small pile between the arms. The red stick lies between the yellow and black ones,
2.2 cm from each, with the green stick just past its far end. The blue stick lies across the middle of the yellow, red
and black sticks, resting on all three, so it traps the red stick. The robot must put the red stick on the mat to its
right without moving the green, yellow or black sticks more than 3 mm or turning them more than 3 degrees, at any time.

```
            green (across the far ends, 1.2 cm past them)
        ===========================
            |      |      |
         yellow   red   black        (along x, 3 cm apart)
        ----+------+------+----      blue: across their middles, resting on all three
            |      |      |
            |      |      |
                 robot                (mat: 20 cm to the robot's right)
```

The dependency is real. `check_physics.py` replays both orders with stand-in fingers in PyBullet:

- **blue first, then red** (the demo): the protected sticks move under 0.1 mm;
- **red first** (lifted straight up with the blue one on it): the blue stick slides off and knocks a protected stick
  97 mm.

Any order of moves that ends with the red stick on the mat and nothing protected disturbed scores full marks.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: sticks, mat, the disturbance watcher, `success()`, `score()` and the scripted `demo()` |
| `shapekit.py` | Bodies made of boxes with their own friction (the sticks) for yam_arena task files |
| `stepkit.py` | Runs the disturbance watcher after every control step |
| `check_physics.py`, `_standins.py` | PyBullet check of the two orders (`pip install pybullet`) |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_pick_up_sticks.py` | Offline checks: the layout, finger clearances, the scorer, the demo's reach |

The kits hook yam_arena without modifying it, so the task runs with the standard `yam_arena` embodiment and
`task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/pick-up-sticks_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1, 3,600 steps: ~/pick-up-sticks_dream_<time>_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras; `<time>` is when the run started. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Remove the red stick from the pile. You may remove the blue stick first, but do not disturb the green, yellow, or black sticks." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/pick-up-sticks-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The right arm lifts the blue
stick straight up by its middle (fingertips 2 mm above the red stick), turns it and lays it on the mat, then lifts the
red stick straight up by its middle (fingers opened 12 mm a side, 6 mm clear of the yellow and black sticks) and lays
it on the mat beside the blue one. About 24 s (480 steps).

## Scoring

Watched every step, scored from the end state (`SCORE_OVER = "end"`):

- 0 if the green, yellow or black stick ever moves more than 3 mm or turns more than 3 degrees from where it first lay.
- 1.0 (success, held 1 s): the red stick lying on the mat.
- 0.5: the blue stick out of the pile (on the mat, or its centre more than 10 cm from the pile's), the red stick not on
  the mat.
- 0 otherwise.

## Runtime and seeds

About 3 to 4 minutes per Dream-1 trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_pick_up_sticks.py -q
python check_physics.py
```

## Real-world build

Five 14 cm lengths of 8 mm square beech dowel, painted red, blue, green, yellow and black. Print the layout at 1:1
(positions in `START` in `task.py`), tape it to the table, and lay the sticks on it in order: red, yellow, black,
green, then blue across the middle of the three. A 16 x 16 cm card is the mat. To score in the real world, film from
above and compare the protected sticks' positions with the template.
