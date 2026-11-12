# Fix a matchstick equation by moving one stick

Eighteen wooden sticks on a black mat between the arms spell a wrong equation in seven-segment digits:

```
 _         _         _
|_|  ___  |_   ___    |
 _|       |_|  ___    |
```

9 - 6 = 7 (read from the robot's side). The robot must move exactly one stick to make it true. Only one move works:
take the top of the 7 (leaving a 1) and lay it in the 6's empty upper right (making an 8): 9 - 8 = 1. The move changes
two digits at once, crosses the equals sign, and the stick has to be turned a quarter turn on the way. Moving the
minus's stick, or reshaping a single digit, never gives a true equation.

`test_matchstick.py` checks this by trying every stick in every empty place, reading digits leniently (a 6 or 9 without
its tail and a 7 with a hook count too): exactly one move gives a true equation.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: mat, sticks, the equation reader, `success()`, `score()` and the scripted `demo()` |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_matchstick.py` | Offline checks: the start is wrong, exactly one move fixes it, the scorer's cases, spacing, the demo's reach |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/matchstick-math_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1: ~/matchstick-math_dream_<time>_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras; `<time>` is when the run started. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Make the equation mathematically correct by moving exactly one matchstick." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/matchstick-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The right arm picks up the 7's top
stick, turns it a quarter turn while carrying it over the 6, and lays it in the 6's upper right. About 10 s (200 steps).

## Scoring

From the end state (`SCORE_OVER = "end"`). A stick lies in a segment's place if its centre is within 1.2 cm of it, its
length within 20 degrees of the segment's direction, and it lies on the mat.

- 1.0 (success, held 1 s): every stick in a place, a true equation, exactly one stick moved.
- 0.25: only the right stick (the 7's top) moved, but it is not in the 6's upper right.
- 0 otherwise: another stick moved, more than one moved, or nothing moved.

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_matchstick.py -q
```

## Real-world build

18 jumbo craft sticks or square dowel cut to 40 x 8 x 8 mm, on a black felt mat taped to the table. Print the layout
(every stick's place is in `SLOTS` in `task.py`) at 1:1 and mark the places on the mat, or lay the sticks on the print
under a clear sheet.
