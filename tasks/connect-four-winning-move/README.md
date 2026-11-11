# Drop the winning disc in Connect Four

A Connect Four game is half played in an upright, standard-size frame between the arms. The robot plays red, and it
is red's turn. The win is a diagonal: red has column 7 row 1, column 6 row 2 and column 4 row 4, so a red disc
landing in column 5 row 3 (on top of column 5's two discs) completes it. Two distractors: red's three-in-a-row along
the bottom is already blocked, and blue threatens to complete its own bottom row in column 3, so blocking there is
tempting but does not win. A spare blue disc stands on the table too; playing it is wrong.

The robot has to lift the spare red disc out of its holder and drop it into a 3.6 cm x 1.1 cm slot 21 cm above the
table.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: frame, discs, `success()`, `score()` and the scripted `demo()` |
| `shapekit.py` | Discs standing on their edge (cylinders along the x axis) for yam_arena task files |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_connect_four.py` | Offline checks: the position is legal with exactly one win and one threat, the scorer's cases, fits, the demo's reach |

`shapekit.py` hooks `yam_arena.scene` without modifying yam_arena, so the task runs with the standard `yam_arena`
embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/connect-four-winning-move_demo_all.mp4 and the setup screenshot
bash run.sh dream    # Dream-1: ~/connect-four-winning-move_dream_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "You are playing red in Connect Four and it is your turn. Drop the spare red disc into the column that wins the game." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/c4-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The left arm, its gripper tilted
50 degrees forward as a hand would be, pinches the red disc by its rim, lifts it out of its holder, carries it over
column 5, lowers it until its edge is in the slot and lets go. A straight-down grip cannot reach 21 cm up, and
yam_arena's scripted waypoints only describe straight-down grips, so `task.py` adds a tilted one (see `PITCHED`).
About 11 s (210 steps).

## Scoring

From the end state (`SCORE_OVER = "end"`):

- 0 if the blue disc is in the frame.
- 1.0 (success, held 1 s): the red disc resting in column 5, row 3.
- 0.25: the red disc in any other column (a legal move that doesn't win).
- 0 otherwise (not in the frame).

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_connect_four.py -q
```

## Real-world build

A standard retail Connect Four with its discs (check it against the dimensions in `task.json`; any set close to them
works), taped to the table. Glue the twelve placed discs in place so a reset only has to shake out the dropped disc.
The two holders are two small blocks each, 4 x 20 x 8 mm, glued 1 mm off the disc's faces.
