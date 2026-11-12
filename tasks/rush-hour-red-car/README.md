# Rush Hour: free the red car

A Rush Hour sliding-block puzzle between the arms, two moves from solved. The red car faces the exit, a gap in the
right-hand wall; a yellow car stands across its row. The yellow car can't go toward the robot (a green car blocks
that way), so it has to slide two cells away from the robot; then the red car slides straight out of the exit onto a
pink apron. Vehicles only slide along their lanes, and nothing may be lifted over anything.

```
far wall
 B B B . . T
 . P . . . T
 . P R R Y .  <- exit
 O . . . Y .
 O . . . G G
 O . . . . .
robot's side
```

R red car, Y yellow car (the blocker), G green car (stops the yellow car going toward the robot), P purple car,
T teal car, B blue truck, O orange truck. The map is seen from above with the robot at the bottom; the exit is
the gap at the right end of the red car's row.

`test_rush_hour.py` solves the puzzle by breadth-first search under Rush Hour's rules: the only shortest solution is
the yellow car two cells away from the robot, then the red car out.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: board, vehicles, the rule watcher, `success()`, `score()` and the scripted `demo()` |
| `shapekit.py` | Bodies made of several boxes (the vehicles with their pegs, the walled board) with their own friction |
| `stepkit.py` | Runs a check after every control step (here: was anything lifted?) |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_rush_hour.py` | Offline checks: the solver, the layout, the scorer's cases, finger clearances, the demo's reach |

Both kits hook yam_arena without modifying it, so the task runs with the standard `yam_arena` embodiment and
`task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/rush-hour-red-car_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1: ~/rush-hour-red-car_dream_<time>_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras; `<time>` is when the run started. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the blocking vehicle, then slide the red car out of the exit." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/rush-hour-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The right arm takes the yellow car
by its peg, with the fingers closing along the car's length so they push on the peg's faces, slides it two cells away
from the robot 3 mm clear of the board, and lets go; then it does the same with the red car, out through the exit.
About 20 s (400 steps).

## Scoring

Watched through the run, scored from the end state (`SCORE_OVER = "end"`):

- 0 if any vehicle was lifted more than 8 mm off the board at any time, or ends out of its lane (more than 8 mm off
  its lane's centre line, or turned more than 15 degrees).
- 1.0 (success, held 1 s): the red car out, its back end past the right wall's outer face.
- 0.5: the yellow car clear of the red car's row, the red car not out.
- 0 otherwise.

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_rush_hour.py -q
```

## Real-world build

3D-print or cut the board (a 216 x 216 x 6 mm base with 6 mm walls 10 mm tall and a 34 mm gap in the right wall) and
the vehicles (28 x 64 x 16 mm cars, 28 x 98 x 16 mm trucks, each with a 12 x 12 x 14 mm peg on the middle of the
roof), in smooth PLA; paint them in the colours above. Add a 44 x 100 mm apron level with the base outside the exit.
Tape the board to the table and mark every vehicle's cells.
