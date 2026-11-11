# Turn and place the last piece of a peg puzzle

A wooden peg puzzle (the children's kind, with a knob on every piece) sits in its tray between the arms with its
middle piece missing. The last piece lies on the table to the robot's left, turned a quarter turn from the way it
fits. Its two tabs and two notches fit the hole only one way round: in each of the other three turns a tab runs
8 mm into a neighbour (checked in `test_jigsaw.py`). The robot has to read the hole's shape, turn the piece to
match, and lower it in.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: tray, pieces (geometry in `piece_boxes`), `success()`, `score()` and the scripted `demo()` |
| `shapekit.py` | Bodies made of several boxes (the pieces) for yam_arena task files |
| `test_jigsaw.py` | Offline checks: the pieces clear each other, the last piece fits one way only, the scorer's cases, the demo's reach |

`shapekit.py` hooks `yam_arena.scene` without modifying yam_arena, so the task runs with the standard `yam_arena`
embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Finish the jigsaw puzzle: put the last piece into the empty space in the tray." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/jigsaw-dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted --max-steps 400`. The left arm grips the
knob, lifts the piece, turns it a quarter turn clockwise while carrying it over the hole, lowers it until its
underside is 1.5 mm above the tray floor, and lets go, opening only partly so its fingers miss the neighbouring
knobs. About 12 s (230 steps).

## Scoring

From the end state (`SCORE_OVER = "end"`):

- 1.0 (success, held 1 s): seated: flat on the floor (within 1.5 mm), centre within 3 mm of the hole's, turned within
  6 degrees of the way it fits.
- 0.5: turned the right way (within 15 degrees), centre within 1.5 cm of the hole's, not seated.
- 0.25: centre within 1.5 cm of the hole's, turned the wrong way.
- 0 otherwise.

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds. The tabs' layout is fixed by `random.Random(7)` in `task.py`.

## Checks

```bash
python -m pytest test_jigsaw.py -q
```

## Real-world build

Cut the nine pieces and the tray from 8 mm and 4 mm plywood with the outlines in `task.py` (each piece is a union of
rectangles), glue a 12 mm square pine knob 14 mm tall on each piece's centre, glue the eight placed pieces to the
tray floor, and paint the last piece orange.
