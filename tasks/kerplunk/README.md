# KerPlunk: remove one rod without dropping a marble

A KerPlunk-style tower between the arms: a clear tube on a wooden base, four coloured rods crossing it about 10 cm
up, and five glass marbles resting on them. The robot must pull out exactly one rod without any marble falling into
the chamber below.

Which rod is safe follows from the geometry, not from a rule in the scorer:

- **orange, green, yellow** are ledges, each running 3 cm from the tube's axis. Their marbles sit between the rod
  and the wall, each marble's centre outside its rod (toward the wall), so the rod and the wall hold it. Nothing else
  is under them: pull the rod and they drop.
- **blue** runs beside the orange ledge, level with the middles of the orange ledge's two marbles, touching them. It
  looks as if it holds them in, but gravity presses those marbles outward onto the orange rod and the wall, not
  against the blue one. It carries no load.

`check_physics.py` builds the tower in PyBullet and pulls each rod straight out from a fresh start: orange drops 2
marbles, blue none, green 2, yellow 1. `python check_physics.py sweep` repeats this at friction 0.2 to 0.7, at 3, 5 and
15 cm/s: the result is the same every time. `python check_physics.py demo` replays the demo's
grip on the cap, lift and pull with the arm up to 2 mm off: the blue rod comes fully out and no marble falls.

The holes are loose (1 mm clear either side, open above the rod, as a loose round hole is to a rod lying in it), and
`test_kerplunk.py` checks that no part of one rod's hole comes near another rod: the orange and blue rods run 2.8 mm
apart, and a rod caught by its neighbour's hole jams when pulled.

```
                 far wall
           .----------------.
         /       o  o        \      o: marbles (this pair on orange, touching blue)
        |   ====orange===     |
        |   ====blue=====     |     orange and blue run left-right (blue just inside orange)
        | o |            |    |
        | o |            |  o |     green (left) and yellow (right) run toward the robot,
        |   |            |    |     each with marbles between it and the wall
        | green      yellow   |
         \                   /
           '----------------'
                 robot
```

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: base, tube, rods, marbles, the fall watcher, `success()`, `score()` and the scripted `demo()` |
| `tower.py` | The tower's geometry: rods, marbles, the wall with its holes, and how far a rod is out |
| `task_geometry.py` | The demo's grip on the blue rod (where, how far it lifts and pulls), shared with the physics check |
| `shapekit.py` | Rods (a cylinder with its cap), marbles and the see-through tube for yam_arena task files |
| `stepkit.py` | Runs the fall watcher after every control step |
| `check_physics.py` | PyBullet check of all four removals (`pip install pybullet`) |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_kerplunk.py` | Offline checks: what each marble touches, rod clearances, the scorer, the demo's reach |

The kits hook yam_arena without modifying it, so the task runs with the standard `yam_arena` embodiment and
`task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/kerplunk_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1, 2,400 steps: ~/kerplunk_dream_<time>_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras; `<time>` is when the run started. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Remove exactly one stick from the tower without allowing any marbles to fall into the bottom chamber." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/kerplunk-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The right arm takes the blue rod
by its cap on the robot's right (the rods stick out unevenly, so it is clear of the orange rod beside it), holding the
cap's flat faces, and lifts that end 2 mm off its hole's sill (pressed down into its holes, a rod drags; lifted, it
slides free). It slides the rod straight out over 5 s until its far end is 3 cm past the wall, carries it clear and
lays it on the table. About 13 s (270 steps).

Each rod has a light 14 mm square cap over its long end: its colour marker, and something a two-finger gripper holds
reliably (these grippers' tapered fingers do not pinch a bare 6 mm round rod well). The caps are on the robot's side
for the green and yellow rods, on the robot's right for blue and its left for orange; each rod's weight stays between
its two holes.

## Scoring

Watched every step, scored from the end state (`SCORE_OVER = "end"`):

- 0 if any marble's centre ever drops 20 mm below the lowest rods, if a marble is handled (lifted 15 mm above where it
  started, or taken out over the rim), or if more than one rod is half out.
- 1.0 (success, held 1 s): one rod slid fully out of the tube (its last end past the tube's outside).
- 0.5: one rod at least half out (its last end has gone half the way to the tube's outside).
- 0 otherwise.

## Runtime and seeds

About 2 to 3 minutes per Dream-1 trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_kerplunk.py -q
python check_physics.py
python check_physics.py sweep
python check_physics.py demo
```

## Real-world build

A clear acrylic tube, 100 mm outside, 94 mm inside, 160 mm long, glued to a 140 x 140 x 10 mm plywood base. Drill
8 mm holes for the four rods where `tower.py` puts them (`crossing_points`): the orange and blue rods along y, the
green and yellow along x, at 110, 117.4, 103.5 and 103.5 mm above the table. Cut 16 cm rods from 6 mm birch dowel,
paint them orange, blue, green and yellow, glue a 14 mm hollow 3D-printed square cap of the same colour over each
one's long end (`CAP_END` in `tower.py`), push them through so they stick out as in `SHIFT`, and set the five 16 mm
marbles against the wall on their ledges. To reset, lift the tube off the base to tip out fallen marbles.
