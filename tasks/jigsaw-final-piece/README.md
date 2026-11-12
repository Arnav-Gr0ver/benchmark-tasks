# Choose and insert the missing jigsaw piece

A 4 x 4 jigsaw of a printed landscape lies assembled on a dark base board between the arms, with one piece missing
near the centre. Four loose pieces lie beside the board, two on each side, each turned a different way. Only one of
them is the missing piece:

| Piece | Shape (tabs and sockets) | Picture |
|---|---|---|
| `piece_a` | wrong: tabs away from the robot and to its left | right |
| `piece_b` | right | wrong: the picture of the piece to the hole's right |
| `piece_c` | right (the missing piece) | right |
| `piece_d` | wrong: three tabs | wrong: the right picture upside down |

The robot has to tell them apart by both cues: the picture near the hole's edge must continue its neighbours'
(the river, the meadow and the hill line run through the hole), and the tabs and sockets must interlock with the
neighbours'. The hole's neighbours toward and away from the robot reach into it with tabs; its left and right
neighbours have sockets. Tabs sit slightly off their edges' middles, so the missing piece fits only one way round:
from where it lies it must be turned 125 degrees, then lowered straight down into the hole (its round-headed tabs
lock it in sideways).

Checked offline (`test_jigsaw.py`): `piece_c` and `piece_b` fit the hole at 1 mm clearance only unturned; every other
piece and turn overlaps a neighbour by at least 20 mm²; only `piece_a` and `piece_c` unturned show the picture the
neighbours continue, every other piece and turn differs along at least a quarter of the hole's edge.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: board, pieces, `success()`, `score()` and the scripted `demo()` |
| `puzzlekit.py` | Jigsaw pieces for yam_arena task files: convex-prism collision, a printed picture, a peg |
| `pieces.json` | Every piece's outline, convex parts and picture (made by `make_pieces.py`) |
| `make_pieces.py` | Draws the pieces and the picture; writes `pieces.json` and `print_and_cut.svg` |
| `print_and_cut.svg` | The real build: the picture with every cut line, 1 unit = 1 mm |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_jigsaw.py` | Offline checks: which pieces fit and which pictures continue the board, the scorer, the demo's reach |

`puzzlekit.py` hooks `yam_arena.scene` without modifying yam_arena, so the task runs with the standard `yam_arena`
embodiment and `task_score` scorer. Running needs nothing beyond yam_arena; regenerating `pieces.json` needs
`pip install shapely mapbox-earcut`.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/jigsaw-final-piece_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1: ~/jigsaw-final-piece_dream_<time>_all.mp4
```

`_all.mp4` shows the top camera above both wrist cameras; `<time>` is when the run started. Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Complete the jigsaw puzzle by choosing and inserting the correct missing piece." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/jigsaw-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`. The right arm takes `piece_c`
by its peg, lifts it, turns it 125 degrees while carrying it over the hole, lowers it until it is 2 mm above the
base, lets go, then presses the peg down once with closed fingers. About 16 s (320 steps).

## Scoring

From the end state (`SCORE_OVER = "end"`):

- 0 if a wrong piece lies in or on the hole (centre within 1.5 cm of the hole's, underside within 1.8 cm of the base).
- 1.0 (success, held 1 s): `piece_c` seated: centre within 2.5 mm of the hole's, turned within 5 degrees of the way
  it fits, underside within 1.5 mm of the base, tilted 3 degrees at most.
- 0.5: `piece_c` over the hole but not seated (centre within 1.5 cm of the hole's, underside within 1.8 cm of the base).
- 0 otherwise.

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds. The tab layout comes from a fixed draw (`SEED = 11` in `make_pieces.py`).

## Checks

```bash
python -m pytest test_jigsaw.py -q
```

## Real-world build

Print `print_and_cut.svg` at 100% on sticker paper, stick it on 6 mm birch plywood and laser-cut the red lines. Each
piece is cut on its own outline, so neighbours end up 1 mm apart; the 1 mm strips between them are scrap. Lightly
sand the bottom edge of the four loose pieces (about 0.8 mm). Glue the 15 board pieces onto a 200 x 200 x 4 mm
dark-painted MDF base in their places; the board piece marked as `piece_c` is the missing piece. Cut `piece_a`,
`piece_b` and `piece_d` from the second column of the sheet. Glue a 12 x 12 x 14 mm pine peg at the centre of each
loose piece (its cell's centre: the middle of the 44 mm square the tabs grow from). Tape the base to the table and
mark the four pieces' starting places and turns.
