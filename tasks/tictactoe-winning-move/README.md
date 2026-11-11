# Make the winning move in tic-tac-toe

A tic-tac-toe game is half played on a board between the arms, with 4 cm toy blocks as pieces. The robot plays red,
and it is red's turn. Red has two in the middle column with the bottom cell open, so putting the spare red block
there wins. Blue also has two in the right column, so the defensive block in the far right cell is tempting but
wrong. A spare blue block is on the table as well; playing it is wrong too.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: board, blocks, `success()`, `score()` and the scripted `demo()` |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_tictactoe.py` | Offline checks: the position has exactly one win and one threat, the scorer's cases, the demo's reach |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo (scores 1.0): ~/tictactoe-winning-move_demo_top.mp4 and the setup screenshot
bash run.sh dream    # Dream-1: ~/tictactoe-winning-move_dream_top.mp4
```

Or directly, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "You are playing red in tic-tac-toe and it is your turn. Make the winning move: put the spare red block in the cell that wins the game." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/ttt-dream
```

The scripted demo is the same command with `--policy yam_scripted --max-steps 600`: the left arm picks the spare
red block at its centre and sets it in the winning cell. About 10 s (200 steps).

## Scoring

From the end state (`SCORE_OVER = "end"`):

- 0 if any block already on the board moved more than 1.5 cm or tipped off its face, or the blue block is on the
  board.
- 1.0 (success, held 1 s): the red block flat on the board (any face down), centre within 2.5 cm of the winning cell's
  centre.
- 0.25: the red block flat in any other empty cell (a legal move that doesn't win).
- 0 otherwise.

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_tictactoe.py -q
```

## Real-world build

Nothing to print: a 24 cm square of 6 mm white acrylic or foam board ruled into 8 cm cells with black tape, taped to
the table, and six 4 cm cubic toy blocks, four painted red and two blue.
