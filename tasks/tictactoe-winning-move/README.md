# Make the winning move in tic-tac-toe

A tic-tac-toe game is half played on a board between the arms. The robot plays red, and it is red's turn. Red has two
in the middle column with the bottom cell open, so placing the spare red counter there wins. Blue also has two in the
right column, so the defensive block in the far right cell is tempting but wrong. A spare blue counter is on the table
as well; playing it is wrong too.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: board, counters, `success()`, `score()` and the scripted `demo()` |
| `test_tictactoe.py` | Offline checks: the position has exactly one win and one threat, the scorer's cases, the demo's reach |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "You are playing red in tic-tac-toe and it is your turn. Make the winning move: put the spare red counter in the cell that wins the game." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/ttt-dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted --max-steps 400`. The left arm picks the
spare red counter and sets it in the winning cell, opening its fingers only partly so they stay clear of the
neighbouring cells. About 10 s (200 steps).

## Scoring

From the end state (`SCORE_OVER = "end"`):

- 0 if any counter already on the board moved more than 1.5 cm or tipped past 15 degrees, or the blue counter is on
  the board.
- 1.0 (success, held 1 s): the red counter flat on the board, centre within 2.5 cm of the winning cell's centre.
- 0.25: the red counter flat in any other empty cell (a legal move that doesn't win).
- 0 otherwise.

## Runtime and seeds

About 1 minute per trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_tictactoe.py -q
```

## Real-world build

A 24 cm square of 6 mm white acrylic or foam board ruled into 8 cm cells with black tape, taped to the table; six
plastic discs 4.4 cm across and 1.6 cm thick (stacks of two draughts checkers glued together work), four red and two
blue.
