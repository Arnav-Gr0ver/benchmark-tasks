# Replace the bit in a cordless drill

A compact cordless drill lies loose on its side between the arms, with a 4 mm bit tight in its keyless chuck. A 6 mm
bit stands in a rack beside it, next to an empty tray. The robot must take the old bit out, put the 6 mm bit in, and
tighten the chuck so the new bit stays put under an axial pull. Then it lets go, leaving the old bit in the tray and
the drill on the table.

Two things make this hard:

- **The chuck only moves through its sleeve.** Turning the sleeve drives the three jaws in or out. Nothing else moves
  them, and pushing a bit in does not grab it. The jaws must open further for the 6 mm bit than they were for the
  4 mm one.
- **The drill is loose.** Turning the sleeve twists the whole drill unless something holds it. In practice that
  means one hand steadies the drill while the other turns the sleeve. The scorer does not require any particular
  hand or order.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: drill, bits, rack and tray; the chuck's thread, release check and retention test (run every step); `success()`, `score()` and the scripted `demo()` |
| `chuck.py` | Geometry of the drill, chuck and bits, the thread logic, the insertion and removal checks, and the pull profile |
| `shapekit.py` | Shapes for yam_arena task files: `Compound`, plus `Mechanism` (an articulation with hinged and sliding links, joint drives and applied forces) |
| `stepkit.py` | Runs the watcher after every control step |
| `check_physics.py` | PyBullet check of the chuck and the retention test (`pip install pybullet`) |
| `run.sh` | Runs a trial and writes its videos to your home folder (see below) |
| `test_drill_bit_swap.py` | Offline checks: chuck geometry, the thread, every score tier, the release check, demo reach |

The kits hook into yam_arena without changing it, so the task runs with the standard `yam_arena` embodiment and the
`task_score` scorer.

## How the chuck works in simulation

The drill is one articulation:

- **Body:** loose on the table, 1.0 kg.
- **Sleeve:** a 36 mm cylinder on a hinge about the chuck axis. It is damped, so it stays where it is left.
- **Jaws:** three steel jaws on radial slides. Each one is driven by a stiff spring (5e5 N/m) toward a commanded
  radius.

Inside a real chuck, a thread turns sleeve rotation into jaw travel, and the jaws cannot drive it backwards. `chuck.Thread`
plays that role. Every control step it reads how far the sleeve has turned and moves the jaws' commanded radius by
5 mm per turn (10 mm of opening per turn).

- **Opening** stops at the end stop: a 10.4 mm opening.
- **Closing** stops 0.3 mm inside the outermost jaw once the jaws touch the bit. Each jaw then pushes with at most
  150 N, roughly the limit of a hand's twist on the sleeve. Turning further makes it no tighter.
- **A jaw knocked outward** never loosens the thread.

The bit is held only by friction between the jaws and its smooth shank (hardened steel, 0.3). Three fixed lands
between the jaws form a bore, so a loose bit cannot drop out sideways. With the jaws open, a bit just slides in and
out.

The retention test is a real applied force. The scorer pulls the bit along the chuck axis and applies the equal and
opposite force to the drill at its sleeve, which is the evaluator holding the drill body. Profile:

| Phase | Force | Duration |
|---|---|---|
| Ramp up | 0 to 50 N | 0.5 s |
| Hold | 50 N | 1 s |
| Unload | 50 N to 0 | 0.5 s |
| Watch | 0 | 1 s |

The bit must slip 1 mm at most, and still be seated at the end: within 3 degrees and 0.6 mm of the axis. If this
build of yam_arena cannot apply a force to a body, the scorer instead checks the clamp. Every jaw must push at least
1.5 x 50 N / (3 x 0.3) = 83 N, and the log says which method was used.

`check_physics.py` builds the same chuck in PyBullet and runs it with the same `chuck.Thread`, at 20 Hz:

| Run | Result |
|---|---|
| The 4 mm bit as prepared | Passes the pull (0.00 mm slip) |
| One opening stroke | Loosens it; the bit draws out, and the unheld drill does not move |
| Three more opening strokes | Open the jaws fully |
| 6 mm bit with four closing strokes | Each jaw pushes 146 N; passes (0.02 mm slip) |
| 6 mm bit with two closing strokes | Pulls out |
| 6 mm bit with two strokes and 32 degrees more (jaws just touching) | Pulls out |

So tightness is judged by whether the bit holds, not by counting turns.

## Scripted demo

1. The left hand holds the drill's handle. The right hand closes on the sleeve and turns it 65 degrees (one opening
   stroke).
2. The left hand draws the old bit out by its fluted end and lays it in the tray, then holds the handle again.
3. The right hand opens the jaws fully (three more strokes) and keeps hold of the sleeve.
4. The left hand takes the 6 mm bit out of the rack along its lean, levels it, and pushes its shank 19 mm into the
   chuck.
5. The right hand makes two closing strokes while the left hand holds the bit. The left hand lets go, and the right
   hand tightens with two more strokes.
6. Both hands back off. The scorer waits 2 s for everything to be still, then runs the pull.

To turn the sleeve, the right hand rolls about the chuck axis. yam_arena's scripted waypoints only describe
straight-down grips, so `task.py` registers those orientations with `yam_arena.demo.grip_rotation` through a tagged
yaw. The demo takes about 77 s, including the final wait, and every waypoint is reachable (worst IK error under 1 mm).

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough:

```bash
bash run.sh          # scripted demo: ~/drill-bit-swap_demo_<time>_all.mp4 and a setup screenshot
bash run.sh dream    # Dream-1, 3,600 steps: ~/drill-bit-swap_dream_<time>_all.mp4
bash run.sh pretest  # the retention test on the 4 mm bit as prepared, then the demo
```

`_all.mp4` shows the top camera above both wrist cameras. `<time>` is when the run started. Or run directly, from
this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Replace the installed drill bit with the larger bit from the rack. Leave the old bit in the tray and the drill on the table with the new bit securely held in its chuck." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/drill-bit-swap-dream
```

The scripted demo is the same command with `--policy yam_scripted`.

## Scoring

The scorer checks the end state; the release watch and the pull run during the episode. It awards the highest tier
reached:

| Score | Outcome |
|---|---|
| 0 | The old bit is still between the jaws, or the drill or a bit fell off the table |
| 0.25 | The old bit is fully out: every point of it 2 mm in front of the jaws, or 2 cm off the axis |
| 0.5 | Also, the 6 mm bit is inserted: its shank end 14 to 22.5 mm past the jaws' front, within 15 degrees and 3 mm of the axis. It may still be held. |
| 0.75 | Also, every finger is 30 mm or more from the drill and the new bit, both are still for 2 s (within 2 mm and 2 degrees), and the bit passes the retention test |
| 1.0 | Also, the old bit is let go and lying entirely inside the tray, and the drill is lying on the table |

More rules:

- A failed retention test caps the score at 0.5.
- Touching the drill or bit while the test runs makes that test invalid. It restarts once they are let go again.
- Loosening the chuck after a pass needs a new test.

The insertion window comes from the chuck: the jaws are 22 mm long, and a bit bottoms out on the sleeve's face at
22 mm. Both bits' smooth shanks (32 and 36 mm) are longer than that, so the jaws only ever grip shank.

## Runtime and seeds

About 4 to 5 minutes per 3,600-step trial on an L40S, plus about 1 minute for Isaac Sim to start. Fixed layout
(`-E object_jitter=0`); no seeds.

## Checks

```bash
python -m pytest test_drill_bit_swap.py -q
python check_physics.py
```

## Real-world build

- **Drill:** a compact drill/driver with a 10 mm (3/8 in) single-sleeve keyless chuck, such as the DeWalt DCD701.
  Battery removed, trigger locked.
- **Bits:** one 4 mm and one 6 mm DIN 338 jobber bit (75 and 93 mm long).
- **Rack:** a wooden block with an 8 mm square socket, 26 mm deep, leaning 45 degrees.
- **Tray:** 120 x 60 mm.

Use the layout in `task.json`. The drill dimensions in `chuck.py` are nominal: measure the actual tool and update
them before collecting physical runs.

To reset:

1. Put the 4 mm bit 19 mm into the open chuck.
2. Close the sleeve by hand until the jaws stop, then give it a firm final twist.
3. Confirm it holds under a 50 N pull. Use a spring scale on the bit, with the drill body held in a bench vise by
   its handle.
4. Stand the 6 mm bit in the rack and empty the tray.

The 50 N load is a benchmark criterion, not a manufacturer rating.

## Related tasks

- PR #400 chucks a bar in a fixed lathe with a key.
- PRs #1288 and #1301 fit a hex screwdriver bit, which has no clamp to operate.
- PR #1827 swaps a drill's battery.

This task is a complete exchange through a hand-operated, adjustable chuck on a loose tool. Robots have changed drill
bits before (Boston Dynamics' Atlas has shown related chuck handling), so this is not claimed as a first.
