"""Closed-loop scripted demo for the tilt maze, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

After every press it reads where the ball came to rest (the simulator's state, which a scripted demo may use),
plans the remaining presses from there with plan.py, and makes the next one: the arm that reaches that side of the
rim goes above it, presses the lid down until the board rests on its stand (7 degrees), holds until the ball has
stopped, and lets the board level. Writes one MP4 per camera (H.264), setup.png (the first top-camera frame) and result.json.

`--policy yam_scripted` plays task.py's demo() instead: the same presses, open loop, from the start.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
MIN_HOLD_S, MAX_HOLD_S = 1.0, 7.0
REST_SPEED, REST_S = 0.005, 0.5   # the ball has stopped: under 5 mm/s for half a second
MAX_PRESSES = 20


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def move(self, pos, duration: float, grip: float | None = None):
        """Generator, one yield per control step: the TCP to ``pos`` pointing straight down; grip 1 open, 0 closed."""
        from yam_arena import ik
        from yam_arena.demo import grip_rotation

        rot = grip_rotation(0.0)
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            target = p0 + s * (np.asarray(pos, float) - p0)
            q, pe, _ = ik.solve(target, ik.matrix_to_quat_xyzw(ik.slerp_matrix(r0, rot, s)), self.q,
                                base_pos=self.base, iters=60)
            self.q, self.pos, self.rot, self.grip = q, target, ik.slerp_matrix(r0, rot, s), g0 + s * (g1 - g0)
            self.worst_mm = max(self.worst_mm, pe * 1000)
            self.cmd = np.r_[q, self.grip]
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def script(task, ctx, log):
    """The demo as a generator: one yield per control step."""
    import maze
    import plan

    T = task
    arms = {"left": ctx.left, "right": ctx.right}
    yield from _both(ctx.left.move(T.PARK["left"], 1.5, grip=0.0), ctx.right.move(T.PARK["right"], 1.5, grip=0.0))
    holds, expected, active = [], None, None
    for press in range(MAX_PRESSES):
        local = T.ball_on_board(ctx.states())
        xy = (float(local[0]), float(local[1]))
        if maze.ring_at(*xy) == 0:
            log(f"ball in the centre after {press} presses")
            break
        if not holds or expected is None or math.dist(xy, expected) > 0.015:
            holds = plan.plan(xy, tilt_deg=math.degrees(T.DEMO_TILT)) or []
            log(f"plan from r={1000 * math.hypot(*xy):.0f} mm az={math.degrees(math.atan2(xy[1], xy[0])) % 360:.0f}:"
                f" {holds}")
            if not holds:
                log("no plan found; stopping")
                break
        az = holds.pop(0)
        model = plan.BallModel()
        model.reset(xy)
        model.hold(az, math.degrees(T.DEMO_TILT))
        expected = (model.x, model.y)
        arm_name = T.arm_for(az)
        arm = arms[arm_name]
        if active is not None and active != arm_name:
            yield from arms[active].move(T.PARK[active], 1.2)
        active = arm_name
        yield from arm.move(T.press_point(az, 0.0, T.HOVER), 1.5)
        yield from arm.move(T.press_point(az, 0.0, 0.002), 0.8)
        yield from arm.move(T.press_point(az, T.DEMO_TILT, -T.PRESS_DEEPER), 1.0)
        held, still = 0.0, 0.0
        while held < MAX_HOLD_S:
            yield
            held += 1.0 / CONTROL_HZ
            speed = float(np.linalg.norm(ctx.states()["ball"].lin_vel))
            still = still + 1.0 / CONTROL_HZ if speed < REST_SPEED else 0.0
            if held >= MIN_HOLD_S and still >= REST_S:
                break
        yield from arm.move(T.press_point(az, 0.0, T.HOVER), 0.8)
        local = T.ball_on_board(ctx.states())
        log(f"press {press + 1} at {az} deg by the {arm_name} arm: ball in ring {maze.ring_at(local[0], local[1])}")
    if active is not None:
        yield from arms[active].move(T.PARK[active], 1.2)


def _both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb)
    notes: list[str] = []

    def log(msg: str) -> None:
        notes.append(msg)
        print(msg, flush=True)

    gen = script(task, ctx, log)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        if ended is None:
            try:
                next(gen)
            except StopIteration:
                ended = steps
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "log": notes,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "log"}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
