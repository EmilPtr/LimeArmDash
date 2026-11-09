"""Tests for arm_script (no hardware, tkinter or ikpy needed).

Run:  .venv/bin/python -m unittest test_arm_script -v
(Don't use plain `unittest discover`: test_send.py is a hardware script.)
"""
import os
import unittest

import arm_script as s

LIMITS = {"base": (135, 270), "shoulder": (0, 120), "elbow": (75, 180), "wrist": (90, 270)}
HERE = os.path.dirname(os.path.abspath(__file__))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeArm:
    """Measured base/shoulder/elbow jump to the commanded value `reach_after` s after
    a frame is sent (never, for joints in `stuck`); wrist and claw echo immediately."""

    def __init__(self, clock, start=(180.0, 0.0, 180.0, 180.0), reach_after=0.1, stuck=()):
        self.clock, self.reach_after, self.stuck = clock, reach_after, set(stuck)
        self.pos = list(start)
        self.cmd, self.cmd_t, self.claw = list(start), 0.0, 0
        self.frames = []

    def send_frame(self, base, shoulder, elbow, wrist, claw=0):
        self._update()
        self.frames.append((base, shoulder, elbow, wrist, claw))
        self.cmd, self.cmd_t, self.claw = [base, shoulder, elbow, wrist], self.clock(), claw

    def _update(self):
        if self.clock() - self.cmd_t >= self.reach_after:
            for i in range(3):
                if i not in self.stuck:
                    self.pos[i] = self.cmd[i]
        self.pos[3] = self.cmd[3]

    def get_joints(self):
        self._update()
        return (*self.pos, self.claw)


def fake_ik(x, y, z):
    if x > 500:
        raise ValueError("Target unreachable")
    return {"base": 180.0 + x / 10, "shoulder": y / 10, "elbow": 90.0 + z / 10, "wrist": 0.0}


def make(clock=None, **arm_kw):
    clock = clock or FakeClock()
    arm = FakeArm(clock, **arm_kw)
    events = []
    runner = s.ScriptRunner(arm, fake_ik, LIMITS, on_event=events.append, clock=clock, sleep=clock.sleep)
    return runner, arm, events, clock


def statuses(events):
    return [(e.line, e.status) for e in events if e.status != "running"]


class ParserTests(unittest.TestCase):
    def test_all_commands_comments_and_defaults(self):
        cmds = s.parse_script(
            "# header\n\nMOVE 10 20.5 -30 timeout=500  # trailing\n"
            "joint Shoulder 45\nclaw open timeout=0\n  claw CLOSE\n")
        self.assertEqual([(c.line, c.kind, c.args, c.timeout_ms) for c in cmds], [
            (3, "move", (10.0, 20.5, -30.0), 500),
            (4, "joint", ("shoulder", 45.0), s.DEFAULT_TIMEOUT_MS),
            (5, "claw", ("open",), 0),
            (6, "claw", ("close",), s.DEFAULT_TIMEOUT_MS),
        ])

    def test_errors_report_every_bad_line(self):
        bad = ("move 1 2\njoint knee 10\nclaw half\nmove 1 2 3 timeout=-5\n"
               "joint base abc\njump 1\nmove 1 2 3 speed=4\nmove 1 2 3 timeout=1 timeout=2\n")
        with self.assertRaises(s.ScriptError) as cm:
            s.parse_script("move 0 0 0\n" + bad)
        lines = [e.split(":")[0] for e in cm.exception.errors]
        self.assertEqual(lines, [f"line {n}" for n in range(2, 10)])

    def test_example_script_parses(self):
        cmds = s.load_script(os.path.join(HERE, "scripts", "example.txt"))
        self.assertEqual({c.kind for c in cmds}, {"move", "joint", "claw", "wrist_abs"})


class RunnerTests(unittest.TestCase):
    def test_reached_early_moves_on_immediately(self):
        runner, arm, events, clock = make(reach_after=0.1)
        runner.run(s.parse_script("joint base 200 timeout=5000\njoint elbow 100 timeout=5000"))
        self.assertEqual(statuses(events), [(1, "reached"), (2, "reached"), (0, "finished")])
        self.assertLess(clock.t, 0.5)  # nowhere near the 10 s of timeouts

    def test_within_tolerance_counts_as_reached(self):
        runner, arm, events, clock = make(reach_after=0.0)
        arm.stuck = {0}
        arm.pos[0] = 200.0 - s.ACCEPTABLE_ERROR_DEG * 0.9
        runner.run(s.parse_script("joint base 200 timeout=1000"))
        self.assertEqual(statuses(events)[0], (1, "reached"))

    def test_timeout_then_continue(self):
        runner, arm, events, clock = make(stuck={1})
        runner.run(s.parse_script("joint shoulder 60 timeout=300\njoint base 200 timeout=3000"))
        self.assertEqual(statuses(events), [(1, "timeout"), (2, "reached"), (0, "finished")])
        self.assertIn("shoulder off by", events[1].message)
        self.assertGreaterEqual(clock.t, 0.3)
        self.assertLess(clock.t, 0.6)

    def test_bad_syntax_rejected_before_any_motion(self):
        runner, arm, events, clock = make()
        with self.assertRaises(s.ScriptError):
            runner.run(s.parse_script("joint base 200\nclaw open\nmove 1 2\n"))
        self.assertEqual(arm.frames, [])
        self.assertEqual(events, [])

    def test_partial_joint_command_keeps_other_joints(self):
        runner, arm, events, clock = make(start=(180.0, 10.0, 150.0, 200.0))
        runner.run(s.parse_script("joint elbow 120 timeout=1000\njoint base 150 timeout=1000"))
        self.assertEqual(arm.frames, [(180.0, 10.0, 120.0, 200.0, 0), (150.0, 10.0, 120.0, 200.0, 0)])

    def test_claw_commands_resend_pose_and_wait_full_timeout(self):
        runner, arm, events, clock = make(start=(180.0, 10.0, 150.0, 200.0))
        runner.run(s.parse_script("joint base 200 timeout=1000\nclaw open timeout=400\n"
                                  "claw close timeout=400\njoint wrist 120 timeout=200"))
        self.assertEqual(arm.frames, [
            (200.0, 10.0, 150.0, 200.0, 0),
            (200.0, 10.0, 150.0, 200.0, s.CLAW_OPEN),
            (200.0, 10.0, 150.0, 200.0, s.CLAW_CLOSE),
            (200.0, 10.0, 150.0, 120.0, 0),
        ])
        self.assertEqual(statuses(events), [(1, "reached"), (2, "done"), (3, "done"), (4, "done"), (0, "finished")])
        self.assertGreaterEqual(clock.t, 0.1 + 0.4 + 0.4 + 0.2 - 1e-9)  # claw/wrist use full timeout
        self.assertLess(clock.t, 1.2)

    def test_move_uses_ik_keeps_wrist_and_clamps(self):
        runner, arm, events, clock = make(start=(180.0, 0.0, 180.0, 210.0))
        runner.run(s.parse_script("move 100 200 300 timeout=1000\nmove 0 2000 0 timeout=1000"))
        self.assertEqual(arm.frames[0], (190.0, 20.0, 120.0, 210.0, 0))
        self.assertEqual(arm.frames[1], (180.0, 120.0, 90.0, 210.0, 0))  # shoulder 200 -> 120
        self.assertIn("clamped: shoulder", [e for e in events if e.line == 2][0].message)
        self.assertEqual(statuses(events), [(1, "reached"), (2, "reached"), (0, "finished")])

    def test_unreachable_ik_is_error_then_continue(self):
        runner, arm, events, clock = make()
        runner.run(s.parse_script("move 900 0 0 timeout=1000\njoint base 200 timeout=1000"))
        self.assertEqual(statuses(events), [(1, "error"), (2, "reached"), (0, "finished")])
        self.assertIn("IK failed", events[0].message)
        self.assertEqual(len(arm.frames), 1)

    def test_stop_cancels_and_holds(self):
        clock = FakeClock()
        runner, arm, events, _ = make(clock, stuck={0})

        def sleep(dt):
            clock.sleep(dt)
            if clock.t > 0.2:
                runner.stop()
        runner.sleep = sleep
        runner.run(s.parse_script("joint base 200 timeout=5000\njoint elbow 100 timeout=5000"))
        self.assertEqual(statuses(events), [(1, "stopped")])
        self.assertEqual(arm.frames[-1], (180.0, 0.0, 180.0, 180.0, 0))  # hold at measured pose
        self.assertEqual(len(arm.frames), 2)

    def test_background_thread(self):
        import time
        arm = FakeArm(time.monotonic, reach_after=0.0)
        events = []
        runner = s.ScriptRunner(arm, fake_ik, LIMITS, on_event=events.append)
        runner.start(s.parse_script("joint base 200 timeout=2000\nclaw open timeout=50"))
        runner.join(5)
        self.assertFalse(runner.running)
        self.assertEqual(statuses(events), [(1, "reached"), (2, "done"), (0, "finished")])


class WristAbsTests(unittest.TestCase):
    def test_parse(self):
        cmds = s.parse_script("wrist_abs on 45\nWRIST_ABS OFF\nwrist_abs on -90")
        self.assertEqual([c.args for c in cmds], [(True, 45.0), (False,), (True, -90.0)])
        for bad in ("wrist_abs", "wrist_abs on", "wrist_abs on 91", "wrist_abs off 3", "wrist_abs maybe"):
            with self.assertRaises(s.ScriptError, msg=bad):
                s.parse_script(bad)

    def test_on_applies_to_every_move_until_off(self):
        runner, arm, events, clock = make(start=(180.0, 0.0, 180.0, 200.0))
        runner.run(s.parse_script(
            "move 0 0 0\nwrist_abs on 0\nmove 100 200 300\nmove 0 400 0\nwrist_abs off\nmove 0 100 0"))
        self.assertEqual(len(arm.frames), 3 + 1)  # wrist_abs lines send no frame
        self.assertEqual(arm.frames[0][3], 200.0)  # off: wrist untouched
        # shoulder 20, elbow 120 -> wrist = 0 + 360 - 140 = 220
        self.assertEqual(arm.frames[1][:4], (190.0, 20.0, 120.0, 220.0))
        # shoulder 40 -> elbow 90: 360 - 130 = 230
        self.assertEqual(arm.frames[2][3], 230.0)
        self.assertEqual(arm.frames[3][3], 230.0)  # off again: keeps last wrist

    def test_angle_and_clamping(self):
        runner, arm, events, clock = make()
        runner.run(s.parse_script("wrist_abs on 30\nmove 0 100 0\nwrist_abs on -90\nmove 0 100 0"))
        # shoulder 10, elbow 90: 30 + 360 - 100 = 290 -> clamped to 270; -90: 170
        self.assertEqual([f[3] for f in arm.frames], [270.0, 170.0])
        self.assertIn("clamped: wrist", [e for e in events if e.line == 2][0].message)

    def test_uses_clamped_shoulder_and_state_resets_each_run(self):
        runner, arm, events, clock = make()
        runner.run(s.parse_script("wrist_abs on 0\nmove 0 2000 0"))  # shoulder 200 -> 120, elbow 90
        self.assertEqual(arm.frames[0][3], 360.0 - 120.0 - 90.0)
        runner.run(s.parse_script("move 0 100 0"))
        self.assertEqual(arm.frames[1][3], arm.frames[0][3])  # state was reset: wrist untouched

    def test_joint_commands_do_not_touch_wrist(self):
        runner, arm, events, clock = make(start=(180.0, 0.0, 180.0, 200.0))
        runner.run(s.parse_script("wrist_abs on 0\njoint shoulder 50\njoint wrist 150"))
        self.assertEqual([f[3] for f in arm.frames], [200.0, 150.0])


if __name__ == "__main__":
    unittest.main()
