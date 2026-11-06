"""Arm script files: parse and run plain-text command scripts on the LimeArm.

No tkinter, ikpy or pyserial in here: the arm (anything with `send_frame` /
`get_joints`), the IK function, the clock and the sleep are all injected, so the
runner can be tested without hardware.

Script syntax
-------------
One command per line, run top to bottom. Keywords are case-insensitive, blank
lines are ignored and `#` starts a comment (whole line or end of line).

    move <x_mm> <y_mm> <z_mm>    [timeout=<ms>]   whole arm to a wrist point (IK)
    joint <joint> <degrees>      [timeout=<ms>]   one joint to a driver angle;
                                                  <joint> = base|shoulder|elbow|wrist
    claw open                    [timeout=<ms>]
    claw close                   [timeout=<ms>]
    wrist_abs on <deg>                            absolute wrist pitch ON: <deg> above the
                                                  horizontal (0 level, 90 upright, -90 down)
    wrist_abs off                                 absolute wrist pitch OFF

`timeout=<ms>` is optional; DEFAULT_TIMEOUT_MS is used when it is left out.
Coordinates are in the same frame/units as the dashboard's IK page (mm, +Y
front, +Z up, target = wrist point). Angles are driver degrees, as on the
Joint Control page.

Semantics
---------
* The whole file is parsed before anything moves; any syntax error aborts the
  run with every bad line reported.
* Each command sends one frame, then waits until the target is reached
  (every checked joint within ACCEPTABLE_ERROR_DEG of its target) or its
  timeout expires, then moves on to the next command either way.
* Unreachable `move` targets (IK error) are reported as an error for that line
  and the runner moves on.
* Targets outside the joint limits are clamped to the limits (same as the
  dashboard's IK page) and the clamp is reported.
* `send_frame` always carries all four joints, so the runner remembers the
  last commanded pose and resends it for the joints a command doesn't touch.
  `move` keeps the last commanded wrist angle (IK gives no wrist).
* Absolute wrist pitch: `wrist_abs on <deg>` / `wrist_abs off` only change runner
  state (no frame is sent, no waiting; the state starts OFF for every run and is
  remembered until changed). While ON, every `move` also computes the wrist
  driver angle that holds the claw at <deg> above horizontal, from the
  (clamped) shoulder and elbow it is about to send:
      claw pitch up = shoulder + elbow + wrist - 360      (URDF pitch-joint mapping)
      wrist = <deg> + 360 - shoulder - elbow              (clamped to wrist limits)
  and sends it with the rest of the pose. While OFF, `move` leaves the wrist
  alone. `joint` commands never touch the wrist except `joint wrist` itself,
  which is an explicit manual override (the next `move` re-applies absolute).
* Feedback: telemetry base/shoulder/elbow are measured, but wrist and claw only
  echo the last command. So commands with no measured joint (`joint wrist ...`,
  `claw ...`) cannot finish early: they always use their full timeout as a
  settle time, then report "done". (A wrist move whose value is not echoed
  back by then is reported as "timeout".)
"""
import math
import threading
import time
from dataclasses import dataclass

# A joint counts as "at target" when |measured - target| <= this (driver degrees).
ACCEPTABLE_ERROR_DEG = 2.0
# Timeout used for lines that don't give `timeout=<ms>`.
DEFAULT_TIMEOUT_MS = 3000
# How often telemetry is checked while waiting for a command to finish.
POLL_INTERVAL_S = 0.02

JOINT_NAMES = ("base", "shoulder", "elbow", "wrist")
MEASURED_JOINTS = (0, 1, 2)  # indices with real telemetry feedback (not wrist)

# Mirrors CLAW_* in serial.py (not imported: that module needs pyserial).
CLAW_HOLD, CLAW_OPEN, CLAW_CLOSE = 0, 1, 2


# ------------------------------------------------------------------ parsing ---
class ScriptError(ValueError):
    """Script failed to parse. `errors` holds one 'line N: ...' string per bad line."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Command:
    line: int        # 1-based line number in the file
    text: str        # the source line, comment stripped
    kind: str        # "move" | "joint" | "claw" | "wrist_abs"
    args: tuple      # move: (x, y, z); joint: (name, deg); claw: ("open"|"close",);
                     # wrist_abs: (True, deg) or (False,)
    timeout_ms: int


def _number(s, what):
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{what} must be a number, got '{s}'") from None
    if not math.isfinite(v):
        raise ValueError(f"{what} must be a finite number, got '{s}'")
    return v


def _parse_line(n, line):
    words = line.split()
    op = words[0].lower()
    timeout = None
    args = []
    for w in words[1:]:
        if "=" in w:
            key, _, val = w.partition("=")
            if key.lower() != "timeout":
                raise ValueError(f"unknown option '{key}=' (only timeout=<ms> is allowed)")
            if timeout is not None:
                raise ValueError("timeout given twice")
            t = _number(val, "timeout")
            if t < 0 or t != int(t):
                raise ValueError(f"timeout must be a whole number of ms >= 0, got '{val}'")
            timeout = int(t)
        else:
            args.append(w)
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_MS

    if op == "move":
        if len(args) != 3:
            raise ValueError("expected: move <x_mm> <y_mm> <z_mm> [timeout=<ms>]")
        parsed = tuple(_number(a, ax) for a, ax in zip(args, "xyz"))
    elif op == "joint":
        if len(args) != 2:
            raise ValueError("expected: joint <base|shoulder|elbow|wrist> <degrees> [timeout=<ms>]")
        name = args[0].lower()
        if name not in JOINT_NAMES:
            raise ValueError(f"unknown joint '{args[0]}' (expected base, shoulder, elbow or wrist)")
        parsed = (name, _number(args[1], "angle"))
    elif op == "claw":
        if len(args) != 1 or args[0].lower() not in ("open", "close"):
            raise ValueError("expected: claw open|close [timeout=<ms>]")
        parsed = (args[0].lower(),)
    elif op == "wrist_abs":
        usage = "expected: wrist_abs on <degrees above horizontal, -90..90>  |  wrist_abs off"
        state = args[0].lower() if args else ""
        if state == "off" and len(args) == 1:
            parsed = (False,)
        elif state == "on" and len(args) == 2:
            deg = _number(args[1], "angle")
            if not -90.0 <= deg <= 90.0:
                raise ValueError(f"absolute wrist angle must be within -90..90, got '{args[1]}'")
            parsed = (True, deg)
        else:
            raise ValueError(usage)
    else:
        raise ValueError(f"unknown command '{words[0]}' (expected move, joint, claw or wrist_abs)")
    return Command(n, line, op, parsed, timeout)


def parse_script(text):
    """Parse a whole script. Returns a list of Commands or raises ScriptError."""
    commands, errors = [], []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            commands.append(_parse_line(n, line))
        except ValueError as e:
            errors.append(f"line {n}: {e}   [{line}]")
    if errors:
        raise ScriptError(errors)
    return commands


def load_script(path):
    with open(path, encoding="utf-8") as f:
        return parse_script(f.read())


# ------------------------------------------------------------------ running ---
@dataclass(frozen=True)
class Event:
    """Progress report. status is one of:
    running  - command sent, waiting for it to finish
    reached  - target reached within ACCEPTABLE_ERROR_DEG
    done     - no-feedback command (wrist / claw) finished its settle time
    timeout  - timeout expired before the target was reached; moving on
    error    - command failed (e.g. IK unreachable); moving on
    stopped  - run cancelled by stop()          (line = interrupted line, or 0)
    finished - all commands processed           (line = 0)
    """
    line: int
    text: str
    status: str
    message: str = ""
    pose: tuple = None  # commanded (base, shoulder, elbow, wrist) when known


class ScriptRunner:
    """Runs parsed commands against an arm.

    arm      - object with send_frame(base, shoulder, elbow, wrist, claw) and
               get_joints() -> (base, shoulder, elbow, wrist, claw)
    ik       - fn(x_mm, y_mm, z_mm) -> {"base", "shoulder", "elbow", ...} degrees,
               raises on unreachable targets
    limits   - {"base": (lo, hi), ...} driver-degree joint limits; targets are clamped
    on_event - callback(Event); called from the runner thread when start() is used
    """

    def __init__(self, arm, ik, limits, on_event=None, clock=time.monotonic, sleep=time.sleep,
                 tolerance=ACCEPTABLE_ERROR_DEG, poll_s=POLL_INTERVAL_S):
        self.arm, self.ik = arm, ik
        self.limits = [tuple(limits[n]) for n in JOINT_NAMES]
        self.on_event = on_event or (lambda ev: None)
        self.clock, self.sleep = clock, sleep
        self.tolerance, self.poll_s = tolerance, poll_s
        self.pose = None  # last commanded (base, shoulder, elbow, wrist)
        self.wrist_abs = None  # None = off, else claw pitch in degrees above horizontal
        self._stop = threading.Event()
        self._thread = None

    # ---- control ----
    def start(self, commands):
        """Run in a background thread (doesn't block the GUI)."""
        if self.running:
            raise RuntimeError("script already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(list(commands),), daemon=True)
        self._thread.start()

    def run(self, commands):
        """Run synchronously in the calling thread."""
        self._stop.clear()
        self._run(list(commands))

    def stop(self):
        """Cancel: the current command is abandoned and the arm told to hold where it is."""
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # ---- internals ----
    def _emit(self, line, text, status, message="", pose=None):
        try:
            self.on_event(Event(line, text, status, message, pose))
        except Exception:
            pass  # a broken UI callback must not kill the run

    def _clamp(self, pose):
        out, clamped = [], []
        for name, v, (lo, hi) in zip(JOINT_NAMES, pose, self.limits):
            c = min(hi, max(lo, v))
            if c != v:
                clamped.append(f"{name} {v:.1f}->{c:.1f}")
            out.append(float(c))
        return tuple(out), clamped

    def _run(self, commands):
        # Start from where the arm is (base/shoulder/elbow measured, wrist = last command).
        self.pose, _ = self._clamp(self.arm.get_joints()[:4])
        self.wrist_abs = None
        for cmd in commands:
            if self._stop.is_set():
                break
            try:
                self._execute(cmd)
            except Exception as e:  # serial failure etc.: report and move on
                self._emit(cmd.line, cmd.text, "error", f"{type(e).__name__}: {e}", self.pose)
            if self._stop.is_set():
                self._hold()
                self._emit(cmd.line, cmd.text, "stopped", "stopped by user", self.pose)
                return
        if self._stop.is_set():
            self._hold()
            self._emit(0, "", "stopped", "stopped by user", self.pose)
        else:
            self._emit(0, "", "finished", f"{len(commands)} command(s) processed", self.pose)

    def _hold(self):
        """Freeze the arm at its measured position (wrist at its last command)."""
        try:
            tel = self.arm.get_joints()
            self.pose, _ = self._clamp((tel[0], tel[1], tel[2], self.pose[3]))
            self.arm.send_frame(*self.pose, CLAW_HOLD)
        except Exception:
            pass

    def _wrist_for(self, shoulder, elbow):
        """Wrist driver angle holding the claw at self.wrist_abs degrees above horizontal."""
        return self.wrist_abs + 360.0 - shoulder - elbow

    def _execute(self, cmd):
        claw = CLAW_HOLD
        if cmd.kind == "wrist_abs":
            self.wrist_abs = cmd.args[1] if cmd.args[0] else None
            self._emit(cmd.line, cmd.text, "done",
                       f"absolute wrist {self.wrist_abs:.1f} deg above horizontal"
                       if self.wrist_abs is not None else "absolute wrist off", self.pose)
            return
        if cmd.kind == "move":
            try:
                sol = self.ik(*cmd.args)
            except Exception as e:
                self._emit(cmd.line, cmd.text, "error", f"IK failed: {e}", self.pose)
                return
            target = (sol["base"], sol["shoulder"], sol["elbow"], self.pose[3])
            if self.wrist_abs is not None:
                # use the shoulder/elbow that will really be sent (after clamping)
                (_, sh, el, _), _ = self._clamp(target)
                target = (*target[:3], self._wrist_for(sh, el))
            check = MEASURED_JOINTS
        elif cmd.kind == "joint":
            i = JOINT_NAMES.index(cmd.args[0])
            target = list(self.pose)
            target[i] = cmd.args[1]
            check = (i,) if i in MEASURED_JOINTS else ()
        else:  # claw
            target = self.pose
            claw = CLAW_OPEN if cmd.args[0] == "open" else CLAW_CLOSE
            check = ()

        target, clamped = self._clamp(target)
        self.pose = target
        self.arm.send_frame(*target, claw)
        self._emit(cmd.line, cmd.text, "running",
                   ("clamped: " + ", ".join(clamped)) if clamped else "", target)

        deadline = self.clock() + cmd.timeout_ms / 1000.0
        while True:
            if self._stop.is_set():
                return
            tel = self.arm.get_joints()
            if check and all(abs(tel[i] - target[i]) <= self.tolerance for i in check):
                self._emit(cmd.line, cmd.text, "reached", "", target)
                return
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            self.sleep(min(self.poll_s, remaining))

        if check:
            errs = ", ".join(f"{JOINT_NAMES[i]} off by {abs(tel[i] - target[i]):.1f}" for i in check
                             if abs(tel[i] - target[i]) > self.tolerance)
            self._emit(cmd.line, cmd.text, "timeout", errs, target)
        elif cmd.kind == "joint" and abs(tel[3] - target[3]) > self.tolerance:
            self._emit(cmd.line, cmd.text, "timeout",
                       f"wrist not echoed (telemetry {tel[3]:.1f})", target)
        else:
            self._emit(cmd.line, cmd.text, "done", "no feedback; waited full timeout", target)
