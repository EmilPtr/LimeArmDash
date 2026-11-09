"""LimeArm dashboard: joint / IK control pages + 3D visualizer (Win95 style).

Run:  python app.py
Visualizer frame = URDF frame, +Z up. Coordinates typed on the IK page are in this
frame, in mm.
"""
import importlib.util
import math
import os
import queue
import sys
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # ik.py loads "arm.urdf" relative to cwd
URDF = os.path.join(HERE, "arm.urdf")

import arm_script  # noqa: E402
import ik  # noqa: E402  (needs HERE on sys.path; it is, as the script dir)

# serial.py in this folder shadows pyserial: load it under another name with
# this folder temporarily off sys.path so its `import serial` finds pyserial.
_saved = list(sys.path)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import serial.tools.list_ports as list_ports  # noqa: E402

_spec = importlib.util.spec_from_file_location("limearm_serial", os.path.join(HERE, "serial.py"))
limearm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(limearm)
sys.path = _saved

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# ---------------------------------------------------------------- joints ----
# name: (min, max, default) in driver degrees (limits from firmware main.cpp)
JOINTS = {
    "Base": (135, 270, 180),
    "Shoulder": (0, 120, 0),
    "Elbow": (75, 180, 180),
    "Wrist": (90, 270, 180),
}
DEFAULTS = tuple(v[2] for v in JOINTS.values())


ENC_RANGE = ((135, 270), (0, 120), (75, 180), (90, 270))


def desired_pose(cmd):
    return tuple(min(hi, max(lo, c)) for c, (lo, hi) in zip(cmd, ENC_RANGE))


def current_pose(tel):
    return tuple(tel[:4])


def driver_to_urdf(base, shoulder, elbow, wrist):
    """Driver degrees -> URDF joint radians (inverse of the mapping in ik.py)."""
    return {
        "base_yaw_joint": math.radians(base - 180.0),
        "shoulder_pitch_joint": math.radians(-shoulder),
        "elbow_pitch_joint": math.radians(180.0 - elbow),
        "wrist_pitch_joint": math.radians(180.0 - wrist),
    }


# ------------------------------------------------------------ URDF model ----
def _vec(s, n=3):
    return [float(v) for v in s.split()] if s else [0.0] * n


def _rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _tf(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_mat(*rpy)
    T[:3, 3] = xyz
    return T


def _axis_rot(axis, a):
    x, y, z = np.array(axis) / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    T = np.eye(4)
    T[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return T


class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        mats = {m.get("name"): _vec(m.find("color").get("rgba"), 4) for m in root.findall("material")}
        self.visuals = {}  # link -> [(T_local, kind, params, rgba)]
        for link in root.findall("link"):
            vis = []
            for v in link.findall("visual"):
                o = v.find("origin")
                T = _tf(_vec(o.get("xyz")), _vec(o.get("rpy"))) if o is not None else np.eye(4)
                g = v.find("geometry")[0]
                rgba = mats.get(v.find("material").get("name"), [0.7, 0.7, 0.7, 1])
                vis.append((T, g.tag, {k: _vec(a) for k, a in g.attrib.items()}, rgba))
            self.visuals[link.get("name")] = vis
        self.joints = []  # in file order: parent-before-child for this URDF
        children = set()
        for j in root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            self.joints.append({
                "name": j.get("name"), "type": j.get("type"),
                "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                "T": _tf(_vec(o.get("xyz")), _vec(o.get("rpy"))),
                "axis": _vec(ax.get("xyz")) if ax is not None else [0, 0, 1],
            })
            children.add(j.find("child").get("link"))
        self.root = next(l for l in self.visuals if l not in children)

    def fk(self, angles):
        """angles: joint name -> radians. Returns link -> 4x4 world transform."""
        T = {self.root: np.eye(4)}
        for j in self.joints:
            t = T[j["parent"]] @ j["T"]
            if j["type"] == "revolute":
                t = t @ _axis_rot(j["axis"], angles.get(j["name"], 0.0))
            T[j["child"]] = t
        return T


def _mesh(kind, p):
    """Local-frame polygons (list of Nx3 arrays) for a URDF primitive."""
    if kind == "box":
        sx, sy, sz = [v / 2 for v in p["size"]]
        c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        return [c[list(f)] for f in faces]
    if kind == "cylinder":
        r, h = p["radius"][0], p["length"][0] / 2
        a = np.linspace(0, 2 * math.pi, 25)
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        polys = [np.array([[*ring[i], -h], [*ring[i + 1], -h], [*ring[i + 1], h], [*ring[i], h]])
                 for i in range(24)]
        polys.append(np.c_[ring, np.full(25, h)])
        polys.append(np.c_[ring, np.full(25, -h)])
        return polys
    if kind == "sphere":
        r = p["radius"][0]
        th = np.linspace(0, math.pi, 9)
        ph = np.linspace(0, 2 * math.pi, 13)
        pt = lambda t, f: [r * math.sin(t) * math.cos(f), r * math.sin(t) * math.sin(f), r * math.cos(t)]
        return [np.array([pt(th[i], ph[k]), pt(th[i + 1], ph[k]), pt(th[i + 1], ph[k + 1]), pt(th[i], ph[k + 1])])
                for i in range(8) for k in range(12)]
    return []


# Display frame == URDF frame (arm at base=180° points along +X).
DISPLAY = np.eye(4)
LIGHT = np.array([-0.4, -0.5, 0.8]) / np.linalg.norm([-0.4, -0.5, 0.8])


def arm_collections(urdf, angles, ghost):
    T = urdf.fk(angles)
    out = []
    for link, vis in urdf.visuals.items():
        for Tv, kind, params, rgba in vis:
            W = DISPLAY @ T[link] @ Tv
            polys, cols = [], []
            for poly in _mesh(kind, params):
                q = poly @ W[:3, :3].T + W[:3, 3]
                n = np.cross(q[1] - q[0], q[2] - q[0])
                n = n / (np.linalg.norm(n) + 1e-12)
                shade = 0.45 + 0.55 * abs(float(n @ LIGHT))
                if ghost:
                    cols.append((0.75 * shade, 0.75 * shade, 0.78 * shade, 0.22))
                else:
                    cols.append((rgba[0] * shade, rgba[1] * shade, rgba[2] * shade, 1.0))
                polys.append(q)
            pc = Poly3DCollection(polys, facecolors=cols,
                                  edgecolors=(0.7, 0.7, 0.75, 0.25) if ghost else (0, 0, 0, 0.45),
                                  linewidths=0.4)
            out.append(pc)
    return out


# -------------------------------------------------------------- visualizer --
BG = "#0b0d12"


class Visualizer:
    H = 0.3  # half extent of the floor, metres

    def __init__(self, parent, urdf):
        self.urdf = urdf
        self.scale = 1.0
        self.fig = Figure(facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self._scroll)
        ax = self.ax
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 0.667), zoom=1.35)
        ax.view_init(elev=24, azim=58)
        self._limits()
        H = self.H
        for t in np.arange(-H, H + 1e-9, 0.05):  # floor grid
            c = "#2a3140" if abs(t) > 1e-9 else "#3d4660"
            ax.plot([-H, H], [t, t], [0, 0], color=c, lw=0.6)
            ax.plot([t, t], [-H, H], [0, 0], color=c, lw=0.6)
        for v, col, lab in (((0.12, 0, 0), "#ff5c5c", "X"), ((0, 0.12, 0), "#5cff8a", "Y (front)"),
                            ((0, 0, 0.12), "#5c9dff", "Z")):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2)
            ax.text(v[0] * 1.15, v[1] * 1.15, v[2] * 1.15 + 0.005, lab, color=col, fontsize=8)
        self.dynamic = []

    def _limits(self):
        s, H = self.scale, self.H
        self.ax.set_xlim(-H * s, H * s)
        self.ax.set_ylim(-H * s, H * s)
        self.ax.set_zlim(0, 0.4 * s)

    def _scroll(self, ev):
        self.scale = min(2.0, max(0.25, self.scale * (0.9 if ev.button == "up" else 1.1)))
        self._limits()
        self.canvas.draw_idle()

    def draw(self, current, desired):
        for a in self.dynamic:
            a.remove()
        self.dynamic = []
        for col in arm_collections(self.urdf, driver_to_urdf(*desired_pose(desired)), ghost=True) + \
                arm_collections(self.urdf, driver_to_urdf(*current_pose(current)), ghost=False):
            self.ax.add_collection3d(col)
            self.dynamic.append(col)
        self.canvas.draw_idle()


# --------------------------------------------------------------------- UI ---
GREY, DARK, NAVY, WHITE = "#c0c0c0", "#808080", "#000080", "#ffffff"
FONT = ("DejaVu Sans", 9)
FONT_B = ("DejaVu Sans", 9, "bold")


def btn(parent, text, cmd, **kw):
    return tk.Button(parent, text=text, command=cmd, bg=GREY, activebackground=GREY, relief="raised",
                     bd=2, font=FONT, padx=10, pady=2, highlightthickness=0, **kw)


def entry(parent, var, width=8):
    return tk.Entry(parent, textvariable=var, width=width, bg=WHITE, relief="sunken", bd=2, font=FONT,
                    highlightthickness=0)


class App:
    def __init__(self, root):
        self.root = root
        self.arm = None
        self.desired = DEFAULTS
        self.last_drawn = None
        self.runner = None  # arm_script.ScriptRunner while/after a script runs
        self.script_events = queue.Queue()  # runner thread -> GUI thread
        root.title("LimeArm Dashboard")
        root.configure(bg=GREY, bd=3, relief="raised")
        root.geometry("900x860")

        # title bar
        tb = tk.Frame(root, bg=NAVY)
        tb.pack(fill="x", padx=2, pady=2)
        tk.Label(tb, text="  LimeArm Dashboard", bg=NAVY, fg=WHITE, font=FONT_B).pack(side="left", pady=2)

        # connection row
        cf = tk.Frame(root, bg=GREY)
        cf.pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(cf, text="Port:", bg=GREY, font=FONT).pack(side="left")
        ports = [p.device for p in list_ports.comports()]
        self.port = tk.StringVar(value="/dev/ttyUSB1")
        entry(cf, self.port, 18).pack(side="left", padx=4)
        self.conn_btn = btn(cf, "Connect", self.toggle_connect)
        self.conn_btn.pack(side="left")
        tk.Label(cf, text=("found: " + ", ".join(ports)) if ports else "no ports found", bg=GREY,
                 fg=DARK, font=FONT).pack(side="left", padx=8)

        # tabs
        tabs = tk.Frame(root, bg=GREY)
        tabs.pack(fill="x", padx=6, pady=(8, 0))
        self.page = tk.StringVar(value="joints")
        for text, val in (("Joint Control", "joints"), ("Inverse Kinematics", "ik"), ("Script", "script")):
            tk.Radiobutton(tabs, text=text, variable=self.page, value=val, indicatoron=False,
                           command=self.show_page, bg=GREY, selectcolor=WHITE, activebackground=GREY,
                           relief="raised", bd=2, font=FONT, padx=12, pady=3,
                           highlightthickness=0).pack(side="left")

        # page area
        self.pagebox = tk.Frame(root, bg=GREY, relief="raised", bd=2)
        self.pagebox.pack(fill="x", padx=6)
        self.build_joint_page()
        self.build_ik_page()
        self.build_script_page()
        self.build_tool_bar()

        # status bar (packed before visualizer so it keeps its space)
        self.status = tk.StringVar(value="Not connected")
        tk.Label(root, textvariable=self.status, bg=GREY, anchor="w", relief="sunken", bd=2,
                 font=FONT).pack(side="bottom", fill="x", padx=6, pady=6)

        # visualizer
        vf = tk.Frame(root, relief="sunken", bd=3, bg=BG)
        vf.pack(fill="both", expand=True, padx=6, pady=6)
        self.viz = Visualizer(vf, Urdf(URDF))

        self.show_page()
        self.tick()

    # ---- pages ----
    def build_joint_page(self):
        f = self.joint_page = tk.Frame(self.pagebox, bg=GREY)
        self.sliders, self.jvars = {}, {}
        for r, (name, (lo, hi, d)) in enumerate(JOINTS.items()):
            tk.Label(f, text=name, bg=GREY, font=FONT, width=9, anchor="w").grid(row=r, column=0, padx=6, pady=3)
            s = tk.Scale(f, from_=lo, to=hi, orient="horizontal", resolution=0.1, showvalue=False,
                         length=520, bg=GREY, troughcolor=WHITE, relief="raised", bd=2,
                         highlightthickness=0, sliderrelief="raised",
                         command=lambda v, n=name: self.slider_moved(n))
            s.set(d)
            s.grid(row=r, column=1, padx=4)
            var = tk.StringVar(value=f"{d:.1f}")
            e = entry(f, var)
            e.grid(row=r, column=2, padx=4)
            e.bind("<Return>", lambda _e, n=name: self.entry_changed(n))
            e.bind("<FocusOut>", lambda _e, n=name: self.entry_changed(n))
            tk.Label(f, text=f"{lo}-{hi}°", bg=GREY, fg=DARK, font=FONT).grid(row=r, column=3, padx=4)
            self.sliders[name], self.jvars[name] = s, var
        btn(f, "Send", self.send_joints).grid(row=len(JOINTS), column=1, pady=6, sticky="e")
        self.fk_text = tk.StringVar()
        tk.Label(f, textvariable=self.fk_text, bg=GREY, fg=DARK, font=FONT, anchor="w").grid(
            row=len(JOINTS) + 1, column=0, columnspan=4, padx=8, pady=(0, 6), sticky="w")

    def build_ik_page(self):
        f = self.ik_page = tk.Frame(self.pagebox, bg=GREY)
        self.xyz = {}
        defaults = {"X": 0, "Y": 250, "Z": 100}
        for c, ax in enumerate("XYZ"):
            tk.Label(f, text=f"{ax} (mm)", bg=GREY, font=FONT).grid(row=0, column=2 * c, padx=(12, 2), pady=10)
            var = tk.StringVar(value=str(defaults[ax]))
            entry(f, var).grid(row=0, column=2 * c + 1, padx=2)
            self.xyz[ax] = var
        btn(f, "Solve IK && Send", self.send_ik).grid(row=0, column=6, padx=16)
        tk.Label(f, text="Target = wrist point.  +Y front, +Z up.", bg=GREY, fg=DARK,
                 font=FONT).grid(row=1, column=0, columnspan=7, pady=(0, 2))
        self.angles_text = tk.StringVar()
        tk.Label(f, textvariable=self.angles_text, bg=GREY, fg=DARK, font=FONT).grid(
            row=2, column=0, columnspan=7, pady=(0, 6))

    def build_script_page(self):
        f = self.script_page = tk.Frame(self.pagebox, bg=GREY)
        self.script_path = tk.StringVar(value=os.path.join(HERE, "scripts", "example.txt"))
        tk.Label(f, text="File:", bg=GREY, font=FONT).grid(row=0, column=0, padx=(6, 2), pady=6)
        entry(f, self.script_path, 62).grid(row=0, column=1, padx=2, sticky="we")
        btn(f, "Browse...", self.browse_script).grid(row=0, column=2, padx=2)
        btn(f, "Run", self.run_script).grid(row=0, column=3, padx=2)
        btn(f, "Stop", self.stop_script).grid(row=0, column=4, padx=(2, 6))
        self.script_log = tk.Listbox(f, height=6, bg=WHITE, relief="sunken", bd=2, font=FONT,
                                     highlightthickness=0, activestyle="none")
        self.script_log.grid(row=1, column=0, columnspan=5, padx=6, pady=(0, 6), sticky="we")
        f.columnconfigure(1, weight=1)

    def build_tool_bar(self):
        f = tk.Frame(self.root, bg=GREY, relief="raised", bd=2)
        f.pack(fill="x", padx=6, pady=(4, 0))
        self.abs_on = tk.BooleanVar(value=False)
        self.abs_angle = tk.StringVar(value="0")
        tk.Checkbutton(f, text="Absolute wrist pitch", variable=self.abs_on, command=self.wrist_mode_changed,
                       bg=GREY, activebackground=GREY, selectcolor=WHITE, font=FONT,
                       highlightthickness=0).pack(side="left", padx=6, pady=4)
        e = entry(f, self.abs_angle, 6)
        e.pack(side="left")
        e.bind("<Return>", lambda _e: self.wrist_mode_changed())
        e.bind("<FocusOut>", lambda _e: self.wrist_mode_changed())
        tk.Label(f, text="deg up   trim:", bg=GREY, fg=DARK, font=FONT).pack(side="left", padx=6)
        self.abs_trim = tk.StringVar(value="0")
        t = entry(f, self.abs_trim, 5)
        t.pack(side="left")
        t.bind("<Return>", lambda _e: self.wrist_mode_changed())
        t.bind("<FocusOut>", lambda _e: self.wrist_mode_changed())
        btn(f, "Close claw", lambda: self.claw(limearm.CLAW_CLOSE)).pack(side="right", padx=4, pady=3)
        btn(f, "Open claw", lambda: self.claw(limearm.CLAW_OPEN)).pack(side="right", padx=4, pady=3)

    def wrist_for(self, shoulder, elbow):
        """Wrist driver value. Absolute mode keeps the claw at a fixed pitch in world space:
        pitch_up = shoulder + elbow + wrist - 360 (from the URDF pitch-joint mapping)."""
        if not self.abs_on.get():
            return float(self.sliders["Wrist"].get())
        try:
            a = float(self.abs_angle.get())
        except ValueError:
            a = 0.0
        try:
            a += float(self.abs_trim.get())  # manual wrist trim, degrees
        except ValueError:
            pass
        lo, hi = JOINTS["Wrist"][:2]
        # claw pitch above horizontal = shoulder + elbow + wrist - 360
        return min(hi, max(lo, a + 360.0 - shoulder - elbow))

    def wrist_mode_changed(self):
        self.sliders["Wrist"].config(state="disabled" if self.abs_on.get() else "normal")
        self.desired = self.slider_values()

    def claw(self, code):
        if self.script_busy():
            return
        if self.arm is None:
            self.status.set("Not connected")
            return
        # resend the arm's present pose so only the claw changes
        self.arm.send_frame(*self.arm.get_joints()[:4], code)
        self.status.set("Claw " + ("open" if code == limearm.CLAW_OPEN else "close"))

    def show_page(self):
        pages = {"joints": self.joint_page, "ik": self.ik_page, "script": self.script_page}
        for p in pages.values():
            p.pack_forget()
        pages[self.page.get()].pack(fill="x")

    # ---- joint page handlers ----
    def slider_moved(self, name):
        self.jvars[name].set(f"{self.sliders[name].get():.1f}")
        self.desired = self.slider_values()

    def entry_changed(self, name):
        try:
            self.sliders[name].set(float(self.jvars[name].get()))
        except ValueError:
            pass
        self.jvars[name].set(f"{self.sliders[name].get():.1f}")
        self.desired = self.slider_values()

    def slider_values(self):
        b, sh, el = (self.sliders[n].get() for n in ("Base", "Shoulder", "Elbow"))
        w = self.wrist_for(sh, el)
        self.jvars["Wrist"].set(f"{w:.1f}")
        return (b, sh, el, w)

    def send(self, angles):
        if self.script_busy():
            return
        self.desired = tuple(angles)
        if self.arm is None:
            self.status.set("Not connected - showing desired pose only")
            return
        self.arm.send_frame(*angles)
        self.status.set("Sent: base %.1f  shoulder %.1f  elbow %.1f  wrist %.1f" % tuple(angles))

    def send_joints(self):
        for n in JOINTS:
            self.entry_changed(n)
        self.send(self.slider_values())

    def send_ik(self):
        try:
            x, y, z = (float(self.xyz[a].get()) for a in "XYZ")
            # display frame == URDF/IK frame now
            sol = ik.inverse_kinematics(x, y, z)
        except Exception as e:  # unreachable target, bad number, ...
            self.status.set(f"IK failed: {e}")
            return
        ang = [sol["base"], sol["shoulder"], sol["elbow"], 180.0]  # IK gives no wrist; level unless absolute
        clamped = []
        for i, (n, (lo, hi, _)) in enumerate(JOINTS.items()):
            if not lo <= ang[i] <= hi:
                clamped.append(n)
            ang[i] = min(hi, max(lo, ang[i]))
            self.sliders[n].set(ang[i])
        if self.abs_on.get():
            ang[3] = self.wrist_for(ang[1], ang[2])
        self.send(ang)
        if clamped:
            self.status.set(self.status.get() + "  (clamped: " + ", ".join(clamped) + ")")

    # ---- script page ----
    def script_busy(self):
        if self.runner and self.runner.running:
            self.status.set("Script running - press Stop on the Script page first")
            return True
        return False

    def browse_script(self):
        path = filedialog.askopenfilename(initialdir=os.path.join(HERE, "scripts"), title="Open arm script",
                                          filetypes=(("Text", "*.txt"), ("All files", "*")))
        if path:
            self.script_path.set(path)

    def script_log_line(self, text):
        self.script_log.insert("end", text)
        self.script_log.see("end")

    def run_script(self):
        if self.runner and self.runner.running:
            self.status.set("Script already running")
            return
        if self.arm is None:
            self.status.set("Not connected - connect before running a script")
            return
        self.script_log.delete(0, "end")
        try:  # parse the whole file before anything moves
            cmds = arm_script.load_script(self.script_path.get())
        except arm_script.ScriptError as e:
            for err in e.errors:
                self.script_log_line(err)
            self.status.set(f"Script not run: {len(e.errors)} syntax error(s)")
            return
        except OSError as e:
            self.status.set(f"Can't open script: {e}")
            return
        limits = {n.lower(): (lo, hi) for n, (lo, hi, _) in JOINTS.items()}
        self.runner = arm_script.ScriptRunner(self.arm, ik.inverse_kinematics, limits,
                                              on_event=self.script_events.put)
        self.runner.start(cmds)
        self.status.set(f"Running script: {len(cmds)} command(s)")

    def stop_script(self):
        if self.runner and self.runner.running:
            self.runner.stop()
            self.status.set("Stopping script...")

    def script_event(self, ev):
        if ev.pose:
            self.desired = ev.pose
        where = f"line {ev.line}: {ev.text}  " if ev.line else ""
        msg = f"{where}[{ev.status}] {ev.message}".rstrip()
        if ev.status != "running" or ev.message:  # "running" is shown in the status bar only
            self.script_log_line(msg)
        self.status.set("Script " + msg)

    # ---- serial ----
    def toggle_connect(self):
        if self.arm:
            if self.runner and self.runner.running:
                self.runner.stop()
                self.runner.join(1)
            self.arm.close()
            self.arm = None
            self.conn_btn.config(text="Connect")
            self.status.set("Disconnected")
            return
        try:
            self.arm = limearm.LimeArmSerial(self.port.get())
        except Exception as e:
            self.status.set(f"Connect failed: {e}")
            return
        self.conn_btn.config(text="Disconnect")
        self.status.set(f"Connected to {self.port.get()}")

    # ---- redraw loop ----
    def update_readouts(self):
        """Joint page: wrist pivot position (forward kinematics) of the desired pose.
        IK page: the desired joint angles (after a solve, the IK solution)."""
        pose = desired_pose(self.desired)
        x, y, z = self.viz.urdf.fk(driver_to_urdf(*pose))["wrist_link"][:3, 3] * 1000.0
        self.fk_text.set(f"Wrist pivot (FK):  X {x:.1f}   Y {y:.1f}   Z {z:.1f}  mm")
        self.angles_text.set("Joint angles:  " + "   ".join(
            f"{n} {a:.1f}°" for n, a in zip(JOINTS, pose)))

    def tick(self):
        while True:  # script progress arrives from the runner thread
            try:
                self.script_event(self.script_events.get_nowait())
            except queue.Empty:
                break
        cur = self.arm.get_joints()[:4] if self.arm else \
            DEFAULTS
        state = (tuple(cur), tuple(self.desired))
        if state != self.last_drawn:
            self.update_readouts()
            self.viz.draw(cur, self.desired)
            self.last_drawn = state
        self.root.after(100, self.tick)


if __name__ == "__main__":
    r = tk.Tk()
    App(r)
    r.mainloop()
