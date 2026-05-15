"""Fullscreen live display for discovered 3D chaotic attractors.

This is the gallery-mode sibling of ``animations.explore``: fewer controls,
more polish. It reads seeds from ``discovered.jsonl``, cycles through the best
systems, and renders smooth particle trails with curated dark palettes.

Controls:
    f           toggle fullscreen
    ← / →       previous / next attractor
    space       pause / resume
    r           reset the current attractor
    p           switch color palette
    + / -       increase / decrease dynamic velocity
    h           toggle the minimal HUD
    q / Esc     quit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vispy import app, scene

from Attractors import RandomPolynomial3D



DEFAULT_PATH = "discovered.jsonl"
DEFAULT_POINTS = 240
DEFAULT_TRAIL_LENGTH = 180
DEFAULT_DT = 0.004
DEFAULT_STEPS_PER_FRAME = 4
BOUND = 1e5
MIN_BBOX_RANGE = 1e-6


@dataclass(frozen=True)
class Palette:
    name: str
    background: tuple[float, float, float, float]
    fog: tuple[float, float, float, float]
    head: tuple[float, float, float, float]
    hot: tuple[float, float, float, float]
    cool: tuple[float, float, float, float]
    ghost: tuple[float, float, float, float]


PALETTES = [
    Palette(
        name="aurora",
        background=(0.006, 0.010, 0.025, 1.0),
        fog=(0.020, 0.030, 0.070, 0.18),
        head=(0.78, 1.00, 0.94, 0.98),
        hot=(0.80, 0.32, 1.00, 0.72),
        cool=(0.08, 0.78, 1.00, 0.62),
        ghost=(0.03, 0.04, 0.08, 0.00),
    ),
    Palette(
        name="ember tide",
        background=(0.020, 0.010, 0.006, 1.0),
        fog=(0.075, 0.025, 0.012, 0.16),
        head=(1.00, 0.86, 0.58, 0.98),
        hot=(1.00, 0.28, 0.16, 0.74),
        cool=(0.96, 0.58, 0.18, 0.56),
        ghost=(0.08, 0.03, 0.01, 0.00),
    ),
    Palette(
        name="glacier",
        background=(0.004, 0.012, 0.018, 1.0),
        fog=(0.010, 0.045, 0.065, 0.18),
        head=(0.88, 0.98, 1.00, 0.98),
        hot=(0.45, 0.92, 1.00, 0.70),
        cool=(0.20, 0.46, 1.00, 0.58),
        ghost=(0.00, 0.02, 0.04, 0.00),
    ),
    Palette(
        name="orchid noir",
        background=(0.014, 0.008, 0.028, 1.0),
        fog=(0.045, 0.020, 0.090, 0.17),
        head=(0.98, 0.88, 1.00, 0.98),
        hot=(1.00, 0.38, 0.76, 0.72),
        cool=(0.40, 0.52, 1.00, 0.58),
        ghost=(0.03, 0.01, 0.07, 0.00),
    ),
]


def load_discovered(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        print(f"No {path} found. Run `python search.py` first.", file=sys.stderr)
        sys.exit(1)

    with path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print(f"{path} is empty. Run `python search.py` first.", file=sys.stderr)
        sys.exit(1)

    return entries


def bbox_center_and_scale(bbox: list[list[float]]) -> tuple[np.ndarray, float]:
    mins = np.array([axis[0] for axis in bbox], dtype=np.float64)
    maxs = np.array([axis[1] for axis in bbox], dtype=np.float64)
    ranges = np.maximum(maxs - mins, MIN_BBOX_RANGE)
    center = (mins + maxs) / 2.0
    scale = float(np.max(ranges))
    return center, scale


def lerp_color(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, float, float, float]:
    return tuple((1.0 - t) * x + t * y for x, y in zip(a, b))  # type: ignore[return-value]


def clamp_color(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(max(0.0, min(1.0, component)) for component in color)  # type: ignore[return-value]


class LiveDisplay:
    def __init__(
        self,
        entries: list[dict],
        *,
        points: int = DEFAULT_POINTS,
        trail_length: int = DEFAULT_TRAIL_LENGTH,
        dt: float = DEFAULT_DT,
        steps_per_frame: int = DEFAULT_STEPS_PER_FRAME,
        auto_seconds: float = 45.0,
        velocity_gain: float = 1.0,
        start_index: int = 0,
        fullscreen: bool = True,
        size: tuple[int, int] = (1600, 1000),
        show_hud: bool = True,
    ):
        self.entries = entries
        self.points = points
        self.trail_length = trail_length
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        self.auto_seconds = auto_seconds
        self.velocity_gain = velocity_gain
        self.idx = start_index % len(entries)
        self.palette_idx = 0
        self.paused = False
        self.show_hud = show_hud
        self.last_switch = time.monotonic()
        self.frame = 0

        palette = self.palette
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            fullscreen=fullscreen,
            size=size,
            bgcolor=palette.background,
            title="Chaotic Attractors — Live Display",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(up="z", fov=46, elevation=24, azimuth=35)
        self.view.camera.interactive = False

        self.line_visual = scene.visuals.Line(method="gl", parent=self.view.scene, width=1.35)
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.halo = scene.visuals.Markers(parent=self.view.scene)
        self.title = scene.visuals.Text(
            "",
            parent=self.canvas.scene,
            color=(0.86, 0.92, 1.00, 0.70),
            font_size=12,
            pos=(28, 32),
            anchor_x="left",
            anchor_y="top",
        )

        self.canvas.events.key_press.connect(self.on_key)
        self.canvas.events.resize.connect(self.on_resize)
        self.load_current(reset_camera=True)
        self.on_resize(None)

    @property
    def palette(self) -> Palette:
        return PALETTES[self.palette_idx % len(PALETTES)]

    def load_current(self, *, reset_camera: bool = False) -> None:
        entry = self.entries[self.idx]
        self.attractor = RandomPolynomial3D(seed=entry["seed"])
        self.coeffs = self.attractor.parameters[0]
        self.center, self.scale = bbox_center_and_scale(entry["bbox"])
        self.display_scale = max(self.scale, 1.0)
        self.bounds = self.display_scale * 2.6 + 2.0

        rng = np.random.default_rng(entry["seed"] + self.points)
        initial = np.asarray(self.attractor.initial_state, dtype=np.float64)
        jitter = max(self.display_scale * 0.0009, 0.006)
        self.states = initial + rng.normal(0.0, jitter, size=(self.points, 3))
        self.trails = [deque(maxlen=self.trail_length) for _ in range(self.points)]
        self.trail_speeds = [deque(maxlen=self.trail_length) for _ in range(self.points)]
        self.speed_phase = rng.uniform(0.0, math.tau, size=self.points)
        self.base_speed = rng.uniform(0.72, 1.34, size=self.points)
        self.speed_ema = np.zeros(self.points, dtype=np.float64)
        self.current_speed_norm = np.zeros(self.points, dtype=np.float64)
        # Deques start empty intentionally; they bloom into trails over the first seconds.

        if reset_camera:
            self.view.camera.center = tuple(self.center)
            self.view.camera.scale_factor = self.display_scale * 1.55 + 1.0

        self.last_switch = time.monotonic()
        self.frame = 0
        self.update_hud()
        print(
            f"[{self.idx + 1}/{len(self.entries)}] seed={entry['seed']}  "
            f"λ={entry['lyapunov']:.3f}  palette={self.palette.name}"
        )

    def derivatives(self, states: np.ndarray) -> np.ndarray:
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        basis = np.stack(
            (np.ones_like(x), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z),
            axis=1,
        )
        return basis @ self.coeffs.T

    def dynamic_timestep(self, tick: int) -> np.ndarray:
        """Per-particle timestep multiplier for alive, non-uniform motion."""
        breath = 1.0 + 0.30 * np.sin(tick * 0.016 + self.speed_phase)
        shimmer = 1.0 + 0.10 * np.sin(tick * 0.049 + self.speed_phase * 0.37)
        return np.clip(self.dt * self.velocity_gain * self.base_speed * breath * shimmer, self.dt * 0.38, self.dt * 2.35)

    def rk4_step(self, tick: int) -> None:
        s = self.states
        h = self.dynamic_timestep(tick)[:, None]
        k1 = self.derivatives(s)
        k2 = self.derivatives(s + 0.5 * h * k1)
        k3 = self.derivatives(s + 0.5 * h * k2)
        k4 = self.derivatives(s + h * k3)
        new_states = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        raw_speed = np.log1p(np.linalg.norm(k1, axis=1) / self.display_scale)
        self.speed_ema = self.speed_ema * 0.88 + raw_speed * 0.12
        speed_ceiling = max(float(np.percentile(self.speed_ema, 92)), 1e-6)
        self.current_speed_norm = np.clip(self.speed_ema / speed_ceiling, 0.0, 1.0)

        finite = np.all(np.isfinite(new_states), axis=1)
        near_center = np.linalg.norm(new_states - self.center, axis=1) < self.bounds
        valid = finite & near_center
        if not np.all(valid):
            rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
            count = int(np.count_nonzero(~valid))
            new_states[~valid] = np.asarray(self.attractor.initial_state) + rng.normal(
                0.0, max(self.display_scale * 0.001, 0.008), size=(count, 3)
            )
            for i, ok in enumerate(valid):
                if not ok:
                    self.trails[i].clear()
                    self.trail_speeds[i].clear()
                    self.speed_ema[i] = 0.0
                    self.current_speed_norm[i] = 0.0
        self.states = new_states

    def update(self, event) -> None:
        if not self.paused:
            tick_base = self.frame * self.steps_per_frame
            for step in range(self.steps_per_frame):
                self.rk4_step(tick_base + step)

            for i, state in enumerate(self.states):
                self.trails[i].appendleft(state.copy())
                self.trail_speeds[i].appendleft(float(self.current_speed_norm[i]))

            self.frame += 1
            self.update_camera_motion()
            self.render_particles()

            if self.auto_seconds > 0 and time.monotonic() - self.last_switch > self.auto_seconds:
                self.next_attractor()

    def update_camera_motion(self) -> None:
        t = self.frame / 60.0
        self.view.camera.azimuth = 35 + t * 4.8
        self.view.camera.elevation = 23 + math.sin(t * 0.19) * 7.0
        self.view.camera.roll = math.sin(t * 0.11) * 1.5

    def render_particles(self) -> None:
        palette = self.palette

        line_points: list[np.ndarray] = []
        line_colors: list[tuple[float, float, float, float]] = []
        for particle_idx, trail in enumerate(self.trails):
            if len(trail) < 2:
                continue
            phase = particle_idx / max(1, self.points - 1)
            body = lerp_color(palette.cool, palette.hot, phase)
            for j in range(len(trail) - 1):
                age = j / max(1, self.trail_length - 1)
                speed = self.trail_speeds[particle_idx][j] if j < len(self.trail_speeds[particle_idx]) else 0.0
                body_with_velocity = lerp_color(body, palette.head, 0.16 + 0.58 * speed)
                color = lerp_color(body_with_velocity, palette.ghost, age**0.82)
                alpha = max(0.0, color[3] * (1.0 - age) ** (0.48 + 0.24 * speed))
                color = clamp_color((color[0], color[1], color[2], alpha * (0.62 + 0.55 * speed)))
                line_points.append(trail[j])
                line_points.append(trail[j + 1])
                line_colors.append(color)
                line_colors.append(color)

        if line_points:
            self.line_visual.set_data(
                pos=np.asarray(line_points, dtype=np.float32),
                color=np.asarray(line_colors, dtype=np.float32),
                connect="segments",
            )

        # Heads glow subtly: one large translucent halo plus one small crisp core.
        head_colors = np.empty((self.points, 4), dtype=np.float32)
        halo_colors = np.empty((self.points, 4), dtype=np.float32)
        head_sizes = np.empty(self.points, dtype=np.float32)
        halo_sizes = np.empty(self.points, dtype=np.float32)
        for i in range(self.points):
            phase = i / max(1, self.points - 1)
            speed = float(self.current_speed_norm[i])
            head = lerp_color(palette.head, palette.hot, min(1.0, 0.22 + speed * 0.78 + 0.18 * math.sin(phase * math.tau)))
            head_colors[i] = clamp_color(head)
            halo_colors[i] = clamp_color((head[0], head[1], head[2], 0.08 + 0.20 * speed))
            head_sizes[i] = 3.6 + 5.4 * speed
            halo_sizes[i] = 12.0 + 24.0 * speed

        self.halo.set_data(self.states.astype(np.float32), edge_color=None, face_color=halo_colors, size=halo_sizes)
        self.scatter.set_data(self.states.astype(np.float32), edge_color=None, face_color=head_colors, size=head_sizes)

    def update_hud(self) -> None:
        if not self.show_hud:
            self.title.text = ""
            return
        entry = self.entries[self.idx]
        self.title.text = (
            f"CHAOTIC ATTRACTORS  ·  {self.idx + 1:02d}/{len(self.entries):02d}  "
            f"seed {entry['seed']}  ·  λ {entry['lyapunov']:.3f}  ·  vel {self.velocity_gain:.2f}×  ·  {self.palette.name}"
        )

    def next_attractor(self) -> None:
        self.idx = (self.idx + 1) % len(self.entries)
        self.load_current(reset_camera=True)

    def previous_attractor(self) -> None:
        self.idx = (self.idx - 1) % len(self.entries)
        self.load_current(reset_camera=True)

    def on_key(self, event) -> None:
        key = event.key.name if event.key is not None else ""
        if key in ("Right", "Down"):
            self.next_attractor()
        elif key in ("Left", "Up"):
            self.previous_attractor()
        elif key == "Space":
            self.paused = not self.paused
        elif key == "R":
            self.load_current(reset_camera=True)
        elif key == "P":
            self.palette_idx = (self.palette_idx + 1) % len(PALETTES)
            self.canvas.bgcolor = self.palette.background
            self.update_hud()
        elif key in ("+", "="):
            self.velocity_gain = min(2.4, self.velocity_gain + 0.12)
            self.update_hud()
        elif key in ("-", "_"):
            self.velocity_gain = max(0.35, self.velocity_gain - 0.12)
            self.update_hud()
        elif key == "H":
            self.show_hud = not self.show_hud
            self.update_hud()
        elif key == "F":
            self.canvas.fullscreen = not self.canvas.fullscreen
        elif key in ("Q", "Escape"):
            app.quit()

    def on_resize(self, event) -> None:
        width, _height = self.canvas.size
        self.title.pos = (28, 32)
        self.title.font_size = 11 if width < 1200 else 13


def sorted_entries(entries: list[dict]) -> list[dict]:
    # Put the most alive-looking systems first without discarding the quieter ones.
    return sorted(entries, key=lambda entry: float(entry.get("lyapunov", 0.0)), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DEFAULT_PATH, help="JSONL file produced by search.py")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="number of live particles")
    parser.add_argument("--trail", type=int, default=DEFAULT_TRAIL_LENGTH, help="trail length per particle")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="RK4 integration timestep")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_FRAME, help="integration steps per frame")
    parser.add_argument("--auto-seconds", type=float, default=45.0, help="seconds before auto-advancing; 0 disables")
    parser.add_argument("--velocity-gain", type=float, default=1.0, help="global multiplier for dynamic particle velocity")
    parser.add_argument("--index", type=int, default=0, help="start index after sorting by Lyapunov exponent")
    parser.add_argument("--windowed", action="store_true", help="start windowed instead of fullscreen")
    parser.add_argument("--no-hud", action="store_true", help="hide the minimal overlay text")
    args = parser.parse_args()

    entries = sorted_entries(load_discovered(args.path))
    print(f"Loaded {len(entries)} discovered attractors for live display.")
    display = LiveDisplay(
        entries,
        points=args.points,
        trail_length=args.trail,
        dt=args.dt,
        steps_per_frame=args.steps,
        auto_seconds=args.auto_seconds,
        velocity_gain=args.velocity_gain,
        start_index=args.index,
        fullscreen=not args.windowed,
        show_hud=not args.no_hud,
    )
    timer = app.Timer(connect=display.update, interval=0.0, start=True)
    app.run()


if __name__ == "__main__":
    main()
