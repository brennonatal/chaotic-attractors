"""Fullscreen live display for discovered 3D chaotic attractors.

This is the gallery-mode sibling of ``animations.explore``: fewer controls,
more polish. It reads seeds from ``discovered.jsonl``, cycles through the best
systems, and renders smooth particle trails with a modern, Pantone-inspired
color language.

Controls:
    f           toggle fullscreen
    ← / →       previous / next attractor
    space       pause / resume
    r           reset the current attractor
    p           switch color palette
    + / -       increase / decrease flow speed
    [ / ]       decrease / increase inter-particle gravity
    , / .       decrease / increase proximity speed boost
    h           toggle the minimal HUD
    q / Esc     quit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vispy import app, scene

from Attractors import RandomPolynomial3D


DEFAULT_PATH = "discovered.jsonl"
BOUND = 1e5
MIN_BBOX_RANGE = 1e-6
QUALITY_PRESETS = {
    # points, trail, dt, steps, stars, glow_width
    # Deliberately lower particle counts: spend the budget on richer particles,
    # longer trails, and cleaner motion instead of noisy quantity.
    "balanced": (21, 320, 0.0030, 5, 45, 4.4),
    "cinema": (33, 420, 0.0026, 6, 80, 6.0),
    "ultra": (55, 520, 0.0022, 7, 120, 7.2),
}
DEFAULT_QUALITY = "cinema"


@dataclass(frozen=True)
class Palette:
    name: str
    background: tuple[float, float, float, float]
    fog: tuple[float, float, float, float]
    head: tuple[float, float, float, float]
    hot: tuple[float, float, float, float]
    cool: tuple[float, float, float, float]
    ghost: tuple[float, float, float, float]
    star: tuple[float, float, float, float]


def hex_color(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a Pantone-style hex reference into VisPy RGBA floats."""
    value = value.removeprefix("#")
    return (int(value[0:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:6], 16) / 255, alpha)


# Pantone-inspired pairings. One quiet dark ground, one warm accent, one cool
# counterpoint, and one bright head. This avoids cheap rainbow visuals.
PALETTES = [
    Palette("mocha periwinkle", hex_color("0B0A10"), hex_color("2A1E22", 0.14), hex_color("F7E1D2", 0.98), hex_color("A47864", 0.76), hex_color("6667AB", 0.66), hex_color("0B0A10", 0.00), hex_color("F2D8C2", 0.22)),
    Palette("peach ink", hex_color("080A12"), hex_color("281B25", 0.15), hex_color("FFE4D6", 0.98), hex_color("FFBE98", 0.78), hex_color("5B7C99", 0.64), hex_color("080A12", 0.00), hex_color("FFD6BF", 0.20)),
    Palette("viva cyan", hex_color("0A0710"), hex_color("2B0E22", 0.16), hex_color("FDE7F0", 0.98), hex_color("BB2649", 0.78), hex_color("00A6A6", 0.64), hex_color("0A0710", 0.00), hex_color("F4A3B7", 0.20)),
    Palette("serenity coral", hex_color("071018"), hex_color("10253A", 0.15), hex_color("F4FBFF", 0.98), hex_color("F7786B", 0.76), hex_color("92A8D1", 0.66), hex_color("071018", 0.00), hex_color("C9D8F2", 0.19)),
    Palette("greenery ultraviolet", hex_color("070C09"), hex_color("102618", 0.14), hex_color("F4FFE8", 0.98), hex_color("88B04B", 0.76), hex_color("5F4B8B", 0.66), hex_color("070C09", 0.00), hex_color("D9F2B4", 0.18)),
]


def load_discovered(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        print(f"No {path} found. Run `uv run python search.py` first.", file=sys.stderr)
        sys.exit(1)

    with path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print(f"{path} is empty. Run `uv run python search.py` first.", file=sys.stderr)
        sys.exit(1)

    return entries


def bbox_center_and_scale(bbox: list[list[float]]) -> tuple[np.ndarray, float]:
    mins = np.array([axis[0] for axis in bbox], dtype=np.float64)
    maxs = np.array([axis[1] for axis in bbox], dtype=np.float64)
    ranges = np.maximum(maxs - mins, MIN_BBOX_RANGE)
    center = (mins + maxs) / 2.0
    scale = float(np.max(ranges))
    return center, scale


def palette_array(color: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(color, dtype=np.float32)


def clamp_color(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(max(0.0, min(1.0, component)) for component in color)  # type: ignore[return-value]


def segment_indices(visible_len: int) -> np.ndarray:
    """Dense head + progressively decimated tail: smooth where the eye tracks."""
    max_start = max(1, visible_len - 1)
    head = np.arange(0, min(92, max_start), 1)
    body = np.arange(min(92, max_start), min(190, max_start), 2)
    tail = np.arange(min(190, max_start), max_start, 4)
    return np.concatenate((head, body, tail)).astype(np.int64)


class LiveDisplay:
    def __init__(
        self,
        entries: list[dict],
        *,
        points: int,
        trail_length: int,
        dt: float,
        steps_per_frame: int,
        star_count: int,
        glow_width: float,
        quality: str,
        auto_seconds: float = 45.0,
        velocity_gain: float = 1.28,
        gravity_strength: float = 0.25,
        proximity_gain: float = 0.85,
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
        self.star_count = star_count
        self.quality = quality
        self.auto_seconds = auto_seconds
        self.velocity_gain = velocity_gain
        self.gravity_strength = gravity_strength
        self.proximity_gain = proximity_gain
        self.idx = start_index % len(entries)
        self.palette_idx = 0
        self.paused = False
        self.show_hud = show_hud
        self.last_switch = time.monotonic()
        self.frame = 0
        self.sim_tick = 0
        self.warmup_len = 1

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
        self.view.camera = scene.TurntableCamera(up="z", fov=42, elevation=24, azimuth=35)
        self.view.camera.interactive = False

        # Glow is a separate low-alpha line pass. Vectorized buffers keep this cheap.
        self.glow_line_visual = scene.visuals.Line(method="gl", parent=self.view.scene, width=glow_width)
        self.line_visual = scene.visuals.Line(method="gl", parent=self.view.scene, width=1.2)
        self.stars = scene.visuals.Markers(parent=self.view.scene)
        self.outer_halo = scene.visuals.Markers(parent=self.view.scene)
        self.inner_halo = scene.visuals.Markers(parent=self.view.scene)
        self.spark = scene.visuals.Markers(parent=self.view.scene)
        self.core = scene.visuals.Markers(parent=self.view.scene)
        self.title = scene.visuals.Text(
            "",
            parent=self.canvas.scene,
            color=(0.86, 0.90, 0.96, 0.64),
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
        # Gravity is meant to be a visible nudge, not a dominating force.
        # Strong chaos (large Lyapunov exponent) keeps the cloud spread out, so
        # it can carry a heavier gravitational pull without collapsing onto a
        # single point. Quieter attractors get gentler gravity. Softening
        # bounds the peak per-pair force.
        lyapunov = float(entry.get("lyapunov", 0.5))
        self.gravity_adapt = max(0.0, min(1.0, (lyapunov / 1.5) ** 2))
        self.gravity_coupling = self.gravity_strength * self.gravity_adapt * self.display_scale ** 2 * 0.18
        self.gravity_softening_sq = max(self.display_scale * 0.18, 0.18) ** 2
        self.proximity_radius = max(self.display_scale * 0.30, 0.30)

        rng = np.random.default_rng(entry["seed"] + self.points)
        self.initial_origin = np.asarray(self.attractor.initial_state, dtype=np.float64)
        self.jitter = max(self.display_scale * 0.0012, 0.008)
        self.states = self.initial_origin + rng.normal(0.0, self.jitter, size=(self.points, 3))
        self.trails = np.repeat(self.states[:, None, :], self.trail_length, axis=1).astype(np.float32)
        self.trail_speeds = np.zeros((self.points, self.trail_length), dtype=np.float32)
        self.trail_curvature = np.zeros((self.points, self.trail_length), dtype=np.float32)
        self.prev_velocity = np.zeros((self.points, 3), dtype=np.float64)
        self.speed_phase = rng.uniform(0.0, math.tau, size=self.points)
        self.base_speed = rng.uniform(0.84, 1.52, size=self.points)
        self.speed_ema = np.zeros(self.points, dtype=np.float64)
        self.current_speed_norm = np.zeros(self.points, dtype=np.float64)
        self.current_curvature_norm = np.zeros(self.points, dtype=np.float64)
        self.star_positions = self.center + rng.uniform(-1.0, 1.0, size=(self.star_count, 3)) * self.display_scale * 1.65
        self.star_sizes = rng.uniform(0.9, 2.8, size=self.star_count).astype(np.float32)

        self.precompute_camera_framing()
        self.apply_camera_framing()
        self.render_starfield()

        self.last_switch = time.monotonic()
        self.frame = 0
        self.sim_tick = 0
        self.warmup_len = 1
        self.update_hud()
        print(
            f"[{self.idx + 1}/{len(self.entries)}] seed={entry['seed']}  "
            f"λ={entry['lyapunov']:.3f}  palette={self.palette.name}  quality={self.quality}"
        )

    def render_starfield(self) -> None:
        palette = self.palette
        colors = np.empty((self.star_count, 4), dtype=np.float32)
        for i in range(self.star_count):
            twinkle = 0.62 + 0.38 * math.sin(i * 12.9898 + self.frame * 0.005)
            colors[i] = clamp_color((palette.star[0], palette.star[1], palette.star[2], palette.star[3] * twinkle))
        self.stars.set_data(self.star_positions.astype(np.float32), edge_color=None, face_color=colors, size=self.star_sizes)

    def derivatives(self, states: np.ndarray) -> np.ndarray:
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        out = np.empty_like(states)
        for axis, coeff in enumerate(self.coeffs):
            out[:, axis] = (
                coeff[0]
                + coeff[1] * x
                + coeff[2] * y
                + coeff[3] * z
                + coeff[4] * xx
                + coeff[5] * yy
                + coeff[6] * zz
                + coeff[7] * xy
                + coeff[8] * xz
                + coeff[9] * yz
            )
        return out

    def dynamic_timestep(self, tick: int) -> np.ndarray:
        """Per-particle timestep multiplier for alive, non-uniform motion."""
        breath = 1.0 + 0.30 * np.sin(tick * 0.007 + self.speed_phase)
        pulse = 1.0 + 0.11 * np.sin(tick * 0.019 + self.speed_phase * 0.43)
        micro = 1.0 + 0.035 * np.sin(tick * 0.043 + self.speed_phase * 1.71)
        return np.clip(self.dt * self.velocity_gain * self.base_speed * breath * pulse * micro, self.dt * 0.45, self.dt * 2.25)

    def gravity_field(self, states: np.ndarray) -> np.ndarray:
        """Pairwise softened inverse-square attraction between particles."""
        if self.gravity_coupling == 0.0:
            return np.zeros_like(states)
        diff = states[None, :, :] - states[:, None, :]                          # (N, N, 3)
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff) + self.gravity_softening_sq
        inv_r3 = dist_sq ** -1.5
        np.fill_diagonal(inv_r3, 0.0)
        return self.gravity_coupling * np.einsum("ij,ijk->ik", inv_r3, diff)

    def proximity_factor(self, states: np.ndarray) -> np.ndarray:
        """Per-particle timestep multiplier: closer to a neighbour → faster."""
        if self.proximity_gain == 0.0:
            return np.ones(len(states), dtype=np.float64)
        diff = states[None, :, :] - states[:, None, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff) + 1e-9)
        np.fill_diagonal(dist, np.inf)
        nearest = np.min(dist, axis=1)
        closeness = np.clip(1.0 - nearest / self.proximity_radius, 0.0, 1.0)
        return 1.0 + self.proximity_gain * closeness * closeness

    def field(self, states: np.ndarray) -> np.ndarray:
        return self.derivatives(states) + self.gravity_field(states)

    def integrate_states(self, states: np.ndarray, tick: int) -> np.ndarray:
        h = (self.dynamic_timestep(tick) * self.proximity_factor(states))[:, None]
        k1 = self.field(states)
        k2 = self.field(states + 0.5 * h * k1)
        k3 = self.field(states + 0.5 * h * k2)
        k4 = self.field(states + h * k3)
        return states + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rk4_step(self, tick: int) -> None:
        s = self.states
        new_states = self.integrate_states(s, tick)

        velocity = new_states - s
        raw_speed = np.log1p(np.linalg.norm(velocity, axis=1) / max(self.display_scale, 1e-6))
        self.speed_ema = self.speed_ema * 0.84 + raw_speed * 0.16
        speed_ceiling = max(float(np.percentile(self.speed_ema, 90)), 1e-6)
        self.current_speed_norm = np.clip(self.speed_ema / speed_ceiling, 0.0, 1.0)

        cross = np.linalg.norm(np.cross(self.prev_velocity, velocity), axis=1)
        denom = np.linalg.norm(self.prev_velocity, axis=1) * np.linalg.norm(velocity, axis=1) + 1e-9
        self.current_curvature_norm = np.clip(cross / denom, 0.0, 1.0)
        self.prev_velocity = velocity

        finite = np.all(np.isfinite(new_states), axis=1)
        near_center = np.linalg.norm(new_states - self.center, axis=1) < self.bounds
        valid = finite & near_center
        if not np.all(valid):
            rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
            count = int(np.count_nonzero(~valid))
            new_states[~valid] = np.asarray(self.attractor.initial_state) + rng.normal(0.0, max(self.display_scale * 0.0012, 0.008), size=(count, 3))
            self.prev_velocity[~valid] = 0.0
            self.speed_ema[~valid] = 0.0
            self.current_speed_norm[~valid] = 0.0
            self.current_curvature_norm[~valid] = 0.0
        self.states = new_states

    def append_trail_samples(self) -> None:
        # Vectorized trail buffer. This replaced Python deques so we can spend the
        # frame budget on actual visual quality instead of object churn.
        self.trails[:, 1:] = self.trails[:, :-1]
        self.trail_speeds[:, 1:] = self.trail_speeds[:, :-1]
        self.trail_curvature[:, 1:] = self.trail_curvature[:, :-1]
        self.trails[:, 0] = self.states.astype(np.float32)
        self.trail_speeds[:, 0] = self.current_speed_norm.astype(np.float32)
        self.trail_curvature[:, 0] = self.current_curvature_norm.astype(np.float32)
        self.warmup_len = min(self.trail_length, self.warmup_len + 1)

    def update(self, event) -> None:
        if self.paused:
            return

        for _ in range(self.steps_per_frame):
            self.rk4_step(self.sim_tick)
            self.append_trail_samples()
            self.sim_tick += 1

        self.frame += 1
        self.update_camera_motion()
        if self.frame % 10 == 0:
            self.render_starfield()
        self.render_particles()

        if self.auto_seconds > 0 and time.monotonic() - self.last_switch > self.auto_seconds:
            self.next_attractor()

    def precompute_camera_framing(self) -> None:
        """Simulate the whole gallery interval before rendering and choose one static fit.

        Live zoom correction looks nervous on a fullscreen display. Instead, use
        the exact same integrator/timestep schedule ahead of time, measure the
        bounds of every finite point that will be produced during the current
        attractor interval, then keep the camera center and scale fixed while it
        renders.
        """
        preview_frames = int((self.auto_seconds if self.auto_seconds > 0 else 45.0) * 60)
        preview_steps = max(self.trail_length, preview_frames * self.steps_per_frame)
        states = self.states.copy()
        mins = np.min(states, axis=0)
        maxs = np.max(states, axis=0)

        reset_rng = np.random.default_rng(self.entries[self.idx]["seed"] + 7919)
        for tick in range(preview_steps):
            new_states = self.integrate_states(states, tick)
            finite = np.all(np.isfinite(new_states), axis=1)
            near_center = np.linalg.norm(new_states - self.center, axis=1) < self.bounds
            valid = finite & near_center
            if np.any(valid):
                valid_states = new_states[valid]
                mins = np.minimum(mins, np.min(valid_states, axis=0))
                maxs = np.maximum(maxs, np.max(valid_states, axis=0))
            if not np.all(valid):
                count = int(np.count_nonzero(~valid))
                new_states[~valid] = self.initial_origin + reset_rng.normal(0.0, self.jitter, size=(count, 3))
            states = new_states

        ranges = np.maximum(maxs - mins, MIN_BBOX_RANGE)
        self.framing_center = (mins + maxs) / 2.0
        max_axis = float(np.max(ranges))
        diagonal_radius = float(np.linalg.norm(ranges) * 0.5)
        self.framing_scale = max(
            max_axis * 1.88 + 1.0,
            diagonal_radius * 1.38 + 1.0,
            self.display_scale * 1.45 + 1.0,
        )
        self.view.camera.center = tuple(self.framing_center)
        self.view.camera.scale_factor = self.framing_scale

    def apply_camera_framing(self) -> None:
        self.view.camera.center = tuple(self.framing_center)
        self.view.camera.scale_factor = self.framing_scale

    def update_camera_motion(self) -> None:
        t = self.frame / 60.0
        self.view.camera.azimuth = 35 + t * 5.0
        self.view.camera.elevation = 24 + math.sin(t * 0.15) * 7.0
        self.view.camera.roll = math.sin(t * 0.09) * 1.8
        # Keep FOV fixed. FOV breathing reads as zoom, and framing is now
        # precomputed exactly before the first rendered frame.
        self.view.camera.fov = 40

    def render_particles(self) -> None:
        palette = self.palette
        idx = segment_indices(self.warmup_len)
        if len(idx) == 0:
            return

        starts = self.trails[:, idx]
        ends = self.trails[:, idx + 1]
        pos = np.empty((self.points, len(idx), 2, 3), dtype=np.float32)
        pos[:, :, 0] = starts
        pos[:, :, 1] = ends
        pos = pos.reshape(-1, 3)

        phase = np.linspace(0.0, 1.0, self.points, dtype=np.float32)[:, None, None]
        age = (idx.astype(np.float32) / max(1, self.trail_length - 1))[None, :, None]
        speed = self.trail_speeds[:, idx][:, :, None]
        curvature = self.trail_curvature[:, idx][:, :, None]
        depth = 0.5 + 0.5 * np.tanh((starts[:, :, 2:3] - self.center[2]) / max(self.display_scale * 0.42, 1e-6))

        cool = palette_array(palette.cool)[None, None, :]
        hot = palette_array(palette.hot)[None, None, :]
        head = palette_array(palette.head)[None, None, :]
        ghost = palette_array(palette.ghost)[None, None, :]

        body = cool * (1.0 - phase) + hot * phase
        excitation = np.clip(0.10 + 0.54 * speed + 0.30 * curvature + 0.14 * depth, 0.0, 1.0)
        color = body * (1.0 - excitation) + head * excitation
        ghost_mix = np.power(age, 0.95)
        color = color * (1.0 - ghost_mix) + ghost * ghost_mix
        # Exponential fade reads as a true "fading trail" — old segments
        # disappear instead of lingering as low-alpha smudges. Fast particles
        # keep slightly longer tails, so velocity becomes legible in the trail.
        fade = np.exp(-age * (2.6 - 0.95 * speed))
        color[:, :, 3:4] *= fade * (0.46 + 0.54 * speed) * (0.68 + 0.35 * depth)
        color = np.clip(color, 0.0, 1.0).astype(np.float32)
        colors = np.repeat(color[:, :, None, :], 2, axis=2).reshape(-1, 4)

        glow = colors.copy()
        glow[:, 3] *= 0.24
        self.glow_line_visual.set_data(pos=pos, color=glow, connect="segments")
        self.line_visual.set_data(pos=pos, color=colors, connect="segments")

        head_speed = self.current_speed_norm.astype(np.float32)
        head_curve = self.current_curvature_norm.astype(np.float32)
        head_depth = 0.5 + 0.5 * np.tanh((self.states[:, 2] - self.center[2]) / max(self.display_scale * 0.42, 1e-6))
        phase_1d = np.linspace(0.0, 1.0, self.points, dtype=np.float32)[:, None]
        body_head = palette_array(palette.cool) * (1.0 - phase_1d) + palette_array(palette.hot) * phase_1d
        excite = np.clip(0.22 + 0.58 * head_speed[:, None] + 0.26 * head_curve[:, None], 0.0, 1.0)
        core_colors = body_head * (1.0 - excite) + palette_array(palette.head) * excite
        core_colors[:, 3] = 0.88 + 0.12 * head_speed

        spark_colors = palette_array(palette.head) * 0.72 + core_colors * 0.28
        spark_colors[:, 3] = np.clip(0.34 + 0.46 * head_speed + 0.24 * head_curve, 0.0, 0.92)

        inner_halo_colors = core_colors.copy()
        inner_halo_colors[:, 3] = np.clip(0.11 + 0.23 * head_speed + 0.16 * head_curve, 0.0, 0.55)
        outer_halo_colors = core_colors.copy()
        outer_halo_colors[:, 3] = np.clip(0.025 + 0.11 * head_speed + 0.08 * head_curve, 0.0, 0.28)

        # Fewer particles, but each reads as a luminous body: large soft aura,
        # tight inner glow, bright spark, then a crisp core with a subtle rim.
        core_sizes = (6.0 + 9.5 * head_speed + 3.8 * head_curve + 1.8 * head_depth).astype(np.float32)
        spark_sizes = (2.5 + 3.7 * head_speed + 2.4 * head_curve).astype(np.float32)
        inner_halo_sizes = (21.0 + 43.0 * head_speed + 26.0 * head_curve + 7.0 * head_depth).astype(np.float32)
        outer_halo_sizes = (48.0 + 92.0 * head_speed + 44.0 * head_curve + 14.0 * head_depth).astype(np.float32)
        rim_colors = np.clip(core_colors * np.array([1.05, 1.05, 1.05, 0.44], dtype=np.float32), 0.0, 1.0)

        states = self.states.astype(np.float32)
        self.outer_halo.set_data(states, edge_color=None, face_color=np.clip(outer_halo_colors, 0.0, 1.0), size=outer_halo_sizes, symbol="disc")
        self.inner_halo.set_data(states, edge_color=None, face_color=np.clip(inner_halo_colors, 0.0, 1.0), size=inner_halo_sizes, symbol="disc")
        self.spark.set_data(states, edge_color=None, face_color=np.clip(spark_colors, 0.0, 1.0), size=spark_sizes, symbol="disc")
        self.core.set_data(states, edge_color=rim_colors, face_color=np.clip(core_colors, 0.0, 1.0), size=core_sizes, symbol="disc")

    def update_hud(self) -> None:
        if not self.show_hud:
            self.title.text = ""
            return
        entry = self.entries[self.idx]
        self.title.text = (
            f"CHAOTIC ATTRACTORS  ·  {self.idx + 1:02d}/{len(self.entries):02d}  "
            f"seed {entry['seed']}  ·  λ {entry['lyapunov']:.3f}  ·  flow {self.velocity_gain:.2f}×  "
            f"·  g {self.gravity_strength:.2f}  ·  prox {self.proximity_gain:.2f}  "
            f"·  {self.quality}  ·  {self.palette.name}"
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
            self.render_starfield()
            self.update_hud()
        elif key in ("+", "="):
            self.velocity_gain = min(3.0, self.velocity_gain + 0.12)
            self.precompute_camera_framing()
            self.apply_camera_framing()
            self.update_hud()
        elif key in ("-", "_"):
            self.velocity_gain = max(0.30, self.velocity_gain - 0.12)
            self.precompute_camera_framing()
            self.apply_camera_framing()
            self.update_hud()
        elif key in ("]", "}"):
            self.gravity_strength = min(3.0, self.gravity_strength + 0.1)
            self.gravity_coupling = self.gravity_strength * self.gravity_adapt * self.display_scale ** 2 * 0.18
            self.precompute_camera_framing()
            self.apply_camera_framing()
            self.update_hud()
        elif key in ("[", "{"):
            self.gravity_strength = max(0.0, self.gravity_strength - 0.1)
            self.gravity_coupling = self.gravity_strength * self.gravity_adapt * self.display_scale ** 2 * 0.18
            self.precompute_camera_framing()
            self.apply_camera_framing()
            self.update_hud()
        elif key in (".", ">"):
            self.proximity_gain = min(3.0, self.proximity_gain + 0.1)
            self.precompute_camera_framing()
            self.apply_camera_framing()
            self.update_hud()
        elif key in (",", "<"):
            self.proximity_gain = max(0.0, self.proximity_gain - 0.1)
            self.precompute_camera_framing()
            self.apply_camera_framing()
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


def quality_defaults(name: str) -> tuple[int, int, float, int, int, float]:
    if name not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality {name!r}; choose one of {', '.join(QUALITY_PRESETS)}")
    return QUALITY_PRESETS[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DEFAULT_PATH, help="JSONL file produced by search.py")
    parser.add_argument("--quality", choices=tuple(QUALITY_PRESETS), default=DEFAULT_QUALITY, help="render quality preset")
    parser.add_argument("--points", type=int, default=None, help="number of live particles")
    parser.add_argument("--trail", type=int, default=None, help="trail length per particle")
    parser.add_argument("--dt", type=float, default=None, help="RK4 integration timestep")
    parser.add_argument("--steps", type=int, default=None, help="integration steps per frame")
    parser.add_argument("--auto-seconds", type=float, default=45.0, help="seconds before auto-advancing; 0 disables")
    parser.add_argument("--velocity-gain", type=float, default=1.28, help="global multiplier for dynamic particle flow")
    parser.add_argument("--gravity", type=float, default=0.25, help="inter-particle gravity strength (0 disables)")
    parser.add_argument("--proximity", type=float, default=0.85, help="speed boost when particles get close (0 disables)")
    parser.add_argument("--index", type=int, default=0, help="start index after sorting by Lyapunov exponent")
    parser.add_argument("--windowed", action="store_true", help="start windowed instead of fullscreen")
    parser.add_argument("--no-hud", action="store_true", help="hide the minimal overlay text")
    args = parser.parse_args()

    preset_points, preset_trail, preset_dt, preset_steps, preset_stars, preset_glow = quality_defaults(args.quality)
    entries = sorted_entries(load_discovered(args.path))
    print(f"Loaded {len(entries)} discovered attractors for live display.")
    display = LiveDisplay(
        entries,
        points=args.points or preset_points,
        trail_length=args.trail or preset_trail,
        dt=args.dt or preset_dt,
        steps_per_frame=args.steps or preset_steps,
        star_count=preset_stars,
        glow_width=preset_glow,
        quality=args.quality,
        auto_seconds=args.auto_seconds,
        velocity_gain=args.velocity_gain,
        gravity_strength=args.gravity,
        proximity_gain=args.proximity,
        start_index=args.index,
        fullscreen=not args.windowed,
        show_hud=not args.no_hud,
    )
    timer = app.Timer(connect=display.update, interval=0.0, start=True)
    app.run()


if __name__ == "__main__":
    main()
