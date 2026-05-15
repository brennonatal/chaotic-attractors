"""Explore discovered 3D chaotic attractors interactively.

Reads seeds from ``discovered.jsonl`` (produced by ``python search.py``),
instantiates each as a ``RandomPolynomial3D``, and animates it with vispy.

Controls:
    ← / →       previous / next attractor
    space       reset the particle cloud for the current attractor
    q / Esc     quit
"""

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from vispy import app, scene

from Attractors import RandomPolynomial3D


NUM_POINTS = 80
TRAIL_LENGTH = 80
DT = 0.01
BOUND = 1e4

DOT_COLOR = (0.45, 0.93, 0.86, 1.0)
TRAIL_COLOR = (0.3, 0.57, 0.82, 0.6)
TRAIL_FADE = (0.1, 0.1, 0.1, 0.0)


def load_discovered(path):
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


class Explorer:
    def __init__(self, entries):
        self.entries = entries
        self.idx = 0
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, size=(1024, 768), bgcolor=(0.02, 0.02, 0.06)
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(up="z", fov=60)
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.line_visual = scene.visuals.Line(
            method="gl", parent=self.view.scene, width=1
        )
        self.canvas.events.key_press.connect(self.on_key)
        self.load_current()

    def load_current(self):
        entry = self.entries[self.idx]
        self.attractor = RandomPolynomial3D(seed=entry["seed"])
        bbox = entry["bbox"]
        centers = [(b[0] + b[1]) / 2 for b in bbox]
        ranges = [b[1] - b[0] for b in bbox]
        self.view.camera.center = centers
        self.view.camera.scale_factor = max(ranges) * 1.4 + 1.0
        ic = self.attractor.initial_state
        self.states = np.tile(ic, (NUM_POINTS, 1)) + np.random.randn(NUM_POINTS, 3) * 0.02
        self.trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(NUM_POINTS)]
        print(
            f"[{self.idx + 1}/{len(self.entries)}] seed={entry['seed']}  "
            f"λ={entry['lyapunov']:.3f}  bbox={bbox}"
        )

    def step(self):
        derivatives = np.apply_along_axis(
            lambda s: self.attractor.next_state(0.0, s, self.attractor.parameters),
            1, self.states,
        )
        new_states = self.states + derivatives * DT
        # Drop any particle that has wandered out of the attractor (rare).
        finite = np.all(np.isfinite(new_states), axis=1)
        in_bound = np.linalg.norm(new_states, axis=1) < BOUND
        mask = finite & in_bound
        new_states[~mask] = self.states[~mask]
        self.states = new_states

    def update(self, ev):
        self.step()
        for i, state in enumerate(self.states):
            self.trails[i].appendleft(state.copy())

        self.scatter.set_data(
            self.states, edge_color=None, face_color=DOT_COLOR, size=6
        )

        line_data = []
        color_data = []
        for trail in self.trails:
            if len(trail) > 1:
                line_data.extend(trail)
                color_data.extend(np.linspace(TRAIL_COLOR, TRAIL_FADE, len(trail)))

        if line_data:
            self.line_visual.set_data(
                pos=np.array(line_data),
                connect="segments",
                color=np.array(color_data),
            )

    def on_key(self, event):
        key = event.key.name if event.key is not None else ""
        if key in ("Right", "Down"):
            self.idx = (self.idx + 1) % len(self.entries)
            self.load_current()
        elif key in ("Left", "Up"):
            self.idx = (self.idx - 1) % len(self.entries)
            self.load_current()
        elif key == "Space":
            self.load_current()
        elif key in ("Q", "Escape"):
            app.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="discovered.jsonl",
                        help="JSONL file produced by search.py")
    args = parser.parse_args()

    entries = load_discovered(args.path)
    print(f"Loaded {len(entries)} discovered attractors.")
    explorer = Explorer(entries)
    timer = app.Timer(connect=explorer.update, interval=0.016)
    timer.start()
    app.run()


if __name__ == "__main__":
    main()
