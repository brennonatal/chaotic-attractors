"""3D Lorenz visualization using vispy.

Multi-particle Euler-step rendering of the Lorenz attractor. Dynamics come
from ``Attractors.LorenzAttractor`` — no inline duplication of the equations.
"""

from collections import deque

import numpy as np
from vispy import app, scene

from Attractors import LorenzAttractor


SCALE = 10.0
NUM_POINTS = 100
TRAIL_LENGTH = 30
DT = 0.01

DOT_COLOR = (0.45, 0.93, 0.86, 1.0)
TRAIL_COLOR = (0.3, 0.57, 0.82, 0.5)
TRAIL_FADE = (0.1, 0.1, 0.1, 0.1)


def main():
    canvas = scene.SceneCanvas(keys="interactive", show=True, size=(1024, 768))
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up="z", fov=60, scale_factor=60)
    view.camera.set_range(x=(-400, 400), y=(-400, 400), z=(-400, 400))

    attractor = LorenzAttractor()
    states = np.random.randn(NUM_POINTS, 3) * 0.01 + np.array([0.1, 0.0, 0.0])
    trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(NUM_POINTS)]

    scatter = scene.visuals.Markers(parent=view.scene)
    line_visual = scene.visuals.Line(method="gl", parent=view.scene, width=1)

    def step(states):
        derivatives = np.apply_along_axis(
            lambda s: attractor.next_state(0, s, attractor.parameters), 1, states
        )
        return states + derivatives * DT

    def update(ev):
        nonlocal states
        states = step(states)
        scaled = states * SCALE
        for i, point in enumerate(scaled):
            trails[i].appendleft(point.copy())

        scatter.set_data(scaled, edge_color=None, face_color=DOT_COLOR, size=10)

        line_data = []
        color_data = []
        for trail in trails:
            if len(trail) > 1:
                line_data.extend(trail)
                color_data.extend(np.linspace(TRAIL_COLOR, TRAIL_FADE, len(trail)))

        if line_data:
            line_visual.set_data(pos=np.array(line_data), connect="segments")
            line_visual.set_data(color=np.array(color_data))

    timer = app.Timer(connect=update, interval=0.016)
    timer.start()
    app.run()


if __name__ == "__main__":
    main()
