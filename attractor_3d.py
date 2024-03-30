import numpy as np
from vispy import scene, app
from collections import deque

# Initialize the scene
canvas = scene.SceneCanvas(keys="interactive", show=True, size=(1024, 768))
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up="z", fov=60, scale_factor=60)
view.camera.set_range(x=(-40, 40), y=(-40, 40), z=(-40, 40))

# Attractor parameters
sigma, rho, beta = 10, 28, 8 / 3
dt = 0.01
num_points = 100
scale = 1
trail_length = 50
dot_color = (0.45, 0.93, 0.86)
trail_color_start = (0.3, 0.57, 0.82)
trail_color_end = (0.1, 0.1, 0.1)

# Initialize states and trails
states = np.random.randn(num_points, 3) * 0.01 + np.array([0.1, 0, 0])
trails = [deque(maxlen=trail_length) for _ in range(num_points)]

# Create visuals
scatter = scene.visuals.Markers(parent=view.scene)
line_visual = scene.visuals.Line(
    color=trail_color_start, method="gl", parent=view.scene
)


def step(states):
    """Evolve the states of the attractor"""
    dx = sigma * (states[:, 1] - states[:, 0]) * dt
    dy = (states[:, 0] * (rho - states[:, 2]) - states[:, 1]) * dt
    dz = (states[:, 0] * states[:, 1] - beta * states[:, 2]) * dt
    return states + np.array([dx, dy, dz]).T


def update(ev):
    global states
    states = step(states)

    # Update trails
    for i, state in enumerate(states):
        trails[i].append(state)

    # Update scatter plot
    scatter.set_data(states * scale, size=8, edge_color=None, face_color=dot_color)

    # Prepare line data
    lines = []
    colors = []
    for trail in trails:
        if len(trail) > 1:
            lines.extend(trail)
            # Create a color gradient for the trail
            n = len(trail)
            color_gradient = np.linspace(trail_color_end, trail_color_start, n)
            colors.extend(color_gradient)

    if lines:
        line_visual.set_data(
            np.array(lines) * scale, connect="segments", color=np.array(colors)
        )


timer = app.Timer(connect=update, interval=0.016)
timer.start()

if __name__ == "__main__":
    app.run()
