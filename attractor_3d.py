import numpy as np
from vispy import scene, app
from collections import deque

# Initialize the scene
canvas = scene.SceneCanvas(keys="interactive", show=True, size=(1024, 768))
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up="z", fov=60, scale_factor=60)
view.camera.set_range(x=(-400, 400), y=(-400, 400), z=(-400, 400))

# Attractor parameters and visualization settings
sigma, rho, beta = 10, 28, 8 / 3
dt = 0.01
num_points = 100
scale = 10
trail_length = 30

# Colors - RGBA format where each component is between 0 and 1
dot_color = (0.45, 0.93, 0.86, 1.0)  # Bright teal for the dots
trail_color = (0.3, 0.57, 0.82, 0.5)  # Slightly transparent blue for the trail start
trail_fade_to_color = (0.1, 0.1, 0.1, 0.1)  # Fade to almost transparent

# Initialize states with slight variations
states = np.random.randn(num_points, 3) * 0.01 + np.array([0.1, 0, 0])
trails = [deque(maxlen=trail_length) for _ in range(num_points)]

# Create visuals
scatter = scene.visuals.Markers(parent=view.scene)
line_visual = scene.visuals.Line(method="gl", parent=view.scene, width=1)


def step(states):
    """Perform one step of the attractor's evolution for all points."""
    dx = sigma * (states[:, 1] - states[:, 0]) * dt
    dy = (states[:, 0] * (rho - states[:, 2]) - states[:, 1]) * dt
    dz = (states[:, 0] * states[:, 1] - beta * states[:, 2]) * dt
    return states + np.array([dx, dy, dz]).T


def update(ev):
    global states
    states = step(states)
    for i, state in enumerate(states):
        trails[i].appendleft(state * scale)  # Prepend to reverse the trail direction

    # Update scatter plot with current states
    scatter.set_data(states * scale, edge_color=None, face_color=dot_color, size=10)

    # Prepare data for trails
    line_data = []
    color_data = []
    for trail in trails:
        if len(trail) > 1:
            line_data.extend(trail)
            # Correctly apply fading effect for trails
            color_gradient = np.linspace(trail_color, trail_fade_to_color, len(trail))
            color_data.extend(color_gradient)

    if line_data:
        line_visual.set_data(pos=np.array(line_data), connect="segments")
        line_visual.set_data(color=np.array(color_data))


# Set up a timer to call the update function periodically
timer = app.Timer(connect=update, interval=0.016)
timer.start()

if __name__ == "__main__":
    app.run()
