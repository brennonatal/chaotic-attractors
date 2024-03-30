import pygame
import numpy as np
import sys
from collections import deque


class Attractor:
    def __init__(
        self,
        sigma=10,
        rho=28,
        beta=8 / 3,
        screen_width=1024,
        screen_height=768,
        scale=10,
        depth=500,
        dot_color=(115, 238, 220),
        trail_color=(77, 145, 209),
        num_points=100,
    ):
        """
        Initializes an attractor model with given parameters and prepares the drawing environment.

        Parameters:
        - sigma, rho, beta: Parameters that affect the attractor's behavior. Changing these can lead to different chaotic effects.
        - screen_width, screen_height: Dimensions of the Pygame window in pixels.
        - scale: A scaling factor for the attractor's coordinates, to adjust its size on the screen.
        - depth: A parameter to adjust the depth effect in the visualization, influencing the brightness of the lines.
        - dot_color: The RGB color for the current point of the attractor as a tuple.
        - trail_color: The RGB color for the trail of the attractor as a tuple.
        - num_points: The number of points to initialize and simulate.

        The state is initialized to a small value near the origin, and an empty list is prepared to store the points of the attractor.
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = scale
        self.depth = depth
        self.dt = 0.01
        self.state = np.array([0.1, 0, 0])
        self.num_points = num_points
        self.states = [
            np.array([0.1 + i * 0.001, 0 + i * 0.001, 0 + i * 0.001])
            for i in range(num_points)
        ]
        self.points = [[] for _ in range(num_points)]
        self.dot_color = dot_color
        self.trail_color = trail_color
        self.max_trail_length = 50
        self.points = [deque(maxlen=self.max_trail_length) for _ in range(num_points)]

    def step(self):
        for j in range(self.num_points):
            dx = self.sigma * (self.states[j][1] - self.states[j][0]) * self.dt
            dy = (
                self.states[j][0] * (self.rho - self.states[j][2]) - self.states[j][1]
            ) * self.dt
            dz = (
                self.states[j][0] * self.states[j][1] - self.beta * self.states[j][2]
            ) * self.dt
            self.states[j] += np.array([dx, dy, dz])

            x, y, z = self.states[j]
            x = (x * self.scale) + (self.screen_width / 2)
            y = (y * self.scale) + (self.screen_height / 2)
            z *= self.scale
            self.points[j].append((x, y, z))

    def draw(self, screen):
        base_point_size = 3
        base_trail_size = 1
        max_additional_size = 2

        for point_set in self.points:
            if len(point_set) > 1:
                z_values = np.array([p[2] for p in point_set])
                min_z = z_values.min()
                max_z = z_values.max()
                z_range = max(max_z - min_z, 1)
                fade_factors = np.linspace(0, 1, len(point_set))

                for i in range(1, len(point_set)):
                    z = z_values[i]
                    brightness = max(
                        min(255, int(255 - (z + self.depth) / self.depth * 100)), 0
                    )
                    fade_factor = fade_factors[i]
                    faded_color = [
                        int(c * fade_factor * brightness / 255)
                        for c in self.trail_color
                    ]
                    size_factor = (z - min_z) / z_range
                    trail_size = int(
                        base_trail_size + size_factor * max_additional_size
                    )

                    pygame.draw.line(
                        screen,
                        faded_color,
                        point_set[i - 1][:2],
                        point_set[i][:2],
                        trail_size,
                    )

        for point_set in self.points:
            if point_set:
                current_point = point_set[-1]
                z = current_point[2]
                current_point_size = base_point_size + int(
                    (z / self.depth) * max_additional_size
                )
                pygame.draw.circle(
                    screen,
                    self.dot_color,
                    (int(current_point[0]), int(current_point[1])),
                    current_point_size,
                )


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1024, 768))
pygame.display.set_caption("Chaotic Attractor Simulation")
clock = pygame.time.Clock()
attractor = Attractor(
    sigma=10,
    rho=28,
    beta=8 / 3,
    screen_width=1024,
    screen_height=768,
    scale=10,
    depth=500,
    dot_color=(115, 238, 220),
    trail_color=(77, 145, 209),
    num_points=100,
)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    attractor.step()

    screen.fill((0, 0, 0))
    attractor.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
