"""2D Lorenz visualization using pygame.

Dynamics come from ``Attractors.LorenzAttractor`` — no inline duplication.
The 2D projection uses z for depth cues (brightness + line thickness).
"""

import sys
from collections import deque

import numpy as np
import pygame

from Attractors import LorenzAttractor


class Lorenz2DView:
    def __init__(
        self,
        screen_width=1024,
        screen_height=768,
        scale=10,
        depth=500,
        dot_color=(115, 238, 220),
        trail_color=(77, 145, 209),
        num_points=100,
        max_trail_length=50,
        dt=0.01,
    ):
        self.attractor = LorenzAttractor()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = scale
        self.depth = depth
        self.dt = dt
        self.num_points = num_points
        self.dot_color = dot_color
        self.trail_color = trail_color
        self.max_trail_length = max_trail_length
        self.states = [
            np.array([0.1 + i * 0.001, i * 0.001, i * 0.001], dtype=float)
            for i in range(num_points)
        ]
        self.points = [deque(maxlen=max_trail_length) for _ in range(num_points)]

    def step(self):
        for j in range(self.num_points):
            deriv = self.attractor.next_state(
                0.0, self.states[j], self.attractor.parameters
            )
            self.states[j] = self.states[j] + np.array(deriv) * self.dt
            x, y, z = self.states[j]
            x = x * self.scale + self.screen_width / 2
            y = y * self.scale + self.screen_height / 2
            z = z * self.scale
            self.points[j].append((x, y, z))

    def draw(self, screen):
        base_point_size = 3
        base_trail_size = 1
        max_additional_size = 2

        for point_set in self.points:
            if len(point_set) <= 1:
                continue
            z_values = np.array([p[2] for p in point_set])
            min_z, max_z = z_values.min(), z_values.max()
            z_range = max(max_z - min_z, 1)
            fade_factors = np.linspace(0, 1, len(point_set))
            for i in range(1, len(point_set)):
                z = z_values[i]
                brightness = max(
                    min(255, int(255 - (z + self.depth) / self.depth * 100)), 0
                )
                fade = fade_factors[i]
                faded_color = [
                    int(c * fade * brightness / 255) for c in self.trail_color
                ]
                size_factor = (z - min_z) / z_range
                trail_size = int(base_trail_size + size_factor * max_additional_size)
                pygame.draw.line(
                    screen,
                    faded_color,
                    point_set[i - 1][:2],
                    point_set[i][:2],
                    trail_size,
                )

        for point_set in self.points:
            if not point_set:
                continue
            current = point_set[-1]
            z = current[2]
            point_size = base_point_size + int((z / self.depth) * max_additional_size)
            pygame.draw.circle(
                screen, self.dot_color, (int(current[0]), int(current[1])), point_size
            )


def main():
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))
    pygame.display.set_caption("Lorenz Attractor (2D)")
    clock = pygame.time.Clock()
    view = Lorenz2DView()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        view.step()
        screen.fill((0, 0, 0))
        view.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
