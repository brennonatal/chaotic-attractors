import pygame
import numpy as np
import sys


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
    ):
        """
        Initializes an attractor model with given parameters and prepares the drawing environment.

        Parameters:
        - sigma, rho, beta: Parameters that affect the attractor's behavior. Changing these can lead to different chaotic effects.
        - screen_width, screen_height: Dimensions of the Pygame window in pixels.
        - scale: A scaling factor for the attractor's coordinates, to adjust its size on the screen.
        - depth: A parameter to adjust the depth effect in the visualization, influencing the brightness of the lines.
        - dot_color: The RGB color for the current point of the attractor.
        - trail_color: The RGB color for the trail of the attractor.

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
        self.points = []
        self.dot_color = pygame.Color(*dot_color)
        self.trail_color = pygame.Color(*trail_color)

    def step(self):
        dx = self.sigma * (self.state[1] - self.state[0]) * self.dt
        dy = (self.state[0] * (self.rho - self.state[2]) - self.state[1]) * self.dt
        dz = (self.state[0] * self.state[1] - self.beta * self.state[2]) * self.dt
        self.state += np.array([dx, dy, dz])

        x, y, z = self.state
        x = (x * self.scale) + (self.screen_width / 2)
        y = (y * self.scale) + (self.screen_height / 2)
        z *= self.scale
        self.points.append((x, y, z))

    def draw(self, screen):
        base_point_size = 3
        base_trail_size = 1
        max_additional_size = 2
        brightness = 0

        # Ensure there is enough points for a trail
        if len(self.points) > 1:
            # Get the minimum and maximum z-values for normalization
            min_z = min(p[2] for p in self.points)
            max_z = max(p[2] for p in self.points)
            z_range = max(max_z - min_z, 1)  # Avoid division by zero

            trail_length = 50
            trail_points = self.points[-trail_length:]

            # Draw the trail
            for i in range(1, len(trail_points)):
                # Calculate brightness based on the z-coordinate
                z = trail_points[i][2]
                brightness = max(
                    min(255, int(255 - (z + self.depth) / self.depth * 100)), 0
                )

                # Fade the color based on its position in the trail
                fade_factor = i / len(trail_points)
                faded_color = self.trail_color.lerp(
                    pygame.Color(0, 0, 0), 1 - fade_factor
                )  # Fading to black
                faded_color.a = int(
                    brightness * fade_factor
                )  # Apply brightness to alpha channel

                # Calculate size variation based on z-coordinate
                size_factor = (z - min_z) / z_range
                trail_size = int(base_trail_size + size_factor * max_additional_size)

                pygame.draw.line(
                    screen,
                    faded_color,
                    trail_points[i - 1][:2],
                    trail_points[i][:2],
                    trail_size,
                )

        # Draw the current point
        if self.points:
            current_point = self.points[-1]
            z = current_point[2]
            current_point_size = base_point_size + int(
                (z / self.depth) * max_additional_size
            )
            self.dot_color.a = brightness
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
attractor = Attractor()

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
