import pygame
import numpy as np
import sys

class Attractor:
    def __init__(self, sigma=10, rho=28, beta=8/3, screen_width=1024, screen_height=768, scale=10, depth=500):
        """
        Initializes an attractor model with given parameters and prepares the drawing environment.
        
        Parameters:
        - sigma, rho, beta: Parameters that affect the attractor's behavior. Changing these can lead to different chaotic effects.
        - screen_width, screen_height: Dimensions of the Pygame window in pixels.
        - scale: A scaling factor for the attractor's coordinates, to adjust its size on the screen.
        - depth: A parameter to adjust the depth effect in the visualization, influencing the brightness of the lines.
        
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

    def lorenz_attractor_step(self):
        dx = self.sigma * (self.state[1] - self.state[0]) * self.dt
        dy = (self.state[0] * (self.rho - self.state[2]) - self.state[1]) * self.dt
        dz = (self.state[0] * self.state[1] - self.beta * self.state[2]) * self.dt
        self.state += np.array([dx, dy, dz])
        
        x, y, z = self.state
        x = (x * self.scale) + (self.screen_width / 2)
        y = (y * self.scale) + (self.screen_height / 2)
        # Adjust z for depth effect
        z *= self.scale
        self.points.append((x, y, z))

    def draw(self, screen):
        trail_length = 50
        line_thickness = 1
        point_size = 4

        # Draw the trail
        if len(self.points) > trail_length:            
            trail_points = self.points[-trail_length:]
        else:
            trail_points = self.points

        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                # Fade the trail color based on the position in the trail
                fade_factor = i / len(trail_points)
                color = pygame.Color(int(255 * fade_factor), int(255 * fade_factor), int(255 * fade_factor))
                pygame.draw.line(screen, color, trail_points[i-1][:2], trail_points[i][:2], line_thickness)

        # Draw the current point
        if self.points:
            current_point = self.points[-1]
            pygame.draw.circle(screen, pygame.Color(255, 255, 255), (int(current_point[0]), int(current_point[1])), point_size)


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

    attractor.lorenz_attractor_step()

    screen.fill((0, 0, 0))
    attractor.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
