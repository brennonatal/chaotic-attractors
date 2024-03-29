import pygame
import numpy as np
import sys

class LorenzAttractor:
    def __init__(self, screen_width=1024, screen_height=768, scale=10, depth=500):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = scale
        self.depth = depth
        self.sigma, self.rho, self.beta = 10, 28, 8/3
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
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                # Use z-coordinate to adjust color brightness
                z = self.points[i][2]
                brightness = max(min(255, int(255 - (z + self.depth) / self.depth * 100)), 0)
                color = pygame.Color(brightness, brightness, brightness)
                
                pygame.draw.line(screen, color, self.points[i-1][:2], self.points[i][:2])

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1024, 768))
pygame.display.set_caption("Lorenz Attractor Simulation")
clock = pygame.time.Clock()
attractor = LorenzAttractor()

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
