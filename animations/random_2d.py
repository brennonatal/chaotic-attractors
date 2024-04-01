import math
import random
import numpy as np
from PIL import Image, ImageDraw
import pygame
import sys
import time


MAX_ITERATIONS = 100000


def generate_coefficients():
    """Generate random coefficients for the attractor."""
    return np.random.uniform(-2, 2, size=6), np.random.uniform(-2, 2, size=6)


def initialize_states():
    """Initialize starting states for x and y."""
    return random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)


def next_state(ax, ay, x_last, y_last):
    """Calculate next state based on current state and coefficients."""
    x_next = (
        ax[0]
        + ax[1] * x_last
        + ax[2] * x_last**2
        + ax[3] * x_last * y_last
        + ax[4] * y_last
        + ax[5] * y_last**2
    )
    y_next = (
        ay[0]
        + ay[1] * x_last
        + ay[2] * x_last**2
        + ay[3] * x_last * y_last
        + ay[4] * y_last
        + ay[5] * y_last**2
    )
    return x_next, y_next


def update_bounds(x, y, xmin, xmax, ymin, ymax):
    """Update min and max bounds based on current points."""
    return min(xmin, x), max(xmax, x), min(ymin, y), max(ymax, y)


def is_series_diverging(xmin, xmax, ymin, ymax, threshold=1e10):
    """Check if the series is diverging to infinity."""
    return (
        xmin < -threshold or ymin < -threshold or xmax > threshold or ymax > threshold
    )


def is_series_converging(dx, dy, threshold=1e-10):
    """Check if the series is converging to a point."""
    return abs(dx) < threshold and abs(dy) < threshold


def create_attractor(attemps=1000):
    for n in range(attemps):
        xmin = 1e32
        xmax = -1e32
        ymin = 1e32
        ymax = -1e32

        ax, ay = generate_coefficients()
        x, y = [initialize_states()[0]], [initialize_states()[1]]
        xe, ye = (
            x[0] + random.uniform(-0.5, 0.5) / 1000.0,
            y[0] + random.uniform(-0.5, 0.5) / 1000.0,
        )

        lyapunov, found = 0, True

        xe = x[0] + random.uniform(-0.5, 0.5) / 1000.0
        ye = y[0] + random.uniform(-0.5, 0.5) / 1000.0
        dx = x[0] - xe
        dy = y[0] - ye
        d0 = math.sqrt(dx * dx + dy * dy)

        for i in range(MAX_ITERATIONS):
            # Calculate next term
            x_last, y_last = x[-1], y[-1]
            x_new, y_new = next_state(ax, ay, x_last, y_last)
            x.append(x_new)
            y.append(y_new)

            # Update the bounds
            xmin, xmax, ymin, ymax = update_bounds(x_new, y_new, xmin, xmax, ymin, ymax)

            # Does the series tend to infinity
            if is_series_diverging(xmin, xmax, ymin, ymax):
                found = False
                break

            # Does the series tend to a point
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            if i > 0 and is_series_converging(dx, dy):
                found = False
                break

            # Calculate the lyapunov exponents
            if i > 1000:
                dx = x[i] - x_new
                dy = y[i] - y_new
                dd = math.sqrt(dx * dx + dy * dy)
                lyapunov += math.log(math.fabs(dd / d0))
                xe = x[i] + d0 * dx / dd
                ye = y[i] + d0 * dy / dd

        # Classify the series according to lyapunov
        if abs(lyapunov) < 10:  # neutrally stable
            continue
        elif lyapunov < 0:  # periodic
            continue

        # Save the image
        if found:
            # print("chaotic {} ".format(lyapunov))
            # save_attractor(n, xmin, xmax, ymin, ymax, x, y)
            return xmin, xmax, ymin, ymax, x, y
        return None


# Function to normalize and scale points
def normalize_and_scale(x, y, xmin, xmax, ymin, ymax, width, height):
    ix = width * (x - xmin) / (xmax - xmin)
    iy = height * (1 - (y - ymin) / (ymax - ymin))  # Pygame's y-axis is inverted
    return ix, iy


def save_attractor(n, xmin, xmax, ymin, ymax, x, y):
    width, height = 500, 500

    # Save the image
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)

    for i in range(MAX_ITERATIONS):
        ix = width * (x[i] - xmin) / (xmax - xmin)
        iy = height * (y[i] - ymin) / (ymax - ymin)
        if i > 100:
            draw.point([ix, iy], fill="black")

    image.save("output/new{}.png".format(n), "PNG")
    print("saved attractor to ./output/{}.png".format(n))


# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Dynamic Chaotic Attractor Animation")
clock = pygame.time.Clock()

# Main loop
running = True
last_time = time.time()
index = 0
while running:
    # current_time = time.time()
    # if current_time - last_time > 10:  # Check if 10 seconds have passed
    #     # Regenerate attractor data
    #     xmin, xmax, ymin, ymax, x, y = create_attractor()
    #     last_time = current_time  # Reset the timer
    att = create_attractor()
    if not att:
        continue
    xmin, xmax, ymin, ymax, x, y = att

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Clear screen with black background

    # Draw attractor
    for i in range(len(x)):
        ix, iy = normalize_and_scale(x[i], y[i], xmin, xmax, ymin, ymax, width, height)
        pygame.draw.circle(screen, (255, 255, 255), (int(ix), int(iy)), 1)  # Draw point

    pygame.display.flip()  # Update the display
    clock.tick(60)  # Limit to 60 frames per second

pygame.quit()
sys.exit()
