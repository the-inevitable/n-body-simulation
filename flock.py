"""
The three core rules of the Boids simulation are as follows:
- Separation: Keep a minimum distance between the boids.
- Alignment: Point each boid in the average direction of movement of its local flockmates.
- Cohesion: Move each boid toward the center of mass of its local flockmates.
"""

import math
import argparse

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist

width, height = 640, 480


class Boids:
    def __init__(self, n):
        # Initial position and velocities.
        self.pos = [width/2, height/2] + 10*np.random.rand(2*n).reshape(n, 2)

        # normalized random velocities.
        angles = 2 * math.pi * np.random.rand(n)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.n = n

        # Minimum distance of approach.
        self.min_dist = 25.0
        # Maximum magnitude of velocities calculated by the "rules".
        self.max_rule_vel = 0.03
        # Maximum magnitude of the final velocity.
        self.max_vel = 2.0

    def tick(self, frame_num, pts, beak):
        """Update the simulation by one time step."""
        # Get pairwise distance.
        self.dist_matrix = squareform(pdist(self.pos))

        # Apply rules:
        self.vel += self.apply_rules()
        self.limit(self.vel, self.max_vel)
        self.pos += self.vel
        self.apply_boundary_conditions()

        # Update data.
        pts.set_data(
            self.pos.reshape(2*self.n)[::2],
            self.pos.reshape(2*self.n)[1::2]
        )
        vec = self.pos + 10 * self.vel / self.max_vel
        beak.set_data(
            vec.reshape(2 * self.n)[::2],
            vec.reshape(2 * self.n)[1::2]
        )

    def limit_vec(self, vec, max_val):
        """Limit the magnitude of the 2D vector."""
        magn = norm(vec)

        if magn > max_val:
            vec[0], vec[1] = vec[0] * max_val / magn, vec[1] * max_val / magn

    def limit(self, X, max_val):
        """limit the magnitude of 2D vectors in array X to max_value."""
        for vec in X:
            self.limit_vec(vec, max_val)

    def apply_boundary_conditions(self):
        """Apply boundary conditions."""
        delta_r = 2.0
        for coord in self.pos:
            if coord[0] > width + delta_r:
                coord[0] = - delta_r
            if coord[0] < - delta_r:
                coord[0] = width + delta_r
            if coord[1] > height + delta_r:
                coord[1] = - delta_r
            if coord[1] < - delta_r:
                coord[1] = height + delta_r

    def apply_rules(self):
        # Apply rule #1: Separation.
        D = self.dist_matrix < 25.0
        vel = self.pos * D.sum(axis=1).reshape(self.n, 1) - D.dot(self.pos)
        self.limit(vel, self.max_rule_vel)

        # Distance threshold for alignment (different from separation).
        D = self.dist_matrix < 50.0

        # Apply rule #2: Alignment.
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.max_rule_vel)
        vel += vel2

        # Apply rule #3: Cohesion.
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.max_rule_vel)
        vel += vel3
        return vel

    def button_press(self, event):
        """Event handler for matplotlib button presses."""
        # Left-click to add a boid.
        if event.button is 1:
            self.pos = np.concatenate((self.pos, np.array([[event.xdata, event.ydata]])), axis=0)

            # Generate a random velocity.
            angles = 2 * math.pi * np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)
            self.n += 1

        # Right-click to scatter boids.
        elif event.button is 3:
            # Add scattering velocity.
            self.vel += 0.1 * (self.pos - np.array([[event.xdata, event.ydata]]))


def tick(frame_num, pts, beak, boids):
    """Update function for animation."""
    boids.tick(frame_num, pts, beak)
    return pts, beak


# Spins up the whole thing.
def main():
    print('Starting boids...')
    parser = argparse.ArgumentParser(description="Implementing Craig Reynold's Boids...")

    # Add arguments.
    parser.add_argument('--num-boids', dest='n', required=False)
    args = parser.parse_args()

    # Set the initial number of boids.
    n = int(args.n) if args.n else 100

    # Create boids.
    boids = Boids(n)

    # Set up plot.
    fig = plt.figure()
    ax = plt.axes(xlim=(0, width), ylim=(0, height))
    pts, = ax.plot([], [], markersize=10, c='k', marker='o', ls='none')
    beak, = ax.plot([], [], markersize=4, c='r', marker='o', ls='none')
    anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, boids), interval=50)

    # Add a "button press" event handler.
    cid = fig.canvas.mpl_connect('button_press_event', boids.button_press)
    plt.show()


if __name__ == '__main__':
    main()

