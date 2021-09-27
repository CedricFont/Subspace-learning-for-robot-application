import numpy as np
import matplotlib.pyplot as plt

class Pendulum(object):
    def __init__(self, dt=0.01):
        self.dt = dt
        self.Dx = 2
        self.Du = 1
        self.b = 1
        self.m = 1
        self.l = 1
        self._g = 9.81

    def set_init_state(self, x0):
        self.x0 = x0

    def g(self, x):
        return self.m * self._g * self.l * np.sin(x[..., 0])

    def f(self, x, u):
        x_next0 = x[..., 0] + x[..., 1] * self.dt
        x_next1 = (1 - self.dt * self.b / (self.m * self.l ** 2)) * x[..., 1] + \
                  - self._g * self.dt * np.sin(x[..., 0]) / self.l + \
                  self.dt * u[..., 0] / (self.m * self.l ** 2)

        x_next = np.concatenate([x_next0[..., None], x_next1[..., None]], -1)
        return x_next

    def plot(self, x, color='k', ax=None):
        px = np.array([0, -self.l * np.sin(x[0])])
        py = np.array([0, self.l * np.cos(x[0])])
        xlim = [-2 * self.l, 2 * self.l]

        if not ax:
            line = plt.plot(px, py, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
            plt.axes().set_aspect('equal')
            plt.axis(xlim + xlim)
        else:
            line = ax.plot(px, py, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
            ax.set_aspect("equal")

        return

