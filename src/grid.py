import numpy as np
from utils import *


class Grid(object):
    def __init__(self, name, elevation, color, p_t=.8, p_o=.8):
        self._name = name
        self.elevation = np.array(elevation)
        self.color = np.array(color)
        assert self.elevation.shape == self.color.shape
        self.p_t = p_t
        self.p_o = p_o

    @property
    def name(self):
        return self._name

    @property
    def states_no(self):
        return self.elevation.size

    @property
    def shape(self):
        return self.elevation.shape

    def get_neighbours(self, state):
        """Returns a list of tuples (neighbour, probability)"""
        y, x = state
        H, W = self.shape

        neighbours = []
        for (dy, dx) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                neighbours.append((ny, nx))

        elevation = [self.elevation[i] for i in neighbours]
        max_e = max(elevation)
        max_no = len([e for e in elevation if e == max_e])
        p_other = (1. - self.p_t) / len(neighbours)
        p_max = (self.p_t / max_no) + p_other
        prob = [p_max if e == max_e else p_other for e in elevation]

        return list(zip(neighbours, prob))

    def get_colors(self, state):
        """Returns a list of tuples (color, probability)"""
        y, x = state
        p_other = (1. - self.p_o) / len(COLORS)
        p_real = self.p_o + p_other
        rc = self.color[y, x]
        return [(i, p_real if i == rc else p_other) for (i, c) in enumerate(COLORS)]


COLORS = ["Black", "Red", "Green", "Blue"]

# Three toy grids to play with
# We'll use the following three grids to test our algorithms.

grid1 = Grid("Grid 1",
             [[1, 2, 3, 5], [2, 2, 1, 2], [3, 2, 1, 1], [0, 0, 0, 0]],  # elevation
             [[0, 3, 1, 2], [3, 1, 2, 0], [2, 2, 0, 0], [3, 0, 3, 1]])  # color

grid2 = Grid("Grid 2",
             [[0, 0, 1, 1], [2, 1, 0, 2], [1, 0, 0, 2], [4, 4, 3, 3]],  # elevation
             [[0, 3, 1, 2], [3, 1, 2, 0], [2, 2, 0, 0], [3, 0, 3, 1]])  # color

grid3 = Grid("Grid 3",
             [[2, 1, 2, 3], [1, 1, 2, 2], [1, 0, 1, 1], [2, 1, 1, 2]],  # elevation
             [[2, 3, 1, 0], [1, 3, 3, 1], [0, 2, 0, 2], [2, 1, 1, 2]])  # color

GRIDS = [grid1, grid2, grid3]
