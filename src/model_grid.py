import numpy as np

from utils import sample_from
from hmm import HMM

"""
The problem: *The Climber Robot*
"""


class Grid(object):
    COLORS = ["Black", "Red", "Green", "Blue"]


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
        p_other = (1. - self.p_o) / len(Grid.COLORS)
        p_real = self.p_o + p_other
        rc = self.color[y, x]
        return [(i, p_real if i == rc else p_other) for (i, c) in enumerate(Grid.COLORS)]


    def get_initial_distribution(self):
        N = self.states_no
        return np.ones(N) / N


    def get_transition_probabilities(self):
        H, W = self.shape
        N = H * W
        A = np.zeros((N, N))

        states = [(i, j) for i in range(H) for j in range(W)]

        for si in states:
            for sj in states:
                si_next = dict(self.get_neighbours(si))

                if sj not in si_next.keys():
                    A[states.index(si), states.index(sj)] = 0
                else:
                    A[states.index(si), states.index(sj)] = si_next[sj]

        return A


    def get_emission_probabilities(self):
        H, W = self.shape
        M = len(self.COLORS)
        B = np.zeros((H * W, M))

        states = [(i, j) for i in range(H) for j in range(W)]

        for si in states:
            for ck in np.arange(M):
                clrs = dict(self.get_colors(si))
                B[states.index(si), ck] = clrs[ck]

        return B


    def get_sequence(self, length):
        """Given a model (a `Grid`), return a sequence of observations and the corresponding states."""
        H, W = self.shape

        states, observations = [], []
        for t in range(length):
            # choose a random init state
            if t == 0:
                state = np.random.randint(H), np.random.randint(W)
            else:
                state = sample_from(self.get_neighbours(state))
            o = sample_from(self.get_colors(state))
            states.append(state)
            observations.append(o)

        return np.array(observations), states


    def get_hmm(self) -> HMM:
        return HMM(
            pi=self.get_initial_distribution(),
            A=self.get_transition_probabilities(),
            B=self.get_emission_probabilities()
        )
