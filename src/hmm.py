# coding: utf-8

# Hidden Markov Models
# skel by Tudor Berariu (tudor.berariu@gmail.com), 2018


import numpy as np

from grid import COLORS
from utils import sample


# The problem: *The Climber Robot*


# ================== Extracting the HMM parameters ==================

# Given a `Grid` object, build the three matrices of parameters of HMM := <pi, A, B>

# Uniform distribution
def get_initial_distribution(grid):
    N = grid.states_no
    return np.ones(N) / N


# Transition probability matrix
def get_transition_probabilities(grid):
    H, W = grid.shape
    N = H * W
    A = np.zeros((N, N))

    states = [(i, j) for i in range(H) for j in range(W)]

    for si in states:
        for sj in states:
            si_next = dict(grid.get_neighbours(si))

            if sj not in si_next.keys():
                A[states.index(si), states.index(sj)] = 0
            else:
                A[states.index(si), states.index(sj)] = si_next[sj]

    return A


# Emission probability matrix
def get_emission_probabilities(grid, num_possible_obs):
    H, W = grid.shape
    N = grid.states_no
    B = np.zeros((H * W, num_possible_obs))

    states = [(i, j) for i in range(H) for j in range(W)]

    for si in states:
        for ck in np.arange(num_possible_obs):
            clrs = dict(grid.get_colors(si))
            B[states.index(si), ck] = clrs[ck]

    return B


# Given a model (a `Grid`), return a sequence of observations and the corresponding states.
def get_sequence(grid, length):
    H, W = grid.shape

    states, observations = [], []
    for t in range(length):
        # choose a random init state
        if t == 0:
            state = np.random.randint(H), np.random.randint(W)
        else:
            state = sample(grid.get_neighbours(state))
        o = sample(grid.get_colors(state))
        states.append(state)
        observations.append(o)

    return observations, states


# ====================== Evaluation =======================
# We'll now evaluate the probability that a given sequence of observations was generated by a given model.
# We will look at a sequence and see if we can figure out which grid generated it.


# Compute the probability that a given sequence comes from a given model
def forward(grid, observations):
    N = grid.states_no
    T = len(observations)
    alpha = np.zeros((T, N))

    pi = get_initial_distribution(grid)
    A = get_transition_probabilities(grid)
    B = get_emission_probabilities(grid, num_possible_obs=len(COLORS))

    for t in range(T):
        for s in range(N):
            if t == 0:
                alpha[t, s] = pi[s] * B[s, observations[0]]
            else:
                alpha[t, s] = np.sum(alpha[t - 1, :] * A[:, s]) * B[s, observations[t]]

    p = alpha[-1, :].sum()
    return p, alpha


# Decoding
# Compute the most probable sequence of states that generated the observations
def viterbi(grid, observations):
    N = grid.states_no
    H, W = grid.shape
    T = len(observations)
    delta = np.zeros((T, N))
    states = np.zeros(T, dtype=int)

    pi = get_initial_distribution(grid)
    A = get_transition_probabilities(grid)
    B = get_emission_probabilities(grid, num_possible_obs=len(COLORS))

    for t in range(T):
        for s in range(N):
            if t == 0:
                delta[t, s] = pi[s] * B[s, observations[0]]
            else:
                delta[t, s] = np.max(delta[t - 1, :] * A[:, s]) * B[s, observations[t]]

    states[T - 1] = np.argmax(delta[T - 1, :])
    for t in range(T - 2, -1, -1):
        states[t] = np.argmax(delta[t, :] * A[:, states[t + 1]])

    return [(s // W, s % W) for s in states], delta
