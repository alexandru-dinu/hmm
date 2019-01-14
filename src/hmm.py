# coding: utf-8

# Hidden Markov Models
# skel by Tudor Berariu (tudor.berariu@gmail.com), 2018


import numpy as np

from grids import COLORS, GRIDS
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
def get_emission_probabilities(grid, num_possible_obs=len(COLORS)):
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
def forward(grid, observations, hmm=None):
    N = grid.states_no
    T = len(observations)
    alpha = np.zeros((T, N))

    if hmm is None:
        pi = get_initial_distribution(grid)
        A = get_transition_probabilities(grid)
        B = get_emission_probabilities(grid, num_possible_obs=len(COLORS))
    else:
        pi, A, B = hmm

    # alpha_0i = pi_i * B_i0
    alpha[0, :] = pi * B[:, observations[0]]

    # alpha_ti = sum_j(alpha_t-1,j * Aji) * B_it
    for t in range(1, T):
        alpha[t, :] = np.sum(alpha[t - 1, :] * np.transpose(A), axis=1) * B[:, observations[t]]

    p = alpha[-1, :].sum()
    return p, alpha


def backward(grid, observations, hmm=None):
    N = grid.states_no
    T = len(observations)
    beta = np.zeros((T, N))

    if hmm is None:
        A = get_transition_probabilities(grid)
        B = get_emission_probabilities(grid, num_possible_obs=len(COLORS))
    else:
        _, A, B = hmm

    # beta_T-1,i = 1
    beta[T - 1, :] = 1

    # beta_ti = sum_j(beta_t+1,j * A_ij * B_j,t+1)
    for t in range(T - 2, -1, -1):
        beta[t, :] = np.sum(beta[t + 1, :] * A * B[:, observations[t + 1]], axis=1)

    return beta


# Decoding: compute the most probable sequence of states that generated the observations
def viterbi(grid, observations, hmm=None):
    N = grid.states_no
    H, W = grid.shape
    T = len(observations)
    delta = np.zeros((T, N))
    states = np.zeros(T, dtype=int)

    if hmm is None:
        pi = get_initial_distribution(grid)
        A = get_transition_probabilities(grid)
        B = get_emission_probabilities(grid, num_possible_obs=len(COLORS))
    else:
        pi, A, B = hmm

    # t == 0
    delta[0, :] = pi * B[:, observations[0]]

    for t in range(1, T):
        delta[t, :] = np.max(delta[t - 1, :] * np.transpose(A), axis=1) * B[:, observations[t]]

    states[T - 1] = np.argmax(delta[T - 1, :])
    for t in range(T - 2, -1, -1):
        states[t] = np.argmax(delta[t, :] * A[:, states[t + 1]])

    return [(s // W, s % W) for s in states], delta


def baum_welch(grid, observations, num_possible_obs, num_it):
    N = grid.states_no
    T = len(observations)
    M = num_possible_obs

    # initial distribution
    pi = np.ones(N) / N

    # initial transition probabilities
    A = np.zeros((N, N))
    for i in range(N):
        A[i, :] = np.random.dirichlet(np.ones(N))
    # A = get_transition_probabilities(grid)

    # initial emission probabilities
    B = np.ones((N, M)) / M

    xi = np.zeros((T - 1, N, N))

    for it in range(1, num_it + 1):
        print(f"Iter {it}")

        # print("\n\n", A[1], "\n", get_transition_probabilities(grid)[1], "\n")
        # import time
        # time.sleep(0.1)

        # E step

        # p_obs = p(obs | theta)
        p_obs, alpha = forward(grid, observations, hmm=(pi, A, B))
        beta = backward(grid, observations, hmm=(None, A, B))

        # p_obs gets too small!! e.g. 7e-166

        # gamma_ti = (alpha_ti * beta_ti) / p(obs | theta) (size = T x N)
        gamma = alpha * beta / p_obs

        # xi_tij = (alpha_ti * A_ij * beta_t+1,j * B_j,t+1) / p(obs | theta) (size = T-1 x N x N)
        for t in range(T - 1):
            xi[t, :, :] = np.tile(alpha[t, :], (N, 1)).transpose() * A * B[:, observations[t + 1]] * beta[t + 1, :]
        xi /= p_obs
        # ---

        # M step
        # update pi
        pi = gamma[0, :]

        # update: A <- sum_t(xi_tij) / sum_t(gamma_ti) | all t's but last one
        for i in range(N):
            for j in range(N):
                A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

        # update: B <- sum_t(1(y_t==k) * gamma_ti) / sum_t(gamma_ti)
        for i in range(N):
            for k in range(M):
                B[i, k] = np.sum((observations == k) * gamma[:, i]) / np.sum(gamma[:, i])
        # ---

    print("Done!")
    return pi, A, B


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    grid = GRIDS[0]
    observations, _ = get_sequence(grid, 500)

    pi, A, B = baum_welch(grid, np.array(observations), num_possible_obs=len(COLORS), num_it=1000)

    pi_true = get_initial_distribution(grid)
    A_true = get_transition_probabilities(grid)
    B_true = get_emission_probabilities(grid, num_possible_obs=len(COLORS))
