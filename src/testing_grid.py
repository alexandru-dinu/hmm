from typing import List

import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

from hmm import forward, viterbi
from model_grid import Grid
from utils_grid import show_grids, show_sequence, show_emission_prob, show_transition_prob, show_viterbi


def test_transition_prob(grids, test_idxs, test_values):
    print("Testing transition matrix...")

    for i, grid in enumerate(grids):
        A = grid.get_hmm().A
        assert A.shape == (grid.states_no, grid.states_no), "Bad shape!"
        assert np.allclose(A.sum(axis=1), np.ones(grid.states_no)), "Rows should sum to one!"
        assert np.allclose(A[test_idxs], test_values[i]), "Bad values!"

    print(colored(">>> Transition matrix looks right!", "green"))
    print("\n")


def test_emission_prob(grids, test_idxs, test_values):
    print("Testing emission matrix...")

    for i, grid in enumerate(grids):
        B = grid.get_hmm().B
        assert B.shape == (grid.states_no, len(Grid.COLORS)), "Bad shape!"
        assert np.allclose(B.sum(axis=1), np.ones(grid.states_no)), "Rows should sum to one!"
        assert np.allclose(B[test_idxs], test_values[i]), "Bad values for " + grid.name + "!"

    print(colored(">>> Emission matrix look right!", "green"))
    print("\n")


def test_forward_algorithm(grids):
    print("Testing forward...")

    grid = np.random.choice(grids)
    print("The real model is \033[1m" + grid.name + "\033[0m")

    T = np.random.randint(2, 10)
    observations, _ = grid.get_sequence(T)
    print("The observed sequence is", ", ".join([Grid.COLORS[i] for i in observations]))

    best_grid, best_p = None, None
    for grid in grids:
        p, _ = forward(observations, grid.get_hmm())
        print("Probability that comes from " + grid.name + " is %.7f." % p)
        if best_grid is None or best_p < p:
            best_grid, best_p = grid.name, p

    print("Most probably the sequence was generated from " + best_grid + ".")
    print("\n")


def test_alpha_values(grid, observations, test_values):
    print("Testing alpha values...")

    p, alpha = forward(observations, grid.get_hmm())

    assert alpha.shape == (3, grid.states_no), "Bad shape!"
    assert np.allclose(alpha, test_values), "Bad values!"
    assert np.allclose(p, sum(test_values[2])), "Bad values!"

    print(colored(">>> Alpha matrix looks right!", "green"))
    print("\n")


def test_viterbi(grid, observations, test_states, test_values):
    print("Testing viterbi...")

    H, W = grid.shape
    states, delta = viterbi(observations, grid.get_hmm())
    states = [(s // H, s % W) for s in states]

    print("States:", states)
    print("TStates:", test_states)

    assert len(states) == len(test_states)
    assert all([s_i == s_j for (s_i, s_j) in zip(states, test_states)])
    assert np.allclose(delta, test_values)

    print(colored(">>> Viterbi looks right!", "green"))
    print("\n")


def test_forward_by_sequence_length(grids, runs_no=1000):
    print("See how sequence length influences p...")

    for T in range(1, 11):
        correct = 0

        for _ in range(runs_no):
            true_grid = np.random.choice(grids)
            observations, _ = true_grid.get_sequence(T)
            best_grid, best_p = None, None
            for grid in grids:
                p, _ = forward(observations, grid.get_hmm())
                if best_grid is None or best_p < p:
                    best_grid, best_p = grid.name, p
            correct += (best_grid == true_grid.name)

        perc = float(correct * 100) / runs_no

        print("%5d / %d (%5.2f%%) for T = %2d" % (correct, runs_no, perc, T))
    print("\n")


def test_decoded_by_sequence_length(grids, runs_no=1000):
    print("Evaluate how good the decoded paths are...")

    for T in range(1, 11):
        correct = 0

        for run_id in range(runs_no):
            grid = np.random.choice(grids)
            H, W = grid.shape
            observations, states = grid.get_sequence(T)
            decoded, _ = viterbi(observations, grid.get_hmm())
            decoded = [(s // H, s % W) for s in decoded]
            correct += sum([a == b for a, b in zip(states, decoded)])
        perc = float(correct * 100) / (runs_no * T)

        print("%5d / %5d (%5.2f%%) for T =%2d" % (correct, runs_no * T, perc, T))
    print("\n")


###

def get_grids() -> List[Grid]:
    grid1 = Grid("Grid 1",
                 [[1, 2, 3, 5], [2, 2, 1, 2], [3, 2, 1, 1], [0, 0, 0, 0]],  # elevation
                 [[0, 3, 1, 2], [3, 1, 2, 0], [2, 2, 0, 0], [3, 0, 3, 1]])  # color

    grid2 = Grid("Grid 2",
                 [[0, 0, 1, 1], [2, 1, 0, 2], [1, 0, 0, 2], [4, 4, 3, 3]],  # elevation
                 [[0, 3, 1, 2], [3, 1, 2, 0], [2, 2, 0, 0], [3, 0, 3, 1]])  # color

    grid3 = Grid("Grid 3",
                 [[2, 1, 2, 3], [1, 1, 2, 2], [1, 0, 1, 1], [2, 1, 1, 2]],  # elevation
                 [[2, 3, 1, 0], [1, 3, 3, 1], [0, 2, 0, 2], [2, 1, 1, 2]])  # color

    return [grid1, grid2, grid3]


def main(grids: List[Grid]) -> None:
    show_grids(grids)
    plt.show()

    show_sequence(grids)
    plt.show()

    show_transition_prob(grids)
    plt.show()

    test_transition_prob(grids,
                         test_idxs=([0, 1, 2, 5, 10, 13, 15], [1, 0, 3, 4, 9, 2, 14]),
                         test_values=np.array([
                             [.5, .2 / 3, .8 + .2 / 3, .8 / 3 + .05, .85, 0, .1],
                             [.1, .2 / 3, .8 + .2 / 3, .85, .05, 0, .9],
                             [.5, .4 + .2 / 3, .8 + .2 / 3, .05, .05, 0, .5]
                         ])
                         )

    show_emission_prob(grids)
    plt.show()

    test_emission_prob(grids,
                       test_idxs=([0, 3, 2, 5, 10, 13, 15], [2, 2, 3, 0, 1, 1, 1]),
                       test_values=np.array([
                           [.05, .85, .05, .05, .05, .05, .85],
                           [.05, .85, .05, .05, .05, .05, .85],
                           [.85, .05, .05, .05, .05, .85, .05]
                       ])
                       )

    test_forward_by_sequence_length(grids, runs_no=1000)
    test_forward_algorithm(grids)
    test_alpha_values(grids[0], observations=[2, 2, 3],
                      test_values=np.array([
                          [3.12500000e-03, 3.12500000e-03, 3.12500000e-03, 5.31250000e-02,
                           3.12500000e-03, 3.12500000e-03, 5.31250000e-02, 3.12500000e-03,
                           5.31250000e-02, 5.31250000e-02, 3.12500000e-03, 3.12500000e-03,
                           3.12500000e-03, 3.12500000e-03, 3.12500000e-03, 3.12500000e-03],
                          [2.08333333e-05, 1.38020833e-04, 4.78385417e-03, 4.60416667e-03,
                           1.36718750e-03, 2.86458333e-04, 6.19791667e-04, 5.33854167e-04,
                           4.30755208e-02, 2.64739583e-02, 4.11458333e-04, 1.58854167e-04,
                           1.87500000e-04, 1.58854167e-04, 3.38541667e-05, 2.08333333e-05],
                          [5.01736111e-06, 3.57044271e-04, 2.39509549e-04, 2.30434028e-04,
                           1.71725825e-02, 7.27517361e-05, 1.94704861e-05, 3.14539931e-05,
                           1.19282552e-03, 1.03400174e-03, 6.97309028e-05, 3.74565972e-06,
                           2.44994792e-03, 6.72352431e-05, 2.82595486e-05, 6.42361111e-07]])
                      )

    show_viterbi(grids)
    plt.show()

    test_decoded_by_sequence_length(grids, runs_no=1000)

    test_viterbi(grids[1], observations=[0, 0, 1, 3],
                 test_states=[(1, 3), (2, 3), (3, 3), (3, 2)],
                 test_values=[[5.31250000e-02, 3.12500000e-03, 3.12500000e-03, 3.12500000e-03,
                               3.12500000e-03, 3.12500000e-03, 3.12500000e-03, 5.31250000e-02,
                               3.12500000e-03, 3.12500000e-03, 5.31250000e-02, 5.31250000e-02,
                               3.12500000e-03, 5.31250000e-02, 3.12500000e-03, 3.12500000e-03],
                              [1.77083333e-04, 2.65625000e-04, 7.29166667e-05, 1.77083333e-04,
                               2.39062500e-03, 7.29166667e-05, 1.77083333e-04, 3.01041667e-03,
                               7.29166667e-05, 1.77083333e-04, 3.01041667e-03, 3.91354167e-02,
                               2.30208333e-03, 2.39062500e-03, 2.25781250e-03, 2.30208333e-03],
                              [7.96875000e-06, 8.85416667e-07, 1.05364583e-04, 1.00347222e-05,
                               7.96875000e-06, 9.48281250e-04, 1.00347222e-05, 1.30451389e-04,
                               5.57812500e-05, 7.96875000e-06, 1.30451389e-04, 1.30451389e-04,
                               1.03593750e-04, 1.03593750e-04, 1.27942708e-04, 2.88297569e-02],
                              [2.65625000e-08, 4.03019531e-05, 5.01736111e-08, 4.56579861e-06,
                               6.85133203e-04, 1.85937500e-07, 2.37070312e-06, 4.51562500e-07,
                               5.17968750e-07, 2.37070312e-06, 4.34837963e-07, 1.44148785e-04,
                               7.63140625e-05, 5.54418403e-06, 2.20547641e-02, 5.65289352e-06]]
                 )


if __name__ == '__main__':
    main(get_grids())
