import numpy as np
from termcolor import colored


def test_transition_prob(grids, As, test_idxs, test_values):
    for i, (grid, A) in enumerate(zip(grids, As)):
        assert A.shape == (grid.states_no, grid.states_no), "Bad shape!"
        assert np.allclose(A.sum(axis=1), np.ones(grid.states_no)), "Rows should sum to one!"
        assert np.allclose(A[test_idxs], test_values[i]), "Bad values!"

    print(colored("\n>>> Transition matrix looks right!", "green"))


def test_emission_prob(grids, colors, Bs, test_idxs, test_values):
    for i, (grid, B) in enumerate(zip(grids, Bs)):
        assert B.shape == (grid.states_no, len(colors)), "Bad shape!"
        assert np.allclose(B.sum(axis=1), np.ones(grid.states_no)), "Rows should sum to one!"
        assert np.allclose(B[test_idxs], test_values[i]), "Bad values for " + grid.name + "!"

    print(colored("\n>>> Emission probabilities look right!", "green"))


def test_forward_algorithm(grids, colors, seq_func, forward_func):
    # See the forward algorithm in action
    print(colored("Forward algorithm", "cyan"))

    grid = np.random.choice(grids)
    print("The real model is \033[1m" + grid.name + "\033[0m")

    T = np.random.randint(2, 10)
    observations, _ = seq_func(grid, T)
    print("The observed sequence is", ", ".join([colors[i] for i in observations]))

    best_grid, best_p = None, None
    for grid in grids:
        p, _ = forward_func(grid, observations)
        print("Probability that comes from " + grid.name + " is %.7f." % (p))
        if best_grid is None or best_p < p:
            best_grid, best_p = grid.name, p

    print("Most probably the sequence was generated from " + best_grid + ".")


def test_alpha_values(grid, observations, func, test_values):
    p, alpha = func(grid, observations)
    assert alpha.shape == (3, grid.states_no), "Bad shape!"
    assert np.allclose(alpha, test_values), "Bad values!"
    assert np.allclose(p, sum(test_values[2])), "Bad values!"

    print(colored("\n>>> Alpha matrix looks right!", "green"))


def test_viterbi(grid, observations, viterbi_func, test_states, test_values):
    states, delta = viterbi_func(grid, observations)
    print(states)
    assert len(states) == len(test_states)
    assert all([s_i == s_j for (s_i, s_j) in zip(states, test_states)])
    assert np.allclose(delta, test_values)

    print(colored("\n>>> Viterbi looks right!", "green"))


def test_forward_by_sequence_length(grids, seq_func, forward_func, runs_no=1000):
    print("See how sequence length influences p")

    for T in range(1, 11):
        correct = 0

        for _ in range(runs_no):
            true_grid = np.random.choice(grids)
            observations, _ = seq_func(true_grid, T)
            best_grid, best_p = None, None
            for grid in grids:
                p, _ = forward_func(grid, observations)
                if best_grid is None or best_p < p:
                    best_grid, best_p = grid.name, p
            correct += (best_grid == true_grid.name)

        perc = float(correct * 100) / runs_no

        print("%5d / %d (%5.2f%%) for T = %2d" % (correct, runs_no, perc, T))


# Evaluate how good the decoded paths are
def test_decoded_by_sequence_length(grids, seq_func, viterbi_func, runs_no=1000):
    print("Number of states correctly decoded.")

    for T in range(1, 11):
        correct = 0

        for run_id in range(runs_no):
            grid = np.random.choice(grids)
            observations, states = seq_func(grid, T)
            decoded, _ = viterbi_func(grid, observations)
            correct += sum([a == b for a, b in zip(states, decoded)])
        perc = float(correct * 100) / (runs_no * T)

        print("%5d / %5d (%5.2f%%) for T =%2d" % (correct, runs_no * T, perc, T))
