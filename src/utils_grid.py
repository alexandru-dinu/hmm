import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from termcolor import colored

from hmm import viterbi
from model_grid import Grid


def show_grids(grids):
    _g_no = len(grids)
    fig, axs = plt.subplots(1, _g_no, figsize=(5 * _g_no, 4), sharey="row")
    for grid, ax in zip(grids, axs):
        cm = LinearSegmentedColormap.from_list("cm", Grid.COLORS)
        sns.heatmap(grid.color, annot=grid.elevation, cmap=cm, square=True, cbar=False, annot_kws={"size": 30}, ax=ax)
        ax.set_title(grid.name)


def show_sequence(grids):
    print(colored("Example of a random sequence from a random model", "cyan"))

    grid = np.random.choice(grids)
    T = np.random.randint(2, 6)
    observations, states = grid.get_sequence(T)

    print("Agent wandered on map \033[1m" + grid.name + "\033[0m")
    print("... going thorugh states", states)
    print("... observing", ", ".join([Grid.COLORS[o] for o in observations]))
    print("\n")

    cm = LinearSegmentedColormap.from_list("cm", Grid.COLORS)
    ax = sns.heatmap(grid.color, annot=grid.elevation, cmap=cm, square=True, cbar=False, annot_kws={"size": 20})
    ax.set_title(grid.name)
    for t in range(T - 1):
        y0, x0 = states[t]
        y0, x0 = y0 + .5, x0 + .5
        y1, x1 = states[t + 1]
        y1, x1 = y1 + .5, x1 + .5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(color="y", width=5.))


def show_transition_prob(grids):
    _g_no = len(grids)
    fig, axs = plt.subplots(1, _g_no, figsize=(6 * _g_no, 4), sharey="row")
    for grid, ax in zip(grids, axs):
        A = grid.get_hmm().A
        sns.heatmap(A, square=True, cbar=True, ax=ax, cmap="Blues")
        ax.set_title(grid.name)


def show_emission_prob(grids):
    _g_no = len(grids)
    fig, axs = plt.subplots(1, _g_no, figsize=(_g_no * 4, 6), sharey="row")

    for grid, ax in zip(grids, axs):
        N = grid.states_no
        B = grid.get_hmm().B
        _colors = np.array([list(range(len(Grid.COLORS))) for _ in range(N)])
        cm = LinearSegmentedColormap.from_list("cm", Grid.COLORS)
        sns.heatmap(_colors, cmap=cm, annot=B, ax=ax)
        ax.set_title(grid.name)


def show_viterbi(grids):
    grid = np.random.choice(grids)
    H, W = grid.shape
    T = np.random.randint(3, 6)
    observations, states = grid.get_sequence(T)
    decoded, _ = viterbi(observations, grid.get_hmm())
    decoded = [(s // H, s % W) for s in decoded]

    print(colored("Viterbi algorithm", "cyan"))
    print("Agent wandered on map \033[1m" + grid.name + "\033[0m")
    print("... going thorugh states", states)
    print("... observing", ", ".join([Grid.COLORS[o] for o in observations]))
    print("\nThe decoded sequence of states is", decoded)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey="row")
    cm = LinearSegmentedColormap.from_list("cm", Grid.COLORS)
    sns.heatmap(grid.color, annot=grid.elevation, cmap=cm, square=True, cbar=False, annot_kws={"size": 20}, ax=axs[0])
    sns.heatmap(grid.color, annot=grid.elevation, cmap=cm, square=True, cbar=False, annot_kws={"size": 20}, ax=axs[1])
    axs[0].set_title(grid.name + " - original path")
    axs[1].set_title(grid.name + " - decoded path")

    for t in range(T - 1):
        (y0, x0), (y1, x1) = states[t], states[t + 1]
        y0, x0, y1, x1 = y0 + .5, x0 + .5, y1 + .5, x1 + .5
        axs[0].annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(color="y", width=5.))
        (y0, x0), (y1, x1) = decoded[t], decoded[t + 1]
        y0, x0, y1, x1 = y0 + .5, x0 + .5, y1 + .5, x1 + .5
        axs[1].annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(color="y", width=5.))
