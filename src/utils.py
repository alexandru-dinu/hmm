import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from termcolor import colored


def sample(probabilities):
    # s, t = .0, np.random.sample()
    # for (value, p) in probabilities:
    #     s += p
    #     if s >= t:
    #         return value
    # raise ValueError("Probabilities " + str(probabilities) + " do not sum to one!")
    vs, ps = zip(*probabilities)
    idx = np.random.choice(np.arange(len(vs)), p=np.array(ps))
    return vs[idx]


def show_grids(grids, colors):
    _g_no = len(grids)
    fig, axs = plt.subplots(1, _g_no, figsize=(5 * _g_no, 4), sharey="row")
    for grid, ax in zip(grids, axs):
        cm = LinearSegmentedColormap.from_list("cm", colors)
        sns.heatmap(grid.color, annot=grid.elevation, cmap=cm,
                    square=True, cbar=False, annot_kws={"size": 30}, ax=ax)
        ax.set_title(grid.name)


def show_transition_prob(grids, As):
    _g_no = len(grids)
    fig, axs = plt.subplots(1, _g_no, figsize=(6 * _g_no, 4), sharey="row")
    for grid, ax, A in zip(grids, axs, As):
        sns.heatmap(A, square=True, cbar=True, ax=ax, cmap="Blues")
        ax.set_title(grid.name)


def show_emission_prob(grids, colors, Bs):
    _g_no = len(grids)
    fig, axs = plt.subplots(1, _g_no, figsize=(_g_no * 4, 6), sharey="row")

    for grid, ax, B in zip(grids, axs, Bs):
        N = grid.states_no
        _colors = np.array([list(range(len(colors))) for _ in range(N)])
        cm = LinearSegmentedColormap.from_list("cm", colors)
        sns.heatmap(_colors, cmap=cm, annot=B, ax=ax)
        ax.set_title(grid.name)


def show_sequence(grids, colors, func):
    # Example of a random sequence from a random model
    print(colored("Example of a random sequence from a random model", "cyan"))
    grid = np.random.choice(grids)
    T = np.random.randint(2, 6)
    observations, states = func(grid, T)

    print("Agent wandered on map \033[1m" + grid.name + "\033[0m")
    print("... going thorugh states", states)
    print("... observing", ", ".join([colors[o] for o in observations]))
    print("\n\n")

    cm = LinearSegmentedColormap.from_list("cm", colors)
    ax = sns.heatmap(grid.color, annot=grid.elevation, cmap=cm,
                     square=True, cbar=False, annot_kws={"size": 20})
    ax.set_title(grid.name)
    for t in range(T - 1):
        y0, x0 = states[t]
        y0, x0 = y0 + .5, x0 + .5
        y1, x1 = states[t + 1]
        y1, x1 = y1 + .5, x1 + .5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(color="y", width=5.))


def show_viterbi(grids, colors, seq_func, viterbi_func):
    # Decoding a state
    grid = np.random.choice(grids)
    T = np.random.randint(3, 6)
    observations, states = seq_func(grid, T)
    decoded, _ = viterbi_func(grid, observations)

    print(colored("Viterbi algorithm", "cyan"))
    print("Agent wandered on map \033[1m" + grid.name + "\033[0m")
    print("... going thorugh states", states)
    print("... observing", ", ".join([colors[o] for o in observations]))
    print("\nThe decoded sequence of states is", decoded)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey="row")
    cm = LinearSegmentedColormap.from_list("cm", colors)
    sns.heatmap(grid.color, annot=grid.elevation, cmap=cm, square=True,
                cbar=False, annot_kws={"size": 20}, ax=axs[0])
    sns.heatmap(grid.color, annot=grid.elevation, cmap=cm, square=True,
                cbar=False, annot_kws={"size": 20}, ax=axs[1])
    axs[0].set_title(grid.name + " - original path")
    axs[1].set_title(grid.name + " - decoded path")

    for t in range(T - 1):
        (y0, x0), (y1, x1) = states[t], states[t + 1]
        y0, x0, y1, x1 = y0 + .5, x0 + .5, y1 + .5, x1 + .5
        axs[0].annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(color="y", width=5.))
        (y0, x0), (y1, x1) = decoded[t], decoded[t + 1]
        y0, x0, y1, x1 = y0 + .5, x0 + .5, y1 + .5, x1 + .5
        axs[1].annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(color="y", width=5.))
