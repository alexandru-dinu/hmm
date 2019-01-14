from grids import GRIDS
from hmm import *
from testing import *
from utils import *

# show_grids(GRIDS, COLORS)
# show_sequence(GRIDS, COLORS, func=get_sequence)


# TRANSITION

# show_transition_prob(GRIDS, As=map(get_transition_probabilities, GRIDS))

test_transition_prob(GRIDS, As=map(get_transition_probabilities, GRIDS),
                     test_idxs=([0, 1, 2, 5, 10, 13, 15], [1, 0, 3, 4, 9, 2, 14]),
                     test_values=np.array([
                         [.5, .2 / 3, .8 + .2 / 3, .8 / 3 + .05, .85, 0, .1],
                         [.1, .2 / 3, .8 + .2 / 3, .85, .05, 0, .9],
                         [.5, .4 + .2 / 3, .8 + .2 / 3, .05, .05, 0, .5]
                     ])
                     )

# EMISSION

# show_emission_prob(GRIDS, COLORS, Bs=[get_emission_probabilities(g, len(COLORS) for g in GRIDS])

test_emission_prob(GRIDS, COLORS, Bs=[get_emission_probabilities(g, len(COLORS)) for g in GRIDS],
                   test_idxs=([0, 3, 2, 5, 10, 13, 15], [2, 2, 3, 0, 1, 1, 1]),
                   test_values=np.array([
                       [.05, .85, .05, .05, .05, .05, .85],
                       [.05, .85, .05, .05, .05, .05, .85],
                       [.85, .05, .05, .05, .05, .85, .05]
                   ])
                   )

# FORWARD
# test_forward_by_sequence_length(GRIDS, seq_func=get_sequence, forward_func=forward, runs_no=1000)

test_forward_algorithm(GRIDS, COLORS, seq_func=get_sequence, forward_func=forward)

test_alpha_values(GRIDS[0], observations=[2, 2, 3], func=forward,
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

# VITERBI
show_viterbi(GRIDS, COLORS, seq_func=get_sequence, viterbi_func=viterbi)

# test_decoded_by_sequence_length(GRIDS, seq_func=get_sequence, viterbi_func=viterbi, runs_no=1000)

test_viterbi(GRIDS[1], observations=[0, 0, 1, 3], viterbi_func=viterbi,
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
