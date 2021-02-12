import os
import sys

import numpy as np

sys.path.insert(0, "../src")
from hmm import HMM, forward, backward, viterbi


hmm = HMM(
    pi=np.array([0.5, 0.5]),
    A=np.array([[0.75, 0.25], [0.25, 0.75]]),
    B=np.array(np.array([[0.45, 0.05, 0.45, 0.05], [0.05, 0.45, 0.05, 0.45]])),
)

obs = np.array([0, 1, 3, 2])
p, alpha = forward(obs, hmm)
q, beta = backward(obs, hmm)

print("p = %f, q = %f" % (p, q))

print("alpha")
for l in alpha:
    print("%f %f" % (l[0], l[1]))
print()

print("beta")
for l in beta:
    print("%f %f" % (l[0], l[1]))
print()

states, delta = viterbi(obs, hmm)

print("states:", states)

print("delta")
for l in delta:
    print("%f %f" % (l[0], l[1]))
print()

print("most prob = ", np.max(delta[-1, :]))

gamma = alpha * beta / p
print("gamma")
for l in gamma:
    print("%f %f" % (l[0], l[1]))
print()
