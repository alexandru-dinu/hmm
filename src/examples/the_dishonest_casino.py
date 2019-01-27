from common import *


hmm = HMM(
    pi=np.array([0.8, 0.2]),
    A=np.array([[0.9, 0.1], [0.1, 0.9]]),
    B=np.array(np.array([
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]
    ]))
)

obs = np.array([1, 4, 3, 6, 6, 4]) - 1 # -1 because the sides are indices

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
