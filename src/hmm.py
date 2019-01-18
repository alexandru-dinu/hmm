from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from utils import sample_from

"""
Hidden Markov Models
initial skel and tests by Tudor Berariu (tudor.berariu@gmail.com), 2018
"""


class HMM:
    """
    A Hiden Markov Model is defined by:
        - N = number of states
        - M = number of possible observations
        - the initial distribution of probabilities - pi[i] = p(s_0 = i) :: N
        - the transition matrix - A[i,j] = p(St+1 = j | St = i) :: N x N
        - the emission matrix - B[i,k] = p(Ot = k | St = i) :: N x M
    """
    
    
    def __init__(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray):
        assert len(pi) == A.shape[0] == A.shape[1] == B.shape[0]
        
        self.pi = pi
        self.A = A
        self.B = B
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
    
    
    def sample_initial(self) -> float:
        return sample_from([(s, self.pi[s]) for s in range(self.N)])
    
    
    def sample_transition(self, from_state: int) -> int:
        return sample_from([(s, self.A[from_state, s]) for s in range(self.N)])
    
    
    def sample_observation(self, state: int) -> int:
        return sample_from([(o, self.B[state, o]) for o in range(self.M)])
    
    
    def sample_sequence(self, length: int) -> np.ndarray:
        seq = []
        s = None
        
        for i in range(length):
            s = self.sample_initial() if i == 0 else self.sample_transition(from_state=s)
            seq += [self.sample_observation(state=s)]
        
        return np.array(seq)


# Compute the probability that a given sequence comes from a given model
def forward(N: int, observations: np.ndarray, hmm: HMM) -> Tuple[float, np.ndarray]:
    T = len(observations)
    alpha = np.zeros((T, N))
    
    # alpha_0i = pi_i * B_i0
    alpha[0, :] = hmm.pi * hmm.B[:, observations[0]]
    
    # alpha_ti = sum_j(alpha_t-1,j * Aji) * B_it
    for t in range(1, T):
        alpha[t, :] = np.sum(alpha[t - 1, :] * np.transpose(hmm.A), axis=1) * hmm.B[:, observations[t]]
    
    # p_obs = p(obs | hmm)
    p_obs = alpha[-1, :].sum()
    
    return p_obs, alpha


def backward(N: int, observations: np.ndarray, hmm: HMM) -> Tuple[float, np.ndarray]:
    T = len(observations)
    beta = np.zeros((T, N))
    
    # beta_T-1,i = 1
    beta[T - 1, :] = 1
    
    # beta_ti = sum_j(beta_t+1,j * A_ij * B_j,t+1)
    for t in range(T - 2, -1, -1):
        beta[t, :] = np.sum(beta[t + 1, :] * hmm.A * hmm.B[:, observations[t + 1]], axis=1)
    
    # p_obs = p(obs | hmm)
    p_obs = np.sum(hmm.pi * hmm.B[:, observations[0]] * beta[0, :])
    
    return p_obs, beta


# Decoding: compute the most probable sequence of states that generated the observations
def viterbi(N: int, observations: np.ndarray, hmm: HMM) -> Tuple[np.ndarray, np.ndarray]:
    T = len(observations)
    delta = np.zeros((T, N))
    states = np.zeros(T, dtype=int)
    
    # t == 0
    delta[0, :] = hmm.pi * hmm.B[:, observations[0]]
    
    for t in range(1, T):
        delta[t, :] = np.max(delta[t - 1, :] * np.transpose(hmm.A), axis=1) * hmm.B[:, observations[t]]
    
    states[T - 1] = np.argmax(delta[T - 1, :])
    for t in range(T - 2, -1, -1):
        states[t] = np.argmax(delta[t, :] * hmm.A[:, states[t + 1]])
    
    return states, delta


# Learning

def initialize_hmm(N: int, M: int) -> HMM:
    # uniform initial distribution
    pi = np.ones(N) / N
    
    # initial transition probabilities
    A = np.zeros((N, N))
    for i in range(N):
        x = np.random.random(N)
        A[i, :] = x / x.sum()
    
    # initial emission probabilities
    B = np.ones((N, M)) / M
    
    return HMM(pi, A, B)


# Expectation-Maximization

def expectation(N: int, samples: List[np.ndarray], hmm: HMM) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    L = len(samples)
    
    logp = 0
    gammas, xis, scales = [], [], []
    
    for l in range(L):
        T = len(samples[l])
        
        p, alpha = forward(N, samples[l], hmm)
        _, beta = backward(N, samples[l], hmm)
        
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            xi[t, :, :] = np.tile(alpha[t, :], (N, 1)).transpose() * hmm.A * hmm.B[:, samples[l][t + 1]] * beta[t + 1, :]
        
        xis += [xi / p]
        gammas += [alpha * beta / p]
        
        logp += np.log(p)
    
    return gammas, xis, logp / L


def maximization(N: int, M: int, samples: List[np.ndarray], gammas: List[np.ndarray], xis: List[np.ndarray], hmm: HMM) -> HMM:
    L = len(samples)
    
    # update pi
    for i in range(N):
        hmm.pi[i] = sum([gammas[l][0, i] for l in range(L)]) / L
    
    # update A
    for i in range(N):
        for j in range(N):
            x, y = 0, 0
            
            for l in range(L):
                x += np.sum(xis[l][:-1, i, j])
                y += np.sum(gammas[l][:-1, i])
            
            hmm.A[i, j] = x / y
        hmm.A[i] /= hmm.A[i].sum()
    
    # update B
    for i in range(N):
        for k in range(M):
            x, y = 0, 0
            
            for l in range(L):
                x += np.sum((samples[l] == k) * gammas[l][:, i])
                y += np.sum(gammas[l][:, i])
            
            hmm.B[i, k] = x / y
        hmm.B[i] /= hmm.B[i].sum()
        
        return hmm


def baum_welch(N, M, samples, num_it=None, plot=True) -> HMM:
    num_it = np.inf if num_it is None else num_it
    
    hmm = initialize_hmm(N, M)
    
    gammas, xis, logp = expectation(N, samples, hmm)
    
    logps, it = [0, logp], 0
    
    while it < num_it:
        
        print("Iter [%5d], logp = %.8f, old_logp = %.8f, diff = [%.8f]" % (it + 1, logps[-1], logps[-2], logps[-1] - logps[-2]))
        
        if plot and (it + 1) % 100 == 0:
            plt.plot(np.arange(it + 1), logps[1:], color='b', lw=1.0)
            plt.show()
        
        hmm = maximization(N, M, samples, gammas, xis, hmm)
        gammas, xis, logp = expectation(N, samples, hmm)
        
        logps += [logp]
        it += 1
    
    print("Baum-Welch done!\n\n")
    
    return hmm
