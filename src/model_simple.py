import numpy as np

from hmm import HMM


def get_simple_models():
    simple_hmm1 = HMM(
        pi=np.array([0.5, 0.5]),
        A=np.array([[0.9, 0.1], [0.8, 0.2]]),
        B=np.array(np.array([[0.05, 0.95], [0.95, 0.05]]))
    )
    
    simple_hmm2 = HMM(
        pi=np.array([0.3, 0.5, 0.2]),
        A=np.array([[0.6, 0.1, 0.3], [0.1, 0.8, 0.1], [0.2, 0.5, 0.3]]),
        B=np.array([[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]])
    )
    
    return [simple_hmm1, simple_hmm2]



