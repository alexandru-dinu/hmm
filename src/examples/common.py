import sys
from os import path

sys.path.append(path.join(path.abspath(path.dirname(__file__)), ".."))

from hmm import HMM, forward, backward, viterbi
import numpy as np
