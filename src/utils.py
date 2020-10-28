import numpy as np


def sample_from(distribution):
    # s, t = .0, np.random.sample()
    # for (value, p) in distribution:
    #     s += p
    #     if s >= t:
    #         return value
    # raise ValueError("Probabilities " + str(distribution) + " do not sum to one!")

    values, probs = zip(*distribution)
    idx = np.random.choice(np.arange(len(values)), p=np.array(probs))

    return values[idx]
