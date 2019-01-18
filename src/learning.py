import argparse

import numpy as np

from hmm import HMM, baum_welch
from model_grid import Grid
from model_simple import get_simple_models
from testing_grid import get_grids

np.set_printoptions(suppress=True, precision=6)


def learning_grid(args: argparse.Namespace) -> HMM:
    grid = get_grids()[args.idx]
    
    samples = []
    for s in range(args.num_samples):
        obs, _ = grid.get_sequence(length=np.random.randint(5, 11))
        samples += [obs]
    
    hmm = baum_welch(N=grid.states_no, M=len(Grid.COLORS), samples=samples)
    
    return hmm


def learn_simple(args: argparse.Namespace) -> HMM:
    true_hmm = get_simple_models()[args.idx]
    
    samples = []
    for s in range(args.num_samples):
        obs = true_hmm.sample_sequence(length=10)
        samples += [obs]
    
    hmm = baum_welch(true_hmm.N, true_hmm.M, samples, num_it=args.num_it, plot=args.plot)
    
    diff(true_hmm, hmm)
    
    return hmm


def diff(true_hmm: HMM, learned_hmm: HMM) -> None:
    print("True HMM pi")
    print(true_hmm.pi, "\n")
    print("Learned HMM pi")
    print(learned_hmm.pi, "\n")
    
    print("True HMM A")
    print(true_hmm.A, "\n")
    print("Learned HMM A")
    print(learned_hmm.A, "\n")
    
    print("True HMM B")
    print(true_hmm.B, "\n")
    print("Learned HMM B")
    print(learned_hmm.B, "\n")


def main(args: argparse.Namespace) -> None:
    models = {
        "grid"  : learning_grid,
        "simple": learn_simple
    }
    
    models[args.model](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to learn (grid/simple)")
    parser.add_argument("--idx", type=int, required=True, help="model index")
    parser.add_argument("--num_samples", type=int, required=True, help="number of samples (i.e. sequences)")
    parser.add_argument("--num_it", type=int, help="number of iterations")
    parser.add_argument("--plot", action="store_true")
    
    main(parser.parse_args())
