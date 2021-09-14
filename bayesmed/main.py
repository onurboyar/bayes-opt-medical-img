import argparse
import functools
import torch

from bayes_opt import BayesianOptimization
from bayesmed.models.trainer import train_unet
from pbounds import p_bounds

def main(args):
    f = functools.partial(
        train_unet,
        epochs = args.epochs,
        batch_size = args.batch_size,
        initial_lr = args.initial_lr,
        scheduler = args.scheduler,
        s_gamma = args.scheduler_gamma,
        s_step = args.scheduler_step,
        train_dir = args.training_datapath,
        eval_dir = args.eval_datapath,
    )

    optimizer = BayesianOptimization(
        f = f,
        pbounds = p_bounds,
        random_state = 1,
    )

    optimizer.maximize(
        init_points = args.bo_init_points,
        n_iter = args.bo_n_iter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--training_datapath",
                        "-td",
                        help="directory for training data",
                        type = str
    )

    parser.add_argument("--eval_datapath",
                        "-ed",
                        help="directory for test data",
                        type = str
    )
    
    parser.add_argument("--batch_size",
                        "-bs",
                        help="batch size for both training and evaluating",
                        type = int,
                        default = 5
    )
    
    parser.add_argument("--epochs",
                        "-ep",
                        help="number of epochs for training",
                        type = int,
                        default = 15
    )
    
    parser.add_argument("--initial_lr",
                        "-il",
                        help="initial learning rate for training",
                        type = float,
                        default = 0.001
    )

    parser.add_argument("--scheduler",
                        "-sc",
                        help="learning rate scheduler: torch.optim.lr_scheduler.StepLR",
                        type = bool,
                        default = True
    )

    parser.add_argument("--scheduler_gamma",
                        "-scg",
                        help="if args.scheduler True: decay parameter",
                        type = float,
                        default = 0.1
    )

    parser.add_argument("--scheduler_step",
                        "-scs",
                        help="if args.scheduler True: step size for multiplying with gamma",
                        type = int,
                        default = 1
    )

    parser.add_argument("--bo_init_points",
                        "-bip",
                        help="nof initial points for bayesian optimization",
                        type = int,
                        default = 20
    )

    parser.add_argument("--bo_n_iter",
                        "-bni",
                        help="nof iterations for bayesian optimization",
                        type = int,
                        default = 20
    )
    
    args = parser.parse_args()

    main(args)
