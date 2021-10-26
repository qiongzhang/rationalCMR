from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
import os
import fitpatterns_opt_drift as fitpatterns
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random import random
import random

def main():
    
    parser = ArgumentParser()
    parser.add_argument("--pos")
    args = parser.parse_args()
 
    # define model
    #filename = "./output/logs_0619optimalCMRprobBind2_nonoprim_pos" + str(int(args.pos)) + ".json"
    filename = "./output/logs_0812optimalCMRprob_noprim_gammaCF_1_pos" + str(int(args.pos)) + ".json"
    # Bounded region of parameter space
    pbounds = {  
        'beta_enc': (0,1),           # rate of context drift during encoding
        'beta_rec': (0,1),           # rate of context drift during recall
        'gamma_fc': (0,1),  # learning rate, feature-to-context
        'gamma_cf': (1,1),  # learning rate, context-to-feature
        'cue_position': [int(args.pos),int(args.pos)], 
    }

    # Bounded region of parameter space
    optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_acc,
        pbounds=pbounds,
        random_state=1,
    )

    # set up an observer (for saving the steps)
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)


    # start the optimization 
    optimizer.maximize(
        init_points = 200, # number of initial random evaluations
        acq='poi',
        xi=0.05,
        random_state= random.randint(1,1000000),
        n_iter = 300, # number of evaluations using bayesian optimization
    )


if __name__ == "__main__":
    main()

