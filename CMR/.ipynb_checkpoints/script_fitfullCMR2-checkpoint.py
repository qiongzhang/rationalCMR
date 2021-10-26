# import package
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
from random import random

# project specific
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import fitpatterns_opt_drift as fitpatterns

def main(): 
    # define model

    filename = "./output/logs0819_fitfullCMR2_K02Behav_poi02.json"

    # Bounded region of parameter space
    pbounds = {  
        'beta_enc': (0.1,1),           # rate of context drift during encoding
        'beta_rec': (0.1,1),           # rate of context drift during recall
        'beta_rec_post': (0.5,1),      # rate of context drift between lists
                                        # (i.e., post-recall)
        'gamma_fc': (0.1,1),  # learning rate, feature-to-context
        #'gamma_cf': (1,1),  # learning rate, context-to-feature
        #'s_cf': (0,0),       # scales influence of semantic similarity
                                # on M_CF matrix
        'phi_s': (1.0,5.0),      # primacy parameter - scale
        'phi_d': (1,3),     # primacy parameter - decay
        'kappa': (0.01,0.5),      # item activation decay rate
        'lamb': (0.01,0.5),        # lateral inhibition
        'eta': (0.01,0.5),        # width of noise gaussian
        'omega': (1,20),     # repetition prevention parameter
        'alpha': (0.5,1),      # repetition prevention parameter 
        'c_thresh': (0,0.2)      # repetition prevention parameter 
         
    }

    # Bounded region of parameter space
    optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_fitfullCMR2,
        pbounds=pbounds,
        random_state=1,
    )
            
            

    # set up logger
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)   

    
    random.seed(1) 
    optimizer.maximize(
        init_points = 200, # number of initial random evaluations
        acq='poi',
        xi=0.02,
        random_state= random.randint(1,1000000),
        n_iter = 300, # number of evaluations using bayesian optimization
    )


if __name__ == "__main__":
    main()

    