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
from load_module_prob import load_module


def main(): 
    # define model

    filename = "./output/logs0930_fitCMRprob_PEERS_Behav_poi02.json"
    
    pbounds = {  
        'beta_enc': (0.1,1),           # rate of context drift during encoding
        'beta_rec': (0.1,1),            # rate of context drift during recall
        'gamma_fc': (0,1),  # learning rate, feature-to-context
        'phi_s': (0.1,5.0),      # primacy parameter - scale
        'phi_d': (0.5,3.0),     # primacy parameter - decay
        'epsilon_d': (0,5),     # stopping probability - scale     
        'k': (0,10),      # luce choice rule 
        }        
    # Bounded region of parameter space
    optimizer = BayesianOptimization(
    f=fitpatterns.obj_func_prob,
    pbounds=pbounds,
    random_state=1,
    )    
            
            

    # set up logger
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)   

    
    random.seed(1) 
    optimizer.maximize(
        init_points = 500, # number of initial random evaluations
        acq='poi',
        xi=0.02,
        random_state= random.randint(1,1000000),
        n_iter = 200, # number of evaluations using bayesian optimization
    )


if __name__ == "__main__":
    main()

