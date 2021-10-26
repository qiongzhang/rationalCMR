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
    #filename = "./output/logs0909_fitCMRprob_rep_K02_p012_poi02.json"
    filename = "./output/logs0909_fitCMRprob_rep_Z09_p012_poi02_2.json"
    #optimizer = load_module(2)
    optimizer = load_module(3)

    # set up logger
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)   


    random.seed(29) 
    optimizer.maximize(
        init_points = 200, # number of initial random evaluations
        acq='poi',
        xi=0.02,
        random_state= random.randint(1,1000000),
        n_iter = 500, # number of evaluations using bayesian optimization
    )


if __name__ == "__main__":
    main()

