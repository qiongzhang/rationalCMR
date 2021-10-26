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
    #filename = "./output/logs_fitCMRprob_Z19GloVe_ucb10_bind2.json"
    #filename = "./output_cat/logs_fitCMRprob0728_P11GloVe_ucb5.json"
    #filename = "./output/logs0830_fitCMRprob2modes_K02_ucb10.json"
    #filename = "./output/logs0826_fitCMRprob2modes_Z19_RC_ucb5.json"
    #filename = "./output/logs0830_fitCMRprob2modesSim_K02_poi02.json"
    #filename = "./output/logs0915_fitCMRprob2modesOptimalEncode_Z19_ucb5.json"
    filename = "./output/logs0919_fitZ19_ucb10.json"
    optimizer = load_module(0)#(5)

    # set up logger
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)   

    random.seed(6)
    # start the optimization
    optimizer.maximize(
        init_points = 500,
        acq='ucb',
        kappa=10,
        random_state= random.randint(1,1000000),
        n_iter = 1000,
    )
    
    #random.seed(29) 
    #optimizer.maximize(
    #    init_points = 200, # number of initial random evaluations
    #    acq='poi',
    #    xi=0.02,
    #    random_state= random.randint(1,1000000),
    #    n_iter = 500, # number of evaluations using bayesian optimization
    #)


if __name__ == "__main__":
    main()

