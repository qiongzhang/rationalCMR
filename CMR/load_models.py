# import package
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
from random import random

# project specific
import numpy as np
import fitpatterns_opt_drift as fitpatterns
    
def load_vanilla():
    pbounds = {  
        'beta_enc': (0.1,1),           # rate of context drift during encoding
        'beta_rec': (0.1,1),            # rate of context drift during recall
        'gamma_fc': (0,1),  # learning rate, feature-to-context
        'phi_s': (0.1,3.0),      # primacy parameter - scale
        'phi_d': (0.1,3.0),     # primacy parameter - decay
        'epsilon_d': (0,5),     # stopping probability - scale     
        'k': (0,10),      # luce choice rule 
    }
    # Bounded region of parameter space
    optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_prob,
        pbounds=pbounds,
        random_state=1,
    )  
    return optimizer

def load_reminder():
    # Bounded region of parameter space
    pbounds = {  
        'beta_enc': (0,1),           # rate of context drift during encoding
        'beta_rec': (0,1),           # rate of context drift during recall
        'gamma_fc': (0,1),  # learning rate, feature-to-context
        'cue_position': (0,10), 
    }

    # Bounded region of parameter space
    optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_prob_reminder,
        pbounds=pbounds,
        random_state=1,
    )
    return optimizer


def load_reminder_noise():
    # Bounded region of parameter space
    pbounds = {  
        'beta_enc': (0,1),           # rate of context drift during encoding
        'beta_rec': (0,1),           # rate of context drift during recall
        'gamma_fc': (0,1),  # learning rate, feature-to-context
        'k': (0,10),           # rate of context drift during recall
        'enc_rate': (0,1),  # learning rate, feature-to-context
        'cue_position': (0,10), 
    }

    # Bounded region of parameter space
    optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_prob_reminder_noise,
        pbounds=pbounds,
        random_state=1,
    )
    return optimizer
  

def load_fullCMR2():
    pbounds = {  
        'beta_enc': (0.1,1),           # rate of context drift during encoding
        'beta_rec': (0.1,1),           # rate of context drift during recall
        'beta_rec_post': (0.5,1),      # rate of context drift between lists
                                        # (i.e., post-recall)
        'gamma_fc': (0.1,1),  # learning rate, feature-to-context
        #'gamma_cf': (0.5,1),  # learning rate, context-to-feature
        #'s_cf': (0.5,3),       # scales influence of semantic similarity
                                # on M_CF matrix
        'phi_s': (1.0,5.0),      # primacy parameter - scale
        'phi_d': (0.5,5),     # primacy parameter - decay
        'kappa': (0.01,0.5),      # item activation decay rate
        'lamb': (0.01,0.5),        # lateral inhibition
        'eta': (0.01,0.5),        # width of noise gaussian
        'omega': (1,20),     # repetition prevention parameter
        'alpha': (0.5,1),      # repetition prevention parameter 
        'c_thresh': (0,0.2),      # editting 
    }
    # Bounded region of parameter space
    optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_fitfullCMR2,
        pbounds=pbounds,
        random_state=1,
    )  
    return optimizer