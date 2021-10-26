# import package
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
from random import random

# project specific
import numpy as np
import fitpatterns_opt_drift as fitpatterns
    
def load_module(number):
    if number==0:
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.5,1),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0,1),  # learning rate, feature-to-context
            'gamma_cf': (0,1),  # learning rate, context-to-feature
            's_cf': (0,1),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            #'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0,5),     # stopping probability - scale     

            'k_intra': (0,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model7,
        pbounds=pbounds,
        random_state=1,
    )
        
    elif number==1:
        pbounds = {  
        'beta_enc': (0,1),           # rate of context drift during encoding
        'beta_rec': (0,1),           # rate of context drift during recall
        'gamma_fc': (0,1),  # learning rate, feature-to-context
        'gamma_cf': (0,1),  # learning rate, context-to-feature
        'cue_position': (0,1), 
    }


        # Bounded region of parameter space
        optimizer = BayesianOptimization(
            f=fitpatterns.obj_func_acc,
            pbounds=pbounds,
            random_state=1,
        )
        
    elif number==2:
        pbounds = {  
            'beta_enc': (0,1),           # rate of context drift during encoding
            'beta_rec': (0,1),            # rate of context drift during recall                              # (i.e., post-recall)
            'gamma_fc': (0,1),  # learning rate, feature-to-context
            #'gamma_cf': (1,1),  # learning rate, context-to-feature
            #'s_cf': (0,0),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (1.0,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            #'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            #'epsilon_d': (0,10),     # stopping probability - scale     
            'Num': (0,10),      # stopping rule
            'k': (0,10),      # luce choice rule 
        }


        # Bounded region of parameter space
        optimizer = BayesianOptimization(
            f=fitpatterns.obj_func_simple_rep,
            pbounds=pbounds,
            random_state=1,
        )
        
    elif number==3:
        pbounds = {  
            'beta_enc': (0.3,1),           # rate of context drift during encoding
            'beta_rec': (0,1),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0,1),  # learning rate, feature-to-context
            'gamma_cf': (0,1),  # learning rate, context-to-feature
            's_cf': (0,1),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (1.0,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'Num': (0,20),      # stopping rule
            'k': (0,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_rep_full,
        pbounds=pbounds,
        random_state=1,
    )    
        
    elif number==4:
        pbounds = {  
            'beta_enc': (0,1),           # rate of context drift during encoding
            'beta_rec': (0,1),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0,1),  # learning rate, feature-to-context
            'gamma_cf': (1,1),  # learning rate, context-to-feature
            's_cf': (0,0),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_d': (0,10),     # stopping probability - scale     

            'k_intra': (0,10),      # luce choice rule 
            'lr': (0.0,5.0),     # learning rate 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model7_OptimalEncode,
        pbounds=pbounds,
        random_state=1,
    )    
    elif number==5:
        pbounds = {  
            'beta_enc': (0,1),           # rate of context drift during encoding
            'beta_rec': (0,1),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0,1),  # learning rate, feature-to-context
            #'gamma_cf': (1,1),  # learning rate, context-to-feature
            #'s_cf': (0,0),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_d': (0,10),     # stopping probability - scale     

            'k_intra': (0,10),      # luce choice rule 
            'lr': (0.0,3.0),     # learning rate 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_OptimalEncode,
        pbounds=pbounds,
        random_state=1,
    )    
    return optimizer 