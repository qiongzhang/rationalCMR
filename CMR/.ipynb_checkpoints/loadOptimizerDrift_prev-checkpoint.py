# import package
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger # for saving 
from bayes_opt.event import Events # for saving
from bayes_opt.util import load_logs # for loading
from random import random

# project specific
import numpy as np
import fitpatterns_opt_drift as fitpatterns

def load_BO(rational):
    if int(rational)==0:
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.1,1),           # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1),  # learning rate, feature-to-context, (0.3,0.7)
            'gamma_cf': (0.1,1),  # learning rate, feature-to-context, (0.3,0.7)
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (1.0,3.0),      # primacy parameter - scale
            'phi_d': (0.1,1.5),     # primacy parameter - decay
            'kappa': (0.01,0.5),      # item activation decay rate
            'lamb': (0.01,0.5),        # lateral inhibition
            'eta': (0.01,0.5),        # width of noise gaussian, (0.01,0.5)
            'c_thresh': (0.001,0.5),   # threshold of context similarity 
            'omega': (1,20),     # repetition prevention parameter
            'alpha': (0.5,1),      # repetition prevention parameter, (0.5,1)    
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_rational0,
        pbounds=pbounds,
        random_state=1,
    )

        
    if int(rational)==1:
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1),  # learning rate, feature-to-context, (0.3,0.7)
            'gamma_cf': (0.1,1),  # learning rate, feature-to-context, (0.3,0.7)
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (1.0,3.0),      # primacy parameter - scale
            'phi_d': (0.1,1.5),     # primacy parameter - decay
            'kappa': (0.01,0.5),      # item activation decay rate
            'lamb': (0.01,0.5),        # lateral inhibition
            'eta': (0.01,0.5),        # width of noise gaussian, (0.01,0.5)
            'c_thresh': (0.001,0.5),   # threshold of context similarity 
            'omega': (1,20),     # repetition prevention parameter
            'alpha': (0.5,1),      # repetition prevention parameter, (0.5,1)    

            'k': (0.001,10)      # scale parameter in cue selection
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_rational1,
        pbounds=pbounds,
        random_state=1,
    )
    return optimizer


def load_BO_models(model):
    
    if model==0: # full model
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,1),           # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model0,
        pbounds=pbounds,
        random_state=1,
    )
        
    if model==1: # simplify stopping probability
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,1),            # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            #'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model1,
        pbounds=pbounds,
        random_state=1,
    )
        


    if model==2: # simply semantics
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,10),             # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            #'gamma_cf': (0.1,1.0),  # learning rate, context-to-feature
            #'s_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model2,
        pbounds=pbounds,
        random_state=1,
    )
 
    if model==3: # simply semantics 2
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,1),            # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model3,
        pbounds=pbounds,
        random_state=1,
    )

    if model==4: # simplify primacy and stopping prob
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,10),             # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            #'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model4,
        pbounds=pbounds,
        random_state=1,
    )


    if model==5: # simply three components
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,1),              # rate of context drift during recall
            'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            #'gamma_cf': (0.1,1.0),  # learning rate, context-to-feature
            #'s_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            #'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model5,
        pbounds=pbounds,
        random_state=1,
    )
        
    if model==6: # simply a set of things
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            #'beta_rec': (0.0001,1),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model6,
        pbounds=pbounds,
        random_state=1,
    )

    if model==7: # simply a set of things
        pbounds = {  
            'beta_enc': (0,1),           # rate of context drift during encoding
            'beta_rec': (0,1),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0,1),  # learning rate, feature-to-context
            'gamma_cf': (0,1),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            #'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model7,
        pbounds=pbounds,
        random_state=1,
    )

    if model==8: # simply a set of things
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,10),            # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            'b_cf': (-0.99,0.99),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            #'s_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            #'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model8,
        pbounds=pbounds,
        random_state=1,
    )
        
        
        
    if model==9: # beta_rec_post = 1
        pbounds = {  
            'beta_enc': (0.1,1),           # rate of context drift during encoding
            'beta_rec': (0.0001,10),           # rate of context drift during recall
            #'beta_rec_post': (0.1,1),      # rate of context drift between lists
                                            # (i.e., post-recall)
            'gamma_fc': (0.1,1.0),  # learning rate, feature-to-context
            'gamma_cf': (0.0001,1.0),  # learning rate, context-to-feature
            's_cf': (0.1,3),       # scales influence of semantic similarity
                                    # on M_CF matrix
            's_fc': (0.1,3),       # scales influence of semantic similarity
                                    # on M_FC matrix
            'phi_s': (0.1,3.0),      # primacy parameter - scale
            'phi_d': (0.1,3.0),     # primacy parameter - decay
            'epsilon_s': (0.0,0.5),     # stopping probability - intercept 
            'epsilon_d': (0.1,10),     # stopping probability - scale     

            'k_intra': (0.01,10),      # luce choice rule 
        }
         
        # Bounded region of parameter space
        optimizer = BayesianOptimization(
        f=fitpatterns.obj_func_simple_model9,
        pbounds=pbounds,
        random_state=1,
    )        
    return optimizer