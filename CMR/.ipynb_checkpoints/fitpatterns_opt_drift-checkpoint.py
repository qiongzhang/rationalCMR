import numpy as np
import os
import fullCMR2 as CMR2
import probCMR as CMR_prob
import matplotlib.pyplot as plt
import itertools


# set parameters
def set_parameter_prob():
    # 0: K02 1:K02_top; 2: K02_bottom; 3:PEERS 4:PEERS_PFR_1 5:PEERS_PFR_2
    # 6:PEERS_ACC_1 7:PEERS_ACC_1
    data_id = 3
    N = 0 # repeat the simulations multiple time
    lls = [10,10,10,16,16,16,16,16]
    lag_examine = 4 # for CRP
    ll = lls[data_id]
    return N, ll, lag_examine, data_id

# load behavioral patterns
def load_patterns(data_id):
    if data_id == 0:# k02 dataset
        data_spc = np.loadtxt('data/K02_files/K02_spc.txt')
        data_pfr = np.loadtxt('data/K02_files/K02_pfr.txt')
        data_crp = np.loadtxt('data/K02_files/K02_crp.txt')
    elif data_id == 1:# k02 dataset - top half
        data_spc = np.loadtxt('data/K02_files/K02_spc_good.txt')
        data_pfr = np.loadtxt('data/K02_files/K02_pfr_good.txt')
        data_crp = np.loadtxt('data/K02_files/K02_crp_good.txt')        
    elif data_id == 2:# k02 dataset - top half
        data_spc = np.loadtxt('data/K02_files/K02_spc_bad.txt')
        data_pfr = np.loadtxt('data/K02_files/K02_pfr_bad.txt')
        data_crp = np.loadtxt('data/K02_files/K02_crp_bad.txt')
    elif data_id == 3:
        data_spc = np.loadtxt('data/PEERS_spc.txt')
        data_pfr = np.loadtxt('data/PEERS_pfr.txt')
        data_crp = np.loadtxt('data/PEERS_crp.txt')        
    elif data_id == 4:
        data_spc = np.loadtxt('data/PEERS_PFR_1_spc.txt')
        data_pfr = np.loadtxt('data/PEERS_PFR_1_pfr.txt')
        data_crp = np.loadtxt('data/PEERS_PFR_1_crp.txt')        
    elif data_id == 5:
        data_spc = np.loadtxt('data/PEERS_PFR_2_spc.txt')
        data_pfr = np.loadtxt('data/PEERS_PFR_2_pfr.txt')
        data_crp = np.loadtxt('data/PEERS_PFR_2_crp.txt')        
    elif data_id == 6:
        data_spc = np.loadtxt('data/PEERS_ACC50_1_spc.txt')
        data_pfr = np.loadtxt('data/PEERS_ACC50_1_pfr.txt')
        data_crp = np.loadtxt('data/PEERS_ACC50_1_crp.txt')               
    elif data_id == 7:
        data_spc = np.loadtxt('data/PEERS_ACC50_2_spc.txt')
        data_pfr = np.loadtxt('data/PEERS_ACC50_2_pfr.txt')
        data_crp = np.loadtxt('data/PEERS_ACC50_2_crp.txt')    

        
    return data_spc, data_pfr, data_crp

# load data
def load_structure(data_id):
    if data_id >-1 and data_id <3:# k02 dataset
        LSA_path = 'data/K02_files/K02_LSA.txt'
        data_path = 'data/K02_files/K02_data.txt'
        data_rec_path = 'data/K02_files/K02_recs.txt'
        subjects_path = 'data/K02_files/K02_list_ids.txt' # assume each list is a subject
    elif data_id == -1: # PEERS single subject
        LSA_path = 'data/K02_files/K02_LSA.txt'
        data_path = 'data/PEERS_subj_structure.txt'
        data_rec_path = 'data/PEERS_subj_structure.txt'
        subjects_path = 'data/PEERS_subj_ids.txt' # assume each list is a subject    
    else:# PEERS
        LSA_path = 'data/K02_files/K02_LSA.txt'
        data_path = 'data/PEERS_data_structure.txt'
        data_rec_path = 'data/PEERS_data_structure.txt'
        subjects_path = 'data/PEERS_list_ids.txt' # assume each list is a subject

        
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)
    return LSA_mat, data_path, data_rec_path, subjects_path

# set parameters
def set_parameter():
    patterns = [0,1,2,3,6,7] # patterns to fit 3,6,7
    # 0 - spc; 1 - pfr; 2 - crp; 3 - intrusion rate
    # 4 - crp for within category; 5 - crp for across categories; 6 - semantic clustering
    N = 1 # repeat the simulations multiple time
    lls = [10,16,24,24]
    lag_examine = 4 # for CRP
    data_id = 0 # 0: K02 1:autolab 2:poly11-un 3:poly11-cat
    ll = lls[data_id]
    return patterns, N, ll, lag_examine, data_id


#
# data recoding 
def data_recode(data_pres,data_rec):
    # recode data into lists without 0-paddings; convert to python indexing from matlab indexing by -1
    presents = [[int(y)-1 for y in list(x) if y>0] for x in list(data_pres)]
    recalls = [[int(y)-1 for y in list(x) if y>0] for x in list(data_rec)]

    # recode recall data into serial positions (set as x+100 if it is a repetition; set as -99 if it is an intrusion)
    recalls_sp = []
    for i,recall in enumerate(recalls):
        recall_sp = []
        this_list = presents[i]
        recallornot = np.zeros(len(this_list))
        for j,item in enumerate(recall):
            try:
                sp = this_list.index(item)
                if recallornot[sp]==0: # non-repetition
                    recall_sp.append(this_list.index(item))
                    recallornot[sp]=1
                else: # repetition
                    recall_sp.append(this_list.index(item)+100)
            except:   # intrusion 
                recall_sp.append(-99)

        recalls_sp.append(recall_sp)    
    return presents,recalls,recalls_sp




# get stopping probability at each output position 
def get_stopping_prob(recalls,list_length):
    lengths = [len(x) for x in recalls]
    counts = np.bincount(lengths)
    probs = []
    max_recall = np.max(lengths)
    for i in range(list_length):
        if i >= max_recall-1:
            prob = 1
        else:    
            prob = (counts[i]+1)/(counts[i]+np.sum(counts[i+1:])+1)
        probs.append(prob)
        
    return probs


# spc and pfr
def get_spc_pfr(recalls_sp,list_length):
    # this function returns serial position curve (spc) and prob of first recall (pfr) 
    # recalls_sp: list of lists of serial positions    
    num_trial = len(recalls_sp)
    spc = np.zeros(list_length)
    pfr = np.zeros(list_length)
    for i,recall_sp in enumerate(recalls_sp):
        recallornot = np.zeros(list_length)
        for j,item in enumerate(recall_sp):           
            if 0 <= item <100:
                #print(item)
                if recallornot[item]==0:
                    spc[item] += 1
                    if j==0:
                        pfr[item] += 1 
                    recallornot[item]=1    

    return spc/num_trial, pfr/num_trial




# crp
def get_crp(recalls_sp,lag_examine,ll):
    # this function returns conditional response probability 
    # recalls_sp: list of lists of serial positions
    # lag_examine: range of lags to examine, can it to 4 usually
    # ll: list length
    #exclude = -1 # if need to exclude first few output positions
    possible_counts = np.zeros(2*lag_examine+1)
    actual_counts = np.zeros(2*lag_examine+1)
    for i in range(len(recalls_sp)):
        recallornot = np.zeros(ll)
        for j in range(len(recalls_sp[i]))[:-1]:
            #if j>exclude:
                sp1 = recalls_sp[i][j]
                sp2 = recalls_sp[i][j+1]
                if 0 <= sp1 <100:
                    recallornot[sp1] = 1
                    if 0 <= sp2 <100:
                        lag = sp2 - sp1
                        if np.abs(lag) <= lag_examine:
                            actual_counts[lag+lag_examine] += 1
                        for k,item in enumerate(recallornot):    
                            if item==0:
                                lag = k - sp1
                                if np.abs(lag) <= lag_examine:
                                    possible_counts[lag+lag_examine] += 1                   
    crp = [(actual_counts[i]+1)/(possible_counts[i]+1) for i in range(len(actual_counts))]
    crp[lag_examine] = 0
    return crp


# crp for same or different categories
def get_crp_cat(recalls_sp,data_cat,lag_examine,ll,same):
    possible_counts = np.zeros(2*lag_examine+1)
    actual_counts = np.zeros(2*lag_examine+1)
    for i in range(len(recalls_sp)):
        recallornot = np.zeros(ll)
        for j in range(len(recalls_sp[i]))[:-1]:
            sp1 = recalls_sp[i][j]
            sp2 = recalls_sp[i][j+1]
            if 0 <= sp1 <100:
                recallornot[sp1] = 1
                if 0 <= sp2 <100:
                    lag = sp2 - sp1
                    if np.abs(lag) <= lag_examine and data_cat[i][sp1]==data_cat[i][sp2] and same == 1:
                        actual_counts[lag+lag_examine] += 1
                    if np.abs(lag) <= lag_examine and data_cat[i][sp1]!=data_cat[i][sp2] and same != 1:
                        actual_counts[lag+lag_examine] += 1
                    for k,item in enumerate(recallornot):    
                        if item==0:
                            lag = k - sp1
                            if np.abs(lag) <= lag_examine:
                                possible_counts[lag+lag_examine] += 1    
    crp = [(actual_counts[i]+1)/(possible_counts[i]+1) for i in range(len(actual_counts))]
    crp[lag_examine] = 0
    return crp

# get semantic LSAs for different lags
def get_semantic(data_pres,data_rec,lag_number,LSA_mat):
    # recode data into lists without 0-paddings; convert to python indexing from matlab indexing by -1
    recalls = [[int(y)-1 for y in list(x) if y>0] for x in list(data_rec)]
    LSAs = []
    for l in lag_number:
        LSA = [0]
        for i in range(len(recalls)):
            if len(recalls[i]) > l:
                for j in range(len(recalls[i]))[:-l]:
                    sim = LSA_mat[recalls[i][j],recalls[i][j+l]]
                    LSA.append(sim)   
        LSAs.append(np.mean(LSA))     
    return LSAs

def normed_RMSE(A,B): 
# return the normed root mean square error
# normed by the scale of A (the target dataset)
    mse = np.mean(np.power(np.subtract(A,B),2))
    normed_rmse = np.sqrt(mse)/(np.max(A)-np.min(A))
    return normed_rmse

def normed_RMSE_singlevalue(A,B): 
# return the normed root mean square error
# normed by the value of A (the target dataset)
    mse = np.mean(np.power(np.subtract(A,B),2))
    normed_rmse = np.sqrt(mse)/np.abs(5*A+0.01)
    return normed_rmse
    

def simualteCMRprob(N, ll, lag_examine, data_path, data_rec_path,subjects_path, LSA_mat,param_dict, data_spc, data_pfr, data_crp):    
    ###############
    #
    #   Simulate using CMR2 
    #
    ###############  

    RMSEs = []
    CMR_spcs = []
    CMR_pfrs = []
    CMR_crps = []
    CMR_intrusions = []
    CMR_crp_sames = []
    CMR_crp_diffs = []
    CMR_LSAs = []
    CMR_sprobs = []
    data_pres = np.loadtxt(data_path, delimiter=',')

    for itr in range(N):
        # run CMR on the data
        #data_pres = np.loadtxt(data_path, delimiter=',')
        #data_recs = np.loadtxt(data_rec_path, delimiter=',')
        #resp, times,_ = CMR_prob.run_CMR2_singleSubj(0, data_pres, data_recs,LSA_mat,param_dict)
        resp, times,_ = CMR_prob.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)
        
        # recode simulations  
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        CMR_spc,CMR_pfr = get_spc_pfr(CMR_sp,ll)
        CMR_crp = get_crp(CMR_sp,lag_examine,ll)
        
        ###############
        #
        #   Calculate fit
        #
        ###############  
        RMSE = 0    
        # include SPC
        RMSE += normed_RMSE(data_spc, CMR_spc)
        CMR_spcs.append(CMR_spc)

        # include FPR   
        RMSE += normed_RMSE(data_pfr, CMR_pfr)
        CMR_pfrs.append(CMR_pfr)

        # include CRP
        CMR_crp = get_crp(CMR_sp,lag_examine,ll)
        RMSE += normed_RMSE(data_crp, CMR_crp) 
        CMR_crps.append(CMR_crp)

        
        RMSEs.append(RMSE) 
        
    if N ==1: # for real-time bayesian optimization
        return RMSE
    else: # for plotting   
       # return RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_crp_same,CMR_crp_sames,data_crp_diff,CMR_crp_diffs,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs
        return RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs

###############
#
#   This is the simulation function for individual PEERS subject
#
###############     
def obj_func_subj(beta_enc,beta_rec,gamma_fc,phi_s,phi_d,epsilon_d,k,primacy):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    ll = 16
    LSA_mat, data_path, data_rec_path, subjects_path = load_structure(-1) # where data structure is from

    # current set up is for fitting the non-emot version of the model
    param_dict = {  
        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec': beta_rec,           # rate of context drift during recall
        'beta_rec_post': 1.0,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': gamma_fc,  # learning rate, feature-to-context
        'gamma_cf': 1.0,  # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 0.0,

        's_cf': 0.0,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': phi_s,      # primacy parameter
        'phi_d': phi_d,      # primacy parameter

        'epsilon_s': 0.0,      # baseline activiation for stopping probability 
        'epsilon_d': epsilon_d,        # scale parameter for stopping probability 
        

        'k': k,   # luce choice rule scale 
        
        'cue_position':-1, # no initial cue
        'primacy':primacy, # control whether to start from the beginning (1) or the end (0)
        'enc_rate':1, # encoding rate
    }
    # run probCMR on the data
    resp, times,_ = CMR_prob.run_CMR2(
            recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)
    data_pres = np.loadtxt(data_path, delimiter=',')
    _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
    acc = np.mean([len(np.nonzero(x)[0]) for x in resp])
    return CMR_sp,acc

    
###############
#
#   This is the objective function for fitting CMR to behavioral patterns in free recall dataset
#
###############     
def obj_func_prob(beta_enc,beta_rec,gamma_fc,phi_s,phi_d,epsilon_d,k):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    N, ll, lag_examine,data_id = set_parameter_prob()
    LSA_mat, data_path, data_rec_path, subjects_path = load_structure(data_id) # where data structure is from
    data_spc, data_pfr, data_crp = load_patterns(data_id) # where summary behavioral data is from
    
    # current set up is for fitting the non-emot version of the model
    param_dict = {  
        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec': beta_rec,           # rate of context drift during recall
        'beta_rec_post': 1.0,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': gamma_fc,  # learning rate, feature-to-context
        'gamma_cf': 1.0,  # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 0.0,

        's_cf': 0.0,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': phi_s,      # primacy parameter
        'phi_d': phi_d,      # primacy parameter

        'epsilon_s': 0.0,      # baseline activiation for stopping probability 
        'epsilon_d': epsilon_d,        # scale parameter for stopping probability 
        

        'k': k,   # luce choice rule scale 
        
        'cue_position':-1, # no initial cue
        'primacy':-1, # always start recalling from the end
        'enc_rate':1, # encoding rate
    }
    # run probCMR on the data
    if N==0: # recall_mode = 0 to simulate based on parameters
        CMR_sps = []
        for i in range(1):
            resp, times,_ = CMR_prob.run_CMR2(
            recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)
            data_pres = np.loadtxt(data_path, delimiter=',')
            _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
            CMR_sps.extend(CMR_sp)
        return CMR_sps
    if N==1:
        RMSEs = []
        for i in range(5):
            RMSE = simualteCMRprob(N, ll, lag_examine, data_path, data_rec_path,subjects_path, LSA_mat, param_dict, data_spc, data_pfr, data_crp)
            RMSEs.append(RMSE)
        return -np.mean(RMSEs)
    else:
        RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs = simualteCMRprob(N, ll, lag_examine, data_path, data_rec_path, subjects_path, LSA_mat, param_dict, data_spc, data_pfr, data_crp)
        plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs)
        return -RMSE    
    

###############
#
#   This is the objective function for simulating full CMR2 
#
###############     
def obj_func_fullCMR2(beta_enc,beta_rec,gamma_fc,cue_position):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    LSA_path = 'data/K02_LSA.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)
    data_path = 'data/K02_data.txt'
    data_rec_path = 'data/K02_recs.txt'
    subjects_path = 'data/K02_subject_ids.txt'
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')
    N, ll, lag_examine,data_id = set_parameter_prob()

    param_dict = {

        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec': beta_rec,           # rate of context drift during recall
        'beta_rec_post': 0.51,#0.802543,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': gamma_fc,  # learning rate, feature-to-context
        'gamma_cf': 1,#0.895261,  # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 0,#1 - 0.895261,
 
        #'gamma_fc': 0.425064,  # learning rate, feature-to-context
        #'gamma_cf': 0.425064,  # learning rate, context-to-feature
        #'scale_fc': 1 - 0.425064,
        #'scale_cf': 1 - 0.425064,
        
        

        's_cf': 0,#1.292411,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': 1.808899,#1.408899,      # primacy parameter
        'phi_d': 0.989567,      # primacy parameter

        'nlists_for_accumulator': 4,    # only the list-length * 4
                                        # strongest items compete

        'kappa': 0.312686,      # item activation decay rate
        'lamb': 0.12962,        # lateral inhibition

        'eta': 0.392847,        # width of noise gaussian

        'rec_time_limit': 30000,  # ms allowed for recall post-study list
        'dt': 10.0,               # time scale for leaky accumulator
        'dt_tau': 0.01,
        'sq_dt_tau': 0.1000,

        'c_thresh': 0.073,#0.073708,   # threshold of context similarity
                                # required for an item to be recalled

        'omega': 11.894106,     # repetition prevention parameter
        'alpha': 0.678955,      # repetition prevention parameter
        
        'cue_position': cue_position, # different initial cues provided
    }
        
    # run probCMR on the data

    resp, times = CMR2.run_CMR2(
    LSA_mat=LSA_mat, data_path=data_path,
    params=param_dict, subj_id_path=subjects_path, sep_files=False)


    data_pres = np.loadtxt(data_path, delimiter=',')
    _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
    
      
    CMR_sp_flatten = [item for sublist in CMR_sp for item in sublist]
    corrects = [1 for x in CMR_sp_flatten if x>-1 and x<99]
    acc = len(corrects)/len(CMR_sp)
    if N == 0:
        _,data_recalls,data_sp = data_recode(data_pres, data_rec)  
        return CMR_sp,data_sp,acc
    else:
        return acc

def obj_func_fitfullCMR2(beta_enc,beta_rec,beta_rec_post,gamma_fc,phi_s,phi_d,kappa,lamb,eta,omega,alpha,c_thresh):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    LSA_path = 'data/K02_LSA.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)
    data_path = 'data/K02_data.txt'
    data_rec_path = 'data/K02_recs.txt'
    subjects_path = 'data/K02_subject_ids.txt'
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')
    N, ll, lag_examine,data_id = set_parameter_prob()

    param_dict = {  
        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec': beta_rec,           # rate of context drift during recall
        'beta_rec_post': beta_rec_post,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': gamma_fc,  # learning rate, feature-to-context
        'gamma_cf': 1,  # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 1 - 1,

        's_cf': 0.0,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': phi_s,      # primacy parameter
        'phi_d': phi_d,      # primacy parameter

        'nlists_for_accumulator': 4,    # only the list-length * 4
                                        # strongest items compete

        'kappa': kappa,      # item activation decay rate
        'lamb': lamb,        # lateral inhibition

        'eta': eta,        # width of noise gaussian

        'rec_time_limit': 30000,  # ms allowed for recall post-study list
        'dt': 10.0,               # time scale for leaky accumulator
        'dt_tau': 0.01,
        'sq_dt_tau': 0.1000,

        'c_thresh': c_thresh,   # threshold of context similarity
                                # required for an item to be recalled

        'omega': omega,     # repetition prevention parameter
        'alpha': alpha,      # repetition prevention parameter   
        
        'cue_position': 10, # different initial cues provided
    }
        
    # run probCMR on the data

    resp, times = CMR2.run_CMR2(
    LSA_mat=LSA_mat, data_path=data_path,
    params=param_dict, subj_id_path=subjects_path, sep_files=False)


    # compare results from dataset and simulation
    _,data_recalls,data_sp = data_recode(data_pres, data_rec)  
    _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
    data_spc,data_pfr = get_spc_pfr(data_sp,ll)
    CMR_spc,CMR_pfr = get_spc_pfr(CMR_sp,ll)

    ###############
    #
    #   Calculate fit
    #
    ###############  
    RMSE = 0    
    # include SPC
    RMSE += normed_RMSE(data_spc, CMR_spc)
        
    # include FPR   
    RMSE += normed_RMSE(data_pfr, CMR_pfr)
        
    # include CRP
    data_crp = get_crp(data_sp,lag_examine,ll)
    CMR_crp = get_crp(CMR_sp,lag_examine,ll)
    RMSE += normed_RMSE(data_crp, CMR_crp) 

    # repeats and intrusion
    data_sp_flatten = [item for sublist in data_sp for item in sublist]
    CMR_sp_flatten = [item for sublist in CMR_sp for item in sublist]
    repeats_data = [1 for x in data_sp_flatten if x>100]
    repeats_model = [1 for x in CMR_sp_flatten if x>100]
    intrusion_data = [1 for x in data_sp_flatten if x<-90]
    intrusion_model = [1 for x in CMR_sp_flatten if x<-90]
    RMSE += normed_RMSE_singlevalue(len(repeats_data), len(repeats_model))
    RMSE += normed_RMSE_singlevalue(len(intrusion_data), len(intrusion_model))
    
    if N==0:
        return CMR_sp,data_sp
    else:
        return -RMSE
###############
#
#   This is the objective function for optimizing CMR given different starting cue
#
###############     

def obj_func_prob_reminder(beta_enc,beta_rec,gamma_fc,cue_position):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    N, ll, lag_examine,data_id = set_parameter_prob()
    LSA_mat, data_path, data_rec_path, subjects_path = load_structure(data_id) # where data structure is from
    data_spc, data_pfr, data_crp = load_patterns(data_id) # where summary behavioral data is from
    
    # current set up is for fitting the non-emot version of the model
    param_dict = {  
        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec': beta_rec,           # rate of context drift during recall
        'beta_rec_post': 1,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': gamma_fc,  # learning rate, feature-to-context
        'gamma_cf': 1,  # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 0,

        's_cf': 0,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0,            # scales influence of semantic similarity
                                # on M_FC matrix.

        'phi_s': 0,       # no primacy 
        'phi_d': 10,      # no primacy

        'epsilon_s': 0,      # baseline activiation for stopping probability 
        'epsilon_d': 2.1130070142836126, #2.738934338758688,        # scale parameter for stopping probability (top: 3.007583053727949; bottom: 2.1130070142836126)
        

        'k': 4.40574248709815,#5.380182482069175,   # luce choice rule scale (top: 5.4694949582292365; bottom: 4.40574248709815)
        
        'cue_position': cue_position, # different initial cues provided
        'primacy':-1, # no primacy constrainst
        'enc_rate':1, # encoding rate
    }
    acc = 0
    if N==0:       
        resp, times,_ = CMR_prob.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)
        data_pres = np.loadtxt(data_path, delimiter=',')
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        acc = np.mean([len(np.nonzero(x)[0]) for x in resp])
        return CMR_sp,acc
    elif N==1:
        temp = []
        for i in range(5):
            resp, times,_ = CMR_prob.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)   
            temp.append(np.mean([len(np.nonzero(x)[0]) for x in resp]))
        acc = np.mean(temp)    
        return acc
    else:     
        RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs = simualteCMRprob(N, ll, lag_examine, data_path, data_rec_path, subjects_path, LSA_mat, param_dict, data_spc, data_pfr, data_crp)
        plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs)
        return -RMSE    


###############
#
#   This is the objective function for optimizing CMR given different starting cue
#
###############     

def obj_func_prob_reminder_noise(beta_enc,beta_rec,gamma_fc,cue_position,k,enc_rate):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    N, ll, lag_examine,data_id = set_parameter_prob()
    LSA_mat, data_path, data_rec_path, subjects_path = load_structure(data_id) # where data structure is from
    data_spc, data_pfr, data_crp = load_patterns(data_id) # where summary behavioral data is from
    
    # current set up is for fitting the non-emot version of the model
    param_dict = {  
        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec': beta_rec,           # rate of context drift during recall
        'beta_rec_post': 1,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': gamma_fc,  # learning rate, feature-to-context
        'gamma_cf': 1,  # learning rate, context-to-feature
        'scale_fc': 1 - gamma_fc,
        'scale_cf': 0,

        's_cf': 0,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0,            # scales influence of semantic similarity
                                # on M_FC matrix.

        'phi_s': 0,       # no primacy 
        'phi_d': 10,      # no primacy

        'epsilon_s': 0,      # baseline activiation for stopping probability 
        'epsilon_d': 2.738934338758688,        # scale parameter for stopping probability (small: 0.5; large: 10)
        

        'k': k,   # luce choice rule scale (5.380182482069175)
        
        'cue_position': cue_position, # different initial cues provided
        'primacy':-1, # no primacy constrainst
        'enc_rate':enc_rate, # encoding rate
    }
    acc = 0
    if N==0:       
        resp, times,_ = CMR_prob.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)
        data_pres = np.loadtxt(data_path, delimiter=',')
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        acc = np.mean([len(np.nonzero(x)[0]) for x in resp])
        return CMR_sp,acc
    elif N==1:
        temp = []
        for i in range(5):
            resp, times,_ = CMR_prob.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)   
            temp.append(np.mean([len(np.nonzero(x)[0]) for x in resp]))
        acc = np.mean(temp)    
        return acc
    else:     
        RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs = simualteCMRprob(N, ll, lag_examine, data_path, data_rec_path, subjects_path, LSA_mat, param_dict, data_spc, data_pfr, data_crp)
        plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs)
        return -RMSE    




    
    
def obj_func_prob_primacy_twostage(beta_enc,beta_rec_bg,beta_rec_ed,gamma_fc_bg,gamma_fc_ed,primacy):  
    """Error function that we want to minimize"""
    ###############
    #
    #   Load parameters and path
    #
    ###############
    N, ll, lag_examine,data_id = set_parameter_prob()
    LSA_mat, data_path, data_rec_path, subjects_path = load_structure(data_id) # where data structure is from
    data_spc, data_pfr, data_crp = load_patterns(data_id) # where summary behavioral data is from
    
    # current set up is for fitting the non-emot version of the model
    param_dict = {  
        'beta_enc': beta_enc,           # rate of context drift during encoding
        'beta_rec_bg': beta_rec_bg,           # rate of context drift during recall
        'beta_rec_ed': beta_rec_ed,           # rate of context drift during recall
        'beta_rec_post': 1,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc_bg': gamma_fc_bg,  # learning rate, feature-to-context
        'gamma_fc_ed': gamma_fc_ed,  # learning rate, feature-to-context
        'gamma_cf': 1,  # learning rate, context-to-feature

        's_cf': 0,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0,            # scales influence of semantic similarity
                                # on M_FC matrix.

        'phi_s': 0,       # no primacy 
        'phi_d': 10,      # no primacy

        'epsilon_s': 0,      # baseline activiation for stopping probability 
        'epsilon_d': 2.738934338758688,        # scale parameter for stopping probability (small: 0.5; large: 10)
        

        'k': 5.380182482069175,   # luce choice rule scale (4.857399308063701; small noise: 10; large noise: 1)
        
        'cue_position': -1, # different initial cues provided
        'primacy':primacy, # no primacy constrainst
        'enc_rate':1, # encoding rate
    }
    acc = 0
    if N==0:       
        resp, times,_ = CMR_prob_twostage.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)
        data_pres = np.loadtxt(data_path, delimiter=',')
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        acc = np.mean([len(np.nonzero(x)[0]) for x in resp])
        return CMR_sp,acc
    elif N==1:
        temp = []
        for i in range(5):
            resp, times,_ = CMR_prob_twostage.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)   
            temp.append(np.mean([len(np.nonzero(x)[0]) for x in resp]))
        acc = np.mean(temp)    
        return acc
    



     
    
def plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs):
#def plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_crp_same,CMR_crp_sames,data_crp_diff,CMR_crp_diffs,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs):

        ###############
    #
    #   Plot results
    #
    ###############
    #plt.rcParams['figure.figsize'] = (16,8)
    plt.rcParams['figure.figsize'] = (15,4)
    #plt.rcParams['figure.figsize'] = (12,4)
    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']



    #plt.subplot(2,3,1)
    plt.subplot(1,4,1)
    #plt.plot(CMR_spc,'coral')
    plt.plot(data_spc,'grey')
    data=np.asarray(CMR_spcs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_spc)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_spc)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('SPC')

    #plt.subplot(2,3,2)
    plt.subplot(1,4,2)
    #plt.plot(CMR_pfr,'coral')
    plt.plot(data_pfr,'grey')
    data=np.asarray(CMR_pfrs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_pfr)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_pfr)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('PFR')

    plt.subplot(1,4,3)
    #plt.subplot(2,3,3)
    plt.plot(data_crp,'*')
    #plt.plot(CMR_crp,'o')
    data=np.asarray(CMR_crps)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_crp)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_crp)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('CRP')
    
    plt.subplot(1,4,4)
    plt.plot(data_LSA,'*')
    #plt.plot(CMR_crp_diff,'o')
    data=np.asarray(CMR_LSAs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_LSA)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_LSA)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Semantic clustering')
    #plt.savefig('./Figs/fitpatterns_1109_pri_K02.png') 
    
'''
    plt.subplot(2,3,4)
    plt.plot(data_crp_same,'*')
    #plt.plot(CMR_crp_same,'o')
    data=np.asarray(CMR_crp_sames)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_crp_same)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_crp_same)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('CRP-within')
    #plt.ylim(0,0.22)

    plt.subplot(2,3,5)
    plt.plot(data_crp_diff,'*')
    #plt.plot(CMR_crp_diff,'o')
    data=np.asarray(CMR_crp_diffs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_crp_diff)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_crp_diff)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('CRP-across')
    #plt.ylim(0,0.22)

    plt.subplot(2,3,6)
    #plt.plot(CMR_spc,'coral')
    plt.plot(data_sprob,'grey')
    data=np.asarray(CMR_sprobs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_sprob)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_sprob)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Stopping Prob') 
    
 
    plt.subplot(2,3,4)
    plt.plot(data_intrusion,'o')
    y = np.mean(CMR_intrusions)
    error = np.std(CMR_intrusions)
    error=0.1
    plt.plot(0, y, 'k*', color='#CC4F1B')
    plt.errorbar(0,y,error)
    plt.title('Intrusion')
    plt.ylim(0,1) 

    '''
       
    
def simualteCMR(simple,model_num,param_dict, patterns, N, ll, lag_examine,LSA_mat, data_path, data_rec_path, subjects_path):    
    ###############
    #
    #   Simulate using CMR2 
    #
    ###############  
    
    
    RMSEs = []
    CMR_spcs = []
    CMR_pfrs = []
    CMR_crps = []
    CMR_intrusions = []
    CMR_crp_sames = []
    CMR_crp_diffs = []
    CMR_LSAs = []
    CMR_sprobs = []
    
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')
    #data_cat = np.loadtxt(data_cat_path, delimiter=',')
    
    _,data_recalls,data_sp = data_recode(data_pres, data_rec)
    data_spc,data_pfr = get_spc_pfr(data_sp,ll)
    #data_crp_same = get_crp_cat(data_sp,data_cat,lag_examine,ll,1)
    #data_crp_diff = get_crp_cat(data_sp,data_cat,lag_examine,ll,0)
    
    for itr in range(N):
        # run CMR2 on the data
        if simple==1:
            resp, times,_ = CMR2_simple.run_CMR2(
            recall_mode=0, model_num = model_num, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)
        elif simple==2: # with repetitions
            resp, times,_ = CMR2_rep.run_CMR2(
            recall_mode=0, model_num = model_num, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)            
        elif simple==3:
            resp, times,_ = CMR_prob.run_CMR2(
            recall_mode=0,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)
        else:
            resp, times, cnt = CMR2.run_CMR2(
                    LSA_mat=LSA_mat, data_path=data_path,
                    params=param_dict, subj_id_path=subjects_path, sep_files=False)

        # recode simulations  
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        CMR_spc,CMR_pfr = get_spc_pfr(CMR_sp,ll)

        ###############
        #
        #   Calculate fit
        #
        ###############  
        RMSE = 0    
        if 0 in patterns: # include SPC
            RMSE += normed_RMSE(data_spc, CMR_spc)
            CMR_spcs.append(CMR_spc)

        if 1 in patterns:  # include FPR   
            RMSE += normed_RMSE(data_pfr, CMR_pfr)
            CMR_pfrs.append(CMR_pfr)

        if 2 in patterns:  # include CRP
            data_crp = get_crp(data_sp,lag_examine,ll)
            CMR_crp = get_crp(CMR_sp,lag_examine,ll)
            RMSE += normed_RMSE(data_crp, CMR_crp) 
            CMR_crps.append(CMR_crp)

        if 3 in patterns:  # include intrusion rate       
            data_allrecalls = list(itertools.chain.from_iterable(data_sp))
            data_intrusion = data_allrecalls.count(-99)/(len(data_allrecalls)+1)   
            CMR_allrecalls = list(itertools.chain.from_iterable(CMR_sp))
            CMR_intrusion = CMR_allrecalls.count(-99)/(len(CMR_allrecalls)+1) 
            RMSE += normed_RMSE_singlevalue(data_intrusion, CMR_intrusion)    
            CMR_intrusions.append(CMR_intrusion)

        if 4 in patterns: # crp for within-categories
            data_crp_same = get_crp_cat(data_sp,data_cat,lag_examine,ll,1)
            CMR_crp_same = get_crp_cat(CMR_sp,data_cat,lag_examine,ll,1)
            RMSE += normed_RMSE(data_crp_same, CMR_crp_same) 
            CMR_crp_sames.append(CMR_crp_same)
            

        if 5 in patterns: # crp for across-categories
            data_crp_diff = get_crp_cat(data_sp,data_cat,lag_examine,ll,0)
            CMR_crp_diff = get_crp_cat(CMR_sp,data_cat,lag_examine,ll,0)
            RMSE += normed_RMSE(data_crp_diff, CMR_crp_diff) 
            CMR_crp_diffs.append(CMR_crp_diff)
            
        if 6 in patterns: # semantic clustering
            data_LSA = get_semantic(data_pres,data_rec,[1,2,3,4],LSA_mat)
            CMR_LSA = get_semantic(data_pres,resp,[1,2,3,4],LSA_mat)
            RMSE += normed_RMSE(data_LSA, CMR_LSA) 
            CMR_LSAs.append(CMR_LSA)  
            
        if 7 in patterns: # semantic clustering
            data_sprob = get_stopping_prob(data_recalls,ll)
            CMR_sprob = get_stopping_prob(CMR_recalls,ll)
            RMSE += normed_RMSE(data_sprob, CMR_sprob) 
            CMR_sprobs.append(CMR_sprob)  
        
        RMSEs.append(RMSE) 
        
    if N ==1: # for real-time bayesian optimization
        return RMSE
    else: # for plotting   
       # return RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_crp_same,CMR_crp_sames,data_crp_diff,CMR_crp_diffs,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs
        return RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs

