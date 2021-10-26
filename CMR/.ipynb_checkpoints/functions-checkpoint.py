import numpy as np
import os
import probCMR as CMR2_simple
import matplotlib.pyplot as plt
import itertools

# load data
def load_data():
    LSA_path = 'datafile/autolab_GloVe.txt'
    data_path = 'datafile/autolab_pres.txt'
    data_rec_path = 'datafile/autolab_recs.txt'
    data_cat_path = 'datafile/autolab_pres_cats.txt'
    subjects_path = 'datafile/autolab_subject_id.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)
    return LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path


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

# get semantic associations for different lags
def get_semantic(data_pres,data_rec,lag_number,LSA_mat):
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
    normed_rmse = np.sqrt(mse)#/np.abs(A)
    return normed_rmse
    
    
    
def plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs):
        ###############
    #
    #   Plot results
    #
    ###############
    plt.rcParams['figure.figsize'] = (15,4)

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

    plt.subplot(1,4,1)
    plt.plot(data_spc,'grey')
    data=np.asarray(CMR_spcs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_spc)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_spc)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('SPC')

    plt.subplot(1,4,2)
    plt.plot(data_pfr,'grey')
    data=np.asarray(CMR_pfrs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_pfr)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_pfr)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('PFR')

    plt.subplot(1,4,3)
    plt.plot(data_crp,'*')
    data=np.asarray(CMR_crps)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_crp)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_crp)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('CRP')
    
    plt.subplot(1,4,4)
    plt.plot(data_LSA,'*')
    data=np.asarray(CMR_LSAs)
    y = np.mean(data,0)
    error = np.std(data,0)
    plt.plot(y)
    plt.plot(range(len(data_LSA)), y, 'k', color='#CC4F1B')
    plt.fill_between(range(len(data_LSA)), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Semantic clustering')

def simualteCMR(param_dict, N, ll, lag_examine,LSA_mat, data_path, data_rec_path, subjects_path):    
    ###############
    #
    #   Simulate probCMR for N times to pool behavioral data before plotting 
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
    _,data_recalls,data_sp = data_recode(data_pres, data_rec)
    data_spc,data_pfr = get_spc_pfr(data_sp,ll)

    for itr in range(N):
        # run CMR2 on the data
        resp, times,_ = CMR2_simple.run_CMR2(
        recall_mode=0, LSA_mat=LSA_mat, data_path=data_path,rec_path = data_rec_path,
        params=param_dict, subj_id_path=subjects_path, sep_files=False)

        # recode simulations  
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        CMR_spc,CMR_pfr = get_spc_pfr(CMR_sp,ll)

        ###############
        #
        #   Calculate fit and behavioral data
        #
        ###############  
        RMSE = 0    
        # SPC
        RMSE += normed_RMSE(data_spc, CMR_spc)
        CMR_spcs.append(CMR_spc)

        # PFR
        RMSE += normed_RMSE(data_pfr, CMR_pfr)
        CMR_pfrs.append(CMR_pfr)

        # CRP
        data_crp = get_crp(data_sp,lag_examine,ll)
        CMR_crp = get_crp(CMR_sp,lag_examine,ll)
        RMSE += normed_RMSE(data_crp, CMR_crp) 
        CMR_crps.append(CMR_crp)

        # Intrusion       
        data_allrecalls = list(itertools.chain.from_iterable(data_sp))
        data_intrusion = data_allrecalls.count(-99)/(len(data_allrecalls)+1)   
        CMR_allrecalls = list(itertools.chain.from_iterable(CMR_sp))
        CMR_intrusion = CMR_allrecalls.count(-99)/(len(CMR_allrecalls)+1) 
        RMSE += normed_RMSE_singlevalue(data_intrusion, CMR_intrusion)    
        CMR_intrusions.append(CMR_intrusion)
   
       # semantic clustering
        data_LSA = get_semantic(data_pres,data_rec,[1,2,3,4],LSA_mat)
        CMR_LSA = get_semantic(data_pres,resp,[1,2,3,4],LSA_mat)
        RMSE += normed_RMSE(data_LSA, CMR_LSA) 
        CMR_LSAs.append(CMR_LSA)  
            
        # stopping probability
        data_sprob = get_stopping_prob(data_recalls,ll)
        CMR_sprob = get_stopping_prob(CMR_recalls,ll)
        RMSE += normed_RMSE(data_sprob, CMR_sprob) 
        CMR_sprobs.append(CMR_sprob)  
        
        RMSEs.append(RMSE) 
    return RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs



def model_probCMR(N, ll, lag_examine):  
    """Error function that we want to minimize"""
    ###############
    #
    #   simulate free recall data
    #
    # N: 0 - obtain lists of recall, in serial positions
    # N: 1 - obtain likelihood given data
    # N >1 - plot behavioral data with error bar with N being the number of times in simulations
    #
    # ll: list length (ll=16)
    #
    # lag_examine: lag used in plotting CRP
    #
    ###############
    LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path = load_data()
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')

    # current set up is for fitting the non-emot version of the model
    # model parameters
    param_dict = {

        'beta_enc':  0.3187893806764954,           # rate of context drift during encoding
        'beta_rec':  0.9371120781560975,           # rate of context drift during recall
        'beta_rec_post': 1,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': 0.1762454837715133,  # learning rate, feature-to-context
        'gamma_cf': 0.5641689110824742,  # learning rate, context-to-feature
        'scale_fc': 1 - 0.1762454837715133,
        'scale_cf': 1 - 0.5641689110824742,


        's_cf': 0.8834467032413329,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': 2.255110764387116,      # primacy parameter
        'phi_d': 0.4882977227079478,      # primacy parameter


        'epsilon_s': 0.0,      # baseline activiation for stopping probability 
        'epsilon_d': 2.2858636787518285,        # scale parameter for stopping probability 

        'k':  6.744153399759922,        # scale parameter in luce choice rule during recall

    }
    
    # run probCMR on the data
    if N==0: # recall_mode = 0 to simulate based on parameters
        resp, times,_ = CMR2_simple.run_CMR2(
            recall_mode=0,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)   
        _,CMR_recalls,CMR_sp = data_recode(data_pres, resp)
        return CMR_sp
    if N==1:# recall_mode = 1 to calculate likelihood of data based on parameters
        _, _,lkh = CMR2_simple.run_CMR2(
            recall_mode=1,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
            params=param_dict, subj_id_path=subjects_path, sep_files=False)   
        return lkh   
    else:
        RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs = simualteCMR(param_dict, N, ll, lag_examine, LSA_mat, data_path, data_rec_path, subjects_path)
        plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs)
        return -RMSE
    
 