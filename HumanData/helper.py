import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import scipy.io
from pathlib import Path
from scipy.stats import mannwhitneyu
from scipy import stats

# statistical tests
def stats_test(forward1,forward2):
    # non-parametric
    stat, p = mannwhitneyu(forward1, forward2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
        
    # indpendent t-test
    print(stats.ttest_ind(forward1, forward2))  
    
# Calculate transitions
def calc_trans(recalls_sp,ll):
    trans = np.zeros((ll,ll,ll)) # output,from,to
    for i in range(len(recalls_sp)):
        temp = [x for x in recalls_sp[i] if x>-1]
        for j in range(np.min([ll-1,len(temp)-1])):
            trans[j][temp[j]][temp[j+1]] = trans[j][temp[j]][temp[j+1]]+1
    return trans   

# spc and pfr
def get_spc_pfr(recalls_sp,list_length):
    num_trial = len(recalls_sp)
    spc = np.zeros(list_length)
    pfr = np.zeros(list_length)
    for i,recall_sp in enumerate(recalls_sp):
        recallornot = np.zeros(list_length)
        for j,item in enumerate(recall_sp):           
            if 0 <= item <list_length:
                #print(item)
                if recallornot[item]==0:
                    spc[item] += 1
                    if j==0:
                        pfr[item] += 1 
                    recallornot[item]=1    

    return spc/num_trial, pfr/num_trial


# crp
def get_crp_2(recalls_sp,lag_examine,ll,exclude):
    possible_counts = np.zeros(2*lag_examine+1)
    actual_counts = np.zeros(2*lag_examine+1)
    TCEs = []
    for i in range(len(recalls_sp)):
        recallornot = np.zeros(ll)
        for j in range(len(recalls_sp[i]))[:-1]:
            if j>exclude:
                sp1 = recalls_sp[i][j]
                sp2 = recalls_sp[i][j+1]
                if 0 <= sp1 <100:
                    recallornot[sp1] = 1
                    if 0 <= sp2 <100:
                        lag = sp2 - sp1
                        N = 1
                        if np.abs(lag) <= lag_examine:
                            actual_counts[lag+lag_examine] += 1
                        actual_lag = np.abs(sp2 - sp1)
                        possible_lags = [actual_lag]

                        for k,item in enumerate(recallornot):    
                            if item==0: # not recalled
                                N = N + 1
                                lag = k - sp1
                                possible_lags.append(np.abs(lag))
                                if np.abs(lag) <= lag_examine:
                                    possible_counts[lag+lag_examine] += 1   
                        sorted_lags = np.sort(possible_lags)  

                        ranks = [r for r,x in enumerate(sorted_lags-actual_lag) if x==0]  
                        if N>1:
                            TCE = (np.mean(ranks))/(N-1)   
                            TCEs.append(TCE)                
    crp = [(actual_counts[i]+1)/(possible_counts[i]+1) for i in range(len(actual_counts))]
    crp[lag_examine] = 0
    return crp, np.mean(TCEs)



def get_crp(recalls_sp,lag_examine,ll,exclude):
    possible_counts = np.zeros(2*lag_examine+1)
    actual_counts = np.zeros(2*lag_examine+1)
    TCEs = []
    for i in range(len(recalls_sp)):
        recallornot = np.zeros(ll)
        for j in range(len(recalls_sp[i]))[:-1]:
            if j>exclude:
                sp1 = recalls_sp[i][j]
                sp2 = recalls_sp[i][j+1]
                if 0 <= sp1 <100:
                    recallornot[sp1] = 1
                    if 0 <= sp2 <100:
                        lag = sp2 - sp1
                        N = 1
                        if np.abs(lag) <= lag_examine:
                            actual_counts[lag+lag_examine] += 1
                        actual_lag = np.abs(sp2 - sp1)
                        
                        possible_lags = []
                        for k,item in enumerate(recallornot):    
                            if item==0: # not recalled
                                N = N + 1
                                lag = k - sp1
                                possible_lags.append(np.abs(lag))
                                if np.abs(lag) <= lag_examine:
                                    possible_counts[lag+lag_examine] += 1   
                        sorted_lags = np.sort(possible_lags)  

                        ranks = [r for r,x in enumerate(sorted_lags-actual_lag) if x==0]  
                        if N>1:
                            TCE = (np.mean(ranks))/(N-1)   
                            TCEs.append(TCE)                
    crp = [(actual_counts[i]+1)/(possible_counts[i]+1) for i in range(len(actual_counts))]
    crp[lag_examine] = 0
    return crp, np.mean(TCEs)


def get_crp_subject(recalls_sp,lag_examine,ll,exclude):
    possible_counts = np.zeros(2*lag_examine+1)
    actual_counts = np.zeros(2*lag_examine+1)
    for i in range(len(recalls_sp)):
        recallornot = np.zeros(ll)
        for j in range(len(recalls_sp[i]))[:-1]:
            sp1 = recalls_sp[i][j]
            sp2 = recalls_sp[i][j+1]
            if 0 <= sp1 <100:
                recallornot[sp1] = 1 
                
            if j> exclude and 0 <= sp1 <100:  # all possible transitions 
                recallornot[sp1] = 1 
                for k,item in enumerate(recallornot): 
                    if item==0: # not recalled
                        lag = k - sp1
                        if np.abs(lag) <= lag_examine:
                            possible_counts[lag+lag_examine] += 1   
       
            if j>exclude and 0 <= sp1 <100 and 0 <= sp2 <100: # actual transition
                lag = sp2 - sp1
                if np.abs(lag) <= lag_examine:
                    actual_counts[lag+lag_examine] += 1
                    
    crp = [actual_counts[i]/possible_counts[i] if possible_counts[i]>0 else -0.01 for i in range(len(actual_counts))]
    crp[lag_examine] = 0
    return crp, 0


def obtain_CRPbehav(crp1,crp2):

    # only consider subjets with non zero CRPs
    marks1 = np.ones(len(crp1))
    for i in range(len(crp1)):
        for j in range(9):
            if crp1[i][j]<0:
                marks1[i]=0

    marks2 = np.ones(len(crp2))
    for i in range(len(crp2)):
        for j in range(9):
            if crp2[i][j]<0:
                marks2[i]=0
    
    forward1 = [np.sum(x[5:9])-np.sum(x[0:4]) for i,x in enumerate(crp1) if marks1[i]==1]
    forward2 = [np.sum(x[5:9])-np.sum(x[0:4]) for i,x in enumerate(crp2) if marks2[i]==1]
    contiguity1 = [np.sum(x[0:4]) for i,x in enumerate(crp1) if marks1[i]==1]
    contiguity2 = [np.sum(x[0:4]) for i,x in enumerate(crp2) if marks2[i]==1]  
    
    
    return forward1,forward2,contiguity1,contiguity2
      


def load_data():
    
################
#Extractin the session 1, young subjects data from PEERS 

#Download entire PEERS dataset from: http://memory.psych.upenn.edu/Data_Archive

#OUTPUT:
# 1. recalls - L lists of recall including information on subj, session, item ID, list number, serial position, IRT per line. 
# 2. recalls_sp - L lists of serial position
# 3. sessionlengths -number of lists per subject
# 4. listlengths (Lx1) - listlengths per list
# 5. subjects (Nx1) - subject IDs from PEERS
# 6. subjs (Lx1) -  subject IDs from extracted lists of data
############  

    # PARAMETERS
    # data directory
    files_dir = '/mnt/cup/people/qiongz/data/'

    # select subjects
    all_subjects = range(63,301)
    subject_dir = files_dir + 'PEERS.data/PEERS_older_adult_subject_list.txt'
    with open(subject_dir) as f:
        content = f.readlines()
        content = [x.split() for x in content]
    subjects_old = [int(x[0]) for x in content]
    subjects_young = [x for x in all_subjects if x not in subjects_old] 

    # options: 1. all_subjects 2. subjects_old 3. subjects_young
    subjects = subjects_young

    # other parameters 
    LL = 16

    # DATA
    recalls_sp = []
    recalls = []
    sessionlengths = []
    listlengths = []
    subjs = []
    subjs_list = [] # re-numbered subjects starting from zero - consistent with 'subjs'
    for s, subj in enumerate(subjects):
        subjs_list.append(s)
        #sessionlength = []
        listlength = []
        length = 0
        for session in [1,2,3,4,5,6]: # excluding the first section which is a practice section
            # Specify data file directory - 3 digits for each subj ID
            if len(str(subj))<3:
                parent_dir = files_dir + 'PEERS.data/LTP0' + str(subj) + '/session_' + str(session) 
            else:  
                parent_dir = files_dir + 'PEERS.data/LTP' + str(subj) + '/session_' + str(session)
            data_dir1 = parent_dir + '/session.log'            
            data_dir2 = parent_dir + '/ffr.par' 
            my_file1 = Path(data_dir1)
            my_file2 = Path(data_dir2)     


            # Load presentation details    
            if my_file1.is_file(): 
                data_dir1 = parent_dir + '/session.log' 
                with open(data_dir1) as f:
                    content = f.readlines()
                    content = [x.split() for x in content]
                pre_list = -1
                dics= []
                for i in range(len(content)):
                    if content[i][2] == 'FR_PRES': # record only stimuli in the encoding stage
                        current_list = int(content[i][3])
                        item = int(content[i][5])
                        if current_list != pre_list:
                            position = 0        
                        dics.append([item,current_list,position]) # create dictionary of word ID, list number, serial position
                        pre_list = current_list
                        position = position + 1

                data_dir = parent_dir + '/15.par' 
                my_file = Path(data_dir)        
                if position == LL and my_file.is_file(): # two kinds of trials (3 or 16 items during encoding), only take trials with 16 items encoded; only take sessions with 16 lists   
                    listlength.append(position)

                    # IMMEDIATE FREE RECALL
                    for listno in range(16): # every selected session has 16 lists
                        data_dir3 = parent_dir + '/' + str(listno)+ '.par' 
                        my_file3 = Path(data_dir3)
                        if my_file3.is_file():   
                            length = length + 1
                            recall = []
                            recall_sp = []
                            subjs.append(s)

                            with open(data_dir3) as f:
                                content = f.readlines()
                                content = [x.split() for x in content]
                            lasttime = 0
                            for i in range(len(content)):
                                item = int(content[i][1])
                                temp = [x for x in dics if x[0]== item] 
                                currenttime = int(content[i][0])
                                RT = (currenttime - lasttime)/1000
                                lasttime = currenttime
                                if len(temp)>0:
                                    recall_sp.append(temp[0][2])
                                    recall.append([subj,session,item,temp[0][1],temp[0][2],RT]) #subj, session, item ID, list number, serial position, IRT
                                else:
                                    recall_sp.append(-1)
                                    recall.append([subj,session,item,-1,-1,RT]) #subj, session, item ID, list number, serial position, IRT             
                            recalls.append(recall)
                            recalls_sp.append(recall_sp)                  
        listlengths.append(listlength)
        sessionlengths.append(length)
    return recalls, recalls_sp, listlengths, sessionlengths, subjects, subjs



def extractTrials(recalls,recalls_sp,subjects,subjs,sessionlengths,percent,exclude,primacy_threshold):
    MAX_OUTPUT = 50 # code maxmimum 50 outputs per list 
    MAX_POS = 10 # consider the first 5 output transitions
    LIST_NUM = 96 # only take subjects that completed all 6 sessions
    LL = 16 # list length
    N = 9 # all trials (all, top 10, bottom 90), primacy trials (all, top 10, bottom 90), non-primacy trials (all, top 10, bottom 90) 
    #exclude = 2 # CRP start calculating from which output position
    #primacy_threshold = 4 # primacy trials with first item occuring earlier than this output position

    thresholds = np.zeros(3)
    included_subjects = [] # list of included subjects in the analysis
    for itr in range(3):
        temp = []
        for s, subj in enumerate(subjects):
            if sessionlengths[s]>= LIST_NUM: # include subjects only if there are enough number of lists
                included_subjects.append(s)
                if itr==0:
                    recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s] 
                elif itr==1:    
                    recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s and (0 in x[0:primacy_threshold])]
                elif itr==2:    
                    recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s and (0 not in x)]

                acc = np.mean([len(np.unique([x for x in rp if 0<=x<LL])) for rp in recall_sp]) 
                temp.append(acc)
        thresholds[itr] = np.percentile(temp,percent)


    # Extract N set of behavioral patterns
    spcs = [[] for i in range(N)]
    pfrs = [[] for i in range(N)]
    crps = [[] for i in range(N)]
    tces = [[] for i in range(N)]
    accs = [[] for i in range(N)]
    nums = [[] for i in range(N)]
    recalls_sp_good = []
    recalls_sp_bad = []
    for itr in range(3):
        for s, subj in enumerate(subjects):
            if sessionlengths[s]>= LIST_NUM: # include subjects only if there are enough number of lists
                if itr==0:
                    recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s] 
                elif itr==1:    
                    recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s and (0 in x[0:primacy_threshold])]
                elif itr==2:    
                    recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s and (0 not in x)]

                spc,pfr = get_spc_pfr(recall_sp,LL)
                crp,tce = get_crp(recall_sp,4,LL,exclude)
                acc = np.mean([len(np.unique([x for x in rp if 0<=x<LL])) for rp in recall_sp]) 
                num = len(recall_sp)
                spcs[itr*3].append(spc)
                pfrs[itr*3].append(pfr)
                crps[itr*3].append(crp)
                accs[itr*3].append(acc)
                tces[itr*3].append(tce)
                nums[itr*3].append(num)

                
                
                if acc > thresholds[itr]:
                    spcs[itr*3+1].append(spc)
                    pfrs[itr*3+1].append(pfr)
                    crps[itr*3+1].append(crp)
                    accs[itr*3+1].append(acc)
                    tces[itr*3+1].append(tce)
                    nums[itr*3+1].append(num)
                    if itr == 0:
                        recalls_sp_good.extend(recall_sp) # joining list of lists
                else:    
                    spcs[itr*3+2].append(spc)
                    pfrs[itr*3+2].append(pfr)
                    crps[itr*3+2].append(crp)
                    accs[itr*3+2].append(acc)
                    tces[itr*3+2].append(tce)
                    nums[itr*3+2].append(num) 
                    if itr == 0:
                        recalls_sp_bad.extend(recall_sp) # joining list of lists 
               
        
    transitions_many = np.zeros((MAX_POS, len(spcs[0]),LL))
    transitions_few = np.zeros((MAX_POS, len(spcs[0]),LL)) 
    count = 0
    for s, subj in enumerate(subjects):
        if sessionlengths[s]>= LIST_NUM: # include subjects only if there are enough number of lists
            # many recalls
            recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s and len(np.unique([y for y in x if 0<=y<LL]))>12] 
            serial_count_subj = [[] for j in range(MAX_POS)]
            for trial_id,trial in enumerate(recall_sp):
                recallornot = np.zeros(LL)
                for item_id,item in enumerate(trial):
                    if item_id < MAX_POS and item > -1 and item <LL and recallornot[item] == 0: # if it is a correct and non-recalled item
                        serial_count_subj[item_id].append(item)
                    recallornot[item] = 1
            for i in range(MAX_POS):
                transitions_many[i,count,:],_ = np.histogram(serial_count_subj[i], bins=range(LL+1))
            if len(recall_sp)>0:
                transitions_many[:,count,:] = transitions_many[:,count,:]/len(recall_sp)
  
            
           # few recalls
            recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s and len(np.unique([y for y in x if 0<=y<LL]))<13]
            serial_count_subj = [[] for j in range(MAX_POS)]
            for trial_id,trial in enumerate(recall_sp):
                recallornot = np.zeros(LL)
                for item_id,item in enumerate(trial):
                    if item_id < MAX_POS and item > -1 and item <LL and recallornot[item] == 0: # if it is a correct and non-recalled item
                        serial_count_subj[item_id].append(item)
                    recallornot[item] = 1
            for i in range(MAX_POS):
                transitions_few[i,count,:],_ = np.histogram(serial_count_subj[i], bins=range(LL+1))
            if len(recall_sp)>0:
                transitions_few[:,count,:] = transitions_few[:,count,:]/len(recall_sp)   
            count = count + 1    
 


    transitions_good = np.zeros((MAX_POS, len(spcs[1]),LL))
    transitions_bad = np.zeros((MAX_POS, len(spcs[2]),LL)) 
    RT_good = np.zeros((MAX_POS, len(spcs[1])))
    RT_bad = np.zeros((MAX_POS, len(spcs[2])))     
    count_good = 0
    count_bad = 0
    for s, subj in enumerate(subjects):
        if sessionlengths[s]>= LIST_NUM: # include subjects only if there are enough number of lists
            recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==s] 
            recall_RT = [x for i,x in enumerate(recalls) if subjs[i]==s] 
            acc = np.mean([len(np.unique([x for x in rp if 0<=x<LL])) for rp in recall_sp])
            
            # calc output transitions
            serial_count_subj = [[] for j in range(MAX_POS)]
            RT_subj = [[] for j in range(MAX_POS)]
            for trial_id,trial in enumerate(recall_sp):
                recallornot = np.zeros(LL)
                for item_id,item in enumerate(trial):
                    if item_id < MAX_POS and item > -1 and item <LL and recallornot[item] == 0: # if it is a correct and non-recalled item
                        serial_count_subj[item_id].append(item)
                        RT_subj[item_id].append(recall_RT[trial_id][item_id][-1])
                    recallornot[item] = 1

            if acc > thresholds[0]:
                for i in range(MAX_POS):
                    transitions_good[i,count_good,:],_ = np.histogram(serial_count_subj[i], bins=range(LL+1))
                    RT_good[i,count_good] = np.mean(RT_subj[i])
                if len(recall_sp)>0:
                    transitions_good[:,count_good,:] = transitions_good[:,count_good,:]/len(recall_sp)
                count_good = count_good+1
            else:
                for i in range(MAX_POS):
                    transitions_bad[i,count_bad,:],_ = np.histogram(serial_count_subj[i], bins=range(LL+1))
                    RT_bad[i,count_bad] = np.mean(RT_subj[i])
                if len(recall_sp)>0:
                    transitions_bad[:,count_bad,:] = transitions_bad[:,count_bad,:]/len(recall_sp)
                count_bad = count_bad+1   
                    
    return spcs, pfrs, crps, accs, tces, nums, recalls_sp_good, recalls_sp_bad, included_subjects,transitions_good,transitions_bad, RT_good, RT_bad, transitions_many, transitions_few

def extractData_HowaKaha(filename):
    data_dir = filename 
    with open(data_dir) as f:
        content = f.readlines()
        content = [x.split() for x in content]

    recall = [] # each item: (correct, position in study, word ID in pool, RT, temporal adjacency)
                #  temporal adjacency (1: adjacent; 0: not adjacent; -1: not applicable)
    study = []
    subject = []
    count = -1
    recall_sp = []
    for x in content: 
        count = count + 1
        if len(x) == 0:           
            count = -1
        else:
            if count == 0:
                subject.append(int(x[0]))
            elif count == 1:
                study.append([int(y)-1 for y in x])
            elif count == 2:
                pos = [int(y)-1 for y in x] 
                correct = [0 + (y > 0) for y in pos]
            elif count == 3:
                word = [int(y)-1 for y in x] 
                #print(word)
            elif count == 4:
                RT = [int(y) for y in x] 

                tmp = [[c,p,w,r] for c,p,w,r in zip(correct,pos,word,RT)]
                tmp_sp = [p for c,p,w,r in zip(correct,pos,word,RT)]
     
                recall.append(tmp) 
                recall_sp.append(tmp_sp)
    return recall_sp, subject


def extractTrials_HoweKaha(recalls_sp,subjects,subjs,percent,exclude,primacy_threshold):
    MAX_OUTPUT = 50 # code maxmimum 50 outputs per list 
    #LIST_NUM = 96 # only take subjects that completed all 6 sessions
    LL = 12 # list length
    N = 3 # all trials (all, top 10, bottom 90), primacy trials (all, top 10, bottom 90), non-primacy trials (all, top 10, bottom 90) 

    thresholds = np.zeros(3)
    included_subjects = [] # list of included subjects in the analysis
    

    temp = []
    for s, subj in enumerate(subjects):

        recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==subj] # all trials
        acc = np.mean([len(np.unique([x for x in rp if 0<=x<LL])) for rp in recall_sp]) 
        #if math.isnan(acc):
        #    print([len(np.unique([x for x in rp if 0<=x<LL])) for rp in recall_sp])
        temp.append(acc)
    threshold = np.percentile(temp,percent)


    # Extract N set of behavioral patterns
    spcs = [[] for i in range(N)]
    pfrs = [[] for i in range(N)]
    crps = [[] for i in range(N)]
    tces = [[] for i in range(N)]
    accs = [[] for i in range(N)]
    nums = [[] for i in range(N)]
    recalls_sp_good = []
    recalls_sp_bad = []
    for s, subj in enumerate(subjects):
        recall_sp = [x for i,x in enumerate(recalls_sp) if subjs[i]==subj] 
        spc,pfr = get_spc_pfr(recall_sp,LL)
        crp,tce = get_crp_subject(recall_sp,4,LL,exclude)
        acc = np.mean([len(np.unique([x for x in rp if 0<=x<LL])) for rp in recall_sp]) 
        num = len(recall_sp)
        spcs[0].append(spc)
        pfrs[0].append(pfr)
        crps[0].append(crp)
        accs[0].append(acc)
        tces[0].append(tce)
        nums[0].append(num)

        if acc > threshold:
            spcs[1].append(spc)
            pfrs[1].append(pfr)
            crps[1].append(crp)
            accs[1].append(acc)
            tces[1].append(tce)
            nums[1].append(num)
            recalls_sp_good.extend(recall_sp) # joining list of lists
        else:    
            spcs[2].append(spc)
            pfrs[2].append(pfr)
            crps[2].append(crp)
            accs[2].append(acc)
            tces[2].append(tce)
            nums[2].append(num) 
            recalls_sp_bad.extend(recall_sp) # joining list of lists                  
    return spcs, pfrs, crps, accs, tces, nums, recalls_sp_good, recalls_sp_bad