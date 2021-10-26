import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

# spc and pfr
def get_spc_pfr(recalls_sp,list_length):
    num_trial = len(recalls_sp)
    spc = np.zeros(list_length)
    pfr = np.zeros(list_length)
    for i,recall_sp in enumerate(recalls_sp):
        for j,item in enumerate(recall_sp):           
            if 0 <= item <100:
                spc[item] += 1
                if j==0:
                    pfr[item] += 1 
            
    return spc/num_trial, pfr/num_trial

# crp
def get_crp(recalls_sp,lag_examine,ll):
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

