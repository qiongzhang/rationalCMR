import numpy as np
import matplotlib.pyplot as plt
import os
import random
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
from scipy import stats

def plot_compare_contiguity(percent,acc,gammas,contiguity,crp1,crp2,label1,label2,filename):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    NUM = 171
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    threshold = np.percentile(acc,percent)
    forward1 = []
    forward2 = []
    contiguity1 = []
    contiguity2 = []
    gamma1 = []
    gamma2 = []
    subj1 = []
    subj2 = []
    for i in range(NUM):
        if acc[i] > threshold:
            #forward1.append(forward[i])
            contiguity1.append(contiguity[i])
            gamma1.append(gammas[i])
            subj1.append(i)
        else:
            #forward2.append(forward[i])
            contiguity2.append(contiguity[i])
            gamma2.append(gammas[i])
            subj2.append(i)
 

    contiguity_1 = [np.sum(x[0:4]) for x in crp1]
    contiguity_2 = [np.sum(x[0:4]) for x in crp2]
    
    plt.rcParams['figure.figsize'] = (19,5.5)
    plt.subplot(1,3,1)
    #plt.title('(d)')
    temp = [x[0:4] for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    plt.plot(range(4), y, color='k')
    
    temp = [x[0:4] for x in crp2]
    data=np.asarray(temp)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot(range(4), y2, color='#CC4F1B')
    plt.legend([label1,label2],frameon=False,loc='upper left')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')   
    plt.fill_between(range(4), y2-error2, y2+error2,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')
 
    
    temp = [x[5:9] for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    #plt.plot(y)
    plt.plot([5,6,7,8], y, color='k')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [x[5:9] for x in crp2]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot([5,6,7,8], y, color='#CC4F1B')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.ylabel('CRP')
    plt.xlabel('Lag')
    
    plt.subplot(1,3,2)  
    data=np.asarray(contiguity_1)
    y1 = np.mean(data,0)
    error1 = np.std(data,0)/np.sqrt(len(contiguity_1)-1)
    data=np.asarray(contiguity_2)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(contiguity_2)-1)
    seed(1)
    # non-parametric
    stat, p = mannwhitneyu(contiguity_1, contiguity_2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    
    plt.bar([1], [y1], width=1, yerr=[error1], color='grey')
    plt.bar([2], [y2], width=1, yerr=[error2], color='peachpuff')
    #plt.legend([label1+' human subjects',label2+' human subjects'],frameon=False)
    plt.annotate('Human Data',xy=(0.65, 0.62),fontsize=14)
    #plt.title('(e)')
    plt.axis([-1,4,0,0.7])
    plt.xticks([1.5,4.5], ('Symmetric Contiguity','')) 

    plt.subplot(1,3,3)  

    data=np.asarray(contiguity1)
    y1c = np.mean(data,0)
    error1c = np.std(data,0)/np.sqrt(len(contiguity1)-1)
    data=np.asarray(contiguity2)
    y2c = np.mean(data,0)
    error2c = np.std(data,0)/np.sqrt(len(contiguity2)-1)
    seed(1)
    # non-parametric
    stat, p = mannwhitneyu(contiguity1, contiguity2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))


    data=np.asarray(gamma1)
    y1g = np.mean(data,0)
    error1g = np.std(data,0)/np.sqrt(len(gamma1)-1)
    data=np.asarray(gamma2)
    y2g = np.mean(data,0)
    error2g = np.std(data,0)/np.sqrt(len(gamma2)-1)
    seed(1)
    # non-parametric
    stat, p = mannwhitneyu(gamma1, gamma2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    
    plt.bar([1,4], [y1c,y1g], width=1, yerr=[error1c,error1g], color='grey')
    plt.bar([2,5], [y2c,y2g], width=1, yerr=[error2c,error2g], color='peachpuff')
    plt.annotate('Simulated Data',xy=(0.9, 0.68),fontsize=14)
    #plt.legend([label1+' simulated subjects',label2+' simulated subjects'],frameon=False)
    #plt.title('(f)')
    plt.axis([-1,6,0,0.76])
    plt.xticks([1.5,4.5], ('Symmetric Contiguity','$\gamma_{fc}$')) 
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.3, hspace=0.5)     
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()    
    
def plot_compare_forward(percent,acc,gammas,forward,crp1,crp2,label1,label2,filename):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    NUM = 171
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    threshold = np.percentile(acc,percent)
    forward1 = []
    forward2 = []
    contiguity1 = []
    contiguity2 = []
    gamma1 = []
    gamma2 = []
    subj1 = []
    subj2 = []
    for i in range(NUM):
        if acc[i] > threshold:
            forward1.append(forward[i])
            #contiguity1.append(contiguity[i])
            gamma1.append(gammas[i])
            subj1.append(i)
        else:
            forward2.append(forward[i])
           # contiguity2.append(contiguity[i])
            gamma2.append(gammas[i])
            subj2.append(i)
 

    forward_1 = [np.sum(x[5:9])-np.sum(x[0:4]) for x in crp1]
    forward_2 = [np.sum(x[5:9])-np.sum(x[0:4]) for x in crp2]

    plt.rcParams['figure.figsize'] = (19,5.5)
    plt.subplot(1,3,1)
    #plt.title('(a)')
    temp = [x[0:4] for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    plt.plot(range(4), y, color='k')
    
    temp = [x[0:4] for x in crp2]
    data=np.asarray(temp)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot(range(4), y2, color='#CC4F1B')
    plt.legend([label1,label2],frameon=False,loc='upper left')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')   
    plt.fill_between(range(4), y2-error2, y2+error2,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')
 
    
    temp = [x[5:9] for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    #plt.plot(y)
    plt.plot([5,6,7,8], y, color='k')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [x[5:9] for x in crp2]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot([5,6,7,8], y, color='#CC4F1B')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.ylabel('CRP')
    plt.xlabel('Lag')
    
    plt.subplot(1,3,2)  
    data=np.asarray(forward_1)
    y1 = np.mean(data,0)
    error1 = np.std(data,0)/np.sqrt(len(forward_1)-1)
    data=np.asarray(forward_2)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(forward_2)-1)
    seed(1)
    # non-parametric
    stat, p = mannwhitneyu(forward_1, forward_2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    plt.bar([1], [y1], width=1, yerr=[error1], color='grey')
    plt.bar([2], [y2], width=1, yerr=[error2], color='peachpuff')
    #plt.legend([label1+' human subjects',label2+' human subjects'],frameon=False)
    plt.annotate('Human Data',xy=(0.75, 0.44),fontsize=14)
    #plt.title('(b)')
    plt.axis([-1,4,0,0.5])
    plt.xticks([1.5,4.5], ('Forward Asymmetry','')) 

    plt.subplot(1,3,3)  

    data=np.asarray(forward1)
    y1c = np.mean(data,0)
    error1c = np.std(data,0)/np.sqrt(len(forward1)-1)
    data=np.asarray(forward2)
    y2c = np.mean(data,0)
    error2c = np.std(data,0)/np.sqrt(len(forward2)-1)
    seed(1)
    # non-parametric
    stat, p = mannwhitneyu(forward1, forward2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))


    data=np.asarray(gamma1)
    y1g = np.mean(data,0)
    error1g = np.std(data,0)/np.sqrt(len(gamma1)-1)
    data=np.asarray(gamma2)
    y2g = np.mean(data,0)
    error2g = np.std(data,0)/np.sqrt(len(gamma2)-1)
    seed(1)
    stat, p = mannwhitneyu(gamma1, gamma2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    plt.bar([1,4], [y1c,y1g], width=1, yerr=[error1c,error1g], color='grey')
    plt.bar([2,5], [y2c,y2g], width=1, yerr=[error2c,error2g], color='peachpuff')
    plt.annotate('Simulated Data',xy=(0.9, 0.8),fontsize=14)
    #plt.legend([label1+' simulated subjects',label2+' simulated subjects'],frameon=False)
    #plt.title('(c)')
    plt.axis([-1,6,0,0.9])
    plt.xticks([1.5,4.5], ('Forward Asymmetry','$\gamma_{fc}$')) 
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.3, hspace=0.5)    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()      
       
def plot_behav(data_spc,data_pfr,data_crp,CMR_spc,CMR_pfr,CMR_crp,ll,filename):    
    # Figure settings
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']


    plt.rcParams['figure.figsize'] = (18,5)
    plt.subplot(1,3,1)
    plt.plot(data_spc,'-ok')
    plt.plot(CMR_spc,':ok',mfc='none')
    plt.axis([-1,ll,0,1])
    plt.legend(['Data','CMR'])
    plt.ylabel('Probability of Recall')
    plt.xlabel('Serial Position')

    plt.subplot(1,3,2)
    plt.plot(data_pfr,'-ok')
    plt.plot(CMR_pfr,':ok',mfc='none')
    plt.axis([-1,ll,0,1])
    plt.ylabel('Probability of First Recall')
    plt.xlabel('Serial Position')


    plt.subplot(1,3,3)
    plt.plot(range(4), data_crp[0:4], '-ok')
    plt.plot(range(4), CMR_crp[0:4], ':ok',mfc='none')
    plt.plot([5,6,7,8], data_crp[5:9], '-ok')
    plt.plot([5,6,7,8], CMR_crp[5:9], ':ok',mfc='none')
    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.axis([-1,9,0,0.7])
    plt.ylabel('Conditional Response Probability')
    plt.xlabel('Lag')

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+ filename) 
    plt.show()    



def plot_simulations(CMR_spc,CMR_pfr,CMR_crp,ll,filename):    
    # Figure settings
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']


    plt.rcParams['figure.figsize'] = (18,5)
    plt.subplot(1,3,1)
    plt.plot(CMR_spc,':ok',mfc='none')
    plt.axis([-1,ll,0,1.05])
    #plt.legend(['Data','CMR'])
    plt.ylabel('Probability of Recall')
    plt.xlabel('Serial Position')

    plt.subplot(1,3,2)
    plt.plot(CMR_pfr,':ok',mfc='none')
    plt.axis([-1,ll,-0.05,1])
    plt.ylabel('Probability of First Recall')
    plt.xlabel('Serial Position')


    plt.subplot(1,3,3)
    plt.plot(range(4), CMR_crp[0:4], ':ok',mfc='none')
    plt.plot([5,6,7,8], CMR_crp[5:9], ':ok',mfc='none')
    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.axis([-1,9,-0.05,1])
    plt.ylabel('Conditional Response Probability')
    plt.xlabel('Lag')

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+ filename) 
    plt.show()
    
def plot_CRP(CMR_spc,CMR_pfr,CMR_crp,ll,filename):    
    # Figure settings
    left = 0.16  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']


    plt.rcParams['figure.figsize'] = (5.5,5)
    plt.plot(range(4), CMR_crp[0:4], ':ok',mfc='none')
    plt.plot([5,6,7,8], CMR_crp[5:9], ':ok',mfc='none')
    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.axis([-1,9,-0.05,0.7])
    plt.ylabel('Conditional Response Probability')
    plt.xlabel('Lag')

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+ filename) 
    plt.show()
    
    
def plot_compare(percent,acc,gammas,primacys,crp,spc,pfr,recalls, NUM,label1,label2,filename):
    # Figure settings
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']


    threshold = np.percentile(acc,percent)
    print(threshold)
    gamma1 = []
    gamma2 = []
    primacy1 = []
    primacy2 = []
    subj1 = []
    subj2 = []
    crp1 = []
    crp2 = []
    spc1 = []
    spc2 = []
    pfr1 = []
    pfr2 = []
    LL = 16
    recalls_sp_good = []
    recalls_sp_bad = []
    for i in range(NUM):
        if acc[i] > threshold:
            gamma1.append(gammas[i])
            primacy1.append(primacys[i])
            subj1.append(i)
            crp1.append(crp[i])
            spc1.append(spc[i]) 
            pfr1.append(pfr[i])
            recalls_sp_good.extend(recalls[i])
        else:
            gamma2.append(gammas[i])
            primacy2.append(primacys[i])
            subj2.append(i)
            crp2.append(crp[i])
            spc2.append(spc[i]) 
            pfr2.append(pfr[i])
            recalls_sp_bad.extend(recalls[i])
 


    data=np.asarray(gamma1)
    y1g = np.mean(data,0)
    error1g = np.std(data,0)/np.sqrt(len(gamma1)-1)
    data=np.asarray(gamma2)
    y2g = np.mean(data,0)
    error2g = np.std(data,0)/np.sqrt(len(gamma2)-1)

    data=np.asarray(primacy1)
    y1c = np.mean(data,0)
    error1c = np.std(data,0)/np.sqrt(len(gamma1)-1)
    data=np.asarray(primacy2)
    y2c = np.mean(data,0)
    error2c = np.std(data,0)/np.sqrt(len(gamma2)-1)

    
    # plotting
    plt.rcParams['figure.figsize'] = (18,5.5)
    plt.subplot(1,3,1)
    data=np.asarray(spc1)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(spc1)-1)
    plt.plot(y,'grey')
    
    data=np.asarray(spc2)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(spc2)-1)
    plt.plot(y2,'orange')
    plt.legend([label1,label2])
    
    plt.plot(range(LL), y, color='k')
    plt.fill_between(range(LL), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')
    plt.plot(range(LL), y2, 'k', color='#CC4F1B')
    plt.fill_between(range(LL), y2-error2, y2+error2,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    #plt.title('Serial position curve')
    plt.ylabel('Probability of Recall')
    plt.xlabel('Serial Position')
    plt.axis([-1,16,0,1.1])

    plt.subplot(1,3,2)
    data=np.asarray(pfr1)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(pfr1)-1)
    plt.plot(y)
    plt.plot(range(LL), y, color='k')
    plt.fill_between(range(LL), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    data=np.asarray(pfr2)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(pfr2)-1)
    plt.plot(y)
    plt.plot(range(LL), y, 'k', color='#CC4F1B')
    plt.fill_between(range(LL), y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.ylabel('Probability of First Recall')
    plt.xlabel('Serial Position')


    plt.subplot(1,3,3)
    temp = [x[0:4] for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    plt.plot(range(4), y, color='k')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [x[5:9] for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    #plt.plot(y)
    plt.plot([5,6,7,8], y, color='k')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [x[0:4] for x in crp2]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot(range(4), y, color='#CC4F1B')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    temp = [x[5:9] for x in crp2]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot([5,6,7,8], y, color='#CC4F1B')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.ylabel('Conditional Response Probability')
    plt.xlabel('Lag')
    
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+ filename) 
    plt.show()
    
def plot_spc(spc1, spc2,spc3, filename):
    # Figure settings
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    LL = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']


    
    # plotting
    plt.rcParams['figure.figsize'] = (6,5.5)
    data=np.asarray(spc1)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(spc1)-1)
    plt.plot(y,'grey')
    
    data=np.asarray(spc2)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(spc2)-1)
    plt.plot(y2,'orange')
    
    data=np.asarray(spc3)
    y3 = np.mean(data,0)
    error3 = np.std(data,0)/np.sqrt(len(spc3)-1)
    plt.plot(y3,'red')
    plt.legend(['Both','Primacy','Recency'])
    
    plt.plot(range(LL), y, color='k')
    plt.fill_between(range(LL), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')
    plt.plot(range(LL), y2, 'k', color='#CC4F1B')
    plt.fill_between(range(LL), y2-error2, y2+error2,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.plot(range(LL), y3, 'k', color='#CC4F1B')
    plt.fill_between(range(LL), y3-error3, y3+error3,alpha=0.5, edgecolor='coral', facecolor='coral')    
    #plt.title('Serial position curve')
    plt.ylabel('Probability of Recall')
    plt.xlabel('Serial Position')
    plt.axis([-1,16,0,1.1])


    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+ filename) 
    plt.show()
    
   