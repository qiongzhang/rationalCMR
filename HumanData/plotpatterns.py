import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from scipy.stats import spearmanr
import math
import helper as helper

def plot_transitions(transitions_good,transitions_bad,label1, label2,filename):
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    plt.rcParams['figure.figsize'] = (20, 10)
    LL = 16
    for i in range(8):
        plt.subplot(2,4,i+1)

        data=transitions_good[i,:,:]
        y = np.mean(data,0)
        error = np.std(data,0)/np.sqrt(transitions_good.shape[1]-1)
        plt.plot(y,'grey')


        data=transitions_bad[i,:,:]
        y2 = np.mean(data,0)
        error2 = np.std(data,0)/np.sqrt(transitions_bad.shape[1]-1)
        plt.plot(y2,'orange')

        bottom, top = plt.ylim() 
        plt.axvline(x=i,color='red',ymin=0,ymax=1,linestyle='dashed',alpha=0.5)

        if i==0:
            plt.legend([label1,label2,'Optimal Policy'])

        plt.plot(range(LL), y, color='k')
        plt.fill_between(range(LL), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

        plt.plot(range(LL), y2, color='orange')
        plt.fill_between(range(LL), y2-error2, y2+error2,alpha=0.5, edgecolor='coral', facecolor='coral')

        plt.title('Output Position = '+str(i))
        plt.axis([-1,LL,bottom,top+4])
        plt.xticks([0,5,10,15], (['0','5','10','15']))
        if i==0 or i==4:
            plt.ylabel('Probability of Recall')
        plt.xlabel('Serial Position')
        if i==0:
            plt.axis([-1,16,0,0.6])
        else:  
            plt.axis([-1,16,0,0.4])
    plt.subplots_adjust(hspace=0.3)     
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()
    
def plot_behav(data_spc,data_pfr,data_crp,CMR_spc,CMR_pfr,CMR_crp,ll,legend1, legend2, filename):    
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
    plt.legend([legend1,legend2])
    plt.ylabel('Probability of Recall')
    plt.xlabel('Serial Position')

    plt.subplot(1,3,2)
    plt.plot(data_pfr,'-ok')
    plt.plot(CMR_pfr,':ok',mfc='none')
    plt.axis([-1,ll,0,0.6])
    plt.ylabel('Probability of First Recall')
    plt.xlabel('Serial Position')


    plt.subplot(1,3,3)
    plt.plot(range(4), data_crp[0:4], '-ok')
    plt.plot(range(4), CMR_crp[0:4], ':ok',mfc='none')
    plt.plot([5,6,7,8], data_crp[5:9], '-ok')
    plt.plot([5,6,7,8], CMR_crp[5:9], ':ok',mfc='none')
    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.axis([-1,9,0,0.4])
    plt.ylabel('Conditional Response Probability')
    plt.xlabel('Lag')

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+ filename) 
    plt.show()    
    
def plot_behav0(LL, spc1, pfr1, crp1, spc2, pfr2, crp2, label1, label2,filename):

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16

    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']



    plt.rcParams['figure.figsize'] = (17,5.5)
    plt.subplot(1,3,1)
    #plt.title('(a)')
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
    plt.axis([-1,LL,0,1.1])

    plt.subplot(1,3,2)
    #plt.title('(b)')
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
   # plt.title('(c)')
    temp = [x[0:4] for i,x in enumerate(crp1) if marks1[i]==1 ]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    plt.plot(range(4), y, color='k')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [x[5:9] for i,x in enumerate(crp1) if marks1[i]==1 ]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    #plt.plot(y)
    plt.plot([5,6,7,8], y, color='k')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [x[0:4] for i,x in enumerate(crp2) if marks2[i]==1 ]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot(range(4), y, color='#CC4F1B')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    temp = [x[5:9] for i,x in enumerate(crp2) if marks2[i]==1 ]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot([5,6,7,8], y, color='#CC4F1B')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.ylabel('CRP')
    plt.xlabel('Lag')



    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()
    
    forward1 = [np.sum(x[5:9])-np.sum(x[0:4]) for i,x in enumerate(crp1) if marks1[i]==1]
    forward2 = [np.sum(x[5:9])-np.sum(x[0:4]) for i,x in enumerate(crp2) if marks2[i]==1]
    lag1 = [x[5] for i,x in enumerate(crp1) ]
    lag2 = [x[5] for i,x in enumerate(crp2) ]
    contiguity1 = [np.sum(x[0:4]) for i,x in enumerate(crp1) if marks1[i]==1]
    contiguity2 = [np.sum(x[0:4]) for i,x in enumerate(crp2) if marks2[i]==1]  
    primacy1 = [x[0] for i,x in enumerate(pfr1) if x[0]>0]
    primacy2 = [x[0] for i,x in enumerate(pfr2) if x[0]>0]
    
    
    return forward1,forward2,lag1,lag2,contiguity1,contiguity2,primacy1,primacy2
    
def plot_behav1(LL, spc1, pfr1, crp1, spc2, pfr2, crp2, label1, label2, spcs, pfrs, crps, accs,filename):

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20

    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']



    plt.rcParams['figure.figsize'] = (24,5.5)
    plt.subplot(1,4,1)
    #plt.title('(a)')
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
    plt.axis([-1,LL,0,1.1])

    plt.subplot(1,4,2)
    #plt.title('(b)')
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


    plt.subplot(1,4,3)
   # plt.title('(c)')
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
    plt.ylabel('CRP')
    plt.xlabel('Lag')

    plt.subplot(1,4,4)
    #plt.title('(d)')
    forward = [np.sum(x[5:9])-np.sum(x[0:4]) for x in crps]
    contiguity = [(np.sum(x[2:3])+np.sum(x[5:6])-np.sum(x[0:1])-np.sum(x[7:8])) for x in crps]
    print(str(spearmanr(accs,forward)))
    print(str(spearmanr(forward,contiguity)))
    plt.rcParams['figure.figsize'] = (5.5,5.5)
    plt.scatter(accs,forward,color='coral',alpha=0.5)
    #plt.legend(['r = -0.80'],frameon=False)
    #plt.annotate('r = '+ str(spearmanr(accs,forward)[0])[0:4],xy=(12, 0.4))
    plt.annotate('r = 0.25',xy=(10, 0.4))
    plt.ylabel('Forward Asymmetry')
    plt.xlabel('Accuracy')

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()
    
    
    
    
def plot_behav2(spc1, pfr1, crp1, spc2, pfr2, crp2, label1, label2, ratio,filename):

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for space between subplots,
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
    
    plt.rcParams['figure.figsize'] = (11.5,5.5)

    plt.subplot(1,2,1)
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
    plt.ylabel('CRP')
    plt.xlabel('Lag')


    plt.subplot(1,2,2)
    if ratio ==1:
        temp = [np.sum(x[5:9])/np.sum(x[0:4]) for x in crp1]
    else:    
        temp = [np.sum(x[5:9])-np.sum(x[0:4]) for x in crp1]
    data=np.asarray(temp)
    y1 = np.mean(data,0)
    error1 = np.std(data,0)/np.sqrt(len(crp1)-1)
    
    if ratio ==1:
        temp = [np.sum(x[5:9])/np.sum(x[0:4]) for x in crp2]
    else:    
        temp = [np.sum(x[5:9])-np.sum(x[0:4]) for x in crp2]
    data=np.asarray(temp)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(crp2)-1)

    temp = [np.sum(x[0:4])for x in crp1]
    data=np.asarray(temp)
    y1c = np.mean(data,0)
    error1c = np.std(data,0)/np.sqrt(len(crp1)-1)
    temp = [np.sum(x[0:4])for x in crp2]
    data=np.asarray(temp)
    y2c = np.mean(data,0)
    error2c = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.bar([1,4], [y1c,y1], width=1, yerr=[error1c,error1], color='grey')
    plt.bar([2,5], [y2c,y2], width=1, yerr=[error2c,error2], color='peachpuff')
    plt.xticks([1.5,4.5], ('Backward Transition','Forward Asymmetry'))
    
    '''
    temp = [np.subtract(x[0:4],x[0:4]) for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    plt.plot(range(4), y, color='k')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [np.subtract(x[5:9],np.flip(x[0:4])) for x in crp1]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp1)-1)
    #plt.plot(y)
    plt.plot([5,6,7,8], y, color='k')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='grey', facecolor='grey')

    temp = [np.subtract(x[0:4],x[0:4]) for x in crp2]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot(range(4), y, color='#CC4F1B')
    plt.fill_between(range(4), y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    temp = [np.subtract(x[5:9],np.flip(x[0:4])) for x in crp2]
    data=np.asarray(temp)
    y = np.mean(data,0)
    error = np.std(data,0)/np.sqrt(len(crp2)-1)
    plt.plot([5,6,7,8], y, color='#CC4F1B')
    plt.fill_between([5,6,7,8], y-error, y+error,alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')

    plt.xticks(range(9), ('-4', '-3', '-2','-1','0','1','2','3','4'))
    plt.ylabel('Corrected CRP')
    plt.xlabel('Lag')
    '''

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()    
    
def plot_behav2_forward(spc1, pfr1, crp1, spc2, pfr2, crp2, label1, label2, filename):

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for space between subplots,
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
    
    plt.rcParams['figure.figsize'] = (11,5.5)

    plt.subplot(1,2,1)
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


    plt.subplot(1,2,2)
    temp = [np.sum(np.subtract(x[5:9],np.flip(x[0:4]))) for x in crp1]
    #temp = [np.sum(x[5:9])/np.sum(x[0:4]) for x in crp1]
    data=np.asarray(temp)
    y1 = np.mean(data,0)
    error1 = np.std(data,0)/np.sqrt(len(crp1)-1)
    temp = [np.sum(np.subtract(x[5:9],np.flip(x[0:4]))) for x in crp2]
    #temp = [np.sum(x[5:9])/np.sum(x[0:4]) for x in crp]
    data=np.asarray(temp)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(crp2)-1)

    temp = [np.sum(x[0:4]) for x in crp1]
    data=np.asarray(temp)
    y1c = np.mean(data,0)
    error1c = np.std(data,0)/np.sqrt(len(crp1)-1)
    temp = [np.sum(x[0:4]) for x in crp2]
    data=np.asarray(temp)
    y2c = np.mean(data,0)
    error2c = np.std(data,0)/np.sqrt(len(crp2)-1)
    #plt.bar([2], [y1], width=1, yerr=[error1], color='grey')
    #plt.bar([3], [y2], width=1, yerr=[error2], color='peachpuff')
    #plt.xticks([2.5], ('Forward Asymmetry'))

    plt.bar([1], [y1], width=1, yerr=[error1], color='grey')
    plt.bar([2], [y2], width=1, yerr=[error2], color='peachpuff')
    plt.axis([-1,4,0,0.5])
    plt.xticks([1.5,4.5], ('Forward Asymmetry','')) 

    
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()      
    
def plot_behav2_contiguity(spc1, pfr1, crp1, spc2, pfr2, crp2, label1, label2, filename):

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for space between subplots,
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
    
    plt.rcParams['figure.figsize'] = (11,5.5)

    plt.subplot(1,2,1)
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


    plt.subplot(1,2,2)
    temp = [np.sum(np.subtract(x[5:9],np.flip(x[0:4]))) for x in crp1]
    data=np.asarray(temp)
    y1 = np.mean(data,0)
    error1 = np.std(data,0)/np.sqrt(len(crp1)-1)
    temp = [np.sum(np.subtract(x[5:9],np.flip(x[0:4]))) for x in crp2]
    data=np.asarray(temp)
    y2 = np.mean(data,0)
    error2 = np.std(data,0)/np.sqrt(len(crp2)-1)

    temp = [np.sum(x[0:4]) for x in crp1]
    data=np.asarray(temp)
    y1c = np.mean(data,0)
    error1c = np.std(data,0)/np.sqrt(len(crp1)-1)
    temp = [np.sum(x[0:4]) for x in crp2]
    data=np.asarray(temp)
    y2c = np.mean(data,0)
    error2c = np.std(data,0)/np.sqrt(len(crp2)-1)
    #plt.bar([2], [y1c], width=1, yerr=[error1c], color='grey')
    #plt.bar([3], [y2c], width=1, yerr=[error2c], color='peachpuff')
    #plt.xticks([2.5], ('Backward Transition'))

    plt.bar([1], [y1c], width=1, yerr=[error1c], color='grey')
    plt.bar([2], [y2c], width=1, yerr=[error2c], color='peachpuff')
    plt.axis([-1,4,0,0.7])
    plt.xticks([1.5,4.5], ('Backward Transition','')) 

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace) 
    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()    
    
        
    
def plot_behav3(spcs, pfrs, crps, tces, accs, filename):

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.3  # the amount of width reserved for space between subplots,
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
    
    # no primacy
    plt.rcParams['figure.figsize'] = (16.5,5.5)
    plt.subplot(1,3,1)
    value1 = [np.sum(x[0:4]) for x in crps]
    value2 = [x/LL for x in accs]
    print(str(spearmanr(value1,value2))) 
    plt.scatter(value1,value2,color='blue',alpha=0.2)
    plt.annotate('r = '+ str(spearmanr(value1,value2)[0])[0:4],xy=(0.6, 0.6))
    plt.ylabel('Accuracy')
    plt.xlabel('Contiguity')
    plt.subplot(1,3,2)
    value1 = [np.sum(x[5:9])-np.sum(x[0:4]) for x in crps]
    value2 = [x/LL for x in accs]
    print(str(spearmanr(value1,value2))) 
    plt.scatter(value1,value2,color='blue',alpha=0.2)
    plt.annotate('r = '+ str(spearmanr(value1,value2)[0])[0:4],xy=(0.25, 0.6))
    #plt.axis([0,0.5,0,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Forward')
    plt.subplot(1,3,3)
    value1 = tces
    value2 = [x/LL for x in accs]
    print(str(spearmanr(value1,value2))) 
    plt.scatter(value1,value2,color='blue',alpha=0.2)
    plt.annotate('r = '+ str(spearmanr(value1,value2)[0])[0:5],xy=(0.2, 0.6))
    #plt.axis([0,0.5,0,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Temporal factor')


    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()    
    
    
def plot_behav4(recalls_sp_good,recalls_sp_bad,filename):
    ll=16
    trans_good = helper.calc_trans(recalls_sp_good,16)
    trans_bad = helper.calc_trans(recalls_sp_bad[0:len(recalls_sp_good)],16)

    # Figure settings
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.3  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']

    plt.rcParams['figure.figsize'] = (22,6.5)
    plt.subplot(1,3,1)
    #plt.title('(e)')
    for i in range(len(recalls_sp_good)):
        temp = [x for x in recalls_sp_good[i] if x>-1]
        plt.plot(temp,alpha=0.02,linewidth=2,ls='-')
    plt.ylabel('Serial Position')
    plt.xlabel('Output Position')
    plt.axis([-1,16,-1,16])

    plt.subplot(1,3,2)
    #plt.title('(f)')
    shuffled_indice = list(range(len(recalls_sp_bad)))
    for i in shuffled_indice[0:len(recalls_sp_good)]: # same number of trials as in recalls_sp_good
        temp = [x for x in recalls_sp_bad[i] if x>-1]
        plt.plot(temp,alpha=0.02,linewidth=2,ls='-')  
    plt.ylabel('Serial Position')
    plt.xlabel('Output Position')
    plt.axis([-1,16,-1,16])

    plt.subplot(1,3,3)
    #plt.title('(g)')
    difference = trans_good - trans_bad
    for i in range(ll):
        for m in range(ll):
            for n in range(ll):
                if difference[i][m][n]>0:
                    for t in range(int(difference[i][m][n])):
                        plt.plot([i,i+1],[m,n],alpha=0.02,linewidth=2,ls='-',color='coral')
                elif difference[i][m][n]<0:        
                    for t in range(-int(difference[i][m][n])):
                        plt.plot([i,i+1],[m,n],alpha=0.02,linewidth=2,ls='-',color='b')    
    plt.ylabel('Serial Position')
    plt.xlabel('Output Position')
    plt.axis([-1,16,-1,16])

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()

def plot_behav5(accs,nums,recalls_sp_good,recalls_sp_bad,filename):
    ll=16
    trans_good = helper.calc_trans(recalls_sp_good,16)
    trans_bad = helper.calc_trans(recalls_sp_bad[0:len(recalls_sp_good)],16)

    # Figure settings
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
    hspace = 0.5  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    LL = 16
    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    colors = ['red','k','grey','silver','lightcoral','maroon','coral','peachpuff','b']



    plt.rcParams['figure.figsize'] = (24,5.5)
    
    
    
    plt.subplot(1,4,1)
    ratio = [nums[3][i]/(nums[3][i]+nums[6][i]) for i in range(len(nums[3]))]
    print(str(spearmanr(accs[0],ratio)))
    #plt.rcParams['figure.figsize'] = (5.5,5.5)
    plt.scatter(accs[0],ratio,color='coral',alpha=0.5)
    plt.annotate('r = 0.59',xy=(7, 0.8))
    plt.ylabel('Proportion of Primacy Trials')
    plt.xlabel('Accuracy')

    
    plt.subplot(1,4,2)
    #plt.title('(e)')
    for i in range(len(recalls_sp_good)):
        temp = [x for x in recalls_sp_good[i] if x>-1]
        plt.plot(temp,alpha=0.02,linewidth=2,ls='-')
    plt.ylabel('Serial Position')
    plt.xlabel('Output Position')
    plt.axis([-1,16,-1,16])

    plt.subplot(1,4,3)
    #plt.title('(f)')
    shuffled_indice = list(range(len(recalls_sp_bad)))
    for i in shuffled_indice[0:len(recalls_sp_good)]: # same number of trials as in recalls_sp_good
        temp = [x for x in recalls_sp_bad[i] if x>-1]
        plt.plot(temp,alpha=0.02,linewidth=2,ls='-')  
    plt.ylabel('Serial Position')
    plt.xlabel('Output Position')
    plt.axis([-1,16,-1,16])

    plt.subplot(1,4,4)
    #plt.title('(g)')
    difference = trans_good - trans_bad
    for i in range(ll):
        for m in range(ll):
            for n in range(ll):
                if difference[i][m][n]>0:
                    for t in range(int(difference[i][m][n])):
                        plt.plot([i,i+1],[m,n],alpha=0.02,linewidth=2,ls='-',color='coral')
                elif difference[i][m][n]<0:        
                    for t in range(-int(difference[i][m][n])):
                        plt.plot([i,i+1],[m,n],alpha=0.02,linewidth=2,ls='-',color='b')    
    plt.ylabel('Serial Position')
    plt.xlabel('Output Position')
    plt.axis([-1,16,-1,16])

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)    
    if filename:
        plt.savefig('./Figs/'+filename) 
    plt.show()
