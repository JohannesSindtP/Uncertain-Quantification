# -*- coding: utf-8 -*-

from scipy.stats import norm
from scipy.stats import multivariate_normal

import pandas as pd  
import numpy as np
from scipy.stats import norm
import MA_Problems
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
from seaborn import heatmap
import pickle
from scipy.stats import ks_2samp
import polytope as pc

#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (3.5, 7)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
#%% Load from Pickle Problems: ED n, ED, MG, DAP

Sample_List = [100,300,500,700,1000,5000,10000]
Repeats = 50
Results = []

for Problem in ['ED n','MG','DAP']:
    Std_Form= pickle.load(open( "Std Form "+Problem+".p", "rb" ) )
    
    A = Std_Form['A']
    b = Std_Form['b']
    c = Std_Form['c']
    Sigma = Std_Form['Sigma']
    mu = np.zeros(Sigma.shape[0])
    C_theta = Std_Form['C_theta']
    B_theta = Std_Form['B_theta']
    #Warm Start
    
    MC_X = pd.read_csv(Problem+' Base MC - X.csv',index_col=(0))
    MC_Z = pd.read_csv(Problem+' Base MC - Z.csv',index_col=(0))
    
    
    Quantiles2 = np.linspace(0.01,0.99,99)
    
    Basis_Q_Values = np.quantile(MC_Z,Quantiles2)
    
    Times = []
    L1 = []
    L2 = []
    L1_std = []
    L2_std = []
    for Samples in Sample_List:
        run_timeLP = []
        run_timeWS = []
        run_timeMB = []
        run_timeCR = []
        run_timeCR_P = []
        L1_MB= []
        L2_MB=[]
        MB_KS=[]
        L1_CR= []
        L2_CR=[]
        CR_KS=[]
        L1_CR_P= []
        L2_CR_P=[]
        CR_P_KS=[]
        for rep in range(Repeats):
            # Make Samples
            Sampled =  MA_Problems.Normal_MC(mu,Sigma,Samples)
            if Problem == 'MG':
                bs = np.linalg.pinv(B_theta)*b
                for i in range(Sampled.shape[0]):
                    for j in range(Sampled.shape[1]):
                        if bs[j]<-Sampled[i,j]:
                            Sampled[i,j] = -bs[j]
            
            # Critical Regions
            start_time = time.time()
            CR_x,CR_z,_ = MA_Problems.Region_Sampling(Sampled,A,b,c,B_theta=B_theta,C_theta=C_theta,N_Regions_Max=10000)
            
            run_timeCR.append(time.time()-start_time)
                    
            result = np.quantile(CR_z,Quantiles2)
            L1_CR.append(sum(abs(Basis_Q_Values-result))/99)
            L2_CR.append(np.sqrt(sum((Basis_Q_Values-result)**2/99)))
            
        Times.append(np.mean(run_timeCR))
        L1.append(np.mean(L1_CR))
        L2.append(np.mean(L2_CR))
        L1_std.append(np.std(L1_CR))
        L2_std.append(np.std(L2_CR))
        
    Times = np.array(Times).T
    L1 = np.array(L1).T
    L2 = np.array(L2).T
    L1_std = np.array(L1_std).T
    L2_std = np.array(L2_std).T
    Results.append([Times,L1,L2,L1_std,L2_std])

pickle.dump( Results, open( "Result MC basis.p", "wb" ) )

#%% Plot computation time vs accuracy
Results= pickle.load(open( "All results.p", "rb" ) )
Colors = ['b','r','orange','orange']
Names=['ED','MG','DAP']
alpha = 0.5
fig, ax = plt.subplots(3,1,sharex=True)
fig.subplots_adjust(hspace=0.04)
for i in range(3):
    
    Norm_value = Results[i][2][1].max()

    ax[i].plot(Results[i][0][0]/Results[i][0][0].max(), Results[i][2][1]/Norm_value, label="Sampling",color = Colors[0])
    ax[i].fill_between(Results[i][0][0]/Results[i][0][0].max(), (Results[i][2][1] + Results[i][4][1])/Norm_value, (Results[i][2][1]- Results[i][4][1])/Norm_value,alpha=alpha,color = Colors[0])

    # ax[i].plot(Results[i][0][1], Results[i][2][1]/Norm_value, label="Sampling with warm start",color = Colors[1])
    # ax[i].fill_between(Results[i][0][1], (Results[i][2][1] + Results[i][4][1])/Norm_value, (Results[i][2][1]- Results[i][4][1])/Norm_value,alpha=alpha,color = Colors[1])

    ax[i].plot(Results[i][0][3]/Results[i][0][0].max(), Results[i][2][1]/Norm_value, label="Basis sampling",color = Colors[2])
    ax[i].fill_between(Results[i][0][3]/Results[i][0][0].max(), (Results[i][2][1] + Results[i][4][1])/Norm_value, (Results[i][2][1]- Results[i][4][1])/Norm_value,alpha=alpha,color = Colors[2])

    # ax[i].plot(Results[i][0][4], Results[i][2][2]/Norm_value, label="Basis sampling with early stopping",color = Colors[3])
    # ax[i].fill_between(Results[i][0][4], (Results[i][2][2] + Results[i][4][2])/Norm_value, (Results[i][2][2]- Results[i][4][2])/Norm_value,alpha=alpha,color = Colors[3])

    ax[i].set_ylabel(Names[i]+' RMSE')
    ax[i].set_xscale('log')
    #ax[i].set_yscale('log')
    #ax[i].set_title(Names[i])
    if i==0:
        ax[i].legend(frameon=True,loc=1)
        
plt.xscale('log')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('Plots\RMSE LP All.pdf')

#%%Point list
Sampled =  MA_Problems.Normal_LHS(mu,Sigma,10000)
if Problem == 'MG':
    bs = np.linalg.pinv(B_theta)*b
    for i in range(Sampled.shape[0]):
        for j in range(Sampled.shape[1]):
            if bs[j]<-Sampled[i,j]:
                Sampled[i,j] = -bs[j]
CR_P_x,CR_P_z,points = MA_Problems.Region_Sampling(Sampled,A,b,c,B_theta=B_theta,C_theta=C_theta,N_Regions_Max=10000)
#%%
plt.plot(points,label='Samples per basis')
N=100
plt.xlabel('Basis number')
plt.ylim(-0.5,10)
plt.ylabel('Samples')
moving = np.convolve(points, np.ones(N)/N, mode='valid')
plt.plot(moving,label='Rolling mean, N=100')
plt.legend(frameon=True)

#%% Plot computation time vs accuracy
Results = pickle.load(open( "All results.p", "rb" ) )

Colors = ['b','r','g','blueviolet','orange','lime']
Names = ['10 PP economic dispatch','micro grid','day ahead planning']

for i in range(3):
    plt.figure()
    plt.plot(Results[i][0][0],Results[i][1][1],label ='Sampling L1',color=Colors[i],linestyle='solid')
    plt.plot(Results[i][0][1],Results[i][1][1],label ='Sampling with warm start L1',color=Colors[i],linestyle='dotted')
    plt.plot(Results[i][0][3],Results[i][1][1],label ='Critical regions L1',color=Colors[i],linestyle='dashed')
    plt.plot(Results[i][0][4],Results[i][1][2],label ='Critical regions with early stopping L1',color=Colors[i],linestyle='dashdot')
    plt.plot(Results[i][0][0],Results[i][2][1],label ='Sampling L2',color=Colors[i+3],linestyle='solid')
    plt.plot(Results[i][0][1],Results[i][2][1],label ='Sampling with warm start L2',color=Colors[i+3],linestyle='dotted')
    plt.plot(Results[i][0][3],Results[i][2][1],label ='Critical regions L2',color=Colors[i+3],linestyle='dashed')
    plt.plot(Results[i][0][4],Results[i][2][2],label ='Critical regions with early stopping L2',color=Colors[i+3],linestyle='dashdot')

    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('[€]')
    plt.xscale('log')

