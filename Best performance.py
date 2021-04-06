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
rcParams['figure.figsize'] = (3.6, 3.6)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
#%% Load from Pickle Problems: ED n, ED, MG, DAP

Sample_List = [100,300,500,700,1000,5000,10000]

#%% Plot computation time vs accuracy
Results= pickle.load(open( "All results.p", "rb" ) )
i=2
x=Results[i][0][3]/Results[i][0][3].max()
y=Results[i][2][1]/Results[i][2][1].max()
metric = (x+y)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x,y , label="Sampling with warm start",color='r')
ax2.plot(x,metric, label='Performance metric')
ax2.scatter(x[metric.argmin()],metric.min(),marker='x',linewidths=3,label='Optimal point',color='r')


plt.xscale('log')
ax1.set_xlabel('Normalized Time')
ax1.set_ylabel('Normalized RMSE',color='r')
ax2.set_ylabel('Metric',color='b')
fig.legend(frameon=True,loc=(0.38,0.75))


#%%Point list

CR_P_x,CR_P_z,points = MA_Problems.Region_Sampling(Sampled,A,b,c,B_theta=B_theta,C_theta=C_theta,N_Regions_Max=10000)
#%%
plt.plot(points)
N=100
plt.xlabel('Region number')
plt.ylim(-0.5,10)
plt.ylabel('Points per region')
moving = np.convolve(points, np.ones(N)/N, mode='valid')
plt.plot(moving)


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

