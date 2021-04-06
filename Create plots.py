# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:02:49 2021

@author: Johannes
"""

# -*- coding: utf-8 -*-

import pandas as pd  
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pickle

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (3.5, 7.0)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False

#%% Plot computation time vs accuracy
Sample_List = [100,300,500,700,1000,5000,10000]
Results= pickle.load(open( "All results sampling.p", "rb" ) )
Colors = ['b','r','g']
Names=['ED','MG','DAP']
alpha = 0.5
fig, ax = plt.subplots(3,1,sharex=True)
fig.subplots_adjust(hspace=0.04)

for i in range(3):
    
    Norm_value = Results[i][2][0].max()
    ax[i].plot(Sample_List, Results[i][2][0]/Norm_value, label="Monte Carlo",color = Colors[0])
    ax[i].fill_between(Sample_List, (Results[i][2][0] + Results[i][5][0])/Norm_value, (Results[i][2][0]- Results[i][5][0])/Norm_value,alpha=alpha,color = Colors[0])

    ax[i].plot(Sample_List, (Results[i][2][1])/Norm_value, label="Latin hypercube",color = Colors[1])
    ax[i].fill_between(Sample_List, (Results[i][2][1] + Results[i][5][1])/Norm_value, (Results[i][2][1]- Results[i][5][1])/Norm_value,alpha=alpha,color = Colors[1])

    ax[i].plot(Sample_List,( Results[i][2][2])/Norm_value, label="Halton sequence",color = Colors[2])
    ax[i].fill_between(Sample_List, (Results[i][2][2] + Results[i][5][2])/Norm_value,( Results[i][2][2]- Results[i][5][2])/Norm_value,alpha=alpha,color = Colors[2])
    ax[i].set_ylabel(Names[i]+' RMSE')
    ax[i].set_xscale('log')
    #ax[i].set_title()
    if i==0:
        ax[i].legend(frameon=True,loc=1)

plt.tight_layout()
plt.xlabel('Samples')
plt.xlim(100,10000)
plt.savefig('Plots\RMSE Sampling All.pdf')

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

    ax[i].plot(Results[i][0][3]/Results[i][0][0].max(), Results[i][2][1]/Norm_value, label="Basis sampling",color = Colors[2])
    ax[i].fill_between(Results[i][0][3]/Results[i][0][0].max(), (Results[i][2][1] + Results[i][4][1])/Norm_value, (Results[i][2][1]- Results[i][4][1])/Norm_value,alpha=alpha,color = Colors[2])

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
