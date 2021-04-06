# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:00:12 2019

@author: Johannes
"""
#Libraries
import pandas as pd  
import numpy as np
from scipy.stats import norm
import MA_Problems
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
import seaborn as sns
import pickle

#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (5.0, 5.0)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
#%% Base
Samples = 10000
Std = 5

#%% Create Std Form
Max = 500
Ramp = 300
Cost = 35
T=48
d=3
A = np.zeros((8*T,T))
b = np.zeros((8*T,1))
C_theta = np.eye(T)
#Number of row restriction
row = 0

#x<Max*y
for t in range (T):
    A[row,t]=1
    b[row]=Max
    row+=1
    

#Ramp Down
for t in range (1,T):
    A[row,t-1]=1
    A[row,t]=-1
    b[row]= Ramp
    row+=1
    
#Ramp Up
for t in range (1,T):
    A[row,t-1]=-1
    A[row,t]=1
    b[row]= Ramp
    row+=1
    

        
A = np.matrix(np.concatenate((A[:row,:],np.eye(row)),axis=1))
b = np.matrix(b[:row,:])
        
#Day Ahead Prices 21-22 Oct - EntsoE
Price = np.matrix([28,28,27,26,27,30,44,50,53,50,44,41,36,31,27,25,24,28,36,29,29,25,24,13,8,1,6,10,11,28,30,45,47,45,42,31,26,27,33,42,46,48,51,53,44,36,33,27]).T
Profit = -Cost*np.ones((T,1))+Price
c = np.matrix(np.concatenate((-Profit,np.zeros((row,1)))))
C_theta = np.matrix(np.concatenate((C_theta, np.zeros((row,T))),axis=0))

m=A.shape[0]
n=A.shape[1]
non_basic_var = n-m

#Create Sigma function
Sigma = np.zeros((C_theta.shape[1],C_theta.shape[1]))
for t1 in range(C_theta.shape[1]):
    for t2 in range(C_theta.shape[1]):
        if t1==t2:
            Sigma[t1,t2]=1
        else:
            Sigma[t1,t2] = np.exp(-(t1-t2)**2/8)

Sigma = Std**2*Sigma


Std_Form = {'A': A, 'b': b,'c':c,'C_theta': C_theta,'B_theta': None,'Sigma':Sigma}

pickle.dump( Std_Form, open( "Std Form DAP.p", "wb" ) )


#%% Heatgrid Sigma
sns.heatmap(Sigma,cbar_kws={'label': 'Variance [(€/MWh)^2]'})

#%%
Price=np.array([28,28,27,26,27,30,44,50,53,50,44,41,36,31,27,25,24,28,36,29,29,25,24,13,8,1,6,10,11,28,30,45,47,45,42,31,26,27,33,42,46,48,51,53,44,36,33,27])
x=np.linspace(0,47,48)
plt.plot(x,Price, label='Market price',color='g')
plt.fill_between(x,Price+Std, Price-Std,color='g',alpha=0.5)
plt.axhline(35, label='Marginal cost',color='r')
plt.xlabel('Time [h]')
plt.ylabel('[€/MWh]')
plt.legend(frameon=True,loc=3)