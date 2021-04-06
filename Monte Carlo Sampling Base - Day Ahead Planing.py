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
rcParams['figure.figsize'] = (7.0, 7.0)
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

#savemat('DAP.mat',Std_Form)

#%% Distributions c

Sampled = np.random.multivariate_normal(np.zeros(Sigma.shape[0]),Sigma,size=10)


for samp in Sampled:
    plt.plot(samp,linewidth=1)
    
plt.ylabel('Price variation [€/MW]')
plt.title('OFC')

Sampled = np.matrix(Sampled)
#%% Sampling c
start_time = time.time()

Resultsc = []
Times =[]
Priofits = []
for i in Sampled:
    c_T_samp = (c+C_theta @ np.matrix(i).T).T
    x,z = MA_Problems.Optimize_LP(A,b,c_T_samp)
    Resultsc.append(x[:T])
    Times.append(opt_time)
    Priofits.append(Profit+np.matrix(i).T)

run_timec = time.time()-start_time

Resultsc = np.array(Resultsc).reshape(-1,)
Times = np.array(Times).reshape(-1,)
Priofits = np.array(Priofits).reshape(-1,)
Resc =pd.DataFrame([Times,Resultsc,Priofits]).T
Resc.columns=['Time [h]','Generation [MWh]','Profit [€/MWh]']

#%% Plot c


g = sns.jointplot(data = Resc, x='Profit [€/MWh]',y='Generation [MWh]', kind="hex",color='forestgreen',gridsize=48,marginal_ticks=True,mincnt=100,vmax=2000)
plt.title('OFC')

plt.figure()
g = sns.JointGrid(data = Resc, x='Time [h]',y='Generation [MWh]',marginal_ticks=True)
g.plot_joint(plt.hexbin,gridsize=48,cmap='Greens',mincnt=100,vmax=2000)
g.ax_marg_x.set_axis_off()
sns.histplot(data = Resc,y='Generation [MWh]',ax=g.ax_marg_y,color='forestgreen')
plt.title('OFC')

plt.figure()
plt.hist(Resc['Generation [MWh]'].loc[Resc['Time [h]']==Resc['Time [h]'].at[12]],color='forestgreen',bins=T)
plt.title('Hour 12 - Mean Profit: 1 [€/MWh] ')
plt.xlabel('Generation [MWh]' )
plt.ylabel('Quantity' )
#%% Save to pickle
Base_cases = {'OFC': Resc,'RunTimeOFC':run_timec}

pickle.dump( Base_cases, open( "BaseCases DayAheadPlaning.p", "wb" ) )

#%%Plots
x,_,_,_ = MA_Problems.Optimize_LP(A,b,c)
plt.plot(Profit,label = 'Profit [€/MWh]')

plt.plot(x[:48],label = 'Generation [MWh]')
plt.xlabel('Time [h]')
plt.legend()