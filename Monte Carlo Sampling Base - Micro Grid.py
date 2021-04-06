# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:00:12 2019

@author: Johannes
"""
#Libraries
import pandas as pd  
import numpy as np
import MA_Problems
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
import seaborn as sns
import pickle
from scipy.io import savemat

#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (7.0, 7.0)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False

#%% Base
Samples = 10000
Std = 0.2

#%% Create Std Form
Max_DG = 1
Hours=24
step = 1 # 1/4 hour steps
T=int(Hours/step)
Max_PV = np.cos(np.linspace(-np.pi, np.pi, T))
for i in range(len(Max_PV)):
    if Max_PV[i]<0:
        Max_PV[i]=0
Max_Bat = 0.75
E_Max_Bat = 0.75
Cost = 70
E_init = 0.25 #MWh
ef = 0.95 
Demand = 0.75
Dis_Rate_Bat = 0.01 #1%/h
A = np.zeros((9*T,6*T))
b = np.zeros((9*T,1))
B_theta = np.zeros((9*T,T))
#Number of row restriction
row = 0

#Variables Order
#X_DG,X_PV,X_BAT+,X_BAT-,E_BAT

#b Order
#Max_DG,Max_PV->Uncertain,Max_Bat,-Max_Bat, E_Max_Bat, 0
#Decision Variables
#ON_DG

#Inequalities
#Disel Generator Max
for t in range (T):
    A[row,t]=1
    b[row]=Max_DG
    row+=1

#Solar Generation
for t in range (T):
    A[row,t+T]=1
    b[row]= Max_PV[t]
    B_theta[row,t]=1
    row+=1
    
#Battery Max
for t in range (T):
    A[row,t+2*T]=1
    A[row,t+3*T]=-1
    b[row]= Max_Bat
    row+=1

#Battery Energy Max
for t in range (T):
    A[row,t+4*T]=1
    b[row]= E_Max_Bat
    row+=1
    
row_eq = row

#Equalties
#E(t=0) is not a variable
A[row,4*T]=1
b[row]= E_init
row+=1

#Energy balance
for t in range (1,T):
    A[row,t+4*T-1]=1-Dis_Rate_Bat*step
    A[row,t+4*T]=-1
    A[row,t+2*T-1]=-1/ef*step
    A[row,t+3*T-1]=ef*step
    row+=1
    
#Power Balance
for t in range (T):
    A[row,t]=1
    A[row,t+T]=1
    A[row,t+2*T]=1
    A[row,t+3*T]=-1
    b[row]= Demand
    row+=1

slacks = np.concatenate((np.eye(row_eq),np.zeros((row-row_eq,row_eq))),axis=0)
A = np.matrix(np.concatenate((A[:row,:],slacks),axis=1))

b = np.matrix(b[:row,:])

#Only uncertain variables for day hours 14 hours
B_theta = np.matrix(B_theta[:row,6*step:18*step])

        

c = np.matrix(np.concatenate((Cost*np.ones(T),np.zeros(A.shape[1]-T)))).T

m=A.shape[0]
n=A.shape[1]
non_basic_var = n-m
index_row = np.arange(m)
index_column = np.arange(n)

#Create Sigma function
# Sigma = np.zeros((B_theta.shape[1],B_theta.shape[1]))
# for t1 in range(B_theta.shape[1]):
#     for t2 in range(B_theta.shape[1]):
#         if t1==t2:
#             Sigma[t1,t2]=1
#         else:
#             Sigma[t1,t2] = np.exp(-(t1-t2)**2/4)

# Sigma = Std**2*Sigma

Vars = Max_PV[6*step:18*step]
Sigma = np.zeros((B_theta.shape[1],B_theta.shape[1]))
for t1 in range(B_theta.shape[1]):
    for t2 in range(B_theta.shape[1]):
        if t1==t2:
            Sigma[t1,t2]=Vars[t1]+0.001
        else:
            Sigma[t1,t2] = (Vars[t1]+Vars[t2])/2*np.exp(-(t1-t2)**2/4)

Sigma = Std**2*Sigma

Std_Form = {'A': A, 'b': b,'c':c,'B_theta': B_theta,'C_theta': None,'Sigma':Sigma}

pickle.dump( Std_Form, open( "Std Form MG.p", "wb" ) )

#%% PV Generation varies normal gaussian
Sampled = np.random.multivariate_normal(np.zeros(Sigma.shape[0]),Sigma,size=10)


for samp in Sampled:
    plt.plot(samp,linewidth=1)
    
plt.ylabel('Maximal solar generation [MW]')
plt.title('RHS')

Sampled = np.matrix(Sampled)
#%% Sampling b
start_time = time.time()

X_Gen=[]
X_PV=[]
X_Bat=[]
X_En=[]
Time=[]


x,_,_,_ = MA_Problems.Optimize_LP(A,b,c)

X_Gen.append(x[:T])
X_PV.append(x[T:2*T])
X_Bat.append(np.array(x[2*T:3*T])-np.array(x[3*T:4*T]))
X_En.append(x[4*T:5*T])

run_timeb = time.time()-start_time

X_Gen = np.array(X_Gen).reshape(-1,)
X_PV = np.array(X_PV).reshape(-1,)
X_Bat = np.array(X_Bat).reshape(-1,)
X_En = np.array(X_En).reshape(-1,)
Time = np.array(Time).reshape(-1,)

plt.plot(X_Gen, label ='Diesel [MW]')
plt.plot(X_PV, label ='Solar [MW]')
plt.plot(X_Bat, label ='Battery power [MW]')
plt.plot(X_En, label ='Battery energy [MWh]')
plt.axhline(y=0.75,linestyle='--', label ='Demand [MW]')
plt.legend()
plt.xlabel('Time [h]')
Resb =pd.DataFrame([Time,X_Gen,X_PV,X_Bat,X_En]).T
Resb.columns=['Time [h]','Diesel generation [MW]','Solar generation [MW]','Battery power [MW]','Battery energy [MWh]']
#%% Plot b
color = 'royalblue'

g = sns.JointGrid(data = Resb, x='Time [h]',y='Diesel generation [MW]',marginal_ticks=True)
g.plot_joint(plt.hexbin,gridsize=48,cmap='Blues',mincnt=100,vmax=2000)
g.ax_marg_x.set_axis_off()
sns.histplot(data = Resb,y='Diesel generation [MW]',ax=g.ax_marg_y,color=color)
plt.title('RHS')

g = sns.JointGrid(data = Resb, x='Time [h]',y='Solar generation [MW]',marginal_ticks=True)
g.plot_joint(plt.hexbin,gridsize=48,cmap='Blues',mincnt=100,vmax=2000)
g.ax_marg_x.set_axis_off()
sns.histplot(data = Resb,y='Solar generation [MW]',ax=g.ax_marg_y,color=color)
plt.title('RHS')

g = sns.JointGrid(data = Resb, x='Time [h]',y='Battery power [MW]',marginal_ticks=True)
g.plot_joint(plt.hexbin,gridsize=48,cmap='Blues',mincnt=100,vmax=2000)
g.ax_marg_x.set_axis_off()
sns.histplot(data = Resb,y='Battery power [MW]',ax=g.ax_marg_y,color=color)
plt.title('RHS')

g = sns.JointGrid(data = Resb, x='Time [h]',y='Battery energy [MWh]',marginal_ticks=True)
g.plot_joint(plt.hexbin,gridsize=48,cmap='Blues',mincnt=100,vmax=2000)
g.ax_marg_x.set_axis_off()
sns.histplot(data = Resb,y='Battery energy [MWh]',ax=g.ax_marg_y,color=color)
plt.title('RHS')
#%% Save to pickle
Base_cases = {'RHS': Resb, 'RunTimeRHS':run_timeb,}

pickle.dump( Base_cases, open( "BaseCases-MicroGrid.p", "wb" ) )

#%% Heatgrid Sigma
sns.heatmap(Sigma,cbar_kws={'label': 'Variance [MW^2]'})
