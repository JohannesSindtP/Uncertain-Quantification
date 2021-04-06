# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:00:12 2019

@author: Johannes
"""
#Libraries
import pandas as pd  
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import time
import MA_Problems
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
from seaborn import heatmap
import pickle
from scipy.stats import ks_2samp


#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (3.6, 3.6)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
#%% Load from Pickle Problems: ED n, ED, MG, DAP
Problem = 'DAP'
Samples = 1000

Std_Form= pickle.load(open( "Std Form "+Problem+".p", "rb" ) )

A = Std_Form['A']
b = Std_Form['b']
c = Std_Form['c']
Sigma = Std_Form['Sigma']
C_theta = Std_Form['C_theta']
B_theta =Std_Form['B_theta']
#Warm Start
x0, basis_vars, VBasis, CBasis = MA_Problems.Optimize_LP(A,b,c)

MC_X = pd.read_csv(Problem+' Base MC - X.csv',index_col=(0))
MC_Z = pd.read_csv(Problem+' Base MC - Z.csv',index_col=(0))

Quantiles2 = np.linspace(0.01,0.99,99)
Basis_Q_Values = np.quantile(MC_Z,Quantiles2)

Sampled = MA_Problems.Normal_LHS(np.zeros(Sigma.shape[0]),Sigma+ 0.005*np.identity(Sigma.shape[0]),Samples)
if Problem == 'MG':
    bs = np.linalg.pinv(B_theta)*b
    for i in range(Sampled.shape[0]):
        for j in range(Sampled.shape[1]):
            if bs[j]<-Sampled[i,j]:
                Sampled[i,j] = -bs[j]

start_time = time.time()

x_list = []
z_list = []
for sample in Sampled:
    
        if C_theta is not None:
            #Optimize with the sample
            x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
            x_list.append(x)
            z_list.append(z)
        
        if B_theta is not None:
            #Optimize with the sample
            
            b_samp = b+B_theta*np.matrix(sample).T
            
            x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b_samp,c, VBasis, CBasis)
            x_list.append(x)
            z_list.append(z)
            
run_time_WS = time.time()-start_time

X = pd.DataFrame(x_list)
Z = pd.DataFrame(z_list)

# Maintaining Basis

if C_theta is not None:
    B_inv, D, c_B, c_D,C_theta_B, C_theta_D = MA_Problems.Process_OFC(A,c,basis_vars,C_theta)
    z_mean = c_B.T*B_inv * b
    z_var = b.T*B_inv.T * C_theta_B * Sigma * C_theta_B.T * B_inv * b
    x_mean = np.array(x0)[basis_vars].reshape(-1,1)
    
if B_theta is not None:
    B_inv = MA_Problems.Process_RHS(A,basis_vars)
    c_B = c[basis_vars]
    x_mean = B_inv * b
    x_var = B_inv @ B_theta @ Sigma @ B_theta.T @ B_inv.T
    z_mean = c_B.T*x_mean
    z_var = c_B.T*x_var*c_B

start_time = time.time()
CR_x,CR_z,_ = MA_Problems.Region_Sampling(Sampled,A,b,c,B_theta=B_theta,C_theta=C_theta,N_Regions_Max=10000)
run_time_CR = time.time()-start_time
#%% Calculate RMSE
Basis_Q_Values = np.quantile(MC_Z,Quantiles2)
result = np.quantile(Z,Quantiles2)
RMSE = np.sqrt(sum((Basis_Q_Values-result)**2/99))

#%%
n_bins = Samples
z_max = (z_mean+3*np.sqrt(z_var))[0,0]+500
z_min = (z_mean-3*np.sqrt(z_var))[0,0]

plt.hist(MC_Z, 1000, density=True, histtype='step',cumulative=True,label = 'Benchmark',range=(z_min,z_max))
z= np.linspace(z_min,z_max,100)
plt.plot(z,multivariate_normal.cdf(z,z_mean,z_var),label = 'Gaussian propagation')
plt.hist(CR_z.T, Samples, density=True, histtype='step',cumulative=True,label = 'Basis sampling',range=(z_min,z_max))

plt.xlabel('z [€]')
plt.ylabel('CDF')
plt.xlim(z_min+1,z_max-1)
plt.legend(frameon=True)

plt.figure()
i=24*4+14
x_max = 0.8
x_min = -0.1
plt.hist(MC_X[str(i)], 1000, density=True, histtype='step',cumulative=True,label = 'Benchmark',range=(x_min,x_max))
x= np.linspace(x_min,x_max,100)
plt.plot(x,multivariate_normal.cdf(x,x_mean[sum(basis_vars[:i])],x_var[sum(basis_vars[:i]),sum(basis_vars[:i])]+0.001),label = 'Gaussian propagation')
plt.hist(CR_x[i,:].T, Samples, density=True, histtype='step',cumulative=True,label = 'Basis sampling',range=(x_min,x_max))

plt.xlabel('Battery energy at h=14 [MWh]')
plt.ylabel('CDF')
plt.legend(frameon=True,loc=2)
plt.xlim(x_min+0.01,x_max-0.01)

Basis_Q_Values_X = np.quantile(MC_X[str(i)],Quantiles2)
result = np.quantile(CR_x[i,:].T,Quantiles2)
RMSE_X = np.sqrt(sum((Basis_Q_Values_X-result)**2/99))

#%% Plot cumulative histograms
n_bins = 500
z_max = (z_mean+3*np.sqrt(z_var))[0,0]
z_min = (z_mean-3*np.sqrt(z_var))[0,0]

plt.hist(MC_Z, 1000, density=True, histtype='step',cumulative=True,label = 'Benchmark',range=(z_min,z_max))
z= np.linspace(z_min,z_max,100)
plt.plot(z,multivariate_normal.cdf(z,z_mean,z_var),label = 'Gaussian propagation')
plt.hist(Z, Samples, density=True, histtype='step',cumulative=True,label = 'Sampling methods',range=(z_min,z_max))
plt.hist(CR_P_z.T, Samples, density=True, histtype='step',cumulative=True,label = 'Basis sampling with early stopping',range=(z_min,z_max))

plt.xlabel('z [€]')
plt.ylabel('CDF')
plt.legend()

#%% Megaplot X
i=18
n_bins = 500
x_max = 1
x_min = 0

plt.hist(MC_X[str(i)], 1000, density=True, histtype='step',cumulative=True,label = 'Benchmark',range=(x_min,x_max))
x= np.linspace(0,0.75,100)
plt.plot(x,multivariate_normal.cdf(x,x_mean[sum(basis_vars[:i])],x_var[sum(basis_vars[:i]),sum(basis_vars[:i])])+0.001,label = 'Gaussian propagation',range=(x_min,x_max))
plt.hist(X[i,:], Samples, density=True, histtype='step',cumulative=True,label = 'Sampling methods',range=(x_min,x_max))
plt.hist(CR_P_x[i,:], Samples, density=True, histtype='step',cumulative=True,label = 'Basis sampling with early stopping',range=(x_min,x_max))


plt.xlabel('X18 [MWh]')
plt.ylabel('CDF')
plt.legend(loc=9)
#%% Depth of discharge
plt.hist(X[4*24+14]/0.75, n_bins, density=True, histtype='step',cumulative=True,color = 'red')
plt.xlabel('Depth of charge')
plt.ylabel('CDF')

