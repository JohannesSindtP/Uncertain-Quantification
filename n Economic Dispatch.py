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
from seaborn import jointplot
import pickle
from scipy.io import savemat

#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (7.0, 7.0)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
#%% Base
Samples = 10000
n = 10 

D_Res = np.matrix(np.concatenate((np.ones(n),np.zeros(n)),axis=0))
Gen_Res = np.concatenate((np.eye(n),np.eye(n)),axis=1)

A = np.concatenate((D_Res,Gen_Res),axis=0)

b =  np.random.randint(5,30,n)*10
b = np.matrix(np.concatenate((np.array([b.sum()/2]),b),axis=0)).T

c = np.matrix(np.concatenate((np.random.randint(30,80,n),np.zeros(10)))).T

B_theta = np.matrix(np.concatenate((np.array([1]),np.zeros(n)),axis=0)).T


m=A.shape[0]
n=A.shape[1]
non_basic_var = n-m
index_row = np.arange(m)
index_column = np.arange(n)

Std = b[0]/5

Std_Form = {'A': A, 'b': b,'c':c,'B_theta':B_theta,'C_theta':None,'Sigma':np.matrix(Std**2)}

#save for matlab
savemat('nED.mat',Std_Form)

#Save for python
pickle.dump( Std_Form, open( "Std Form ED n.p", "wb" ) )
#%% Demand is a normal gaussian
Std_Form=  pickle.load(open( "Std Form ED n.p", "rb" ) )


Sampled = np.random.normal(0,Std,(Samples,1))

x = np.linspace(60,140,100)
plt.plot(x,norm.pdf(x,loc=100,scale=Std))
plt.xlabel('Demand [MWh]')
plt.ylabel('Probability')
plt.title('RHS')
#%% Sampling b
start_time = time.time()

Resultsb = []
for samp in Sampled:
    b_samp=b+B_theta @ i
    x,z = MA_Problems.Optimize_LP(A,b_samp,c.T)
    Resultsb.append(x[:2])

run_timeb = time.time()-start_time
#%% Plot b
Resb =pd.DataFrame(Resultsb,columns=['X1 [MWh]','X2 [MWh]'])
jointplot(data = Resb, x='X1 [MWh]',y='X2 [MWh]', kind="hex",color='red',marginal_ticks=True)
plt.title('RHS')
#%% Distributions c

Sampled = np.random.normal([0,0],Std,(Samples,2))
x = np.linspace(20,110,100)

plt.plot(x,norm.pdf(x,loc=50,scale=Std),label='X1')
plt.plot(x,norm.pdf(x,loc=70,scale=Std),label='X2')
plt.ylabel('Probability')
plt.xlabel('Price [€/MWh]')
plt.legend()
plt.title('OFC')
#%% Sampling c
start_time = time.time()

Resultsc = []
for i in Sampled:
    c_T_samp = (c+C_theta @ np.matrix(i).T).T
    x,z = MA_Problems.Optimize_LP(A,b,c_T_samp)
    Resultsc.append(x[:2])

run_timec = time.time()-start_time
#%% Plot c
Resc =pd.DataFrame(Resultsc,columns=['X1 [MWh]','X2 [MWh]'])
jointplot(data = Resc, x='X1 [MWh]',y='X2 [MWh]', kind="hex",color='red',marginal_ticks=True)
plt.title('OFC')

#%% Save to pickle
Base_cases = {'RHS': Resb, 'OFC': Resc,'RunTimeRHS':run_timeb,'RunTimeOFC':run_timec}

pickle.dump( Base_cases, open( "BaseCases.p", "wb" ) )

#%% Open pickle

Base_cases=pickle.load(open( "BaseCases.p", "rb" ) )
Resb = Base_cases['RHS']
Resc = Base_cases['OFC']

#%% Sampling example
x = np.linspace(60,140,100)
samples= 5
uniform = np.random.uniform(size=samples)
sampled = norm.ppf(uniform, loc= 100, scale=Std)

plt.plot(x,norm.pdf(x, loc= 100, scale=Std))
for samp in sampled:
    plt.axvline(x=samp,color='black')
plt.xlabel('Demand [MWh]')
plt.ylabel('Probability')
plt.title('RHS')
plt.legend(('Probability density function','Samples'))

print (MA_Problems.bmatrix(np.matrix(uniform)))
print (MA_Problems.bmatrix(np.matrix(sampled)))

#%% Without Uncertainties
x,_,_,_ = MA_Problems.Optimize_LP(Std_Form['A'],Std_Form['b'],Std_Form['c'])

MA_Problems.bmatrix(np.matrix(x).T)

MA_Problems.bmatrix(Std_Form['b'])

MA_Problems.bmatrix(Std_Form['c'])