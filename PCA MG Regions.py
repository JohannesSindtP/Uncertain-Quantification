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

import MA_Problems
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
from seaborn import heatmap
import pickle
from scipy.stats import ks_2samp
from numpy.linalg import eig

import polytope as pc


#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (30,30)
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False

Std_Form_DAP=  pickle.load(open( "Std Form MG.p", "rb" ) )
A = Std_Form_DAP['A']
b = Std_Form_DAP['b']
c = Std_Form_DAP['c']
Sigma = Std_Form_DAP['Sigma']

OFC = MA_Problems.Optimize_Gaussian_Diferential(A,b,c,np.zeros(Sigma.shape[0]), Sigma, 10000,B_theta=Std_Form_DAP['B_theta'])
#%%

values, vectors = eig(Sigma)
components = 2
P = Sigma@vectors[:,:components]

#Plot for 2 eigenvalues
for i in range(2):
    plt.plot(vectors[:,i],label='Component '+str(i+1))
plt.legend()
plt.xlabel('Uncertain variable')
plt.ylabel('Vector value')


#%%
j=5
for i in range(j,j+10):
    plt.figure()
    rcParams['axes.spines.bottom'] = False
    rcParams['axes.spines.left'] = False
    plt.rcParams["axes.grid"] = False
    axes_lim = 1
    ax = plt.axes(xlim=(-axes_lim,axes_lim),ylim=(-axes_lim,axes_lim))
    for ind in OFC.index:
        A_p = OFC['A_p'].at[ind][:-2,:]@vectors[:,:components]
        b_p = OFC['b_p'].at[ind]
        poly=pc.Polytope(np.array(A_p), b_p)
        poly.plot(ax=ax,linewidth=7,linestyle='-')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('C:/Users/Johannes/Documents/Master/Masterarbeit/CR fotos/'+str(i)+'.pdf')
