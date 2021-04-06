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

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
#Figure Parameters
plt.style.use(['seaborn-whitegrid'])
rcParams['figure.figsize'] = (7.0, 7.0)
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
    KS = []
    L1_std = []
    L2_std = []
    KS_std = []
    for Samples in Sample_List:
        run_timeMC = []
        L1MC = []
        L2MC = []
        KSMC = []
        run_timeLH = []
        L1LH = []
        L2LH = []
        KSLH = []
        run_timeHS = []
        L1HS = []
        L2HS = []
        KSHS = []
        for rep in range(Repeats):
                        # Make Samples
            Sampled =  MA_Problems.Normal_MC(mu,Sigma,Samples)
            if Problem == 'MG':
                bs = np.linalg.pinv(B_theta)*b
                for i in range(Sampled.shape[0]):
                    for j in range(Sampled.shape[1]):
                        if bs[j]<-Sampled[i,j]:
                            Sampled[i,j] = -bs[j]
            
            # Warm Start
            start_time = time.time()
            x0, basis_vars, VBasis, CBasis = MA_Problems.Optimize_LP(A,b,c)
            x_list = []
            z_list = []
            for sample in Sampled:
                
                if C_theta is not None:
                    #Optimize with the sample
                    x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
                    if x == False:
                        continue
                    x_list.append(x)
                    z_list.append(z)
                
                if B_theta is not None:
                    #Optimize with the sample
                    b_samp = b+B_theta*np.matrix(sample).T
                    
                    x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b_samp,c, VBasis, CBasis)
                    if x == False:
                        continue
                    x_list.append(x)
                    z_list.append(z)
                    
            Quasi_MC_X_WS = pd.DataFrame(x_list)
            Quasi_MC_Z_WS = pd.DataFrame(z_list)
            
            run_timeMC.append(time.time()-start_time)
            
            result = np.quantile(Quasi_MC_Z_WS,Quantiles2)
            L1MC.append(sum(abs(Basis_Q_Values-result))/99)
            L2MC.append(np.sqrt(sum((Basis_Q_Values-result)**2/99)))
            KSMC.append(ks_2samp(MC_Z['0'],Quasi_MC_Z_WS[0]).statistic)
            
            # Make Samples
            Sampled =  MA_Problems.Normal_LHS(mu,Sigma,Samples)
            if Problem == 'MG':
                bs = np.linalg.pinv(B_theta)*b
                for i in range(Sampled.shape[0]):
                    for j in range(Sampled.shape[1]):
                        if bs[j]<-Sampled[i,j]:
                            Sampled[i,j] = -bs[j]
            
            # Warm Start
            start_time = time.time()
            x0, basis_vars, VBasis, CBasis = MA_Problems.Optimize_LP(A,b,c)
            x_list = []
            z_list = []
            for sample in Sampled:
                
                if C_theta is not None:
                    #Optimize with the sample
                    x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
                    if x == False:
                        continue
                    x_list.append(x)
                    z_list.append(z)
                
                if B_theta is not None:
                    #Optimize with the sample
                    b_samp = b+B_theta*np.matrix(sample).T
                    
                    x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b_samp,c, VBasis, CBasis)
                    if x == False:
                        continue
                    x_list.append(x)
                    z_list.append(z)
                    
            Quasi_MC_X_WS = pd.DataFrame(x_list)
            Quasi_MC_Z_WS = pd.DataFrame(z_list)
            
            run_timeLH.append(time.time()-start_time)
            
            result = np.quantile(Quasi_MC_Z_WS,Quantiles2)
            L1LH.append(sum(abs(Basis_Q_Values-result))/99)
            L2LH.append(np.sqrt(sum((Basis_Q_Values-result)**2/99)))
            KSLH.append(ks_2samp(MC_Z['0'],Quasi_MC_Z_WS[0]).statistic)
            
                    # Make Samples
            Sampled =  MA_Problems.Normal_HAL(mu,Sigma,Samples)
            if Problem == 'MG':
                bs = np.linalg.pinv(B_theta)*b
                for i in range(Sampled.shape[0]):
                    for j in range(Sampled.shape[1]):
                        if bs[j]<-Sampled[i,j]:
                            Sampled[i,j] = -bs[j]
            
            # Warm Start
            start_time = time.time()
            x0, basis_vars, VBasis, CBasis = MA_Problems.Optimize_LP(A,b,c)
            x_list = []
            z_list = []
            for sample in Sampled:
                
                if C_theta is not None:
                    #Optimize with the sample
                    x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
                    if x == False:
                        continue
                    x_list.append(x)
                    z_list.append(z)
                
                if B_theta is not None:
                    #Optimize with the sample
                    b_samp = b+B_theta*np.matrix(sample).T
                    
                    x, _,z = MA_Problems.Optimize_LP_WarmStart_withOBJ(A,b_samp,c, VBasis, CBasis)
                    if x == False:
                        continue
                    x_list.append(x)
                    z_list.append(z)
                    
            Quasi_MC_X_WS = pd.DataFrame(x_list)
            Quasi_MC_Z_WS = pd.DataFrame(z_list)
            
            run_timeHS.append(time.time()-start_time)
            
            result = np.quantile(Quasi_MC_Z_WS,Quantiles2)
            L1HS.append(sum(abs(Basis_Q_Values-result))/99)
            L2HS.append(np.sqrt(sum((Basis_Q_Values-result)**2/99)))
            KSHS.append(ks_2samp(MC_Z['0'],Quasi_MC_Z_WS[0]).statistic)
            
        Times.append([np.mean(run_timeMC),np.mean(run_timeLH),np.mean(run_timeHS)])
        L1.append([np.mean(L1MC),np.mean(L1LH),np.mean(L1HS)])
        L2.append([np.mean(L2MC),np.mean(L2LH),np.mean(L2HS)])
        KS.append([np.mean(KSMC),np.mean(KSLH),np.mean(KSHS)])
        L1_std.append([np.std(L1MC),np.std(L1LH),np.std(L1HS)])
        L2_std.append([np.std(L2MC),np.std(L2LH),np.std(L2HS)])
        KS_std.append([np.std(KSMC),np.std(KSLH),np.std(KSHS)])
        
    Times = np.array(Times).T
    L1 = np.array(L1).T
    L2 = np.array(L2).T
    KS = np.array(KS).T
    L1_std = np.array(L1_std).T
    L2_std = np.array(L2_std).T
    KS_std = np.array(KS_std).T
    Results.append([Times,L1,L2,KS,L1_std,L2_std,KS_std])

pickle.dump( Results, open( "All results sampling.p", "wb" ) )

#%% Plot computation time vs accuracy
Results= pickle.load(open( "All results sampling.p", "rb" ) )
Sample_List = [100,300,500,700,1000,5000,10000]
Colors = ['b','r','g']
alpha = 0.5
for i in range(3):
    
    Norm_value = Results[i][2][0].max()
    plt.figure()
    plt.plot(Sample_List, Results[i][2][0], label="Benchmark",color = Colors[0])
    plt.fill_between(Sample_List, Results[i][2][0] + Results[i][5][0], Results[i][2][0]- Results[i][5][0],alpha=alpha,color = Colors[0])

    plt.plot(Sample_List, Results[i][2][1], label="Latin hypercube",color = Colors[1])
    plt.fill_between(Sample_List, Results[i][2][1] + Results[i][5][1], Results[i][2][1]- Results[i][5][1],alpha=alpha,color = Colors[1])

    plt.plot(Sample_List, Results[i][2][2], label="Halton sequence",color = Colors[2])
    plt.fill_between(Sample_List, Results[i][2][2] + Results[i][5][2], Results[i][2][2]- Results[i][5][2],alpha=alpha,color = Colors[2])
    plt.xscale('log')
    plt.xlim(100,10000)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('RMSE [€]')
    
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
plt.savefig('Plots\RMSE All.pdf')

#%% Plot computation time vs accuracy
Results= pickle.load(open( "All results sampling.p", "rb" ) )
Colors = ['b','r','g']

for i in range(3):
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    
    par1 = host.twinx()
    par2 = host.twinx()
    
    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))
    
    par2.axis["right"].toggle(all=True)
    par1.axis["right"].toggle(all=True)

    
    host.set_xlabel("Samples")
    host.set_ylabel("L1 [€]")
    par1.set_ylabel("L2 [€]")
    par2.set_ylabel("KS statistic")
    
    host.plot(Sample_List, Results[i][1][0], label="L1 Monte Carlo",linestyle='solid',color = Colors[0])
    host.plot(Sample_List, Results[i][1][1], label="L1 latin hypercube",linestyle='dotted',color = Colors[0])
    host.plot(Sample_List, Results[i][1][2], label="L1 Halton sequence",linestyle='dashed',color = Colors[0])
    
    par1.plot(Sample_List, np.sqrt(Results[i][2][0]), label="L2 Monte Carlo",linestyle='solid',color = Colors[1])
    par1.plot(Sample_List, np.sqrt(Results[i][2][1]), label="L2 latin hypercube",linestyle='dotted',color = Colors[1])
    par1.plot(Sample_List, np.sqrt(Results[i][2][2]), label="L2 Halton sequence",linestyle='dashed',color = Colors[1])

    par2.plot(Sample_List, Results[i][3][0], label="KS Monte Carlo",linestyle='solid',color = Colors[2])
    par2.plot(Sample_List, Results[i][3][1], label="KS latin hypercube",linestyle='dotted',color = Colors[2])
    par2.plot(Sample_List, Results[i][3][2], label="KS Halton sequence",linestyle='dashed',color = Colors[2])
    
    host.legend()
    plt.xscale('log')
    
    host.axis["left"].label.set_color(Colors[0])
    par1.axis["right"].label.set_color(Colors[1])
    par2.axis["right"].label.set_color(Colors[2])
    
    plt.draw()
    plt.show()


