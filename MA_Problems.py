# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:16:23 2019

@author: U0003573
"""
#General
import pandas as pd  
import numpy as np

#Sampling
from chaospy import create_latin_hypercube_samples, create_halton_samples

#Probability
from scipy.stats import norm

#Sparse inverse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

#Optimization
import gurobipy as gp
from gurobipy import GRB

#Statistic
from scipy.stats import ks_2samp


def Optimize_LP(A,b,c):
    n=A.shape[1]
    b = np.array(b.T)[0]
    c = np.array(c.T)[0]
    
    model = gp.Model()
    model.Params.LogToConsole = 0
    #Variables
    x = model.addMVar(shape=n,vtype=GRB.CONTINUOUS, name="x")
    
    #Restrictions
    model.addConstr(A @ x == b, name="constr")

    #OFC
    model.setObjective(c @ x, GRB.MINIMIZE)
    
    model.optimize()
    
            #Check optimality
    if model.Status!= 2:
        print('No optimal solution found.')
        return False, False, False, False
    
    return model.x,  np.array(model.VBasis)==0, model.VBasis, model.CBasis

def Optimize_LP_withOBJ(A,b,c):
    n=A.shape[1]
    b = np.array(b.T)[0]
    c = np.array(c.T)[0]
    
    model = gp.Model()
    model.Params.LogToConsole = 0
    #Variables
    x = model.addMVar(shape=n,vtype=GRB.CONTINUOUS, name="x")
    
    #Restrictions
    model.addConstr(A @ x == b, name="constr")

    #OFC
    model.setObjective(c @ x, GRB.MINIMIZE)
    
    model.optimize()
    
            #Check optimality
    if model.Status!= 2:
        print('No optimal solution found.')
        return False, False
    
    return model.x, model.objVal

def Optimize_LP_WarmStart(A,b,c,VBasis, CBasis):
    n=A.shape[1]
    b = np.array(b.T)[0]
    c = np.array(c.T)[0]
    
    model = gp.Model()
    model.Params.LogToConsole = 0
    
    #Variables
    x = model.addMVar(shape=n,vtype=GRB.CONTINUOUS, name="x")
    
    #Restrictions
    model.addConstr(A @ x == b, name="constr")

    #OFC
    model.setObjective(c @ x, GRB.MINIMIZE)
    
    #Warm start
    model.update()
    model.setAttr("VBasis", model.getVars(),VBasis)
    model.setAttr("CBasis", model.getConstrs(),CBasis)
    
    model.optimize()
    
            #Check optimality
    if model.Status!= 2:
        print('No optimal solution found.')
        return False, False
    
    return model.x,  np.array(model.VBasis)==0

def Optimize_LP_WarmStart_withOBJ(A,b,c,VBasis, CBasis):
    n=A.shape[1]
    b = np.array(b.T)[0]
    c = np.array(c.T)[0]
    
    model = gp.Model()
    model.Params.LogToConsole = 0
    
    #Variables
    x = model.addMVar(shape=n,vtype=GRB.CONTINUOUS, name="x")
    
    #Restrictions
    model.addConstr(A @ x == b, name="constr")

    #OFC
    model.setObjective(c @ x, GRB.MINIMIZE)
    
    #Warm start
    model.update()
    model.setAttr("VBasis", model.getVars(),VBasis)
    model.setAttr("CBasis", model.getConstrs(),CBasis)
    
    model.optimize()
    
            #Check optimality
    if model.Status!= 2:
        print('No optimal solution found.')
        return False, False, False
    
    return model.x,  np.array(model.VBasis)==0, model.objVal

def Normal_MC(means,Sigma,Samples):
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma+ 0.005*np.identity(d))
    u = np.random.normal(size =Samples*d).reshape(d,Samples)
    x = np.dot(L, u)
    for i in range(Samples):
        x[:,i]= means + x[:,i]
    return x.T

def Normal_LHS(means,Sigma,Samples):
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma+ 0.005*np.identity(d))
    u = create_latin_hypercube_samples(order=Samples, dim=d)
    for i in range(Samples):
        u[:, i] = norm.ppf(u[:, i])
    x = np.dot(L, u)
    for i in range(Samples):
        x[:,i]= means + x[:,i]
    return x.T

def Normal_HAL(means,Sigma,Samples):
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma+ 0.005*np.identity(d))
    u = create_halton_samples(order=Samples, dim=d)
    for i in range(Samples):
        u[:, i] = norm.ppf(u[:, i])
    x = np.dot(L, u)
    for i in range(Samples):
        x[:,i]= means + x[:,i]
    return x.T

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return print('\n'.join(rv)+ '\n')

def Process_RHS_Sparse(A,basis_vars):
    #Basis Matrix
    B=csc_matrix(A[:,basis_vars])
    
    B_inv = inv(B)
    return B_inv.todense()

def Process_RHS(A,basis_vars):
    #Basis Matrix
    B=A[:,basis_vars]
    
    B_inv = np.linalg.inv(B)
    return B_inv

def Process_LP(A,c,basis_vars,C_theta):
    B=A[:,basis_vars]
    D=A[:,~basis_vars]
    c_B = c[basis_vars]
    c_D  = c[~basis_vars]
    C_theta_B = C_theta[basis_vars,:]
    C_theta_D = C_theta[~basis_vars,:]
    return B, D, c_B, c_D,C_theta_B, C_theta_D

def Process_OFC_Sparse(A,c,basis_vars,C_theta):
    B=A[:,basis_vars]
    D=A[:,~basis_vars]
    c_B = c[basis_vars]
    c_D  = c[~basis_vars]
    C_theta_B = C_theta[basis_vars,:]
    C_theta_D = C_theta[~basis_vars,:]
    B_inv = inv(csc_matrix(B)).todense()
    return B_inv, D, c_B, c_D,C_theta_B, C_theta_D

def Process_OFC(A,c,basis_vars,C_theta):
    B=A[:,basis_vars]
    D=A[:,~basis_vars]
    c_B = c[basis_vars]
    c_D  = c[~basis_vars]
    C_theta_B = C_theta[basis_vars,:]
    C_theta_D = C_theta[~basis_vars,:]
    B_inv = np.linalg.inv(B)
    return B_inv, D, c_B, c_D,C_theta_B, C_theta_D

def Region_Sampling_Intelligent(Sampled,A,b,c,B_theta=None,C_theta=None,epsilon=10**-10,N_Regions_Max=500,Stop_Percent = 100,Sample_Try = 100, KS_Statistic=0.01):
    #Warm start solution
    x0, basis_vars, VBasis, CBasis = Optimize_LP(A,b,c)
    #Number of variables
    m = A.shape[1]
    Total_Samples = Sampled.shape[0]
    
    Critical_Regions = 0
    points_past = 0
    first = True
    first_evaluation = True
    KS = []
    #Determine critical regions of the samples
    while len(Sampled)>0:
        #Region Number
        #Take first point
        sample = Sampled[0]
        if C_theta is not None:
            #Optimize with the sample
            x, basis_vars, z = Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
    
            #Calculate Matices of basis B
            B, D, c_B, c_D,C_theta_B, C_theta_D = Process_LP(A,c,basis_vars,C_theta)
            
            B_inv_D_T = spsolve(B,D).T
            
            #Determine polytrope representation
            A_p = -C_theta_D+B_inv_D_T*C_theta_B
            b_p = c_D-B_inv_D_T*c_B
            
            #Means 
            mu_x = np.matrix(x).T
            
        if B_theta is not None:
            #Optimize with the sample
            x, basis_vars, z = Optimize_LP_WarmStart_withOBJ(A,b+B_theta*np.matrix(sample).T,c, VBasis, CBasis)
            
            #Delete infeasable sample
            if x == False:
                Sampled = Sampled[1:]
                continue
            
            #Calculate Matices of basis B
            B=A[:,basis_vars]
            B_inv_bB_theta = spsolve(B,np.concatenate((b,B_theta),axis=1))
            b_p = np.matrix(B_inv_bB_theta[:,0]).T
            A_p = -B_inv_bB_theta[:,1:]
    
        #Discard sampling points that are in critical region
        condition = A_p@Sampled.T
        condition = condition<= b_p +epsilon*np.ones((b_p.shape[0],1))
        condition = np.ones((1,b_p.shape[0]))@condition
        condition = np.array(condition==b_p.shape[0])[0]
        
        Points = Sampled[condition].T
        Sampled = Sampled[~condition]
        
        #Break if no points are beeing eliminated
        if Points.shape[1]==0:
            print('No point was eliminated, normaly because of numerical errors. \n Take a bigger epsilon value.')
            Sampled = Sampled[1:]
            continue
        
        #Interpolate samples
        if C_theta is not None:
            X_points = np.hstack([mu_x]*Points.shape[1])
            Z_points = (mu_x.T*(c+C_theta*Points))
            
        if B_theta is not None:
            X_points_2 = np.hstack([b_p]*Points.shape[1]) -A_p@Points
            X_points = np.zeros((m,Points.shape[1]))
            X_points[basis_vars,:]=X_points_2
            Z_points = c.T@X_points
    
        #Append points
        if first:
            X = X_points
            Z = Z_points
            first = False
        else:
            X = np.concatenate((X,X_points),axis=1)
            Z = np.concatenate((Z,Z_points),axis=1) 
            
        Critical_Regions +=1
        print('Points to be evaluated:' + str(Sampled.shape[0]))
        print('Number of Regions: '+str(Critical_Regions))
        
        #Break if number of regions over the maximum
        if Critical_Regions>N_Regions_Max:
            print('Maximal region amount achieved.')
            return X,Z,KS
        
        Evaluated = Total_Samples-Sampled.shape[0]
        #Break if number of regions over the maximum
        if 100*(Evaluated)/Total_Samples>=Stop_Percent:
            print('Sample percentage achieved.')
            return X,Z,KS
        
        if first_evaluation:
            if Evaluated >Sample_Try:
                first_evaluation = False
                Z_eval = Z
                next_try = Sample_Try+(Evaluated//Sample_Try)*Sample_Try
                
        elif Evaluated>next_try:
            next_try = Sample_Try+(Evaluated//Sample_Try)*Sample_Try
            stat = ks_2samp(np.array(Z)[0],np.array(Z_eval)[0]).statistic
            KS.append(stat)
            if stat<KS_Statistic:
                print('Statistic achieved')
                return X,Z,KS
            Z_eval = Z
        
    return X,Z,KS

def Region_Sampling(Sampled,A,b,c,B_theta=None,C_theta=None,epsilon=10**-10,N_Regions_Max=500,Stop_Percent = 100):
    #Warm start solution
    x0, basis_vars, VBasis, CBasis = Optimize_LP(A,b,c)
    #Number of variables
    m = A.shape[1]
    Total_Samples = Sampled.shape[0]
    
    Critical_Regions = 0
    points_past = 0
    Point_list = []
    first = True
    #Determine critical regions of the samples
    while len(Sampled)>0:
        #Region Number
        #Take first point
        sample = Sampled[0]
        if C_theta is not None:
            #Optimize with the sample
            x, basis_vars, z = Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
    
            #Calculate Matices of basis B
            B, D, c_B, c_D,C_theta_B, C_theta_D = Process_LP(A,c,basis_vars,C_theta)
            
            B_inv_D_T = spsolve(B,D).T
            
            #Determine polytrope representation
            A_p = -C_theta_D+B_inv_D_T*C_theta_B
            b_p = c_D-B_inv_D_T*c_B
            
            #Means 
            mu_x = np.matrix(x).T
            
        if B_theta is not None:
            #Optimize with the sample
            x, basis_vars, z = Optimize_LP_WarmStart_withOBJ(A,b+B_theta*np.matrix(sample).T,c, VBasis, CBasis)
            
            #Delete infeasable sample
            if x == False:
                Sampled = Sampled[1:]
                continue
            
            #Calculate Matices of basis B
            B=A[:,basis_vars]
            B_inv_bB_theta = spsolve(B,np.concatenate((b,B_theta),axis=1))
            b_p = np.matrix(B_inv_bB_theta[:,0]).T
            A_p = -B_inv_bB_theta[:,1:]
    
        #Discard sampling points that are in critical region
        condition = A_p@Sampled.T
        condition = condition<= b_p +epsilon*np.ones((b_p.shape[0],1))
        condition = np.ones((1,b_p.shape[0]))@condition
        condition = np.array(condition==b_p.shape[0])[0]
        
        Points = Sampled[condition].T
        Sampled = Sampled[~condition]
        
        #Break if no points are beeing eliminated
        if Points.shape[1]==0:
            print('No point was eliminated, normaly because of numerical errors. \n Take a bigger epsilon value.')
            Sampled = Sampled[1:]
            continue
            
        #Interpolate samples
        if C_theta is not None:
            X_points = np.hstack([mu_x]*Points.shape[1])
            Z_points = (mu_x.T*(c+C_theta*Points))
            
        if B_theta is not None:
            X_points_2 = np.hstack([b_p]*Points.shape[1]) -A_p@Points
            X_points = np.zeros((m,Points.shape[1]))
            X_points[basis_vars,:]=X_points_2
            Z_points = c.T@X_points
    
        #Append points
        if first:
            X = X_points
            Z = Z_points
            first = False
        else:
            X = np.concatenate((X,X_points),axis=1)
            Z = np.concatenate((Z,Z_points),axis=1) 
            
        Critical_Regions +=1
        Point_list.append(Points.shape[1])
        print('Points to be evaluated:' + str(Sampled.shape[0]))
        print('Number of Regions: '+str(Critical_Regions))
    
        #Break if number of regions over the maximum
        if Critical_Regions>N_Regions_Max:
            print('Maximal region amount achieved.')
            return X,Z,Point_list
            
        #Break if number of regions over the maximum
        if 100*(Total_Samples-Sampled.shape[0])/Total_Samples>=Stop_Percent:
            print('Sample percentage achieved.')
            return X,Z,Point_list
        
        
    return X,Z,Point_list


def Characterize_regions(A,b,c,mean, cov, Prob_Samples, B_theta=None,C_theta=None, G= None, g = None, epsilon = 10**-15):

    Sampled = np.random.multivariate_normal(mean,cov,size=Prob_Samples)
    
    #Warm start solution
    x0, basis_vars, VBasis, CBasis = Optimize_LP(A,b,c)
    #Number of variables
    m = A.shape[1]
    
    Critical_Regions = []
    points_past = 0
    #Determine critical regions of the samples
    while len(Sampled)>0:
        #Region Number
        
        #Take first point
        sample = Sampled[0]
        
        if C_theta is not None:
            #Optimize with the sample
            x, basis_vars, z = Optimize_LP_WarmStart_withOBJ(A,b,c+C_theta*np.matrix(sample).T, VBasis, CBasis)
    
            #Calculate Matices of basis B
            B_inv, D, c_B, c_D,C_theta_B, C_theta_D = Process_OFC(A,c,basis_vars,C_theta)
            
            #Determine polytrope representation
            A_p = np.array(-C_theta_D+D.T*B_inv.T*C_theta_B)
            b_p = np.array(c_D-D.T*B_inv.T*c_B).T[0]
            if G is not None:
                A_p = np.concatenate((A_p,G),axis=0)
                b_p = np.concatenate((b_p,g),axis=0)
            
            #H-Representation for calculating the integral
            A_p_int = b.T*B_inv.T*C_theta_B
            A_p_int = np.concatenate((A_p,A_p_int,-A_p_int),axis=0)
            
            #H-Representation for calculating the integral in z
            A_p_x = np.zeros((m,1))
            
            #Means 
            mu_x = x
            mu_z = c.T*np.matrix(x).T
            
        if B_theta is not None:
            
            b_samp = b+B_theta*np.matrix(sample).T
            
            #Optimize with the sample
            x, basis_vars, z = Optimize_LP_WarmStart_withOBJ(A,b_samp,c, VBasis, CBasis)
            
            #Delete infeasable sample
            if x == False:
                Sampled = Sampled[1:]
                continue
            
            #Calculate Matices of basis B
            B_inv = Process_RHS(A,basis_vars)
            c_B = c[basis_vars]
            
            #Determine polytrope representation
            A_p = np.array(-B_inv*B_theta)
            b_p = np.array((B_inv * b).T)[0]
            if G is not None:
                A_p = np.concatenate((A_p,G),axis=0)
                b_p = np.concatenate((b_p,g),axis=0)
            
            
            #H-Representation for calculating the integral in z
            A_p_int = c_B.T*B_inv*B_theta
            A_p_int = np.concatenate((A_p,A_p_int,-A_p_int),axis=0)
            
            #H-Representation for calculating the integral in x
            A_p_x = np.matrix(A_p)
            
            #Means 
            mu_x = np.zeros(m)
            mu_x[basis_vars] =  np.array(B_inv*b).T[0]
            mu_z = c.T*np.matrix(x).T
            
        #Discard sampling points that are in critical region
        condition = A_p@Sampled.T
        condition2 = []
        for i in range(condition.shape[1]):
            n_true = 0
            for j in range(condition.shape[0]):
                if condition[j,i]-b_p[j]<= epsilon:
                    n_true+=1
                    if n_true ==condition.shape[0]:
                        condition2.append(False)
                else:
                    condition2.append(True)
                    break
                
        Sampled = Sampled[condition2]
        
        Critical_Regions.append([x,z,sample,A_p_int,b_p,mu_z,mu_x,A_p_x])
        print('Points to be evaluated:' + str(sum(condition2)))
        print('Number of Regions: '+str(len(Critical_Regions)))
        
        #Break if no points are beeing eliminated
        if points_past == sum(condition2):
            print('No point was eliminated, normaly because of numerical errors. \n Take a bigger epsilon value.')
            return [x,z,sample,A_p_int,b_p,mu_z,mu_x]
        points_past = sum(condition2)
    
    Critical_Regions = pd.DataFrame(Critical_Regions,columns=['X','Z','Sample','A_p','b_p','mu_z','mu_x','BinvBtheta'])
    
    return Critical_Regions

