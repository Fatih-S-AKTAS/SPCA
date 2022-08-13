# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:52:55 2021

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

from scipy.cluster.vq import vq, kmeans, whiten

# import networkx as nx

from sklearn.cluster import SpectralClustering

#rest
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh


from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

from scipy import eye, asarray, dot, sum
from scipy.linalg import svd, polar

from PCA_SPAR import SPCA
import time

#%% Varimax Algo
def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):

    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)

def PRS(A,k,gamma):
    U,S,V = randomized_svd(A, n_components=k, n_iter=3, random_state=None)

    U2 = varimax(U)             #Varimax
    U3 = np.abs(U2) - gamma*np.ones((U2.shape))   # Soft Threshold
    U3 = np.where(U3 > gamma, U2, 0) #Hard Threshold
    U3 = np.sign(U2)*U3
    return U3

def SCA(A,k,gamma):
    Z, Sig, Y = randomized_svd(A, n_components=k, n_iter=3, random_state=None) #init
    
    for i in range (1):
        Y = PRS(A.T@Z , k ,gamma)    #PRS
        Z, _ = polar(A@Y, side='left')  #Polar
    return Y



#%% Varimax Paper Synthetic Data 1

def varimax_paper_data(dim1, dim2):
    #example dims: dim1 = 200, dim2 = 16
    
    #Generate unitary U,V
    V = ortho_group.rvs(dim2)
    U_large = ortho_group.rvs(dim1)
    U = U_large[:,0:dim2]
    
    #Generate Diagonal
    S = np.zeros((dim2,dim2))
    for i in range (dim2):
        S[i,i] = 10- np.sqrt(i+1)
    
    #Generate A and make it sparse using gamma 
    A = U@S@V.T
    
    #Sparse Loadings
    Y_large = ortho_group.rvs(dim1)
    Y = Y_large[:,0:dim2]
    
    gamma = 0.15
    Y = np.where(np.abs(Y) > gamma, Y, 0) #Hard Threshold
    
    # Generate Error Matrix
    E = np.random.normal(0,0.01,(dim1,dim1))
    
    #Data
    X = A@Y.T + E
    return X

def varimax_paper_data_mk2(dim1, dim2,dim3):
    #example dims: dim1 = 200, dim2 = 16
    
    #Generate unitary U,V
    V = ortho_group.rvs(dim2)
    U_large = ortho_group.rvs(dim1)
    U = U_large[:,0:dim2]
    
    #Generate Diagonal
    S = np.zeros((dim2,dim2))
    for i in range (dim2):
        S[i,i] = 10- np.sqrt(i+1)
    
    #Generate A and make it sparse using gamma 
    A = U@S@V.T
    
    #Sparse Loadings
    Y_large = ortho_group.rvs(dim3)
    Y = Y_large[:,0:dim2]
    
    gamma = 0.15
    Y = np.where(np.abs(Y) > gamma, Y, 0) #Hard Threshold
    
    # Generate Error Matrix
    E = np.random.normal(0,0.01,(dim1,dim3))
    
    #Data
    X = A@Y.T + E
    return X


def review_paper_data(dim1,dim2): #X= UDV.T + Z  #dim1=n dim2=p
    U = np.random.standard_normal((dim1,dim2))
    V = ortho_group.rvs(dim2)
    D = np.zeros((dim2,dim2))
    for i in range (dim2):
        D[i,i] = 10- np.sqrt(i+1)
    Z = np.random.normal(0,0.01,(dim1,dim2))
    
    X = U@D@V.T + Z
    return X


def graph_data(node_number,group_number):
    B = np.array(([0.7,0.1,0.1,0.1],[0.2,0.7,0.05,0.05],[0.1,0.05,0.7,0.15],[0.1,0.05,0.15,0.7]),dtype='float64') 
    B = B*0.5

    Nodes = np.random.randint(group_number,size=(node_number))

    A = np.zeros((node_number,node_number))

    for i in range(node_number):
        for j in range(node_number):
            A[i,j] = np.random.binomial(1,B[Nodes[i],Nodes[j]])
            if (i==j):
                A[i,j] = 1
        
    sums = np.sum(A,axis=1)
    D = np.diag(sums)            
                
    return A,Nodes,D

"""

#%% Graph
    
#node_number = 900
#group_number = 4
#
#B = np.array(([0.6,0.2,0.1,0.1],[0.2,0.7,0.05,0.05],[0.1,0.05,0.6,0.25],[0.1,0.05,0.25,0.6]),dtype='float64') 
#B = B*0.05
#
#Nodes = np.random.randint(group_number,size=(node_number))
#
#A = np.zeros((node_number,node_number))
#
#for i in range(node_number):
#    for j in range(node_number):
#        A[i,j] = np.random.binomial(1,B[Nodes[i],Nodes[j]])
#    
#plt.spy(A)
#   
    
success = np.zeros((1))
success2 = np.zeros((1))

node_number = 100
for iter in range(1):
    A,Nodes,D = graph_data(node_number,4)
    L = D-A
    
    s = np.floor(node_number/4)
    s = int(s)

    k = 4
    
#    evals_all, evecs_all = eigh(A)
    
    omega = SPCA(A,s)
    omega.search_multiplier = node_number/s
    
    pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
    pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
    
    #0-Pad loadings
    loading1 = np.zeros((node_number,k))
    loading4 = np.zeros((node_number,k))
    
    for i in range(4):    
        for j in range(s):
            loading1[pattern1[i][j],i] = load1[j,i]
            loading4[pattern4[i][j],i] = load4[j,i]
    
    
    #kmeans
    centroids, _ = kmeans(loading1, 4)
    clx,_ = vq(loading1,centroids)
    
    #max
    pred = np.argmax(np.abs(loading1),axis=1)
    
    #library
    clustering = SpectralClustering(n_clusters=4,
       assign_labels="kmeans",
       affinity="precomputed",
       random_state=0).fit(A)
    
    for i in range(node_number):
        if (pred[i] == Nodes[i]):
            success[iter] = success[iter] + 1
        if (clustering.labels_[i]==Nodes[i]):
            success2[iter] = success2[iter] + 1

            



print(np.mean(success))
print(np.mean(success2))
#%% Graph Plot
G2 = nx.from_numpy_matrix(A)

color_map = []
for node in G2:
    if (Nodes[node]==0):
        color_map.append('red')
    elif (Nodes[node]==1):
        color_map.append('blue')
    elif (Nodes[node]==2):
        color_map.append('yellow')
    else:
        color_map.append('green')

nx.draw(G2,node_color=color_map,with_labels=True)

#%% Pred Graph Plot max

color_map2 = []
for node in G2:
    if (pred[node]==0):
        color_map2.append('red')
    elif (pred[node]==1):
        color_map2.append('blue')
    elif (pred[node]==2):
        color_map2.append('yellow')
    else:
        color_map2.append('green')

nx.draw(G2,node_color=color_map2,with_labels=True)



#%% Pred Graph Plot kmeans

color_map3 = []
for node in G2:
    if (clx[node]==0):
        color_map3.append('red')
    elif (clx[node]==1):
        color_map3.append('blue')
    elif (clx[node]==2):
        color_map3.append('yellow')
    else:
        color_map3.append('green')

nx.draw(G2,node_color=color_map3,with_labels=True)

#%% Pred Spectral Lib Plot

color_map4 = []
for node in G2:
    if (clustering.labels_[node]==0):
        color_map4.append('red')
    elif (clustering.labels_[node]==1):
        color_map4.append('blue')
    elif (clustering.labels_[node]==2):
        color_map4.append('yellow')
    else:
        color_map4.append('green')

nx.draw(G2,node_color=color_map4,with_labels=True)

#%% Together Plot
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.flatten()
nx.draw(G2,node_color=color_map,with_labels=True, ax=ax[0])
nx.draw(G2,node_color=color_map2,with_labels=True, ax=ax[1])
nx.draw(G2,node_color=color_map3,with_labels=True, ax=ax[2])
nx.draw(G2,node_color=color_map4,with_labels=True, ax=ax[3])
plt.show()



#%%
var_results = np.zeros((6,1,10))  #algo,different val, mean
time_results = np.zeros((6,1,10))

gamma = 0.04  #X_sub is 3-sparse approximately with this gamma


for iter in range(1):
    for iter2 in range (1):
        print("-------iter:",iter,".",iter2,"--------")
        X = review_paper_data(100+100*iter,1000+500*iter)
        s = 3
        k = 1
        omega = SPCA(X,s)
        
        omega.search_multiplier = 100/s
        solve_sdp = False
        
        t0 = time.process_time()
        pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
        t1 = time.process_time()
        omega.restart()
        print("gerschgorin done")
        
        
        t2 = time.process_time()
        pattern2,eigens2,load2,component2,variance2 = omega.find_component("CCW",k)
        t3 = time.process_time()
        omega.restart()
        print("CCW done ")
        
        t4 = time.process_time()
        pattern3,eigens3,load3,component3,variance3 = omega.find_component("FCW",k)
        t5 = time.process_time()
        omega.restart()
        print("FCW done ")
        
        t6 = time.process_time()
        pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
        t7 = time.process_time()
        omega.restart()
        print("EM done ")
        
        t8 = time.process_time()
        pattern5,eigens5,load5,component5,variance5 = omega.find_component("Path",k)
        t9 = time.process_time()
        omega.restart()
        print("Path/Chol done ")
        
        t10 = time.process_time()
#        pattern6,eigens6,load6,component6,variance6 = omega.find_component("PCW",k)
        
        t11 = time.process_time()
        omega.restart()
        print("PCW done ")
        
        t12 = time.process_time()
        # pattern7,eigens7,load7,component7,variance7 = omega.find_component("GCW",k)
        t13 = time.process_time()
        omega.restart()
        print("GCW done ")
        
        t14 = time.process_time()
        res = SCA(X.T@X,k,gamma)
        t15 = time.process_time()
        varimax_sup = np.nonzero(res)
        X_sub = np.zeros((X.shape[0],varimax_sup[0].size))
        for col in range(varimax_sup[0].size):
            X_sub[:,col] = X[:,varimax_sup[0][col]]
        print("Varimax done ")
        
        varimax_eig, _ = eigh(X_sub.T@X_sub) 

        
        eig_orig,_ = eigh(X.T@X)
        

        
        print("----------------------------")
        print("gerschgorin  ",t1-t0)
        time_results[0,iter,iter2] = t1-t0
        print("correlation  ",t3-t2)
        time_results[1,iter,iter2] = t3-t2
        print("frobenius    ",t5-t4)
        time_results[2,iter,iter2] = t5-t4
        print("em           ",t7-t6)
        time_results[3,iter,iter2] = t7-t6
        print("cholesky     ",t9-t8)
        time_results[4,iter,iter2] = t9-t8
        print("PCW          ",t11-t10)
        time_results[5,iter,iter2] = t11-t10
        print("GCW          ",t13-t12)
        
        print('Varimax      ',t15-t14)
        
        print("----------------------------")
        print("gerschgorin  ",sum(variance1))
        var_results[0,iter,iter2] = sum(variance1)
        print("correlation  ",sum(variance2))
        var_results[1,iter,iter2] = sum(variance2)
        print("frobenius    ",sum(variance3))
        var_results[2,iter,iter2] = sum(variance3)
        print("em           ",sum(variance4))
        var_results[3,iter,iter2] = sum(variance4)
        print("cholesky     ",sum(variance5))
        var_results[4,iter,iter2] = sum(variance5)
        print("varimax      ",sum(varimax_eig[-1]))
#        print("PCW          ",sum(variance6))
#        var_results[5,iter,iter2] = sum(variance6)
#        print("GCW          ",sum(variance7))
        print('original     ',sum(eig_orig[-1]))
    
    
#%% Padding
loading1 = np.zeros((1000,k))
loading4 = np.zeros((1000,k))

for i in range(k):    
    for j in range(s):
        loading1[pattern1[i][j],i] = load1[j,i]
        loading4[pattern4[i][j],i] = load4[j,i]
        
evals_all, evecs_all = eigh(X.T@X) 
  
    

#%%
plt.figure()
plt.title('time')
plt.plot(np.mean(time_results[0,:,:],axis=1), color='red',label='gersch')
plt.plot(np.mean(time_results[1,:,:],axis=1), color='green',label='correlation')
plt.plot(np.mean(time_results[2,:,:],axis=1), color='orange',label='frob')
plt.plot(np.mean(time_results[3,:,:],axis=1), color='blue',label='em')
plt.plot(np.mean(time_results[4,:,:],axis=1), color='yellow',label='chol')
plt.legend(loc="upper left")
plt.xlabel('dimension')
plt.ylabel('run-time')
plt.savefig("timeplot5.pdf")

plt.figure()
plt.title('var')
plt.plot(np.mean(var_results[0,:,:],axis=1), color='red',label='gersch')
plt.plot(np.mean(var_results[1,:,:],axis=1), color='green',label='correlation')
plt.plot(np.mean(var_results[2,:,:],axis=1), color='orange',label='frob')
plt.plot(np.mean(var_results[3,:,:],axis=1), color='blue',label='em')
plt.plot(np.mean(var_results[4,:,:],axis=1), color='yellow',label='chol')
plt.legend(loc="upper left")
plt.xlabel('dimension')
plt.ylabel('total var')
plt.savefig("varplot5.pdf")

"""

