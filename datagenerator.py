import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

from scipy.cluster.vq import vq, kmeans, whiten

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
    Y = np.where(A > gamma, A, 0) #Hard Threshold
    
    # Generate Error Matrix
    E = np.random.normal(0,0.01,(dim1,dim1))
    
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

# def review_paper_data(dim1,dim2): #X= UDV.T + Z  #dim1=n dim2=p
#     U = np.random.standard_normal((dim1,dim2))
#     V = ortho_group.rvs(dim2)
#     D = np.zeros((dim2,dim2))
#     for i in range (dim2):
#         D[i,i] = 10- np.sqrt(i+1)
#     Z = np.random.normal(0,0.01,(dim1,dim2))
    
#     X = U@D@V.T + Z
#     return X


def graph_data(node_number,group_number):
    B = np.array(([0.6,0.2,0.1,0.1],[0.2,0.7,0.05,0.05],[0.1,0.05,0.6,0.25],[0.1,0.05,0.25,0.6]),dtype='float64') 
    B = B*00.5

    Nodes = np.random.randint(group_number,size=(node_number))

    A = np.zeros((node_number,node_number))

    for i in range(node_number):
        for j in range(node_number):
            A[i,j] = np.random.binomial(1,B[Nodes[i],Nodes[j]])
    return A,Nodes

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
    
success = np.zeros((10))
for iter in range(10):
    A,Nodes = graph_data(900,4)
    s = 20
    k = 4
    omega = SPCA(A,s)
    
    pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
    pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
    
    centroids, _ = kmeans(component4, 4)
    clx,_ = vq(component1,centroids)
    
    pred = np.argmax(component4,axis=1)
    
    for i in range(900):
        if (pred[i] == Nodes[i]):
            success[iter] = success[iter] + 1

print(np.mean(success))

#%%
var_results = np.zeros((6,5,10))  #algo,different val, mean
time_results = np.zeros((6,5,10))
    


for iter in range(5):
    for iter2 in range (10):
        print("-------iter:",iter,".",iter2,"--------")
        X = review_paper_data(500+100*iter,5000+500*iter)
        s = 10
        k = 4
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
#        print("PCW          ",sum(variance6))
#        var_results[5,iter,iter2] = sum(variance6)
    # print("GCW          ",sum(variance7))

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