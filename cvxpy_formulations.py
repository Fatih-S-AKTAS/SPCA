import gc

sadge = gc.collect()

from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,argsort,eye
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm,qr
from scipy.sparse.linalg import eigsh
from scipy.special import huber
import time
from static_questions import pitprops
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure
from datagenerator import varimax_paper_data,review_paper_data,graph_data
import cvxpy


A = random.normal(4,10,[100,25])
# A = varimax_paper_data(1000,400)
# A = review_paper_data(400,4000)
# A = graph_data(2000,4)

# l,d,p = ldl(pitprops)
# A = l

m,n = shape(A)
    
mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA

A2 = A.T.dot(A)


sadge1 = gc.collect()

s = 8

omega = SPCA(A,s)


omega.search_multiplier = min(200/s,n/s)

k = 1

t0 = time.process_time()
pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
t1 = time.process_time()
omega.restart()
print("gerschgorin done ")

sadge1 = gc.collect()

t2 = time.process_time()
pattern2,eigens2,load2,component2,variance2 = omega.find_component("CCW",k)
t3 = time.process_time()
omega.restart()
print("CCW done ")

sadge1 = gc.collect()

t4 = time.process_time()
pattern3,eigens3,load3,component3,variance3 = omega.find_component("FCW",k)
t5 = time.process_time()
omega.restart()
print("FCW done ")

sadge1 = gc.collect()

t6 = time.process_time()
pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
t7 = time.process_time()
omega.restart()
print("EM done ")

sadge1 = gc.collect()

t8 = time.process_time()
pattern5,eigens5,load5,component5,variance5 = omega.find_component("Path",k)
t9 = time.process_time()
omega.restart()
print("Path/Chol done ")

sadge1 = gc.collect()

t10 = time.process_time()
# pattern6,eigens6,load6,component6,variance6 = omega.find_component("PCW",k)
t11 = time.process_time()
# omega.restart()
print("PCW done ")

# sadge1 = gc.collect()

t12 = time.process_time()
pattern7,eigens7,load7,component7,variance7 = omega.find_component("nesterov",k)
t13 = time.process_time()
omega.restart()
print("Nesterov done ")
sadge1 = gc.collect()

print("----------------------------")
print("gerschgorin  ",t1-t0)
print("correlation  ",t3-t2)
print("frobenius    ",t5-t4)
print("em           ",t7-t6)
print("cholesky     ",t9-t8)
print("PCW          ",t11-t10)
print("nesterov     ",t13-t12)
print("----------------------------")
print("gerschgorin  ",sum(variance1))
print("correlation  ",sum(variance2))
print("frobenius    ",sum(variance3))
print("em           ",sum(variance4))
print("cholesky     ",sum(variance5))
# print("PCW          ",sum(variance6))
print("nesterov     ",sum(variance7))

best_val = max([sum(variance1),sum(variance2),sum(variance3),sum(variance4),sum(variance5),sum(variance7)])

#%%

rho = 1
# cvxpy 1 norm constraint + trace maximization (fantope bullshit)
if n <= 25:
    X = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints = []
    constraints += [X >> 0]
    constraints += [eye(n) >> X]
    constraints += [cvxpy.trace(X) == k * 1]
    # constraints += [cvxpy.pnorm(X, 1) <= s]
    # constraints += [cvxpy.sum(cvxpy.abs(X)) <= k * s]
    
    
    obj = cvxpy.Maximize(cvxpy.trace(A2 @ X) - rho * cvxpy.pnorm(X, 1) )
    
    prob = cvxpy.Problem(obj,constraints)
    prob.solve(solver = cvxpy.CVXOPT)
    
    Z0 = X.value
    Z = Z0.copy()
    Z[abs(Z)<1e-3] = 0
    
    lamda0,vector0 = eigsh(Z0,k = k,tol = 1e-4)
    svector0 = vector0.copy()
    svector0[abs(svector0) < 1e-3] = 0
    
    pattern0 = argsort(abs(vector0[:,:k]),axis = 0)[-s:]
    svector1 = zeros([n,k])
    for i in range(k):
        svector1[pattern0[:,i],i] = vector0[pattern0[:,i],i]/norm(vector0[pattern0[:,i],i])
    
    q,r = qr(A.dot(svector1),mode = "economic")
    var_cvx = diag(r) ** 2 
    print("var",sum(var_cvx))
    

#%%

# cvxpy 1 norm constraint + trace maximization
if n <= 25:
    X = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints = []
    constraints += [X >> 0]
    constraints += [eye(n) >> X]
    constraints += [cvxpy.trace(X) == k * 1]
    # constraints += [cvxpy.pnorm(X, 1) <= s]
    constraints += [cvxpy.sum(cvxpy.abs(X)) <= k * s + 2*s]
    
    
    obj = cvxpy.Maximize(cvxpy.trace(A2 @ X))
    
    prob = cvxpy.Problem(obj,constraints)
    prob.solve(solver = cvxpy.CVXOPT)
    
    Z0 = X.value
    Z = Z0.copy()
    Z[abs(Z)<1e-3] = 0
    
    lamda0,vector0 = eigsh(Z0,k = k,tol = 1e-4)
    svector0 = vector0.copy()
    svector0[abs(svector0) < 1e-3] = 0
    
    pattern0 = argsort(abs(vector0[:,:k]),axis = 0)[-s:]
    svector1 = zeros([n,k])
    for i in range(k):
        svector1[pattern0[:,i],i] = vector0[pattern0[:,i],i]/norm(vector0[pattern0[:,i],i])
    
    q,r = qr(A.dot(svector1),mode = "economic")
    var_cvx = diag(r) ** 2 
        

#%%

# cvxpy trace constraint + 1 norm minimization
if n <= 25:
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X2 >> 0]
    constraints2 += [eye(n) >> X2]
    constraints2 += [cvxpy.trace(X2) == k]
    constraints2 += [cvxpy.trace(A2 @ X2) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.pnorm(X2,1))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.CVXOPT)
    
    first_copy = X2.value
    Y0 = X2.value
    Y = Y0.copy()
    Y[abs(Y)<1e-3] = 0
    
    lamda1,vector1 = eigsh(Y0,k = k,tol = 1e-4)
    svector1 = vector1.copy()
    svector1[abs(svector1) < 1e-3] = 0
    
    pattern1 = argsort(abs(vector1[:,0]))[-s:]
    sparsity1 = where(abs(svector1) > 0)[0]
    
    pattern0 = argsort(abs(vector1[:,:k]),axis = 0)[-s:]
    svector1 = zeros([n,k])
    for i in range(k):
        svector1[pattern0[:,i],i] = vector1[pattern0[:,i],i]/norm(vector1[pattern0[:,i],i])
    
    q,r = qr(A.dot(svector1),mode = "economic")
    var_cvx = diag(r) ** 2 

#%%
    
k = 2
# cvxpy trace constraint + 1 norm minimization
if n <= 25:
    X1 = cvxpy.Variable(shape = (n,n),symmetric = True)
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X1 >> 0]
    constraints2 += [X2 >> 0]
    constraints2 += [eye(n) >> X1 + X2]
    constraints2 += [cvxpy.trace(X1) == 1]
    constraints2 += [cvxpy.trace(X2) == 1]
    constraints2 += [cvxpy.trace(A2 @ X2) + cvxpy.trace(A2 @ X1) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.pnorm(X1,1) + cvxpy.pnorm(X2,1))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.CVXOPT)
    
    pc1 = X1.value
    pc2 = X2.value
    
    lamda1,vector1 = eigsh(pc1,k = k,tol = 1e-4)
    lamda2,vector2 = eigsh(pc2,k = k,tol = 1e-4)
    
    svector1 = vector1.copy()
    svector1[abs(svector1) < 1e-3] = 0
    
    pattern1 = argsort(abs(vector1[:,0]))[-s:]
    sparsity1 = where(abs(svector1) > 0)[0]
    
    svector2 = vector2.copy()
    svector2[abs(svector2) < 1e-3] = 0
    
    pattern2 = argsort(abs(vector2[:,0]))[-s:]
    sparsity2 = where(abs(svector2) > 0)[0]
    
    pattern0 = argsort(abs(vector1[:,:k]),axis = 0)[-s:]
    svector1 = zeros([n,k])
    for i in range(k):
        svector1[pattern0[:,i],i] = vector1[pattern0[:,i],i]/norm(vector1[pattern0[:,i],i])
    
    q,r = qr(A.dot(svector1),mode = "economic")
    var_cvx = diag(r) ** 2 
    
#%%
# cvxpy test, trace constraint + frobenius norm minimization
if n <= 25:
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X2 >> 0]
    constraints2 += [cvxpy.trace(X2) == 1]
    constraints2 += [cvxpy.trace(A2 @ X2) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.norm(X2,"fro"))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.CVXOPT)
    
    first_copy = X2.value
    Y0_3 = X2.value
    Y = Y0.copy()
    Y[abs(Y)<1e-3] = 0
    
    lamda1,vector1 = eigsh(Y0,k = 1,tol = 1e-4)
    svector1 = vector1.copy()
    svector1[abs(svector1) < 1e-3] = 0
    
    pattern1 = argsort(abs(vector1[:,0]))[-s:]
    sparsity1 = where(abs(svector1) > 0)[0]
    
#%%
# cvxpy test, trace constraint + reweighted 1 norm minimization

if n <= 25:
    L = 1/(abs(Y0) + 1e-3)
    
    for i in range(4):
        X3 = cvxpy.Variable(shape = (n,n),symmetric = True)
        constraints3 = []
        constraints3 += [X3 >> 0]
        constraints3 += [cvxpy.trace(X3) == 1]
        constraints3 += [cvxpy.trace(A2 @ X3) >= best_val]
        
        
        obj3 = cvxpy.Minimize(cvxpy.pnorm(cvxpy.multiply(L,X3),1))
        # obj3 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X3)))
        
        prob3 = cvxpy.Problem(obj3,constraints3)
        prob3.solve(solver = cvxpy.MOSEK)
        
        W0 = X3.value
        W = W0.copy()
        W[abs(W)<1e-3] = 0
        L = 1/(abs(W0) + 1e-3)


    lamda2,vector2 = eigsh(W0,k = 1,tol = 1e-4)
    svector2 = vector2.copy()
    svector2[abs(svector2) < 1e-3] = 0
    
    pattern2 = argsort(abs(vector2[:,0]))[-s:]
    sparsity2 = where(abs(svector2) > 0)[0]
    
    print("----------------------------------------------------------")
    print("real sparsity level",s)
    print("sparsity 1",len(sparsity1),omega.eigen_upperbound(sparsity1))
    print("sparsity 2",len(sparsity2),omega.eigen_upperbound(sparsity2))
    print("----------------------------------------------------------")
    print("pattern 0",omega.eigen_upperbound(pattern0))
    print("pattern 1",omega.eigen_upperbound(pattern1))
    print("pattern 2",omega.eigen_upperbound(pattern2))
