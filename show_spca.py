import gc

sadge = gc.collect()

#%%
from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,zeros
from numpy.linalg import matrix_rank,qr
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm
from scipy.sparse.linalg import eigsh
from scipy.special import huber
# from static_questions import * 
from matplotlib.pyplot import *
# import cvxpy
# from nesterov_wrapper import run_formulation

#%%

A = random.normal(4,10,[20,40])

# l,d,p = ldl(pitprops)
# A = l

# A = at_t_faces

m,n = shape(A)

mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA

A2 = A.T.dot(A)

#%%

s = 10
k = 4
omega = SPCA(A,s)

omega.search_multiplier = 4

solve_sdp = False

# gd,gd_val = omega.column_norm_1()
# gd_val2,gd_vector = omega.eigen_pair(gd)

t0 = time.process_time()
pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
t1 = time.process_time()
omega.restart()
print("gerschgorin done ")


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
# pattern6,eigens6,load6,component6,variance6 = omega.find_component("PCW",k)
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
print("correlation  ",t3-t2)
print("frobenius    ",t5-t4)
print("em           ",t7-t6)
print("cholesky     ",t9-t8)
print("PCW          ",t11-t10)
print("GCW          ",t13-t12)
print("----------------------------")
print("gerschgorin  ",sum(variance1))
print("correlation  ",sum(variance2))
print("frobenius    ",sum(variance3))
print("em           ",sum(variance4))
print("cholesky     ",sum(variance5))
# print("PCW          ",sum(variance6))
# print("GCW          ",sum(variance7))

#%%

# cvxpy 1 norm constraint + trace maximization
if n <= 25:
    X = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints = []
    constraints += [X >> 0]
    constraints += [cvxpy.trace(X) == 1]
    # constraints += [cvxpy.pnorm(X, 1) <= s]
    constraints += [cvxpy.sum(cvxpy.abs(X)) <= s]
    
    
    obj = cvxpy.Maximize(cvxpy.trace(A2 @ X))
    
    prob = cvxpy.Problem(obj,constraints)
    prob.solve(solver = cvxpy.CVXOPT)
    
    Z0 = X.value
    Z = Z0.copy()
    Z[abs(Z)<1e-3] = 0
    
    lamda0,vector0 = eigsh(Z0,k = 1,tol = 1e-4)
    svector0 = vector0.copy()
    svector0[abs(svector0) < 1e-3] = 0
    
    pattern0 = argsort(abs(vector0[:,0]))[-s:]

#%%
# cvxpy trace constraint + 1 norm minimization
if n <= 25:
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X2 >> 0]
    constraints2 += [cvxpy.trace(X2) == 1]
    constraints2 += [cvxpy.trace(A2 @ X2) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.pnorm(X2,1))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.CVXOPT)
    
    first_copy = X2.value
    Y0 = X2.value
    Y = Y0.copy()
    Y[abs(Y)<1e-3] = 0
    
    lamda1,vector1 = eigsh(Y0,k = 1,tol = 1e-4)
    svector1 = vector1.copy()
    svector1[abs(svector1) < 1e-3] = 0
    
    pattern1 = argsort(abs(vector1[:,0]))[-s:]
    sparsity1 = where(abs(svector1) > 0)[0]

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
    
# L = 1/(abs(Y0) + 1e-3)

# for i in range(4):
#     X3 = cvxpy.Variable(shape = (n,n),symmetric = True)
#     constraints3 = []
#     constraints3 += [X3 >> 0]
#     constraints3 += [cvxpy.trace(X3) == 1]
#     constraints3 += [cvxpy.trace(A2 @ X3) >= best_val]
    
    
#     obj3 = cvxpy.Minimize(cvxpy.pnorm(cvxpy.multiply(L,X3),1))
#     # obj3 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X3)))
    
#     prob3 = cvxpy.Problem(obj3,constraints3)
#     prob3.solve(solver = cvxpy.MOSEK)
    
#     W0 = X3.value
#     W = W0.copy()
#     W[abs(W)<1e-3] = 0
#     L = 1/(abs(W0) + 1e-3)

# lamda2,vector2 = eigsh(W0,k = 1,tol = 1e-4)
# svector2 = vector2.copy()
# svector2[abs(svector2) < 1e-3] = 0

# pattern2 = argsort(abs(vector2[:,0]))[-s:]
# sparsity2 = where(abs(svector2) > 0)[0]

# print("----------------------------------------------------------")
# print("real sparsity level",s)
# print("sparsity 1",len(sparsity1),omega.eigen_upperbound(sparsity1))
# print("sparsity 2",len(sparsity2),omega.eigen_upperbound(sparsity2))
# print("----------------------------------------------------------")
# print("pattern 0",omega.eigen_upperbound(pattern0))
# print("pattern 1",omega.eigen_upperbound(pattern1))
# print("pattern 2",omega.eigen_upperbound(pattern2))

#%%

# K = random.randn(n,n)
# K = K.T.dot(K)
# K = K/norm(K)

#%%
# cgal SDP, trace constraint + frobenius norm minimization
if solve_sdp and False:
    A_F = norm(A2) ** 2
    
    X = zeros([n,n])
    y = 0
    # b = 80
    b = best_val
    beta0 = 1
    t = 0
    
    A3 = A2/norm(A2)
    b = b/norm(A2)
    constraint_diff = b
    while t <= 4000:
        t += 1
        beta = beta0 * t ** 0.5
        eta = 2/t
        
        # ful_matrix = K + A3 * (y + beta * constraint_diff)
        val,vec = eigsh(X + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
        # ful_val,ful_vec = eigh(sign(X)/beta + A3 * (y + beta * constraint_diff))
        
        X = (1 - eta) * X + eta * vec.dot(vec.T)
        
        constraint_diff = tensordot(A3,X) - b
        print("constraint diff",constraint_diff)
        
        gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
        y = y + gamma * (constraint_diff)
    d,v = eigsh(X,k=3)
    cgal = argsort(-1 * abs(v[:,-1]))[:s]
    cgal_val = omega.eigen_upperbound(cgal)

#%%
# cgal SDP, trace constraint + reweighted frobenius norm minimization for 1 norm minimization
if solve_sdp and False:
    # b = 80
    b = best_val
    L = ones([n,n])
    A3 = A2/norm(A2)
    b = b/norm(A2)
    constraint_diff = b
    beta0 = 1
    for rew in range(4):
        X = zeros([n,n])
        y = 0
        t = 0
        while t <= 4000:
            t += 1
            beta = beta0 * t ** 0.5
            eta = 2/t
            
            # ful_matrix = K + A3 * (y + beta * constraint_diff)
            val,vec = eigsh(L * X + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
            # ful_val,ful_vec = eigh(sign(X)/beta + A3 * (y + beta * constraint_diff))
            
            X = (1 - eta) * X + eta * vec.dot(vec.T)
            
            constraint_diff = tensordot(A3,X) - b
            print("constraint diff",constraint_diff)
            
            gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
            y = y + gamma * (constraint_diff)
        L = 1/(1e-4 + abs(X)) ** 4
        L = L/norm(L) * n
        d,v = eigsh(X,k=3)
        cgal = argsort(-1 * abs(v[:,-1]))[:s]
        cgal_val = omega.eigen_upperbound(cgal)

#%%
# varimax stuff

# from sklearn.utils.extmath import randomized_svd
# import numpy as np
# from scipy.linalg import svd,polar

# def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):

#     p,k = Phi.shape
#     R = np.eye(k)
#     d=0
#     for i in range(q):
#         d_old = d
#         Lambda = np.dot(Phi, R)
#         u,s,vh = svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
#         R = np.dot(u,vh)
#         d = np.sum(s)
#         if d_old!=0 and d/d_old < 1 + tol: break
#     return np.dot(Phi, R)

# def PRS(A,k,gamma):
#     U,S,V = randomized_svd(A, n_components=k, n_iter=3, random_state=None)
#           #SVD
#     U2 = varimax(U)         #Varimax
#     U3 = np.where(U2 > gamma, U2, 0) #Hard Threshold
#     return U3

# def SCA(A,k,gamma):
#     Z, Sig, Y = randomized_svd(A, n_components=k, n_iter=3, random_state=None) #init

#     for i in range (10):
#         Y = PRS(A.T@Z , k ,gamma)    #PRS
#         Z,P  = polar(A@Y, side='left')  #Polar
#     return Y

# if n <= 1000:
#     Y = SCA(A,1,0.15)
#     pattern = where(abs(Y[:,0]) > 0)[0]

#%%

# cgal SDP, trace constraint + Huber norm minimization
# it works
if solve_sdp:
    A_F = norm(A2) ** 2
    
    X = zeros([n,n])
    momentum = zeros([n,n])
    y = 0
    # b = 80
    b = best_val
    beta0 = 1
    t = 0
    
    A3 = A2/norm(A2)
    b = b/norm(A2)
    constraint_diff = b
    # teta0 = 1e-2
    teta0 = 1/sqrt(n) * 0.2
    
    while t <= 10000:
        t += 1
        beta = beta0 * t ** 0.5
        teta = teta0/beta
        
        eta = 2/t
        
        # gradient = X.copy()
        gradient = X + 0
        i,j = where(X > teta)
        k,l = where(X < -teta)
        gradient[i,j] = teta
        gradient[k,l] = -teta
        
        val,vec = eigsh(gradient + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-3,which = "SA",maxiter = t ** 0.25 * log(n))
    
        X = (1 - eta) * X + eta * vec.dot(vec.T)
    
        constraint_diff = tensordot(A3,X) - b
        print("iteration",t,"constraint diff",constraint_diff)
        
        gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
        y = y + gamma * (constraint_diff)
    d2,v2 = eigsh(X,k=3)
    cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
    cgal_val2 = omega.eigen_upperbound(cgal2)
    sv2 = v2[:,-1].copy()
    sv2[abs(sv2)<1e-2] = 0
    fake_pattern = where(abs(v2[:,-1]) > 1e-2)[0]

#%%

# cgal SDP, trace constraint + Huber norm minimization
# it works
correction = [1e-3,1e-3,1e-4,1e-4,1e-5]
# lamda_max = [0]
if solve_sdp:
    L = ones([n,n])
    for r in range(5):
        X = zeros([n,n])
        y = 0
        # b = 80
        b = best_val
        beta0 = 1
        t = 0
        
        A3 = A2/norm(A2)
        b = b/norm(A2)
        constraint_diff = b
        # teta0 = 1e-2
        teta0 = 1/sqrt(n) * 0.2
        
        while t <= 10000:
            t += 1
            beta = beta0 * t ** 0.5
            teta = teta0/beta
            
            eta = 2/t
            
            gradient = X + 0
    
            i,j = where(X > teta)
            k,l = where(X < -teta)
            gradient[i,j] = teta
            gradient[k,l] = -teta
            
            val,vec = eigsh(L * gradient + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA",maxiter = t ** 0.25 * log(n),ncv = 20)
        
            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            constraint_diff = tensordot(A3,X) - b
            
            gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
            y = y + gamma * (constraint_diff)
            print("r",r,"iteration",t,"constraint diff",constraint_diff)
        d2,v2 = eigsh(X,k=3)
        cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
        cgal_val2 = omega.eigen_upperbound(cgal2)
        sv2 = v2[:,-1].copy()
        sv2[abs(sv2)<1e-2] = 0
        fake_pattern = where(abs(v2[:,-1]) > 1e-2)[0]
        L = 1/(correction[r] + abs(X))
        L = L/norm(L) * n
    
    
#%%

# cgal SDP, Huber norm constraint + trace maximization
# it works, change updates tho xd
# constraint value should be changed although it works anyway
if solve_sdp:
    A_F = norm(A2) ** 2
    
    X = zeros([n,n])
    momentum = zeros([n,n])
    y = 0
    # b = 80
    b = s
    beta0 = 1
    t = 0
    
    A3 = A2/norm(A2)
    b = b/norm(A2)
    b = b/5
    constraint_diff = b
    # teta0 = 1e-2
    teta0 = 1/sqrt(n) * 0.2
    
    while t <= 10000:
        t += 1
        beta = beta0 * t ** 0.5
        teta = teta0/beta
        
        eta = 2/t
        
        gradient = X.copy()
        i,j = where(X > teta)
        k,l = where(X < -teta)
        g,h = where(abs(X) <= teta)
        gradient[i,j] = teta
        gradient[k,l] = -teta
        
        val,vec = eigsh(-A3 + gradient * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
        # ful_val,ful_vec = eigh(A3 + gradient * (y + beta * constraint_diff))
    
        X = (1 - eta) * X + eta * vec.dot(vec.T)
        
        # huber_norm = teta * (sum(abs(X[i,j])) + sum(abs(X[k,l])) - 0.5 * teta * (len(i) + len(j)) ) + 0.5 * sum(X[g,h] ** 2)
        huber_norm = sum(huber(teta,X))
        constraint_diff = huber_norm - b
        print("iteration",t,"constraint diff",constraint_diff)
        
        gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
        y = y + gamma * (constraint_diff)
    d3,v3 = eigsh(X,k=3)
    cgal3 = argsort(-1 * abs(v3[:,-1]))[:s]
    cgal_val3 = omega.eigen_upperbound(cgal3)
    sv3 = v3[:,-1].copy()
    sv3[abs(sv3)<1e-2 * 5] = 0
    fake_pattern = where(abs(v3[:,-1]) > 1e-2 *5)[0]

#%%

# cgal SDP, Huber norm constraint + trace maximization
# it works, change updates tho xd
if solve_sdp:
    A_F = norm(A2) ** 2
    
    X = zeros([n,n])
    momentum = zeros([n,n])
    y = 0
    # b = 80
    b = s
    beta0 = 1
    t = 0
    
    A3 = A2/norm(A2)
    b = b/norm(A2)
    constraint_diff = b
    # teta0 = 1e-2
    teta0 = 1/sqrt(n) * 0.2
    
    while t <= 10000:
        t += 1
        beta = beta0 * t ** 0.5
        teta = teta0/beta
        b = 2 * (teta * s - 0.5 * teta ** 2 * n ** 2)
        # b = b/norm(A2)
        eta = 2/t
        
        gradient = X.copy()
        i,j = where(X > teta)
        k,l = where(X < -teta)
        g,h = where(abs(X) <= teta)
        gradient[i,j] = teta
        gradient[k,l] = -teta
        
        val,vec = eigsh(-A3 + gradient * (y + beta * constraint_diff),k = 1,tol = 1e-4,which = "SA")
        # ful_val,ful_vec = eigh(A3 + gradient * (y + beta * constraint_diff))
    
        X = (1 - eta) * X + eta * vec.dot(vec.T)
        
        # huber_norm = teta * (sum(abs(X[i,j])) + sum(abs(X[k,l])) - 0.5 * teta * (len(i) + len(j)) ) + 0.5 * sum(X[g,h] ** 2)
        huber_norm = sum(huber(teta,X))
        constraint_diff = huber_norm - b
        print("iteration",t,"constraint diff",constraint_diff)
        
        gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
        y = y + gamma * (constraint_diff)
    d3,v3 = eigsh(X,k=3)
    cgal3 = argsort(-1 * abs(v3[:,-1]))[:s]
    cgal_val3 = omega.eigen_upperbound(cgal3)
    sv3 = v3[:,-1].copy()
    sv3[abs(sv3)<1e-2 * 5] = 0
    fake_pattern = where(abs(v3[:,-1]) > 1e-2 *5)[0]
    