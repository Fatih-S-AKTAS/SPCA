import gc

gc.collect()

from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,argsort,sqrt
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,eye
from scipy.special import huber
from static_questions import pitprops
import time
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure


A = random.normal(4,10,[100,50])
# A = varimax_paper_data(1000,400)
# A = review_paper_data(400,4000)
# A = graph_data(2000,4)

# L = cholesky(pitprops)
# A = L

m,n = shape(A)

A = A - A.mean(axis = 0)

sA = std(A,axis = 0)

A = A/sA

A2 = A.T.dot(A)


gc.collect()

s = 20

omega = SPCA(A,s)


omega.search_multiplier = min(200/s,n/s)

k = 1

solve_sdp = False

t0 = time.process_time()
pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
t1 = time.process_time()
omega.restart()
print("gerschgorin done ")

gc.collect()

t2 = time.process_time()
pattern2,eigens2,load2,component2,variance2 = omega.find_component("CCW",k)
t3 = time.process_time()
omega.restart()
print("CCW done ")

gc.collect()

t4 = time.process_time()
pattern3,eigens3,load3,component3,variance3 = omega.find_component("FCW",k)
t5 = time.process_time()
omega.restart()
print("FCW done ")

gc.collect()

t6 = time.process_time()
pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
t7 = time.process_time()
omega.restart()
print("EM done ")

gc.collect()

t8 = time.process_time()
pattern5,eigens5,load5,component5,variance5 = omega.find_component("Path",k)
t9 = time.process_time()
omega.restart()
print("Path/Chol done ")

gc.collect()

t10 = time.process_time()
# pattern6,eigens6,load6,component6,variance6 = omega.find_component("PCW",k)
t11 = time.process_time()
# omega.restart()
print("PCW done ")

gc.collect()

t12 = time.process_time()
pattern7,eigens7,load7,component7,variance7 = omega.find_component("nesterov",k)
t13 = time.process_time()
omega.restart()
print("Nesterov done ")
gc.collect()

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

# # cgal SDP, trace constraint + frobenius norm minimization
# if solve_sdp:
#     A_F = norm(A2) ** 2
    
#     X = zeros([n,n])
#     y = 0
#     # b = 80
#     b = best_val
#     beta0 = 1
#     t = 0
    
#     A3 = A2/norm(A2)
#     b = b/norm(A2)
#     constraint_diff = b
#     while t <= 4000:
#         t += 1
#         beta = beta0 * t ** 0.5
#         eta = 2/t
        
#         # ful_matrix = K + A3 * (y + beta * constraint_diff)
#         val,vec = eigsh(X + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
#         # ful_val,ful_vec = eigh(sign(X)/beta + A3 * (y + beta * constraint_diff))
        
#         X = (1 - eta) * X + eta * vec.dot(vec.T)
        
#         constraint_diff = tensordot(A3,X) - b
#         print("constraint diff",constraint_diff)
        
#         gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
#         y = y + gamma * (constraint_diff)
#     d,v = eigsh(X,k=3)
#     cgal = argsort(-1 * abs(v[:,-1]))[:s]
#     cgal_val = omega.eigen_upperbound(cgal)

# #%%
# # cgal SDP, trace constraint + reweighted frobenius norm minimization for 1 norm minimization
# if solve_sdp:
#     # b = 80
#     b = best_val
#     L = ones([n,n])
#     A3 = A2/norm(A2)
#     b = b/norm(A2)
#     constraint_diff = b
#     beta0 = 1
#     for rew in range(4):
#         X = zeros([n,n])
#         y = 0
#         t = 0
#         while t <= 4000:
#             t += 1
#             beta = beta0 * t ** 0.5
#             eta = 2/t
            
#             # ful_matrix = K + A3 * (y + beta * constraint_diff)
#             val,vec = eigsh(L * X + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
#             # ful_val,ful_vec = eigh(sign(X)/beta + A3 * (y + beta * constraint_diff))
            
#             X = (1 - eta) * X + eta * vec.dot(vec.T)
            
#             constraint_diff = tensordot(A3,X) - b
#             print("constraint diff",constraint_diff)
            
#             gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
#             y = y + gamma * (constraint_diff)
#         L = 1/(1e-4 + abs(X)) ** 4
#         L = L/norm(L) * n
#         d,v = eigsh(X,k=3)
#         cgal = argsort(-1 * abs(v[:,-1]))[:s]
#         cgal_val = omega.eigen_upperbound(cgal)


#%%

# cgal SDP, trace constraint + Huber norm minimization
# it works
if solve_sdp and False:
    X = random.randn(n,n)
    X = X/norm(X)
    
    y = 0
    # b = 80
    b = best_val
    beta0 = 1
    t = 0
    
    A3 = A2/norm(A2)
    b = b/norm(A2)
    constraint_diff = b
    # teta0 = 1e-2
    teta0 = 2/sqrt(n)
    
    while t <= 6000:
        t += 1
        beta = beta0 * t ** 0.5
        teta = teta0/beta
        
        eta = 2/t
        
        # gradient = X + 0
        gradient = X.copy()
        
        i,j = where(X > teta)
        k,l = where(X < -teta)
        gradient[i,j] = teta
        gradient[k,l] = -teta
        
        gradient = gradient/1
        
        val,vec = eigsh(gradient + A3 * (y + beta * constraint_diff) + 10 * eye(n),k = 1,tol = 1e-2,which = "SA")
    
        X = (1 - eta) * X + eta * vec.dot(vec.T)
    
        constraint_diff = tensordot(A3,X) - b
        print("iteration",t,"constraint diff",constraint_diff)
        if t % 200 == 0:
            gc.collect()
        gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
        y = y + gamma * (constraint_diff)
    d2,v2 = eigsh(X,k=3)
    cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
    cgal_val2 = omega.eigen_upperbound(cgal2)
    sv2 = v2[:,-1].copy()
    sv2[abs(sv2)<1e-2] = 0
    fake_pattern = where(abs(v2[:,-1]) > 1e-2)[0]
    filtered = argsort(abs(v2[:,-1]))[-s:]

#%%

# cgal SDP, trace constraint + Huber norm minimization
# it works

correction = [1e-3,1e-3,1e-4,1e-4,1e-5]
# lamda_max = [0]
if solve_sdp and False:
    L = ones([n,n])
    for r in range(5):
        X = random.randn(n,n)
        X = X/norm(X)
        y = 0
        
        b = best_val
        
        beta0 = 1

        t = 1
        
        A3 = A2/norm(A2)
        b = b/norm(A2)
        constraint_diff = b
        # teta0 = 1e-2
        teta0 = 1/sqrt(n)
        
        while t <= 10000:
            t += 1
            beta = beta0 * t ** 0.5
            teta = teta0/beta
            
            eta = 2/t
            
            gradient = X.copy()
    
            i,j = where(X > teta)
            k,l = where(X < -teta)
            gradient[i,j] = teta
            gradient[k,l] = -teta
            
            val,vec = eigsh(L * gradient + A3 * (y + beta * constraint_diff) + 10 * eye(n),k = 1,tol = 1e-2,which = "SA",maxiter = t ** 0.25 * log(n),ncv = 20)
        
            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            constraint_diff = tensordot(A3,X) - b
            
            gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
            y = y + gamma * (constraint_diff)
            print("r",r,"iteration",t,"constraint diff",constraint_diff)
            if t % 200 == 0:
                gc.collect()
        d2,v2 = eigsh(X,k=3)
        cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
        cgal_val2 = omega.eigen_upperbound(cgal2)
        sv2 = v2[:,-1].copy()
        sv2[abs(sv2)<1e-2] = 0
        fake_pattern = where(abs(v2[:,-1]) > 1e-2)[0]
        L = 1/(correction[r] + abs(X))
        L = L/norm(L) * n
        filtered = argsort(abs(v2[:,-1]))[-s:]
    
#%%

def min_eig(A,max_iter):
    m,n = shape(A)
    sigma = 2 * norm(A,axis = 1,ord = 1).max()
    v = random.randn(n,1)
    v = v/norm(v)
    for i in range(max_iter):
        v = sigma * v - A.dot(v)
        v = v/norm(v)
    del m
    del n
    del sigma
    return v
# fresh version from 2019 paper
# could also add orthogonality whatever

correction = [1e-3,1e-4]
# correction = [1e-3,1e-3,1e-4,1e-4,1e-5,1e-5]
# correction = [1e-3,1e-3,1e-4,1e-4,1e-4,1e-5,1e-5,1e-5]
# correction = [1e-3] * 10

import os
import psutil
import gc
process = psutil.Process(os.getpid())
def str_mem_usage():
    mu = process.memory_info().rss / 1024**2
    return 'Memory usage: {:.2f} MB'.format(mu)
# gc.enable()
# gc.set_debug(gc.DEBUG_LEAK)

# lamda_max = [0]
if solve_sdp and True:
    L = ones([n,n])
    operator_norm = L.max()
    
    x = random.randn(n,1)
    x = x/norm(x)
    X = x.dot(x.T)
    X = A2/norm(A2)
    
    y = 0
    
    b = best_val
    
    A3 = A2/norm(A2)
    b = b/norm(A2)
    for rw in range(len(correction)):
        
        lamda0 = 1
        beta0 = 1/sqrt(n)
        # beta0 = 0.1
        
        r = tensordot(A3,X) - b

        t = 1
        change = 1
        while t <= 8000:
            # print("step",t,"change",change)
            # print('rw ', rw, " t ", t, '\t', str_mem_usage())
            # gc.collect()
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = X.copy()
    
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            # w = w/1
            
            w = L * w
            
            v = y * A3 + lamda * A3 * r
            
            val,vec = eigsh(w + v -10 * eye(n),k = 1,tol = 1e-4,which = "SA")
            # vec = min_eig(w+v,max_iter = 200)

            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b
            
            lipschitz = lamda + 1/beta * operator_norm ** 2
            
            # sigma = max(lamda,1/4*eta ** 2 * lipschitz * 4/(r ** 2))
            sigma = lamda

            y = y + sigma * r
            if t % 500 == 0:
                # print('rw ', rw, " t ", t, '\t', str_mem_usage())
                print("r",rw,"iteration",t,"constraint diff",r)
                # d2,v2 = eigsh(X,k=3)
                # print("d2\n",d2)
                gc.collect()
        d2,v2 = eigsh(X,k=3)
        cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
        cgal_val2 = omega.eigen_upperbound(cgal2)
        print(cgal_val2)
        print(d2)
        # sv2 = v2[:,-1].copy()
        # sv2[abs(sv2)<1e-2] = 0
        # fake_pattern = where(abs(v2[:,-1]) > 1e-2)[0]
        L = 1/(correction[rw] + abs(X))
        L = L/L.max()
        operator_norm = L.max()
        # filtered = argsort(abs(v2[:,-1]))[-s:]
        # print("d2",d2)

#%%

# # cgal SDP, Huber norm constraint + trace maximization
# # it works, change updates tho xd
# # constraint value should be changed although it works anyway
# if solve_sdp:
#     A_F = norm(A2) ** 2
    
#     X = zeros([n,n])
#     momentum = zeros([n,n])
#     y = 0
#     # b = 80
#     b = s
#     beta0 = 1
#     t = 0
    
#     A3 = A2/norm(A2)
#     b = b/norm(A2)
#     b = b/5
#     constraint_diff = b
#     # teta0 = 1e-2
#     teta0 = 1/sqrt(n) * 0.2
    
#     while t <= 10000:
#         t += 1
#         beta = beta0 * t ** 0.5
#         teta = teta0/beta
        
#         eta = 2/t
        
#         gradient = X.copy()
#         i,j = where(X > teta)
#         k,l = where(X < -teta)
#         g,h = where(abs(X) <= teta)
#         gradient[i,j] = teta
#         gradient[k,l] = -teta
        
#         val,vec = eigsh(-A3 + gradient * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
#         # ful_val,ful_vec = eigh(A3 + gradient * (y + beta * constraint_diff))
    
#         X = (1 - eta) * X + eta * vec.dot(vec.T)
        
#         # huber_norm = teta * (sum(abs(X[i,j])) + sum(abs(X[k,l])) - 0.5 * teta * (len(i) + len(j)) ) + 0.5 * sum(X[g,h] ** 2)
#         huber_norm = huber(teta,X).sum()
#         constraint_diff = huber_norm - b
#         print("iteration",t,"constraint diff",constraint_diff)
        
#         gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
#         y = y + gamma * (constraint_diff)
#         if t % 200 == 0:
#             gc.collect()
#     d3,v3 = eigsh(X,k=3)
#     cgal3 = argsort(-1 * abs(v3[:,-1]))[:s]
#     cgal_val3 = omega.eigen_upperbound(cgal3)
#     sv3 = v3[:,-1].copy()
#     sv3[abs(sv3)<1e-2 * 5] = 0
#     fake_pattern = where(abs(v3[:,-1]) > 1e-4)[0]

#%%

# # cgal SDP, Huber norm constraint + trace maximization
# # it works, change updates tho xd
# if solve_sdp:
#     A_F = norm(A2) ** 2
    
#     X = zeros([n,n])
#     r = random.randn(n,1)

#     r = r/norm(r)
#     X = r.dot(r.T)

#     momentum = zeros([n,n])
#     y = 0
#     beta0 = 1
#     t = 0
    
#     A3 = A2/norm(A2)
#     b = best_val/norm(A2)
#     constraint_diff = b
#     # teta0 = 1e-2
#     teta0 = 1/sqrt(n)
    
#     while t <= 40000:
#         t += 1
#         beta = beta0 * t ** 0.5
#         teta = teta0/beta
#         # b = b/norm(A2)
#         eta = 2/t
        
#         gradient = X.copy()
#         i,j = where(X > teta)
#         k,l = where(X < -teta)
#         g,h = where(abs(X) <= teta)
#         gradient[i,j] = teta
#         gradient[k,l] = -teta
        
#         val = random.randn(1)
#         vec = random.randn(n,1)
        
#         val,vec = eigsh(-A3 + gradient * y + 10 *eye(n),k = 1,tol = 1e-3,which = "SA")
    
#         X = (1 - eta) * X + eta * vec.dot(vec.T)
        
#         huber_norm = huber(teta,X).sum()
#         constraint_diff = huber_norm - b
#         print("iteration",t,"constraint diff",constraint_diff)
        
#         y = y + beta0 * (constraint_diff)
#         if t % 200 == 0:
#             gc.collect()
#     d3,v3 = eigsh(X,k=3)
#     cgal3 = argsort(-1 * abs(v3[:,-1]))[:s]
#     cgal_val3 = omega.eigen_upperbound(cgal3)
#     sv3 = v3[:,-1].copy()
#     sv3[abs(sv3)<1e-2 * 5] = 0
#     fake_pattern = where(abs(v3[:,-1]) > 1e-2 *5)[0]
#     filtered = argsort(abs(v3[:,-1]))[-s:]

#%%
# m,n = [1000,12000]

# A = random.randn(m,n)
# A2 = A.T.dot(A) + 10 * eye(n)
# # A2s = csr_matrix(A2)

# t0 = time.process_time()
# eigsh(A2,k=1,tol = 1e-4,which = "SA")
# t1 = time.process_time()
# eigsh(A2,k=1,tol = 1e-4,which = "LA")
# t2 = time.process_time()
# print("1",t1-t0)
# print("2",t2-t1)

#%%

# correction = [1e-3,1e-3,1e-4,1e-4,1e-5]
# # lamda_max = [0]
# if solve_sdp and True:
#     L = ones([n,n])
#     for r in range(5):
#         X = random.randn(n,n)
#         X = X/norm(X)
#         y = 0
        
#         b = best_val
        
#         eta0 = 2
#         lamda0 = 2
#         beta0 = 1/sqrt(n)

#         t = 1
        
#         A3 = A2/norm(A2)
#         b = b/norm(A2)
#         constraint_diff = b
#         # teta0 = 1e-2
#         teta0 = 1/sqrt(n)
        
#         while t <= 10000:
#             t += 1
#             eta = eta0 * t ** 0.5
#             lamda = lamda0 * t ** 0.5
#             beta = beta0 / t ** 0.5
            
            
#             gradient = X.copy()
    
#             i,j = where(X > teta)
#             k,l = where(X < -teta)
#             gradient[i,j] = teta
#             gradient[k,l] = -teta
            
#             val,vec = eigsh(L * gradient + A3 * (y + beta * constraint_diff) + 10 * eye(n),k = 1,tol = 1e-2,which = "SA",maxiter = t ** 0.25 * log(n),ncv = 20)
        
#             X = (1 - eta) * X + eta * vec.dot(vec.T)
        
#             constraint_diff = tensordot(A3,X) - b
            
#             gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
#             y = y + gamma * (constraint_diff)
#             print("r",r,"iteration",t,"constraint diff",constraint_diff)
#             if t % 200 == 0:
#                 gc.collect()
#         d2,v2 = eigsh(X,k=3)
#         cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
#         cgal_val2 = omega.eigen_upperbound(cgal2)
#         sv2 = v2[:,-1].copy()
#         sv2[abs(sv2)<1e-2] = 0
#         fake_pattern = where(abs(v2[:,-1]) > 1e-2)[0]
#         L = 1/(correction[r] + abs(X))
#         L = L/norm(L) * n
#         filtered = argsort(abs(v2[:,-1]))[-s:]
        