import gc

sadge = gc.collect()

#%%
from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,argsort,sqrt
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm
from scipy.sparse.linalg import eigsh
from scipy.special import huber
import time
from static_questions import pitprops
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure
from datagenerator import varimax_paper_data,review_paper_data,graph_data
import cvxpy

#%%

A = random.normal(4,10,[200,2000])
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

#%%

sadge1 = gc.collect()

s = 20

omega = SPCA(A,s)


#%%

omega.search_multiplier = min(200/s,n/s)

k = 1

solve_sdp = True

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
# cgal SDP, trace constraint + frobenius norm minimization
if solve_sdp:
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
if solve_sdp:
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
    