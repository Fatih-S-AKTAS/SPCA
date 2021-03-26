import gc

sadge = gc.collect()

#%%
from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,eye,argsort
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm,svd
from scipy.sparse.linalg import eigsh,svds
from scipy.special import huber
import time
from static_questions import pitprops
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure
from datagenerator import varimax_paper_data,review_paper_data,graph_data

#%%

A = random.normal(4,10,[100,2000])
# A = varimax_paper_data(1000,400)
# A = review_paper_data(400,4000)
# A = graph_data(2000,4)

# l,d,p = ldl(pitprops)
# A = l

# A = at_t_faces
# A = load("at_t_faces_save.npy")

m,n = shape(A)
    
mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)
# sA = std(A,axis = 0) * (1 + 0.01 * random.randn(n))
A = A/sA

#%%

sadge1 = gc.collect()

s = 50

omega = SPCA(A,s)


#%%

omega.search_multiplier = min(400/s,n/s)

k = 1

solve_sdp = False

# gd,gd_val = omega.column_norm_1()
# gd_val2,gd_vector = omega.eigen_pair(gd)

t0 = time.process_time()
pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
# gd_set,gd_val = omega.column_norm_1()
t1 = time.process_time()
omega.restart()
print("gerschgorin done ")

sadge1 = gc.collect()

t2 = time.process_time()
pattern2,eigens2,load2,component2,variance2 = omega.find_component("CCW",k)
# ccw_set,ccw_val = omega.correlation_cw()
t3 = time.process_time()
omega.restart()
print("CCW done ")

sadge1 = gc.collect()

t4 = time.process_time()
pattern3,eigens3,load3,component3,variance3 = omega.find_component("FCW",k)
# fcw_set,fcw_val = omega.frobenius_cw()
t5 = time.process_time()
omega.restart()
print("FCW done ")

sadge1 = gc.collect()

t6 = time.process_time()
pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
# em_set,em_val = omega.EM()
t7 = time.process_time()
omega.restart()
print("EM done ")

sadge1 = gc.collect()

t8 = time.process_time()
pattern5,eigens5,load5,component5,variance5 = omega.find_component("Path",k)
# path_set,path_val = omega.cholesky_mk2()
t9 = time.process_time()
omega.restart()
print("Path/Chol done ")

sadge1 = gc.collect()

t14 = time.process_time()
# pattern8,eigens8,load8,component8,variance8 = omega.find_component("Path_mk2",k)
t15 = time.process_time()
# omega.restart()
print("Path/Chol_mk2 done ")

# sadge1 = gc.collect()

t10 = time.process_time()
# pattern6,eigens6,load6,component6,variance6 = omega.find_component("PCW",k)
# pcw_possible,pcw_set,pcw_val = omega.PCW1()
t11 = time.process_time()
# omega.restart()
print("PCW done ")

# sadge1 = gc.collect()

t12 = time.process_time()
pattern7,eigens7,load7,component7,variance7 = omega.find_component("nesterov",k)
# nesterov_set,nesterov_val = omega.nesterov()
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
print("cholesky mk2 ",t15-t14)
print("PCW          ",t11-t10)
print("nesterov     ",t13-t12)
print("----------------------------")
print("gerschgorin  ",sum(variance1))
print("correlation  ",sum(variance2))
print("frobenius    ",sum(variance3))
print("em           ",sum(variance4))
print("cholesky     ",sum(variance5))
# print("cholesky mk2 ",sum(variance8))
# print("PCW          ",sum(variance6))
print("nesterov     ",sum(variance7))

#%%

def SPCArt(A,s):
    u,d,vt = svds(A,k=1)
    n,p = shape(A)
    R = eye(1)
    X = zeros([1,p])
    t = 0
    change = 1
    while change > 1e-3 and t < 400:
        t += 1
        Z = vt.T.dot(R.T)
        argsorted = argsort(-1 * abs(Z[:,0]))
        X_old = X
        X = zeros([1,p])
        X[0,argsorted[:s]] = Z[argsorted[:s],0]
        u1,d1,vt1 = svd(X.dot(vt.T))
        R = u1.dot(vt1)
        change = norm(X_old - X)
        print("t",t,"change",change)
    return X/norm(X)
    
X = SPCArt(A,s)

print("SPCArt",X.dot(A.T).dot(A.dot(X.T))[0][0])