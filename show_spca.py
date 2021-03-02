from PCA_SPAR import *
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal
from numpy.linalg import matrix_rank
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky
from scipy.sparse.linalg import eigsh
from static_questions import * 
from matplotlib.pyplot import *
import cvxpy


#%%

A = random.normal(4,10,[200,50])

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

# A2 -= val * vec0.dot(vec0.T)

#%%

s = 10

omega = SPCA(A,s)


omega.search_multiplier = 4

#%%

t0 = time.process_time()
omega.column_norm_1()
t1 = time.process_time()

tv0 = time.process_time()
# omega.column_norm_1l()
tv1 = time.process_time()

print("gerschgorin done ")


t2 = time.process_time()
# pcw_index2,pcw_val2 = omega.PCW1_iterative(list(range(n)))
t3 = time.process_time()

print("pcw done ")

t4 = time.process_time()
# pcw_index,pcw_val = omega.PCW1(list(range(n)))
t5 = time.process_time()



t6 = time.process_time()
# gcw_index,gcw_val = omega.GCW1(list(range(n)))
t7 = time.process_time()



t14 = time.process_time()
# gcw_index2,gcw_val2 = omega.GCW1_iterative(list(range(n)))
t15 = time.process_time()

print("gcw done ")

t8 = time.process_time()
omega.frobenius_cw()
t9 = time.process_time()

print("frobenius done ")

t10 = time.process_time()
omega.correlation_cw()
t11 = time.process_time()

print("correlation done ")

t12 = time.process_time()
# omega.solve_spca(list(range(n)))
t13 = time.process_time()

t16 = time.process_time()
em_index,em_val = omega.EM()
t17 = time.process_time()

print("em done ")

t18 = time.process_time()
chol2,chol_val2 = omega.cholesky_mk2()
t19 = time.process_time()

print("chol2 done ")

t20 = time.process_time()
# chol,chol_val = omega.cholesky()
t21 = time.process_time()

print("chol done ")

t22 = time.process_time()
# spi,spi_val = omega.SPI()
t23 = time.process_time()

print("spi done ")

print("Gerschgorin  ",t1-t0)
print("correlation  ",t11-t10)
print("Frobenius    ",t9-t8)
print("cholesky     ",t21-t20)
print("cholesky  2  ",t19-t18)
print("partial      ",t3-t2)
print("greedy       ",t15-t14)
print("expect       ",t17-t16)
print("thresholding ",t23-t22)

nope = False

# nope = True
# bulduk = sorted(omega.eigenindices[0])
# val_bulduk = omega.eigenvalues[0]

if nope:
    print("optimal      ",val_bulduk)
print("Gerschgorin  ",omega.Rval)
print("correlation  ",omega.R2val)
print("Frobenius    ",omega.R4val)
# print("cholesky     ",chol_val)
print("cholesky  2  ",chol_val2)
print("expect       ",em_val)
# print("partial      ",pcw_val2)
# print("greedy       ",gcw_val2)
# print("thresholding ",spi_val)

best_val = max(em_val,omega.Rval,omega.R2val,omega.R4val,chol_val2)

#%%

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

if n <= 25:
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X2 >> 0]
    constraints2 += [cvxpy.trace(X2) == 1]
    constraints2 += [cvxpy.trace(A2 @ X2) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.pnorm(X2,1))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.MOSEK)
    
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

if n <= 25:
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X2 >> 0]
    constraints2 += [cvxpy.trace(X2) == 1]
    constraints2 += [cvxpy.trace(A2 @ X2) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.pnorm(X2,2))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.MOSEK)
    
    first_copy = X2.value
    Y0_2 = X2.value
    Y = Y0.copy()
    Y[abs(Y)<1e-3] = 0
    
    lamda1,vector1 = eigsh(Y0,k = 1,tol = 1e-4)
    svector1 = vector1.copy()
    svector1[abs(svector1) < 1e-3] = 0
    
    pattern1 = argsort(abs(vector1[:,0]))[-s:]
    sparsity1 = where(abs(svector1) > 0)[0]

#%%

if n <= 25:
    X2 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints2 = []
    constraints2 += [X2 >> 0]
    constraints2 += [cvxpy.trace(X2) == 1]
    constraints2 += [cvxpy.trace(A2 @ X2) >= best_val]
    
    
    obj2 = cvxpy.Minimize(cvxpy.norm(X2,"fro"))
    # obj2 = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(X2)))
    
    prob2 = cvxpy.Problem(obj2,constraints2)
    prob2.solve(solver = cvxpy.MOSEK)
    
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
while t <= 1:
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
while t <= 1:
    t += 1
    beta = beta0 * t ** 0.5
    eta = 2/t
    old_sign = sign(X)
    # ful_matrix = K + A3 * (y + beta * constraint_diff)
    val,vec = eigsh(old_sign / beta + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
    # val,vec = eigsh(A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
    # ful_val,ful_vec = eigh(sign(X)/beta + A3 * (y + beta * constraint_diff))
    
    X = (1 - eta) * X + eta * vec.dot(vec.T)
    new_sign = sign(X)
    consistency = (old_sign + new_sign)/2
    X = X * consistency
    constraint_diff = tensordot(A3,X) - b
    print("constraint diff",constraint_diff)
    
    gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
    y = y + gamma * (constraint_diff)
d,v = eigsh(X,k=3)
cgal = argsort(-1 * abs(v[:,-1]))[:s]
cgal_val = omega.eigen_upperbound(cgal)

#%%

if n <= 25:
    X4 = cvxpy.Variable(shape = (n,n),symmetric = True)
    constraints4 = []
    constraints4 += [X4 >> 0]
    constraints4 += [cvxpy.trace(X4) == 1]
    constraints4 += [cvxpy.trace(A3 @ X4) == b]
    
    obj4 = cvxpy.Minimize(cvxpy.pnorm(X4,1))
    # obj4 = cvxpy.Minimize(cvxpy.trace(K @ X4))
    
    prob4 = cvxpy.Problem(obj4,constraints4)
    prob4.solve(solver = cvxpy.MOSEK)
    
    first_copy = X4.value
    L0 = X4.value
    L = L0.copy()
    L[abs(L)<1e-3] = 0
    
    lamda1,vector1 = eigsh(Y0,k = 1,tol = 1e-4)
    svector1 = vector1.copy()
    svector1[abs(svector1) < 1e-3] = 0
    
    pattern1 = argsort(abs(vector1[:,0]))[-s:]
    sparsity1 = where(abs(svector1) > 0)[0]

#%%

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
teta0 = 1/sqrt(n) * 0.1

while t <= 1:
    t += 1
    beta = beta0 * t ** 0.5
    teta = teta0/beta
    eta = 2/t
    
    
    gradient = X.copy()
    old_sign = sign(X)
    i,j = where(X > teta)
    k,l = where(X < -teta)
    gradient[i,j] = teta
    gradient[k,l] = -teta
    
    val,vec = eigsh(gradient + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")

    X = (1 - eta) * X + eta * vec.dot(vec.T)

    new_sign = sign(X)
    consistency = sign(old_sign + new_sign)
    # X = X * consistency
    constraint_diff = tensordot(A3,X) - b
    print("constraint diff",constraint_diff)
    
    gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
    y = y + gamma * (constraint_diff)
d2,v2 = eigsh(X,k=3)
cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
cgal_val2 = omega.eigen_upperbound(cgal2)
sv2 = v2[:,-1].copy()
sv2[abs(sv2)<1e-2 * 5] = 0
fake_pattern = where(abs(v2[:,-1]) > 1e-2 *5)[0]

#%%

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
# teta0 = 1e-2
teta0 = 1/n * 0.1

while t <= 1:
    t += 1
    beta = beta0 * t ** 0.5
    teta = teta0/beta
    eta = 2/t
    
    gradient = X.copy()
    # old_sign = sign(X)
    i,j = where(X > teta)
    k,l = where(X < -teta)
    gradient[i,j] = teta
    gradient[k,l] = -teta
    
    val,vec = eigsh(gradient + A3 * beta * constraint_diff,k = 1,tol = 1e-2,which = "SA")
    
    X = (1 - eta) * X + eta * vec.dot(vec.T)
    # new_sign = sign(X)
    # consistency = sign(old_sign + new_sign)
    # X = X * consistency
    constraint_diff = tensordot(A3,X) - b
    print("constraint diff",constraint_diff)
    
d2,v2 = eigsh(X,k=3)
cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
cgal_val2 = omega.eigen_upperbound(cgal2)


#%%

# A_F = norm(A2) ** 2

# X = zeros([n,n])
# y = 0
# # b = 80
# b = best_val
# beta0 = 1
# t = 0

# A3 = A2/norm(A2)
# b = b/norm(A2)
# constraint_diff = b

# while t <= 40:
#     t += 1
#     beta = beta0 * t ** 0.5
#     eta = 2/t
    
#     # ful_matrix = K + A3 * (y + beta * constraint_diff)
#     val,vec = eigsh(sign(X)/beta * eta + A3 * (y + beta * constraint_diff),k = 1,tol = 1e-2,which = "SA")
#     # ful_val,ful_vec = eigh(sign(X)/beta + A3 * (y + beta * constraint_diff))
    
#     X = (1 - eta) * X + eta * vec.dot(vec.T)
    
#     constraint_diff = tensordot(A3,X) - b
#     print("constraint diff",constraint_diff)
    
#     gamma = min(beta0,beta*eta ** 2/(constraint_diff ** 2))
#     y = y + gamma * (constraint_diff)
# d,v = eigsh(X,k=3)
# cgal = argsort(-1 * abs(v[:,-1]))[:s]
# cgal_val = omega.eigen_upperbound(cgal)