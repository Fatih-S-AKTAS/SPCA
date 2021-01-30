from PCA_SPAR import *
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal
from numpy.linalg import matrix_rank
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv
from scipy.sparse.linalg import eigsh
from static_questions import * 
from matplotlib.pyplot import *
import cvxpy

#%%

A = random.normal(4,10,[100,13])

# A = ozone_X2

m,n = shape(A)

mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA 

A2 = A.T.dot(A)
vals = eigvalsh(A2)
A2con = A2/vals[0]
A2_con_inv = inv(A2con)
q,r,p = qr(A,pivoting = True)
 
sign_vector = sign(random.randn(50,1))
fake_matrix = random.uniform(0,1,[50,50])

matrix = sign_vector.dot(sign_vector.T) * fake_matrix
matrix = matrix + matrix.T
matrix = matrix + eye(50)

#%%

s = 4

omega = SPCA(A,s)
omega.A2 = pitprops
omega.squared_A2 = pitprops ** 2
omega.abs_A2 = abs(pitprops)
omega.abs_A2s = abs(pitprops) - eye(13)
# vals = eigvalsh(pitprops)
# A2con = pitprops/vals[0]
A2_con_inv = inv(pitprops)

# omega.A2 = matrix
# omega.squared_A2 = matrix ** 2
# omega.abs_A2 = abs(matrix)
# omega.abs_A2s = abs(matrix) - eye(50)

t0 = time.process_time()
omega.column_norm_1()
t1 = time.process_time()


omega.column_norm_2()

print("first done ")

t2 = time.process_time()
# pcw_index2,pcw_val2 = omega.PCW1_iterative(list(range(n)))
t3 = time.process_time()

print("second done ")

t4 = time.process_time()
# pcw_index,pcw_val = omega.PCW1(list(range(n)))
t5 = time.process_time()

print("third done ")

t6 = time.process_time()
# gcw_index,gcw_value = omega.GCW1(list(range(n)))
t7 = time.process_time()

print("fourth done ")


t8 = time.process_time()
omega.frobenius_cw()
t9 = time.process_time()

print("fifth done ")

t10 = time.process_time()
omega.correlation_cw()
t11 = time.process_time()


t12 = time.process_time()
omega.solve_spca_mk6(list(range(n)))
t13 = time.process_time()

t14 = time.process_time()
x_final_row = omega.convex_lasso(lamda0 = 2)
t15 = time.process_time()

print("mk6 {:1.4f} mk2 {:1.4f} mk3 {:1.4f} mk4 {:1.4f}  mk5 {:1.4f}".format(t1-t0,t3-t2,t5-t4,t7-t6,t9-t8))
# print("eigenvalue",omega.eigenvalues[0],"eigenindex",omega.eigenindices[0],"frobenius",norm(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:]))
# print(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:])
# print(sum(abs(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:])))
# print("sadge",sorted(omega.eigenindices[0]) in omega.listR)
# print("sadge2",sorted(omega.eigenindices[0]) in omega.R2)
x_final = reshape(x_final_row,[13,1])
x_final[abs(x_final) < 1e-4] = 0
print(x_final)

# bulduk = sorted(omega.eigenindices[0])
# val_bulduk = omega.eigenvalues[0]

# print("optimal",val_bulduk)
# print("heuristic",pcw_val)
print("mk1",omega.Rval)
print("mk2",omega.R2val)
print("mk3",omega.R3val)
print("mk4",omega.R4val)
# print("mk5",omega.R5val)
# print("mk6",omega.R6val)
# val,vec = eigh(A2)

# l,d,p = ldl(A2)
# ut_l = l[p,:].T + l[p,:] - eye(n)

#%%
alpha = 1.6
# forsenCD 
X = cvxpy.Variable((13,13),symmetric = True)
constraints = [X >> 0]
constraints += [cvxpy.sum(cvxpy.abs(X)) <= alpha]
# constraints += [cvxpy.norm(X,1) <= alpha]

obj = cvxpy.Maximize(cvxpy.log_det(X) - cvxpy.trace(inv(pitprops) @ X))

prob = cvxpy.Problem(obj,constraints)

prob.solve(solver = cvxpy.MOSEK)
construct_X = X.value
construct_X[abs(construct_X) <= 1e-5] = 0
# print("eigs",eigvalsh(X.value))
# print("X")
# print(X.value)
fake_X = X.value.copy()
fill_diagonal(fake_X,0)
xc,yc = where(fake_X >0)
# print(unique(xc))

#%%

# alpha = 4.1
# # forsenCD 
# X = cvxpy.Variable((4,4),symmetric = True)
# constraints = [X >> 0]
# constraints += [cvxpy.sum(cvxpy.abs(X)) <= alpha]
# # constraints += [cvxpy.norm(X,1) <= alpha]

# obj = cvxpy.Maximize(cvxpy.log_det(X) - cvxpy.trace(invT @ X))

# prob = cvxpy.Problem(obj,constraints)

# prob.solve(solver = cvxpy.MOSEK)
# construct_X = X.value
# construct_X[abs(construct_X) <= 1e-4] = 0
# print("eigs",eigvalsh(X.value))
# print("X")
# print(X.value)
