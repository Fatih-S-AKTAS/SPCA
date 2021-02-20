from PCA_SPAR import *
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal
from numpy.linalg import matrix_rank
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky
from scipy.sparse.linalg import eigsh
from static_questions import * 
from matplotlib.pyplot import *
import cvxpy

#%%

# A = random.normal(4,10,[10,20])

A = ozone_X2

m,n = shape(A)

mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA

A2 = A.T.dot(A)
vals = eigvalsh(A2)


#%%

s = 5

omega = SPCA(A,s)
# omega.A2 = pitprops
# omega.squared_A2 = pitprops ** 2
# omega.abs_A2 = abs(pitprops)
# omega.abs_A2s = abs(pitprops) - eye(13)
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
pcw_index2,pcw_val2 = omega.PCW1_iterative(list(range(n)))
t3 = time.process_time()

print("second done ")

t4 = time.process_time()
# pcw_index,pcw_val = omega.PCW1(list(range(n)))
t5 = time.process_time()

print("third done ")

t6 = time.process_time()
# gcw_index,gcw_val = omega.GCW1(list(range(n)))
t7 = time.process_time()

print("fourth done ")

t14 = time.process_time()
gcw_index2,gcw_val2 = omega.GCW1_iterative(list(range(n)))
t15 = time.process_time()

t8 = time.process_time()
omega.frobenius_cw()
t9 = time.process_time()

print("fifth done ")

t10 = time.process_time()
omega.correlation_cw()
t11 = time.process_time()


t12 = time.process_time()
# omega.solve_spca_mk6(list(range(n)))
t13 = time.process_time()

t16 = time.process_time()
chol = omega.cholesky()
t17 = time.process_time()

t18 = time.process_time()
chol2 = omega.cholesky_mk2()
t19 = time.process_time()

print("first {:1.4f} second {:1.4f} third {:1.4f} fourth {:1.4f}  fifth {:1.4f}".format(t1-t0,t3-t2,t5-t4,t7-t6,t9-t8))
# print("eigenvalue",omega.eigenvalues[0],"eigenindex",omega.eigenindices[0],"frobenius",norm(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:]))
# print(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:])
# print(sum(abs(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:])))

nope = False

# nope = True
# bulduk = sorted(omega.eigenindices[0])
# val_bulduk = omega.eigenvalues[0]

if nope:
    print("optimal      ",val_bulduk)
print("Gerschgorin  ",omega.Rval)
print("correlation  ",omega.R2val)
print("Column norm  ",omega.R3val)
print("Frobenius    ",omega.R4val)
print("cholesky     ",omega.eigen_upperbound(chol))
print("partial      ",pcw_val2)
print("greedy       ",gcw_val2)
