from PCA_SPAR import *
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal
from numpy.linalg import matrix_rank
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky
from scipy.sparse.linalg import eigsh
from static_questions import * 
from matplotlib.pyplot import *
import cvxpy

#%%

# A = random.normal(4,10,[200,1000])

A = at_t_faces

m,n = shape(A)

mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA

A2 = A.T.dot(A)


#%%

s = 20

omega = SPCA(A,s)
# omega.A2 = pitprops
# omega.squared_A2 = pitprops ** 2
# omega.abs_A2 = abs(pitprops)
# omega.abs_A2s = abs(pitprops) - eye(13)

omega.search_multiplier = 20

#%%

t0 = time.process_time()
omega.column_norm_1_fast()
t1 = time.process_time()

tv0 = time.process_time()
# omega.column_norm_1l()
tv1 = time.process_time()

print("gerschgorin done ")

#%%

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
omega.frobenius_cw_fast()
t9 = time.process_time()

print("frobenius done ")

t10 = time.process_time()
omega.correlation_cw_fast()
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

