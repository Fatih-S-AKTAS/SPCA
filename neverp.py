import cvxpy as cp
from numpy import *
from scipy.linalg import *
from numpy.linalg import matrix_rank

#%%

eye_6 = eye(6)
L = []
for i in range(4):
    M = zeros([6,6])
    M[i+1,i+2] = 1
    M[i+2,i+1] = 1
    L.append(M)

M = zeros([6,6])
M[1,5] = 1
M[5,1] = 1
L.append(M)

K = []
for i in range(5):
    M = zeros([6,6])
    M[i+1,i+1] = 1
    M[0,i+1] = -0.5
    M[i+1,0] = -0.5
    K.append(M)
    

X = cp.Variable((6,6),symmetric = True)
constraints = [X >> 0]
for i in range(5):
    constraints += [ cp.trace(L[i] @ X) == 0]

N = zeros([6,6])
N[0,0] = 1
constraints += [ cp.trace(N @ X) == 1]

for i in range(5):
    constraints += [ cp.trace(K[i] @ X) == 0]


C = zeros([6,6])
C[0,1:] = 1
C[1:,0] = 1

obj = cp.Maximize(cp.trace(C @ X))
prob = cp.Problem(obj,constraints)
prob.solve(verbose = True)
print("X")
print(X.value)
xval = X.value[1:,1:]

#%%

eye_6 = eye(6)
L = []
for i in range(4):
    M = zeros([6,6])
    M[i+1,i+2] = 0.5
    M[i+2,i+1] = 0.5
    M[0,i+1] = -0.5
    M[i+1,0] = -0.5
    M[0,i+2] = -0.5
    M[i+2,0] = -0.5
    L.append(M)

M = zeros([6,6])
M[1,5] = 0.5
M[5,1] = 0.5
M[0,5] = -0.5
M[5,0] = -0.5
M[0,1] = -0.5
M[1,0] = -0.5
L.append(M)

K = []
for i in range(6):
    M = zeros([6,6])
    M[i,i] = 1
    K.append(M)
    

X = cp.Variable((6,6),symmetric = True)
constraints = [X >> 0]
for i in range(5):
    constraints += [ cp.trace(L[i] @ X) == -1]


for i in range(6):
    constraints += [ cp.trace(K[i] @ X) == 1]


C = zeros([6,6])
C[0,1:] = 1
C[1:,0] = 1
C /= 2

obj = cp.Minimize(cp.trace(C @ X))
prob = cp.Problem(obj,constraints)
prob.solve(verbose = True)
print("X")
print(X.value)
xval = X.value[1:,1:]

#%%

eye_5 = eye(5)
L = []
for i in range(4):
    M = zeros([5,5])
    M[i,i+1] = 1
    M[i+1,i] = 1
    L.append(M)
    
M = zeros([5,5])
M[0,4] = 1
M[4,0] = 1
L.append(M)

X = cp.Variable((5,5),symmetric = True)
constraints = [X >> 0]
for i in range(5):
    constraints += [ cp.trace(L[i] @ X) == 0]
constraints += [cp.trace(eye_5 @ X) == 1]

J = ones([5,5])
obj = cp.Maximize(cp.trace(J @ X))
prob = cp.Problem(obj,constraints)
prob.solve(verbose = True)
print("X")
print(X.value)
    
#%%

eye_5 = eye(5)
L = []
for i in range(4):
    M = zeros([5,5])
    M[i,i+1] = 1
    M[i+1,i] = 1
    L.append(M)
    
M = zeros([5,5])
M[0,4] = 1
M[4,0] = 1
L.append(M)

X = cp.Variable((5,5),symmetric = True)
constraints = [X >> 0]
for i in range(5):
    constraints += [ cp.trace(L[i] @ X) == 0]
constraints += [cp.trace(eye_5 @ X) == 1]

J = ones([5,5])
obj = cp.Maximize(cp.trace(J @ X))
prob = cp.Problem(obj,constraints)
prob.solve(verbose = True)
print("X")
print(X.value)
    

