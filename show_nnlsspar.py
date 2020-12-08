from PCA_SPAR import *
from numpy import random,reshape,mean,ones,array
from scipy.linalg import eigvalsh,ldl,det
from static_questions import * 
from matplotlib.pyplot import *

#%%

A = random.normal(4,10,[400,13])

# A = ozone_X2

m,n = shape(A)

mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA 

A2 = A.T.dot(A)

q,r,p = qr(A,pivoting = True)


#%%

omega = SPCA(A,6)
omega.A2 = pitprops
omega.AA2 = abs(pitprops)
omega.AA2s = abs(pitprops) - eye(13)


t0 = time.process_time()
# omega.solve_spca_mk6(list(range(n)))
t1 = time.process_time()

print("sixth done ")

t2 = time.process_time()
# omega.solve_spca_mk2(list(range(n)))
t3 = time.process_time()

print("second done ")

t4 = time.process_time()
# gcw_value,gcw_index = omega.GCW1(list(range(n)))
t5 = time.process_time()

print("third done ")

t6 = time.process_time()
# omega.solve_spca_mk6(list(range(n)))
t7 = time.process_time()

print("fourth done ")

t8 = time.process_time()
gwa = omega.FH_mk2(list(range(n)))
t9 = time.process_time()

print("fifth done ")


print("mk6 {:1.4f} mk2 {:1.4f} mk3 {:1.4f} mk4 {:1.4f}  mk5 {:1.4f}".format(t1-t0,t3-t2,t5-t4,t7-t6,t9-t8))
# print("eigenvalue",omega.eigenvalues[0],"eigenindex",omega.eigenindices[0],"frobenius",norm(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:]))
# print(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:])
# print(sum(abs(A2[:,omega.eigenindices[0]][omega.eigenindices[0],:])))
# print("sadge",sorted(omega.eigenindices[0]) in omega.listR)
# print("sadge2",sorted(omega.eigenindices[0]) in omega.R2)

#%%


# s1 = [11,14]

# s3 = [11, 14,  8]

# s4 = [11, 14, 32, 41]
# maxheur = 0
# for i in range(n):
#     if i in s3:
#         continue
#     st = s3 + [i]
#     print("i",i)
#     print(sum(A2[:,st][st,:]))
#     maxheur = max(maxheur,sum(A2[:,st][st,:]))
# print("----------------------")
# print("champion",sum(A2[:,s4][s4,:]))

#%%

# fakeA = copy(A)
# fakeA[:,11] *= -1
# fakeA2 = fakeA.T.dot(fakeA)


