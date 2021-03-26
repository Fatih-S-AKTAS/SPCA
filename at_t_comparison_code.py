import gc

sadge = gc.collect()

#%%
from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm
from scipy.sparse.linalg import eigsh
from scipy.special import huber
import time
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure
from datagenerator import varimax_paper_data,review_paper_data,graph_data

#%%

# A = load("at_t_faces_save.npy")
A = random.randn(50,400)

m,n = shape(A)

mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA

#%%

sadge1 = gc.collect()

s = 20

omega = SPCA(A,s)


#%%
# omega.search_multiplier = min(400/s,n/s)

# k = 1

up_to = 4

s_var = zeros([up_to,7])
s_cpu = zeros([up_to,7])

sparsity = 5
omega.s = sparsity
omega.args["sparsity"] = sparsity
print("!!! Current Sparsity level",sparsity,"!!!")


t0 = time.process_time()
gd_set,gd_val = omega.column_norm_1()
t1 = time.process_time()
print("gerschgorin done ")

sadge1 = gc.collect()

t2 = time.process_time()
ccw_set,ccw_val = omega.correlation_cw()
t3 = time.process_time()
print("CCW done ")

sadge1 = gc.collect()

t4 = time.process_time()
fcw_set,fcw_val = omega.frobenius_cw()
t5 = time.process_time()
print("FCW done ")

sadge1 = gc.collect()

t6 = time.process_time()
em_set,em_val = omega.EM()
t7 = time.process_time()
print("EM done ")

t8 = time.process_time()
path_set,path_val = omega.cholesky_mk2()
t9 = time.process_time()
print("Path/Chol done ")

sadge1 = gc.collect()

t10 = time.process_time()
gp_set,gp_val = omega.nesterov()
t11 = time.process_time()

print("Nesterov done ")

t12 = time.process_time()
P_pcw,pcw_set,pcw_val = omega.PCW1()
t13 = time.process_time()

print("PCW step 1 done ")

sadge1 = gc.collect()

print("----------------------------")
print("gerschgorin  ",t1-t0)
print("correlation  ",t3-t2)
print("frobenius    ",t5-t4)
print("cholesky     ",t7-t6)
print("PCW          ",t13-t12)
print("----------------------------")
print("gerschgorin  ",gd_val)
print("correlation  ",ccw_val)
print("frobenius    ",fcw_val)
print("cholesky     ",path_val)
print("PCW          ",pcw_val)

s_cpu[0,0] = t1-t0
s_cpu[0,1] = t3-t2
s_cpu[0,2] = t5-t4
s_cpu[0,3] = t7-t6
s_cpu[0,4] = t9-t8
s_cpu[0,5] = t11-t10
s_cpu[0,6] = t13-t12

s_var[0,0] = gd_val
s_var[0,1] = ccw_val
s_var[0,2] = fcw_val
s_var[0,3] = em_val
s_var[0,4] = path_val
s_var[0,5] = gp_val
s_var[0,6] = pcw_val

for iteration in range(1,up_to):
    sparsity = (iteration+1) * 5
    omega.s = sparsity
    omega.args["sparsity"] = sparsity
    print("!!! Current Sparsity level",sparsity,"!!!")
    omega.search_multiplier = min(400/sparsity,n/sparsity)
    
    t0 = time.process_time()
    gd_set,gd_val = omega.column_norm_1()
    t1 = time.process_time()
    print("gerschgorin done ")
    
    sadge1 = gc.collect()
    
    t2 = time.process_time()
    ccw_set,ccw_val = omega.correlation_cw()
    t3 = time.process_time()
    print("CCW done ")
    
    sadge1 = gc.collect()
    
    t4 = time.process_time()
    fcw_set,fcw_val = omega.frobenius_cw()
    t5 = time.process_time()
    print("FCW done ")
    
    sadge1 = gc.collect()
    
    t6 = time.process_time()
    em_set,em_val = omega.EM()
    t7 = time.process_time()
    print("EM done ")
    
    t8 = time.process_time()
    path_set,path_val = omega.cholesky_mk2()
    t9 = time.process_time()
    print("Path/Chol done ")
    
    sadge1 = gc.collect()
    
    t10 = time.process_time()
    gp_set,gp_val = omega.nesterov()
    t11 = time.process_time()
    
    print("Nesterov done ")
    
    t12 = time.process_time()
    P_pcw,pcw_set,pcw_val = omega.PCW1_efficient(P_pcw,pcw_set,pcw_val)
    t13 = time.process_time()
    
    print("PCW step 1 done ")
    
    sadge1 = gc.collect()
    
    print("----------------------------")
    print("gerschgorin  ",t1-t0)
    print("correlation  ",t3-t2)
    print("frobenius    ",t5-t4)
    print("Path     ",t7-t6)
    print("PCW          ",t13-t12)
    print("----------------------------")
    print("gerschgorin  ",gd_val)
    print("correlation  ",ccw_val)
    print("frobenius    ",fcw_val)
    print("Path     ",path_val)
    print("PCW          ",pcw_val)
    
    s_cpu[iteration,0] = t1-t0
    s_cpu[iteration,1] = t3-t2
    s_cpu[iteration,2] = t5-t4
    s_cpu[iteration,3] = t7-t6
    s_cpu[iteration,4] = t9-t8
    s_cpu[iteration,5] = t11-t10
    s_cpu[iteration,6] = t13-t12 + s_cpu[iteration-1,6]
    
    s_var[iteration,0] = gd_val
    s_var[iteration,1] = ccw_val
    s_var[iteration,2] = fcw_val
    s_var[iteration,3] = em_val
    s_var[iteration,4] = path_val
    s_var[iteration,5] = gp_val
    s_var[iteration,6] = pcw_val

#%%
sparsity = arange(5,5 * (up_to + 1),5)

varf = figure()
grid(True)
plot(sparsity,s_var[:,0],color = "blue")
plot(sparsity,s_var[:,1],color = "red")
plot(sparsity,s_var[:,2],color = "green")
plot(sparsity,s_var[:,3],color = "olive")
plot(sparsity,s_var[:,4],color = "orange")
plot(sparsity,s_var[:,4],color = "purple")
plot(sparsity,s_var[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("Variance")
title("Variance Against Sparisty Level Pattern Generation")

cpuf = figure()
grid(True)
plot(sparsity,s_cpu[:,0],color = "blue")
plot(sparsity,s_cpu[:,1],color = "red")
plot(sparsity,s_cpu[:,2],color = "green")
plot(sparsity,s_cpu[:,3],color = "olive")
plot(sparsity,s_cpu[:,4],color = "orange")
plot(sparsity,s_cpu[:,5],color = "purple")
plot(sparsity,s_cpu[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("CPU time (s)")
title("CPU time Against Sparisty Level Pattern Generation")

varf.savefig("at_t_variance.eps",format = "eps")
cpuf.savefig("at_t_cpu.eps",format = "eps")


