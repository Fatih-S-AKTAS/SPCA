import gc

gc.collect()

#%%
from PCA_SPAR import SPCA
from numpy import random,reshape,mean,ones,arange,shape,std,zeros
import time
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure

#%%

A = random.normal(4,10,[100,5000])

m,n = shape(A)
    
mA = reshape(mean(A,axis = 0),[1,n])

A = A - ones([m,1]).dot(mA)

sA = std(A,axis = 0)

A = A/sA

#%%

gc.collect()

s = 20

omega = SPCA(A,s)


#%%

omega.search_multiplier = min(400/s,n/s)

k = 1
    
up_to = 50

s_var = zeros([up_to,5])
s_cpu = zeros([up_to,5])

ss_var = zeros([up_to,5])
ss_cpu = zeros([up_to,5])

sparsity = 5
print("!!! Current Sparsity level",sparsity,"!!!")


t0 = time.process_time()
gd_set,gd_val = omega.column_norm_1()
t1 = time.process_time()
print("gerschgorin done ")

gc.collect()

t2 = time.process_time()
ccw_set,ccw_val = omega.correlation_cw()
t3 = time.process_time()
print("CCW done ")

gc.collect()

t4 = time.process_time()
fcw_set,fcw_val = omega.frobenius_cw()
t5 = time.process_time()
print("FCW done ")

gc.collect()

t6 = time.process_time()
path_set,path_val = omega.cholesky_mk2()
t7 = time.process_time()
print("Path/Chol done ")

gc.collect()

t8 = time.process_time()
P_pcw,pcw_set,pcw_val = omega.greedy_forward()
t9 = time.process_time()

print("PCW done ")

gc.collect()

print("----------------------------")
print("gerschgorin  ",t1-t0)
print("correlation  ",t3-t2)
print("frobenius    ",t5-t4)
print("cholesky     ",t7-t6)
print("PCW          ",t9-t8)
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

s_var[0,0] = gd_val
s_var[0,1] = ccw_val
s_var[0,2] = fcw_val
s_var[0,3] = path_val
s_var[0,4] = pcw_val

for iteration in range(1,up_to):
    sparsity = (iteration+1) * 5
    omega.s = sparsity
    print("!!! Current Sparsity level",sparsity,"!!!")
    omega.search_multiplier = 400/sparsity
    
    t0 = time.process_time()
    gd_set,gd_val = omega.column_norm_1()
    t1 = time.process_time()
    print("gerschgorin done ")
    
    gc.collect()
    
    t2 = time.process_time()
    ccw_set,ccw_val = omega.correlation_cw()
    t3 = time.process_time()
    print("CCW done ")
    
    gc.collect()
    
    t4 = time.process_time()
    fcw_set,fcw_val = omega.frobenius_cw()
    t5 = time.process_time()
    print("FCW done ")
    
    gc.collect()
    
    t6 = time.process_time()
    path_set,path_val = omega.cholesky_mk2()
    t7 = time.process_time()
    print("Path/Chol done ")
    
    gc.collect()
    
    t8 = time.process_time()
    P_pcw,pcw_set,pcw_val = omega.greedy_forward_efficient(P_pcw,pcw_set,pcw_val)
    t9 = time.process_time()
    
    print("PCW done ")
    
    gc.collect()
    
    print("----------------------------")
    print("gerschgorin  ",t1-t0)
    print("correlation  ",t3-t2)
    print("frobenius    ",t5-t4)
    print("cholesky     ",t7-t6)
    print("PCW          ",t9-t8)
    print("----------------------------")
    print("gerschgorin  ",gd_val)
    print("correlation  ",ccw_val)
    print("frobenius    ",fcw_val)
    print("cholesky     ",path_val)
    print("PCW          ",pcw_val)
    
    
    # P_gd = list(range(n))
    # for element in gd_set:
    #     P_gd.remove(element)
    # _,x_gd = omega.eigen_pair(gd_set)
    
    # t10 = time.process_time()
    # gdf,gdfv = omega.PCW2_iterative(gd_val,x_gd,P_gd,gd_set)
    # t11 = time.process_time()
    # print("GD done ")
    
    # gc.collect()
    
    # P_ccw = list(range(n))
    # for element in ccw_set:
    #     P_ccw.remove(element)
    # _,x_ccw = omega.eigen_pair(ccw_set)
    
    # t12 = time.process_time()
    # ccwf,ccwfv = omega.PCW2_iterative(ccw_val,x_ccw,P_ccw,ccw_set)
    # t13 = time.process_time()
    # print("CCW done ")
    
    # gc.collect()
    
    # P_fcw = list(range(n))
    # for element in fcw_set:
    #     P_fcw.remove(element)
    # _,x_fcw = omega.eigen_pair(fcw_set)
    
    # t14 = time.process_time()
    # fcwf,fcwfv = omega.PCW2_iterative(fcw_val,x_fcw,P_fcw,fcw_set)
    # t15 = time.process_time()
    # print("FCW done ")
    
    # gc.collect()
    
    # P_path = list(range(n))
    # for element in path_set:
    #     P_path.remove(element)
    # _,x_path = omega.eigen_pair(path_set)
    
    # t16 = time.process_time()
    # pathf,pathfv = omega.PCW2_iterative(path_val,x_path,P_path,path_set)
    # t17 = time.process_time()
    # print("Path done ")
    
    # gc.collect()
    
    # _,x_pcw = omega.eigen_pair(pcw_set)
    
    # t18 = time.process_time()
    # pcwf,pcwfv = omega.PCW2_iterative(pcw_val,x_pcw,P_pcw,pcw_set)
    # t19 = time.process_time()
    # print("PCW done ")
    
    # gc.collect()
    
    # print("Stronger Version")
    # print("++++++++++++++++++++++++++++")
    # print("gerschgorin  ",t11-t10)
    # print("correlation  ",t13-t12)
    # print("frobenius    ",t15-t14)
    # print("cholesky     ",t17-t16)
    # print("PCW          ",t19-t18)
    # print("++++++++++++++++++++++++++++")
    # print("gerschgorin  ",gdfv)
    # print("correlation  ",ccwfv)
    # print("frobenius    ",fcwfv)
    # print("cholesky     ",pathfv)
    # print("PCW          ",pcwfv)
    
    s_cpu[iteration,0] = t1-t0
    s_cpu[iteration,1] = t3-t2
    s_cpu[iteration,2] = t5-t4
    s_cpu[iteration,3] = t7-t6
    s_cpu[iteration,4] = t9-t8 + s_cpu[iteration-1,4]
    
    s_var[iteration,0] = gd_val
    s_var[iteration,1] = ccw_val
    s_var[iteration,2] = fcw_val
    s_var[iteration,3] = path_val
    s_var[iteration,4] = pcw_val
    
    # ss_cpu[iteration,0] = t11-t10
    # ss_cpu[iteration,1] = t13-t12
    # ss_cpu[iteration,2] = t15-t14
    # ss_cpu[iteration,3] = t17-t16
    # ss_cpu[iteration,4] = t19-t18
    
    # ss_var[iteration,0] = gdfv
    # ss_var[iteration,1] = ccwfv
    # ss_var[iteration,2] = fcwfv
    # ss_var[iteration,3] = pathfv
    # ss_var[iteration,4] = pcwfv

# ts_cpu = s_cpu+ss_cpu

#%%
sparsity = arange(5,5 * (up_to +1),5)

varf = figure()
grid(True)
plot(sparsity,s_var[:,0],color = "blue")
plot(sparsity,s_var[:,1],color = "red")
plot(sparsity,s_var[:,2],color = "green")
plot(sparsity,s_var[:,3],color = "orange")
plot(sparsity,s_var[:,4],color = "yellow")
legend(["GD","CCW","FCW","Path","Greedy"])
xlabel("Sparsity")
ylabel("Variance")
title("Variance Against Sparisty Level Pattern Generation")

cpuf = figure()
grid(True)
plot(sparsity,s_cpu[:,0],color = "blue")
plot(sparsity,s_cpu[:,1],color = "red")
plot(sparsity,s_cpu[:,2],color = "green")
plot(sparsity,s_cpu[:,3],color = "orange")
plot(sparsity,s_cpu[:,4],color = "yellow")
legend(["GD","CCW","FCW","Path","Greedy"])
xlabel("Sparsity")
ylabel("CPU time (s)")
title("CPU time Against Sparisty Level Pattern Generation")

# varf2 = figure()
# grid(True)
# plot(sparsity,ss_var[:,0],color = "blue")
# plot(sparsity,ss_var[:,1],color = "red")
# plot(sparsity,ss_var[:,2],color = "green")
# plot(sparsity,ss_var[:,3],color = "orange")
# plot(sparsity,ss_var[:,4],color = "cyan")
# legend(["GD","CCW","FCW","Path","PCW"])
# xlabel("Sparsity")
# ylabel("Variance")
# title("Variance Against Sparisty Level Imposing Stationarity")

# cpuf2 = figure()
# grid(True)
# plot(sparsity,ss_cpu[:,0],color = "blue")
# plot(sparsity,ss_cpu[:,1],color = "red")
# plot(sparsity,ss_cpu[:,2],color = "green")
# plot(sparsity,ss_cpu[:,3],color = "orange")
# plot(sparsity,ss_cpu[:,4],color = "cyan")
# legend(["GD","CCW","FCW","Path","PCW"])
# xlabel("Sparsity")
# ylabel("CPU time (s)")
# title("CPU time Against Sparisty Level Imposing Stationarity")

# cpuf3 = figure()
# grid(True)
# plot(sparsity,ts_cpu[:,0],color = "blue")
# plot(sparsity,ts_cpu[:,1],color = "red")
# plot(sparsity,ts_cpu[:,2],color = "green")
# plot(sparsity,ts_cpu[:,3],color = "orange")
# plot(sparsity,ts_cpu[:,4],color = "cyan")
# legend(["GD","CCW","FCW","Path","PCW"])
# xlabel("Sparsity")
# ylabel("CPU time (s)")
# title("Total CPU time Against Sparisty Level")

varf.savefig("CW5_comparison_variance.eps",format = "eps")
cpuf.savefig("CW5_comparison_cpu.eps",format = "eps")

# varf2.savefig("CW5_comparison_variance2.eps",format = "eps")
# cpuf2.savefig("CW5_comparison_cpu2.eps",format = "eps")

# cpuf3.savefig("CW5_comparison_cpu3.eps",format = "eps")
