import gc

gc.collect()

#%%
from PCA_SPAR import SPCA
from numpy import reshape,mean,ones,arange,shape,std,zeros,load,random
import time
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure

#%%

A = random.normal(4,10,[100,2000])

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

omega.search_multiplier = min(200/s,n/s)

k = 1

up_to = 50

s_var = zeros([up_to,7])
s_cpu = zeros([up_to,7])

ss_var = zeros([up_to,7])
ss_cpu = zeros([up_to,7])

sss_var = zeros([up_to,7])
sss_cpu = zeros([up_to,7])

sparsity = 5
omega.s = sparsity
omega.args["sparsity"] = sparsity
print("!!! Current Sparsity level",sparsity,"!!!")


t0 = time.process_time()
gd_set,gd_val = omega.column_norm_1_old()
t1 = time.process_time()
print("gerschgorin done ")

t2 = time.process_time()
ccw_set,ccw_val = omega.correlation_cw_old()
t3 = time.process_time()
print("CCW done ")

t4 = time.process_time()
fcw_set,fcw_val = omega.frobenius_cw_old()
t5 = time.process_time()
print("FCW done ")

t6 = time.process_time()
em_set,em_val = omega.EM()
t7 = time.process_time()
print("EM done ")

t8 = time.process_time()
path_set,path_val = omega.cholesky_mk2_old()
t9 = time.process_time()
print("Path/Chol done ")

t10 = time.process_time()
gp_set,gp_val = omega.GPower()
t11 = time.process_time()

print("GPower done ")

t12 = time.process_time()
P_pcw,pcw_set,pcw_val = omega.greedy_forward()
t13 = time.process_time()

print("PCW step 1 done ")

# print("----------------------------")
# print("gerschgorin  ",t1-t0)
# print("correlation  ",t3-t2)
# print("frobenius    ",t5-t4)
# print("cholesky     ",t7-t6)
# print("PCW          ",t13-t12)
# print("----------------------------")
# print("gerschgorin  ",gd_val)
# print("correlation  ",ccw_val)
# print("frobenius    ",fcw_val)
# print("cholesky     ",path_val)
# print("PCW          ",pcw_val)

print("Imposing Stationarity")
val,vec,vec0 = omega.eigen_pair0(gd_set)

t14 = time.process_time()
gdf,gdfv = omega.EM_mk2(vec0,gd_set)
t15 = time.process_time()
print("GD done ")

val,vec,vec0 = omega.eigen_pair0(ccw_set)

t16 = time.process_time()
ccwf,ccwfv = omega.EM_mk2(vec0,ccw_set)
t17 = time.process_time()
print("CCW done ")

val,vec,vec0 = omega.eigen_pair0(fcw_set)

t18 = time.process_time()
fcwf,fcwfv = omega.EM_mk2(vec0,fcw_set)
t19 = time.process_time()
print("FCW done ")

val,vec,vec0 = omega.eigen_pair0(path_set)

t22 = time.process_time()
pathf,pathfv = omega.EM_mk2(vec0,path_set)
t23 = time.process_time()
print("Path done ")

val,vec,vec0 = omega.eigen_pair0(pcw_set)

t26 = time.process_time()
pcwf,pcwfv = omega.EM_mk2(vec0,pcw_set)
t27 = time.process_time()
print("PCW done ")

print("Imposing CW-Optimality")
P_gd = list(range(n))
for element in gd_set:
    P_gd.remove(element)
_,x_gd = omega.eigen_pair(gd_set)

t28 = time.process_time()
gdf,gdfv2 = omega.PCW2_iterative(gd_val,x_gd,P_gd,gd_set)
t29 = time.process_time()
print("GD done ")

P_ccw = list(range(n))
for element in ccw_set:
    P_ccw.remove(element)
_,x_ccw = omega.eigen_pair(ccw_set)

t30 = time.process_time()
ccwf,ccwfv2 = omega.PCW2_iterative(ccw_val,x_ccw,P_ccw,ccw_set)
t31 = time.process_time()
print("CCW done ")

P_fcw = list(range(n))
for element in fcw_set:
    P_fcw.remove(element)
_,x_fcw = omega.eigen_pair(fcw_set)

t32 = time.process_time()
fcwf,fcwfv2 = omega.PCW2_iterative(fcw_val,x_fcw,P_fcw,fcw_set)
t33 = time.process_time()
print("FCW done ")

P_em = list(range(n))
for element in em_set:
    P_em.remove(element)
_,x_em = omega.eigen_pair(em_set)

t34 = time.process_time()
emf,emfv2 = omega.PCW2_iterative(em_val,x_em,P_em,list(em_set))
t35 = time.process_time()
print("EM done ")

P_path = list(range(n))
for element in path_set:
    P_path.remove(element)
_,x_path = omega.eigen_pair(path_set)

t36 = time.process_time()
pathf,pathfv2 = omega.PCW2_iterative(path_val,x_path,P_path,path_set)
t37 = time.process_time()
print("Path done ")

P_gp = list(range(n))
for element in gp_set:
    P_gp.remove(element)
_,x_gp = omega.eigen_pair(gp_set)

t38 = time.process_time()
gpf,gpfv2 = omega.PCW2_iterative(gp_val,x_gp,P_gp,list(gp_set))
t39 = time.process_time()
print("AM done ")

_,x_pcw = omega.eigen_pair(pcw_set)

t40 = time.process_time()
pcwf,pcwfv2 = omega.PCW2_iterative(pcw_val,x_pcw,P_pcw,pcw_set)
t41 = time.process_time()
print("PCW done ")
    
s_cpu[0,0] = t1-t0
s_cpu[0,1] = t3-t2
s_cpu[0,2] = t5-t4
s_cpu[0,3] = t7-t6
s_cpu[0,4] = t9-t8
s_cpu[0,5] = t11-t10
s_cpu[0,6] = t13-t12

ss_cpu[0,0] = t15-t14
ss_cpu[0,1] = t17-t16
ss_cpu[0,2] = t19-t18
ss_cpu[0,3] = 0
ss_cpu[0,4] = t23-t22
ss_cpu[0,5] = 0
ss_cpu[0,6] = t27-t26

sss_cpu[0,0] = t29-t28
sss_cpu[0,1] = t31-t30
sss_cpu[0,2] = t33-t32
sss_cpu[0,3] = t35-t34
sss_cpu[0,4] = t37-t36
sss_cpu[0,5] = t39-t38
sss_cpu[0,6] = t41-t40

s_var[0,0] = gd_val
s_var[0,1] = ccw_val
s_var[0,2] = fcw_val
s_var[0,3] = em_val
s_var[0,4] = path_val
s_var[0,5] = gp_val
s_var[0,6] = pcw_val

ss_var[0,0] = gdfv
ss_var[0,1] = ccwfv
ss_var[0,2] = fcwfv
ss_var[0,3] = em_val
ss_var[0,4] = pathfv
ss_var[0,5] = gp_val
ss_var[0,6] = pcwfv

sss_var[0,0] = gdfv2
sss_var[0,1] = ccwfv2
sss_var[0,2] = fcwfv2
sss_var[0,3] = emfv2
sss_var[0,4] = pathfv2
sss_var[0,5] = gpfv2
sss_var[0,6] = pcwfv2
    
for iteration in range(1,up_to):
    gc.collect()
    
    sparsity = (iteration+1) * 5
    omega.s = sparsity
    omega.args["sparsity"] = sparsity
    print("!!! Current Sparsity level",sparsity,"!!!")
    omega.search_multiplier = min(200/sparsity,n/sparsity)
    
    t0 = time.process_time()
    gd_set,gd_val = omega.column_norm_1_old()
    t1 = time.process_time()
    print("gerschgorin done ")
    
    t2 = time.process_time()
    ccw_set,ccw_val = omega.correlation_cw_old()
    t3 = time.process_time()
    print("CCW done ")
    
    t4 = time.process_time()
    fcw_set,fcw_val = omega.frobenius_cw_old()
    t5 = time.process_time()
    print("FCW done ")
    
    t6 = time.process_time()
    em_set,em_val = omega.EM()
    t7 = time.process_time()
    print("EM done ")
    
    t8 = time.process_time()
    path_set,path_val = omega.cholesky_mk2_old()
    t9 = time.process_time()
    print("Path/Chol done ")
    
    t10 = time.process_time()
    gp_set,gp_val = omega.GPower()
    t11 = time.process_time()
    
    print("GPower done ")
    
    t12 = time.process_time()
    P_pcw,pcw_set,pcw_val = omega.greedy_forward_efficient(P_pcw,pcw_set,pcw_val)
    t13 = time.process_time()
    
    print("PCW step 1 done ")
    
    # print("----------------------------")
    # print("gerschgorin  ",t1-t0)
    # print("correlation  ",t3-t2)
    # print("frobenius    ",t5-t4)
    # print("Path     ",t7-t6)
    # print("PCW          ",t13-t12)
    # print("----------------------------")
    # print("gerschgorin  ",gd_val)
    # print("correlation  ",ccw_val)
    # print("frobenius    ",fcw_val)
    # print("Path     ",path_val)
    # print("PCW          ",pcw_val)
    
    print("Imposing Stationarity")
    val,vec,vec0 = omega.eigen_pair0(gd_set)
    
    t14 = time.process_time()
    gdf,gdfv = omega.EM_mk2(vec0,gd_set)
    t15 = time.process_time()
    print("GD done ")
    
    val,vec,vec0 = omega.eigen_pair0(ccw_set)
    
    t16 = time.process_time()
    ccwf,ccwfv = omega.EM_mk2(vec0,ccw_set)
    t17 = time.process_time()
    print("CCW done ")
    
    val,vec,vec0 = omega.eigen_pair0(fcw_set)
    
    t18 = time.process_time()
    fcwf,fcwfv = omega.EM_mk2(vec0,fcw_set)
    t19 = time.process_time()
    print("FCW done ")
    
    val,vec,vec0 = omega.eigen_pair0(path_set)
    
    t22 = time.process_time()
    pathf,pathfv = omega.EM_mk2(vec0,path_set)
    t23 = time.process_time()
    print("Path done ")
    
    val,vec,vec0 = omega.eigen_pair0(pcw_set)
    
    t26 = time.process_time()
    pcwf,pcwfv = omega.EM_mk2(vec0,pcw_set)
    t27 = time.process_time()
    print("PCW done ")

    print("Imposing CW-Optimality")
    P_gd = list(range(n))
    for element in gd_set:
        P_gd.remove(element)
    _,x_gd = omega.eigen_pair(gd_set)
    
    t28 = time.process_time()
    gdf,gdfv2 = omega.PCW2_iterative(gd_val,x_gd,P_gd,gd_set)
    t29 = time.process_time()
    print("GD done ")
    
    P_ccw = list(range(n))
    for element in ccw_set:
        P_ccw.remove(element)
    _,x_ccw = omega.eigen_pair(ccw_set)
    
    t30 = time.process_time()
    ccwf,ccwfv2 = omega.PCW2_iterative(ccw_val,x_ccw,P_ccw,ccw_set)
    t31 = time.process_time()
    print("CCW done ")
    
    P_fcw = list(range(n))
    for element in fcw_set:
        P_fcw.remove(element)
    _,x_fcw = omega.eigen_pair(fcw_set)
    
    t32 = time.process_time()
    fcwf,fcwfv2 = omega.PCW2_iterative(fcw_val,x_fcw,P_fcw,fcw_set)
    t33 = time.process_time()
    print("FCW done ")
    
    P_em = list(range(n))
    for element in em_set:
        P_em.remove(element)
    _,x_em = omega.eigen_pair(em_set)
    
    t34 = time.process_time()
    emf,emfv2 = omega.PCW2_iterative(em_val,x_em,P_em,list(em_set))
    t35 = time.process_time()
    print("EM done ")
    
    P_path = list(range(n))
    for element in path_set:
        P_path.remove(element)
    _,x_path = omega.eigen_pair(path_set)
    
    t36 = time.process_time()
    pathf,pathfv2 = omega.PCW2_iterative(path_val,x_path,P_path,path_set)
    t37 = time.process_time()
    print("Path done ")
    
    P_gp = list(range(n))
    for element in gp_set:
        P_gp.remove(element)
    _,x_gp = omega.eigen_pair(gp_set)
    
    t38 = time.process_time()
    gpf,gpfv2 = omega.PCW2_iterative(gp_val,x_gp,P_gp,list(gp_set))
    t39 = time.process_time()
    print("AM done ")
    
    _,x_pcw = omega.eigen_pair(pcw_set)
    
    t40 = time.process_time()
    pcwf,pcwfv2 = omega.PCW2_iterative(pcw_val,x_pcw,P_pcw,pcw_set)
    t41 = time.process_time()
    print("PCW done ")
    
    s_cpu[iteration,0] = t1-t0
    s_cpu[iteration,1] = t3-t2
    s_cpu[iteration,2] = t5-t4
    s_cpu[iteration,3] = t7-t6
    s_cpu[iteration,4] = t9-t8
    s_cpu[iteration,5] = t11-t10
    s_cpu[iteration,6] = t13-t12 + s_cpu[iteration-1,6]
    
    ss_cpu[iteration,0] = t15-t14
    ss_cpu[iteration,1] = t17-t16
    ss_cpu[iteration,2] = t19-t18
    ss_cpu[iteration,3] = 0
    ss_cpu[iteration,4] = t23-t22
    ss_cpu[iteration,5] = 0
    ss_cpu[iteration,6] = t27-t26
    
    sss_cpu[iteration,0] = t29-t28
    sss_cpu[iteration,1] = t31-t30
    sss_cpu[iteration,2] = t33-t32
    sss_cpu[iteration,3] = t35-t34
    sss_cpu[iteration,4] = t37-t36
    sss_cpu[iteration,5] = t39-t38
    sss_cpu[iteration,6] = t41-t40
    
    s_var[iteration,0] = gd_val
    s_var[iteration,1] = ccw_val
    s_var[iteration,2] = fcw_val
    s_var[iteration,3] = em_val
    s_var[iteration,4] = path_val
    s_var[iteration,5] = gp_val
    s_var[iteration,6] = pcw_val
    
    ss_var[iteration,0] = gdfv
    ss_var[iteration,1] = ccwfv
    ss_var[iteration,2] = fcwfv
    ss_var[iteration,3] = em_val
    ss_var[iteration,4] = pathfv
    ss_var[iteration,5] = gp_val
    ss_var[iteration,6] = pcwfv

    sss_var[iteration,0] = gdfv2
    sss_var[iteration,1] = ccwfv2
    sss_var[iteration,2] = fcwfv2
    sss_var[iteration,3] = emfv2
    sss_var[iteration,4] = pathfv2
    sss_var[iteration,5] = gpfv2
    sss_var[iteration,6] = pcwfv2
    
ts_cpu = s_cpu+ss_cpu
tss_cpu = s_cpu+sss_cpu

#%%

sparsity = arange(5,5 * (up_to + 1),5)

varf = figure()
grid(True)
plot(sparsity,s_var[:,0],color = "blue")
plot(sparsity,s_var[:,1],color = "red")
plot(sparsity,s_var[:,2],color = "green")
plot(sparsity,s_var[:,3],color = "olive")
plot(sparsity,s_var[:,4],color = "orange")
plot(sparsity,s_var[:,5],color = "purple")
plot(sparsity,s_var[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("Variance")
title("Variance Against Sparsity Level Pattern Generation")

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
title("CPU time Against Sparsity Level Pattern Generation")

varf2 = figure()
grid(True)
plot(sparsity,ss_var[:,0],color = "blue")
plot(sparsity,ss_var[:,1],color = "red")
plot(sparsity,ss_var[:,2],color = "green")
plot(sparsity,ss_var[:,3],color = "olive")
plot(sparsity,ss_var[:,4],color = "orange")
plot(sparsity,ss_var[:,5],color = "purple")
plot(sparsity,ss_var[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("Variance")
title("Variance Against Sparsity Level Imposing Stationarity")

cpuf2 = figure()
grid(True)
plot(sparsity,ts_cpu[:,0],color = "blue")
plot(sparsity,ts_cpu[:,1],color = "red")
plot(sparsity,ts_cpu[:,2],color = "green")
plot(sparsity,ts_cpu[:,3],color = "olive")
plot(sparsity,ts_cpu[:,4],color = "orange")
plot(sparsity,ts_cpu[:,5],color = "purple")
plot(sparsity,ts_cpu[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("CPU time (s)")
title("Total CPU time Against Sparsity Level Imposing Stationarity")


varf3 = figure()
grid(True)
plot(sparsity,sss_var[:,0],color = "blue")
plot(sparsity,sss_var[:,1],color = "red")
plot(sparsity,sss_var[:,2],color = "green")
plot(sparsity,sss_var[:,3],color = "olive")
plot(sparsity,sss_var[:,4],color = "orange")
plot(sparsity,sss_var[:,5],color = "purple")
plot(sparsity,sss_var[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("Variance")
title("Variance Against Sparsity Level Imposing CW-Optimality")

cpuf3 = figure()
grid(True)
plot(sparsity,tss_cpu[:,0],color = "blue")
plot(sparsity,tss_cpu[:,1],color = "red")
plot(sparsity,tss_cpu[:,2],color = "green")
plot(sparsity,tss_cpu[:,3],color = "olive")
plot(sparsity,tss_cpu[:,4],color = "orange")
plot(sparsity,tss_cpu[:,5],color = "purple")
plot(sparsity,tss_cpu[:,6],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","Greedy"])
xlabel("Sparsity")
ylabel("CPU time (s)")
title("Total CPU time Against Sparsity Level Imposing CW-Optimality")

varf.savefig("CW_comparison_variance.eps",format = "eps")
cpuf.savefig("CW_comparison_cpu.eps",format = "eps")

varf2.savefig("CW_comparison_variance2.eps",format = "eps")
cpuf2.savefig("CW_comparison_cpu2.eps",format = "eps")

varf3.savefig("CW_comparison_variance3.eps",format = "eps")
cpuf3.savefig("CW_comparison_cpu3.eps",format = "eps")

