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

s = 20

omega = SPCA(A,s)

#%%

k = 10

up_to = 50

s_var = zeros([up_to,7])
s_cpu = zeros([up_to,7])
    
for iteration in range(0,up_to):
    gc.collect()
    
    sparsity = (iteration+1) * 5
    omega.s = sparsity
    omega.args["sparsity"] = sparsity
    print("!!! Current Sparsity level",sparsity,"!!!")
    omega.search_multiplier = min(200/sparsity,n/sparsity)
    
    t0 = time.process_time()
    pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
    t1 = time.process_time()
    print("gerschgorin done ")
    
    t2 = time.process_time()
    pattern2,eigens2,load2,component2,variance2 = omega.find_component("CCW",k)
    t3 = time.process_time()
    print("CCW done ")
    
    t4 = time.process_time()
    pattern3,eigens3,load3,component3,variance3 = omega.find_component("FCW",k)
    t5 = time.process_time()
    print("FCW done ")
    
    t6 = time.process_time()
    pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
    t7 = time.process_time()
    print("EM done ")
    
    t8 = time.process_time()
    pattern5,eigens5,load5,component5,variance5 = omega.find_component("Path",k)
    t9 = time.process_time()
    print("Path/Chol done ")
    
    t10 = time.process_time()
    pattern6,eigens6,load6,component6,variance6 = omega.find_component("GPower",k)
    t11 = time.process_time()
    print("PCW done ")
    
    t12 = time.process_time()
    pattern7,eigens7,load7,component7,variance7 = omega.find_component("Greedy",k)
    t13 = time.process_time()
    print("GPower done ")
    
    print("----------------------------")
    print("gerschgorin  ",t1-t0)
    print("correlation  ",t3-t2)
    print("frobenius    ",t5-t4)
    print("EM     ",t7-t6)
    print("Path     ",t9-t8)
    print("GPower     ",t11-t10)
    print("PCW          ",t13-t12)
    print("----------------------------")
    print("gerschgorin  ",sum(variance1))
    print("correlation  ",sum(variance2))
    print("frobenius    ",sum(variance3))
    print("EM          ",sum(variance4))
    print("Path     ",sum(variance5))
    print("GPower    ",sum(variance6))
    print("PCW      ",sum(variance7))
    
    s_cpu[iteration,0] = t1-t0
    s_cpu[iteration,1] = t3-t2
    s_cpu[iteration,2] = t5-t4
    s_cpu[iteration,3] = t7-t6
    s_cpu[iteration,4] = t9-t8
    s_cpu[iteration,5] = t11-t10
    s_cpu[iteration,6] = t13-t12
    
    s_var[iteration,0] = sum(variance1)
    s_var[iteration,1] = sum(variance2)
    s_var[iteration,2] = sum(variance3)
    s_var[iteration,3] = sum(variance4)
    s_var[iteration,4] = sum(variance5)
    s_var[iteration,5] = sum(variance6)
    s_var[iteration,6] = sum(variance7)
    

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
title("Total Variance Against Sparsity Level Pattern Generation")

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

varf.savefig("CW_comparison_multiple_variance.eps",format = "eps")
cpuf.savefig("CW_comparison_multiple_cpu.eps",format = "eps")
