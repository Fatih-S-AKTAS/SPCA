from PCA_SPAR import SPCA
from numpy import reshape,mean,ones,arange,shape,std,zeros,load,random
import time
from matplotlib.pyplot import plot,grid,xlabel,ylabel,legend,title,figure

from numpy import save as npsave
import gc
from scipy.io import savemat
#%%

k = 1

repeat = 30
up_to = 50

s_var = zeros([repeat,up_to,9])
s_cpu = zeros([repeat,up_to,9])
s_wall = zeros([repeat,up_to,9])

progress = open("ccw10_pcw_progress.txt","w")
try:
    for rep in range(repeat):
        A = random.normal(4,10,[100,10000])
        
        m,n = shape(A)
        
        A = A - A.mean(axis = 0)
        
        sA = std(A,axis = 0)
        
        A = A/sA
        
        s = 20
        
        mdic = {"data": A, "label": "experiment"}
        savemat("data_matrix.mat",mdic)

        omega = SPCA(A,s)
        lambda_max,lambda_v = omega.eigen_pair(list(range(n)))
        for iteration in range(0,up_to):
            progress.write("repetition "+str(rep)+" iteration "+str(iteration)+" begins\n")
            
            sparsity = (iteration+1) * 5
            omega.set_sparsity(sparsity)
            omega.args["numOfTrials"] = 100
            print("!!! Current Repeat",rep,"!!!")
            print("!!! Current Sparsity level",sparsity,"!!!")
            omega.search = min(200,n)
            
            
            w0 = time.time()
            t0 = time.process_time()
            pattern1,eigens1,load1,component1,variance1 = omega.find_component("GD",k)
            t1 = time.process_time()
            w1 = time.time()
            print("gerschgorin done ")
    
            progress.write("gerschgorin passed\n")
            
            w2 = time.time()
            t2 = time.process_time()
            pattern2,eigens2,load2,component2,variance2 = omega.find_component("CCW",k)
            t3 = time.process_time()
            w3 = time.time()
            print("CCW done ")
            
            progress.write("CCW passed\n")
            
            w4 = time.time()
            t4 = time.process_time()
            pattern3,eigens3,load3,component3,variance3 = omega.find_component("FCW",k)
            t5 = time.process_time()
            w5 = time.time()
            print("FCW done ")
            
            progress.write("FCW passed\n")
            
            w6 = time.time()
            t6 = time.process_time()
            pattern4,eigens4,load4,component4,variance4 = omega.find_component("EM",k)
            t7 = time.process_time()
            w7 = time.time()
            print("EM done ")
            
            progress.write("EM passed\n")
            
            w8 = time.time()
            t8 = time.process_time()
            pattern5,eigens5,load5,component5,variance5 = omega.find_component("Path",k)
            t9 = time.process_time()
            w9 = time.time()
            print("Path/Chol done ")
            
            progress.write("Path passed\n")
            
            w10 = time.time()
            t10 = time.process_time()
            pattern6,eigens6,load6,component6,variance6 = omega.find_component("GPower",k)
            t11 = time.process_time()
            w11 = time.time()
            print("GPower done ")
            
            progress.write("Gpower passed\n")
            
            w12 = time.time()
            t12 = time.process_time()
            pattern7,eigens7,load7,component7,variance7 = omega.find_component("PCW_memory",k)
            t13 = time.process_time()
            w13 = time.time()
            print("PCW done ")
            
            progress.write("PCW passed\n")
            
            w14 = time.time()
            t14 = time.process_time()
            pattern8,eigens8,load8,component8,variance8 = omega.find_component("GCW_memory",k)
            t15 = time.process_time()
            w15 = time.time()
            print("GCW done ")
            
            progress.write("GCW passed\n")
            
            w16 = time.time()
            t16 = time.process_time()
            pattern9,eigens9,load9,component9,variance9 = omega.find_component("gpbbls_memory",k)
            t17 = time.process_time()
            w17 = time.time()
            print("GPBB-ls done ")
            
            progress.write("gp-bb-ls passed\n")
                    
            print("----------------------------")
            print("gerschgorin  ",t1-t0)
            print("correlation  ",t3-t2)
            print("frobenius    ",t5-t4)
            print("EM     ",t7-t6)
            print("Path     ",t9-t8)
            print("GPower     ",t11-t10)
            print("PCW          ",t13-t12)
            print("GCW     ",t15-t14)
            print("GPBB-ls    ",t17-t16)
            
            print("----------------------------")
            print("gerschgorin  ",sum(variance1))
            print("correlation  ",sum(variance2))
            print("frobenius    ",sum(variance3))
            print("EM          ",sum(variance4))
            print("Path     ",sum(variance5))
            print("GPower    ",sum(variance6))
            print("PCW      ",sum(variance7))
            print("GCW     ",sum(variance8))
            print("GPBB-ls    ",sum(variance9))
            
            progress.write("print passed\n")
            
            s_cpu[rep,iteration,0] = t1-t0
            s_cpu[rep,iteration,1] = t3-t2
            s_cpu[rep,iteration,2] = t5-t4
            s_cpu[rep,iteration,3] = t7-t6
            s_cpu[rep,iteration,4] = t9-t8
            s_cpu[rep,iteration,5] = t11-t10
            s_cpu[rep,iteration,6] = t13-t12
            s_cpu[rep,iteration,7] = t15-t14
            s_cpu[rep,iteration,8] = t17-t16
            
            s_wall[rep,iteration,0] = w1-w0
            s_wall[rep,iteration,1] = w3-w2
            s_wall[rep,iteration,2] = w5-w4
            s_wall[rep,iteration,3] = w7-w6
            s_wall[rep,iteration,4] = w9-w8
            s_wall[rep,iteration,5] = w11-w10
            s_wall[rep,iteration,6] = w13-w12
            s_wall[rep,iteration,7] = w15-w14
            s_wall[rep,iteration,8] = w17-w16
            
            s_var[rep,iteration,0] = sum(variance1)/lambda_max
            s_var[rep,iteration,1] = sum(variance2)/lambda_max
            s_var[rep,iteration,2] = sum(variance3)/lambda_max
            s_var[rep,iteration,3] = sum(variance4)/lambda_max
            s_var[rep,iteration,4] = sum(variance5)/lambda_max
            s_var[rep,iteration,5] = sum(variance6)/lambda_max
            s_var[rep,iteration,6] = sum(variance7)/lambda_max
            s_var[rep,iteration,7] = sum(variance8)/lambda_max
            s_var[rep,iteration,8] = sum(variance9)/lambda_max
            
            gc.collect()
            progress.write("value register passed\n")
except KeyboardInterrupt:
    print("code aborted because some retard closed session")
    progress.write("code aborted because some retard closed session \n")
except Exception as exc:
    print("code aborted with reason")
    print(exc)
    progress.write("code aborted with reason \n")
    progress.write(str(exc))
    
#%%

progress.write("algorithms passed\n")
        
ss_var = s_var.mean(axis = 0)
ss_cpu = s_cpu.mean(axis = 0)
ss_wall = s_wall.mean(axis = 0)

#%%
sparsity = arange(5,5 * (up_to + 1),5)

varf = figure()
grid(True)
plot(sparsity,ss_var[:,0],color = "lightcoral")
plot(sparsity,ss_var[:,1],color = "red")
plot(sparsity,ss_var[:,2],color = "firebrick")
plot(sparsity,ss_var[:,3],color = "wheat")
plot(sparsity,ss_var[:,4],color = "blue")
plot(sparsity,ss_var[:,5],color = "orange")
plot(sparsity,ss_var[:,6],color = "purple")
plot(sparsity,ss_var[:,7],color = "magenta")
plot(sparsity,ss_var[:,8],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","PCW","GCW","GPBB-ls"])
xlabel("Sparsity")
ylabel("Variance")
title("Percentage of Variance Explained Against Sparsity Level")
# title("Percentage of Total Variance Against Sparsity Level")

cpuf = figure()
grid(True)
plot(sparsity,ss_cpu[:,0],color = "lightcoral")
plot(sparsity,ss_cpu[:,1],color = "red")
plot(sparsity,ss_cpu[:,2],color = "firebrick")
plot(sparsity,ss_cpu[:,3],color = "wheat")
plot(sparsity,ss_cpu[:,4],color = "blue")
plot(sparsity,ss_cpu[:,5],color = "orange")
plot(sparsity,ss_cpu[:,6],color = "purple")
plot(sparsity,ss_cpu[:,7],color = "magenta")
plot(sparsity,ss_cpu[:,8],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","PCW","GCW","GPBB-ls"])
xlabel("Sparsity")
ylabel("CPU Time (s)")
title("CPU Time Against Sparsity Level")

wallf = figure()
grid(True)
plot(sparsity,ss_wall[:,0],color = "lightcoral")
plot(sparsity,ss_wall[:,1],color = "red")
plot(sparsity,ss_wall[:,2],color = "firebrick")
plot(sparsity,ss_wall[:,3],color = "wheat")
plot(sparsity,ss_wall[:,4],color = "blue")
plot(sparsity,ss_wall[:,5],color = "orange")
plot(sparsity,ss_wall[:,6],color = "purple")
plot(sparsity,ss_wall[:,7],color = "magenta")
plot(sparsity,ss_wall[:,8],color = "yellow")
legend(["GD","CCW","FCW","EM","Path","GPower","PCW","GCW","GPBB-ls"])
xlabel("Sparsity")
ylabel("Wall Time (s)")
title("Wall Time Against Sparsity Level")

varf.savefig("CW10_comparison_variance.eps",format = "eps")
cpuf.savefig("CW10_comparison_cpu.eps",format = "eps")
wallf.savefig("CW10_comparison_wall.eps",format = "eps")

varf.savefig("CW10_comparison_variance_png.png",format = "png")
cpuf.savefig("CW10_comparison_cpu_png.png",format = "png")
wallf.savefig("CW10_comparison_wall_png.png",format = "png")

npsave("CW10_comparison_variance.npy",s_var)
npsave("CW10_comparison_cpu.npy",s_cpu)
npsave("CW10_comparison_wall.npy",s_wall)

progress.write("terminated succesfully \n")
progress.close()