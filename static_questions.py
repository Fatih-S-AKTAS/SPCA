from numpy import array,zeros
import gc

pitprops = zeros([13,13])
z8 = open("pitprops.txt","r")
header8 = z8.readline()
for i in range(13):
    line = z8.readline()
    v = line.strip()
    v = v.split(" ")
    fv = list(map(float,v))
    pitprops[:,i] = fv
    
z8.close()

micromass = zeros([571,1300])
z10 = open("pure_spectra_matrix.csv","r")
for i in range(452):
    line = z10.readline()
    v = line.strip()
    v = v.split(";")
    fv = list(map(float,v))
    micromass[i,:] = fv
z10.close()