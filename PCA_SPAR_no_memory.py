from numpy import Inf,ndarray,shape,argmax,zeros,argsort,where,diag,sum,\
    arange,delete,absolute,asarray_chkfinite
from scipy.sparse.linalg import eigsh
from scipy.sparse import random as scipy_random
from numpy.linalg import qr,norm,eigvalsh
from nesterov_functions import CustomDistribution
from nesterov_wrapper import run_formulation
from bisect import bisect_left

class SPCA_LM:
    def __init__(self,A,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x n data matrix
        s       = sparsity level
        
        Please contact selim.aktas@ug.bilkent.edu.tr for any bugs, errors and recommendations.
        """
        asarray_chkfinite(A)  
        if type(A) != ndarray:
            print("A should be a numpy ndarray")
        elif type(s) != int:
            print("s should be an integer")
        else:
            self.s = s
            """ registering the sparsity level s """
            self.m,self.n = shape(A)
            """ saving the shape of matrix A explicitly not to repeat use of shape() function """
            self.rem_qsize = []

            self.out = 1
            """ initializing the output choice of algorithm """
            
            self.A0 = A.copy()
            self.A = A.copy()
            d = zeros(self.n)
            for i in range(self.n):
                d[i] = norm(A[:,i]) ** 2
            self.diag = d
            self.diag2 = d ** 2
            
            self.search_multiplier = 10
            self.AM_trial = 10
            self.args = {'formulation': 'L2var_L0cons', 'dataname': 'ATandT_Database_of_Faces', 'dataDir': './data/',\
    'resultsDir': './results/', 'seed': 1, 'density_of_SP': 1, 'sparsity': self.s, 'tol': 1e-06, \
        'maxIter': 200, 'numOfTrials': self.AM_trial, 'stabilityIter': 30, 'incDelta': 0.001}

    def show(self,ind):
        print(self.A[:,ind].T.dot(self.A[:,ind]))
        return self.A[:,ind].T.dot(self.A[:,ind])
    
    def frobenius(self,ind):
        return norm(self.A[:,ind].T.dot(self.A[:,ind]))
    
    def all_eigens(self,ind):
        if len(ind) <= self.m:
            return eigvalsh(self.A[:,ind].T.dot(self.A[:,ind]))
        else:
            return eigvalsh(self.A[:,ind].dot(self.A[:,ind].T))

    def restart(self):
        self.A = self.A0.copy()
        d = zeros(self.n)
        for i in range(self.n):
            d[i] = norm(self.A[:,i]) ** 2
        self.diag = d
        self.diag2 = d ** 2
        
    def deflation_sparse_vector(self,svector,indices):        
        dual_vector = self.A[:,indices].dot(svector)
        self.A[:,indices] = self.A[:,indices] - dual_vector.dot(svector.T)
        d = zeros(self.n)
        for i in range(self.n):
            d[i] = norm(self.A[:,i]) ** 2
        self.diag = d
        self.diag2 = d ** 2
    
    def eigen_upperbound(self,ind):
        if len(ind) <= self.m:
            dominant_eigen_value = eigsh(self.A[:,ind].T.dot(self.A[:,ind]),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
            return dominant_eigen_value[0]
        else:
            dominant_eigen_value = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
            return dominant_eigen_value[0]
    
    def eigen_pair(self,ind):
        if len(ind) <= self.m:
            eigenvalue,eigenvector = eigsh(self.A[:,ind].T.dot(self.A[:,ind]),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
            return eigenvalue,eigenvector
        else:
            eigenvalue,dualeigenvector = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
            eigenvector = dualeigenvector.T.dot(self.A[:,ind])
            eigenvector = eigenvector/norm(eigenvector)
            return eigenvalue,eigenvector[:,ind].T
    
    def eigen_pair0(self,ind):
        if len(ind) <= self.m:
            eigenvalue,eigenvector = eigsh(self.A[:,ind].T.dot(self.A[:,ind]),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
        else:
            eigenvalue,dualeigenvector = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
            eigenvector = dualeigenvector.T.dot(self.A[:,ind])
            eigenvector = eigenvector/norm(eigenvector)
            eigenvector = eigenvector.T
        eigenvector_padded = zeros([self.n,1])
        eigenvector_padded[ind] = eigenvector
        return eigenvalue,eigenvector,eigenvector_padded
    
    def GCW1(self):
        index1 = 0
        index2 = 0
        max_cor = 0
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            col[i] = 0
            j = argmax(col)
            if col[j] > max_cor:
                max_cor = col[j]
                index1 = i
                index2 = j
        C = [index1,index2]
        P = list(range(self.n))
        P.remove(index1)
        P.remove(index2)
        best_value = self.eigen_upperbound(C)
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value = self.eigen_upperbound(C_temp)
                if value > best_value:
                    best_value = value
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return self.GCW2(best_value,P,C)
            
    def GCW2(self,best_value,P,C):
        len_p = len(P)
        len_c = len(C)
        
        temp_best = 0
        enter = -1
        leave = -1
        for i in range(len_c):
            for j in range(len_p):
                C_temp = C + [P[j]]
                C_temp.remove(C[i])
                value = self.eigen_upperbound(C_temp)
                if value > temp_best:
                    temp_best = value
                    enter = j
                    leave = i
        if temp_best > best_value:
            C_temp = C +[P[enter]]
            C_temp.remove(C[leave])
            P += [C[leave]]
            P.remove(P[enter])
            return self.GCW2(value,P,C_temp)
        else:
            return C,best_value

    def GCW1_iterative(self):
        index1 = 0
        index2 = 0
        max_cor = 0
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            col[i] = 0
            j = argmax(col)
            if col[j] > max_cor:
                max_cor = col[j]
                index1 = i
                index2 = j
        C = [index1,index2]
        P = list(range(self.n))
        P.remove(index1)
        P.remove(index2)
        best_value = self.eigen_upperbound(C)
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value = self.eigen_upperbound(C_temp)
                if value > best_value:
                    best_value = value
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return self.GCW2_iterative(best_value,P,C)
            
    def GCW2_iterative(self,best_value,P,C):
        len_p = len(P)
        len_c = len(C)
        
        change = True
        while change:
            change = False
            temp_best = 0
            enter = -1
            leave = -1
            for i in range(len_c):
                for j in range(len_p):
                    C_temp = C + [P[j]]
                    C_temp.remove(C[i])
                    value = self.eigen_upperbound(C_temp)
                    if value > temp_best:
                        temp_best = value
                        enter = j
                        leave = i
            if temp_best > best_value:
                best_value = temp_best
                C = C + [P[enter]]
                P += [C[leave]]
                C.remove(C[leave])
                P.remove(P[enter])
                change = True
        
        return C,best_value
           
    def greedy_forward(self):
        index1 = 0
        index2 = 0
        max_cor = 0
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            col[i] = 0
            j = argmax(col)
            if col[j] > max_cor:
                max_cor = col[j]
                index1 = i
                index2 = j
        C = [index1,index2]
        P = list(range(self.n))
        P.remove(index1)
        P.remove(index2)
        best_value = [self.eigen_upperbound(C)]
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value = self.eigen_upperbound(C_temp)
                if value > best_value:
                    best_value = value
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return P,C,best_value

    def greedy_forward_stationary(self):
        index1 = 0
        index2 = 0
        max_cor = 0
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            col[i] = 0
            j = argmax(col)
            if col[j] > max_cor:
                max_cor = col[j]
                index1 = i
                index2 = j
        C = [index1,index2]
        P = list(range(self.n))
        P.remove(index1)
        P.remove(index2)
        best_value = [self.eigen_upperbound(C)]
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value = self.eigen_upperbound(C_temp)
                if value > best_value:
                    best_value = value
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        
        val,vec,vec0 = self.eigen_pair0(C)
        best_set,best_val = self.EM_mk2(vec0,C)
        return best_set,best_val
    
    def greedy_forward_efficient(self,P,C,best_value):
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value = self.eigen_upperbound(C_temp)
                if value > best_value:
                    best_value = value
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return P,C,best_value
    
    def PCW1(self):
        index1 = 0
        index2 = 0
        max_cor = 0
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            col[i] = 0
            j = argmax(col)
            if col[j] > max_cor:
                max_cor = col[j]
                index1 = i
                index2 = j
        C = [index1,index2]
        P = list(range(self.n))
        P.remove(index1)
        P.remove(index2)
        best_value = [self.eigen_upperbound(C)]
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value,vector = self.eigen_pair(C_temp)
                if value > best_value:
                    best_value = value
                    best_vector = vector
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return self.PCW2(best_value[0],best_vector,P,C)
     
    def PCW2(self,best_value,best_vector,P,C):
        len_p = len(P)
        len_c = len(C)
        sorted_vector = argsort(abs(best_vector[:,0]))
        t = 0
        while t < len_c:
            i = sorted_vector[t]
            t += 1
            temp_best = 0
            enter = -1
            for j in range(len_p):
                C_temp = C + [P[j]]
                C_temp.remove(C[i])
                value,vector = self.eigen_pair(C_temp)
                if value[0] > temp_best:
                    temp_best = value[0]
                    enter = j
            if temp_best > best_value:
                C_temp = C + [P[enter]]
                C_temp.remove(C[i])
                P += [C[i]]
                P.remove(P[enter])
                return self.PCW2(temp_best,vector,P,C_temp)
            
        return C,best_value

    def PCW1_iterative(self):
        index1 = 0
        index2 = 0
        max_cor = 0
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            col[i] = 0
            j = argmax(col)
            if col[j] > max_cor:
                max_cor = col[j]
                index1 = i
                index2 = j
        C = [index1,index2]
        P = list(range(self.n))
        P.remove(index1)
        P.remove(index2)
        best_value = self.eigen_upperbound(C)
        len_c = len(C)
        len_p = len(P)
        while len_c < self.s:
            len_c += 1
            len_p -= 1
            best_index = 0
            best_value = 0
            for i in range(len_p):
                C_temp = C + [P[i]]
                value,vector = self.eigen_pair(C_temp)
                if value > best_value:
                    best_value = value
                    best_vector = vector
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return self.PCW2_iterative(best_value[0],best_vector,P,C)
    
    def PCW2_iterative(self,best_value,best_vector,P,C):
        len_p = len(P)
        len_c = len(C)
        sorted_vector = argsort(abs(best_vector[:,0]))
        change = True
        while change:
            change = False
            t = 0
            while t < len_c:
                i = sorted_vector[t]
                t += 1
                temp_best = 0
                enter = -1
                for j in range(len_p):
                    C_temp = C + [P[j]]
                    C_temp.remove(C[i])
                    value,vector = self.eigen_pair(C_temp)
                    if value[0] > temp_best:
                        temp_best = value[0]
                        enter = j
                if temp_best > best_value:
                    best_value = temp_best
                    P += [C[i]]
                    C += [P[enter]]
                    P.remove(P[enter])
                    C.remove(C[i])
                    sorted_vector = argsort(abs(vector[:,0]))
                    change = True
                    break
            
        return C,best_value

    def column_norm_1(self):
        top_s = zeros(int(self.s * self.search_multiplier))
        R = zeros((int(self.s * self.search_multiplier),self.s),dtype = int)
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            disk_set = argsort(col)[-self.s:]
            disk_radius = col.sum()
            if disk_radius > top_s[0]:
                insert_location = bisect_left(top_s,disk_radius)
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = disk_radius
                R[:insert_location-1,:] = R[1:insert_location,:]
                R[insert_location-1,:] = disk_set
        best_set = []
        best_val = 0
        for i in range(int(self.s * self.search_multiplier)):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
        
        val,vec,vec0 = self.eigen_pair0(best_set)
        best_set,best_val = self.EM_mk2(vec0,best_set)
        return best_set,best_val

    def column_norm_1_old(self):
        top_s = zeros(int(self.s * self.search_multiplier))
        R = zeros((int(self.s * self.search_multiplier),self.s),dtype = int)
        for i in range(self.n):
            col = abs(self.A[:,i].T.dot(self.A))
            disk_set = argsort(col)[-self.s:]
            disk_radius = col.sum()
            if disk_radius > top_s[0]:
                insert_location = bisect_left(top_s,disk_radius)
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = disk_radius
                R[:insert_location-1,:] = R[1:insert_location,:]
                R[insert_location-1,:] = disk_set
        best_set = []
        best_val = 0
        for i in range(int(self.s * self.search_multiplier)):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
        
        return best_set,best_val
    
    def correlation_cw(self):
        top_s = zeros(int(self.s * self.search_multiplier),dtype = int)
        top_s_disk_radius = zeros(int(self.s * self.search_multiplier))
        for i in range(self.n):
            disk_radius = norm(self.A[:,i].T.dot(self.A),ord = 1)
            if disk_radius > top_s_disk_radius[0]:
                insert_location = bisect_left(top_s_disk_radius,disk_radius)
                top_s_disk_radius[:insert_location-1] = top_s_disk_radius[1:insert_location]
                top_s_disk_radius[insert_location-1] = disk_radius
                
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = i
        best_set = []
        best_val = 0
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(absolute(self.A[:,current].T.dot(self.A[:,possible])),axis = 0) + self.diag[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        val,vec,vec0 = self.eigen_pair0(best_set)
        best_set,best_val = self.EM_mk2(vec0,best_set)
        return best_set,best_val

    def correlation_cw_old(self):
        top_s = zeros(int(self.s * self.search_multiplier),dtype = int)
        top_s_disk_radius = zeros(int(self.s * self.search_multiplier))
        for i in range(self.n):
            disk_radius = norm(self.A[:,i].T.dot(self.A),ord = 1)
            if disk_radius > top_s_disk_radius[0]:
                insert_location = bisect_left(top_s_disk_radius,disk_radius)
                top_s_disk_radius[:insert_location-1] = top_s_disk_radius[1:insert_location]
                top_s_disk_radius[insert_location-1] = disk_radius
                
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = i
        best_set = []
        best_val = 0
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(absolute(self.A[:,current].T.dot(self.A[:,possible])),axis = 0) + self.diag[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        return best_set,best_val
    
    def frobenius_cw(self):
        top_s = zeros(int(self.s * self.search_multiplier),dtype = int)
        top_s_disk_radius = zeros(int(self.s * self.search_multiplier))
        for i in range(self.n):
            disk_radius = norm(self.A[:,i].T.dot(self.A),ord = 1)
            if disk_radius > top_s_disk_radius[0]:
                insert_location = bisect_left(top_s_disk_radius,disk_radius)
                top_s_disk_radius[:insert_location-1] = top_s_disk_radius[1:insert_location]
                top_s_disk_radius[insert_location-1] = disk_radius
                
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = i
        best_set = []
        best_val = 0
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(self.A[:,current].T.dot(self.A[:,possible]) ** 2,axis = 0) + self.diag2[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        val,vec,vec0 = self.eigen_pair0(best_set)
        best_set,best_val = self.EM_mk2(vec0,best_set)
        return best_set,best_val

    def frobenius_cw_old(self):
        top_s = zeros(int(self.s * self.search_multiplier),dtype = int)
        top_s_disk_radius = zeros(int(self.s * self.search_multiplier))
        for i in range(self.n):
            disk_radius = norm(self.A[:,i].T.dot(self.A),ord = 1)
            if disk_radius > top_s_disk_radius[0]:
                insert_location = bisect_left(top_s_disk_radius,disk_radius)
                top_s_disk_radius[:insert_location-1] = top_s_disk_radius[1:insert_location]
                top_s_disk_radius[insert_location-1] = disk_radius
                
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = i
        best_set = []
        best_val = 0
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(self.A[:,current].T.dot(self.A[:,possible]) ** 2,axis = 0) + self.diag2[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        return best_set,best_val
    
    def cholesky_mk2_old(self):
        i = 0
        max_col_norm = 0
        for j in range(self.n):
            col_norm = norm(self.A[:,j].T.dot(self.A),ord = 2)
            if col_norm > max_col_norm:
                max_col_norm = col_norm
                i = j

        P = arange(self.n)
        
        current = [i]
        P = delete(P,i)
        
        x = self.A[:,[i]]/norm(self.A[:,i])
        vals = x.T.dot(self.A[:,P]) ** 2
        new_index = argmax(vals)
        current.append(P[new_index])
        P = delete(P,new_index)
        
        for i in range(self.s - 2):
            value,vector = self.eigen_pair(current)
            x = self.A[:,current].dot(vector)
            x = x/norm(x)
            vals = x.T.dot(self.A[:,P]) ** 2
            new_index = argmax(vals)
            current.append(P[new_index])
            P = delete(P,new_index)
        
        return current,self.eigen_upperbound(current)

    def cholesky_mk2(self):
        i = 0
        max_col_norm = 0
        for j in range(self.n):
            col_norm = norm(self.A[:,j].T.dot(self.A),ord = 2)
            if col_norm > max_col_norm:
                max_col_norm = col_norm
                i = j

        P = arange(self.n)
        
        current = [i]
        P = delete(P,i)
        
        x = self.A[:,[i]]/norm(self.A[:,i])
        vals = x.T.dot(self.A[:,P]) ** 2
        new_index = argmax(vals)
        current.append(P[new_index])
        P = delete(P,new_index)
        
        for i in range(self.s - 2):
            value,vector = self.eigen_pair(current)
            x = self.A[:,current].dot(vector)
            x = x/norm(x)
            vals = x.T.dot(self.A[:,P]) ** 2
            new_index = argmax(vals)
            current.append(P[new_index])
            P = delete(P,new_index)
        
        val,vec,vec0 = self.eigen_pair0(current)
        best_set,best_val = self.EM_mk2(vec0,current)
        return best_set,best_val
    
    def cholesky_mk3(self):
        top_s = zeros(int(self.s * self.search_multiplier),dtype = int)
        top_s_disk_radius = zeros(int(self.s * self.search_multiplier))
        for i in range(self.n):
            disk_radius = norm(self.A[:,i].T.dot(self.A),ord = 2)
            if disk_radius > top_s[0]:
                insert_location = bisect_left(top_s_disk_radius,disk_radius)
                top_s_disk_radius[:insert_location-1] = top_s_disk_radius[1:insert_location]
                top_s_disk_radius[insert_location-1] = disk_radius
                
                top_s[:insert_location-1] = top_s[1:insert_location]
                top_s[insert_location-1] = i
        best_val = 0
        best_set = []
        for i in top_s:
            current = [i]
            P = arange(self.n)
            P = delete(P,i)
            
            x = self.A[:,[i]]/norm(self.A[:,i])
            vals = x.T.dot(self.A[:,P]) ** 2
            new_index = argmax(vals)
            current.append(P[new_index])
            P = delete(P,new_index)
            
            for i in range(self.s - 2):
                value,vector = self.eigen_pair(current)
                x = self.A[:,current].dot(vector)
                x = x/norm(x)
                vals = x.T.dot(self.A[:,P]) ** 2
                new_index = argmax(vals)
                current.append(P[new_index])
                P = delete(P,new_index)
            final_value = self.eigen_upperbound(current)
            if final_value > best_val:
                best_val = final_value
                best_set = current
        return best_set,best_val
        
    def EM(self):
        v,wt = self.eigen_pair(list(range(self.n)))
        order = arange(self.n)
        t = 0
        diff = 1
        while diff >= 1e-6:
            t += 1
            y = self.A[:,order].dot(wt[order])
            w = y.T.dot(self.A)/(norm(y) ** 2)
            order = argsort(-1 * abs(w[0]))[:self.s]
            old_wt = wt
            wt = zeros([self.n,1])
            wt[order,0] = w[0][order]
            wt = wt/norm(wt)
            diff = norm(wt-old_wt)
        pattern = where(abs(wt) > 0)[0]
        value = self.eigen_upperbound(pattern)
        return pattern,value
    
    def EM_mk2(self,x,support):
        t = 0
        diff = 1
        while diff >= 1e-6:
            t += 1
            y = self.A[:,support].dot(x[support])
            w = y.T.dot(self.A)
            w = w/norm(w)
            support = argsort(-1 * abs(w[0]))[:self.s]
            old_x = x.copy()
            x = zeros([self.n,1])
            x[support,0] = w[0][support]
            x = x/norm(x)
            diff = norm(x-old_x)
        pattern = where(abs(x) > 0)[0]
        value = self.eigen_upperbound(pattern)
        return pattern,value
    
    def nesterov(self):
        R = CustomDistribution(seed=self.args['seed'])
        R_obj = R()  # get a frozen version of the distribution
        best_x = None
        bestVar = -Inf
        for seed in range(self.args['numOfTrials']):
            
            self.args['seed'] = seed
            
            # initial point w.r.t. the mentioned seed
            X0 = scipy_random(self.n, 1, density = self.args['density_of_SP'], 
                        random_state = self.args['seed'], 
                        data_rvs = R_obj.rvs)
            
            x, expVar = run_formulation(self.args, self.A, X0, self.n)
            
            if expVar > bestVar:
                bestVar = expVar
                best_x = x
        return best_x.row,bestVar
    
    def find_component(self,algo,k):
        if algo not in ["PCW","GCW","GD","FCW","CCW","EM","Path","Path_mk2","nesterov"]:
            print("bad algortihm choice, choose of one of the following")
            print("GD, CCW, FCW, PCW, GCW, EM, Path,Path_mk2, nesterov")
        elif algo == "PCW":
            algorithm = self.PCW1_iterative
        elif algo == "GCW":
            algorithm = self.GCW1_iterative
        elif algo == "FCW":
            algorithm = self.frobenius_cw
        elif algo == "CCW":
            algorithm = self.correlation_cw
        elif algo == "GD":
            algorithm = self.column_norm_1
        elif algo == "EM":
            algorithm = self.EM
        elif algo == "Path":
            algorithm = self.cholesky_mk2
        elif algo == "Path_mk2":
            algorithm = self.cholesky_mk3
        elif algo == "nesterov":
            algorithm = self.nesterov
        all_loadings = zeros([self.s,k])
        components = zeros([self.m,k])
        loading_patterns = []
        eigen_values = []
        for i in range(k):
            pattern,value = algorithm()
            eigenvalue,eigenvector = self.eigen_pair(pattern)
            loading_patterns.append(pattern)
            eigen_values.append(value)
            all_loadings[:,i] = eigenvector[:,0]
            components[:,i] = self.A0[:,pattern].dot(eigenvector[:,0])
            self.deflation_sparse_vector(eigenvector,pattern)
        r = qr(components,mode = "r")
        variance = diag(r) ** 2
        return loading_patterns,eigen_values,all_loadings,components,variance

            