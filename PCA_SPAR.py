from numpy import Inf,ndarray,shape,argmax,zeros,argsort,where,diag,sum,\
    arange,delete,absolute,asarray_chkfinite
from scipy.sparse.linalg import eigsh
from scipy.sparse import random as scipy_random
from scipy.linalg import qr
from numpy.linalg import norm,eigvalsh

from queue import PriorityQueue

from AM_functions import CustomDistribution
from AM_wrapper import run_formulation

"""
This is a class based implementation of SPCA algorithms described in the paper

Fatih S. Aktaş, Ömer Ekmekcioğlu, Mustafa Ç. Pınar, "Improved Greedy Enumeration Algorithms for Sparse PCA"

If you use this software in your project, please cite the related paper.

Please contact selim.aktas@bilkent.edu.tr for any bugs, errors and recommendations.
"""

class SPCA:
    def __init__(self,A,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x n data matrix
        s       = sparsity level
        -------------------------------------------------------------------------------------
        Suggested use of code;
        
        class_instance = SPCA(A,s)
        pattern,eigens,load,component,variance = class_instance.find_component(algorithm,number_of_components)
        
        pattern       = list of size number_of_components, of lists, contains the indices of the sparse patterns
        eigens        = list of size number_of_components, contains the eigenvalues corresponding to sparse patterns
                        given in the order described by variable (pattern)
        load          = array of size s by number_of_components, that contains the loadings i.e sparse eigenvectors
                        corresponding to patterns described by variable (pattern) in same order
        component     = array of size m by number_of_components that contains components, projection of data matrix
                        to sparse loadings given by variable (load)
        variance      = similar to eigens list of size number_of_components, contains the adjusted variance of sparse
                        patterns given in the order described by variable (pattern)
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
            self.A2 = A.T.dot(A)
            self.diag = diag(self.A2)
            self.diag2 = self.diag ** 2

            self.eigenvalues = []
            self.eigenindices = []
            self.eigenvectors = []
            
            self.search_multiplier = 10
            self.args = {'formulation': 'L2var_L0cons', 'dataname': 'ATandT_Database_of_Faces', 'dataDir': './data/',\
    'resultsDir': './results/', 'seed': 1, 'density_of_SP': 1, 'sparsity': self.s, 'tol': 1e-06, \
        'maxIter': 200, 'numOfTrials': 10, 'stabilityIter': 30, 'incDelta': 0.001}
    
    def set_s(self,s):
        # changes sparsity level
        self.s = s
        self.args["s"] = s
        
    def show(self,ind):
        # utility function to show a submatrix 
        # induced by set "ind"
        print(self.A2[:,ind][ind,:])
        return self.A2[:,ind][ind,:]
    
    def frobenius(self,ind):
        # returns frobenius norm of a submatrix induced by set "ind"
        return norm(self.A2[:,ind][ind,:])
    
    def all_eigens(self,ind):
        # returns all eigenvalues of a submatrix induced by set "ind"
        return eigvalsh(self.A2[:,ind][ind,:])

    def restart(self):
        # deflation schemes change the matrix stored, in order to use the algorithms again
        # all related cached information should be restarted with this function
        self.A = self.A0.copy()
        self.A2 = self.A.T.dot(self.A)
        self.diag = diag(self.A2)
        self.diag2 = self.diag ** 2
        
    def deflation_sparse_vector(self,svector,indices):        
        # deflation using sparse vector "svector" and its coordinates "indices"
        dual_vector = self.A[:,indices].dot(svector)
        self.A[:,indices] = self.A[:,indices] - dual_vector.dot(svector.T)
        self.A2[:,indices] = self.A.T.dot(self.A[:,indices])
        self.A2[indices,:] = self.A2[:,indices].T
        self.diag = diag(self.A2)
        self.diag2 = self.diag ** 2
    
    def eigen_upperbound(self,ind):
        # returns the dominant eigenvalue
        # if logic controls the computation scheme, if set "ind" is small, uses
        # submatrix to compute eigenvalue, otherwise uses A * A.T to compute eigenvalues
        # depending on which one is cheaper
        if len(ind) <= self.m:
            dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
            return dominant_eigen_value[0]
        else:
            dominant_eigen_value = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
            return dominant_eigen_value[0]
    
    def eigen_pair(self,ind):
        # returns the dominant eigenpair
        # if logic controls the computation scheme, if set "ind" is small, uses
        # submatrix to compute eigenvalue, otherwise uses A * A.T to compute eigenvalues
        # depending on which one is cheaper
        if len(ind) <= self.m:
            eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
            return eigenvalue,eigenvector
        else:
            eigenvalue,dualeigenvector = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
            eigenvector = dualeigenvector.T.dot(self.A[:,ind])
            eigenvector = eigenvector/norm(eigenvector)

            return eigenvalue,eigenvector.T

    def eigen_pair0(self,ind):
        # returns eigenvalue, eigenvector, and 0 padded eigenvector
        if len(ind) <= self.m:
            eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
        else:
            eigenvalue,dualeigenvector = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
            eigenvector = dualeigenvector.T.dot(self.A[:,ind])
            eigenvector = eigenvector/norm(eigenvector)
            eigenvector = eigenvector.T
        eigenvector_padded = zeros([self.n,1])
        eigenvector_padded[ind] = eigenvector
        return eigenvalue,eigenvector,eigenvector_padded
    
    """
    Following functions;
    (Algorithm name + Algorithm step + whether iterative or recursive)
    GCW1
    GCW2
    GCW1_iterative
    GCW2_iterative
    
    PCW1
    PCW2
    PCW1_iterative
    PCW2_iterative
    
    implementations of algorithms described by Beck & Vaisbourd 2016, for more information 
    check their paper titled "The Sparse Principal Component Analysis Problem:
    Optimality Conditions and Algorithms"
    
    greedy_forward
    greedy_forward_stationary
    greedy_forward_efficient
    
    these algorithms are equivalent to first step of PCW1 described in the same paper. Though 
    this algorithm is initially investigated by Moghaddam et al 2006.
    """
    
    def GCW1(self):
        two_indices = argmax(abs(self.A2)-diag(self.diag),axis = None)
        index1 = two_indices % self.n 
        index2 = two_indices // self.n
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
        two_indices = argmax(abs(self.A2)-diag(self.diag),axis = None)
        index1 = two_indices % self.n 
        index2 = two_indices // self.n
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
        two_indices = argmax(abs(self.A2)-diag(self.diag),axis = None)
        index1 = two_indices % self.n 
        index2 = two_indices // self.n
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
            
        return P,C,best_value

    def greedy_forward_stationary(self):
        two_indices = argmax(abs(self.A2)-diag(self.diag),axis = None)
        index1 = two_indices % self.n 
        index2 = two_indices // self.n
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
        two_indices = argmax(abs(self.A2)-diag(self.diag),axis = None)
        index1 = two_indices % self.n 
        index2 = two_indices // self.n
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
        two_indices = argmax(abs(self.A2)-diag(self.diag),axis = None)
        index1 = two_indices % self.n 
        index2 = two_indices // self.n
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

    # Algorithm 3.1 of the paper
    # Sparse PCA algorithm based on Gerschgorin Discs
    # imposes stationarity condition in the end
    def column_norm_1(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
        R = argsort(absolute(self.A2[top_s,:]),axis = 1)[:,self.n-self.s:].tolist()
        for i in range(int(self.s * self.search_multiplier)):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
                
        val,vec,vec0 = self.eigen_pair0(best_set)
        best_set,best_val = self.EM_mk2(vec0,best_set)
        return best_set,best_val

    # Algorithm 3.1 of the paper
    # Sparse PCA algorithm based on Gerschgorin Discs
    def column_norm_1_old(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
        R = argsort(absolute(self.A2[top_s,:]),axis = 1)[:,self.n-self.s:].tolist()
        for i in range(int(self.s * self.search_multiplier)):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
                
        return best_set,best_val
    
    # Algorithm 3.2 of the paper
    # Sparse PCA algorithm based on l1 norm or correlations
    # imposes stationarity condition in the end
    def correlation_cw(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(absolute(self.A2[current,:][:,possible]),axis = 0) + self.diag[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        val,vec,vec0 = self.eigen_pair0(best_set)
        best_set,best_val = self.EM_mk2(vec0,best_set)
        return best_set,best_val
    
    # Algorithm 3.2 of the paper
    # Sparse PCA algorithm based on l1 norm or correlations
    def correlation_cw_old(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(absolute(self.A2[current,:][:,possible]),axis = 0) + self.diag[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        return best_set,best_val
    
    # Algorithm 3.3 of the paper
    # Sparse PCA algorithm based on frobenius norm
    # imposes stationarity condition in the end
    def frobenius_cw(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(self.A2[current,:][:,possible] ** 2,axis = 0) + self.diag2[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        val,vec,vec0 = self.eigen_pair0(best_set)
        best_set,best_val = self.EM_mk2(vec0,best_set)
        return best_set,best_val

    # Algorithm 3.3 of the paper
    # Sparse PCA algorithm based on frobenius norm
    def frobenius_cw_old(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(self.A2[current,:][:,possible] ** 2,axis = 0) + self.diag2[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current
                
        return best_set,best_val
    
    # Following functions (named with cholesky) are implementations of 
    # Approximate Greedy Search Algorithm by d’Aspremont, Bach and Ghaoui 2008 from paper titled
    # "Optimal Solutions for Sparse Principal Component Analysis"
    
    
    # Approximate Greedy Search Algorithm
    # initialized with variable with the largest column norm
    def cholesky_mk2_old(self):
        
        P = arange(self.n)
        i = argmax(norm(self.A2,axis = 1))
        
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

    # Approximate Greedy Search Algorithm
    # initialized with variable with the largest column norm
    # imposes stationarity condition in the end
    def cholesky_mk2(self):
        
        P = arange(self.n)
        i = argmax(norm(self.A2,axis = 1))
        
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
    
    # Approximate Greedy Search Algorithm
    # initialized with variables with largest column norm
    # for (sparsity_level * search_multiplier) many variables
    def cholesky_mk3(self):
        best_val = 0
        best_set = []
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:int(self.s * self.search_multiplier)]
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

    # Following functions (named with EM) are implementations of Expectation-Maximization
    # Algorithm described in paper "Expectation-Maximization for Sparse and Non-Negative PCA" by
    # Sigg and Buhmann 2008.
    
    
    # Expectation Maximization Algorithm
    # initialized with first PC
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
    
    # Expectation Maximization Algorithm
    # uses given input sparse vector and indices
    # uses indices to avoid unnecessary flop
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
        pattern = list(where(abs(x) > 0)[0])
        value = self.eigen_upperbound(pattern)
        return pattern,value

    # Expectation Maximization Algorithm
    # uses given input vector
    def EM_mk3(self,x):
        t = 0
        diff = 1
        while diff >= 1e-6:
            t += 1
            y = self.A.dot(x)
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
    
    # Following function GPower compiles GPower algorithm described in paper "Generalized Power Method 
    # for Sparse Principal Component Analysis" by Journee et al 2010. Following this paper, equivalent
    # method and code is given in Alternating Maximization is described in "Alternating maximization: unifying
    # framework for 8 sparse PCA formulations and efficient parallel codes" by Richtarik et al 2020.
    # This function uses the software given by second paper. The software is modified and adapted for use
    # with other implemented functions.
    def GPower(self):
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
        # this algorithm is an compiler for previously implemented methods
        # it calls the stationarity imposed version of the algorithms
        # inputs for the algorithm;
        #
        # algo             : choice of algorithm
        # k                : number of components to compute
        # 
        # it returns;
        #
        # loading_patterns : sparse patterns
        # eigen_values     : eigenvalues of the submatrix induced by sparse patterns
        # all_loadings     : sparse eigenvectors that correspond to sparse patterns
        # components       : components produced by loadings
        # variance         : adjusted variance of each component
        if algo not in ["PCW","GCW","GD","FCW","CCW","EM","Path","Path_mk2","GPower","Greedy"]:
            print("bad algortihm choice, choose of one of the following")
            print("GD, CCW, FCW, PCW, GCW, Greedy, EM, Path,Path_mk2, GPower")
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
        elif algo == "GPower":
            algorithm = self.GPower
        elif algo == "Greedy":
            algorithm = self.greedy_forward_stationary
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
        r = qr(components,mode = "r",pivoting = True)
        variance = diag(r[0]) ** 2
        self.restart()
        return loading_patterns,eigen_values,all_loadings,components,variance
    
    def solve_spca(self,P ,C = []):
        """ 
        This is a basic search algorithm to solve SPCA problem exactly, that uses
        sparse column norm approximation of eigenvalues.
        
        This algorithm assumes columns of the original data matrix are scaled, though
        this assumption is only relevant for heuristic reordering of variables
        
        """
        V = zeros(len(P))
        V2 = zeros(len(P))
        R = argsort(abs(self.A2),axis = 1)[P,len(P)-self.s:len(P)-len(C)]
        for i in range(len(P)):
            current_set = R[i,:]
            V[i] = sum(abs(self.A2)[i,current_set])
            V2[i] = sum(abs(self.A2)[current_set,:][:,current_set])

        P = list(map(P.__getitem__,argsort(-1*V)))
        self.order = P
        
        q = PriorityQueue()
        f =  self.eigen_upperbound(P+C)
        lenp = len(P)
        q.put([-1*f,[P,C,len(C),lenp]])
        while q.qsize() > 0:
            [eigen,[P,C,len_c,len_p]] = q.get()
            # print("eigen value",eigen,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if len_c < self.s:
                if  len_c + len_p <= self.s:
                    eigenvalue,eigenvector = self.eigen_pair(C+P)
                    self.eigenvalues.append(eigenvalue[0])
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    break
                else:
                    C1 = C + P[0:1]
                    P1 = P[1:]
                    
                    eigen2 = self.eigen_upperbound(C+P1)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        eigenvalue,eigenvector = self.eigen_pair(C1)
                        q.put([-1*eigenvalue,[eigenvector,C1,len_c1,len_p]])
                        q.put([-1*eigen2,[P1,C,len_c,len_p]])
                    else:
                        q.put([ eigen ,[P1,C1,len_c1,len_p]])
                        q.put([-1*eigen2 ,[P1,C,len_c,len_p]])
            else:
                self.eigenvalues.append(-1*eigen[0])
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                break

            