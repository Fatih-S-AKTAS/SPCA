from numpy import Inf,ndarray,shape,argmax,zeros,argsort,where,diag,sum,\
    arange,delete,absolute,asarray_chkfinite,asarray,concatenate,unique,sort,isnan,NaN
from scipy.sparse.linalg import eigsh
from scipy.sparse import random as scipy_random
from scipy.linalg import qr
from scipy.io import savemat
from numpy.linalg import norm,eigvalsh

from queue import PriorityQueue

from AM_functions import CustomDistribution
from AM_wrapper import run_formulation

import matlab
import matlab.engine


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

        algorithm     : choice of spca algorithm
        components    : number of components to compute
        
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

            self.eigenvalues = []
            self.eigenindices = []
            self.eigenvectors = []
            
            self.search = min(200,self.n)
            self.args = {'formulation': 'L2var_L0cons', 'dataname': 'ATandT_Database_of_Faces', 'dataDir': './data/',\
    'resultsDir': './results/', 'seed': 1, 'density_of_SP': 1, 'sparsity': self.s, 'tol': 1e-06, \
        'maxIter': 200, 'numOfTrials': 10, 'stabilityIter': 30, 'incDelta': 0.001}

            self.mat_eng = matlab.engine.start_matlab()
    
    def PCW(self):
        x,f = self.mat_eng.cwPCA(matlab.double(self.A.tolist()),self.s,"type","PCW","mat_type","DMat",nargout = 2)
        x = asarray(x)
        return where(asarray(x) != 0)[0],x.T.dot(self.A2.dot(x))
    def GCW(self):
        x,f = self.mat_eng.cwPCA(matlab.double(self.A.tolist()),self.s,"type","GCW","mat_type","DMat",nargout = 2)
        x = asarray(x)
        return where(asarray(x) != 0)[0],x.T.dot(self.A2.dot(x))

    def PCW_memory(self):
        x,f = self.mat_eng.cwPCA_memory(self.s,"type","PCW","mat_type","DMat",nargout = 2)
        x = asarray(x)
        return where(asarray(x) != 0)[0],x.T.dot(self.A2.dot(x))
    def GCW_memory(self):
        x,f = self.mat_eng.cwPCA_memory(self.s,"type","GCW","mat_type","DMat",nargout = 2)
        x = asarray(x)
        return where(asarray(x) != 0)[0],x.T.dot(self.A2.dot(x))
    
    def gpbbls(self):
        x,f = self.mat_eng.gpbbls(matlab.double(self.A.tolist()),self.s,nargout = 2)
        # x = asarray(x)
        return where(asarray(x) != 0)[0],f
    
    def gpbbls_memory(self):
        x,f = self.mat_eng.gpbbls_memory(self.s,nargout = 2)
        # x = asarray(x)
        return where(asarray(x) != 0)[0],f
    
    def set_sparsity(self,s):
        # changes sparsity level
        self.s = s
        self.args["sparsity"] = s
        
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
        mdic = {"data": self.A0, "label": "experiment"}
        savemat("data_matrix.mat",mdic)
        
    def deflation_sparse_vector(self,svector,indices):        
        # deflation using sparse vector "svector" and its coordinates "indices"
        dual_vector = self.A[:,indices].dot(svector)
        self.A[:,indices] = self.A[:,indices] - dual_vector.dot(svector.T)
        mdic = {"data": self.A, "label": "experiment"}
        savemat("data_matrix.mat",mdic)
        self.A2[:,indices] = self.A.T.dot(self.A[:,indices])
        self.A2[indices,:] = self.A2[:,indices].T
    
    def eigen_upperbound(self,ind):
        # returns the dominant eigenvalue
        # if logic controls the computation scheme, if set "ind" is small, uses
        # submatrix to compute eigenvalue, otherwise uses A * A.T to compute eigenvalues
        # depending on which one is cheaper
        if len(ind) <= self.m:
            dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-8,return_eigenvectors = False)
            return dominant_eigen_value[0]
        else:
            dominant_eigen_value = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-8,return_eigenvectors = False)
            return dominant_eigen_value[0]
    
    def eigen_pair(self,ind):
        # returns the dominant eigenpair
        # if logic controls the computation scheme, if set "ind" is small, uses
        # submatrix to compute eigenvalue, otherwise uses A * A.T to compute eigenvalues
        # depending on which one is cheaper
        if len(ind) <= self.m:
            eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-8,return_eigenvectors = True)
            return eigenvalue,eigenvector
        else:
            eigenvalue,dualeigenvector = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-8,return_eigenvectors = True)
            eigenvector = dualeigenvector.T.dot(self.A[:,ind])
            eigenvector = eigenvector/norm(eigenvector)

            return eigenvalue,eigenvector.T

    def eigen_pair0(self,ind):
        # returns eigenvalue, eigenvector, and 0 padded eigenvector
        if len(ind) <= self.m:
            eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-8,return_eigenvectors = True)
        else:
            eigenvalue,dualeigenvector = eigsh(self.A[:,ind].dot(self.A[:,ind].T),k = 1, which = "LA", tol = 1e-8,return_eigenvectors = True)
            eigenvector = dualeigenvector.T.dot(self.A[:,ind])
            eigenvector = eigenvector/norm(eigenvector)
            eigenvector = eigenvector.T
        eigenvector_padded = zeros([self.n,1])
        eigenvector_padded[ind] = eigenvector
        return eigenvalue,eigenvector,eigenvector_padded

    # Algorithm 3.1 of the paper
    # Sparse PCA algorithm based on Gerschgorin Discs
    # imposes stationarity condition in the end
    def column_norm_1(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:self.search]
        R = argsort(absolute(self.A2[top_s,:]),axis = 1)[:,self.n-self.s:]
        R = unique(sort(R),axis = 0)
        for i in range(R.shape[0]):
            val,vec,vec0 = self.eigen_pair0(R[i])
            new_set,new_val = self.EM_mk2(vec0,R[i])
            if new_val > best_val:
                best_val = new_val
                best_set = new_set
    
        return best_set,best_val
    
    # Algorithm 3.1 of the paper
    # Sparse PCA algorithm based on Gerschgorin Discs
    def column_norm_1_old(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s = argsorted[:self.search]
        R = argsort(absolute(self.A2[top_s,:]),axis = 1)[:,self.n-self.s:].tolist()
        for i in range(self.search):
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
        argsorted = argsort(-1 * norm(self.A2,axis = 1,keepdims = True),axis = 0)
        top_s = argsorted[:self.search]
        rang = arange(self.search).reshape([self.search,1])
        for j in range(self.s-1):
            S = absolute(self.A2[top_s,:]).sum(axis = 1)
            S[rang,top_s] = 0
            new_column = S.argmax(axis = 1).reshape([self.search,1])
            top_s = concatenate((top_s,new_column),axis = 1)
        R = unique(sort(top_s),axis = 0)
        for i in range(R.shape[0]):
            val,vec,vec0 = self.eigen_pair0(R[i])
            new_set,new_val = self.EM_mk2(vec0,R[i])
            if new_val > best_val:
                best_val = new_val
                best_set = new_set
                
        return best_set,best_val
    
    # Algorithm 3.2 of the paper
    # Sparse PCA algorithm based on l1 norm or correlations
    def correlation_cw_old(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1,keepdims = True),axis = 0)
        top_s = argsorted[:self.search]
        rang = arange(self.search).reshape([self.search,1])
        for j in range(self.s-1):
            S = absolute(self.A2[top_s,:]).sum(axis = 1)
            S[rang,top_s] = 0
            new_column = S.argmax(axis = 1).reshape([self.search,1])
            top_s = concatenate((top_s,new_column),axis = 1)
        R = unique(sort(top_s),axis = 0)
        for i in range(R.shape[0]):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
                
        return best_set,best_val
    
    # Algorithm 3.3 of the paper
    # Sparse PCA algorithm based on frobenius norm
    # imposes stationarity condition in the end
    def frobenius_cw(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1,keepdims = True),axis = 0)
        top_s = argsorted[:self.search]
        rang = arange(self.search).reshape([self.search,1])
        for j in range(self.s-1):
            S = (self.A2[top_s,:] ** 2).sum(axis = 1)
            S[rang,top_s] = 0
            new_column = S.argmax(axis = 1).reshape([self.search,1])
            top_s = concatenate((top_s,new_column),axis = 1)
        R = unique(sort(top_s),axis = 0)
        for i in range(R.shape[0]):
            val,vec,vec0 = self.eigen_pair0(R[i])
            new_set,new_val = self.EM_mk2(vec0,R[i])
            if new_val > best_val:
                best_val = new_val
                best_set = new_set
                
        return best_set,best_val
    
    # Algorithm 3.3 of the paper
    # Sparse PCA algorithm based on frobenius norm
    def frobenius_cw_old(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1,keepdims = True),axis = 0)
        top_s = argsorted[:self.search]
        rang = arange(self.search).reshape([self.search,1])
        for j in range(self.s-1):
            S = (self.A2[top_s,:] ** 2).sum(axis = 1)
            S[rang,top_s] = 0
            new_column = S.argmax(axis = 1).reshape([self.search,1])
            top_s = concatenate((top_s,new_column),axis = 1)
        R = unique(sort(top_s),axis = 0)
        for i in range(R.shape[0]):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
                
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
        top_s = argsorted[:self.search]
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
        if algo not in ["PCW","GCW","GD","FCW","CCW","EM","Path","Path_mk2","GPower","gpbbls",\
                        "gpbbls_memory","PCW_memory","GCW_memory"]:
            print("bad algortihm choice, choose of one of the following")
            print("GD, CCW, FCW, PCW, GCW, Greedy, EM, Path,Path_mk2, GPower, gpbbls")
        elif algo == "PCW":
            algorithm = self.PCW
        elif algo == "GCW":
            algorithm = self.GCW
        elif algo == "PCW_memory":
            algorithm = self.PCW_memory
        elif algo == "GCW_memory":
            algorithm = self.GCW_memory
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
        elif algo == "gpbbls":
            algorithm = self.gpbbls
        elif algo == "gpbbls_memory":
            algorithm = self.gpbbls_memory
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

    def find_component_var_spar(self,algo,k,s_list):
        # see above algorithm for details
        #
        # this algorithm solves the problem of varying sparsify levels for components
        #
        # s_list             : a list of size k that specifies the sparsify levels for the components
        #                      
        if len(s_list) != k:
            print("number of components and number of specified sparsity levels should be equal")            
        if algo not in ["PCW","GCW","GD","FCW","CCW","EM","Path","Path_mk2","GPower","gpbbls"]:
            print("bad algortihm choice, choose of one of the following")
            print("GD, CCW, FCW, PCW, GCW, Greedy, EM, Path,Path_mk2, GPower, gpbbls")
        elif algo == "PCW":
            algorithm = self.PCW
        elif algo == "GCW":
            algorithm = self.GCW
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
        elif algo == "gpbbls":
            algorithm = self.gpbbls
        all_loadings = zeros([max(s_list),k])
        components = zeros([self.m,k])
        loading_patterns = []
        eigen_values = []
        for i in range(k):
            self.s = s_list[i]
            self.args["sparsity"] = self.s
            self.search_multiplier = min(self.n/self.s,10)
            pattern,value = algorithm()
            eigenvalue,eigenvector = self.eigen_pair(pattern)
            loading_patterns.append(pattern)
            eigen_values.append(value)
            all_loadings[:self.s,i] = eigenvector[:,0]
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

            