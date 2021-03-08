from scipy.linalg import ldl
from numpy import isnan,Inf,float64,ndarray,shape,std,copy,argmax,zeros,argsort,where,random,diag,sum,eye,\
    hstack,stack,product,sign,sqrt,arange,delete,fill_diagonal
from scipy.sparse.linalg import eigsh
from numpy.linalg import qr,norm,eigvalsh,cholesky
from queue import PriorityQueue
import time


class SPCA:
    def __init__(self,A,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x n data matrix
        s       = sparsity level
        
        Please contact selim.aktas@ug.bilkent.edu.tr for any bugs, errors and recommendations.
        """
        
        # for i in range(len(A)):
        #     for j in range(len(A[0])):
        #         if isnan(A[i,j]) or abs(A[i,j]) == Inf:
        #             print("Matrix A has NAN or Inf values, it will give linear algebra errors")
        #             break
        # for i in range(len(A)):
        #     for j in range(len(A[0])):
        #         if type(A[i,j]) != float64:
        #             print("Matrix A should be registered as float64, otherwise computations will be wrong")
                    
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
            
            self.abs_A2 = abs(self.A2)
            self.squared_A2 = self.A2 ** 2
            self.abs_A2s = self.abs_A2.copy()
            fill_diagonal(self.abs_A2s,0)
 
            
            self.tablelookup = {}
            """ lookup table to store and not solve for the same sparsity pattern multiple times"""

            self.eigenvalues = []
            self.eigenindices = []
            self.eigenvectors = []
            
            self.search_multiplier = 10

    def show(self,ind):
        print(self.A2[:,ind][ind,:])
        return self.A2[:,ind][ind,:]
    
    def frobenius(self,ind):
        return norm(self.A2[:,ind][ind,:])
    
    def all_eigens(self,ind):
        return eigvalsh(self.A2[:,ind][ind,:])
    
    def restart(self):
        self.A = self.A0.copy()
        self.A2 = self.A.T.dot(self.A)
        self.diag = diag(self.A2)
        self.diag2 = self.diag ** 2
        
        self.abs_A2 = abs(self.A2)
        self.squared_A2 = self.A2 ** 2
        self.abs_A2s = self.abs_A2.copy()
        fill_diagonal(self.abs_A2s,0)
        
    def deflation_sparse_vector(self,svector,indices):        
        dual_vector = self.A[:,indices].dot(svector)
        self.A[:,indices] = self.A[:,indices] - dual_vector.dot(svector.T)
        self.A2[:,indices] = self.A.T.dot(self.A[:,indices])
        self.A2[indices,:] = self.A2[:,indices].T
        self.diag = diag(self.A2)
        self.diag2 = self.diag ** 2
        self.abs_A2 = abs(self.A2)
        self.squared_A2 = self.A2 ** 2
        self.abs_A2s = self.abs_A2.copy()
        fill_diagonal(self.abs_A2s,0)
    
    def eigen_upperbound(self,ind):
        dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
        return dominant_eigen_value[0]
    
    def eigen_pair(self,ind):
        eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
        return eigenvalue,eigenvector
    
    def eigen_pair0(self,ind):
        eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
        eigenvector_padded = zeros([self.n,1])
        eigenvector_padded[ind] = eigenvector
        return eigenvalue,eigenvector,eigenvector_padded
    
    def SPI(self):
        """ Sparse Power Iteration """
        val,x1 = self.eigen_pair(list(range(self.n)))
        x1t = zeros([self.n,1])
        t = 0
        diff = 1
        while t <= 1000 and diff > 1e-4:
            t += 1
            x0 = x1t
            x1 = self.A2.dot(x1)
            x1 = x1/norm(x1)
            xi = argsort(abs(x1[:,0]))
            x1t = zeros([self.n,1])
            x1t[xi[:self.s],0] = x1[xi[:self.s],0]
            diff = norm(x0-x1t)
        pattern = where(abs(x1t) > 0)[0]
        value = self.eigen_upperbound(pattern)
        return pattern,value
    
    def GCW1(self):
        two_indices = argmax(self.abs_A2s,axis = None)
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
        two_indices = argmax(self.abs_A2s,axis = None)
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
           
    def PCW1(self):
        two_indices = argmax(self.abs_A2s,axis = None)
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
        two_indices = argmax(self.abs_A2s,axis = None)
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

    def column_norm_1(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        # top_s_2 = argsorted[:self.s * self.search_multiplier]
        top_s_2 = argsorted[:int(self.s * self.search_multiplier)]
        R = argsort(self.abs_A2[top_s_2,:],axis = 1)[:,self.n-self.s:].tolist()
        for i in range(int(self.s * self.search_multiplier)):
            val = self.eigen_upperbound(R[i])
            if val > best_val:
                best_val = val
                best_set = R[i]
         
        return best_set,best_val

    def correlation_cw(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        # top_s_2 = argsorted[:self.s * self.search_multiplier]
        top_s_2 = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s_2:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(self.abs_A2[current,:][:,possible],axis = 0) + self.diag[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current

        return best_set,best_val
        
    def frobenius_cw(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        # top_s_2 = argsorted[:self.s * self.search_multiplier]
        top_s_2 = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s_2:
            possible = list(range(self.n))
            possible.pop(i)
            current = [i]
            for j in range(self.s-1):
                k = argmax(sum(self.squared_A2[current,:][:,possible],axis = 0) + self.diag2[possible])
                current.append(possible[k])
                del possible[k]
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current

        return best_set,best_val
        
    def cholesky(self):
        l,d,p = ldl(self.A2)
        
        P = arange(self.n)
        i = argmax(norm(self.A2,axis = 1))
        
        current = [i]
        P = delete(P,i)
        
        x = l[[i],:]/norm(l[i,:])
        vals = x.dot(l.T[:,P]) ** 2
        new_index = argmax(vals)
        current.append(P[new_index])
        P = delete(P,new_index)
        
        for i in range(self.s - 2):
            value,vector = self.eigen_pair(current)
            x = l.T[:,current].dot(vector)
            x = x/norm(x)
            vals = x.T.dot(l.T[:,P]) ** 2
            new_index = argmax(vals)
            current.append(P[new_index])
            P = delete(P,new_index)
        
        return current,self.eigen_upperbound(current)

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
        
        return current,self.eigen_upperbound(current)

    def cholesky_mk3(self):
        best_val = 0
        best_set = []
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s_2 = argsorted[:int(self.s * self.search_multiplier)]
        for i in top_s_2:
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
        t = 0
        diff = 1
        while diff >= 1e-4:
            t += 1
            y = self.A.dot(wt)
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

    def find_component(self,algo,k):
        if algo not in ["PCW","GCW","GD","FCW","CCW","EM","Path"]:
            print("bad algortihm choice, choose of one of the following")
            print("GD, CCW, FCW, PCW, GCW, EM, Path")
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
    
    def solve_spca(self,P ,C = []):
        """ 

        sparse principal component with 
        
        Sparse Column norm approximation of eigenvalues 
        
        This algorithm assumes columns of the original data matrix are scaled
        
        """
        V = zeros(len(P))
        V2 = zeros(len(P))
        R = argsort(self.abs_A2,axis = 1)[P,len(P)-self.s:len(P)-len(C)]
        for i in range(len(P)):
            current_set = R[i,:]
            V[i] = sum(self.abs_A2[i,current_set])
            V2[i] = sum(self.abs_A2[current_set,:][:,current_set])

        P = list(map(P.__getitem__,argsort(-1*V)))
        self.order = P
            
        # """ reordering indices in P according to column norms"""
        
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.eigen_upperbound(P+C)
        lenp = len(P)
        q.put([-1*f,[P,C,len(C),lenp]])
        count_best = 0
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [eigen,[P,C,len_c,len_p]] = q.get()
            # print("eigen value",eigen,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if len_c < self.s:
                if  len_c + len_p <= self.s:
                    eigenvalue,eigenvector = self.eigen_pair(C+P)
                    self.eigenvalues.append(eigenvalue[0])
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    break
                else:
                    C1 = C + P[0:1]
                    P1 = P[1:]
                    
                    eigen2 = self.eigen_upperbound(C+P1)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        eigenvalue,eigenvector = self.eigen_pair(C1)
                        q.put([-1*eigenvalue,[eigenvector,C1,len_c1,len_p]])
                        q.put([-1*eigen2,[P1,C,len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([ eigen ,[P1,C1,len_c1,len_p]])
                        q.put([-1*eigen2 ,[P1,C,len_c,len_p]])
            else:
                self.eigenvalues.append(-1*eigen[0])
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                break

            