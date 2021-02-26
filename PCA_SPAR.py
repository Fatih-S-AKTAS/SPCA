from scipy.linalg import norm,eigvalsh,ldl
from numpy import isnan,Inf,float64,ndarray,shape,std,copy,argmax,zeros,argsort,where,random,diag,sum,eye,\
    hstack,stack,product,sign,sqrt,arange,delete
from scipy.sparse.linalg import eigsh
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
            
            self.A = A
            self.A2 = A.T.dot(A)
            self.abs_A2 = abs(self.A2)
            self.squared_A2 = self.A2 ** 2
            self.abs_A2s = self.abs_A2 - self.m * eye(self.n)
            
            self.tablelookup = {}
            """ lookup table to store and not solve for the same sparsity pattern multiple times"""
            self.many = 1

            self.eigenvalues = []
            self.eigenindices = []
            self.eigenvectors = []
            
            self.search_multiplier = 10
            self.best = -Inf
            self.look_up = 0

    def show(self,ind):
        print(self.A2[:,ind][ind,:])
        return self.A2[:,ind][ind,:]
    
    def frobenius(self,ind):
        return norm(self.A2[:,ind][ind,:])
    
    def all_eigens(self,ind):
        return eigvalsh(self.A2[:,ind][ind,:])
    
    def eigen_upperbound(self,ind):
        dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
        return dominant_eigen_value[0]
    
    def eigen_pair(self,ind):
        eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = True)
        return eigenvalue,eigenvector
    
    def eigen_upperboundl(self,ind):
        check = str(ind)
        if check in self.tablelookup:
            self.look_up += 1
            return self.tablelookup[check]
        dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-3,return_eigenvectors = False)
        self.tablelookup[check] = dominant_eigen_value[0]
        return dominant_eigen_value[0]


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
    
    def GCW1(self,P,C = []):
        if C == []:
            two_indices = argmax(self.abs_A2s[P,:][:,P],axis = None)
            index1 = two_indices % len(P) 
            index2 = two_indices // len(P)
            C = [index1,index2]
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

    def GCW1_iterative(self,P,C = []):
        if C == []:
            two_indices = argmax(self.abs_A2s[P,:][:,P],axis = None)
            index1 = two_indices % len(P) 
            index2 = two_indices // len(P)
            C = [index1,index2]
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
           
    def PCW1(self,P,C = []):
        if C == []:
            two_indices = argmax(self.abs_A2s[P,:][:,P],axis = None)
            index1 = two_indices % len(P) 
            index2 = two_indices // len(P)
            C = [index1,index2]
            P.remove(index1)
            P.remove(index2)
            best_value,best_vector = self.eigen_pair(C)
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

    def PCW1_iterative(self,P,C = []):
        if C == []:
            two_indices = argmax(self.abs_A2s[P,:][:,P],axis = None)
            index1 = two_indices % len(P) 
            index2 = two_indices // len(P)
            C = [index1,index2]
            P.remove(index1)
            P.remove(index2)
            best_value,best_vector = self.eigen_pair(C)
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
        R = argsort(self.abs_A2,axis = 1)[:,self.n-self.s:].tolist()
        best_val = 0
        for i in range(self.n):
            current = sorted(R[i])
            R[i] = current.copy()
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
            R[i].append(val)
         
        self.R = R
        self.Rval = best_val
        
    def column_norm_1l(self):
        R = argsort(self.abs_A2,axis = 1)[:,self.n-self.s:].tolist()
        best_val = 0
        for i in range(self.n):
            current = R[i]
            R[i] = current.copy()
            val = self.eigen_upperboundl(current)
            if val > best_val:
                best_val = val
            R[i].append(val)
         
        self.R = R
        self.Rval = best_val

    def column_norm_1_fast(self):
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s_2 = argsorted[:self.s * self.search_multiplier]
        R = argsort(self.abs_A2[top_s_2,:],axis = 1)[:,self.n-self.s:].tolist()
        for i in range(self.s * self.search_multiplier):
            current = sorted(R[i])
            R[i] = current.copy()
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
            R[i].append(val)
         
        self.R = R
        self.Rval = best_val
        
    def correlation_cw(self):
        best_set = []
        best_val = 0
        for i in range(self.n):
            current = [i]
            for j in range(self.s-1):
                argsortk = argsort(sum(self.abs_A2[current,:],axis = 0))[-len(current)-1:][::-1]
                k = 0
                found = False
                while not found and k < len(current) + 1:
                    if argsortk[k] not in current:
                        current.append(argsortk[k])
                        found = True
                    k += 1
            current = sorted(current)
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current

        self.R2 = best_set
        self.R2val = best_val
        
    def frobenius_cw(self):
        best_set = []
        best_val = 0
        for i in range(self.n):
            current = [i]
            for j in range(self.s-1):
                argsortk = argsort(sum(self.squared_A2[current,:],axis = 0))[-len(current)-1:]
                k = 0
                found = False
                while not found and k < len(current) + 1:
                    if argsortk[k] not in current:
                        current.append(argsortk[k])
                        found = True
                    k += 1
            current = sorted(current)
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current

        self.R4 = best_set
        self.R4val = best_val

    def correlation_cw_fast(self):
        visited = []
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s_2 = argsorted[:self.s * self.search_multiplier]
        for i in top_s_2:
            current = [i]
            for j in range(self.s-1):
                argsortk = argsort(sum(self.abs_A2[current,:],axis = 0))[-len(current)-1:][::-1]
                k = 0
                found = False
                while not found and k < len(current) + 1:
                    if argsortk[k] not in current:
                        current.append(argsortk[k])
                        found = True
                    k += 1
                if current in visited:
                    break
                else:
                    visited.append(current.copy())
            current = sorted(current)
            val = self.eigen_upperboundl(current)
            if val > best_val:
                best_val = val
                best_set = current

        self.R2 = best_set
        self.R2val = best_val
        
    def frobenius_cw_fast(self):
        best_set = []
        best_val = 0
        argsorted = argsort(-1 * norm(self.A2,axis = 1))
        top_s_2 = argsorted[:self.s * self.search_multiplier]
        for i in top_s_2:
            current = [i]
            for j in range(self.s-1):
                argsortk = argsort(sum(self.squared_A2[current,:],axis = 0))[-len(current)-1:]
                k = 0
                found = False
                while not found and k < len(current) + 1:
                    if argsortk[k] not in current:
                        current.append(argsortk[k])
                        found = True
                    k += 1
            current = sorted(current)
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
                best_set = current

        self.R4 = best_set
        self.R4val = best_val
        
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
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        print("first")
                        break
                    """ termination condition of the algorithm """
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
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    print("second")
                    break
                """ termination condition of the algorithm """

            