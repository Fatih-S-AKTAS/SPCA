from scipy.linalg import qr,svd,norm,eigvalsh,det,ldl
from numpy import isnan,Inf,float64,ndarray,shape,std,copy,argmax,zeros,argsort,where,random,diag,sum,eye,\
    hstack,stack,product,sign,sqrt,arange,delete
from scipy.sparse.linalg import eigsh
from scipy.special import comb
from queue import PriorityQueue
from guppy import hpy
import time
import os
import sys


class SPCA:
    def __init__(self,A,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x m covariance matrix
        s       = sparsity level
        
        Please contact selim.aktas@ug.bilkent.edu.tr for any bugs, errors and recommendations.
        """
        
        for i in range(len(A)):
            for j in range(len(A[0])):
                if isnan(A[i,j]) or abs(A[i,j]) == Inf:
                    print("Matrix A has NAN or Inf values, it will give linear algebra errors")
                    break
        for i in range(len(A)):
            for j in range(len(A[0])):
                if type(A[i,j]) != float64:
                    print("Matrix A should be registered as float64, otherwise computations will be wrong")
                    
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
            """ initializing remaining queue size after the algorithm finishes """
            self.out = 1
            """ initializing the output choice of algorithm """
            self.original_stdout = sys.stdout
            """  saving original stdout, after manipulating it for changing values of out, we should be
            able to recover it """
            
            self.A = A
            self.A2 = A.T.dot(A)
            
            self.abs_A2 = abs(self.A2)
            self.squared_A2 = self.A2 ** 2
            
            self.abs_A2s = self.abs_A2 - self.m * eye(self.n)
            
            """ QR with pivoting to find columns that are large in norm and orthogonal, construct the covarince matrix accordingly"""
            
            self.tablelookup = {}
            """ lookup tables for solving all subset problem """
            self.many = 1
            """ initializing the number of solutions to be found"""
            self.eigenvalues = []
            self.eigenindices = []
            self.eigenvectors = []
            """ initializing the arrays of solution parameters """
            self.memory = 0
            """ initializing memory usage of the algorithm """
            self.cpu = 0
            """ initializing cpu usage of the algorithm """
            self.best = -Inf

    
    def show(self,ind):
        print(self.A2[:,ind][ind,:])
        
    def frobenius(self,ind):
        return norm(self.A2[:,ind][ind,:])
    
    def all_eigens(self,ind):
        return eigvalsh(self.A2[:,ind][ind,:])
    
    def eigen_upperbound(self,ind):
        dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-2,return_eigenvectors = False)
        return dominant_eigen_value[0]
    
    def eigen_pair(self,ind):
        eigenvalue,eigenvector = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-2,return_eigenvectors = True)
        return eigenvalue,eigenvector
    
    def eigen_upperboundl(self,ind):
        check = str(ind)
        if check in self.tablelookup:
            return self.tablelookup[check]
        dominant_eigen_value = eigsh(self.A2[:,ind][ind,:],k = 1, which = "LA", tol = 1e-2,return_eigenvectors = False)
        self.tablelookup_nnls[check] = dominant_eigen_value[0]
        return dominant_eigen_value[0]


    def SPI(self,ind,x = False):
        """ Sparse Power Iteration """
        l = len(ind)
        A = self.A[:,ind]
        u,d,v = svd(A)
        step = 1/d[0] ** 2
        if x is False:
            x1 = zeros(l)
        else:
            x1 = copy(x)    
        diff = 1
        t = 0
        x2 = x1
        while diff >= 1e-6 and t <= 400:
            # print("dif",diff,"time",t)
            t += 1
            grad = self.rtilde[:,ind].T.dot(self.rtilde[:,ind].dot(x1) - self.qb[:,0])
            x1 = x1 - step * grad
            
            si = argsort(x1)
            x1[si[:-self.s]] = 0
            diff = norm(x2-x1,2)
            x2 = x1

        return x1
    
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
        Rc = []
        RC = []
        RF = []
        best_val = 0
        for i in range(self.n):
            current = sorted(R[i])
            R[i] = current.copy()
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
            R[i].append(val)
            Rc.append(current + [sum(self.abs_A2[i,current])])
            RC.append(current + [sum(self.abs_A2[current,:][:,current])])
            RF.append(current + [sum(self.squared_A2[current,:][:,current])])
         
        self.R = R
        self.Rc = Rc
        self.RC = RC
        self.RF = RF
        self.Rval = best_val
        
    def correlation_cw(self):
        R = []
        Rc = []
        RC = []
        RF = []
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
            R.append(current + [val])
            Rc.append(current + [sum(self.abs_A2[i,current])])
            RC.append(current + [sum(self.abs_A2[current,:][:,current])])
            RF.append(current + [sum(self.squared_A2[current,:][:,current])])

        self.R2 = R
        self.R2c = Rc
        self.R2C = RC
        self.R2F = RF
        self.R2val = best_val
        
    def column_norm_2(self):
        R = argsort(self.squared_A2,axis = 1)[:,self.n-self.s:].tolist()
        Rc = R.copy()
        RC = R.copy()
        RF = R.copy()
        best_val = 0
        for i in range(self.n):
            current = sorted(R[i])
            R[i] = current.copy()
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
            R[i].append(val)
            Rc.append(current + [sum(self.abs_A2[i,current])])
            RC.append(current + [sum(self.abs_A2[current,:][:,current])])
            RF.append(current + [sum(self.squared_A2[current,:][:,current])])
         
        self.R3 = R
        self.R3c = Rc
        self.R3C = RC
        self.R3F = RF
        self.R3val = best_val
        
    def frobenius_cw(self):
        R = []
        Rc = []
        RC = []
        RF = []
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
            R.append(current + [val])
            Rc.append(current + [sum(self.abs_A2[i,current])])
            RC.append(current + [sum(self.abs_A2[current,:][:,current])])
            RF.append(current + [sum(self.squared_A2[current,:][:,current])])

        self.R4 = R
        self.R4c = Rc
        self.R4C = RC
        self.R4F = RF
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
        
        return current

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
        
        return current    

        
    def EM(self):
        w = self.eigen_pair(list(range(self.n)))
        y = self.A.dot(w)
        return w
    
    def cassini_oval(self):
        R = []
        Rc = []
        RC = []
        RF = []
        best_val = 0
        for i in range(self.n):
            current = [i]
            for j in range(self.s-1):
                k = 0
                chosen = 0
                c_oval = 0
                while k < self.n:
                    if k in current:
                        k += 1
                        continue
                    oval = product(sorted(sum(self.abs_A2[current + [k],:][:,current + [k]],axis = 1))[-2:])
                    if oval > c_oval:
                        c_oval = oval
                        chosen = k
                    k += 1
                current = current + [chosen]
            current = sorted(current)
            val = self.eigen_upperbound(current)
            if val > best_val:
                best_val = val
            R.append(current + [val])
            Rc.append(current + [sum(self.abs_A2[i,current])])
            RC.append(current + [sum(self.abs_A2[current,:][:,current])])
            RF.append(current + [sum(self.squared_A2[current,:][:,current])])

        self.R6 = R
        self.R6c = Rc
        self.R6C = RC
        self.R6F = RF
        self.R6val = best_val

    def solve_spca_mk6(self,P ,C = []):
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
            
        # self.V = V
        # self.V2 = V2
        # self.R = R
        # listR = []
        # for i in range(self.n):
        #     listR.append(sorted(list(self.R[i,:])))
        # self.listR = listR
        # R2 = []
        # R2f = []
        # R2e = []
        # for i in range(len(P)):
        #     current = [i]
        #     for j in range(self.s-len(C)-1):
        #         argsortk = argsort(sum(self.abs_A2[current,:],axis = 0))
        #         current.append(argsortk[-len(current)-1])
        #     currentf = sorted(current)
        #     currente = currentf[:]
        #     currentf.append(self.frobenius(current))
        #     currente.append(self.eigen_upperbound(current))

        #     R2.append(sorted(current))
        #     R2f.append(currentf)
        #     R2e.append(currente)
        # self.R2 = R2
        # self.R2f = R2f
        # self.R2e = R2e
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

            