from scipy.linalg import qr,svd,norm,eigvalsh
from numpy import isnan,Inf,float64,ndarray,shape,std,copy,argmax,zeros,argsort,where,random,diag,sum,eye
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
            self.means = abs(A.mean(axis = 0))
            """ storing the mean values of each independent variable which will be used for 
            selection criteria in solver algortihms"""
            self.sterror = std(A,axis = 0)
            """ storing the standard deviation and variance of each independent variable which will
            be used for selection criteria in solver algortihms"""
            self.rem_qsize = []
            """ initializing remaining queue size after the algorithm finishes """
            self.out = 1
            """ initializing the output choice of algorithm """
            self.original_stdout = sys.stdout
            """  saving original stdout, after manipulating it for changing values of out, we should be
            able to recover it """
            
            q,r,p = qr(A,pivoting = True)
            self.q = q
            self.r = r
            self.p = p
            
            self.A2 = A.T.dot(A)
            self.eigenvalues_A2 = eigvalsh(self.A2)
            
            self.AA2 = abs(self.A2)
            self.AA2s = self.AA2 - self.m*eye(self.n)
            
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

    def SPI(self,ind,x = False):
        """ Sparse Power Iteration """
        l = len(ind)
        A = self.rtilde[:,ind]
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

    def qr_pivoting(self,ind):
        q,r,p = qr(self.r[:,ind],pivoting = True)
        return list(map(ind.__getitem__,p))
    
    def GCW1(self,P,C = []):
        if C == []:
            two_indices = argmax(self.AA2s[P,:][:,P],axis = None)
            index1 = two_indices % len(P) 
            index2 = two_indices // len(P)
            C = [index1,index2]
            for i in C:   
                P.remove(i)
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
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return self.GCW2(best_value,P,C)
            
    def GCW2(self,best_value,P,C = []):
        print("recursion")
        len_p = len(P)
        len_c = len(C)
        for i in range(len_c):
            for j in range(len_p):
                # print("i",i,"j",j)
                C_temp = C + [P[j]]
                C_temp.remove(C[i])
                value,vector = self.eigen_pair(C_temp)
                if value > best_value:
                    P += [C[i]]
                    P.remove(P[j])
                    return self.GCW2(value,P,C_temp)
        return [best_value,C]
                
    def PCW1(self,P,C = []):
        if C == []:
            two_indices = argmax(self.AA2s[P,:][:,P],axis = None)
            index1 = two_indices % len(P) 
            index2 = two_indices // len(P)
            C = [index1,index2]
            for i in C:   
                P.remove(i)
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
                    best_index = i
            C = C + [P[best_index]]
            P.remove(P[best_index])
        return self.GCW2(best_value,P,C)
            
    def PCW2(self,best_value,P,C = []):
        len_p = len(P)
        len_c = len(C)
        for i in range(len_c):
            for j in range(len_p):
                # print("i",i,"j",j)
                C_temp = C + [P[j]]
                C_temp.remove(C[i])
                value,vector = self.eigen_pair(C_temp)
                if value > best_value:
                    P += [C[i]]
                    P.remove(P[j])
                    return self.GCW2(value,P,C_temp)
        return C
           
    def FH(self,P,C = []):
        V = zeros(len(P))
        V2 = zeros(len(P))
        R = argsort(self.AA2,axis = 1)[P,len(P)-self.s:len(P)-len(C)]
        for i in range(len(P)):
            current_set = R[i,:]
            V[i] = sum(self.AA2[i,current_set])
            V2[i] = sum(self.AA2[current_set,:][:,current_set])
            
        self.V = V
        self.V2 = V2
        self.R = R
        listR = []
        for i in range(self.n):
            listR.append(sorted(list(self.R[i,:])))
        self.listR = listR
        R2 = []
        R2f = []
        R2e = []
        for i in range(len(P)):
            current = [i]
            for j in range(self.s-len(C)-1):
                argsortk = argsort(sum(self.AA2[current,:],axis = 0))
                current.append(argsortk[-len(current)-1])
            currentf = sorted(current)
            currente = currentf[:]
            currentf.append(self.frobenius(current))
            currente.append(self.eigen_upperbound(current))

            R2.append(sorted(current))
            R2f.append(currentf)
            R2e.append(currente)
        self.R2 = R2
        self.R2f = R2f
        self.R2e = R2e
        
    def FH_mk2(self,P,C = []):
        """ Also do not touch the initial algorithm above  and fix the problem here
        the problem is, current chosen variables do not have to be the best so you have to go over in 
        order one by one to see if you can add it to the current set"""
        V = zeros(len(P))
        V2 = zeros(len(P))
        R = argsort(self.AA2,axis = 1)[P,len(P)-self.s:len(P)-len(C)]
        for i in range(len(P)):
            current_set = R[i,:]
            V[i] = sum(self.AA2[i,current_set])
            V2[i] = sum(self.AA2[current_set,:][:,current_set])
            
        self.V = V
        self.V2 = V2
        self.R = R
        listR = []
        for i in range(self.n):
            listR.append(sorted(list(self.R[i,:])))
        self.listR = listR
        R2 = []
        R2f = []
        R2e = []
        for i in range(len(P)):
            current = [i]
            for j in range(self.s-len(C)-1):
                argsortk = argsort(sum(self.AA2[current,:],axis = 0))[-len(current)-1:]
                k = 0
                found = False
                while not found and k < len(current) + 1:
                    if argsortk[k] not in current:
                        current.append(argsortk[k])
                        found = True
                    k += 1
            currentf = sorted(current)
            currente = currentf[:]
            currentf.append(self.frobenius(current))
            currente.append(self.eigen_upperbound(current))

            R2.append(sorted(current))
            R2f.append(currentf)
            R2e.append(currente)
        self.R2 = R2
        self.R2f = R2f
        self.R2e = R2e
        
    def solve_spca_mk1(self,P ,C = []):
        """ 
        
        sparse principal component with 
        
        QR Pivoting where C is included

        """
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                    """ termination condition of the algorithm """
                else:
                    C1 = C + P[0:1]
                    P1 = P[1:]
                    
                    P1 = self.qr_pivoting(C+P1)
                    P1 = [x for x in P1 if x not in C1]
                    
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """
                
    def solve_spca_mk2(self,P ,C = []):
        """ 
        
        
        sparse principal component with 
        
        QR Pivoting where only P is used
        
        
        """
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                    """ termination condition of the algorithm """
                else:
                    C1 = C + P[0:1]
                    P1 = P[1:]
                    
                    P1 = self.qr_pivoting(P1)
                    
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """

    def solve_spca_mk3(self,P ,C = []):
        """ 

        sparse principal component with 
        
        Column norm approximation of eigenvalues
        
        """
        
        P = list(map(P.__getitem__,argsort(-1*norm(self.A[:,P],2,axis = 0))))
        """ reordering indices in P according to column norms"""
        
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """

    def solve_spca_mk4(self,P ,C = []):
        
        """ 

        sparse principal component with 
        
        QR approximation of dominant eigenvalue of Covariance Matrix
        
        This algorithm assumes columns of the original data matrix are scaled
        
        """
        
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                    """ termination condition of the algorithm """
                else:
                    if C == []:
                        aggregate_double_index = argmax(self.AA2s[:,P][P,:])
                        aggregate_index0 = aggregate_double_index // len_p
                        aggregate_index1 = aggregate_double_index % len_p
                        if sum(self.AA2s[P,aggregate_index0]) >= sum(self.AA2s[P,aggregate_index1],axis = 0):
                            aggregate_index = aggregate_index0
                        else:
                            aggregate_index = aggregate_index1
                    else:
                        aggregate_index = argmax(sum(self.AA2s[:,C][P,:]))
                    correct_index = P[aggregate_index]
                    C1 = C + [correct_index]

                    P1 = P[:]
                    del P1[aggregate_index]
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """

    def solve_spca_mk5(self,P ,C = []):
        """ 

        sparse principal component with 
        
        QR approximation of dominant eigenvalue of Covariance Matrix 
        
        This algorithm assumes columns of the original data matrix are scaled
        
        with Randomization
        
        """
        
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                    """ termination condition of the algorithm """
                else:
                    if C == []:
                        correct_index = random.choice(a = P,p = diag(self.A2)[P]/sum(diag(self.A2)[P]))
                    else:
                        correct_index = random.choice(a = P,p = sum(self.AA2s[:,C][P,:],axis = 1)/sum(self.AA2s[:,C][P,:]))
                    C1 = C + [correct_index]

                    P1 = P[:]
                    P1.remove(correct_index)
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """

    def solve_spca_mk6(self,P ,C = []):
        """ 

        sparse principal component with 
        
        Sparse Column norm approximation of eigenvalues 
        
        This algorithm assumes columns of the original data matrix are scaled
        
        """
        V = zeros(len(P))
        V2 = zeros(len(P))
        R = argsort(self.AA2,axis = 1)[P,len(P)-self.s:len(P)-len(C)]
        for i in range(len(P)):
            current_set = R[i,:]
            V[i] = sum(self.AA2[i,current_set])
            V2[i] = sum(self.AA2[current_set,:][:,current_set])
            
        self.V = V
        self.V2 = V2
        self.R = R
        listR = []
        for i in range(self.n):
            listR.append(sorted(list(self.R[i,:])))
        self.listR = listR
        R2 = []
        R2f = []
        R2e = []
        for i in range(len(P)):
            current = [i]
            for j in range(self.s-len(C)-1):
                argsortk = argsort(sum(self.AA2[current,:],axis = 0))
                current.append(argsortk[-len(current)-1])
            currentf = sorted(current)
            currente = currentf[:]
            currentf.append(self.frobenius(current))
            currente.append(self.eigen_upperbound(current))

            R2.append(sorted(current))
            R2f.append(currentf)
            R2e.append(currente)
        self.R2 = R2
        self.R2f = R2f
        self.R2e = R2e
        P = list(map(P.__getitem__,argsort(-1*V)))
        self.order = P
            
        """ reordering indices in P according to column norms"""
        
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

    def solve_spca_mk10(self,P ,C = []):
        """ 

        sparse principal component with 
        
        Sparse Column norm approximation of eigenvalues 
        
        This algorithm assumes columns of the original data matrix are scaled
        
        """
        self.count = 0
        V = zeros(len(P))
        V2 = zeros(len(P))
        R = argsort(self.AA2,axis = 1)[P,len(P)-self.s:len(P)-len(C)]
        for i in range(len(P)):
            current_set = R[i,:]
            V[i] = sum(self.AA2[i,current_set])
            V2[i] = sum(self.AA2[current_set,:][:,current_set])
            
        self.V = V
        self.V2 = V2
        self.R = R
        listR = []
        for i in range(self.n):
            listR.append(sorted(list(self.R[i,:])))
        self.listR = listR
        R2 = []
        R2f = []
        R2e = []
        for i in range(len(P)):
            current = [i]
            for j in range(self.s-len(C)-1):
                argsortk = argsort(sum(self.AA2[current,:],axis = 0))
                current.append(argsortk[-len(current)-1])
            currentf = sorted(current)
            currente = currentf[:]
            currentf.append(self.frobenius(current))
            currente.append(self.eigen_upperbound(current))

            R2.append(sorted(current))
            R2f.append(currentf)
            R2e.append(currente)
        self.R2 = R2
        self.R2f = R2f
        self.R2e = R2e
        P = list(map(P.__getitem__,argsort(-1*V)))
        self.order = P
            
        """ reordering indices in P according to column norms"""
        
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
                    self.eigenvalues.append(eigenvalue)
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
                    if max(self.V[P]) <= self.best:
                        self.count += 1
                        continue
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
                        self.best = max(self.best,-1*eigenvalue)
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
                
                
    def solve_spca_mk7(self,P ,C = []):
        """ 

        sparse principal component with 
        
        Sparse Column norm approximation of eigenvalues 
        
        This algorithm assumes columns of the original data matrix are scaled
        
        """
        
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                    """ termination condition of the algorithm """
                else:
                    V = zeros(len(P))
                    R = argsort(self.AA2[P,:][:,P],axis = 1)[:,len_p-self.s+len_c:]
                    for i in range(len(P)):
                        V[i] = sum(self.AA2[i,R[i,:]])
                    aggregate_index = argmax(V)                
                    correct_index = P[aggregate_index]
                    
                    C1 = C + [correct_index]

                    P1 = P[:]
                    del P1[aggregate_index]
                    
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """

    def solve_spca_mk8(self,P ,C = []):
        """ 

        sparse principal component with 
        
        Sparse Column norm approximation of eigenvalues 
        
        This algorithm assumes columns of the original data matrix are scaled
        
        """
        
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
                    self.eigenvalues.append(eigenvalue)
                    self.eigenindices.append(C+P)
                    self.eigenvectors.append(eigenvector)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                    """ termination condition of the algorithm """
                else:
                    V = zeros(len(P))
                    R = argsort(self.AA2[P,:][:,P],axis = 1)[:,len_p-self.s+len_c:]
                    for i in range(len(P)):
                        V[i] = sum(self.AA2[i,R[i,:]]) + sum(self.AA2[i,C])
                    aggregate_index = argmax(V)                
                    correct_index = P[aggregate_index]
                    
                    C1 = C + [correct_index]

                    P1 = P[:]
                    del P1[aggregate_index]
                    
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
                self.eigenvalues.append(-1*eigen)
                self.eigenindices.append(C)
                self.eigenvectors.append(P)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                """ termination condition of the algorithm """
                
#
#  add sparse mk4 and mk5 with not a single value but like s - len(C) values
#  P by : with s - len_c this is mk6
#  P by P with s - len_c    = mk7
                



    def solve_mk0(self,P ,C = []):
        """ 
        
        This is a function to solve the all subsets problem with exploiting previously solved problems with table look ups.
        
        """

        L = [0]*(self.s+1)
        for i in range(self.s+1):
            L[i] = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_nnls(C+P)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        for i in range(len(L)):
            L[i].put([f[1],[P,C,f[0][lenc:],lenc,lenp]])
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            # print("Started solving for sparsity level ",self.s)
            count_best = 0
            while L[i].qsize() >= 1:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = L[i].get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                # print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
                if len_c < self.s:
                    if  len(coef) <= self.s:
                        self.residual_squared.append(low)
                        index = list(where(coef > 0)[0])
                        all_variables = P+C
                        real_indexes = [all_variables[i] for i in index]
                        self.indexes.append(real_indexes)
                        self.coefficients.append(coef[index])
                        """ registering the solution"""
                        count_best += 1
                        if count_best == self.many:
                            self.rem_qsize.append(L[i].qsize())
                            break
                        """ termination condition of the algorithm """
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = (sdx+xbar)*coef[:len_p]
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef1 = copy(coef)
                        coef1[l_index_bb:-1],coef1[-1] = coef1[l_index_bb+1:],coef1[l_index_bb]
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indexes"""
                        lower2 = self.qr_nnlsl(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indexes ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.qr_nnls(C1)
                            L[i].put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            L[i].put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            L[i].put([low ,[P1,C1,coef1,len_c1,len_p]])
                            L[i].put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indexes.append(C)
                    self.coefficients.append(coef)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(L[i].qsize())
                        break
                    """ termination condition of the algorithm """
    
    
    def solve(self,C = []):
            
        if self.out not in [0,1,2]:
            print("OUT parameter should be a integer >=0  and <= 2")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            return None
        elif self.many > comb(self.n,self.s):
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        
        if not type(C) == list and C != []:
            print("C should be a list, C is taken empty list now ")
            C = []
        elif C != [] and (max(C) >= self.n or min(C) < 0):
            print("Values of C should be valid, in the range 0 <= n-1, C is taken empty list")
            C = []
        elif len(set(C)) != len(C):
            print("Values of C should be unique, C is taken empty list")
            C = []
        elif len(C) > self.s:
            print(" Length of C cannot be greater than spartsity level s,C is taken empty list")
            C = []
            
            
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
        
        P = list(range(self.n))
        if C != []:
            for i in range(len(C)):
                P.remove(C[i])
        """ Preparing the P and C that will be passed to the function """
        
        """ Another if list to find and call the right function """
        
        t3 = time.time()
        self.solve_nnls(P,C)
        t4 = time.time()
        finish = time.process_time()
        duration = finish-t0
        self.cpu = duration
        self.real = t4-t3
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout
        print("CPU time of the algorithm",duration,"seconds")
        m = mem.heap()
        # print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout

        
    def solve_allsubsets(self):
        
        """ For performance concerns, only the best algorithm is offered to use for finding all best k subsets for sparsity level
        from 1 to s, """
        

        if self.many > self.n:
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            
            return None
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout
        """ whether user wants to print every step of the algorithm or not """
            
        P = list(range(self.n))
        C = []
        """ Preparing the P and C that will be passed to the function """
        
        
        t3 = time.time()
        self.solve_mk0(P,C)
        """ mk0 also works, but this is in general faster """
        t4 = time.time()
        finish = time.process_time()
        duration = finish-t0
        self.cpu = duration
        self.real = t4-t3
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout   
        print("CPU time of the algorithm",duration,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout

            