#!/usr/bin/env python
from static_questions import pitprops,ozone_X2
from numpy import array,real,dot,column_stack,append,row_stack,zeros
import numpy
ra = numpy.random
la = numpy.linalg

def PathSPCA(A,k):
    M,N=A.shape
    # Loop through variables
    As = la.norm(A.T.dot(A),axis = 1)
    # As=((A*A).sum(axis=0));
    vmax=As.max();vp=As.argmax();subset=[vp];
    variances=[];res=subset;rhos=[(A[:,vp]*A[:,vp]).sum()];
    Stemp=array([rhos])
    for i in range(1,k):
        lev,v = la.eig(Stemp)
        variances.append(real(lev).max())
        vp=real(lev).argmax()
        x=dot(A[:,subset],v[:,vp])
        x=x/la.norm(x)
        seto = list(range(0,N))
        for j in subset:
            seto.remove(j)
        vals=dot(x.T,A[:,seto]);vals=vals*vals
        rhos.append(vals.max())
        vpo=seto[vals.argmax()]
        Stemp=column_stack((Stemp,dot(A[:,subset].T,A[:,vpo])))
        vbuf=append(dot(A[:,vpo].T,A[:,subset]),array([(A[:,vpo]*A[:,vpo]).sum()]))
        Stemp=row_stack((Stemp,vbuf))
        subset.append(vpo)
    lev,v=la.eig(Stemp)
    variances.append(real(lev).max())
    return variances,res,rhos

# **** Run quick demo ****
# Simple data matrix with N=7 variables and M=3 samples
k=5 # target cardinality
A = array([[1,2,3,4,3,2,1],[4,2,1,4,3,2,1],[5,2,3,4,3,3,1]])
A = ozone_X2[:20,:]

m,n = numpy.shape(A)
mA = numpy.reshape(numpy.mean(A,axis = 0),[1,n])
A = A - numpy.ones([m,1]).dot(mA)
sA = numpy.std(A,axis = 0)
A = A/sA


# Call function
variances,res,rhos=PathSPCA(A,k)
print( res )
print( variances )
print( rhos )

def cholesky
n=size(S,1);A=chol(S);
subset=[1];subres=[subset;zeros(n-length(subset),1)];
res=[];rhobreaks=[sum(A(:,1).^2)];sol=[];vars=[];

% Loop through variables
for i=1:n
    % Compute solution at current subset
    [v,mv]=maxeig(S(subset,subset));
    vsol=zeros(n,1);vsol(subset)=v;
    sol=[sol,vsol];vars=[vars,mv];
    % Compute x at current subset
    x=A(:,subset)*v;x=x/norm(x);
    res=[res,[subset;zeros(n-length(subset),1)]];
    % Compute next rho breakpoint
    set=1:n;set(subset)=[];
    vals=(x*A(:,set)).^2;
    [rhomax,vpos]=max(vals);
    rhobreaks=[rhobreaks;rhomax];
    subset=[subset,set(vpos)];
end