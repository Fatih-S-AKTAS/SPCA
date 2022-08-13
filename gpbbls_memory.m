function [ output_args, output_eigv ,numiter,gpcrrnt] = gpbbls_memory( k,xini, datm )
% Non-convex gradient projection with BB approx., w/ non-monotone line search
% as in the paper. JJ Z
load("data_matrix.mat")
% read from outer file, python-matlab transfer leaks memory
maxiter=1000;
testiter=200;
gpcrrnt=zeros(testiter,1);
Q=-data'*data;
ai=1;
if nargin == 3
    A=datm;
end
[m n]=size(Q);
if nargin >1
    x=xini;
else
    x=zeros(n,1);
    x(1)=1;
end
oldx=zeros(n,1);
switch nargin
    case {1,2}
        qx=Q * x;
        gradfx=2* qx;
        gradfxold=gradfx;
    case {3,4}
        qx=(-A') * (A * x);
        gradfx=2* qx;
        gradfxold=gradfx;
end
j=1;
xqx=1;
for j=1:testiter
    if j>maxiter
        break
    end
    if j>2
        ai=((x-oldx)'*(gradfx-gradfxold))/((x-oldx)'*(x-oldx));
        if isnan(ai)
           ai = 0;
        end
    end
    oldx=x;
    gradfxold=gradfx;
    if gradfx==0
        break
    end
    i_search=0;
    while i_search==0 || (j>2 && (-xqx < gpcrrnt(j-1,1)) )
        i_search=i_search+1;
        if i_search >30
            break
        end
        x=proj(ai*x - gradfx,k);
        switch nargin
            case {1,2}
                qx=Q*x;
            case {3,4}
                qx=(-A') * (A * x);
        end
        xqx=x'*qx;
        ai=.25*ai;
    end
    gradfx=2* qx;
    gpcrrnt(j,1)=-xqx;
end
output_args=x;
output_eigv=-xqx;
numiter=j;
end