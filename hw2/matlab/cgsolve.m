function [x, niters, relres] = cgsolve(Atimes, getb, n)
% cgsolve : Solve a linear system A*x=b by conjugate gradients
%
% [x, niters, relres] = cgsolve(n, @Atimes, @getb);
%
% This routine solves the linear system A*x=b for x.
% The matrix A is not given explicitly, but rather as a
% user-supplied function that multiplies A by a given vector.
% The right-hand side b is given by a user-supplied function
% that returns individual elements of b.
%
% Inputs:
%   n          integer: dimension of the matrix and vectors
%   Atimes     function: y = Atimes(z,n) should return A*z
%   getb       function: bi = getb(i,n) should return b(i)
% Outputs:
%   x          vector: computed solution to A*x=b
%   niters     integer: number of CG iterations performed
%   relres     double: relative residual norm, defined as
%                      norm(b-A*x)/norm(b)
%
% The CG algorithm requires the matrix to be symmetric and
% positive definite, though we do not check this.  We iterate
% either until the relative residual is less than 10^-6
% or for at most max(1000,10*sqrt(n)) iterations.  A more
% robust code would let the user specify the stopping condition.
%
% This is a sequential Matlab template -- CS240A homework 2 is 
% to implement this in parallel, including an "Atimes" routine
% that applies the 5-point model problem to a vector in parallel
% without ever actually forming the matrix.  In the parallel
% code, the vectors b, x, r, and d will all be distributed across 
% processors, and all the operations on vectors will be done by 
% calls to subroutines you write using MPI.
%
% John R. Gilbert     3 April 2011

% Note: for initial debugging, set maxiters to something small like 3.
maxiters = max(1000,5*sqrt(n));
b = zeros(n,1); 
for i=1:n
    b(i) = getb(i,n);
end;
normb = sqrt(b'*b);

x = zeros(n,1);
r = b;
rtr = r'*r;
relres = 1;
d = r;
niters = 0;
while relres > 1e-6  &&  niters < maxiters
    niters = niters+1;
    Ad = Atimes(d,n);
    alpha = rtr / (d'*Ad);
    x = x + alpha * d;
    r = r - alpha * Ad;
    rtrold = rtr;
    rtr = r'*r;
    beta = rtr / rtrold;
    d = r + beta * d;
    relres = sqrt(rtr) / normb;
end;