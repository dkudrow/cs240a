function cgtest(k)
% cgtest : Test cgsolve on the model problem
%
% cgtest(k)  creates a random right-hand-side b with k^2
%            elements, then runs "cgsolve" to solve A*x=b
%            where A is the 5-point model problem.

n = k^2;
b = rand(n,1);

[xcg, niters, relres] = cgsolve(@modelmatvec, @(i,n)b(i), n);
niters
relres
