function v = modelmatvec(w,n)
% modelmatvec : multiply the model-problem matrix by a vector
%
% v = modelmatvec(w,n)   returns A*w, 
%                        where w is an n-vector and A is the
%                        n-by-n matrix of the "model problem"
%
% The "model problem" is a 5-point-stencil finite-difference
% discretization of Poisson's equation on a 2D square grid
% of k by k points.  The matrix is k^2-by-k^2.
%         
% The matrix A has a very structured form:  Each row represents
% the connections between one grid point and its (usually four)
% neighbors; thus, each matrix row has at most five nonzeros
% in a simple pattern.
%
% This routine does not form A explicitly.  Rather, we compute
% the effect of A on w point-by-point.
%
% Here are the details of the structure of the matrix:  A has 
% n = k^2 rows, one for each point in the k-by-k grid.  Grid
% point (r,s) corresponds to matrix row number i = (r-1)*k+s,
% for each r and s in 1:k.  Most of the rows of A have 5 nonzeros:
% A(i,i) = 4, and A(i,i-k) = A(i,i-1) = A(i,i+1) = A(i,i+k) = -1.
% The exceptions are when i corresponds to a boundary grid point,
% that is, a point with r or s equal to 1 or k.

k = sqrt(n);
if round(k) ~= k, error('size must be a perfect square'); end;
if length(w) ~= n, error('input vector is the wrong size'); end

v = zeros(n,1);
for r = 1:k
    for s = 1:k
        i = (r-1)*k + s;
        v(i) = 4*w(i);
        if r~=1, v(i) = v(i) - w(i-k); end;  % all but top row
        if s~=1, v(i) = v(i) - w(i-1); end;  % all but left edge
        if s~=k, v(i) = v(i) - w(i+1); end;  % all but right edge
        if r~=k, v(i) = v(i) - w(i+k); end;  % all but bottom row
    end;
end;
