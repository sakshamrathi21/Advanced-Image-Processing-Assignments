function [x,J] = fista(y,H,Ht,lambda,alpha,Nit)
% [x, J] = ista_fns(y, H, lambda, alpha, Nit)
% L1-regularized signal restoration using the fast iterated
% soft-thresholding algorithm (FISTA)
% Minimizes J(x) = norm2(y-H*x)^2 + lambda*norm1(x)
% INPUT
% y - observed signal
% H - function handle
% Ht - function handle for H’
% lambda - regularization parameter
% alpha - need alpha >= max(eig(H’*H))
% Nit - number of iterations
% OUTPUT
% x - result of deconvolution
% J - objective function
J = zeros(1, Nit); % Objective function
x = 0*Ht*(y); % Initialize x
T = lambda/(2*alpha);
tk = 1;

Hx = H*x;
yk = x + (Ht*(y - Hx))/alpha;

for k = 1:Nit
    J(k) = sum(abs(Hx-y(:)).^2) + lambda*sum(abs(x(:)));

    oldx = x;
    x = yk;
    x(abs(x) <= T) = 0;
    x(x > T) = x(x > T)-T;
    x(x < -T) = x(x < -T)+T;

    tk1 = (1 + sqrt(1+4*tk*tk))/2;
    yk = x + (tk-1)*(x-oldx)/(tk1);

    tk = tk1;
    Hx = H*x;
end

plot(J);