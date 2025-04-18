clear;
n1 = 500;
n2 = 800;
r = 50;
s = round(0.1*n1*n2);

A = randn(n1,n2);
[U,Sig,V] = svd(A);
L = U(:,1:r)*Sig(1:r,1:r)*(V(:,1:r))';
S = zeros(n1*n2,1);
indices = randperm(n1*n2);
S(indices(1:s)) = 5*randn(s,1);
S = reshape(S,n1,n2);

M = L+S;

mu = 0.25*n1*n2/(4*sum(abs(M(:)))); recmu = 1/mu;
lambda = 1/sqrt(max([n1 n2]));
lr = lambda*recmu;

converged = 0;
estS = zeros(n1,n2); estL = estS; Y = estS;

count = 0;
while converged ~= 1
    fprintf('\n%d',count);
    old_estL = estL;
    old_estS = estS;

    L1 = M-estS+Y/mu;
    [U,D,V] = svd(L1);
    D = wthresh(D,"s",recmu);
    estL = U*D*V';

    estS = M-estL+Y/mu;
    %estS(abs(estS) > lr) = (estS(abs(estS) > lr)-lr).*sign(estS(abs(estS)>lr));
    estS = wthresh(estS,"s",lr);

    Y = Y + mu*(M-estL-estS);

    delta =  norm(estL(:)-old_estL(:))/norm(old_estL(:));
    fprintf(' %f',delta);
    if delta < 0.001, converged = 1; end;
    count = count+1;
end

errL = norm(L(:)-estL(:))/norm(L(:));
errS = norm(S(:)-estS(:))/norm(S(:));
fprintf('\nerrL = %f, errS = %f',errL,errS);

