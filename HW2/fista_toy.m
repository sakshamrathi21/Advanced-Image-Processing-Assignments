clear;
clc;

n = 200;
m = 190;
s = 5;

Phi = randn(m,n);

x = zeros(n,1);
ind = randperm(n);

x(ind(1:s)) = randn(s,1);

sig = 0.01*mean(abs(Phi*x));
y = Phi*x + sig*randn(m,1);

[VV,DD] = eig((Phi'*Phi));
alpha = max(DD(:))+1;
lambda = 0.005;

[xest,J] = fista(y,Phi,Phi',lambda,alpha,200);

plot(x); hold on; plot(xest,'color','red');