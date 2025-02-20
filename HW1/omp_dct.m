clear;
clc;

n = 1024;
k = 200;
mvals = [100:100:1000];

for ii=1:length(mvals)
    m = mvals(ii);
    Phi = rand(m,n);Phi(Phi < 0.5) = -1; Phi(Phi >= 0.5) = 1;
    Psi = kron(dctmtx(32)',dctmtx(32)');

    indices = randperm(1024);
    thetas = zeros(1024,1);
    thetas(indices(1:k)) = rand(k,1);

    im = Psi*thetas; im = reshape(im,32,32);
    imagesc(im,[min(im(:)) max(im(:))]); colorbar;

    A = Phi*Psi; normA = A;
    for i=1:n, normA(:,i) = A(:,i)/norm(A(:,i),2);end;
    
    y = A*thetas;
    r = y;
    supp_set = [];
    for it=1:k
        dp = abs(r'*normA);
        [maxval,maxind] = max(dp); j = maxind(1); 
        supp_set = [supp_set j];
        theta_supp_set = A(:,supp_set)\y;
        r = y-A(:,supp_set)*theta_supp_set;
    end

    im_rec = Psi(:,supp_set)*theta_supp_set;
    im_true = Psi*thetas;

    rmseval(ii) = norm(im_rec-im_true,2)/norm(im_true,2);
    fprintf('\nrmseval = %f',rmseval(ii));
    im_rec = reshape(im_rec,32,32);
    figure,imagesc(im_rec,[min(im_rec(:)) max(im_rec(:))]); colorbar;
end
figure,plot(mvals,rmseval);
%%%%%%%
close all;
%%%%%%
m = 500;
rmseval = [];
kvals = [5,10,20,30,40,50,100,150,200];
Phi = rand(m,n);Phi(Phi < 0.5) = -1; Phi(Phi >= 0.5) = 1;
Psi = kron(dctmtx(32)',dctmtx(32)');
A = Phi*Psi; normA = A;
for i=1:n, normA(:,i) = A(:,i)/norm(A(:,i),2);end;

for ii=1:length(kvals)
    indices = randperm(1024);
    thetas = zeros(1024,1);

    thetas(indices(1:kvals(ii))) = rand(kvals(ii),1);

    im = Psi*thetas; im = reshape(im,32,32);
    imagesc(im,[min(im(:)) max(im(:))]); colorbar;
   
    y = A*thetas;
    r = y;
    supp_set = [];
    for it=1:kvals(ii)
        dp = abs(r'*normA);
        [maxval,maxind] = max(dp); j = maxind(1); 
        supp_set = [supp_set j];
        theta_supp_set = A(:,supp_set)\y;
        r = y-A(:,supp_set)*theta_supp_set;
    end

    im_rec = Psi(:,supp_set)*theta_supp_set;
    im_true = Psi*thetas;

    rmseval(ii) = norm(im_rec-im_true,2)/norm(im_true,2);
    fprintf('\nrmseval = %f',rmseval);
    im_rec = reshape(im_rec,32,32);
    figure,imagesc(im_rec,[min(im_rec(:)) max(im_rec(:))]); colorbar;
end
figure,plot(kvals,rmseval);

