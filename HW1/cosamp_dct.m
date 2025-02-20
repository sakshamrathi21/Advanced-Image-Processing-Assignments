clear;
clc;

n = 1024;
k = 30;
mvals = [100:100:1000];

for ii=1:length(mvals)
    m = mvals(ii);
    Phi = rand(m,n);Phi(Phi < 0.5) = -1; Phi(Phi >= 0.5) = 1;
    Psi = kron(dctmtx(32)',dctmtx(32)');

    indices = randperm(1024);
    thetas = zeros(1024,1);
    thetas(indices(1:k)) = rand(k,1);

    im = Psi*thetas; im = reshape(im,32,32);
    %figure,imagesc(im,[min(im(:)) max(im(:))]); colorbar;

    A = Phi*Psi; normA = A;
    for i=1:n, normA(:,i) = A(:,i)/norm(A(:,i),2);end;
    A = normA;

    y = A*thetas;
    r = y;
    supp_set = [];
    for it=1:k
        ty = abs(A'*r);
        [sorted_ty,indices_ty] = sort(ty,'descend');
        
        new_supp_set = indices_ty(1:2*k);
        supp_set = union(supp_set,new_supp_set);
        
        b = A(:,supp_set)\y;
        temp_thetas(:) = 0;
        temp_thetas(supp_set) = b;

        signs = sign(temp_thetas);
        [sorted_thetas,indices_thetas] = sort(abs(temp_thetas),'descend');

        est_thetas(1:n) = 0;
        est_thetas(indices_thetas(1:k)) = signs(indices_thetas(1:k)).*sorted_thetas(1:k);

        r = y - A*est_thetas';
    end

    im_rec = Psi*est_thetas';
    im_true = Psi*thetas;

    rmseval(ii) = norm(im_rec-im_true,2)/norm(im_true,2);
    fprintf('\n, m = %d, rmseval = %f',mvals(ii),rmseval(ii));
    im_rec = reshape(im_rec,32,32);
    %figure,imagesc(im_rec,[min(im_rec(:)) max(im_rec(:))]); colorbar;
end
figure,plot(mvals,rmseval);
% %%%%%%%
% close all;
% %%%%%%

m = 700;
rmseval = [];
kvals = [5,10,20,30,40,50,60,70,80,90,100];
Phi = rand(m,n);Phi(Phi < 0.5) = -1; Phi(Phi >= 0.5) = 1;
Psi = kron(dctmtx(32)',dctmtx(32)');
A = Phi*Psi; normA = A;
for i=1:n, normA(:,i) = A(:,i)/norm(A(:,i),2);end;
A = normA;

for ii=1:length(kvals)
    indices = randperm(1024);
    thetas = zeros(1024,1);

    thetas(indices(1:kvals(ii))) = rand(kvals(ii),1);

    im = Psi*thetas; im = reshape(im,32,32);
    %imagesc(im,[min(im(:)) max(im(:))]); colorbar;
   
    y = A*thetas;
    r = y;

    supp_set = [];
    for it=1:kvals(ii)
        ty = abs(A'*r);
        [sorted_ty,indices_ty] = sort(ty,'descend');
        
        new_supp_set = indices_ty(1:2*k);
        supp_set = union(supp_set,new_supp_set);
        
        b = A(:,supp_set)\y;
        temp_thetas(:) = 0;
        temp_thetas(supp_set) = b;

        signs = sign(temp_thetas);
        [sorted_thetas,indices_thetas] = sort(abs(temp_thetas),'descend');

        est_thetas(1:n) = 0;
        est_thetas(indices_thetas(1:k)) = signs(indices_thetas(1:k)).*sorted_thetas(1:k);

        r = y - A*est_thetas';
    end

    im_rec = Psi*est_thetas';
    im_true = Psi*thetas;

    rmseval(ii) = norm(im_rec-im_true,2)/norm(im_true,2);
    fprintf('\n, k = %d,rmseval = %f',kvals(ii),rmseval(ii));
    im_rec = reshape(im_rec,32,32);
    %figure,imagesc(im_rec,[min(im_rec(:)) max(im_rec(:))]); colorbar;
end
figure,plot(kvals,rmseval);

