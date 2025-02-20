clear;
clc;

ps = 8;

im = double(imread ('barbara256.png')); %im = im(1:128,1:128); 
[H,W] = size(im);

N = 60000;
patches = zeros(ps*ps,N);
cnt = 0;
for i=1:H-ps+1
    for j=1:W-ps+1
        cnt = cnt+1;
        p = im(i:i+ps-1,j:j+ps-1);
        patches(:,cnt) = p(:);
    end
end
N = size(patches,2);

D = dctmtx(ps)';
D = kron(D,D);

frac = 0.5; %:-0.1:0.1];
lfrac = length(frac);
Phi = randn(ps*ps,ps*ps);

[VV,DD] = eig(D'*(Phi'*Phi)*D);
alpha = max(DD(:))+2;
lambda = 0.005;

for f=1:lfrac
    m = floor(frac(f)*ps*ps);
    Phi = Phi(1:m,:)/sqrt(m);

    coded_patches = Phi*patches;
    sigval = 1;
    coded_patches = coded_patches + randn(size(coded_patches))*sigval;

    % errorGoal = sigval*sqrt(2);
    %thetas = OMPerr(Phi*D,coded_patches,errorGoal); 

    rec_patches = zeros(ps*ps,size(coded_patches,2));
    rec_im = im; rec_im(:,:) = 0; numcount = rec_im;
    cnt = 0;
    for i=1:H-ps+1
        if mod(i,20) == 0, fprintf (' %d',i); end;
        for j=1:W-ps+1
            cnt = cnt+1;
            
            [r,J] = ista(coded_patches(:,cnt),Phi*D,D'*Phi',lambda,alpha,200);
            rec_patches(:,cnt) = D*r;
            
            rec_im(i:i+ps-1,j:j+ps-1) = rec_im(i:i+ps-1,j:j+ps-1) + reshape(rec_patches(:,cnt),ps,ps);
            numcount(i:i+ps-1,j:j+ps-1) = numcount(i:i+ps-1,j:j+ps-1) + 1;        
        end
    end
    rec_im = rec_im./numcount;

    err_im(f) = sum((rec_im(:)-im(:)).^2)/(sum(im(:).^2));
    err_patches(f) = sum((rec_patches(:)-patches(:)).^2)/(sum(patches(:).^2));
    fprintf ('\nm = %f, image error = %f, patch error = %f',frac(f),err_im(f),err_patches(f));
    
    figure; imshow(im/255); title ('original image');   
    figure;imshow(rec_im/255); title ('reconstructed image');
end

