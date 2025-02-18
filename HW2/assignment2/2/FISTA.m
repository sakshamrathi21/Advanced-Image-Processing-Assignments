clc; clear; close all;

barbaraOriginal = imread("barbara256.png");
barbaraOriginal = double(barbaraOriginal);
barbaraNoisy = barbaraOriginal + 4.0 * randn(size(barbaraOriginal));
barbaraNoisy = barbaraNoisy / 256.0;

[height, width] = size(barbaraOriginal);
patchSize = 8;
patches = im2col(barbaraNoisy, [patchSize, patchSize], 'sliding');
numPatches = size(patches, 2);

%%

Phi = randn(32, 64);
DCT = kron(dctmtx(8)', dctmtx(8)');
A = Phi * DCT;
alpha = max(eig(Phi * (DCT * DCT') * Phi'));
lambda = 1.0;
maxIter = 1000;

sparsity = zeros(size(patches, 2), 1);
for i = 1:size(patches, 2)
    sparsity(i) = nnz(DCT * patches(:, i));
end
disp("Sparsity: " + mean(sparsity));
% Check the mean entry in the patches
disp("Mean Entry: " + mean(patches(:)));

%%

function [output] = softThresholding(input, threshold)
    output = sign(input) .* max(abs(input) - threshold, 0);
end

%% 

function [reconstructedPatch] = patchISTA(patch, A, lambda, alpha, maxIter)
    reconstructedPatch = zeros(64, 1);
    threshold = lambda / (2 * alpha);
    for i = 1:maxIter
        gradient = A' * (patch - A * reconstructedPatch);
        reconstructedPatch = softThresholding(reconstructedPatch + (1 / alpha) * gradient, threshold);
    end
end

%%

function [reconstructedPatch] = patchFISTA(patch, A, lambda, alpha, maxIter)
    reconstructedPatch = zeros(64, 1);
    threshold = lambda / (2 * alpha);
    y = reconstructedPatch;
    t = 1;
    for i = 1:maxIter
        gradient = A' * (patch - A * y);
        x = softThresholding(y + (1 / alpha) * gradient, threshold);
        tNew = (1 + sqrt(1 + 4 * t^2)) / 2;
        y = x + ((t - 1) / tNew) * (x - reconstructedPatch);
        reconstructedPatch = x;
        t = tNew;
    end
end

%%

reconstructedPatches = zeros(64, numPatches);
for i = 1:numPatches
    currPatch = Phi * patches(:, i);
    reconstructedPatches(:, i) = DCT * patchFISTA(currPatch, A, lambda, alpha, maxIter);
end

%% 


function reconstructedImage = reconstructFromPatches(reconstructedPatches, patchSize, imageSize)
    % reconstructedPatches: Matrix of size [patchSize^2 x numPatches]
    % patchSize: Size of each patch (e.g., [8, 8])
    % imageSize: Size of the final image (e.g., [256, 256])
    
    % Initialize the reconstructed image and weight matrix
    reconstructedImage = zeros(imageSize);
    weightMatrix = zeros(imageSize);
    
    % Calculate the number of patches along rows and columns
    numPatchesX = imageSize(2) - patchSize(2) + 1;
    numPatchesY = imageSize(1) - patchSize(1) + 1;
    
    % Iterate over each patch and place it in the correct position
    patchIndex = 1;
    for y = 1:numPatchesY
        for x = 1:numPatchesX
            % Extract the current patch and reshape it correctly
            patch = reshape(reconstructedPatches(:, patchIndex), patchSize(1), patchSize(2));
            
            % Define the region in the image where the patch will be placed
            rowRange = y:(y + patchSize(1) - 1);
            colRange = x:(x + patchSize(2) - 1);
            
            % Add the patch to the reconstructed image
            reconstructedImage(rowRange, colRange) = reconstructedImage(rowRange, colRange) + patch;
            
            % Update the weight matrix
            weightMatrix(rowRange, colRange) = weightMatrix(rowRange, colRange) + 1;
            
            % Move to the next patch
            patchIndex = patchIndex + 1;
        end
    end
    
    % Normalize the reconstructed image by dividing by the weight matrix
    reconstructedImage = reconstructedImage ./ weightMatrix;
    reconstructedImage = reconstructedImage';
    reconstructedImage = reconstructedImage * 256.0;
end

%% 

reconstructedImage = reconstructFromPatches(reconstructedPatches, [patchSize patchSize], [height, width]);
figure;
subplot(1, 3, 1);
imshow(uint8(barbaraOriginal));
title("Original Image");
subplot(1, 3, 2);
imshow(uint8(barbaraNoisy * 256.0));
title("Noisy Image");
subplot(1, 3, 3);
imshow(uint8(reconstructedImage));
title("Reconstructed Image");
saveas(gcf, "FISTA_Barbara.png");

%%
% RMSE = 0.14027
barbaraRMSE = sqrt(mean((barbaraOriginal(:) - reconstructedImage(:)).^2)) / sqrt(mean(barbaraOriginal(:).^2));
disp("RMSE: " + barbaraRMSE);

%%

% Perform the same process on the Goldhill image take the top left 256*256
goldhillOriginal = imread("goldhill.png");
goldhillOriginal = goldhillOriginal(1:256, 1:256);
goldhillOriginal = double(goldhillOriginal);

% Add noise to the Goldhill image
noise = 4.0 * randn(size(goldhillOriginal));
goldhillNoisy = goldhillOriginal + noise;
goldhillNoisy = goldhillNoisy / 256.0;
% Divide the images into patches of 8*8
[height, width] = size(goldhillNoisy);
% Define the 8*8 patches with overlapping
patches = im2col(goldhillNoisy, [patchSize, patchSize], 'sliding');

%%

% Define reconstructedPatches as an 32 * 32 cell array of 64 * 1 vectors
numPatches = size(patches, 2);
reconstructedPatches = zeros(64, numPatches);

% Iterate over the patches and reconstruct them using ISTA
for i = 1:numPatches
    if mod(i, 1000) == 0
        disp("Patch " + i);
    end
    currPatch = patches(:,i);
    measuredPatch = Phi * currPatch;
    reconstructedPatches(:, i) = DCT * patchFISTA(measuredPatch, A, lambda, alpha, maxIter);
end

%%

% Reconstruct the image from the patches
reconstructedImage = reconstructFromPatches(reconstructedPatches, [patchSize patchSize], [height width]);

% Compute the RMSE between the original image and the reconstructed image
GoldhillRMSE = sqrt(mean((goldhillOriginal(:) - reconstructedImage(:)).^2)) / sqrt(mean(goldhillOriginal(:).^2));
disp("RMSE: " + GoldhillRMSE);
% RMSE = 0.088566
% Display the original image, noisy image and the reconstructed image
figure;
subplot(1, 3, 1);
imshow(uint8(goldhillOriginal));
title("Original Image");
subplot(1, 3, 2);
imshow(uint8(goldhillNoisy * 256.0));
title("Noisy Image");
subplot(1, 3, 3);
imshow(uint8(re/home/kshitij-vaidya/CS747_Assignments/references.txtconstructedImage));
title("Reconstructed Image");
% Save the figure
saveas(gcf, "FISTA_Goldhill.png");

%%

% Analyse the variation of the RMSE with respect to the regularization parameter lambda from 10^-3 to 10^5 in log scale
lambdaRange = logspace(-3, 5, 9);
barbaraOriginal = double(imread("barbara256.png"));
barbaraNoisy = barbaraOriginal + 4.0 * randn(size(barbaraOriginal));
barbaraNoisy = barbaraNoisy / 256.0;
patches = im2col(barbaraNoisy, [patchSize, patchSize], 'sliding');
numPatches = size(patches, 2);
% Define the sensing matrix as a random normal 32*64 matrix
Phi = randn(32, 64);
% Define the 2D DCT matrix as a 64*64 matrix
DCT = kron(dctmtx(8)', dctmtx(8)');
% Define the A matrix as the product of Phi and U
A = Phi * DCT;
% Determine the value of alpha as the maximum eigenvalue of Phi * DCT * DCT' * Phi'
alpha = max(eig(Phi * (DCT * DCT') * Phi'));
% Define the maximum iterations as 1000
maxIter = 1000;
% Perform the analysis on the Barbara image
RMSEBarbara = zeros(size(lambdaRange));
for i = 1:length(lambdaRange)
    lambda = lambdaRange(i);
    disp("Lambda: " + lambda);
    % Define reconstructedPatches as an 32 * 32 cell array of 64 * 1 vectors
    reconstructedPatches = zeros(64, numPatches);
    % Iterate over the patches and reconstruct them using ISTA
    for j = 1:numPatches
        currPatch = patches(:,j);
        measuredPatch = Phi * currPatch;
        reconstructedPatches(:, j) = DCT * patchFISTA(measuredPatch, A, lambda, alpha, maxIter);
    end
    % Reconstruct the image from the patches
    reconstructedImage = reconstructFromPatches(reconstructedPatches, [patchSize patchSize], [height width]);
    % Compute the RMSE between the original image and the reconstructed image
    RMSEBarbara(i) = sqrt(mean((barbaraOriginal(:) - reconstructedImage(:)).^2)) / sqrt(mean(barbaraOriginal(:).^2));
end

%%

% Plot the RMSE vs lambda for the Barbara image
figure;
semilogx(lambdaRange, RMSEBarbara, 'LineWidth', 2);
xlabel("Regularization Parameter (\lambda)", FontSize=14);
ylabel("RMSE", FontSize=14);
title("RMSE vs \lambda for Barbara Image", FontSize=16);
grid on;
% Save the figure
saveas(gcf, "FISTA_RMSE_Barbara.png");

