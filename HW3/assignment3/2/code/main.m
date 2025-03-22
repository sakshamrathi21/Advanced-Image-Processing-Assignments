clc, clear, close all;

imageFolderPath = 'C:\Users\Kshitij Vaidya\Documents\CS754-Assignments\HW3\assignment3\2\images\';
path = 'cars.avi';
A = mmread(path);
T = 3;

%%

% Read the first T frames of the video
X = zeros(size(A.frames(1).cdata, 1), size(A.frames(1).cdata, 2), T);
for i=1:T
    X(:, :, i) = double(rgb2gray(A.frames(i).cdata));
end


%%

% Extract the 120*240 portion from the lower part of the video
X = X(end-119:end, end-239:end, :);
[H, W, T] = size(X);

% Display the original video frames
figure;
for i=1:T
    subplot(1, T, i);
    imshow(uint8(X(:, :, i)));
    title("Original Frame " + i);
end
saveas(gcf, fullfile(imageFolderPath, 'original_frames.png'));

%% 

% Generate the random code pattern C of size H*W*T and entries in {0, 1}
C = randi([0, 1], H, W, T);
% Compute the coded snapshot E
E = sum(X .* C, 3);
% Display the coded snapshot
figure;
imshow(uint8(E));
title("Coded Snapshot");
saveas(gcf, fullfile(imageFolderPath, 'coded_snapshot.png'));

%%

% Add noise to the coded snapshot with standard deviation 2
noise = 2 * randn(size(E));
noisyE = E + noise;

% Display the noisy coded snapshot and the original coded snapshot
figure;
subplot(1, 2, 1);
imshow(uint8(noisyE));
title("Noisy Coded Snapshot");
subplot(1, 2, 2);
imshow(uint8(E));
title("Original Coded Snapshot");
saveas(gcf, fullfile(imageFolderPath, 'noisy_coded_snapshot.png'));
%%

% Define the Soft Thresholding function
function [output] = softThresholding(input, threshold)
    output = sign(input) .* max(abs(input) - threshold, 0);
end

% Define the ISTA function for a single patch
function [reconstructedPatch] = patchISTA(patch, A, lambda, alpha, maxIter)
    % Initialize the reconstructed patch as a vector of zeros
    reconstructedPatch = zeros(192, 1);
    % Compute the threshold as lambda / (2 * alpha)
    threshold = lambda / (2 * alpha);
    % Iterate for maxIter times
    for i = 1:maxIter
        % Perform the update using the Soft Thresholding function
        gradient = A' * (patch - A * reconstructedPatch);
        reconstructedPatch = softThresholding(reconstructedPatch + (1 / alpha) * gradient, threshold);
    end
end

%%
patchSize = 8;
D1 = dctmtx(patchSize);
D = kron(D1, D1);
reconstructedX = zeros(H, W, T);

for i=1:patchSize:H
    for j=1:patchSize:W
        disp(i + " " + j);
        iEnd = i + patchSize - 1;
        jEnd = j + patchSize - 1;
        b = noisyE(i:iEnd, j:jEnd);
        b = b(:);

        patchA = [];
        for t=1:T
            patchCt = C(i:iEnd, j:jEnd, t);
            vecCt = patchCt(:);
            At = bsxfun(@times, D, vecCt);
            patchA = [patchA, At];
        end
        alpha = max(eig(patchA' * patchA));
        lambda = 0.1 * alpha;
        maxIter = 1000;
        %disp("Shape of b: " + size(b));
        %disp("Shape of patchA: " + size(patchA));
        output = patchISTA(b, patchA, lambda, alpha, maxIter);
        splitOutput = reshape(output, [], T);

        for t = 1:T
            patchT = D * splitOutput(:, t);
            reconstructedX(i:iEnd, j:jEnd, t) = reshape(patchT, patchSize, patchSize);
        end
    end
end

%%

for i=1:T
    subplot(1, T, i);
    imshow(uint8(reconstructedX(:, :, i)));
    title("Reconstructed Frame " + i);
end
saveas(gcf, fullfile(imageFolderPath, 'reconstructed_frames.png'));

%%

% Find the RMSE between the original and reconstructed frames
rmse = zeros(1, T);
for i=1:T
    rmse(i) = sqrt(mean((X(:, :, i) - reconstructedX(:, :, i)).^2, 'all'));
end
disp("RMSE: " + rmse);