%% Part (a): Read video and extract T frames
clear; close all; clc;
imageFolderPath = 'C:\Users\Kshitij Vaidya\Documents\CS754-Assignments\HW3\assignment3\2\images\';

% Read video
A = mmread('cars.avi');
T = 3; % Can be changed to 5 or 7 later
frames = [];
for i = 1:T
    frame = rgb2gray(A.frames(i).cdata);
    frames(:,:,i) = double(frame);
end

% Crop to region around lowermost car (120x240)
[H_orig, W_orig, ~] = size(frames);
frames_crop = frames(end-119:end, :, :); % Bottom 120 rows
mid_col = floor(size(frames_crop, 2)/2);
frames_crop = frames_crop(:, mid_col-119:mid_col+120, :);
[H, W, ~] = size(frames_crop);

% Display cropped frames
figure;
numRows = ceil(T/2);
numCols = 2;
for t = 1:T
    subplot(numRows, numCols, t);
    imshow(frames_crop(:,:,t), []);
    title(['Frame ', num2str(t)]);
end
saveas(gcf, fullfile(imageFolderPath, 'originalFrames.png'));

%% Part (b): Generate code and coded snapshot with noise
% Generate random binary codes
Ct = randi([0 1], H, W, T);

% Create coded snapshot
Eu = zeros(H, W);
for t = 1:T
    Eu = Eu + Ct(:,:,t) .* frames_crop(:,:,t);
end

% Add noise
noise_std = 2;
Eu_noisy = Eu + noise_std * randn(size(Eu));
%Eu_noisy = Eu_noisy;
% Display coded snapshot
figure;
imshow(Eu_noisy, []);
title('Noisy Coded Snapshot');
saveas(gcf, fullfile(imageFolderPath, 'codedSnapshot.png'));

%%
% Parameters
patch_size = 8;
step = 2; % Overlap step size
k_sparse = 40; % Sparsity level for OMP

% Create DCT basis for 8x8 patches
D = dctmtx(patch_size);
DCT_basis = kron(D, D); % 2D DCT basis (64x64)

% Initialize reconstructed frames
recon_frames = zeros(H, W, T);
overlap_count = zeros(H, W, T);

% Process each patch
for row = 1:step:H-patch_size+1
    for col = 1:step:W-patch_size+1
        % Get current patch from coded snapshot (b)
        b_patch = Eu_noisy(row:row+patch_size-1, col:col+patch_size-1);
        b_patch = b_patch(:);
        
        % Get code patches for all frames
        Ct_patch = Ct(row:row+patch_size-1, col:col+patch_size-1, :);
        
        % Construct measurement matrix A for this patch
        A_patch = zeros(patch_size^2, patch_size^2 * T);
        for t = 1:T
            ct = Ct_patch(:,:,t);
            ct = ct(:);
            A_patch(:,(t-1)*patch_size^2+1:t*patch_size^2) = diag(ct) * DCT_basis;
        end
        
        % Solve using OMP
        x_patch = omp(A_patch, b_patch, k_sparse);
        
        % Reconstruct each frame's patch
        for t = 1:T
            theta = x_patch((t-1)*patch_size^2+1:t*patch_size^2);
            ft_patch = DCT_basis * theta;
            ft_patch = reshape(ft_patch, patch_size, patch_size);
            
            % Accumulate into reconstructed frames
            recon_frames(row:row+patch_size-1, col:col+patch_size-1, t) = ...
                recon_frames(row:row+patch_size-1, col:col+patch_size-1, t) + ft_patch;
            overlap_count(row:row+patch_size-1, col:col+patch_size-1, t) = ...
                overlap_count(row:row+patch_size-1, col:col+patch_size-1, t) + 1;
        end
    end
end

% Average overlapping regions
recon_frames = recon_frames ./ overlap_count;

% Calculate RMSE
rmse = sqrt(mean((recon_frames(:) - frames_crop(:)).^2)) / sqrt(mean(frames_crop(:).^2));
fprintf('Relative RMSE for T=%d: %.4f\n', T, rmse);

% Display results
for t = 1:T
    figure;
    subplot(1,2,1); imshow(frames_crop(:,:,t), []); 
    title(['Original Frame ', num2str(t)]);
    subplot(1,2,2); imshow(recon_frames(:,:,t), []); 
    title(['Reconstructed Frame ', num2str(t)]);
end

%% Part (f): Repeat for T=5 and T=7
% (Same code as above, just change T and re-run)

%% Part (h): Repeat with flame video
% (Similar code as above but with flame video)

%% OMP Implementation
function x = omp(A, b, k)
    residual = b;
    idx = [];
    x = zeros(size(A,2),1);
    
    for iter = 1:k
        % Find the column of A most correlated with residual
        correlations = A' * residual;
        [~, new_idx] = max(abs(correlations));
        
        % Check if this index was already selected
        if ismember(new_idx, idx)
            break;
        end
        
        % Add new index to selected set
        idx = [idx, new_idx];
        
        % Solve least squares problem with selected columns
        A_selected = A(:, idx);
        x_ls = A_selected \ b;
        
        % Update residual
        residual = b - A_selected * x_ls;
        
        % Check stopping criterion
        if norm(residual) < 1e-6
            break;
        end
    end
    
    % Put the solution in the correct locations
    x(idx) = x_ls;
end