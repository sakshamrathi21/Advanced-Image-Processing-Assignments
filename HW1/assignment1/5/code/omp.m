tic;
img_dim = 32;
algorithm_name = "OMP"; % Updated algorithm name
sparsity_levels = [5, 10, 20, 30, 50, 100, 150, 200];
sample_sizes = 100:100:1000;
sparsity_plot_indices = [1,5,8];
sample_plot_indices = [5,7];
error_matrix = zeros(length(sparsity_levels), length(sample_sizes));
true_image = zeros(img_dim, img_dim, length(sparsity_levels));
recon_images_1 = zeros(img_dim, img_dim, length(sample_sizes), length(sparsity_plot_indices));
recon_images_2 = zeros(img_dim, img_dim, length(sparsity_levels), length(sample_plot_indices));

for s_idx = 1:length(sparsity_levels)
    s = sparsity_levels(s_idx);
    coeff_vector = zeros(img_dim^2, 1);
    chosen_indices = randperm(img_dim^2, s);
    coeff_vector(chosen_indices) = randn(s, 1);
    transform_matrix = dctmtx(img_dim^2);
    signal_vector = transform_matrix * coeff_vector;
    reshaped_image = reshape(signal_vector, img_dim, img_dim);
    true_image(:, :, s_idx) = reshaped_image;
    
    for sm_idx = 1:length(sample_sizes)
        sm = sample_sizes(sm_idx);
        sensing_matrix = randi([0, 1], sm, img_dim^2);
        sensing_matrix(sensing_matrix == 0) = -1;
        measurement_matrix = sensing_matrix * transform_matrix;
        observations = sensing_matrix * signal_vector;
        estimated_coeffs = OMP_Algo(measurement_matrix, observations, s);
        reconstructed_image = reshape(transform_matrix * estimated_coeffs, img_dim, img_dim);
        
        if any(sparsity_plot_indices == s_idx)
          recon_images_1(:, :, sm_idx, find(sparsity_plot_indices == s_idx)) = reconstructed_image;
        end
        if any(sample_plot_indices == sm_idx)
          recon_images_2(:, :, s_idx, find(sample_plot_indices == sm_idx)) = reconstructed_image;
        end
        
        error_matrix(s_idx, sm_idx) = norm(signal_vector - reconstructed_image(:)) / norm(signal_vector);
    end
end

% Save images
for i = 1:length(sparsity_levels)
  imwrite(true_image(:,:,i), sprintf("../images/omp/Ground_Truth_k_%d.png", sparsity_levels(i)));
end
for i = 1:length(sparsity_plot_indices)
  for j = 1:length(sample_sizes)
    imwrite(recon_images_1(:,:,j,i), sprintf("../images/omp/Reconstructed_k_%d_m_%d.png", sparsity_levels(sparsity_plot_indices(i)), sample_sizes(j)));
  end
end
for i = 1:length(sample_plot_indices)
  for j = 1:length(sparsity_levels)
    imwrite(recon_images_2(:,:,j,i), sprintf("../images/omp/Reconstructed_k_%d_m_%d.png", sparsity_levels(j), sample_sizes(sample_plot_indices(i))));
  end
end

% Plot RMSE
figure;
hold on;
legend_entries = [];
for idx = 1:length(sparsity_plot_indices)
    s_idx = sparsity_plot_indices(idx);
    s = sparsity_levels(s_idx);
    plot(sample_sizes, error_matrix(s_idx,:));
    legend_entries = [legend_entries, sprintf("k=%d", s)];
end
xlabel('m');
ylabel('RMSE');
title(sprintf("RMSE vs m for varying k - %s", algorithm_name));
legend(legend_entries);
hold off;
saveas(gcf, sprintf("../images/omp/omp_k.png"));

figure;
hold on;
legend_entries = [];
for idx = 1:length(sample_plot_indices)
    sm_idx = sample_plot_indices(idx);
    sm = sample_sizes(sm_idx);
    plot(sparsity_levels, error_matrix(:,sm_idx));
    legend_entries = [legend_entries, sprintf("m=%d", sm)];
end
xlabel('k');
ylabel('RMSE');
title(sprintf("RMSE vs k for varying m - %s", algorithm_name));
legend(legend_entries);
hold off;
saveas(gcf, sprintf("../images/omp/omp_m.png"));

toc;

function result = OMP_Algo(A, y, sparsity)
    % Modified Orthogonal Matching Pursuit
    [num_rows, num_cols] = size(A);
    normalized_A = A ./ sqrt(sum(A.^2, 1));
    residual = y;
    selected_atoms = [];
    iter = 0;

    while iter < num_rows && norm(residual)^2 > 1e-10
        [~, max_idx] = max(abs(normalized_A' * residual));
        if ismember(max_idx, selected_atoms)
          break;
        end
        selected_atoms = [selected_atoms, max_idx];
        theta_estimate = A(:, selected_atoms) \ y;
        residual = y - A(:, selected_atoms) * theta_estimate;
        iter = iter + 1;
    end

    result = zeros(num_cols, 1);
    result(selected_atoms) = theta_estimate;
end