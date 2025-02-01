tic;

image_size = 32;
algo_name = "OMP";

k_values = [5, 10, 20, 30, 50, 100, 150, 200];
m_values = 100:100:1000;

plot_k_idxs = [1,5,8];
plot_m_idxs = [5,7];

RMSE = zeros(length(k_values), length(m_values));
original_image = zeros(image_size, image_size, length(k_values));
reconstructed_images_1 = zeros(image_size, image_size, length(m_values), length(plot_k_idxs));
reconstructed_images_2 = zeros(image_size, image_size, length(k_values), length(plot_m_idxs));

for idx_k = 1:length(k_values)
    k = k_values(idx_k);
    coefficients = zeros(image_size^2, 1);
    selected_indices = randperm(image_size^2, k);
    coefficients(selected_indices) = randn(k, 1);
    psi = dctmtx(image_size^2);
    vec_f = psi * coefficients;
    f = reshape(vec_f, image_size, image_size);
    original_image(:, :, idx_k) = f;
    for idx_m = 1:length(m_values)
        m = m_values(idx_m);
        phi = randi([0, 1], m, image_size^2);
        phi(phi == 0) = -1;
        A = phi * psi;
        y = phi * vec_f;
        theta = OMP(A, y, k);
        f_hat = reshape(psi*theta, image_size, image_size);
        if any(plot_k_idxs == idx_k)
          reconstructed_images_1(:, :, idx_m, find(plot_k_idxs == idx_k)) = f_hat;
        end
        if any(plot_m_idxs == idx_m)
          reconstructed_images_2(:, :, idx_k, find(plot_m_idxs == idx_m)) = f_hat;
        end
        RMSE(idx_k, idx_m) = norm(vec_f - f_hat(:)) / norm(vec_f);
    end
end

for i = 1:length(k_values)
  z = original_image(:,:,i);
  imwrite(z, sprintf("../images/omp/Ground_Truth_k_%d.png", k_values(i)));
end

for i = 1:length(plot_k_idxs)
  for j = 1:length(m_values)
    z = reconstructed_images_1(:,:,j,i);
    imwrite(z, sprintf("../images/omp/Reconstructed_k_%d_m_%d.png", k_values(plot_k_idxs(i)), m_values(j)));
  end
end

for i = 1:length(plot_m_idxs)
  for j = 1:length(k_values)
    z = reconstructed_images_2(:,:,j,i);
    imwrite(z, sprintf("../images/omp/Reconstructed_k_%d_m_%d.png", k_values(j), m_values(plot_m_idxs(i))));
  end
end

figure;
hold on;
legends = [];
for idx = 1:length(plot_k_idxs)
    k_idx = plot_k_idxs(idx);
    k = k_values(k_idx);

    y_axis = RMSE(k_idx,:);
    x_axis = m_values;
    plot(x_axis, y_axis);
    legends = [legends, sprintf("k=%d",k)];
end
xlabel('m');
ylabel('RMSE');
title(sprintf("RMSE vs m for varying k - %s", algo_name));
legend(legends);
hold off;
saveas(gcf, sprintf("../images/omp/omp_k.png"));

figure;
hold on;
legends = [];
for idx = 1:length(plot_m_idxs)
    m_idx = plot_m_idxs(idx);
    m = m_values(m_idx);

    y_axis = RMSE(:,m_idx);
    x_axis = k_values;
    plot(x_axis, y_axis);
    legends = [legends, sprintf("m=%d",m)];
end
xlabel('k');
ylabel('RMSE');
title(sprintf("RMSE vs k for varying m - %s", algo_name));
legend(legends);
hold off;
saveas(gcf, sprintf("../images/omp/omp_m.png"));

toc;

function res = OMP(A, y, k)
    [m, n] = size(A);
    norm_A = A ./ sqrt(sum(A.^2, 1));
    r = y;
    T = [];
    iterations = 0;

    while iterations < m && norm(r)^2 > 1e-10
        [~, j] = max(abs(norm_A' * r));
        if ismember(j,T)
          break;
        end
        T = [T, j];
        theta = A(:, T) \ y;
        r = y - A(:, T) * theta;
        iterations = iterations + 1;
    end

    res = zeros(n, 1);
    res(T) = theta;
end