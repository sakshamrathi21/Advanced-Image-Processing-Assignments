tic;

image_size = 32;
algo_name = "COSAMP"; % Name of the algorithm
k_values = [5, 10, 20, 30, 50, 100, 150, 200];
m_values = 100:100:1000;
k_indices_to_plot = [1,5,8];
m_indices_to_plot = [5,7];

RMSE = zeros(length(k_values), length(m_values));
original_image = zeros(image_size, image_size, length(k_values));
reconstructed_images_1 = zeros(image_size, image_size, length(m_values), length(k_indices_to_plot));
reconstructed_images_2 = zeros(image_size, image_size, length(k_values), length(m_indices_to_plot));

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
        theta = cosamp_algo(A, y, k);
        f_hat = reshape(psi*theta, image_size, image_size);
        if any(k_indices_to_plot == idx_k)
          reconstructed_images_1(:, :, idx_m, find(k_indices_to_plot == idx_k)) = f_hat;
        end
        if any(m_indices_to_plot == idx_m)
          reconstructed_images_2(:, :, idx_k, find(m_indices_to_plot == idx_m)) = f_hat;
        end
        RMSE(idx_k, idx_m) = norm(vec_f - f_hat(:)) / norm(vec_f);
    end
end


% Save the ground truth and reconstructed images
for i = 1:length(k_values)
  z = original_image(:,:,i);
  imwrite(z, sprintf("../images/cosamp/Ground_Truth_k_%d.png", k_values(i)));
end
for i = 1:length(k_indices_to_plot)
  for j = 1:length(m_values)
    z = reconstructed_images_1(:,:,j,i);
    imwrite(z, sprintf("../images/cosamp/Reconstructed_k_%d_m_%d.png", k_values(k_indices_to_plot(i)), m_values(j)));
  end
end
for i = 1:length(m_indices_to_plot)
  for j = 1:length(k_values)
    z = reconstructed_images_2(:,:,j,i);
    imwrite(z, sprintf("../images/cosamp/Reconstructed_k_%d_m_%d.png", k_values(j), m_values(m_indices_to_plot(i))));
  end
end

% Plot RMSE vs k and m
figure;
hold on;
legends = [];
for idx = 1:length(k_indices_to_plot)
    k_idx = k_indices_to_plot(idx);
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
saveas(gcf, sprintf("../images/cosamp/cosamp_k.png"));
figure;
hold on;
legends = [];
for idx = 1:length(m_indices_to_plot)
    m_idx = m_indices_to_plot(idx);
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
saveas(gcf, sprintf("../images/cosamp/cosamp_m.png"));

toc;

function [Sest, iterations, error] = cosamp_algo(Phi,u,K)
  % CoSaMP: Compressive Sampling Matching Pursuit
    Sest = zeros(size(Phi,2),1);
    tol = 1e-6;
    maxiterations = K;
    v = u;
    t = 1; 
    numericalprecision = 1e-12;
    T = [];
    error=norm(v)/norm(u);
    while (t <= maxiterations) && (error > tol)
      y = abs(Phi'*v);
      [vals,z] = sort(y,'descend');
      Omega = find(y >= vals(2*K) & y > numericalprecision);
      T = union(Omega,T);
      b = pinv(Phi(:,T))*u;
      [vals,z] = sort(abs(b),'descend');
      Kgoodindices = (abs(b) >= vals(K) & abs(b) > numericalprecision);
      T = T(Kgoodindices);
      Sest = zeros(size(Phi,2),1);
      b = b(Kgoodindices);
      Sest(T) = b;
      v = u - Phi(:,T)*b;
      t = t+1;
      error=norm(v)/norm(u);
    end
    iterations=t;
end