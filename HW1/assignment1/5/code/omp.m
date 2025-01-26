function sparse_image_reconstruction_omp()
    N = 32;
    image_size = N * N;
    k_values = [5, 50, 200];
    m_values = 100:100:1000;

    DCT_basis = dctmtx(N);
    DCT2_basis = kron(DCT_basis, DCT_basis);

    for k = k_values
        f = generate_sparse_image(DCT2_basis, k, N);
        imwrite(reshape(f, N, N), sprintf('../images/omp/Ground_Truth_k_%d.png', k));

        rmse_vs_m = zeros(size(m_values));

        for idx = 1:length(m_values)
            m = m_values(idx);
            Phi = radamacher_matrix(m, image_size);
            y = Phi * f(:);
            f_hat = omp_algo(Phi, y, k);
            rmse = norm(f(:) - f_hat) / norm(f(:));
            rmse_vs_m(idx) = rmse;

            imwrite(reshape(f_hat, N, N), ...
                sprintf('../images/omp/Reconstructed_k_%d_m_%d.png', k, m));
        end

        figure;
        plot(m_values, rmse_vs_m, '-o');
        xlabel('Number of measurements (m)');
        ylabel('RMSE');
        title(sprintf('RMSE vs m (k = %d)', k));
        saveas(gcf, sprintf('../images/omp/RMSE_vs_m_k_%d.png', k));
        close;
    end

    m_fixed = [500, 700];
    k_values_all = [5, 10, 20, 30, 50, 100, 150, 200];
    for m = m_fixed
        rmse_vs_k = zeros(size(k_values_all));

        for idx = 1:length(k_values_all)
            k = k_values_all(idx);
            f = generate_sparse_image(DCT2_basis, k, N);
            Phi = radamacher_matrix(m, image_size);
            y = Phi * f(:);
            f_hat = omp_algo(Phi, y, k);
            rmse = norm(f(:) - f_hat) / norm(f(:));
            rmse_vs_k(idx) = rmse;
        end

        figure;
        plot(k_values_all, rmse_vs_k, '-o');
        xlabel('Sparsity Level (k)');
        ylabel('RMSE');
        title(sprintf('RMSE vs k (m = %d)', m));
        saveas(gcf, sprintf('../images/omp/RMSE_vs_k_m_%d.png', m));
        close;
    end
end

function f = generate_sparse_image(DCT2_basis, k, N)
    coeffs = zeros(N^2, 1);
    selected_indices = randperm(N^2, k);
    coeffs(selected_indices) = randn(k, 1);
    f = DCT2_basis * coeffs;
end

function Phi = radamacher_matrix(m, n)
    Phi = sign(randn(m, n));
end

function x = omp_algo(Phi, y, k)
    residual = y;
    idx_set = [];
    x = zeros(size(Phi, 2), 1);

    for i = 1:k
        correlations = abs(Phi' * residual);
        [~, idx] = max(correlations);
        idx_set = [idx_set, idx];
        Phi_restricted = Phi(:, idx_set);
        x_restricted = Phi_restricted \ y;
        residual = y - Phi_restricted * x_restricted;
    end

    x(idx_set) = x_restricted;
end
