function main()
    n1 = 800;
    n2 = 900;
    
    rank_values = [10, 30, 50, 75, 100, 125, 150, 200];
    sparsity_fractions = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15];
    num_trials = 15;
    success_prob = zeros(length(rank_values), length(sparsity_fractions));
    successful_case = [];
    unsuccessful_case = [];
    
    for i = 1:length(rank_values)
        r = rank_values(i);
        
        for j = 1:length(sparsity_fractions)
            fs = sparsity_fractions(j);
            
            fprintf('Testing r = %d, fs = %.3f\n', r, fs);
            
            successes = 0;
            
            for trial = 1:num_trials
                [L_true, S_true, M] = generate_test_data(n1, n2, r, fs);
                lambda = 1/sqrt(max(n1, n2));
                tic;
                [L_hat, S_hat] = rpca_alm(M, lambda);
                elapsed_time = toc;
                L_error = norm(L_true - L_hat, 'fro') / norm(L_true, 'fro');
                S_error = norm(S_true - S_hat, 'fro') / norm(S_true, 'fro');
                
                is_success = (L_error <= 0.001) && (S_error <= 0.001);
                
                if is_success
                    successes = successes + 1;
                end
                
                if isempty(successful_case) && is_success
                    successful_case.r = r;
                    successful_case.fs = fs;
                    successful_case.L_true = L_true;
                    successful_case.S_true = S_true;
                    successful_case.L_hat = L_hat;
                    successful_case.S_hat = S_hat;
                end
                
                if isempty(unsuccessful_case) && ~is_success && trial > 5
                    unsuccessful_case.r = r;
                    unsuccessful_case.fs = fs;
                    unsuccessful_case.L_true = L_true;
                    unsuccessful_case.S_true = S_true;
                    unsuccessful_case.L_hat = L_hat;
                    unsuccessful_case.S_hat = S_hat;
                end
                
                fprintf('  Trial %d: Success = %d, Time = %.2f s, L_error = %.2e, S_error = %.2e\n', trial, is_success, elapsed_time, L_error, S_error);

            end
            success_prob(i, j) = successes / num_trials;
        end
    end

    fig = figure('Position', [100, 100, 800, 600]);
    imagesc(success_prob);
    colormap('gray');
    colorbar;
    
    xticks(1:length(sparsity_fractions));
    yticks(1:length(rank_values));
    xticklabels(sparsity_fractions);
    yticklabels(rank_values);
    
    xlabel('Sparsity Fraction (fs)');
    ylabel('Rank (r)');
    title('Success Probability of RPCA-ALM');
    saveas(fig, 'success_probability_heatmap.png');
    
    if ~isempty(successful_case)
        visualize_case(successful_case, 'Successful Case');
    end
    
    if ~isempty(unsuccessful_case)
        visualize_case(unsuccessful_case, 'Unsuccessful Case');
    end
    
end

function [L, S, M] = generate_test_data(n1, n2, r, fs)
    A = randn(n1, r);
    B = randn(n2, r);
    L = A * B';
    
    L = L / norm(L, 'fro') * n1;
    
    S = zeros(n1, n2);
    s = round(fs * n1 * n2);
    
    idx = randperm(n1 * n2, s);
    [I, J] = ind2sub([n1, n2], idx);
    
    for k = 1:length(I)
        S(I(k), J(k)) = 3 * randn(); 
    end

    M = L + S;
end

% function [L, S] = rpca_alm(D, lambda)
%     [m, n] = size(D);
%     Y = D / max(norm(D, 2), lambda^(-1) * norm(D, Inf));
%     L = zeros(m, n);
%     S = zeros(m, n);
%     mu = 1.25 / norm(D, 2);
%     mu_bar = mu * 1e7;
%     rho = 1.5;
%     tol = 1e-7;
%     max_iter = 500;
    
%     iter = 0;
%     converged = false;
    
%     while ~converged && iter < max_iter
%         iter = iter + 1;
%         temp = D - S + Y/mu;
%         [U, Sigma, V] = svd(temp, 'econ');
%         sigma = diag(Sigma);
%         svt = soft_threshold(sigma, 1/mu);
%         rank_L = sum(svt > 0);
%         L = U(:, 1:rank_L) * diag(svt(1:rank_L)) * V(:, 1:rank_L)';
%         temp = D - L + Y/mu;
%         S = sign(temp) .* max(abs(temp) - lambda/mu, 0);
%         Z = D - L - S;
%         Y = Y + mu * Z;
%         mu = min(rho * mu, mu_bar);
%         err = norm(Z, 'fro') / norm(D, 'fro');
%         if err < tol
%             converged = true;
%         end
%     end
% end

function [L, S] = rpca_alm(D, lambda)
    mu = 25 * lambda;
    tol = 1e-5;
    max_iter = 1000;
    [L, S] = RobustPCA(D, lambda, mu, tol, max_iter);
end

function [L, S] = RobustPCA(X, lambda, mu, tol, max_iter)
    [M, N] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');
    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);
    for iter = 1:max_iter
        L = D_op(1/mu, X - S + (1/mu)*Y);
        S = soft_thresh(lambda/mu, X - L + (1/mu)*Y);
        Z = X - L - S;
        Z(unobserved) = 0;
        Y = Y + mu*Z;
        err = norm(Z, 'fro') / normX;
        if err < tol
            break;
        end
    end
end

function r = soft_thresh(tau, X)
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = D_op(tau, X)
    [U, S, V] = svd(X, 'econ');
    S_thresh = soft_thresh(tau, S);
    r = U * S_thresh * V';
end


% function y = soft_threshold(x, t)
%     y = sign(x) .* max(abs(x) - t, 0);
% end

function visualize_case(case_data, title_str)
    fig = figure('Position', [100, 100, 1200, 300]);
    subplot(2, 4, 1);
    imagesc(case_data.L_true(1:200, 1:200));
    title('True L (subset)');
    colorbar;
    
    subplot(2, 4, 2);
    imagesc(case_data.S_true(1:200, 1:200));
    title('True S (subset)');
    colorbar;
    
    subplot(2, 4, 3);
    imagesc(case_data.L_true(1:200, 1:200) + case_data.S_true(1:200, 1:200));
    title('M = L + S (subset)');
    colorbar;
    
    subplot(2, 4, 5);
    imagesc(case_data.L_hat(1:200, 1:200));
    title('Estimated L (subset)');
    colorbar;
    
    subplot(2, 4, 6);
    imagesc(case_data.S_hat(1:200, 1:200));
    title('Estimated S (subset)');
    colorbar;
    
    subplot(2, 4, 7);
    L_error = norm(case_data.L_true - case_data.L_hat, 'fro') / norm(case_data.L_true, 'fro');
    S_error = norm(case_data.S_true - case_data.S_hat, 'fro') / norm(case_data.S_true, 'fro');
    imagesc(abs(case_data.L_true(1:200, 1:200) - case_data.L_hat(1:200, 1:200)));
    title(['L Error: ' num2str(L_error, '%.6f')]);
    colorbar;
    
    subplot(2, 4, 8);
    imagesc(abs(case_data.S_true(1:200, 1:200) - case_data.S_hat(1:200, 1:200)));
    title(['S Error: ' num2str(S_error, '%.6f')]);
    colorbar;
    
    % suptitle([title_str ': r = ' num2str(case_data.r) ', fs = ' num2str(case_data.fs)]);
    sgtitle([title_str ': r = ' num2str(case_data.r) ', fs = ' num2str(case_data.fs)]);

    % Save figure with descriptive name
    filename = [lower(strrep(title_str, ' ', '_')) '_r' num2str(case_data.r) '_fs' num2str(case_data.fs) '.png'];
    saveas(fig, filename);
end