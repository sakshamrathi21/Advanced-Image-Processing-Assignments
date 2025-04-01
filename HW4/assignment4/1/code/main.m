clc; clear; close all;

% Define the parameters for the system
N = 500;
M = 300;
sparsityLevels = [5, 10, 15, 20];
lambdaValues = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 100];

% Generate the Sensing Matrix Phi as Bernoulli (+- 1/sqrt(M))
Phi = (2 * (rand(M, N) > 0.5) - 1) / sqrt(M);

% Initialize the error vector
validationError = zeros(length(sparsityLevels), length(lambdaValues));
RMSE = zeros(length(sparsityLevels), length(lambdaValues));

for sparsityIndex = 1:length(sparsityLevels)
    sparsity = sparsityLevels(sparsityIndex);
    % Generate the sparse signal x with the given sparsity levels
    theta = zeros(N, 1);
    nonZeroIndices = randperm(N, sparsity);
    % Define its values in Uniform distribution [0, 1000]
    theta(nonZeroIndices) = rand(sparsity, 1) * 1000;
    % Generate the measurement with noise
    yOriginal = Phi * theta;
    sigma = 0.025 * mean(abs(yOriginal));
    noise = sigma * randn(M, 1);
    y = yOriginal + noise;
    % Split the data in Reconstruction(90%) and Validation(10%)
    permutation = randperm(M);
    % Reconstruction set indices
    Rindices = permutation(1:round(0.9 * M));
    % Validation set indices
    Vindices = permutation(1 + round(0.9 * M):M);
    % Split the matrix and the vector
    PhiR = Phi(Rindices, :);
    PhiV = Phi(Vindices, :);
    yR = y(Rindices);
    yV = y(Vindices);
    % Loop over the different lambda values
    for lambdaIndex = 1:length(lambdaValues)
        lambda = lambdaValues(lambdaIndex);
        % Solve LASSO using CVX
        disp("Solving for " + sparsity + " sparsity and lambda = " + lambda);
        cvx_clear
        cvx_begin quiet
            variable x(N)
            minimize(square_pos(norm(yR - PhiR * x, 2)) + lambda * norm(x, 1))
        cvx_end
        % Compute the validation error
        validationError(sparsityIndex, lambdaIndex) = norm(yV - PhiV * x, 2)^2 / length(Vindices);
        % Compute the RMSE = ||x^ - x|| / ||x||
        RMSE(sparsityIndex, lambdaIndex) = norm(x - theta, 2) / norm(theta, 2);
    end
end

%%
% directoryPath = fullfile('..', 'report');
% Plot the validation errors for different sparsity levels in the same figure
figure;
hold on;
for sparsityIndex = 1:length(sparsityLevels)
    plot(lambdaValues, validationError(sparsityIndex, :), 'DisplayName', ['Sparsity = ' num2str(sparsityLevels(sparsityIndex))], 'LineStyle', '-', 'LineWidth', 1, 'Marker', 'o', 'MarkerSize', 3);
end
hold off;
xlabel('Lambda Values');
ylabel('Validation Error');
title('Validation Error vs Lambda for Different Sparsity Levels');
legend('show');
grid on;
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
% saveas(gcf, fullfile(directoryPath, 'validationError.png'));


%%

% Plot the RMSE for different sparsity levels in the same figure
figure;
hold on;
for sparsityIndex = 1:length(sparsityLevels)
    plot(lambdaValues, RMSE(sparsityIndex, :), 'DisplayName', ['Sparsity = ' num2str(sparsityLevels(sparsityIndex))], 'LineStyle', '-', 'LineWidth', 1, 'Marker', 'o', 'MarkerSize', 3);
end
hold off;
xlabel('Lambda Values');
ylabel('RMSE');
title('RMSE vs Lambda for Different Sparsity Levels');
legend('show');
grid on;
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
%saveas(gcf, '../report/rmse.png');

%%

% Plot in the Linear Scale for the RMSE and the Validation Error
figure;
hold on;
for sparsityIndex = 1:length(sparsityLevels)
    plot(lambdaValues, RMSE(sparsityIndex, :), 'DisplayName', ['Sparsity = ' num2str(sparsityLevels(sparsityIndex))], 'LineStyle', '-', 'LineWidth', 1, 'Marker', 'o', 'MarkerSize', 3);
end
hold off;
xlabel('Lambda Values');
ylabel('RMSE');
title('RMSE vs Lambda for Different Sparsity Levels (Linear Scale)');
legend('show');
grid on;
% set(gca, 'XScale', 'log');
% set(gca, 'YScale', 'log');
% saveas(gcf, fullfile(directoryPath, 'rmseLinear.png'));

%%

figure;
hold on;
for sparsityIndex = 1:length(sparsityLevels)
    plot(lambdaValues, validationError(sparsityIndex, :), 'DisplayName', ['Sparsity = ' num2str(sparsityLevels(sparsityIndex))], 'LineStyle', '-', 'LineWidth', 1, 'Marker', 'o', 'MarkerSize', 3);
end
hold off;
xlabel('Lambda Values');
ylabel('Validation Error');
title('Validation Error vs Lambda for Different Sparsity Levels (Linear Scale)');
legend('show');
grid on;
% set(gca, 'XScale', 'log');
% set(gca, 'YScale', 'log');

%%

% Use the Mozorov Discrepancy Principle to find the best lambda for all sparsity levels
function discrepancyArray = selectLambda(y, Phi, lambda, sigma2)
    M = size(y, 1);
    target = M * sigma2;
    discrepancyArray = zeros(length(lambda), 1);

    for i = 1:length(lambda)
        l = lambda(i);
        disp("Solving for lambda = " + l);
        cvx_clear
        cvx_begin quiet
            variable x(size(Phi, 2))
            minimize(square_pos(norm(y - Phi * x, 2)) + l * norm(x, 1))
        cvx_end

        % Compute the discrepancy
        discrepancy = abs(norm(y - Phi * x, 2)^2 - target);
        discrepancyArray(i) = discrepancy;
    end
end

% Find the best lambda for each sparsity level
discrepancy = zeros(length(sparsityLevels), length(lambdaValues));
for sparsityIndex = 1:length(sparsityLevels)
    sparsity = sparsityLevels(sparsityIndex);
    disp("Sparsity = " + sparsity);
    % Generate the sparse signal x with the given sparsity levels
    theta = zeros(N, 1);
    nonZeroIndices = randperm(N, sparsity);
    % Define its values in Uniform distribution [0, 1000]
    theta(nonZeroIndices) = rand(sparsity, 1) * 1000;
    % Generate the measurement with noise
    yOriginal = Phi * theta;
    sigma = 0.025 * mean(abs(yOriginal));
    noise = sigma * randn(M, 1);
    y = yOriginal + noise;
    % Compute the discrepancy for each lambda value
    discrepancy(sparsityIndex, :) = selectLambda(y, Phi, lambdaValues, sigma^2);
end

% Plot the discrepancy for different sparsity levels in the same figure
figure;
hold on;
for sparsityIndex = 1:length(sparsityLevels)
    plot(lambdaValues, discrepancy(sparsityIndex, :), 'DisplayName', ['Sparsity = ' num2str(sparsityLevels(sparsityIndex))], 'LineStyle', '-', 'LineWidth', 1, 'Marker', 'o', 'MarkerSize', 3);
end
hold off;
xlabel('Lambda Values');
ylabel('Discrepancy');
title('Discrepancy vs Lambda for Different Sparsity Levels');
legend('show');
grid on;
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');





