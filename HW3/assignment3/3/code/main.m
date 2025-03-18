% CryoEM Image Reconstruction using Laplacian Eigenmaps

% The following array stores the projections which we will use:
projection_counts = [50, 100, 500, 1000, 2000, 5000, 10000];

% We need to load the original image and convert it to double
original_image = double(imread("../images/cryoem.png"));

% We need to iterate over each value of N
for i = 1:length(projection_counts)
    tic; % we need to start the timer
    num_projections = projection_counts(i); % getting the corresponding value of n
    fprintf('Processing with %d projections...\n', num_projections);

    true_angles = sort(360 * rand(num_projections, 1)); % generating random numbers which belong to the range [0, 360)

    % We will get the projections at these angles through the inbuilt radon function
    image_projections = radon(original_image, true_angles);
    
    % We will build pairwise distance matrix between projections
    similarity_matrix = buildSimilarityGraph(image_projections);
    
    % We will apply Laplacian eigenmaps to estimate projection angles
    estimated_angles = computeLaplacianEigenmaps(similarity_matrix);
    
    % We will normalize the angles to belong to the range [0, 360)
    estimated_angles = normalizeAngles(estimated_angles);
    
    % We will reconstruct the image at these angles using the iradon function
    reconstructed_image = double(iradon(image_projections, estimated_angles));
    
    % We will crop the image to the origianl image's size
    reconstructed_image = reconstructed_image(1:size(original_image, 1), 1:size(original_image, 2));
    
    % We will correct for rotation (finding the best angle)
    rotation_corrected_image = findOptimalRotation(original_image, reconstructed_image, num_projections);

    % We will compute both RMSE and RRMSE
    original = double(original_image(:)); 
    reconstructed = double(rotation_corrected_image(:)); 
    rmse = norm(original - reconstructed) / sqrt(numel(original));
    rrmse = norm(original - reconstructed) / norm(original);

    elapsed_time = toc;
    fprintf('Completed reconstruction with %d projections in %.2f seconds (RMSE: %.6f) (RRMSE: %.6f)\n\n', ...
        num_projections, elapsed_time, rmse, rrmse);
    
    % We will save the image
    figure('Visible', 'off');
    colormap(gray);
    imagesc(rotation_corrected_image);
    axis image off;
    title(sprintf("N=%d, RMSE: %f RRMSE: %.2f (Time taken: %.2f seconds)", num_projections, rmse, rrmse, elapsed_time));
    saveas(gcf, sprintf("../images/reconstructed_N%d.png", num_projections));
    close(gcf);
end

function similarity_matrix = buildSimilarityGraph(projections)
    % This function builds the similarity matrix using cosine similarity
    num_projections = size(projections, 2);
    similarity_matrix = zeros(num_projections, num_projections);
    
    % We will compute pairwise similarity
    for i = 1:num_projections
        for j = i:num_projections
            % Cosine similarity between the projections
            similarity = sum(projections(:,i) .* projections(:,j)) / ...
                         (norm(projections(:,i)) * norm(projections(:,j)));
            
            % We will also apply gaussian kernel
            similarity = exp(-acos(min(max(similarity, -1), 1))^2 / 0.1);
            
            similarity_matrix(i,j) = similarity;
            similarity_matrix(j,i) = similarity;
        end
    end
    
    % We will just keep k nearest neighbours, and will make others 0
    k = min(15, num_projections-1); % Number of nearest neighbors to keep
    for i = 1:num_projections
        [~, sorted_indices] = sort(similarity_matrix(i,:), 'descend');
        similarity_matrix(i, sorted_indices(k+2:end)) = 0;
    end

    % We will take max of Aij and Aji
    similarity_matrix = max(similarity_matrix, similarity_matrix');
end

function estimated_angles = computeLaplacianEigenmaps(similarity_matrix)
    % This function computes the Laplacian eigenmaps embedding to estimate angles

    % We will compute the graph Laplacian
    degree_matrix = diag(sum(similarity_matrix, 2));
    laplacian = degree_matrix - similarity_matrix;
    
    % We will get the eigne bvectors
    [eigenvectors, eigenvalues] = eig(laplacian, degree_matrix);
    [~, idx] = sort(diag(eigenvalues));
    eigenvectors = eigenvectors(:, idx);
    
    % Embedding using second and third eigen vector
    embedding = [eigenvectors(:, 2), eigenvectors(:, 3)];
    
    % We will convert embeddings to angles
    estimated_angles = atan2(embedding(:, 2), embedding(:, 1));
end

function normalized_angles = normalizeAngles(angles)
    % This function normalizes angles from [-pi, pi) to [0, 360)
    angles = mod(angles, 2*pi);
    normalized_angles = angles * 180 / pi;
end

function optimal_image = findOptimalRotation(reference_image, test_image, projection_count)
    % This function finds the optimal rotation of test_image to match reference_image
    num_angles_try = max(projection_count, 360);
    test_angles = linspace(0, 359, num_angles_try);
    min_error = inf;
    optimal_image = test_image;
    
    % We will try various rotations and find the best one (which minimizes RRMSE)
    for i = 1:length(test_angles)
        angle = test_angles(i);
        rotated_image = imrotate(test_image, angle, "bilinear", "crop");
        current_error = norm(reference_image(:) - rotated_image(:)) / norm(reference_image(:));
        if (current_error < min_error)
            min_error = current_error;
            optimal_image = rotated_image;
        end
    end
    
    % We will also try the flipped rotation
    flipped_image = fliplr(test_image);
    for i = 1:length(test_angles)
        angle = test_angles(i);
        rotated_image = imrotate(flipped_image, angle, "bilinear", "crop");
        current_error = norm(reference_image(:) - rotated_image(:)) / norm(reference_image(:));
        
        if (current_error < min_error)
            min_error = current_error;
            optimal_image = rotated_image;
        end
    end
end