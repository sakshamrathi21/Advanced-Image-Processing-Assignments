N_vals = [50,100,500,1000,2000,5000,10000];
I = double(imread("../images/cryoem.png"));

for i = 1:length(N_vals)
  tic;
  N = N_vals(i);
  angles = 360 * rand(N,1);
  
  projections = radon(I, angles);
  ordered_projections = sort_projections(projections);
  reconstructed = double(iradon(ordered_projections, linspace(360/N,360,N)));
  reconstructed = reconstructed(1:size(I,1),1:size(I,2));
  
  rotated_recon = fix_global_rotation(I,reconstructed);
  rmse = norm(I(:)-rotated_recon(:)) / norm(I(:));
  
  figure,imshow(rotated_recon, []);
  title(sprintf("RMSE: %f for N=%d", rmse, N));
  saveas(gcf,sprintf("../images/N%d.png",N));
  toc;
end

function res = sort_projections(projections)
  N = size(projections,2);
  res = zeros(size(projections));
  dup_proj = projections;

  res(:,1) = dup_proj(:,1);
  dup_proj(:,1) = [];
  for i = 1:(N-1)
    [~, n1] = min(sum((dup_proj - res(:,i)).^2));
    res(:,i+1) = dup_proj(:,n1);
    dup_proj(:,n1) = [];
  end
end

function res = fix_global_rotation(I,X)
  angles = linspace(1,360,1000);
  min_rmse = 1000000;

  for i = 1:length(angles)
    angle = angles(i);
    rotated_X = imrotate(X,angle,"bilinear","crop");
    current_rmse = norm(I(:)-rotated_X(:)) / norm(I(:));
    if (current_rmse < min_rmse)
      min_rmse = current_rmse;
      res = rotated_X;
    end
  end

  Z = fliplr(X);
  for i = 1:length(angles)
    angle = angles(i);
    rotated_X = imrotate(Z,angle,"bilinear","crop");
    current_rmse = norm(I(:)-rotated_X(:)) / norm(I(:));
    if (current_rmse < min_rmse)
      min_rmse = current_rmse;
      res = rotated_X;
    end
  end
end