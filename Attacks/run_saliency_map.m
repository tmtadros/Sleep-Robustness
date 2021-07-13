function [ idx, pix_sign ] = run_saliency_map( net, adversary, target, num_labels, mask )

[~,num_features] = size(adversary);
alphas = nnbp_deepfool(net, adversary, target, num_labels).*mask;

% Pixel influence on sum of residual classes
betas = zeros(num_labels,num_features);
for i = 1:num_labels
    betas(i,:) = nnbp_deepfool(net, adversary, i, num_labels).*mask - alphas;
end
betas = sum(betas, 1);

% Compute saliency map
salmap = abs(alphas) .* abs(betas) .* sign(alphas.*betas);

% Find optimal pixel and direction of perturbation
[~, idx] = min(salmap);
pix_sign = sign(alphas(idx));

end

