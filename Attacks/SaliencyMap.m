function [ adversarialX ] = SaliencyMap( net, x, y, theta, max_iter, max_perturbations_per_pixel )
% Computes an adversarial example of the input x based on the algorithm
% described in "The Limitations of Deep Learning in Adversarial Settings"
% by Papernot et. al 2016.

[~, num_classes] = size(y);
[~, original_class] = max(y, [], 2);

net = nnff(net, x, y);
logits = net.a{end};

[~, ind1] = max(logits);
new_logits = logits;
new_logits(ind1) = -Inf;
[~, ind2] = max(new_logits);
target_class = ind2;

min_ = min(x);
max_ = max(x);

adversarialX = x;
% the mask defines the search domain
% each modified pixel with border value is set to zero in mask
mask = ones(size(x));
% count tracks how often each pixel was changed
counts = zeros(size(x));

for i = 1:max_iter
    prediction = nnpredict(net, adversarialX);
    if prediction ~= original_class
        break
    end
    
    % Get pixel location with highest influence on class
    [idx,p_sign] = run_saliency_map(net, adversarialX, target_class, num_classes, mask);
    
    % apply perturbation
    adversarialX(idx) = adversarialX(idx) - p_sign * theta * (max_ - min_);
    
    % track number of updates for each pixel
    counts(idx) = counts(idx) + 1;
    
    % remove index from search domain if it exceeds bounds
    if adversarialX(idx) <= min_ || adversarialX(idx) >= max_
        mask(idx) = 0;
    end
    
    % remove pixel if changed too often
    if counts(idx) >= max_perturbations_per_pixel
        mask(idx) = 0;
    end
    adversarialX = clip(adversarialX, min_, max_);
end

end

