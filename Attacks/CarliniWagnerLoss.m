function [ adv_loss, total_loss, total_loss_grad ] = CarliniWagnerLoss(net, const, x, logits, reconstructed_original, label, confidence, min_, max_ )
%Returns the loss and the gradient of the loss w.r.t. x,
%        assuming that logits = model(x).

% Find best other class (untargeted case)
[~, ind1] = max(logits);
new_logits = logits;
new_logits(ind1) = -Inf;
[~, ind2] = max(new_logits);

c_minimize = label; % true class
c_maximize = ind2; % closest class

% 
is_adv_loss = logits(c_minimize) - logits(c_maximize);


% is_adv is True as soon as the is_adv_loss goes below 0
% but sometimes we want additional confidence
is_adv_loss = is_adv_loss + confidence;
is_adv_loss = max([0, is_adv_loss]);
adv_loss = is_adv_loss * const;
s = max_ - min_;
squared_l2_distance = sum((x-reconstructed_original).^2)/s^2;
total_loss = squared_l2_distance + const*is_adv_loss;

% Calculate the gradient of loss w.r.t x
logits_diff_grad = zeros(size(logits));
logits_diff_grad(c_minimize) = 1;
logits_diff_grad(c_maximize) = -1;
is_adv_loss_grad = nnbp_CW(net, x, logits_diff_grad);

squared_l2_distance_grad = (2/s^2) * (x-reconstructed_original);
total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad;

end

