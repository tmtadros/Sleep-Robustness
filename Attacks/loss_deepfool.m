function [ loss ] = loss_deepfool(nn, x, label, num_classes)
% Computes loss w.r.t artficial label
y = zeros(1,num_classes);
y(label) = 1;

% Compute error w.r.t to new label
nn = nnff(nn, x, y);
loss = nn.L;

end

