function [ signed_eta ] = fast_gradient_sign_adversary(net, label, x)
%Computes the sign of the gradient with respect to the input and returns
%this value

net.testing = 1;
% get activations for example image
net = nnff(net, x, label);
derivative = nnbp_adversary(net,x);
net.testing = 0;

signed_eta = sign(derivative);
end

