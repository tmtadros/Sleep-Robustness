function [ x_perturbed ] = DeepFool(net, x, y)
% Computes an adversarial example x_hat based on the DeepFool Algorithm
% in Moosavi-Dezfooli et al. CVPR 2016

net = nnff(net, x, y);
[dummy, i] = max(net.a{end},[],2);
label = i;


[~, num_classes] = size(y); 
[~, num_features] = size(x);

predicted_label = label;
i = 1;

x_perturbed = x;
residual_labels = 1:num_classes;
residual_labels(label) = [];
while predicted_label == label && i < 100
    gradients = zeros(num_classes-1, num_features);
    losses = zeros(num_classes-1,1);
    distances = zeros(num_classes-1,1);
    for j = 1:length(residual_labels)
        gradients(j,:) = nnbp_deepfool(net, x, residual_labels(j), num_classes) - nnbp_deepfool(net, x_perturbed, label, num_classes);
        losses(j,:) = loss_deepfool(net, x, label, num_classes) - loss_deepfool(net, x, residual_labels(j), num_classes);
        
        % Calculate distances, can choose any norm
        distances(j) = abs(losses(j)) / norm(gradients(j,:), 2);
    end
    [~, I] = min(distances);
    optimal = I;
    
    % Choose optimal loss and gradient
    df = losses(optimal);
    dg = gradients(optimal,:);
    
    % Apply perturbation
    perturbation = abs(df)/ norm(dg,2)^2 * -dg;
    x_perturbed = x_perturbed + 1.05 * perturbation;
    %x_perturbed = clip(x_perturbed, 0, 1);

    i = i + 1;
    % Check for misclassification
    predicted_label = nnpredict(net, x_perturbed);
end

end

