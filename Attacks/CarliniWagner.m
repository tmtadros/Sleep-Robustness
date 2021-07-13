function [ x_perturbed ] = CarliniWagner( net, x, y )
%Creates adversarial examples according to the method created by Carlini
%and Wagner in "Towards Evaulating the Robustness of Neural Networks".

% Find true label
[~, true_label] = max(y, [], 2);        
        
% SET up parameters
learningrate = 5e-3; % Larger values converge faster to less accurate results
binary_search_steps = 20; % number of times to adjust the connstant with binary search
max_iterations = 1000; % number of iterations to perform gradient descent
confidence = 0; % how strong adversary should be
initial_const = 0.01;
state.alpha = learningrate;

[~, num_features] = size(x);

min_ = min(x);
max_ = max(x);


att_original = CW_to_attack_space(x, min_, max_);
[reconstructed_original, ~] = CW_to_model_space(att_original, min_, max_);

% Binary search finds the smallest const for which we find an adversarial
% example
const = initial_const;
lower_bound = 0;
upper_bound = Inf;

for i = 1:binary_search_steps
    att_perturbation = zeros(size(att_original));
     
    found_adversarial = 0;
    
    loss_at_previous_check = Inf;
    
    for j = 1:max_iterations
        [x, dxdp] = CW_to_model_space(att_original + att_perturbation, min_, max_);
        net = nnff(net, x, y);
        logits = net.a{end};
        [adv_loss, loss, dldx] = CarliniWagnerLoss(net, const, x, logits, reconstructed_original, ...
                                    true_label, confidence, min_, max_);
        
        if adv_loss <= 0
            found_adversarial = 1;
        else
            found_adversarial = 0;
        end
        
        % backprop the gradient of the loss w.r.t. x further
        % to get the gradient of the loss w.r.t. att_perturbation
        gradient = dldx .* dxdp;
        
        
        [att_perturb, state] = Adam(gradient, state);
        att_perturbation = att_perturbation + att_perturb;

    end
    
    if found_adversarial == 1
        fprintf('Found adversarial with const %f.\n', const);
        upper_bound = const;
    else
        lower_bound = const;
    end

    if upper_bound == Inf
        const = const * 10;
    else
        const = (lower_bound + upper_bound) / 2;
    end

end
[x,~] = CW_to_model_space(att_original + att_perturbation, min_, max_);
x_perturbed = x;


end

