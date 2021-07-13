function [ x_perturbed ] = BoundaryAttack( net, x, y, test_x, test_y, num_steps)
% Creates an adversarial example of x based on the Boundary ATtack
% algorithm presented in Brendel et al., ICLR 2018.
[~,labels] = max(test_y, [], 2);
[~,attack_class] = max(y, [], 2);

[~, num_features] = size(x);
% Find target and adversarial sample

% Pick random target class
possible_targets = 1:10;
possible_targets(attack_class) = [];
possible_targets = possible_targets(randperm(length(possible_targets)));
target_class = possible_targets(1);

% Pick random image from target class
[im_labels, im_indices] = sort(labels);
label_inds = find(im_labels == target_class);

random_ind = randi(length(label_inds));
actual_label = -1;
while actual_label ~= target_class
    actual_label = nnpredict(net, test_x(im_indices(label_inds(random_ind)),:));
    random_ind = randi(length(label_inds));
end

% Target sample is starting image, adversarial sample is where we want to
% get to (input example)
target_sample = test_x(im_indices(label_inds(random_ind)), :);
adversarial_sample = x;

n_steps = 0;
n_calls = 0;
epsilon = 1.0;
delta = 0.1;

% Move first step to the boundary (forward perturbation between target and
% input such that the example is classified as the input
while 1==1
   perturbation = forward_perturbation(epsilon * norm(adversarial_sample - target_sample), adversarial_sample, target_sample);
   trial_sample = adversarial_sample + perturbation;
   
   predicted_label = nnpredict(net, trial_sample);
   n_calls = n_calls + 1;
   if predicted_label == attack_class
       adversarial_sample = trial_sample;
       break
   else
       epsilon = epsilon * 0.9;
   end
end

while n_steps < num_steps
    d_step = 0;
    while 1 == 1
        d_step = d_step + 1;
        trial_samples = zeros(10,num_features);
        for i = 1:10
            ortho_perturbation = orthogonal_perturbation(delta, adversarial_sample, target_sample);
            trial_samples(i,:) = adversarial_sample + ortho_perturbation;
        end
        labels = nnpredict(net, trial_samples);
        n_calls = n_calls + 10;
        d_score = mean(labels == attack_class);
        if d_score > 0
            if d_score < 0.3
                delta = delta * 0.9;
            elseif d_score > 0.7
                delta = delta / 0.9;
            end
            ind = find(labels == attack_class);
            adversarial_sample = trial_samples(ind(1),:);
            break
        else
            delta = delta * 0.9;
        end
    end
    if mod(n_steps, 50) == 0
        figure;
        for i = 1:10
            subplot(5,2,i)
            imshow(reshape(trial_samples(i,:),28,28)');
            label = nnpredict(net, trial_samples(i,:));
            title(num2str(label-1));
        end
    end

    e_step = 0;
    while 1 == 1
        e_step = e_step + 1;
        forward_perturb = forward_perturbation(epsilon * norm(adversarial_sample - x), adversarial_sample, x);
        trial_sample = adversarial_sample + forward_perturb;
        n_calls = n_calls + 1;
        label = nnpredict(net, trial_sample);
        if label == attack_class
            adversarial_sample = trial_sample;
            epsilon = epsilon / 0.5;
            break
        elseif e_step > 500
            break
        else
            epsilon = epsilon * 0.5;
        end 
    end
    
    n_steps = n_steps + 1;
    if epsilon < 1e-7
        break
    end
end
x_perturbed = adversarial_sample;   
    

end

