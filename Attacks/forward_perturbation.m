function [ perturb ] = forward_perturbation(epsilon, prev_sample, target_sample)
perturb = target_sample - prev_sample;
perturb = perturb / norm(perturb);
perturb = perturb * epsilon;
end

