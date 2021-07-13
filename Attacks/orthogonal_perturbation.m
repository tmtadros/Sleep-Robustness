function [ perturb ] = orthogonal_perturbation(delta, prev_sample, target_sample )
perturb = randn(size(prev_sample));
perturb = perturb / norm(perturb - zeros(size(perturb)));
perturb = perturb * delta * norm(target_sample - prev_sample);

% Project perturbation onto sphere around target
diff = target_sample - prev_sample;
diff = diff / norm(diff);
perturb = perturb - diff * perturb' * diff;

mean = 0.1307;
% Check overflow and underflow

overflow = (prev_sample + perturb) - ones(size(prev_sample)) * (1-mean);
perturb = perturb - overflow .* (overflow > 0);
underflow = ones(size(prev_sample)) * -mean - (prev_sample + perturb);
perturb = perturb + underflow .* (underflow > 0);
end

