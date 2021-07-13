function [ epsilon, delta ] = update_step_sizes( epsilon, delta, sphericalSuccessProb, stepSuccessProb, stepAdaptationRate, doSpherical )

if doSpherical
    if sphericalSuccessProb > 0.5
        delta = delta * stepAdaptationRate;
        epsilon = epsilon * stepAdaptationRate;
    elseif sphericalSuccessProb < 0.2
        delta = delta / stepAdaptationRate;
        epsilon = epsilon / stepAdaptationRate;
    end
end

if stepSuccessProb > 0.5
    epsilon = epsilon * stepAdaptationRate;
elseif stepSuccessProb < 0.2
    epsilon = epsilon / stepAdaptationRate;
end

end

