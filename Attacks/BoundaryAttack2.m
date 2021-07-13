function [ adversarialImage ] = BoundaryAttack2( net, x, y, num_steps)
% Creates an adversarial example of x based on the Boundary ATtack
% algorithm presented in Brendel et al., ICLR 2018.


% Parameters
numDirectionsToExplore = 25;
epsilon = 0.5;
delta = 0.5;
stepAdaptationRate = 0.5;
convergenceThreshold = 1e-7;
minDomainValue = min(x);
maxDomainValue = max(x);
normalize = 0;

[~,originalClass] = max(y, [], 2);
originalPrediction = nnpredict(net, x);
originalImage = x;

% Initialize the adversarial example
adversarialLabel = originalClass;
while adversarialLabel == originalClass
    adversarialImage = rand(size(x));
    adversarialLabel = nnpredict(net, adversarialImage);
    fprintf('Image successfull initialized!\n');
    fprintf('Original image label %d, Original image prediction %d, Adversarial image prediction %d\n',originalClass-1, originalPrediction-1, adversarialLabel-1); 
end

% Perform updates on adversarial image so create copy
initialAdversarialImage = adversarialImage;


% Iterate over num_steps to perform updates
n_steps = 0;
while n_steps < num_steps
    doSpherical = mod(n_steps, 10) == 0;
    
    % Variables for statistics
    numSuccessSpherical = 0;
    numSuccessSteps = 0;
    numTotalSteps = numDirectionsToExplore;
    
    % Compute unit vector pointing from original to adversarial image
    [originalImageVector, originalImageDirection, originalImageNorm] = compute_original_image_direction(originalImage, adversarialImage);
    distance = compute_distances(originalImage, adversarialImage, normalize, minDomainValue, maxDomainValue);
    fprintf('Distance between original and adversarial image: %f\n', distance);
    
    % check if adversarial example converged
    if epsilon < convergenceThreshold
        fprintf('Attack converged after %d steps\n', n_steps);
        convergenceStep = n_steps - 1;
        adversarialImagePredictedLabel = nnpredict(net, adversarialImage);
        break
    end
    
    newAdversarialImageDist = Inf;
    newAdversarialImage = zeros(size(originalImage));
    
    for i = 1:numDirectionsToExplore
        % Sample adversarial update from iid distributed with range [0,1]
        adversarialUpdate = rand(size(originalImage));
        
        % Generate candidates based on input
        [candidate, sphericalCandidate] = generate_candidates(originalImage, adversarialUpdate, originalImageVector, originalImageDirection, originalImageNorm, epsilon, delta, minDomainValue, maxDomainValue);
        
        if doSpherical
            % check if spherical candidate is adversarial
            sphericalCandidatePredictedLabel = nnpredict(net, sphericalCandidate);
            isCandidateAdversarial = 0;
            if sphericalCandidatePredictedLabel ~= originalClass
                numSuccessSpherical = numSuccessSpherical + 1;
                candidatePredictedLabel = nnpredict(net, candidate);
                isCandidateAdversarial = originalClass ~= candidatePredictedLabel;
            else
                continue
            end
            
            if isCandidateAdversarial
                currentDist = compute_distances(originalImage, candidate, normalize, minDomainValue, maxDomainValue);
                if currentDist < newAdversarialImageDist
                    newAdversarialImageDist = currentDist;
                    numSuccessSteps = numSuccessSteps + 1;
                    newAdversarialImage = candidate;
                    newCandidatePredictedLabel = candidatePredictedLabel;
                end
            end
        else
            candidatePredictedLabel = nnpredict(net, candidate);
            isCandidateAdversarial = originalClass ~= candidatePredictedLabel;
            if isCandidateAdversarial
                currentDist = compute_distances(originalImage, candidate, normalize, minDomainValue, maxDomainValue);
                if currentDist < newAdversarialImageDist
                    newAdversarialImageDist = currentDist;
                    numSuccessSteps = numSuccessSteps + 1;
                    newAdversarialImage = candidate;
                    newCandidatePredictedLabel = candidatePredictedLabel;
                end
            end
        end     
    end
    
    % Handle found adversarial example
    if sum(newAdversarialImage) ~= 0
        if newAdversarialImageDist > distance
            fprintf('Warning, new adversarial image has larger distane than original\n');
        else
            absoluteImprovement = distance - newAdversarialImageDist;
            relativeImprovement = absoluteImprovement/distance;
            fprintf('Absolute improvement: %f, relative improvement: %f\n', absoluteImprovement, relativeImprovement);
            adversarialImage = newAdversarialImage;
            distance = newAdversarialImageDist;
        end
    end
    
    % Update epsilon and delta based on success probability
    sphericalSuccessProbability = numSuccessSpherical/numTotalSteps;
    stepSuccessProbability = numSuccessSteps/numTotalSteps;
    fprintf('Step: %d, Total Attempts: %d, Successful Attempts (spherical): %d, Successful Attempts (candidate): %d, Spherical success prob: %f, Step success prob: %f\n',...
            n_steps, numTotalSteps, numSuccessSpherical, numSuccessSteps, sphericalSuccessProbability, stepSuccessProbability);
    
    [epsilon, delta] = update_step_sizes(epsilon, delta, sphericalSuccessProbability, stepSuccessProbability, stepAdaptationRate, doSpherical);
    n_steps = n_steps + 1;
end
adversarialLabel = nnpredict(net, adversarialImage);
fprintf("Original image label: %d | Original image prediction: %d | Final adverserial image prediction: %d\n",...
		originalClass - 1, originalPrediction - 1, adversarialLabel - 1);
end