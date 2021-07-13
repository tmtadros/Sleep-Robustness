function [ candidate, sphericalCandidate ] = generate_candidates(originalImage, adverserialUpdate, originalImageVector, originalImageDirection, originalImageNorm, epsilon, delta, minDomainValue, maxDomainValue)
% Project point onto a sphere
projection = dot(adverserialUpdate, originalImageDirection);
adverserialUpdate = adverserialUpdate - projection * originalImageDirection;
adverserialUpdate = adverserialUpdate * delta * originalImageNorm / norm(adverserialUpdate);

D = 1.0/(delta^2+1)^(1/2);
direction = adverserialUpdate - originalImageVector;
sphericalCandidate = originalImageVector + D * direction;
sphericalCandidate = clip(sphericalCandidate,minDomainValue, maxDomainValue);

% Add perturbation in direction of source
newOriginalImageDirection = originalImage - sphericalCandidate;
newOriginalImageNorm = norm(newOriginalImageDirection);

% Length of vector assuming spherical candidate to be exactly on sphere
lengthOfSphericalCandidate = epsilon * originalImageNorm;
deviation = newOriginalImageNorm - originalImageNorm; 
lengthOfSphericalCandidate = lengthOfSphericalCandidate + deviation;
lengthOfSphericalCandidate = max([0 lengthOfSphericalCandidate]);
lengthOfSphericalCandidate = lengthOfSphericalCandidate/newOriginalImageNorm;

candidate = sphericalCandidate + lengthOfSphericalCandidate * newOriginalImageDirection;
candidate = clip(candidate, minDomainValue, maxDomainValue);

end

