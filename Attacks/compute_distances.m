function [ dist ] = compute_distances(firstImage, secondImage, normalize, minDomainValue, maxDomainValue)

dist = mean((firstImage - secondImage).^2);
if normalize
    n = prod(size(firstImage));
    normalizer = n * (maxDomainValue - minDomainValue)^2;
    dist = dist/normalizer;
end

end

