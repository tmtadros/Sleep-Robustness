function [ noise_X ] = add_noise(test_X, mean, var)
%blurs images according to imgaussfilt

[num_images, num_features] = size(test_X);
noise_X = zeros(num_images, 28, 28);

for i = 1:num_images
    noise_X(i,:,:) = imnoise(reshape(test_X(i,:), 28, 28), 'gaussian', mean, var);
end

noise_X = reshape(noise_X, num_images, num_features);
end

