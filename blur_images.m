function [ blur_X ] = blur_images(test_X, sigma)
%blurs images according to imgaussfilt

[num_images, num_features] = size(test_X);
width = sqrt(num_features);
blur_X = zeros(num_images, width, width);

for i = 1:num_images
    blur_X(i,:,:) = imgaussfilt(reshape(test_X(i,:), width, width), sigma);
end

blur_X = reshape(blur_X, num_images, num_features);
end

