function [ originalImageVector, originalImageDirection, originalImageNorm ] = compute_original_image_direction( firstImage, secondImage )
originalImageVector = firstImage - secondImage;
originalImageNorm = norm(originalImageVector);
originalImageDirection = originalImageVector / originalImageNorm;
end

