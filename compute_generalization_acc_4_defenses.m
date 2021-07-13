function [ noise_acc, blur_acc] = compute_generalization_acc_4_defenses( NNs, testX, testY, noise_test_sets, blur_test_sets )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
numNoiseSets = length(noise_test_sets);
numBlurSets = length(blur_test_sets);

num_networks = length(NNs);
blur_acc = zeros(num_networks, numBlurSets);
noise_acc = zeros(num_networks, numNoiseSets);

for j = 1:num_networks
    nn= NNs{j};
    for i = 1:numNoiseSets
        X = noise_test_sets{i};
        [er, ~] = nntest(nn, X, testY);
        noise_acc(j,i) = (1-er)*100;
    end
    for i = 1:numBlurSets
        X = blur_test_sets{i};
        [er, ~] = nntest(nn, X, testY);
        blur_acc(j,i) = (1-er)*100;

    end
end
end