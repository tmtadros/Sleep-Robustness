%% MNIST Adversarial Learning Example
%    Load paths
clear all; close all;
addpath(genpath('../../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../../utils'));
addpath(genpath('../../generalization_utils'));
addpath(genpath('../Attacks'));

%% load mnist_uint8 and create noisy datasets for test set;
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% Set up neural network
%Initialize net
nn = nnsetup([784 1200 1200 10]);
% Rescale weights for ReLU
for i = 2 : nn.n   
    % Weights - choose between [-0.1 0.1]
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
    nn.vW{i - 1} = zeros(size(nn.W{i-1}));
end

% ReLU Train
% Set up learning constants
nn.activation_function = 'relu';
nn.output ='relu';
nn.learningRate = 0.1;
nn.momentum = 0.5;
nn.dropoutFraction = 0.2;
nn.learn_bias = 0;
opts.numepochs =  2;
opts.batchsize = 100;
%% 
num_examples = 5000;
nn1 = nntrain(nn, train_x, train_y, opts);
[er,bad] = nntest(nn1, test_x, test_y);

%% Create adversarial examples based on current network
adversarialX = create_MNIST_FGSM_adversaries(test_x, test_y, nn1);

%% Plot adversarial examples


figure;
for i = 1:10
    plot_im = i;
    subplot(5,2,i)
    imshow(reshape(test_x(plot_im,:) + 0.25*adversarialX(plot_im,:), 28, 28)')
    labels = nnpredict(nn1, test_x(plot_im,:) + 0.25*adversarialX(plot_im,:));
    title(num2str(labels-1))
end

figure;
for i = 1:10
    plot_im = i;
    subplot(5,2,i)
    imshow(reshape(test_x(plot_im,:), 28, 28)')
    labels = nnpredict(nn1, test_x(plot_im,:));
    title(num2str(labels-1))
end

%%
[~, real_labels] = max(test_y, [], 2);
etas = 0:0.005:0.4
acc = zeros(length(etas),1);
for i = 1:length(etas)
    labels = nnpredict(nn1, clip(test_x + etas(i)*adversarialX, 0, 1));
    acc(i) = sum(labels == real_labels)/length(labels);
end
figure;
plot(etas, acc, 'LineWidth', 2)
xlabel('Eta', 'FontSize', 20)
ylabel('Accuracy', 'FontSize', 20)

%% Figure of image vs. eta
figure;
for i = 1:length(etas)
    subplot(4,21,i)
    imshow(reshape(clip(test_x(1,:) + etas(i) * adversarialX(1,:),0,1), 28,28)');
    title(num2str(etas(i)))
end
    
    
    
