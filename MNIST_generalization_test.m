%%
clear all; close all;
addpath(genpath('../../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../../utils'));
addpath(genpath('../../sleep'));
addpath(genpath('../Attacks'));
addpath(genpath('../MNIST'));
addpath('./');


%% setup random for repeatability
%rng('default');
%rng(1000);
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
%% Blur images 
blur_level = [0.0001, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
blur_test_sets = {};
for i = 1:length(blur_level)
    blur_test_sets{i} = clip(blur_images(test_x, blur_level(i)),0,1);
end
noise_level = [0.0 0.1, 0.3, 0.5, 0.7, 0.9];
noise_test_sets = {};
for i = 1:length(noise_level)
    noise_test_sets{i} = clip(add_noise(test_x, 0, noise_level(i)),0,1);
end

noisyacc = zeros(5,2, length(noise_level));
bluracc = zeros(5,2,length(blur_level));
%% Set up neural network
%Initialize net
for j = 1:1
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
    num_examples = 50000;
    nn1 = nntrain(nn, train_x(1:num_examples,:), train_y(1:num_examples,:), opts);

    %%
    x = [39.037119 0.063997 0.069220 36.186585 23.362736 36.384713 2.193698 27105.000000];
    inp_rate=x(1);
    inc=x(2); 
    dec=x(3);
    b1=x(4);
    b2=x(5);
    b3=x(6);
    alpha_scale=x(7);
    sleep_dur=int16(x(8));
    sleep_opts.DC=0;
    t_opts = struct;
    t_opts.t_ref        = 0.000;
    t_opts.threshold    =   1.0;
    t_opts.dt           = 0.001;
    t_opts.duration     = 0.035;
    t_opts.report_every = 0.001;
    t_opts.max_rate     =  inp_rate;

    sleep_opts.beta = [b1 b2 b3];
    % sleep_opts.alpha = [2.50 4.0 7.5]*1.25; % -- This is reset later
    sleep_opts.decay = 0.999; 

    sleep_opts.W_inh=0.0;
    sleep_opts.normW = 0;
    sleep_opts.inc = inc;
    sleep_opts.dec = dec;

    sleep_opts.delta_min=1000;
    sleep_opts.delta_max=1000;

    sleep_opts.theta = 0.0;
    numiterations = sleep_dur;

    [~, norm_constants] = normalize_nn_data(nn1, train_x(1:num_examples,:));
    sleep_opts.alpha = norm_constants*alpha_scale;
    sleep_input = train_x(1:numiterations,:);
    % Run NREM
    Snn = sleepnn_old(nn1, numiterations, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales
    %% Plot results
    [accuracy, noise_acc, blur_acc] = compute_generalization_acc({nn1, Snn}, test_x, test_y, noise_test_sets, blur_test_sets);
    bluracc(j,:,:) = blur_acc;
    noisyacc(j,:,:) = noise_acc;
end
%%
plot_generalization_accuracy(accuracy, noisyacc, bluracc, noise_level, blur_level)
