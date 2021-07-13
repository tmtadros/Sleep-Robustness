%% FGSM Test 
% Control network + one of 3 defenses
% 1. Defensive distillation
% 2. Fine-tuning
% 3. Sleep
clear all; close all;
addpath(genpath('../../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../../utils'));
addpath(genpath('../../generalization_utils'));
addpath(genpath('../Attacks'));
addpath(genpath('../../sleep'));

%% Load data and create network architecture
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

for j = 1:3
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

    %% Set up distillation network
    distillation_nn = nn;
    distilatlion_nn.output ='distillation';

    %% Train control network
    control_nn = nntrain(nn, train_x, train_y, opts);

    %% Train distillation network for soft labels
    distillation_nn.Temp = 50;
    distillation_nn = nntrain(distillation_nn, train_x, train_y, opts);

    %% Train distillation network on soft labels
    distillation_nn2 = nnff(distillation_nn, train_x, train_y);
    soft_y = distillation_nn2.a{end};
    distillation_nn = nntrain(distillation_nn, train_x, soft_y, opts);
    distillation_nn.Temp = 1;
    %% Fine-tune control network 
    finetuneX = create_MNIST_FGSM_adversaries(train_x(1:5000,:), train_y(1:5000,:), control_nn);
    finetune_nn = control_nn;
    finetune_nn.learningRate=0.05;
    opts.numepochs = 2;
    newX = [clip(rand(size(finetuneX))/10.*finetuneX+train_x(1:5000,:),0,1); train_x(5001:end,:)];
    finetune_nn = nntrain(finetune_nn, newX, train_y, opts);

    %% sleep network
    x = [39.037119 0.063997 0.069220 36.186585 23.362736 36.384713 2.193698 27105.000000];
    inp_rate=x(1);
    inc=x(2); 
    dec=x(3);
    b1=x(4);
    b2=x(5);
    b3=x(6);
    alpha_scale=x(7);
    sleep_dur=int16(x(8));
    t_opts = struct;
    t_opts.t_ref        = 0.000;
    t_opts.threshold    =   1.0;
    t_opts.dt           = 0.001;
    t_opts.duration     = 0.035;
    t_opts.report_every = 0.001;
    t_opts.max_rate     =  inp_rate;
    %t_opts.max_rate=16;

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

    [norm_nn, norm_constants] = normalize_nn_data(control_nn, train_x);
    sleep_opts.alpha = norm_constants*alpha_scale;
    sleep_input = train_x(1:numiterations,:);
    % Run NREM
    sleep_nn = sleepnn_old(control_nn, numiterations, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales


    %% Create adversaries for each defense
    controlX = create_MNIST_FGSM_adversaries(test_x, test_y, control_nn);
    distilledX = create_MNIST_FGSM_adversaries(test_x, test_y, distillation_nn);
    finetuneX = create_MNIST_FGSM_adversaries(test_x, test_y, finetune_nn);
    sleepX = create_MNIST_FGSM_adversaries(test_x, test_y, sleep_nn);
    
    % Save networks
    save(strcat('FGSM/', 'networks_blurtuned', num2str(j)), 'control_nn', 'distillation_nn', 'finetune_nn', 'sleep_nn');
    
    % Save adversarial examples
    save(strcat('FGSM/', 'adversaries_blurtuned', num2str(j)), 'controlX', 'distilledX', 'finetuneX', 'sleepX');

end