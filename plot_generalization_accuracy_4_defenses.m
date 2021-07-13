function [ f ] = plot_generalization_accuracy_4_defenses( noisy_acc, blur_acc, noise_levels, blur_levels )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
f = figure();
subplot(121)
errorbar(noise_levels, squeeze(mean(noisy_acc(:,1,:))), squeeze(std(noisy_acc(:,1,:))), 'LineWidth', 2); hold on
errorbar(noise_levels, squeeze(mean(noisy_acc(:,2,:))), squeeze(std(noisy_acc(:,2,:))), 'LineWidth', 2); hold on
errorbar(noise_levels, squeeze(mean(noisy_acc(:,3,:))), squeeze(std(noisy_acc(:,3,:))), 'LineWidth', 2); hold on
errorbar(noise_levels, squeeze(mean(noisy_acc(:,4,:))), squeeze(std(noisy_acc(:,4,:))), 'LineWidth', 2); hold on
errorbar(noise_levels, squeeze(mean(noisy_acc(:,5,:))), squeeze(std(noisy_acc(:,5,:))), 'LineWidth', 2); hold on

legend('Control', 'Defensive Distillation', 'Finetuning-noise', 'Finetuning-blur', 'Sleep')
ylabel('Accuracy', 'FontSize', 20)
xlabel('Noise std', 'FontSize', 20)
title('Noise generalization', 'FontSize', 20)
subplot(122)
errorbar(blur_levels, squeeze(mean(blur_acc(:,1,:))), squeeze(std(blur_acc(:,1,:))), 'LineWidth', 2); hold on
errorbar(blur_levels, squeeze(mean(blur_acc(:,2,:))), squeeze(std(blur_acc(:,2,:))), 'LineWidth', 2); hold on
errorbar(blur_levels, squeeze(mean(blur_acc(:,3,:))), squeeze(std(blur_acc(:,3,:))), 'LineWidth', 2); hold on
errorbar(blur_levels, squeeze(mean(blur_acc(:,4,:))), squeeze(std(blur_acc(:,4,:))), 'LineWidth', 2); hold on
errorbar(blur_levels, squeeze(mean(blur_acc(:,5,:))), squeeze(std(blur_acc(:,5,:))), 'LineWidth', 2); hold on

ylabel('Accuracy', 'FontSize', 20)
xlabel('Sigma', 'FontSize', 20)
title('Blur generalization', 'FontSize', 20)

end
