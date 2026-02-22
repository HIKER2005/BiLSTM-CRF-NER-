%% SSVEP分类快速演示脚本
% 使用模拟SSVEP数据测试所有分类算法
% 无需真实数据即可运行，用于验证代码正确性

clear all; close all; clc;
rng(42);

fprintf('====== SSVEP信号分类 - 模拟数据演示 ======\n\n');

%% 参数设置
params.fs          = 250;
params.n_subjects  = 3;  % 演示仅用3名模拟被试
params.n_classes   = 3;
params.epoch_time  = [-0.5, 2];
params.stim_freqs  = [8, 10, 12];
params.markers     = [11, 21, 31];
params.K_fold      = 5;  % 演示用5折
params.n_harmonics = 3;
params.wavelet_name = 'db4';
params.wavelet_level = 5;
params.freq_bands = struct('delta',[1 4],'theta',[4 8],'alpha',[8 13],'beta',[13 30],'ssvep',[6 40]);

method_names = {'SVM_Linear', 'SVM_RBF', 'Wavelet_SVM', 'HDCA', ...
                'xDAWN_SVM', 'DCPM_SVM', 'ENT_SVM', 'TRCA', 'eTRCA'};
n_methods = length(method_names);

all_results = struct();

for sub = 1:params.n_subjects
    fprintf('\n========== 模拟被试 %d / %d ==========\n', sub, params.n_subjects);
    
    % 生成模拟数据
    [EEG_epochs, labels, ~] = load_subject_data('./data/', sub, params);
    
    n_channels   = size(EEG_epochs, 1);
    n_timepoints = size(EEG_epochs, 2);
    n_trials     = size(EEG_epochs, 3);
    fprintf('  通道数: %d, 时间点: %d, 试次数: %d\n', n_channels, n_timepoints, n_trials);
    
    % 预处理
    EEG_preprocessed = preprocess_eeg(EEG_epochs, params);
    
    % 交叉验证分折
    cv_indices = crossvalind('Kfold', labels, params.K_fold);
    
    % PSD特征 (SVM共用)
    features_psd = extract_psd_features(EEG_preprocessed, params);
    
    % 方法1: SVM Linear
    fprintf('  [1/9] SVM (线性核)...\n');
    all_results(sub).SVM_Linear = classify_svm(features_psd, labels, cv_indices, params, 'linear');
    
    % 方法2: SVM RBF
    fprintf('  [2/9] SVM (RBF核)...\n');
    all_results(sub).SVM_RBF = classify_svm(features_psd, labels, cv_indices, params, 'rbf');
    
    % 方法3: Wavelet + SVM
    fprintf('  [3/9] Wavelet + SVM...\n');
    features_wavelet = extract_wavelet_features(EEG_preprocessed, params);
    all_results(sub).Wavelet_SVM = classify_svm(features_wavelet, labels, cv_indices, params, 'linear');
    
    % 方法4: HDCA
    fprintf('  [4/9] HDCA...\n');
    all_results(sub).HDCA = classify_hdca(EEG_preprocessed, labels, cv_indices, params);
    
    % 方法5: xDAWN + SVM
    fprintf('  [5/9] xDAWN + SVM...\n');
    all_results(sub).xDAWN_SVM = classify_xdawn_svm(EEG_preprocessed, labels, cv_indices, params);
    
    % 方法6: DCPM + SVM
    fprintf('  [6/9] DCPM + SVM...\n');
    all_results(sub).DCPM_SVM = classify_dcpm_svm(EEG_preprocessed, labels, cv_indices, params);
    
    % 方法7: ENT + SVM
    fprintf('  [7/9] ENT + SVM...\n');
    features_ent = extract_entropy_features(EEG_preprocessed, params);
    all_results(sub).ENT_SVM = classify_svm(features_ent, labels, cv_indices, params, 'linear');
    
    % 方法8: TRCA
    fprintf('  [8/9] TRCA...\n');
    all_results(sub).TRCA = classify_trca(EEG_preprocessed, labels, cv_indices, params);
    
    % 方法9: eTRCA
    fprintf('  [9/9] eTRCA...\n');
    all_results(sub).eTRCA = classify_etrca(EEG_preprocessed, labels, cv_indices, params);
end

%% 结果汇总与可视化
fprintf('\n\n');
summarize_and_plot_results(all_results, method_names, params);

save('Demo_Results.mat', 'all_results', 'method_names', 'params');
fprintf('\n演示完成! 结果已保存至 Demo_Results.mat\n');
