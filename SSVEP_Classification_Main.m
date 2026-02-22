%% SSVEP信号分类主脚本
% 利用9种分类算法对三种条件刺激(A/B/C)下10名被试的SSVEP脑电信号进行分类
%
% 标记含义：
%   11 = A刺激, 21 = B刺激, 31 = C刺激
%   41 = 视觉搜索任务出现时刻（有目标）
%   42 = 视觉搜索任务出现时刻（无目标）
%   12 = 回答正确, 22 = 回答错误, 32 = 无反应
%
% 分类算法：
%   1. SVM (线性核)
%   2. SVM (RBF核)
%   3. Wavelet + SVM
%   4. HDCA (分层判别成分分析)
%   5. xDAWN + SVM
%   6. DCPM + SVM
%   7. ENT + SVM (熵特征)
%   8. TRCA (任务相关成分分析)
%   9. eTRCA (集成TRCA)

clear all; close all; clc;
rng(42);

%% ==================== 参数设置 ====================
params.fs          = 250;       % 采样率 (Hz), 根据实际数据修改
params.n_subjects  = 10;        % 被试数量
params.n_classes   = 3;         % 类别数 (A/B/C三种刺激)
params.epoch_time  = [-0.5, 3]; % 截取时间窗 (s), 相对于刺激onset
params.stim_freqs  = [8, 10, 12]; % 三种刺激对应的SSVEP频率 (Hz), 请根据实际修改
params.markers     = [11, 21, 31]; % 三种刺激的标记
params.K_fold      = 10;        % 交叉验证折数
params.n_harmonics = 3;         % SSVEP谐波数量
params.wavelet_name = 'db4';    % 小波基函数
params.wavelet_level = 5;       % 小波分解层数

% 频带定义
params.freq_bands = struct(...
    'delta', [1 4], ...
    'theta', [4 8], ...
    'alpha', [8 13], ...
    'beta',  [13 30], ...
    'ssvep', [6 40]);

% 方法列表
method_names = {'SVM_Linear', 'SVM_RBF', 'Wavelet_SVM', 'HDCA', ...
                'xDAWN_SVM', 'DCPM_SVM', 'ENT_SVM', 'TRCA', 'eTRCA'};
n_methods = length(method_names);

%% ==================== 数据加载 ====================
fprintf('====== SSVEP信号分类分析 ======\n');
fprintf('正在加载数据...\n');

% ====== 请在此处修改数据加载路径 ======
data_path = './data/';  % 数据文件夹路径

% 初始化存储结构
all_results = struct();

for sub = 1:params.n_subjects
    fprintf('\n========== 被试 %d / %d ==========\n', sub, params.n_subjects);
    
    %% 加载单个被试数据
    % 请根据实际数据格式修改此函数
    [EEG_epochs, labels, chan_locs] = load_subject_data(data_path, sub, params);
    
    % EEG_epochs: [n_channels x n_timepoints x n_trials]
    % labels:     [n_trials x 1], 值为 1, 2, 3 分别对应 A, B, C 刺激
    
    n_channels  = size(EEG_epochs, 1);
    n_timepoints = size(EEG_epochs, 2);
    n_trials    = size(EEG_epochs, 3);
    
    fprintf('  通道数: %d, 时间点: %d, 试次数: %d\n', n_channels, n_timepoints, n_trials);
    fprintf('  各类别试次数: A=%d, B=%d, C=%d\n', ...
        sum(labels==1), sum(labels==2), sum(labels==3));
    
    %% 数据预处理
    EEG_preprocessed = preprocess_eeg(EEG_epochs, params);
    
    %% 交叉验证分折
    cv_indices = crossvalind('Kfold', labels, params.K_fold);
    
    %% ===== 方法1: SVM (线性核) =====
    fprintf('  [1/9] SVM (线性核)...\n');
    features_psd = extract_psd_features(EEG_preprocessed, params);
    results_svm_lin = classify_svm(features_psd, labels, cv_indices, params, 'linear');
    all_results(sub).SVM_Linear = results_svm_lin;
    
    %% ===== 方法2: SVM (RBF核) =====
    fprintf('  [2/9] SVM (RBF核)...\n');
    results_svm_rbf = classify_svm(features_psd, labels, cv_indices, params, 'rbf');
    all_results(sub).SVM_RBF = results_svm_rbf;
    
    %% ===== 方法3: Wavelet + SVM =====
    fprintf('  [3/9] Wavelet + SVM...\n');
    features_wavelet = extract_wavelet_features(EEG_preprocessed, params);
    results_wav_svm = classify_svm(features_wavelet, labels, cv_indices, params, 'linear');
    all_results(sub).Wavelet_SVM = results_wav_svm;
    
    %% ===== 方法4: HDCA =====
    fprintf('  [4/9] HDCA...\n');
    results_hdca = classify_hdca(EEG_preprocessed, labels, cv_indices, params);
    all_results(sub).HDCA = results_hdca;
    
    %% ===== 方法5: xDAWN + SVM =====
    fprintf('  [5/9] xDAWN + SVM...\n');
    results_xdawn = classify_xdawn_svm(EEG_preprocessed, labels, cv_indices, params);
    all_results(sub).xDAWN_SVM = results_xdawn;
    
    %% ===== 方法6: DCPM + SVM =====
    fprintf('  [6/9] DCPM + SVM...\n');
    results_dcpm = classify_dcpm_svm(EEG_preprocessed, labels, cv_indices, params);
    all_results(sub).DCPM_SVM = results_dcpm;
    
    %% ===== 方法7: ENT + SVM =====
    fprintf('  [7/9] ENT + SVM...\n');
    features_ent = extract_entropy_features(EEG_preprocessed, params);
    results_ent_svm = classify_svm(features_ent, labels, cv_indices, params, 'linear');
    all_results(sub).ENT_SVM = results_ent_svm;
    
    %% ===== 方法8: TRCA =====
    fprintf('  [8/9] TRCA...\n');
    results_trca = classify_trca(EEG_preprocessed, labels, cv_indices, params);
    all_results(sub).TRCA = results_trca;
    
    %% ===== 方法9: eTRCA =====
    fprintf('  [9/9] eTRCA...\n');
    results_etrca = classify_etrca(EEG_preprocessed, labels, cv_indices, params);
    all_results(sub).eTRCA = results_etrca;
    
    %% ===== 可选方法10: EEGNet (需要Deep Learning Toolbox) =====
    % 如需运行EEGNet，取消下方注释:
    % fprintf('  [10] EEGNet...\n');
    % results_eegnet = classify_eegnet(EEG_preprocessed, labels, cv_indices, params);
    % all_results(sub).EEGNet = results_eegnet;
    
end

%% ==================== 结果汇总 ====================
fprintf('\n\n====== 分类结果汇总 ======\n');
summarize_and_plot_results(all_results, method_names, params);

%% 保存结果
save('SSVEP_Classification_Results.mat', 'all_results', 'method_names', 'params');
fprintf('\n结果已保存至 SSVEP_Classification_Results.mat\n');
