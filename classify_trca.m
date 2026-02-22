function results = classify_trca(EEG_epochs, labels, cv_indices, params)
% CLASSIFY_TRCA 任务相关成分分析 (Task-Related Component Analysis)
%
% TRCA通过最大化同类试次间的相关性来提取空间滤波器,
% 特别适用于SSVEP信号的频率识别。
%
% 参考: Nakanishi et al. (2018), "Enhancing Detection of SSVEPs for a
%       High-Speed Brain Speller Using Task-Related Component Analysis"
%
% 输入:
%   EEG_epochs - [n_channels x n_timepoints x n_trials]
%   labels     - [n_trials x 1] 类别标签
%   cv_indices - [n_trials x 1] 交叉验证折号
%   params     - 参数结构体
%
% 输出:
%   results    - 分类结果结构体

    [n_channels, n_timepoints, n_trials] = size(EEG_epochs);
    K = params.K_fold;
    n_classes = params.n_classes;
    
    all_pred = zeros(size(labels));
    fold_acc = zeros(K, 1);
    confusion_total = zeros(n_classes, n_classes);
    
    for k = 1:K
        test_idx  = (cv_indices == k);
        train_idx = (cv_indices ~= k);
        
        X_train = EEG_epochs(:, :, train_idx);
        Y_train = labels(train_idx);
        X_test  = EEG_epochs(:, :, test_idx);
        Y_test  = labels(test_idx);
        
        n_test = sum(test_idx);
        
        %% 对每个类别训练TRCA空间滤波器
        W_trca = cell(n_classes, 1);
        templates = cell(n_classes, 1);
        
        for c = 1:n_classes
            class_data = X_train(:, :, Y_train == c); % [ch x time x trials_c]
            n_c = size(class_data, 3);
            
            % 计算 S 矩阵 (最大化试次间协方差)
            S = zeros(n_channels, n_channels);
            for i = 1:n_c
                for j = i+1:n_c
                    x_i = class_data(:, :, i);
                    x_j = class_data(:, :, j);
                    % 去均值
                    x_i = x_i - mean(x_i, 2);
                    x_j = x_j - mean(x_j, 2);
                    S = S + x_i * x_j' + x_j * x_i';
                end
            end
            
            % 计算 Q 矩阵 (总方差)
            Q = zeros(n_channels, n_channels);
            for i = 1:n_c
                x_i = class_data(:, :, i);
                x_i = x_i - mean(x_i, 2);
                Q = Q + x_i * x_i';
            end
            Q = Q + 1e-6 * eye(n_channels);
            
            % 广义特征值分解: S * w = lambda * Q * w
            [V, D] = eig(S, Q);
            [~, sort_idx] = sort(diag(real(D)), 'descend');
            V = real(V(:, sort_idx));
            
            n_components = min(3, n_channels);
            W_trca{c} = V(:, 1:n_components);
            
            % 计算类别模板 (训练集平均)
            templates{c} = mean(class_data, 3); % [n_channels x n_timepoints]
        end
        
        %% 分类: 计算测试样本与各类别模板在TRCA空间下的相关系数
        pred_labels = zeros(n_test, 1);
        
        for t = 1:n_test
            x_test = X_test(:, :, t); % [n_channels x n_timepoints]
            
            corr_scores = zeros(n_classes, 1);
            for c = 1:n_classes
                w = W_trca{c}; % [n_channels x n_components]
                
                % 空间滤波
                y_test     = w' * x_test;        % [n_comp x n_time]
                y_template = w' * templates{c};   % [n_comp x n_time]
                
                % 计算Pearson相关系数（所有成分的平均相关）
                r_sum = 0;
                for nc = 1:size(w, 2)
                    r = corrcoef(y_test(nc, :), y_template(nc, :));
                    r_sum = r_sum + r(1, 2);
                end
                corr_scores(c) = r_sum / size(w, 2);
            end
            
            [~, pred_labels(t)] = max(corr_scores);
        end
        
        all_pred(test_idx) = pred_labels;
        fold_acc(k) = mean(pred_labels == Y_test);
        
        for i = 1:length(Y_test)
            confusion_total(Y_test(i), pred_labels(i)) = ...
                confusion_total(Y_test(i), pred_labels(i)) + 1;
        end
    end
    
    results.accuracy = mean(fold_acc);
    results.std_acc  = std(fold_acc);
    results.fold_acc = fold_acc;
    results.confusion_matrix = confusion_total;
    results.all_predictions = all_pred;
    results.all_labels = labels;
    
    for c = 1:n_classes
        TP = confusion_total(c, c);
        FP = sum(confusion_total(:, c)) - TP;
        FN = sum(confusion_total(c, :)) - TP;
        results.precision(c) = TP / (TP + FP + eps);
        results.recall(c)    = TP / (TP + FN + eps);
        results.f1_score(c)  = 2 * results.precision(c) * results.recall(c) / ...
                               (results.precision(c) + results.recall(c) + eps);
    end
    results.macro_f1 = mean(results.f1_score);
    
    fprintf('    准确率: %.2f%% (±%.2f%%)\n', results.accuracy*100, results.std_acc*100);
end
