function results = classify_hdca(EEG_epochs, labels, cv_indices, params)
% CLASSIFY_HDCA 分层判别成分分析 (Hierarchical Discriminant Component Analysis)
%
% HDCA通过两个层级实现分类：
%   层1: 对每个时间窗口内的各通道数据计算空间权重
%   层2: 对时间窗口的加权值训练线性分类器
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
    
    % HDCA时间窗口参数
    win_size = round(0.1 * params.fs); % 100ms窗口
    win_step = round(0.05 * params.fs); % 50ms步进
    win_starts = 1:win_step:(n_timepoints - win_size + 1);
    n_windows = length(win_starts);
    
    all_pred = zeros(size(labels));
    fold_acc = zeros(K, 1);
    confusion_total = zeros(n_classes, n_classes);
    
    % 采用 One-vs-Rest 策略实现多类分类
    for k = 1:K
        test_idx  = (cv_indices == k);
        train_idx = (cv_indices ~= k);
        
        X_train = EEG_epochs(:, :, train_idx);
        Y_train = labels(train_idx);
        X_test  = EEG_epochs(:, :, test_idx);
        Y_test  = labels(test_idx);
        
        n_train = sum(train_idx);
        n_test  = sum(test_idx);
        
        % 对每个类别训练一个二分类HDCA模型 (OvR)
        scores = zeros(n_test, n_classes);
        
        for c = 1:n_classes
            Y_binary = double(Y_train == c);
            
            % 层1: 对每个时间窗口计算空间权重 (Fisher判别)
            spatial_weights = zeros(n_channels, n_windows);
            windowed_features_train = zeros(n_train, n_windows);
            windowed_features_test  = zeros(n_test, n_windows);
            
            for w = 1:n_windows
                win_range = win_starts(w):(win_starts(w) + win_size - 1);
                
                % 提取该窗口所有trial的数据: [n_channels x win_size x n_trials]
                win_data_train = X_train(:, win_range, :);
                
                % 计算每个trial在该窗口的通道均值: [n_channels x n_trials]
                chan_means_train = squeeze(mean(win_data_train, 2));
                if size(chan_means_train, 2) ~= n_train
                    chan_means_train = chan_means_train';
                end
                
                % Fisher判别空间权重
                pos_idx = (Y_binary == 1);
                neg_idx = (Y_binary == 0);
                
                mu_pos = mean(chan_means_train(:, pos_idx), 2);
                mu_neg = mean(chan_means_train(:, neg_idx), 2);
                
                cov_pos = cov(chan_means_train(:, pos_idx)');
                cov_neg = cov(chan_means_train(:, neg_idx)');
                Sw = cov_pos + cov_neg + 1e-6 * eye(n_channels);
                
                w_spatial = Sw \ (mu_pos - mu_neg);
                spatial_weights(:, w) = w_spatial;
                
                % 对训练集应用空间权重
                windowed_features_train(:, w) = (w_spatial' * chan_means_train)';
                
                % 对测试集应用空间权重
                win_data_test = X_test(:, win_range, :);
                chan_means_test = squeeze(mean(win_data_test, 2));
                if size(chan_means_test, 2) ~= n_test
                    chan_means_test = chan_means_test';
                end
                windowed_features_test(:, w) = (w_spatial' * chan_means_test)';
            end
            
            % 层2: 线性判别
            mu_pos_t = mean(windowed_features_train(Y_binary==1, :), 1);
            mu_neg_t = mean(windowed_features_train(Y_binary==0, :), 1);
            
            cov_pos_t = cov(windowed_features_train(Y_binary==1, :));
            cov_neg_t = cov(windowed_features_train(Y_binary==0, :));
            Sw_t = cov_pos_t + cov_neg_t + 1e-6 * eye(n_windows);
            
            w_temporal = Sw_t \ (mu_pos_t - mu_neg_t)';
            
            scores(:, c) = windowed_features_test * w_temporal;
        end
        
        [~, pred_labels] = max(scores, [], 2);
        
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
