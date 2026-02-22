function results = classify_dcpm_svm(EEG_epochs, labels, cv_indices, params)
% CLASSIFY_DCPM_SVM 判别典型模式匹配 + SVM (DCPM + SVM)
%
% DCPM (Discriminative Canonical Pattern Matching):
%   利用空间滤波和模板匹配相结合的方法。
%   首先通过DSP(判别空间模式)提取空间滤波器,
%   然后计算滤波后信号与类别模板的相关系数作为特征,
%   最后用SVM进行分类。
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
    n_filters = min(4, n_channels); % 空间滤波器数量
    
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
        
        n_train = sum(train_idx);
        n_test  = sum(test_idx);
        
        %% 计算DSP (Discriminative Spatial Pattern) 空间滤波器
        % 类间散布矩阵 Sb
        grand_mean = mean(mean(X_train, 3), 2); % [n_channels x 1]
        Sb = zeros(n_channels, n_channels);
        Sw = zeros(n_channels, n_channels);
        
        for c = 1:n_classes
            class_data = X_train(:, :, Y_train == c);
            n_c = size(class_data, 3);
            
            class_mean = mean(mean(class_data, 3), 2); % [n_channels x 1]
            diff = class_mean - grand_mean;
            Sb = Sb + n_c * (diff * diff');
            
            % 类内散布矩阵
            for t = 1:n_c
                trial_mean = mean(class_data(:, :, t), 2);
                diff_w = trial_mean - class_mean;
                Sw = Sw + diff_w * diff_w';
            end
        end
        
        Sw = Sw + 1e-6 * eye(n_channels);
        
        % 广义特征值分解
        [W, D] = eig(Sb, Sw);
        [~, sort_idx] = sort(diag(D), 'descend');
        W = W(:, sort_idx(1:n_filters));
        
        % 归一化
        for f = 1:n_filters
            W(:, f) = W(:, f) / norm(W(:, f));
        end
        
        %% 计算类别模板
        templates = zeros(n_filters, n_timepoints, n_classes);
        for c = 1:n_classes
            class_data = X_train(:, :, Y_train == c);
            class_avg = mean(class_data, 3); % [n_channels x n_timepoints]
            templates(:, :, c) = W' * class_avg; % [n_filters x n_timepoints]
        end
        
        %% 提取特征 (与各类别模板的相关系数)
        feat_train = zeros(n_train, n_classes * n_filters);
        feat_test  = zeros(n_test,  n_classes * n_filters);
        
        for t = 1:n_train
            filtered = W' * X_train(:, :, t); % [n_filters x n_timepoints]
            fi = 0;
            for c = 1:n_classes
                for f = 1:n_filters
                    fi = fi + 1;
                    r = corrcoef(filtered(f, :), templates(f, :, c));
                    feat_train(t, fi) = r(1, 2);
                end
            end
        end
        
        for t = 1:n_test
            filtered = W' * X_test(:, :, t);
            fi = 0;
            for c = 1:n_classes
                for f = 1:n_filters
                    fi = fi + 1;
                    r = corrcoef(filtered(f, :), templates(f, :, c));
                    feat_test(t, fi) = r(1, 2);
                end
            end
        end
        
        % 处理NaN
        feat_train(isnan(feat_train)) = 0;
        feat_test(isnan(feat_test))   = 0;
        
        %% 标准化
        [feat_train, mu, sigma] = zscore(feat_train);
        sigma(sigma == 0) = 1;
        feat_test = (feat_test - mu) ./ sigma;
        
        %% SVM分类
        t_svm = templateSVM('KernelFunction', 'linear', 'Standardize', false, ...
            'BoxConstraint', 1);
        model = fitcecoc(feat_train, Y_train, 'Learners', t_svm, 'Coding', 'onevsone');
        pred_labels = predict(model, feat_test);
        
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
