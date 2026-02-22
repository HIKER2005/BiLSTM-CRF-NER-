function results = classify_svm(features, labels, cv_indices, params, kernel_type)
% CLASSIFY_SVM 使用SVM进行多类别分类 (One-vs-One策略)
%
% 输入:
%   features    - [n_trials x n_features] 特征矩阵
%   labels      - [n_trials x 1] 类别标签
%   cv_indices  - [n_trials x 1] 交叉验证折号
%   params      - 参数结构体
%   kernel_type - 核函数类型: 'linear' 或 'rbf'
%
% 输出:
%   results     - 结构体包含 accuracy, confusion_matrix, per_fold_acc 等

    K = params.K_fold;
    n_classes = params.n_classes;
    
    all_pred = zeros(size(labels));
    fold_acc = zeros(K, 1);
    confusion_total = zeros(n_classes, n_classes);
    
    for k = 1:K
        test_idx  = (cv_indices == k);
        train_idx = (cv_indices ~= k);
        
        X_train = features(train_idx, :);
        Y_train = labels(train_idx);
        X_test  = features(test_idx, :);
        Y_test  = labels(test_idx);
        
        % 特征标准化 (z-score)
        [X_train, mu, sigma] = zscore(X_train);
        sigma(sigma == 0) = 1;
        X_test = (X_test - mu) ./ sigma;
        
        % 特征选择: 去除方差为0的特征
        valid_feat = var(X_train) > 0;
        X_train = X_train(:, valid_feat);
        X_test  = X_test(:, valid_feat);
        
        % 使用MATLAB自带的fitcecoc进行多类SVM
        if strcmp(kernel_type, 'linear')
            t = templateSVM('KernelFunction', 'linear', 'Standardize', false, ...
                'BoxConstraint', 1);
        elseif strcmp(kernel_type, 'rbf')
            t = templateSVM('KernelFunction', 'rbf', 'Standardize', false, ...
                'BoxConstraint', 1, 'KernelScale', 'auto');
        else
            t = templateSVM('KernelFunction', kernel_type, 'Standardize', false);
        end
        
        svm_model = fitcecoc(X_train, Y_train, 'Learners', t, 'Coding', 'onevsone');
        
        pred_labels = predict(svm_model, X_test);
        
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
    
    % 多类别性能指标
    for c = 1:n_classes
        TP = confusion_total(c, c);
        FP = sum(confusion_total(:, c)) - TP;
        FN = sum(confusion_total(c, :)) - TP;
        TN = sum(confusion_total(:)) - TP - FP - FN;
        
        results.precision(c) = TP / (TP + FP + eps);
        results.recall(c)    = TP / (TP + FN + eps);
        results.f1_score(c)  = 2 * results.precision(c) * results.recall(c) / ...
                               (results.precision(c) + results.recall(c) + eps);
    end
    
    results.macro_precision = mean(results.precision);
    results.macro_recall    = mean(results.recall);
    results.macro_f1        = mean(results.f1_score);
    
    fprintf('    准确率: %.2f%% (±%.2f%%)\n', results.accuracy*100, results.std_acc*100);
end
