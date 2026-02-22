function results = classify_xdawn_svm(EEG_epochs, labels, cv_indices, params)
% CLASSIFY_XDAWN_SVM xDAWN空间滤波 + SVM分类
%
% xDAWN通过最大化信噪比来提取最优空间滤波器，
% 增强事件相关响应(ERP/SSVEP)的信号成分。
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
    n_components = min(6, n_channels); % xDAWN保留的成分数
    
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
        
        %% xDAWN空间滤波器训练
        % 计算全局协方差
        X_concat = reshape(X_train, n_channels, []);
        C_total = (X_concat * X_concat') / size(X_concat, 2);
        C_total = C_total + 1e-6 * eye(n_channels);
        
        % 对每个类别计算类别平均协方差
        xdawn_filters = zeros(n_channels, n_components * n_classes);
        comp_idx = 0;
        
        for c = 1:n_classes
            class_trials = X_train(:, :, Y_train == c);
            avg_response = mean(class_trials, 3); % [n_channels x n_timepoints]
            
            C_signal = (avg_response * avg_response') / n_timepoints;
            C_signal = C_signal + 1e-6 * eye(n_channels);
            
            % 广义特征值分解
            [V, D] = eig(C_signal, C_total);
            [~, sort_idx] = sort(diag(D), 'descend');
            V = V(:, sort_idx);
            
            for nc = 1:n_components
                V(:, nc) = V(:, nc) / norm(V(:, nc));
            end
            
            xdawn_filters(:, comp_idx+1:comp_idx+n_components) = V(:, 1:n_components);
            comp_idx = comp_idx + n_components;
        end
        
        %% 应用xDAWN滤波并提取特征
        n_total_components = size(xdawn_filters, 2);
        feat_train = zeros(n_train, n_total_components * 2);
        feat_test  = zeros(n_test,  n_total_components * 2);
        
        for t = 1:n_train
            filtered = xdawn_filters' * X_train(:, :, t);
            feat_train(t, :) = [mean(filtered, 2)', var(filtered, 0, 2)'];
        end
        
        for t = 1:n_test
            filtered = xdawn_filters' * X_test(:, :, t);
            feat_test(t, :) = [mean(filtered, 2)', var(filtered, 0, 2)'];
        end
        
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
