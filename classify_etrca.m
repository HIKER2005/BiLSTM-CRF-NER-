function results = classify_etrca(EEG_epochs, labels, cv_indices, params)
% CLASSIFY_ETRCA 集成任务相关成分分析 (Ensemble TRCA)
%
% eTRCA利用所有类别的训练数据共同计算TRCA空间滤波器，
% 然后与各类别模板进行相关性比较。
% 相比标准TRCA，eTRCA在训练样本有限时具有更好的鲁棒性。
%
% 参考: Nakanishi et al. (2018)
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
    n_components = min(3, n_channels);
    
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
        
        %% 对每个类别计算S和Q矩阵, 然后汇总
        S_total = zeros(n_channels, n_channels);
        Q_total = zeros(n_channels, n_channels);
        
        W_individual = cell(n_classes, 1);  % 各类别独立的TRCA滤波器
        templates = cell(n_classes, 1);
        
        for c = 1:n_classes
            class_data = X_train(:, :, Y_train == c);
            n_c = size(class_data, 3);
            
            S_c = zeros(n_channels, n_channels);
            Q_c = zeros(n_channels, n_channels);
            
            for i = 1:n_c
                x_i = class_data(:, :, i) - mean(class_data(:, :, i), 2);
                Q_c = Q_c + x_i * x_i';
                for j = i+1:n_c
                    x_j = class_data(:, :, j) - mean(class_data(:, :, j), 2);
                    S_c = S_c + x_i * x_j' + x_j * x_i';
                end
            end
            
            S_total = S_total + S_c;
            Q_total = Q_total + Q_c;
            
            % 各类别独立的TRCA
            Q_c_reg = Q_c + 1e-6 * eye(n_channels);
            [V, D] = eig(S_c, Q_c_reg);
            [~, sort_idx] = sort(diag(real(D)), 'descend');
            V = real(V(:, sort_idx));
            W_individual{c} = V(:, 1:n_components);
            
            templates{c} = mean(class_data, 3);
        end
        
        %% 集成TRCA: 使用总S和Q矩阵求解空间滤波器
        Q_total = Q_total + 1e-6 * eye(n_channels);
        [V_e, D_e] = eig(S_total, Q_total);
        [~, sort_idx_e] = sort(diag(real(D_e)), 'descend');
        V_e = real(V_e(:, sort_idx_e));
        W_ensemble = V_e(:, 1:n_components);
        
        %% 分类: 结合个体TRCA和集成TRCA的相关系数
        pred_labels = zeros(n_test, 1);
        
        for t = 1:n_test
            x_test = X_test(:, :, t);
            
            corr_scores = zeros(n_classes, 1);
            for c = 1:n_classes
                % 使用集成滤波器
                y_test_e     = W_ensemble' * x_test;
                y_template_e = W_ensemble' * templates{c};
                
                r_ensemble = 0;
                for nc = 1:n_components
                    r = corrcoef(y_test_e(nc,:), y_template_e(nc,:));
                    r_ensemble = r_ensemble + r(1,2);
                end
                r_ensemble = r_ensemble / n_components;
                
                % 使用个体滤波器
                w_ind = W_individual{c};
                y_test_i     = w_ind' * x_test;
                y_template_i = w_ind' * templates{c};
                
                r_individual = 0;
                for nc = 1:n_components
                    r = corrcoef(y_test_i(nc,:), y_template_i(nc,:));
                    r_individual = r_individual + r(1,2);
                end
                r_individual = r_individual / n_components;
                
                % Fisher z变换后加权融合
                z_e = atanh(r_ensemble);
                z_i = atanh(r_individual);
                corr_scores(c) = z_e + z_i;
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
