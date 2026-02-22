function [p_value, null_distribution] = permutation_test(EEG_epochs, labels, params, method_name, n_permutations)
% PERMUTATION_TEST 置换检验评估分类结果的统计显著性
%
% 通过随机打乱标签并重新分类来构建零分布,
% 计算实际准确率超过零分布的比例作为p值。
%
% 输入:
%   EEG_epochs     - [n_channels x n_timepoints x n_trials]
%   labels         - [n_trials x 1] 类别标签
%   params         - 参数结构体
%   method_name    - 方法名称 ('SVM_Linear', 'TRCA', 'eTRCA' 等)
%   n_permutations - 置换次数 (建议 >= 500)
%
% 输出:
%   p_value           - 统计显著性p值
%   null_distribution - 零分布准确率数组

    if nargin < 5
        n_permutations = 1000;
    end
    
    n_trials = length(labels);
    K = params.K_fold;
    
    %% 计算真实准确率
    cv_indices = crossvalind('Kfold', labels, K);
    
    switch method_name
        case 'SVM_Linear'
            features = extract_psd_features(EEG_epochs, params);
            real_results = classify_svm(features, labels, cv_indices, params, 'linear');
        case 'TRCA'
            real_results = classify_trca(EEG_epochs, labels, cv_indices, params);
        case 'eTRCA'
            real_results = classify_etrca(EEG_epochs, labels, cv_indices, params);
        case 'HDCA'
            real_results = classify_hdca(EEG_epochs, labels, cv_indices, params);
        otherwise
            features = extract_psd_features(EEG_epochs, params);
            real_results = classify_svm(features, labels, cv_indices, params, 'linear');
    end
    
    real_acc = real_results.accuracy;
    fprintf('实际分类准确率: %.2f%%\n', real_acc * 100);
    
    %% 置换检验
    null_distribution = zeros(n_permutations, 1);
    
    for i = 1:n_permutations
        if mod(i, 100) == 0
            fprintf('  置换检验进度: %d / %d\n', i, n_permutations);
        end
        
        perm_labels = labels(randperm(n_trials));
        perm_cv = crossvalind('Kfold', perm_labels, K);
        
        switch method_name
            case 'SVM_Linear'
                perm_results = classify_svm(features, perm_labels, perm_cv, params, 'linear');
            case 'TRCA'
                perm_results = classify_trca(EEG_epochs, perm_labels, perm_cv, params);
            case 'eTRCA'
                perm_results = classify_etrca(EEG_epochs, perm_labels, perm_cv, params);
            case 'HDCA'
                perm_results = classify_hdca(EEG_epochs, perm_labels, perm_cv, params);
            otherwise
                perm_results = classify_svm(features, perm_labels, perm_cv, params, 'linear');
        end
        
        null_distribution(i) = perm_results.accuracy;
    end
    
    %% 计算p值
    p_value = sum(null_distribution >= real_acc) / n_permutations;
    
    fprintf('置换检验 p值: %.4f\n', p_value);
    
    %% 可视化
    figure('Name', sprintf('置换检验 - %s', method_name), 'Position', [100 100 700 400]);
    
    histogram(null_distribution * 100, 30, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
    hold on;
    xline(real_acc * 100, 'r-', 'LineWidth', 2);
    xline(100/params.n_classes, 'b--', 'LineWidth', 1.5);
    
    xlabel('分类准确率 (%)');
    ylabel('频次');
    title(sprintf('置换检验 - %s (p=%.4f, n=%d)', method_name, p_value, n_permutations));
    legend({'零分布', sprintf('实际准确率 (%.1f%%)', real_acc*100), '随机水平'}, ...
        'Location', 'best');
    
    saveas(gcf, sprintf('Permutation_Test_%s.png', method_name));
end
