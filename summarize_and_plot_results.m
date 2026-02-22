function summarize_and_plot_results(all_results, method_names, params)
% SUMMARIZE_AND_PLOT_RESULTS 汇总所有被试在所有方法上的分类结果并可视化
%
% 输入:
%   all_results  - 结构体数组, all_results(sub).MethodName = results
%   method_names - 方法名称元胞数组
%   params       - 参数结构体

    n_subjects = length(all_results);
    n_methods  = length(method_names);
    
    %% 提取各方法各被试的准确率
    acc_matrix = zeros(n_subjects, n_methods); % [subjects x methods]
    f1_matrix  = zeros(n_subjects, n_methods);
    
    for s = 1:n_subjects
        for m = 1:n_methods
            method = method_names{m};
            if isfield(all_results(s), method)
                r = all_results(s).(method);
                acc_matrix(s, m) = r.accuracy;
                if isfield(r, 'macro_f1')
                    f1_matrix(s, m) = r.macro_f1;
                end
            end
        end
    end
    
    %% 打印汇总表格
    fprintf('\n%s\n', repmat('=', 1, 90));
    fprintf('%-20s', '方法');
    fprintf('%-12s', '平均ACC(%)');
    fprintf('%-12s', '标准差(%)');
    fprintf('%-12s', '最高ACC(%)');
    fprintf('%-12s', '最低ACC(%)');
    fprintf('%-12s', '平均F1');
    fprintf('\n%s\n', repmat('-', 1, 90));
    
    for m = 1:n_methods
        acc_vals = acc_matrix(:, m) * 100;
        f1_vals  = f1_matrix(:, m);
        fprintf('%-20s', method_names{m});
        fprintf('%-12.2f', mean(acc_vals));
        fprintf('%-12.2f', std(acc_vals));
        fprintf('%-12.2f', max(acc_vals));
        fprintf('%-12.2f', min(acc_vals));
        fprintf('%-12.4f', mean(f1_vals));
        fprintf('\n');
    end
    fprintf('%s\n', repmat('=', 1, 90));
    
    %% 打印各被试详细结果
    fprintf('\n各被试分类准确率 (%%):\n');
    fprintf('%-10s', '被试');
    for m = 1:n_methods
        fprintf('%-14s', method_names{m});
    end
    fprintf('\n%s\n', repmat('-', 1, 10 + 14*n_methods));
    
    for s = 1:n_subjects
        fprintf('%-10s', sprintf('Sub%02d', s));
        for m = 1:n_methods
            fprintf('%-14.2f', acc_matrix(s, m) * 100);
        end
        fprintf('\n');
    end
    
    %% ===== 图1: 各方法平均准确率柱状图 =====
    figure('Name', '各方法平均分类准确率', 'Position', [100 100 900 500]);
    
    mean_acc = mean(acc_matrix, 1) * 100;
    std_acc  = std(acc_matrix, 0, 1) * 100;
    
    bar_handle = bar(mean_acc, 'FaceColor', 'flat');
    hold on;
    errorbar(1:n_methods, mean_acc, std_acc, 'k.', 'LineWidth', 1.5);
    
    colors = lines(n_methods);
    for m = 1:n_methods
        bar_handle.CData(m, :) = colors(m, :);
    end
    
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', method_names, 'XTickLabelRotation', 30);
    ylabel('分类准确率 (%)');
    title('各方法在10名被试上的平均分类准确率');
    ylim([0 100]);
    yline(100/params.n_classes, '--r', '随机水平', 'LineWidth', 1.5);
    grid on;
    
    saveas(gcf, 'Fig1_Average_Accuracy.png');
    fprintf('图1已保存: Fig1_Average_Accuracy.png\n');
    
    %% ===== 图2: 各方法箱线图 =====
    figure('Name', '各方法分类准确率分布', 'Position', [100 100 900 500]);
    
    boxplot(acc_matrix * 100, 'Labels', method_names);
    ylabel('分类准确率 (%)');
    title('各方法在10名被试上的分类准确率分布');
    set(gca, 'XTickLabelRotation', 30);
    yline(100/params.n_classes, '--r', '随机水平', 'LineWidth', 1.5);
    grid on;
    
    saveas(gcf, 'Fig2_Accuracy_Boxplot.png');
    fprintf('图2已保存: Fig2_Accuracy_Boxplot.png\n');
    
    %% ===== 图3: 各被试各方法热力图 =====
    figure('Name', '各被试各方法分类准确率热力图', 'Position', [100 100 1000 600]);
    
    imagesc(acc_matrix' * 100);
    colormap(jet);
    colorbar;
    caxis([0 100]);
    
    set(gca, 'XTick', 1:n_subjects, 'XTickLabel', arrayfun(@(x) sprintf('S%02d', x), 1:n_subjects, 'UniformOutput', false));
    set(gca, 'YTick', 1:n_methods, 'YTickLabel', method_names);
    xlabel('被试');
    ylabel('方法');
    title('各被试各方法分类准确率热力图 (%)');
    
    for s = 1:n_subjects
        for m = 1:n_methods
            text(s, m, sprintf('%.1f', acc_matrix(s, m)*100), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    
    saveas(gcf, 'Fig3_Heatmap.png');
    fprintf('图3已保存: Fig3_Heatmap.png\n');
    
    %% ===== 图4: 混淆矩阵 (最佳方法) =====
    [~, best_method_idx] = max(mean_acc);
    best_method = method_names{best_method_idx};
    
    total_cm = zeros(params.n_classes, params.n_classes);
    for s = 1:n_subjects
        if isfield(all_results(s), best_method)
            total_cm = total_cm + all_results(s).(best_method).confusion_matrix;
        end
    end
    
    figure('Name', '最佳方法混淆矩阵', 'Position', [100 100 600 500]);
    
    cm_norm = total_cm ./ sum(total_cm, 2) * 100;
    imagesc(cm_norm);
    colormap(flipud(hot));
    colorbar;
    caxis([0 100]);
    
    class_labels = {'A刺激', 'B刺激', 'C刺激'};
    set(gca, 'XTick', 1:params.n_classes, 'XTickLabel', class_labels);
    set(gca, 'YTick', 1:params.n_classes, 'YTickLabel', class_labels);
    xlabel('预测类别');
    ylabel('真实类别');
    title(sprintf('混淆矩阵 - %s (所有被试汇总, %%)', best_method));
    
    for i = 1:params.n_classes
        for j = 1:params.n_classes
            text(j, i, sprintf('%.1f%%\n(%d)', cm_norm(i,j), total_cm(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 11);
        end
    end
    
    saveas(gcf, 'Fig4_Confusion_Matrix.png');
    fprintf('图4已保存: Fig4_Confusion_Matrix.png\n');
    
    %% ===== 图5: 各方法F1分数雷达图 =====
    figure('Name', '各方法宏平均F1分数', 'Position', [100 100 700 500]);
    
    mean_f1 = mean(f1_matrix, 1);
    bar(mean_f1, 'FaceColor', [0.3 0.6 0.9]);
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', method_names, 'XTickLabelRotation', 30);
    ylabel('宏平均F1分数');
    title('各方法宏平均F1分数');
    ylim([0 1]);
    grid on;
    
    saveas(gcf, 'Fig5_F1_Scores.png');
    fprintf('图5已保存: Fig5_F1_Scores.png\n');
    
    %% ===== 统计检验: 方法间配对t检验 =====
    fprintf('\n===== 方法间配对t检验 (p值) =====\n');
    fprintf('%-20s', '');
    for m = 1:n_methods
        fprintf('%-12s', method_names{m});
    end
    fprintf('\n');
    
    p_matrix = ones(n_methods, n_methods);
    for m1 = 1:n_methods
        fprintf('%-20s', method_names{m1});
        for m2 = 1:n_methods
            if m1 ~= m2
                [~, p] = ttest(acc_matrix(:, m1), acc_matrix(:, m2));
                p_matrix(m1, m2) = p;
                if p < 0.05
                    fprintf('%-12s', sprintf('%.4f*', p));
                else
                    fprintf('%-12.4f', p);
                end
            else
                fprintf('%-12s', '-');
            end
        end
        fprintf('\n');
    end
    
    fprintf('\n* 表示 p < 0.05\n');
end
