%% ========================================================================
%% 科研论文统计图 —— MATLAB 版本
%% 假设：A 刺激对视觉搜索任务阶段的优化效用强于 B 和 C
%% ========================================================================
%  运行方式：直接在 MATLAB 中运行本脚本
%  输出：6 张 PNG + 6 张 PDF 图片，保存到脚本所在目录
%% ========================================================================

clear; clc; close all;

% 保存目录 = 脚本所在目录（自动检测）
save_dir = fileparts(mfilename('fullpath'));
if isempty(save_dir), save_dir = pwd; end
fprintf('图片保存目录: %s\n\n', save_dir);

%% ========================= 颜色与标签 =========================
color_A = [230, 75, 53] / 255;    % #E64B35 红
color_B = [77, 187, 213] / 255;   % #4DBBD5 青蓝
color_C = [0, 160, 135] / 255;    % #00A087 青绿
colors3 = [color_A; color_B; color_C];
cond_labels = {'A', 'B', 'C'};

%% ========================= 数据 =========================
% 行为学（10 被试 × 3 条件）
acc_A = [79.17, 85.83, 77.5, 75.83, 77.5, 80.83, 60.89, 85.83, 79.17, 80];
acc_B = [63.33, 72.5, 76.67, 59.17, 82.5, 53.33, 73.33, 69.17, 62.5, 75.83];
acc_C = [61.67, 75, 65.83, 55.83, 74.17, 46.67, 66.11, 70, 54.17, 77.5];

rt_A = [1052.93, 929.87, 1030.86, 1090.78, 1064.96, 1186.88, 1144.08, 1087.67, 1087.75, 1020.54];
rt_B = [1164.48, 1106.42, 1191.62, 1200.97, 1227.08, 1305.02, 1233.39, 1186.55, 1263.67, 1112.5];
rt_C = [1158.66, 1153.52, 1278.88, 1215.61, 1235.7, 1302.71, 1300.77, 1242.93, 1268.78, 1188.09];

dp_A = [1.629, 2.184, 1.515, 1.72, 1.526, 1.764, 0.553, 2.147, 1.625, 1.806];
dp_B = [0.681, 1.205, 1.458, 0.605, 1.878, 0.169, 1.252, 1.004, 0.65, 1.813];
dp_C = [0.612, 1.351, 0.817, 0.262, 1.353, -0.169, 0.839, 1.05, 0.212, 1.561];

% EEG ROI
n2_A = [-0.7136, -2.4264, -2.4977, -1.58, 0.2046, -2.4518, -0.982, 6.5011, -1.1916, -1.0591];
n2_B = [-1.5733, -1.3852, -1.2393, -1.1006, 1.3451, -3.9794, -1.073, -0.4501, -1.1712, -1.681];
n2_C = [-0.9812, -1.5905, -2.6002, -1.9756, -1.8674, -1.3861, -1.2298, 1.6493, -1.3386, -0.3058];

p3_A = [0.794, -1.5683, -0.9722, 1.2479, 0.7767, -0.0365, -0.417, 11.5699, -0.0692, -0.0076];
p3_B = [-0.2022, -0.5261, -3.3552, -1.8521, 2.9047, -0.2866, -2.2362, -3.3749, -1.0619, -0.134];
p3_C = [0.529, -2.7457, -2.0961, 0.9872, 1.5118, -1.0833, -1.5726, 3.6788, -0.8347, 1.3431];

n2lat_A = [274, 260, 252, 266, 274, 262, 250, 256, 278, 256];
n2lat_B = [272, 278, 350, 264, 280, 254, 250, 270, 280, 270];
n2lat_C = [274, 300, 350, 266, 322, 254, 236, 254, 284, 236];

p3lat_A = [388, 566, 282, 288, 354, 340, 346, 440, 588, 328];
p3lat_B = [378, 554, 280, 280, 532, 338, 282, 356, 264, 334];
p3lat_C = [370, 314, 278, 276, 600, 330, 348, 428, 272, 324];

nSubj = 10;

%% ========================================================================
%% 图1：行为学结果（准确率 / 反应时 / d'）
%% ========================================================================
fig1 = figure('Position', [50 50 1400 480], 'Color', 'w');

data_sets = { [acc_A', acc_B', acc_C'], [rt_A', rt_B', rt_C'], [dp_A', dp_B', dp_C'] };
ylabels1  = {'准确率 (%)', '反应时 (ms)', "d' (敏感性)"};
panels1   = {'a', 'b', 'c'};

for sp = 1:3
    ax = subplot(1, 3, sp);
    D = data_sets{sp};
    draw_bar_panel(ax, D, colors3, cond_labels, ylabels1{sp}, panels1{sp}, nSubj);
end

% 图例放右侧
lg = legend(ax, {'A 条件', 'B 条件', 'C 条件'}, 'FontSize', 12, 'Box', 'on', ...
    'Location', 'eastoutside');
lg.Position(1) = 0.92;
lg.Position(2) = 0.4;

sgtitle('图1  行为学结果', 'FontSize', 16, 'FontWeight', 'bold');
save_fig(fig1, save_dir, 'Figure1_Behavioral_Results');

%% ========================================================================
%% 图2：ERP ROI 成分分析（N2/P3 振幅 + N2/P3 潜伏期）
%% ========================================================================
fig2 = figure('Position', [50 50 1800 480], 'Color', 'w');

data_sets2 = { [n2_A', n2_B', n2_C'], [p3_A', p3_B', p3_C'], ...
               [n2lat_A', n2lat_B', n2lat_C'], [p3lat_A', p3lat_B', p3lat_C'] };
ylabels2 = {'N2 平均振幅 (\muV)', 'P3 平均振幅 (\muV)', ...
            'N2 峰值潜伏期 (ms)', 'P3 峰值潜伏期 (ms)'};
titles2  = {'N2 ROI (FCz+Fz+Cz)', 'P3 ROI (Pz+Cz)', 'N2 峰值潜伏期', 'P3 峰值潜伏期'};
panels2  = {'a', 'b', 'c', 'd'};

for sp = 1:4
    ax = subplot(1, 4, sp);
    D = data_sets2{sp};
    draw_bar_panel(ax, D, colors3, cond_labels, ylabels2{sp}, panels2{sp}, nSubj);
    title(titles2{sp}, 'FontSize', 12, 'FontWeight', 'bold');
end

lg = legend(ax, {'A 条件', 'B 条件', 'C 条件'}, 'FontSize', 12, 'Box', 'on', ...
    'Location', 'eastoutside');
lg.Position(1) = 0.93;
lg.Position(2) = 0.4;

sgtitle('图2  ERP ROI 成分分析', 'FontSize', 16, 'FontWeight', 'bold');
save_fig(fig2, save_dir, 'Figure2_ERP_ROI_Results');

%% ========================================================================
%% 图3：综合面板（行为学 + 相关 + ERP）
%% ========================================================================
fig3 = figure('Position', [30 30 1800 900], 'Color', 'w');

% 第一行：行为学 3 面板 + 1 相关图
subplot(2, 4, 1);
draw_bar_panel(gca, [acc_A', acc_B', acc_C'], colors3, cond_labels, '准确率 (%)', 'a', nSubj);
subplot(2, 4, 2);
draw_bar_panel(gca, [rt_A', rt_B', rt_C'], colors3, cond_labels, '反应时 (ms)', 'b', nSubj);
subplot(2, 4, 3);
draw_bar_panel(gca, [dp_A', dp_B', dp_C'], colors3, cond_labels, "d'", 'c', nSubj);

subplot(2, 4, 4);
draw_corr_panel(gca, dp_A - dp_C, p3_A - p3_C, "\Deltad' (A-C)", '\DeltaP3 振幅 (A-C, \muV)', 'd');

% 第二行：ERP 4 面板
erp_labels3 = {'N2 振幅 (\muV)', 'P3 振幅 (\muV)', 'N2 潜伏期 (ms)', 'P3 潜伏期 (ms)'};
erp_panels3 = {'e', 'f', 'g', 'h'};
for sp = 1:4
    subplot(2, 4, 4 + sp);
    draw_bar_panel(gca, data_sets2{sp}, colors3, cond_labels, erp_labels3{sp}, erp_panels3{sp}, nSubj);
end

sgtitle('图3  行为学与ERP综合分析', 'FontSize', 18, 'FontWeight', 'bold');
save_fig(fig3, save_dir, 'Figure3_Comprehensive_Summary');

%% ========================================================================
%% 图4：雷达图（多维度性能概况）
%% ========================================================================
fig4 = figure('Position', [100 100 700 600], 'Color', 'w');

norm01 = @(x) (x - min(x)) / (max(x) - min(x) + 1e-10);
cats = {'准确率', '速度(1/RT)', "d'", 'P3 振幅', '-N2 振幅'};
nCats = length(cats);

vals = [ norm01([mean(acc_A), mean(acc_B), mean(acc_C)]); ...
         norm01(1000 ./ [mean(rt_A), mean(rt_B), mean(rt_C)]); ...
         norm01([mean(dp_A), mean(dp_B), mean(dp_C)]); ...
         norm01([mean(p3_A), mean(p3_B), mean(p3_C)]); ...
         norm01(-[mean(n2_A), mean(n2_B), mean(n2_C)]) ]';  % 3×5

angles = linspace(0, 2*pi, nCats + 1);
ax4 = polaraxes;
hold(ax4, 'on');

for c = 1:3
    v = [vals(c, :), vals(c, 1)];
    polarplot(ax4, angles, v, '-o', 'Color', colors3(c,:), 'LineWidth', 2, 'MarkerSize', 6, ...
        'MarkerFaceColor', colors3(c,:));
end

ax4.ThetaTick = rad2deg(angles(1:end-1));
ax4.ThetaTickLabel = cats;
ax4.RLim = [0 1.15];
ax4.RTickLabel = {'', '0.25', '0.50', '0.75', '1.00'};
ax4.FontSize = 12;
legend({'A 条件', 'B 条件', 'C 条件'}, 'FontSize', 12, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'Box', 'on');
title('图4  多维度性能概况', 'FontSize', 16, 'FontWeight', 'bold');

save_fig(fig4, save_dir, 'Figure4_Radar_Chart');

%% ========================================================================
%% 图5：9 种算法分类精度对比
%% ========================================================================
fprintf('====== 9 种算法分类 (LOSO-CV) ======\n');

% 构建特征矩阵
X = []; Y = []; G = [];
for si = 1:nSubj
    X = [X; acc_A(si), rt_A(si), dp_A(si), n2_A(si), p3_A(si), n2lat_A(si), p3lat_A(si)]; Y = [Y; 1]; G = [G; si];
    X = [X; acc_B(si), rt_B(si), dp_B(si), n2_B(si), p3_B(si), n2lat_B(si), p3lat_B(si)]; Y = [Y; 2]; G = [G; si];
    X = [X; acc_C(si), rt_C(si), dp_C(si), n2_C(si), p3_C(si), n2lat_C(si), p3lat_C(si)]; Y = [Y; 3]; G = [G; si];
end

feature_names = {'Accuracy', 'RT', "d'", 'N2 Amp', 'P3 Amp', 'N2 Lat', 'P3 Lat'};

algo_names = {'SVM (Linear)', 'SVM (RBF)', 'KNN (k=3)', 'Decision Tree', ...
              'Random Forest', 'Logistic Reg.', 'Naive Bayes', 'LDA', 'Gradient Boost'};
nAlgos = length(algo_names);
accuracy_all = zeros(nAlgos, 1);
y_pred_all = zeros(length(Y), nAlgos);

for fold = 1:nSubj
    test_idx = (G == fold);
    train_idx = ~test_idx;

    X_tr = X(train_idx, :);
    Y_tr = Y(train_idx);
    X_te = X(test_idx, :);

    mu = mean(X_tr); sig_val = std(X_tr); sig_val(sig_val == 0) = 1;
    X_tr_z = (X_tr - mu) ./ sig_val;
    X_te_z = (X_te - mu) ./ sig_val;

    for ai = 1:nAlgos
        try
            switch ai
                case 1
                    t = templateSVM('KernelFunction', 'linear');
                    mdl = fitcecoc(X_tr_z, Y_tr, 'Learners', t);
                case 2
                    t = templateSVM('KernelFunction', 'rbf');
                    mdl = fitcecoc(X_tr_z, Y_tr, 'Learners', t);
                case 3
                    mdl = fitcknn(X_tr_z, Y_tr, 'NumNeighbors', 3);
                case 4
                    mdl = fitctree(X_tr_z, Y_tr, 'MaxNumSplits', 5);
                case 5
                    mdl = TreeBagger(100, X_tr_z, Y_tr, 'Method', 'classification', 'MaxNumSplits', 5);
                case 6
                    mdl = fitclinear(X_tr_z, Y_tr, 'Learner', 'logistic');
                case 7
                    mdl = fitcnb(X_tr_z, Y_tr);
                case 8
                    mdl = fitcdiscr(X_tr_z, Y_tr, 'DiscrimType', 'linear');
                case 9
                    t = templateTree('MaxNumSplits', 3);
                    mdl = fitcensemble(X_tr_z, Y_tr, 'Method', 'AdaBoostM2', 'Learners', t, 'NumLearningCycles', 50);
            end

            if ai == 5
                yp = str2double(predict(mdl, X_te_z));
            elseif ai == 6
                yp = predict(mdl, X_te_z);
            else
                yp = predict(mdl, X_te_z);
            end
            y_pred_all(test_idx, ai) = yp;
        catch ME
            fprintf('  [警告] %s fold %d 失败: %s\n', algo_names{ai}, fold, ME.message);
            y_pred_all(test_idx, ai) = 0;
        end
    end
end

for ai = 1:nAlgos
    accuracy_all(ai) = sum(y_pred_all(:, ai) == Y) / length(Y) * 100;
    fprintf('  %-20s: %.1f%%\n', algo_names{ai}, accuracy_all(ai));
end

[sorted_acc, sort_idx] = sort(accuracy_all, 'descend');
sorted_names = algo_names(sort_idx);

algo_colors = [
    31, 78, 121;    % dark blue
    230, 75, 53;    % red
    0, 160, 135;    % green
    112, 48, 160;   % purple
    243, 155, 127;  % orange
    44, 44, 44;     % near black
    139, 105, 20;   % olive
    0, 112, 192;    % blue
    106, 13, 173;   % dark purple
] / 255;

fig5 = figure('Position', [50 50 1100 550], 'Color', 'w');
bh = barh(1:nAlgos, sorted_acc, 0.6, 'EdgeColor', 'k', 'LineWidth', 0.6);
bh.FaceColor = 'flat';
for bi = 1:nAlgos
    bh.CData(bi, :) = algo_colors(bi, :);
end
hold on;
xline(100/3, '--r', 'LineWidth', 1.5);
text(100/3 + 1, nAlgos - 0.2, sprintf('机会水平\n(33.3%%)'), 'Color', 'r', 'FontSize', 11);

for bi = 1:nAlgos
    text(sorted_acc(bi) + 1, bi, sprintf('%.1f%%', sorted_acc(bi)), ...
        'FontWeight', 'bold', 'FontSize', 11, 'VerticalAlignment', 'middle');
end

set(gca, 'YTick', 1:nAlgos, 'YTickLabel', sorted_names, 'FontSize', 12, ...
    'YDir', 'reverse', 'Box', 'off');
ax5 = gca;
ax5.XAxis.FontSize = 12;
ax5.YAxis.FontSize = 12;
xlabel('分类精度 (%)', 'FontSize', 14);
xlim([0, max(sorted_acc) + 15]);
title('图5  9种算法分类精度对比 (LOSO-CV)', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'TickDir', 'out');
box off;

save_fig(fig5, save_dir, 'Figure5_ML_Classification');

%% ========================================================================
%% 图6：Random Forest 特征重要性
%% ========================================================================
X_z = zscore(X);
rf_mdl = TreeBagger(200, X_z, Y, 'Method', 'classification', ...
    'OOBPredictorImportance', 'on', 'MaxNumSplits', 5);
importance = rf_mdl.OOBPermutedPredictorDeltaError;

[sorted_imp, imp_idx] = sort(importance, 'ascend');

cmap_imp = [
    69, 117, 180;
    116, 173, 209;
    171, 217, 233;
    254, 224, 144;
    253, 174, 97;
    244, 109, 67;
    215, 48, 39;
] / 255;

fig6 = figure('Position', [50 50 900 480], 'Color', 'w');
bh6 = barh(1:length(feature_names), sorted_imp, 0.55, 'EdgeColor', 'k', 'LineWidth', 0.6);
bh6.FaceColor = 'flat';
for bi = 1:length(feature_names)
    bh6.CData(bi, :) = cmap_imp(bi, :);
end

hold on;
for bi = 1:length(feature_names)
    text(sorted_imp(bi) + max(sorted_imp)*0.02, bi, sprintf('%.3f', sorted_imp(bi)), ...
        'FontWeight', 'bold', 'FontSize', 11, 'VerticalAlignment', 'middle');
end

set(gca, 'YTick', 1:length(feature_names), 'YTickLabel', feature_names(imp_idx), ...
    'FontSize', 12, 'Box', 'off');
xlabel('特征重要性 (OOB Permutation)', 'FontSize', 14);
title('图6  Random Forest 特征重要性', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'TickDir', 'out');
box off;

save_fig(fig6, save_dir, 'Figure6_Feature_Importance');

fprintf('\n====== 全部 6 张图生成完毕！ ======\n');
fprintf('保存位置: %s\n', save_dir);

%% ========================================================================
%% ======================= 局部函数 ======================================
%% ========================================================================

function draw_bar_panel(ax, D, colors3, cond_labels, ylbl, panel_label, nSubj)
    axes(ax); hold on;
    [n, k] = size(D);
    means = mean(D);
    sems = std(D) / sqrt(n);

    x = 1:k;
    bh = bar(x, means, 0.55, 'EdgeColor', 'k', 'LineWidth', 0.6);
    bh.FaceColor = 'flat';
    for c = 1:k
        bh.CData(c, :) = colors3(c, :);
    end

    errorbar(x, means, sems, 'k.', 'LineWidth', 1.2, 'CapSize', 5);

    rng_state = rng;
    for c = 1:k
        rng(c * 100 + 42);
        jit = randn(nSubj, 1) * 0.04;
        scatter(x(c) + jit, D(:, c), 22, colors3(c, :), 'filled', ...
            'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.6, 'LineWidth', 0.4);
    end
    rng(rng_state);

    set(gca, 'XTick', x, 'XTickLabel', cond_labels, 'FontSize', 12, 'Box', 'off');
    ylabel(ylbl, 'FontSize', 13);
    ax_obj = gca;
    ax_obj.XAxis.FontSize = 13;
    ax_obj.XAxis.FontWeight = 'bold';
    set(gca, 'TickDir', 'out');
    box off;

    [F, p_val, eta2, df1, df2] = rm_anova_calc(D);

    yrng = max(means + sems) - min(means - sems);
    ymax = max(means + sems);
    gap = yrng * 0.08;

    pairs = [1 2; 1 3; 2 3];
    pair_labels_txt = {'A vs B', 'A vs C', 'B vs C'};
    for pi = 1:3
        c1 = pairs(pi, 1); c2 = pairs(pi, 2);
        [~, p_raw] = ttest(D(:, c1), D(:, c2));
        p_bonf = min(p_raw * 3, 1);
        draw_bracket(x(c1), x(c2), ymax + gap * pi, p_bonf);
    end
    ylim_top = ymax + gap * 4.5;
    if ylim_top > ax_obj.YLim(2)
        ylim([ax_obj.YLim(1), ylim_top]);
    end

    s = sig_str(p_val);
    txt = sprintf('F(%d,%d)=%.1f\np=%.3f %s', df1, df2, F, p_val, s);
    text(0.97, 0.97, txt, 'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'FontSize', 8.5, 'BackgroundColor', [1 1 0.9], ...
        'EdgeColor', [0.8 0.8 0.8], 'Margin', 3);

    text(-0.12, 1.06, panel_label, 'Units', 'normalized', 'FontSize', 18, ...
        'FontWeight', 'bold', 'VerticalAlignment', 'top');
end

function draw_corr_panel(ax, xd, yd, xlbl, ylbl, panel_label)
    axes(ax); hold on;
    scatter(xd, yd, 50, [60 80 136]/255, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
    [r, p_val] = corr(xd(:), yd(:), 'Type', 'Pearson');
    coeffs = polyfit(xd, yd, 1);
    xline_vals = linspace(min(xd), max(xd), 100);
    plot(xline_vals, polyval(coeffs, xline_vals), '--', 'Color', [230 75 53]/255, 'LineWidth', 1.5);
    xlabel(xlbl, 'FontSize', 11);
    ylabel(ylbl, 'FontSize', 11);
    title(sprintf('r = %.3f, p = %.3f %s', r, p_val, sig_str(p_val)), 'FontSize', 10);
    set(gca, 'Box', 'off', 'TickDir', 'out');
    text(-0.12, 1.06, panel_label, 'Units', 'normalized', 'FontSize', 18, ...
        'FontWeight', 'bold', 'VerticalAlignment', 'top');
end

function [F, p, eta2, df1, df2] = rm_anova_calc(D)
    [n, k] = size(D);
    gm = mean(D(:));
    SS_c = n * sum((mean(D) - gm).^2);
    SS_s = k * sum((mean(D, 2) - gm).^2);
    SS_t = sum((D(:) - gm).^2);
    SS_e = SS_t - SS_c - SS_s;
    df1 = k - 1;
    df2 = (k - 1) * (n - 1);
    F = (SS_c / df1) / (SS_e / df2);
    p = 1 - fcdf(F, df1, df2);
    eta2 = SS_c / (SS_c + SS_e);
end

function draw_bracket(x1, x2, y, p)
    s = sig_str(p);
    if strcmp(s, 'n.s.'), return; end
    yl = ylim;
    bh = (yl(2) - yl(1)) * 0.015;
    plot([x1 x1 x2 x2], [y, y+bh, y+bh, y], 'k-', 'LineWidth', 0.9, 'Clipping', 'off');
    text((x1+x2)/2, y + bh * 1.5, s, 'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold');
end

function s = sig_str(p)
    if p < 0.001,     s = '***';
    elseif p < 0.01,  s = '**';
    elseif p < 0.05,  s = '*';
    else,              s = 'n.s.';
    end
end

function save_fig(fig_handle, dir_path, name)
    saveas(fig_handle, fullfile(dir_path, [name '.png']));
    saveas(fig_handle, fullfile(dir_path, [name '.pdf']));
    fprintf('已保存: %s.png / .pdf\n', name);
end
