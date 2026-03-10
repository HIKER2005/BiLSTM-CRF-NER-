%% ========================================================================
%% 9种机器学习算法分类 —— 基于行为学+EEG特征区分 A/B/C 三种刺激条件
%% ========================================================================
%  使用 Leave-One-Subject-Out 交叉验证（LOSO-CV）
%  特征：准确率、反应时、d'、N2平均振幅、P3平均振幅、N2潜伏期、P3潜伏期
%  标签：A=1, B=2, C=3
%  
%  9种算法：
%    1. SVM (Linear)        - 线性支持向量机
%    2. SVM (RBF)           - 径向基核支持向量机
%    3. KNN                 - K近邻
%    4. Decision Tree       - 决策树
%    5. Random Forest       - 随机森林
%    6. Naive Bayes         - 朴素贝叶斯
%    7. LDA                 - 线性判别分析
%    8. AdaBoost            - 自适应提升
%    9. Neural Network      - 浅层神经网络
%% ========================================================================

clear; clc;
fprintf('====== 9种算法分类分析 ======\n');
fprintf('交叉验证方式: Leave-One-Subject-Out (LOSO)\n\n');

%% ========================= 数据定义 =========================
% 行为学数据 (10 subjects × 3 conditions)
acc_A = [79.17, 85.83, 77.5, 75.83, 77.5, 80.83, 60.89, 85.83, 79.17, 80];
acc_B = [63.33, 72.5, 76.67, 59.17, 82.5, 53.33, 73.33, 69.17, 62.5, 75.83];
acc_C = [61.67, 75, 65.83, 55.83, 74.17, 46.67, 66.11, 70, 54.17, 77.5];

rt_A = [1052.93, 929.87, 1030.86, 1090.78, 1064.96, 1186.88, 1144.08, 1087.67, 1087.75, 1020.54];
rt_B = [1164.48, 1106.42, 1191.62, 1200.97, 1227.08, 1305.02, 1233.39, 1186.55, 1263.67, 1112.5];
rt_C = [1158.66, 1153.52, 1278.88, 1215.61, 1235.7, 1302.71, 1300.77, 1242.93, 1268.78, 1188.09];

dp_A = [1.629, 2.184, 1.515, 1.72, 1.526, 1.764, 0.553, 2.147, 1.625, 1.806];
dp_B = [0.681, 1.205, 1.458, 0.605, 1.878, 0.169, 1.252, 1.004, 0.65, 1.813];
dp_C = [0.612, 1.351, 0.817, 0.262, 1.353, -0.169, 0.839, 1.05, 0.212, 1.561];

% EEG ROI 数据
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
feature_names = {'Accuracy', 'RT', 'd''', 'N2 Amp', 'P3 Amp', 'N2 Lat', 'P3 Lat'};

%% ========================= 构建特征矩阵 =========================
% X: 30×7 (10被试 × 3条件, 7个特征)
% y: 30×1 (标签: 1=A, 2=B, 3=C)
% groups: 30×1 (被试编号, 用于 LOSO-CV)

X = []; y = []; groups = [];
for si = 1:nSubj
    % 条件 A
    X = [X; acc_A(si), rt_A(si), dp_A(si), n2_A(si), p3_A(si), n2lat_A(si), p3lat_A(si)];
    y = [y; 1]; groups = [groups; si];
    % 条件 B
    X = [X; acc_B(si), rt_B(si), dp_B(si), n2_B(si), p3_B(si), n2lat_B(si), p3lat_B(si)];
    y = [y; 2]; groups = [groups; si];
    % 条件 C
    X = [X; acc_C(si), rt_C(si), dp_C(si), n2_C(si), p3_C(si), n2lat_C(si), p3lat_C(si)];
    y = [y; 3]; groups = [groups; si];
end

nSamples = size(X, 1);
nFeatures = size(X, 2);
fprintf('样本数: %d, 特征数: %d\n', nSamples, nFeatures);
fprintf('条件分布: A=%d, B=%d, C=%d\n', sum(y==1), sum(y==2), sum(y==3));
fprintf('机会水平: %.1f%%\n\n', 100/3);

%% ========================= 定义9种分类器 =========================
algo_names = {
    'SVM (Linear)', ...
    'SVM (RBF)', ...
    'KNN (k=3)', ...
    'Decision Tree', ...
    'Random Forest', ...
    'Naive Bayes', ...
    'LDA', ...
    'AdaBoost', ...
    'Neural Network'
};
nAlgos = length(algo_names);

%% ========================= LOSO 交叉验证 =========================
fprintf('====== 开始 LOSO 交叉验证 ======\n');

accuracy_all = zeros(nAlgos, 1);
y_pred_all = zeros(nSamples, nAlgos);
confusion_all = zeros(3, 3, nAlgos);

for fold = 1:nSubj
    test_idx = (groups == fold);
    train_idx = ~test_idx;

    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);

    % Z-score 标准化（基于训练集）
    mu = mean(X_train);
    sigma = std(X_train);
    sigma(sigma == 0) = 1;
    X_train_z = (X_train - mu) ./ sigma;
    X_test_z = (X_test - mu) ./ sigma;

    for ai = 1:nAlgos
        switch ai
            case 1  % SVM Linear
                t = templateSVM('KernelFunction', 'linear', 'Standardize', false);
                mdl = fitcecoc(X_train_z, y_train, 'Learners', t);
            case 2  % SVM RBF
                t = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
                mdl = fitcecoc(X_train_z, y_train, 'Learners', t);
            case 3  % KNN
                mdl = fitcknn(X_train_z, y_train, 'NumNeighbors', 3);
            case 4  % Decision Tree
                mdl = fitctree(X_train_z, y_train, 'MaxNumSplits', 5);
            case 5  % Random Forest
                mdl = TreeBagger(100, X_train_z, y_train, ...
                    'Method', 'classification', 'MaxNumSplits', 5, ...
                    'OOBPrediction', 'off');
            case 6  % Naive Bayes
                mdl = fitcnb(X_train_z, y_train);
            case 7  % LDA
                mdl = fitcdiscr(X_train_z, y_train, 'DiscrimType', 'linear');
            case 8  % AdaBoost
                t = templateTree('MaxNumSplits', 3);
                mdl = fitcensemble(X_train_z, y_train, 'Method', 'AdaBoostM2', ...
                    'Learners', t, 'NumLearningCycles', 50);
            case 9  % Neural Network
                try
                    mdl = fitcnet(X_train_z, y_train, ...
                        'LayerSizes', [16 8], ...
                        'Activations', 'relu', ...
                        'Standardize', false);
                catch
                    % fitcnet 不可用时 (R2021a 之前)，用 fitcecoc+SVM 替代
                    t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 2);
                    mdl = fitcecoc(X_train_z, y_train, 'Learners', t);
                end
        end

        % 预测
        if ai == 5  % TreeBagger 返回 cell
            y_pred_cell = predict(mdl, X_test_z);
            y_pred = str2double(y_pred_cell);
        else
            y_pred = predict(mdl, X_test_z);
        end

        y_pred_all(test_idx, ai) = y_pred;
    end
end

%% ========================= 计算各算法指标 =========================
fprintf('\n====== 分类结果汇总 ======\n');
fprintf('%-20s  准确率(%%)\n', '算法');
fprintf('%s\n', repmat('-', 1, 35));

for ai = 1:nAlgos
    accuracy_all(ai) = sum(y_pred_all(:, ai) == y) / nSamples * 100;
    confusion_all(:,:,ai) = confusionmat(y, y_pred_all(:, ai));
    fprintf('%-20s  %.1f\n', algo_names{ai}, accuracy_all(ai));
end

fprintf('%s\n', repmat('-', 1, 35));
fprintf('机会水平:             33.3\n');

%% ========================= 图1: 分类准确率对比柱状图 =========================
[sorted_acc, sort_idx] = sort(accuracy_all, 'descend');
sorted_names = algo_names(sort_idx);

figure('Position', [100 100 800 450], 'Color', 'w');

% 根据准确率设定颜色渐变
cmap = [linspace(0.2, 0.9, nAlgos)', linspace(0.6, 0.95, nAlgos)', linspace(0.2, 0.3, nAlgos)'];
bar_h = barh(1:nAlgos, sorted_acc, 0.6);
bar_h.FaceColor = 'flat';
for bi = 1:nAlgos
    bar_h.CData(bi,:) = cmap(bi,:);
end
bar_h.EdgeColor = 'k';
bar_h.LineWidth = 0.5;

hold on;
line([100/3 100/3], [0 nAlgos+1], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
text(100/3 + 1, nAlgos + 0.3, sprintf('Chance = %.1f%%', 100/3), ...
    'Color', 'r', 'FontSize', 9);

for bi = 1:nAlgos
    text(sorted_acc(bi) + 1, bi, sprintf('%.1f%%', sorted_acc(bi)), ...
        'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'middle');
end

set(gca, 'YTick', 1:nAlgos, 'YTickLabel', sorted_names, 'FontSize', 10, ...
    'YDir', 'reverse', 'Box', 'off');
xlabel('Classification Accuracy (%)', 'FontSize', 12);
title({'9-Algorithm Classification Comparison', ...
    '(LOSO-CV, Behavioral + EEG Features)'}, 'FontSize', 13, 'FontWeight', 'bold');
xlim([0, max(sorted_acc) + 12]);

% 保存
try
    saveas(gcf, fullfile('ML_Classification_Comparison.png'));
    fprintf('\n已保存: ML_Classification_Comparison.png\n');
catch
    warning('保存失败');
end

%% ========================= 图2: 最佳算法的混淆矩阵 =========================
[~, best_idx] = max(accuracy_all);
best_cm = confusion_all(:,:,best_idx);

figure('Position', [100 100 500 400], 'Color', 'w');
imagesc(best_cm);
colormap(flipud(bone));
colorbar;

for i = 1:3
    for j = 1:3
        text(j, i, num2str(best_cm(i,j)), 'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
end

set(gca, 'XTick', 1:3, 'XTickLabel', {'A','B','C'}, ...
    'YTick', 1:3, 'YTickLabel', {'A','B','C'}, 'FontSize', 12);
xlabel('Predicted Label', 'FontSize', 12);
ylabel('True Label', 'FontSize', 12);
title(sprintf('Confusion Matrix - %s (%.1f%%)', algo_names{best_idx}, accuracy_all(best_idx)), ...
    'FontSize', 12, 'FontWeight', 'bold');

try
    saveas(gcf, fullfile('ML_Confusion_Matrix.png'));
    fprintf('已保存: ML_Confusion_Matrix.png\n');
catch
    warning('保存失败');
end

%% ========================= 图3: 特征重要性（Random Forest）=========================
% 训练完整数据集上的随机森林，获取特征重要性
X_z = zscore(X);
rf_model = TreeBagger(200, X_z, y, 'Method', 'classification', ...
    'OOBPredictorImportance', 'on', 'MaxNumSplits', 5);
importance = rf_model.OOBPermutedPredictorDeltaError;

[sorted_imp, imp_idx] = sort(importance, 'ascend');

figure('Position', [100 100 600 350], 'Color', 'w');
barh(1:nFeatures, sorted_imp, 0.5, 'FaceColor', [0.3 0.7 0.85], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);
set(gca, 'YTick', 1:nFeatures, 'YTickLabel', feature_names(imp_idx), ...
    'FontSize', 10, 'Box', 'off');
xlabel('Feature Importance (OOB Permutation)', 'FontSize', 11);
title('Random Forest Feature Importance', 'FontSize', 12, 'FontWeight', 'bold');

try
    saveas(gcf, fullfile('ML_Feature_Importance.png'));
    fprintf('已保存: ML_Feature_Importance.png\n');
catch
    warning('保存失败');
end

%% ========================= 图4: 各条件特征分布 (PCA 可视化) =========================
[coeff, score, ~, ~, explained] = pca(X_z);

figure('Position', [100 100 600 500], 'Color', 'w');
colors_pca = {'r', 'b', [0 0.6 0.4]};
markers = {'o', 's', 'd'};
cond_labels_pca = {'A', 'B', 'C'};

hold on;
for c = 1:3
    idx_c = (y == c);
    scatter(score(idx_c, 1), score(idx_c, 2), 80, colors_pca{c}, ...
        markers{c}, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end

% 画各条件的椭圆（95% CI）
for c = 1:3
    idx_c = (y == c);
    mu_c = mean(score(idx_c, 1:2));
    cov_c = cov(score(idx_c, 1:2));
    theta = linspace(0, 2*pi, 100);
    [V, D] = eig(cov_c);
    % 95% 置信椭圆 (chi2inv(0.95, 2) ≈ 5.991)
    r = sqrt(5.991);
    ellipse = r * [cos(theta); sin(theta)];
    ellipse = V * sqrt(D) * ellipse;
    plot(ellipse(1,:) + mu_c(1), ellipse(2,:) + mu_c(2), ...
        'Color', colors_pca{c}, 'LineWidth', 1.5, 'LineStyle', '--');
end

legend(cond_labels_pca, 'FontSize', 11, 'Location', 'best');
xlabel(sprintf('PC1 (%.1f%% variance)', explained(1)), 'FontSize', 11);
ylabel(sprintf('PC2 (%.1f%% variance)', explained(2)), 'FontSize', 11);
title('PCA Visualization of Condition Separability', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box off;

try
    saveas(gcf, fullfile('ML_PCA_Visualization.png'));
    fprintf('已保存: ML_PCA_Visualization.png\n');
catch
    warning('保存失败');
end

%% ========================= 导出结果表格 =========================
fprintf('\n====== 导出结果 ======\n');

% CSV: 算法对比
fid = fopen('ML_Results_Comparison.csv', 'w');
fprintf(fid, 'Rank,Algorithm,Accuracy(%%),Above_Chance(%%)\n');
for bi = 1:nAlgos
    fprintf(fid, '%d,%s,%.1f,%.1f\n', bi, sorted_names{bi}, sorted_acc(bi), sorted_acc(bi)-100/3);
end
fclose(fid);
fprintf('已导出: ML_Results_Comparison.csv\n');

% CSV: 混淆矩阵
fid = fopen('ML_Confusion_Matrices.csv', 'w');
fprintf(fid, 'Algorithm,Accuracy,TrueA_PredA,TrueA_PredB,TrueA_PredC,TrueB_PredA,TrueB_PredB,TrueB_PredC,TrueC_PredA,TrueC_PredB,TrueC_PredC\n');
for ai = 1:nAlgos
    cm = confusion_all(:,:,ai);
    fprintf(fid, '%s,%.1f,%d,%d,%d,%d,%d,%d,%d,%d,%d\n', ...
        algo_names{ai}, accuracy_all(ai), ...
        cm(1,1),cm(1,2),cm(1,3),cm(2,1),cm(2,2),cm(2,3),cm(3,1),cm(3,2),cm(3,3));
end
fclose(fid);
fprintf('已导出: ML_Confusion_Matrices.csv\n');

fprintf('\n====== 所有分析完成！ ======\n');
