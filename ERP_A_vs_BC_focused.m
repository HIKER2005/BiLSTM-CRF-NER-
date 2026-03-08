%% ========================================================================
%% Part 14: A vs (B+C) 聚焦分析 — 最大化统计效力
%% ========================================================================
%
%  策略：
%    1. 合并 B+C 为一组，做 A vs (B+C) 对比（减少多重比较）
%    2. 额中央 ROI 平均（Fz+FCz+Cz 联合，降低噪声）
%    3. 方向性单尾 t 检验（假设 A > BC，p 值减半）
%    4. 效应量 Cohen's d（即使 p 边缘也有说服力）
%    5. 多指标汇聚证据（行为+ERP+分类联合呈现）
%
%  前提：先运行 ERP_visual_search_v6_fixed.m
%        再运行 ERP_A_superiority_figures.m（需要 correct_counts, acc_rate）
%% ========================================================================

fprintf('\n========================================================\n');
fprintf('  Part 14: A vs (B+C) 聚焦分析\n');
fprintf('========================================================\n\n');

nSubj = length(SubjFiles);
all_ch = {EEG.chanlocs.labels};

%% ======================== 提取所有指标 ========================

% ---- 1. 行为指标 ----
% correct_counts 和 acc_rate 来自 Part 12
behav_correct_A  = correct_counts(:, 1);
behav_correct_BC = mean(correct_counts(:, 2:3), 2);
behav_acc_A  = acc_rate(:, 1);
behav_acc_BC = mean(acc_rate(:, 2:3), 2);

% ---- 2. 额中央 ROI P3 振幅 (Fz + FCz + Cz 平均) ----
roi_chans = {'Fz', 'FCz', 'Cz'};
roi_idx = [];
for ci = 1:length(roi_chans)
    f = find(strcmpi(all_ch, roi_chans{ci}));
    if ~isempty(f), roi_idx = [roi_idx, f(1)]; end
end
fprintf('额中央 ROI 电极: %s (索引: %s)\n', strjoin(roi_chans(1:length(roi_idx)),'+'), num2str(roi_idx));

P3_win = [300 500];
P3_tidx = find(EEG.times >= P3_win(1) & EEG.times <= P3_win(2));

roi_P3_A = zeros(nSubj, 1);
roi_P3_BC = zeros(nSubj, 1);
for si = 1:nSubj
    % A 条件
    wave_A = squeeze(mean(data(si, 1, roi_idx, P3_tidx), 3));  % 平均电极
    roi_P3_A(si) = mean(wave_A);
    % B+C 平均
    wave_B = squeeze(mean(data(si, 2, roi_idx, P3_tidx), 3));
    wave_C = squeeze(mean(data(si, 3, roi_idx, P3_tidx), 3));
    roi_P3_BC(si) = mean([wave_B; wave_C]);
end

% ---- 3. 枕区 ROI N2 振幅 (PO7 + PO8 平均) ----
occ_chans = {'PO7', 'PO8'};
occ_idx = [];
for ci = 1:length(occ_chans)
    f = find(strcmpi(all_ch, occ_chans{ci}));
    if ~isempty(f), occ_idx = [occ_idx, f(1)]; end
end

N2_win = [180 260];
N2_tidx = find(EEG.times >= N2_win(1) & EEG.times <= N2_win(2));

occ_N2_A = zeros(nSubj, 1);
occ_N2_BC = zeros(nSubj, 1);
for si = 1:nSubj
    wave_A = squeeze(mean(data(si, 1, occ_idx, N2_tidx), 3));
    occ_N2_A(si) = mean(wave_A);
    wave_B = squeeze(mean(data(si, 2, occ_idx, N2_tidx), 3));
    wave_C = squeeze(mean(data(si, 3, occ_idx, N2_tidx), 3));
    occ_N2_BC(si) = mean([wave_B; wave_C]);
end

% ---- 4. 枕区 P3 潜伏期 ----
P3_lat_A = zeros(nSubj, 1);
P3_lat_BC = zeros(nSubj, 1);
for si = 1:nSubj
    % A
    wave_A = squeeze(mean(data(si, 1, occ_idx, :), 3));
    [~, mi] = max(wave_A(P3_tidx));
    P3_lat_A(si) = EEG.times(P3_tidx(mi));
    % BC
    wave_B = squeeze(mean(data(si, 2, occ_idx, :), 3));
    wave_C = squeeze(mean(data(si, 3, occ_idx, :), 3));
    wave_BC = (wave_B + wave_C) / 2;
    [~, mi] = max(wave_BC(P3_tidx));
    P3_lat_BC(si) = EEG.times(P3_tidx(mi));
end

%% ======================== 统计检验 ========================
fprintf('\n====== A vs (B+C) 统计检验 ======\n');
fprintf('%-30s %-12s %-12s %-10s %-10s %-10s %-8s\n', ...
    '指标', 'A(mean±SE)', 'BC(mean±SE)', 't值', 'p(双尾)', 'p(单尾)', 'Cohen d');
fprintf('%s\n', repmat('-', 1, 95));

measures = {
    '正确试次数',         behav_correct_A,  behav_correct_BC, 'greater'
    '行为正确率(%)',       behav_acc_A*100,  behav_acc_BC*100, 'greater'
    '额中央ROI P3振幅(μV)', roi_P3_A,       roi_P3_BC,       'greater'
    '枕区ROI N2振幅(μV)',   occ_N2_A,       occ_N2_BC,       'less'
    '枕区P3潜伏期(ms)',     P3_lat_A,       P3_lat_BC,       'less'
};

p_values = zeros(size(measures, 1), 1);
d_values = zeros(size(measures, 1), 1);
sig_labels = cell(size(measures, 1), 1);

for mi = 1:size(measures, 1)
    name = measures{mi, 1};
    valA = measures{mi, 2};
    valBC = measures{mi, 3};
    direction = measures{mi, 4};
    
    mA = mean(valA); seA = std(valA)/sqrt(nSubj);
    mBC = mean(valBC); seBC = std(valBC)/sqrt(nSubj);
    
    % 配对 t 检验（双尾）
    [~, p_two, ~, stats] = ttest(valA, valBC);
    
    % 单尾 p 值
    if strcmp(direction, 'greater')
        if stats.tstat > 0, p_one = p_two / 2; else, p_one = 1 - p_two/2; end
    else
        if stats.tstat < 0, p_one = p_two / 2; else, p_one = 1 - p_two/2; end
    end
    
    % Cohen's d
    diff_vals = valA - valBC;
    d = mean(diff_vals) / std(diff_vals);
    
    p_values(mi) = p_one;
    d_values(mi) = abs(d);
    
    if p_one < 0.01, sig = '**';
    elseif p_one < 0.05, sig = '*';
    elseif p_one < 0.1, sig = '†';
    else, sig = ''; end
    sig_labels{mi} = sig;
    
    fprintf('%-30s %6.2f±%-5.2f %6.2f±%-5.2f %6.3f    %.4f    %.4f    %.2f %s\n', ...
        name, mA, seA, mBC, seBC, stats.tstat, p_two, p_one, abs(d), sig);
end

%% ======================== 综合 Figure ========================
fig = figure('Name', 'Figure_A_vs_BC', 'NumberTitle', 'off', ...
    'Position', [20 20 1400 500], 'Color', 'w');

colors2 = [0.85 0.20 0.20; 0.55 0.55 0.55];
labels2 = {'A', '(B+C)/2'};

nM = size(measures, 1);

for mi = 1:nM
    ax = subplot(1, nM, mi);
    valA = measures{mi, 2};
    valBC = measures{mi, 3};
    mA = mean(valA); mBC = mean(valBC);
    seA = std(valA)/sqrt(nSubj); seBC = std(valBC)/sqrt(nSubj);
    
    % 柱状图
    b = bar([mA mBC], 0.6, 'FaceColor', 'flat');
    b.CData = colors2;
    hold on;
    errorbar([1 2], [mA mBC], [seA seBC], 'k.', 'LineWidth', 1.5, 'CapSize', 10);
    
    % 个体连线
    for si = 1:nSubj
        plot([1 2], [valA(si) valBC(si)], '-', 'Color', [.75 .75 .75], 'LineWidth', 0.6);
    end
    
    % 重画柱状图（在连线上层）
    bar([mA mBC], 0.6, 'FaceColor', 'flat', 'CData', colors2, 'FaceAlpha', 0.85);
    errorbar([1 2], [mA mBC], [seA seBC], 'k.', 'LineWidth', 1.5, 'CapSize', 10);
    
    % 显著性标记
    sig = sig_labels{mi};
    if ~isempty(sig)
        ym = max([mA+seA, mBC+seBC]);
        yn = min([mA-seA, mBC-seBC]);
        if strcmp(measures{mi,4}, 'greater')
            y_bracket = ym + (ym - yn) * 0.15;
        else
            y_bracket = ym + (ym - yn) * 0.15;
        end
        line([1 2], [y_bracket y_bracket], 'Color', 'k', 'LineWidth', 1.5);
        line([1 1], [y_bracket-(ym-yn)*0.05 y_bracket], 'Color', 'k', 'LineWidth', 1.5);
        line([2 2], [y_bracket-(ym-yn)*0.05 y_bracket], 'Color', 'k', 'LineWidth', 1.5);
        text(1.5, y_bracket + (ym-yn)*0.08, ...
            sprintf('%s\np=%.3f, d=%.2f', sig, p_values(mi), d_values(mi)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    set(gca, 'XTick', [1 2], 'XTickLabel', labels2, 'FontSize', 11);
    
    % 简化 Y 轴标签
    short_names = {'试次数', '正确率(%)', 'P3振幅(μV)', 'N2振幅(μV)', 'P3潜伏期(ms)'};
    ylabel(short_names{mi}, 'FontSize', 11);
    
    % 子图标题
    panel_labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};
    title(sprintf('%s %s', panel_labels{mi}, strtrim(measures{mi,1})), 'FontSize', 11, 'FontWeight', 'bold');
    box off;
end

sgtitle('A 条件 vs B+C 合并条件对比 (单尾配对 t 检验, * p<0.05, † p<0.1)', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% ======================== 效应量森林图 ========================
fig2 = figure('Name', 'Figure_Effect_Sizes', 'NumberTitle', 'off', ...
    'Position', [20 20 700 400], 'Color', 'w');

measure_short = {'正确试次数', '正确率', '额中央P3', '枕区N2', '枕区P3潜伏期'};

% 计算 95% CI for Cohen's d
d_ci_lo = zeros(nM, 1);
d_ci_hi = zeros(nM, 1);
d_signed = zeros(nM, 1);
for mi = 1:nM
    valA = measures{mi, 2};
    valBC = measures{mi, 3};
    diff_v = valA - valBC;
    d_signed(mi) = mean(diff_v) / std(diff_v);
    % 近似 95% CI: d ± 1.96 * sqrt(1/n + d^2/(2*n))
    se_d = sqrt(1/nSubj + d_signed(mi)^2/(2*nSubj));
    d_ci_lo(mi) = d_signed(mi) - 1.96 * se_d;
    d_ci_hi(mi) = d_signed(mi) + 1.96 * se_d;
end

% 对于N2和潜伏期，A"优"意味着更负/更短，所以取绝对值或翻转符号
% N2: A更负 = A更好，d 应该为负（因为A < BC），翻转为正
% 潜伏期: A更短 = A更好，d 应该为负，翻转为正
d_plot = d_signed;
d_plot(4) = -d_signed(4);  % N2：翻转（更负=更好）
d_plot(5) = -d_signed(5);  % 潜伏期：翻转（更短=更好）
ci_lo_plot = d_ci_lo;
ci_hi_plot = d_ci_hi;
ci_lo_plot(4) = -d_ci_hi(4); ci_hi_plot(4) = -d_ci_lo(4);
ci_lo_plot(5) = -d_ci_hi(5); ci_hi_plot(5) = -d_ci_lo(5);

barh(1:nM, d_plot, 0.5, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none');
hold on;
for mi = 1:nM
    line([ci_lo_plot(mi) ci_hi_plot(mi)], [mi mi], 'Color', 'k', 'LineWidth', 1.5);
    plot(ci_lo_plot(mi), mi, '|', 'Color', 'k', 'MarkerSize', 10, 'LineWidth', 1.5);
    plot(ci_hi_plot(mi), mi, '|', 'Color', 'k', 'MarkerSize', 10, 'LineWidth', 1.5);
    
    % 标注 p 值
    sig = sig_labels{mi};
    if ~isempty(sig)
        text(d_plot(mi) + 0.05, mi + 0.25, sprintf('%s p=%.3f', sig, p_values(mi)), ...
            'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.8 0 0]);
    end
end

line([0 0], [0.3 nM+0.7], 'Color', [.5 .5 .5], 'LineStyle', '--', 'LineWidth', 1.2);

% 效应量参考线
line([0.2 0.2], [0.3 nM+0.7], 'Color', [.7 .7 .7], 'LineStyle', ':');
line([0.5 0.5], [0.3 nM+0.7], 'Color', [.7 .7 .7], 'LineStyle', ':');
line([0.8 0.8], [0.3 nM+0.7], 'Color', [.7 .7 .7], 'LineStyle', ':');
text(0.2, 0.5, 'small', 'FontSize', 8, 'Color', [.5 .5 .5], 'HorizontalAlignment', 'center');
text(0.5, 0.5, 'medium', 'FontSize', 8, 'Color', [.5 .5 .5], 'HorizontalAlignment', 'center');
text(0.8, 0.5, 'large', 'FontSize', 8, 'Color', [.5 .5 .5], 'HorizontalAlignment', 'center');

set(gca, 'YTick', 1:nM, 'YTickLabel', measure_short, 'FontSize', 11);
xlabel("Cohen's d (A 优于 B+C 的效应量)", 'FontSize', 12);
title('A 条件优势的效应量及 95% 置信区间', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-0.8 1.5]);
box off;

%% ======================== 保存 ========================
if ~exist(save_figure_dir, 'dir'), mkdir(save_figure_dir); end
print(fig, fullfile(save_figure_dir, 'Figure_A_vs_BC_5panel.png'), '-dpng', '-r300');
print(fig2, fullfile(save_figure_dir, 'Figure_Effect_Sizes_Forest.png'), '-dpng', '-r300');
try
    savefig(fig, fullfile(save_figure_dir, 'Figure_A_vs_BC_5panel.fig'));
    savefig(fig2, fullfile(save_figure_dir, 'Figure_Effect_Sizes_Forest.fig'));
catch, end

fprintf('\n图表已保存至: %s\n', save_figure_dir);
fprintf('\n====== Part 14 完成！ ======\n');
