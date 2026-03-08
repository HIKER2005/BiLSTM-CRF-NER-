%% ========================================================================
%% Part 13: 综合对比图 — 论文级 Figure
%% ========================================================================
%
%  将行为数据、ERP 成分、分类精度整合到 2-3 张论文级综合图中
%  前提：先运行 ERP_visual_search_v6_fixed.m + ERP_A_superiority_figures.m
%        需要工作空间中存在: data, data_diff, EEG, correct_counts, acc_rate 等
%% ========================================================================

fprintf('\n========================================================\n');
fprintf('  Part 13: 综合对比图 — 论文级 Figure\n');
fprintf('========================================================\n\n');

nSubj = length(SubjFiles);
cond_labels = {'A', 'B', 'C'};
colors3 = [0.85 0.20 0.20; 0.20 0.45 0.85; 0.25 0.25 0.25];

all_ch_labels = {EEG.chanlocs.labels};
N2_win = [200 250]; P3_win = [300 500];
N2_tidx = find(EEG.times >= N2_win(1) & EEG.times <= N2_win(2));
P3_tidx = find(EEG.times >= P3_win(1) & EEG.times <= P3_win(2));

target_chans = {'PO7','PO8','Fz','Cz'};
chan_idx = zeros(1,4);
for ci = 1:4
    f = find(strcmpi(all_ch_labels, target_chans{ci}));
    if ~isempty(f), chan_idx(ci) = f(1); end
end

%% ======================== Figure 1: 综合 6 面板 ========================
fig1 = figure('Name', 'Figure_Comprehensive', 'NumberTitle', 'off', ...
    'Position', [20 20 1400 900], 'Color', 'w');

% 统一参数
pairs = {[1,2],[1,3],[2,3]};

% ---- (a) 行为正确率 ----
ax1 = subplot(2,3,1);
mA = mean(acc_rate)*100; seA = std(acc_rate)/sqrt(nSubj)*100;
b1 = bar(mA, 0.65, 'FaceColor', 'flat');
for k=1:3, b1.CData(k,:)=colors3(k,:); end
hold on;
errorbar(1:3, mA, seA, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
add_sig_brackets(ax1, pairs, acc_rate, mA, seA, 'up', 2.5);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('正确率 (%)','FontSize',12);
title('(a) 行为正确率','FontSize',13,'FontWeight','bold');
box off;

% ---- (b) 正确试次数 ----
ax2 = subplot(2,3,2);
mN = mean(correct_counts); seN = std(correct_counts)/sqrt(nSubj);
b2 = bar(mN, 0.65, 'FaceColor', 'flat');
for k=1:3, b2.CData(k,:)=colors3(k,:); end
hold on;
errorbar(1:3, mN, seN, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
add_sig_brackets(ax2, pairs, correct_counts, mN, seN, 'up', 2);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('试次数 (个)','FontSize',12);
title('(b) 正确检测试次数','FontSize',13,'FontWeight','bold');
box off;

% ---- (c) PO7 N2 平均振幅 ----
ax3 = subplot(2,3,3);
ch = chan_idx(1);
N2_amp_po7 = zeros(nSubj,3);
for si=1:nSubj, for ci=1:3
    N2_amp_po7(si,ci) = mean(squeeze(data(si,ci,ch,N2_tidx)));
end, end
mV = mean(N2_amp_po7); seV = std(N2_amp_po7)/sqrt(nSubj);
b3 = bar(mV, 0.65, 'FaceColor', 'flat');
for k=1:3, b3.CData(k,:)=colors3(k,:); end
hold on;
errorbar(1:3, mV, seV, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
add_sig_brackets(ax3, pairs, N2_amp_po7, mV, seV, 'down', 0.5);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('振幅 (\muV)','FontSize',12);
title('(c) PO7 N2 振幅 (200-250ms)','FontSize',13,'FontWeight','bold');
box off;

% ---- (d) Fz P3 平均振幅 ----
ax4 = subplot(2,3,4);
ch_fz = chan_idx(3);
P3_amp_fz = zeros(nSubj,3);
for si=1:nSubj, for ci=1:3
    P3_amp_fz(si,ci) = mean(squeeze(data(si,ci,ch_fz,P3_tidx)));
end, end
mV4 = mean(P3_amp_fz); seV4 = std(P3_amp_fz)/sqrt(nSubj);
b4 = bar(mV4, 0.65, 'FaceColor', 'flat');
for k=1:3, b4.CData(k,:)=colors3(k,:); end
hold on;
errorbar(1:3, mV4, seV4, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
add_sig_brackets(ax4, pairs, P3_amp_fz, mV4, seV4, 'up', 0.4);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('振幅 (\muV)','FontSize',12);
title('(d) Fz P3 振幅 (300-500ms)','FontSize',13,'FontWeight','bold');
box off;

% ---- (e) Cz P3 平均振幅 ----
ax5 = subplot(2,3,5);
ch_cz = chan_idx(4);
P3_amp_cz = zeros(nSubj,3);
for si=1:nSubj, for ci=1:3
    P3_amp_cz(si,ci) = mean(squeeze(data(si,ci,ch_cz,P3_tidx)));
end, end
mV5 = mean(P3_amp_cz); seV5 = std(P3_amp_cz)/sqrt(nSubj);
b5 = bar(mV5, 0.65, 'FaceColor', 'flat');
for k=1:3, b5.CData(k,:)=colors3(k,:); end
hold on;
errorbar(1:3, mV5, seV5, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
add_sig_brackets(ax5, pairs, P3_amp_cz, mV5, seV5, 'up', 0.3);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('振幅 (\muV)','FontSize',12);
title('(e) Cz P3 振幅 (300-500ms)','FontSize',13,'FontWeight','bold');
box off;

% ---- (f) PO8 P3 潜伏期 ----
ax6 = subplot(2,3,6);
ch_po8 = chan_idx(2);
P3_lat_po8 = zeros(nSubj,3);
for si=1:nSubj, for ci=1:3
    wave = squeeze(data(si,ci,ch_po8,:));
    [~,mi] = max(wave(P3_tidx));
    P3_lat_po8(si,ci) = EEG.times(P3_tidx(mi));
end, end
mL = mean(P3_lat_po8); seL = std(P3_lat_po8)/sqrt(nSubj);
b6 = bar(mL, 0.65, 'FaceColor', 'flat');
for k=1:3, b6.CData(k,:)=colors3(k,:); end
hold on;
errorbar(1:3, mL, seL, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
add_sig_brackets(ax6, pairs, P3_lat_po8, mL, seL, 'up', 8);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('潜伏期 (ms)','FontSize',12);
title('(f) PO8 P3 潜伏期 (300-500ms)','FontSize',13,'FontWeight','bold');
box off;

sgtitle('三种刺激条件的行为表现与 ERP 成分对比', 'FontSize', 16, 'FontWeight', 'bold');

%% ======================== Figure 2: ERP 波形对比 ========================
fig2 = figure('Name', 'Figure_ERP_Waveforms', 'NumberTitle', 'off', ...
    'Position', [20 20 1200 700], 'Color', 'w');

for chi = 1:4
    ch = chan_idx(chi);
    if ch == 0, continue; end
    
    subplot(2,2,chi);
    hold on;
    
    for ci = 1:3
        wave = squeeze(mean(data(:,ci,ch,:), 1));
        plot(EEG.times, wave, 'Color', colors3(ci,:), 'LineWidth', 2);
    end
    
    % N2 时间窗阴影
    yl = ylim;
    fill([N2_win(1) N2_win(2) N2_win(2) N2_win(1)], ...
         [yl(1) yl(1) yl(2) yl(2)], [0.9 0.9 1], ...
         'EdgeColor', 'none', 'FaceAlpha', 0.3);
    % P3 时间窗阴影
    fill([P3_win(1) P3_win(2) P3_win(2) P3_win(1)], ...
         [yl(1) yl(1) yl(2) yl(2)], [1 0.9 0.9], ...
         'EdgeColor', 'none', 'FaceAlpha', 0.3);
    
    % 重画波形（在阴影上层）
    for ci = 1:3
        wave = squeeze(mean(data(:,ci,ch,:), 1));
        plot(EEG.times, wave, 'Color', colors3(ci,:), 'LineWidth', 2);
    end
    
    line([0 0], ylim, 'Color', [.5 .5 .5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [.5 .5 .5], 'LineStyle', '-');
    xlim([-100 800]);
    
    xlabel('时间 (ms)', 'FontSize', 11);
    ylabel('振幅 (\muV)', 'FontSize', 11);
    title(target_chans{chi}, 'FontSize', 14, 'FontWeight', 'bold');
    
    if chi == 1
        legend('A', 'B', 'C', 'Location', 'best', 'FontSize', 10);
    end
    box off;
    set(gca, 'YDir', 'normal');
end

sgtitle('三种刺激条件 ERP 波形对比 (蓝色=N2窗, 红色=P3窗)', 'FontSize', 15, 'FontWeight', 'bold');

%% ======================== Figure 3: 个体趋势 + 散点 ========================
fig3 = figure('Name', 'Figure_Individual_Trends', 'NumberTitle', 'off', ...
    'Position', [20 20 1100 400], 'Color', 'w');

% (a) 正确率个体连线
subplot(1,3,1); hold on;
for si = 1:nSubj
    plot(1:3, acc_rate(si,:)*100, 'o-', 'Color', [.7 .7 .7], 'LineWidth', 0.8, 'MarkerSize', 4);
end
plot(1:3, mean(acc_rate)*100, 's-', 'Color', [.8 .1 .1], 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', [.8 .1 .1]);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('正确率 (%)','FontSize',12); xlim([0.5 3.5]);
title('(a) 正确率','FontSize',13,'FontWeight','bold'); box off;

% (b) 正确试次数个体连线
subplot(1,3,2); hold on;
for si = 1:nSubj
    plot(1:3, correct_counts(si,:), 'o-', 'Color', [.7 .7 .7], 'LineWidth', 0.8, 'MarkerSize', 4);
end
plot(1:3, mean(correct_counts), 's-', 'Color', [.1 .1 .8], 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', [.1 .1 .8]);
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('试次数 (个)','FontSize',12); xlim([0.5 3.5]);
title('(b) 正确试次数','FontSize',13,'FontWeight','bold'); box off;

% (c) Fz P3 振幅个体连线
subplot(1,3,3); hold on;
for si = 1:nSubj
    plot(1:3, P3_amp_fz(si,:), 'o-', 'Color', [.7 .7 .7], 'LineWidth', 0.8, 'MarkerSize', 4);
end
plot(1:3, mean(P3_amp_fz), 's-', 'Color', [.1 .6 .1], 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', [.1 .6 .1]);
line([0.5 3.5], [0 0], 'Color', [.5 .5 .5], 'LineStyle', '--');
set(gca,'XTick',1:3,'XTickLabel',cond_labels,'FontSize',11);
ylabel('Fz P3 振幅 (\muV)','FontSize',12); xlim([0.5 3.5]);
title('(c) Fz P3 振幅','FontSize',13,'FontWeight','bold'); box off;

sgtitle('个体水平条件对比 (灰线=个体, 彩色=组平均)', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================== 保存 ========================
if ~exist(save_figure_dir, 'dir'), mkdir(save_figure_dir); end

% 高分辨率保存
print(fig1, fullfile(save_figure_dir, 'Figure_Comprehensive_6panel.png'), '-dpng', '-r300');
print(fig2, fullfile(save_figure_dir, 'Figure_ERP_Waveforms_4chan.png'), '-dpng', '-r300');
print(fig3, fullfile(save_figure_dir, 'Figure_Individual_Trends.png'), '-dpng', '-r300');

try
    savefig(fig1, fullfile(save_figure_dir, 'Figure_Comprehensive_6panel.fig'));
    savefig(fig2, fullfile(save_figure_dir, 'Figure_ERP_Waveforms_4chan.fig'));
    savefig(fig3, fullfile(save_figure_dir, 'Figure_Individual_Trends.fig'));
catch, end

fprintf('\n图表已保存至: %s\n', save_figure_dir);

%% ======================== 打印统计表（可直接复制到论文）========================
fprintf('\n====== 论文用统计表 ======\n\n');
fprintf('Table 1. 三种刺激条件的行为表现和 ERP 成分比较\n');
fprintf('%-20s %-15s %-15s %-15s %-10s\n', '指标', 'A (Mean±SE)', 'B (Mean±SE)', 'C (Mean±SE)', 'p(AvB/AvC/BvC)');
fprintf('%s\n', repmat('-', 1, 80));

% 行为正确率
[~,p1]=ttest(acc_rate(:,1),acc_rate(:,2));
[~,p2]=ttest(acc_rate(:,1),acc_rate(:,3));
[~,p3]=ttest(acc_rate(:,2),acc_rate(:,3));
fprintf('%-20s %.1f±%.1f%%     %.1f±%.1f%%     %.1f±%.1f%%     %.3f/%.3f/%.3f\n', ...
    '正确率(%)', mean(acc_rate(:,1))*100, std(acc_rate(:,1))/sqrt(nSubj)*100, ...
    mean(acc_rate(:,2))*100, std(acc_rate(:,2))/sqrt(nSubj)*100, ...
    mean(acc_rate(:,3))*100, std(acc_rate(:,3))/sqrt(nSubj)*100, p1,p2,p3);

% 正确试次数
[~,p1]=ttest(correct_counts(:,1),correct_counts(:,2));
[~,p2]=ttest(correct_counts(:,1),correct_counts(:,3));
[~,p3]=ttest(correct_counts(:,2),correct_counts(:,3));
fprintf('%-20s %.1f±%.1f       %.1f±%.1f       %.1f±%.1f       %.3f/%.3f/%.3f\n', ...
    '正确试次数', mean(correct_counts(:,1)), std(correct_counts(:,1))/sqrt(nSubj), ...
    mean(correct_counts(:,2)), std(correct_counts(:,2))/sqrt(nSubj), ...
    mean(correct_counts(:,3)), std(correct_counts(:,3))/sqrt(nSubj), p1,p2,p3);

% Fz P3
[~,p1]=ttest(P3_amp_fz(:,1),P3_amp_fz(:,2));
[~,p2]=ttest(P3_amp_fz(:,1),P3_amp_fz(:,3));
[~,p3]=ttest(P3_amp_fz(:,2),P3_amp_fz(:,3));
fprintf('%-20s %.2f±%.2f      %.2f±%.2f      %.2f±%.2f      %.3f/%.3f/%.3f\n', ...
    'Fz P3振幅(μV)', mean(P3_amp_fz(:,1)), std(P3_amp_fz(:,1))/sqrt(nSubj), ...
    mean(P3_amp_fz(:,2)), std(P3_amp_fz(:,2))/sqrt(nSubj), ...
    mean(P3_amp_fz(:,3)), std(P3_amp_fz(:,3))/sqrt(nSubj), p1,p2,p3);

% Cz P3
[~,p1]=ttest(P3_amp_cz(:,1),P3_amp_cz(:,2));
[~,p2]=ttest(P3_amp_cz(:,1),P3_amp_cz(:,3));
[~,p3]=ttest(P3_amp_cz(:,2),P3_amp_cz(:,3));
fprintf('%-20s %.2f±%.2f      %.2f±%.2f      %.2f±%.2f      %.3f/%.3f/%.3f\n', ...
    'Cz P3振幅(μV)', mean(P3_amp_cz(:,1)), std(P3_amp_cz(:,1))/sqrt(nSubj), ...
    mean(P3_amp_cz(:,2)), std(P3_amp_cz(:,2))/sqrt(nSubj), ...
    mean(P3_amp_cz(:,3)), std(P3_amp_cz(:,3))/sqrt(nSubj), p1,p2,p3);

fprintf('\n====== Part 13 完成！ ======\n');

%% ======================== 辅助函数 ========================
function add_sig_brackets(ax, pairs, data_mat, m, se, direction, gap)
    axes(ax); %#ok<LAXES>
    if strcmp(direction, 'up')
        ybase = max(m + se);
    else
        ybase = min(m - se);
    end
    
    for pi = 1:length(pairs)
        c1 = pairs{pi}(1); c2 = pairs{pi}(2);
        [~, p] = ttest(data_mat(:,c1), data_mat(:,c2));
        
        if p < 0.001, sig = '***';
        elseif p < 0.01, sig = '**';
        elseif p < 0.05, sig = '*';
        elseif p < 0.1, sig = '\dagger';
        else, sig = ''; end
        
        if isempty(sig), continue; end
        
        if strcmp(direction, 'up')
            y = ybase + gap * pi;
            tick = -gap * 0.3;
        else
            y = ybase - gap * pi;
            tick = gap * 0.3;
        end
        
        line([c1 c2], [y y], 'Color', 'k', 'LineWidth', 1.2);
        line([c1 c1], [y+tick y], 'Color', 'k', 'LineWidth', 1.2);
        line([c2 c2], [y+tick y], 'Color', 'k', 'LineWidth', 1.2);
        
        if strcmp(direction, 'up')
            text((c1+c2)/2, y + gap*0.4, sprintf('%s', sig), ...
                'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
                'Interpreter', 'tex');
        else
            text((c1+c2)/2, y - gap*0.4, sprintf('%s', sig), ...
                'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
                'Interpreter', 'tex');
        end
    end
end
