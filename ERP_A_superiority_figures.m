%% ========================================================================
%% Part 12: A 条件优势证据 — 多维度显著性图
%% ========================================================================
%
%  从行为数据、ERP 成分、分类特征三个维度证明 A 条件优于 B/C
%  直接使用 Part 1 (ERP_visual_search_v6_fixed) 中已计算的变量
%
%  前提：先运行 ERP_visual_search_v6_fixed.m（Part 1-8 全部）
%        需要工作空间中存在: data, data_nt, data_diff, EEG, SubjFiles 等变量
%% ========================================================================

fprintf('\n========================================================\n');
fprintf('  Part 12: A 条件优势证据 — 多维度显著性图\n');
fprintf('========================================================\n\n');

nSubj = length(SubjFiles);
cond_names_abc = {'A', 'B', 'C'};
colors3 = [0.9 0.2 0.2; 0.2 0.5 0.9; 0.3 0.3 0.3];

%% ========================================================================
%% 图1: 行为正确率条件对比（从试次数推算）
%% ========================================================================
fprintf('====== 图1: 行为正确率 ======\n');

% 从 data 中提取每个被试每种条件的正确试次数
% data 维度: [nSubj nCond nChan nTime], 条件1-3为A/B/C正确, 4-6为A/B/C错误
% 我们需要从原始 epoch 数推算正确率

% 从之前的输出中手动提取（或从 .set 文件重新计算）
% 这里我们用一种更可靠的方式：利用已加载的数据重新扫描
correct_counts = zeros(nSubj, 3);
incorrect_counts = zeros(nSubj, 3);

for si = 1:nSubj
    fn = SubjFiles{si};
    EEG_tmp = pop_loadset('filename', fn, 'filepath', file_path);
    
    if ndims(EEG_tmp.data) == 3
        time_lock_types = zeros(1, EEG_tmp.trials);
        for ep = 1:EEG_tmp.trials
            ep_types = EEG_tmp.epoch(ep).eventtype; ep_lats = EEG_tmp.epoch(ep).eventlatency;
            if iscell(ep_lats), lats=cellfun(@double,ep_lats); else, lats=double(ep_lats); end
            [~,zi]=min(abs(lats));
            if iscell(ep_types), tl=ep_types{zi}; else, tl=ep_types(zi); end
            if ischar(tl)||isstring(tl), tl=str2double(tl); end
            time_lock_types(ep) = tl;
        end
        
        epoch_cond = zeros(1, EEG_tmp.trials);
        for ep = 1:EEG_tmp.trials
            if time_lock_types(ep) ~= 41, continue; end
            ep_types=EEG_tmp.epoch(ep).eventtype; ep_lats=EEG_tmp.epoch(ep).eventlatency;
            nev=length(ep_types); types_num=nan(1,nev); lats_num=nan(1,nev);
            for k=1:nev
                if iscell(ep_types),t=ep_types{k};else,t=ep_types(k);end
                if ischar(t)||isstring(t),t=str2double(t);end; types_num(k)=t;
                if iscell(ep_lats),lats_num(k)=double(ep_lats{k});else,lats_num(k)=double(ep_lats(k));end
            end
            resp_code=0;
            for ri=find(lats_num>0)
                if types_num(ri)==12,resp_code=1;break;end
                if types_num(ri)==22,resp_code=2;break;end
                if types_num(ri)==32,resp_code=3;break;end
            end
            if resp_code==0
                for ne=(ep+1):min(EEG_tmp.trials,ep+5)
                    tln=time_lock_types(ne);
                    if tln==12,resp_code=1;break;end
                    if tln==22,resp_code=2;break;end
                    if tln==32,resp_code=3;break;end
                    if ismember(tln,[11,21,31,41,42]),break;end
                end
            end
            stim_code=0;
            for si_k=find(lats_num<0)
                if ismember(types_num(si_k),[11,21,31]),stim_code=round(types_num(si_k)/10);break;end
            end
            if stim_code==0
                for pe=(ep-1):-1:max(1,ep-15)
                    tlp=time_lock_types(pe);
                    if ismember(tlp,[11,21,31]),stim_code=round(tlp/10);break;end
                    if ismember(tlp,[41,42]),break;end
                end
            end
            if stim_code>0 && resp_code>0
                epoch_cond(ep) = stim_code*100+resp_code;
            end
        end
        
        for ci = 1:3
            correct_counts(si, ci) = sum(epoch_cond == ci*100+1);
            incorrect_counts(si, ci) = sum(epoch_cond == ci*100+2);
        end
    end
    fprintf('  被试 %d: 正确[A=%d B=%d C=%d] 错误[A=%d B=%d C=%d]\n', si, ...
        correct_counts(si,1), correct_counts(si,2), correct_counts(si,3), ...
        incorrect_counts(si,1), incorrect_counts(si,2), incorrect_counts(si,3));
end

% 正确率 = 正确 / (正确+错误)  (不含无反应)
total_counts = correct_counts + incorrect_counts;
acc_rate = correct_counts ./ max(total_counts, 1);

% ---- 图1: 行为正确率柱状图 ----
figure('Name', '图1-行为正确率', 'NumberTitle', 'off', ...
    'Position', [50 50 500 450], 'Color', 'w');
mAcc = mean(acc_rate) * 100;
seAcc = std(acc_rate) / sqrt(nSubj) * 100;
b = bar(mAcc, 0.6, 'FaceColor', 'flat');
for k = 1:3, b.CData(k,:) = colors3(k,:); end
hold on;
errorbar(1:3, mAcc, seAcc, 'k.', 'LineWidth', 1.5, 'CapSize', 10);

% 配对比较
pairs = {[1,2],[1,3],[2,3]}; pair_labels = {'A vs B','A vs C','B vs C'};
ymax = max(mAcc + seAcc);
for pi = 1:3
    c1=pairs{pi}(1); c2=pairs{pi}(2);
    [~,p] = ttest(acc_rate(:,c1), acc_rate(:,c2));
    if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
    y = ymax + 2 + (pi-1)*4;
    line([c1 c2],[y y],'Color','k','LineWidth',1.5);
    line([c1 c1],[y-0.8 y],'Color','k','LineWidth',1.5);
    line([c2 c2],[y-0.8 y],'Color','k','LineWidth',1.5);
    text((c1+c2)/2, y+1, sprintf('%s (p=%.3f)',sig,p), 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
end
set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
ylabel('行为正确率 (%)', 'FontSize', 13);
title('三种刺激条件的目标检测正确率', 'FontSize', 14);
ylim([40 ymax+16]);
box off;

%% ========================================================================
%% 图2: 行为正确率个体连线图
%% ========================================================================
figure('Name', '图2-正确率个体连线', 'NumberTitle', 'off', ...
    'Position', [50 50 400 450], 'Color', 'w');
hold on;
for si = 1:nSubj
    plot(1:3, acc_rate(si,:)*100, 'o-', 'Color', [.6 .6 .6], 'LineWidth', 1, 'MarkerSize', 5);
end
plot(1:3, mAcc, 's-', 'Color', 'r', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
ylabel('正确率 (%)', 'FontSize', 13);
title('个体正确率变化趋势 (灰=个体, 红=均值)', 'FontSize', 13);
xlim([0.5 3.5]);
box off;

%% ========================================================================
%% 图3: 正确试次数条件对比
%% ========================================================================
figure('Name', '图3-正确试次数', 'NumberTitle', 'off', ...
    'Position', [50 50 500 450], 'Color', 'w');
mN = mean(correct_counts);
seN = std(correct_counts) / sqrt(nSubj);
b3 = bar(mN, 0.6, 'FaceColor', 'flat');
for k = 1:3, b3.CData(k,:) = colors3(k,:); end
hold on;
errorbar(1:3, mN, seN, 'k.', 'LineWidth', 1.5, 'CapSize', 10);

ymax3 = max(mN + seN);
for pi = 1:3
    c1=pairs{pi}(1); c2=pairs{pi}(2);
    [~,p] = ttest(correct_counts(:,c1), correct_counts(:,c2));
    if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
    y = ymax3 + 2 + (pi-1)*4;
    line([c1 c2],[y y],'Color','k','LineWidth',1.5);
    line([c1 c1],[y-0.8 y],'Color','k','LineWidth',1.5);
    line([c2 c2],[y-0.8 y],'Color','k','LineWidth',1.5);
    text((c1+c2)/2, y+1, sprintf('%s (p=%.3f)',sig,p), 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
end
set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
ylabel('正确试次数 (个)', 'FontSize', 13);
title('三种条件的正确检测试次数', 'FontSize', 14);
box off;

%% ========================================================================
%% 图4-5: ERP N2 振幅条件对比（PO7, PO8）
%% ========================================================================
fprintf('\n====== 图4-5: N2 振幅条件对比 ======\n');

% N2 时间窗
N2_win = [200 250];
N2_tidx = find(EEG.times >= N2_win(1) & EEG.times <= N2_win(2));

% 电极列表
erp_chans = {'PO7','PO8','Fz','Cz'};
erp_chan_idx = zeros(1, length(erp_chans));
all_labels = {EEG.chanlocs.labels};
for ci = 1:length(erp_chans)
    f = find(strcmpi(all_labels, erp_chans{ci}));
    if ~isempty(f), erp_chan_idx(ci) = f(1); end
end

% P3 时间窗
P3_win = [300 500];
P3_tidx = find(EEG.times >= P3_win(1) & EEG.times <= P3_win(2));

fig_count = 3;

% 对每个电极画 N2 和 P3
for chi = 1:length(erp_chans)
    ch = erp_chan_idx(chi);
    if ch == 0, continue; end
    ch_name = erp_chans{chi};
    
    % N2 平均振幅（在时间窗内取平均）
    N2_amp = zeros(nSubj, 3);
    P3_amp = zeros(nSubj, 3);
    N2_lat = zeros(nSubj, 3);
    P3_lat = zeros(nSubj, 3);
    
    for si = 1:nSubj
        for ci = 1:3
            wave = squeeze(data(si, ci, ch, :));
            % N2: 时间窗内平均振幅
            N2_amp(si, ci) = mean(wave(N2_tidx));
            % N2: 峰值潜伏期
            [~, mi] = min(wave(N2_tidx));
            N2_lat(si, ci) = EEG.times(N2_tidx(mi));
            % P3: 时间窗内平均振幅
            P3_amp(si, ci) = mean(wave(P3_tidx));
            % P3: 峰值潜伏期
            [~, mi] = max(wave(P3_tidx));
            P3_lat(si, ci) = EEG.times(P3_tidx(mi));
        end
    end
    
    % ---- N2 振幅柱状图 ----
    fig_count = fig_count + 1;
    figure('Name', sprintf('图%d-N2振幅_%s', fig_count, ch_name), 'NumberTitle', 'off', ...
        'Position', [50 50 500 450], 'Color', 'w');
    mV = mean(N2_amp);
    seV = std(N2_amp) / sqrt(nSubj);
    b = bar(mV, 0.6, 'FaceColor', 'flat');
    for k = 1:3, b.CData(k,:) = colors3(k,:); end
    hold on;
    errorbar(1:3, mV, seV, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
    
    ymin = min(mV - seV); ymax_n2 = max(mV + seV);
    for pi = 1:3
        c1=pairs{pi}(1); c2=pairs{pi}(2);
        [~,p] = ttest(N2_amp(:,c1), N2_amp(:,c2));
        if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
        y = ymin - 0.5 - (pi-1)*0.8;
        line([c1 c2],[y y],'Color','k','LineWidth',1.5);
        line([c1 c1],[y y+0.2],'Color','k','LineWidth',1.5);
        line([c2 c2],[y y+0.2],'Color','k','LineWidth',1.5);
        text((c1+c2)/2, y-0.3, sprintf('%s\np=%.3f',sig,p), 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    end
    set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
    ylabel('N2 平均振幅 (\muV)', 'FontSize', 13);
    title(sprintf('%s 电极 N2 振幅 (%d-%dms)', ch_name, N2_win(1), N2_win(2)), 'FontSize', 14);
    box off;
    
    % ---- P3 振幅柱状图 ----
    fig_count = fig_count + 1;
    figure('Name', sprintf('图%d-P3振幅_%s', fig_count, ch_name), 'NumberTitle', 'off', ...
        'Position', [50 50 500 450], 'Color', 'w');
    mV = mean(P3_amp);
    seV = std(P3_amp) / sqrt(nSubj);
    b = bar(mV, 0.6, 'FaceColor', 'flat');
    for k = 1:3, b.CData(k,:) = colors3(k,:); end
    hold on;
    errorbar(1:3, mV, seV, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
    
    ymaxp = max(mV + seV);
    for pi = 1:3
        c1=pairs{pi}(1); c2=pairs{pi}(2);
        [~,p] = ttest(P3_amp(:,c1), P3_amp(:,c2));
        if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
        y = ymaxp + 0.5 + (pi-1)*0.8;
        line([c1 c2],[y y],'Color','k','LineWidth',1.5);
        line([c1 c1],[y-0.2 y],'Color','k','LineWidth',1.5);
        line([c2 c2],[y-0.2 y],'Color','k','LineWidth',1.5);
        text((c1+c2)/2, y+0.3, sprintf('%s\np=%.3f',sig,p), 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    end
    set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
    ylabel('P3 平均振幅 (\muV)', 'FontSize', 13);
    title(sprintf('%s 电极 P3 振幅 (%d-%dms)', ch_name, P3_win(1), P3_win(2)), 'FontSize', 14);
    box off;
    
    % ---- P3 潜伏期柱状图 ----
    fig_count = fig_count + 1;
    figure('Name', sprintf('图%d-P3潜伏期_%s', fig_count, ch_name), 'NumberTitle', 'off', ...
        'Position', [50 50 500 450], 'Color', 'w');
    mL = mean(P3_lat);
    seL = std(P3_lat) / sqrt(nSubj);
    b = bar(mL, 0.6, 'FaceColor', 'flat');
    for k = 1:3, b.CData(k,:) = colors3(k,:); end
    hold on;
    errorbar(1:3, mL, seL, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
    
    ymaxl = max(mL + seL);
    for pi = 1:3
        c1=pairs{pi}(1); c2=pairs{pi}(2);
        [~,p] = ttest(P3_lat(:,c1), P3_lat(:,c2));
        if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
        y = ymaxl + 5 + (pi-1)*12;
        line([c1 c2],[y y],'Color','k','LineWidth',1.5);
        line([c1 c1],[y-2 y],'Color','k','LineWidth',1.5);
        line([c2 c2],[y-2 y],'Color','k','LineWidth',1.5);
        text((c1+c2)/2, y+4, sprintf('%s\np=%.3f',sig,p), 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    end
    set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
    ylabel('P3 峰值潜伏期 (ms)', 'FontSize', 13);
    title(sprintf('%s 电极 P3 潜伏期 (%d-%dms)', ch_name, P3_win(1), P3_win(2)), 'FontSize', 14);
    box off;
    
    % 打印统计
    fprintf('  [%s] N2振幅: A=%.2f±%.2f B=%.2f±%.2f C=%.2f±%.2f\n', ch_name, ...
        mean(N2_amp(:,1)),std(N2_amp(:,1)), mean(N2_amp(:,2)),std(N2_amp(:,2)), mean(N2_amp(:,3)),std(N2_amp(:,3)));
    fprintf('  [%s] P3振幅: A=%.2f±%.2f B=%.2f±%.2f C=%.2f±%.2f\n', ch_name, ...
        mean(P3_amp(:,1)),std(P3_amp(:,1)), mean(P3_amp(:,2)),std(P3_amp(:,2)), mean(P3_amp(:,3)),std(P3_amp(:,3)));
    fprintf('  [%s] P3潜伏期: A=%.1f±%.1f B=%.1f±%.1f C=%.1f±%.1f\n', ch_name, ...
        mean(P3_lat(:,1)),std(P3_lat(:,1)), mean(P3_lat(:,2)),std(P3_lat(:,2)), mean(P3_lat(:,3)),std(P3_lat(:,3)));
end

%% ========================================================================
%% 图: SSVEP 减法后的 ERP 成分对比（data_diff）
%% ========================================================================
if exist('data_diff', 'var')
    fprintf('\n====== SSVEP减法后 ERP 成分对比 ======\n');
    
    for chi = 1:length(erp_chans)
        ch = erp_chan_idx(chi);
        if ch == 0, continue; end
        ch_name = erp_chans{chi};
        
        P3_amp_diff = zeros(nSubj, 3);
        for si = 1:nSubj
            for ci = 1:3
                wave = squeeze(data_diff(si, ci, ch, :));
                P3_amp_diff(si, ci) = mean(wave(P3_tidx));
            end
        end
        
        fig_count = fig_count + 1;
        figure('Name', sprintf('图%d-目标诱发P3_%s', fig_count, ch_name), 'NumberTitle', 'off', ...
            'Position', [50 50 500 450], 'Color', 'w');
        mV = mean(P3_amp_diff);
        seV = std(P3_amp_diff) / sqrt(nSubj);
        b = bar(mV, 0.6, 'FaceColor', 'flat');
        for k = 1:3, b.CData(k,:) = colors3(k,:); end
        hold on;
        errorbar(1:3, mV, seV, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
        
        ymaxd = max(mV + seV);
        ymind = min(mV - seV);
        for pi = 1:3
            c1=pairs{pi}(1); c2=pairs{pi}(2);
            [~,p] = ttest(P3_amp_diff(:,c1), P3_amp_diff(:,c2));
            if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
            y = ymaxd + 0.3 + (pi-1)*0.7;
            line([c1 c2],[y y],'Color','k','LineWidth',1.5);
            line([c1 c1],[y-0.15 y],'Color','k','LineWidth',1.5);
            line([c2 c2],[y-0.15 y],'Color','k','LineWidth',1.5);
            text((c1+c2)/2, y+0.2, sprintf('%s (p=%.3f)',sig,p), 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
        end
        set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
        ylabel('目标诱发 P3 振幅 (\muV)', 'FontSize', 13);
        title(sprintf('%s 目标诱发P3 (SSVEP消除, %d-%dms)', ch_name, P3_win(1), P3_win(2)), 'FontSize', 14);
        box off;
        
        fprintf('  [%s] 目标诱发P3: A=%.2f±%.2f B=%.2f±%.2f C=%.2f±%.2f\n', ch_name, ...
            mean(P3_amp_diff(:,1)),std(P3_amp_diff(:,1)), ...
            mean(P3_amp_diff(:,2)),std(P3_amp_diff(:,2)), ...
            mean(P3_amp_diff(:,3)),std(P3_amp_diff(:,3)));
    end
end

%% ========================================================================
%% 图: 成对分类精度中 A 的优势（从 Part 10 数据）
%% ========================================================================
fprintf('\n====== 成对分类中 A 的优势 ======\n');

% 如果 pair_acc 存在（从 Part 10），用它
% 否则从数据中可以看到 A vs B/C 精度通常高于 B vs C
if exist('pair_acc', 'var')
    fig_count = fig_count + 1;
    figure('Name', sprintf('图%d-HDCA成对分类精度', fig_count), 'NumberTitle', 'off', ...
        'Position', [50 50 500 450], 'Color', 'w');
    
    % HDCA 的成对精度 (pair_acc: nSubj × 4alg × 3pairs)
    hdca_pair = squeeze(pair_acc(:, 1, :)) * 100;  % nSubj × 3pairs
    
    mP = mean(hdca_pair);
    seP = std(hdca_pair) / sqrt(nSubj);
    
    pair_colors = [0.8 0.2 0.6; 0.2 0.7 0.3; 0.5 0.5 0.5];
    b = bar(mP, 0.6, 'FaceColor', 'flat');
    for k = 1:3, b.CData(k,:) = pair_colors(k,:); end
    hold on;
    errorbar(1:3, mP, seP, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
    line([0.5 3.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--', 'LineWidth', 1.2);
    
    % t 检验 vs 50%
    for pi = 1:3
        vals = hdca_pair(:, pi);
        [~, p] = ttest(vals, 50);
        ypos = mP(pi) + seP(pi) + 1.5;
        if p<0.001, sig='***'; elseif p<0.01, sig='**'; elseif p<0.05, sig='*'; else, sig='n.s.'; end
        text(pi, ypos, sig, 'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color','r');
        text(pi, ypos + 3, sprintf('%.1f%%', mP(pi)), 'HorizontalAlignment','center','FontSize',10);
    end
    
    % 配对间比较
    [~, p_ab_bc] = ttest(hdca_pair(:,1), hdca_pair(:,3));  % AvB vs BvC
    [~, p_ac_bc] = ttest(hdca_pair(:,2), hdca_pair(:,3));  % AvC vs BvC
    
    ymaxp2 = max(mP+seP);
    y = ymaxp2 + 8;
    line([1 3],[y y],'Color','k','LineWidth',1.5);
    line([1 1],[y-1 y],'Color','k','LineWidth',1.5);
    line([3 3],[y-1 y],'Color','k','LineWidth',1.5);
    if p_ab_bc<0.05, sig_ab='*'; else, sig_ab='n.s.'; end
    text(2, y+1.5, sprintf('A参与 vs 无A: %s (p=%.3f)', sig_ab, p_ab_bc), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    
    set(gca, 'XTick', 1:3, 'XTickLabel', {'A vs B', 'A vs C', 'B vs C'}, 'FontSize', 12);
    ylabel('HDCA 分类精度 (%)', 'FontSize', 13);
    title('成对二分类: A 参与的配对精度更高', 'FontSize', 14);
    ylim([40 ymaxp2 + 16]);
    box off;
end

%% ========================================================================
%% 统计汇总表
%% ========================================================================
fprintf('\n====== 统计汇总 ======\n\n');

fprintf('--- 行为正确率 ---\n');
for ci = 1:3
    fprintf('  %s: %.1f%% ± %.1f%%\n', cond_names_abc{ci}, mean(acc_rate(:,ci))*100, std(acc_rate(:,ci))*100);
end
fprintf('  配对比较:\n');
for pi = 1:3
    c1=pairs{pi}(1); c2=pairs{pi}(2);
    [~,p,~,stats] = ttest(acc_rate(:,c1), acc_rate(:,c2));
    fprintf('    %s vs %s: t(%d)=%.3f, p=%.4f\n', cond_names_abc{c1}, cond_names_abc{c2}, stats.df, stats.tstat, p);
end

fprintf('\n--- 正确试次数 ---\n');
for ci = 1:3
    fprintf('  %s: %.1f ± %.1f\n', cond_names_abc{ci}, mean(correct_counts(:,ci)), std(correct_counts(:,ci)));
end
for pi = 1:3
    c1=pairs{pi}(1); c2=pairs{pi}(2);
    [~,p,~,stats] = ttest(correct_counts(:,c1), correct_counts(:,c2));
    fprintf('    %s vs %s: t(%d)=%.3f, p=%.4f\n', cond_names_abc{c1}, cond_names_abc{c2}, stats.df, stats.tstat, p);
end

%% ========================================================================
%% 保存
%% ========================================================================
if ~exist(save_figure_dir, 'dir'), mkdir(save_figure_dir); end
all_figs = findobj('Type', 'figure');
for fi = 1:length(all_figs)
    h = all_figs(fi); fn = get(h, 'Name');
    if isempty(fn), fn = sprintf('A_sup_Fig_%02d', fi); end
    fn = regexprep(fn, '[\\/:*?"<>|]', '_');
    try
        saveas(h, fullfile(save_figure_dir, [fn '.png']));
        savefig(h, fullfile(save_figure_dir, [fn '.fig']));
    catch, end
end

% 导出 CSV
fname = fullfile(saca_csv_dir, 'A_superiority_data.csv');
fid = fopen(fname, 'w');
fprintf(fid, 'Subject,AccRate_A,AccRate_B,AccRate_C,CorrectN_A,CorrectN_B,CorrectN_C\n');
for si = 1:nSubj
    [~,nm,~] = fileparts(SubjFiles{si});
    fprintf(fid, '%s,%.4f,%.4f,%.4f,%d,%d,%d\n', nm, ...
        acc_rate(si,1), acc_rate(si,2), acc_rate(si,3), ...
        correct_counts(si,1), correct_counts(si,2), correct_counts(si,3));
end
fclose(fid);
fprintf('\n已导出: %s\n', fname);

fprintf('\n====== Part 12 全部完成！ ======\n');
