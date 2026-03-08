%% ========================================================================
%% Part 10: 增强版 ERP 分类分析 —— 4 种方案出显著性图
%% ========================================================================
%
%  方案1: 正确 vs 错误 二分类（全通道 HDCA）→ 类似论文目标/非目标
%  方案2: 时间解码曲线（滑动窗口分类精度随时间变化）→ 类似论文 Fig 2.13
%  方案3: 成对二分类 A-B/A-C/B-C + 统计检验 → 算法间差异显著性
%  方案4: 全通道 vs 4通道 → 空间信息对分类的贡献
%
%  前提：先运行 ERP_visual_search_v6_fixed.m (Part 1-8)
%        或确保 file_path / SubjFiles 等变量在工作空间中
%
%  输出：5-8 张出版级图表（带误差线、显著性标记、标准配色）
%% ========================================================================

fprintf('\n========================================================\n');
fprintf('  Part 10: 增强版分类分析 — 4 种方案出显著性图\n');
fprintf('========================================================\n\n');

%%% ===== 路径 =====
if ~exist('file_path', 'var')
    file_path = 'D:\实验一数据\闪烁光实验一\预处理结束\';
    save_figure_dir = 'C:\Users\wangw\Desktop\ERP-视觉搜索阶段-v6\v6_中性电极成分';
    saca_csv_dir = save_figure_dir;
    auto_scan = true;
    eeglab;
end

%%% ===== 事件标记 =====
if ~exist('stim_markers', 'var')
    stim_markers = [11, 21, 31];
    target_marker = 41; no_target_marker = 42;
    correct_marker = 12; incorrect_marker = 22; no_response_marker = 32;
    response_markers = [correct_marker, incorrect_marker, no_response_marker];
    epoch_window = [-0.1 0.8]; baseline_window = [-100 0];
end

%%% ===== 参数 =====
n_folds = 10;
hdca_win_ms = 40; hdca_step_ms = 40;

%%% ===== 被试扫描 =====
if ~exist('SubjFiles', 'var')
    if auto_scan
        set_files = dir(fullfile(file_path, '*.set'));
        SubjFiles = {set_files.name}';
    end
end
nSubj = length(SubjFiles);

%% ========================= 数据提取 =========================
% 对每个被试：提取单试次全通道 3D 数据 + 条件标签
% 同时提取正确和错误反应试次（用于方案1的二分类）

fprintf('====== 提取全通道单试次数据 ======\n');

subj_data = cell(nSubj, 1);     % {si}.X_3d, .Y_cond(1/2/3), .Y_acc(1=正确/2=错误)
global_srate = [];
global_times = [];
global_chanlocs = [];

for si = 1:nSubj
    fn = SubjFiles{si};
    fp = fullfile(file_path, fn);
    if ~exist(fp, 'file'), continue; end
    
    fprintf('  被试 %d/%d: %s\n', si, nSubj, fn);
    EEG = pop_loadset('filename', fn, 'filepath', file_path);
    
    if isempty(global_srate)
        global_srate = EEG.srate;
        global_times = EEG.times;
        global_chanlocs = EEG.chanlocs;
    end
    
    nChan = EEG.nbchan; nPts = EEG.pnts;
    
    % 复合打标（与 Part 1 相同逻辑）
    is_epoched = (ndims(EEG.data) == 3);
    
    X_trials = [];  Y_cond = [];  Y_acc = [];
    
    if is_epoched
        time_lock_types = zeros(1, EEG.trials);
        for ep = 1:EEG.trials
            ep_types = EEG.epoch(ep).eventtype;
            ep_lats  = EEG.epoch(ep).eventlatency;
            if iscell(ep_lats), lats = cellfun(@double, ep_lats); else, lats = double(ep_lats); end
            [~, zi] = min(abs(lats));
            if iscell(ep_types), tl = ep_types{zi}; else, tl = ep_types(zi); end
            if ischar(tl)||isstring(tl), tl = str2double(tl); end
            time_lock_types(ep) = tl;
        end
        
        epoch_cond = zeros(1, EEG.trials);
        for ep = 1:EEG.trials
            if time_lock_types(ep) ~= target_marker, continue; end
            ep_types = EEG.epoch(ep).eventtype;
            ep_lats  = EEG.epoch(ep).eventlatency;
            nev = length(ep_types);
            types_num = nan(1,nev); lats_num = nan(1,nev);
            for k = 1:nev
                if iscell(ep_types), t=ep_types{k}; else, t=ep_types(k); end
                if ischar(t)||isstring(t), t=str2double(t); end
                types_num(k)=t;
                if iscell(ep_lats), lats_num(k)=double(ep_lats{k}); else, lats_num(k)=double(ep_lats(k)); end
            end
            
            resp_code=0;
            for ri=find(lats_num>0)
                if types_num(ri)==correct_marker, resp_code=1; break; end
                if types_num(ri)==incorrect_marker, resp_code=2; break; end
                if types_num(ri)==no_response_marker, resp_code=3; break; end
            end
            if resp_code==0
                for ne=(ep+1):min(EEG.trials,ep+5)
                    tln=time_lock_types(ne);
                    if tln==correct_marker, resp_code=1; break; end
                    if tln==incorrect_marker, resp_code=2; break; end
                    if tln==no_response_marker, resp_code=3; break; end
                    if ismember(tln,[stim_markers,target_marker,no_target_marker]), break; end
                end
            end
            
            stim_code=0;
            for si_k=find(lats_num<0)
                if ismember(types_num(si_k), stim_markers)
                    stim_code=round(types_num(si_k)/10); break;
                end
            end
            if stim_code==0
                for pe=(ep-1):-1:max(1,ep-15)
                    tlp=time_lock_types(pe);
                    if ismember(tlp,stim_markers), stim_code=round(tlp/10); break; end
                    if ismember(tlp,[target_marker,no_target_marker]), break; end
                end
            end
            
            if stim_code > 0 && (resp_code == 1 || resp_code == 2)
                epoch_cond(ep) = stim_code * 100 + resp_code;
            end
        end
        
        for ep = 1:EEG.trials
            ec = epoch_cond(ep);
            if ec == 0, continue; end
            stim_class = floor(ec / 100);   % 1, 2, 3
            resp_class = mod(ec, 100);       % 1=正确, 2=错误
            
            trial_3d = EEG.data(:, :, ep);   % nChan × nPts
            X_trials = cat(3, X_trials, trial_3d);
            Y_cond = [Y_cond; stim_class];
            Y_acc  = [Y_acc; resp_class];
        end
    end
    
    % 存储
    sd = struct();
    if ~isempty(X_trials)
        sd.X = permute(X_trials, [3 1 2]);  % trials × nChan × nPts
    else
        sd.X = zeros(0, nChan, nPts);
    end
    sd.Y_cond = Y_cond;    % 刺激类型 1/2/3
    sd.Y_acc  = Y_acc;     % 反应正确性 1/2
    subj_data{si} = sd;
    
    nCorr = sum(Y_acc==1); nIncorr = sum(Y_acc==2);
    fprintf('    正确=%d, 错误=%d (A=%d B=%d C=%d)\n', nCorr, nIncorr, ...
        sum(Y_cond==1), sum(Y_cond==2), sum(Y_cond==3));
end

fprintf('数据提取完成\n\n');

%% ========================================================================
%% 方案 1: 正确 vs 错误 二分类（全通道 + 9 种算法）
%% ========================================================================
fprintf('====== 方案1: 正确 vs 错误 二分类（全通道）======\n');

% 只选时间窗 0-600ms 以减小维度
t_idx_spatial = find(global_times >= 0 & global_times <= 600);
nT_sp = length(t_idx_spatial);

acc_corr_vs_err = nan(nSubj, 4);  % 4种算法
alg4_names = {'HDCA', 'SVM', 'TRCA', 'eTRCA'};

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    X = sd.X(:, :, t_idx_spatial);
    Y = sd.Y_acc;  % 1=正确, 2=错误
    
    n1 = sum(Y==1); n2 = sum(Y==2);
    if n1 < 5 || n2 < 5, continue; end
    
    % 平衡类别：下采样多数类
    idx1 = find(Y==1); idx2 = find(Y==2);
    nMin = min(n1, n2);
    rng(si);
    idx1_sel = idx1(randperm(n1, nMin));
    idx2_sel = idx2(randperm(n2, nMin));
    sel = sort([idx1_sel; idx2_sel]);
    X = X(sel, :, :); Y = Y(sel);
    
    nTr = length(Y);
    if nTr < n_folds, continue; end
    cv = cvpartition(Y, 'KFold', n_folds);
    
    fprintf('  被试 %d: 正确=%d 错误=%d (平衡后各%d)\n', si, n1, n2, nMin);
    
    for ai = 1:4
        Yp = nan(nTr, 1);
        for fold = 1:n_folds
            tr = cv.training(fold); te = cv.test(fold);
            try
                switch ai
                    case 1, Yp(te) = hdca_binary(X(tr,:,:), Y(tr), X(te,:,:), global_srate, hdca_win_ms, hdca_step_ms);
                    case 2, Yp(te) = svm_binary_raw(X(tr,:,:), Y(tr), X(te,:,:), t_idx_spatial, global_times);
                    case 3, Yp(te) = trca_classify(X(tr,:,:), Y(tr), X(te,:,:));
                    case 4, Yp(te) = etrca_classify(X(tr,:,:), Y(tr), X(te,:,:));
                end
            catch
                Yp(te) = mode(Y(tr));
            end
        end
        v = ~isnan(Yp);
        acc_corr_vs_err(si, ai) = mean(Y(v) == Yp(v));
    end
    
    fprintf('    ');
    for ai = 1:4, fprintf('%s=%.1f%%  ', alg4_names{ai}, acc_corr_vs_err(si,ai)*100); end
    fprintf('\n');
end

% ---- 方案1出图：正确vs错误二分类 ----
figure('Name', '方案1-正确vs错误二分类', 'NumberTitle', 'off', ...
    'Position', [50 50 700 500], 'Color', 'w');

vs = ~isnan(acc_corr_vs_err(:,1));
mAcc = nanmean(acc_corr_vs_err) * 100;
sAcc = nanstd(acc_corr_vs_err) / sqrt(sum(vs)) * 100;

b = bar(mAcc, 0.6, 'FaceColor', 'flat');
cmap4 = [0.2 0.4 0.8; 0.9 0.2 0.2; 0.8 0.6 0.1; 0.1 0.5 0.8];
for k = 1:4, b.CData(k,:) = cmap4(k,:); end
hold on;
errorbar(1:4, mAcc, sAcc, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
line([0.5 4.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--', 'LineWidth', 1.2);
text(4.3, 50, '随机水平', 'Color', [.5 .5 .5], 'FontSize', 10);

% t 检验显著性标记
for ai = 1:4
    vals = acc_corr_vs_err(vs, ai);
    [~, p] = ttest(vals, 0.5);
    ypos = mAcc(ai) + sAcc(ai) + 2;
    if p < 0.001, sig = '***';
    elseif p < 0.01, sig = '**';
    elseif p < 0.05, sig = '*';
    else, sig = 'n.s.'; end
    text(ai, ypos, sig, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
    text(ai, mAcc(ai) + sAcc(ai) + 5.5, sprintf('%.1f%%', mAcc(ai)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

set(gca, 'XTick', 1:4, 'XTickLabel', alg4_names, 'FontSize', 12);
ylabel('分类精度 (%)', 'FontSize', 13);
title('正确 vs 错误反应 二分类精度 (全通道, 10-fold CV)', 'FontSize', 14);
ylim([30 100]);
box off;

%% ========================================================================
%% 方案 2: 时间解码曲线（分类精度随时间变化）
%% ========================================================================
fprintf('\n====== 方案2: 时间解码曲线 ======\n');

% 使用 HDCA 在不同时间起点做滑动窗口分类
decode_win_ms = 100;        % 解码窗口宽度
decode_step_ms = 20;        % 步长
decode_starts_ms = -50:decode_step_ms:700;
nDecodeWin = length(decode_starts_ms);

% 三分类时间解码
decode_acc_3class = nan(nSubj, nDecodeWin);
% 二分类（正确vs错误）时间解码
decode_acc_binary = nan(nSubj, nDecodeWin);

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    fprintf('  被试 %d/%d 时间解码...\n', si, nSubj);
    
    for di = 1:nDecodeWin
        t_start = decode_starts_ms(di);
        t_end = t_start + decode_win_ms;
        t_idx = find(global_times >= t_start & global_times <= t_end);
        if length(t_idx) < 5, continue; end
        
        % ---- 三分类: A vs B vs C (仅正确) ----
        mask3 = (sd.Y_acc == 1);
        X3 = sd.X(mask3, :, t_idx);
        Y3 = sd.Y_cond(mask3);
        if length(Y3) >= n_folds && length(unique(Y3)) >= 3
            cv3 = cvpartition(Y3, 'KFold', n_folds);
            Yp3 = nan(length(Y3), 1);
            for fold = 1:n_folds
                tr = cv3.training(fold); te = cv3.test(fold);
                try
                    Yp3(te) = hdca_multiclass(X3(tr,:,:), Y3(tr), X3(te,:,:), global_srate, 40, 40);
                catch
                    Yp3(te) = mode(Y3(tr));
                end
            end
            v = ~isnan(Yp3);
            decode_acc_3class(si, di) = mean(Y3(v) == Yp3(v));
        end
        
        % ---- 二分类: 正确 vs 错误 ----
        X2 = sd.X(:, :, t_idx);
        Y2 = sd.Y_acc;
        n1 = sum(Y2==1); n2 = sum(Y2==2);
        if n1 >= 5 && n2 >= 5
            nMin = min(n1, n2);
            idx1 = find(Y2==1); idx2 = find(Y2==2);
            rng(si+di*100);
            sel = sort([idx1(randperm(n1,nMin)); idx2(randperm(n2,nMin))]);
            X2b = X2(sel,:,:); Y2b = Y2(sel);
            if length(Y2b) >= n_folds
                cv2 = cvpartition(Y2b, 'KFold', n_folds);
                Yp2 = nan(length(Y2b), 1);
                for fold = 1:n_folds
                    tr = cv2.training(fold); te = cv2.test(fold);
                    try
                        Yp2(te) = hdca_binary(X2b(tr,:,:), Y2b(tr), X2b(te,:,:), global_srate, 40, 40);
                    catch
                        Yp2(te) = mode(Y2b(tr));
                    end
                end
                v = ~isnan(Yp2);
                decode_acc_binary(si, di) = mean(Y2b(v) == Yp2(v));
            end
        end
    end
end

% ---- 方案2出图：时间解码曲线 ----
figure('Name', '方案2-时间解码曲线', 'NumberTitle', 'off', ...
    'Position', [50 50 900 450], 'Color', 'w');
hold on;

center_times = decode_starts_ms + decode_win_ms/2;

% 三分类曲线
m3 = nanmean(decode_acc_3class) * 100;
se3 = nanstd(decode_acc_3class) / sqrt(sum(~isnan(decode_acc_3class(:,1)))) * 100;
fill([center_times, fliplr(center_times)], [m3+se3, fliplr(m3-se3)], ...
    [0.7 0.7 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(center_times, m3, 'b-', 'LineWidth', 2);

% 二分类曲线
m2 = nanmean(decode_acc_binary) * 100;
se2 = nanstd(decode_acc_binary) / sqrt(sum(~isnan(decode_acc_binary(:,1)))) * 100;
fill([center_times, fliplr(center_times)], [m2+se2, fliplr(m2-se2)], ...
    [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(center_times, m2, 'r-', 'LineWidth', 2);

% 标记显著高于随机水平的时间点（t 检验）
for di = 1:nDecodeWin
    vals3 = decode_acc_3class(:, di);
    vals3 = vals3(~isnan(vals3));
    if length(vals3) >= 5
        [~, p3] = ttest(vals3, 1/3);
        if p3 < 0.05
            plot(center_times(di), 32, 'b.', 'MarkerSize', 8);
        end
    end
    vals2 = decode_acc_binary(:, di);
    vals2 = vals2(~isnan(vals2));
    if length(vals2) >= 5
        [~, p2] = ttest(vals2, 0.5);
        if p2 < 0.05
            plot(center_times(di), 30, 'r.', 'MarkerSize', 8);
        end
    end
end

line([0 0], ylim, 'Color', [.5 .5 .5], 'LineStyle', '--');
line(xlim, [33.33 33.33], 'Color', 'b', 'LineStyle', ':', 'LineWidth', 1);
line(xlim, [50 50], 'Color', 'r', 'LineStyle', ':', 'LineWidth', 1);

xlabel('时间 (ms)', 'FontSize', 13);
ylabel('分类精度 (%)', 'FontSize', 13);
title('HDCA 时间解码曲线 (全通道)', 'FontSize', 14);
legend({'A vs B vs C ±SE', 'A vs B vs C (三分类)', ...
        '正确 vs 错误 ±SE', '正确 vs 错误 (二分类)'}, ...
    'Location', 'best', 'FontSize', 10);
text(550, 34, '三分类随机', 'Color', 'b', 'FontSize', 9);
text(550, 51, '二分类随机', 'Color', 'r', 'FontSize', 9);
xlim([-50 750]);
box off;

%% ========================================================================
%% 方案 3: 成对二分类 A-B / A-C / B-C + 算法对比统计
%% ========================================================================
fprintf('\n====== 方案3: 成对二分类 + 算法间统计比较 ======\n');

pairs = {[1,2], [1,3], [2,3]};
pair_names = {'A vs B', 'A vs C', 'B vs C'};
nPairs = 3;

% 4 种算法 × 3 种配对 × 10 被试
pair_acc = nan(nSubj, 4, nPairs);

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    mask_corr = (sd.Y_acc == 1);
    X_all = sd.X(mask_corr, :, t_idx_spatial);
    Y_all = sd.Y_cond(mask_corr);
    
    for pi = 1:nPairs
        c1 = pairs{pi}(1); c2 = pairs{pi}(2);
        mask_pair = (Y_all == c1 | Y_all == c2);
        Xp = X_all(mask_pair, :, :);
        Yp = Y_all(mask_pair);
        
        if length(Yp) < n_folds || length(unique(Yp)) < 2, continue; end
        cv = cvpartition(Yp, 'KFold', n_folds);
        
        for ai = 1:4
            Ypred = nan(length(Yp), 1);
            for fold = 1:n_folds
                tr = cv.training(fold); te = cv.test(fold);
                try
                    switch ai
                        case 1, Ypred(te) = hdca_binary(Xp(tr,:,:), Yp(tr), Xp(te,:,:), global_srate, hdca_win_ms, hdca_step_ms);
                        case 2, Ypred(te) = svm_binary_raw(Xp(tr,:,:), Yp(tr), Xp(te,:,:), t_idx_spatial, global_times);
                        case 3, Ypred(te) = trca_classify(Xp(tr,:,:), Yp(tr), Xp(te,:,:));
                        case 4, Ypred(te) = etrca_classify(Xp(tr,:,:), Yp(tr), Xp(te,:,:));
                    end
                catch
                    Ypred(te) = mode(Yp(tr));
                end
            end
            v = ~isnan(Ypred);
            pair_acc(si, ai, pi) = mean(Yp(v) == Ypred(v));
        end
    end
    
    fprintf('  被试 %d:', si);
    for pi = 1:nPairs
        fprintf('  %s[', pair_names{pi});
        for ai = 1:4, fprintf('%.0f', pair_acc(si,ai,pi)*100); if ai<4, fprintf('/'); end; end
        fprintf(']');
    end
    fprintf('\n');
end

% ---- 方案3出图：成对二分类柱状图 ----
figure('Name', '方案3-成对二分类', 'NumberTitle', 'off', ...
    'Position', [50 50 1000 450], 'Color', 'w');

for pi = 1:nPairs
    subplot(1, 3, pi);
    mA = squeeze(nanmean(pair_acc(:, :, pi))) * 100;
    seA = squeeze(nanstd(pair_acc(:, :, pi))) / sqrt(nSubj) * 100;
    
    b = bar(mA, 0.65, 'FaceColor', 'flat');
    for k = 1:4, b.CData(k,:) = cmap4(k,:); end
    hold on;
    errorbar(1:4, mA, seA, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
    line([0.5 4.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--');
    
    % t 检验
    for ai = 1:4
        vals = pair_acc(~isnan(pair_acc(:,ai,pi)), ai, pi);
        if length(vals) >= 5
            [~, p] = ttest(vals, 0.5);
            yp = mA(ai) + seA(ai) + 2;
            if p < 0.001, sig = '***';
            elseif p < 0.01, sig = '**';
            elseif p < 0.05, sig = '*';
            else, sig = ''; end
            text(ai, yp, sig, 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'r');
        end
    end
    
    set(gca, 'XTick', 1:4, 'XTickLabel', alg4_names, 'FontSize', 10);
    ylabel('精度 (%)'); ylim([35 85]);
    title(pair_names{pi}, 'FontSize', 13, 'FontWeight', 'bold');
    box off;
end
sgtitle('成对二分类精度 (全通道, 仅正确反应试次)', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% 方案 4: 全通道 vs 4 通道对比
%% ========================================================================
fprintf('\n====== 方案4: 全通道 vs 4通道 ======\n');

ch_labels = {global_chanlocs.labels};
four_chan_idx = [];
for ci = 1:length({'PO7','PO8','Fz','Cz'})
    names4 = {'PO7','PO8','Fz','Cz'};
    f = find(strcmpi(ch_labels, names4{ci}));
    if ~isempty(f), four_chan_idx = [four_chan_idx, f(1)]; end
end

acc_full = nan(nSubj, 1);  % 全通道 HDCA
acc_4ch  = nan(nSubj, 1);  % 4通道 HDCA

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    mask_corr = (sd.Y_acc == 1);
    Y = sd.Y_cond(mask_corr);
    if length(Y) < n_folds || length(unique(Y)) < 3, continue; end
    
    X_full = sd.X(mask_corr, :, t_idx_spatial);
    X_4ch  = sd.X(mask_corr, four_chan_idx, t_idx_spatial);
    
    cv = cvpartition(Y, 'KFold', n_folds);
    Yp_f = nan(length(Y),1); Yp_4 = nan(length(Y),1);
    
    for fold = 1:n_folds
        tr = cv.training(fold); te = cv.test(fold);
        try
            Yp_f(te) = hdca_multiclass(X_full(tr,:,:), Y(tr), X_full(te,:,:), global_srate, hdca_win_ms, hdca_step_ms);
        catch, Yp_f(te) = mode(Y(tr)); end
        try
            Yp_4(te) = hdca_multiclass(X_4ch(tr,:,:), Y(tr), X_4ch(te,:,:), global_srate, hdca_win_ms, hdca_step_ms);
        catch, Yp_4(te) = mode(Y(tr)); end
    end
    
    v = ~isnan(Yp_f); acc_full(si) = mean(Y(v) == Yp_f(v));
    v = ~isnan(Yp_4); acc_4ch(si)  = mean(Y(v) == Yp_4(v));
    fprintf('  被试 %d: 全通道=%.1f%% 4通道=%.1f%%\n', si, acc_full(si)*100, acc_4ch(si)*100);
end

% ---- 方案4出图：配对比较 ----
figure('Name', '方案4-全通道vs4通道', 'NumberTitle', 'off', ...
    'Position', [50 50 500 450], 'Color', 'w');

vs = ~isnan(acc_full) & ~isnan(acc_4ch);
mF = mean(acc_full(vs))*100; mC = mean(acc_4ch(vs))*100;
seF = std(acc_full(vs))/sqrt(sum(vs))*100;
seC = std(acc_4ch(vs))/sqrt(sum(vs))*100;

b = bar([mC, mF], 0.5, 'FaceColor', 'flat');
b.CData = [0.7 0.7 0.7; 0.2 0.5 0.9];
hold on;
errorbar([1 2], [mC mF], [seC seF], 'k.', 'LineWidth', 1.5, 'CapSize', 10);
line([0.5 2.5], [33.33 33.33], 'Color', [.5 .5 .5], 'LineStyle', '--');

% 配对 t 检验
[~, p_paired] = ttest(acc_full(vs), acc_4ch(vs));
y_bracket = max(mF+seF, mC+seC) + 4;
line([1 2], [y_bracket y_bracket], 'Color', 'k', 'LineWidth', 1.5);
line([1 1], [y_bracket-1 y_bracket], 'Color', 'k', 'LineWidth', 1.5);
line([2 2], [y_bracket-1 y_bracket], 'Color', 'k', 'LineWidth', 1.5);
if p_paired < 0.001, ps = sprintf('p < 0.001 ***');
elseif p_paired < 0.01, ps = sprintf('p = %.3f **', p_paired);
elseif p_paired < 0.05, ps = sprintf('p = %.3f *', p_paired);
else, ps = sprintf('p = %.3f n.s.', p_paired); end
text(1.5, y_bracket + 2, ps, 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'XTick', [1 2], 'XTickLabel', {'4通道 (PO7/PO8/Fz/Cz)', '全通道 (59通道)'}, 'FontSize', 11);
ylabel('HDCA 三分类精度 (%)', 'FontSize', 13);
title('空间信息对分类的贡献', 'FontSize', 14);
ylim([25 85]);
box off;

% 配对连线图
figure('Name', '方案4-个体配对图', 'NumberTitle', 'off', ...
    'Position', [50 50 400 450], 'Color', 'w');
hold on;
for si = find(vs')
    plot([1 2], [acc_4ch(si) acc_full(si)]*100, 'o-', 'Color', [.6 .6 .6], 'LineWidth', 1.2, 'MarkerSize', 6);
end
plot([1 2], [mC mF], 's-', 'Color', 'r', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
set(gca, 'XTick', [1 2], 'XTickLabel', {'4通道', '全通道'}, 'FontSize', 12);
ylabel('HDCA 三分类精度 (%)', 'FontSize', 13);
title(sprintf('全通道 vs 4通道 (%s)', ps), 'FontSize', 13);
xlim([0.5 2.5]);
box off;

%% ========================= 汇总图：所有方案 =========================
figure('Name', '汇总-四种分类任务对比', 'NumberTitle', 'off', ...
    'Position', [50 50 900 500], 'Color', 'w');

% 收集所有结果
task_names = {'三分类\n(A vs B vs C)\n4通道', '三分类\n(A vs B vs C)\n全通道', ...
              '二分类\n(正确vs错误)\n全通道', '成对均值\n(A-B/A-C/B-C)\n全通道'};
task_acc_mean = nan(1, 4);
task_acc_se = nan(1, 4);

% 三分类4通道
vs4 = ~isnan(acc_4ch);
task_acc_mean(1) = mean(acc_4ch(vs4))*100;
task_acc_se(1) = std(acc_4ch(vs4))/sqrt(sum(vs4))*100;

% 三分类全通道
vsf = ~isnan(acc_full);
task_acc_mean(2) = mean(acc_full(vsf))*100;
task_acc_se(2) = std(acc_full(vsf))/sqrt(sum(vsf))*100;

% 二分类全通道 (HDCA)
vs1 = ~isnan(acc_corr_vs_err(:,1));
task_acc_mean(3) = mean(acc_corr_vs_err(vs1,1))*100;
task_acc_se(3) = std(acc_corr_vs_err(vs1,1))/sqrt(sum(vs1))*100;

% 成对均值 (HDCA)
pair_hdca_mean = squeeze(nanmean(pair_acc(:,1,:), 3));
vsp = ~isnan(pair_hdca_mean);
task_acc_mean(4) = mean(pair_hdca_mean(vsp))*100;
task_acc_se(4) = std(pair_hdca_mean(vsp))/sqrt(sum(vsp))*100;

colors_summary = [0.6 0.6 0.6; 0.2 0.5 0.9; 0.9 0.2 0.2; 0.2 0.7 0.3];
b = bar(task_acc_mean, 0.6, 'FaceColor', 'flat');
for k = 1:4, b.CData(k,:) = colors_summary(k,:); end
hold on;
errorbar(1:4, task_acc_mean, task_acc_se, 'k.', 'LineWidth', 1.5, 'CapSize', 10);

% 随机水平线
line([0.5 2.5], [33.33 33.33], 'Color', 'b', 'LineStyle', '--', 'LineWidth', 1);
line([2.5 3.5], [50 50], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1);
line([3.5 4.5], [50 50], 'Color', [0 .6 0], 'LineStyle', '--', 'LineWidth', 1);

for k = 1:4
    text(k, task_acc_mean(k) + task_acc_se(k) + 2, sprintf('%.1f%%', task_acc_mean(k)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

set(gca, 'XTick', 1:4, 'XTickLabel', {'三分类\n4通道', '三分类\n全通道', '二分类\n正确vs错误', '成对均值\nA-B/A-C/B-C'}, ...
    'FontSize', 10);
ylabel('HDCA 分类精度 (%)', 'FontSize', 13);
title('四种分类任务 HDCA 精度对比', 'FontSize', 14);
ylim([20 90]);
box off;

%% ========================= 导出 CSV =========================
fname = fullfile(saca_csv_dir, 'Enhanced_Classification_Results.csv');
fid = fopen(fname, 'w');
fprintf(fid, 'Subject,3class_4ch,3class_allCh,Binary_CorrErr_HDCA,Binary_CorrErr_SVM,Binary_CorrErr_TRCA,Binary_CorrErr_eTRCA');
for pi = 1:nPairs
    for ai = 1:4
        fprintf(fid, ',%s_%s', pair_names{pi}, alg4_names{ai});
    end
end
fprintf(fid, '\n');
for si = 1:nSubj
    [~,nm,~] = fileparts(SubjFiles{si});
    fprintf(fid, '%s,%.4f,%.4f', nm, acc_4ch(si), acc_full(si));
    for ai = 1:4, fprintf(fid, ',%.4f', acc_corr_vs_err(si,ai)); end
    for pi = 1:nPairs
        for ai = 1:4, fprintf(fid, ',%.4f', pair_acc(si,ai,pi)); end
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('\n结果已导出: %s\n', fname);

%% ========================= 保存图表 =========================
if ~exist(save_figure_dir, 'dir'), mkdir(save_figure_dir); end
all_figs = findobj('Type', 'figure');
for fi = 1:length(all_figs)
    h = all_figs(fi); fn_fig = get(h, 'Name');
    if isempty(fn_fig), fn_fig = sprintf('Enhanced_Fig_%02d', fi); end
    fn_fig = regexprep(fn_fig, '[\\/:*?"<>|]', '_');
    try
        saveas(h, fullfile(save_figure_dir, [fn_fig '.png']));
        savefig(h, fullfile(save_figure_dir, [fn_fig '.fig']));
    catch, end
end

fprintf('\n====== Part 10 全部完成！ ======\n');

%% ========================================================================
%% 辅助函数
%% ========================================================================

function Y_pred = hdca_binary(Xtr, Ytr, Xte, srate, win_ms, step_ms)
    [~, nC, nT] = size(Xtr);
    ws = max(1, round(win_ms/1000*srate));
    ss = max(1, round(step_ms/1000*srate));
    starts = 1:ss:(nT-ws+1);
    nW = length(starts);
    
    classes = unique(Ytr);
    idx1 = (Ytr==classes(1)); idx0 = (Ytr==classes(2));
    
    proj_tr = zeros(size(Xtr,1), nW);
    proj_te = zeros(size(Xte,1), nW);
    
    for wi = 1:nW
        tidx = starts(wi):min(starts(wi)+ws-1, nT);
        X1 = squeeze(mean(Xtr(idx1,:,tidx),3));
        X0 = squeeze(mean(Xtr(idx0,:,tidx),3));
        if size(X1,1)==1, X1=X1(:)'; end
        if size(X0,1)==1, X0=X0(:)'; end
        Sw = cov(X1)+cov(X0)+eye(nC)*1e-6;
        w = Sw\(mean(X1,1)'-mean(X0,1)');
        proj_tr(:,wi) = squeeze(mean(Xtr(:,:,tidx),3))*w;
        proj_te(:,wi) = squeeze(mean(Xte(:,:,tidx),3))*w;
    end
    
    mdl = fitcdiscr(proj_tr, Ytr, 'DiscrimType', 'linear');
    Y_pred = predict(mdl, proj_te);
end

function Y_pred = hdca_multiclass(Xtr, Ytr, Xte, srate, win_ms, step_ms)
    [~, nC, nT] = size(Xtr);
    ws = max(1, round(win_ms/1000*srate));
    ss = max(1, round(step_ms/1000*srate));
    starts = 1:ss:(nT-ws+1);
    nW = length(starts);
    classes = unique(Ytr); nCls = length(classes);
    
    proj_tr = zeros(size(Xtr,1), nW*nCls);
    proj_te = zeros(size(Xte,1), nW*nCls);
    
    for ci = 1:nCls
        Ybin = double(Ytr==classes(ci));
        for wi = 1:nW
            tidx = starts(wi):min(starts(wi)+ws-1, nT);
            X1 = squeeze(mean(Xtr(Ybin==1,:,tidx),3));
            X0 = squeeze(mean(Xtr(Ybin==0,:,tidx),3));
            if size(X1,1)==1, X1=X1(:)'; end
            if size(X0,1)==1, X0=X0(:)'; end
            Sw = cov(X1)+cov(X0)+eye(nC)*1e-6;
            w = Sw\(mean(X1,1)'-mean(X0,1)');
            col = (ci-1)*nW+wi;
            Xtr_w = squeeze(mean(Xtr(:,:,tidx),3));
            Xte_w = squeeze(mean(Xte(:,:,tidx),3));
            if size(Xtr_w,2)~=nC, Xtr_w=reshape(Xtr_w,[],nC); end
            if size(Xte_w,2)~=nC, Xte_w=reshape(Xte_w,[],nC); end
            proj_tr(:,col) = Xtr_w*w;
            proj_te(:,col) = Xte_w*w;
        end
    end
    
    mdl = fitcdiscr(proj_tr, Ytr, 'DiscrimType', 'linear');
    Y_pred = predict(mdl, proj_te);
end

function Y_pred = svm_binary_raw(Xtr, Ytr, Xte, ~, ~)
    Ftr = reshape(Xtr, size(Xtr,1), []);
    Fte = reshape(Xte, size(Xte,1), []);
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte-mu)./sg;
    mdl = fitcsvm(Ftr, Ytr, 'KernelFunction', 'rbf', 'Standardize', false);
    Y_pred = predict(mdl, Fte);
end

function Y_pred = trca_classify(Xtr, Ytr, Xte)
    [~, nC, nT] = size(Xtr);
    classes = unique(Ytr); nCls = length(classes);
    W = zeros(nC, nCls); templates = zeros(nCls, nT);
    
    for ci = 1:nCls
        idx = find(Ytr==classes(ci)); ni = length(idx);
        X_ci = Xtr(idx,:,:);
        S = zeros(nC,nC);
        for j1=1:ni, for j2=(j1+1):ni
            Xj1=squeeze(X_ci(j1,:,:)); Xj2=squeeze(X_ci(j2,:,:));
            S=S+Xj1*Xj2'+Xj2*Xj1';
        end, end
        Q = zeros(nC,nC);
        for j=1:ni, Xj=squeeze(X_ci(j,:,:)); Q=Q+Xj*Xj'; end
        Q=Q+eye(nC)*1e-6;
        [V,D]=eig(S,Q); [~,mi]=max(real(diag(D)));
        W(:,ci)=real(V(:,mi));
        templates(ci,:) = W(:,ci)' * squeeze(mean(X_ci,1));
    end
    
    nTe = size(Xte,1); Y_pred = zeros(nTe,1);
    for ti = 1:nTe
        Xi = squeeze(Xte(ti,:,:));
        corrs = zeros(1,nCls);
        for ci=1:nCls, corrs(ci) = corr((W(:,ci)'*Xi)', templates(ci,:)'); end
        [~,Y_pred(ti)] = max(corrs);
        Y_pred(ti) = classes(Y_pred(ti));
    end
end

function Y_pred = etrca_classify(Xtr, Ytr, Xte)
    [~, nC, nT] = size(Xtr);
    classes = unique(Ytr); nCls = length(classes);
    W = zeros(nC, nCls); avg_all = zeros(nCls, nC, nT);
    
    for ci = 1:nCls
        idx = find(Ytr==classes(ci)); ni = length(idx);
        X_ci = Xtr(idx,:,:);
        S = zeros(nC,nC);
        for j1=1:ni, for j2=(j1+1):ni
            Xj1=squeeze(X_ci(j1,:,:)); Xj2=squeeze(X_ci(j2,:,:));
            S=S+Xj1*Xj2'+Xj2*Xj1';
        end, end
        Q = zeros(nC,nC);
        for j=1:ni, Xj=squeeze(X_ci(j,:,:)); Q=Q+Xj*Xj'; end
        Q=Q+eye(nC)*1e-6;
        [V,D]=eig(S,Q); [~,mi]=max(real(diag(D)));
        W(:,ci)=real(V(:,mi));
        avg_all(ci,:,:) = mean(X_ci,1);
    end
    
    nTe = size(Xte,1); Y_pred = zeros(nTe,1);
    for ti = 1:nTe
        Xi = squeeze(Xte(ti,:,:));
        scores = zeros(1,nCls);
        for ci=1:nCls
            tmpl = squeeze(avg_all(ci,:,:));
            for fi=1:nCls
                scores(ci) = scores(ci) + corr((W(:,fi)'*Xi)', (W(:,fi)'*tmpl)');
            end
        end
        [~,Y_pred(ti)] = max(scores);
        Y_pred(ti) = classes(Y_pred(ti));
    end
end
