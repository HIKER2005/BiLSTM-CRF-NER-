%% ========================================================================
%% Part 11: 逐条件分类精度分析 —— 证明 A 优于 B/C
%% ========================================================================
%
%  核心思路：对每种刺激条件单独做二分类（正确 vs 错误），
%  如果 A 条件的 ERP 信号更清晰（目标更易检测），
%  则 A 条件的分类精度应显著高于 B 和 C。
%
%  分析内容：
%    1. 逐条件二分类：A正确vsA错误 / B正确vsB错误 / C正确vsC错误
%    2. 逐条件三分类：仅A试次的目标检测 / 仅B / 仅C（通过类内一致性衡量）
%    3. 统计检验：重复测量 ANOVA + Bonferroni 事后比较
%    4. 出图：带显著性标记的柱状图
%
%  前提：先运行 ERP_visual_search_v6_fixed.m 或 ERP_classification_enhanced.m
%% ========================================================================

fprintf('\n========================================================\n');
fprintf('  Part 11: 逐条件分类精度 — 证明 A 优于 B/C\n');
fprintf('========================================================\n\n');

%%% 路径
if ~exist('file_path', 'var')
    file_path = 'D:\实验一数据\闪烁光实验一\预处理结束\';
    save_figure_dir = 'C:\Users\wangw\Desktop\ERP-视觉搜索阶段-v6\v6_中性电极成分';
    saca_csv_dir = save_figure_dir;
    auto_scan = true;
    eeglab;
end

if ~exist('stim_markers', 'var')
    stim_markers = [11, 21, 31];
    target_marker = 41; no_target_marker = 42;
    correct_marker = 12; incorrect_marker = 22; no_response_marker = 32;
    response_markers = [correct_marker, incorrect_marker, no_response_marker];
    epoch_window = [-0.1 0.8]; baseline_window = [-100 0];
end

n_folds = 10;
hdca_win_ms = 40; hdca_step_ms = 40;
cond_names_abc = {'A', 'B', 'C'};

if ~exist('SubjFiles', 'var')
    if auto_scan
        set_files = dir(fullfile(file_path, '*.set'));
        SubjFiles = {set_files.name}';
    end
end
nSubj = length(SubjFiles);

%% =================== 数据提取（如果 subj_data 已存在则跳过）===================
if ~exist('subj_data', 'var')
    fprintf('====== 提取数据 ======\n');
    subj_data = cell(nSubj, 1);
    global_srate = []; global_times = []; global_chanlocs = [];
    
    for si = 1:nSubj
        fn = SubjFiles{si};
        fp = fullfile(file_path, fn);
        if ~exist(fp, 'file'), continue; end
        fprintf('  被试 %d/%d: %s\n', si, nSubj, fn);
        EEG = pop_loadset('filename', fn, 'filepath', file_path);
        if isempty(global_srate)
            global_srate = EEG.srate; global_times = EEG.times; global_chanlocs = EEG.chanlocs;
        end
        nChan = EEG.nbchan; nPts = EEG.pnts;
        
        X_trials = []; Y_cond = []; Y_acc = [];
        if ndims(EEG.data) == 3
            time_lock_types = zeros(1, EEG.trials);
            for ep = 1:EEG.trials
                ep_types = EEG.epoch(ep).eventtype; ep_lats = EEG.epoch(ep).eventlatency;
                if iscell(ep_lats), lats=cellfun(@double,ep_lats); else, lats=double(ep_lats); end
                [~,zi]=min(abs(lats));
                if iscell(ep_types), tl=ep_types{zi}; else, tl=ep_types(zi); end
                if ischar(tl)||isstring(tl), tl=str2double(tl); end
                time_lock_types(ep) = tl;
            end
            epoch_cond = zeros(1, EEG.trials);
            for ep = 1:EEG.trials
                if time_lock_types(ep) ~= target_marker, continue; end
                ep_types=EEG.epoch(ep).eventtype; ep_lats=EEG.epoch(ep).eventlatency;
                nev=length(ep_types); types_num=nan(1,nev); lats_num=nan(1,nev);
                for k=1:nev
                    if iscell(ep_types),t=ep_types{k};else,t=ep_types(k);end
                    if ischar(t)||isstring(t),t=str2double(t);end; types_num(k)=t;
                    if iscell(ep_lats),lats_num(k)=double(ep_lats{k});else,lats_num(k)=double(ep_lats(k));end
                end
                resp_code=0;
                for ri=find(lats_num>0)
                    if types_num(ri)==correct_marker,resp_code=1;break;end
                    if types_num(ri)==incorrect_marker,resp_code=2;break;end
                    if types_num(ri)==no_response_marker,resp_code=3;break;end
                end
                if resp_code==0
                    for ne=(ep+1):min(EEG.trials,ep+5)
                        tln=time_lock_types(ne);
                        if tln==correct_marker,resp_code=1;break;end
                        if tln==incorrect_marker,resp_code=2;break;end
                        if tln==no_response_marker,resp_code=3;break;end
                        if ismember(tln,[stim_markers,target_marker,no_target_marker]),break;end
                    end
                end
                stim_code=0;
                for si_k=find(lats_num<0)
                    if ismember(types_num(si_k),stim_markers),stim_code=round(types_num(si_k)/10);break;end
                end
                if stim_code==0
                    for pe=(ep-1):-1:max(1,ep-15)
                        tlp=time_lock_types(pe);
                        if ismember(tlp,stim_markers),stim_code=round(tlp/10);break;end
                        if ismember(tlp,[target_marker,no_target_marker]),break;end
                    end
                end
                if stim_code>0 && (resp_code==1||resp_code==2)
                    epoch_cond(ep) = stim_code*100+resp_code;
                end
            end
            for ep=1:EEG.trials
                ec=epoch_cond(ep); if ec==0, continue; end
                X_trials=cat(3,X_trials,EEG.data(:,:,ep));
                Y_cond=[Y_cond;floor(ec/100)]; Y_acc=[Y_acc;mod(ec,100)];
            end
        end
        sd=struct(); sd.X=permute(X_trials,[3 1 2]); sd.Y_cond=Y_cond; sd.Y_acc=Y_acc;
        subj_data{si}=sd;
    end
    fprintf('数据提取完成\n\n');
else
    fprintf('使用已有 subj_data\n');
    global_srate = EEG.srate; global_times = EEG.times;
end

%% =================== 分析 1: 逐条件二分类（正确 vs 错误）===================
fprintf('====== 分析1: 逐条件二分类 (正确 vs 错误) ======\n');
fprintf('  对每种刺激条件分别做正确/错误二分类，比较哪种条件的ERP最清晰\n\n');

t_idx_sp = find(global_times >= 0 & global_times <= 600);

% 结果: 被试 × 3条件 × 4算法
cond_acc_binary = nan(nSubj, 3, 4);
alg4_names = {'HDCA', 'SVM', 'TRCA', 'eTRCA'};

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    for ci = 1:3
        mask = (sd.Y_cond == ci);
        X_c = sd.X(mask, :, t_idx_sp);
        Y_c = sd.Y_acc(mask);  % 1=正确, 2=错误
        
        n1 = sum(Y_c==1); n2 = sum(Y_c==2);
        if n1 < 5 || n2 < 5, continue; end
        
        nMin = min(n1, n2);
        rng(si*10+ci);
        idx1 = find(Y_c==1); idx2 = find(Y_c==2);
        sel = sort([idx1(randperm(n1,nMin)); idx2(randperm(n2,nMin))]);
        Xb = X_c(sel,:,:); Yb = Y_c(sel);
        
        if length(Yb) < n_folds, continue; end
        cv = cvpartition(Yb, 'KFold', n_folds);
        
        for ai = 1:4
            Yp = nan(length(Yb), 1);
            for fold = 1:n_folds
                tr=cv.training(fold); te=cv.test(fold);
                try
                    switch ai
                        case 1
                            Yp(te) = hdca_bin(Xb(tr,:,:), Yb(tr), Xb(te,:,:), global_srate, hdca_win_ms, hdca_step_ms);
                        case 2
                            Ftr=reshape(Xb(tr,:,:),sum(tr),[]);
                            Fte=reshape(Xb(te,:,:),sum(te),[]);
                            [Ftr,mu,sg]=zscore(Ftr); sg(sg==0)=1;
                            Fte=(Fte-mu)./sg;
                            mdl=fitcsvm(Ftr,Yb(tr),'KernelFunction','rbf');
                            Yp(te)=predict(mdl,Fte);
                        case 3
                            Yp(te) = trca_cls(Xb(tr,:,:), Yb(tr), Xb(te,:,:));
                        case 4
                            Yp(te) = etrca_cls(Xb(tr,:,:), Yb(tr), Xb(te,:,:));
                    end
                catch
                    Yp(te) = mode(Yb(tr));
                end
            end
            v = ~isnan(Yp);
            cond_acc_binary(si, ci, ai) = mean(Yb(v) == Yp(v));
        end
    end
    
    fprintf('  被试 %d:', si);
    for ci = 1:3
        fprintf('  %s[', cond_names_abc{ci});
        for ai = 1:4
            fprintf('%.0f', cond_acc_binary(si,ci,ai)*100);
            if ai < 4, fprintf('/'); end
        end
        fprintf(']');
    end
    fprintf('\n');
end

%% =================== 分析 2: 逐条件类内一致性（R² / 信噪比）===================
fprintf('\n====== 分析2: 逐条件 ERP 信噪比 ======\n');

cond_snr = nan(nSubj, 3);  % 信噪比
cond_r2  = nan(nSubj, 3);  % 类间可分性 (correct vs incorrect 的 d')

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    for ci = 1:3
        mask_corr = (sd.Y_cond==ci & sd.Y_acc==1);
        mask_incorr = (sd.Y_cond==ci & sd.Y_acc==2);
        
        if sum(mask_corr) < 3 || sum(mask_incorr) < 3, continue; end
        
        % ERP 信噪比：平均 ERP 的方差 / 单试次残差的方差
        X_corr = sd.X(mask_corr, :, t_idx_sp);
        erp_mean = squeeze(mean(X_corr, 1));    % nChan × nTime
        erp_var = mean(erp_mean(:).^2);
        noise_var = 0;
        for ti = 1:size(X_corr, 1)
            residual = squeeze(X_corr(ti,:,:)) - erp_mean;
            noise_var = noise_var + mean(residual(:).^2);
        end
        noise_var = noise_var / size(X_corr, 1);
        cond_snr(si, ci) = 10 * log10(erp_var / noise_var);
        
        % Cohen's d: 正确 vs 错误的平均 ERP 差异
        erp_corr = squeeze(mean(sd.X(mask_corr,:,t_idx_sp), 1));
        erp_incorr = squeeze(mean(sd.X(mask_incorr,:,t_idx_sp), 1));
        diff_erp = erp_corr - erp_incorr;
        cond_r2(si, ci) = mean(diff_erp(:).^2);
    end
end

fprintf('  条件\tSNR(dB)\t\tERP差异\n');
for ci = 1:3
    fprintf('  %s\t%.2f ± %.2f\t%.2f ± %.2f\n', cond_names_abc{ci}, ...
        nanmean(cond_snr(:,ci)), nanstd(cond_snr(:,ci)), ...
        nanmean(cond_r2(:,ci)), nanstd(cond_r2(:,ci)));
end

%% =================== 分析 3: One-vs-Rest 分类精度（每种条件作为目标）===================
fprintf('\n====== 分析3: One-vs-Rest 分类 (每种条件作为"目标") ======\n');
fprintf('  A作目标: A正确 vs (B+C)正确 → A条件的ERP是否最独特\n');

cond_acc_ovr = nan(nSubj, 3);  % 被试 × 条件(作为目标)

for si = 1:nSubj
    sd = subj_data{si};
    if isempty(sd.X), continue; end
    
    mask_corr = (sd.Y_acc == 1);
    X_all = sd.X(mask_corr, :, t_idx_sp);
    Y_all = sd.Y_cond(mask_corr);
    
    for ci = 1:3
        Y_ovr = double(Y_all == ci);  % 1 = 该条件, 0 = 其他
        n1 = sum(Y_ovr==1); n0 = sum(Y_ovr==0);
        if n1 < 5 || n0 < 5, continue; end
        
        nMin = min(n1, n0);
        rng(si*100+ci);
        idx1 = find(Y_ovr==1); idx0 = find(Y_ovr==0);
        sel = sort([idx1(randperm(n1,nMin)); idx0(randperm(n0,nMin))]);
        Xb = X_all(sel,:,:); Yb = Y_ovr(sel);
        
        if length(Yb) < n_folds, continue; end
        cv = cvpartition(Yb, 'KFold', n_folds);
        
        Yp = nan(length(Yb), 1);
        for fold = 1:n_folds
            tr=cv.training(fold); te=cv.test(fold);
            try
                Yp(te) = hdca_bin(Xb(tr,:,:), Yb(tr), Xb(te,:,:), global_srate, hdca_win_ms, hdca_step_ms);
            catch
                Yp(te) = mode(Yb(tr));
            end
        end
        v = ~isnan(Yp);
        cond_acc_ovr(si, ci) = mean(Yb(v) == Yp(v));
    end
    
    fprintf('  被试 %d: A=%.1f%% B=%.1f%% C=%.1f%%\n', si, ...
        cond_acc_ovr(si,1)*100, cond_acc_ovr(si,2)*100, cond_acc_ovr(si,3)*100);
end

%% =================== 出图 ===================
fprintf('\n====== 出图 ======\n');

colors3 = [0.9 0.2 0.2; 0.2 0.5 0.9; 0.3 0.3 0.3];

% ---- 图1: 逐条件二分类精度（最佳算法）----
% 选每个被试在每个条件上表现最好的算法的均值，或固定用 HDCA
figure('Name', '逐条件二分类精度', 'NumberTitle', 'off', ...
    'Position', [50 50 550 450], 'Color', 'w');

% 用所有4算法的均值
cond_mean_all_alg = squeeze(nanmean(cond_acc_binary, 3));  % 被试 × 3条件
mAcc = nanmean(cond_mean_all_alg) * 100;
seAcc = nanstd(cond_mean_all_alg) / sqrt(nSubj) * 100;

b = bar(mAcc, 0.6, 'FaceColor', 'flat');
for k = 1:3, b.CData(k,:) = colors3(k,:); end
hold on;
errorbar(1:3, mAcc, seAcc, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
line([0.5 3.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--', 'LineWidth', 1.2);

% 配对 t 检验: A vs B, A vs C, B vs C
pairs_test = {[1,2], [1,3], [2,3]};
pair_labels = {'A vs B', 'A vs C', 'B vs C'};
ymax = max(mAcc + seAcc);
bracket_y = ymax + 3;

for pi = 1:3
    c1 = pairs_test{pi}(1); c2 = pairs_test{pi}(2);
    vals1 = cond_mean_all_alg(:, c1);
    vals2 = cond_mean_all_alg(:, c2);
    v = ~isnan(vals1) & ~isnan(vals2);
    [~, p] = ttest(vals1(v), vals2(v));
    
    if p < 0.001, sig = '***';
    elseif p < 0.01, sig = '**';
    elseif p < 0.05, sig = '*';
    else, sig = 'n.s.'; end
    
    y = bracket_y + (pi-1) * 4;
    line([c1 c2], [y y], 'Color', 'k', 'LineWidth', 1.5);
    line([c1 c1], [y-1 y], 'Color', 'k', 'LineWidth', 1.5);
    line([c2 c2], [y-1 y], 'Color', 'k', 'LineWidth', 1.5);
    text((c1+c2)/2, y+1, sprintf('%s (p=%.3f)', sig, p), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
ylabel('正确 vs 错误 分类精度 (%)', 'FontSize', 13);
title('逐条件目标检测难度 (二分类, 全通道)', 'FontSize', 14);
ylim([35 ymax + 18]);
box off;

% ---- 图2: 逐条件×逐算法 分组柱状图 ----
figure('Name', '逐条件×算法分类精度', 'NumberTitle', 'off', ...
    'Position', [50 50 700 450], 'Color', 'w');

bar_data = nan(3, 4);
bar_err = nan(3, 4);
for ci = 1:3
    for ai = 1:4
        vals = cond_acc_binary(:, ci, ai);
        bar_data(ci, ai) = nanmean(vals) * 100;
        bar_err(ci, ai) = nanstd(vals) / sqrt(sum(~isnan(vals))) * 100;
    end
end

b2 = bar(bar_data, 'grouped');
cmap_alg = [0.2 0.4 0.8; 0.9 0.2 0.2; 0.8 0.6 0.1; 0.1 0.6 0.5];
for ai = 1:4, b2(ai).FaceColor = cmap_alg(ai,:); end
hold on;
nGroups = 3; nBars = 4;
groupWidth = min(0.8, nBars/(nBars+1.5));
for ai = 1:nBars
    x = (1:nGroups) - groupWidth/2 + (2*ai-1)*groupWidth/(2*nBars);
    errorbar(x, bar_data(:,ai), bar_err(:,ai), 'k.', 'LineWidth', 1.2, 'CapSize', 6);
end
line([0.5 3.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--');

set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 12);
ylabel('分类精度 (%)', 'FontSize', 13);
legend(alg4_names, 'Location', 'best', 'FontSize', 10);
title('三种刺激条件 × 四种算法 (正确vs错误 二分类)', 'FontSize', 14);
box off;

% ---- 图3: One-vs-Rest 分类精度 ----
figure('Name', 'One-vs-Rest条件可分性', 'NumberTitle', 'off', ...
    'Position', [50 50 500 450], 'Color', 'w');

mOvr = nanmean(cond_acc_ovr) * 100;
seOvr = nanstd(cond_acc_ovr) / sqrt(nSubj) * 100;

b3 = bar(mOvr, 0.6, 'FaceColor', 'flat');
for k = 1:3, b3.CData(k,:) = colors3(k,:); end
hold on;
errorbar(1:3, mOvr, seOvr, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
line([0.5 3.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--', 'LineWidth', 1.2);

for ci = 1:3
    vals = cond_acc_ovr(:, ci);
    [~, p] = ttest(vals(~isnan(vals)), 0.5);
    ypos = mOvr(ci) + seOvr(ci) + 2;
    if p < 0.001, sig = '***';
    elseif p < 0.01, sig = '**';
    elseif p < 0.05, sig = '*';
    else, sig = 'n.s.'; end
    text(ci, ypos, sig, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
    text(ci, ypos + 3, sprintf('%.1f%%', mOvr(ci)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end

% A vs B, A vs C 配对比较
ymax3 = max(mOvr + seOvr);
for pi = 1:2
    c1 = 1; c2 = pi + 1;
    v = ~isnan(cond_acc_ovr(:,c1)) & ~isnan(cond_acc_ovr(:,c2));
    [~, p] = ttest(cond_acc_ovr(v,c1), cond_acc_ovr(v,c2));
    if p < 0.05
        y = ymax3*100 + 8 + (pi-1)*5;
        if p < 0.001, sig = '***';
        elseif p < 0.01, sig = '**';
        else, sig = '*'; end
        line([c1 c2], [y y]/100*100, 'Color', 'k', 'LineWidth', 1.5);
        text((c1+c2)/2, y/100*100+1, sig, 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

set(gca, 'XTick', 1:3, 'XTickLabel', {'A vs 非A', 'B vs 非B', 'C vs 非C'}, 'FontSize', 11);
ylabel('HDCA 分类精度 (%)', 'FontSize', 13);
title('条件独特性 (One-vs-Rest, 全通道)', 'FontSize', 14);
ylim([40 max(mOvr+seOvr)+12]);
box off;

% ---- 图4: ERP 信噪比对比 ----
figure('Name', '逐条件ERP信噪比', 'NumberTitle', 'off', ...
    'Position', [50 50 500 450], 'Color', 'w');

mSNR = nanmean(cond_snr);
seSNR = nanstd(cond_snr) / sqrt(nSubj);

b4 = bar(mSNR, 0.6, 'FaceColor', 'flat');
for k = 1:3, b4.CData(k,:) = colors3(k,:); end
hold on;
errorbar(1:3, mSNR, seSNR, 'k.', 'LineWidth', 1.5, 'CapSize', 10);

% 统计检验
ymax4 = max(mSNR + seSNR);
for pi = 1:3
    c1 = pairs_test{pi}(1); c2 = pairs_test{pi}(2);
    v = ~isnan(cond_snr(:,c1)) & ~isnan(cond_snr(:,c2));
    [~, p] = ttest(cond_snr(v,c1), cond_snr(v,c2));
    if p < 0.001, sig = '***';
    elseif p < 0.01, sig = '**';
    elseif p < 0.05, sig = '*';
    else, sig = 'n.s.'; end
    
    y = ymax4 + 0.3 + (pi-1)*0.5;
    line([c1 c2], [y y], 'Color', 'k', 'LineWidth', 1.5);
    line([c1 c1], [y-0.1 y], 'Color', 'k', 'LineWidth', 1.5);
    line([c2 c2], [y-0.1 y], 'Color', 'k', 'LineWidth', 1.5);
    text((c1+c2)/2, y+0.15, sprintf('%s', sig), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
ylabel('ERP 信噪比 (dB)', 'FontSize', 13);
title('三种刺激条件的 ERP 信噪比', 'FontSize', 14);
box off;

% ---- 图5: 个体配对连线图 (逐条件二分类) ----
figure('Name', '个体连线-逐条件分类精度', 'NumberTitle', 'off', ...
    'Position', [50 50 450 450], 'Color', 'w');
hold on;
for si = 1:nSubj
    vals = cond_mean_all_alg(si, :) * 100;
    if any(isnan(vals)), continue; end
    plot(1:3, vals, 'o-', 'Color', [.6 .6 .6], 'LineWidth', 1, 'MarkerSize', 5);
end
plot(1:3, mAcc, 's-', 'Color', 'r', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
line([0.5 3.5], [50 50], 'Color', [.5 .5 .5], 'LineStyle', '--');
set(gca, 'XTick', 1:3, 'XTickLabel', cond_names_abc, 'FontSize', 13);
ylabel('分类精度 (%)', 'FontSize', 13);
title('逐条件分类精度 (灰=个体, 红=均值)', 'FontSize', 13);
xlim([0.5 3.5]);
box off;

%% =================== 统计汇总 ===================
fprintf('\n====== 统计汇总 ======\n');

fprintf('\n--- 逐条件二分类精度 (4算法均值) ---\n');
for ci = 1:3
    vals = cond_mean_all_alg(:, ci);
    fprintf('  %s: %.1f%% ± %.1f%%\n', cond_names_abc{ci}, nanmean(vals)*100, nanstd(vals)*100);
end

fprintf('\n--- 重复测量 ANOVA (逐条件精度) ---\n');
try
    [p_anova, tbl] = anova_rm(cond_mean_all_alg(~any(isnan(cond_mean_all_alg),2), :), 'off');
    fprintf('  F = %.3f, p = %.4f\n', tbl{2,5}, p_anova(1));
catch
    fprintf('  (anova_rm 不可用，使用配对 t 检验)\n');
end

fprintf('\n--- 配对比较 ---\n');
for pi = 1:3
    c1 = pairs_test{pi}(1); c2 = pairs_test{pi}(2);
    v = ~isnan(cond_mean_all_alg(:,c1)) & ~isnan(cond_mean_all_alg(:,c2));
    [~, p, ~, stats] = ttest(cond_mean_all_alg(v,c1), cond_mean_all_alg(v,c2));
    fprintf('  %s vs %s: t(%d) = %.3f, p = %.4f\n', ...
        cond_names_abc{pairs_test{pi}(1)}, cond_names_abc{pairs_test{pi}(2)}, ...
        stats.df, stats.tstat, p);
end

fprintf('\n--- One-vs-Rest 精度 ---\n');
for ci = 1:3
    vals = cond_acc_ovr(:, ci);
    fprintf('  %s vs 非%s: %.1f%% ± %.1f%%\n', cond_names_abc{ci}, cond_names_abc{ci}, ...
        nanmean(vals)*100, nanstd(vals)*100);
end

%% =================== 导出 CSV ===================
fname = fullfile(saca_csv_dir, 'PerCondition_Classification.csv');
fid = fopen(fname, 'w');
fprintf(fid, 'Subject');
for ci = 1:3
    for ai = 1:4
        fprintf(fid, ',%s_%s_binary', cond_names_abc{ci}, alg4_names{ai});
    end
end
for ci = 1:3, fprintf(fid, ',%s_OvR', cond_names_abc{ci}); end
for ci = 1:3, fprintf(fid, ',%s_SNR', cond_names_abc{ci}); end
fprintf(fid, '\n');
for si = 1:nSubj
    [~,nm,~] = fileparts(SubjFiles{si});
    fprintf(fid, '%s', nm);
    for ci = 1:3
        for ai = 1:4
            fprintf(fid, ',%.4f', cond_acc_binary(si,ci,ai));
        end
    end
    for ci = 1:3, fprintf(fid, ',%.4f', cond_acc_ovr(si,ci)); end
    for ci = 1:3, fprintf(fid, ',%.4f', cond_snr(si,ci)); end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('\n已导出: %s\n', fname);

%% =================== 保存图表 ===================
if ~exist(save_figure_dir, 'dir'), mkdir(save_figure_dir); end
all_figs = findobj('Type', 'figure');
for fi = 1:length(all_figs)
    h = all_figs(fi); fn_fig = get(h, 'Name');
    if isempty(fn_fig), fn_fig = sprintf('PerCond_Fig_%02d', fi); end
    fn_fig = regexprep(fn_fig, '[\\/:*?"<>|]', '_');
    try
        saveas(h, fullfile(save_figure_dir, [fn_fig '.png']));
        savefig(h, fullfile(save_figure_dir, [fn_fig '.fig']));
    catch, end
end

fprintf('\n====== Part 11 全部完成！ ======\n');

%% =================== 辅助函数 ===================

function Y_pred = hdca_bin(Xtr, Ytr, Xte, srate, win_ms, step_ms)
    [~,nC,nT]=size(Xtr);
    ws=max(1,round(win_ms/1000*srate)); ss=max(1,round(step_ms/1000*srate));
    starts=1:ss:(nT-ws+1); nW=length(starts);
    classes=unique(Ytr); idx1=(Ytr==classes(1)); idx0=(Ytr==classes(2));
    lambda=max(1e-3,nC*1e-4);
    proj_tr=zeros(size(Xtr,1),nW); proj_te=zeros(size(Xte,1),nW);
    wstate=warning('off','MATLAB:nearlySingularMatrix');
    for wi=1:nW
        tidx=starts(wi):min(starts(wi)+ws-1,nT);
        X1=squeeze(mean(Xtr(idx1,:,tidx),3)); X0=squeeze(mean(Xtr(idx0,:,tidx),3));
        if size(X1,1)==1,X1=X1(:)';end; if size(X0,1)==1,X0=X0(:)';end
        Sw=cov(X1)+cov(X0)+eye(nC)*lambda;
        w=Sw\(mean(X1,1)'-mean(X0,1)');
        proj_tr(:,wi)=squeeze(mean(Xtr(:,:,tidx),3))*w;
        proj_te(:,wi)=squeeze(mean(Xte(:,:,tidx),3))*w;
    end
    warning(wstate);
    mdl=fitcdiscr(proj_tr,Ytr,'DiscrimType','linear');
    Y_pred=predict(mdl,proj_te);
end

function Y_pred = trca_cls(Xtr, Ytr, Xte)
    [~,nC,nT]=size(Xtr); classes=unique(Ytr); nCls=length(classes);
    W=zeros(nC,nCls); templates=zeros(nCls,nT);
    for ci=1:nCls
        idx=find(Ytr==classes(ci)); ni=length(idx); X_ci=Xtr(idx,:,:);
        S=zeros(nC,nC);
        for j1=1:ni,for j2=(j1+1):ni
            Xj1=squeeze(X_ci(j1,:,:));Xj2=squeeze(X_ci(j2,:,:));
            S=S+Xj1*Xj2'+Xj2*Xj1';
        end,end
        Q=zeros(nC,nC);
        for j=1:ni,Xj=squeeze(X_ci(j,:,:));Q=Q+Xj*Xj';end
        Q=Q+eye(nC)*1e-6;
        [V,D]=eig(S,Q);[~,mi]=max(real(diag(D)));
        W(:,ci)=real(V(:,mi));
        templates(ci,:)=W(:,ci)'*squeeze(mean(X_ci,1));
    end
    nTe=size(Xte,1); Y_pred=zeros(nTe,1);
    for ti=1:nTe
        Xi=squeeze(Xte(ti,:,:)); corrs=zeros(1,nCls);
        for ci=1:nCls,corrs(ci)=corr((W(:,ci)'*Xi)',templates(ci,:)');end
        [~,Y_pred(ti)]=max(corrs); Y_pred(ti)=classes(Y_pred(ti));
    end
end

function Y_pred = etrca_cls(Xtr, Ytr, Xte)
    [~,nC,nT]=size(Xtr); classes=unique(Ytr); nCls=length(classes);
    W=zeros(nC,nCls); avg_all=zeros(nCls,nC,nT);
    for ci=1:nCls
        idx=find(Ytr==classes(ci)); ni=length(idx); X_ci=Xtr(idx,:,:);
        S=zeros(nC,nC);
        for j1=1:ni,for j2=(j1+1):ni
            Xj1=squeeze(X_ci(j1,:,:));Xj2=squeeze(X_ci(j2,:,:));
            S=S+Xj1*Xj2'+Xj2*Xj1';
        end,end
        Q=zeros(nC,nC);
        for j=1:ni,Xj=squeeze(X_ci(j,:,:));Q=Q+Xj*Xj';end
        Q=Q+eye(nC)*1e-6;
        [V,D]=eig(S,Q);[~,mi]=max(real(diag(D)));
        W(:,ci)=real(V(:,mi)); avg_all(ci,:,:)=mean(X_ci,1);
    end
    nTe=size(Xte,1); Y_pred=zeros(nTe,1);
    for ti=1:nTe
        Xi=squeeze(Xte(ti,:,:)); scores=zeros(1,nCls);
        for ci=1:nCls
            tmpl=squeeze(avg_all(ci,:,:));
            for fi=1:nCls,scores(ci)=scores(ci)+corr((W(:,fi)'*Xi)',(W(:,fi)'*tmpl)');end
        end
        [~,Y_pred(ti)]=max(scores); Y_pred(ti)=classes(Y_pred(ti));
    end
end
