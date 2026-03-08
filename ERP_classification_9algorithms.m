%% ========================================================================
%% Part 9: ERP 成分分类分析 —— 9 种算法（参照论文 RSVP 分类方法）
%% ========================================================================
%
%  参照论文方法，采用以下 9 种算法对三种刺激条件（A/B/C）进行分类：
%    1. HDCA  — 层次判别成分分析（滑动窗口空间权重 + 时间LDA）
%    2. SVM   — 支持向量机（ERP 均值/峰值特征）
%    3. Wavelet+SVM — 小波时频特征提取 + SVM
%    4. xDAWN+SVM   — xDAWN 空间滤波增强 ERP + SVM
%    5. DCPM+SVM    — 判别典型模式匹配 + SVM
%    6. ENT+SVM     — 熵特征（样本熵/谱熵）+ SVM
%    7. TRCA  — 任务相关成分分析
%    8. eTRCA — 集成 TRCA
%    9. EEGNet — 紧凑型卷积神经网络
%
%  分类特征电极与时间窗：
%    N2 成分 → PO7、PO8 电极，200-250 ms
%    P3 成分 → Fz、Cz 电极，300-350 ms
%    空间算法（HDCA/xDAWN/TRCA/eTRCA/EEGNet）使用以上 4 个电极的完整时段
%
%  分类任务：A vs B vs C（三分类，仅正确反应试次）
%  交叉验证：被试内 10 折 CV + 跨被试 LOSO-CV
%
%  依赖：EEGLAB, Statistics and Machine Learning Toolbox,
%         Signal Processing Toolbox (小波), Deep Learning Toolbox (EEGNet)
%% ========================================================================

%% ========================= 9.0 参数设置 =========================
fprintf('\n========================================================\n');
fprintf('  Part 9: ERP 成分分类 — 9 种算法（参照论文方法）\n');
fprintf('========================================================\n\n');

%%% 路径（独立运行时需要设置）
if ~exist('file_path', 'var')
    file_path = 'D:\实验一数据\闪烁光实验一\预处理结束\';
    save_figure_dir = 'C:\Users\wangw\Desktop\ERP-视觉搜索阶段-v6\v6_中性电极成分';
    saca_csv_dir = 'C:\Users\wangw\Desktop\ERP-视觉搜索阶段-v6\v6_中性电极成分';
    auto_scan = true;
    eeglab;
end

%%% 分类电极
clf_N2_chans = {'PO7', 'PO8'};          % N2 成分电极
clf_P3_chans = {'Fz', 'Cz'};            % P3 成分电极
clf_N2_window = [200, 250];              % N2 时间窗 (ms)
clf_P3_window = [300, 350];              % P3 时间窗 (ms)

%%% 空间算法使用的电极和时间窗（HDCA/xDAWN/TRCA/eTRCA/EEGNet）
clf_spatial_chans = [clf_N2_chans, clf_P3_chans];  % PO7, PO8, Fz, Cz
clf_spatial_window = [0, 600];           %%% 空间算法使用的时间窗 (ms)

%%% HDCA 参数（参照论文：窗口长度 40ms，步长 40ms，10 折 CV）
hdca_win_ms  = 40;                       %%% 每个窗口长度(ms)，论文中为 10 采样点@250Hz=40ms
hdca_step_ms = 40;                       %%% 窗口步长(ms)

%%% EEGNet 参数
eegnet_epochs  = 200;                    %%% 训练轮数
eegnet_F1 = 8;                           %%% 时间卷积核数
eegnet_D  = 2;                           %%% 深度卷积倍数
eegnet_F2 = 16;                          %%% 可分离卷积核数
eegnet_dropout = 0.25;

%%% 交叉验证
n_folds = 10;                            %%% K 折交叉验证
do_loso = true;                          %%% 是否做 LOSO-CV

%%% 分类条件（仅正确反应）
clf_markers = [101, 201, 301];
clf_labels  = [1, 2, 3];
clf_names   = {'A', 'B', 'C'};
nClasses = 3;

%%% 事件标记（复合打标逻辑）
if ~exist('stim_markers', 'var')
    stim_markers = [11, 21, 31];
    target_marker = 41; no_target_marker = 42;
    correct_marker = 12; incorrect_marker = 22; no_response_marker = 32;
    response_markers = [correct_marker, incorrect_marker, no_response_marker];
end
if ~exist('epoch_window', 'var')
    epoch_window = [-0.1 0.8];
    baseline_window = [-100 0];
end

%%% 9 种算法名称
alg_names = {'HDCA','SVM','Wavelet+SVM','xDAWN+SVM','DCPM+SVM',...
             'ENT+SVM','TRCA','eTRCA','EEGNet'};
nAlg = length(alg_names);

%% ========================= 9.1 被试文件扫描 =========================
if ~exist('SubjFiles', 'var') || ~exist('nSubj', 'var')
    if auto_scan
        set_files = dir(fullfile(file_path, '*.set'));
        SubjFiles = {set_files.name}';
    end
    nSubj = length(SubjFiles);
end
fprintf('共 %d 个被试\n', nSubj);

%% ========================= 9.2 单试次数据提取 =========================
fprintf('\n====== 提取单试次数据 ======\n');
fprintf('N2: %s, %d-%dms | P3: %s, %d-%dms\n', ...
    strjoin(clf_N2_chans,'/'), clf_N2_window(1), clf_N2_window(2), ...
    strjoin(clf_P3_chans,'/'), clf_P3_window(1), clf_P3_window(2));
fprintf('空间算法电极: %s, %d-%dms\n', ...
    strjoin(clf_spatial_chans,'/'), clf_spatial_window(1), clf_spatial_window(2));

% 存储：单试次 3D 数据 + 标签 + 被试ID
all_X_3d = [];        % total_trials × n_spatial_chans × n_spatial_time
all_Y    = [];        % total_trials × 1
all_sid  = [];        % total_trials × 1
chan_idx_found = false;

for si = 1:nSubj
    file_name = SubjFiles{si};
    full_path = fullfile(file_path, file_name);
    if ~exist(full_path, 'file'), continue; end
    
    fprintf('--- 被试 %d/%d: %s ---\n', si, nSubj, file_name);
    EEG = pop_loadset('filename', file_name, 'filepath', file_path);
    
    % 首次查找电极索引
    if ~chan_idx_found
        ch_labels = {EEG.chanlocs.labels};
        
        N2_idx = zeros(1, length(clf_N2_chans));
        for ci = 1:length(clf_N2_chans)
            f = find(strcmpi(ch_labels, clf_N2_chans{ci}));
            if ~isempty(f), N2_idx(ci) = f(1); end
        end
        P3_idx = zeros(1, length(clf_P3_chans));
        for ci = 1:length(clf_P3_chans)
            f = find(strcmpi(ch_labels, clf_P3_chans{ci}));
            if ~isempty(f), P3_idx(ci) = f(1); end
        end
        
        spatial_idx = zeros(1, length(clf_spatial_chans));
        for ci = 1:length(clf_spatial_chans)
            f = find(strcmpi(ch_labels, clf_spatial_chans{ci}));
            if ~isempty(f), spatial_idx(ci) = f(1); end
        end
        
        % 移除未找到的
        N2_idx = N2_idx(N2_idx > 0);
        P3_idx = P3_idx(P3_idx > 0);
        spatial_idx = spatial_idx(spatial_idx > 0);
        
        fprintf('  N2电极索引: [%s]\n', num2str(N2_idx));
        fprintf('  P3电极索引: [%s]\n', num2str(P3_idx));
        fprintf('  空间电极索引: [%s]\n', num2str(spatial_idx));
        
        % 时间索引
        N2_tidx = find(EEG.times >= clf_N2_window(1) & EEG.times <= clf_N2_window(2));
        P3_tidx = find(EEG.times >= clf_P3_window(1) & EEG.times <= clf_P3_window(2));
        spatial_tidx = find(EEG.times >= clf_spatial_window(1) & EEG.times <= clf_spatial_window(2));
        
        fprintf('  N2 时间点: %d 个 | P3 时间点: %d 个 | 空间窗时间点: %d 个\n', ...
            length(N2_tidx), length(P3_tidx), length(spatial_tidx));
        
        nSpatialChan = length(spatial_idx);
        nSpatialTime = length(spatial_tidx);
        srate_clf = EEG.srate;
        times_spatial = EEG.times(spatial_tidx);
        times_full = EEG.times;
        
        chan_idx_found = true;
    end
    
    % ---- 复合打标逻辑（与 Part 1 完全一致）----
    is_epoched = (ndims(EEG.data) == 3);
    
    if is_epoched
        % 已分段数据
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
            
            if stim_code>0 && resp_code>0
                epoch_cond(ep) = stim_code*100 + resp_code;
            end
        end
        
        % 提取单试次 3D 数据
        for ep = 1:EEG.trials
            cidx = find(clf_markers == epoch_cond(ep));
            if ~isempty(cidx)
                trial_3d = EEG.data(spatial_idx, spatial_tidx, ep);  % nChan × nTime
                all_X_3d = cat(1, all_X_3d, reshape(trial_3d, [1, nSpatialChan, nSpatialTime]));
                all_Y = [all_Y; clf_labels(cidx)];
                all_sid = [all_sid; si];
            end
        end
        
    else
        % 连续数据：先打标再分段
        nevents = length(EEG.event);
        for ev = 1:nevents
            ct = EEG.event(ev).type;
            if ischar(ct)||isstring(ct), ct=str2double(ct); end
            if ct == target_marker
                st=NaN;
                for pe=(ev-1):-1:1
                    pt=EEG.event(pe).type; if ischar(pt)||isstring(pt), pt=str2double(pt); end
                    if ismember(pt,stim_markers), st=pt; break; end
                    if ismember(pt,[target_marker,no_target_marker]), break; end
                end
                rt=NaN;
                for ne=(ev+1):nevents
                    nt2=EEG.event(ne).type; if ischar(nt2)||isstring(nt2), nt2=str2double(nt2); end
                    if ismember(nt2,response_markers), rt=nt2; break; end
                    if ismember(nt2,[stim_markers,target_marker,no_target_marker]), break; end
                end
                if ~isnan(st)
                    sc=round(st/10);
                    if rt==correct_marker, rc=1; elseif rt==incorrect_marker, rc=2;
                    elseif rt==no_response_marker, rc=3; else, rc=0; end
                    if rc>0, EEG.event(ev).type=sc*100+rc; end
                end
            end
        end
        EEG = pop_epoch(EEG, num2cell(clf_markers), epoch_window, ...
            'newname', 'clf_epoched', 'epochinfo', 'yes');
        EEG = pop_rmbase(EEG, baseline_window);
        
        for ep = 1:EEG.trials
            ep_type=EEG.epoch(ep).eventtype;
            if iscell(ep_type)
                el=EEG.epoch(ep).eventlatency;
                if iscell(el), ls=cellfun(@double,el); else, ls=double(el); end
                [~,zi]=min(abs(ls)); evt=ep_type{zi};
            else, evt=ep_type; end
            if ischar(evt)||isstring(evt), evt=str2double(evt); end
            cidx=find(clf_markers==evt);
            if ~isempty(cidx)
                trial_3d = EEG.data(spatial_idx, spatial_tidx, ep);
                all_X_3d = cat(1, all_X_3d, reshape(trial_3d, [1, nSpatialChan, nSpatialTime]));
                all_Y = [all_Y; clf_labels(cidx)];
                all_sid = [all_sid; si];
            end
        end
    end
    
    nA=sum(all_sid==si & all_Y==1); nB=sum(all_sid==si & all_Y==2); nC=sum(all_sid==si & all_Y==3);
    fprintf('  A=%d, B=%d, C=%d\n', nA, nB, nC);
end

nTotal = size(all_X_3d, 1);
fprintf('\n总试次: %d (A=%d, B=%d, C=%d)\n', nTotal, sum(all_Y==1), sum(all_Y==2), sum(all_Y==3));
fprintf('3D数据维度: %d × %d × %d (试次 × 电极 × 时间点)\n', size(all_X_3d));

% 预计算特征索引（在 spatial 窗口内定位 N2/P3 子窗口）
% N2 和 P3 在 spatial 时间窗内的相对索引
N2_rel_tidx = find(times_spatial >= clf_N2_window(1) & times_spatial <= clf_N2_window(2));
P3_rel_tidx = find(times_spatial >= clf_P3_window(1) & times_spatial <= clf_P3_window(2));
% N2 电极在 spatial_idx 中的相对位置
N2_rel_cidx = []; P3_rel_cidx = [];
for ci = 1:length(N2_idx)
    N2_rel_cidx = [N2_rel_cidx, find(spatial_idx == N2_idx(ci))];
end
for ci = 1:length(P3_idx)
    P3_rel_cidx = [P3_rel_cidx, find(spatial_idx == P3_idx(ci))];
end

% 算法参数结构体
params = struct();
params.srate = srate_clf;
params.times = times_spatial;
params.N2_cidx = N2_rel_cidx;  params.P3_cidx = P3_rel_cidx;
params.N2_tidx = N2_rel_tidx;  params.P3_tidx = P3_rel_tidx;
params.nClasses = nClasses;
params.hdca_win_ms = hdca_win_ms;  params.hdca_step_ms = hdca_step_ms;
params.eegnet_F1 = eegnet_F1;  params.eegnet_D = eegnet_D;
params.eegnet_F2 = eegnet_F2;  params.eegnet_dropout = eegnet_dropout;
params.eegnet_epochs = eegnet_epochs;

% 保存提取的数据
save(fullfile(file_path, 'clf_data_3d.mat'), 'all_X_3d', 'all_Y', 'all_sid', ...
    'params', 'spatial_idx', 'N2_idx', 'P3_idx', 'clf_names', 'alg_names', ...
    'nSubj', 'SubjFiles', '-v7.3');
fprintf('数据已保存: %s\n', fullfile(file_path, 'clf_data_3d.mat'));

%% ========================= 9.3 被试内 10 折交叉验证 =========================
fprintf('\n====== 被试内 %d 折交叉验证 (9 种算法) ======\n', n_folds);

within_acc   = nan(nSubj, nAlg);
within_kappa = nan(nSubj, nAlg);
within_f1    = nan(nSubj, nAlg, nClasses);
within_cm    = cell(nSubj, nAlg);

for si = 1:nSubj
    mask = (all_sid == si);
    X_s = all_X_3d(mask, :, :);
    Y_s = all_Y(mask);
    ns  = length(Y_s);
    
    if ns < n_folds || length(unique(Y_s)) < nClasses
        fprintf('  被试 %d: 试次不足或类别不全，跳过\n', si);
        continue;
    end
    
    fprintf('\n  被试 %d (%s): %d 试次 [A=%d B=%d C=%d]\n', ...
        si, SubjFiles{si}, ns, sum(Y_s==1), sum(Y_s==2), sum(Y_s==3));
    
    cv = cvpartition(Y_s, 'KFold', n_folds);
    
    for ai = 1:nAlg
        Y_pred_all = nan(ns, 1);
        
        for fold = 1:n_folds
            tr = cv.training(fold); te = cv.test(fold);
            Xtr = X_s(tr,:,:); Ytr = Y_s(tr);
            Xte = X_s(te,:,:); Yte = Y_s(te);
            
            try
                Ypred = run_classifier(ai, Xtr, Ytr, Xte, params);
                Y_pred_all(te) = Ypred;
            catch ME
                fprintf('    [%s] fold%d 出错: %s\n', alg_names{ai}, fold, ME.message);
                Y_pred_all(te) = mode(Ytr);
            end
        end
        
        v = ~isnan(Y_pred_all);
        within_acc(si, ai) = mean(Y_s(v) == Y_pred_all(v));
        cm = confusionmat(Y_s(v), Y_pred_all(v), 'Order', [1 2 3]);
        within_cm{si, ai} = cm;
        within_kappa(si, ai) = cohen_kappa(cm);
        [~, ~, f1s] = prf_from_cm(cm);
        within_f1(si, ai, :) = f1s;
    end
    
    fprintf('    ');
    for ai = 1:nAlg
        fprintf('%s=%.1f%%  ', alg_names{ai}, within_acc(si, ai)*100);
    end
    fprintf('\n');
end

% 汇总
fprintf('\n====== 被试内 %d 折 CV 结果汇总 (A vs B vs C 三分类) ======\n', n_folds);
fprintf('%-16s  Acc(mean±SD)       Kappa(mean±SD)     F1_macro(mean±SD)\n', '算法');
fprintf('%s\n', repmat('-', 1, 75));
mean_acc_w = nan(1, nAlg);
for ai = 1:nAlg
    vs = ~isnan(within_acc(:, ai));
    a = within_acc(vs, ai); k = within_kappa(vs, ai);
    f1m = mean(squeeze(within_f1(vs, ai, :)), 2);
    mean_acc_w(ai) = mean(a);
    fprintf('%-16s  %.2f%% ± %.2f%%     %.3f ± %.3f       %.3f ± %.3f\n', ...
        alg_names{ai}, mean(a)*100, std(a)*100, mean(k), std(k), mean(f1m), std(f1m));
end

%% ========================= 9.4 LOSO-CV =========================
if do_loso
    fprintf('\n====== 跨被试留一交叉验证 (LOSO-CV) ======\n');
    loso_acc = nan(nSubj, nAlg);
    loso_kappa = nan(nSubj, nAlg);
    
    for si = 1:nSubj
        te_mask = (all_sid == si);
        tr_mask = ~te_mask;
        Xtr = all_X_3d(tr_mask,:,:); Ytr = all_Y(tr_mask);
        Xte = all_X_3d(te_mask,:,:); Yte = all_Y(te_mask);
        
        if isempty(Xte) || length(unique(Ytr)) < nClasses, continue; end
        
        fprintf('  LOSO 被试 %d/%d (train:%d, test:%d)...', si, nSubj, sum(tr_mask), sum(te_mask));
        for ai = 1:nAlg
            try
                Yp = run_classifier(ai, Xtr, Ytr, Xte, params);
                loso_acc(si, ai) = mean(Yte == Yp);
                cm = confusionmat(Yte, Yp, 'Order', [1 2 3]);
                loso_kappa(si, ai) = cohen_kappa(cm);
            catch ME
                fprintf('[%s ERR]', alg_names{ai});
            end
        end
        fprintf(' 完成\n');
    end
    
    fprintf('\n====== LOSO-CV 结果汇总 ======\n');
    fprintf('%-16s  Acc(mean±SD)\n', '算法');
    mean_acc_l = nan(1, nAlg);
    for ai = 1:nAlg
        vs = ~isnan(loso_acc(:, ai));
        a = loso_acc(vs, ai);
        mean_acc_l(ai) = mean(a);
        fprintf('%-16s  %.2f%% ± %.2f%%\n', alg_names{ai}, mean(a)*100, std(a)*100);
    end
end

%% ========================= 9.5 N2 vs P3 成分分类能力对比 =========================
fprintf('\n====== N2 vs P3 成分分类能力对比（仅 SVM）======\n');

feat_sets = {'N2_only', 'P3_only', 'N2+P3'};
comp_acc = nan(3, nSubj);

for fs = 1:3
    for si = 1:nSubj
        mask = (all_sid == si);
        X_s = all_X_3d(mask,:,:); Y_s = all_Y(mask);
        if length(Y_s) < n_folds || length(unique(Y_s)) < nClasses, continue; end
        
        % 根据特征集选择提取方式
        feat = extract_component_features(X_s, params, fs);
        
        cv = cvpartition(Y_s, 'KFold', n_folds);
        Yp = nan(length(Y_s), 1);
        for fold = 1:n_folds
            tr = cv.training(fold); te = cv.test(fold);
            Ftr = feat(tr,:); Fte = feat(te,:);
            [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
            Fte = (Fte - mu) ./ sg;
            mdl = fitcecoc(Ftr, Y_s(tr), 'Learners', templateSVM('KernelFunction','rbf'));
            Yp(te) = predict(mdl, Fte);
        end
        v = ~isnan(Yp);
        comp_acc(fs, si) = mean(Y_s(v) == Yp(v));
    end
end

fprintf('%-15s', ''); for ai = 1:3, fprintf('  %-12s', feat_sets{ai}); end; fprintf('\n');
fprintf('%-15s', 'SVM Acc(mean)');
for fs = 1:3
    vs = ~isnan(comp_acc(fs,:));
    fprintf('  %-12s', sprintf('%.1f%%', mean(comp_acc(fs, vs))*100));
end
fprintf('\n');

%% ========================= 9.6 可视化 =========================
fprintf('\n====== 绘制分类结果图表 ======\n');

fig_base = 200;
cmap9 = lines(nAlg);

% ---- 被试内 CV 柱状图 ----
figure('Name', '被试内CV准确率', 'NumberTitle', 'off', ...
    'Position', [50 50 1100 500], 'Color', 'w');
bar_d = mean_acc_w * 100;
bar_e = nan(1, nAlg);
for ai = 1:nAlg
    vs = ~isnan(within_acc(:,ai));
    bar_e(ai) = std(within_acc(vs,ai))*100;
end
b = bar(1:nAlg, bar_d, 0.7, 'FaceColor', 'flat');
for k = 1:nAlg, b.CData(k,:) = cmap9(k,:); end
hold on;
errorbar(1:nAlg, bar_d, bar_e, 'k.', 'LineWidth', 1.5, 'CapSize', 6);
line([0.5 nAlg+0.5], [33.33 33.33], 'Color', [.5 .5 .5], 'LineStyle', '--');
set(gca, 'XTick', 1:nAlg, 'XTickLabel', alg_names, 'XTickLabelRotation', 25);
ylabel('分类准确率 (%)'); box off;
for k = 1:nAlg
    text(k, bar_d(k)+bar_e(k)+1.5, sprintf('%.1f%%', bar_d(k)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end
ylim([0 max(bar_d+bar_e)+12]);
title(sprintf('被试内 %d 折 CV 准确率 (A vs B vs C)', n_folds), 'FontSize', 14);

% ---- 热力图 ----
figure('Name', '被试×算法热力图', 'NumberTitle', 'off', ...
    'Position', [50 50 1000 500], 'Color', 'w');
imagesc(within_acc * 100); colormap(parula); colorbar; caxis([20 70]);
set(gca, 'XTick', 1:nAlg, 'XTickLabel', alg_names, 'XTickLabelRotation', 25);
sn = cell(nSubj,1);
for si = 1:nSubj, [~,sn{si},~] = fileparts(SubjFiles{si}); end
set(gca, 'YTick', 1:nSubj, 'YTickLabel', sn);
xlabel('算法'); ylabel('被试');
title(sprintf('分类准确率热力图 (%d-fold CV)', n_folds), 'FontSize', 14);
for si = 1:nSubj
    for ai = 1:nAlg
        if ~isnan(within_acc(si,ai))
            text(ai, si, sprintf('%.0f', within_acc(si,ai)*100), ...
                'HorizontalAlignment','center','FontSize',7,'Color','w','FontWeight','bold');
        end
    end
end

% ---- LOSO vs Within 对比 ----
if do_loso
    figure('Name', '被试内 vs LOSO 对比', 'NumberTitle', 'off', ...
        'Position', [50 50 1000 500], 'Color', 'w');
    cmp = [mean_acc_w'*100, mean_acc_l'*100];
    b3 = bar(cmp, 'grouped');
    b3(1).FaceColor = [0.2 0.6 0.9]; b3(2).FaceColor = [0.9 0.4 0.3];
    set(gca, 'XTick', 1:nAlg, 'XTickLabel', alg_names, 'XTickLabelRotation', 25);
    ylabel('准确率 (%)');
    title('被试内 CV vs 跨被试 LOSO-CV', 'FontSize', 14);
    legend(sprintf('%d-fold CV', n_folds), 'LOSO-CV', 'Location', 'best');
    line([0.5 nAlg+0.5], [33.33 33.33], 'Color', [.5 .5 .5], 'LineStyle', '--');
    box off;
end

% ---- N2 vs P3 对比 ----
figure('Name', 'N2 vs P3 分类能力对比', 'NumberTitle', 'off', ...
    'Position', [50 50 600 400], 'Color', 'w');
np_mean = nan(1,3); np_se = nan(1,3);
for fs = 1:3
    vs = ~isnan(comp_acc(fs,:));
    np_mean(fs) = mean(comp_acc(fs,vs))*100;
    np_se(fs) = std(comp_acc(fs,vs))/sqrt(sum(vs))*100;
end
bp = bar(np_mean, 0.6, 'FaceColor', 'flat');
bp.CData = [0.2 0.5 0.9; 0.9 0.3 0.2; 0.3 0.8 0.3];
hold on; errorbar(1:3, np_mean, np_se, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'N2 (PO7/PO8)', 'P3 (Fz/Cz)', 'N2+P3'});
ylabel('SVM 准确率 (%)');
title('N2 vs P3 成分分类能力', 'FontSize', 13);
line([0.5 3.5], [33.33 33.33], 'Color', [.5 .5 .5], 'LineStyle', '--');
box off;

% ---- 最佳算法混淆矩阵 ----
[~, best_ai] = max(mean_acc_w);
figure('Name', sprintf('混淆矩阵 - %s', alg_names{best_ai}), ...
    'NumberTitle', 'off', 'Position', [50 50 450 400], 'Color', 'w');
scm = zeros(3,3);
for si = 1:nSubj
    if ~isempty(within_cm{si, best_ai}), scm = scm + within_cm{si, best_ai}; end
end
scm_pct = scm ./ max(sum(scm,2), 1) * 100;
imagesc(scm_pct); colormap(flipud(hot)); colorbar; caxis([0 100]);
set(gca, 'XTick', 1:3, 'XTickLabel', clf_names, 'YTick', 1:3, 'YTickLabel', clf_names);
xlabel('预测'); ylabel('真实');
title(sprintf('混淆矩阵 - %s (全部被试)', alg_names{best_ai}), 'FontSize', 12);
for r = 1:3
    for c = 1:3
        text(c, r, sprintf('%d\n%.1f%%', scm(r,c), scm_pct(r,c)), ...
            'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
    end
end

%% ========================= 9.7 导出 CSV =========================
fprintf('\n====== 导出分类结果 ======\n');
fname = fullfile(saca_csv_dir, 'Classification_9Alg_WithinSubj.csv');
fid = fopen(fname, 'w');
fprintf(fid, 'Subject');
for ai = 1:nAlg, fprintf(fid, ',%s', alg_names{ai}); end
fprintf(fid, '\n');
for si = 1:nSubj
    [~,nm,~] = fileparts(SubjFiles{si});
    fprintf(fid, '%s', nm);
    for ai = 1:nAlg, fprintf(fid, ',%.4f', within_acc(si,ai)); end
    fprintf(fid, '\n');
end
fprintf(fid, 'Mean');
for ai = 1:nAlg, fprintf(fid, ',%.4f', nanmean(within_acc(:,ai))); end
fprintf(fid, '\nSD');
for ai = 1:nAlg, fprintf(fid, ',%.4f', nanstd(within_acc(:,ai))); end
fprintf(fid, '\n');
fclose(fid);
fprintf('已导出: %s\n', fname);

if do_loso
    fname = fullfile(saca_csv_dir, 'Classification_9Alg_LOSO.csv');
    fid = fopen(fname, 'w');
    fprintf(fid, 'Subject');
    for ai = 1:nAlg, fprintf(fid, ',%s', alg_names{ai}); end
    fprintf(fid, '\n');
    for si = 1:nSubj
        [~,nm,~] = fileparts(SubjFiles{si});
        fprintf(fid, '%s', nm);
        for ai = 1:nAlg, fprintf(fid, ',%.4f', loso_acc(si,ai)); end
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('已导出: %s\n', fname);
end

% 保存完整结果
save(fullfile(file_path, 'classification_9alg_results.mat'), ...
    'within_acc', 'within_kappa', 'within_f1', 'within_cm', ...
    'alg_names', 'mean_acc_w', 'comp_acc', 'feat_sets');
if do_loso
    save(fullfile(file_path, 'classification_9alg_results.mat'), ...
        'loso_acc', 'loso_kappa', 'mean_acc_l', '-append');
end

% 保存图表
if ~exist(save_figure_dir, 'dir'), mkdir(save_figure_dir); end
all_figs = findobj('Type', 'figure');
for fi = 1:length(all_figs)
    h = all_figs(fi); fn = get(h, 'Name');
    if isempty(fn), fn = sprintf('CLF_Fig_%02d', fi); end
    fn = regexprep(fn, '[\\/:*?"<>|]', '_');
    try
        saveas(h, fullfile(save_figure_dir, [fn '.png']));
        savefig(h, fullfile(save_figure_dir, [fn '.fig']));
    catch, end
end

fprintf('\n====== Part 9 全部完成！ ======\n');

%% ========================================================================
%% ========================= 辅助函数 =====================================
%% ========================================================================

function Y_pred = run_classifier(alg_idx, Xtr, Ytr, Xte, p)
%RUN_CLASSIFIER 分发到 9 种算法
    switch alg_idx
        case 1, Y_pred = clf_hdca(Xtr, Ytr, Xte, p);
        case 2, Y_pred = clf_svm(Xtr, Ytr, Xte, p);
        case 3, Y_pred = clf_wavelet_svm(Xtr, Ytr, Xte, p);
        case 4, Y_pred = clf_xdawn_svm(Xtr, Ytr, Xte, p);
        case 5, Y_pred = clf_dcpm_svm(Xtr, Ytr, Xte, p);
        case 6, Y_pred = clf_ent_svm(Xtr, Ytr, Xte, p);
        case 7, Y_pred = clf_trca(Xtr, Ytr, Xte, p);
        case 8, Y_pred = clf_etrca(Xtr, Ytr, Xte, p);
        case 9, Y_pred = clf_eegnet(Xtr, Ytr, Xte, p);
    end
end

%% ---- 1. HDCA: 层次判别成分分析 ----
function Y_pred = clf_hdca(Xtr, Ytr, Xte, p)
    [nTr, nC, nT] = size(Xtr);
    nTe = size(Xte, 1);
    classes = unique(Ytr);
    nCls = length(classes);
    
    win_samp = max(1, round(p.hdca_win_ms / 1000 * p.srate));
    step_samp = max(1, round(p.hdca_step_ms / 1000 * p.srate));
    win_starts = 1:step_samp:(nT - win_samp + 1);
    nWin = length(win_starts);
    
    % One-vs-rest: 每个类别训练一组空间权重
    proj_tr = zeros(nTr, nWin * nCls);
    proj_te = zeros(nTe, nWin * nCls);
    
    for ci = 1:nCls
        Ybin = double(Ytr == classes(ci));
        idx1 = (Ybin == 1); idx0 = (Ybin == 0);
        
        for wi = 1:nWin
            tidx = win_starts(wi):(win_starts(wi) + win_samp - 1);
            tidx = tidx(tidx <= nT);
            
            X1 = squeeze(mean(Xtr(idx1, :, tidx), 3));  % n1 × nChan
            X0 = squeeze(mean(Xtr(idx0, :, tidx), 3));
            if size(X1,1) == 1, X1 = X1(:)'; end
            if size(X0,1) == 1, X0 = X0(:)'; end
            
            mu1 = mean(X1, 1)'; mu0 = mean(X0, 1)';
            Sw = cov(X1) + cov(X0) + eye(nC) * 1e-6;
            w = Sw \ (mu1 - mu0);
            
            col = (ci-1)*nWin + wi;
            Xtr_w = squeeze(mean(Xtr(:, :, tidx), 3));
            Xte_w = squeeze(mean(Xte(:, :, tidx), 3));
            if size(Xtr_w,2) ~= nC, Xtr_w = reshape(Xtr_w, [], nC); end
            if size(Xte_w,2) ~= nC, Xte_w = reshape(Xte_w, [], nC); end
            proj_tr(:, col) = Xtr_w * w;
            proj_te(:, col) = Xte_w * w;
        end
    end
    
    mdl = fitcdiscr(proj_tr, Ytr, 'DiscrimType', 'linear');
    Y_pred = predict(mdl, proj_te);
end

%% ---- 2. SVM: 支持向量机（ERP 特征）----
function Y_pred = clf_svm(Xtr, Ytr, Xte, p)
    Ftr = extract_component_features(Xtr, p, 3);
    Fte = extract_component_features(Xte, p, 3);
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte - mu) ./ sg;
    mdl = fitcecoc(Ftr, Ytr, 'Learners', templateSVM('KernelFunction','rbf'));
    Y_pred = predict(mdl, Fte);
end

%% ---- 3. Wavelet+SVM: 小波时频特征 + SVM ----
function Y_pred = clf_wavelet_svm(Xtr, Ytr, Xte, p)
    Ftr = extract_wavelet_features(Xtr, p);
    Fte = extract_wavelet_features(Xte, p);
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte - mu) ./ sg;
    mdl = fitcecoc(Ftr, Ytr, 'Learners', templateSVM('KernelFunction','rbf'));
    Y_pred = predict(mdl, Fte);
end

%% ---- 4. xDAWN+SVM: xDAWN 空间滤波 + SVM ----
function Y_pred = clf_xdawn_svm(Xtr, Ytr, Xte, p)
    [nTr, nC, nT] = size(Xtr);
    classes = unique(Ytr);
    nCls = length(classes);
    nComp = min(nC, 3);  % 每类取前 3 个成分
    
    % xDAWN: 对每个类别找增强 ERP 的空间滤波器
    filters_all = [];
    for ci = 1:nCls
        idx = (Ytr == classes(ci));
        A_mean = squeeze(mean(Xtr(idx, :, :), 1));  % nC × nT: 该类别的平均 ERP
        
        % 信号协方差 (ERP template)
        Cs = A_mean * A_mean';
        % 噪声协方差 (所有试次)
        Cn = zeros(nC, nC);
        for ti = 1:nTr
            Xi = squeeze(Xtr(ti, :, :));
            Cn = Cn + Xi * Xi';
        end
        Cn = Cn / nTr + eye(nC) * 1e-6;
        
        [V, D] = eig(Cs, Cn);
        [~, ord] = sort(diag(D), 'descend');
        V = V(:, ord(1:nComp));
        filters_all = [filters_all, V];
    end
    
    % 投影并提取特征
    nFilt = size(filters_all, 2);
    Ftr = zeros(nTr, nFilt * 3);  % 每个滤波器：均值、方差、峰值
    for ti = 1:nTr
        Xi = squeeze(Xtr(ti, :, :));
        proj = filters_all' * Xi;  % nFilt × nT
        for fi = 1:nFilt
            Ftr(ti, (fi-1)*3+1) = mean(proj(fi, :));
            Ftr(ti, (fi-1)*3+2) = var(proj(fi, :));
            Ftr(ti, (fi-1)*3+3) = max(abs(proj(fi, :)));
        end
    end
    
    nTe = size(Xte, 1);
    Fte = zeros(nTe, nFilt * 3);
    for ti = 1:nTe
        Xi = squeeze(Xte(ti, :, :));
        proj = filters_all' * Xi;
        for fi = 1:nFilt
            Fte(ti, (fi-1)*3+1) = mean(proj(fi, :));
            Fte(ti, (fi-1)*3+2) = var(proj(fi, :));
            Fte(ti, (fi-1)*3+3) = max(abs(proj(fi, :)));
        end
    end
    
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte - mu) ./ sg;
    mdl = fitcecoc(Ftr, Ytr, 'Learners', templateSVM('KernelFunction','rbf'));
    Y_pred = predict(mdl, Fte);
end

%% ---- 5. DCPM+SVM: 判别典型模式匹配 + SVM ----
function Y_pred = clf_dcpm_svm(Xtr, Ytr, Xte, p)
    [nTr, nC, nT] = size(Xtr);
    classes = unique(Ytr);
    nCls = length(classes);
    
    % 计算每个类别的平均模板
    templates = zeros(nCls, nC, nT);
    for ci = 1:nCls
        templates(ci, :, :) = mean(Xtr(Ytr == classes(ci), :, :), 1);
    end
    
    % 计算类间和类内协方差的判别空间滤波器
    Sb = zeros(nC, nC);
    Sw = zeros(nC, nC);
    grand_mean = squeeze(mean(Xtr, 1));  % nC × nT
    
    for ci = 1:nCls
        ni = sum(Ytr == classes(ci));
        diff_i = squeeze(templates(ci,:,:)) - grand_mean;
        Sb = Sb + ni * (diff_i * diff_i');
        
        idx_ci = find(Ytr == classes(ci));
        for ti = 1:length(idx_ci)
            diff_w = squeeze(Xtr(idx_ci(ti),:,:)) - squeeze(templates(ci,:,:));
            Sw = Sw + diff_w * diff_w';
        end
    end
    Sw = Sw + eye(nC) * 1e-6;
    
    [V, D] = eig(Sb, Sw);
    [~, ord] = sort(diag(D), 'descend');
    nComp = min(nC, nCls);
    W = V(:, ord(1:nComp));
    
    % 投影到判别空间并计算与模板的相关性作为特征
    Ftr = zeros(nTr, nCls * nComp);
    for ti = 1:nTr
        Xi = squeeze(Xtr(ti,:,:));
        proj_trial = W' * Xi;  % nComp × nT
        for ci = 1:nCls
            proj_tmpl = W' * squeeze(templates(ci,:,:));
            for ki = 1:nComp
                Ftr(ti, (ci-1)*nComp+ki) = corr(proj_trial(ki,:)', proj_tmpl(ki,:)');
            end
        end
    end
    
    nTe = size(Xte, 1);
    Fte = zeros(nTe, nCls * nComp);
    for ti = 1:nTe
        Xi = squeeze(Xte(ti,:,:));
        proj_trial = W' * Xi;
        for ci = 1:nCls
            proj_tmpl = W' * squeeze(templates(ci,:,:));
            for ki = 1:nComp
                Fte(ti, (ci-1)*nComp+ki) = corr(proj_trial(ki,:)', proj_tmpl(ki,:)');
            end
        end
    end
    
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte - mu) ./ sg;
    mdl = fitcecoc(Ftr, Ytr, 'Learners', templateSVM('KernelFunction','rbf'));
    Y_pred = predict(mdl, Fte);
end

%% ---- 6. ENT+SVM: 熵特征 + SVM ----
function Y_pred = clf_ent_svm(Xtr, Ytr, Xte, p)
    Ftr = extract_entropy_features(Xtr, p);
    Fte = extract_entropy_features(Xte, p);
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte - mu) ./ sg;
    mdl = fitcecoc(Ftr, Ytr, 'Learners', templateSVM('KernelFunction','rbf'));
    Y_pred = predict(mdl, Fte);
end

%% ---- 7. TRCA: 任务相关成分分析 ----
function Y_pred = clf_trca(Xtr, Ytr, Xte, p)
    [~, nC, nT] = size(Xtr);
    classes = unique(Ytr);
    nCls = length(classes);
    
    W = zeros(nC, nCls);         % 每个类别一个空间滤波器
    templates = zeros(nCls, nT); % 滤波后的模板
    
    for ci = 1:nCls
        idx = find(Ytr == classes(ci));
        ni = length(idx);
        X_ci = Xtr(idx, :, :);  % ni × nC × nT
        
        % S 矩阵：试次间协方差（最大化试次间一致性）
        S = zeros(nC, nC);
        for j1 = 1:ni
            for j2 = (j1+1):ni
                Xj1 = squeeze(X_ci(j1,:,:));
                Xj2 = squeeze(X_ci(j2,:,:));
                S = S + Xj1 * Xj2' + Xj2 * Xj1';
            end
        end
        
        % Q 矩阵：总协方差
        Q = zeros(nC, nC);
        for j = 1:ni
            Xj = squeeze(X_ci(j,:,:));
            Q = Q + Xj * Xj';
        end
        Q = Q + eye(nC) * 1e-6;
        
        [V, D] = eig(S, Q);
        [~, maxidx] = max(real(diag(D)));
        w = real(V(:, maxidx));
        W(:, ci) = w;
        
        % 模板：该类别所有试次的平均 → 空间滤波
        avg_ci = squeeze(mean(X_ci, 1));  % nC × nT
        templates(ci, :) = w' * avg_ci;
    end
    
    % 分类：测试试次投影到每个类的空间滤波器，与模板求相关
    nTe = size(Xte, 1);
    Y_pred = zeros(nTe, 1);
    for ti = 1:nTe
        Xi = squeeze(Xte(ti, :, :));
        corrs = zeros(1, nCls);
        for ci = 1:nCls
            proj = W(:, ci)' * Xi;
            corrs(ci) = corr(proj', templates(ci,:)');
        end
        [~, Y_pred(ti)] = max(corrs);
        Y_pred(ti) = classes(Y_pred(ti));
    end
end

%% ---- 8. eTRCA: 集成 TRCA ----
function Y_pred = clf_etrca(Xtr, Ytr, Xte, p)
    [~, nC, nT] = size(Xtr);
    classes = unique(Ytr);
    nCls = length(classes);
    
    % 训练每个类别的空间滤波器（同 TRCA）
    W = zeros(nC, nCls);
    avg_all = zeros(nCls, nC, nT);
    
    for ci = 1:nCls
        idx = find(Ytr == classes(ci));
        ni = length(idx);
        X_ci = Xtr(idx, :, :);
        
        S = zeros(nC, nC);
        for j1 = 1:ni
            for j2 = (j1+1):ni
                Xj1 = squeeze(X_ci(j1,:,:));
                Xj2 = squeeze(X_ci(j2,:,:));
                S = S + Xj1*Xj2' + Xj2*Xj1';
            end
        end
        Q = zeros(nC, nC);
        for j = 1:ni
            Xj = squeeze(X_ci(j,:,:));
            Q = Q + Xj*Xj';
        end
        Q = Q + eye(nC)*1e-6;
        
        [V, D] = eig(S, Q);
        [~, mi] = max(real(diag(D)));
        W(:, ci) = real(V(:, mi));
        avg_all(ci, :, :) = mean(X_ci, 1);
    end
    
    % 集成分类：对每个类别模板，使用所有类的空间滤波器的相关性之和
    nTe = size(Xte, 1);
    Y_pred = zeros(nTe, 1);
    for ti = 1:nTe
        Xi = squeeze(Xte(ti, :, :));
        scores = zeros(1, nCls);
        for ci = 1:nCls
            tmpl_ci = squeeze(avg_all(ci, :, :));  % nC × nT
            for fi = 1:nCls  % 用所有类别的滤波器
                proj_test = W(:, fi)' * Xi;
                proj_tmpl = W(:, fi)' * tmpl_ci;
                scores(ci) = scores(ci) + corr(proj_test', proj_tmpl');
            end
        end
        [~, Y_pred(ti)] = max(scores);
        Y_pred(ti) = classes(Y_pred(ti));
    end
end

%% ---- 9. EEGNet: 紧凑型卷积神经网络 ----
function Y_pred = clf_eegnet(Xtr, Ytr, Xte, p)
    [nTr, nC, nT] = size(Xtr);
    nTe = size(Xte, 1);
    classes = unique(Ytr);
    nCls = length(classes);
    
    F1 = p.eegnet_F1; D = p.eegnet_D; F2 = p.eegnet_F2;
    kernLength = min(nT, max(32, round(p.srate * 0.25)));  % ~250ms
    
    % 重排为 4D: nC × nT × 1 × nTrials
    Xtr_4d = permute(Xtr, [2 3 4 1]);                 % nC × nT × 1 × nTr
    Xtr_4d = reshape(Xtr_4d, nC, nT, 1, nTr);
    Xte_4d = permute(Xte, [2 3 4 1]);
    Xte_4d = reshape(Xte_4d, nC, nT, 1, nTe);
    
    % 标签 → categorical
    Ytr_cat = categorical(Ytr);
    
    try
        % EEGNet 架构
        layers = [
            imageInputLayer([nC nT 1], 'Normalization', 'none', 'Name', 'input')
            
            % Block 1: 时间卷积
            convolution2dLayer([1 kernLength], F1, 'Padding', 'same', 'Name', 'conv_temporal')
            batchNormalizationLayer('Name', 'bn1')
            
            % 深度卷积（空间滤波）
            groupedConvolution2dLayer([nC 1], D, F1, 'Name', 'conv_depth')
            batchNormalizationLayer('Name', 'bn2')
            eluLayer(1, 'Name', 'elu1')
            averagePooling2dLayer([1 4], 'Stride', [1 4], 'Name', 'pool1')
            dropoutLayer(p.eegnet_dropout, 'Name', 'drop1')
            
            % Block 2: 可分离卷积
            convolution2dLayer([1 16], F2, 'Padding', 'same', 'Name', 'conv_sep')
            batchNormalizationLayer('Name', 'bn3')
            eluLayer(1, 'Name', 'elu2')
            averagePooling2dLayer([1 8], 'Stride', [1 8], 'Name', 'pool2')
            dropoutLayer(p.eegnet_dropout, 'Name', 'drop2')
            
            % 分类器
            fullyConnectedLayer(nCls, 'Name', 'fc')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];
        
        opts = trainingOptions('adam', ...
            'MaxEpochs', p.eegnet_epochs, ...
            'MiniBatchSize', min(32, floor(nTr/2)), ...
            'InitialLearnRate', 1e-3, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'ValidationFrequency', 50, ...
            'L2Regularization', 1e-4);
        
        net = trainNetwork(Xtr_4d, Ytr_cat, layers, opts);
        Ypred_cat = classify(net, Xte_4d);
        Y_pred = double(Ypred_cat);
    catch ME
        % Deep Learning Toolbox 不可用时回退到 BP 神经网络
        warning('EEGNet 构建失败 (%s)，回退到 BP 神经网络', ME.message);
        Y_pred = clf_bpnn_fallback(Xtr, Ytr, Xte, p);
    end
end

%% ---- BP 神经网络回退方案 ----
function Y_pred = clf_bpnn_fallback(Xtr, Ytr, Xte, p)
    Ftr = extract_component_features(Xtr, p, 3);
    Fte = extract_component_features(Xte, p, 3);
    [Ftr, mu, sg] = zscore(Ftr); sg(sg==0)=1;
    Fte = (Fte - mu) ./ sg;
    
    classes = unique(Ytr); nCls = length(classes);
    T = zeros(nCls, length(Ytr));
    for k = 1:nCls, T(k, Ytr==classes(k)) = 1; end
    
    net = patternnet([20 10]);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = 200;
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0;
    net = train(net, Ftr', T);
    
    Yout = net(Fte');
    [~, pidx] = max(Yout, [], 1);
    Y_pred = classes(pidx)';
end

%% ========================= 特征提取辅助函数 =========================

function feat = extract_component_features(X3d, p, set_id)
%EXTRACT_COMPONENT_FEATURES 提取 N2/P3 均值+峰值+潜伏期特征
%   set_id: 1=N2 only, 2=P3 only, 3=N2+P3
    nTrials = size(X3d, 1);
    feat = [];
    
    if set_id == 1 || set_id == 3
        % N2: PO7/PO8 × [200-250ms]
        for ci = 1:length(p.N2_cidx)
            ch = p.N2_cidx(ci);
            for ti = 1:nTrials
                seg = squeeze(X3d(ti, ch, p.N2_tidx));
                feat(ti, end+1) = mean(seg);             % 均值
                [pk, pi] = min(seg);                      % N2 负峰
                feat(ti, end+1) = pk;
                feat(ti, end+1) = p.times(p.N2_tidx(pi)); % 潜伏期
            end
        end
    end
    
    if set_id == 2 || set_id == 3
        % P3: Fz/Cz × [300-350ms]
        for ci = 1:length(p.P3_cidx)
            ch = p.P3_cidx(ci);
            for ti = 1:nTrials
                seg = squeeze(X3d(ti, ch, p.P3_tidx));
                feat(ti, end+1) = mean(seg);             % 均值
                [pk, pi] = max(seg);                      % P3 正峰
                feat(ti, end+1) = pk;
                feat(ti, end+1) = p.times(p.P3_tidx(pi));
            end
        end
    end
end

function feat = extract_wavelet_features(X3d, p)
%EXTRACT_WAVELET_FEATURES 小波时频特征提取
    nTrials = size(X3d, 1);
    all_cidx = [p.N2_cidx, p.P3_cidx];
    all_tidx = unique([p.N2_tidx, p.P3_tidx]);
    
    feat = [];
    for ti = 1:nTrials
        fvec = [];
        for ci = 1:length(all_cidx)
            ch = all_cidx(ci);
            sig = squeeze(X3d(ti, ch, all_tidx));
            
            % 小波分解 (db4, 3层)
            try
                [C, L] = wavedec(sig, 3, 'db4');
                % 每层的能量和均值
                for lv = 1:3
                    d = detcoef(C, L, lv);
                    fvec = [fvec, mean(d), std(d), sum(d.^2)/length(d)];
                end
                a = appcoef(C, L, 'db4', 3);
                fvec = [fvec, mean(a), std(a), sum(a.^2)/length(a)];
            catch
                % 信号太短时用原始统计量
                fvec = [fvec, mean(sig), std(sig), max(sig), min(sig), ...
                         mean(sig.^2), skewness(sig)];
            end
        end
        feat(ti, :) = fvec;
    end
end

function feat = extract_entropy_features(X3d, p)
%EXTRACT_ENTROPY_FEATURES 多种熵特征提取
    nTrials = size(X3d, 1);
    all_cidx = [p.N2_cidx, p.P3_cidx];
    all_tidx_sets = {p.N2_tidx, p.P3_tidx};
    
    feat = [];
    for ti = 1:nTrials
        fvec = [];
        for ci = 1:length(all_cidx)
            ch = all_cidx(ci);
            for si = 1:length(all_tidx_sets)
                sig = squeeze(X3d(ti, ch, all_tidx_sets{si}));
                sig = sig(:)';
                
                % 1. 样本熵 (简化版)
                fvec(end+1) = sample_entropy(sig, 2, 0.2*std(sig));
                
                % 2. 谱熵
                psd = abs(fft(sig)).^2;
                psd = psd(1:floor(length(psd)/2)+1);
                psd_norm = psd / sum(psd);
                psd_norm(psd_norm == 0) = eps;
                fvec(end+1) = -sum(psd_norm .* log2(psd_norm));
                
                % 3. 排列熵 (order=3)
                fvec(end+1) = perm_entropy(sig, 3);
                
                % 4. 对数能量熵
                fvec(end+1) = sum(log(sig.^2 + eps));
            end
        end
        feat(ti, :) = fvec;
    end
end

function se = sample_entropy(sig, m, r)
%SAMPLE_ENTROPY 计算样本熵
    N = length(sig);
    if N < m + 2, se = 0; return; end
    
    count = zeros(1, 2);
    for dim = m:(m+1)
        templates = zeros(N-dim, dim);
        for i = 1:(N-dim)
            templates(i,:) = sig(i:(i+dim-1));
        end
        n_tmpl = size(templates, 1);
        cnt = 0;
        for i = 1:n_tmpl
            for j = (i+1):n_tmpl
                if max(abs(templates(i,:) - templates(j,:))) < r
                    cnt = cnt + 1;
                end
            end
        end
        count(dim - m + 1) = cnt;
    end
    
    if count(1) == 0 || count(2) == 0
        se = 0;
    else
        se = -log(count(2) / count(1));
    end
end

function pe = perm_entropy(sig, order)
%PERM_ENTROPY 计算排列熵
    N = length(sig);
    if N < order + 1, pe = 0; return; end
    
    n_perm = factorial(order);
    pattern_count = zeros(1, n_perm);
    
    for i = 1:(N - order + 1)
        seg = sig(i:(i+order-1));
        [~, patt] = sort(seg);
        
        % 将排列模式编码为唯一索引
        idx = 0;
        for k = 1:order
            idx = idx + (patt(k)-1) * factorial(order - k);
        end
        pattern_count(idx + 1) = pattern_count(idx + 1) + 1;
    end
    
    p = pattern_count / sum(pattern_count);
    p(p == 0) = [];
    pe = -sum(p .* log2(p)) / log2(n_perm);
end

%% ========================= 评估辅助函数 =========================

function [prec, rec, f1] = prf_from_cm(cm)
    nC = size(cm, 1);
    prec = zeros(1, nC); rec = zeros(1, nC); f1 = zeros(1, nC);
    for c = 1:nC
        tp = cm(c,c); fp = sum(cm(:,c))-tp; fn = sum(cm(c,:))-tp;
        if tp+fp > 0, prec(c) = tp/(tp+fp); end
        if tp+fn > 0, rec(c) = tp/(tp+fn); end
        if prec(c)+rec(c) > 0, f1(c) = 2*prec(c)*rec(c)/(prec(c)+rec(c)); end
    end
end

function k = cohen_kappa(cm)
    n = sum(cm(:));
    if n == 0, k = 0; return; end
    po = trace(cm)/n;
    pe = sum(sum(cm,2).*sum(cm,1)') / n^2;
    if pe == 1, k = 0; else, k = (po-pe)/(1-pe); end
end
