%% ========================================================================
%%  SSVEP脑电信号分类完整脚本 (单文件版)
%%  利用9种算法对三种条件刺激(A/B/C)下10名被试的SSVEP信号进行分类
%%  数据格式: EEGLAB .set / .fdt 文件
%% ========================================================================
%
%  标记含义:
%    11 = A刺激       21 = B刺激       31 = C刺激
%    41 = 视觉搜索任务(有目标)  42 = 视觉搜索任务(无目标)
%    12 = 回答正确     22 = 回答错误     32 = 无反应
%
%  分类算法:
%    1. SVM (线性核)    2. SVM (RBF核)     3. Wavelet+SVM
%    4. HDCA           5. xDAWN+SVM       6. DCPM+SVM
%    7. ENT+SVM        8. TRCA            9. eTRCA
%
%  运行前请确认:
%    1. 已安装 EEGLAB 并加入路径 (addpath)
%    2. 已安装 Statistics and Machine Learning Toolbox
%    3. 已安装 Signal Processing Toolbox
%    4. 已安装 Wavelet Toolbox
%    5. 修改下方 "用户配置区" 中的路径和参数

clear all; close all; clc;
rng(42);

%% ====================== 用户配置区 (请修改) ==========================

% ---------- 数据路径 ----------
% .set文件所在文件夹 (文件夹内放置所有被试的 .set 和 .fdt 文件)
data_folder = 'D:\SSVEP_Data\';   % <-- 修改为你的数据路径

% 被试文件名列表 (不含后缀, 对应 xxx.set 和 xxx.fdt)
subject_files = {
    'subject01', ...
    'subject02', ...
    'subject03', ...
    'subject04', ...
    'subject05', ...
    'subject06', ...
    'subject07', ...
    'subject08', ...
    'subject09', ...
    'subject10'  ...
};  % <-- 修改为你的文件名

% ---------- 实验参数 ----------
fs             = 250;          % 采样率 (Hz), 根据实际数据修改
epoch_time     = [-0.5, 3.0];  % 截取时间窗 (秒), 相对于刺激onset
stim_markers   = [11, 21, 31]; % 三种刺激的event标记
stim_freqs     = [8, 10, 12];  % 三种刺激对应的SSVEP频率 (Hz)
n_harmonics    = 3;            % 分析的谐波数量
K_fold         = 10;           % 交叉验证折数
wavelet_name   = 'db4';        % 小波基函数
wavelet_level  = 5;            % 小波分解层数

% ====================== 配置区结束 ====================================

n_subjects = length(subject_files);
n_classes  = length(stim_markers);

method_names = {'SVM_Linear','SVM_RBF','Wavelet_SVM','HDCA',...
                'xDAWN_SVM','DCPM_SVM','ENT_SVM','TRCA','eTRCA'};
n_methods = length(method_names);

% 汇总结果矩阵 [subjects x methods]
acc_all = zeros(n_subjects, n_methods);
f1_all  = zeros(n_subjects, n_methods);

% 保存每个被试的完整结果
all_results = struct();

fprintf('============================================================\n');
fprintf('   SSVEP 脑电信号多方法分类分析\n');
fprintf('   被试数: %d   类别数: %d   方法数: %d\n', n_subjects, n_classes, n_methods);
fprintf('============================================================\n');

%% ====================== 主循环: 逐被试分析 ===========================
for sub = 1:n_subjects

    fprintf('\n==================== 被试 %d / %d ====================\n', sub, n_subjects);

    %% ---------- 1. 加载 .set 文件 ----------
    set_file = fullfile(data_folder, [subject_files{sub}, '.set']);
    fprintf('  加载文件: %s\n', set_file);

    EEG = pop_loadset('filename', [subject_files{sub}, '.set'], 'filepath', data_folder);

    actual_fs = EEG.srate;
    n_channels_raw = EEG.nbchan;
    fprintf('  采样率: %d Hz, 通道数: %d, 总时长: %.1f s\n', ...
        actual_fs, n_channels_raw, EEG.pnts/actual_fs);

    % 如果实际采样率与设定不同, 进行重采样
    if actual_fs ~= fs
        fprintf('  重采样: %d -> %d Hz\n', actual_fs, fs);
        EEG = pop_resample(EEG, fs);
    end

    %% ---------- 2. 提取事件并分段 ----------
    epoch_samples = round(epoch_time * fs);
    n_epoch_pts   = epoch_samples(2) - epoch_samples(1) + 1;
    n_ch          = EEG.nbchan;

    stim_events = [];
    for e = 1:length(EEG.event)
        evt = EEG.event(e).type;
        if isnumeric(evt)
            mk = evt;
        elseif ischar(evt) || isstring(evt)
            mk = str2double(evt);
        else
            continue;
        end
        if ~isnan(mk) && ismember(mk, stim_markers)
            stim_events = [stim_events; mk, round(EEG.event(e).latency)]; %#ok<AGROW>
        end
    end

    n_trials = size(stim_events, 1);
    fprintf('  检测到 %d 个刺激事件 (A=%d, B=%d, C=%d)\n', n_trials, ...
        sum(stim_events(:,1)==stim_markers(1)), ...
        sum(stim_events(:,1)==stim_markers(2)), ...
        sum(stim_events(:,1)==stim_markers(3)));

    if n_trials == 0
        fprintf('  [警告] 未找到刺激事件, 跳过该被试\n');
        continue;
    end

    EEG_epochs = zeros(n_ch, n_epoch_pts, n_trials);
    labels     = zeros(n_trials, 1);
    valid      = true(n_trials, 1);

    for t = 1:n_trials
        s_start = stim_events(t,2) + epoch_samples(1);
        s_end   = stim_events(t,2) + epoch_samples(2);

        if s_start < 1 || s_end > size(EEG.data, 2)
            valid(t) = false;
            continue;
        end

        EEG_epochs(:,:,t) = EEG.data(:, s_start:s_end);

        for c = 1:n_classes
            if stim_events(t,1) == stim_markers(c)
                labels(t) = c;
            end
        end
    end

    EEG_epochs = EEG_epochs(:,:,valid);
    labels     = labels(valid);
    n_trials   = sum(valid);
    fprintf('  有效试次: %d (A=%d, B=%d, C=%d)\n', n_trials, ...
        sum(labels==1), sum(labels==2), sum(labels==3));

    %% ---------- 3. 预处理 ----------
    fprintf('  预处理...\n');
    baseline_end = abs(epoch_time(1)) * fs;
    [b_bp, a_bp] = butter(4, [1 40]/(fs/2), 'bandpass');

    for t = 1:n_trials
        dat = double(EEG_epochs(:,:,t));
        if baseline_end > 0
            bl = mean(dat(:, 1:round(baseline_end)), 2);
            dat = dat - bl;
        end
        for ch = 1:n_ch
            dat(ch,:) = filtfilt(b_bp, a_bp, dat(ch,:));
            dat(ch,:) = detrend(dat(ch,:));
        end
        EEG_epochs(:,:,t) = dat;
    end

    % 伪迹剔除 (幅值 > 100 μV)
    bad = false(n_trials,1);
    for t = 1:n_trials
        if max(abs(EEG_epochs(:,:,t)),[],'all') > 100
            bad(t) = true;
        end
    end
    if any(bad)
        fprintf('  去除 %d 个伪迹试次\n', sum(bad));
        EEG_epochs(:,:,bad) = [];
        labels(bad) = [];
        n_trials = length(labels);
    end

    %% ---------- 4. 交叉验证分折 ----------
    cv_idx = crossvalind('Kfold', labels, K_fold);

    %% ---------- 5. 特征提取 ----------
    fprintf('  提取特征...\n');

    % --- PSD特征 (SVM方法1,2共用) ---
    freqs_interest = [];
    for c = 1:n_classes
        for h = 1:n_harmonics
            freqs_interest = [freqs_interest, stim_freqs(c)*h]; %#ok<AGROW>
        end
    end
    nfft_psd = 2^nextpow2(n_epoch_pts);
    bw = 1; % Hz
    feat_psd = zeros(n_trials, n_ch * length(freqs_interest));
    for t = 1:n_trials
        fi = 0;
        for ch = 1:n_ch
            sig = detrend(EEG_epochs(ch,:,t));
            [pxx, f_psd] = pwelch(sig,[],[],nfft_psd,fs);
            for fq = 1:length(freqs_interest)
                fi = fi + 1;
                idx_f = (f_psd >= freqs_interest(fq)-bw) & (f_psd <= freqs_interest(fq)+bw);
                feat_psd(t,fi) = mean(pxx(idx_f));
            end
        end
    end
    feat_psd = log10(feat_psd + eps);

    % --- 小波特征 (方法3) ---
    n_feat_per_lv = 4;
    feat_wav = zeros(n_trials, n_ch*(wavelet_level+1)*n_feat_per_lv);
    for t = 1:n_trials
        vec = [];
        for ch = 1:n_ch
            sig = double(EEG_epochs(ch,:,t));
            [C,L] = wavedec(sig, wavelet_level, wavelet_name);
            for lv = 1:wavelet_level
                d = detcoef(C,L,lv);
                vec = [vec, wstats(d)]; %#ok<AGROW>
            end
            a = appcoef(C,L,wavelet_name,wavelet_level);
            vec = [vec, wstats(a)]; %#ok<AGROW>
        end
        feat_wav(t,:) = vec;
    end

    % --- 熵特征 (方法7) ---
    feat_ent = zeros(n_trials, n_ch*4);
    for t = 1:n_trials
        vec = [];
        for ch = 1:n_ch
            sig = double(EEG_epochs(ch,:,t));
            se  = fn_sample_entropy(sig, 2, 0.2*std(sig));
            fe  = fn_fuzzy_entropy(sig, 2, 0.2*std(sig));
            pe  = fn_permutation_entropy(sig, 3, 1);
            spe = fn_spectral_entropy(sig, fs);
            vec = [vec, se, fe, pe, spe]; %#ok<AGROW>
        end
        feat_ent(t,:) = vec;
    end
    feat_ent(isnan(feat_ent)) = 0;
    feat_ent(isinf(feat_ent)) = 0;

    %% ---------- 6. 运行9种分类方法 ----------

    % ===== 方法 1: SVM Linear =====
    fprintf('  [1/9] SVM (线性核)...\n');
    res = fn_classify_svm(feat_psd, labels, cv_idx, K_fold, n_classes, 'linear');
    all_results(sub).SVM_Linear = res;
    acc_all(sub,1) = res.accuracy; f1_all(sub,1) = res.macro_f1;

    % ===== 方法 2: SVM RBF =====
    fprintf('  [2/9] SVM (RBF核)...\n');
    res = fn_classify_svm(feat_psd, labels, cv_idx, K_fold, n_classes, 'rbf');
    all_results(sub).SVM_RBF = res;
    acc_all(sub,2) = res.accuracy; f1_all(sub,2) = res.macro_f1;

    % ===== 方法 3: Wavelet + SVM =====
    fprintf('  [3/9] Wavelet+SVM...\n');
    res = fn_classify_svm(feat_wav, labels, cv_idx, K_fold, n_classes, 'linear');
    all_results(sub).Wavelet_SVM = res;
    acc_all(sub,3) = res.accuracy; f1_all(sub,3) = res.macro_f1;

    % ===== 方法 4: HDCA =====
    fprintf('  [4/9] HDCA...\n');
    res = fn_classify_hdca(EEG_epochs, labels, cv_idx, K_fold, n_classes, fs);
    all_results(sub).HDCA = res;
    acc_all(sub,4) = res.accuracy; f1_all(sub,4) = res.macro_f1;

    % ===== 方法 5: xDAWN + SVM =====
    fprintf('  [5/9] xDAWN+SVM...\n');
    res = fn_classify_xdawn(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).xDAWN_SVM = res;
    acc_all(sub,5) = res.accuracy; f1_all(sub,5) = res.macro_f1;

    % ===== 方法 6: DCPM + SVM =====
    fprintf('  [6/9] DCPM+SVM...\n');
    res = fn_classify_dcpm(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).DCPM_SVM = res;
    acc_all(sub,6) = res.accuracy; f1_all(sub,6) = res.macro_f1;

    % ===== 方法 7: ENT + SVM =====
    fprintf('  [7/9] ENT+SVM...\n');
    res = fn_classify_svm(feat_ent, labels, cv_idx, K_fold, n_classes, 'linear');
    all_results(sub).ENT_SVM = res;
    acc_all(sub,7) = res.accuracy; f1_all(sub,7) = res.macro_f1;

    % ===== 方法 8: TRCA =====
    fprintf('  [8/9] TRCA...\n');
    res = fn_classify_trca(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).TRCA = res;
    acc_all(sub,8) = res.accuracy; f1_all(sub,8) = res.macro_f1;

    % ===== 方法 9: eTRCA =====
    fprintf('  [9/9] eTRCA...\n');
    res = fn_classify_etrca(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).eTRCA = res;
    acc_all(sub,9) = res.accuracy; f1_all(sub,9) = res.macro_f1;

end % --- 被试循环结束 ---

%% ====================== 结果汇总与可视化 =============================
fprintf('\n\n');
fprintf('================================================================\n');
fprintf('                    分 类 结 果 汇 总\n');
fprintf('================================================================\n');
fprintf('%-18s %-10s %-10s %-10s %-10s %-10s\n', ...
    '方法','平均ACC%','标准差%','最高%','最低%','平均F1');
fprintf('%s\n', repmat('-',1,68));
for m = 1:n_methods
    a = acc_all(:,m)*100;
    fprintf('%-18s %-10.2f %-10.2f %-10.2f %-10.2f %-10.4f\n', ...
        method_names{m}, mean(a), std(a), max(a), min(a), mean(f1_all(:,m)));
end
fprintf('%s\n', repmat('=',1,68));

fprintf('\n各被试分类准确率 (%%):\n');
fprintf('%-8s','被试');
for m = 1:n_methods, fprintf('%-13s', method_names{m}); end
fprintf('\n%s\n', repmat('-',1,8+13*n_methods));
for s = 1:n_subjects
    fprintf('%-8s', sprintf('S%02d',s));
    for m = 1:n_methods
        fprintf('%-13.2f', acc_all(s,m)*100);
    end
    fprintf('\n');
end

% --- 方法间配对 t 检验 ---
fprintf('\n方法间配对 t 检验 (p 值, * 表示 p<0.05):\n');
fprintf('%-18s','');
for m = 1:n_methods, fprintf('%-12s', method_names{m}); end
fprintf('\n');
for m1 = 1:n_methods
    fprintf('%-18s', method_names{m1});
    for m2 = 1:n_methods
        if m1==m2
            fprintf('%-12s', '-');
        else
            [~,p] = ttest(acc_all(:,m1), acc_all(:,m2));
            if p < 0.05
                fprintf('%-12s', sprintf('%.4f*',p));
            else
                fprintf('%-12.4f', p);
            end
        end
    end
    fprintf('\n');
end

%% ---------- 图1: 平均准确率柱状图 ----------
figure('Name','平均分类准确率','Position',[50 50 950 500]);
mn = mean(acc_all,1)*100; sd = std(acc_all,0,1)*100;
bh = bar(mn,'FaceColor','flat'); hold on;
errorbar(1:n_methods, mn, sd, 'k.','LineWidth',1.5);
colors = lines(n_methods);
for m = 1:n_methods, bh.CData(m,:) = colors(m,:); end
set(gca,'XTick',1:n_methods,'XTickLabel',method_names,'XTickLabelRotation',30);
ylabel('分类准确率 (%)'); ylim([0 105]);
title('各方法在10名被试上的平均分类准确率');
yline(100/n_classes,'--r','随机水平','LineWidth',1.5);
grid on;
saveas(gcf,'Fig1_Average_Accuracy.png');

%% ---------- 图2: 箱线图 ----------
figure('Name','准确率分布','Position',[50 50 950 500]);
boxplot(acc_all*100,'Labels',method_names);
set(gca,'XTickLabelRotation',30);
ylabel('分类准确率 (%)');
title('各方法分类准确率分布 (箱线图)');
yline(100/n_classes,'--r','随机水平','LineWidth',1.5);
grid on;
saveas(gcf,'Fig2_Accuracy_Boxplot.png');

%% ---------- 图3: 热力图 ----------
figure('Name','热力图','Position',[50 50 1050 600]);
imagesc(acc_all'*100); colormap(jet); colorbar; caxis([0 100]);
set(gca,'XTick',1:n_subjects,'XTickLabel',arrayfun(@(x) sprintf('S%02d',x),1:n_subjects,'Uni',0));
set(gca,'YTick',1:n_methods,'YTickLabel',method_names);
xlabel('被试'); ylabel('方法');
title('各被试×各方法 分类准确率 (%)');
for s = 1:n_subjects
    for m = 1:n_methods
        text(s,m,sprintf('%.1f',acc_all(s,m)*100),'HorizontalAlignment','center','FontSize',7);
    end
end
saveas(gcf,'Fig3_Heatmap.png');

%% ---------- 图4: 最佳方法混淆矩阵 ----------
[~,best_m] = max(mean(acc_all,1));
cm_total = zeros(n_classes);
for s = 1:n_subjects
    r = all_results(s).(method_names{best_m});
    if isfield(r,'confusion_matrix')
        cm_total = cm_total + r.confusion_matrix;
    end
end
cm_pct = cm_total ./ sum(cm_total,2) * 100;
clbl = {'A刺激','B刺激','C刺激'};

figure('Name','混淆矩阵','Position',[50 50 550 500]);
imagesc(cm_pct); colormap(flipud(hot)); colorbar; caxis([0 100]);
set(gca,'XTick',1:n_classes,'XTickLabel',clbl,'YTick',1:n_classes,'YTickLabel',clbl);
xlabel('预测类别'); ylabel('真实类别');
title(sprintf('混淆矩阵 - %s (所有被试汇总, %%)', method_names{best_m}));
for i = 1:n_classes
    for j = 1:n_classes
        text(j,i,sprintf('%.1f%%\n(%d)',cm_pct(i,j),cm_total(i,j)),...
            'HorizontalAlignment','center','FontSize',11);
    end
end
saveas(gcf,'Fig4_Confusion_Matrix.png');

%% ---------- 图5: F1分数 ----------
figure('Name','F1分数','Position',[50 50 900 450]);
bar(mean(f1_all,1),'FaceColor',[0.3 0.6 0.9]); hold on;
errorbar(1:n_methods, mean(f1_all,1), std(f1_all,0,1), 'k.','LineWidth',1.5);
set(gca,'XTick',1:n_methods,'XTickLabel',method_names,'XTickLabelRotation',30);
ylabel('宏平均 F1 分数'); ylim([0 1.1]);
title('各方法宏平均 F1 分数'); grid on;
saveas(gcf,'Fig5_F1_Scores.png');

%% 保存
save('SSVEP_Classification_Results.mat','all_results','acc_all','f1_all','method_names');
fprintf('\n所有结果已保存至 SSVEP_Classification_Results.mat\n');
fprintf('所有图形已保存为 PNG 文件\n');
fprintf('============================================================\n');
fprintf('                    分析完成!\n');
fprintf('============================================================\n');


%% #####################################################################
%%                    以下为所有内部函数
%% #####################################################################

%% =================== 小波统计量 ======================================
function s = wstats(c)
    energy = sum(c.^2);
    avg = mean(c);
    sd = std(c);
    p = (c.^2)/(sum(c.^2)+eps); p(p==0) = eps;
    ent = -sum(p.*log2(p));
    s = [energy, avg, sd, ent];
end

%% =================== 样本熵 ==========================================
function se = fn_sample_entropy(x, m, r)
    N = length(x);
    if N < m+2, se = 0; return; end
    count = zeros(1,2);
    for dim = m:m+1
        tmpl = zeros(N-dim, dim);
        for i = 1:N-dim, tmpl(i,:) = x(i:i+dim-1); end
        nm = 0;
        for i = 1:size(tmpl,1)
            for j = i+1:size(tmpl,1)
                if max(abs(tmpl(i,:)-tmpl(j,:))) < r, nm = nm+1; end
            end
        end
        count(dim-m+1) = nm;
    end
    if count(1)==0||count(2)==0, se = 0;
    else, se = -log(count(2)/count(1)); end
end

%% =================== 模糊熵 ==========================================
function fe = fn_fuzzy_entropy(x, m, r)
    N = length(x);
    if N < m+2, fe = 0; return; end
    phi = zeros(1,2);
    for dim = m:m+1
        tmpl = zeros(N-dim, dim);
        for i = 1:N-dim
            tmpl(i,:) = x(i:i+dim-1);
            tmpl(i,:) = tmpl(i,:) - mean(tmpl(i,:));
        end
        nt = size(tmpl,1); ss = 0;
        for i = 1:nt
            for j = i+1:nt
                d = max(abs(tmpl(i,:)-tmpl(j,:)));
                ss = ss + exp(-(d/r)^2);
            end
        end
        phi(dim-m+1) = ss/(nt*(nt-1)/2+eps);
    end
    if phi(1)==0, fe = 0;
    else, fe = -log(phi(2)/(phi(1)+eps)+eps); end
end

%% =================== 排列熵 ==========================================
function pe = fn_permutation_entropy(x, m, tau)
    N = length(x);
    np = N-(m-1)*tau;
    if np < 1, pe = 0; return; end
    patt = zeros(np, m);
    for i = 1:np
        idx = i:tau:i+(m-1)*tau;
        patt(i,:) = x(idx);
    end
    [~,pi_] = sort(patt,2);
    [~,~,ic] = unique(pi_,'rows');
    cnt = accumarray(ic,1);
    prb = cnt/sum(cnt);
    pe = -sum(prb.*log2(prb+eps))/log2(factorial(m));
end

%% =================== 谱熵 ============================================
function spe = fn_spectral_entropy(x, fs_in)
    nf = 2^nextpow2(length(x));
    [pxx,~] = pwelch(x,[],[],nf,fs_in);
    pn = pxx/(sum(pxx)+eps); pn(pn==0) = eps;
    spe = -sum(pn.*log2(pn))/log2(length(pxx));
end

%% =================== SVM分类 (多类 One-vs-One) =======================
function res = fn_classify_svm(features, labels, cv_idx, K, nc, kernel)
    all_pred = zeros(size(labels));
    fold_acc = zeros(K,1);
    cm = zeros(nc);

    for k = 1:K
        te = (cv_idx==k); tr = ~te;
        Xtr = features(tr,:); Ytr = labels(tr);
        Xte = features(te,:); Yte = labels(te);

        [Xtr,mu,sigma] = zscore(Xtr); sigma(sigma==0) = 1;
        Xte = (Xte-mu)./sigma;
        vf = var(Xtr)>0; Xtr = Xtr(:,vf); Xte = Xte(:,vf);

        if strcmp(kernel,'rbf')
            t_ = templateSVM('KernelFunction','rbf','Standardize',false,...
                'BoxConstraint',1,'KernelScale','auto');
        else
            t_ = templateSVM('KernelFunction','linear','Standardize',false,'BoxConstraint',1);
        end
        mdl = fitcecoc(Xtr,Ytr,'Learners',t_,'Coding','onevsone');
        pred = predict(mdl,Xte);
        all_pred(te) = pred; fold_acc(k) = mean(pred==Yte);
        for i = 1:length(Yte), cm(Yte(i),pred(i)) = cm(Yte(i),pred(i))+1; end
    end

    res.accuracy = mean(fold_acc); res.std_acc = std(fold_acc);
    res.fold_acc = fold_acc; res.confusion_matrix = cm;
    res.all_predictions = all_pred; res.all_labels = labels;
    [res.precision, res.recall, res.f1_score, res.macro_f1] = compute_metrics(cm, nc);
    fprintf('    ACC: %.2f%% (±%.2f%%)\n', res.accuracy*100, res.std_acc*100);
end

%% =================== HDCA 分类 =======================================
function res = fn_classify_hdca(EEG_ep, labels, cv_idx, K, nc, fs_in)
    [nch,ntp,~] = size(EEG_ep);
    ws = round(0.1*fs_in); st = round(0.05*fs_in);
    wstarts = 1:st:(ntp-ws+1); nw = length(wstarts);

    all_pred = zeros(size(labels)); fold_acc = zeros(K,1); cm = zeros(nc);

    for k = 1:K
        te = (cv_idx==k); tr = ~te;
        Xtr = EEG_ep(:,:,tr); Ytr = labels(tr);
        Xte = EEG_ep(:,:,te); Yte = labels(te);
        ntr = sum(tr); nte = sum(te);
        scores = zeros(nte,nc);

        for c = 1:nc
            Yb = double(Ytr==c);
            wfeat_tr = zeros(ntr,nw); wfeat_te = zeros(nte,nw);

            for w = 1:nw
                wr = wstarts(w):(wstarts(w)+ws-1);
                cm_tr = squeeze(mean(Xtr(:,wr,:),2));
                if size(cm_tr,2)~=ntr, cm_tr = cm_tr'; end

                mp = mean(cm_tr(:,Yb==1),2); mn = mean(cm_tr(:,Yb==0),2);
                Sw_ = cov(cm_tr(:,Yb==1)') + cov(cm_tr(:,Yb==0)') + 1e-6*eye(nch);
                ws_ = Sw_\(mp-mn);

                wfeat_tr(:,w) = (ws_'*cm_tr)';

                cm_te = squeeze(mean(Xte(:,wr,:),2));
                if size(cm_te,2)~=nte, cm_te = cm_te'; end
                wfeat_te(:,w) = (ws_'*cm_te)';
            end

            mp_t = mean(wfeat_tr(Yb==1,:),1); mn_t = mean(wfeat_tr(Yb==0,:),1);
            Sw_t = cov(wfeat_tr(Yb==1,:)) + cov(wfeat_tr(Yb==0,:)) + 1e-6*eye(nw);
            wt = Sw_t\(mp_t-mn_t)';
            scores(:,c) = wfeat_te*wt;
        end

        [~,pred] = max(scores,[],2);
        all_pred(te) = pred; fold_acc(k) = mean(pred==Yte);
        for i = 1:length(Yte), cm(Yte(i),pred(i)) = cm(Yte(i),pred(i))+1; end
    end

    res.accuracy = mean(fold_acc); res.std_acc = std(fold_acc);
    res.fold_acc = fold_acc; res.confusion_matrix = cm;
    res.all_predictions = all_pred; res.all_labels = labels;
    [res.precision,res.recall,res.f1_score,res.macro_f1] = compute_metrics(cm,nc);
    fprintf('    ACC: %.2f%% (±%.2f%%)\n', res.accuracy*100, res.std_acc*100);
end

%% =================== xDAWN + SVM =====================================
function res = fn_classify_xdawn(EEG_ep, labels, cv_idx, K, nc)
    [nch,ntp,~] = size(EEG_ep);
    ncomp = min(6,nch);
    all_pred = zeros(size(labels)); fold_acc = zeros(K,1); cm = zeros(nc);

    for k = 1:K
        te = (cv_idx==k); tr = ~te;
        Xtr = EEG_ep(:,:,tr); Ytr = labels(tr);
        Xte = EEG_ep(:,:,te); Yte = labels(te);
        ntr = sum(tr); nte = sum(te);

        Xc = reshape(Xtr,nch,[]);
        Ct = (Xc*Xc')/size(Xc,2) + 1e-6*eye(nch);

        filt = zeros(nch, ncomp*nc); ci = 0;
        for c = 1:nc
            avg_r = mean(Xtr(:,:,Ytr==c),3);
            Cs = (avg_r*avg_r')/ntp + 1e-6*eye(nch);
            [V,D] = eig(Cs,Ct);
            [~,si] = sort(diag(D),'descend'); V = V(:,si);
            for j = 1:ncomp, V(:,j) = V(:,j)/norm(V(:,j)); end
            filt(:,ci+1:ci+ncomp) = V(:,1:ncomp); ci = ci+ncomp;
        end

        ntf = size(filt,2);
        ftr = zeros(ntr,ntf*2); fte = zeros(nte,ntf*2);
        for t = 1:ntr, fl = filt'*Xtr(:,:,t); ftr(t,:) = [mean(fl,2)',var(fl,0,2)']; end
        for t = 1:nte, fl = filt'*Xte(:,:,t); fte(t,:) = [mean(fl,2)',var(fl,0,2)']; end

        [ftr,mu,sg] = zscore(ftr); sg(sg==0) = 1; fte = (fte-mu)./sg;
        t_ = templateSVM('KernelFunction','linear','Standardize',false,'BoxConstraint',1);
        mdl = fitcecoc(ftr,Ytr,'Learners',t_,'Coding','onevsone');
        pred = predict(mdl,fte);
        all_pred(te) = pred; fold_acc(k) = mean(pred==Yte);
        for i = 1:length(Yte), cm(Yte(i),pred(i)) = cm(Yte(i),pred(i))+1; end
    end

    res.accuracy = mean(fold_acc); res.std_acc = std(fold_acc);
    res.fold_acc = fold_acc; res.confusion_matrix = cm;
    res.all_predictions = all_pred; res.all_labels = labels;
    [res.precision,res.recall,res.f1_score,res.macro_f1] = compute_metrics(cm,nc);
    fprintf('    ACC: %.2f%% (±%.2f%%)\n', res.accuracy*100, res.std_acc*100);
end

%% =================== DCPM + SVM ======================================
function res = fn_classify_dcpm(EEG_ep, labels, cv_idx, K, nc)
    [nch,ntp,~] = size(EEG_ep);
    nf = min(4,nch);
    all_pred = zeros(size(labels)); fold_acc = zeros(K,1); cm = zeros(nc);

    for k = 1:K
        te = (cv_idx==k); tr = ~te;
        Xtr = EEG_ep(:,:,tr); Ytr = labels(tr);
        Xte = EEG_ep(:,:,te); Yte = labels(te);
        ntr = sum(tr); nte = sum(te);

        gm = mean(mean(Xtr,3),2);
        Sb = zeros(nch); Sw = zeros(nch);
        for c = 1:nc
            cd = Xtr(:,:,Ytr==c); n_c = size(cd,3);
            cmn = mean(mean(cd,3),2); d = cmn-gm;
            Sb = Sb + n_c*(d*d');
            for t = 1:n_c
                dw = mean(cd(:,:,t),2)-cmn;
                Sw = Sw + dw*dw';
            end
        end
        Sw = Sw + 1e-6*eye(nch);
        [W,D] = eig(Sb,Sw); [~,si] = sort(diag(D),'descend');
        W = W(:,si(1:nf));
        for f = 1:nf, W(:,f) = W(:,f)/norm(W(:,f)); end

        tmpl = zeros(nf,ntp,nc);
        for c = 1:nc, tmpl(:,:,c) = W'*mean(Xtr(:,:,Ytr==c),3); end

        ftr = zeros(ntr,nc*nf); fte = zeros(nte,nc*nf);
        for t = 1:ntr
            fl = W'*Xtr(:,:,t); fi = 0;
            for c = 1:nc, for f = 1:nf
                fi = fi+1; r = corrcoef(fl(f,:),tmpl(f,:,c)); ftr(t,fi) = r(1,2);
            end, end
        end
        for t = 1:nte
            fl = W'*Xte(:,:,t); fi = 0;
            for c = 1:nc, for f = 1:nf
                fi = fi+1; r = corrcoef(fl(f,:),tmpl(f,:,c)); fte(t,fi) = r(1,2);
            end, end
        end
        ftr(isnan(ftr)) = 0; fte(isnan(fte)) = 0;
        [ftr,mu,sg] = zscore(ftr); sg(sg==0) = 1; fte = (fte-mu)./sg;

        t_ = templateSVM('KernelFunction','linear','Standardize',false,'BoxConstraint',1);
        mdl = fitcecoc(ftr,Ytr,'Learners',t_,'Coding','onevsone');
        pred = predict(mdl,fte);
        all_pred(te) = pred; fold_acc(k) = mean(pred==Yte);
        for i = 1:length(Yte), cm(Yte(i),pred(i)) = cm(Yte(i),pred(i))+1; end
    end

    res.accuracy = mean(fold_acc); res.std_acc = std(fold_acc);
    res.fold_acc = fold_acc; res.confusion_matrix = cm;
    res.all_predictions = all_pred; res.all_labels = labels;
    [res.precision,res.recall,res.f1_score,res.macro_f1] = compute_metrics(cm,nc);
    fprintf('    ACC: %.2f%% (±%.2f%%)\n', res.accuracy*100, res.std_acc*100);
end

%% =================== TRCA ============================================
function res = fn_classify_trca(EEG_ep, labels, cv_idx, K, nc)
    [nch,~,~] = size(EEG_ep);
    ncomp = min(3,nch);
    all_pred = zeros(size(labels)); fold_acc = zeros(K,1); cm = zeros(nc);

    for k = 1:K
        te = (cv_idx==k); tr = ~te;
        Xtr = EEG_ep(:,:,tr); Ytr = labels(tr);
        Xte = EEG_ep(:,:,te); Yte = labels(te);
        nte = sum(te);

        Wc = cell(nc,1); tmpl = cell(nc,1);
        for c = 1:nc
            cd = Xtr(:,:,Ytr==c); n_c = size(cd,3);
            S = zeros(nch); Q = zeros(nch);
            for i = 1:n_c
                xi = cd(:,:,i)-mean(cd(:,:,i),2);
                Q = Q + xi*xi';
                for j = i+1:n_c
                    xj = cd(:,:,j)-mean(cd(:,:,j),2);
                    S = S + xi*xj' + xj*xi';
                end
            end
            Q = Q + 1e-6*eye(nch);
            [V,D] = eig(S,Q); [~,si] = sort(diag(real(D)),'descend');
            Wc{c} = real(V(:,si(1:ncomp)));
            tmpl{c} = mean(cd,3);
        end

        pred = zeros(nte,1);
        for t = 1:nte
            xt = Xte(:,:,t); sc = zeros(nc,1);
            for c = 1:nc
                w = Wc{c};
                yt = w'*xt; yr = w'*tmpl{c};
                rs = 0;
                for j = 1:ncomp, r = corrcoef(yt(j,:),yr(j,:)); rs = rs+r(1,2); end
                sc(c) = rs/ncomp;
            end
            [~,pred(t)] = max(sc);
        end

        all_pred(te) = pred; fold_acc(k) = mean(pred==Yte);
        for i = 1:length(Yte), cm(Yte(i),pred(i)) = cm(Yte(i),pred(i))+1; end
    end

    res.accuracy = mean(fold_acc); res.std_acc = std(fold_acc);
    res.fold_acc = fold_acc; res.confusion_matrix = cm;
    res.all_predictions = all_pred; res.all_labels = labels;
    [res.precision,res.recall,res.f1_score,res.macro_f1] = compute_metrics(cm,nc);
    fprintf('    ACC: %.2f%% (±%.2f%%)\n', res.accuracy*100, res.std_acc*100);
end

%% =================== eTRCA (集成TRCA) ================================
function res = fn_classify_etrca(EEG_ep, labels, cv_idx, K, nc)
    [nch,~,~] = size(EEG_ep);
    ncomp = min(3,nch);
    all_pred = zeros(size(labels)); fold_acc = zeros(K,1); cm = zeros(nc);

    for k = 1:K
        te = (cv_idx==k); tr = ~te;
        Xtr = EEG_ep(:,:,tr); Ytr = labels(tr);
        Xte = EEG_ep(:,:,te); Yte = labels(te);
        nte = sum(te);

        St = zeros(nch); Qt = zeros(nch);
        Wi = cell(nc,1); tmpl = cell(nc,1);

        for c = 1:nc
            cd = Xtr(:,:,Ytr==c); n_c = size(cd,3);
            Sc = zeros(nch); Qc = zeros(nch);
            for i = 1:n_c
                xi = cd(:,:,i)-mean(cd(:,:,i),2);
                Qc = Qc + xi*xi';
                for j = i+1:n_c
                    xj = cd(:,:,j)-mean(cd(:,:,j),2);
                    Sc = Sc + xi*xj' + xj*xi';
                end
            end
            St = St+Sc; Qt = Qt+Qc;
            Qc = Qc+1e-6*eye(nch);
            [V,D] = eig(Sc,Qc); [~,si] = sort(diag(real(D)),'descend');
            Wi{c} = real(V(:,si(1:ncomp)));
            tmpl{c} = mean(cd,3);
        end

        Qt = Qt+1e-6*eye(nch);
        [Ve,De] = eig(St,Qt); [~,si] = sort(diag(real(De)),'descend');
        We = real(Ve(:,si(1:ncomp)));

        pred = zeros(nte,1);
        for t = 1:nte
            xt = Xte(:,:,t); sc = zeros(nc,1);
            for c = 1:nc
                ye = We'*xt; yer = We'*tmpl{c};
                re = 0;
                for j = 1:ncomp, r = corrcoef(ye(j,:),yer(j,:)); re = re+r(1,2); end
                re = re/ncomp;

                yi = Wi{c}'*xt; yir = Wi{c}'*tmpl{c};
                ri = 0;
                for j = 1:ncomp, r = corrcoef(yi(j,:),yir(j,:)); ri = ri+r(1,2); end
                ri = ri/ncomp;

                sc(c) = atanh(re)+atanh(ri);
            end
            [~,pred(t)] = max(sc);
        end

        all_pred(te) = pred; fold_acc(k) = mean(pred==Yte);
        for i = 1:length(Yte), cm(Yte(i),pred(i)) = cm(Yte(i),pred(i))+1; end
    end

    res.accuracy = mean(fold_acc); res.std_acc = std(fold_acc);
    res.fold_acc = fold_acc; res.confusion_matrix = cm;
    res.all_predictions = all_pred; res.all_labels = labels;
    [res.precision,res.recall,res.f1_score,res.macro_f1] = compute_metrics(cm,nc);
    fprintf('    ACC: %.2f%% (±%.2f%%)\n', res.accuracy*100, res.std_acc*100);
end

%% =================== 多类别性能指标计算 ===============================
function [prec, rec, f1, macro_f1] = compute_metrics(cm, nc)
    prec = zeros(1,nc); rec = zeros(1,nc); f1 = zeros(1,nc);
    for c = 1:nc
        TP = cm(c,c);
        FP = sum(cm(:,c))-TP;
        FN = sum(cm(c,:))-TP;
        prec(c) = TP/(TP+FP+eps);
        rec(c)  = TP/(TP+FN+eps);
        f1(c)   = 2*prec(c)*rec(c)/(prec(c)+rec(c)+eps);
    end
    macro_f1 = mean(f1);
end
