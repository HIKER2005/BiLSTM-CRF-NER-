%% ========================================================================
%%  SSVEP脑电信号分类完整脚本 (单文件版, 全自动参数检测)
%%  利用9种算法对三种条件刺激(A/B/C)下10名被试的SSVEP信号进行分类
%%  数据格式: EEGLAB .set / .fdt 文件 (已预处理)
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

clear all; close all; clc;
rng(42);

%% ==================== 唯一需要修改的：数据路径 ========================

data_folder = 'D:\实验一数据\闪光实验一\预处理结束';  % <-- 修改为你的 .set/.fdt 文件所在路径

%% ==================== 以下全部自动检测，无需修改 ======================

% ---------- 自动扫描文件夹中所有 .set 文件 ----------
set_list = dir(fullfile(data_folder, '*.set'));
if isempty(set_list)
    error('在 %s 中未找到 .set 文件，请检查路径是否正确。', data_folder);
end
subject_files = {set_list.name};
n_subjects = length(subject_files);

fprintf('============================================================\n');
fprintf('  自动检测到 %d 个 .set 文件:\n', n_subjects);
for i = 1:n_subjects
    fprintf('    [%2d] %s\n', i, subject_files{i});
end
fprintf('============================================================\n');

% ---------- 加载第1个被试用于自动检测参数 ----------
fprintf('\n正在从第1个文件自动检测实验参数...\n');
EEG_probe = pop_loadset('filename', subject_files{1}, 'filepath', data_folder);

% 自动获取采样率
fs = EEG_probe.srate;
fprintf('  采样率: %d Hz\n', fs);

% 自动获取通道数
n_ch_detected = EEG_probe.nbchan;
fprintf('  通道数: %d\n', n_ch_detected);

% 自动检测是已分段数据还是连续数据
is_epoched = (EEG_probe.trials > 1);
if is_epoched
    fprintf('  数据类型: 已分段 (epoched), trials=%d\n', EEG_probe.trials);
    fprintf('  时间窗: [%.3f, %.3f] 秒\n', EEG_probe.xmin, EEG_probe.xmax);
else
    fprintf('  数据类型: 连续 (continuous), 总时长=%.1f 秒\n', EEG_probe.pnts/fs);
end

% 自动扫描所有 event 类型
all_event_types = [];
for e = 1:length(EEG_probe.event)
    evt = EEG_probe.event(e).type;
    if isnumeric(evt)
        all_event_types = [all_event_types; evt]; %#ok<AGROW>
    elseif ischar(evt) || isstring(evt)
        val = str2double(evt);
        if ~isnan(val)
            all_event_types = [all_event_types; val]; %#ok<AGROW>
        end
    end
end
unique_events = unique(all_event_types);
fprintf('  检测到的事件类型: [%s]\n', num2str(unique_events'));

% 识别三种刺激标记 (11, 21, 31)
stim_markers = [11, 21, 31];
found_markers = intersect(stim_markers, unique_events);
if length(found_markers) == 3
    fprintf('  已确认三种刺激标记: %d, %d, %d\n', found_markers);
    stim_markers = found_markers(:)';
else
    fprintf('  [警告] 未完整找到标记11/21/31, 实际找到: [%s]\n', num2str(found_markers'));
    fprintf('  将尝试使用事件中出现次数最多的3种标记作为刺激条件\n');
    event_counts = arrayfun(@(x) sum(all_event_types==x), unique_events);
    [~, sort_idx] = sort(event_counts, 'descend');
    stim_markers = sort(unique_events(sort_idx(1:min(3,length(sort_idx)))))';
    fprintf('  自动选取的刺激标记: [%s]\n', num2str(stim_markers));
end

n_classes = length(stim_markers);
stim_counts = arrayfun(@(x) sum(all_event_types==x), stim_markers);
for c = 1:n_classes
    fprintf('  标记 %d: %d 次\n', stim_markers(c), stim_counts(c));
end

% ---------- 确定分段方式和时间窗 ----------
if is_epoched
    epoch_time = [EEG_probe.xmin, EEG_probe.xmax];
else
    % 连续数据默认截取: 刺激前0.5秒 到 刺激后3秒
    epoch_time = [-0.5, 3.0];
    fprintf('  连续数据，使用默认分段时间窗: [%.1f, %.1f] 秒\n', epoch_time);
end

% ---------- 自动估算SSVEP刺激频率 (从数据PSD峰值) ----------
fprintf('\n正在从数据中自动估算SSVEP刺激频率...\n');
stim_freqs = fn_auto_detect_ssvep_freqs(EEG_probe, stim_markers, is_epoched, epoch_time, fs);
fprintf('  估算的SSVEP频率: [%s] Hz\n', num2str(stim_freqs, '%.1f '));

% ---------- 其余固定参数 ----------
n_harmonics   = 3;
K_fold        = 10;
wavelet_name  = 'db4';
wavelet_level = 5;

clear EEG_probe;

method_names = {'SVM_Linear','SVM_RBF','Wavelet_SVM','HDCA',...
                'xDAWN_SVM','DCPM_SVM','ENT_SVM','TRCA','eTRCA'};
n_methods = length(method_names);

acc_all = zeros(n_subjects, n_methods);
f1_all  = zeros(n_subjects, n_methods);
all_results = struct();
subject_names = cell(n_subjects, 1);

fprintf('\n============================================================\n');
fprintf('  开始分类分析\n');
fprintf('  被试数: %d   类别数: %d   方法数: %d   %d折交叉验证\n', ...
    n_subjects, n_classes, n_methods, K_fold);
fprintf('============================================================\n');

%% ====================== 主循环: 逐被试分析 ===========================
for sub = 1:n_subjects

    fprintf('\n==================== 被试 %d / %d ====================\n', sub, n_subjects);
    fprintf('  文件: %s\n', subject_files{sub});

    % 提取被试姓名用于显示
    [~, fname, ~] = fileparts(subject_files{sub});
    subject_names{sub} = strrep(fname, '_processed', '');

    %% ---------- 1. 加载 .set 文件 (已预处理, 不再做额外预处理) ----------
    EEG = pop_loadset('filename', subject_files{sub}, 'filepath', data_folder);
    n_ch = EEG.nbchan;
    fprintf('  采样率: %d Hz, 通道数: %d\n', EEG.srate, n_ch);

    %% ---------- 2. 提取分段 ----------
    if EEG.trials > 1
        % ====== 数据已是分段格式 ======
        fprintf('  数据已分段, trials=%d\n', EEG.trials);
        n_epoch_pts = EEG.pnts;

        % 从每个epoch的event中提取标签
        n_total_epochs = EEG.trials;
        labels = zeros(n_total_epochs, 1);

        for t = 1:n_total_epochs
            epoch_events = [];
            for e = 1:length(EEG.event)
                if EEG.event(e).epoch == t
                    evt = EEG.event(e).type;
                    if isnumeric(evt), mk = evt;
                    elseif ischar(evt)||isstring(evt), mk = str2double(evt);
                    else, continue; end
                    if ~isnan(mk) && ismember(mk, stim_markers)
                        epoch_events = [epoch_events, mk]; %#ok<AGROW>
                    end
                end
            end
            if ~isempty(epoch_events)
                for c = 1:n_classes
                    if any(epoch_events == stim_markers(c))
                        labels(t) = c;
                        break;
                    end
                end
            end
        end

        valid = (labels > 0);
        EEG_epochs = double(EEG.data(:,:,valid));
        labels = labels(valid);
        n_trials = length(labels);

    else
        % ====== 数据是连续格式，手动分段 ======
        fprintf('  连续数据, 手动分段...\n');
        epoch_samples = round(epoch_time * fs);
        n_epoch_pts = epoch_samples(2) - epoch_samples(1) + 1;

        stim_events = [];
        for e = 1:length(EEG.event)
            evt = EEG.event(e).type;
            if isnumeric(evt), mk = evt;
            elseif ischar(evt)||isstring(evt), mk = str2double(evt);
            else, continue; end
            if ~isnan(mk) && ismember(mk, stim_markers)
                stim_events = [stim_events; mk, round(EEG.event(e).latency)]; %#ok<AGROW>
            end
        end

        n_trials_raw = size(stim_events, 1);
        EEG_epochs = zeros(n_ch, n_epoch_pts, n_trials_raw);
        labels = zeros(n_trials_raw, 1);
        valid = true(n_trials_raw, 1);

        for t = 1:n_trials_raw
            s1 = stim_events(t,2) + epoch_samples(1);
            s2 = stim_events(t,2) + epoch_samples(2);
            if s1 < 1 || s2 > size(EEG.data, 2)
                valid(t) = false; continue;
            end
            EEG_epochs(:,:,t) = double(EEG.data(:, s1:s2));
            for c = 1:n_classes
                if stim_events(t,1) == stim_markers(c)
                    labels(t) = c; break;
                end
            end
        end

        EEG_epochs = EEG_epochs(:,:,valid);
        labels = labels(valid);
        n_trials = sum(valid);
    end

    fprintf('  有效试次: %d', n_trials);
    for c = 1:n_classes
        fprintf(', 标记%d=%d', stim_markers(c), sum(labels==c));
    end
    fprintf('\n');

    if n_trials < K_fold * n_classes
        fprintf('  [警告] 试次数过少，跳过该被试\n');
        continue;
    end

    %% ---------- 3. 数据已预处理, 仅确保为double ----------
    EEG_epochs = double(EEG_epochs);
    clear EEG;

    %% ---------- 4. 交叉验证分折 ----------
    cv_idx = crossvalind('Kfold', labels, K_fold);

    %% ---------- 5. 特征提取 ----------
    fprintf('  提取特征...\n');
    n_epoch_pts = size(EEG_epochs, 2);

    % --- PSD特征 ---
    freqs_interest = [];
    for c = 1:n_classes
        for h = 1:n_harmonics
            f_h = stim_freqs(c) * h;
            if f_h < fs/2
                freqs_interest = [freqs_interest, f_h]; %#ok<AGROW>
            end
        end
    end
    freqs_interest = unique(freqs_interest);
    nfft_psd = 2^nextpow2(n_epoch_pts);
    bw = 1;
    feat_psd = zeros(n_trials, n_ch * length(freqs_interest));
    for t = 1:n_trials
        fi = 0;
        for ch = 1:n_ch
            sig = detrend(double(EEG_epochs(ch,:,t)));
            [pxx, f_psd] = pwelch(sig,[],[],nfft_psd,fs);
            for fq = 1:length(freqs_interest)
                fi = fi + 1;
                idx_f = (f_psd >= freqs_interest(fq)-bw) & (f_psd <= freqs_interest(fq)+bw);
                feat_psd(t,fi) = mean(pxx(idx_f));
            end
        end
    end
    feat_psd = log10(feat_psd + eps);

    % --- 小波特征 ---
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

    % --- 熵特征 ---
    % 为加速，对信号进行降采样后计算熵
    ds_factor = max(1, round(n_epoch_pts / 200));
    feat_ent = zeros(n_trials, n_ch*4);
    for t = 1:n_trials
        vec = [];
        for ch = 1:n_ch
            sig = double(EEG_epochs(ch, 1:ds_factor:end, t));
            r_thr = 0.2 * std(sig);
            se  = fn_sample_entropy(sig, 2, r_thr);
            fe  = fn_fuzzy_entropy(sig, 2, r_thr);
            pe  = fn_permutation_entropy(sig, 3, 1);
            spe = fn_spectral_entropy(sig, round(fs/ds_factor));
            vec = [vec, se, fe, pe, spe]; %#ok<AGROW>
        end
        feat_ent(t,:) = vec;
    end
    feat_ent(isnan(feat_ent)) = 0;
    feat_ent(isinf(feat_ent)) = 0;

    %% ---------- 6. 运行9种分类方法 ----------

    fprintf('  [1/9] SVM (线性核)...\n');
    res = fn_classify_svm(feat_psd, labels, cv_idx, K_fold, n_classes, 'linear');
    all_results(sub).SVM_Linear = res;
    acc_all(sub,1) = res.accuracy; f1_all(sub,1) = res.macro_f1;

    fprintf('  [2/9] SVM (RBF核)...\n');
    res = fn_classify_svm(feat_psd, labels, cv_idx, K_fold, n_classes, 'rbf');
    all_results(sub).SVM_RBF = res;
    acc_all(sub,2) = res.accuracy; f1_all(sub,2) = res.macro_f1;

    fprintf('  [3/9] Wavelet+SVM...\n');
    res = fn_classify_svm(feat_wav, labels, cv_idx, K_fold, n_classes, 'linear');
    all_results(sub).Wavelet_SVM = res;
    acc_all(sub,3) = res.accuracy; f1_all(sub,3) = res.macro_f1;

    fprintf('  [4/9] HDCA...\n');
    res = fn_classify_hdca(EEG_epochs, labels, cv_idx, K_fold, n_classes, fs);
    all_results(sub).HDCA = res;
    acc_all(sub,4) = res.accuracy; f1_all(sub,4) = res.macro_f1;

    fprintf('  [5/9] xDAWN+SVM...\n');
    res = fn_classify_xdawn(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).xDAWN_SVM = res;
    acc_all(sub,5) = res.accuracy; f1_all(sub,5) = res.macro_f1;

    fprintf('  [6/9] DCPM+SVM...\n');
    res = fn_classify_dcpm(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).DCPM_SVM = res;
    acc_all(sub,6) = res.accuracy; f1_all(sub,6) = res.macro_f1;

    fprintf('  [7/9] ENT+SVM...\n');
    res = fn_classify_svm(feat_ent, labels, cv_idx, K_fold, n_classes, 'linear');
    all_results(sub).ENT_SVM = res;
    acc_all(sub,7) = res.accuracy; f1_all(sub,7) = res.macro_f1;

    fprintf('  [8/9] TRCA...\n');
    res = fn_classify_trca(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).TRCA = res;
    acc_all(sub,8) = res.accuracy; f1_all(sub,8) = res.macro_f1;

    fprintf('  [9/9] eTRCA...\n');
    res = fn_classify_etrca(EEG_epochs, labels, cv_idx, K_fold, n_classes);
    all_results(sub).eTRCA = res;
    acc_all(sub,9) = res.accuracy; f1_all(sub,9) = res.macro_f1;

end

%% ====================== 结果汇总与可视化 =============================
% 找出实际完成分析的被试 (跳过试次不足的)
valid_subs = find(any(acc_all > 0, 2));
n_valid = length(valid_subs);
acc_valid = acc_all(valid_subs, :);
f1_valid  = f1_all(valid_subs, :);

fprintf('\n\n');
fprintf('================================================================\n');
fprintf('                    分 类 结 果 汇 总\n');
fprintf('================================================================\n');
fprintf('%-18s %-10s %-10s %-10s %-10s %-10s\n', ...
    '方法','平均ACC%','标准差%','最高%','最低%','平均F1');
fprintf('%s\n', repmat('-',1,68));
for m = 1:n_methods
    a = acc_valid(:,m)*100;
    fprintf('%-18s %-10.2f %-10.2f %-10.2f %-10.2f %-10.4f\n', ...
        method_names{m}, mean(a), std(a), max(a), min(a), mean(f1_valid(:,m)));
end
fprintf('%s\n', repmat('=',1,68));

fprintf('\n各被试分类准确率 (%%):\n');
fprintf('%-15s','被试');
for m = 1:n_methods, fprintf('%-13s', method_names{m}); end
fprintf('\n%s\n', repmat('-',1,15+13*n_methods));
for idx = 1:n_valid
    s = valid_subs(idx);
    fprintf('%-15s', subject_names{s});
    for m = 1:n_methods
        fprintf('%-13.2f', acc_all(s,m)*100);
    end
    fprintf('\n');
end

% 方法间配对 t 检验
if n_valid >= 3
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
                [~,p] = ttest(acc_valid(:,m1), acc_valid(:,m2));
                if isnan(p), fprintf('%-12s','NaN');
                elseif p < 0.05, fprintf('%-12s', sprintf('%.4f*',p));
                else, fprintf('%-12.4f', p); end
            end
        end
        fprintf('\n');
    end
end

%% ---------- 图1: 平均准确率柱状图 ----------
figure('Name','平均分类准确率','Position',[50 50 950 500]);
mn = mean(acc_valid,1)*100; sd = std(acc_valid,0,1)*100;
bh = bar(mn,'FaceColor','flat'); hold on;
errorbar(1:n_methods, mn, sd, 'k.','LineWidth',1.5);
colors = lines(n_methods);
for m = 1:n_methods, bh.CData(m,:) = colors(m,:); end
set(gca,'XTick',1:n_methods,'XTickLabel',method_names,'XTickLabelRotation',30,'FontSize',10);
ylabel('分类准确率 (%)'); ylim([0 105]);
title(sprintf('各方法在%d名被试上的平均分类准确率', n_valid));
yline(100/n_classes,'--r','随机水平','LineWidth',1.5);
grid on;
saveas(gcf,'Fig1_Average_Accuracy.png');
saveas(gcf,'Fig1_Average_Accuracy.fig');

%% ---------- 图2: 箱线图 ----------
figure('Name','准确率分布','Position',[50 50 950 500]);
boxplot(acc_valid*100,'Labels',method_names);
set(gca,'XTickLabelRotation',30,'FontSize',10);
ylabel('分类准确率 (%)');
title('各方法分类准确率分布 (箱线图)');
yline(100/n_classes,'--r','随机水平','LineWidth',1.5);
grid on;
saveas(gcf,'Fig2_Accuracy_Boxplot.png');
saveas(gcf,'Fig2_Accuracy_Boxplot.fig');

%% ---------- 图3: 热力图 ----------
figure('Name','热力图','Position',[50 50 1100 600]);
imagesc(acc_valid'*100); colormap(jet); cb = colorbar; ylabel(cb,'准确率 (%)');
caxis([max(0, min(acc_valid(:))*100-5), min(100, max(acc_valid(:))*100+5)]);
set(gca,'XTick',1:n_valid,'XTickLabel',subject_names(valid_subs),'XTickLabelRotation',30);
set(gca,'YTick',1:n_methods,'YTickLabel',method_names,'FontSize',9);
xlabel('被试'); ylabel('方法');
title('各被试 × 各方法 分类准确率 (%)');
for si = 1:n_valid
    for m = 1:n_methods
        text(si,m,sprintf('%.1f',acc_valid(si,m)*100),...
            'HorizontalAlignment','center','FontSize',7,'FontWeight','bold');
    end
end
saveas(gcf,'Fig3_Heatmap.png');
saveas(gcf,'Fig3_Heatmap.fig');

%% ---------- 图4: 最佳方法混淆矩阵 ----------
[~,best_m] = max(mean(acc_valid,1));
cm_total = zeros(n_classes);
for idx = 1:n_valid
    s = valid_subs(idx);
    r = all_results(s).(method_names{best_m});
    if isfield(r,'confusion_matrix')
        cm_total = cm_total + r.confusion_matrix;
    end
end
cm_pct = cm_total ./ (sum(cm_total,2) + eps) * 100;
clbl = cell(1, n_classes);
stim_labels_map = containers.Map([11,21,31], {'A刺激','B刺激','C刺激'});
for c = 1:n_classes
    if stim_labels_map.isKey(stim_markers(c))
        clbl{c} = stim_labels_map(stim_markers(c));
    else
        clbl{c} = sprintf('类别%d (标记%d)', c, stim_markers(c));
    end
end

figure('Name','混淆矩阵','Position',[50 50 600 520]);
imagesc(cm_pct); colormap(flipud(hot)); colorbar; caxis([0 100]);
set(gca,'XTick',1:n_classes,'XTickLabel',clbl,'YTick',1:n_classes,'YTickLabel',clbl,'FontSize',11);
xlabel('预测类别'); ylabel('真实类别');
title(sprintf('混淆矩阵 - %s (所有被试汇总, %%)', method_names{best_m}));
for i = 1:n_classes
    for j = 1:n_classes
        text(j,i,sprintf('%.1f%%\n(%d)',cm_pct(i,j),cm_total(i,j)),...
            'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');
    end
end
saveas(gcf,'Fig4_Confusion_Matrix.png');
saveas(gcf,'Fig4_Confusion_Matrix.fig');

%% ---------- 图5: F1分数 ----------
figure('Name','F1分数','Position',[50 50 950 450]);
bar(mean(f1_valid,1),'FaceColor',[0.3 0.6 0.9]); hold on;
errorbar(1:n_methods, mean(f1_valid,1), std(f1_valid,0,1), 'k.','LineWidth',1.5);
set(gca,'XTick',1:n_methods,'XTickLabel',method_names,'XTickLabelRotation',30,'FontSize',10);
ylabel('宏平均 F1 分数'); ylim([0 1.1]);
title('各方法宏平均 F1 分数'); grid on;
saveas(gcf,'Fig5_F1_Scores.png');
saveas(gcf,'Fig5_F1_Scores.fig');

%% 保存
save('SSVEP_Classification_Results.mat', ...
    'all_results','acc_all','f1_all','method_names','subject_names', ...
    'stim_markers','stim_freqs','fs','epoch_time','n_classes');
fprintf('\n所有结果已保存至 SSVEP_Classification_Results.mat\n');
fprintf('所有图形已保存为 PNG + FIG 文件\n');
fprintf('============================================================\n');
fprintf('                    分析完成!\n');
fprintf('============================================================\n');


%% #####################################################################
%%                    以下为所有内部函数
%% #####################################################################

%% =================== 自动检测SSVEP频率 ===============================
function stim_freqs = fn_auto_detect_ssvep_freqs(EEG, stim_markers, is_epoched, epoch_time, fs)
    n_classes = length(stim_markers);
    stim_freqs = zeros(1, n_classes);
    nfft = 1024;
    search_range = [4 50]; % 在4-50Hz范围内搜索SSVEP峰值

    for c = 1:n_classes
        if is_epoched
            % 已分段数据: 找出属于该标记的epoch
            epoch_idx = [];
            for e = 1:length(EEG.event)
                evt = EEG.event(e).type;
                if isnumeric(evt), mk = evt;
                elseif ischar(evt)||isstring(evt), mk = str2double(evt);
                else, continue; end
                if mk == stim_markers(c) && isfield(EEG.event(e),'epoch')
                    epoch_idx = [epoch_idx, EEG.event(e).epoch]; %#ok<AGROW>
                end
            end
            epoch_idx = unique(epoch_idx);
            if isempty(epoch_idx), stim_freqs(c) = 10*c; continue; end

            % 计算平均PSD (使用枕区通道或所有通道)
            avg_psd = zeros(nfft/2+1, 1);
            n_avg = 0;
            for t = epoch_idx
                if t <= EEG.trials
                    for ch = 1:EEG.nbchan
                        sig = double(EEG.data(ch,:,t));
                        [pxx, f] = pwelch(detrend(sig),[],[],nfft,fs);
                        avg_psd = avg_psd + pxx;
                        n_avg = n_avg + 1;
                    end
                end
            end
        else
            % 连续数据: 截取该标记周围的信号段
            avg_psd = zeros(nfft/2+1, 1);
            n_avg = 0;
            ep_samp = round(epoch_time * fs);
            for e = 1:length(EEG.event)
                evt = EEG.event(e).type;
                if isnumeric(evt), mk = evt;
                elseif ischar(evt)||isstring(evt), mk = str2double(evt);
                else, continue; end
                if mk == stim_markers(c)
                    lat = round(EEG.event(e).latency);
                    s1 = lat + ep_samp(1); s2 = lat + ep_samp(2);
                    if s1 >= 1 && s2 <= size(EEG.data,2)
                        for ch = 1:EEG.nbchan
                            sig = double(EEG.data(ch, s1:s2));
                            [pxx, f] = pwelch(detrend(sig),[],[],nfft,fs);
                            avg_psd = avg_psd + pxx;
                            n_avg = n_avg + 1;
                        end
                    end
                end
            end
        end

        if n_avg > 0
            avg_psd = avg_psd / n_avg;
        end

        % 在搜索范围内找到最大峰值
        valid_f = (f >= search_range(1)) & (f <= search_range(2));
        f_valid = f(valid_f);
        psd_valid = avg_psd(valid_f);

        [~, peak_idx] = max(psd_valid);
        stim_freqs(c) = round(f_valid(peak_idx) * 2) / 2; % 四舍五入到0.5Hz
    end

    % 如果检测到重复频率或异常值，回退到常见SSVEP频率
    if length(unique(stim_freqs)) < n_classes || any(stim_freqs < 4)
        fprintf('  [提示] 自动频率检测不够可靠，使用常见默认值\n');
        default_freqs = [8, 10, 12, 15, 20, 25];
        stim_freqs = default_freqs(1:n_classes);
    end
end

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
