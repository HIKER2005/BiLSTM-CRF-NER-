%% ========================================================================
%% EEG ERP 分析脚本 —— 视觉搜索任务（三种刺激条件 × 有目标 × 正确反应）
%% ========================================================================
%
%  实验设计：
%    标记含义：
%      11 = A刺激         21 = B刺激         31 = C刺激
%      41 = 视觉搜索任务出现（有目标）
%      42 = 视觉搜索任务出现（无目标）
%      12 = 回答正确       22 = 回答错误       32 = 无反应
%
%  提取条件（正确反应）：
%    条件1 (Cond1): 刺激类型11 + 有目标41 + 反应正确12
%    条件2 (Cond2): 刺激类型21 + 有目标41 + 反应正确12
%    条件3 (Cond3): 刺激类型31 + 有目标41 + 反应正确12
%
%  可选对比条件（错误反应）：
%    条件4 (Cond4): 刺激类型11 + 有目标41 + 反应错误22
%    条件5 (Cond5): 刺激类型21 + 有目标41 + 反应错误22
%    条件6 (Cond6): 刺激类型31 + 有目标41 + 反应错误22
%
%  分析重点：以41标记（有目标出现）的时间点为 0 点，分析 -0.1s 到 0.8s 的 ERP
%
%  复合标记编码方案（脚本自动生成）：
%    101 = A刺激 + 有目标 + 正确     201 = B刺激 + 有目标 + 正确     301 = C刺激 + 有目标 + 正确
%    102 = A刺激 + 有目标 + 错误     202 = B刺激 + 有目标 + 错误     302 = C刺激 + 有目标 + 错误
%    103 = A刺激 + 有目标 + 无反应   203 = B刺激 + 有目标 + 无反应   303 = C刺激 + 有目标 + 无反应
%% ========================================================================

clear; clc;
script_version = 'v6_2026-02-11';
fprintf('\n====== ERP分析脚本 版本: %s ======\n', script_version);
fprintf('如果版本号不是 %s，请重新下载脚本！\n\n', script_version);
eeglab;

%% ========================= 参数设置（需要修改的部分）=========================
file_path = 'D:\实验一数据\闪烁光实验一\预处理结束\'; %%% 数据文件所在路径，修改为你的实际路径
save_figure_dir = 'C:\Users\wangw\Desktop\ERP-视觉搜索阶段-v6\v6_中性电极成分';

%%% =================== 被试文件定义 ===================
%%% 方式1（推荐）：自动读取目录下所有 .set 文件作为被试
%%% 脚本会自动扫描 file_path 中的所有 .set 文件，每个文件视为一个被试
auto_scan = true;  % true=自动扫描目录, false=手动指定文件名列表

%%% 方式2：手动列出每个被试的文件名（当 auto_scan = false 时使用）
%%% 取消下面的注释并填入你的文件名
% SubjFiles = {
%     'denghaowei_processed.set'
%     'houyuejiao_processed.set'
%     'huangjingjie_processed.set'
%     'jiangyan_processed.set'
%     'jinpengyu_processed.set'
%     'jiqiang_processed.set'
%     'qiuyuanrong_processed.set'
%     'niuhuan_processed.set'
%     'wanghongqi_processed.set'
%     'zhouxiangling_processed.set'
% };
%% ========== 扫描/确认被试文件列表 ==========
fprintf('\n====== 数据目录: %s ======\n', file_path);
if ~exist(file_path, 'dir')
    error('错误：目录不存在: %s\n请检查 file_path 设置。', file_path);
end

if auto_scan
    set_files = dir(fullfile(file_path, '*.set'));
    if isempty(set_files)
        error('错误：目录 %s 中没有找到任何 .set 文件！', file_path);
    end
    SubjFiles = {set_files.name}';
end

nSubj = length(SubjFiles);
fprintf('共找到 %d 个被试文件:\n', nSubj);
for fi = 1:nSubj
    fprintf('  被试%2d: %s\n', fi, SubjFiles{fi});
end
fprintf('\n');

Subj = 1:nSubj;

% 分段参数
epoch_window = [-0.1 0.8];             % 分段时间窗，单位：秒（-100ms 到 800ms）
baseline_window = [-100 0];             % 基线校正窗口，单位：毫秒

% ====== ERP 波形平滑参数 ======
apply_lowpass = true;                   %%% 是否对平均后的波形做低通滤波
lowpass_cutoff = 20;                    %%% 低通滤波截止频率(Hz)，推荐 20~30Hz
lowpass_order = 4;                      %%% Butterworth 滤波器阶数（通常 2~6）

% 原始事件标记
stim_markers = [11, 21, 31];            % A/B/C 三种刺激类型标记
target_marker = 41;                     % 有目标视觉搜索标记
no_target_marker = 42;                  % 无目标视觉搜索标记
correct_marker = 12;                    % 正确反应标记
incorrect_marker = 22;                  % 错误反应标记
no_response_marker = 32;               % 无反应标记
response_markers = [correct_marker, incorrect_marker, no_response_marker];

% 复合标记编码（自动生成，无需修改）
composite_correct   = [101, 201, 301];
composite_incorrect = [102, 202, 302];
composite_noresp    = [103, 203, 303];

% 分析条件定义
Cond_markers = composite_correct;
Cond_names = {'A_correct', 'B_correct', 'C_correct'};
nCond = length(Cond_markers);

% —— 是否同时提取错误反应条件 ——
include_error = true;                   %%% 设为 true 则同时提取错误反应条件
if include_error
    Cond_markers = [composite_correct, composite_incorrect];
    Cond_names = {'A_correct','B_correct','C_correct','A_incorrect','B_incorrect','C_incorrect'};
    nCond = length(Cond_markers);
end

% 感兴趣电极列表
chans_of_interest = {'POz', 'Oz', 'FCz', 'Fz', 'Fpz', 'Pz', 'Cz',  ...
                    };
chan_indices = [];

% 兼容旧变量名
chan_of_interest = chans_of_interest{1};
chan_idx = [];

%% ========================= Part 1: 复合标记创建与分段 =========================
fprintf('\n====== 开始处理数据 ======\n');

for i = 1:nSubj
    file_name = SubjFiles{i};

    full_path = fullfile(file_path, file_name);
    if ~exist(full_path, 'file')
        warning('文件不存在，跳过: %s', full_path);
        continue;
    end

    fprintf('\n--- 正在处理被试 %d/%d: %s ---\n', i, nSubj, file_name);

    EEG = pop_loadset('filename', file_name, 'filepath', file_path);

    if isempty(chan_indices)
        fprintf('查找感兴趣电极:\n');
        all_labels = {EEG.chanlocs.labels};
        chan_indices = zeros(1, length(chans_of_interest));
        for ci = 1:length(chans_of_interest)
            found = find(strcmpi(all_labels, chans_of_interest{ci}));
            if ~isempty(found)
                chan_indices(ci) = found(1);
                fprintf('  %s -> 索引 %d\n', chans_of_interest{ci}, chan_indices(ci));
            else
                warning('未找到电极 %s！', chans_of_interest{ci});
                chan_indices(ci) = 0;
            end
        end
        valid = chan_indices > 0;
        chans_of_interest = chans_of_interest(valid);
        chan_indices = chan_indices(valid);
        if isempty(chan_indices)
            error('没有找到任何指定的电极，请检查 chans_of_interest 中的电极名称！');
        end
        fprintf('共找到 %d 个有效电极\n', length(chan_indices));
        chan_of_interest = chans_of_interest{1};
        chan_idx = chan_indices(1);
    end

    %% ---- 判断数据类型：连续 or 已分段 ----
    is_epoched = (ndims(EEG.data) == 3);

    if ~is_epoched
        %% ============ 情况A: 连续数据 ============
        fprintf('  检测到连续数据，正在创建复合标记并分段...\n');

        nevents = length(EEG.event);
        trial_count = zeros(1, 6);

        for ev = 1:nevents
            current_type = EEG.event(ev).type;
            if ischar(current_type) || isstring(current_type)
                current_type_num = str2double(current_type);
            else
                current_type_num = current_type;
            end

            if current_type_num == target_marker
                % 向前查找刺激类型
                stim_type = NaN;
                for prev_ev = (ev - 1):-1:1
                    pt = EEG.event(prev_ev).type;
                    if ischar(pt)||isstring(pt), pt = str2double(pt); end
                    if ismember(pt, stim_markers), stim_type = pt; break; end
                    if ismember(pt, [target_marker, no_target_marker]), break; end
                end
                % 向后查找反应类型
                resp_type = NaN;
                for next_ev = (ev + 1):nevents
                    nt = EEG.event(next_ev).type;
                    if ischar(nt)||isstring(nt), nt = str2double(nt); end
                    if ismember(nt, response_markers), resp_type = nt; break; end
                    if ismember(nt, [stim_markers, target_marker, no_target_marker]), break; end
                end
                % 创建复合标记
                if ~isnan(stim_type)
                    stim_code = round(stim_type / 10);
                    if resp_type == correct_marker, resp_code = 1;
                    elseif resp_type == incorrect_marker, resp_code = 2;
                    elseif resp_type == no_response_marker, resp_code = 3;
                    else, resp_code = 0; end
                    if resp_code > 0
                        EEG.event(ev).type = stim_code * 100 + resp_code;
                        if stim_code <= 3 && resp_code <= 2
                            trial_count((stim_code-1)*2 + resp_code) = trial_count((stim_code-1)*2 + resp_code) + 1;
                        end
                    end
                end
            end
        end

        fprintf('  被试 %d (%s) 试次统计:\n', i, file_name);
        fprintf('    A正确: %d, B正确: %d, C正确: %d\n', trial_count(1), trial_count(3), trial_count(5));
        fprintf('    A错误: %d, B错误: %d, C错误: %d\n', trial_count(2), trial_count(4), trial_count(6));

        EEG = pop_epoch(EEG, num2cell(Cond_markers), epoch_window, ...
            'newname', [file_name '_epoched'], 'epochinfo', 'yes');
        EEG = pop_rmbase(EEG, baseline_window);
        fprintf('  分段完成，共 %d 个 epoch\n', EEG.trials);

        for j = 1:nCond
            EEG_temp = pop_selectevent(EEG, 'type', Cond_markers(j), ...
                'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
            fprintf('  条件 %s (标记 %d): %d 个 epoch\n', Cond_names{j}, Cond_markers(j), EEG_temp.trials);
            if EEG_temp.trials > 0
                data(i, j, :, :) = squeeze(mean(EEG_temp.data, 3));
            else
                warning('  被试 %d 条件 %s 无有效 epoch！', i, Cond_names{j});
                data(i, j, :, :) = zeros(EEG.nbchan, EEG.pnts);
            end
        end

    else
        %% ============ 情况B: 已分段数据 ============
        fprintf('  检测到已分段数据（%d trials），正在分析事件结构...\n', EEG.trials);

        time_lock_types = zeros(1, EEG.trials);

        for ep = 1:EEG.trials
            ep_types = EEG.epoch(ep).eventtype;
            ep_lats  = EEG.epoch(ep).eventlatency;

            if iscell(ep_lats)
                lats = cellfun(@(x) double(x), ep_lats);
            else
                lats = double(ep_lats);
            end

            [~, zero_idx] = min(abs(lats));
            if iscell(ep_types)
                tl = ep_types{zero_idx};
            else
                tl = ep_types(zero_idx);
            end
            if ischar(tl) || isstring(tl), tl = str2double(tl); end
            time_lock_types(ep) = tl;
        end

        unique_tl = unique(time_lock_types);
        fprintf('  锁时事件类型统计:\n');
        for ut = 1:length(unique_tl)
            fprintf('    标记 %g: %d 个 epoch\n', unique_tl(ut), sum(time_lock_types == unique_tl(ut)));
        end

        first41 = find(time_lock_types == target_marker, 1);
        if ~isempty(first41)
            fprintf('  [调试] 第一个41-epoch (ep=%d) 内的事件:\n', first41);
            ep_types = EEG.epoch(first41).eventtype;
            ep_lats  = EEG.epoch(first41).eventlatency;
            nev_debug = length(ep_types);
            for kk = 1:nev_debug
                if iscell(ep_types), tt = ep_types{kk}; else, tt = ep_types(kk); end
                if iscell(ep_lats),  ll = ep_lats{kk};  else, ll = ep_lats(kk);  end
                fprintf('    事件: type=%s, latency=%.1fms\n', num2str(tt), ll);
            end
        end

        epoch_condition = zeros(1, EEG.trials);
        debug_no_stim = 0;
        debug_no_resp = 0;

        for ep = 1:EEG.trials
            is_target = (time_lock_types(ep) == target_marker);
            is_notarget = (time_lock_types(ep) == no_target_marker);
            if ~is_target && ~is_notarget
                continue;
            end

            ep_types = EEG.epoch(ep).eventtype;
            ep_lats  = EEG.epoch(ep).eventlatency;
            nev = length(ep_types);

            types_num = nan(1, nev);
            lats_num  = nan(1, nev);
            for k = 1:nev
                if iscell(ep_types), t = ep_types{k}; else, t = ep_types(k); end
                if ischar(t) || isstring(t), t = str2double(t); end
                types_num(k) = t;
                if iscell(ep_lats), lats_num(k) = double(ep_lats{k}); else, lats_num(k) = double(ep_lats(k)); end
            end

            resp_code = 0;
            resp_events_idx = find(lats_num > 0);
            for ri = 1:length(resp_events_idx)
                k = resp_events_idx(ri);
                if types_num(k) == correct_marker,     resp_code = 1; break; end
                if types_num(k) == incorrect_marker,    resp_code = 2; break; end
                if types_num(k) == no_response_marker,  resp_code = 3; break; end
            end

            if resp_code == 0
                for next_ep = (ep + 1):min(EEG.trials, ep + 5)
                    tl_next = time_lock_types(next_ep);
                    if tl_next == correct_marker,     resp_code = 1; break; end
                    if tl_next == incorrect_marker,    resp_code = 2; break; end
                    if tl_next == no_response_marker,  resp_code = 3; break; end
                    if ismember(tl_next, [stim_markers, target_marker, no_target_marker])
                        break;
                    end
                end
            end

            stim_code = 0;

            stim_events_idx = find(lats_num < 0);
            for si_k = 1:length(stim_events_idx)
                k = stim_events_idx(si_k);
                if ismember(types_num(k), stim_markers)
                    stim_code = round(types_num(k) / 10);
                    break;
                end
            end

            if stim_code == 0
                for prev_ep = (ep - 1):-1:max(1, ep - 15)
                    tl_prev = time_lock_types(prev_ep);
                    if ismember(tl_prev, stim_markers)
                        stim_code = round(tl_prev / 10);
                        break;
                    end
                    if ismember(tl_prev, [target_marker, no_target_marker])
                        break;
                    end
                end
            end

            if stim_code > 0 && resp_code > 0
                base_code = stim_code * 100 + resp_code;
                if is_notarget
                    base_code = base_code + 1000;
                end
                epoch_condition(ep) = base_code;
            else
                if stim_code == 0, debug_no_stim = debug_no_stim + 1; end
                if resp_code == 0, debug_no_resp = debug_no_resp + 1; end
            end
        end

        unique_conds = unique(epoch_condition(epoch_condition > 0));
        if ~isempty(unique_conds)
            fprintf('  [调试] epoch_condition 实际值分布:\n');
            for uc = 1:length(unique_conds)
                fprintf('    编码 %.1f: %d 个 epoch\n', unique_conds(uc), sum(epoch_condition == unique_conds(uc)));
            end
        else
            fprintf('  [调试] 警告：没有任何epoch被成功分配条件！\n');
        end

        fprintf('  条件分配统计 —— 有目标(41):\n');
        for c = 1:length(Cond_markers)
            n_ep = sum(epoch_condition == Cond_markers(c));
            fprintf('    %s (编码 %d): %d 个 epoch\n', Cond_names{c}, Cond_markers(c), n_ep);
        end
        fprintf('  条件分配统计 —— 无目标(42):\n');
        NT_correct_markers = [1101, 1201, 1301];
        NT_names = {'A_NT_correct', 'B_NT_correct', 'C_NT_correct'};
        for c = 1:3
            n_ep = sum(epoch_condition == NT_correct_markers(c));
            fprintf('    %s (编码 %d): %d 个 epoch\n', NT_names{c}, NT_correct_markers(c), n_ep);
        end
        n_unassigned = sum(ismember(time_lock_types, [target_marker, no_target_marker]) & epoch_condition == 0);
        if n_unassigned > 0
            fprintf('    未能分配条件的41/42-epoch: %d 个 (找不到刺激: %d, 找不到反应: %d)\n', ...
                n_unassigned, debug_no_stim, debug_no_resp);
        end

        % 有目标条件
        for j = 1:nCond
            cond_epoch_idx = find(epoch_condition == Cond_markers(j));
            fprintf('  [有目标] %s (编码 %d): %d 个 epoch\n', Cond_names{j}, Cond_markers(j), length(cond_epoch_idx));
            if ~isempty(cond_epoch_idx)
                EEG_temp = pop_select(EEG, 'trial', cond_epoch_idx);
                data(i, j, :, :) = squeeze(mean(EEG_temp.data, 3));
            else
                warning('  被试 %d 条件 %s 无有效 epoch！', i, Cond_names{j});
                data(i, j, :, :) = zeros(EEG.nbchan, EEG.pnts);
            end
        end

        % 无目标条件
        for c = 1:3
            nt_epoch_idx = find(epoch_condition == NT_correct_markers(c));
            fprintf('  [无目标] %s (编码 %d): %d 个 epoch\n', NT_names{c}, NT_correct_markers(c), length(nt_epoch_idx));
            if ~isempty(nt_epoch_idx)
                EEG_temp = pop_select(EEG, 'trial', nt_epoch_idx);
                data_nt(i, c, :, :) = squeeze(mean(EEG_temp.data, 3));
            else
                warning('  被试 %d 无目标条件 %s 无有效 epoch！', i, NT_names{c});
                data_nt(i, c, :, :) = zeros(EEG.nbchan, EEG.pnts);
            end
        end
    end
end

%% ========================= 保存时间和电极信息 =========================
srate = EEG.srate;
tepoch = EEG.times;
chanloc = EEG.chanlocs;
nbchan = EEG.nbchan;
EEG = [];
EEG.times = tepoch;
EEG.chanlocs = chanloc;
EEG.srate = srate;
EEG.nbchan = nbchan;

save_path = fullfile(file_path, 'all_data.mat');
save(save_path, 'data', 'data_nt', 'EEG', 'Subj', 'SubjFiles', 'Cond_names', 'Cond_markers', ...
    'chan_idx', 'chan_of_interest', 'chan_indices', 'chans_of_interest', 'srate');
fprintf('\n数据已保存至: %s\n', save_path);
fprintf('data 维度: %s (被试 × 条件 × 电极 × 时间点)\n', mat2str(size(data)));

%% ========================= ERP 波形低通滤波 =========================
if apply_lowpass
    fprintf('\n====== 对平均后的ERP波形进行低通滤波 ======\n');
    fprintf('截止频率: %d Hz, 滤波器阶数: %d, 采样率: %d Hz\n', lowpass_cutoff, lowpass_order, srate);

    Wn = lowpass_cutoff / (srate / 2);
    if Wn >= 1
        warning('截止频率 %d Hz 超过奈奎斯特频率 %d Hz，跳过滤波！', lowpass_cutoff, srate/2);
    else
        [filt_b, filt_a] = butter(lowpass_order, Wn, 'low');

        fprintf('正在滤波 data (有目标)...\n');
        for si = 1:size(data, 1)
            for ci = 1:size(data, 2)
                for ch = 1:size(data, 3)
                    data(si, ci, ch, :) = filtfilt(filt_b, filt_a, squeeze(data(si, ci, ch, :)));
                end
            end
        end

        fprintf('正在滤波 data_nt (无目标)...\n');
        for si = 1:size(data_nt, 1)
            for ci = 1:size(data_nt, 2)
                for ch = 1:size(data_nt, 3)
                    data_nt(si, ci, ch, :) = filtfilt(filt_b, filt_a, squeeze(data_nt(si, ci, ch, :)));
                end
            end
        end

        fprintf('低通滤波完成！波形已平滑。\n');
    end
end

%% ========================================================================
%% ========================= Part 2: 画波形图（多电极）====================
%% ========================================================================

colors_correct = {'r', 'b', 'k'};
nChans = length(chans_of_interest);

disp_xlim = [epoch_window(1)*1000, epoch_window(2)*1000];

fig_num = 0;

%% ---- 2.1 所有电极三种正确条件对比波形（子图拼接）----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 多电极三种正确条件对比波形', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
nRows = ceil(nChans / 3);
nCols = min(nChans, 3);

for ci = 1:nChans
    ch = chan_indices(ci);
    subplot(nRows, nCols, ci);
    hold on;
    set(gca, 'YDir', 'normal');
    for c = 1:3
        plot(EEG.times, squeeze(mean(data(:, c, ch, :), 1)), ...
            'Color', colors_correct{c}, 'LineWidth', 1.5);
    end
    xlim(disp_xlim);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
    xlabel('ms'); ylabel('\muV');
    if ci == 1
        legend('A-正确', 'B-正确', 'C-正确', 'Location', 'best', 'FontSize', 8);
    end
    box off;
end
sgtitle(sprintf('图%d 三种刺激条件组平均波形 (正确反应)', fig_num), 'fontsize', 15, 'FontWeight', 'bold');

%% ---- 2.2 如果包含错误条件，所有电极正确 vs 错误对比 ----
if include_error && nCond >= 6
    fig_num = fig_num + 1;
    figure('Name', sprintf('图%d 多电极正确 vs 错误对比', fig_num), 'NumberTitle', 'off', ...
        'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
    for ci = 1:nChans
        ch = chan_indices(ci);
        subplot(nRows, nCols, ci);
        hold on;
        set(gca, 'YDir', 'reverse');
        % 正确条件（实线）
        for c = 1:3
            plot(EEG.times, squeeze(mean(data(:, c, ch, :), 1)), ...
                'Color', colors_correct{c}, 'LineWidth', 1.5, 'LineStyle', '-');
        end
        % 错误条件（虚线）
        for c = 4:6
            plot(EEG.times, squeeze(mean(data(:, c, ch, :), 1)), ...
                'Color', colors_correct{c-3}, 'LineWidth', 1.5, 'LineStyle', '--');
        end
        xlim(disp_xlim);
        line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
        line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
        title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
        xlabel('ms'); ylabel('\muV');
        if ci == 1
            legend('A-正确','B-正确','C-正确','A-错误','B-错误','C-错误', ...
                'Location', 'best', 'FontSize', 7);
        end
        box off;
    end
    sgtitle(sprintf('图%d 正确 vs 错误反应对比 (实线=正确, 虚线=错误)', fig_num), 'fontsize', 15, 'FontWeight', 'bold');
end

%% ---- 2.3 所有电极差异波 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 多电极差异波', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
for ci = 1:nChans
    ch = chan_indices(ci);
    subplot(nRows, nCols, ci);
    hold on; set(gca, 'YDir', 'reverse');
    diff_BA = squeeze(mean(data(:,2,ch,:),1)) - squeeze(mean(data(:,1,ch,:),1));
    diff_CA = squeeze(mean(data(:,3,ch,:),1)) - squeeze(mean(data(:,1,ch,:),1));
    plot(EEG.times, diff_BA, 'Color', 'b', 'LineWidth', 1.5);
    plot(EEG.times, diff_CA, 'Color', 'k', 'LineWidth', 1.5);
    xlim(disp_xlim);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
    xlabel('ms'); ylabel('\muV');
    if ci == 1
        legend('B-A', 'C-A', 'Location', 'best', 'FontSize', 9);
    end
    box off;
end
sgtitle(sprintf('图%d 差异波 (B-A 和 C-A)', fig_num), 'fontsize', 15, 'FontWeight', 'bold');

%% ========================================================================
%% ========================= Part 3: 地形图 ===============================
%% ========================================================================

%%% 请根据实际波形调整以下峰值潜伏期 %%%
P1_peak_ms = 100;
N1_peak_ms = 170;
N2_peak_ms = 250;
P3_peak_ms = 400;

P1_idx = find(EEG.times >= P1_peak_ms, 1, 'first');
N1_idx = find(EEG.times >= N1_peak_ms, 1, 'first');
N2_idx = find(EEG.times >= N2_peak_ms, 1, 'first');
P3_idx = find(EEG.times >= P3_peak_ms, 1, 'first');

%% ---- 3.2 所有正确条件总平均各成分地形图 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 各ERP成分地形图', fig_num), 'NumberTitle', 'off');

subplot(141);
topoplot(squeeze(mean(mean(data(:, 1:3, :, P1_idx), 1), 2)), EEG.chanlocs);
title(sprintf('P1 (%d ms)', P1_peak_ms), 'fontsize', 12);

subplot(142);
topoplot(squeeze(mean(mean(data(:, 1:3, :, N1_idx), 1), 2)), EEG.chanlocs);
title(sprintf('N1 (%d ms)', N1_peak_ms), 'fontsize', 12);

subplot(143);
topoplot(squeeze(mean(mean(data(:, 1:3, :, N2_idx), 1), 2)), EEG.chanlocs);
title(sprintf('N2 (%d ms)', N2_peak_ms), 'fontsize', 12);

subplot(144);
topoplot(squeeze(mean(mean(data(:, 1:3, :, P3_idx), 1), 2)), EEG.chanlocs);
title(sprintf('P3 (%d ms)', P3_peak_ms), 'fontsize', 12);

sgtitle(sprintf('图%d 各ERP成分地形图', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

%% ---- 3.3 各条件在 N2 时间窗口内的平均地形图 ----
N2_window_ms = 50;
N2_start_idx = find(EEG.times >= (N2_peak_ms - N2_window_ms), 1, 'first');
N2_end_idx = find(EEG.times >= (N2_peak_ms + N2_window_ms), 1, 'first');

fig_num = fig_num + 1;
figure('Name', sprintf('图%d 各条件 N2 地形图', fig_num), 'NumberTitle', 'off');
for c = 1:3
    subplot(1, 3, c);
    condition_data = squeeze(mean(mean(data(:, c, :, N2_start_idx:N2_end_idx), 1), 4));
    topoplot(condition_data, EEG.chanlocs, 'maplimits', 'maxmin');
    title(sprintf('%s N2 (%d±%dms)', Cond_names{c}, N2_peak_ms, N2_window_ms), 'fontsize', 11);
end
sgtitle(sprintf('图%d 各条件 N2 地形图', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

%% ---- 3.4 各条件在 P3 时间窗口内的平均地形图 ----
P3_window_ms = 100;
P3_start_idx = find(EEG.times >= (P3_peak_ms - P3_window_ms), 1, 'first');
P3_end_idx = find(EEG.times >= (P3_peak_ms + P3_window_ms), 1, 'first');

fig_num = fig_num + 1;
figure('Name', sprintf('图%d 各条件 P3 地形图', fig_num), 'NumberTitle', 'off');
for c = 1:3
    subplot(1, 3, c);
    condition_data = squeeze(mean(mean(data(:, c, :, P3_start_idx:P3_end_idx), 1), 4));
    topoplot(condition_data, EEG.chanlocs, 'maplimits', 'maxmin');
    title(sprintf('%s P3 (%d±%dms)', Cond_names{c}, P3_peak_ms, P3_window_ms), 'fontsize', 11);
end
sgtitle(sprintf('图%d 各条件 P3 地形图', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

%% ---- 3.5 多条件逐时间窗地形图矩阵（时空演变图）----
fprintf('\n====== 绘制多条件逐时间窗地形图矩阵 ======\n');

topo_step_ms  = 50;   %%% 时间步长(ms)
topo_half_win = 25;    %%% 每个地形图取中心时间 ±25ms 的平均值
topo_times_ms = (epoch_window(1)*1000) : topo_step_ms : (epoch_window(2)*1000);
nTopo = length(topo_times_ms);

if include_error && nCond >= 6
    plot_nCond_topo = 6;
else
    plot_nCond_topo = 3;
end

topo_data = zeros(plot_nCond_topo, nTopo, EEG.nbchan);
for c = 1:plot_nCond_topo
    for ti = 1:nTopo
        t_idx = find(EEG.times >= (topo_times_ms(ti) - topo_half_win) & ...
                     EEG.times <= (topo_times_ms(ti) + topo_half_win));
        if isempty(t_idx)
            [~, t_idx] = min(abs(EEG.times - topo_times_ms(ti)));
        end
        topo_data(c, ti, :) = squeeze(mean(mean(data(:, c, :, t_idx), 1), 4));
    end
end

global_clim = 3;
if global_clim < 0.01, global_clim = 1; end
fprintf('地形图统一色标范围: [%.2f, %.2f] μV\n', -global_clim, global_clim);

margin_l = 0.07;
margin_r = 0.05;
margin_t = 0.06;
margin_b = 0.02;
cbar_w   = 0.02;
gap_x    = 0.002;
gap_y    = 0.005;
cell_w = (1 - margin_l - margin_r - cbar_w - gap_x*(nTopo-1)) / nTopo;
cell_h = (1 - margin_t - margin_b - gap_y*(plot_nCond_topo-1)) / plot_nCond_topo;

fig_num = fig_num + 1;
fig_w = min(1920, max(1200, nTopo * 80 + 150));
fig_h = min(1000, max(400, plot_nCond_topo * 130 + 60));
fig_topo = figure('Name', sprintf('图%d 各条件ERP地形图时空演变', fig_num), ...
    'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [10 10 fig_w fig_h]);

for c = 1:plot_nCond_topo
    row_bottom = 1 - margin_t - c * cell_h - (c-1) * gap_y;

    for ti = 1:nTopo
        col_left = margin_l + (ti-1) * (cell_w + gap_x);
        ax = axes('Position', [col_left, row_bottom, cell_w, cell_h]);

        topoplot(squeeze(topo_data(c, ti, :)), EEG.chanlocs, ...
            'maplimits', [-global_clim global_clim], ...
            'electrodes', 'pts', 'conv', 'on', ...
            'shading', 'interp', 'style', 'map');
        colormap(ax, jet);

        if c == 1
            title(sprintf('%d', topo_times_ms(ti)), 'FontSize', 7, 'FontWeight', 'normal');
        end
    end

    cond_label = strrep(Cond_names{c}, '_', ' ');
    annotation(fig_topo, 'textbox', ...
        [0, row_bottom, margin_l, cell_h], ...
        'String', cond_label, 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 9, 'FontWeight', 'bold', 'Interpreter', 'none');
end

total_h = plot_nCond_topo * cell_h + (plot_nCond_topo - 1) * gap_y;
cb_left   = 1 - margin_r - cbar_w + 0.005;
cb_bottom = 1 - margin_t - total_h + total_h * 0.1;
cb_h      = total_h * 0.8;
ax_cb = axes('Position', [cb_left, cb_bottom, cbar_w * 0.5, cb_h]);
imagesc(ax_cb, 1, linspace(-global_clim, global_clim, 256), linspace(-global_clim, global_clim, 256)');
set(ax_cb, 'YDir', 'normal', 'XTick', [], 'YAxisLocation', 'right', 'FontSize', 7);
yticks(ax_cb, [-global_clim, 0, global_clim]);
yticklabels(ax_cb, {sprintf('%.1f', -global_clim), '0', sprintf('%.1f', global_clim)});
colormap(ax_cb, jet);
ylabel(ax_cb, '\muV', 'FontSize', 8);

sgtitle(sprintf('图%d 各条件ERP地形图时空演变 (SSVEP已消除, 每%dms, \\pm%dms平均)', ...
    fig_num, topo_step_ms, topo_half_win), 'fontsize', 13, 'FontWeight', 'bold');

try
    save_fname = fullfile(file_path, 'Topomap_timeseries_all_conditions.png');
    saveas(gcf, save_fname);
    fprintf('已保存地形图序列至: %s\n', save_fname);
    save_fname_fig = fullfile(file_path, 'Topomap_timeseries_all_conditions.fig');
    savefig(gcf, save_fname_fig);
    fprintf('已保存 .fig 文件至: %s\n', save_fname_fig);
catch
    warning('保存地形图序列失败，请检查 file_path 是否可写。');
end

%% ========================================================================
%% ========================= Part 4: 峰值与潜伏期测量 =====================
%% ========================================================================

%% ---- 4.1 自动峰值检测：N2 和 P3（多电极）----
N2_search_window = [150 350];   %%% N2 搜索窗口
P3_search_window = [250 600];   %%% P3 搜索窗口

N2_win_idx = find(EEG.times >= N2_search_window(1) & EEG.times <= N2_search_window(2));
P3_win_idx = find(EEG.times >= P3_search_window(1) & EEG.times <= P3_search_window(2));

for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};

    N2_amp = zeros(nSubj, 3);
    N2_lat = zeros(nSubj, 3);
    P3_amp = zeros(nSubj, 3);
    P3_lat = zeros(nSubj, 3);

    for i = 1:nSubj
        for c = 1:3
            wave = squeeze(data(i, c, ch, :));

            [min_val, min_pos] = min(wave(N2_win_idx));
            N2_amp(i, c) = min_val;
            N2_lat(i, c) = EEG.times(N2_win_idx(min_pos));

            [max_val, max_pos] = max(wave(P3_win_idx));
            P3_amp(i, c) = max_val;
            P3_lat(i, c) = EEG.times(P3_win_idx(max_pos));
        end
    end

    fprintf('\n====== N2 振幅和潜伏期 (电极: %s) ======\n', ch_name);
    fprintf('条件\t\t平均振幅(uV)\t标准差\t\t平均潜伏期(ms)\t标准差\n');
    for c = 1:3
        fprintf('%s\t%.2f\t\t%.2f\t\t%.1f\t\t%.1f\n', ...
            Cond_names{c}, mean(N2_amp(:,c)), std(N2_amp(:,c)), ...
            mean(N2_lat(:,c)), std(N2_lat(:,c)));
    end

    fprintf('\n====== P3 振幅和潜伏期 (电极: %s) ======\n', ch_name);
    fprintf('条件\t\t平均振幅(uV)\t标准差\t\t平均潜伏期(ms)\t标准差\n');
    for c = 1:3
        fprintf('%s\t%.2f\t\t%.2f\t\t%.1f\t\t%.1f\n', ...
            Cond_names{c}, mean(P3_amp(:,c)), std(P3_amp(:,c)), ...
            mean(P3_lat(:,c)), std(P3_lat(:,c)));
    end

    all_N2_amp{ci} = N2_amp;
    all_N2_lat{ci} = N2_lat;
    all_P3_amp{ci} = P3_amp;
    all_P3_lat{ci} = P3_lat;
end

%% ---- 4.2 平均振幅测量（多电极）----
for ci = 1:nChans
    ch = chan_indices(ci);
    N2_mean_amp_tmp = zeros(nSubj, 3);
    P3_mean_amp_tmp = zeros(nSubj, 3);
    for i = 1:nSubj
        for c = 1:3
            N2_mean_amp_tmp(i, c) = mean(squeeze(data(i, c, ch, N2_start_idx:N2_end_idx)));
            P3_mean_amp_tmp(i, c) = mean(squeeze(data(i, c, ch, P3_start_idx:P3_end_idx)));
        end
    end
    all_N2_mean_amp{ci} = N2_mean_amp_tmp;
    all_P3_mean_amp{ci} = P3_mean_amp_tmp;
end

N2_amp = all_N2_amp{1}; N2_lat = all_N2_lat{1};
P3_amp = all_P3_amp{1}; P3_lat = all_P3_lat{1};
N2_mean_amp = all_N2_mean_amp{1}; P3_mean_amp = all_P3_mean_amp{1};

%% ---- 4.3 多电极 N2/P3 平均振幅柱状图 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 多电极 N2 平均振幅', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
for ci = 1:nChans
    subplot(nRows, nCols, ci); hold on;
    means_N2 = mean(all_N2_mean_amp{ci});
    se_N2 = std(all_N2_mean_amp{ci}) / sqrt(nSubj);
    bar_h = bar(means_N2);
    bar_h.FaceColor = 'flat';
    bar_h.CData = [1 0 0; 0 0 1; 0 0 0];
    errorbar(1:3, means_N2, se_N2, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', {'A', 'B', 'C'});
    ylabel('\muV');
    title(sprintf('%s N2 (%d±%dms)', chans_of_interest{ci}, N2_peak_ms, N2_window_ms), 'fontsize', 11);
    box off;
end
sgtitle(sprintf('图%d N2 平均振幅 (各电极)', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

fig_num = fig_num + 1;
figure('Name', sprintf('图%d 多电极 P3 平均振幅', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
for ci = 1:nChans
    subplot(nRows, nCols, ci); hold on;
    means_P3 = mean(all_P3_mean_amp{ci});
    se_P3 = std(all_P3_mean_amp{ci}) / sqrt(nSubj);
    bar_h = bar(means_P3);
    bar_h.FaceColor = 'flat';
    bar_h.CData = [1 0 0; 0 0 1; 0 0 0];
    errorbar(1:3, means_P3, se_P3, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', {'A', 'B', 'C'});
    ylabel('\muV');
    title(sprintf('%s P3 (%d±%dms)', chans_of_interest{ci}, P3_peak_ms, P3_window_ms), 'fontsize', 11);
    box off;
end
sgtitle(sprintf('图%d P3 平均振幅 (各电极)', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% ======= Part 6: 逐时间点重复测量方差分析 ===============================
%% ========================================================================

fprintf('\n====== 正在进行逐时间点重复测量方差分析（多电极）... ======\n');

nTimepoints = size(data, 4);

for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};
    fprintf('  电极 %s...\n', ch_name);

    F_vals = zeros(1, nTimepoints);
    P_vals = ones(1, nTimepoints);

    for t = 1:nTimepoints
        anova_data = squeeze(data(:, 1:3, ch, t));
        try
            [p, tbl] = anova_rm(anova_data, 'off');
            P_vals(t) = p(1);
            F_vals(t) = tbl{2, 5};
        catch
            P_vals(t) = 1;
            F_vals(t) = 0;
        end
    end

    try
        [p_fdr, ~] = fdr(P_vals, 0.05);
        fprintf('    %s FDR 校正阈值: p = %.6f\n', ch_name, p_fdr);
    catch
        p_fdr = 0.05;
    end

    all_F_vals{ci} = F_vals;
    all_P_vals{ci} = P_vals;
    all_p_fdr(ci) = p_fdr;
end

%% ---- 6.1 多电极 波形图 + p 值图 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 多电极逐时间点ANOVA', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 350*ceil(nChans/3)*1.0]);

for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};

    subplot(2, nChans, ci); hold on;
    set(gca, 'YDir', 'reverse');
    plot(EEG.times, squeeze(mean(data(:,1,ch,:),1)), '-r', 'LineWidth', 1.2);
    plot(EEG.times, squeeze(mean(data(:,2,ch,:),1)), '-b', 'LineWidth', 1.2);
    plot(EEG.times, squeeze(mean(data(:,3,ch,:),1)), '-k', 'LineWidth', 1.2);
    xlim(disp_xlim);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    title(ch_name, 'fontsize', 12, 'FontWeight', 'bold');
    if ci == 1, ylabel('\muV'); end
    xlabel('ms');
    if ci == 1, legend('A','B','C', 'Location', 'best', 'FontSize', 7); end
    box off;

    subplot(2, nChans, nChans + ci); hold on;
    plot(EEG.times, all_P_vals{ci}, 'b', 'LineWidth', 1);
    line(disp_xlim, [0.05 0.05], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(disp_xlim, [all_p_fdr(ci) all_p_fdr(ci)], 'Color', 'r', 'LineWidth', 1.5);
    xlim(disp_xlim); ylim([0 0.1]);
    title(sprintf('p值 (FDR=%.4f)', all_p_fdr(ci)), 'fontsize', 10);
    if ci == 1, ylabel('p value'); end
    xlabel('ms');
    box off;
end
sgtitle(sprintf('图%d 逐时间点重复测量方差分析 (各电极)', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% ============ Part 7: 导出数据 ==========================================
%% ========================================================================

fprintf('\n====== 导出统计数据 ======\n');

SubjNames = cell(nSubj, 1);
for si = 1:nSubj
    [~, SubjNames{si}, ~] = fileparts(SubjFiles{si});
end

csv_header = 'Subject,A_correct,B_correct,C_correct\n';

for ci = 1:nChans
    ch_name = chans_of_interest{ci};

    fname = fullfile(file_path, sprintf('N2_mean_amplitude_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.4f,%.4f,%.4f\n', SubjNames{si}, ...
            all_N2_mean_amp{ci}(si,1), all_N2_mean_amp{ci}(si,2), all_N2_mean_amp{ci}(si,3));
    end
    fclose(fid);

    fname = fullfile(file_path, sprintf('P3_mean_amplitude_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.4f,%.4f,%.4f\n', SubjNames{si}, ...
            all_P3_mean_amp{ci}(si,1), all_P3_mean_amp{ci}(si,2), all_P3_mean_amp{ci}(si,3));
    end
    fclose(fid);

    fname = fullfile(file_path, sprintf('N2_peak_latency_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.1f,%.1f,%.1f\n', SubjNames{si}, ...
            all_N2_lat{ci}(si,1), all_N2_lat{ci}(si,2), all_N2_lat{ci}(si,3));
    end
    fclose(fid);

    fname = fullfile(file_path, sprintf('P3_peak_latency_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.1f,%.1f,%.1f\n', SubjNames{si}, ...
            all_P3_lat{ci}(si,1), all_P3_lat{ci}(si,2), all_P3_lat{ci}(si,3));
    end
    fclose(fid);

    fprintf('电极 %s 的数据已导出 (N2振幅/潜伏期, P3振幅/潜伏期)\n', ch_name);
end

%% ========================================================================
%% ===== Part 8: SSVEP 减法 —— 有目标 - 无目标，提取目标诱发 ERP =========
%% ========================================================================

fprintf('\n====== Part 8: SSVEP 减法分析 (有目标 - 无目标) ======\n');

data_diff = data(:, 1:3, :, :) - data_nt(:, 1:3, :, :);
fprintf('data_diff 维度: %s (被试 × 条件 × 电极 × 时间点)\n', mat2str(size(data_diff)));

Cond_labels_diff = {'A (目标-非目标)', 'B (目标-非目标)', 'C (目标-非目标)'};

%% ---- 8.1 多电极：目标诱发ERP波形 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 目标诱发ERP (SSVEP已消除) - 多电极', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);

for ci = 1:nChans
    ch = chan_indices(ci);
    subplot(nRows, nCols, ci);
    hold on; set(gca, 'YDir', 'reverse');
    for c = 1:3
        plot(EEG.times, squeeze(mean(data_diff(:, c, ch, :), 1)), ...
            'Color', colors_correct{c}, 'LineWidth', 1.5);
    end
    xlim(disp_xlim);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
    xlabel('ms'); ylabel('\muV');
    if ci == 1
        legend('A', 'B', 'C', 'Location', 'best', 'FontSize', 9);
    end
    box off;
end
sgtitle(sprintf('图%d 目标诱发ERP波形 (有目标 - 无目标, SSVEP已消除)', fig_num), 'fontsize', 15, 'FontWeight', 'bold');

%% ---- 8.2 多电极：有目标 vs 无目标 对比 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 有目标 vs 无目标 vs 差值 - 多电极', fig_num), 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);

for ci = 1:nChans
    ch = chan_indices(ci);
    subplot(nRows, nCols, ci);
    hold on; set(gca, 'YDir', 'reverse');

    target_avg    = squeeze(mean(mean(data(:, 1:3, ch, :), 1), 2));
    nontarget_avg = squeeze(mean(mean(data_nt(:, 1:3, ch, :), 1), 2));
    diff_avg      = squeeze(mean(mean(data_diff(:, 1:3, ch, :), 1), 2));

    plot(EEG.times, target_avg, '-r', 'LineWidth', 1.5);
    plot(EEG.times, nontarget_avg, '-b', 'LineWidth', 1.5);
    plot(EEG.times, diff_avg, '-k', 'LineWidth', 2);

    xlim(disp_xlim);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
    xlabel('ms'); ylabel('\muV');
    if ci == 1
        legend('有目标', '无目标', '差值(目标诱发ERP)', 'Location', 'best', 'FontSize', 7);
    end
    box off;
end
sgtitle(sprintf('图%d 有目标 vs 无目标 vs 目标诱发ERP (SSVEP消除)', fig_num), 'fontsize', 15, 'FontWeight', 'bold');

%% ---- 8.3 目标诱发ERP的N2和P3分析（多电极）----
fprintf('\n====== 目标诱发ERP (SSVEP消除后) 各电极N2/P3 ======\n');

for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};

    N2_amp_diff = zeros(nSubj, 3);
    N2_lat_diff = zeros(nSubj, 3);
    P3_amp_diff = zeros(nSubj, 3);
    P3_lat_diff = zeros(nSubj, 3);

    for s = 1:nSubj
        for c = 1:3
            wave = squeeze(data_diff(s, c, ch, :));
            [min_val, min_pos] = min(wave(N2_win_idx));
            N2_amp_diff(s, c) = min_val;
            N2_lat_diff(s, c) = EEG.times(N2_win_idx(min_pos));
            [max_val, max_pos] = max(wave(P3_win_idx));
            P3_amp_diff(s, c) = max_val;
            P3_lat_diff(s, c) = EEG.times(P3_win_idx(max_pos));
        end
    end

    fprintf('\n  [%s] 目标诱发ERP N2:\n', ch_name);
    fprintf('  条件\t\t平均振幅(uV)\t标准差\t\t平均潜伏期(ms)\t标准差\n');
    for c = 1:3
        fprintf('  %s\t%.2f\t\t%.2f\t\t%.1f\t\t%.1f\n', ...
            Cond_names{c}, mean(N2_amp_diff(:,c)), std(N2_amp_diff(:,c)), ...
            mean(N2_lat_diff(:,c)), std(N2_lat_diff(:,c)));
    end
    fprintf('  [%s] 目标诱发ERP P3:\n', ch_name);
    fprintf('  条件\t\t平均振幅(uV)\t标准差\t\t平均潜伏期(ms)\t标准差\n');
    for c = 1:3
        fprintf('  %s\t%.2f\t\t%.2f\t\t%.1f\t\t%.1f\n', ...
            Cond_names{c}, mean(P3_amp_diff(:,c)), std(P3_amp_diff(:,c)), ...
            mean(P3_lat_diff(:,c)), std(P3_lat_diff(:,c)));
    end
end

%% ---- 8.4 目标诱发ERP 地形图 ----
fig_num = fig_num + 1;
figure('Name', sprintf('图%d 目标诱发ERP 地形图 (SSVEP消除)', fig_num), 'NumberTitle', 'off');

subplot(141);
topoplot(squeeze(mean(mean(data_diff(:, 1:3, :, P1_idx), 1), 2)), EEG.chanlocs);
title(sprintf('P1 (%d ms)', P1_peak_ms), 'fontsize', 12);

subplot(142);
topoplot(squeeze(mean(mean(data_diff(:, 1:3, :, N1_idx), 1), 2)), EEG.chanlocs);
title(sprintf('N1 (%d ms)', N1_peak_ms), 'fontsize', 12);

subplot(143);
topoplot(squeeze(mean(mean(data_diff(:, 1:3, :, N2_idx), 1), 2)), EEG.chanlocs);
title(sprintf('N2 (%d ms)', N2_peak_ms), 'fontsize', 12);

subplot(144);
topoplot(squeeze(mean(mean(data_diff(:, 1:3, :, P3_idx), 1), 2)), EEG.chanlocs);
title(sprintf('P3 (%d ms)', P3_peak_ms), 'fontsize', 12);

sgtitle(sprintf('图%d 目标诱发ERP 地形图 (有目标-无目标)', fig_num), 'fontsize', 14, 'FontWeight', 'bold');

%% ---- 8.5 导出目标诱发ERP数据 ----
for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};

    fname = fullfile(file_path, sprintf('TargetERP_N2_amplitude_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, 'Subject,A_correct,B_correct,C_correct\n');
    for si = 1:nSubj
        N2_mean_diff = zeros(1, 3);
        for c = 1:3
            N2_mean_diff(c) = mean(squeeze(data_diff(si, c, ch, N2_start_idx:N2_end_idx)));
        end
        fprintf(fid, '%s,%.4f,%.4f,%.4f\n', SubjNames{si}, N2_mean_diff(1), N2_mean_diff(2), N2_mean_diff(3));
    end
    fclose(fid);

    fname = fullfile(file_path, sprintf('TargetERP_P3_amplitude_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, 'Subject,A_correct,B_correct,C_correct\n');
    for si = 1:nSubj
        P3_mean_diff = zeros(1, 3);
        for c = 1:3
            P3_mean_diff(c) = mean(squeeze(data_diff(si, c, ch, P3_start_idx:P3_end_idx)));
        end
        fprintf(fid, '%s,%.4f,%.4f,%.4f\n', SubjNames{si}, P3_mean_diff(1), P3_mean_diff(2), P3_mean_diff(3));
    end
    fclose(fid);

    fprintf('电极 %s 目标诱发ERP数据已导出\n', ch_name);
end

fprintf('\n====== 所有分析完成！ ======\n');

%% =====================================================================
%% 保存所有图表到指定目录
%% =====================================================================

if ~exist(save_figure_dir, 'dir')
    mkdir(save_figure_dir);
    fprintf('\n创建保存目录: %s\n', save_figure_dir);
end

all_figs = findobj('Type', 'figure');
all_figs = sort(all_figs);

for i = 1:length(all_figs)
    h = all_figs(i);
    fig_name = get(h, 'Name');

    if isempty(fig_name)
        fig_name = sprintf('Figure_%02d', i);
    end

    fig_name_clean = regexprep(fig_name, '[\\/:*?"<>|]', '_');

    try
        saveas(h, fullfile(save_figure_dir, [fig_name_clean '.png']));
    catch ME
        warning('保存图表 "%s" 时出错: %s', fig_name, ME.message);
    end
end

fprintf('共保存 %d 张图表至: %s\n', length(all_figs), save_figure_dir);
