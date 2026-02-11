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
%     'niuhuan_processed.set'
%     'qiuyuanrong_processed.set'
%     'wanghongqi_processed.set'
%     'zhouxiangling_processed.set'
% };

%% ========== 扫描/确认被试文件列表 ==========
fprintf('\n====== 数据目录: %s ======\n', file_path);
if ~exist(file_path, 'dir')
    error('错误：目录不存在: %s\n请检查 file_path 设置。', file_path);
end

if auto_scan
    % 自动扫描目录中所有 .set 文件
    set_files = dir(fullfile(file_path, '*.set'));
    if isempty(set_files)
        error('错误：目录 %s 中没有找到任何 .set 文件！', file_path);
    end
    SubjFiles = {set_files.name}';  % 转为列向量 cell 数组
end

nSubj = length(SubjFiles);
fprintf('共找到 %d 个被试文件:\n', nSubj);
for fi = 1:nSubj
    fprintf('  被试%2d: %s\n', fi, SubjFiles{fi});
end
fprintf('\n');

% 定义被试编号（1 到 nSubj）
Subj = 1:nSubj;

% 分段参数
epoch_window = [-0.1 0.8];             % 分段时间窗，单位：秒（-100ms 到 800ms）
baseline_window = [-100 0];             % 基线校正窗口，单位：毫秒

% 原始事件标记
stim_markers = [11, 21, 31];            % A/B/C 三种刺激类型标记
target_marker = 41;                     % 有目标视觉搜索标记
no_target_marker = 42;                  % 无目标视觉搜索标记
correct_marker = 12;                    % 正确反应标记
incorrect_marker = 22;                  % 错误反应标记
no_response_marker = 32;               % 无反应标记
response_markers = [correct_marker, incorrect_marker, no_response_marker];

% 复合标记编码（自动生成，无需修改）
%   百位 = 刺激类型编号（1=A, 2=B, 3=C）
%   个位 = 反应类型编号（1=正确, 2=错误, 3=无反应）
composite_correct   = [101, 201, 301];  % 三种刺激 × 正确反应
composite_incorrect = [102, 202, 302];  % 三种刺激 × 错误反应
composite_noresp    = [103, 203, 303];  % 三种刺激 × 无反应

% 分析条件定义
% —— 主要分析条件（正确反应）——
Cond_markers = composite_correct;       % 用于主分析的条件标记 [101, 201, 301]
Cond_names = {'A_correct', 'B_correct', 'C_correct'};  % 条件名称
nCond = length(Cond_markers);

% —— 是否同时提取错误反应条件 ——
include_error = true;                   %%% 设为 true 则同时提取错误反应条件
if include_error
    Cond_markers = [composite_correct, composite_incorrect]; % [101,201,301,102,202,302]
    Cond_names = {'A_correct','B_correct','C_correct','A_incorrect','B_incorrect','C_incorrect'};
    nCond = length(Cond_markers);
end

% 感兴趣电极列表（可以指定多个电极，将分别出图和统计）
% 常见选择：Fz, Cz, Pz, Oz, PO7, PO8, P3, P4, O1, O2 等
chans_of_interest = {'Fz', 'Cz', 'Pz', 'Oz', 'PO7', 'PO8'};  %%% 修改为你想查看的电极列表
chan_indices = [];                      % 稍后自动查找索引

% 兼容旧变量名（后续部分画图代码用）
chan_of_interest = chans_of_interest{1};  % 默认第一个电极
chan_idx = [];

%% ========================= Part 1: 复合标记创建与分段 =========================
fprintf('\n====== 开始处理数据 ======\n');

for i = 1:nSubj
    file_name = SubjFiles{i};
    
    % 检查文件是否存在
    full_path = fullfile(file_path, file_name);
    if ~exist(full_path, 'file')
        warning('文件不存在，跳过: %s', full_path);
        continue;
    end
    
    fprintf('\n--- 正在处理被试 %d/%d: %s ---\n', i, nSubj, file_name);
    
    EEG = pop_loadset('filename', file_name, 'filepath', file_path);
    
    % 第一次成功加载时查找所有感兴趣电极的索引
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
                chan_indices(ci) = 0;  % 标记未找到
            end
        end
        % 移除未找到的电极
        valid = chan_indices > 0;
        chans_of_interest = chans_of_interest(valid);
        chan_indices = chan_indices(valid);
        if isempty(chan_indices)
            error('没有找到任何指定的电极，请检查 chans_of_interest 中的电极名称！');
        end
        fprintf('共找到 %d 个有效电极\n', length(chan_indices));
        % 兼容旧变量
        chan_of_interest = chans_of_interest{1};
        chan_idx = chan_indices(1);
    end
    
    %% ---- 辅助函数：将事件类型统一转为数值 ----
    % （定义为内联逻辑，在下面多处使用）
    
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
                    stim_code = round(stim_type / 10);  % 11->1, 21->2, 31->3（用round避免浮点问题）
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
        
        % 按条件提取（连续数据已创建复合标记，可用 pop_selectevent）
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
        
        % ---- Step 1: 获取每个epoch的锁时事件类型 和 epoch内所有事件信息 ----
        time_lock_types = zeros(1, EEG.trials);
        
        for ep = 1:EEG.trials
            ep_types = EEG.epoch(ep).eventtype;
            ep_lats  = EEG.epoch(ep).eventlatency;
            
            % eventlatency 可能是 cell 或数组，统一处理
            if iscell(ep_lats)
                lats = cellfun(@(x) double(x), ep_lats);
            else
                lats = double(ep_lats);
            end
            
            % 找到 latency 最接近 0 的事件作为锁时事件
            [~, zero_idx] = min(abs(lats));
            if iscell(ep_types)
                tl = ep_types{zero_idx};
            else
                tl = ep_types(zero_idx);
            end
            if ischar(tl) || isstring(tl), tl = str2double(tl); end
            time_lock_types(ep) = tl;
        end
        
        % 打印锁时事件统计
        unique_tl = unique(time_lock_types);
        fprintf('  锁时事件类型统计:\n');
        for ut = 1:length(unique_tl)
            fprintf('    标记 %g: %d 个 epoch\n', unique_tl(ut), sum(time_lock_types == unique_tl(ut)));
        end
        
        % 打印第一个41-locked epoch的内部事件（调试用）
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
        
        % ---- Step 2: 为每个41-locked epoch分配条件 ----
        % 策略：
        %   反应类型 → 在当前epoch的事件列表中，查找latency>0的12/22/32
        %   刺激类型 → 先查当前epoch事件列表(latency<0的11/21/31)
        %              若找不到，再按epoch时间顺序向前查找前面epoch的锁时事件
        %              (因为epoch按时间排列，所以前面的epoch对应更早的事件)
        
        epoch_condition = zeros(1, EEG.trials);  % 0=未分配
        debug_no_stim = 0;
        debug_no_resp = 0;
        
        for ep = 1:EEG.trials
            % 只处理以41为锁时点的epoch
            if time_lock_types(ep) ~= target_marker
                continue;
            end
            
            % 获取当前epoch内所有事件的类型和潜伏期
            ep_types = EEG.epoch(ep).eventtype;
            ep_lats  = EEG.epoch(ep).eventlatency;
            nev = length(ep_types);
            
            % 转换为数值数组
            types_num = nan(1, nev);
            lats_num  = nan(1, nev);
            for k = 1:nev
                if iscell(ep_types), t = ep_types{k}; else, t = ep_types(k); end
                if ischar(t) || isstring(t), t = str2double(t); end
                types_num(k) = t;
                if iscell(ep_lats), lats_num(k) = double(ep_lats{k}); else, lats_num(k) = double(ep_lats(k)); end
            end
            
            % ---- 查找反应类型（在当前epoch内，latency > 0）----
            resp_code = 0;
            % 先找epoch内 latency > 0 的反应标记
            resp_events_idx = find(lats_num > 0);
            for ri = 1:length(resp_events_idx)
                k = resp_events_idx(ri);
                if types_num(k) == correct_marker,     resp_code = 1; break; end
                if types_num(k) == incorrect_marker,    resp_code = 2; break; end
                if types_num(k) == no_response_marker,  resp_code = 3; break; end
            end
            
            % 如果epoch内没找到反应，向后查找后续epoch的锁时事件
            if resp_code == 0
                for next_ep = (ep + 1):min(EEG.trials, ep + 5)
                    tl_next = time_lock_types(next_ep);
                    if tl_next == correct_marker,     resp_code = 1; break; end
                    if tl_next == incorrect_marker,    resp_code = 2; break; end
                    if tl_next == no_response_marker,  resp_code = 3; break; end
                    % 碰到下一个刺激或搜索标记就停止
                    if ismember(tl_next, [stim_markers, target_marker, no_target_marker])
                        break;
                    end
                end
            end
            
            % ---- 查找刺激类型 ----
            stim_code = 0;
            
            % 方法1: 在当前epoch事件列表中查找 latency < 0 的刺激标记
            stim_events_idx = find(lats_num < 0);
            for si_k = 1:length(stim_events_idx)
                k = stim_events_idx(si_k);
                if ismember(types_num(k), stim_markers)
                    stim_code = round(types_num(k) / 10);  % 11->1, 21->2, 31->3
                    break;
                end
            end
            
            % 方法2: 如果epoch内没找到刺激，按epoch顺序向前查找
            % (epochs 按原始时间排列，前一个epoch的锁时事件对应更早的事件)
            if stim_code == 0
                for prev_ep = (ep - 1):-1:max(1, ep - 15)
                    tl_prev = time_lock_types(prev_ep);
                    if ismember(tl_prev, stim_markers)
                        stim_code = round(tl_prev / 10);  % 11->1, 21->2, 31->3
                        break;
                    end
                    % 碰到另一个搜索标记(41/42)就停止，说明跨试次了
                    if ismember(tl_prev, [target_marker, no_target_marker])
                        break;
                    end
                end
            end
            
            % ---- 分配条件编码 ----
            if stim_code > 0 && resp_code > 0
                epoch_condition(ep) = stim_code * 100 + resp_code;
            else
                if stim_code == 0, debug_no_stim = debug_no_stim + 1; end
                if resp_code == 0, debug_no_resp = debug_no_resp + 1; end
            end
        end
        
        % 打印条件分配统计
        % 先打印 epoch_condition 的实际值分布（调试用）
        unique_conds = unique(epoch_condition(epoch_condition > 0));
        if ~isempty(unique_conds)
            fprintf('  [调试] epoch_condition 实际值分布:\n');
            for uc = 1:length(unique_conds)
                fprintf('    编码 %.1f: %d 个 epoch\n', unique_conds(uc), sum(epoch_condition == unique_conds(uc)));
            end
        else
            fprintf('  [调试] 警告：没有任何epoch被成功分配条件！\n');
        end
        
        fprintf('  条件分配统计（以标记41为锁时点的epoch）:\n');
        for c = 1:length(Cond_markers)
            n_ep = sum(epoch_condition == Cond_markers(c));
            fprintf('    %s (编码 %d): %d 个 epoch\n', Cond_names{c}, Cond_markers(c), n_ep);
        end
        n_unassigned_41 = sum(time_lock_types == target_marker & epoch_condition == 0);
        if n_unassigned_41 > 0
            fprintf('    未能分配条件的41-epoch: %d 个 (找不到刺激: %d, 找不到反应: %d)\n', ...
                n_unassigned_41, debug_no_stim, debug_no_resp);
        end
        n_non41 = sum(time_lock_types ~= target_marker);
        fprintf('    非41锁时的epoch（已忽略）: %d 个\n', n_non41);
        
        % ---- Step 3: 按条件选择epoch并计算平均 ----
        for j = 1:nCond
            cond_epoch_idx = find(epoch_condition == Cond_markers(j));
            fprintf('  条件 %s (编码 %d): %d 个 epoch\n', Cond_names{j}, Cond_markers(j), length(cond_epoch_idx));
            
            if ~isempty(cond_epoch_idx)
                EEG_temp = pop_select(EEG, 'trial', cond_epoch_idx);
                data(i, j, :, :) = squeeze(mean(EEG_temp.data, 3));
            else
                warning('  被试 %d (%s) 条件 %s 无有效 epoch！', i, file_name, Cond_names{j});
                data(i, j, :, :) = zeros(EEG.nbchan, EEG.pnts);
            end
        end
    end
end

%% ========================= 保存时间和电极信息 =========================
tepoch = EEG.times;
chanloc = EEG.chanlocs;
EEG = [];
EEG.times = tepoch;
EEG.chanlocs = chanloc;

% 保存数据
save_path = fullfile(file_path, 'all_data.mat');
save(save_path, 'data', 'EEG', 'Subj', 'SubjFiles', 'Cond_names', 'Cond_markers', ...
    'chan_idx', 'chan_of_interest', 'chan_indices', 'chans_of_interest');
fprintf('\n数据已保存至: %s\n', save_path);
fprintf('data 维度: %s (被试 × 条件 × 电极 × 时间点)\n', mat2str(size(data)));

%% ========================================================================
%% ========================= Part 2: 画波形图（多电极）====================
%% ========================================================================
% 以下画图部分使用保存好的 data，可以直接 load 后运行
% load(fullfile(file_path, 'all_data.mat'));

colors_correct = {'b', [0 0.6 0], 'r'};  % A=蓝, B=绿, C=红
nChans = length(chans_of_interest);

%% ---- 2.1 所有电极三种正确条件对比波形（子图拼接）----
figure('Name', '多电极三种正确条件对比波形', 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
nRows = ceil(nChans / 3);
nCols = min(nChans, 3);

for ci = 1:nChans
    ch = chan_indices(ci);
    subplot(nRows, nCols, ci);
    hold on; set(gca, 'YDir', 'reverse');
    for c = 1:3
        plot(EEG.times, squeeze(mean(data(:, c, ch, :), 1)), ...
            'Color', colors_correct{c}, 'LineWidth', 1.5);
    end
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
    xlabel('ms'); ylabel('\muV');
    if ci == 1
        legend('A-正确', 'B-正确', 'C-正确', 'Location', 'best', 'FontSize', 8);
    end
    box off;
end
sgtitle('三种刺激条件组平均波形 (正确反应)', 'fontsize', 15, 'FontWeight', 'bold');

%% ---- 2.2 如果包含错误条件，所有电极正确 vs 错误对比 ----
if include_error && nCond >= 6
    figure('Name', '多电极正确 vs 错误对比', 'NumberTitle', 'off', ...
        'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
    for ci = 1:nChans
        ch = chan_indices(ci);
        subplot(nRows, nCols, ci);
        hold on; set(gca, 'YDir', 'reverse');
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
    sgtitle('正确 vs 错误反应对比 (实线=正确, 虚线=错误)', 'fontsize', 15, 'FontWeight', 'bold');
end

%% ---- 2.3 所有电极差异波 ----
figure('Name', '多电极差异波', 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
for ci = 1:nChans
    ch = chan_indices(ci);
    subplot(nRows, nCols, ci);
    hold on; set(gca, 'YDir', 'reverse');
    diff_BA = squeeze(mean(data(:,2,ch,:),1)) - squeeze(mean(data(:,1,ch,:),1));
    diff_CA = squeeze(mean(data(:,3,ch,:),1)) - squeeze(mean(data(:,1,ch,:),1));
    plot(EEG.times, diff_BA, 'Color', [0 0.6 0], 'LineWidth', 1.5);
    plot(EEG.times, diff_CA, '-r', 'LineWidth', 1.5);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    title(chans_of_interest{ci}, 'fontsize', 13, 'FontWeight', 'bold');
    xlabel('ms'); ylabel('\muV');
    if ci == 1
        legend('B-A', 'C-A', 'Location', 'best', 'FontSize', 9);
    end
    box off;
end
sgtitle('差异波 (B-A 和 C-A)', 'fontsize', 15, 'FontWeight', 'bold');

%% ========================================================================
%% ========================= Part 3: 地形图 ===============================
%% ========================================================================

%% ---- 3.1 选择感兴趣的 ERP 成分时间窗口 ----
% 根据你的波形图结果调整以下时间窗口参数
% 常见视觉搜索 ERP 成分及其典型时间窗口：
%   P1:   80 - 130 ms
%   N1:   130 - 200 ms
%   N2pc: 200 - 300 ms（对侧-同侧差异）
%   P3:   300 - 600 ms

%%% 请根据实际波形调整以下峰值潜伏期 %%%
P1_peak_ms = 100;   % P1 峰值潜伏期（毫秒），请根据波形图调整
N1_peak_ms = 170;   % N1 峰值潜伏期
N2_peak_ms = 250;   % N2/N2pc 峰值潜伏期
P3_peak_ms = 400;   % P3 峰值潜伏期

% 找到对应的时间点索引
P1_idx = find(EEG.times >= P1_peak_ms, 1, 'first');
N1_idx = find(EEG.times >= N1_peak_ms, 1, 'first');
N2_idx = find(EEG.times >= N2_peak_ms, 1, 'first');
P3_idx = find(EEG.times >= P3_peak_ms, 1, 'first');

%% ---- 3.2 所有正确条件总平均各成分地形图 ----
figure('Name', '各ERP成分地形图', 'NumberTitle', 'off');

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

%% ---- 3.3 各条件在 N2 时间窗口内的平均地形图 ----
N2_window_ms = 50;  % N2 前后各50ms
N2_start_idx = find(EEG.times >= (N2_peak_ms - N2_window_ms), 1, 'first');
N2_end_idx = find(EEG.times >= (N2_peak_ms + N2_window_ms), 1, 'first');

figure('Name', '各条件 N2 地形图', 'NumberTitle', 'off');
for c = 1:3
    subplot(1, 3, c);
    condition_data = squeeze(mean(mean(data(:, c, :, N2_start_idx:N2_end_idx), 1), 4));
    topoplot(condition_data, EEG.chanlocs, 'maplimits', 'maxmin');
    title(sprintf('%s N2 (%d±%dms)', Cond_names{c}, N2_peak_ms, N2_window_ms), 'fontsize', 11);
end

%% ---- 3.4 各条件在 P3 时间窗口内的平均地形图 ----
P3_window_ms = 100;  % P3 前后各100ms
P3_start_idx = find(EEG.times >= (P3_peak_ms - P3_window_ms), 1, 'first');
P3_end_idx = find(EEG.times >= (P3_peak_ms + P3_window_ms), 1, 'first');

figure('Name', '各条件 P3 地形图', 'NumberTitle', 'off');
for c = 1:3
    subplot(1, 3, c);
    condition_data = squeeze(mean(mean(data(:, c, :, P3_start_idx:P3_end_idx), 1), 4));
    topoplot(condition_data, EEG.chanlocs, 'maplimits', 'maxmin');
    title(sprintf('%s P3 (%d±%dms)', Cond_names{c}, P3_peak_ms, P3_window_ms), 'fontsize', 11);
end

%% ========================================================================
%% ========================= Part 4: 峰值与潜伏期测量 =====================
%% ========================================================================

%% ---- 4.1 自动峰值检测：N2 和 P3（多电极）----
% 定义检测时间窗口（毫秒）
N2_search_window = [150 350];   %%% N2 搜索窗口，请根据波形调整
P3_search_window = [250 600];   %%% P3 搜索窗口，请根据波形调整

% 找到对应的索引范围
N2_win_idx = find(EEG.times >= N2_search_window(1) & EEG.times <= N2_search_window(2));
P3_win_idx = find(EEG.times >= P3_search_window(1) & EEG.times <= P3_search_window(2));

% 为每个电极分别计算
for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};
    
    % 维度: 被试 × 条件
    N2_amp = zeros(nSubj, 3);
    N2_lat = zeros(nSubj, 3);
    P3_amp = zeros(nSubj, 3);
    P3_lat = zeros(nSubj, 3);
    
    for i = 1:nSubj
        for c = 1:3  % 只对三个正确条件
            wave = squeeze(data(i, c, ch, :));
            
            % N2: 在搜索窗口内找极小值（负波峰）
            [min_val, min_pos] = min(wave(N2_win_idx));
            N2_amp(i, c) = min_val;
            N2_lat(i, c) = EEG.times(N2_win_idx(min_pos));
            
            % P3: 在搜索窗口内找极大值（正波峰）
            [max_val, max_pos] = max(wave(P3_win_idx));
            P3_amp(i, c) = max_val;
            P3_lat(i, c) = EEG.times(P3_win_idx(max_pos));
        end
    end
    
    % 打印结果
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
    
    % 保存到总结构体中（用于后续导出）
    all_N2_amp{ci} = N2_amp;
    all_N2_lat{ci} = N2_lat;
    all_P3_amp{ci} = P3_amp;
    all_P3_lat{ci} = P3_lat;
end

%% ---- 4.2 平均振幅测量（多电极，用于统计分析更稳健）----
% 在成分峰值前后一个时间窗口内取平均振幅，比单点峰值更稳健
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

% 兼容旧变量（用第一个电极的数据）
N2_amp = all_N2_amp{1}; N2_lat = all_N2_lat{1};
P3_amp = all_P3_amp{1}; P3_lat = all_P3_lat{1};
N2_mean_amp = all_N2_mean_amp{1}; P3_mean_amp = all_P3_mean_amp{1};

%% ---- 4.3 多电极 N2/P3 平均振幅柱状图 ----
figure('Name', '多电极 N2 平均振幅', 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
for ci = 1:nChans
    subplot(nRows, nCols, ci); hold on;
    means_N2 = mean(all_N2_mean_amp{ci});
    se_N2 = std(all_N2_mean_amp{ci}) / sqrt(nSubj);
    bar_h = bar(means_N2);
    bar_h.FaceColor = 'flat';
    bar_h.CData = [0 0 1; 0 0.6 0; 1 0 0];
    errorbar(1:3, means_N2, se_N2, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', {'A', 'B', 'C'});
    ylabel('\muV');
    title(sprintf('%s N2 (%d±%dms)', chans_of_interest{ci}, N2_peak_ms, N2_window_ms), 'fontsize', 11);
    box off;
end
sgtitle('N2 平均振幅 (各电极)', 'fontsize', 14, 'FontWeight', 'bold');

figure('Name', '多电极 P3 平均振幅', 'NumberTitle', 'off', ...
    'Position', [50 50 1200 200*ceil(nChans/3)*1.2]);
for ci = 1:nChans
    subplot(nRows, nCols, ci); hold on;
    means_P3 = mean(all_P3_mean_amp{ci});
    se_P3 = std(all_P3_mean_amp{ci}) / sqrt(nSubj);
    bar_h = bar(means_P3);
    bar_h.FaceColor = 'flat';
    bar_h.CData = [0 0 1; 0 0.6 0; 1 0 0];
    errorbar(1:3, means_P3, se_P3, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', {'A', 'B', 'C'});
    ylabel('\muV');
    title(sprintf('%s P3 (%d±%dms)', chans_of_interest{ci}, P3_peak_ms, P3_window_ms), 'fontsize', 11);
    box off;
end
sgtitle('P3 平均振幅 (各电极)', 'fontsize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% ============ Part 5: 单个被试波峰和潜伏期手动测量（交互式）==============
%% ========================================================================
% 如果需要手动选择波峰，取消下面这段的注释

% Lat_Amp = nan(nSubj, 4);  % [N2_lat, P3_lat, N2_amp, P3_amp]
% for i = 1:nSubj
%     figure; hold on;
%     set(gca, 'YDir', 'reverse');
%     
%     % 当前被试所有正确条件平均波形（红色）
%     temp = squeeze(mean(data(i, 1:3, chan_idx, :), 2));
%     plot(EEG.times, temp, 'r', 'LineWidth', 1.5);
%     
%     % 组平均波形（黑色，作参考）
%     plot(EEG.times, squeeze(mean(mean(data(:, 1:3, chan_idx, :), 2), 1)), 'k', 'LineWidth', 1);
%     
%     title(sprintf('被试 %d (%s) - 红=个体, 黑=组平均 (请依次点击 N2 和 P3)', i, SubjFiles{i}));
%     legend('个体波形', '组平均波形');
%     xlabel('Latency (ms)'); ylabel('Amplitude (\muV)');
%     
%     % 手动选取 N2 和 P3 峰值
%     [x, ~] = ginput(2);
%     while length(x) < 2
%         disp('请选择两个波峰（先 N2 后 P3）');
%         [x, ~] = ginput(2);
%     end
%     
%     % 在选定点前后 ±20ms 范围内精确查找
%     search_range = 20;  % ms
%     
%     % N2 (极小值)
%     n2_range = find(EEG.times >= (x(1) - search_range) & EEG.times <= (x(1) + search_range));
%     [N2_val, N2_pos] = min(temp(n2_range));
%     N2_latency = EEG.times(n2_range(N2_pos));
%     
%     % P3 (极大值)
%     p3_range = find(EEG.times >= (x(2) - search_range) & EEG.times <= (x(2) + search_range));
%     [P3_val, P3_pos] = max(temp(p3_range));
%     P3_latency = EEG.times(p3_range(P3_pos));
%     
%     Lat_Amp(i, :) = [N2_latency, P3_latency, N2_val, P3_val];
%     close;
% end

%% ========================================================================
%% ======= Part 6: 逐时间点重复测量方差分析（三种正确条件对比）============
%% ========================================================================
% 注意：此部分需要 anova_rm 函数（来自 EEGLAB 插件或 File Exchange）

fprintf('\n====== 正在进行逐时间点重复测量方差分析（多电极）... ======\n');

nTimepoints = size(data, 4);

% 对每个电极分别做 ANOVA
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
    
    % FDR 校正
    try
        [p_fdr, ~] = fdr(P_vals, 0.05);
        fprintf('    %s FDR 校正阈值: p = %.6f\n', ch_name, p_fdr);
    catch
        p_fdr = 0.05;
    end
    
    % 保存结果
    all_F_vals{ci} = F_vals;
    all_P_vals{ci} = P_vals;
    all_p_fdr(ci) = p_fdr;
end

%% ---- 6.1 多电极 波形图 + p 值图 ----
figure('Name', '多电极逐时间点ANOVA', 'NumberTitle', 'off', ...
    'Position', [50 50 1200 350*ceil(nChans/3)*1.0]);

for ci = 1:nChans
    ch = chan_indices(ci);
    ch_name = chans_of_interest{ci};
    
    % 上排：波形
    subplot(2, nChans, ci); hold on;
    set(gca, 'YDir', 'reverse');
    plot(EEG.times, squeeze(mean(data(:,1,ch,:),1)), '-b', 'LineWidth', 1.2);
    plot(EEG.times, squeeze(mean(data(:,2,ch,:),1)), 'Color', [0 0.6 0], 'LineWidth', 1.2);
    plot(EEG.times, squeeze(mean(data(:,3,ch,:),1)), '-r', 'LineWidth', 1.2);
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    title(ch_name, 'fontsize', 12, 'FontWeight', 'bold');
    if ci == 1, ylabel('\muV'); end
    xlabel('ms');
    if ci == 1, legend('A','B','C', 'Location', 'best', 'FontSize', 7); end
    box off;
    
    % 下排：p 值
    subplot(2, nChans, nChans + ci); hold on;
    plot(EEG.times, all_P_vals{ci}, 'b', 'LineWidth', 1);
    line([EEG.times(1) EEG.times(end)], [0.05 0.05], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line([EEG.times(1) EEG.times(end)], [all_p_fdr(ci) all_p_fdr(ci)], 'Color', 'r', 'LineWidth', 1.5);
    ylim([0 0.1]);
    title(sprintf('p值 (FDR=%.4f)', all_p_fdr(ci)), 'fontsize', 10);
    if ci == 1, ylabel('p value'); end
    xlabel('ms');
    box off;
end
sgtitle('逐时间点重复测量方差分析 (各电极)', 'fontsize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% ============ Part 7: 导出数据（方便后续统计软件分析）====================
%% ========================================================================

% 导出平均振幅数据到 CSV（用于 SPSS / R / jamovi 等统计软件）
fprintf('\n====== 导出统计数据 ======\n');

% 提取被试姓名（去掉 _processed.set 后缀）作为标识
SubjNames = cell(nSubj, 1);
for si = 1:nSubj
    [~, SubjNames{si}, ~] = fileparts(SubjFiles{si});
end

% 为每个电极分别导出 CSV
csv_header = 'Subject,A_correct,B_correct,C_correct\n';

for ci = 1:nChans
    ch_name = chans_of_interest{ci};
    
    % N2 平均振幅
    fname = fullfile(file_path, sprintf('N2_mean_amplitude_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.4f,%.4f,%.4f\n', SubjNames{si}, ...
            all_N2_mean_amp{ci}(si,1), all_N2_mean_amp{ci}(si,2), all_N2_mean_amp{ci}(si,3));
    end
    fclose(fid);
    
    % P3 平均振幅
    fname = fullfile(file_path, sprintf('P3_mean_amplitude_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.4f,%.4f,%.4f\n', SubjNames{si}, ...
            all_P3_mean_amp{ci}(si,1), all_P3_mean_amp{ci}(si,2), all_P3_mean_amp{ci}(si,3));
    end
    fclose(fid);
    
    % N2 峰值潜伏期
    fname = fullfile(file_path, sprintf('N2_peak_latency_%s.csv', ch_name));
    fid = fopen(fname, 'w');
    fprintf(fid, csv_header);
    for si = 1:nSubj
        fprintf(fid, '%s,%.1f,%.1f,%.1f\n', SubjNames{si}, ...
            all_N2_lat{ci}(si,1), all_N2_lat{ci}(si,2), all_N2_lat{ci}(si,3));
    end
    fclose(fid);
    
    % P3 峰值潜伏期
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

fprintf('\n====== 所有分析完成！ ======\n');
