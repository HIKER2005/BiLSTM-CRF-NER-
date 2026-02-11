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

clear; clc; eeglab;

%% ========================= 参数设置（需要修改的部分）=========================
Subj = [1:10];                          %%% 被试编号，根据实际被试数量修改
file_path = 'D:\EEG_data\';            %%% 数据文件所在路径，需要修改为你的实际路径
file_suffix = '.set';                   %%% 文件后缀
file_prefix = '';                       %%% 文件前缀（如有），如 'sub'
file_postfix = '_preprocessed';         %%% 文件名后部分（如有），如 '_preprocessed'

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

% 感兴趣电极（根据你的电极布局修改）
% 常见选择：Cz, Pz, Oz, PO7, PO8 等
chan_of_interest = 'Cz';                %%% 感兴趣电极名称
chan_idx = [];                          % 稍后自动查找索引

%% ========================= Part 1: 复合标记创建与分段 =========================
fprintf('\n====== 开始处理数据 ======\n');

for i = 1:length(Subj)
    % 构建文件名，根据你的命名规则修改
    file_name = [file_prefix num2str(Subj(i)) file_postfix file_suffix];
    %%% 例如：'1_preprocessed.set' 或 'sub01.set'，请根据实际情况修改上面的前后缀
    fprintf('\n--- 正在处理被试 %d: %s ---\n', Subj(i), file_name);
    
    EEG = pop_loadset('filename', file_name, 'filepath', file_path);
    
    % 第一次循环时查找感兴趣电极的索引
    if i == 1
        for ch = 1:length(EEG.chanlocs)
            if strcmpi(EEG.chanlocs(ch).labels, chan_of_interest)
                chan_idx = ch;
                break;
            end
        end
        if isempty(chan_idx)
            warning('未找到电极 %s，将使用第1个电极！请检查电极名称。', chan_of_interest);
            chan_idx = 1;
        else
            fprintf('找到电极 %s，索引编号为 %d\n', chan_of_interest, chan_idx);
        end
    end
    
    %% ---- 判断数据类型：连续 or 已分段 ----
    if ndims(EEG.data) == 2  % 连续数据，需要创建复合标记并分段
        fprintf('  检测到连续数据，正在创建复合标记...\n');
        
        nevents = length(EEG.event);
        trial_count = zeros(1, 6); % 记录各条件的试次数 [A正确,B正确,C正确,A错误,B错误,C错误]
        
        for ev = 1:nevents
            % 获取当前事件类型（统一转换为数值）
            current_type = EEG.event(ev).type;
            if ischar(current_type) || isstring(current_type)
                current_type_num = str2double(current_type);
            else
                current_type_num = current_type;
            end
            
            % 仅处理"有目标"标记（41）
            if current_type_num == target_marker
                
                % —— 向前查找：找到最近的刺激类型标记（11/21/31）——
                stim_type = NaN;
                for prev_ev = (ev - 1):-1:1
                    prev_type = EEG.event(prev_ev).type;
                    if ischar(prev_type) || isstring(prev_type)
                        prev_type_num = str2double(prev_type);
                    else
                        prev_type_num = prev_type;
                    end
                    
                    if ismember(prev_type_num, stim_markers)
                        stim_type = prev_type_num;
                        break;
                    end
                    % 如果遇到另一个搜索任务标记，说明序列断裂，停止查找
                    if ismember(prev_type_num, [target_marker, no_target_marker])
                        break;
                    end
                end
                
                % —— 向后查找：找到最近的反应标记（12/22/32）——
                resp_type = NaN;
                for next_ev = (ev + 1):nevents
                    next_type = EEG.event(next_ev).type;
                    if ischar(next_type) || isstring(next_type)
                        next_type_num = str2double(next_type);
                    else
                        next_type_num = next_type;
                    end
                    
                    if ismember(next_type_num, response_markers)
                        resp_type = next_type_num;
                        break;
                    end
                    % 如果遇到下一个刺激或搜索标记，说明该试次无反应或序列断裂
                    if ismember(next_type_num, [stim_markers, target_marker, no_target_marker])
                        break;
                    end
                end
                
                % —— 根据刺激类型和反应类型创建复合标记 ——
                % 编码规则：百位=刺激编号(1/2/3), 个位=反应编号(1=正确,2=错误,3=无反应)
                if ~isnan(stim_type)
                    stim_code = stim_type / 10;  % 11->1, 21->2, 31->3
                    
                    if resp_type == correct_marker
                        resp_code = 1;
                    elseif resp_type == incorrect_marker
                        resp_code = 2;
                    elseif resp_type == no_response_marker
                        resp_code = 3;
                    else
                        resp_code = 0; % 未知反应
                    end
                    
                    if resp_code > 0
                        new_marker = stim_code * 100 + resp_code;
                        EEG.event(ev).type = new_marker;
                        
                        % 统计试次数
                        if stim_code <= 3 && resp_code <= 2
                            trial_count((stim_code - 1) * 2 + resp_code) = ...
                                trial_count((stim_code - 1) * 2 + resp_code) + 1;
                        end
                    end
                end
            end
        end
        
        % 打印试次统计
        fprintf('  被试 %d 试次统计:\n', Subj(i));
        fprintf('    A正确: %d, B正确: %d, C正确: %d\n', trial_count(1), trial_count(3), trial_count(5));
        fprintf('    A错误: %d, B错误: %d, C错误: %d\n', trial_count(2), trial_count(4), trial_count(6));
        
        % 确定需要提取的所有复合标记
        all_markers_to_epoch = Cond_markers;
        
        % 分段：以复合标记时间点为 0 点
        EEG = pop_epoch(EEG, num2cell(all_markers_to_epoch), epoch_window, ...
            'newname', ['sub' num2str(Subj(i)) '_epoched'], 'epochinfo', 'yes');
        
        % 基线校正
        EEG = pop_rmbase(EEG, baseline_window);
        
        fprintf('  分段完成，共 %d 个 epoch\n', EEG.trials);
        
    else  % 数据已经分段（假设已经围绕 41 标记分段）
        fprintf('  检测到已分段数据（%d trials），将直接根据事件筛选...\n', EEG.trials);
        % 注意：如果数据已经分段但没有复合标记，你需要先在预处理阶段创建复合标记
        % 或者确保数据中已包含复合标记（101,201,301等）
    end
    
    %% ---- 按条件提取 epoch 并计算平均 ----
    for j = 1:nCond
        EEG_temp = EEG;
        
        % 根据复合标记选择特定条件的 epoch
        EEG_temp = pop_selectevent(EEG_temp, 'type', Cond_markers(j), ...
            'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
        
        fprintf('  条件 %s (标记 %d): %d 个 epoch\n', Cond_names{j}, Cond_markers(j), EEG_temp.trials);
        
        if EEG_temp.trials > 0
            % data 为 subj × cond × channel × timepoints 的四维数组
            data(i, j, :, :) = squeeze(mean(EEG_temp.data, 3));
        else
            warning('  被试 %d 条件 %s 无有效 epoch！', Subj(i), Cond_names{j});
            data(i, j, :, :) = zeros(EEG.nbchan, EEG.pnts);
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
save_path = [file_path 'all_data.mat'];
save(save_path, 'data', 'EEG', 'Subj', 'Cond_names', 'Cond_markers', 'chan_idx', 'chan_of_interest');
fprintf('\n数据已保存至: %s\n', save_path);
fprintf('data 维度: %s (被试 × 条件 × 电极 × 时间点)\n', mat2str(size(data)));

%% ========================================================================
%% ========================= Part 2: 画波形图 =============================
%% ========================================================================
% 以下画图部分使用保存好的 data，可以直接 load 后运行
% load([file_path 'all_data.mat']);

%% ---- 2.1 所有条件总平均波形图（感兴趣电极）----
figure('Name', '所有条件总平均波形', 'NumberTitle', 'off');
plot(EEG.times, squeeze(mean(mean(data(:, 1:3, chan_idx, :), 1), 2)), '-r', 'LineWidth', 1.5);
set(gca, 'YDir', 'reverse');  % 负极朝上
title(sprintf('Group-level %s waveform (all correct conditions)', chan_of_interest), 'fontsize', 14);
xlabel('Latency (ms)', 'fontsize', 14);
ylabel('Amplitude (\\muV)', 'fontsize', 14);
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');  % 刺激出现时刻的竖线
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');   % 零线
box off;

%% ---- 2.2 三种正确反应条件的波形对比图 ----
colors_correct = {'b', [0 0.6 0], 'r'};  % A=蓝, B=绿, C=红
figure('Name', '三种正确条件对比波形', 'NumberTitle', 'off');
hold on;
set(gca, 'YDir', 'reverse');
for c = 1:3
    plot(EEG.times, squeeze(mean(data(:, c, chan_idx, :), 1)), ...
        'Color', colors_correct{c}, 'LineWidth', 1.5);
end
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
legend('A刺激-正确', 'B刺激-正确', 'C刺激-正确', 'Location', 'best');
title(sprintf('Group-level %s waveforms by stimulus type (correct)', chan_of_interest), 'fontsize', 14);
xlabel('Latency (ms)', 'fontsize', 14);
ylabel('Amplitude (\\muV)', 'fontsize', 14);
box off;

%% ---- 2.3 如果包含错误条件，画正确 vs 错误对比 ----
if include_error && nCond >= 6
    figure('Name', '正确 vs 错误反应对比', 'NumberTitle', 'off');
    hold on;
    set(gca, 'YDir', 'reverse');
    
    % 正确条件（实线）
    for c = 1:3
        plot(EEG.times, squeeze(mean(data(:, c, chan_idx, :), 1)), ...
            'Color', colors_correct{c}, 'LineWidth', 1.5, 'LineStyle', '-');
    end
    % 错误条件（虚线）
    for c = 4:6
        plot(EEG.times, squeeze(mean(data(:, c, chan_idx, :), 1)), ...
            'Color', colors_correct{c - 3}, 'LineWidth', 1.5, 'LineStyle', '--');
    end
    
    line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
    legend('A-正确','B-正确','C-正确','A-错误','B-错误','C-错误', 'Location', 'best');
    title(sprintf('Group-level %s: Correct vs Incorrect', chan_of_interest), 'fontsize', 14);
    xlabel('Latency (ms)', 'fontsize', 14);
    ylabel('Amplitude (\\muV)', 'fontsize', 14);
    box off;
end

%% ---- 2.4 差异波（以 C-A 和 B-A 为例）----
figure('Name', '差异波', 'NumberTitle', 'off');
hold on;
set(gca, 'YDir', 'reverse');

diff_BA = squeeze(mean(data(:, 2, chan_idx, :), 1)) - squeeze(mean(data(:, 1, chan_idx, :), 1));
diff_CA = squeeze(mean(data(:, 3, chan_idx, :), 1)) - squeeze(mean(data(:, 1, chan_idx, :), 1));

plot(EEG.times, diff_BA, '-g', 'LineWidth', 1.5);
plot(EEG.times, diff_CA, '-r', 'LineWidth', 1.5);
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
legend('B - A', 'C - A', 'Location', 'best');
title(sprintf('Group-level difference waves at %s', chan_of_interest), 'fontsize', 14);
xlabel('Latency (ms)', 'fontsize', 14);
ylabel('Amplitude (\\muV)', 'fontsize', 14);
box off;

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

%% ---- 4.1 自动峰值检测：N2 和 P3 ----
% 定义检测时间窗口（毫秒）
N2_search_window = [150 350];   %%% N2 搜索窗口，请根据波形调整
P3_search_window = [250 600];   %%% P3 搜索窗口，请根据波形调整

% 找到对应的索引范围
N2_win_idx = find(EEG.times >= N2_search_window(1) & EEG.times <= N2_search_window(2));
P3_win_idx = find(EEG.times >= P3_search_window(1) & EEG.times <= P3_search_window(2));

% 为每个被试每个条件提取 N2 和 P3 的振幅和潜伏期
% 维度: 被试 × 条件
N2_amp = zeros(length(Subj), 3);
N2_lat = zeros(length(Subj), 3);
P3_amp = zeros(length(Subj), 3);
P3_lat = zeros(length(Subj), 3);

for i = 1:length(Subj)
    for c = 1:3  % 只对三个正确条件
        wave = squeeze(data(i, c, chan_idx, :));
        
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
fprintf('\n====== N2 振幅和潜伏期 (电极: %s) ======\n', chan_of_interest);
fprintf('条件\t\t平均振幅(uV)\t标准差\t\t平均潜伏期(ms)\t标准差\n');
for c = 1:3
    fprintf('%s\t%.2f\t\t%.2f\t\t%.1f\t\t%.1f\n', ...
        Cond_names{c}, mean(N2_amp(:,c)), std(N2_amp(:,c)), ...
        mean(N2_lat(:,c)), std(N2_lat(:,c)));
end

fprintf('\n====== P3 振幅和潜伏期 (电极: %s) ======\n', chan_of_interest);
fprintf('条件\t\t平均振幅(uV)\t标准差\t\t平均潜伏期(ms)\t标准差\n');
for c = 1:3
    fprintf('%s\t%.2f\t\t%.2f\t\t%.1f\t\t%.1f\n', ...
        Cond_names{c}, mean(P3_amp(:,c)), std(P3_amp(:,c)), ...
        mean(P3_lat(:,c)), std(P3_lat(:,c)));
end

%% ---- 4.2 平均振幅测量（用于统计分析更稳健）----
% 在成分峰值前后一个时间窗口内取平均振幅，比单点峰值更稳健
N2_mean_amp = zeros(length(Subj), 3);
P3_mean_amp = zeros(length(Subj), 3);

for i = 1:length(Subj)
    for c = 1:3
        N2_mean_amp(i, c) = mean(squeeze(data(i, c, chan_idx, N2_start_idx:N2_end_idx)));
        P3_mean_amp(i, c) = mean(squeeze(data(i, c, chan_idx, P3_start_idx:P3_end_idx)));
    end
end

%% ---- 4.3 柱状图 + 误差线（振幅对比）----
figure('Name', 'N2 和 P3 振幅柱状图', 'NumberTitle', 'off');

% N2 平均振幅柱状图
subplot(121); hold on;
means_N2 = mean(N2_mean_amp);
se_N2 = std(N2_mean_amp) / sqrt(length(Subj));
bar_handle = bar(means_N2);
bar_handle.FaceColor = 'flat';
bar_handle.CData = [0 0 1; 0 0.6 0; 1 0 0];  % A=蓝, B=绿, C=红
errorbar(1:3, means_N2, se_N2, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'A刺激', 'B刺激', 'C刺激'});
ylabel('Amplitude (\muV)', 'fontsize', 12);
title(sprintf('N2 Mean Amplitude (%d±%dms)', N2_peak_ms, N2_window_ms), 'fontsize', 13);
box off;

% P3 平均振幅柱状图
subplot(122); hold on;
means_P3 = mean(P3_mean_amp);
se_P3 = std(P3_mean_amp) / sqrt(length(Subj));
bar_handle = bar(means_P3);
bar_handle.FaceColor = 'flat';
bar_handle.CData = [0 0 1; 0 0.6 0; 1 0 0];
errorbar(1:3, means_P3, se_P3, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'A刺激', 'B刺激', 'C刺激'});
ylabel('Amplitude (\muV)', 'fontsize', 12);
title(sprintf('P3 Mean Amplitude (%d±%dms)', P3_peak_ms, P3_window_ms), 'fontsize', 13);
box off;

%% ---- 4.4 潜伏期柱状图 ----
figure('Name', 'N2 和 P3 潜伏期柱状图', 'NumberTitle', 'off');

subplot(121); hold on;
means_N2_lat = mean(N2_lat);
se_N2_lat = std(N2_lat) / sqrt(length(Subj));
bar_handle = bar(means_N2_lat);
bar_handle.FaceColor = 'flat';
bar_handle.CData = [0 0 1; 0 0.6 0; 1 0 0];
errorbar(1:3, means_N2_lat, se_N2_lat, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'A刺激', 'B刺激', 'C刺激'});
ylabel('Latency (ms)', 'fontsize', 12);
title('N2 Peak Latency', 'fontsize', 13);
box off;

subplot(122); hold on;
means_P3_lat = mean(P3_lat);
se_P3_lat = std(P3_lat) / sqrt(length(Subj));
bar_handle = bar(means_P3_lat);
bar_handle.FaceColor = 'flat';
bar_handle.CData = [0 0 1; 0 0.6 0; 1 0 0];
errorbar(1:3, means_P3_lat, se_P3_lat, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'A刺激', 'B刺激', 'C刺激'});
ylabel('Latency (ms)', 'fontsize', 12);
title('P3 Peak Latency', 'fontsize', 13);
box off;

%% ========================================================================
%% ============ Part 5: 单个被试波峰和潜伏期手动测量（交互式）==============
%% ========================================================================
% 如果需要手动选择波峰，取消下面这段的注释

% Lat_Amp = nan(length(Subj), 4);  % [N2_lat, P3_lat, N2_amp, P3_amp]
% for i = 1:length(Subj)
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
%     title(sprintf('被试 %d - 红=个体, 黑=组平均 (请依次点击 N2 和 P3)', Subj(i)));
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

fprintf('\n====== 正在进行逐时间点重复测量方差分析... ======\n');

clear F_vals P_vals
nTimepoints = size(data, 4);

F_vals = zeros(1, nTimepoints);
P_vals = ones(1, nTimepoints);  % 默认 p=1

for t = 1:nTimepoints
    % 提取所有被试三种正确条件在当前时间点的数据
    % 维度: 被试 × 3条件
    anova_data = squeeze(data(:, 1:3, chan_idx, t));
    
    try
        [p, table] = anova_rm(anova_data, 'off');
        P_vals(t) = p(1);
        F_vals(t) = table{2, 5};
    catch
        P_vals(t) = 1;
        F_vals(t) = 0;
    end
end

% FDR 校正
try
    [p_fdr, p_masked] = fdr(P_vals, 0.05);
    fprintf('FDR 校正阈值: p = %.6f\n', p_fdr);
catch
    warning('fdr 函数不可用，跳过 FDR 校正');
    p_fdr = 0.05;
    p_masked = P_vals < 0.05;
end

%% ---- 6.1 波形图 + p 值图 ----
figure('Name', '逐时间点方差分析结果', 'NumberTitle', 'off');

% 上图：三种条件波形
subplot(211); hold on;
set(gca, 'YDir', 'reverse');
plot(EEG.times, squeeze(mean(data(:, 1, chan_idx, :), 1)), '-b', 'LineWidth', 1.5);
plot(EEG.times, squeeze(mean(data(:, 2, chan_idx, :), 1)), 'Color', [0 0.6 0], 'LineWidth', 1.5);
plot(EEG.times, squeeze(mean(data(:, 3, chan_idx, :), 1)), '-r', 'LineWidth', 1.5);
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
legend('A刺激-正确', 'B刺激-正确', 'C刺激-正确', 'Location', 'best');
title(sprintf('三种刺激条件组平均波形 (%s)', chan_of_interest), 'fontsize', 14);
xlabel('Latency (ms)', 'fontsize', 12);
ylabel('Amplitude (\muV)', 'fontsize', 12);
xlim([epoch_window(1)*1000 epoch_window(2)*1000]);
box off;

% 下图：p 值
subplot(212); hold on;
plot(EEG.times, P_vals, 'b', 'LineWidth', 1);
line([EEG.times(1) EEG.times(end)], [0.05 0.05], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
line([EEG.times(1) EEG.times(end)], [p_fdr p_fdr], 'Color', [1 0 0], 'LineStyle', '-', 'LineWidth', 1.5);
legend('p value', 'p=0.05', sprintf('FDR threshold (p=%.4f)', p_fdr), 'Location', 'best');
title('逐时间点重复测量方差分析 p 值', 'fontsize', 14);
xlabel('Latency (ms)', 'fontsize', 12);
ylabel('p value', 'fontsize', 12);
xlim([epoch_window(1)*1000 epoch_window(2)*1000]);
ylim([0 0.1]);
box off;

%% ---- 6.2 F 值热图 ----
figure('Name', 'F值热图', 'NumberTitle', 'off');
hold on;
imagesc(EEG.times, 1, F_vals);
xlim([epoch_window(1)*1000 epoch_window(2)*1000]);
colorbar;
title('逐时间点重复测量方差分析 F 值', 'fontsize', 14);
xlabel('Latency (ms)', 'fontsize', 12);

% 在显著时间点上方添加标记
sig_times = EEG.times(P_vals < p_fdr);
if ~isempty(sig_times)
    plot(sig_times, ones(size(sig_times)) * 1.3, 'r*', 'MarkerSize', 3);
end

%% ========================================================================
%% ============ Part 7: 导出数据（方便后续统计软件分析）====================
%% ========================================================================

% 导出平均振幅数据到 CSV（用于 SPSS / R / jamovi 等统计软件）
fprintf('\n====== 导出统计数据 ======\n');

% N2 平均振幅
T_N2 = table();
T_N2.Subject = Subj(:);
for c = 1:3
    T_N2.(Cond_names{c}) = N2_mean_amp(:, c);
end
writetable(T_N2, [file_path 'N2_mean_amplitude.csv']);
fprintf('N2 平均振幅已导出至: %s\n', [file_path 'N2_mean_amplitude.csv']);

% P3 平均振幅
T_P3 = table();
T_P3.Subject = Subj(:);
for c = 1:3
    T_P3.(Cond_names{c}) = P3_mean_amp(:, c);
end
writetable(T_P3, [file_path 'P3_mean_amplitude.csv']);
fprintf('P3 平均振幅已导出至: %s\n', [file_path 'P3_mean_amplitude.csv']);

% N2 峰值潜伏期
T_N2_lat = table();
T_N2_lat.Subject = Subj(:);
for c = 1:3
    T_N2_lat.(Cond_names{c}) = N2_lat(:, c);
end
writetable(T_N2_lat, [file_path 'N2_peak_latency.csv']);

% P3 峰值潜伏期
T_P3_lat = table();
T_P3_lat.Subject = Subj(:);
for c = 1:3
    T_P3_lat.(Cond_names{c}) = P3_lat(:, c);
end
writetable(T_P3_lat, [file_path 'P3_peak_latency.csv']);

fprintf('\n====== 所有分析完成！ ======\n');
