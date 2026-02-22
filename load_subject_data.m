function [EEG_epochs, labels, chan_locs] = load_subject_data(data_path, sub_id, params)
% LOAD_SUBJECT_DATA 加载单个被试的EEG数据并按标记进行分段
%
% 输入:
%   data_path - 数据文件夹路径
%   sub_id    - 被试编号
%   params    - 参数结构体
%
% 输出:
%   EEG_epochs - [n_channels x n_timepoints x n_trials] 分段后的EEG数据
%   labels     - [n_trials x 1] 类别标签 (1=A刺激, 2=B刺激, 3=C刺激)
%   chan_locs   - 通道位置信息
%
% ====== 请根据实际数据格式修改此函数 ======

    % --- 方式1: 从.mat文件加载 ---
    filename = fullfile(data_path, sprintf('subject%02d.mat', sub_id));
    if exist(filename, 'file')
        data = load(filename);
        
        % 假设 .mat 文件包含:
        %   data.EEG    - [n_channels x n_timepoints_total] 连续EEG数据
        %   data.events - [n_events x 2] 列1=事件标记, 列2=事件时间点(样本索引)
        %   data.chan_locs - 通道位置信息 (可选)
        
        if isfield(data, 'EEG') && isfield(data, 'events')
            EEG_raw = data.EEG;
            events  = data.events;
            chan_locs = [];
            if isfield(data, 'chan_locs')
                chan_locs = data.chan_locs;
            end
            
            n_channels = size(EEG_raw, 1);
            epoch_samples = round(params.epoch_time * params.fs);
            n_epoch_points = epoch_samples(2) - epoch_samples(1) + 1;
            
            stim_idx = find(ismember(events(:,1), params.markers));
            n_trials = length(stim_idx);
            
            EEG_epochs = zeros(n_channels, n_epoch_points, n_trials);
            labels = zeros(n_trials, 1);
            
            for t = 1:n_trials
                event_marker = events(stim_idx(t), 1);
                event_sample = events(stim_idx(t), 2);
                
                start_sample = event_sample + epoch_samples(1);
                end_sample   = event_sample + epoch_samples(2);
                
                if start_sample >= 1 && end_sample <= size(EEG_raw, 2)
                    EEG_epochs(:, :, t) = EEG_raw(:, start_sample:end_sample);
                end
                
                switch event_marker
                    case 11, labels(t) = 1; % A刺激
                    case 21, labels(t) = 2; % B刺激
                    case 31, labels(t) = 3; % C刺激
                end
            end
            
            valid_trials = labels > 0;
            EEG_epochs = EEG_epochs(:, :, valid_trials);
            labels = labels(valid_trials);
            return;
        end
    end
    
    % --- 方式2: 从.set文件加载 (EEGLAB格式) ---
    filename_set = fullfile(data_path, sprintf('subject%02d.set', sub_id));
    if exist(filename_set, 'file')
        EEG = pop_loadset(filename_set);
        chan_locs = EEG.chanlocs;
        
        epoch_samples = round(params.epoch_time * EEG.srate);
        n_epoch_points = epoch_samples(2) - epoch_samples(1) + 1;
        n_channels = EEG.nbchan;
        
        stim_events = [];
        for e = 1:length(EEG.event)
            evt_type = EEG.event(e).type;
            if isnumeric(evt_type)
                marker = evt_type;
            else
                marker = str2double(evt_type);
            end
            if ismember(marker, params.markers)
                stim_events = [stim_events; marker, round(EEG.event(e).latency)];
            end
        end
        
        n_trials = size(stim_events, 1);
        EEG_epochs = zeros(n_channels, n_epoch_points, n_trials);
        labels = zeros(n_trials, 1);
        
        for t = 1:n_trials
            start_sample = stim_events(t,2) + epoch_samples(1);
            end_sample   = stim_events(t,2) + epoch_samples(2);
            
            if start_sample >= 1 && end_sample <= size(EEG.data, 2)
                EEG_epochs(:, :, t) = EEG.data(:, start_sample:end_sample);
            end
            
            switch stim_events(t,1)
                case 11, labels(t) = 1;
                case 21, labels(t) = 2;
                case 31, labels(t) = 3;
            end
        end
        
        valid_trials = labels > 0;
        EEG_epochs = EEG_epochs(:, :, valid_trials);
        labels = labels(valid_trials);
        return;
    end
    
    % --- 方式3: 已分段的数据 ---
    filename_epoch = fullfile(data_path, sprintf('subject%02d_epochs.mat', sub_id));
    if exist(filename_epoch, 'file')
        data = load(filename_epoch);
        EEG_epochs = data.EEG_epochs;
        labels = data.labels;
        chan_locs = [];
        if isfield(data, 'chan_locs')
            chan_locs = data.chan_locs;
        end
        return;
    end
    
    % --- 方式4: 生成模拟数据 (用于测试代码) ---
    fprintf('  [警告] 未找到被试%d的数据文件，生成模拟SSVEP数据用于代码测试\n', sub_id);
    [EEG_epochs, labels, chan_locs] = generate_simulated_ssvep(params);
end


function [EEG_epochs, labels, chan_locs] = generate_simulated_ssvep(params)
% 生成模拟SSVEP数据用于代码调试和测试
    n_channels = 16;
    n_epoch_points = round((params.epoch_time(2) - params.epoch_time(1)) * params.fs) + 1;
    n_trials_per_class = 30;
    n_trials = n_trials_per_class * params.n_classes;
    
    t = linspace(params.epoch_time(1), params.epoch_time(2), n_epoch_points);
    
    EEG_epochs = zeros(n_channels, n_epoch_points, n_trials);
    labels = zeros(n_trials, 1);
    
    occipital_channels = round(n_channels*0.6):n_channels;
    
    trial_idx = 0;
    for c = 1:params.n_classes
        freq = params.stim_freqs(c);
        for tr = 1:n_trials_per_class
            trial_idx = trial_idx + 1;
            labels(trial_idx) = c;
            
            noise = 5 * randn(n_channels, n_epoch_points);
            
            for h = 1:params.n_harmonics
                amp = 3 / h;
                phase = 2 * pi * rand();
                ssvep_signal = amp * sin(2 * pi * h * freq * t + phase);
                for ch = occipital_channels
                    noise(ch, :) = noise(ch, :) + ssvep_signal * (0.5 + 0.5*rand());
                end
            end
            
            EEG_epochs(:, :, trial_idx) = noise;
        end
    end
    
    shuffle_idx = randperm(n_trials);
    EEG_epochs = EEG_epochs(:, :, shuffle_idx);
    labels = labels(shuffle_idx);
    
    chan_locs = [];
end
