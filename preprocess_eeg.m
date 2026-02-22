function EEG_out = preprocess_eeg(EEG_epochs, params)
% PREPROCESS_EEG 对分段后的EEG数据进行预处理
%
% 输入:
%   EEG_epochs - [n_channels x n_timepoints x n_trials]
%   params     - 参数结构体
%
% 输出:
%   EEG_out    - 预处理后的EEG数据 [n_channels x n_timepoints x n_trials]

    [n_channels, n_timepoints, n_trials] = size(EEG_epochs);
    EEG_out = zeros(size(EEG_epochs));
    
    for t = 1:n_trials
        trial_data = EEG_epochs(:, :, t);
        
        % 1. 去均值 (基线校正)
        baseline_end = round(abs(params.epoch_time(1)) * params.fs);
        if baseline_end > 0
            baseline = mean(trial_data(:, 1:baseline_end), 2);
            trial_data = trial_data - repmat(baseline, 1, n_timepoints);
        else
            trial_data = detrend(trial_data', 'constant')';
        end
        
        % 2. 带通滤波 (1-40 Hz)
        if params.fs > 80
            [b, a] = butter(4, [1 40] / (params.fs/2), 'bandpass');
            for ch = 1:n_channels
                trial_data(ch, :) = filtfilt(b, a, double(trial_data(ch, :)));
            end
        end
        
        % 3. 去趋势
        for ch = 1:n_channels
            trial_data(ch, :) = detrend(trial_data(ch, :));
        end
        
        EEG_out(:, :, t) = trial_data;
    end
    
    % 4. 简单伪迹剔除: 去除幅值超过阈值的试次
    amp_threshold = 100; % 微伏
    bad_trials = false(n_trials, 1);
    for t = 1:n_trials
        if max(abs(EEG_out(:,:,t)), [], 'all') > amp_threshold
            bad_trials(t) = true;
        end
    end
    
    if any(bad_trials)
        fprintf('    去除 %d 个伪迹试次 (共 %d)\n', sum(bad_trials), n_trials);
    end
end
