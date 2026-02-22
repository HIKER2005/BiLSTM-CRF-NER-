function features = extract_psd_features(EEG_epochs, params)
% EXTRACT_PSD_FEATURES 提取功率谱密度特征
%
% 输入:
%   EEG_epochs - [n_channels x n_timepoints x n_trials]
%   params     - 参数结构体
%
% 输出:
%   features   - [n_trials x n_features] 特征矩阵

    [n_channels, n_timepoints, n_trials] = size(EEG_epochs);
    nfft = 2^nextpow2(n_timepoints);
    
    % 对SSVEP相关频带和谐波频率提取功率
    freq_resolution = params.fs / nfft;
    freqs_of_interest = [];
    for c = 1:params.n_classes
        for h = 1:params.n_harmonics
            f_center = params.stim_freqs(c) * h;
            freqs_of_interest = [freqs_of_interest, f_center];
        end
    end
    
    n_freq_features = length(freqs_of_interest);
    bandwidth = 1; % Hz, 每个频率点取 ±bandwidth 范围内的功率
    
    features = zeros(n_trials, n_channels * n_freq_features);
    
    for t = 1:n_trials
        feat_idx = 0;
        for ch = 1:n_channels
            signal = detrend(EEG_epochs(ch, :, t));
            [pxx, f] = pwelch(signal, [], [], nfft, params.fs);
            
            for fi = 1:n_freq_features
                f_target = freqs_of_interest(fi);
                freq_idx = (f >= f_target - bandwidth) & (f <= f_target + bandwidth);
                feat_idx = feat_idx + 1;
                features(t, feat_idx) = mean(pxx(freq_idx));
            end
        end
    end
    
    % 对数变换使特征分布更接近正态
    features = log10(features + eps);
end
