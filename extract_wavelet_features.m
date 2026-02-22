function features = extract_wavelet_features(EEG_epochs, params)
% EXTRACT_WAVELET_FEATURES 提取小波变换特征
%
% 输入:
%   EEG_epochs - [n_channels x n_timepoints x n_trials]
%   params     - 参数结构体
%
% 输出:
%   features   - [n_trials x n_features] 特征矩阵

    [n_channels, n_timepoints, n_trials] = size(EEG_epochs);
    
    wname = params.wavelet_name;
    level = params.wavelet_level;
    
    % 每层产生的特征: 能量、均值、标准差、熵
    n_feat_per_level = 4;
    n_features = n_channels * (level + 1) * n_feat_per_level;
    features = zeros(n_trials, n_features);
    
    for t = 1:n_trials
        feat_vec = [];
        for ch = 1:n_channels
            signal = double(EEG_epochs(ch, :, t));
            
            [C, L] = wavedec(signal, level, wname);
            
            % 逐层提取特征
            for lv = 1:level
                d = detcoef(C, L, lv); % 细节系数
                feat_vec = [feat_vec, compute_wavelet_stats(d)];
            end
            
            % 近似系数
            a = appcoef(C, L, wname, level);
            feat_vec = [feat_vec, compute_wavelet_stats(a)];
        end
        features(t, :) = feat_vec;
    end
end


function stats = compute_wavelet_stats(coeffs)
% 计算小波系数的统计特征
    energy = sum(coeffs.^2);
    avg    = mean(coeffs);
    sd     = std(coeffs);
    
    p = (coeffs.^2) / (sum(coeffs.^2) + eps);
    p(p == 0) = eps;
    ent = -sum(p .* log2(p));
    
    stats = [energy, avg, sd, ent];
end
