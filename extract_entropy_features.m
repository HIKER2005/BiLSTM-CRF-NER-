function features = extract_entropy_features(EEG_epochs, params)
% EXTRACT_ENTROPY_FEATURES 提取多种熵特征 (ENT)
%
% 输入:
%   EEG_epochs - [n_channels x n_timepoints x n_trials]
%   params     - 参数结构体
%
% 输出:
%   features   - [n_trials x n_features] 特征矩阵
%
% 包含: 样本熵, 模糊熵, 排列熵, 谱熵

    [n_channels, ~, n_trials] = size(EEG_epochs);
    
    n_ent_types = 4; % 4种熵
    n_features = n_channels * n_ent_types;
    features = zeros(n_trials, n_features);
    
    for t = 1:n_trials
        feat_vec = [];
        for ch = 1:n_channels
            signal = double(EEG_epochs(ch, :, t));
            
            se  = sample_entropy(signal, 2, 0.2 * std(signal));
            fe  = fuzzy_entropy(signal, 2, 0.2 * std(signal));
            pe  = permutation_entropy(signal, 3, 1);
            spe = spectral_entropy(signal, params.fs);
            
            feat_vec = [feat_vec, se, fe, pe, spe];
        end
        features(t, :) = feat_vec;
    end
    
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
end


function se = sample_entropy(x, m, r)
% 样本熵 (Sample Entropy)
    N = length(x);
    if N < m + 2
        se = 0;
        return;
    end
    
    count = zeros(1, 2);
    for dim = m:m+1
        templates = zeros(N - dim, dim);
        for i = 1:N-dim
            templates(i, :) = x(i:i+dim-1);
        end
        
        n_matches = 0;
        for i = 1:size(templates, 1)
            for j = i+1:size(templates, 1)
                if max(abs(templates(i,:) - templates(j,:))) < r
                    n_matches = n_matches + 1;
                end
            end
        end
        count(dim - m + 1) = n_matches;
    end
    
    if count(1) == 0 || count(2) == 0
        se = 0;
    else
        se = -log(count(2) / count(1));
    end
end


function fe = fuzzy_entropy(x, m, r)
% 模糊熵 (Fuzzy Entropy)
    N = length(x);
    if N < m + 2
        fe = 0;
        return;
    end
    
    phi = zeros(1, 2);
    for dim = m:m+1
        templates = zeros(N - dim, dim);
        for i = 1:N-dim
            templates(i, :) = x(i:i+dim-1);
            templates(i, :) = templates(i, :) - mean(templates(i, :));
        end
        
        n_templ = size(templates, 1);
        similarity_sum = 0;
        for i = 1:n_templ
            for j = i+1:n_templ
                d = max(abs(templates(i,:) - templates(j,:)));
                similarity_sum = similarity_sum + exp(-(d/r)^2);
            end
        end
        phi(dim - m + 1) = similarity_sum / (n_templ * (n_templ - 1) / 2 + eps);
    end
    
    if phi(1) == 0
        fe = 0;
    else
        fe = -log(phi(2) / (phi(1) + eps) + eps);
    end
end


function pe = permutation_entropy(x, m, tau)
% 排列熵 (Permutation Entropy)
    N = length(x);
    n_patterns = N - (m - 1) * tau;
    
    if n_patterns < 1
        pe = 0;
        return;
    end
    
    patterns = zeros(n_patterns, m);
    for i = 1:n_patterns
        idx = i:tau:i+(m-1)*tau;
        patterns(i, :) = x(idx);
    end
    
    [~, perm_indices] = sort(patterns, 2);
    
    [unique_perms, ~, ic] = unique(perm_indices, 'rows');
    counts = accumarray(ic, 1);
    probs = counts / sum(counts);
    
    pe = -sum(probs .* log2(probs + eps));
    pe = pe / log2(factorial(m)); % 归一化
end


function spe = spectral_entropy(x, fs)
% 谱熵 (Spectral Entropy)
    nfft = 2^nextpow2(length(x));
    [pxx, ~] = pwelch(x, [], [], nfft, fs);
    
    pxx_norm = pxx / (sum(pxx) + eps);
    pxx_norm(pxx_norm == 0) = eps;
    
    spe = -sum(pxx_norm .* log2(pxx_norm));
    spe = spe / log2(length(pxx)); % 归一化
end
