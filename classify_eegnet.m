function results = classify_eegnet(EEG_epochs, labels, cv_indices, params)
% CLASSIFY_EEGNET 使用EEGNet深度学习模型进行SSVEP分类
%
% EEGNet是一种紧凑的卷积神经网络,专门设计用于EEG信号的解码。
% 包含: 时间卷积 -> 深度可分离卷积(空间) -> 可分离卷积 -> 全连接
%
% 需要: MATLAB Deep Learning Toolbox
%
% 参考: Lawhern et al. (2018), "EEGNet: A Compact Convolutional Neural
%       Network for EEG-based Brain-Computer Interfaces"
%
% 输入:
%   EEG_epochs - [n_channels x n_timepoints x n_trials]
%   labels     - [n_trials x 1] 类别标签
%   cv_indices - [n_trials x 1] 交叉验证折号
%   params     - 参数结构体
%
% 输出:
%   results    - 分类结果结构体

    [n_channels, n_timepoints, n_trials] = size(EEG_epochs);
    K = params.K_fold;
    n_classes = params.n_classes;
    
    % EEGNet超参数
    F1 = 8;   % 时间滤波器数量
    D  = 2;   % 深度乘数
    F2 = F1 * D; % 可分离卷积滤波器数量
    kernel_length = round(params.fs / 2); % 时间卷积核长度 (约0.5秒)
    dropout_rate = 0.25;
    
    all_pred = zeros(size(labels));
    fold_acc = zeros(K, 1);
    confusion_total = zeros(n_classes, n_classes);
    
    for k = 1:K
        test_idx  = (cv_indices == k);
        train_idx = (cv_indices ~= k);
        
        % 准备数据: [height x width x channels x batch] = [channels x timepoints x 1 x trials]
        X_train = reshape(EEG_epochs(:, :, train_idx), [n_channels, n_timepoints, 1, sum(train_idx)]);
        Y_train = categorical(labels(train_idx));
        X_test  = reshape(EEG_epochs(:, :, test_idx),  [n_channels, n_timepoints, 1, sum(test_idx)]);
        Y_test  = labels(test_idx);
        
        % 数据归一化
        mu = mean(X_train, 4);
        sd = std(X_train, 0, 4);
        sd(sd == 0) = 1;
        X_train = (X_train - mu) ./ sd;
        X_test  = (X_test - mu) ./ sd;
        
        %% 构建EEGNet网络
        layers = [
            imageInputLayer([n_channels n_timepoints 1], 'Normalization', 'none', 'Name', 'input')
            
            % Block 1: 时间卷积
            convolution2dLayer([1 kernel_length], F1, 'Padding', 'same', 'Name', 'conv_temporal')
            batchNormalizationLayer('Name', 'bn1')
            
            % 深度可分离卷积 (空间滤波)
            groupedConvolution2dLayer([n_channels 1], 1, F1, 'Name', 'conv_spatial')
            batchNormalizationLayer('Name', 'bn2')
            eluLayer(1, 'Name', 'elu1')
            averagePooling2dLayer([1 4], 'Stride', [1 4], 'Name', 'pool1')
            dropoutLayer(dropout_rate, 'Name', 'drop1')
            
            % Block 2: 可分离卷积
            convolution2dLayer([1 16], F2, 'Padding', 'same', 'Name', 'conv_separable')
            batchNormalizationLayer('Name', 'bn3')
            eluLayer(1, 'Name', 'elu2')
            averagePooling2dLayer([1 8], 'Stride', [1 8], 'Name', 'pool2')
            dropoutLayer(dropout_rate, 'Name', 'drop2')
            
            % 分类层
            fullyConnectedLayer(n_classes, 'Name', 'fc')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];
        
        %% 训练选项
        options = trainingOptions('adam', ...
            'InitialLearnRate', 1e-3, ...
            'MaxEpochs', 100, ...
            'MiniBatchSize', min(32, sum(train_idx)), ...
            'Shuffle', 'every-epoch', ...
            'ValidationFrequency', 10, ...
            'L2Regularization', 1e-4, ...
            'GradientThreshold', 1, ...
            'Verbose', false, ...
            'Plots', 'none');
        
        %% 训练
        try
            net = trainNetwork(X_train, Y_train, layers, options);
            pred_cat = classify(net, X_test);
            pred_labels = double(pred_cat);
        catch ME
            fprintf('    [警告] EEGNet训练失败: %s\n', ME.message);
            fprintf('    使用备选方案: 全连接网络\n');
            
            % 备选方案: 简单的全连接网络
            X_train_flat = reshape(X_train, [], sum(train_idx))';
            X_test_flat  = reshape(X_test,  [], sum(test_idx))';
            
            layers_fc = [
                featureInputLayer(size(X_train_flat, 2), 'Name', 'input')
                fullyConnectedLayer(64, 'Name', 'fc1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                dropoutLayer(0.5, 'Name', 'drop1')
                fullyConnectedLayer(32, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(n_classes, 'Name', 'fc3')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            
            net = trainNetwork(X_train_flat, Y_train, layers_fc, options);
            pred_cat = classify(net, X_test_flat);
            pred_labels = double(pred_cat);
        end
        
        all_pred(test_idx) = pred_labels;
        fold_acc(k) = mean(pred_labels == Y_test);
        
        for i = 1:length(Y_test)
            confusion_total(Y_test(i), pred_labels(i)) = ...
                confusion_total(Y_test(i), pred_labels(i)) + 1;
        end
    end
    
    results.accuracy = mean(fold_acc);
    results.std_acc  = std(fold_acc);
    results.fold_acc = fold_acc;
    results.confusion_matrix = confusion_total;
    results.all_predictions = all_pred;
    results.all_labels = labels;
    
    for c = 1:n_classes
        TP = confusion_total(c, c);
        FP = sum(confusion_total(:, c)) - TP;
        FN = sum(confusion_total(c, :)) - TP;
        results.precision(c) = TP / (TP + FP + eps);
        results.recall(c)    = TP / (TP + FN + eps);
        results.f1_score(c)  = 2 * results.precision(c) * results.recall(c) / ...
                               (results.precision(c) + results.recall(c) + eps);
    end
    results.macro_f1 = mean(results.f1_score);
    
    fprintf('    准确率: %.2f%% (±%.2f%%)\n', results.accuracy*100, results.std_acc*100);
end
