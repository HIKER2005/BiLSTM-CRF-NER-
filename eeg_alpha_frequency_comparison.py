import os
import re
import mne
import time
import fooof
from fooof import FOOOF
import numpy as np
# import pandas as pd
from NDFSysMNE import mneNDF
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib.pyplot import isinteractive
from NDFSysParser import FileNEF, FileNTP, FileNSF, FileNDF, FolderNHF, FolderNSF
from NDFSysUtility import FolderCatergory, NDFUtility
from ReadNDF import ReadNDFChannels, ReadOneChannel

base_path = r'C:\Users\lee23\Desktop\小论文返修\静息脑电'

# 创建空列表来存储数据路径
data_paths = []
FOOOF_iaf_gs_data = []
FOOOF_iaf_periodic_data = []
FOOOF_iaf_sg_data = []

# 直接处理所有符合条件的文件夹
for folder in sorted(os.listdir(base_path)):
    if re.match(r'\d{14}_.*?_静息', folder):
        folder_path = os.path.join(base_path, folder)

        # 获取子文件夹
        for subitem in os.listdir(folder_path):
            subpath = os.path.join(folder_path, subitem)
            if os.path.isdir(subpath):
                data_paths.append(subpath)
                print(f"数据路径: {subpath}")
                break

# 打印数组长度和内容
print(f"\n总共找到 {len(data_paths)} 个数据路径")
print("数据路径数组:", data_paths)

# 现在你可以循环读取这些路径了
print("\n开始循环读取:")
for i, path in enumerate(data_paths, 1):
    print(f"\n {path}")
    # 使用系统中文字体（推荐）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 电极位置配置：使用标准的10-20系统电极放置法
    montage = mne.channels.make_standard_montage("standard_1020")
    # 带通滤波参数：保留1-30Hz频段（包含Delta, Theta, Alpha, Beta波段）
    low_freq = 1
    high_freq = 30
    # 自动处理标志：是否自动检测坏道和进行ICA降噪
    isautobadChans = True
    isautoICA = True
    downSampleRate = 500  # 降采样率：将数据降采样到500Hz以减少计算量

    # mne加载NDF文件
    filepath = path  # 被试数据路径
    print("\n正在载入数据...")

    # 博瑞康py加载NDF文件
    # 创建mneNDF对象读取NDF格式的脑电数据
    # 第一个参数是数据文件夹路径，第二个参数是.nef配置文件路径
    ndfMneObj = mneNDF(filepath, filepath + r"\..\neuracle.nef")
    # 将NDF数据转换为MNE的Raw对象格式
    raw = ndfMneObj.read2MneRaw()

    # 获取数据信息：通道名称、采样点数等
    infoTest = raw.info
    channels = raw.ch_names
    timeLen = raw.n_times
    print(channels)

    # 电极定位：将通道名称与10-20系统标准位置对应
    raw.set_montage(montage, match_case=False)

    # 重参考：使用平均参考（所有电极的平均值作为参考）
    print("\n设置参考电极...")
    raw.set_eeg_reference(ref_channels="average")

    # 选择特定通道进行分析：主要选择枕区和顶枕区的电极
    # Oz（枕中线）、O1/O2（左/右枕区）、PO3/PO4（左/右顶枕区）
    # 这些区域对视觉处理和Alpha节律最敏感
    raw.pick(["Oz", "O1", "O2", "PO3", "PO4"])

    # 带通滤波器：保留1-30Hz频段，去除低频漂移和高频噪声
    print("\n进行滤波处理...")
    raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="iir",  # 使用IIR滤波器（巴特沃斯）
        iir_params=dict(order=3, ftype="butter"),  # 3阶巴特沃斯滤波器
    )

    # 陷波滤波器：去除50Hz工频干扰（中国电网频率）
    raw.notch_filter(freqs=50, method="iir")

    # 降采样：将数据降至500Hz，降低数据量和计算复杂度
    print("\n调整采样率...")
    raw.resample(sfreq=downSampleRate)

    # 坏道自动检测与插值处理
    if isautobadChans:
        print("\n进行坏导处理...")
        # 使用LOF（局部异常因子）算法检测坏道
        noisy_chslist = mne.preprocessing.find_bad_channels_lof(raw, n_neighbors=8)
        # 插值修复坏道数据
        raw.interpolate_bads(exclude=noisy_chslist)

    # ICA（独立成分分析）去除伪迹
    if isautoICA:
        print("\n进行ICA处理...")
        # 创建ICA对象，使用infomax算法
        ica = mne.preprocessing.ICA(
            n_components=None,  # 自动确定成分数量
            method="infomax",  # 使用infomax算法
            max_iter="auto"  # 自动确定最大迭代次数
        ).fit(raw)

        exclude_idx = []  # 存储需要去除的成分索引
        try:
            # 检测并去除肌肉伪迹（EMG）
            emg_indices, ecmg_scores = ica.find_bads_muscle(raw, threshold=0.95)
            exclude_idx.extend(emg_indices)
            # 检测并去除眼电伪迹（EOG）
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            exclude_idx.extend(eog_indices)
        except:
            pass  # 如果检测失败，跳过

        # 去除重复的索引
        exclude_idx = np.unique(exclude_idx)
        print(f"去除成分编号 = {exclude_idx}")
        # 应用ICA，去除伪迹成分
        ica.apply(raw, exclude=exclude_idx.tolist())

    # 获取处理后的数据：取最后5个通道的数据（实际只选择了5个通道，所以这里是全部数据）
    data, times = raw[-5:, :]  # 获取数据矩阵和时间点
    dataname = raw.ch_names[-5:]  # 获取通道名称

    # 计算所有通道的平均值（创建虚拟的"avg"通道）
    data_averages = np.average(data, axis=0)  # 沿通道轴计算平均值
    dataname = np.append(dataname, "avg")  # 在通道名列表中添加"avg"
    # 将平均值作为新的一行添加到数据中
    data = np.insert(data, 5, data_averages, axis=0)

    # 获取数据的维度信息
    num_channels = data.shape[0]  # 通道数量（现在是6个：5个原始+1个平均）
    num_samples = data.shape[1]  # 时间点数

    # ===================== PSD计算（FFT方法）=====================

    # 应用汉宁窗 (减少频谱泄漏)：为每个样本点乘以汉宁窗系数
    window = np.hanning(num_samples)  # 生成汉宁窗
    windowed_data = data * window[np.newaxis, :]  # 将窗函数应用于每个通道

    # 执行FFT (沿时间轴)：使用实FFT只计算正频率部分
    fft_result = np.fft.rfft(windowed_data, axis=1)  # 对每个通道进行FFT

    # 计算频率轴 (只含正频率)
    # 基于采样率和样本数计算对应的频率值
    freqs = np.fft.rfftfreq(num_samples, 1 / downSampleRate)

    # 计算功率谱密度 (单位: V²/Hz)
    n_fft = fft_result.shape[1]  # FFT结果的长度
    # 功率谱计算：|FFT|² / (采样率 × 窗函数能量)
    power_spectrum = (np.abs(fft_result) ** 2) / (downSampleRate * (window ** 2).sum())

    # 在计算完 power_spectrum 之后，平滑处理之前插入以下代码：
    # （找到 power_spectrum = ... 这一行后面）

    # ===================== FOOOF分析（在平滑之前）=====================
    print("\n进行FOOOF分析...")

    # 创建FOOOF对象
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, min_peak_height=0.1)

    # 定义感兴趣的频率范围（Alpha波段及其周边）
    freq_range = [3, 30]

    # 存储当前数据路径的FOOOF IAF结果（6个通道）
    current_fooof_iaf_periodic = []

    # 对每个通道进行FOOOF拟合
    for ch in range(num_channels):
        try:
            # 拟合FOOOF模型到原始功率谱
            fm.fit(freqs, power_spectrum[ch], freq_range)

            # 提取peak frequencies（周期性成分的频率）
            peak_params = fm.peak_params_  # [center_freq, power, bandwidth]

            # 在Alpha频段(8-13Hz)寻找最强的peak作为IAF
            iaf_periodic = None
            if len(peak_params) > 0:
                alpha_peaks = peak_params[(peak_params[:, 0] >= 8) & (peak_params[:, 0] <= 13)]
                if len(alpha_peaks) > 0:
                    # 选择功率最大的peak
                    iaf_periodic = alpha_peaks[np.argmax(alpha_peaks[:, 1]), 0]

            current_fooof_iaf_periodic.append(iaf_periodic if iaf_periodic else np.nan)

            print(
                f"  通道 {dataname[ch]}: IAF (FOOOF) = {iaf_periodic:.2f} Hz" if iaf_periodic else f"  通道 {dataname[ch]}: 未检测到Alpha峰")

        except Exception as e:
            print(f"  通道 {dataname[ch]}: FOOOF拟合失败 - {e}")
            current_fooof_iaf_periodic.append(np.nan)

    # 将当前数据的结果添加到总列表中
    FOOOF_iaf_periodic_data.append(current_fooof_iaf_periodic)

    # ===================== 继续原有的平滑处理 =====================

    # 平滑功率谱曲线，使用两种不同的平滑方法
    # Savitzky-Golay滤波器：251点的窗口，2阶多项式
    power_spectrum_smooth_sg = savgol_filter(power_spectrum, 251, 2)
    # 高斯滤波：标准差为10的高斯核
    power_spectrum_smooth_gs = gaussian_filter1d(power_spectrum, 10)
    # 存储当前数据路径的高斯平滑IAF
    current_iaf_gs = []

    # 创建图形窗口，设置标题
    plt.figure(figsize=(12, 8))
    plt.suptitle("Multi-channel Power Spectrum Analysis", y=1.02)

    # 为每个通道创建子图进行可视化
    for ch in range(num_channels):
        ax = plt.subplot(num_channels, 1, ch + 1)  # 创建子图，垂直排列

        # 在alpha频段（8-13Hz）内寻找功率最大的频率点
        # 计算原始数据的最大功率频率
        max_power_idx = (
                np.argmax(
                    power_spectrum[
                        ch, 8 * num_samples // downSampleRate: 13 * num_samples // downSampleRate
                    ]
                )
                + 8 * num_samples // downSampleRate  # 加上起始索引偏移
        )
        max_freq = freqs[max_power_idx]  # 获取对应的频率值
        print(f"\n-----------------功率最大频率({dataname[ch]}通道)-----------------")
        print(f"原始功率最大的频率（蓝色线）: {max_freq:.2f} Hz")

        # 计算Savitzky-Golay平滑后数据的最大功率频率
        max_power_idx = (
                np.argmax(
                    power_spectrum_smooth_sg[
                        ch, 8 * num_samples // downSampleRate: 13 * num_samples // downSampleRate
                    ]
                )
                + 8 * num_samples // downSampleRate
        )
        max_freq = freqs[max_power_idx]
        print(f"Savitzky-Golay平滑功率最大的频率（红色线）: {max_freq:.2f} Hz")

        # 计算高斯平滑后数据的最大功率频率
        max_power_idx = (
                np.argmax(
                    power_spectrum_smooth_gs[
                        ch, 8 * num_samples // downSampleRate: 13 * num_samples // downSampleRate
                    ]
                )
                + 8 * num_samples // downSampleRate
        )
        max_freq = freqs[max_power_idx]
        print(f"高斯平滑功率最大的频率（黄色线）: {max_freq:.2f} Hz")
        print(f"-----------------------结束-----------------------")
        # 存储高斯平滑的IAF
        current_iaf_gs.append(max_freq)

        # =====================
        # 4. 可视化设置
        # =====================
        # 绘制三种功率谱曲线：原始、SG平滑、高斯平滑
        plt.semilogy(freqs, power_spectrum[ch], color="blue", linewidth=1)  # 原始（蓝色）
        plt.semilogy(freqs, power_spectrum_smooth_sg[ch], color="red", linewidth=2)  # SG平滑（红色）
        plt.semilogy(freqs, power_spectrum_smooth_gs[ch], color="yellow", linewidth=2)  # 高斯平滑（黄色）

        # 坐标轴设置：聚焦alpha频段（5-15Hz）
        plt.xlim(5, 15)  # X轴范围
        plt.ylim(1e-16, 1e-10)  # Y轴范围（对数坐标）
        plt.grid(True, which="both", linestyle="--", alpha=0.6)  # 显示网格
        plt.ylabel(f"Channel {dataname[ch]}\nPower (V²/Hz)")  # Y轴标签

        # 只在最后一个子图显示X轴标签
        if ch == num_channels - 1:
            plt.xlabel("Frequency (Hz)")
        else:
            plt.tick_params(labelbottom=False)  # 隐藏其他子图的X轴标签

    # 调整子图布局，确保不重叠
    plt.tight_layout()
    # 显示图形
    plt.show()
    FOOOF_iaf_gs_data.append(current_iaf_gs)

# 输出所有数据的avg通道IAF对比
print("\n" + "="*80)
print("AVG通道IAF汇总（FOOOF vs 高斯平滑）")
print("="*80)
for i, path in enumerate(data_paths, 1):
    fooof_iaf = FOOOF_iaf_periodic_data[i-1][5]  # avg通道索引为5
    gs_iaf = FOOOF_iaf_gs_data[i-1][5]
    print(f"{i}. {os.path.basename(path)}")
    print(f"   FOOOF IAF: {fooof_iaf:.2f} Hz | 高斯平滑 IAF: {gs_iaf:.2f} Hz | 差值: {abs(fooof_iaf - gs_iaf):.2f} Hz")
