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
# 方法一：FOOOF去除非周期成分 + 平滑 得出的IAF
method1_iaf_sg_data = []   # SG平滑
method1_iaf_gs_data = []   # 高斯平滑
# 方法二：直接对原始功率谱平滑 得出的IAF
method2_iaf_sg_data = []   # SG平滑
method2_iaf_gs_data = []   # 高斯平滑

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

    # ===================== FOOOF分析 + 提取周期成分 =====================
    print("\n进行FOOOF分析...")

    # 创建FOOOF对象
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, min_peak_height=0.1)

    # 定义感兴趣的频率范围（Alpha波段及其周边）
    freq_range = [3, 30]

    # 初始化周期成分功率谱（去除非周期1/f背景后的功率）
    periodic_spectrum = np.zeros_like(power_spectrum)

    # 对每个通道进行FOOOF拟合，提取周期成分
    for ch in range(num_channels):
        try:
            # 拟合FOOOF模型到原始功率谱
            fm.fit(freqs, power_spectrum[ch], freq_range)

            # 提取周期成分（去除非周期/1f背景）
            # fm._ap_fit 是非周期成分在对数空间的拟合
            # 周期成分 = 原始功率 - 非周期功率（线性空间减法）
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            ap_linear = 10 ** fm._ap_fit  # 非周期成分转回线性空间
            periodic_spectrum[ch, freq_mask] = power_spectrum[ch, freq_mask] - ap_linear
            periodic_spectrum[ch, freq_mask] = np.maximum(periodic_spectrum[ch, freq_mask], 0)

            print(f"  通道 {dataname[ch]}: FOOOF拟合成功，已提取周期成分")

        except Exception as e:
            print(f"  通道 {dataname[ch]}: FOOOF拟合失败 - {e}，使用原始功率谱")
            periodic_spectrum[ch] = power_spectrum[ch]  # 拟合失败时退回原始功率谱

    # ===================== 平滑处理 =====================

    # --- 方法二：直接对原始功率谱做平滑 ---
    # Savitzky-Golay滤波器：251点的窗口，2阶多项式
    power_spectrum_smooth_sg = savgol_filter(power_spectrum, 251, 2)
    # 高斯滤波：标准差为10的高斯核
    power_spectrum_smooth_gs = gaussian_filter1d(power_spectrum, 10)

    # --- 方法一：对FOOOF周期成分（去除非周期背景后）做平滑 ---
    periodic_smooth_sg = savgol_filter(periodic_spectrum, 251, 2)
    periodic_smooth_gs = gaussian_filter1d(periodic_spectrum, 10)

    # 存储当前数据路径各方法的IAF
    current_method1_iaf_sg = []  # 方法一 SG平滑
    current_method1_iaf_gs = []  # 方法一 高斯平滑
    current_method2_iaf_sg = []  # 方法二 SG平滑
    current_method2_iaf_gs = []  # 方法二 高斯平滑

    # 创建图形窗口，双列布局：左列方法二，右列方法一
    plt.figure(figsize=(16, 12))
    plt.suptitle("IAF对比: 方法二(原始功率谱+平滑) vs 方法一(FOOOF去非周期+平滑)", y=1.02)

    # Alpha频段索引范围
    alpha_start_idx = 8 * num_samples // downSampleRate
    alpha_end_idx = 13 * num_samples // downSampleRate

    # 为每个通道创建子图进行可视化
    for ch in range(num_channels):
        print(f"\n-----------------IAF对比({dataname[ch]}通道)-----------------")

        # ===== 方法二：直接对原始功率谱平滑 =====
        # SG平滑 IAF
        max_power_idx = (
                np.argmax(power_spectrum_smooth_sg[ch, alpha_start_idx:alpha_end_idx])
                + alpha_start_idx
        )
        method2_sg_freq = freqs[max_power_idx]
        current_method2_iaf_sg.append(method2_sg_freq)
        print(f"  方法二 SG平滑 IAF: {method2_sg_freq:.2f} Hz")

        # 高斯平滑 IAF
        max_power_idx = (
                np.argmax(power_spectrum_smooth_gs[ch, alpha_start_idx:alpha_end_idx])
                + alpha_start_idx
        )
        method2_gs_freq = freqs[max_power_idx]
        current_method2_iaf_gs.append(method2_gs_freq)
        print(f"  方法二 高斯平滑 IAF: {method2_gs_freq:.2f} Hz")

        # ===== 方法一：FOOOF去除非周期成分后平滑 =====
        # SG平滑 IAF
        max_power_idx = (
                np.argmax(periodic_smooth_sg[ch, alpha_start_idx:alpha_end_idx])
                + alpha_start_idx
        )
        method1_sg_freq = freqs[max_power_idx]
        current_method1_iaf_sg.append(method1_sg_freq)
        print(f"  方法一 FOOOF+SG平滑 IAF: {method1_sg_freq:.2f} Hz")

        # 高斯平滑 IAF
        max_power_idx = (
                np.argmax(periodic_smooth_gs[ch, alpha_start_idx:alpha_end_idx])
                + alpha_start_idx
        )
        method1_gs_freq = freqs[max_power_idx]
        current_method1_iaf_gs.append(method1_gs_freq)
        print(f"  方法一 FOOOF+高斯平滑 IAF: {method1_gs_freq:.2f} Hz")

        # 差值
        diff_sg = abs(method1_sg_freq - method2_sg_freq)
        diff_gs = abs(method1_gs_freq - method2_gs_freq)
        print(f"  SG平滑差值: {diff_sg:.2f} Hz | 高斯平滑差值: {diff_gs:.2f} Hz")
        print(f"-----------------------结束-----------------------")

        # ===== 可视化 =====
        # 左列：方法二 - 原始功率谱 + 平滑
        ax1 = plt.subplot(num_channels, 2, ch * 2 + 1)
        plt.semilogy(freqs, power_spectrum[ch], color="blue", linewidth=1, label="原始")
        plt.semilogy(freqs, power_spectrum_smooth_sg[ch], color="red", linewidth=2, label="SG平滑")
        plt.semilogy(freqs, power_spectrum_smooth_gs[ch], color="orange", linewidth=2, label="高斯平滑")
        plt.axvline(x=method2_gs_freq, color="green", linestyle="--", alpha=0.7, label=f"IAF={method2_gs_freq:.1f}Hz")
        plt.xlim(5, 15)
        plt.ylim(1e-16, 1e-10)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.ylabel(f"{dataname[ch]}\nPower (V²/Hz)")
        if ch == 0:
            plt.title("方法二: 原始功率谱 + 平滑")
        if ch == num_channels - 1:
            plt.xlabel("Frequency (Hz)")
        else:
            plt.tick_params(labelbottom=False)
        if ch == 0:
            plt.legend(fontsize=7, loc="upper right")

        # 右列：方法一 - FOOOF周期成分 + 平滑
        ax2 = plt.subplot(num_channels, 2, ch * 2 + 2)
        plt.plot(freqs, periodic_spectrum[ch], color="blue", linewidth=1, label="周期成分")
        plt.plot(freqs, periodic_smooth_sg[ch], color="red", linewidth=2, label="SG平滑")
        plt.plot(freqs, periodic_smooth_gs[ch], color="orange", linewidth=2, label="高斯平滑")
        plt.axvline(x=method1_gs_freq, color="green", linestyle="--", alpha=0.7, label=f"IAF={method1_gs_freq:.1f}Hz")
        plt.xlim(5, 15)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.ylabel(f"{dataname[ch]}\nPeriodic Power")
        if ch == 0:
            plt.title("方法一: FOOOF周期成分 + 平滑")
        if ch == num_channels - 1:
            plt.xlabel("Frequency (Hz)")
        else:
            plt.tick_params(labelbottom=False)
        if ch == 0:
            plt.legend(fontsize=7, loc="upper right")

    # 调整子图布局，确保不重叠
    plt.tight_layout()
    # 显示图形
    plt.show()

    # 存储当前被试的IAF结果
    method1_iaf_sg_data.append(current_method1_iaf_sg)
    method1_iaf_gs_data.append(current_method1_iaf_gs)
    method2_iaf_sg_data.append(current_method2_iaf_sg)
    method2_iaf_gs_data.append(current_method2_iaf_gs)

# ===================== 汇总输出：两种方法IAF对比 =====================
print("\n" + "="*120)
print("AVG通道 IAF 汇总对比")
print("方法一: FOOOF去除非周期成分 + 平滑    方法二: 直接原始功率谱 + 平滑")
print("="*120)
print(f"{'序号':<6}{'被试':<30}{'方法一SG':<12}{'方法一GS':<12}{'方法二SG':<12}{'方法二GS':<12}{'差值SG':<10}{'差值GS':<10}")
print("-"*120)

all_diff_sg = []
all_diff_gs = []
for i, path in enumerate(data_paths, 1):
    m1_sg = method1_iaf_sg_data[i-1][5]  # avg通道索引为5
    m1_gs = method1_iaf_gs_data[i-1][5]
    m2_sg = method2_iaf_sg_data[i-1][5]
    m2_gs = method2_iaf_gs_data[i-1][5]
    diff_sg = abs(m1_sg - m2_sg)
    diff_gs = abs(m1_gs - m2_gs)
    all_diff_sg.append(diff_sg)
    all_diff_gs.append(diff_gs)
    print(f"{i:<6}{os.path.basename(path):<30}{m1_sg:<12.2f}{m1_gs:<12.2f}{m2_sg:<12.2f}{m2_gs:<12.2f}{diff_sg:<10.2f}{diff_gs:<10.2f}")

print("-"*120)
print(f"{'平均差值':<72}{np.nanmean(all_diff_sg):<10.2f}{np.nanmean(all_diff_gs):<10.2f}")
print(f"{'最大差值':<72}{np.nanmax(all_diff_sg):<10.2f}{np.nanmax(all_diff_gs):<10.2f}")
print(f"{'最小差值':<72}{np.nanmin(all_diff_sg):<10.2f}{np.nanmin(all_diff_gs):<10.2f}")
print(f"{'标准差':<72}{np.nanstd(all_diff_sg):<10.2f}{np.nanstd(all_diff_gs):<10.2f}")
print("="*120)
