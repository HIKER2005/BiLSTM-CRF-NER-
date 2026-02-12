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
# 方法一：FOOOF参数化频谱分解，从拟合的高斯峰中提取Alpha峰中心频率作为IAF
fooof_iaf_data = []
# 方法二：直接对原始功率谱平滑后在Alpha频段找峰值作为IAF
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

    # ===================== 方法一：FOOOF参数化频谱分析 =====================
    # FOOOF (fitting oscillations & one over f) 的核心原理：
    #   1. 将功率谱建模为：Power(f) = Aperiodic(f) + Σ Gaussian_peaks(f)
    #   2. Aperiodic(f) = b + log10(f^(-χ)) 是1/f非周期背景
    #   3. 每个振荡峰用高斯函数拟合：G(f) = a * exp(-(f-μ)²/(2σ²))
    #   4. peak_params_ 输出每个峰的 [μ(中心频率), a(功率), σ(带宽)]
    # 所以FOOOF提取IAF的正确方式是：从peak_params_中找Alpha频段内功率最大的峰的中心频率
    print("\n进行FOOOF参数化频谱分析...")

    # 创建FOOOF对象
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, min_peak_height=0.1)

    # 定义感兴趣的频率范围（Alpha波段及其周边）
    freq_range = [3, 30]

    # 存储当前被试的FOOOF IAF（6个通道）
    current_fooof_iaf = []
    # 初始化周期成分功率谱（用于可视化）
    periodic_spectrum = np.zeros_like(power_spectrum)

    # 对每个通道进行FOOOF拟合
    for ch in range(num_channels):
        try:
            # 拟合FOOOF模型到原始功率谱
            fm.fit(freqs, power_spectrum[ch], freq_range)

            # 提取周期成分功率谱（用于可视化）
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            ap_linear = 10 ** fm._ap_fit  # 非周期成分转回线性空间
            periodic_spectrum[ch, freq_mask] = power_spectrum[ch, freq_mask] - ap_linear
            periodic_spectrum[ch, freq_mask] = np.maximum(periodic_spectrum[ch, freq_mask], 0)

            # ★ FOOOF的核心价值：参数化高斯峰拟合 ★
            # peak_params_ = [[中心频率, 功率, 带宽], ...] 每行一个峰
            peak_params = fm.peak_params_

            # 在Alpha频段(8-13Hz)内寻找功率最大的高斯峰
            iaf = None
            if len(peak_params) > 0:
                alpha_peaks = peak_params[(peak_params[:, 0] >= 8) & (peak_params[:, 0] <= 13)]
                if len(alpha_peaks) > 0:
                    # 选择功率最大的峰，其中心频率即为IAF
                    iaf = alpha_peaks[np.argmax(alpha_peaks[:, 1]), 0]

            current_fooof_iaf.append(iaf if iaf is not None else np.nan)

            if iaf is not None:
                print(f"  通道 {dataname[ch]}: FOOOF IAF = {iaf:.2f} Hz "
                      f"(拟合R²={fm.r_squared_:.3f}, 检测到{len(peak_params)}个峰)")
            else:
                print(f"  通道 {dataname[ch]}: 未检测到Alpha峰 "
                      f"(拟合R²={fm.r_squared_:.3f}, 检测到{len(peak_params)}个峰)")

        except Exception as e:
            print(f"  通道 {dataname[ch]}: FOOOF拟合失败 - {e}")
            current_fooof_iaf.append(np.nan)
            periodic_spectrum[ch] = power_spectrum[ch]

    fooof_iaf_data.append(current_fooof_iaf)

    # ===================== 方法二：原始功率谱 + 平滑峰值检测 =====================

    # Savitzky-Golay滤波器：251点的窗口，2阶多项式
    power_spectrum_smooth_sg = savgol_filter(power_spectrum, 251, 2)
    # 高斯滤波：标准差为10的高斯核
    power_spectrum_smooth_gs = gaussian_filter1d(power_spectrum, 10)

    # 存储当前数据路径方法二的IAF
    current_method2_iaf_sg = []  # 方法二 SG平滑
    current_method2_iaf_gs = []  # 方法二 高斯平滑

    # 创建图形窗口，双列布局：左列方法二，右列方法一
    plt.figure(figsize=(16, 12))
    plt.suptitle("IAF对比: 方法一(FOOOF参数化拟合) vs 方法二(功率谱平滑+峰值检测)", y=1.02)

    # Alpha频段索引范围
    alpha_start_idx = 8 * num_samples // downSampleRate
    alpha_end_idx = 13 * num_samples // downSampleRate

    # 为每个通道创建子图进行可视化
    for ch in range(num_channels):
        print(f"\n-----------------IAF对比({dataname[ch]}通道)-----------------")

        # ===== 方法一：FOOOF参数化IAF（已在上面的循环中计算） =====
        fooof_iaf = current_fooof_iaf[ch]
        if not np.isnan(fooof_iaf):
            print(f"  方法一 FOOOF参数化 IAF: {fooof_iaf:.2f} Hz")
        else:
            print(f"  方法一 FOOOF参数化 IAF: 未检测到Alpha峰")

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

        # 差值
        if not np.isnan(fooof_iaf):
            diff_sg = abs(fooof_iaf - method2_sg_freq)
            diff_gs = abs(fooof_iaf - method2_gs_freq)
            print(f"  差值: FOOOF vs SG = {diff_sg:.2f} Hz | FOOOF vs 高斯 = {diff_gs:.2f} Hz")
        print(f"-----------------------结束-----------------------")

        # ===== 可视化 =====
        # 左列：方法二 - 原始功率谱 + 平滑
        ax1 = plt.subplot(num_channels, 2, ch * 2 + 1)
        plt.semilogy(freqs, power_spectrum[ch], color="blue", linewidth=1, label="原始")
        plt.semilogy(freqs, power_spectrum_smooth_sg[ch], color="red", linewidth=2, label="SG平滑")
        plt.semilogy(freqs, power_spectrum_smooth_gs[ch], color="orange", linewidth=2, label="高斯平滑")
        plt.axvline(x=method2_gs_freq, color="green", linestyle="--", alpha=0.7,
                    label=f"GS IAF={method2_gs_freq:.1f}Hz")
        plt.axvline(x=method2_sg_freq, color="red", linestyle=":", alpha=0.7,
                    label=f"SG IAF={method2_sg_freq:.1f}Hz")
        plt.xlim(5, 15)
        plt.ylim(1e-16, 1e-10)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.ylabel(f"{dataname[ch]}\nPower (V²/Hz)")
        if ch == 0:
            plt.title("方法二: 功率谱平滑 + 峰值检测")
        if ch == num_channels - 1:
            plt.xlabel("Frequency (Hz)")
        else:
            plt.tick_params(labelbottom=False)
        if ch == 0:
            plt.legend(fontsize=7, loc="upper right")

        # 右列：方法一 - FOOOF参数化分解
        ax2 = plt.subplot(num_channels, 2, ch * 2 + 2)
        # 绘制去除非周期成分后的周期成分功率谱
        plt.plot(freqs, periodic_spectrum[ch], color="blue", linewidth=1, label="周期成分(去1/f)")
        # 标注FOOOF检测到的IAF位置
        if not np.isnan(fooof_iaf):
            plt.axvline(x=fooof_iaf, color="green", linestyle="--", linewidth=2, alpha=0.8,
                        label=f"FOOOF IAF={fooof_iaf:.2f}Hz")
        plt.xlim(5, 15)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.ylabel(f"{dataname[ch]}\nPeriodic Power")
        if ch == 0:
            plt.title("方法一: FOOOF参数化分解")
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
    method2_iaf_sg_data.append(current_method2_iaf_sg)
    method2_iaf_gs_data.append(current_method2_iaf_gs)

# ===================== 汇总输出：两种方法IAF对比 =====================
print("\n" + "="*120)
print("AVG通道 IAF 汇总对比")
print("方法一: FOOOF参数化频谱分解（高斯峰拟合）    方法二: 原始功率谱 + 平滑峰值检测")
print("="*120)
print(f"{'序号':<6}{'被试':<30}{'FOOOF IAF':<14}{'SG平滑IAF':<14}{'高斯平滑IAF':<14}{'差值vs SG':<12}{'差值vs GS':<12}")
print("-"*120)

all_diff_sg = []
all_diff_gs = []
for i, path in enumerate(data_paths, 1):
    f_iaf = fooof_iaf_data[i-1][5]  # avg通道索引为5
    sg_iaf = method2_iaf_sg_data[i-1][5]
    gs_iaf = method2_iaf_gs_data[i-1][5]
    diff_sg = abs(f_iaf - sg_iaf) if not np.isnan(f_iaf) else np.nan
    diff_gs = abs(f_iaf - gs_iaf) if not np.isnan(f_iaf) else np.nan
    all_diff_sg.append(diff_sg)
    all_diff_gs.append(diff_gs)
    f_str = f"{f_iaf:.2f}" if not np.isnan(f_iaf) else "N/A"
    d_sg_str = f"{diff_sg:.2f}" if not np.isnan(diff_sg) else "N/A"
    d_gs_str = f"{diff_gs:.2f}" if not np.isnan(diff_gs) else "N/A"
    print(f"{i:<6}{os.path.basename(path):<30}{f_str:<14}{sg_iaf:<14.2f}{gs_iaf:<14.2f}{d_sg_str:<12}{d_gs_str:<12}")

print("-"*120)
print(f"{'平均差值':<72}{np.nanmean(all_diff_sg):<12.2f}{np.nanmean(all_diff_gs):<12.2f}")
print(f"{'最大差值':<72}{np.nanmax(all_diff_sg):<12.2f}{np.nanmax(all_diff_gs):<12.2f}")
print(f"{'最小差值':<72}{np.nanmin(all_diff_sg):<12.2f}{np.nanmin(all_diff_gs):<12.2f}")
print(f"{'标准差':<72}{np.nanstd(all_diff_sg):<12.2f}{np.nanstd(all_diff_gs):<12.2f}")
print("="*120)
