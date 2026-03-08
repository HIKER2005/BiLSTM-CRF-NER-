#!/usr/bin/env python3
"""
Professional Scientific Figures for ERP Visual Search Study
Hypothesis: Stimulus A optimizes visual search performance more than B and C
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Global style settings
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {'A': '#E64B35', 'B': '#4DBBD5', 'C': '#00A087'}
COLOR_LIST = [COLORS['A'], COLORS['B'], COLORS['C']]
COND_LABELS = ['A', 'B', 'C']

# ============================================================
# Data
# ============================================================
# Behavioral data (10 subjects × 3 conditions)
acc_A = np.array([79.17, 85.83, 77.5, 75.83, 77.5, 80.83, 60.89, 85.83, 79.17, 80])
acc_B = np.array([63.33, 72.5, 76.67, 59.17, 82.5, 53.33, 73.33, 69.17, 62.5, 75.83])
acc_C = np.array([61.67, 75, 65.83, 55.83, 74.17, 46.67, 66.11, 70, 54.17, 77.5])

rt_A = np.array([1052.93, 929.87, 1030.86, 1090.78, 1064.96, 1186.88, 1144.08, 1087.67, 1087.75, 1020.54])
rt_B = np.array([1164.48, 1106.42, 1191.62, 1200.97, 1227.08, 1305.02, 1233.39, 1186.55, 1263.67, 1112.5])
rt_C = np.array([1158.66, 1153.52, 1278.88, 1215.61, 1235.7, 1302.71, 1300.77, 1242.93, 1268.78, 1188.09])

dp_A = np.array([1.629, 2.184, 1.515, 1.72, 1.526, 1.764, 0.553, 2.147, 1.625, 1.806])
dp_B = np.array([0.681, 1.205, 1.458, 0.605, 1.878, 0.169, 1.252, 1.004, 0.65, 1.813])
dp_C = np.array([0.612, 1.351, 0.817, 0.262, 1.353, -0.169, 0.839, 1.05, 0.212, 1.561])

# EEG ROI data (10 subjects × 3 conditions)
n2_A = np.array([-0.7136, -2.4264, -2.4977, -1.58, 0.2046, -2.4518, -0.982, 6.5011, -1.1916, -1.0591])
n2_B = np.array([-1.5733, -1.3852, -1.2393, -1.1006, 1.3451, -3.9794, -1.073, -0.4501, -1.1712, -1.681])
n2_C = np.array([-0.9812, -1.5905, -2.6002, -1.9756, -1.8674, -1.3861, -1.2298, 1.6493, -1.3386, -0.3058])

p3_A = np.array([0.794, -1.5683, -0.9722, 1.2479, 0.7767, -0.0365, -0.417, 11.5699, -0.0692, -0.0076])
p3_B = np.array([-0.2022, -0.5261, -3.3552, -1.8521, 2.9047, -0.2866, -2.2362, -3.3749, -1.0619, -0.134])
p3_C = np.array([0.529, -2.7457, -2.0961, 0.9872, 1.5118, -1.0833, -1.5726, 3.6788, -0.8347, 1.3431])

n2lat_A = np.array([274, 260, 252, 266, 274, 262, 250, 256, 278, 256])
n2lat_B = np.array([272, 278, 350, 264, 280, 254, 250, 270, 280, 270])
n2lat_C = np.array([274, 300, 350, 266, 322, 254, 236, 254, 284, 236])

p3lat_A = np.array([388, 566, 282, 288, 354, 340, 346, 440, 588, 328])
p3lat_B = np.array([378, 554, 280, 280, 532, 338, 282, 356, 264, 334])
p3lat_C = np.array([370, 314, 278, 276, 600, 330, 348, 428, 272, 324])

n_subj = 10

# ============================================================
# Statistical functions
# ============================================================
def rm_anova_oneway(data_matrix):
    """One-way repeated measures ANOVA. data_matrix: (n_subjects, n_conditions)"""
    n, k = data_matrix.shape
    grand_mean = data_matrix.mean()
    cond_means = data_matrix.mean(axis=0)
    subj_means = data_matrix.mean(axis=1)
    SS_cond = n * np.sum((cond_means - grand_mean)**2)
    SS_subj = k * np.sum((subj_means - grand_mean)**2)
    SS_total = np.sum((data_matrix - grand_mean)**2)
    SS_error = SS_total - SS_cond - SS_subj
    df_cond = k - 1
    df_error = (k - 1) * (n - 1)
    MS_cond = SS_cond / df_cond
    MS_error = SS_error / df_error
    F = MS_cond / MS_error
    p = 1 - stats.f.cdf(F, df_cond, df_error)
    eta2 = SS_cond / (SS_cond + SS_error)
    return F, p, eta2, df_cond, df_error


def posthoc_paired(a, b, n_comparisons=3):
    """Paired t-test with Bonferroni correction."""
    t, p_raw = stats.ttest_rel(a, b)
    d = np.mean(a - b) / np.std(a - b, ddof=1)
    p_bonf = min(p_raw * n_comparisons, 1.0)
    return t, p_raw, p_bonf, d


def sig_str(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def add_significance_bracket(ax, x1, x2, y, p, dh=0.02, barh=0.015):
    """Draw significance bracket between bars."""
    lw = 0.8
    s = sig_str(p)
    if s == 'n.s.':
        return
    ax_ylim = ax.get_ylim()
    range_y = ax_ylim[1] - ax_ylim[0]
    y_abs = y
    bh = barh * range_y
    ax.plot([x1, x1, x2, x2], [y_abs, y_abs + bh, y_abs + bh, y_abs],
            lw=lw, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, y_abs + bh, s, ha='center', va='bottom',
            fontsize=9, fontweight='bold')


# ============================================================
# Figure 1: Behavioral Results
# ============================================================
def create_figure1():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    fig.subplots_adjust(wspace=0.38)

    datasets = [
        (np.column_stack([acc_A, acc_B, acc_C]), 'Accuracy (%)', 'Accuracy'),
        (np.column_stack([rt_A, rt_B, rt_C]), 'Reaction Time (ms)', 'RT'),
        (np.column_stack([dp_A, dp_B, dp_C]), "d' (Sensitivity)", "d'"),
    ]

    panel_labels = ['a', 'b', 'c']

    for idx, (data, ylabel, title) in enumerate(datasets):
        ax = axes[idx]
        means = data.mean(axis=0)
        sems = data.std(axis=0, ddof=1) / np.sqrt(n_subj)
        F, p_anova, eta2, df1, df2 = rm_anova_oneway(data)

        x = np.arange(3)
        bars = ax.bar(x, means, width=0.55, color=COLOR_LIST, edgecolor='black',
                      linewidth=0.6, alpha=0.85, zorder=3)
        ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black',
                    capsize=4, capthick=1, elinewidth=1, zorder=4)

        for i in range(3):
            jitter = np.random.normal(0, 0.06, n_subj)
            ax.scatter(x[i] + jitter, data[:, i], color=COLOR_LIST[i],
                       edgecolor='white', s=18, linewidth=0.5, zorder=5, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Post-hoc comparisons
        pairs = [(0, 1), (0, 2), (1, 2)]
        pair_names = ['A vs B', 'A vs C', 'B vs C']
        y_max = max(means + sems) * 1.02
        bracket_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06

        for pi, (c1, c2) in enumerate(pairs):
            _, p_raw, p_bonf, d = posthoc_paired(data[:, c1], data[:, c2])
            y_bracket = y_max + bracket_offset * pi
            add_significance_bracket(ax, c1, c2, y_bracket, p_bonf)

        ax.set_ylim(ax.get_ylim()[0], y_max + bracket_offset * 3.5)

        anova_txt = f'F({df1},{df2})={F:.2f}, p={p_anova:.3f}\n' \
                    f'$\\eta_p^2$={eta2:.3f}'
        ax.text(0.98, 0.98, anova_txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

        ax.text(-0.15, 1.08, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    fig.suptitle('Behavioral Results', fontsize=13, fontweight='bold', y=1.02)
    fig.savefig('/workspace/Figure1_Behavioral_Results.png')
    fig.savefig('/workspace/Figure1_Behavioral_Results.pdf')
    plt.close(fig)
    print('Figure 1 saved.')

    # Print behavioral statistics
    for name, data in [('Accuracy', np.column_stack([acc_A, acc_B, acc_C])),
                        ('RT', np.column_stack([rt_A, rt_B, rt_C])),
                        ("d'", np.column_stack([dp_A, dp_B, dp_C]))]:
        F, p, eta2, df1, df2 = rm_anova_oneway(data)
        print(f'\n{name}: F({df1},{df2})={F:.3f}, p={p:.4f}, partial η²={eta2:.3f}')
        for c1, c2, label in [(0,1,'A-B'), (0,2,'A-C'), (1,2,'B-C')]:
            t, p_raw, p_bonf, d = posthoc_paired(data[:, c1], data[:, c2])
            print(f'  {label}: t={t:.3f}, p_raw={p_raw:.4f}, p_bonf={p_bonf:.4f}, d={d:.3f} {sig_str(p_bonf)}')


# ============================================================
# Figure 2: ERP ROI Results
# ============================================================
def create_figure2():
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.8))
    fig.subplots_adjust(wspace=0.45)

    datasets = [
        (np.column_stack([n2_A, n2_B, n2_C]),
         'N2 Mean Amplitude (μV)', f'N2 (FCz+Fz+Cz)\n200–300 ms'),
        (np.column_stack([p3_A, p3_B, p3_C]),
         'P3 Mean Amplitude (μV)', f'P3 (Pz+Cz)\n350–550 ms'),
        (np.column_stack([n2lat_A, n2lat_B, n2lat_C]),
         'N2 Peak Latency (ms)', f'N2 Latency'),
        (np.column_stack([p3lat_A, p3lat_B, p3lat_C]),
         'P3 Peak Latency (ms)', f'P3 Latency'),
    ]
    panel_labels = ['a', 'b', 'c', 'd']

    for idx, (data, ylabel, title) in enumerate(datasets):
        ax = axes[idx]
        means = data.mean(axis=0)
        sems = data.std(axis=0, ddof=1) / np.sqrt(n_subj)
        F, p_anova, eta2, df1, df2 = rm_anova_oneway(data)

        x = np.arange(3)
        bars = ax.bar(x, means, width=0.55, color=COLOR_LIST, edgecolor='black',
                      linewidth=0.6, alpha=0.85, zorder=3)
        ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black',
                    capsize=4, capthick=1, elinewidth=1, zorder=4)

        for i in range(3):
            jitter = np.random.normal(0, 0.06, n_subj)
            ax.scatter(x[i] + jitter, data[:, i], color=COLOR_LIST[i],
                       edgecolor='white', s=15, linewidth=0.4, zorder=5, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Significance brackets
        y_range = max(means + sems) - min(means - sems)
        y_max = max(means + sems) + y_range * 0.05
        bracket_offset = y_range * 0.08
        for pi, (c1, c2) in enumerate([(0,1), (0,2), (1,2)]):
            _, _, p_bonf, _ = posthoc_paired(data[:, c1], data[:, c2])
            add_significance_bracket(ax, c1, c2, y_max + bracket_offset * pi, p_bonf)
        ax.set_ylim(ax.get_ylim()[0], y_max + bracket_offset * 3.5)

        anova_txt = f'F={F:.2f}, p={p_anova:.3f}'
        ax.text(0.98, 0.98, anova_txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.6))

        ax.text(-0.2, 1.15, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    fig.suptitle('ERP ROI Results', fontsize=13, fontweight='bold', y=1.05)
    fig.savefig('/workspace/Figure2_ERP_ROI_Results.png')
    fig.savefig('/workspace/Figure2_ERP_ROI_Results.pdf')
    plt.close(fig)
    print('Figure 2 saved.')


# ============================================================
# Figure 3: Comprehensive Multi-Panel Summary
# ============================================================
def create_figure3():
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)

    # Row 1: Behavioral (Accuracy, RT, d')
    beh_data = [
        (np.column_stack([acc_A, acc_B, acc_C]), 'Accuracy (%)', 'a'),
        (np.column_stack([rt_A, rt_B, rt_C]), 'RT (ms)', 'b'),
        (np.column_stack([dp_A, dp_B, dp_C]), "d'", 'c'),
    ]
    for col, (data, ylabel, label) in enumerate(beh_data):
        ax = fig.add_subplot(gs[0, col])
        _plot_bar_with_stats(ax, data, ylabel, label)

    # Row 2: ERP amplitude
    erp_amp_data = [
        (np.column_stack([n2_A, n2_B, n2_C]), 'N2 Amp (μV)\nFCz+Fz+Cz', 'd'),
        (np.column_stack([p3_A, p3_B, p3_C]), 'P3 Amp (μV)\nPz+Cz', 'e'),
    ]
    for col, (data, ylabel, label) in enumerate(erp_amp_data):
        ax = fig.add_subplot(gs[1, col])
        _plot_bar_with_stats(ax, data, ylabel, label)

    # Row 2, col 3: ERP latency (grouped bar)
    ax = fig.add_subplot(gs[1, 2])
    _plot_grouped_latency(ax)
    ax.text(-0.2, 1.12, 'f', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    # Row 3: Correlations
    corr_data = [
        (acc_A - acc_C, n2_A - n2_C, 'ΔAccuracy (A−C, %)', 'ΔN2 Amp (A−C, μV)', 'g'),
        (rt_A - rt_C, p3_A - p3_C, 'ΔRT (A−C, ms)', 'ΔP3 Amp (A−C, μV)', 'h'),
        (dp_A - dp_C, p3_A - p3_C, "Δd' (A−C)", 'ΔP3 Amp (A−C, μV)', 'i'),
    ]
    for col, (x_data, y_data, xlabel, ylabel, label) in enumerate(corr_data):
        ax = fig.add_subplot(gs[2, col])
        _plot_correlation(ax, x_data, y_data, xlabel, ylabel, label)

    fig.suptitle('Comprehensive Results Summary', fontsize=14, fontweight='bold', y=0.98)
    fig.savefig('/workspace/Figure3_Comprehensive_Summary.png')
    fig.savefig('/workspace/Figure3_Comprehensive_Summary.pdf')
    plt.close(fig)
    print('Figure 3 saved.')


def _plot_bar_with_stats(ax, data, ylabel, panel_label):
    means = data.mean(axis=0)
    sems = data.std(axis=0, ddof=1) / np.sqrt(n_subj)
    F, p_anova, eta2, df1, df2 = rm_anova_oneway(data)
    x = np.arange(3)
    ax.bar(x, means, width=0.55, color=COLOR_LIST, edgecolor='black',
           linewidth=0.5, alpha=0.85, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black',
                capsize=3, capthick=0.8, elinewidth=0.8, zorder=4)
    for i in range(3):
        jitter = np.random.normal(0, 0.05, n_subj)
        ax.scatter(x[i] + jitter, data[:, i], color=COLOR_LIST[i],
                   edgecolor='white', s=12, linewidth=0.3, zorder=5, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    y_range = max(means + sems) - min(means - sems)
    y_max = max(means + sems) + y_range * 0.05
    bracket_offset = y_range * 0.08
    for pi, (c1, c2) in enumerate([(0,1), (0,2), (1,2)]):
        _, _, p_bonf, _ = posthoc_paired(data[:, c1], data[:, c2])
        add_significance_bracket(ax, c1, c2, y_max + bracket_offset * pi, p_bonf)
    ax.set_ylim(ax.get_ylim()[0], y_max + bracket_offset * 3.5)

    s = sig_str(p_anova)
    ax.set_title(f'F={F:.1f}, p={p_anova:.3f} {s}', fontsize=8, pad=4)
    ax.text(-0.2, 1.12, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


def _plot_grouped_latency(ax):
    n2_data = np.column_stack([n2lat_A, n2lat_B, n2lat_C])
    p3_data = np.column_stack([p3lat_A, p3lat_B, p3lat_C])
    x = np.arange(3)
    w = 0.3
    n2_means = n2_data.mean(axis=0)
    n2_sems = n2_data.std(axis=0, ddof=1) / np.sqrt(n_subj)
    p3_means = p3_data.mean(axis=0)
    p3_sems = p3_data.std(axis=0, ddof=1) / np.sqrt(n_subj)
    ax.bar(x - w/2, n2_means, w, color='#3C5488', edgecolor='black',
           linewidth=0.5, alpha=0.8, label='N2', zorder=3)
    ax.bar(x + w/2, p3_means, w, color='#F39B7F', edgecolor='black',
           linewidth=0.5, alpha=0.8, label='P3', zorder=3)
    ax.errorbar(x - w/2, n2_means, yerr=n2_sems, fmt='none', ecolor='black',
                capsize=3, capthick=0.8, elinewidth=0.8, zorder=4)
    ax.errorbar(x + w/2, p3_means, yerr=p3_sems, fmt='none', ecolor='black',
                capsize=3, capthick=0.8, elinewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS)
    ax.set_ylabel('Peak Latency (ms)', fontsize=9)
    ax.set_title('Peak Latency', fontsize=9, pad=4)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_correlation(ax, x, y, xlabel, ylabel, panel_label):
    ax.scatter(x, y, c='#3C5488', s=30, edgecolor='white', linewidth=0.5, zorder=3)
    r, p = stats.pearsonr(x, y)
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, '--', color='#E64B35',
                linewidth=1.2, alpha=0.8, zorder=2)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    s = sig_str(p)
    ax.set_title(f'r={r:.3f}, p={p:.3f} {s}', fontsize=8, pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.2, 1.12, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


# ============================================================
# Figure 4: Radar Chart - Condition Comparison
# ============================================================
def create_figure4_radar():
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    categories = ['Accuracy', 'Speed\n(1/RT)', "d'", 'P3 Amp', '−N2 Amp']

    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

    acc_all = np.array([acc_A.mean(), acc_B.mean(), acc_C.mean()])
    speed_all = 1000 / np.array([rt_A.mean(), rt_B.mean(), rt_C.mean()])
    dp_all = np.array([dp_A.mean(), dp_B.mean(), dp_C.mean()])
    p3_all = np.array([p3_A.mean(), p3_B.mean(), p3_C.mean()])
    neg_n2_all = -np.array([n2_A.mean(), n2_B.mean(), n2_C.mean()])

    values = np.column_stack([
        normalize(acc_all), normalize(speed_all), normalize(dp_all),
        normalize(p3_all), normalize(neg_n2_all)
    ])

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for i, (cond, color) in enumerate(zip(COND_LABELS, COLOR_LIST)):
        vals = values[i].tolist() + [values[i][0]]
        ax.plot(angles, vals, 'o-', linewidth=1.5, color=color, label=cond, markersize=4)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
    ax.set_title('Multi-Dimensional Performance Profile', fontsize=11,
                 fontweight='bold', pad=20)

    fig.savefig('/workspace/Figure4_Radar_Chart.png')
    fig.savefig('/workspace/Figure4_Radar_Chart.pdf')
    plt.close(fig)
    print('Figure 4 saved.')


# ============================================================
# Figure 5: ML Classification Results
# ============================================================
def create_figure5_ml():
    print('\n====== Machine Learning Classification ======')
    # Build feature matrix: each row = one subject-condition sample
    features_list = []
    labels_list = []
    groups_list = []

    for subj_idx in range(n_subj):
        for cond_idx, cond_label in enumerate(['A', 'B', 'C']):
            feat = [
                [acc_A, acc_B, acc_C][cond_idx][subj_idx],
                [rt_A, rt_B, rt_C][cond_idx][subj_idx],
                [dp_A, dp_B, dp_C][cond_idx][subj_idx],
                [n2_A, n2_B, n2_C][cond_idx][subj_idx],
                [p3_A, p3_B, p3_C][cond_idx][subj_idx],
                [n2lat_A, n2lat_B, n2lat_C][cond_idx][subj_idx],
                [p3lat_A, p3lat_B, p3lat_C][cond_idx][subj_idx],
            ]
            features_list.append(feat)
            labels_list.append(cond_idx)
            groups_list.append(subj_idx)

    X = np.array(features_list)
    y = np.array(labels_list)
    groups = np.array(groups_list)

    feature_names = ['Accuracy', 'RT', "d'", 'N2 Amp', 'P3 Amp', 'N2 Lat', 'P3 Lat']

    classifiers = {
        'SVM (Linear)': SVC(kernel='linear', C=1.0),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'Logistic Reg.': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'Gradient Boost': GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42),
    }

    logo = LeaveOneGroupOut()
    results = {}

    for name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        y_pred_all = np.zeros_like(y)
        for train_idx, test_idx in logo.split(X, y, groups):
            pipe.fit(X[train_idx], y[train_idx])
            y_pred_all[test_idx] = pipe.predict(X[test_idx])
        acc = accuracy_score(y, y_pred_all) * 100
        results[name] = acc
        print(f'  {name:20s}: {acc:.1f}%')

    # Sort by accuracy
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    names = list(sorted_results.keys())
    accs = list(sorted_results.values())

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(names))
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))

    bars = ax.barh(x, accs, height=0.6, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.axvline(x=33.33, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Chance (33.3%)')

    for i, (bar, acc_val) in enumerate(zip(bars, accs)):
        ax.text(acc_val + 1, i, f'{acc_val:.1f}%', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Classification Accuracy (%)', fontsize=11)
    ax.set_xlim(0, max(accs) + 15)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('9-Algorithm Classification Comparison\n(Leave-One-Subject-Out CV, Behavioral + EEG Features)',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    fig.savefig('/workspace/Figure5_ML_Classification.png')
    fig.savefig('/workspace/Figure5_ML_Classification.pdf')
    plt.close(fig)
    print('Figure 5 saved.')

    # Feature importance from Random Forest
    pipe_rf = Pipeline([('scaler', StandardScaler()),
                         ('clf', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42))])
    pipe_rf.fit(X, y)
    importances = pipe_rf.named_steps['clf'].feature_importances_
    sorted_idx = np.argsort(importances)

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.barh(np.arange(len(feature_names)), importances[sorted_idx],
             color='#4DBBD5', edgecolor='black', linewidth=0.5)
    ax2.set_yticks(np.arange(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax2.set_xlabel('Feature Importance', fontsize=11)
    ax2.set_title('Random Forest Feature Importance', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.savefig('/workspace/Figure6_Feature_Importance.png')
    fig2.savefig('/workspace/Figure6_Feature_Importance.pdf')
    plt.close(fig2)
    print('Figure 6 saved.')

    return sorted_results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    print('='*60)
    print(' Generating Professional Scientific Figures')
    print('='*60)

    create_figure1()
    create_figure2()
    create_figure3()
    create_figure4_radar()
    ml_results = create_figure5_ml()

    print('\n' + '='*60)
    print(' All figures generated successfully!')
    print('='*60)
    print('\nOutput files:')
    print('  Figure1_Behavioral_Results.png/pdf')
    print('  Figure2_ERP_ROI_Results.png/pdf')
    print('  Figure3_Comprehensive_Summary.png/pdf')
    print('  Figure4_Radar_Chart.png/pdf')
    print('  Figure5_ML_Classification.png/pdf')
    print('  Figure6_Feature_Importance.png/pdf')
