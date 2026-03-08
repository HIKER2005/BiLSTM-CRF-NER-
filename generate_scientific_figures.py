#!/usr/bin/env python3
"""
Professional Scientific Figures — Spacious Style (matching reference)
Hypothesis: Stimulus A optimizes visual search performance more than B and C
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Global style — spacious, clean, matching reference
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 13,
    'axes.linewidth': 1.0,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

COLORS = {'A': '#E64B35', 'B': '#4DBBD5', 'C': '#00A087'}
COLOR_LIST = [COLORS['A'], COLORS['B'], COLORS['C']]
COND_LABELS = ['A', 'B', 'C']

# ============================================================
# Data
# ============================================================
acc_A = np.array([79.17, 85.83, 77.5, 75.83, 77.5, 80.83, 60.89, 85.83, 79.17, 80])
acc_B = np.array([63.33, 72.5, 76.67, 59.17, 82.5, 53.33, 73.33, 69.17, 62.5, 75.83])
acc_C = np.array([61.67, 75, 65.83, 55.83, 74.17, 46.67, 66.11, 70, 54.17, 77.5])

rt_A = np.array([1052.93, 929.87, 1030.86, 1090.78, 1064.96, 1186.88, 1144.08, 1087.67, 1087.75, 1020.54])
rt_B = np.array([1164.48, 1106.42, 1191.62, 1200.97, 1227.08, 1305.02, 1233.39, 1186.55, 1263.67, 1112.5])
rt_C = np.array([1158.66, 1153.52, 1278.88, 1215.61, 1235.7, 1302.71, 1300.77, 1242.93, 1268.78, 1188.09])

dp_A = np.array([1.629, 2.184, 1.515, 1.72, 1.526, 1.764, 0.553, 2.147, 1.625, 1.806])
dp_B = np.array([0.681, 1.205, 1.458, 0.605, 1.878, 0.169, 1.252, 1.004, 0.65, 1.813])
dp_C = np.array([0.612, 1.351, 0.817, 0.262, 1.353, -0.169, 0.839, 1.05, 0.212, 1.561])

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
# Statistics
# ============================================================
def rm_anova(data):
    n, k = data.shape
    gm = data.mean()
    SS_c = n * np.sum((data.mean(0) - gm)**2)
    SS_s = k * np.sum((data.mean(1) - gm)**2)
    SS_t = np.sum((data - gm)**2)
    SS_e = SS_t - SS_c - SS_s
    df1, df2 = k - 1, (k-1)*(n-1)
    F = (SS_c / df1) / (SS_e / df2)
    p = 1 - stats.f.cdf(F, df1, df2)
    eta2 = SS_c / (SS_c + SS_e)
    return F, p, eta2, df1, df2


def paired_t(a, b, nc=3):
    t, pr = stats.ttest_rel(a, b)
    d = np.mean(a - b) / np.std(a - b, ddof=1)
    return t, pr, min(pr * nc, 1.0), d


def sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def bracket(ax, x1, x2, y, p, lw=0.9):
    s = sig(p)
    if s == 'n.s.':
        return
    rng = ax.get_ylim()[1] - ax.get_ylim()[0]
    bh = rng * 0.015
    ax.plot([x1, x1, x2, x2], [y, y+bh, y+bh, y], lw=lw, c='k', clip_on=False)
    ax.text((x1+x2)/2, y + bh * 1.3, s, ha='center', va='bottom',
            fontsize=11, fontweight='bold')


# ============================================================
# Figure 1: Behavioral Results — 3 spacious panels
# ============================================================
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.88)

    datasets = [
        (np.column_stack([acc_A, acc_B, acc_C]), '准确率 (%)', 'Accuracy'),
        (np.column_stack([rt_A, rt_B, rt_C]), '反应时 (ms)', 'Reaction Time'),
        (np.column_stack([dp_A, dp_B, dp_C]), "d' (敏感性)", "d' (Sensitivity)"),
    ]
    panels = ['a', 'b', 'c']

    for idx, (data, ylabel, _title) in enumerate(datasets):
        ax = axes[idx]
        means = data.mean(0)
        sems = data.std(0, ddof=1) / np.sqrt(n_subj)
        F, p, eta2, d1, d2 = rm_anova(data)

        x = np.arange(3)
        w = 0.50
        bars = ax.bar(x, means, width=w, color=COLOR_LIST, edgecolor='black',
                      linewidth=0.7, zorder=3)
        ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black',
                    capsize=5, capthick=1.2, elinewidth=1.2, zorder=4)

        for i in range(3):
            jit = np.random.normal(0, 0.04, n_subj)
            ax.scatter(x[i] + jit, data[:, i], color=COLOR_LIST[i],
                       edgecolor='white', s=22, linewidth=0.5, zorder=5, alpha=0.65)

        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        yrng = max(means + sems) - min(min(means - sems), 0)
        ymax = max(means + sems)
        gap = yrng * 0.07
        for pi, (c1, c2) in enumerate([(0,1), (0,2), (1,2)]):
            _, _, pb, _ = paired_t(data[:, c1], data[:, c2])
            bracket(ax, c1, c2, ymax + gap * (pi + 1), pb)
        ax.set_ylim(bottom=ax.get_ylim()[0], top=ymax + gap * 5)

        anova_txt = f'F({d1},{d2}) = {F:.2f}\np = {p:.4f}\n$\\eta_p^2$ = {eta2:.3f}'
        ax.text(0.97, 0.97, anova_txt, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.7))

        ax.text(-0.12, 1.05, panels[idx], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')

    # Legend outside right
    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=c, ec='black', lw=0.6, label=l)
                       for c, l in zip(COLOR_LIST, ['A 条件', 'B 条件', 'C 条件'])]
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.98, 0.5), fontsize=13, frameon=True,
               edgecolor='gray', fancybox=True)

    fig.suptitle('图1  行为学结果', fontsize=16, fontweight='bold', y=1.01)
    fig.savefig('/workspace/Figure1_Behavioral_Results.png')
    fig.savefig('/workspace/Figure1_Behavioral_Results.pdf')
    plt.close(fig)
    print('Figure 1 saved.')


# ============================================================
# Figure 2: ERP ROI Results — 4 spacious panels
# ============================================================
def fig2():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.35, left=0.05, right=0.90)

    datasets = [
        (np.column_stack([n2_A, n2_B, n2_C]),
         'N2 平均振幅 (μV)', 'N2 ROI (FCz+Fz+Cz)\n200–300 ms'),
        (np.column_stack([p3_A, p3_B, p3_C]),
         'P3 平均振幅 (μV)', 'P3 ROI (Pz+Cz)\n350–550 ms'),
        (np.column_stack([n2lat_A, n2lat_B, n2lat_C]),
         'N2 峰值潜伏期 (ms)', 'N2 峰值潜伏期'),
        (np.column_stack([p3lat_A, p3lat_B, p3lat_C]),
         'P3 峰值潜伏期 (ms)', 'P3 峰值潜伏期'),
    ]
    panels = ['a', 'b', 'c', 'd']

    for idx, (data, ylabel, title) in enumerate(datasets):
        ax = axes[idx]
        means = data.mean(0)
        sems = data.std(0, ddof=1) / np.sqrt(n_subj)
        F, p, eta2, d1, d2 = rm_anova(data)

        x = np.arange(3)
        w = 0.50
        ax.bar(x, means, width=w, color=COLOR_LIST, edgecolor='black',
               linewidth=0.7, zorder=3)
        ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black',
                    capsize=5, capthick=1.2, elinewidth=1.2, zorder=4)

        for i in range(3):
            jit = np.random.normal(0, 0.04, n_subj)
            ax.scatter(x[i] + jit, data[:, i], color=COLOR_LIST[i],
                       edgecolor='white', s=18, linewidth=0.4, zorder=5, alpha=0.55)

        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        yrng = max(means + sems) - min(means - sems)
        ymax = max(means + sems)
        gap = yrng * 0.07
        for pi, (c1, c2) in enumerate([(0,1), (0,2), (1,2)]):
            _, _, pb, _ = paired_t(data[:, c1], data[:, c2])
            bracket(ax, c1, c2, ymax + gap * (pi + 1), pb)
        ax.set_ylim(top=ymax + gap * 5.5)

        s_txt = sig(p)
        ax.text(0.97, 0.97, f'F={F:.2f}, p={p:.3f} {s_txt}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.6))

        ax.text(-0.15, 1.10, panels[idx], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')

    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=c, ec='black', lw=0.6, label=l)
                       for c, l in zip(COLOR_LIST, ['A 条件', 'B 条件', 'C 条件'])]
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.99, 0.5), fontsize=13, frameon=True,
               edgecolor='gray', fancybox=True)

    fig.suptitle('图2  ERP ROI 成分分析', fontsize=16, fontweight='bold', y=1.03)
    fig.savefig('/workspace/Figure2_ERP_ROI_Results.png')
    fig.savefig('/workspace/Figure2_ERP_ROI_Results.pdf')
    plt.close(fig)
    print('Figure 2 saved.')


# ============================================================
# Figure 3: Comprehensive — 2 rows, row1=behavioral, row2=ERP
# ============================================================
def fig3():
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35,
                          left=0.06, right=0.90, top=0.92, bottom=0.08)

    # Row 1: Behavioral (3 panels, leave 4th for legend)
    beh = [
        (np.column_stack([acc_A, acc_B, acc_C]), '准确率 (%)'),
        (np.column_stack([rt_A, rt_B, rt_C]), '反应时 (ms)'),
        (np.column_stack([dp_A, dp_B, dp_C]), "d'"),
    ]
    panels_row1 = ['a', 'b', 'c']

    for col, (data, ylabel) in enumerate(beh):
        ax = fig.add_subplot(gs[0, col])
        _bar_panel(ax, data, ylabel, panels_row1[col])

    # Row 1 col 4: correlation (Accuracy diff vs P3 diff)
    ax = fig.add_subplot(gs[0, 3])
    _corr_panel(ax, dp_A - dp_C, p3_A - p3_C,
                "Δd' (A−C)", 'ΔP3 振幅 (A−C, μV)', 'd')

    # Row 2: ERP (4 panels)
    erp = [
        (np.column_stack([n2_A, n2_B, n2_C]), 'N2 振幅 (μV)'),
        (np.column_stack([p3_A, p3_B, p3_C]), 'P3 振幅 (μV)'),
        (np.column_stack([n2lat_A, n2lat_B, n2lat_C]), 'N2 潜伏期 (ms)'),
        (np.column_stack([p3lat_A, p3lat_B, p3lat_C]), 'P3 潜伏期 (ms)'),
    ]
    panels_row2 = ['e', 'f', 'g', 'h']
    for col, (data, ylabel) in enumerate(erp):
        ax = fig.add_subplot(gs[1, col])
        _bar_panel(ax, data, ylabel, panels_row2[col])

    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=c, ec='black', lw=0.6, label=l)
                       for c, l in zip(COLOR_LIST, ['A 条件', 'B 条件', 'C 条件'])]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.97), fontsize=13, frameon=True,
               edgecolor='gray', fancybox=True)

    fig.suptitle('图3  行为学与ERP综合分析', fontsize=18, fontweight='bold', y=0.98)
    fig.savefig('/workspace/Figure3_Comprehensive_Summary.png')
    fig.savefig('/workspace/Figure3_Comprehensive_Summary.pdf')
    plt.close(fig)
    print('Figure 3 saved.')


def _bar_panel(ax, data, ylabel, label):
    means = data.mean(0)
    sems = data.std(0, ddof=1) / np.sqrt(n_subj)
    F, p, eta2, d1, d2 = rm_anova(data)

    x = np.arange(3)
    ax.bar(x, means, width=0.50, color=COLOR_LIST, edgecolor='black',
           linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black',
                capsize=5, capthick=1, elinewidth=1, zorder=4)
    for i in range(3):
        jit = np.random.normal(0, 0.04, n_subj)
        ax.scatter(x[i] + jit, data[:, i], color=COLOR_LIST[i],
                   edgecolor='white', s=16, linewidth=0.4, zorder=5, alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    yrng = max(means + sems) - min(means - sems)
    ymax = max(means + sems)
    gap = yrng * 0.07
    for pi, (c1, c2) in enumerate([(0,1), (0,2), (1,2)]):
        _, _, pb, _ = paired_t(data[:, c1], data[:, c2])
        bracket(ax, c1, c2, ymax + gap * (pi + 1), pb)
    ax.set_ylim(top=ymax + gap * 5.5)

    s = sig(p)
    ax.text(0.97, 0.97, f'F={F:.1f}, p={p:.3f} {s}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='lightyellow', alpha=0.6))
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


def _corr_panel(ax, xd, yd, xl, yl, label):
    ax.scatter(xd, yd, c='#3C5488', s=50, edgecolor='white', linewidth=0.6, zorder=3)
    r, p = stats.pearsonr(xd, yd)
    sl, it = np.polyfit(xd, yd, 1)
    xline = np.linspace(xd.min(), xd.max(), 100)
    ax.plot(xline, sl * xline + it, '--', color='#E64B35', lw=1.5, alpha=0.8, zorder=2)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f'r = {r:.3f}, p = {p:.3f} {sig(p)}', fontsize=10, pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


# ============================================================
# Figure 4: Radar — larger, cleaner
# ============================================================
def fig4():
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.1, right=0.75, top=0.88, bottom=0.08)

    cats = ['准确率', '速度\n(1/RT)', "d'", 'P3 振幅', '−N2 振幅']

    def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-10)
    vals = np.column_stack([
        norm(np.array([acc_A.mean(), acc_B.mean(), acc_C.mean()])),
        norm(1000 / np.array([rt_A.mean(), rt_B.mean(), rt_C.mean()])),
        norm(np.array([dp_A.mean(), dp_B.mean(), dp_C.mean()])),
        norm(np.array([p3_A.mean(), p3_B.mean(), p3_C.mean()])),
        norm(-np.array([n2_A.mean(), n2_B.mean(), n2_C.mean()])),
    ])
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    for i, (cond, color) in enumerate(zip(COND_LABELS, COLOR_LIST)):
        v = vals[i].tolist() + [vals[i][0]]
        ax.plot(angles, v, 'o-', lw=2.0, color=color, label=f'{cond} 条件', ms=6)
        ax.fill(angles, v, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=9, color='gray')
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=13,
              frameon=True, edgecolor='gray', fancybox=True)
    ax.set_title('图4  多维度性能概况', fontsize=16, fontweight='bold', pad=25)

    fig.savefig('/workspace/Figure4_Radar_Chart.png')
    fig.savefig('/workspace/Figure4_Radar_Chart.pdf')
    plt.close(fig)
    print('Figure 4 saved.')


# ============================================================
# Figure 5: ML Classification — spacious horizontal bars
# ============================================================
def fig5():
    print('\n====== ML Classification (LOSO-CV) ======')
    X, y, groups = [], [], []
    for si in range(n_subj):
        for ci, arrs in enumerate([
            (acc_A, rt_A, dp_A, n2_A, p3_A, n2lat_A, p3lat_A),
            (acc_B, rt_B, dp_B, n2_B, p3_B, n2lat_B, p3lat_B),
            (acc_C, rt_C, dp_C, n2_C, p3_C, n2lat_C, p3lat_C),
        ]):
            X.append([a[si] for a in arrs])
            y.append(ci)
            groups.append(si)
    X, y, groups = np.array(X), np.array(y), np.array(groups)

    clfs = {
        'SVM (Linear)':    SVC(kernel='linear', C=1.0),
        'SVM (RBF)':       SVC(kernel='rbf', C=1.0, gamma='scale'),
        'KNN':             KNeighborsClassifier(n_neighbors=3),
        'Decision Tree':   DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest':   RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'Logistic Reg.':   LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes':     GaussianNB(),
        'LDA':             LinearDiscriminantAnalysis(),
        'Gradient Boost':  GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42),
    }

    logo = LeaveOneGroupOut()
    results = {}
    for name, clf in clfs.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        yp = np.zeros_like(y)
        for tri, tei in logo.split(X, y, groups):
            pipe.fit(X[tri], y[tri])
            yp[tei] = pipe.predict(X[tei])
        acc = accuracy_score(y, yp) * 100
        results[name] = acc
        print(f'  {name:20s}: {acc:.1f}%')

    # Color palette matching reference (9 distinct colors)
    algo_colors = [
        '#1F4E79',   # dark blue
        '#E64B35',   # red
        '#00A087',   # green
        '#7030A0',   # purple
        '#F39B7F',   # orange
        '#2C2C2C',   # near black
        '#8B6914',   # olive/brown
        '#0070C0',   # blue
        '#6A0DAD',   # dark purple
    ]

    sorted_res = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    names = list(sorted_res.keys())
    accs = list(sorted_res.values())

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.10)

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, height=0.60, edgecolor='black', linewidth=0.6, zorder=3)
    for i, bar in enumerate(bars):
        bar.set_facecolor(algo_colors[i % len(algo_colors)])

    ax.axvline(x=33.33, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    ax.text(34.5, len(names) - 0.3, f'机会水平\n(33.3%)', color='red', fontsize=11, va='top')

    for i, v in enumerate(accs):
        ax.text(v + 1.0, i, f'{v:.1f}%', va='center', fontsize=12, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=13)
    ax.set_xlabel('分类精度 (%)', fontsize=14)
    ax.set_xlim(0, max(accs) + 15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.set_title('图5  9种算法分类精度对比 (LOSO-CV)', fontsize=16, fontweight='bold', pad=12)

    fig.savefig('/workspace/Figure5_ML_Classification.png')
    fig.savefig('/workspace/Figure5_ML_Classification.pdf')
    plt.close(fig)
    print('Figure 5 saved.')

    # Feature importance
    pipe_rf = Pipeline([('scaler', StandardScaler()),
                         ('clf', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42))])
    pipe_rf.fit(X, y)
    imp = pipe_rf.named_steps['clf'].feature_importances_
    fnames = ['Accuracy', 'RT', "d'", 'N2 Amp', 'P3 Amp', 'N2 Lat', 'P3 Lat']
    si = np.argsort(imp)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.subplots_adjust(left=0.15, right=0.88, top=0.88, bottom=0.10)
    colors_imp = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(fnames)))
    ax2.barh(np.arange(len(fnames)), imp[si], height=0.55,
             color=colors_imp, edgecolor='black', linewidth=0.6)

    for i, v in enumerate(imp[si]):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

    ax2.set_yticks(np.arange(len(fnames)))
    ax2.set_yticklabels([fnames[i] for i in si], fontsize=13)
    ax2.set_xlabel('特征重要性', fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('图6  Random Forest 特征重要性', fontsize=16, fontweight='bold', pad=12)

    fig2.savefig('/workspace/Figure6_Feature_Importance.png')
    fig2.savefig('/workspace/Figure6_Feature_Importance.pdf')
    plt.close(fig2)
    print('Figure 6 saved.')

    return results


# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    print('=' * 60)
    print(' Generating Spacious Scientific Figures')
    print('=' * 60)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print('\n' + '=' * 60)
    print(' All 6 figures regenerated!')
    print('=' * 60)
