#!/usr/bin/env python3
"""
GraphPad-style scatter dot plots for P3 peak amplitude & latency
3 electrodes (FCz, Fz, Cz) × 2 measures = 6 figures
"""
import numpy as np
import matplotlib
import sys
if sys.platform != 'win32':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 13, 'axes.linewidth': 1.2,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

COLORS = ['#E64B35', '#4DBBD5', '#00A087']  # A=红, B=青蓝, C=青绿
LABELS = ['A', 'B', 'C']

# ==================== Data ====================
# P3 peak amplitude (10 subjects × 3 conditions per electrode)
amp = {
    'FCz': {
        'A': np.array([-0.6514,-1.5343,0.2324,0.4599,2.3497,-0.1217,-0.1420,10.7658,-0.4483,1.8775]),
        'B': np.array([-0.2690,-0.9078,-1.1250,0.9778,2.7398,-2.8028,-0.1194,0.7005,-1.0273,0.4239]),
        'C': np.array([-1.4204,-2.5303,-2.0959,-1.5825,-1.0011,0.1693,-0.4517,2.0429,0.5086,0.9671]),
    },
    'Fz': {
        'A': np.array([-0.7570,-0.5667,0.4685,-0.6058,1.5035,0.7833,-0.4445,18.1716,-0.7312,4.1906]),
        'B': np.array([-0.0498,-0.6620,-0.4790,1.0960,3.8476,-1.8366,0.2149,1.1057,-0.5116,2.9913]),
        'C': np.array([-1.8974,-1.7040,-1.8604,-0.9344,-1.7845,0.5505,-0.9629,10.1149,0.3492,2.7018]),
    },
    'Cz': {
        'A': np.array([0.6549,-1.4525,0.1280,2.6232,2.3709,0.6347,0.1166,10.9160,1.0341,0.6906]),
        'B': np.array([-0.9478,-0.8827,-1.7308,1.2667,4.0294,0.1897,-1.4922,0.0707,-0.2980,-0.0405]),
        'C': np.array([0.2059,-2.7279,-1.3576,0.0381,0.8299,0.0966,-0.1665,1.8773,0.1228,1.5284]),
    },
}

lat = {
    'FCz': {
        'A': np.array([332,352,438,318,386,308,358,434,332,332], dtype=float),
        'B': np.array([326,346,280,312,414,304,372,416,326,384], dtype=float),
        'C': np.array([338,352,288,322,600,302,306,312,326,300], dtype=float),
    },
    'Fz': {
        'A': np.array([334,354,434,320,390,308,358,408,330,390], dtype=float),
        'B': np.array([320,348,404,316,408,302,378,308,326,392], dtype=float),
        'C': np.array([250,270,288,598,266,308,300,412,326,392], dtype=float),
    },
    'Cz': {
        'A': np.array([332,350,440,316,386,314,352,436,336,330], dtype=float),
        'B': np.array([340,348,282,304,418,406,292,410,446,338], dtype=float),
        'C': np.array([342,328,286,318,600,302,312,422,330,302], dtype=float),
    },
}

# ==================== Plot function ====================
def plot_scatter(ax, data_dict, ylabel, title_text, electrode):
    n_subj = len(data_dict['A'])
    x_positions = [1, 2, 3]

    for i, (cond, color) in enumerate(zip(LABELS, COLORS)):
        vals = data_dict[cond]
        x = x_positions[i]
        jitter = np.random.normal(0, 0.08, n_subj)
        ax.scatter(x + jitter, vals, c=color, s=40, alpha=0.75,
                   edgecolors='white', linewidth=0.5, zorder=3)
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(n_subj)
        ax.plot([x - 0.25, x + 0.25], [m, m], color='black', lw=2, zorder=4)
        ax.plot([x, x], [m - se, m + se], color='black', lw=1.2, zorder=4)
        ax.plot([x - 0.08, x + 0.08], [m - se, m - se], color='black', lw=1.2, zorder=4)
        ax.plot([x - 0.08, x + 0.08], [m + se, m + se], color='black', lw=1.2, zorder=4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(LABELS, fontsize=14, fontweight='bold')
    ax.set_xlim(0.4, 3.6)
    ax.set_ylabel(ylabel, fontsize=14, fontfamily='sans-serif')
    ax.set_title(f'{electrode}', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1)

    # Significance brackets
    pairs = [(0, 1, 'A', 'B'), (0, 2, 'A', 'C'), (1, 2, 'B', 'C')]
    all_vals = np.concatenate([data_dict[c] for c in LABELS])
    y_max = np.max(all_vals)
    y_range = np.max(all_vals) - np.min(all_vals)
    gap = y_range * 0.08

    bracket_y = y_max + gap
    for pi, (i1, i2, c1, c2) in enumerate(pairs):
        _, p_raw = stats.ttest_rel(data_dict[c1], data_dict[c2])
        p_bonf = min(p_raw * 3, 1.0)
        if p_bonf >= 0.05:
            continue
        if p_bonf < 0.001:   s = '***'
        elif p_bonf < 0.01:  s = '**'
        elif p_bonf < 0.05:  s = '*'
        else: continue

        x1, x2 = x_positions[i1], x_positions[i2]
        by = bracket_y + gap * pi
        bh = y_range * 0.012
        ax.plot([x1, x1, x2, x2], [by, by + bh, by + bh, by],
                'k-', lw=1, clip_on=False)
        ax.text((x1 + x2) / 2, by + bh * 1.5, s, ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_ylim(top=bracket_y + gap * 4)


# ==================== Generate 6 figures ====================
np.random.seed(42)

# --- 3 amplitude figures ---
fig_amp, axes_amp = plt.subplots(1, 3, figsize=(12, 4.5))
fig_amp.subplots_adjust(wspace=0.38, left=0.08, right=0.95)
for i, elec in enumerate(['FCz', 'Fz', 'Cz']):
    plot_scatter(axes_amp[i], amp[elec], 'Amplitude ($\\mu$V)', f'P3 Peak Amplitude', elec)
    axes_amp[i].axhline(y=0, color='gray', linestyle=':', linewidth=0.8, zorder=1)
    axes_amp[i].text(-0.15, 1.08, chr(ord('a') + i), transform=axes_amp[i].transAxes,
                     fontsize=18, fontweight='bold', va='top')
fig_amp.savefig(os.path.join(SAVE_DIR, 'P3_amplitude_scatter_FCz_Fz_Cz.png'))
fig_amp.savefig(os.path.join(SAVE_DIR, 'P3_amplitude_scatter_FCz_Fz_Cz.pdf'))
plt.close(fig_amp)
print('Saved: P3_amplitude_scatter_FCz_Fz_Cz')

# --- 3 latency figures ---
fig_lat, axes_lat = plt.subplots(1, 3, figsize=(12, 4.5))
fig_lat.subplots_adjust(wspace=0.38, left=0.08, right=0.95)
for i, elec in enumerate(['FCz', 'Fz', 'Cz']):
    plot_scatter(axes_lat[i], lat[elec], 'Latency (ms)', f'P3 Peak Latency', elec)
    axes_lat[i].text(-0.15, 1.08, chr(ord('d') + i), transform=axes_lat[i].transAxes,
                     fontsize=18, fontweight='bold', va='top')
fig_lat.savefig(os.path.join(SAVE_DIR, 'P3_latency_scatter_FCz_Fz_Cz.png'))
fig_lat.savefig(os.path.join(SAVE_DIR, 'P3_latency_scatter_FCz_Fz_Cz.pdf'))
plt.close(fig_lat)
print('Saved: P3_latency_scatter_FCz_Fz_Cz')

# --- Combined 2×3 figure ---
fig_all, axes_all = plt.subplots(2, 3, figsize=(14, 9))
fig_all.subplots_adjust(wspace=0.35, hspace=0.4)
np.random.seed(42)
for i, elec in enumerate(['FCz', 'Fz', 'Cz']):
    plot_scatter(axes_all[0, i], amp[elec], 'Amplitude ($\\mu$V)', '', elec)
    axes_all[0, i].axhline(y=0, color='gray', linestyle=':', linewidth=0.8, zorder=1)
    axes_all[0, i].set_title(f'{elec} - P3 Amplitude', fontsize=13, fontweight='bold')
    axes_all[0, i].text(-0.18, 1.10, chr(ord('a') + i), transform=axes_all[0, i].transAxes,
                        fontsize=18, fontweight='bold', va='top')
np.random.seed(42)
for i, elec in enumerate(['FCz', 'Fz', 'Cz']):
    plot_scatter(axes_all[1, i], lat[elec], 'Latency (ms)', '', elec)
    axes_all[1, i].set_title(f'{elec} - P3 Latency', fontsize=13, fontweight='bold')
    axes_all[1, i].text(-0.18, 1.10, chr(ord('d') + i), transform=axes_all[1, i].transAxes,
                        fontsize=18, fontweight='bold', va='top')

fig_all.savefig(os.path.join(SAVE_DIR, 'P3_scatter_all_6panels.png'))
fig_all.savefig(os.path.join(SAVE_DIR, 'P3_scatter_all_6panels.pdf'))
plt.close(fig_all)
print('Saved: P3_scatter_all_6panels')

# --- Print statistics ---
print('\n====== P3 Statistics ======')
for elec in ['FCz', 'Fz', 'Cz']:
    print(f'\n--- {elec} P3 Peak Amplitude ---')
    for c in LABELS:
        m, s = np.mean(amp[elec][c]), np.std(amp[elec][c], ddof=1)
        print(f'  {c}: {m:.3f} ± {s:.3f}')
    for c1, c2 in [('A','B'),('A','C'),('B','C')]:
        t, p = stats.ttest_rel(amp[elec][c1], amp[elec][c2])
        print(f'  {c1} vs {c2}: t={t:.3f}, p={p:.4f}, p_bonf={min(p*3,1):.4f}')

    print(f'\n--- {elec} P3 Peak Latency ---')
    for c in LABELS:
        m, s = np.mean(lat[elec][c]), np.std(lat[elec][c], ddof=1)
        print(f'  {c}: {m:.1f} ± {s:.1f}')
    for c1, c2 in [('A','B'),('A','C'),('B','C')]:
        t, p = stats.ttest_rel(lat[elec][c1], lat[elec][c2])
        print(f'  {c1} vs {c2}: t={t:.3f}, p={p:.4f}, p_bonf={min(p*3,1):.4f}')

print('\nDone!')
