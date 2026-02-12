"""
静息脑电 Individual Alpha Frequency (IAF) 方法对比 —— 科研绘图
方法一: FOOOF 参数化频谱分解（高斯峰拟合）
方法二: 原始功率谱 + 平滑峰值检测（SG / 高斯平滑）

生成图表：
  Figure 1 — 相关性散点图 + 线性回归（FOOOF vs SG, FOOOF vs Gaussian）
  Figure 2 — Bland-Altman 一致性分析图
  Figure 3 — 被试配对折线图 + 差值分布直方图
  Figure 4 — 小提琴图 + 配对 t 检验
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# =====================================================================
# 0. 数据（43 名被试 avg 通道的 IAF，单位 Hz）
# =====================================================================
subjects = [
    "wangweiyong", "lilincheng", "zhangzijian", "zhangzhaowei", "shaowenyu",
    "wulanzhou", "zhaoyongqi", "liangshuo", "wuzhinneg", "yanhongxiang",
    "yangyizhi", "yangzhen", "wufushan", "lijiahang", "zhangxiang",
    "huchunxin", "zengbaiyi", "chenweifan", "liuhaohang", "qiutian",
    "lidongyang", "mayingkai", "zhanglongtao", "xushengye", "wangweiming",
    "weidonghui", "jiachenshuo", "dushu", "tujiahuang", "jiangxuyong",
    "wangyisen", "liuxiangmin", "tanli", "zhongshijie", "yangwenjie",
    "dinghaoling", "yangyongcun", "chenyunhui", "yandongfei", "huxun",
    "zhaochunan", "xuaochao", "dongminghao",
]

fooof_iaf = np.array([
    9.30, 10.68, 10.87, 9.85, 8.75,
    10.71, 10.73, 10.89, 10.13, 10.57,
    10.22, 9.98, 9.44, 10.39, 12.53,
    9.19, 8.94, 12.48, 9.68, 9.78,
    10.61, 8.80, 9.82, 8.98, 10.32,
    8.40, 10.64, 10.64, 10.61, 8.80,
    9.25, 9.58, 9.77, 10.36, 10.32,
    9.68, 9.18, 10.83, 9.19, 10.80,
    9.73, 10.95, 9.54,
])

sg_iaf = np.array([
    9.20, 10.47, 10.90, 9.89, 8.60,
    10.66, 10.79, 10.76, 10.08, 10.44,
    10.22, 10.01, 9.55, 10.23, 12.46,
    9.19, 8.96, 12.61, 9.68, 9.73,
    10.17, 8.97, 9.62, 9.00, 10.20,
    8.47, 8.00, 10.58, 10.58, 9.20,
    9.45, 9.63, 9.72, 10.26, 10.18,
    9.51, 9.21, 9.12, 9.03, 10.77,
    9.76, 10.64, 9.51,
])

gs_iaf = np.array([
    9.12, 10.43, 10.94, 9.73, 8.69,
    10.53, 10.94, 10.71, 10.09, 10.68,
    10.34, 10.16, 9.39, 10.04, 12.24,
    9.10, 9.05, 12.53, 9.52, 9.87,
    10.17, 9.05, 9.55, 8.76, 10.34,
    8.26, 10.35, 10.37, 10.51, 9.07,
    9.51, 9.73, 9.45, 10.16, 10.31,
    9.51, 9.24, 11.24, 8.87, 10.85,
    9.78, 10.64, 9.50,
])

N = len(subjects)

# =====================================================================
# 全局绘图样式
# =====================================================================
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "SimSun", "DejaVu Serif"],
    "font.sans-serif":   ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# 配色
C_FOOOF = "#2171B5"   # 蓝色 — FOOOF
C_SG    = "#E6550D"   # 橙红 — SG
C_GS    = "#31A354"   # 绿色 — Gaussian
C_GRAY  = "#969696"

# =====================================================================
# 辅助函数
# =====================================================================
def add_identity(ax, **kwargs):
    """在散点图上画 y=x 参考线"""
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, ls="--", color=C_GRAY, lw=1, zorder=0, **kwargs)
    ax.set_xlim(lims); ax.set_ylim(lims)

def stat_text(x, y):
    """返回 r, p, ICC 等统计文本"""
    r, p = stats.pearsonr(x, y)
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff  = np.std(diff, ddof=1)
    return r, p, mean_diff, std_diff

def p_to_stars(p):
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."

# =====================================================================
# Figure 1 — 相关性散点图 + 线性回归
# =====================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))

for ax, y, color, label in zip(
    axes1,
    [sg_iaf, gs_iaf],
    [C_SG, C_GS],
    ["SG Smoothing", "Gaussian Smoothing"],
):
    r, p, md, sd = stat_text(fooof_iaf, y)
    # 线性回归
    slope, intercept, r_val, p_val, se = stats.linregress(fooof_iaf, y)
    x_fit = np.linspace(7.5, 13.5, 100)
    y_fit = slope * x_fit + intercept

    ax.scatter(fooof_iaf, y, s=42, c=color, alpha=0.75, edgecolors="white",
               linewidths=0.5, zorder=3)
    ax.plot(x_fit, y_fit, color=color, lw=2, zorder=2,
            label=f"y = {slope:.2f}x + {intercept:.2f}")
    add_identity(ax)

    # 统计标注
    ax.text(0.05, 0.95,
            f"r = {r:.3f}  (p < 0.001)\n"
            f"Mean diff = {md:+.3f} Hz\n"
            f"SD of diff = {sd:.3f} Hz\n"
            f"N = {N}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax.set_xlabel("FOOOF IAF (Hz)")
    ax.set_ylabel(f"{label} IAF (Hz)")
    ax.set_title(f"FOOOF  vs  {label}")
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_aspect("equal", adjustable="box")

fig1.suptitle("Figure 1.  Correlation between FOOOF and Smoothing-based IAF",
              fontweight="bold", y=1.02)
fig1.tight_layout()
fig1.savefig("fig1_correlation.png")
fig1.savefig("fig1_correlation.svg")
print("[OK] fig1_correlation saved.")

# =====================================================================
# Figure 2 — Bland-Altman 一致性分析图
# =====================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

for ax, y, color, label in zip(
    axes2,
    [sg_iaf, gs_iaf],
    [C_SG, C_GS],
    ["SG Smoothing", "Gaussian Smoothing"],
):
    mean_pair = (fooof_iaf + y) / 2
    diff_pair = fooof_iaf - y
    md  = np.mean(diff_pair)
    sd  = np.std(diff_pair, ddof=1)
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd

    ax.scatter(mean_pair, diff_pair, s=42, c=color, alpha=0.75,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Mean difference 线
    ax.axhline(md, color="black", ls="-", lw=1.2, zorder=2)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else 8.0,
            md + 0.03, f"Mean = {md:+.3f}",
            fontsize=9, va="bottom")

    # 95% LoA
    ax.axhline(loa_upper, color=color, ls="--", lw=1, zorder=2)
    ax.axhline(loa_lower, color=color, ls="--", lw=1, zorder=2)
    ax.fill_between(ax.get_xlim() if ax.get_xlim()[0] != 0 else [7.5, 13.5],
                     loa_lower, loa_upper, color=color, alpha=0.08, zorder=0)

    # LoA 标注
    ax.text(13.0, loa_upper + 0.03, f"+1.96SD = {loa_upper:+.3f}",
            fontsize=8, color=color, ha="right", va="bottom")
    ax.text(13.0, loa_lower - 0.03, f"−1.96SD = {loa_lower:+.3f}",
            fontsize=8, color=color, ha="right", va="top")

    ax.axhline(0, color=C_GRAY, ls=":", lw=0.8, zorder=1)
    ax.set_xlabel("Mean of Two Methods (Hz)")
    ax.set_ylabel("Difference: FOOOF − Smoothing (Hz)")
    ax.set_title(f"FOOOF  vs  {label}")

fig2.suptitle("Figure 2.  Bland-Altman Agreement Analysis",
              fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig("fig2_bland_altman.png")
fig2.savefig("fig2_bland_altman.svg")
print("[OK] fig2_bland_altman saved.")

# =====================================================================
# Figure 3 — 被试配对折线图 + 差值分布直方图
# =====================================================================
fig3 = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1.2, 1.2], wspace=0.35)

# --- 3a 配对折线图 ---
ax3a = fig3.add_subplot(gs[0])
x_idx = np.arange(N)

ax3a.plot(x_idx, fooof_iaf, "o-", color=C_FOOOF, ms=5, lw=1.2,
          label="FOOOF", zorder=3)
ax3a.plot(x_idx, sg_iaf, "s-", color=C_SG, ms=4, lw=1, alpha=0.8,
          label="SG Smoothing", zorder=2)
ax3a.plot(x_idx, gs_iaf, "^-", color=C_GS, ms=4, lw=1, alpha=0.8,
          label="Gaussian Smoothing", zorder=2)

ax3a.set_xlabel("Subject Index")
ax3a.set_ylabel("IAF (Hz)")
ax3a.set_title("(a) IAF per Subject")
ax3a.legend(loc="upper left", framealpha=0.9)
ax3a.set_xlim(-1, N)
ax3a.grid(axis="y", ls="--", alpha=0.4)

# --- 3b 差值直方图 FOOOF−SG ---
ax3b = fig3.add_subplot(gs[1])
diff_sg = fooof_iaf - sg_iaf
ax3b.hist(diff_sg, bins=15, color=C_SG, alpha=0.7, edgecolor="white")
ax3b.axvline(np.mean(diff_sg), color="black", ls="-", lw=1.3,
             label=f"Mean={np.mean(diff_sg):+.2f}")
ax3b.axvline(0, color=C_GRAY, ls=":", lw=0.8)
ax3b.set_xlabel("FOOOF − SG (Hz)")
ax3b.set_ylabel("Count")
ax3b.set_title("(b) Diff: FOOOF − SG")
ax3b.legend(fontsize=9)

# --- 3c 差值直方图 FOOOF−GS ---
ax3c = fig3.add_subplot(gs[2])
diff_gs = fooof_iaf - gs_iaf
ax3c.hist(diff_gs, bins=15, color=C_GS, alpha=0.7, edgecolor="white")
ax3c.axvline(np.mean(diff_gs), color="black", ls="-", lw=1.3,
             label=f"Mean={np.mean(diff_gs):+.2f}")
ax3c.axvline(0, color=C_GRAY, ls=":", lw=0.8)
ax3c.set_xlabel("FOOOF − Gaussian (Hz)")
ax3c.set_ylabel("Count")
ax3c.set_title("(c) Diff: FOOOF − Gaussian")
ax3c.legend(fontsize=9)

fig3.suptitle("Figure 3.  Subject-level IAF Comparison and Difference Distribution",
              fontweight="bold", y=1.02)
fig3.savefig("fig3_paired_and_histogram.png")
fig3.savefig("fig3_paired_and_histogram.svg")
print("[OK] fig3_paired_and_histogram saved.")

# =====================================================================
# Figure 4 — 小提琴图 / 箱线图 + 配对 t 检验
# =====================================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5.5))

# --- 4a 三种方法的IAF分布 ---
ax4a = axes4[0]
data_all = [fooof_iaf, sg_iaf, gs_iaf]
labels_all = ["FOOOF", "SG Smoothing", "Gaussian\nSmoothing"]
colors_all = [C_FOOOF, C_SG, C_GS]

parts = ax4a.violinplot(data_all, positions=[1, 2, 3], showmeans=True,
                         showmedians=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors_all[i])
    pc.set_alpha(0.35)
    pc.set_edgecolor(colors_all[i])
parts["cmeans"].set_color("black")
parts["cmeans"].set_linewidth(1.5)
parts["cmedians"].set_color(C_GRAY)
parts["cmedians"].set_linewidth(1)
parts["cmedians"].set_linestyle("--")

# 叠加 strip (jitter) 点
for i, d in enumerate(data_all):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
    ax4a.scatter(np.full(len(d), i + 1) + jitter, d, s=18,
                 c=colors_all[i], alpha=0.6, edgecolors="white",
                 linewidths=0.3, zorder=3)

ax4a.set_xticks([1, 2, 3])
ax4a.set_xticklabels(labels_all)
ax4a.set_ylabel("IAF (Hz)")
ax4a.set_title("(a) IAF Distribution by Method")
ax4a.grid(axis="y", ls="--", alpha=0.3)

# 添加配对 t 检验标注
t_sg, p_sg = stats.ttest_rel(fooof_iaf, sg_iaf)
t_gs, p_gs = stats.ttest_rel(fooof_iaf, gs_iaf)

def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="black")
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=9)

ymax = max(fooof_iaf.max(), sg_iaf.max(), gs_iaf.max())
add_bracket(ax4a, 1, 2, ymax + 0.3, 0.12,
            f"p={p_sg:.4f} {p_to_stars(p_sg)}")
add_bracket(ax4a, 1, 3, ymax + 0.8, 0.12,
            f"p={p_gs:.4f} {p_to_stars(p_gs)}")

# --- 4b 差值箱线图 ---
ax4b = axes4[1]
diff_data = [fooof_iaf - sg_iaf, fooof_iaf - gs_iaf]
bp = ax4b.boxplot(diff_data, positions=[1, 2], widths=0.5, patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker="D", markerfacecolor="black",
                                 markeredgecolor="black", markersize=5),
                  medianprops=dict(color="black", lw=1.5),
                  flierprops=dict(marker="o", markersize=4, alpha=0.5))
for patch, color in zip(bp["boxes"], [C_SG, C_GS]):
    patch.set_facecolor(color)
    patch.set_alpha(0.35)

# 叠加 strip 点
for i, d in enumerate(diff_data):
    jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(d))
    ax4b.scatter(np.full(len(d), i + 1) + jitter, d, s=18,
                 c=[C_SG, C_GS][i], alpha=0.6, edgecolors="white",
                 linewidths=0.3, zorder=3)

ax4b.axhline(0, color=C_GRAY, ls=":", lw=0.8)
ax4b.set_xticks([1, 2])
ax4b.set_xticklabels(["FOOOF − SG", "FOOOF − Gaussian"])
ax4b.set_ylabel("Difference (Hz)")
ax4b.set_title("(b) Paired Differences")
ax4b.grid(axis="y", ls="--", alpha=0.3)

# one-sample t-test: diff vs 0
_, p_sg_0 = stats.ttest_1samp(fooof_iaf - sg_iaf, 0)
_, p_gs_0 = stats.ttest_1samp(fooof_iaf - gs_iaf, 0)
ax4b.text(1, ax4b.get_ylim()[1] * 0.90,
          f"vs 0: p={p_sg_0:.4f}\n{p_to_stars(p_sg_0)}",
          ha="center", fontsize=9, color=C_SG)
ax4b.text(2, ax4b.get_ylim()[1] * 0.90,
          f"vs 0: p={p_gs_0:.4f}\n{p_to_stars(p_gs_0)}",
          ha="center", fontsize=9, color=C_GS)

fig4.suptitle("Figure 4.  Statistical Comparison of IAF Methods",
              fontweight="bold", y=1.02)
fig4.tight_layout()
fig4.savefig("fig4_violin_box.png")
fig4.savefig("fig4_violin_box.svg")
print("[OK] fig4_violin_box saved.")

# =====================================================================
# 打印统计摘要
# =====================================================================
print("\n" + "=" * 70)
print("Statistical Summary")
print("=" * 70)
print(f"N = {N} subjects")
print(f"\nFOOOF IAF:          {fooof_iaf.mean():.2f} ± {fooof_iaf.std(ddof=1):.2f} Hz")
print(f"SG Smoothing IAF:   {sg_iaf.mean():.2f} ± {sg_iaf.std(ddof=1):.2f} Hz")
print(f"Gaussian Smth IAF:  {gs_iaf.mean():.2f} ± {gs_iaf.std(ddof=1):.2f} Hz")

print(f"\n--- FOOOF vs SG Smoothing ---")
r_sg, p_r_sg = stats.pearsonr(fooof_iaf, sg_iaf)
t_sg, p_t_sg = stats.ttest_rel(fooof_iaf, sg_iaf)
diff_sg = fooof_iaf - sg_iaf
print(f"  Pearson r = {r_sg:.4f}  (p = {p_r_sg:.2e})")
print(f"  Paired t-test: t = {t_sg:.3f}, p = {p_t_sg:.4f}  {p_to_stars(p_t_sg)}")
print(f"  Mean diff = {diff_sg.mean():+.3f} Hz, SD = {diff_sg.std(ddof=1):.3f} Hz")
print(f"  95% LoA = [{diff_sg.mean()-1.96*diff_sg.std(ddof=1):.3f}, "
      f"{diff_sg.mean()+1.96*diff_sg.std(ddof=1):.3f}]")

print(f"\n--- FOOOF vs Gaussian Smoothing ---")
r_gs, p_r_gs = stats.pearsonr(fooof_iaf, gs_iaf)
t_gs, p_t_gs = stats.ttest_rel(fooof_iaf, gs_iaf)
diff_gs = fooof_iaf - gs_iaf
print(f"  Pearson r = {r_gs:.4f}  (p = {p_r_gs:.2e})")
print(f"  Paired t-test: t = {t_gs:.3f}, p = {p_t_gs:.4f}  {p_to_stars(p_t_gs)}")
print(f"  Mean diff = {diff_gs.mean():+.3f} Hz, SD = {diff_gs.std(ddof=1):.3f} Hz")
print(f"  95% LoA = [{diff_gs.mean()-1.96*diff_gs.std(ddof=1):.3f}, "
      f"{diff_gs.mean()+1.96*diff_gs.std(ddof=1):.3f}]")

print(f"\n--- SG vs Gaussian Smoothing ---")
r_sg_gs, p_sg_gs = stats.pearsonr(sg_iaf, gs_iaf)
t_sg_gs, p_t_sg_gs = stats.ttest_rel(sg_iaf, gs_iaf)
diff_sg_gs = sg_iaf - gs_iaf
print(f"  Pearson r = {r_sg_gs:.4f}  (p = {p_sg_gs:.2e})")
print(f"  Paired t-test: t = {t_sg_gs:.3f}, p = {p_t_sg_gs:.4f}  {p_to_stars(p_t_sg_gs)}")
print(f"  Mean diff = {diff_sg_gs.mean():+.3f} Hz, SD = {diff_sg_gs.std(ddof=1):.3f} Hz")

print("=" * 70)
print("\n[DONE] All figures saved (PNG + SVG).")
plt.show()
