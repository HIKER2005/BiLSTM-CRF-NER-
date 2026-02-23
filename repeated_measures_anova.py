#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重复测量资料的方差分析 (Repeated Measures ANOVA)

支持三种分析方法：
1. 混合效应模型 (Mixed Effects Model) — 推荐
2. 一般线性模型 / 传统重复测量方差分析
3. 多元方差分析 (MANOVA) 方式

数据格式要求（长格式，Tab分隔）：
  id    region    channel    treat    value
  1.0   额极区    Fpz       A(节律)   2.109

使用方式：
  python repeated_measures_anova.py --input data.tsv
  python repeated_measures_anova.py --input data.tsv --method all
  python repeated_measures_anova.py --input data.tsv --method mixed --cov_struct cs
"""

import argparse
import sys
import warnings
from io import StringIO
from itertools import combinations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ========== 内置示例数据 ==========
SAMPLE_DATA = """\
1.0\t额极区 (Frontopolar Area)\tFpz\tA(节律)\t2.1091051058590000
1.0\t额极区 (Frontopolar Area)\tFpz\tB(无节律)\t1.8572336611599500
1.0\t额极区 (Frontopolar Area)\tFp1\tA(节律)\t2.3262671519486000
1.0\t额极区 (Frontopolar Area)\tFp1\tB(无节律)\t2.0447859079259700
1.0\t额极区 (Frontopolar Area)\tFp2\tA(节律)\t2.1138083205895900
1.0\t额极区 (Frontopolar Area)\tFp2\tB(无节律)\t1.8419211817669900
1.0\t前颞区 (Anterior Temporal Area)\tT7\tA(节律)\t.2718246589058320
1.0\t前颞区 (Anterior Temporal Area)\tT7\tB(无节律)\t.2773186935898810
1.0\t前颞区 (Anterior Temporal Area)\tT8\tA(节律)\t.5098487030374550
1.0\t前颞区 (Anterior Temporal Area)\tT8\tB(无节律)\t.5115313776156220
1.0\t前颞区 (Anterior Temporal Area)\tTP7\tA(节律)\t.5441073876453610
1.0\t前颞区 (Anterior Temporal Area)\tTP7\tB(无节律)\t.5483373965490610
1.0\t顶叶区 (Parietal Lobe)\tPz\tA(节律)\t.6708590411457500
1.0\t顶叶区 (Parietal Lobe)\tPz\tB(无节律)\t.6304502800412150
1.0\t顶叶区 (Parietal Lobe)\tP3\tA(节律)\t.6622108382650440
1.0\t顶叶区 (Parietal Lobe)\tP3\tB(无节律)\t.5922789236139140
1.0\t顶叶区 (Parietal Lobe)\tP4\tA(节律)\t1.2753742098046200
1.0\t顶叶区 (Parietal Lobe)\tP4\tB(无节律)\t1.1634924434361900
1.0\t枕叶区 (Occipital Lobe)\tOz\tA(节律)\t2.7519551193177200
1.0\t枕叶区 (Occipital Lobe)\tOz\tB(无节律)\t2.0997994850390600
1.0\t枕叶区 (Occipital Lobe)\tO1\tA(节律)\t2.0322070316901900
1.0\t枕叶区 (Occipital Lobe)\tO1\tB(无节律)\t1.6177824230651900
1.0\t枕叶区 (Occipital Lobe)\tO2\tA(节律)\t2.1491383301703300
1.0\t枕叶区 (Occipital Lobe)\tO2\tB(无节律)\t1.8058060985165900
2.0\t额极区 (Frontopolar Area)\tFpz\tA(节律)\t1.7520505441157300
2.0\t额极区 (Frontopolar Area)\tFpz\tB(无节律)\t1.7747271347396800
2.0\t额极区 (Frontopolar Area)\tFp1\tA(节律)\t1.6637114352277000
2.0\t额极区 (Frontopolar Area)\tFp1\tB(无节律)\t1.7470700782242100
2.0\t额极区 (Frontopolar Area)\tFp2\tA(节律)\t2.0036584506747900
2.0\t额极区 (Frontopolar Area)\tFp2\tB(无节律)\t2.0261680392024300
2.0\t前颞区 (Anterior Temporal Area)\tT7\tA(节律)\t1.4511966990055500
2.0\t前颞区 (Anterior Temporal Area)\tT7\tB(无节律)\t1.4149214175722400
2.0\t前颞区 (Anterior Temporal Area)\tT8\tA(节律)\t1.5397530584689900
2.0\t前颞区 (Anterior Temporal Area)\tT8\tB(无节律)\t1.5087020266781900
2.0\t前颞区 (Anterior Temporal Area)\tTP7\tA(节律)\t2.4368244515179400
2.0\t前颞区 (Anterior Temporal Area)\tTP7\tB(无节律)\t2.4446301629441200
2.0\t顶叶区 (Parietal Lobe)\tPz\tA(节律)\t2.3423838051438500
2.0\t顶叶区 (Parietal Lobe)\tPz\tB(无节律)\t2.2980759326968500
2.0\t顶叶区 (Parietal Lobe)\tP3\tA(节律)\t3.5539613189949300
2.0\t顶叶区 (Parietal Lobe)\tP3\tB(无节律)\t3.4457384241539600
2.0\t顶叶区 (Parietal Lobe)\tP4\tA(节律)\t2.7391227493051700
2.0\t顶叶区 (Parietal Lobe)\tP4\tB(无节律)\t2.7651263655088500
2.0\t枕叶区 (Occipital Lobe)\tOz\tA(节律)\t2.8770209360327900
2.0\t枕叶区 (Occipital Lobe)\tOz\tB(无节律)\t2.8395723214149900
2.0\t枕叶区 (Occipital Lobe)\tO1\tA(节律)\t3.9763160139673900
2.0\t枕叶区 (Occipital Lobe)\tO1\tB(无节律)\t3.9345798865289500
2.0\t枕叶区 (Occipital Lobe)\tO2\tA(节律)\t3.6671498760092600
2.0\t枕叶区 (Occipital Lobe)\tO2\tB(无节律)\t3.8752839704212500
3.0\t额极区 (Frontopolar Area)\tFpz\tA(节律)\t3.8376052166559100
3.0\t额极区 (Frontopolar Area)\tFpz\tB(无节律)\t3.7941925427716700
3.0\t额极区 (Frontopolar Area)\tFp1\tA(节律)\t5.5900490295862400
3.0\t额极区 (Frontopolar Area)\tFp1\tB(无节律)\t5.4370002765168500
3.0\t额极区 (Frontopolar Area)\tFp2\tA(节律)\t3.3026352394902200
3.0\t额极区 (Frontopolar Area)\tFp2\tB(无节律)\t3.2524794098062000
3.0\t前颞区 (Anterior Temporal Area)\tT7\tA(节律)\t.6864601297777550
3.0\t前颞区 (Anterior Temporal Area)\tT7\tB(无节律)\t.6484524679259900
3.0\t前颞区 (Anterior Temporal Area)\tT8\tA(节律)\t1.1354385747569500
3.0\t前颞区 (Anterior Temporal Area)\tT8\tB(无节律)\t1.1058767474595100
3.0\t前颞区 (Anterior Temporal Area)\tTP7\tA(节律)\t1.4108703953372000
3.0\t前颞区 (Anterior Temporal Area)\tTP7\tB(无节律)\t1.2431809769610800
3.0\t顶叶区 (Parietal Lobe)\tPz\tA(节律)\t1.8793426122717800
3.0\t顶叶区 (Parietal Lobe)\tPz\tB(无节律)\t1.8724252206370300
3.0\t顶叶区 (Parietal Lobe)\tP3\tA(节律)\t1.6327993059473200
3.0\t顶叶区 (Parietal Lobe)\tP3\tB(无节律)\t1.6957463450805600
3.0\t顶叶区 (Parietal Lobe)\tP4\tA(节律)\t1.5562885643531300
3.0\t顶叶区 (Parietal Lobe)\tP4\tB(无节律)\t1.5213605289424400
"""


def load_data(input_path=None):
    """加载数据，支持外部文件或使用内置示例数据"""
    col_names = ["id", "region", "channel", "treat", "value"]

    if input_path:
        sep = "\t"
        if input_path.endswith(".csv"):
            sep = ","
        df = pd.read_csv(input_path, sep=sep, header=None, names=col_names)
    else:
        df = pd.read_csv(StringIO(SAMPLE_DATA), sep="\t", header=None, names=col_names)

    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["treat"] = df["treat"].astype("category")
    df["channel"] = df["channel"].astype("category")
    df["region"] = df["region"].astype("category")
    df["id"] = df["id"].astype("category")

    return df


def print_separator(title=""):
    width = 80
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print(f"{'-' * width}")


def descriptive_stats(df):
    """描述性统计"""
    print_separator("描述性统计")

    print("\n【按 treat 分组】")
    desc_treat = df.groupby("treat")["value"].agg(["count", "mean", "std", "min", "max"])
    desc_treat.columns = ["样本量", "均值", "标准差", "最小值", "最大值"]
    print(desc_treat.to_string())

    print("\n【按 treat × channel 分组】")
    desc_cross = df.groupby(["treat", "channel"])["value"].agg(["count", "mean", "std"])
    desc_cross.columns = ["样本量", "均值", "标准差"]
    print(desc_cross.to_string())

    ci_data = []
    for (treat, channel), grp in df.groupby(["treat", "channel"]):
        n = len(grp)
        m = grp["value"].mean()
        se = grp["value"].std() / np.sqrt(n) if n > 1 else 0
        ci_low = m - 1.96 * se
        ci_high = m + 1.96 * se
        ci_data.append({
            "treat": treat, "channel": channel,
            "均值": m, "95%CI下限": ci_low, "95%CI上限": ci_high
        })
    ci_df = pd.DataFrame(ci_data)
    print("\n【均数及95%置信区间】")
    print(ci_df.to_string(index=False))

    return ci_df


def method_mixed_effects(df, cov_struct="vc"):
    """
    方法1：混合效应模型 (Linear Mixed Effects Model)
    使用 statsmodels MixedLM
    treat 和 channel 为固定效应，id 为随机效应（随机截距）
    """
    print_separator("方法1：混合效应模型 (Mixed Effects Model)")

    cov_struct_map = {
        "vc": "方差分量 (Variance Components)",
        "cs": "复合对称 (Compound Symmetry)",
    }
    print(f"\n协方差结构: {cov_struct_map.get(cov_struct, cov_struct)}")

    df_model = df.copy()
    df_model["treat"] = df_model["treat"].astype(str)
    df_model["channel"] = df_model["channel"].astype(str)
    df_model["region"] = df_model["region"].astype(str)
    df_model["id"] = df_model["id"].astype(str)

    print("\n--- 模型1: treat + channel + treat:channel 作为固定效应，id 作为随机截距 ---")
    try:
        model = smf.mixedlm(
            "value ~ C(treat) * C(channel)",
            data=df_model,
            groups=df_model["id"],
        )
        result = model.fit(reml=True)
        print(result.summary())
    except Exception as e:
        print(f"模型拟合失败: {e}")
        result = None

    print("\n--- 模型2: treat 作为固定效应，id 作为随机截距（不含 channel 交互） ---")
    try:
        model2 = smf.mixedlm(
            "value ~ C(treat)",
            data=df_model,
            groups=df_model["id"],
        )
        result2 = model2.fit(reml=True)
        print(result2.summary())
    except Exception as e:
        print(f"模型拟合失败: {e}")
        result2 = None

    print("\n--- 模型3: 按脑区分别分析 treat 效应 ---")
    regions = df_model["region"].unique()
    region_results = {}
    for region in sorted(regions):
        df_sub = df_model[df_model["region"] == region].copy()
        try:
            m = smf.mixedlm(
                "value ~ C(treat) * C(channel)",
                data=df_sub,
                groups=df_sub["id"],
            )
            r = m.fit(reml=True)
            region_results[region] = r
            print(f"\n  【{region}】")
            fixed = r.summary().tables[1]
            print(f"  {fixed}")
        except Exception as e:
            print(f"\n  【{region}】 拟合失败: {e}")

    return result, region_results


def _balance_data(df, within_cols, subject_col="id", dv_col="value"):
    """确保数据平衡：仅保留在所有条件组合中都有数据的被试"""
    df_bal = df.copy()
    combo_cols = within_cols if isinstance(within_cols, list) else [within_cols]
    all_combos = df_bal.groupby(combo_cols).ngroups
    subject_combo_count = df_bal.groupby(subject_col)[combo_cols[0]].count()
    valid_subjects = subject_combo_count[subject_combo_count >= all_combos].index
    df_bal = df_bal[df_bal[subject_col].isin(valid_subjects)]
    return df_bal


def method_rm_anova(df):
    """
    方法2：传统重复测量方差分析
    优先使用 statsmodels.AnovaRM（兼容性更好），pingouin 作为补充
    treat 和 channel 均为被试内因素
    """
    print_separator("方法2：传统重复测量方差分析 (Repeated Measures ANOVA)")

    df_rm = df.copy()
    df_rm["id"] = df_rm["id"].astype(str)
    df_rm["treat"] = df_rm["treat"].astype(str)
    df_rm["channel"] = df_rm["channel"].astype(str)

    from statsmodels.stats.anova import AnovaRM

    # --- 双因素 ---
    print("\n--- 双因素重复测量方差分析: treat × channel ---")
    aov = None
    df_bal = _balance_data(df_rm, ["treat", "channel"])
    n_subj = df_bal["id"].nunique()
    if n_subj < 2:
        print("  平衡数据后被试数不足，无法进行双因素分析")
    else:
        try:
            aovrm = AnovaRM(df_bal, depvar="value", subject="id", within=["treat", "channel"])
            res = aovrm.fit()
            print(res.summary())
            aov = res
        except Exception as e1:
            print(f"  statsmodels AnovaRM 失败: {e1}")
            try:
                aov_pg = pg.rm_anova(
                    data=df_bal, dv="value", within=["treat", "channel"],
                    subject="id", detailed=True
                )
                print(aov_pg.to_string())
                aov = aov_pg
            except Exception as e2:
                print(f"  pingouin rm_anova 也失败: {e2}")

        try:
            print("\n球形性检验 (Mauchly's Test):")
            spher, W, chi2, dof, pval = pg.sphericity(
                data=df_bal, dv="value", within=["treat", "channel"], subject="id"
            )
            print(f"  W = {W:.4f}, chi2 = {chi2:.4f}, df = {dof}, p = {pval:.4f}")
            if pval < 0.05:
                print("  → 球形性假设不成立，建议使用 Greenhouse-Geisser 或 Huynh-Feldt 校正")
            else:
                print("  → 球形性假设成立")
        except Exception:
            print("  球形性检验无法完成（可能是数据维度不足）")

    # --- 单因素: treat ---
    print("\n--- 单因素重复测量方差分析: treat 效应 ---")
    df_treat = df_rm.groupby(["id", "treat"])["value"].mean().reset_index()
    try:
        aovrm_t = AnovaRM(df_treat, depvar="value", subject="id", within=["treat"])
        res_t = aovrm_t.fit()
        print(res_t.summary())
    except Exception as e1:
        print(f"  statsmodels 失败: {e1}")
        try:
            aov_treat = pg.rm_anova(data=df_treat, dv="value", within="treat",
                                     subject="id", detailed=True)
            print(aov_treat.to_string())
        except Exception as e2:
            print(f"  分析失败: {e2}")

    # --- 单因素: channel ---
    print("\n--- 单因素重复测量方差分析: channel 效应 ---")
    df_ch = df_rm.groupby(["id", "channel"])["value"].mean().reset_index()
    df_ch_bal = _balance_data(df_ch, ["channel"])
    try:
        aovrm_c = AnovaRM(df_ch_bal, depvar="value", subject="id", within=["channel"])
        res_c = aovrm_c.fit()
        print(res_c.summary())
    except Exception as e1:
        print(f"  statsmodels 失败: {e1}")
        try:
            aov_ch = pg.rm_anova(data=df_ch_bal, dv="value", within="channel",
                                  subject="id", detailed=True)
            print(aov_ch.to_string())
        except Exception as e2:
            print(f"  分析失败: {e2}")

    return aov


def pairwise_comparisons(df):
    """事后两两比较 (Bonferroni 校正)"""
    print_separator("事后两两比较 (Post-hoc Pairwise Comparisons)")

    df_pw = df.copy()
    df_pw["id"] = df_pw["id"].astype(str)
    df_pw["treat"] = df_pw["treat"].astype(str)
    df_pw["channel"] = df_pw["channel"].astype(str)

    # 1) treat 在每个 channel 水平上的比较
    print("\n【1】在每个 channel 水平上比较 treat 的差异 (Bonferroni 校正)")
    print("    等价于 SPSS: /EMMEANS=TABLES(treat*channel) COMPARE(treat) ADJ(BONFERRONI)\n")

    channels = sorted(df_pw["channel"].unique())
    results_treat_by_channel = []

    for ch in channels:
        sub = df_pw[df_pw["channel"] == ch]
        treats = sorted(sub["treat"].unique())
        if len(treats) < 2:
            continue
        try:
            pw = pg.pairwise_tests(
                data=sub, dv="value", within="treat", subject="id",
                padjust="bonf", return_desc=True
            )
            pw.insert(0, "channel", ch)
            results_treat_by_channel.append(pw)
        except Exception:
            pass

    if results_treat_by_channel:
        df_res1 = pd.concat(results_treat_by_channel, ignore_index=True)
        cols = [c for c in ["channel", "A", "B", "mean(A)", "mean(B)", "T", "dof", "p-unc", "p-corr", "hedges"] if c in df_res1.columns]
        print(df_res1[cols].to_string(index=False))
    else:
        print("  无足够数据进行比较")

    # 2) channel 在每个 treat 水平上的比较
    print("\n\n【2】在每个 treat 水平上比较 channel 的差异 (Bonferroni 校正)")
    print("    等价于 SPSS: /EMMEANS=TABLES(treat*channel) COMPARE(channel) ADJ(BONFERRONI)\n")

    treats = sorted(df_pw["treat"].unique())
    results_ch_by_treat = []

    for tr in treats:
        sub = df_pw[df_pw["treat"] == tr]
        try:
            pw = pg.pairwise_tests(
                data=sub, dv="value", within="channel", subject="id",
                padjust="bonf", return_desc=True
            )
            pw.insert(0, "treat", tr)
            results_ch_by_treat.append(pw)
        except Exception:
            pass

    if results_ch_by_treat:
        df_res2 = pd.concat(results_ch_by_treat, ignore_index=True)
        cols = [c for c in ["treat", "A", "B", "mean(A)", "mean(B)", "T", "dof", "p-unc", "p-corr", "hedges"] if c in df_res2.columns]
        print(df_res2[cols].to_string(index=False))
    else:
        print("  无足够数据进行比较")

    # 3) treat × channel 交互: 所有组合的两两比较
    print("\n\n【3】treat × channel 所有组合的两两比较")
    df_pw["group"] = df_pw["treat"] + " | " + df_pw["channel"]
    groups = sorted(df_pw["group"].unique())
    group_means = df_pw.groupby("group")["value"].agg(["mean", "std", "count"])

    pair_results = []
    for g1, g2 in combinations(groups, 2):
        vals1 = df_pw[df_pw["group"] == g1]["value"].values
        vals2 = df_pw[df_pw["group"] == g2]["value"].values
        if len(vals1) < 2 or len(vals2) < 2:
            continue
        t_stat, p_val = stats.ttest_rel(vals1, vals2) if len(vals1) == len(vals2) else stats.ttest_ind(vals1, vals2)
        pair_results.append({
            "组1": g1, "组2": g2,
            "均值1": np.mean(vals1), "均值2": np.mean(vals2),
            "均值差": np.mean(vals1) - np.mean(vals2),
            "t": t_stat, "p(未校正)": p_val
        })

    if pair_results:
        df_pairs = pd.DataFrame(pair_results)
        n_comparisons = len(df_pairs)
        df_pairs["p(Bonferroni)"] = np.minimum(df_pairs["p(未校正)"] * n_comparisons, 1.0)
        df_pairs["显著性"] = df_pairs["p(Bonferroni)"].apply(
            lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
        sig_pairs = df_pairs[df_pairs["p(Bonferroni)"] < 0.05]
        print(f"\n  共 {n_comparisons} 个比较，其中 {len(sig_pairs)} 个在 Bonferroni 校正后显著 (p<0.05)")
        if not sig_pairs.empty:
            print("\n  显著的两两比较:")
            print(sig_pairs.to_string(index=False))
        else:
            print("  Bonferroni 校正后没有显著的两两比较")

    return


def method_manova(df):
    """方法3：多元方差分析视角 (MANOVA-like approach)"""
    print_separator("方法3：多元方差分析视角 (MANOVA)")

    df_wide = df.pivot_table(
        index="id", columns=["treat", "channel"], values="value"
    )
    df_wide.columns = [f"{t}_{c}" for t, c in df_wide.columns]
    df_wide = df_wide.reset_index()

    print("\n宽格式数据预览:")
    print(df_wide.head().to_string())

    treat_a_cols = [c for c in df_wide.columns if c.startswith("A(")]
    treat_b_cols = [c for c in df_wide.columns if c.startswith("B(")]

    if treat_a_cols and treat_b_cols:
        a_mean = df_wide[treat_a_cols].mean(axis=1)
        b_mean = df_wide[treat_b_cols].mean(axis=1)
        t_stat, p_val = stats.ttest_rel(a_mean, b_mean)
        print(f"\n配对 t 检验 (A均值 vs B均值): t = {t_stat:.4f}, p = {p_val:.4f}")

        hotelling_results = []
        channels = df["channel"].unique()
        for ch in sorted(channels):
            col_a = f"A(节律)_{ch}"
            col_b = f"B(无节律)_{ch}"
            if col_a in df_wide.columns and col_b in df_wide.columns:
                diff = df_wide[col_a] - df_wide[col_b]
                t_s, p_v = stats.ttest_1samp(diff, 0)
                hotelling_results.append({
                    "channel": ch,
                    "A均值": df_wide[col_a].mean(),
                    "B均值": df_wide[col_b].mean(),
                    "差值均值": diff.mean(),
                    "t": t_s, "p": p_v
                })
        if hotelling_results:
            df_hot = pd.DataFrame(hotelling_results)
            n_comp = len(df_hot)
            df_hot["p(Bonferroni)"] = np.minimum(df_hot["p"] * n_comp, 1.0)
            print("\n逐通道配对 t 检验 (A - B):")
            print(df_hot.to_string(index=False))


def plot_results(df, ci_df, output_prefix="rm_anova"):
    """可视化"""
    print_separator("生成可视化图表")

    df_plot = df.copy()
    df_plot["treat"] = df_plot["treat"].astype(str)
    df_plot["channel"] = df_plot["channel"].astype(str)
    df_plot["region"] = df_plot["region"].astype(str)

    # 图1: treat × channel 交互作用图
    fig, ax = plt.subplots(figsize=(14, 6))
    means = df_plot.groupby(["channel", "treat"])["value"].mean().reset_index()
    means_pivot = means.pivot(index="channel", columns="treat", values="value")
    means_pivot.plot(kind="bar", ax=ax, width=0.7, edgecolor="black", alpha=0.8)
    ax.set_title("treat × channel 交互效应 (各组均值)", fontsize=14)
    ax.set_xlabel("通道 (Channel)", fontsize=12)
    ax.set_ylabel("测量值 (Value)", fontsize=12)
    ax.legend(title="条件 (Treat)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fname1 = f"{output_prefix}_interaction_bar.png"
    fig.savefig(fname1, dpi=150)
    plt.close(fig)
    print(f"  保存: {fname1}")

    # 图2: 按脑区分面的箱线图
    regions = sorted(df_plot["region"].unique())
    n_regions = len(regions)
    fig, axes = plt.subplots(1, n_regions, figsize=(6 * n_regions, 5), sharey=False)
    if n_regions == 1:
        axes = [axes]
    for i, region in enumerate(regions):
        sub = df_plot[df_plot["region"] == region]
        sns.boxplot(data=sub, x="channel", y="value", hue="treat", ax=axes[i])
        axes[i].set_title(region, fontsize=11)
        axes[i].tick_params(axis="x", rotation=45)
        if i > 0:
            axes[i].get_legend().remove()
    plt.suptitle("按脑区分组的条件比较", fontsize=14, y=1.02)
    plt.tight_layout()
    fname2 = f"{output_prefix}_boxplot_by_region.png"
    fig.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {fname2}")

    # 图3: 置信区间图
    if ci_df is not None and not ci_df.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        ci_sorted = ci_df.sort_values(["channel", "treat"])
        x_labels = [f"{r['channel']}|{r['treat']}" for _, r in ci_sorted.iterrows()]
        x_pos = np.arange(len(ci_sorted))
        colors = ["#1f77b4" if "A" in str(t) else "#ff7f0e" for t in ci_sorted["treat"]]
        ax.errorbar(
            x_pos, ci_sorted["均值"],
            yerr=[
                ci_sorted["均值"] - ci_sorted["95%CI下限"],
                ci_sorted["95%CI上限"] - ci_sorted["均值"]
            ],
            fmt="o", capsize=4, capthick=1.5, markersize=6,
            color="black", ecolor="gray"
        )
        for xi, yi, c in zip(x_pos, ci_sorted["均值"], colors):
            ax.scatter(xi, yi, color=c, s=60, zorder=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
        ax.set_title("各组均数及95%置信区间", fontsize=14)
        ax.set_ylabel("测量值", fontsize=12)
        ax.axhline(y=df["value"].mean(), color="red", linestyle="--", alpha=0.5, label="总均值")
        ax.legend()
        plt.tight_layout()
        fname3 = f"{output_prefix}_ci_plot.png"
        fig.savefig(fname3, dpi=150)
        plt.close(fig)
        print(f"  保存: {fname3}")

    # 图4: 个体轨迹图
    fig, ax = plt.subplots(figsize=(12, 6))
    for subj_id in df_plot["id"].unique():
        sub = df_plot[df_plot["id"] == subj_id]
        sub_a = sub[sub["treat"].str.contains("A")].sort_values("channel")
        sub_b = sub[sub["treat"].str.contains("B")].sort_values("channel")
        channels_sorted = sorted(sub["channel"].unique())
        ch_idx = {c: i for i, c in enumerate(channels_sorted)}
        if not sub_a.empty:
            ax.plot([ch_idx[c] for c in sub_a["channel"]], sub_a["value"].values,
                    "o-", alpha=0.5, label=f"id={subj_id} A" if subj_id == df_plot["id"].unique()[0] else "")
        if not sub_b.empty:
            ax.plot([ch_idx[c] for c in sub_b["channel"]], sub_b["value"].values,
                    "s--", alpha=0.5, label=f"id={subj_id} B" if subj_id == df_plot["id"].unique()[0] else "")
    ax.set_xticks(range(len(channels_sorted)))
    ax.set_xticklabels(channels_sorted, rotation=45)
    ax.set_title("个体在各通道上的测量轨迹", fontsize=14)
    ax.set_xlabel("通道 (Channel)")
    ax.set_ylabel("测量值")
    plt.tight_layout()
    fname4 = f"{output_prefix}_individual_trajectories.png"
    fig.savefig(fname4, dpi=150)
    plt.close(fig)
    print(f"  保存: {fname4}")


def main():
    parser = argparse.ArgumentParser(
        description="重复测量资料的方差分析 (Repeated Measures ANOVA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python repeated_measures_anova.py                         # 使用内置示例数据，运行所有方法
  python repeated_measures_anova.py --input data.tsv        # 使用外部数据文件
  python repeated_measures_anova.py --method mixed          # 仅运行混合效应模型
  python repeated_measures_anova.py --method rm_anova       # 仅运行传统重复测量方差分析
  python repeated_measures_anova.py --cov_struct cs         # 使用复合对称协方差结构

数据文件格式 (Tab分隔，无表头):
  id  region  channel  treat  value
  1.0 额极区  Fpz      A(节律) 2.109
        """
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入数据文件路径 (TSV/CSV)，不提供则使用内置示例数据")
    parser.add_argument("--method", "-m", type=str, default="all",
                        choices=["all", "mixed", "rm_anova", "manova"],
                        help="分析方法: all=全部, mixed=混合效应模型, rm_anova=传统重复测量, manova=多元方差分析")
    parser.add_argument("--cov_struct", type=str, default="vc",
                        choices=["vc", "cs"],
                        help="混合效应模型协方差结构: vc=方差分量, cs=复合对称")
    parser.add_argument("--no_plot", action="store_true",
                        help="不生成可视化图表")
    parser.add_argument("--output_prefix", "-o", type=str, default="rm_anova",
                        help="输出文件名前缀")

    args = parser.parse_args()

    print_separator("重复测量资料的方差分析")
    print(f"  分析方法: {args.method}")
    if args.input:
        print(f"  数据文件: {args.input}")
    else:
        print("  数据来源: 内置示例数据")

    df = load_data(args.input)

    print(f"\n  数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  被试数量: {df['id'].nunique()}")
    print(f"  条件水平: {list(df['treat'].cat.categories)}")
    print(f"  通道数量: {df['channel'].nunique()} ({', '.join(sorted(df['channel'].unique().astype(str)))})")
    print(f"  脑区数量: {df['region'].nunique()}")

    ci_df = descriptive_stats(df)

    if args.method in ("all", "mixed"):
        method_mixed_effects(df, cov_struct=args.cov_struct)

    if args.method in ("all", "rm_anova"):
        method_rm_anova(df)

    if args.method in ("all", "manova"):
        method_manova(df)

    pairwise_comparisons(df)

    if not args.no_plot:
        plot_results(df, ci_df, output_prefix=args.output_prefix)

    print_separator("分析完成")
    print("""
分析说明:
  1. 混合效应模型 (推荐): 以 id 为随机效应，treat 和 channel 为固定效应
     - 可处理不完整数据和不等方差
     - 可选择不同协方差结构 (方差分量、复合对称等)
  2. 传统重复测量方差分析: 基于一般线性模型
     - 需满足球形性假设 (Mauchly's test)
     - 不满足时需使用 Greenhouse-Geisser 校正
  3. 多元方差分析: 不需要球形性假设
     - 但对样本量要求较高

  事后两两比较采用 Bonferroni 校正控制多重比较的 I 类错误率。
  等价于 SPSS 中的:
    /EMMEANS=TABLES(treat*channel) COMPARE(treat) ADJ(BONFERRONI)
    /EMMEANS=TABLES(treat*channel) COMPARE(channel) ADJ(BONFERRONI)
""")


if __name__ == "__main__":
    main()
