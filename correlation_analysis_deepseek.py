#!/usr/bin/env python3
"""Correlation analysis (9 pairs) and DeepSeek interpretation."""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


# 你给的原始数据（制表符/空格都可）
RAW_DATA = """
d	O2	Oz	O1	rt	correct
0.86	0.34	0.65	0.41	-49.18	3.32
0.2	-0.21	0.04	0.04	-21.25	3.55
-0.17	-0.08	-0.04	0.01	1.82	-1.17
0.18	-0.17	-0.17	-0.3	11.21	3.77
0.28	-0.13	0	-0.04	-10.24	3.47
-0.37	-0.5	-0.58	-0.92	31.39	-5.84
0.45	0.22	0.16	0.19	-43.24	2.04
-0.31	0.41	0.38	0.85	-4.09	-4.6
0.17	0.28	0.27	0.31	-25.12	2.54
-0.11	4.21	4.25	6.28	-30.05	0.55
-0.17	1.19	1.58	1.94	-7.59	-3.52
-0.05	0.37	0.52	0.36	-31	-0.06
0.26	0.38	0.43	0.9	-5.05	0.32
1.08	0.55	0.4	0.22	-0.82	3.83
0.04	0.09	-0.02	-0.11	5.62	2.45
0.25	0.51	0.73	1.01	-9.49	0.42
0.12	-0.21	-0.37	-0.51	35.8	1.28
-0.47	-0.22	-0.25	-0.3	-1.37	0.43
0.33	6.69	3.6	3.19	1.07	-0.54
0.19	1.19	0.85	0.69	4.14	3.76
0.52	0.09	0.06	0.2	-37.32	1.25
0.79	2.17	1.74	1.9	18.78	1.31
0.45	0.28	0.11	0.02	-22.55	-0.36
0.95	0.27	0.31	0.34	2.92	2.94
0.41	-2.75	-2.67	-2.85	10.67	3.6
0.16	1.32	1.15	1.21	21.34	2.28
0.37	0.05	0.13	0.05	10.61	3.3
0.37	1.03	1.52	1.94	-1.29	4.11
0.01	0.13	0.34	0.03	-32.91	0
0.06	0.79	0.67	-1.04	6.18	1.01
0	0.77	0.96	1.63	-36.33	0
-0.22	-0.39	-0.3	-0.34	13.51	0.1
0.19	0.6	1.16	1.2	-12.24	3.44
0.02	0.12	0.11	0.11	-11.24	0.42
0.29	0.49	0.32	0.44	-17.06	4.17
0.68	0.38	0.33	0.01	1.08	0.88
-0.09	-0.21	-0.15	-0.16	-17.88	0.01
0.63	-0.17	-0.08	0	-3.01	3.9
0.06	1.25	0.93	0.74	-17.52	5.06
0.12	0.08	0	-0.07	-31.75	-4.35
0.08	1.32	0.86	0.92	-18.61	0.62
0.04	0.29	0.43	1.09	-13.42	0.54
0.01	0.19	0.4	0.48	-28.73	-0.2
""".strip()

CHANNELS = ["O2", "Oz", "O1"]
TARGETS = ["d", "rt", "correct"]
MISSING_TOKENS = {"", "na", "nan", "null", "none", "-"}


@dataclass
class CorrelationResult:
    x_var: str
    y_var: str
    n: int
    r: float
    r_squared: float
    t_stat: float
    p_value: float
    ci95_low: float
    ci95_high: float
    significant: bool
    effect_size: str


def _to_float(token: str) -> float | None:
    value = token.strip()
    if value.lower() in MISSING_TOKENS:
        return None
    return float(value)


def parse_table(raw_text: str) -> dict[str, list[float | None]]:
    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("数据行数不足，至少需要表头 + 1行数据")

    headers = lines[0].split()
    data: dict[str, list[float | None]] = {h: [] for h in headers}

    for row in lines[1:]:
        parts = row.split()
        if len(parts) != len(headers):
            raise ValueError(f"数据列数不匹配: {row!r}")
        for h, token in zip(headers, parts):
            data[h].append(_to_float(token))

    return data


def student_t_pdf(t: float, df: int) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    half_df = 0.5 * df
    log_coef = math.lgamma(0.5 * (df + 1.0)) - math.lgamma(half_df)
    log_coef -= 0.5 * (math.log(df) + math.log(math.pi))
    log_shape = -0.5 * (df + 1.0) * math.log1p((t * t) / df)
    return math.exp(log_coef + log_shape)


def _simpson(f, a: float, b: float) -> float:
    c = 0.5 * (a + b)
    return (b - a) * (f(a) + 4.0 * f(c) + f(b)) / 6.0


def _adaptive_simpson(f, a: float, b: float, eps: float, depth: int) -> float:
    whole = _simpson(f, a, b)

    def recurse(left: float, right: float, budget: float, prev: float, n_depth: int) -> float:
        mid = 0.5 * (left + right)
        left_area = _simpson(f, left, mid)
        right_area = _simpson(f, mid, right)
        delta = left_area + right_area - prev
        if n_depth <= 0 or abs(delta) <= 15.0 * budget:
            return left_area + right_area + delta / 15.0
        return recurse(left, mid, budget / 2.0, left_area, n_depth - 1) + recurse(
            mid, right, budget / 2.0, right_area, n_depth - 1
        )

    return recurse(a, b, eps, whole, depth)


def student_t_cdf(t: float, df: int) -> float:
    if t == 0.0:
        return 0.5
    if t < 0.0:
        return 1.0 - student_t_cdf(-t, df)
    integral = _adaptive_simpson(lambda u: student_t_pdf(u, df), 0.0, t, eps=1e-10, depth=25)
    cdf = 0.5 + integral
    return max(0.0, min(1.0, cdf))


def effect_size_label(abs_r: float) -> str:
    if abs_r < 0.1:
        return "可忽略"
    if abs_r < 0.3:
        return "弱相关"
    if abs_r < 0.5:
        return "中等相关"
    if abs_r < 0.7:
        return "较强相关"
    return "强相关"


def pearson_correlation_with_stats(
    x_values: list[float | None], y_values: list[float | None], alpha: float = 0.05
) -> tuple[int, float, float, float, float, float]:
    pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    n = len(pairs)
    if n < 4:
        raise ValueError("有效样本太少，至少需要4个有效配对值")

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    syy = sum((y - y_mean) ** 2 for y in ys)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        raise ValueError("某一变量方差为0，无法计算相关系数")

    r = sxy / math.sqrt(sxx * syy)
    r = max(min(r, 1.0), -1.0)

    df = n - 2
    if abs(r) == 1.0:
        t_stat = math.inf
        p_value = 0.0
    else:
        t_stat = r * math.sqrt(df / (1.0 - r * r))
        p_value = 2.0 * (1.0 - student_t_cdf(abs(t_stat), df))

    # Fisher z-transform 近似95%CI
    z = math.atanh(max(min(r, 0.999999999), -0.999999999))
    se_z = 1.0 / math.sqrt(n - 3)
    z_crit = 1.959963984540054
    ci_low = math.tanh(z - z_crit * se_z)
    ci_high = math.tanh(z + z_crit * se_z)

    return n, r, t_stat, p_value, ci_low, ci_high


def compute_all_correlations(data: dict[str, list[float | None]]) -> list[CorrelationResult]:
    for required in CHANNELS + TARGETS:
        if required not in data:
            raise ValueError(f"数据中缺少列: {required}")

    results: list[CorrelationResult] = []
    for ch in CHANNELS:
        for target in TARGETS:
            n, r, t_stat, p_value, ci_low, ci_high = pearson_correlation_with_stats(
                data[ch], data[target]
            )
            results.append(
                CorrelationResult(
                    x_var=ch,
                    y_var=target,
                    n=n,
                    r=r,
                    r_squared=r * r,
                    t_stat=t_stat,
                    p_value=p_value,
                    ci95_low=ci_low,
                    ci95_high=ci_high,
                    significant=p_value < 0.05,
                    effect_size=effect_size_label(abs(r)),
                )
            )
    return results


def print_correlation_table(results: list[CorrelationResult]) -> None:
    print("9种相关性分析结果（Pearson）")
    print("-" * 105)
    print(
        f"{'通道':<6}{'变量':<10}{'n':>5}{'r':>12}{'R^2':>12}{'p值':>14}"
        f"{'95%CI(r)':>28}{'显著性':>10}{'强度':>10}"
    )
    print("-" * 105)
    for r in results:
        ci_text = f"[{r.ci95_low:.3f}, {r.ci95_high:.3f}]"
        sig = "显著" if r.significant else "不显著"
        print(
            f"{r.x_var:<6}{r.y_var:<10}{r.n:>5d}{r.r:>12.4f}{r.r_squared:>12.4f}"
            f"{r.p_value:>14.6f}{ci_text:>28}{sig:>10}{r.effect_size:>10}"
        )
    print("-" * 105)


def summarize_results(results: list[CorrelationResult]) -> dict[str, Any]:
    sorted_by_abs = sorted(results, key=lambda item: abs(item.r), reverse=True)
    significant_pairs = [r for r in results if r.significant]
    return {
        "total_pairs": len(results),
        "significant_pairs_count": len(significant_pairs),
        "significant_pairs": [
            {"pair": f"{r.x_var} vs {r.y_var}", "r": r.r, "p": r.p_value} for r in significant_pairs
        ],
        "strongest_pair": {
            "pair": f"{sorted_by_abs[0].x_var} vs {sorted_by_abs[0].y_var}",
            "r": sorted_by_abs[0].r,
            "p": sorted_by_abs[0].p_value,
        },
        "weakest_pair": {
            "pair": f"{sorted_by_abs[-1].x_var} vs {sorted_by_abs[-1].y_var}",
            "r": sorted_by_abs[-1].r,
            "p": sorted_by_abs[-1].p_value,
        },
    }


def build_prompt(results: list[CorrelationResult], summary: dict[str, Any]) -> str:
    results_json = json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        你是一个严谨的生物信号统计分析顾问。下面是 9 组 Pearson 相关性分析结果，
        对象是 O2/Oz/O1 通道与 d/rt/correct 三个行为变量之间的相关关系。

        【9组相关性结果】
        {results_json}

        【自动汇总】
        {summary_json}

        请输出中文解读，结构如下：
        1) 总体结论（哪些关系最值得关注）
        2) 按目标变量分组解读（d、rt、correct 各自对应 O2/Oz/O1）
        3) 统计显著性与效应大小的区别（避免只看 p 值）
        4) 潜在数据问题与方法局限（样本量、异常值、多重比较等）
        5) 下一步可执行建议（至少 5 条）
        6) 非统计背景读者版摘要（80~120字）

        注意：
        - 不要杜撰实验背景或单位；
        - 明确指出“相关不代表因果”；
        - 如果结果整体相关性弱，要直接说明。
        """
    ).strip()


def call_deepseek_api(
    api_key: str,
    prompt: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.2,
    max_tokens: int = 1800,
    timeout: int = 60,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨、客观、实事求是的统计分析助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP错误: {exc.code}, detail={detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"DeepSeek API网络错误: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"无法解析 DeepSeek 响应: {body}") from exc


def build_markdown_report(
    results: list[CorrelationResult],
    summary: dict[str, Any],
    interpretation: str,
    source: str,
    model: str,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    results_json = json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        # O2/Oz/O1 与 d/rt/correct 相关性分析报告（DeepSeek）

        - 生成时间: {generated_at}
        - 数据来源: {source}
        - 模型: {model}

        ## 统计结果（程序计算）

        ### 9组相关性详情
        ```json
        {results_json}
        ```

        ### 汇总
        ```json
        {summary_json}
        ```

        ## DeepSeek 解读

        {interpretation}
        """
    ).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算9组相关性（O2/Oz/O1 × d/rt/correct）并调用DeepSeek解读"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="可选：输入数据文件路径，需包含表头 d O2 Oz O1 rt correct。",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DeepSeek API Key（优先级高于环境变量 DEEPSEEK_API_KEY）。",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek Base URL。",
    )
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek 模型名。")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度。")
    parser.add_argument("--max-tokens", type=int, default=1800, help="最大输出 token。")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP 超时（秒）。")
    parser.add_argument("--output", type=str, default=None, help="可选：保存 Markdown 报告路径。")
    parser.add_argument("--save-json", type=str, default=None, help="可选：保存原始统计结果 JSON。")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调用 API。")
    return parser.parse_args()


def load_data_text(file_path: str | None) -> tuple[str, str]:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8"), file_path
    return RAW_DATA, "内置数据"


def main() -> None:
    args = parse_args()
    raw_text, source_name = load_data_text(args.data_file)
    data = parse_table(raw_text)
    results = compute_all_correlations(data)
    summary = summarize_results(results)

    print_correlation_table(results)
    print("汇总信息:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(
                {
                    "source": source_name,
                    "results": [asdict(item) for item in results],
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n原始统计结果已保存到: {args.save_json}")

    prompt = build_prompt(results, summary)
    if args.dry_run:
        print("\n=== DRY RUN: Prompt Preview ===")
        print(prompt)
        return

    # API Key 填写位置（推荐方式）
    # 1) 命令行传入: --api-key "sk-xxx"
    # 2) 环境变量: DEEPSEEK_API_KEY
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print(
            "\n未检测到 DeepSeek API Key，已跳过 AI 解读。\n"
            "请在以下任一位置填写：\n"
            "  1) 运行参数 --api-key \"sk-...\"\n"
            "  2) 环境变量 DEEPSEEK_API_KEY（推荐）"
        )
        return

    interpretation = call_deepseek_api(
        api_key=api_key,
        prompt=prompt,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    print("\n=== DeepSeek 解读结果 ===")
    print(interpretation)

    if args.output:
        report = build_markdown_report(
            results=results,
            summary=summary,
            interpretation=interpretation,
            source=source_name,
            model=args.model,
        )
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
