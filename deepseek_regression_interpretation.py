#!/usr/bin/env python3
"""Use DeepSeek API to interpret linear regression validation results."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from linear_regression_validation import (
    RAW_DATA,
    REFERENCE,
    RegressionResult,
    linear_regression_with_ci,
    parse_data,
)


def load_xy_data(path: str | None) -> tuple[list[float], list[float]]:
    if path is None:
        return parse_data(RAW_DATA)
    file_text = Path(path).read_text(encoding="utf-8")
    return parse_data(file_text)


def result_to_dict(result: RegressionResult) -> dict[str, Any]:
    return {
        "best_fit": {
            "slope": result.slope,
            "y_intercept": result.intercept,
            "x_intercept": result.x_intercept,
            "inv_slope": result.inv_slope,
            "equation": f"Y = {result.slope:.5f}*X {result.intercept:+.5f}",
        },
        "std_error": {
            "slope": result.se_slope,
            "y_intercept": result.se_intercept,
        },
        "confidence_interval_95": {
            "slope": {"low": result.ci_slope[0], "high": result.ci_slope[1]},
            "y_intercept": {"low": result.ci_intercept[0], "high": result.ci_intercept[1]},
            "x_intercept": {"low": result.ci_x_intercept[0], "high": result.ci_x_intercept[1]},
        },
        "goodness_of_fit": {
            "r_squared": result.r2,
            "sy_x": result.syx,
        },
        "slope_significance": {
            "f": result.f_stat,
            "dfn": result.dfn,
            "dfd": result.dfd,
            "p_value": result.p_value,
            "significant": result.p_value < 0.05,
        },
        "data_info": {
            "n_x_values": result.n,
            "max_y_replicates": result.max_y_replicates,
            "total_values": result.total_values,
            "missing_values": result.missing_values,
        },
    }


def build_prompt(metrics: dict[str, Any], reference: dict[str, Any]) -> str:
    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    reference_text = json.dumps(reference, ensure_ascii=False, indent=2)

    return textwrap.dedent(
        f"""
        你是一位严谨的统计分析顾问。请基于下面的线性回归结果进行中文解读。

        【计算得到的统计结果】
        {metrics_text}

        【用于对照的目标结果（来自外部软件）】
        {reference_text}

        请按以下结构输出，并保证结论可落地：
        1) 一句话总评（是否存在显著线性相关）
        2) 关键统计解释（斜率、截距、R²、F、P值、95%CI）
        3) 结果可靠性与局限（样本量、离群点敏感性、解释力不足风险）
        4) 对下一步分析的建议（至少4条，可执行）
        5) 给非统计背景读者的通俗说明（100字以内）

        注意：
        - 不能杜撰未提供的信息（如变量单位、实验背景）。
        - 强调“统计显著”与“实际意义”可能不同。
        - 输出使用简体中文，条理清晰，尽量使用要点符号。
        """
    ).strip()


def call_deepseek_api(
    api_key: str,
    prompt: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.2,
    max_tokens: int = 1400,
    timeout: int = 60,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是严谨、客观、可解释的统计顾问，擅长线性回归结果解读。",
            },
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
        raise RuntimeError(f"无法解析DeepSeek响应: {body}") from exc


def build_markdown_report(
    metrics: dict[str, Any],
    interpretation: str,
    data_source: str,
    model: str,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics_block = json.dumps(metrics, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        # 线性回归结果解读报告（DeepSeek）

        - 生成时间: {generated_at}
        - 数据来源: {data_source}
        - 模型: {model}

        ## 回归统计结果（程序计算）

        ```json
        {metrics_block}
        ```

        ## DeepSeek 解读

        {interpretation}
        """
    ).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用 DeepSeek API 对线性回归统计结果进行自动解读"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="可选：输入数据文件路径（每行两个数：X Y）。不传则使用内置数据。",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DeepSeek API Key（不传则从环境变量 DEEPSEEK_API_KEY 读取）。",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek Base URL。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="模型名，例如 deepseek-chat 或 deepseek-reasoner。",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度。")
    parser.add_argument("--max-tokens", type=int, default=1400, help="最大输出token。")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP超时（秒）。")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：将完整报告保存到 Markdown 文件。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要发送的 prompt，不调用 API。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xs, ys = load_xy_data(args.data_file)
    result = linear_regression_with_ci(xs, ys)
    metrics = result_to_dict(result)
    prompt = build_prompt(metrics, REFERENCE)

    if args.dry_run:
        print("=== DRY RUN: Prompt Preview ===")
        print(prompt)
        return

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "未检测到 API Key。请通过 --api-key 传入，或设置环境变量 DEEPSEEK_API_KEY。"
        )

    interpretation = call_deepseek_api(
        api_key=api_key,
        prompt=prompt,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    print("=== DeepSeek 解读结果 ===")
    print(interpretation)

    if args.output:
        report = build_markdown_report(
            metrics=metrics,
            interpretation=interpretation,
            data_source=args.data_file or "内置数据",
            model=args.model,
        )
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
