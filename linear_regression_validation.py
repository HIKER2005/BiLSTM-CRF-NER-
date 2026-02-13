#!/usr/bin/env python3
"""Linear regression validation for the provided (X, Y) dataset."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


RAW_DATA = """
0.074074147    -49.18
0.042175717    -21.25
-0.010432505   1.82
-0.324693005   11.21
-0.159130379   -10.24
-0.067149903   31.39
0.018945788    -43.24
0.170206422    -4.09
0.00120942     -25.12
0.419911958    -30.05
0.119656597    -7.59
0.053735301    -31
0.014590166    -5.05
0.072722102    -0.82
0.037729436    5.62
0.019260902    -9.49
-0.660610389   35.8
-0.12201123    -1.37
0.055712719    1.07
0.091488908    4.14
0.015649958    -37.32
0.138696965    18.78
0.012981518    -22.55
0.147314406    2.92
-0.585465507   10.67
0.023762297    21.34
-0.044165184   10.61
0.147291987    -1.29
-0.03599094    -32.91
0.381625906    6.18
0.10950026     -36.33
-0.024857021   13.51
0.03740627     -12.24
0.020332066    -11.24
0.020312753    -17.06
-0.029324311   1.08
-0.095281844   -17.88
0.060885882    -3.01
0.241030075    -17.52
-0.087149299   -31.75
0.390932191    -18.61
0.086411893    -13.42
0.009651307    -28.73
""".strip()


REFERENCE = {
    "slope": -34.71,
    "intercept": -7.833,
    "x_intercept": -0.2257,
    "inv_slope": -0.02881,
    "se_slope": 14.50,
    "se_intercept": 2.837,
    "ci_slope_low": -63.99,
    "ci_slope_high": -5.420,
    "ci_intercept_low": -13.56,
    "ci_intercept_high": -2.104,
    "ci_x_intercept_low": -1.607,
    "ci_x_intercept_high": -0.05119,
    "r2": 0.1226,
    "syx": 18.52,
    "f": 5.728,
    "p_value": 0.0214,
    "dfn": 1,
    "dfd": 41,
    "n": 43,
}


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    x_intercept: float
    inv_slope: float
    se_slope: float
    se_intercept: float
    ci_slope: tuple[float, float]
    ci_intercept: tuple[float, float]
    ci_x_intercept: tuple[float, float]
    r2: float
    syx: float
    f_stat: float
    p_value: float
    dfn: int
    dfd: int
    n: int
    max_y_replicates: int
    total_values: int
    missing_values: int


def parse_data(raw: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid line: {line!r}")
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))
    return xs, ys


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


def student_t_ppf(prob: float, df: int) -> float:
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be in (0, 1)")
    if prob == 0.5:
        return 0.0
    if prob < 0.5:
        return -student_t_ppf(1.0 - prob, df)

    low = 0.0
    high = 1.0
    while student_t_cdf(high, df) < prob:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("Unable to bracket t quantile")

    for _ in range(80):
        mid = 0.5 * (low + high)
        if student_t_cdf(mid, df) < prob:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def linear_regression_with_ci(xs: list[float], ys: list[float], alpha: float = 0.05) -> RegressionResult:
    n = len(xs)
    if n != len(ys):
        raise ValueError("X and Y lengths differ")
    if n < 3:
        raise ValueError("At least 3 data points are required")

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    syy = sum((y - y_mean) ** 2 for y in ys)

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    y_hat = [intercept + slope * x for x in xs]
    residuals = [y - y_pred for y, y_pred in zip(ys, y_hat)]

    sse = sum(r * r for r in residuals)
    ssr = sum((pred - y_mean) ** 2 for pred in y_hat)

    dfn = 1
    dfd = n - 2
    mse = sse / dfd
    syx = math.sqrt(mse)

    se_slope = syx / math.sqrt(sxx)
    se_intercept = syx * math.sqrt((1.0 / n) + (x_mean * x_mean) / sxx)

    t_crit = student_t_ppf(1.0 - alpha / 2.0, dfd)
    ci_slope = (slope - t_crit * se_slope, slope + t_crit * se_slope)
    ci_intercept = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)

    x_intercept = -intercept / slope
    inv_slope = 1.0 / slope

    # Fieller confidence interval for ratio -intercept / slope.
    cov_b0_b1 = -x_mean * mse / sxx
    t2 = t_crit * t_crit
    a = slope * slope - t2 * se_slope * se_slope
    b = 2.0 * (intercept * slope - t2 * cov_b0_b1)
    c = intercept * intercept - t2 * se_intercept * se_intercept
    disc = b * b - 4.0 * a * c
    if disc < 0:
        raise RuntimeError("Fieller CI failed: discriminant < 0")
    root_disc = math.sqrt(disc)
    x_ci_low = (-b - root_disc) / (2.0 * a)
    x_ci_high = (-b + root_disc) / (2.0 * a)
    ci_x_intercept = (min(x_ci_low, x_ci_high), max(x_ci_low, x_ci_high))

    r2 = ssr / syy
    f_stat = (ssr / dfn) / mse
    t_stat = slope / se_slope
    p_value = 2.0 * (1.0 - student_t_cdf(abs(t_stat), dfd))

    max_y_replicates = max(Counter(xs).values())
    return RegressionResult(
        slope=slope,
        intercept=intercept,
        x_intercept=x_intercept,
        inv_slope=inv_slope,
        se_slope=se_slope,
        se_intercept=se_intercept,
        ci_slope=ci_slope,
        ci_intercept=ci_intercept,
        ci_x_intercept=ci_x_intercept,
        r2=r2,
        syx=syx,
        f_stat=f_stat,
        p_value=p_value,
        dfn=dfn,
        dfd=dfd,
        n=n,
        max_y_replicates=max_y_replicates,
        total_values=n,
        missing_values=0,
    )


def print_report(result: RegressionResult) -> None:
    print("Best-fit values")
    print(f"  Slope                  {result.slope:.5f}")
    print(f"  Y-intercept            {result.intercept:.5f}")
    print(f"  X-intercept            {result.x_intercept:.6f}")
    print(f"  1/slope                {result.inv_slope:.5f}")
    print()

    print("Std. Error")
    print(f"  Slope                  {result.se_slope:.5f}")
    print(f"  Y-intercept            {result.se_intercept:.5f}")
    print()

    print("95% Confidence Intervals")
    print(f"  Slope                  {result.ci_slope[0]:.5f} to {result.ci_slope[1]:.5f}")
    print(f"  Y-intercept            {result.ci_intercept[0]:.5f} to {result.ci_intercept[1]:.5f}")
    print(
        "  X-intercept            "
        f"{result.ci_x_intercept[0]:.6f} to {result.ci_x_intercept[1]:.6f}"
    )
    print()

    print("Goodness of Fit")
    print(f"  R squared              {result.r2:.6f}")
    print(f"  Sy.x                   {result.syx:.5f}")
    print()

    print("Is slope significantly non-zero?")
    print(f"  F                      {result.f_stat:.5f}")
    print(f"  DFn, DFd               {result.dfn}, {result.dfd}")
    print(f"  P value                {result.p_value:.6f}")
    print(f"  Deviation from zero?   {'Significant' if result.p_value < 0.05 else 'Not significant'}")
    print()

    print("Equation")
    print(f"  Y = {result.slope:.5f}*X {result.intercept:+.5f}")
    print()

    print("Data")
    print(f"  Number of X values     {result.n}")
    print(f"  Maximum Y replicates   {result.max_y_replicates}")
    print(f"  Total number of values {result.total_values}")
    print(f"  Number of missing      {result.missing_values}")
    print()


def validate_against_reference(result: RegressionResult) -> None:
    checks = [
        ("Slope", result.slope, REFERENCE["slope"], 0.01),
        ("Y-intercept", result.intercept, REFERENCE["intercept"], 0.01),
        ("X-intercept", result.x_intercept, REFERENCE["x_intercept"], 0.001),
        ("1/slope", result.inv_slope, REFERENCE["inv_slope"], 0.0001),
        ("SE(slope)", result.se_slope, REFERENCE["se_slope"], 0.01),
        ("SE(intercept)", result.se_intercept, REFERENCE["se_intercept"], 0.01),
        ("CI slope low", result.ci_slope[0], REFERENCE["ci_slope_low"], 0.01),
        ("CI slope high", result.ci_slope[1], REFERENCE["ci_slope_high"], 0.01),
        ("CI intercept low", result.ci_intercept[0], REFERENCE["ci_intercept_low"], 0.01),
        ("CI intercept high", result.ci_intercept[1], REFERENCE["ci_intercept_high"], 0.01),
        ("CI X-int low", result.ci_x_intercept[0], REFERENCE["ci_x_intercept_low"], 0.001),
        ("CI X-int high", result.ci_x_intercept[1], REFERENCE["ci_x_intercept_high"], 0.001),
        ("R squared", result.r2, REFERENCE["r2"], 0.0001),
        ("Sy.x", result.syx, REFERENCE["syx"], 0.01),
        ("F", result.f_stat, REFERENCE["f"], 0.001),
        ("P value", result.p_value, REFERENCE["p_value"], 0.0002),
        ("DFn", float(result.dfn), float(REFERENCE["dfn"]), 0.0),
        ("DFd", float(result.dfd), float(REFERENCE["dfd"]), 0.0),
        ("N", float(result.n), float(REFERENCE["n"]), 0.0),
    ]

    print("Validation against target values")
    all_ok = True
    for name, calc, target, tol in checks:
        diff = abs(calc - target)
        ok = diff <= tol
        all_ok = all_ok and ok
        status = "OK" if ok else "CHECK"
        print(
            f"  {name:<18} calc={calc:>11.6f} "
            f"target={target:>11.6f} diff={diff:>10.6f} [{status}]"
        )
    print()
    print(f"Overall validation: {'PASS' if all_ok else 'NEEDS REVIEW'}")


def main() -> None:
    xs, ys = parse_data(RAW_DATA)
    result = linear_regression_with_ci(xs, ys)
    print_report(result)
    validate_against_reference(result)


if __name__ == "__main__":
    main()
