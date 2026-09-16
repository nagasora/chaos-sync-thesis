"""E2E の臨界減速を測る平均・差分座標の数値積分器。

有限差分では tan の加減公式を積形式で評価し、十分小さい差分だけを
同期多様体まわりの線形化で進める。線形化は局所近似であり、有限差分の
力学を大域的に置き換えるものではない。
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


LOG_FLOAT_MAX = math.log(np.finfo(np.float64).max)
EMPTY_CHECKPOINTS = np.empty(0, dtype=np.int64)


@njit(cache=True)
def step(
    s: float,
    log_abs_d: float,
    beta: float,
    kappa: float,
    linear_eta: float = 1e-6,
) -> tuple[float, float, bool, bool]:
    """平均 ``s`` と ``log|d|`` を1ステップ進める。

    ``d=(x-y)/2`` の符号は交換対称性により不要なので絶対値だけを保持する。
    解決不能な非有限値または特異点では、値を丸めず ``valid=False`` を返す。
    """
    if (
        not math.isfinite(s)
        or not math.isfinite(beta)
        or beta <= 0.0
        or not math.isfinite(kappa)
        or not math.isfinite(linear_eta)
        or linear_eta <= 0.0
        or math.isnan(log_abs_d)
        or log_abs_d == math.inf
    ):
        return math.nan, math.nan, False, False

    a = beta * s
    if not math.isfinite(a):
        return math.nan, math.nan, False, False

    cos_a = math.cos(a)
    abs_cos_a = abs(cos_a)
    if log_abs_d == -math.inf:
        s_new = math.tan(a)
        if not math.isfinite(s_new):
            return math.nan, math.nan, True, False
        return s_new, -math.inf, True, True

    transverse = abs(1.0 - 2.0 * kappa)
    if not math.isfinite(transverse):
        return math.nan, math.nan, False, False

    log_beta = math.log(beta)
    log_abs_b = log_abs_d + log_beta
    log_eta = math.log(linear_eta)
    use_linear = False
    if abs_cos_a > 0.0 and log_abs_b <= log_eta:
        # d を実数へ戻す前に比較し、log|d| の underflow を床値で隠さない。
        use_linear = log_abs_b - math.log(abs_cos_a) <= log_eta

    if use_linear:
        s_new = math.tan(a)
        if not math.isfinite(s_new):
            return math.nan, math.nan, True, False
        if transverse == 0.0:
            return s_new, -math.inf, True, True
        log_d_new = (
            log_abs_d
            + math.log(transverse)
            + log_beta
            - 2.0 * math.log(abs_cos_a)
        )
        if math.isnan(log_d_new) or log_d_new == math.inf:
            return math.nan, math.nan, True, False
        return s_new, log_d_new, True, True

    # Why not: exp を上限で clip すると、本来解決不能な巨大差分を別軌道として進めてしまう。
    if log_abs_d > LOG_FLOAT_MAX:
        return math.nan, math.nan, False, False
    abs_d = math.exp(log_abs_d)
    if abs_d == 0.0 or not math.isfinite(abs_d):
        return math.nan, math.nan, False, False
    b = beta * abs_d
    if not math.isfinite(b):
        return math.nan, math.nan, False, False

    sin_a = math.sin(a)
    sin_b = math.sin(b)
    cos_b = math.cos(b)
    common = cos_a * cos_b
    cross = sin_a * sin_b
    denominator = (common - cross) * (common + cross)
    if denominator == 0.0 or not math.isfinite(denominator):
        return math.nan, math.nan, False, False

    s_new = sin_a * cos_a / denominator
    if not math.isfinite(s_new):
        return math.nan, math.nan, False, False

    if transverse == 0.0:
        return s_new, -math.inf, False, True
    abs_sin_b = abs(sin_b)
    abs_cos_b = abs(cos_b)
    if abs_sin_b == 0.0 or abs_cos_b == 0.0:
        return math.nan, math.nan, False, False
    log_d_new = (
        math.log(transverse)
        + math.log(abs_sin_b)
        + math.log(abs_cos_b)
        - math.log(abs(denominator))
    )
    if math.isnan(log_d_new) or log_d_new == math.inf:
        return math.nan, math.nan, False, False
    return s_new, log_d_new, False, True


@njit(cache=True)
def trace(
    beta: float,
    kappa: float,
    s0: float,
    logd0: float,
    steps: int,
    eta: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """単一軌道の状態、対数差分、線形分岐、妥当性を全時刻で返す。"""
    if steps < 0:
        raise ValueError("steps must be non-negative")

    s_values = np.full(steps + 1, np.nan, dtype=np.float64)
    logd_values = np.full(steps + 1, np.nan, dtype=np.float64)
    used_linear = np.zeros(steps + 1, dtype=np.bool_)
    valid = np.zeros(steps + 1, dtype=np.bool_)
    initial_valid = (
        math.isfinite(s0)
        and math.isfinite(beta)
        and beta > 0.0
        and math.isfinite(kappa)
        and math.isfinite(eta)
        and eta > 0.0
        and not math.isnan(logd0)
        and logd0 != math.inf
    )
    if not initial_valid:
        return s_values, logd_values, used_linear, valid

    s_values[0] = s0
    logd_values[0] = logd0
    valid[0] = True
    for t in range(1, steps + 1):
        s_new, logd_new, linear, step_valid = step(
            s_values[t - 1], logd_values[t - 1], beta, kappa, eta
        )
        used_linear[t] = linear
        valid[t] = step_valid
        if not step_valid:
            break
        s_values[t] = s_new
        logd_values[t] = logd_new
    return s_values, logd_values, used_linear, valid


@njit(parallel=True, cache=True)
def measure(
    beta: float,
    kappa: float,
    s0: np.ndarray,
    logd0: np.ndarray,
    steps: int,
    eta: float = 1e-6,
    entry_tol: float = 1e-8,
    exit_tol: float = 1e-5,
    enter_hold: int = 10,
    tail_hold: int = 500,
    checkpoints: np.ndarray = EMPTY_CHECKPOINTS,
) -> np.ndarray:
    """独立軌道ごとの有限窓指標を各 checkpoint で返す。

    最終軸は ``first_entry, last_outside, tail_dwell, releave, logd,
    invalid, linear_count, first_invalid_step`` の順。``tail_dwell`` は
    checkpoint まで連続して閾値内にいた全期間を保存する。無効化後の checkpoint は
    ``logd=nan``、``tail_dwell=-1`` とし、古い状態を有効値として残さない。
    """
    if s0.ndim != 1 or logd0.ndim != 1 or s0.size != logd0.size:
        raise ValueError("s0 and logd0 must be one-dimensional arrays of equal size")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if (
        not math.isfinite(beta)
        or beta <= 0.0
        or not math.isfinite(kappa)
        or not math.isfinite(eta)
        or eta <= 0.0
        or not math.isfinite(entry_tol)
        or entry_tol <= 0.0
        or not math.isfinite(exit_tol)
        or exit_tol <= 0.0
        or entry_tol >= exit_tol
        or enter_hold <= 0
        or tail_hold <= 0
    ):
        raise ValueError("invalid integration or finite-window configuration")
    for j in range(checkpoints.size):
        if checkpoints[j] <= 0 or checkpoints[j] > steps:
            raise ValueError("checkpoints must lie in [1, steps]")
        if j > 0 and checkpoints[j] <= checkpoints[j - 1]:
            raise ValueError("checkpoints must be strictly increasing")

    output = np.full((s0.size, checkpoints.size, 8), np.nan, dtype=np.float64)
    log_entry = math.log(entry_tol)
    log_exit = math.log(exit_tol)

    for i in prange(s0.size):
        s = s0[i]
        logd = logd0[i]
        first_entry = -1
        last_outside = 0 if logd >= log_exit else -1
        tail_dwell = 0
        entry_run = 0
        entered = False
        releaved = False
        invalid = (
            not math.isfinite(s)
            or math.isnan(logd)
            or logd == math.inf
        )
        first_invalid_step = 0 if invalid else -1
        linear_count = 0
        checkpoint_index = 0

        for t in range(1, steps + 1):
            if invalid:
                break
            s_new, logd_new, linear, step_valid = step(s, logd, beta, kappa, eta)
            if linear:
                linear_count += 1
            if not step_valid:
                invalid = True
                first_invalid_step = t
            else:
                s = s_new
                logd = logd_new
                if logd < log_entry:
                    entry_run += 1
                else:
                    entry_run = 0
                if first_entry < 0 and entry_run >= enter_hold:
                    first_entry = t - enter_hold + 1
                    entered = True

                if logd >= log_exit:
                    if entered:
                        releaved = True
                    last_outside = t
                    tail_dwell = 0
                else:
                    tail_dwell += 1

            if checkpoint_index < checkpoints.size and t == checkpoints[checkpoint_index]:
                output[i, checkpoint_index, 0] = first_entry
                output[i, checkpoint_index, 1] = last_outside
                output[i, checkpoint_index, 2] = -1 if invalid else tail_dwell
                output[i, checkpoint_index, 3] = 1.0 if releaved else 0.0
                output[i, checkpoint_index, 4] = math.nan if invalid else logd
                output[i, checkpoint_index, 5] = 1.0 if invalid else 0.0
                output[i, checkpoint_index, 6] = linear_count
                output[i, checkpoint_index, 7] = first_invalid_step
                checkpoint_index += 1

        while checkpoint_index < checkpoints.size:
            output[i, checkpoint_index, 0] = first_entry
            output[i, checkpoint_index, 1] = last_outside
            output[i, checkpoint_index, 2] = -1.0
            output[i, checkpoint_index, 3] = 1.0 if releaved else 0.0
            output[i, checkpoint_index, 4] = math.nan
            output[i, checkpoint_index, 5] = 1.0
            output[i, checkpoint_index, 6] = linear_count
            output[i, checkpoint_index, 7] = first_invalid_step
            checkpoint_index += 1

    return output
