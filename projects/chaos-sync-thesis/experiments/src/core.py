"""全実験で共有する一般化Boole写像と決定論的軌道生成。"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def generalized_boole(
    state: FloatArray, alpha: FloatArray, near_zero: float = 1e-8
) -> Tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
    """F=alpha*(x-1/x)、非有限出力マスク、特異点近傍入力マスクを返す。

    stateのfloat32/float64精度を保つ。ゼロ除算やoverflowはマスクへ明示し、
    clipや代替値による軌道継続を行わない。
    """
    x = np.asarray(state)
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("stateはfloat32またはfloat64にしてください。")
    a = np.asarray(alpha, dtype=x.dtype)
    if not np.all(np.isfinite(a) & (a > 0) & (a < 1)):
        raise ValueError("alphaは有限かつ0<alpha<1が必要です。")
    if not np.isfinite(near_zero) or near_zero <= 0:
        raise ValueError("near_zeroは有限の正数にしてください。")
    # IEEE非有限値を返す診断APIであり、失敗の成功扱いや値の置換は行わない。
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        updated = a * (x - x.dtype.type(1) / x)
    return updated, ~np.isfinite(updated), np.abs(x) < near_zero


def boole_log_derivative(state: FloatArray, alpha: FloatArray) -> NDArray[np.float64]:
    """有限・非零の観測点のlog|F'|をfloat64で集計する。

    中間値x**(-2)のoverflowを避け、同値式をlogaddexpで評価する。
    これは観測量の評価方法であり、軌道更新式を変更しない。
    """
    x, a = np.asarray(state, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    if np.any(~np.isfinite(x) | (x == 0)):
        raise ValueError("導関数の評価点は有限かつ非零にしてください。")
    if np.any(~np.isfinite(a) | (a <= 0) | (a >= 1)):
        raise ValueError("alphaは0<alpha<1にしてください。")
    return np.log(a) + np.logaddexp(0, -2 * np.log(np.abs(x)))


def boole_theory(alpha: FloatArray) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """非結合系の不変Cauchy尺度と自然対数Lyapunov指数を返す。"""
    a = np.asarray(alpha, dtype=np.float64)
    if np.any(~np.isfinite(a) | (a <= 0) | (a >= 1)):
        raise ValueError("alphaは0<alpha<1にしてください。")
    return np.sqrt(a / (1 - a)), 2 * np.log(np.sqrt(a) + np.sqrt(1 - a))


def simulate_boole_orbits(
    initial_states: FloatArray, alpha: FloatArray, burn_in: int,
    observation_length: int, thresholds: Tuple[float, ...] = (1e-12, 1e-10, 1e-8),
) -> Tuple[FloatArray, Dict[str, NDArray]]:
    """独立ノードを一括反復し、観測軌道と失敗・近傍通過診断を返す。

    観測はx_(burn_in+1)から。失敗ノードは更新を停止し、未生成部分はNaNとする。
    failure_step=-1だけが完走を意味する。近傍回数はburn-inを含む更新入力で数える。
    """
    if burn_in < 0 or observation_length < 1:
        raise ValueError("burn_in>=0、observation_length>=1が必要です。")
    x = np.array(initial_states, copy=True)
    if x.ndim != 1 or not np.all(np.isfinite(x) & (x != 0)):
        raise ValueError("初期状態は有限・非零の1次元配列にしてください。")
    a = np.broadcast_to(np.asarray(alpha, dtype=x.dtype), x.shape)
    generalized_boole(x, a)
    limits = np.asarray(thresholds, dtype=np.float64)
    if limits.ndim != 1 or not np.all(np.isfinite(limits) & (limits > 0)):
        raise ValueError("thresholdsは有限の正数列にしてください。")
    orbit = np.full((len(x), observation_length), np.nan, dtype=x.dtype)
    failure = np.full(len(x), -1, dtype=np.int64)
    counts = np.zeros((len(x), len(limits)), dtype=np.int64)
    nonfinite_counts = np.zeros(len(x), dtype=np.int64)
    active = np.ones(len(x), dtype=bool)
    for step in range(burn_in + observation_length):
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            break
        counts[indices] += np.abs(x[indices, None]).astype(np.float64) < limits
        updated, nonfinite, _ = generalized_boole(x[indices], a[indices])
        bad = nonfinite | (updated == 0)
        nonfinite_counts[indices] += nonfinite
        failure[indices[bad]] = step + 1
        active[indices[bad]] = False
        x[indices] = updated
        if step >= burn_in:
            orbit[indices, step - burn_in] = updated
    return orbit, {"failure_step": failure, "near_zero_counts": counts,
                   "nonfinite_update_count": nonfinite_counts}



def cayley_modes(
    state: FloatArray, orders: Tuple[int, ...], mu: float = 0.0, gamma: float = 1.0,
) -> NDArray[np.complex128]:
    """Cauchy測度に対応するCayley power modesを末尾の次数軸に並べる。

    stateは有限実数、gammaは正。標準Cauchy以外は中心mu・尺度gammaを指定する。
    一般の異なる極を持つTM系や、Koopman固有関数を自動構築する関数ではない。
    """
    raw = np.asarray(state)
    if np.iscomplexobj(raw):
        raise TypeError("stateは実数にしてください。")
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(mu) or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("muは有限、gammaは有限の正数にしてください。")
    if not orders or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) for k in orders):
        raise ValueError("ordersは空でない整数列にしてください。")
    if not np.all(np.isfinite(x)):
        raise ValueError("stateは有限にしてください。")
    # 位相の指数関数で実装するとK-foldの検証と同じ計算経路になるため、有理式を使う。
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        shifted = x - mu
        q = (shifted - 1j * gamma) / (shifted + 1j * gamma)
        return q[..., None] ** np.asarray(orders)


def kfold_cotangent(
    state: FloatArray, K: int, pole_sine_threshold: float = 1e-10,
) -> Tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """枝をatan2(1,x)で固定したH_K、非有限出力、極近傍マスクを返す。

    極近傍はclipせず生の値と診断を返す。厳密なx=0では偶数Kは未定義、
    奇数Kは0という解析値を適用する。その他の極近傍はマスクで明示する。
    """
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("Kは2以上の整数にしてください。")
    raw = np.asarray(state)
    if np.iscomplexobj(raw):
        raise TypeError("stateは実数にしてください。")
    x = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("stateは有限にしてください。")
    if not np.isfinite(pole_sine_threshold) or pole_sine_threshold <= 0:
        raise ValueError("pole_sine_thresholdは有限の正数にしてください。")
    angle = K * np.arctan2(1.0, x)
    denominator = np.sin(angle)
    near = np.abs(denominator) < pole_sine_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        mapped = np.cos(angle) / denominator
    # sin(n*pi)の丸め残差から巨大な有限値を返すと、既知の特異点を見逃す。
    mapped = np.where(x == 0, np.nan if K % 2 == 0 else 0.0, mapped)
    near = near | ((x == 0) & (K % 2 == 0))
    return mapped, ~np.isfinite(mapped), near
