"""有限軌道のrobust標準化、固定時間特徴、train-only ridge読み出し。"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from .core import cayley_modes

FloatArray = NDArray[np.float64]


def robust_standardize(orbit: FloatArray) -> Tuple[FloatArray, float, float]:
    """観測窓自身のmedian・half-IQRを除く。母平均・母分散は使用しない。"""
    if orbit.ndim != 1 or not np.all(np.isfinite(orbit)):
        raise ValueError("軌道は有限の1次元配列が必要です。")
    q1, median, q3 = np.quantile(orbit, [.25, .5, .75])
    scale = float((q3-q1)/2)
    if scale <= np.finfo(float).eps:
        raise ValueError("正のrobust尺度が必要です。")
    return (orbit-median)/scale, float(median), scale


def fourier_features(values: FloatArray, bands: int) -> FloatArray:
    """Hann窓の有限標本powerを帯域平均し対数化する。母PSDとは解釈しない。"""
    window = np.hanning(len(values))
    spectrum = np.fft.rfft((values-values.mean())*window)[1:]
    power = np.abs(spectrum)**2 / np.sum(window**2)
    if len(power) < bands:
        raise ValueError("FFT帯域数以上の周波数binが必要です。")
    return np.array([np.log(1e-12+band.mean()) for band in np.array_split(power, bands)])


def fixed_readout_features(orbit: FloatArray, permutation: NDArray[np.int64]) -> Tuple[Dict[str, FloatArray], float, float]:
    """同じ標準化窓とshuffleから8種の固定16次元特徴を返す。"""
    z, location, scale = robust_standardize(orbit)
    if not np.array_equal(np.sort(permutation), np.arange(len(z))):
        raise ValueError("permutationは時刻の全単射にしてください。")
    q = cayley_modes(z, (1,))[:, 0]
    features = {}
    for suffix, values, phase in (("", z, q), ("_shuffle", z[permutation], q[permutation])):
        correlations = np.array([np.mean(phase[lag:]**k * np.conj(phase[:-lag]**k))
                                 for k in range(1, 5) for lag in (1, 2)])
        features["tm"+suffix] = np.column_stack((correlations.real, correlations.imag)).ravel()
        features["fourier_raw"+suffix] = fourier_features(values, 16)
        features["fourier_cayley"+suffix] = np.concatenate([fourier_features(part, 8) for part in (phase.real, phase.imag)])
    moments = np.array([np.mean(q**k) for k in range(1, 9)])
    features["tm_instant"] = np.column_stack((moments.real, moments.imag)).ravel()
    features["quantiles"] = np.quantile(z, np.linspace(.05, .95, 16))
    return features, location, scale


def fit_ridge(features: FloatArray, targets: FloatArray, penalty: float) -> Dict[str, FloatArray]:
    """旧E1Aのtrain標準化付きridgeを、係数も保存できる形で再利用する。"""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("ridge penaltyは有限正数が必要です。")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    # 定数列は中心化後にゼロとなる。除算だけを正規化し情報を追加しない。
    scale = np.where(scale > 1e-12, scale, 1.)
    x = (features-mean)/scale
    intercept = np.array(targets.mean())
    coefficients = np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]), x.T@(targets-intercept))
    return dict(mean=mean, scale=scale, intercept=intercept, coefficients=coefficients)


def predict_ridge(model: Dict[str, FloatArray], features: FloatArray) -> FloatArray:
    """凍結済みのtrain統計・係数で予測する。評価データではfitしない。"""
    return model["intercept"]+(features-model["mean"])/model["scale"]@model["coefficients"]
