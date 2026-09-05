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
    """train標準化付きridgeの係数を返す。targetsは(n,)または(n,出力数)。"""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("ridge penaltyは有限正数が必要です。")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    # 定数列は中心化後にゼロとなる。除算だけを正規化し情報を追加しない。
    scale = np.where(scale > 1e-12, scale, 1.)
    x = (features-mean)/scale
    intercept = np.array(targets.mean(axis=0))
    # 正規方程式は辞書の条件数を二乗してしまうため、拡大最小二乗を直接解く。
    augmented = np.vstack((x, np.sqrt(penalty)*np.eye(x.shape[1])))
    padded = np.concatenate((targets-intercept, np.zeros((x.shape[1],)+targets.shape[1:])))
    coefficients = np.linalg.lstsq(augmented, padded, rcond=None)[0]
    return dict(mean=mean, scale=scale, intercept=intercept, coefficients=coefficients)


def predict_ridge(model: Dict[str, FloatArray], features: FloatArray) -> FloatArray:
    """凍結済みのtrain統計・係数で予測する。評価データではfitしない。"""
    return model["intercept"]+(features-model["mean"])/model["scale"]@model["coefficients"]


def state_dictionary(state: FloatArray, name: str) -> FloatArray:
    """標準Cauchy状態へ固定16実数の観測辞書を適用する。時間FFTではない。"""
    x = np.asarray(state, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("状態は有限な1次元配列が必要です。")
    u = .5+np.arctan(x)/np.pi
    if name in ("tm", "phase_fourier", "euclidean_fourier"):
        k = np.arange(1,9)
        if name == "tm":
            modes = cayley_modes(x,tuple(range(1,9)))
        else:
            angle = -2*np.arctan2(1.,x) if name == "phase_fourier" else x
            modes = np.exp(1j*angle[:,None]*k)
        return np.sqrt(2)*np.stack((modes.real,modes.imag),axis=-1).reshape(len(x),16)
    if name == "cdf_cosine":
        return np.sqrt(2)*np.cos(np.pi*u[:,None]*np.arange(1,17))
    if name == "cdf_legendre":
        return np.polynomial.legendre.legvander(2*u-1,16)[:,1:]*np.sqrt(2*np.arange(1,17)+1)
    if name == "cdf_rbf":
        centers=(2*np.arange(16)+1)/16-1
        return np.exp(-.5*((2*u[:,None]-1-centers)/.25)**2)
    raise ValueError("未知の状態辞書です: "+name)
