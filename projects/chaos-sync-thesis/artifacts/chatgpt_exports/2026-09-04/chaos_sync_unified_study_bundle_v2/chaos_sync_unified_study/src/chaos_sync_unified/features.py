from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def robust_center_scale(values: FloatArray, axis: int = -1) -> tuple[FloatArray, FloatArray]:
    """How: median と half-IQR を用いて Cauchy 系でも有限な中心・尺度を求める。"""
    array = np.asarray(values, dtype=np.float64)
    center = np.nanmedian(array, axis=axis, keepdims=True)
    q25 = np.nanquantile(array, 0.25, axis=axis, keepdims=True)
    q75 = np.nanquantile(array, 0.75, axis=axis, keepdims=True)
    scale = 0.5 * (q75 - q25)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return center, scale


def robust_standardize(values: FloatArray, axis: int = -1) -> FloatArray:
    """How: train/test 各軌道内部の周辺尺度を除き、時間構造へ焦点を移す。"""
    center, scale = robust_center_scale(values, axis=axis)
    return (np.asarray(values, dtype=np.float64) - center) / scale


def cayley(values: FloatArray, scale: float = 1.0) -> ComplexArray:
    """How: 実数直線を単位円へ写す Cayley 変換を適用する。"""
    array = np.asarray(values, dtype=np.float64)
    gamma = float(scale)
    if gamma <= 0.0:
        raise ValueError("scale は正である必要がある")
    return (array - 1j * gamma) / (array + 1j * gamma)


def interpolate_missing(values: FloatArray) -> FloatArray:
    """How: Fourier 比較に必要な等間隔列を、観測済み点だけで線形補間する。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values は [sample,time] である必要がある")
    result = array.copy()
    x_full = np.arange(array.shape[1], dtype=np.float64)
    for row in range(array.shape[0]):
        observed = np.isfinite(array[row])
        if observed.sum() == 0:
            result[row] = 0.0
        elif observed.sum() == 1:
            result[row] = array[row, observed][0]
        else:
            result[row] = np.interp(x_full, x_full[observed], array[row, observed])
    return result


@dataclass(frozen=True)
class TMTemporalExtractor:
    """How: TM 位相の自己相関を同一容量の実数特徴へ展開する。"""

    orders: tuple[int, ...] = (1, 2, 3, 4)
    lags: tuple[int, ...] = (1, 2)

    @property
    def output_dim(self) -> int:
        return 2 * len(self.orders) * len(self.lags)

    def transform(self, trajectories: FloatArray) -> FloatArray:
        values = np.asarray(trajectories, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("trajectories は [sample,time] である必要がある")
        standardized = robust_standardize(values, axis=1)
        z = cayley(standardized)
        columns: list[FloatArray] = []
        for order in self.orders:
            mode = z**order
            for lag in self.lags:
                if lag >= values.shape[1]:
                    raise ValueError("lag は観測長より小さくする必要がある")
                product = np.conjugate(mode[:, :-lag]) * mode[:, lag:]
                valid = np.isfinite(values[:, :-lag]) & np.isfinite(values[:, lag:])
                product = np.where(valid, product, np.nan + 1j * np.nan)
                coefficient = np.nanmean(product, axis=1)
                coefficient = np.nan_to_num(coefficient, nan=0.0)
                columns.extend([coefficient.real, coefficient.imag])
        return np.column_stack(columns).astype(np.float64, copy=False)


@dataclass(frozen=True)
class FourierBandExtractor:
    """How: 補間済み軌道の対数帯域エネルギーを固定次元へ集約する。"""

    n_bands: int = 16

    @property
    def output_dim(self) -> int:
        return self.n_bands

    def transform(self, trajectories: FloatArray) -> FloatArray:
        values = np.asarray(trajectories, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("trajectories は [sample,time] である必要がある")
        filled = interpolate_missing(values)
        standardized = robust_standardize(filled, axis=1)
        spectrum = np.abs(np.fft.rfft(standardized, axis=1)) ** 2
        spectrum = spectrum[:, 1:]  # DC は軌道内標準化で意味を持たない。
        edges = np.linspace(0, spectrum.shape[1], self.n_bands + 1, dtype=int)
        bands = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            right = max(right, left + 1)
            bands.append(np.log1p(np.mean(spectrum[:, left:right], axis=1)))
        return np.column_stack(bands).astype(np.float64, copy=False)


def multichannel_features(
    trajectories: FloatArray,
    extractor: TMTemporalExtractor | FourierBandExtractor,
) -> FloatArray:
    """How: 各観測チャネルを同じ特徴写像で処理し、順序を保って連結する。"""
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("trajectories は [sample,channel,time] である必要がある")
    return np.concatenate([extractor.transform(values[:, channel, :]) for channel in range(values.shape[1])], axis=1)


def make_graph_basis(n_nodes: int) -> FloatArray:
    """How: 一様モード、二群コントラスト、残余直交モードの順に基底を構成する。"""
    if n_nodes < 2 or n_nodes % 2 != 0:
        raise ValueError("n_nodes は2以上の偶数である必要がある")
    uniform = np.ones(n_nodes, dtype=np.float64)
    uniform /= np.linalg.norm(uniform)
    contrast = np.concatenate([np.ones(n_nodes // 2), -np.ones(n_nodes // 2)]).astype(np.float64)
    contrast /= np.linalg.norm(contrast)
    candidates = np.eye(n_nodes, dtype=np.float64)
    matrix = np.column_stack([uniform, contrast, candidates])
    q, _ = np.linalg.qr(matrix)
    # QR の符号不定性を最初の2列だけ固定する。
    if np.dot(q[:, 0], uniform) < 0:
        q[:, 0] *= -1
    if np.dot(q[:, 1], contrast) < 0:
        q[:, 1] *= -1
    return q[:, :n_nodes]


def graph_tm_late_features(
    trajectories: FloatArray,
    graph_basis: FloatArray,
    *,
    late_window: int = 24,
    orders: Iterable[int] = (1, 2, 3, 4),
    n_modes: int = 2,
) -> FloatArray:
    """How: 後半軌道の TM モードをグラフ一様・コントラスト方向へ射影する。"""
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("trajectories は [sample,time,node] である必要がある")
    if graph_basis.shape[0] != values.shape[2]:
        raise ValueError("graph_basis と node 数が一致しない")
    late = values[:, -late_window:, :]
    # Why not: node ごとの独立標準化は同期誤差を消すため、全 late-window を sample 単位で標準化する。
    flattened = late.reshape(late.shape[0], -1)
    center, scale = robust_center_scale(flattened, axis=1)
    standardized = (late - center[:, None, :]) / scale[:, None, :]
    z = cayley(standardized)
    columns: list[FloatArray] = []
    basis = graph_basis[:, :n_modes]
    for order in orders:
        projected = np.einsum("stn,nm->stm", z**order, basis, optimize=True)
        mean_mode = np.mean(projected, axis=1)
        lag_mode = np.mean(np.conjugate(projected[:, :-1, :]) * projected[:, 1:, :], axis=1)
        for statistic in (mean_mode, lag_mode):
            columns.extend([statistic.real, statistic.imag])
    return np.concatenate(columns, axis=1).astype(np.float64, copy=False)


def graph_raw_late_features(
    trajectories: FloatArray,
    graph_basis: FloatArray,
    *,
    late_window: int = 24,
    n_modes: int = 2,
) -> FloatArray:
    """How: 生状態を同じグラフモードへ射影し、32次元のロバスト時間統計へする。"""
    values = np.asarray(trajectories, dtype=np.float64)
    late = values[:, -late_window:, :]
    projected = np.einsum("stn,nm->stm", late, graph_basis[:, :n_modes], optimize=True)
    standardized = np.empty_like(projected)
    for mode in range(n_modes):
        standardized[:, :, mode] = robust_standardize(projected[:, :, mode], axis=1)
    stats: list[FloatArray] = []
    for mode in range(n_modes):
        series = standardized[:, :, mode]
        q10 = np.quantile(series, 0.10, axis=1)
        q25 = np.quantile(series, 0.25, axis=1)
        q50 = np.quantile(series, 0.50, axis=1)
        q75 = np.quantile(series, 0.75, axis=1)
        q90 = np.quantile(series, 0.90, axis=1)
        moments = [
            np.mean(series, axis=1),
            np.std(series, axis=1),
            q10,
            q25,
            q50,
            q75,
            q90,
            series[:, -1],
        ]
        autocorrs = []
        for lag in (1, 2, 4, 8):
            left = series[:, :-lag]
            right = series[:, lag:]
            autocorrs.append(np.mean(left * right, axis=1))
        diff_stats = [
            np.mean(np.abs(np.diff(series, axis=1)), axis=1),
            np.std(np.diff(series, axis=1), axis=1),
            np.max(np.abs(series), axis=1),
            np.mean(series**3, axis=1),
        ]
        stats.extend(moments + autocorrs + diff_stats)
    return np.column_stack(stats).astype(np.float64, copy=False)


def tm_temporal_spectrum(
    trajectories: FloatArray,
    *,
    orders: tuple[int, ...] = (1, 2, 3, 4),
    n_coefficients: int = 8,
) -> FloatArray:
    """How: node 平均 TM 位相の低周波 DCT 係数を圧縮候補として返す。"""
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("trajectories は [sample,time,node] である必要がある")
    center, scale = robust_center_scale(values.reshape(values.shape[0], -1), axis=1)
    standardized = (values - center[:, None, :]) / scale[:, None, :]
    z_mean = np.mean(cayley(standardized), axis=2)
    blocks = []
    for order in orders:
        mode = z_mean**order
        real_coeff = dct(mode.real, axis=1, norm="ortho")[:, :n_coefficients]
        imag_coeff = dct(mode.imag, axis=1, norm="ortho")[:, :n_coefficients]
        blocks.extend([real_coeff, imag_coeff])
    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)


def raw_temporal_spectrum(trajectories: FloatArray, *, n_coefficients: int = 64) -> FloatArray:
    """How: node平均生状態をロバスト標準化し、DCT係数へ変換する。"""
    values = np.asarray(trajectories, dtype=np.float64)
    mean_state = np.mean(values, axis=2)
    standardized = robust_standardize(mean_state, axis=1)
    return dct(standardized, axis=1, norm="ortho")[:, :n_coefficients].astype(np.float64, copy=False)


def synchronization_rms(trajectories: FloatArray, *, window: int = 16) -> FloatArray:
    """How: 終端窓における一様モードからの RMS 偏差を sample ごとに測る。"""
    values = np.asarray(trajectories, dtype=np.float64)[:, -window:, :]
    centered = values - np.mean(values, axis=2, keepdims=True)
    return np.sqrt(np.mean(centered * centered, axis=(1, 2)))


def effective_rank(features: FloatArray) -> float:
    """How: 特徴共分散の特異値から entropy effective rank を計算する。"""
    values = np.asarray(features, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular * singular
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))
