"""有限サイズランダム結合Boole系の同期転移をTM位相で検証する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]

COUPLING_DISTRIBUTIONS: Tuple[str, ...] = (
    "uniform_positive",
    "positive_binary",
    "biased_sign",
    "rademacher",
)
COUPLING_SEED_OFFSETS: Dict[str, int] = {
    "uniform_positive": 11,
    "positive_binary": 23,
    "biased_sign": 37,
    "rademacher": 53,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """E3A pilotの数値条件と事前判定基準を保持する。"""

    alpha: float
    node_counts: Tuple[int, ...]
    matrix_seeds: Tuple[int, ...]
    coupling_distributions: Tuple[str, ...]
    k_values: Tuple[float, ...]
    burn_in: int
    evaluation_steps: int
    local_growth_steps: int
    local_perturbation: float
    near_zero_threshold: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    success_mean_order: float
    success_q05_order: float
    phase_heatmap_k_values: Tuple[float, ...]
    save_figures: bool = True
    theory_kc_tolerance: float = 0.10
    assessment_scope: str = "pilot"

    def validate(self) -> None:
        """写像、走査、bootstrapが意味を持つ範囲へ設定を制限する。"""

        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha は 0 < alpha < 1 を満たす必要があります。")
        if not self.node_counts or any(node_count < 4 for node_count in self.node_counts):
            raise ValueError("node_counts は4以上の正整数を含む必要があります。")
        if len(set(self.node_counts)) != len(self.node_counts):
            raise ValueError("node_counts に重複を含められません。")
        if not self.matrix_seeds or len(set(self.matrix_seeds)) != len(self.matrix_seeds):
            raise ValueError("matrix_seeds は重複のない1個以上のseedにしてください。")
        unknown = set(self.coupling_distributions) - set(COUPLING_DISTRIBUTIONS)
        if unknown:
            raise ValueError(f"未対応の結合分布です: {sorted(unknown)}")
        if not self.k_values or any(k_value < 0.0 for k_value in self.k_values):
            raise ValueError("k_values は0以上の値を含む必要があります。")
        if tuple(sorted(self.k_values)) != self.k_values or len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values は重複のない昇順にしてください。")
        if max(self.k_values) >= 1.0 - self.alpha:
            raise ValueError("pilotでは nominal m=1 のCauchy固定点が存在する範囲だけを走査します。")
        if self.burn_in <= 0 or self.evaluation_steps <= 0:
            raise ValueError("burn_in と evaluation_steps は正にしてください。")
        if not 1 <= self.local_growth_steps <= self.burn_in:
            raise ValueError("local_growth_steps は1以上burn_in以下にしてください。")
        if self.local_perturbation <= 0.0:
            raise ValueError("local_perturbation は正にしてください。")
        if self.near_zero_threshold <= 0.0:
            raise ValueError("near_zero_threshold は正にしてください。")
        if self.bootstrap_repetitions <= 0:
            raise ValueError("bootstrap_repetitions は正にしてください。")
        if not 0.0 < self.success_q05_order <= self.success_mean_order <= 1.0:
            raise ValueError("同期成功閾値は 0 < q05 <= mean <= 1 を満たす必要があります。")
        if self.theory_kc_tolerance <= 0.0:
            raise ValueError("theory_kc_tolerance は正にしてください。")
        if self.assessment_scope not in {"pilot", "confirmation"}:
            raise ValueError("assessment_scope は pilot または confirmation にしてください。")


@dataclass(frozen=True)
class CouplingStatistics:
    """一つの固定結合行列から得る監査統計を保持する。"""

    distribution: str
    node_count: int
    matrix_seed: int
    target_absolute_mean: float
    target_signed_mean: float
    sample_absolute_mean: float
    sample_signed_mean: float
    row_mean_std: float
    row_absolute_mean_std: float
    symmetric: bool
    diagonal_zero: bool


@dataclass(frozen=True)
class SimulationMetrics:
    """一つの行列・初期化・結合強度に対する時系列集約値を保持する。"""

    distribution: str
    node_count: int
    matrix_seed: int
    initialization: str
    coupling_strength: float
    sample_absolute_mean: float
    sample_signed_mean: float
    row_mean_std: float
    mean_tm_order: float
    q05_tm_order: float
    mean_phase_mad: float
    median_sync_residual: float
    median_adaptive_scale: float
    max_tm_modulus_error: float
    local_initial_growth_rate: float
    near_zero_rate: float
    nonfinite_event_rate: float
    finite_completed: bool
    first_nonfinite_step: int
    synchronized: bool


def theoretical_cauchy_scale(alpha: float, coupling_strength: float, moment: float) -> float:
    """論文の絶対一次モーメント置換に基づくCauchy尺度を返す。"""

    denominator = 1.0 - alpha - coupling_strength * moment
    if denominator <= 0.0:
        raise ValueError("正のCauchy固定点が存在しないパラメータです。")
    return math.sqrt(alpha / denominator)


def conditional_lyapunov(alpha: float, coupling_strength: float, moment: float) -> float:
    """論文式または明示したmoment置換で条件付きLyapunov指数を計算する。"""

    radicand = 1.0 - alpha - coupling_strength * moment
    if radicand < 0.0:
        return math.nan
    return 2.0 * math.log(math.sqrt(alpha) + math.sqrt(radicand))


def critical_coupling(alpha: float, moment: float) -> float:
    """正のmomentに対する解析的臨界結合を返す。"""

    if moment <= 0.0:
        raise ValueError("critical_coupling の moment は正にしてください。")
    return 2.0 * math.sqrt(alpha) * (1.0 - math.sqrt(alpha)) / moment


def tm_transform(states: FloatArray, scale: float) -> ComplexArray:
    """実数状態をCauchy尺度に対応する単位円上のTM位相へ写す。"""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("TM尺度は有限かつ正にしてください。")
    return np.asarray((states - 1j * scale) / (states + 1j * scale), dtype=np.complex128)


def tm_phase_metrics(states: FloatArray) -> Tuple[float, float, float, float, FloatArray]:
    """適応Cauchy尺度でTM秩序、位相MAD、単位円誤差を求める。"""

    if states.ndim != 1 or states.size == 0 or not np.all(np.isfinite(states)):
        raise ValueError("TM指標には有限な1次元状態列が必要です。")
    scale = float(np.median(np.abs(states)))
    if scale <= 0.0:
        raise FloatingPointError("適応TM尺度が0となりました。")
    tm_values = tm_transform(states, scale)
    mean_value = complex(np.mean(tm_values))
    order = abs(mean_value)
    phases = np.angle(tm_values)
    # Why not: 非同期相では円周中央値が一意でないため、平均方向でunwrapしてから標本中央値を取る。
    anchor = math.atan2(mean_value.imag, mean_value.real) if order > 1.0e-15 else float(phases[0])
    unwrapped = np.angle(np.exp(1j * (phases - anchor)))
    circular_median = anchor + float(np.median(unwrapped))
    distances = np.abs(np.angle(np.exp(1j * (phases - circular_median))))
    phase_mad = float(np.median(distances))
    modulus_error = float(np.max(np.abs(np.abs(tm_values) - 1.0)))
    return order, phase_mad, scale, modulus_error, phases


def normalized_sync_residual(states: FloatArray) -> float:
    """Cauchy外れ値に頑健なノード間同期残差を返す。"""

    center = float(np.median(states))
    numerator = float(np.median(np.abs(states - center)))
    denominator = 1.0 + float(np.median(np.abs(states)))
    return numerator / denominator


def coupling_target_moments(distribution: str) -> Tuple[float, float]:
    """実験で定義した結合分布の母絶対平均と符号付き平均を返す。"""

    if distribution == "uniform_positive":
        return 1.0, 1.0
    if distribution == "positive_binary":
        return 1.0, 1.0
    if distribution == "biased_sign":
        return 1.0, 0.75
    if distribution == "rademacher":
        return 1.0, 0.0
    raise ValueError(f"未対応の結合分布です: {distribution}")


def generate_symmetric_coupling(
    node_count: int,
    distribution: str,
    matrix_seed: int,
) -> FloatArray:
    """上三角を独立生成し、対称・対角0の固定結合行列を返す。"""

    if node_count < 2:
        raise ValueError("node_count は2以上にしてください。")
    if distribution not in COUPLING_SEED_OFFSETS:
        raise ValueError(f"未対応の結合分布です: {distribution}")
    if distribution == "uniform_positive":
        matrix = np.ones((node_count, node_count), dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    seed_sequence = np.random.SeedSequence(
        [matrix_seed, node_count, COUPLING_SEED_OFFSETS[distribution]]
    )
    rng = np.random.default_rng(seed_sequence)
    upper_indices = np.triu_indices(node_count, k=1)
    edge_count = upper_indices[0].size
    if distribution == "positive_binary":
        upper_values = 2.0 * rng.integers(0, 2, size=edge_count)
    elif distribution == "biased_sign":
        upper_values = np.where(rng.random(edge_count) < 0.875, 1.0, -1.0)
    else:
        upper_values = np.where(rng.random(edge_count) < 0.5, 1.0, -1.0)

    matrix = np.zeros((node_count, node_count), dtype=np.float64)
    matrix[upper_indices] = upper_values
    matrix[(upper_indices[1], upper_indices[0])] = upper_values
    return matrix


def calculate_coupling_statistics(
    matrix: FloatArray,
    distribution: str,
    matrix_seed: int,
) -> CouplingStatistics:
    """行列の標本momentと有限サイズ行平均揺らぎを計算する。"""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("結合行列は正方行列にしてください。")
    node_count = matrix.shape[0]
    upper_values = matrix[np.triu_indices(node_count, k=1)]
    row_means = np.sum(matrix, axis=1) / (node_count - 1)
    row_absolute_means = np.sum(np.abs(matrix), axis=1) / (node_count - 1)
    target_absolute_mean, target_signed_mean = coupling_target_moments(distribution)
    return CouplingStatistics(
        distribution=distribution,
        node_count=node_count,
        matrix_seed=matrix_seed,
        target_absolute_mean=target_absolute_mean,
        target_signed_mean=target_signed_mean,
        sample_absolute_mean=float(np.mean(np.abs(upper_values))),
        sample_signed_mean=float(np.mean(upper_values)),
        row_mean_std=float(np.std(row_means)),
        row_absolute_mean_std=float(np.std(row_absolute_means)),
        symmetric=bool(np.array_equal(matrix, matrix.T)),
        diagonal_zero=bool(np.all(np.diag(matrix) == 0.0)),
    )


def coupled_step(
    states: FloatArray,
    coupling_matrix: FloatArray,
    alpha: float,
    coupling_strength: float,
) -> FloatArray:
    """モデル式を一ステップだけ評価する。"""

    if states.ndim != 1 or coupling_matrix.shape != (states.size, states.size):
        raise ValueError("状態と結合行列の次元が一致しません。")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        local = alpha * (states - 1.0 / states)
        field = coupling_matrix @ states / (states.size - 1)
        return np.asarray(local + coupling_strength * field, dtype=np.float64)


def coupled_batch_step(
    states: FloatArray,
    coupling_matrix: FloatArray,
    alpha: float,
    coupling_strengths: FloatArray,
    use_uniform_complete_graph: bool = False,
) -> FloatArray:
    """同じ固定行列上の複数K・初期化を共通乱数で一括更新する。"""

    if states.ndim != 2 or coupling_matrix.shape != (states.shape[1], states.shape[1]):
        raise ValueError("batch状態と結合行列の次元が一致しません。")
    if coupling_strengths.shape != (states.shape[0],):
        raise ValueError("coupling_strengths はbatch行数と一致させてください。")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        local = alpha * (states - 1.0 / states)
        if use_uniform_complete_graph:
            # Why not: 一様完全グラフでは密行列積は同じ和をN回再計算するため使用しない。
            fields = (
                np.sum(states, axis=1, keepdims=True) - states
            ) / (states.shape[1] - 1)
        else:
            fields = states @ coupling_matrix.T / (states.shape[1] - 1)
        return np.asarray(local + coupling_strengths[:, None] * fields, dtype=np.float64)

def sample_finite_cauchy(
    rng: np.random.Generator,
    size: int,
    scale: float,
) -> FloatArray:
    """有限かつ厳密な0でないCauchy初期値を生成する。"""

    values = np.asarray(scale * rng.standard_cauchy(size), dtype=np.float64)
    invalid = ~np.isfinite(values) | (values == 0.0)
    attempts = 0
    while np.any(invalid) and attempts < 100:
        values[invalid] = scale * rng.standard_cauchy(int(np.sum(invalid)))
        invalid = ~np.isfinite(values) | (values == 0.0)
        attempts += 1
    if np.any(invalid):
        raise RuntimeError("有限かつゼロでないCauchy初期値を生成できませんでした。")
    return values


def prepare_initial_states(
    node_count: int,
    matrix_seed: int,
    alpha: float,
    k_values: Sequence[float],
    local_perturbation: float,
) -> Tuple[FloatArray, FloatArray, Tuple[str, ...]]:
    """全Kで同じglobal/local初期状態を共有するbatchを作る。"""

    rng = np.random.default_rng(np.random.SeedSequence([matrix_seed, node_count, 101]))
    uncoupled_scale = math.sqrt(alpha / (1.0 - alpha))
    global_state = sample_finite_cauchy(rng, node_count, uncoupled_scale)
    common_state = float(sample_finite_cauchy(rng, 1, uncoupled_scale)[0])
    perturbation = rng.normal(size=node_count)
    perturbation -= np.mean(perturbation)
    perturbation_std = float(np.std(perturbation))
    if perturbation_std == 0.0:
        raise RuntimeError("local初期化の摂動分散が0となりました。")
    perturbation /= perturbation_std
    local_state = common_state + local_perturbation * (1.0 + abs(common_state)) * perturbation

    k_array = np.asarray(tuple(k_values), dtype=np.float64)
    global_batch = np.repeat(global_state[None, :], k_array.size, axis=0)
    local_batch = np.repeat(local_state[None, :], k_array.size, axis=0)
    states = np.vstack([global_batch, local_batch])
    coupling_strengths = np.concatenate([k_array, k_array])
    initializations = tuple(["global"] * k_array.size + ["local"] * k_array.size)
    return states, coupling_strengths, initializations


def estimate_log_growth(residuals: FloatArray) -> float:
    """初期同期残差の対数傾きからlocal成長率を推定する。"""

    indices = np.arange(residuals.size, dtype=np.float64)
    valid = np.isfinite(residuals) & (residuals > 0.0)
    if int(np.sum(valid)) < 3:
        return math.nan
    log_residuals = np.log(np.maximum(residuals[valid], np.finfo(np.float64).tiny))
    slope, _ = np.polyfit(indices[valid], log_residuals, 1)
    return float(slope)


def simulate_coupling_case(
    coupling_matrix: FloatArray,
    statistics: CouplingStatistics,
    config: ExperimentConfig,
    capture_phases: bool = False,
) -> Tuple[List[SimulationMetrics], Dict[str, FloatArray]]:
    """一つの固定行列で全K・二初期化を実行し、時系列を集約する。"""

    node_count = coupling_matrix.shape[0]
    use_uniform_complete_graph = statistics.distribution == "uniform_positive"
    if use_uniform_complete_graph:
        upper_values = coupling_matrix[np.triu_indices(node_count, k=1)]
        if (
            not statistics.symmetric
            or not statistics.diagonal_zero
            or not np.all(upper_values == 1.0)
        ):
            raise ValueError("一様完全グラフ高速経路にはoff-diagonal=1が必要です。")
    states, coupling_strengths, initializations = prepare_initial_states(
        node_count,
        statistics.matrix_seed,
        config.alpha,
        config.k_values,
        config.local_perturbation,
    )
    row_count = states.shape[0]
    k_count = len(config.k_values)
    active = np.ones(row_count, dtype=np.bool_)
    first_nonfinite_step = np.full(row_count, -1, dtype=np.int64)
    near_zero_counts = np.zeros(row_count, dtype=np.int64)
    attempted_counts = np.zeros(row_count, dtype=np.int64)
    nonfinite_counts = np.zeros(row_count, dtype=np.int64)
    local_residual_history = np.full(
        (config.local_growth_steps + 1, k_count),
        np.nan,
        dtype=np.float64,
    )
    for k_index in range(k_count):
        local_residual_history[0, k_index] = normalized_sync_residual(states[k_count + k_index])

    order_history = np.full((config.evaluation_steps, row_count), np.nan, dtype=np.float64)
    phase_mad_history = np.full_like(order_history, np.nan)
    residual_history = np.full_like(order_history, np.nan)
    scale_history = np.full_like(order_history, np.nan)
    modulus_error_history = np.full_like(order_history, np.nan)

    capture_rows: Dict[int, str] = {}
    phase_buffers: Dict[str, List[FloatArray]] = {}
    if capture_phases:
        for target_k in config.phase_heatmap_k_values:
            nearest_index = int(np.argmin(np.abs(np.asarray(config.k_values) - target_k)))
            key = f"{statistics.distribution}__global__K_{config.k_values[nearest_index]:.3f}"
            capture_rows[nearest_index] = key
            phase_buffers[key] = []

    total_steps = config.burn_in + config.evaluation_steps
    for step_index in range(1, total_steps + 1):
        active_before = active.copy()
        if np.any(active_before):
            attempted_counts[active_before] += node_count
            near_zero_counts[active_before] += np.sum(
                np.abs(states[active_before]) < config.near_zero_threshold,
                axis=1,
            )
        candidate = coupled_batch_step(
            states,
            coupling_matrix,
            config.alpha,
            coupling_strengths,
            use_uniform_complete_graph=use_uniform_complete_graph,
        )
        finite_rows = np.all(np.isfinite(candidate), axis=1)
        newly_failed = active_before & ~finite_rows
        if np.any(newly_failed):
            nonfinite_counts[newly_failed] += np.sum(~np.isfinite(candidate[newly_failed]), axis=1)
            first_nonfinite_step[newly_failed] = step_index
            active[newly_failed] = False
        candidate[~active] = np.nan
        states = candidate

        if step_index <= config.local_growth_steps:
            for k_index in range(k_count):
                row_index = k_count + k_index
                if active[row_index]:
                    local_residual_history[step_index, k_index] = normalized_sync_residual(states[row_index])

        if step_index <= config.burn_in:
            continue
        evaluation_index = step_index - config.burn_in - 1
        for row_index in np.flatnonzero(active):
            order, phase_mad, scale, modulus_error, phases = tm_phase_metrics(states[row_index])
            order_history[evaluation_index, row_index] = order
            phase_mad_history[evaluation_index, row_index] = phase_mad
            residual_history[evaluation_index, row_index] = normalized_sync_residual(states[row_index])
            scale_history[evaluation_index, row_index] = scale
            modulus_error_history[evaluation_index, row_index] = modulus_error
            if row_index in capture_rows:
                phase_buffers[capture_rows[row_index]].append(phases.copy())

    metrics: List[SimulationMetrics] = []
    for row_index, initialization in enumerate(initializations):
        k_index = row_index % k_count
        finite_completed = bool(active[row_index])
        if finite_completed:
            order_values = order_history[:, row_index]
            mean_order = float(np.mean(order_values))
            q05_order = float(np.quantile(order_values, 0.05))
            mean_phase_mad = float(np.mean(phase_mad_history[:, row_index]))
            median_residual = float(np.median(residual_history[:, row_index]))
            median_scale = float(np.median(scale_history[:, row_index]))
            max_modulus_error = float(np.max(modulus_error_history[:, row_index]))
        else:
            mean_order = math.nan
            q05_order = math.nan
            mean_phase_mad = math.nan
            median_residual = math.nan
            median_scale = math.nan
            max_modulus_error = math.nan
        local_growth_rate = (
            estimate_log_growth(local_residual_history[:, k_index])
            if initialization == "local"
            else math.nan
        )
        metrics.append(
            SimulationMetrics(
                distribution=statistics.distribution,
                node_count=node_count,
                matrix_seed=statistics.matrix_seed,
                initialization=initialization,
                coupling_strength=float(coupling_strengths[row_index]),
                sample_absolute_mean=statistics.sample_absolute_mean,
                sample_signed_mean=statistics.sample_signed_mean,
                row_mean_std=statistics.row_mean_std,
                mean_tm_order=mean_order,
                q05_tm_order=q05_order,
                mean_phase_mad=mean_phase_mad,
                median_sync_residual=median_residual,
                median_adaptive_scale=median_scale,
                max_tm_modulus_error=max_modulus_error,
                local_initial_growth_rate=local_growth_rate,
                near_zero_rate=(
                    float(near_zero_counts[row_index] / attempted_counts[row_index])
                    if attempted_counts[row_index] > 0
                    else math.nan
                ),
                nonfinite_event_rate=(
                    float(nonfinite_counts[row_index] / attempted_counts[row_index])
                    if attempted_counts[row_index] > 0
                    else math.nan
                ),
                finite_completed=finite_completed,
                first_nonfinite_step=int(first_nonfinite_step[row_index]),
                synchronized=bool(
                    finite_completed
                    and mean_order >= config.success_mean_order
                    and q05_order >= config.success_q05_order
                ),
            )
        )

    captured = {
        key: np.stack(values, axis=0)
        for key, values in phase_buffers.items()
        if len(values) == config.evaluation_steps
    }
    return metrics, captured


def threshold_crossing(k_values: FloatArray, curve: FloatArray, level: float = 0.5) -> float:
    """秩序曲線が指定levelを初めて上向きに横切る補間点を返す。"""

    if k_values.size != curve.size or k_values.size < 2:
        return math.nan
    if curve[0] >= level:
        return float(k_values[0])
    for index in range(1, curve.size):
        if curve[index - 1] < level <= curve[index]:
            delta = curve[index] - curve[index - 1]
            if delta == 0.0:
                return float(k_values[index])
            fraction = (level - curve[index - 1]) / delta
            return float(k_values[index - 1] + fraction * (k_values[index] - k_values[index - 1]))
    return math.nan


def maximum_slope_threshold(k_values: FloatArray, curve: FloatArray) -> float:
    """有限差分勾配が最大となる走査点を返す。"""

    if k_values.size != curve.size or k_values.size < 3:
        return math.nan
    gradient = np.gradient(curve, k_values)
    return float(k_values[int(np.nanargmax(gradient))])


def group_metrics(
    metrics: Iterable[SimulationMetrics],
) -> Dict[Tuple[str, int, str], List[SimulationMetrics]]:
    """閾値推定単位でseed別結果をまとめる。"""

    grouped: Dict[Tuple[str, int, str], List[SimulationMetrics]] = {}
    for metric in metrics:
        key = (metric.distribution, metric.node_count, metric.initialization)
        grouped.setdefault(key, []).append(metric)
    return grouped


def build_order_curves(metrics: Iterable[SimulationMetrics]) -> List[Dict[str, Any]]:
    """分布・N・初期化・Kごとのseed集約曲線を作る。"""

    buckets: Dict[Tuple[str, int, str, float], List[SimulationMetrics]] = {}
    for metric in metrics:
        key = (
            metric.distribution,
            metric.node_count,
            metric.initialization,
            metric.coupling_strength,
        )
        buckets.setdefault(key, []).append(metric)

    rows: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        values = buckets[key]
        finite_values = [value for value in values if value.finite_completed]
        orders = np.asarray([value.mean_tm_order for value in finite_values], dtype=np.float64)
        rows.append(
            {
                "distribution": key[0],
                "node_count": key[1],
                "initialization": key[2],
                "coupling_strength": key[3],
                "seed_count": len(values),
                "finite_seed_count": len(finite_values),
                "mean_tm_order": float(np.mean(orders)) if orders.size else math.nan,
                "std_tm_order": float(np.std(orders, ddof=1)) if orders.size > 1 else 0.0,
                "mean_q05_tm_order": (
                    float(np.mean([value.q05_tm_order for value in finite_values]))
                    if finite_values
                    else math.nan
                ),
                "median_sync_residual": (
                    float(np.median([value.median_sync_residual for value in finite_values]))
                    if finite_values
                    else math.nan
                ),
                "synchronization_success_rate": (
                    float(np.mean([value.synchronized for value in values])) if values else math.nan
                ),
            }
        )
    return rows


def bootstrap_thresholds(
    values: Sequence[SimulationMetrics],
    k_values: Sequence[float],
    repetitions: int,
    bootstrap_seed: int,
) -> Dict[str, float]:
    """seed単位でTM凝集点と持続同期成功率50%点の区間を推定する。"""

    ordered_k = np.asarray(tuple(k_values), dtype=np.float64)
    seed_labels = sorted({value.matrix_seed for value in values})
    seed_order_curves: List[FloatArray] = []
    seed_sync_curves: List[FloatArray] = []
    for seed in seed_labels:
        by_k = {
            value.coupling_strength: value
            for value in values
            if value.matrix_seed == seed and value.finite_completed
        }
        if all(k_value in by_k for k_value in k_values):
            seed_order_curves.append(
                np.asarray(
                    [by_k[k_value].mean_tm_order for k_value in k_values],
                    dtype=np.float64,
                )
            )
            seed_sync_curves.append(
                np.asarray(
                    [float(by_k[k_value].synchronized) for k_value in k_values],
                    dtype=np.float64,
                )
            )
    if not seed_order_curves:
        return {
            "r50": math.nan,
            "r50_ci_lower": math.nan,
            "r50_ci_upper": math.nan,
            "r50_bootstrap_valid_fraction": 0.0,
            "sustained_sync_k50": math.nan,
            "sustained_sync_k50_ci_lower": math.nan,
            "sustained_sync_k50_ci_upper": math.nan,
            "sustained_sync_k50_bootstrap_valid_fraction": 0.0,
            "max_slope": math.nan,
            "max_slope_ci_lower": math.nan,
            "max_slope_ci_upper": math.nan,
            "bootstrap_seed_count": 0.0,
        }

    order_matrix = np.stack(seed_order_curves, axis=0)
    sync_matrix = np.stack(seed_sync_curves, axis=0)
    mean_order_curve = np.mean(order_matrix, axis=0)
    mean_sync_curve = np.mean(sync_matrix, axis=0)
    r50 = threshold_crossing(ordered_k, mean_order_curve)
    sustained_sync_k50 = threshold_crossing(ordered_k, mean_sync_curve)
    max_slope = maximum_slope_threshold(ordered_k, mean_order_curve)
    rng = np.random.default_rng(bootstrap_seed)
    r50_samples: List[float] = []
    sync_k50_samples: List[float] = []
    slope_samples: List[float] = []
    for _ in range(repetitions):
        sampled_indices = rng.integers(0, order_matrix.shape[0], size=order_matrix.shape[0])
        sampled_order_curve = np.mean(order_matrix[sampled_indices], axis=0)
        sampled_sync_curve = np.mean(sync_matrix[sampled_indices], axis=0)
        sampled_r50 = threshold_crossing(ordered_k, sampled_order_curve)
        if math.isfinite(sampled_r50):
            r50_samples.append(sampled_r50)
        sampled_sync_k50 = threshold_crossing(ordered_k, sampled_sync_curve)
        if math.isfinite(sampled_sync_k50):
            sync_k50_samples.append(sampled_sync_k50)
        sampled_slope = maximum_slope_threshold(ordered_k, sampled_order_curve)
        if math.isfinite(sampled_slope):
            slope_samples.append(sampled_slope)

    def interval(samples: Sequence[float]) -> Tuple[float, float]:
        """bootstrap標本の両側95%パーセンタイル区間を返す。"""

        if not samples:
            return math.nan, math.nan
        lower, upper = np.quantile(np.asarray(samples), [0.025, 0.975])
        return float(lower), float(upper)

    r50_lower, r50_upper = interval(r50_samples)
    sync_k50_lower, sync_k50_upper = interval(sync_k50_samples)
    slope_lower, slope_upper = interval(slope_samples)
    return {
        "r50": r50,
        "r50_ci_lower": r50_lower,
        "r50_ci_upper": r50_upper,
        "r50_bootstrap_valid_fraction": len(r50_samples) / repetitions,
        "sustained_sync_k50": sustained_sync_k50,
        "sustained_sync_k50_ci_lower": sync_k50_lower,
        "sustained_sync_k50_ci_upper": sync_k50_upper,
        "sustained_sync_k50_bootstrap_valid_fraction": (
            len(sync_k50_samples) / repetitions
        ),
        "max_slope": max_slope,
        "max_slope_ci_lower": slope_lower,
        "max_slope_ci_upper": slope_upper,
        "bootstrap_seed_count": float(order_matrix.shape[0]),
    }

def estimate_all_thresholds(
    metrics: Sequence[SimulationMetrics],
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """全分布・N・初期化について閾値と理論比較量を作る。"""

    rows: List[Dict[str, Any]] = []
    for (distribution, node_count, initialization), values in sorted(group_metrics(metrics).items()):
        bootstrap = bootstrap_thresholds(
            values,
            config.k_values,
            config.bootstrap_repetitions,
            config.bootstrap_seed + node_count + COUPLING_SEED_OFFSETS[distribution],
        )
        sample_absolute_mean = float(np.mean([value.sample_absolute_mean for value in values]))
        sample_signed_mean = float(np.mean([value.sample_signed_mean for value in values]))
        row: Dict[str, Any] = {
            "distribution": distribution,
            "node_count": node_count,
            "initialization": initialization,
            "mean_sample_absolute_mean": sample_absolute_mean,
            "mean_sample_signed_mean": sample_signed_mean,
            "mean_row_mean_std": float(np.mean([value.row_mean_std for value in values])),
            "theory_kc_absolute_sample": critical_coupling(config.alpha, sample_absolute_mean),
            "signed_mean_consistency_kc": (
                critical_coupling(config.alpha, sample_signed_mean)
                if sample_signed_mean > 0.0
                else math.nan
            ),
            **bootstrap,
        }
        row["r50_times_absolute_mean"] = (
            row["r50"] * sample_absolute_mean if math.isfinite(row["r50"]) else math.nan
        )
        row["r50_times_signed_mean"] = (
            row["r50"] * sample_signed_mean if math.isfinite(row["r50"]) else math.nan
        )
        row["sustained_sync_k50_times_absolute_mean"] = (
            row["sustained_sync_k50"] * sample_absolute_mean
            if math.isfinite(row["sustained_sync_k50"])
            else math.nan
        )
        row["sustained_sync_k50_times_signed_mean"] = (
            row["sustained_sync_k50"] * sample_signed_mean
            if math.isfinite(row["sustained_sync_k50"])
            else math.nan
        )
        rows.append(row)
    return rows


def finite_size_diagnostics(
    coupling_statistics: Sequence[CouplingStatistics],
    metrics: Sequence[SimulationMetrics],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """行平均揺らぎのスケーリングと同期残差との記述的相関を求める。"""

    scaling: Dict[str, Dict[str, Any]] = {}
    for distribution in config.coupling_distributions:
        per_n: List[Tuple[int, float]] = []
        for node_count in config.node_counts:
            values = [
                item.row_mean_std
                for item in coupling_statistics
                if item.distribution == distribution and item.node_count == node_count
            ]
            if values:
                per_n.append((node_count, float(np.mean(values))))
        positive = [(node_count, value) for node_count, value in per_n if value > 0.0]
        exponent = math.nan
        if len(positive) >= 2:
            slope, _ = np.polyfit(
                np.log([item[0] for item in positive]),
                np.log([item[1] for item in positive]),
                1,
            )
            exponent = float(slope)
        scaling[distribution] = {
            "per_node_count": [
                {"node_count": node_count, "mean_row_mean_std": value}
                for node_count, value in per_n
            ],
            "log_log_exponent": exponent,
        }

    max_k = max(config.k_values)
    residual_rows = [
        item
        for item in metrics
        if item.initialization == "local"
        and item.coupling_strength == max_k
        and item.distribution != "uniform_positive"
        and item.finite_completed
    ]
    row_stds = np.asarray([item.row_mean_std for item in residual_rows], dtype=np.float64)
    residuals = np.asarray([item.median_sync_residual for item in residual_rows], dtype=np.float64)
    correlation = math.nan
    if row_stds.size >= 3 and np.std(row_stds) > 0.0 and np.std(residuals) > 0.0:
        correlation = float(np.corrcoef(row_stds, residuals)[0, 1])
    return {
        "row_mean_std_scaling": scaling,
        "local_residual_correlation_at_max_k": correlation,
        "correlation_sample_count": int(row_stds.size),
        "correlation_is_descriptive_only": True,
    }


def hypothesis_assessment(
    metrics: Sequence[SimulationMetrics],
    curves: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """事前定義したゲートでH1-H4の観察を機械可読化する。"""

    max_n = max(config.node_counts)
    max_k = max(config.k_values)
    theory_kc = critical_coupling(config.alpha, 1.0)
    scope = config.assessment_scope

    def threshold_row(distribution: str) -> Mapping[str, Any] | None:
        """最大N・localの閾値行を取得する。"""

        return next(
            (
                row
                for row in thresholds
                if row["distribution"] == distribution
                and row["node_count"] == max_n
                and row["initialization"] == "local"
            ),
            None,
        )

    def sustained_sync_k50(distribution: str) -> float:
        """持続同期成功率が50%へ達するKを最大N・local曲線から求める。"""

        row = threshold_row(distribution)
        if row is not None:
            saved_value = row.get("sustained_sync_k50")
            if saved_value is not None and math.isfinite(float(saved_value)):
                return float(saved_value)
        matching = sorted(
            (
                curve
                for curve in curves
                if curve["distribution"] == distribution
                and curve["node_count"] == max_n
                and curve["initialization"] == "local"
            ),
            key=lambda curve: float(curve["coupling_strength"]),
        )
        if not matching:
            return math.nan
        return threshold_crossing(
            np.asarray(
                [curve["coupling_strength"] for curve in matching],
                dtype=np.float64,
            ),
            np.asarray(
                [curve["synchronization_success_rate"] for curve in matching],
                dtype=np.float64,
            ),
        )

    uniform_threshold = threshold_row("uniform_positive")
    uniform_tm_r50 = float(uniform_threshold["r50"]) if uniform_threshold else math.nan
    uniform_sync_k50 = sustained_sync_k50("uniform_positive")
    h1_supported = (
        math.isfinite(uniform_sync_k50)
        and abs(uniform_sync_k50 - theory_kc) <= config.theory_kc_tolerance
    )
    h1_status = (
        f"{scope}_inconclusive"
        if not math.isfinite(uniform_sync_k50)
        else (f"{scope}_supported" if h1_supported else f"{scope}_not_supported")
    )

    normalized_sync_thresholds: Dict[str, float] = {}
    normalized_tm_thresholds: Dict[str, float] = {}
    for distribution in config.coupling_distributions:
        row = threshold_row(distribution)
        if row is not None and math.isfinite(float(row["r50_times_absolute_mean"])):
            normalized_tm_thresholds[distribution] = float(row["r50_times_absolute_mean"])
        sync_k50 = sustained_sync_k50(distribution)
        if (
            row is not None
            and "mean_sample_absolute_mean" in row
            and math.isfinite(sync_k50)
        ):
            normalized_sync_thresholds[distribution] = (
                sync_k50 * float(row["mean_sample_absolute_mean"])
            )
    h2_all_observed = (
        len(config.coupling_distributions) >= 2
        and len(normalized_sync_thresholds) == len(config.coupling_distributions)
    )
    h2_range = (
        max(normalized_sync_thresholds.values()) - min(normalized_sync_thresholds.values())
        if h2_all_observed
        else math.nan
    )
    h2_supported = h2_all_observed and h2_range <= 0.10
    h2_status = (
        f"{scope}_inconclusive"
        if not h2_all_observed
        else (f"{scope}_supported" if h2_supported else f"{scope}_not_supported")
    )

    max_k_orders: Dict[str, float] = {}
    for distribution in config.coupling_distributions:
        matching = [
            float(row["mean_tm_order"])
            for row in curves
            if row["distribution"] == distribution
            and row["node_count"] == max_n
            and row["initialization"] == "local"
            and row["coupling_strength"] == max_k
        ]
        if matching:
            max_k_orders[distribution] = matching[0]
    h3_required = {
        "uniform_positive",
        "positive_binary",
        "biased_sign",
        "rademacher",
    }
    h3_observed = h3_required.issubset(max_k_orders)
    h3_supported = (
        h3_observed
        and max_k_orders["uniform_positive"] >= 0.80
        and max_k_orders["positive_binary"] >= 0.70
        and max_k_orders["rademacher"] <= 0.40
        and max_k_orders["biased_sign"] > max_k_orders["rademacher"]
    )
    h3_status = (
        f"{scope}_inconclusive"
        if not h3_observed
        else (f"{scope}_supported" if h3_supported else f"{scope}_not_supported")
    )

    k_zero_orders = [
        item.mean_tm_order * math.sqrt(item.node_count)
        for item in metrics
        if item.initialization == "global"
        and item.coupling_strength == 0.0
        and item.finite_completed
    ]
    modulus_errors = [
        item.max_tm_modulus_error for item in metrics if item.finite_completed
    ]
    h4_floor = float(np.median(k_zero_orders)) if k_zero_orders else math.nan
    h4_observed = bool(k_zero_orders and modulus_errors and max_k_orders)
    h4_supported = (
        h4_observed
        and math.isfinite(h4_floor)
        and 0.25 <= h4_floor <= 2.5
        and max(max_k_orders.values()) >= 0.80
        and max(modulus_errors) <= 1.0e-12
    )
    h4_status = (
        f"{scope}_inconclusive"
        if not h4_observed
        else (f"{scope}_supported" if h4_supported else f"{scope}_not_supported")
    )
    return {
        "H1_theory_reproduction": {
            "status": h1_status,
            "uniform_local_sustained_sync_k50_max_n": uniform_sync_k50,
            "uniform_local_tm_r50_max_n": uniform_tm_r50,
            "theory_kc": theory_kc,
            "predefined_tolerance": config.theory_kc_tolerance,
            "comparison_metric": "sustained synchronization success rate = 0.5",
        },
        "H2_absolute_moment_universality": {
            "status": h2_status,
            "sustained_sync_k50_times_absolute_mean": normalized_sync_thresholds,
            "tm_r50_times_absolute_mean": normalized_tm_thresholds,
            "range_if_all_sustained_sync_observed": h2_range,
            "all_distributions_reached_sustained_sync_k50": h2_all_observed,
        },
        "H3_signed_mean_consistency": {
            "status": h3_status,
            "local_mean_order_at_max_k": max_k_orders,
            "note": "符号付き平均による置換は独立な整合性proxyであり、論文の解析式ではない。",
        },
        "H4_tm_order_parameter": {
            "status": h4_status,
            "median_k0_order_times_sqrt_n": h4_floor,
            "maximum_tm_modulus_error": max(modulus_errors, default=math.nan),
        },
    }

def write_csv_records(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """同一schemaの辞書列をUTF-8 BOM付きCSVへ保存する。"""

    if not records:
        raise ValueError(f"空のCSVは保存できません: {output_path}")
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def json_safe(value: Any) -> Any:
    """非有限浮動小数をnullへ変換し、標準JSONへ収める。"""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(value: Mapping[str, Any], output_path: Path) -> None:
    """標準JSONとして監査可能な改行付きファイルへ保存する。"""

    output_path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )




def parse_csv_boolean(value: str) -> bool:
    """CSVへ保存した真偽値を曖昧なtruthinessなしで復元する。"""

    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"CSV真偽値として解釈できません: {value}")


def load_simulation_metrics(metrics_path: Path) -> List[SimulationMetrics]:
    """既存のseed別CSVからSimulationMetricsを型付きで復元する。"""

    metrics: List[SimulationMetrics] = []
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            metrics.append(
                SimulationMetrics(
                    distribution=row["distribution"],
                    node_count=int(row["node_count"]),
                    matrix_seed=int(row["matrix_seed"]),
                    initialization=row["initialization"],
                    coupling_strength=float(row["coupling_strength"]),
                    sample_absolute_mean=float(row["sample_absolute_mean"]),
                    sample_signed_mean=float(row["sample_signed_mean"]),
                    row_mean_std=float(row["row_mean_std"]),
                    mean_tm_order=float(row["mean_tm_order"]),
                    q05_tm_order=float(row["q05_tm_order"]),
                    mean_phase_mad=float(row["mean_phase_mad"]),
                    median_sync_residual=float(row["median_sync_residual"]),
                    median_adaptive_scale=float(row["median_adaptive_scale"]),
                    max_tm_modulus_error=float(row["max_tm_modulus_error"]),
                    local_initial_growth_rate=float(row["local_initial_growth_rate"]),
                    near_zero_rate=float(row["near_zero_rate"]),
                    nonfinite_event_rate=float(row["nonfinite_event_rate"]),
                    finite_completed=parse_csv_boolean(row["finite_completed"]),
                    first_nonfinite_step=int(row["first_nonfinite_step"]),
                    synchronized=parse_csv_boolean(row["synchronized"]),
                )
            )
    if not metrics:
        raise ValueError(f"seed別メトリクスCSVが空です: {metrics_path}")
    return metrics


def analysis_caveats(config: ExperimentConfig) -> List[str]:
    """実験結果へ常に添える解釈上の制約を返す。"""

    return [
        (
            f"seed bootstrap区間は{len(config.matrix_seeds)}個の固定seed群内の不確実性であり、"
            "別seed群・別K格子による独立追試を代替しない。"
        ),
        (
            "TM R=0.5は位相の部分凝集クロスオーバーであり、論文のlambda_c=0と同一視しない。"
            "理論Kcとの比較には事前定義した持続同期成功率50%点を用いる。"
        ),
        "持続同期K50も有限K格子とmean R>=0.8かつq05 R>=0.5の判定規則に依存する。",
        "符号付き平均によるKc置換は同期多様体整合性のproxyであり、解析的に証明した式ではない。",
        "globalとlocalの差は局所安定性と吸引域を分けて解釈する。",
    ]

def file_sha256(path: Path) -> str:
    """実行コードと設定の同一性を追跡するSHA-256を返す。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(run_directory: Path) -> Dict[str, Any]:
    """現在のcommitとdirty状態を取得し、失敗時は未確認として残す。"""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=run_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=run_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        return {"git_commit": commit, "git_worktree_dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"git_commit": None, "git_worktree_dirty": None, "git_error": str(error)}


def load_config(config_path: Path) -> ExperimentConfig:
    """実験記録JSONから実際に使用するsimulation設定を読み込む。"""

    record = json.loads(config_path.read_text(encoding="utf-8"))
    simulation = record["simulation"]
    config = ExperimentConfig(
        alpha=float(simulation["alpha"]),
        node_counts=tuple(int(value) for value in simulation["node_counts"]),
        matrix_seeds=tuple(int(value) for value in simulation["matrix_seeds"]),
        coupling_distributions=tuple(str(value) for value in simulation["coupling_distributions"]),
        k_values=tuple(float(value) for value in simulation["k_values"]),
        burn_in=int(simulation["burn_in"]),
        evaluation_steps=int(simulation["evaluation_steps"]),
        local_growth_steps=int(simulation["local_growth_steps"]),
        local_perturbation=float(simulation["local_perturbation"]),
        near_zero_threshold=float(simulation["near_zero_threshold"]),
        bootstrap_repetitions=int(simulation["bootstrap_repetitions"]),
        bootstrap_seed=int(simulation["bootstrap_seed"]),
        success_mean_order=float(simulation["success_mean_order"]),
        success_q05_order=float(simulation["success_q05_order"]),
        phase_heatmap_k_values=tuple(float(value) for value in simulation["phase_heatmap_k_values"]),
        save_figures=bool(simulation.get("save_figures", True)),
        theory_kc_tolerance=float(simulation.get("theory_kc_tolerance", 0.10)),
        assessment_scope=str(simulation.get("assessment_scope", "pilot")),
    )
    config.validate()
    return config


def write_figures(
    artifacts_directory: Path,
    config: ExperimentConfig,
    curves: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, Any]],
    metrics: Sequence[SimulationMetrics],
    captured_phases: Mapping[str, FloatArray],
) -> List[str]:
    """発表で必要な理論・主結果・有限サイズ・位相図を保存する。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: List[str] = []
    k_dense = np.linspace(0.0, max(config.k_values), 400)
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    absolute_curve = [conditional_lyapunov(config.alpha, value, 1.0) for value in k_dense]
    signed_proxy = [conditional_lyapunov(config.alpha, value, 0.75) for value in k_dense]
    axis.plot(k_dense, absolute_curve, label="paper prediction: moment = E|eps| = 1")
    axis.plot(k_dense, signed_proxy, linestyle="--", label="signed-mean consistency proxy: E eps = 0.75")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(critical_coupling(config.alpha, 1.0), color="tab:blue", linestyle=":", label="Kc(abs) = 0.5")
    axis.axvline(critical_coupling(config.alpha, 0.75), color="tab:orange", linestyle=":", label="Kc(mu=0.75)")
    axis.axvline(1.0 - config.alpha, color="tab:red", linestyle="-.", label="Kmax(abs) = 0.75")
    axis.set(xlabel="coupling K", ylabel="conditional Lyapunov exponent", title="Analytical prediction and signed-mean consistency check")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    theory_path = artifacts_directory / "theory_phase_diagram.png"
    fig.savefig(theory_path, dpi=180)
    plt.close(fig)
    figure_paths.append(theory_path.name)

    max_n = max(config.node_counts)
    colors = {
        "uniform_positive": "tab:blue",
        "positive_binary": "tab:green",
        "biased_sign": "tab:orange",
        "rademacher": "tab:red",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True, sharey=True, constrained_layout=True)
    for axis, initialization in zip(axes, ("global", "local")):
        for distribution in config.coupling_distributions:
            rows = sorted(
                (
                    row
                    for row in curves
                    if row["distribution"] == distribution
                    and row["node_count"] == max_n
                    and row["initialization"] == initialization
                ),
                key=lambda row: float(row["coupling_strength"]),
            )
            if not rows:
                continue
            x_values = np.asarray([row["coupling_strength"] for row in rows], dtype=np.float64)
            means = np.asarray([row["mean_tm_order"] for row in rows], dtype=np.float64)
            stds = np.asarray([row["std_tm_order"] for row in rows], dtype=np.float64)
            axis.plot(x_values, means, marker="o", markersize=3, color=colors[distribution], label=distribution)
            axis.fill_between(x_values, np.clip(means - stds, 0.0, 1.0), np.clip(means + stds, 0.0, 1.0), color=colors[distribution], alpha=0.12)
        axis.axvline(critical_coupling(config.alpha, 1.0), color="black", linestyle=":", linewidth=1.0)
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(f"{initialization} initialization, N={max_n}")
        axis.set_xlabel("coupling K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("time-mean adaptive TM order R")
    axes[1].legend(fontsize=8)
    order_path = artifacts_directory / "tm_order_curves.png"
    fig.savefig(order_path, dpi=180)
    plt.close(fig)
    figure_paths.append(order_path.name)

    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for distribution in config.coupling_distributions:
        rows = sorted(
            (
                row
                for row in thresholds
                if row["distribution"] == distribution and row["initialization"] == "local"
            ),
            key=lambda row: int(row["node_count"]),
        )
        if not rows:
            continue
        x_values = np.asarray(
            [1.0 / math.sqrt(int(row["node_count"])) for row in rows],
            dtype=np.float64,
        )
        sync_crossing = np.asarray(
            [row["sustained_sync_k50"] for row in rows],
            dtype=np.float64,
        )
        sync_lower = np.asarray(
            [row["sustained_sync_k50_ci_lower"] for row in rows],
            dtype=np.float64,
        )
        sync_upper = np.asarray(
            [row["sustained_sync_k50_ci_upper"] for row in rows],
            dtype=np.float64,
        )
        finite_sync = np.isfinite(sync_crossing)
        if np.any(finite_sync):
            axis.errorbar(
                x_values[finite_sync],
                sync_crossing[finite_sync],
                yerr=np.vstack(
                    [
                        sync_crossing[finite_sync] - sync_lower[finite_sync],
                        sync_upper[finite_sync] - sync_crossing[finite_sync],
                    ]
                ),
                marker="o",
                capsize=3,
                color=colors[distribution],
                label=distribution,
            )
        else:
            axis.plot([], [], marker="o", color=colors[distribution], label=distribution)
        tm_crossing = np.asarray([row["r50"] for row in rows], dtype=np.float64)
        finite_tm = np.isfinite(tm_crossing)
        axis.plot(
            x_values[finite_tm],
            tm_crossing[finite_tm],
            marker="x",
            linestyle=":",
            color=colors[distribution],
            alpha=0.45,
        )
    axis.axhline(
        critical_coupling(config.alpha, 1.0),
        color="black",
        linestyle="--",
        label="paper Kc(abs)",
    )
    axis.plot(
        [],
        [],
        color="gray",
        marker="o",
        label="sustained synchronization 50%",
    )
    axis.plot([], [], color="gray", marker="x", linestyle=":", label="TM coherence R=0.5")
    axis.set(
        xlabel="1 / sqrt(N)",
        ylabel="empirical coupling K",
        title="Sustained synchronization onset and TM coherence crossover (local)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    finite_path = artifacts_directory / "finite_size_thresholds.png"
    fig.savefig(finite_path, dpi=180)
    plt.close(fig)
    figure_paths.append(finite_path.name)

    max_k = max(config.k_values)
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    plotted_residuals: List[float] = []
    for distribution in config.coupling_distributions:
        rows = [
            row
            for row in metrics
            if row.distribution == distribution
            and row.initialization == "local"
            and row.coupling_strength == max_k
            and row.finite_completed
        ]
        if rows:
            plotted_residuals.extend(row.median_sync_residual for row in rows)
            axis.scatter(
                [row.row_mean_std for row in rows],
                [row.median_sync_residual for row in rows],
                color=colors[distribution],
                label=distribution,
                alpha=0.8,
            )
    axis.set(xlabel="std of signed row means", ylabel="median robust sync residual", title=f"Row-sum heterogeneity and residual at K={max_k:.2f}")
    # 完全同期では residual=0 のみとなるため、その場合は定義できない対数軸を使わない。
    if any(value > 0.0 for value in plotted_residuals):
        axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    residual_path = artifacts_directory / "row_sum_vs_sync_residual.png"
    fig.savefig(residual_path, dpi=180)
    plt.close(fig)
    figure_paths.append(residual_path.name)

    if captured_phases:
        keys = sorted(captured_phases)
        fig, axes = plt.subplots(len(keys), 1, figsize=(9.0, 2.3 * len(keys)), constrained_layout=True)
        axes_array = np.atleast_1d(axes)
        for axis, key in zip(axes_array, keys):
            image = axis.imshow(
                captured_phases[key].T,
                aspect="auto",
                origin="lower",
                cmap="twilight",
                vmin=-math.pi,
                vmax=math.pi,
                interpolation="nearest",
            )
            axis.set(title=key, xlabel="evaluation time", ylabel="node")
        fig.colorbar(image, ax=list(axes_array), label="TM phase [rad]", shrink=0.85)
        heatmap_path = artifacts_directory / "tm_phase_heatmaps.png"
        fig.savefig(heatmap_path, dpi=180)
        plt.close(fig)
        figure_paths.append(heatmap_path.name)
    return figure_paths




def reassess_existing_run(
    run_directory: Path,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """高価なシミュレーションを再実行せず、既存seed別CSVから解析を更新する。"""

    artifacts_directory = run_directory / "artifacts"
    metrics_path = artifacts_directory / "per_run_metrics.csv"
    summary_path = artifacts_directory / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("再解析にはper_run_metrics.csvとsummary.jsonが必要です。")

    metrics = load_simulation_metrics(metrics_path)
    curves = build_order_curves(metrics)
    thresholds = estimate_all_thresholds(metrics, config)
    hypotheses = hypothesis_assessment(metrics, curves, thresholds, config)
    write_csv_records(curves, artifacts_directory / "order_curves.csv")
    write_csv_records(thresholds, artifacts_directory / "thresholds.csv")

    captured_phases: Dict[str, FloatArray] = {}
    phase_path = artifacts_directory / "representative_tm_phases.npz"
    if phase_path.is_file():
        with np.load(phase_path, allow_pickle=False) as archive:
            captured_phases = {
                key: np.asarray(archive[key], dtype=np.float64)
                for key in archive.files
            }
    figure_paths = write_figures(
        artifacts_directory,
        config,
        curves,
        thresholds,
        metrics,
        captured_phases,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["analysis_revision"] = 2
    summary["analysis_correction"] = {
        "reason": "TM R=0.5 coherence crossover was incorrectly compared directly with lambda_c=0.",
        "theory_comparison_metric": "sustained synchronization success rate = 0.5",
        "raw_metrics_sha256": file_sha256(metrics_path),
    }
    summary["assessment_scope"] = config.assessment_scope
    summary["hypothesis_assessment"] = hypotheses
    summary["artifacts"]["figures"] = [
        f"artifacts/{figure_path}" for figure_path in figure_paths
    ]
    summary["caveats"] = analysis_caveats(config)
    write_json(summary, summary_path)

    metrics_record = {
        "schema_version": 2,
        "status": "completed",
        "assessment_scope": config.assessment_scope,
        "primary": {
            "hypothesis_assessment": hypotheses,
            "all_conditions_finite": summary["execution"]["all_conditions_finite"],
        },
        "secondary": summary["artifacts"],
        "notes": summary["caveats"],
    }
    write_json(metrics_record, run_directory / "metrics.json")

    environment_path = run_directory / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment.update(
        {
            "postprocessed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "postprocessing_command": (
                "python e3a_finite_size_sync.py --run-directory . --config config.json"
                f" --assessment-scope {config.assessment_scope} --reassess-existing"
            ),
            "postprocessor_script_sha256": file_sha256(Path(__file__).resolve()),
            "postprocessed_metrics_sha256": file_sha256(metrics_path),
        }
    )
    write_json(environment, environment_path)
    return summary

def run_experiment(run_directory: Path, config: ExperimentConfig) -> Dict[str, Any]:
    """全条件を実行し、seed別結果、閾値、図、環境をrunへ保存する。"""

    config.validate()
    artifacts_directory = run_directory / "artifacts"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    all_metrics: List[SimulationMetrics] = []
    all_statistics: List[CouplingStatistics] = []
    all_captured_phases: Dict[str, FloatArray] = {}
    max_n = max(config.node_counts)
    first_seed = config.matrix_seeds[0]

    for node_count in config.node_counts:
        for distribution in config.coupling_distributions:
            for matrix_seed in config.matrix_seeds:
                matrix = generate_symmetric_coupling(node_count, distribution, matrix_seed)
                statistics = calculate_coupling_statistics(matrix, distribution, matrix_seed)
                if not statistics.symmetric or not statistics.diagonal_zero:
                    raise AssertionError("生成した結合行列が対称・対角0の契約を破りました。")
                all_statistics.append(statistics)
                capture = (
                    node_count == max_n
                    and matrix_seed == first_seed
                    and distribution in {"uniform_positive", "rademacher"}
                )
                case_metrics, captured = simulate_coupling_case(
                    matrix,
                    statistics,
                    config,
                    capture_phases=capture,
                )
                all_metrics.extend(case_metrics)
                all_captured_phases.update(captured)

    metric_records = [asdict(metric) for metric in all_metrics]
    statistic_records = [asdict(statistic) for statistic in all_statistics]
    curves = build_order_curves(all_metrics)
    thresholds = estimate_all_thresholds(all_metrics, config)
    finite_size = finite_size_diagnostics(all_statistics, all_metrics, config)
    hypotheses = hypothesis_assessment(all_metrics, curves, thresholds, config)
    write_csv_records(metric_records, artifacts_directory / "per_run_metrics.csv")
    write_csv_records(statistic_records, artifacts_directory / "coupling_statistics.csv")
    write_csv_records(curves, artifacts_directory / "order_curves.csv")
    write_csv_records(thresholds, artifacts_directory / "thresholds.csv")
    if all_captured_phases:
        np.savez_compressed(artifacts_directory / "representative_tm_phases.npz", **all_captured_phases)

    nonfinite_conditions = sum(not metric.finite_completed for metric in all_metrics)
    figure_paths = (
        write_figures(
            artifacts_directory,
            config,
            curves,
            thresholds,
            all_metrics,
            all_captured_phases,
        )
        if config.save_figures
        else []
    )
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "assessment_scope": config.assessment_scope,
        "question": "有限サイズで論文のKc(abs)を再現でき、同一E|eps|・異なるEepsの結合が同じ同期曲線を持つか。",
        "config": asdict(config),
        "theory": {
            "paper_critical_coupling_for_nominal_m_1": critical_coupling(config.alpha, 1.0),
            "paper_cauchy_existence_upper_k_for_nominal_m_1": (1.0 - config.alpha),
            "source_scope": "infinite-dimensional independent-Cauchy scale closure",
            "signed_mean_proxy_is_not_paper_theory": True,
        },
        "execution": {
            "metric_row_count": len(all_metrics),
            "coupling_matrix_count": len(all_statistics),
            "nonfinite_condition_count": nonfinite_conditions,
            "all_conditions_finite": nonfinite_conditions == 0,
        },
        "hypothesis_assessment": hypotheses,
        "finite_size_diagnostics": finite_size,
        "artifacts": {
            "per_run_metrics": "artifacts/per_run_metrics.csv",
            "coupling_statistics": "artifacts/coupling_statistics.csv",
            "order_curves": "artifacts/order_curves.csv",
            "thresholds": "artifacts/thresholds.csv",
            "representative_tm_phases": (
                "artifacts/representative_tm_phases.npz" if all_captured_phases else None
            ),
            "figures": [f"artifacts/{path}" for path in figure_paths],
        },
        "caveats": analysis_caveats(config),
    }
    write_json(summary, artifacts_directory / "summary.json")

    metrics_record = {
        "schema_version": 1,
        "status": "completed",
        "assessment_scope": config.assessment_scope,
        "primary": {
            "hypothesis_assessment": hypotheses,
            "all_conditions_finite": nonfinite_conditions == 0,
        },
        "secondary": summary["artifacts"],
        "notes": summary["caveats"],
    }
    write_json(metrics_record, run_directory / "metrics.json")

    environment_path = run_directory / "environment.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.is_file()
        else {}
    )
    environment.update(
        {
            "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": [f"numpy=={np.__version__}"],
            "execution_command": (
                "python e3a_finite_size_sync.py --run-directory . --config config.json"
                + (
                    f" --assessment-scope {config.assessment_scope}"
                    if config.assessment_scope != "pilot"
                    else ""
                )
            ),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "config_sha256": file_sha256(run_directory / "config.json"),
            **git_metadata(run_directory),
        }
    )
    write_json(environment, environment_path)
    return summary


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """runディレクトリ、設定、実行または再解析モードを受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="実験記録ディレクトリ。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="run-directoryからの相対パス、または絶対パスの設定JSON。",
    )
    parser.add_argument(
        "--assessment-scope",
        choices=("pilot", "confirmation"),
        default=None,
        help="機械可読な判定ラベルの根拠段階。省略時はconfigまたはpilot。",
    )
    parser.add_argument(
        "--reassess-existing",
        action="store_true",
        help="既存per_run_metrics.csvから閾値・判定・図だけを再生成する。",
    )
    return parser.parse_args(tuple(arguments))


def main(arguments: Sequence[str] | None = None) -> int:
    """設定を読み、E3Aの全実行または既存成果物の再解析を行う。"""

    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    run_directory = args.run_directory.resolve()
    config_path = args.config if args.config.is_absolute() else run_directory / args.config
    config = load_config(config_path)
    if args.assessment_scope is not None:
        config = replace(config, assessment_scope=args.assessment_scope)
    summary = (
        reassess_existing_run(run_directory, config)
        if args.reassess_existing
        else run_experiment(run_directory, config)
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "all_conditions_finite": summary["execution"]["all_conditions_finite"],
                "hypothesis_assessment": summary["hypothesis_assessment"],
            },
            ensure_ascii=False,
        )
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
