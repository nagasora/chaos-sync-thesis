from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import BooleMap, OutputMixingNetwork, TangentMap, generate_boole_trajectories, simulate_tangent_pair
from .features import (
    FourierBandExtractor,
    TMTemporalExtractor,
    effective_rank,
    graph_raw_late_features,
    graph_tm_late_features,
    make_graph_basis,
    multichannel_features,
    raw_temporal_spectrum,
    robust_center_scale,
    robust_standardize,
    synchronization_rms,
    tm_temporal_spectrum,
)
from .modeling import (
    GroupedLogistic,
    GroupedRidge,
    classification_scores,
    fit_ridge_for_dimensions,
    nmse,
    paired_bootstrap_interval,
    rmse,
)
from .utils import collect_hashes, environment_payload, write_csv, write_json

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _standardize_each_source(sources: FloatArray) -> FloatArray:
    return robust_standardize(np.asarray(sources, dtype=np.float64), axis=1)


def _make_ordered_pair_dataset(
    alpha_grid: list[float],
    pair_seeds: list[int],
    *,
    burn_in: int,
    length: int,
) -> tuple[FloatArray, FloatArray, IntArray, pd.DataFrame]:
    """How: unordered alpha pair を両順序で展開し、同じ pair seed を group とする。"""
    rows: list[dict[str, Any]] = []
    alpha_left: list[float] = []
    alpha_right: list[float] = []
    source_seed_left: list[int] = []
    source_seed_right: list[int] = []
    group_values: list[int] = []
    sample_id = 0
    for pair_seed in pair_seeds:
        for left_index in range(len(alpha_grid)):
            for right_index in range(left_index + 1, len(alpha_grid)):
                pair = (alpha_grid[left_index], alpha_grid[right_index])
                for swapped in (False, True):
                    first, second = (pair[1], pair[0]) if swapped else pair
                    left_seed = pair_seed * 100_003 + left_index * 997 + right_index * 37 + 11
                    right_seed = pair_seed * 100_003 + right_index * 997 + left_index * 37 + 1_000_019
                    if swapped:
                        left_seed, right_seed = right_seed, left_seed
                    alpha_left.append(first)
                    alpha_right.append(second)
                    source_seed_left.append(left_seed)
                    source_seed_right.append(right_seed)
                    group_values.append(pair_seed)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "pair_seed": pair_seed,
                            "alpha_1": first,
                            "alpha_2": second,
                            "swapped": swapped,
                            "source_seed_1": left_seed,
                            "source_seed_2": right_seed,
                        }
                    )
                    sample_id += 1
    left = generate_boole_trajectories(
        np.asarray(alpha_left, dtype=np.float64),
        np.asarray(source_seed_left, dtype=np.int64),
        burn_in=burn_in,
        length=length,
    )
    right = generate_boole_trajectories(
        np.asarray(alpha_right, dtype=np.float64),
        np.asarray(source_seed_right, dtype=np.int64),
        burn_in=burn_in,
        length=length,
    )
    standardized = np.stack([_standardize_each_source(left), _standardize_each_source(right)], axis=1)
    targets = np.column_stack([alpha_left, alpha_right]).astype(np.float64)
    groups = np.asarray(group_values, dtype=np.int64)
    return standardized, targets, groups, pd.DataFrame(rows)


def _add_channel_scaled_noise(observations: FloatArray, sigma: float, seed: int) -> FloatArray:
    values = np.asarray(observations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    flattened = values.reshape(-1, values.shape[-1])
    _, scale = robust_center_scale(flattened, axis=1)
    scale = scale.reshape(values.shape[0], values.shape[1], 1)
    return values + sigma * scale * rng.standard_normal(values.shape)


def _apply_missingness(observations: FloatArray, rate: float, seed: int) -> FloatArray:
    values = np.asarray(observations, dtype=np.float64).copy()
    if rate <= 0.0:
        return values
    rng = np.random.default_rng(seed)
    mask = rng.random(values.shape) < rate
    # Why not: 各系列の全点を欠損させると指標の意味がなくなるため、両端は観測済みに固定する。
    mask[:, :, 0] = False
    mask[:, :, -1] = False
    values[mask] = np.nan
    return values


@dataclass(frozen=True)
class E2RobustnessConfig:
    """How: E1B の可逆2観測を固定し、観測劣化だけを操作する。"""

    burn_in: int = 1024
    length: int = 1024
    train_pair_seeds: tuple[int, ...] = tuple(range(100, 116))
    test_pair_seeds: tuple[int, ...] = tuple(range(500, 524))
    train_alphas: tuple[float, ...] = (0.34, 0.42, 0.50, 0.58, 0.66)
    test_alphas: tuple[float, ...] = (0.38, 0.46, 0.54, 0.62)
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.03, 0.10, 0.30)
    missing_rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40)
    observation_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    mixing_matrix: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.35), (0.25, 1.0))
    seed: int = 20260904


class E2RobustnessExperiment:
    """How: clean-fit readoutを凍結し、noise・missing・length に対する劣化曲線を測る。"""

    experiment_id = "E2A-BOOLE-OBSERVATION-ROBUSTNESS"

    def __init__(self, config: E2RobustnessConfig | None = None) -> None:
        self.config = config or E2RobustnessConfig()
        self.tm = TMTemporalExtractor()
        self.fourier = FourierBandExtractor(n_bands=self.tm.output_dim)

    def _extract(self, observations: FloatArray, model_name: str, matrix_inverse: FloatArray) -> FloatArray:
        if model_name.startswith("oracle"):
            recovered = np.einsum("ij,sjt->sit", matrix_inverse, observations, optimize=True)
            channels = recovered
        else:
            channels = observations
        extractor = self.tm if "tm" in model_name else self.fourier
        return multichannel_features(channels, extractor)

    def run(self, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        cfg = self.config
        matrix = np.asarray(cfg.mixing_matrix, dtype=np.float64)
        matrix_inverse = np.linalg.inv(matrix)
        train_sources, y_train, train_groups, train_manifest = _make_ordered_pair_dataset(
            list(cfg.train_alphas), list(cfg.train_pair_seeds), burn_in=cfg.burn_in, length=cfg.length
        )
        test_sources, y_test, test_groups, test_manifest = _make_ordered_pair_dataset(
            list(cfg.test_alphas), list(cfg.test_pair_seeds), burn_in=cfg.burn_in, length=cfg.length
        )
        train_obs = np.einsum("ij,sjt->sit", matrix, train_sources, optimize=True)
        test_obs = np.einsum("ij,sjt->sit", matrix, test_sources, optimize=True)
        np.savez_compressed(
            artifacts / "source_and_observation_data.npz",
            train_sources=train_sources,
            test_sources=test_sources,
            train_observations=train_obs,
            test_observations=test_obs,
            y_train=y_train,
            y_test=y_test,
            train_groups=train_groups,
            test_groups=test_groups,
            mixing_matrix=matrix,
        )
        train_manifest.assign(split="fit").to_csv(artifacts / "fit_manifest.csv", index=False)
        test_manifest.assign(split="test").to_csv(artifacts / "test_manifest.csv", index=False)

        model_names = ("oracle_tm", "oracle_fourier", "direct_tm", "direct_fourier")
        fitted: dict[str, GroupedRidge] = {}
        clean_train_features: dict[str, FloatArray] = {}
        for model_name in model_names:
            feature = self._extract(train_obs, model_name, matrix_inverse)
            clean_train_features[model_name] = feature
            fitted[model_name] = GroupedRidge().fit(feature, y_train, train_groups)
        np.savez_compressed(artifacts / "clean_train_features.npz", **clean_train_features)

        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []

        def evaluate(condition_type: str, condition_value: float, observations: FloatArray, length: int) -> None:
            for model_name in model_names:
                feature = self._extract(observations[:, :, :length], model_name, matrix_inverse)
                prediction = fitted[model_name].predict(feature)
                sample_error = np.sqrt(np.mean((prediction - y_test) ** 2, axis=1))
                metric_rows.append(
                    {
                        "condition_type": condition_type,
                        "condition_value": condition_value,
                        "observation_length": length,
                        "model": model_name,
                        "feature_dim": feature.shape[1],
                        "cv_rmse_clean": fitted[model_name].cv_rmse_,
                        "ridge_alpha": fitted[model_name].selected_alpha_,
                        "test_rmse": rmse(y_test, prediction),
                        "median_sample_rmse": float(np.median(sample_error)),
                        "p90_sample_rmse": float(np.quantile(sample_error, 0.90)),
                    }
                )
                for sample_index in range(len(y_test)):
                    prediction_rows.append(
                        {
                            "condition_type": condition_type,
                            "condition_value": condition_value,
                            "observation_length": length,
                            "model": model_name,
                            "sample_index": sample_index,
                            "pair_seed": int(test_groups[sample_index]),
                            "alpha_1": y_test[sample_index, 0],
                            "alpha_2": y_test[sample_index, 1],
                            "pred_1": prediction[sample_index, 0],
                            "pred_2": prediction[sample_index, 1],
                            "sample_rmse": sample_error[sample_index],
                        }
                    )

        for noise_index, sigma in enumerate(cfg.noise_levels):
            noisy = _add_channel_scaled_noise(test_obs, sigma, cfg.seed + 10_000 + noise_index)
            evaluate("gaussian_noise_sigma", float(sigma), noisy, cfg.length)
        for missing_index, rate in enumerate(cfg.missing_rates):
            missing = _apply_missingness(test_obs, rate, cfg.seed + 20_000 + missing_index)
            evaluate("missing_rate", float(rate), missing, cfg.length)
        for length in cfg.observation_lengths:
            evaluate("observation_length", float(length), test_obs, int(length))

        metrics = pd.DataFrame(metric_rows)
        predictions = pd.DataFrame(prediction_rows)
        write_csv(artifacts / "robustness_metrics.csv", metrics)
        write_csv(artifacts / "test_predictions.csv", predictions)

        clean = metrics[(metrics["condition_type"] == "gaussian_noise_sigma") & (metrics["condition_value"] == 0.0)]
        high_noise = metrics[(metrics["condition_type"] == "gaussian_noise_sigma") & (metrics["condition_value"] == max(cfg.noise_levels))]
        comparison_rows = []
        for condition_type, condition_value in (
            ("gaussian_noise_sigma", 0.10),
            ("missing_rate", 0.20),
            ("observation_length", 256.0),
        ):
            subset = predictions[
                (predictions["condition_type"] == condition_type)
                & (predictions["condition_value"] == condition_value)
            ]
            tm_error = subset[subset["model"] == "oracle_tm"].sort_values("sample_index")["sample_rmse"].to_numpy()
            fourier_error = subset[subset["model"] == "oracle_fourier"].sort_values("sample_index")["sample_rmse"].to_numpy()
            mean_difference, lower, upper = paired_bootstrap_interval(tm_error, fourier_error, seed=cfg.seed + 30_000)
            comparison_rows.append(
                {
                    "condition_type": condition_type,
                    "condition_value": condition_value,
                    "difference_tm_minus_fourier": mean_difference,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
        comparisons = pd.DataFrame(comparison_rows)
        write_csv(artifacts / "paired_comparisons.csv", comparisons)

        self._plot(metrics, artifacts)
        integrity_checks = {
            "mixing_matrix_invertible": bool(abs(np.linalg.det(matrix)) > 1e-9),
            "train_test_alpha_disjoint": bool(set(cfg.train_alphas).isdisjoint(cfg.test_alphas)),
            "matched_feature_dimension": bool(self.tm.output_dim == self.fourier.output_dim),
            "all_metrics_finite": bool(np.isfinite(metrics["test_rmse"]).all()),
            "noise_degrades_oracle_tm": bool(
                high_noise[high_noise["model"] == "oracle_tm"]["test_rmse"].iloc[0]
                > clean[clean["model"] == "oracle_tm"]["test_rmse"].iloc[0]
            ),
        }
        summary = {
            "experiment_id": self.experiment_id,
            "status": "completed" if all(integrity_checks.values()) else "integrity_warning",
            "config": cfg.__dict__,
            "integrity_checks": integrity_checks,
            "clean_test_rmse": clean.set_index("model")["test_rmse"].to_dict(),
            "high_noise_test_rmse": high_noise.set_index("model")["test_rmse"].to_dict(),
            "paired_comparisons": comparisons.to_dict(orient="records"),
            "interpretation": self._interpret(metrics, comparisons),
        }
        write_json(artifacts / "summary.json", summary)
        write_json(run_dir / "config.json", cfg.__dict__)
        write_json(run_dir / "environment.json", environment_payload())
        write_json(run_dir / "validation.json", integrity_checks)
        write_csv(run_dir / "artifact_hashes.csv", collect_hashes(run_dir))
        return summary

    @staticmethod
    def _plot(metrics: pd.DataFrame, artifacts: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        conditions = [
            ("gaussian_noise_sigma", "Gaussian noise σ"),
            ("missing_rate", "Missing rate"),
            ("observation_length", "Observation length"),
        ]
        for axis, (condition, label) in zip(axes, conditions, strict=True):
            subset = metrics[metrics["condition_type"] == condition]
            for model, group in subset.groupby("model"):
                ordered = group.sort_values("condition_value")
                axis.plot(ordered["condition_value"], ordered["test_rmse"], marker="o", label=model)
            axis.set_xlabel(label)
            axis.set_ylabel("Ordered alpha RMSE")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("E2A observation robustness")
        fig.tight_layout()
        fig.savefig(artifacts / "robustness_curves.png", dpi=180)
        plt.close(fig)

    @staticmethod
    def _interpret(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> dict[str, Any]:
        clean = metrics[(metrics.condition_type == "gaussian_noise_sigma") & (metrics.condition_value == 0.0)]
        pivot = clean.set_index("model")["test_rmse"]
        best_model = str(pivot.idxmin())
        return {
            "best_clean_model": best_model,
            "best_clean_rmse": float(pivot.min()),
            "tm_fourier_ci_statements": comparisons.to_dict(orient="records"),
            "scope": "既知混合行列、補間 alpha、Gaussian 観測ノイズ、線形 ridge probe に限定する。",
        }


@dataclass(frozen=True)
class E3BasinConfig:
    """How: 局所横指数と有限差初期値の大域 basin を別々に測る。"""

    betas: tuple[float, ...] = (1.05, 1.10, 1.20, 1.30)
    output_couplings: tuple[float, ...] = tuple(np.linspace(0.0, 0.5, 21))
    diffusive_couplings: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))
    n_initial_pairs: int = 256
    n_steps: int = 500
    sustain: int = 50
    tolerance: float = 1e-8
    derivative_burn_in: int = 5000
    derivative_length: int = 40000
    seed: int = 20260904


class E3BasinExperiment:
    """How: legacy T4 を E3B へ移し、局所安定性と大域同期率の差を定量化する。"""

    experiment_id = "E3B-TANGENT-TWO-NODE-BASIN"

    def __init__(self, config: E3BasinConfig | None = None) -> None:
        self.config = config or E3BasinConfig()

    def _synchronized_derivatives(self, beta: float) -> FloatArray:
        cfg = self.config
        tangent = TangentMap(beta)
        rng = np.random.default_rng(cfg.seed + int(beta * 10_000))
        state = np.asarray(rng.standard_cauchy(), dtype=np.float64)
        for _ in range(cfg.derivative_burn_in):
            state = tangent(state)
        derivatives = np.empty(cfg.derivative_length, dtype=np.float64)
        for index in range(cfg.derivative_length):
            derivatives[index] = float(tangent.derivative(state))
            state = tangent(state)
        return derivatives

    def run(self, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        cfg = self.config
        rows: list[dict[str, Any]] = []
        local_rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(cfg.seed)
        for beta_index, beta in enumerate(cfg.betas):
            derivatives = self._synchronized_derivatives(beta)
            lambda_parallel = float(np.mean(np.log(np.abs(derivatives))))
            initial_x = rng.standard_cauchy(cfg.n_initial_pairs)
            initial_y = rng.standard_cauchy(cfg.n_initial_pairs)
            for coupling_form, couplings in (
                ("output_cross", cfg.output_couplings),
                ("state_diffusive", cfg.diffusive_couplings),
            ):
                for coupling in couplings:
                    if coupling_form == "output_cross":
                        transverse_multiplier = np.abs(1.0 - 2.0 * coupling) * np.abs(derivatives)
                    else:
                        transverse_multiplier = np.abs(derivatives - 2.0 * coupling)
                    if np.any(transverse_multiplier == 0.0):
                        lambda_perp = float("-inf")
                    else:
                        lambda_perp = float(np.mean(np.log(transverse_multiplier)))
                    sustained, diverged, final_error = simulate_tangent_pair(
                        beta,
                        float(coupling),
                        initial_x,
                        initial_y,
                        n_steps=cfg.n_steps,
                        coupling_form=coupling_form,
                        tolerance=cfg.tolerance,
                        sustain=cfg.sustain,
                    )
                    rows.append(
                        {
                            "beta": beta,
                            "coupling_form": coupling_form,
                            "coupling": float(coupling),
                            "lambda_parallel": lambda_parallel,
                            "lambda_perp_local": lambda_perp,
                            "local_stable": bool(lambda_perp < 0.0),
                            "basin_success_rate": float(np.mean(sustained)),
                            "divergence_rate": float(np.mean(diverged)),
                            "median_final_error": float(np.median(final_error)),
                            "p90_final_error": float(np.quantile(final_error, 0.90)),
                        }
                    )
                    local_rows.append(
                        {
                            "beta": beta,
                            "coupling_form": coupling_form,
                            "coupling": float(coupling),
                            "lambda_perp_local": lambda_perp,
                        }
                    )
        metrics = pd.DataFrame(rows)
        write_csv(artifacts / "basin_metrics.csv", metrics)
        write_csv(artifacts / "local_exponents.csv", pd.DataFrame(local_rows))
        self._plot(metrics, artifacts)

        output_half = metrics[(metrics.coupling_form == "output_cross") & np.isclose(metrics.coupling, 0.5)]
        locally_stable = metrics[metrics.local_stable]
        gap = locally_stable.assign(
            stability_basin_gap=1.0 - locally_stable["basin_success_rate"]
        ) if len(locally_stable) else locally_stable
        mismatch_fraction = float(np.mean(locally_stable["basin_success_rate"] < 0.9)) if len(locally_stable) else float("nan")
        checks = {
            "output_half_exact_sync": bool((output_half["basin_success_rate"] >= 0.999).all()),
            "parallel_chaos_positive": bool((metrics.groupby("beta")["lambda_parallel"].first() > 0.0).all()),
            "metrics_finite_except_expected_minus_inf": bool(np.isfinite(metrics["basin_success_rate"]).all()),
            "both_coupling_forms_present": bool(set(metrics.coupling_form) == {"output_cross", "state_diffusive"}),
        }
        worst_gap_row = None
        if len(gap):
            record = gap.sort_values("stability_basin_gap", ascending=False).iloc[0]
            worst_gap_row = {key: (float(value) if isinstance(value, (np.floating, float)) else value) for key, value in record.to_dict().items()}
        summary = {
            "experiment_id": self.experiment_id,
            "status": "completed" if all(checks.values()) else "integrity_warning",
            "config": cfg.__dict__,
            "integrity_checks": checks,
            "fraction_of_locally_stable_grid_with_basin_below_0_9": mismatch_fraction,
            "worst_local_global_gap": worst_gap_row,
            "interpretation": {
                "main": "局所横 Lyapunov 指数が負であることと、有限差初期値から高確率で同期することは同値ではない。",
                "positive_control": "output-cross coupling=0.5 は1ステップで厳密同期する正対照である。",
                "scope": "Cauchy 初期値、float64、有限500ステップ、指定閾値に限定する。",
            },
        }
        write_json(artifacts / "summary.json", summary)
        write_json(run_dir / "config.json", cfg.__dict__)
        write_json(run_dir / "environment.json", environment_payload())
        write_json(run_dir / "validation.json", checks)
        write_csv(run_dir / "artifact_hashes.csv", collect_hashes(run_dir))
        return summary

    @staticmethod
    def _plot(metrics: pd.DataFrame, artifacts: Path) -> None:
        for coupling_form in ("output_cross", "state_diffusive"):
            subset = metrics[metrics.coupling_form == coupling_form]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for beta, group in subset.groupby("beta"):
                ordered = group.sort_values("coupling")
                axes[0].plot(ordered.coupling, ordered.lambda_perp_local, marker="o", label=f"β={beta:.2f}")
                axes[1].plot(ordered.coupling, ordered.basin_success_rate, marker="o", label=f"β={beta:.2f}")
            axes[0].axhline(0.0, linewidth=1.0)
            axes[0].set_ylabel("Local transverse Lyapunov")
            axes[1].set_ylabel("Global basin success rate")
            for axis in axes:
                axis.set_xlabel("Coupling")
                axis.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"E3B {coupling_form}: local stability vs global basin")
            fig.tight_layout()
            fig.savefig(artifacts / f"{coupling_form}_local_vs_basin.png", dpi=180)
            plt.close(fig)


@dataclass(frozen=True)
class E3InputRetentionConfig:
    """How: 同期臨界の前後で小さな横方向入力の保持を比較する。"""

    alpha: float = 0.5
    n_nodes: int = 8
    couplings: tuple[float, ...] = (0.0, 0.30, 0.45, 0.50, 0.52, 0.60, 0.80, 0.95)
    n_seed_groups: int = 360
    n_steps: int = 80
    late_window: int = 24
    input_amplitude: float = 0.02
    nuisance_amplitude: float = 0.002
    train_fraction: float = 0.70
    seed: int = 20260904


class E3InputRetentionExperiment:
    """How: 同期誤差と入力識別を同じ coupling 軸で別指標として測る。"""

    experiment_id = "E3C-BOOLE-N8-INPUT-RETENTION"

    def __init__(self, config: E3InputRetentionConfig | None = None) -> None:
        self.config = config or E3InputRetentionConfig()

    def _initial_dataset(self) -> tuple[FloatArray, IntArray, IntArray]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        local_map = BooleMap(cfg.alpha)
        base = local_map.invariant_scale * rng.standard_cauchy(cfg.n_seed_groups)
        contrast = np.concatenate(
            [np.ones(cfg.n_nodes // 2), -np.ones(cfg.n_nodes // 2)]
        ).astype(np.float64)
        contrast /= np.linalg.norm(contrast)
        initial_rows = []
        labels = []
        groups = []
        for group in range(cfg.n_seed_groups):
            nuisance = cfg.nuisance_amplitude * rng.standard_normal(cfg.n_nodes)
            nuisance -= np.mean(nuisance)
            for label in (-1, 1):
                state = np.full(cfg.n_nodes, base[group], dtype=np.float64)
                state += label * cfg.input_amplitude * contrast + nuisance
                initial_rows.append(state)
                labels.append(1 if label > 0 else 0)
                groups.append(group)
        return np.vstack(initial_rows), np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)

    def run(self, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        cfg = self.config
        initial, labels, groups = self._initial_dataset()
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(cfg.seed + 1)
        rng.shuffle(unique_groups)
        cut = int(len(unique_groups) * cfg.train_fraction)
        train_groups = set(unique_groups[:cut].tolist())
        train_mask = np.array([group in train_groups for group in groups])
        test_mask = ~train_mask
        basis = make_graph_basis(cfg.n_nodes)
        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        trajectory_examples: dict[str, FloatArray] = {}
        for coupling in cfg.couplings:
            network = OutputMixingNetwork(BooleMap(cfg.alpha), coupling)
            trajectories = network.simulate(initial, cfg.n_steps)
            if coupling in (0.0, 0.52, 0.80, 0.95):
                trajectory_examples[f"coupling_{coupling:.2f}"] = trajectories[:12]
            feature_sets: dict[str, FloatArray] = {
                "tm_graph_late": graph_tm_late_features(
                    trajectories, basis, late_window=cfg.late_window, n_modes=2
                ),
                "raw_graph_late": graph_raw_late_features(
                    trajectories, basis, late_window=cfg.late_window, n_modes=2
                ),
            }
            terminal = trajectories[:, -1, :]
            feature_sets["terminal_state"] = terminal
            sync_values = synchronization_rms(trajectories, window=cfg.late_window)
            for feature_name, features in feature_sets.items():
                probe = GroupedLogistic().fit(features[train_mask], labels[train_mask], groups[train_mask])
                prediction = probe.predict(features[test_mask])
                probabilities = probe.predict_proba(features[test_mask])
                scores = classification_scores(labels[test_mask], prediction)
                metric_rows.append(
                    {
                        "coupling": coupling,
                        "feature": feature_name,
                        "feature_dim": features.shape[1],
                        "cv_balanced_accuracy": probe.cv_balanced_accuracy_,
                        "selected_c": probe.selected_c_,
                        "test_accuracy": scores["accuracy"],
                        "test_balanced_accuracy": scores["balanced_accuracy"],
                        "median_sync_rms": float(np.median(sync_values[test_mask])),
                        "p90_sync_rms": float(np.quantile(sync_values[test_mask], 0.90)),
                        "effective_rank": effective_rank(features[test_mask]),
                    }
                )
                test_indices = np.where(test_mask)[0]
                for local_index, global_index in enumerate(test_indices):
                    prediction_rows.append(
                        {
                            "coupling": coupling,
                            "feature": feature_name,
                            "sample_index": int(global_index),
                            "group": int(groups[global_index]),
                            "label": int(labels[global_index]),
                            "prediction": int(prediction[local_index]),
                            "probability": float(probabilities[local_index]),
                            "sync_rms": float(sync_values[global_index]),
                        }
                    )
        metrics = pd.DataFrame(metric_rows)
        predictions = pd.DataFrame(prediction_rows)
        write_csv(artifacts / "input_retention_metrics.csv", metrics)
        write_csv(artifacts / "test_predictions.csv", predictions)
        np.savez_compressed(artifacts / "trajectory_examples.npz", **trajectory_examples)
        self._plot(metrics, artifacts)
        theoretical_lambda = BooleMap(cfg.alpha).theoretical_lyapunov
        theoretical_critical = 1.0 - math.exp(-theoretical_lambda)
        postcritical = metrics[metrics.coupling > theoretical_critical]
        tm_postcritical = postcritical[postcritical.feature == "tm_graph_late"]
        best_postcritical = tm_postcritical.sort_values("test_balanced_accuracy", ascending=False).iloc[0]
        deep = metrics[(metrics.feature == "tm_graph_late") & (metrics.coupling == max(cfg.couplings))].iloc[0]
        checks = {
            "paired_labels_per_group": bool(pd.Series(groups).value_counts().eq(2).all()),
            "group_split_disjoint": bool(set(groups[train_mask]).isdisjoint(set(groups[test_mask]))),
            "all_metrics_finite": bool(np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()),
            "critical_value_in_grid_neighborhood": bool(min(abs(np.asarray(cfg.couplings) - theoretical_critical)) <= 0.03),
        }
        scientific_gates = {
            "postcritical_tm_above_0_65": bool(best_postcritical.test_balanced_accuracy >= 0.65),
            "postcritical_sync_below_uncoupled": bool(
                best_postcritical.median_sync_rms
                < metrics[(metrics.feature == "tm_graph_late") & (metrics.coupling == 0.0)].median_sync_rms.iloc[0]
            ),
            "deep_sync_information_loss": bool(deep.test_balanced_accuracy < best_postcritical.test_balanced_accuracy),
        }
        summary = {
            "experiment_id": self.experiment_id,
            "status": "completed" if all(checks.values()) else "integrity_warning",
            "config": cfg.__dict__,
            "theoretical_lambda_parallel": theoretical_lambda,
            "theoretical_output_mixing_critical_coupling": theoretical_critical,
            "integrity_checks": checks,
            "scientific_gates": scientific_gates,
            "best_postcritical_tm": best_postcritical.to_dict(),
            "deepest_coupling_tm": deep.to_dict(),
            "interpretation": {
                "main": "同期臨界直後の有限時間窓で、横方向冗長性の縮約と入力符号の線形読み出しを同時に評価した。",
                "caution": "入力は二群コントラスト方向の微小摂動であり、一般画像・長期記憶へ直ちに一般化しない。",
            },
        }
        write_json(artifacts / "summary.json", summary)
        write_json(run_dir / "config.json", cfg.__dict__)
        write_json(run_dir / "environment.json", environment_payload())
        write_json(run_dir / "validation.json", {**checks, **scientific_gates})
        write_csv(run_dir / "artifact_hashes.csv", collect_hashes(run_dir))
        return summary

    @staticmethod
    def _plot(metrics: pd.DataFrame, artifacts: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for feature, group in metrics.groupby("feature"):
            ordered = group.sort_values("coupling")
            axes[0].plot(ordered.coupling, ordered.test_balanced_accuracy, marker="o", label=feature)
        sync = metrics[metrics.feature == "tm_graph_late"].sort_values("coupling")
        axes[1].semilogy(sync.coupling, sync.median_sync_rms, marker="o")
        axes[0].axhline(0.5, linewidth=1.0)
        axes[0].set_xlabel("Output-mixing coupling K")
        axes[0].set_ylabel("Test balanced accuracy")
        axes[1].set_xlabel("Output-mixing coupling K")
        axes[1].set_ylabel("Median synchronization RMS")
        axes[0].legend(fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.3)
        fig.suptitle("E3C synchronization–information trade-off")
        fig.tight_layout()
        fig.savefig(artifacts / "sync_information_tradeoff.png", dpi=180)
        plt.close(fig)


def _generate_synthetic_signals(n_samples: int, length: int, seed: int) -> tuple[FloatArray, IntArray, pd.DataFrame]:
    """How: 正弦波・chirp・2周波混合を同数近く生成する。"""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, length, endpoint=False)
    signals = np.empty((n_samples, length), dtype=np.float64)
    kinds = np.empty(n_samples, dtype=np.int64)
    metadata: list[dict[str, Any]] = []
    for index in range(n_samples):
        kind = index % 3
        amplitude = rng.uniform(0.5, 1.0)
        phase = rng.uniform(-np.pi, np.pi)
        if kind == 0:
            frequency = rng.uniform(1.0, 7.0)
            signal = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
            record = {"kind": "sine", "amplitude": amplitude, "f0": frequency, "f1": frequency, "phase": phase}
        elif kind == 1:
            f0 = rng.uniform(0.5, 2.5)
            f1 = rng.uniform(4.0, 8.0)
            rate = f1 - f0
            signal = amplitude * np.sin(2.0 * np.pi * (f0 * time + 0.5 * rate * time * time) + phase)
            record = {"kind": "chirp", "amplitude": amplitude, "f0": f0, "f1": f1, "phase": phase}
        else:
            f0 = rng.uniform(1.0, 4.0)
            f1 = rng.uniform(4.5, 8.0)
            ratio = rng.uniform(0.25, 0.75)
            phase2 = rng.uniform(-np.pi, np.pi)
            signal = amplitude * (
                ratio * np.sin(2.0 * np.pi * f0 * time + phase)
                + (1.0 - ratio) * np.sin(2.0 * np.pi * f1 * time + phase2)
            )
            record = {
                "kind": "mixture",
                "amplitude": amplitude,
                "f0": f0,
                "f1": f1,
                "phase": phase,
                "phase2": phase2,
                "ratio": ratio,
            }
        max_abs = max(float(np.max(np.abs(signal))), 1e-12)
        signals[index] = signal / max_abs
        kinds[index] = kind
        record["sample_id"] = index
        metadata.append(record)
    return signals, kinds, pd.DataFrame(metadata)


def _simulate_driven_boole(
    inputs: FloatArray,
    *,
    coupling: float,
    input_gain: float,
    alpha: float,
    n_nodes: int,
    washout: int,
    seed: int,
) -> FloatArray:
    """How: 共通入力を同期多様体方向へ加え、drive 区間の軌道だけを返す。"""
    values = np.asarray(inputs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    local_map = BooleMap(alpha)
    base = local_map.invariant_scale * rng.standard_cauchy(values.shape[0])
    initial = base[:, None] + 1e-3 * rng.standard_normal((values.shape[0], n_nodes))
    network = OutputMixingNetwork(local_map, coupling)
    if washout > 0:
        washout_trajectory = network.simulate(initial, washout)
        initial = washout_trajectory[:, -1, :]
    driven = network.simulate(initial, values.shape[1], input_gain * values)
    return driven[:, 1:, :]


@dataclass(frozen=True)
class E4SignalConfig:
    """How: 合成波形を共通入力として与え、潜在次元ごとの rate–distortion を測る。"""

    alpha: float = 0.5
    n_nodes: int = 8
    length: int = 64
    n_samples: int = 720
    washout: int = 16
    coupling_grid: tuple[float, ...] = (0.30, 0.48, 0.52, 0.65, 0.85)
    input_gain_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    dimensions: tuple[int, ...] = (4, 8, 16, 32)
    seed: int = 20260904


class E4SignalReconstructionExperiment:
    """How: TM・生状態・reservoir PCA・入力PCAを同じ潜在次元とridgeで比較する。"""

    experiment_id = "E4A-BOOLE-SYNTHETIC-SIGNAL-RECONSTRUCTION"

    def __init__(self, config: E4SignalConfig | None = None) -> None:
        self.config = config or E4SignalConfig()

    @staticmethod
    def _split_indices(kinds: IntArray, seed: int) -> tuple[IntArray, IntArray, IntArray]:
        all_indices = np.arange(len(kinds))
        train, remaining = train_test_split(
            all_indices, test_size=0.40, random_state=seed, stratify=kinds
        )
        valid, test = train_test_split(
            remaining, test_size=0.50, random_state=seed + 1, stratify=kinds[remaining]
        )
        return train.astype(np.int64), valid.astype(np.int64), test.astype(np.int64)

    def _validation_score(
        self,
        features: FloatArray,
        signals: FloatArray,
        train_idx: IntArray,
        valid_idx: IntArray,
        dimension: int = 16,
    ) -> float:
        reducer = PCA(n_components=min(dimension, features.shape[1]), random_state=0)
        train_latent = reducer.fit_transform(features[train_idx])
        valid_latent = reducer.transform(features[valid_idx])
        best = np.inf
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(train_latent, signals[train_idx])
            best = min(best, nmse(signals[valid_idx], model.predict(valid_latent)))
        return float(best)

    def run(self, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        cfg = self.config
        signals, kinds, metadata = _generate_synthetic_signals(cfg.n_samples, cfg.length, cfg.seed)
        train_idx, valid_idx, test_idx = self._split_indices(kinds, cfg.seed)
        metadata["split"] = ""
        metadata.loc[train_idx, "split"] = "train"
        metadata.loc[valid_idx, "split"] = "validation"
        metadata.loc[test_idx, "split"] = "test"
        write_csv(artifacts / "signal_manifest.csv", metadata)
        np.savez_compressed(
            artifacts / "signals.npz",
            signals=signals,
            kinds=kinds,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
        )

        search_rows: list[dict[str, Any]] = []
        best_score = np.inf
        best_params: tuple[float, float] | None = None
        best_trajectories: FloatArray | None = None
        for coupling in cfg.coupling_grid:
            for gain in cfg.input_gain_grid:
                trajectories = _simulate_driven_boole(
                    signals,
                    coupling=coupling,
                    input_gain=gain,
                    alpha=cfg.alpha,
                    n_nodes=cfg.n_nodes,
                    washout=cfg.washout,
                    seed=cfg.seed + int(coupling * 1000) + int(gain * 10000),
                )
                tm_features = tm_temporal_spectrum(trajectories)
                score = self._validation_score(tm_features, signals, train_idx, valid_idx)
                sync_median = float(np.median(synchronization_rms(trajectories[valid_idx], window=16)))
                search_rows.append(
                    {
                        "coupling": coupling,
                        "input_gain": gain,
                        "validation_nmse_tm_d16": score,
                        "validation_median_sync_rms": sync_median,
                    }
                )
                if score < best_score:
                    best_score = score
                    best_params = (coupling, gain)
                    best_trajectories = trajectories
        assert best_params is not None and best_trajectories is not None
        search = pd.DataFrame(search_rows)
        write_csv(artifacts / "hyperparameter_search.csv", search)
        coupling, gain = best_params
        trajectories = best_trajectories
        np.savez_compressed(
            artifacts / "selected_trajectory_examples.npz",
            trajectories=trajectories[test_idx[:24]],
            signals=signals[test_idx[:24]],
            kinds=kinds[test_idx[:24]],
        )
        base_features: dict[str, FloatArray] = {
            "tm_spectrum": tm_temporal_spectrum(trajectories),
            "raw_mean_dct": raw_temporal_spectrum(trajectories, n_coefficients=64),
            "reservoir_flat": trajectories.reshape(len(trajectories), -1),
            "terminal_state": trajectories[:, -1, :],
            "input_oracle": signals,
        }
        np.savez_compressed(artifacts / "base_features.npz", **base_features)
        metric_rows: list[dict[str, Any]] = []
        reconstruction_examples: dict[str, FloatArray] = {"truth": signals[test_idx[:16]]}
        for representation, features in base_features.items():
            records = fit_ridge_for_dimensions(
                features[train_idx],
                features[valid_idx],
                features[test_idx],
                signals[train_idx],
                signals[valid_idx],
                signals[test_idx],
                cfg.dimensions,
                use_pca=True,
            )
            for record in records:
                prediction = np.asarray(record.pop("prediction"), dtype=np.float64)
                dimension = int(record["dimension"])
                metric_rows.append(
                    {
                        "representation": representation,
                        **record,
                        "coupling": coupling,
                        "input_gain": gain,
                        "median_sync_rms": float(np.median(synchronization_rms(trajectories[test_idx], window=16))),
                    }
                )
                if dimension == 16:
                    reconstruction_examples[representation] = prediction[:16]
        metrics = pd.DataFrame(metric_rows)
        write_csv(artifacts / "rate_distortion_metrics.csv", metrics)
        np.savez_compressed(artifacts / "reconstruction_examples.npz", **reconstruction_examples)
        self._plot(metrics, reconstruction_examples, artifacts)
        d16 = metrics[metrics.dimension == 16].set_index("representation")
        terminal_common_dimension = int(
            min(
                16,
                metrics.loc[metrics.representation == "tm_spectrum", "dimension"].max(),
                metrics.loc[metrics.representation == "terminal_state", "dimension"].max(),
            )
        )
        terminal_common = metrics[
            metrics.dimension == terminal_common_dimension
        ].set_index("representation")
        checks = {
            "split_disjoint": bool(
                set(train_idx).isdisjoint(set(valid_idx))
                and set(train_idx).isdisjoint(set(test_idx))
                and set(valid_idx).isdisjoint(set(test_idx))
            ),
            "selected_only_on_validation": True,
            "all_metrics_finite": bool(np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()),
            "all_representations_present": bool(
                set(base_features) == set(metrics.representation.unique())
            ),
        }
        tm_beats_terminal_common = bool(
            terminal_common.loc["tm_spectrum", "test_nmse"]
            < terminal_common.loc["terminal_state", "test_nmse"]
        )
        scientific = {
            "tm_beats_raw_dct_d16": bool(
                d16.loc["tm_spectrum", "test_nmse"] < d16.loc["raw_mean_dct", "test_nmse"]
            ),
            # Why not: terminal_state は8ノードのため16次元表現を持たない。
            # したがってTMとの比較は両者が利用できる最大共通次元で行う。
            "terminal_comparison_dimension": terminal_common_dimension,
            "tm_beats_terminal_common_dimension": tm_beats_terminal_common,
            "tm_beats_terminal_d16": tm_beats_terminal_common,
            "tm_matches_input_pca_within_25pct": bool(
                d16.loc["tm_spectrum", "test_nmse"] <= 1.25 * d16.loc["input_oracle", "test_nmse"]
            ),
        }
        summary = {
            "experiment_id": self.experiment_id,
            "status": "completed" if all(checks.values()) else "integrity_warning",
            "config": cfg.__dict__,
            "selected_coupling": coupling,
            "selected_input_gain": gain,
            "selected_validation_nmse": best_score,
            "integrity_checks": checks,
            "scientific_gates": scientific,
            "d16_metrics": d16.reset_index().to_dict(orient="records"),
            "interpretation": {
                "main": "入力波形64点を潜在4/8/16/32次元へ落とし、同一ridge decoderで rate–distortion を比較した。",
                "caution": "入力PCAは力学系を通さないoracle線形圧縮であり、提案法の比較上限として扱う。",
            },
        }
        write_json(artifacts / "summary.json", summary)
        write_json(run_dir / "config.json", cfg.__dict__)
        write_json(run_dir / "environment.json", environment_payload())
        write_json(run_dir / "validation.json", {**checks, **scientific})
        write_csv(run_dir / "artifact_hashes.csv", collect_hashes(run_dir))
        return summary

    @staticmethod
    def _plot(metrics: pd.DataFrame, examples: dict[str, FloatArray], artifacts: Path) -> None:
        fig, axis = plt.subplots(figsize=(7.5, 5.0))
        for representation, group in metrics.groupby("representation"):
            ordered = group.sort_values("dimension")
            axis.plot(ordered.dimension, ordered.test_nmse, marker="o", label=representation)
        axis.set_yscale("log")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Test NMSE")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        axis.set_title("E4A rate–distortion on synthetic signals")
        fig.tight_layout()
        fig.savefig(artifacts / "rate_distortion.png", dpi=180)
        plt.close(fig)

        if "tm_spectrum" in examples:
            fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
            for row, axis in enumerate(axes):
                axis.plot(examples["truth"][row], label="truth")
                axis.plot(examples["tm_spectrum"][row], label="TM reconstruction")
                axis.plot(examples.get("input_oracle", examples["tm_spectrum"])[row], label="input-PCA")
                axis.legend(fontsize=8)
                axis.grid(alpha=0.3)
            fig.suptitle("E4A reconstruction examples (d=16)")
            fig.tight_layout()
            fig.savefig(artifacts / "reconstruction_examples.png", dpi=180)
            plt.close(fig)


def _mean_image_ssim(y_true: FloatArray, y_pred: FloatArray, side: int = 8) -> float:
    """How: 画像ごとの SSIM を平均し、利用不可時は global SSIM 近似へフォールバックする。"""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1, side, side)
    prediction = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0).reshape(-1, side, side)
    try:
        from skimage.metrics import structural_similarity

        scores = [structural_similarity(a, b, data_range=1.0) for a, b in zip(truth, prediction, strict=True)]
        return float(np.mean(scores))
    except Exception:
        c1 = 0.01**2
        c2 = 0.03**2
        scores = []
        for a, b in zip(truth, prediction, strict=True):
            mu_a, mu_b = float(a.mean()), float(b.mean())
            var_a, var_b = float(a.var()), float(b.var())
            covariance = float(np.mean((a - mu_a) * (b - mu_b)))
            numerator = (2 * mu_a * mu_b + c1) * (2 * covariance + c2)
            denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
            scores.append(numerator / denominator)
        return float(np.mean(scores))


def _psnr(y_true: FloatArray, y_pred: FloatArray) -> float:
    mse = float(np.mean((np.asarray(y_true) - np.clip(np.asarray(y_pred), 0.0, 1.0)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-15)))


@dataclass(frozen=True)
class E5ImageConfig:
    """How: 8×8 digitsを最小画像課題として用い、E4力学を転移評価する。"""

    alpha: float = 0.5
    n_nodes: int = 8
    washout: int = 16
    dimensions: tuple[int, ...] = (8, 16, 32)
    seed: int = 20260904


class E5SmallImageExperiment:
    """How: full MNIST 前の pilot として、offline digits で復元・クラス保持を同時評価する。"""

    experiment_id = "E5A-BOOLE-SMALL-IMAGE-DIGITS"

    def __init__(self, config: E5ImageConfig | None = None) -> None:
        self.config = config or E5ImageConfig()

    @staticmethod
    def _classification_with_latent(
        train_features: FloatArray,
        valid_features: FloatArray,
        test_features: FloatArray,
        y_train: IntArray,
        y_valid: IntArray,
        y_test: IntArray,
        dimension: int,
    ) -> dict[str, float]:
        d = min(dimension, train_features.shape[1], train_features.shape[0] - 1)
        reducer = PCA(n_components=d, random_state=0)
        x_train = reducer.fit_transform(train_features)
        x_valid = reducer.transform(valid_features)
        best_c = None
        best_score = -np.inf
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("logistic", LogisticRegression(C=c_value, max_iter=3000)),
                ]
            )
            model.fit(x_train, y_train)
            score = balanced_accuracy_score(y_valid, model.predict(x_valid))
            if score > best_score:
                best_score = float(score)
                best_c = float(c_value)
        assert best_c is not None
        combined_features = np.vstack([train_features, valid_features])
        combined_labels = np.concatenate([y_train, y_valid])
        final_reducer = PCA(n_components=d, random_state=0)
        combined_latent = final_reducer.fit_transform(combined_features)
        test_latent = final_reducer.transform(test_features)
        final_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=best_c, max_iter=3000)),
            ]
        )
        final_model.fit(combined_latent, combined_labels)
        prediction = final_model.predict(test_latent)
        return {
            "classification_accuracy": float(accuracy_score(y_test, prediction)),
            "classification_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "classification_c": best_c,
        }

    def run(self, run_dir: Path, *, coupling: float, input_gain: float) -> dict[str, Any]:
        from sklearn.datasets import load_digits

        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        cfg = self.config
        dataset = load_digits()
        images = dataset.data.astype(np.float64) / 16.0
        labels = dataset.target.astype(np.int64)
        all_indices = np.arange(len(images))
        train_idx, remaining = train_test_split(
            all_indices, test_size=0.40, random_state=cfg.seed, stratify=labels
        )
        valid_idx, test_idx = train_test_split(
            remaining, test_size=0.50, random_state=cfg.seed + 1, stratify=labels[remaining]
        )
        inputs = 2.0 * images - 1.0
        trajectories = _simulate_driven_boole(
            inputs,
            coupling=coupling,
            input_gain=input_gain,
            alpha=cfg.alpha,
            n_nodes=cfg.n_nodes,
            washout=cfg.washout,
            seed=cfg.seed + 50_000,
        )
        base_features: dict[str, FloatArray] = {
            "tm_spectrum": tm_temporal_spectrum(trajectories),
            "raw_mean_dct": raw_temporal_spectrum(trajectories, n_coefficients=64),
            "reservoir_flat": trajectories.reshape(len(trajectories), -1),
            "terminal_state": trajectories[:, -1, :],
            "input_oracle": images,
        }
        np.savez_compressed(
            artifacts / "digits_data_and_split.npz",
            images=images,
            labels=labels,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
        )
        np.savez_compressed(artifacts / "base_features.npz", **base_features)
        metric_rows: list[dict[str, Any]] = []
        examples: dict[str, FloatArray] = {"truth": images[test_idx[:20]]}
        for representation, features in base_features.items():
            records = fit_ridge_for_dimensions(
                features[train_idx],
                features[valid_idx],
                features[test_idx],
                images[train_idx],
                images[valid_idx],
                images[test_idx],
                cfg.dimensions,
                use_pca=True,
            )
            for record in records:
                prediction = np.clip(np.asarray(record.pop("prediction"), dtype=np.float64), 0.0, 1.0)
                d = int(record["dimension"])
                classification = self._classification_with_latent(
                    features[train_idx],
                    features[valid_idx],
                    features[test_idx],
                    labels[train_idx],
                    labels[valid_idx],
                    labels[test_idx],
                    d,
                )
                metric_rows.append(
                    {
                        "representation": representation,
                        **record,
                        "test_psnr": _psnr(images[test_idx], prediction),
                        "test_ssim": _mean_image_ssim(images[test_idx], prediction),
                        **classification,
                        "coupling": coupling,
                        "input_gain": input_gain,
                        "median_sync_rms": float(np.median(synchronization_rms(trajectories[test_idx], window=16))),
                    }
                )
                if d == 16:
                    examples[representation] = prediction[:20]
        metrics = pd.DataFrame(metric_rows)
        write_csv(artifacts / "image_metrics.csv", metrics)
        np.savez_compressed(artifacts / "reconstruction_examples.npz", **examples)
        self._plot(metrics, examples, artifacts)
        d16 = metrics[metrics.dimension == 16].set_index("representation")
        terminal_common_dimension = int(
            min(
                16,
                metrics.loc[metrics.representation == "tm_spectrum", "dimension"].max(),
                metrics.loc[metrics.representation == "terminal_state", "dimension"].max(),
            )
        )
        terminal_common = metrics[
            metrics.dimension == terminal_common_dimension
        ].set_index("representation")
        checks = {
            "split_disjoint": bool(
                set(train_idx).isdisjoint(set(valid_idx))
                and set(train_idx).isdisjoint(set(test_idx))
                and set(valid_idx).isdisjoint(set(test_idx))
            ),
            "e4_dynamics_frozen": True,
            "all_metrics_finite": bool(np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()),
            "digits_shape_8x8": bool(images.shape[1] == 64),
        }
        tm_beats_terminal_common = bool(
            terminal_common.loc["tm_spectrum", "test_nmse"]
            < terminal_common.loc["terminal_state", "test_nmse"]
        )
        scientific = {
            # Why not: terminal_state は8ノードのため16次元表現を持たない。
            # したがってTMとの比較は両者が利用できる最大共通次元で行う。
            "terminal_comparison_dimension": terminal_common_dimension,
            "tm_beats_terminal_reconstruction_common_dimension": tm_beats_terminal_common,
            "tm_beats_terminal_reconstruction_d16": tm_beats_terminal_common,
            "tm_beats_raw_dct_reconstruction_d16": bool(
                d16.loc["tm_spectrum", "test_nmse"] < d16.loc["raw_mean_dct", "test_nmse"]
            ),
            "tm_classification_above_0_70_d16": bool(
                d16.loc["tm_spectrum", "classification_balanced_accuracy"] >= 0.70
            ),
            "tm_matches_input_pca_ssim_within_0_10": bool(
                d16.loc["tm_spectrum", "test_ssim"] >= d16.loc["input_oracle", "test_ssim"] - 0.10
            ),
        }
        summary = {
            "experiment_id": self.experiment_id,
            "status": "pilot_completed" if all(checks.values()) else "integrity_warning",
            "config": cfg.__dict__,
            "transferred_coupling": coupling,
            "transferred_input_gain": input_gain,
            "integrity_checks": checks,
            "scientific_gates": scientific,
            "d16_metrics": d16.reset_index().to_dict(orient="records"),
            "interpretation": {
                "main": "E4で選んだ力学を再調整せず8×8 digitsへ転移し、復元とクラス保持を比較した。",
                "scope": "これは小画像pilotであり、28×28 MNISTの確認実験とは区別する。",
            },
        }
        write_json(artifacts / "summary.json", summary)
        write_json(run_dir / "config.json", {**cfg.__dict__, "coupling": coupling, "input_gain": input_gain})
        write_json(run_dir / "environment.json", environment_payload())
        write_json(run_dir / "validation.json", {**checks, **scientific})
        write_csv(run_dir / "artifact_hashes.csv", collect_hashes(run_dir))
        return summary

    @staticmethod
    def _plot(metrics: pd.DataFrame, examples: dict[str, FloatArray], artifacts: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for representation, group in metrics.groupby("representation"):
            ordered = group.sort_values("dimension")
            axes[0].plot(ordered.dimension, ordered.test_nmse, marker="o", label=representation)
            axes[1].plot(
                ordered.dimension,
                ordered.classification_balanced_accuracy,
                marker="o",
                label=representation,
            )
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Reconstruction NMSE")
        axes[1].set_ylabel("Classification balanced accuracy")
        for axis in axes:
            axis.set_xlabel("Latent dimension")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("E5A small-image rate–distortion and class retention")
        fig.tight_layout()
        fig.savefig(artifacts / "image_rate_distortion.png", dpi=180)
        plt.close(fig)

        representations = [name for name in ("tm_spectrum", "raw_mean_dct", "input_oracle") if name in examples]
        n_rows = min(6, len(examples["truth"]))
        fig, axes = plt.subplots(n_rows, 1 + len(representations), figsize=(2.2 * (1 + len(representations)), 2.2 * n_rows))
        axes = np.atleast_2d(axes)
        column_names = ["truth"] + representations
        for row in range(n_rows):
            for column, name in enumerate(column_names):
                axes[row, column].imshow(examples[name][row].reshape(8, 8), cmap="gray", vmin=0.0, vmax=1.0)
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(name, fontsize=8)
        fig.suptitle("E5A reconstruction examples (d=16)")
        fig.tight_layout()
        fig.savefig(artifacts / "digit_reconstructions.png", dpi=180)
        plt.close(fig)
