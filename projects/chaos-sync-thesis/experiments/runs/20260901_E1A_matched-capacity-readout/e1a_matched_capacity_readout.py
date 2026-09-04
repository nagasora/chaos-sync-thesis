"""TMとFourierを同じ16次元へ揃えたalpha読み出し追試を実行する。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
Coordinate = Tuple[int, int]
CandidateSpecs = Dict[str, Tuple[Coordinate, ...]]

PILOT_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "20260830_E1A_temporal-alpha-readout"
    / "e1a_temporal_alpha_readout.py"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """容量一致追試の分割、特徴選択、評価条件を保持する。"""

    fit_alphas: Tuple[float, ...]
    test_alphas: Tuple[float, ...]
    fit_seeds: Tuple[int, ...]
    test_seeds: Tuple[int, ...]
    burn_in: int
    observation_length: int
    candidate_specs: CandidateSpecs
    cv_folds: int
    selection_seed: int
    ridge_penalties: Tuple[float, ...]
    bootstrap_repetitions: int
    bootstrap_seed: int
    fourier_band_count: int = 8
    tm_full_max_order: int = 8
    tm_full_lags: Tuple[int, ...] = (1, 2, 4, 8)

    def validate(self) -> None:
        """未使用testと16次元容量一致が成立する設定か検証する。"""

        if not self.fit_alphas or not self.test_alphas:
            raise ValueError("fit_alphas と test_alphas は空にできません。")
        if set(self.fit_alphas) & set(self.test_alphas):
            raise ValueError("test alpha はfitに未使用でなければなりません。")
        if not (
            min(self.fit_alphas) < min(self.test_alphas)
            and max(self.test_alphas) < max(self.fit_alphas)
        ):
            raise ValueError("test alpha はfit範囲内の補間点にしてください。")
        if set(self.fit_seeds) & set(self.test_seeds):
            raise ValueError("fit seed と test seed は重複できません。")
        if self.cv_folds < 2 or self.cv_folds > len(self.fit_seeds):
            raise ValueError("cv_folds は2以上かつfit seed数以下にしてください。")
        if self.burn_in < 0 or self.observation_length <= 0:
            raise ValueError("burn_in は非負、observation_length は正にしてください。")
        if self.fourier_band_count * 2 != 16:
            raise ValueError("Fourier特徴は16次元へ固定してください。")
        if not self.candidate_specs:
            raise ValueError("TM候補を少なくとも一つ指定してください。")
        for name, coordinates in self.candidate_specs.items():
            if len(coordinates) != 8 or len(set(coordinates)) != 8:
                raise ValueError(f"{name} は重複なしの8複素座標にしてください。")
            if any(order <= 0 for order, _ in coordinates):
                raise ValueError(f"{name} のTM次数は正にしてください。")
            if any(lag < 0 or lag >= self.observation_length for _, lag in coordinates):
                raise ValueError(f"{name} の遅延は観測長未満の非負値にしてください。")
        if any(penalty <= 0.0 for penalty in self.ridge_penalties):
            raise ValueError("ridge_penalties は正にしてください。")
        if self.bootstrap_repetitions <= 0:
            raise ValueError("bootstrap_repetitions は正にしてください。")


@dataclass(frozen=True)
class FeatureDataset:
    """軌道識別子と回帰へ実際に使用する特徴行列を保持する。"""

    split: StringArray
    alpha: FloatArray
    seed: IntArray
    estimated_location: FloatArray
    estimated_scale: FloatArray
    features: Dict[str, FloatArray]


def load_pilot_module() -> ModuleType:
    """E0由来の検証済み軌道生成・前処理・ridge実装を読み込む。"""

    spec = importlib.util.spec_from_file_location("e1a_temporal_alpha_pilot", PILOT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"pilot実装を読み込めません: {PILOT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> ExperimentConfig:
    """実験前に固定したconfig.jsonから実行設定を構築する。"""

    record = json.loads(path.read_text(encoding="utf-8"))
    data = record["data"]
    training = record["training"]
    candidate_specs = {
        name: tuple((int(order), int(lag)) for order, lag in coordinates)
        for name, coordinates in record["tm_candidate_specs"].items()
    }
    config = ExperimentConfig(
        fit_alphas=tuple(float(value) for value in data["fit_alphas"]),
        test_alphas=tuple(float(value) for value in data["test_alphas"]),
        fit_seeds=tuple(int(value) for value in data["fit_seeds"]),
        test_seeds=tuple(int(value) for value in data["test_seeds"]),
        burn_in=int(data["burn_in"]),
        observation_length=int(data["observation_length"]),
        candidate_specs=candidate_specs,
        cv_folds=int(training["cv_folds"]),
        selection_seed=int(training["selection_seed"]),
        ridge_penalties=tuple(float(value) for value in training["ridge_penalties"]),
        bootstrap_repetitions=int(record["bootstrap_repetitions"]),
        bootstrap_seed=int(record["bootstrap_seed"]),
    )
    config.validate()
    return config


def tm_coordinate_features(
    cayley_values: NDArray[np.complex128],
    coordinates: Tuple[Coordinate, ...],
    pilot: ModuleType,
) -> FloatArray:
    """指定したTM次数・遅延の複素座標を実部・虚部へ展開する。"""

    values: List[complex] = []
    for order, lag in coordinates:
        mode = np.power(cayley_values, order)
        if lag == 0:
            values.append(complex(np.mean(mode)))
        else:
            values.append(complex(np.mean(mode[lag:] * np.conjugate(mode[:-lag]))))
    return pilot.complex_to_real_features(values)


def extract_feature_row(
    orbit: FloatArray,
    config: ExperimentConfig,
    pilot: ModuleType,
) -> Tuple[Dict[str, FloatArray], float, float]:
    """一つの軌道から全候補と比較対象を同時に抽出する。"""

    standardized, location, scale = pilot.robust_standardize(orbit)
    cayley_values = pilot.cayley_transform(standardized)
    features: Dict[str, FloatArray] = {
        "fourier_16": pilot.fourier_band_features(cayley_values, config.fourier_band_count),
        "tm_full_80": pilot.tm_delay_features(
            cayley_values,
            config.tm_full_max_order,
            config.tm_full_lags,
        ),
        "tm_instant_16": pilot.tm_instantaneous_features(cayley_values, 8),
    }
    for name, coordinates in config.candidate_specs.items():
        features[f"tm16_{name}"] = tm_coordinate_features(
            cayley_values,
            coordinates,
            pilot,
        )
    expected_dimensions = {"fourier_16": 16, "tm_full_80": 80, "tm_instant_16": 16}
    expected_dimensions.update({f"tm16_{name}": 16 for name in config.candidate_specs})
    for name, values in features.items():
        if values.shape != (expected_dimensions[name],):
            raise ValueError(f"{name} の次元が不正です: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} に非有限値が発生しました。")
    return features, float(location), float(scale)


def build_feature_dataset(config: ExperimentConfig, pilot: ModuleType) -> FeatureDataset:
    """configのalphaとseedから決定論的に軌道と特徴データを生成する。"""

    split_values: List[str] = []
    alpha_values: List[float] = []
    seed_values: List[int] = []
    location_values: List[float] = []
    scale_values: List[float] = []
    feature_rows: Dict[str, List[FloatArray]] = {}

    for split, alphas, seeds in (
        ("fit", config.fit_alphas, config.fit_seeds),
        ("test", config.test_alphas, config.test_seeds),
    ):
        for alpha in alphas:
            for seed in seeds:
                orbit = pilot.E0_MODULE.generate_orbit(
                    alpha,
                    seed,
                    config.burn_in,
                    config.observation_length,
                )
                features, location, scale = extract_feature_row(orbit, config, pilot)
                split_values.append(split)
                alpha_values.append(alpha)
                seed_values.append(seed)
                location_values.append(location)
                scale_values.append(scale)
                for name, values in features.items():
                    feature_rows.setdefault(name, []).append(values)

    return FeatureDataset(
        split=np.asarray(split_values, dtype=np.str_),
        alpha=np.asarray(alpha_values, dtype=np.float64),
        seed=np.asarray(seed_values, dtype=np.int64),
        estimated_location=np.asarray(location_values, dtype=np.float64),
        estimated_scale=np.asarray(scale_values, dtype=np.float64),
        features={name: np.stack(rows) for name, rows in feature_rows.items()},
    )


def save_feature_dataset(path: Path, dataset: FeatureDataset) -> None:
    """モデル入力を再利用可能な圧縮NPZとして保存する。"""

    arrays = {
        "split": dataset.split,
        "alpha": dataset.alpha,
        "seed": dataset.seed,
        "estimated_location": dataset.estimated_location,
        "estimated_scale": dataset.estimated_scale,
    }
    arrays.update({f"feature__{name}": values for name, values in dataset.features.items()})
    np.savez_compressed(path, **arrays)


def load_feature_dataset(path: Path) -> FeatureDataset:
    """保存NPZから軌道再生成なしで同じモデル入力を復元する。"""

    with np.load(path, allow_pickle=False) as archive:
        features = {
            name.removeprefix("feature__"): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith("feature__")
        }
        return FeatureDataset(
            split=np.asarray(archive["split"], dtype=np.str_),
            alpha=np.asarray(archive["alpha"], dtype=np.float64),
            seed=np.asarray(archive["seed"], dtype=np.int64),
            estimated_location=np.asarray(archive["estimated_location"], dtype=np.float64),
            estimated_scale=np.asarray(archive["estimated_scale"], dtype=np.float64),
            features=features,
        )


def validate_feature_dataset(dataset: FeatureDataset, config: ExperimentConfig) -> None:
    """保存または生成した特徴データが設定と一致するか検証する。"""

    expected_rows = (
        len(config.fit_alphas) * len(config.fit_seeds)
        + len(config.test_alphas) * len(config.test_seeds)
    )
    if dataset.alpha.size != expected_rows:
        raise ValueError(f"特徴データ行数が不正です: {dataset.alpha.size} != {expected_rows}")
    if any(values.shape[0] != expected_rows for values in dataset.features.values()):
        raise ValueError("特徴行列間で行数が一致しません。")
    if set(dataset.seed[dataset.split == "fit"].tolist()) != set(config.fit_seeds):
        raise ValueError("fit seedがconfigと一致しません。")
    if set(dataset.seed[dataset.split == "test"].tolist()) != set(config.test_seeds):
        raise ValueError("test seedがconfigと一致しません。")
    if any(not np.all(np.isfinite(values)) for values in dataset.features.values()):
        raise FloatingPointError("保存特徴に非有限値があります。")


def assign_grouped_folds(config: ExperimentConfig) -> Dict[int, int]:
    """同じseedの全alphaを同一foldへ置く決定論的割当を作る。"""

    seeds = np.asarray(config.fit_seeds, dtype=np.int64)
    shuffled = np.random.default_rng(config.selection_seed).permutation(seeds)
    return {int(seed): int(index % config.cv_folds) for index, seed in enumerate(shuffled)}


def grouped_cv_predictions(
    features: FloatArray,
    targets: FloatArray,
    seeds: IntArray,
    fold_by_seed: Dict[int, int],
    penalty: float,
    pilot: ModuleType,
) -> FloatArray:
    """seed-grouped foldごとに学習し、全fit行のout-of-fold予測を返す。"""

    predictions = np.empty_like(targets)
    fold_labels = np.asarray([fold_by_seed[int(seed)] for seed in seeds], dtype=np.int64)
    for fold in sorted(set(fold_by_seed.values())):
        validation_mask = fold_labels == fold
        train_mask = ~validation_mask
        predictions[validation_mask] = pilot.fit_ridge(
            features[train_mask],
            targets[train_mask],
            features[validation_mask],
            penalty,
        )
    return predictions


def select_penalty(
    feature_name: str,
    features: FloatArray,
    targets: FloatArray,
    seeds: IntArray,
    fold_by_seed: Dict[int, int],
    penalties: Tuple[float, ...],
    pilot: ModuleType,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """grouped CV RMSEだけでridge強度を選ぶ。"""

    rows: List[Dict[str, object]] = []
    for penalty in penalties:
        predictions = grouped_cv_predictions(
            features,
            targets,
            seeds,
            fold_by_seed,
            penalty,
            pilot,
        )
        rows.append(
            {
                "feature_name": feature_name,
                "feature_dimension": int(features.shape[1]),
                "penalty": float(penalty),
                "cv_rmse": pilot.rmse(targets, predictions),
            }
        )
    selected = min(rows, key=lambda row: (float(row["cv_rmse"]), float(row["penalty"])))
    for row in rows:
        row["selected"] = bool(row is selected)
    return float(selected["penalty"]), float(selected["cv_rmse"]), rows


def cluster_bootstrap_difference(
    targets: FloatArray,
    tm_predictions: FloatArray,
    fourier_predictions: FloatArray,
    seeds: IntArray,
    repetitions: int,
    bootstrap_seed: int,
    pilot: ModuleType,
) -> Tuple[Dict[str, float], FloatArray]:
    """seedをクラスタとしてFourier RMSE minus TM RMSEを再標本化する。"""

    unique_seeds = np.unique(seeds)
    indices_by_seed = {int(seed): np.flatnonzero(seeds == seed) for seed in unique_seeds}
    rng = np.random.default_rng(bootstrap_seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.choice(unique_seeds, size=unique_seeds.size, replace=True)
        row_indices = np.concatenate([indices_by_seed[int(seed)] for seed in sampled])
        differences[index] = pilot.rmse(
            targets[row_indices],
            fourier_predictions[row_indices],
        ) - pilot.rmse(targets[row_indices], tm_predictions[row_indices])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return (
        {
            "point_difference": pilot.rmse(targets, fourier_predictions)
            - pilot.rmse(targets, tm_predictions),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
        },
        differences,
    )


def file_sha256(path: Path) -> str:
    """再現対象コードのSHA-256を計算する。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(pilot: ModuleType, path: Path, rows: List[Dict[str, object]]) -> None:
    """空でない辞書行を既存E1Aと同じCSV形式で保存する。"""

    if not rows:
        raise ValueError(f"空のCSVは保存できません: {path}")
    pilot.write_csv(path, list(rows[0].keys()), rows)


def create_result_plots(
    model_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    bootstrap_differences: FloatArray,
    comparison: Dict[str, float],
    output_directory: Path,
) -> None:
    """model比較・不確実性・TM候補選択を静的PNGへ保存する。"""

    colors = {
        "tm_selected_16": "#2F5D8A",
        "fourier_16": "#C7922B",
        "tm_full_80": "#737A82",
        "tm_instant_16": "#AAB0B6",
    }
    labels = {
        "tm_selected_16": "Selected TM (16D)",
        "fourier_16": "Fourier (16D)",
        "tm_full_80": "Full TM (80D)",
        "tm_instant_16": "Instant TM (16D)",
    }
    by_model = {str(row["model"]): row for row in model_rows}
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    main_names = ["tm_selected_16", "fourier_16", "tm_full_80"]
    main_values = [float(by_model[name]["test_rmse"]) for name in main_names]
    bars = axes[0].barh(
        [labels[name] for name in main_names],
        main_values,
        color=[colors[name] for name in main_names],
    )
    axes[0].set_title("Test RMSE: temporal features")
    axes[0].set_xlabel("RMSE")
    axes[0].set_xlim(0.0, max(main_values) * 1.28)
    axes[0].bar_label(bars, fmt="%.6f", padding=3, fontsize=8)

    control_names = ["tm_selected_16", "tm_instant_16"]
    control_values = [float(by_model[name]["test_rmse"]) for name in control_names]
    bars = axes[1].barh(
        [labels[name] for name in control_names],
        control_values,
        color=[colors[name] for name in control_names],
    )
    axes[1].set_title("Delay ablation")
    axes[1].set_xlabel("RMSE")
    axes[1].set_xlim(0.0, max(control_values) * 1.20)
    axes[1].bar_label(bars, fmt="%.6f", padding=3, fontsize=8)

    axes[2].hist(bootstrap_differences, bins=36, color="#2F5D8A", alpha=0.82)
    axes[2].axvline(
        0.0,
        color="#34383D",
        linestyle="--",
        linewidth=1.2,
        label="no difference",
    )
    axes[2].axvline(
        comparison["ci_95_lower"],
        color="#C7922B",
        linestyle=":",
        linewidth=1.4,
    )
    axes[2].axvline(
        comparison["ci_95_upper"],
        color="#C7922B",
        linestyle=":",
        linewidth=1.4,
    )
    axes[2].set_title("Seed-cluster bootstrap")
    axes[2].set_xlabel("Fourier RMSE - TM RMSE")
    axes[2].set_ylabel("Replicates")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="#E3E6E8", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle("E1A matched-capacity readout", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(
        output_directory / "matched_capacity_results.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    best_by_candidate: Dict[str, Dict[str, object]] = {}
    for row in candidate_rows:
        name = str(row["candidate"])
        if (
            name not in best_by_candidate
            or float(row["cv_rmse"]) < float(best_by_candidate[name]["cv_rmse"])
        ):
            best_by_candidate[name] = row
    ordered = sorted(
        best_by_candidate.values(),
        key=lambda row: float(row["cv_rmse"]),
        reverse=True,
    )
    selected_name = next(
        str(row["candidate"]) for row in candidate_rows if bool(row["selected"])
    )
    candidate_labels = [str(row["candidate"]) for row in ordered]
    candidate_values = [float(row["cv_rmse"]) for row in ordered]
    candidate_colors = [
        "#2F5D8A" if name == selected_name else "#AAB0B6"
        for name in candidate_labels
    ]
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    bars = axis.barh(candidate_labels, candidate_values, color=candidate_colors)
    axis.set_title("TM 16D candidate selection on grouped CV")
    axis.set_xlabel("Out-of-fold RMSE")
    axis.set_xlim(0.0, max(candidate_values) * 1.20)
    axis.bar_label(bars, fmt="%.6f", padding=3, fontsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#E3E6E8", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(
        output_directory / "tm_candidate_selection.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_experiment(
    run_directory: Path,
    config: ExperimentConfig,
    reuse_features_path: Path | None = None,
) -> Dict[str, object]:
    """特徴生成、grouped CV、凍結test、保存、プロットを一括実行する。"""

    pilot = load_pilot_module()
    artifacts_directory = run_directory / "artifacts"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    dataset = (
        load_feature_dataset(reuse_features_path)
        if reuse_features_path is not None
        else build_feature_dataset(config, pilot)
    )
    validate_feature_dataset(dataset, config)
    feature_store_path = artifacts_directory / "features.npz"
    save_feature_dataset(feature_store_path, dataset)

    manifest_rows = [
        {
            "row_id": index,
            "split": str(dataset.split[index]),
            "alpha": float(dataset.alpha[index]),
            "seed": int(dataset.seed[index]),
            "estimated_location": float(dataset.estimated_location[index]),
            "estimated_scale": float(dataset.estimated_scale[index]),
        }
        for index in range(dataset.alpha.size)
    ]
    write_csv(pilot, artifacts_directory / "trajectory_manifest.csv", manifest_rows)

    fit_mask = dataset.split == "fit"
    test_mask = dataset.split == "test"
    fit_targets = dataset.alpha[fit_mask]
    fit_seeds = dataset.seed[fit_mask]
    test_targets = dataset.alpha[test_mask]
    test_seeds = dataset.seed[test_mask]
    fold_by_seed = assign_grouped_folds(config)
    fold_rows = [
        {"seed": seed, "fold": fold_by_seed[seed]}
        for seed in sorted(fold_by_seed)
    ]
    write_csv(pilot, artifacts_directory / "fold_assignments.csv", fold_rows)

    candidate_rows: List[Dict[str, object]] = []
    for candidate in config.candidate_specs:
        feature_name = f"tm16_{candidate}"
        _, _, rows = select_penalty(
            feature_name,
            dataset.features[feature_name][fit_mask],
            fit_targets,
            fit_seeds,
            fold_by_seed,
            config.ridge_penalties,
            pilot,
        )
        for row in rows:
            row["candidate"] = candidate
            row["selected"] = False
            candidate_rows.append(row)
    selected_candidate_row = min(
        candidate_rows,
        key=lambda row: (
            float(row["cv_rmse"]),
            str(row["candidate"]),
            float(row["penalty"]),
        ),
    )
    selected_candidate_row["selected"] = True
    selected_candidate = str(selected_candidate_row["candidate"])
    selected_feature_name = str(selected_candidate_row["feature_name"])
    write_csv(pilot, artifacts_directory / "candidate_cv_metrics.csv", candidate_rows)

    model_feature_names = {
        "tm_selected_16": selected_feature_name,
        "fourier_16": "fourier_16",
        "tm_full_80": "tm_full_80",
        "tm_instant_16": "tm_instant_16",
    }
    model_penalty_rows: List[Dict[str, object]] = []
    selected_model_settings: Dict[str, Tuple[float, float]] = {}
    for model, feature_name in model_feature_names.items():
        if model == "tm_selected_16":
            selected_model_settings[model] = (
                float(selected_candidate_row["penalty"]),
                float(selected_candidate_row["cv_rmse"]),
            )
            continue
        penalty, cv_rmse, rows = select_penalty(
            feature_name,
            dataset.features[feature_name][fit_mask],
            fit_targets,
            fit_seeds,
            fold_by_seed,
            config.ridge_penalties,
            pilot,
        )
        selected_model_settings[model] = (penalty, cv_rmse)
        for row in rows:
            row["model"] = model
            model_penalty_rows.append(row)
    write_csv(
        pilot,
        artifacts_directory / "model_penalty_cv_metrics.csv",
        model_penalty_rows,
    )

    predictions_by_model: Dict[str, FloatArray] = {}
    model_rows: List[Dict[str, object]] = []
    for model, feature_name in model_feature_names.items():
        feature_matrix = dataset.features[feature_name]
        penalty, cv_rmse = selected_model_settings[model]
        test_predictions = pilot.fit_ridge(
            feature_matrix[fit_mask],
            fit_targets,
            feature_matrix[test_mask],
            penalty,
        )
        fit_predictions = pilot.fit_ridge(
            feature_matrix[fit_mask],
            fit_targets,
            feature_matrix[fit_mask],
            penalty,
        )
        predictions_by_model[model] = test_predictions
        model_rows.append(
            {
                "model": model,
                "feature_name": feature_name,
                "feature_dimension": int(feature_matrix.shape[1]),
                "selected_penalty": penalty,
                "cv_rmse": cv_rmse,
                "fit_rmse": pilot.rmse(fit_targets, fit_predictions),
                "test_rmse": pilot.rmse(test_targets, test_predictions),
            }
        )
    write_csv(pilot, artifacts_directory / "model_metrics.csv", model_rows)

    prediction_rows: List[Dict[str, object]] = []
    test_indices = np.flatnonzero(test_mask)
    for local_index, row_index in enumerate(test_indices):
        row: Dict[str, object] = {
            "alpha": float(dataset.alpha[row_index]),
            "seed": int(dataset.seed[row_index]),
            "estimated_location": float(dataset.estimated_location[row_index]),
            "estimated_scale": float(dataset.estimated_scale[row_index]),
        }
        for model, predictions in predictions_by_model.items():
            row[f"prediction_{model}"] = float(predictions[local_index])
        prediction_rows.append(row)
    write_csv(pilot, artifacts_directory / "test_predictions.csv", prediction_rows)

    alpha_rows: List[Dict[str, object]] = []
    for alpha in config.test_alphas:
        alpha_mask = test_targets == alpha
        for model, predictions in predictions_by_model.items():
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "model": model,
                    "row_count": int(np.sum(alpha_mask)),
                    "rmse": pilot.rmse(test_targets[alpha_mask], predictions[alpha_mask]),
                }
            )
    write_csv(pilot, artifacts_directory / "alpha_metrics.csv", alpha_rows)

    seed_rows: List[Dict[str, object]] = []
    for seed in sorted(set(test_seeds.tolist())):
        seed_mask = test_seeds == seed
        tm_rmse = pilot.rmse(
            test_targets[seed_mask],
            predictions_by_model["tm_selected_16"][seed_mask],
        )
        fourier_rmse = pilot.rmse(
            test_targets[seed_mask],
            predictions_by_model["fourier_16"][seed_mask],
        )
        seed_rows.append(
            {
                "seed": seed,
                "alpha_count": int(np.sum(seed_mask)),
                "tm_selected_16_rmse": tm_rmse,
                "fourier_16_rmse": fourier_rmse,
                "fourier_minus_tm_rmse": fourier_rmse - tm_rmse,
            }
        )
    write_csv(pilot, artifacts_directory / "seed_metrics.csv", seed_rows)

    comparison, bootstrap_differences = cluster_bootstrap_difference(
        test_targets,
        predictions_by_model["tm_selected_16"],
        predictions_by_model["fourier_16"],
        test_seeds,
        config.bootstrap_repetitions,
        config.bootstrap_seed,
        pilot,
    )
    bootstrap_rows = [
        {"replicate": index, "fourier_minus_tm_rmse": float(value)}
        for index, value in enumerate(bootstrap_differences)
    ]
    write_csv(
        pilot,
        artifacts_directory / "bootstrap_differences.csv",
        bootstrap_rows,
    )
    create_result_plots(
        model_rows,
        candidate_rows,
        bootstrap_differences,
        comparison,
        artifacts_directory,
    )

    success = bool(comparison["ci_95_lower"] > 0.0)
    summary: Dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "question": "TMとFourierを16次元へ揃えても未使用alpha回収の性能差が残るか。",
        "config": asdict(config),
        "sample_counts": {
            "fit": int(np.sum(fit_mask)),
            "test": int(np.sum(test_mask)),
        },
        "selected_tm_candidate": selected_candidate,
        "selected_tm_coordinates": [
            list(value) for value in config.candidate_specs[selected_candidate]
        ],
        "per_model": model_rows,
        "fourier_minus_tm_bootstrap": comparison,
        "success": success,
        "artifacts": {
            "feature_store": "artifacts/features.npz",
            "trajectory_manifest": "artifacts/trajectory_manifest.csv",
            "test_predictions": "artifacts/test_predictions.csv",
            "model_metrics": "artifacts/model_metrics.csv",
            "candidate_cv_metrics": "artifacts/candidate_cv_metrics.csv",
            "alpha_metrics": "artifacts/alpha_metrics.csv",
            "seed_metrics": "artifacts/seed_metrics.csv",
            "bootstrap_differences": "artifacts/bootstrap_differences.csv",
            "plots": [
                "artifacts/matched_capacity_results.png",
                "artifacts/tm_candidate_selection.png",
            ],
        },
        "caveats": [
            "単一源alpha回収であり、同期、源分離、圧縮・復号を検証していない。",
            "test alphaはfit範囲内の未使用補間点であり、外挿ではない。",
            "TM候補選択の自由度はFourierの固定帯域より大きいため、成功しても基底固有の普遍的優位を意味しない。",
            "軌道ごとの尺度推定誤差が有限時間情報へ残る可能性を排除していない。",
        ],
    }
    (artifacts_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    model_by_name = {str(row["model"]): row for row in model_rows}
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "primary": {
            "success": success,
            "selected_tm_candidate": selected_candidate,
            "tm_selected_16_test_rmse": model_by_name["tm_selected_16"]["test_rmse"],
            "fourier_16_test_rmse": model_by_name["fourier_16"]["test_rmse"],
            "fourier_minus_tm_ci_95": [
                comparison["ci_95_lower"],
                comparison["ci_95_upper"],
            ],
        },
        "secondary": {"model_metrics": model_rows},
        "artifacts": summary["artifacts"],
        "notes": summary["caveats"],
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config_path = run_directory / "config.json"
    config_record = json.loads(config_path.read_text(encoding="utf-8"))
    config_record["status"] = "completed"
    config_record["actual_run_config"] = asdict(config)
    config_path.write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    validator_path = script_path.with_name("validate_results.py")
    environment_path = run_directory / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment.update(
        {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": [
                f"numpy=={np.__version__}",
                f"matplotlib=={matplotlib.__version__}",
            ],
            "execution_command": "python e1a_matched_capacity_readout.py --run-directory .",
            "reuse_command": (
                "python e1a_matched_capacity_readout.py --run-directory . "
                "--reuse-features artifacts/features.npz"
            ),
            "reused_feature_store": str(reuse_features_path) if reuse_features_path else None,
            "code_sha256": {
                script_path.name: file_sha256(script_path),
                validator_path.name: file_sha256(validator_path),
                PILOT_SCRIPT_PATH.name: file_sha256(PILOT_SCRIPT_PATH),
            },
            "data_provenance": {
                "generator": "F_alpha(x)=alpha*(x-1/x)",
                "trajectory_identifiers": "artifacts/trajectory_manifest.csv",
                "model_input_store": "artifacts/features.npz",
            },
            "artifact_sha256": {
                name: file_sha256(artifacts_directory / name)
                for name in (
                    "features.npz",
                    "trajectory_manifest.csv",
                    "fold_assignments.csv",
                    "candidate_cv_metrics.csv",
                    "model_penalty_cv_metrics.csv",
                    "model_metrics.csv",
                    "test_predictions.csv",
                    "alpha_metrics.csv",
                    "seed_metrics.csv",
                    "bootstrap_differences.csv",
                    "matched_capacity_results.png",
                    "tm_candidate_selection.png",
                )
            },
        }
    )
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_arguments(arguments: List[str]) -> argparse.Namespace:
    """実験runと任意の再利用特徴ストアをCLIから受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path.cwd())
    parser.add_argument("--reuse-features", type=Path)
    return parser.parse_args(arguments)


def main(arguments: List[str] | None = None) -> int:
    """固定configで追試を実行し、主要結果を標準出力へ返す。"""

    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    run_directory = args.run_directory.resolve()
    config = load_config(run_directory / "config.json")
    reuse_path = args.reuse_features.resolve() if args.reuse_features else None
    summary = run_experiment(run_directory, config, reuse_path)
    print(
        json.dumps(
            {
                "success": summary["success"],
                "selected_tm_candidate": summary["selected_tm_candidate"],
                "comparison": summary["fourier_minus_tm_bootstrap"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
