"""2源full-rank観測と対称1観測の識別可能性対照を実行する。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
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

MATCHED_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "20260901_E1A_matched-capacity-readout"
    / "e1a_matched_capacity_readout.py"
)
MODEL_ORDER = (
    "oracle_tm_full_rank",
    "oracle_fourier_full_rank",
    "direct_tm_full_rank",
    "direct_fourier_full_rank",
    "direct_tm_rank_one",
    "direct_fourier_rank_one",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """2源識別可能性実験の生成、特徴、評価条件を保持する。"""

    fit_alphas: Tuple[float, ...]
    test_alphas: Tuple[float, ...]
    fit_pair_seeds: Tuple[int, ...]
    test_pair_seeds: Tuple[int, ...]
    right_source_seed_offset: int
    burn_in: int
    observation_length: int
    mixing_matrix: FloatArray
    tm_coordinates: Tuple[Coordinate, ...]
    fourier_band_count: int
    cv_folds: int
    selection_seed: int
    ridge_penalties: Tuple[float, ...]
    oracle_tm_rmse_threshold: float
    collision_tolerance: float

    def validate(self) -> None:
        """事前登録した可逆性、分割、特徴次元を検証する。"""

        if len(self.fit_alphas) < 2 or len(self.test_alphas) < 2:
            raise ValueError("fit/test alphaはそれぞれ2値以上必要です。")
        if set(self.fit_alphas) & set(self.test_alphas):
            raise ValueError("test alphaはfitに未使用でなければなりません。")
        if not (
            min(self.fit_alphas) < min(self.test_alphas)
            and max(self.test_alphas) < max(self.fit_alphas)
        ):
            raise ValueError("test alphaはfit範囲内の補間点にしてください。")
        if set(self.fit_pair_seeds) & set(self.test_pair_seeds):
            raise ValueError("fit/test pair seedは重複できません。")
        if self.right_source_seed_offset <= 0:
            raise ValueError("right_source_seed_offsetは正にしてください。")
        if self.burn_in < 0 or self.observation_length <= 1:
            raise ValueError("burn_inは非負、observation_lengthは2以上にしてください。")
        if self.mixing_matrix.shape != (2, 2):
            raise ValueError("mixing_matrixは2x2にしてください。")
        if abs(float(np.linalg.det(self.mixing_matrix))) <= 1.0e-12:
            raise ValueError("full-rank観測行列は可逆でなければなりません。")
        if len(self.tm_coordinates) != 8 or len(set(self.tm_coordinates)) != 8:
            raise ValueError("TM特徴は重複なし8複素座標=16実次元にしてください。")
        if any(order <= 0 or lag < 0 for order, lag in self.tm_coordinates):
            raise ValueError("TM次数は正、遅延は非負にしてください。")
        if max(lag for _, lag in self.tm_coordinates) >= self.observation_length:
            raise ValueError("TM遅延は観測長未満にしてください。")
        if self.fourier_band_count != 8:
            raise ValueError("Fourier特徴は16次元へ固定してください。")
        if self.cv_folds < 2 or self.cv_folds > len(self.fit_pair_seeds):
            raise ValueError("cv_foldsは2以上かつfit pair seed数以下にしてください。")
        if any(value <= 0.0 for value in self.ridge_penalties):
            raise ValueError("ridge_penaltiesは正にしてください。")
        if self.oracle_tm_rmse_threshold <= 0.0 or self.collision_tolerance < 0.0:
            raise ValueError("成功閾値が不正です。")


@dataclass(frozen=True)
class SourceDataset:
    """実験に使用した標準化済み2源軌道と由来を保持する。"""

    split: StringArray
    pair_seed: IntArray
    alpha_low: FloatArray
    alpha_high: FloatArray
    left_seed: IntArray
    right_seed: IntArray
    low_location: FloatArray
    low_scale: FloatArray
    high_location: FloatArray
    high_scale: FloatArray
    source_low: FloatArray
    source_high: FloatArray


@dataclass(frozen=True)
class FeatureDataset:
    """衝突対の順序付き目標と全モデル入力を保持する。"""

    split: StringArray
    collision_id: IntArray
    pair_seed: IntArray
    order_index: IntArray
    alpha1: FloatArray
    alpha2: FloatArray
    source_seed1: IntArray
    source_seed2: IntArray
    features: Dict[str, FloatArray]


def load_matched_module() -> ModuleType:
    """E1Aで検証した特徴抽出と単一源実装を読み込む。"""

    spec = importlib.util.spec_from_file_location("e1a_matched_capacity", MATCHED_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"E1A実装を読み込めません: {MATCHED_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> ExperimentConfig:
    """事前登録config.jsonから実行設定を構築する。"""

    record = json.loads(path.read_text(encoding="utf-8"))
    data = record["data"]
    features = record["features"]
    training = record["training"]
    criteria = record["success_criteria"]
    threshold_text = next(value for value in criteria if value.startswith("oracle_tm"))
    tolerance_text = next(value for value in criteria if "collision_feature" in value)
    config = ExperimentConfig(
        fit_alphas=tuple(float(value) for value in data["fit_alphas"]),
        test_alphas=tuple(float(value) for value in data["test_alphas"]),
        fit_pair_seeds=tuple(
            range(
                int(data["fit_pair_seed_start"]),
                int(data["fit_pair_seed_start"]) + int(data["fit_pair_seed_count"]),
            )
        ),
        test_pair_seeds=tuple(
            range(
                int(data["test_pair_seed_start"]),
                int(data["test_pair_seed_start"]) + int(data["test_pair_seed_count"]),
            )
        ),
        right_source_seed_offset=int(data["right_source_seed_offset"]),
        burn_in=int(data["burn_in"]),
        observation_length=int(data["observation_length"]),
        mixing_matrix=np.asarray(
            record["theory"]["full_rank_observation_matrix"],
            dtype=np.float64,
        ),
        tm_coordinates=tuple(
            (int(order), int(lag))
            for order, lag in features["frozen_tm_coordinates"]
        ),
        fourier_band_count=int(features["fourier_band_count"]),
        cv_folds=int(training["cv_folds"]),
        selection_seed=int(training["selection_seed"]),
        ridge_penalties=tuple(float(value) for value in training["ridge_penalties"]),
        oracle_tm_rmse_threshold=float(threshold_text.rsplit(" ", 1)[-1]),
        collision_tolerance=float(tolerance_text.rsplit(" ", 1)[-1]),
    )
    config.validate()
    return config


def alpha_pairs(alphas: Tuple[float, ...]) -> Tuple[Tuple[float, float], ...]:
    """昇順alphaから異なる値のunordered pairを固定順で返す。"""

    return tuple((float(low), float(high)) for low, high in combinations(sorted(alphas), 2))


def build_source_dataset(
    config: ExperimentConfig,
    pilot: ModuleType,
) -> SourceDataset:
    """alphaとseedから実使用の標準化済み源軌道を生成する。"""

    rows: Dict[str, List[object]] = {
        name: []
        for name in (
            "split",
            "pair_seed",
            "alpha_low",
            "alpha_high",
            "left_seed",
            "right_seed",
            "low_location",
            "low_scale",
            "high_location",
            "high_scale",
            "source_low",
            "source_high",
        )
    }
    orbit_cache: Dict[Tuple[float, int], Tuple[FloatArray, float, float]] = {}

    def standardized_orbit(alpha: float, seed: int) -> Tuple[FloatArray, float, float]:
        key = (alpha, seed)
        if key not in orbit_cache:
            orbit = pilot.E0_MODULE.generate_orbit(
                alpha,
                seed,
                config.burn_in,
                config.observation_length,
            )
            standardized, location, scale = pilot.robust_standardize(orbit)
            orbit_cache[key] = (
                np.asarray(standardized, dtype=np.float64),
                float(location),
                float(scale),
            )
        return orbit_cache[key]

    for split, alphas, pair_seeds in (
        ("fit", config.fit_alphas, config.fit_pair_seeds),
        ("test", config.test_alphas, config.test_pair_seeds),
    ):
        for pair_seed in pair_seeds:
            left_seed = int(pair_seed)
            right_seed = int(pair_seed + config.right_source_seed_offset)
            for low, high in alpha_pairs(alphas):
                low_orbit, low_location, low_scale = standardized_orbit(low, left_seed)
                high_orbit, high_location, high_scale = standardized_orbit(high, right_seed)
                values = {
                    "split": split,
                    "pair_seed": pair_seed,
                    "alpha_low": low,
                    "alpha_high": high,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "low_location": low_location,
                    "low_scale": low_scale,
                    "high_location": high_location,
                    "high_scale": high_scale,
                    "source_low": low_orbit,
                    "source_high": high_orbit,
                }
                for name, value in values.items():
                    rows[name].append(value)
    return SourceDataset(
        split=np.asarray(rows["split"], dtype=np.str_),
        pair_seed=np.asarray(rows["pair_seed"], dtype=np.int64),
        alpha_low=np.asarray(rows["alpha_low"], dtype=np.float64),
        alpha_high=np.asarray(rows["alpha_high"], dtype=np.float64),
        left_seed=np.asarray(rows["left_seed"], dtype=np.int64),
        right_seed=np.asarray(rows["right_seed"], dtype=np.int64),
        low_location=np.asarray(rows["low_location"], dtype=np.float64),
        low_scale=np.asarray(rows["low_scale"], dtype=np.float64),
        high_location=np.asarray(rows["high_location"], dtype=np.float64),
        high_scale=np.asarray(rows["high_scale"], dtype=np.float64),
        source_low=np.stack(rows["source_low"]).astype(np.float64),
        source_high=np.stack(rows["source_high"]).astype(np.float64),
    )


def save_source_dataset(path: Path, dataset: SourceDataset) -> None:
    """実使用の源軌道と由来を圧縮NPZへ保存する。"""

    np.savez_compressed(path, **asdict(dataset))


def load_source_dataset(path: Path) -> SourceDataset:
    """保存NPZから源軌道を復元する。"""

    with np.load(path, allow_pickle=False) as archive:
        return SourceDataset(
            split=np.asarray(archive["split"], dtype=np.str_),
            pair_seed=np.asarray(archive["pair_seed"], dtype=np.int64),
            alpha_low=np.asarray(archive["alpha_low"], dtype=np.float64),
            alpha_high=np.asarray(archive["alpha_high"], dtype=np.float64),
            left_seed=np.asarray(archive["left_seed"], dtype=np.int64),
            right_seed=np.asarray(archive["right_seed"], dtype=np.int64),
            low_location=np.asarray(archive["low_location"], dtype=np.float64),
            low_scale=np.asarray(archive["low_scale"], dtype=np.float64),
            high_location=np.asarray(archive["high_location"], dtype=np.float64),
            high_scale=np.asarray(archive["high_scale"], dtype=np.float64),
            source_low=np.asarray(archive["source_low"], dtype=np.float64),
            source_high=np.asarray(archive["source_high"], dtype=np.float64),
        )


def validate_source_dataset(dataset: SourceDataset, config: ExperimentConfig) -> None:
    """保存または生成した源データがconfigと一致するか検証する。"""

    expected_fit = len(config.fit_pair_seeds) * len(alpha_pairs(config.fit_alphas))
    expected_test = len(config.test_pair_seeds) * len(alpha_pairs(config.test_alphas))
    expected_rows = expected_fit + expected_test
    if dataset.split.size != expected_rows:
        raise ValueError(f"源データ行数が不正です: {dataset.split.size} != {expected_rows}")
    if dataset.source_low.shape != (expected_rows, config.observation_length):
        raise ValueError("source_lowの形状が不正です。")
    if dataset.source_high.shape != dataset.source_low.shape:
        raise ValueError("source_highの形状が不正です。")
    if int(np.sum(dataset.split == "fit")) != expected_fit:
        raise ValueError("fit源データ行数が不正です。")
    if int(np.sum(dataset.split == "test")) != expected_test:
        raise ValueError("test源データ行数が不正です。")
    if set(dataset.pair_seed[dataset.split == "fit"].tolist()) != set(config.fit_pair_seeds):
        raise ValueError("fit pair seedがconfigと一致しません。")
    if set(dataset.pair_seed[dataset.split == "test"].tolist()) != set(config.test_pair_seeds):
        raise ValueError("test pair seedがconfigと一致しません。")
    numeric_arrays = (
        dataset.alpha_low,
        dataset.alpha_high,
        dataset.low_location,
        dataset.low_scale,
        dataset.high_location,
        dataset.high_scale,
        dataset.source_low,
        dataset.source_high,
    )
    if any(not np.all(np.isfinite(values)) for values in numeric_arrays):
        raise FloatingPointError("源データに非有限値があります。")
    if np.any(dataset.low_scale <= 0.0) or np.any(dataset.high_scale <= 0.0):
        raise ValueError("源軌道のrobust尺度は正でなければなりません。")


def extract_channel_features(
    signal: FloatArray,
    config: ExperimentConfig,
    pilot: ModuleType,
    matched: ModuleType,
) -> Tuple[FloatArray, FloatArray]:
    """1観測チャネルから凍結TM16とFourier16を抽出する。"""

    standardized, _, _ = pilot.robust_standardize(signal)
    cayley = pilot.cayley_transform(standardized)
    tm_values = matched.tm_coordinate_features(cayley, config.tm_coordinates, pilot)
    fourier_values = pilot.fourier_band_features(cayley, config.fourier_band_count)
    if tm_values.shape != (16,) or fourier_values.shape != (16,):
        raise ValueError("1チャネル特徴はTM/Fourierとも16次元でなければなりません。")
    if not np.all(np.isfinite(tm_values)) or not np.all(np.isfinite(fourier_values)):
        raise FloatingPointError("特徴に非有限値があります。")
    return (
        np.asarray(tm_values, dtype=np.float64),
        np.asarray(fourier_values, dtype=np.float64),
    )


def build_feature_dataset(
    source: SourceDataset,
    config: ExperimentConfig,
    pilot: ModuleType,
    matched: ModuleType,
) -> FeatureDataset:
    """2源からfull-rankとrank-one観測を作り全モデル入力を抽出する。"""

    inverse_matrix = np.linalg.inv(config.mixing_matrix)
    metadata: Dict[str, List[object]] = {
        name: []
        for name in (
            "split",
            "collision_id",
            "pair_seed",
            "order_index",
            "alpha1",
            "alpha2",
            "source_seed1",
            "source_seed2",
        )
    }
    feature_rows: Dict[str, List[FloatArray]] = {name: [] for name in MODEL_ORDER}

    for collision_id in range(source.split.size):
        low = source.source_low[collision_id]
        high = source.source_high[collision_id]
        rank_one_signal = (low + high) / np.sqrt(2.0)
        rank_tm, rank_fourier = extract_channel_features(
            rank_one_signal,
            config,
            pilot,
            matched,
        )
        order_values = (
            (
                low,
                high,
                float(source.alpha_low[collision_id]),
                float(source.alpha_high[collision_id]),
                int(source.left_seed[collision_id]),
                int(source.right_seed[collision_id]),
            ),
            (
                high,
                low,
                float(source.alpha_high[collision_id]),
                float(source.alpha_low[collision_id]),
                int(source.right_seed[collision_id]),
                int(source.left_seed[collision_id]),
            ),
        )
        for order_index, (signal1, signal2, alpha1, alpha2, seed1, seed2) in enumerate(
            order_values
        ):
            sources = np.vstack((signal1, signal2))
            observations = config.mixing_matrix @ sources
            recovered = inverse_matrix @ observations
            oracle1_tm, oracle1_fourier = extract_channel_features(
                recovered[0],
                config,
                pilot,
                matched,
            )
            oracle2_tm, oracle2_fourier = extract_channel_features(
                recovered[1],
                config,
                pilot,
                matched,
            )
            direct1_tm, direct1_fourier = extract_channel_features(
                observations[0],
                config,
                pilot,
                matched,
            )
            direct2_tm, direct2_fourier = extract_channel_features(
                observations[1],
                config,
                pilot,
                matched,
            )
            values = {
                "split": str(source.split[collision_id]),
                "collision_id": collision_id,
                "pair_seed": int(source.pair_seed[collision_id]),
                "order_index": order_index,
                "alpha1": alpha1,
                "alpha2": alpha2,
                "source_seed1": seed1,
                "source_seed2": seed2,
            }
            for name, value in values.items():
                metadata[name].append(value)
            feature_rows["oracle_tm_full_rank"].append(
                np.concatenate((oracle1_tm, oracle2_tm))
            )
            feature_rows["oracle_fourier_full_rank"].append(
                np.concatenate((oracle1_fourier, oracle2_fourier))
            )
            feature_rows["direct_tm_full_rank"].append(
                np.concatenate((direct1_tm, direct2_tm))
            )
            feature_rows["direct_fourier_full_rank"].append(
                np.concatenate((direct1_fourier, direct2_fourier))
            )
            # Why not: swapped行で再抽出すると丸め差が衝突検証へ混入するため、
            # 数学的に同一なrank-one観測には同じ保存特徴を明示的に再利用する。
            feature_rows["direct_tm_rank_one"].append(rank_tm.copy())
            feature_rows["direct_fourier_rank_one"].append(rank_fourier.copy())

    return FeatureDataset(
        split=np.asarray(metadata["split"], dtype=np.str_),
        collision_id=np.asarray(metadata["collision_id"], dtype=np.int64),
        pair_seed=np.asarray(metadata["pair_seed"], dtype=np.int64),
        order_index=np.asarray(metadata["order_index"], dtype=np.int64),
        alpha1=np.asarray(metadata["alpha1"], dtype=np.float64),
        alpha2=np.asarray(metadata["alpha2"], dtype=np.float64),
        source_seed1=np.asarray(metadata["source_seed1"], dtype=np.int64),
        source_seed2=np.asarray(metadata["source_seed2"], dtype=np.int64),
        features={name: np.stack(rows) for name, rows in feature_rows.items()},
    )


def save_feature_dataset(path: Path, dataset: FeatureDataset) -> None:
    """全モデル入力を再利用可能な圧縮NPZへ保存する。"""

    arrays = {
        "split": dataset.split,
        "collision_id": dataset.collision_id,
        "pair_seed": dataset.pair_seed,
        "order_index": dataset.order_index,
        "alpha1": dataset.alpha1,
        "alpha2": dataset.alpha2,
        "source_seed1": dataset.source_seed1,
        "source_seed2": dataset.source_seed2,
    }
    arrays.update({f"feature__{name}": values for name, values in dataset.features.items()})
    np.savez_compressed(path, **arrays)


def load_feature_dataset(path: Path) -> FeatureDataset:
    """保存NPZから特徴データを復元する。"""

    with np.load(path, allow_pickle=False) as archive:
        features = {
            name.removeprefix("feature__"): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith("feature__")
        }
        return FeatureDataset(
            split=np.asarray(archive["split"], dtype=np.str_),
            collision_id=np.asarray(archive["collision_id"], dtype=np.int64),
            pair_seed=np.asarray(archive["pair_seed"], dtype=np.int64),
            order_index=np.asarray(archive["order_index"], dtype=np.int64),
            alpha1=np.asarray(archive["alpha1"], dtype=np.float64),
            alpha2=np.asarray(archive["alpha2"], dtype=np.float64),
            source_seed1=np.asarray(archive["source_seed1"], dtype=np.int64),
            source_seed2=np.asarray(archive["source_seed2"], dtype=np.int64),
            features=features,
        )


def validate_feature_dataset(dataset: FeatureDataset, config: ExperimentConfig) -> None:
    """特徴行数、次元、衝突対、分割を検証する。"""

    expected_fit = len(config.fit_pair_seeds) * len(alpha_pairs(config.fit_alphas)) * 2
    expected_test = len(config.test_pair_seeds) * len(alpha_pairs(config.test_alphas)) * 2
    expected_rows = expected_fit + expected_test
    if dataset.split.size != expected_rows:
        raise ValueError(f"特徴行数が不正です: {dataset.split.size} != {expected_rows}")
    expected_dimensions = {
        "oracle_tm_full_rank": 32,
        "oracle_fourier_full_rank": 32,
        "direct_tm_full_rank": 32,
        "direct_fourier_full_rank": 32,
        "direct_tm_rank_one": 16,
        "direct_fourier_rank_one": 16,
    }
    if set(dataset.features) != set(expected_dimensions):
        raise ValueError("保存特徴のモデル集合が不正です。")
    for name, dimension in expected_dimensions.items():
        values = dataset.features[name]
        if values.shape != (expected_rows, dimension):
            raise ValueError(f"{name}の形状が不正です: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name}に非有限値があります。")
    if int(np.sum(dataset.split == "fit")) != expected_fit:
        raise ValueError("fit特徴行数が不正です。")
    if int(np.sum(dataset.split == "test")) != expected_test:
        raise ValueError("test特徴行数が不正です。")
    unique_collisions, counts = np.unique(dataset.collision_id, return_counts=True)
    if unique_collisions.size * 2 != expected_rows or not np.all(counts == 2):
        raise ValueError("各collision_idはforward/swappedの2行必要です。")
    if not np.all(dataset.alpha1 + dataset.alpha2 > 0.0):
        raise ValueError("alpha目標が不正です。")


def ordered_pair_rmse(targets: FloatArray, predictions: FloatArray) -> float:
    """順序付き2座標の全要素RMSEを返す。"""

    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targetsとpredictionsは同じ(N,2)形状にしてください。")
    return float(np.sqrt(np.mean(np.square(targets - predictions))))


def permutation_invariant_rmse(targets: FloatArray, predictions: FloatArray) -> float:
    """各行で正順・逆順の小さい二乗誤差を採用したRMSEを返す。"""

    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targetsとpredictionsは同じ(N,2)形状にしてください。")
    direct = np.mean(np.square(targets - predictions), axis=1)
    swapped = np.mean(np.square(targets[:, ::-1] - predictions), axis=1)
    return float(np.sqrt(np.mean(np.minimum(direct, swapped))))


def fit_multioutput_ridge(
    fit_features: FloatArray,
    fit_targets: FloatArray,
    evaluation_features: FloatArray,
    penalty: float,
    pilot: ModuleType,
) -> FloatArray:
    """既存のtrain-only標準化ridgeを2目的変数へ独立適用する。"""

    if fit_targets.ndim != 2 or fit_targets.shape[1] != 2:
        raise ValueError("fit_targetsは(N,2)にしてください。")
    return np.column_stack(
        [
            pilot.fit_ridge(
                fit_features,
                fit_targets[:, coordinate],
                evaluation_features,
                penalty,
            )
            for coordinate in range(2)
        ]
    ).astype(np.float64)


def assign_grouped_folds(config: ExperimentConfig) -> Dict[int, int]:
    """同じpair seedを一つのfoldへ置く決定論的割当を返す。"""

    seeds = np.asarray(config.fit_pair_seeds, dtype=np.int64)
    shuffled = np.random.default_rng(config.selection_seed).permutation(seeds)
    return {int(seed): int(index % config.cv_folds) for index, seed in enumerate(shuffled)}


def grouped_cv_predictions(
    features: FloatArray,
    targets: FloatArray,
    pair_seeds: IntArray,
    fold_by_seed: Dict[int, int],
    penalty: float,
    pilot: ModuleType,
) -> FloatArray:
    """pair-seed grouped foldのout-of-fold 2目的予測を返す。"""

    predictions = np.empty_like(targets)
    fold_labels = np.asarray(
        [fold_by_seed[int(seed)] for seed in pair_seeds],
        dtype=np.int64,
    )
    for fold in sorted(set(fold_by_seed.values())):
        validation_mask = fold_labels == fold
        train_mask = ~validation_mask
        predictions[validation_mask] = fit_multioutput_ridge(
            features[train_mask],
            targets[train_mask],
            features[validation_mask],
            penalty,
            pilot,
        )
    return predictions


def select_penalty(
    model: str,
    features: FloatArray,
    targets: FloatArray,
    pair_seeds: IntArray,
    fold_by_seed: Dict[int, int],
    penalties: Tuple[float, ...],
    pilot: ModuleType,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """grouped CVの順序付きRMSEだけでridge強度を選ぶ。"""

    rows: List[Dict[str, object]] = []
    for penalty in penalties:
        predictions = grouped_cv_predictions(
            features,
            targets,
            pair_seeds,
            fold_by_seed,
            penalty,
            pilot,
        )
        rows.append(
            {
                "model": model,
                "feature_dimension": int(features.shape[1]),
                "penalty": float(penalty),
                "cv_ordered_rmse": ordered_pair_rmse(targets, predictions),
                "cv_permutation_invariant_rmse": permutation_invariant_rmse(
                    targets,
                    predictions,
                ),
            }
        )
    selected = min(
        rows,
        key=lambda row: (float(row["cv_ordered_rmse"]), float(row["penalty"])),
    )
    for row in rows:
        row["selected"] = bool(row is selected)
    return float(selected["penalty"]), float(selected["cv_ordered_rmse"]), rows


def theoretical_collision_floor(targets: FloatArray) -> float:
    """完全衝突を持つ順序付き2目的問題のRMSE下限を返す。"""

    return float(np.sqrt(np.mean(np.square((targets[:, 0] - targets[:, 1]) / 2.0))))


def collision_feature_differences(
    dataset: FeatureDataset,
    mask: NDArray[np.bool_],
) -> List[Dict[str, object]]:
    """rank-one衝突対の特徴差と理論下限寄与を行別に返す。"""

    rows: List[Dict[str, object]] = []
    for collision_id in np.unique(dataset.collision_id[mask]):
        indices = np.flatnonzero(mask & (dataset.collision_id == collision_id))
        if indices.size != 2:
            raise ValueError(f"collision_id={collision_id}が2行ではありません。")
        first, second = int(indices[0]), int(indices[1])
        rows.append(
            {
                "collision_id": int(collision_id),
                "split": str(dataset.split[first]),
                "pair_seed": int(dataset.pair_seed[first]),
                "alpha_low": float(min(dataset.alpha1[first], dataset.alpha2[first])),
                "alpha_high": float(max(dataset.alpha1[first], dataset.alpha2[first])),
                "target_gap": float(abs(dataset.alpha1[first] - dataset.alpha2[first])),
                "rmse_floor": float(abs(dataset.alpha1[first] - dataset.alpha2[first]) / 2.0),
                "tm_feature_max_abs_difference": float(
                    np.max(
                        np.abs(
                            dataset.features["direct_tm_rank_one"][first]
                            - dataset.features["direct_tm_rank_one"][second]
                        )
                    )
                ),
                "fourier_feature_max_abs_difference": float(
                    np.max(
                        np.abs(
                            dataset.features["direct_fourier_rank_one"][first]
                            - dataset.features["direct_fourier_rank_one"][second]
                        )
                    )
                ),
            }
        )
    return rows


def file_sha256(path: Path) -> str:
    """再現対象ファイルのSHA-256を計算する。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(pilot: ModuleType, path: Path, rows: List[Dict[str, object]]) -> None:
    """空でない辞書行をE1Aと同じUTF-8 CSV形式で保存する。"""

    if not rows:
        raise ValueError(f"空のCSVは保存できません: {path}")
    pilot.write_csv(path, list(rows[0].keys()), rows)


def create_result_plots(
    model_rows: List[Dict[str, object]],
    prediction_rows: List[Dict[str, object]],
    theoretical_floor: float,
    artifacts_directory: Path,
) -> None:
    """model誤差と真値対予測を再現可能なPNGへ描画する。"""

    labels = {
        "oracle_tm_full_rank": "Oracle TM (2 obs.)",
        "oracle_fourier_full_rank": "Oracle Fourier (2 obs.)",
        "direct_tm_full_rank": "Direct TM (2 obs.)",
        "direct_fourier_full_rank": "Direct Fourier (2 obs.)",
        "direct_tm_rank_one": "TM (1 symmetric obs.)",
        "direct_fourier_rank_one": "Fourier (1 symmetric obs.)",
    }
    colors = {
        "oracle_tm_full_rank": "#0072B2",
        "oracle_fourier_full_rank": "#56B4E9",
        "direct_tm_full_rank": "#009E73",
        "direct_fourier_full_rank": "#CC79A7",
        "direct_tm_rank_one": "#D55E00",
        "direct_fourier_rank_one": "#E69F00",
    }
    by_model = {str(row["model"]): row for row in model_rows}
    positions = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharey=True)
    for axis, metric, title in (
        (axes[0], "test_ordered_rmse", "Ordered pair RMSE"),
        (
            axes[1],
            "test_permutation_invariant_rmse",
            "Permutation-invariant RMSE",
        ),
    ):
        values = [float(by_model[model][metric]) for model in MODEL_ORDER]
        axis.barh(
            positions,
            values,
            color=[colors[model] for model in MODEL_ORDER],
            edgecolor="#333333",
            linewidth=0.7,
        )
        axis.set_xlim(left=0.0)
        axis.set_xlabel("Test RMSE")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        for position, value in zip(positions, values):
            axis.text(value, position, f"  {value:.4f}", va="center", fontsize=8)
    axes[0].axvline(
        theoretical_floor,
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label=f"rank-one ordered floor = {theoretical_floor:.4f}",
    )
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[0].set_yticks(positions, [labels[model] for model in MODEL_ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("Two-source identifiability: invertible vs symmetric observation")
    fig.tight_layout()
    fig.savefig(artifacts_directory / "identifiability_results.png", dpi=180)
    plt.close(fig)

    targets = np.asarray(
        [[float(row["alpha1"]), float(row["alpha2"])] for row in prediction_rows],
        dtype=np.float64,
    )
    scatter_models = (
        "oracle_tm_full_rank",
        "direct_tm_full_rank",
        "direct_tm_rank_one",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharex=True, sharey=True)
    bounds = (
        float(min(np.min(targets), min(config_value for config_value in targets.ravel()))) - 0.02,
        float(max(np.max(targets), max(config_value for config_value in targets.ravel()))) + 0.02,
    )
    for axis, model in zip(axes, scatter_models):
        predictions = np.asarray(
            [
                [
                    float(row[f"prediction_{model}_alpha1"]),
                    float(row[f"prediction_{model}_alpha2"]),
                ]
                for row in prediction_rows
            ],
            dtype=np.float64,
        )
        axis.scatter(
            targets[:, 0],
            predictions[:, 0],
            s=13,
            alpha=0.42,
            color="#0072B2",
            label="coordinate 1",
        )
        axis.scatter(
            targets[:, 1],
            predictions[:, 1],
            s=13,
            alpha=0.42,
            color="#D55E00",
            marker="x",
            label="coordinate 2",
        )
        axis.plot(bounds, bounds, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_xlim(bounds)
        axis.set_ylim(bounds)
        axis.set_title(labels[model])
        axis.set_xlabel("True alpha")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted alpha")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ordered alpha recovery on unseen test alphas")
    fig.tight_layout()
    fig.savefig(artifacts_directory / "prediction_scatter.png", dpi=180)
    plt.close(fig)


def run_experiment(
    run_directory: Path,
    config: ExperimentConfig,
    reuse_data_path: Path | None = None,
    reuse_features_path: Path | None = None,
) -> Dict[str, object]:
    """データ生成、特徴抽出、grouped CV、凍結test、保存を一括実行する。"""

    matched = load_matched_module()
    pilot = matched.load_pilot_module()
    artifacts = run_directory / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    local_source_path = artifacts / "source_trajectories.npz"
    if reuse_data_path is not None:
        source = load_source_dataset(reuse_data_path)
    elif reuse_features_path is not None and local_source_path.is_file():
        source = load_source_dataset(local_source_path)
    else:
        source = build_source_dataset(config, pilot)
    validate_source_dataset(source, config)
    save_source_dataset(local_source_path, source)

    dataset = (
        load_feature_dataset(reuse_features_path)
        if reuse_features_path is not None
        else build_feature_dataset(source, config, pilot, matched)
    )
    validate_feature_dataset(dataset, config)
    feature_path = artifacts / "features.npz"
    save_feature_dataset(feature_path, dataset)

    source_manifest_rows = [
        {
            "collision_id": index,
            "split": str(source.split[index]),
            "pair_seed": int(source.pair_seed[index]),
            "alpha_low": float(source.alpha_low[index]),
            "alpha_high": float(source.alpha_high[index]),
            "left_seed": int(source.left_seed[index]),
            "right_seed": int(source.right_seed[index]),
            "low_location": float(source.low_location[index]),
            "low_scale": float(source.low_scale[index]),
            "high_location": float(source.high_location[index]),
            "high_scale": float(source.high_scale[index]),
        }
        for index in range(source.split.size)
    ]
    write_csv(pilot, artifacts / "source_trajectory_manifest.csv", source_manifest_rows)
    manifest_rows = [
        {
            "row_id": index,
            "split": str(dataset.split[index]),
            "collision_id": int(dataset.collision_id[index]),
            "pair_seed": int(dataset.pair_seed[index]),
            "order_index": int(dataset.order_index[index]),
            "alpha1": float(dataset.alpha1[index]),
            "alpha2": float(dataset.alpha2[index]),
            "source_seed1": int(dataset.source_seed1[index]),
            "source_seed2": int(dataset.source_seed2[index]),
        }
        for index in range(dataset.split.size)
    ]
    write_csv(pilot, artifacts / "trajectory_manifest.csv", manifest_rows)

    fit_mask = dataset.split == "fit"
    test_mask = dataset.split == "test"
    fit_targets = np.column_stack((dataset.alpha1[fit_mask], dataset.alpha2[fit_mask]))
    test_targets = np.column_stack((dataset.alpha1[test_mask], dataset.alpha2[test_mask]))
    fit_pair_seeds = dataset.pair_seed[fit_mask]
    test_pair_seeds = dataset.pair_seed[test_mask]
    fold_by_seed = assign_grouped_folds(config)
    fold_rows = [
        {"pair_seed": seed, "fold": fold_by_seed[seed]}
        for seed in sorted(fold_by_seed)
    ]
    write_csv(pilot, artifacts / "fold_assignments.csv", fold_rows)

    cv_rows: List[Dict[str, object]] = []
    selected_settings: Dict[str, Tuple[float, float]] = {}
    for model in MODEL_ORDER:
        penalty, cv_rmse, rows = select_penalty(
            model,
            dataset.features[model][fit_mask],
            fit_targets,
            fit_pair_seeds,
            fold_by_seed,
            config.ridge_penalties,
            pilot,
        )
        selected_settings[model] = (penalty, cv_rmse)
        cv_rows.extend(rows)
    write_csv(pilot, artifacts / "model_cv_metrics.csv", cv_rows)

    predictions_by_model: Dict[str, FloatArray] = {}
    model_rows: List[Dict[str, object]] = []
    for model in MODEL_ORDER:
        matrix = dataset.features[model]
        penalty, cv_rmse = selected_settings[model]
        fit_predictions = fit_multioutput_ridge(
            matrix[fit_mask],
            fit_targets,
            matrix[fit_mask],
            penalty,
            pilot,
        )
        test_predictions = fit_multioutput_ridge(
            matrix[fit_mask],
            fit_targets,
            matrix[test_mask],
            penalty,
            pilot,
        )
        predictions_by_model[model] = test_predictions
        selected_cv_row = next(
            row
            for row in cv_rows
            if row["model"] == model and bool(row["selected"])
        )
        model_rows.append(
            {
                "model": model,
                "observation_rank": 1 if model.endswith("rank_one") else 2,
                "feature_dimension": int(matrix.shape[1]),
                "selected_penalty": penalty,
                "cv_ordered_rmse": cv_rmse,
                "cv_permutation_invariant_rmse": float(
                    selected_cv_row["cv_permutation_invariant_rmse"]
                ),
                "fit_ordered_rmse": ordered_pair_rmse(fit_targets, fit_predictions),
                "fit_permutation_invariant_rmse": permutation_invariant_rmse(
                    fit_targets,
                    fit_predictions,
                ),
                "test_ordered_rmse": ordered_pair_rmse(test_targets, test_predictions),
                "test_permutation_invariant_rmse": permutation_invariant_rmse(
                    test_targets,
                    test_predictions,
                ),
            }
        )
    write_csv(pilot, artifacts / "model_metrics.csv", model_rows)

    prediction_rows: List[Dict[str, object]] = []
    test_indices = np.flatnonzero(test_mask)
    for local_index, row_index in enumerate(test_indices):
        row: Dict[str, object] = {
            "row_id": int(row_index),
            "collision_id": int(dataset.collision_id[row_index]),
            "pair_seed": int(dataset.pair_seed[row_index]),
            "order_index": int(dataset.order_index[row_index]),
            "alpha1": float(dataset.alpha1[row_index]),
            "alpha2": float(dataset.alpha2[row_index]),
        }
        for model, predictions in predictions_by_model.items():
            row[f"prediction_{model}_alpha1"] = float(predictions[local_index, 0])
            row[f"prediction_{model}_alpha2"] = float(predictions[local_index, 1])
        prediction_rows.append(row)
    write_csv(pilot, artifacts / "test_predictions.csv", prediction_rows)

    pair_rows: List[Dict[str, object]] = []
    for low, high in alpha_pairs(config.test_alphas):
        pair_mask = np.isclose(np.minimum(test_targets[:, 0], test_targets[:, 1]), low) & np.isclose(
            np.maximum(test_targets[:, 0], test_targets[:, 1]),
            high,
        )
        for model in MODEL_ORDER:
            pair_rows.append(
                {
                    "alpha_low": low,
                    "alpha_high": high,
                    "model": model,
                    "row_count": int(np.sum(pair_mask)),
                    "ordered_rmse": ordered_pair_rmse(
                        test_targets[pair_mask],
                        predictions_by_model[model][pair_mask],
                    ),
                    "permutation_invariant_rmse": permutation_invariant_rmse(
                        test_targets[pair_mask],
                        predictions_by_model[model][pair_mask],
                    ),
                }
            )
    write_csv(pilot, artifacts / "pair_metrics.csv", pair_rows)

    seed_rows: List[Dict[str, object]] = []
    for pair_seed in sorted(set(test_pair_seeds.tolist())):
        seed_mask = test_pair_seeds == pair_seed
        for model in MODEL_ORDER:
            seed_rows.append(
                {
                    "pair_seed": pair_seed,
                    "model": model,
                    "row_count": int(np.sum(seed_mask)),
                    "ordered_rmse": ordered_pair_rmse(
                        test_targets[seed_mask],
                        predictions_by_model[model][seed_mask],
                    ),
                    "permutation_invariant_rmse": permutation_invariant_rmse(
                        test_targets[seed_mask],
                        predictions_by_model[model][seed_mask],
                    ),
                }
            )
    write_csv(pilot, artifacts / "seed_metrics.csv", seed_rows)


    collision_rows = collision_feature_differences(dataset, test_mask)
    write_csv(pilot, artifacts / "collision_diagnostics.csv", collision_rows)
    rank_one_max_difference = max(
        max(
            float(row["tm_feature_max_abs_difference"]),
            float(row["fourier_feature_max_abs_difference"]),
        )
        for row in collision_rows
    )
    collision_floor = theoretical_collision_floor(test_targets)
    model_by_name = {str(row["model"]): row for row in model_rows}
    oracle_tm_rmse = float(
        model_by_name["oracle_tm_full_rank"]["test_ordered_rmse"]
    )
    rank_one_floor_checks = {
        model: bool(
            float(model_by_name[model]["test_ordered_rmse"])
            >= collision_floor - config.collision_tolerance
        )
        for model in ("direct_tm_rank_one", "direct_fourier_rank_one")
    }
    success_components = {
        "oracle_tm_below_threshold": bool(
            oracle_tm_rmse < config.oracle_tm_rmse_threshold
        ),
        "rank_one_collision_exact": bool(
            rank_one_max_difference <= config.collision_tolerance
        ),
        "rank_one_ordered_floor_tm": rank_one_floor_checks["direct_tm_rank_one"],
        "rank_one_ordered_floor_fourier": rank_one_floor_checks[
            "direct_fourier_rank_one"
        ],
    }
    success = bool(all(success_components.values()))
    create_result_plots(
        model_rows,
        prediction_rows,
        collision_floor,
        artifacts,
    )

    summary: Dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "question": (
            "既知full-rank 2観測では2源alphaを回収でき、"
            "対称1観測では順序付き目標が識別不能になるか。"
        ),
        "config": {
            **asdict(config),
            "mixing_matrix": config.mixing_matrix.tolist(),
        },
        "sample_counts": {
            "source_collision_pairs": int(source.split.size),
            "fit_rows": int(np.sum(fit_mask)),
            "test_rows": int(np.sum(test_mask)),
        },
        "mixing_matrix_determinant": float(np.linalg.det(config.mixing_matrix)),
        "theoretical_collision_floor": collision_floor,
        "rank_one_collision_feature_max_abs_difference": rank_one_max_difference,
        "per_model": model_rows,
        "success_components": success_components,
        "success": success,
        "artifacts": {
            "source_data": "artifacts/source_trajectories.npz",
            "feature_store": "artifacts/features.npz",
            "source_manifest": "artifacts/source_trajectory_manifest.csv",
            "trajectory_manifest": "artifacts/trajectory_manifest.csv",
            "fold_assignments": "artifacts/fold_assignments.csv",
            "model_cv_metrics": "artifacts/model_cv_metrics.csv",
            "model_metrics": "artifacts/model_metrics.csv",
            "test_predictions": "artifacts/test_predictions.csv",
            "pair_metrics": "artifacts/pair_metrics.csv",
            "seed_metrics": "artifacts/seed_metrics.csv",
            "collision_diagnostics": "artifacts/collision_diagnostics.csv",
            "plots": [
                "artifacts/identifiability_results.png",
                "artifacts/prediction_scatter.png",
            ],
        },
        "caveats": [
            "既知Aのoracle逆変換はblind source separationではない。",
            "源ごとの混合前標準化を使う理想化実験であり、未知混合の実データでは直接利用できない。",
            "test alphaはfit範囲内の未使用補間点であり、外挿ではない。",
            "識別可能性と情報圧縮、同期、画像復号は別問題である。",
            "rank-oneの非識別可能性は順序付き目標に対する主張であり、unordered pair推定を否定しない。",
        ],
    }
    summary_path = artifacts / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = {
        "schema_version": 1,
        "status": "completed",
        "primary": {
            "success": success,
            "success_components": success_components,
            "oracle_tm_full_rank_ordered_test_rmse": oracle_tm_rmse,
            "oracle_tm_rmse_threshold": config.oracle_tm_rmse_threshold,
            "theoretical_collision_floor": collision_floor,
            "rank_one_collision_feature_max_abs_difference": rank_one_max_difference,
            "collision_tolerance": config.collision_tolerance,
            "rank_one_ordered_test_rmse": {
                model: float(model_by_name[model]["test_ordered_rmse"])
                for model in ("direct_tm_rank_one", "direct_fourier_rank_one")
            },
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
    config_record["actual_run_config"] = {
        **asdict(config),
        "mixing_matrix": config.mixing_matrix.tolist(),
    }
    config_path.write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    validator_path = script_path.with_name("validate_results.py")
    environment_path = run_directory / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    artifact_names = (
        "source_trajectories.npz",
        "features.npz",
        "source_trajectory_manifest.csv",
        "trajectory_manifest.csv",
        "fold_assignments.csv",
        "model_cv_metrics.csv",
        "model_metrics.csv",
        "test_predictions.csv",
        "pair_metrics.csv",
        "seed_metrics.csv",
        "collision_diagnostics.csv",
        "identifiability_results.png",
        "prediction_scatter.png",
        "summary.json",
    )
    environment.update(
        {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": [
                f"numpy=={np.__version__}",
                f"matplotlib=={matplotlib.__version__}",
            ],
            "execution_command": (
                "python e1b_two_source_identifiability.py --run-directory ."
            ),
            "reuse_data_command": (
                "python e1b_two_source_identifiability.py --run-directory . "
                "--reuse-data artifacts/source_trajectories.npz"
            ),
            "reuse_features_command": (
                "python e1b_two_source_identifiability.py --run-directory . "
                "--reuse-features artifacts/features.npz"
            ),
            "reused_data_store": str(reuse_data_path) if reuse_data_path else None,
            "reused_feature_store": (
                str(reuse_features_path) if reuse_features_path else None
            ),
            "code_sha256": {
                script_path.name: file_sha256(script_path),
                validator_path.name: file_sha256(validator_path),
                MATCHED_SCRIPT_PATH.name: file_sha256(MATCHED_SCRIPT_PATH),
                matched.PILOT_SCRIPT_PATH.name: file_sha256(matched.PILOT_SCRIPT_PATH),
            },
            "data_provenance": {
                "generator": "F_alpha(x)=alpha*(x-1/x)",
                "source_data": "artifacts/source_trajectories.npz",
                "source_identifiers": "artifacts/source_trajectory_manifest.csv",
                "model_input_store": "artifacts/features.npz",
                "observation_rule": {
                    "full_rank": config.mixing_matrix.tolist(),
                    "rank_one": [float(1.0 / np.sqrt(2.0))] * 2,
                },
            },
            "artifact_sha256": {
                name: file_sha256(artifacts / name) for name in artifact_names
            },
        }
    )
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_arguments(arguments: List[str]) -> argparse.Namespace:
    """runと任意の再利用データをCLIから受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path.cwd())
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-data", type=Path)
    reuse.add_argument("--reuse-features", type=Path)
    return parser.parse_args(arguments)


def main(arguments: List[str] | None = None) -> int:
    """事前登録configで実験を実行し主要結果を標準出力へ返す。"""

    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    run_directory = args.run_directory.resolve()
    config = load_config(run_directory / "config.json")
    summary = run_experiment(
        run_directory,
        config,
        args.reuse_data.resolve() if args.reuse_data else None,
        args.reuse_features.resolve() if args.reuse_features else None,
    )
    print(
        json.dumps(
            {
                "success": summary["success"],
                "theoretical_collision_floor": summary[
                    "theoretical_collision_floor"
                ],
                "rank_one_collision_feature_max_abs_difference": summary[
                    "rank_one_collision_feature_max_abs_difference"
                ],
                "oracle_tm_ordered_rmse": next(
                    row["test_ordered_rmse"]
                    for row in summary["per_model"]
                    if row["model"] == "oracle_tm_full_rank"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
