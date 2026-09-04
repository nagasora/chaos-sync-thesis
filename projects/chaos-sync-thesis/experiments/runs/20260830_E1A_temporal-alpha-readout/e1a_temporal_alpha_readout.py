"""周辺尺度を除いた一般化Boole軌道から写像パラメータを回収する。"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

E0_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "20260806_E0_boole-invariant-measure"
    / "e0_boole_validation.py"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """E1A pilotの軌道生成、特徴抽出、評価条件を保持する。"""

    train_alphas: Tuple[float, ...] = (0.34, 0.42, 0.50, 0.58, 0.66)
    test_alphas: Tuple[float, ...] = (0.38, 0.46, 0.54, 0.62)
    train_seeds: Tuple[int, ...] = tuple(range(20260830, 20260850))
    validation_seeds: Tuple[int, ...] = tuple(range(20261830, 20261838))
    test_seeds: Tuple[int, ...] = tuple(range(20262830, 20262842))
    burn_in: int = 2_048
    observation_length: int = 4_096
    tm_max_order: int = 8
    delay_lags: Tuple[int, ...] = (1, 2, 4, 8)
    raw_window_length: int = 64
    fourier_band_count: int = 8
    ridge_penalties: Tuple[float, ...] = (
        1.0e-6,
        1.0e-4,
        1.0e-2,
        1.0,
        100.0,
    )
    bootstrap_repetitions: int = 2_000
    bootstrap_seed: int = 20260830
    feature_shuffle_seed: int = 20260830

    def validate(self) -> None:
        """未使用alpha補間と有限時間特徴が成立する設定か検証する。"""

        if not self.train_alphas or not self.test_alphas:
            raise ValueError("train_alphas と test_alphas は空にできません。")
        all_alphas = self.train_alphas + self.test_alphas
        if any(not 0.0 < alpha < 1.0 for alpha in all_alphas):
            raise ValueError("alpha は 0 < alpha < 1 を満たす必要があります。")
        if set(self.train_alphas) & set(self.test_alphas):
            raise ValueError("test alpha は学習に未使用でなければなりません。")
        if not (
            min(self.train_alphas) < min(self.test_alphas)
            and max(self.test_alphas) < max(self.train_alphas)
        ):
            raise ValueError("test alpha はtrain alpha範囲内の補間点にしてください。")
        seed_sets = [
            set(self.train_seeds),
            set(self.validation_seeds),
            set(self.test_seeds),
        ]
        if any(not seeds for seeds in seed_sets):
            raise ValueError("各splitには少なくとも一つのseedが必要です。")
        if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("train/validation/test seed は重複できません。")
        if self.burn_in < 0 or self.observation_length <= 0:
            raise ValueError("burn_in は非負、observation_length は正にしてください。")
        if self.tm_max_order <= 0 or not self.delay_lags:
            raise ValueError("TM次数と時間遅延は正にしてください。")
        if any(lag <= 0 or lag >= self.observation_length for lag in self.delay_lags):
            raise ValueError("時間遅延は0より大きく観測長より小さくしてください。")
        if not 0 < self.raw_window_length <= self.observation_length:
            raise ValueError("raw_window_length は観測長以下の正数にしてください。")
        if self.fourier_band_count <= 0:
            raise ValueError("fourier_band_count は正にしてください。")
        if any(penalty <= 0.0 for penalty in self.ridge_penalties):
            raise ValueError("ridge_penalties は正にしてください。")
        if self.bootstrap_repetitions <= 0:
            raise ValueError("bootstrap_repetitions は正にしてください。")


@dataclass(frozen=True)
class TrajectorySample:
    """一つの軌道の識別子、前処理診断値、特徴量を保持する。"""

    split: str
    alpha: float
    seed: int
    estimated_location: float
    estimated_scale: float
    features: Dict[str, FloatArray]


@dataclass(frozen=True)
class ProbeResult:
    """一つの特徴集合に対するridge probeの選択結果を保持する。"""

    feature_name: str
    feature_dimension: int
    selected_penalty: float
    train_rmse: float
    validation_rmse: float
    test_rmse: float
    predictions: Dict[str, FloatArray]


def load_e0_module() -> ModuleType:
    """検証済みE0の軌道生成を、コード複製せず読み込む。"""

    spec = importlib.util.spec_from_file_location("e0_boole_for_e1a", E0_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"E0実験コードを読み込めません: {E0_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Why not: exec_moduleだけではE0内dataclassが参照するsys.modulesに登録されない。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E0_MODULE = load_e0_module()


def robust_standardize(orbit: FloatArray) -> Tuple[FloatArray, float, float]:
    """軌道ごとの中央値と半四分位幅を除き、周辺尺度差を消す。"""

    first_quartile, median, third_quartile = np.quantile(orbit, [0.25, 0.5, 0.75])
    scale = float((third_quartile - first_quartile) / 2.0)
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError("軌道のロバスト尺度を正に推定できません。")
    standardized = (orbit - median) / scale
    return np.asarray(standardized, dtype=np.float64), float(median), scale


def cayley_transform(values: FloatArray) -> ComplexArray:
    """尺度1のCayley変換で重い裾を単位円上の有界値へ写す。"""

    transformed = (values - 1j) / (values + 1j)
    return np.asarray(transformed, dtype=np.complex128)


def complex_to_real_features(values: List[complex]) -> FloatArray:
    """複素特徴をridgeへ入力できる実部・虚部の並びへ変換する。"""

    return np.asarray(
        [component for value in values for component in (value.real, value.imag)],
        dtype=np.float64,
    )


def tm_instantaneous_features(cayley_values: ComplexArray, max_order: int) -> FloatArray:
    """有限軌道上のCayley/TMモード平均を次数ごとに計算する。"""

    moments = [complex(np.mean(np.power(cayley_values, order))) for order in range(1, max_order + 1)]
    return complex_to_real_features(moments)


def tm_delay_features(
    cayley_values: ComplexArray,
    max_order: int,
    lags: Tuple[int, ...],
) -> FloatArray:
    """TMモード平均と各次数の時間遅延相関を結合する。"""

    values: List[complex] = []
    for order in range(1, max_order + 1):
        mode = np.power(cayley_values, order)
        values.append(complex(np.mean(mode)))
        for lag in lags:
            values.append(complex(np.mean(mode[lag:] * np.conjugate(mode[:-lag]))))
    return complex_to_real_features(values)


def fourier_band_features(cayley_values: ComplexArray, band_count: int) -> FloatArray:
    """Cayley系列の実部・虚部について周波数帯域別対数パワーを返す。"""

    band_features: List[float] = []
    for component in (cayley_values.real, cayley_values.imag):
        centered = component - np.mean(component)
        power = np.square(np.abs(np.fft.rfft(centered)))[1:]
        for band in np.array_split(power, band_count):
            band_features.append(float(np.log1p(np.mean(band))))
    return np.asarray(band_features, dtype=np.float64)


def extract_features(
    orbit: FloatArray,
    seed: int,
    alpha: float,
    config: ExperimentConfig,
) -> Tuple[Dict[str, FloatArray], float, float]:
    """同じ軌道から主特徴、比較対象、時間順序破壊対照を作る。"""

    standardized, location, scale = robust_standardize(orbit)
    cayley_values = cayley_transform(standardized)
    shuffle_seed = seed ^ int(round(alpha * 1_000_000.0)) ^ config.feature_shuffle_seed
    permutation = np.random.default_rng(shuffle_seed).permutation(cayley_values.size)
    shuffled = cayley_values[permutation]
    quantile_probabilities = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    features = {
        "robust_quantiles": np.asarray(
            np.quantile(standardized, quantile_probabilities),
            dtype=np.float64,
        ),
        "raw_window": np.concatenate(
            [
                cayley_values.real[: config.raw_window_length],
                cayley_values.imag[: config.raw_window_length],
            ]
        ).astype(np.float64),
        "fourier_bands": fourier_band_features(cayley_values, config.fourier_band_count),
        "cayley_delay_k1": tm_delay_features(cayley_values, 1, config.delay_lags),
        "tm_no_delay": tm_instantaneous_features(cayley_values, config.tm_max_order),
        "tm_delay_shuffled": tm_delay_features(
            shuffled,
            config.tm_max_order,
            config.delay_lags,
        ),
        "tm_delay": tm_delay_features(
            cayley_values,
            config.tm_max_order,
            config.delay_lags,
        ),
    }
    if any(not np.all(np.isfinite(feature)) for feature in features.values()):
        raise FloatingPointError("特徴量に非有限値が発生しました。")
    return features, location, scale


def build_samples(config: ExperimentConfig) -> List[TrajectorySample]:
    """alphaとseedの事前固定splitから独立軌道標本を生成する。"""

    config.validate()
    split_specs = [
        ("train", config.train_alphas, config.train_seeds),
        ("validation", config.train_alphas, config.validation_seeds),
        ("test", config.test_alphas, config.test_seeds),
    ]
    samples: List[TrajectorySample] = []
    for split, alphas, seeds in split_specs:
        for alpha in alphas:
            for seed in seeds:
                orbit = E0_MODULE.generate_orbit(
                    alpha,
                    seed,
                    config.burn_in,
                    config.observation_length,
                )
                features, location, scale = extract_features(orbit, seed, alpha, config)
                samples.append(
                    TrajectorySample(
                        split=split,
                        alpha=alpha,
                        seed=seed,
                        estimated_location=location,
                        estimated_scale=scale,
                        features=features,
                    )
                )
    return samples


def split_feature_matrix(
    samples: List[TrajectorySample],
    feature_name: str,
    split: str,
) -> Tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """指定splitの特徴行列、alpha、seedを同じ行順で返す。"""

    selected = [sample for sample in samples if sample.split == split]
    matrix = np.stack([sample.features[feature_name] for sample in selected])
    targets = np.asarray([sample.alpha for sample in selected], dtype=np.float64)
    seeds = np.asarray([sample.seed for sample in selected], dtype=np.int64)
    return matrix, targets, seeds


def rmse(targets: FloatArray, predictions: FloatArray) -> float:
    """alpha回収の二乗平均平方根誤差を計算する。"""

    return float(np.sqrt(np.mean(np.square(targets - predictions))))


def fit_ridge(
    train_features: FloatArray,
    train_targets: FloatArray,
    evaluation_features: FloatArray,
    penalty: float,
) -> FloatArray:
    """train統計だけで特徴を標準化し、切片付きridge予測を返す。"""

    feature_mean = np.mean(train_features, axis=0)
    feature_scale = np.std(train_features, axis=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    standardized_train = (train_features - feature_mean) / feature_scale
    standardized_evaluation = (evaluation_features - feature_mean) / feature_scale
    target_mean = float(np.mean(train_targets))
    centered_targets = train_targets - target_mean
    gram = standardized_train.T @ standardized_train
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float64),
        standardized_train.T @ centered_targets,
    )
    return np.asarray(
        target_mean + standardized_evaluation @ coefficients,
        dtype=np.float64,
    )


def evaluate_feature(
    samples: List[TrajectorySample],
    feature_name: str,
    penalties: Tuple[float, ...],
) -> ProbeResult:
    """validation RMSEでridge強度を選び、固定したprobeをtest評価する。"""

    train_x, train_y, _ = split_feature_matrix(samples, feature_name, "train")
    validation_x, validation_y, _ = split_feature_matrix(samples, feature_name, "validation")
    test_x, test_y, _ = split_feature_matrix(samples, feature_name, "test")

    validation_candidates: List[Tuple[float, float]] = []
    for penalty in penalties:
        prediction = fit_ridge(train_x, train_y, validation_x, penalty)
        validation_candidates.append((rmse(validation_y, prediction), penalty))
    selected_validation_rmse, selected_penalty = min(validation_candidates)
    predictions = {
        "train": fit_ridge(train_x, train_y, train_x, selected_penalty),
        "validation": fit_ridge(train_x, train_y, validation_x, selected_penalty),
        "test": fit_ridge(train_x, train_y, test_x, selected_penalty),
    }
    return ProbeResult(
        feature_name=feature_name,
        feature_dimension=int(train_x.shape[1]),
        selected_penalty=float(selected_penalty),
        train_rmse=rmse(train_y, predictions["train"]),
        validation_rmse=float(selected_validation_rmse),
        test_rmse=rmse(test_y, predictions["test"]),
        predictions=predictions,
    )


def bootstrap_rmse_difference(
    targets: FloatArray,
    tm_predictions: FloatArray,
    baseline_predictions: FloatArray,
    seed_labels: NDArray[np.int64],
    repetitions: int,
    bootstrap_seed: int,
) -> Dict[str, float]:
    """seedをクラスタとしてbaseline RMSE minus TM RMSEの区間を推定する。"""

    unique_seeds = np.unique(seed_labels)
    indices_by_seed = {seed: np.flatnonzero(seed_labels == seed) for seed in unique_seeds}
    rng = np.random.default_rng(bootstrap_seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled_seeds = rng.choice(unique_seeds, size=unique_seeds.size, replace=True)
        indices = np.concatenate([indices_by_seed[int(seed)] for seed in sampled_seeds])
        differences[repetition] = rmse(
            targets[indices], baseline_predictions[indices]
        ) - rmse(targets[indices], tm_predictions[indices])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "point_difference": rmse(targets, baseline_predictions) - rmse(targets, tm_predictions),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    """実験表をExcelでも読めるUTF-8-SIG CSVとして保存する。"""

    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_as_dict(config: ExperimentConfig) -> Dict[str, object]:
    """tupleをJSON配列へ変換した実行設定を返す。"""

    value = asdict(config)
    return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}


def run_experiment(run_directory: Path, config: ExperimentConfig) -> Dict[str, object]:
    """全特徴を同一split・probeで評価し、監査可能な成果物を保存する。"""

    config.validate()
    artifacts_directory = run_directory / "artifacts"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    samples = build_samples(config)
    feature_names = list(samples[0].features.keys())
    results = [evaluate_feature(samples, name, config.ridge_penalties) for name in feature_names]
    result_by_name = {result.feature_name: result for result in results}
    baseline_names = [
        "robust_quantiles",
        "raw_window",
        "fourier_bands",
        "cayley_delay_k1",
    ]
    best_baseline_name = min(
        baseline_names,
        key=lambda name: result_by_name[name].validation_rmse,
    )
    _, test_targets, test_seed_labels = split_feature_matrix(samples, "tm_delay", "test")
    confidence_interval = bootstrap_rmse_difference(
        test_targets,
        result_by_name["tm_delay"].predictions["test"],
        result_by_name[best_baseline_name].predictions["test"],
        test_seed_labels,
        config.bootstrap_repetitions,
        config.bootstrap_seed,
    )
    success = bool(confidence_interval["ci_95_lower"] > 0.0)

    metric_rows: List[Dict[str, object]] = [
        {
            "feature": result.feature_name,
            "feature_dimension": result.feature_dimension,
            "selected_penalty": result.selected_penalty,
            "train_rmse": result.train_rmse,
            "validation_rmse": result.validation_rmse,
            "test_rmse": result.test_rmse,
        }
        for result in results
    ]
    write_csv(
        artifacts_directory / "per_feature_metrics.csv",
        list(metric_rows[0].keys()),
        metric_rows,
    )

    test_samples = [sample for sample in samples if sample.split == "test"]
    prediction_rows: List[Dict[str, object]] = []
    for index, sample in enumerate(test_samples):
        row: Dict[str, object] = {
            "alpha": sample.alpha,
            "seed": sample.seed,
            "estimated_location": sample.estimated_location,
            "estimated_scale": sample.estimated_scale,
        }
        for result in results:
            row[f"prediction_{result.feature_name}"] = float(result.predictions["test"][index])
        prediction_rows.append(row)
    write_csv(
        artifacts_directory / "test_predictions.csv",
        list(prediction_rows[0].keys()),
        prediction_rows,
    )

    summary: Dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "question": "周辺Cauchy尺度を軌道ごとに除去しても、TM時間遅延特徴から未使用alphaを回収できるか。",
        "config": config_as_dict(config),
        "sample_counts": {
            split: sum(sample.split == split for sample in samples)
            for split in ("train", "validation", "test")
        },
        "per_feature": metric_rows,
        "best_baseline_selected_on_validation": best_baseline_name,
        "paired_seed_bootstrap": confidence_interval,
        "success": success,
        "observations": {
            "tm_delay_test_rmse": result_by_name["tm_delay"].test_rmse,
            "tm_no_delay_test_rmse": result_by_name["tm_no_delay"].test_rmse,
            "shuffled_tm_delay_test_rmse": result_by_name["tm_delay_shuffled"].test_rmse,
            "best_baseline_test_rmse": result_by_name[best_baseline_name].test_rmse,
        },
        "caveats": [
            "本pilotは単一源のalpha回収だけを扱い、2源分離と識別不能対照は未実行である。",
            "軌道ごとのロバスト標準化は周辺尺度差を抑えるが、有限標本の尺度推定誤差を完全には消さない。",
            "bootstrapはtest seedをクラスタとして再標本化した工学的区間で、理論的保証ではない。",
            "test alphaは学習範囲内の未使用補間点であり、範囲外外挿は検証していない。",
        ],
    }
    (artifacts_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_record = {
        "schema_version": 1,
        "status": "completed",
        "primary": {
            "success": success,
            "tm_delay_test_rmse": result_by_name["tm_delay"].test_rmse,
            "best_baseline_name": best_baseline_name,
            "best_baseline_test_rmse": result_by_name[best_baseline_name].test_rmse,
            "baseline_minus_tm_ci_95": [
                confidence_interval["ci_95_lower"],
                confidence_interval["ci_95_upper"],
            ],
        },
        "secondary": {
            "tm_no_delay_test_rmse": result_by_name["tm_no_delay"].test_rmse,
            "shuffled_tm_delay_test_rmse": result_by_name["tm_delay_shuffled"].test_rmse,
            "summary_artifact": "artifacts/summary.json",
            "per_feature_artifact": "artifacts/per_feature_metrics.csv",
        },
        "per_seed_artifact": "artifacts/test_predictions.csv",
        "notes": summary["caveats"],
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config_path = run_directory / "config.json"
    recorded_config = json.loads(config_path.read_text(encoding="utf-8"))
    recorded_config["status"] = "completed"
    recorded_config["actual_run_config"] = config_as_dict(config)
    config_path.write_text(
        json.dumps(recorded_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    environment_path = run_directory / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment.update(
        {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": [f"numpy=={np.__version__}"],
            "execution_command": "python e1a_temporal_alpha_readout.py --run-directory .",
        }
    )
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_arguments(arguments: List[str]) -> argparse.Namespace:
    """実験出力先をCLIから受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="実験記録ディレクトリ。既定値はスクリプト所在フォルダ。",
    )
    return parser.parse_args(arguments)


def main(arguments: List[str] | None = None) -> int:
    """E1A pilotを実行し、主成功条件を表示する。"""

    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    summary = run_experiment(args.run_directory.resolve(), ExperimentConfig())
    print(
        json.dumps(
            {
                "success": summary["success"],
                "best_baseline": summary["best_baseline_selected_on_validation"],
                "observations": summary["observations"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
