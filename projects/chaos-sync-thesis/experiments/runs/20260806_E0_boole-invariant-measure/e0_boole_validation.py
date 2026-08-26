"""一般化Boole写像の理論値を複数seedの数値軌道で検証する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ValidationThresholds:
    """数値再現を合格とみなす工学的な許容誤差を保持する。"""

    max_scale_relative_error: float = 0.05
    max_lyapunov_absolute_error: float = 0.015
    max_ks_distance: float = 0.02
    max_tm_nonzero_magnitude: float = 0.02
    max_quadrature_lyapunov_error: float = 1.0e-5


@dataclass(frozen=True)
class ExperimentConfig:
    """E0実験の入力条件と再現性パラメータを保持する。"""

    alpha_values: Tuple[float, ...] = (0.4, 0.5, 0.6)
    seeds: Tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)
    burn_in: int = 20_000
    observation_length: int = 100_000
    tm_max_order: int = 8
    diagnostic_sample_size: int = 20_000
    quadrature_points: int = 200_000
    thresholds: ValidationThresholds = ValidationThresholds()

    def validate(self) -> None:
        """写像と数値計算が意味を持つ範囲へ設定を制限する。"""

        if not self.alpha_values:
            raise ValueError("alpha_values は空にできません。")
        if any(not 0.0 < alpha < 1.0 for alpha in self.alpha_values):
            raise ValueError("alpha は 0 < alpha < 1 を満たす必要があります。")
        if not self.seeds:
            raise ValueError("seeds は空にできません。")
        if self.burn_in < 0 or self.observation_length <= 0:
            raise ValueError("burn_in は非負、observation_length は正にしてください。")
        if self.tm_max_order <= 0:
            raise ValueError("tm_max_order は正にしてください。")
        if not 0 < self.diagnostic_sample_size <= self.observation_length:
            raise ValueError("diagnostic_sample_size は観測長以下の正数にしてください。")


@dataclass(frozen=True)
class OrbitMetrics:
    """一つのalphaとseedから得た検証指標を保持する。"""

    alpha: float
    seed: int
    theoretical_scale: float
    estimated_scale: float
    scale_relative_error: float
    theoretical_lyapunov: float
    estimated_lyapunov: float
    lyapunov_absolute_error: float
    ks_distance: float
    tm_max_nonzero_magnitude: float
    minimum_absolute_state: float
    maximum_absolute_state: float
    exact_repeat_count: int
    finite: bool


def generalized_boole_map(state: float, alpha: float) -> float:
    """実数状態へ一般化Boole写像 F_alpha(x)=alpha(x-1/x) を適用する。"""

    if state == 0.0:
        raise ZeroDivisionError("状態が厳密に0となり、一般化Boole写像を評価できません。")
    return alpha * (state - 1.0 / state)


def generalized_boole_derivative(state: FloatArray, alpha: float) -> FloatArray:
    """軌道上でLyapunov指数を評価するため導関数を計算する。"""

    return alpha * (1.0 + 1.0 / np.square(state))


def theoretical_cauchy_scale(alpha: float) -> float:
    """非結合一般化Boole写像の理論Cauchy尺度を返す。"""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha は 0 < alpha < 1 を満たす必要があります。")
    return math.sqrt(alpha / (1.0 - alpha))


def theoretical_lyapunov(alpha: float) -> float:
    """非結合一般化Boole写像の閉形式Lyapunov指数を返す。"""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha は 0 < alpha < 1 を満たす必要があります。")
    return 2.0 * math.log(math.sqrt(alpha) + math.sqrt(1.0 - alpha))


def sample_cauchy_initial_state(rng: np.random.Generator, scale: float) -> float:
    """逆変換法で中心0のCauchy分布からゼロでない初期値を生成する。"""

    for _ in range(100):
        uniform_value = float(rng.random())
        state = scale * math.tan(math.pi * (uniform_value - 0.5))
        if state != 0.0 and math.isfinite(state):
            return state
    raise RuntimeError("有限かつゼロでないCauchy初期値を生成できませんでした。")


def generate_orbit(
    alpha: float,
    seed: int,
    burn_in: int,
    observation_length: int,
) -> FloatArray:
    """理論Cauchy測度から開始し、burn-in後の決定論的軌道を生成する。"""

    rng = np.random.default_rng(seed)
    state = sample_cauchy_initial_state(rng, theoretical_cauchy_scale(alpha))
    for _ in range(burn_in):
        state = generalized_boole_map(state, alpha)
        if not math.isfinite(state):
            raise FloatingPointError("burn-in中に非有限状態が発生しました。")

    orbit = np.empty(observation_length, dtype=np.float64)
    for index in range(observation_length):
        state = generalized_boole_map(state, alpha)
        if not math.isfinite(state):
            raise FloatingPointError(f"観測index={index}で非有限状態が発生しました。")
        orbit[index] = state
    return orbit


def estimate_cauchy_scale(orbit: FloatArray) -> float:
    """Cauchy分布の四分位幅から尺度をロバストに推定する。"""

    first_quartile, third_quartile = np.quantile(orbit, [0.25, 0.75])
    return float((third_quartile - first_quartile) / 2.0)


def cauchy_cdf(values: FloatArray, scale: float) -> FloatArray:
    """中心0のCauchy分布の累積分布関数を評価する。"""

    return 0.5 + np.arctan(values / scale) / math.pi


def cauchy_ks_distance(orbit: FloatArray, scale: float) -> float:
    """経験分布と理論Cauchy分布の両側Kolmogorov-Smirnov距離を求める。"""

    sorted_values = np.sort(orbit)
    theoretical_cdf = cauchy_cdf(sorted_values, scale)
    sample_size = sorted_values.size
    upper_empirical = np.arange(1, sample_size + 1, dtype=np.float64) / sample_size
    lower_empirical = np.arange(0, sample_size, dtype=np.float64) / sample_size
    return float(
        max(
            np.max(upper_empirical - theoretical_cdf),
            np.max(theoretical_cdf - lower_empirical),
        )
    )


def estimate_lyapunov(orbit: FloatArray, alpha: float) -> float:
    """軌道平均 log|F'(x)| から有限時間Lyapunov指数を推定する。"""

    derivative = generalized_boole_derivative(orbit, alpha)
    return float(np.mean(np.log(np.abs(derivative))))


def tm_coefficients(orbit: FloatArray, scale: float, max_order: int) -> Dict[int, complex]:
    """Cayley変換後の有限TMモード係数を軌道平均で推定する。"""

    cayley_values = (orbit - 1j * scale) / (orbit + 1j * scale)
    return {
        order: complex(np.mean(np.power(cayley_values, order)))
        for order in range(1, max_order + 1)
    }


def quadrature_lyapunov(alpha: float, point_count: int) -> float:
    """Cauchy変数の角度表示を用いてLyapunov積分を独立に数値積分する。"""

    scale = theoretical_cauchy_scale(alpha)
    # Why not: 無限区間を直接切るとCauchy尾部の打ち切り誤差が支配するため、等確率の角度で積分する。
    angles = -math.pi / 2.0 + (np.arange(point_count) + 0.5) * math.pi / point_count
    states = scale * np.tan(angles)
    log_derivative = np.log(np.abs(generalized_boole_derivative(states, alpha)))
    return float(np.mean(log_derivative))


def calculate_orbit_metrics(alpha: float, seed: int, config: ExperimentConfig) -> Tuple[OrbitMetrics, FloatArray]:
    """一軌道を生成し、理論値との差とTM係数をまとめて計算する。"""

    orbit = generate_orbit(alpha, seed, config.burn_in, config.observation_length)
    scale = theoretical_cauchy_scale(alpha)
    estimated_scale = estimate_cauchy_scale(orbit)
    lyapunov = theoretical_lyapunov(alpha)
    estimated_lyapunov = estimate_lyapunov(orbit, alpha)
    coefficients = tm_coefficients(orbit, scale, config.tm_max_order)
    metrics = OrbitMetrics(
        alpha=alpha,
        seed=seed,
        theoretical_scale=scale,
        estimated_scale=estimated_scale,
        scale_relative_error=abs(estimated_scale - scale) / scale,
        theoretical_lyapunov=lyapunov,
        estimated_lyapunov=estimated_lyapunov,
        lyapunov_absolute_error=abs(estimated_lyapunov - lyapunov),
        ks_distance=cauchy_ks_distance(orbit, scale),
        tm_max_nonzero_magnitude=max(abs(value) for value in coefficients.values()),
        minimum_absolute_state=float(np.min(np.abs(orbit))),
        maximum_absolute_state=float(np.max(np.abs(orbit))),
        exact_repeat_count=int(orbit.size - np.unique(orbit).size),
        finite=bool(np.all(np.isfinite(orbit))),
    )
    return metrics, orbit


def summarize_results(metrics: List[OrbitMetrics], config: ExperimentConfig) -> Dict[str, object]:
    """seed別指標をalpha別に集約し、事前定義したゲートを判定する。"""

    per_alpha: List[Dict[str, object]] = []
    for alpha in config.alpha_values:
        alpha_metrics = [metric for metric in metrics if metric.alpha == alpha]
        quadrature_value = quadrature_lyapunov(alpha, config.quadrature_points)
        quadrature_error = abs(quadrature_value - theoretical_lyapunov(alpha))
        row: Dict[str, object] = {
            "alpha": alpha,
            "seed_count": len(alpha_metrics),
            "theoretical_scale": theoretical_cauchy_scale(alpha),
            "median_scale_relative_error": float(np.median([item.scale_relative_error for item in alpha_metrics])),
            "max_scale_relative_error": max(item.scale_relative_error for item in alpha_metrics),
            "theoretical_lyapunov": theoretical_lyapunov(alpha),
            "median_lyapunov_absolute_error": float(np.median([item.lyapunov_absolute_error for item in alpha_metrics])),
            "max_lyapunov_absolute_error": max(item.lyapunov_absolute_error for item in alpha_metrics),
            "max_ks_distance": max(item.ks_distance for item in alpha_metrics),
            "max_tm_nonzero_magnitude": max(item.tm_max_nonzero_magnitude for item in alpha_metrics),
            "quadrature_lyapunov": quadrature_value,
            "quadrature_lyapunov_error": quadrature_error,
            "all_states_finite": all(item.finite for item in alpha_metrics),
            "total_exact_repeat_count": sum(item.exact_repeat_count for item in alpha_metrics),
        }
        thresholds = config.thresholds
        row["gates"] = {
            "scale": row["max_scale_relative_error"] <= thresholds.max_scale_relative_error,
            "lyapunov": row["max_lyapunov_absolute_error"] <= thresholds.max_lyapunov_absolute_error,
            "ks": row["max_ks_distance"] <= thresholds.max_ks_distance,
            "tm": row["max_tm_nonzero_magnitude"] <= thresholds.max_tm_nonzero_magnitude,
            "quadrature": row["quadrature_lyapunov_error"] <= thresholds.max_quadrature_lyapunov_error,
            "finite": row["all_states_finite"],
        }
        row["passed"] = all(bool(value) for value in row["gates"].values())
        per_alpha.append(row)

    return {
        "schema_version": 1,
        "status": "completed",
        "question": "一般化Boole写像のCauchy不変測度・Lyapunov指数・TM非零モード0を有限軌道で再現できるか。",
        "config": {
            **asdict(config),
            "alpha_values": list(config.alpha_values),
            "seeds": list(config.seeds),
        },
        "per_alpha": per_alpha,
        "overall_passed": all(bool(row["passed"]) for row in per_alpha),
        "caveats": [
            "合格ゲートは実装回帰を検出する工学的閾値であり、統計的有意水準ではない。",
            "TM係数の検証は簡略化したCayleyモード q_gamma(x)^k に対するもので、一般のTM系全体の検証ではない。",
            "有限精度で重複がないことは、長時間周期化が存在しない証明ではない。",
        ],
    }


def write_per_seed_csv(metrics: List[OrbitMetrics], output_path: Path) -> None:
    """seed別指標を監査可能なCSVとして保存する。"""

    rows = [asdict(metric) for metric in metrics]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(run_directory: Path, config: ExperimentConfig) -> Dict[str, object]:
    """全alpha・seedを実行し、指標・診断標本・環境をrun内へ保存する。"""

    config.validate()
    artifacts_directory = run_directory / "artifacts"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    metrics: List[OrbitMetrics] = []
    diagnostic_samples: Dict[str, FloatArray] = {}

    for alpha in config.alpha_values:
        for seed_index, seed in enumerate(config.seeds):
            orbit_metrics, orbit = calculate_orbit_metrics(alpha, seed, config)
            metrics.append(orbit_metrics)
            if seed_index == 0:
                sample_indices = np.linspace(
                    0,
                    orbit.size - 1,
                    config.diagnostic_sample_size,
                    dtype=np.int64,
                )
                diagnostic_samples[f"alpha_{alpha:.2f}"] = orbit[sample_indices]

    summary = summarize_results(metrics, config)
    write_per_seed_csv(metrics, artifacts_directory / "per_seed_metrics.csv")
    np.savez_compressed(artifacts_directory / "diagnostic_orbit_samples.npz", **diagnostic_samples)
    (artifacts_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_record = {
        "schema_version": 1,
        "status": "completed",
        "primary": {
            "overall_passed": summary["overall_passed"],
            "alpha_count": len(config.alpha_values),
            "seed_count_per_alpha": len(config.seeds),
        },
        "secondary": {
            "summary_artifact": "artifacts/summary.json",
            "per_seed_artifact": "artifacts/per_seed_metrics.csv",
            "diagnostic_samples": "artifacts/diagnostic_orbit_samples.npz",
        },
        "per_seed_artifact": "artifacts/per_seed_metrics.csv",
        "notes": summary["caveats"],
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics_record, ensure_ascii=False, indent=2) + "\n",
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
            "execution_command": "python e0_boole_validation.py --run-directory .",
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
    """E0実験を実行し、最終ゲート判定を表示する。"""

    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    run_directory = args.run_directory.resolve()
    summary = run_experiment(run_directory, ExperimentConfig())
    print(json.dumps({"overall_passed": summary["overall_passed"]}, ensure_ascii=False))
    return 0 if summary["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
