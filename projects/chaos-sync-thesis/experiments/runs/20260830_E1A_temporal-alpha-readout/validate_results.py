"""E1A保存成果物を実験実装とは別経路で再計算し検証する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class ValidationCheck:
    """一つの保存成果物検証と、その判定根拠を保持する。"""

    name: str
    passed: bool
    detail: str


def load_csv(path: Path) -> List[Dict[str, str]]:
    """UTF-8-SIGの実験表を行辞書として読む。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def calculate_rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    """保存予測からRMSEを独立に再計算する。"""

    return float(np.sqrt(np.mean(np.square(targets - predictions))))


def validate(run_directory: Path) -> Dict[str, object]:
    """設定、split、表の粒度、RMSE、選択規則、成功ゲートを照合する。"""

    artifacts_directory = run_directory / "artifacts"
    summary = json.loads((artifacts_directory / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    environment = json.loads((run_directory / "environment.json").read_text(encoding="utf-8"))
    metric_rows = load_csv(artifacts_directory / "per_feature_metrics.csv")
    prediction_rows = load_csv(artifacts_directory / "test_predictions.csv")
    checks: List[ValidationCheck] = []

    actual = config["actual_run_config"]
    train_alphas = {float(value) for value in actual["train_alphas"]}
    test_alphas = {float(value) for value in actual["test_alphas"]}
    train_seeds = {int(value) for value in actual["train_seeds"]}
    validation_seeds = {int(value) for value in actual["validation_seeds"]}
    test_seeds = {int(value) for value in actual["test_seeds"]}
    checks.append(
        ValidationCheck(
            "split_disjoint",
            not (train_alphas & test_alphas)
            and not (train_seeds & validation_seeds)
            and not (train_seeds & test_seeds)
            and not (validation_seeds & test_seeds),
            "train/test alphaと全seed splitの非重複を確認",
        )
    )
    checks.append(
        ValidationCheck(
            "test_row_grain",
            len(prediction_rows) == len(test_alphas) * len(test_seeds)
            and {(float(row["alpha"]), int(row["seed"])) for row in prediction_rows}
            == {(alpha, seed) for alpha in test_alphas for seed in test_seeds},
            f"rows={len(prediction_rows)}, expected={len(test_alphas) * len(test_seeds)}",
        )
    )

    recorded_by_feature = {row["feature"]: row for row in summary["per_feature"]}
    csv_by_feature = {row["feature"]: row for row in metric_rows}
    checks.append(
        ValidationCheck(
            "feature_set",
            set(recorded_by_feature) == set(csv_by_feature)
            and len(recorded_by_feature) == 7,
            f"features={sorted(recorded_by_feature)}",
        )
    )
    targets = np.asarray([float(row["alpha"]) for row in prediction_rows])
    for feature_name, recorded in recorded_by_feature.items():
        predictions = np.asarray(
            [float(row[f"prediction_{feature_name}"]) for row in prediction_rows]
        )
        recomputed = calculate_rmse(targets, predictions)
        csv_value = float(csv_by_feature[feature_name]["test_rmse"])
        recorded_value = float(recorded["test_rmse"])
        checks.append(
            ValidationCheck(
                f"test_rmse_{feature_name}",
                math.isclose(recomputed, recorded_value, rel_tol=0.0, abs_tol=1.0e-15)
                and math.isclose(csv_value, recorded_value, rel_tol=0.0, abs_tol=1.0e-15),
                f"recomputed={recomputed:.16g}, recorded={recorded_value:.16g}",
            )
        )

    baseline_names = {
        "robust_quantiles",
        "raw_window",
        "fourier_bands",
        "cayley_delay_k1",
    }
    expected_baseline = min(
        baseline_names,
        key=lambda name: float(recorded_by_feature[name]["validation_rmse"]),
    )
    recorded_baseline = summary["best_baseline_selected_on_validation"]
    checks.append(
        ValidationCheck(
            "validation_selected_baseline",
            recorded_baseline == expected_baseline,
            f"expected={expected_baseline}, recorded={recorded_baseline}",
        )
    )

    tm_rmse = float(recorded_by_feature["tm_delay"]["test_rmse"])
    baseline_rmse = float(recorded_by_feature[recorded_baseline]["test_rmse"])
    point_difference = baseline_rmse - tm_rmse
    recorded_difference = float(summary["paired_seed_bootstrap"]["point_difference"])
    ci_lower = float(summary["paired_seed_bootstrap"]["ci_95_lower"])
    checks.append(
        ValidationCheck(
            "success_gate",
            math.isclose(point_difference, recorded_difference, rel_tol=0.0, abs_tol=1.0e-15)
            and bool(summary["success"]) == (ci_lower > 0.0)
            and bool(metrics["primary"]["success"]) == bool(summary["success"]),
            f"difference={point_difference:.16g}, ci_lower={ci_lower:.16g}, success={summary['success']}",
        )
    )
    checks.append(
        ValidationCheck(
            "environment",
            any(str(item).startswith("numpy==") for item in environment["dependencies"])
            and bool(environment["python_executable"])
            and bool(environment["execution_command"]),
            f"dependencies={environment['dependencies']}",
        )
    )

    failed = [asdict(check) for check in checks if not check.passed]
    return {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": [asdict(check) for check in checks],
        "failed_checks": failed,
    }


def parse_arguments() -> argparse.Namespace:
    """検証対象runディレクトリを受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> int:
    """独立検証を実行し、run内へ結果を保存する。"""

    run_directory = parse_arguments().run_directory.resolve()
    result = validate(run_directory)
    output_path = run_directory / "artifacts" / "validation.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "check_count": result["check_count"]}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
