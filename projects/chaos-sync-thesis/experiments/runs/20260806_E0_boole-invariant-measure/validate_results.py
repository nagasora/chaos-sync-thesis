"""E0保存成果物を実験コードとは別経路で再集計し、整合性を検証する。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ValidationCheck:
    """一つの独立検証項目と判定結果を保持する。"""

    name: str
    passed: bool
    detail: str


def load_per_seed_rows(path: Path) -> List[Dict[str, str]]:
    """seed別CSVを行辞書として読み込む。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def group_rows_by_alpha(rows: List[Dict[str, str]]) -> Dict[float, List[Dict[str, str]]]:
    """独立再集計のためCSV行をalpha単位へ分割する。"""

    grouped: Dict[float, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["alpha"]), []).append(row)
    return grouped


def check_notebook(notebook_path: Path) -> Tuple[int, int, int]:
    """Notebookの全コードセル実行、エラー出力、図出力をJSONから確認する。"""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    executed_count = sum(cell.get("execution_count") is not None for cell in code_cells)
    error_count = sum(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    return executed_count, error_count, image_count


def validate(run_directory: Path) -> Dict[str, object]:
    """件数、集約値、ゲート、診断標本、Notebookを独立に照合する。"""

    artifacts_directory = run_directory / "artifacts"
    summary = json.loads((artifacts_directory / "summary.json").read_text(encoding="utf-8"))
    rows = load_per_seed_rows(artifacts_directory / "per_seed_metrics.csv")
    grouped = group_rows_by_alpha(rows)
    expected_alphas = [float(value) for value in summary["config"]["alpha_values"]]
    expected_seeds = [int(value) for value in summary["config"]["seeds"]]
    checks: List[ValidationCheck] = []

    experiment_config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    metrics_record = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    environment_record = json.loads((run_directory / "environment.json").read_text(encoding="utf-8"))
    checks.append(
        ValidationCheck(
            "experiment_config",
            experiment_config["status"] == "completed"
            and [float(value) for value in experiment_config["data"]["alpha_values"]]
            == expected_alphas
            and [int(value) for value in experiment_config["seeds"]] == expected_seeds
            and experiment_config["thresholds"] == summary["config"]["thresholds"],
            "status、alpha、seed、閾値をsummaryと照合",
        )
    )
    checks.append(
        ValidationCheck(
            "metrics_record",
            metrics_record["status"] == "completed"
            and bool(metrics_record["primary"]["overall_passed"])
            == bool(summary["overall_passed"]),
            "metrics.jsonの状態と総合判定をsummaryと照合",
        )
    )
    dependency_names = {str(value).split("==")[0] for value in environment_record["dependencies"]}
    checks.append(
        ValidationCheck(
            "environment_record",
            {"numpy", "nbformat", "nbclient", "matplotlib", "ipykernel"}.issubset(
                dependency_names
            )
            and bool(environment_record["python_executable"]),
            f"dependencies={sorted(dependency_names)}",
        )
    )

    unique_keys = {(float(row["alpha"]), int(row["seed"])) for row in rows}
    expected_keys = {(alpha, seed) for alpha in expected_alphas for seed in expected_seeds}
    checks.append(
        ValidationCheck(
            "row_grain",
            unique_keys == expected_keys and len(rows) == len(expected_keys),
            f"expected={len(expected_keys)}, actual={len(rows)}, unique={len(unique_keys)}",
        )
    )
    checks.append(
        ValidationCheck(
            "finite_states",
            all(row["finite"].lower() == "true" for row in rows),
            "全seed行のfinite列を確認",
        )
    )

    summary_by_alpha = {float(row["alpha"]): row for row in summary["per_alpha"]}
    aggregate_fields = [
        "scale_relative_error",
        "lyapunov_absolute_error",
        "ks_distance",
        "tm_max_nonzero_magnitude",
    ]
    summary_fields = [
        "max_scale_relative_error",
        "max_lyapunov_absolute_error",
        "max_ks_distance",
        "max_tm_nonzero_magnitude",
    ]
    for alpha in expected_alphas:
        alpha_rows = grouped[alpha]
        alpha_summary = summary_by_alpha[alpha]
        for csv_field, summary_field in zip(aggregate_fields, summary_fields):
            recomputed = max(float(row[csv_field]) for row in alpha_rows)
            recorded = float(alpha_summary[summary_field])
            checks.append(
                ValidationCheck(
                    f"aggregate_alpha_{alpha}_{csv_field}",
                    math.isclose(recomputed, recorded, rel_tol=0.0, abs_tol=1.0e-15),
                    f"recomputed={recomputed:.16g}, recorded={recorded:.16g}",
                )
            )

    thresholds = summary["config"]["thresholds"]
    threshold_mapping = {
        "scale": ("max_scale_relative_error", "max_scale_relative_error"),
        "lyapunov": ("max_lyapunov_absolute_error", "max_lyapunov_absolute_error"),
        "ks": ("max_ks_distance", "max_ks_distance"),
        "tm": ("max_tm_nonzero_magnitude", "max_tm_nonzero_magnitude"),
        "quadrature": ("quadrature_lyapunov_error", "max_quadrature_lyapunov_error"),
    }
    for alpha, alpha_summary in summary_by_alpha.items():
        for gate_name, (metric_name, threshold_name) in threshold_mapping.items():
            recomputed_gate = float(alpha_summary[metric_name]) <= float(thresholds[threshold_name])
            recorded_gate = bool(alpha_summary["gates"][gate_name])
            checks.append(
                ValidationCheck(
                    f"gate_alpha_{alpha}_{gate_name}",
                    recomputed_gate == recorded_gate,
                    f"metric={alpha_summary[metric_name]}, threshold={thresholds[threshold_name]}",
                )
            )

    alpha_half = summary_by_alpha[0.5]
    checks.append(
        ValidationCheck(
            "alpha_half_special_values",
            math.isclose(float(alpha_half["theoretical_scale"]), 1.0, abs_tol=1.0e-15)
            and math.isclose(
                float(alpha_half["theoretical_lyapunov"]),
                math.log(2.0),
                abs_tol=1.0e-15,
            ),
            "alpha=0.5でscale=1、Lyapunov=log(2)を確認",
        )
    )

    samples = np.load(artifacts_directory / "diagnostic_orbit_samples.npz")
    sample_keys = sorted(samples.files)
    sample_lengths = {key: int(samples[key].size) for key in sample_keys}
    sample_finite = all(bool(np.all(np.isfinite(samples[key]))) for key in sample_keys)
    checks.append(
        ValidationCheck(
            "diagnostic_samples",
            sample_keys == ["alpha_0.40", "alpha_0.50", "alpha_0.60"]
            and set(sample_lengths.values()) == {20_000}
            and sample_finite,
            f"keys={sample_keys}, lengths={sample_lengths}, finite={sample_finite}",
        )
    )

    executed_count, error_count, image_count = check_notebook(
        run_directory / "E0_boole_validation.ipynb"
    )
    checks.append(
        ValidationCheck(
            "executed_notebook",
            executed_count == 5 and error_count == 0 and image_count == 1,
            f"executed_code_cells={executed_count}, errors={error_count}, images={image_count}",
        )
    )
    chart_path = artifacts_directory / "e0_metric_gates.png"
    checks.append(
        ValidationCheck(
            "exported_chart",
            chart_path.is_file() and chart_path.stat().st_size > 10_000,
            f"path={chart_path.name}, bytes={chart_path.stat().st_size if chart_path.exists() else 0}",
        )
    )

    checks.append(
        ValidationCheck(
            "overall_gate",
            bool(summary["overall_passed"])
            == all(bool(row["passed"]) for row in summary["per_alpha"]),
            f"recorded={summary['overall_passed']}",
        )
    )

    failed_checks = [asdict_check(check) for check in checks if not check.passed]
    return {
        "schema_version": 1,
        "status": "passed" if not failed_checks else "failed",
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": [asdict_check(check) for check in checks],
        "failed_checks": failed_checks,
    }


def asdict_check(check: ValidationCheck) -> Dict[str, object]:
    """検証項目をJSONへ保存できる辞書へ変換する。"""

    return {"name": check.name, "passed": check.passed, "detail": check.detail}


def main() -> int:
    """run成果物を検証し、機械可読な検証記録を保存する。"""

    run_directory = Path(__file__).resolve().parent
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
