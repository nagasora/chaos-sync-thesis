from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import write_json


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, (int, str, bool)):
        return str(value)
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def build_digest(root: Path) -> dict[str, Any]:
    """How: 各runの数値正本から、人が読める統合要約を再生成する。"""
    summaries = json.loads((root / "all_experiment_summaries.json").read_text(encoding="utf-8"))
    e2_metrics = pd.read_csv(root / "runs/20260904_E2A_boole_observation_robustness/artifacts/robustness_metrics.csv")
    e3b_metrics = pd.read_csv(root / "runs/20260904_E3B_tangent_two_node_basin/artifacts/basin_metrics.csv")
    e3c_metrics = pd.read_csv(root / "runs/20260904_E3C_boole_n8_input_retention/artifacts/input_retention_metrics.csv")
    e4_metrics = pd.read_csv(root / "runs/20260904_E4A_boole_synthetic_signal/artifacts/rate_distortion_metrics.csv")
    e5_metrics = pd.read_csv(root / "runs/20260904_E5A_boole_small_image_digits/artifacts/image_metrics.csv")

    e2_conditions: list[dict[str, Any]] = []
    for condition_type, value in (
        ("gaussian_noise_sigma", 0.0),
        ("gaussian_noise_sigma", 0.1),
        ("gaussian_noise_sigma", 0.3),
        ("missing_rate", 0.2),
        ("missing_rate", 0.4),
        ("observation_length", 128.0),
        ("observation_length", 256.0),
    ):
        subset = e2_metrics[(e2_metrics.condition_type == condition_type) & (e2_metrics.condition_value == value)]
        e2_conditions.extend(subset[["condition_type", "condition_value", "model", "feature_dim", "test_rmse"]].to_dict(orient="records"))

    e3b_key: list[dict[str, Any]] = []
    for beta in sorted(e3b_metrics.beta.unique()):
        for form in ("output_cross", "state_diffusive"):
            subset = e3b_metrics[(e3b_metrics.beta == beta) & (e3b_metrics.coupling_form == form)]
            stable = subset[subset.local_stable]
            best = subset.sort_values("basin_success_rate", ascending=False).iloc[0]
            e3b_key.append(
                {
                    "beta": beta,
                    "form": form,
                    "local_stable_count": int(len(stable)),
                    "min_local_stable_coupling": None if stable.empty else float(stable.coupling.min()),
                    "max_local_stable_coupling": None if stable.empty else float(stable.coupling.max()),
                    "best_basin_coupling": float(best.coupling),
                    "best_basin_success": float(best.basin_success_rate),
                    "best_divergence_rate": float(best.divergence_rate),
                }
            )

    e3c_tm = e3c_metrics[e3c_metrics.feature == "tm_graph_late"].sort_values("coupling")
    e3c_key = e3c_tm[
        ["coupling", "test_balanced_accuracy", "median_sync_rms", "effective_rank"]
    ].to_dict(orient="records")
    e4_d16 = e4_metrics[e4_metrics.dimension == 16][
        ["representation", "test_nmse", "test_rmse", "validation_nmse"]
    ].sort_values("test_nmse").to_dict(orient="records")
    e5_d16 = e5_metrics[e5_metrics.dimension == 16][
        [
            "representation",
            "test_nmse",
            "test_psnr",
            "test_ssim",
            "classification_balanced_accuracy",
        ]
    ].sort_values("test_nmse").to_dict(orient="records")

    digest = {
        "suite_status": {key: value["status"] for key, value in summaries.items()},
        "e2_conditions": e2_conditions,
        "e2_paired_comparisons": summaries["e2"]["paired_comparisons"],
        "e3b_key": e3b_key,
        "e3b_mismatch_fraction": summaries["e3b"]["fraction_of_locally_stable_grid_with_basin_below_0_9"],
        "e3c_key": e3c_key,
        "e3c_best_postcritical": summaries["e3c"]["best_postcritical_tm"],
        "e3c_scientific_gates": summaries["e3c"]["scientific_gates"],
        "e4_selected": {
            "coupling": summaries["e4"]["selected_coupling"],
            "input_gain": summaries["e4"]["selected_input_gain"],
            "validation_nmse": summaries["e4"]["selected_validation_nmse"],
        },
        "e4_d16": e4_d16,
        "e4_scientific_gates": summaries["e4"]["scientific_gates"],
        "e5_d16": e5_d16,
        "e5_scientific_gates": summaries["e5"]["scientific_gates"],
    }
    write_json(root / "RESULTS_DIGEST.json", digest)

    lines = [
        "# 統一実験スイート結果要約",
        "",
        "## 完了状態",
        "",
    ]
    for key, status in digest["suite_status"].items():
        lines.append(f"- {key}: `{status}`")
    lines.extend(["", "## E2A 頑健性（抜粋）", "", "| 条件 | 値 | model | dim | RMSE |", "|---|---:|---|---:|---:|"])
    for row in e2_conditions:
        lines.append(
            f"| {row['condition_type']} | {_fmt(row['condition_value'])} | {row['model']} | {row['feature_dim']} | {_fmt(row['test_rmse'])} |"
        )
    lines.extend(["", "## E3B 局所安定性と大域basin", "", "| beta | form | local stable grid | range | best basin |", "|---:|---|---:|---|---:|"])
    for row in e3b_key:
        range_text = f"{_fmt(row['min_local_stable_coupling'])}–{_fmt(row['max_local_stable_coupling'])}"
        lines.append(
            f"| {_fmt(row['beta'])} | {row['form']} | {row['local_stable_count']} | {range_text} | {_fmt(row['best_basin_success'])} |"
        )
    lines.extend(["", "## E3C 同期と入力保持（TM）", "", "| K | balanced acc | median sync RMS | effective rank |", "|---:|---:|---:|---:|"])
    for row in e3c_key:
        lines.append(
            f"| {_fmt(row['coupling'])} | {_fmt(row['test_balanced_accuracy'])} | {_fmt(row['median_sync_rms'])} | {_fmt(row['effective_rank'])} |"
        )
    lines.extend(["", "## E4A d=16", "", "| representation | test NMSE | test RMSE |", "|---|---:|---:|"])
    for row in e4_d16:
        lines.append(f"| {row['representation']} | {_fmt(row['test_nmse'])} | {_fmt(row['test_rmse'])} |")
    lines.extend(["", "## E5A d=16", "", "| representation | NMSE | PSNR | SSIM | class balanced acc |", "|---|---:|---:|---:|---:|"])
    for row in e5_d16:
        lines.append(
            f"| {row['representation']} | {_fmt(row['test_nmse'])} | {_fmt(row['test_psnr'])} | {_fmt(row['test_ssim'])} | {_fmt(row['classification_balanced_accuracy'])} |"
        )
    (root / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return digest
