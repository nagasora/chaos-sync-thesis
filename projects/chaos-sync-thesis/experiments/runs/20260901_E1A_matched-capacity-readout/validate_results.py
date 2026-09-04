"""E1A容量一致追試の保存成果物を独立に再計算して検証する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Check:
    """一つの独立検証結果を機械可読に保持する。"""

    name: str
    passed: bool
    detail: str


def read_csv(path: Path) -> List[Dict[str, str]]:
    """UTF-8 CSVを辞書行として読み込む。"""

    # Why not: pilotのCSV writerはExcel互換のBOMを付けるため、utf-8では先頭列名が変わる。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rmse(targets: FloatArray, predictions: FloatArray) -> float:
    """保存予測からRMSEを独立に再計算する。"""

    return float(np.sqrt(np.mean(np.square(targets - predictions))))


def file_sha256(path: Path) -> str:
    """保存済みコードハッシュと比較するSHA-256を計算する。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_check(checks: List[Check], name: str, passed: bool, detail: str) -> None:
    """検証名、成否、根拠を一行へ追加する。"""

    checks.append(Check(name=name, passed=bool(passed), detail=detail))


def validate_results(run_directory: Path) -> Dict[str, object]:
    """JSON、CSV、NPZ、PNGを相互照合しvalidation.jsonを保存する。"""

    artifacts = run_directory / "artifacts"
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads((run_directory / "environment.json").read_text(encoding="utf-8"))
    checks: List[Check] = []

    statuses = (config["status"], metrics["status"], summary["status"])
    append_check(
        checks,
        "completed_status",
        statuses == ("completed", "completed", "completed"),
        f"statuses={statuses}",
    )

    data_config = config["data"]
    fit_seeds = {int(seed) for seed in data_config["fit_seeds"]}
    test_seeds = {int(seed) for seed in data_config["test_seeds"]}
    parent_summary_path = (
        run_directory.parent
        / "20260830_E1A_temporal-alpha-replication"
        / "artifacts"
        / "summary.json"
    )
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_test_seeds = {int(seed) for seed in parent_summary["config"]["test_seeds"]}
    split_passed = not (fit_seeds & test_seeds) and not (parent_test_seeds & test_seeds)
    append_check(
        checks,
        "seed_split_disjoint",
        split_passed,
        f"fit={len(fit_seeds)}, test={len(test_seeds)}, parent_overlap={len(parent_test_seeds & test_seeds)}",
    )

    with np.load(artifacts / "features.npz", allow_pickle=False) as archive:
        split = np.asarray(archive["split"], dtype=np.str_)
        alpha = np.asarray(archive["alpha"], dtype=np.float64)
        seed = np.asarray(archive["seed"], dtype=np.int64)
        feature_arrays = {
            name.removeprefix("feature__"): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith("feature__")
        }
    expected_fit_rows = len(data_config["fit_alphas"]) * len(fit_seeds)
    expected_test_rows = len(data_config["test_alphas"]) * len(test_seeds)
    row_passed = (
        int(np.sum(split == "fit")) == expected_fit_rows
        and int(np.sum(split == "test")) == expected_test_rows
        and alpha.size == expected_fit_rows + expected_test_rows
    )
    append_check(
        checks,
        "feature_store_row_grain",
        row_passed,
        f"fit={np.sum(split == 'fit')}, test={np.sum(split == 'test')}",
    )

    expected_dimensions = {
        "fourier_16": 16,
        "tm_full_80": 80,
        "tm_instant_16": 16,
        **{f"tm16_{name}": 16 for name in config["tm_candidate_specs"]},
    }
    feature_passed = (
        set(feature_arrays) == set(expected_dimensions)
        and all(feature_arrays[name].shape == (alpha.size, dimension) for name, dimension in expected_dimensions.items())
        and all(np.all(np.isfinite(values)) for values in feature_arrays.values())
    )
    append_check(
        checks,
        "feature_store_dimensions",
        feature_passed,
        f"dimensions={ {name: values.shape[1] for name, values in feature_arrays.items()} }",
    )

    manifest_rows = read_csv(artifacts / "trajectory_manifest.csv")
    manifest_passed = len(manifest_rows) == alpha.size and all(
        row["split"] == split[index]
        and float(row["alpha"]) == alpha[index]
        and int(row["seed"]) == seed[index]
        for index, row in enumerate(manifest_rows)
    )
    append_check(
        checks,
        "trajectory_manifest",
        manifest_passed,
        f"rows={len(manifest_rows)}",
    )

    fold_rows = read_csv(artifacts / "fold_assignments.csv")
    fold_counts: Dict[int, int] = {}
    fold_seed_set = set()
    for row in fold_rows:
        fold = int(row["fold"])
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
        fold_seed_set.add(int(row["seed"]))
    fold_passed = (
        fold_seed_set == fit_seeds
        and set(fold_counts) == set(range(int(config["training"]["cv_folds"])))
        and max(fold_counts.values()) - min(fold_counts.values()) <= 1
    )
    append_check(
        checks,
        "grouped_fold_assignment",
        fold_passed,
        f"fold_counts={fold_counts}",
    )

    candidate_rows = read_csv(artifacts / "candidate_cv_metrics.csv")
    selected_candidates = [row for row in candidate_rows if row["selected"].lower() == "true"]
    minimum_candidate = min(candidate_rows, key=lambda row: float(row["cv_rmse"]))
    candidate_passed = (
        len(candidate_rows)
        == len(config["tm_candidate_specs"]) * len(config["training"]["ridge_penalties"])
        and len(selected_candidates) == 1
        and selected_candidates[0]["candidate"] == minimum_candidate["candidate"]
        and float(selected_candidates[0]["cv_rmse"]) == float(minimum_candidate["cv_rmse"])
        and summary["selected_tm_candidate"] == minimum_candidate["candidate"]
    )
    append_check(
        checks,
        "tm_candidate_selection",
        candidate_passed,
        f"selected={summary['selected_tm_candidate']}, cv_rmse={minimum_candidate['cv_rmse']}",
    )

    model_rows = read_csv(artifacts / "model_metrics.csv")
    model_by_name = {row["model"]: row for row in model_rows}
    dimension_passed = (
        set(model_by_name)
        == {"tm_selected_16", "fourier_16", "tm_full_80", "tm_instant_16"}
        and int(model_by_name["tm_selected_16"]["feature_dimension"]) == 16
        and int(model_by_name["fourier_16"]["feature_dimension"]) == 16
        and int(model_by_name["tm_full_80"]["feature_dimension"]) == 80
        and int(model_by_name["tm_instant_16"]["feature_dimension"]) == 16
    )
    append_check(
        checks,
        "matched_feature_dimensions",
        dimension_passed,
        f"dimensions={ {name: row['feature_dimension'] for name, row in model_by_name.items()} }",
    )

    penalty_rows = read_csv(artifacts / "model_penalty_cv_metrics.csv")
    selected_by_model: Dict[str, List[Dict[str, str]]] = {}
    for row in penalty_rows:
        if row["selected"].lower() == "true":
            selected_by_model.setdefault(row["model"], []).append(row)
    penalty_passed = all(
        len(selected_by_model.get(model, [])) == 1
        for model in ("fourier_16", "tm_full_80", "tm_instant_16")
    )
    append_check(
        checks,
        "model_penalty_selection",
        penalty_passed,
        f"selected_models={sorted(selected_by_model)}",
    )

    prediction_rows = read_csv(artifacts / "test_predictions.csv")
    targets = np.asarray([float(row["alpha"]) for row in prediction_rows], dtype=np.float64)
    prediction_seeds = np.asarray([int(row["seed"]) for row in prediction_rows], dtype=np.int64)
    recomputed_rmse: Dict[str, float] = {}
    for model in model_by_name:
        predictions = np.asarray(
            [float(row[f"prediction_{model}"]) for row in prediction_rows],
            dtype=np.float64,
        )
        recomputed_rmse[model] = rmse(targets, predictions)
    rmse_passed = all(
        np.isclose(recomputed_rmse[model], float(model_by_name[model]["test_rmse"]), rtol=0.0, atol=1.0e-15)
        for model in model_by_name
    )
    append_check(
        checks,
        "test_rmse_recomputation",
        rmse_passed,
        f"rmse={recomputed_rmse}",
    )

    summary_models = {row["model"]: row for row in summary["per_model"]}
    summary_passed = all(
        np.isclose(
            recomputed_rmse[model],
            float(summary_models[model]["test_rmse"]),
            rtol=0.0,
            atol=1.0e-15,
        )
        for model in model_by_name
    )
    append_check(
        checks,
        "summary_model_metrics",
        summary_passed,
        f"models={sorted(summary_models)}",
    )

    tm_predictions = np.asarray(
        [float(row["prediction_tm_selected_16"]) for row in prediction_rows],
        dtype=np.float64,
    )
    fourier_predictions = np.asarray(
        [float(row["prediction_fourier_16"]) for row in prediction_rows],
        dtype=np.float64,
    )
    point_difference = rmse(targets, fourier_predictions) - rmse(targets, tm_predictions)
    bootstrap_rows = read_csv(artifacts / "bootstrap_differences.csv")
    bootstrap_values = np.asarray(
        [float(row["fourier_minus_tm_rmse"]) for row in bootstrap_rows],
        dtype=np.float64,
    )
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    comparison = summary["fourier_minus_tm_bootstrap"]
    bootstrap_passed = (
        bootstrap_values.size == int(config["bootstrap_repetitions"])
        and np.isclose(point_difference, float(comparison["point_difference"]), atol=1.0e-15)
        and np.isclose(lower, float(comparison["ci_95_lower"]), atol=1.0e-15)
        and np.isclose(upper, float(comparison["ci_95_upper"]), atol=1.0e-15)
    )
    append_check(
        checks,
        "bootstrap_recomputation",
        bootstrap_passed,
        f"difference={point_difference}, ci=[{lower}, {upper}]",
    )

    expected_success = bool(lower > 0.0)
    gate_passed = expected_success == bool(summary["success"]) == bool(metrics["primary"]["success"])
    append_check(
        checks,
        "success_gate",
        gate_passed,
        f"expected={expected_success}, recorded={summary['success']}",
    )

    alpha_rows = read_csv(artifacts / "alpha_metrics.csv")
    alpha_passed = True
    for row in alpha_rows:
        alpha_value = float(row["alpha"])
        model = row["model"]
        mask = targets == alpha_value
        predictions = np.asarray(
            [float(item[f"prediction_{model}"]) for item in prediction_rows],
            dtype=np.float64,
        )
        alpha_passed &= int(row["row_count"]) == int(np.sum(mask))
        alpha_passed &= bool(
            np.isclose(rmse(targets[mask], predictions[mask]), float(row["rmse"]), atol=1.0e-15)
        )
    append_check(
        checks,
        "alpha_metrics_recomputation",
        alpha_passed,
        f"rows={len(alpha_rows)}",
    )

    seed_rows = read_csv(artifacts / "seed_metrics.csv")
    seed_passed = len(seed_rows) == len(test_seeds)
    for row in seed_rows:
        seed_value = int(row["seed"])
        mask = prediction_seeds == seed_value
        tm_seed_rmse = rmse(targets[mask], tm_predictions[mask])
        fourier_seed_rmse = rmse(targets[mask], fourier_predictions[mask])
        seed_passed &= bool(
            np.isclose(tm_seed_rmse, float(row["tm_selected_16_rmse"]), atol=1.0e-15)
            and np.isclose(fourier_seed_rmse, float(row["fourier_16_rmse"]), atol=1.0e-15)
            and np.isclose(
                fourier_seed_rmse - tm_seed_rmse,
                float(row["fourier_minus_tm_rmse"]),
                atol=1.0e-15,
            )
        )
    append_check(
        checks,
        "seed_metrics_recomputation",
        seed_passed,
        f"seeds={len(seed_rows)}",
    )

    plot_paths = [
        artifacts / "matched_capacity_results.png",
        artifacts / "tm_candidate_selection.png",
    ]
    plot_passed = all(path.is_file() and path.stat().st_size > 10_000 for path in plot_paths)
    append_check(
        checks,
        "plot_artifacts",
        plot_passed,
        f"sizes={ {path.name: path.stat().st_size if path.exists() else 0 for path in plot_paths} }",
    )

    code_hashes = environment.get("code_sha256", {})
    hash_passed = bool(code_hashes) and all(
        (run_directory / name).is_file()
        and file_sha256(run_directory / name) == expected_hash
        for name, expected_hash in code_hashes.items()
        if name != "e1a_temporal_alpha_readout.py"
    )
    pilot_name = "e1a_temporal_alpha_readout.py"
    pilot_path = run_directory.parent / "20260830_E1A_temporal-alpha-readout" / pilot_name
    hash_passed &= code_hashes.get(pilot_name) == file_sha256(pilot_path)
    append_check(
        checks,
        "code_sha256",
        hash_passed,
        f"files={sorted(code_hashes)}",
    )

    artifact_hashes = environment.get("artifact_sha256", {})
    artifact_hash_passed = bool(artifact_hashes) and all(
        (artifacts / name).is_file()
        and file_sha256(artifacts / name) == expected_hash
        for name, expected_hash in artifact_hashes.items()
    )
    append_check(
        checks,
        "artifact_sha256",
        artifact_hash_passed,
        f"files={sorted(artifact_hashes)}",
    )

    dependencies = environment.get("dependencies", [])
    environment_passed = any(str(value).startswith("numpy==") for value in dependencies) and any(
        str(value).startswith("matplotlib==") for value in dependencies
    )
    append_check(
        checks,
        "environment_record",
        environment_passed,
        f"dependencies={dependencies}",
    )

    failed_checks = [check.name for check in checks if not check.passed]
    validation: Dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if not failed_checks else "failed",
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": [asdict(check) for check in checks],
        "failed_checks": failed_checks,
    }
    (artifacts / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validation


def parse_arguments(arguments: List[str]) -> argparse.Namespace:
    """検証対象runをCLIから受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path.cwd())
    return parser.parse_args(arguments)


def main(arguments: List[str] | None = None) -> int:
    """検証結果を保存・表示し、失敗時は非0で終了する。"""

    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    validation = validate_results(args.run_directory.resolve())
    print(json.dumps(validation, ensure_ascii=False))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
