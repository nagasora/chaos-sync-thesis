"""E1B 2源識別可能性実験の保存成果物を独立に再計算して検証する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
MODEL_ORDER = (
    "oracle_tm_full_rank",
    "oracle_fourier_full_rank",
    "direct_tm_full_rank",
    "direct_fourier_full_rank",
    "direct_tm_rank_one",
    "direct_fourier_rank_one",
)


@dataclass(frozen=True)
class Check:
    """一つの独立検証結果を機械可読に保持する。"""

    name: str
    passed: bool
    detail: str


def read_csv(path: Path) -> List[Dict[str, str]]:
    """BOMを許容してCSVを辞書行として読む。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ordered_rmse(targets: FloatArray, predictions: FloatArray) -> float:
    """保存予測から順序付きRMSEを独立計算する。"""

    return float(np.sqrt(np.mean(np.square(targets - predictions))))


def permutation_invariant_rmse(
    targets: FloatArray,
    predictions: FloatArray,
) -> float:
    """保存予測から順序不変RMSEを独立計算する。"""

    direct = np.mean(np.square(targets - predictions), axis=1)
    swapped = np.mean(np.square(targets[:, ::-1] - predictions), axis=1)
    return float(np.sqrt(np.mean(np.minimum(direct, swapped))))


def file_sha256(path: Path) -> str:
    """保存済みSHA-256と比較するハッシュを計算する。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_check(checks: List[Check], name: str, passed: bool, detail: str) -> None:
    """検証名、成否、根拠を一行へ追加する。"""

    checks.append(Check(name=name, passed=bool(passed), detail=detail))


def load_predictions(rows: List[Dict[str, str]], model: str) -> FloatArray:
    """test予測CSVから指定modelの2座標を配列へ戻す。"""

    return np.asarray(
        [
            [
                float(row[f"prediction_{model}_alpha1"]),
                float(row[f"prediction_{model}_alpha2"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def validate_results(run_directory: Path) -> Dict[str, object]:
    """JSON、CSV、NPZ、PNGを相互照合しvalidation.jsonを保存する。"""

    artifacts = run_directory / "artifacts"
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads(
        (run_directory / "environment.json").read_text(encoding="utf-8")
    )
    checks: List[Check] = []

    statuses = (config["status"], metrics["status"], summary["status"])
    append_check(
        checks,
        "completed_status",
        statuses == ("completed", "completed", "completed"),
        f"statuses={statuses}",
    )

    matrix = np.asarray(
        config["theory"]["full_rank_observation_matrix"],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(matrix))
    theory_passed = (
        matrix.shape == (2, 2)
        and abs(determinant) > 1.0e-12
        and np.isclose(
            determinant,
            float(summary["mixing_matrix_determinant"]),
            atol=1.0e-15,
        )
    )
    append_check(
        checks,
        "full_rank_observation",
        theory_passed,
        f"determinant={determinant}",
    )

    data = config["data"]
    fit_seeds = set(
        range(
            int(data["fit_pair_seed_start"]),
            int(data["fit_pair_seed_start"]) + int(data["fit_pair_seed_count"]),
        )
    )
    test_seeds = set(
        range(
            int(data["test_pair_seed_start"]),
            int(data["test_pair_seed_start"]) + int(data["test_pair_seed_count"]),
        )
    )
    parent_config = json.loads(
        (
            run_directory.parent
            / "20260901_E1A_matched-capacity-readout"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    parent_seeds = set(parent_config["data"]["fit_seeds"]) | set(
        parent_config["data"]["test_seeds"]
    )
    split_passed = not (fit_seeds & test_seeds) and not (
        (fit_seeds | test_seeds) & parent_seeds
    )
    append_check(
        checks,
        "pair_seed_split_disjoint",
        split_passed,
        (
            f"fit={len(fit_seeds)}, test={len(test_seeds)}, "
            f"parent_overlap={len((fit_seeds | test_seeds) & parent_seeds)}"
        ),
    )

    fit_pairs = len(list(combinations(data["fit_alphas"], 2)))
    test_pairs = len(list(combinations(data["test_alphas"], 2)))
    expected_source_rows = len(fit_seeds) * fit_pairs + len(test_seeds) * test_pairs
    expected_feature_rows = expected_source_rows * 2
    with np.load(artifacts / "source_trajectories.npz", allow_pickle=False) as archive:
        source_split = np.asarray(archive["split"], dtype=np.str_)
        source_low = np.asarray(archive["source_low"], dtype=np.float64)
        source_high = np.asarray(archive["source_high"], dtype=np.float64)
        source_pair_seed = np.asarray(archive["pair_seed"], dtype=np.int64)
        low_scale = np.asarray(archive["low_scale"], dtype=np.float64)
        high_scale = np.asarray(archive["high_scale"], dtype=np.float64)
    source_passed = (
        source_split.size == expected_source_rows
        and source_low.shape
        == (expected_source_rows, int(data["observation_length"]))
        and source_high.shape == source_low.shape
        and np.all(np.isfinite(source_low))
        and np.all(np.isfinite(source_high))
        and np.all(low_scale > 0.0)
        and np.all(high_scale > 0.0)
        and set(source_pair_seed[source_split == "fit"].tolist()) == fit_seeds
        and set(source_pair_seed[source_split == "test"].tolist()) == test_seeds
    )
    append_check(
        checks,
        "source_data_store",
        source_passed,
        f"rows={source_split.size}, shape={source_low.shape}",
    )

    source_manifest = read_csv(artifacts / "source_trajectory_manifest.csv")
    source_manifest_passed = len(source_manifest) == expected_source_rows and all(
        int(row["collision_id"]) == index
        and row["split"] == source_split[index]
        and int(row["pair_seed"]) == source_pair_seed[index]
        for index, row in enumerate(source_manifest)
    )
    append_check(
        checks,
        "source_manifest",
        source_manifest_passed,
        f"rows={len(source_manifest)}",
    )

    with np.load(artifacts / "features.npz", allow_pickle=False) as archive:
        split = np.asarray(archive["split"], dtype=np.str_)
        collision_id = np.asarray(archive["collision_id"], dtype=np.int64)
        pair_seed = np.asarray(archive["pair_seed"], dtype=np.int64)
        order_index = np.asarray(archive["order_index"], dtype=np.int64)
        alpha1 = np.asarray(archive["alpha1"], dtype=np.float64)
        alpha2 = np.asarray(archive["alpha2"], dtype=np.float64)
        feature_arrays = {
            name.removeprefix("feature__"): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith("feature__")
        }
    expected_dimensions = {
        "oracle_tm_full_rank": 32,
        "oracle_fourier_full_rank": 32,
        "direct_tm_full_rank": 32,
        "direct_fourier_full_rank": 32,
        "direct_tm_rank_one": 16,
        "direct_fourier_rank_one": 16,
    }
    feature_passed = (
        split.size == expected_feature_rows
        and int(np.sum(split == "fit")) == len(fit_seeds) * fit_pairs * 2
        and int(np.sum(split == "test")) == len(test_seeds) * test_pairs * 2
        and set(feature_arrays) == set(expected_dimensions)
        and all(
            feature_arrays[name].shape == (expected_feature_rows, dimension)
            for name, dimension in expected_dimensions.items()
        )
        and all(np.all(np.isfinite(values)) for values in feature_arrays.values())
    )
    append_check(
        checks,
        "feature_store_dimensions",
        feature_passed,
        f"rows={split.size}, dimensions={expected_dimensions}",
    )

    unique_collisions, collision_counts = np.unique(collision_id, return_counts=True)
    target_swap_passed = unique_collisions.size == expected_source_rows and np.all(
        collision_counts == 2
    )
    for value in unique_collisions:
        indices = np.flatnonzero(collision_id == value)
        target_swap_passed &= bool(
            np.array_equal(order_index[indices], np.asarray([0, 1], dtype=np.int64))
            and np.isclose(alpha1[indices[0]], alpha2[indices[1]])
            and np.isclose(alpha2[indices[0]], alpha1[indices[1]])
        )
    append_check(
        checks,
        "collision_target_swap",
        target_swap_passed,
        f"collision_count={unique_collisions.size}",
    )

    rank_differences: List[float] = []
    for value in unique_collisions:
        indices = np.flatnonzero(collision_id == value)
        for model in ("direct_tm_rank_one", "direct_fourier_rank_one"):
            rank_differences.append(
                float(
                    np.max(
                        np.abs(
                            feature_arrays[model][indices[0]]
                            - feature_arrays[model][indices[1]]
                        )
                    )
                )
            )
    max_rank_difference = max(rank_differences)
    tolerance = float(metrics["primary"]["collision_tolerance"])
    append_check(
        checks,
        "rank_one_exact_feature_collision",
        max_rank_difference <= tolerance
        and np.isclose(
            max_rank_difference,
            float(summary["rank_one_collision_feature_max_abs_difference"]),
            atol=0.0,
        ),
        f"max_difference={max_rank_difference}, tolerance={tolerance}",
    )

    manifest_rows = read_csv(artifacts / "trajectory_manifest.csv")
    manifest_passed = len(manifest_rows) == expected_feature_rows and all(
        int(row["row_id"]) == index
        and row["split"] == split[index]
        and int(row["collision_id"]) == collision_id[index]
        and int(row["pair_seed"]) == pair_seed[index]
        and int(row["order_index"]) == order_index[index]
        and float(row["alpha1"]) == alpha1[index]
        and float(row["alpha2"]) == alpha2[index]
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
        fold_seed_set.add(int(row["pair_seed"]))
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

    cv_rows = read_csv(artifacts / "model_cv_metrics.csv")
    cv_passed = True
    for model in MODEL_ORDER:
        model_cv = [row for row in cv_rows if row["model"] == model]
        selected = [row for row in model_cv if row["selected"].lower() == "true"]
        minimum = min(
            model_cv,
            key=lambda row: (
                float(row["cv_ordered_rmse"]),
                float(row["penalty"]),
            ),
        )
        cv_passed &= (
            len(model_cv) == len(config["training"]["ridge_penalties"])
            and len(selected) == 1
            and selected[0]["penalty"] == minimum["penalty"]
            and selected[0]["cv_ordered_rmse"] == minimum["cv_ordered_rmse"]
        )
    append_check(
        checks,
        "ridge_penalty_selection",
        cv_passed,
        f"rows={len(cv_rows)}",
    )

    prediction_rows = read_csv(artifacts / "test_predictions.csv")
    targets = np.asarray(
        [[float(row["alpha1"]), float(row["alpha2"])] for row in prediction_rows],
        dtype=np.float64,
    )
    model_rows = read_csv(artifacts / "model_metrics.csv")
    model_by_name = {row["model"]: row for row in model_rows}
    metric_passed = set(model_by_name) == set(MODEL_ORDER)
    recomputed: Dict[str, Dict[str, float]] = {}
    for model in MODEL_ORDER:
        predictions = load_predictions(prediction_rows, model)
        ordered = ordered_rmse(targets, predictions)
        invariant = permutation_invariant_rmse(targets, predictions)
        recomputed[model] = {"ordered": ordered, "invariant": invariant}
        metric_passed &= bool(
            np.isclose(
                ordered,
                float(model_by_name[model]["test_ordered_rmse"]),
                atol=1.0e-15,
            )
            and np.isclose(
                invariant,
                float(model_by_name[model]["test_permutation_invariant_rmse"]),
                atol=1.0e-15,
            )
        )
    append_check(
        checks,
        "test_metric_recomputation",
        metric_passed,
        f"metrics={recomputed}",
    )

    floor = float(
        np.sqrt(np.mean(np.square((targets[:, 0] - targets[:, 1]) / 2.0)))
    )
    floor_passed = np.isclose(
        floor,
        float(summary["theoretical_collision_floor"]),
        atol=1.0e-15,
    ) and all(
        recomputed[model]["ordered"] >= floor - tolerance
        for model in ("direct_tm_rank_one", "direct_fourier_rank_one")
    )
    append_check(
        checks,
        "theoretical_collision_floor",
        floor_passed,
        (
            f"floor={floor}, tm={recomputed['direct_tm_rank_one']['ordered']}, "
            f"fourier={recomputed['direct_fourier_rank_one']['ordered']}"
        ),
    )

    threshold = float(metrics["primary"]["oracle_tm_rmse_threshold"])
    expected_components = {
        "oracle_tm_below_threshold": bool(
            recomputed["oracle_tm_full_rank"]["ordered"] < threshold
        ),
        "rank_one_collision_exact": bool(max_rank_difference <= tolerance),
        "rank_one_ordered_floor_tm": bool(
            recomputed["direct_tm_rank_one"]["ordered"] >= floor - tolerance
        ),
        "rank_one_ordered_floor_fourier": bool(
            recomputed["direct_fourier_rank_one"]["ordered"] >= floor - tolerance
        ),
    }
    expected_success = bool(all(expected_components.values()))
    gate_passed = (
        expected_components == summary["success_components"]
        and expected_components == metrics["primary"]["success_components"]
        and expected_success == bool(summary["success"])
        and expected_success == bool(metrics["primary"]["success"])
    )
    append_check(
        checks,
        "success_gate",
        gate_passed,
        f"components={expected_components}, success={expected_success}",
    )

    pair_rows = read_csv(artifacts / "pair_metrics.csv")
    seed_rows = read_csv(artifacts / "seed_metrics.csv")
    aggregation_passed = (
        len(pair_rows) == test_pairs * len(MODEL_ORDER)
        and len(seed_rows) == len(test_seeds) * len(MODEL_ORDER)
        and all(int(row["row_count"]) > 0 for row in pair_rows + seed_rows)
    )
    append_check(
        checks,
        "pair_seed_aggregations",
        aggregation_passed,
        f"pair_rows={len(pair_rows)}, seed_rows={len(seed_rows)}",
    )

    collision_rows = read_csv(artifacts / "collision_diagnostics.csv")
    collision_table_passed = (
        len(collision_rows) == len(test_seeds) * test_pairs
        and max(
            max(
                float(row["tm_feature_max_abs_difference"]),
                float(row["fourier_feature_max_abs_difference"]),
            )
            for row in collision_rows
        )
        == max_rank_difference
    )
    append_check(
        checks,
        "collision_diagnostics",
        collision_table_passed,
        f"rows={len(collision_rows)}",
    )

    plot_paths = [
        artifacts / "identifiability_results.png",
        artifacts / "prediction_scatter.png",
    ]
    plot_passed = all(path.is_file() and path.stat().st_size > 10_000 for path in plot_paths)
    append_check(
        checks,
        "plot_artifacts",
        plot_passed,
        f"sizes={ {path.name: path.stat().st_size if path.exists() else 0 for path in plot_paths} }",
    )

    code_hashes = environment.get("code_sha256", {})
    code_locations = {
        "e1b_two_source_identifiability.py": run_directory
        / "e1b_two_source_identifiability.py",
        "validate_results.py": run_directory / "validate_results.py",
        "e1a_matched_capacity_readout.py": run_directory.parent
        / "20260901_E1A_matched-capacity-readout"
        / "e1a_matched_capacity_readout.py",
        "e1a_temporal_alpha_readout.py": run_directory.parent
        / "20260830_E1A_temporal-alpha-readout"
        / "e1a_temporal_alpha_readout.py",
    }
    code_hash_passed = set(code_hashes) == set(code_locations) and all(
        path.is_file() and file_sha256(path) == code_hashes[name]
        for name, path in code_locations.items()
    )
    append_check(
        checks,
        "code_sha256",
        code_hash_passed,
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
        f"files={len(artifact_hashes)}",
    )

    dependencies = environment.get("dependencies", [])
    environment_passed = any(
        str(value).startswith("numpy==") for value in dependencies
    ) and any(str(value).startswith("matplotlib==") for value in dependencies)
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
