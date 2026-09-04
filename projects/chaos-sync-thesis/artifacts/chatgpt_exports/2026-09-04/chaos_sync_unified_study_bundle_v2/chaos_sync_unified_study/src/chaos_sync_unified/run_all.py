from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .experiments import (
    E2RobustnessExperiment,
    E3BasinExperiment,
    E3InputRetentionExperiment,
    E4SignalReconstructionExperiment,
    E5SmallImageExperiment,
)
from .registry import write_registry
from .utils import collect_hashes, environment_payload, write_csv, write_json


def run_all(root: Path, *, force: bool = False) -> dict[str, object]:
    """How: 依存順にE2A→E3B→E3C→E4A→E5Aを実行し、統合manifestを確定する。"""
    root.mkdir(parents=True, exist_ok=True)
    write_registry(root)
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    run_specs = [
        ("e2", runs / "20260904_E2A_boole_observation_robustness", E2RobustnessExperiment()),
        ("e3b", runs / "20260904_E3B_tangent_two_node_basin", E3BasinExperiment()),
        ("e3c", runs / "20260904_E3C_boole_n8_input_retention", E3InputRetentionExperiment()),
    ]
    summaries: dict[str, object] = {}
    for key, directory, experiment in run_specs:
        if force and directory.exists():
            shutil.rmtree(directory)
        summary_path = directory / "artifacts" / "summary.json"
        if summary_path.exists() and not force:
            summaries[key] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summaries[key] = experiment.run(directory)
    e4_dir = runs / "20260904_E4A_boole_synthetic_signal"
    if force and e4_dir.exists():
        shutil.rmtree(e4_dir)
    e4_summary_path = e4_dir / "artifacts" / "summary.json"
    if e4_summary_path.exists() and not force:
        e4_summary = json.loads(e4_summary_path.read_text(encoding="utf-8"))
    else:
        e4_summary = E4SignalReconstructionExperiment().run(e4_dir)
    summaries["e4"] = e4_summary
    e5_dir = runs / "20260904_E5A_boole_small_image_digits"
    if force and e5_dir.exists():
        shutil.rmtree(e5_dir)
    e5_summary_path = e5_dir / "artifacts" / "summary.json"
    if e5_summary_path.exists() and not force:
        e5_summary = json.loads(e5_summary_path.read_text(encoding="utf-8"))
    else:
        e5_summary = E5SmallImageExperiment().run(
            e5_dir,
            coupling=float(e4_summary["selected_coupling"]),
            input_gain=float(e4_summary["selected_input_gain"]),
        )
    summaries["e5"] = e5_summary
    write_json(root / "all_experiment_summaries.json", summaries)
    write_json(root / "environment.json", environment_payload())
    write_csv(root / "artifact_hashes.csv", collect_hashes(root))
    integrity = {
        key: value.get("status") in {"completed", "pilot_completed"}
        for key, value in summaries.items()
        if isinstance(value, dict)
    }
    write_json(root / "suite_validation.json", integrity)
    if not all(integrity.values()):
        raise RuntimeError(f"統合suiteのintegrity statusに警告がある: {integrity}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_all(args.root, force=args.force)


if __name__ == "__main__":
    main()
