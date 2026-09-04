"""E1A pilotを固定したまま独立test seedだけを増やして追試する。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import List


PILOT_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "20260830_E1A_temporal-alpha-readout"
    / "e1a_temporal_alpha_readout.py"
)


def load_pilot_module() -> ModuleType:
    """検証済みpilot実装を複製せず、追試用に読み込む。"""

    spec = importlib.util.spec_from_file_location("e1a_temporal_alpha_pilot", PILOT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"pilot実装を読み込めません: {PILOT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(arguments: List[str] | None = None) -> int:
    """学習条件を変えず、48個の新規test seedで同じ評価を再実行する。"""

    module = load_pilot_module()
    args = module.parse_arguments(sys.argv[1:] if arguments is None else arguments)
    config = module.ExperimentConfig(
        test_seeds=tuple(range(20263830, 20263878)),
        bootstrap_repetitions=5_000,
        bootstrap_seed=20263830,
    )
    run_directory = args.run_directory.resolve()
    summary = module.run_experiment(run_directory, config)

    environment_path = run_directory / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["execution_command"] = (
        "python e1a_temporal_alpha_replication.py --run-directory ."
    )
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
