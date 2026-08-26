"""E0一般化Boole写像検証コードの計算契約を確認する。"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "20260806_E0_boole-invariant-measure"
    / "e0_boole_validation.py"
)


def load_script_module() -> ModuleType:
    """What: run内の実験コードを直接読み込み、公開計算関数を検証する。"""

    spec = importlib.util.spec_from_file_location("e0_boole_validation", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"実験コードを読み込めません: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BooleValidationTests(unittest.TestCase):
    """What: 理論特殊値、写像、分布指標、最小run出力を確認する。"""

    @classmethod
    def setUpClass(cls) -> None:
        """What: 全テストで同じ実験モジュールを使用する。"""

        cls.module = load_script_module()

    def test_maximum_chaos_special_values(self) -> None:
        """What: alpha=1/2で尺度1、Lyapunov指数log(2)になる。"""

        self.assertAlmostEqual(1.0, self.module.theoretical_cauchy_scale(0.5))
        self.assertAlmostEqual(math.log(2.0), self.module.theoretical_lyapunov(0.5))

    def test_generalized_boole_map_known_value(self) -> None:
        """What: alpha=1/2、x=2の一ステップが0.75になる。"""

        self.assertAlmostEqual(0.75, self.module.generalized_boole_map(2.0, 0.5))

    def test_cauchy_quantile_grid_has_small_ks_distance(self) -> None:
        """What: 理論分位点から作った標本はCauchy CDFと整合する。"""

        probabilities = (np.arange(10_000, dtype=np.float64) + 0.5) / 10_000
        values = np.tan(math.pi * (probabilities - 0.5))
        self.assertLess(self.module.cauchy_ks_distance(values, 1.0), 1.0e-4)
        self.assertAlmostEqual(1.0, self.module.estimate_cauchy_scale(values), places=3)

    def test_quadrature_matches_closed_form(self) -> None:
        """What: 角度積分のLyapunov指数が閉形式と独立に一致する。"""

        numerical = self.module.quadrature_lyapunov(0.5, 100_000)
        self.assertLess(abs(numerical - math.log(2.0)), 2.0e-5)

    def test_small_run_writes_machine_readable_artifacts(self) -> None:
        """What: 小規模runがJSON・CSV・NPZを一貫した場所へ保存する。"""

        config = self.module.ExperimentConfig(
            alpha_values=(0.5,),
            seeds=(11,),
            burn_in=100,
            observation_length=1_000,
            tm_max_order=2,
            diagnostic_sample_size=100,
            quadrature_points=10_000,
            thresholds=self.module.ValidationThresholds(
                max_scale_relative_error=1.0,
                max_lyapunov_absolute_error=1.0,
                max_ks_distance=1.0,
                max_tm_nonzero_magnitude=1.0,
                max_quadrature_lyapunov_error=1.0,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            (run_directory / "environment.json").write_text("{}", encoding="utf-8")
            summary = self.module.run_experiment(run_directory, config)
            saved_summary = json.loads(
                (run_directory / "artifacts" / "summary.json").read_text(encoding="utf-8")
            )

            self.assertTrue(summary["overall_passed"])
            self.assertEqual(summary["overall_passed"], saved_summary["overall_passed"])
            self.assertTrue((run_directory / "artifacts" / "per_seed_metrics.csv").is_file())
            self.assertTrue((run_directory / "artifacts" / "diagnostic_orbit_samples.npz").is_file())


if __name__ == "__main__":
    unittest.main()
