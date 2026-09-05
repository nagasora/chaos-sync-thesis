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



class SharedBooleCoreTests(unittest.TestCase):
    """共通写像の精度・特異点診断・旧実装との整合性を確認する。"""

    def test_shared_core_precision_and_failure_contract(self) -> None:
        """既知値、真のdtype、失敗停止、旧軌道との一致を一つの境界で検証する。"""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.core import generalized_boole, boole_log_derivative, boole_theory, simulate_boole_orbits
        for dtype in (np.float32,np.float64):
            with self.subTest(dtype=dtype):
                x = np.array([2.0,0.0,1e-15],dtype=dtype)
                y,bad,near = generalized_boole(x,np.array(.5))
                self.assertEqual(y.dtype,np.dtype(dtype))
                self.assertEqual(float(y[0]),.75)
                self.assertTrue(bad[1])
                np.testing.assert_array_equal(near,[False,True,True])
                orbit,diag = simulate_boole_orbits(np.array([1.0,2.0],dtype=dtype),np.array(.5),0,4)
                self.assertEqual(diag["failure_step"][0],1)
                self.assertEqual(float(orbit[0,0]),0.0)
                self.assertTrue(np.isnan(orbit[0,1:]).all())
                self.assertEqual(diag["failure_step"][1],-1)
        gamma,lyap = boole_theory(np.array(.5))
        self.assertAlmostEqual(float(gamma),1)
        self.assertAlmostEqual(float(lyap),math.log(2))
        self.assertAlmostEqual(float(boole_log_derivative(np.array([2.]),np.array(.5))[0]),math.log(.625))
        old = load_script_module()
        actual,_ = simulate_boole_orbits(np.array([2.],dtype=np.float64),np.array(.4),10,100)
        expected = []
        state = 2.
        for i in range(110):
            state = old.generalized_boole_map(state,.4)
            if i >= 10:
                expected.append(state)
        np.testing.assert_array_equal(actual[0],expected)
        with self.assertRaises(ValueError):
            boole_theory(np.array([0.,1.]))
        with self.assertRaises(ValueError):
            boole_log_derivative(np.array([0.]),np.array(.5))



    def test_runner_marks_unmeasured_precision_and_serializes_validation(self) -> None:
        """未実施精度を合格にせず、独立検証結果をJSONへ保存できる。"""
        base = Path(__file__).resolve().parents[1] / "E0"
        modules = {}
        for name in ("run_e0", "validate_results"):
            spec = importlib.util.spec_from_file_location(name, base / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
        config = json.loads((base / "config.json").read_text(encoding="utf-8-sig"))
        config.update(alpha_values=[.5], seeds=[20260905], initial_distributions=["gaussian"],
                      precision=["float64"], burn_in=10, observation_lengths=[100])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "single_precision"
            summary = modules["run_e0"].run(config, out)
            self.assertIsNone(summary["float32_accuracy_passed"])
            self.assertTrue(modules["validate_results"].validate(out)["passed"])
            saved = json.loads((out / "validation.json").read_text(encoding="utf-8"))
            self.assertIs(saved["passed"], True)



    def test_kfold_tm_identity_gram_and_exact_poles(self) -> None:
        """枝・直交性・shiftの正負対照・厳密極の契約を確認する。"""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.core import cayley_modes, kfold_cotangent
        theta = np.pi*(np.arange(2048)+.371)/2048
        x = np.cos(theta)/np.sin(theta)
        orders = tuple(range(-4,5))
        modes = cayley_modes(x,orders)
        np.testing.assert_allclose(modes,np.exp(-2j*theta[:,None]*np.array(orders)),atol=2e-14,rtol=0)
        np.testing.assert_allclose(modes.conj().T@modes/len(x),np.eye(9),atol=2e-14,rtol=0)
        for K in (2,3,5):
            y,bad,near = kfold_cotangent(x,K)
            self.assertFalse(bad.any())
            for k in (1,2,3,4):
                np.testing.assert_allclose(cayley_modes(y,(k,)),cayley_modes(x,(K*k,)),atol=1e-12,rtol=0)
        y,bad,near = kfold_cotangent(np.array([0.]),2)
        self.assertTrue(bad[0] and near[0] and np.isnan(y[0]))
        y,bad,near = kfold_cotangent(np.array([0.]),3)
        self.assertEqual(y[0],0)
        self.assertFalse(bad[0])
        for alpha in (.25,.75):
            y=alpha*x-(1-alpha)/x
            residual=np.abs(cayley_modes(y,(1,))-cayley_modes(x,(2,)))
            self.assertGreater(float(np.sqrt(np.mean(residual**2))),.01)
        for bad_K in (1,2.5,True):
            with self.assertRaises(ValueError):
                kfold_cotangent(x,bad_K)
        with self.assertRaises(ValueError):
            cayley_modes(x,(1,),gamma=0)
        with self.assertRaises(ValueError):
            cayley_modes(x,(.5,))


if __name__ == "__main__":
    unittest.main()
