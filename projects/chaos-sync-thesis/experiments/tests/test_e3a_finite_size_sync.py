"""E3A有限サイズ同期転移pilotの計算契約を確認する。"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "20260830_E3A_finite-size-sync-transition"
    / "e3a_finite_size_sync.py"
)


def load_script_module() -> ModuleType:
    """What: run内のE3Aコードを配布形態のまま読み込む。"""

    spec = importlib.util.spec_from_file_location("e3a_finite_size_sync", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"実験コードを読み込めません: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FiniteSizeSyncTests(unittest.TestCase):
    """What: 理論特殊値、結合、TM指標、同期多様体、再現性を検証する。"""

    @classmethod
    def setUpClass(cls) -> None:
        """What: 全テストで同じE3Aモジュールを使用する。"""

        cls.module = load_script_module()

    def test_alpha_quarter_theory_values(self) -> None:
        """What: alpha=1/4、moment=1でKc=0.5、Kmax=0.75になる。"""

        self.assertAlmostEqual(0.5, self.module.critical_coupling(0.25, 1.0))
        self.assertAlmostEqual(0.75, 1.0 - 0.25)
        self.assertAlmostEqual(0.0, self.module.conditional_lyapunov(0.25, 0.5, 1.0))

    def test_generated_couplings_are_symmetric_with_zero_diagonal(self) -> None:
        """What: 全結合分布で対称性・対角0・絶対moment有限が成立する。"""

        for distribution in self.module.COUPLING_DISTRIBUTIONS:
            matrix = self.module.generate_symmetric_coupling(32, distribution, 17)
            statistics = self.module.calculate_coupling_statistics(matrix, distribution, 17)
            np.testing.assert_array_equal(matrix, matrix.T)
            np.testing.assert_array_equal(np.diag(matrix), np.zeros(32))
            self.assertTrue(statistics.symmetric)
            self.assertTrue(statistics.diagonal_zero)
            self.assertTrue(math.isfinite(statistics.sample_absolute_mean))

    def test_tm_transform_has_unit_modulus(self) -> None:
        """What: 有限実数状態のTM像は数値誤差内で単位円上にある。"""

        states = np.asarray([-1.0e6, -3.0, -0.1, 0.0, 0.2, 4.0, 1.0e6])
        transformed = self.module.tm_transform(states, 1.3)
        np.testing.assert_allclose(np.abs(transformed), 1.0, atol=2.0e-15)

    def test_uniform_coupling_preserves_exact_sync_manifold(self) -> None:
        """What: 一様結合で全状態が同じなら一ステップ後も同じになる。"""

        matrix = self.module.generate_symmetric_coupling(16, "uniform_positive", 1)
        states = np.full(16, 1.7, dtype=np.float64)
        updated = self.module.coupled_step(states, matrix, 0.25, 0.6)
        np.testing.assert_allclose(updated, updated[0], rtol=0.0, atol=1.0e-15)

    def test_random_coupling_breaks_exact_sync_when_row_sums_differ(self) -> None:
        """What: 有限ランダム行列では同期状態から行和依存の差が生じる。"""

        matrix = self.module.generate_symmetric_coupling(32, "rademacher", 5)
        states = np.full(32, 1.7, dtype=np.float64)
        updated = self.module.coupled_step(states, matrix, 0.25, 0.6)
        self.assertGreater(float(np.std(updated)), 0.0)

    def test_same_seed_reproduces_small_simulation(self) -> None:
        """What: 同じ行列・初期値seedと設定で集約値が完全一致する。"""

        config = self.module.ExperimentConfig(
            alpha=0.25,
            node_counts=(16,),
            matrix_seeds=(7,),
            coupling_distributions=("biased_sign",),
            k_values=(0.0, 0.4, 0.6),
            burn_in=20,
            evaluation_steps=30,
            local_growth_steps=10,
            local_perturbation=1.0e-8,
            near_zero_threshold=1.0e-12,
            bootstrap_repetitions=20,
            bootstrap_seed=9,
            success_mean_order=0.8,
            success_q05_order=0.5,
            phase_heatmap_k_values=(0.4,),
            save_figures=False,
        )
        matrix = self.module.generate_symmetric_coupling(16, "biased_sign", 7)
        statistics = self.module.calculate_coupling_statistics(matrix, "biased_sign", 7)
        first, _ = self.module.simulate_coupling_case(matrix, statistics, config)
        second, _ = self.module.simulate_coupling_case(matrix, statistics, config)
        self.assertEqual(first, second)

    def test_independent_cauchy_tm_order_has_inverse_sqrt_n_scale(self) -> None:
        """What: 適応尺度TM秩序は独立Cauchy標本で小さい。"""

        rng = np.random.default_rng(123)
        states = rng.standard_cauchy(4096)
        order, _, _, modulus_error, _ = self.module.tm_phase_metrics(states)
        self.assertLess(order, 0.06)
        self.assertLess(modulus_error, 2.0e-15)

    def test_h1_uses_sustained_sync_onset_instead_of_tm_r50(self) -> None:
        """What: 理論Kcとの比較にはTM凝集R50でなく持続同期の立上りを使う。"""

        config = self.module.ExperimentConfig(
            alpha=0.25,
            node_counts=(256,),
            matrix_seeds=(1, 2, 3),
            coupling_distributions=("uniform_positive",),
            k_values=(0.0, 0.25, 0.5),
            burn_in=20,
            evaluation_steps=20,
            local_growth_steps=5,
            local_perturbation=1.0e-8,
            near_zero_threshold=1.0e-12,
            bootstrap_repetitions=20,
            bootstrap_seed=9,
            success_mean_order=0.8,
            success_q05_order=0.5,
            phase_heatmap_k_values=(0.5,),
            save_figures=False,
        )
        thresholds = [
            {
                "distribution": "uniform_positive",
                "node_count": 256,
                "initialization": "local",
                "r50": 0.25,
                "r50_times_absolute_mean": 0.25,
                "mean_sample_absolute_mean": 1.0,
                "sustained_sync_k50": 0.5,
                "sustained_sync_k50_times_absolute_mean": 0.5,
            }
        ]

        assessment = self.module.hypothesis_assessment([], [], thresholds, config)

        self.assertEqual(
            "pilot_supported",
            assessment["H1_theory_reproduction"]["status"],
        )
        self.assertEqual(
            0.5,
            assessment["H1_theory_reproduction"][
                "uniform_local_sustained_sync_k50_max_n"
            ],
        )
        self.assertEqual(
            0.25,
            assessment["H1_theory_reproduction"]["uniform_local_tm_r50_max_n"],
        )
        self.assertEqual(
            "pilot_inconclusive",
            assessment["H2_absolute_moment_universality"]["status"],
        )

        confirmation_config = replace(config, assessment_scope="confirmation")
        confirmation = self.module.hypothesis_assessment(
            [], [], thresholds, confirmation_config
        )
        self.assertEqual(
            "confirmation_supported",
            confirmation["H1_theory_reproduction"]["status"],
        )

    def test_bootstrap_separates_tm_coherence_from_sustained_sync(self) -> None:
        """What: TM凝集点と持続同期成功率50%点を別々に推定する。"""

        values = []
        for matrix_seed in (1, 2, 3):
            for coupling_strength, mean_tm_order, synchronized in (
                (0.0, 0.1, False),
                (0.25, 0.6, False),
                (0.5, 1.0, True),
            ):
                values.append(
                    SimpleNamespace(
                        matrix_seed=matrix_seed,
                        coupling_strength=coupling_strength,
                        mean_tm_order=mean_tm_order,
                        finite_completed=True,
                        synchronized=synchronized,
                    )
                )

        result = self.module.bootstrap_thresholds(
            values,
            (0.0, 0.25, 0.5),
            repetitions=20,
            bootstrap_seed=9,
        )

        self.assertAlmostEqual(0.2, result["r50"])
        self.assertAlmostEqual(0.375, result["sustained_sync_k50"])
        self.assertEqual(1.0, result["sustained_sync_k50_bootstrap_valid_fraction"])


    def test_uniform_batch_fast_path_matches_dense_product(self) -> None:
        """What: 一様完全グラフのO(N)場計算は密行列積と一致する。"""

        rng = np.random.default_rng(77)
        states = rng.normal(size=(6, 16))
        states[np.abs(states) < 0.1] += 0.2
        matrix = self.module.generate_symmetric_coupling(16, "uniform_positive", 1)
        coupling_strengths = np.linspace(0.0, 0.7, states.shape[0])
        dense = self.module.coupled_batch_step(
            states,
            matrix,
            0.25,
            coupling_strengths,
        )
        fast = self.module.coupled_batch_step(
            states,
            matrix,
            0.25,
            coupling_strengths,
            use_uniform_complete_graph=True,
        )

        np.testing.assert_allclose(fast, dense, rtol=0.0, atol=2.0e-15)

if __name__ == "__main__":
    unittest.main()


def test_e3d_cauchy_distribution_contract() -> None:
    """E3Dの代数的一歩同期と分布oracleを独立な定義で照合する。"""
    path = SCRIPT_PATH.parent.parent / "20260908_E3D_cauchy-distribution-sync/run_experiment.py"
    spec = importlib.util.spec_from_file_location("e3d_distribution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states = np.array([[-.7, .2], [1.1, -2.3], [0., 0.5]])
    beta = 1.5
    expected = np.tan(beta * states).mean(axis=1)
    advanced = module.advance(states, beta, .5)
    np.testing.assert_array_equal(advanced[:, 0], advanced[:, 1])
    np.testing.assert_allclose(advanced[:, 0], expected, rtol=1e-15)
    gamma = module.scale_fixed(beta)
    assert abs(gamma-math.tanh(beta*gamma)) < 1e-12
    params, records = module.parameter_path(np.array([1j*gamma, 1j*gamma]), beta, .5)
    np.testing.assert_allclose(params, 1j*gamma, atol=2e-12)
    assert max(r["float64_parameter_error"] for r in records) < 1e-14
    n = 1000
    cauchy_quantiles = .3+.7*np.tan(np.pi*((np.arange(n)+.5)/n-.5))
    assert abs(module.cauchy_ks(cauchy_quantiles, .3+.7j)-.5/n) < 1e-13
    row = module.measure(states, np.array([0j, 0j]), beta, gamma, .25, "stationary", 0, 2)[0]
    assert not row["closure_supported"]
    assert row["reference_gamma"] == gamma
