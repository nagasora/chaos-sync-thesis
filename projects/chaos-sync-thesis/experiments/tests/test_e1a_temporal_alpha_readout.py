"""E1Aの時間構造読み出しpilotが守る計算契約を確認する。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "20260830_E1A_temporal-alpha-readout"
    / "e1a_temporal_alpha_readout.py"
)
MATCHED_CAPACITY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "20260901_E1A_matched-capacity-readout"
    / "e1a_matched_capacity_readout.py"
)
MATCHED_CAPACITY_VALIDATOR_PATH = MATCHED_CAPACITY_SCRIPT_PATH.with_name("validate_results.py")
TWO_SOURCE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "20260901_E1B_two-source-identifiability"
    / "e1b_two_source_identifiability.py"
)
TWO_SOURCE_VALIDATOR_PATH = TWO_SOURCE_SCRIPT_PATH.with_name("validate_results.py")


def load_script_module() -> ModuleType:
    """What: run内のE1A実験コードを配布形態のまま読み込む。"""

    spec = importlib.util.spec_from_file_location("e1a_temporal_alpha_readout", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"実験コードを読み込めません: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matched_capacity_module() -> ModuleType:
    """What: 容量一致追試コードを配布形態のまま読み込む。"""

    spec = importlib.util.spec_from_file_location(
        "e1a_matched_capacity_readout",
        MATCHED_CAPACITY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"実験コードを読み込めません: {MATCHED_CAPACITY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matched_capacity_validator() -> ModuleType:
    """What: 容量一致validatorを配布形態のまま読み込む。"""

    spec = importlib.util.spec_from_file_location(
        "e1a_matched_capacity_validator",
        MATCHED_CAPACITY_VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"validatorを読み込めません: {MATCHED_CAPACITY_VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_two_source_module() -> ModuleType:
    """What: 2源識別可能性実験コードを配布形態のまま読み込む。"""

    spec = importlib.util.spec_from_file_location(
        "e1b_two_source_identifiability",
        TWO_SOURCE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"実験コードを読み込めません: {TWO_SOURCE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_two_source_validator() -> ModuleType:
    """What: 2源識別可能性validatorを配布形態のまま読み込む。"""

    spec = importlib.util.spec_from_file_location(
        "e1b_two_source_validator",
        TWO_SOURCE_VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"validatorを読み込めません: {TWO_SOURCE_VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TemporalAlphaReadoutTests(unittest.TestCase):
    """What: 分割、特徴、ridge、bootstrap、最小run成果物を検証する。"""

    @classmethod
    def setUpClass(cls) -> None:
        """What: 全ケースで同じE1Aモジュールを使用する。"""

        cls.module = load_script_module()

    def test_config_rejects_seen_test_alpha(self) -> None:
        """What: test alphaがtrainへ混入した設定を拒否する。"""

        config = self.module.ExperimentConfig(
            train_alphas=(0.4, 0.5, 0.6),
            test_alphas=(0.5,),
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_shuffle_preserves_instantaneous_tm_moments(self) -> None:
        """What: 時間順序破壊対照は周辺TMモーメントを変えない。"""

        values = np.linspace(-3.0, 3.0, 257, dtype=np.float64)
        cayley = self.module.cayley_transform(values)
        shuffled = cayley[np.random.default_rng(7).permutation(cayley.size)]
        original = self.module.tm_instantaneous_features(cayley, 5)
        control = self.module.tm_instantaneous_features(shuffled, 5)
        np.testing.assert_allclose(original, control, atol=1.0e-14)

    def test_bootstrap_seed_does_not_change_shuffle_control(self) -> None:
        """What: 区間推定seedの変更は時間順序破壊対照を変えない。"""

        orbit = self.module.E0_MODULE.generate_orbit(0.5, 17, 32, 128)
        first_config = self.module.ExperimentConfig(
            observation_length=128,
            delay_lags=(1, 2),
            raw_window_length=16,
            bootstrap_seed=1,
            feature_shuffle_seed=99,
        )
        second_config = self.module.ExperimentConfig(
            observation_length=128,
            delay_lags=(1, 2),
            raw_window_length=16,
            bootstrap_seed=2,
            feature_shuffle_seed=99,
        )
        first_features, _, _ = self.module.extract_features(
            orbit,
            seed=17,
            alpha=0.5,
            config=first_config,
        )
        second_features, _, _ = self.module.extract_features(
            orbit,
            seed=17,
            alpha=0.5,
            config=second_config,
        )
        np.testing.assert_allclose(
            first_features["tm_delay_shuffled"],
            second_features["tm_delay_shuffled"],
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_ridge_recovers_linear_target(self) -> None:
        """What: ridge実装が線形な既知目標を十分小さい誤差で回収する。"""

        features = np.arange(1.0, 21.0, dtype=np.float64).reshape(-1, 1)
        targets = 0.3 + 0.02 * features[:, 0]
        predictions = self.module.fit_ridge(features, targets, features, 1.0e-8)
        self.assertLess(self.module.rmse(targets, predictions), 1.0e-8)

    def test_cluster_bootstrap_reports_clear_improvement(self) -> None:
        """What: 完全予測は定数baselineより正のRMSE差区間を持つ。"""

        targets = np.asarray([0.4, 0.6, 0.4, 0.6], dtype=np.float64)
        tm_predictions = targets.copy()
        baseline_predictions = np.full_like(targets, 0.5)
        seed_labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
        result = self.module.bootstrap_rmse_difference(
            targets,
            tm_predictions,
            baseline_predictions,
            seed_labels,
            repetitions=100,
            bootstrap_seed=11,
        )
        self.assertGreater(result["ci_95_lower"], 0.0)

    def test_small_run_writes_reproducible_artifacts(self) -> None:
        """What: 縮小設定でも全特徴評価と機械可読成果物を保存する。"""

        config = self.module.ExperimentConfig(
            train_alphas=(0.34, 0.50, 0.66),
            test_alphas=(0.42, 0.58),
            train_seeds=(11, 12, 13, 14),
            validation_seeds=(21, 22),
            test_seeds=(31, 32, 33),
            burn_in=128,
            observation_length=512,
            tm_max_order=3,
            delay_lags=(1, 2),
            raw_window_length=16,
            fourier_band_count=4,
            ridge_penalties=(1.0e-4, 1.0e-2),
            bootstrap_repetitions=50,
            bootstrap_seed=9,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            (run_directory / "config.json").write_text(
                json.dumps({"status": "planned"}),
                encoding="utf-8",
            )
            (run_directory / "environment.json").write_text("{}", encoding="utf-8")
            summary = self.module.run_experiment(run_directory, config)

            self.assertEqual("completed", summary["status"])
            self.assertEqual(7, len(summary["per_feature"]))
            self.assertTrue((run_directory / "artifacts" / "summary.json").is_file())
            self.assertTrue((run_directory / "artifacts" / "per_feature_metrics.csv").is_file())
            self.assertTrue((run_directory / "artifacts" / "test_predictions.csv").is_file())


class MatchedCapacityReadoutTests(unittest.TestCase):
    """What: 容量一致、grouped CV、再利用可能成果物を検証する。"""

    @classmethod
    def setUpClass(cls) -> None:
        """What: 全ケースで同じ容量一致モジュールを使用する。"""

        cls.module = load_matched_capacity_module()

    def build_small_config(self) -> object:
        """What: 単体テストで完走できる縮小設定を返す。"""

        candidate_specs = {
            "short_lag_orders": (
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
            ),
            "order1_long_lags": (
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 4),
                (1, 8),
                (1, 16),
                (1, 32),
                (1, 64),
            ),
        }
        return self.module.ExperimentConfig(
            fit_alphas=(0.34, 0.50, 0.66),
            test_alphas=(0.42, 0.58),
            fit_seeds=tuple(range(11, 19)),
            test_seeds=(31, 32, 33),
            burn_in=128,
            observation_length=256,
            candidate_specs=candidate_specs,
            cv_folds=2,
            selection_seed=7,
            ridge_penalties=(1.0e-2, 1.0),
            bootstrap_repetitions=50,
            bootstrap_seed=9,
        )

    def test_candidate_specs_are_exactly_16_dimensions(self) -> None:
        """What: 各TM候補が8複素座標、実16次元である。"""

        config = self.build_small_config()
        config.validate()
        for coordinates in config.candidate_specs.values():
            self.assertEqual(8, len(coordinates))
            self.assertEqual(8, len(set(coordinates)))

    def test_grouped_folds_keep_each_seed_together(self) -> None:
        """What: 全fit seedが一つのfoldだけへ割り当てられる。"""

        config = self.build_small_config()
        assignments = self.module.assign_grouped_folds(config)
        self.assertEqual(set(config.fit_seeds), set(assignments))
        counts = np.bincount(list(assignments.values()))
        self.assertLessEqual(int(np.max(counts) - np.min(counts)), 1)
        self.assertFalse(set(config.test_seeds) & set(assignments))

    def test_validator_reads_bom_csv_header(self) -> None:
        """What: validatorはBOM付きCSVの先頭列名を欠損させない。"""

        validator = load_matched_capacity_validator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "folds.csv"
            path.write_text("seed,fold\n11,0\n", encoding="utf-8-sig")
            rows = validator.read_csv(path)
        self.assertEqual("11", rows[0]["seed"])
        self.assertEqual("0", rows[0]["fold"])

    def test_small_run_preserves_features_metrics_and_plots(self) -> None:
        """What: 縮小runでも再利用NPZ、数値CSV、PNGを保存する。"""

        config = self.build_small_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            (run_directory / "config.json").write_text(
                json.dumps({"status": "planned"}),
                encoding="utf-8",
            )
            (run_directory / "environment.json").write_text("{}", encoding="utf-8")
            summary = self.module.run_experiment(run_directory, config)

            self.assertEqual("completed", summary["status"])
            self.assertEqual(16, summary["per_model"][0]["feature_dimension"])
            for name in (
                "features.npz",
                "trajectory_manifest.csv",
                "model_metrics.csv",
                "candidate_cv_metrics.csv",
                "test_predictions.csv",
                "bootstrap_differences.csv",
                "matched_capacity_results.png",
                "tm_candidate_selection.png",
            ):
                self.assertTrue((run_directory / "artifacts" / name).is_file(), name)
            restored = self.module.load_feature_dataset(
                run_directory / "artifacts" / "features.npz"
            )
            self.module.validate_feature_dataset(restored, config)


class TwoSourceIdentifiabilityTests(unittest.TestCase):
    """What: 2源の可逆観測、完全衝突、保存契約を検証する。"""

    @classmethod
    def setUpClass(cls) -> None:
        """What: 全ケースで同じE1Bモジュールを使用する。"""

        cls.module = load_two_source_module()

    def build_small_config(self) -> object:
        """What: 単体テストで完走できる縮小設定を返す。"""

        return self.module.ExperimentConfig(
            fit_alphas=(0.34, 0.50, 0.66),
            test_alphas=(0.42, 0.58),
            fit_pair_seeds=(101, 102, 103, 104),
            test_pair_seeds=(201, 202, 203),
            right_source_seed_offset=1000,
            burn_in=128,
            observation_length=256,
            mixing_matrix=np.asarray(
                [[1.0, 0.35], [0.25, 1.0]],
                dtype=np.float64,
            ),
            tm_coordinates=(
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
            ),
            fourier_band_count=8,
            cv_folds=2,
            selection_seed=7,
            ridge_penalties=(1.0e-4, 1.0),
            oracle_tm_rmse_threshold=1.0,
            collision_tolerance=1.0e-12,
        )

    def test_rank_one_swaps_are_exact_feature_collisions(self) -> None:
        """What: swapped目標の対称1観測特徴は点ごとに同一になる。"""

        config = self.build_small_config()
        config.validate()
        matched = self.module.load_matched_module()
        pilot = matched.load_pilot_module()
        source = self.module.build_source_dataset(config, pilot)
        dataset = self.module.build_feature_dataset(source, config, pilot, matched)
        for collision_id in np.unique(dataset.collision_id):
            indices = np.flatnonzero(dataset.collision_id == collision_id)
            self.assertEqual(2, indices.size)
            np.testing.assert_array_equal(
                dataset.features["direct_tm_rank_one"][indices[0]],
                dataset.features["direct_tm_rank_one"][indices[1]],
            )
            np.testing.assert_array_equal(
                dataset.features["direct_fourier_rank_one"][indices[0]],
                dataset.features["direct_fourier_rank_one"][indices[1]],
            )
            self.assertEqual(dataset.alpha1[indices[0]], dataset.alpha2[indices[1]])
            self.assertEqual(dataset.alpha2[indices[0]], dataset.alpha1[indices[1]])

    def test_theoretical_floor_matches_two_target_collision(self) -> None:
        """What: 同一入力の(a,b),(b,a)に対する順序付きRMSE下限は|a-b|/2。"""

        targets = np.asarray([[0.4, 0.6], [0.6, 0.4]], dtype=np.float64)
        predictions = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
        self.assertAlmostEqual(0.1, self.module.theoretical_collision_floor(targets))
        self.assertAlmostEqual(0.1, self.module.ordered_pair_rmse(targets, predictions))

    def test_multioutput_ridge_recovers_known_linear_targets(self) -> None:
        """What: 2目的ridgeが既知の線形写像を同時に回収する。"""

        matched = self.module.load_matched_module()
        pilot = matched.load_pilot_module()
        features = np.arange(1.0, 41.0, dtype=np.float64).reshape(-1, 2)
        targets = np.column_stack(
            (
                0.2 + 0.03 * features[:, 0],
                0.7 - 0.02 * features[:, 1],
            )
        )
        predictions = self.module.fit_multioutput_ridge(
            features,
            targets,
            features,
            1.0e-8,
            pilot,
        )
        self.assertLess(self.module.ordered_pair_rmse(targets, predictions), 1.0e-7)

    def test_small_run_saves_data_numbers_and_plots(self) -> None:
        """What: 縮小runでも実データ、特徴、全数値、PNGを保存する。"""

        config = self.build_small_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            (run_directory / "config.json").write_text(
                json.dumps({"status": "planned"}),
                encoding="utf-8",
            )
            (run_directory / "environment.json").write_text("{}", encoding="utf-8")
            summary = self.module.run_experiment(run_directory, config)

            self.assertEqual("completed", summary["status"])
            self.assertEqual(6, len(summary["per_model"]))
            self.assertLessEqual(
                summary["rank_one_collision_feature_max_abs_difference"],
                config.collision_tolerance,
            )
            for name in (
                "source_trajectories.npz",
                "features.npz",
                "source_trajectory_manifest.csv",
                "trajectory_manifest.csv",
                "fold_assignments.csv",
                "model_cv_metrics.csv",
                "model_metrics.csv",
                "test_predictions.csv",
                "pair_metrics.csv",
                "seed_metrics.csv",
                "collision_diagnostics.csv",
                "identifiability_results.png",
                "prediction_scatter.png",
                "summary.json",
            ):
                self.assertTrue((run_directory / "artifacts" / name).is_file(), name)
            restored_source = self.module.load_source_dataset(
                run_directory / "artifacts" / "source_trajectories.npz"
            )
            restored_features = self.module.load_feature_dataset(
                run_directory / "artifacts" / "features.npz"
            )
            self.module.validate_source_dataset(restored_source, config)
            self.module.validate_feature_dataset(restored_features, config)


if __name__ == "__main__":
    unittest.main()
