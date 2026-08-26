"""実験記録の作成契約を確認するテスト。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import List


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "new_experiment.py"


def load_script_module() -> ModuleType:
    """What: 配布形態のまま CLI スクリプトを読み込み、公開関数をテストする。"""

    spec = importlib.util.spec_from_file_location("new_experiment", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"スクリプトを読み込めません: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Why not: exec_module だけでは dataclass が参照する sys.modules に登録されない。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CreateExperimentTests(unittest.TestCase):
    """What: 作成物、上書き防止、入力検証の三つを確認する。"""

    def setUp(self) -> None:
        """What: 各テスト専用の一時 experiments ルートへ雛形を複製する。"""

        self.module = load_script_module()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.experiments_root = Path(self.temporary_directory.name)
        self.templates_dir = self.experiments_root / "templates"
        self.templates_dir.mkdir()
        source_templates = SCRIPT_PATH.parent / "templates"
        for name in ["experiment_record.md", "config.json", "metrics.json"]:
            (self.templates_dir / name).write_text(
                (source_templates / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        """What: テスト専用ディレクトリを破棄する。"""

        self.temporary_directory.cleanup()

    def test_create_experiment_writes_required_files(self) -> None:
        """What: 必須ファイルと seed が初回作成で保存される。"""

        spec = self.module.ExperimentSpec("E1A", "tm-linear-mixture", "読み出し", 42)
        created_at = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)
        experiment_dir = self.module.create_experiment(
            spec,
            self.experiments_root,
            created_at,
        )

        expected_names: List[str] = [
            "README.md",
            "config.json",
            "environment.json",
            "metrics.json",
            "artifacts",
            "logs",
        ]
        self.assertEqual(
            sorted(expected_names),
            sorted(path.name for path in experiment_dir.iterdir()),
        )
        config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual([42], config["seeds"])
        self.assertEqual("20260806_E1A_tm-linear-mixture", config["experiment_id"])

    def test_config_template_is_valid_json(self) -> None:
        """What: 未展開の設定テンプレートもJSONツールで直接検証できる。"""

        template_text = (self.templates_dir / "config.json").read_text(encoding="utf-8")
        config = json.loads(template_text)
        self.assertEqual([0], config["seeds"])

    def test_existing_experiment_is_not_overwritten(self) -> None:
        """What: 同一 ID の二回目作成は既存記録を保護して失敗する。"""

        spec = self.module.ExperimentSpec("E1A", "duplicate", "重複", 7)
        created_at = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)
        self.module.create_experiment(spec, self.experiments_root, created_at)

        with self.assertRaises(FileExistsError):
            self.module.create_experiment(spec, self.experiments_root, created_at)

    def test_invalid_slug_is_rejected(self) -> None:
        """What: パスとして曖昧な slug を拒否する。"""

        spec = self.module.ExperimentSpec("E1A", "Invalid Slug", "不正", 1)
        with self.assertRaises(ValueError):
            spec.validate()

    def test_title_with_quote_remains_valid_json(self) -> None:
        """What: 引用符を含む実験名でも config JSON を正しく保存する。"""

        spec = self.module.ExperimentSpec("E0", "quoted-title", '尺度 "gamma" の確認', 3)
        created_at = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)
        experiment_dir = self.module.create_experiment(
            spec,
            self.experiments_root,
            created_at,
        )

        config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual('尺度 "gamma" の確認', config["title"])


if __name__ == "__main__":
    unittest.main()
