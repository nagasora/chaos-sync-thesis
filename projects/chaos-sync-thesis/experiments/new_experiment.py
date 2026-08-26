"""再現可能な実験記録フォルダをテンプレートから作成する。"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
STAGE_PATTERN = re.compile(r"^E\d+[A-Z]?$")


@dataclass(frozen=True)
class ExperimentSpec:
    """実験 ID と初期メタデータの生成に必要な値を保持する。"""

    stage: str
    slug: str
    title: str
    seed: int

    def validate(self) -> None:
        """許可した形式だけを受け入れ、安全で読みやすい ID を保つ。"""

        if not STAGE_PATTERN.fullmatch(self.stage):
            raise ValueError("stage は E0、E1A のような形式で指定してください。")
        if not SLUG_PATTERN.fullmatch(self.slug):
            raise ValueError("slug は小文字英数字をハイフンまたはアンダースコアで区切ってください。")
        if not self.title.strip():
            raise ValueError("title は空にできません。")
        if "\n" in self.title or "\r" in self.title:
            raise ValueError("title に改行は使用できません。")
        if self.seed < 0:
            raise ValueError("seed は 0 以上にしてください。")


def build_experiment_id(spec: ExperimentSpec, created_at: datetime) -> str:
    """作成日、段階、slug を連結して一意な実験 ID を組み立てる。"""

    spec.validate()
    return f"{created_at:%Y%m%d}_{spec.stage}_{spec.slug}"


def replace_tokens(text: str, values: Dict[str, str]) -> str:
    """単純なトークン置換で Markdown と JSON の雛形を展開する。"""

    rendered = text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def create_environment_record(created_at: datetime) -> Dict[str, object]:
    """実験作成時の Python と OS の識別情報を辞書へまとめる。"""

    return {
        "schema_version": 1,
        "recorded_at": created_at.isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": [],
        "execution_command": "",
        "notes": ["実行時に依存パッケージと実コマンドを追記する。"],
    }


def create_experiment(
    spec: ExperimentSpec,
    experiments_root: Path,
    created_at: datetime,
) -> Path:
    """テンプレートを展開し、上書きせずに実験記録一式を作成する。"""

    experiment_id = build_experiment_id(spec, created_at)
    experiment_dir = experiments_root / "runs" / experiment_id
    templates_dir = experiments_root / "templates"
    if experiment_dir.exists():
        raise FileExistsError(f"既存の実験は上書きできません: {experiment_dir}")

    required_templates: List[Tuple[str, str]] = [
        ("experiment_record.md", "README.md"),
        ("config.json", "config.json"),
        ("metrics.json", "metrics.json"),
    ]
    missing_templates = [
        source_name
        for source_name, _ in required_templates
        if not (templates_dir / source_name).is_file()
    ]
    if missing_templates:
        raise FileNotFoundError(f"テンプレートが不足しています: {missing_templates}")

    values = {
        "EXPERIMENT_ID": experiment_id,
        "TITLE": spec.title,
        "STAGE": spec.stage,
        "CREATED_AT": created_at.isoformat(),
        "SEED": str(spec.seed),
    }

    experiment_dir.mkdir(parents=True, exist_ok=False)
    for directory_name in ["artifacts", "logs"]:
        (experiment_dir / directory_name).mkdir()

    for source_name, destination_name in required_templates:
        source_text = (templates_dir / source_name).read_text(encoding="utf-8")
        destination_values = values
        if destination_name.endswith(".json"):
            # Why not: 生文字列を JSON へ差し込むと、引用符を含む実験名で構文が壊れる。
            destination_values = {
                **values,
                "TITLE": json.dumps(spec.title, ensure_ascii=False)[1:-1],
            }
        destination_text = replace_tokens(source_text, destination_values)
        if destination_name == "config.json":
            config = json.loads(destination_text)
            config["seeds"] = [spec.seed]
            destination_text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        (experiment_dir / destination_name).write_text(destination_text, encoding="utf-8")

    environment = create_environment_record(created_at)
    # Why not: YAML は追加依存が必要なため、標準ライブラリで検証できる JSON を正本にする。
    (experiment_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return experiment_dir


def parse_args(arguments: List[str]) -> argparse.Namespace:
    """コマンドライン引数を検証可能な名前空間へ変換する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="E0、E1A などの実験段階")
    parser.add_argument("--slug", required=True, help="小文字英数字の短い識別子")
    parser.add_argument("--title", required=True, help="人が読む実験名")
    parser.add_argument("--seed", required=True, type=int, help="最初に記録する乱数 seed")
    return parser.parse_args(arguments)


def main(arguments: List[str] | None = None) -> int:
    """CLI 入力から実験記録を作成し、作成先を表示する。"""

    args = parse_args(sys.argv[1:] if arguments is None else arguments)
    spec = ExperimentSpec(
        stage=args.stage,
        slug=args.slug,
        title=args.title,
        seed=args.seed,
    )
    experiments_root = Path(__file__).resolve().parent
    created_at = datetime.now().astimezone()
    experiment_dir = create_experiment(spec, experiments_root, created_at)
    print(experiment_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
