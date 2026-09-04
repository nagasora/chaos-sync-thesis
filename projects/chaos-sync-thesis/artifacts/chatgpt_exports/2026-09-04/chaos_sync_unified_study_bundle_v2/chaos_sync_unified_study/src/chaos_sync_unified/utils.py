from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    """How: NumPy scalar・配列を再現可能な JSON へ変換する。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def write_json(path: Path, payload: Any) -> None:
    """How: 親フォルダを作り UTF-8 JSON を原子的に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """How: index を除き UTF-8 CSV として保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8")


def sha256_file(path: Path) -> str:
    """How: artifact の byte-level provenance を SHA-256 で固定する。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_hashes(root: Path, *, exclude_suffixes: tuple[str, ...] = (".pyc",)) -> pd.DataFrame:
    """How: run 配下の全ファイルを相対パスと hash の台帳へする。"""
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix in exclude_suffixes or "__pycache__" in path.parts:
            continue
        rows.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def environment_payload() -> dict[str, Any]:
    """How: Python・OS・主要依存版・git状態を記録する。"""
    versions: dict[str, str] = {}
    for module_name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "reportlab"):
        module = __import__(module_name)
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "dependencies": versions,
        "git_commit": git_commit,
    }
