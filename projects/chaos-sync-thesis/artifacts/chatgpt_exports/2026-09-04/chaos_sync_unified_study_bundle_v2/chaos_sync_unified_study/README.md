# カオス同期研究：統一E系列実験スイート

旧E系統とT系統を、研究依存順の正規ID `E0→E1→E2→E3→E4→E5` へ統一した再現パッケージである。既存runのフォルダ名は改名せず、`experiment_registry.json` の `legacy_ids` で別名解決する。

## 実行

```bash
cd chaos_sync_unified_study
PYTHONPATH=src python -m chaos_sync_unified.run_all --root . --force
pytest
```

## 新規に完了するrun

- `E2A-BOOLE-OBSERVATION-ROBUSTNESS`: ノイズ・欠損・有限観測長
- `E3B-TANGENT-TWO-NODE-BASIN`: 局所横安定性と大域同期basin
- `E3C-BOOLE-N8-INPUT-RETENTION`: 同期と入力保持のトレードオフ
- `E4A-BOOLE-SYNTHETIC-SIGNAL-RECONSTRUCTION`: 合成波形rate–distortion
- `E5A-BOOLE-SMALL-IMAGE-DIGITS`: 8×8小画像pilot

`E5B`の28×28 MNIST本確認は、E5Aと区別して未実行としてregistryに残す。

## 完了条件

各runは `config.json`, `environment.json`, `validation.json`, 数値CSV/JSON, 予測, 図, 入力/特徴NPZ, SHA-256台帳を保存する。科学仮説が棄却されても、実験完全性が通れば `completed` とする。
