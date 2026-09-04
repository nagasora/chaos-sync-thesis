# Chaos Sync / TM Self Implementation

このフォルダは、一般化 Boole 写像・Cauchy 不変測度・TM 基底・カオス同期・情報保持実験を、実験テキストに沿って自力で実装するための作業領域である。

## 目的

- 理論式を自分で導出し、最小単位の数値検証から実装する。
- 「同期による自由度縮約」と「入力情報を保持した圧縮・復号」を別々に評価する。
- 添付原稿の主張をそのまま前提にせず、再現計算とテストで確認する。
- notebook に処理を閉じ込めず、再利用可能な処理は `src/`、仕様は `tests/`、条件は `configs/` に分離する。

## フォルダ構成

```text
chaos-sync-tm-self-implementation/
├── README.md       # この案内
├── docs/           # 実験テキスト、LaTeX正本、配布ZIP
├── notebooks/      # 探索・可視化用。正本ロジックは置かない
├── src/            # 写像、特徴抽出、結合系、評価指標の実装
├── tests/          # 数式・不変量・入出力仕様のテスト
├── configs/        # E0〜E5の実験条件
├── runs/           # 実行ごとの不変成果物
└── reports/        # 結果・考察・採否判断
```

## 実験の順序

1. **E0**: 単一写像、Cauchy 不変測度、Lyapunov 指数、TM 基底
2. **E1**: TM 時間特徴の読み出しと識別可能性
3. **E2**: 観測雑音、欠測、有限観測長への頑健性
4. **E3**: 同期転移、局所安定性、大域 basin、入力情報保持
5. **E4**: 合成時系列の圧縮・復元
6. **E5**: 8×8 digits による画像 pilot

E0〜E3を卒論の最小スコープとし、E4・E5は基礎検証を通過した後に進める。

## 実装ルール

- 関数には型ヒントと Docstring を付ける。
- コードは **How**、テストは **What**、コミットメッセージは **Why** を明確にする。
- コメントには、採用しなかった素朴な方法とその理由、すなわち **Why not** を残す。
- 理論値、数値値、添付原稿由来の主張、仮説を混同しない。
- `runs/YYYYMMDD_<EXPERIMENT_ID>_<slug>/` に設定、seed、環境、全予測、集約値、図の元データ、考察を保存する。

## 最初の作業

1. `docs/chaos_sync_textbook_starter.zip` をローカルまたは Colab に展開する。
2. 次の3関数から実装する。
   - `generalized_boole`
   - `boole_derivative`
   - `simulate_boole_orbit`
3. smoke test を通す。

```bash
python -m pip install -e ".[dev]"
pytest -q tests/test_scaffold.py
```

4. E0 の仕様テストを通す。

```bash
pytest -q -m exercise
```

5. 数値結果が出たら、必ず `reports/` に理論値との差、誤差原因、採否、次の単一仮説を書く。

## Colab での利用

```python
from google.colab import drive
drive.mount('/content/drive')
```

Drive 上のファイルを直接編集するより、実行時は `/content` にコピーし、終了時に設定・結果・レポートを `runs/` へ同期する。これにより Drive I/O による遅延や中途半端な成果物を避ける。
