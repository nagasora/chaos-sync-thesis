# 実験記録ガイド

## 現行レイアウト（2026-09-05以降）

新規実験の正本は段階別の experiments/E0、E1… とし、共通実装は experiments/src に置く。各段階は README.md（レポート）、事前設定、実験固有runner・validator、artifacts/<目的が分かる実行名>/ を持つ。[E0](E0/README.md)を具体例とする。

数値はCSV/NPZ/JSON、図はPNG、実際の入力または決定論的生成条件、実行時ソース、環境、SHA-256、独立検証を保存する。出力先が存在する場合は停止し、新しい実行名を指定する。合否未達でも全証拠と考察を揃えた実験記録は完了になり得るが、研究仮説の支持や後段への移行許可とは区別する。

以下の runs/YYYYMMDD_STAGE_slug と new_experiment.py の説明は旧run向けである。旧成果物を改名・移動せず、既存リンクとハッシュを維持する。

## 目的

実験の成功・失敗・未実行を同じ形式で残し、「どの仮説を、何と比較し、何を根拠に次へ進んだか」を後から再現できるようにする。

## ディレクトリ構成

```text
experiments/
├─ new_experiment.py
├─ templates/
└─ runs/
   └─ YYYYMMDD_STAGE_slug/
      ├─ README.md
      ├─ config.json
      ├─ environment.json
      ├─ metrics.json
      ├─ artifacts/
      └─ logs/
```

`README.md` は人間向けの正本、`config.json` と `metrics.json` は機械可読な正本とする。大きな生データは複製せず、由来、取得日、ハッシュ、前処理コードを記録する。

## 新規実験の作成

プロジェクト直下で次を実行する。

```powershell
python experiments/new_experiment.py --stage E1A --slug tm-linear-mixture --title "TM基底による人工カオス線形混合の読み出し" --seed 20260806
```

同じ実験 ID が存在する場合は失敗し、既存記録を上書きしない。

## 実験ライフサイクル

1. `planned`: 仮説、主指標、比較対象、成功条件、反証条件を先に書く。
2. `running`: 実行コマンド、環境、実データ識別子を記録する。
3. `completed` または `failed`: 図表と指標だけでなく、解釈と失敗理由を書く。
4. `superseded`: 後継実験 ID と、置き換えた理由を書く。

## 実行前チェック

- 一度に変更する主要因は一つか。
- train/validation/test を軌道または seed 単位で分けたか。
- 前処理、基底尺度、ハイパーパラメータを test に適合していないか。
- ベースラインと反証条件が書かれているか。
- Cauchy のような重い裾に対して、存在しない平均・分散を前提にしていないか。
- 学習する結合と固定する結合を区別したか。
- 状態時間 `t` と学習ステップ `n` の更新順を記録したか。
- エントロピー項だけでなく、復号・同期・結合正則化を別々に記録するか。

## 実行後チェック

- `config.json` に実際の設定が反映されているか。
- `environment.json` に実行環境と依存関係を追記したか。
- `metrics.json` に集約値だけでなく seed 別結果への参照があるか。
- 図の元データを `artifacts/` に保存したか。
- 仮説を支持しなかった結果も削除していないか。
- `README.md` の「判断」と「次の一手」を更新したか。
- test 時に結合と読み出しを凍結したか。

## ID と上書き方針

- ID は `YYYYMMDD_STAGE_slug` とする。
- 同条件の再実行も新しい ID または `replicate` を含む slug にする。
- 既存の成果物は上書きせず、修正実験から元実験 ID を参照する。
- 一時ファイルと最終成果物を分ける。
