# 周辺尺度除去後のTM時間遅延特徴によるBoole写像パラメータ回収

- 実験 ID: `20260830_E1A_temporal-alpha-readout`
- 段階: `E1A`
- 状態: `completed`
- 作成日時: `2026-08-30T14:23:16.958233+09:00`
- seed: `20260830`

## 研究質問

一般化 Boole 写像の各軌道から中央値と半四分位幅を除き、周辺 Cauchy 尺度による識別を抑えた後でも、TM 高次モードの時間遅延特徴から未使用の補間 `alpha` を回収できるか。

## 仮説

写像パラメータ `alpha` は周辺尺度だけでなく時間発展にも影響する。したがって、Cayley 変換後の高次 TM モードに対する有限時間遅延相関には `alpha` 依存構造が残り、同じ ridge probe を使ったロバスト分位点、raw 窓、Fourier 帯域、1 次 Cayley 遅延より test RMSE が小さくなると予想する。

## 反証条件

- validation のみで選んだ最良 baseline と TM 時間遅延特徴の test RMSE 差について、seed-cluster bootstrap 95% 区間の下限が 0 以下になる。
- 軌道の時刻順をランダム化しても TM 時間遅延特徴の性能が保たれ、時間構造ではなく周辺分布または有限標本誤差を読んでいる疑いが残る。
- 未使用 seed ではなく時点分割でしか改善しない。

## データ

- 生成元または取得元: E0 で検証済みの `F_alpha(x)=alpha(x-1/x)` 軌道生成関数。
- 入力と目的変数: 入力は1軌道の有限時間特徴、目的変数は連続値 `alpha`。
- 軌道数・観測長: train 100、validation 40、test 48軌道。各4,096点。
- burn-in: 2,048点。
- train/validation/test の分割単位: seed単位。testはseedに加えて `alpha={0.38,0.46,0.54,0.62}` を丸ごと未使用にする。
- Cauchy サンプリングを使う箇所と理由: E0と同じく理論不変測度から初期化し、初期分布差を主要因にしない。

## 方法

- 前処理: 各軌道の中央値と半四分位幅で標準化する。尺度推定値そのものはprobeへ渡さない。
- モデルまたは写像: 非結合の一般化 Boole 写像。同期系はまだ導入しない。
- TM 基底の定義と尺度: 標準化後の `q(x)=(x-i)/(x+i)` と `q(x)^k`、`k=1,...,8`。
- 時間遅延: `tau={1,2,4,8}` で `E[q_t^k conjugate(q_{t-tau}^k)]` を有限軌道平均する。
- 学習方法: 全特徴集合に同一の ridge 線形 probe を使い、正則化をvalidation RMSEだけで選ぶ。

## 学習設計

- 学習するパラメータ: ridge probe の重みと切片。
- 固定するパラメータ: 写像、特徴定義、split、TM次数、時間遅延。
- 状態時間 `t` と学習ステップ `n` の更新順: 固定条件で全軌道を生成してからprobeを学習する。
- 損失項と各重み: `alpha` の二乗誤差のみ。
- 結合行列の制約: E1Aでは結合を導入しない。
- optimizer、勾配 clipping、停止条件: 閉形式ridge。正則化候補 `10^-6` から `10^2` をvalidationで選ぶ。
- test 時に凍結する対象: 前処理式、特徴定義、正則化、probeをすべて凍結する。

## 比較対象とアブレーション

- ベースライン: ロバスト分位点、Cayley raw窓、Fourier帯域、1次Cayley遅延。
- 除去する構成要素: TM遅延なし、および時刻順ランダム化後のTM遅延。

## 評価

- 主指標: 未使用 `alpha`・未使用 seed に対するRMSE、baseline RMSE minus TM RMSE。
- 副指標: validation RMSE、特徴次元、遅延なし・時刻順破壊対照のtest RMSE。
- seed 間の集約方法: 同じseedの異なる `alpha` を1クラスタとして2,000回bootstrapする。
- 成功条件: baseline minus TM の95% bootstrap区間下限が正。

## 実行

```powershell
python e1a_temporal_alpha_readout.py --run-directory .
```

## 結果

主成功条件は未達だった。

| 特徴 | validation RMSE | test RMSE |
|---|---:|---:|
| TM時間遅延 | 0.001881 | 0.002097 |
| Fourier帯域 | 0.002912 | 0.002382 |
| 1次Cayley遅延 | 0.011757 | 0.012217 |
| TM遅延なし | 0.109383 | 0.090611 |
| 時刻順破壊TM遅延 | 0.126543 | 0.104366 |

validationで選ばれた最良baselineはFourier帯域だった。testの点推定ではTM時間遅延が0.000286小さかったが、baseline minus TMのseed-cluster bootstrap 95%区間は `[-0.000334, 0.000886]` で0を含んだ。TMは4つの未使用 `alpha` のうち3つ、12 test seedのうち8つでFourierより良かった。

保存値は [summary.json](artifacts/summary.json)、[特徴別指標](artifacts/per_feature_metrics.csv)、[test予測](artifacts/test_predictions.csv) に記録した。独立再計算13項目は [validation.json](artifacts/validation.json) ですべて通過した。

## 解釈

TM時間遅延、Fourier帯域、1次Cayley遅延が周辺分位点やraw窓より大幅に良かったため、周辺尺度を除いた軌道にも `alpha` 依存の時間構造が残るという観察は得られた。遅延なしTMと時刻順破壊対照が悪化したことも、有限時間の順序情報が必要であるという説明と整合する。

一方、TM時間遅延がFourierより固有に優れるとは言えない。TMは80次元、Fourierは16次元であり、小さい点推定差は特徴次元、正則化、有限seedのいずれでも説明できる。同期、圧縮、源分離の有用性はこの実験の対象外である。

## 判断

`revise`: 「時間構造から未使用alphaを高精度に回収できる」は支持されたが、「TM時間遅延が最良baselineより優れる」は事前成功条件を満たさなかった。

## 次の一手

モデルを変更せずtest seedだけを増やす後継追試 [20260830_E1A_temporal-alpha-replication](../20260830_E1A_temporal-alpha-replication/README.md) を実行した。
