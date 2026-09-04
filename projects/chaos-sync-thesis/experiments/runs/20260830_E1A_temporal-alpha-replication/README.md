# TM時間遅延alpha回収の独立seed精度追試

- 実験 ID: `20260830_E1A_temporal-alpha-replication`
- 段階: `E1A`
- 状態: `completed`
- 作成日時: `2026-08-30T14:33:32.068699+09:00`
- seed: `20263830`

## 研究質問

pilotで観測したTM時間遅延特徴の小さなtest RMSE改善は、学習・validation・特徴・alpha集合を変えず、独立test seedを12から48へ増やしても再現するか。

## 仮説

pilotの点推定差が安定した効果なら、独立seedを4倍にした追試では baseline RMSE minus TM RMSE のseed-cluster bootstrap 95%区間下限が正になる。

## 反証条件

- 95%区間が引き続き0を含む。
- 点推定がFourier baseline優位へ反転する。

## データ

- 生成元または取得元: `20260830_E1A_temporal-alpha-readout` と同一実装。
- 入力と目的変数: 同一。
- 軌道数・観測長: train 100、validation 40は同一。testのみ48 seed x 4 alpha = 192軌道へ増やす。
- burn-in: 2,048点、観測4,096点で同一。
- train/validation/test の分割単位: seed単位。test seedはpilotとも非重複。
- Cauchy サンプリングを使う箇所と理由: pilotと同一。

## 方法

- 前処理: pilotから変更なし。
- モデルまたは写像: pilotから変更なし。
- TM 基底の定義と尺度: pilotから変更なし。
- 時間遅延: pilotから変更なし。
- 学習方法: pilotと同じtrain/validationで同じridge選択を再現する。

## 学習設計

- 学習するパラメータ: pilotと同じridge probeのみ。
- 固定するパラメータ: test seed以外の全条件。
- 状態時間 `t` と学習ステップ `n` の更新順: pilotと同一。
- 損失項と各重み: pilotと同一。
- 結合行列の制約: 結合なし。
- optimizer、勾配 clipping、停止条件: pilotと同一。
- test 時に凍結する対象: 前処理、特徴、正則化選択規則、probe。

## 比較対象とアブレーション

- ベースライン: pilotのvalidationで選ばれるFourier帯域を含む同一集合。
- 除去する構成要素: なし。今回はtest標本数だけを変更する。

## 評価

- 主指標: baseline RMSE minus TM RMSEの95% bootstrap区間。
- 副指標: test RMSE点推定、alpha別RMSE、seed別勝敗。
- seed 間の集約方法: 48 test seedをクラスタとして5,000回bootstrapする。
- 成功条件: 区間下限が正。

## 実行

```powershell
python e1a_temporal_alpha_replication.py --run-directory .
```

## 結果

主成功条件は追試でも未達だった。

| 特徴 | test RMSE |
|---|---:|
| TM時間遅延 | 0.002096 |
| Fourier帯域 | 0.002356 |
| 1次Cayley遅延 | 0.012181 |
| TM遅延なし | 0.090291 |
| 時刻順破壊TM遅延 | 0.099228 |

TMの点推定差はpilotとほぼ同じ `+0.000260` だったが、95%区間は `[-0.000023, 0.000561]` でわずかに0を含んだ。TMは4つの未使用 `alpha` のうち3つ、48 test seedのうち27でFourierより良かった。`alpha=0.54` ではFourierが良く、差が一様ではない。

保存値は [summary.json](artifacts/summary.json)、[特徴別指標](artifacts/per_feature_metrics.csv)、[test予測](artifacts/test_predictions.csv) に記録した。独立再計算13項目は [validation.json](artifacts/validation.json) ですべて通過した。

## 解釈

独立seedを4倍にしても点推定差とTMの絶対RMSEがほぼ再現したため、pilotの数値が少数seedだけの偶然だった可能性は下がった。ただし事前ゲートは越えず、TMのFourierに対する優位を確定できない。時間順序を破壊すると性能が2桁悪化するため、有限時間の動力学が回収に必要という観察は再現した。

この追試は標本数だけを変えており、80次元TMと16次元Fourierの容量差、`alpha=0.54` の局所逆転、観測長依存は未解決である。

## 判断

`revise`: TM優位の主張は採用しない。より限定した結論「TMとFourierはいずれも周辺尺度除去後の時間構造からalphaを回収でき、TMの小さな点推定改善は未確定」を採用する。

## 次の一手

seedだけを追加する追試はここで止める。次は特徴次元を16へ揃えたTM次数・遅延のnested-validationアブレーションを新しいtest seedで行い、差が80次元の容量によるものかを1要因だけ検証する。
