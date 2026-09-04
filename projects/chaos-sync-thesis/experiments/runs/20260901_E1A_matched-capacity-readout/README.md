# TMとFourierの16次元容量一致alpha読み出し

- 実験 ID: `20260901_E1A_matched-capacity-readout`
- 段階: `E1A`
- 状態: `completed`
- 作成日時: `2026-09-01T01:03:58.367787+09:00`
- 親実験: `20260830_E1A_temporal-alpha-replication`

## 研究質問

80次元TM時間遅延と16次元Fourierの容量差を除き、両者を16次元へ揃えても、周辺Cauchy尺度を除去した有限軌道から未使用 `alpha` を読む性能差が残るか。

## 仮説

前回のTM点推定改善が単なる80次元の容量差でなければ、fit seedだけのgrouped CVでTM次数・遅延を選び、16次元へ固定した後も、新規test seedにおける `Fourier RMSE - TM RMSE` のseed-cluster bootstrap 95%区間下限が0より大きくなる。

## 反証条件

- 主比較の95%区間が0を含む。
- 点推定がFourier優位へ反転する。
- 選択TMまたはFourierの最終特徴次元が16でない。
- test seed、test alpha、test予測を特徴・ridge選択へ使用する。

## データ

- 生成元: 親実験で検証した一般化Boole写像 `F_alpha(x)=alpha(x-1/x)` の決定論的軌道生成器。
- 入力と目的変数: 軌道から抽出した有限時間特徴から写像パラメータ `alpha` を回帰する。
- fit: 5 alpha x 28 seed = 140軌道。以前のtrain/validation seedをまとめ、seed単位4-fold CVだけに使用する。
- test: 未使用補間4 alpha x 新規48 seed = 192軌道。一度だけ最終評価する。
- burn-in: 2,048点。観測長: 4,096点。
- 使用データの保存: 全軌道識別子を `artifacts/trajectory_manifest.csv`、実際に回帰へ使う特徴行列を `artifacts/features.npz` に保存する。軌道そのものは設定とseedから決定論的に再生成できるため複製しない。

## 方法

- 前処理: 軌道ごとの中央値と半四分位幅で標準化し、尺度1のCayley変換へ写す。
- Fourier: 実部・虚部を各8周波数帯へ分けた対数パワー16次元。
- TM候補: 8個の複素座標を実部・虚部へ展開した16次元。候補4組は実行前に `config.json` へ固定する。
- 選択: 28 fit seedを4 foldへ分け、TM候補とridgeをgrouped CVで同時選択する。Fourier、80次元TM参照、遅延なしTM16次元も同じfoldでridgeを選ぶ。
- test: 選択後に全fit軌道で再学習し、前処理・特徴・ridgeを凍結して新規testだけを評価する。

## 比較対象とアブレーション

- 主比較: nested選択TM 16次元 vs Fourier 16次元。
- 容量参照: TM時間遅延80次元。
- 時間順序対照: 瞬時TMモーメント16次元。

## 評価

- 主指標: `Fourier test RMSE - selected TM test RMSE`。
- 区間推定: 48 test seedをクラスタとして10,000回bootstrapする。
- 副指標: model別CV/test RMSE、alpha別RMSE、seed別RMSE差、選択TM座標。
- 成功条件: 主指標の95% bootstrap区間下限が0より大きい。

## 再現性・保存契約

- 実行コード、独立validator、設定、環境、コードSHA-256をrun内に保存する。
- test予測、model別指標、candidate別CV、fold割当、alpha別・seed別指標、bootstrap標本をCSV/JSONで保存する。
- プロット元CSVとPNGを同じ `artifacts/` に保存し、コードから再生成できるようにする。
- `--reuse-features artifacts/features.npz` で、軌道再生成なしに解析とプロットを再実行できるようにする。

## 実行

```powershell
python e1a_matched_capacity_readout.py --run-directory .
python validate_results.py --run-directory .
```

## 結果

主成功条件を満たした。fit seedだけのgrouped CVは、瞬時 `k=1,...,4` と遅延1の `k=1,...,4` を組み合わせた `short_lag_orders` を選択した。

| model | 次元 | grouped CV RMSE | test RMSE |
|---|---:|---:|---:|
| 選択TM時間遅延 | 16 | 0.001826 | 0.001342 |
| Fourier帯域 | 16 | 0.003090 | 0.002614 |
| 全TM時間遅延参照 | 80 | 0.001760 | 0.001318 |
| 遅延なしTM | 16 | 0.113775 | 0.094659 |

主比較 `Fourier RMSE - TM RMSE` は `0.001272`、48 test seedのcluster bootstrap 95%区間は `[0.000985, 0.001560]` で0より大きかった。選択TMは48 seed中44 seed、4個すべての未使用 `alpha` でFourierより小さいRMSEとなった。Fourierに対する相対RMSE低下は点推定で約48.6%だった。

数値正本は [summary.json](artifacts/summary.json)、[model_metrics.csv](artifacts/model_metrics.csv)、[test_predictions.csv](artifacts/test_predictions.csv)、[alpha_metrics.csv](artifacts/alpha_metrics.csv)、[seed_metrics.csv](artifacts/seed_metrics.csv) に保存した。独立validatorは19/19項目を通過した。

プロットは [容量一致結果](artifacts/matched_capacity_results.png) と [TM候補選択](artifacts/tm_candidate_selection.png) に保存した。図の元データは `model_metrics.csv`、`candidate_cv_metrics.csv`、`bootstrap_differences.csv` である。

再利用可能なモデル入力は [features.npz](artifacts/features.npz)、使用軌道の識別子は [trajectory_manifest.csv](artifacts/trajectory_manifest.csv)、コード・全artifactのSHA-256は [environment.json](environment.json) に保存した。

## 解釈

16次元同士でもTM時間遅延がFourierより良く、80対16次元という容量差だけでは前回のTM点推定改善を説明できない。選択されたのは長遅延や単一次数ではなく、複数次数の瞬時モードと最短遅延の組であり、この写像では短時間の次数横断構造が `alpha` 回収に有効という経験的証拠になった。80次元TMとの差は小さく、主要情報を16次元へ縮約できた。

一方、これはTM基底の一般的優位を証明しない。TM側は4候補からgrouped CVで選択したのに対しFourierの帯域定義は固定であり、選択自由度は非対称である。また単一源・補間alpha・線形ridgeだけの結果で、同期、複数源分離、圧縮符号長、画像復号は未検証である。有限標本の尺度推定誤差を読んでいる可能性も残る。

## 判断

`adopt`: 「一般化Boole単一源の有限時間alpha回収では、容量を16次元へ揃えても、選択TM短遅延特徴は固定Fourier帯域より良かった」を採用する。主張はこのデータ生成系と選択手順に限定する。

## 次の一手

同じtestへ追加適合しない。次は選択済み16次元TMを凍結し、2源2観測full-rank混合と2源1観測の識別不能対照を同一生成条件で比較する。
