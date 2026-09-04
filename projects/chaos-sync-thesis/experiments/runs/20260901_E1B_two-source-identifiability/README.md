# 2源full-rank混合と1観測の識別可能性対照

- 実験 ID: `20260901_E1B_two-source-identifiability`
- 段階: `E1B`
- 状態: `completed`
- 作成日時: `2026-09-01T01:39:28.879462+09:00`
- 親実験: `20260901_E1A_matched-capacity-readout`

## 研究質問

E1Aで凍結した16次元TM短遅延特徴は、2つの独立な一般化Boole源を既知の可逆2観測から復元した場合にも、順序付きパラメータ `(alpha_1, alpha_2)` を読めるか。また、対称な1観測では源入替えにより順序付き目標が原理的に識別不能になることを、完全衝突と誤差下限で確認できるか。

## 理論的定式化

独立な2源を

```text
s_{j,t+1} = F_{alpha_j}(s_{j,t}),
F_alpha(x) = alpha (x - 1/x),  j = 1,2
```

とする。各有限軌道は混合前に中央値と半四分位幅で標準化し、Cauchy周辺尺度だけでalphaを読む経路を除く。

### full-rank 2観測

```text
y_t = A s_t,
A = [[1, 0.35],
     [0.25, 1]],
det(A) = 0.9125 != 0.
```

`A` が既知なら `s_t = A^{-1} y_t` で源の順序まで復元できる。したがって、凍結した単一源特徴写像とridge probeが各alphaを回収できる範囲では、順序付き2パラメータも識別可能である。これはblind source separationではなく、観測写像が情報を失っていないことを確認する正対照である。

### 対称なrank-one 1観測

```text
z_t = (s_{1,t} + s_{2,t}) / sqrt(2).
```

変換 `(s_1, alpha_1) <-> (s_2, alpha_2)` は全時刻の `z_t` を変えない。そこで各unordered alpha pairとseed対から、同じ `z_t` を持つ2行に異なる順序付き目標 `(a,b)` と `(b,a)` を割り当てる。

同一観測に対する決定論的予測を `(p_1,p_2)` とすると、衝突2行の平均二乗誤差は `(p_1,p_2)=((a+b)/2,(a+b)/2)` で最小になる。そのとき全座標に対するRMSE下限は

```text
RMSE_floor(a,b) = |a-b| / 2.
```

複数衝突対を等重みで集約したtest下限は `sqrt(mean(((alpha_1-alpha_2)/2)^2))` である。順序を無視するpermutation-invariant RMSEは小さくなり得るため、順序付きRMSEと分けて保存する。

## 仮説と反証条件

- H1: oracle full-rank TMの順序付きtest RMSEは `0.005` 未満になる。
- H2: swapped衝突対のrank-one特徴の最大絶対差は `1e-12` 以下になる。
- H3: rank-one TM/Fourierの順序付きtest RMSEは理論衝突下限を下回らない。

H1が不成立なら、可逆観測でも凍結E1A特徴またはprobeが2源alpha回収へ移らない。H2が不成立なら、衝突構成または前処理実装が理論対称性を壊している。H3が不成立なら、指標・行対応・リークのいずれかに誤りがある。test alphaまたはtest pair seedをridge選択へ使った場合も実験を無効とする。

## データ

- fit alpha: `0.34, 0.42, 0.50, 0.58, 0.66` の異なる2値全10組。
- test alpha: 未使用補間値 `0.38, 0.46, 0.54, 0.62` の異なる2値全6組。
- fit: 16 pair seed x 10組 x 2順序 = 320行。
- test: 32 pair seed x 6組 x 2順序 = 384行。
- burn-in: 2,048点。観測長: 4,096点。
- 分割: pair seed単位4-fold grouped CV。fit/test seedは非重複。
- 左源seedはpair seed、右源seedは `pair_seed + 1,000,000`。swapped行では同じ2軌道の順序だけを交換する。
- 使用データは標準化済み源軌道を `artifacts/source_trajectories.npz`、モデル入力を `artifacts/features.npz`、行識別子を `artifacts/trajectory_manifest.csv` に保存する。

## 方法と比較対象

E1Aのgrouped CVで選んだTM座標 `[(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(4,1)]` を再選択せず凍結する。各複素座標を実部・虚部へ展開して1チャネル16次元とする。Fourierも1チャネル16次元へ固定する。

| model | 観測 | 特徴次元 | 役割 |
|---|---|---:|---|
| `oracle_tm_full_rank` | 既知Aで逆変換 | 32 | 主正対照 |
| `oracle_fourier_full_rank` | 既知Aで逆変換 | 32 | 基底baseline |
| `direct_tm_full_rank` | 2混合チャネル | 32 | 直接読み出し探索 |
| `direct_fourier_full_rank` | 2混合チャネル | 32 | 直接baseline |
| `direct_tm_rank_one` | 対称1チャネル | 16 | 識別不能対照 |
| `direct_fourier_rank_one` | 対称1チャネル | 16 | 識別不能baseline |

全モデルは同じmulti-output ridgeを使い、ridge強度だけをfit pair seedのgrouped CVで選ぶ。testでは特徴定義、逆変換、前処理、ridgeをすべて凍結する。

## 評価と保存契約

- 主指標: H1のoracle TM順序付きRMSE、H2の衝突特徴差、H3のrank-one RMSEと理論下限。
- 副指標: model別CV/fit/test、permutation-invariant RMSE、alpha pair別・seed別RMSE。
- 数値正本: JSONとCSV。図の元表も同じCSVとして保存する。
- 図: model別誤差と理論下限、真値対予測の2枚をPNGで保存する。
- 再現性: config、データNPZ、特徴NPZ、コード、validator、依存版、コード・artifact SHA-256をrun内に保存する。
- 再利用: `--reuse-data` で軌道生成を省略し、`--reuse-features` で特徴抽出も省略して再集計・再描画できるようにする。

## 実行

```powershell
python e1b_two_source_identifiability.py --run-directory .
python validate_results.py --run-directory .
```

## 結果

H1〜H3の事前成功条件をすべて満たした。標準化済み源軌道352組、順序付き特徴704行（fit 320、test 384）を生成し、fit pair seedだけの4-fold grouped CVでridge強度を選んだ。

| model | 次元 | CV ordered RMSE | test ordered RMSE | test permutation-invariant RMSE |
|---|---:|---:|---:|---:|
| oracle TM full-rank | 32 | 0.002429 | 0.002083 | 0.002083 |
| oracle Fourier full-rank | 32 | 0.003109 | 0.003051 | 0.003051 |
| direct TM full-rank | 32 | 0.009502 | 0.007740 | 0.007740 |
| direct Fourier full-rank | 32 | 0.011008 | 0.009070 | 0.009070 |
| direct TM rank-one | 16 | 0.090014 | 0.073710 | 0.073710 |
| direct Fourier rank-one | 16 | 0.090335 | 0.073891 | 0.073891 |

- H1: oracle TMのtest RMSEは `0.002083 < 0.005` で通過した。
- H2: 全352衝突対でrank-one TM/Fourier特徴の最大絶対差は厳密に `0.0` だった。
- H3: testの理論衝突下限は `0.073030`。rank-one TMは `0.073710`、Fourierは `0.073891` で、いずれも下限を下回らなかった。
- oracle条件のTMはFourierより点推定で約31.7%低いRMSE、direct full-rankでは約14.7%低かった。ただしこの差の区間推定は事前主指標ではないため、TM一般優位とは扱わない。
- direct full-rank TMはrank-one理論下限より約89.4%低く、明示的な逆変換なしの固定2チャネル特徴にも順序付きalpha情報が残った。

数値正本は [summary.json](artifacts/summary.json)、[model_metrics.csv](artifacts/model_metrics.csv)、[test_predictions.csv](artifacts/test_predictions.csv)、[pair_metrics.csv](artifacts/pair_metrics.csv)、[seed_metrics.csv](artifacts/seed_metrics.csv)、[collision_diagnostics.csv](artifacts/collision_diagnostics.csv) に保存した。独立validatorは20/20項目を通過した。

使用した標準化済み源軌道は [source_trajectories.npz](artifacts/source_trajectories.npz)、全モデル入力は [features.npz](artifacts/features.npz)、由来と行対応は [source manifest](artifacts/source_trajectory_manifest.csv) と [trajectory manifest](artifacts/trajectory_manifest.csv) に保存した。

プロットは [識別可能性比較](artifacts/identifiability_results.png) と [真値対予測](artifacts/prediction_scatter.png) に保存した。図の元データは `model_metrics.csv` と `test_predictions.csv` である。コード・全artifactのSHA-256と依存版は [environment.json](environment.json) に保存した。

## 解釈

観察として、既知の可逆2観測では源を逆変換した後の凍結TM特徴が2つの未使用alphaを小さい誤差で回収し、E1Aの単一源読み出しが2源の正対照へ移った。直接2観測でも誤差はoracleより増えるが、rank-one下限からは十分離れており、固定混合チャネルの時間特徴だけでも順序情報の多くを読めた。

一方、対称1観測ではforward/swappedの時系列と特徴が完全に一致する。この完全衝突はモデル性能や標本数では解消できず、順序付き目標のRMSEが理論下限付近に張り付いた。したがって「TMが失われた源順序を復元した」とは言えず、観測写像が捨てた自由度を特徴基底は作り直せないという研究上の境界を数値で確認した。

permutation-invariant RMSEも本runではordered RMSEと同じだった。これは順序付き教師で学習したridgeが衝突対の中点に近い同一2座標を返したためであり、rank-one観測からunordered alpha pairが識別不能だと証明したものではない。unordered目標を評価するには、sorted target専用probeを別実験として学習する必要がある。

限界は、`A`既知のoracle、混合前の源別標準化、補間alpha、線形ridgeという理想化である。本runはblind source separation、同期、圧縮符号長、画像復号を検証していない。TMとFourierの点推定差も、この1つの混合行列と固定特徴定義に限定する。

## 判断

`adopt`。「可逆2観測では凍結時間特徴から順序付き2源alphaを回収できるが、対称1観測では源入替えの完全衝突により順序付き復号に厳密な誤差下限が生じる」を、この理想化生成系に限定して採用する。

## 次の一手

E2Aとして現在のfull-rank行列と全分割を固定し、観測ノイズ強度だけを変える。理論上は逆変換誤差が `||A^{-1}||` により増幅されるため、oracle/direct TM・FourierがどのSNRで順序付き復号を失うかを事前閾値つきで測る。行列条件数の走査はノイズ強度を固定した後の別実験とし、二要因を同時に変えない。
