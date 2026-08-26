# 一般化Boole写像のCauchy不変測度・Lyapunov指数・TM係数再現

- 実験 ID: `20260806_E0_boole-invariant-measure`
- 段階: `E0`
- 状態: `completed`
- 作成日・実行日: 2026-08-06
- 総合判定: `PASS`

## 研究質問

一般化Boole写像

\[
F_\alpha(x)=\alpha\left(x-\frac{1}{x}\right)
\]

の有限精度・有限長軌道で、理論Cauchy尺度、閉形式Lyapunov指数、非零Cayleyモード期待値0を再現できるか。

## 仮説

`0 < alpha < 1` の検証点では、理論尺度

\[
\gamma_\alpha=\sqrt{\frac{\alpha}{1-\alpha}}
\]

と理論Lyapunov指数

\[
\lambda(\alpha)=2\log(\sqrt{\alpha}+\sqrt{1-\alpha})
\]

が有限軌道から事前定義誤差内で回収され、理論Cauchy測度に対する非零Cayleyモード平均は0近傍になる。

## 反証条件

- いずれかの `alpha` で事前定義ゲートを外れる。
- 非有限状態が発生する。
- 保存集約値と独立再集計が一致しない。

## データ

- 生成元: 一般化Boole写像による人工軌道。
- `alpha`: 0.4、0.5、0.6。
- seed: 20260806〜20260810の5個。
- 各軌道: burn-in 20,000点、観測100,000点。
- 合計: 15軌道、150万観測点。
- 初期化: 各 `alpha` の理論Cauchy不変測度。
- 保存: seed別指標CSVと、各`alpha`の先頭seedから20,000点の診断標本。

## 方法

- Cauchy尺度: `(Q75-Q25)/2` で推定。
- 分布整合: 理論尺度に対する両側KS距離。
- Lyapunov指数: 軌道平均 `mean(log|F'(x)|)`。
- 独立照合: Cauchy変数の角度座標を用いた20万点数値積分。
- TM候補: `q_gamma(x)^k`、`k=1,...,8` の軌道平均。
- 数値診断: 最小・最大絶対状態、非有限値、厳密な浮動小数点重複。

## 学習設計

学習は行わない。単一非結合ノードの数学・実装整合性だけを検証する。

## 評価結果

| alpha | 理論scale | scale最大相対誤差 | Lyapunov最大絶対誤差 | KS最大値 | TM非零最大値 | 判定 |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.4 | 0.816497 | 0.013948 | 0.001175 | 0.006502 | 0.007621 | PASS |
| 0.5 | 1.000000 | 0.007904 | 0.000031 | 0.004497 | 0.004496 | PASS |
| 0.6 | 1.224745 | 0.007178 | 0.000810 | 0.005901 | 0.006524 | PASS |

全alpha・全ゲートが通過した。全150万状態は有限で、各観測軌道内の厳密な浮動小数点重複は0だった。観測された最大絶対状態は約 `2.307e6`、最小絶対状態は約 `2.167e-7` で、Cauchy系とBoole写像の数値的な極端値も記録した。

## 成果物

- 実行済みNotebook: [E0_boole_validation.ipynb](E0_boole_validation.ipynb)
- 実験コード: [e0_boole_validation.py](e0_boole_validation.py)
- 独立検証コード: [validate_results.py](validate_results.py)
- 集約結果: [artifacts/summary.json](artifacts/summary.json)
- seed別結果: [artifacts/per_seed_metrics.csv](artifacts/per_seed_metrics.csv)
- 指標図: [artifacts/e0_metric_gates.png](artifacts/e0_metric_gates.png)
- 独立検証結果: [artifacts/validation.json](artifacts/validation.json)
- 検証報告: [VALIDATION_REPORT.md](VALIDATION_REPORT.md)

## 実行

```powershell
python e0_boole_validation.py --run-directory .
```

Notebookは研究専用kernelで上から実行し、14セル中5コードセルがすべて実行済み、エラー0、図1点を確認した。

## 解釈

今回の範囲では、一般化Boole写像を「Cauchy不変測度とLyapunov指数を解析・数値照合できる局所カオス」として実装できた。これは各ノードへBoole写像を置く数学的理由のうち、可解性の実装基盤を支持する。

一方、この結果は複数ノードを結合する必要性、同期による情報圧縮、TM特徴の源分離性能を示していない。

## 判断

`adopt`: E1Aの人工カオス源生成器としてこの実装を採用する。独立検証は37項目中37項目を通過した。

## 次の一手

E1Aで独立なBoole軌道の線形混合を生成し、raw・自己相関・Fourier・ICAとTM特徴の読み出し性能を比較する。非定常初期値からの収束速度と長時間周期化はE0の追加頑健性検証として残す。
