# ランダム結合一般化Boole写像の有限サイズ同期転移pilot

- 実験 ID: `20260830_E3A_finite-size-sync-transition`
- 段階: `E3A`
- 状態: 完了（解析 revision 2、2026-08-31）
- 親実験: `20260806_E0_boole-invariant-measure`

## 研究質問

無限次元の独立Cauchyスケール閉包が予測する臨界値

\[
K_c=\frac{2\sqrt{\alpha}(1-\sqrt{\alpha})}{\mathbb E|\varepsilon|}
\]

は有限サイズの対称結合系で観測されるか。また、同じ `E|eps|` を持つ正結合と符号付き結合は同じ同期曲線を持つか。

## 理論上の区別

- 論文由来: 非同期で独立に近いCauchy変数の線形和では、尺度が `|eps|` に比例する。
- 独立な整合性検討: 同期近傍では各辺へ同じ状態が入るため、結合場は符号付き行平均を含む。
- 有限サイズ: 行平均はノードごとに異なり、ランダム行列では完全同期多様体が厳密には不変でない。

符号付き平均を論文式へ代入した値は比較用proxyであり、解析的に証明した閾値として扱わない。

## 仮説と反証条件

- H1: 一様正結合のlocal初期化で `Kc=0.5` 付近にクロスオーバーが現れる。
- H2: `E|eps|=1` なら全分布の閾値が一致する。
- H3: 符号付き結合では `E eps` と行平均揺らぎに応じてH2から外れる可能性がある。
- H4: 適応尺度TM秩序はK=0で概ね `N^-1/2`、同期時に1へ近づく。

H1は持続同期成功率50%点と `Kc` の差、H2は全分布の持続同期K50を比較する。`R=0.5` はTM位相の部分凝集クロスオーバーとして別に保存し、論文の `lambda_c=0` と同一視しない。H3/H4も設定済みのpilotゲートで判定し、これは本実験の統計的検定ではない。

## 条件

- `alpha=0.25`
- `N=64,128,256`
- 3 matrix/initial-condition seeds
- `K=0.0,...,0.7` の11点
- burn-in 300、評価300
- global / local初期化
- 一様正、正の二点、bias付き符号、Rademacher結合
- float64、非有限値はclipしない

## 指標

- 適応尺度TM秩序 `R`
- `R` の下位5%点
- unwrap標本円周中央値に対する位相MAD
- ロバスト同期残差
- local初期残差の対数成長率
- `R=0.5` 補間点と最大勾配点
- 符号付き行平均の標準偏差と非有限値率

## 実行

```powershell
python e3a_finite_size_sync.py --run-directory . --config config.json
```

## 結果

792条件・36結合行列はすべて有限値で完走した。rawの `per_run_metrics.csv` は変更せず、最初の判定で混同していたTM凝集点と持続同期点を分離して再解析した。

| 仮説 | pilot判定 | 主な観測 |
|---|---|---|
| H1 理論再現 | 支持 | 一様正結合・local・\(N=256\) の持続同期K50は `0.4875`、bootstrap 95%区間は `[0.475, 0.525]`。理論値 `0.5` を含む。 |
| H2 絶対一次モーメント普遍性 | 判定不能 | \(N=256\) で持続同期K50へ達したのは一様正結合だけで、全分布の閾値比較が成立しない。 |
| H3 符号付き平均整合性 | 支持 | \(K=0.7,N=256\), localの平均TM秩序は一様正 `1.000`、正二点 `0.810`、bias付き符号 `0.578`、Rademacher `0.053`。 |
| H4 TM秩序変数 | 支持 | \(K=0\) の \(\operatorname{median}(\overline R\sqrt N)=0.6914\)、TM単位円誤差最大 `4.44e-16`。 |

補助観測:

- 一様正結合のTM凝集点は `0.2489` で、持続同期K50 `0.4875` より早い。部分凝集と持続同期は同じ閾値ではない。
- ランダム3分布の行平均標準偏差は、log-log傾き `-0.498`〜`-0.506` で概ね \(N^{-1/2}\) に従った。
- positive binaryは平均TM秩序が高くても \(N=256\) では持続同期ゲートを越えず、有限行和揺らぎの影響が残る。
- H2を直ちに不支持とはしない。有限ランダム系では完全同期多様体自体が不変でなく、現在の持続同期ゲートが厳しすぎる可能性もある。

## 成果物

- [seed別メトリクス](artifacts/per_run_metrics.csv)
- [結合統計](artifacts/coupling_statistics.csv)
- [集約曲線](artifacts/order_curves.csv)
- [閾値とbootstrap区間](artifacts/thresholds.csv)
- [機械可読summary](artifacts/summary.json)
- [TM秩序曲線](artifacts/tm_order_curves.png)
- [有限サイズ閾値](artifacts/finite_size_thresholds.png)
- [行和揺らぎと同期残差](artifacts/row_sum_vs_sync_residual.png)
- [TM位相ヒートマップ](artifacts/tm_phase_heatmaps.png)

## 再現コマンド

全シミュレーション:

```powershell
python e3a_finite_size_sync.py --run-directory . --config config.json
```

既存raw metricsから解析だけ再生成:

```powershell
python e3a_finite_size_sync.py --run-directory . --config config.json --reassess-existing
```

## 次の単一実験

一様正結合だけを用い、10 seed、\(N=64,128,256,512\)、burn-in / 評価各1000、\(K_c=0.5\) 近傍の細密格子でH1を確認する。H1確認後にのみ、ランダム結合のH2/H3確認へ進む。
