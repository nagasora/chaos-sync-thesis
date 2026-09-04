# E3A 一様正結合同期臨界値の確認

## 目的

`20260830_E3A_finite-size-sync-transition` の pilot で得た結果を、一様正結合だけに限定して確認する。比較対象は TM 秩序変数の部分的な立ち上がりではなく、local 初期化における持続同期成功率の K50 とする。

理論値は一般化 Boole 写像の条件付き Lyapunov 指数

\[
\lambda_c
=2\log\left(
\sqrt{\alpha}
+\sqrt{1-\alpha-K\,\mathbb{E}[|\varepsilon_{ij}|]}
\right)
\]

の零点から得る

\[
K_c
=\frac{2\sqrt{\alpha}\left(1-\sqrt{\alpha}\right)}
{\mathbb{E}[|\varepsilon_{ij}|]}.
\]

一様正結合では \(\mathbb{E}[|\varepsilon_{ij}|]=1\)、今回の \(\alpha=1/4\) では \(K_c=0.5\) である。実験前に許容幅を \(\pm 0.05\) と固定する。

## pilot から変更する変数

- 分布: 4種類から `uniform_positive` のみに限定
- seed: 3から10へ増加
- 最大系サイズ: 256から512へ増加
- burn-in / 評価時間: 各300から各1000へ増加
- K走査: 閾値近傍を0.025刻みで細分化

写像、初期化、TM 秩序変数、持続同期判定（平均0.8以上かつ5%点0.5以上）は変更しない。

## 判定

- 支持: 最大Nの local 条件で持続同期 K50 が有限かつ \(|K_{50}-0.5|\leq0.05\)
- 不支持: K50 が有限だが事前許容幅の外側
- 判定不能: 走査範囲内で K50 が得られない

## 実行

このrunでは既存の検証済み実装を再利用する。

```powershell
python ..\20260830_E3A_finite-size-sync-transition\e3a_finite_size_sync.py `
  --run-directory . `
  --config config.json `
  --assessment-scope confirmation
```

## 結果

2026-08-31に完走した。40個の結合行列から800条件を評価し、非有限値で終了した条件は0件だった。

| N | local持続同期K50 | seed bootstrap 95%区間 |
|---:|---:|---:|
| 64 | 0.465000 | [0.450000, 0.479167] |
| 128 | 0.465625 | [0.462500, 0.475000] |
| 256 | 0.470833 | [0.463889, 0.484375] |
| 512 | 0.465625 | [0.462500, 0.475000] |

最大NではK=0.45の持続同期成功率が0/10、K=0.475が8/10、K=0.5以上が10/10だった。線形補間したK50は `0.465625` で、理論値 `0.5` との差は `-0.034375`。事前許容幅 `±0.05` の内側なので、H1は **`confirmation_supported`** と判定する。

TM秩序変数のR50は `0.256735`（95%区間 `[0.248517, 0.268155]`）だった。これは部分的な位相凝集点であり、横方向Lyapunov指数が0となる理論閾値との比較には使わない。K=0におけるTM秩序の `sqrt(N)` 規格化中央値は `0.670299`、TM像の最大単位円誤差は `4.44e-16` で、H4も確認実験の基準内だった。

## 解釈の境界

- このrunは一様正結合だけを対象とするため、分布普遍性H2と符号付き結合H3は判定しない。
- K50は有限K格子と「平均R>=0.8かつ5%点R>=0.5」という持続同期規則に依存する。
- 95%区間は10個の固定seed群内のbootstrapであり、別seed群・別K格子による独立追試を置き換えない。
- これは有限サイズでの数値的整合であり、熱力学極限の証明ではない。

## 監査可能性

- 実走config SHA-256: `83b924d59ddaae1baab42041357c5acb7e7fd0050d699da7ba2fe4c561040724`
- 生の `per_run_metrics.csv` SHA-256: `bb9d711478e1006798fc69cbf5cebb58248e40bd6439abf75d565fbfac563993`
- 数値実走コードSHAと後処理コードSHAは `environment.json` へ分離して保存した。
- `config.json` は実走時の内容を保つため `status=planned` のまま凍結し、完走状態は `summary.json` と `metrics.json` を正とする。

主要成果物は `artifacts/summary.json`、`artifacts/thresholds.csv`、`artifacts/order_curves.csv`、`artifacts/finite_size_thresholds.png`、`artifacts/tm_order_curves.png` である。
