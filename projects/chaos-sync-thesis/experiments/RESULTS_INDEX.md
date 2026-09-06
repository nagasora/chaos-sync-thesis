# 実験結果・データ対応台帳

確認日: 2026-09-06。パスは原則として本ファイルのある `experiments/` からの相対パス。
現存する設定・CSV・NPZ・実行記録に基づく索引であり、数値の正本はリンク先の成果物に置く。今後の追記方法は [記録ガイド](RECORDING_GUIDE.md)。

## リポジトリ内の役割

| 場所 | 役割・使い分け |
| --- | --- |
| [プロジェクトREADME](../README.md)、[docs](../docs/PROJECT_CONTEXT.md) | 研究の目的・現在地・判断。個別結果は本台帳からたどる |
| [E0](E0/README.md)、[E1](E1/README.md) | 9月4日版教材に基づく現行系列。段階の設定・runner・レポートと `artifacts/<実行名>/` |
| [src](src) / [tests](tests) | 共通数値処理・読み出し / 実装の振る舞いの検証 |
| [runs](runs) | 旧系列の実行単位の履歴。現行系列と同じE番号でも同じ条件とは限らない |
| [templates](templates/experiment_record.md) | 記録項目の共通テンプレート。新規の空ディレクトリを量産しない |
| [教材](chaos-sync-tm-self-implementation/README.md) | 原典教材・starter・自習用コード。現行共通実装の正本は `src/` |
| [クラウド添付](../artifacts/chatgpt_exports) / [archive](../archive) | 受領資料・旧資料。添付内のregistryやrunをローカル検証済み結果へ自動的に数えない |
| [presentations](../presentations/README.md) | 発表用資料。採用した実行と図の元データを参照する |
| [references](../references/README.md) | 原典と文献ハッシュ。実験データの置き場とは分ける |

このチェックアウトのGitルートは `C:/研究`、本プロジェクトはその `projects/chaos-sync-thesis/` 配下。TeX正本はプロジェクト外の `C:/研究/latex/` にある。別checkoutではGitルート・リモートを再確認する。

## 識別方法と状態の読み方

**一つの結果は「系列＋実験ID＋成果物パス」で指定する。** E0Aの `pilot` と `boole_local_validation` は保存済み `experiment_id` が同じでも別実行である。過去IDを改変せず、パスで区別する。

- 現行系列: 9月4日版教材に基づく段階別実験。TM-Aはその追加監査。
- 旧系列: `runs/` に記録した従来計画。旧E3B/Cの結合学習と現行E3B tangent basin/E3C N8入力保持は異なる。
- 実行済み・検証通過・仮説支持は別状態。以下の検証件数は**保存済みの検証記録**であり、この棚卸しで数値実験を再実行した件数ではない。
- 設定内の `status: planned` だけで未実行と判断しない。旧E3Aは設定にplannedが残るが、`metrics.json`・`artifacts/summary.json` はcompletedで数値も存在する。原記録は保持する。

## 現行系列

| 実験 / 実行名 | 設定・レポート | データ規模 | 実行状態 / 科学的判断 | 保存先 |
| --- | --- | --- | --- | --- |
| E0A / `boole_local_validation` | [報告](E0/README.md)、[実行時設定](E0/artifacts/boole_local_validation/config.json) | 75初期条件×2精度=150軌道、3観測長で450指標行 | 実行済み、検証50項目通過。float64支持、float32精度ゲート不合格 | [成果物](E0/artifacts/boole_local_validation) |
| E0A / `pilot` | [設定](E0/artifacts/pilot/config.json)、[結果](E0/artifacts/pilot/summary.json) | α=.5・Gaussian・1 seed・float64・10,000点 | 縮小動作確認。本実験の代替にしない。float32未評価 | [成果物](E0/artifacts/pilot) |
| E0B / `kfold_tm_validation` | [報告](E0/README.md)、[実行時設定](E0/artifacts/kfold_tm_validation/config.json) | 10 seed、IID標本1,000/10,000/100,000、K=2,3,5 | 実行済み、検証104項目通過、科学的12ゲート通過 | [成果物](E0/artifacts/kfold_tm_validation) |
| E1A / `fixed_readout` | [報告](E1/README.md)、[実行時設定](E1/artifacts/fixed_readout/config.json) | train160 / validation80 / test192軌道、各4096点 | 実行済み、検証155項目通過。時間情報支持、固定TM16優位は棄却 | [成果物](E1/artifacts/fixed_readout) |
| E1 TM-A / `tm_dynamics` | [報告](E1/TM_DYNAMICS.md)、[実行時設定](E1/artifacts/tm_dynamics/config.json) | train72 / validation24 / test72軌道、6辞書・4 horizon・2 target | 実行済み、検証396項目通過。有限射影の限界を確認、普遍的TM優位は未支持 | [成果物](E1/artifacts/tm_dynamics) |
| E1B / 固定二源読み出し | [事前計画](E1/IDENTIFIABILITY.md)、[設定](E1/config_identifiability.json) | 計画: 928源軌道、464交換対 | **未実行**。計画に記載の `run_identifiability.py` は棚卸し時点で存在しない | 結果フォルダなし |

現行E2以降・TM-C/E3Cは計画段階。旧E1B/E3Aの実行済み結果で現行の完了扱いにしない。

### E0A: 初期条件と精度・観測長

実験ID: `E0A-BOOLE-LOCAL-VALIDATION`。本実行の基準フォルダは [boole_local_validation](E0/artifacts/boole_local_validation)。

| たどる対象 | ファイル・対応キー |
| --- | --- |
| 入力条件 | [conditions.csv](E0/artifacts/boole_local_validation/conditions.csv): 75行。`condition_index` → α、seed、初期分布、初期値 |
| 全観測軌道 | `orbits_float64.npz` / `orbits_float32.npz` の `orbit` は各75×100000。行は `condition_index`、列はburn-in後の時刻 |
| 条件別結果 | [metrics.csv](E0/artifacts/boole_local_validation/metrics.csv): `(condition_index, dtype, length)` が1結果。3観測長は同じ軌道の先頭窓で、独立軌道ではない |
| 密度・QQ図 | `figures/density_qq.png` ← [plot_data.csv](E0/artifacts/boole_local_validation/plot_data.csv) の `kind, alpha, dtype, x, y`。Gaussian初期化・先頭seedの観測を理論尺度で正規化。抽出処理は [run_e0.py](E0/run_e0.py) の `plot_results` |
| 理論一致・精度監査図 | `figures/theory_agreement.png`、`figures/length_precision_audit.png` ← `metrics.csv` の尺度・指数・再訪診断と `summary.json` の `convergence`（同CSVからの集約） |
| 判定・整合性 | [summary.json](E0/artifacts/boole_local_validation/summary.json) / [validation.json](E0/artifacts/boole_local_validation/validation.json) |

α=0.25,0.4,0.5,0.6,0.75、seed=20260905–20260909、burn-in=20000。学習・splitは該当なし。
pilotはburn-in=1000・1条件であり、同名指標でも本実行と混ぜない。pilotの設定に残る「150条件」等のゲート説明を、実際の評価数として引用しない。

### E0B: IID標本とK-fold写像

実験ID: `E0B-KFOLD-TM-SHIFT-CHECK`。基準フォルダは [kfold_tm_validation](E0/artifacts/kfold_tm_validation)。

- 入力は反復軌道ではなくPCG64によるIID Cauchy標本。seed=20260915–20260924。burn-in・学習splitは該当なし。
- `samples_seed_<seed>.npz`: `theta, x, q` は100000標本。`mapped` の第0軸は設定の `K_values`、`grams` の第0軸は `sample_lengths`。小さいNは同じ標本の先頭部分。
- [shift_metrics.csv](E0/artifacts/kfold_tm_validation/shift_metrics.csv)、[control_metrics.csv](E0/artifacts/kfold_tm_validation/control_metrics.csv)、[pole_metrics.csv](E0/artifacts/kfold_tm_validation/pole_metrics.csv) → `figures/shift_and_controls.png`。
- [gram_metrics.csv](E0/artifacts/kfold_tm_validation/gram_metrics.csv) と先頭seedの `grams` → `figures/gram_convergence.png`。生成処理は [run_e0b.py](E0/run_e0b.py) の `plot_results`。
- [summary.json](E0/artifacts/kfold_tm_validation/summary.json) の12判定と [validation.json](E0/artifacts/kfold_tm_validation/validation.json) の整合性検証を区別する。

### E1A: 固定16次元読み出し

実験ID: `E1A-BOOLE-FIXED-READOUT`。基準フォルダは [fixed_readout](E1/artifacts/fixed_readout)。

| 段階 | データ・行対応 |
| --- | --- |
| 初期条件・入力 | `train/validation/test_conditions.csv` と対応する `*_orbits.npz` の同じ行。NPZ内 `seeds, target` でも `(seed, alpha)` を照合できる |
| 前処理・特徴 | `*_conditions.csv` の `location, scale` と `*_features.npz`。特徴名（例 `tm`）の配列は条件表と同じ行順。testは192×16 |
| 選択・凍結 | `validation_candidates.csv` → `selection.json` → `models.npz`。係数・train平均/尺度を保存 |
| 予測 | [predictions.csv](E1/artifacts/fixed_readout/predictions.csv): `(model, seed, alpha)` が1予測。192 test軌道×10モデル=1920行 |
| 指標・区間 | [metrics.csv](E1/artifacts/fixed_readout/metrics.csv)、`seed_metrics.csv`、`alpha_metrics.csv`、[contrasts.csv](E1/artifacts/fixed_readout/contrasts.csv)、`bootstrap.npz` |
| 図 | `figures/readout_metrics.png` ← `metrics.csv`。`figures/paired_differences.png` ← `contrasts.csv` と `predictions.csv`。生成: [run_e1.py](E1/run_e1.py) の `plot_results` |

train seed510000–510031、validation520000–520015、test530000–530047。burn-in=20000、観測4096点。test α=0.38,0.46,0.54,0.62。
RMSEはTM `0.006507`、raw Fourier `0.003912`。区間はseed平均RMSE差で、集約RMSE差と同一視しない。
最終判断は [summary.json](E1/artifacts/fixed_readout/summary.json)、照合記録は [validation.json](E1/artifacts/fixed_readout/validation.json)。

### E1 TM-A: 力学適合性

実験ID: `E1-TMA-DYNAMICS-AUDIT`。基準フォルダは [tm_dynamics](E1/artifacts/tm_dynamics)。依頼原文は [request_text.txt](E1/artifacts/tm_dynamics/request_text.txt)。

| 段階 | データ・行対応 |
| --- | --- |
| 条件・入力 | `*_conditions.csv` ↔ `*_orbits.npz` の同じ行、キーは `(split, alpha, seed)`。全軌道168本、各1028点（評価1024点＋未来4点） |
| 辞書・学習 | 状態辞書は軌道と設定から再計算。`models_alpha_25/50/75.npz` に辞書別operator/decoder・尺度・target分散を保存。候補全54行は `validation_candidates.csv` |
| 予測 | `predictions_alpha_25/50/75.npz` の辞書名配列は `(horizon, test seed, 時刻, target成分)` = `(4,24,1024,3)`。seed順は設定の `test_seeds`、horizon順は `horizons` |
| target成分 | 最終軸はCayleyの実部・虚部・`2*atan(x)/pi`。2種類の評価target `cayley/cdf` へ集約。既知尺度による正規化を使用 |
| 指標 | [metrics.csv](E1/artifacts/tm_dynamics/metrics.csv): `(alpha, dictionary, horizon, target)` ごと144行。`seed_metrics.csv` はさらにseedを含み3456行 |
| 差分・構造 | `contrasts.csv` と `bootstrap.npz` はhorizon=1の比較。`structure.csv` はα×辞書の18行 |
| 共通未来予測図 | `figures/common_prediction.png` ← `metrics.csv` |
| 作用素図 | `figures/operator_structure.png` ← `theory.npz` と `models_alpha_50.npz` の係数・尺度。疎性指標CSVだけでは図を復元できない |

α=.25,.5,.75、train seed610000–610023、validation620000–620007、test630000–630023、burn-in=20000。
生成処理は [run_dynamics.py](E1/run_dynamics.py)。理論正対照は `theory.npz / theory.json / theory_metrics.csv`、判定は [summary.json](E1/artifacts/tm_dynamics/summary.json)、整合性は [validation.json](E1/artifacts/tm_dynamics/validation.json)。入力αの読み出し実験とは評価対象が異なる。

### 現行成果物の由来と再確認

各実行フォルダの `config.json` が実行時設定、`source_code.zip` が実行時ソース、`environment.json` が環境・由来、`sha256.txt` が成果物ハッシュの正本。段階直下の設定・ソースはその後に変更され得る。レポートの後日追記と、ZIP内の実行前仕様を区別する。

保存済み結果のvalidatorコマンド（作業ディレクトリはプロジェクト直下）は次のとおり。validatorは検証JSONを書き直す場合があるため、凍結記録を厳密に保持する場合は先に複製先で検証する。

```powershell
python experiments/E0/validate_results.py --output experiments/E0/artifacts/boole_local_validation
python experiments/E0/validate_results.py --output experiments/E0/artifacts/kfold_tm_validation
python experiments/E1/validate_results.py --output experiments/E1/artifacts/fixed_readout
python experiments/E1/validate_results.py --output experiments/E1/artifacts/tm_dynamics
```

再実験は各レポートのコマンドと**保存された設定**を使い、別の `--output` を指定する。今回の整理では実験・validatorの再実行を行わず、ファイル対応と保存ハッシュを読み取り検証した。

## 旧系列の対応表

以下の実行IDは `runs/` 直下のフォルダ名そのもの。各行の「成果物」リンク内に記載のファイルがある。設定・環境は各run直下の `config.json / environment.json`、実行状態は `metrics.json` とレポートを併読する。

| 実行ID・レポート | 入力・保存データ | 結果と対応成果物 | 状態・限界 |
| --- | --- | --- | --- |
| [20260806_E0_boole-invariant-measure](runs/20260806_E0_boole-invariant-measure/README.md) | α=.4,.5,.6×5 seed、各100000点。`diagnostic_orbit_samples.npz` は各αの先頭seedの20000点だけ | [成果物](runs/20260806_E0_boole-invariant-measure/artifacts): `per_seed_metrics.csv` → `summary.json`、`e0_metric_gates.png` | 実行済み、検証37項目通過。全軌道保存ではなく、記録済み条件・`e0_boole_validation.py` で再生成 |
| [20260806_E1A_tm-linear-mixture](runs/20260806_E1A_tm-linear-mixture/README.md) | 設定・計画のみ | `metrics.json` は `not_run`。数値成果物なし | 未実行。後続E1A/E1Bの結果と混ぜない |
| [20260830_E1A_temporal-alpha-readout](runs/20260830_E1A_temporal-alpha-readout/README.md) | train100 / validation40 / test48、burn-in2048・観測4096。生軌道・特徴NPZなし | [成果物](runs/20260830_E1A_temporal-alpha-readout/artifacts): `test_predictions.csv` → `per_feature_metrics.csv / summary.json` | 実行済み、検証13項目通過。TM80対Fourier16、差の区間は0を含む。生成器・設定から再生成が必要 |
| [20260830_E1A_temporal-alpha-replication](runs/20260830_E1A_temporal-alpha-replication/README.md) | train/validationはpilotと同条件、testのみ48 seed×4α=192。生軌道・特徴NPZなし | [成果物](runs/20260830_E1A_temporal-alpha-replication/artifacts): `test_predictions.csv / per_feature_metrics.csv / summary.json` | 実行済み、検証13項目通過。TM RMSE .002096、Fourier .002356、優位未確定。pilotの実装を再利用 |
| [20260901_E1A_matched-capacity-readout](runs/20260901_E1A_matched-capacity-readout/README.md) | fit140 / test192。`features.npz` 全332行、`trajectory_manifest.csv`、`fold_assignments.csv`。生軌道は再生成方針 | [成果物](runs/20260901_E1A_matched-capacity-readout/artifacts): `test_predictions.csv / model_metrics.csv / seed_metrics.csv / alpha_metrics.csv / bootstrap_differences.csv` | 実行済み、検証19項目通過。選択TM16 .001342、Fourier16 .002614。当該条件内で支持、現行の固定TM実験とは別 |
| [20260901_E1B_two-source-identifiability](runs/20260901_E1B_two-source-identifiability/README.md) | `source_trajectories.npz`: 標準化済み源352組×2本×4096点。`features.npz`: 順序付き704行（fit320 / test384）。各manifestを参照 | [成果物](runs/20260901_E1B_two-source-identifiability/artifacts): `test_predictions.csv / model_metrics.csv / collision_diagnostics.csv / pair_metrics.csv` | 実行済み、検証20項目通過。oracle32次元 .002083、対称観測の順序情報損失を確認。現行計画の全体16次元とは別 |
| [20260830_E3A_finite-size-sync-transition](runs/20260830_E3A_finite-size-sync-transition/README.md) | 792条件、4結合分布・3 seed、burn-in/評価各300。`config.json` の `simulation` から行列・初期値を生成 | [成果物](runs/20260830_E3A_finite-size-sync-transition/artifacts): `per_run_metrics.csv / coupling_statistics.csv` → `order_curves.csv / thresholds.csv / summary.json` | pilot実行済み。持続同期K50=.4875。全状態軌道は未保存、位相NPZは代表例。独立validation.jsonなし |
| [20260831_E3A_uniform-positive-confirmation](runs/20260831_E3A_uniform-positive-confirmation/README.md) | 800条件、一様正結合・10 seed・N=64,128,256,512、burn-in/評価各1000。pilotのrunnerを再利用 | [成果物](runs/20260831_E3A_uniform-positive-confirmation/artifacts): `per_run_metrics.csv / coupling_statistics.csv / order_curves.csv / thresholds.csv / summary.json` | 確認実行済み。持続同期K50=.465625、H1/H4支持。全状態軌道・独立validation.jsonなし。同期後保持は未検証 |

旧容量一致E1AのNPZは `split, seed, alpha` と `feature__<特徴名>` を同じ行で対応させる。旧E1Bの源NPZは `split, pair_seed, alpha_low, alpha_high` で識別し、同じ源対からforward/swappedの2行を作る。配列を2倍の独立源として数えない。

### 旧図・発表資料から結果をたどる

- 容量一致E1A: `matched_capacity_results.png` ← `model_metrics.csv / bootstrap_differences.csv` と `summary.json` の比較区間、`tm_candidate_selection.png` ← `candidate_cv_metrics.csv`。実装は同runの `e1a_matched_capacity_readout.py`。
- 旧E1B: `identifiability_results.png` ← `model_metrics.csv`、`prediction_scatter.png` ← `test_predictions.csv`。実装は同runの `e1b_two_source_identifiability.py`。
- 旧E3Aの両run: `tm_order_curves.png` ← `order_curves.csv`、`finite_size_thresholds.png` ← `thresholds.csv`、`tm_phase_heatmaps.png` ← `representative_tm_phases.npz`、`row_sum_vs_sync_residual.png` ← `per_run_metrics.csv / coupling_statistics.csv`。理論相図の条件は各 `config.json` とpilotの `e3a_finite_size_sync.py`。
- [有限サイズ発表資料](../presentations/README.md)は**旧E0・旧E3A pilot・旧E3A一様正結合確認**を使用。現行E0AやTM-Aの図を使った資料とは扱わない。
- 旧E3A確認の `environment.json` にあるrunner名は実体が同フォルダにない。正しい再利用パスは同runのREADMEにある `../20260830_E3A_finite-size-sync-transition/e3a_finite_size_sync.py`。

## 保存範囲と既知の不足

- 現行5実行のCSV/JSON/NPZ/PNG/ZIPは棚卸し時点でローカルに存在しGit追跡済み。リモートへの同期・別環境での再計算は今回の確認対象ではない。
- 旧E0のNPZは診断標本だけ。旧E1A pilot/追試の生軌道・特徴ストアはない。旧容量一致E1Aは特徴保存・軌道再生成、旧E1Bは**標準化済み**源保存であり、生の源軌道とは区別する。
- 旧E3Aは代表位相と集計表を保存し、全状態軌道の保存・独立validator記録はない。新規の完全保存契約を遡って達成したと扱わない。
- 旧runのハッシュ保存形式は一様ではない。容量一致E1A・旧E1Bは `environment.json` 内の `code_sha256 / artifact_sha256`、旧E3Aは実装・設定・後処理のハッシュを参照する。
- 親Gitのignore規則には `*.npz / *.png / *.pdf` がある。**今あるファイルが追跡済みでも、次回生成したファイルが自動的に追跡されるわけではない。** 共有時は追跡状況または外部保管先・ハッシュを確認する。
- 教材側の `src/` とクラウド添付のコードは参照物として残す。新しい共通コードの置き場は増やさず、`experiments/src/` を使用する。

台帳への追加時は、実行行・データ対応・図の元データ・親実行・保存不足を同時に更新する。結果の詳細な説明は既存レポートへ集約する。


## 今回の棚卸し検証（2026-09-06）

- 追加・更新した8文書のローカルMarkdownリンク130件が実在することを確認。
- 現行5実行・旧8記録を台帳が網羅し、現行106ファイルのローカル存在とGit追跡を確認。
- 現行5実行の既存 `sha256.txt` に登録された96ファイルを再ハッシュし、全件一致。台帳自身5件と `validation.json` 5件はこの96件の対象外。
- E0A本実行の75条件・150軌道・450指標行のキーと、E1A/TM-Aの全splitの条件表とNPZ内seed/alphaの行順、E1Aの特徴行数を照合。
- `git diff --check` 通過。実験コード・設定・保存データ・数値・図は変更せず、科学的結果の再実験・validatorの再実行は行っていない。

新規文書は本台帳とプロジェクト直下の `AGENTS.md` の2件。既存文書6件を更新し、ディレクトリ追加・成果物移動・削除は行っていない。
