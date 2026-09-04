# 統一実験ID体系

正規IDは `E0→E1→E2→E3→E4→E5` の研究依存順に固定する。旧E/T名は `legacy_ids` として残し、既存runフォルダは改名・上書きしない。

| 正規ID | 段階 | family | 状態 | 旧ID |
|---|---|---|---|---|
| E0A-BOOLE-LOCAL-VALIDATION | E0 基礎力学 | generalized_boole | confirmed_existing | E0<br>20260806_E0_boole-invariant-measure |
| E0B-TANGENT-FOUNDATION | E0 基礎力学 | tangent | legacy_theory_grouped | T0-*<br>T1<br>T2<br>T3（理論ノート内の局所写像・Cauchy・TM節） |
| E1A-BOOLE-SINGLE-SOURCE-READOUT | E1 読み出し・識別可能性 | generalized_boole | confirmed_existing | E1A<br>20260830_E1A_temporal-alpha-readout<br>20260830_E1A_temporal-alpha-replication<br>20260901_E1A_matched-capacity-readout |
| E1B-BOOLE-TWO-SOURCE-IDENTIFIABILITY | E1 読み出し・識別可能性 | generalized_boole | confirmed_existing | E1B<br>20260901_E1B_two-source-identifiability |
| E2A-BOOLE-OBSERVATION-ROBUSTNESS | E2 頑健性 | generalized_boole | new_completed_by_unified_suite | 旧E2 noise/missing/length計画 |
| E3A-BOOLE-FINITE-SIZE-SYNC | E3 同期力学 | generalized_boole_random_coupling | confirmed_existing | E3A<br>20260830_E3A_finite-size-sync-transition<br>20260831_E3A_uniform-positive-confirmation |
| E3B-TANGENT-TWO-NODE-BASIN | E3 同期力学 | tangent_two_node | new_completed_by_unified_suite | T4<br>旧E2 2自由度タンジェント同期相図 |
| E3C-BOOLE-N8-INPUT-RETENTION | E3 同期力学 | boole_output_mixing | new_completed_by_unified_suite | 旧E3 小規模ネットワーク入力保持計画 |
| E4A-BOOLE-SYNTHETIC-SIGNAL-RECONSTRUCTION | E4 信号圧縮・復元 | boole_output_mixing_driven | new_completed_by_unified_suite | 旧E4 合成信号計画 |
| E5A-BOOLE-SMALL-IMAGE-DIGITS | E5 画像pilot | boole_output_mixing_driven | pilot_completed_by_unified_suite | 旧E5 MNIST/AE計画 |
| E5B-BOOLE-MNIST-CONFIRMATION | E5 画像確認 | boole_output_mixing_driven | not_run_requires_external_dataset_compute | MNIST本確認 |

## 命名規則

- run folder: `YYYYMMDD_<canonical short id>_<slug>`
- config内には `canonical_id`, `legacy_aliases`, `model_family`, `depends_on` を必須保存する。
- 既存runは不変。registryだけで別名解決し、履歴のSHA-256を壊さない。
- `completed` は数値artifact・予測・validator・hashを含む場合だけ用いる。
- `pilot_completed` と `confirmed_existing` を区別し、pilotを本確認として扱わない。