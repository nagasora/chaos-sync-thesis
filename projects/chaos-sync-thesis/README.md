# カオス同期による動的情報圧縮の研究

2026-09-05: [E1A固定読み出し追試](experiments/E1/README.md)を432軌道で完了。時間順序の有用性を支持したが、固定TM16（RMSE 0.006507）は実数Fourier16（0.003912）より誤差が大きく、TM優位仮説は棄却。独立検証155/155。次は教材E1Bの二源識別可能性。

続くE0Bも10 seedで完了し、基底直交性とK-foldのshift正対照を確認した（独立検証104/104）。一般Booleへ恒等式を流用しない負対照も確認済み。E0Aのfloat32不合格を維持し、この次計画だったE1Aは上記のとおり実施済み。

## 研究の核

> 神経系に見られる動的同期に着想を得て、カオス同期を情報圧縮へ応用できるか検討する。

中心的な問いは、多数のカオス自由度が同期によって縮約されるとき、入力に必要な情報を残し、その情報を安定に読み出して復号できるか、という点です。

同期で自由度が減ることだけでは、情報圧縮が成立したとはいえません。すべての入力が同一の同期軌道・同一の不変測度へ収束すれば、入力差は失われます。本研究では、同期によって消える冗長成分と、同期後も残る入力依存成分を分離して測定します。

## 現在の研究仮説

同期多様体に垂直な方向では収縮し、同期多様体に沿う方向ではカオス性を残す領域、

\[
\lambda_\perp < 0,\qquad \lambda_\parallel > 0
\]

において、冗長性の縮約と入力依存情報の保持が両立する可能性があります。特に同期臨界点直後では、横断方向の差を抑えながら有限時間の入力依存軌道を比較的長く保持できる、という仮説を検証します。

## 読み出し候補

- Takenaka–Malmquist（TM）基底: Cauchy 不変測度を持つ可解カオス軌道の状態・時間モード
- グラフフーリエ変換: ノード間の同期、クラスタ、空間モード
- 時間遅延・Koopman 解析: 遷移や時間発展構造
- 横断 Lyapunov 指数: 同期多様体の安定性

潜在表現の候補は、グラフ周波数・TM 次数・時間遅延を統合した

\[
z(x)=\{\widetilde c_{\ell k,\tau}(x)\}
\]

です。

## 現在地

2026-09-05: ユーザー指定の実験テキストに沿い、段階別 experiments/E0… と共通 experiments/src を正本として整備した。[E0Aレポート](experiments/E0/README.md)には150条件の全軌道・数値・図・独立検証を保存した。float64の整合性を支持する一方、float32の全軌道周期化とα=0.75の尺度誤差基準超過を確認。このE0A時点の次計画だったE0Bは、上記のとおり実施済みである。以下は過去の実験履歴。


E0では非結合一般化Boole写像のCauchy不変測度、Lyapunov指数、簡略TMモードの数値整合性を確認済みです。E1Aでは周辺尺度を軌道ごとに除いた単一源から未使用 `alpha` を読むpilot・独立seed追試に続き、TMとFourierを16次元へ揃えた容量一致追試を実行しました。新規48 test seedでTM短遅延はRMSE `0.001342`、Fourierは `0.002614`、差の95%区間は `[0.000985, 0.001560]` となり、容量差だけではTM改善を説明できない結果でした。ただし候補選択自由度、単一源、補間alphaという限定は維持します。E1Bでは2源へ進み、既知full-rank 2観測のoracle TMは順序付きRMSE `0.002083`、対称1観測は完全衝突差 `0.0`、理論下限 `0.073030` に対してTM `0.073710` となりました。可逆観測の正対照と、源入替えによる厳密な識別不能対照を同一条件で確認しています。

E3Aでは有限サイズ結合系792条件のpilotを完走しました。一様正結合の持続同期K50は `0.4875`（95% bootstrap区間 `[0.475, 0.525]`）で理論値 `0.5` と整合しました。一方、同じ `E|eps|=1` でもRademacher結合は凝集せず、正二点・bias付き符号も最大Nで持続同期ゲートへ達しませんでした。したがって分布普遍性H2は判定不能、符号付き平均整合性H3はpilot支持です。

一様正結合の本確認は10 seed・最大 `N=512`・burn-in/評価各1000・臨界近傍細密格子の800条件で完走しました。最大N・localの持続同期K50は `0.465625`（95%区間 `[0.4625, 0.475]`）で、事前基準 `K_c=0.5±0.05` を満たしました。H1とTM秩序H4は確認実験として支持します。短期発表ではこの結果を主結論として凍結し、H2/H3の10-seed検証は未完の限界として明示します。読み出し側の次段階は、E1Bのfull-rank行列と分割を固定し、観測ノイズ強度だけを変えるE2A頑健性実験です。

## まず読むファイル

1. [PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)
2. [MIGRATION_MANIFEST.md](docs/MIGRATION_MANIFEST.md)
3. [CLOUD_CONVERSATION_INDEX.md](docs/CLOUD_CONVERSATION_INDEX.md)
4. [REFERENCE_INVENTORY.md](docs/REFERENCE_INVENTORY.md)
5. [experiments/README.md](experiments/README.md)
6. [発表フィードバック整理（2026-08-06）](docs/PRESENTATION_FEEDBACK_2026-08-06.md)
7. [実験記録ガイド](experiments/RECORDING_GUIDE.md)
8. [モデル学習と結合強度の設計](docs/MODEL_TRAINING_DESIGN.md)

## 作業場所

- 研究文書・実験: この `projects/chaos-sync-thesis` 配下
- TeXソースと生成PDF: [`../../latex`](../../latex/README.md)
- PowerPointと発表ノート: [`presentations`](presentations/README.md)
- 中核論文: [`references/papers`](references/README.md)

## 参考文献

主要 PDF は [references/papers](references/papers) に保存されています。特に、無限次元ランダム結合 Boole 系、Artificial Kuramoto Oscillatory Neurons、TM–Koopman 関連論文の役割を区別して利用します。旧名称とSHA-256は[参考文献台帳](references/README.md)に記録しています。
