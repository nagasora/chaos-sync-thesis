# 有限サイズカオス同期相転移：発表用カンペ

対象資料: ../finite-size-chaos-sync-presentation.pptx  
作成日: 2026-08-31  
標準時間: 約11分30秒

## 最初に覚える結論

> α=1/4の一様正結合系では、N=512の持続同期K50は0.465625、bootstrap 95%区間は[0.4625, 0.475]で、事前許容幅0.5±0.05に入った。一方、TM位相秩序R=0.5の点は約0.257で、部分凝集と持続同期は別の転移として観測された。

K_c（理論値）、持続同期K50（数値判定）、TM R50（部分凝集）を混同しない。符号付き結合での普遍性、同期後の情報保持、情報圧縮は未検証である。

## スライド別カンペ

### スライド1｜有限サイズカオス同期相転移

[Sources]
- C:/研究/projects/chaos-sync-thesis/docs/PROJECT_CONTEXT.md, sections 1 and 7.
- C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/README.md.
[/Sources]
目安時間: 0:30
本発表では、長期目標の情報圧縮そのものではなく、その前提となる同期相転移を有限サイズで検証します。結論を先に言うと、一様正結合では理論値0.5の近傍で持続同期が立ち上がりました。一方、TM位相の部分凝集と完全な持続同期は別の現象でした。

### スライド2｜研究質問：無限Nの臨界値は有限Nでも見えるか

[Sources]
- S-003: C:/研究/projects/chaos-sync-thesis/references/papers/infinite-dimensional-chaotic-synchronization.pdf, p.7 Eq. (43), p.8 Eq. (45).
- E3A-confirmation: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/artifacts/summary.json.
[/Sources]
目安時間: 0:55
研究質問は単純です。無限次元理論が予測するKc=0.5が、有限Nの数値計算でも持続同期の立ち上がりとして見えるかを調べました。N=512ではK50が0.4656で、事前に固定した0.5±0.05に入りました。ただし、TM秩序変数のR=0.5点は約0.257であり、同じ閾値ではありません。

### スライド3｜モデル：一般化Boole写像の結合系

[Sources]
- S-003: C:/研究/projects/chaos-sync-thesis/references/papers/infinite-dimensional-chaotic-synchronization.pdf, p.2 Eq. (1)-(2) and assumptions.
- E3A-confirmation: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/config.json.
[/Sources]
目安時間: 1:05
各ノードは一般化Boole写像で時間発展し、他ノードから線形結合を受けます。理論原稿は対称なi.i.d.重みと無限次元極限を扱います。今回の確認実験は、その一様正結合極限だけを有限Nで調べました。αは1/4です。E0で使ったα=1/2とは別条件なので混同しません。

### スライド4｜理論：α=1/4の臨界値はK_c=0.5

[Sources]
- S-003: C:/研究/projects/chaos-sync-thesis/references/papers/infinite-dimensional-chaotic-synchronization.pdf, p.3 Eq. (9)-(10), p.7 Eq. (39)-(43), p.8 Eq. (45).
[/Sources]
目安時間: 1:10
理論原稿では、Cauchy尺度を用いて条件付きLyapunov指数が閉形式で与えられます。その零点がKcです。α=1/4かつ一様重みでE|ε|=1とするとKc=0.5になります。正のCauchy尺度が存在する上限はK<0.75です。この原稿は2026年7月5日付のローカルpreprintであり、有限Nでの再現を本研究の数値的な検証対象にしています。

### スライド5｜判定指標：持続同期とTM位相秩序を分ける

[Sources]
- E3A implementation: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260830_E3A_finite-size-sync-transition/e3a_finite_size_sync.py, functions tm_transform, tm_phase_metrics, and bootstrap_thresholds.
- E3A tests: C:/研究/projects/chaos-sync-thesis/experiments/tests/test_e3a_finite_size_sync.py.
[/Sources]
目安時間: 1:10
ここが本発表の重要な区別です。TM変換は実数状態を単位円上へ写し、その位相がどれだけ揃うかをRで測ります。しかしRが0.5になっただけでは、評価区間全体で安定に同期したとは言えません。理論Kcとの比較には、local初期化からの持続同期成功率を使います。

### スライド6｜確認実験：条件を事前固定

[Sources]
- E3A-confirmation: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/config.json.
- E3A-confirmation protocol: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/README.md.
[/Sources]
目安時間: 0:55
pilotを見た後に自由に条件を変えることを避けるため、確認実験は一様正結合のH1だけへ限定しました。N、seed、K格子、計算長、bootstrap回数、許容幅をconfigへ固定し、800条件を実行しました。raw CSVのハッシュも保存しています。

### スライド7｜E0：単体写像の実装検証

[Sources]
- E0: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260806_E0_boole-invariant-measure/metrics.json.
- E0 validation: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260806_E0_boole-invariant-measure/artifacts/validation.json.
[/Sources]
目安時間: 0:55
結合系へ進む前に、単体写像の実装を検証しました。3つのα、各5 seedで、Cauchy尺度、Lyapunov指数、KS距離、簡略TMモードの事前ゲートと独立チェック37件を通過しています。ただし、これは実装回帰を検出する工学的ゲートであり、統計的有意性や結合系の正しさを証明するものではありません。

### スライド8｜N=512：TM秩序と持続同期成功率

[Sources]
- E3A-confirmation figure: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/artifacts/tm_order_curves.png.
- E3A-confirmation raw metrics: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/artifacts/per_run_metrics.csv.
[/Sources]
目安時間: 1:00
N=512のTM秩序曲線です。global初期化でもlocal初期化でもKとともにRが上昇します。持続同期の判定では、K=0.45で成功0、0.475で8、0.5以上で10 seedすべて成功しました。図のR曲線だけから臨界値を決めず、下の成功率からK50を推定しています。

### スライド9｜有限サイズ閾値

[Sources]
- E3A-confirmation thresholds: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/artifacts/thresholds.csv.
- E3A-confirmation summary: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/artifacts/summary.json.
[/Sources]
目安時間: 1:05
横軸は1/sqrt(N)です。持続同期K50はNを変えてもおよそ0.47にあり、N=512では0.465625、95%区間は0.4625から0.475です。理論0.5との差は事前許容0.05の範囲内です。一方、TMのR=0.5点は約0.257で、有限サイズを変えても別の低いクロスオーバーとして残りました。

### スライド10｜pilot：結合分布で挙動が異なる

[Sources]
- E3A-pilot figure: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260830_E3A_finite-size-sync-transition/artifacts/tm_order_curves.png.
- E3A-pilot summary: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260830_E3A_finite-size-sync-transition/artifacts/summary.json.
- E3A-pilot row-sum diagnostic: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260830_E3A_finite-size-sync-transition/artifacts/row_sum_vs_sync_residual.png.
[/Sources]
目安時間: 1:05
pilotでは4つの重み分布を比較しました。一様正結合は持続同期へ達しましたが、他の分布は同じK範囲で達しませんでした。TM秩序の上がり方にも差があります。ただし各3 seedで、分布ごとに同期が生成される条件も異なります。この結果だけでE|ε|の普遍性を反証・支持せず、H2は判定不能とします。

### スライド11｜結論・限界・次の一手

[Sources]
- E3A-confirmation: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260831_E3A_uniform-positive-confirmation/README.md.
- E3A-pilot: C:/研究/projects/chaos-sync-thesis/experiments/runs/20260830_E3A_finite-size-sync-transition/README.md.
- Research boundary: C:/研究/projects/chaos-sync-thesis/docs/PROJECT_CONTEXT.md, sections 5, 7, and 10.
[/Sources]
目安時間: 1:05
確認できたのは、一様正結合という限定条件での有限サイズ同期転移です。TM変換の数値的な単位模性も確認しました。反対に、結合分布をまたぐ普遍性、同期後の情報保持、計算量や省エネルギー性は未検証です。次は符号付き分布だけを10 seedへ増やし、完全同期多様体の不変性に関わる行和揺らぎを主解析にします。

### スライド12｜参考文献・再現入口

[Sources]
- S-001: C:/研究/projects/chaos-sync-thesis/references/papers/synchronization-of-chaotic-systems.pdf.
- S-002: C:/研究/projects/chaos-sync-thesis/references/papers/chaotic-synchronization-mutually-coupled-systems.pdf.
- S-003: C:/研究/projects/chaos-sync-thesis/references/papers/infinite-dimensional-chaotic-synchronization.pdf.
- E0, E3A-pilot, and E3A-confirmation run directories listed on the slide.
[/Sources]
目安時間: 0:30
理論の中核は3本のローカル原典で確認しました。2026年原稿は査読済み論文とは表現せず、日付付きpreprintとして扱います。数値結果はrunディレクトリ、config、raw CSV、summary、ハッシュから追跡できます。

## 8分へ短縮する場合

- スライド3・4・7・10は結論だけを一文で述べる。
- スライド8・9・11は省略しない。
- 数式導出より、理論値・数値判定・未検証範囲の区別を優先する。

## 想定問答の要点

- **なぜ理論値0.5より低いか**: 有限N・有限時間・有限seed・離散K格子による数値閾値であり、厳密一致ではなく事前許容幅で判定した。
- **95%区間に0.5が入らないのでは**: これはseedに対するK50推定区間で、理論値との同値性検定ではない。支持条件は事前固定した差0.05以内である。
- **R50≈0.257の意味**: TM位相の部分凝集であり、評価区間全体の持続同期ではない。
- **pilotで他分布が同期しない理由**: 各3 seedで行和揺らぎも異なるため判定不能。次の確認実験で分離する。
- **圧縮できたか**: まだ同期条件だけである。復号誤差・符号長・比較ベースラインは別実験が必要。

## 発表直前チェック

- [ ] K_c、持続同期K50、TM R50を別の言葉で説明できる。
- [ ] α=1/4、一様正結合、10 seedという適用範囲を言える。
- [ ] 0.465625 [0.4625, 0.475]を言える。
- [ ] K=0.45: 0/10、0.475: 8/10、0.5以上: 10/10を言える。
- [ ] 2026年原稿を査読済み論文と呼ばない。
- [ ] 同期だけでは情報圧縮ではない、と答えられる。
