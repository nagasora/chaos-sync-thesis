# E1: 単一tan系のTM・複素平面解析

状態: preregistered。設定・理論確認を軌道解析より先に固定する。
対象はE0保存済み代表seed（float64 12条件、float32 3条件）のみ。
新規軌道生成、同期、学習、複数seedの性能比較は行わない。

## 理論と予想（解析前固定）

主系F(x)=tan(beta*x)、beta>1、gamma=tanh(beta*gamma)の正の固定点。
q_s(x)=(x-is)/(x+is)、M_k=q_s^kを用いる。実数軌道の像は単位円に載る。
これはCauchy測度上の共通極Cayleyモードで、一般の極列のTM系とは区別する。
非負次数だけで任意のL2関数を完全展開できるとは主張しない。

### 周辺分布

X~C(0,gamma)なら、a=(gamma-s)/(gamma+s)に対して

    p(theta)=(1-a^2)/(2*pi*(1-2*a*cos(theta)+a^2)),
    E[q_s^k]=a^k,
    G_kl=E[conj(q_s^k)*q_s^l]=a^abs(l-k).

q_gammaの角度は一様であり、尺度変更による円板自己同型の境界Jacobianが
上のPoisson核を与える。Poisson核のFourier級数からモード平均とGramを得る。
s=gammaなら一様、s>gammaなら-1側、s<gammaなら+1側に集中する。
円形自体はカオスや情報保持の証拠ではない。

### 時間構造

H=q_gamma o F o q_gamma^{-1}は単位円板の正則自己写像でH(0)=0。
tanは上半平面を上半平面へ写し、境界上は零測度の極を除き実数を取る。
従って境界上のHはinnerであり、一様円周測度を保存する。
連鎖律でH'(0)=F'(i*gamma)=beta*sech(beta*gamma)^2=r。
Hのtau回合成は原点で微分r^tau、定数項0を持つので、
(H^tau(z))^kのz^k係数はr^(k*tau)。円周Fourier係数を取れば

    E[z_(t+tau)^k * conj(z_t^k)] = r^(k*tau).

複素平面にw=z_(t+tau)^k*conj(z_t^k)を描き、平均を実軸の理論値と比較する。
臨界近傍ではrが1に近く、同じ一様周辺分布でも長い時間相関が予想される。
尺度不適合の遅延相関に、この尺度適合時の式を流用しない。

## 固定条件と判定

- E0のx_1〜x_200000、全区間と4個の50000点窓。窓の開始・終端は0-based半開区間。
- s=1, gamma/2, gamma, 2*gamma。モード1〜8、Gramのみ0〜8。
- 遅延1,2,4,8,16,64,256,1024,4096,16384。窓を跨ぐ対は使わない。
- 角度64ビン、理論ビン確率はPoisson核をビン内積分する。
- 図の代表beta=1.0001,1.01,1.1,2。全条件の数値を保存する。
- 散布図は等間隔で最大4096点、時間着色は先頭2048点。
- 位相差図はk=1、tau=1,64,4096。全k・tauの相関はCSVへ保存。
- 並べ替えは軌道キーごとにSeedSequence([20260906,1,sorted_key_index,start,stop])。
  窓ごとに同じ点を並べ替えて周辺量を保つ。全区間用と4窓用は別のpermutation。
  順列後の相関の有限標本条件付き期待値は
  (abs(sum(z^k))^2-N)/(N*(N-1))。理想の定常平均0と区別して保存する。
- 理論ゲート: 全beta、4096/8192点、半径.5の複素円積分でtau=1,2,4,8,16、k=1〜8。
  長遅延は上記合成・係数の証明から計算する。数値係数誤差<=1e-10。
  単位円、逆変換、Poisson規格化、平均、Gramも独立に照合する。
- 理論ゲート不通過、入力ハッシュ不一致、非有限状態、単位円誤差>1e-12では解析を停止。
  クリップ、状態の除去、条件の差替えはしない。
- 軌道結果は理論との差・窓間変動として報告し、窓を独立seedとした95%区間は作らない。

## 実行と保存

このフォルダだけで解析・再描画できる。prepare時のみE0の入力をコピーする。
NumPy/SciPy/matplotlibを使用。古いrunのPython実装はimportしない。

    python tm_complex_plane.py --prepare --source ../20260906_E0_tangent-cauchy
    python tm_complex_plane.py --theory
    python tm_complex_plane.py --analyze
    python tm_complex_plane.py --plot
    python tm_complex_plane.py --validate

既存結果は上書き拒否。再試行は--outputで別フォルダを指定する。
artifacts/source_orbits.npzが入力正本。入力由来とSHAはprovenance.json。
moments.csv, density.csv, gram.csv, correlations.csv, diagnostics.csvが条件・窓別の結果。
plot_points.npzは表示に実際に用いる複素点・時刻・位相差点のストア。
設定・理論結果・コード・入力ハッシュを解析開始時に記録し、完成時に全成果物をハッシュ化する。

## 図とキャプション

全図は代表seedのみ。PNG/PDF/SVGをartifactsに保存する。
背景は印刷用白、経験値は青、理論値は濃灰、比較は琥珀色。
時間色は知覚的に順序のあるviridis。複素平面は共通軸・等アスペクト。

1. fig1_geometry_density: 固定尺度と理論尺度のCayley点群・角度密度。
   点は最大4096点、ヒストグラムは全200000点、線は理論ビン平均密度。
2. fig2_time_colored: 尺度適合Cayley像の先頭2048点を時刻で着色。
   離散時刻の点群であり連続時間の運動ではない。
3. fig3_modes: z,z^2,z^4の点群と全区間平均。理論平均は原点。
4. fig4_phase_difference: k=1の遅延位相差点と全有効対の平均。星がr^tau。
5. fig5_correlations: k=1,2,4,8の相関実部、虚部、絶対誤差と事前理論。
   4窓の変動は別CSVに保存。信頼区間ではない。
6. fig6_gram: 尺度gamma/2,gamma,2gammaで理論・経験Gram実部と複素誤差の絶対値。
   主ファイルはbeta=1.0001。他の代表条件はfig6_gram_beta_*として同じサイズで出力する。
   誤差パネルは飽和で差を隠さないようラベル付きの個別色範囲とする。
7. supplement_controls: 並べ替え対照とfloat32比較。入力精度の影響を混同しない。

各図の元データは上のCSVおよびplot_points.npz。PDF/SVGは論文・発表用、PNGは確認用。
理論予想と有限軌道の差は隠さず考察する。TMの優位、同期、復号性能は結論しない。


## 実行結果と解釈（2026-09-06）

理論確認を先に完了し、その後に保存済み15軌道を解析した。新しい軌道は生成していない。
理論チェック1308項目は全て通過（最大誤差2.243e-14、基準1e-10）。
単位円からの最大逸脱は2.221e-16。実装テスト8件、保存結果の再検証109項目が通過した。

| 指標 | 観測値 | 解釈 |
|---|---:|---|
| beta=1.0001、尺度適合、k=1の平均 | 0.00300644 - 0.29182229 i | 理論平均0に対し角度の非対称性が残る |
| beta=2、同じ平均の絶対値 | 0.00298864 | 臨界近傍より小さい有限軌道の偏り |
| float64全期間の最大相関誤差 | 0.18094841 | beta=1.0001、k=5、tau=1024 |
| 並べ替え対照の最大相関誤差 | 0.00618065 | 対照の有限標本期待値との比較 |

最大相関誤差の条件では経験値0.38019612+0.17972181i、理論値0.35916280。
同条件の4窓の誤差は順に0.14910、0.03771、0.36903、0.21684であり、窓依存も大きい。
これを理論との完全一致や収束済みとは判定しない。長い相関と有限観測長による影響は
候補説明だが、この代表seedだけでは原因の確定や統計的棄却はできない。

複素平面上では全条件が単位円に乗る一方、角度密度・複素モード平均・遅延位相差の
平均は異なる。円形という幾何学的制約と時間相関を区別する必要がある。
並べ替えは周辺モード平均を保存し、時間順序の相関を弱めるという対照として機能した。
float32は補足図と条件別CSVに分離し、精度差を理論的な相の差とは解釈しない。
次段階で原因を検証する場合は、独立seed追加や観測長延長を別の事前登録実験とする。

### 図へのリンク

- [点群・角度密度](artifacts/fig1_geometry_density.pdf)
- [時間色の複素点群](artifacts/fig2_time_colored.pdf)
- [複素モード](artifacts/fig3_modes.pdf)
- [遅延位相差](artifacts/fig4_phase_difference.pdf)
- [相関と理論誤差](artifacts/fig5_correlations.pdf)
- [Gram比較](artifacts/fig6_gram.pdf)
- [並べ替え・精度比較](artifacts/supplement_controls.pdf)

同名のPNGとSVGも保存済み。主図6種類、追加Gram3枚、補足1枚の計10枚・30ファイル。
