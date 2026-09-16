# E2：相互加算タンジェント結合の同期可能性

## 事前登録（軌道生成前）

対象は x'=tan(beta x)+epsilon y、y'=tan(beta y)+epsilon x。
beta>1、0<=epsilon<1。拡散結合や写像出力の混合とは区別する。
今回の目的は、同期多様体上の不変測度と横方向線形安定性を確認すること。
吸引域の調査、一般の二変数分布のCauchy閉包、復号・学習は行わない。

## 理論

### 同期多様体上の閉包を改めて導出する

x=y=s は不変で、s'=f(s)=tan(beta s)+epsilon s。
上半平面では Im tan(beta z)>0 であり、epsilon>=0ならfも上半平面を自身へ写す。
u>=0に対して exp(i u f(z)) は有界解析関数。その実部・虚部へのPoisson表示から

    E[exp(i u f(S))] = exp(i u f(i gamma))
                      = exp[-u{tanh(beta gamma)+epsilon gamma}],
    S ~ C(0,gamma).

負のuは共役で従う。実軸上の極は測度0、境界値はほとんど至る所で存在する。
従ってこの1変数写像に限って、厳密尺度更新 G(gamma)=tanh(beta gamma)+epsilon gamma。
これは依存するCauchy変数の和を独立とみなした導出ではない。
Poisson核の式の参考: [Complex Analysis, Theorem 10.3.1](https://complexanalysis.org/web/sec_poisson-integral-formula.html)。
本写像への適用と以下の安定性計算は、この実験で行った導出である。
有界解析関数の実部・虚部を用いれば、境界の不連続点を除いて有界調和関数の表示を適用できる。

正の固定尺度gammaは (1-epsilon)gamma=tanh(beta gamma) の唯一の正解。
Gは正領域で狭義凹、G'(0)=beta+epsilon>1、epsilon<1でG(gamma)/gammaの極限はepsilon。
従って固定点が一つあり、G'(gamma)<1。

### 縦・横方向指数

同期点でのJacobianは [[D,epsilon],[epsilon,D]]、D=beta sec^2(beta s)。
固有方向(1,1),(1,-1)の倍率はD+epsilon、D-epsilon。
U=tan(beta S) ~ C(0,a)、a=tanh(beta gamma)=(1-epsilon)gamma とおけば

    lambda_parallel = log(beta)+2 log(a+sqrt(1+epsilon/beta)),
    lambda_perp     = log(beta)+2 log(a+sqrt(1-epsilon/beta)).

ここではepsilon<1<betaなので横倍率は常に正。
積分 E log(U^2+c^2)=2log(a+c) は、U=a tan(theta)、thetaが一様という置換で
独立数値積分も行う。

尺度適合Cayley像Hは単位円板のinner写像、H(0)=0、回転写像ではない。
Schwarzの補題から反復H^nは円板内部のコンパクト集合で0へ収束する。
H^nの正次数べきのTaylor係数も0へ収束し、Fourier単項式間の相関は消える。
三角多項式の稠密性と測度保存からmixing、従ってergodic。
上の対数倍率は積分可能なので、Birkhoffから指数はこの不変測度に関してほとんど確実に上式。
例外的な固定点・周期点すべての指数を同じと主張しない。

### 同期不安定性の事前予想

B=beta/(1-epsilon)、atanh(a)/a=B とし、
atanh(a)/a < 1/(1-a^2) より a^2>1-(1-epsilon)/beta。
c=sqrt(1-epsilon/beta)>0 とすると

    beta(a+c)^2 > beta(a^2+c^2) > 2 beta-1 > 1.
    lambda_perp > log(2 beta-1) > 0.

従って、指定領域の不変Cauchy測度上の典型的完全同期は横方向に不安定と予想する。
epsilon>=1ではこの正の有限固定尺度は存在しないため、この式で外挿しない。

## 固定実験条件

- beta=1.0001,1.01,1.1,2、epsilon=0,0.01,0.1,0.5,0.9。
- 32独立seed、各200000ステップ。SeedSequence([20260907,2,beta_index,epsilon_index,seed])。
- S0~C(0,gamma)、burn-inなし。float64、Numba標準tan、fastmathなし。
- 主評価は対数横倍率と縦倍率の時間平均。独立seedの平均とStudent-t近似95%区間。
- 尺度は半IQR、Cauchy CDFとのKS距離、Cayley平均を補助記録。
- 同期多様体上の軌道と線形変分の検証であり、非同期初期値の吸引域実験ではない。
- 単位円誤差、非有限値、最大状態を監査。クリップ、再抽選、seed交換はしない。
- 理論ゲート: 固定点残差、円板積分、対数積分、Jacobian有限差分。
  積分/残差基準1e-9、有限差分1e-7。失敗時は軌道生成しない。
- beta=2,epsilon=0.9ではaがfloat64で1へ丸められ得る。aの飽和を記録し、
  atanh(a)の数値評価で無限大を作らずgammaの区間付き求根を使用する。

## 図・保存と再現

1. 結合強度に対する縦横指数と不変尺度：理論曲線、seed平均と区間。
2. epsilon=0,0.5のCayley点群と角度密度：各betaのseed=0、最大4096表示点、密度は全点。
3. 代表seedの横方向対数感度累積：2の累乗時刻と最終時刻、理論lambda_perp*tと比較。

PNG/PDF/SVG、元CSV、代表軌道NPZ、設定・理論・環境・ハッシュをこのrunに保存する。
GitHubに送るのはコードとテストだけで、生成データ・図はローカルに留める。
入口はadditive_sync.pyの1件。既存E1テストを拡張し、旧run実装へのimportは追加しない。
白背景、青=横方向、琥珀=縦方向、破線=理論、点=数値。複素平面は等アスペクト共通軸。

実行: `python additive_sync.py --theory`、`--run`、`--plot`、`--validate`。
生成結果は上書きせず、追試は別フォルダで行う。

## 結果

事前登録時は未実行。以下に結果を追記する。


## 実行結果（2026-09-07）

20条件 x 32seed、計640軌道・1.28億更新を29.01秒で実行した。
事前登録の本文はpreregistration.mdおよびコード内定数に保存し、結果による条件変更はない。
理論260項目は通過（最大絶対差4.4021e-11、Jacobian差分の基準は1e-7）。
結合系の固定点・変分・理論ゲートのテストが通過。
代表20軌道の完全再生と別式による指数再計算が通過（最大差2.22045e-16）。

| beta | epsilon | 横指数：理論 | 横指数：32seed平均 | 近似95%半幅 |
|---:|---:|---:|---:|---:|
| 1.0001 | 0 | 0.0344414 | 0.0332284 | 0.0035357 |
| 1.0001 | 0.5 | 1.0193481 | 1.0189698 | 0.0019038 |
| 1.0001 | 0.9 | 0.5498559 | 0.5503500 | 0.0020309 |
| 2 | 0 | 2.0364876 | 2.0381001 | 0.0013739 |
| 2 | 0.5 | 1.9400457 | 1.9400383 | 0.0015250 |
| 2 | 0.9 | 1.8027784 | 1.8030623 | 0.0014523 |

640軌道すべてで有限時間横指数が正、最小0.0199080。
20条件すべてでseed平均の近似95%区間も正だった。
横指数の理論値を区間内に含んだのは19/20条件で、beta=2,epsilon=0が区間外。
この区間は近似・多重比較未補正なので、19/20を一括合否として扱わない。
ゼロ出現0、非有限値0、単位円からの最大逸脱2.22045e-16。
高精度での軌道追跡やfloat32比較は今回行っていない。

### 解釈

Hypothesis: 正の相互加算結合では、指定領域の典型的な完全同期は横方向に不安定。
Change: 単一系から同期多様体上の写像tan(beta s)+epsilon sへ進み、不変測度を改めて導出。
Metric: 縦横指数、seed単位区間、尺度とCDF、Cayley点群、対数感度累積。
Result: すべての測定条件で横方向拡大を観測し、解析的正値下限を支持した。
Interpretation: 同じ円周上の点群やCauchy周辺分布でも、同期安定性は保証されない。
これは同期多様体近傍の局所的・不変測度に関する結論で、任意の初期値や
例外的周期点、負結合、epsilon>=1への結論ではない。
Next Action: この結合で安定完全同期を探す実験を続ける根拠は得られなかった。
次は写像出力を混合する拡散結合など、同期上の写像を保ったまま横倍率を制御できる
別の結合式を独立に事前登録する。相互加算結合の失敗をtan写像一般の失敗とはしない。

### 図と元データ

- [縦横指数・尺度](artifacts/fig1_stability_scale.pdf): predictions.csv、artifacts/summary.csv。
  近似95%区間は描画済みだが、多くの点でマーカーより小さい。数値幅はCSVに保存。
- [複素平面と角度密度](artifacts/fig2_complex_plane.pdf): artifacts/representative_orbits.npz。
  同期上の代表seedのみ。円の形自体を同期やカオスの証拠にしない。
- [横方向対数感度](artifacts/fig3_transverse_gain.pdf): artifacts/log_gain.csv。
  線形変分の対数累積であり、有限の二軌道間距離そのものではない。

各図にPNG/SVGもある。seed別の初期値・尺度・KS距離・指数・数値監査はartifacts/seeds.csv。
結果再描画は --plot、ハッシュと代表軌道の再検証は --validate。
再実行は空フォルダにadditive_sync.pyだけを置いて --theory → --run → --plot → --validate。
記録された環境での再現を基本とし、別の数学ライブラリで点ごとの一致は保証しない。

コード1ファイル追加、既存E1テスト1ケース拡張。生成物の新規保存先はこのrunのみ。
E0・E1の既存結果は保持。コードとテスト以外はGitHubに送信しない。
