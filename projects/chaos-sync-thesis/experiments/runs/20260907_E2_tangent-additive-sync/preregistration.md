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
