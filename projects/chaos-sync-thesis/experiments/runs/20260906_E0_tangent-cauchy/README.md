# E0: tan(beta x) のコーシー保存・臨界挙動

状態: completed（全条件実行済み、全仮説採択ではない）。実行結果は metrics.json、条件は config.json を正本とする。
既存Boole・K-fold実験は変更しない。E0A〜Fは本run内の検証項目である。

## 理論前提（実験前固定）

中心Cauchy X~C(0,gamma)、beta,gamma>0 とする。a=beta*gamma,
u=atan(y) とすると逆像 x_n=(u+n*pi)/beta の密度和は

    p_Y(y) = a/[pi*(1+y^2)] * sum_n 1/[(u+n*pi)^2+a^2].

周期化核のFourier係数は、実線上のCauchy核の積分から
exp(-2*a*abs(k))/a となる（周期pi）。幾何級数を足すと

    sum_n 1/[(u+n*pi)^2+a^2]
      = sinh(2*a)/[a*(cosh(2*a)-cos(2*u))].

cos(2*u)=(1-y^2)/(1+y^2) を代入すると

    p_Y(y) = tanh(a)/[pi*(y^2+tanh(a)^2)].

従って中心Cauchy族の押し出しは gamma'=tanh(beta*gamma) に閉じる。
極は可算な零測度集合。任意の密度や結合した確率変数の閉包は主張しない。
数値branch和には有限打切り誤差があるため、閉形式との差を尾部上界で判定する。

beta>1の正の固定点を atanh(gamma)/gamma=beta で求める。
不変測度上の積分 E log(1+X^2)=2 log(1+gamma) により
lambda=log(beta)+2 log(1+gamma*)。独立角度求積で照合する。
beta=1では coth(gamma)^2-gamma^(-2)=2/3+gamma^2/15+O(gamma^4)。
gamma_t~sqrt(3/(2t))。Q_t=gamma_t^(-2)-2t/3 の定数収束は要求しない。

## 固定条件・判定

- E0A: beta=[.5,.9,1,1.01,1.1,1.5,2]、尺度=[.1,1,10]、10 seed、10万標本。
  理論尺度固定KS。float64の210実行が主検定族、family alpha=.01のDKW。
  float32は同じ初期乱数の精度診断であり、主検定族へ混ぜない。
- E0B: 広域は同じbeta/尺度、5 seed、4096標本、4096 step。
  長時間はbeta=[.9999,1,1.0001]、尺度1、5 seed、512標本、200000 step。
  保存時刻は0、2の累乗、終端。再抽選なし。
  理論系列は広域と1±epsilonの和集合、尺度3条件。
  上限max(4096,ceil(20/abs(beta-1)))、beta=1は200000。
  beta>1は固定点相対誤差<=1e-8、beta<1は尺度<=1e-12。
  beta=1は代数緩和を記録し、収束判定を付けない。
- epsilon=[.1,.05,.02,.01,.005,.002,.001,.0005,.0002,.0001]。
- E0C: beta>1の格子和集合、20 seed、200000 step、固定点Cauchy初期化、burn-inなし。
  50000 stepごとの4窓と全区間。KSは距離のみでiid検定なし。
- E0D: R_gamma、理論と経験のlog-log fit（全域とepsilon<=.01）。
  緩和時間は局所乗数と固定点+相対1e-3の摂動を1e-6まで減衰させた推定を比較。
- E0E: Cの軌道を再利用。log(beta)+2*log(hypot(1,x_next))。
  seed単位Student-t 95%区間（独立seedを仮定した近似区間）、R_lambdaを保存。
  理論を区間が含むかは診断であり、多重比較を調整した採択検定ではない。
- E0F: float32はA全条件とCのbeta=[1.0001,1.01,2]。
  全状態実行から極近傍100点・大引数100点（重複保持）を100桁監査。
  極距離のfloat近似による選抜なので全点の高精度最小値の保証ではない。
  非有限値は該当標本の更新を停止し位置を保存。成功標本に置換しない。
  seed全体の指標を不完全とし、有限標本だけの値で合格させない。
- K-fold: K=2,3,4、一ステップの標準Cauchyとモード恒等式だけを確認。

seedはSeedSequence([20260906, stage, condition_index, replicate])。
同じ条件のfloat32/64には同じfloat64初期標本を使い、型への丸めだけを変える。
裾指標はP(|X|>10*gamma),P(|X|>100*gamma)。母分散を使用しない。
E0Bの有限標本誤差、C/Eの有限時間区間は記述的評価。
不一致・未収束を含めて報告し、結果によるseed/観測窓変更はしない。

## 実装と再現

既存BooleのCDF・KS・IQR実装をimportして再利用する。
NumPy, SciPy, Numba, mpmath, matplotlib（環境に導入済み）を使用。
Numbaはfastmathなし。float32の引数乗算もfloat32へ明示的に丸める。
全時刻の状態更新は実行し、集団はブロック処理でメモリを制限する。

    python experiments/runs/20260906_E0_tangent-cauchy/e0_tangent_cauchy.py --prepare
    python -m pytest experiments/tests/test_e0_tangent_cauchy.py
    python experiments/runs/20260906_E0_tangent-cauchy/e0_tangent_cauchy.py --benchmark
    python experiments/runs/20260906_E0_tangent-cauchy/e0_tangent_cauchy.py --run
    python experiments/runs/20260906_E0_tangent-cauchy/e0_tangent_cauchy.py --validate

設定とコードのSHA-256を実行開始前に固定。既存metrics/artifactsは上書き拒否。
再実行は別の出力先を--outputで指定する。--validateは保存NPZ/CSVとハッシュを照合。
主図6枚・補足、図の元CSV、代表軌道、初期値台帳、監査点を保存する。
全軌道はseed・コード・環境で再生成可能。浮動小数点軌道の他環境でのbit一致は保証しない。

## 結論の境界

中心Cauchy族の閉包と有限時間統計を区別する。弱収束を全初期値の収束、
尺度有限を状態有界、状態微分下限をBPTTの安定性と読み替えない。
同期・圧縮・TM復号・学習は対象外。次の結合系は不変測度から再導出する。


## 実行結果と判断

全条件を331.9秒で計算（図生成時間は別）。A:420、B:120、C:300実行。
非有限値を生じた実行は0件。Bの低betaでは浮動小数点のゼロ吸収があり、
ゼロの累積出現数は185720339（個別軌道数ではない）。

| 項目 | 結果 | 解釈 |
|---|---|---|
| E0A float64 | DKW 209/210通過 | 全条件通過の主張はしない |
| E0A超過条件 | beta=1, gamma0=1, condition=7, replicate=4 | KS=0.0073901139 > 0.0072956922 |
| 同一標本の100桁診断 | KSは同じ。CDF差の最大7.07e-17 | 数値誤差による説明は支持されない。有限標本の不一致を保持 |
| beta=1, gamma0=1, t=200000 | 正規化尺度0.9999917979、逆二乗増分0.6666671667 | 代数緩和の理論系列と整合 |
| E0E float64 | 12/12 betaで理論値がseed 95%区間内 | 記述的な有限時間整合、多重検定の合格ではない |
| epsilon<=.01の経験尺度fit | 指数0.5020531 | 有限範囲の補助fit |
| 同じ範囲の経験Lyapunov fit | 指数0.4915899 | 同範囲の理論fit自体も0.4890760。厳密な指数推定としない |
| float32, beta=1.01 | lambda=0.3542450、理論0.3267965 | 精度依存のずれ。主計算には使用しない |

主図は artifacts/fig1〜fig6、補足は supplement_numerical_audit.png。
個々の初期条件、窓別値、集団チェックポイント、図の元データはCSV/NPZに保存した。

追加診断は事後解析として a_failed_case_diagnosis.json と repetition_diagnosis.csv に分離した。
前者は同じSeedSequenceの10万標本を100桁tanで再評価しSciPy KSと照合したもの。
後者は保存済み代表軌道末尾50000点のnp.uniqueによる状態数である。
float32のbeta=1.01では3500種類、beta=2では4187種類だった。
これは有限精度による反復の診断であり、全seedの周期分類や誤差原因の完全同定ではない。

### 検証と実行ソース

- 対象テスト12件通過。最新mainの既存実験群を含む49件すべて通過。既存テスト用の実在PDFをGit管理外で配置した。
- 保存物の検証317項目通過。検証項目の追加により件数が増える場合はvalidation.jsonを正本とする。
- 最初のvalidatorはnp.bool_をJSON化できず失敗した。演算結果をPython boolへ変換して修正し、
  不完全bundleでも不合格JSONを出せる回帰テストを追加した。
- 数値実験を行ったソースはartifacts/executed_source.pyに保存。
  environment.jsonのscript_sha256と一致し、現行コードとはvalidate関数だけが異なることをASTで検証する。
  このスナップショットは直接実行する入口ではない。
- SHA監査はバイト列に厳密。既存Boole依存はLF版を使用したため、checkoutの改行差に注意。
  PRでは当該依存のLFと本runのバイト列保持を.gitattributesに明示する。

## 次の行動

E0Aの不一致1件とfloat32の精度依存を残した状態で、一次元の理論と実装証拠を確定する。
この結果だけでコーシー閉包の破綻、全軌道のエルゴード性、圧縮性能、学習安定性を主張しない。
2自由度へ進む場合は結合式・同期多様体・不変測度を別途事前登録する。
