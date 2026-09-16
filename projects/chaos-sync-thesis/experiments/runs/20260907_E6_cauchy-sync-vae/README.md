# E6：Cauchy潜在とtan同期VAEの段階比較

状態：completed（2026-09-07、pilot）。既存runを保存して追加する独立pilot。

## 目的・仮説・反証条件

添付の「軌道読み出しから分布表現へ」を、学習・事前生成まで実行する最小VAEへ接続する。
H1：同じ潜在予算でCauchy事後がGaussianよりtest負ELBOを改善する。
H2：Cauchyへtan力学を追加すると改善する。
H3：同じtan力学へ同期結合を追加すると改善し、入力依存情報も残る。
各比較は同一split・3学習seedで行う。平均差が改善しなければpilot支持なし。
改善しても3 seedのみでは優位確立としない。同期誤差ゼロ単独ではH3を支持しない。

## 事前固定した構成

- sklearn同梱digits、画素/16を0.5以上で二値化。固定層化60/20/20、split seed 607。
- 64→64→(位置8,尺度8) encoder、8→64→64 decoder。全モデル同じ学習パラメータ数。
- Gaussian / Cauchy、Cauchy+tan(beta=1.5,T=4,kappa=0)、同kappa=0.25,0.45,0.5。
- 潜在8実数を4組の2ノードに割り当て、出力拡散結合を各組で適用。
- 全モデルに固定観測2*atan(z)/piを使用。これはTM係数ではない。
- 位置=3*tanh(raw)、尺度=0.1+1.9*sigmoid(raw)、decoder logits=8*tanh(raw/8)。
- 初期潜在の標準Gaussianまたは標準Cauchy prior。生成にも同じ力学・観測を適用。
- KL重み1、Adam lr=0.001、40 epoch、batch=128、4独立posterior sample。
- 全モデルでscore-function勾配＋leave-one-out baselineを使う。tan反復を微分しない。
- validation負ELBO（16 sample）だけでepoch選択。testは選択後64 sampleで一度評価。
- 学習seed 607,608,609。CPU float64。再構成・KL・負ELBO・同期・生成物を別保存。

## 数学上の区別

q(z0|x)からz0を引き、固定写像Gを通したp(x|G(z0))を尤度とする。
ELBO=E_q log p(x|G(z0))-KL(q(z0|x)||p(z0))。Gは非可逆でもよい。
終端分布を独立Cauchyと近似してKLを計算しない。decoderへ位置・尺度を直送しない。
有限有界Bernoulli logitsなので再構成負対数尤度は有界。Cauchy状態の二乗期待値を仮定しない。

本pilotは固定beta・初期分布符号化型。既存TANGENT_SYNC_VAE_THEORY.mdの
beta保持・時間相関読み出し型とは別の比較であり、その復号定理は移用しない。
分布閉包は非結合一段で別検証。中間結合で生じる依存を無視した閉包は使わない。

## 実行

```powershell
python experiments/runs/20260907_E6_cauchy-sync-vae/run_experiment.py
python -m pytest experiments/tests/test_e6_cauchy_sync_vae.py -q
python experiments/runs/20260907_E6_cauchy-sync-vae/validate_artifacts.py
```

出力はこのrunのartifacts/へ保存し、既存artifactsがあれば拒否する。
再実験は --output に新しいrunの出力先を渡す。
生成画像は事前からの無条件sampleと確率画像を区別する。画質や新規性は別評価。

## 次の比較

E6D Adaptive TMは未実装。極と係数の両方の実数自由度・伝送bit数を予算化する。
E0G location-scale閉包、E0H有限標本推定、E3D分布同期を独立ゲートとして追加する。
量子化と実符号長、非カオス対照、大規模MNIST、発表題目変更は本pilotの対象外。

## 実行結果（2026-09-07）

状態：**completed / pilotで性能優位を支持せず**。
1078 train / 359 validation / 360 test、3学習seed×6モデルを完走。
全18条件で最良epochは40（予算末端）なので、最適化の収束は確認できていない。

| モデル | test負ELBO↓ | NLL↓ | KL | 有界ノード差↓ |
|---|---:|---:|---:|---:|
| Gaussian | 23.539155 | 21.512208 | 2.026947 | 0.247689 |
| Cauchy | 24.828569 | 23.851874 | 0.976695 | 0.339672 |
| Cauchy+tan、kappa=0 | 25.366643 | 25.364777 | 0.001865 | 0.317525 |
| 同kappa=0.25 | 25.362904 | 25.361146 | 0.001758 | 0.128205 |
| 同kappa=0.45 | 25.357141 | 25.355703 | 0.001437 | 0.008343 |
| 同kappa=0.5 | 25.357538 | 25.356129 | 0.001409 | 0.000000 |

各値は3 seed平均、NLL/KL/負ELBOの単位はnat/画像。
train画素頻度だけの入力非依存Bernoulli対照のtest NLLは25.079382。
この対照は潜在なし尤度でありVAEの再構成NLLとの意味を区別する。
VAEの負ELBOは周辺NLLの上界であり、上界の順序だけでは真の周辺尤度順位を確定しない。

- H1（Cauchyの改善）：支持なし。Gaussianが全seedで良い。
- H2（カオスの改善）：支持なし。tan追加でKLは約0.002へ落ち、負ELBOが悪化した。
- H3（同期と入力保持の両立による改善）：有用性を支持しない。
  kappa=0.5は全評価sampleと事前生成sampleでノードが一致。
  kappa=0に対して負ELBOは平均約0.0091だけ小さいが、Cauchy単体・Gaussianを上回らず、
  KLは約0.0014、画像は入力差に乏しい。点推定の微差を同期生成性能の確立とはしない。
- beta=1.5の不変尺度0.858560、縦指数1.645069、
  横断線形安定の下側境界kappa≈0.403500（0〜0.5範囲）。
  0.25は境界以下、0.45は以上。ただし4stepの有限距離収束を線形指数だけで保証しない。
- E0G予備検証：非結合一段9条件がDKW閾値0.008657680を全通過。
  最大KS=0.004865513。結合系全体の閉包や生成有用性は証明しない。

## 検証と再利用可能な成果物

- 数学・モデル契約テスト：8 passed。
- 保存物の整合性監査：262/262（[validation.json](artifacts/validation.json)）。
  これは保存物のチェック数であり、独立した262研究実験の成功ではない。
- [config](artifacts/config.json)、[environment](artifacts/environment.json)、
  [seed別数値](artifacts/metrics.json)、[閉包判定](artifacts/closure.json)、
  [hash](artifacts/sha256.json)を保存。
- data_split.npzに元画素、二値入力、ラベル、全split index。
  各モデルの.ptはvalidationで選んだ推論checkpoint（optimizer再開用ではない）。
- 各.npzにtest個票のNLL/KL/負ELBO、再構成確率、同期指標、prior初期値・終端値・画像を保存。
- 全40epochのvalidation推移、代表seedの確率画像・Bernoulli画像・再構成一覧を保存。
- 視覚確認：Gaussianも粗い再構成に留まる。同期モデルはほぼ同じ確率画像を出し、
  prior Bernoulli画像は画素雑音が強い。良質な数字生成は達成していない。

## 判断と次の実験

**VAEとして学習・再構成・事前生成する実装は成立。高品質生成と同期の有用性は未達。**
コード故障の迂回で結果を作ったものではなく、有限勾配を持つscore推定で学習したpilotの負結果である。
ただし最適化難度と初期分布情報の混合の寄与は未分離である。

1. T=0,1,2,4とscore推定分散をtrain/validationで診断し、現在のtestは選択に再使用しない。
2. 非結合および一歩完全同期で厳密な閉包が使える条件では、
   w'=tan(beta*w)を直接進めて終端分布からsampleする分布パラメータ学習を比較する。
   これは周辺分布からdecoderが読む本構成の期待値を解析的に周辺化する候補で、
   時間相関を読むモデルでは軌道分布を別途保持する必要がある。
3. 別系列のbeta保持・時間相関型VAEを、解析相関直結と単一軌道の対照付きで実装する。
   入力を初期分布だけへ置く方式の負結果を、beta保持型の否定へ一般化しない。
4. E1C/E6D Adaptive TMは残存情報と実数・bit予算を確認してから追加する。
