# E1B: 二源の読み出しと順序の識別不能性

## 目的・位置づけ

E1Aで認めた時間構造の読み出しが二源へ拡張できるか、観測で消えた情報をTMが回復したと誤認しないかを確認する。教材E1Bと[TM-B方針](TM_DYNAMICS.md)に対応する。同期・圧縮率・信号復元はこの実験の対象外。旧 `runs/20260901_E1B_two-source-identifiability` は変更せず、今回の固定特徴・独立validation方式を別の追試として記録する。

## 理論と事前仮説

独立源は $x_{j,t+1}=α_j(x_{j,t}-1/x_{j,t})$。各源の有限窓からmedian・half-IQRを除去して $s_j$ とする。これは原写像への共役座標の移動であり、標準化後の源を原式と同じ写像と仮定しない。尺度の自明な手掛かりを抑えるための、源にアクセスする制御実験である。

- 可逆正対照: 既知 $A=[[1,.35],[-.2,1]]$、$y=As$、oracleは $A^{-1}y$、directは $y$ を読む。逆行列を知るoracleは盲分離ではない。
- 対称負対照: $y=(s_1+s_2)/√2$。**同じ2配列を交換**した `(a,b)` と `(b,a)` を必ず同じsplitに同数含める。有限標本でも観測は同一。
- 決定論的予測 $p(y)$ の座標平均MSEには厳密恒等式 $MSE=E[(a-b)^2]/4+E[∥p-(a+b)/2∥^2/2]$ がある。従って順序付きRMSEの下限は $√{E[(a-b)^2]/4}$。交換順序の損失であり、順序なしパラメータ集合の完全損失ではない。
- H1: oracle TMの未使用alpha test RMSE < 0.02。未達なら固定読み出しの二源拡張を支持しない。
- H2: 対称観測の交換対で特徴・予測が一致し、恒等式が成立、理論下限を下回らない。違反は手法の成功ではなく評価実装を調査する。
- full-rank directの精度とTM/Fourier差は探索的に報告。TM優位を成功条件にしない。順序なし誤差は**順序付き学習の予測を並べ替えた診断**で、順序なし復号器の最良性能ではない。

## 実験手法・凍結条件

`config_identifiability.json` を結果を見る前に固定する。float64、burn-in 20000、観測4096。trainは5alphaから異なる2値の10組×24seed×2順序=480行、validationは同10組×8seed×2=160行、testは未使用4alphaの6組×24seed×2=288行。合計464衝突対・928源軌道。seed・alpha組・源役割からSeedSequenceで初期値を生成し、交換対は元軌道を共有する。

全手法は**観測サンプル全体で16実数**。2観測では各8、1観測では16。TMは各チャネルのCayleyモードk=1..2（1観測では1..4）、lag=1,2の自己相関Re/Im。raw Fourierは各チャネル8（1観測16）帯域、Cayley FourierはRe/Im各4（1観測8）帯域。全チャネルを窓ごとにrobust標準化する。Hann・DC除去・log有限標本powerを使い、Cauchy母PSDとは解釈しない。次元は名目容量であり実効自由度の一致を保証しない。観測数による1チャネルの割当差を含めて報告する。

全9条件で同じridge候補5個、trainのみでfit、独立validationで順序付きMSE最小を選択し、全モデルを凍結してからtest生成。test結果による特徴・penalty・閾値の変更は行わない。train平均の定数予測を対照に追加。交換対とalpha組をまとめたseed単位のpaired bootstrap 10000回・95%点別区間を報告（多重比較補正なし）。初期値・生軌道・正規化値・特徴・全予測・seed/alpha組別指標・図の元CSV・コードZIP・環境・SHA256を保存。失敗軌道を除外／再生成しない。

## 結果・考察・展望

未実行。実行後に事前条件を保持したまま追記する。

## 再現コマンド

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python experiments/E1/run_identifiability.py --output experiments/E1/artifacts/fixed_identifiability
python experiments/E1/validate_results.py --output experiments/E1/artifacts/fixed_identifiability
```

出力先の既存成果は上書きしない。再実行は別のoutputを指定する。
