# Validation Report

## Overall Assessment: Ready to share within E0 scope

この評価は、一般化Boole写像の単一非結合ノードについて、Cauchy不変測度・Lyapunov指数・簡略Cayleyモードを再現できるという限定された主張を対象とする。結合系、同期、情報保持、一般のTM基底の有効性は対象外である。

## Methodology Review

- データ源: 記録済みパラメータから生成した人工決定論的軌道。
- 実行日: 2026-08-06 JST。
- 粒度: 3 alpha × 5 seed = 15軌道。
- 観測: 各軌道100,000点、合計1,500,000点。各軌道の前に20,000点をburn-in。
- 分布評価: Cauchy分布では平均・分散を使わず、IQR尺度とKS距離を採用。
- 理論照合: 閉形式に加え、軌道とは独立な角度座標数値積分でLyapunov指数を再計算。
- 判定: 実行前にコードへ固定した工学的ゲートを使用。統計的有意水準とは解釈しない。

質問、データ、指標、結論はE0の目的に整合している。

## Issues Found

1. Severity: Medium - 初期値を理論不変測度から生成しているため、非定常初期値からCauchy測度へ収束する速さは未検証。
2. Severity: Medium - TM検証は `q_gamma(x)^k` の有限モードに限定され、一般のTakenaka–Malmquist系の完全性・読み出し性能を保証しない。
3. Severity: Low - 100,000点区間で厳密重複はなかったが、より長時間の浮動小数点周期化を否定できない。
4. Severity: Low - 合格ゲートは実装回帰用であり、seed依存性に対する仮説検定や信頼区間ではない。

E0の限定主張を覆す不整合は見つからなかった。

## Calculation Spot-Checks

- 行粒度: Verified - 期待15行、実15行、`alpha × seed`の一意キー15個。
- 有限値: Verified - 全seed行で有限状態。
- alpha別最大値: Verified - CSVから再計算した16集約値が`summary.json`と絶対誤差`1e-15`以内で一致。
- ゲート: Verified - 15個の数値ゲートを閾値から再計算し、保存判定と一致。
- `alpha=0.5`: Verified - 理論scale 1、理論Lyapunov `log(2)`。
- 診断標本: Verified - 3 alpha、各20,000点、全値有限。
- 総合判定: Verified - alpha別判定の論理積と`overall_passed=true`が一致。
- 独立検証: 37項目中37項目通過、失敗0。実験設定、指標記録、環境記録、出力図も照合済み。

## Visualization Review

実行済みNotebookの4パネル図を目視確認した。全指標でseed点が事前ゲート線より下にあり、対数軸・alpha・ゲートが判読できる。初回の日本語フォント欠落は英語ラベルへ修正し、最終図に欠落文字、重なり、切れはない。

## Suggested Improvements

1. 非定常初期値を複数分布から与え、観測窓開始時刻に対する尺度・KS誤差を測る。
2. 観測長を対数的に増やし、誤差収束率と有限精度周期化を測る。
3. E1AではTM次数と尺度をtrainだけで選び、未使用seedで読み出し性能を評価する。
4. 結合系へ進む前に、独立軌道の線形和がCauchy安定則と整合することを確認する。

## Required Caveats for Stakeholders

- E0はBoole写像の可解性を数値再現したもので、複数力学系を結合する有用性を示していない。
- 自由度縮約と情報保持は未検証であり、「情報圧縮に成功した」とは言えない。
- TMに関する結論は簡略Cayleyモードの定常平均に限定する。
