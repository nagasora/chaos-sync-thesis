# GitHub研究運用

## 目的

GitHubを、研究の結論を先に決める場所ではなく、未解決点・根拠・検証・判断のつながりを追跡する場所として使う。
実験の再現可能な正本はリポジトリ内のコード・設定・結果メモであり、Issueはその入口と判断ログである。

## テーマとProjectの対応

リポジトリでは `projects/<research-theme>/` を研究テーマの正本にする。GitHub Projectはテーマごとに1件を基本とする。

最初のProjectは **Chaos Synchronization Thesis** とし、`projects/chaos-sync-thesis` を対応する正本にする。

| Project field | 用途 |
| --- | --- |
| Status | `Todo` → `In Progress` → `Done` |
| Research state | `Needs evidence`、`Ready to experiment`、`Review`、`Parked` |
| Research item | `Question`、`Experiment`、`Decision`、`Literature` |
| Stage | `E0`〜`E5`、`Cross-cutting` |
| Evidence | 原典・再現計算・未検証を区別する短い根拠 |
| Next action | 次に一つだけ行う検証または判断 |
| Priority | 卒論の最小スコープ E0〜E3 を優先するための順位 |

## Issueの単位

Issue一件には一つの研究上の対象だけを置く。

- **Question**: まだ答えがない研究上の問い。根拠、候補仮説、反証条件、次の小さな検証を残す。
- **Experiment**: 一つの仮説を変数一つで検証する実験。基準、変更、指標、失敗条件、成果物パスを残す。
- **Decision**: 複数の選択肢の採否。結論だけでなく、採った理由と再検討条件を残す。
- **Literature**: 原典確認が必要な式・主張・比較。論文名と該当箇所を明示する。

「同期が起きる」と「入力を復号できる」は別のIssueにする。同期による自由度縮約を、必要情報を保持した圧縮と同一視しない。

## 研究を進める流れ

1. 未解決点をIssueフォームで起票し、対応するProjectへ追加する。
2. 原典または既存の再現計算を確認し、`Evidence`欄とIssueへ根拠を追記する。
3. 実験が必要なら、仮説を一つに絞ったExperiment Issueを作る。基準実験は保持する。
4. 現行実験のコード・設定・レポートは `experiments/E<段階>/`、実行時データ・数値・図はその `artifacts/<実行名>/` に保存し、[結果・データ対応台帳](../experiments/RESULTS_INDEX.md) とIssueからリンクする。旧 `experiments/runs/<experiment-id>/` は履歴として保持する。
5. 結果を「支持・反証・未判定」に分けて記録する。スコアや結論を未実行のまま記載しない。
6. 判断が確定したらDecision Issueへ理由と再検討条件を残し、Projectの状態を更新する。

## 最初に起票する候補

1. **E1AでTM・時間遅延読み出しが既知の人工カオス源を回収できる条件は何か**
2. **第一の保存対象を入力全体・クラス・両方のどれに置くか**
3. **長時間平均と有限時間過渡のどちらに入力依存情報が残るか**
4. **有限個のTM・グラフモードで識別可能性を評価する指標は何か**
5. **結合Boole系で同期安定性と復号可能性が両立する領域はどこか**

これらは結論ではなく、既存の研究コンテキストから整理した最初の問いである。各Issueでは原典または再現計算によって更新する。
