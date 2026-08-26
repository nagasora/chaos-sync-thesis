# 研究プロジェクト運用ガイド

このワークスペースの中心は `projects/chaos-sync-thesis` にある卒業研究です。LaTeXソースは `latex` に集約されています。

## 作業開始時

1. `projects/chaos-sync-thesis/docs/PROJECT_CONTEXT.md` を読む。
2. `projects/chaos-sync-thesis/docs/MIGRATION_MANIFEST.md` でクラウドからの移行状況を確認する。
3. 先行研究の根拠が必要な場合は、まず `projects/chaos-sync-thesis/references/papers` の原典を確認する。

## 研究上の最重要方針

- 最上位目的は「神経系に見られる動的同期に着想を得て、カオス同期を情報圧縮へ応用できるか検討する」である。
- 「同期による自由度縮約」と「必要情報を保持した圧縮・復号」を同一視しない。
- 脳の省エネルギー性や長期記憶の原因がカオス同期である、とは断定しない。
- TM 基底は情報を新たに作る手法ではなく、軌道に残っている入力依存情報を読み出す候補座標系として扱う。
- 数式上の主張、論文中の式、実験結果は、原典または再現計算で確認してから確定事項として記載する。
- 卒論の最小スコープは E0〜E3。VAE・MNIST・大規模生成モデルは、基礎検証後の拡張とする。

## 成果物

- 人が読む研究概要: `projects/chaos-sync-thesis/README.md`
- 統合研究コンテキスト: `projects/chaos-sync-thesis/docs/PROJECT_CONTEXT.md`
- クラウド会話索引: `projects/chaos-sync-thesis/docs/CLOUD_CONVERSATION_INDEX.md`
- 実験の入口: `projects/chaos-sync-thesis/experiments/README.md`
- LaTeX作業領域: `latex`
- PowerPoint・発表ノート: `projects/chaos-sync-thesis/presentations`
- 中核論文: `projects/chaos-sync-thesis/references/papers`
