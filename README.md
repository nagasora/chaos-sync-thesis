# 研究ワークスペース

このリポジトリは、卒業研究、LaTeX文書、論文ライブラリを用途別に管理します。研究テーマの概要は[カオス同期による動的情報圧縮の研究](projects/chaos-sync-thesis/README.md)を参照してください。

## 入口

- [LaTeX作業領域](latex/README.md): TeXソースと文書別のビルド
- [卒業研究](projects/chaos-sync-thesis/README.md): 研究コンテキスト、実験、発表資料、中核論文
- [一般論文ライブラリ](library/README.md)
- [運用ガイド](AGENTS.md)
- [GitHub研究運用](projects/chaos-sync-thesis/docs/GITHUB_RESEARCH_WORKFLOW.md)

研究状況と実験記録の索引:

- [研究状況（2026-10-02時点の公開リポジトリ記録）](projects/chaos-sync-thesis/docs/RESEARCH_STATUS_2026-10-02.md)
- [実験インデックスと再現ガイド](projects/chaos-sync-thesis/experiments/EXPERIMENT_INDEX_2026-10-02.md)

輪読資料と大学院出願書類は、このリポジトリには含まれていません。

## ディレクトリ構成

```text
.
├── latex/                         # TeXソースと文書別build
├── projects/chaos-sync-thesis/   # 卒業研究本体
└── library/                       # 一般論文ライブラリ
```

`.tmp`、`.codex_tmp`、`.pnpm-store`、`.tmp.drive*` はツール管理用のため、ルートに残しています。

## GitHubでの研究管理

- `projects/<research-theme>/` をテーマごとの正本にする。
- GitHub Project はテーマ全体の問い・実験・判断を一望する場、Issue は一つの未解決点または検証単位にする。
- 実験結果は Issue だけで完結させず、再現に必要な設定・コード・結果メモをテーマ配下へ保存してリンクする。
- 著作権のある論文 PDF、個人情報を含む出願書類、生成 PDF/PPTX/DOCX は GitHub に含めない。台帳、Markdown、LaTeX、実験コードを正本とする。
