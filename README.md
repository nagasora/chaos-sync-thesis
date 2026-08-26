# 研究ワークスペース

`C:\研究` は、卒業研究、LaTeX文書、論文ライブラリ、輪読資料、出願書類を用途別の英数字パスで管理するワークスペースです。研究の再現可能なソース・記録は GitHub の [`nagasora/Sora_research`](https://github.com/nagasora/Sora_research) と同期します。

## 入口

- [LaTeX作業領域](latex/README.md): VS Codeで `C:\研究\latex` を開き、`.tex` を保存してビルド
- [卒業研究](projects/chaos-sync-thesis/README.md): 研究コンテキスト、実験、発表資料、中核論文
- [一般論文ライブラリ](library/README.md)
- [輪読資料](seminars/README.md)
- [大学院出願書類](applications/README.md)
- [運用ガイド](AGENTS.md)
- [GitHub研究運用](projects/chaos-sync-thesis/docs/GITHUB_RESEARCH_WORKFLOW.md)

## ディレクトリ構成

```text
C:\研究
├── latex/                         # TeXソースと文書別build
├── projects/chaos-sync-thesis/   # 卒業研究本体
├── library/papers/               # 一般論文
├── seminars/                     # 年度・ゼミ別の輪読資料
├── applications/graduate-school/ # 出願書類
└── archive/                       # 旧設定・検証用一時成果物
```

`.tmp`、`.codex_tmp`、`.pnpm-store`、`.tmp.drive*` はツール管理用のため、ルートに残しています。

## GitHubでの研究管理

- `projects/<research-theme>/` をテーマごとの正本にする。
- GitHub Project はテーマ全体の問い・実験・判断を一望する場、Issue は一つの未解決点または検証単位にする。
- 実験結果は Issue だけで完結させず、再現に必要な設定・コード・結果メモをテーマ配下へ保存してリンクする。
- 著作権のある論文 PDF、個人情報を含む出願書類、生成 PDF/PPTX/DOCX は GitHub に含めない。台帳、Markdown、LaTeX、実験コードを正本とする。
