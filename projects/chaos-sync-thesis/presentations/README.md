# 発表資料

PowerPoint、書き出しPDF、発表者向けノートを保存します。LaTeXソースとその生成物は、ワークスペース共通の [`../../../latex`](../../../latex/README.md) に分離しています。

## 研究テーマ発表

- `research-topic.pptx`: 編集可能なPowerPoint正本。発表者ノートを含む
- `research-topic.pdf`: 閲覧用PDF
- `notes/research-topic-speaker-notes.md`: 発表用カンペ・理論補足
- `notes/research-topic-speaker-notes.docx`: 上記のWord版
- `notes/research-topic-theory-qa.md`: 詳細理論ノート・想定問答のMarkdown正本

対応するTeXは `C:\研究\latex\notes\research-topic-qa.tex`、生成PDFは `C:\研究\latex\build\research-topic-qa\research-topic-qa.pdf` です。

## Beamer版

研究発表のBeamer正本は `C:\研究\latex\slides\chaos-sync-presentation.tex` です。ビルド方法は [`C:\研究\latex\README.md`](../../../latex/README.md) を参照してください。

## テンプレート

- `templates/research-presentation-template.pptx`: PowerPoint版
- `C:\研究\latex\templates\research-presentation-template.tex`: LuaLaTeX Beamer版
- `C:\研究\latex\build\research-presentation-template\research-presentation-template.pdf`: 生成PDF

PowerPointテンプレートの内部構成は変更していません。
