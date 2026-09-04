# 発表資料

PowerPoint、書き出しPDF、発表者向けノートを保存します。LaTeXソースとその生成物は、ワークスペース共通の [`../../../latex`](../../../latex/README.md) に分離しています。

## 研究テーマ発表

- `research-topic.pptx`: 編集可能なPowerPoint正本。発表者ノートを含む
- `research-topic.pdf`: 閲覧用PDF
- `notes/research-topic-speaker-notes.md`: 発表用カンペ・理論補足
- `notes/research-topic-speaker-notes.docx`: 上記のWord版
- `notes/research-topic-theory-qa.md`: 詳細理論ノート・想定問答のMarkdown正本

対応するTeXは `C:\研究\latex\notes\research-topic-qa.tex`、生成PDFは `C:\研究\latex\build\research-topic-qa\research-topic-qa.pdf` です。

## 有限サイズカオス同期相転移

- [finite-size-chaos-sync-presentation.pptx](finite-size-chaos-sync-presentation.pptx): E0・E3A pilot・一様正結合確認実験をまとめた12枚のPowerPoint。各スライドに出典と発表者ノートを埋め込み済み
- [発表用カンペ](notes/finite-size-chaos-sync-speaker-notes.md): 約11分30秒の読み上げ案、8分短縮案、想定問答、発表直前チェック
- 対応実験: ../experiments/runs/20260831_E3A_uniform-positive-confirmation/

この資料が確認したのは α=1/4 の一様正結合における有限サイズ同期転移です。符号付き分布での普遍性、同期後の情報保持、情報圧縮は未検証として区別しています。

## Beamer版

研究発表のBeamer正本は `C:\研究\latex\slides\chaos-sync-presentation.tex` です。ビルド方法は [`C:\研究\latex\README.md`](../../../latex/README.md) を参照してください。

## テンプレート

- `templates/research-presentation-template.pptx`: PowerPoint版
- `C:\研究\latex\templates\research-presentation-template.tex`: LuaLaTeX Beamer版
- `C:\研究\latex\build\research-presentation-template\research-presentation-template.pdf`: 生成PDF

PowerPointテンプレートの内部構成は変更していません。
