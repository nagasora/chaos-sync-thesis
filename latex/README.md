# LaTeX 作業領域

このフォルダだけを VS Code で開けば、卒論・発表資料・理論ノート・テンプレート・輪読資料を同じ操作で編集できます。標準エンジンは LuaLaTeX、文字コードは UTF-8 です。

## VS Code で使う

1. VS Code で `C:\研究\latex` をフォルダとして開く。
2. 推奨拡張機能 `James-Yu.latex-workshop` をインストールする。
3. 対象の `.tex` を開いて保存する。

保存時に文書が自動ビルドされ、PDFはVS Code内のタブで更新されます。SyncTeXも有効です。生成物はソースの隣ではなく `build/<文書名>/` に保存されます。

## 文書一覧

| 文書 | ソース | 出力先 |
|---|---|---|
| 卒業論文 | `thesis/main.tex` | `build/main/` |
| 研究発表 | `slides/chaos-sync-presentation.tex` | `build/chaos-sync-presentation/` |
| 理論ノート・想定問答 | `notes/research-topic-qa.tex` | `build/research-topic-qa/` |
| E0・E1定式化と結果 | `notes/e0-e1-formulation-results.tex` | `build/e0-e1-formulation-results/` |
| 研究発表テンプレート | `templates/research-presentation-template.tex` | `build/research-presentation-template/` |
| B4輪読 2026-05-27 | `seminars/b4-2026-05-27.tex` | `build/b4-2026-05-27/` |

## PowerShell から使う

1文書をビルドします。

```powershell
.\scripts\build.ps1 -Source .\thesis\main.tex
```

登録済みの全TeXを個別の出力先へビルドします。

```powershell
.\scripts\build.ps1 -All
```

対象文書の生成物だけを削除して再構築します。

```powershell
.\scripts\build.ps1 -Source .\slides\chaos-sync-presentation.tex -Clean
```

`-MaxPasses` は2〜5で指定でき、既定値は5です。終了コードは成功が0、コンパイル失敗が1、不正な引数が2です。BibLaTeXの `.bcf` が生成された場合はBiber、BibTeXの指定を検出した場合はBibTeXを自動実行します。

## 構成

- `shared/figures/`: 複数文書で共有する図
- `shared/bibliography/`: 複数文書で共有する文献データ
- `shared/styles/`: 複数文書で共有するスタイル
- `build/`: PDF・ログ・SyncTeXなどの生成物。手作業で編集しません。

旧 `.latexmkrc` は再利用せず、履歴保全のため `..\archive\legacy-config\` に保存しています。
