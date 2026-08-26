# クラウド研究プロジェクト移行記録

移行日: 2026-08-05

## 対象

- 移行元: ChatGPT クラウドプロジェクト「研究」
- クラウド project ID: `g-p-6a01ac73c360819187113e618b50a83d`
- 2026-08-05時点の移行先（旧表記）: Codex ローカルプロジェクト `C:\研究`
- 2026-08-07以降の研究プロジェクト: `C:\研究\projects\chaos-sync-thesis`
- 2026-08-07以降のLaTeX作業領域: `C:\研究\latex`
- ローカル project ID: `8375c784-3054-4bc1-a518-bc6897cc5a47`
- 同一パスには別のローカル登録 `local-dedf2a6c71ada8fd6235dab9cf5e042c` も存在する。

## 移行済み

- クラウド会話から研究目的、理論、仮説、未解決点、実験計画を統合した。
- 将来の Codex 作業で自動参照できるよう、ワークスペース直下に `AGENTS.md` を追加した。
- 主要な先行研究PDFを `references/papers` に集約し、旧名称・新パス・SHA-256を `references/README.md` に記録した。
- 卒論の最小スコープを E0〜E3 として明文化した。
- クラウド会話と成果物の索引を作成した。

## ローカルに存在する主要原典

- `ARTIFICIAL KURAMOTO OSCILLATORY NEURONS.pdf`
- `Infinite Dimensional Chaotic Synchronization for Randomly Coupling Systems and Dynamical Phase Transition.pdf`
- `Universal Mass Gap Formula ... .pdf`
- `ランダム結合系の大自由度極限でのカオス同期現象.pdf`
- `Synchronization of chaotic systems.pdf`
- `Chaotic synchronization of mutually coupled systems arbitrary proportional linear relations.pdf`

## クラウドで生成されたことを確認した成果物

- `カオス同期による情報圧縮_研究構想詳細整理.md`
- `chaotic_synchronization_information_compression.tex`
- `chaotic_synchronization_information_compression.pdf`
- `カオス同期生成モデル_教授相談用研究構想メモ.docx`
- `カオス同期型エンコーダデコーダ_数理モデル構築構想レポート.docx`
- `カオス同期型エンコーダデコーダ_数式TeX組版版.pdf`

## 制約

ChatGPT クラウド会話内で生成されたファイルは `sandbox:/workspace/...` の一時 URL で参照されており、Codex のローカルファイル API から直接コピーできない。したがって、今回の移行では内容・判断・再生成に必要な研究コンテキストを先にローカル化した。

クラウドのチャット自体をローカル Codex プロジェクトへ付け替える公式なハンドオフ機能はない。会話はクラウド側に残し、ローカル側ではこの文書群を正本の作業コンテキストとして扱う。

## 再構成済み成果物

- 旧パス: `slides/chaotic_synchronization_information_compression.tex`
  - 現在の正本: `C:\研究\latex\slides\chaos-sync-presentation.tex`
  - クラウドで確認できた「本編20枚＋補足4枚」の内容を、ローカル正本として再構成した。
  - 旧クラウドファイルそのものではないため、取得できた場合は差分を確認する。

## 2026-08-07のワークスペース再編

- 旧プロジェクト名 `2026B4_1029350850_長松蒼空` を `projects/chaos-sync-thesis` へ移動した。
- すべてのTeXソースを `C:\研究\latex` に集約した。
- PDFとLaTeX中間生成物の出力先を `latex/build/<文書名>/` に分離した。
- 旧 `.latexmkrc` 2件は `C:\研究\archive\legacy-config` に保存した。
- 旧日本語ファイル名は台帳に残し、現在の作業パスを英数字へ統一した。
- 5件のTeXをLuaLaTeXで再ビルドし、全84ページを描画確認した。
- PowerPointの発表者ノート内にあった旧絶対パス12件を現行パスへ更新し、表示差分がないことを確認した。
- 詳細な検証記録は `C:\研究\archive\tooling\workspace-restructure-20260807\README.md` に保存した。

## 残作業

- 旧 Beamer ソースをクラウド回答本文から回収できた場合、`archive/cloud_chatgpt` に保存して差分確認する。
- 取得できないバイナリ成果物は、ローカルの Markdown / TeX 正本から再生成する。
- クラウド原本を取得できた場合は、現在の正本との差分を確認する。
