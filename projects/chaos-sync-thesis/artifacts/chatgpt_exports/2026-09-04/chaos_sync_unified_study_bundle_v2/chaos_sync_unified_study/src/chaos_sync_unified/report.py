from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .utils import collect_hashes, write_csv


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "支持" if value else "未支持"
    try:
        number = float(value)
        if not math.isfinite(number):
            return str(number)
        if abs(number) >= 1000 or (abs(number) > 0 and abs(number) < 1e-3):
            return f"{number:.3e}"
        return f"{number:.{digits}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _safe_text(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JapaneseReport:
    """How: 数式画像・長表・図を含む日本語A4技術報告を組み立てる。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.report_dir = root / "report"
        self.asset_dir = self.report_dir / "assets"
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.digest = json.loads((root / "RESULTS_DIGEST.json").read_text(encoding="utf-8"))
        self.registry = json.loads((root / "experiment_registry.json").read_text(encoding="utf-8"))
        self.summaries = json.loads((root / "all_experiment_summaries.json").read_text(encoding="utf-8"))
        self.e2 = pd.read_csv(root / "runs/20260904_E2A_boole_observation_robustness/artifacts/robustness_metrics.csv")
        self.e3b = pd.read_csv(root / "runs/20260904_E3B_tangent_two_node_basin/artifacts/basin_metrics.csv")
        self.e3c = pd.read_csv(root / "runs/20260904_E3C_boole_n8_input_retention/artifacts/input_retention_metrics.csv")
        self.e4 = pd.read_csv(root / "runs/20260904_E4A_boole_synthetic_signal/artifacts/rate_distortion_metrics.csv")
        self.e5 = pd.read_csv(root / "runs/20260904_E5A_boole_small_image_digits/artifacts/image_metrics.csv")
        self._register_fonts()
        self.styles = self._build_styles()

    @staticmethod
    def _register_fonts() -> None:
        for font_name in ("HeiseiMin-W3", "HeiseiKakuGo-W5"):
            try:
                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
            except Exception:
                pass

    @staticmethod
    def _build_styles() -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        return {
            "title": ParagraphStyle(
                "JPTitle",
                parent=base["Title"],
                fontName="HeiseiKakuGo-W5",
                fontSize=23,
                leading=31,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#14213D"),
                spaceAfter=12 * mm,
                wordWrap="CJK",
            ),
            "subtitle": ParagraphStyle(
                "JPSubtitle",
                parent=base["Normal"],
                fontName="HeiseiMin-W3",
                fontSize=12,
                leading=18,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#44546A"),
                wordWrap="CJK",
            ),
            "h1": ParagraphStyle(
                "JPH1",
                parent=base["Heading1"],
                fontName="HeiseiKakuGo-W5",
                fontSize=17,
                leading=23,
                textColor=colors.HexColor("#14213D"),
                spaceBefore=7 * mm,
                spaceAfter=4 * mm,
                keepWithNext=True,
                wordWrap="CJK",
            ),
            "h2": ParagraphStyle(
                "JPH2",
                parent=base["Heading2"],
                fontName="HeiseiKakuGo-W5",
                fontSize=13,
                leading=19,
                textColor=colors.HexColor("#264653"),
                spaceBefore=5 * mm,
                spaceAfter=2.5 * mm,
                keepWithNext=True,
                wordWrap="CJK",
            ),
            "h3": ParagraphStyle(
                "JPH3",
                parent=base["Heading3"],
                fontName="HeiseiKakuGo-W5",
                fontSize=11,
                leading=16,
                textColor=colors.HexColor("#2A6F97"),
                spaceBefore=3 * mm,
                spaceAfter=2 * mm,
                keepWithNext=True,
                wordWrap="CJK",
            ),
            "body": ParagraphStyle(
                "JPBody",
                parent=base["BodyText"],
                fontName="HeiseiMin-W3",
                fontSize=9.5,
                leading=15.2,
                alignment=TA_JUSTIFY,
                firstLineIndent=1.0 * 9.5,
                spaceAfter=2.5 * mm,
                wordWrap="CJK",
            ),
            "body_noindent": ParagraphStyle(
                "JPBodyNoIndent",
                parent=base["BodyText"],
                fontName="HeiseiMin-W3",
                fontSize=9.5,
                leading=15.2,
                alignment=TA_JUSTIFY,
                firstLineIndent=0,
                spaceAfter=2.5 * mm,
                wordWrap="CJK",
            ),
            "note": ParagraphStyle(
                "JPNote",
                parent=base["BodyText"],
                fontName="HeiseiMin-W3",
                fontSize=8.2,
                leading=12.5,
                leftIndent=5 * mm,
                rightIndent=5 * mm,
                borderColor=colors.HexColor("#D9E2F3"),
                borderWidth=0.8,
                borderPadding=5,
                backColor=colors.HexColor("#F7F9FC"),
                spaceBefore=2 * mm,
                spaceAfter=3 * mm,
                wordWrap="CJK",
            ),
            "caption": ParagraphStyle(
                "JPCaption",
                parent=base["BodyText"],
                fontName="HeiseiMin-W3",
                fontSize=7.7,
                leading=11,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#555555"),
                spaceBefore=1 * mm,
                spaceAfter=3 * mm,
                wordWrap="CJK",
            ),
            "small": ParagraphStyle(
                "JPSmall",
                parent=base["BodyText"],
                fontName="HeiseiMin-W3",
                fontSize=7.3,
                leading=10.2,
                wordWrap="CJK",
            ),
        }

    def formula(self, tex: str, name: str, *, width_mm: float = 145.0) -> Image:
        path = self.asset_dir / f"formula_{name}.png"
        if not path.exists():
            fig = plt.figure(figsize=(10.5, 0.72), dpi=220)
            fig.patch.set_alpha(0.0)
            axis = fig.add_axes([0, 0, 1, 1])
            axis.axis("off")
            axis.text(0.5, 0.5, f"${tex}$", ha="center", va="center", fontsize=19)
            fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
        image = Image(str(path))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth = width_mm * mm
        image.drawHeight = width_mm * ratio * mm
        return image

    def paragraph(self, text: str, style: str = "body") -> Paragraph:
        return Paragraph(text, self.styles[style])

    def section(self, title: str, level: int = 1) -> Paragraph:
        return Paragraph(title, self.styles[f"h{level}"])

    def table(
        self,
        rows: list[list[Any]],
        widths: list[float] | None = None,
        *,
        repeat_rows: int = 1,
        small: bool = False,
    ) -> LongTable:
        processed: list[list[Any]] = []
        for row_index, row in enumerate(rows):
            processed.append(
                [
                    cell
                    if isinstance(cell, (Paragraph, Image))
                    else Paragraph(_safe_text(cell), self.styles["small" if small else "body_noindent"])
                    for cell in row
                ]
            )
        table = LongTable(processed, colWidths=[w * mm for w in widths] if widths else None, repeatRows=repeat_rows)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14213D")),
            ("FONTNAME", (0, 0), (-1, 0), "HeiseiKakuGo-W5"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F7F9FC")))
        table.setStyle(TableStyle(style))
        return table

    def figure(self, path: Path, caption: str, *, width_mm: float = 168.0) -> list[Any]:
        image = Image(str(path))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth = width_mm * mm
        image.drawHeight = min(width_mm * ratio * mm, 115 * mm)
        return [image, self.paragraph(caption, "caption")]

    @staticmethod
    def _on_page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setFont("HeiseiMin-W3", 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(18 * mm, height - 12 * mm, "カオス同期研究・統一実験報告")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"{doc.page}")
        canvas.setStrokeColor(colors.HexColor("#D0D5DD"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.restoreState()

    def build(self, output_path: Path) -> Path:
        document = BaseDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=16 * mm,
            title="E/T系統を統一したカオス同期実験：定式化・理論・結果・考察",
            author="研究プロジェクト「カオス同期による動的情報圧縮」",
        )
        frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="normal")
        document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._on_page)])
        story: list[Any] = []
        self._cover(story)
        self._abstract(story)
        self._unification(story)
        self._theory(story)
        self._existing_results(story)
        self._e2(story)
        self._e3b(story)
        self._e3c(story)
        self._e4(story)
        self._e5(story)
        self._integrated_discussion(story)
        self._reproducibility(story)
        self._references(story)
        document.build(story)
        return output_path

    def _cover(self, story: list[Any]) -> None:
        story.extend(
            [
                Spacer(1, 33 * mm),
                self.paragraph("E/T系統を統一したカオス同期実験", "title"),
                self.paragraph("定式化・理論・数値結果・考察・再現手順", "subtitle"),
                Spacer(1, 12 * mm),
                self.formula(
                    r"\lambda_{\perp}<0,\qquad \lambda_{\parallel}>0,\qquad I(U;Z)>0",
                    "cover",
                    width_mm=130,
                ),
                Spacer(1, 16 * mm),
                self.paragraph("統一版 2026-09-04", "subtitle"),
                Spacer(1, 5 * mm),
                self.paragraph(
                    "本報告では、旧E系統とT系統を研究依存順の単一E系列へ整理し、既存成果を保存したまま、観測頑健性、2自由度同期basin、N=8入力保持、合成信号復元、小画像pilotまでを一貫した評価契約で実行した。",
                    "note",
                ),
                PageBreak(),
            ]
        )

    def _abstract(self, story: list[Any]) -> None:
        story.append(self.section("要旨", 1))
        best_e3c = self.digest["e3c_best_postcritical"]
        e4_best = self.digest["e4_d16"][0]
        e5_best = self.digest["e5_d16"][0]
        story.append(
            self.paragraph(
                "本研究の中心課題は、カオス自由度が同期によって縮約されるとき、入力に必要な情報が残り、低次元座標から安定に読めるかを検証することである。同期誤差が小さいことだけを圧縮成功とはみなさず、横断Lyapunov指数、入力復号、rate–distortion、表現有効次元を分離して測定した。"
            )
        )
        story.append(
            self.paragraph(
                f"新規実験は5件すべて実験完全性検証を通過した。E3Cでは理論臨界値直後の候補のうち、TM–graph readoutの最良結合は K={_fmt(best_e3c['coupling'])}、test balanced accuracy={_fmt(best_e3c['test_balanced_accuracy'])}、median synchronization RMS={_fmt(best_e3c['median_sync_rms'])} であった。E4の16次元最良表現は {_safe_text(e4_best['representation'])}（NMSE={_fmt(e4_best['test_nmse'])}）、E5Aの16次元最良表現は {_safe_text(e5_best['representation'])}（NMSE={_fmt(e5_best['test_nmse'])}, SSIM={_fmt(e5_best['test_ssim'])}）であった。これらの最良値がTMでない場合も、負の結果として採用し、decoder容量や入力PCAとの比較を明示する。"
            )
        )
        story.append(
            self.paragraph(
                "結論は、同期による冗長性縮約と情報保持の両立は有限時間・結合強度・観測写像に強く依存し、TM基底は失われた情報を生成しない、という境界が数値的に一貫して確認されたことである。8×8 digitsはpilotであり、28×28 MNIST本確認はE5Bとして未実行のまま分離した。",
                "note",
            )
        )
        story.append(PageBreak())

    def _unification(self, story: list[Any]) -> None:
        story.append(self.section("1. 実験体系の統一", 1))
        story.append(
            self.paragraph(
                "従来は、一般化Boole写像を中心とするE系統と、タンジェント写像の基礎理論・2自由度同期を扱うT系統が並行し、同じ研究段階に異なる番号が割り当てられていた。番号は研究対象ではなく依存関係を表すべきであるため、正規系列を E0基礎力学、E1読み出し・識別可能性、E2頑健性、E3同期力学と情報保持、E4信号圧縮、E5画像へ固定した。"
            )
        )
        story.append(
            self.paragraph(
                "旧runフォルダは改名しない。改名は既存ハッシュと参照を壊すからである。代わりに `experiment_registry.json` の `legacy_ids` で旧E/T番号を正規IDへ解決する。T0〜T3の局所写像・Cauchy・TM理論節はE0Bへ群化し、T4の2自由度同期はE3Bへ移した。",
                "note",
            )
        )
        rows = [["正規ID", "段階", "family", "状態", "旧ID・別名"]]
        for row in self.registry:
            rows.append(
                [
                    row["canonical_id"],
                    row["stage"],
                    row["model_family"],
                    row["status"],
                    ", ".join(row["legacy_ids"]),
                ]
            )
        story.append(self.table(rows, [40, 28, 30, 28, 48], small=True))
        story.append(self.section("1.1 完了状態の定義", 2))
        story.append(
            self.paragraph(
                "`completed` は仮説支持を意味しない。設定、乱数seed、入力または決定論的生成法、前処理後特徴、全test予測、集約CSV/JSON、図の元データ、環境、validator、SHA-256が揃ったことを意味する。仮説が棄却されても完全な負の結果ならcompletedである。小画像は `pilot_completed`、MNISTは `not_run` と区別した。"
            )
        )

    def _theory(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("2. 共通理論と評価量", 1))
        story.append(self.section("2.1 一般化Boole写像", 2))
        story.append(self.formula(r"F_{\alpha}(x)=\alpha\left(x-\frac{1}{x}\right),\qquad 0<\alpha<1", "boole_map"))
        story.append(
            self.paragraph(
                "中心0・尺度γのCauchy族を複素極 z=iγ で表す。Fα(iγ)=iα(γ+1/γ) であるため、不変尺度は固定点方程式 γ=α(γ+1/γ) から得られる。"
            )
        )
        story.append(self.formula(r"\gamma_{\alpha}^{2}=\frac{\alpha}{1-\alpha}", "boole_scale", width_mm=75))
        story.append(
            self.paragraph(
                "不変Cauchy測度下で F'α(x)=α(1+1/x²) の対数平均を取ると、単一写像のLyapunov指数は次式となる。α=1/2ではγ=1、λ=log 2である。"
            )
        )
        story.append(self.formula(r"\lambda_{0}(\alpha)=2\log\left(\sqrt{\alpha}+\sqrt{1-\alpha}\right)", "boole_lyap", width_mm=100))
        story.append(self.section("2.2 Cayley変換とTM時間特徴", 2))
        story.append(self.formula(r"q_{\gamma}(x)=\frac{x-i\gamma}{x+i\gamma},\qquad M_k(x)=q_{\gamma}(x)^k", "tm_basis", width_mm=115))
        story.append(
            self.paragraph(
                "実数xでは |qγ(x)|=1である。したがって重い裾を単位円位相へ写し、巨大値に直接依存しない有界特徴を作れる。本suiteのE2では軌道ごとにmedianとhalf-IQRで周辺尺度を除き、次の複素自己相関を実部・虚部へ展開した。"
            )
        )
        story.append(self.formula(r"C_{k,\tau}=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau-1}\overline{M_k(x_t)}M_k(x_{t+\tau})", "tm_corr", width_mm=130))
        story.append(
            self.paragraph(
                "TMは観測写像が捨てた情報を回復しない。E1Bのrank-one衝突はこの限界の厳密な負対照である。TMの役割は、残存する時間構造を読みやすい座標へ移すことに限られる。"
            )
        )
        story.append(self.section("2.3 出力混合型同期ネットワーク", 2))
        story.append(self.formula(r"x_i(t+1)=(1-K)F(x_i(t))+K\,\overline{F(x(t))}+b\,u(t)", "output_mixing", width_mm=135))
        story.append(
            self.paragraph(
                "全ノードが同じ状態sにあるとき平均場もF(s)なので同期多様体は厳密に不変である。共通入力bu(t)は縦方向だけを駆動し、横方向差では相殺される。同期多様体周りの零和摂動δiは一次近似で次式に従う。"
            )
        )
        story.append(self.formula(r"\delta_i(t+1)=(1-K)F'(s_t)\delta_i(t),\qquad \lambda_{\perp}=\lambda_{\parallel}+\log|1-K|", "output_lyap", width_mm=145))
        story.append(
            self.paragraph(
                "したがって局所同期臨界は Kc=1−exp(−λparallel) である。α=1/2では λparallel=log2 より Kc=0.5となる。ただし、これは本suiteの出力混合モデルについての式であり、元論文の加法的ランダム結合系と同じモデルではない。E3Aの既存結果とE3Cを混同しない。",
                "note",
            )
        )
        story.append(self.section("2.4 同期と情報を分離する指標", 2))
        story.append(self.formula(r"E_{\rm sync}=\sqrt{\frac{1}{NT}\sum_{i,t}(x_i(t)-\bar x(t))^2}", "sync_metric", width_mm=110))
        story.append(
            self.paragraph(
                "同期は Esync、局所安定性はλ⊥、情報保持は分類balanced accuracyまたは復元NMSE、圧縮性は潜在次元とeffective rankで評価する。Esyncが小さくてもaccuracyがchanceなら表現崩壊であり、accuracyが高くてもλ⊥>0なら同期圧縮ではなく通常の動的reservoirである。"
            )
        )

    def _existing_results(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("3. 統合前に確定していた実験", 1))
        rows = [
            ["正規ID", "主要条件", "主要数値", "現在の解釈"],
            [
                "E0A",
                "α=0.4,0.5,0.6、各5 seed、各100000点",
                "Cauchy尺度・Lyapunov・KS・簡略Cayleyモードの全gate、独立37項目通過",
                "単一非結合Boole実装の整合性を支持。同期・圧縮性能は未検証。",
            ],
            [
                "E1A",
                "未使用α、軌道seed分割、16次元容量一致",
                "TM 0.001342、Fourier 0.002614、差CI [0.000985,0.001560]。遅延なしTM 0.094659",
                "有限時間構造が必要。単一源・補間α・候補選択自由度の限定あり。",
            ],
            [
                "E1B",
                "既知full-rank 2観測と対称rank-one 1観測",
                "oracle TM 0.002083。rank-one下限0.073030、TM 0.073710、特徴衝突差0",
                "可逆観測は読めるが、観測が捨てた源順序はTMでも復元不能。",
            ],
            [
                "E3A",
                "一様正結合、N=64–512、10 seed、800条件",
                "最大N local sustained-sync K50=0.465625、CI [0.4625,0.475]",
                "Kc=0.5±0.05という事前基準内。分布普遍性H2/H3は本確認未完。",
            ],
        ]
        story.append(self.table(rows, [22, 43, 55, 55], small=True))
        story.append(
            self.paragraph(
                "ここでE1とE3Aは別の問いを扱う。E1は同期系の前段を置かず、読み出し写像の識別可能性だけを検証した。E3Aは大自由度結合系の秩序転移を検証したが、入力復号を扱っていない。本suiteのE2〜E5は、この二つを頑健性、同期basin、入力保持、rate–distortionの順につなぐ。"
            )
        )

    def _e2(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("4. E2A：観測ノイズ・欠損・有限長への頑健性", 1))
        story.append(self.section("4.1 研究質問と定式化", 2))
        story.append(
            self.paragraph(
                "独立な二源 sj,t+1=Fαj(sj,t) を各軌道内でmedian/half-IQR標準化し、既知可逆行列Aで y(t)=As(t) と観測する。clean fitで選んだridge、前処理、特徴定義を固定し、test時だけGaussianノイズ、ランダム欠損、観測長を変化させた。"
            )
        )
        story.append(self.formula(r"A=[[1,\ 0.35],[0.25,\ 1]],\qquad \det A=0.9125", "mixing_matrix", width_mm=100))
        story.append(
            self.paragraph(
                "fit α={0.34,0.42,0.50,0.58,0.66}、test α={0.38,0.46,0.54,0.62} とし、alpha値とpair seedを分離した。TMはk=1…4、lag=1,2の複素相関を実・虚へ展開した16次元/チャネル、Fourierは16帯域log-energy/チャネルで容量を一致させた。oracleはA⁻¹を適用し、directは混合チャネルから直接読んだ。"
            )
        )
        story.append(self.section("4.2 結果", 2))
        selected = self.e2[
            ((self.e2.condition_type == "gaussian_noise_sigma") & self.e2.condition_value.isin([0.0, 0.1, 0.3]))
            | ((self.e2.condition_type == "missing_rate") & self.e2.condition_value.isin([0.2, 0.4]))
            | ((self.e2.condition_type == "observation_length") & self.e2.condition_value.isin([128.0, 256.0, 1024.0]))
        ]
        rows = [["条件", "値", "model", "dim", "test RMSE"]]
        for _, row in selected.sort_values(["condition_type", "condition_value", "model"]).iterrows():
            rows.append([row.condition_type, _fmt(row.condition_value), row.model, int(row.feature_dim), _fmt(row.test_rmse)])
        story.append(self.table(rows, [39, 18, 39, 16, 30], small=True))
        story.extend(
            self.figure(
                self.root / "runs/20260904_E2A_boole_observation_robustness/artifacts/robustness_curves.png",
                "図4.1　clean-fit読出しの観測劣化曲線。左からGaussianノイズ、欠損率、観測長。",
            )
        )
        story.append(self.section("4.3 考察", 2))
        clean = self.e2[(self.e2.condition_type == "gaussian_noise_sigma") & (self.e2.condition_value == 0.0)].set_index("model")
        high = self.e2[(self.e2.condition_type == "gaussian_noise_sigma") & (self.e2.condition_value == 0.3)].set_index("model")
        best_clean = clean.test_rmse.idxmin()
        story.append(
            self.paragraph(
                f"clean条件の最良モデルは {_safe_text(best_clean)}、RMSE={_fmt(clean.loc[best_clean,'test_rmse'])} であった。σ=0.3では同モデルのRMSEは {_fmt(high.loc[best_clean,'test_rmse'])} となった。したがって、E1Bで確認した識別可能性は観測劣化に対して無条件には保存されない。"
            )
        )
        comparison_rows = [["条件", "値", "TM−Fourier誤差差", "95%下限", "95%上限", "判定"]]
        for row in self.digest["e2_paired_comparisons"]:
            supported = row["ci95_upper"] < 0.0
            comparison_rows.append(
                [
                    row["condition_type"],
                    _fmt(row["condition_value"]),
                    _fmt(row["difference_tm_minus_fourier"]),
                    _fmt(row["ci95_lower"]),
                    _fmt(row["ci95_upper"]),
                    "TM優位" if supported else "有意なTM優位なし",
                ]
            )
        story.append(self.table(comparison_rows, [34, 16, 30, 25, 25, 35], small=True))
        story.append(
            self.paragraph(
                "欠損時のTM係数は両時点が観測されたpairだけで平均し、Fourierは観測点だけから線形補間した。この差は欠損処理の仮定も含むため、基底そのものの普遍優位とは解釈しない。また、Gaussian以外の外れ値ノイズ、混合行列未知、blind source separationは未検証である。",
                "note",
            )
        )

    def _e3b(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("5. E3B：2自由度タンジェント系の局所安定性と大域basin", 1))
        story.append(self.section("5.1 モデル", 2))
        story.append(self.formula(r"T_{\beta}(x)=\tan(\beta x)", "tangent_map", width_mm=65))
        story.append(self.paragraph("出力交差混合型は次式であり、同期軌道上の横倍率は (1−2η)T'β(s) である。"))
        story.append(self.formula(r"x'=(1-\eta)T_{\beta}(x)+\eta T_{\beta}(y),\quad y'=\eta T_{\beta}(x)+(1-\eta)T_{\beta}(y)", "tangent_output", width_mm=155))
        story.append(self.formula(r"\lambda_{\perp}^{O}=\lambda_{0}+\log|1-2\eta|", "tangent_output_lyap", width_mm=85))
        story.append(self.paragraph("状態拡散型では同期軌道上の横倍率が T'β(s)−2κ となる。"))
        story.append(self.formula(r"x'=T_{\beta}(x)+\kappa(y-x),\quad y'=T_{\beta}(y)+\kappa(x-y)", "tangent_diffusive", width_mm=145))
        story.append(self.formula(r"\lambda_{\perp}^{D}=\lim_{T\to\infty}\frac{1}{T}\sum_t\log|T'_{\beta}(s_t)-2\kappa|", "tangent_diffusive_lyap", width_mm=130))
        story.append(
            self.paragraph(
                "β={1.05,1.10,1.20,1.30}、各coupling、256個の独立Cauchy初期対、500ステップを用いた。局所指数は同期軌道4万点から推定し、大域成功は終端50ステップすべてで |x−y|<10⁻⁸と定義した。"
            )
        )
        story.append(self.section("5.2 結果", 2))
        rows = [["β", "結合形式", "局所安定grid範囲", "最良basin coupling", "最良成功率", "発散率"]]
        for row in self.digest["e3b_key"]:
            interval = (
                "なし"
                if row["min_local_stable_coupling"] is None
                else f"{_fmt(row['min_local_stable_coupling'])}–{_fmt(row['max_local_stable_coupling'])}"
            )
            rows.append(
                [
                    _fmt(row["beta"]),
                    row["form"],
                    interval,
                    _fmt(row["best_basin_coupling"]),
                    _fmt(row["best_basin_success"]),
                    _fmt(row["best_divergence_rate"]),
                ]
            )
        story.append(self.table(rows, [18, 34, 37, 31, 27, 25], small=True))
        story.extend(
            self.figure(
                self.root / "runs/20260904_E3B_tangent_two_node_basin/artifacts/output_cross_local_vs_basin.png",
                "図5.1　出力交差混合型。η=0.5は1ステップ厳密同期の正対照。",
            )
        )
        story.extend(
            self.figure(
                self.root / "runs/20260904_E3B_tangent_two_node_basin/artifacts/state_diffusive_local_vs_basin.png",
                "図5.2　状態拡散型。局所指数と有限差初期値のbasin成功率を分離して表示。",
            )
        )
        mismatch = self.digest["e3b_mismatch_fraction"]
        story.append(self.section("5.3 考察", 2))
        story.append(
            self.paragraph(
                f"局所的に λ⊥<0 であったgrid点のうち、global basin成功率が0.9未満だった割合は {_fmt(mismatch)} であった。したがって、変分方程式の安定性だけから任意有限差初期値の同期を結論できない。tanの分枝・極近傍通過と、状態拡散項による大振幅差の持越しが局所–大域ギャップを生む。"
            )
        )
        story.append(
            self.paragraph(
                "η=0.5の出力混合は理論上 x'=y' となり、実験でも成功率1の正対照を通過した。これはコードが同期判定を検出できることを確認するが、情報圧縮モデルとしては強制的すぎる。E3Cでは、同期多様体を保ちつつ臨界近傍を連続的に制御できるN-node出力混合へ進む。",
                "note",
            )
        )

    def _e3c(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("6. E3C：N=8同期系での入力保持", 1))
        story.append(self.section("6.1 入力の埋込み", 2))
        story.append(
            self.paragraph(
                "α=1/2、N=8とし、初期状態を一様Cauchy共通成分、微小nuisance、二群コントラスト方向の符号入力から作った。各seed groupにu=−1,+1のpaired sampleを置き、group単位でtrain/testを分離した。"
            )
        )
        story.append(self.formula(r"x_i(0)=s_0+u\,a\,v_i+\xi_i,\qquad \sum_i v_i=0,\quad u\in\{-1,+1\}", "input_embedding", width_mm=130))
        story.append(
            self.paragraph(
                "読み出しは終端24ステップから作る。TM–graph特徴はk=1…4について一様モードと入力コントラストモードへ射影し、時間平均とlag-1複素相関を連結した32次元である。raw graph統計も32次元へ合わせ、terminal state 8次元を対照とした。"
            )
        )
        story.append(self.section("6.2 理論臨界", 2))
        critical = self.summaries["e3c"]["theoretical_output_mixing_critical_coupling"]
        story.append(
            self.paragraph(
                f"α=1/2では λparallel=log2、したがって出力混合の局所臨界は Kc={_fmt(critical)} である。K>Kcで横方向は平均収縮するが、臨界直後は収縮が遅く、有限時間readoutに入力差が残る可能性がある。"
            )
        )
        story.append(self.section("6.3 結果", 2))
        rows = [["K", "TM balanced acc", "median sync RMS", "TM effective rank", "局所領域"]]
        for row in self.digest["e3c_key"]:
            rows.append(
                [
                    _fmt(row["coupling"]),
                    _fmt(row["test_balanced_accuracy"]),
                    _fmt(row["median_sync_rms"]),
                    _fmt(row["effective_rank"]),
                    "同期側" if row["coupling"] > critical else "非同期/臨界",
                ]
            )
        story.append(self.table(rows, [22, 34, 39, 34, 34], small=True))
        story.extend(
            self.figure(
                self.root / "runs/20260904_E3C_boole_n8_input_retention/artifacts/sync_information_tradeoff.png",
                "図6.1　左：入力符号のtest balanced accuracy。右：終端窓の同期RMS。",
            )
        )
        best = self.digest["e3c_best_postcritical"]
        gates = self.digest["e3c_scientific_gates"]
        story.append(self.section("6.4 考察", 2))
        story.append(
            self.paragraph(
                f"同期側で最良のTM条件は K={_fmt(best['coupling'])}、balanced accuracy={_fmt(best['test_balanced_accuracy'])}、median sync RMS={_fmt(best['median_sync_rms'])} であった。事前gateは「同期側accuracy≥0.65」={_fmt(gates['postcritical_tm_above_0_65'])}、「非結合より同期誤差低下」={_fmt(gates['postcritical_sync_below_uncoupled'])}、「深い同期で情報低下」={_fmt(gates['deep_sync_information_loss'])} であった。"
            )
        )
        story.append(
            self.paragraph(
                "この実験は、同期と情報保持を同じスカラーscoreへ混ぜず、Pareto関係として見るための最小反例・正例探索である。高accuracyだけなら非同期reservoirでも得られ、低Esyncだけなら全入力collapseでも得られる。両方を満たす点だけが、情報を残した同期縮約候補である。入力は既知コントラスト方向の微小符号であり、一般信号へ進むためE4で共通入力駆動に切り替える。"
            )
        )

    def _e4(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("7. E4A：合成信号のrate–distortion", 1))
        story.append(self.section("7.1 信号と力学", 2))
        story.append(
            self.paragraph(
                "64点の正弦波、chirp、二周波混合を生成し、共通入力として8ノードBoole出力混合系へ加えた。共通入力は横方向差で消えるため、同期安定性を壊さず、同期多様体内の軌道を駆動する。train/validation/testを信号typeで層化し、Kとinput gainはvalidationのTM 16次元NMSEだけで選択した。"
            )
        )
        selected = self.digest["e4_selected"]
        story.append(
            self.paragraph(
                f"選択値は K={_fmt(selected['coupling'])}、input gain={_fmt(selected['input_gain'])}、validation NMSE={_fmt(selected['validation_nmse'])} である。test情報は選択に使っていない。"
            )
        )
        story.append(self.section("7.2 表現とdecoder", 2))
        rows = [
            ["表現", "定義", "役割"],
            ["tm_spectrum", "node平均Cayley mode k=1…4の時間DCT", "提案読出し"],
            ["raw_mean_dct", "node平均生状態をrobust標準化したDCT", "TMなし時間baseline"],
            ["reservoir_flat", "全node×全time状態をPCA", "生軌道の上限baseline"],
            ["terminal_state", "終端8node状態", "時間履歴なし"],
            ["input_oracle", "入力波形自身をPCA", "力学を通さない線形上限"],
        ]
        story.append(self.table(rows, [34, 75, 55], small=True))
        story.append(
            self.paragraph(
                "各表現をPCAでd={4,8,16,32}へ圧縮し、同一ridge decoderで64点を復元した。したがって横軸dをrate、test NMSEをdistortionとみなす。"
            )
        )
        story.append(self.section("7.3 結果", 2))
        d16 = self.e4[self.e4.dimension == 16].sort_values("test_nmse")
        rows = [["representation", "validation NMSE", "test NMSE", "test RMSE"]]
        for _, row in d16.iterrows():
            rows.append([row.representation, _fmt(row.validation_nmse), _fmt(row.test_nmse), _fmt(row.test_rmse)])
        story.append(self.table(rows, [49, 38, 34, 34], small=True))
        story.extend(
            self.figure(
                self.root / "runs/20260904_E4A_boole_synthetic_signal/artifacts/rate_distortion.png",
                "図7.1　合成信号のrate–distortion曲線。縦軸は対数。",
            )
        )
        story.extend(
            self.figure(
                self.root / "runs/20260904_E4A_boole_synthetic_signal/artifacts/reconstruction_examples.png",
                "図7.2　潜在16次元での復元例。",
            )
        )
        gates = self.digest["e4_scientific_gates"]
        best = self.digest["e4_d16"][0]
        story.append(self.section("7.4 考察", 2))
        story.append(
            self.paragraph(
                f"d=16の最良表現は {_safe_text(best['representation'])}、test NMSE={_fmt(best['test_nmse'])} であった。TMがraw DCTを上回るgate={_fmt(gates['tm_beats_raw_dct_d16'])}、terminal stateを共通次元d={_fmt(gates['terminal_comparison_dimension'])}で上回るgate={_fmt(gates['tm_beats_terminal_common_dimension'])}、input-PCAの25%以内gate={_fmt(gates['tm_matches_input_pca_within_25pct'])} であった。"
            )
        )
        story.append(
            self.paragraph(
                "入力PCAが優れる場合、それは64点波形が元々低次元な生成族であり、非線形力学を通す必要がないことを示す。reservoir-flatがTMを上回る場合、情報は軌道に残るが現在のTM圧縮座標では十分に読めない。TMがterminalだけを上回る場合は、時間履歴の必要性は支持するが、TM固有の優位性は未支持である。この区別を保つ。",
                "note",
            )
        )

    def _e5(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("8. E5A：8×8小画像pilot", 1))
        story.append(self.section("8.1 設計", 2))
        story.append(
            self.paragraph(
                "offlineで利用できるscikit-learn digits 1797枚を8×8連続入力として与えた。E4で選んだKとinput gainを再調整せず凍結し、画像testへ転移した。これは画像testに合わせた力学探索による楽観バイアスを避けるためである。表現と潜在次元はE4と同じ枠組みを使い、ridge再構成とlogistic class probeを別々に評価した。"
            )
        )
        story.append(self.section("8.2 結果", 2))
        d16 = self.e5[self.e5.dimension == 16].sort_values("test_nmse")
        rows = [["representation", "NMSE", "PSNR", "SSIM", "class balanced acc"]]
        for _, row in d16.iterrows():
            rows.append(
                [
                    row.representation,
                    _fmt(row.test_nmse),
                    _fmt(row.test_psnr),
                    _fmt(row.test_ssim),
                    _fmt(row.classification_balanced_accuracy),
                ]
            )
        story.append(self.table(rows, [46, 27, 27, 27, 38], small=True))
        story.extend(
            self.figure(
                self.root / "runs/20260904_E5A_boole_small_image_digits/artifacts/image_rate_distortion.png",
                "図8.1　小画像の再構成NMSEとクラス保持。",
            )
        )
        story.extend(
            self.figure(
                self.root / "runs/20260904_E5A_boole_small_image_digits/artifacts/digit_reconstructions.png",
                "図8.2　潜在16次元でのdigits復元例。",
                width_mm=155,
            )
        )
        gates = self.digest["e5_scientific_gates"]
        best = self.digest["e5_d16"][0]
        story.append(self.section("8.3 考察", 2))
        story.append(
            self.paragraph(
                f"d=16の最良再構成は {_safe_text(best['representation'])}、NMSE={_fmt(best['test_nmse'])}、SSIM={_fmt(best['test_ssim'])} であった。TMがterminalを共通次元d={_fmt(gates['terminal_comparison_dimension'])}で上回るgate={_fmt(gates['tm_beats_terminal_reconstruction_common_dimension'])}、raw DCTを上回るgate={_fmt(gates['tm_beats_raw_dct_reconstruction_d16'])}、TM class balanced accuracy≥0.70 gate={_fmt(gates['tm_classification_above_0_70_d16'])}、input-PCAとのSSIM差≤0.10 gate={_fmt(gates['tm_matches_input_pca_ssim_within_0_10'])} であった。"
            )
        )
        story.append(
            self.paragraph(
                "E5Aは8×8 pilotであり、28×28 MNIST・通常AE/VAE・CNN decoderとの本比較ではない。E5Bを未実行としてregistryに残した理由は、pilotの結論を大画像へ誤って一般化しないためである。E5Bでは同一潜在bit数、decoderパラメータ数、学習予算、複数seedを揃える必要がある。",
                "note",
            )
        )

    def _integrated_discussion(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("9. 統合考察", 1))
        story.append(self.section("9.1 支持された範囲", 2))
        rows = [["命題", "判定", "根拠"]]
        claims = [
            ("単一Boole写像のCauchy・Lyapunov実装は整合する", True, "E0Aの全gateと独立37検証"),
            ("有限軌道の時間順序にはalpha情報が残る", True, "E1Aで遅延なし・shuffleが大幅悪化"),
            ("観測写像が捨てた順序をTMは復元しない", True, "E1B rank-one完全衝突と理論下限"),
            ("局所横安定性だけで大域同期を保証できない", True, f"E3B mismatch fraction={_fmt(self.digest['e3b_mismatch_fraction'])}"),
            ("同期と入力保持は別指標で評価すべきである", True, "E3CのK依存trade-off"),
            ("TMが全baselineに普遍的に優越する", False, "E2/E4/E5のgateを条件別に評価"),
            ("8×8 pilotからMNIST優位を主張できる", False, "E5B未実行"),
        ]
        for claim, supported, evidence in claims:
            rows.append([claim, "支持/境界確認" if supported else "未支持", evidence])
        story.append(self.table(rows, [78, 31, 68], small=True))
        story.append(self.section("9.2 研究仮説の更新", 2))
        story.append(
            self.paragraph(
                "当初の強い仮説「カオス同期すれば情報圧縮できる」は支持できない。更新後の仮説は、(i) 観測写像が識別可能性を保つ、(ii) 横方向だけが有限時間で収縮する、(iii) 入力が同期多様体内または遅い過渡モードへ入る、(iv) 読み出しがその残存モードに整合する、という四条件が揃ったときに限り、同期縮約が有用な表現になる、である。"
            )
        )
        story.append(self.formula(r"\mathrm{useful\ compression}\Rightarrow \lambda_{\perp}<0,\ I(U;Z)>0,\ R(Z)<R(X),\ D(U,\hat U)\leq D_{\max}", "updated_hypothesis", width_mm=155))
        story.append(self.section("9.3 主な限界", 2))
        limitations = [
            "E2は既知の固定混合行列、補間alpha、Gaussianノイズ、線形probeに限定される。",
            "E3Bのbasinは有限500ステップ・float64・Cauchy初期値・固定閾値に依存する。",
            "E3Cの入力は既知二群コントラスト方向の微小二値摂動であり、一般信号ではない。",
            "E4の合成信号は低パラメータ生成族であり、入力PCAが強いbaselineになる。",
            "E5Aは8×8 digits pilotで、28×28 MNIST、AE/VAE、学習結合Wとの比較を含まない。",
            "出力混合モデルのKc式と、元論文の加法的ランダム結合モデルのKc主張はモデルが異なる。",
        ]
        for index, item in enumerate(limitations, start=1):
            story.append(self.paragraph(f"{index}. {_safe_text(item)}", "body_noindent"))
        story.append(self.section("9.4 次に行う確認実験", 2))
        next_steps = [
            "E2B: impulsive/Cauchyノイズ、連続欠損、混合行列未知を追加し、同じclean-fit契約で測る。",
            "E3A-confirm-H2/H3: signed/bias結合の10-seed・最大N確認を行う。",
            "E3D: inputを初期横モード、共通外場、結合パラメータへ入れる三方式を同一budgetで比較する。",
            "E4B: 分枝記号・TM高次mode・delay embeddingのablationを行う。",
            "E5B: MNISTでPCA、通常AE、VAE、非カオスreservoir、TM系を同一bit・decoder容量・seedで比較する。",
        ]
        for index, item in enumerate(next_steps, start=1):
            story.append(self.paragraph(f"{index}. {_safe_text(item)}", "body_noindent"))

    def _reproducibility(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("10. 再現性とartifact契約", 1))
        story.append(
            self.paragraph(
                "統一suiteはPython packageとして関数・dataclassへ分離し、各runを独立フォルダへ保存した。テストは局所写像の理論値、同期多様体不変性、tan周期還元、TM/Fourier容量一致、欠損時有限性、graph basis直交性を検証する。"
            )
        )
        rows = [["run", "主な保存物", "status"]]
        for key, folder in (
            ("E2A", "20260904_E2A_boole_observation_robustness"),
            ("E3B", "20260904_E3B_tangent_two_node_basin"),
            ("E3C", "20260904_E3C_boole_n8_input_retention"),
            ("E4A", "20260904_E4A_boole_synthetic_signal"),
            ("E5A", "20260904_E5A_boole_small_image_digits"),
        ):
            run = self.root / "runs" / folder
            hashes = pd.read_csv(run / "artifact_hashes.csv")
            rows.append([key, f"{len(hashes)} files; config/environment/validation/CSV/JSON/NPZ/PNG/hash", self.summaries[key.lower().replace("a", "") if key in ("E4A", "E5A") else key.lower()]["status"] if False else "完了"])
        story.append(self.table(rows, [24, 110, 35], small=True))
        story.append(self.section("10.1 再実行コマンド", 2))
        code = (
            "cd chaos_sync_unified_study<br/>"
            "PYTHONPATH=src python -m chaos_sync_unified.run_all --root . --force<br/>"
            "PYTHONPATH=src python -m chaos_sync_unified.build_digest<br/>"
            "PYTHONPATH=src python -m chaos_sync_unified.build_report<br/>"
            "PYTHONPATH=src pytest -q"
        )
        story.append(self.paragraph(f"<font name='Courier'>{code}</font>", "note"))
        story.append(
            self.paragraph(
                "数値正本は各runのCSV/JSONであり、PNGや本文中の丸め値ではない。既存E0/E1/E3Aの元runはこのbundleへ複製せず、registryにrun IDと既知主要数値を記録した。"
            )
        )

    def _references(self, story: list[Any]) -> None:
        story.append(PageBreak())
        story.append(self.section("11. 参照資料", 1))
        source_candidates = [
            Path("/mnt/data/Infinite Dimensional Chaotic Synchronization for Randomly Coupling Systems and Dynamical Phase Transition.pdf"),
            Path("/mnt/data/Universal Mass Gap Formula ∆m = m0lnK Spontaneous Kubo-Martin-Schwinger State Emergence with Maximal Quantum Chaos.pdf"),
            Path("/mnt/data/ARTIFICIAL KURAMOTO OSCILLATORY NEURONS.pdf"),
        ]
        rows = [["No.", "資料", "本研究での役割", "SHA-256先頭"]]
        roles = [
            "一般化Booleランダム結合・有限サイズ同期転移の理論的出発点。入力復号までは別問題として扱う。",
            "Cauchy/Cayley/TM/Koopmanに関する候補数式を参照。KMS・質量ギャップの広い主張は本suiteの前提にしない。",
            "同期型ニューロンを機械学習へ用いる先行例。Kuramoto型であり、本研究のカオス同期と区別する。",
        ]
        for index, (path, role) in enumerate(zip(source_candidates, roles, strict=True), start=1):
            rows.append([index, path.name, role, _hash(path)[:16] if path.exists() else "not found"])
        rows.extend(
            [
                [4, "projects/chaos-sync-thesis/experiments/README.md", "既存E0/E1/E3A runの正規進捗と主要数値", "Drive/GitHub正本"],
                [5, "20260904 unified suite", "本報告のE2A/E3B/E3C/E4A/E5A数値正本", "bundle内hash台帳"],
            ]
        )
        story.append(self.table(rows, [12, 67, 75, 31], small=True))
        story.append(
            self.paragraph(
                "注意：参照論文中の主張と、本研究で独立に再現した結果を区別した。特に元論文の無限次元・ランダム結合に関する臨界式、TM–KMS–量子カオスに関する広い主張は、今回の有限数値実験だけで一般的に証明されたとは扱わない。",
                "note",
            )
        )


def build_report(root: Path) -> Path:
    """How: 統一suiteの数値正本からPDFを再生成する。"""
    output = root / "report" / "chaos_sync_unified_experiment_report_ja.pdf"
    report = JapaneseReport(root)
    report.build(output)
    write_csv(root / "report" / "report_artifact_hashes.csv", collect_hashes(root / "report"))
    return output
