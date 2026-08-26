"""E0の保存済み結果から読者向けNotebookを作成し、上から順に実行する。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import nbformat
from nbclient import NotebookClient
from nbformat.notebooknode import NotebookNode


def format_per_alpha_table(per_alpha: List[Dict[str, object]]) -> str:
    """主要結果をMarkdown表へ整形する。"""

    lines = [
        "| alpha | 理論scale | scale最大相対誤差 | Lyapunov最大絶対誤差 | KS最大値 | TM非零最大値 | 判定 |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in per_alpha:
        lines.append(
            "| {alpha:.1f} | {scale:.6f} | {scale_error:.6f} | {lyapunov_error:.6f} | "
            "{ks:.6f} | {tm:.6f} | {passed} |".format(
                alpha=float(row["alpha"]),
                scale=float(row["theoretical_scale"]),
                scale_error=float(row["max_scale_relative_error"]),
                lyapunov_error=float(row["max_lyapunov_absolute_error"]),
                ks=float(row["max_ks_distance"]),
                tm=float(row["max_tm_nonzero_magnitude"]),
                passed="PASS" if bool(row["passed"]) else "FAIL",
            )
        )
    return "\n".join(lines)


def build_notebook(run_directory: Path, summary: Dict[str, object]) -> NotebookNode:
    """分析レポート形式のNotebookを、保存済み結果に基づいて構築する。"""

    per_alpha = summary["per_alpha"]
    assert isinstance(per_alpha, list)
    result_table = format_per_alpha_table(per_alpha)
    max_scale_error = max(float(row["max_scale_relative_error"]) for row in per_alpha)
    max_lyapunov_error = max(float(row["max_lyapunov_absolute_error"]) for row in per_alpha)
    max_ks_distance = max(float(row["max_ks_distance"]) for row in per_alpha)
    max_tm_magnitude = max(float(row["max_tm_nonzero_magnitude"]) for row in per_alpha)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Research Python 3.12",
        "language": "python",
        "name": "research-python",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# E0: 一般化Boole写像の理論整合性確認\n\n"
            "## tl;dr\n\n"
            f"`alpha = 0.4, 0.5, 0.6`、各5 seed、各10万観測点で検証し、総合判定は "
            f"**{'PASS' if bool(summary['overall_passed']) else 'FAIL'}** だった。"
            f"尺度の最大相対誤差は `{max_scale_error:.6f}`、Lyapunov指数の最大絶対誤差は "
            f"`{max_lyapunov_error:.6f}`、KS距離最大値は `{max_ks_distance:.6f}`、"
            f"非零TMモードの最大絶対値は `{max_tm_magnitude:.6f}` だった。\n\n"
            "この結果は、一般化Boole写像をCauchy可解モデルとして実装するE0基盤を支持する。"
            "ただし、結合系の同期、情報保持、一般のTM基底の有効性はまだ検証していない。"
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### 研究質問\n\n"
            "一般化Boole写像\n\n"
            "$$F_\\alpha(x)=\\alpha\\left(x-\\frac{1}{x}\\right),\\qquad 0<\\alpha<1$$\n\n"
            "について、次の理論値を有限精度・有限長軌道で再現できるかを確認する。\n\n"
            "$$\\gamma_\\alpha=\\sqrt{\\frac{\\alpha}{1-\\alpha}},\\qquad"
            "\\lambda(\\alpha)=2\\log(\\sqrt{\\alpha}+\\sqrt{1-\\alpha})$$\n\n"
            "Cayleyモードは\n\n"
            "$$q_\\gamma(x)=\\frac{x-i\\gamma}{x+i\\gamma},\\qquad M_k(x)=q_\\gamma(x)^k$$\n\n"
            "とし、中心0・尺度$\\gamma$のCauchy測度では非零次数の期待値が0になることを検証する。\n\n"
            "### Key Assumptions\n\n"
            "- 初期値は理論Cauchy測度から生成する。\n"
            "- 四分位幅からCauchy尺度を推定し、平均・分散は使わない。\n"
            "- KS距離は理論尺度に対して計算する。\n"
            "- TM検証は簡略化したCayleyモードに限定する。\n"
            "- 合格閾値は統計的有意水準ではなく、実装回帰を検出する事前定義ゲートである。\n\n"
            "### Source\n\n"
            "実験コード: `e0_boole_validation.py`  \n"
            "原典: `../../../references/papers/infinite-dimensional-chaotic-synchronization.pdf`"
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. 保存済み実験結果を読み込む"),
        nbformat.v4.new_code_cell(
            "from __future__ import annotations\n\n"
            "import csv\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "RUN_DIRECTORY = Path.cwd()\n"
            "ARTIFACTS_DIRECTORY = RUN_DIRECTORY / 'artifacts'\n"
            "summary = json.loads((ARTIFACTS_DIRECTORY / 'summary.json').read_text(encoding='utf-8'))\n"
            "with (ARTIFACTS_DIRECTORY / 'per_seed_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:\n"
            "    per_seed_rows = list(csv.DictReader(stream))\n\n"
            "print(f\"alpha数: {len(summary['per_alpha'])}\")\n"
            "print(f\"seed別行数: {len(per_seed_rows)}\")\n"
            "print(f\"総合判定: {summary['overall_passed']}\")"
        ),
        nbformat.v4.new_markdown_cell(
            "### 2. 入力件数と完全性を確認する\n\n"
            "期待する粒度は `3 alpha × 5 seed = 15行`。各行は一つの決定論的軌道を表す。"
        ),
        nbformat.v4.new_code_cell(
            "expected_row_count = len(summary['config']['alpha_values']) * len(summary['config']['seeds'])\n"
            "assert len(per_seed_rows) == expected_row_count\n"
            "assert all(row['finite'].lower() == 'true' for row in per_seed_rows)\n"
            "assert len({(row['alpha'], row['seed']) for row in per_seed_rows}) == expected_row_count\n"
            "print({'expected_rows': expected_row_count, 'actual_rows': len(per_seed_rows), 'finite': True, 'duplicates': 0})"
        ),
        nbformat.v4.new_markdown_cell("## Results\n\n### 3. alpha別の主要指標"),
        nbformat.v4.new_markdown_cell(result_table),
        nbformat.v4.new_code_cell(
            "for row in summary['per_alpha']:\n"
            "    failed_gates = [name for name, passed in row['gates'].items() if not passed]\n"
            "    print(f\"alpha={row['alpha']:.1f}: passed={row['passed']}, failed_gates={failed_gates}\")"
        ),
        nbformat.v4.new_markdown_cell(
            "### 4. seed別誤差を可視化する\n\n"
            "**Analytical question:** 4つのseed別指標は、すべて事前定義ゲートを下回るか。  \n"
            "**Takeaway:** 15軌道×4指標の全点がゲートを下回った。  \n"
            "破線は事前定義した合格上限。対数軸を使い、小さい誤差の差を読みやすくする。"
        ),
        nbformat.v4.new_code_cell(
            "numeric_fields = ['scale_relative_error', 'lyapunov_absolute_error', 'ks_distance', 'tm_max_nonzero_magnitude']\n"
            "metric_titles = ['Cauchy scale relative error', 'Lyapunov absolute error', 'KS distance', 'Max nonzero TM magnitude']\n"
            "thresholds = [\n"
            "    summary['config']['thresholds']['max_scale_relative_error'],\n"
            "    summary['config']['thresholds']['max_lyapunov_absolute_error'],\n"
            "    summary['config']['thresholds']['max_ks_distance'],\n"
            "    summary['config']['thresholds']['max_tm_nonzero_magnitude'],\n"
            "]\n"
            "alphas = sorted({float(row['alpha']) for row in per_seed_rows})\n"
            "palette = {0.4: '#3B82F6', 0.5: '#E28A2B', 0.6: '#758E4F'}\n"
            "fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)\n"
            "for axis, field, title, threshold in zip(axes.flat, numeric_fields, metric_titles, thresholds):\n"
            "    for alpha in alphas:\n"
            "        values = [float(row[field]) for row in per_seed_rows if float(row['alpha']) == alpha]\n"
            "        axis.scatter([alpha] * len(values), values, label=f'alpha={alpha:.1f}', color=palette[alpha], alpha=0.85)\n"
            "    axis.axhline(threshold, color='#374151', linestyle='--', linewidth=1.4, label='pre-set gate')\n"
            "    axis.set_yscale('log')\n"
            "    axis.set_title(title)\n"
            "    axis.set_xlabel('alpha')\n"
            "    axis.set_xticks(alphas)\n"
            "    axis.grid(True, which='both', alpha=0.25)\n"
            "fig.suptitle('E0 validation metrics by alpha and seed\\n5 seeds per alpha; 100,000 observations per trajectory; dashed line = pre-set gate')\n"
            "handles, labels = axes.flat[0].get_legend_handles_labels()\n"
            "fig.legend(handles, labels, loc='outside lower center', ncol=4)\n"
            "fig.savefig(ARTIFACTS_DIRECTORY / 'e0_metric_gates.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 5. 理論Lyapunov指数を独立な数値積分で照合する\n\n"
            "Cauchy変数を角度座標へ変換した等確率積分を用い、軌道平均とは別経路で閉形式を確認する。"
        ),
        nbformat.v4.new_code_cell(
            "quadrature_rows = [\n"
            "    {\n"
            "        'alpha': row['alpha'],\n"
            "        'closed_form': row['theoretical_lyapunov'],\n"
            "        'quadrature': row['quadrature_lyapunov'],\n"
            "        'absolute_error': row['quadrature_lyapunov_error'],\n"
            "    }\n"
            "    for row in summary['per_alpha']\n"
            "]\n"
            "quadrature_rows"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. 検証した3つの`alpha`では、有限軌道の経験分布が理論Cauchy測度と事前ゲート内で整合した。\n"
            "2. 軌道平均と独立な角度積分の両方が、閉形式Lyapunov指数と整合した。\n"
            "3. 非零Cayleyモードの平均は0近傍となり、E1AでTM特徴を実装するための最小チェックを通過した。\n"
            "4. これは各ノードへ一般化Boole写像を置く数学的実装基盤を支持するが、複数ノードを結合する有用性は示していない。\n\n"
            "### Caveats\n\n"
            "- 簡略Cayleyモードの検証であり、一般のTakenaka–Malmquist基底全体の完全性や読み出し性能を保証しない。\n"
            "- 初期値を不変測度から生成している。非定常初期値からの収束速度は別実験が必要である。\n"
            "- 観測区間に厳密な浮動小数点重複はなかったが、より長時間の周期化を否定しない。\n"
            "- 結合系の同期臨界式と入力情報保持は未検証である。\n\n"
            "### Next Step\n\n"
            "E1Aでは、独立な一般化Boole軌道の線形混合を作り、raw・自己相関・Fourier・ICAとTM特徴を比較する。"
        ),
    ]
    return notebook


def execute_notebook(notebook: NotebookNode, run_directory: Path) -> NotebookNode:
    """研究専用kernelでNotebookを上から順に実行する。"""

    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="research-python",
        resources={"metadata": {"path": str(run_directory)}},
        allow_errors=False,
    )
    return client.execute()


def main() -> int:
    """summaryを読み、Notebookを構築・実行・保存する。"""

    run_directory = Path(__file__).resolve().parent
    summary = json.loads(
        (run_directory / "artifacts" / "summary.json").read_text(encoding="utf-8")
    )
    notebook = build_notebook(run_directory, summary)
    executed_notebook = execute_notebook(notebook, run_directory)
    output_path = run_directory / "E0_boole_validation.ipynb"
    nbformat.write(executed_notebook, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
