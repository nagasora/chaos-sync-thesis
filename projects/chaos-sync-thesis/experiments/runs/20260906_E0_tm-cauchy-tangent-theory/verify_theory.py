"""TM型座標の理論予測を独立な実数写像と有限軌道で照合する。"""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent

def cayley(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """実状態を指定尺度のCayley座標へ写す。"""
    return (x - 1j * scale) / (x + 1j * scale)

def dictionary(q: np.ndarray, degree: int) -> np.ndarray:
    """共通極の実直交辞書を次数順に返す。"""
    powers = q[:, None] ** np.arange(1, degree + 1)
    return np.stack([powers.real, powers.imag], axis=-1).reshape(len(q), -1) * np.sqrt(2)

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """条件別結果を丸めずに保存する。"""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_projection() -> Dict[str, object]:
    """異なる求積点で数値最小二乗射影を適合・検証する。"""
    rows: List[Dict[str, object]] = []
    for m, omega in [(2, -np.pi/2), (3, 0.17)]:
        x = np.tan(-np.pi/2 + np.pi*(np.arange(8192)+0.5)/8192)
        y = np.tan(-np.pi/2 + np.pi*(np.arange(16384)+0.37)/16384)
        train_phi, test_phi = dictionary(cayley(x), 8), dictionary(cayley(y), 8)
        for horizon in range(1, 6):
            x = np.tan(m*np.arctan(x)+omega)
            y = np.tan(m*np.arctan(y)+omega)
            coef = np.linalg.lstsq(train_phi, dictionary(cayley(x), 8), rcond=None)[0]
            nmse = float(np.mean((dictionary(cayley(y), 8)-test_phi@coef)**2))
            theory = 1-(8//m**horizon)/8
            rows.append(dict(m=m, horizon=horizon, nmse=nmse, theory=theory, error=abs(nmse-theory)))
    result: Dict[str, object] = dict(method="8192-point fit / shifted 16384-point test",
        max_error=max(row["error"] for row in rows), rows=rows)
    if result["max_error"] >= 1e-8:
        raise AssertionError("独立数値射影が理論値に一致しません")
    result["status"] = "passed"
    return result

def main() -> None:
    """事前設定の全条件を実行し、失敗も含めて数値と判定を保存する。"""
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    n, degree = cfg["quadrature_points"], cfg["degree"]
    theta = -np.pi / 2 + np.pi * (np.arange(n) + 0.5) / n
    x = np.tan(theta)
    q = cayley(x)
    phi = dictionary(q, degree)
    checks: List[Dict[str, object]] = []
    def check(name: str, error: float, tolerance: float) -> None:
        """失敗を握り潰さず、事前閾値に対する照合値を記録する。"""
        checks.append(dict(name=name, error=float(error), tolerance=tolerance,
                           passed=bool(np.isfinite(error) and error <= tolerance)))
    gram_rows: List[Dict[str, object]] = []
    for scale in cfg["reference_scales"]:
        ph = dictionary(cayley(x, scale), degree)
        gram = ph.T @ ph / n
        err = float(np.linalg.norm(gram - np.eye(2 * degree)))
        gram_rows.append(dict(scale=scale, gram_error=err,
                              condition_number=float(np.linalg.cond(gram)),
                              mean_mode_real=float(cayley(x, scale).mean().real),
                              mean_mode_theory=(1-scale)/(1+scale)))
        check("scale_moment_" + str(scale),
              abs(cayley(x, scale).mean() - (1-scale)/(1+scale)), 1e-10)
        if scale == 1:
            check("matched_gram", err, 1e-10)
    write_csv(out / "gram.csv", gram_rows)
    correlation_rows: List[Dict[str, object]] = []
    for alpha in cfg["alphas"]:
        r = 2 * alpha - 1
        evolved = x.copy()
        for lag in cfg["lags"]:
            # 角度写像の同じ式を左右で比較せず、実数Boole写像を進める。
            evolved = alpha * evolved - (1-alpha) / evolved
            qh = cayley(evolved)
            for k in cfg["correlation_degrees"]:
                observed = np.mean(qh**k * np.conjugate(q**k))
                expected = r ** (lag * k)
                error = abs(observed-expected)
                correlation_rows.append(dict(alpha=alpha, lag=lag, degree=k,
                    theory=expected, observed_real=float(observed.real),
                    observed_imag=float(observed.imag), error=float(error)))
                check(f"correlation_{alpha}_{lag}_{k}", error, 1e-9)
    write_csv(out / "correlation_quadrature.csv", correlation_rows)
    operator_rows: List[Dict[str, object]] = []
    for m, omega in cfg["tangent_maps"]:
        evolved = x.copy()
        chi = (-1)**(m+1) * np.exp(2j*omega)
        for horizon in range(1, cfg["horizon"]+1):
            evolved = np.tan(m*np.arctan(evolved)+omega)
            actual_q = cayley(evolved)
            chi_h = chi ** ((m**horizon-1)//(m-1))
            analytic_q = chi_h * q**(m**horizon)
            check(f"tangent_conjugacy_{m}_{omega}_{horizon}",
                  np.max(abs(actual_q-analytic_q)), 1e-8)
            targets = dictionary(actual_q, degree)
            # 高次数の真の将来値を実数写像から計算し、辞書内の直交射影だけで予測する。
            predictions = np.zeros_like(targets)
            retained = degree // m**horizon
            if retained:
                predictions[:, :2*retained] = dictionary(analytic_q, retained)
            nmse = float(np.sum((targets-predictions)**2) / np.sum(targets**2))
            theoretical_nmse = 1-retained/degree
            q1_nmse = float(np.mean(abs(actual_q - (analytic_q if retained else 0))**2))
            rank = int(np.linalg.matrix_rank(phi.T @ targets/n, tol=1e-7))
            operator_rows.append(dict(m=m, omega=omega, horizon=horizon,
                degree=degree, retained=retained, theory_rank=2*retained,
                observed_rank=rank, theory_nmse=theoretical_nmse, observed_nmse=nmse,
                q1_theory_nmse=int(retained == 0), q1_observed_nmse=q1_nmse))
            check(f"operator_nmse_{m}_{omega}_{horizon}", abs(nmse-theoretical_nmse), 1e-8)
            check(f"operator_rank_{m}_{omega}_{horizon}", abs(rank-2*retained), 0)
    write_csv(out / "operator.csv", operator_rows)
    seed_rows: List[Dict[str, object]] = []
    finite = True
    min_abs = float("inf")
    for alpha in cfg["alphas"]:
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(seed)
            state = float(rng.standard_cauchy())
            trajectory = np.empty(cfg["observations"])
            for t in range(cfg["burn_in"]+cfg["observations"]):
                min_abs = min(min_abs, abs(state))
                if state == 0 or not np.isfinite(state):
                    raise FloatingPointError("Boole軌道の特異点または非有限値")
                state = alpha*state-(1-alpha)/state
                if t >= cfg["burn_in"]:
                    trajectory[t-cfg["burn_in"]] = state
            qs = cayley(trajectory)
            for lag in cfg["lags"]:
                for k in cfg["correlation_degrees"]:
                    value = np.mean(qs[lag:]**k * np.conjugate(qs[:-lag]**k))
                    seed_rows.append(dict(alpha=alpha, seed=seed, lag=lag, degree=k,
                         theory=(2*alpha-1)**(lag*k), real=float(value.real), imag=float(value.imag)))
    write_csv(out / "correlation_seeds.csv", seed_rows)
    aggregated: List[Dict[str, object]] = []
    for row in correlation_rows:
        values = [complex(s["real"],s["imag"]) for s in seed_rows if all(
            s[key] == row[key] for key in ["alpha","lag","degree"])]
        mean = np.mean(values)
        se = np.std(np.real(values), ddof=1)/np.sqrt(len(values))
        aggregated.append(dict(alpha=row["alpha"], lag=row["lag"], degree=row["degree"],
            theory=row["theory"], mean_real=float(mean.real), mean_imag=float(mean.imag),
            seed_standard_error=float(se), absolute_error=float(abs(mean-row["theory"]))))
    write_csv(out / "correlation_trajectories.csv", aggregated)
    max_orbit_error = max(r["absolute_error"] for r in aggregated)
    check("finite_trajectory_mean", max_orbit_error, cfg["trajectory_tolerance"])
    environment = dict(python=sys.version, executable=sys.executable,
                       numpy=np.__version__, platform=platform.platform(),
                       command="python verify_theory.py", finite_precision="float64/complex128")
    (ROOT/"environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    summary = dict(status="passed" if all(c["passed"] for c in checks) else "failed",
        check_count=len(checks), checks=checks, max_orbit_mean_error=max_orbit_error,
        minimum_absolute_state=min_abs, interpretation="理論値の数値照合。学習性能の比較ではない。")
    (ROOT/"metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    independent = independent_projection()
    (out/"independent_projection_validation.json").write_text(json.dumps(independent, indent=2), encoding="utf-8")
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), ROOT/"config.json", ROOT/"environment.json", ROOT/"metrics.json", out/"independent_projection_validation.json", *out.glob("*.csv")]}
    (out/"sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "checks"}, ensure_ascii=False))
    if summary["status"] != "passed":
        print(json.dumps([c for c in checks if not c["passed"]]))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
