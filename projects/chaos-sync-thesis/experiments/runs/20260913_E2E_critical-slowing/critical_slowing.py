"""共有会話のE2Eを有限観測窓・数値感度を分離して実行、検証する。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
from numba import njit
from scipy.optimize import brentq

import e2e_dynamics as dynamics

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCE = PROJECT / "artifacts/E2_tangent_two_system_deep_20260913"
FIELDS = ["first_entry", "last_outside", "tail_dwell", "releave", "logd", "invalid", "linear_count", "first_invalid_step"]
CONFIG = {
    "seed": 20260913, "main_beta": 1.2, "main_pairs": 1024,
    "deltas": [-0.01, 0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    "replication_betas": [1.05, 1.5], "replication_pairs": 512,
    "replication_deltas": [0.002, 0.005, 0.01, 0.02],
    "checkpoints": [7500, 15000, 30000], "eta": 1e-6,
    "sensitivity_etas": [1e-4, 1e-6, 1e-8], "sensitivity_pairs": 512,
    "sensitivity_deltas": [-0.01, 0.002, 0.01], "sensitivity_steps": 7500,
    "entry_tol": 1e-8, "exit_tol": 1e-5, "entry_hold": 10, "tail_hold": 500,
    "ftle_pairs": 256, "ftle_steps": 32768, "bootstrap": 2000,
    "ftle_max_error_gate": 0.01, "sensitivity_span_gate": 0.10,
    "fit_min_confirmed": 0.95, "fit_min_conditions": 4,
}


def dump(path: Path, value: Any) -> None:
    """非有限値を暗黙にJSONへ混入させず保存する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameters(beta: float) -> Tuple[float, float, float]:
    """既存出力混合runと同じ正尺度・指数・下側臨界を計算する。"""
    g = float(brentq(lambda z: np.tanh(beta*z)/z-1 if z else beta-1, 0, 1, xtol=5e-15))
    lam = math.log(beta)+2*math.log1p(g)
    return g, lam, -math.expm1(-lam)/2


def preflight() -> None:
    """実行前に条件・元資料ハッシュと入口コードを固定する。"""
    if (ROOT / "config.json").exists():
        raise FileExistsError("preflight already exists")
    source_paths = [SOURCE / "note/E2_tangent_two_system_theory_experiment_note_v3.tex",
                    SOURCE / "E2D_global_basin_20260913/code/e2d_reference.py",
                    SOURCE / "E2D_global_basin_20260913/code/e2d_numba.py",
                    SOURCE / "E2D_global_basin_20260913/code/run_e2d.py",
                    SOURCE / "E2D_global_basin_20260913/results/E2D_SUMMARY.json"]
    snapshot = PROJECT.parents[1] / "shared-e2-discussion.html"
    if snapshot.exists():
        (ROOT / "artifacts").mkdir(exist_ok=True)
        snapshot.rename(ROOT / "artifacts/shared-discussion.html")
    dump(ROOT / "config.json", CONFIG)
    dump(ROOT / "provenance.json", {
        "shared_url": "https://chatgpt.com/share/6aa616e9-dbc4-83ee-82e4-1cf6241c6e07",
        "sources": {p.relative_to(PROJECT).as_posix(): sha(p) for p in source_paths},
        "claim": "Local continuation, not a rerun of the original cloud bundle.",
    })
    dump(ROOT / "preflight.json", {"passed": True, "hashes": {
        name: sha(ROOT/name) for name in ["critical_slowing.py", "e2e_dynamics.py", "README.md", "config.json", "provenance.json"]}})
    print("preflight passed", flush=True)


@njit
def diagonal_samples(beta: float, s0: np.ndarray, steps: int) -> np.ndarray:
    """同期多様体の独立初期値ごとの時間平均指数を記録する。"""
    output = np.empty(s0.size)
    for i in range(s0.size):
        s, total = s0[i], 0.0
        for _ in range(steps):
            s = math.tan(beta*s)
            total += math.log(beta)+2*math.log(math.hypot(1.0, s))
        output[i] = total/steps
    return output


def summarize(values: np.ndarray, horizon: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """全初期値を分母にし、未確認・数値無効を明示する。"""
    valid = values[:, 5] == 0
    entered = valid & (values[:, 0] >= 0)
    confirmed = valid & (values[:, 2] >= cfg["tail_hold"])
    times = np.where(confirmed, values[:, 1]+1, np.inf)
    median = float(np.median(times))
    p = float(confirmed.mean())
    n = len(values)
    z = 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"horizon": horizon, "pairs": n, "invalid": int((~valid).sum()),
            "confirmed_rate": p, "confirmed_ci_low": center-half, "confirmed_ci_high": center+half,
            "unconfirmed_rate": float((~confirmed).mean()), "ever_entered_rate": float(entered.mean()),
            "releave_given_entered": float(values[entered, 3].mean()) if entered.any() else None,
            "median_candidate_time": median if math.isfinite(median) else None,
            "median_tail_dwell": float(np.median(values[valid, 2])) if valid.any() else None}


def fit_slopes(data: Dict[str, np.ndarray], metadata: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """未確認を無限大とした中央値を用い、同じ初期値の対応をbootstrapで保持する。"""
    fits = []
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 99]))
    for bi, beta in enumerate([cfg["main_beta"]]+cfg["replication_betas"]):
        cases = [m for m in metadata if m["suite"] == "main" and m["beta"] == beta and m["delta"] > 0]
        for ci, horizon in enumerate(cfg["checkpoints"]):
            eligible, times = [], []
            for m in cases:
                v = data[m["key"]][:, ci]
                confirmed = (v[:, 5] == 0) & (v[:, 2] >= cfg["tail_hold"])
                if confirmed.mean() >= cfg["fit_min_confirmed"]:
                    eligible.append(m["delta"])
                    times.append(np.where(confirmed, v[:, 1]+1, np.inf))
            row: Dict[str, Any] = {"beta": beta, "horizon": horizon, "eligible_deltas": eligible,
                                    "slope": None, "ci95": None, "bootstrap_invalid": 0,
                                    "status": "insufficient_eligible_conditions"}
            if len(times) >= cfg["fit_min_conditions"]:
                arr = np.asarray(times)
                med = np.median(arr, axis=1)
                if np.all(np.isfinite(med) & (med > 0)):
                    x = np.log(eligible)
                    row["slope"] = float(np.polyfit(x, np.log(med), 1)[0])
                    boot = []
                    for _ in range(cfg["bootstrap"]):
                        ids = rng.integers(0, arr.shape[1], arr.shape[1])
                        bm = np.median(arr[:, ids], axis=1)
                        if np.all(np.isfinite(bm) & (bm > 0)):
                            boot.append(float(np.polyfit(x, np.log(bm), 1)[0]))
                    row["bootstrap_invalid"] = cfg["bootstrap"]-len(boot)
                    if boot:
                        row["ci95"] = np.quantile(boot, [.025, .975]).tolist()
                        row["status"] = "finite_window_descriptive_fit"
            fits.append(row)
    return fits


def run() -> None:
    """固定条件を最後まで実行し、軌道別・条件別の値を保存する。"""
    gate = json.loads((ROOT / "preflight.json").read_text(encoding="utf-8"))
    if not gate["passed"] or any(sha(ROOT/name) != h for name, h in gate["hashes"].items()):
        raise ValueError("preflight code/config changed")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    out = ROOT / "artifacts"
    if (out / "checkpoints.npz").exists():
        raise FileExistsError("results already exist")
    out.mkdir(exist_ok=True)
    dump(ROOT / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__,
        "matplotlib": matplotlib.__version__, "numba_threads": numba.get_num_threads(), "fastmath": False})
    started = time.perf_counter()
    data, initials, metadata, summary, rows, ftle_rows = {}, {}, [], [], [], []
    betas = [cfg["main_beta"]]+cfg["replication_betas"]
    for bi, beta in enumerate(betas):
        gamma, lam, kc = parameters(beta)
        n = cfg["main_pairs"] if bi == 0 else cfg["replication_pairs"]
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1, bi]))
        x, y = gamma*rng.standard_cauchy((2, n))
        s, logd = (x+y)/2, np.log(abs(x-y)/2)
        initials[f"main_{bi}"] = np.array([s, logd])
        for di, delta in enumerate(cfg["deltas"] if bi == 0 else cfg["replication_deltas"]):
            key = f"main_{bi}_{di}"
            meta = {"key": key, "initial_key": f"main_{bi}", "suite": "main", "beta": beta,
                    "gamma": gamma, "kappa_c": kc, "delta": delta, "kappa": kc+delta,
                    "lambda_perp": lam+math.log(abs(1-2*(kc+delta))), "eta": cfg["eta"]}
            values = dynamics.measure(beta, kc+delta, s, logd, max(cfg["checkpoints"]),
                cfg["eta"], cfg["entry_tol"], cfg["exit_tol"], cfg["entry_hold"], cfg["tail_hold"], np.array(cfg["checkpoints"]))
            data[key] = values
            metadata.append(meta)
            for ci, horizon in enumerate(cfg["checkpoints"]):
                summary.append({**meta, **summarize(values[:, ci], horizon, cfg)})
            print(f"main beta={beta} delta={delta}: complete", flush=True)
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 2, bi]))
        s0 = gamma*rng.standard_cauchy(cfg["ftle_pairs"])
        initials[f"ftle_{bi}"] = s0
        samples = diagonal_samples(beta, s0, cfg["ftle_steps"])
        for i, val in enumerate(samples):
            ftle_rows.append({"beta": beta, "pair": i, "initial": s0[i], "estimate": val, "theory": lam})
    beta = cfg["main_beta"]
    gamma, lam, kc = parameters(beta)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 3]))
    x, y = gamma*rng.standard_cauchy((2, cfg["sensitivity_pairs"]))
    s, logd = (x+y)/2, np.log(abs(x-y)/2)
    initials["sensitivity"] = np.array([s, logd])
    for di, delta in enumerate(cfg["sensitivity_deltas"]):
        for ei, eta in enumerate(cfg["sensitivity_etas"]):
            key = f"sensitivity_{di}_{ei}"
            meta = {"key": key, "initial_key": "sensitivity", "suite": "sensitivity", "beta": beta,
                    "gamma": gamma, "kappa_c": kc, "delta": delta, "kappa": kc+delta,
                    "lambda_perp": lam+math.log(abs(1-2*(kc+delta))), "eta": eta}
            values = dynamics.measure(beta, kc+delta, s, logd, cfg["sensitivity_steps"], eta,
                cfg["entry_tol"], cfg["exit_tol"], cfg["entry_hold"], cfg["tail_hold"], np.array([cfg["sensitivity_steps"]]))
            data[key] = values
            metadata.append(meta)
            summary.append({**meta, **summarize(values[:, 0], cfg["sensitivity_steps"], cfg)})
            print(f"sensitivity delta={delta} eta={eta}: complete", flush=True)
    for meta in metadata:
        values = data[meta["key"]]
        horizons = cfg["checkpoints"] if meta["suite"] == "main" else [cfg["sensitivity_steps"]]
        for i in range(len(values)):
            for ci, horizon in enumerate(horizons):
                rows.append({"key": meta["key"], "pair": i, "horizon": horizon,
                             **{name: values[i, ci, j] for j, name in enumerate(FIELDS)}})
    np.savez_compressed(out / "initials.npz", **initials)
    np.savez_compressed(out / "checkpoints.npz", **data)
    table(out / "per_pair.csv", rows)
    table(out / "summary.csv", summary)
    table(out / "ftle.csv", ftle_rows)
    dump(out / "conditions.json", metadata)
    fits = fit_slopes(data, metadata, cfg)
    dump(out / "scaling.json", fits)
    ftle_summary = []
    for beta in betas:
        r = [v for v in ftle_rows if v["beta"] == beta]
        vals = np.array([v["estimate"] for v in r])
        ftle_summary.append({"beta": beta, "mean": float(vals.mean()), "theory": r[0]["theory"],
            "error": float(vals.mean()-r[0]["theory"]), "seed_se": float(vals.std(ddof=1)/np.sqrt(len(vals)))})
    sensitivity = []
    for delta in cfg["sensitivity_deltas"]:
        rr = [r for r in summary if r["suite"] == "sensitivity" and r["delta"] == delta]
        sensitivity.append({"delta": delta, "rate_span": max(r["confirmed_rate"] for r in rr)-min(r["confirmed_rate"] for r in rr)})
    metrics = {"status": "executed", "conditions": len(metadata), "pair_conditions": sum(len(v) for v in data.values()),
        "checkpoint_rows": len(rows), "seconds": time.perf_counter()-started, "ftle": ftle_summary,
        "ftle_gate": all(abs(r["error"]) <= cfg["ftle_max_error_gate"] for r in ftle_summary),
        "sensitivity": sensitivity, "sensitivity_gate": all(r["rate_span"] <= cfg["sensitivity_span_gate"] for r in sensitivity),
        "invalid_checkpoint_count": sum(r["invalid"] for r in summary),
        "scope": "finite-window stability and log-distance hybrid approximation; no global or permanent synchronization proof"}
    dump(ROOT / "metrics.json", metrics)
    plot(summary, fits, ftle_summary)
    dump(ROOT / "sha256.json", {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.name not in ["sha256.json", "validation.json", "RESULTS.md"]})
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def plot(summary: List[Dict[str, Any]], fits: List[Dict[str, Any]], ftle: List[Dict[str, Any]]) -> None:
    """保存した観測量を有限窓の意味が分かるラベルで描く。"""
    plt.rcParams.update({"figure.facecolor": "#F9F8F6", "axes.facecolor": "#FFFFFF",
        "text.color": "#1E1E1E", "axes.labelcolor": "#1E1E1E", "axes.edgecolor": "#EAE8E3",
        "axes.prop_cycle": plt.cycler(color=["#2C3E35", "#8C6D46", "#736E65"])})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    main = [r for r in summary if r["suite"] == "main" and r["beta"] == 1.2]
    for horizon in CONFIG["checkpoints"]:
        rr = [r for r in main if r["horizon"] == horizon]
        axes[0, 0].plot([r["delta"] for r in rr], [r["confirmed_rate"] for r in rr], "o-", label=f"T={horizon}")
        positive = [r for r in rr if r["delta"] > 0 and r["median_candidate_time"] is not None]
        axes[0, 1].loglog([r["delta"] for r in positive], [r["median_candidate_time"] for r in positive], "o-", label=f"T={horizon}")
    axes[0, 0].axvline(0, color="gray", ls=":")
    axes[0, 0].set(xlabel="kappa - kappa_c", ylabel="Fraction: final dwell >= 500", ylim=(-.03, 1.03), title="Finite-window confirmation, beta=1.2")
    axes[0, 1].set(xlabel="kappa - kappa_c", ylabel="Median candidate L_T+1", title="Unconfirmed trajectories assigned infinity")
    for eta in CONFIG["sensitivity_etas"]:
        rr = [r for r in summary if r["suite"] == "sensitivity" and r["eta"] == eta]
        axes[1, 0].plot([r["delta"] for r in rr], [r["confirmed_rate"] for r in rr], "o-", label=f"eta={eta:g}")
    axes[1, 0].set(xlabel="kappa - kappa_c", ylabel="Final dwell fraction, T=7500", title="Linearization threshold sensitivity")
    axes[1, 1].errorbar([r["beta"] for r in ftle], [r["error"] for r in ftle],
                      yerr=[1.96*r["seed_se"] for r in ftle], fmt="o")
    axes[1, 1].axhline(0, color="gray", ls=":")
    axes[1, 1].set(xlabel="beta", ylabel="Mean lambda0 - theory", title="Independent orbit mean +/- 1.96 SE")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend(fontsize=8)
    fig.suptitle("E2E output-mixed tangent pair | finite observation, not permanent synchronization")
    for ext in ["png", "pdf"]:
        fig.savefig(ROOT / "artifacts" / f"critical_slowing.{ext}", dpi=180)
    plt.close(fig)


def validate() -> None:
    """条件網羅・全CSV転記・代表再生・集計・source固定を独立に照合する。"""
    hashes = json.loads((ROOT / "sha256.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT/p) == h for p, h in hashes.items()), "hash mismatch"
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "artifacts/conditions.json").read_text(encoding="utf-8"))
    expected = len(cfg["deltas"])+len(cfg["replication_betas"])*len(cfg["replication_deltas"])+len(cfg["sensitivity_etas"])*len(cfg["sensitivity_deltas"])
    assert len(metadata) == expected and len({m["key"] for m in metadata}) == expected
    replayed = 0
    with np.load(ROOT / "artifacts/checkpoints.npz") as data, np.load(ROOT / "artifacts/initials.npz") as initial:
        with (ROOT / "artifacts/per_pair.csv").open(encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == sum(v.shape[0]*v.shape[1] for v in data.values())
        for r in csv_rows:
            ci = cfg["checkpoints"].index(int(r["horizon"]))
            np.testing.assert_array_equal(np.array([float(r[f]) for f in FIELDS]), data[r["key"]][int(r["pair"]), ci])
        with (ROOT / "artifacts/summary.csv").open(encoding="utf-8") as stream:
            summaries = list(csv.DictReader(stream))
        for meta in metadata:
            cp = cfg["checkpoints"] if meta["suite"] == "main" else [cfg["sensitivity_steps"]]
            source = initial[meta["initial_key"]]
            actual = dynamics.measure(meta["beta"], meta["kappa"], source[0, :2], source[1, :2], max(cp),
                meta["eta"], cfg["entry_tol"], cfg["exit_tol"], cfg["entry_hold"], cfg["tail_hold"], np.array(cp))
            np.testing.assert_array_equal(actual, data[meta["key"]][:2])
            for ci, horizon in enumerate(cp):
                recorded = next(r for r in summaries if r["key"] == meta["key"] and int(r["horizon"]) == horizon)
                fresh = summarize(data[meta["key"]][:, ci], horizon, cfg)
                assert float(recorded["confirmed_rate"]) == fresh["confirmed_rate"]
                assert int(recorded["invalid"]) == fresh["invalid"]
            replayed += 2
    dump(ROOT / "validation.json", {"passed": True, "hashes": len(hashes), "conditions": expected,
        "csv_rows": len(csv_rows), "replayed_pairs": replayed, "meaning": "integrity and reproducibility, not proof of asymptotic synchronization"})
    print("validation passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "run", "validate"])
    args = parser.parse_args()
    globals()[args.action]()
