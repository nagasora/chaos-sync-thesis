"""E0BのK-fold/TM正対照を独立Cauchy標本で検証する。"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_e0 import write_csv
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import cayley_modes, kfold_cotangent, generalized_boole, boole_theory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def uniform_ks(values: np.ndarray) -> float:
    """[0,1]標本と一様CDFの距離を返す。p値や独立性の推定は行わない。"""
    ordered = np.sort(values)
    n = len(ordered)
    return float(max(np.max(np.arange(1, n + 1) / n - ordered),
                     np.max(ordered - np.arange(n) / n)))


def summarize(shift: List[Dict], gram: List[Dict], controls: List[Dict],
              poles: List[Dict], config: Dict) -> Dict:
    """事前固定ゲートを独立性・極の区別を保ったまま適用する。"""
    largest = max(config["sample_lengths"])
    smallest = min(config["sample_lengths"])
    longest = [r for r in gram if r["length"] == largest]
    shortest = [r for r in gram if r["length"] == smallest]
    finite_poles = [r for r in poles if r["status"] == "finite"]
    g = config["gates"]
    checks = {
        "iid_finite": all(r["invalid_count"] == 0 for r in shift),
        "shift_identity": all(r["max_residual"] is not None and r["max_residual"] <= g["max_shift_residual"] for r in shift),
        "unit_modulus": all(r["modulus_error"] <= g["max_modulus_error"] for r in longest),
        "cauchy_input": all(r["input_uniform_ks"] <= g["max_uniform_ks"] for r in longest),
        "cauchy_output": all(r["output_uniform_ks"] <= g["max_uniform_ks"] for r in shift),
        "gram_accuracy_all_seeds": all(r["frobenius"] <= g["max_gram_frobenius"] for r in longest),
        "gram_convergence_all_seeds": all(next(r["frobenius"] for r in longest if r["seed"] == s)
            < next(r["frobenius"] for r in shortest if r["seed"] == s) for s in config["seeds"]),
        "gram_convergence_median": float(np.median([r["frobenius"] for r in longest])) < float(np.median([r["frobenius"] for r in shortest])),
        "finite_pole_shift": bool(finite_poles) and all(r["residual"] <= g["max_shift_residual"] for r in finite_poles),
        "pole_diagnostics": all(r["status"] == ("undefined_pole" if r["K"] % 2 == 0 else "finite") for r in poles if r["case"] == "exact_zero"),
        "positive_control": all(r["rms"] <= g["max_shift_residual"] for r in controls if r["alpha"] == 0.5),
        "negative_control": all(r["rms"] >= g["min_negative_control_rms"] for r in controls if r["alpha"] != 0.5),
    }
    return dict(experiment_id=config["experiment_id"], scientific_gates=checks,
        overall_passed=bool(all(checks.values())), seed_count=len(config["seeds"]),
        maximum_shift_residual=max(r["max_residual"] for r in shift if r["max_residual"] is not None),
        maximum_near_pole_residual=max(r["residual"] for r in finite_poles if r["near_pole"]),
        gram_median_by_length={str(n):float(np.median([r["frobenius"] for r in gram if r["length"] == n])) for n in config["sample_lengths"]},
        negative_rms_by_alpha={str(a):float(np.median([r["rms"] for r in controls if r["alpha"] == a])) for a in config["negative_alphas"]})


def plot_results(out: Path, shift: List[Dict], gram: List[Dict], controls: List[Dict],
                 poles: List[Dict], config: Dict) -> None:
    """未集約CSVと保存Gram行列を元に、正対照・収束・負対照の図を描く。"""
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#F9F8F6", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for K in config["K_values"]:
        rows = [r for r in shift if r["K"] == K]
        axes[0].scatter([r["k"] for r in rows], [r["max_residual"] for r in rows], alpha=.5, label=f"K={K}")
    axes[0].axhline(config["gates"]["max_shift_residual"], color="black", ls="--", label="Gate")
    axes[0].set(xlabel="Mode k", ylabel="Maximum shift residual", yscale="log")
    axes[0].legend(fontsize=8)
    rows = [r for r in poles if r["case"] == "offset" and r["status"] == "finite"]
    for K in config["K_values"]:
        group = [r for r in rows if r["K"] == K]
        axes[1].scatter([abs(r["offset"]) for r in group], [max(r["residual"],1e-18) for r in group], alpha=.4, label=f"K={K}")
    axes[1].axhline(config["gates"]["max_shift_residual"], color="black", ls="--")
    axes[1].set(xlabel="Angular distance to pole", ylabel="Residual (zeros shown at 1e-18)", xscale="log", yscale="log")
    for a in config["negative_alphas"]:
        group = [r for r in controls if r["alpha"] == a]
        axes[2].scatter([a] * len(group), [r["rms"] for r in group], color="#2C3E35")
    axes[2].axhline(config["gates"]["min_negative_control_rms"], color="black", ls="--")
    axes[2].set(xlabel="Boole alpha (scale standardized)", ylabel="K=2 shift RMS", yscale="log")
    fig.tight_layout(); fig.savefig(figdir / "shift_and_controls.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for seed in config["seeds"]:
        rows = [r for r in gram if r["seed"] == seed]
        axes[0].loglog([r["length"] for r in rows], [r["frobenius"] for r in rows], color="#8C6D46", alpha=.3)
    ns = np.array(config["sample_lengths"])
    d = len(config["gram_orders"])
    medians = [np.median([r["frobenius"] for r in gram if r["length"] == n]) for n in ns]
    axes[0].loglog(ns, medians, "o-", color="#2C3E35", label="Seed median")
    axes[0].loglog(ns, np.sqrt(d*(d-1)/ns), "k--", label="RMS theory sqrt(d(d-1)/N)")
    axes[0].set(xlabel="IID sample count", ylabel="Gram Frobenius error")
    axes[0].legend(fontsize=7)
    with np.load(out / f"samples_seed_{config['seeds'][0]}.npz") as sample:
        for i, index in enumerate([0, len(ns)-1]):
            delta = np.abs(sample["grams"][index] - np.eye(d))
            im = axes[i+1].imshow(delta, vmin=0, vmax=.05, cmap="Greys")
            axes[i+1].set(title=f"First seed: N={ns[index]}", xlabel="Mode", ylabel="Mode",
                          xticks=range(d), xticklabels=config["gram_orders"],
                          yticks=range(d), yticklabels=config["gram_orders"])
            fig.colorbar(im, ax=axes[i+1], fraction=.046)
    fig.tight_layout(); fig.savefig(figdir / "gram_convergence.png", dpi=160); plt.close(fig)


def run(config: Dict, out: Path) -> Dict:
    """設定を凍結し、全入力・写像出力・Cayley座標・Gram行列と判定を保存する。"""
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shift, gram, controls, poles = [], [], [], []
    orders = tuple(config["gram_orders"])
    for seed in config["seeds"]:
        rng = np.random.Generator(np.random.PCG64(seed))
        u = rng.random(max(config["sample_lengths"]))
        if np.any((u <= 0) | (u >= 1)):
            raise ValueError("乱数がCauchy変換の端点に到達しました。")
        theta = np.pi * u
        x = np.cos(theta) / np.sin(theta)
        modes = cayley_modes(x, orders)
        q = cayley_modes(x, (1,))[:, 0]
        grams = []
        for n in config["sample_lengths"]:
            G = modes[:n].conj().T @ modes[:n] / n
            grams.append(G)
            gram.append(dict(seed=seed, length=n, frobenius=float(np.linalg.norm(G-np.eye(len(orders)))),
                modulus_error=float(np.max(np.abs(np.abs(modes[:n])-1))),
                input_uniform_ks=uniform_ks(.5+np.arctan(x[:n])/np.pi),
                theoretical_squared_error=len(orders)*(len(orders)-1)/n))
        mapped = []
        for K in config["K_values"]:
            y, invalid, near = kfold_cotangent(x, K, config["pole_sine_threshold"])
            mapped.append(y)
            valid = ~invalid
            for k in config["shift_orders"]:
                residual = np.abs(cayley_modes(y[valid], (k,))[:, 0] - q[valid]**(K*k))
                shift.append(dict(seed=seed, K=K, k=k, count=len(x), invalid_count=int(invalid.sum()),
                    near_count=int(near.sum()), max_residual=float(np.max(residual)) if len(residual) else None,
                    rms=float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
                    output_uniform_ks=uniform_ks(.5+np.arctan(y[valid])/np.pi) if len(residual) else float("inf")))
        for alpha in config["negative_alphas"]:
            gamma = float(boole_theory(np.array(alpha))[0])
            physical = gamma*x
            y, invalid, _ = generalized_boole(physical, np.array(alpha))
            if np.any(invalid):
                raise FloatingPointError("負対照に非有限値が発生しました。")
            residual = np.abs(cayley_modes(y, (1,), gamma=gamma)[:,0] - cayley_modes(physical, (2,), gamma=gamma)[:,0])
            controls.append(dict(seed=seed, alpha=alpha, rms=float(np.sqrt(np.mean(residual**2))), maximum=float(np.max(residual))))
        np.savez_compressed(out/f"samples_seed_{seed}.npz", theta=theta, x=x, q=q,
                            mapped=np.asarray(mapped), grams=np.asarray(grams))
        print(f"E0B seed {seed} saved", flush=True)
    for K in config["K_values"]:
        cases = [("offset", j, offset, float(np.cos(j*np.pi/K+offset)/np.sin(j*np.pi/K+offset)))
                 for j in range(1, K) for offset in config["pole_offsets"]]
        cases.append(("exact_zero", 0, 0.0, 0.0))
        for case, j, offset, x in cases:
            y, invalid, near = kfold_cotangent(np.array([x]), K, config["pole_sine_threshold"])
            for k in config["shift_orders"]:
                residual = None if invalid[0] else float(np.abs(cayley_modes(y,(k,))[0,0]-cayley_modes(np.array([x]),(K*k,))[0,0]))
                poles.append(dict(case=case, K=K, k=k, pole_index=j, offset=offset, x=x,
                    y=None if invalid[0] else float(y[0]), near_pole=bool(near[0]),
                    status="undefined_pole" if invalid[0] else "finite", residual=residual))
    write_csv(out/"shift_metrics.csv",shift)
    write_csv(out/"gram_metrics.csv",gram)
    write_csv(out/"control_metrics.csv",controls)
    write_csv(out/"pole_metrics.csv",poles)
    summary = summarize(shift,gram,controls,poles,config)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    plot_results(out,shift,gram,controls,poles,config)
    source_paths = [HERE/"run_e0b.py",HERE/"run_e0.py",HERE/"validate_results.py",HERE/"config_e0b.json",
                    ROOT/"experiments/src/core.py",ROOT/"experiments/tests/test_e0_boole_validation.py"]
    env = dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        platform=platform.platform(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        command=sys.argv,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        reference_sha256={config["source_pdf"]:hashlib.sha256((ROOT/config["source_pdf"]).read_bytes()).hexdigest()},
        generation="IID PCG64, not an iterative chaotic orbit; no burn-in",
        core_precision="float64 / complex128")
    (out/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    with zipfile.ZipFile(out/"source_code.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in source_paths:
            archive.write(p,p.relative_to(ROOT).as_posix())
    hashes = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}"
              for p in sorted(out.rglob("*")) if p.is_file()]
    (out/"sha256.txt").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HERE/"config_e0b.json")
    parser.add_argument("--output",type=Path,default=HERE/"artifacts"/"kfold_tm_validation")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8-sig")),args.output.resolve())
