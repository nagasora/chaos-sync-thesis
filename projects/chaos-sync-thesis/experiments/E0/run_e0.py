"""テキストE0Aの初期分布・精度監査を実行し、軌道・数値・図を保存する。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import boole_log_derivative, boole_theory, simulate_boole_orbits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def write_csv(path: Path, rows: List[Dict]) -> None:
    """条件表をUTF-8 CSVへ保存する。"""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_repeat(values: np.ndarray) -> Tuple[int, int, int]:
    """最初の厳密な再訪位置と周期を返す。添字は観測窓内0始まり。"""
    seen: Dict[float, int] = {}
    for index, value in enumerate(values):
        key = float(value)
        if not np.isfinite(key):
            break
        if key in seen:
            return seen[key], index, index - seen[key]
        seen[key] = index
    return -1, -1, 0


def summarize(rows: List[Dict], config: Dict) -> Dict:
    """事前固定した群別中央値ゲートを全条件に適用する。"""
    longest = max(config["observation_lengths"])
    full = [r for r in rows if r["length"] == longest]
    groups, convergence = [], []
    for dtype in config["precision"]:
        for alpha in config["alpha_values"]:
            for distribution in config["initial_distributions"]:
                group = [r for r in full if r["dtype"] == dtype and r["alpha"] == alpha
                         and r["initial_distribution"] == distribution]
                complete = all(r["completed"] for r in group)
                mad = float(np.median([r["mad_relative_error"] for r in group])) if complete else None
                lyap = float(np.median([r["lyapunov_absolute_error"] for r in group])) if complete else None
                passed = complete and mad <= config["gates"]["max_group_median_mad_relative_error"] and lyap <= config["gates"]["max_group_median_lyapunov_absolute_error"]
                groups.append(dict(dtype=dtype, alpha=alpha, initial_distribution=distribution,
                    completed=complete, median_mad_relative_error=mad,
                    median_lyapunov_absolute_error=lyap, passed=bool(passed)))
        for length in config["observation_lengths"]:
            group = [r for r in rows if r["dtype"] == dtype and r["length"] == length]
            complete = all(r["completed"] for r in group)
            convergence.append(dict(dtype=dtype, length=length,
                median_mad_relative_error=float(np.median([r["mad_relative_error"] for r in group])) if complete else None,
                median_lyapunov_absolute_error=float(np.median([r["lyapunov_absolute_error"] for r in group])) if complete else None))
    main = [r for r in convergence if r["dtype"] == "float64"]
    trend = bool(main) and all(main[-1][k] is not None and main[0][k] is not None and main[-1][k] <= main[0][k]
                for k in ["median_mad_relative_error", "median_lyapunov_absolute_error"])
    return dict(experiment_id=config["experiment_id"], groups=groups, convergence=convergence,
        float64_accuracy_passed=all(g["passed"] for g in groups if g["dtype"] == "float64") if "float64" in config["precision"] else None,
        float32_accuracy_passed=all(g["passed"] for g in groups if g["dtype"] == "float32") if "float32" in config["precision"] else None,
        float64_convergence_passed=trend, all_conditions_finite=all(r["completed"] for r in full),
        exact_repeat_conditions={d:sum(r["period"] > 0 for r in full if r["dtype"] == d) for d in config["precision"]},
        overall_passed=bool(all(g["passed"] for g in groups) and trend),
        condition_count=len(full), row_count=len(rows))


def plot_results(out: Path, rows: List[Dict], conditions: List[Dict], config: Dict, summary: Dict) -> None:
    """保存NPZと指標から必須図と精度監査図を再生成する。"""
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10,
                         "figure.facecolor":"#F9F8F6"})
    colors = {"float64":"#2C3E35", "float32":"#8C6D46"}
    archives = {d:np.load(out/f"orbits_{d}.npz")["orbit"] for d in config["precision"]}
    fig, axes = plt.subplots(2, len(config["alpha_values"]), figsize=(16,6), squeeze=False)
    plot_rows = []
    for j, alpha in enumerate(config["alpha_values"]):
        gamma = float(boole_theory(np.array(alpha))[0])
        index = next(i for i,r in enumerate(conditions) if r["alpha"] == alpha
                     and r["initial_distribution"] == "gaussian" and r["seed"] == config["seeds"][0])
        bins, probs = np.linspace(-10,10,101), np.linspace(.01,.99,99)
        theory_q = np.tan(np.pi*(probs-.5))
        for dtype in config["precision"]:
            x = archives[dtype][index].astype(np.float64)
            x = x[np.isfinite(x) & (x != 0)] / gamma
            if len(x) == 0:
                continue
            hist,_ = np.histogram(x,bins=bins)
            density = hist/(len(x)*np.diff(bins))
            centers = (bins[1:]+bins[:-1])/2
            quantiles = np.quantile(x,probs)
            axes[0,j].plot(centers,density,color=colors[dtype],label=dtype)
            axes[1,j].plot(theory_q,quantiles,".",ms=3,color=colors[dtype])
            for kind,xx,yy in [("density",centers,density),("qq",theory_q,quantiles)]:
                plot_rows.extend(dict(kind=kind,alpha=alpha,dtype=dtype,x=u,y=v) for u,v in zip(xx,yy))
        grid = np.linspace(-10,10,500)
        axes[0,j].plot(grid,1/(np.pi*(1+grid**2)),"k--",lw=1,label="Cauchy")
        axes[0,j].set(title=f"alpha={alpha}",xlabel="x / gamma",ylabel="Density")
        axes[1,j].plot([-32,32],[-32,32],"k--",lw=1)
        axes[1,j].set(xlabel="Theory quantile / gamma",ylabel="Observed quantile / gamma")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("E0A: preselected Gaussian seed; density normalization includes all tails")
    fig.tight_layout(); fig.savefig(figures/"density_qq.png",dpi=160); plt.close(fig)
    write_csv(out/"plot_data.csv",plot_rows)
    full = [r for r in rows if r["length"] == max(config["observation_lengths"]) and r["completed"]]
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for dtype in config["precision"]:
        group = [r for r in full if r["dtype"] == dtype]
        for ax,theory,observed in [(axes[0],"theoretical_scale","mad"),(axes[1],"theoretical_lyapunov","estimated_lyapunov")]:
            ax.scatter([r[theory] for r in group],[r[observed] for r in group],s=18,alpha=.55,color=colors[dtype],label=dtype)
            limits=[min(r[theory] for r in full),max(r[theory] for r in full)]
            ax.plot(limits,limits,"k--",lw=1)
            ax.set(xlabel=theory.replace("_"," "),ylabel=observed.replace("_"," "))
    axes[0].legend(); fig.tight_layout(); fig.savefig(figures/"theory_agreement.png",dpi=160); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    for dtype in config["precision"]:
        group = [r for r in summary["convergence"] if r["dtype"] == dtype]
        for ax,key in zip(axes[:2],["median_mad_relative_error","median_lyapunov_absolute_error"]):
            ax.loglog([r["length"] for r in group],[r[key] for r in group],"o-",color=colors[dtype],label=dtype)
            ax.set(xlabel="Observation length",ylabel=key.replace("_"," "))
        group = [r for r in full if r["dtype"] == dtype]
        axes[2].scatter([r["alpha"] for r in group],[r["period"] or np.nan for r in group],color=colors[dtype],alpha=.4)
    axes[0].legend(); axes[2].set(xlabel="alpha",ylabel="Exact repeat period (no repeat omitted)",yscale="log")
    fig.tight_layout(); fig.savefig(figures/"length_precision_audit.png",dpi=160); plt.close(fig)


def run(config: Dict, out: Path) -> Dict:
    """設定を複製してから実行し、全軌道・指標・由来を上書きせず保存する。"""
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    conditions = []
    for alpha in config["alpha_values"]:
        for distribution_index,distribution in enumerate(config["initial_distributions"]):
            for seed in config["seeds"]:
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,distribution_index])))
                initial = float({"cauchy":rng.standard_cauchy,"gaussian":rng.standard_normal,
                                 "uniform":lambda:rng.uniform(-1,1)}[distribution]())
                conditions.append(dict(condition_index=len(conditions),alpha=alpha,seed=seed,
                                       initial_distribution=distribution,initial_state=initial))
    write_csv(out/"conditions.csv",conditions)
    alphas = np.array([r["alpha"] for r in conditions])
    scales,lyaps = boole_theory(alphas)
    rows, timings = [], {}
    for dtype in config["precision"]:
        start = time.perf_counter()
        orbit,diag = simulate_boole_orbits(np.array([r["initial_state"] for r in conditions],dtype=dtype),
            alphas,config["burn_in"],max(config["observation_lengths"]),tuple(config["near_zero_thresholds"]))
        elapsed = time.perf_counter()-start
        timings[dtype] = elapsed
        np.savez_compressed(out/f"orbits_{dtype}.npz",orbit=orbit,**diag)
        for i,condition in enumerate(conditions):
            begin,end,period = first_repeat(orbit[i])
            for length in config["observation_lengths"]:
                x = orbit[i,:length].astype(np.float64)
                complete = bool(np.all(np.isfinite(x)&(x!=0)))
                row = dict(**condition,dtype=dtype,length=length,completed=complete,
                    failure_step=int(diag["failure_step"][i]),theoretical_scale=float(scales[i]),
                    theoretical_lyapunov=float(lyaps[i]),median=None,mad=None,half_iqr=None,
                    mad_absolute_error=None,mad_relative_error=None,estimated_lyapunov=None,
                    lyapunov_absolute_error=None,sample_mean=None,sample_std=None,
                    nonfinite_observation_rate=float(np.mean(~np.isfinite(x))),
                    nonfinite_update_count=int(diag["nonfinite_update_count"][i]),
                    first_repeat_start=begin if end<length else -1,
                    first_repeat_end=end if end<length else -1,period=period if end<length else 0,
                    batch_simulation_seconds=elapsed)
                for k,threshold in enumerate(config["near_zero_thresholds"]):
                    row[f"near_zero_{threshold:g}_all_updates"] = int(diag["near_zero_counts"][i,k])
                    row[f"near_zero_{threshold:g}_observed"] = int(np.sum(np.abs(x)<threshold))
                if complete:
                    median = float(np.median(x)); mad = float(np.median(np.abs(x-median)))
                    lyap = float(np.mean(boole_log_derivative(x,alphas[i])))
                    row.update(median=median,mad=mad,half_iqr=float(np.diff(np.quantile(x,[.25,.75]))[0]/2),
                        mad_absolute_error=abs(mad-scales[i]),mad_relative_error=abs(mad/scales[i]-1),
                        estimated_lyapunov=lyap,lyapunov_absolute_error=abs(lyap-lyaps[i]),
                        sample_mean=float(np.mean(x)),sample_std=float(np.std(x)))
                rows.append(row)
        print(f"{dtype}: {len(conditions)} conditions saved; simulation {elapsed:.2f}s",flush=True)
    write_csv(out/"metrics.csv",rows)
    summary = summarize(rows,config)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    plot_results(out,rows,conditions,config,summary)
    sources = [HERE/"run_e0.py",HERE/"validate_results.py",ROOT/"experiments/src/core.py",HERE/"config.json",
               ROOT/config["source_pdf"],ROOT/"references/papers/infinite-dimensional-chaotic-synchronization.pdf",
               ROOT/"experiments/tests/test_e0_boole_validation.py"]
    env = dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        python_executable=sys.executable,platform=platform.platform(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        simulation_seconds=timings,command=sys.argv,
        float_info={d:dict(bits=np.finfo(d).bits,eps=float(np.finfo(d).eps),nmant=np.finfo(d).nmant) for d in ["float32","float64","longdouble"]},
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/"environment.json").write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf-8")
    with zipfile.ZipFile(out/"source_code.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            if source.suffix in [".py",".json"]:
                archive.write(source,source.relative_to(ROOT).as_posix())
    hashes = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}"
              for p in sorted(out.rglob("*")) if p.is_file()]
    (out/"sha256.txt").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["groups","convergence"]},indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HERE/"config.json")
    parser.add_argument("--output",type=Path,default=HERE/"artifacts"/"boole_local_validation")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8-sig")),args.output.resolve())
