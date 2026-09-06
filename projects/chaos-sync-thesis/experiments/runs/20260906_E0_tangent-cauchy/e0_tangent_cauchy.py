"""tan(beta*x) の事前登録E0A〜Fを実行・保存・再検証する。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numba
import numpy as np
import scipy
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
BOOLE = HERE.parent / "20260806_E0_boole-invariant-measure/e0_boole_validation.py"
spec = importlib.util.spec_from_file_location("boole_e0_reused", BOOLE)
assert spec is not None and spec.loader is not None
boole = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = boole
spec.loader.exec_module(boole)


def configuration() -> Dict[str, Any]:
    """承認済み条件を単一の機械可読設定として返す。"""
    return dict(beta=[.5, .9, 1., 1.01, 1.1, 1.5, 2.],
                epsilon=[.1, .05, .02, .01, .005, .002, .001, .0005, .0002, .0001],
                scales=[.1, 1., 10.], seed_root=20260906,
                a_seeds=10, a_n=100000, family_alpha=.01,
                b_seeds=5, b_n=4096, b_steps=4096,
                long_beta=[.9999, 1., 1.0001], long_n=512, long_steps=200000,
                c_seeds=20, c_steps=200000, window=50000,
                fp32_beta=[1.0001, 1.01, 2.], audit_k=100, mp_digits=100,
                block=1024, gamma_floor=1e-12, fixed_rtol=1e-8,
                fit_cutoffs=[.1, .01], tail_multipliers=[10, 100],
                bootstrap=None, interval="independent-seed Student-t 95%, descriptive",
                classification="DKW gate only for A-float64; other comparisons descriptive")


def write_json(path: Path, value: Any) -> None:
    """非有限JSONを許さずUTF-8で記録する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """同一責務の条件別行をCSVへ保存する。"""
    if not rows:
        raise ValueError(f"空の結果: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """保存済みCSVを読み込む。"""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    """保存物のSHA-256を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_scale(beta: float) -> float:
    """ゼロ根を除外した尺度固定点を区間付き求根で求める。"""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("betaは有限正数")
    if beta <= 1:
        return 0.
    def residual(g: float) -> float:
        if g < .01:
            z = g * g
            return z * (1/3 + z * (1/5 + z * (1/7 + z/9))) - (beta - 1)
        return math.atanh(g) / g - beta
    return brentq(residual, 0., np.nextafter(1., 0.), xtol=5e-15, rtol=1e-14)


def lyapunov_theory(beta: float) -> float:
    """非退化不変Cauchy測度上の理論値を返す。"""
    if beta <= 1:
        raise ValueError("非退化不変測度の比較はbeta>1")
    return math.log(beta) + 2 * math.log1p(positive_scale(beta))


def critical_increment(g: float) -> float:
    """逆二乗差の桁落ちを避けてbeta=1の補助量を返す。"""
    if g < .01:
        return 2/3 + g*g/15 - 2*g**4/189 + g**6/675
    return 1/math.tanh(g)**2 - 1/g**2


@njit(cache=True)
def evolve(initial: np.ndarray, beta: float, steps: int) -> np.ndarray:
    """型を保った軌道ブロックを生成し、非有限標本の以後をNaNにする。"""
    states = np.empty((steps+1, initial.size), dtype=initial.dtype)
    states[0] = initial
    # float32の積をfloat64へ暗黙昇格させないため引数も同じ配列型へ格納する。
    argument = np.empty_like(initial)
    for t in range(steps):
        for j in range(initial.size):
            if np.isfinite(states[t, j]):
                argument[j] = beta * states[t, j]
                states[t+1, j] = np.tan(argument[j])
            else:
                states[t+1, j] = np.nan
    return states


@njit(cache=True)
def scale_series(beta: float, initial: float, steps: int) -> np.ndarray:
    """状態集団とは独立に尺度漸化式だけを計算する。"""
    result = np.empty(steps+1)
    result[0] = initial
    for i in range(steps):
        result[i+1] = np.tanh(beta * result[i])
    return result


def sample(stage: int, condition: int, seed: int, size: int, scale: float) -> np.ndarray:
    """再抽選を行わず固定SeedSequenceから初期標本を生成する。"""
    rng = np.random.default_rng(np.random.SeedSequence([20260906, stage, condition, seed]))
    return scale * rng.standard_cauchy(size)


def checkpoints(steps: int) -> List[int]:
    """0、2冪、終端を重複なく返す。"""
    return sorted({0, steps, *(2**k for k in range(steps.bit_length()) if 2**k <= steps)})


def distribution(values: np.ndarray, scale: float) -> Dict[str, Any]:
    """Cauchyの存在しない母平均・母分散を用いず分布を評価する。"""
    if not np.isfinite(values).all():
        return dict(status="数値精度上の制約", n=values.size, median=None,
                    estimated_scale=None, ks=None, tail10=None, tail100=None)
    return dict(status="記述的評価", n=values.size, median=float(np.median(values)),
                estimated_scale=boole.estimate_cauchy_scale(values),
                ks=boole.cauchy_ks_distance(values, scale) if scale > 0 else None,
                tail10=float(np.mean(np.abs(values) > 10*scale)),
                tail100=float(np.mean(np.abs(values) > 100*scale)))


def audit_update(candidates: Dict[str, np.ndarray], states: np.ndarray, beta: float,
                 offset: int, k: int) -> Dict[str, Any]:
    """ブロックから極・大引数候補と非有限診断を集計する。"""
    x, y = states[:-1].reshape(-1), states[1:].reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        argument = np.multiply(x, np.asarray(beta, dtype=x.dtype))
        distance = np.abs(np.remainder(argument.astype(float), np.pi) - np.pi/2)
    valid = np.isfinite(x) & np.isfinite(argument) & np.isfinite(y)
    pool = np.flatnonzero(valid)
    for kind, scores in (("pole", distance), ("large", -np.abs(argument.astype(float)))):
        if pool.size:
            count = min(k, pool.size)
            chosen = pool[np.argpartition(scores[pool], count-1)[:count]]
            records = np.column_stack([x[chosen], argument[chosen], y[chosen],
                                       offset+chosen//states.shape[1], chosen % states.shape[1],
                                       distance[chosen]])
            records = np.vstack([candidates[kind], records])
            rank = records[:, 5] if kind == "pole" else -np.abs(records[:, 1])
            candidates[kind] = records[np.argsort(rank, kind="stable")[:k]]
    failures = np.argwhere(np.isfinite(states[:-1]) & ~np.isfinite(states[1:]))
    return dict(nonfinite=int(np.count_nonzero(~np.isfinite(y))),
                first_failure_step=int(offset+failures[0, 0]+1) if failures.size else None,
                first_failure_index=int(failures[0, 1]) if failures.size else None,
                zero_count=int(np.count_nonzero(y == 0)),
                max_abs=float(np.max(np.abs(states[np.isfinite(states)]))) if np.isfinite(states).any() else None,
                min_pole=float(np.min(distance[valid])) if valid.any() else None)


def finish_audit(candidates: Dict[str, np.ndarray], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """100桁で入力積とtan評価の誤差を分離する。"""
    rows = []
    with mp.workdps(100):
        b = mp.mpf(float(meta["beta"]))
        for kind, points in candidates.items():
            for x, arg, y, step, index, distance in points:
                exact_arg = b * mp.mpf(float(x))
                rounded_arg = mp.mpf(float(arg))
                reference = mp.tan(exact_arg)
                rounded_reference = mp.tan(rounded_arg)
                n = mp.nint((exact_arg-mp.pi/2)/mp.pi)
                row = dict(meta, kind=kind, x=float(x), argument=float(arg), y=float(y),
                           step=int(step), sample_index=int(index), pole_distance_fp=float(distance),
                           pole_distance_mp=str(abs(exact_arg-(mp.pi/2+n*mp.pi))),
                           reference=str(reference),
                           argument_error=str(abs(exact_arg-rounded_arg)),
                           multiplication_effect=str(abs(reference-rounded_reference)),
                           tan_evaluation_error=str(abs(mp.mpf(float(y))-rounded_reference)),
                           total_error=str(abs(mp.mpf(float(y))-reference)),
                           scaled_error=float(abs(mp.mpf(float(y))-reference)/max(1, abs(reference))))
                rows.append(row)
    return rows


def empty_candidates() -> Dict[str, np.ndarray]:
    return {"pole": np.empty((0, 6)), "large": np.empty((0, 6))}


def theory_checks() -> List[Dict[str, Any]]:
    """有限branch和の尾部上界・独立積分・K-fold対照を照合する。"""
    rows = []
    for a in [.01, .1, 1., 10.]:
        for y in [-10., -.3, 0., 2.]:
            u, n = math.atan(y), 10000
            indices = np.arange(-n, n+1)
            summed = a/(math.pi*(1+y*y))*np.sum(1/((u+indices*np.pi)**2+a*a))
            exact = math.tanh(a)/(math.pi*(y*y+math.tanh(a)**2))
            bound = a/(math.pi*(1+y*y))*2/(math.pi**2*(n-.5))
            rows.append(dict(check="branch_sum", a=a, y=y, value=summed, expected=exact,
                             error=abs(summed-exact), bound=bound,
                             passed=bool(abs(summed-exact) <= bound+1e-12)))
    for g in [.001, .01, .1, .5, 1., 10.]:
        value, _ = quad(lambda theta: 2*math.log(math.hypot(1, g*math.tan(theta)))/math.pi,
                        -math.pi/2, math.pi/2, epsabs=1e-11)
        expected = 2*math.log1p(g)
        rows.append(dict(check="cauchy_integral", a=g, value=value, expected=expected,
                         error=abs(value-expected), bound=1e-9, passed=abs(value-expected)<1e-9))
    x = sample(9, 0, 0, 100000, 1.)
    for k in [2, 3, 4]:
        y = np.tan(k*np.arctan(x))
        error = float(np.max(np.abs(np.exp(2j*np.arctan(y))-np.exp(2j*k*np.arctan(x)))))
        ks = boole.cauchy_ks_distance(y, 1.)
        rows.append(dict(check="kfold", a=k, error=error, ks=ks, bound=1e-11,
                         passed=error<1e-11 and ks<math.sqrt(math.log(600)/200000)))
    return rows


def run_a(out: Path, cfg: Dict[str, Any]) -> None:
    """E0Aとその精度監査を実行する。"""
    rows, audits, diagnostics, cdf = [], [], [], []
    gate = math.sqrt(math.log(2*len(cfg["beta"])*len(cfg["scales"])*cfg["a_seeds"]/.01)/(2*cfg["a_n"]))
    condition = 0
    for beta in cfg["beta"]:
        for gamma in cfg["scales"]:
            target = math.tanh(beta*gamma)
            for seed in range(cfg["a_seeds"]):
                original = sample(1, condition, seed, cfg["a_n"], gamma)
                for dtype in ["float64", "float32"]:
                    meta = dict(stage="A", beta=beta, gamma0=gamma, condition=condition, seed=seed, dtype=dtype)
                    states = evolve(original.astype(dtype), np.dtype(dtype).type(beta), 1)
                    candidates = empty_candidates()
                    diagnostics.append(dict(meta, **audit_update(candidates, states, beta, 0, cfg["audit_k"])))
                    audits.extend(finish_audit(candidates, meta))
                    stats = distribution(states[1], target)
                    passed = stats["ks"] is not None and stats["ks"] <= gate
                    rows.append(dict(meta, theoretical_scale=target, gate=gate, **stats,
                                     decision=("整合" if passed else "不一致") if dtype=="float64" else "精度診断"))
                    if seed == 0 and dtype == "float64":
                        sorted_y = np.sort(states[1])
                        for q in np.linspace(.001, .999, 201):
                            j = int(q*(len(sorted_y)-1))
                            cdf.append(dict(beta=beta, gamma0=gamma, x=float(sorted_y[j]),
                                            empirical=(j+1)/len(sorted_y),
                                            theoretical=float(boole.cauchy_cdf(sorted_y[j], target))))
            condition += 1
    for name, data in [("a", rows), ("a_audit", audits), ("a_diagnostics", diagnostics), ("cdf", cdf)]:
        write_csv(out/f"{name}.csv", data)


def run_b(out: Path, cfg: Dict[str, Any]) -> None:
    """状態集団の固定チェックポイントと理論系列を保存する。"""
    rows, audits, diagnostics = [], [], []
    groups = [("wide", cfg["beta"], cfg["scales"], cfg["b_n"], cfg["b_steps"]),
              ("long", cfg["long_beta"], [1.], cfg["long_n"], cfg["long_steps"])]
    condition = 0
    for group, betas, scales, size, steps in groups:
        for beta in betas:
            for gamma in scales:
                theory = scale_series(beta, gamma, steps)
                times = checkpoints(steps)
                for seed in range(cfg["b_seeds"]):
                    meta = dict(stage="B", group=group, beta=beta, gamma0=gamma, condition=condition,
                                seed=seed, dtype="float64")
                    state = sample(2, condition, seed, size, gamma)
                    candidates = empty_candidates()
                    diag = dict(nonfinite=0, zero_count=0, max_abs=float(np.max(np.abs(state))), min_pole=math.pi/2,
                                first_failure_step=None, first_failure_index=None)
                    for checkpoint_index, end in enumerate(times):
                        start = times[checkpoint_index-1] if checkpoint_index else 0
                        for offset in range(start, end, cfg["block"]):
                            block = evolve(state, beta, min(cfg["block"], end-offset))
                            d = audit_update(candidates, block, beta, offset, cfg["audit_k"])
                            for key in ["nonfinite", "zero_count"]:
                                diag[key] += d[key]
                            for key, op in [("max_abs", max), ("min_pole", min)]:
                                if d[key] is not None:
                                    diag[key] = op(diag[key], d[key])
                            if diag["first_failure_step"] is None and d["first_failure_step"] is not None:
                                diag["first_failure_step"], diag["first_failure_index"] = d["first_failure_step"], d["first_failure_index"]
                            state = block[-1].copy()
                        stats = distribution(state, float(theory[end]))
                        estimate = stats["estimated_scale"]
                        error = abs(estimate-theory[end]) if estimate is not None else None
                        rows.append(dict(meta, t=end, theoretical_scale=float(theory[end]), **stats,
                                         absolute_error=error,
                                         floored_error=error/max(theory[end], 1e-12) if error is not None else None))
                    audits.extend(finish_audit(candidates, meta))
                    diagnostics.append(dict(meta, **diag))
                print(f"B {group} beta={beta} gamma={gamma}", flush=True)
                condition += 1
    for name, data in [("b", rows), ("b_audit", audits), ("b_diagnostics", diagnostics)]:
        write_csv(out/f"{name}.csv", data)


def run_theory(out: Path, cfg: Dict[str, Any]) -> None:
    """理論尺度・臨界量・局所緩和を経験系列から分離して保存する。"""
    betas = sorted(set(cfg["beta"] + [round(1+s*e, 8) for e in cfg["epsilon"] for s in [-1, 1]]))
    rows, fixed, relaxation = [], [], []
    for beta in betas:
        target = positive_scale(beta)
        steps = 200000 if beta == 1 else max(4096, math.ceil(20/abs(beta-1)))
        for gamma in cfg["scales"]:
            values = scale_series(beta, gamma, steps)
            passed = np.flatnonzero(np.abs(values-target)/target <= 1e-8) if beta>1 else np.flatnonzero(values<=1e-12)
            hit = int(passed[0]) if passed.size and beta != 1 else None
            for t in checkpoints(steps):
                g = float(values[t])
                rows.append(dict(beta=beta, gamma0=gamma, t=t, gamma=g, fixed=target, convergence_step=hit,
                                 status="代数緩和" if beta==1 else ("整合" if hit is not None else "未収束"),
                                 A=g*math.sqrt(2*t/3) if beta==1 else None,
                                 H=1/g**2 if beta==1 else None,
                                 H_over_t=1/(g*g*t) if beta==1 and t else None,
                                 delta_H=critical_increment(g) if beta==1 else None,
                                 Q=1/g**2-2*t/3 if beta==1 else None))
        if beta > 1:
            lam = lyapunov_theory(beta)
            fixed.append(dict(beta=beta, epsilon=beta-1, gamma=target, lyapunov=lam,
                              R_gamma=target/math.sqrt(3*(beta-1)), R_lambda=lam/(2*math.sqrt(3*(beta-1)))))
            multiplier = beta*(1-target*target)
            trajectory = scale_series(beta, target*(1+1e-3), steps)
            delta = trajectory-target
            mask = (delta/target <= 1e-3) & (delta/target >= 1e-6)
            indices = np.flatnonzero(mask)
            slope = float(np.polyfit(indices, np.log(delta[indices]), 1)[0]) if len(indices)>2 else None
            relaxation.append(dict(beta=beta, multiplier=multiplier, tau=-1/math.log(multiplier),
                                   tau_empirical=-1/slope if slope is not None else None,
                                   selected_points=len(indices), asymptotic_tau=1/(2*(beta-1))))
    write_csv(out/"theory.csv", rows)
    write_csv(out/"fixed.csv", fixed)
    write_csv(out/"relaxation.csv", relaxation)


def run_c(out: Path, cfg: Dict[str, Any]) -> None:
    """E0C/Eの同一軌道から窓別統計・対数感度・監査を保存する。"""
    rows, audits, diagnostics, initial_rows = [], [], [], []
    representatives = {}
    betas = sorted(set([b for b in cfg["beta"] if b>1] + [round(1+e, 8) for e in cfg["epsilon"]]))
    for condition, beta in enumerate(betas):
        gamma = positive_scale(beta)
        for seed in range(cfg["c_seeds"]):
            original = sample(3, condition, seed, 1, gamma)
            for dtype in (["float64", "float32"] if beta in cfg["fp32_beta"] else ["float64"]):
                meta = dict(stage="C", beta=beta, gamma0=gamma, condition=condition, seed=seed, dtype=dtype)
                initial_rows.append(dict(meta, initial=float(original[0]), cast_initial=float(original.astype(dtype)[0])))
                states = evolve(original.astype(dtype), np.dtype(dtype).type(beta), cfg["c_steps"])
                candidates = empty_candidates()
                diagnostics.append(dict(meta, **audit_update(candidates, states, beta, 0, cfg["audit_k"])))
                audits.extend(finish_audit(candidates, meta))
                values = states[1:, 0].astype(float)
                logd = math.log(beta)+2*np.log(np.hypot(1, values))
                windows = [(0, len(values))] + [(s, min(s+cfg["window"], len(values))) for s in range(0, len(values), cfg["window"])]
                for start, stop in windows:
                    stats = distribution(values[start:stop], gamma)
                    lam = float(np.mean(logd[start:stop])) if np.isfinite(logd[start:stop]).all() else None
                    rows.append(dict(meta, start=start, stop=stop, **stats, theoretical_scale=gamma,
                                     lyapunov=lam, theoretical_lyapunov=lyapunov_theory(beta),
                                     log_sensitivity=float(np.sum(logd[start:stop])) if lam is not None else None))
                if seed == 0:
                    representatives[f"b{beta}_{dtype}"] = states[:, 0]
        print(f"C beta={beta}", flush=True)
    for name, data in [("c", rows), ("c_audit", audits), ("c_diagnostics", diagnostics), ("initial_states", initial_rows)]:
        write_csv(out/f"{name}.csv", data)
    np.savez_compressed(out/"representative_orbits.npz", **representatives)


def summarize(out: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """seed単位の近似区間と事前範囲fitを保存CSVだけから再生成する。"""
    c = read_csv(out/"c.csv")
    full = [r for r in c if int(r["start"])==0 and int(r["stop"])==cfg["c_steps"]]
    aggregates = []
    for beta, dtype in sorted({(float(r["beta"]), r["dtype"]) for r in full}):
        selected = [r for r in full if float(r["beta"])==beta and r["dtype"]==dtype]
        for metric in ["lyapunov", "estimated_scale", "ks"]:
            if any(not r[metric] for r in selected):
                aggregates.append(dict(beta=beta, dtype=dtype, metric=metric, status="数値精度上の制約"))
                continue
            values = np.array([float(r[metric]) for r in selected])
            mean = float(np.mean(values))
            half = float(student_t.ppf(.975, len(values)-1)*np.std(values, ddof=1)/math.sqrt(len(values)))
            theoretical = lyapunov_theory(beta) if metric=="lyapunov" else (positive_scale(beta) if metric=="estimated_scale" else None)
            aggregates.append(dict(beta=beta, dtype=dtype, metric=metric, n=len(values), mean=mean,
                                   low=mean-half, high=mean+half, theoretical=theoretical,
                                   error=mean-theoretical if theoretical is not None else None,
                                   status="記述的評価" if theoretical is None else ("整合" if mean-half<=theoretical<=mean+half else "不一致（有限時間診断）")))
    write_csv(out/"aggregate.csv", aggregates)
    fits = []
    for cutoff in cfg["fit_cutoffs"]:
        eps = np.array([e for e in cfg["epsilon"] if e <= cutoff])
        for source in ["theory", "orbit"]:
            for metric in ["estimated_scale", "lyapunov"]:
                if source == "theory":
                    y = [positive_scale(1+e) if metric=="estimated_scale" else lyapunov_theory(1+e) for e in eps]
                else:
                    matching = [next(r for r in aggregates if r["dtype"]=="float64" and r["metric"]==metric and abs(r["beta"]-(1+e))<1e-12) for e in eps]
                    if any("mean" not in r or r["mean"] <= 0 for r in matching):
                        fits.append(dict(source=source, metric=metric, cutoff=cutoff, status="数値精度上の制約"))
                        continue
                    y = [r["mean"] for r in matching]
                slope, intercept = np.polyfit(np.log(eps), np.log(y), 1)
                fits.append(dict(source=source, metric=metric, cutoff=cutoff, n=len(eps), slope=float(slope), intercept=float(intercept), status="記述的評価"))
    write_csv(out/"fits.csv", fits)
    a = read_csv(out/"a.csv")
    primary = [r for r in a if r["dtype"]=="float64"]
    diagnostics = [r for stage in "abc" for r in read_csv(out/f"{stage}_diagnostics.csv")]
    return dict(status="completed", a_passed=sum(r["decision"]=="整合" for r in primary), a_total=len(primary),
                nonfinite_runs=sum(int(r["nonfinite"])>0 for r in diagnostics),
                orbit_diagnostics=aggregates, fits=fits,
                scope="中心Cauchy閉包と有限時間数値統計。同期・圧縮・学習・エルゴード性の証明ではない。")


def plot_results(out: Path) -> None:
    """保存CSVから主図6枚と監査補足を再描画する。"""
    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(out/f"{name}.png", dpi=160)
        plt.close()
    cdf = read_csv(out/"cdf.csv")
    plt.figure(figsize=(7, 4))
    for beta in [.5, 1., 2.]:
        r = [r for r in cdf if float(r["beta"])==beta and float(r["gamma0"])==1]
        x = [float(v["x"]) for v in r]
        plt.plot(x, [float(v["theoretical"]) for v in r], label=f"theory beta={beta}")
        plt.plot(x[::8], [float(v["empirical"]) for v in r][::8], ".")
    plt.xlim(-5, 5); plt.xlabel("x"); plt.ylabel("CDF"); plt.legend(); save("fig1_cauchy_cdf")
    theory, b = read_csv(out/"theory.csv"), read_csv(out/"b.csv")
    plt.figure(figsize=(7, 4))
    for beta in [.9999, 1., 1.0001]:
        r = [r for r in theory if float(r["beta"])==beta and float(r["gamma0"])==1 and int(r["t"])>0]
        plt.loglog([int(v["t"]) for v in r], [float(v["gamma"]) for v in r], label=f"theory {beta}")
        r = [r for r in b if r["group"]=="long" and float(r["beta"])==beta and int(r["seed"])==0 and int(r["t"])>0 and r["estimated_scale"]]
        plt.loglog([int(v["t"]) for v in r], [float(v["estimated_scale"]) for v in r], ".")
    plt.xlabel("t"); plt.ylabel("scale (dots: seed 0 ensemble)"); plt.legend(); save("fig2_scale_dynamics")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for gamma in [.1, 1., 10.]:
        r = [r for r in theory if float(r["beta"])==1 and float(r["gamma0"])==gamma and int(r["t"])>0]
        for ax, field, target in zip(axes, ["A", "H_over_t", "delta_H"], [1, 2/3, 2/3]):
            ax.semilogx([int(v["t"]) for v in r], [float(v[field]) for v in r], label=f"gamma0={gamma}")
            ax.axhline(target, color="gray", linestyle=":"); ax.set_xlabel("t"); ax.set_ylabel(field)
    axes[0].legend(); save("fig3_critical_relaxation")
    fixed = [r for r in read_csv(out/"fixed.csv") if float(r["epsilon"])<=.100000001]
    eps = np.array([float(r["epsilon"]) for r in fixed])
    agg = read_csv(out/"aggregate.csv")
    plt.figure(figsize=(7, 4))
    plt.loglog(eps, [float(r["gamma"]) for r in fixed], label="fixed point")
    plt.loglog(eps, np.sqrt(3*eps), "--", label="sqrt(3 epsilon)")
    r = [r for r in agg if r["dtype"]=="float64" and r["metric"]=="estimated_scale" and float(r["beta"])<=1.100000001 and r.get("mean")]
    plt.loglog([float(v["beta"])-1 for v in r], [float(v["mean"]) for v in r], ".", label="orbit mean")
    plt.xlabel("beta - 1"); plt.ylabel("scale"); plt.legend(); save("fig4_scale_emergence")
    plt.figure(figsize=(7, 4))
    for field in ["R_gamma", "R_lambda"]:
        plt.semilogx(eps, [float(r[field]) for r in fixed], label=field+" theory")
    for metric, factor, label in [("estimated_scale", 1, "R_gamma orbit"), ("lyapunov", 2, "R_lambda orbit")]:
        r = [r for r in agg if r["dtype"]=="float64" and r["metric"]==metric and float(r["beta"])<=1.100000001 and r.get("mean")]
        plt.semilogx([float(v["beta"])-1 for v in r], [float(v["mean"])/(factor*math.sqrt(3*(float(v["beta"])-1))) for v in r], ".", label=label)
    plt.axhline(1, color="gray", linestyle=":"); plt.xlabel("beta - 1"); plt.ylabel("critical ratio"); plt.legend(); save("fig5_critical_ratios")
    plt.figure(figsize=(7, 4))
    for dtype in ["float64", "float32"]:
        r = [r for r in agg if r["dtype"]==dtype and r["metric"]=="lyapunov" and r.get("mean")]
        plt.errorbar([float(v["beta"]) for v in r], [float(v["mean"]) for v in r],
                     yerr=[float(v["high"])-float(v["mean"]) for v in r], fmt=".", label=dtype+" seed 95% CI")
    x = sorted({float(r["beta"]) for r in agg})
    plt.plot(x, [lyapunov_theory(v) for v in x], label="theory")
    plt.xlabel("beta"); plt.ylabel("Lyapunov"); plt.legend(); save("fig6_lyapunov")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dtype in ["float32", "float64"]:
        r = [r for r in read_csv(out/"c_audit.csv") if r["dtype"]==dtype]
        axes[0].hist([math.log10(max(float(v["scaled_error"]), 1e-30)) for v in r], bins=40, alpha=.5, label=dtype)
        axes[1].hist([math.log10(max(float(v["pole_distance_mp"]), 1e-100)) for v in r], bins=40, alpha=.5, label=dtype)
    axes[0].set_xlabel("log10 scaled one-step error"); axes[1].set_xlabel("log10 audited pole distance")
    axes[0].legend(); save("supplement_numerical_audit")


def prepare(root: Path) -> None:
    """実験前設定を固定し、別条件による上書きを拒否する。"""
    root.mkdir(parents=True, exist_ok=True)
    path = root/"config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != configuration():
            raise ValueError("既存設定が異なります")
    else:
        write_json(path, configuration())


def benchmark(root: Path) -> Dict[str, float]:
    """本実験と異なる固定入力でJIT後の短い性能測定を行う。"""
    path = root/"benchmark.json"
    if path.exists():
        raise FileExistsError(path)
    state = np.linspace(-.3, .3, 512)
    evolve(state, 1.0001, 2)
    started = time.perf_counter()
    candidates = empty_candidates()
    for offset in range(0, 4096, 1024):
        block = evolve(state, 1.0001, 1024)
        audit_update(candidates, block, 1.0001, offset, 100)
        state = block[-1].copy()
    elapsed = time.perf_counter()-started
    result = dict(seconds=elapsed, evaluations=512*4096,
                  estimated_long_ensemble_seconds=elapsed*(200000/4096)*15,
                  includes_jit=False, includes_high_precision=False)
    write_json(path, result)
    return result


def run(root: Path) -> Dict[str, Any]:
    """事前設定を凍結したE0全条件を上書きなしで実行する。"""
    prepare(root)
    if not (root/"benchmark.json").exists():
        raise FileNotFoundError("--benchmarkを先に実行してください")
    out = root/"artifacts"
    out.mkdir(exist_ok=False)
    cfg = json.loads((root/"config.json").read_text(encoding="utf-8"))
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    environment = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                       scipy=scipy.__version__, numba=numba.__version__, mpmath=mp.__version__,
                       matplotlib=matplotlib.__version__, git_head=git,
                       script_sha256=sha(Path(__file__)), config_sha256=sha(root/"config.json"),
                       reused_boole_sha256=sha(BOOLE), command=sys.argv,
                       started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(root/"environment.json", environment)
    started = time.perf_counter()
    checks = theory_checks()
    write_csv(out/"theory_checks.csv", checks)
    if not all(r["passed"] for r in checks):
        raise ArithmeticError("理論前提の独立照合が失敗")
    run_theory(out, cfg)
    for stage, func in [("A", run_a), ("B", run_b), ("C/E/F", run_c)]:
        print(f"START {stage}", flush=True)
        func(out, cfg)
        print(f"DONE {stage}: elapsed {time.perf_counter()-started:.1f}s", flush=True)
    metrics = summarize(out, cfg)
    metrics["elapsed_seconds"] = time.perf_counter()-started
    plot_results(out)
    write_json(root/"metrics.json", metrics)
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix in {".csv", ".npz", ".png", ".json"} and p.name not in {"sha256.json", "validation.json"}]
    write_json(out/"sha256.json", {p.relative_to(root).as_posix(): sha(p) for p in paths})
    return metrics


def validate(root: Path) -> Dict[str, Any]:
    """ハッシュ、件数、代表軌道の保存値から集計を独立に確認する。"""
    out = root/"artifacts"
    checks = []
    for relative, digest in json.loads((out/"sha256.json").read_text()).items():
        checks.append(dict(check="sha256 "+relative, passed=sha(root/relative)==digest))
    cfg = json.loads((root/"config.json").read_text())
    checks.append(dict(check="A primary count", passed=len([r for r in read_csv(out/"a.csv") if r["dtype"]=="float64"])==210))
    checks.append(dict(check="A precision count", passed=len(read_csv(out/"a.csv"))==420))
    for stage, count in [("a", 420), ("b", 120), ("c", 300)]:
        checks.append(dict(check=stage+" diagnostic count", passed=len(read_csv(out/f"{stage}_diagnostics.csv"))==count))
        checks.append(dict(check=stage+" audit count", passed=len(read_csv(out/f"{stage}_audit.csv"))==count*200))
    expected_b = 105*len(checkpoints(cfg["b_steps"]))+15*len(checkpoints(cfg["long_steps"]))
    checks.append(dict(check="B checkpoint count", passed=len(read_csv(out/"b.csv"))==expected_b))
    rows = read_csv(out/"c.csv")
    checks.append(dict(check="C window count", passed=len(rows)==1500))
    for row in read_csv(out/"a.csv"):
        if row["dtype"]=="float64":
            expected = "整合" if row["ks"] and float(row["ks"])<=float(row["gate"]) else "不一致"
            checks.append(dict(check="A gate "+row["condition"]+"/"+row["seed"], passed=row["decision"]==expected))
    for row in read_csv(out/"aggregate.csv"):
        selected = [r for r in rows if r["beta"]==row["beta"] and r["dtype"]==row["dtype"] and int(r["start"])==0 and int(r["stop"])==cfg["c_steps"]]
        field = row["metric"]
        if row.get("mean"):
            values = np.array([float(r[field]) for r in selected])
            mean = float(np.sum(values)/len(values))
            half = float(student_t.ppf(.975, len(values)-1)*np.std(values, ddof=1)/math.sqrt(len(values)))
            checks.append(dict(check="aggregate "+row["beta"]+"/"+row["dtype"]+"/"+field,
                               passed=abs(mean-float(row["mean"]))<1e-12 and abs(mean+half-float(row["high"]))<1e-12))
    with np.load(out/"representative_orbits.npz") as archive:
        for key in archive.files:
            beta_text, dtype = key[1:].split("_")
            beta = float(beta_text)
            trajectory = archive[key].astype(float)
            found = next(r for r in rows if float(r["beta"])==beta and r["dtype"]==dtype and int(r["seed"])==0 and int(r["start"])==0 and int(r["stop"])==cfg["c_steps"])
            if np.isfinite(trajectory).all():
                lam = float(np.mean(np.log(beta)+np.logaddexp(0., 2*np.log(np.abs(trajectory[1:])))))
                q = np.quantile(trajectory[1:], [.25, .75])
                checks.append(dict(check="orbit "+key, passed=bool(abs(lam-float(found["lyapunov"]))<1e-12 and abs((q[1]-q[0])/2-float(found["estimated_scale"]))<1e-12)))
            else:
                checks.append(dict(check="failed orbit "+key, passed=not found["lyapunov"]))
    env = json.loads((root/"environment.json").read_text())
    snapshot = out/"executed_source.py"
    if snapshot.exists():
        import ast
        executed_ast = ast.parse(snapshot.read_text(encoding="utf-8"))
        current_ast = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for tree in [executed_ast, current_ast]:
            tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name=="validate")]
        checks.append(dict(check="executed source hash", passed=sha(snapshot)==env["script_sha256"]))
        checks.append(dict(check="only validator changed since execution", passed=ast.dump(executed_ast)==ast.dump(current_ast)))
    else:
        checks.append(dict(check="script unchanged since run", passed=sha(Path(__file__))==env["script_sha256"]))
    checks.append(dict(check="reused Boole unchanged", passed=sha(BOOLE)==env["reused_boole_sha256"]))
    result = dict(passed=all(r["passed"] for r in checks), checks=checks)
    write_json(root/"validation.json", result)
    return result


def main() -> None:
    """準備・性能測定・本実験・保存物検証を明示的に選択する。"""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ["prepare", "benchmark", "run", "validate"]:
        modes.add_argument("--"+name, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output)
    elif args.benchmark:
        print(json.dumps(benchmark(args.output)))
    elif args.run:
        result = run(args.output)
        print(json.dumps({k: result[k] for k in ["status", "a_passed", "a_total", "nonfinite_runs", "elapsed_seconds"]}))
    else:
        result = validate(args.output)
        print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]))))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
