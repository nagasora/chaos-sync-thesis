"""独立ensembleで状態同期とCauchy周辺分布を区別し、保存値から再検証する。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
CONFIG = dict(betas=[1.01, 1.1, 1.5, 2.0], seeds=[810, 811, 812],
              couplings=[0.0, 0.25, 0.5], n=20000, steps=16,
              dkw_alpha=0.01, hoeffding_alpha=0.01, precision_digits=80,
              dkw_tests=1272, distance_tests=204, dtype="float64")


def dump(path: Path, value: Any) -> None:
    """NaNを成功値として保存しないJSON出力。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    """保存コードと成果物のSHA-256を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scale_fixed(beta: float) -> float:
    """ゼロ根を除外して正のCauchy固定尺度を求める。"""
    if not math.isfinite(beta) or beta <= 1:
        raise ValueError("beta>1 is required")
    return brentq(lambda g: math.atanh(g)/g-beta, 1e-7, np.nextafter(1., 0.))


def advance(state: np.ndarray, beta: float, coupling: float) -> np.ndarray:
    """同時更新でtan出力拡散を計算し、非有限値を隠さない。"""
    mapped = np.tan(beta*state)
    result = (1-coupling)*mapped+coupling*mapped[:, ::-1]
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite tan state")
    return result


def cayley(state: np.ndarray, gamma: float) -> np.ndarray:
    """裾での二乗誤差の代わりに有界の円周座標へ写す。"""
    return np.exp(1j*(2*np.arctan2(state, gamma)-np.pi))


def cauchy_ks(values: np.ndarray, parameter: complex) -> float:
    """既知位置尺度との両側経験CDF距離を計算する。"""
    if parameter.imag <= 0 or not np.isfinite(values).all():
        raise ValueError("finite data and positive scale required")
    cdf = .5+np.arctan((np.sort(values)-parameter.real)/parameter.imag)/np.pi
    n = len(values)
    return float(max(np.max(np.arange(1, n+1)/n-cdf), np.max(cdf-np.arange(n)/n)))


def measure(state: np.ndarray, prediction: np.ndarray, beta: float, gamma: float,
            coupling: float, regime: str, seed: int, step: int) -> List[Dict[str, Any]]:
    """各時刻の有界距離と分位点を保存し、未証明閉包を区別する。"""
    q = cayley(state, gamma)
    distance = float(np.mean(abs(q[:, 0]-q[:, 1])**2))
    pair_ks = float(ks_2samp(state[:, 0], state[:, 1], method="asymp").statistic)
    supported = coupling in [0., .5] or step <= 1
    rows = []
    for node in range(2):
        q25, median, q75 = np.quantile(state[:, node], [.25, .5, .75])
        expected = prediction[node] if supported else 1j*gamma
        rows.append(dict(beta=beta, regime=regime, seed=seed, coupling=coupling, step=step,
                         node=node, closure_supported=supported,
                         reference_mu=float(expected.real), reference_gamma=float(expected.imag),
                         ks_reference=cauchy_ks(state[:, node], expected),
                         ks_stationary=cauchy_ks(state[:, node], 1j*gamma),
                         median=float(median), half_iqr=float((q75-q25)/2),
                         paired_distance=distance, pair_ks=pair_ks,
                         exact_equal=int(np.count_nonzero(state[:, 0] == state[:, 1]))))
    return rows


def precise_step(pair: np.ndarray, beta: float, coupling: float) -> Dict[str, str]:
    """同じbinary64入力から80桁で一段更新し、丸め一致と区別する。"""
    with mp.workdps(CONFIG["precision_digits"]):
        x, y = (mp.mpf(float(v)) for v in pair)
        b, k = mp.mpf(beta), mp.mpf(coupling)
        fx, fy = mp.tan(b*x), mp.tan(b*y)
        gap = ((1-k)*fx+k*fy)-(k*fx+(1-k)*fy)
        return dict(previous_x=repr(float(pair[0])), previous_y=repr(float(pair[1])),
                    high_precision_gap=mp.nstr(gap, 70))


def parameter_path(parameters: np.ndarray, beta: float, coupling: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """分布パラメータを80桁で反復し、実軌道とは別に保存する。"""
    values, records = [], []
    with mp.workdps(CONFIG["precision_digits"]):
        w = [mp.mpc(float(z.real), float(z.imag)) for z in parameters]
        b, k = mp.mpf(beta), mp.mpf(coupling)
        fp = parameters.copy()
        for step in range(CONFIG["steps"]+1):
            # 未証明の依存Cauchy和を解析oracleとして延長しない。
            if coupling == .25 and step >= 2:
                values.append([0j, 0j])
                continue
            values.append([complex(z) for z in w])
            for node, z in enumerate(w):
                records.append(dict(step=step, node=node, mu=mp.nstr(z.real, 70),
                                    gamma=mp.nstr(z.imag, 70),
                                    float64_parameter_error=float(abs(complex(z)-fp[node]))))
            mapped = [mp.tan(b*z) for z in w]
            w = [(1-k)*mapped[0]+k*mapped[1], k*mapped[0]+(1-k)*mapped[1]]
            mf = np.tan(beta*fp)
            fp = (1-coupling)*mf+coupling*mf[::-1]
    return np.array(values), records


def simulate(initial: np.ndarray, parameters: np.ndarray, beta: float, gamma: float,
             coupling: float, regime: str, seed: int) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """初期値から時刻別観測と丸め診断を再現する。"""
    state = initial.copy()
    predictions, _ = parameter_path(parameters, beta, coupling)
    rows, diagnostics = [], []
    for step in range(CONFIG["steps"]+1):
        rows.extend(measure(state, predictions[step], beta, gamma, coupling, regime, seed, step))
        if step == CONFIG["steps"]:
            break
        updated = advance(state, beta, coupling)
        collisions = np.flatnonzero((updated[:, 0] == updated[:, 1]) & (state[:, 0] != state[:, 1]))
        if len(collisions):
            record = dict(beta=beta, regime=regime, seed=seed, coupling=coupling,
                          step=step+1, collision_count=len(collisions),
                          representative_pair=int(collisions[0]))
            record.update(precise_step(state[collisions[0]], beta, coupling))
            diagnostics.append(record)
        state = updated
    return state, rows, diagnostics


def summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """定理照合、数値同期、未知閉包の診断を別々に集計する。"""
    asserted = [r for r in rows if r["closure_supported"]]
    distances = [r for r in rows if r["regime"] == "stationary" and r["coupling"] == 0 and r["node"] == 0]
    if len(asserted) != CONFIG["dkw_tests"] or len(distances) != CONFIG["distance_tests"]:
        raise AssertionError("preregistered test count mismatch")
    eps = math.sqrt(math.log(2*len(asserted)/CONFIG["dkw_alpha"])/(2*CONFIG["n"]))
    eta = math.sqrt(8*math.log(2*len(distances)/CONFIG["hoeffding_alpha"])/CONFIG["n"])
    half = [r for r in rows if r["coupling"] == .5 and r["step"] > 0]
    unknown = [r for r in rows if not r["closure_supported"]]
    return dict(status="completed", cases=48, rows=len(rows), dkw_tests=len(asserted),
                dkw_epsilon=eps, dkw_exceedances=sum(r["ks_reference"] > eps for r in asserted),
                dkw_max_ks=max(r["ks_reference"] for r in asserted),
                distance_tests=len(distances), distance_epsilon=eta,
                distance_exceedances=sum(abs(r["paired_distance"]-2) > eta for r in distances),
                uncoupled_distance_range=[min(r["paired_distance"] for r in distances), max(r["paired_distance"] for r in distances)],
                half_all_exact=all(r["exact_equal"] == CONFIG["n"] for r in half),
                half_max_distance=max(r["paired_distance"] for r in half),
                quarter_unknown_max_ks=max(r["ks_reference"] for r in unknown),
                nonhalf_equal_rows=sum(r["exact_equal"] > 0 for r in rows if r["coupling"] < .5 and r["node"] == 0))


def compute(root: Path, replay: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """固定seedのensemble生成、または保存初期値による再実行を行う。"""
    source = np.load(root/"artifacts/ensembles.npz") if replay else None
    arrays, rows, diagnostics, oracles = {}, [], [], {}
    for bi, beta in enumerate(CONFIG["betas"]):
        gamma = scale_fixed(beta)
        for ri, regime in enumerate(["stationary", "heterogeneous"]):
            parameters = np.array([1j*gamma, 1j*gamma] if ri == 0 else [-.5+.6j*gamma, .5+1.4j*gamma])
            for seed in CONFIG["seeds"]:
                key = f"b{bi}_r{ri}_s{seed}"
                rng = np.random.default_rng(np.random.SeedSequence([seed, bi, ri]))
                initial = source[key] if replay else parameters.real+parameters.imag*rng.standard_cauchy((CONFIG["n"], 2))
                arrays[key] = initial
                for coupling in CONFIG["couplings"] if ri == 0 else [0.]:
                    _, oracle = parameter_path(parameters, beta, coupling)
                    oracles[f"b{bi}_r{ri}_k{coupling}"] = oracle
                    final, case_rows, case_diagnostics = simulate(initial, parameters, beta, gamma, coupling, regime, seed)
                    endkey = key+f"_k{coupling}_end"
                    if replay and not np.array_equal(final, source[endkey]):
                        raise AssertionError(f"terminal replay differs: {endkey}")
                    arrays[endkey] = final
                    rows.extend(case_rows)
                    diagnostics.extend(case_diagnostics)
    if not replay:
        np.savez_compressed(root/"artifacts/ensembles.npz", **arrays)
        dump(root/"artifacts/oracle.json", oracles)
    else:
        assert oracles == json.loads((root/"artifacts/oracle.json").read_text(encoding="utf-8"))
    if source is not None:
        source.close()
    return rows, diagnostics


def plot(root: Path, rows: List[Dict[str, Any]]) -> None:
    """CSVと同じ観測から状態距離と周辺KSを可視化する。"""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for col, beta in enumerate(CONFIG["betas"]):
        for coupling in CONFIG["couplings"]:
            data = [r for r in rows if r["beta"] == beta and r["regime"] == "stationary" and r["coupling"] == coupling and r["node"] == 0]
            for ax, metric in zip(axes[:, col], ["paired_distance", "ks_stationary"]):
                values = np.array([[r[metric] for r in data if r["step"] == t] for t in range(CONFIG["steps"]+1)])
                ax.plot(values.mean(axis=1), label=f"k={coupling}")
                ax.fill_between(np.arange(len(values)), values.min(axis=1), values.max(axis=1), alpha=.12)
            axes[0, col].set_title(f"beta={beta}")
        axes[0, col].set_ylim(-.05, 2.2)
        axes[1, col].set_xlabel("time")
    axes[0, 0].set_ylabel("Paired Cayley distance")
    axes[1, 0].set_ylabel("KS to stationary Cauchy")
    axes[0, -1].legend()
    fig.savefig(root/"artifacts/state_distribution.png", dpi=140)
    fig.savefig(root/"artifacts/state_distribution.pdf")
    plt.close(fig)


def main() -> None:
    """本実験は上書きせず、検証は保存初期値とhashから実行する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "validate"])
    args = parser.parse_args()
    if args.action == "validate":
        hashes = json.loads((HERE/"sha256.json").read_text(encoding="utf-8"))
        for relative, expected in hashes.items():
            if digest(HERE/relative) != expected:
                raise AssertionError(f"hash mismatch: {relative}")
        rows, diagnostics = compute(HERE, replay=True)
        stored = json.loads((HERE/"artifacts/observations.json").read_text(encoding="utf-8"))
        assert rows == stored
        assert diagnostics == json.loads((HERE/"artifacts/precision.json").read_text(encoding="utf-8"))
        assert summary(rows) == json.loads((HERE/"metrics.json").read_text(encoding="utf-8"))
        result = dict(passed=True, hashed_files=len(hashes), replayed_cases=48, observation_rows=len(rows),
                      precision_records=len(diagnostics), finite_all=True)
        dump(HERE/"validation.json", result)
        print(json.dumps(result))
        return
    if (HERE/"metrics.json").exists() or (HERE/"artifacts/ensembles.npz").exists():
        raise FileExistsError("existing run is never overwritten")
    (HERE/"artifacts").mkdir(exist_ok=True)
    dump(HERE/"config.json", CONFIG)
    dump(HERE/"environment.json", dict(python=platform.python_version(), numpy=np.__version__,
          scipy=scipy.__version__, mpmath=mp.__version__, platform=platform.platform()))
    rows, diagnostics = compute(HERE)
    dump(HERE/"artifacts/observations.json", rows)
    with (HERE/"artifacts/observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(HERE/"artifacts/precision.json", diagnostics)
    result = summary(rows)
    dump(HERE/"metrics.json", result)
    plot(HERE, rows)
    paths = [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in ["README.md", "sha256.json", "validation.json"]]
    dump(HERE/"sha256.json", {p.relative_to(HERE).as_posix(): digest(p) for p in paths})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
