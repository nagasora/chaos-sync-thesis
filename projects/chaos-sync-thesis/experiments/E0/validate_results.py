"""E0保存軌道を実験実装と独立な式・積分・再集計で監査する。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import zipfile
import scipy
from pathlib import Path
from typing import Dict
import numpy as np
from scipy.integrate import quad


def validate(out: Path) -> Dict:
    """全軌道の一ステップ式、尺度・指数、周期、集計・ハッシュを検証する。"""
    config = json.loads((out/"config.json").read_text(encoding="utf-8"))
    if config['experiment_id'] == 'E0B-KFOLD-TM-SHIFT-CHECK':
        return validate_e0b(out)
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    env = json.loads((out/"environment.json").read_text(encoding="utf-8"))
    with (out/"metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (out/"conditions.csv").open(encoding="utf-8") as f:
        conditions = list(csv.DictReader(f))
    checks = {}
    expected = len(config["alpha_values"])*len(config["seeds"])*len(config["initial_distributions"])
    checks["row_count"] = len(rows) == expected*len(config["precision"])*len(config["observation_lengths"])
    checks["unique_conditions"] = len({(r["dtype"],r["alpha"],r["seed"],r["initial_distribution"],r["length"]) for r in rows}) == len(rows)
    checks["condition_grid"] = {(float(r["alpha"]),int(r["seed"]),r["initial_distribution"]) for r in conditions} == {
        (a,s,d) for a in config["alpha_values"] for s in config["seeds"] for d in config["initial_distributions"]}
    theory = []
    cycles = []
    for alpha in config["alpha_values"]:
        gamma = np.sqrt(alpha/(1-alpha))
        integral,error = quad(lambda theta: (np.log(alpha)+np.log1p((np.tan(theta)/gamma)**2))*2/np.pi,0,np.pi/2,epsabs=1e-10)
        closed = np.log1p(2*np.sqrt(alpha*(1-alpha)))
        checks[f"quadrature_{alpha}"] = abs(integral-closed) < 1e-8
        theory.append(dict(alpha=alpha,gamma=float(gamma),closed_form=float(closed),quadrature=integral,quadrature_error=error))
    for dtype in config["precision"]:
        data = np.load(out/f"orbits_{dtype}.npz")
        orbit = data["orbit"]
        checks[f"shape_dtype_{dtype}"] = orbit.shape == (expected,max(config["observation_lengths"])) and orbit.dtype.name == dtype
        a = np.array([float(r["alpha"]) for r in conditions],dtype=dtype)
        x = np.array([float(r["initial_state"]) for r in conditions],dtype=dtype)
        near = np.zeros((len(x),len(config["near_zero_thresholds"])),dtype=np.int64)
        failures = np.full(len(x),-1,dtype=np.int64)
        nf = np.zeros(len(x),dtype=np.int64)
        active = np.ones(len(x),dtype=bool)
        match = True
        # 共通モジュールをimportせず、保存初期値から全軌道を再演算する。
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            for step in range(config["burn_in"]+orbit.shape[1]):
                idx = np.flatnonzero(active)
                near[idx] += np.abs(x[idx,None]).astype(np.float64) < np.array(config["near_zero_thresholds"])
                reciprocal = np.divide(np.ones(len(idx),dtype=dtype),x[idx])
                x[idx] = np.multiply(a[idx],np.subtract(x[idx],reciprocal))
                bad = ~np.isfinite(x[idx]) | (x[idx] == 0)
                nf[idx] += ~np.isfinite(x[idx])
                failures[idx[bad]] = step+1
                active[idx[bad]] = False
                if step >= config["burn_in"]:
                    observed = orbit[:,step-config["burn_in"]]
                    expected_col = np.full(len(x),np.nan,dtype=dtype)
                    expected_col[idx] = x[idx]
                    match = match and np.array_equal(observed,expected_col,equal_nan=True)
        checks[f"independent_exact_replay_{dtype}"] = match
        checks[f"diagnostic_counts_{dtype}"] = bool(np.array_equal(near,data["near_zero_counts"]) and np.array_equal(failures,data["failure_step"]) and np.array_equal(nf,data["nonfinite_update_count"]))
        metric_ok = True
        period_ok = True
        for r in [r for r in rows if r["dtype"] == dtype]:
            values = orbit[int(r["condition_index"]),:int(r["length"])].astype(np.float64)
            complete = bool(np.all(np.isfinite(values) & (values != 0)))
            metric_ok &= complete == (r["completed"] == "True")
            if complete:
                alpha = float(r["alpha"])
                gamma = np.sqrt(alpha/(1-alpha))
                med = np.quantile(values,.5)
                mad = np.quantile(np.abs(values-med),.5)
                lyap = np.mean(np.log(alpha*(1+1/values**2)))
                targets = dict(median=med,mad=mad,half_iqr=(np.quantile(values,.75)-np.quantile(values,.25))/2,
                    mad_absolute_error=abs(mad-gamma),mad_relative_error=abs(mad/gamma-1),
                    estimated_lyapunov=lyap,lyapunov_absolute_error=abs(lyap-np.log1p(2*np.sqrt(alpha*(1-alpha)))),
                    sample_mean=np.mean(values),sample_std=np.std(values))
                metric_ok &= all(np.isclose(float(r[k]),v,rtol=1e-11,atol=1e-12) for k,v in targets.items())
                unique, first, counts = np.unique(values,return_index=True,return_counts=True)
                has_repeat = np.any(counts>1)
                period = int(r["period"])
                period_ok &= has_repeat == (period>0)
                if period:
                    start,end = int(r["first_repeat_start"]),int(r["first_repeat_end"])
                    suffix_matches = np.array_equal(values[start:-period],values[end:])
                    period_ok &= end-start == period and end < len(values) and suffix_matches
                    if int(r["length"]) == max(config["observation_lengths"]):
                        cycle = values[start:end]
                        cycle_mad = float(np.median(np.abs(cycle-np.median(cycle))))
                        cycles.append(dict(condition_index=int(r["condition_index"]),dtype=dtype,
                            alpha=float(r["alpha"]),seed=int(r["seed"]),initial_distribution=r["initial_distribution"],
                            period=period,start=start,exact_suffix_matches=bool(suffix_matches),
                            cycle_mad=cycle_mad,cycle_mad_relative_error=abs(cycle_mad/np.sqrt(float(r["alpha"])/(1-float(r["alpha"])))-1),
                            cycle_sha256=hashlib.sha256(np.sort(cycle).tobytes()).hexdigest()))
        checks[f"independent_metrics_{dtype}"] = bool(metric_ok)
        checks[f"exact_period_suffix_{dtype}"] = bool(period_ok)
    for group in summary["groups"]:
        subset = [r for r in rows if r["dtype"] == group["dtype"] and float(r["alpha"]) == group["alpha"]
                  and r["initial_distribution"] == group["initial_distribution"] and int(r["length"]) == max(config["observation_lengths"])]
        complete = all(r["completed"] == "True" for r in subset)
        passed = complete and np.median([float(r["mad_relative_error"]) for r in subset]) <= config["gates"]["max_group_median_mad_relative_error"] and np.median([float(r["lyapunov_absolute_error"]) for r in subset]) <= config["gates"]["max_group_median_lyapunov_absolute_error"]
        checks[f"group_{group['dtype']}_{group['alpha']}_{group['initial_distribution']}"] = bool(passed == group["passed"])
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(out/"source_code.zip") as archive:
        checks["source_hashes"] = all(hashlib.sha256(archive.read(p) if p in archive.namelist() else (root/p).read_bytes()).hexdigest() == h for p,h in env["source_sha256"].items())
    checks["artifact_hashes"] = all(hashlib.sha256((out/p).read_bytes()).hexdigest() == h
        for h,p in (line.split("  ",1) for line in (out/"sha256.txt").read_text(encoding="utf-8").splitlines()))
    checks = {name:bool(value) for name,value in checks.items()}
    result = dict(passed=all(checks.values()),checks=checks,quadrature=theory,cycles=cycles,
                  scipy_version=scipy.__version__,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation="Artifact integrity is separate from scientific gate success.")
    (out/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "cycles"},indent=2))
    return result



def validate_e0b(out: Path) -> Dict:
    """E0Bを位相表示・複素多項式と独立集計で検証する。共通実装は呼ばない。"""
    config = json.loads((out/"config.json").read_text(encoding="utf-8"))
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    env = json.loads((out/"environment.json").read_text(encoding="utf-8"))
    tables = {}
    for name in ("shift", "gram", "control", "pole"):
        with (out/f"{name}_metrics.csv").open(encoding="utf-8") as f:
            tables[name] = list(csv.DictReader(f))
    checks = {}
    seeds, Ks, orders = config["seeds"], config["K_values"], np.array(config["gram_orders"])
    checks["shift_grid"] = {(int(r["seed"]),int(r["K"]),int(r["k"])) for r in tables["shift"]} == {
        (s,K,k) for s in seeds for K in Ks for k in config["shift_orders"]} and len(tables["shift"]) == len(seeds)*len(Ks)*len(config["shift_orders"])
    checks["gram_grid"] = {(int(r["seed"]),int(r["length"])) for r in tables["gram"]} == {
        (s,n) for s in seeds for n in config["sample_lengths"]} and len(tables["gram"]) == len(seeds)*len(config["sample_lengths"])
    checks["control_grid"] = {(int(r["seed"]),float(r["alpha"])) for r in tables["control"]} == {
        (s,a) for s in seeds for a in config["negative_alphas"]} and len(tables["control"]) == len(seeds)*len(config["negative_alphas"])
    expected_poles = {(K,k,j,float(offset)) for K in Ks for k in config["shift_orders"] for j in range(1,K) for offset in config["pole_offsets"]}
    actual_poles = {(int(r["K"]),int(r["k"]),int(r["pole_index"]),float(r["offset"])) for r in tables["pole"] if r["case"] == "offset"}
    checks["pole_grid"] = actual_poles == expected_poles and len(tables["pole"]) == len(expected_poles)+len(Ks)*len(config["shift_orders"])
    shift_ok, gram_ok, control_ok, ks_ok = True, True, True, True
    for seed in seeds:
        with np.load(out/f"samples_seed_{seed}.npz") as data:
            theta, x, q, images, grams = (data[k] for k in ("theta","x","q","mapped","grams"))
        expected_theta = np.pi * np.random.Generator(np.random.PCG64(seed)).random(max(config["sample_lengths"]))
        checks[f"input_replay_{seed}"] = np.array_equal(theta,expected_theta) and np.array_equal(x,np.cos(theta)/np.sin(theta))
        phase = np.exp(-2j*theta)
        checks[f"phase_coordinate_{seed}"] = bool(np.max(np.abs(q-phase)) < 1e-13)
        checks[f"shapes_finite_{seed}"] = (images.shape == (len(Ks),len(x)) and grams.shape == (len(config["sample_lengths"]),len(orders),len(orders))
            and x.dtype == np.float64 and q.dtype == np.complex128 and np.isfinite(images).all())
        Fourier = np.exp(-2j*theta[:,None]*orders)
        for i,n in enumerate(config["sample_lengths"]):
            G = Fourier[:n].conj().T@Fourier[:n]/n
            row = next(r for r in tables["gram"] if int(r["seed"]) == seed and int(r["length"]) == n)
            gram_ok &= np.allclose(G,grams[i],atol=1e-13,rtol=0)
            gram_ok &= np.isclose(float(row["frobenius"]),np.linalg.norm(G-np.eye(len(orders))),atol=1e-13,rtol=0)
            gram_ok &= np.isclose(float(row["theoretical_squared_error"]),len(orders)*(len(orders)-1)/n)
            # CDFは角度から直接求め、runnerのarctan経路と照合する。
            u = np.sort(1-theta[:n]/np.pi)
            D = max(np.max(np.arange(1,n+1)/n-u),np.max(u-np.arange(n)/n))
            ks_ok &= abs(D-float(row["input_uniform_ks"])) < 1e-13
        for j,K in enumerate(Ks):
            y = images[j]
            w = ((x+1j)/np.hypot(x,1))**K
            rational_y = w.real/w.imag
            checks[f"independent_map_{seed}_{K}"] = bool(np.allclose(y,rational_y,rtol=1e-7,atol=1e-10))
            shifted = (y-1j)/(y+1j)
            checks[f"independent_shift_{seed}_{K}"] = bool(np.max(np.abs(shifted-np.exp(-2j*K*theta))) < 1e-12)
            n = len(y)
            u = np.sort(.5+np.arctan(y)/np.pi)
            D = max(np.max(np.arange(1,n+1)/n-u),np.max(u-np.arange(n)/n))
            for k in config["shift_orders"]:
                row = next(r for r in tables["shift"] if int(r["seed"]) == seed and int(r["K"]) == K and int(r["k"]) == k)
                residual = np.abs(shifted**k-q**(K*k))
                shift_ok &= abs(float(row["max_residual"])-float(residual.max())) < 1e-14
                shift_ok &= abs(float(row["rms"])-float(np.sqrt(np.mean(residual**2)))) < 1e-14
                shift_ok &= int(row["invalid_count"]) == int(np.sum(~np.isfinite(y)))
                shift_ok &= int(row["near_count"]) == int(np.sum(np.abs(np.sin(K*np.arctan2(1,x))) < config["pole_sine_threshold"]))
                ks_ok &= abs(D-float(row["output_uniform_ks"])) < 1e-13
        for a in config["negative_alphas"]:
            z = a*x-(1-a)/x
            error = np.abs((z-1j)/(z+1j)-phase**2)
            row = next(r for r in tables["control"] if int(r["seed"]) == seed and float(r["alpha"]) == a)
            control_ok &= abs(float(row["rms"])-float(np.sqrt(np.mean(error**2)))) < 1e-13
    checks.update(independent_gram_metrics=bool(gram_ok),independent_shift_metrics=bool(shift_ok),
                  independent_control_metrics=bool(control_ok),independent_cdf_metrics=bool(ks_ok))
    pole_ok = True
    for row in tables["pole"]:
        K,k,x = int(row["K"]),int(row["k"]),float(row["x"])
        expected_undefined = row["case"] == "exact_zero" and K%2 == 0
        pole_ok &= (row["status"] == "undefined_pole") == expected_undefined
        if expected_undefined:
            pole_ok &= row["y"] == "" and row["residual"] == "" and row["near_pole"] == "True"
        else:
            y = float(row["y"])
            error = abs(((y-1j)/(y+1j))**k-((x-1j)/(x+1j))**(K*k))
            pole_ok &= abs(error-float(row["residual"])) < 1e-14
    checks["pole_metrics"] = bool(pole_ok)
    g = config["gates"]
    full = [r for r in tables["gram"] if int(r["length"]) == max(config["sample_lengths"])]
    first = [r for r in tables["gram"] if int(r["length"]) == min(config["sample_lengths"])]
    expected = {
        "iid_finite": all(int(r["invalid_count"]) == 0 for r in tables["shift"]),
        "shift_identity": all(float(r["max_residual"]) <= g["max_shift_residual"] for r in tables["shift"]),
        "unit_modulus": all(float(r["modulus_error"]) <= g["max_modulus_error"] for r in full),
        "cauchy_input": all(float(r["input_uniform_ks"]) <= g["max_uniform_ks"] for r in full),
        "cauchy_output": all(float(r["output_uniform_ks"]) <= g["max_uniform_ks"] for r in tables["shift"]),
        "gram_accuracy_all_seeds": all(float(r["frobenius"]) <= g["max_gram_frobenius"] for r in full),
        "gram_convergence_all_seeds": all(float(next(r["frobenius"] for r in full if int(r["seed"]) == s)) < float(next(r["frobenius"] for r in first if int(r["seed"]) == s)) for s in seeds),
        "gram_convergence_median": np.median([float(r["frobenius"]) for r in full]) < np.median([float(r["frobenius"]) for r in first]),
        "finite_pole_shift": all(float(r["residual"]) <= g["max_shift_residual"] for r in tables["pole"] if r["status"] == "finite"),
        "pole_diagnostics": all(r["status"] == ("undefined_pole" if int(r["K"])%2 == 0 else "finite") for r in tables["pole"] if r["case"] == "exact_zero"),
        "positive_control": all(float(r["rms"]) <= g["max_shift_residual"] for r in tables["control"] if float(r["alpha"]) == .5),
        "negative_control": all(float(r["rms"]) >= g["min_negative_control_rms"] for r in tables["control"] if float(r["alpha"]) != .5),
    }
    checks["all_scientific_gate_decisions"] = {k:bool(v) for k,v in expected.items()} == summary["scientific_gates"]
    checks["overall_decision"] = all(expected.values()) == summary["overall_passed"]
    checks["summary_values"] = all(abs(float(np.median([float(r["frobenius"]) for r in tables["gram"] if int(r["length"]) == n]))-summary["gram_median_by_length"][str(n)]) < 1e-13 for n in config["sample_lengths"])
    with zipfile.ZipFile(out/"source_code.zip") as archive:
        checks["source_hashes"] = all(hashlib.sha256(archive.read(p)).hexdigest() == h for p,h in env["source_sha256"].items())
    checks["artifact_hashes"] = all(hashlib.sha256((out/p).read_bytes()).hexdigest() == h for h,p in
        (line.split("  ",1) for line in (out/"sha256.txt").read_text().splitlines()))
    checks = {k:bool(v) for k,v in checks.items()}
    result = dict(passed=all(checks.values()),checks=checks,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation="Integrity validation is separate from scientific hypothesis support.")
    (out/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"artifacts"/"boole_local_validation")
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.output) ["passed"] else 1)
