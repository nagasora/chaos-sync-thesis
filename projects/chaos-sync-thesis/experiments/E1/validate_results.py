"""E1Aの全軌道・特徴・選択・予測をrunnerの関数をimportせず独立照合する。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Dict, List
import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    """保存した未集約表を読み込む。"""
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def independent_features(x: np.ndarray, permutation: np.ndarray) -> Dict[str, np.ndarray]:
    """有理式でなく角度の指数関数から同じ観測量を組み立てる。"""
    lower,center,upper=np.percentile(x,[25,50,75])
    z=(x-center)/((upper-lower)/2)
    phase=np.exp(-2j*np.arctan2(1.,z))
    result={}
    for suffix,signal,q in (("",z,phase),("_shuffle",z[permutation],phase[permutation])):
        row=[]
        for k in range(1,5):
            for lag in (1,2):
                c=np.vdot(q[:-lag]**k,q[lag:]**k)/(len(q)-lag)
                row.extend([c.real,c.imag])
        result["tm"+suffix]=np.array(row)
        for name,parts,bands in (("fourier_raw",[signal],16),("fourier_cayley",[q.real,q.imag],8)):
            row=[]
            for part in parts:
                window=.5-.5*np.cos(2*np.pi*np.arange(len(part))/(len(part)-1))
                power=np.abs(np.fft.rfft(window*(part-part.mean())))[1:]**2/np.dot(window,window)
                row.extend(np.log(1e-12+np.mean(band)) for band in np.array_split(power,bands))
            result[name+suffix]=np.array(row)
    moments=[np.mean(phase**k) for k in range(1,9)]
    result["tm_instant"]=np.array([(v.real,v.imag) for v in moments]).ravel()
    result["quantiles"]=np.quantile(z,np.linspace(.05,.95,16))
    return result


def validate(out: Path) -> Dict:
    """事後整合性検証。科学的仮説が不合格でも記録の整合性は検証できる。"""
    config=json.loads((out/"config.json").read_text())
    summary=json.loads((out/"summary.json").read_text())
    frozen=json.loads((out/"selection.json").read_text())
    env=json.loads((out/"environment.json").read_text())
    checks={}
    seeds=[set(config[s+"_seeds"]) for s in ("train","validation","test")]
    checks["seed_disjoint"]=all(not seeds[i]&seeds[j] for i in range(3) for j in range(i))
    checks["unseen_interpolation"]=not set(config["train_alphas"])&set(config["test_alphas"]) and min(config["train_alphas"])<min(config["test_alphas"])<=max(config["test_alphas"])<max(config["train_alphas"])
    checks["freeze_order"]=env["started_utc"]<=frozen["frozen_utc"]<=env["test_generation_started_utc"]<=env["completed_utc"]
    checks["frozen_model_hash"]=hashlib.sha256((out/"models.npz").read_bytes()).hexdigest()==frozen["model_sha256"]
    tables={}; targets={}; all_rows={}; repeated=0
    for split in ("train","validation","test"):
        rows=read_csv(out/(split+"_conditions.csv")); all_rows[split]=rows
        with np.load(out/(split+"_orbits.npz")) as data:
            x=data["orbits"]; initial=data["initial"]; y=data["target"]
            alphas=config["test_alphas" if split=="test" else "train_alphas"]
            expected=[(s,a) for s in config[split+"_seeds"] for a in alphas]
            checks[split+"_grid"]=[(int(r["seed"]),float(r["alpha"])) for r in rows]==expected and np.array_equal(y,[a for s,a in expected]) and np.array_equal(data["seeds"],[s for s,a in expected])
            generated=np.array([np.random.default_rng(np.random.SeedSequence([s,round(a*1000000)])).standard_cauchy() for s,a in expected])
            checks[split+"_initial"]=np.array_equal(initial,generated) and np.array_equal(initial,[float(r["initial_state"]) for r in rows])
            state=initial.copy(); near=np.zeros((len(initial),3),dtype=np.int64); replay=True
            for step in range(config["burn_in"]+config["observation_length"]):
                near+=np.abs(state[:,None])<np.array([1e-12,1e-10,1e-8])
                state=y*(state-1/state)
                if step>=config["burn_in"]:
                    replay=replay and np.array_equal(state,x[:,step-config["burn_in"]])
            checks[split+"_exact_replay"]=replay and x.dtype==np.float64 and x.shape==(len(expected),config["observation_length"])
            checks[split+"_finite_diagnostics"]=np.isfinite(x).all() and np.all(data["failure_step"]==-1) and np.all(data["nonfinite_update_count"]==0) and np.array_equal(near,data["near_zero_counts"])
            values={}; row_diagnostics=True
            for i,(row,orbit) in enumerate(zip(rows,x)):
                permutation=np.random.default_rng(np.random.SeedSequence([config["shuffle_seed"],int(row["seed"]),round(float(row["alpha"])*1000000)])).permutation(len(orbit))
                for name,value in independent_features(orbit,permutation).items():
                    values.setdefault(name,[]).append(value)
                is_repeated=len(np.unique(orbit))!=len(orbit); repeated+=is_repeated
                q1,loc,q3=np.quantile(orbit,[.25,.5,.75])
                row_diagnostics=row_diagnostics and (row["repeated"]==str(is_repeated)) and float(row["location"])==loc and float(row["scale"])==(q3-q1)/2 and np.array_equal(near[i],[int(row[k]) for k in ("near_1e12","near_1e10","near_1e8")])
            checks[split+"_condition_diagnostics"]=row_diagnostics
        with np.load(out/(split+"_features.npz")) as data:
            tables[split]={k:data[k] for k in data.files}
            checks[split+"_feature_names"]=set(data.files)==set(values)
            for name,value in values.items():
                checks[split+"_features_"+name]=data[name].shape==(len(rows),16) and np.allclose(data[name],value,atol=2e-12,rtol=2e-12)
        targets[split]=y
    checks["sample_counts"]=summary["sample_counts"]=={s:len(y) for s,y in targets.items()}
    predictions={}; candidate_rows=read_csv(out/"validation_candidates.csv"); model_count=len(tables["train"])
    checks["candidate_count"]=len(candidate_rows)==model_count*len(config["ridge_penalties"])
    with np.load(out/"models.npz") as models:
        for name,x in tables["train"].items():
            mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<=1e-12]=1
            z=(x-mean)/scale; center=targets["train"].mean(); candidates=[]
            for penalty in config["ridge_penalties"]:
                augmented=np.vstack([z,np.sqrt(penalty)*np.eye(16)])
                beta=np.linalg.lstsq(augmented,np.r_[targets["train"]-center,np.zeros(16)],rcond=None)[0]
                v=center+(tables["validation"][name]-mean)/scale@beta
                rmse=float(np.sqrt(np.mean((v-targets["validation"])**2)))
                candidates.append((rmse,penalty,beta))
                matching=[r for r in candidate_rows if r["model"]==name and float(r["penalty"])==penalty]
                checks[name+"_candidate_"+str(penalty)]=len(matching)==1 and abs(float(matching[0]["validation_rmse"])-rmse)<1e-8
            loss,penalty,beta=min(candidates,key=lambda r:(r[0],r[1]))
            checks[name+"_selection"]=frozen["selection"][name]["penalty"]==penalty and abs(frozen["selection"][name]["validation_rmse"]-loss)<1e-8 and frozen["selection"][name]["dimension"]==16
            checks[name+"_model"]=np.allclose(models[name+"__mean"],mean,atol=0,rtol=0) and np.array_equal(models[name+"__scale"],scale) and float(models[name+"__intercept"])==center and np.allclose(models[name+"__coefficients"],beta,atol=1e-7,rtol=1e-7)
            predictions[name]=center+(tables["test"][name]-mean)/scale@beta
    predictions["constant"]=np.full(len(targets["test"]),targets["train"].mean())
    predictions["analytic"]=(1+tables["test"]["tm"][:,0])/2
    rows=read_csv(out/"predictions.csv"); seed_rows=read_csv(out/"seed_metrics.csv"); alpha_rows=read_csv(out/"alpha_metrics.csv"); metrics=read_csv(out/"metrics.csv")
    checks["output_row_counts"]=len(rows)==len(predictions)*len(targets["test"]) and len(seed_rows)==len(predictions)*len(config["test_seeds"]) and len(alpha_rows)==len(predictions)*len(config["test_alphas"])
    errors={}; y=targets["test"]; test_seeds=np.array([int(r["seed"]) for r in all_rows["test"]])
    for name,pred in predictions.items():
        saved=[r for r in rows if r["model"]==name]
        checks[name+"_predictions"]=np.allclose(pred,[float(r["prediction"]) for r in saved],atol=1e-7,rtol=1e-7) and [(int(r["seed"]),float(r["alpha"])) for r in saved]==[(int(r["seed"]),float(r["alpha"])) for r in all_rows["test"]] and np.allclose(pred-y,[float(r["error"]) for r in saved],atol=1e-7)
        score=dict(rmse=float(np.sqrt(np.mean((pred-y)**2))),mae=float(np.mean(np.abs(pred-y))),r2=float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)))
        checks[name+"_metrics"]=all(abs(float(next(r[k] for r in metrics if r["model"]==name))-v)<1e-7 and abs(next(r[k] for r in summary["metrics"] if r["model"]==name)-v)<1e-7 for k,v in score.items())
        errors[name]=np.array([np.sqrt(np.mean((pred[test_seeds==s]-y[test_seeds==s])**2)) for s in config["test_seeds"]])
        checks[name+"_seed_rmse"]=np.allclose(errors[name],[float(r["rmse"]) for r in seed_rows if r["model"]==name],atol=1e-7)
        checks[name+"_alpha_rmse"]=all(abs(float(r["rmse"])-np.sqrt(np.mean((pred[y==float(r["alpha"])]-float(r["alpha"]))**2)))<1e-7 for r in alpha_rows if r["model"]==name)
    draws=np.random.default_rng(config["bootstrap_seed"]).integers(0,len(config["test_seeds"]),size=(config["bootstrap_repetitions"],len(config["test_seeds"])))
    contrast_rows=read_csv(out/"contrasts.csv")
    with np.load(out/"bootstrap.npz") as data:
        checks["bootstrap_draws"]=np.array_equal(draws,data["draws"])
        for row,saved in zip(contrast_rows,summary["contrasts"]):
            left,right=row["left"],row["right"]; coverage=float(row["coverage"])
            delta=errors[left]-errors[right]; samples=delta[draws].mean(axis=1)
            lo,hi=np.quantile(samples,[(1-coverage)/2,(1+coverage)/2])
            checks["bootstrap_"+left+"_"+right]=np.allclose(samples,data[left+"__"+right],atol=1e-7) and all(abs(float(row[k])-v)<1e-7 and abs(saved[k]-v)<1e-7 for k,v in (("mean_seed_difference",delta.mean()),("lower",lo),("upper",hi)))
    gates=dict(temporal_information=any(float(r["upper"])<0 for r in contrast_rows[:2]),tm_superiority_raw=float(contrast_rows[2]["upper"])<0,lag_information=float(contrast_rows[3]["upper"])<0,
               dimension_budget=all(v.shape[1]==16 for v in tables["train"].values()),no_observed_repeats=repeated==0,split_contract=checks["seed_disjoint"] and checks["unseen_interpolation"],all_finite=all(checks[s+"_finite_diagnostics"] for s in targets))
    checks["scientific_gates"]=gates==summary["scientific_gates"] and repeated==summary["repeated_conditions"]
    checks["proceed_gate"]=summary["proceed_to_e1b"]==all(gates[k] for k in ("temporal_information","dimension_budget","no_observed_repeats","split_contract","all_finite"))
    with zipfile.ZipFile(out/"source_code.zip") as archive:
        checks["source_hashes"]=all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in env["source_sha256"].items())
    checks["artifact_hashes"]=all(hashlib.sha256((out/p).read_bytes()).hexdigest()==h for h,p in (line.split("  ",1) for line in (out/"sha256.txt").read_text().splitlines()))
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(passed=all(checks.values()),check_count=len(checks),checks=checks,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation="Independent integrity checks; scientific gates are reported separately.")
    (out/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"],check_count=len(checks),failed=[k for k,v in checks.items() if not v]),indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"artifacts/fixed_readout")
    args=parser.parse_args()
    raise SystemExit(0 if validate(args.output)["passed"] else 1)
