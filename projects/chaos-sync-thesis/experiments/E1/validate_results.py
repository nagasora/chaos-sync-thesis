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
    if config["experiment_id"] == "E1-TMA-DYNAMICS-AUDIT":
        return validate_dynamics(out)
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


def audit_dictionary(x: np.ndarray, name: str) -> np.ndarray:
    """runnerの辞書をimportせず、TMは位相・多項式は漸化式で照合する。"""
    u=.5+np.arctan(x)/np.pi; t=2*u-1
    if name in ("tm","phase_fourier","euclidean_fourier"):
        if name=="tm":
            modes=np.exp(-2j*np.arctan2(1,x)[:,None]*np.arange(1,9))
        elif name=="phase_fourier":
            modes=((x-1j)/(x+1j))[:,None]**np.arange(1,9)
        else:
            angles=x[:,None]*np.arange(1,9);modes=np.cos(angles)+1j*np.sin(angles)
        return np.sqrt(2)*np.stack([modes.real,modes.imag],axis=-1).reshape(len(x),16)
    if name=="cdf_cosine":return np.sqrt(2)*np.cos(u[:,None]*np.pi*np.arange(1,17))
    if name=="cdf_rbf":return np.column_stack([np.exp(-((t-((2*j+1)/16-1))**2)/(2*.25**2)) for j in range(16)])
    previous=np.ones(len(t));current=t.copy();columns=[np.sqrt(3)*current]
    for k in range(2,17):
        following=((2*k-1)*t*current-(k-1)*previous)/k
        columns.append(np.sqrt(2*k+1)*following);previous,current=current,following
    return np.column_stack(columns)


def audit_targets(x: np.ndarray) -> np.ndarray:
    """共通targetを角度だけで評価する。"""
    phase=-2*np.arctan2(1,x)
    return np.column_stack((np.cos(phase),np.sin(phase),2*np.arctan(x)/np.pi))


def validate_dynamics(out: Path) -> Dict:
    """TM-A全軌道・作用素/decoder・共通予測・bootstrapと理論対照を独立検証。"""
    c=json.loads((out/"config.json").read_text());summary=json.loads((out/"summary.json").read_text())
    env=json.loads((out/"environment.json").read_text());frozen=json.loads((out/"selection.json").read_text())
    checks={};data={};repeats=0;T=c["observation_length"]
    sets=[set(c[s+"_seeds"]) for s in ("train","validation","test")]
    checks["seed_disjoint"]=all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    checks["freeze_order"]=env["started_utc"]<=frozen["frozen_utc"]<=env["test_generation_started_utc"]<=env["completed_utc"]
    for split in ("train","validation","test"):
        rows=read_csv(out/(split+"_conditions.csv"))
        expected=[(a,s) for a in c["alphas"] for s in c[split+"_seeds"]]
        with np.load(out/(split+"_orbits.npz")) as d:
            raw=d["orbits"];a=d["alpha"];gamma=np.sqrt(a/(1-a))
            initial=gamma*np.array([np.random.default_rng(np.random.SeedSequence([seed,round(alpha*1000000)])).standard_cauchy() for alpha,seed in expected])
            checks[split+"_manifest"]=[(float(r["alpha"]),int(r["seed"])) for r in rows]==expected and np.array_equal(a,[v for v,s in expected]) and np.array_equal(d["seeds"],[s for v,s in expected]) and np.array_equal(d["initial"],initial)
            state=initial.copy();near=np.zeros((len(state),3),dtype=int);replay=True
            for step in range(c["burn_in"]+T+4):
                near+=np.abs(state[:,None])<np.array([1e-12,1e-10,1e-8]);state=a*(state-1/state)
                if step>=c["burn_in"]:replay=replay and np.array_equal(state,raw[:,step-c["burn_in"]])
            checks[split+"_replay"]=replay and raw.shape==(len(expected),T+4) and raw.dtype==np.float64
            checks[split+"_finite"]=np.isfinite(raw).all() and np.all(d["failure_step"]==-1) and np.all(d["nonfinite_update_count"]==0) and np.array_equal(near,d["near_zero_counts"])
            row_ok=True
            for i,orbit in enumerate(raw):
                repeated=np.unique(orbit).size!=orbit.size;repeats+=repeated
                row_ok=row_ok and rows[i]["repeated"]==str(repeated) and int(rows[i]["near_1e8"])==near[i,2] and float(rows[i]["initial_state"])==initial[i]
            checks[split+"_diagnostics"]=row_ok
            data[split]={round(alpha*100):raw[a==alpha]/np.sqrt(alpha/(1-alpha)) for alpha in c["alphas"]}
    metrics=read_csv(out/"metrics.csv");seeds=read_csv(out/"seed_metrics.csv");structure=read_csv(out/"structure.csv");candidates=read_csv(out/"validation_candidates.csv");contrasts=read_csv(out/"contrasts.csv")
    checks["table_counts"]=(len(metrics)==3*6*4*2 and len(seeds)==len(metrics)*len(c["test_seeds"]) and len(candidates)==3*6*len(c["ridge_penalties"]) and len(structure)==18 and len(contrasts)==24)
    all_errors={};own_half=None;low_half=None
    for a in c["alphas"]:
        key=round(a*100);train=data["train"][key];val=data["validation"][key];test=data["test"][key]
        target=audit_targets(train[:,:T].ravel());variance=target.var(axis=0)
        modelpath=out/f"models_alpha_{key}.npz"
        checks[f"model_hash_{key}"]=hashlib.sha256(modelpath.read_bytes()).hexdigest()==frozen["model_sha256"][modelpath.name]
        with np.load(modelpath) as saved,np.load(out/f"predictions_alpha_{key}.npz") as predictions:
            checks[f"variance_{key}"]=np.allclose(saved["target_variance"],variance,rtol=1e-12,atol=1e-12)
            for name in c["dictionaries"]:
                prefix=f"{key}_{name}"
                x=audit_dictionary(train[:,:T].ravel(),name);future=audit_dictionary(train[:,1:T+1].ravel(),name)
                mu=x.mean(axis=0);scale=x.std(axis=0);scale[scale<=1e-12]=1;z=(x-mu)/scale
                chosen=next(r for r in frozen["selection"] if r["alpha"]==a and r["dictionary"]==name)
                selected_rows=[r for r in candidates if float(r["alpha"])==a and r["dictionary"]==name]
                minimum=min(selected_rows,key=lambda r:(float(r["validation_nmse"]),float(r["penalty"])))
                checks[prefix+"_selection"]=chosen["penalty"]==float(minimum["penalty"]) and chosen["validation_nmse"]==float(minimum["validation_nmse"])
                verified={}
                for penalty in c["ridge_penalties"]:
                    augmented=np.vstack([z,np.sqrt(penalty)*np.eye(16)])
                    means=[future.mean(axis=0),target.mean(axis=0)]
                    betas=[np.linalg.lstsq(augmented,np.vstack([y-center,np.zeros((16,y.shape[1]))]),rcond=None)[0] for y,center in zip((future,target),means)]
                    current=audit_dictionary(val[:,:T].ravel(),name);loss=[]
                    for h in c["horizons"]:
                        current=means[0]+(current-mu)/scale@betas[0]
                        pred=(means[1]+(current-mu)/scale@betas[1]).reshape(len(val),T,3)
                        truth=audit_targets(val[:,h:h+T].ravel()).reshape(len(val),T,3)
                        mse=np.mean((pred-truth)**2,axis=1);loss.extend([np.mean(mse[:,:2].sum(axis=1)/variance[:2].sum()),np.mean(mse[:,2]/variance[2])])
                    value=np.mean(loss)
                    recorded=float(next(r["validation_nmse"] for r in selected_rows if float(r["penalty"])==penalty))
                    checks[prefix+"_candidate_"+str(penalty)]=np.isclose(value,recorded,atol=2e-5,rtol=2e-5)
                    if penalty==chosen["penalty"]:verified=dict(betas=betas,means=means)
                for j,part in enumerate(("operator","decoder")):
                    checks[prefix+"_"+part]=np.allclose(saved[name+"__"+part+"__mean"],mu,atol=1e-12) and np.allclose(saved[name+"__"+part+"__scale"],scale,atol=1e-12) and np.allclose(saved[name+"__"+part+"__intercept"],verified["means"][j],atol=1e-12) and np.allclose(saved[name+"__"+part+"__coefficients"],verified["betas"][j],atol=2e-4,rtol=2e-4)
                current=audit_dictionary(test[:,:T].ravel(),name);errors=[]
                for hi,h in enumerate(c["horizons"]):
                    current=verified["means"][0]+(current-mu)/scale@verified["betas"][0]
                    pred=(verified["means"][1]+(current-mu)/scale@verified["betas"][1]).reshape(len(test),T,3)
                    truth=audit_targets(test[:,h:h+T].ravel()).reshape(len(test),T,3)
                    checks[prefix+f"_pred_h{h}"]=np.allclose(pred,predictions[name][hi],atol=2e-5,rtol=2e-5)
                    mse=np.mean((pred-truth)**2,axis=1);err=np.column_stack([mse[:,:2].sum(axis=1)/variance[:2].sum(),mse[:,2]/variance[2]]);errors.append(err)
                    for ti,t in enumerate(c["targets"]):
                        rows=[r for r in seeds if float(r["alpha"])==a and r["dictionary"]==name and int(r["horizon"])==h and r["target"]==t]
                        recorded=float(next(r["nmse"] for r in metrics if float(r["alpha"])==a and r["dictionary"]==name and int(r["horizon"])==h and r["target"]==t))
                        checks[prefix+f"_metric_{h}_{t}"]=[int(r["seed"]) for r in rows]==c["test_seeds"] and np.allclose(err[:,ti],[float(r["nmse"]) for r in rows],atol=2e-5,rtol=2e-5) and np.isclose(err[:,ti].mean(),recorded,atol=2e-5,rtol=2e-5)
                    if h==1:
                        phi_future=audit_dictionary(test[:,1:T+1].ravel(),name);residual=np.mean((current-phi_future)**2,axis=0)
                        own=residual.sum()/x.var(axis=0).sum();low=residual[:8].sum()/x.var(axis=0)[:8].sum()
                all_errors[(key,name)]=np.array(errors)
                B=verified["betas"][0]/scale[None,:];sv=np.linalg.svd(B,compute_uv=False);gram=z.T@z/len(z)
                row=next(r for r in structure if float(r["alpha"])==a and r["dictionary"]==name)
                checks[prefix+"_structure"]=np.allclose(B,predictions[name+"__B_standardized"],atol=2e-4,rtol=2e-4) and np.allclose(gram,predictions[name+"__gram"],atol=1e-12) and abs(float(row["own_dictionary_nmse"])-own)<2e-5 and abs(float(row["first8_nmse"])-low)<2e-5 and abs(float(row["stable_rank"])-np.sum(sv**2)/sv[0]**2)<2e-4
                if key==50 and name=="tm":own_half=own;low_half=low
    draws=np.random.default_rng(c["bootstrap_seed"]).integers(0,len(c["test_seeds"]),size=(c["bootstrap_repetitions"],len(c["test_seeds"])))
    with np.load(out/"bootstrap.npz") as d:
        checks["bootstrap_indices"]=np.array_equal(draws,d["draws"])
        for row in contrasts:
            key=round(float(row["alpha"])*100);name=row["baseline"];target=row["target"];ti=c["targets"].index(target)
            delta=all_errors[(key,"tm")][0,:,ti]-all_errors[(key,name)][0,:,ti];samples=delta[draws].mean(axis=1);lo,hi=np.quantile(samples,[.025,.975])
            checks[f"bootstrap_{key}_{name}_{target}"]=np.allclose(samples,d[f"{key}__{name}__{target}"],atol=2e-5,rtol=2e-5) and all(abs(float(row[k])-v)<2e-5 for k,v in (("mean_difference",delta.mean()),("lower",lo),("upper",hi)))
    theory=json.loads((out/"theory.json").read_text());theory_rows=read_csv(out/"theory_metrics.csv")
    with np.load(out/"theory.npz") as d:
        B=d["operator"];expected=np.zeros((16,16))
        for j in range(8):expected[2*(j//2+1)*2-2+j%2,j]=1
        checks["exact_shift_matrix"]=np.array_equal(B,expected) and np.linalg.matrix_rank(B)==8 and np.count_nonzero(B)==8 and np.all(np.linalg.matrix_power(B,4)==0)
        checks["analytic_gram"]=np.linalg.norm(d["gram"]-np.eye(16))<1e-10 and np.linalg.norm(d["mismatch_gram"]-np.eye(16))>.5
    checks["theoretical_truncation"]=np.allclose([float(r["full_nmse"]) for r in theory_rows],[.5,.75,.875,1],atol=1e-10) and np.allclose([float(r["first_mode_nmse"]) for r in theory_rows],[0,0,0,1],atol=1e-10)
    g=c["gates"];h4=all_errors[(50,"tm")][3,:,0].mean()
    checks["gate_decisions"]=(summary["scientific_gates"]["low_modes_representable"]==(low_half<g["low_modes_nmse_max"]) and summary["scientific_gates"]["finite_dictionary_half_loss"]==(abs(own_half-.5)<g["full_dictionary_nmse_tolerance"]) and summary["scientific_gates"]["four_step_escape"]==(h4>g["horizon4_cayley_nmse_min"]) and summary["scientific_gates"]["tm_uniform_advantage"]==all(float(r["upper"])<0 for r in contrasts) and summary["scientific_gates"]["finite_no_repeats"]==(repeats==0))
    checks["equivalent_predictions"]=all(np.max(np.abs(all_errors[(round(a*100),"tm")]-all_errors[(round(a*100),"phase_fourier")]))<2e-5 for a in c["alphas"])
    checks["request_hash"]=hashlib.sha256((out/"request_text.txt").read_bytes()).hexdigest()==env["request_sha256"]
    with zipfile.ZipFile(out/"source_code.zip") as archive:checks["source_hashes"]=all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in env["source_sha256"].items())
    checks["artifact_hashes"]=all(hashlib.sha256((out/p).read_bytes()).hexdigest()==h for h,p in (line.split("  ",1) for line in (out/"sha256.txt").read_text().splitlines()))
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(passed=all(checks.values()),check_count=len(checks),checks=checks,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numerical_tolerance="Independent least-squares predictions/metrics atol/rtol 2e-5; coefficients atol/rtol 2e-4 for ill-conditioned fixed RBF dictionary; exact orbit replay.")
    (out/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"],check_count=len(checks),failed=[k for k,v in checks.items() if not v]),indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"artifacts/fixed_readout")
    args=parser.parse_args()
    raise SystemExit(0 if validate(args.output)["passed"] else 1)
