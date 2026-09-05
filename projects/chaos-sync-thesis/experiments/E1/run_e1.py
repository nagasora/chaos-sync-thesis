"""事前固定E1Aをtrain/validation選択、凍結、testの順で実行する。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from src.core import simulate_boole_orbits
from src.readout import fixed_readout_features, fit_ridge, predict_ridge
from E0.run_e0 import write_csv, first_repeat


def validate_config(config: Dict) -> None:
    """seed分離・補間test・固定特徴容量を実行前に検査する。"""
    seed_sets = [set(config[s+"_seeds"]) for s in ("train", "validation", "test")]
    if any(not group for group in seed_sets) or any(seed_sets[i]&seed_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("split間でseedを重複できません。")
    for split in ("train", "validation", "test"):
        if len(config[split+"_seeds"]) != len(set(config[split+"_seeds"])):
            raise ValueError("seedはsplit内でも一意にしてください。")
    a, b = config["train_alphas"], config["test_alphas"]
    if set(a)&set(b) or not 0 < min(a) < min(b) <= max(b) < max(a) < 1:
        raise ValueError("test alphaはtrain未使用の補間点が必要です。")
    if len(a)!=len(set(a)) or len(b)!=len(set(b)):
        raise ValueError("alphaは重複できません。")
    if config["tm_orders"] != [1,2,3,4] or config["tm_lags"] != [1,2] or config["feature_dimension"] != 16:
        raise ValueError("特徴は事前固定の16次元です。")
    if config["precision"] != "float64" or config["observation_length"] < 32 or config["burn_in"] < 0:
        raise ValueError("float64と十分な軌道長が必要です。")
    if any(p <= 0 or not np.isfinite(p) for p in config["ridge_penalties"]):
        raise ValueError("penaltyは有限正数です。")


def generate_split(config: Dict, split: str, out: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[Dict]]:
    """全軌道・診断を先に保存する。失敗時は代替標本を生成せず停止する。"""
    alphas = config["test_alphas" if split == "test" else "train_alphas"]
    rows = [dict(split=split, seed=s, alpha=a) for s in config[split+"_seeds"] for a in alphas]
    initial = np.array([np.random.default_rng(np.random.SeedSequence([r["seed"], round(r["alpha"]*1000000)] )).standard_cauchy() for r in rows])
    target = np.array([r["alpha"] for r in rows])
    orbits, diag = simulate_boole_orbits(initial, target, config["burn_in"], config["observation_length"])
    np.savez_compressed(out/(split+"_orbits.npz"), orbits=orbits, initial=initial, target=target,
                        seeds=np.array([r["seed"] for r in rows]), **diag)
    if np.any(diag["failure_step"] != -1) or not np.isfinite(orbits).all():
        raise RuntimeError("軌道生成失敗。保存診断を確認してください。失敗標本は除外しません。")
    values = {}
    for i, (row, orbit) in enumerate(zip(rows, orbits)):
        permutation = np.random.default_rng(np.random.SeedSequence([config["shuffle_seed"],row["seed"],round(row["alpha"]*1000000)])).permutation(len(orbit))
        features, location, scale = fixed_readout_features(orbit, permutation)
        for name, feature in features.items():
            values.setdefault(name, []).append(feature)
        repeat = first_repeat(orbit)
        row.update(initial_state=float(initial[i]), location=location, scale=scale, repeated=repeat[0]>=0,
                   near_1e12=int(diag["near_zero_counts"][i,0]), near_1e10=int(diag["near_zero_counts"][i,1]), near_1e8=int(diag["near_zero_counts"][i,2]))
    matrices = {name:np.array(value) for name,value in values.items()}
    np.savez_compressed(out/(split+"_features.npz"), **matrices)
    write_csv(out/(split+"_conditions.csv"), rows)
    print(f"{split}: {len(rows)} trajectories and features saved", flush=True)
    return matrices, target, rows


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> Dict:
    """poolしたRMSE、MAE、R2を返す。seed平均RMSEと区別する。"""
    error = prediction-y
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)))


def plot_results(out: Path, metrics: List[Dict], contrasts: List[Dict], predictions: List[Dict]) -> None:
    """保存表と同一の値から性能比較・区間・test予測を描画する。"""
    directory = out/"figures"
    directory.mkdir()
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#F9F8F6","font.size":9})
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh([r["model"] for r in metrics],[r["rmse"] for r in metrics],color="#2C3E35")
    ax.set(xlabel="Test RMSE (pooled trajectories)",title="Unseen alpha readout: fixed features, 48 independent test seeds")
    fig.tight_layout(); fig.savefig(directory/"readout_metrics.png",dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for i,r in enumerate(contrasts):
        axes[0].plot([r["lower"],r["upper"]],[i,i],color="#2C3E35")
        axes[0].scatter(r["mean_seed_difference"],i,color="#8C6D46",s=25)
    axes[0].set_yticks(range(len(contrasts)),[r["contrast"] for r in contrasts])
    axes[0].axvline(0,color="gray",linestyle="--")
    axes[0].set(xlabel="Mean paired seed RMSE difference (left model - right)",title="Seed bootstrap intervals (coverage shown in labels)")
    for model in ("tm","fourier_raw","fourier_cayley","analytic"):
        rows=[r for r in predictions if r["model"]==model]
        axes[1].scatter([r["alpha"] for r in rows],[r["prediction"] for r in rows],s=9,alpha=.4,label=model)
    axes[1].plot([.3,.7],[.3,.7],color="gray",linestyle="--")
    axes[1].set(xlabel="True alpha",ylabel="Predicted alpha",title="192 held-out trajectories")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory/"paired_differences.png",dpi=160); plt.close(fig)


def run(config: Dict, out: Path) -> Dict:
    """test生成前に選択を凍結し、全予測・モデル・証拠を保存する。"""
    validate_config(config)
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    source_paths=[HERE/"run_e1.py",HERE/"validate_results.py",HERE/"README.md",HERE/"config.json",HERE.parent/"src/core.py",HERE.parent/"src/readout.py",HERE.parent/"E0/run_e0.py",HERE.parent/"tests/test_e1a_temporal_alpha_readout.py"]
    env=dict(started_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),numpy=np.__version__,matplotlib=matplotlib.__version__,command=sys.argv,
             threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
             source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
             reference_sha256={config["source_text"]:hashlib.sha256((ROOT/config["source_text"]).read_bytes()).hexdigest()})
    with zipfile.ZipFile(out/"source_code.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in source_paths:
            archive.write(p,p.relative_to(ROOT).as_posix())
    train,ytrain,train_rows=generate_split(config,"train",out)
    val,yval,val_rows=generate_split(config,"validation",out)
    models,selection,cv_rows = {},{},[]
    for name in train:
        candidates=[]
        for penalty in config["ridge_penalties"]:
            model=fit_ridge(train[name],ytrain,penalty)
            loss=regression_metrics(yval,predict_ridge(model,val[name]))["rmse"]
            cv_rows.append(dict(model=name,penalty=penalty,validation_rmse=loss))
            candidates.append((loss,penalty,model))
        loss,penalty,model=min(candidates,key=lambda item:(item[0],item[1]))
        models[name]=model
        selection[name]=dict(penalty=penalty,validation_rmse=loss,dimension=train[name].shape[1])
    write_csv(out/"validation_candidates.csv",cv_rows)
    np.savez_compressed(out/"models.npz",**{name+"__"+key:value for name,model in models.items() for key,value in model.items()})
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),selection=selection,model_sha256=hashlib.sha256((out/"models.npz").read_bytes()).hexdigest())
    (out/"selection.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    env["test_generation_started_utc"]=datetime.now(timezone.utc).isoformat()
    test,ytest,test_rows=generate_split(config,"test",out)
    predicted={name:predict_ridge(models[name],features) for name,features in test.items()}
    predicted["constant"]=np.full(len(ytest),ytrain.mean())
    predicted["analytic"]=(1+test["tm"][:,0])/2
    metrics,predictions,seed_metrics,alpha_metrics=[],[],[],[]
    seed_order=np.array([r["seed"] for r in test_rows])
    errors={}
    for name,prediction in predicted.items():
        metrics.append(dict(model=name,dimension=16 if name in models else 0,**regression_metrics(ytest,prediction)))
        errors[name]=[]
        for seed in config["test_seeds"]:
            mask=seed_order==seed
            score=regression_metrics(ytest[mask],prediction[mask])
            errors[name].append(score["rmse"])
            seed_metrics.append(dict(model=name,seed=seed,**score))
        for alpha in config["test_alphas"]:
            mask=ytest==alpha
            alpha_metrics.append(dict(model=name,alpha=alpha,rmse=float(np.sqrt(np.mean((prediction[mask]-alpha)**2))),mae=float(np.mean(np.abs(prediction[mask]-alpha)))))
        predictions.extend(dict(model=name,seed=r["seed"],alpha=r["alpha"],prediction=float(p),error=float(p-r["alpha"])) for r,p in zip(test_rows,prediction))
    draws=np.random.default_rng(config["bootstrap_seed"]).integers(0,len(config["test_seeds"]),size=(config["bootstrap_repetitions"],len(config["test_seeds"])))
    comparisons=[("tm","tm_shuffle",.975),("fourier_raw","fourier_raw_shuffle",.975),("tm","fourier_raw",.95),("tm","tm_instant",.95),("tm","fourier_cayley",.95)]
    contrasts=[]; bootstrap={}
    for left,right,coverage in comparisons:
        delta=np.array(errors[left])-errors[right]
        samples=delta[draws].mean(axis=1)
        low,high=np.quantile(samples,[(1-coverage)/2,(1+coverage)/2])
        key=left+" - "+right
        contrasts.append(dict(contrast=key+f" ({coverage*100:g}%)",left=left,right=right,coverage=coverage,mean_seed_difference=float(delta.mean()),lower=float(low),upper=float(high)))
        bootstrap[left+"__"+right]=samples
    np.savez_compressed(out/"bootstrap.npz",draws=draws,**bootstrap)
    for name,rows in (("metrics",metrics),("predictions",predictions),("seed_metrics",seed_metrics),("alpha_metrics",alpha_metrics),("contrasts",contrasts)):
        write_csv(out/(name+".csv"),rows)
    repeated=sum(r["repeated"] for r in train_rows+val_rows+test_rows)
    gates=dict(temporal_information=any(r["upper"]<0 for r in contrasts[:2]),tm_superiority_raw=contrasts[2]["upper"]<0,lag_information=contrasts[3]["upper"]<0,
               dimension_budget=all(r["dimension"]==16 for r in selection.values()),no_observed_repeats=repeated==0,split_contract=True,all_finite=True)
    summary=dict(experiment_id=config["experiment_id"],scientific_gates=gates,
                 proceed_to_e1b=all(gates[k] for k in ("temporal_information","dimension_budget","no_observed_repeats","split_contract","all_finite")),
                 repeated_conditions=repeated,sample_counts=dict(train=len(ytrain),validation=len(yval),test=len(ytest)),metrics=metrics,contrasts=contrasts,
                 interpretation="Finite-window parameter readout; not synchronization, compression, or universal TM superiority.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    plot_results(out,metrics,contrasts,predictions)
    env["completed_utc"]=datetime.now(timezone.utc).isoformat()
    (out/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    hashes=[f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file()]
    (out/"sha256.txt").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HERE/"config.json")
    parser.add_argument("--output",type=Path,default=HERE/"artifacts/fixed_readout")
    args=parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8-sig")),args.output)
