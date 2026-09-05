"""TMの状態辞書適合性を共通target予測と有限Koopman射影で監査する。"""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from src.core import simulate_boole_orbits, cayley_modes
from src.readout import state_dictionary, fit_ridge, predict_ridge
from E0.run_e0 import write_csv, first_repeat


def common_targets(x: np.ndarray) -> np.ndarray:
    """同じ有限状態から全辞書共通の2種3実数targetを作る。"""
    q=cayley_modes(x,(1,))[:,0]
    return np.column_stack((q.real,q.imag,2*np.arctan(x)/np.pi))


def validate_config(c: Dict) -> None:
    """固定基底、split分離、未来点の取り扱いを検査する。"""
    groups=[c[s+"_seeds"] for s in ("train","validation","test")]
    if any(not g or len(g)!=len(set(g)) for g in groups) or any(set(groups[i])&set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("seedはsplit内外で一意にしてください。")
    if c["alphas"] != [.25,.5,.75] or c["horizons"] != [1,2,3,4] or c["dimension"] != 16:
        raise ValueError("alpha、horizon、次元は本監査の固定条件です。")
    if c["dictionaries"] != ["tm","phase_fourier","euclidean_fourier","cdf_cosine","cdf_legendre","cdf_rbf"]:
        raise ValueError("辞書の比較集合を変更できません。")
    if c["observation_length"]<16 or c["burn_in"]<0 or any(p<=0 for p in c["ridge_penalties"]):
        raise ValueError("軌道長・burn-in・penaltyが不正です。")


def generate(c: Dict, split: str, out: Path) -> Tuple[Dict[int,np.ndarray],List[Dict]]:
    """全αの同一splitを一括生成し、生軌道と診断を保存する。"""
    rows=[dict(split=split,alpha=a,seed=s) for a in c["alphas"] for s in c[split+"_seeds"]]
    alpha=np.array([r["alpha"] for r in rows]); gamma=np.sqrt(alpha/(1-alpha))
    initial=gamma*np.array([np.random.default_rng(np.random.SeedSequence([r["seed"],round(r["alpha"]*1000000)])).standard_cauchy() for r in rows])
    raw,diag=simulate_boole_orbits(initial,alpha,c["burn_in"],c["observation_length"]+max(c["horizons"]))
    np.savez_compressed(out/(split+"_orbits.npz"),orbits=raw,initial=initial,alpha=alpha,seeds=np.array([r["seed"] for r in rows]),**diag)
    if not np.isfinite(raw).all() or np.any(diag["failure_step"]!=-1):
        raise RuntimeError("保存済み軌道の生成が失敗しました。失敗除外・再生成はしません。")
    for i,r in enumerate(rows):
        r.update(initial_state=float(initial[i]),repeated=first_repeat(raw[i])[0]>=0,near_1e8=int(diag["near_zero_counts"][i,2]))
    write_csv(out/(split+"_conditions.csv"),rows)
    print(f"{split}: {len(rows)} trajectories saved",flush=True)
    return {round(a*100):raw[alpha==a]/np.sqrt(a/(1-a)) for a in c["alphas"]},rows


def target_nmse(pred: np.ndarray, truth: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """時間点を平均し、seedごとにtrain分散で規格化した2target誤差を返す。"""
    mse=np.mean((pred-truth)**2,axis=1)
    return np.column_stack((mse[:,:2].sum(axis=1)/variance[:2].sum(),mse[:,2]/variance[2]))


def evaluate(model: Dict, x: np.ndarray, name: str, c: Dict) -> Tuple[np.ndarray,np.ndarray]:
    """凍結作用素を反復し、全seed・horizonの共通target予測を返す。"""
    count,T=len(x),c["observation_length"]
    lifted=state_dictionary(x[:,:T].ravel(),name)
    predictions=[]; errors=[]
    for h in c["horizons"]:
        lifted=predict_ridge(model["operator"],lifted)
        pred=predict_ridge(model["decoder"],lifted).reshape(count,T,3)
        truth=common_targets(x[:,h:h+T].ravel()).reshape(count,T,3)
        predictions.append(pred);errors.append(target_nmse(pred,truth,model["target_variance"]))
    return np.array(predictions),np.array(errors)


def theory_audit(c: Dict, out: Path) -> Dict:
    """α=.5の無限次元shiftと有限16次元射影を分けた解析正対照。"""
    angle=np.pi*(np.arange(c["theory_grid_size"])+.37)/c["theory_grid_size"]
    x=1/np.tan(angle); phi=state_dictionary(x,"tm")
    B=np.zeros((16,16))
    for k in range(1,5):
        B[2*(2*k-1),2*(k-1)]=1;B[2*(2*k-1)+1,2*(k-1)+1]=1
    rows=[]
    q=cayley_modes(x,(1,))[:,0]; predicted=phi.copy()
    for h in c["horizons"]:
        predicted=predicted@B
        modes=(q[:,None]**(2**h*np.arange(1,9)))
        true=np.sqrt(2)*np.stack((modes.real,modes.imag),axis=-1).reshape(len(x),16)
        rows.append(dict(horizon=h,full_nmse=float(np.mean((predicted-true)**2)),first_mode_nmse=float(np.mean((predicted[:,:2]-true[:,:2])**2))))
    gram=phi.T@phi/len(x)
    mismatch=state_dictionary(x/2,"tm"); mismatch_gram=mismatch.T@mismatch/len(x)
    error=float(np.max(np.abs(phi-state_dictionary(x,"phase_fourier"))))
    np.savez_compressed(out/"theory.npz",operator=B,gram=gram,mismatch_gram=mismatch_gram)
    write_csv(out/"theory_metrics.csv",rows)
    result=dict(equivalence_max=error,gram_frobenius=float(np.linalg.norm(gram-np.eye(16))),mismatch_gram_frobenius=float(np.linalg.norm(mismatch_gram-np.eye(16))),exact_rank=int(np.linalg.matrix_rank(B)),exact_nonzero_fraction=float(np.count_nonzero(B)/B.size),rows=rows)
    (out/"theory.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def plot_results(out: Path, metrics: List[Dict], structure: List[Dict], models: Dict, c: Dict) -> None:
    """共通予測・有限射影の構造を別図にし、数値はCSV/NPZに保存する。"""
    figdir=out/"figures";figdir.mkdir()
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#F9F8F6","font.size":9})
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for col,a in enumerate(c["alphas"]):
        for row,target in enumerate(c["targets"]):
            ax=axes[row,col]
            for name in c["dictionaries"]:
                if name=="phase_fourier":continue
                data=[r for r in metrics if r["alpha"]==a and r["target"]==target and r["dictionary"]==name]
                ax.plot([r["horizon"] for r in data],[r["nmse"] for r in data],marker="o",markersize=3,label=name)
            ax.set(title=f"alpha={a}; target={target}",ylabel="Test NMSE",xlabel="Prediction horizon")
            ax.axhline(1,color="gray",linestyle="--",linewidth=.7)
    axes[0,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(figdir/"common_prediction.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,(key,title) in zip(axes,[("theory","Exact population projection"),("tm","Fitted TM (standardized)"),("cdf_cosine","Fitted CDF cosine (standardized)")]):
        if key=="theory":
            with np.load(out/"theory.npz") as d: B=d["operator"]
        else:
            model=models[(50,key)]["operator"]; B=model["coefficients"]/model["scale"][None,:]
        im=ax.imshow(B,cmap="RdBu_r",vmin=-1,vmax=1);ax.set(title=title,xlabel="Next observable",ylabel="Current observable")
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle("alpha=0.5: matrix sparsity alone is not predictive closure")
    fig.tight_layout();fig.savefig(figdir/"operator_structure.png",dpi=160);plt.close(fig)


def run(c: Dict, out: Path, request: Path) -> Dict:
    """仕様・原文保存、train/validation選択、凍結、testの順を固定する。"""
    validate_config(c)
    request_bytes=request.read_bytes()
    out.mkdir(parents=True,exist_ok=False)
    (out/"request_text.txt").write_bytes(request_bytes)
    (out/"config.json").write_text(json.dumps(c,indent=2),encoding="utf-8")
    sources=[HERE/"run_dynamics.py",HERE/"validate_results.py",HERE/"TM_DYNAMICS.md",HERE/"config_dynamics.json",HERE.parent/"src/readout.py",HERE.parent/"src/core.py",HERE.parent/"E0/run_e0.py",HERE.parent/"tests/test_e1a_temporal_alpha_readout.py"]
    env=dict(started_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},request_sha256=hashlib.sha256(request_bytes).hexdigest(),reference_sha256={c["source_text"]:hashlib.sha256((ROOT/c["source_text"]).read_bytes()).hexdigest()})
    with zipfile.ZipFile(out/"source_code.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sources:archive.write(p,p.relative_to(ROOT).as_posix())
    theory=theory_audit(c,out)
    train,tr_rows=generate(c,"train",out); val,va_rows=generate(c,"validation",out)
    models={}; candidates=[]; selections=[]; T=c["observation_length"]
    for a in c["alphas"]:
        key=round(a*100); x=train[key]; target=common_targets(x[:,:T].ravel()); variance=target.var(axis=0)
        for name in c["dictionaries"]:
            phi=state_dictionary(x[:,:T].ravel(),name); future=state_dictionary(x[:,1:T+1].ravel(),name)
            options=[]
            for penalty in c["ridge_penalties"]:
                model=dict(operator=fit_ridge(phi,future,penalty),decoder=fit_ridge(phi,target,penalty),target_variance=variance)
                _,errors=evaluate(model,val[key],name,c); score=float(errors.mean())
                candidates.append(dict(alpha=a,dictionary=name,penalty=penalty,validation_nmse=score)); options.append((score,penalty,model))
            score,penalty,model=min(options,key=lambda v:(v[0],v[1]));models[(key,name)]=model
            selections.append(dict(alpha=a,dictionary=name,penalty=penalty,validation_nmse=score))
        saved={name+"__"+part+"__"+k:v for (ak,name),model in models.items() if ak==key for part in ("operator","decoder") for k,v in model[part].items()}
        saved["target_variance"]=variance
        np.savez_compressed(out/f"models_alpha_{key}.npz",**saved)
    write_csv(out/"validation_candidates.csv",candidates)
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),selection=selections,model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("models_alpha_*.npz")})
    (out/"selection.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    env["test_generation_started_utc"]=datetime.now(timezone.utc).isoformat()
    test,te_rows=generate(c,"test",out)
    metrics=[]; seed_metrics=[]; structure=[]; error_map={}; low=[]; all_equivalence=[]
    for a in c["alphas"]:
        key=round(a*100); x=test[key]; saved={}
        all_equivalence.append(float(np.max(np.abs(state_dictionary(x.ravel(),"tm")-state_dictionary(x.ravel(),"phase_fourier")))))
        for name in c["dictionaries"]:
            model=models[(key,name)]; pred,errors=evaluate(model,x,name,c); saved[name]=pred;error_map[(key,name)]=errors
            for j,h in enumerate(c["horizons"]):
                for target_index,target_name in enumerate(c["targets"]):
                    values=errors[j,:,target_index]
                    metrics.append(dict(alpha=a,dictionary=name,horizon=h,target=target_name,nmse=float(values.mean())))
                    seed_metrics.extend(dict(alpha=a,dictionary=name,horizon=h,target=target_name,seed=s,nmse=float(v)) for s,v in zip(c["test_seeds"],values))
            phi=state_dictionary(train[key][:,:T].ravel(),name); z=(phi-model["operator"]["mean"])/model["operator"]["scale"]
            gram=z.T@z/len(z); B=model["operator"]["coefficients"]/model["operator"]["scale"][None,:]
            singular=np.linalg.svd(B,compute_uv=False)
            future=state_dictionary(x[:,1:T+1].ravel(),name)
            forecast=predict_ridge(model["operator"],state_dictionary(x[:,:T].ravel(),name))
            variance=phi.var(axis=0)
            full=float(np.mean((future-forecast)**2,axis=0).sum()/variance.sum())
            low_nmse=float(np.mean((future[:,:8]-forecast[:,:8])**2,axis=0).sum()/variance[:8].sum())
            structure.append(dict(alpha=a,dictionary=name,gram_condition=float(np.linalg.cond(gram)),stable_rank=float(np.sum(singular**2)/singular[0]**2),energy_rank90=int(np.searchsorted(np.cumsum(singular**2)/np.sum(singular**2),.9)+1),nonzero_fraction=float(np.mean(np.abs(B)>.01*np.max(np.abs(B)))),own_dictionary_nmse=full,first8_nmse=low_nmse))
            saved[name+"__gram"]=gram;saved[name+"__B_standardized"]=B;saved[name+"__singular_values"]=singular
        np.savez_compressed(out/f"predictions_alpha_{key}.npz",**saved)
    contrasts=[]; bootstrap={}
    draws=np.random.default_rng(c["bootstrap_seed"]).integers(0,len(c["test_seeds"]),size=(c["bootstrap_repetitions"],len(c["test_seeds"])))
    for a in c["alphas"]:
        key=round(a*100)
        for name in c["dictionaries"][2:]:
            for ti,target in enumerate(c["targets"]):
                delta=error_map[(key,"tm")][0,:,ti]-error_map[(key,name)][0,:,ti]
                samples=delta[draws].mean(axis=1);lo,hi=np.quantile(samples,[.025,.975])
                contrasts.append(dict(alpha=a,baseline=name,target=target,horizon=1,mean_difference=float(delta.mean()),lower=float(lo),upper=float(hi)))
                bootstrap[f"{key}__{name}__{target}"]=samples
    np.savez_compressed(out/"bootstrap.npz",draws=draws,**bootstrap)
    for name,rows in (("metrics",metrics),("seed_metrics",seed_metrics),("structure",structure),("contrasts",contrasts)):
        write_csv(out/(name+".csv"),rows)
    fitted=next(r for r in structure if r["alpha"]==.5 and r["dictionary"]=="tm")
    h4=next(r["nmse"] for r in metrics if r["alpha"]==.5 and r["dictionary"]=="tm" and r["horizon"]==4 and r["target"]=="cayley")
    prediction_equivalence=max(float(np.max(np.abs(error_map[(round(a*100),"tm")]-error_map[(round(a*100),"phase_fourier")]))) for a in c["alphas"])
    g=c["gates"]
    gates=dict(angle_fourier_identity=max(all_equivalence+[theory["equivalence_max"]])<=g["equivalence_atol"],low_modes_representable=fitted["first8_nmse"]<g["low_modes_nmse_max"],finite_dictionary_half_loss=abs(fitted["own_dictionary_nmse"]-.5)<g["full_dictionary_nmse_tolerance"],four_step_escape=h4>g["horizon4_cayley_nmse_min"],tm_uniform_advantage=all(r["upper"]<0 for r in contrasts),finite_no_repeats=not any(r["repeated"] for r in tr_rows+va_rows+te_rows))
    summary=dict(experiment_id=c["experiment_id"],scientific_gates=gates,alpha_half=fitted,horizon4_cayley_nmse=h4,angle_equivalence_max=max(all_equivalence),prediction_equivalence_max=prediction_equivalence,trajectory_count=len(tr_rows+va_rows+te_rows),interpretation="Finite dictionary closure and common-target prediction; no synchronization/input-retention/compression claim.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    plot_results(out,metrics,structure,models,c)
    env["completed_utc"]=datetime.now(timezone.utc).isoformat()
    (out/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    hashes=[f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file()]
    (out/"sha256.txt").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2));return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HERE/"config_dynamics.json")
    parser.add_argument("--output",type=Path,default=HERE/"artifacts/tm_dynamics")
    parser.add_argument("--request",type=Path,default=HERE/"artifacts/tm_dynamics/request_text.txt")
    args=parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8-sig")),args.output,args.request)
