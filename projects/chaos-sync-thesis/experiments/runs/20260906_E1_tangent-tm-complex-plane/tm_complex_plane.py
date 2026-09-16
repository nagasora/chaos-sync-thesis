"""保存済みtan軌道を、理論ゲートの後にCayley/TM座標で解析する。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
BLUE, INK, AMBER = "#236C93", "#303640", "#AA7228"


def configuration() -> Dict[str, Any]:
    """承認済みの解析条件を返す。"""
    return dict(betas=[1.0001, 1.0002, 1.0005, 1.001, 1.002, 1.005, 1.01, 1.02, 1.05, 1.1, 1.5, 2.],
                fp32_betas=[1.0001, 1.01, 2.], representatives=[1.0001, 1.01, 1.1, 2.],
                n=200000, window=50000, scales=["fixed", "half", "matched", "double"],
                degree=8, lags=[1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384],
                bins=64, scatter_n=4096, time_n=2048, phase_lags=[1, 64, 4096],
                quadrature_n=[4096, 8192], contour_lags=[1, 2, 4, 8, 16], radius=.5,
                theory_tolerance=1e-10, unit_tolerance=1e-12, shuffle_seed=20260906,
                source_run="20260906_E0_tangent-cauchy", selected_seed=0)


def dump(path: Path, value: Any) -> None:
    """非有限値を許さずJSONをLF/UTF-8で保存する。"""
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path, rows: List[Dict[str, Any]]) -> None:
    """条件別データをCSVへ保存する。"""
    if not rows:
        raise ValueError(f"空の表: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def scale_fixed(beta: float) -> float:
    """ゼロ根を除外して非退化Cauchy尺度を求める。"""
    if not math.isfinite(beta) or beta <= 1:
        raise ValueError("beta>1が必要")
    def residual(g: float) -> float:
        z = g*g
        return z*(1/3+z*(1/5+z*(1/7+z/9)))-(beta-1) if g<.01 else math.atanh(g)/g-beta
    return brentq(residual, 0., np.nextafter(1., 0.), xtol=5e-15, rtol=1e-14)


def cayley(x: np.ndarray, scale: float) -> np.ndarray:
    """実数の有限入力を桁あふれなしに単位円へ写す。"""
    if not np.isfinite(x).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("有限状態と有限正尺度が必要")
    # 複素除算やx/scaleは極端な比で桁あふれし得るため角度で同じ写像を評価する。
    return np.exp(1j*(2*np.arctan2(x.astype(float), scale)-np.pi))


def scales(gamma: float) -> Dict[str, float]:
    return dict(fixed=1., half=gamma/2, matched=gamma, double=2*gamma)


def poisson(theta: float, a: float) -> float:
    return (1-a*a)/(2*np.pi*(1-2*a*np.cos(theta)+a*a))


def expected_keys(cfg: Dict[str, Any]) -> List[str]:
    return sorted([f"b{b}_float64" for b in cfg["betas"]]+[f"b{b}_float32" for b in cfg["fp32_betas"]])


def seal(root: Path) -> None:
    """入力から図までの成果物をハッシュ台帳へ固定する。"""
    paths = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.name not in {"sha256.json", "validation.json"}]
    dump(root/"sha256.json", {p.relative_to(root).as_posix(): digest(p) for p in paths})


def prepare(root: Path, source: Path) -> None:
    """E0を変更せず入力をコピーし、由来と解析設定を固定する。"""
    if (root/"config.json").exists():
        raise FileExistsError("設定済み。既存実験は上書きしません")
    original = source/"artifacts/representative_orbits.npz"
    expected = load(source/"artifacts/sha256.json")["artifacts/representative_orbits.npz"]
    if digest(original) != expected:
        raise ValueError("E0入力のハッシュが不一致")
    (root/"artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, root/"artifacts/source_orbits.npz")
    dump(root/"config.json", configuration())
    dump(root/"provenance.json", dict(source_run=source.name, source_relative="artifacts/representative_orbits.npz",
         source_sha256=expected, used_indices="[1:200001]", representative_seed=0,
         source_environment=load(source/"environment.json"), e0_gate="209/210; one DKW exceedance retained"))


def theory(root: Path) -> Dict[str, Any]:
    """軌道を読まず独立求積と解析予測を保存し、解析開始を制御する。"""
    if (root/"theory.json").exists():
        raise FileExistsError("理論結果は上書きしません")
    cfg = load(root/"config.json")
    checks, predictions = [], []
    def check(name: str, beta: float, error: float) -> None:
        checks.append(dict(name=name, beta=beta, error=float(error), passed=bool(error<=cfg["theory_tolerance"])))
    for beta in cfg["betas"]:
        gamma = scale_fixed(beta)
        r = beta*(1-gamma*gamma)
        for label, s in scales(gamma).items():
            a = (gamma-s)/(gamma+s)
            edges = np.linspace(-np.pi, np.pi, cfg["bins"]+1)
            mass = np.array([quad(poisson, left, right, args=(a,), epsabs=1e-13)[0] for left, right in zip(edges[:-1], edges[1:])])
            check("Poisson normalization "+label, beta, abs(mass.sum()-1))
            predictions.append(dict(beta=beta, gamma=gamma, scale_name=label, scale=s, a=a, r=r,
                                    bin_probabilities=mass.tolist(), moments=[a**k for k in range(cfg["degree"]+1)],
                                    lag_correlations=[[r**(k*lag) for lag in cfg["lags"]] for k in range(1,cfg["degree"]+1)]))
            for n in cfg["quadrature_n"]:
                angles=-np.pi/2+np.pi*(np.arange(n)+.5)/n
                x=gamma*np.tan(angles)
                q=cayley(x, s)
                powers=q[:,None]**np.arange(cfg["degree"]+1)
                target=a**np.abs(np.arange(cfg["degree"]+1)[:,None]-np.arange(cfg["degree"]+1))
                check(f"moments {label} n={n}", beta, float(np.max(abs(powers.mean(axis=0)-a**np.arange(cfg["degree"]+1)))))
                check(f"Gram {label} n={n}", beta, float(np.max(abs(powers.conj().T@powers/n-target))))
                check(f"unit {label} n={n}", beta, float(np.max(abs(abs(q)-1))))
        # 単位円境界の極を数値的に追わず、円板内部のCauchy積分で係数を求める。
        for n in cfg["quadrature_n"]:
            z=cfg["radius"]*np.exp(2j*np.pi*(np.arange(n)+.5)/n)
            h=z.copy()
            for lag in range(1, max(cfg["contour_lags"])+1):
                x=1j*gamma*(1+h)/(1-h)
                y=np.tan(beta*x)
                h=(y-1j*gamma)/(y+1j*gamma)
                if lag in cfg["contour_lags"]:
                    for k in range(1,cfg["degree"]+1):
                        coefficient=np.mean((h/z)**k)
                        check(f"contour n={n} lag={lag} k={k}", beta, abs(coefficient-r**(lag*k)))
        x=np.array([-20., -1., 0., 1., 20.])*gamma
        z=cayley(x,gamma)
        restored=1j*gamma*(1+z)/(1-z)
        check("inverse", beta, float(np.max(abs(restored-x)/np.maximum(gamma,abs(x)))))
    result=dict(passed=all(r["passed"] for r in checks), max_error=max(r["error"] for r in checks),
                config_sha256=digest(root/"config.json"), code_sha256=digest(Path(__file__)),
                checks=checks, predictions=predictions)
    dump(root/"theory.json",result)
    if not result["passed"]:
        raise ArithmeticError("理論ゲート不通過。軌道解析は実行不可")
    return result


def analyze(root: Path) -> Dict[str, Any]:
    """理論ゲートと入力同一性を確認して代表軌道だけを解析する。"""
    cfg, gate=load(root/"config.json"), load(root/"theory.json")
    if not gate["passed"] or gate["config_sha256"]!=digest(root/"config.json") or gate["code_sha256"]!=digest(Path(__file__)):
        raise ValueError("理論確認後のコード・設定変更、または理論ゲート不通過")
    if (root/"metrics.json").exists() or (root/"artifacts/moments.csv").exists():
        raise FileExistsError("解析結果は上書きしません")
    out=root/"artifacts"
    if digest(out/"source_orbits.npz") != load(root/"provenance.json")["source_sha256"]:
        raise ValueError("入力ハッシュ不一致")
    moments, density, gram, correlations, diagnostics = [], [], [], [], []
    points: Dict[str,np.ndarray]={}
    windows=[(0,cfg["n"])]+[(i,i+cfg["window"]) for i in range(0,cfg["n"],cfg["window"])]
    with np.load(out/"source_orbits.npz") as archive:
        if sorted(archive.files)!=expected_keys(cfg):
            raise ValueError("入力軌道キーが事前条件と異なる")
        for index,key in enumerate(sorted(archive.files)):
            beta=float(key[1:].split("_")[0]); dtype=key.split("_")[1]
            raw=archive[key]
            if len(raw)!=cfg["n"]+1 or not np.isfinite(raw).all():
                dump(root/"failure.json",dict(key=key,reason="nonfinite input or unexpected length"))
                raise ValueError("入力に非有限値または長さ不一致")
            x=raw[1:].astype(float)
            gamma=scale_fixed(beta); r=beta*(1-gamma*gamma)
            for label,s in scales(gamma).items():
                q=cayley(x,s)
                unit_error=float(np.max(abs(abs(q)-1)))
                diagnostics.append(dict(key=key,scale_name=label,unit_error=unit_error,n=len(x),finite=True))
                if unit_error>cfg["unit_tolerance"]:
                    dump(root/"failure.json",diagnostics[-1]); raise ArithmeticError("単位円から逸脱")
                powers=q[:,None]**np.arange(cfg["degree"]+1)
                a=(gamma-s)/(gamma+s)
                target=next(t for t in gate["predictions"] if t["beta"]==beta and t["scale_name"]==label)
                sample_indices=np.linspace(0,len(q)-1,min(cfg["scatter_n"],len(q)),dtype=int)
                points[key+"_"+label]=q[sample_indices]
                points[key+"_"+label+"_indices"]=sample_indices+1
                if label=="matched":
                    points[key+"_time"]=q[:cfg["time_n"]]
                    for lag in cfg["phase_lags"]:
                        w=q[lag:]*q[:-lag].conj()
                        ix=np.linspace(0,len(w)-1,min(cfg["scatter_n"],len(w)),dtype=int)
                        points[key+f"_phase{lag}"]=w[ix]
                        points[key+f"_phase{lag}_indices"]=ix+1
                for start,stop in windows:
                    p=powers[start:stop]; n=len(p)
                    meta=dict(key=key,beta=beta,dtype=dtype,scale_name=label,scale=s,start=start,stop=stop,n=n)
                    mean=p.mean(axis=0); g=p.conj().T@p/n
                    for k in range(1,cfg["degree"]+1):
                        moments.append(dict(meta,k=k,real=float(mean[k].real),imag=float(mean[k].imag),theory=a**k,error=float(abs(mean[k]-a**k))))
                    counts,edges=np.histogram(np.angle(q[start:stop]),bins=np.linspace(-np.pi,np.pi,cfg["bins"]+1))
                    for j,count in enumerate(counts):
                        expected=target["bin_probabilities"][j]
                        density.append(dict(meta,bin=j,left=float(edges[j]),right=float(edges[j+1]),count=int(count),probability=float(count/n),theory=expected,error=float(count/n-expected)))
                    for k in range(cfg["degree"]+1):
                        for ell in range(cfg["degree"]+1):
                            expected=a**abs(ell-k)
                            gram.append(dict(meta,k=k,ell=ell,real=float(g[k,ell].real),imag=float(g[k,ell].imag),theory=expected,error=float(abs(g[k,ell]-expected))))
                    if label=="matched":
                        rng=np.random.default_rng(np.random.SeedSequence([cfg["shuffle_seed"],1,index,start,stop]))
                        permutation=rng.permutation(n)
                        shuffled=p[permutation]
                        finite_null=(abs(p.sum(axis=0))**2-n)/(n*(n-1))
                        for order,values in [("original",p),("shuffled",shuffled)]:
                            for lag in cfg["lags"]:
                                c=np.mean(values[lag:,1:]*values[:-lag,1:].conj(),axis=0)
                                for k,value in enumerate(c,1):
                                    expected=r**(lag*k) if order=="original" else float(finite_null[k])
                                    correlations.append(dict(meta,order=order,lag=lag,k=k,pairs=n-lag,real=float(value.real),imag=float(value.imag),theory=expected,error=float(abs(value-expected))))
                        if start==0 and stop==cfg["n"]:
                            diagnostics[-1]["shuffle_mean_error"]=float(np.max(abs(shuffled.mean(axis=0)-mean)))
            print("analyzed "+key,flush=True)
    # 診断列は尺度ごとに同じschemaへ揃える。
    for d in diagnostics:
        d.setdefault("shuffle_mean_error",None)
    for name,rows in [("moments",moments),("density",density),("gram",gram),("correlations",correlations),("diagnostics",diagnostics)]:
        table(out/(name+".csv"),rows)
    np.savez_compressed(out/"plot_points.npz",**points)
    full=[r for r in correlations if r["dtype"]=="float64" and r["start"]==0 and r["stop"]==cfg["n"]]
    summary=dict(status="completed",orbits=len(expected_keys(cfg)),unit_error=max(r["unit_error"] for r in diagnostics),
                 max_original_correlation_error=max(r["error"] for r in full if r["order"]=="original"),
                 max_shuffled_correlation_error=max(r["error"] for r in full if r["order"]=="shuffled"),
                 row_counts={name:len(rows) for name,rows in [("moments",moments),("density",density),("gram",gram),("correlations",correlations)]},
                 limitation="One representative seed per condition; windows are not independent replicates.")
    dump(root/"metrics.json",summary)
    dump(root/"environment.json",dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                                     code_sha256=digest(Path(__file__)),theory_sha256=digest(root/"theory.json"),config_sha256=digest(root/"config.json")))
    seal(root)
    return summary


def draw(root: Path) -> None:
    """保存した表示点・集計CSVだけから論文用の三形式を描画する。"""
    cfg=load(root/"config.json"); out=root/"artifacts"
    if not (root/"metrics.json").exists():
        raise FileNotFoundError("解析を先に実行してください")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":INK,"text.color":INK,"svg.fonttype":"none","pdf.fonttype":42})
    data={name:read_table(out/(name+".csv")) for name in ["moments","density","gram","correlations"]}
    representatives=cfg["representatives"]; count=len(representatives)
    def rows(name: str, beta: float, label: str="matched", dtype: str="float64", **filters: Any) -> List[Dict[str,str]]:
        return [r for r in data[name] if float(r["beta"])==beta and r["dtype"]==dtype and r["scale_name"]==label
                and int(r["start"])==0 and int(r["stop"])==cfg["n"] and all(r[k]==str(v) for k,v in filters.items())]
    def circle(ax: Any) -> None:
        t=np.linspace(-np.pi,np.pi,361); ax.plot(np.cos(t),np.sin(t),color="#BCC1C5",lw=.7,zorder=0)
        ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),xlabel="Real",ylabel="Imaginary")
        ax.set_aspect("equal"); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
    def save(fig: Any, name: str, title: str) -> None:
        fig.suptitle(title,fontsize=14)
        fig.text(.5,.01,f"E0 saved representative seed 0 | N={cfg['n']:,} | theoretical scale unless labeled",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.95))
        for ext in ["png","pdf","svg"]:
            fig.savefig(out/f"{name}.{ext}",dpi=240,facecolor="white")
        plt.close(fig)
    def markers(ax: Any, empirical: complex, theoretical: float) -> None:
        ax.plot(empirical.real,empirical.imag,"o",color=AMBER,ms=5,label="Empirical mean",zorder=5)
        ax.plot(theoretical,0,"*",color=INK,ms=9,label="Theory mean",zorder=6)
    with np.load(out/"plot_points.npz") as points:
        fig,axes=plt.subplots(count,4,figsize=(13,3.1*count),squeeze=False)
        for i,beta in enumerate(representatives):
            key=f"b{beta}_float64"
            for j,label in enumerate(["fixed","matched"]):
                z=points[key+"_"+label]; ax=axes[i,2*j]; circle(ax)
                ax.scatter(z.real,z.imag,s=2,alpha=.25,color=BLUE)
                ax.set_title(f"beta={beta}, s={'1' if label=='fixed' else 'gamma*'}")
                ax=axes[i,2*j+1]; d=rows("density",beta,label)
                middle=[(float(r["left"])+float(r["right"]))/2 for r in d]; width=2*np.pi/cfg["bins"]
                ax.stairs([float(r["probability"])/width for r in d],np.linspace(-np.pi,np.pi,cfg["bins"]+1),color=BLUE,label="Empirical")
                ax.plot(middle,[float(r["theory"])/width for r in d],"--",color=INK,label="Theory bin average")
                ax.set(xlabel="Angle (radians)",ylabel="Density",xlim=(-np.pi,np.pi)); ax.legend(fontsize=8)
        save(fig,"fig1_geometry_density","Cayley geometry and angular density: scale matters")
        fig,axes=plt.subplots(1,count+1,figsize=(3.6*count+.6,4),squeeze=False,gridspec_kw={"width_ratios":[1]*count+[.05]})
        for ax,beta in zip(axes[0,:count],representatives):
            z=points[f"b{beta}_float64_time"]; circle(ax)
            artist=ax.scatter(z.real,z.imag,c=np.arange(1,len(z)+1),cmap="viridis",s=5,vmin=1,vmax=cfg["time_n"])
            ax.set_title(f"beta={beta}")
        fig.colorbar(artist,cax=axes[0,-1],label="Discrete time t")
        save(fig,"fig2_time_colored","First 2,048 states on the complex plane")
        for name,columns in [("fig3_modes",[1,2,4]),("fig4_phase_difference",cfg["phase_lags"])]:
            fig,axes=plt.subplots(count,len(columns),figsize=(3.8*len(columns),3.2*count),squeeze=False)
            for i,beta in enumerate(representatives):
                key=f"b{beta}_float64"
                for j,column in enumerate(columns):
                    ax=axes[i,j]; circle(ax)
                    if name=="fig3_modes":
                        z=points[key+"_matched"]**column; r=rows("moments",beta,k=column)[0]
                        label=f"k={column}"
                    else:
                        z=points[key+f"_phase{column}"]; r=rows("correlations",beta,k=1,lag=column,order="original")[0]
                        label=f"lag={column}"
                    ax.scatter(z.real,z.imag,s=2,color=BLUE,alpha=.2)
                    markers(ax,complex(float(r["real"]),float(r["imag"])),float(r["theory"]))
                    ax.set_title(f"beta={beta}, {label}")
            axes[0,0].legend(loc="upper right",fontsize=7)
            save(fig,name,"TM modes on the unit circle" if name=="fig3_modes" else "Lag phase differences: mean reveals temporal structure")
    colors={1:BLUE,2:AMBER,4:"#6D5894",8:"#39846B"}
    fig,axes=plt.subplots(count,3,figsize=(12,3*count),squeeze=False)
    for i,beta in enumerate(representatives):
        for k in [1,2,4,8]:
            r=rows("correlations",beta,k=k,order="original"); x=[int(v["lag"]) for v in r]
            for j,field in enumerate(["real","imag","error"]):
                axes[i,j].semilogx(x,[float(v[field]) for v in r],"o-",ms=3,color=colors[k],label=f"k={k}")
                axes[i,j].set(xlabel="Lag",ylabel=["Real correlation","Imaginary correlation","Absolute complex error"][j])
            axes[i,0].semilogx(x,[float(v["theory"]) for v in r],"--",color=colors[k],alpha=.65)
        axes[i,0].set_title(f"beta={beta}: solid empirical / dashed theory"); axes[i,1].axhline(0,color=INK,lw=.6)
        axes[i,0].legend(fontsize=8,ncol=2)
    save(fig,"fig5_correlations","Temporal correlations versus analytic predictions")
    # 理論・経験・誤差を別パネルとし、理論/経験には共通の色範囲を使う。
    for i,beta in enumerate(representatives):
        fig,axes=plt.subplots(3,3,figsize=(10,8),squeeze=False)
        for j,label in enumerate(["half","matched","double"]):
            r=rows("gram",beta,label); matrices=[]
            for field in ["theory","real","error"]:
                matrix=np.empty((cfg["degree"]+1,cfg["degree"]+1))
                for v in r: matrix[int(v["k"]),int(v["ell"])]=float(v[field])
                matrices.append(matrix)
            error_max=max(float(np.max(matrices[2])),1e-12)
            for col,(field,matrix) in enumerate(zip(["Theory Re G","Empirical Re G","Complex error magnitude"],matrices)):
                ax=axes[j,col]
                im=ax.imshow(matrix,cmap="coolwarm" if col<2 else "magma",vmin=-1 if col<2 else 0,vmax=1 if col<2 else error_max)
                ax.set(title=f"{label}: {field}",xlabel="l",ylabel="k")
                fig.colorbar(im,ax=ax,fraction=.046)
        name="fig6_gram" if i==0 else f"fig6_gram_beta_{beta}"
        save(fig,name,f"Gram matrices: beta={beta}; error color scale labeled separately")
    fig,axes=plt.subplots(count,2,figsize=(10,3*count),squeeze=False)
    for i,beta in enumerate(representatives):
        for order,color in [("original",BLUE),("shuffled",AMBER)]:
            r=rows("correlations",beta,k=1,order=order)
            axes[i,0].semilogx([int(v["lag"]) for v in r],[float(v["real"]) for v in r],"o-",color=color,label=order)
            axes[i,0].semilogx([int(v["lag"]) for v in r],[float(v["theory"]) for v in r],"--",color=color)
        for dtype,color in [("float64",BLUE),("float32",AMBER)]:
            r=rows("correlations",beta,dtype=dtype,k=1,order="original")
            if r: axes[i,1].semilogx([int(v["lag"]) for v in r],[float(v["error"]) for v in r],"o-",color=color,label=dtype)
        for j in [0,1]:
            axes[i,j].set(xlabel="Lag",ylabel="Real C1" if j==0 else "Absolute complex error",title=f"beta={beta}"); axes[i,j].legend(fontsize=8)
    save(fig,"supplement_controls","Shuffle control and input precision sensitivity")
    seal(root)


def validate(root: Path) -> Dict[str, Any]:
    """保存ファイルの完全性・集計・点の時刻対応を独立に確認する。"""
    cfg=load(root/"config.json"); out=root/"artifacts"; checks=[]
    for relative,expected in load(root/"sha256.json").items():
        checks.append(dict(name="hash "+relative,passed=digest(root/relative)==expected))
    keys=expected_keys(cfg); nw=1+cfg["n"]//cfg["window"]
    for name,count in [("moments",len(keys)*4*nw*cfg["degree"]),("density",len(keys)*4*nw*cfg["bins"]),
                       ("gram",len(keys)*4*nw*(cfg["degree"]+1)**2),("correlations",len(keys)*nw*2*len(cfg["lags"])*cfg["degree"])]:
        checks.append(dict(name="count "+name,passed=len(read_table(out/(name+".csv")))==count))
    moments=read_table(out/"moments.csv"); corr=read_table(out/"correlations.csv")
    with np.load(out/"source_orbits.npz") as source, np.load(out/"plot_points.npz") as points:
        for key in keys:
            x=source[key][1:].astype(float); beta=float(key[1:].split("_")[0]); gamma=scale_fixed(beta)
            # 実数式の複素除算を独立経路として使い、保存した角度実装と照合する。
            z=(x-1j*gamma)/(x+1j*gamma)
            selected=next(r for r in moments if r["key"]==key and r["scale_name"]=="matched" and int(r["start"])==0 and int(r["stop"])==len(x) and int(r["k"])==1)
            saved=complex(float(selected["real"]),float(selected["imag"]))
            checks.append(dict(name="mean "+key,passed=bool(abs(z.mean()-saved)<1e-12)))
            indices=points[key+"_matched_indices"]-1
            checks.append(dict(name="plot positions "+key,passed=bool(np.max(abs(z[indices]-points[key+"_matched"]))<1e-12)))
            for lag in [cfg["lags"][0],cfg["lags"][-1]]:
                selected=next(r for r in corr if r["key"]==key and r["order"]=="original" and int(r["start"])==0 and int(r["stop"])==len(x) and int(r["k"])==1 and int(r["lag"])==lag)
                value=np.vdot(z[:-lag],z[lag:])/(len(z)-lag)
                checks.append(dict(name=f"lag {key}/{lag}",passed=bool(abs(value-complex(float(selected["real"]),float(selected["imag"])))<1e-12)))
    gate=load(root/"theory.json")
    checks.append(dict(name="theory gate and unchanged code",passed=gate["passed"] and gate["code_sha256"]==digest(Path(__file__))))
    result=dict(passed=all(r["passed"] for r in checks),checks=checks)
    dump(root/"validation.json",result)
    return result


def main() -> None:
    """入力準備、理論、解析、描画、検証を明示的に分離する。"""
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--source",type=Path,default=HERE.parent/"20260906_E0_tangent-cauchy")
    group=parser.add_mutually_exclusive_group(required=True)
    for mode in ["prepare","theory","analyze","plot","validate"]:
        group.add_argument("--"+mode,action="store_true")
    args=parser.parse_args()
    if args.prepare: prepare(args.output,args.source)
    elif args.theory:
        r=theory(args.output); print(json.dumps(dict(passed=r["passed"],max_error=r["max_error"],checks=len(r["checks"]))))
    elif args.analyze: print(json.dumps(analyze(args.output)))
    elif args.plot: draw(args.output)
    else:
        r=validate(args.output); print(json.dumps(dict(passed=r["passed"],checks=len(r["checks"]))))
        if not r["passed"]: raise SystemExit(1)


if __name__=="__main__":
    main()
