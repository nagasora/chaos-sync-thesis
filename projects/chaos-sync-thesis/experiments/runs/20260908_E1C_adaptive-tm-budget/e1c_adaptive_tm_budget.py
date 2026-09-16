from pathlib import Path
import numpy as np,json,csv,hashlib,struct,platform,os
HERE=Path(__file__).parent
SRC=HERE.parent/"20260906_E1_tangent-tm-complex-plane/artifacts/source_orbits.npz"
SHA=HERE.parent/"20260906_E1_tangent-tm-complex-plane/sha256.json"
CASES=["b1.01_float64","b1.1_float64","b1.5_float64","b2.0_float64"]; METHODS=["fixed_tm_a0","fourier_low","adaptive_tm"]
def dg(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def grid(): return [0j]+[r*np.exp(2j*np.pi*k/16) for r in [.3,.6,.85] for k in range(16)]
def tm(x,p):
 if not p:return np.ones(len(x),complex)
 a=p[-1];v=np.sqrt(1-abs(a)**2)/(1-np.conj(a)*x)
 for b in p[:-1]:v*= (x-b)/(1-np.conj(b)*x)
 return v
def fit(A,y):
 c=np.linalg.lstsq(A,y,rcond=None)[0];r=y-A@c;return c,r,float(np.mean(abs(r)**2))
def model(y,x,meth,m):
 if meth=="adaptive_tm":
  pool=grid();p=[];cols=[];e=np.mean(abs(y)**2)
  for _ in range(m):
   best=None
   for a in pool:
    q=tm(x,p+[a]);c,r,z=fit(np.column_stack(cols+[q]),y)
    if best is None or z<best[0]:best=(z,a,q)
   e,a,q=best;p.append(a);cols.append(q);pool.remove(a)
  A=np.column_stack(cols)
 elif meth=="fixed_tm_a0":
  p=[0j]*m;A=np.column_stack([tm(x,p[:i+1]) for i in range(m)])
 else:
  o=[0]+[v for k in range(1,m) for v in (k,-k)][:m-1];p=[complex(v) for v in o];A=np.column_stack([x**v for v in o])
 c,r,e=fit(A,y);return np.asarray(p),c,r,e
def rec(x,p,c,meth):
 A=np.column_stack([x**int(a.real) for a in p]) if meth=="fourier_low" else np.column_stack([tm(x,list(p[:i+1])) for i in range(len(p))]);return A@c
def enc(m,p,c):
 b=bytearray(struct.pack("<4sHBBBB6x",b"E1C1",128,len(c),METHODS.index(m),1,0));b.extend(np.asarray(c,np.complex64).tobytes())
 if m=="adaptive_tm":
  g=grid();b.extend(bytes(min(range(len(g)),key=lambda i:abs(g[i]-a)) for a in p))
 return bytes(b)
def dec(b,m):
 _,T,n,mid,_,_=struct.unpack("<4sHBBBB6x",b[:16]);assert T==128 and mid==METHODS.index(m)
 c=np.frombuffer(b[16:16+8*n],np.complex64).astype(complex)
 if m=="adaptive_tm":p=np.array([grid()[i] for i in b[16+8*n:16+9*n]])
 elif m=="fixed_tm_a0":p=np.zeros(n,complex)
 else:p=np.array([complex(v) for v in ([0]+[k for j in range(1,n) for k in (j,-j)])[:n]])
 return p,c
def main():
 cfg=json.loads((HERE/"config.json").read_text());expected=json.loads(SHA.read_text())["artifacts/source_orbits.npz"];assert dg(SRC)==expected
 out=HERE/"artifacts";out.mkdir();rows=[];sp={};pr={};bits={};wins={}
 with np.load(SRC) as z:
  for case in CASES:
   a=z[case][1:]
   for w in range(12):
    y=np.exp(1j*(2*np.arctan2(a[w*128:(w+1)*128],1)-np.pi));wins[f"{case}_w{w}"]=y;x=np.exp(2j*np.pi*np.arange(128)/128)
    for meth in METHODS:
     for B in cfg["total_byte_budgets"]:
      m=max(1,(B-16)//(9 if meth=="adaptive_tm" else 8));p,c,r,e=model(y,x,meth,m);b=enc(meth,p,c);dp,dc=dec(b,meth);q=rec(x,dp,dc,meth);tag=f"{case}_w{w}_{meth}_B{B}";sp[tag]=dp;pr[tag]=q;bits[tag]=np.frombuffer(b,dtype=np.uint8)
      rows.append({"case":case,"window":w,"method":meth,"budget":B,"terms":m,"actual_bytes":len(b),"under_budget":len(b)<=B,"qr_prequant_mse":e,"decoded_mse":float(np.mean(abs(y-q)**2)),"codec_roundtrip":True})
    for m in cfg["diagnostic_m"]:
     for meth in ["fixed_tm_a0","adaptive_tm"]:
      p,c,r,e=model(y,x,meth,m);rows.append({"case":case,"window":w,"method":meth,"budget":"diagnostic_m","terms":m,"actual_bytes":"","under_budget":"","qr_prequant_mse":e,"decoded_mse":float(np.mean(abs(y-rec(x,p,c,meth))**2)),"codec_roundtrip":""})
 with (out/"case_metrics.csv").open("w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(out/"selected_poles.npz",**sp);np.savez_compressed(out/"predictions.npz",**pr);np.savez_compressed(out/"windows.npz",**wins);np.savez_compressed(out/"bitstreams.npz",**bits)
 (out/"summary.json").write_text(json.dumps({"status":"completed","windows":48,"primary_rows":432,"raw_complex64_bytes":1040,"source_sha256":expected},indent=2),encoding="utf8")
 (HERE/"environment.json").write_text(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"threads":os.getenv("OMP_NUM_THREADS","unset"),"source_sha256":expected,"code_sha256":dg(Path(__file__))},indent=2),encoding="utf8")
def validate():
 r=list(csv.DictReader((HERE/"artifacts/case_metrics.csv").open(encoding="utf8")));p=[x for x in r if x["budget"]!="diagnostic_m"]
 bs=np.load(HERE/"artifacts/bitstreams.npz");ww=np.load(HERE/"artifacts/windows.npz");file_ok=True
 for x in p:
  tag=f"{x["case"]}_w{x["window"]}_{x["method"]}_B{x["budget"]}"
  payload=bytes(bs[tag].tolist());dp,dc=dec(payload,x["method"]);xx=np.exp(2j*np.pi*np.arange(128)/128);q=rec(xx,dp,dc,x["method"])
  file_ok=file_ok and abs(float(np.mean(abs(ww[f"{x["case"]}_w{x["window"]}"]-q)**2))-float(x["decoded_mse"]))<1e-12 and len(payload)==int(x["actual_bytes"])
 v={"passed":len(p)==432 and file_ok and all(x["under_budget"]=="True" and np.isfinite(float(x["decoded_mse"])) for x in p),"file_decode":file_ok}
 (HERE/"artifacts/validation.json").write_text(json.dumps(v,indent=2),encoding="utf8");print(v)
if __name__=="__main__":main();validate()
