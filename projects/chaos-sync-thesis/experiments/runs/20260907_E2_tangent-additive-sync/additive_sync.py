"""相互加算結合の同期多様体と横方向指数を理論先行で検証する。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import t as student

PREREGISTRATION = "# E2：相互加算タンジェント結合の同期可能性\n\n## 事前登録（軌道生成前）\n\n対象は x'=tan(beta x)+epsilon y、y'=tan(beta y)+epsilon x。\nbeta>1、0<=epsilon<1。拡散結合や写像出力の混合とは区別する。\n今回の目的は、同期多様体上の不変測度と横方向線形安定性を確認すること。\n吸引域の調査、一般の二変数分布のCauchy閉包、復号・学習は行わない。\n\n## 理論\n\n### 同期多様体上の閉包を改めて導出する\n\nx=y=s は不変で、s'=f(s)=tan(beta s)+epsilon s。\n上半平面では Im tan(beta z)>0 であり、epsilon>=0ならfも上半平面を自身へ写す。\nu>=0に対して exp(i u f(z)) は有界解析関数。その実部・虚部へのPoisson表示から\n\n    E[exp(i u f(S))] = exp(i u f(i gamma))\n                      = exp[-u{tanh(beta gamma)+epsilon gamma}],\n    S ~ C(0,gamma).\n\n負のuは共役で従う。実軸上の極は測度0、境界値はほとんど至る所で存在する。\n従ってこの1変数写像に限って、厳密尺度更新 G(gamma)=tanh(beta gamma)+epsilon gamma。\nこれは依存するCauchy変数の和を独立とみなした導出ではない。\nPoisson核の式の参考: [Complex Analysis, Theorem 10.3.1](https://complexanalysis.org/web/sec_poisson-integral-formula.html)。\n本写像への適用と以下の安定性計算は、この実験で行った導出である。\n有界解析関数の実部・虚部を用いれば、境界の不連続点を除いて有界調和関数の表示を適用できる。\n\n正の固定尺度gammaは (1-epsilon)gamma=tanh(beta gamma) の唯一の正解。\nGは正領域で狭義凹、G'(0)=beta+epsilon>1、epsilon<1でG(gamma)/gammaの極限はepsilon。\n従って固定点が一つあり、G'(gamma)<1。\n\n### 縦・横方向指数\n\n同期点でのJacobianは [[D,epsilon],[epsilon,D]]、D=beta sec^2(beta s)。\n固有方向(1,1),(1,-1)の倍率はD+epsilon、D-epsilon。\nU=tan(beta S) ~ C(0,a)、a=tanh(beta gamma)=(1-epsilon)gamma とおけば\n\n    lambda_parallel = log(beta)+2 log(a+sqrt(1+epsilon/beta)),\n    lambda_perp     = log(beta)+2 log(a+sqrt(1-epsilon/beta)).\n\nここではepsilon<1<betaなので横倍率は常に正。\n積分 E log(U^2+c^2)=2log(a+c) は、U=a tan(theta)、thetaが一様という置換で\n独立数値積分も行う。\n\n尺度適合Cayley像Hは単位円板のinner写像、H(0)=0、回転写像ではない。\nSchwarzの補題から反復H^nは円板内部のコンパクト集合で0へ収束する。\nH^nの正次数べきのTaylor係数も0へ収束し、Fourier単項式間の相関は消える。\n三角多項式の稠密性と測度保存からmixing、従ってergodic。\n上の対数倍率は積分可能なので、Birkhoffから指数はこの不変測度に関してほとんど確実に上式。\n例外的な固定点・周期点すべての指数を同じと主張しない。\n\n### 同期不安定性の事前予想\n\nB=beta/(1-epsilon)、atanh(a)/a=B とし、\natanh(a)/a < 1/(1-a^2) より a^2>1-(1-epsilon)/beta。\nc=sqrt(1-epsilon/beta)>0 とすると\n\n    beta(a+c)^2 > beta(a^2+c^2) > 2 beta-1 > 1.\n    lambda_perp > log(2 beta-1) > 0.\n\n従って、指定領域の不変Cauchy測度上の典型的完全同期は横方向に不安定と予想する。\nepsilon>=1ではこの正の有限固定尺度は存在しないため、この式で外挿しない。\n\n## 固定実験条件\n\n- beta=1.0001,1.01,1.1,2、epsilon=0,0.01,0.1,0.5,0.9。\n- 32独立seed、各200000ステップ。SeedSequence([20260907,2,beta_index,epsilon_index,seed])。\n- S0~C(0,gamma)、burn-inなし。float64、Numba標準tan、fastmathなし。\n- 主評価は対数横倍率と縦倍率の時間平均。独立seedの平均とStudent-t近似95%区間。\n- 尺度は半IQR、Cauchy CDFとのKS距離、Cayley平均を補助記録。\n- 同期多様体上の軌道と線形変分の検証であり、非同期初期値の吸引域実験ではない。\n- 単位円誤差、非有限値、最大状態を監査。クリップ、再抽選、seed交換はしない。\n- 理論ゲート: 固定点残差、円板積分、対数積分、Jacobian有限差分。\n  積分/残差基準1e-9、有限差分1e-7。失敗時は軌道生成しない。\n- beta=2,epsilon=0.9ではaがfloat64で1へ丸められ得る。aの飽和を記録し、\n  atanh(a)の数値評価で無限大を作らずgammaの区間付き求根を使用する。\n\n## 図・保存と再現\n\n1. 結合強度に対する縦横指数と不変尺度：理論曲線、seed平均と区間。\n2. epsilon=0,0.5のCayley点群と角度密度：各betaのseed=0、最大4096表示点、密度は全点。\n3. 代表seedの横方向対数感度累積：2の累乗時刻と最終時刻、理論lambda_perp*tと比較。\n\nPNG/PDF/SVG、元CSV、代表軌道NPZ、設定・理論・環境・ハッシュをこのrunに保存する。\nGitHubに送るのはコードとテストだけで、生成データ・図はローカルに留める。\n入口はadditive_sync.pyの1件。既存E1テストを拡張し、旧run実装へのimportは追加しない。\n白背景、青=横方向、琥珀=縦方向、破線=理論、点=数値。複素平面は等アスペクト共通軸。\n\n実行: `python additive_sync.py --theory`、`--run`、`--plot`、`--validate`。\n生成結果は上書きせず、追試は別フォルダで行う。\n\n## 結果\n\n事前登録時は未実行。以下に結果を追記する。\n"

ROOT = Path(__file__).resolve().parent
CONFIG = dict(betas=[1.0001, 1.01, 1.1, 2.], epsilons=[0., .01, .1, .5, .9],
              seeds=32, n=200000, entropy=20260907)
BLUE, AMBER = '#236C93', '#AA7228'


def dump(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_table(path: Path) -> List[Dict[str, float]]:
    with path.open(encoding='utf-8') as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def predictions(beta: float, epsilon: float) -> Dict[str, float]:
    """正の固定尺度と不変測度に関する縦横指数を返す。"""
    if not beta > 1 or not 0 <= epsilon < 1:
        raise ValueError('requires beta>1 and 0<=epsilon<1')
    def ratio(g: float) -> float:
        return beta+epsilon-1 if g == 0 else np.tanh(beta*g)/g+epsilon-1
    g = float(brentq(ratio, 0., 1/(1-epsilon), xtol=5e-15))
    a = float(np.tanh(beta*g))
    return dict(beta=beta, epsilon=epsilon, gamma=g, a=a,
                parallel=float(np.log(beta)+2*np.log(a+np.sqrt(1+epsilon/beta))),
                transverse=float(np.log(beta)+2*np.log(a+np.sqrt(1-epsilon/beta))),
                lower_bound=float(np.log(2*beta-1)), saturated_a=int(a == 1.))


@njit
def orbit(beta: float, epsilon: float, x0: float, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同期軌道と縦横の対数倍率を、再正規化せず対数で記録する。"""
    states = np.empty(n); parallel = np.empty(n); transverse = np.empty(n)
    s = x0
    for i in range(n):
        u = np.tan(beta*s)
        parallel[i] = np.log(beta)+2*np.log(np.hypot(u, np.sqrt(1+epsilon/beta)))
        transverse[i] = np.log(beta)+2*np.log(np.hypot(u, np.sqrt(1-epsilon/beta)))
        s = u+epsilon*s
        states[i] = s
    return states, parallel, transverse


def theory(root: Path) -> None:
    """独立積分とJacobian差分を通すまで軌道生成を許可しない。"""
    if (root/'theory.json').exists():
        raise FileExistsError('theory exists')
    root.mkdir(parents=True, exist_ok=True)
    (root/'preregistration.md').write_bytes(PREREGISTRATION.encode('utf-8'))
    dump(root/'config.json', CONFIG)
    checks, rows = [], []
    def check(name: str, beta: float, epsilon: float, error: float, tolerance: float = 1e-9) -> None:
        checks.append(dict(name=name, beta=beta, epsilon=epsilon, error=float(error), tolerance=tolerance, passed=bool(error <= tolerance)))
    for beta in CONFIG['betas']:
        for epsilon in CONFIG['epsilons']:
            p = predictions(beta, epsilon); rows.append(p); g, a = p['gamma'], p['a']
            check('fixed point', beta, epsilon, abs((1-epsilon)*g-a))
            for key, sign in [('parallel', 1), ('transverse', -1)]:
                c = np.sqrt(1+sign*epsilon/beta)
                value, _ = quad(lambda theta: (np.log(beta)+2*np.log(np.hypot(a*np.tan(theta), c)))/np.pi,
                                -np.pi/2, np.pi/2, epsabs=1e-11, epsrel=1e-11)
                check('log integral '+key, beta, epsilon, abs(value-p[key]))
            for nq in [4096, 8192]:
                z = .5*np.exp(2j*np.pi*(np.arange(nq)+.5)/nq)
                s = 1j*g*(1+z)/(1-z); y = np.tan(beta*s)+epsilon*s; h = (y-1j*g)/(y+1j*g)
                for k in [1, 2, 4]:
                    check(f'Cayley mean nq={nq} k={k}', beta, epsilon, abs(np.mean(h**k)))
            for s in [0., .1, -.2]:
                d = 1e-6
                dx = np.tan(beta*(s+d))+epsilon*(s-d)
                dy = np.tan(beta*(s-d))+epsilon*(s+d)
                expected = beta/(np.cos(beta*s)**2)-epsilon
                check('transverse Jacobian', beta, epsilon, abs((dx-dy)/(2*d)-expected), 1e-7)
            check('positive lower bound', beta, epsilon, 0. if p['transverse'] > p['lower_bound'] > 0 else 1.)
    table(root/'predictions.csv', rows)
    passed = all(c['passed'] for c in checks)
    dump(root/'theory.json', dict(passed=passed, checks=checks, code_sha256=digest(Path(__file__)),
         config_sha256=digest(root/'config.json'), predictions_sha256=digest(root/'predictions.csv'),
         preregistration_sha256=digest(root/'preregistration.md')))
    if not passed:
        raise ArithmeticError('theory gate failed')
    print(json.dumps(dict(theory_passed=True, checks=len(checks), max_error=max(c['error'] for c in checks))))


def run(root: Path) -> None:
    """固定条件の同期軌道を独立seedで生成し、状態と変分を分けて保存する。"""
    gate = json.loads((root/'theory.json').read_text(encoding='utf-8'))
    paths = dict(code=Path(__file__), config=root/'config.json', predictions=root/'predictions.csv', preregistration=root/'preregistration.md')
    if not gate['passed'] or any(gate[key+'_sha256'] != digest(path) for key, path in paths.items()):
        raise ValueError('theory/code/config mismatch')
    out = root/'artifacts'
    if out.exists():
        raise FileExistsError('run exists')
    out.mkdir(); cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    dump(root/'environment.json', dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
         numba=numba.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(), fastmath=False))
    orbit(1.1, .1, .2, 4)
    started = time.perf_counter(); rows, traces, reps = [], [], {}
    times = sorted(set([1]+[2**j for j in range(1, 19) if 2**j <= cfg['n']]+[cfg['n']]))
    for bi, beta in enumerate(cfg['betas']):
        for ei, epsilon in enumerate(cfg['epsilons']):
            p = predictions(beta, epsilon); g = p['gamma']
            for seed in range(cfg['seeds']):
                rng = np.random.default_rng(np.random.SeedSequence([cfg['entropy'], 2, bi, ei, seed]))
                x0 = float(g*rng.standard_cauchy()); x, lp, lt = orbit(beta, epsilon, x0, cfg['n'])
                if not all(np.isfinite(v).all() for v in [x, lp, lt]):
                    dump(out/'failure.json', dict(beta=beta, epsilon=epsilon, seed=seed, x0=x0)); raise ArithmeticError('nonfinite')
                z = np.exp(1j*(2*np.arctan2(x, g)-np.pi)); avg = z.mean(); unit = float(np.max(abs(abs(z)-1)))
                if unit > 1e-12:
                    raise ArithmeticError('unit circle deviation')
                sorted_x = np.sort(x); cdf = .5+np.arctan(sorted_x/g)/np.pi; ranks = np.arange(1, len(x)+1)/len(x)
                ks = float(max(np.max(ranks-cdf), np.max(cdf-(ranks-1/len(x)))))
                q = np.quantile(x, [.25, .75])
                rows.append(dict(beta=beta, epsilon=epsilon, seed=seed, x0=x0, parallel=lp.mean(), transverse=lt.mean(),
                                 scale=(q[1]-q[0])/2, ks=ks, cayley_real=avg.real, cayley_imag=avg.imag,
                                 unit_error=unit, max_abs=np.max(abs(x)), zeros=int(np.sum(x == 0))))
                if seed == 0:
                    reps[f'b{beta}_e{epsilon}'] = x
                    cumulative = np.cumsum(lt)
                    for t in times:
                        traces.append(dict(beta=beta, epsilon=epsilon, t=t, log_gain=cumulative[t-1], theory=p['transverse']*t))
            print(f'beta={beta} epsilon={epsilon}: {cfg["seeds"]} seeds', flush=True)
    table(out/'seeds.csv', rows); table(out/'log_gain.csv', traces)
    np.savez_compressed(out/'representative_orbits.npz', **reps)
    dump(root/'execution.json', dict(seconds=time.perf_counter()-started, trajectories=len(rows), updates=len(rows)*cfg['n']))


def plot(root: Path) -> None:
    """保存CSVと代表点から論文用の図を再生成する。"""
    out = root/'artifacts'; rows = read_table(out/'seeds.csv'); pred = read_table(root/'predictions.csv')
    summary = []
    for p in pred:
        matching = [v for v in rows if v['beta'] == p['beta'] and v['epsilon'] == p['epsilon']]
        for key in ['parallel', 'transverse', 'scale']:
            vals = np.array([r[key] for r in matching]); mean = float(vals.mean())
            half = float(student.ppf(.975, len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals)))
            target = p['gamma'] if key == 'scale' else p[key]
            summary.append(dict(beta=p['beta'], epsilon=p['epsilon'], observable=key, mean=mean, half95=half, theory=target, error=mean-target))
    table(out/'summary.csv', summary)
    def save(fig: Any, name: str) -> None:
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'{name}.{ext}', dpi=240)
        plt.close(fig)
    betas = sorted(set(p['beta'] for p in pred)); fig, axes = plt.subplots(4, 2, figsize=(10, 12), layout='constrained')
    for bi, beta in enumerate(betas):
        for key, color in [('transverse', BLUE), ('parallel', AMBER), ('scale', BLUE)]:
            ax = axes[bi, int(key == 'scale')]; rr = [r for r in summary if r['beta'] == beta and r['observable'] == key]
            ax.errorbar([r['epsilon'] for r in rr], [r['mean'] for r in rr], yerr=[r['half95'] for r in rr], fmt='o', color=color, label=key)
            ax.plot([r['epsilon'] for r in rr], [r['theory'] for r in rr], '--', color=color)
            ax.set(xlabel='epsilon', ylabel='Scale' if key == 'scale' else 'Lyapunov exponent', title=f'beta={beta}'); ax.legend(fontsize=8)
        axes[bi, 0].axhline(0, color='black', lw=.7)
    fig.suptitle('Additive coupling: theory (dashed) and 32-seed means | approximate 95% CI')
    save(fig, 'fig1_stability_scale')
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), layout='constrained')
    with np.load(out/'representative_orbits.npz') as data:
        for bi, beta in enumerate(betas):
            for ei, epsilon in enumerate([0., .5]):
                x = data[f'b{beta}_e{epsilon}']; g = predictions(beta, epsilon)['gamma']; z = np.exp(1j*(2*np.arctan2(x, g)-np.pi))
                index = np.linspace(0, len(z)-1, 4096, dtype=int); ax = axes[bi, 2*ei]
                ax.scatter(z[index].real, z[index].imag, s=2, color=BLUE); ax.set_aspect('equal')
                ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel='Real', ylabel='Imaginary', title=f'beta={beta}, epsilon={epsilon}')
                ax = axes[bi, 2*ei+1]; ax.hist(np.angle(z), bins=np.linspace(-np.pi, np.pi, 65), density=True, histtype='step', color=BLUE)
                ax.axhline(1/(2*np.pi), color='black', ls='--'); ax.set(xlabel='Angle', ylabel='Density')
    fig.suptitle('Matched Cayley coordinates on synchronization manifold | representative seed=0')
    save(fig, 'fig2_complex_plane')
    traces = read_table(out/'log_gain.csv'); fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, beta in zip(axes.flat, betas):
        for ei, epsilon in enumerate(CONFIG['epsilons']):
            rr = [v for v in traces if v['beta'] == beta and v['epsilon'] == epsilon]; color = plt.get_cmap('tab10')(ei)
            ax.plot([v['t'] for v in rr], [v['log_gain'] for v in rr], color=color, label=f'epsilon={epsilon}')
            ax.plot([v['t'] for v in rr], [v['theory'] for v in rr], '--', color=color)
        ax.set(xscale='log', xlabel='Time', ylabel='Cumulative log transverse gain', title=f'beta={beta}'); ax.legend(fontsize=7)
    fig.suptitle('Linear transverse sensitivity | representative seed=0 | dashed: theory')
    save(fig, 'fig3_transverse_gain'); seal(root)


def seal(root: Path) -> None:
    dump(root/'sha256.json', {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()
         and '__pycache__' not in p.parts and p.name not in ['sha256.json', 'validation.json']})


def validate(root: Path) -> None:
    """代表軌道を再計算し、保存状態と指数を独立に検証する。"""
    hashes = json.loads((root/'sha256.json').read_text(encoding='utf-8'))
    assert all(digest(root/p) == h for p, h in hashes.items())
    rows = read_table(root/'artifacts/seeds.csv'); cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert len(rows) == len(cfg['betas'])*len(cfg['epsilons'])*cfg['seeds']
    error = 0.
    with np.load(root/'artifacts/representative_orbits.npz') as data:
        for row in rows:
            if row['seed'] != 0:
                continue
            beta, epsilon = row['beta'], row['epsilon']; x = data[f'b{beta}_e{epsilon}']
            replay, _, _ = orbit(beta, epsilon, row['x0'], cfg['n']); assert np.array_equal(replay, x)
            previous = np.r_[row['x0'], x[:-1]]; u = np.tan(beta*previous)
            estimate = np.mean(np.log(beta*(1+u*u)-epsilon))
            error = max(error, abs(estimate-row['transverse']))
    assert error < 1e-9
    dump(root/'validation.json', dict(passed=True, hashes=len(hashes), replayed_seeds=20, max_log_error=error))
    print(json.dumps(dict(validated=True, hashes=len(hashes), max_log_error=error)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for action in ['theory', 'run', 'plot', 'validate']:
        group.add_argument('--'+action, action='store_true')
    args = parser.parse_args()
    for action in ['theory', 'run', 'plot', 'validate']:
        if getattr(args, action):
            globals()[action](ROOT)
