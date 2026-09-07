"""写像出力の拡散結合で局所安定性と有限距離同期を分けて検証する。"""
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
from scipy.optimize import brentq
from scipy.stats import t as student

PREREGISTRATION = "# E3：タンジェント写像出力の拡散結合\n\n## 事前登録\n\nE2相互加算結合と区別し、F(x)=tan(beta x)に対して\nx'=(1-kappa)F(x)+kappa F(y)、y'=kappa F(x)+(1-kappa)F(y)を調べる。\nbeta>1、0<=kappa<=1。同期上ではs'=F(s)であり、中心Cauchy不変尺度gammaは\ngamma=tanh(beta gamma)の正解のまま。一般の非同期二変数分布の閉包は仮定しない。\n\n## 理論と予想（実行前）\n\n同期上のJacobianはF'(s)[[1-kappa,kappa],[kappa,1-kappa]]。\n縦倍率はF'(s)、横倍率は(1-2kappa)F'(s)。不変測度上で\n\n    lambda_0=log(beta)+2log(1+gamma),\n    lambda_perp=lambda_0+log|1-2kappa|,\n    (1-exp(-lambda_0))/2 < kappa < (1+exp(-lambda_0))/2\n\nが局所横安定条件。等号は中立であり収束保証はない。\nkappa=1/2では両出力が同じ和となり、有限入力・出力なら厳密実数でも一歩で同期。\nこの特例の指数は負の無限大であり、有限数値の平均と混ぜずNoneで記録する。\n結合行列のモード分解は[master stability functionの原論文](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.2109)\nに対応する考え方で、上の離散tan系の式はここでJacobianから導出した。\n\nlambda_0を典型軌道の指数と結びつける際は、E1/E2で整理した尺度適合Cayley像のinner写像・混合性の議論を用いる。\n枝内のF'(x)>=betaだけから、極と無限枝を含む大域的ergodicityを結論しない。\n\n負の条件付き指数は、有限距離の初期条件すべての同期を保証しない。\ntanの極をまたぐ有限摂動と、有限精度による完全一致・吸収を区別する。\n境界近傍では有限時間指数と理論値の符号が異なる可能性がある。\n\n## 固定条件\n\n- beta=1.0001,1.01,1.1,2、32独立seed、各200000step。\n- q=-1,-0.25,0,0.25,0.75 とし、kappa=(1-exp((q-1)lambda_0))/2。\n  理論横指数がq*lambda_0となる5条件に、kappa=0と1/2を加えた7条件。\n  数値走査は下半区間のみ。上半区間との対称性は写像の交換対称性で確認する。\n- SeedSequence([20260907,3,beta_index,seed])。各seedで中心s0と独立x0,y0をCauchy生成。\n- near初期値: (s0+gamma*1e-8/2,s0-gamma*1e-8/2)。\n- independent初期値: 二つの独立C(0,gamma)。同じ初期値を全kappaで使用する。\n- 各seedの同期上軌道を一度だけ生成してlambda_0を推定。kappaごとに厳密な対数係数を足し、\n  非同期の二軌道統計とは別に保存する。\n- 同期判定は最後の20000stepの最大|x-y|/gammaと最大Cayley弦距離がともに1e-6以下。\n  初回完全一致時刻(x==y)を別記録。正指数側での完全一致は数値吸収として警告し、\n  理論的な安定同期の証拠とはしない。初期時点ですでに同値なら記録し再抽選しない。\n- nearの|x-y|/gammaが最初に0.01を超える時刻も記録。収束時間と混同しない。\n- 時系列は2の累乗時刻・最終時刻、複素点群は代表seedの等間隔4096点を保存。\n  モードz_x conjugate(z_y)の配置も表示し、単に両者が円周にあることと一致を区別する。\n- 分率は32seedの記述統計とし、吸引域体積の精密推定や母集団保証とはしない。\n- float64、Numba標準tan、fastmathなし。クリップ、ノイズ、再抽選、結果に応じた延長はしない。\n- 理論ゲート: 固定尺度、Jacobian、交換対称性、kappa=1/2の一歩同期、\n  有限区間の指数恒等式。有限差分基準1e-7、他1e-10。\n\n## 成果物・実装\n\n入口output_diffusion.pyの1件。既存E1テストに一つのテスト単位を追加する。\n理論・条件はコードに埋め込み、--theoryでpreregistration.mdと理論値を先に生成する。\n--run、--plot、--validateの順で実行。上書きは拒否し、別runで再実行する。\n設定、環境、seed初期値、条件別CSV、表示点NPZ、監査、ハッシュ、PNG/PDF/SVGをこのrunに保存。\n全生軌道は保存しない。初期値・コード・環境から代表軌道を再生して保存点と指標を照合する。\nGitHubにはコードとテストのみを送信し、データ・図・結果記録はローカルに保持する。\n\n図は、(1)理論安定境界と同期上の数値横指数、(2)near/independent別の有限時間同期率と完全一致率、\n(3)代表seedのCayley相対位相の複素平面、(4)代表seedの距離時系列。\n複素平面は等アスペクト共通軸、背景白。安定側青、不安定側琥珀、中立灰、\n完全一致を別マーカーで識別する。ゼロ距離は対数軸から黙って消さず、別に件数・時刻を表示する。\n\n## 結果\n\n事前登録時は未実行。以下に観測と解釈を追記する。\n"

ROOT = Path(__file__).resolve().parent
CONFIG = dict(betas=[1.0001, 1.01, 1.1, 2.], ratios=[-1., -.25, 0., .25, .75],
              seeds=32, n=200000, tail=20000, threshold=1e-6, entropy=20260907)


def dump(path: Path, obj: Any) -> None:
    path.write_bytes(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parameters(beta: float) -> Tuple[float, float]:
    """元のtan系の正尺度と不変測度上の指数を返す。"""
    if beta <= 1:
        raise ValueError('requires beta>1')
    def residual(g: float) -> float:
        return beta-1 if g == 0 else np.tanh(beta*g)/g-1
    g = float(brentq(residual, 0., 1., xtol=5e-15))
    return g, float(np.log(beta)+2*np.log1p(g))


def conditions(beta: float) -> List[Dict[str, Any]]:
    """結果を見ずに横指数の比から結合強度を定める。"""
    g, lam = parameters(beta)
    rows = [dict(beta=beta, gamma=g, lambda0=lam, name='uncoupled', kappa=0., target=lam, regime='unstable')]
    for q in CONFIG['ratios']:
        k = float(-np.expm1((q-1)*lam)/2)
        rows.append(dict(beta=beta, gamma=g, lambda0=lam, name=f'q{q}', kappa=k, target=q*lam,
                         regime='stable' if q < 0 else 'unstable' if q > 0 else 'neutral'))
    rows.append(dict(beta=beta, gamma=g, lambda0=lam, name='one_step', kappa=.5, target=None, regime='one_step'))
    return rows


@njit
def pair(beta: float, kappa: float, x0: float, y0: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """同時更新で二状態を反復する。等値への強制置換やクリップは行わない。"""
    xs = np.empty(n); ys = np.empty(n)
    x, y = x0, y0
    for i in range(n):
        u, v = np.tan(beta*x), np.tan(beta*y)
        x, y = (1-kappa)*u+kappa*v, kappa*u+(1-kappa)*v
        xs[i], ys[i] = x, y
    return xs, ys


@njit
def diagonal_exponent(beta: float, s0: float, n: int) -> float:
    """非同期二軌道とは独立に、同期多様体上で基準指数を測る。"""
    s, total = s0, 0.
    for _ in range(n):
        s = np.tan(beta*s)
        total += np.log(beta)+2*np.log(np.hypot(1., s))
    return total/n


def theory(root: Path) -> None:
    """理論と差分・対称性を確認してから実験を許可する。"""
    if (root/'theory.json').exists():
        raise FileExistsError('theory exists')
    root.mkdir(parents=True, exist_ok=True)
    (root/'preregistration.md').write_bytes(PREREGISTRATION.encode('utf-8')); dump(root/'config.json', CONFIG)
    checks, rows = [], []
    for beta in CONFIG['betas']:
        for c in conditions(beta):
            rows.append(c); k = c['kappa']; g = c['gamma']
            checks.append(dict(name='fixed scale', error=abs(g-np.tanh(beta*g)), tolerance=1e-10))
            for s in [0., .1, -.2]:
                d = 1e-6; x, y = pair(beta, k, s+d, s-d, 1)
                expected = (1-2*k)*beta/(np.cos(beta*s)**2)
                checks.append(dict(name='transverse Jacobian', error=abs((x[0]-y[0])/(2*d)-expected), tolerance=1e-7))
            x, y = pair(beta, k, .13, -.07, 1); a, b = pair(beta, 1-k, .13, -.07, 1)
            checks.append(dict(name='exchange symmetry', error=max(abs(x[0]-b[0]), abs(y[0]-a[0])), tolerance=1e-10))
            if k == .5:
                checks.append(dict(name='one step identity', error=abs(x[0]-y[0]), tolerance=0.))
            else:
                checks.append(dict(name='exponent target', error=abs(c['lambda0']+np.log1p(-2*k)-c['target']), tolerance=1e-10))
                x, _ = pair(beta, 0., .031, .031, 32)
                state = np.r_[.031, x[:-1]]; derivative = beta/np.cos(beta*state)**2
                left = np.mean(np.log(abs((1-2*k)*derivative)))
                right = diagonal_exponent(beta, .031, 32)+np.log1p(-2*k)
                checks.append(dict(name='finite time identity', error=abs(left-right), tolerance=1e-10))
    passed = all(r['error'] <= r['tolerance'] for r in checks)
    table(root/'predictions.csv', rows)
    dump(root/'theory.json', dict(passed=passed, checks=checks, code_sha256=sha(Path(__file__)),
         config_sha256=sha(root/'config.json'), predictions_sha256=sha(root/'predictions.csv'), preregistration_sha256=sha(root/'preregistration.md')))
    if not passed:
        raise ArithmeticError('theory gate failed')
    print(json.dumps(dict(theory_passed=True, checks=len(checks), max_error=float(max(r['error'] for r in checks)))))


def diagnostics(x: np.ndarray, y: np.ndarray, gamma: float, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    """有限距離とCayley距離を併用し、丸めによる完全一致を別記録する。"""
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ArithmeticError('nonfinite orbit')
    error = abs(x-y)/gamma
    chord = 2*gamma*abs(x-y)/(np.hypot(x, gamma)*np.hypot(y, gamma))
    exact = np.flatnonzero(x == y); escape = np.flatnonzero(error > .01)
    tail_real = float(error[-cfg['tail']:].max()); tail_chord = float(chord[-cfg['tail']:].max())
    return dict(tail_real=tail_real, tail_chord=tail_chord,
                finite_sync=int(tail_real <= cfg['threshold'] and tail_chord <= cfg['threshold']),
                first_exact=int(exact[0]+1) if len(exact) else -1,
                first_escape=int(escape[0]+1) if len(escape) else -1,
                max_abs=float(max(abs(x).max(), abs(y).max()))), error


def run(root: Path) -> None:
    """同じ初期値で結合条件を比較し、理論安定性と有限時間同期率を保存する。"""
    gate = json.loads((root/'theory.json').read_text(encoding='utf-8'))
    paths = dict(code=Path(__file__), config=root/'config.json', predictions=root/'predictions.csv', preregistration=root/'preregistration.md')
    if not gate['passed'] or any(gate[k+'_sha256'] != sha(v) for k, v in paths.items()):
        raise ValueError('theory/code/config mismatch')
    out = root/'artifacts'
    if out.exists():
        raise FileExistsError('run exists')
    out.mkdir(); cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    dump(root/'environment.json', dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
         numba=numba.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(), fastmath=False))
    pair(1.1, .1, .1, .2, 2); diagonal_exponent(1.1, .1, 2)
    started = time.perf_counter(); rows, linear, traces, points = [], [], [], {}
    times = sorted(set([1]+[2**j for j in range(1, 19) if 2**j <= cfg['n']]+[cfg['n']]))
    indices = np.linspace(0, cfg['n']-1, 4096, dtype=int); points['indices'] = indices
    for bi, beta in enumerate(cfg['betas']):
        g, _ = parameters(beta)
        for seed in range(cfg['seeds']):
            rng = np.random.default_rng(np.random.SeedSequence([cfg['entropy'], 3, bi, seed]))
            center, ix, iy = g*rng.standard_cauchy(3)
            lam = diagonal_exponent(beta, center, cfg['n'])
            for c in conditions(beta):
                k = c['kappa']; key = f'b{beta}_{c["name"]}'
                linear.append(dict(beta=beta, name=c['name'], seed=seed, center=center, lambda0=lam,
                                   estimate=lam+np.log1p(-2*k) if k < .5 else None, target=c['target']))
                for initial, x0, y0 in [('near', center+g*1e-8/2, center-g*1e-8/2), ('independent', ix, iy)]:
                    x, y = pair(beta, k, x0, y0, cfg['n'])
                    result, error = diagnostics(x, y, g, cfg)
                    rows.append(dict(beta=beta, name=c['name'], kappa=k, regime=c['regime'], seed=seed,
                                     initial=initial, x0=x0, y0=y0, initial_equal=int(x0 == y0), **result))
                    if seed == 0:
                        points[key+'_'+initial] = np.array([x[indices], y[indices]])
                        for t in times:
                            traces.append(dict(beta=beta, name=c['name'], initial=initial, t=t, error=error[t-1]))
            if (seed+1) % 16 == 0:
                print(f'beta={beta}: {seed+1}/{cfg["seeds"]} seeds', flush=True)
        table(out/'pairs.csv', rows); table(out/'linear.csv', linear); table(out/'traces.csv', traces)
    np.savez_compressed(out/'points.npz', **points)
    dump(root/'execution.json', dict(seconds=time.perf_counter()-started, pair_runs=len(rows), pair_state_updates=2*len(rows)*cfg['n']))


def plot(root: Path) -> None:
    """保存CSVから局所指数と有限時間の観測を別パネルで描く。"""
    out = root/'artifacts'; rows = read(out/'pairs.csv'); linear = read(out/'linear.csv'); summary = []
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    colors = dict(stable='#236C93', unstable='#AA7228', neutral='#737373', one_step='#72589A')
    def save(fig: Any, name: str) -> None:
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'{name}.{ext}', dpi=240)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, beta in zip(axes.flat, cfg['betas']):
        g, lam = parameters(beta); grid = np.linspace(0, .499, 300)
        ax.plot(grid, lam+np.log1p(-2*grid), color='black', label='Theory')
        for c in conditions(beta):
            if c['kappa'] == .5:
                continue
            vals = np.array([float(r['estimate']) for r in linear if float(r['beta']) == beta and r['name'] == c['name']])
            half = student.ppf(.975, len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))
            ax.errorbar(c['kappa'], vals.mean(), yerr=half, fmt='o', color=colors[c['regime']])
        ax.axhline(0, color='gray', ls=':'); ax.axvline(-np.expm1(-lam)/2, color='gray', ls='--')
        ax.set(xlabel='kappa', ylabel='Transverse exponent', title=f'beta={beta}')
    fig.suptitle('Local transverse stability | seed-level approximate 95% CI | kappa=0.5: exact one-step synchronization')
    save(fig, 'fig1_local_stability')
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout='constrained')
    for bi, beta in enumerate(cfg['betas']):
        cs = conditions(beta)
        for ii, initial in enumerate(['near', 'independent']):
            for j, c in enumerate(cs):
                rr = [r for r in rows if float(r['beta']) == beta and r['name'] == c['name'] and r['initial'] == initial]
                synced = sum(int(r['finite_sync']) for r in rr); exact = sum(int(r['first_exact']) >= 0 for r in rr)
                summary.append(dict(beta=beta, name=c['name'], kappa=c['kappa'], regime=c['regime'], initial=initial,
                                    runs=len(rr), synchronized=synced, exact=exact))
                axes[bi, ii].bar(j, synced/len(rr), color=colors[c['regime']], alpha=.6)
                axes[bi, ii].scatter(j, exact/len(rr), color='black', marker='x')
            axes[bi, ii].set(ylim=(-.05, 1.08), xticks=range(len(cs)), xticklabels=[c['name'] for c in cs],
                             ylabel='Fraction of 32 seeds', title=f'beta={beta}, {initial}')
            axes[bi, ii].tick_params(axis='x', rotation=30)
    fig.suptitle('Finite-time synchronization (bars) and floating-point equality (crosses)\nNeither is a global basin guarantee; positive-exponent equality is numerical absorption')
    save(fig, 'fig2_sync_fractions'); table(out/'summary.csv', summary)
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), layout='constrained')
    with np.load(out/'points.npz') as data:
        for bi, beta in enumerate(cfg['betas']):
            g, _ = parameters(beta)
            for j, name in enumerate(['q-1.0', 'q0.0', 'q0.75']):
                x, y = data[f'b{beta}_{name}_independent']; zx = np.exp(1j*(2*np.arctan2(x, g)-np.pi)); zy = np.exp(1j*(2*np.arctan2(y, g)-np.pi)); w = zx*zy.conj()
                ax = axes[bi, j]; ax.scatter(w.real, w.imag, s=3, color=colors[['stable', 'neutral', 'unstable'][j]])
                ax.scatter([1], [0], color='black', marker='+', s=80)
                ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel='Real relative phase', ylabel='Imaginary relative phase', title=f'beta={beta}, {name}'); ax.set_aspect('equal')
    fig.suptitle('Cayley relative phase z_x conjugate(z_y) | seed=0, independent initialization\nCross at 1 indicates equal phases; real-state distance is assessed separately')
    save(fig, 'fig3_complex_relative_phase')
    traces = read(out/'traces.csv'); fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout='constrained')
    for bi, beta in enumerate(cfg['betas']):
        for ii, initial in enumerate(['near', 'independent']):
            for c in conditions(beta):
                rr = [r for r in traces if float(r['beta']) == beta and r['name'] == c['name'] and r['initial'] == initial]
                axes[bi, ii].plot([int(r['t']) for r in rr], [float(r['error']) for r in rr], label=c['name'])
            axes[bi, ii].set(xscale='log', yscale='symlog', xlabel='Time', ylabel='|x-y| / gamma', title=f'beta={beta}, {initial}')
            axes[bi, ii].set_yscale('symlog', linthresh=1e-12); axes[bi, ii].legend(fontsize=6)
    fig.suptitle('Representative distances | symlog retains exact zeros | seed=0')
    save(fig, 'fig4_distance_traces'); seal(root)


def seal(root: Path) -> None:
    dump(root/'sha256.json', {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()
         and '__pycache__' not in p.parts and p.name not in ['sha256.json', 'validation.json']})


def validate(root: Path) -> None:
    """代表二軌道を再生し、保存点と同期判定を照合する。"""
    hashes = json.loads((root/'sha256.json').read_text(encoding='utf-8'))
    assert all(sha(root/p) == h for p, h in hashes.items())
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8')); rows = read(root/'artifacts/pairs.csv')
    assert len(rows) == len(cfg['betas'])*7*2*cfg['seeds']
    count = 0
    with np.load(root/'artifacts/points.npz') as data:
        for r in rows:
            if int(r['seed']) != 0:
                continue
            beta, k = float(r['beta']), float(r['kappa']); x, y = pair(beta, k, float(r['x0']), float(r['y0']), cfg['n'])
            expected = data[f'b{beta}_{r["name"]}_{r["initial"]}']
            assert np.array_equal(np.array([x[data['indices']], y[data['indices']]]), expected)
            result, _ = diagnostics(x, y, parameters(beta)[0], cfg)
            assert result['finite_sync'] == int(r['finite_sync']) and result['first_exact'] == int(r['first_exact'])
            count += 1
    dump(root/'validation.json', dict(passed=True, hashes=len(hashes), replayed_pairs=count))
    print(json.dumps(dict(validated=True, hashes=len(hashes), replayed_pairs=count)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for name in ['theory', 'run', 'plot', 'validate']:
        group.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    for name in ['theory', 'run', 'plot', 'validate']:
        if getattr(args, name):
            globals()[name](ROOT)
