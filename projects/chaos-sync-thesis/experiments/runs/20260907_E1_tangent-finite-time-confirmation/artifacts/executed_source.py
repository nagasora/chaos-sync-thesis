"""独立初期値のTM平均と有限時間理論を比較する追試入口。"""
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

ROOT = Path(__file__).resolve().parent
CONFIG = dict(betas=[1.0001, 1.01, 1.1, 2.0], seeds=64,
              lengths=[50000, 200000, 1000000], modes=[1, 2, 4, 8],
              lags=[1, 64, 1024, 4096], correlation_modes=[1, 4], entropy=20260907)


def dump(path: Path, obj: Any) -> None:
    path.write_bytes(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, float]]:
    with path.open(encoding='utf-8') as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def gamma_fixed(beta: float) -> float:
    """ゼロ根を除外して中心Cauchy不変尺度を求める。"""
    if beta <= 1:
        raise ValueError('beta > 1 required')
    def residual(g: float) -> float:
        z = g*g
        return z*(1/3+z*(1/5+z*(1/7+z/9)))-(beta-1) if g < .01 else np.arctanh(g)/g-beta
    return float(brentq(residual, 0, np.nextafter(1., 0.), xtol=5e-15))


def mean_square(n: int, rho: float) -> float:
    """相関rho^hから有限長の二乗平均を安定な有限和で計算する。"""
    if n < 1 or not 0 <= rho <= 1:
        raise ValueError('invalid covariance')
    h = np.arange(1, n, dtype=float)
    return float((n+2*np.sum((n-h)*rho**h))/n**2)


@njit
def orbit(beta: float, x0: float, n: int) -> np.ndarray:
    """fastmathやクリップを使わずx1以降の実数軌道を生成する。"""
    x = np.empty(n, dtype=np.float64)
    value = x0
    for i in range(n):
        value = np.tan(beta*value)
        x[i] = value
    return x


def interval(values: np.ndarray) -> Tuple[float, float]:
    """独立seedを単位とした近似95%区間の中心と半幅。"""
    return float(np.mean(values)), float(student.ppf(.975, len(values)-1)*np.std(values, ddof=1)/np.sqrt(len(values)))


def theory(root: Path) -> None:
    """軌道生成前に理論予測と独立な係数照合を固定する。"""
    if (root/'theory.json').exists():
        raise FileExistsError('theory exists')
    root.mkdir(parents=True, exist_ok=True)
    dump(root/'config.json', CONFIG)
    errors, rows = [], []
    for beta in CONFIG['betas']:
        g = gamma_fixed(beta); r = beta*(1-g*g)
        errors.append(abs(g-np.tanh(beta*g)))
        for nquad in [4096, 8192]:
            z = .5*np.exp(2j*np.pi*(np.arange(nquad)+.5)/nquad)
            x = 1j*g*(1+z)/(1-z); y = np.tan(beta*x); hz = (y-1j*g)/(y+1j*g)
            for k in CONFIG['modes']:
                errors.append(float(abs(np.mean((hz/z)**k)-r**k)))
        for k in CONFIG['modes']:
            rho = r**k
            matrix = rho**abs(np.arange(31)[:, None]-np.arange(31)[None, :])
            errors.append(abs(mean_square(31, rho)-matrix.mean()))
            for n in CONFIG['lengths']:
                v = mean_square(n, rho)
                rows.append(dict(beta=beta, gamma=g, r=r, k=k, n=n, variance=v,
                                 rms=np.sqrt(v), effective_n=1/v))
    passed = max(errors) <= 1e-10
    write_csv(root/'predictions.csv', rows)
    dump(root/'theory.json', dict(passed=passed, max_error=max(errors), checks=len(errors),
         code_sha256=sha(Path(__file__)), config_sha256=sha(root/'config.json'),
         predictions_sha256=sha(root/'predictions.csv'), preregistration_sha256=sha(root/'README.md'),
         preregistration_text=(root/'README.md').read_text(encoding='utf-8')))
    if not passed:
        raise ArithmeticError('theory gate failed')
    print(json.dumps(dict(theory_passed=passed, checks=len(errors), max_error=max(errors))))


def run(root: Path) -> None:
    """独立seedの軌道と各観測長の指標を事前条件のまま記録する。"""
    gate = json.loads((root/'theory.json').read_text(encoding='utf-8'))
    if not gate['passed'] or gate['code_sha256'] != sha(Path(__file__)) or gate['config_sha256'] != sha(root/'config.json') or gate['predictions_sha256'] != sha(root/'predictions.csv') or gate['preregistration_sha256'] != sha(root/'README.md'):
        raise ValueError('theory/code/config/preregistration mismatch')
    out = root/'artifacts'
    if out.exists():
        raise FileExistsError('run exists')
    out.mkdir()
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    dump(root/'environment.json', dict(python=platform.python_version(), numpy=np.__version__,
         scipy=scipy.__version__, numba=numba.__version__, matplotlib=matplotlib.__version__,
         platform=platform.platform(), fastmath=False, engine='Numba float64 tan', code_sha256=sha(Path(__file__))))
    orbit(1.1, .2, 8)
    started = time.perf_counter()
    moments, correlations, diagnostics, densities = [], [], [], []
    samples = {}
    edges = np.linspace(-np.pi, np.pi, 65)
    for bi, beta in enumerate(cfg['betas']):
        g = gamma_fixed(beta)
        for seed in range(cfg['seeds']):
            rng = np.random.default_rng(np.random.SeedSequence([cfg['entropy'], bi, seed]))
            x0 = float(g*rng.standard_cauchy()); x = orbit(beta, x0, max(cfg['lengths']))
            if not np.isfinite(x).all():
                dump(out/'failure.json', dict(beta=beta, seed=seed, x0=x0, reason='nonfinite'))
                raise ArithmeticError('nonfinite orbit')
            z = np.exp(1j*(2*np.arctan2(x, g)-np.pi))
            unit = float(np.max(abs(abs(z)-1)))
            if unit > 1e-12:
                raise ArithmeticError('unit circle deviation')
            diagnostics.append(dict(beta=beta, seed=seed, x0=x0, zero_count=int(np.sum(x == 0)),
                                    max_abs=float(np.max(abs(x))), unit_error=unit))
            if seed == 0:
                samples[f'x_{beta}'] = x
            for k in cfg['modes']:
                zk = z**k
                for n in cfg['lengths']:
                    avg = zk[:n].mean()
                    moments.append(dict(beta=beta, seed=seed, n=n, k=k, real=avg.real, imag=avg.imag, square=abs(avg)**2))
                    if k in cfg['correlation_modes']:
                        for lag in cfg['lags']:
                            c = np.mean(zk[lag:n]*zk[:n-lag].conj())
                            correlations.append(dict(beta=beta, seed=seed, n=n, k=k, lag=lag, real=c.real, imag=c.imag))
            for n in cfg['lengths']:
                count = np.histogram(np.angle(z[:n]), bins=edges)[0]
                for b, value in enumerate(count):
                    densities.append(dict(beta=beta, seed=seed, n=n, bin=b, probability=value/n))
            if (seed+1) % 16 == 0:
                print(f'beta={beta} seeds={seed+1}/{cfg["seeds"]}', flush=True)
        write_csv(out/'moments.csv', moments)
        write_csv(out/'correlations.csv', correlations)
        write_csv(out/'diagnostics.csv', diagnostics)
        write_csv(out/'density.csv', densities)
    np.savez_compressed(out/'representative_orbits.npz', **samples)
    dump(root/'execution.json', dict(seconds=time.perf_counter()-started, orbits=len(diagnostics),
         updates=len(diagnostics)*max(cfg['lengths']), moments=len(moments), correlations=len(correlations)))


def summarize(root: Path) -> List[Dict[str, Any]]:
    """保存済みseed指標のみから理論比較を再集計する。"""
    moments = read_csv(root/'artifacts/moments.csv'); predictions = read_csv(root/'predictions.csv')
    result = []
    for p in predictions:
        values = np.array([v['square'] for v in moments if all(v[key] == p[key] for key in ['beta', 'k', 'n'])])
        mean, half = interval(values)
        result.append(dict(**p, empirical=mean, half95=half, ratio=mean/p['variance'],
                           interval_contains_theory=int(abs(mean-p['variance']) <= half)))
    write_csv(root/'artifacts/summary.csv', result)
    return result


def plot(root: Path) -> None:
    """保存した点・指標から複素平面と誤差図を3形式で出力する。"""
    out = root/'artifacts'; summary = summarize(root); moments = read_csv(out/'moments.csv')
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    colors = ['#236C93', '#AA7228', '#72589A']
    def save(fig: Any, name: str) -> None:
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'{name}.{ext}', dpi=240)
        plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(11, 12), layout='constrained')
    for bi, beta in enumerate(cfg['betas']):
        for ti, n in enumerate(cfg['lengths']):
            ax = axes[bi, ti]; rows = [v for v in moments if v['beta'] == beta and v['n'] == n and v['k'] == 1]
            p = next(v for v in summary if v['beta'] == beta and v['n'] == n and v['k'] == 1)
            theta = np.linspace(0, 2*np.pi, 256); radius = p['rms']
            ax.plot(radius*np.cos(theta), radius*np.sin(theta), '--', color='#303640', label='Theory RMS radius')
            ax.scatter([v['real'] for v in rows], [v['imag'] for v in rows], s=12, color=colors[ti])
            ax.scatter([0], [0], marker='+', color='black')
            limit = max(.002, 3*max(v['rms'] for v in summary if v['beta'] == beta and v['k'] == 1))
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel='Real mean', ylabel='Imaginary mean', title=f'beta={beta}, T={n:,}')
            ax.set_aspect('equal')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Complex mode mean: 64 independent seeds | k=1\nDashed circle is RMS radius, not a confidence region')
    save(fig, 'fig1_complex_means')
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), layout='constrained')
    for bi, beta in enumerate(cfg['betas']):
        for ki, k in enumerate(cfg['modes']):
            rows = [v for v in summary if v['beta'] == beta and v['k'] == k]
            ns = np.array([v['n'] for v in rows]); empirical = np.array([v['empirical'] for v in rows]); err = np.array([v['half95'] for v in rows]); theory_v = np.array([v['variance'] for v in rows])
            color = plt.get_cmap('tab10')(ki)
            axes[bi, 0].errorbar(ns, empirical, yerr=err, fmt='o', color=color, label=f'k={k}')
            axes[bi, 0].plot(ns, theory_v, '--', color=color)
            axes[bi, 1].errorbar(ns, empirical/theory_v, yerr=err/theory_v, fmt='o-', color=color)
        axes[bi, 0].set(xscale='log', yscale='log', xlabel='T', ylabel='Mean squared mode mean', title=f'beta={beta}'); axes[bi, 0].legend()
        axes[bi, 1].axhline(1, color='black', ls='--'); axes[bi, 1].set(xscale='log', xlabel='T', ylabel='Empirical / theory')
    fig.suptitle('Finite-time fluctuations | dashed: exact theory | bars: seed-level approximate 95% CI')
    save(fig, 'fig2_finite_time_scaling')
    density = read_csv(out/'density.csv')
    with np.load(out/'representative_orbits.npz') as data:
        fig, axes = plt.subplots(4, 2, figsize=(10, 12), layout='constrained')
        for bi, beta in enumerate(cfg['betas']):
            x = data[f'x_{beta}']; inds = np.linspace(0, len(x)-1, 4096, dtype=int)
            z = np.exp(1j*(2*np.arctan2(x[inds], gamma_fixed(beta))-np.pi))
            axes[bi, 0].scatter(z.real, z.imag, s=3, color=colors[0]); axes[bi, 0].set_aspect('equal')
            axes[bi, 0].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel='Real', ylabel='Imaginary', title=f'beta={beta}, seed=0')
            for ti, n in enumerate(cfg['lengths']):
                prob = np.array([v['probability'] for v in density if v['beta'] == beta and v['n'] == n]).reshape(cfg['seeds'], 64).mean(axis=0)
                axes[bi, 1].stairs(prob/(2*np.pi/64), np.linspace(-np.pi, np.pi, 65), color=colors[ti], label=f'T={n:,}')
            axes[bi, 1].axhline(1/(2*np.pi), ls='--', color='black'); axes[bi, 1].set(xlabel='Angle (radians)', ylabel='Density'); axes[bi, 1].legend(fontsize=8)
        fig.suptitle('Matched Cayley geometry and pooled angular density | 64 seeds')
        save(fig, 'fig3_circle_density')
    corr = read_csv(out/'correlations.csv'); fig, axes = plt.subplots(4, 2, figsize=(10, 12), layout='constrained')
    for bi, beta in enumerate(cfg['betas']):
        r = beta*(1-gamma_fixed(beta)**2)
        for ki, k in enumerate(cfg['correlation_modes']):
            for j, field in enumerate(['real', 'imag']):
                vals = [interval(np.array([v[field] for v in corr if v['beta'] == beta and v['n'] == max(cfg['lengths']) and v['k'] == k and v['lag'] == lag])) for lag in cfg['lags']]
                axes[bi, j].errorbar(cfg['lags'], [v[0] for v in vals], yerr=[v[1] for v in vals], fmt='o-', color=colors[ki], label=f'k={k}')
                axes[bi, j].plot(cfg['lags'], [r**(k*h) if j == 0 else 0 for h in cfg['lags']], '--', color=colors[ki])
                axes[bi, j].set(xscale='log', xlabel='Lag', ylabel=field, title=f'beta={beta}'); axes[bi, j].legend()
    fig.suptitle('Lag correlations at T=1,000,000 | 64 independent seeds | approximate 95% CI')
    save(fig, 'fig4_correlations')
    seal(root)


def seal(root: Path) -> None:
    dump(root/'sha256.json', {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['sha256.json', 'validation.json']})


def validate(root: Path) -> None:
    """ハッシュと代表軌道からの別式で集計の再現を確認する。"""
    hashes = json.loads((root/'sha256.json').read_text(encoding='utf-8'))
    assert all(sha(root/p) == expected for p, expected in hashes.items())
    rows = read_csv(root/'artifacts/moments.csv'); cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert len(rows) == len(cfg['betas'])*cfg['seeds']*len(cfg['lengths'])*len(cfg['modes'])
    error = 0.
    with np.load(root/'artifacts/representative_orbits.npz') as data:
        for beta in cfg['betas']:
            x = data[f'x_{beta}']; g = gamma_fixed(beta); z = (x-1j*g)/(x+1j*g)
            for row in rows:
                if row['beta'] == beta and row['seed'] == 0:
                    avg = np.mean(z[:int(row['n'])]**int(row['k']))
                    error = max(error, abs(avg-complex(row['real'], row['imag'])))
    assert error < 1e-10
    dump(root/'validation.json', dict(passed=True, hashes=len(hashes), max_independent_error=error, rows=len(rows)))
    print(json.dumps(dict(validation_passed=True, hashes=len(hashes), error=error)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for name in ['theory', 'run', 'plot', 'validate']:
        group.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    for name in ['theory', 'run', 'plot', 'validate']:
        if getattr(args, name):
            globals()[name](ROOT)
