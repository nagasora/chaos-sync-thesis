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

PREREGISTRATION = '# E1追試：TM時間平均の有限時間変動\n\n## 事前登録（軌道生成前）\n\n前回 `20260906_E1_tangent-tm-complex-plane` の代表seedで観測された偏りを、\n独立初期値と観測長で検証する。Booleの既存結論は変更しない。\n同期、学習、TMと他の読み出し器の優劣は対象外。\n\n中心Cauchy不変測度から初期化した厳密実数のtan(beta x)を理論対象とする。\nq(x)=(x-i gamma)/(x+i gamma)、gamma=tanh(beta gamma)>0、\nr=beta(1-gamma^2) とし、尺度適合した正次数モードを用いる。\n前回の円板内Taylor係数による導出から E[z]=0、\nE[z_(t+h)^k conjugate(z_t^k)]=r^(kh)。\n\n時間平均 m_(T,k)=T^(-1) sum_(t=1)^T z_t^k の二乗を展開すると、\n対角項T個と距離hの対が各T-h個あり、厳密に\n\n    V(T,k)=E[|m_(T,k)|^2]\n          =[T+2 sum_(h=1)^(T-1) (T-h) r^(kh)]/T^2.\n\nさらに H(0)=0 により正次数の非共役積の平均は0なので E[m^2]=0。\n従って Re m、Im m の分散はいずれも V/2、共分散は0。\nこれは複素平均が正規分布であることを意味しない。\nTが相関時間より十分長ければ V ~ (1+r^k)/(1-r^k)/T。\n有効標本数はこの二乗平均に関する量 N_eff=1/V と定義する。\nbeta=1.0001、k=1、T=200000ではRMSは約0.224と予想されるため、\n前回の平均の絶対値0.292だけで理論不一致とは言えない。\n\n## 固定条件と判定\n\n- beta: 1.0001, 1.01, 1.1, 2.0。float64のみ。\n- 64独立seed。NumPy SeedSequence([20260907, beta_index, seed])、seed=0..63。\n- 初期化 X0=gamma*standard_cauchy()。burn-inなし、x1から観測。\n- 同じ軌道の先頭T=50000,200000,1000000を集計。T間は独立ではない。\n- モード k=1,2,4,8。相関lag=1,64,1024,4096、k=1,4。\n- 主評価: seed平均 |m|^2 と厳密V、その比。補助: Re/Im平均と相関。\n- 独立seedを単位とするStudent-t 95%区間を表示。有限seedでの近似区間であり、\n  多重比較補正を行った一括合否や厳密な被覆保証には使わない。\n- 理論Vの明示和・Toeplitz行列・円板積分を先に検証。誤差基準1e-10。\n- 非有限値は停止。ゼロの出現、最大状態、単位円誤差を記録。クリップしない。\n- 窓変更・seed交換・結果を見た延長はしない。\n\n## 実装と図の契約\n\n入口1ファイル、既存E1テストファイルを拡張する。旧run実装へのimport依存なし。\n小さいCayley式・固定点式は同じ規約を保つ。軌道生成はNumbaの標準float64 tan、\nfastmathなし。E0と異なる実行エンジンなので同一seedの点ごとの一致は主張しない。\n\n主図（白背景、経験値青、理論黒、観測長は色に加え凡例で識別）:\n\n1. 各betaの複素平均64点をT別に等アスペクトで表示。円は理論RMS半径で信頼領域ではない。\n2. 観測長に対するseed平均二乗偏差・理論値、比とseed単位95%区間。\n3. seed=0のCayley点群と全64seedの64ビン角度密度。点群最大4096点を等間隔表示。\n4. lag相関のseed平均Re/Im・95%区間と理論。T=1000000、k=1,4。\n\nPNG/PDF/SVG、図の元CSV、複素点NPZ、seed初期値、理論ゲート、環境、コードと\n成果物ハッシュをこのフォルダに保存する。全生軌道は保存しないが、初期値・コード・\n環境から再生成でき、seed=0の各betaの全軌道を代表スナップショットとして保存する。\n\n実行: `python finite_time.py --theory` → `python finite_time.py --run` →\n`python finite_time.py --plot` → `python finite_time.py --validate`。\n既存の理論・実験データは上書きしない。図と集計は保存CSVから再生成できる。\n\n## 結果\n\n未実行。結果と解釈は事前条件を変更せず以下に追記する。\n'

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
    (root/'preregistration.md').write_bytes(PREREGISTRATION.encode('utf-8'))
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
         predictions_sha256=sha(root/'predictions.csv'), preregistration_sha256=sha(root/'preregistration.md'),
         preregistration_text=(root/'preregistration.md').read_text(encoding='utf-8')))
    if not passed:
        raise ArithmeticError('theory gate failed')
    print(json.dumps(dict(theory_passed=passed, checks=len(errors), max_error=max(errors))))


def run(root: Path) -> None:
    """独立seedの軌道と各観測長の指標を事前条件のまま記録する。"""
    gate = json.loads((root/'theory.json').read_text(encoding='utf-8'))
    if not gate['passed'] or gate['code_sha256'] != sha(Path(__file__)) or gate['config_sha256'] != sha(root/'config.json') or gate['predictions_sha256'] != sha(root/'predictions.csv') or gate['preregistration_sha256'] != sha(root/'preregistration.md'):
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
