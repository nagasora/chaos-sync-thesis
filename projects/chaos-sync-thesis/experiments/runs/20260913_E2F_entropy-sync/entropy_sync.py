"""E2F の有限分解能エントロピーと同期面横安定性を測定する。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numba
import numpy as np
from numba import njit, prange
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CONFIG_PATH = ROOT / "config.json"
METRIC_NAMES = (
    "hx_bits", "hy_bits", "hxy_bits", "hcond_bits", "mi_bits",
    "mismatch_probability", "fano_bound_bits", "sync_fraction",
    "median_logd", "invalid_count", "valid_count", "shuffle_mi_bits",
    "ever_entered_fraction", "reexit_fraction", "mean_exit_count",
)


def load_dynamics() -> Tuple[Any, Path]:
    """意味名で E2E 積分器を一意に解決して読み込む。"""
    matches = sorted(ROOT.parent.glob("*_E2E_critical-slowing/e2e_dynamics.py"))
    if len(matches) != 1:
        raise RuntimeError(f"E2E dynamics must resolve exactly once, found {len(matches)}")
    path = matches[0]
    # Numba cache は関数定義元のモジュール名を復元するため、元名で事前登録する。
    spec = importlib.util.spec_from_file_location("e2e_dynamics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E2E dynamics: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


DYNAMICS, DYNAMICS_PATH = load_dynamics()
STEP = DYNAMICS.step


def dynamics_ledger_path() -> str:
    """run 基準の明示的な相対パスを台帳用に返す。"""
    return (Path("..") / DYNAMICS_PATH.relative_to(ROOT.parent)).as_posix()


def read_config() -> Dict[str, Any]:
    """凍結済み JSON 設定を読む。"""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    """非標準 NaN を許さず JSON を保存する。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    """ファイルの SHA-256 を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """辞書行を同じ列順で CSV に保存する。"""
    if not rows:
        raise ValueError("CSV rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameters(beta: float) -> Tuple[float, float, float]:
    """正の Cauchy 尺度、同期面指数、下側臨界結合を返す。"""
    gamma = float(brentq(lambda z: math.tanh(beta * z) - z, 1e-15, 1.0))
    lambda0 = math.log(beta) + 2.0 * math.log1p(gamma)
    kappa_c = -math.expm1(-lambda0) / 2.0
    return gamma, lambda0, kappa_c


def kappa_grid(config: Dict[str, Any], kappa_c: float) -> np.ndarray:
    """事前登録した特殊値と臨界差から結合格子を組み立てる。"""
    offsets = config["kappa_offsets"]
    values = [
        config["kappa_special"][0] if offset is None and i == 0
        else config["kappa_special"][1] if offset is None
        else kappa_c + float(offset)
        for i, offset in enumerate(offsets)
    ]
    return np.asarray(values, dtype=np.float64)


@njit(parallel=True, cache=True)
def simulate_snapshots(
    beta: float,
    kappas: np.ndarray,
    s0: np.ndarray,
    logd0: np.ndarray,
    snapshots: np.ndarray,
    eta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """全結合・独立初期値を進め、指定時刻の ``s, log|d|`` を保存する。"""
    n_kappa, n_pairs, n_snapshots = kappas.size, s0.size, snapshots.size
    states = np.full((n_kappa, n_snapshots, n_pairs), np.nan)
    logds = np.full((n_kappa, n_snapshots, n_pairs), np.nan)
    invalid = np.ones((n_kappa, n_snapshots, n_pairs), dtype=np.bool_)
    entered_output = np.zeros((n_kappa, n_snapshots, n_pairs), dtype=np.bool_)
    exited_output = np.zeros((n_kappa, n_snapshots, n_pairs), dtype=np.bool_)
    exit_count_output = np.zeros((n_kappa, n_snapshots, n_pairs), dtype=np.int32)
    log_threshold = math.log(1e-5)
    for flat in prange(n_kappa * n_pairs):
        ki, pi = flat // n_pairs, flat % n_pairs
        s, logd = s0[pi], logd0[pi]
        valid = math.isfinite(s) and not math.isnan(logd) and logd != math.inf
        below = valid and logd < log_threshold
        entered = below
        exited = False
        exit_count = 0
        si = 0
        if snapshots[0] == 0 and valid:
            states[ki, 0, pi] = s
            logds[ki, 0, pi] = logd
            invalid[ki, 0, pi] = False
            entered_output[ki, 0, pi] = entered
            si = 1
        for t in range(1, snapshots[-1] + 1):
            if not valid:
                break
            s, logd, _, valid = STEP(s, logd, beta, kappas[ki], eta)
            if valid:
                new_below = logd < log_threshold
                if entered and below and not new_below:
                    exited = True
                    exit_count += 1
                if new_below:
                    entered = True
                below = new_below
            if si < n_snapshots and t == snapshots[si]:
                if valid:
                    states[ki, si, pi] = s
                    logds[ki, si, pi] = logd
                    invalid[ki, si, pi] = False
                entered_output[ki, si, pi] = entered
                exited_output[ki, si, pi] = exited
                exit_count_output[ki, si, pi] = exit_count
                si += 1
        while si < n_snapshots:
            entered_output[ki, si, pi] = entered
            exited_output[ki, si, pi] = exited
            exit_count_output[ki, si, pi] = exit_count
            si += 1
    return states, logds, invalid, entered_output, exited_output, exit_count_output


@njit(parallel=True, cache=True)
def diagonal_ftle(
    beta: float,
    s0: np.ndarray,
    steps: int,
    kappas: np.ndarray,
    block_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """独立対角軌道ごとの平均指数と非重複窓の横指数正値率を返す。"""
    means = np.empty(s0.size)
    positive = np.zeros((s0.size, kappas.size, block_lengths.size))
    for i in prange(s0.size):
        increments = np.empty(steps)
        s = s0[i]
        for t in range(steps):
            s = math.tan(beta * s)
            increments[t] = math.log(beta) + 2.0 * math.log(math.hypot(1.0, s))
        means[i] = increments.mean()
        for ki in range(kappas.size):
            transverse = abs(1.0 - 2.0 * kappas[ki])
            shift = -math.inf if transverse == 0.0 else math.log(transverse)
            for li in range(block_lengths.size):
                width = block_lengths[li]
                blocks = steps // width
                count = 0
                for block in range(blocks):
                    total = 0.0
                    for j in range(block * width, (block + 1) * width):
                        total += increments[j]
                    if total / width + shift > 0.0:
                        count += 1
                positive[i, ki, li] = count / blocks
    return means, positive


def entropy_from_counts(counts: np.ndarray) -> float:
    """度数から plugin エントロピーを bit 単位で計算する。"""
    total = float(counts.sum())
    if total <= 0.0:
        return math.nan
    probabilities = counts[counts > 0.0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def binary_entropy(probability: float) -> float:
    """ベルヌーイ変数の二値エントロピーを返す。"""
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(-probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability))


def metrics_from_counts(counts: np.ndarray) -> Dict[str, float]:
    """同時度数から連鎖律と Fano 上界に関わる量を計算する。"""
    hx = entropy_from_counts(counts.sum(axis=1))
    hy = entropy_from_counts(counts.sum(axis=0))
    hxy = entropy_from_counts(counts)
    hcond = hxy - hx
    total = float(counts.sum())
    mismatch = 1.0 - float(np.trace(counts)) / total
    bins = counts.shape[0]
    fano = binary_entropy(mismatch) + mismatch * math.log2(bins - 1)
    return {
        "hx_bits": hx,
        "hy_bits": hy,
        "hxy_bits": hxy,
        "hcond_bits": hcond,
        "mi_bits": hx + hy - hxy,
        "mismatch_probability": mismatch,
        "fano_bound_bits": fano,
    }


def binned_counts(
    states: np.ndarray,
    logds: np.ndarray,
    invalid: np.ndarray,
    gamma: float,
    bins: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """交換対称な同時度数と、ラベル乱択後に Y を置換した対照を返す。"""
    usable = ~invalid & np.isfinite(states) & ~np.isnan(logds) & (logds <= math.log(np.finfo(np.float64).max))
    s = states[usable]
    logd = logds[usable]
    distance = np.exp(logd)
    reconstructable = np.isfinite(distance)
    s, distance = s[reconstructable], distance[reconstructable]
    low, high = s - distance, s + distance

    def indexes(values: np.ndarray) -> np.ndarray:
        u = 0.5 + np.arctan(values / gamma) / math.pi
        return np.minimum((np.clip(u, 0.0, np.nextafter(1.0, 0.0)) * bins).astype(np.int64), bins - 1)

    low_index, high_index = indexes(low), indexes(high)
    raw = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(raw, (high_index, low_index), 1.0)
    symmetric = 0.5 * (raw + raw.T)

    signs = rng.integers(0, 2, low.size, dtype=np.int8).astype(bool)
    x_index = np.where(signs, high_index, low_index)
    y_index = np.where(signs, low_index, high_index)
    shuffled_y = y_index[rng.permutation(y_index.size)]
    control_raw = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(control_raw, (x_index, shuffled_y), 1.0)
    control = 0.5 * (control_raw + control_raw.T)
    return symmetric, control, int(invalid.size - low.size)


def evaluate_snapshot(
    suite: str,
    batch: int,
    kappa_index: int,
    kappa: float,
    snapshot_index: int,
    snapshot: int,
    eta: float,
    states: np.ndarray,
    logds: np.ndarray,
    invalid: np.ndarray,
    entered: np.ndarray,
    exited: np.ndarray,
    exit_count: np.ndarray,
    gamma: float,
    bins_values: List[int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """一つの ensemble snapshot の全量子化指標と度数を返す。"""
    rows: List[Dict[str, Any]] = []
    counts: Dict[int, np.ndarray] = {}
    controls: Dict[int, np.ndarray] = {}
    valid_logd = logds[~invalid & ~np.isnan(logds)]
    sync_fraction = float(np.mean(valid_logd < read_config()["sync_log_threshold"])) if valid_logd.size else math.nan
    median_logd = float(np.median(valid_logd)) if valid_logd.size else math.nan
    for bins in bins_values:
        # eta 間で同じラベル乱択と置換順を使い、数値切替以外の差を加えない。
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31, batch, kappa_index, snapshot_index, bins]))
        joint, control, reconstruction_invalid = binned_counts(states, logds, invalid, gamma, bins, rng)
        metric = metrics_from_counts(joint)
        control_metric = metrics_from_counts(control)
        row: Dict[str, Any] = {
            "suite": suite, "batch": batch, "eta": eta,
            "kappa_index": kappa_index, "kappa": kappa,
            "snapshot": snapshot, "bins": bins,
            **metric, "sync_fraction": sync_fraction, "median_logd": median_logd,
            "invalid_count": reconstruction_invalid,
            "valid_count": int(joint.sum()),
            "shuffle_mi_bits": control_metric["mi_bits"],
            "ever_entered_fraction": float(np.mean(entered)),
            "reexit_fraction": float(np.mean(exited)),
            "mean_exit_count": float(np.mean(exit_count)),
        }
        rows.append(row)
        counts[bins], controls[bins] = joint, control
    return rows, counts, controls


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同条件の独立バッチ平均とバッチ標準偏差を計算する。"""
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["suite"], row["eta"], row["kappa_index"], row["kappa"], row["snapshot"], row["bins"])
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        result: Dict[str, Any] = {
            "suite": key[0], "eta": key[1], "kappa_index": key[2],
            "kappa": key[3], "snapshot": key[4], "bins": key[5],
            "batch_count": len(group),
        }
        for name in METRIC_NAMES:
            values = np.asarray([float(row[name]) for row in group])
            result[f"{name}_mean"] = float(values.mean())
            result[f"{name}_batch_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(result)
    return output


def preflight() -> None:
    """既知度数、連鎖律、Fano 境界、再利用積分器の数値境界を確認する。"""
    config = read_config()
    ARTIFACTS.mkdir(exist_ok=True)
    bins = 8
    independent = np.full((bins, bins), 3.0)
    diagonal = np.eye(bins) * 24.0
    anti_diagonal = np.fliplr(np.eye(bins)) * 24.0
    checks: Dict[str, bool] = {}
    for name, counts, expected_cond in [
        ("independent", independent, math.log2(bins)),
        ("diagonal", diagonal, 0.0),
        ("anti_diagonal", anti_diagonal, 0.0),
    ]:
        metric = metrics_from_counts(counts)
        checks[f"{name}_conditional_entropy"] = abs(metric["hcond_bits"] - expected_cond) < 1e-12
        checks[f"{name}_chain_rule"] = abs(metric["hxy_bits"] - metric["hx_bits"] - metric["hcond_bits"]) < 1e-12
        checks[f"{name}_fano"] = metric["hcond_bits"] <= metric["fano_bound_bits"] + 1e-12
    _, synced_logd, _, valid = STEP(0.25, math.log(0.4), config["beta"], 0.5, config["linear_eta"])
    checks["one_step_kappa_half"] = valid and synced_logd == -math.inf
    underflow_logd = -1000.0
    _, retained_logd, linear, valid = STEP(0.25, underflow_logd, config["beta"], 0.3, config["linear_eta"])
    checks["logd_below_underflow_retained"] = valid and linear and math.isfinite(retained_logd) and retained_logd < -900.0
    gamma, lambda0, kappa_c = parameters(config["beta"])
    kappas = kappa_grid(config, kappa_c)
    checks["kappa_grid"] = bool(kappas.size == 9 and kappas[0] == 0.0 and kappas[-1] == 0.5)
    trial_s0 = np.asarray([0.1, -0.2, 0.3, -0.4])
    trial_logd0 = np.log(np.asarray([0.2, 0.3, 0.4, 0.5]))
    trial = simulate_snapshots(
        config["beta"], kappas[:1], trial_s0, trial_logd0,
        np.asarray([0, 1], dtype=np.int64), config["linear_eta"],
    )
    trial_rows, trial_counts, _ = evaluate_snapshot(
        "preflight", 0, 0, float(kappas[0]), 1, 1, config["linear_eta"],
        trial[0][0, 1], trial[1][0, 1], trial[2][0, 1],
        trial[3][0, 1], trial[4][0, 1], trial[5][0, 1],
        gamma, [8], config["seed"],
    )
    checks["simulate_evaluate_pipeline"] = bool(
        len(trial_rows) == 1 and trial_counts[8].sum() == 4
    )
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})
    dump_json(ARTIFACTS / "preflight.json", {
        "passed": True, "checks": checks, "gamma": gamma,
        "lambda0_theory": lambda0, "kappa_c": kappa_c,
        "frozen_hashes": {
            "config.json": sha256(CONFIG_PATH),
            "entropy_sync.py": sha256(Path(__file__)),
            dynamics_ledger_path(): sha256(DYNAMICS_PATH),
        },
    })
    print(json.dumps({"preflight": "passed", "checks": len(checks), "kappa_c": kappa_c}), flush=True)


def assert_frozen() -> Dict[str, Any]:
    """preflight 後に設定・コード・積分器が変化していないことを確認する。"""
    record = json.loads((ARTIFACTS / "preflight.json").read_text(encoding="utf-8"))
    current = {
        "config.json": sha256(CONFIG_PATH),
        "entropy_sync.py": sha256(Path(__file__)),
        dynamics_ledger_path(): sha256(DYNAMICS_PATH),
    }
    if not record["passed"] or current != record["frozen_hashes"]:
        raise RuntimeError("code, config, or E2E dependency changed after preflight")
    return read_config()


def save_plot(
    aggregate: List[Dict[str, Any]],
    kappas: np.ndarray,
    kappa_c: float,
    ftle_positive: np.ndarray,
    block_lengths: np.ndarray,
) -> None:
    """臨界近傍のエントロピー、小差分率、有限窓横指数を描画する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in aggregate if row["suite"] == "main" and row["bins"] == 16]
    near_indexes = range(1, 8)
    near_deltas = kappas[1:8] - kappa_c
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for snapshot in [1000, 7500, 30000]:
        rows = sorted([row for row in selected if row["snapshot"] == snapshot], key=lambda row: row["kappa_index"])
        by_index = {int(row["kappa_index"]): row for row in rows}
        axes[0, 0].plot(near_deltas, [by_index[i]["hcond_bits_mean"] for i in near_indexes], "o-", label=f"T={snapshot}")
        axes[0, 1].plot(near_deltas, [by_index[i]["sync_fraction_mean"] for i in near_indexes], "o-", label=f"T={snapshot}")
    axes[0, 0].set(xlabel="kappa - kappa_c", ylabel="plugin H(Y|X) [bit]", title="Fixed 16-bin conditional entropy")
    axes[0, 1].set(xlabel="kappa - kappa_c", ylabel="fraction log|d| < log(1e-5)", title="Finite-resolution small difference")
    for li, block in enumerate(block_lengths):
        axes[1, 0].plot(near_deltas, ftle_positive[:, 1:8, li].mean(axis=0), "o-", label=f"window={block}")
    axes[1, 0].set(xlabel="kappa - kappa_c", ylabel="positive-window fraction", title="Diagonal transverse FTLE")
    for ki, label in [(0, "kappa=0"), (7, "kappa_c+0.05"), (8, "kappa=0.5")]:
        rows = sorted(
            [row for row in aggregate if row["suite"] == "main" and row["snapshot"] == 30000 and row["kappa_index"] == ki],
            key=lambda row: row["bins"],
        )
        axes[1, 1].plot([row["bins"] for row in rows], [row["hcond_bits_mean"] for row in rows], "o-", label=label)
    axes[1, 1].set(xlabel="bins", ylabel="plugin H(Y|X) [bit]", title="Final endpoint resolution")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(ARTIFACTS / "entropy_sync_summary.png", dpi=180)
    plt.close(fig)


def run() -> None:
    """凍結条件で主計算、感度計算、独立対角 FTLE を実行して保存する。"""
    config = assert_frozen()
    if (ARTIFACTS / "main_snapshots.npz").exists():
        raise FileExistsError("main results already exist")
    started = time.perf_counter()
    beta = float(config["beta"])
    gamma, lambda0, kappa_c = parameters(beta)
    kappas = kappa_grid(config, kappa_c)
    snapshots = np.asarray(config["snapshots"], dtype=np.int64)
    bins_values = [int(value) for value in config["bins"]]
    main_shape = (config["main_batches"], kappas.size, snapshots.size, config["main_pairs_per_batch"])
    main_s = np.full(main_shape, np.nan)
    main_logd = np.full(main_shape, np.nan)
    main_invalid = np.ones(main_shape, dtype=np.bool_)
    main_entered = np.zeros(main_shape, dtype=np.bool_)
    main_exited = np.zeros(main_shape, dtype=np.bool_)
    main_exit_count = np.zeros(main_shape, dtype=np.int32)
    main_s0 = np.empty((config["main_batches"], config["main_pairs_per_batch"]))
    main_logd0 = np.empty_like(main_s0)
    batch_rows: List[Dict[str, Any]] = []
    main_counts = {bins: np.zeros(main_shape[:3] + (bins, bins)) for bins in bins_values}
    main_controls = {bins: np.zeros(main_shape[:3] + (bins, bins)) for bins in bins_values}
    for batch in range(config["main_batches"]):
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1, batch]))
        x, y = gamma * rng.standard_cauchy((2, config["main_pairs_per_batch"]))
        main_s0[batch], main_logd0[batch] = (x + y) / 2.0, np.log(np.abs(x - y) / 2.0)
        state, logd, invalid, entered, exited, exit_count = simulate_snapshots(
            beta, kappas, main_s0[batch], main_logd0[batch], snapshots, config["linear_eta"]
        )
        main_s[batch], main_logd[batch], main_invalid[batch] = state, logd, invalid
        main_entered[batch], main_exited[batch], main_exit_count[batch] = entered, exited, exit_count
        for ki, kappa in enumerate(kappas):
            for si, snapshot in enumerate(snapshots):
                rows, counts, controls = evaluate_snapshot(
                    "main", batch, ki, float(kappa), si, int(snapshot), config["linear_eta"],
                    state[ki, si], logd[ki, si], invalid[ki, si],
                    entered[ki, si], exited[ki, si], exit_count[ki, si],
                    gamma, bins_values, config["seed"],
                )
                batch_rows.extend(rows)
                for bins in bins_values:
                    main_counts[bins][batch, ki, si] = counts[bins]
                    main_controls[bins][batch, ki, si] = controls[bins]
        print(f"main batch {batch + 1}/{config['main_batches']} complete: {time.perf_counter() - started:.1f}s", flush=True)
    np.savez_compressed(
        ARTIFACTS / "main_snapshots.npz", s=main_s, logd=main_logd, invalid=main_invalid,
        entered=main_entered, exited=main_exited, exit_count=main_exit_count,
        s0=main_s0, logd0=main_logd0, kappas=kappas, snapshots=snapshots,
    )
    np.savez_compressed(
        ARTIFACTS / "main_counts.npz",
        **{f"counts_b{bins}": main_counts[bins] for bins in bins_values},
        **{f"shuffle_counts_b{bins}": main_controls[bins] for bins in bins_values},
    )

    sensitivity_kappas = np.asarray([kappa_c + offset for offset in config["sensitivity_offsets"]])
    sensitivity_etas = np.asarray(config["sensitivity_etas"], dtype=np.float64)
    sensitivity_shape = (
        sensitivity_etas.size, config["sensitivity_batches"], sensitivity_kappas.size,
        snapshots.size, config["sensitivity_pairs_per_batch"],
    )
    sensitivity_s = np.full(sensitivity_shape, np.nan)
    sensitivity_logd = np.full(sensitivity_shape, np.nan)
    sensitivity_invalid = np.ones(sensitivity_shape, dtype=np.bool_)
    sensitivity_entered = np.zeros(sensitivity_shape, dtype=np.bool_)
    sensitivity_exited = np.zeros(sensitivity_shape, dtype=np.bool_)
    sensitivity_exit_count = np.zeros(sensitivity_shape, dtype=np.int32)
    sensitivity_s0 = np.empty((config["sensitivity_batches"], config["sensitivity_pairs_per_batch"]))
    sensitivity_logd0 = np.empty_like(sensitivity_s0)
    sensitivity_counts = {bins: np.zeros(sensitivity_shape[:4] + (bins, bins)) for bins in bins_values}
    sensitivity_controls = {bins: np.zeros(sensitivity_shape[:4] + (bins, bins)) for bins in bins_values}
    for batch in range(config["sensitivity_batches"]):
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 2, batch]))
        x, y = gamma * rng.standard_cauchy((2, config["sensitivity_pairs_per_batch"]))
        sensitivity_s0[batch], sensitivity_logd0[batch] = (x + y) / 2.0, np.log(np.abs(x - y) / 2.0)
        for ei, eta in enumerate(sensitivity_etas):
            state, logd, invalid, entered, exited, exit_count = simulate_snapshots(
                beta, sensitivity_kappas, sensitivity_s0[batch], sensitivity_logd0[batch], snapshots, eta
            )
            sensitivity_s[ei, batch], sensitivity_logd[ei, batch], sensitivity_invalid[ei, batch] = state, logd, invalid
            sensitivity_entered[ei, batch] = entered
            sensitivity_exited[ei, batch] = exited
            sensitivity_exit_count[ei, batch] = exit_count
            for ki, kappa in enumerate(sensitivity_kappas):
                for si, snapshot in enumerate(snapshots):
                    rows, counts, controls = evaluate_snapshot(
                        "sensitivity", batch, ki, float(kappa), si, int(snapshot), float(eta),
                        state[ki, si], logd[ki, si], invalid[ki, si],
                        entered[ki, si], exited[ki, si], exit_count[ki, si],
                        gamma, bins_values, config["seed"],
                    )
                    batch_rows.extend(rows)
                    for bins in bins_values:
                        sensitivity_counts[bins][ei, batch, ki, si] = counts[bins]
                        sensitivity_controls[bins][ei, batch, ki, si] = controls[bins]
            print(f"sensitivity batch={batch + 1} eta={eta:g} complete: {time.perf_counter() - started:.1f}s", flush=True)
    np.savez_compressed(
        ARTIFACTS / "sensitivity_snapshots.npz", s=sensitivity_s, logd=sensitivity_logd,
        invalid=sensitivity_invalid, entered=sensitivity_entered, exited=sensitivity_exited,
        exit_count=sensitivity_exit_count, s0=sensitivity_s0, logd0=sensitivity_logd0,
        kappas=sensitivity_kappas, etas=sensitivity_etas, snapshots=snapshots,
    )
    np.savez_compressed(
        ARTIFACTS / "sensitivity_counts.npz",
        **{f"counts_b{bins}": sensitivity_counts[bins] for bins in bins_values},
        **{f"shuffle_counts_b{bins}": sensitivity_controls[bins] for bins in bins_values},
    )

    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 3]))
    ftle_s0 = gamma * rng.standard_cauchy(config["ftle_trajectories"])
    block_lengths = np.asarray(config["ftle_block_lengths"], dtype=np.int64)
    ftle_means, ftle_positive = diagonal_ftle(beta, ftle_s0, config["ftle_steps"], kappas, block_lengths)
    np.savez_compressed(
        ARTIFACTS / "ftle.npz", s0=ftle_s0, common_mean=ftle_means,
        transverse_positive_fraction=ftle_positive, kappas=kappas, block_lengths=block_lengths,
    )
    ftle_rows: List[Dict[str, Any]] = []
    for trajectory, mean in enumerate(ftle_means):
        row: Dict[str, Any] = {
            "trajectory": trajectory, "s0": ftle_s0[trajectory],
            "common_mean": mean, "common_theory": lambda0,
        }
        for ki in range(kappas.size):
            for li, block in enumerate(block_lengths):
                row[f"k{ki}_positive_fraction_w{block}"] = ftle_positive[trajectory, ki, li]
        ftle_rows.append(row)
    write_csv(ARTIFACTS / "ftle_per_trajectory.csv", ftle_rows)

    aggregate = aggregate_rows(batch_rows)
    write_csv(ARTIFACTS / "batch_metrics.csv", batch_rows)
    write_csv(ARTIFACTS / "aggregate_metrics.csv", aggregate)
    save_plot(aggregate, kappas, kappa_c, ftle_positive, block_lengths)
    main_final = [row for row in aggregate if row["suite"] == "main" and row["snapshot"] == 30000]
    gate_details = []
    for bins in bins_values:
        uncoupled = next(row for row in main_final if row["kappa_index"] == 0 and row["bins"] == bins)
        stable = next(row for row in main_final if row["kappa_index"] == 7 and row["bins"] == bins)
        ratio = stable["hcond_bits_mean"] / uncoupled["hcond_bits_mean"]
        gate_details.append({
            "bins": bins, "hcond_ratio": ratio,
            "hcond_pass": ratio < config["main_hcond_ratio_gate"],
            "sync_fraction": stable["sync_fraction_mean"],
            "sync_pass": stable["sync_fraction_mean"] >= config["main_sync_fraction_gate"],
        })
    ftle_error = float(ftle_means.mean() - lambda0)
    metrics = {
        "status": "executed", "beta": beta, "gamma": gamma,
        "lambda0_theory": lambda0, "kappa_c": kappa_c,
        "main_batches": config["main_batches"],
        "main_pairs_per_batch": config["main_pairs_per_batch"],
        "main_gate_details": gate_details,
        "main_gate_pass": all(row["hcond_pass"] and row["sync_pass"] for row in gate_details),
        "diagonal_mean": float(ftle_means.mean()), "diagonal_mean_error": ftle_error,
        "diagonal_gate_pass": abs(ftle_error) <= config["diagonal_mean_error_gate"],
        "ftle_window_note": "Non-overlapping windows within a trajectory are descriptive and are not treated as independent replicates.",
        "shuffle_note": "Finite-sample plugin MI baseline is reported separately and is not subtracted.",
        "state_note": "Reconstructed equal floating values at tiny logd indicate finite resolution, not exact synchronization.",
        "scope": "finite-time fixed-bin ensemble entropy and diagonal transverse FTLE; no full spectrum, KS entropy, or permanent synchronization claim",
        "runtime_seconds": time.perf_counter() - started,
    }
    dump_json(ARTIFACTS / "metrics.json", metrics)
    dump_json(ARTIFACTS / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": __import__("scipy").__version__,
        "numba": numba.__version__, "numba_threads": numba.get_num_threads(),
        "fastmath": False, "config_sha256": sha256(CONFIG_PATH),
        "e2e_dynamics_path": dynamics_ledger_path(),
        "e2e_dynamics_sha256": sha256(DYNAMICS_PATH),
    })
    manifest: Dict[str, str] = {
        "config.json": sha256(CONFIG_PATH), "entropy_sync.py": sha256(Path(__file__)),
        dynamics_ledger_path(): sha256(DYNAMICS_PATH),
    }
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_file() and path.name != "hash_manifest.json":
            manifest[f"artifacts/{path.name}"] = sha256(path)
    dump_json(ARTIFACTS / "hash_manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def replay_counts(
    config: Dict[str, Any], gamma: float, batch_rows: List[Dict[str, str]],
) -> int:
    """保存 snapshot から主・感度の全度数と指標を再計算する。"""
    checked = 0
    bins_values = [int(value) for value in config["bins"]]
    with np.load(ARTIFACTS / "main_snapshots.npz") as snapshots, np.load(ARTIFACTS / "main_counts.npz") as recorded:
        for batch in range(config["main_batches"]):
            for ki, kappa in enumerate(snapshots["kappas"]):
                for si, snapshot in enumerate(snapshots["snapshots"]):
                    rows, counts, controls = evaluate_snapshot(
                        "main", batch, ki, float(kappa), si, int(snapshot), config["linear_eta"],
                        snapshots["s"][batch, ki, si], snapshots["logd"][batch, ki, si],
                        snapshots["invalid"][batch, ki, si], snapshots["entered"][batch, ki, si],
                        snapshots["exited"][batch, ki, si], snapshots["exit_count"][batch, ki, si],
                        gamma, bins_values, config["seed"],
                    )
                    for row in rows:
                        bins = row["bins"]
                        np.testing.assert_array_equal(counts[bins], recorded[f"counts_b{bins}"][batch, ki, si])
                        np.testing.assert_array_equal(controls[bins], recorded[f"shuffle_counts_b{bins}"][batch, ki, si])
                        saved = next(item for item in batch_rows if item["suite"] == "main" and int(item["batch"]) == batch and int(item["kappa_index"]) == ki and int(item["snapshot"]) == int(snapshot) and int(item["bins"]) == bins)
                        for name in METRIC_NAMES:
                            np.testing.assert_allclose(float(saved[name]), float(row[name]), rtol=0.0, atol=1e-14)
                        checked += 1
    with np.load(ARTIFACTS / "sensitivity_snapshots.npz") as snapshots, np.load(ARTIFACTS / "sensitivity_counts.npz") as recorded:
        for ei, eta in enumerate(snapshots["etas"]):
            for batch in range(config["sensitivity_batches"]):
                for ki, kappa in enumerate(snapshots["kappas"]):
                    for si, snapshot in enumerate(snapshots["snapshots"]):
                        rows, counts, controls = evaluate_snapshot(
                            "sensitivity", batch, ki, float(kappa), si, int(snapshot), float(eta),
                            snapshots["s"][ei, batch, ki, si], snapshots["logd"][ei, batch, ki, si],
                            snapshots["invalid"][ei, batch, ki, si], snapshots["entered"][ei, batch, ki, si],
                            snapshots["exited"][ei, batch, ki, si], snapshots["exit_count"][ei, batch, ki, si],
                            gamma, bins_values, config["seed"],
                        )
                        for row in rows:
                            bins = row["bins"]
                            np.testing.assert_array_equal(counts[bins], recorded[f"counts_b{bins}"][ei, batch, ki, si])
                            np.testing.assert_array_equal(controls[bins], recorded[f"shuffle_counts_b{bins}"][ei, batch, ki, si])
                            saved = next(
                                item for item in batch_rows
                                if item["suite"] == "sensitivity"
                                and int(item["batch"]) == batch
                                and float(item["eta"]) == float(eta)
                                and int(item["kappa_index"]) == ki
                                and int(item["snapshot"]) == int(snapshot)
                                and int(item["bins"]) == bins
                            )
                            for name in METRIC_NAMES:
                                np.testing.assert_allclose(float(saved[name]), float(row[name]), rtol=0.0, atol=1e-14)
                            checked += 1
    return checked


def validate() -> None:
    """ハッシュ、保存再集計、代表軌道再生、集計転記を照合する。"""
    config = assert_frozen()
    manifest = json.loads((ARTIFACTS / "hash_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise AssertionError(f"hash mismatch: {relative}")
    gamma, _, _ = parameters(config["beta"])
    with (ARTIFACTS / "batch_metrics.csv").open(encoding="utf-8") as stream:
        batch_rows = list(csv.DictReader(stream))
    count_checks = replay_counts(config, gamma, batch_rows)
    with (ARTIFACTS / "aggregate_metrics.csv").open(encoding="utf-8") as stream:
        aggregate_saved = list(csv.DictReader(stream))
    aggregate_fresh = aggregate_rows([{key: (float(value) if key not in {"suite"} else value) for key, value in row.items()} for row in batch_rows])
    if len(aggregate_saved) != len(aggregate_fresh):
        raise AssertionError("aggregate row count mismatch")
    for saved, fresh in zip(aggregate_saved, aggregate_fresh):
        for key, value in fresh.items():
            if key == "suite":
                assert saved[key] == value
            else:
                np.testing.assert_allclose(float(saved[key]), float(value), rtol=0.0, atol=1e-12)
    with np.load(ARTIFACTS / "main_snapshots.npz") as saved:
        replay = simulate_snapshots(
            config["beta"], saved["kappas"], saved["s0"][0, :2], saved["logd0"][0, :2],
            saved["snapshots"], config["linear_eta"],
        )
        for fresh, name in zip(replay, ["s", "logd", "invalid", "entered", "exited", "exit_count"]):
            np.testing.assert_array_equal(fresh, saved[name][0, :, :, :2])
    sensitivity_replays = 0
    with np.load(ARTIFACTS / "sensitivity_snapshots.npz") as saved:
        for ei, eta in enumerate(saved["etas"]):
            replay = simulate_snapshots(
                config["beta"], saved["kappas"], saved["s0"][0, :2], saved["logd0"][0, :2],
                saved["snapshots"], float(eta),
            )
            for fresh, name in zip(replay, ["s", "logd", "invalid", "entered", "exited", "exit_count"]):
                np.testing.assert_array_equal(fresh, saved[name][ei, 0, :, :, :2])
            sensitivity_replays += 2
    dump_json(ROOT / "validation.json", {
        "passed": True, "hashes_checked": len(manifest),
        "count_metric_checks": count_checks,
        "aggregate_rows_checked": len(aggregate_fresh),
        "main_dynamics_replayed_pairs": 2,
        "sensitivity_dynamics_replayed_pairs": sensitivity_replays,
        "meaning": "integrity and deterministic replay; not proof of asymptotic synchronization or estimator unbiasedness",
    })
    print(json.dumps({"validation": "passed", "count_metric_checks": count_checks}), flush=True)


def main() -> None:
    """CLI から preflight、run、validate の一つを実行する。"""
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
    elif arguments.run:
        run()
    else:
        validate()


if __name__ == "__main__":
    main()
