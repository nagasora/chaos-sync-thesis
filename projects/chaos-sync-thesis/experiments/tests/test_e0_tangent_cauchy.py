"""タンジェントE0の理論・型・保存結果に必要な計算契約を検証する。"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"runs/20260906_E0_tangent-cauchy/e0_tangent_cauchy.py"
spec = importlib.util.spec_from_file_location("e0_tangent_cauchy", PATH)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.mark.parametrize("beta", [1.0001, 1.01, 1.1, 2.])
def test_positive_root_and_derivative(beta: float) -> None:
    """正の根と微分が独立な高精度・有限差分評価に一致する。"""
    g = m.positive_scale(beta)
    assert 0 < g < 1
    assert abs(g-math.tanh(beta*g)) < 2e-14
    x, h = .123, 1e-6
    numerical = (math.tan(beta*(x+h))-math.tan(beta*(x-h)))/(2*h)
    analytic = beta*math.hypot(1, math.tan(beta*x))**2
    assert numerical == pytest.approx(analytic, rel=1e-9)


def test_branch_sum_integral_and_control() -> None:
    """密度和の打切り上界、独立積分、K-fold対照が成立する。"""
    assert all(r["passed"] for r in m.theory_checks())


def test_critical_increment_and_slow_relaxation() -> None:
    """桁落ちのない逆二乗差と臨界時の代数減衰を確認する。"""
    with mp.workdps(80):
        g = mp.mpf("0.0001")
        expected = float(1/mp.tanh(g)**2-1/g**2)
    assert m.critical_increment(float(g)) == pytest.approx(expected, abs=2e-16)
    values = m.scale_series(1., 1., 200000)
    assert values[-1]*math.sqrt(400000/3) == pytest.approx(1., rel=1e-4)
    assert 1/values[-1]**2-400000/3 > 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_state_dtype_and_failure_recording(dtype: str) -> None:
    """入力精度を保ち、非有限の標本を正常出力へ置換しない。"""
    x = np.array([.12, -.5, np.inf], dtype=dtype)
    beta = np.dtype(dtype).type(1.01)
    states = m.evolve(x, beta, 3)
    assert states.dtype == x.dtype
    expected = np.tan(np.multiply(x[:2], beta))
    np.testing.assert_allclose(states[1, :2], expected, rtol=2e-7 if dtype=="float32" else 1e-14)
    assert np.isnan(states[1:, 2]).all()
    assert m.distribution(states[-1], 1)["ks"] is None


def test_audit_reproduces_exact_float_input() -> None:
    """監査入力・時刻と誤差分解を保存できる。"""
    states = m.evolve(np.array([.1, 1.2]), 1.1, 3)
    candidates = m.empty_candidates()
    result = m.audit_update(candidates, states, 1.1, 7, 2)
    assert result["nonfinite"] == 0
    rows = m.finish_audit(candidates, dict(beta=1.1))
    assert len(rows) == 4
    assert all(7 <= r["step"] < 10 for r in rows)
    assert all(r["scaled_error"] < 1e-12 for r in rows)


def test_small_pipeline_saved_aggregate(tmp_path: Path) -> None:
    """短縮runは本実験と区別し、全段のCSV/NPZと再集計を確認する。"""
    cfg = m.configuration()
    cfg.update(beta=[.9, 1., 1.01, 1.1], epsilon=[.1, .01], scales=[1.],
               a_seeds=2, a_n=512, b_seeds=2, b_n=16, b_steps=8,
               long_beta=[1.], long_n=8, long_steps=16,
               c_seeds=3, c_steps=100, window=25, audit_k=2, fit_cutoffs=[.1])
    m.run_a(tmp_path, cfg)
    m.run_b(tmp_path, cfg)
    m.run_c(tmp_path, cfg)
    summary = m.summarize(tmp_path, cfg)
    assert summary["a_total"] == 8
    assert (tmp_path/"representative_orbits.npz").exists()
    rows = m.read_csv(tmp_path/"aggregate.csv")
    assert all(int(r["n"]) == 3 for r in rows)
    with np.load(tmp_path/"representative_orbits.npz") as archive:
        assert all(len(archive[k]) == 101 for k in archive.files)


def test_preregistration_rejects_replacement(tmp_path: Path) -> None:
    """既存の異なる事前設定を上書きしない。"""
    m.prepare(tmp_path)
    cfg = json.loads((tmp_path/"config.json").read_text())
    cfg["a_n"] = 1
    (tmp_path/"config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        m.prepare(tmp_path)


def test_validator_reports_incomplete_bundle_as_json(tmp_path: Path) -> None:
    """不足した保存物は不合格JSONになり、NumPy真偽値で保存が失敗しない。"""
    out = tmp_path/"artifacts"
    out.mkdir()
    m.write_json(out/"sha256.json", {})
    m.write_json(tmp_path/"config.json", dict(c_steps=4, b_steps=4, long_steps=4))
    m.write_json(tmp_path/"environment.json", dict(script_sha256=m.sha(PATH), reused_boole_sha256=m.sha(m.BOOLE)))
    for name in ["a", "b", "aggregate", "a_diagnostics", "b_diagnostics", "c_diagnostics", "a_audit", "b_audit", "c_audit"]:
        (out/(name+".csv")).write_text("dtype\n", encoding="utf-8")
    trajectory = np.array([.1, .2, .3, .4, .5])
    lam = np.mean(np.log(1.1)+2*np.log(np.hypot(1, trajectory[1:])))
    m.write_csv(out/"c.csv", [dict(beta=1.1, dtype="float64", seed=0, start=0, stop=4,
        lyapunov=float(lam), estimated_scale=.075)])
    np.savez(out/"representative_orbits.npz", **{"b1.1_float64": trajectory})
    result = m.validate(tmp_path)
    assert result["passed"] is False
    saved = json.loads((tmp_path/"validation.json").read_text())
    assert any(r["check"] == "orbit b1.1_float64" and r["passed"] for r in saved["checks"])
