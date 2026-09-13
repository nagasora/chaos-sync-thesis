"""TM複素平面解析の理論ゲート・入力保護・保存計算を検証する。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"runs/20260906_E1_tangent-tm-complex-plane/tm_complex_plane.py"
spec=importlib.util.spec_from_file_location("tm_complex_plane",PATH)
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


@pytest.mark.parametrize("scale",[.001,1.,100.])
def test_cayley_matches_rational_map(scale: float) -> None:
    """角度による安定実装が規約の複素有理式と一致する。"""
    x=np.array([-100.,-1.,0.,1.,100.])*scale
    z=m.cayley(x,scale)
    np.testing.assert_allclose(z,(x-1j*scale)/(x+1j*scale),atol=1e-14)
    assert np.max(abs(abs(z)-1))<1e-14


def test_invalid_input_is_not_removed() -> None:
    """不正な状態・尺度を削除や既定値で隠さない。"""
    with pytest.raises(ValueError): m.cayley(np.array([0.,np.inf]),1.)
    with pytest.raises(ValueError): m.cayley(np.array([0.]),0.)


def fixture_run(root: Path) -> dict:
    """本実験と区別した少数点fixtureを構成する。"""
    (root/"artifacts").mkdir()
    cfg=m.configuration()
    cfg.update(betas=[1.1],fp32_betas=[],representatives=[1.1],n=512,window=128,
               lags=[1,2,4,8,16,64],phase_lags=[1,16,64],quadrature_n=[512,1024],scatter_n=64,time_n=64)
    m.dump(root/"config.json",cfg)
    x=np.random.default_rng(7).standard_cauchy(513)*m.scale_fixed(1.1)
    # 本物と同じキーを使うが、軌道でなく独立標本なので経験的相関の合格は要求しない。
    np.savez(root/"artifacts/source_orbits.npz",**{"b1.1_float64":x})
    m.dump(root/"provenance.json",dict(source_sha256=m.digest(root/"artifacts/source_orbits.npz")))
    return cfg


def test_theory_gate_and_saved_pipeline(tmp_path: Path) -> None:
    """理論確認後に集計・図・入力時刻を検証できる。"""
    fixture_run(tmp_path)
    result=m.theory(tmp_path)
    assert result["passed"]
    result=m.analyze(tmp_path)
    assert result["orbits"]==1
    m.draw(tmp_path)
    checked=m.validate(tmp_path)
    assert checked["passed"]
    assert len(list((tmp_path/"artifacts").glob("*.pdf")))==7
    assert len(list((tmp_path/"artifacts").glob("*.svg")))==7
    with pytest.raises(FileExistsError): m.analyze(tmp_path)


def test_failed_theory_blocks_analysis(tmp_path: Path) -> None:
    """理論ゲートの失敗を無視して解析しない。"""
    fixture_run(tmp_path)
    m.theory(tmp_path)
    gate=m.load(tmp_path/"theory.json"); gate["passed"]=False
    m.dump(tmp_path/"theory.json",gate)
    with pytest.raises(ValueError): m.analyze(tmp_path)
    assert not (tmp_path/"artifacts/moments.csv").exists()


def test_changed_input_blocks_analysis(tmp_path: Path) -> None:
    """異なる入力を同じ実験の結果に混入させない。"""
    fixture_run(tmp_path)
    m.theory(tmp_path)
    with (tmp_path/"artifacts/source_orbits.npz").open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError): m.analyze(tmp_path)


def test_shuffle_conditional_mean() -> None:
    """有限集合の順列対照は0でなく全異なる対の平均を理論値とする。"""
    z=np.exp(1j*np.array([.1,.3,.9,1.5]))
    expected=(abs(z.sum())**2-len(z))/(len(z)*(len(z)-1))
    direct=np.mean([a*np.conj(b) for i,a in enumerate(z) for j,b in enumerate(z) if i!=j])
    assert abs(expected-direct)<1e-14


import math

E2E_PATH = PATH.parent.parent / "20260913_E2E_critical-slowing/e2e_dynamics.py"
E2E_SPEC = importlib.util.spec_from_file_location("e2e_dynamics", E2E_PATH)
assert E2E_SPEC is not None and E2E_SPEC.loader is not None
e2e = importlib.util.module_from_spec(E2E_SPEC)
sys.modules[E2E_SPEC.name] = e2e
E2E_SPEC.loader.exec_module(e2e)


def _mp_e2e_step(
    s: float,
    logd: float,
    beta: float,
    kappa: float,
) -> tuple[float, float]:
    """binary64入力から独立な80桁 tan 更新を計算する。"""
    import mpmath as mp

    with mp.workdps(80):
        ms = mp.mpf(s)
        mbeta = mp.mpf(beta)
        mkappa = mp.mpf(kappa)
        d = mp.mpf("0") if logd == -math.inf else mp.exp(mp.mpf(logd))
        plus = mp.tan(mbeta * (ms + d))
        minus = mp.tan(mbeta * (ms - d))
        s_new = (plus + minus) / 2
        d_new = abs(1 - 2 * mkappa) * abs(plus - minus) / 2
        return float(s_new), -math.inf if d_new == 0 else float(mp.log(d_new))


@pytest.mark.parametrize(
    ("gap_scale", "expected_linear"),
    [(0.5e-6, True), (2.0e-6, False)],
)
def test_e2e_one_step_branch_crossing_matches_mpmath(
    gap_scale: float,
    expected_linear: bool,
) -> None:
    """線形化境界の両側で一歩更新が80桁の有限差分更新と一致する。"""
    beta, kappa, s = 1.01, 0.37, 0.23
    d = gap_scale * abs(math.cos(beta * s)) / beta
    logd = math.log(d)
    expected_s, expected_logd = _mp_e2e_step(s, logd, beta, kappa)
    actual_s, actual_logd, used_linear, valid = e2e.step(
        s, logd, beta, kappa, 1e-6
    )
    assert valid
    assert used_linear is expected_linear
    assert actual_s == pytest.approx(expected_s, rel=1e-9, abs=1e-9)
    assert actual_logd == pytest.approx(expected_logd, rel=1e-9, abs=1e-9)


def test_e2e_near_pole_product_avoids_cosine_sum_cancellation() -> None:
    """極近傍の有限結果を、相殺する旧分母より高精度に評価する。"""
    s = math.pi / 2.0 - 5e-7
    logd = math.log(4e-7)
    expected_s, _ = _mp_e2e_step(s, logd, 1.0, 0.31)
    actual_s, _, used_linear, valid = e2e.step(s, logd, 1.0, 0.31, 1e-6)
    d = math.exp(logd)
    legacy_s = math.sin(2.0 * s) / (
        math.cos(2.0 * s) + math.cos(2.0 * d)
    )
    assert valid and not used_linear and math.isfinite(actual_s)
    assert abs((actual_s - expected_s) / expected_s) < 2e-9
    assert abs((legacy_s - expected_s) / expected_s) > 1e-5


def test_e2e_small_gap_linearization_is_nonuniform_near_pole() -> None:
    """絶対差だけ小さい場合は、極近傍で線形近似へ入らない。"""
    s = math.pi / 2.0 - 1e-4
    logd = math.log(9e-7)
    expected_s, expected_logd = _mp_e2e_step(s, logd, 1.0, 0.2)
    actual_s, actual_logd, used_linear, valid = e2e.step(
        s, logd, 1.0, 0.2, 1e-6
    )
    naive_linear_s = math.tan(s)
    assert valid and not used_linear
    assert actual_s == pytest.approx(expected_s, rel=1e-9)
    assert actual_logd == pytest.approx(expected_logd, rel=1e-9)
    assert abs(naive_linear_s - expected_s) > 0.1


def test_e2e_exact_zero_coupling_and_unresolvable_gap() -> None:
    """κ=1/2とd=0は真の零差分を保ち、巨大差分は無効化する。"""
    s_new, logd_new, used_linear, valid = e2e.step(
        0.2, math.log(0.3), 1.01, 0.5, 1e-6
    )
    assert valid and not used_linear and math.isfinite(s_new)
    assert logd_new == -math.inf
    _, second_logd, second_linear, second_valid = e2e.step(
        s_new, logd_new, 1.01, 0.5, 1e-6
    )
    assert second_valid and second_linear and second_logd == -math.inf
    _, _, _, overflow_valid = e2e.step(0.2, 1_000.0, 1.01, 0.2, 1e-6)
    assert not overflow_valid


def test_e2e_measure_preserves_finite_window_history_and_invalidity() -> None:
    """checkpointごとに再離脱・窓長を残し、無効軌道を明示する。"""
    beta = 2.0
    target = (math.pi / 2.0 - 1e-8) / beta
    s_before_pole = math.atan(target) / beta
    checkpoints = np.array([1, 2, 3], dtype=np.int64)
    result = e2e.measure(
        beta,
        0.45,
        np.array([s_before_pole, 1e308]),
        np.array([math.log(1e-12), math.log(1e-4)]),
        3,
        1e-6,
        1e-8,
        1e-5,
        1,
        2,
        checkpoints,
    )
    assert result.shape == (2, 3, 8)
    np.testing.assert_array_equal(result[0, :, 0], [1, 1, 1])
    assert result[0, 1, 1] == 2
    assert result[0, 1, 3] == 1
    assert result[0, 0, 2] == 1
    assert result[0, 1, 2] == 0
    np.testing.assert_array_equal(result[1, :, 5], 1)
    np.testing.assert_array_equal(result[1, :, 7], 1)
    np.testing.assert_array_equal(result[1, :, 2], -1)
    assert np.isnan(result[1, :, 4]).all()