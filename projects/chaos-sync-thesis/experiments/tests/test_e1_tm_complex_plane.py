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


def test_finite_time_covariance_and_orbit(tmp_path: Path) -> None:
    """追試の有限長分散と実軌道が独立な定義に一致する。"""
    path = PATH.parent.parent / "20260907_E1_tangent-finite-time-confirmation/finite_time.py"
    spec = importlib.util.spec_from_file_location("finite_time", path)
    assert spec is not None and spec.loader is not None
    followup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(followup)
    for rho in [0., .3, .9999, 1.]:
        for n in [1, 2, 31]:
            covariance = rho ** abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            assert followup.mean_square(n, rho) == pytest.approx(covariance.mean())
    beta, x0 = 1.1, .03
    expected = []
    for _ in range(12):
        x0 = float(np.tan(beta*x0))
        expected.append(x0)
    np.testing.assert_allclose(followup.orbit(beta, .03, 12), expected, rtol=1e-13)
    g = followup.gamma_fixed(beta)
    assert g > 0 and abs(g-np.tanh(beta*g)) < 1e-12
    with pytest.raises(ValueError):
        followup.mean_square(0, .5)

    followup.theory(tmp_path)
    assert json.loads((tmp_path / "theory.json").read_text(encoding="utf-8"))["passed"]
    assert (tmp_path / "preregistration.md").exists()

def test_additive_sync_theory_and_variation(tmp_path: Path) -> None:
    """相互加算結合の尺度・横変分・理論ゲートを確認する。"""
    path = PATH.parent.parent / "20260907_E2_tangent-additive-sync/additive_sync.py"
    spec = importlib.util.spec_from_file_location("additive_sync", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for beta, epsilon in [(1.0001, 0.), (1.1, .5), (2., .9)]:
        p = module.predictions(beta, epsilon)
        assert p["transverse"] > p["lower_bound"] > 0
        assert abs((1-epsilon)*p["gamma"]-np.tanh(beta*p["gamma"])) < 1e-12
    with pytest.raises(ValueError):
        module.predictions(1.1, 1.)
    beta, epsilon, x0 = 1.1, .1, .02
    states, parallel, transverse = module.orbit(beta, epsilon, x0, 12)
    expected = []
    for i in range(12):
        derivative = beta/np.cos(beta*x0)**2
        assert parallel[i] == pytest.approx(np.log(derivative+epsilon))
        assert transverse[i] == pytest.approx(np.log(derivative-epsilon))
        x0 = np.tan(beta*x0)+epsilon*x0
        expected.append(x0)
    np.testing.assert_allclose(states, expected, atol=1e-12)
    module.theory(tmp_path)
    gate = json.loads((tmp_path / "theory.json").read_text(encoding="utf-8"))
    assert gate["passed"]
    gate["passed"] = False
    (tmp_path / "theory.json").write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError):
        module.run(tmp_path)
    assert not (tmp_path / "artifacts").exists()
