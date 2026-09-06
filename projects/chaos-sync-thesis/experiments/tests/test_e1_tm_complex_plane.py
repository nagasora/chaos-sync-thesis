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
