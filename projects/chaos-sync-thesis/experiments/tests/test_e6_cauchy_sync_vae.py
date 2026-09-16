"""VAEの確率契約・KL・同期・score勾配を独立計算と照合する。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import cauchy

PATH = Path(__file__).resolve().parents[1] / "runs/20260907_E6_cauchy-sync-vae/run_experiment.py"
SPEC = importlib.util.spec_from_file_location("e6", PATH)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.mark.parametrize("loc,scale", [(0., 1.), (0.7, 0.2), (-1.3, 1.8)])
def test_kl_matches_independent_quadrature(loc: float, scale: float) -> None:
    """KLが密度比の独立積分に一致し、自己KLは0になる。"""
    actual = m.cauchy_kl(torch.tensor([loc], dtype=torch.float64), torch.tensor([scale], dtype=torch.float64))
    expected = quad(lambda z: cauchy.pdf(z, loc, scale) *
                    (cauchy.logpdf(z, loc, scale) - cauchy.logpdf(z)), -np.inf, np.inf)[0]
    assert actual.item() == pytest.approx(expected, abs=1e-9)


def test_sync_and_primary_path_failure() -> None:
    """一歩同期、不変対角、非結合、非有限停止を直接式と照合する。"""
    z = torch.tensor([[.12, .8, -.2, .4]], dtype=torch.float64)
    expected = torch.tan(1.5 * z).reshape(1, 2, 2).mean(-1)
    out = m.evolve(z, 1, .5).reshape(1, 2, 2)
    torch.testing.assert_close(out[..., 0], expected)
    assert torch.equal(out[..., 0], out[..., 1])
    follow = m.evolve(out.reshape(1, 4), 1, .5)
    torch.testing.assert_close(follow, torch.tan(1.5 * out.reshape(1, 4)))
    torch.testing.assert_close(m.evolve(z, 1, 0), torch.tan(1.5 * z))
    with pytest.raises(FloatingPointError):
        m.evolve(torch.tensor([[float("inf"), 0.]]), 1, .5)


def test_score_gradient_matches_smooth_expectation() -> None:
    """Cauchy位置のscore期待値を、解析的に微分可能な有界観測と照合する。"""
    # E cos(Z)=exp(-scale)*cos(loc)の微分が基準。乱数誤差を避け分位点求積する。
    loc = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(.7, dtype=torch.float64)
    u = (torch.arange(200000, dtype=torch.float64) + .5) / 200000
    z = (loc.detach() + scale * torch.tan(torch.pi * (u - .5))).detach()
    q = torch.distributions.Cauchy(loc, scale)
    (z.cos() * q.log_prob(z)).mean().backward()
    assert loc.grad.item() == pytest.approx(-np.exp(-.7) * np.sin(.4), abs=2e-4)


@pytest.mark.parametrize("family,steps,coupling", [("normal", 0, 0.), ("cauchy", 0, 0.), ("cauchy", 4, .5)])
def test_learning_and_prior_generation(family: str, steps: int, coupling: float) -> None:
    """encoder/decoder双方へ有限勾配が届き、同じdecodeで事前生成できる。"""
    torch.manual_seed(17)
    model = m.VAE(family, steps, coupling).double()
    x = torch.randint(0, 2, (6, 64)).double()
    loss = model.objective(x, 4)
    loss.backward()
    for block in (model.encoder, model.decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert sum(p.grad.abs().sum().item() for p in block.parameters()) > 0
    logits, _ = model.decode(torch.randn(9, 8, dtype=torch.float64))
    assert logits.shape == (9, 64) and (logits.abs() <= 8).all()
    scores = m.evaluate(model, x, 4, 2)
    np.testing.assert_allclose(scores["negative_elbo"], scores["nll"] + scores["kl"])
