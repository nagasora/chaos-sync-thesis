"""What: 局所写像・同期多様体・特徴次元の不変条件を検証する。"""

import numpy as np

from chaos_sync_unified.core import BooleMap, OutputMixingNetwork, TangentMap
from chaos_sync_unified.features import FourierBandExtractor, TMTemporalExtractor, make_graph_basis


def test_boole_invariant_scale_formula() -> None:
    model = BooleMap(0.5)
    assert np.isclose(model.invariant_scale, 1.0)
    assert np.isclose(model.theoretical_lyapunov, np.log(2.0))


def test_output_mixing_preserves_synchrony() -> None:
    model = OutputMixingNetwork(BooleMap(0.5), 0.73)
    states = np.full((4, 8), 0.37)
    next_state = model.step(states)
    assert np.max(np.ptp(next_state, axis=1)) < 1e-14


def test_tangent_period_reduction() -> None:
    model = TangentMap(1.2)
    x = np.array([0.1, 0.5, -1.2])
    shifted = x + np.pi / 1.2
    assert np.allclose(model(x), model(shifted), rtol=1e-12, atol=1e-12)


def test_matched_readout_dimension() -> None:
    assert TMTemporalExtractor().output_dim == FourierBandExtractor(16).output_dim == 16


def test_graph_basis_is_orthonormal() -> None:
    basis = make_graph_basis(8)
    assert np.allclose(basis.T @ basis, np.eye(8), atol=1e-12)
