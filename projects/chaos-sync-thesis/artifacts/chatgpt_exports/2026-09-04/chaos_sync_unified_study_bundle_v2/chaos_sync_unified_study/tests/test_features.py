"""What: 欠損を含む軌道でもTM/Fourier特徴が有限で再現可能かを検証する。"""

import numpy as np

from chaos_sync_unified.features import FourierBandExtractor, TMTemporalExtractor, graph_tm_late_features, make_graph_basis


def test_temporal_features_with_missing_values() -> None:
    rng = np.random.default_rng(1)
    values = rng.standard_cauchy((12, 128))
    values[:, 10:15] = np.nan
    tm = TMTemporalExtractor().transform(values)
    fourier = FourierBandExtractor(16).transform(values)
    assert tm.shape == (12, 16)
    assert fourier.shape == (12, 16)
    assert np.isfinite(tm).all()
    assert np.isfinite(fourier).all()


def test_graph_tm_feature_dimension() -> None:
    rng = np.random.default_rng(2)
    trajectories = rng.standard_normal((10, 40, 8))
    features = graph_tm_late_features(trajectories, make_graph_basis(8), late_window=20)
    assert features.shape == (10, 32)
    assert np.isfinite(features).all()
