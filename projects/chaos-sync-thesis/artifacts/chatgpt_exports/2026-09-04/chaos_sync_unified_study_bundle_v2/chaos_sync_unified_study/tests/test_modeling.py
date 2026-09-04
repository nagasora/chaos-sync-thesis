from __future__ import annotations

import numpy as np

from chaos_sync_unified.modeling import fit_ridge_for_dimensions


def test_capped_latent_dimensions_are_not_duplicated() -> None:
    """What: 特徴次元を超える複数要求が同じ実効次元の重複行を作らない。"""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(30, 3))
    valid = rng.normal(size=(10, 3))
    test = rng.normal(size=(10, 3))
    y_train = rng.normal(size=(30, 2))
    y_valid = rng.normal(size=(10, 2))
    y_test = rng.normal(size=(10, 2))

    records = fit_ridge_for_dimensions(
        train, valid, test, y_train, y_valid, y_test, [2, 3, 4, 8], use_pca=True
    )

    assert [record["dimension"] for record in records] == [2, 3]
