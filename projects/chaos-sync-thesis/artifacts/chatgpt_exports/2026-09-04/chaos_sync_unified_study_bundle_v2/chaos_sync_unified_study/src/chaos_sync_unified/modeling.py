from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """How: 全 sample・全出力座標を同じ重みで RMSE 集約する。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """How: test target の総分散で規格化した MSE を返す。"""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    denominator = float(np.mean((truth - np.mean(truth, axis=0, keepdims=True)) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return float(np.mean((truth - prediction) ** 2) / denominator)


@dataclass
class GroupedRidge:
    """How: group leakage を避けた CV で ridge 強度を選び、全 fit データへ再適合する。"""

    alphas: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    n_splits: int = 4
    model_: Pipeline | None = None
    selected_alpha_: float | None = None
    cv_rmse_: float | None = None

    def fit(self, x: FloatArray, y: FloatArray, groups: IntArray) -> "GroupedRidge":
        values = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        group_values = np.asarray(groups, dtype=np.int64)
        unique_groups = np.unique(group_values)
        n_splits = min(self.n_splits, len(unique_groups))
        if n_splits < 2:
            raise ValueError("grouped CV には2個以上の group が必要")
        splitter = GroupKFold(n_splits=n_splits)
        best_score = np.inf
        best_alpha = None
        for alpha in self.alphas:
            fold_scores = []
            for train_index, valid_index in splitter.split(values, targets, group_values):
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("ridge", Ridge(alpha=float(alpha))),
                    ]
                )
                model.fit(values[train_index], targets[train_index])
                fold_scores.append(rmse(targets[valid_index], model.predict(values[valid_index])))
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        assert best_alpha is not None
        self.selected_alpha_ = best_alpha
        self.cv_rmse_ = best_score
        self.model_ = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=best_alpha)),
            ]
        )
        self.model_.fit(values, targets)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        if self.model_ is None:
            raise RuntimeError("fit を先に呼ぶ必要がある")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=np.float64)), dtype=np.float64)


@dataclass
class GroupedLogistic:
    """How: group split で正則化を選ぶ二値ロジスティック probe。"""

    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    n_splits: int = 4
    model_: Pipeline | None = None
    selected_c_: float | None = None
    cv_balanced_accuracy_: float | None = None

    def fit(self, x: FloatArray, y: IntArray, groups: IntArray) -> "GroupedLogistic":
        values = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.int64)
        group_values = np.asarray(groups, dtype=np.int64)
        n_splits = min(self.n_splits, len(np.unique(group_values)))
        splitter = GroupKFold(n_splits=n_splits)
        best_score = -np.inf
        best_c = None
        for c_value in self.c_values:
            scores = []
            for train_index, valid_index in splitter.split(values, targets, group_values):
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "logistic",
                            LogisticRegression(C=float(c_value), max_iter=3000, solver="lbfgs"),
                        ),
                    ]
                )
                model.fit(values[train_index], targets[train_index])
                prediction = model.predict(values[valid_index])
                scores.append(balanced_accuracy_score(targets[valid_index], prediction))
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_c = float(c_value)
        assert best_c is not None
        self.selected_c_ = best_c
        self.cv_balanced_accuracy_ = best_score
        self.model_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs"),
                ),
            ]
        )
        self.model_.fit(values, targets)
        return self

    def predict(self, x: FloatArray) -> IntArray:
        if self.model_ is None:
            raise RuntimeError("fit を先に呼ぶ必要がある")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=np.float64)), dtype=np.int64)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        if self.model_ is None:
            raise RuntimeError("fit を先に呼ぶ必要がある")
        return np.asarray(self.model_.predict_proba(np.asarray(x, dtype=np.float64))[:, 1], dtype=np.float64)


def fit_ridge_for_dimensions(
    train_features: FloatArray,
    valid_features: FloatArray,
    test_features: FloatArray,
    y_train: FloatArray,
    y_valid: FloatArray,
    y_test: FloatArray,
    dimensions: Iterable[int],
    *,
    use_pca: bool,
) -> list[dict[str, float | int | FloatArray]]:
    """How: 各潜在次元で validation 選択済み ridge を学習し test 再構成を返す。"""
    records: list[dict[str, float | int | FloatArray]] = []
    max_available = train_features.shape[1]
    seen_effective_dimensions: set[int] = set()
    for dimension in dimensions:
        requested_dimension = int(dimension)
        d = min(requested_dimension, max_available, train_features.shape[0] - 1)
        if d <= 0 or d in seen_effective_dimensions:
            # Why not: feature 次元を超えた要求を同じ実効次元で重複評価すると、
            # 指標表に同一条件が複数行入り、比較時の行選択が曖昧になるため除外する。
            continue
        seen_effective_dimensions.add(d)
        if use_pca:
            reducer = PCA(n_components=d, random_state=0)
            x_train = reducer.fit_transform(train_features)
            x_valid = reducer.transform(valid_features)
            x_test = reducer.transform(test_features)
        else:
            reducer = None
            x_train = train_features[:, :d]
            x_valid = valid_features[:, :d]
            x_test = test_features[:, :d]
        best_alpha = None
        best_nmse = np.inf
        for alpha in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(x_train, y_train)
            score = nmse(y_valid, model.predict(x_valid))
            if score < best_nmse:
                best_nmse = score
                best_alpha = float(alpha)
        assert best_alpha is not None
        final_x = np.vstack([x_train, x_valid])
        final_y = np.vstack([y_train, y_valid])
        final_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
        final_model.fit(final_x, final_y)
        prediction = np.asarray(final_model.predict(x_test), dtype=np.float64)
        records.append(
            {
                "dimension": d,
                "alpha": best_alpha,
                "validation_nmse": best_nmse,
                "test_nmse": nmse(y_test, prediction),
                "test_rmse": rmse(y_test, prediction),
                "prediction": prediction,
            }
        )
    return records


def classification_scores(y_true: IntArray, y_pred: IntArray) -> dict[str, float]:
    """How: class imbalance に頑健な指標と通常 accuracy を併記する。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def paired_bootstrap_interval(
    first_errors: FloatArray,
    second_errors: FloatArray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260904,
) -> tuple[float, float, float]:
    """How: sample 対応を保った誤差差 first-second の bootstrap 区間を求める。"""
    a = np.asarray(first_errors, dtype=np.float64)
    b = np.asarray(second_errors, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired errors の形は一致する必要がある")
    difference = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    means = np.mean(difference[indices], axis=1)
    return float(np.mean(difference)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
