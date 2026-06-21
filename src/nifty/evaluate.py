"""Walk-forward evaluation and model comparison.

Uses :class:`sklearn.model_selection.TimeSeriesSplit` so every validation fold is
strictly *after* its training data — no shuffling, no peeking into the future.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from .config import Config


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, p)


def evaluate_models(
    factories: dict[str, Callable[[], object]],
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
) -> pd.DataFrame:
    """Walk-forward CV for each model; return a metrics comparison table."""
    n_splits = config.evaluate["n_splits"]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    Xv, yv = X.values, y.values

    rows = []
    for name, factory in factories.items():
        accs, aucs, precs, recs, f1s = [], [], [], [], []
        for train_idx, test_idx in splitter.split(Xv):
            model = factory()
            model.fit(Xv[train_idx], yv[train_idx])
            proba = np.asarray(model.predict_proba(Xv[test_idx]))
            pred = (proba > 0.5).astype(int)
            y_true = yv[test_idx]

            accs.append(accuracy_score(y_true, pred))
            aucs.append(_safe_auc(y_true, proba))
            precs.append(precision_score(y_true, pred, zero_division=0))
            recs.append(recall_score(y_true, pred, zero_division=0))
            f1s.append(f1_score(y_true, pred, zero_division=0))

        rows.append(
            {
                "model": name,
                "accuracy": np.mean(accs),
                "roc_auc": np.nanmean(aucs),
                "precision": np.mean(precs),
                "recall": np.mean(recs),
                "f1": np.mean(f1s),
                "accuracy_std": np.std(accs),
            }
        )
        print(
            f"  {name:>14s} | acc={np.mean(accs):.4f} "
            f"auc={np.nanmean(aucs):.4f} f1={np.mean(f1s):.4f}"
        )

    table = pd.DataFrame(rows).set_index("model")
    return table


def select_best(table: pd.DataFrame, config: Config) -> str:
    """Pick the best model name by the configured selection metric."""
    metric = config.evaluate.get("select_metric", "roc_auc")
    if metric not in table.columns:
        metric = "accuracy"
    return table[metric].astype(float).idxmax()
