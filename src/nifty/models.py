"""Model zoo behind a common ``fit`` / ``predict_proba`` interface.

Each builder returns an object exposing:
    * ``fit(X, y)``
    * ``predict_proba(X) -> np.ndarray`` of shape (n,) giving P(up)

so the evaluation and backtest code can treat every model identically.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


class SklearnProbModel:
    """Adapter giving any sklearn classifier a uniform P(up) interface."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class LSTMModel:
    """Keras LSTM over a sliding window of features (lazily imports tensorflow)."""

    def __init__(self, window: int, epochs: int, batch_size: int, units: int, seed: int):
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.seed = seed
        self.model = None
        self._mean = None
        self._std = None

    def _windowize(self, X: np.ndarray) -> np.ndarray:
        # Build (n - window + 1, window, n_features) sequences ending at each row.
        n, f = X.shape
        if n < self.window:
            return np.empty((0, self.window, f))
        idx = np.arange(self.window)[None, :] + np.arange(n - self.window + 1)[:, None]
        return X[idx]

    def _scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        return (X - self._mean) / self._std

    def fit(self, X, y):
        import tensorflow as tf

        tf.random.set_seed(self.seed)
        Xv = self._scale(np.asarray(X, dtype="float32"), fit=True)
        yv = np.asarray(y, dtype="float32")
        seqs = self._windowize(Xv)
        targets = yv[self.window - 1 :]  # label aligned to the last day of window

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.window, Xv.shape[1])),
                tf.keras.layers.LSTM(self.units),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(
            seqs, targets, epochs=self.epochs, batch_size=self.batch_size, verbose=0
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xv = self._scale(np.asarray(X, dtype="float32"), fit=False)
        seqs = self._windowize(Xv)
        preds = self.model.predict(seqs, verbose=0).ravel()
        # First (window-1) rows lack a full window; pad with the neutral 0.5.
        pad = np.full(self.window - 1, 0.5, dtype="float32")
        return np.concatenate([pad, preds])


def build_models(config: Config) -> dict[str, Callable[[], object]]:
    """Return ``{name: factory}`` for every model enabled in the config."""
    seed = config.random_seed
    toggles = config.models
    factories: dict[str, Callable[[], object]] = {}

    if toggles.get("logistic"):
        factories["logistic"] = lambda: SklearnProbModel(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            )
        )

    if toggles.get("random_forest"):
        factories["random_forest"] = lambda: SklearnProbModel(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                random_state=seed,
                n_jobs=-1,
            )
        )

    if toggles.get("xgboost"):
        def _xgb():
            from xgboost import XGBClassifier

            return SklearnProbModel(
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=-1,
                )
            )

        factories["xgboost"] = _xgb

    if toggles.get("lightgbm"):
        def _lgbm():
            from lightgbm import LGBMClassifier

            return SklearnProbModel(
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            )

        factories["lightgbm"] = _lgbm

    if toggles.get("lstm"):
        lcfg = config.lstm
        factories["lstm"] = lambda: LSTMModel(
            window=lcfg["window"],
            epochs=lcfg["epochs"],
            batch_size=lcfg["batch_size"],
            units=lcfg["units"],
            seed=seed,
        )

    if not factories:
        raise ValueError("No models enabled in config.yaml [models].")
    return factories
