"""Load and lightly validate the YAML configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"
)


@dataclass
class Config:
    """Thin wrapper around the parsed YAML giving attribute + dict access."""

    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # Convenience accessors used across the codebase --------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def models(self) -> dict[str, bool]:
        return self.raw["models"]

    @property
    def lstm(self) -> dict[str, Any]:
        return self.raw["lstm"]

    @property
    def evaluate(self) -> dict[str, Any]:
        return self.raw["evaluate"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def output(self) -> dict[str, Any]:
        return self.raw["output"]

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("random_seed", 42))


def load_config(path: str | None = None) -> Config:
    """Read ``config.yaml`` (or the given path) into a :class:`Config`."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    required = {"data", "features", "models", "evaluate", "backtest", "output"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"config is missing required sections: {sorted(missing)}")

    return Config(raw=raw)
