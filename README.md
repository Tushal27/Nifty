# Nifty 50 — Next-Day Direction Prediction & Trade Analysis

A from-scratch Python pipeline that loads Nifty 50 history, engineers technical
indicators, **trains and compares several models** to predict the **next day's
direction (up/down)**, and runs a **walk-forward backtest** that produces trade
signals, performance metrics, and charts.

> ⚠️ **Honest disclaimer.** Daily price direction is *near-random*. This project
> is an educational, methodologically-careful framework — proper time-series
> validation, no look-ahead bias, and an honest benchmark against buy-and-hold.
> It is **not** financial advice and makes **no** promise of profit. Treat
> walk-forward accuracy near ~50% as the expected, honest result; anything far
> above that usually signals data leakage, not a money machine.

## What's inside

| Stage | Module | What it does |
|-------|--------|--------------|
| Data | `src/nifty/data_loader.py` | Fetch via `yfinance` (`^NSEI`) **or** read your own CSV; clean + cache |
| Features | `src/nifty/features.py` | Returns, SMA/EMA ratios, RSI, MACD, Bollinger %, ATR, volatility, momentum, volume — all trailing-only |
| Models | `src/nifty/models.py` | Logistic Regression, Random Forest, XGBoost, LightGBM, and an optional Keras LSTM, behind one interface |
| Evaluate | `src/nifty/evaluate.py` | `TimeSeriesSplit` walk-forward CV; accuracy / AUC / F1 comparison |
| Backtest | `src/nifty/backtest.py` | Out-of-sample equity curve, Sharpe, max drawdown, win rate, costs, live signal |
| Plots | `src/nifty/plots.py` | Equity curve, signals-on-price, feature importance, model comparison |

## Setup

```bash
pip install -r requirements.txt
# Optional, only if you enable the LSTM model in config.yaml:
# pip install tensorflow
```

## Data

By default the pipeline pulls Nifty 50 (`^NSEI`) from Yahoo Finance.

> **Note on "50 years":** free APIs typically only reach back to ~2007 for
> `^NSEI`, not a full 50 years. To use your own long-history file, set
> `data.csv_path` in `config.yaml` to a CSV with columns
> `Date, Open, High, Low, Close, Volume` (case-insensitive). The CSV path takes
> precedence over the API.

```bash
python scripts/fetch_data.py            # download + cache to data/nifty.parquet
python scripts/fetch_data.py --refresh  # ignore cache and re-download
```

## Run the full pipeline

```bash
python scripts/run_pipeline.py
# overrides:
python scripts/run_pipeline.py --refresh --threshold 0.55 --ticker ^NSEI
```

### Outputs (written to `outputs/`)

- `metrics.csv` — per-model walk-forward comparison table
- `backtest_metrics.json` — strategy vs buy-and-hold performance
- `latest_signal.json` — the most recent up/down call with its probability
- `charts/` — `equity_curve.png`, `signals_on_price.png`,
  `feature_importance.png`, `model_comparison.png`

## Configuration

All knobs live in `config.yaml`: ticker/date range or CSV path, indicator
parameters, which models to enable, walk-forward fold count, and backtest
settings (probability threshold, long/short, transaction cost, annualisation).

## Tests

```bash
pytest tests/
```

The tests assert RSI bounds, that the label matches the realised next-day move,
and — crucially — that altering a *future* price never changes a past feature
row (i.e. no look-ahead leakage).

## How it avoids common mistakes

- **No shuffling.** All splits are time-ordered via `TimeSeriesSplit`.
- **No look-ahead.** Features use only trailing windows; positions are applied to
  the *next* day's return; tests enforce this.
- **Honest benchmark.** Every backtest is reported alongside buy-and-hold.
- **Transaction costs.** Charged on every position change (configurable bps).
