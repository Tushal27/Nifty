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
| Data | `src/nifty/data_loader.py` | Fetch via `yfinance` (`^NSEI`, ~2007+), **NSE** niftyindices.com (~1996+), **or** your own CSV; clean + cache |
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

Pick a source with `data.source` in `config.yaml`:

| `data.source` | Provider | History depth | Notes |
|---|---|---|---|
| `yfinance` (default) | Yahoo Finance `^NSEI` | ~**2007** → today | Easiest; limited by Yahoo's depth |
| `nse` | niftyindices.com API | ~**1996** → today | **Maximum real Nifty 50 history** (index inception); paged one year at a time |
| *(CSV)* | `data.csv_path` | whatever your file holds | Always overrides the API |

> **Reality on "50 years":** the Nifty 50 index itself only began on **3 Nov 1995**
> (launched Apr 1996), so ~**30 years** is the true maximum — there is no 50-year
> Nifty 50 series. Use `source: "nse"` to get that full span. For your own file,
> set `data.csv_path` to a CSV with columns `Date, Open, High, Low, Close, Volume`
> (case-insensitive; `Volume` optional for index data).

```bash
python scripts/fetch_data.py            # download + cache to data/nifty.parquet
python scripts/fetch_data.py --refresh  # ignore cache and re-download
```

> Network note: both Yahoo and niftyindices.com must be reachable. In restricted
> environments the fetch fails with a clear message and you should use a CSV.

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

## Prediction horizon (and why "70% accuracy" is a trap)

Set `target.horizon` in `config.yaml` (or `--horizon N`):

- `horizon: 1` — next-day direction. The hardest case: ~52–55% accuracy, which is
  near the information limit of daily price data.
- `horizon: 5–20+` — "will price be higher N days ahead?" (multi-day **trend**).
  The accuracy *number* climbs toward 60–65% as N grows — **but** that rise is the
  market's upward **drift (base rate)**, not model skill.

To keep this honest, every run prints a **majority-class baseline** ("always
predict the more common direction") and the model's **skill = accuracy −
baseline**. Empirically on this data the skill is *negative* at every horizon — a
trivial "always UP" model matches or beats the trained models. So a high headline
accuracy at long horizons is **drift, not prediction**. Folds are also embargoed by
`horizon − 1` days (`TimeSeriesSplit(gap=...)`) so overlapping labels can't leak.

Takeaway: judge models by **skill above baseline / ROC-AUC**, never by raw
accuracy on an imbalanced target.

## How it avoids common mistakes

- **No shuffling.** All splits are time-ordered via `TimeSeriesSplit`.
- **No look-ahead.** Features use only trailing windows; positions are applied to
  the *next* day's return; tests enforce this.
- **Honest benchmark.** Every backtest is reported alongside buy-and-hold.
- **Transaction costs.** Charged on every position change (configurable bps).
