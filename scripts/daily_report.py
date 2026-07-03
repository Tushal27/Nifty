#!/usr/bin/env python
"""Generate the daily Nifty report (next-day direction + options ticket) as HTML.

Designed to run in CI each evening: it fetches fresh spot (^NSEI) and India VIX
(^INDIAVIX) from Yahoo when the network allows, falls back to the committed CSVs
otherwise, then writes ``outputs/daily_report.html`` (and a .txt) for emailing.

Honest by construction: the direction call is a thin-edge model (~0.55 AUC) and
the options ticket only says SELL when the variance risk premium is favourable.
Not financial advice.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from nifty.config import load_config  # noqa: E402
from nifty.data_loader import _fetch_yfinance, load_data  # noqa: E402
from nifty.features import build_dataset  # noqa: E402
from nifty.models import build_models  # noqa: E402
from nifty.options.backtest import OptionsConfig, run_options_backtest  # noqa: E402
from nifty.options.sizing import SizingConfig  # noqa: E402
from nifty.options.ticket import build_ticket  # noqa: E402


def _load_vix_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    dc = next(c for c in ("date", "datetime") if c in df.columns)
    df[dc] = pd.to_datetime(df[dc])
    return (df.set_index(dc)["close"].sort_index() / 100.0).rename("vix")


def get_data(config):
    """Return (ohlcv_df, vix_decimal_series, source_label). Live first, CSV fallback."""
    try:
        df = _fetch_yfinance("^NSEI", "2005-01-01", None)
        import yfinance as yf

        vraw = yf.download("^INDIAVIX", start="2008-01-01", auto_adjust=True,
                           progress=False)
        if vraw is None or vraw.empty:
            raise RuntimeError("empty VIX")
        if isinstance(vraw.columns, pd.MultiIndex):
            vraw.columns = vraw.columns.get_level_values(0)
        vix = (vraw["Close"] / 100.0)
        vix.index = pd.to_datetime(vix.index)
        vix.name = "vix"
        return df, vix, "live (Yahoo ^NSEI + ^INDIAVIX)"
    except Exception as exc:  # noqa: BLE001 - fall back to committed data
        df = load_data(config)
        vpath = config.raw.get("options", {}).get("vix_csv",
                                                  "external_data/india_vix.csv")
        vix = _load_vix_csv(vpath)
        return df, vix, f"cached CSV (live fetch failed: {str(exc)[:80]})"


def direction_call(config, df):
    """Train a fast model on all history and predict the next day's direction."""
    cfg = load_config()  # a clean copy so we can override safely
    cfg.raw["target"] = {"mode": "close_to_close", "horizon": 1}
    cfg.raw["external"] = {"enabled": False}  # price-only: robust, no extra files
    cfg.raw["models"] = {"logistic": False, "random_forest": True,
                         "xgboost": False, "lightgbm": False, "lstm": False}
    X, y, _ = build_dataset(df, cfg)
    model = build_models(cfg)["random_forest"]().fit(X.values, y.values)
    prob = float(model.predict_proba(X.values)[-1])
    return {
        "as_of": str(X.index[-1].date()),
        "prob_up": round(prob, 4),
        "signal": "UP" if prob > 0.5 else "DOWN",
        "confidence": abs(prob - 0.5),
    }


def options_ticket(config, df, vix):
    opt_raw = dict(config.raw.get("options", {}))
    opt_cfg = OptionsConfig.from_dict(opt_raw)
    sizing_cfg = SizingConfig.from_dict({
        **dict(opt_raw.get("sizing", {})),
        "cycle_days": opt_cfg.cycle_days, "trading_days": opt_cfg.trading_days,
    })
    live = config.raw.get("live", {})
    spot = df["close"]
    v = vix.reindex(spot.index).ffill()
    bt, _ = run_options_backtest(spot, v, opt_cfg)
    return build_ticket(spot, vix, opt_cfg, sizing_cfg, per_cycle_ret=bt["ret"],
                        lot_size=int(live.get("lot_size", 75)),
                        vrp_min=float(live.get("vrp_min_to_trade", 0.02)),
                        realized_window=int(opt_raw.get("realized_window", 20)))


def render_html(direction, ticket, source) -> str:
    d, t = direction, ticket
    dir_colour = "#1a7f37" if d["signal"] == "UP" else "#cf222e"
    sell = t["action"].startswith("SELL")
    legs_html = "".join(
        f"<li><b>{html.escape(l['side'])}</b> {l['lots']} lot "
        f"{html.escape(l['kind'])} @ {l['strike']:.0f}</li>"
        for l in t["legs"]
    ) if sell else "<li>—</li>"
    return f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:640px;margin:auto;color:#1f2328">
  <h2 style="margin-bottom:2px">📈 Nifty 50 — Daily Prediction</h2>
  <div style="color:#57606a;font-size:13px">as of {html.escape(t['entry_date'])} &nbsp;·&nbsp; data: {html.escape(source)}</div>

  <div style="border:1px solid #d0d7de;border-radius:10px;padding:14px 16px;margin:14px 0">
    <div style="font-size:13px;color:#57606a">NEXT-DAY DIRECTION (thin-edge model)</div>
    <div style="font-size:26px;font-weight:700;color:{dir_colour}">{d['signal']}
      <span style="font-size:15px;color:#57606a;font-weight:400">
      &nbsp;P(up)={d['prob_up']:.3f}</span></div>
    <div style="font-size:12px;color:#8b949e">Spot {t['spot']:.1f} · low-confidence by
      design (daily direction is ~random)</div>
  </div>

  <div style="border:1px solid #d0d7de;border-radius:10px;padding:14px 16px;margin:14px 0">
    <div style="font-size:13px;color:#57606a">OPTIONS TRADE TICKET (variance risk premium)</div>
    <div style="font-size:20px;font-weight:700;margin:4px 0">{html.escape(t['action'])}</div>
    <div style="font-size:13px">VIX {t['implied_vol_vix']*100:.1f} vs realized
      {t['realized_vol']*100:.1f} &nbsp;→&nbsp; <b>VRP {t['variance_risk_premium']:+.4f}</b>
      (sell only if &gt; {t['vrp_min_to_trade']})</div>
    {"<div style='margin-top:8px;font-size:13px'>Expiry ≈ " + html.escape(t['expiry_approx'])
      + f" · expected move ±{t['expected_move_pts']:.0f} pts · leverage {t['leverage_used']}x</div>"
      + "<ul style='margin:6px 0'>" + legs_html + "</ul>"
      + f"<div style='font-size:13px'>Credit ≈ ₹{t['credit_total_rupees']:,.0f} · "
        f"margin ≈ ₹{t['margin_estimate']:,.0f} · "
        f"breakevens {t['breakeven_lower']:.0f}/{t['breakeven_upper']:.0f}</div>"
        f"<div style='font-size:12px;color:#8b949e'>Stop: {html.escape(t['stop_loss_rule'])} · "
        f"{html.escape(t['risk_note'])}</div>"
      if sell else "<div style='font-size:13px;color:#57606a'>No favourable premium-selling edge today.</div>"}
  </div>

  <div style="font-size:11px;color:#8b949e;border-top:1px solid #eaeef2;padding-top:8px">
    ⚠ Educational only. Strikes/credit are Black-Scholes-from-VIX estimates (no skew);
    real fills differ. Short options carry large tail risk. Paper-trade first.
    <b>Not financial advice.</b>
  </div>
</div>"""


def render_text(direction, ticket, source) -> str:
    d, t = direction, ticket
    lines = [
        f"Nifty 50 Daily Prediction — as of {t['entry_date']}  (data: {source})",
        "",
        f"NEXT-DAY DIRECTION: {d['signal']}  (P(up)={d['prob_up']:.3f})  spot {t['spot']:.1f}",
        "  (thin-edge model; daily direction is ~random by nature)",
        "",
        f"OPTIONS TICKET: {t['action']}",
        f"  VIX {t['implied_vol_vix']*100:.1f} vs realized {t['realized_vol']*100:.1f}"
        f"  -> VRP {t['variance_risk_premium']:+.4f} (sell if > {t['vrp_min_to_trade']})",
    ]
    if t["action"].startswith("SELL"):
        lines.append(f"  expiry ~{t['expiry_approx']}, ±{t['expected_move_pts']:.0f} pts, "
                     f"leverage {t['leverage_used']}x")
        for l in t["legs"]:
            lines.append(f"    {l['side']} {l['lots']} lot {l['kind']} @ {l['strike']:.0f}")
        lines.append(f"  credit ~Rs{t['credit_total_rupees']:,.0f}, margin ~Rs{t['margin_estimate']:,.0f}, "
                     f"breakevens {t['breakeven_lower']:.0f}/{t['breakeven_upper']:.0f}")
        lines.append(f"  stop: {t['stop_loss_rule']}; {t['risk_note']}")
    lines += ["", "Educational only. Short options carry large tail risk. "
              "Paper-trade first. Not financial advice."]
    return "\n".join(lines)


def main():
    config = load_config()
    df, vix, source = get_data(config)
    direction = direction_call(config, df)
    ticket = options_ticket(config, df, vix)

    os.makedirs(config.output["dir"], exist_ok=True)
    html_path = os.path.join(config.output["dir"], "daily_report.html")
    txt_path = os.path.join(config.output["dir"], "daily_report.txt")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(direction, ticket, source))
    text = render_text(direction, ticket, source)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    print(text)
    print(f"\nWrote {html_path} and {txt_path}  (generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC)")


if __name__ == "__main__":
    main()
