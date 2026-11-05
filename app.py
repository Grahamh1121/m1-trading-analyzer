"""Educational live M1 analyzer using Coinbase Exchange public candles."""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

API = "https://api.exchange.coinbase.com/products/{}/candles"
DB = "m1_signals.sqlite3"
PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD"]


def fetch_candles(pair):
    response = requests.get(API.format(pair), params={"granularity": 60},
                            headers={"User-Agent": "Educational-M1-Analyzer/1.0"}, timeout=12)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Unexpected candle response: {rows}")
    df = pd.DataFrame(rows, columns=["timestamp", "low", "high", "open", "close", "volume"])
    for col in ["timestamp", "low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Coinbase may include the candle still forming: exclude it.
    current_minute = int(time.time() // 60 * 60)
    return df[df.timestamp < current_minute].copy().reset_index(drop=True)


def indicators(df):
    df = df.copy()
    close, high, low = df.close, df.high, df.low
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    df.loc[(loss == 0) & (gain > 0), "rsi14"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi14"] = 50
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["plus_di"] = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    df["minus_di"] = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (df.plus_di-df.minus_di).abs() / (df.plus_di+df.minus_di).replace(0, np.nan)
    df["adx14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return df


def rules(df):
    x = df.iloc[-1]
    required = [x.ema9, x.ema21, x.rsi14, x.adx14, x.plus_di, x.minus_di]
    if not all(np.isfinite(v) for v in required):
        return "WAIT", "Insufficient indicator history"
    if x.adx14 < 20:
        return "WAIT", "ADX below 20; weak or ranging trend"
    if x.ema9 > x.ema21 and x.plus_di > x.minus_di and 50 < x.rsi14 < 70 and x.close > x.ema9:
        return "BUY", "EMA uptrend, +DI leads, RSI supports upward momentum"
    if x.ema9 < x.ema21 and x.minus_di > x.plus_di and 30 < x.rsi14 < 50 and x.close < x.ema9:
        return "SELL", "EMA downtrend, -DI leads, RSI supports downward momentum"
    return "WAIT", "Indicators do not align sufficiently"


def explain_with_ai(pair, df, signal, rationale, model):
    x = df.iloc[-1]
    snapshot = {"pair": pair, "candle_start_utc": datetime.fromtimestamp(int(x.timestamp), timezone.utc).isoformat(),
                "open": float(x.open), "high": float(x.high), "low": float(x.low), "close": float(x.close),
                "ema9": float(x.ema9), "ema21": float(x.ema21), "rsi14": float(x.rsi14),
                "adx14": float(x.adx14), "plus_di": float(x.plus_di), "minus_di": float(x.minus_di),
                "rule_signal": signal, "rule_reason": rationale}
    response = OpenAI().responses.create(
        model=model,
        instructions=("You explain an educational 1-minute trading signal. Use only the supplied numeric "
                      "data. Do not invent price levels, probabilities, chart patterns, or performance. "
                      "Do not override the rule signal. Give 2 short sentences: why the signal arose "
                      "and the main uncertainty. This is not personalized investment advice."),
        input=json.dumps(snapshot), max_output_tokens=160)
    return response.output_text


def db_connect():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        pair TEXT NOT NULL, candle INTEGER NOT NULL, signal TEXT NOT NULL,
        close REAL NOT NULL, next_close REAL, outcome TEXT,
        PRIMARY KEY(pair, candle))""")
    return con


def record_and_score(con, pair, df, signal):
    x = df.iloc[-1]
    con.execute("INSERT OR IGNORE INTO signals(pair,candle,signal,close) VALUES(?,?,?,?)",
                (pair, int(x.timestamp), signal, float(x.close)))
    # Score previous signals only when the immediately next minute's candle is present.
    rows = con.execute("SELECT candle,signal,close FROM signals WHERE pair=? AND outcome IS NULL AND signal != 'WAIT'",
                       (pair,)).fetchall()
    lookup = {int(r.timestamp): float(r.close) for r in df.itertuples()}
    for candle, old_signal, old_close in rows:
        next_close = lookup.get(candle + 60)
        if next_close is None:
            continue
        outcome = "TIE" if next_close == old_close else (
            "WIN" if (next_close > old_close) == (old_signal == "BUY") else "LOSS")
        con.execute("UPDATE signals SET next_close=?,outcome=? WHERE pair=? AND candle=?",
                    (next_close, outcome, pair, candle))
    con.commit()


st.set_page_config(page_title="Live M1 Analyzer", layout="wide")
st.title("Live M1 Trading Analyzer")
st.caption("Educational signals only • Coinbase Exchange public data • no order execution")
selected = st.multiselect("Crypto pairs", PAIRS, default=PAIRS)
interval = st.slider("Refresh interval (seconds)", 10, 60, 20, 5)
use_ai = st.checkbox("Use OpenAI for commentary on each new candle", value=False)
model = st.text_input("OpenAI model", value="gpt-4.1-mini") if use_ai else "gpt-4.1-mini"
if use_ai and not os.environ.get("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in your environment before enabling AI commentary.")

# Streamlit fragment reruns automatically while the browser session is active.
@st.fragment(run_every=f"{interval}s")
def dashboard():
    if not selected:
        st.info("Select at least one pair")
        return
    con = db_connect()
    try:
        for pair in selected:
            try:
                df = indicators(fetch_candles(pair))
                if len(df) < 60:
                    st.warning(f"{pair}: not enough completed candles")
                    continue
                signal, reason = rules(df)
                x = df.iloc[-1]
                prior = con.execute("SELECT 1 FROM signals WHERE pair=? AND candle=?", (pair, int(x.timestamp))).fetchone()
                record_and_score(con, pair, df, signal)
                st.subheader(f"{pair} — {signal}")
                st.write(f"Last completed candle: {datetime.fromtimestamp(int(x.timestamp), timezone.utc):%Y-%m-%d %H:%M} UTC")
                st.write(f"Close: {x.close:,.4f} | EMA9: {x.ema9:,.4f} | EMA21: {x.ema21:,.4f} | RSI: {x.rsi14:.1f} | ADX: {x.adx14:.1f}")
                st.write(reason)
                chart = df.tail(100).copy()
                chart["time"] = pd.to_datetime(chart.timestamp, unit="s", utc=True)
                st.line_chart(chart.set_index("time")[["close", "ema9", "ema21"]])
                if use_ai and os.environ.get("OPENAI_API_KEY") and not prior:
                    try:
                        st.info(explain_with_ai(pair, df, signal, reason, model))
                    except Exception as exc:
                        st.warning(f"AI commentary unavailable: {exc}")
            except (requests.RequestException, ValueError, KeyError) as exc:
                st.error(f"{pair}: market data unavailable: {exc}")
        history = pd.read_sql_query("SELECT * FROM signals ORDER BY candle DESC LIMIT 100", con)
        st.subheader("Signal history")
        st.dataframe(history, use_container_width=True)
        scored = history[history.outcome.isin(["WIN", "LOSS"])] if not history.empty else history
        if len(scored):
            st.metric("Directional win rate (scored, non-tie)", f"{(scored.outcome == 'WIN').mean():.1%}",
                      help="Uses completed candle close as hypothetical entry; ignores latency, spreads, and fees.")
    finally:
        con.close()

dashboard()
st.caption(f"Polling every {interval}s while the dashboard is open. Coinbase may omit intervals with no trades.")
