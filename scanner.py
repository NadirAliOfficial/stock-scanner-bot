#!/usr/bin/env python3
"""
Stock Scanner — Daily post-market technical analysis bot.
Runs once per day after US market close.
Outputs a shortlist of trade candidates based on predefined technical criteria.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Configuration ────────────────────────────────────────────────────────────

# Watchlist file: accepts .csv, .json, or .txt (one symbol per line)
WATCHLIST_FILE = "watchlist.csv"

OUTPUT_DIR = "output"

# Support/resistance lookback window in trading days (~1 year)
SR_PERIOD = 252

# Filter thresholds
MIN_AVG_DOLLAR_VOLUME = 20_000_000   # $20M
MIN_AMPLITUDE        = 0.10          # 10%
MAX_SUPPORT_DISTANCE = 0.05          # 5%
MIN_RESISTANCE_DIST  = 0.05          # 5%


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("scanner")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─── Watchlist loading ────────────────────────────────────────────────────────

def load_watchlist(path: str) -> list[str]:
    """
    Load symbol list from:
      - .json  → list of strings or list of objects with a "symbol" key
      - .csv   → first column is treated as symbols (header optional)
      - .txt   → one symbol per line, # comments ignored
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                symbols = data
            else:
                symbols = [item["symbol"] for item in data]
        elif isinstance(data, dict):
            symbols = data.get("symbols", list(data.keys()))
        else:
            raise ValueError("Unrecognised JSON structure in watchlist")

    elif ext == ".csv":
        df = pd.read_csv(p, header=None)
        raw = df.iloc[:, 0].astype(str).tolist()
        # Skip a header row if the first cell looks like a label
        symbols = [s.strip() for s in raw
                   if s.strip() and not s.strip().lower() in ("symbol", "ticker", "#")]

    else:  # .txt or anything else
        with open(p) as f:
            lines = f.readlines()
        symbols = [l.strip() for l in lines
                   if l.strip() and not l.strip().startswith("#")]

    return [s.upper() for s in symbols]


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_daily(symbol: str) -> pd.DataFrame:
    """~2 years of daily OHLCV, fully closed candles only."""
    df = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def fetch_weekly(symbol: str) -> pd.DataFrame:
    """~5 years of weekly OHLCV. Last incomplete week is dropped."""
    df = yf.Ticker(symbol).history(period="5y", interval="1wk", auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    # Drop current (incomplete) week: weekly bar opens Monday, closes Friday
    today = pd.Timestamp.today().normalize()
    if not df.empty:
        week_end = df.index[-1] + pd.Timedelta(days=4)   # Friday
        if week_end >= today:
            df = df.iloc[:-1]
    return df


# ─── Indicators ───────────────────────────────────────────────────────────────

def weekly_ma200_metrics(weekly: pd.DataFrame) -> tuple[float, float, bool, int]:
    """
    Returns (ma200_current, ma200_prev, trending_up, weeks_above_ma200).
    Requires at least 201 weekly bars.
    """
    if len(weekly) < 201:
        return np.nan, np.nan, False, 0

    closes = weekly["Close"]
    ma200  = closes.rolling(200).mean()

    current_ma200 = float(ma200.iloc[-1])
    prev_ma200    = float(ma200.iloc[-2])
    trending_up   = current_ma200 > prev_ma200

    # Count consecutive weeks where close > MA200 (from most recent backward)
    weeks_above = 0
    for i in range(len(closes) - 1, -1, -1):
        if pd.isna(ma200.iloc[i]):
            break
        if closes.iloc[i] > ma200.iloc[i]:
            weeks_above += 1
        else:
            break

    return current_ma200, prev_ma200, trending_up, weeks_above


def calc_support_resistance(daily: pd.DataFrame, period: int) -> tuple[float, float]:
    window = daily.iloc[-period:]
    return float(window["Low"].min()), float(window["High"].max())


def calc_avg_dollar_volume(daily: pd.DataFrame, n: int = 20) -> float:
    if len(daily) < n:
        return np.nan
    last_n = daily.iloc[-n:]
    return float((last_n["Close"] * last_n["Volume"]).mean())


def calc_atr(daily: pd.DataFrame, n: int = 14) -> float:
    """Average True Range over last n days."""
    if len(daily) < n + 1:
        return np.nan
    high  = daily["High"]
    low   = daily["Low"]
    close = daily["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def is_two_green_candles(daily: pd.DataFrame) -> bool:
    """Last two completed daily candles are both bullish (close > open)."""
    if len(daily) < 2:
        return False
    last2 = daily.iloc[-2:]
    return bool((last2["Close"] > last2["Open"]).all())


def is_rebound_pattern(daily: pd.DataFrame) -> bool:
    """
    Bullish rejection / bounce on the latest bar (hammer):
      - lower wick >= 2× body
      - close in upper 40% of candle range
    """
    if len(daily) < 1:
        return False
    bar = daily.iloc[-1]
    o, h, l, c = bar["Open"], bar["High"], bar["Low"], bar["Close"]
    body        = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return False
    lower_wick     = min(o, c) - l
    close_position = (c - l) / candle_range
    return bool((lower_wick >= 2 * body) and (close_position >= 0.60))


def is_volume_confirmed(daily: pd.DataFrame, n: int = 20) -> bool:
    if len(daily) < n + 1:
        return False
    avg_vol    = daily["Volume"].iloc[-(n + 1):-1].mean()
    latest_vol = daily["Volume"].iloc[-1]
    return bool(latest_vol > avg_vol)


# ─── Filters ─────────────────────────────────────────────────────────────────

def apply_filters(m: dict) -> bool:
    """All five mandatory filters must pass."""
    if m["close"] <= m["weekly_ma200"] or not m["weekly_ma200_trending_up"]:
        return False
    if m["avg_dollar_volume"] < MIN_AVG_DOLLAR_VOLUME:
        return False
    if m["amplitude"] < MIN_AMPLITUDE:
        return False
    if m["support_distance"] > MAX_SUPPORT_DISTANCE:
        return False
    if m["resistance_distance"] < MIN_RESISTANCE_DIST:
        return False
    return True


# ─── Scoring ──────────────────────────────────────────────────────────────────

def calculate_score(m: dict) -> int:
    score = 0

    # 1. Weekly Trend (0 or 2)
    if m["close"] > m["weekly_ma200"] and m["weekly_ma200_trending_up"]:
        score += 2

    # 2. Support Proximity (0, 1 or 2)
    sd = m["support_distance"]
    if sd <= 0.02:
        score += 2
    elif sd <= 0.05:
        score += 1

    # 3. Amplitude (0, 1 or 2)
    amp = m["amplitude"]
    if amp >= 0.20:
        score += 2
    elif amp >= 0.10:
        score += 1

    # 4. Rebound Pattern (0 or 2)
    if m["rebound_pattern"]:
        score += 2

    # 5. Volume Confirmation (0 or 1)
    if m["volume_confirmed"]:
        score += 1

    # 6. Liquidity Bonus (0 or 1)
    if m["avg_dollar_volume"] >= MIN_AVG_DOLLAR_VOLUME:
        score += 1

    return score


def build_comment(m: dict, passes: bool) -> str:
    if pd.isna(m.get("weekly_ma200", np.nan)):
        return "Insufficient data"
    parts = []
    if passes:
        if m["support_distance"] <= 0.02:
            parts.append("Very close to support")
        elif m["support_distance"] <= 0.05:
            parts.append("Near support")
        if m["amplitude"] >= 0.20:
            parts.append("Wide amplitude")
        if m["rebound_pattern"]:
            parts.append("Rebound pattern")
        if m["volume_confirmed"]:
            parts.append("Volume confirmed")
        return " / ".join(parts) if parts else "Passes all filters"
    else:
        reasons = []
        if m["close"] <= m["weekly_ma200"]:
            reasons.append("Below MA200")
        if not m["weekly_ma200_trending_up"]:
            reasons.append("MA200 not trending up")
        if m["avg_dollar_volume"] < MIN_AVG_DOLLAR_VOLUME:
            reasons.append("Low liquidity")
        if m["amplitude"] < MIN_AMPLITUDE:
            reasons.append("Amplitude < 10%")
        if m["support_distance"] > MAX_SUPPORT_DISTANCE:
            reasons.append(f"Support too far ({m['support_distance']*100:.1f}%)")
        if m["resistance_distance"] < MIN_RESISTANCE_DIST:
            reasons.append(f"Resistance too close ({m['resistance_distance']*100:.1f}%)")
        return " / ".join(reasons)


# ─── Per-symbol processing ────────────────────────────────────────────────────

def process_symbol(symbol: str, logger: logging.Logger) -> dict | None:
    try:
        daily  = fetch_daily(symbol)
        weekly = fetch_weekly(symbol)

        if daily.empty or len(daily) < 22:
            logger.warning("%s: insufficient daily data (%d bars)", symbol, len(daily))
            return None
        if weekly.empty or len(weekly) < 201:
            logger.warning("%s: insufficient weekly data (%d bars)", symbol, len(weekly))
            return None

        close = float(daily["Close"].iloc[-1])
        support, resistance = calc_support_resistance(daily, SR_PERIOD)
        ma200, prev_ma200, ma200_up, weeks_above = weekly_ma200_metrics(weekly)
        avg_dollar_volume = calc_avg_dollar_volume(daily)
        avg_volume_20d    = float(daily["Volume"].iloc[-21:-1].mean())
        latest_volume     = int(daily["Volume"].iloc[-1])
        atr14             = calc_atr(daily, 14)

        if np.isnan(ma200) or np.isnan(avg_dollar_volume):
            logger.warning("%s: could not compute required metrics", symbol)
            return None

        support_distance    = (close - support) / support
        resistance_distance = (resistance - close) / close
        amplitude           = (resistance - support) / support
        atr_pct             = (atr14 / close) if close > 0 else np.nan

        m = {
            "close":                   close,
            "support":                 support,
            "resistance":              resistance,
            "support_distance":        support_distance,
            "resistance_distance":     resistance_distance,
            "amplitude":               amplitude,
            "weekly_ma200":            ma200,
            "weekly_ma200_trending_up": ma200_up,
            "avg_dollar_volume":       avg_dollar_volume,
            "rebound_pattern":         is_rebound_pattern(daily),
            "volume_confirmed":        is_volume_confirmed(daily),
        }

        passes = apply_filters(m)
        score  = calculate_score(m)
        comment = build_comment(m, passes)

        return {
            # ── Core spec columns ──────────────────────────────────────
            "Symbol":               symbol,
            "Close":                round(close, 4),
            "Support":              round(support, 4),
            "Resistance":           round(resistance, 4),
            "SupportDistance":      round(support_distance, 4),
            "ResistanceDistance":   round(resistance_distance, 4),
            "Amplitude":            round(amplitude, 4),
            "Score":                score,
            "CandidateFlag":        "TRUE" if passes else "FALSE",
            # ── Weekly trend detail ────────────────────────────────────
            "WeeklyMA200":          round(ma200, 4),
            "PriceAboveMA200":      close > ma200,
            "WeeksAboveMA200":      weeks_above,
            "MA200Slope":           "UP" if ma200_up else "DOWN",
            # ── Support / resistance flags ─────────────────────────────
            "NearSupport":          support_distance <= MAX_SUPPORT_DISTANCE,
            "NearResistance":       resistance_distance <= MIN_RESISTANCE_DIST,
            "AmplitudeValid":       amplitude >= MIN_AMPLITUDE,
            # ── Candle / rebound ──────────────────────────────────────
            "TwoGreenCandles":      is_two_green_candles(daily),
            "ReboundPattern":       m["rebound_pattern"],
            # ── Volume ────────────────────────────────────────────────
            "LatestVolume":         latest_volume,
            "AvgVolume20D":         int(round(avg_volume_20d)),
            "VolumeConfirmed":      m["volume_confirmed"],
            # ── ATR / volatility ──────────────────────────────────────
            "ATR14":                round(atr14, 4) if not np.isnan(atr14) else None,
            "ATR_Pct":              round(atr_pct * 100, 2) if not np.isnan(atr_pct) else None,
            # ── Liquidity ─────────────────────────────────────────────
            "AvgDollarVolume20":    int(round(avg_dollar_volume)),
            "LiquidityValid":       avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME,
            # ── Comment ───────────────────────────────────────────────
            "Comment":              comment,
        }

    except Exception as e:
        logger.error("%s: %s", symbol, e)
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(OUTPUT_DIR, f"scan_{ts}.log")
    csv_path = os.path.join(OUTPUT_DIR, f"scan_{ts}.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"scan_{ts}.xlsx")

    logger = setup_logging(log_path)
    start_time = datetime.now()
    logger.info("Scanner started")

    symbols = load_watchlist(WATCHLIST_FILE)
    logger.info("Symbols to scan: %d", len(symbols))

    results = []
    errors  = 0

    for symbol in symbols:
        logger.info("Processing: %s", symbol)
        row = process_symbol(symbol, logger)
        if row is not None:
            results.append(row)
        else:
            errors += 1

    df = pd.DataFrame(results)

    # ── CSV output ────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    logger.info("CSV written:  %s", csv_path)

    # ── XLSX output ───────────────────────────────────────────────────────────
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Scan")
        # Auto-fit column widths
        ws = writer.sheets["Scan"]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0
                         for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)
    logger.info("XLSX written: %s", xlsx_path)

    candidates = df[df["CandidateFlag"] == "TRUE"] if not df.empty else df

    end_time = datetime.now()
    elapsed  = (end_time - start_time).total_seconds()

    logger.info("─── Summary ───────────────────────────────────")
    logger.info("Start:      %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("End:        %s", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Elapsed:    %.1fs", elapsed)
    logger.info("Scanned:    %d", len(symbols))
    logger.info("Processed:  %d", len(results))
    logger.info("Candidates: %d", len(candidates))
    logger.info("Errors:     %d", errors)
    logger.info("───────────────────────────────────────────────")

    if not candidates.empty:
        print("\n=== CANDIDATES ===")
        cols = ["Symbol", "Close", "SupportDistance", "ResistanceDistance",
                "Amplitude", "Score", "Comment"]
        print(candidates[cols].to_string(index=False))
    else:
        print("\nNo candidates found today.")


if __name__ == "__main__":
    main()
