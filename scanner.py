#!/usr/bin/env python3
"""
Stock Scanner — Daily post-market technical analysis bot.
Runs once per day after US market close.
Data source: Interactive Brokers (IB Gateway / TWS) via ib_insync.
Outputs a shortlist of trade candidates based on predefined technical criteria.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, util

# ─── Configuration ────────────────────────────────────────────────────────────

# Watchlist file: accepts .csv, .json, or .txt (one symbol per line)
WATCHLIST_FILE = "watchlist.csv"

OUTPUT_DIR = "output"

# IBKR connection settings
IB_HOST      = "127.0.0.1"
IB_PORT      = 4001        # IB Gateway live
IB_CLIENT_ID = 3           # confirmed by client

# Symbol overrides: maps watchlist symbol → (ib_symbol, exchange, currency)
# Use this for tickers that need non-default IBKR contract definitions
SYMBOL_MAP = {
    "BRK-B":   ("BRK B",  "SMART", "USD"),
    "NESN.SW": ("NESN",   "EBS",   "CHF"),
    "CSPX.L":  ("CSPX",   "LSE",   "USD"),
    "SIE.DE":  ("SIE",    "XETRA", "EUR"),
    "ABB":     ("ABB",    "NYSE",  "USD"),
}

# Support/resistance lookback window in trading days (~1 year)
SR_PERIOD = 252

# Filter thresholds
MIN_AVG_DOLLAR_VOLUME = 20_000_000   # $20M
MIN_AMPLITUDE        = 0.10          # 10%
MAX_SUPPORT_DISTANCE = 0.05          # 5%
MIN_RESISTANCE_DIST  = 0.05          # 5%

# Volume confirmation multiplier (spec: >= 1.5x)
VOLUME_CONFIRM_MULT = 1.5

# Volatility flag threshold (ATR% above this → high volatility)
VOLATILITY_ATR_PCT_THRESH = 2.0      # 2% of price


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
      - .csv   → first column treated as symbols (header optional)
      - .txt   → one symbol per line, # comments ignored
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            symbols = data if data and isinstance(data[0], str) else [i["symbol"] for i in data]
        elif isinstance(data, dict):
            symbols = data.get("symbols", list(data.keys()))
        else:
            raise ValueError("Unrecognised JSON structure in watchlist")

    elif ext == ".csv":
        df = pd.read_csv(p, header=None)
        raw = df.iloc[:, 0].astype(str).tolist()
        symbols = [s.strip() for s in raw
                   if s.strip() and s.strip().lower() not in ("symbol", "ticker", "#")]

    else:
        with open(p) as f:
            lines = f.readlines()
        symbols = [l.strip() for l in lines
                   if l.strip() and not l.strip().startswith("#")]

    return [s.upper() for s in symbols]


# ─── IBKR data fetching ───────────────────────────────────────────────────────

def make_contract(symbol: str) -> Stock:
    ib_sym, exchange, currency = SYMBOL_MAP.get(symbol, (symbol, "SMART", "USD"))
    return Stock(ib_sym, exchange, currency)


def fetch_daily(ib: IB, symbol: str) -> pd.DataFrame:
    """Fetch ~2 years of daily OHLCV from IBKR."""
    contract = make_contract(symbol)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def fetch_weekly(ib: IB, symbol: str) -> pd.DataFrame:
    """Fetch ~5 years of weekly OHLCV from IBKR. Drops the current incomplete week."""
    contract = make_contract(symbol)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="5 Y",
        barSizeSetting="1 week",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Drop current incomplete week (bar opens Monday, closes Friday)
    today = pd.Timestamp.today().normalize()
    if not df.empty:
        week_end = df.index[-1] + pd.Timedelta(days=4)
        if week_end >= today:
            df = df.iloc[:-1]

    return df


# ─── Indicators ───────────────────────────────────────────────────────────────

def weekly_ma200_metrics(weekly: pd.DataFrame) -> tuple[float, bool, int]:
    if len(weekly) < 201:
        return np.nan, False, 0

    closes = weekly["Close"]
    ma200  = closes.rolling(200).mean()

    current_ma200 = float(ma200.iloc[-1])
    trending_up   = current_ma200 > float(ma200.iloc[-2])

    weeks_above = 0
    for i in range(len(closes) - 1, -1, -1):
        if pd.isna(ma200.iloc[i]):
            break
        if closes.iloc[i] > ma200.iloc[i]:
            weeks_above += 1
        else:
            break

    return current_ma200, trending_up, weeks_above


# ── 2.1  Support / Resistance — pivot-point detection with clustering ─────────

def _find_pivot_lows(daily: pd.DataFrame, order: int = 5) -> list[float]:
    """Detect local minima (pivot lows) using a rolling window of ±order bars."""
    lows = daily["Low"].values
    pivots = []
    for i in range(order, len(lows) - order):
        if lows[i] == min(lows[i - order:i + order + 1]):
            pivots.append(float(lows[i]))
    return pivots


def _find_pivot_highs(daily: pd.DataFrame, order: int = 5) -> list[float]:
    """Detect local maxima (pivot highs) using a rolling window of ±order bars."""
    highs = daily["High"].values
    pivots = []
    for i in range(order, len(highs) - order):
        if highs[i] == max(highs[i - order:i + order + 1]):
            pivots.append(float(highs[i]))
    return pivots


def _cluster_levels(levels: list[float], tolerance: float = 0.02) -> list[float]:
    """Cluster nearby price levels (within tolerance %) and return the mean of each cluster."""
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters = [[sorted_levels[0]]]
    for lvl in sorted_levels[1:]:
        if (lvl - clusters[-1][0]) / clusters[-1][0] <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    return [np.mean(c) for c in clusters]


def calc_support_resistance(daily: pd.DataFrame, period: int) -> tuple[float, float]:
    """
    Detect recent support and resistance levels using pivot-point clustering.
    Support  = nearest clustered pivot-low zone at or below current price.
    Resistance = nearest clustered pivot-high zone at or above current price.
    Falls back to period min/max if no pivot zones are found.
    """
    window = daily.iloc[-period:]
    close = float(window["Close"].iloc[-1])

    # Find and cluster pivot points
    pivot_lows = _find_pivot_lows(window)
    pivot_highs = _find_pivot_highs(window)

    support_zones = _cluster_levels(pivot_lows)
    resistance_zones = _cluster_levels(pivot_highs)

    # Nearest support at or below price
    supports_below = [s for s in support_zones if s <= close]
    support = max(supports_below) if supports_below else float(window["Low"].min())

    # Nearest resistance at or above price
    resistances_above = [r for r in resistance_zones if r >= close]
    resistance = min(resistances_above) if resistances_above else float(window["High"].max())

    # Guard: resistance must be above support
    if resistance <= support:
        resistance = float(window["High"].max())

    return support, resistance


# ── 2.4  RSI(14) ──────────────────────────────────────────────────────────────

def calc_rsi(daily: pd.DataFrame, n: int = 14) -> float:
    """Wilder's RSI over n periods."""
    if len(daily) < n + 1:
        return np.nan
    delta = daily["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])


# ── 2.4  52-week high / low distances ────────────────────────────────────────

def calc_52w_distances(daily: pd.DataFrame) -> tuple[float, float]:
    """Return (distance_from_52w_high, distance_from_52w_low) as fractions."""
    window = daily.iloc[-252:] if len(daily) >= 252 else daily
    close = float(window["Close"].iloc[-1])
    high_52w = float(window["High"].max())
    low_52w = float(window["Low"].min())
    dist_high = (close - high_52w) / high_52w   # negative when below high
    dist_low = (close - low_52w) / low_52w       # positive when above low
    return dist_high, dist_low


def calc_avg_dollar_volume(daily: pd.DataFrame, n: int = 20) -> float:
    if len(daily) < n:
        return np.nan
    last_n = daily.iloc[-n:]
    return float((last_n["Close"] * last_n["Volume"]).mean())


def calc_atr(daily: pd.DataFrame, n: int = 14) -> float:
    if len(daily) < n + 1:
        return np.nan
    high       = daily["High"]
    low        = daily["Low"]
    close      = daily["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


# ── 2.6  Strong green candles ─────────────────────────────────────────────────

def is_two_strong_green_candles(daily: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Check if the last two candles are strong bullish candles.
    "Strong" = close > open AND body size >= median body size over lookback.
    """
    if len(daily) < max(2, lookback):
        return False
    last2 = daily.iloc[-2:]
    # Both must be green
    if not (last2["Close"] > last2["Open"]).all():
        return False
    # Median body size over recent history as threshold for "strong"
    bodies = (daily["Close"].iloc[-lookback:] - daily["Open"].iloc[-lookback:]).abs()
    median_body = float(bodies.median())
    if median_body == 0:
        return False
    for _, bar in last2.iterrows():
        body = bar["Close"] - bar["Open"]  # green so Close > Open
        if body < median_body:
            return False
    return True


def is_rebound_pattern(daily: pd.DataFrame) -> bool:
    """Hammer: lower wick >= 2x body, close in upper 40% of range."""
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


# ── 2.2  Volume confirmation — >= 1.5x average ───────────────────────────────

def is_volume_confirmed(daily: pd.DataFrame, n: int = 20) -> bool:
    if len(daily) < n + 1:
        return False
    avg_vol    = daily["Volume"].iloc[-(n + 1):-1].mean()
    latest_vol = daily["Volume"].iloc[-1]
    return bool(latest_vol >= VOLUME_CONFIRM_MULT * avg_vol)


# ─── Filters ─────────────────────────────────────────────────────────────────

def apply_filters(m: dict) -> bool:
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
    # Require at least one confirmation signal
    confirmations = (
        m["rebound_pattern"],
        m["volume_confirmed"],
        m["two_strong_green"],
    )
    if not any(confirmations):
        return False
    return True


# ─── Scoring ──────────────────────────────────────────────────────────────────

def calculate_score(m: dict) -> int:
    score = 0

    # Weekly trend
    if m["close"] > m["weekly_ma200"] and m["weekly_ma200_trending_up"]:
        score += 2

    # Support proximity
    sd = m["support_distance"]
    if sd <= 0.02:
        score += 2
    elif sd <= 0.05:
        score += 1

    # Amplitude
    amp = m["amplitude"]
    if amp >= 0.20:
        score += 2
    elif amp >= 0.10:
        score += 1

    # Rebound pattern
    if m["rebound_pattern"]:
        score += 2

    # Volume (2.2: using 1.5x threshold)
    if m["volume_confirmed"]:
        score += 1

    # Liquidity
    if m["avg_dollar_volume"] >= MIN_AVG_DOLLAR_VOLUME:
        score += 1

    # 2.5  Volatility factor
    if m["volatility_flag"]:
        score += 1

    return score


def build_comment(m: dict, passes: bool) -> str:
    if pd.isna(m.get("weekly_ma200", np.nan)):
        return "Insufficient data"
    if passes:
        parts = []
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
        if m["volatility_flag"]:
            parts.append("High volatility")
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
        if not any((m["rebound_pattern"], m["volume_confirmed"], m["two_strong_green"])):
            reasons.append("No confirmation signal")
        return " / ".join(reasons)


# ─── Per-symbol processing ────────────────────────────────────────────────────

def process_symbol(ib: IB, symbol: str, logger: logging.Logger) -> dict | None:
    try:
        daily  = fetch_daily(ib, symbol)
        weekly = fetch_weekly(ib, symbol)

        if daily.empty or len(daily) < 22:
            logger.warning("%s: insufficient daily data (%d bars)", symbol, len(daily))
            return None
        if weekly.empty or len(weekly) < 201:
            logger.warning("%s: insufficient weekly data (%d bars)", symbol, len(weekly))
            return None

        close = float(daily["Close"].iloc[-1])
        support, resistance = calc_support_resistance(daily, SR_PERIOD)
        ma200, ma200_up, weeks_above = weekly_ma200_metrics(weekly)
        avg_dollar_volume = calc_avg_dollar_volume(daily)
        avg_volume_20d    = float(daily["Volume"].iloc[-21:-1].mean())
        latest_volume     = int(daily["Volume"].iloc[-1])
        atr14             = calc_atr(daily, 14)
        rsi14             = calc_rsi(daily, 14)
        dist_52w_high, dist_52w_low = calc_52w_distances(daily)

        if np.isnan(ma200) or np.isnan(avg_dollar_volume):
            logger.warning("%s: could not compute required metrics", symbol)
            return None

        support_distance    = (close - support) / support
        resistance_distance = (resistance - close) / close
        amplitude           = (resistance - support) / support
        atr_pct             = (atr14 / close * 100) if close > 0 else np.nan
        volatility_flag     = bool(not np.isnan(atr_pct) and atr_pct >= VOLATILITY_ATR_PCT_THRESH)

        m = {
            "close":                    close,
            "support":                  support,
            "resistance":               resistance,
            "support_distance":         support_distance,
            "resistance_distance":      resistance_distance,
            "amplitude":                amplitude,
            "weekly_ma200":             ma200,
            "weekly_ma200_trending_up": ma200_up,
            "avg_dollar_volume":        avg_dollar_volume,
            "rebound_pattern":          is_rebound_pattern(daily),
            "volume_confirmed":         is_volume_confirmed(daily),
            "two_strong_green":         is_two_strong_green_candles(daily),
            "volatility_flag":          volatility_flag,
        }

        passes  = apply_filters(m)
        score   = calculate_score(m)
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
            # ── RSI ───────────────────────────────────────────────────
            "RSI14":                round(rsi14, 2) if not np.isnan(rsi14) else None,
            # ── 52-week distances ─────────────────────────────────────
            "Dist52WkHigh":         round(dist_52w_high * 100, 2),
            "Dist52WkLow":          round(dist_52w_low * 100, 2),
            # ── Candle / rebound ──────────────────────────────────────
            "TwoStrongGreenCandles": m["two_strong_green"],
            "ReboundPattern":       m["rebound_pattern"],
            # ── Volume ────────────────────────────────────────────────
            "LatestVolume":         latest_volume,
            "AvgDailyVolume":       int(round(avg_volume_20d)),
            "VolumeConfirmed":      m["volume_confirmed"],
            # ── ATR / volatility ──────────────────────────────────────
            "ATR14":                round(atr14, 4) if not np.isnan(atr14) else None,
            "ATR_Pct":              round(atr_pct, 2) if not np.isnan(atr_pct) else None,
            "VolatilityFlag":       volatility_flag,
            # ── Liquidity ─────────────────────────────────────────────
            "AvgDollarVolume":      int(round(avg_dollar_volume)),
            "LiquidityFlag":        avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME,
            # ── Comment ───────────────────────────────────────────────
            "Comment":              comment,
        }

    except Exception as e:
        logger.error("%s: %s", symbol, e)
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",      default=IB_HOST)
    parser.add_argument("--port",      type=int, default=IB_PORT)
    parser.add_argument("--client-id", type=int, default=IB_CLIENT_ID)
    parser.add_argument("--watchlist", default=WATCHLIST_FILE)
    parser.add_argument("--output",    default=OUTPUT_DIR)
    args = parser.parse_args()

    host       = args.host
    port       = args.port
    client_id  = args.client_id
    watchlist  = args.watchlist
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    log_path  = os.path.join(output_dir, f"scan_{ts}.log")
    csv_path  = os.path.join(output_dir, f"scan_{ts}.csv")
    xlsx_path = os.path.join(output_dir, f"scan_{ts}.xlsx")

    logger = setup_logging(log_path)
    start_time = datetime.now()
    logger.info("Scanner started")

    # Connect to IBKR
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=20)
    logger.info("Connected to IBKR at %s:%s (clientId=%s)", host, port, client_id)

    symbols = load_watchlist(watchlist)
    logger.info("Symbols to scan: %d", len(symbols))

    results = []
    errors  = 0

    for symbol in symbols:
        logger.info("Processing: %s", symbol)
        row = process_symbol(ib, symbol, logger)
        if row is not None:
            results.append(row)
        else:
            errors += 1

    ib.disconnect()
    logger.info("Disconnected from IBKR")

    df = pd.DataFrame(results)

    # ── CSV output ────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    logger.info("CSV written:  %s", csv_path)

    # ── XLSX output ───────────────────────────────────────────────────────────
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Scan")
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
