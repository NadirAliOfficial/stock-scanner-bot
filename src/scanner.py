#!/usr/bin/env python3
"""
Stock Scanner — Daily post-market technical analysis bot.
Runs once per day after US market close.
Data source: Interactive Brokers (IB Gateway / TWS) via ib_insync.
Outputs a shortlist of trade candidates based on predefined technical criteria.
"""

import json
import operator
import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ib_insync import IB, Stock, util

# ─── Configuration ────────────────────────────────────────────────────────────

WATCHLIST_FILE  = "config/watchlist.csv"
SCORING_CONFIG  = "config/scoring.yaml"
OUTPUT_DIR      = "output"

IB_HOST      = "127.0.0.1"
IB_PORT      = 4002
IB_CLIENT_ID = 3

SYMBOL_MAP = {
    "BRK-B":   ("BRK B",  "SMART", "USD"),
    "NESN.SW": ("NESN",   "EBS",   "CHF"),
    "CSPX.L":  ("CSPX",   "LSE",   "USD"),
    "SIE.DE":  ("SIE",    "XETRA", "EUR"),
    "ABB":     ("ABB",    "NYSE",  "USD"),
}

SR_PERIOD               = 252
MIN_AVG_DOLLAR_VOLUME   = 20_000_000
MIN_AMPLITUDE           = 0.10
MAX_SUPPORT_DISTANCE    = 0.05
MIN_RESISTANCE_DIST     = 0.05
VOLUME_CONFIRM_MULT     = 1.5
VOLATILITY_ATR_PCT_THRESH = 2.0


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
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        with open(p, encoding="utf-8") as f:
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
        with open(p, encoding="utf-8") as f:
            lines = f.readlines()
        symbols = [l.strip() for l in lines
                   if l.strip() and not l.strip().startswith("#")]

    return [s.upper() for s in symbols]


# ─── Scoring rules ────────────────────────────────────────────────────────────

_OPS = {
    "eq":  operator.eq,
    "ne":  operator.ne,
    "lt":  operator.lt,
    "lte": operator.le,
    "gt":  operator.gt,
    "gte": operator.ge,
}


def load_scoring_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def calculate_score(m: dict, rules: dict) -> int:
    score = 0

    for rule in rules.get("rules", []):
        field  = rule["field"]
        op     = rule["op"]
        value  = rule["value"]
        points = rule["points"]

        field_val = m.get(field)
        if field_val is None:
            continue
        if isinstance(field_val, float) and np.isnan(field_val):
            continue

        # Round floats to 4 decimal places before comparing to avoid floating
        # point precision mismatches (e.g. 0.020000000000000004 failing <= 0.02)
        if isinstance(field_val, float):
            field_val = round(field_val, 4)

        if _OPS[op](field_val, value):
            score += points

    comment = m.get("comment", "")
    for crule in rules.get("comment_rules", []):
        if crule["contains"].lower() in comment.lower():
            score += crule["points"]

    return score


# ─── IBKR data fetching ───────────────────────────────────────────────────────

def make_contract(symbol: str) -> Stock:
    ib_sym, exchange, currency = SYMBOL_MAP.get(symbol, (symbol, "SMART", "USD"))
    return Stock(ib_sym, exchange, currency)


def fetch_daily(ib: IB, symbol: str) -> pd.DataFrame:
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


def _find_pivot_lows(daily: pd.DataFrame, order: int = 5) -> list[float]:
    lows = daily["Low"].values
    pivots = []
    for i in range(order, len(lows) - order):
        if lows[i] == min(lows[i - order:i + order + 1]):
            pivots.append(float(lows[i]))
    return pivots


def _find_pivot_highs(daily: pd.DataFrame, order: int = 5) -> list[float]:
    highs = daily["High"].values
    pivots = []
    for i in range(order, len(highs) - order):
        if highs[i] == max(highs[i - order:i + order + 1]):
            pivots.append(float(highs[i]))
    return pivots


def _cluster_levels(levels: list[float], tolerance: float = 0.02) -> list[float]:
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters = [[sorted_levels[0]]]
    for lvl in sorted_levels[1:]:
        centroid = float(np.mean(clusters[-1]))
        if (lvl - centroid) / centroid <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    return [float(np.mean(c)) for c in clusters]


def calc_support_resistance(daily: pd.DataFrame, period: int) -> tuple[float, float]:
    window = daily.iloc[-period:]
    close = float(window["Close"].iloc[-1])

    pivot_lows  = _find_pivot_lows(window)
    pivot_highs = _find_pivot_highs(window)

    support_zones    = _cluster_levels(pivot_lows)
    resistance_zones = _cluster_levels(pivot_highs)

    supports_below = [s for s in support_zones if s <= close]
    support = max(supports_below) if supports_below else float(window["Low"].min())

    resistances_above = [r for r in resistance_zones if r >= close]
    resistance = min(resistances_above) if resistances_above else float(window["High"].max())

    if resistance <= support:
        resistance = float(window["High"].max())

    return support, resistance


def calc_rsi(daily: pd.DataFrame, n: int = 14) -> float:
    if len(daily) < n + 1:
        return np.nan
    delta    = daily["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])


def calc_52w_distances(daily: pd.DataFrame) -> tuple[float, float]:
    window   = daily.iloc[-252:] if len(daily) >= 252 else daily
    close    = float(window["Close"].iloc[-1])
    high_52w = float(window["High"].max())
    low_52w  = float(window["Low"].min())
    dist_high = (close - high_52w) / high_52w
    dist_low  = (close - low_52w) / low_52w
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


def is_two_strong_green_candles(daily: pd.DataFrame, lookback: int = 20) -> bool:
    if len(daily) < max(2, lookback):
        return False
    last2 = daily.iloc[-2:]
    if not (last2["Close"] > last2["Open"]).all():
        return False
    bodies      = (daily["Close"].iloc[-lookback:] - daily["Open"].iloc[-lookback:]).abs()
    median_body = float(bodies.median())
    if median_body == 0:
        return False
    for _, bar in last2.iterrows():
        body = bar["Close"] - bar["Open"]
        if body < median_body:
            return False
    return True


def is_rebound_pattern(daily: pd.DataFrame) -> bool:
    if len(daily) < 1:
        return False
    bar = daily.iloc[-1]
    o, h, l, c  = bar["Open"], bar["High"], bar["Low"], bar["Close"]
    body         = abs(c - o)
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
    if not any((m["rebound_pattern"], m["volume_confirmed"], m["two_strong_green"])):
        return False
    return True


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

def _resolve_contract_info(ib: IB, symbol: str) -> tuple[str, str]:
    """Return (company_name, sector) from IBKR contract details."""
    try:
        contract = make_contract(symbol)
        details  = ib.reqContractDetails(contract)
        if details:
            d = details[0]
            return (d.longName or ""), (d.industry or "")
    except Exception:
        pass
    return "", ""


def process_symbol(ib: IB, symbol: str, logger: logging.Logger,
                   scoring_rules: dict) -> dict | None:
    try:
        company_name, sector = _resolve_contract_info(ib, symbol)
        daily  = fetch_daily(ib, symbol)
        weekly = fetch_weekly(ib, symbol)

        if daily.empty or len(daily) < 22:
            logger.warning("%s: insufficient daily data (%d bars)", symbol, len(daily))
            return None
        if weekly.empty or len(weekly) < 201:
            logger.warning("%s: insufficient weekly data (%d bars)", symbol, len(weekly))
            return None

        close             = float(daily["Close"].iloc[-1])
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
            "near_support":             support_distance <= MAX_SUPPORT_DISTANCE,
            "near_resistance":          resistance_distance <= MIN_RESISTANCE_DIST,
            "amplitude":                amplitude,
            "amplitude_valid":          amplitude >= MIN_AMPLITUDE,
            "weekly_ma200":             ma200,
            "weekly_ma200_trending_up": ma200_up,
            "price_above_ma200":        close > ma200,
            "ma200_slope":              "UP" if ma200_up else "DOWN",
            "weeks_above_ma200":        weeks_above,
            "rsi14":                    rsi14,
            "dist_52w_high":            dist_52w_high,
            "avg_dollar_volume":        avg_dollar_volume,
            "rebound_pattern":          is_rebound_pattern(daily),
            "volume_confirmed":         is_volume_confirmed(daily),
            "two_strong_green":         is_two_strong_green_candles(daily),
            "volatility_flag":          volatility_flag,
            "liquidity_flag":           avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME,
        }

        passes  = apply_filters(m)
        comment = build_comment(m, passes)
        m["comment"] = comment
        score   = calculate_score(m, scoring_rules)

        return {
            "Symbol":                symbol,
            "CompanyName":           company_name,
            "Sector":                sector,
            "Close":                 round(close, 4),
            "Support":               round(support, 4),
            "Resistance":            round(resistance, 4),
            "SupportDistance":       round(support_distance, 4),
            "ResistanceDistance":    round(resistance_distance, 4),
            "Amplitude":             round(amplitude, 4),
            "Score":                 score,
            "CandidateFlag":         "TRUE" if passes else "FALSE",
            "WeeklyMA200":           round(ma200, 4),
            "PriceAboveMA200":       close > ma200,
            "WeeksAboveMA200":       weeks_above,
            "MA200Slope":            "UP" if ma200_up else "DOWN",
            "NearSupport":           support_distance <= MAX_SUPPORT_DISTANCE,
            "NearResistance":        resistance_distance <= MIN_RESISTANCE_DIST,
            "AmplitudeValid":        amplitude >= MIN_AMPLITUDE,
            "RSI14":                 round(rsi14, 2) if not np.isnan(rsi14) else None,
            "Dist52WkHigh":          round(dist_52w_high * 100, 2),
            "Dist52WkLow":           round(dist_52w_low * 100, 2),
            "TwoStrongGreenCandles": m["two_strong_green"],
            "ReboundPattern":        m["rebound_pattern"],
            "LatestVolume":          latest_volume,
            "AvgDailyVolume":        int(round(avg_volume_20d)),
            "VolumeConfirmed":       m["volume_confirmed"],
            "ATR14":                 round(atr14, 4) if not np.isnan(atr14) else None,
            "ATR_Pct":               round(atr_pct, 2) if not np.isnan(atr_pct) else None,
            "VolatilityFlag":        volatility_flag,
            "AvgDollarVolume":       int(round(avg_dollar_volume)),
            "LiquidityFlag":         avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME,
            "Comment":               comment,
        }

    except Exception as e:
        logger.error("%s: %s", symbol, e)
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    app_config_path = "config/config.yaml"
    market_mode = None
    if os.path.exists(app_config_path):
        with open(app_config_path, "r", encoding="utf-8") as fh:
            app_config = yaml.safe_load(fh) or {}
        market_mode = app_config.get("market_mode")

    scoring_default = (
        f"config/scoring_{market_mode}.yaml" if market_mode else SCORING_CONFIG
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",      default=IB_HOST)
    parser.add_argument("--port",      type=int, default=IB_PORT)
    parser.add_argument("--client-id", type=int, default=IB_CLIENT_ID)
    parser.add_argument("--watchlist", default=WATCHLIST_FILE)
    parser.add_argument("--output",    default=OUTPUT_DIR)
    parser.add_argument("--scoring",   default=scoring_default)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    log_path  = os.path.join(args.output, f"scan_{ts}.log")
    csv_path  = os.path.join(args.output, f"scan_{ts}.csv")
    xlsx_path = os.path.join(args.output, f"scan_{ts}.xlsx")

    logger = setup_logging(log_path)
    start_time = datetime.now()
    logger.info("Scanner started")
    logger.info("Active scoring profile: %s", market_mode or "default")

    scoring_rules = load_scoring_rules(args.scoring)
    logger.info("Scoring rules loaded from %s", args.scoring)

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
    except Exception as exc:
        logger.error("Failed to connect to IBKR at %s:%s — %s", args.host, args.port, exc)
        sys.sys.sys.sys.exit(1)
    logger.info("Connected to IBKR at %s:%s (clientId=%s)", args.host, args.port, args.client_id)

    symbols = load_watchlist(args.watchlist)
    logger.info("Symbols to scan: %d", len(symbols))

    results = []
    errors  = 0

    for symbol in symbols:
        logger.info("Processing: %s", symbol)
        row = process_symbol(ib, symbol, logger, scoring_rules)
        if row is not None:
            results.append(row)
        else:
            errors += 1

    ib.disconnect()
    logger.info("Disconnected from IBKR")

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    logger.info("CSV written:  %s", csv_path)

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
