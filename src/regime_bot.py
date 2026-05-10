#!/usr/bin/env python3
"""
Market Regime Bot
Determines the active market regime (bull / defensive / range) using
configurable indicators fetched from TWS and updates market_mode in
config/config.yaml. Run this before scanner.py.
"""

import logging
import os
import tempfile
from datetime import datetime

import yaml
from ib_insync import IB, Index, Stock

REGIME_CONFIG = "config/regime.yaml"
APP_CONFIG    = "config/config.yaml"
LOG_DIR       = "output"


def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("regime_bot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"regime_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_current_mode(path: str) -> str:
    try:
        return load_yaml(path).get("market_mode", "range")
    except Exception:
        return "range"


def write_market_mode(path: str, mode: str) -> None:
    try:
        cfg = load_yaml(path)
    except Exception:
        cfg = {}
    cfg["market_mode"] = mode
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def fetch_bars(ib: IB, contract, duration: str) -> list:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        raise ValueError(f"No data returned for {contract.symbol}")
    return bars


def calc_atr(bars: list, period: int) -> float:
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        raise ValueError(f"Not enough data for ATR{period} ({len(trs)} bars)")
    return sum(trs[-period:]) / period


def calc_ma(values: list[float], period: int) -> float:
    if len(values) < period:
        raise ValueError(f"Not enough data for MA{period} ({len(values)} values)")
    return sum(values[-period:]) / period


def fetch_indicators(ib: IB, cfg: dict, logger: logging.Logger) -> dict:
    ind      = cfg["indicators"]
    spy_ma   = int(ind["spy_ma_period"])
    qqq_ma   = int(ind["qqq_ma_period"])
    atr_per  = int(ind.get("spy_atr_period", 14))
    rsp_ma   = int(ind.get("rsp_ma_period", 50))
    duration = f"{int(ind['lookback_days'])} D"

    spy_contract = Stock(ind["spy_ticker"], "SMART", "USD")
    qqq_contract = Stock(ind["qqq_ticker"], "SMART", "USD")
    vix_contract = Index(ind["vix_ticker"], "CBOE")
    rsp_contract = Stock(ind.get("rsp_ticker", "RSP"), "SMART", "USD")

    logger.info("Fetching SPY daily bars from TWS (%s)", duration)
    spy_bars   = fetch_bars(ib, spy_contract, duration)
    spy_closes = [b.close for b in spy_bars]

    logger.info("Fetching QQQ daily bars from TWS (%s)", duration)
    qqq_bars   = fetch_bars(ib, qqq_contract, duration)
    qqq_closes = [b.close for b in qqq_bars]

    logger.info("Fetching VIX daily bars from TWS (%s)", duration)
    vix_bars   = fetch_bars(ib, vix_contract, duration)
    vix_closes = [b.close for b in vix_bars]

    logger.info("Fetching RSP daily bars from TWS (%s)", duration)
    rsp_bars   = fetch_bars(ib, rsp_contract, duration)
    rsp_closes = [b.close for b in rsp_bars]

    # ── Core MA values ────────────────────────────────────────────────────────
    spy_latest      = spy_closes[-1]
    qqq_latest      = qqq_closes[-1]
    vix_latest      = vix_closes[-1]
    spy_ma_val      = calc_ma(spy_closes, spy_ma)
    qqq_ma_val      = calc_ma(qqq_closes, qqq_ma)
    spy_above_ma    = spy_latest > spy_ma_val
    qqq_above_ma    = qqq_latest > qqq_ma_val
    spy_ma_dist_pct = (spy_latest - spy_ma_val) / spy_ma_val * 100

    # ── ATR volatility ────────────────────────────────────────────────────────
    atr_val     = calc_atr(spy_bars, atr_per)
    atr_pct     = atr_val / spy_latest * 100

    # ── Market breadth (RSP above its MA) ────────────────────────────────────
    rsp_latest      = rsp_closes[-1]
    rsp_ma_val      = calc_ma(rsp_closes, rsp_ma)
    breadth_above_ma = rsp_latest > rsp_ma_val

    # ── Logs ─────────────────────────────────────────────────────────────────
    logger.info("SPY %.2f | MA%d: %.2f | dist: %+.2f%% | %s MA",
                spy_latest, spy_ma, spy_ma_val, spy_ma_dist_pct,
                "above" if spy_above_ma else "below")
    logger.info("QQQ %.2f | MA%d: %.2f | %s MA",
                qqq_latest, qqq_ma, qqq_ma_val,
                "above" if qqq_above_ma else "below")
    logger.info("VIX %.2f", vix_latest)
    logger.info("SPY ATR%d: %.4f | ATR%%: %.2f%%", atr_per, atr_val, atr_pct)
    logger.info("RSP %.2f | MA%d: %.2f | breadth: %s MA",
                rsp_latest, rsp_ma, rsp_ma_val,
                "above" if breadth_above_ma else "below")

    return {
        "spy_above_ma":     spy_above_ma,
        "qqq_above_ma":     qqq_above_ma,
        "vix":              vix_latest,
        "atr_pct":          atr_pct,
        "spy_ma_dist_pct":  spy_ma_dist_pct,
        "breadth_above_ma": breadth_above_ma,
    }


def determine_regime(cfg: dict, indicators: dict, logger: logging.Logger) -> str:
    regimes       = cfg["regimes"]
    defensive_cfg = regimes.get("defensive", {})
    bull_cfg      = regimes.get("bull", {})
    range_cfg     = regimes.get("range", {})
    fallback      = cfg.get("fallback_mode", "range")

    # ── 1. Defensive — VIX spike OR high ATR triggers (OR logic) ─────────────
    def_vix_min = defensive_cfg.get("vix_min")
    def_atr_min = defensive_cfg.get("atr_pct_min")

    vix_defensive = def_vix_min is not None and indicators["vix"] >= float(def_vix_min)
    atr_defensive = def_atr_min is not None and indicators["atr_pct"] >= float(def_atr_min)

    if vix_defensive:
        logger.info("Regime: defensive | trigger: VIX %.2f >= %.2f",
                    indicators["vix"], def_vix_min)
        return "defensive"
    if atr_defensive:
        logger.info("Regime: defensive | trigger: ATR%% %.2f%% >= %.2f%%",
                    indicators["atr_pct"], def_atr_min)
        return "defensive"

    # ── 2. Bull — all configured conditions must pass (AND logic) ────────────
    bull_pass = []
    bull_log  = []

    if bull_cfg.get("spy_above_ma"):
        ok = indicators["spy_above_ma"]
        bull_pass.append(ok)
        bull_log.append(f"SPY above MA200: {'yes' if ok else 'NO'}")

    if bull_cfg.get("qqq_above_ma"):
        ok = indicators["qqq_above_ma"]
        bull_pass.append(ok)
        bull_log.append(f"QQQ above MA200: {'yes' if ok else 'NO'}")

    vix_max = bull_cfg.get("vix_max")
    if vix_max is not None:
        ok = indicators["vix"] <= float(vix_max)
        bull_pass.append(ok)
        bull_log.append(f"VIX {indicators['vix']:.2f} <= {vix_max}: {'yes' if ok else 'NO'}")

    atr_max = bull_cfg.get("atr_pct_max")
    if atr_max is not None:
        ok = indicators["atr_pct"] <= float(atr_max)
        bull_pass.append(ok)
        bull_log.append(f"ATR% {indicators['atr_pct']:.2f}% <= {atr_max}%: {'yes' if ok else 'NO'}")

    if bull_cfg.get("breadth_above_ma"):
        ok = indicators["breadth_above_ma"]
        bull_pass.append(ok)
        bull_log.append(f"RSP above MA (breadth): {'yes' if ok else 'NO'}")

    dist_min = bull_cfg.get("spy_ma_distance_min")
    if dist_min is not None:
        ok = indicators["spy_ma_dist_pct"] >= float(dist_min)
        bull_pass.append(ok)
        bull_log.append(f"SPY dist from MA {indicators['spy_ma_dist_pct']:+.2f}% >= {dist_min}%: {'yes' if ok else 'NO'}")

    logger.info("Bull check | %s", " | ".join(bull_log) if bull_log else "no conditions")

    if bull_pass and all(bull_pass):
        logger.info("Regime: bull | all conditions met")
        return "bull"

    # ── 3. Range — explicitly detected via VIX band ───────────────────────────
    range_pass = []
    r_vix_min  = range_cfg.get("vix_min")
    r_vix_max  = range_cfg.get("vix_max")

    if r_vix_min is not None:
        range_pass.append(indicators["vix"] >= float(r_vix_min))
    if r_vix_max is not None:
        range_pass.append(indicators["vix"] < float(r_vix_max))

    if range_pass and all(range_pass):
        logger.info("Regime: range | VIX %.2f within band [%.2f, %.2f)",
                    indicators["vix"], r_vix_min or 0, r_vix_max or 0)
        return "range"

    # ── 4. Fallback ───────────────────────────────────────────────────────────
    logger.info("Regime: %s (fallback — no explicit regime matched)", fallback)
    return fallback


def main():
    logger = setup_logging()
    logger.info("Regime bot started")

    previous_mode = read_current_mode(APP_CONFIG)

    try:
        regime_cfg = load_yaml(REGIME_CONFIG)
        ibkr_cfg   = regime_cfg.get("ibkr", {})
        host       = ibkr_cfg.get("host", "127.0.0.1")
        port       = int(ibkr_cfg.get("port", 4002))
        client_id  = int(ibkr_cfg.get("client_id", 4))

        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=20)
        logger.info("Connected to TWS at %s:%s (clientId=%s)", host, port, client_id)

        try:
            indicators = fetch_indicators(ib, regime_cfg, logger)
            new_mode   = determine_regime(regime_cfg, indicators, logger)
            write_market_mode(APP_CONFIG, new_mode)
            logger.info("market_mode updated: %s -> %s", previous_mode, new_mode)
        finally:
            ib.disconnect()
            logger.info("Disconnected from TWS")

    except Exception as exc:
        logger.warning("Regime analysis failed — keeping previous mode '%s'. Error: %s",
                       previous_mode, exc)

    logger.info("Regime bot finished")


if __name__ == "__main__":
    main()
