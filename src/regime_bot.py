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
from ib_insync import IB, Index, Stock, util

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


def fetch_close(ib: IB, contract, duration: str) -> list[float]:
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
    return [b.close for b in bars]


def fetch_indicators(ib: IB, cfg: dict, logger: logging.Logger) -> dict:
    ind      = cfg["indicators"]
    spy_ma   = int(ind["spy_ma_period"])
    qqq_ma   = int(ind["qqq_ma_period"])
    duration = f"{int(ind['lookback_days'])} D"

    spy_contract = Stock(ind["spy_ticker"], "SMART", "USD")
    qqq_contract = Stock(ind["qqq_ticker"], "SMART", "USD")
    vix_contract = Index(ind["vix_ticker"], "CBOE")

    logger.info("Fetching SPY daily data from TWS (%s)", duration)
    spy_closes = fetch_close(ib, spy_contract, duration)

    logger.info("Fetching QQQ daily data from TWS (%s)", duration)
    qqq_closes = fetch_close(ib, qqq_contract, duration)

    logger.info("Fetching VIX daily data from TWS (%s)", duration)
    vix_closes = fetch_close(ib, vix_contract, duration)

    if len(spy_closes) < spy_ma:
        raise ValueError(f"Not enough SPY data for MA{spy_ma} ({len(spy_closes)} bars)")
    if len(qqq_closes) < qqq_ma:
        raise ValueError(f"Not enough QQQ data for MA{qqq_ma} ({len(qqq_closes)} bars)")

    spy_latest   = spy_closes[-1]
    qqq_latest   = qqq_closes[-1]
    vix_latest   = vix_closes[-1]
    spy_ma_val   = sum(spy_closes[-spy_ma:]) / spy_ma
    qqq_ma_val   = sum(qqq_closes[-qqq_ma:]) / qqq_ma
    spy_above_ma = spy_latest > spy_ma_val
    qqq_above_ma = qqq_latest > qqq_ma_val

    logger.info("SPY %.2f (MA%d: %.2f) — %s MA",
                spy_latest, spy_ma, spy_ma_val, "above" if spy_above_ma else "below")
    logger.info("QQQ %.2f (MA%d: %.2f) — %s MA",
                qqq_latest, qqq_ma, qqq_ma_val, "above" if qqq_above_ma else "below")
    logger.info("VIX %.2f", vix_latest)

    return {
        "spy_above_ma": spy_above_ma,
        "qqq_above_ma": qqq_above_ma,
        "vix": vix_latest,
    }


def determine_regime(cfg: dict, indicators: dict, logger: logging.Logger) -> str:
    regimes       = cfg["regimes"]
    defensive_cfg = regimes.get("defensive", {})
    bull_cfg      = regimes.get("bull", {})
    range_cfg     = regimes.get("range", {})
    fallback      = cfg.get("fallback_mode", "range")

    # 1. Defensive — VIX spike overrides everything
    vix_min = defensive_cfg.get("vix_min")
    if vix_min is not None and indicators["vix"] >= float(vix_min):
        logger.info("Regime: defensive (VIX %.2f >= %.2f)", indicators["vix"], vix_min)
        return "defensive"

    # 2. Bull — all configured conditions must pass
    bull_conditions = []
    if bull_cfg.get("spy_above_ma"):
        bull_conditions.append(indicators["spy_above_ma"])
    if bull_cfg.get("qqq_above_ma"):
        bull_conditions.append(indicators["qqq_above_ma"])
    vix_max = bull_cfg.get("vix_max")
    if vix_max is not None:
        bull_conditions.append(indicators["vix"] <= float(vix_max))

    if bull_conditions and all(bull_conditions):
        logger.info("Regime: bull (all bull conditions met)")
        return "bull"

    # 3. Range — explicitly detected via VIX band
    range_conditions = []
    r_vix_min = range_cfg.get("vix_min")
    r_vix_max = range_cfg.get("vix_max")
    if r_vix_min is not None:
        range_conditions.append(indicators["vix"] >= float(r_vix_min))
    if r_vix_max is not None:
        range_conditions.append(indicators["vix"] < float(r_vix_max))

    if range_conditions and all(range_conditions):
        logger.info("Regime: range (VIX %.2f within range band %.2f-%.2f)",
                    indicators["vix"], r_vix_min or 0, r_vix_max or 0)
        return "range"

    # 4. Fallback — no regime explicitly matched
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
            logger.info("market_mode updated: %s → %s", previous_mode, new_mode)
        finally:
            ib.disconnect()
            logger.info("Disconnected from TWS")

    except Exception as exc:
        logger.warning("Regime analysis failed — keeping previous mode '%s'. Error: %s",
                       previous_mode, exc)

    logger.info("Regime bot finished")


if __name__ == "__main__":
    main()
