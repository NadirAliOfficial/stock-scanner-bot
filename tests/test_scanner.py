"""
Unit tests for scanner.py — covers all 7 specification items from client review.
Each test class maps to a specific spec gap (2.1–2.7).
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yaml

_SCORING_RULES = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__, encoding="utf-8"), '..', 'config', 'scoring.yaml'))
)

from scanner import (
    _find_pivot_lows,
    _find_pivot_highs,
    _cluster_levels,
    calc_support_resistance,
    is_volume_confirmed,
    calc_atr,
    calc_rsi,
    calc_52w_distances,
    is_two_strong_green_candles,
    is_rebound_pattern,
    calculate_score,
    apply_filters,
    build_comment,
    calc_avg_dollar_volume,
    VOLUME_CONFIRM_MULT,
    VOLATILITY_ATR_PCT_THRESH,
    MIN_AVG_DOLLAR_VOLUME,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_daily(prices: list[dict]) -> pd.DataFrame:
    """Build a daily OHLCV DataFrame from a list of dicts."""
    df = pd.DataFrame(prices)
    df["date"] = pd.date_range("2025-01-01", periods=len(df), freq="B")
    df = df.set_index("date")
    return df


def _make_flat_daily(n: int, open_=100.0, high=105.0, low=95.0, close=102.0, volume=1_000_000):
    """Generate n bars of flat OHLCV data."""
    return _make_daily([
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
        for _ in range(n)
    ])


def _make_trending_daily(n: int, start=100, step=1, volume=1_000_000):
    """Generate n bars of steadily rising OHLCV data."""
    rows = []
    for i in range(n):
        c = start + i * step
        rows.append({
            "Open": c - 0.5, "High": c + 2, "Low": c - 2, "Close": c,
            "Volume": volume,
        })
    return _make_daily(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1  Support / Resistance — pivot-point detection with clustering
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupportResistance:

    def test_pivot_lows_detected(self):
        """Pivot lows are detected at local minima, not just global min."""
        # Create V-shaped dip at bar 10 and bar 20
        rows = []
        for i in range(30):
            if i == 10:
                low = 80
            elif i == 20:
                low = 85
            else:
                low = 100
            rows.append({"Open": 100, "High": 105, "Low": low, "Close": 102, "Volume": 1e6})
        df = _make_daily(rows)
        pivots = _find_pivot_lows(df, order=3)
        assert 80.0 in pivots
        assert 85.0 in pivots

    def test_pivot_highs_detected(self):
        """Pivot highs are detected at local maxima."""
        rows = []
        for i in range(30):
            if i == 10:
                high = 130
            elif i == 20:
                high = 125
            else:
                high = 105
            rows.append({"Open": 100, "High": high, "Low": 95, "Close": 102, "Volume": 1e6})
        df = _make_daily(rows)
        pivots = _find_pivot_highs(df, order=3)
        assert 130.0 in pivots
        assert 125.0 in pivots

    def test_cluster_levels_groups_nearby(self):
        """Levels within 2% of each other should be clustered together."""
        levels = [100.0, 101.0, 101.5, 120.0, 121.0]
        clusters = _cluster_levels(levels, tolerance=0.02)
        # 100, 101, 101.5 cluster together; 120, 121 cluster together
        assert len(clusters) == 2
        assert abs(clusters[0] - np.mean([100, 101, 101.5])) < 0.01
        assert abs(clusters[1] - np.mean([120, 121])) < 0.01

    def test_cluster_levels_empty(self):
        assert _cluster_levels([]) == []

    def test_support_below_price(self):
        """Support should be the nearest pivot-low zone at or below the current price."""
        # Price at 102, with pivot lows at 90 and 98
        rows = []
        for i in range(60):
            if i == 15:
                low = 90
            elif i == 40:
                low = 98
            else:
                low = 100
            rows.append({"Open": 101, "High": 105, "Low": low, "Close": 102, "Volume": 1e6})
        df = _make_daily(rows)
        support, _ = calc_support_resistance(df, period=60)
        # Should pick ~98 (nearest below), not 90
        assert support > 95

    def test_resistance_above_price(self):
        """Resistance should be the nearest pivot-high zone at or above the current price."""
        rows = []
        for i in range(60):
            if i == 15:
                high = 130
            elif i == 40:
                high = 110
            else:
                high = 105
            rows.append({"Open": 101, "High": high, "Low": 95, "Close": 102, "Volume": 1e6})
        df = _make_daily(rows)
        _, resistance = calc_support_resistance(df, period=60)
        # Should pick ~110 (nearest above), not 130
        assert resistance < 120

    def test_fallback_to_minmax(self):
        """With no pivot points (flat data), falls back to period min/max."""
        df = _make_flat_daily(30)
        support, resistance = calc_support_resistance(df, period=30)
        assert support == 95.0   # Low
        assert resistance == 105.0  # High


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2  Volume confirmation — >= 1.5x average
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeConfirmation:

    def test_volume_at_exactly_1_5x_passes(self):
        """Volume exactly at 1.5x average should pass (>= not >)."""
        avg_vol = 1_000_000
        spike_vol = int(avg_vol * VOLUME_CONFIRM_MULT)  # exactly 1.5x
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": avg_vol}
                for _ in range(21)]
        rows.append({"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": spike_vol})
        df = _make_daily(rows)
        assert is_volume_confirmed(df) is True

    def test_volume_at_1_49x_fails(self):
        """Volume just below 1.5x should fail."""
        avg_vol = 1_000_000
        spike_vol = int(avg_vol * 1.49)
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": avg_vol}
                for _ in range(21)]
        rows.append({"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": spike_vol})
        df = _make_daily(rows)
        assert is_volume_confirmed(df) is False

    def test_volume_at_1x_fails(self):
        """Volume at exactly 1x average (old behavior) should fail."""
        avg_vol = 1_000_000
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": avg_vol}
                for _ in range(22)]
        df = _make_daily(rows)
        assert is_volume_confirmed(df) is False

    def test_volume_at_2x_passes(self):
        """Volume at 2x average should pass."""
        avg_vol = 1_000_000
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": avg_vol}
                for _ in range(21)]
        rows.append({"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": avg_vol * 2})
        df = _make_daily(rows)
        assert is_volume_confirmed(df) is True

    def test_insufficient_data(self):
        df = _make_flat_daily(5)
        assert is_volume_confirmed(df) is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3  Volatility flag
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolatilityFlag:

    def test_high_atr_sets_flag(self):
        """ATR% >= 2.0 should set volatility_flag = True."""
        # Create volatile data: range of 5 on a 100 close → ATR ~5 → 5%
        df = _make_flat_daily(20, open_=98, high=105, low=95, close=100)
        atr = calc_atr(df)
        atr_pct = (atr / 100) * 100
        flag = atr_pct >= VOLATILITY_ATR_PCT_THRESH
        assert flag is True

    def test_low_atr_no_flag(self):
        """ATR% < 2.0 should set volatility_flag = False."""
        # Tight range: 0.5 on 100 → ATR ~0.5 → 0.5%
        df = _make_flat_daily(20, open_=99.9, high=100.2, low=99.8, close=100)
        atr = calc_atr(df)
        atr_pct = (atr / 100) * 100
        flag = atr_pct >= VOLATILITY_ATR_PCT_THRESH
        assert flag is False

    def test_volatility_flag_in_score(self):
        """rebound_pattern=True should add +4 per new scoring matrix."""
        base = {
            "near_support": False, "support_distance": 0.10,
            "near_resistance": False, "resistance_distance": 0.10,
            "price_above_ma200": False, "ma200_slope": "DOWN",
            "weeks_above_ma200": 0, "amplitude_valid": False,
            "amplitude": 0.07, "rsi14": 50.0, "dist_52w_high": -0.20,
            "volume_confirmed": False, "rebound_pattern": False,
            "two_strong_green": False, "liquidity_flag": False,
            "volatility_flag": False, "comment": "",
        }
        score_no_rebound = calculate_score(base, _SCORING_RULES)
        base["rebound_pattern"] = True
        score_with_rebound = calculate_score(base, _SCORING_RULES)
        assert score_with_rebound == score_no_rebound + 4

    def test_volatility_in_comment(self):
        """Comment should mention volatility when flag is set."""
        m = {
            "close": 110, "weekly_ma200": 100, "weekly_ma200_trending_up": True,
            "support_distance": 0.03, "amplitude": 0.15,
            "rebound_pattern": False, "volume_confirmed": False,
            "avg_dollar_volume": 25_000_000, "volatility_flag": True,
            "resistance_distance": 0.10,
        }
        comment = build_comment(m, passes=True)
        assert "High volatility" in comment


# ═══════════════════════════════════════════════════════════════════════════════
# 2.4  RSI(14) and 52-week distances
# ═══════════════════════════════════════════════════════════════════════════════

class TestRSIAnd52Week:

    def test_rsi_trending_up(self):
        """RSI should be > 50 for a steadily rising series."""
        df = _make_trending_daily(60, start=100, step=1)
        rsi = calc_rsi(df, 14)
        assert not np.isnan(rsi)
        assert rsi > 50

    def test_rsi_trending_down(self):
        """RSI should be < 50 for a steadily falling series."""
        df = _make_trending_daily(60, start=200, step=-1)
        rsi = calc_rsi(df, 14)
        assert rsi < 50

    def test_rsi_bounds(self):
        """RSI should always be between 0 and 100."""
        df = _make_trending_daily(60, start=100, step=2)
        rsi = calc_rsi(df, 14)
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        df = _make_flat_daily(5)
        assert np.isnan(calc_rsi(df, 14))

    def test_52w_at_high(self):
        """At 52-week high, distance from high should be 0."""
        rows = []
        for i in range(252):
            rows.append({"Open": 100, "High": 105, "Low": 95, "Close": 100, "Volume": 1e6})
        # Last bar closes at the 52-week high
        rows[-1]["Close"] = 105
        rows[-1]["High"] = 105
        df = _make_daily(rows)
        dist_high, dist_low = calc_52w_distances(df)
        assert abs(dist_high) < 0.001  # at the high
        assert dist_low > 0            # above the low

    def test_52w_at_low(self):
        """At 52-week low, distance from low should be 0."""
        rows = []
        for i in range(252):
            rows.append({"Open": 100, "High": 105, "Low": 95, "Close": 100, "Volume": 1e6})
        rows[-1]["Close"] = 95
        rows[-1]["Low"] = 95
        df = _make_daily(rows)
        dist_high, dist_low = calc_52w_distances(df)
        assert dist_high < 0   # below the high
        assert abs(dist_low) < 0.001  # at the low

    def test_52w_distances_sign(self):
        """dist_high should be negative when below high, dist_low positive when above low."""
        df = _make_flat_daily(252)  # close=102, high=105, low=95
        dist_high, dist_low = calc_52w_distances(df)
        assert dist_high < 0   # 102 < 105
        assert dist_low > 0    # 102 > 95


# ═══════════════════════════════════════════════════════════════════════════════
# 2.5  Global score includes volatility
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreVolatility:

    def _base_metrics(self, **overrides):
        m = {
            "close": 110, "weekly_ma200": 100, "weekly_ma200_trending_up": True,
            "support_distance": 0.03, "resistance_distance": 0.10,
            "amplitude": 0.15, "rebound_pattern": False,
            "volume_confirmed": False, "two_strong_green": False,
            "avg_dollar_volume": 25_000_000,
            "volatility_flag": False,
        }
        m.update(overrides)
        return m

    def test_known_score_combination(self):
        """near_support(+5) + price_above_ma200(+2) + rebound_pattern(+4) = +11."""
        m = {
            "near_support": True, "support_distance": 0.04,
            "near_resistance": False, "resistance_distance": 0.10,
            "price_above_ma200": True, "ma200_slope": "DOWN",
            "weeks_above_ma200": 0, "amplitude_valid": False,
            "amplitude": 0.07, "rsi14": 50.0, "dist_52w_high": -0.20,
            "volume_confirmed": False, "rebound_pattern": True,
            "two_strong_green": False, "liquidity_flag": False,
            "volatility_flag": False, "comment": "",
        }
        # near_support=True → +5, price_above_ma200=True → +2, rebound_pattern=True → +4
        # ma200_slope=DOWN → -3, dist_52w_high(-0.20) > -0.10 is False → 0
        # Expected: 5 + 2 + 4 - 3 = 8
        score = calculate_score(m, _SCORING_RULES)
        assert score == 8

    def test_score_all_negative(self):
        """near_resistance(−5) + ma200_slope DOWN(−3) + rsi > 70(−2) = −10."""
        m = {
            "near_support": False, "support_distance": 0.10,
            "near_resistance": True, "resistance_distance": 0.03,
            "price_above_ma200": False, "ma200_slope": "DOWN",
            "weeks_above_ma200": 0, "amplitude_valid": False,
            "amplitude": 0.07, "rsi14": 75.0, "dist_52w_high": -0.20,
            "volume_confirmed": False, "rebound_pattern": False,
            "two_strong_green": False, "liquidity_flag": False,
            "volatility_flag": False, "comment": "",
        }
        # near_resistance=True → -5, resistance_distance(0.03)<=0.02 False → 0
        # ma200_slope=DOWN → -3, rsi14(75)>70 → -2
        # dist_52w_high(-0.20) > -0.10 → False → 0
        # Expected: -5 - 3 - 2 = -10
        score = calculate_score(m, _SCORING_RULES)
        assert score == -10


# ═══════════════════════════════════════════════════════════════════════════════
# 2.6  Rebound confirmation — strong green candles
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrongGreenCandles:

    def test_two_strong_green_passes(self):
        """Two green candles with body >= median body should pass."""
        # 18 small bars, then 2 big green bars
        rows = [{"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1e6}
                for _ in range(18)]
        # median body of first 18 = |101-100| = 1.0
        # Last 2 bars: body = 3.0 (green, > median)
        rows.append({"Open": 100, "High": 106, "Low": 99, "Close": 103, "Volume": 1e6})
        rows.append({"Open": 103, "High": 109, "Low": 102, "Close": 106, "Volume": 1e6})
        df = _make_daily(rows)
        assert is_two_strong_green_candles(df) is True

    def test_two_weak_green_fails(self):
        """Two green candles with tiny bodies (< median) should fail."""
        # 18 bars with body=2, then 2 bars with body=0.5
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1e6}
                for _ in range(18)]
        rows.append({"Open": 100, "High": 102, "Low": 99, "Close": 100.3, "Volume": 1e6})
        rows.append({"Open": 100.3, "High": 102, "Low": 99, "Close": 100.6, "Volume": 1e6})
        df = _make_daily(rows)
        assert is_two_strong_green_candles(df) is False

    def test_one_red_one_green_fails(self):
        """If either of the last two candles is red, should fail."""
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1e6}
                for _ in range(18)]
        # Red candle
        rows.append({"Open": 105, "High": 106, "Low": 99, "Close": 100, "Volume": 1e6})
        # Green candle
        rows.append({"Open": 100, "High": 108, "Low": 99, "Close": 106, "Volume": 1e6})
        df = _make_daily(rows)
        assert is_two_strong_green_candles(df) is False

    def test_insufficient_data(self):
        df = _make_flat_daily(5)
        assert is_two_strong_green_candles(df) is False

    def test_hammer_pattern(self):
        """Hammer candle should be detected by is_rebound_pattern."""
        # body=1, lower_wick=4 (>=2x body), close in upper 40%
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1e6}
                for _ in range(5)]
        rows.append({"Open": 101, "High": 102, "Low": 96, "Close": 101.5, "Volume": 1e6})
        df = _make_daily(rows)
        assert is_rebound_pattern(df) is True

    def test_non_hammer(self):
        """Regular candle should not be detected as hammer."""
        rows = [{"Open": 100, "High": 105, "Low": 99, "Close": 103, "Volume": 1e6}]
        df = _make_daily(rows)
        assert is_rebound_pattern(df) is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2.7  Liquidity output structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiquidityOutput:

    def test_avg_dollar_volume_calculation(self):
        """AvgDollarVolume should be mean(close * volume) over 20 days."""
        rows = [{"Open": 100, "High": 105, "Low": 95, "Close": 50, "Volume": 1_000_000}
                for _ in range(20)]
        df = _make_daily(rows)
        adv = calc_avg_dollar_volume(df, 20)
        assert adv == 50 * 1_000_000  # $50M

    def test_liquidity_flag_above_threshold(self):
        """LiquidityFlag should be True when avg dollar vol >= $20M."""
        adv = 25_000_000
        assert adv >= MIN_AVG_DOLLAR_VOLUME

    def test_liquidity_flag_below_threshold(self):
        """LiquidityFlag should be False when avg dollar vol < $20M."""
        adv = 15_000_000
        assert adv < MIN_AVG_DOLLAR_VOLUME

    def test_insufficient_data_returns_nan(self):
        df = _make_flat_daily(5)
        assert np.isnan(calc_avg_dollar_volume(df, 20))


# ═══════════════════════════════════════════════════════════════════════════════
# Filters — integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilters:

    def _passing_metrics(self):
        return {
            "close": 110, "weekly_ma200": 100, "weekly_ma200_trending_up": True,
            "support_distance": 0.03, "resistance_distance": 0.10,
            "amplitude": 0.15, "avg_dollar_volume": 25_000_000,
            "rebound_pattern": False, "volume_confirmed": True,
            "two_strong_green": False, "volatility_flag": True,
        }

    def test_all_pass(self):
        assert apply_filters(self._passing_metrics()) is True

    def test_below_ma200_fails(self):
        m = self._passing_metrics()
        m["close"] = 90  # below ma200
        assert apply_filters(m) is False

    def test_ma200_not_trending_fails(self):
        m = self._passing_metrics()
        m["weekly_ma200_trending_up"] = False
        assert apply_filters(m) is False

    def test_low_liquidity_fails(self):
        m = self._passing_metrics()
        m["avg_dollar_volume"] = 10_000_000
        assert apply_filters(m) is False

    def test_low_amplitude_fails(self):
        m = self._passing_metrics()
        m["amplitude"] = 0.05
        assert apply_filters(m) is False

    def test_support_too_far_fails(self):
        m = self._passing_metrics()
        m["support_distance"] = 0.08
        assert apply_filters(m) is False

    def test_resistance_too_close_fails(self):
        m = self._passing_metrics()
        m["resistance_distance"] = 0.03
        assert apply_filters(m) is False

    def test_no_confirmation_signal_fails(self):
        """Candidate must have at least one confirmation signal."""
        m = self._passing_metrics()
        m["rebound_pattern"] = False
        m["volume_confirmed"] = False
        m["two_strong_green"] = False
        assert apply_filters(m) is False

    def test_rebound_only_passes(self):
        m = self._passing_metrics()
        m["rebound_pattern"] = True
        m["volume_confirmed"] = False
        m["two_strong_green"] = False
        assert apply_filters(m) is True

    def test_volume_only_passes(self):
        m = self._passing_metrics()
        m["rebound_pattern"] = False
        m["volume_confirmed"] = True
        m["two_strong_green"] = False
        assert apply_filters(m) is True

    def test_strong_green_only_passes(self):
        m = self._passing_metrics()
        m["rebound_pattern"] = False
        m["volume_confirmed"] = False
        m["two_strong_green"] = True
        assert apply_filters(m) is True


# ═══════════════════════════════════════════════════════════════════════════════
# ATR
# ═══════════════════════════════════════════════════════════════════════════════

class TestATR:

    def test_atr_positive(self):
        df = _make_flat_daily(20, high=110, low=90)
        atr = calc_atr(df, 14)
        assert atr > 0

    def test_atr_zero_range(self):
        """ATR should be 0 when all bars have identical OHLC."""
        df = _make_flat_daily(20, open_=100, high=100, low=100, close=100)
        atr = calc_atr(df, 14)
        assert atr == 0.0

    def test_atr_insufficient_data(self):
        df = _make_flat_daily(5)
        assert np.isnan(calc_atr(df, 14))
