# Stock Scanner Bot

A daily post-market technical analysis scanner for US stocks and ETFs.
Connects to **Interactive Brokers (IB Gateway / TWS)**, evaluates a watchlist against technical filters with confirmation signals, scores each symbol, and exports results to CSV and Excel.

## Project Structure

```
stock_scanner/
├── src/
│   └── scanner.py          # Main scanner logic
├── tests/
│   └── test_scanner.py     # Unit tests (49 tests)
├── config/
│   └── watchlist.csv       # Symbol watchlist
├── docs/
│   └── Scan_Bot_Spec_Detailed_18-03-2026.pdf
├── output/                  # Generated scan results (gitignored)
├── requirements.txt
└── README.md
```

## Features

- Loads watchlist from **CSV, JSON, or TXT**
- Fetches **daily and weekly** OHLCV data via IBKR
- **Pivot-point support/resistance** detection with clustering (not simple min/max)
- **6 mandatory filters** including at least one confirmation signal
- **Score from 0–11** based on 7 technical criteria (including volatility)
- **RSI(14)**, 52-week high/low distances, ATR, volatility flag
- Timestamped **CSV + XLSX + log** on every run

## Filters (all must pass)

| # | Filter | Condition |
|---|--------|-----------|
| 1 | Weekly Trend | Close > Weekly MA200 and MA200 trending upward |
| 2 | Liquidity | Avg Dollar Volume (20d) >= $20M |
| 3 | Amplitude | (Resistance - Support) / Support >= 10% |
| 4 | Support Proximity | (Close - Support) / Support <= 5% |
| 5 | Resistance Distance | (Resistance - Close) / Close >= 5% |
| 6 | Confirmation Signal | At least one of: rebound pattern, volume >= 1.5x avg, or two strong green candles |

## Scoring Model (0–11)

| Criterion | Points |
|-----------|--------|
| Weekly trend confirmed | 0 or 2 |
| Support proximity <= 2% / <= 5% | 2 / 1 / 0 |
| Amplitude >= 20% / >= 10% | 2 / 1 / 0 |
| Rebound / bullish rejection pattern | 0 or 2 |
| Volume confirmation (>= 1.5x 20d avg) | 0 or 1 |
| Liquidity (avg dollar vol >= $20M) | 0 or 1 |
| Volatility flag (ATR% >= 2.0) | 0 or 1 |

## Output Columns

| Column | Description |
|--------|-------------|
| Symbol | Ticker |
| Close | Last daily close |
| Support | Nearest pivot-low support zone |
| Resistance | Nearest pivot-high resistance zone |
| SupportDistance | (Close - Support) / Support |
| ResistanceDistance | (Resistance - Close) / Close |
| Amplitude | (Resistance - Support) / Support |
| Score | 0–11 |
| CandidateFlag | TRUE / FALSE |
| WeeklyMA200 | 200-week moving average |
| PriceAboveMA200 | Boolean |
| WeeksAboveMA200 | Consecutive weeks above MA200 |
| MA200Slope | UP / DOWN |
| RSI14 | 14-day RSI (Wilder's) |
| Dist52WkHigh | % distance from 52-week high |
| Dist52WkLow | % distance from 52-week low |
| TwoStrongGreenCandles | Last 2 candles green with body >= median |
| ReboundPattern | Hammer / bullish rejection detected |
| LatestVolume | Most recent session volume |
| AvgDailyVolume | 20-day average volume |
| VolumeConfirmed | Latest volume >= 1.5x 20d average |
| ATR14 | 14-day Average True Range |
| ATR_Pct | ATR as % of close |
| VolatilityFlag | ATR% >= 2.0 |
| AvgDollarVolume | 20-day average dollar volume |
| LiquidityFlag | AvgDollarVolume >= $20M |
| Comment | Summary of pass/fail reasons |

## Setup

```bash
git clone https://github.com/NadirAliOfficial/stock-scanner-bot.git
cd stock-scanner-bot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Edit `config/watchlist.csv` with your symbols (one per row)
2. Make sure IB Gateway or TWS is running with API enabled
3. Run the scanner:

```bash
python src/scanner.py
```

4. Results are saved to `output/`:

```
output/
  scan_20260320_2155.csv
  scan_20260320_2155.xlsx
  scan_20260320_2155.log
```

### CLI Options

```bash
python src/scanner.py --host 127.0.0.1 --port 4001 --client-id 3 --watchlist config/watchlist.csv --output output
```

## Configuration

All thresholds are defined at the top of `src/scanner.py`:

```python
SR_PERIOD             = 252          # Support/resistance lookback (trading days)
MIN_AVG_DOLLAR_VOLUME = 20_000_000   # $20M
MIN_AMPLITUDE         = 0.10         # 10%
MAX_SUPPORT_DISTANCE  = 0.05         # 5%
MIN_RESISTANCE_DIST   = 0.05         # 5%
VOLUME_CONFIRM_MULT   = 1.5          # 1.5x average volume
VOLATILITY_ATR_PCT_THRESH = 2.0      # 2% ATR threshold
```

## IBKR Connection

| Setting | Default | Notes |
|---------|---------|-------|
| Host | 127.0.0.1 | localhost |
| Port | 4001 | IB Gateway live (4002 paper, 7496 TWS live, 7497 TWS paper) |
| Client ID | 3 | Must be unique across connected bots |

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

## Notes

- Only **fully closed candles** are used (daily and weekly)
- Symbols with insufficient history are skipped and logged
- The bot performs **analysis only** — no trade execution
- Foreign symbols require corresponding IBKR market data subscriptions
<!-- updated: 2026-05-03-04 -->
