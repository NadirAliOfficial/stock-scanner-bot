# Stock Scanner Bot

A daily post-market technical analysis scanner for US stocks and ETFs.
Runs after market close via **Interactive Brokers (IB Gateway)**, evaluates a predefined watchlist against technical filters, scores each symbol, and exports results to CSV and Excel.

---

## Features

- Loads watchlist from **CSV or JSON**
- Fetches **daily and weekly** OHLCV data via **IBKR** (IB Gateway / TWS)
- Applies **5 mandatory filters** — only fully qualified symbols proceed to scoring
- Computes a **score from 0 to 10** based on 6 technical criteria
- Exports a **timestamped CSV + XLSX** on every run (previous results never overwritten)
- Full **log file** per run with start/end time, symbols scanned, candidates found, and per-symbol errors
- Robust error handling — invalid or missing data is skipped and logged, execution continues

---

## Filters (all must pass)

| # | Filter | Condition |
|---|--------|-----------|
| 1 | Weekly Trend | Close > Weekly MA200 and MA200 trending upward |
| 2 | Liquidity | Avg Dollar Volume (20d) ≥ $20M |
| 3 | Amplitude | (Resistance − Support) / Support ≥ 10% |
| 4 | Support Proximity | (Close − Support) / Support ≤ 5% |
| 5 | Resistance Distance | (Resistance − Close) / Close ≥ 5% |

---

## Scoring Model (0–10)

| Criterion | Points |
|-----------|--------|
| Weekly trend confirmed | 0 or 2 |
| Support proximity ≤ 2% | 2 · ≤ 5% → 1 · > 5% → 0 |
| Amplitude ≥ 20% | 2 · ≥ 10% → 1 · < 10% → 0 |
| Rebound / bullish rejection pattern | 0 or 2 |
| Volume confirmation (latest > 20d avg) | 0 or 1 |
| Liquidity bonus (avg dollar vol ≥ $20M) | 0 or 1 |

---

## Output Columns

| Column | Description |
|--------|-------------|
| Symbol | Ticker |
| Close | Last daily close |
| Support | Lowest low over lookback period |
| Resistance | Highest high over lookback period |
| SupportDistance | (Close − Support) / Support |
| ResistanceDistance | (Resistance − Close) / Close |
| Amplitude | (Resistance − Support) / Support |
| Score | 0–10 |
| CandidateFlag | TRUE / FALSE |
| WeeklyMA200 | 200-week moving average |
| PriceAboveMA200 | Boolean |
| WeeksAboveMA200 | Consecutive weeks above MA200 |
| MA200Slope | UP / DOWN |
| NearSupport | SupportDistance ≤ 5% |
| NearResistance | ResistanceDistance ≤ 5% |
| AmplitudeValid | Amplitude ≥ 10% |
| TwoGreenCandles | Last 2 daily candles both bullish |
| ReboundPattern | Hammer / bullish rejection detected |
| LatestVolume | Most recent session volume |
| AvgVolume20D | 20-day average volume |
| VolumeConfirmed | Latest volume > 20d average |
| ATR14 | 14-day Average True Range |
| ATR_Pct | ATR as % of close |
| AvgDollarVolume20 | 20-day average dollar volume |
| LiquidityValid | AvgDollarVolume20 ≥ $20M |
| Comment | Summary of pass/fail reasons |

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd stock_scanner

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

1. Edit `watchlist.csv` — add your symbols (one per row, header `Symbol`)
2. Run the scanner:

```bash
python scanner.py
```

3. Results are written to the `output/` folder:

```
output/
  scan_20260318_2215.csv
  scan_20260318_2215.xlsx
  scan_20260318_2215.log
```

---

## Watchlist Format

**CSV** (recommended):
```csv
Symbol
AAPL
MSFT
NVDA
```

**JSON** (also supported):
```json
["AAPL", "MSFT", "NVDA"]
```

To use a different file, update `WATCHLIST_FILE` at the top of `scanner.py`.

---

## Configuration

All thresholds are defined at the top of `scanner.py`:

```python
SR_PERIOD             = 252    # Support/resistance lookback (trading days)
MIN_AVG_DOLLAR_VOLUME = 20_000_000
MIN_AMPLITUDE         = 0.10
MAX_SUPPORT_DISTANCE  = 0.05
MIN_RESISTANCE_DIST   = 0.05
```

---

## IBKR Configuration

Connection settings are defined at the top of `scanner.py`:

```python
IB_HOST      = "127.0.0.1"
IB_PORT      = 4001        # IB Gateway live (7497 for paper)
IB_CLIENT_ID = 3           # must be unique across all connected bots
```

IB Gateway or TWS must be running before executing the scanner.

---

## Notes

- Only **fully closed candles** are used (daily and weekly)
- Symbols with insufficient history (e.g. recent IPOs) are skipped and logged
- The bot performs **analysis only** — no trade execution
- Foreign exchange symbols (e.g. Swiss, German, London) require the corresponding IBKR market data subscription
