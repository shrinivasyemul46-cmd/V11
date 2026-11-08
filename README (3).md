# 🏆 NSE TOP 5 — POSITIONAL POWER SCANNER — V11

Scans **all NSE cash (EQ-series) stocks** and shows **at most 5** — each with a ready plan (entry trigger, stop, targets, quantity).
It never forces a trade: 0-4 stocks on a weak day is a correct result.

## Files (all 4 in the GitHub repo root, exact names)
`app.py` · `engine.py` · `requirements.txt` · `README.md` — upload over the old ones, then **Reboot** the app.

## Universe: all NSE cash stocks (default)
Funnel, cheapest step first: **NSE EQ list (~2,000)** → daily prices (cached chunks of 50, progress bar, resumes after an
interruption) → drop short history / price below limit / illiquid (20-day turnover) using those prices → market cap only for the
liquid survivors (cached 24h) → detailed analysis. The screen shows the counts at every step. Sectors exist only for stocks that
are also in the Nifty 500 list; others are "Unknown" and are NOT excluded by the top-sector rule. If NSE's full list cannot be
downloaded the app falls back to Nifty 500 and says so. Nifty 500 / 200 / "My symbols" are still available in the Universe box.
First full scan can take several minutes (Yahoo is rate-limited); later scans are fast. Not yet timed on real data.

## Two strategies (top of the sidebar)
* **🏆 Auto Top 5** — best 5 across: base breakout (6-24 months), cup base, 63-day / 6-month / 52-week high breakout,
  weekly accumulation and the new **Quality Pullback**. Fundamental score ≥ 60.
* **🎯 My quality pullback (original rules)** — your uploaded scanner's logic: monthly RSI > 60, weekly RSI > 60, daily RSI
  pullback zone 39-42, MACD above zero (line + histogram) on daily/weekly/monthly, daily and weekly volume ≥ 2x, plus hard
  quality limits (market cap ≥ ₹500 Cr, ROE ≥ 15%, revenue growth ≥ 10%, earnings growth ≥ 10%, debt/equity ≤ 0.10).
  Missing fundamental data = NOT passed (listed for manual check). Banks/NBFCs skip the debt test.

## What is new in V11
* **Weekly + monthly engine:** RSI, MACD and weekly volume ratio from completed weeks/months only (an unfinished week or
  month is never used, nothing from the future leaks into a signal - checked by test). Every setup must have weekly AND
  monthly RSI ≥ 50 (base breakouts get 10 points of slack, because a long flat base keeps monthly RSI near 50).
* **Quality Pullback setup:** a dip inside a strong weekly+monthly trend: daily RSI inside the zone (within the last 3
  sessions) + a bullish bounce bar + daily momentum turning up, price above the rising 200-EMA. Daily RSI in the zone alone
  is NOT a buy signal; the bounce bar waits for price to stabilise.
* **Faster than the uploaded scanner:** chart gates run first on all stocks; Yahoo fundamentals are fetched only for the
  ranked shortlist (max 60 lookups, cached 24h) instead of for all 500 stocks.
* **Top 5 cap:** Simple mode never shows more than 5. Advanced lets you raise it.
* RSI fix: a stock with no down periods now reads RSI 100 (it used to read "missing").

## 🧪 Test it on REAL data (sidebar bottom → "Test data connection")
One tap (30-60 s) checks the real NSE / Yahoo path: Nifty 500 list, prices for 6 liquid stocks (rows, last candle date, split/bonus
jumps, zero volume), Nifty 500 index, 11 world/India indices, market cap, fundamentals, and runs the engine on those stocks
with your current settings. It shows PASS / WARN / FAIL per check, a rough full-scan download time, and the last 5 closes of
each stock so you can compare them with your broker chart. A copy-able text report is included - send it to get it interpreted.

## Everything from V10 is kept
Positional / Swing horizon switch, volume-check window, base-breakout + coiled "ready" list, MFI/CCI/RSI gates, market-cap /
price / circuit / liquidity gates, world-India regime, sector ranking, reward:risk room, position sizing, backtest of
3 entries × 2 exits, journal, CSV downloads, plain-language "why no trade today".

## Honest limits
* Your original rules together are very narrow (a strong weekly RSI > 60 AND a daily RSI 39-42 AND 2x daily AND 2x weekly
  volume AND all-timeframe MACD > 0): expect few or no stocks most days. Heavy volume during a pullback can also mean selling.
  The Backtest tab shows how often the Quality Pullback setup actually triggered and its Avg R.
* Free end-of-day Yahoo data (see "price data as of"); Yahoo fundamentals are patchy for small caps and are latest values.
  The Backtest tests chart rules only (no fundamentals, regime, sector or market-cap replay) on today's index members.
* Tested on synthetic data only so far, not yet on real NSE history. Score is not a probability. Paper-trade first.
* Educational tool, not investment advice.
