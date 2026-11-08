"""SWING PRO V11 engine (pure pandas/numpy, no Streamlit).

normalize -> features (per bar) -> signals (score + gates + reward:risk) -> summarize / backtest
Top-down helpers: index_stats -> regime, sector_strength.  Fundamentals: fund_score().
V9 focus: long consolidation (6-24 month) breakouts + coiled-and-ready watchlist.
"""
import numpy as np
import pandas as pd

CANDLES = {"BullEngulf": "Bullish Engulfing", "Hammer": "Hammer", "BullPin": "Bullish Pin Bar",
           "Marubozu": "Bullish Marubozu", "InsideBreak": "Inside-Bar Breakout",
           "ThreeBull": "Three Bullish Candles"}
SETUPS = {"TL": "Trendline Breakout", "BO": "63D Breakout", "SW": "Liquidity Sweep", "PB": "EMA20 Pullback",
          "WA": "Weekly Accumulation", "CB": "Cup Base Breakout (long-term)",
          "CO": "Base Breakout (6-24 month consolidation)",
          "MP": "Quality Pullback (weekly+monthly strong)"}
BASE_WINDOWS = (126, 189, 252, 378, 504)          # ~6, 9, 12, 18, 24 months of trading days
GATES = {"G_setup": "No setup", "G_vol": "Volume low", "G_candle": "Weak candle", "G_trend": "Trend weak",
         "G_liq": "Low liquidity", "G_mkt": "Market weak", "G_risk": "Stop too wide", "G_score": "Score low",
         "G_rr": "Reward:Risk low", "G_circuit": "Frozen/no-range candle", "G_mcap": "Market cap too small",
         "G_price": "Price too low (penny-stock risk)", "G_mom": "Momentum weak (RSI/MFI/CCI)",
         "G_mtf": "Weekly/Monthly trend weak", "G_mpvol": "Pullback volume low"}
ENTRY_MODES = {"Next open / signal close": "open", "Buy-stop above signal high": "buystop",
               "Pullback to 50% of signal candle": "pullback"}
DEFAULTS = dict(min_score=70, min_vol=2.0, max_risk=8.0, min_turnover=10.0, use_mkt=True,
                min_rr=2.5, entry_mode="buystop", min_mcap_cr=1000.0, min_price=20.0,
                min_cup_months=36, max_cup_months=120, cup_min_depth=0.15, cb_max_ext=0.04, cb_min_vol=2.0,
                min_rsi=55.0, max_rsi=82.0, min_mfi=55.0, min_cci=60.0,
                max_base_range=0.40, base_max_drift=0.30, base_drift_frac=0.5, ceil_tol=0.05, base_min_pos=0.5, co_max_ext=0.06, coil_pct=0.05,
                vol_sessions=1, mtf_min_rsi=50.0,
                mp_m_rsi=60.0, mp_w_rsi=60.0, mp_rsi_lo=38.0, mp_rsi_hi=50.0, mp_daily_macd="improving",
                mp_need_bounce=True, mp_vol_d=1.0, mp_vol_w=1.0, mp_hist_strict=False, mp_min_rr=1.0,
                allowed=tuple(SETUPS))
GLOBAL = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "Nikkei 225": "^N225",
          "Hang Seng": "^HSI", "Shanghai": "000001.SS", "DAX": "^GDAXI", "FTSE 100": "^FTSE"}
INDIA = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "India VIX": "^INDIAVIX"}
ALIASES = {"date": "Date", "datetime": "Date", "timestamp": "Date", "open": "Open", "high": "High",
           "low": "Low", "close": "Close", "ltp": "Close", "price": "Close", "volume": "Volume", "vol": "Volume"}
NEED = ["Open", "High", "Low", "Close", "Volume"]


# ---------------- basic indicators ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / ad                                           # no losses at all -> inf -> RSI 100; 0/0 stays NaN
    return 100 - 100 / (1 + rs)


def macd(s):
    line = ema(s, 12) - ema(s, 26)
    sig = ema(line, 9)
    return line, sig, line - sig


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def mfi(df, n=14):
    """Money Flow Index: volume-weighted RSI (0-100). High = money flowing in."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    flow, d = tp * df["Volume"], tp.diff()
    pos = flow.where(d > 0, 0.0).rolling(n).sum()
    neg = flow.where(d < 0, 0.0).rolling(n).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pos / neg                                        # neg == 0 -> inf -> MFI 100; 0/0 stays NaN
    return 100 - 100 / (1 + ratio)


def cci(df, n=20):
    """Commodity Channel Index, vectorised mean-absolute-deviation (no slow rolling.apply)."""
    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy()
    out = np.full(len(tp), np.nan)
    if len(tp) >= n:
        w = np.lib.stride_tricks.sliding_window_view(tp, n)
        mean = w.mean(axis=1)
        mad = np.abs(w - mean[:, None]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = (tp[n - 1:] - mean) / (0.015 * np.where(mad == 0, np.nan, mad))
    return pd.Series(out, index=df.index)


# ---------------- data cleaning ----------------
def _strip_tz(s):
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def normalize(raw):
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: ALIASES[c] for c in df.columns if c in ALIASES})
    miss = [c for c in NEED if c not in df.columns]
    if miss:
        raise ValueError("Missing columns: " + ", ".join(miss))
    if "Date" in df.columns:
        df["Date"] = _strip_tz(pd.to_datetime(df["Date"], errors="coerce"))
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    for c in NEED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in ["Date"] + NEED if c in df.columns]
    return df[keep].dropna(subset=NEED).reset_index(drop=True)


def drop_live_candle(df):
    """Before ~4 PM IST today's candle is incomplete: drop it so signals use closed candles only."""
    if "Date" not in df or df.empty:
        return df
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    if now.weekday() < 5 and now.hour < 16 and df["Date"].iloc[-1].date() == now.date():
        return df.iloc[:-1].reset_index(drop=True)
    return df


def prep_index(idx):
    """Index data -> market regime per day (close > EMA50) and 60-day momentum (for relative strength)."""
    d = idx.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    d = d.rename(columns={"adj close": "close"})[["date", "close"]]
    d["date"] = _strip_tz(pd.to_datetime(d["date"], errors="coerce"))
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    c = d.dropna().drop_duplicates("date").sort_values("date").set_index("date")["close"]
    return pd.DataFrame({"MktOK": (c > ema(c, 50)).astype(float), "IdxMom60": c.pct_change(60)})


# ---------------- trendline + resistance ----------------
def trendline_break(high, close, t, look=120, k=4, gap=8, win=5, max_ext=0.08):
    """Falling trendline through two swing highs (data up to bar t only). True if price crossed above it
    within the last `win` bars, is still above it, and is not more than max_ext above the line."""
    s = max(0, t - look)
    hi, cl = high[s:t + 1], close[s:t + 1]
    n = len(hi)
    if n < 40:
        return False
    piv = [i for i in range(k, n - k) if hi[i] == hi[i - k:i + k + 1].max()]
    lim = n - 1 - win
    best = None
    for ai, a in enumerate(piv):
        for b in piv[ai + 1:]:
            if b - a < gap or hi[b] >= hi[a] or b > lim:
                continue
            slope = (hi[b] - hi[a]) / (b - a)
            line = hi[a] + slope * (np.arange(a, lim + 1) - a)
            viol = int((cl[a:lim + 1] > line * 1.01).sum())
            if viol <= 1 and (best is None or (viol, -(b - a)) < best[0]):
                best = ((viol, -(b - a)), a, slope)
    if best is None:
        return False
    _, a, slope = best
    xs = np.arange(n - 1 - win, n)
    line = hi[a] + slope * (xs - a)
    above = cl[xs] > line
    crossed = any(above[j] and not above[j - 1] for j in range(1, len(xs)))
    return bool(above[-1] and crossed and cl[-1] / line[-1] - 1 <= max_ext)


def weekly_accumulation(d, min_weeks=18):
    """Flags the last daily row of a COMPLETED week where volume accumulated steadily: this week's volume
    beat the volume 1,2,3 weeks and 1,2,3 months (approx 4/9/13 weeks) ago, AND price held or rose over the
    same 1-week and ~4-month (16-week) window. Needs a 'Date' column; returns all-False otherwise."""
    n = len(d)
    out = np.zeros(n, dtype=bool)
    if "Date" not in d.columns or n < 40:
        return out
    wk = d.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    if len(wk) < min_weeks:
        return out
    today = pd.Timestamp.now().normalize()
    if wk.index[-1].date() >= today.date() and len(wk) > 1:
        wk = wk.iloc[:-1]                                   # drop the still-forming week
    vol_ok = pd.concat([wk["Volume"] > wk["Volume"].shift(k) for k in (2, 3, 4, 5, 9, 13)], axis=1).all(axis=1)
    price_ok = (wk["Close"] >= wk["Close"].shift(1)) & (wk["Close"] >= wk["Close"].shift(16))
    hit_weeks = wk.index[(vol_ok & price_ok).fillna(False)]
    if not len(hit_weeks):
        return out
    dates = d["Date"].to_numpy()
    for w_end in hit_weeks.to_numpy():
        idx = np.flatnonzero(dates <= w_end)
        if len(idx):
            out[idx[-1]] = True                             # mark the last trading day of that completed week
    return out


def cup_breakout(d, p=None):
    """Long-term 'cup' breakout on MONTHLY bars: price makes a high (the cup lip / neckline), falls at least
    cup_min_depth below it, then climbs back and closes above that old high for the first time, within
    cb_max_ext of it (not already run away), on a strong bullish month with cb_min_vol+ volume vs the trailing
    6 months. The old high must be min_cup_months to max_cup_months back - old enough to be a real multi-year
    base, bounded so it stays relevant. Returns 5 arrays aligned to d's row index: CB (bool), CupNeckline,
    CupDepthPct, CupAgeYears, CupVolX (NaN where not applicable). Needs a 'Date' column; returns all-empty
    otherwise. This is a resistance-breakout detector, not a dip-buy: it flags the breakout month only, not a
    later retest of it - treat it as a trigger to watch, and consider waiting for a pullback toward the
    neckline before sizing up, since nothing here confirms that retest in advance.
    """
    p = {**DEFAULTS, **(p or {})}
    n = len(d)
    out = dict(CB=np.zeros(n, bool), CupNeckline=np.full(n, np.nan), CupDepthPct=np.full(n, np.nan),
               CupAgeYears=np.full(n, np.nan), CupVolX=np.full(n, np.nan))
    if "Date" not in d.columns or n < 40:
        return out
    m = d.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    today = pd.Timestamp.now().normalize()
    if len(m) and m.index[-1].year == today.year and m.index[-1].month == today.month and len(m) > 1:
        m = m.iloc[:-1]                                      # drop the still-forming month
    mn = len(m)
    min_c, max_c = int(p["min_cup_months"]), int(p["max_cup_months"])
    if mn <= min_c:
        return out
    hi, lo, cl, op, vol = (m[c].to_numpy() for c in ("High", "Low", "Close", "Open", "Volume"))
    hits = []
    for i in range(min_c, mn):
        lo_b, hi_b = max(0, i - max_c), i - min_c
        if hi_b < lo_b:
            continue
        w = hi[lo_b:hi_b + 1]
        neckline = w.max()
        neck_pos = lo_b + int(w.argmax())
        if neckline <= 0 or cl[i - 1] > neckline:
            continue                                         # already broken out earlier - not fresh
        if not (neckline < cl[i] <= neckline * (1 + p["cb_max_ext"])):
            continue                                         # not a fresh breakout within the entry filter
        trough = cl[neck_pos:i].min() if i > neck_pos else neckline
        depth = (neckline - trough) / neckline
        if depth < p["cup_min_depth"]:
            continue                                         # too shallow to be a real base, not noise
        rng = hi[i] - lo[i]
        strong_candle = (cl[i] > op[i]) and (rng <= 0 or (cl[i] - lo[i]) / rng >= 0.55)
        if not strong_candle:
            continue
        prior = vol[max(0, i - 6):i]
        vol_x = vol[i] / prior.mean() if len(prior) and prior.mean() > 0 else np.nan
        if not (pd.notna(vol_x) and vol_x >= p["cb_min_vol"]):
            continue
        hits.append((i, neckline, depth, (i - neck_pos) / 12.0, vol_x))
    if not hits:
        return out
    dates = d["Date"].to_numpy()
    for i, neckline, depth, age, vol_x in hits:
        m_end = m.index[i].to_numpy()
        idx = np.flatnonzero(dates <= m_end)
        if len(idx):
            j = idx[-1]
            out["CB"][j], out["CupNeckline"][j] = True, neckline
            out["CupDepthPct"][j], out["CupAgeYears"][j], out["CupVolX"][j] = depth * 100, age, vol_x
    return out


def mtf_features(d):
    """Weekly (W-FRI) and monthly (ME) RSI / MACD / volume-ratio, mapped onto each DAILY row using only periods that
    were already COMPLETE on that date (period label date <= row date), so nothing from an unfinished week/month and
    nothing from the future leaks into a signal. Early bars where an indicator has not warmed up are NaN (not guessed).
    Columns: W_RSI, W_MACD, W_MACDH, W_VolX, M_RSI, M_MACD, M_MACDH."""
    cols = ["W_RSI", "W_MACD", "W_MACDH", "W_VolX", "M_RSI", "M_MACD", "M_MACDH"]
    out = pd.DataFrame(np.nan, index=d.index, columns=cols)
    if "Date" not in d.columns or len(d) < 120:
        return out
    base = d.set_index("Date")[["Close", "Volume"]]
    dates = pd.DatetimeIndex(d["Date"])
    for pre, rule in (("W", "W-FRI"), ("M", "ME")):
        bars = base.resample(rule).agg({"Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
        if len(bars) < 20:
            continue
        c = bars["Close"]
        line, _, hist = macd(c)
        r = rsi(c)
        r.iloc[:15] = np.nan                                  # RSI(14) needs warm-up
        line.iloc[:26], hist.iloc[:26] = np.nan, np.nan       # slow EMA(26) needs warm-up
        feat = pd.DataFrame({f"{pre}_RSI": r, f"{pre}_MACD": line, f"{pre}_MACDH": hist})
        if pre == "W":
            feat["W_VolX"] = bars["Volume"] / bars["Volume"].rolling(20).mean().shift(1)
        mapped = feat.reindex(dates, method="ffill")
        for col in feat.columns:
            out[col] = mapped[col].to_numpy()
    return out


def _base_setup(d, p):
    """Long-consolidation breakout / coil detector built from the BH{L}/BL{L} columns made in features().
    A 'base' = the last L trading days (L in BASE_WINDOWS ~ 6..24 months) ENDING YESTERDAY stayed inside a band
    no wider than max_base_range * (L/252)^0.25 (a little extra room for longer bases), was FLAT (net price change over the
    window no more than base_drift_frac of the band width and within base_max_drift, and the ceiling already reached in the OLDER half of the window within
    ceil_tol - so a steady uptrend that merely keeps making new highs is NOT mistaken for a base) and the previous
    close sat in the upper part of the band (base_min_pos) - coiling near the top, not sagging at the bottom.
      CO     = today's close is the first close above the base high, within co_max_ext of it (not extended).
      Coiled = still below the base high but within coil_pct of it (ready, not yet triggered).
    When several windows qualify the LONGEST one is reported. Returns arrays aligned to d."""
    n = len(d)
    c = d["Close"].to_numpy()
    pc = d["Close"].shift(1).to_numpy()
    co, coil = np.zeros(n, bool), np.zeros(n, bool)
    co_len, coil_len = np.zeros(n), np.zeros(n)
    co_hi, co_lo = np.full(n, np.nan), np.full(n, np.nan)
    cl_hi, cl_lo = np.full(n, np.nan), np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for L in BASE_WINDOWS:                                # ascending: a longer valid base overwrites a shorter
            hi, lo = d[f"BH{L}"].to_numpy(), d[f"BL{L}"].to_numpy()
            c0 = d["Close"].shift(L + 1).to_numpy()                # close at the START of the base window
            width = (hi - lo) / lo
            pos = (pc - lo) / (hi - lo)
            drift = np.abs(pc / c0 - 1)                           # flat base: start and end prices are close
            ceiling = d[f"BO{L}"].to_numpy() >= hi * (1 - p["ceil_tol"])   # ceiling already set in the older half
            ok = ((width <= p["max_base_range"] * (L / 252.0) ** 0.25) & (pos >= p["base_min_pos"])
                  & (drift <= p["base_max_drift"]) & (drift <= p["base_drift_frac"] * width) & ceiling)
            brk = ok & (c > hi) & (c <= hi * (1 + p["co_max_ext"]))
            cl = ok & (c <= hi) & (c >= hi * (1 - p["coil_pct"]))
            co_len, co_hi, co_lo = np.where(brk, L, co_len), np.where(brk, hi, co_hi), np.where(brk, lo, co_lo)
            coil_len, cl_hi, cl_lo = np.where(cl, L, coil_len), np.where(cl, hi, cl_hi), np.where(cl, lo, cl_lo)
            co |= brk
            coil |= cl
    # one signal per breakout: not if there was already a flagged breakout in the previous 5 sessions
    recent = pd.Series(co).shift(1, fill_value=False).rolling(5, min_periods=1).max().fillna(0).to_numpy() > 0
    co = co & ~recent
    coil = coil & ~co
    length = np.where(co, co_len, np.where(coil, coil_len, np.nan))
    hi = np.where(co, co_hi, np.where(coil, cl_hi, np.nan))
    lo = np.where(co, co_lo, np.where(coil, cl_lo, np.nan))
    return dict(CO=co, Coiled=coil, CoLen=length, CoHigh=hi, CoLow=lo,
                CoRangePct=np.where(np.isnan(hi), np.nan, (hi - lo) / lo * 100))


def nearest_resistance(high, t, entry, look=252, k=5):
    """Lowest confirmed swing high above `entry` in the last `look` bars (data up to t). NaN = blue sky."""
    hh = high[max(0, t - look):t + 1]
    n = len(hh)
    if n < 2 * k + 1:
        return np.nan
    mid = hh[k:n - k]
    piv = mid[mid == np.lib.stride_tricks.sliding_window_view(hh, 2 * k + 1).max(axis=1)]
    above = piv[piv > entry * 1.005]
    return float(above.min()) if len(above) else np.nan


# ---------------- per-bar features ----------------
def features(raw, idx=None, full_tl=False):
    d = raw.copy()
    o, h, l, c, v = (d[x] for x in NEED)
    d["EMA20"], d["EMA50"], d["EMA200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    d["RSI"] = rsi(c)
    d["MFI"], d["CCI"] = mfi(d), cci(d)
    _mt = mtf_features(d)
    for _c in _mt.columns:
        d[_c] = _mt[_c].to_numpy()
    d["MACD"], d["MACDSignal"], d["MACDHist"] = macd(c)
    d["ATR"] = atr(d)
    d["VolRatio"] = v / v.rolling(20).mean().shift(1).replace(0, np.nan)   # today excluded from average
    d["Turnover20"] = (c * v).rolling(20).mean()
    upv, dnv = v.where(c > c.shift(1), 0.0).rolling(50).sum(), v.where(c < c.shift(1), 0.0).rolling(50).sum()
    d["UpDnVol"] = upv / dnv.replace(0, np.nan)                 # >1 = more volume on up days (accumulation)
    d["Dryup"] = v.rolling(10).mean() / v.rolling(60).mean().replace(0, np.nan)   # <1 = volume drying up (coil)
    d["Tight20"] = (h.rolling(20).max() - l.rolling(20).min()) / c               # 20-day range as % of price

    rng = (h - l).replace(0, np.nan)
    bp = (c - o).abs() / rng
    up = (h - np.maximum(o, c)) / rng
    lw = (np.minimum(o, c) - l) / rng
    d["CloseLoc"] = (c - l) / rng
    pl20 = l.rolling(20).min().shift(1)
    supp = (l <= pl20 * 1.03) | ((l <= d["EMA50"] * 1.02) & (l >= d["EMA50"] * 0.97))
    po, pc = o.shift(1), c.shift(1)
    d["BullEngulf"] = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    d["Hammer"] = (lw >= 2 * bp) & (up <= 0.35) & (c > o) & supp
    d["BullPin"] = (lw >= 2.2 * bp) & (c > o) & (c >= l + 0.65 * rng) & supp
    d["Marubozu"] = (c > o) & (bp >= 0.70) & (up <= 0.15) & (lw <= 0.15)
    inside = (h < h.shift(1)) & (l > l.shift(1))
    d["InsideBreak"] = inside.shift(1, fill_value=False) & (c > h.shift(2))
    bull = c > o
    d["ThreeBull"] = (bull & bull.shift(1, fill_value=False) & bull.shift(2, fill_value=False)
                      & (c > pc) & (pc > c.shift(2)))

    hi63, lo63 = h.rolling(63).max().shift(1), l.rolling(63).min().shift(1)
    hi105, lo105 = h.rolling(105).max().shift(1), l.rolling(105).min().shift(1)
    d["Base3M"] = (hi63 - lo63) / pc <= 0.28
    d["Base5M"] = (hi105 - lo105) / pc <= 0.38
    bo = c > hi63
    d["BO"] = bo & ~bo.shift(1, fill_value=False)            # fresh breakout only
    d["SW"] = (l < pl20) & (c > pl20)
    d["WA"] = weekly_accumulation(d)
    cb = cup_breakout(d)
    for k, v in cb.items():
        d[k] = v
    trend_full = (d["EMA20"] > d["EMA50"]) & (c > d["EMA50"]) & (d["EMA50"] > d["EMA200"])
    d["PB"] = (trend_full & (l.rolling(3).min() <= d["EMA20"] * 1.01) & (c > d["EMA20"]) & (c > o)
               & d["RSI"].between(42, 65) & (c > pc))
    d["Ext"] = c / d["EMA20"] - 1
    for L in BASE_WINDOWS:                                       # base bands ending YESTERDAY (breakout bar excluded)
        d[f"BH{L}"] = h.rolling(L).max().shift(1)
        d[f"BL{L}"] = l.rolling(L).min().shift(1)
        d[f"BO{L}"] = h.rolling(L // 2).max().shift(L // 2 + 1)    # high of the OLDER half of the window
    d["Stop"] = l.rolling(3).min() - 0.1 * d["ATR"]           # structural stop: below last 3 lows

    d["MktOK"], d["RS"] = True, np.nan
    if idx is not None and "Date" in d.columns:
        r = idx.reindex(pd.DatetimeIndex(d["Date"]), method="ffill")
        d["MktOK"] = r["MktOK"].fillna(1.0).to_numpy() > 0.5
        d["RS"] = c.pct_change(60).to_numpy() - r["IdxMom60"].to_numpy()

    tl = np.zeros(len(d), dtype=bool)
    cand = ((d["VolRatio"] >= 1.5) & (c > o) & (c > d["EMA20"])).to_numpy()
    hv, cv = h.to_numpy(), c.to_numpy()
    for t in (range(len(d)) if full_tl else [len(d) - 1]):
        if t >= 60 and cand[t]:
            tl[t] = trendline_break(hv, cv, t)
    d["TL"] = tl
    return d


# ---------------- score + gates + reward:risk ----------------
def signals(d, p):
    d = d.copy()
    p = {**DEFAULTS, **p}
    # long-consolidation breakout / coil (needs the BH/BL columns from features())
    if f"BH{BASE_WINDOWS[0]}" in d.columns:
        for k, v in _base_setup(d, p).items():
            d[k] = v
    else:
        for k in ("CO", "Coiled"):
            d[k] = False
        for k in ("CoLen", "CoHigh", "CoLow", "CoRangePct"):
            d[k] = np.nan
    if "CO" not in set(p["allowed"]):
        d["Coiled"] = False
    # CO breakouts: failed-breakout stop = back inside the base (3% under the base high), never wider than 3-bar low
    d["Stop"] = np.where(d["CO"], np.maximum(d["Stop"], d["CoHigh"] * 0.97 - 0.1 * d["ATR"]), d["Stop"])
    for _c in ("W_RSI", "W_MACD", "W_MACDH", "W_VolX", "M_RSI", "M_MACD", "M_MACDH"):
        if _c not in d.columns:
            d[_c] = np.nan
    # Quality Pullback: strong weekly AND monthly momentum (RSI + MACD above zero and rising), daily RSI pulled back into
    # a zone, momentum turning up on the daily, and (optionally) a bullish bounce bar - a dip bought inside a strong trend.
    mtf_strong = (d["M_RSI"] > p["mp_m_rsi"]) & (d["W_RSI"] > p["mp_w_rsi"]) & (d["M_MACD"] > 0) & (d["W_MACD"] > 0)
    if p["mp_hist_strict"]:                              # also demand rising weekly + monthly MACD histograms
        mtf_strong = mtf_strong & (d["M_MACDH"] > 0) & (d["W_MACDH"] > 0)
    d_macd_ok = ((d["MACD"] > 0) & (d["MACDHist"] > 0)) if str(p["mp_daily_macd"]).startswith("strict") \
        else (d["MACDHist"] > d["MACDHist"].shift(1))
    bounce = ((d["Close"] > d["Open"]) & (d["Close"] > d["Close"].shift(1)) & (d["CloseLoc"] >= 0.5)) \
        if p["mp_need_bounce"] else True
    if p["mp_need_bounce"]:        # the dip was in the zone within the last 3 sessions; today's bounce confirms it
        in_zone = d["RSI"].rolling(3).min().between(p["mp_rsi_lo"], p["mp_rsi_hi"]) & (d["RSI"] <= p["mp_rsi_hi"] + 8)
    else:                           # literal rule: today's daily RSI is inside the zone
        in_zone = d["RSI"].between(p["mp_rsi_lo"], p["mp_rsi_hi"])
    d["MP"] = (mtf_strong & d_macd_ok & in_zone & bounce
               & (d["Close"] > d["EMA200"]) & (d["EMA50"] > d["EMA200"]))
    for k in SETUPS:                                        # only the setups the user allowed can fire
        if k not in set(p["allowed"]):
            d[k] = False

    vol = d["VolRatio"].fillna(0)
    candle_any = d[list(CANDLES)].any(axis=1)
    trend_full = (d["Close"] > d["EMA20"]) & (d["EMA20"] > d["EMA50"]) & (d["EMA50"] > d["EMA200"])
    trend_part = (d["Close"] > d["EMA20"]) & (d["Close"] > d["EMA50"])
    e200_ok = d["EMA200"] >= d["EMA200"].shift(20) * 0.99    # 200-EMA not falling
    macd_ok = (d["MACD"] > d["MACDSignal"]) & (d["MACDHist"] > d["MACDHist"].shift(1))

    score = (np.maximum.reduce([d["TL"]*30, d["BO"]*30, d["CB"]*30, d["CO"]*30, d["MP"]*28, d["SW"]*25, d["PB"]*22, d["WA"]*24])  # setup 30
             + np.where(vol >= 3, 13, np.where(vol >= 2, 10, np.where(vol >= 1.5, 5, 0)))  # volume   13
             + np.where(candle_any, 6, 0)                                                    # candle    6
             + np.where((d["W_RSI"] >= 55) & (d["M_RSI"] >= 55), 5,
                        np.where((d["W_RSI"] >= 55) | (d["M_RSI"] >= 55), 2, 0))             # weekly+monthly trend 5
             + np.where(d["CO"], 10, np.where(d["Base5M"], 10, np.where(d["Base3M"], 7, 0)))  # base    10
             + np.where(trend_full, 8, np.where(trend_part, 4, 0))                           # trend     8
             + np.where(d["RS"] > 0.05, 10, np.where(d["RS"] > 0, 5, 0))                     # RS       10
             + np.where(d["RSI"].between(50, 78), 3, 0)                                      # RSI       3
             + np.where(d["MFI"] >= p["min_mfi"], 3, 0)                                      # MFI       3
             + np.where(d["CCI"] >= p["min_cci"], 3, 0)                                      # CCI       3
             + np.where(macd_ok, 3, 0)                                                       # MACD      3
             + np.where(d["Ext"] <= 0.12, 4, 0)                                              # not stretched 4
             + np.where(d["UpDnVol"] >= 1.2, 2, 0))                                          # accumulation 2
    score = score + np.where(d["MP"], 12, 0)      # a pullback scores low on volume/momentum by nature; the strong weekly+monthly
    d["Score"] = np.minimum(score, 100).astype(int)   # trend the setup requires is the equivalent evidence

    # planned entry by mode -> risk is measured from the PLANNED entry (pullback entry = smaller risk)
    mode = p["entry_mode"]
    entry = d["High"] * 1.001 if mode == "buystop" else ((d["High"] + d["Low"]) / 2 if mode == "pullback" else d["Close"])
    d["PlanEntry"], d["PlanRisk"] = entry, entry - d["Stop"]
    d["RiskPct"] = d["PlanRisk"] / entry

    brk = d["TL"] | d["BO"] | d["CB"] | d["CO"]
    soft = d["SW"] | d["PB"] | d["WA"] | d["MP"]
    mp_only = d["MP"] & ~brk & ~(d["SW"] | d["PB"] | d["WA"])
    base_like = d["CO"] | d["CB"]       # long-base breakouts (they usually fire together with BO/WA): the EMA stack lags a
                                        # multi-month base, so they get the lenient trend test and a lower weekly/monthly floor
    tl_only = d["TL"] & ~(d["BO"] | d["CB"] | d["CO"] | soft)
    d["G_setup"] = brk | soft
    # volume: breakout needs the AVERAGE volume ratio of the last vol_sessions sessions >= min_vol (1 = breakout day
    # alone), and the breakout day itself must still be a real surge (>= 1.5x). Today is never in its own average.
    vs = max(1, int(p["vol_sessions"]))
    vavg = vol.rolling(vs, min_periods=vs).mean()
    d["G_vol"] = np.where(brk, (vavg >= p["min_vol"]) & (vol >= min(p["min_vol"], 1.5)), vol >= 1.0)
    d["G_candle"] = np.where(brk, (d["Close"] > d["Open"]) & (d["CloseLoc"] >= 0.6), candle_any | d["WA"] | d["MP"])
    d["G_trend"] = np.where(base_like, trend_part & (d["Close"] > d["EMA200"]) & e200_ok,
                            np.where(mp_only, (d["Close"] > d["EMA200"]) & (d["EMA50"] > d["EMA200"]) & e200_ok,
                                     np.where(tl_only, trend_part, trend_full)))
    # weekly + monthly trend alignment for every setup (set mtf_min_rsi to 0 to switch off)
    floor = np.where(base_like, p["mtf_min_rsi"] - 10, p["mtf_min_rsi"])   # a long flat base keeps monthly RSI near 50
    d["G_mtf"] = True if p["mtf_min_rsi"] <= 0 else ((d["W_RSI"] >= floor) & (d["M_RSI"] >= floor))
    d["G_mpvol"] = np.where(mp_only, (vol >= p["mp_vol_d"]) & (d["W_VolX"].fillna(0) >= p["mp_vol_w"]), True)
    d["G_mom"] = np.where(brk, (d["RSI"] >= p["min_rsi"]) & (d["RSI"] <= p["max_rsi"])
                          & (d["MFI"] >= p["min_mfi"]) & (d["CCI"] >= p["min_cci"]), True)
    d["G_liq"] = d["Turnover20"] >= p["min_turnover"] * 1e7
    d["G_mkt"] = d["MktOK"].astype(bool) if p["use_mkt"] else True
    d["G_risk"] = (d["RiskPct"] <= p["max_risk"] / 100) & (d["RiskPct"] >= 0.008)
    d["G_score"] = d["Score"] >= p["min_score"]
    d["G_circuit"] = (d["High"] - d["Low"]) > 0                            # not frozen at a circuit limit
    d["G_price"] = d["Close"] >= p.get("min_price", 0)
    # Market cap is a universe-level filter (no historical series available), not a per-bar one: the gate only
    # bites when the caller has actually attached a McapCr column (live scan); it is a pass-through otherwise
    # (CSV mode, backtest) so those paths are never silently zeroed out by a missing market-cap lookup.
    if "McapCr" in d.columns and p.get("min_mcap_cr", 0) > 0:
        d["G_mcap"] = d["McapCr"].fillna(-1) >= p["min_mcap_cr"]
    else:
        d["G_mcap"] = True
    base = [g for g in GATES if g != "G_rr"]
    d[base] = d[base].fillna(False).astype(bool)

    # "ready to move" watchlist: coiled just under a 6-24 month base high, quality gates passed, not yet triggered
    d["CoilReady"] = (d["Coiled"] & d["G_liq"] & d["G_mcap"] & d["G_price"] & d["G_circuit"]
                      & (d["Close"] > d["EMA50"]) & (d["Close"] > d["EMA200"]) & e200_ok & (d["RSI"] >= 45))

    # reward:risk room to the nearest overhead swing-high (only computed where everything else passes + last bar)
    n, hv, ev, rv = len(d), d["High"].to_numpy(), entry.to_numpy(), d["PlanRisk"].to_numpy()
    res, rr = np.full(n, np.nan), np.full(n, np.nan)
    for t in set(np.flatnonzero(d[base].all(axis=1).to_numpy()).tolist()) | {n - 1}:
        if rv[t] > 0:
            res[t] = nearest_resistance(hv, t, ev[t], look=504)
            rr[t] = 4.0 if np.isnan(res[t]) else (res[t] - ev[t]) / rv[t]     # blue sky counts as 4R of room
    d["Res"], d["RR_room"] = res, rr
    # a pullback inside an uptrend sits just under its own recent high, so that high is a target to retest, not a wall:
    # Quality Pullback uses the lower mp_min_rr room requirement instead of min_rr
    d["G_rr"] = np.where(mp_only, d["RR_room"] >= p["mp_min_rr"], d["RR_room"] >= p["min_rr"])
    d["Buy"] = d[list(GATES)].all(axis=1)
    return d


def _names(x, mapping):
    return [n for k, n in mapping.items() if bool(x[k])]


def with_mcap(d, mcap_cr):
    """Attach a market-cap-in-crore value (float or NaN) to every row, so signals()/summarize() can gate on it."""
    d = d.copy()
    d["McapCr"] = mcap_cr
    return d


def summarize(sym, sector, d, p=None):
    p = {**DEFAULTS, **(p or {})}
    x = d.iloc[-1]
    entry = float(x["PlanEntry"])
    stop = float(x["Stop"]) if pd.notna(x["Stop"]) else np.nan
    risk = entry - stop
    trig = {"open": f"Buy near ₹{entry:,.2f} (signal close)",
            "buystop": f"Buy-stop: buy only if price trades above ₹{entry:,.2f} (valid 3 sessions)",
            "pullback": f"Limit buy at ₹{entry:,.2f} (valid 3 sessions; skip if 2R runs without filling)"}[p["entry_mode"]]
    fails = [w for g, w in GATES.items() if not x[g]]
    mc = float(x["McapCr"]) if "McapCr" in d.columns and pd.notna(x["McapCr"]) else np.nan
    return {"Symbol": sym, "Sector": sector, "BUY": bool(x["Buy"]), "Score": int(x["Score"]), "McapCr": round(mc) if pd.notna(mc) else np.nan,
            "Setup": ", ".join(_names(x, SETUPS)) or "-", "Candle": ", ".join(_names(x, CANDLES)) or "-",
            "Vol_x": round(float(x["VolRatio"]), 1) if pd.notna(x["VolRatio"]) else 0.0,
            "RSI": round(float(x["RSI"]), 1), "Base": "5M+" if x["Base5M"] else ("3M+" if x["Base3M"] else "No"),
            "RS_vs_Index%": round(float(x["RS"]) * 100, 1) if pd.notna(x["RS"]) else np.nan,
            "Close": round(float(x["Close"]), 2), "Entry": round(entry, 2), "Stop": round(stop, 2),
            "Risk%": round(risk / entry * 100, 1),
            "T1_2R": round(entry + 2 * risk, 2), "T2_3R": round(entry + 3 * risk, 2),
            "Resistance": round(float(x["Res"]), 2) if pd.notna(x["Res"]) else np.nan,
            "RR_room": round(float(x["RR_room"]), 1) if pd.notna(x["RR_room"]) else np.nan,
            "Trigger": trig, "Why_not_BUY": "" if x["Buy"] else ", ".join(fails),
            "CupAgeYears": round(float(x["CupAgeYears"]), 1) if pd.notna(x.get("CupAgeYears")) else np.nan,
            "CupDepthPct": round(float(x["CupDepthPct"]), 0) if pd.notna(x.get("CupDepthPct")) else np.nan,
            "CupVolX": round(float(x["CupVolX"]), 1) if pd.notna(x.get("CupVolX")) else np.nan,
            "Classic_T1_10pct": round(entry * 1.10, 2), "Classic_T2_20pct": round(entry * 1.20, 2),
            "Classic_T3_30pct": round(entry * 1.30, 2),
            "BreakoutLevel": ("52-week high" if pd.notna(x.get("BH252")) and x["Close"] > x["BH252"] else
                              "6-month high" if pd.notna(x.get("BH126")) and x["Close"] > x["BH126"] else
                              "3-month high" if bool(x["BO"]) else "-"),
            "W_RSI": round(float(x["W_RSI"]), 0) if pd.notna(x.get("W_RSI")) else np.nan,
            "M_RSI": round(float(x["M_RSI"]), 0) if pd.notna(x.get("M_RSI")) else np.nan,
            "W_VolX": round(float(x["W_VolX"]), 1) if pd.notna(x.get("W_VolX")) else np.nan,
            "MACD_WM": ("W✓ " if pd.notna(x.get("W_MACDH")) and x["W_MACD"] > 0 and x["W_MACDH"] > 0 else "W✗ ")
                       + ("M✓" if pd.notna(x.get("M_MACDH")) and x["M_MACD"] > 0 and x["M_MACDH"] > 0 else "M✗"),
            "MFI": round(float(x["MFI"]), 0) if pd.notna(x.get("MFI")) else np.nan,
            "CCI": round(float(x["CCI"]), 0) if pd.notna(x.get("CCI")) else np.nan,
            "UpDnVol": round(float(x["UpDnVol"]), 2) if pd.notna(x.get("UpDnVol")) else np.nan,
            "Dryup": round(float(x["Dryup"]), 2) if pd.notna(x.get("Dryup")) else np.nan,
            "Tight20%": round(float(x["Tight20"]) * 100, 1) if pd.notna(x.get("Tight20")) else np.nan,
            "BaseMonths": round(float(x["CoLen"]) / 21, 0) if pd.notna(x.get("CoLen")) else np.nan,
            "BaseRange%": round(float(x["CoRangePct"]), 0) if pd.notna(x.get("CoRangePct")) else np.nan,
            "BaseHigh": round(float(x["CoHigh"]), 2) if pd.notna(x.get("CoHigh")) else np.nan,
            "BaseLow": round(float(x["CoLow"]), 2) if pd.notna(x.get("CoLow")) else np.nan,
            "BaseTarget": (round(float(x["CoHigh"] + (x["CoHigh"] - x["CoLow"])), 2) if pd.notna(x.get("CoHigh")) else np.nan),
            "Dist_to_base%": (round((float(x["CoHigh"]) / float(x["Close"]) - 1) * 100, 1) if pd.notna(x.get("CoHigh")) else np.nan),
            "BreakoutTrigger": (round(float(x["CoHigh"]) * 1.001, 2) if pd.notna(x.get("CoHigh")) else np.nan),
            "CoilReady": bool(x["CoilReady"]) if "CoilReady" in d.columns else False}


def size(entry, stop, capital, risk_pct, max_pos_pct):
    """Qty from risk, capped so one stock never exceeds max_pos_pct of capital."""
    per_share = entry - stop
    if per_share <= 0:
        return 0
    return int(max(0, min(np.floor(capital * risk_pct / 100 / per_share),
                          np.floor(capital * max_pos_pct / 100 / entry))))


# ---------------- fundamentals ----------------
def quality_pass(f, sector="", min_roe=15.0, min_rev=10.0, min_eps=10.0, max_de=0.10):
    """Hard fundamental screen using fixed limits (ROE %, revenue growth %, earnings growth %, debt/equity ratio).
    Missing data is NEVER a pass: returns (ok, failed, missing). Banks/NBFCs skip the debt test (debt is their raw
    material, so a low debt/equity limit is not comparable for them). f uses the same keys as fund_score()."""
    fin = any(k in str(sector).lower() for k in ("financ", "bank", "nbfc", "insurance"))
    need = [("ROE", "roe", lambda v: v >= min_roe), ("Revenue growth", "rg", lambda v: v >= min_rev),
            ("Earnings growth", "eg", lambda v: v >= min_eps)]
    if not fin:
        need.append(("Debt/Equity", "de", lambda v: v <= max_de))
    failed, missing = [], []
    for label, key, test in need:
        v = f.get(key)
        if v is None or pd.isna(v):
            missing.append(label)
        elif not test(v):
            failed.append(label)
    return (not failed and not missing), failed, missing


def fund_score(f, sector=""):
    """Transparent 0-100 fundamental score from Yahoo-style numbers. f keys (None = unknown): roe %, de (debt/equity
    ratio), eg (earnings growth %), rg (revenue growth %), pm (profit margin %), cr (current ratio), pe, ins (insider
    holding %, a rough promoter proxy). Points are only counted for data that exists; if less than 60% of the
    attainable points could be measured the result is NA (not a guess, not a pass). Banks/NBFCs skip debt and
    current-ratio (not meaningful for them). Returns dict(score, label, coverage, detail)."""
    fin = any(k in str(sector).lower() for k in ("financ", "bank", "nbfc", "insurance"))

    def tier(v, tiers):
        for th, pts in tiers:
            if v >= th:
                return pts
        return 0

    g = lambda k: f.get(k) if f.get(k) is not None and not pd.isna(f.get(k)) else None
    items = []                                                     # (label, earned, max, text)
    if g("roe") is not None:
        items.append(("ROE", tier(g("roe"), [(15, 20), (10, 10), (5, 4)]), 20, f"ROE {g('roe'):.0f}%"))
    if g("de") is not None and not fin:
        de = g("de")
        items.append(("D/E", 15 if de <= 0.5 else 8 if de <= 1.0 else 0, 15, f"D/E {de:.2f}"))
    if g("eg") is not None:
        items.append(("EPS growth", tier(g("eg"), [(15, 20), (5, 10), (0, 4)]), 20, f"EPS g {g('eg'):.0f}%"))
    if g("rg") is not None:
        items.append(("Sales growth", tier(g("rg"), [(10, 15), (5, 8), (0, 3)]), 15, f"Sales g {g('rg'):.0f}%"))
    if g("pm") is not None:
        items.append(("Margin", tier(g("pm"), [(10, 10), (5, 5)]), 10, f"Margin {g('pm'):.0f}%"))
    if g("cr") is not None and not fin:
        items.append(("Current ratio", 5 if g("cr") >= 1.2 else 2 if g("cr") >= 1.0 else 0, 5, f"CR {g('cr'):.1f}"))
    if g("pe") is not None and g("pe") > 0:
        pe = g("pe")
        items.append(("PE", 10 if pe <= 45 else 5 if pe <= 80 else 0, 10, f"PE {pe:.0f}"))
    if g("ins") is not None:
        items.append(("Insider hold", tier(g("ins"), [(40, 5), (25, 3)]), 5, f"Insider {g('ins'):.0f}%"))
    avail, earned = sum(i[2] for i in items), sum(i[1] for i in items)
    attainable = 80 if fin else 100
    coverage = avail / attainable
    detail = " · ".join(i[3] for i in items) or "no data"
    if coverage < 0.6 or avail == 0:
        return dict(score=None, label="NA", coverage=round(coverage, 2), detail=detail)
    score = int(round(100 * earned / avail))
    label = "STRONG" if score >= 75 else "GOOD" if score >= 60 else "AVERAGE" if score >= 45 else "WEAK"
    return dict(score=score, label=label, coverage=round(coverage, 2), detail=detail)


# ---------------- top-down: global / India regime, sectors ----------------
def index_stats(df):
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(c) < 60:
        return None
    e200 = ema(c, 200).iloc[-1] if len(c) >= 200 else np.nan
    return {"Last": round(float(c.iloc[-1]), 2), "1D%": round((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
            "5D%": round((c.iloc[-1] / c.iloc[-6] - 1) * 100, 2), "20D%": round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2),
            "Above50": bool(c.iloc[-1] > ema(c, 50).iloc[-1]),
            "Above200": bool(c.iloc[-1] > e200) if pd.notna(e200) else None}


def regime(stats):
    """0-100 risk appetite from world markets (40), Nifty/BankNifty trend (45) and India VIX (15).
    >=65 RISK-ON (full risk) | 45-65 NEUTRAL (half risk, max 3 trades) | <45 RISK-OFF (no new trades)."""
    g = [s for k, s in stats.items() if k in GLOBAL and s]
    pts_g = 40 * (0.625 * np.mean([s["Above50"] for s in g]) + 0.375 * np.mean([s["5D%"] > 0 for s in g])) if g else 20
    n, b, v = stats.get("Nifty 50"), stats.get("Bank Nifty"), stats.get("India VIX")
    pts_i = 22.5
    if n:
        pts_i = 15 * n["Above50"] + 10 * bool(n["Above200"]) + 10 * (n["20D%"] > 0) + (10 * b["Above50"] if b else 5)
    pts_v = 8 if not v else (15 if v["Last"] <= 15 else 8 if v["Last"] <= 20 else 3 if v["Last"] <= 25 else 0)
    score = int(round(pts_g + pts_i + pts_v))
    label, mult = ("RISK-ON", 1.0) if score >= 65 else (("NEUTRAL", 0.5) if score >= 45 else ("RISK-OFF", 0.0))
    return {"score": score, "label": label, "mult": mult, "parts": (round(pts_g), round(pts_i), round(pts_v))}


def sector_strength(feats, sectors, min_n=4):
    """Rank industry groups from their own stocks: 60D/20D median return + % of stocks above EMA50 and EMA200.
    Groups with fewer than min_n stocks are ignored (a 2-stock 'sector' is noise)."""
    rows = []
    for sym, d in feats.items():
        sec = sectors.get(sym, "Unknown")
        if sec == "Unknown" or len(d) < 62:
            continue
        x, c = d.iloc[-1], d["Close"]
        rows.append((sec, c.iloc[-1] / c.iloc[-21] - 1, c.iloc[-1] / c.iloc[-61] - 1, bool(x["Close"] > x["EMA50"]),
                     bool(x["Close"] > x["EMA200"])))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Sector", "R20", "R60", "A50", "A200"])
    g = df.groupby("Sector").agg(Stocks=("R20", "size"), R20=("R20", "median"), R60=("R60", "median"),
                                 Breadth=("A50", "mean"), Breadth200=("A200", "mean"))
    g = g[g["Stocks"] >= min_n].copy()
    if g.empty:
        return pd.DataFrame()
    g["Score"] = (0.35 * g["R60"].rank(pct=True) * 100 + 0.25 * g["R20"].rank(pct=True) * 100
                  + 0.2 * g["Breadth"] * 100 + 0.2 * g["Breadth200"] * 100).round(1)
    g["R20"], g["R60"] = (g["R20"] * 100).round(1), (g["R60"] * 100).round(1)
    g["Breadth"], g["Breadth200"] = (g["Breadth"] * 100).round(0), (g["Breadth200"] * 100).round(0)
    return g.sort_values("Score", ascending=False).reset_index()


# ---------------- backtest (entry mode x exit style) ----------------
def _fill(mode, i, o, h, l, stop, n, valid, rr):
    """Return (bar, price) of the fill or None. All rules use only data after the signal bar i."""
    if mode == "open":
        return (i + 1, o[i + 1]) if i + 1 < n else None
    lvl = h[i] * 1.001 if mode == "buystop" else (h[i] + l[i]) / 2
    for j in range(i + 1, min(n, i + 1 + valid)):
        if mode == "buystop":
            if h[j] >= lvl:
                return j, max(lvl, o[j])                    # gap above trigger fills at the open
            if l[j] <= stop:
                return None                                 # setup failed before triggering
        else:
            if o[j] <= stop:
                return None
            if l[j] <= lvl:
                return j, min(lvl, o[j])
            if h[j] >= lvl + rr * (lvl - stop):
                return None                                 # move ran without us
    return None


def _manage(o, h, l, c, e20, jf, fp, stop, rr, hold, n, style):
    """Return (exit bar, R multiple). 'fixed': all out at rr. 'scale': 50% at rr, stop to breakeven,
    rest trails: exit on a close below EMA20. Stop is checked first when both hit on one bar (conservative)."""
    risk, target, cur, half = fp - stop, fp + rr * (fp - stop), stop, False
    last = min(n, jf + hold)
    for j in range(jf, last):
        if j > jf and o[j] <= cur:
            ex = o[j]
        elif l[j] <= cur:
            ex = cur
        else:
            ex = None
        if ex is not None:
            r = (ex - fp) / risk
            return j, (0.5 * rr + 0.5 * r if half else r)
        if not half and h[j] >= target:
            if style == "fixed":
                return j, (max(target, o[j]) - fp) / risk if j > jf else rr
            half, cur = True, fp
        elif half and c[j] < e20[j]:
            return j, 0.5 * rr + 0.5 * (c[j] - fp) / risk
    j = last - 1
    r = (c[j] - fp) / risk
    return j, (0.5 * rr + 0.5 * r if half else r)


def backtest(frames, mode="open", style="fixed", rr=2.0, hold=15, cost=0.002, valid=3, trail="EMA20"):
    """frames: {symbol: signals() output on FULL history built with the same entry_mode}. trail = the EMA column the
    scale-out style trails under (EMA20 for swing, EMA50 for positional holds)."""
    rows = []
    for sym, d in frames.items():
        o, h, l, c, e20 = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", trail))
        stops, buys, n = d["Stop"].to_numpy(), d["Buy"].to_numpy(), len(d)
        dates = d["Date"].to_numpy() if "Date" in d else np.arange(n)
        flags = {k: d[k].to_numpy() for k in SETUPS}
        free = 0
        for i in np.flatnonzero(buys):
            if i < 220 or i + 1 >= n or i < free:
                continue
            f = _fill(mode, i, o, h, l, stops[i], n, valid, rr)
            if f is None:
                continue
            jf, fp = f
            risk = fp - stops[i]
            if not (risk > 0) or risk / fp > 0.12:
                continue
            j_exit, r = _manage(o, h, l, c, e20, jf, fp, stops[i], rr, hold, n, style)
            r -= cost * fp / risk
            rows.append({"Symbol": sym, "Date": dates[i], "Setup": next((SETUPS[k] for k in SETUPS if flags[k][i]), "-"),
                         "Entry": round(fp, 2), "Days": j_exit - i, "R": round(r, 2)})
            free = j_exit + 1
    return pd.DataFrame(rows)


def backtest_stats(t):
    if t.empty:
        return {}
    t = t.sort_values("Date")
    w, ls = t.loc[t["R"] > 0, "R"], t.loc[t["R"] <= 0, "R"]
    cum = t["R"].cumsum()
    streak = best = 0
    for r in t["R"]:
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)
    return {"Trades": len(t), "Win %": round(len(w) / len(t) * 100, 1), "Avg R": round(float(t["R"].mean()), 2),
            "Profit factor": round(float(w.sum() / abs(ls.sum())), 2) if ls.sum() != 0 else np.inf,
            "Max losing streak": best, "Max drawdown (R)": round(float((cum - cum.cummax()).min()), 1),
            "Avg days": round(float(t["Days"].mean()), 1)}
