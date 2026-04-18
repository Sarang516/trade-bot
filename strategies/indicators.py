"""
strategies/indicators.py — Technical indicator library.

Pure numpy/pandas — no ta-lib, no API calls.
Every function is stateless: pass a DataFrame, get a Series back.
All functions handle NaN gracefully: the first `period-1` rows are NaN.

Public API
----------
  Trend        : ema, sma, dema, vwap
  Volatility   : atr, bollinger_bands
  Momentum     : rsi, macd, stochastic, williams_r
  Volume       : volume_ma, volume_surge, relative_volume, obv
  Composite    : supertrend, pivot_points
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _close(df_or_series: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    """Extract the close price regardless of whether input is DataFrame or Series."""
    if isinstance(df_or_series, pd.Series):
        return df_or_series
    if "close" not in df_or_series.columns:
        raise ValueError("DataFrame must have a 'close' column")
    return df_or_series["close"]


def _require_cols(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _date_groups(df: pd.DataFrame):
    """Return a GroupBy object keyed by calendar date (for VWAP reset)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.groupby(df.index.normalize())
    if "datetime" in df.columns:
        key = pd.to_datetime(df["datetime"]).dt.normalize()
        return df.groupby(key)
    raise ValueError("DataFrame needs a DatetimeIndex or 'datetime' column for VWAP")


# ═════════════════════════════════════════════════════════════════════════════
# Trend indicators
# ═════════════════════════════════════════════════════════════════════════════

def ema(
    df_or_series: Union[pd.DataFrame, pd.Series],
    period: int,
) -> pd.Series:
    """
    Exponential Moving Average.

    Uses pandas ewm with span=period and adjust=False, which matches
    the standard EMA formula used on most charting platforms.
    """
    s = _close(df_or_series)
    return s.ewm(span=period, adjust=False).mean()


def sma(
    df_or_series: Union[pd.DataFrame, pd.Series],
    period: int,
) -> pd.Series:
    """Simple Moving Average."""
    s = _close(df_or_series)
    return s.rolling(window=period, min_periods=period).mean()


def dema(
    df_or_series: Union[pd.DataFrame, pd.Series],
    period: int,
) -> pd.Series:
    """
    Double Exponential Moving Average — reduces lag compared to EMA.

    Formula: DEMA = 2 * EMA(close, n) - EMA(EMA(close, n), n)
    """
    s = _close(df_or_series)
    e1 = s.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    return 2 * e1 - e2


def vwap(
    df: pd.DataFrame,
    num_std: float = 1.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Volume Weighted Average Price with standard-deviation bands.

    Resets at the start of each calendar day so it stays meaningful
    for intraday trading.

    Formula
    -------
    Typical Price (TP)  = (high + low + close) / 3
    VWAP                = cumsum(TP × volume) / cumsum(volume)
    VWAP Variance       = cumsum((TP - VWAP)² × volume) / cumsum(volume)
    Upper Band          = VWAP + num_std × sqrt(VWAP Variance)
    Lower Band          = VWAP - num_std × sqrt(VWAP Variance)

    Returns
    -------
    (vwap_series, upper_band, lower_band)
    """
    _require_cols(df, "high", "low", "close", "volume")

    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap_parts, upper_parts, lower_parts = [], [], []

    for _, group in _date_groups(df):
        tp_g = tp.loc[group.index]
        vol_g = group["volume"]
        cum_vol = vol_g.cumsum().replace(0, np.nan)

        vwap_g = (tp_g * vol_g).cumsum() / cum_vol
        # Volume-weighted variance of TP around VWAP
        variance_g = ((tp_g - vwap_g) ** 2 * vol_g).cumsum() / cum_vol
        std_g = variance_g.apply(np.sqrt)

        vwap_parts.append(vwap_g)
        upper_parts.append(vwap_g + num_std * std_g)
        lower_parts.append(vwap_g - num_std * std_g)

    vwap_s = pd.concat(vwap_parts).reindex(df.index)
    upper_s = pd.concat(upper_parts).reindex(df.index)
    lower_s = pd.concat(lower_parts).reindex(df.index)
    return vwap_s, upper_s, lower_s


# ═════════════════════════════════════════════════════════════════════════════
# Volatility indicators
# ═════════════════════════════════════════════════════════════════════════════

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing (alpha = 1/period).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder's EMA of True Range
    """
    _require_cols(df, "high", "low", "close")
    high, low = df["high"], df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def bollinger_bands(
    df_or_series: Union[pd.DataFrame, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns
    -------
    (middle_band, upper_band, lower_band)
    middle = SMA(close, period)
    upper  = middle + std_dev × rolling_std
    lower  = middle - std_dev × rolling_std
    """
    s = _close(df_or_series)
    middle = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std(ddof=0)
    return middle, middle + std_dev * std, middle - std_dev * std


# ═════════════════════════════════════════════════════════════════════════════
# Momentum indicators
# ═════════════════════════════════════════════════════════════════════════════

def rsi(
    df_or_series: Union[pd.DataFrame, pd.Series],
    period: int = 14,
) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    Uses alpha = 1/period (not span=period) to match the original
    Wilder definition and match most trading platforms.

    Returns values in the range [0, 100].
    """
    s = _close(df_or_series)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).clip(0.0, 100.0)


def macd(
    df_or_series: Union[pd.DataFrame, pd.Series],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Returns
    -------
    (macd_line, signal_line, histogram)
    macd_line  = EMA(close, fast) - EMA(close, slow)
    signal_line = EMA(macd_line, signal)
    histogram   = macd_line - signal_line
    """
    s = _close(df_or_series)
    fast_ema = s.ewm(span=fast, adjust=False).mean()
    slow_ema = s.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D).

    %K = (close - lowest_low) / (highest_high - lowest_low) × 100
    %D = SMA(%K, d_period)

    Returns
    -------
    (percent_k, percent_d)
    """
    _require_cols(df, "high", "low", "close")
    low_n = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_n = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_n - low_n).replace(0.0, np.nan)
    k = ((df["close"] - low_n) / denom * 100.0).clip(0.0, 100.0)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R momentum oscillator.

    Range: -100 (oversold) to 0 (overbought).
    Overbought zone: > -20   |   Oversold zone: < -80

    Formula: (highest_high - close) / (highest_high - lowest_low) × -100
    """
    _require_cols(df, "high", "low", "close")
    high_n = df["high"].rolling(window=period, min_periods=period).max()
    low_n = df["low"].rolling(window=period, min_periods=period).min()
    denom = (high_n - low_n).replace(0.0, np.nan)
    return ((high_n - df["close"]) / denom * -100.0).clip(-100.0, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# Volume indicators
# ═════════════════════════════════════════════════════════════════════════════

def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple moving average of volume over `period` bars."""
    _require_cols(df, "volume")
    return df["volume"].rolling(window=period, min_periods=period).mean()


def volume_surge(df: pd.DataFrame, multiplier: float = 2.0, period: int = 20) -> pd.Series:
    """
    Boolean Series — True when current volume is `multiplier` × volume_ma.

    Identifies unusual activity spikes likely driven by institutions.
    """
    vma = volume_ma(df, period)
    return df["volume"] > multiplier * vma


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Relative volume ratio: current_volume / volume_ma.

    Values > 1 mean above-average activity; 2.0 = twice the average.
    """
    _require_cols(df, "volume")
    vma = volume_ma(df, period).replace(0.0, np.nan)
    return df["volume"] / vma


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — cumulative volume indicator.

    +volume on up-days, -volume on down-days.
    Divergence between OBV and price often precedes reversals.
    """
    _require_cols(df, "close", "volume")
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ═════════════════════════════════════════════════════════════════════════════
# Composite indicators
# ═════════════════════════════════════════════════════════════════════════════

def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.

    A trend-following overlay that stays below price in an uptrend and
    above it in a downtrend.  Direction switches trigger buy/sell signals.

    Algorithm
    ---------
    1. Compute ATR(period).
    2. Basic Upper = (H+L)/2 + multiplier × ATR
       Basic Lower = (H+L)/2 - multiplier × ATR
    3. Apply trend-following adjustment (upper can only move down,
       lower can only move up — unless price crosses the band).
    4. Supertrend = lower_band when bullish, upper_band when bearish.

    Returns
    -------
    (direction, supertrend_values)
    direction : pd.Series of int  — 1 = bullish, -1 = bearish
    st_values : pd.Series of float — the supertrend line price
    """
    _require_cols(df, "high", "low", "close")

    close_arr = df["close"].to_numpy(dtype=float)
    atr_arr = atr(df, period).to_numpy(dtype=float)
    hl_mid = ((df["high"] + df["low"]) / 2).to_numpy(dtype=float)

    n = len(df)
    basic_upper = hl_mid + multiplier * atr_arr
    basic_lower = hl_mid - multiplier * atr_arr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = np.ones(n, dtype=np.int8)
    st_vals = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(atr_arr[i]):
            continue

        # Final upper: can only decrease (or reset if price was above it)
        final_upper[i] = (
            basic_upper[i]
            if (basic_upper[i] < final_upper[i - 1]
                or close_arr[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )

        # Final lower: can only increase (or reset if price was below it)
        final_lower[i] = (
            basic_lower[i]
            if (basic_lower[i] > final_lower[i - 1]
                or close_arr[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        # Determine direction from previous supertrend value
        if np.isnan(st_vals[i - 1]):
            # Bootstrap: initialise based on current price vs bands
            if close_arr[i] > final_upper[i]:
                direction[i], st_vals[i] = 1, final_lower[i]
            else:
                direction[i], st_vals[i] = -1, final_upper[i]
        elif direction[i - 1] == 1:  # Bullish — track lower band
            if close_arr[i] < final_lower[i]:
                direction[i], st_vals[i] = -1, final_upper[i]  # flip bearish
            else:
                direction[i], st_vals[i] = 1, final_lower[i]
        else:                         # Bearish — track upper band
            if close_arr[i] > final_upper[i]:
                direction[i], st_vals[i] = 1, final_lower[i]   # flip bullish
            else:
                direction[i], st_vals[i] = -1, final_upper[i]

    return (
        pd.Series(direction, index=df.index, name="supertrend_dir"),
        pd.Series(st_vals, index=df.index, name="supertrend"),
    )


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic Pivot Points — daily support/resistance levels.

    Uses the previous bar's OHLC to compute the current bar's levels.
    Designed to be called on a daily DataFrame (one row per trading day).

    Levels returned
    ---------------
    PP  = (H + L + C) / 3
    R1  = 2×PP - L    |  S1 = 2×PP - H
    R2  = PP + (H-L)  |  S2 = PP - (H-L)
    R3  = H + 2×(PP-L)|  S3 = L - 2×(H-PP)

    Returns a DataFrame with columns: pp, r1, r2, r3, s1, s2, s3.
    All values are shifted forward by 1 bar (today's levels come from
    yesterday's OHLC).
    """
    _require_cols(df, "high", "low", "close")
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    c = df["close"].shift(1)
    pp = (h + l + c) / 3
    return pd.DataFrame({
        "pp": pp,
        "r1": 2 * pp - l,
        "r2": pp + (h - l),
        "r3": h + 2 * (pp - l),
        "s1": 2 * pp - h,
        "s2": pp - (h - l),
        "s3": l - 2 * (h - pp),
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: compute all indicators at once
# ═════════════════════════════════════════════════════════════════════════════

def add_all(
    df: pd.DataFrame,
    ema_period: int = 9,
    rsi_period: int = 14,
    atr_period: int = 14,
    vol_ma_period: int = 20,
    bb_period: int = 20,
    st_period: int = 10,
    st_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Compute all indicators and attach them as new columns on a copy of `df`.

    Useful for strategy development and backtesting.  Live strategies should
    call individual functions to avoid recalculating unused indicators.

    Added columns
    -------------
    ema{n}, sma20, dema{n}, vwap, vwap_upper, vwap_lower,
    atr, bb_mid, bb_upper, bb_lower,
    rsi, macd, macd_signal, macd_hist,
    stoch_k, stoch_d, williams_r,
    vol_ma, rel_vol, vol_surge, obv,
    supertrend_dir, supertrend
    """
    out = df.copy()

    out[f"ema{ema_period}"] = ema(out, ema_period)
    out["sma20"] = sma(out, 20)
    out[f"dema{ema_period}"] = dema(out, ema_period)

    if "volume" in out.columns:
        vwap_s, vwap_u, vwap_l = vwap(out)
        out["vwap"] = vwap_s
        out["vwap_upper"] = vwap_u
        out["vwap_lower"] = vwap_l

    out["atr"] = atr(out, atr_period)
    bb_m, bb_u, bb_l = bollinger_bands(out, bb_period)
    out["bb_mid"] = bb_m
    out["bb_upper"] = bb_u
    out["bb_lower"] = bb_l

    out["rsi"] = rsi(out, rsi_period)
    ml, sl_s, hist = macd(out)
    out["macd"] = ml
    out["macd_signal"] = sl_s
    out["macd_hist"] = hist

    if all(c in out.columns for c in ("high", "low")):
        sk, sd = stochastic(out)
        out["stoch_k"] = sk
        out["stoch_d"] = sd
        out["williams_r"] = williams_r(out)

    if "volume" in out.columns:
        out["vol_ma"] = volume_ma(out, vol_ma_period)
        out["rel_vol"] = relative_volume(out, vol_ma_period)
        out["vol_surge"] = volume_surge(out, period=vol_ma_period)
        out["obv"] = obv(out)

    if all(c in out.columns for c in ("high", "low")):
        st_dir, st_val = supertrend(out, st_period, st_mult)
        out["supertrend_dir"] = st_dir
        out["supertrend"] = st_val

    return out


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def _make_test_df(n: int = 300) -> pd.DataFrame:
    """Generate realistic-looking synthetic OHLCV data for tests."""
    rng = np.random.default_rng(42)
    close = 18_000 + np.cumsum(rng.normal(0, 50, n))
    spread = rng.uniform(20, 120, n)
    high = close + spread * rng.uniform(0.3, 1.0, n)
    low = close - spread * rng.uniform(0.3, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0, 1, n)
    volume = rng.integers(10_000, 500_000, n)

    idx = pd.date_range("2026-01-02 09:15", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _run_tests() -> None:
    df = _make_test_df(300)
    errors: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if not cond:
            errors.append(f"FAIL  {name}" + (f": {detail}" if detail else ""))
        else:
            print(f"  ok  {name}")

    print("\nRunning indicator unit tests...")

    # ── EMA / SMA ────────────────────────────────────────────────────
    e9 = ema(df, 9)
    check("ema returns correct length", len(e9) == len(df))
    check("ema first 0 rows are NaN", not np.isnan(e9.iloc[1]))  # EMA is defined from bar 0
    check("ema last value is finite", np.isfinite(e9.iloc[-1]))

    s20 = sma(df, 20)
    check("sma first 19 rows are NaN", all(np.isnan(s20.iloc[:19])))
    check("sma 20th row is finite", np.isfinite(s20.iloc[19]))

    de9 = dema(df, 9)
    check("dema returns correct length", len(de9) == len(df))
    check("dema last value is finite", np.isfinite(de9.iloc[-1]))

    # ── ATR ──────────────────────────────────────────────────────────
    a14 = atr(df, 14)
    check("atr all non-negative", (a14.dropna() >= 0).all())
    check("atr last value is finite", np.isfinite(a14.iloc[-1]))

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_m, bb_u, bb_l = bollinger_bands(df, 20)
    check("bb upper >= middle", (bb_u.dropna() >= bb_m.dropna()).all())
    check("bb lower <= middle", (bb_l.dropna() <= bb_m.dropna()).all())

    # ── RSI ──────────────────────────────────────────────────────────
    r14 = rsi(df, 14)
    check("rsi in [0, 100]", r14.dropna().between(0, 100).all())
    check("rsi last value finite", np.isfinite(r14.iloc[-1]))

    # ── MACD ─────────────────────────────────────────────────────────
    ml, sl_, hist = macd(df)
    check("macd histogram = line - signal",
          np.allclose(hist.dropna(), (ml - sl_).dropna(), atol=1e-8))
    check("macd returns 3 series", len(ml) == len(sl_) == len(hist))

    # ── Stochastic ───────────────────────────────────────────────────
    sk, sd = stochastic(df, 14, 3)
    check("stoch_k in [0, 100]", sk.dropna().between(0, 100).all())
    check("stoch_d in [0, 100]", sd.dropna().between(0, 100).all())

    # ── Williams %R ──────────────────────────────────────────────────
    wr = williams_r(df, 14)
    check("williams_r in [-100, 0]", wr.dropna().between(-100, 0).all())

    # ── VWAP ─────────────────────────────────────────────────────────
    vw, vw_u, vw_l = vwap(df)
    check("vwap upper >= vwap", (vw_u.dropna() >= vw.dropna()).all())
    check("vwap lower <= vwap", (vw_l.dropna() <= vw.dropna()).all())
    check("vwap in price range",
          vw.dropna().between(df["low"].min(), df["high"].max()).all())

    # ── Volume ───────────────────────────────────────────────────────
    vma = volume_ma(df, 20)
    check("volume_ma all positive", (vma.dropna() > 0).all())
    rv = relative_volume(df, 20)
    check("relative_volume all positive", (rv.dropna() > 0).all())
    vs = volume_surge(df, 2.0, 20)
    check("volume_surge is bool Series", vs.dtype == bool)
    ob = obv(df)
    check("obv returns correct length", len(ob) == len(df))

    # ── Supertrend ───────────────────────────────────────────────────
    st_dir, st_val = supertrend(df, 10, 3.0)
    valid_dir = st_dir.dropna()
    check("supertrend direction only +1/-1",
          set(valid_dir.unique()).issubset({1, -1}))
    check("supertrend values finite after warmup",
          st_val.iloc[20:].notna().all())

    # ── Pivot Points ─────────────────────────────────────────────────
    daily_df = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    piv = pivot_points(daily_df)
    check("pivot_points has 7 columns", list(piv.columns) == ["pp","r1","r2","r3","s1","s2","s3"])
    check("pivot r1 > pp", (piv["r1"].dropna() > piv["pp"].dropna()).all())
    check("pivot s1 < pp", (piv["s1"].dropna() < piv["pp"].dropna()).all())

    # ── add_all ──────────────────────────────────────────────────────
    enriched = add_all(df)
    expected_cols = ["ema9", "sma20", "rsi", "macd", "atr", "vwap",
                     "supertrend_dir", "supertrend", "obv"]
    for col in expected_cols:
        check(f"add_all has column '{col}'", col in enriched.columns)

    print()
    if errors:
        for e in errors:
            print(e)
        raise AssertionError(f"{len(errors)} test(s) failed")
    else:
        print(f"All {28 + len(expected_cols)} tests passed.\n")


if __name__ == "__main__":
    _run_tests()
