"""
strategies/regime_detector.py - Market Regime Detector.

Classifies the current market into one of four regimes every morning
so the bot can load the best-performing parameter set for that condition.

Regimes
-------
  BULL      - Strong uptrend (price above rising EMAs, positive momentum)
  BEAR      - Strong downtrend (price below falling EMAs, negative momentum)
  RANGE     - Sideways / low volatility (ADX < threshold, narrow Bollinger)
  VOLATILE  - High-volatility whipsaw (ATR spike, VIX-like conditions)

Detection logic
---------------
1. Fetch last 60 days of DAILY closes for the symbol.
2. Compute:
     - 20-day EMA slope (positive = trending up, negative = down)
     - 50-day EMA: is price above or below?
     - ADX(14): > 25 = trending, < 20 = ranging
     - ATR% (ATR / close): > 2.5% = volatile
     - 10-day return: positive = bullish momentum, negative = bearish
3. Score each regime and pick the highest score.

Usage
-----
    from strategies.regime_detector import RegimeDetector, Regime
    detector = RegimeDetector(broker, settings)
    regime = detector.detect("RELIANCE")
    print(regime)  # Regime.BEAR
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd
from loguru import logger


class Regime(str, Enum):
    BULL     = "BULL"
    BEAR     = "BEAR"
    RANGE    = "RANGE"
    VOLATILE = "VOLATILE"
    UNKNOWN  = "UNKNOWN"    # not enough data


# ── Indicator helpers (pure pandas, no external dep) ─────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX without ta-lib."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, float("nan"))

    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ── Main class ───────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Detects market regime using daily OHLCV data.

    Parameters
    ----------
    broker      : ZerodhaBroker (or any BaseBroker) - used to fetch daily data
    settings    : Settings instance
    lookback    : Calendar days of daily history to use for detection
    """

    # Thresholds — tune these if the classifier misfires on your symbols
    ADX_TREND_THRESHOLD  = 25    # ADX > 25 = trending (BULL or BEAR)
    ADX_RANGE_THRESHOLD  = 20    # ADX < 20 = ranging
    ATR_VOLATILE_PCT     = 2.5   # ATR / close > 2.5% = volatile
    EMA_SLOPE_BULLISH    = 0.0   # positive 5-day EMA slope = bullish direction
    RETURN_10D_THRESHOLD = 3.0   # 10-day return > +3% = strong bull, < -3% = strong bear

    def __init__(self, broker, settings, lookback: int = 90) -> None:
        self._broker   = broker
        self._settings = settings
        self._lookback = lookback
        self._cache: dict[str, tuple[datetime, Regime]] = {}   # symbol -> (timestamp, regime)

    def detect(self, symbol: str, use_cache_minutes: int = 60) -> Regime:
        """
        Return the current regime for the symbol.
        Result is cached for `use_cache_minutes` to avoid redundant API calls.
        """
        now = datetime.now()
        if symbol in self._cache:
            ts, cached_regime = self._cache[symbol]
            age_minutes = (now - ts).total_seconds() / 60
            if age_minutes < use_cache_minutes:
                logger.debug("Regime cache hit for {}: {}", symbol, cached_regime)
                return cached_regime

        regime = self._compute(symbol)
        self._cache[symbol] = (now, regime)
        logger.info("Regime for {}: {}", symbol, regime.value)
        return regime

    def _compute(self, symbol: str) -> Regime:
        try:
            df = self._fetch_daily(symbol)
        except Exception as exc:
            logger.warning("RegimeDetector: could not fetch data for {}: {}", symbol, exc)
            return Regime.UNKNOWN

        if len(df) < 30:
            logger.warning("RegimeDetector: not enough daily data for {} ({})", symbol, len(df))
            return Regime.UNKNOWN

        close = df["close"]

        # -- Indicator values (last bar) -----------------------------------
        ema20        = _ema(close, 20)
        ema50        = _ema(close, 50)
        atr_series   = _atr(df, 14)
        adx_series   = _adx(df, 14)

        last_close   = close.iloc[-1]
        last_ema20   = ema20.iloc[-1]
        last_ema50   = ema50.iloc[-1]
        last_adx     = adx_series.iloc[-1]
        last_atr_pct = (atr_series.iloc[-1] / last_close * 100) if last_close > 0 else 0

        # EMA20 slope: compare last value vs 5 bars ago (% change)
        ema20_5ago  = ema20.iloc[-6] if len(ema20) >= 6 else ema20.iloc[0]
        ema20_slope = (last_ema20 - ema20_5ago) / ema20_5ago * 100 if ema20_5ago > 0 else 0

        # 10-day return
        ret10 = 0.0
        if len(close) >= 11:
            ret10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100

        # -- Regime scoring -----------------------------------------------
        scores = {
            Regime.BULL:     0,
            Regime.BEAR:     0,
            Regime.RANGE:    0,
            Regime.VOLATILE: 0,
        }

        # Volatility check first (overrides trend signals when ATR is extreme)
        if last_atr_pct > self.ATR_VOLATILE_PCT:
            scores[Regime.VOLATILE] += 3

        # Trending vs ranging (ADX)
        if last_adx > self.ADX_TREND_THRESHOLD:
            # It's trending — determine direction
            if ema20_slope > self.EMA_SLOPE_BULLISH and last_close > last_ema50:
                scores[Regime.BULL] += 3
            elif ema20_slope < self.EMA_SLOPE_BULLISH and last_close < last_ema50:
                scores[Regime.BEAR] += 3
            else:
                # Trending but direction unclear — use 10d return as tiebreaker
                if ret10 > 0:
                    scores[Regime.BULL] += 1
                else:
                    scores[Regime.BEAR] += 1
        elif last_adx < self.ADX_RANGE_THRESHOLD:
            scores[Regime.RANGE] += 3

        # 10-day momentum confirmation
        if ret10 > self.RETURN_10D_THRESHOLD:
            scores[Regime.BULL] += 2
        elif ret10 < -self.RETURN_10D_THRESHOLD:
            scores[Regime.BEAR] += 2
        elif abs(ret10) < 1.5:
            scores[Regime.RANGE] += 1

        # Price vs EMA confirmation
        if last_close > last_ema20 > last_ema50:
            scores[Regime.BULL] += 1
        elif last_close < last_ema20 < last_ema50:
            scores[Regime.BEAR] += 1

        best = max(scores, key=lambda r: scores[r])

        logger.info(
            "Regime detection {} | ADX={:.1f} ATR%={:.2f} slope={:.2f}% ret10={:.1f}% | scores={} -> {}",
            symbol, last_adx, last_atr_pct, ema20_slope, ret10,
            {r.value: s for r, s in scores.items()}, best.value,
        )
        return best

    def _fetch_daily(self, symbol: str) -> pd.DataFrame:
        """Fetch daily OHLCV from broker. Returns DataFrame indexed by date."""
        from brokers.base_broker import Exchange
        from data.feed import HistoricalData

        to_date   = datetime.now()
        from_date = to_date - timedelta(days=self._lookback)

        hist = HistoricalData(self._broker)
        # Use 60-min data resampled to daily — avoids a separate daily API call
        df = hist.fetch(symbol, Exchange.NSE, from_date, to_date, interval_minutes=60)

        if df.empty:
            return df

        daily = df.resample("D").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["close"])

        return daily


# ── Convenience function ─────────────────────────────────────────────────────

def describe_regime(regime: Regime) -> str:
    """Return a human-readable description for logging / dashboard."""
    return {
        Regime.BULL:     "BULL - Strong uptrend: favour LONG entries, wider targets",
        Regime.BEAR:     "BEAR - Strong downtrend: favour SHORT entries, tighter targets",
        Regime.RANGE:    "RANGE - Sideways: mean-reversion setups, tight RR, quick exits",
        Regime.VOLATILE: "VOLATILE - Choppy: reduce position size, wider stops, fewer entries",
        Regime.UNKNOWN:  "UNKNOWN - Not enough data to classify",
    }[regime]
