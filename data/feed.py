"""
data/feed.py — Live market data feed, candle building, and historical data.

Classes
-------
MarketHours      NSE trading hours + holiday calendar + time utilities
CandleBuilder    Aggregates live ticks into OHLCV candles for any interval
HistoricalData   Fetch + SQLite-cached OHLCV bars for warmup / backtest
InstrumentLookup Symbol → Zerodha instrument token with daily file cache
TickDataFeed     Thread-safe live tick subscriber with auto-reconnect
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time as _time
from collections import deque
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pytz
from loguru import logger

from brokers.base_broker import BaseBroker, Exchange
from strategies.base_strategy import Candle

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
_DATA_DIR = _ROOT / "data"
_CACHE_DB = _DATA_DIR / "market_cache.db"
_INSTRUMENT_CACHE = _DATA_DIR / "instruments_cache.json"
_HOLIDAY_FILE = _DATA_DIR / "nse_holidays.json"


# ── NSE Holiday Calendar ───────────────────────────────────────────────────────
# Hardcoded NSE trading holidays.  Update _HOLIDAY_FILE (data/nse_holidays.json)
# each year using NSE's official circular — that file takes priority over this dict.
_BUILTIN_HOLIDAYS: dict[int, set[date]] = {
    2025: {
        date(2025, 2, 26),   # Mahashivratri
        date(2025, 3, 14),   # Holi
        date(2025, 3, 31),   # Id-Ul-Fitr (Eid)
        date(2025, 4, 10),   # Shri Ram Navami
        date(2025, 4, 14),   # Dr. Ambedkar Jayanti
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 1),    # Maharashtra Day
        date(2025, 8, 15),   # Independence Day
        date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
        date(2025, 10, 24),  # Diwali (Laxmi Pujan)
        date(2025, 11, 5),   # Diwali (Balipratipada)
        date(2025, 11, 15),  # Gurunanak Jayanti
        date(2025, 12, 25),  # Christmas
    },
    2026: {
        date(2026, 1, 26),   # Republic Day
        date(2026, 2, 16),   # Mahashivratri
        date(2026, 3, 20),   # Id-Ul-Fitr (Eid) — tentative, lunar dependent
        date(2026, 4, 3),    # Good Friday
        date(2026, 4, 14),   # Dr. Ambedkar Jayanti
        date(2026, 5, 1),    # Maharashtra Day
        date(2026, 10, 2),   # Gandhi Jayanti
        date(2026, 10, 19),  # Dussehra (tentative)
        date(2026, 11, 7),   # Diwali Laxmi Pujan (tentative)
        date(2026, 11, 8),   # Diwali Balipratipada (tentative)
        date(2026, 11, 14),  # Gurunanak Jayanti (tentative)
        date(2026, 12, 25),  # Christmas
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# MarketHours
# ═════════════════════════════════════════════════════════════════════════════

class MarketHours:
    """
    NSE market hours helper (all times in IST).

    Holiday list is loaded from data/nse_holidays.json if it exists,
    falling back to the built-in _BUILTIN_HOLIDAYS dict.
    Update the JSON file each year using NSE's official holiday circular.
    """

    IST = pytz.timezone("Asia/Kolkata")
    _OPEN = time(9, 15)
    _CLOSE = time(15, 30)

    def __init__(self) -> None:
        self._holidays: dict[int, set[date]] = dict(_BUILTIN_HOLIDAYS)
        self._load_holiday_file()

    def _load_holiday_file(self) -> None:
        if not _HOLIDAY_FILE.exists():
            return
        try:
            raw: dict[str, list[str]] = json.loads(_HOLIDAY_FILE.read_text())
            for year_str, dates in raw.items():
                year = int(year_str)
                self._holidays[year] = {
                    date.fromisoformat(d) for d in dates
                }
            logger.debug("Loaded NSE holidays from {}", _HOLIDAY_FILE)
        except Exception as exc:
            logger.warning("Could not load holiday file: {}", exc)

    def _now_ist(self, dt: datetime | None = None) -> datetime:
        d = dt or datetime.now()
        if d.tzinfo is None:
            d = pytz.utc.localize(d)
        return d.astimezone(self.IST)

    def is_trading_day(self, dt: datetime | None = None) -> bool:
        """Return True if the given date is a weekday that is not an NSE holiday."""
        d = self._now_ist(dt).date()
        if d.weekday() >= 5:   # Saturday=5, Sunday=6
            return False
        holidays = self._holidays.get(d.year, set())
        return d not in holidays

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Return True if NSE is currently open (trading day + within hours)."""
        now_ist = self._now_ist(dt)
        if not self.is_trading_day(now_ist):
            return False
        t = now_ist.time()
        return self._OPEN <= t <= self._CLOSE

    def minutes_to_open(self, dt: datetime | None = None) -> float:
        """Minutes until the next market open (returns 0 if market is open now)."""
        now_ist = self._now_ist(dt)
        if self.is_market_open(now_ist):
            return 0.0
        # Find next trading day at 9:15 AM
        candidate = now_ist.replace(
            hour=9, minute=15, second=0, microsecond=0
        )
        if now_ist.time() >= self._OPEN:
            candidate += timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return (candidate - now_ist).total_seconds() / 60

    def minutes_to_close(self, dt: datetime | None = None) -> float:
        """Minutes until market close today (negative if already closed)."""
        now_ist = self._now_ist(dt)
        close_dt = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        return (close_dt - now_ist).total_seconds() / 60


# ═════════════════════════════════════════════════════════════════════════════
# CandleBuilder
# ═════════════════════════════════════════════════════════════════════════════

class CandleBuilder:
    """
    Aggregates raw price ticks into fixed-interval OHLCV candles.

    Key design points
    -----------------
    - Zerodha tick volume is cumulative for the day — candle volume is the
      delta between the cumulative value at window-start and window-end.
    - Emits completed candles both via return value AND registered callbacks,
      so it works in both pull mode (main loop) and push mode (TickDataFeed).
    - Stores last `max_candles` candles in memory for indicator warm-up.
    - Thread-safe: all state protected by a single lock.
    """

    def __init__(
        self,
        interval_minutes: int = 5,
        max_candles: int = 200,
        on_candle: Optional[Callable[[Candle], None]] = None,
    ) -> None:
        self.interval = interval_minutes
        self._lock = threading.Lock()
        self._history: deque[Candle] = deque(maxlen=max_candles)
        self._callbacks: list[Callable[[Candle], None]] = []
        if on_candle:
            self._callbacks.append(on_candle)

        # Current window state
        self._current_start: datetime | None = None
        self._open = self._high = self._low = self._close = 0.0
        self._vol_start = 0
        self._oi = 0

    def subscribe(self, callback: Callable[[Candle], None]) -> None:
        """Register a function to be called whenever a candle closes."""
        self._callbacks.append(callback)

    def process_tick(self, tick: dict) -> Candle | None:
        """
        Feed one tick.  Returns the just-closed Candle when the interval
        rolls over, otherwise None.  Also fires all registered callbacks.
        """
        with self._lock:
            return self._process(tick)

    def _process(self, tick: dict) -> Candle | None:
        ts: datetime = tick.get("timestamp", datetime.now())
        ltp: float = float(tick.get("ltp", 0))
        vol: int = int(tick.get("volume", 0))
        oi: int = int(tick.get("oi", 0))

        if ltp <= 0:
            return None

        # Floor to nearest interval boundary
        floored = ts.replace(
            minute=(ts.minute // self.interval) * self.interval,
            second=0,
            microsecond=0,
        )

        if self._current_start is None:
            self._current_start = floored
            self._open = self._high = self._low = self._close = ltp
            self._vol_start = vol
            self._oi = oi
            return None

        if floored == self._current_start:
            self._high = max(self._high, ltp)
            self._low = min(self._low, ltp)
            self._close = ltp
            self._oi = oi
            return None

        # New window — emit completed candle
        candle = Candle(
            datetime=self._current_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=max(0, vol - self._vol_start),
            interval_minutes=self.interval,
        )
        self._history.append(candle)
        self._fire_callbacks(candle)

        # Start new window
        self._current_start = floored
        self._open = self._high = self._low = self._close = ltp
        self._vol_start = vol
        self._oi = oi

        return candle

    def _fire_callbacks(self, candle: Candle) -> None:
        for cb in self._callbacks:
            try:
                cb(candle)
            except Exception as exc:
                logger.error("CandleBuilder callback error: {}", exc)

    def get_candles(self) -> list[Candle]:
        """Return a snapshot of recent candle history (oldest first)."""
        with self._lock:
            return list(self._history)

    def get_df(self) -> pd.DataFrame:
        """Return candle history as a DataFrame with datetime index."""
        candles = self.get_candles()
        if not candles:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame([c.to_dict() for c in candles])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    def reset(self) -> None:
        """Clear all state — call at market open each day."""
        with self._lock:
            self._current_start = None
            self._open = self._high = self._low = self._close = 0.0
            self._vol_start = 0
            self._oi = 0
            self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


# ═════════════════════════════════════════════════════════════════════════════
# HistoricalData
# ═════════════════════════════════════════════════════════════════════════════

class HistoricalData:
    """
    Fetch OHLCV candles from the broker with a local SQLite cache.

    First call for a date range downloads from the broker and stores to
    cache.  Subsequent calls return from cache instantly — no API hit.
    Cache is keyed by (symbol, exchange, interval, datetime).
    """

    # Interval string used by Kite Connect API
    KITE_INTERVALS = {1: "minute", 3: "3minute", 5: "5minute",
                      15: "15minute", 30: "30minute", 60: "60minute"}

    def __init__(
        self,
        broker: BaseBroker,
        db_path: Path = _CACHE_DB,
    ) -> None:
        self._broker = broker
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol   TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    open     REAL,
                    high     REAL,
                    low      REAL,
                    close    REAL,
                    volume   INTEGER,
                    PRIMARY KEY (symbol, exchange, interval, datetime)
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # Zerodha Kite Connect hard limits per request (calendar days):
    #   1-min  : 60 days    3-min  : 60 days   5-min  : 60 days
    #   15-min : 60 days   30-min  : 60 days   60-min : 60 days
    #   daily  : 2000 days (not used here — all intraday)
    # No API limit on total history depth — can fetch many years by chunking.
    _CHUNK_DAYS = {1: 60, 3: 60, 5: 60, 15: 60, 30: 60, 60: 60}

    def fetch(
        self,
        symbol: str,
        exchange: Exchange,
        from_date: datetime,
        to_date: datetime,
        interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for the requested range.
        Loads from cache where available, fetches missing ranges from broker.
        Automatically chunks large date ranges to stay within Zerodha's 60-day
        per-request limit for intraday intervals.
        """
        interval_str = self.KITE_INTERVALS.get(interval_minutes, "5minute")
        cached = self._load_cache(symbol, exchange.value, interval_str,
                                  from_date, to_date)
        if cached is not None and not cached.empty:
            logger.debug("Cache hit for {} {} {} rows={}", symbol,
                         interval_str, exchange.value, len(cached))
            return cached

        chunk_days = self._CHUNK_DAYS.get(interval_minutes, 60)
        total_days = (to_date - from_date).days

        if total_days <= chunk_days:
            return self._fetch_single(symbol, exchange, from_date, to_date,
                                      interval_str, interval_minutes)

        # Split into chunks and concatenate
        logger.info(
            "Date range {} days exceeds {} day limit — fetching in chunks",
            total_days, chunk_days,
        )
        chunks: list[pd.DataFrame] = []
        chunk_start = from_date
        while chunk_start < to_date:
            chunk_end = min(
                chunk_start + timedelta(days=chunk_days),
                to_date,
            )
            part = self._fetch_single(symbol, exchange, chunk_start, chunk_end,
                                      interval_str, interval_minutes)
            if not part.empty:
                chunks.append(part)
            chunk_start = chunk_end

        if not chunks:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if not df.empty:
            self._save_cache(df, symbol, exchange.value, interval_str)
        return df

    def _fetch_single(
        self,
        symbol: str,
        exchange: Exchange,
        from_date: datetime,
        to_date: datetime,
        interval_str: str,
        interval_minutes: int,
    ) -> pd.DataFrame:
        logger.info("Fetching {} {} {} → {}", symbol, interval_str,
                    from_date.date(), to_date.date())
        try:
            df = self._broker.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                from_date=from_date,
                to_date=to_date,
                interval=interval_str,
            )
        except Exception as exc:
            logger.error("Historical data fetch failed: {}", exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if not df.empty:
            self._save_cache(df, symbol, exchange.value, interval_str)
        return df

    def warmup_candles(
        self,
        symbol: str,
        exchange: Exchange,
        interval_minutes: int = 5,
        n_candles: int = 200,
    ) -> list[Candle]:
        """
        Return the last `n_candles` historical candles as Candle objects.
        Used to pre-populate strategy candle history before live trading.
        """
        # Request enough history to cover n_candles (add buffer for weekends)
        trading_days_needed = (n_candles * interval_minutes // 375) + 5
        to_date = datetime.now()
        from_date = to_date - timedelta(days=trading_days_needed + 7)

        df = self.fetch(symbol, exchange, from_date, to_date, interval_minutes)
        if df.empty:
            logger.warning("No historical data available for warmup")
            return []

        df = df.tail(n_candles)
        candles: list[Candle] = []
        for idx, row in df.iterrows():
            dt = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            candles.append(Candle(
                datetime=dt,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                interval_minutes=interval_minutes,
            ))
        logger.info("Warmup: loaded {} candles for {}", len(candles), symbol)
        return candles

    def _load_cache(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame | None:
        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT datetime, open, high, low, close, volume
                    FROM candles
                    WHERE symbol=? AND exchange=? AND interval=?
                      AND datetime >= ? AND datetime <= ?
                    ORDER BY datetime ASC
                """, (symbol, exchange, interval,
                      from_date.isoformat(), to_date.isoformat())).fetchall()
            if not rows:
                return None
            df = pd.DataFrame(rows,
                              columns=["datetime", "open", "high",
                                       "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Cache read failed: {}", exc)
            return None

    def _save_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
    ) -> None:
        try:
            records = [
                (symbol, exchange, interval,
                 str(idx), row["open"], row["high"],
                 row["low"], row["close"], int(row["volume"]))
                for idx, row in df.iterrows()
            ]
            with self._connect() as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO candles
                    (symbol, exchange, interval, datetime,
                     open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
            logger.debug("Cached {} rows for {} {} {}", len(records),
                         symbol, exchange, interval)
        except Exception as exc:
            logger.warning("Cache write failed: {}", exc)


# ═════════════════════════════════════════════════════════════════════════════
# InstrumentLookup
# ═════════════════════════════════════════════════════════════════════════════

class InstrumentLookup:
    """
    Resolves trading symbols to Zerodha instrument tokens.

    Tokens are cached in data/instruments_cache.json and refreshed once per
    day on first access (Zerodha instrument lists update overnight).
    The broker's own in-memory cache is used within a session; this class
    provides persistence across bot restarts.
    """

    def __init__(
        self,
        broker: BaseBroker,
        cache_path: Path = _INSTRUMENT_CACHE,
    ) -> None:
        self._broker = broker
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, int]] = {}  # exchange -> symbol -> token
        self._loaded_at: datetime | None = None
        self._lock = threading.Lock()

    def get_token(self, symbol: str, exchange: Exchange) -> int:
        """Return the instrument token for a symbol, refreshing cache if needed."""
        with self._lock:
            self._ensure_fresh()
            return self._cache.get(exchange.value, {}).get(symbol, 0)

    def search(self, query: str, exchange: Exchange) -> list[dict]:
        """Return all instruments whose symbol or name contains `query`."""
        try:
            return self._broker.search_instruments(query, exchange)
        except Exception as exc:
            logger.error("Instrument search failed: {}", exc)
            return []

    def refresh(self) -> None:
        """Force a full refresh from the broker for all active exchanges."""
        with self._lock:
            self._fetch_and_cache()

    def _ensure_fresh(self) -> None:
        today = date.today()
        cache_date = (self._loaded_at.date()
                      if self._loaded_at else None)
        if cache_date == today and self._cache:
            return
        # Try to load from file first
        if self._cache_path.exists():
            try:
                payload = json.loads(self._cache_path.read_text())
                saved_date = date.fromisoformat(payload.get("date", "1970-01-01"))
                if saved_date == today:
                    self._cache = payload.get("instruments", {})
                    self._loaded_at = datetime.now()
                    logger.debug("Instruments loaded from cache ({})", today)
                    return
            except Exception as exc:
                logger.warning("Instrument cache file corrupt: {}", exc)
        self._fetch_and_cache()

    def _fetch_and_cache(self) -> None:
        logger.info("Refreshing instrument cache from broker...")
        new_cache: dict[str, dict[str, int]] = {}
        for exchange in (Exchange.NSE, Exchange.NFO, Exchange.BSE):
            try:
                instruments = self._broker.search_instruments("", exchange)
                new_cache[exchange.value] = {
                    inst.get("tradingsymbol", ""): inst.get("instrument_token", 0)
                    for inst in instruments
                    if inst.get("tradingsymbol")
                }
                logger.debug("Cached {} instruments for {}",
                             len(new_cache[exchange.value]), exchange.value)
            except Exception as exc:
                logger.warning("Could not fetch instruments for {}: {}",
                               exchange.value, exc)
        if new_cache:
            self._cache = new_cache
            self._loaded_at = datetime.now()
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(json.dumps({
                    "date": date.today().isoformat(),
                    "instruments": new_cache,
                }))
            except Exception as exc:
                logger.warning("Could not write instrument cache: {}", exc)


# ═════════════════════════════════════════════════════════════════════════════
# TickDataFeed
# ═════════════════════════════════════════════════════════════════════════════

class TickDataFeed:
    """
    Thread-safe live tick subscriber with automatic reconnection.

    Responsibilities
    ----------------
    1. Subscribe to broker WebSocket ticks for a list of symbols.
    2. Buffer incoming ticks in a thread-safe deque (maxlen configurable).
    3. Dispatch raw ticks to registered tick callbacks.
    4. Dispatch completed candles to per-interval candle builders.
    5. Monitor connection health and reconnect with exponential backoff.

    Usage
    -----
        feed = TickDataFeed(broker, symbols=["NIFTY 50"], settings=settings)
        feed.subscribe_tick(my_tick_handler)
        feed.subscribe_candle(my_candle_handler, interval_minutes=5)
        feed.start()
        ...
        feed.stop()
    """

    _BACKOFF_BASE = 2.0      # seconds — doubles on each retry
    _BACKOFF_MAX = 60.0      # cap at 60 s
    _MONITOR_INTERVAL = 30   # seconds between connection checks

    def __init__(
        self,
        broker: BaseBroker,
        symbols: list[str],
        exchange: Exchange = Exchange.NSE,
        max_ticks: int = 1000,
        settings=None,
    ) -> None:
        self._broker = broker
        self._symbols = symbols
        self._exchange = exchange
        self._settings = settings

        self._tick_buffer: deque[dict] = deque(maxlen=max_ticks)
        self._tick_callbacks: list[Callable[[dict], None]] = []
        self._candle_builders: dict[int, CandleBuilder] = {}  # interval -> builder

        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._retry_count = 0

    # ── Public API ────────────────────────────────────────────────────

    def subscribe_tick(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for every incoming raw tick."""
        self._tick_callbacks.append(callback)

    def subscribe_candle(
        self,
        callback: Callable[[Candle], None],
        interval_minutes: int = 5,
    ) -> None:
        """Register a callback for completed candles at the given interval."""
        if interval_minutes not in self._candle_builders:
            self._candle_builders[interval_minutes] = CandleBuilder(
                interval_minutes=interval_minutes
            )
        self._candle_builders[interval_minutes].subscribe(callback)

    def get_candle_builder(self, interval_minutes: int = 5) -> CandleBuilder:
        """Return (or create) the CandleBuilder for the given interval."""
        if interval_minutes not in self._candle_builders:
            self._candle_builders[interval_minutes] = CandleBuilder(
                interval_minutes=interval_minutes
            )
        return self._candle_builders[interval_minutes]

    def start(self) -> None:
        """Subscribe to broker ticks and start the reconnect monitor."""
        self._running = True
        self._broker.register_tick_callback(self._on_tick)
        self._connect()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="FeedMonitor"
        )
        self._monitor_thread.start()
        logger.info("TickDataFeed started | symbols={} exchange={}",
                    self._symbols, self._exchange.value)

    def stop(self) -> None:
        """Stop monitoring and unsubscribe ticks."""
        self._running = False
        try:
            self._broker.unsubscribe_ticks(self._symbols)
        except Exception:
            pass
        logger.info("TickDataFeed stopped")

    def get_recent_ticks(self, n: int = 100) -> list[dict]:
        """Return the last `n` ticks from the buffer (most recent last)."""
        with self._lock:
            ticks = list(self._tick_buffer)
        return ticks[-n:]

    def reset_candles(self) -> None:
        """Reset all candle builders — call at market open each morning."""
        for builder in self._candle_builders.values():
            builder.reset()

    # ── Internal ──────────────────────────────────────────────────────

    def _on_tick(self, tick: dict) -> None:
        """Receive one tick from the broker and fan it out."""
        with self._lock:
            self._tick_buffer.append(tick)

        # Raw tick callbacks
        for cb in self._tick_callbacks:
            try:
                cb(tick)
            except Exception as exc:
                logger.error("Tick callback error: {}", exc)

        # Feed into each interval's candle builder
        for builder in self._candle_builders.values():
            builder.process_tick(tick)

    def _connect(self) -> bool:
        try:
            if not self._broker.is_connected():
                self._broker.connect()
            self._broker.subscribe_ticks(self._symbols, self._exchange)
            self._retry_count = 0
            logger.info("Feed connected | {}", self._symbols)
            return True
        except Exception as exc:
            logger.error("Feed connect failed: {}", exc)
            return False

    def _monitor_loop(self) -> None:
        """Background thread: detect disconnection and reconnect with backoff."""
        while self._running:
            _time.sleep(self._MONITOR_INTERVAL)
            if not self._running:
                break
            if not self._broker.is_connected():
                self._retry_count += 1
                backoff = min(
                    self._BACKOFF_BASE ** self._retry_count,
                    self._BACKOFF_MAX,
                )
                logger.warning(
                    "Feed disconnected — reconnecting in {:.0f}s (attempt {})",
                    backoff, self._retry_count,
                )
                _time.sleep(backoff)
                if self._connect():
                    logger.info("Feed reconnected successfully")
                    self._retry_count = 0
