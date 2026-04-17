"""data/feed.py — Built in Phase 3."""
from datetime import datetime, time
import pytz

class CandleBuilder:
    def __init__(self, interval_minutes=5):
        self.interval = interval_minutes
        self._ticks = []
        self._current_candle_start = None
    def process_tick(self, tick): return None  # stub

class MarketHours:
    IST = pytz.timezone("Asia/Kolkata")
    OPEN = time(9, 15)
    CLOSE = time(15, 30)
    def is_market_open(self, dt=None):
        now = (dt or datetime.now()).astimezone(self.IST).time()
        return self.OPEN <= now <= self.CLOSE
    def minutes_to_open(self, dt=None):
        now = (dt or datetime.now()).astimezone(self.IST)
        from datetime import timedelta
        open_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        if now.time() > self.OPEN:
            open_dt += timedelta(days=1)
        return (open_dt - now).seconds / 60
