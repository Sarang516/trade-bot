"""
scripts/checklist.py - Pre-live safety checker.

Run BEFORE switching to live trading:
    python scripts/checklist.py

Exits with code 0 if all checks pass, 1 if any FAIL.
Prints a colour-coded table - green PASS, yellow WARN, red FAIL.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    _rich = True
except ImportError:
    _rich = False

# ── result helpers ────────────────────────────────────────────────────────────

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_results: list[tuple[str, str, str]] = []   # (category, detail, status)


def record(category: str, detail: str, status: str) -> None:
    _results.append((category, detail, status))


def check(category: str, detail: str, fn: Callable[[], str]) -> str:
    """Run fn(), record result, return status."""
    try:
        status = fn()
    except Exception as exc:
        status = FAIL
        detail = f"{detail} - {exc}"
    record(category, detail, status)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python version
# ─────────────────────────────────────────────────────────────────────────────

def _check_python() -> str:
    v = sys.version_info
    if v >= (3, 11):
        return PASS
    if v >= (3, 9):
        return WARN   # works, but 3.11+ recommended
    return FAIL       # async bug risks on 3.8-


check("Python", f"Version {sys.version.split()[0]} (need >= 3.9, recommend 3.11+)", _check_python)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Required packages
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("kiteconnect",          "kiteconnect"),
    ("pydantic",             "pydantic"),
    ("pydantic_settings",    "pydantic_settings"),
    ("pandas",               "pandas"),
    ("numpy",                "numpy"),
    ("sqlalchemy",           "sqlalchemy"),
    ("flask",                "flask"),
    ("loguru",               "loguru"),
    ("rich",                 "rich"),
    ("click",                "click"),
    ("filelock",             "filelock"),
    ("python-telegram-bot",  "telegram"),
    ("requests",             "requests"),
    ("tenacity",             "tenacity"),
]

for pkg_name, import_name in REQUIRED_PACKAGES:
    def _mk_pkg_check(imp):
        def _fn():
            importlib.import_module(imp)
            return PASS
        return _fn
    check("Package", f"{pkg_name}", _mk_pkg_check(import_name))


# ─────────────────────────────────────────────────────────────────────────────
# 3. .env file + credentials
# ─────────────────────────────────────────────────────────────────────────────

def _check_env_file() -> str:
    env = ROOT / ".env"
    if env.exists():
        return PASS
    example = ROOT / ".env.example"
    if example.exists():
        return FAIL   # file missing but example present - user needs to copy it
    return WARN       # no .env and no example - might be using real env vars


check(".env", ".env file exists", _check_env_file)


def _load_env_settings():
    """Return Settings without crashing if import fails."""
    try:
        from config import Settings
        return Settings()
    except Exception:
        return None


_s = _load_env_settings()


def _credential_check(name: str, value: str, dummy: str) -> str:
    if not value or value == dummy:
        return FAIL
    if "DUMMY" in value.upper() or value == dummy:
        return FAIL
    return PASS


if _s:
    check("Credentials", "ZERODHA_API_KEY set",
          lambda: _credential_check("key", _s.zerodha_api_key, "DUMMY_ZERODHA_KEY"))
    check("Credentials", "ZERODHA_API_SECRET set",
          lambda: _credential_check("secret", _s.zerodha_api_secret, "DUMMY_ZERODHA_SECRET"))
    check("Credentials", "ZERODHA_ACCESS_TOKEN set",
          lambda: _credential_check("token", _s.zerodha_access_token, "DUMMY_ACCESS_TOKEN"))

    def _check_paper_mode() -> str:
        if _s.paper_trade:
            return WARN   # warn: paper mode on; user may want to confirm
        return PASS
    check("Trading", "PAPER_TRADE mode (WARN = paper is on, expected for first run)",
          _check_paper_mode)

    def _check_capital() -> str:
        if _s.trading_capital < 10_000:
            return WARN   # unrealistically low
        return PASS
    check("Trading", f"TRADING_CAPITAL Rs.{_s.trading_capital:,.0f}", _check_capital)

    def _check_risk() -> str:
        if _s.risk_per_trade_pct > 3.0:
            return WARN   # aggressive
        return PASS
    check("Risk", f"RISK_PER_TRADE_PCT {_s.risk_per_trade_pct}% (warn if > 3%)", _check_risk)

    def _check_daily_loss() -> str:
        if _s.max_daily_loss_inr <= 0:
            return FAIL
        return PASS
    check("Risk", f"MAX_DAILY_LOSS_INR Rs.{_s.max_daily_loss_inr:,.0f}", _check_daily_loss)
else:
    record("Credentials", "Could not load config.py - check for syntax errors", FAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Logs directory is writable
# ─────────────────────────────────────────────────────────────────────────────

def _check_log_dir() -> str:
    log_dir = (_s.log_dir if _s else ROOT / "logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        test_file = log_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return PASS
    except Exception:
        return FAIL


check("Filesystem", "logs/ directory is writable", _check_log_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SQLite DB can be created / opened
# ─────────────────────────────────────────────────────────────────────────────

def _check_db() -> str:
    try:
        from db.trade_logger import TradeLogger
        tl = TradeLogger()
        # Just opening the logger creates the tables
        return PASS
    except Exception as exc:
        raise RuntimeError(f"DB init failed: {exc}") from exc


check("Database", "SQLite trade DB initialises", _check_db)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Broker connection (paper-safe: just checks auth, no order placed)
# ─────────────────────────────────────────────────────────────────────────────

def _check_broker_connect() -> str:
    if _s is None:
        return FAIL
    if _s.zerodha_access_token in ("DUMMY_ACCESS_TOKEN", ""):
        return WARN   # no token set - skip actual connect attempt
    try:
        from brokers import get_broker
        broker = get_broker(_s)
        broker.connect()
        broker.disconnect()
        return PASS
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


check("Broker", "Zerodha broker connect / disconnect", _check_broker_connect)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Telegram (optional - just validate token format, no real HTTP call)
# ─────────────────────────────────────────────────────────────────────────────

def _check_telegram() -> str:
    if _s is None:
        return WARN
    token = _s.telegram_bot_token
    chat  = _s.telegram_chat_id
    if token in ("DUMMY_TELEGRAM_TOKEN", "") or chat in ("000000000", ""):
        return WARN   # not configured - OK, alerts go to console
    # Basic token format: <digits>:<alphanum>
    parts = token.split(":")
    if len(parts) != 2 or not parts[0].isdigit():
        return FAIL
    return PASS


check("Telegram", "Token format valid (WARN = not configured)", _check_telegram)


# ─────────────────────────────────────────────────────────────────────────────
# 8. No stale PID lock from a previous crashed instance
# ─────────────────────────────────────────────────────────────────────────────

def _check_pid_lock() -> str:
    log_dir = (_s.log_dir if _s else ROOT / "logs")
    lock_file = log_dir / "trading_bot.pid.lock"
    if lock_file.exists():
        return WARN   # may be stale from a crash - bot will still start (filelock re-acquires)
    return PASS


check("Process", "No stale PID lock file", _check_pid_lock)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Run pytest (skip slow integration tests)
# ─────────────────────────────────────────────────────────────────────────────

def _check_unit_tests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no",
         "-m", "not slow", "--timeout=30"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        # Extract summary line
        lines = result.stdout.strip().splitlines()
        summary = lines[-1] if lines else "passed"
        return PASS
    # Check if it's just "no tests collected" (tests dir missing etc.)
    if "no tests ran" in result.stdout or "collected 0" in result.stdout:
        return WARN
    return FAIL


check("Tests", "pytest unit tests pass", _check_unit_tests)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Strategy imports cleanly
# ─────────────────────────────────────────────────────────────────────────────

def _check_strategy() -> str:
    from strategies import get_strategy
    if _s:
        strategy = get_strategy(_s.strategy, symbol="NIFTY", settings=_s)
    else:
        from config import Settings
        strategy = get_strategy("vwap_volume", symbol="NIFTY", settings=Settings())
    return PASS


check("Strategy", f"Strategy '{(_s.strategy if _s else 'vwap_volume')}' imports + constructs", _check_strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────────────────────

def _print_results() -> bool:
    failures = sum(1 for _, _, s in _results if s == FAIL)
    warnings = sum(1 for _, _, s in _results if s == WARN)

    if _rich:
        table = Table(title="Pre-Live Safety Checklist", show_lines=True)
        table.add_column("Category", style="bold cyan", min_width=14)
        table.add_column("Check", min_width=55)
        table.add_column("Result", justify="center", min_width=6)

        colours = {PASS: "green", WARN: "yellow", FAIL: "red"}
        for cat, detail, status in _results:
            table.add_row(cat, detail, f"[{colours[status]}]{status}[/{colours[status]}]")

        console.print()
        console.print(table)
        console.print()
        if failures:
            console.print(f"[red bold]X  {failures} FAIL - fix before going live.[/red bold]")
        if warnings:
            console.print(f"[yellow]!  {warnings} WARN - review before going live.[/yellow]")
        if not failures:
            console.print("[green bold]OK  All required checks passed.[/green bold]")
    else:
        # Plain-text fallback
        print("\nPre-Live Safety Checklist")
        print("=" * 80)
        for cat, detail, status in _results:
            print(f"  [{status}]  {cat:14s}  {detail}")
        print("=" * 80)
        if failures:
            print(f"FAIL: {failures} check(s) failed.")
        if warnings:
            print(f"WARN: {warnings} warning(s).")
        if not failures:
            print("All required checks passed.")

    return failures == 0


if __name__ == "__main__":
    ok = _print_results()
    sys.exit(0 if ok else 1)
