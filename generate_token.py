"""
generate_token.py — Zerodha daily access token generator.

Run this script every morning BEFORE starting the bot:
    python generate_token.py

What it does:
  1. Opens the Zerodha login page in your browser.
  2. You log in and Zerodha redirects to a URL containing a request_token.
  3. Paste that full redirect URL here.
  4. The script exchanges it for an access_token and saves it to your .env file.

Access tokens expire at 6 AM IST the next day — run this script once each morning.
"""

from __future__ import annotations

import re
import sys
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
ENV_FILE = ROOT_DIR / ".env"


def _read_env() -> dict[str, str]:
    """Parse .env file into a key→value dict."""
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def _update_env(key: str, new_value: str) -> None:
    """Replace the value of `key` in .env file (adds it if not present)."""
    if not ENV_FILE.exists():
        print(f"\n[ERROR] .env file not found at {ENV_FILE}")
        print("  → Copy .env.example to .env and fill in your credentials first.")
        sys.exit(1)

    content = ENV_FILE.read_text(encoding="utf-8")
    pattern = rf"^{re.escape(key)}\s*=.*$"
    replacement = f"{key}={new_value}"

    if re.search(pattern, content, flags=re.MULTILINE):
        updated = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        updated = content.rstrip() + f"\n{replacement}\n"

    ENV_FILE.write_text(updated, encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("  Zerodha Access Token Generator")
    print("=" * 60)

    # ── Load credentials from .env ─────────────────────────────────
    env = _read_env()
    api_key = env.get("ZERODHA_API_KEY", "")
    api_secret = env.get("ZERODHA_API_SECRET", "")

    if not api_key or api_key == "your_kite_api_key_here":
        print("\n[ERROR] ZERODHA_API_KEY not set in .env")
        print("  → Open .env and fill in your Kite Connect API key.")
        sys.exit(1)

    if not api_secret or api_secret == "your_kite_api_secret_here":
        print("\n[ERROR] ZERODHA_API_SECRET not set in .env")
        print("  → Open .env and fill in your Kite Connect API secret.")
        sys.exit(1)

    # ── Open Zerodha login in browser ──────────────────────────────
    login_url = f"https://kite.trade/connect/login?api_key={api_key}&v=3"
    print(f"\nStep 1 — Opening Zerodha login in your browser...")
    print(f"  URL: {login_url}")
    webbrowser.open(login_url)

    print("\nStep 2 — Log in with your Zerodha credentials.")
    print("  After login, your browser will redirect to a URL like:")
    print("  https://127.0.0.1/?request_token=XXXXXXXXXXXXXX&action=login&status=success")
    print("\n  (The page may show an error — that is normal. Just copy the full URL.)")

    # ── Get request_token from user ────────────────────────────────
    print("\nStep 3 — Paste the full redirect URL below and press Enter:")
    redirect_url = input("  URL: ").strip()

    match = re.search(r"request_token=([A-Za-z0-9]+)", redirect_url)
    if not match:
        # Maybe they pasted just the token itself
        if re.fullmatch(r"[A-Za-z0-9]{20,}", redirect_url):
            request_token = redirect_url
        else:
            print("\n[ERROR] Could not find request_token in the URL you pasted.")
            print("  → Make sure you copied the full redirect URL from the browser address bar.")
            sys.exit(1)
    else:
        request_token = match.group(1)

    print(f"\n  request_token: {request_token[:8]}{'*' * (len(request_token) - 8)}")

    # ── Exchange request_token for access_token ────────────────────
    print("\nStep 4 — Generating access token...")
    try:
        from kiteconnect import KiteConnect  # type: ignore
    except ImportError:
        print("\n[ERROR] kiteconnect package not installed.")
        print("  → Run: pip install kiteconnect")
        sys.exit(1)

    try:
        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token: str = session["access_token"]
        user_name: str = session.get("user_name", "Unknown")
        user_id: str = session.get("user_id", "")
    except Exception as exc:
        print(f"\n[ERROR] Token generation failed: {exc}")
        print("  Common causes:")
        print("  - request_token already used (each token is single-use)")
        print("  - API secret is wrong in .env")
        print("  - Kite Connect app redirect URL not configured correctly")
        sys.exit(1)

    # ── Save access_token to .env ──────────────────────────────────
    _update_env("ZERODHA_ACCESS_TOKEN", access_token)

    print(f"\n  Logged in as: {user_name} ({user_id})")
    print(f"  Access token: {access_token[:8]}{'*' * (len(access_token) - 8)}")
    print(f"\n  Saved to: {ENV_FILE}")
    print("\n" + "=" * 60)
    print("  Done! You can now start the bot:")
    print("    python main.py --paper     (paper trading)")
    print("    python main.py             (live trading)")
    print("=" * 60)


if __name__ == "__main__":
    main()
