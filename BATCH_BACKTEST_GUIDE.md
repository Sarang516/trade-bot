"""
BATCH BACKTEST GUIDE - Populate Registry for Agent Learning

The agent needs diverse training data to make good decisions. This guide walks you
through running systematic backtests to populate the registry.

QUICK START (5-10 minutes):
==========================
Run 3 quick tests to see the difference between symbols on 15-min candles:

    cd c:\Project\trade-bot
    python scripts/batch_backtest.py --quick

This tests:
  - RELIANCE 15min (2024-2025 bear market)
  - BANKNIFTY 15min (2024-2025 bear market)
  - FINNIFTY 15min (2024-2025 bear market)

Expected outcome: Different symbols should show different regime characteristics
and win rates. Smaller caps (BANKNIFTY, FINNIFTY) may trade better than large cap
RELIANCE in bear market.


COMPARE 5-MIN vs 15-MIN (15 minutes):
=====================================
See if 15-min candles really outperform 5-min:

    # First, your baseline (5-min RELIANCE - already done):
    python backtest_run.py --symbol RELIANCE --interval 5 --from 2024-06-01 --to 2025-06-01

    # Now test 15-min same period:
    python backtest_run.py --symbol RELIANCE --interval 15 --from 2024-06-01 --to 2025-06-01

Compare the win rates and Sharpe ratios. 15-min should be cleaner signals.


TEST DIFFERENT REGIME (Bull market 2022):
==========================================
The current data is all bear market. Let's see how strategy performs in bull market:

    python backtest_run.py --symbol RELIANCE --interval 15 --from 2022-01-01 --to 2022-12-31
    python backtest_run.py --symbol BANKNIFTY --interval 15 --from 2022-01-01 --to 2022-12-31

This will populate registry with "what works in BULL" vs "what works in BEAR".
Agent can then adapt parameters based on detected regime.


FULL GRID (45-60 minutes):
==========================
Run all combinations of symbol x interval x period:

    python scripts/batch_backtest.py --full

This tests:
  - RELIANCE, BANKNIFTY, FINNIFTY on 15-min (2024-2025 bear)
  - RELIANCE, BANKNIFTY on 15-min (2022 bull)
  - RELIANCE on 5-min (2024-2025 bear) - for comparison

Results populate registry with regime/symbol/interval combinations.


CUSTOM TEST:
============
Test a specific symbol/date range:

    python scripts/batch_backtest.py --symbol FINNIFTY --from 2023-01-01 --to 2024-01-01

Or test only 15-min candles:

    python scripts/batch_backtest.py --15min


VIEW REGISTRY:
==============
After running backtests, see what the agent learned:

    python scripts/regime_report.py                    # All results
    python scripts/regime_report.py --symbol RELIANCE   # RELIANCE only


TEST THE AGENT:
===============
Once registry is populated, see what agent recommends:

    python scripts/agent_manual.py --symbol RELIANCE
    python scripts/agent_manual.py --symbol BANKNIFTY


START BOT WITH ENRICHED DATA:
=============================
The bot now uses agent's learned parameters:

    python main.py


EXPECTED OUTCOMES:
==================
1. QUICK TEST: Should see 3 different regime classifications
2. 5min vs 15min: 15-min should have fewer trades, better quality (higher win rate)
3. Bull vs Bear: Different parameters should work in each regime
4. Agent decision: Agent will pick parameters from best-performing regime


NEXT STEPS IF STILL UNPROFITABLE:
==================================
If all backtests are still losing money:
1. Strategy may be fundamentally wrong for this market
2. Need to try different entry/exit logic
3. Consider other strategies (mean-reversion, momentum, etc)
4. Use only high-quality signals (reduce volume_surge_multiplier threshold)
5. Focus on symbols that trade well (BANKNIFTY/FINNIFTY vs RELIANCE)


REMEMBER:
=========
- Backtests with few trades (<5) are auto-skipped from registry
- Negative sharpe ratios still get logged (agent needs to know what doesn't work)
- Agent will prefer DISABLED mode if all known parameters are unprofitable
- Position size reduction (0.5x) helps when market conditions are uncertain
"""
