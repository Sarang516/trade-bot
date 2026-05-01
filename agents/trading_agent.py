"""
agents/trading_agent.py - Claude AI-powered trading agent.

The agent uses the Anthropic API to analyze market conditions and make
trading decisions. It has tools to:
  - Detect market regime (BULL/BEAR/RANGE/VOLATILE)
  - Analyze bull/bear signals and profitability patterns
  - Query the parameter registry for best-known params
  - Assess risk and recommend position sizing
  - Make final trading decisions

The agent is deterministic and explainable — it reasons through each step
and returns both a decision and the reasoning behind it.

Usage
-----
    from agents.trading_agent import TradingAgent
    agent = TradingAgent(broker, settings)
    decision = agent.analyze_and_decide("RELIANCE")
    print(decision.parameters)
    print(decision.reasoning)
    print(decision.risk_override)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger

try:
    import anthropic
except ImportError:
    raise ImportError("Install anthropic: pip install anthropic")


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentDecision:
    """The agent's recommendation for today's trading."""
    symbol: str
    regime: str
    decision_timestamp: datetime

    # Core decision
    should_trade: bool
    trading_mode: str  # "LONG_PREFERRED", "SHORT_PREFERRED", "BOTH", "DISABLED"

    # Parameters to apply
    parameters: dict[str, Any]

    # Risk adjustments
    position_size_multiplier: float  # 0.5 = half size, 2.0 = double
    trailing_sl_enabled: bool
    max_open_positions: int

    # Decision reasoning (for logging / dashboard)
    reasoning: str
    confidence_score: float  # 0-100

    # Data snapshot used for decision
    market_data: dict[str, Any]


# ── Claude Agent ──────────────────────────────────────────────────────────────

class TradingAgent:
    """
    AI-powered agent that makes trading decisions using Claude.

    Each morning it analyzes:
      - Current market regime
      - Bull/bear signals
      - Historical parameter performance
      - Recent live trading results
      - Economic events / news

    And decides what parameters to use and how much risk to take.
    """

    def __init__(self, broker, settings) -> None:
        self._broker = broker
        self._settings = settings
        api_key = getattr(settings, "anthropic_api_key", "") or ""
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = "claude-sonnet-4-6"

    def analyze_and_decide(self, symbol: str) -> AgentDecision:
        """
        Main entry point: analyze market for the symbol and return a decision.
        """
        logger.info("Agent: Starting analysis for {}", symbol)

        # Collect all available market data
        market_data = self._gather_market_data(symbol)

        # Build the prompt for Claude
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(symbol, market_data)

        # Call Claude with tool use
        response = self._call_claude_with_tools(system_prompt, user_message)

        # Parse Claude's decision
        decision = self._parse_agent_response(symbol, response, market_data)

        logger.info(
            "Agent decision for {}: {} | regime={} | confidence={:.0f}% | "
            "pos_size_mult={:.1f}x",
            symbol, decision.trading_mode, decision.regime,
            decision.confidence_score, decision.position_size_multiplier,
        )

        return decision

    # ── Data gathering ────────────────────────────────────────────────────────

    def _gather_market_data(self, symbol: str) -> dict[str, Any]:
        """Collect all relevant market and trading data."""
        from strategies.regime_detector import RegimeDetector, describe_regime
        from data.parameter_registry import ParameterRegistry
        from db.trade_logger import TradeLogger

        logger.info("Agent: Gathering market data for {}", symbol)

        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Regime detection
        try:
            detector = RegimeDetector(self._broker, self._settings)
            regime = detector.detect(symbol, use_cache_minutes=0)
            data["current_regime"] = regime.value
            data["regime_description"] = describe_regime(regime)
            logger.info("Agent: Regime = {}", regime.value)
        except Exception as e:
            logger.warning("Agent: Regime detection failed: {}", e)
            data["current_regime"] = "UNKNOWN"
            data["regime_description"] = str(e)

        # 2. Historical price action (bull/bear indicators)
        try:
            price_data = self._get_price_analysis(symbol)
            data.update(price_data)
            logger.info(
                "Agent: Price action | 10d_return={:.1f}% | 20d_return={:.1f}%",
                price_data.get("return_10d", 0), price_data.get("return_20d", 0),
            )
        except Exception as e:
            logger.warning("Agent: Price analysis failed: {}", e)

        # 3. Parameter registry — best known params per regime
        try:
            reg = ParameterRegistry()
            summaries = reg.get_regime_summary()
            data["parameter_history"] = [
                s for s in summaries if s["symbol"] == symbol.upper()
            ]
            logger.info("Agent: {} regime(s) in registry", len(data["parameter_history"]))
        except Exception as e:
            logger.warning("Agent: Registry query failed: {}", e)
            data["parameter_history"] = []

        # 4. Recent live trading performance (last 5 days)
        try:
            tl = TradeLogger()
            recent = tl.get_trades()
            recent_5d = [
                t for t in recent
                if t and t.get("exit_time")
                and datetime.fromisoformat(t["exit_time"]) > datetime.now() - timedelta(days=5)
            ]
            data["recent_trades"] = recent_5d[:10]
            if recent_5d:
                win_count = sum(1 for t in recent_5d if t.get("pnl", 0) > 0)
                data["recent_win_rate_pct"] = win_count / len(recent_5d) * 100
                data["recent_avg_pnl"] = sum(t.get("pnl", 0) for t in recent_5d) / len(recent_5d)
            logger.info(
                "Agent: Recent performance | trades={} | win_rate={:.0f}%",
                len(recent_5d), data.get("recent_win_rate_pct", 0),
            )
        except Exception as e:
            logger.warning("Agent: Trade history fetch failed: {}", e)
            data["recent_trades"] = []

        return data

    def _get_price_analysis(self, symbol: str) -> dict[str, Any]:
        """Analyze recent price action to detect bull/bear signals."""
        from brokers.base_broker import Exchange
        from data.feed import HistoricalData

        hist = HistoricalData(self._broker)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=60)

        df = hist.fetch(symbol, Exchange.NSE, from_date, to_date, interval_minutes=60)
        if df.empty or len(df) < 20:
            return {}

        daily = df["close"].resample("D").last().dropna()

        returns = {
            "return_5d": (daily.iloc[-1] / daily.iloc[-6] - 1) * 100 if len(daily) >= 6 else 0,
            "return_10d": (daily.iloc[-1] / daily.iloc[-11] - 1) * 100 if len(daily) >= 11 else 0,
            "return_20d": (daily.iloc[-1] / daily.iloc[-21] - 1) * 100 if len(daily) >= 21 else 0,
        }

        # EMA slope
        ema20 = daily.ewm(span=20).mean()
        ema_slope = (ema20.iloc[-1] - ema20.iloc[-6]) / ema20.iloc[-6] * 100 if len(ema20) >= 6 else 0
        returns["ema20_slope_pct"] = ema_slope

        # Volatility
        returns["volatility_20d_pct"] = daily.pct_change().std() * 100

        return returns

    # ── Claude integration ────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return """You are an expert trading agent for Indian stock market intraday trading.

Your role:
1. Analyze the current market regime (BULL/BEAR/RANGE/VOLATILE)
2. Review historical parameter performance in each regime
3. Assess recent trading results and recent price action
4. Make trading decisions that maximize profitability for TODAY's conditions

Guidelines:
- In BULL markets: prioritize LONG signals, be aggressive with entries, wider targets
- In BEAR markets: prioritize SHORT signals OR disable trading if recent losses, tighter stops
- In RANGE markets: use mean-reversion params, quick exits, tight risk
- In VOLATILE markets: reduce position size 50%, wider stops, fewer entries
- If recent performance is poor (< 30% win rate), recommend DISABLED mode
- Always explain your reasoning clearly

Output format:
Return a JSON object with these exact fields:
{
  "should_trade": bool,
  "trading_mode": "LONG_PREFERRED" | "SHORT_PREFERRED" | "BOTH" | "DISABLED",
  "recommended_parameters": { key: value, ... },
  "position_size_multiplier": 0.5 to 2.0,
  "trailing_sl_enabled": bool,
  "max_open_positions": 1-5,
  "confidence_score": 0-100,
  "reasoning": "explanation of the decision"
}"""

    def _build_user_message(self, symbol: str, market_data: dict) -> str:
        return f"""Analyze today's trading opportunity for {symbol}.

Current market data:
{json.dumps(market_data, indent=2, default=str)}

Current settings:
- Default strategy: {self._settings.strategy}
- Capital: Rs. {self._settings.trading_capital:,.0f}
- Risk per trade: {self._settings.risk_per_trade_pct}%
- Max daily loss: Rs. {self._settings.max_daily_loss_inr:,.0f}

Based on this, decide:
1. Should we trade today?
2. What mode (LONG/SHORT/BOTH/DISABLED)?
3. Which parameters should we use?
4. What position size (0.5x to 2.0x default)?
5. Confidence in this decision (0-100%)?

Respond ONLY with valid JSON (no markdown, no explanation outside JSON)."""

    def _call_claude_with_tools(self, system_prompt: str, user_message: str) -> str:
        """Call Claude API and parse the response."""
        logger.info("Agent: Calling Claude API...")

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
        )

        # Extract text from response
        content = response.content[0]
        if content.type == "text":
            return content.text
        else:
            raise ValueError(f"Unexpected response type: {content.type}")

    def _parse_agent_response(
        self,
        symbol: str,
        response_text: str,
        market_data: dict,
    ) -> AgentDecision:
        """Parse Claude's JSON response into an AgentDecision."""
        try:
            agent_output = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error("Agent: Failed to parse JSON response: {}", e)
            logger.error("Response was: {}", response_text[:200])
            # Fallback: safe defaults
            agent_output = {
                "should_trade": False,
                "trading_mode": "DISABLED",
                "recommended_parameters": {},
                "position_size_multiplier": 1.0,
                "trailing_sl_enabled": True,
                "max_open_positions": 1,
                "confidence_score": 0,
                "reasoning": f"Agent parsing failed. Error: {e}",
            }

        # Get default parameters from strategy
        from strategies.vwap_volume import VWAPVolumeConfig
        default_cfg = VWAPVolumeConfig()

        # Merge recommended params with defaults
        recommended = agent_output.get("recommended_parameters", {})
        final_params = {
            "volume_surge_multiplier": recommended.get("volume_surge_multiplier",
                                                       default_cfg.volume_surge_multiplier),
            "volume_ma_period": recommended.get("volume_ma_period",
                                               default_cfg.volume_ma_period),
            "rsi_long_min": recommended.get("rsi_long_min",
                                           default_cfg.rsi_long_min),
            "rsi_long_max": recommended.get("rsi_long_max",
                                           default_cfg.rsi_long_max),
            "rsi_short_min": recommended.get("rsi_short_min",
                                            default_cfg.rsi_short_min),
            "rsi_short_max": recommended.get("rsi_short_max",
                                            default_cfg.rsi_short_max),
            "ema_period": recommended.get("ema_period", default_cfg.ema_period),
            "ema_trend_period": recommended.get("ema_trend_period",
                                               default_cfg.ema_trend_period),
            "sl_atr_multiplier": recommended.get("sl_atr_multiplier",
                                                 default_cfg.sl_atr_multiplier),
            "rr_ratio": recommended.get("rr_ratio", default_cfg.rr_ratio),
            "vwap_exit_candles": recommended.get("vwap_exit_candles",
                                                 default_cfg.vwap_exit_candles),
        }

        return AgentDecision(
            symbol=symbol,
            regime=market_data.get("current_regime", "UNKNOWN"),
            decision_timestamp=datetime.now(),
            should_trade=agent_output.get("should_trade", False),
            trading_mode=agent_output.get("trading_mode", "DISABLED"),
            parameters=final_params,
            position_size_multiplier=float(agent_output.get("position_size_multiplier", 1.0)),
            trailing_sl_enabled=agent_output.get("trailing_sl_enabled", True),
            max_open_positions=int(agent_output.get("max_open_positions", 3)),
            reasoning=agent_output.get("reasoning", ""),
            confidence_score=float(agent_output.get("confidence_score", 0)),
            market_data=market_data,
        )
