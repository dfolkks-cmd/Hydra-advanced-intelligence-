"""
agents/specialists.py — The four specialist sub-agents and their node functions.

Each agent:
  1. Receives the current AgentState
  2. Uses its bound tools to gather data
  3. Returns a partial state update
"""

from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from state import AgentState
from tools.crypto_tools import (
    MARKET_TOOLS,
    RISK_TOOLS,
    STRATEGY_TOOLS,
    TECHNICAL_TOOLS,
    calculate_technical_indicators,
    get_historical_prices,
)

# CDP wallet tools — imported conditionally so the system works even if
# cdp-sdk isn't installed yet (e.g. during local development without live keys).
# When CDP IS configured, the Risk Manager gets real on-chain balance data
# before it calculates any position size — preventing recommendations that
# exceed what's actually in the wallet.
try:
    from cdp.wallet_manager import CDP_READONLY_TOOLS
    CDP_ENABLED = True
except Exception:
    CDP_READONLY_TOOLS = []
    CDP_ENABLED = False


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _model(model_name: str = "claude-sonnet-4-6") -> ChatAnthropic:
    return ChatAnthropic(model=model_name, temperature=0.1, max_tokens=4096)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET ANALYST
# ─────────────────────────────────────────────────────────────────────────────

MARKET_ANALYST_SYSTEM = """\
You are an expert crypto market analyst with 15+ years of experience.
Your job: gather real-time price data, market sentiment, and news for the
requested cryptocurrency.

When called:
1. Use get_crypto_price() to fetch current price and 24h stats.
2. Use get_fear_greed_index() to gauge overall market sentiment.
3. Synthesise findings into a clear, concise market snapshot.

Output a JSON block with this exact structure inside ```json ... ```:
{
  "price_usd": float,
  "price_change_24h": float,
  "market_cap": float,
  "volume_24h": float,
  "sentiment": "bullish" | "bearish" | "neutral",
  "fear_greed_index": int,
  "news_headlines": ["headline1", ...],
  "key_observations": ["obs1", "obs2", ...]
}

Be concise. Only include verifiable data from tools.
"""


def market_analyst_node(state: AgentState) -> dict:
    """LangGraph node: Market Analyst agent."""
    llm = _model().bind_tools(MARKET_TOOLS)
    messages = [
        SystemMessage(content=MARKET_ANALYST_SYSTEM),
        HumanMessage(
            content=f"Analyse the current market conditions for {state['target_symbol']}. "
                    f"Portfolio context: ${state['portfolio_value']:,.0f} total value. "
                    f"User query: {state['user_query']}"
        ),
    ]

    # Simple ReAct loop (up to 5 tool calls)
    for _ in range(5):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_fn = {t.name: t for t in MARKET_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                from langchain_core.messages import ToolMessage
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

    # Parse the JSON block from the final response
    content = response.content if hasattr(response, "content") else str(response)
    market_data: dict[str, Any] = {"symbol": state["target_symbol"]}
    try:
        import re
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            market_data.update(parsed)
    except Exception:
        market_data["raw_data"] = {"response": content}

    return {
        "market_data": market_data,
        "current_step": "market_analysis_complete",
        "messages": messages[2:],  # skip the system prompt from history
    }


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYST
# ─────────────────────────────────────────────────────────────────────────────

TECHNICAL_ANALYST_SYSTEM = """\
You are a professional crypto technical analyst and quant trader.
Your expertise: RSI, MACD, Bollinger Bands, EMA crossovers, support/resistance.

When called:
1. Use get_historical_prices() to fetch OHLCV data (30 days default).
2. Use calculate_technical_indicators() on that data.
3. Optionally use execute_python() for custom calculations.
4. Provide a clear technical verdict.

Output a JSON block inside ```json ... ```:
{
  "timeframe": "30d",
  "rsi": float,
  "macd_signal": "buy" | "sell" | "neutral",
  "bb_position": string,
  "ema_trend": "uptrend" | "downtrend" | "sideways",
  "support_levels": [float, ...],
  "resistance_levels": [float, ...],
  "overall_signal": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
  "confidence": float (0-1),
  "summary": "2-3 sentence technical summary"
}
"""


def technical_analyst_node(state: AgentState) -> dict:
    """LangGraph node: Technical Analyst agent."""
    llm = _model().bind_tools(TECHNICAL_TOOLS)
    messages = [
        SystemMessage(content=TECHNICAL_ANALYST_SYSTEM),
        HumanMessage(
            content=f"Perform technical analysis on {state['target_symbol']}. "
                    f"User query context: {state['user_query']}"
        ),
    ]

    for _ in range(6):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_fn = {t.name: t for t in TECHNICAL_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    content = response.content if hasattr(response, "content") else str(response)
    ta: dict[str, Any] = {"symbol": state["target_symbol"]}
    try:
        import re
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            ta.update(json.loads(match.group(1)))
    except Exception:
        ta["summary"] = content[:500]

    return {
        "technical_analysis": ta,
        "current_step": "technical_analysis_complete",
        "messages": messages[2:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

RISK_MANAGER_SYSTEM = """\
You are a seasoned crypto risk manager. Your mission: protect capital above all else.
Principles: never risk more than you can afford to lose; always use stop-losses;
size positions based on Kelly Criterion and fixed-fractional methods.

When called:
1. Use calculate_position_size() to determine optimal sizing.
2. Use assess_portfolio_risk() to evaluate current exposure.
3. Factor in market volatility and the Fear & Greed Index.
4. Always recommend a specific stop-loss and take-profit level.

Output a JSON block inside ```json ... ```:
{
  "risk_level": "low" | "medium" | "high" | "extreme",
  "max_position_size_pct": float,
  "max_position_size_usd": float,
  "stop_loss_pct": float,
  "take_profit_pct": float,
  "risk_reward_ratio": float,
  "warnings": ["warning1", ...],
  "rationale": "2-3 sentence risk rationale"
}
"""


def risk_manager_node(state: AgentState) -> dict:
    """LangGraph node: Risk Manager agent.

    When CDP is enabled this agent has two layers of position-size data:
      1. The theoretical size from calculate_position_size() (Kelly Criterion)
      2. The actual on-chain balance from cdp_get_all_balances()

    The final recommended size is the MINIMUM of the two — you can never
    trade more than you actually hold in the wallet, regardless of what
    Kelly says is theoretically optimal.
    """
    # Merge standard risk tools with CDP readonly tools when available.
    # This gives the LLM access to real wallet balance data so it can
    # sanity-check position size against actual holdings.
    active_tools = RISK_TOOLS + CDP_READONLY_TOOLS if CDP_ENABLED else RISK_TOOLS
    llm = _model().bind_tools(active_tools)

    # Pull context produced by the upstream agents this same run
    price = state.get("market_data", {}).get("price_usd", 0)
    fgi   = state.get("market_data", {}).get("fear_greed_index", 50)
    ta    = state.get("technical_analysis", {})

    # Build a note about CDP status so the LLM knows what tools it has
    cdp_note = (
        "CDP wallet is CONNECTED — use cdp_get_all_balances() to fetch the "
        "real on-chain balance before finalising position size. "
        "The recommended position must not exceed the actual wallet balance."
        if CDP_ENABLED
        else "CDP wallet is NOT configured — base position sizing on portfolio_value only."
    )

    messages = [
        SystemMessage(content=RISK_MANAGER_SYSTEM),
        HumanMessage(
            content=(
                f"Assess risk for a potential {state['target_symbol']} position.\n\n"
                f"Portfolio (total):  ${state['portfolio_value']:,.0f}\n"
                f"Risk tolerance:     {state['risk_tolerance']}\n"
                f"Current price:      ${price:,.4f}\n"
                f"Fear & Greed Index: {fgi}\n"
                f"Technical signal:   {ta.get('overall_signal', 'unknown')}\n"
                f"Current holdings:   {json.dumps(state.get('current_holdings', {}))}\n\n"
                f"CDP status: {cdp_note}\n\n"
                f"User query: {state['user_query']}"
            )
        ),
    ]

    for _ in range(4):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_fn = {t.name: t for t in RISK_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    content = response.content if hasattr(response, "content") else str(response)
    risk: dict[str, Any] = {"portfolio_value": state["portfolio_value"]}
    try:
        import re
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            risk.update(json.loads(match.group(1)))
    except Exception:
        risk["rationale"] = content[:500]

    return {
        "risk_assessment": risk,
        "current_step": "risk_assessment_complete",
        "messages": messages[2:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_ADVISOR_SYSTEM = """\
You are an elite crypto trading strategist with expertise in:
- Momentum trading, swing trading, DCA (dollar-cost averaging)
- On-chain analysis interpretation
- Macro crypto cycle analysis (Bitcoin halving cycles, alt seasons)
- Derivatives and options hedging concepts

You synthesise inputs from the Market Analyst, Technical Analyst, and Risk Manager
to produce a final, actionable trade recommendation.

Be specific: give exact entry zones, targets, and stop levels.
Consider the user's risk tolerance and portfolio context.

Output a JSON block inside ```json ... ```:
{
  "action": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell" | "avoid",
  "strategy_used": "momentum" | "swing" | "dca" | "breakout" | "mean_reversion",
  "entry_zone": {"low": float, "high": float},
  "target_price": float,
  "stop_loss": float,
  "timeframe": "1d" | "1w" | "1m",
  "confidence": float (0-1),
  "rationale": "3-4 sentence strategy rationale",
  "key_risks": ["risk1", "risk2", ...],
  "dca_plan": "optional DCA suggestion if applicable"
}
"""


def strategy_advisor_node(state: AgentState) -> dict:
    """LangGraph node: Strategy Advisor agent."""
    llm = _model().bind_tools(STRATEGY_TOOLS)

    md  = state.get("market_data", {})
    ta  = state.get("technical_analysis", {})
    ra  = state.get("risk_assessment", {})

    messages = [
        SystemMessage(content=STRATEGY_ADVISOR_SYSTEM),
        HumanMessage(
            content=(
                f"Synthesise a trade strategy for {state['target_symbol']}.\n\n"
                f"MARKET DATA:\n{json.dumps(md, indent=2)}\n\n"
                f"TECHNICAL ANALYSIS:\n{json.dumps(ta, indent=2)}\n\n"
                f"RISK ASSESSMENT:\n{json.dumps(ra, indent=2)}\n\n"
                f"Portfolio: ${state['portfolio_value']:,.0f} | "
                f"Risk tolerance: {state['risk_tolerance']}\n"
                f"Memory context:\n{state.get('memory_context', 'None')}\n\n"
                f"User query: {state['user_query']}"
            )
        ),
    ]

    for _ in range(4):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_fn = {t.name: t for t in STRATEGY_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    content = response.content if hasattr(response, "content") else str(response)
    rec: dict[str, Any] = {}
    try:
        import re
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            rec.update(json.loads(match.group(1)))
    except Exception:
        rec["rationale"] = content[:500]

    return {
        "trade_recommendation": rec,
        "current_step": "strategy_complete",
        "messages": messages[2:],
    }
