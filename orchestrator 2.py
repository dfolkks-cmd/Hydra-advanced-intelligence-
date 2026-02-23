"""
agents/orchestrator.py — The master orchestrator agent.

Responsibilities:
  1. Parse the user query and extract intent + target symbol
  2. Decide which specialists to invoke
  3. Synthesise all specialist outputs into the final response
  4. Save a session summary to memory
"""

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState


ORCHESTRATOR_SYSTEM = """\
You are the master orchestrator of a professional crypto trading AI system.
You manage a team of specialist agents:
  - market_analyst:    real-time price, sentiment, news
  - technical_analyst: RSI, MACD, Bollinger Bands, EMA, support/resistance
  - risk_manager:      position sizing, stop-losses, portfolio exposure
  - strategy_advisor:  final trade recommendation synthesising all inputs

Your responsibilities:
  1. Parse the user's query to extract:
     - target_symbol (e.g. BTC, ETH, SOL) — default to BTC if unclear
     - which agents are needed for this query
     - portfolio details if mentioned
  2. After all agents have run, synthesise their outputs into a clear,
     professional final response that a serious trader would find valuable.
  3. Include: price snapshot, key technical signals, risk parameters,
     and a clear actionable recommendation.

ROUTING RULES:
- Price/news/sentiment queries → [market_analyst]
- Chart/indicator queries      → [technical_analyst]
- "Should I buy/sell" queries  → [market_analyst, technical_analyst, risk_manager, strategy_advisor]
- Portfolio/risk queries       → [market_analyst, risk_manager]
- Full analysis                → all four agents

Always include disclaimers: crypto is highly volatile; not financial advice.
"""


def orchestrator_parse_node(state: AgentState) -> dict:
    """
    First orchestrator pass: parse query, set routing, extract symbol.
    """
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(
            content=(
                f"Parse this user query and decide routing.\n\n"
                f"Query: {state['user_query']}\n"
                f"Memory context: {state.get('memory_context', 'None')}\n\n"
                "Reply ONLY with a JSON object inside ```json ... ```:\n"
                "{\n"
                '  "target_symbol": "BTC",\n'
                '  "active_agents": ["market_analyst", "technical_analyst", '
                '"risk_manager", "strategy_advisor"],\n'
                '  "portfolio_value": null,\n'
                '  "risk_tolerance": "moderate",\n'
                '  "intent": "one-line description of what user wants"\n'
                "}"
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content

    updates: dict = {}
    try:
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            updates["target_symbol"]  = parsed.get("target_symbol", "BTC").upper()
            updates["active_agents"]  = parsed.get(
                "active_agents",
                ["market_analyst", "technical_analyst", "risk_manager", "strategy_advisor"],
            )
            if parsed.get("portfolio_value") and state.get("portfolio_value", 0) == 0:
                updates["portfolio_value"] = float(parsed["portfolio_value"])
            if parsed.get("risk_tolerance"):
                updates["risk_tolerance"] = parsed["risk_tolerance"]
    except Exception:
        updates["target_symbol"] = "BTC"
        updates["active_agents"] = ["market_analyst", "technical_analyst",
                                    "risk_manager", "strategy_advisor"]

    updates["current_step"] = "routing_complete"
    updates["iteration"]    = state.get("iteration", 0) + 1
    return updates


def orchestrator_synthesise_node(state: AgentState) -> dict:
    """
    Final orchestrator pass: write the response the user sees.
    """
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0.2, max_tokens=2000)

    md  = state.get("market_data", {})
    ta  = state.get("technical_analysis", {})
    ra  = state.get("risk_assessment", {})
    rec = state.get("trade_recommendation", {})

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(
            content=(
                f"Synthesise the following specialist outputs into a final response "
                f"for the user query: '{state['user_query']}'\n\n"
                f"TARGET: {state.get('target_symbol', 'BTC')}\n"
                f"PORTFOLIO: ${state.get('portfolio_value', 0):,.0f} | "
                f"Risk tolerance: {state.get('risk_tolerance', 'moderate')}\n\n"
                f"MARKET DATA:\n{json.dumps(md, indent=2)}\n\n"
                f"TECHNICAL ANALYSIS:\n{json.dumps(ta, indent=2)}\n\n"
                f"RISK ASSESSMENT:\n{json.dumps(ra, indent=2)}\n\n"
                f"TRADE RECOMMENDATION:\n{json.dumps(rec, indent=2)}\n\n"
                "Format the response professionally. Use clear sections:\n"
                "1. 📊 Market Snapshot\n"
                "2. 📈 Technical Picture\n"
                "3. ⚖️  Risk Parameters\n"
                "4. 🎯 Recommendation\n"
                "5. ⚠️  Key Risks & Disclaimer\n"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {
        "final_response": response.content,
        "current_step": "synthesis_complete",
    }
