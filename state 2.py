"""
state.py — Shared state schema for the Hydra Sentinel-X agent graph.

TWO STATE TYPES LIVE HERE:

  AgentState  — the original flat state used by the four specialist agents.
                Unchanged from before; all existing code continues to work.

  GoTState    — extends AgentState with Graph of Thoughts fields. The GoT
                reasoning layer operates on this extended state so it has
                access to both the specialist outputs AND the thought graph.

WHY EXTEND RATHER THAN REPLACE:
  The specialist agents produce rich structured data (prices, RSI, Kelly-
  sized positions) that the GoT layer depends on as its raw material.
  Replacing that pipeline would be wasteful. Instead:

    1. Specialists run first  → populate market_data, technical_analysis, etc.
    2. GoT layer runs second  → reads that data, generates competing hypotheses,
                                stress-tests them adversarially, and aggregates
                                survivors into a richer trade_recommendation.

  The GoT layer is pure reasoning. It never calls CoinGecko or computes RSI —
  it reads what the specialists already fetched and argues over it.

THOUGHT NODE LIFECYCLE:
  Each ThoughtNode is an immutable record of one agent's hypothesis at one
  moment. The parent_ids field creates a lineage chain: when the aggregator
  merges two surviving thoughts, the consensus node's parent_ids points to
  both parents. Every final recommendation is fully traceable back through
  the chain of reasoning and critique that produced it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


# ── Reducers ────────────────────────────────────────────────────────────────

def add_messages(left: list, right: list) -> list:
    return left + right


def merge_thoughts(
    left: list,
    right: list,
) -> list:
    """
    Append new ThoughtNodes without overwriting existing ones.
    Idempotent: nodes already present (by node_id) are skipped.
    This preserves the full reasoning lineage across loop iterations.
    """
    seen_ids = {t["node_id"] for t in left}
    new_nodes = [t for t in right if t["node_id"] not in seen_ids]
    return left + new_nodes


# ── Specialist output payloads ───────────────────────────────────────────────

class MarketData(TypedDict, total=False):
    symbol: str
    price_usd: float
    price_change_24h: float
    market_cap: float
    volume_24h: float
    news_headlines: list[str]
    sentiment: Literal["bullish", "bearish", "neutral"]
    fear_greed_index: int
    raw_data: dict[str, Any]


class TechnicalAnalysis(TypedDict, total=False):
    symbol: str
    timeframe: str
    rsi: float
    macd_signal: Literal["buy", "sell", "neutral"]
    macd_value: float
    macd_histogram: float
    bb_position: str
    ema_trend: Literal["uptrend", "downtrend", "sideways"]
    support_levels: list[float]
    resistance_levels: list[float]
    summary: str
    chart_ascii: str


class RiskAssessment(TypedDict, total=False):
    portfolio_value: float
    risk_level: Literal["low", "medium", "high", "extreme"]
    max_position_size_pct: float
    max_position_size_usd: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_reward_ratio: float
    warnings: list[str]
    rationale: str


class TradeRecommendation(TypedDict, total=False):
    action: Literal["strong_buy", "buy", "hold", "sell", "strong_sell", "avoid"]
    entry_zone: dict[str, float]
    entry_price: float
    target_price: float
    stop_loss: float
    timeframe: str
    confidence: float
    strategy_used: str
    rationale: str
    key_risks: list[str]
    dca_plan: str


# ── Graph of Thoughts types ─────────────────────────────────────────────────

class ThoughtNode(TypedDict):
    """
    One hypothesis in the reasoning graph — a Post-it from one analyst.

    node_id:          UUID, unique across the entire session.
    parent_ids:       IDs of prior ThoughtNodes this was derived from.
                      Empty = root hypothesis with no prior thought.
    agent_origin:     Which persona generated this (STINKMEANER, SAMUEL, etc.)
    content:          The actual hypothesis: thesis, supporting evidence,
                      proposed entry/exit, timeframe reasoning.
    confidence_score: 0.0–1.0 assigned by the generating agent.
                      The adversary node will reduce this under stress tests.
    generation:       Which pass through the generate→critique cycle made this.
    critique_notes:   Populated by the adversary: what attacks were attempted.
    survived_critique: True once the adversary has evaluated this node and it
                      passed. Distinguishes "not yet evaluated" from "passed".
    """
    node_id:           str
    parent_ids:        list[str]
    agent_origin:      str
    content:           dict[str, Any]
    confidence_score:  float
    generation:        int
    critique_notes:    list[str]
    survived_critique: bool


class AdversarialAttack(TypedDict):
    """
    One stress-test scenario applied to one ThoughtNode by GRANDDAD.

    Preserving all attacks in state gives the approval dashboard a clear
    picture of which risks were considered and survived — the human reviewer
    can see "this thesis was tested against a 20% flash-crash and held."
    """
    attack_id:        str
    target_node_id:   str
    attack_type:      str     # "flash_crash" | "volume_spoof" | "macro_shock" | etc.
    scenario:         str     # human-readable description of the stress test
    confidence_delta: float   # negative = lowers confidence
    thesis_survives:  bool
    explanation:      str


# ── Original state (unchanged — all existing code continues to work) ─────────

class AgentState(TypedDict):
    messages:             Annotated[list[BaseMessage], add_messages]
    user_query:           str
    target_symbol:        str
    portfolio_value:      float
    current_holdings:     dict[str, float]
    risk_tolerance:       Literal["conservative", "moderate", "aggressive"]
    active_agents:        list[str]
    current_step:         str
    iteration:            int
    market_data:          MarketData
    technical_analysis:   TechnicalAnalysis
    risk_assessment:      RiskAssessment
    trade_recommendation: TradeRecommendation
    memory_context:       str
    final_response:       str
    errors:               list[str]


# ── Extended GoT state ───────────────────────────────────────────────────────

class GoTState(AgentState):
    """
    Extends AgentState with the Graph of Thoughts reasoning layer.

    The GoT fields are populated AFTER the four specialists run. The
    specialists fill market_data, technical_analysis, risk_assessment,
    and a first-pass trade_recommendation. The GoT layer then reads those
    fields, generates competing ThoughtNodes, critiques them adversarially,
    and writes a richer, battle-tested recommendation back to
    trade_recommendation — augmented with the full reasoning audit trail.

    thought_graph:       All ThoughtNodes across all generations.
                         Uses merge_thoughts reducer so nodes from loop
                         iteration 1 survive when iteration 2 adds new ones.
    adversarial_attacks: Every stress test applied; preserved for the audit trail.
    got_generation:      Current pass through the generate→critique cycle.
    got_max_generations: Hard ceiling on retries (prevents infinite loops).
    consensus_reached:   Set True by the aggregator. Conditional edge uses this
                         to decide: route to END or loop back to generate.
    surviving_thoughts:  Subset of thought_graph that passed adversarial critique.
                         The aggregator works from this list, not the full graph.
    """
    thought_graph:        Annotated[list[ThoughtNode], merge_thoughts]
    adversarial_attacks:  Annotated[list[AdversarialAttack], operator.add]
    got_generation:       int
    got_max_generations:  int
    consensus_reached:    bool
    surviving_thoughts:   list[ThoughtNode]
